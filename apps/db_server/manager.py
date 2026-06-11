"""Manages the database"""

import asyncio
import fnmatch
import hashlib
import io
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiofiles
import jstyleson
from sqlalchemy.orm import make_transient

from apps import CACHE_DIR, THUMB_CACHE_MAX_MB
from apps.db_server.hash_index import HashIndex
from apps.helpers.consts import TaskStatus
from apps.helpers.db.db_client import (
    DbClient,
    TagKind,
)
from apps.helpers.db.types import (
    AVAILABLE_MODEL_TYPENAMES,
    FALSE_POS_TAG_UUID,
    UNVERIFIED_TAG_UUID,
    BBoxUpdate,
    DatasetCreate,
    DatasetRead,
    DatasetUpdate,
    ImageLabelRead,
    ImageLabelStatus,
    ImageRecord,
    ImageRecordRead,
    ImageRecordUpdate,
    ImageSearchParams,
    Label,
    LabelType,
    ModelConfigCreate,
    ModelConfigNodeWrite,
    ModelConfigRead,
    ModelConfigUpdate,
    Source,
    Tag,
    TagUpdate,
    TaskConfig,
    TaskConfigRead,
    TaskInstance,
    TaskInstanceRead,
    TaskParams,
    TaskResultData,
    TaskResumeData,
)
from apps.helpers.fileUtils import get_safe_path
from apps.helpers.hashUtils import (
    compute_bbox_phash,
    compute_image_phash,
    phash_from_numpy,
    phash_from_numpy_crop,
)
from apps.helpers.imageProviders.Registry import image_provider_registry
from apps.helpers.types import (
    FileEntryRead,
)

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

# Bboxes with any of these tags are excluded from PRESENT image-label tracking.
_BBOX_SKIP_TAGS = frozenset({UNVERIFIED_TAG_UUID, FALSE_POS_TAG_UUID})


class Manager:
    """Manages interactions with the database"""

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks

    def __init__(self, db_client: DbClient, files_root: Path):
        """Initialize the Manager with a database client"""
        self._db_client = db_client
        self._files_root: Path = files_root
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations
        self._config = {}
        self._image_hash_size: int = int(os.environ.get("IMAGE_HASH_SIZE", "8"))
        self._bbox_hash_size: int = int(os.environ.get("BBOX_HASH_SIZE", "16"))
        self._hash_index = HashIndex(
            db_client,
            image_hash_size=self._image_hash_size,
            bbox_hash_size=self._bbox_hash_size,
        )
        self._rebuild_in_progress: bool = False
        self._hash_index_ready: bool = False  # False until initial build completes on startup
        self._build_phase: str = ""  # "loading" or "constructing" during initial index build
        # Short-lived cache for expensive count queries.  Key is an MD5 of the
        # filter predicates (sort/page fields excluded); value is (count, ts).
        # TTL is intentionally short so writes that happen while browsing are
        # reflected within a few seconds when the user hits Refresh.
        self._count_cache: dict[str, tuple[int, float]] = {}
        self._COUNT_CACHE_TTL = 60.0
        # Maximum seconds to wait for the count query before returning -1.
        # count_images_search does a full table scan with no LIMIT; on large
        # datasets (100k+ images) with complex multi-EXISTS filters it can take
        # many seconds.  The list query (LIMIT 51) completes quickly thanks to
        # index early-termination, so we cap the count independently.
        self._COUNT_QUERY_TIMEOUT = 3.0  # seconds

    async def get_server_config(self) -> Dict[str, Any]:
        return self._config.copy()

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._worker_task())

    async def _init(self):
        """Initialization that should run on event loop"""
        await self._db_client.init_db()
        await self._hash_index.build(on_phase=lambda p: setattr(self, "_build_phase", p))
        self._build_phase = ""
        self._hash_index_ready = True

    async def _worker_task(self):
        """Periodic worker task that runs at regular intervals"""
        try:
            await self._init()
            while True:
                await asyncio.sleep(Manager.POLLING_INTERVAL)
                # Opportunistically rebuild the hash index when the tombstone
                # ratio has crossed the threshold.  Doing it here (between
                # requests rather than inside a request handler) avoids opening
                # a concurrent DB session while another transaction is active.
                await self._hash_index.rebuild_if_needed()
        except asyncio.CancelledError:
            print("Periodic task was cancelled.")
        finally:
            print("Periodic task cleanup.")

    ################################################################################
    # LABELS API
    ################################################################################

    async def create_new_label(
        self, name: str, color: str, parent_uuid: Optional[str] = None, label_type: LabelType = LabelType.STANDARD
    ) -> Label:
        """
        Adds a new label to the database

        Returns:
            Label: Label added to database
        """
        ret = await self._db_client.add_label(name, color, parent_uuid=parent_uuid, label_type=label_type)
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))
        return ret

    async def delete_label(self, label_uuid: str) -> None:
        """
        Deletes a label from the database
        """
        # Check if the label has a parent
        child_uuids = await self._db_client.get_label_children(label_uuid)
        if child_uuids:
            raise ValueError(f"Cannot delete label '{label_uuid}' because it has child labels.")

        await self._db_client.delete_label(label_uuid)
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))

    async def update_label(
        self,
        label_uuid: str,
        name: Optional[str] = None,
        color: Optional[str] = None,
        label_type: Optional[LabelType] = None,
    ) -> None:
        """
        Update a label's name, color, and/or label_type.

        Args:
            label_uuid (str): uuid of the label to update.
            name (Optional[str]): New name for the label.
            color (Optional[str]): New color for the label.
            label_type (Optional[LabelType]): New label type.
        """
        await self._db_client.update_label(
            label_uuid=label_uuid, name=name, color=color,
            **({} if label_type is None else {"label_type": label_type}),
        )
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))

    async def get_label_uuid_map(self) -> Dict[str, Label]:
        flat_list: List[Label] = await self._db_client.list_labels()

        # Manually create parent<->child relationships
        uuid_to_node: Dict[str, Label] = {}
        for x in flat_list:
            # Severe the SQL link and manually initialize children
            make_transient(x)
            x.children = []
            uuid_to_node[x.uuid] = x

        for node in uuid_to_node.values():
            if node.parent_uuid:
                parent_node = uuid_to_node.get(node.parent_uuid)
                if parent_node:
                    parent_node.children.append(node)

        return uuid_to_node

    async def get_labels(self) -> List[Label]:
        """
        Gets a list of all labels
        """
        return list((await self.get_label_uuid_map()).values())

    ################################################################################
    # SOURCES API
    ################################################################################

    async def get_avail_sources(self) -> List[Source]:
        """
        List all sources managed by the Manager.
        """
        sources = await self._db_client.get_sources()
        return sources

    async def create_new_source(self, image_provider: str, provider_params: Dict[str, Any], source_name: str) -> Source:
        """
        Create a new source from an image source as a preset image_provider/params

        Args:
            image_provider (str): The image provider for this source
            params (Dict[str, Any]): The parameters to pass to the image provider
            source_name (str): The name given for the new source
        """
        if image_provider not in image_provider_registry:
            raise ValueError(f"Unknown image provider: {image_provider}")

        ret: Source = await self._db_client.add_source(
            name=source_name, typename=image_provider, params=provider_params
        )
        return ret

    async def delete_sources(self, uuid_list: list[str]) -> None:
        """
        Delete sources by their UUIDs.

        Args:
            uuid_list (list[str]): List of source UUIDs to delete.
        """
        await self._db_client.delete_sources(uuid_list)

    async def update_source(
        self, source_uuid: str, image_provider: str, provider_params: Dict[str, Any], source_name: str
    ) -> Source:
        """
        Updates sources by their UUIDs.

        Args:
            uuid_list (list[str]): List of source UUIDs to delete.
        """
        return await self._db_client.update_source(
            source_uuid=source_uuid,
            name=source_name,
            typename=image_provider,
            params=provider_params,
        )

    async def list_avail_image_providers(self) -> List[str]:
        """
        List all available image providers.
        """
        return list(image_provider_registry.keys())

    async def get_image_provider_schema(self, image_provider: str) -> Dict[str, Any]:
        """
        Get the schema for a specific image provider.

        Args:
            image_provider (str): The name of the image provider.

        Returns:
            Dict[str, Any]: The schema for the image provider.
        """
        if image_provider not in image_provider_registry:
            raise ValueError(f"Unknown image provider: {image_provider}")

        return image_provider_registry[image_provider].params_schema()

    ################################################################################
    # TAGS API
    ################################################################################
    async def create_new_tag(
        self,
        name: str,
        color: Optional[str] = None,
        icon: Optional[str] = None,
        protected: bool = False,
        kind: TagKind = TagKind.GENERIC,
        exclusive_group: Optional[str] = None,
        uuid: Optional[str] = None,
    ) -> Tag:
        """
        Adds a new tag to the database
        """
        ret = await self._db_client.add_tag(
            name=name,
            color=color,
            icon=icon,
            protected=protected,
            kind=kind,
            exclusive_group=exclusive_group,
            uuid=uuid,
        )
        return ret

    async def delete_tag(self, tag_uuid: str) -> bool:
        """
        Deletes a tag from the database.
        Returns True if deleted, False if not found.
        """
        return await self._db_client.delete_tag(tag_uuid)

    async def update_tag(self, tag_uuid: str, update_data: TagUpdate) -> Optional[Tag]:
        """
        Update a tag's fields.

        Args:
            tag_uuid: The "Who" - used to find the row.
            update_data: The "What" - the validated fields to change.
        """
        changes = update_data.model_dump(exclude_unset=True)

        if not changes:
            return await self.get_tag_by_uuid(tag_uuid)

        return await self._db_client.update_tag(tag_uuid, **changes)

    async def get_tag_by_uuid(self, uuid: str) -> Optional[Tag]:
        """Retrieve a tag by its primary key UUID."""
        return await self._db_client.get_tag_by_uuid(uuid)

    async def get_tag_uuid_map(self) -> Dict[str, Tag]:
        """
        Get a mapping of tag UUIDs to their SQLModel objects.
        """
        tags: List[Tag] = await self._db_client.list_tags()
        return {tag.uuid: tag for tag in tags}

    async def get_tags(self) -> List[Tag]:
        """
        Gets a list of all tags
        """
        return await self._db_client.list_tags()

    ################################################################################
    # USER SETTINGS API
    ################################################################################

    async def get_setting(self, key: str) -> Optional[str]:
        """Return stored value for key, or None if not set."""
        return await self._db_client.get_setting(key)

    async def set_setting(self, key: str, value: str) -> None:
        """Upsert a user setting."""
        await self._db_client.set_setting(key, value)

    ################################################################################
    # IMAGES API
    ################################################################################

    async def get_image_metadata(self, image_path: Path) -> Optional[ImageRecordRead]:
        """
        Get metadata for image by resolving image ID from filename.

        Args:
            image_path (Path): The path to the image file.

        Returns:
            Optional[ImageRecordRead]: Full metadata object or None.
        """
        safe_path = get_safe_path(self._files_root, image_path)
        if not safe_path or not safe_path.exists():
            return None
        image_record = await self._db_client.get_image_by_filename(safe_path)
        if image_record is None or len(image_record.bboxes) == 0:
            await self._db_client.sync_image_from_json(safe_path)
            image_record = await self._db_client.get_image_by_filename(safe_path)
        if image_record is None:
            return None
        return ImageRecordRead.model_validate(image_record)

    def _get_json_sidefile(self, image_path: Path) -> Path:
        return image_path.with_suffix(".json")

    def _fixup_entry(self, entry):
        logger.info(f"TODO: Fixup entry {entry}")

    async def update_image_metadata(self, image_rel_path: str, update: ImageRecordUpdate) -> Optional[ImageRecordRead]:
        """
        Replace the bboxes for an image, persisting to both the DB and the JSON sidecar.

        The sidecar is written after re-reading the DB so that label and tag *names* are
        stored alongside UUIDs.  Names serve as fallback resolution keys if UUIDs change
        on a future DB rebuild.

        Args:
            image_rel_path (str): Relative image path.
            update (ImageRecordUpdate): New bbox data.

        Returns:
            The updated ImageRecordRead, or None if the image path is invalid.
        """
        safe_path = get_safe_path(self._files_root, image_rel_path)
        if safe_path is None or not safe_path.exists():
            logger.warning(f"Path not found or inaccessible: {image_rel_path}")
            return None
        image = await self._db_client.add_image(safe_path)
        await self._db_client.write_image_metadata_to_db(image.uuid, update)

        # Compute phashes on first write (image.phash is None for new images).
        image_phash: Optional[str] = image.phash
        if image_phash is None:
            try:
                image_bgr = await asyncio.to_thread(__import__("cv2").imread, str(safe_path))
                if image_bgr is not None:
                    image_phash = await asyncio.to_thread(
                        phash_from_numpy, image_bgr, self._image_hash_size
                    )
                    # Compute per-bbox phashes using the re-written update bboxes.
                    bbox_phash_list = []
                    for b in update.bboxes:
                        try:
                            bp = await asyncio.to_thread(
                                phash_from_numpy_crop,
                                image_bgr, b.x, b.y, b.width, b.height,
                                self._bbox_hash_size,
                            )
                            bbox_phash_list.append({"uuid": b.uuid, "phash": bp})
                        except Exception:
                            logger.exception("bbox phash error for %s", b.uuid)
                    await self._db_client.update_phashes(image.uuid, image_phash, bbox_phash_list)
                    # Update hash index.
                    self._hash_index.add_image(image.uuid, image_phash)
                    for bp_entry in bbox_phash_list:
                        label_uuid = next((b.label_uuid for b in update.bboxes if b.uuid == bp_entry["uuid"]), None)
                        if label_uuid:
                            self._hash_index.add_bbox(bp_entry["uuid"], label_uuid, bp_entry["phash"])
            except Exception:
                logger.exception("phash computation failed for %s", image_rel_path)
                image_phash = None

        # Re-read from DB; get_image_by_filename eager-loads bboxes → label + tags,
        # so we can include names in the sidecar for rebuild robustness.
        updated = await self._db_client.get_image_by_filename(safe_path)
        if updated:
            enriched_update = ImageRecordUpdate(
                bboxes=[
                    BBoxUpdate(
                        uuid=b.uuid,
                        label_uuid=b.label_uuid,
                        label_name=b.label.name if b.label else None,
                        x=b.x,
                        y=b.y,
                        width=b.width,
                        height=b.height,
                        tag_uuids=[t.uuid for t in (b.tags or [])],
                        tag_names=[t.name for t in (b.tags or [])],
                        phash=b.phash,
                    )
                    for b in updated.bboxes
                ]
            )
            await asyncio.to_thread(
                self._db_client.write_image_metadata_to_json,
                self._get_json_sidefile(safe_path),
                image.uuid,
                image.filename,
                enriched_update,
                updated.phash,
            )
            await self._recompute_image_labels(image.uuid, enriched_update)
        else:
            # DB re-read failed; fall back to the raw caller payload for label recomputation.
            await self._recompute_image_labels(image.uuid, update)
        return ImageRecordRead.model_validate(updated) if updated else None

    async def update_single_bbox(self, bbox_uuid: str, label_uuid: Optional[str], tag_uuids: List[str]) -> bool:
        """Update one bbox's label and tags, then sync sidecar and image_labels."""
        result = await self._db_client.get_bbox_with_image(bbox_uuid)
        if result is None or result[1] is None:
            return False
        bbox, image_record = result
        old_label_uuid = bbox.label_uuid
        await self._db_client.update_single_bbox(bbox_uuid, label_uuid, tag_uuids)
        # Move bbox in hash index if label changed and bbox has a phash.
        if label_uuid and label_uuid != old_label_uuid and bbox.phash and old_label_uuid:
            self._hash_index.move_bbox(bbox_uuid, old_label_uuid, label_uuid, bbox.phash)
        await self._sync_image_state(image_record)
        return True

    async def delete_single_bbox(self, bbox_uuid: str) -> bool:
        """Delete one bbox, then sync sidecar and image_labels."""
        result = await self._db_client.get_bbox_with_image(bbox_uuid)
        if result is None or result[1] is None:
            return False
        bbox, image_record = result
        # Tombstone in hash index before DB deletion.
        if bbox.label_uuid:
            self._hash_index.remove_bbox(bbox_uuid, bbox.label_uuid)
        await self._db_client.delete_single_bbox(bbox_uuid)
        await self._sync_image_state(image_record)
        return True

    async def _sync_image_state(self, image_record: ImageRecord) -> None:
        """Rebuild JSON sidecar and image_labels after a single-bbox mutation."""
        safe_path = get_safe_path(self._files_root, image_record.filename)
        # get_bboxes_for_image eager-loads both tags and label so names are available.
        bboxes = await self._db_client.get_bboxes_for_image(image_record.uuid)
        update = ImageRecordUpdate(
            bboxes=[
                BBoxUpdate(
                    uuid=b.uuid,
                    label_uuid=b.label_uuid,
                    label_name=b.label.name if b.label else None,
                    x=b.x,
                    y=b.y,
                    width=b.width,
                    height=b.height,
                    tag_uuids=[t.uuid for t in (b.tags or [])],
                    tag_names=[t.name for t in (b.tags or [])],
                )
                for b in bboxes
            ]
        )
        if safe_path is not None:
            await asyncio.to_thread(
                self._db_client.write_image_metadata_to_json,
                self._get_json_sidefile(safe_path),
                image_record.uuid,
                image_record.filename,
                update,
            )
        else:
            logger.warning(f"Skipping sidecar write for {image_record.filename!r}: path not under FILES_ROOT")
        await self._recompute_image_labels(image_record.uuid, update)

    async def _recompute_image_labels(self, image_uuid: str, update: ImageRecordUpdate) -> None:
        """Sync PRESENT rows in image_labels from reviewed bboxes.

        Only PRESENT rows are written/deleted here.  ABSENT rows are always
        set explicitly by the user via set_image_label and are never touched.

        Bboxes tagged UNVERIFIED or FALSE_POSITIVE are excluded from PRESENT tracking,
        as they have not been confirmed as true positive annotations.
        """
        present_label_uuids = {
            b.label_uuid for b in update.bboxes if b.label_uuid and not _BBOX_SKIP_TAGS.intersection(b.tag_uuids)
        }
        for label_uuid in present_label_uuids:
            await self._db_client.set_image_label(image_uuid, label_uuid, ImageLabelStatus.PRESENT)

        existing = await self._db_client.get_image_labels(image_uuid)
        stale_present = [
            r.label_uuid
            for r in existing
            if r.status == ImageLabelStatus.PRESENT and r.label_uuid not in present_label_uuids
        ]
        await self._db_client.delete_image_labels(image_uuid, stale_present)

    async def _resolve_image_uuid(self, image_path: str) -> Optional[str]:
        """Resolve an image path to its DB uuid, or None if not found."""
        safe_path = get_safe_path(self._files_root, image_path)
        if safe_path is None:
            return None
        return await self._db_client.get_image_uuid_by_filename(safe_path)

    async def set_image_label(self, image_path: str, label_uuid: str, status: ImageLabelStatus) -> bool:
        """Explicitly set a label state for an image.

        ABSENT creates/updates the row.  UNCERTAIN clears an existing explicit
        ABSENT row (deletes it so the label reverts to the implicit default).
        PRESENT cannot be set this way — it is always derived from verified bboxes.
        """
        image_uuid = await self._resolve_image_uuid(image_path)
        if image_uuid is None:
            return False
        if status == ImageLabelStatus.PRESENT:
            return False  # PRESENT is always derived from verified bboxes, never set directly
        if status == ImageLabelStatus.UNCERTAIN:
            await self._db_client.delete_image_labels(image_uuid, [label_uuid])
        else:
            await self._db_client.set_image_label(image_uuid, label_uuid, status)
        return True

    async def get_image_labels(self, image_path: str) -> Optional[List[ImageLabelRead]]:
        """Return all label statuses for an image."""
        image_uuid = await self._resolve_image_uuid(image_path)
        if image_uuid is None:
            return None
        rows = await self._db_client.get_image_labels(image_uuid)
        return [ImageLabelRead.model_validate(r) for r in rows]

    def _image_rel_path(self, image_filename: str) -> str:
        """Convert an absolute stored filename back to a FILES_ROOT-relative path."""
        try:
            return str(Path(image_filename).relative_to(self._files_root))
        except ValueError:
            return image_filename

    async def _count_images_bounded(self, params: ImageSearchParams) -> int:
        """Run count_images_search with a timeout; return -1 if it exceeds the limit.

        Uses asyncio.shield so the inner DB coroutine is NOT cancelled on timeout —
        it continues running in the background and properly closes its session when
        done.  This avoids leaving dangling SQLite connections while still letting
        the caller return quickly with -1 instead of blocking for minutes on large
        datasets.
        """
        t0 = time.monotonic()
        inner: asyncio.Task[int] = asyncio.ensure_future(
            self._db_client.count_images_search(params)
        )
        try:
            result = await asyncio.wait_for(asyncio.shield(inner), timeout=self._COUNT_QUERY_TIMEOUT)
            logger.info("[review] count query completed in %.3fs → %d", time.monotonic() - t0, result)
            return result
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            logger.warning(
                "[review] count query timed out after %.3fs (timeout=%.1fs); "
                "returning -1, background query continues",
                elapsed,
                self._COUNT_QUERY_TIMEOUT,
            )
            # Consume any future exception so Python doesn't warn about
            # "Task exception was never retrieved" when the task eventually ends.
            inner.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            return -1

    async def get_review_queue(self, params: ImageSearchParams) -> tuple[List[ImageRecordRead], bool, int]:
        """Return (images, has_more, total_count) for the review queue."""
        t0 = time.monotonic()
        logger.info(
            "[review] request: label_uuids=%s tag_uuids=%s require_bboxes=%s after_uuid=%s",
            params.label_uuids,
            params.tag_uuids,
            params.require_bboxes,
            params.after_uuid,
        )
        fetch_params = params.model_copy(update={"limit": params.limit + 1})

        if params.after_uuid is None:
            cache_key = hashlib.md5(
                params.model_dump_json(
                    exclude={"sort_by", "sort_order", "limit", "after_uuid", "after_sort_value"}
                ).encode(),
                usedforsecurity=False,
            ).hexdigest()
            now = time.monotonic()
            cached = self._count_cache.get(cache_key)

            if cached is not None and (now - cached[1]) < self._COUNT_CACHE_TTL:
                logger.info("[review] count cache hit (key=%s, age=%.1fs)", cache_key[:8], now - cached[1])
                t1 = time.monotonic()
                batch = await self._db_client.list_images_search(fetch_params)
                logger.info("[review] list query completed in %.3fs → %d rows", time.monotonic() - t1, len(batch))
                total_count = cached[0]
            else:
                logger.info("[review] count cache miss (key=%s) — running list+count in parallel", cache_key[:8])
                t1 = time.monotonic()
                batch, total_count = await asyncio.gather(
                    self._db_client.list_images_search(fetch_params),
                    self._count_images_bounded(params),
                )
                logger.info(
                    "[review] gather done in %.3fs: list=%d rows, total_count=%d",
                    time.monotonic() - t1,
                    len(batch),
                    total_count,
                )
                if total_count >= 0:
                    self._count_cache[cache_key] = (total_count, now)
        else:
            t1 = time.monotonic()
            batch = await self._db_client.list_images_search(fetch_params)
            logger.info("[review] list query (paginated) completed in %.3fs → %d rows", time.monotonic() - t1, len(batch))
            total_count = -1

        has_more = len(batch) > params.limit
        images_page = batch[: params.limit]

        results = []
        for img in images_page:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)

        logger.info("[review] total handler time %.3fs → %d images, has_more=%s", time.monotonic() - t0, len(results), has_more)
        return results, has_more, total_count

    async def list_files(
        self,
        rel_path: str = "/",
        patterns: List[str] = ["*"],
        exclude_patterns: List[str] = ["*.json"],
        recursive: bool = False,
        include_db_status: bool = False,
        offset: int = 0,
        limit: int = 200,
    ) -> tuple[List[FileEntryRead], bool]:
        """
        List files and directories in the given relative path under the root directory.

        Args:
            rel_path (str): Relative path from the root directory (default: "/").
            patterns (List[str]): Glob pattern(s) to include (default: "*").
            exclude_patterns (List[str]): Glob pattern(s) to exclude (default: "").
            recursive (bool): Whether to list files recursively (default: False).
            include_db_status (bool): When True, populate uuid on entries that are
                registered in the image database (default: False).
            offset (int): Number of entries to skip (default: 0).
            limit (int): Maximum number of entries to return (default: 200).

        Returns:
            Tuple of (entries page, has_more flag).
        """
        if not self._files_root or not self._files_root.is_dir():
            return [], False

        safe_path = get_safe_path(self._files_root, rel_path)
        if not safe_path or not safe_path.is_dir():
            return [], False

        search_pattern = "**/*" if recursive else None

        # Collect and sort all matching paths by name (string sort — no stat calls yet).
        raw = sorted(safe_path.glob(search_pattern) if recursive else safe_path.iterdir(), key=lambda p: p.name)

        filtered = []
        for p in raw:
            if p.name.startswith("."):
                continue
            rel_str = str(p.relative_to(self._files_root))
            is_matched = any(fnmatch.fnmatch(rel_str, pat) for pat in patterns)
            is_excluded = any(fnmatch.fnmatch(rel_str, pat) for pat in exclude_patterns)
            if is_matched and not is_excluded:
                filtered.append(p)

        has_more = len(filtered) > offset + limit
        page = filtered[offset : offset + limit]

        # Stat calls (is_dir) and FileEntryRead construction only for the page slice.
        entries = [FileEntryRead(name=p.name, is_dir=p.is_dir(), path=str(p.relative_to(self._files_root))) for p in page]

        if include_db_status and entries:
            abs_paths = [str(self._files_root / e.path) for e in entries if not e.is_dir]
            in_db_map = await self._db_client.get_filenames_in_db(abs_paths)
            for e in entries:
                if not e.is_dir:
                    info = in_db_map.get(str(self._files_root / e.path))
                    if info:
                        e.uuid = info["uuid"]
                        e.bbox_count = info["bbox_count"]

        return entries, has_more

    async def create_directory(self, rel_path: str) -> None:
        """
        Create a directory under FILES_ROOT.

        Args:
            rel_path: Path relative to FILES_ROOT (must resolve inside it).

        Raises:
            ValueError: If the path resolves outside FILES_ROOT or already exists as a file.
        """
        if not self._files_root or not self._files_root.is_dir():
            raise ValueError("FILES_ROOT is not configured")
        safe_path = get_safe_path(self._files_root, rel_path)
        if safe_path is None:
            raise ValueError(f"Path is outside FILES_ROOT: {rel_path!r}")
        if safe_path.exists() and not safe_path.is_dir():
            raise ValueError(f"Path already exists as a file: {rel_path!r}")
        safe_path.mkdir(parents=True, exist_ok=True)

    ################################################################################
    # Image Reconciliation API
    ################################################################################

    async def list_all_image_stubs(self) -> List[Dict[str, str]]:
        """Return [{uuid, filename}] for every image record without pagination."""
        return await self._db_client.get_all_image_stubs()

    async def list_image_stubs_needing_phash(self, image_hex_len: int, bbox_hex_len: int) -> List[Dict[str, str]]:
        """Return [{uuid, filename}] for images whose phash is missing or was computed at a different size."""
        return await self._db_client.get_image_stubs_needing_phash(image_hex_len, bbox_hex_len)

    async def reconcile_image(self, rel_path: str, sync_mode: str) -> Dict[str, Any]:
        """Move-aware reconcile of a single image file against the DB.

        1. Reads the sidecar to get the image UUID.
        2. If that UUID exists in the DB under a different filename and the old
           file is gone, updates the record's filename (move detection).
        3. Applies sync_mode: replace_db_with_json | replace_json_with_db |
           merge | keep_newer.

        Returns a dict with "action" and "rel_path".
        """
        image_path = get_safe_path(self._files_root, rel_path)
        if image_path is None:
            return {"action": "skipped", "reason": "path outside files_root", "rel_path": rel_path}

        # Sidecars replace the image extension: image.jpg → image.json (matches _get_json_sidefile)
        sidecar_path = self._get_json_sidefile(image_path)

        if not sidecar_path.exists():
            logger.debug("reconcile_image: no sidecar for %s, skipping", rel_path)
            return {"action": "skipped", "reason": "no sidecar", "rel_path": rel_path}

        try:
            sidecar_data = jstyleson.loads(sidecar_path.read_text())
        except Exception as e:
            logger.warning("reconcile_image: failed to parse sidecar %s: %s", sidecar_path, e)
            return {"action": "error", "reason": str(e), "rel_path": rel_path}

        sidecar_uuid = sidecar_data.get("uuid")
        if not sidecar_uuid:
            return {"action": "skipped", "reason": "sidecar has no uuid", "rel_path": rel_path}

        # Move detection: UUID exists but under a different absolute filename.
        # The DB stores absolute paths (e.g. /files_root/subdir/img.png).
        existing = await self._db_client.get_image(sidecar_uuid)
        action = "synced"
        if existing is not None and existing.filename != str(image_path):
            old_abs = (
                Path(existing.filename)
                if Path(existing.filename).is_absolute()
                else get_safe_path(self._files_root, existing.filename)
            )
            if old_abs is None or not old_abs.exists():
                # The old file is gone — this is a move.
                logger.info("reconcile_image: move detected %s → %s", existing.filename, image_path)
                await self._db_client.update_image_filename(sidecar_uuid, str(image_path))
                action = "moved"

        # Apply sync_mode.
        # Every branch that mutates the DB calls _sync_image_state so that
        # image_labels stays consistent with the new bbox set.
        if sync_mode == "replace_db_with_json":
            record = await self._db_client.sync_image_from_json(image_path, overwrite=True)
            await self._sync_image_state(record)
        elif sync_mode == "replace_json_with_db":
            # DB wins: write current DB bboxes back to the sidecar.
            updated_record = await self._db_client.get_image(sidecar_uuid)
            if updated_record:
                await self._sync_image_state(updated_record)
        elif sync_mode == "merge":
            record = await self._db_client.sync_image_from_json(image_path, overwrite=False)
            # Write the merged result back to the sidecar.
            await self._sync_image_state(record)
        elif sync_mode == "keep_newer":
            updated_record = await self._db_client.get_image(sidecar_uuid)
            if updated_record and updated_record.updated_at:
                db_ts = updated_record.updated_at
                if db_ts.tzinfo is None:
                    db_ts = db_ts.replace(tzinfo=timezone.utc)
                sidecar_mtime = datetime.fromtimestamp(sidecar_path.stat().st_mtime, tz=timezone.utc)
                if sidecar_mtime > db_ts:
                    # Sidecar is newer — import from JSON and rebuild image_labels.
                    record = await self._db_client.sync_image_from_json(image_path, overwrite=True)
                    await self._sync_image_state(record)
                else:
                    # DB is newer — write DB state back to the sidecar.
                    await self._sync_image_state(updated_record)
            else:
                # No DB record yet — import from JSON.
                record = await self._db_client.sync_image_from_json(image_path, overwrite=True)
                await self._sync_image_state(record)
        else:
            return {"action": "error", "reason": f"unknown sync_mode: {sync_mode}", "rel_path": rel_path}

        return {"action": action, "rel_path": rel_path}

    async def prune_missing_images(self, known_disk_uuids: List[str]) -> int:
        """Delete DB records for images whose file is missing and whose UUID is not on disk.

        The known_disk_uuids set is the set of UUIDs seen in sidecars during the
        reconcile pass.  Any DB record whose file is absent AND whose UUID is not in
        that set is truly gone (not just moved) and can be safely deleted.

        Returns the number of records deleted.
        """
        known_set: Set[str] = set(known_disk_uuids)
        stubs = await self._db_client.get_all_image_stubs()
        deleted = 0
        for stub in stubs:
            uuid = stub["uuid"]
            filename = stub["filename"]
            if uuid in known_set:
                continue
            abs_path = get_safe_path(self._files_root, filename)
            if abs_path is not None and abs_path.exists():
                continue
            logger.info("prune_missing_images: deleting %s (%s)", filename, uuid)
            # Remove from hash index before DB deletion.
            image_record = await self._db_client.get_image_by_uuid(uuid)
            if image_record:
                if image_record.phash:
                    self._hash_index.remove_image(uuid)
                for bbox in (image_record.bboxes or []):
                    if bbox.phash and bbox.label_uuid:
                        self._hash_index.remove_bbox(bbox.uuid, bbox.label_uuid)
            await self._db_client.delete_image(uuid)
            deleted += 1
        return deleted

    # ─────────────────────────────────────────────────────────────────────────
    # Perceptual hash index public API
    # ─────────────────────────────────────────────────────────────────────────

    def find_similar_images(self, phash: str, threshold: int) -> List[Tuple[str, int]]:
        """Return [(image_uuid, distance)] for images within *threshold* of *phash*."""
        return self._hash_index.find_similar_images(phash, threshold)

    def find_similar_bboxes(
        self,
        phash: str,
        label_uuid: Optional[str],
        threshold: int,
    ) -> List[Tuple[str, int, str]]:
        """Return [(bbox_uuid, distance, label_uuid)] within *threshold* of *phash*."""
        return self._hash_index.find_similar_bboxes(phash, label_uuid, threshold)

    async def find_similar_bboxes_enriched(
        self,
        phash: str,
        label_uuid: Optional[str],
        threshold: int,
    ) -> List[Tuple[str, Optional[str], int, str]]:
        """Return [(bbox_uuid, image_uuid, distance, label_uuid)] within *threshold* of *phash*.

        Like find_similar_bboxes but enriches each result with its parent image UUID via a
        single batch query, so callers never have to deal with a missing image_uuid.
        """
        matches = self._hash_index.find_similar_bboxes(phash, label_uuid, threshold)
        if not matches:
            return []
        bbox_uuids = [b_uuid for b_uuid, _dist, _lbl in matches]
        image_uuid_map = await self._db_client.get_image_uuids_for_bboxes(bbox_uuids)
        return [(b_uuid, image_uuid_map.get(b_uuid), dist, lbl) for b_uuid, dist, lbl in matches]

    async def find_similar_images_by_uuid(self, image_uuid: str, threshold: int) -> List[Tuple[str, int]]:
        """Return [(image_uuid, distance)] for images similar to the given image UUID, including the query image itself (distance 0)."""
        image = await self._db_client.get_image_by_uuid(image_uuid)
        if image is None or image.phash is None:
            return []
        return self._hash_index.find_similar_images(image.phash, threshold)

    async def find_similar_bboxes_by_uuid(
        self,
        bbox_uuid: str,
        label_uuid: Optional[str],
        threshold: int,
    ) -> List[Tuple[str, Optional[str], int, str]]:
        """Return [(bbox_uuid, image_uuid, distance, label_uuid)] for bboxes similar to the given bbox UUID, including the query bbox itself (distance 0)."""
        result = await self._db_client.get_bbox_with_image(bbox_uuid)
        if result is None:
            return []
        bbox, _ = result
        if bbox.phash is None:
            return []
        matches = self._hash_index.find_similar_bboxes(bbox.phash, label_uuid, threshold)
        if not matches:
            return []
        # Enrich with image UUIDs via a single batch query
        bbox_uuids = [b_uuid for b_uuid, _dist, _lbl in matches]
        image_uuid_map = await self._db_client.get_image_uuids_for_bboxes(bbox_uuids)
        return [(b_uuid, image_uuid_map.get(b_uuid), dist, lbl) for b_uuid, dist, lbl in matches]

    async def find_similar_bboxes_by_coords(
        self,
        image_uuid: str,
        x: float,
        y: float,
        w: float,
        h: float,
        label_uuid: Optional[str],
        threshold: int,
    ) -> List[Tuple[str, Optional[str], int, str]]:
        """Compute a phash from a (possibly unsaved) bbox region and return similar bboxes."""
        image = await self._db_client.get_image_by_uuid(image_uuid)
        if image is None:
            return []
        image_path = Path(image.filename)
        if not image_path.is_absolute():
            image_path = self._files_root / image_path
        phash = compute_bbox_phash(image_path, x, y, w, h)
        return await self.find_similar_bboxes_enriched(phash, label_uuid, threshold)

    async def get_images_by_uuids(self, image_uuids: List[str]) -> List[ImageRecordRead]:
        """Return full ImageRecordRead objects for a list of image UUIDs, in any order."""
        records = await self._db_client.get_images_by_uuids(image_uuids)
        result = []
        for img in records:
            read = ImageRecordRead.model_validate(img)
            read.filename = self._image_rel_path(img.filename)
            result.append(read)
        return result

    async def get_status(self) -> dict:
        """Return service health and diagnostic info."""
        notices: list[dict] = []
        if not self._hash_index_ready:
            phase_detail = {
                "loading": " (loading hashes from DB…)",
                "constructing": " (constructing search trees…)",
            }.get(self._build_phase, "…")
            notices.append({
                "key": "hash_build",
                "level": "info",
                "message": f"Building similarity index{phase_detail} — similar search unavailable until complete",
            })
        elif self._rebuild_in_progress:
            notices.append({"key": "hash_rebuild", "level": "warning", "message": "Hash index rebuilding — similar search may return stale results"})
        return {
            "service": "db_server",
            "status": "ok",
            "hash_index": self._hash_index.stats(),
            "notices": notices,
        }

    async def get_image_filename_by_uuid(self, image_uuid: str) -> Optional[str]:
        """Return the relative filename for an image UUID, or None if not found."""
        image = await self._db_client.get_image_by_uuid(image_uuid)
        return self._image_rel_path(image.filename) if image else None

    def check_dedup_bulk(self, image_phash: str, image_threshold: int, bboxes: list) -> dict:
        """Bulk duplicate check: one image phash + N bbox phashes → one in-memory call."""
        return self._hash_index.check_dedup_bulk(image_phash, image_threshold, bboxes)

    async def rebuild_hash_index(self) -> dict:
        """Rebuild the BK-tree index from the DB and return stats."""
        self._rebuild_in_progress = True
        try:
            stats = await self._hash_index.rebuild()
        finally:
            self._rebuild_in_progress = False
        return {
            "image_count": stats.image_count,
            "bbox_count": stats.bbox_count,
            "duration_ms": stats.duration_ms,
        }

    async def recompute_image_phash(
        self,
        image_uuid: str,
        image_hash_size: Optional[int] = None,
        bbox_hash_size: Optional[int] = None,
    ) -> Optional[dict]:
        """Re-read image from disk, recompute phashes, update DB + index.

        Returns {"image_phash": str, "bbox_count": int} on success, None if not found.
        """
        import cv2 as _cv2

        img_hs = image_hash_size or self._image_hash_size
        bbox_hs = bbox_hash_size or self._bbox_hash_size

        image_record = await self._db_client.get_image_by_uuid(image_uuid)
        if image_record is None:
            return None

        safe_path = get_safe_path(self._files_root, image_record.filename)
        if safe_path is None or not safe_path.exists():
            logger.warning("recompute_image_phash: file not found for %s", image_uuid)
            return None

        image_bgr = await asyncio.to_thread(_cv2.imread, str(safe_path))
        if image_bgr is None:
            logger.warning("recompute_image_phash: cv2.imread failed for %s", safe_path)
            return None

        image_phash = await asyncio.to_thread(phash_from_numpy, image_bgr, img_hs)

        bbox_phash_list = []
        for bbox in (image_record.bboxes or []):
            try:
                bp = await asyncio.to_thread(
                    phash_from_numpy_crop,
                    image_bgr, bbox.x, bbox.y, bbox.width, bbox.height, bbox_hs,
                )
                bbox_phash_list.append({"uuid": bbox.uuid, "phash": bp})
            except Exception:
                logger.exception("bbox phash error for %s", bbox.uuid)

        await self._db_client.update_phashes(image_uuid, image_phash, bbox_phash_list)

        # Remove old hash index entries, then re-add updated ones.
        if image_record.phash:
            self._hash_index.remove_image(image_uuid)
        for bbox in (image_record.bboxes or []):
            if bbox.phash and bbox.label_uuid:
                self._hash_index.remove_bbox(bbox.uuid, bbox.label_uuid)

        self._hash_index.add_image(image_uuid, image_phash)
        for bp_entry in bbox_phash_list:
            bbox_obj = next((b for b in (image_record.bboxes or []) if b.uuid == bp_entry["uuid"]), None)
            if bbox_obj and bbox_obj.label_uuid:
                self._hash_index.add_bbox(bp_entry["uuid"], bbox_obj.label_uuid, bp_entry["phash"])

        # Refresh sidecar with updated phashes.
        updated_record = await self._db_client.get_image_by_filename(safe_path)
        if updated_record:
            enriched_update = ImageRecordUpdate(
                bboxes=[
                    BBoxUpdate(
                        uuid=b.uuid,
                        label_uuid=b.label_uuid,
                        label_name=b.label.name if b.label else None,
                        x=b.x,
                        y=b.y,
                        width=b.width,
                        height=b.height,
                        tag_uuids=[t.uuid for t in (b.tags or [])],
                        tag_names=[t.name for t in (b.tags or [])],
                        phash=b.phash,
                    )
                    for b in updated_record.bboxes
                ]
            )
            await asyncio.to_thread(
                self._db_client.write_image_metadata_to_json,
                self._get_json_sidefile(safe_path),
                image_uuid,
                image_record.filename,
                enriched_update,
                image_phash,
            )

        return {"image_phash": image_phash, "bbox_count": len(bbox_phash_list)}

    async def clear_orphan_generated_tags(self) -> int:
        """Delete TASK_GENERATED tags with no associated bboxes.

        Returns the number of tags deleted.
        """
        return await self._db_client.clear_orphan_generated_tags()

    ################################################################################
    # TASK CONFIGS API
    ################################################################################

    async def create_task_config(
        self,
        name: str,
        typename: str,
        params: TaskParams,
        description: Optional[str] = None,
    ) -> TaskConfig:
        return await self._db_client.add_task_config(
            name=name, typename=typename, params=params, description=description
        )

    async def list_task_configs(
        self,
        config_uuids: Optional[List[str]] = None,
        typename: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, TaskConfig]:
        return await self._db_client.get_task_configs(config_uuids=config_uuids, typename=typename, name=name)

    async def update_task_config(
        self,
        config_uuid: str,
        name: Optional[str] = None,
        typename: Optional[str] = None,
        params: Optional[TaskParams] = None,
        description: Optional[str] = None,
        marked_for_delete: Optional[bool] = None,
    ) -> None:
        await self._db_client.update_task_config(
            config_uuid=config_uuid,
            name=name,
            typename=typename,
            params=params,
            description=description,
            marked_for_delete=marked_for_delete,
        )

    async def delete_task_configs(self, config_uuids: List[str]) -> None:
        await self._db_client.delete_task_config(config_uuids)

    ################################################################################
    # TASK INSTANCES API
    ################################################################################

    async def create_task_instance(self, config_uuid: str) -> TaskInstance:
        return await self._db_client.insert_new_active_task(config_uuid)

    async def list_task_instances(
        self,
        task_uuids: Optional[List[str]] = None,
        resumable: Optional[bool] = None,
    ) -> List[TaskInstance]:
        return await self._db_client.get_tasks(task_uuid=task_uuids, resumable=resumable)

    async def delete_task_instances(self, task_uuids: List[str]) -> None:
        await self._db_client.delete_active_tasks(task_uuids)

    async def set_task_status(self, instance_uuid: str, status: TaskStatus) -> None:
        await self._db_client.set_task_status(instance_uuid, status)

    async def set_task_result(self, instance_uuid: str, result: TaskResultData) -> None:
        await self._db_client.set_task_result(instance_uuid, result)

    async def set_task_resume_data(self, instance_uuid: str, resume_data: TaskResumeData) -> None:
        await self._db_client.set_task_resume_data(instance_uuid, resume_data)

    async def set_task_timestamps_and_logs(
        self,
        instance_uuid: str,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
        log_json: Optional[List[str]] = None,
    ) -> None:
        await self._db_client.set_task_timestamps_and_logs(
            instance_uuid, started_at=started_at, ended_at=ended_at, log_json=log_json
        )

    async def update_task_heartbeat(self, instance_uuid: str) -> None:
        await self._db_client.update_task_heartbeat(instance_uuid)

    async def get_task_results(self, task_uuids: List[str]) -> Dict[str, TaskResultData]:
        results = await self._db_client.get_task_results(task_uuids)
        return results or {}

    ################################################################################
    # THUMBNAIL API
    ################################################################################

    def _thumb_cache_dir(self) -> Path:
        return CACHE_DIR / "thumb_cache"

    def _thumb_cache_path(self, bbox_uuid: str, width: int) -> Path:
        return self._thumb_cache_dir() / f"{bbox_uuid}_{width}.png"

    def _trim_thumb_cache_sync(self, max_bytes: int) -> None:
        """Evict least-recently-used thumbnail files until total cache size is under max_bytes."""
        cache_dir = self._thumb_cache_dir()
        if not cache_dir.exists():
            return
        entries = []
        total = 0
        for p in cache_dir.glob("*.png"):
            try:
                st = p.stat()
                entries.append((st.st_mtime, st.st_size, p))
                total += st.st_size
            except FileNotFoundError:
                pass
        if total <= max_bytes:
            return
        entries.sort()  # oldest mtime first — evict these first
        for _, size, path in entries:
            if total <= max_bytes:
                break
            try:
                path.unlink(missing_ok=True)
                total -= size
                logger.info(f"Thumbnail cache evicted: {path.name}")
            except Exception as e:
                logger.warning(f"Failed to evict thumbnail {path.name}: {e}")

    def _schedule_thumb_cache_trim(self) -> None:
        """Fire-and-forget: trim the cache in a background thread if a size limit is set."""
        if THUMB_CACHE_MAX_MB <= 0:
            return
        max_bytes = THUMB_CACHE_MAX_MB * 1024 * 1024
        asyncio.create_task(asyncio.to_thread(self._trim_thumb_cache_sync, max_bytes))

    def _resolve_image_file_path(self, image_path: str) -> Optional[Path]:
        """Resolve an image path under _files_root; falls back to treating it as an absolute path."""
        safe = get_safe_path(self._files_root, Path(image_path))
        if safe is not None and safe.is_file():
            return safe
        fallback = Path(image_path)
        return fallback if fallback.is_file() else None

    async def get_bbox_thumbnail(self, bbox_uuid: str, width: int = 200) -> Optional[bytes]:
        """
        Return JPEG bytes for a cropped+resized thumbnail of the given bbox.
        Serves from disk cache if available and fresh, otherwise generates and caches it.
        Cache freshness is validated against bbox.updated_at so edits made outside the
        review page (e.g. from the image-tagging canvas) are always reflected.
        """
        from PIL import Image as PILImage
        from PIL import ImageDraw

        # Fetch bbox first so we can validate cache freshness against bbox.updated_at.
        result = await self._db_client.get_bbox_with_image(bbox_uuid)
        if result is None:
            return None
        bbox, image_record = result
        if image_record is None:
            return None

        cache_path = self._thumb_cache_path(bbox_uuid, width)
        if cache_path.exists():
            try:
                cache_mtime = cache_path.stat().st_mtime
                if cache_mtime >= bbox.updated_at.timestamp():
                    cache_path.touch()  # update mtime for LRU tracking
                    async with aiofiles.open(cache_path, "rb") as f:
                        return await f.read()
                # Cache is older than the bbox — delete it and regenerate below.
                cache_path.unlink(missing_ok=True)
            except OSError:
                pass

        safe_path = self._resolve_image_file_path(image_record.filename)
        if safe_path is None:
            return None

        # Extract coords and look up label color before the thread (avoids detached ORM objects).
        bx, by, bw, bh = bbox.x, bbox.y, bbox.width, bbox.height
        label_color: tuple[int, int, int] = (255, 215, 0)  # fallback gold
        if bbox.label_uuid:
            label = await self._db_client.get_label_by_uuid(bbox.label_uuid)
            if label and label.color:
                try:
                    h = label.color.lstrip("#")
                    label_color = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
                except (ValueError, IndexError):
                    pass

        def _generate() -> Optional[bytes]:
            try:
                with PILImage.open(safe_path) as img:
                    img = img.convert("RGB")
                    img_w, img_h = img.size
                    pad_x = bw * img_w * 0.2
                    pad_y = bh * img_h * 0.2
                    left = max(0, bx * img_w - pad_x)
                    top = max(0, by * img_h - pad_y)
                    right = min(img_w, (bx + bw) * img_w + pad_x)
                    bottom = min(img_h, (by + bh) * img_h + pad_y)
                    cropped = img.crop((left, top, right, bottom))
                    draw = ImageDraw.Draw(cropped)
                    draw.rectangle(
                        [bx * img_w - left, by * img_h - top, (bx + bw) * img_w - left, (by + bh) * img_h - top],
                        outline=label_color,
                        width=3,
                    )
                    del draw
                    aspect = cropped.width / cropped.height if cropped.height else 1
                    height = max(1, round(width / aspect))
                    cropped = cropped.resize((width, height), PILImage.LANCZOS)
                    buf = io.BytesIO()
                    cropped.save(buf, format="PNG")
                    return buf.getvalue()
            except Exception as e:
                logger.warning(f"Failed to generate thumbnail for bbox {bbox_uuid}: {e}")
                return None

        jpeg_bytes = await asyncio.to_thread(_generate)
        if jpeg_bytes is None:
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(cache_path, "wb") as f:
            await f.write(jpeg_bytes)

        self._schedule_thumb_cache_trim()
        return jpeg_bytes

    async def invalidate_bbox_thumbnails(self, bbox_uuids: List[str]) -> None:
        """Delete all cached thumbnail sizes for the given bbox UUIDs."""
        cache_dir = self._thumb_cache_dir()

        def _remove_files() -> None:
            for uuid in bbox_uuids:
                for path in cache_dir.glob(f"{uuid}_*.png"):
                    try:
                        path.unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"Failed to delete thumbnail cache for {uuid}: {e}")

        await asyncio.to_thread(_remove_files)

    def _image_thumb_cache_path(self, image_path: str, width: int) -> Path:
        key = hashlib.sha1(image_path.encode()).hexdigest()
        return self._thumb_cache_dir() / f"img_{key}_{width}.png"

    async def get_image_thumbnail(self, image_path: str, width: int = 640) -> Optional[bytes]:
        """Return PNG bytes for a full-image thumbnail, resized to `width` px.
        Serves from disk cache if available, otherwise generates and caches it.
        """
        from PIL import Image as PILImage

        cache_path = self._image_thumb_cache_path(image_path, width)
        if cache_path.exists():
            try:
                cache_path.touch()  # update mtime for LRU tracking
            except OSError:
                pass
            async with aiofiles.open(cache_path, "rb") as f:
                return await f.read()

        safe_path = self._resolve_image_file_path(image_path)
        if safe_path is None:
            return None

        def _generate() -> Optional[bytes]:
            try:
                with PILImage.open(safe_path) as img:
                    img = img.convert("RGB")
                    aspect = img.width / img.height if img.height else 1
                    height = max(1, round(width / aspect))
                    resized = img.resize((width, height), PILImage.LANCZOS)
                    buf = io.BytesIO()
                    resized.save(buf, format="PNG")
                    return buf.getvalue()
            except Exception as e:
                logger.warning(f"Failed to generate image thumbnail for {image_path}: {e}")
                return None

        png_bytes = await asyncio.to_thread(_generate)
        if png_bytes is None:
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(cache_path, "wb") as f:
            await f.write(png_bytes)

        self._schedule_thumb_cache_trim()
        return png_bytes

    async def open_file(self, rel_path: str) -> Optional[Tuple[aiofiles.threadpool.binary.AsyncBufferedReader, str]]:
        """
        Securely open a file under the root and return an aiofiles stream and filename.

        Args:
            rel_path (str): Relative path under the image root.

        Returns:
            Tuple[aiofiles.AsyncBufferedReader, str] or None
        """
        if not self._files_root or not os.path.isdir(self._files_root):
            logger.warning("Root directory not set or invalid")
            return None

        try:
            safe_path = get_safe_path(self._files_root, rel_path)

            if safe_path is None:
                logger.warning(f"Blocked directory traversal attempt: {rel_path}")
                return None

            if not safe_path.is_file():
                logger.warning(f"File not found: {safe_path}")
                return None

            f = await aiofiles.open(safe_path, mode="rb")
            filename = safe_path.name
            return f, filename

        except Exception as e:
            logger.error(f"Failed to open file {rel_path}: {e}", exc_info=True)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset management
    # ─────────────────────────────────────────────────────────────────────────

    async def list_datasets(self) -> List[DatasetRead]:
        return await self._db_client.list_datasets()

    async def get_dataset(self, uuid: str) -> Optional[DatasetRead]:
        return await self._db_client.get_dataset(uuid)

    async def create_dataset(self, data: DatasetCreate) -> DatasetRead:
        return await self._db_client.create_dataset(data)

    async def update_dataset(self, uuid: str, data: DatasetUpdate) -> Optional[DatasetRead]:
        return await self._db_client.update_dataset(uuid, data)

    async def delete_dataset(self, uuid: str) -> bool:
        return await self._db_client.delete_dataset(uuid)

    async def get_dataset_images(
        self,
        uuid: str,
        *,
        limit: int = 50,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
        after_uuid: Optional[str] = None,
        after_sort_value: Optional[datetime] = None,
    ) -> Optional[tuple[List[ImageRecordRead], bool]]:
        result = await self._db_client.list_dataset_images(
            uuid,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
            after_uuid=after_uuid,
            after_sort_value=after_sort_value,
        )
        if result is None:
            return None
        images, has_more = result
        records = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            records.append(record)
        return records, has_more

    async def get_dataset_labels(self, uuid: str) -> Optional[List[Label]]:
        """Return the Label objects for a dataset's relevant labels."""
        dataset = await self._db_client.get_dataset(uuid)
        if dataset is None:
            return None
        if not dataset.label_uuids:
            return []
        labels_map = await self._db_client.get_labels_by_uuids(dataset.label_uuids)
        return list(labels_map.values())

    async def preview_dataset_images(
        self,
        filter_json: dict,
        label_uuids: List[str],
        *,
        limit: int = 500,
    ) -> tuple[List[ImageRecordRead], bool]:
        """Execute filter criteria in-memory without saving a dataset (unsaved preview)."""
        images, has_more = await self._db_client.preview_dataset_images(filter_json, label_uuids, limit=limit)
        records = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            records.append(record)
        return records, has_more

    # ─── Model Configurations ─────────────────────────────────────────────────

    def get_available_model_typenames(self) -> List[str]:
        return list(AVAILABLE_MODEL_TYPENAMES)

    async def _validate_model_config_nodes(self, nodes: List[ModelConfigNodeWrite]) -> None:
        """Validate typename and label_uuid for every node write object."""
        for node in nodes:
            if node.model_typename not in AVAILABLE_MODEL_TYPENAMES:
                raise ValueError(
                    f"Unknown model typename: {node.model_typename!r}. " f"Valid choices: {AVAILABLE_MODEL_TYPENAMES}"
                )
            if node.label_uuid is not None:
                label = await self._db_client.get_label_by_uuid(node.label_uuid)
                if label is None:
                    raise ValueError(f"Label not found: {node.label_uuid!r}")

    async def list_model_configs(self) -> List[ModelConfigRead]:
        return await self._db_client.list_model_configs()

    async def get_model_config(self, uuid: str) -> Optional[ModelConfigRead]:
        return await self._db_client.get_model_config(uuid)

    async def create_model_config(self, data: ModelConfigCreate) -> ModelConfigRead:
        await self._validate_model_config_nodes(data.nodes)
        return await self._db_client.create_model_config(data)

    async def update_model_config(self, uuid: str, data: ModelConfigUpdate) -> Optional[ModelConfigRead]:
        if data.nodes is not None:
            await self._validate_model_config_nodes(data.nodes)
        return await self._db_client.update_model_config(uuid, data)

    async def delete_model_config(self, uuid: str) -> bool:
        return await self._db_client.delete_model_config(uuid)
