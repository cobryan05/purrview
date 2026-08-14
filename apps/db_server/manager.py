"""Manages the database"""

import asyncio
import fnmatch
import hashlib
import io
import logging
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
from uuid import uuid4

import aiofiles
import jstyleson
from sqlalchemy.orm import make_transient

from apps import CACHE_DIR, STREAM_BUFFER_RETENTION_S, THUMB_CACHE_MAX_MB
from apps.db_server.hash_index import HashIndex
from apps.helpers.consts import TaskStatus
from apps.helpers.db.db_client import BBOX_VERIFICATION_STATE_TO_TAG_UUID, DbClient, TagKind
from apps.helpers.db.types import (
    MODEL_TYPE_ARCHITECTURES,
    UNVERIFIED_TAG_UUID,
    VERIFIED_TAG_UUID,
    BBox,
    BBoxConfidenceRange,
    BBoxUpdate,
    DatasetCreate,
    DatasetRead,
    DatasetUpdate,
    FilterChainCreate,
    FilterChainRead,
    FilterChainUpdate,
    ImageItemStatus,
    ImageLabelRead,
    ImageLabelWithFilename,
    ImageRecord,
    ImageRecordRead,
    ImageRecordUpdate,
    ImageSearchParams,
    ImageTagStatusRead,
    Label,
    LabelScope,
    LabelType,
    ModelConfigCreate,
    ModelConfigNodeWrite,
    ModelConfigRead,
    ModelConfigUpdate,
    ModelGroupCreate,
    ModelGroupRead,
    ModelGroupUpdate,
    PreviewApplyResult,
    PreviewCandidate,
    PreviewCandidateBulkUpdate,
    PreviewCandidateCreate,
    PreviewCandidateQuery,
    PreviewCandidateRead,
    PreviewConflictPolicy,
    PreviewDecision,
    PreviewDecisionAction,
    PreviewRunCreate,
    PreviewRunRead,
    PreviewRunUpdate,
    RecordingRead,
    Source,
    Tag,
    TagRead,
    TagUpdate,
    TaskConfig,
    TaskConfigRead,
    TaskInstance,
    TaskInstanceRead,
    TaskParams,
    TaskResultData,
    TaskResumeData,
    WatchEventRead,
    WatchProfileCreate,
    WatchProfileRead,
    WatchProfileUpdate,
    confidence_in_band,
)
from apps.helpers.fileUtils import get_safe_path
from apps.helpers.hashUtils import (
    compute_bbox_phash,
    compute_image_phash,
    phash_from_numpy,
    phash_from_numpy_crop,
)
from apps.helpers.imageProviders.Registry import image_provider_registry
from apps.helpers.serviceStatus import row, section
from apps.helpers.types import FileEntryRead

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

# Only bboxes explicitly carrying this tag produce PRESENT image-label rows.
_BBOX_VERIFIED_TAG = VERIFIED_TAG_UUID

# Sentinel for Manager.update_label's parent_uuid param: `None` is itself a
# meaningful value there (promote to root), so it can't also mean "not supplied".
_PARENT_UUID_NOT_SET = object()

# Crop-thumbnail outline weight, as a fraction of the RENDERED width (3px on the
# 300px crops the review grid asks for). See Manager._render_outlined_crop for
# why it can't be a source-pixel constant.
_OUTLINE_WIDTH_RATIO = 0.01
_OUTLINE_MIN_PX = 2


def _decode_image_bytes(data: bytes):
    """Decode uploaded bytes, or None if they are not a readable image.

    Deliberately decodes with the same library that will later read the file
    off disk, rather than trusting the extension or sniffing magic bytes: the
    question worth answering is not "is this a PNG?" but "can this app open
    it?", and those differ for truncated or subtly-corrupt files.

    Runs on a worker thread (see the caller) — decoding a large image is slow
    enough to stall the event loop.
    """
    import cv2
    import numpy as np

    try:
        buffer = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    except Exception:
        logger.exception("Uploaded bytes could not be decoded")
        return None


class Manager:
    """Manages interactions with the database"""

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks
    BULK_APPLY_PAGE_SIZE: int = 200  # Keyset page size for bulk_apply_bboxes (class attr so tests can shrink it)

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
        self._build_phase: str | None = None  # "migrating", "loading", or "constructing" during startup
        self._build_img_loaded: int = 0  # running count during loading phase
        self._build_bbox_loaded: int = 0
        self._db_ready: bool = False  # True after init_db (alembic) succeeds; gates normal requests
        self._init_error: str | None = None  # Set if _init() fails; surfaced as an error notice
        # Short-lived cache for expensive count queries.  Key is an MD5 of the
        # filter predicates (sort/page fields excluded); value is (count, ts).
        # TTL is intentionally short so writes that happen while browsing are
        # reflected within a few seconds when the user hits Refresh.
        self._count_cache: dict[str, tuple[int, float]] = {}
        self._COUNT_CACHE_TTL = 60.0
        # Expired entries are only ever overwritten on an identical key, so a browsing
        # session that keeps trying new filter combinations would grow this dict
        # forever. Sweep expired keys once the dict crosses a soft ceiling.
        self._COUNT_CACHE_MAX_ENTRIES = 512
        # Thumbnail generation decodes the *full-resolution* source image (PIL
        # open + convert) even when the output is a small bbox crop. Dispatched
        # via asyncio.to_thread that runs on the default executor, so a grid view
        # firing dozens of thumbnail requests at once could hold
        # min(32, cpu+4) full-res decodes in memory simultaneously — hundreds of
        # MB of spike for 4K sources. Cap the concurrent decodes instead.
        self._thumb_decode_sem = asyncio.Semaphore(int(os.environ.get("THUMB_DECODE_CONCURRENCY", "4")))
        # Held reference to the in-flight cache-trim task: asyncio only keeps weak
        # references to tasks, so a bare create_task() can be garbage-collected
        # mid-run. Doubles as a coalescing flag — one trim at a time is plenty.
        self._thumb_trim_task: Optional[asyncio.Task] = None
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
        self._build_phase = "migrating"
        try:
            await self._db_client.init_db()
        except Exception as e:
            self._init_error = str(e)
            self._build_phase = None
            logger.error("Database initialization (alembic) failed: %s", e, exc_info=True)
            raise  # re-raise so _worker_task stops and doesn't enter the polling loop
        self._build_phase = None
        self._db_ready = True

        # Hash sizes are stored in DB settings so the review page, tasks, and the
        # db_server all reference the same value.  Read them here (after init_db
        # ensures the settings table exists) and override the env-var defaults set
        # in __init__.  Falls back silently to the env-var value when the setting
        # is absent (e.g. first run before the user has opened the settings page).
        raw_img = await self._db_client.get_setting("dedup_image_hash_size")
        raw_bbox = await self._db_client.get_setting("dedup_bbox_hash_size")
        if raw_img is not None:
            try:
                self._image_hash_size = int(raw_img)
            except (ValueError, TypeError):
                pass
        if raw_bbox is not None:
            try:
                self._bbox_hash_size = int(raw_bbox)
            except (ValueError, TypeError):
                pass
        # Propagate the resolved sizes to the HashIndex before building.
        self._hash_index._image_hash_size = self._image_hash_size
        self._hash_index._bbox_hash_size = self._bbox_hash_size

        def _on_phase(p: str) -> None:
            self._build_phase = p
            if p == "constructing":
                self._build_img_loaded = 0
                self._build_bbox_loaded = 0

        def _on_progress(img: int, bbox: int) -> None:
            self._build_img_loaded = img
            self._build_bbox_loaded = bbox

        await self._hash_index.build(on_phase=_on_phase, on_progress=_on_progress)
        self._build_phase = None
        self._build_img_loaded = 0
        self._build_bbox_loaded = 0
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
        except Exception as e:
            # _init() already set _init_error and logged before re-raising.
            # The worker stops here; the error notice remains visible in /status.
            logger.error("Worker task terminated after initialization failure: %s", e)
        finally:
            print("Periodic task cleanup.")

    ################################################################################
    # LABELS API
    ################################################################################

    async def create_new_label(
        self,
        name: str,
        color: str,
        parent_uuid: Optional[str] = None,
        label_type: LabelType = LabelType.STANDARD,
        scope: LabelScope = LabelScope.OBJECT,
    ) -> Label:
        """
        Adds a new label to the database

        Returns:
            Label: Label added to database
        """
        ret = await self._db_client.add_label(name, color, parent_uuid=parent_uuid, label_type=label_type, scope=scope)
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
        scope: Optional[LabelScope] = None,
        parent_uuid: Any = _PARENT_UUID_NOT_SET,
    ) -> None:
        """
        Update a label's name, color, label_type, scope, and/or parent.

        Args:
            label_uuid (str): uuid of the label to update.
            name (Optional[str]): New name for the label.
            color (Optional[str]): New color for the label.
            label_type (Optional[LabelType]): New label type.
            scope (Optional[LabelScope]): New label scope.
            parent_uuid: New parent uuid ("move" the label), or None to promote
                the label to root. Uses the sentinel default `_PARENT_UUID_NOT_SET`
                (rather than `None`) because `None` is itself a meaningful value
                here (promote to root) distinct from "the caller didn't ask to
                change the parent at all".

        Raises:
            ValueError: if `parent_uuid` would make the label its own parent, or
                a descendant of itself (a cycle).
        """
        updates = {
            k: v
            for k, v in {
                "name": name,
                "color": color,
                "label_type": label_type,
                "scope": scope,
            }.items()
            if v is not None
        }

        if parent_uuid is not _PARENT_UUID_NOT_SET:
            if parent_uuid is not None:
                await self._assert_no_reparent_cycle(label_uuid, parent_uuid)
            updates["parent_uuid"] = parent_uuid

        await self._db_client.update_label(label_uuid=label_uuid, **updates)
        # TODO CJO: await self._db_client.export_labels_from_db_to_json(str(self._labels_json))

    async def _assert_no_reparent_cycle(self, label_uuid: str, new_parent_uuid: str) -> None:
        """
        Walk up from `new_parent_uuid` toward the root. If we encounter
        `label_uuid` along the way (including immediately, i.e. new_parent_uuid
        == label_uuid), the move would make the label its own ancestor — reject it.

        Complexity: O(V) to build the uuid->label map (get_label_uuid_map already
        does this walk once) + O(depth) for the upward walk.
        """
        uuid_to_node = await self.get_label_uuid_map()
        current: Optional[str] = new_parent_uuid
        seen: Set[str] = set()
        while current is not None:
            if current == label_uuid:
                raise ValueError(
                    f"Cannot re-parent label '{label_uuid}': the target parent "
                    f"'{new_parent_uuid}' is the label itself or one of its own descendants."
                )
            if current in seen:
                break  # defensive: pre-existing cycle in stored data; don't loop forever
            seen.add(current)
            node = uuid_to_node.get(current)
            current = node.parent_uuid if node else None

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
        self,
        source_uuid: str,
        image_provider: str,
        provider_params: Dict[str, Any],
        source_name: str,
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

        # Capture old bbox entries BEFORE the DB write so we can remove stale hash
        # index entries.  write_image_metadata_to_db does a full DELETE + INSERT, so
        # old phashes are gone once it returns.
        old_image_record = await self._db_client.get_image_by_uuid(image.uuid)
        old_bboxes = old_image_record.bboxes if old_image_record else []

        # Carry each bbox's stored metadata_json across the DELETE+INSERT unless
        # the caller explicitly supplied one. The canvas save posts only
        # geometry/label/tags, so without this every save silently wiped
        # source_model / source_confidence / run_model_added — and
        # relabel_prev_label_uuid, which "revert filter-chain relabel" reads.
        # `{}` still clears, so a caller can deliberately empty it.
        old_metadata = {b.uuid: (b.metadata_json or {}) for b in old_bboxes}
        for b in update.bboxes:
            if b.metadata_json is None:
                b.metadata_json = old_metadata.get(b.uuid, {})

        await self._db_client.write_image_metadata_to_db(image.uuid, update)

        # Tombstone all previous bbox entries from the hash index (covers
        # deleted, moved, and resized boxes). This must happen outside the
        # phash-recompute conditional below: deleting the LAST bbox leaves
        # update.bboxes empty, skipping recompute entirely — but the old
        # entries still have to go, or deleted boxes keep matching in
        # similarity searches.
        for old_bbox in old_bboxes:
            if old_bbox.phash and old_bbox.label_uuid:
                self._hash_index.remove_bbox(old_bbox.uuid, old_bbox.label_uuid)

        # Image phash is computed once (first write only — the file never changes).
        # Bbox phashes must always be (re-)computed because the user may have moved or
        # resized boxes since the last save, which changes the cropped pixel region.
        image_phash: Optional[str] = image.phash
        if image_phash is None or update.bboxes:
            try:
                image_bgr = await asyncio.to_thread(__import__("cv2").imread, str(safe_path))
                if image_bgr is not None:
                    if image_phash is None:
                        image_phash = await asyncio.to_thread(phash_from_numpy, image_bgr, self._image_hash_size)
                    # Always (re-)compute per-bbox phashes.
                    bbox_phash_list = []
                    for b in update.bboxes:
                        try:
                            bp = await asyncio.to_thread(
                                phash_from_numpy_crop,
                                image_bgr,
                                b.x,
                                b.y,
                                b.width,
                                b.height,
                                self._bbox_hash_size,
                            )
                            bbox_phash_list.append({"uuid": b.uuid, "phash": bp})
                        except Exception:
                            logger.exception("bbox phash error for %s", b.uuid)
                    await self._db_client.update_phashes(image.uuid, image_phash, bbox_phash_list)
                    # Update hash index (old entries were tombstoned above).
                    if image.phash is None and image_phash:
                        self._hash_index.add_image(image.uuid, image_phash)
                    uuid_to_label = {b.uuid: b.label_uuid for b in update.bboxes}
                    for bp_entry in bbox_phash_list:
                        label_uuid = uuid_to_label.get(bp_entry["uuid"])
                        if label_uuid:
                            self._hash_index.add_bbox(bp_entry["uuid"], label_uuid, bp_entry["phash"])
            except Exception:
                logger.exception("phash computation failed for %s", image_rel_path)
                image_phash = image.phash  # revert to previously-stored value on error

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

    async def update_single_bbox(
        self,
        bbox_uuid: str,
        label_uuid: Optional[str],
        tag_uuids: List[str],
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update one bbox's label and tags, then sync sidecar and image_labels."""
        result = await self._db_client.get_bbox_with_image(bbox_uuid)
        if result is None or result[1] is None:
            return False
        bbox, image_record = result
        old_label_uuid = bbox.label_uuid
        await self._db_client.update_single_bbox(bbox_uuid, label_uuid, tag_uuids, metadata_json=metadata_json)
        # Move bbox in hash index if label changed and bbox has a phash.
        if label_uuid and label_uuid != old_label_uuid and bbox.phash and old_label_uuid:
            self._hash_index.move_bbox(bbox_uuid, old_label_uuid, label_uuid, bbox.phash)
        await self._sync_image_state(image_record)
        return True

    async def add_single_bbox(
        self,
        image_rel_path: str,
        label_uuid: Optional[str],
        x: float,
        y: float,
        width: float,
        height: float,
        tag_uuids: List[str],
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Insert one new bbox on an image (leaving siblings untouched), then sync
        its phash / hash-index entry / JSON sidecar / image_labels. Returns the new
        bbox uuid, or None if the image path is invalid.

        Unlike update_image_metadata (a full bbox replace that re-writes and
        re-phashes every box on the image), this is a targeted single-row insert —
        the right primitive for programmatic callers such as the filter-chain
        add_bbox action.
        """
        safe_path = get_safe_path(self._files_root, image_rel_path)
        if safe_path is None or not safe_path.exists():
            logger.warning(f"Path not found or inaccessible: {image_rel_path}")
            return None
        image = await self._db_client.add_image(safe_path)
        bbox_uuid = str(uuid4())
        await self._db_client.add_single_bbox(
            image.uuid, bbox_uuid, label_uuid, x, y, width, height, tag_uuids, metadata_json=metadata_json
        )

        # Compute the new crop's phash and register it in the hash index (mirrors
        # the per-bbox phash bookkeeping in update_image_metadata).
        try:
            image_bgr = await asyncio.to_thread(__import__("cv2").imread, str(safe_path))
            if image_bgr is not None:
                bp = await asyncio.to_thread(
                    phash_from_numpy_crop, image_bgr, x, y, width, height, self._bbox_hash_size
                )
                await self._db_client.update_phashes(image.uuid, image.phash, [{"uuid": bbox_uuid, "phash": bp}])
                if label_uuid and bp:
                    self._hash_index.add_bbox(bbox_uuid, label_uuid, bp)
        except Exception:
            logger.exception("phash computation failed for new bbox on %s", image_rel_path)

        image_record = await self._db_client.get_image_by_uuid(image.uuid)
        if image_record is not None:
            await self._sync_image_state(image_record)
        return bbox_uuid

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

    async def bulk_apply_bboxes(
        self,
        search_params: ImageSearchParams,
        action: str,
        label_uuid: Optional[str],
        tag_uuids: List[str],
        bbox_tag_filter: List[str],
        bbox_label_filter: List[str],
        show_all_bboxes: bool = False,
        bbox_verification: Optional[str] = None,
        bbox_confidence: Optional[BBoxConfidenceRange] = None,
    ) -> Tuple[int, int]:
        """Apply action to every bbox matching the filter, across all pages.

        Pages through list_images_search in batches of 200, applies the requested
        action to each matching bbox, and calls _sync_image_state once per image.

        bbox_tag_filter / bbox_label_filter / bbox_verification mirror the
        client-side bbox subset logic:
          - bbox_tag_filter: bbox must carry at least one of these tag UUIDs
            (ignored when show_all_bboxes is True or the list is empty)
          - bbox_label_filter: bbox label_uuid must be in this set
            (ignored when the list is empty)
          - bbox_verification: bbox must carry the matching system verification
            tag. ANDed with bbox_tag_filter rather than folded into it — that
            list is OR-semantics, so merging the verification tag in would
            widen the scope instead of narrowing it.
          - bbox_confidence: bbox's stored model confidence must fall in the
            band. Same reason it is a bbox-level argument rather than only a
            filter term: the term picks images that HAVE an in-band bbox, which
            would still let the action hit their out-of-band boxes.

        Returns (modified_count, fail_count).
        """
        tag_filter_set: Optional[Set[str]] = (
            None if show_all_bboxes else (set(bbox_tag_filter) if bbox_tag_filter else None)
        )
        label_filter_set: Optional[Set[str]] = set(bbox_label_filter) if bbox_label_filter else None
        # 'any' is the UI's no-op tri-state value; treat it like an omitted filter.
        verification_tag_uuid: Optional[str] = (
            BBOX_VERIFICATION_STATE_TO_TAG_UUID.get(bbox_verification) if bbox_verification else None
        )
        confidence_band = bbox_confidence if (bbox_confidence and bbox_confidence.is_active()) else None

        # Paginate by uuid regardless of the caller's requested sort. The action
        # mutates image.updated_at, so keyset-paginating by updated_at (asc) would
        # bump every processed row past the cursor and re-match it on the next
        # page — an infinite loop that never terminates. Traversal order is
        # irrelevant to the action (each matching bbox is touched exactly once).
        search_params = search_params.model_copy(update={"sort_by": "uuid", "sort_order": "asc"})

        modified_count = 0
        fail_count = 0
        _PAGE_SIZE = self.BULK_APPLY_PAGE_SIZE
        cursor_uuid: Optional[str] = None
        cursor_sort_value = None

        while True:
            page_params = search_params.model_copy(
                update={
                    "limit": _PAGE_SIZE + 1,
                    "after_uuid": cursor_uuid,
                    "after_sort_value": cursor_sort_value,
                }
            )
            batch = await self._db_client.list_images_search(page_params)
            has_more = len(batch) > _PAGE_SIZE
            batch = batch[:_PAGE_SIZE]

            if not batch:
                break

            for img in batch:
                # Replicate the client-side bbox subset filter.
                matching = []
                for bbox in img.bboxes or []:
                    if tag_filter_set is not None and not any(t.uuid in tag_filter_set for t in bbox.tags or []):
                        continue
                    if label_filter_set is not None and bbox.label_uuid not in label_filter_set:
                        continue
                    if verification_tag_uuid is not None and not any(
                        t.uuid == verification_tag_uuid for t in bbox.tags or []
                    ):
                        continue
                    if confidence_band is not None and not confidence_in_band(
                        # The generated column, already loaded with the row —
                        # identical to re-parsing metadata_json, minus the parse.
                        bbox.confidence,
                        confidence_band.min,
                        confidence_band.max,
                        confidence_band.include_missing,
                    ):
                        continue
                    matching.append(bbox)

                if not matching:
                    continue

                any_modified = False
                for bbox in matching:
                    try:
                        await self._apply_bbox_bulk_action(bbox, action, label_uuid, tag_uuids)
                        modified_count += 1
                        any_modified = True
                    except Exception as e:
                        logger.error("bulk_apply_bboxes: bbox %s failed: %s", bbox.uuid, e)
                        fail_count += 1

                if any_modified:
                    try:
                        await self._sync_image_state(img)
                    except Exception as e:
                        logger.error(
                            "bulk_apply_bboxes: _sync_image_state failed for %s: %s",
                            img.uuid,
                            e,
                        )

            if not has_more:
                break

            last = batch[-1]
            sort_by = search_params.sort_by
            cursor_uuid = last.uuid
            cursor_sort_value = (
                (last.updated_at if sort_by == "updated_at" else last.created_at)
                if sort_by not in ("uuid", "filename")
                else None
            )

        return modified_count, fail_count

    async def count_bboxes_matching(
        self,
        search_params: ImageSearchParams,
        bbox_tag_filter: List[str],
        bbox_label_filter: List[str],
        show_all_bboxes: bool = False,
        bbox_verification: Optional[str] = None,
        bbox_confidence: Optional[BBoxConfidenceRange] = None,
    ) -> int:
        """How many bboxes bulk_apply_bboxes would touch for this exact filter.

        Same argument shape as bulk_apply_bboxes so the number the UI shows and
        the number it then modifies cannot drift.
        """
        verification_tag_uuid = (
            BBOX_VERIFICATION_STATE_TO_TAG_UUID.get(bbox_verification) if bbox_verification else None
        )
        return await self._db_client.count_bboxes_search(
            search_params,
            bbox_tag_filter=bbox_tag_filter,
            bbox_label_filter=bbox_label_filter,
            show_all_bboxes=show_all_bboxes,
            bbox_verification_tag_uuid=verification_tag_uuid,
            bbox_confidence=bbox_confidence,
        )

    async def _apply_bbox_bulk_action(
        self,
        bbox: BBox,
        action: str,
        label_uuid: Optional[str],
        tag_uuids: List[str],
        geometry: Optional[Dict[str, float]] = None,
        mark_verified: bool = True,
    ) -> None:
        """Apply one bulk action (relabel/delete/addTags/removeTags/revertRelabel) to a single bbox.

        Shared by bulk_apply_bboxes (filter-derived candidates) and
        bulk_apply_bboxes_by_uuids (explicit-list candidates) so the two entry
        points can't drift. Raises on failure — caller is responsible for
        catching and counting it.

        `mark_verified` governs the relabel branch only. A relabel is normally a
        human decision, so it promotes the box to VERIFIED; callers applying a
        *model's* suggestion ("Run Model on Results") pass False so the box keeps
        whatever verification state it already had and stays reviewable.
        """
        _VERIFICATION_UUIDS: Set[str] = {UNVERIFIED_TAG_UUID}
        current_tags = [t.uuid for t in bbox.tags or []]
        if action == "delete":
            if bbox.label_uuid:
                self._hash_index.remove_bbox(bbox.uuid, bbox.label_uuid)
            await self._db_client.delete_single_bbox(bbox.uuid)
        elif action == "addTags":
            # Pass [existing, new] — db_client resolves exclusive-group conflicts
            # so new tags win any group conflict.
            await self._db_client.update_single_bbox(bbox.uuid, None, [*current_tags, *tag_uuids])
        elif action == "removeTags":
            remove_set = set(tag_uuids)
            new_tags = [u for u in current_tags if u not in remove_set]
            await self._db_client.update_single_bbox(bbox.uuid, None, new_tags)
        elif action == "revertRelabel":
            # Undo a filter-chain `relabel_bbox` step: restore the label stashed in
            # metadata_json by ApplyFilterChainTask and drop the review tag(s)
            # (tag_uuids — typically the caller's current tag filter).
            metadata = dict(bbox.metadata_json or {})
            prev_label = metadata.pop("relabel_prev_label_uuid", None)
            if prev_label is None:
                raise ValueError(f"bbox {bbox.uuid} has no relabel_prev_label_uuid to revert")
            remove_set = set(tag_uuids)
            new_tags = [u for u in current_tags if u not in remove_set]
            if bbox.label_uuid and prev_label != bbox.label_uuid and bbox.phash:
                self._hash_index.move_bbox(bbox.uuid, bbox.label_uuid, prev_label, bbox.phash)
            await self._db_client.update_single_bbox(bbox.uuid, prev_label, new_tags, metadata_json=metadata)
        else:  # relabel
            assigning = bool(label_uuid)
            new_label = label_uuid if assigning else bbox.label_uuid
            metadata_json = None
            if assigning:
                new_tags = (
                    [u for u in current_tags if u not in _VERIFICATION_UUIDS] + [VERIFIED_TAG_UUID]
                    if mark_verified
                    else current_tags
                )
                if bbox.label_uuid and label_uuid != bbox.label_uuid and bbox.phash:
                    self._hash_index.move_bbox(bbox.uuid, bbox.label_uuid, label_uuid, bbox.phash)
                # A manual relabel supersedes any prior filter-chain relabel: drop the
                # stashed original label so a later "revert filter-chain relabel" can't
                # silently restore a now-stale label over the user's manual choice.
                if bbox.metadata_json and "relabel_prev_label_uuid" in bbox.metadata_json:
                    metadata_json = {k: v for k, v in bbox.metadata_json.items() if k != "relabel_prev_label_uuid"}
            else:
                new_tags = current_tags
            await self._db_client.update_single_bbox(
                bbox.uuid, new_label, new_tags, metadata_json=metadata_json, geometry=geometry if assigning else None
            )

    async def bulk_apply_bboxes_by_uuids(
        self,
        bbox_uuids: List[str],
        action: str,
        label_uuid: Optional[str],
        tag_uuids: List[str],
        bbox_geometry: Optional[Dict[str, Dict[str, float]]] = None,
        mark_verified: bool = True,
    ) -> Tuple[int, int]:
        """Apply action to an explicit list of bboxes in one server-side pass.

        Used by review modals whose candidate set has no server-side filter
        equivalent (e.g. similarity/diversity search results ranked by phash
        distance) — the caller already knows the exact UUID list and only needs
        the action applied without round-tripping one HTTP call per bbox.

        `bbox_geometry`, if given, maps bbox_uuid -> {"x","y","width","height"} —
        used by "Run Model on Results" relabels to replace a bbox's coordinates
        with the refinement model's suggested box instead of keeping the original
        (the model that ran is assumed to be more precise than whatever produced
        the original box).

        `mark_verified=False` relabels without promoting the box to VERIFIED —
        used when the relabel comes from a model suggestion rather than a human.

        Returns (modified_count, fail_count).
        """
        modified_count = 0
        fail_count = 0
        synced_images: Dict[str, ImageRecord] = {}

        for bbox_uuid in bbox_uuids:
            result = await self._db_client.get_bbox_with_image(bbox_uuid)
            if result is None or result[1] is None:
                fail_count += 1
                continue
            bbox, img = result
            try:
                geometry = bbox_geometry.get(bbox_uuid) if bbox_geometry else None
                await self._apply_bbox_bulk_action(
                    bbox, action, label_uuid, tag_uuids, geometry=geometry, mark_verified=mark_verified
                )
                modified_count += 1
                synced_images[img.uuid] = img
            except Exception as e:
                logger.error("bulk_apply_bboxes_by_uuids: bbox %s failed: %s", bbox_uuid, e)
                fail_count += 1

        for img in synced_images.values():
            try:
                await self._sync_image_state(img)
            except Exception as e:
                logger.error(
                    "bulk_apply_bboxes_by_uuids: _sync_image_state failed for %s: %s",
                    img.uuid,
                    e,
                )

        return modified_count, fail_count

    async def _apply_image_bulk_action(
        self,
        image_uuid: str,
        action: str,
        label_uuids: List[str],
        tag_uuids: List[str],
    ) -> str:
        """Apply one image-level bulk action to a single image uuid.

        Shared by bulk_apply_images_by_uuids (explicit-list candidates) and
        bulk_apply_images (filter-derived candidates) so the two entry points
        can't drift.

        Label actions accept several labels and apply to each in turn. The
        return value stays per-image so the counts callers report keep meaning
        "images", not "(image, label) pairs": "skipped" is returned only when
        *every* requested label was skipped, "modified" when at least one landed.

        Returns "modified" or "skipped" ("skipped" == setAbsent refused because
        the image already has a verified bbox for that label — a legitimate
        skip, not a failure). Raises on real failure — caller is responsible
        for catching and counting it as a fail.
        """
        if action == "delete":
            rel_path = await self.delete_image_permanently(image_uuid)
            if rel_path is None:
                raise ValueError(f"image {image_uuid} not found or already deleted")
        elif action == "removeFromDb":
            # Forget the image without touching the file: DB record + bboxes go
            # away, the sidecar goes with them (otherwise reconcile resurrects
            # the record), and the image file itself stays on disk.
            rel_path = await self.delete_image_db_only(image_uuid)
            if rel_path is None:
                raise ValueError(f"image {image_uuid} not found or already removed")
        elif action == "setUncertain":
            await self._db_client.delete_image_labels(image_uuid, list(label_uuids))
        elif action == "addTags":
            await self._db_client.add_image_tags(image_uuid, tag_uuids)
        elif action == "removeTags":
            await self._db_client.remove_image_tags(image_uuid, tag_uuids)
        elif action == "setTagAbsent":
            for tag_uuid in tag_uuids:
                await self._db_client.set_image_tag_status(image_uuid, tag_uuid, ImageItemStatus.ABSENT)
        elif action == "setTagUncertain":
            for tag_uuid in tag_uuids:
                await self._db_client.set_image_tag_status(image_uuid, tag_uuid, ImageItemStatus.UNCERTAIN)
        elif action == "setAbsent":
            # Mirror the single-image path's rule (set_image_label_status): ABSENT is
            # refused when a verified bbox already contradicts it. A bulk run treats
            # that as a skip, not a failure. Batched into one query + one upsert per
            # image regardless of label count, rather than a round trip per label.
            verified = await self._db_client.labels_with_verified_bbox(image_uuid, label_uuids)
            applicable = [label_uuid for label_uuid in label_uuids if label_uuid not in verified]
            if applicable:
                await self._db_client.set_image_labels_status(image_uuid, applicable, ImageItemStatus.ABSENT)
            if label_uuids and not applicable:
                return "skipped"
        elif action == "setPresent":
            await self._db_client.set_image_labels_status(image_uuid, label_uuids, ImageItemStatus.PRESENT)
        else:
            raise ValueError(f"Unknown bulk image action: {action}")
        return "modified"

    async def bulk_apply_images_by_uuids(
        self,
        image_uuids: List[str],
        action: str,
        label_uuids: Optional[List[str]] = None,
        tag_uuids: Optional[List[str]] = None,
    ) -> Tuple[int, int, int]:
        """Apply an image-level bulk action (setPresent/setAbsent/setUncertain/delete/
        addTags/removeTags/setTagAbsent/setTagUncertain) to an explicit list of images
        in one server-side pass — the image-kind counterpart of bulk_apply_bboxes_by_uuids.

        Returns (modified_count, fail_count, skipped_count).
        """
        tag_uuids = tag_uuids or []
        label_uuids = label_uuids or []
        if action in ("addTags", "removeTags", "setTagAbsent", "setTagUncertain") and not tag_uuids:
            # Nothing to do — don't report images as "modified" for a no-op.
            return 0, 0, 0
        if action in ("setPresent", "setAbsent", "setUncertain") and not label_uuids:
            # Same no-op guard as the tag actions above: a label action with no
            # labels would otherwise count every image as "modified".
            return 0, 0, 0
        modified_count = 0
        fail_count = 0
        skipped_count = 0
        for image_uuid in image_uuids:
            try:
                result = await self._apply_image_bulk_action(image_uuid, action, label_uuids, tag_uuids)
                if result == "skipped":
                    skipped_count += 1
                else:
                    modified_count += 1
            except Exception as e:
                logger.error("bulk_apply_images_by_uuids: image %s failed: %s", image_uuid, e)
                fail_count += 1
        return modified_count, fail_count, skipped_count

    async def bulk_apply_images(
        self,
        search_params: ImageSearchParams,
        action: str,
        label_uuids: Optional[List[str]] = None,
        tag_uuids: Optional[List[str]] = None,
    ) -> Tuple[int, int, int]:
        """Apply an image-level action to every image matching the filter, across all pages.

        Pages through list_images_search in batches of self.BULK_APPLY_PAGE_SIZE and applies
        the requested action to each image — the image-kind counterpart of bulk_apply_bboxes.

        Returns (modified_count, fail_count, skipped_count).
        """
        tag_uuids = tag_uuids or []
        label_uuids = label_uuids or []
        if action in ("addTags", "removeTags", "setTagAbsent", "setTagUncertain") and not tag_uuids:
            return 0, 0, 0
        if action in ("setPresent", "setAbsent", "setUncertain") and not label_uuids:
            return 0, 0, 0

        # Paginate by uuid regardless of the caller's requested sort — mirrors
        # bulk_apply_bboxes: the action can mutate rows the cursor depends on
        # (e.g. updated_at), so keyset-paginating by anything but uuid risks
        # re-matching an already-processed row on the next page. Traversal
        # order is irrelevant to the action (each matching image is touched
        # exactly once).
        search_params = search_params.model_copy(update={"sort_by": "uuid", "sort_order": "asc"})

        modified_count = 0
        fail_count = 0
        skipped_count = 0
        _PAGE_SIZE = self.BULK_APPLY_PAGE_SIZE
        cursor_uuid: Optional[str] = None
        cursor_sort_value = None

        while True:
            page_params = search_params.model_copy(
                update={
                    "limit": _PAGE_SIZE + 1,
                    "after_uuid": cursor_uuid,
                    "after_sort_value": cursor_sort_value,
                }
            )
            batch = await self._db_client.list_images_search(page_params)
            has_more = len(batch) > _PAGE_SIZE
            batch = batch[:_PAGE_SIZE]

            if not batch:
                break

            for img in batch:
                try:
                    result = await self._apply_image_bulk_action(img.uuid, action, label_uuids, tag_uuids)
                    if result == "skipped":
                        skipped_count += 1
                    else:
                        modified_count += 1
                except Exception as e:
                    logger.error("bulk_apply_images: image %s failed: %s", img.uuid, e)
                    fail_count += 1

            if not has_more:
                break

            last = batch[-1]
            sort_by = search_params.sort_by
            cursor_uuid = last.uuid
            # sort_by is forced to "uuid" above, so this is always None — kept in
            # this shape (rather than hardcoding None) to mirror bulk_apply_bboxes
            # exactly, in case that forcing is ever relaxed for either function.
            cursor_sort_value = (
                (last.updated_at if sort_by == "updated_at" else last.created_at)
                if sort_by not in ("uuid", "filename")
                else None
            )

        return modified_count, fail_count, skipped_count

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
        set explicitly by the user via set_image_label_status and are never touched.

        Bboxes tagged UNVERIFIED are excluded from PRESENT tracking, as they have
        not been confirmed as true annotations.

        The stale-PRESENT sweep below only applies to object-scope labels, whose
        PRESENT state is owned by bbox sync (derived from verified bboxes). Image-scope
        labels can never have a bbox — the UI short-circuits the draw tool for them — so
        their PRESENT state is user-asserted, exactly like ABSENT, and must never be
        treated as stale here. A candidate whose label no longer exists (deleted) is not
        exempt: it's an orphan and is still swept.
        """
        present_label_uuids = {
            b.label_uuid for b in update.bboxes if b.label_uuid and _BBOX_VERIFIED_TAG in b.tag_uuids
        }
        for label_uuid in present_label_uuids:
            await self._db_client.set_image_label_status(image_uuid, label_uuid, ImageItemStatus.PRESENT)

        existing = await self._db_client.get_image_labels(image_uuid)
        stale_candidates = [
            r.label_uuid
            for r in existing
            if r.status == ImageItemStatus.PRESENT and r.label_uuid not in present_label_uuids
        ]
        stale_present = stale_candidates
        if stale_candidates:
            labels = await self._db_client.get_labels_by_uuids(stale_candidates)
            stale_present = [
                u for u in stale_candidates if labels.get(u) is None or labels[u].scope != LabelScope.IMAGE
            ]
        await self._db_client.delete_image_labels(image_uuid, stale_present)

    async def _resolve_image_uuid(self, image_path: str) -> Optional[str]:
        """Resolve an image path to its DB uuid, or None if not found."""
        safe_path = get_safe_path(self._files_root, image_path)
        if safe_path is None:
            return None
        return await self._db_client.get_image_uuid_by_filename(safe_path)

    async def set_image_label_status(self, image_path: str, label_uuid: str, status: ImageItemStatus) -> str:
        """Explicitly set a label state for an image.

        PRESENT and ABSENT create/update the row.  UNCERTAIN clears an existing
        row (deletes it so the label reverts to the implicit default).

        ABSENT is blocked when the image already has a verified bbox with the
        same label — that would be contradictory evidence, not a failure. This
        mirrors the bulk paths' "setAbsent" handling in _apply_image_bulk_action:
        same predicate (image_has_verified_bbox_for_label), same "skipped" outcome
        vocabulary, so a single-image save and a bulk action never disagree about
        what counts as a legitimate skip vs. a real error.

        Returns "modified", "skipped" (verified-bbox conflict), or "not_found".
        """
        image_uuid = await self._resolve_image_uuid(image_path)
        if image_uuid is None:
            return "not_found"
        if status == ImageItemStatus.ABSENT:
            if await self._db_client.image_has_verified_bbox_for_label(image_uuid, label_uuid):
                return "skipped"
        if status == ImageItemStatus.UNCERTAIN:
            await self._db_client.delete_image_labels(image_uuid, [label_uuid])
        else:
            await self._db_client.set_image_label_status(image_uuid, label_uuid, status)
        return "modified"

    async def get_image_labels(self, image_path: str) -> Optional[List[ImageLabelRead]]:
        """Return all label statuses for an image."""
        image_uuid = await self._resolve_image_uuid(image_path)
        if image_uuid is None:
            return None
        rows = await self._db_client.get_image_labels(image_uuid)
        return [ImageLabelRead.model_validate(r) for r in rows]

    async def get_image_labels_bulk(self, image_uuids: List[str]) -> Dict[str, List[ImageLabelRead]]:
        """Return {image_uuid: [label rows]} for a batch of image UUIDs.

        Images with no rows are omitted; unknown UUIDs are silently ignored.
        Lets the review grid resolve whole-image label chips for a page of
        cards in one request instead of one GET per card.
        """
        rows = await self._db_client.get_image_labels_bulk(image_uuids)
        out: Dict[str, List[ImageLabelRead]] = {}
        for r in rows:
            out.setdefault(r.image_uuid, []).append(ImageLabelRead.model_validate(r))
        return out

    async def list_images_for_label(
        self,
        label_uuid: str,
        statuses: List[ImageItemStatus],
    ) -> List["ImageLabelWithFilename"]:
        """Return image-label rows (with filename) for the given label + status filter."""
        return await self._db_client.list_images_for_label(label_uuid, statuses)

    async def set_image_tag(self, image_path: str, tag_uuid: str, action: str) -> bool:
        """Add or remove a whole-image tag. action is 'add' or 'remove'."""
        image_uuid = await self._resolve_image_uuid(image_path)
        if image_uuid is None:
            return False
        if action == "remove":
            await self._db_client.remove_image_tag(image_uuid, tag_uuid)
        else:
            await self._db_client.add_image_tag(image_uuid, tag_uuid)
        return True

    async def set_image_tag_status(self, image_path: str, tag_uuid: str, status: ImageItemStatus) -> bool:
        """Explicitly set PRESENT, ABSENT, or UNCERTAIN for a whole-image tag.

        Unlike labels, all three states are user-settable here — tags have no
        derived-PRESENT concept to protect.
        """
        image_uuid = await self._resolve_image_uuid(image_path)
        if image_uuid is None:
            return False
        await self._db_client.set_image_tag_status(image_uuid, tag_uuid, status)
        return True

    async def get_image_tag_statuses(self, image_path: str) -> Optional[List[ImageTagStatusRead]]:
        """Return all tag statuses (PRESENT and ABSENT) for an image."""
        image_uuid = await self._resolve_image_uuid(image_path)
        if image_uuid is None:
            return None
        rows = await self._db_client.get_image_tag_statuses(image_uuid)
        return [ImageTagStatusRead.model_validate(r) for r in rows]

    async def get_image_tags(self, image_path: str) -> Optional[List[TagRead]]:
        """Return all tags linked to an image."""
        image_uuid = await self._resolve_image_uuid(image_path)
        if image_uuid is None:
            return None
        image = await self._db_client.get_image_by_uuid(image_uuid)
        if image is None:
            return None
        return [TagRead.model_validate(t) for t in image.tags]

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
        inner: asyncio.Task[int] = asyncio.ensure_future(self._db_client.count_images_search(params))
        try:
            result = await asyncio.wait_for(asyncio.shield(inner), timeout=self._COUNT_QUERY_TIMEOUT)
            logger.info(
                "[review] count query completed in %.3fs → %d",
                time.monotonic() - t0,
                result,
            )
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
            "[review] request: terms=%s require_bboxes=%s after_uuid=%s",
            params.terms,
            params.require_bboxes,
            params.after_uuid,
        )
        fetch_params = params.model_copy(update={"limit": params.limit + 1})

        if params.after_uuid is None:
            cache_key = hashlib.md5(
                params.model_dump_json(
                    exclude={
                        "sort_by",
                        "sort_order",
                        "limit",
                        "after_uuid",
                        "after_sort_value",
                    }
                ).encode(),
                usedforsecurity=False,
            ).hexdigest()
            now = time.monotonic()
            cached = self._count_cache.get(cache_key)

            if cached is not None and (now - cached[1]) < self._COUNT_CACHE_TTL:
                logger.info(
                    "[review] count cache hit (key=%s, age=%.1fs)",
                    cache_key[:8],
                    now - cached[1],
                )
                t1 = time.monotonic()
                batch = await self._db_client.list_images_search(fetch_params)
                logger.info(
                    "[review] list query completed in %.3fs → %d rows",
                    time.monotonic() - t1,
                    len(batch),
                )
                total_count = cached[0]
            else:
                logger.info(
                    "[review] count cache miss (key=%s) — running list+count in parallel",
                    cache_key[:8],
                )
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
                    self._evict_expired_counts(now)
        else:
            t1 = time.monotonic()
            batch = await self._db_client.list_images_search(fetch_params)
            logger.info(
                "[review] list query (paginated) completed in %.3fs → %d rows",
                time.monotonic() - t1,
                len(batch),
            )
            total_count = -1

        has_more = len(batch) > params.limit
        images_page = batch[: params.limit]

        results = []
        for img in images_page:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)

        logger.info(
            "[review] total handler time %.3fs → %d images, has_more=%s",
            time.monotonic() - t0,
            len(results),
            has_more,
        )
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

        def _scan() -> tuple[List[FileEntryRead], bool]:
            # Recursive glob walks the whole tree synchronously — run in a
            # worker thread so a large FILES_ROOT doesn't freeze the event
            # loop (and every other request) for the duration of the walk.
            # Collect and sort all matching paths by name (string sort — no stat calls yet).
            raw = sorted(
                safe_path.glob(search_pattern) if recursive else safe_path.iterdir(),
                key=lambda p: p.name,
            )

            filtered = []
            for p in raw:
                if p.name.startswith("."):
                    continue
                rel_str = str(p.relative_to(self._files_root))
                is_matched = any(fnmatch.fnmatch(rel_str, pat) for pat in patterns)
                is_excluded = any(fnmatch.fnmatch(rel_str, pat) for pat in exclude_patterns)
                if is_matched and not is_excluded:
                    filtered.append(p)

            more = len(filtered) > offset + limit
            page = filtered[offset : offset + limit]

            # Stat calls (is_dir) and FileEntryRead construction only for the page slice.
            return [
                FileEntryRead(
                    name=p.name,
                    is_dir=p.is_dir(),
                    path=str(p.relative_to(self._files_root)),
                )
                for p in page
            ], more

        entries, has_more = await asyncio.to_thread(_scan)

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

    # Extensions accepted on upload. An allowlist, not a denylist: the file is
    # written under FILES_ROOT where the browser and every scan will pick it up,
    # so "anything not obviously bad" is the wrong default.
    UPLOAD_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

    async def save_uploaded_image(
        self,
        rel_dir: str,
        filename: str,
        data: bytes,
        *,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Write an uploaded image under FILES_ROOT and register it.

        Registering here rather than leaving it to lazy registration is what
        makes an upload immediately findable in Review: a record only appears on
        its own once something writes annotations to the image, so an untouched
        upload would otherwise be invisible to every search that matters.

        Near-duplicates are reported, never rejected. Re-importing an image you
        already have is a real thing to want to do (a better scan of the same
        frame, say), so the caller is told and decides.

        Args:
            rel_dir: Destination directory, relative to FILES_ROOT.
            filename: Client-supplied name; only its basename is honoured.
            data: Raw file bytes.
            overwrite: Replace an existing file instead of uniquifying the name.

        Returns:
            {path, uuid, duplicate_of, duplicate_distance, renamed_from}

        Raises:
            ValueError: unusable destination, rejected extension, or undecodable image.
        """
        if not self._files_root or not self._files_root.is_dir():
            raise ValueError("FILES_ROOT is not configured")

        # Path components in the *filename* are the classic traversal vector, and
        # get_safe_path below would reject them anyway — but silently dropping
        # them here means "../../etc/passwd.jpg" uploads as "passwd.jpg" rather
        # than failing confusingly.
        base_name = Path(filename).name
        if not base_name or base_name in {".", ".."}:
            raise ValueError(f"Unusable filename: {filename!r}")

        suffix = Path(base_name).suffix.lower()
        if suffix not in self.UPLOAD_EXTENSIONS:
            allowed = ", ".join(sorted(self.UPLOAD_EXTENSIONS))
            raise ValueError(f"{base_name!r} is not an image this app reads (accepts: {allowed})")

        dest_dir = get_safe_path(self._files_root, rel_dir or "")
        if dest_dir is None:
            raise ValueError(f"Path is outside FILES_ROOT: {rel_dir!r}")
        if dest_dir.exists() and not dest_dir.is_dir():
            raise ValueError(f"Upload destination is a file, not a folder: {rel_dir!r}")
        dest_dir.mkdir(parents=True, exist_ok=True)

        target = dest_dir / base_name
        renamed_from = None
        if target.exists() and not overwrite:
            # Uniquify rather than clobber: two cameras happily produce the same
            # "snapshot.jpg", and silently replacing the first one loses both the
            # file and every annotation already made against that path.
            stem, ext = Path(base_name).stem, Path(base_name).suffix
            for n in range(1, 1000):
                candidate = dest_dir / f"{stem}_{n}{ext}"
                if not candidate.exists():
                    renamed_from, target = base_name, candidate
                    break
            else:
                raise ValueError(f"Too many files named like {base_name!r} in {rel_dir!r}")

        # Decode before writing: an unreadable file under FILES_ROOT is worse
        # than a rejected upload, because every later scan trips over it.
        frame = await asyncio.to_thread(_decode_image_bytes, data)
        if frame is None:
            raise ValueError(f"{base_name!r} could not be decoded as an image")

        await asyncio.to_thread(target.write_bytes, data)

        try:
            image = await self._db_client.add_image(target)
        except Exception:
            # Do not leave a file the DB does not know about — it would show in
            # the browser and be missing from every search.
            await asyncio.to_thread(target.unlink, True)
            raise

        duplicate_of, duplicate_distance = await self._find_upload_duplicate(image.uuid, target)

        return {
            "path": self._image_rel_path(str(target)),
            "uuid": image.uuid,
            "duplicate_of": duplicate_of,
            "duplicate_distance": duplicate_distance,
            "renamed_from": renamed_from,
        }

    async def _find_upload_duplicate(self, image_uuid: str, path: Path) -> Tuple[Optional[str], Optional[int]]:
        """Nearest already-known image within the configured dedup threshold.

        Best-effort: a hash index that is cold, rebuilding, or sized differently
        must not fail the upload — the file is already written and registered,
        and "we could not check" is not the same as "it is a duplicate".
        """
        try:
            threshold_raw = await self._db_client.get_setting("dedup_image_threshold")
            threshold = int(threshold_raw) if threshold_raw is not None else 5
            phash = await asyncio.to_thread(compute_image_phash, path, self._hash_index._image_hash_size)
            self._hash_index.add_image(image_uuid, phash)
            for other_uuid, distance in self._hash_index.find_similar_images(phash, threshold):
                if other_uuid != image_uuid:
                    return other_uuid, distance
        except Exception:
            logger.exception("Duplicate check failed for %s; treating as not-a-duplicate", path)
        return None, None

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
            return {
                "action": "skipped",
                "reason": "path outside files_root",
                "rel_path": rel_path,
            }

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
            return {
                "action": "skipped",
                "reason": "sidecar has no uuid",
                "rel_path": rel_path,
            }

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
                logger.info(
                    "reconcile_image: move detected %s → %s",
                    existing.filename,
                    image_path,
                )
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
            return {
                "action": "error",
                "reason": f"unknown sync_mode: {sync_mode}",
                "rel_path": rel_path,
            }

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
            image_record = await self._db_client.get_image_by_uuid(uuid)
            if image_record:
                self._remove_from_hash_index(image_record)
            await self._db_client.delete_image(uuid)
            deleted += 1
        return deleted

    async def delete_images_by_directory(self, directories: List[str]) -> int:
        """Delete all DB image records (and their bboxes) whose path falls under any given directory.

        Each entry in *directories* is a path relative to FILES_ROOT.  The DB stores
        filenames as absolute paths, so each input directory is resolved to an absolute
        path before prefix-matching.  Hash-index entries are removed before deletion
        so the BK-tree stays consistent.

        Returns the number of image records deleted.
        """
        if not directories:
            return 0
        # Resolve to absolute paths — the DB stores filenames as absolute paths
        # (e.g. /app/image_datasets/timelapse_out2/frame001.jpg).
        norm_dirs = [str(p).rstrip("/") for d in directories if (p := get_safe_path(self._files_root, d))]
        if not norm_dirs:
            return 0

        # Two targeted queries instead of a full-table scan + N per-record fetches.
        image_rows, bbox_rows = await self._db_client.get_image_hash_data_by_prefixes(norm_dirs)
        if not image_rows:
            return 0

        # Update the in-memory hash index before bulk-deleting.
        for img_uuid, img_phash in image_rows:
            if img_phash:
                self._hash_index.remove_image(img_uuid)
        for bbox_uuid, bbox_phash, bbox_label_uuid in bbox_rows:
            if bbox_phash and bbox_label_uuid:
                self._hash_index.remove_bbox(bbox_uuid, bbox_label_uuid)

        image_uuids = [r[0] for r in image_rows]
        deleted = await self._db_client.delete_images_bulk(image_uuids)
        logger.info("delete_images_by_directory: deleted %d images under %s", deleted, norm_dirs)
        return deleted

    def _resolve_image_path(self, filename: str) -> Optional[Path]:
        """Resolve a filename (possibly absolute or relative) to a Path.

        Returns an absolute Path, or None if ``get_safe_path`` rejects it.
        """
        _p = Path(filename)
        return _p if _p.is_absolute() else get_safe_path(self._files_root, filename)

    def _remove_from_hash_index(self, image_record) -> None:
        """Remove an image and all its bboxes from the in-memory hash index."""
        if image_record.phash:
            self._hash_index.remove_image(image_record.uuid)
        for bbox in image_record.bboxes or []:
            if bbox.phash and bbox.label_uuid:
                self._hash_index.remove_bbox(bbox.uuid, bbox.label_uuid)

    async def delete_image_permanently(self, image_uuid: str) -> Optional[str]:
        """Delete a single image from the DB and from disk (file + JSON sidecar).

        Removes the image and its bboxes from the hash index before DB deletion
        so the BK-tree stays consistent.

        Returns the FILES_ROOT-relative path of the deleted image, or None if
        the image UUID was not found in the DB.
        """
        image_record = await self._db_client.get_image_by_uuid(image_uuid)
        if not image_record:
            return None

        filename = image_record.filename
        image_path = self._resolve_image_path(filename)

        # Delete the JSON sidecar FIRST and let any failure propagate BEFORE the
        # DB is touched. If the DB record were deleted first and the sidecar
        # deletion then failed (swallowed), the orphaned sidecar would cause the
        # next reconcile to silently re-import (resurrect) the deleted image.
        if image_path:
            sidecar_path = self._get_json_sidefile(image_path)
            if sidecar_path.exists():
                sidecar_path.unlink()  # let exception propagate; caller handles it

        # Remove from hash index before DB deletion.
        self._remove_from_hash_index(image_record)

        # Delete DB record (bboxes cascade).
        await self._db_client.delete_image(image_uuid)

        # Delete the image file last. A leftover image file without its sidecar
        # is harmless (reconcile only re-imports images that have a sidecar), so
        # this is best-effort and does not gate the delete.
        if image_path and image_path.exists():
            try:
                image_path.unlink()
            except Exception as e:
                logger.warning(
                    "delete_image_permanently: failed to delete file %s: %s",
                    image_path,
                    e,
                )

        rel_path = self._image_rel_path(filename)
        logger.info("delete_image_permanently: deleted %s (%s)", filename, image_uuid)
        return rel_path

    async def delete_image_db_only(self, image_uuid: str) -> Optional[str]:
        """Remove a single image from the DB (and hash index) without deleting the image file.

        The JSON sidecar is deleted so that a subsequent reconcile does not
        re-import the old record back into the DB.  The image file itself is
        left on disk.

        Returns the FILES_ROOT-relative path of the removed image, or None if
        the image UUID was not found in the DB.
        """
        image_record = await self._db_client.get_image_by_uuid(image_uuid)
        if not image_record:
            return None

        filename = image_record.filename

        # Delete the JSON sidecar FIRST so that if it cannot be removed we
        # abort before touching the DB.  A swallowed failure here would leave
        # the sidecar on disk after the DB record is gone, causing the next
        # reconcile run to silently re-import (resurrect) the deleted record.
        image_path = self._resolve_image_path(filename)
        if image_path:
            sidecar_path = self._get_json_sidefile(image_path)
            if sidecar_path.exists():
                sidecar_path.unlink()  # let exception propagate; caller handles it

        # Remove from hash index before DB deletion.
        self._remove_from_hash_index(image_record)

        # Delete DB record (bboxes cascade); image file is untouched.
        await self._db_client.delete_image(image_uuid)

        rel_path = self._image_rel_path(filename)
        logger.info("delete_image_db_only: removed from DB %s (%s)", filename, image_uuid)
        return rel_path

    # ─────────────────────────────────────────────────────────────────────────
    # Perceptual hash index public API
    # ─────────────────────────────────────────────────────────────────────────

    def find_similar_images(self, phash: str, threshold: int) -> List[Tuple[str, int]]:
        """Return [(image_uuid, distance)] for images within *threshold* of *phash*."""
        return self._hash_index.find_similar_images(phash, threshold)

    def find_similar_bboxes(
        self,
        phash: str,
        label_uuids: Optional[List[str]],
        threshold: int,
    ) -> List[Tuple[str, int, str]]:
        """Return [(bbox_uuid, distance, label_uuid)] within *threshold* of *phash*."""
        return self._hash_index.find_similar_bboxes(phash, label_uuids, threshold)

    async def find_similar_bboxes_enriched(
        self,
        phash: str,
        label_uuids: Optional[List[str]],
        threshold: int,
        tag_uuids: Optional[List[str]] = None,
        exclude_tag_uuids: Optional[List[str]] = None,
        tag_match_mode: Literal["any", "all"] = "any",
    ) -> List[Tuple[str, Optional[str], int, str]]:
        """Return [(bbox_uuid, image_uuid, distance, label_uuid)] within *threshold* of *phash*.

        Like find_similar_bboxes but enriches each result with its parent image UUID via a
        single batch query, so callers never have to deal with a missing image_uuid.
        Optional tag_uuids / exclude_tag_uuids filter the matching bboxes by their own tags.
        label_uuids restricts the BK-tree search to those label trees (None/empty = all labels).
        """
        matches = self._hash_index.find_similar_bboxes(phash, label_uuids, threshold)
        if not matches:
            return []

        # Apply tag filter if any tags are specified.
        if tag_uuids or exclude_tag_uuids:
            passing = await self._db_client.filter_bboxes_by_tags(
                [b for b, _, _ in matches],
                tag_uuids or [],
                exclude_tag_uuids or [],
                tag_match_mode,
            )
            matches = [(b, d, l) for b, d, l in matches if b in passing]
            if not matches:
                return []

        image_uuid_map = await self._db_client.get_image_uuids_for_bboxes([b for b, _, _ in matches])
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
        label_uuids: Optional[List[str]],
        threshold: int,
        tag_uuids: Optional[List[str]] = None,
        exclude_tag_uuids: Optional[List[str]] = None,
        tag_match_mode: Literal["any", "all"] = "any",
    ) -> List[Tuple[str, Optional[str], int, str]]:
        """Return [(bbox_uuid, image_uuid, distance, label_uuid)] for bboxes similar to the given bbox UUID, including the query bbox itself (distance 0)."""
        result = await self._db_client.get_bbox_with_image(bbox_uuid)
        if result is None:
            return []
        bbox, _ = result
        if bbox.phash is None:
            return []
        return await self.find_similar_bboxes_enriched(
            bbox.phash,
            label_uuids,
            threshold,
            tag_uuids,
            exclude_tag_uuids,
            tag_match_mode,
        )

    async def rank_images_by_diversity(
        self,
        seed_uuid: Optional[str],
        candidate_uuids: Optional[Set[str]],
        count: int,
    ) -> List[str]:
        """Return up to *count* image UUIDs ordered most-diverse first (farthest-point sampling).

        If *seed_uuid* is None, a seed phash is chosen automatically from the candidate
        set (v1 approximation: first available candidate — the diversity walk is largely
        insensitive to the exact seed choice).
        If *candidate_uuids* is None, all valid images participate.
        """
        if seed_uuid is not None:
            image = await self._db_client.get_image_by_uuid(seed_uuid)
            if image is None or image.phash is None:
                return []
            seed_phash = image.phash
        else:
            seed_phash = self._hash_index.pick_any_image_phash(candidate_uuids)
            if seed_phash is None:
                return []

        # rank_images_by_diversity is async: it materialises the candidate list on
        # the event loop (safe against concurrent index mutation) and offloads only
        # the CPU-bound ranking to a worker thread internally.
        return await self._hash_index.rank_images_by_diversity(
            seed_phash,
            candidate_uuids,
            count,
        )

    async def rank_bboxes_by_diversity(
        self,
        seed_bbox_uuid: str,
        label_uuids: Optional[List[str]],
        count: int,
    ) -> List[Tuple[str, Optional[str], str]]:
        """Return up to *count* (bbox_uuid, image_uuid, label_uuid) ordered most-diverse first."""
        result = await self._db_client.get_bbox_with_image(seed_bbox_uuid)
        if result is None:
            return []
        bbox, _ = result
        if bbox.phash is None:
            return []

        ranked = await self._hash_index.rank_bboxes_by_diversity(
            bbox.phash,
            label_uuids or None,
            count,
        )
        if not ranked:
            return []

        image_uuid_map = await self._db_client.get_image_uuids_for_bboxes([b for b, _ in ranked])
        return [(b_uuid, image_uuid_map.get(b_uuid), lbl) for b_uuid, lbl in ranked]

    async def get_most_unique_images(
        self,
        count: int,
        seed_uuid: Optional[str] = None,
        candidate_uuids: Optional[List[str]] = None,
    ) -> List[str]:
        """Return up to *count* image UUIDs ranked by visual diversity.

        Default scope: images that have at least one bbox tagged UNVERIFIED_TAG_UUID.
        Falls back to all images if no unverified images exist.

        When *candidate_uuids* is given (the caller's already-filtered set, e.g. the
        Review tab's active filters), that set is the scope instead — an explicit
        filter wins over the unverified default, otherwise filtering to, say,
        verified-only images would silently return unverified ones. An empty list
        means "nothing matched the filter" and yields no results.
        """
        if candidate_uuids is not None:
            if not candidate_uuids:
                return []
            return await self.rank_images_by_diversity(seed_uuid, self._cap_candidates(set(candidate_uuids)), count)

        unverified_uuids = self._cap_candidates(await self._db_client.get_image_uuids_by_bbox_tag(UNVERIFIED_TAG_UUID))
        return await self.rank_images_by_diversity(
            seed_uuid,
            unverified_uuids if unverified_uuids else None,
            count,
        )

    @staticmethod
    def _cap_candidates(uuids: Set[str]) -> Set[str]:
        """Bound a diversity candidate pool to the hash index's documented capacity.

        Truncating the query itself would bias which images are even eligible (e.g.
        always favoring lower-rowid images), so sample randomly instead to keep the
        pool representative.
        """
        if len(uuids) <= HashIndex.MAX_DIVERSITY_CANDIDATES:
            return uuids
        return set(random.sample(sorted(uuids), HashIndex.MAX_DIVERSITY_CANDIDATES))

    async def find_similar_bboxes_by_coords(
        self,
        image_uuid: str,
        x: float,
        y: float,
        w: float,
        h: float,
        label_uuids: Optional[List[str]],
        threshold: int,
        tag_uuids: Optional[List[str]] = None,
        exclude_tag_uuids: Optional[List[str]] = None,
        tag_match_mode: Literal["any", "all"] = "any",
    ) -> List[Tuple[str, Optional[str], int, str]]:
        """Compute a phash from a (possibly unsaved) bbox region and return similar bboxes."""
        image = await self._db_client.get_image_by_uuid(image_uuid)
        if image is None:
            return []
        image_path = Path(image.filename)
        if not image_path.is_absolute():
            image_path = self._files_root / image_path
        phash = compute_bbox_phash(image_path, x, y, w, h)
        return await self.find_similar_bboxes_enriched(
            phash, label_uuids, threshold, tag_uuids, exclude_tag_uuids, tag_match_mode
        )

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

        if self._init_error:
            notices.append(
                {
                    "key": "db_init_error",
                    "level": "error",
                    "message": f"Database initialization failed: {self._init_error}",
                }
            )
        elif self._build_phase == "migrating":
            notices.append(
                {
                    "key": "db_migrating",
                    "level": "info",
                    "message": "Database migrating — other requests paused until complete",
                }
            )
        elif not self._hash_index_ready:
            if self._build_phase == "loading":
                img, bbox = self._build_img_loaded, self._build_bbox_loaded
                if img or bbox:
                    count_str = f" — {img:,} images, {bbox:,} bboxes loaded"
                else:
                    count_str = ""
                phase_detail = f" (loading hashes from DB…{count_str})"
            else:
                phase_detail = {
                    "constructing": " (constructing search trees…)",
                }.get(self._build_phase, "…")
            notices.append(
                {
                    "key": "hash_build",
                    "level": "info",
                    "message": f"Building similarity index{phase_detail} — similarity searches unavailable until complete",
                }
            )
        elif self._rebuild_in_progress:
            notices.append(
                {
                    "key": "hash_rebuild",
                    "level": "warning",
                    "message": "Hash index rebuilding — similar search may return stale results",
                }
            )
        stats = self._hash_index.stats()

        summary = f"index: {stats.get('image_valid', 0):,} imgs"
        if stats.get("needs_rebuild"):
            summary += " ⚠rebuild"

        # Bbox trees are keyed by label *uuid*, and resolving those to names means
        # a DB round-trip — which /status can't afford, since the notices loop
        # polls it every few seconds. Aggregate instead: a list of 28 raw uuids
        # tells a reader nothing anyway.
        bbox_sizes = stats.get("bbox_tree_sizes_by_label", {}) or {}
        tombstones = stats.get("bbox_tombstones_by_label", {}) or {}
        bbox_total = sum(bbox_sizes.values())
        bbox_dead = sum(tombstones.values())

        index_rows = [
            row("Images indexed", f"{stats.get('image_valid', 0):,}"),
            row("Image tree size", f"{stats.get('image_tree_size', 0):,} ({stats.get('image_tombstones', 0):,} dead)"),
            row("Bbox trees", f"{len(bbox_sizes):,} labels, {bbox_total:,} entries ({bbox_dead:,} dead)"),
            row("Largest bbox tree", f"{max(bbox_sizes.values()):,}" if bbox_sizes else "—"),
            row("Build phase", self._build_phase or "—"),
            row("Ready", "yes" if self._hash_index_ready else "no"),
            row("Needs rebuild", "yes" if stats.get("needs_rebuild") else "no"),
        ]

        return {
            "service": "db_server",
            "status": "ok",
            "hash_index": stats,
            "summary": summary,
            "notices": notices,
            "details": [section("Similarity index", index_rows)],
        }

    async def get_image_filename_by_uuid(self, image_uuid: str) -> Optional[str]:
        """Return the relative filename for an image UUID, or None if not found."""
        image = await self._db_client.get_image_by_uuid(image_uuid)
        return self._image_rel_path(image.filename) if image else None

    def check_dedup_bulk(self, image_phash: str, image_threshold: int, bboxes: list) -> dict:
        """Bulk duplicate check: one image phash + N bbox phashes → one in-memory call."""
        return self._hash_index.check_dedup_bulk(image_phash, image_threshold, bboxes)

    async def rebuild_hash_index(self, clear_first: bool = False) -> dict:
        """Rebuild the BK-tree index from the DB and return stats.

        If clear_first is True, all stored phashes are NULLed before the rebuild
        so the tree starts empty. Use this when the global hash size changes —
        old hashes are incompatible and would produce nonsense distances.
        """
        self._rebuild_in_progress = True
        try:
            if clear_first:
                await self._db_client.clear_all_phashes()
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
        for bbox in image_record.bboxes or []:
            try:
                bp = await asyncio.to_thread(
                    phash_from_numpy_crop,
                    image_bgr,
                    bbox.x,
                    bbox.y,
                    bbox.width,
                    bbox.height,
                    bbox_hs,
                )
                bbox_phash_list.append({"uuid": bbox.uuid, "phash": bp})
            except Exception:
                logger.exception("bbox phash error for %s", bbox.uuid)

        await self._db_client.update_phashes(image_uuid, image_phash, bbox_phash_list)

        # Remove old hash index entries, then re-add updated ones.
        if image_record.phash:
            self._hash_index.remove_image(image_uuid)
        for bbox in image_record.bboxes or []:
            if bbox.phash and bbox.label_uuid:
                self._hash_index.remove_bbox(bbox.uuid, bbox.label_uuid)

        self._hash_index.add_image(image_uuid, image_phash)
        for bp_entry in bbox_phash_list:
            bbox_obj = next(
                (b for b in (image_record.bboxes or []) if b.uuid == bp_entry["uuid"]),
                None,
            )
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
        is_temporary: Optional[bool] = False,
    ) -> TaskConfig:
        return await self._db_client.add_task_config(
            name=name, typename=typename, params=params, description=description, is_temporary=bool(is_temporary)
        )

    async def list_task_configs(
        self,
        config_uuids: Optional[List[str]] = None,
        typename: Optional[str] = None,
        name: Optional[str] = None,
        is_temporary: Optional[bool] = None,
    ) -> Dict[str, TaskConfig]:
        return await self._db_client.get_task_configs(
            config_uuids=config_uuids, typename=typename, name=name, is_temporary=is_temporary
        )

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

    async def set_task_status(
        self, instance_uuid: str, status: TaskStatus, error_message: Optional[str] = None
    ) -> None:
        await self._db_client.set_task_status(instance_uuid, status, error_message=error_message)

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
        status_data_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self._db_client.set_task_timestamps_and_logs(
            instance_uuid,
            started_at=started_at,
            ended_at=ended_at,
            log_json=log_json,
            status_data_json=status_data_json,
        )

    async def update_task_heartbeat(self, instance_uuid: str) -> None:
        await self._db_client.update_task_heartbeat(instance_uuid)

    async def get_task_results(self, task_uuids: List[str]) -> Dict[str, TaskResultData]:
        results = await self._db_client.get_task_results(task_uuids)
        return results or {}

    ################################################################################
    # THUMBNAIL API
    ################################################################################

    def _evict_expired_counts(self, now: float) -> None:
        """Drop timed-out entries from the count cache once it grows past the ceiling.

        Entries are keyed by a hash of the filter predicates, so every distinct filter
        combination a user tries leaves one behind; without this they are only ever
        replaced by an identical query and the dict grows for the life of the process.
        """
        if len(self._count_cache) <= self._COUNT_CACHE_MAX_ENTRIES:
            return
        stale = [key for key, (_, ts) in self._count_cache.items() if now - ts >= self._COUNT_CACHE_TTL]
        for key in stale:
            del self._count_cache[key]
        if len(self._count_cache) > self._COUNT_CACHE_MAX_ENTRIES:
            # Everything is still live (a burst of distinct filters inside one TTL
            # window). Drop the oldest to keep the ceiling meaningful.
            excess = len(self._count_cache) - self._COUNT_CACHE_MAX_ENTRIES
            for key in sorted(self._count_cache, key=lambda k: self._count_cache[k][1])[:excess]:
                del self._count_cache[key]

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
        """Trim the cache in a background thread if a size limit is set.

        Coalesced: each trim globs and stats the whole cache directory, so running one
        per thumbnail miss (which is how often this is called) is pure waste. If a trim
        is already in flight, this is a no-op — that trim will observe whatever the
        current on-disk state is by the time it runs.
        """
        if THUMB_CACHE_MAX_MB <= 0:
            return
        if self._thumb_trim_task is not None and not self._thumb_trim_task.done():
            return
        max_bytes = THUMB_CACHE_MAX_MB * 1024 * 1024
        # Keep a strong reference (asyncio holds only a weak one) and log failures
        # instead of letting them vanish into an un-awaited task.
        task = asyncio.create_task(asyncio.to_thread(self._trim_thumb_cache_sync, max_bytes))
        self._thumb_trim_task = task

        def _log_failure(t: asyncio.Task) -> None:
            if not t.cancelled() and t.exception() is not None:
                logger.warning(f"Thumbnail cache trim failed: {t.exception()}")

        task.add_done_callback(_log_failure)

    def _resolve_image_file_path(self, image_path: str) -> Optional[Path]:
        """Resolve an image path to a file under _files_root, or None.

        Only paths that resolve within _files_root are accepted; there is no
        raw-path fallback (that would let a user-supplied path escape the root
        via an absolute path or traversal — arbitrary file read). Trusted
        callers pass absolute paths already under _files_root, which
        get_safe_path resolves correctly.
        """
        safe = get_safe_path(self._files_root, Path(image_path))
        if safe is not None and safe.is_file():
            return safe
        # Not silent: legacy DB rows may hold absolute paths outside the root
        # (imported before FILES_ROOT was enforced) — surface why the
        # thumbnail is missing instead of returning a bare None.
        logger.warning(
            f"Image path {image_path!r} does not resolve to a file under {self._files_root}; refusing to serve it"
        )
        return None

    @staticmethod
    def _render_outlined_crop(
        img,
        box: Tuple[float, float, float, float],
        out_width: int,
        color: Tuple[int, int, int],
        pad_frac: float = 0.2,
    ) -> bytes:
        """Crop `img` around `box`, resize to `out_width`, then stroke the box.

        `box` is (left, top, right, bottom) in source pixels; `pad_frac` is the
        context padding as a fraction of the box's own size, so the box occupies
        the same share of the thumbnail whatever its source size.

        The stroke is drawn AFTER the resize, at a width proportional to the
        output. Drawn before (at a fixed 3 source px) it was a third of the crop
        for a 20px box — which reads as a small box floating in context rather
        than a filled card — and less than one pixel for a 900px box, which the
        LANCZOS downscale then blurred into the background.

        Returns PNG bytes.
        """
        from PIL import Image as PILImage
        from PIL import ImageDraw

        img_w, img_h = img.size
        bl, bt, br, bb = box
        pad_x = (br - bl) * pad_frac
        pad_y = (bb - bt) * pad_frac
        left = max(0, bl - pad_x)
        top = max(0, bt - pad_y)
        right = min(img_w, br + pad_x)
        bottom = min(img_h, bb + pad_y)
        # A box smaller than a pixel (or an image only a few pixels across) would
        # otherwise crop to nothing, and a zero-size crop cannot be resized.
        right = max(right, min(img_w, left + 1))
        bottom = max(bottom, min(img_h, top + 1))

        cropped = img.crop((left, top, right, bottom))
        crop_w, crop_h = cropped.size
        aspect = crop_w / crop_h if crop_h else 1
        out_height = max(1, round(out_width / aspect))
        cropped = cropped.resize((out_width, out_height), PILImage.LANCZOS)

        sx = out_width / crop_w if crop_w else 1
        sy = out_height / crop_h if crop_h else 1
        stroke = max(_OUTLINE_MIN_PX, round(out_width * _OUTLINE_WIDTH_RATIO))
        draw = ImageDraw.Draw(cropped)
        draw.rectangle(
            [
                (bl - left) * sx,
                (bt - top) * sy,
                (br - left) * sx,
                (bb - top) * sy,
            ],
            outline=color,
            width=stroke,
        )
        del draw

        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()

    async def get_bbox_thumbnail(self, bbox_uuid: str, width: int = 200) -> Optional[bytes]:
        """
        Return JPEG bytes for a cropped+resized thumbnail of the given bbox.
        Serves from disk cache if available and fresh, otherwise generates and caches it.
        Cache freshness is validated against bbox.updated_at so edits made outside the
        review page (e.g. from the image-tagging canvas) are always reflected.
        """
        from PIL import Image as PILImage

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
                    return self._render_outlined_crop(
                        img,
                        (bx * img_w, by * img_h, (bx + bw) * img_w, (by + bh) * img_h),
                        width,
                        label_color,
                    )
            except Exception as e:
                logger.warning(f"Failed to generate thumbnail for bbox {bbox_uuid}: {e}")
                return None

        async with self._thumb_decode_sem:
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

    def _region_thumb_cache_path(self, image_path: str, region: Tuple[float, float, float, float], width: int) -> Path:
        # The region is part of the key, so moving a proposed box simply misses
        # the cache rather than needing an invalidation hook.
        key = hashlib.sha1(f"{image_path}|{region[0]:.6f},{region[1]:.6f},{region[2]:.6f},{region[3]:.6f}".encode())
        return self._thumb_cache_dir() / f"rgn_{key.hexdigest()}_{width}.png"

    async def get_region_thumbnail(
        self,
        image_path: str,
        x: float,
        y: float,
        width_frac: float,
        height_frac: float,
        width: int = 200,
        color: Optional[str] = None,
    ) -> Optional[bytes]:
        """PNG bytes for a cropped+resized thumbnail of an arbitrary normalized
        region of an image, with the region outlined.

        This is `get_bbox_thumbnail` for a box that has no bbox row: a preview
        run's proposed *new* boxes exist only as candidate geometry, so a card
        for one had to fall back to the whole frame — in which a small proposed
        box is invisible. Same 20% context padding as the bbox crop.
        """
        from PIL import Image as PILImage

        bx = min(max(x, 0.0), 1.0)
        by = min(max(y, 0.0), 1.0)
        bw = min(max(width_frac, 0.0), 1.0 - bx)
        bh = min(max(height_frac, 0.0), 1.0 - by)
        if bw <= 0 or bh <= 0:
            return None

        cache_path = self._region_thumb_cache_path(image_path, (bx, by, bw, bh), width)
        if cache_path.exists():
            try:
                cache_path.touch()  # update mtime for LRU tracking
                async with aiofiles.open(cache_path, "rb") as f:
                    return await f.read()
            except OSError:
                pass

        safe_path = self._resolve_image_file_path(image_path)
        if safe_path is None:
            return None

        outline: Tuple[int, int, int] = (92, 196, 92)  # the "new box" green
        if color:
            try:
                h = color.lstrip("#")
                outline = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
            except (ValueError, IndexError):
                pass

        def _generate() -> Optional[bytes]:
            try:
                with PILImage.open(safe_path) as img:
                    img = img.convert("RGB")
                    img_w, img_h = img.size
                    return self._render_outlined_crop(
                        img,
                        (bx * img_w, by * img_h, (bx + bw) * img_w, (by + bh) * img_h),
                        width,
                        outline,
                    )
            except Exception as e:
                logger.warning(f"Failed to generate region thumbnail for {image_path}: {e}")
                return None

        async with self._thumb_decode_sem:
            png_bytes = await asyncio.to_thread(_generate)
        if png_bytes is None:
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(cache_path, "wb") as f:
            await f.write(png_bytes)

        self._schedule_thumb_cache_trim()
        return png_bytes

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

        async with self._thumb_decode_sem:
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

    async def list_datasets(self, include_views: bool = False) -> List[DatasetRead]:
        return await self._db_client.list_datasets(include_views=include_views)

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

    def get_model_type_architectures(self) -> dict:
        return dict(MODEL_TYPE_ARCHITECTURES)

    async def _validate_model_config_nodes(self, nodes: List[ModelConfigNodeWrite]) -> None:
        """Validate model_type, architecture, and label_uuid for every node write object."""
        for node in nodes:
            if node.model_type not in MODEL_TYPE_ARCHITECTURES:
                raise ValueError(
                    f"Unknown model_type: {node.model_type!r}. "
                    f"Valid choices: {list(MODEL_TYPE_ARCHITECTURES.keys())}"
                )
            if node.architecture not in MODEL_TYPE_ARCHITECTURES[node.model_type]:
                raise ValueError(
                    f"Architecture {node.architecture!r} is not valid for model_type "
                    f"{node.model_type!r}. Valid choices: {MODEL_TYPE_ARCHITECTURES[node.model_type]}"
                )
            if node.label_uuid is not None:
                label = await self._db_client.get_label_by_uuid(node.label_uuid)
                if label is None:
                    raise ValueError(f"Label not found: {node.label_uuid!r}")

    async def list_model_configs(self) -> List[ModelConfigRead]:
        return await self._db_client.list_model_configs()

    async def get_model_config(self, uuid: str) -> Optional[ModelConfigRead]:
        return await self._db_client.get_model_config(uuid)

    async def get_model_config_node(self, node_uuid: str) -> Optional["ModelConfigNodeRead"]:
        return await self._db_client.get_model_config_node(node_uuid)

    async def derive_training_targets(self, model_config_uuid: str) -> Optional[list]:
        return await self._db_client.derive_training_targets(model_config_uuid)

    async def get_training_target_detection_dataset(
        self,
        model_config_uuid: str,
        node_uuid: str,
        dataset_uuids: list[str],
        label_uuids: Optional[list[str]] = None,
        class_remap: Optional[dict[str, str]] = None,
        negative_ratio: Optional[float] = None,
        negative_ratio_floor: Optional[int] = None,
    ) -> Optional[dict]:
        result = await self._db_client.get_training_target_detection_dataset(
            model_config_uuid, node_uuid, dataset_uuids, label_uuids=label_uuids, class_remap=class_remap
        )
        if result is None:
            return None
        return await self._apply_negative_ratio_cap(result, negative_ratio, negative_ratio_floor)

    # Default cap for a detection dataset's negative (no-annotation) images, expressed
    # as a multiple of the positive count. Bulk-verifying "absent" across a whole dataset
    # (e.g. tagging every indoor photo as dog-absent right after adding a few outdoor dog
    # photos) otherwise silently drowns a new class in negatives with no signal in the loss.
    NEGATIVE_RATIO_DEFAULT = 4.0
    NEGATIVE_RATIO_FLOOR_DEFAULT = 20

    async def _apply_negative_ratio_cap(
        self,
        result: dict,
        negative_ratio: Optional[float],
        negative_ratio_floor: Optional[int],
    ) -> dict:
        """Cap a detection dataset's negatives at ``negative_ratio`` * positive count
        (never below ``negative_ratio_floor``), keeping the perceptually most diverse
        negatives via ``HashIndex`` so a handful of near-duplicate frames from one
        static source can't be what survives the cap. No-op if already under the cap.
        """
        images = result.get("images", [])
        positives = [img for img in images if img["annotations"]]
        negatives = [img for img in images if not img["annotations"]]

        ratio = negative_ratio if negative_ratio is not None else self.NEGATIVE_RATIO_DEFAULT
        floor = negative_ratio_floor if negative_ratio_floor is not None else self.NEGATIVE_RATIO_FLOOR_DEFAULT
        max_negatives = max(int(len(positives) * ratio), floor)

        if len(negatives) <= max_negatives:
            return result

        candidate_uuids = {img["image_uuid"] for img in negatives}
        if len(candidate_uuids) > HashIndex.MAX_DIVERSITY_CANDIDATES:
            candidate_uuids = set(random.sample(sorted(candidate_uuids), HashIndex.MAX_DIVERSITY_CANDIDATES))

        kept_uuids = set(await self.rank_images_by_diversity(None, candidate_uuids, max_negatives))

        if len(kept_uuids) < max_negatives:
            # Negatives with no indexed phash can't be ranked by diversity at all; top
            # them up deterministically so the ratio/floor guarantee still holds.
            remaining = sorted(uuid for uuid in candidate_uuids if uuid not in kept_uuids)
            kept_uuids.update(remaining[: max_negatives - len(kept_uuids)])

        kept_negatives = [img for img in negatives if img["image_uuid"] in kept_uuids]
        result["images"] = positives + kept_negatives
        return result

    async def create_model_config(self, data: ModelConfigCreate) -> ModelConfigRead:
        await self._validate_model_config_nodes(data.nodes)
        return await self._db_client.create_model_config(data)

    async def update_model_config(self, uuid: str, data: ModelConfigUpdate) -> Optional[ModelConfigRead]:
        if data.nodes is not None:
            await self._validate_model_config_nodes(data.nodes)
        return await self._db_client.update_model_config(uuid, data)

    async def delete_model_config(self, uuid: str) -> bool:
        return await self._db_client.delete_model_config(uuid)

    # ── Filter chains ──────────────────────────────────────────────────────────

    async def list_filter_chains(self) -> List[FilterChainRead]:
        return await self._db_client.list_filter_chains()

    async def get_filter_chain(self, uuid: str) -> Optional[FilterChainRead]:
        return await self._db_client.get_filter_chain(uuid)

    async def create_filter_chain(self, data: FilterChainCreate) -> FilterChainRead:
        return await self._db_client.create_filter_chain(data)

    async def update_filter_chain(self, uuid: str, data: FilterChainUpdate) -> Optional[FilterChainRead]:
        return await self._db_client.update_filter_chain(uuid, data)

    async def delete_filter_chain(self, uuid: str) -> bool:
        return await self._db_client.delete_filter_chain(uuid)

    # ── Watch profiles (StreamWatcher / NVR) ───────────────────────────────────

    async def list_watch_profiles(self) -> List[WatchProfileRead]:
        return await self._db_client.list_watch_profiles()

    async def get_watch_profile(self, uuid: str) -> Optional[WatchProfileRead]:
        return await self._db_client.get_watch_profile(uuid)

    async def create_watch_profile(self, data: WatchProfileCreate) -> WatchProfileRead:
        return await self._db_client.create_watch_profile(data)

    async def update_watch_profile(self, uuid: str, data: WatchProfileUpdate) -> Optional[WatchProfileRead]:
        return await self._db_client.update_watch_profile(uuid, data)

    async def delete_watch_profile(self, uuid: str) -> bool:
        return await self._db_client.delete_watch_profile(uuid)

    async def validate_watch_profile(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Report configuration problems that only show up at run time.

        Worth its own endpoint because every one of these fails *silently*: a rule
        with no zone never fires, a pre-roll longer than the buffer quietly produces a
        short clip, and a source with no models is analysed against nothing.
        """
        profile = await self._db_client.get_watch_profile(uuid)
        if profile is None:
            return None

        errors: List[str] = []
        warnings: List[str] = []

        if not profile.sources:
            errors.append("This watcher has no sources, so it will never analyse anything.")

        enabled_rules = [r for r in profile.rules if r.enabled]
        if not enabled_rules:
            warnings.append("No enabled rules — the watcher will track objects but never raise an event.")

        zone_uuids = {z.uuid for s in profile.sources for z in s.zones}
        for rule in profile.rules:
            for kind, ref, side in (
                (rule.subject_a_kind, rule.subject_a_ref, "A"),
                (rule.subject_b_kind, rule.subject_b_ref, "B"),
            ):
                if kind == "zone" and ref not in zone_uuids:
                    errors.append(f"Rule {rule.name!r}: subject {side} references a zone that no longer exists.")

        for src in profile.sources:
            if src.model_group_uuid is None and profile.default_model_group_uuid is None:
                errors.append(
                    f"Source {src.source_uuid} has no model stack and the profile sets no default, "
                    "so nothing will be detected on it."
                )
            fps = src.analysis_fps or profile.default_analysis_fps
            if fps > 10:
                warnings.append(
                    f"Source {src.source_uuid} analyses at {fps} fps. Per-frame inference over "
                    "base64 HTTP is the throughput ceiling; 2–5 fps is usually plenty for tracking."
                )

        # Pre-roll longer than the retention can never be satisfied — the buffer simply
        # does not hold that much history. Catch it at save time, not at cut time.
        retentions = {src.source_uuid: (src.record_retention_s or STREAM_BUFFER_RETENTION_S) for src in profile.sources}
        for action in profile.actions:
            if action.kind != "record_clip":
                continue
            preroll = float(action.params_json.get("preroll_s", 0) or 0)
            for source_uuid, retention in retentions.items():
                if preroll > retention:
                    warnings.append(
                        f"A record_clip action asks for {preroll:g}s of pre-roll, but source "
                        f"{source_uuid} only buffers {retention:g}s. The clip will be clamped."
                    )

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
        }

    # ── Watch events and recordings ────────────────────────────────────────────

    async def list_watch_events(self, **kwargs) -> List[WatchEventRead]:
        return await self._db_client.list_watch_events(**kwargs)

    async def list_recordings(self, **kwargs) -> List[RecordingRead]:
        return await self._db_client.list_recordings(**kwargs)

    async def get_recording(self, uuid: str) -> Optional[RecordingRead]:
        return await self._db_client.get_recording(uuid)

    async def delete_recording(self, uuid: str) -> bool:
        """Delete a recording row and the clip file it points at.

        The file is removed first-effort but a failure there must not leave an
        undeletable row: the row is the index, the file is derived from it.
        """
        recording = await self._db_client.get_recording(uuid)
        if recording is None:
            return False
        for path_str in (recording.file_path, recording.thumbnail_path):
            if not path_str:
                continue
            try:
                Path(path_str).unlink(missing_ok=True)
            except OSError as e:
                logger.warning("Could not delete recording file %s: %s", path_str, e)
        return await self._db_client.delete_recording(uuid)

    # ── Model groups ───────────────────────────────────────────────────────────

    async def list_model_groups(self) -> List[ModelGroupRead]:
        return await self._db_client.list_model_groups()

    async def get_model_group(self, uuid: str) -> Optional[ModelGroupRead]:
        return await self._db_client.get_model_group(uuid)

    async def create_model_group(self, data: ModelGroupCreate) -> ModelGroupRead:
        return await self._db_client.create_model_group(data)

    async def update_model_group(self, uuid: str, data: ModelGroupUpdate) -> Optional[ModelGroupRead]:
        return await self._db_client.update_model_group(uuid, data)

    async def delete_model_group(self, uuid: str) -> bool:
        return await self._db_client.delete_model_group(uuid)

    # ── Training configs ──────────────────────────────────────────────────────

    async def create_training_config(self, data) -> "TrainingConfigRead":
        return await self._db_client.create_training_config(data)

    async def list_training_configs(self) -> "list[TrainingConfigRead]":
        return await self._db_client.list_training_configs()

    async def get_training_config(self, uuid: str) -> "Optional[TrainingConfigRead]":
        return await self._db_client.get_training_config(uuid)

    async def update_training_config(self, uuid: str, data) -> "Optional[TrainingConfigRead]":
        return await self._db_client.update_training_config(uuid, data)

    async def delete_training_config(self, uuid: str) -> bool:
        return await self._db_client.delete_training_config(uuid)

    async def get_training_config_readiness(self, uuid: str) -> "Optional[dict]":
        return await self._db_client.get_training_config_readiness(uuid)

    async def get_training_config_detection_dataset(self, uuid: str) -> "Optional[dict]":
        result = await self._db_client.get_training_config_detection_dataset(uuid)
        if result is None:
            return None
        tc = await self._db_client.get_training_config(uuid)
        hyperparams = dict(tc.hyperparams_json or {}) if tc is not None else {}
        return await self._apply_negative_ratio_cap(
            result, hyperparams.get("negative_ratio"), hyperparams.get("negative_ratio_floor")
        )

    async def get_dataset_image_uuids(self, dataset_uuids: list[str]) -> list[str]:
        return await self._db_client.get_dataset_image_uuids(dataset_uuids)

    async def get_training_config_label_stats(
        self, uuid: str, restrict_to_image_uuids: "Optional[list[str]]" = None
    ) -> "Optional[dict]":
        return await self._db_client.get_training_config_label_stats(
            uuid, restrict_to_image_uuids=restrict_to_image_uuids
        )

    async def get_training_config_needs_review_images(
        self, uuid: str, payload: "NeedsReviewImagesPayload"
    ) -> "tuple[list[ImageRecordRead], bool]":
        images, has_more = await self._db_client.get_training_config_needs_review_images(
            uuid,
            limit=payload.limit,
            after_uuid=payload.after_uuid,
            after_sort_value=payload.after_sort_value,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
            extra_label_uuids=payload.label_uuids or None,
        )
        results = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)
        return results, has_more

    async def get_training_config_ready_images(
        self, uuid: str, payload: "ReadyImagesPayload"
    ) -> "tuple[list[ImageRecordRead], bool]":
        images, has_more = await self._db_client.get_training_config_ready_images(
            uuid,
            limit=payload.limit,
            after_uuid=payload.after_uuid,
            after_sort_value=payload.after_sort_value,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
        )
        results = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)
        return results, has_more

    async def get_training_config_ready_positive_images(
        self, uuid: str, payload: "ReadyImagesPayload"
    ) -> "tuple[list[ImageRecordRead], bool]":
        images, has_more = await self._db_client.get_training_config_ready_positive_images(
            uuid,
            limit=payload.limit,
            after_uuid=payload.after_uuid,
            after_sort_value=payload.after_sort_value,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
        )
        results = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)
        return results, has_more

    async def get_training_config_ready_negative_images(
        self, uuid: str, payload: "ReadyImagesPayload"
    ) -> "tuple[list[ImageRecordRead], bool]":
        images, has_more = await self._db_client.get_training_config_ready_negative_images(
            uuid,
            limit=payload.limit,
            after_uuid=payload.after_uuid,
            after_sort_value=payload.after_sort_value,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
        )
        results = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)
        return results, has_more

    async def get_training_config_partial_review_images(
        self, uuid: str, payload: "ReadyImagesPayload"
    ) -> "tuple[list[ImageRecordRead], bool]":
        images, has_more = await self._db_client.get_training_config_partial_review_images(
            uuid,
            limit=payload.limit,
            after_uuid=payload.after_uuid,
            after_sort_value=payload.after_sort_value,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
        )
        results = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)
        return results, has_more

    async def get_training_config_not_started_images(
        self, uuid: str, payload: "ReadyImagesPayload"
    ) -> "tuple[list[ImageRecordRead], bool]":
        images, has_more = await self._db_client.get_training_config_not_started_images(
            uuid,
            limit=payload.limit,
            after_uuid=payload.after_uuid,
            after_sort_value=payload.after_sort_value,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
        )
        results = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)
        return results, has_more

    async def get_training_config_image_gaps(self, uuid: str, image_uuids: "list[str]") -> "dict[str, list[dict]]":
        return await self._db_client.get_training_config_image_gaps(uuid, image_uuids)

    async def get_training_config_images_by_label_status(
        self, uuid: str, payload: "ImagesByLabelStatusPayload"
    ) -> "tuple[list[ImageRecordRead], bool]":
        images, has_more = await self._db_client.get_training_config_images_by_label_status(
            uuid,
            payload.label_uuid,
            payload.status,
            rollup_label_uuids=payload.rollup_label_uuids,
            missing_label_uuid=payload.missing_label_uuid,
            limit=payload.limit,
            after_uuid=payload.after_uuid,
            after_sort_value=payload.after_sort_value,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
        )
        results = []
        for img in images:
            record = ImageRecordRead.model_validate(img)
            record.filename = self._image_rel_path(img.filename)
            results.append(record)
        return results, has_more

    # ── Preview runs ───────────────────────────────────────────────────────

    async def create_preview_run(self, payload: "PreviewRunCreate") -> PreviewRunRead:
        name = payload.name or f"{payload.model_name} — {datetime.now(timezone.utc):%Y-%m-%d %H:%M}"
        run = await self._db_client.create_preview_run(
            name=name,
            model_name=payload.model_name,
            task_typename=payload.task_typename,
            params=payload.params,
            scope=payload.scope,
        )
        return PreviewRunRead.model_validate(run)

    async def list_preview_runs(self) -> List[PreviewRunRead]:
        """Sweeps expired runs first — the list is the only reliably-visited
        endpoint, so it is where the TTL gets enforced (§9 of the plan: a
        scheduler for a delete this cheap would be its own failure mode)."""
        try:
            await self._db_client.sweep_expired_preview_runs()
        except Exception:
            logger.exception("preview run TTL sweep failed; listing anyway")

        runs = await self._db_client.get_preview_runs()
        counts = await self._db_client.count_preview_decisions([r.uuid for r in runs])
        results = []
        for run in runs:
            read = PreviewRunRead.model_validate(run)
            read.decision_counts = counts.get(run.uuid, {})
            read.candidate_count = sum(read.decision_counts.values())
            results.append(read)
        return results

    async def get_preview_run(self, run_uuid: str) -> Optional[PreviewRunRead]:
        run = await self._db_client.get_preview_run(run_uuid)
        if run is None:
            return None
        read = PreviewRunRead.model_validate(run)
        read.decision_counts = (await self._db_client.count_preview_decisions([run_uuid])).get(run_uuid, {})
        read.candidate_count = sum(read.decision_counts.values())
        return read

    async def update_preview_run(self, run_uuid: str, payload: "PreviewRunUpdate") -> Optional[PreviewRunRead]:
        run = await self._db_client.update_preview_run(
            run_uuid, name=payload.name, pinned=payload.pinned, stats=payload.stats
        )
        return None if run is None else PreviewRunRead.model_validate(run)

    async def delete_preview_run(self, run_uuid: str) -> bool:
        return await self._db_client.delete_preview_runs(run_uuid) > 0

    async def add_preview_candidates(
        self,
        run_uuid: str,
        candidates: List["PreviewCandidateCreate"],
        conflict_policy: PreviewConflictPolicy = PreviewConflictPolicy.LAST,
    ) -> int:
        return await self._db_client.add_preview_candidates(run_uuid, candidates, conflict_policy)

    async def _annotate_preview_staleness(self, candidates: List[PreviewCandidate]) -> List[PreviewCandidateRead]:
        """Reconcile a run's snapshot against the live rows.

        A saved run holds UUIDs captured when the model ran. By the time the user
        comes back the bbox may have been relabeled or deleted and the image may
        be gone, so this is read-time reconciliation, not stored state:

          - bbox row gone (or image gone, for an is_new candidate) → `stale`
          - bbox still there but its label moved off the snapshot →
            `label_changed_since_run`, which is still applicable but the user
            should see it before accepting.

        `stale` is written back so the counts in the saved-runs list agree with
        what the panel shows.
        """
        bbox_uuids = {c.bbox_uuid for c in candidates if c.bbox_uuid}
        # Fetched for *every* candidate, not just the boxless ones: the panel
        # builds each card's thumbnail URL from the path, and an absent key is
        # also how a vanished image is detected.
        image_uuids = {c.image_uuid for c in candidates if c.image_uuid}

        live_labels: Dict[str, Optional[str]] = {}
        if bbox_uuids:
            live_labels = await self._db_client.get_bbox_label_uuids(list(bbox_uuids))
        live_images: Dict[str, str] = {}
        if image_uuids:
            live_images = await self._db_client.get_image_filenames(list(image_uuids))

        results: List[PreviewCandidateRead] = []
        newly_stale: List[str] = []
        for c in candidates:
            read = PreviewCandidateRead.model_validate(c)
            filename = live_images.get(c.image_uuid)
            read.image_filename = self._image_rel_path(filename) if filename else ""
            vanished = (
                c.image_uuid not in live_images if (c.is_new or not c.bbox_uuid) else c.bbox_uuid not in live_labels
            )
            if vanished:
                # Already-applied candidates are history, not stale work — an
                # applied delete legitimately removed its own bbox.
                if c.decision != PreviewDecision.APPLIED:
                    read.decision = PreviewDecision.STALE
                    if c.decision != PreviewDecision.STALE:
                        newly_stale.append(c.uuid)
            elif c.bbox_uuid and c.current_label_uuid is not None:
                read.label_changed_since_run = live_labels.get(c.bbox_uuid) != c.current_label_uuid
            results.append(read)

        if newly_stale:
            run_uuid = candidates[0].run_uuid
            await self._db_client.update_preview_candidates(run_uuid, newly_stale, decision=PreviewDecision.STALE)

        return results

    async def query_preview_candidates(
        self, run_uuid: str, query: "PreviewCandidateQuery"
    ) -> Tuple[List[PreviewCandidateRead], int]:
        candidates, total = await self._db_client.query_preview_candidates(run_uuid, query)
        return await self._annotate_preview_staleness(candidates), total

    async def update_preview_candidates(self, run_uuid: str, payload: "PreviewCandidateBulkUpdate") -> int:
        return await self._db_client.update_preview_candidates(
            run_uuid,
            payload.candidate_uuids,
            decision=payload.decision,
            decision_action=payload.decision_action,
            proposed_label_uuid=payload.proposed_label_uuid,
            geometry=payload.geometry,
        )

    async def apply_preview_run(self, run_uuid: str, mark_verified: bool = False) -> PreviewApplyResult:
        """Apply every `accepted` candidate in one server-side pass.

        Doing this server-side (rather than the browser's per-group loop) is what
        makes closing the laptop mid-apply safe on a several-thousand-candidate
        run: each candidate is marked `applied` as it lands, so a re-run picks up
        where it stopped instead of re-applying everything.

        Semantics match the client loop it replaces exactly — relabels grouped by
        proposed label with the model's geometry replacing the original, new boxes
        created UNVERIFIED (the model's suggestion is not a human verification)
        unless `mark_verified`, then bbox thumbnails invalidated.
        """
        result = PreviewApplyResult()
        accepted = await self._db_client.get_preview_candidates(run_uuid, decisions=[PreviewDecision.ACCEPTED])
        if not accepted:
            return result

        # Re-check liveness *inside* the apply, not only on read: the run may
        # have been sitting in a tab for days. Never silently skip — the stale
        # count is surfaced in the toast.
        reconciled = await self._annotate_preview_staleness(accepted)
        stale_uuids = {r.uuid for r in reconciled if r.decision == PreviewDecision.STALE}
        result.stale = len(stale_uuids)
        live = [c for c in accepted if c.uuid not in stale_uuids]

        applied_uuids: List[str] = []
        touched_bbox_uuids: List[str] = []

        # ── Deletes ────────────────────────────────────────────────────────
        to_delete = [c for c in live if c.decision_action == PreviewDecisionAction.DELETE and c.bbox_uuid]
        if to_delete:
            modified, failed = await self.bulk_apply_bboxes_by_uuids(
                [c.bbox_uuid for c in to_delete], "delete", None, []
            )
            result.deleted += modified
            result.failed += failed
            applied_uuids.extend(c.uuid for c in to_delete)
            touched_bbox_uuids.extend(c.bbox_uuid for c in to_delete)

        # ── Relabels, grouped by proposed label ────────────────────────────
        relabels = [c for c in live if c.decision_action != PreviewDecisionAction.DELETE and not c.is_new]
        groups: Dict[str, List[PreviewCandidate]] = defaultdict(list)
        for c in relabels:
            if c.bbox_uuid:
                groups[c.proposed_label_uuid].append(c)
        for label_uuid, group in groups.items():
            bbox_uuids = [c.bbox_uuid for c in group]
            geometry = {c.bbox_uuid: c.geometry_json for c in group if c.geometry_json}
            modified, failed = await self.bulk_apply_bboxes_by_uuids(
                bbox_uuids,
                "relabel",
                label_uuid,
                [],
                bbox_geometry=geometry or None,
                mark_verified=mark_verified,
            )
            result.modified += modified
            result.failed += failed
            applied_uuids.extend(c.uuid for c in group)
            touched_bbox_uuids.extend(bbox_uuids)

        # ── New boxes ──────────────────────────────────────────────────────
        new_boxes = [c for c in live if c.is_new and c.decision_action != PreviewDecisionAction.DELETE]
        for c in new_boxes:
            geo = c.geometry_json or {}
            image = await self._db_client.get_image_by_uuid(c.image_uuid)
            if image is None or not geo:
                result.failed += 1
                continue
            try:
                bbox_uuid = await self.add_single_bbox(
                    image.filename,
                    c.proposed_label_uuid,
                    geo["x"],
                    geo["y"],
                    geo["width"],
                    geo["height"],
                    [VERIFIED_TAG_UUID if mark_verified else UNVERIFIED_TAG_UUID],
                    # `confidence` is the key CollectTrainingImagesTask (the
                    # highest-volume producer) writes and the one the canvas
                    # label and review card were built around.
                    # ApplyFilterChainTask writes `source_confidence` instead —
                    # readers accept both, see CONFIDENCE_KEYS in
                    # static/js/shared/domain/image/confidence.js.
                    metadata_json={
                        "run_model_added": True,
                        "source_model": c.source_model,
                        "confidence": c.confidence,
                    },
                )
            except Exception:
                logger.exception("apply_preview_run: creating new bbox on image %s failed", c.image_uuid)
                bbox_uuid = None
            if bbox_uuid is None:
                result.failed += 1
                continue
            result.added += 1
            applied_uuids.append(c.uuid)

        if applied_uuids:
            await self._db_client.update_preview_candidates(
                run_uuid,
                applied_uuids,
                decision=PreviewDecision.APPLIED,
                applied_at=datetime.now(timezone.utc),
            )
        if touched_bbox_uuids:
            try:
                await self.invalidate_bbox_thumbnails(touched_bbox_uuids)
            except Exception:
                logger.exception("apply_preview_run: thumbnail invalidation failed")

        return result
