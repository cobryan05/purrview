"""Database maintenance task: reconcile image files, prune missing records, clear orphan tags,
and optionally recompute perceptual hashes for all images using the global hash-size settings.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

import db_client
import jstyleson
from db_client.api.images_api import ImagesApi
from db_client.api.settings_api import SettingsApi
from db_client.api.tags_api import TagsApi
from db_client.models.prune_images_payload import PruneImagesPayload
from db_client.models.recompute_phash_payload import RecomputePhashPayload
from db_client.models.reconcile_image_payload import ReconcileImagePayload
from db_client.models.rebuild_hash_index_response import RebuildHashIndexResponse

from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

_VALID_SYNC_MODES = {"merge", "replace_db_with_json", "replace_json_with_db", "keep_newer"}


@register_task()
class DatabaseMaintenanceTask(Task):
    """Reconcile the DB against image files on disk, and optionally recompute all perceptual hashes.

    Scan order guarantees both move orderings converge:
      1. Reconcile each file found on disk (move detection + sidecar sync).
         Collect sidecar UUIDs as known_disk_uuids.
      2. Prune DB records whose file is missing and whose UUID is not known.
      3. Clear orphan TASK_GENERATED tags.
      4. (Optional) Recompute perceptual hashes for every image using hash sizes
         from global Settings, then rebuild the BK-tree index.
    """

    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        self._status_msg: str = "Initializing"
        self._remove_deleted: bool = bool(params.get("remove_deleted", True))
        self._sync_jsons: bool = bool(params.get("sync_jsons", True))
        self._sync_mode: str = params.get("sync_mode", "merge")
        self._clear_orphan_tags: bool = bool(params.get("clear_orphan_tags", True))
        self._recompute_phashes: bool = bool(params.get("recompute_phashes", False))

        if self._sync_mode not in _VALID_SYNC_MODES:
            raise ValueError(f"Invalid sync_mode: {self._sync_mode!r}. Must be one of {_VALID_SYNC_MODES}")

    async def _run(self) -> dict[str, Any]:
        db_api_client = self._manager.get_db_api_client()
        images_api = ImagesApi(db_api_client)
        tags_api = TagsApi(db_api_client)

        # ── 1. List files on disk ─────────────────────────────────────────────
        self._status_msg = "Listing files"
        self.log("Listing image files on disk")
        list_resp = await asyncio.to_thread(images_api.list_images, path="/", pattern="*", recursive=True)
        file_entries = list_resp.files or []
        image_files = [e for e in file_entries if not e.is_dir]
        total = len(image_files)
        self.log(f"Found {total} image files")

        # Progress budget: share 0–55 % across reconcile/prune/tags; leave 55–100 for phashes.
        reconcile_end = 55.0 if self._recompute_phashes else 70.0
        prune_end = 75.0 if self._recompute_phashes else 90.0
        tags_end = 80.0 if self._recompute_phashes else 100.0

        # ── 2. Reconcile each file + collect known disk UUIDs ─────────────────
        reconciled = 0
        moved = 0
        errors = 0
        known_disk_uuids: list[str] = []

        label = "Reconciling" if self._sync_jsons else "Scanning"
        self._status_msg = f"{label} 0/{total}"

        for i, entry in enumerate(image_files):
            if self._cancel_flag.is_set():
                self.log("Cancelled during file reconcile")
                break
            await self._check_pause()

            rel_path = entry.path
            sidecar_uuid = await self._get_sidecar_uuid(rel_path)
            if sidecar_uuid:
                known_disk_uuids.append(sidecar_uuid)

            if self._sync_jsons:
                try:
                    result = await asyncio.to_thread(
                        images_api.reconcile_image,
                        ReconcileImagePayload(rel_path=rel_path, sync_mode=self._sync_mode),
                    )
                    action = getattr(result, "action", "")
                    if action == "moved":
                        moved += 1
                    elif action not in ("skipped",):
                        reconciled += 1
                except Exception:
                    logger.exception("reconcile_image failed for %s", rel_path)
                    errors += 1

            self._update_progress((i + 1) / max(total, 1) * reconcile_end)
            self._status_msg = f"{label} {i + 1}/{total}"

        # ── 3. Prune missing images ───────────────────────────────────────────
        pruned = 0
        if self._remove_deleted and not self._cancel_flag.is_set():
            self._status_msg = "Pruning missing images"
            self.log("Pruning DB records for missing images")
            try:
                prune_resp = await asyncio.to_thread(
                    images_api.prune_missing_images,
                    PruneImagesPayload(known_disk_uuids=known_disk_uuids),
                )
                pruned = getattr(prune_resp, "deleted_count", 0) or 0
                self.log(f"Pruned {pruned} missing image records")
            except Exception:
                logger.exception("prune_missing_images failed")
            self._update_progress(prune_end)

        # ── 4. Clear orphan generated tags ────────────────────────────────────
        tags_cleared = 0
        if self._clear_orphan_tags and not self._cancel_flag.is_set():
            self._status_msg = "Clearing orphan tags"
            self.log("Clearing orphan TASK_GENERATED tags")
            try:
                tags_resp = await asyncio.to_thread(tags_api.clear_orphan_generated_tags)
                tags_cleared = getattr(tags_resp, "deleted_count", 0) or 0
                self.log(f"Cleared {tags_cleared} orphan tags")
            except Exception:
                logger.exception("clear_orphan_generated_tags failed")
            self._update_progress(tags_end)

        # ── 5. Recompute perceptual hashes ────────────────────────────────────
        phash_processed = 0
        phash_errors = 0
        if self._recompute_phashes and not self._cancel_flag.is_set():
            settings_api = SettingsApi(db_api_client)
            image_hash_size, bbox_hash_size = await asyncio.gather(
                self._read_int_setting(settings_api, "dedup_image_hash_size", 8),
                self._read_int_setting(settings_api, "dedup_bbox_hash_size", 16),
            )
            self.log(f"Recomputing phashes (image_hash_size={image_hash_size}, bbox_hash_size={bbox_hash_size})")

            # Re-list stubs after reconcile/prune so the set is current.
            self._status_msg = "Listing images for phash recompute"
            stubs_resp = await asyncio.to_thread(images_api.list_image_stubs)
            stubs = stubs_resp.stubs or []
            phash_total = len(stubs)
            self.log(f"Recomputing hashes for {phash_total} images")

            for i, stub in enumerate(stubs):
                if self._cancel_flag.is_set():
                    self.log("Cancelled during phash recompute")
                    break
                await self._check_pause()

                try:
                    await asyncio.to_thread(
                        images_api.recompute_image_phash,
                        stub.uuid,
                        RecomputePhashPayload(
                            image_hash_size=image_hash_size,
                            bbox_hash_size=bbox_hash_size,
                        ),
                    )
                    phash_processed += 1
                except Exception:
                    logger.exception("recompute_image_phash failed for %s", stub.uuid)
                    self.log(f"Error hashing {stub.uuid}")
                    phash_errors += 1

                self._update_progress(tags_end + (i + 1) / max(phash_total, 1) * (95.0 - tags_end))
                self._status_msg = f"Hashing {i + 1}/{phash_total} — {phash_processed} done, {phash_errors} errors"
                await asyncio.sleep(0)

            if not self._cancel_flag.is_set():
                self._status_msg = "Rebuilding hash index"
                self.log("Rebuilding BK-tree hash index")
                try:
                    rebuild_resp: RebuildHashIndexResponse = await asyncio.to_thread(
                        images_api.rebuild_hash_index
                    )
                    img_cnt = getattr(rebuild_resp, "image_count", 0) or 0
                    bbox_cnt = getattr(rebuild_resp, "bbox_count", 0) or 0
                    self.log(f"Index rebuilt: {img_cnt} images, {bbox_cnt} bboxes")
                except Exception:
                    logger.exception("rebuild_hash_index failed")
                    self.log("Failed to rebuild hash index")

        self._update_progress(100.0)
        self._status_msg = "Done"
        summary: dict[str, Any] = {
            "files_found": total,
            "reconciled": reconciled,
            "moved": moved,
            "errors": errors,
            "pruned": pruned,
            "tags_cleared": tags_cleared,
        }
        log_parts = [
            f"{total} files",
            f"{reconciled} synced",
            f"{moved} moves detected",
            f"{pruned} pruned",
            f"{tags_cleared} orphan tags cleared",
        ]
        if self._recompute_phashes:
            summary["phash_processed"] = phash_processed
            summary["phash_errors"] = phash_errors
            log_parts.append(f"{phash_processed} hashes recomputed")
        self.log("Complete: " + ", ".join(log_parts))
        return summary

    async def _get_sidecar_uuid(self, rel_path: str) -> Optional[str]:
        """Read the sidecar for rel_path and return its uuid field, or None."""
        try:
            files_root = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))
            sidecar_path = (files_root / rel_path).with_suffix(".json")
            if not sidecar_path.exists():
                return None
            data = jstyleson.loads(sidecar_path.read_text())
            return data.get("uuid")
        except Exception:
            return None

    async def _deinit(self) -> None:
        pass

    def get_status_message(self) -> str:
        return self._status_msg

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        return {
            "meta": {
                "order": [
                    "sync_jsons",
                    "sync_mode",
                    "remove_deleted",
                    "clear_orphan_tags",
                    "recompute_phashes",
                ]
            },
            "sync_jsons": {
                "type": "boolean",
                "label": "Sync image JSON sidecars",
                "description": "Reconcile each image file against its JSON sidecar. Detects moved files.",
                "default": True,
            },
            "sync_mode": {
                "type": "select",
                "label": "Sync mode",
                "description": "Direction of data flow when reconciling image sidecars with the DB.",
                "options": [
                    {"value": "merge", "label": "Merge (add JSON-only bboxes to DB, keep existing)"},
                    {"value": "replace_db_with_json", "label": "Replace DB with JSON (JSON wins)"},
                    {"value": "replace_json_with_db", "label": "Replace JSON with DB (DB wins)"},
                    {"value": "keep_newer", "label": "Keep newer (compare timestamps)"},
                ],
                "default": "merge",
            },
            "remove_deleted": {
                "type": "boolean",
                "label": "Remove deleted images",
                "description": "Delete DB records for images that are no longer on disk.",
                "default": True,
            },
            "clear_orphan_tags": {
                "type": "boolean",
                "label": "Clear orphan generated tags",
                "description": "Remove task-generated tags that are no longer attached to any bounding box.",
                "default": True,
            },
            "recompute_phashes": {
                "type": "boolean",
                "label": "Recompute perceptual hashes",
                "description": (
                    "Recompute phashes for every image using the hash sizes configured in Settings, "
                    "then rebuild the BK-tree deduplication index. Required after changing hash sizes."
                ),
                "default": False,
            },
        }
