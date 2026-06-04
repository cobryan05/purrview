import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

import db_client
import jstyleson
from db_client.api.images_api import ImagesApi
from db_client.api.tags_api import TagsApi
from db_client.models.prune_images_payload import PruneImagesPayload
from db_client.models.reconcile_image_payload import ReconcileImagePayload

from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

_VALID_SYNC_MODES = {"merge", "replace_db_with_json", "replace_json_with_db", "keep_newer"}


@register_task()
class RefreshDatabaseTask(Task):
    """Reconcile the DB against image files on disk.

    Scan order guarantees both move orderings converge:
      1. Reconcile each file found on disk (move detection + sidecar sync).
         Collect sidecar UUIDs as known_disk_uuids.
      2. Prune DB records whose file is missing and whose UUID is not known.
      3. Clear orphan TASK_GENERATED tags.
    """

    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        self._status_msg: str = "Initializing"
        self._remove_deleted: bool = bool(params.get("remove_deleted", True))
        self._sync_jsons: bool = bool(params.get("sync_jsons", True))
        self._sync_mode: str = params.get("sync_mode", "merge")
        self._clear_orphan_tags: bool = bool(params.get("clear_orphan_tags", True))

        if self._sync_mode not in _VALID_SYNC_MODES:
            raise ValueError(f"Invalid sync_mode: {self._sync_mode!r}. Must be one of {_VALID_SYNC_MODES}")

    async def _run(self) -> dict[str, Any]:
        db_api_client = self._manager.get_db_api_client()
        images_api = ImagesApi(db_api_client)
        tags_api = TagsApi(db_api_client)

        # 1. List files on disk via the db_server API.
        self._status_msg = "Listing files"
        self.log("Listing image files on disk")
        list_resp = await asyncio.to_thread(images_api.list_images, path="/", pattern="*", recursive=True)
        file_entries = list_resp.files or []
        image_files = [e for e in file_entries if not e.is_dir]
        total = len(image_files)
        self.log(f"Found {total} image files")

        # 2. Reconcile each file + collect known disk UUIDs.
        reconciled = 0
        moved = 0
        errors = 0
        known_disk_uuids: list[str] = []

        if self._sync_jsons:
            self._status_msg = f"Reconciling 0/{total}"
        else:
            self._status_msg = f"Scanning 0/{total}"

        for i, entry in enumerate(image_files):
            if self._cancel_flag.is_set():
                self.log("Cancelled during file reconcile")
                break
            await self._check_pause()

            rel_path = entry.path

            # Collect the sidecar UUID regardless of whether we sync.
            # We need it for the prune step's move guard.
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

            progress = (i + 1) / max(total, 1) * (70.0 if self._remove_deleted else 100.0)
            self._update_progress(progress)
            label = "Reconciling" if self._sync_jsons else "Scanning"
            self._status_msg = f"{label} {i + 1}/{total}"

        # 3. Prune missing images.
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
            self._update_progress(90.0)

        # 4. Clear orphan generated tags.
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
            self._update_progress(100.0)

        self._status_msg = "Done"
        summary = {
            "files_found": total,
            "reconciled": reconciled,
            "moved": moved,
            "errors": errors,
            "pruned": pruned,
            "tags_cleared": tags_cleared,
        }
        self.log(
            f"Complete: {total} files, {reconciled} synced, {moved} moves detected, "
            f"{pruned} pruned, {tags_cleared} orphan tags cleared"
        )
        return summary

    async def _get_sidecar_uuid(self, rel_path: str) -> Optional[str]:
        """Read the sidecar for rel_path and return its uuid field, or None.

        The sidecar sits at <FILES_ROOT>/<rel_path_without_extension>.json
        (matching the _get_json_sidefile convention in db_server/manager.py).
        """
        try:
            files_root = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))
            image_path = files_root / rel_path
            sidecar_path = image_path.with_suffix(".json")
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
            "meta": {"order": ["remove_deleted", "sync_jsons", "sync_mode", "clear_orphan_tags"]},
            "remove_deleted": {
                "type": "boolean",
                "label": "Remove deleted images",
                "description": "Delete DB records for images that are no longer on disk.",
                "default": True,
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
            "clear_orphan_tags": {
                "type": "boolean",
                "label": "Clear orphan generated tags",
                "description": "Remove task-generated tags that are no longer attached to any bounding box.",
                "default": True,
            },
        }
