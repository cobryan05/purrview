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
from db_client.models.image_stubs_needing_phash_payload import ImageStubsNeedingPhashPayload
from db_client.models.prune_images_payload import PruneImagesPayload
from db_client.models.rebuild_hash_index_response import RebuildHashIndexResponse
from db_client.models.recompute_phash_payload import RecomputePhashPayload
from db_client.models.delete_images_by_directory_payload import DeleteImagesByDirectoryPayload
from db_client.models.reconcile_image_payload import ReconcileImagePayload

from apps.helpers.consts import TaskStatus

from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

_VALID_SYNC_MODES = {"merge", "replace_db_with_json", "replace_json_with_db", "keep_newer"}


@register_task()
class DatabaseMaintenanceTask(Task):
    """Modular DB maintenance: each option is independent and skips unnecessary work.

    Phases (each independently toggleable):
      sync_jsons      — Walk disk, sync each file's JSON sidecar with the DB.
                        Also the only phase that detects moved files.
      remove_deleted  — Delete DB records whose image file is gone.
                        If sync_jsons ran first, sidecar UUIDs protect moved files.
                        If sync_jsons is off, the backend checks file existence directly
                        (no disk walk needed).
      clear_orphan_tags — Remove TASK_GENERATED tags no longer attached to any bbox.
      fill_missing_phashes — Hash only images/bboxes missing a hash or hashed at a
                        different size.  Much faster than a full recompute.
      recompute_phashes — Recompute every phash regardless of existing values.
                        Implies fill_missing_phashes (subsumes it).

    Disk listing is skipped entirely when only hash work or tag clearing is requested.
    """

    # Approximate relative cost of each phase — used to portion the progress bar.
    _PHASE_WEIGHTS: dict[str, float] = {
        "reconcile": 50,
        "remove_dirs": 15,
        "prune": 10,
        "tags": 3,
        "hash": 30,
        "rebuild": 7,
    }

    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        self._status_msg: str = "Initializing"
        self._remove_deleted: bool = bool(params.get("remove_deleted", True))
        self._sync_jsons: bool = bool(params.get("sync_jsons", True))
        self._sync_mode: str = params.get("sync_mode", "merge")
        self._clear_orphan_tags: bool = bool(params.get("clear_orphan_tags", True))
        self._fill_missing_phashes: bool = bool(params.get("fill_missing_phashes", False))
        self._recompute_phashes: bool = bool(params.get("recompute_phashes", False))
        self._phash_concurrency: int = max(1, int(params.get("phash_concurrency", 4)))
        self._remove_directories: list[str] = list(params.get("remove_directories") or [])
        self._remove_directories_delete_sidecars: bool = bool(
            params.get("remove_directories_delete_sidecars", True)
        )

        if self._sync_mode not in _VALID_SYNC_MODES:
            raise ValueError(f"Invalid sync_mode: {self._sync_mode!r}. Must be one of {_VALID_SYNC_MODES}")

    # ── Orchestrator ───────────────────────────────────────────────────────────

    async def _run(self) -> dict[str, Any]:
        db_api_client = self._manager.get_db_api_client()
        images_api = ImagesApi(db_api_client)
        tags_api = TagsApi(db_api_client)

        _has_hash_work = self._recompute_phashes or self._fill_missing_phashes

        # Build a dynamic progress budget: only allocate space to active phases.
        active = []
        if self._sync_jsons:                      active.append("reconcile")
        if self._remove_directories:              active.append("remove_dirs")
        if self._remove_deleted:                  active.append("prune")
        if self._clear_orphan_tags:               active.append("tags")
        if _has_hash_work:                        active.extend(["hash", "rebuild"])

        total_w = sum(self._PHASE_WEIGHTS[p] for p in active) or 1
        ends: dict[str, float] = {}
        cum = 0.0
        for p in active:
            cum += self._PHASE_WEIGHTS[p] / total_w * 100.0
            ends[p] = cum

        summary: dict[str, Any] = {}

        # ── 1. Reconcile (disk scan + sidecar sync) ───────────────────────────
        known_disk_uuids: list[str] = []
        if self._sync_jsons and not self._cancel_flag.is_set():
            r = await self._step_reconcile(images_api, 0.0, ends["reconcile"])
            known_disk_uuids = r.pop("known_disk_uuids")
            summary.update(r)

        # ── 2. Remove directories ─────────────────────────────────────────────
        if self._remove_directories and not self._cancel_flag.is_set():
            r = await self._step_remove_directories(images_api, ends.get("remove_dirs", 100.0))
            summary.update(r)

        # ── 3. Prune ──────────────────────────────────────────────────────────
        if self._remove_deleted and not self._cancel_flag.is_set():
            # If reconcile ran, known_disk_uuids carries sidecar UUIDs so moved
            # files are protected.  If reconcile was skipped, pass [] — the backend
            # checks file existence directly (no second disk walk needed).
            summary["pruned"] = await self._step_prune(
                images_api, known_disk_uuids, ends.get("prune", 100.0)
            )

        # ── 4. Clear orphan tags ──────────────────────────────────────────────
        if self._clear_orphan_tags and not self._cancel_flag.is_set():
            summary["tags_cleared"] = await self._step_clear_tags(tags_api, ends.get("tags", 100.0))

        # ── 5. Hash work ──────────────────────────────────────────────────────
        if _has_hash_work and not self._cancel_flag.is_set():
            settings_api = SettingsApi(db_api_client)
            h = await self._step_hash(
                images_api, settings_api,
                hash_end=ends.get("hash", 95.0),
                rebuild_end=ends.get("rebuild", 100.0),
            )
            summary.update(h)

        self._update_progress(100.0)
        self._status_msg = "Done"

        log_parts: list[str] = []
        if "files_found" in summary:      log_parts.append(f"{summary['files_found']} files on disk")
        if "reconciled" in summary:       log_parts.append(f"{summary['reconciled']} synced")
        if "moved" in summary:            log_parts.append(f"{summary['moved']} moves detected")
        if "errors" in summary:           log_parts.append(f"{summary['errors']} errors")
        if "dirs_removed" in summary:     log_parts.append(f"{summary['dirs_removed']} dir records removed")
        if "sidecars_deleted" in summary: log_parts.append(f"{summary['sidecars_deleted']} sidecars deleted")
        if "pruned" in summary:           log_parts.append(f"{summary['pruned']} pruned")
        if "tags_cleared" in summary:     log_parts.append(f"{summary['tags_cleared']} orphan tags cleared")
        if "phash_processed" in summary:
            verb = "recomputed" if self._recompute_phashes else "filled"
            log_parts.append(f"{summary['phash_processed']} hashes {verb}")
        self.log("Complete: " + (", ".join(log_parts) or "nothing to do"))
        return summary

    # ── Phase helpers ──────────────────────────────────────────────────────────

    async def _step_reconcile(self, images_api, start_pct: float, end_pct: float) -> dict:
        """Walk disk, sync each file's JSON sidecar with the DB, collect sidecar UUIDs."""
        self._status_msg = "Listing files on disk…"
        self.log("Listing image files on disk")
        await self.heartbeat(force=True)

        list_resp = await asyncio.to_thread(images_api.list_images, path="/", pattern="*", recursive=True)
        image_files = [e for e in (list_resp.files or []) if not e.is_dir]
        total = len(image_files)
        self.log(f"Found {total} image files")

        label = "Reconciling" if self._sync_jsons else "Scanning"
        self._status_msg = f"{label} 0 / {total}"

        known_disk_uuids: list[str] = []
        reconciled = moved = errors = 0

        for i, entry in enumerate(image_files):
            if self._cancel_flag.is_set():
                self.log("Cancelled during file reconcile")
                break
            await self._check_pause()

            rel_path = entry.path
            sidecar_uuid = await self._get_sidecar_uuid(rel_path)
            if sidecar_uuid:
                known_disk_uuids.append(sidecar_uuid)

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

            self._update_progress(start_pct + (i + 1) / max(total, 1) * (end_pct - start_pct))
            self._status_msg = f"{label} {i + 1} / {total}"

        return {
            "known_disk_uuids": known_disk_uuids,
            "files_found": total,
            "reconciled": reconciled,
            "moved": moved,
            "errors": errors,
        }

    async def _step_prune(self, images_api, known_disk_uuids: list[str], end_pct: float) -> int:
        """Delete DB records for images no longer on disk.

        When known_disk_uuids is non-empty (populated by a preceding reconcile pass),
        those UUIDs shield moved files from deletion.  When empty (reconcile was
        skipped), the backend checks each record's file path directly — no disk walk
        needed in the task.
        """
        self._status_msg = "Pruning missing images…"
        self.log("Pruning DB records for missing images")
        await self.heartbeat(force=True)
        pruned = 0
        try:
            prune_resp = await asyncio.to_thread(
                images_api.prune_missing_images,
                PruneImagesPayload(known_disk_uuids=known_disk_uuids),
            )
            pruned = getattr(prune_resp, "deleted_count", 0) or 0
            self.log(f"Pruned {pruned} missing image records")
        except Exception:
            logger.exception("prune_missing_images failed")
        self._update_progress(end_pct)
        return pruned

    async def _step_remove_directories(self, images_api, end_pct: float) -> dict:
        """Delete all DB records for images under the requested directories.

        Optionally also removes the JSON sidecar files from disk.
        """
        dirs = self._remove_directories
        self._status_msg = f"Removing {len(dirs)} director{'y' if len(dirs) == 1 else 'ies'} from DB…"
        self.log(f"Removing DB records for images under: {dirs}")
        await self.heartbeat(force=True)

        dirs_removed = 0
        sidecars_deleted = 0
        try:
            del_resp = await asyncio.to_thread(
                images_api.delete_images_by_directory,
                DeleteImagesByDirectoryPayload(directories=dirs),
            )
            dirs_removed = getattr(del_resp, "deleted_count", 0) or 0
            self.log(f"Removed {dirs_removed} image records from DB")
        except Exception:
            logger.exception("delete_images_by_directory failed")
        else:
            if self._remove_directories_delete_sidecars and not self._cancel_flag.is_set():
                self._status_msg = "Deleting sidecar files…"
                files_root = Path(os.environ.get("FILES_ROOT", "/app/image_datasets")).resolve()
                for rel_dir in dirs:
                    if self._cancel_flag.is_set():
                        break
                    # Guard against path traversal: resolve and verify the result stays under files_root.
                    abs_dir = (files_root / rel_dir.lstrip("/")).resolve()
                    try:
                        abs_dir.relative_to(files_root)
                    except ValueError:
                        logger.warning("Skipping directory outside files root: %s", rel_dir)
                        continue
                    if not abs_dir.is_dir():
                        continue
                    for sidecar in abs_dir.rglob("*.json"):
                        if self._cancel_flag.is_set():
                            break
                        try:
                            sidecar.unlink()
                            sidecars_deleted += 1
                        except Exception:
                            logger.exception("Failed to delete sidecar %s", sidecar)
                self.log(f"Deleted {sidecars_deleted} sidecar files")

        self._update_progress(end_pct)
        return {"dirs_removed": dirs_removed, "sidecars_deleted": sidecars_deleted}

    async def _step_clear_tags(self, tags_api, end_pct: float) -> int:
        """Remove TASK_GENERATED tags no longer attached to any bounding box."""
        self._status_msg = "Clearing orphan tags…"
        self.log("Clearing orphan TASK_GENERATED tags")
        await self.heartbeat(force=True)
        tags_cleared = 0
        try:
            tags_resp = await asyncio.to_thread(tags_api.clear_orphan_generated_tags)
            tags_cleared = getattr(tags_resp, "deleted_count", 0) or 0
            self.log(f"Cleared {tags_cleared} orphan tags")
        except Exception:
            logger.exception("clear_orphan_generated_tags failed")
        self._update_progress(end_pct)
        return tags_cleared

    async def _step_hash(
        self,
        images_api,
        settings_api,
        hash_end: float,
        rebuild_end: float,
    ) -> dict:
        """Compute perceptual hashes concurrently, then rebuild the BK-tree index."""
        image_hash_size, bbox_hash_size = await asyncio.gather(
            self._read_int_setting(settings_api, "dedup_image_hash_size", 8),
            self._read_int_setting(settings_api, "dedup_bbox_hash_size", 16),
        )
        image_hex_len = image_hash_size * image_hash_size // 4
        bbox_hex_len = bbox_hash_size * bbox_hash_size // 4

        if self._recompute_phashes:
            self._status_msg = "Listing images for full phash recompute…"
            self.log(f"Recomputing ALL phashes (image_hash_size={image_hash_size}, bbox_hash_size={bbox_hash_size})")
            await self.heartbeat(force=True)
            stubs_resp = await asyncio.to_thread(images_api.list_image_stubs)
            mode_label = "Recomputing"
        else:
            self._status_msg = "Finding images with missing / wrong-size phashes…"
            self.log(f"Filling missing phashes (image_hex_len={image_hex_len}, bbox_hex_len={bbox_hex_len})")
            await self.heartbeat(force=True)
            stubs_resp = await asyncio.to_thread(
                images_api.list_image_stubs_needing_phash,
                ImageStubsNeedingPhashPayload(image_hex_len=image_hex_len, bbox_hex_len=bbox_hex_len),
            )
            mode_label = "Hashing"

        stubs = stubs_resp.stubs or []
        phash_total = len(stubs)
        self.log(f"{mode_label} hashes for {phash_total} images (concurrency={self._phash_concurrency})")
        self._status_msg = f"{mode_label} 0 / {phash_total}"

        # hash_end is the progress target after all images are done; reserve the
        # last slice of that range for the rebuild step.
        hash_start = self._progress
        hash_work_end = hash_start + (hash_end - hash_start) * 0.92

        phash_processed = 0
        phash_errors = 0
        sem = asyncio.Semaphore(self._phash_concurrency)
        payload = RecomputePhashPayload(image_hash_size=image_hash_size, bbox_hash_size=bbox_hash_size)

        async def _hash_one(stub) -> None:
            nonlocal phash_processed, phash_errors
            if self._cancel_flag.is_set():
                return
            async with sem:
                if self._cancel_flag.is_set():
                    return
                try:
                    await asyncio.to_thread(images_api.recompute_image_phash, stub.uuid, payload)
                    phash_processed += 1
                except Exception:
                    logger.exception("recompute_image_phash failed for %s", stub.uuid)
                    self.log(f"Error hashing {stub.uuid}")
                    phash_errors += 1

                done = phash_processed + phash_errors
                self._update_progress(hash_start + done / max(phash_total, 1) * (hash_work_end - hash_start))
                self._status_msg = (
                    f"{mode_label} {done} / {phash_total} — {phash_processed} done"
                    + (f", {phash_errors} errors" if phash_errors else "")
                )
                await self.heartbeat()

        await asyncio.gather(*[_hash_one(stub) for stub in stubs], return_exceptions=True)

        if not self._cancel_flag.is_set():
            self._status_msg = "Rebuilding hash index…"
            self.log("Rebuilding BK-tree hash index")
            await self.heartbeat(force=True)
            try:
                rebuild_resp: RebuildHashIndexResponse = await asyncio.to_thread(images_api.rebuild_hash_index)
                img_cnt = getattr(rebuild_resp, "image_count", 0) or 0
                bbox_cnt = getattr(rebuild_resp, "bbox_count", 0) or 0
                self.log(f"Index rebuilt: {img_cnt} images, {bbox_cnt} bboxes")
            except Exception:
                logger.exception("rebuild_hash_index failed")
                self.log("Failed to rebuild hash index")
            self._update_progress(rebuild_end)

        return {"phash_processed": phash_processed, "phash_errors": phash_errors}

    # ── Utilities ──────────────────────────────────────────────────────────────

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

    def get_notices(self) -> list[dict]:
        status = self.get_status()
        if status not in (TaskStatus.RUNNING, TaskStatus.PAUSED):
            return []
        recompute = getattr(self, "_recompute_phashes", False)
        fill_missing = getattr(self, "_fill_missing_phashes", False)
        if recompute:
            msg = "Recomputing all perceptual hashes — duplicate search results may be incomplete"
        elif fill_missing:
            msg = "Computing missing perceptual hashes — duplicate search results may be incomplete"
        else:
            return []
        return [{"key": "hash_in_progress", "level": "warning", "message": msg}]

    def get_status_message(self) -> str:
        return self._status_msg

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        return {
            "meta": {
                "order": ["sidecar_sync", "remove_records", "tag_cleanup", "perceptual_hashes"]
            },
            "sidecar_sync": {
                "type": "group",
                "label": "Maintenance",
                "collapsed": False,
                "fields": {
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
                        "label": "Remove missing images (DB only — does NOT delete files)",
                        "description": (
                            "Remove DB records for images whose file no longer exists on disk. "
                            "Does NOT delete any image files from the filesystem. "
                            "If 'Sync sidecars' is also on, moved files are protected via sidecar UUIDs. "
                            "If 'Sync sidecars' is off, file existence is checked directly — no extra disk scan."
                        ),
                        "default": True,
                    },
                },
            },
            "remove_records": {
                "type": "group",
                "label": "Remove Records",
                "collapsed": False,
                "fields": {
                    "remove_directories": {
                        "type": "array",
                        "label": "Remove directories from DB",
                        "description": (
                            "Remove all DB records (images and their bounding boxes) for images under these "
                            "directories. Does not delete image files from disk."
                        ),
                        "default": [],
                        "items": {
                            "type": "directory",
                            "label": "Directory",
                            "description": "Directory path relative to the image dataset root",
                            "required": True,
                        },
                    },
                    "remove_directories_delete_sidecars": {
                        "type": "boolean",
                        "label": "Also delete sidecar files from the directories",
                        "description": (
                            "Also delete the JSON sidecar (.json) files from disk for each directory listed "
                            "in 'Remove directories from DB'. Only applies when that list is non-empty — "
                            "has no effect otherwise."
                        ),
                        "default": True,
                    },
                },
            },
            "tag_cleanup": {
                "type": "group",
                "label": "Tag Cleanup",
                "collapsed": False,
                "fields": {
                    "clear_orphan_tags": {
                        "type": "boolean",
                        "label": "Clear orphan generated tags",
                        "description": "Remove task-generated tags that are no longer attached to any bounding box.",
                        "default": True,
                    },
                },
            },
            "perceptual_hashes": {
                "type": "group",
                "label": "Perceptual Hashes",
                "collapsed": False,
                "fields": {
                    "fill_missing_phashes": {
                        "type": "boolean",
                        "label": "Fill missing perceptual hashes",
                        "description": (
                            "Compute phashes only for images/bboxes that have no hash or were hashed at a "
                            "different size than the current Settings. Much faster than a full recompute when "
                            "most data is already hashed. Rebuilds the BK-tree index when done."
                        ),
                        "default": False,
                    },
                    "recompute_phashes": {
                        "type": "boolean",
                        "label": "Recompute ALL perceptual hashes",
                        "description": (
                            "Recompute phashes for every image using the hash sizes configured in Settings, "
                            "then rebuild the BK-tree deduplication index. Required after changing hash sizes. "
                            "Use 'Fill missing' instead if you only want to backfill new images."
                        ),
                        "default": False,
                    },
                    "phash_concurrency": {
                        "type": "number",
                        "label": "Hash concurrency",
                        "description": (
                            "Number of images to hash in parallel. Higher values saturate disk and CPU faster "
                            "but use more memory. 4 is a safe default; raise to 8–16 on fast SSDs."
                        ),
                        "default": 4,
                        "min": 1,
                        "max": 32,
                    },
                },
            },
        }
