import asyncio
import logging
from typing import Any, Optional

from db_client.api.images_api import ImagesApi
from db_client.api.labels_api import LabelsApi
from db_client.models.batch_by_uuids_payload import BatchByUuidsPayload

from apps import FILES_ROOT
from apps.helpers.bboxUtils import iou_xywh
from apps.helpers.fileUtils import get_safe_path

from .Registry import register_task
from .Task import Task
from .utils.labelHelper import build_label_ancestry_map

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

# Images fetched per DB round-trip.
_BATCH_SIZE = 100


@register_task()
class DeduplicateOverlappingBBoxesTask(Task):
    """
    Scan images in the database and delete bounding boxes made redundant by a
    more-specific (child-label) overlapping bbox on the same image.

    For each pair of bboxes with IoU >= threshold:
    - Same label: keep the first encountered, delete the duplicate.
    - One label is an ancestor of the other in the label hierarchy: keep the
      descendant (more specific), delete the ancestor-labelled bbox.
    - Unrelated labels: keep both.

    Defaults to dry_run=True so users can preview before committing deletions.
    """

    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        self._status_msg = "Initializing"
        self._directory: str = params.get("directory", "").strip().rstrip("/")
        self._iou_threshold: float = float(params.get("iou_threshold", 0.5))
        self._dry_run: bool = bool(params.get("dry_run", True))

        self._images_processed: int = resume_data.get("images_processed", 0) if resume_data else 0
        self._bboxes_removed: int = resume_data.get("bboxes_removed", 0) if resume_data else 0
        self._images_with_changes: int = resume_data.get("images_with_changes", 0) if resume_data else 0

    async def _run(self) -> dict[str, Any]:
        db_api_client = self._manager.get_db_api_client()
        images_api = ImagesApi(db_api_client)
        labels_api = LabelsApi(db_api_client)

        # Build label ancestry map: {label_uuid: set_of_ancestor_uuids}
        self._status_msg = "Loading label hierarchy"
        label_ancestry = await build_label_ancestry_map(labels_api)
        self.log(f"Loaded ancestry for {len(label_ancestry)} labels")

        # Get all image stubs (uuid + filename, no pagination)
        self._status_msg = "Loading image list"
        stubs_resp = await asyncio.to_thread(images_api.list_image_stubs)
        all_stubs = stubs_resp.stubs or []

        # Optional directory filter. The DB stores filenames as absolute paths
        # (e.g. /app/image_datasets/foo/bar.jpg), so the input directory must be
        # resolved to an absolute path under FILES_ROOT before prefix-matching.
        if self._directory:
            abs_dir = get_safe_path(FILES_ROOT, self._directory)
            if abs_dir is not None:
                prefix = str(abs_dir) + "/"
                all_stubs = [s for s in all_stubs if s.filename.startswith(prefix) or s.filename == str(abs_dir)]
            else:
                all_stubs = []

        total = len(all_stubs)
        dry_label = " [DRY RUN]" if self._dry_run else ""
        self.log(
            f"Processing {total} images"
            f" (IoU={self._iou_threshold}"
            f"{', dir=' + self._directory if self._directory else ''})"
            f"{dry_label}"
        )

        uuid_list = [s.uuid for s in all_stubs]

        for batch_start in range(0, total, _BATCH_SIZE):
            if self._cancel_flag.is_set():
                break
            await self._check_pause()

            batch_uuids = uuid_list[batch_start : batch_start + _BATCH_SIZE]
            batch_end = min(batch_start + _BATCH_SIZE, total)
            self._status_msg = f"Processing {batch_end}/{total} images, {self._bboxes_removed} redundant bbox(es) found"

            try:
                resp = await asyncio.to_thread(
                    images_api.batch_images_by_uuids,
                    BatchByUuidsPayload(image_uuids=batch_uuids),
                )
                images_data = resp.images or []
            except Exception:
                logger.exception("Failed to fetch image batch starting at index %d", batch_start)
                self._images_processed += len(batch_uuids)
                continue

            for img_dict in images_data:
                if self._cancel_flag.is_set():
                    break

                img_uuid = img_dict.get("uuid", "?")
                bboxes: list[dict] = img_dict.get("bboxes") or []
                if not bboxes:
                    self._images_processed += 1
                    continue

                to_delete = _find_redundant_bboxes(bboxes, label_ancestry, self._iou_threshold)

                if to_delete:
                    if not self._dry_run:
                        for bbox_uuid in to_delete:
                            try:
                                await asyncio.to_thread(images_api.delete_bbox, bbox_uuid)
                            except Exception:
                                logger.exception("Failed to delete bbox %s on image %s", bbox_uuid, img_uuid)
                    self._bboxes_removed += len(to_delete)
                    self._images_with_changes += 1
                    logger.debug(
                        "Image %s: %s %d redundant bbox(es)",
                        img_uuid,
                        "would remove" if self._dry_run else "removed",
                        len(to_delete),
                    )

                self._images_processed += 1

            # Yield to event loop between batches
            await asyncio.sleep(0)

            if self._data_req_flag.is_set():
                self._data_req_flag.clear()
                self._resume_data = self._build_resume_data()
                self._data_ready_flag.set()

        self._resume_data = self._build_resume_data()
        self._data_ready_flag.set()

        action_word = "would be removed" if self._dry_run else "removed"
        self.log(
            f"Complete{dry_label}: {self._images_processed} images checked, "
            f"{self._images_with_changes} affected, "
            f"{self._bboxes_removed} redundant bbox(es) {action_word}"
        )

        return {
            "images_processed": self._images_processed,
            "images_with_changes": self._images_with_changes,
            "bboxes_removed": self._bboxes_removed,
            "dry_run": self._dry_run,
        }

    def _build_resume_data(self) -> dict[str, Any]:
        return {
            "images_processed": self._images_processed,
            "bboxes_removed": self._bboxes_removed,
            "images_with_changes": self._images_with_changes,
        }

    async def _deinit(self) -> None:
        pass

    def get_status_message(self) -> str:
        return self._status_msg

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        return {
            "meta": {
                "order": ["directory", "iou_threshold", "dry_run"],
            },
            "directory": {
                "type": "directory",
                "label": "Directory Filter (optional)",
                "required": False,
                "description": ("Only process images under this directory. Leave blank to scan all images."),
            },
            "iou_threshold": {
                "type": "float",
                "label": "IoU Threshold",
                "description": (
                    "Two bboxes are considered overlapping when their intersection-over-union "
                    "is at or above this value (0–1). The less-specific (ancestor-label) bbox "
                    "is deleted. 0.5 is a reasonable starting point."
                ),
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            },
            "dry_run": {
                "type": "boolean",
                "label": "Dry Run",
                "description": (
                    "When enabled, count redundant bboxes without deleting anything. "
                    "Run dry first to verify the threshold is correct before committing."
                ),
                "default": True,
            },
        }


# ---------------------------------------------------------------------------
# Module-level helpers (not instance methods so they are testable in isolation)
# ---------------------------------------------------------------------------


def _find_redundant_bboxes(
    bboxes: list[dict],
    label_ancestry: dict[str, set[str]],
    iou_threshold: float,
) -> set[str]:
    """Return the set of bbox UUIDs that should be deleted from this image.

    Processes bboxes in the order they appear in the list (earlier = higher
    priority when labels are equal).  For each pair with IoU >= threshold:
    - same label_uuid: delete the later one.
    - one label is an ancestor of the other: delete the ancestor-labelled bbox.
    - unrelated labels: keep both.
    """
    to_delete: set[str] = set()
    kept: list[dict] = []

    for bbox in bboxes:
        b_uuid = bbox.get("uuid")
        if not b_uuid or b_uuid in to_delete:
            continue

        # NOTE: these dicts come from batch_images_by_uuids, which serializes with
        # by_alias=True (alias_generator=to_camel) — the JSON/dict key is "labelUuid",
        # not "label_uuid". Reading the wrong key here always returns None, silently
        # disabling all same-label/ancestor-label matching below (bug fixed in passing).
        b_label = bbox.get("labelUuid")
        dominated = False
        to_supersede: list[int] = []

        for i, other in enumerate(kept):
            if other.get("uuid") in to_delete:
                continue

            if iou_xywh(bbox, other) < iou_threshold:
                continue

            o_label = other.get("labelUuid")

            if b_label is not None and b_label == o_label:
                # Same label — first encountered wins
                dominated = True
                break
            elif b_label and b_label in label_ancestry.get(o_label or "", ()):
                # bbox label is an ancestor of other's → other is more specific → drop bbox
                dominated = True
                break
            elif o_label and o_label in label_ancestry.get(b_label or "", ()):
                # other label is an ancestor of bbox's → bbox is more specific → supersede other
                to_supersede.append(i)
            # else: unrelated labels — keep both

        if dominated:
            to_delete.add(b_uuid)
        else:
            for i in reversed(to_supersede):
                superseded_uuid = kept[i].get("uuid")
                if superseded_uuid:
                    to_delete.add(superseded_uuid)
                kept.pop(i)
            kept.append(bbox)

    return to_delete
