"""Task that backfills perceptual hashes for all images/bboxes that have phash=NULL,
then triggers a full BK-tree index rebuild.
"""

import asyncio
import logging
from typing import Any, Optional

import db_client
from db_client.api.images_api import ImagesApi
from db_client.models.recompute_phash_payload import RecomputePhashPayload
from db_client.models.rebuild_hash_index_response import RebuildHashIndexResponse

from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@register_task()
class ComputePhashesTask(Task):
    """Backfill perceptual hashes for images/bboxes with phash=NULL, then rebuild the BK-tree index."""

    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        self._status_msg: str = "Initializing"
        self._image_hash_size: int = int(params.get("image_hash_size", 8))
        self._bbox_hash_size: int = int(params.get("bbox_hash_size", 16))
        self._force_recompute: bool = bool(params.get("force_recompute", False))

    async def _run(self) -> dict[str, Any]:
        db_api_client = self._manager.get_db_api_client()
        images_api = ImagesApi(db_api_client)

        self._status_msg = "Listing images"
        self.log("Listing all images")
        stubs_resp = await asyncio.to_thread(images_api.list_image_stubs)
        stubs = stubs_resp.stubs or []
        total = len(stubs)
        self.log(f"Found {total} images")

        processed = 0
        errors = 0

        for i, stub in enumerate(stubs):
            if self._cancel_flag.is_set():
                self.log("Cancelled")
                break
            await self._check_pause()

            image_uuid = stub.uuid

            try:
                await asyncio.to_thread(
                    images_api.recompute_image_phash,
                    image_uuid,
                    RecomputePhashPayload(
                        image_hash_size=self._image_hash_size,
                        bbox_hash_size=self._bbox_hash_size,
                    ),
                )
                processed += 1
            except Exception:
                logger.exception("recompute_image_phash failed for %s", image_uuid)
                self.log(f"Error processing {image_uuid}")
                errors += 1

            self._update_progress(i / max(total, 1) * 90.0)
            self._status_msg = f"Processing {i + 1}/{total} — {processed} done, {errors} errors"
            await asyncio.sleep(0)

        # Rebuild the hash index after backfill.
        if not self._cancel_flag.is_set():
            self._status_msg = "Rebuilding hash index"
            self.log("Rebuilding BK-tree hash index")
            try:
                rebuild_resp: RebuildHashIndexResponse = await asyncio.to_thread(
                    images_api.rebuild_hash_index
                )
                image_count = getattr(rebuild_resp, "image_count", 0) or 0
                bbox_count = getattr(rebuild_resp, "bbox_count", 0) or 0
                self.log(f"Index rebuilt: {image_count} images, {bbox_count} bboxes")
            except Exception:
                logger.exception("rebuild_hash_index failed")
                self.log("Failed to rebuild hash index")

        self._update_progress(100.0)
        self._status_msg = "Done"
        self.log(f"Complete: {processed} processed, {errors} errors")
        return {"processed": processed, "errors": errors}

    async def _deinit(self) -> None:
        pass

    def get_status_message(self) -> str:
        return self._status_msg

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        return {
            "meta": {
                "order": ["image_hash_size", "bbox_hash_size", "force_recompute"],
            },
            "image_hash_size": {
                "type": "int",
                "label": "Image Hash Size",
                "default": 8,
                "description": (
                    "Hash grid dimension for full images (8=64-bit, 16=256-bit). "
                    "Larger = more sensitive to small differences."
                ),
            },
            "bbox_hash_size": {
                "type": "int",
                "label": "BBox Hash Size",
                "default": 16,
                "description": (
                    "Hash grid dimension for bbox crops. Higher than image hash "
                    "to distinguish similar small regions."
                ),
            },
            "force_recompute": {
                "type": "boolean",
                "label": "Force Recompute",
                "default": False,
                "description": "Recompute even if hash already exists.",
            },
        }
