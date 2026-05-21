import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import cv2
import db_client
from db_client.api.images_api import ImagesApi
from db_client.api.tags_api import TagsApi
from db_client.models.bounding_box_metadata_model import BoundingBoxMetadataModel
from db_client.models.update_metadata_payload import UpdateMetadataPayload
import inference_client
from inference_client.api.inference_api import InferenceApi
from inference_client.api.models_api import ModelsApi
from inference_client.models.inference_payload import InferencePayload
from inference_client.models.pin_model_payload import PinModelPayload

from apps.helpers.imageUtils import base64_encode_png

from .Registry import register_task
from .Task import Task
from .utils.imageHelper import ImageHelper
from .utils.sourceHelper import SourceHelper

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@dataclass
class TargetLabel:
    label_uuid: str
    min_confidence: float


@register_task()
class CollectTrainingImagesTask(Task):
    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        self._status_msg: str = "Initializing"
        self._source_uuid: str = params["source_uuid"]
        self._output_dir: str = params["output_dir"]
        self._model_names: list[str] = params.get("model_names", [])
        self._target_labels: list[TargetLabel] = [
            TargetLabel(label_uuid=t["label_uuid"], min_confidence=float(t["min_confidence"]))
            for t in params.get("target_labels", [])
        ]
        self._target_label_map: dict[str, float] = {
            t.label_uuid: t.min_confidence for t in self._target_labels
        }
        self._min_all_conf: float = min((t.min_confidence for t in self._target_labels), default=0.5)
        self._images_saved: int = resume_data.get("images_saved", 0) if resume_data else 0
        self._unverified_tag_uuid: Optional[str] = None

    async def _run(self) -> dict[str, Any]:
        self._status_msg = "Looking up system tags"
        db_api_client = self._manager.get_db_api_client()
        inference_api_client = self._manager.get_inference_api_client()

        tags_resp = await asyncio.to_thread(TagsApi(db_api_client).list_tags)
        self._unverified_tag_uuid = next(
            (t.uuid for t in (tags_resp.tags or []) if t.name == "Unverified"), None
        )
        if not self._unverified_tag_uuid:
            raise RuntimeError("'Unverified' system tag not found in DB")

        self._status_msg = "Getting image provider"
        source_helper = SourceHelper(db_api_client)
        provider = source_helper.get_image_provider(self._source_uuid)

        self._status_msg = "Pinning models"
        models_api = ModelsApi(inference_api_client)
        pin_ids: dict[str, Optional[str]] = {}
        for name in self._model_names:
            try:
                resp = await asyncio.to_thread(
                    models_api.pin_model_api_api_models_model_name_pin_post,
                    name,
                    PinModelPayload(model_name=name, duration=3600),
                )
                pin_ids[name] = resp.pin_id
            except Exception:
                logger.exception(f"Failed to pin model {name}")
                pin_ids[name] = None

        images_api = ImagesApi(db_api_client)
        inference_api = InferenceApi(inference_api_client)
        files_root = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))

        self._status_msg = "Running"
        await provider.start()
        try:
            while not self._cancel_flag.is_set():
                provided = await provider.getNextImage()
                if provided is None:
                    break

                base64_img = base64_encode_png(provided.image)

                matching = []
                for name in self._model_names:
                    try:
                        result = await asyncio.to_thread(
                            inference_api.inference,
                            name,
                            InferencePayload(
                                image_base64=base64_img,
                                conf_thresh=self._min_all_conf,
                                pin_id=pin_ids.get(name),
                            ),
                        )
                        for det in result.result.detections or []:
                            luuid = det.bbox.label_uuid
                            if luuid and luuid in self._target_label_map:
                                if det.confidence >= self._target_label_map[luuid]:
                                    matching.append(det)
                    except Exception:
                        logger.exception(f"Inference error on model {name}")

                if not matching:
                    continue

                source_name = ImageHelper.sanitize_filename(provided.metadata.source or "unknown")
                frame_idx = provided.metadata.frame_idx or 0
                filename = f"{source_name}_{frame_idx}.png"
                rel_path = str(Path(self._output_dir) / filename)
                abs_path = files_root / self._output_dir / filename
                abs_path.parent.mkdir(parents=True, exist_ok=True)

                if not cv2.imwrite(str(abs_path), provided.image):
                    logger.warning(f"Failed to write image to {abs_path}")
                    continue

                boxes = [
                    BoundingBoxMetadataModel(
                        bbox_uuid=str(uuid4()),
                        x=det.bbox.x,
                        y=det.bbox.y,
                        width=det.bbox.width,
                        height=det.bbox.height,
                        label_uuid=det.bbox.label_uuid,
                        class_str=det.bbox.class_str,
                        tag_uuids=[self._unverified_tag_uuid],
                    )
                    for det in matching
                ]

                try:
                    await asyncio.to_thread(
                        images_api.update_image_metadata,
                        UpdateMetadataPayload(image_path=rel_path, boxes=boxes),
                    )
                except Exception:
                    logger.exception(f"Failed to save metadata for {rel_path}")
                    continue

                self._images_saved += 1
                self._status_msg = f"Saved {self._images_saved} images"
                self._update_progress(self._images_saved)

                if self._data_req_flag.is_set():
                    self._data_req_flag.clear()
                    self._resume_data = {"images_saved": self._images_saved}
                    self._data_ready_flag.set()

        finally:
            await provider.stop()

        self._resume_data = {"images_saved": self._images_saved}
        self._data_ready_flag.set()
        return {"images_saved": self._images_saved}

    async def _deinit(self) -> None:
        pass

    def get_status_message(self) -> str:
        return self._status_msg

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        return {
            "meta": {
                "order": ["source_uuid", "output_dir", "model_label_config", "min_capture_interval"]
            },
            "source_uuid": {
                "type": "source_uuid",
                "label": "Image Source",
                "required": True,
                "description": "Pre-configured source to pull images from",
            },
            "output_dir": {
                "type": "string",
                "label": "Output Directory",
                "required": True,
                "description": "Directory relative to image dataset root where images will be saved",
            },
            "model_label_config": {
                "type": "array",
                "required": "true",
                "label": "Model/Label Configuration",
                "items": {
                    "type": "model_label",
                    "label": "Model/Label selection",
                    "required": True,
                    "description": "Labels to collect",
                },
            },
            "min_capture_interval": {
                "type": "int",
                "label": "Min Capture Interval (s)",
                "default": 0,
                "required": False,
                "description": "Minimum seconds between frame grabs (reserved for live streams)",
            },
        }
