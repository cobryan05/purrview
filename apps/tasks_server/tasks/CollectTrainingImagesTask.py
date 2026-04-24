import asyncio
import base64
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from dacite import Config, from_dict
from inference_client import InferenceResultResponse
from inference_client.api.inference_api import InferenceApi
from inference_client.api.models_api import ModelsApi
from inference_client.models.pin_model_payload import PinModelPayload
from inference_client.models.recognize_payload import RecognizePayload

from apps.helpers.consts import JsonValues
from apps.helpers.db.types import ImageMetadata
from apps.helpers.imageProviders.imageProvider import (
    ImageProvider,
    ImageWithProviderMetadata,
)

from .Registry import register_task
from .Task import Task
from .utils.imageHelper import ImageHelper
from .utils.sourceHelper import SourceHelper

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@dataclass
class ModelLabelValues:
    modelName: str
    classesValues: Dict[str, float]


@register_task()
class CollectTrainingImagesTask(Task):
    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        """Run any initialization logic for the task."""
        self._status_msg: str = "Creating Task"
        self._source_uuid: str = params.get("source_uuid", "")
        self._output_dir: str = params.get("output_dir", "")
        self._min_capture_interval: int = int(params.get("min_capture_interval", 0))
        self._model_label_configs: list[ModelLabelValues] = [
            from_dict(ModelLabelValues, item, Config(cast=[float])) for item in params.get("model_label_config", [])
        ]

    async def _run(self) -> dict[str, Any]:
        """Run the main logic of the task."""
        ret = {}
        db_api_client = self._manager.get_db_api_client()
        inference_api_client = self._manager.get_inference_api_client()
        status: str = JsonValues.SUCCESS
        self._status_msg = "Initializing Provider"

        source_helper: SourceHelper = SourceHelper(db_api_client)
        image_provider: ImageProvider = source_helper.get_image_provider(self._source_uuid)
        if not image_provider:
            raise KeyError("Failed to get image provider")

        inference_api: InferenceApi = InferenceApi(inference_api_client)
        models_api: ModelsApi = ModelsApi(inference_api_client)

        pin_ids = {}
        for config in self._model_label_configs:
            pin_payload: PinModelPayload = PinModelPayload(model_name=config.modelName, duration=60)
            pin_response = await asyncio.to_thread(models_api.pin_model, pin_payload)
            pin_ids[config.modelName] = pin_response.get("pin_id")

        image_helper: ImageHelper = ImageHelper(db_api_client)
        self._status_msg = "Running"

        await image_provider.start()
        try:
            while self._cancel_flag.is_set() is False:
                image_with_metadata: ImageWithProviderMetadata = await image_provider.getNextImage()
                if image_with_metadata is None:
                    break

                # Determine an output filename
                sanitized_name: str = ImageHelper.sanitize_filename(
                    f"{image_with_metadata.metadata.source}_{image_with_metadata.metadata.frame_idx}"
                )
                output_path = Path(self._output_dir) / f"{sanitized_name}.png"

                # Check if this file already exists in the database
                image_metadata: ImageMetadata = await asyncio.to_thread(
                    image_helper.get_image_metadata, str(output_path)
                )
                if image_metadata:
                    logger.info(f"Output image {output_path} already exists")
                else:
                    logger.info(f"Output image {output_path} is new")
                    image_metadata = ImageMetadata(filename=str(output_path))

                success, buf = cv2.imencode(".png", image_with_metadata.image)
                image_base64 = base64.b64encode(buf).decode("utf-8")
                results: Dict[str, InferenceResultResponse] = {}
                for config in self._model_label_configs:
                    # Get inference results
                    min_conf = min(config.classesValues.values())
                    payload: RecognizePayload = RecognizePayload(
                        model_name=config.modelName,
                        image_base64=image_base64,
                        conf_thresh=min_conf,
                        pin_id=pin_ids.get(config.modelName),
                    )

                    recognize_res = await asyncio.to_thread(inference_api.recognize, payload)
                    # Filter the results to only keep configured classes
                    filtered_detections = []
                    for detection in recognize_res.detections:
                        if detection.confidence > config.classesValues.get(detection.class_uuid, 100.0):
                            filtered_detections.append(detection)
                    results[config.modelName] = InferenceResultResponse(detections=filtered_detections)

                # TODO: De-dupe results?

                # TODO: Convert BBOX
                image_metadata.boxes = []
                for config, result in results.items():
                    image_metadata.boxes.extend(result.detections)
                image_with_metadata.metadata.boxes = image_metadata.boxes
                await asyncio.to_thread(image_helper.update_image_metadata, str(output_path), image_metadata)
                logger.info(results)

        finally:
            await image_provider.stop()

        self._update_resume_data()
        ret["status"] = status
        self._status_msg = "Done"
        return ret

    async def _deinit(self) -> None:
        # TODO: Does killing clean up
        pass

    def get_status_message(self) -> str:
        return self._status_msg

    def _update_resume_data(self) -> None:
        self._resume_data = {}
        self._data_ready_flag.set()

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        """
        Return a schema describing the parameters for this Task.
        Each key is a parameter name, value is a dict with:
            - type: str
            - required: bool
            - default: Any (optional)
            - description: str (optional)
            - options: list (optional, for enums)
            - schema: dict (optional, for nested objects)
        """
        return {
            "meta": {"order": ["source_uuid", "model_label_config", "output_dir", "min_capture_interval"]},
            "source_uuid": {
                "type": "source_uuid",
                "label": "Image Capture Source",
                "required": True,
                "description": "UUID of pre-configured Source to use",
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
            "output_dir": {
                "type": "string",
                "label": "Image Output Directory",
                "required": True,
                "description": "Directory to save the collected images",
            },
            "min_capture_interval": {
                "type": "int",
                "label": "Minimum capture Interval",
                "default": 0,
                "required": True,
                "description": "Minimum time (in seconds) between captures",
            },
        }
