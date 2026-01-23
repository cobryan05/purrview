import base64
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
from dacite import Config, from_dict
from db_client.api.sources_api import SourcesApi
from inference_client.api.inference_api import InferenceApi
from inference_client.api.models_api import ModelsApi
from inference_client.models.recognize_payload import RecognizePayload
from inference_client.models.pin_model_payload import PinModelPayload
from apps.helpers.consts import JsonValues
from apps.helpers.imageProviders.imageProvider import ImageProvider
from apps.helpers.imageProviders.Registry import image_provider_registry

from .Registry import register_task
from .Task import Task

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
        status: str = JsonValues.SUCCESS
        self._status_msg = "Initializing Provider"

        # Get ImageProvider
        sources_api: SourcesApi = SourcesApi(self._manager.get_db_api_client())
        api_response = sources_api.get_sources()
        if api_response.status != JsonValues.SUCCESS:
            raise Exception("Failed to retrieve source list")
        source_metadata = api_response.sources.get(self._source_uuid, None) if api_response.sources else None
        if not source_metadata:
            raise Exception(f"Source {self._source_uuid} not found")
        image_provider: ImageProvider = image_provider_registry[source_metadata.typename](**source_metadata.params)

        # Get ImageRecognizer
        inference_api: InferenceApi = InferenceApi(self._manager.get_inference_api_client())

        models_api: ModelsApi = ModelsApi(self._manager.get_inference_api_client())

        pin_ids = {}
        for config in self._model_label_configs:
            pin_payload: PinModelPayload = PinModelPayload(model_name=config.modelName,duration=10)
            pin_response = models_api.pin_model(pin_payload)
            pin_ids[config.modelName] = pin_response.get("pin_id")


        self._status_msg = "Running"
        await image_provider.start()
        try:
            while self._cancel_flag.is_set() is False:
                image_with_metadata = await image_provider.getNextImage()
                if image_with_metadata is None:
                    break
                success, buf = cv2.imencode(".png", image_with_metadata.image)
                image_base64 = base64.b64encode(buf).decode("utf-8")
                results = {}
                for config in self._model_label_configs:
                    payload: RecognizePayload = RecognizePayload(
                        model_name=config.modelName,
                        image_base64=image_base64,
                        conf_thresh=min(config.classesValues.values()),
                    )
                    result = inference_api.recognize(payload)
                    logger.info(result)

        finally:
            await image_provider.stop()

        # TODO: Unpin models

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
