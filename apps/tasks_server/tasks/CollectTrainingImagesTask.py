from typing import Any, Dict, Optional

from .Registry import register_task
from .Task import Task
from dataclasses import dataclass
from dacite import from_dict, Config

@dataclass
class ModelLabelValues:
    modelName: str
    labelValues: Dict[str, float]


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


        print("HI")

    async def _run(self) -> dict[str, Any]:
        """Run the main logic of the task."""
        ret = {}
        status: str = "success"
        self._status_msg = "Initializing Provider"

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
                "required": False,
                "description": "Minimum time (in seconds) between captures",
            },
        }
