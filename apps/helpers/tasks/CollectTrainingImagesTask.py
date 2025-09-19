import asyncio
import os
from typing import Any, Optional

from .Registry import register_task
from apps.helpers.imageProviders.Registry import image_provider_registry
from .Task import Task
from apps.helpers.imageUtils import base64_encode_png


@register_task()
class CollectTrainingImagesTask(Task):
    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        """Run any initialization logic for the task."""
        self._status_msg: str = "Creating Task"
        self._provider: str = params.get("provider", "")
        self._provider_params: dict[str, Any] = params.get("provider_params", {})

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
            "source_uuid": {
                "type": "source_uuid",
                "required": True,
                "description": "UUID of pre-configured Source to use",
            },
            "labels_uuid": {
                "type": "array",
                "required": True,
                "items": {"type": "string", "description": "Label UUIDs to associate with the images, one per line"},
            },
            "models": {
                "type": "array",
                "required": True,
                "items": {"type": "string", "description": "Model names to use for inference, one per line"},
            },
            "output_dir": {"type": "string", "required": True, "description": "Directory to save the collected images"},
            "capture_interval": {
                "type": "int",
                "required": False,
                "description": "Interval (in seconds) to capture images",
            },
        }
