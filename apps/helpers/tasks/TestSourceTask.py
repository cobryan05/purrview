import asyncio
import os
from typing import Any, Optional

from .Registry import register_task
from apps.helpers.imageProviders.Registry import image_provider_registry
from .Task import Task
from apps.helpers.imageUtils import base64_encode_png

@register_task()
class TestSourceTask(Task):
    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        """Run any initialization logic for the task."""
        self._status_msg: str = "Creating Task"
        self._status: str = "pending"
        self._provider: str = params.get("provider", "")
        self._provider_params: dict[str, Any] = params.get("provider_params", {})

    async def _run(self) -> dict[str, Any]:
        """Run the main logic of the task."""
        ret = {}
        status: str = "success"
        self._status_msg = "Initializing Provider"
        provider = image_provider_registry[self._provider](**self._provider_params)
        if not provider:
            status = "error"
        else:
            self._status_msg = "Waiting for image"
            image = await provider.getNextImage()
            image_base64 = base64_encode_png(image)
            ret["image"] = image_base64

        self._update_resume_data()
        ret["status"] = status
        self._status_msg = f"Image received: {status}"
        await provider.stop()
        return ret

    async def _deinit(self) -> None:
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
            - help: str (optional)
            - options: list (optional, for enums)
            - schema: dict (optional, for nested objects)
        """
        return {
            "image_provider": {"type": "string", "required": True, "help": "Image provider to test"},
            "params": {"type": "string", "required": False, "help": "Additional parameters json for the image provider"},
        }
