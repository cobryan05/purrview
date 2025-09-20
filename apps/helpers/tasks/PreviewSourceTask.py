import asyncio
import logging
import os
from typing import Any, Optional

from apps.helpers.imageProviders.Registry import image_provider_registry
from apps.helpers.imageUtils import base64_encode_png

from .Registry import register_task
from .Task import Task


logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@register_task()
class PreviewSourceTask(Task):
    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        """Run any initialization logic for the task."""
        self._status_msg: str = "Creating Task"
        self._provider: str = params.get("provider", "")
        self._provider_params: dict[str, Any] = params.get("provider_params", {})
        self._params[Task.InternalKeys.ONESHOT_RESULT] = True

    async def _run(self) -> dict[str, Any]:
        """Run the main logic of the task."""
        ret = {}
        status: str = "success"
        self._status_msg = "Initializing Provider"
        provider = image_provider_registry[self._provider](**self._provider_params)
        if not provider:
            status = "error"
        else:
            self._status_msg = "Starting Provider"

            await asyncio.create_task(provider.start())
            self._status_msg = "Waiting for image"

            cancel_task = asyncio.create_task(self._cancel_flag.wait())
            image_task = asyncio.create_task(provider.getNextImage())

            try:
                done, pending = await asyncio.wait(
                    {image_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if image_task in done:
                    try:
                        image_base64 = base64_encode_png(image_task.result().image)
                        ret["image"] = image_base64
                    except Exception as e:
                        logger.exception(f"Error fetching image")
                        self._status_msg = "Error"
                        status = "error"
                elif cancel_task in done:
                    self._status_msg = "Canceled"
                    status = "canceled"

            finally:
                # Cancel any tasks still pending (cleanup)
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        self._update_resume_data()
        ret["status"] = status
        self._status_msg = f"Image received: {status}"
        await provider.stop()
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
            - help: str (optional)
            - options: list (optional, for enums)
            - schema: dict (optional, for nested objects)
        """
        return {
            "provider": {"type": "image_provider_name", "label": "Image Provider", "required": True, "description": "Image provider to test"},
            "provider_params": {
                "type": "string",
                "label": "Provider Parameters",
                "required": False,
                "description": "Additional parameters json for the image provider",
            },
        }
