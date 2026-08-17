import asyncio
import logging
from typing import Any, Optional

from apps.helpers.imageProviders.Registry import image_provider_registry
from apps.helpers.imageUtils import base64_encode_jpg
from apps.helpers.logUtils import LOG_LEVEL

from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


@register_task()
class PreviewSourceTask(Task):
    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        """Run any initialization logic for the task."""
        config = params.get("source_config", {})
        self._status_msg: str = "Creating Task"
        self._provider: str = config.get("provider", "")
        self._provider_params: dict[str, Any] = config.get("provider_params", {})
        self._params[Task.InternalKeys.ONESHOT_RESULT] = True
        self.log(f"Initializing preview for provider: {self._provider}")

    async def _run(self) -> dict[str, Any]:
        """Run the main logic of the task."""
        ret = {}
        status: str = "success"
        self._status_msg = "Initializing Provider"
        self.log("Initializing provider")
        provider_cls = image_provider_registry.get(self._provider)
        if provider_cls is None:
            # Raise instead of a bare registry KeyError: Task._execute turns
            # this into an error result with a message that names the type.
            self.log(f"Error: unknown provider type {self._provider!r}")
            raise ValueError(
                f"Unknown image provider type {self._provider!r} (registered: {sorted(image_provider_registry)})"
            )
        provider = provider_cls(**self._provider_params)
        if not provider:
            status = "error"
            self.log("Error: provider not found")
        else:
            self._status_msg = "Starting Provider"
            self.log("Starting provider")

            await asyncio.create_task(provider.start())
            self._status_msg = "Waiting for image"
            self.log("Waiting for image")

            cancel_task = asyncio.create_task(self._cancel_flag.wait())
            image_task = asyncio.create_task(provider.getNextImage())

            await self.heartbeat(force=True)
            try:
                done, pending = await asyncio.wait(
                    {image_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if image_task in done:
                    try:
                        result = image_task.result()
                        if not result:
                            raise ValueError("Invalid image result")
                        image_base64 = base64_encode_jpg(result.image)
                        ret["image"] = image_base64
                        self.log("Image received successfully")
                    except Exception:
                        logger.exception(f"Error fetching image")
                        self._status_msg = "Error"
                        self.log("Error fetching image")
                        status = "error"
                elif cancel_task in done:
                    self._status_msg = "Canceled"
                    self.log("Task canceled")
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
            "source_config": {
                "type": "source_config",
                "label": "Source Config",
                "required": False,
                "description": "Configuration for the source",
            },
        }
