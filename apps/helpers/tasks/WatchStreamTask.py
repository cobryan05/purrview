import asyncio
import os
from typing import Any, Optional

from .Registry import register_task
from .Task import Task


@register_task()
class WatchStreamTask(Task):
    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]] = None) -> None:
        """Run any initialization logic for the task."""
        pass

    async def _run(self) -> dict[str, Any]:
        """Run the main logic of the task."""
        pass

    async def _deinit(self) -> None:
        pass

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
            "rtsp_url": {"type": "string", "required": True, "help": "RTSP stream URL to watch"},
            "interval": {
                "type": "number",
                "required": False,
                "default": 10.0,
                "help": "Interval in seconds between frame processing",
            },
            "models": {
                "type": "list",
                "required": True,
                "help": "List of models to run on each frame",
                "schema": {
                    "model_name": {"type": "string", "required": True, "help": "Name of the model"},
                    "classes": {"type": "list", "required": False, "help": "List of classes to detect"},
                    "min_confidence": {
                        "type": "number",
                        "required": False,
                        "default": 0.5,
                        "help": "Minimum confidence threshold",
                    },
                },
            },
            "output_dir": {"type": "string", "required": True, "help": "Directory to save interesting results"},
        }
