"""Object detection trainer stub.

Trains YOLO-family object detection models. Not yet implemented.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from . import register_trainer
from .base import Trainer


@register_trainer("object_detection")
class ObjectDetectionTrainer(Trainer):
    async def train(
        self,
        config: dict[str, Any],
        progress_cb: Callable[[dict], None],
        cancel_event: asyncio.Event,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "ObjectDetectionTrainer is not yet implemented. "
            "This stub is a placeholder for future YOLO-family training support."
        )
