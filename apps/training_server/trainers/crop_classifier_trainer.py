"""Crop classifier trainer stub.

Trains image classifiers on bounding-box crops (one class per direct child label).
Not yet implemented.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from . import register_trainer
from .base import Trainer


@register_trainer("crop_classifier")
class CropClassifierTrainer(Trainer):
    async def train(
        self,
        config: dict[str, Any],
        progress_cb: Callable[[dict], None],
        cancel_event: asyncio.Event,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "CropClassifierTrainer is not yet implemented. "
            "This stub is a placeholder for future crop-level classification training."
        )
