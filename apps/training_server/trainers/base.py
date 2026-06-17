"""Abstract base class for all trainers."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable


class Trainer(ABC):
    @abstractmethod
    async def train(
        self,
        config: dict[str, Any],
        progress_cb: Callable[[dict], None],
        cancel_event: asyncio.Event,
    ) -> dict[str, Any]:
        """Run the training job.

        Args:
            config:       Full job config dict (trainer-specific fields included).
            progress_cb:  Call each epoch with a progress dict, e.g.:
                          {"progress": 0.42, "epoch": 21, "total_epochs": 50,
                           "val_accuracy": 0.87, "log": "Epoch 21 val_loss=0.31"}
            cancel_event: Asyncio event set when cancellation is requested.
                          Poll ``cancel_event.is_set()`` between epochs and return
                          early (the queue will mark the job cancelled).

        Returns:
            Result dict written to the job; typically includes ``model_path`` and
            ``val_accuracy``.
        """
        ...
