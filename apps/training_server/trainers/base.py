"""Abstract base class for all trainers."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable


class Trainer(ABC):
    # Single source of truth for this trainer's hyperparam defaults — the
    # keys it reads out of ``config`` via ``config.get(key, default)`` and
    # what it falls back to when a caller (API payload, UI) omits them.
    # Subclasses override this; also used to serve GET
    # /api/training/trainers/{trainer_type}/hyperparams so the frontend can
    # show real defaults instead of guessing.
    HYPERPARAM_DEFAULTS: dict[str, int] = {}

    def log_effective_hyperparams(
        self,
        logger: logging.Logger,
        progress_cb: Callable[[dict], None] | None = None,
        **resolved: Any,
    ) -> None:
        """Log the concrete hyperparam values this run resolved to.

        Call with whatever keys are relevant to this trainer, e.g.
        ``self.log_effective_hyperparams(logger, progress_cb, epochs=max_epochs, batch=batch_size)``.

        Also forwards the message through ``progress_cb`` (if given) so it
        reaches the frontend task log — plain ``logger.info`` only goes to
        the training server's own process logs, which the UI never shows.
        """
        parts = " ".join(f"{k}={v}" for k, v in resolved.items())
        msg = f"Effective hyperparams: {parts}"
        logger.info(msg)
        if progress_cb is not None:
            progress_cb({"progress": 0.0, "log": msg})

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
