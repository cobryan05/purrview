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

    @staticmethod
    def release_gpu_memory() -> None:
        """Best-effort release of GPU memory back to the driver after a job ends.

        A trainer's ``model``/optimizer/loader locals going out of scope is not
        enough on its own: PyTorch's caching allocator holds freed CUDA memory
        for reuse rather than returning it to the driver, and the training_server
        process stays alive between jobs (single in-process worker). Without an
        explicit ``empty_cache()``, VRAM used by one job appears "stuck" even
        when idle. Call this from a ``finally`` block wrapping every trainer's
        ``train()`` so it runs on success, error, and cancellation alike.

        Only touches ``torch`` if it is already imported (it always will be in
        production — every real trainer imports it before this runs). Avoids
        triggering a fresh, first-time ``import torch`` as a side effect of
        cleanup, which is both pointless (nothing to release yet) and unsafe to
        do concurrently from a background thread — importing torch for the
        first time from two threads at once can hit a known thread-safety bug
        in ``torch.overrides`` (docstring double-registration).
        """
        import gc
        import sys

        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

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
