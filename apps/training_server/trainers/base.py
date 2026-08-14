"""Abstract base class for all trainers."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

#: Job-config key for the per-training-run "remember exact image list" opt-out.
#: Absent means ON — every trainer has always recorded the manifest, so the
#: default must preserve that. Threaded explicitly by TrainModelTask rather than
#: left to the ``**hyperparams`` merge; see NOTES.md on trainers disagreeing over
#: hyperparam key names. One key, read in exactly one place: ``image_manifest_fields``.
RECORD_IMAGE_MANIFEST_KEY = "record_image_manifest"


def image_manifest_fields(
    config: dict[str, Any],
    train_image_uuids: list[str],
    val_image_uuids: list[str],
) -> dict[str, list[str]]:
    """Return the sidecar's train/val image-uuid fields, or ``{}`` if opted out.

    Every trainer splats this into its sidecar dict instead of writing
    ``train_image_uuids``/``val_image_uuids`` directly, so the opt-out cannot be
    honoured by three trainers and silently ignored by the fourth. Opting out
    omits the keys entirely rather than writing empty lists — a reader can then
    tell "not recorded" from "recorded, and it was empty".
    """
    if not config.get(RECORD_IMAGE_MANIFEST_KEY, True):
        return {}
    return {
        "train_image_uuids": list(train_image_uuids),
        "val_image_uuids": list(val_image_uuids),
    }


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

    @staticmethod
    def shutdown_dataloaders(*loaders: Any) -> None:
        """Release a DataLoader's worker processes and the pipes they hold.

        Needed for the same reason ``release_gpu_memory`` is: the
        training_server process outlives every job it runs, so whatever a job
        fails to hand back is leaked for the life of the server rather than
        reclaimed at exit. Each worker costs file descriptors, and at the
        default ``ulimit -n`` of 1024 the *next* job then dies on an
        unrelated-looking ``[Errno 24] Too many open files`` while opening a
        temp file — nowhere near the actual cause.

        Which loaders actually leak is worth being precise about, because it is
        not all of them:

        - A plain ``torch`` ``DataLoader`` with ``persistent_workers=False``
          (the default, and what our classifier trainers use) builds a fresh
          iterator per ``__iter__`` and keeps **no** reference to it — its
          ``_iterator`` attribute stays ``None``. The iterator is a local of the
          training loop, so refcounting frees it on loop exit (``break``
          included) and its ``__del__`` stops the workers. Nothing to do; these
          are passed in only so the gc pass below can catch one held alive by a
          lingering exception traceback.
        - Ultralytics' ``InfiniteDataLoader`` assigns
          ``self.iterator = super().__iter__()`` in ``__init__`` and holds that
          strong reference for the object's whole life, deliberately, to reuse
          workers across epochs. Its workers therefore live exactly as long as
          the loader does — and ultralytics parks the loader on the trainer it
          hangs off the model. That is the real leak, and it is why this exists.

        Uses each loader's own public ``close()`` where it has one — that is
        ultralytics' documented "shut down persistent workers when the
        DataLoader is no longer needed", and it handles unregistering the
        workers from torch's SIGCHLD watchdog too. No torch internals are
        touched here: for plain loaders the supported mechanism is simply
        dropping the last reference, which the ``gc.collect()`` below forces.

        Best-effort throughout. Cleanup runs in a ``finally`` and must never be
        the thing that fails a training run that otherwise succeeded.
        """
        import gc

        for loader in loaders:
            if loader is None:
                continue
            close = getattr(loader, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception:  # noqa: BLE001 — see docstring: never fail the job
                # Module-local logger: this file imports `logging` but defines no
                # module-level logger, and each trainer makes its own.
                logging.getLogger(__name__).exception("Failed to close a DataLoader")

        # Backstop for loaders with no close(): frees any iterator still held by
        # a reference cycle or an exception traceback, whose __del__ is torch's
        # own supported worker shutdown.
        gc.collect()

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
