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


#: Machine tokens for why a training run stopped. Kept short and stable — the
#: human sentence lives beside them in ``stop_detail``, which is free to be reworded.
STOP_COMPLETED = "completed"  # ran the full epoch budget
STOP_EARLY = "early_stopping"  # patience exhausted before the budget
STOP_CANCELLED = "cancelled"  # a human asked it to stop


def stop_summary(
    *,
    reason: str,
    detail: str,
    epochs_run: int,
    max_epochs: int,
    best_epoch: int | None = None,
) -> dict[str, Any]:
    """The "why did it stop" fields every trainer returns from ``train()``.

    Whatever ``train()`` returns is splatted straight into the job's terminal
    ``done`` SSE event (see ``job_queue``), so these reach the task — and its log
    — with no plumbing of their own. That is the whole point: a run that stopped
    at epoch 14 of 50 used to produce a final log line identical to one that ran
    all 50, and the reason existed only in the training server's own process log,
    which the UI never shows. "Finished" and "gave up early" are the two facts a
    user most needs from a finished run, and they were indistinguishable.

    Written through one helper so a fourth trainer cannot report the same thing
    under a different key — the failure mode `image_manifest_fields` above exists
    to prevent, and the one NOTES.md records for the patience keys.
    """
    return {
        "stop_reason": reason,
        "stop_detail": detail,
        "epochs_run": int(epochs_run),
        "max_epochs": int(max_epochs),
        # Omitted rather than sent as 0 when unknown: a reader can then tell
        # "this trainer does not track a best epoch" from "the first one was best".
        **({"best_epoch": int(best_epoch)} if best_epoch is not None else {}),
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
    def warn_if_cpu(logger: logging.Logger, progress_cb: Callable[[dict], None] | None = None) -> None:
        """Loudly flag a CPU-fallback run.

        No GPU visible to this process is usually not a deliberate choice — it
        is the container having quietly lost GPU access (a host driver update,
        a passthrough change) — and CPU training is 10-50x slower with
        otherwise no signal anything is wrong: the job just looks unusually
        slow rather than broken.

        Call once near the top of ``train()``, before any CUDA-dependent setup.
        Sets ``device_warning`` (in addition to ``log``) in the progress event
        so the task can persist it as a standing status message rather than
        letting it scroll out of the log — see
        ``_training_job_runner.py``'s ``_handle_event``.
        """
        import torch

        if torch.cuda.is_available():
            return
        msg = (
            "WARNING: no GPU detected — training will run on CPU, roughly 10-50x "
            "slower than GPU. If this is unexpected, check that the container "
            "still has GPU access (a host driver update or a GPU passthrough "
            "change can silently drop it)."
        )
        logger.warning(msg)
        if progress_cb is not None:
            progress_cb({"log": msg, "device_warning": msg})

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
        - A plain torch ``DataLoader`` built with ``persistent_workers=True``
          (libreyolo's default — see ``libreyolo/training/config.py``) is a third
          case, and the one that has no public shutdown at all: torch parks the
          iterator on ``loader._iterator`` and reuses it across epochs, but
          ``DataLoader`` exposes no ``close()``. Torch's own shutdown path is
          ``_MultiProcessingDataLoaderIter.__del__`` → ``_shutdown_workers()``,
          so dropping that last reference *is* the supported mechanism; there is
          simply no public name for the reference. Clearing it is therefore the
          documented "drop the last reference" rule applied to the attribute
          torch itself set, not a reach into torch's shutdown internals.

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
            # Module-local logger: this file imports `logging` but defines no
            # module-level logger, and each trainer makes its own.
            log = logging.getLogger(__name__)

            close = getattr(loader, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 — see docstring: never fail the job
                    log.exception("Failed to close a DataLoader")
                continue

            # No close() — if torch parked a persistent iterator here, dropping it
            # is the only shutdown path there is (see docstring). Guarded by the
            # `is not None` check so a persistent_workers=False loader, whose
            # _iterator is always None, is still left entirely to refcounting.
            if getattr(loader, "_iterator", None) is not None:
                try:
                    loader._iterator = None
                except Exception:  # noqa: BLE001 — see docstring: never fail the job
                    log.exception("Failed to release a persistent DataLoader iterator")

        # Backstop for loaders with no close(): frees any iterator still held by
        # a reference cycle or an exception traceback, whose __del__ is torch's
        # own supported worker shutdown.
        gc.collect()

    @classmethod
    def shutdown_dataloaders_in_traceback(cls, exc: BaseException | None) -> None:
        """Release DataLoader workers stranded by an *in-flight* exception.

        The trainers above can name their loaders because they own them. This
        exists for the case where they cannot: libreyolo builds its ``Trainer``
        as a **local** of ``model.train()`` and never hands it back (it is not
        stored on the model), so there is nothing to read the loaders off once
        ``train()`` has raised.

        That matters because the loaders are only unreachable *and* still alive
        in the failure case. On the success path ``model.train()`` returns, its
        frame dies, refcounting drops the trainer, and the workers go with it —
        which is why completed jobs cleaned up fine and only cancelled and
        errored ones leaked. Observed in production on 2026-08-19 as eight
        orphaned worker processes holding ~1.5 GB, ten minutes after a cancelled
        job.

        The exception's traceback holds every frame it passed through, including
        ``train()``'s, which makes it a reliable *route* back to the trainer. Be
        careful about the stronger claim, though: it is not necessarily the only
        thing keeping the trainer alive. Measured on 2026-08-19, clearing those
        frames with ``traceback.clear_frames`` left the libreyolo trainer, and
        its CUDA tensors, alive and unchanged — so libreyolo retains it
        somewhere else too. That is precisely why this shuts the loaders down
        *explicitly* rather than dropping references and trusting refcounting:
        the shutdown must not depend on being the last referrer.

        So this walks the traceback and shuts down every loader it can reach.
        Deliberately
        duck-typed and read-only: it identifies loaders structurally rather than
        by ``isinstance`` (torch is not imported here) and only ever reads a
        fixed set of attribute names, so an unrelated object sitting in a frame
        cannot be provoked into running code.

        Args:
            exc: The exception being handled, or None on the success path —
                 typically ``sys.exc_info()[1]`` from inside a ``finally``.
                 None is a no-op, so the call site needs no branch.
        """
        if exc is None:
            return

        log = logging.getLogger(__name__)

        def _is_loader(obj: Any) -> bool:
            # A DataLoader, structurally: torch sets both of these in __init__.
            # `_iterator` alone would also match the iterator itself.
            return hasattr(obj, "_iterator") and hasattr(obj, "num_workers")

        found: list[Any] = []
        seen: set[int] = set()

        def _consider(obj: Any) -> None:
            if obj is None or id(obj) in seen:
                return
            seen.add(id(obj))
            if _is_loader(obj):
                found.append(obj)

        try:
            tb = exc.__traceback__
            while tb is not None:
                for value in list(tb.tb_frame.f_locals.values()):
                    _consider(value)
                    # Loaders usually hang off the trainer rather than sitting
                    # in a frame directly. These are the names libreyolo and
                    # ultralytics both use.
                    for attr in ("train_loader", "val_loader", "test_loader", "dataloader"):
                        try:
                            _consider(getattr(value, attr, None))
                        except Exception:  # noqa: BLE001 — a hostile __getattr__ is not our problem
                            continue
                tb = tb.tb_next
        except Exception:  # noqa: BLE001 — cleanup must never mask the real failure
            log.exception("Failed to walk the traceback for stranded DataLoaders")

        if found:
            log.info("Releasing %d DataLoader(s) stranded by a failed training run", len(found))
        cls.shutdown_dataloaders(*found)

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
