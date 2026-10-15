"""Object detection trainer using libreyolo.

Trains YOLO-family object detection models (YOLOv9 via LibreYOLO9) and exports
to ONNX for serving by the inference server.

Supported architectures:
    yolo_v9_c  → LibreYOLO9 size='c'
    yolo_v9_e  → LibreYOLO9 size='c'  (no distinct 'e' size in libreyolo; uses largest)

Adding new architectures: extend ``_ARCH_MAP`` with a ``(ModelClass, size)`` entry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ..dataset import build_yolo_detection_dataset
from . import register_trainer
from .base import STOP_COMPLETED, STOP_EARLY, Trainer, device_kwarg, image_manifest_fields, stop_summary

logger = logging.getLogger(__name__)

# Maps DB architecture name → (libreyolo class name, size string).
# Only the class *name* is stored here; the class is imported lazily inside
# train() so that libreyolo is an optional dependency at import time.
_ARCH_MAP: dict[str, tuple[str, str]] = {
    "yolo_v9_c": ("LibreYOLO9", "c"),
    "yolo_v9_e": ("LibreYOLO9", "c"),  # libreyolo has no distinct 'e' size
    "yolo_v9_t": ("LibreYOLO9", "t"),
    "yolo_v9_s": ("LibreYOLO9", "s"),
    "yolo_v9_m": ("LibreYOLO9", "m"),
}


def _get_model_class(class_name: str):
    import libreyolo

    cls = getattr(libreyolo, class_name, None)
    if cls is None:
        raise ImportError(f"libreyolo has no class {class_name!r}. Install/upgrade libreyolo.")
    return cls


@register_trainer("object_detection")
class ObjectDetectionTrainer(Trainer):
    # NOTE: this trainer reads "patience", unlike the classifier trainers
    # which read "early_stopping_patience" — see NOTES.md. Keep this key
    # name here as the source of truth for what this trainer accepts.
    HYPERPARAM_DEFAULTS: dict[str, int] = {
        "max_epochs": 50,
        "imgsz": 640,
        "batch_size": 16,
        "patience": 20,
    }

    async def train(
        self,
        config: dict[str, Any],
        progress_cb: Callable[[dict], None],
        cancel_event: asyncio.Event,
    ) -> dict[str, Any]:
        d = self.HYPERPARAM_DEFAULTS
        architecture: str = config.get("architecture", "yolo_v9_c")
        class_names: list[str] = list(config.get("class_names") or [])
        annotations: dict = dict(config.get("annotations") or {})
        model_name: str = str(config.get("model_name", "model"))
        cache_dir: str = str(config.get("cache_dir", ""))
        artifact_dir: str = str(config.get("artifact_dir", ""))
        manifest: list[dict] = list(config.get("manifest") or [])

        role: str = str(config.get("role", ""))
        model_config_uuid: str = str(config.get("model_config_uuid", ""))
        model_config_node_uuid: str = str(config.get("model_config_node_uuid", ""))
        label_uuids: list[str] = list(config.get("label_uuids") or [])

        max_epochs: int = int(config.get("max_epochs", d["max_epochs"]))
        imgsz: int = int(config.get("imgsz", d["imgsz"]))
        batch_size: int = int(config.get("batch_size", d["batch_size"]))
        patience: int = int(config.get("patience", d["patience"]))
        self.warn_if_cpu(logger, progress_cb)

        if not class_names:
            raise ValueError("object_detection trainer requires at least one class in class_names")
        if not manifest:
            raise ValueError("object_detection trainer received an empty manifest")
        if not cache_dir:
            raise ValueError("object_detection trainer requires cache_dir")
        if not artifact_dir:
            raise ValueError("object_detection trainer requires artifact_dir")

        if architecture not in _ARCH_MAP:
            raise ValueError(
                f"Unsupported architecture {architecture!r} for object_detection trainer. "
                f"Supported: {sorted(_ARCH_MAP)}. "
                f"To add more, extend _ARCH_MAP in object_detection_trainer.py."
            )
        model_class_name, size = _ARCH_MAP[architecture]

        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Build YOLO dataset layout from cache.
            dataset_info = build_yolo_detection_dataset(
                manifest=manifest,
                annotations=annotations,
                class_names=class_names,
                cache_dir=cache_dir,
                out_dir=tmp_dir,
            )
            data_yaml = dataset_info["data_yaml"]
            train_uuids = dataset_info["train_uuids"]
            val_uuids = dataset_info["val_uuids"]
            logger.info(
                "Detection dataset: %d train / %d val images, %d classes",
                len(train_uuids),
                len(val_uuids),
                len(class_names),
            )

            # libreyolo owns the patience logic, so the only evidence of an early
            # stop here is which epoch the callback last saw: a run that used its
            # whole budget ends on `max_epochs`, one that gave up ends short.
            # Tracked in a dict rather than via `nonlocal` — the callback is defined
            # far from the epilogue that reads these, and a plain name would be easy
            # to shadow in between.
            observed = {"last_epoch": 0, "best_epoch": None, "best_map50": None}

            # Epoch callback — wired via libreyolo's callback parameter.
            # Receives TrainEpochEvent dataclasses.
            def _epoch_cb(event) -> None:
                epoch: int = event.epoch
                total: int = event.total_epochs
                loss: float = float(event.train_loss)
                # libreyolo validates every `eval_interval` epochs (10 by
                # default), so most epochs carry no mAP50 at all. Reporting the
                # missing measurement as 0.0 drew a chart that was mostly zeros
                # with a spike wherever validation actually ran — omit the key
                # instead and let the consumer show a gap. val_loss is likewise
                # omitted: libreyolo doesn't expose it separately.
                map50_raw = event.val_metrics.get("metrics/mAP50")
                map50: Optional[float] = float(map50_raw) if isinstance(map50_raw, (int, float)) else None
                payload: dict[str, Any] = {
                    "progress": epoch / max(total, 1),
                    "epoch": epoch,
                    "total_epochs": total,
                    "train_loss": round(loss, 4),
                    "log": (
                        f"Epoch {epoch}/{total}  loss={loss:.4f}  "
                        f"mAP50={f'{map50:.4f}' if map50 is not None else 'n/a'}"
                    ),
                }
                if map50 is not None:
                    payload["val_accuracy"] = round(map50, 4)
                progress_cb(payload)
                observed["last_epoch"] = epoch
                # Only validated epochs carry a mAP50 at all (see above), so the
                # "best epoch" this reports is the best *measured* one.
                if map50 is not None and (observed["best_map50"] is None or map50 > observed["best_map50"]):
                    observed["best_map50"] = map50
                    observed["best_epoch"] = epoch
                if cancel_event.is_set():
                    raise RuntimeError("Training cancelled by user")

            def _on_train_start(event) -> None:
                # Emitted once, before the first epoch: on slow (e.g. CPU)
                # hardware an epoch can take minutes, and with no event between
                # "job started" and "epoch 1 finished" the job looks stuck
                # rather than merely slow. libreyolo has no per-epoch
                # "epoch N starting" hook (only on_train_epoch_end), so this
                # is the earliest signal available.
                total = event.total_epochs or max_epochs
                progress_cb(
                    {
                        "progress": 0.0,
                        "epoch": event.start_epoch,
                        "total_epochs": total,
                        "log": f"Training started — epoch {event.start_epoch}/{total} running...",
                    }
                )

            # Object exposing on_train_start, dispatched alongside the plain
            # _epoch_cb callable (which libreyolo routes to on_train_epoch_end
            # only) — see libreyolo.training.callbacks.TrainCallbackList.
            start_announcer = types.SimpleNamespace(on_train_start=_on_train_start)

            # Load model (downloads pretrained weights from HuggingFace on first use).
            # Omit nb_classes so it defaults to the checkpoint's nc (80 for COCO);
            # model.train() adapts the head to the dataset's class count automatically.
            model_cls = _get_model_class(model_class_name)
            weight_filename = f"{model_cls.FILENAME_PREFIX}{size}{model_cls.WEIGHT_EXT}"
            logger.info("Loading %s (size=%s) from %s", model_class_name, size, weight_filename)

            model = model_cls(weight_filename, size=size)

            try:
                # Run training in a thread (libreyolo train() is synchronous).
                logger.info("Starting training: arch=%s", architecture)
                self.log_effective_hyperparams(
                    logger,
                    progress_cb,
                    architecture=architecture,
                    epochs=max_epochs,
                    imgsz=imgsz,
                    batch=batch_size,
                    patience=patience,
                    device=device_kwarg(config).get("device", "auto"),
                )
                results: dict = await asyncio.to_thread(
                    model.train,
                    data_yaml,
                    epochs=max_epochs,
                    imgsz=imgsz,
                    batch=batch_size,
                    patience=patience,
                    project=str(artifact_path / "runs"),
                    name=model_name,
                    exist_ok=True,
                    callbacks=[_epoch_cb, start_announcer],
                    **device_kwarg(config),
                )

                best_map50: float = float(results.get("best_mAP50", 0.0))
                logger.info("Training complete. best_mAP50=%.4f", best_map50)

                epochs_run = observed["last_epoch"] or max_epochs
                if epochs_run < max_epochs:
                    stop_reason = STOP_EARLY
                    stop_detail = (
                        f"early stopping at epoch {epochs_run} of {max_epochs}: mAP50 did not improve "
                        f"for {patience} epoch(s) (patience={patience})"
                    )
                else:
                    stop_reason = STOP_COMPLETED
                    stop_detail = f"ran the full {max_epochs} epoch(s)"
                if observed["best_epoch"] is not None:
                    stop_detail += f"; best mAP50 {best_map50:.4f} at epoch {observed['best_epoch']}"
                # Through progress_cb, not just logger.info: the latter reaches only
                # this server's process log, which the task UI never shows.
                progress_cb({"log": stop_detail})

                # Export trained model to ONNX.
                onnx_dest = str(artifact_path / f"{model_name}.onnx")
                logger.info("Exporting ONNX → %s", onnx_dest)
                exported_path: str = await asyncio.to_thread(
                    model.export,
                    "onnx",
                    output_path=onnx_dest,
                )
                # If export wrote to a different path (shouldn't happen), copy it.
                if str(exported_path) != onnx_dest:
                    shutil.copy2(exported_path, onnx_dest)
            finally:
                # libreyolo builds its Trainer as a local of model.train() and
                # never stores it on the model, so there is nothing here to read
                # the loaders off — and its loaders default to
                # persistent_workers=True, whose workers live as long as the
                # loader does. When train() returns normally that is fine: the
                # frame dies and refcounting takes the workers with it. When it
                # *raises* — a cancel from _epoch_cb, or a real training error —
                # the traceback pins that frame, and in a server that outlives
                # the job the workers are stranded for good. The traceback is
                # also the only route back to them.
                self.shutdown_dataloaders_in_traceback(sys.exc_info()[1])
                del model
                self.release_gpu_memory()

            # Write sidecar JSON understood by load_yolo_onnx.
            sidecar = {
                "model_kind": "object_detection",
                "trainer": "object_detection",
                "architecture": architecture,
                "classes": class_names,
                "yolo": 9,
                "model_family": "libreyolo9",
                "val_accuracy": best_map50,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                **image_manifest_fields(config, train_uuids, val_uuids),
                "role": role,
                "model_config_uuid": model_config_uuid,
                "model_config_node_uuid": model_config_node_uuid,
                "label_uuids": label_uuids,
            }
            sidecar_path = artifact_path / f"{model_name}.json"
            sidecar_path.write_text(json.dumps(sidecar, indent=2))
            logger.info("Sidecar written → %s", sidecar_path)

        return {
            "model_path": onnx_dest,
            "model_name": model_name,
            "val_accuracy": best_map50,
            **stop_summary(
                reason=stop_reason,
                detail=stop_detail,
                epochs_run=epochs_run,
                max_epochs=max_epochs,
                best_epoch=observed["best_epoch"],
            ),
        }
