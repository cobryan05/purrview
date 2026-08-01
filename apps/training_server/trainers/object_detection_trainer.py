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
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..dataset import build_yolo_detection_dataset
from . import register_trainer
from .base import Trainer

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

            # Epoch callback — wired via libreyolo's callback parameter.
            # Receives TrainEpochEvent dataclasses.
            def _epoch_cb(event) -> None:
                epoch: int = event.epoch
                total: int = event.total_epochs
                map50: float = float(event.val_metrics.get("metrics/mAP50", 0.0))
                loss: float = float(event.train_loss)
                progress_cb(
                    {
                        "progress": epoch / max(total, 1),
                        "epoch": epoch,
                        "total_epochs": total,
                        "val_accuracy": round(map50, 4),
                        "train_loss": round(loss, 4),
                        "val_loss": 0.0,  # libreyolo doesn't expose val_loss separately
                        "log": (f"Epoch {epoch}/{total}  loss={loss:.4f}  " f"mAP50={map50:.4f}"),
                    }
                )
                if cancel_event.is_set():
                    raise RuntimeError("Training cancelled by user")

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
                    callbacks=_epoch_cb,
                )

                best_map50: float = float(results.get("best_mAP50", 0.0))
                logger.info("Training complete. best_mAP50=%.4f", best_map50)

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
                "train_image_uuids": train_uuids,
                "val_image_uuids": val_uuids,
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
        }
