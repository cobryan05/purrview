"""Object detection trainer using Ultralytics (YOLOv8 / Ultralytics-YOLOv9 /
YOLO11 / YOLO12 / YOLOv10 / YOLO26).

Trains YOLO-family object detection models via the ``ultralytics`` package's
``YOLO(...).train()`` API and exports to ONNX for serving by the inference
server. Every architecture in ``_ARCH_MAP`` trains through the identical
``YOLO(...).train()``/``model.export(format="onnx")`` call shape — only the
exported tensor layout (and therefore the inference-side decode/sidecar
``model_family``) differs per architecture, via ``_MODEL_FAMILY_MAP``:

- v8 / Ultralytics-v9 / v11 / v12 share the "classic" cxcywh + pre-sigmoid'd
  per-class-score layout, decoded by ``UltralyticsYoloV8InferenceProvider``
  under family ``"ultralytics_detect"``.
- v10 / YOLO26 are NMS-free ("end-to-end") exports: the graph itself already
  returns a fixed top-k list of picked ``[x1, y1, x2, y2, conf, cls]`` rows —
  confirmed empirically against real ``yolov10n.onnx``/``yolo26n.onnx``
  exports from this installed ``ultralytics`` version (see
  MORE_YOLO_PLAN.md Section 2) — decoded by
  ``UltralyticsE2eInferenceProvider`` under family ``"ultralytics_e2e"``.

Note: Ultralytics' own YOLOv9 (GELAN backbone, arch names ``yolov9t/s/m/c/e``)
is architecturally distinct from LibreYOLO9 (``yolo_v9_t/s/m/c/e``, trained by
``ObjectDetectionTrainer``) — both remain independently selectable in the DB
architecture catalogue; dispatch between the two trainers is handled by
``apps.helpers.model_type_map.resolve_trainer_type``.

Adding new architectures: extend ``_ARCH_MAP`` with the Ultralytics model
filename stem (e.g. ``"yolov8n"`` → loads ``yolov8n.pt``), and
``_MODEL_FAMILY_MAP`` with the sidecar ``model_family`` its ONNX export
requires (confirm the real exported tensor layout first — do not guess).
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
from .base import Trainer, image_manifest_fields

logger = logging.getLogger(__name__)

# Maps DB architecture name → Ultralytics model filename stem (without
# extension). Ultralytics auto-downloads the matching COCO-pretrained ``.pt``
# weights on first use (same shape as the existing LibreYOLO9 HuggingFace
# download in object_detection_trainer.py).
_ARCH_MAP: dict[str, str] = {
    "yolov8n": "yolov8n",
    "yolov8s": "yolov8s",
    "yolov8m": "yolov8m",
    "yolov8l": "yolov8l",
    "yolov8x": "yolov8x",
    "yolov9t": "yolov9t",
    "yolov9s": "yolov9s",
    "yolov9m": "yolov9m",
    "yolov9c": "yolov9c",
    "yolov9e": "yolov9e",
    "yolo11n": "yolo11n",
    "yolo11s": "yolo11s",
    "yolo11m": "yolo11m",
    "yolo11l": "yolo11l",
    "yolo11x": "yolo11x",
    "yolo12n": "yolo12n",
    "yolo12s": "yolo12s",
    "yolo12m": "yolo12m",
    "yolo12l": "yolo12l",
    "yolo12x": "yolo12x",
    # YOLOv10 (no "t" scale in this ultralytics version's cfg — only n/s/m/b/l/x).
    "yolov10n": "yolov10n",
    "yolov10s": "yolov10s",
    "yolov10m": "yolov10m",
    "yolov10b": "yolov10b",
    "yolov10l": "yolov10l",
    "yolov10x": "yolov10x",
    # YOLO26 (no "t"/"b" scale — n/s/m/l/x).
    "yolo26n": "yolo26n",
    "yolo26s": "yolo26s",
    "yolo26m": "yolo26m",
    "yolo26l": "yolo26l",
    "yolo26x": "yolo26x",
}

# Arch name -> sidecar model_family, chosen by architecture rather than
# hardcoded to one family (see module docstring). Any arch NOT listed here
# falls back to "ultralytics_detect" (the classic v8-shape layout) below.
_MODEL_FAMILY_MAP: dict[str, str] = {
    **{arch: "ultralytics_e2e" for arch in _ARCH_MAP if arch.startswith("yolov10") or arch.startswith("yolo26")},
}


def _model_family_for_arch(architecture: str) -> str:
    return _MODEL_FAMILY_MAP.get(architecture, "ultralytics_detect")


def _get_yolo_class():
    from ultralytics import YOLO

    return YOLO


@register_trainer("ultralytics_object_detection")
class UltralyticsDetectionTrainer(Trainer):
    # NOTE: this trainer reads "patience", matching object_detection_trainer —
    # NOT "early_stopping_patience" (used by the classifier trainers). See
    # NOTES.md's hyperparams_json key-name gotcha.
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
        architecture: str = config.get("architecture", "yolov8n")
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
            raise ValueError("ultralytics_object_detection trainer requires at least one class in class_names")
        if not manifest:
            raise ValueError("ultralytics_object_detection trainer received an empty manifest")
        if not cache_dir:
            raise ValueError("ultralytics_object_detection trainer requires cache_dir")
        if not artifact_dir:
            raise ValueError("ultralytics_object_detection trainer requires artifact_dir")

        if architecture not in _ARCH_MAP:
            raise ValueError(
                f"Unsupported architecture {architecture!r} for ultralytics_object_detection trainer. "
                f"Supported: {sorted(_ARCH_MAP)}. "
                f"To add more, extend _ARCH_MAP in ultralytics_detection_trainer.py."
            )
        weights_stem = _ARCH_MAP[architecture]

        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Build YOLO dataset layout from cache — same on-disk layout
            # (images/{train,val}, labels/{train,val}, data.yaml with
            # train/val/nc/names keys) that Ultralytics' YOLO(...).train(data=...)
            # expects natively, so this is reused as-is (no adapter needed).
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

            # Progress state captured by the epoch callback closure.
            progress_state: dict[str, Any] = {"total_epochs": max_epochs}

            def _on_train_epoch_end(trainer) -> None:
                epoch = int(getattr(trainer, "epoch", 0)) + 1
                total = int(progress_state["total_epochs"])
                metrics = dict(getattr(trainer, "metrics", None) or {})
                map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
                loss_items = getattr(trainer, "loss_items", None)
                try:
                    train_loss = float(loss_items.mean()) if loss_items is not None else 0.0
                except Exception:
                    train_loss = 0.0
                progress_cb(
                    {
                        "progress": epoch / max(total, 1),
                        "epoch": epoch,
                        "total_epochs": total,
                        "val_accuracy": round(map50, 4),
                        "train_loss": round(train_loss, 4),
                        "val_loss": 0.0,
                        "log": (f"Epoch {epoch}/{total}  loss={train_loss:.4f}  " f"mAP50={map50:.4f}"),
                    }
                )
                if cancel_event.is_set():
                    trainer.stop_training = True

            def _run_training() -> dict:
                YOLO = _get_yolo_class()
                model = YOLO(f"{weights_stem}.pt")
                try:
                    model.add_callback("on_train_epoch_end", _on_train_epoch_end)
                    results = model.train(
                        data=data_yaml,
                        epochs=max_epochs,
                        imgsz=imgsz,
                        batch=batch_size,
                        patience=patience,
                        project=str(artifact_path / "runs"),
                        name=model_name,
                        exist_ok=True,
                    )
                    # results.results_dict has keys like "metrics/mAP50(B)".
                    results_dict = dict(getattr(results, "results_dict", None) or {})
                    best_map50 = float(results_dict.get("metrics/mAP50(B)", 0.0))

                    # Export to ONNX. Ultralytics writes next to the best weights;
                    # move/copy it to our stable artifact path.
                    exported = model.export(format="onnx")
                    return {"exported_path": str(exported), "best_map50": best_map50}
                finally:
                    del model
                    self.release_gpu_memory()

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
            train_result: dict = await asyncio.to_thread(_run_training)

            best_map50 = train_result["best_map50"]
            logger.info("Training complete. best_mAP50=%.4f", best_map50)

            onnx_dest = str(artifact_path / f"{model_name}.onnx")
            exported_path = train_result["exported_path"]
            if str(exported_path) != onnx_dest:
                shutil.copy2(exported_path, onnx_dest)
            logger.info("Exported ONNX → %s", onnx_dest)

            # Write sidecar JSON understood by load_yolo_onnx. model_family is
            # chosen per-architecture (see _MODEL_FAMILY_MAP): "ultralytics_detect"
            # (classic cxcywh/pre-sigmoid layout, v8/Ultralytics-v9/v11/v12) is
            # registered on UltralyticsYoloV8InferenceProvider; "ultralytics_e2e"
            # (NMS-free top-k layout, v10/YOLO26) is registered on
            # UltralyticsE2eInferenceProvider.
            model_family = _model_family_for_arch(architecture)
            sidecar = {
                "model_kind": "object_detection",
                "trainer": "ultralytics_object_detection",
                "architecture": architecture,
                "classes": class_names,
                "model_family": model_family,
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
        }
