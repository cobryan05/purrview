"""Model-conversion trainer: turns an uploaded PyTorch ``.pt`` checkpoint into ONNX.

Unlike the other trainers, this one does not train anything — it inspects an
arbitrary user-supplied checkpoint, figures out which model family produced it,
and exports it to ONNX (writing the same kind of sidecar JSON that
``object_detection_trainer`` writes, so the inference server / UI can load the
result the same way).

Two conversion paths are attempted, in order:

1. **libreyolo** — the checkpoint is loaded with
   ``libreyolo.models.load_untrusted_torch_file`` (safe loader) and probed
   against every concrete ``libreyolo.models.BaseModel`` subclass via its
   ``can_load(weights_dict)`` classmethod. The first match is instantiated and
   exported.
2. **ultralytics fallback** — if no libreyolo family recognizes the checkpoint,
   we try loading it directly with ``ultralytics.YOLO`` (covers plain
   YOLOv5/v8-style ``.pt`` files that libreyolo doesn't claim) and export via
   its own ``.export(format="onnx")``.

``model_family`` sidecar mapping: only ``"libreyolo9"`` and
``"ultralytics_yolov8"`` have registered inference providers today (see
``apps/helpers/yoloUtils.py`` + ``apps/helpers/metadataUtils.py::resolve_legacy_model_family``).
A libreyolo9 match is written as ``"libreyolo9"``. Any other libreyolo family
(RT-DETR, YOLO-NAS, YOLOX, ...) is written as ``f"libreyolo_{FAMILY}"`` — there
is no inference provider registered for these yet, so the exported ONNX won't
be loadable by the inference server today, but the conversion job itself still
succeeds and lands a valid ONNX + sidecar on disk. This is intentional per the
Chunk B spec; wiring up more providers is future work.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from apps.helpers.metadataUtils import save_model_json_metadata

from . import register_trainer
from .base import Trainer

logger = logging.getLogger(__name__)


def _libreyolo_class_names(weights: dict) -> list[str]:
    """Best-effort extraction of an ordered class-name list from a checkpoint's
    ``names`` metadata (dict[int, str] or list[str]). Returns [] if absent/unparsable."""
    names = weights.get("names") if isinstance(weights, dict) else None
    if names is None:
        return []
    try:
        if isinstance(names, dict):
            return [str(name) for _, name in sorted(names.items(), key=lambda kv: int(kv[0]))]
        if isinstance(names, (list, tuple)):
            return [str(name) for name in names]
    except Exception:
        logger.warning("Could not parse checkpoint 'names' metadata; falling back to [].", exc_info=True)
    return []


def _find_libreyolo_class(weights: dict):
    """Return the first concrete libreyolo BaseModel subclass whose can_load()
    accepts *weights*, or None if none match."""
    import libreyolo.models as lm

    for name in dir(lm):
        obj = getattr(lm, name)
        if not (isinstance(obj, type) and issubclass(obj, lm.BaseModel) and obj is not lm.BaseModel):
            continue
        can_load = getattr(obj, "can_load", None)
        if can_load is None:
            continue
        try:
            if can_load(weights):
                return obj
        except Exception:
            logger.warning("can_load() raised for candidate class %s; skipping.", name, exc_info=True)
    return None


@register_trainer("convert_model")
class ModelConverter(Trainer):
    async def train(
        self,
        config: dict[str, Any],
        progress_cb: Callable[[dict], None],
        cancel_event: asyncio.Event,
    ) -> dict[str, Any]:
        source_path = config.get("source_model_path")
        model_name: str = str(config.get("model_name", "converted_model"))
        artifact_dir_str = config.get("artifact_dir", "")

        if not source_path:
            raise ValueError("convert_model trainer requires 'source_model_path' in config")
        if not Path(source_path).is_file():
            raise ValueError(f"source_model_path does not exist or is not a file: {source_path!r}")
        if not artifact_dir_str:
            raise ValueError("convert_model trainer requires 'artifact_dir'")

        artifact_path = Path(artifact_dir_str)
        artifact_path.mkdir(parents=True, exist_ok=True)

        progress_cb({"progress": 0.05, "log": "Loading source model..."})

        import libreyolo.models as lm

        try:
            weights = await asyncio.to_thread(lm.load_untrusted_torch_file, source_path)
        except Exception as e:
            raise ValueError(
                f"Unsupported model checkpoint: could not load {source_path!r} as a PyTorch " f"checkpoint ({e})."
            ) from e

        if cancel_event.is_set():
            return {}

        onnx_path = artifact_path / f"{model_name}.onnx"
        model_family: str
        class_list: list[str]

        libreyolo_cls = await asyncio.to_thread(_find_libreyolo_class, weights)

        if libreyolo_cls is not None:
            progress_cb({"progress": 0.2, "log": f"Detected libreyolo family: {libreyolo_cls.FAMILY}"})

            size = libreyolo_cls.detect_size(weights)
            nb_classes = libreyolo_cls.detect_nb_classes(weights)
            task = libreyolo_cls.detect_checkpoint_task(weights)

            kwargs: dict[str, Any] = {"device": "cpu"}
            if size is not None:
                kwargs["size"] = size
            if nb_classes is not None:
                kwargs["nb_classes"] = nb_classes
            if task is not None:
                kwargs["task"] = task

            def _build_and_export():
                model = libreyolo_cls(source_path, **kwargs)
                return model.export("onnx", output_path=str(onnx_path))

            if cancel_event.is_set():
                return {}

            progress_cb({"progress": 0.5, "log": "Exporting to ONNX..."})
            exported_path = await asyncio.to_thread(_build_and_export)
            if str(exported_path) != str(onnx_path):
                shutil.copy2(exported_path, onnx_path)

            family = str(libreyolo_cls.FAMILY)
            model_family = "libreyolo9" if family == "yolo9" else f"libreyolo_{family}"
            class_list = _libreyolo_class_names(weights)
            if not class_list and nb_classes:
                class_list = [str(i) for i in range(nb_classes)]
        else:
            progress_cb({"progress": 0.2, "log": "No libreyolo family matched; trying ultralytics fallback..."})
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise ValueError(
                    "Unsupported model checkpoint: no libreyolo family could load it, and "
                    "ultralytics is not installed; cannot convert this checkpoint."
                ) from e

            if cancel_event.is_set():
                return {}

            def _ultralytics_export():
                m = YOLO(source_path)
                exported = m.export(format="onnx")
                return exported, dict(m.names or {})

            progress_cb({"progress": 0.5, "log": "Exporting to ONNX via ultralytics..."})
            try:
                exported_path, names = await asyncio.to_thread(_ultralytics_export)
            except Exception as e:
                raise ValueError(
                    "Unsupported model checkpoint: no libreyolo family matched, and ultralytics "
                    f"could not load it either ({e})."
                ) from e
            if str(exported_path) != str(onnx_path):
                shutil.copy2(exported_path, onnx_path)

            model_family = "ultralytics_yolov8"
            class_list = [str(name) for _, name in sorted(names.items(), key=lambda kv: int(kv[0]))]

        sidecar = {
            "model_family": model_family,
            "classes": class_list,
            "class_map": {},
            "device": "auto",
        }
        save_model_json_metadata(str(onnx_path), sidecar)

        progress_cb({"progress": 1.0, "log": "Conversion complete"})

        return {
            "model_path": str(onnx_path),
            "model_name": model_name,
            "classes": class_list,
        }
