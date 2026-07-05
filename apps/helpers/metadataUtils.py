"""Utility functions for models and json metadata"""

import ast
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import onnx

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

DEFAULT_MODEL_DEVICE = "auto"
VALID_MODEL_DEVICES = ("auto", "gpu", "cpu")


def validate_bare_model_name(model_name: str) -> None:
    """model_name is joined onto a models-root path to build .onnx/.json paths in both
    the upload endpoint (inference_server.manager.import_model_upload) and
    ImportModelTask — reject anything that could escape that root via separators or
    '..' traversal. This is the only place this guard should live."""
    if (
        not model_name
        or "/" in model_name
        or "\\" in model_name
        or model_name in (".", "..")
        or Path(model_name).is_absolute()
    ):
        raise ValueError(f"Invalid model_name {model_name!r}: must be a bare model name (no path separators or '..')")


def validate_onnx_bytes(onnx_bytes: bytes) -> None:
    """Parse `onnx_bytes` in memory to confirm it's a well-formed ONNX model before any
    caller writes it to disk. Raises ValueError on garbage/corrupt input — callers must
    do this check before touching the filesystem so a bad upload/download can't clobber
    an existing good model or leave an unparseable orphan file behind."""
    try:
        onnx.load_model_from_string(onnx_bytes)
    except Exception as e:
        raise ValueError(f"Not a valid ONNX model: {e}") from e


def providers_for_device(device: str) -> List[str]:
    """Map a model's device preference to an ordered list of ONNX execution providers.

    `auto` and `gpu` both prefer CUDA with a CPU fallback (so a GPU-less box still
    runs); `cpu` forces CPU-only. Unknown values fall back to `auto`'s behavior.
    """
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device not in VALID_MODEL_DEVICES:
        logger.warning(f"Unknown model device {device!r}; falling back to 'auto'")
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def get_model_metadata(model_path: str, create_json_if_missing: bool = True) -> Dict[str, Any]:
    create_json = create_json_if_missing
    try:
        metadata = get_model_json_metadata(model_path)
        create_json = False
    except FileNotFoundError:
        metadata = {}
        onnx_metadata = get_onnx_metadata(model_path)
        description = onnx_metadata.get("description", "")
        # Case insensitive check for Yolov7/Yolov8/YOLO11 in description (real
        # ultralytics exports embed e.g. "Ultralytics YOLOv8x model trained on
        # coco.yaml" / "Ultralytics YOLO11s model trained on ..."). YOLO11 uses
        # the same Detect head and ONNX output layout as YOLOv8 — both are
        # handled by ultralytics_yolov8 (see resolve_legacy_model_family) — so
        # it's mapped to yolo version 8 rather than a separate number.
        yoloVer = 5
        description_lower = description.lower()
        if "yolov7" in description_lower:
            yoloVer = 7
        elif "yolov8" in description_lower or "yolo11" in description_lower or "yolov11" in description_lower:
            yoloVer = 8
        else:
            logger.warning(f"onnx metadata not explicit. Assuming YoloV5 for [{model_path}]")
        metadata["yolo"] = yoloVer

        classes: Dict[int, str] = onnx_metadata.get("names", {})
        if isinstance(classes, str):
            try:
                classes = ast.literal_eval(classes)
            except Exception as e:
                raise ValueError(f"Failed to parse 'names' metadata: {e}")

        class_list = [name for _, name in sorted(classes.items())]
        metadata["classes"] = class_list

    if create_json and metadata:
        save_model_json_metadata(model_path, metadata)

    return metadata


def resolve_legacy_model_family(metadata: Dict[str, Any]) -> str:
    """Map a sidecar with no `model_family` key to one, using the legacy `yolo` int.

    This is the only place this legacy-mapping heuristic should live.
    """
    yolo_version = metadata.get("yolo", 8)
    if yolo_version == 8:
        return "ultralytics_yolov8"
    if yolo_version == 9:
        return "libreyolo9"
    if yolo_version in (5, 7):
        # No provider is registered for this family (see
        # INFERENCE_PROVIDER_REGISTRY_PLAN.md open question) — preserves
        # today's behavior of these sidecars being unloadable, just via the
        # "unhandled family" path instead of a bare version check.
        return "ultralytics_yolov5"
    raise Exception(f"Unhandled YOLO version: {yolo_version}")


def get_onnx_metadata(model_path: str) -> Dict[str, Any]:
    model = onnx.load(model_path)
    metadata_dict = {prop.key: prop.value for prop in model.metadata_props}
    return metadata_dict


def get_model_json_metadata(model_path: str) -> Dict[str, Any]:
    json_path = Path(model_path).with_suffix(".json")
    with open(json_path, "r") as f:
        metadata = json.load(f)
    return metadata


def save_model_json_metadata(model_path: str, metadata: Dict[str, Any]) -> None:
    json_path = Path(model_path).with_suffix(".json")
    with open(json_path, "w") as f:
        f.write(json.dumps(metadata, indent=2))
