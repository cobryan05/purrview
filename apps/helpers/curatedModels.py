"""Curated list of pretrained base models offered for cold-start import.

Both entries point at Ultralytics' own GitHub release assets, which publish
ready-to-run `.onnx` exports directly (verified via the GitHub API: 200 OK,
real byte content) — no `.pt` -> `.onnx` conversion step is needed.

YOLO11's detection head is architecturally identical to YOLOv8's (anchor-free,
cxcywh boxes, raw class logits, no objectness channel — see the LibreYOLO9
gotcha in NOTES.md for why this output layout matters to the inference
provider). That's why these entries use `model_family="ultralytics_yolov8"`:
the registry key names a *resolver* (an output-decoding strategy), not a
literal architecture-version requirement. Do not "fix" this to a
`ultralytics_yolo11` family — there is no such provider, and none is needed.
"""

from typing import Any, Dict, List

CURATED_BASE_MODELS: List[Dict[str, Any]] = [
    {
        "key": "yolo11n-coco",
        "display_name": "YOLO11n (COCO, 80 classes, fastest)",
        "onnx_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx",
        "model_family": "ultralytics_yolov8",
    },
    {
        "key": "yolo11s-coco",
        "display_name": "YOLO11s (COCO, 80 classes, balanced)",
        "onnx_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.onnx",
        "model_family": "ultralytics_yolov8",
    },
]


def get_curated_model(key: str) -> Dict[str, Any]:
    """Look up a curated base model entry by its `key`.

    Raises:
        KeyError: if `key` doesn't match any curated entry.
    """
    for entry in CURATED_BASE_MODELS:
        if entry["key"] == key:
            return entry
    raise KeyError(f"Unknown curated model key: {key!r}")
