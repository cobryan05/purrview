"""Curated list of pretrained base models offered for cold-start import.

All entries point at the LibreYOLO project's own Hugging Face model repo
(https://huggingface.co/LibreYOLO/libreyolo-web), which publishes ready-to-run
`.onnx` exports directly (verified via HTTP HEAD: 200 OK, real byte content,
sizes matching the repo's file listing) — no `.pt` -> `.onnx` conversion step
is needed.

These are exported by the LibreYOLO project itself using the same
`model.export("onnx")` path the training server's ObjectDetectionTrainer uses
after fine-tuning (see object_detection_trainer.py), so their output layout
matches what LibreYolo9InferenceProvider expects (boxes already xyxy in
canvas-pixel space, scores already sigmoid-activated, no objectness channel).
Do not substitute generic/third-party YOLOv9 ONNX exports (e.g. from the
original WongKinYiu/yolov9 repo) here — their output layout differs and will
silently produce garbage detections through this provider.
"""

from typing import Any, Dict, List

CURATED_BASE_MODELS: List[Dict[str, Any]] = [
    {
        "key": "yolo9_t-coco",
        "display_name": "YOLO9 Tiny (COCO, 80 classes, fastest)",
        "onnx_url": "https://huggingface.co/LibreYOLO/libreyolo-web/resolve/main/yolo9_t.onnx",
        "model_family": "libreyolo9",
    },
    {
        "key": "yolo9_s-coco",
        "display_name": "YOLO9 Small (COCO, 80 classes, balanced)",
        "onnx_url": "https://huggingface.co/LibreYOLO/libreyolo-web/resolve/main/yolo9_s.onnx",
        "model_family": "libreyolo9",
    },
    {
        "key": "yolo9_m-coco",
        "display_name": "YOLO9 Medium (COCO, 80 classes)",
        "onnx_url": "https://huggingface.co/LibreYOLO/libreyolo-web/resolve/main/yolo9_m.onnx",
        "model_family": "libreyolo9",
    },
    {
        "key": "yolo9_c-coco",
        "display_name": "YOLO9 Compact (COCO, 80 classes, most accurate)",
        "onnx_url": "https://huggingface.co/LibreYOLO/libreyolo-web/resolve/main/yolo9_c.onnx",
        "model_family": "libreyolo9",
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
