"""Curated list of pretrained base models offered for cold-start import.

Two kinds of entries, both surfaced in the same Curated tab (see
`ui/models.js::buildCuratedForm`):

1. **Direct ONNX** (no `requires_conversion` key, or `requires_conversion:
   False`) — a maintainer-verified, directly-usable `.onnx` export. Verified
   via HTTP HEAD (200 OK, real byte content) AND an actual layout check
   (loaded through onnxruntime / `apps.helpers.yoloUtils.load_yolo_onnx`) that
   the export's tensor shape matches what the named `model_family`'s inference
   provider decodes. Do NOT add a generic/third-party ONNX export here without
   that layout verification — a plausible-looking export with a subtly
   different channel order/activation silently produces garbage detections
   through the wrong provider, with no exception raised.

   - The four `yolo9_*` entries point at the LibreYOLO project's own Hugging
     Face model repo (https://huggingface.co/LibreYOLO/libreyolo-web), which
     publishes ready-to-run `.onnx` exports using the same
     `model.export("onnx")` path the training server's
     ObjectDetectionTrainer uses after fine-tuning (see
     object_detection_trainer.py) — output layout matches
     LibreYolo9InferenceProvider (boxes already xyxy in canvas-pixel space,
     scores already sigmoid-activated, no objectness channel). Do not
     substitute generic/third-party YOLOv9 ONNX exports (e.g. from the
     original WongKinYiu/yolov9 repo) here — their layout differs.
   - `yolo11n-coco` points at Ultralytics' own GitHub release asset
     (`ultralytics/assets` release `v8.3.0`), which is the ONE Ultralytics
     detection family with a directly-published `.onnx` (confirmed: v8/v9/v10/
     v12/YOLO26 have no `.onnx` asset at that release, only `.pt` — see the
     `requires_conversion` entries below for those). Verified end-to-end: HTTP
     HEAD 200, real ONNX bytes, output tensor shape `(batch, 84, anchors)` —
     the classic wide multi-class-score layout (`nms=False` in the embedded
     export args) — matching `ultralytics_detect`'s decode
     (`UltralyticsYoloV8InferenceProvider`), and successfully loaded end-to-end
     through `load_yolo_onnx` + `processImage`.

2. **Needs conversion** (`requires_conversion: True`) — a public `.pt` URL
   (Ultralytics' own GitHub release assets) with no directly-usable ONNX
   export. The UI shows a "via training server" badge and disables the
   download button when the training server is unreachable (see
   `ImportCuratedModelTask` / `ui/models.js`). These entries carry no
   `model_family` — the training server's `convert_model` trainer determines
   it from the ACTUAL exported ONNX output shape (see
   `model_converter.py::_detect_ultralytics_model_family`), not from this
   catalogue, so a wrong guess here can't corrupt the sidecar.

   All three `.pt` URLs below were verified reachable (HTTP HEAD/GET 200,
   redirecting to a real release asset) at
   `https://github.com/ultralytics/assets/releases/download/v8.3.0/`.
"""

from typing import Any, Dict, List

_ULTRALYTICS_RELEASE_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"

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
    {
        "key": "yolo11n-coco",
        "display_name": "YOLO11 Nano (COCO, 80 classes, fastest Ultralytics)",
        "onnx_url": f"{_ULTRALYTICS_RELEASE_BASE}/yolo11n.onnx",
        "model_family": "ultralytics_detect",
    },
    # ── Needs conversion: public .pt only, no maintainer-published .onnx ──
    {
        "key": "yolov8n-coco-pt",
        "display_name": "YOLOv8 Nano (COCO, 80 classes) — via training server",
        "pt_url": f"{_ULTRALYTICS_RELEASE_BASE}/yolov8n.pt",
        "requires_conversion": True,
    },
    {
        "key": "yolo11n-coco-pt",
        "display_name": "YOLO11 Nano (COCO, 80 classes) — via training server",
        "pt_url": f"{_ULTRALYTICS_RELEASE_BASE}/yolo11n.pt",
        "requires_conversion": True,
    },
    {
        "key": "yolov10n-coco-pt",
        "display_name": "YOLOv10 Nano (COCO, 80 classes, NMS-free) — via training server",
        "pt_url": f"{_ULTRALYTICS_RELEASE_BASE}/yolov10n.pt",
        "requires_conversion": True,
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
