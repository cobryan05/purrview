from typing import Tuple

import numpy as np

from .onnxYoloInferenceProvider import OnnxYoloInferenceProvider
from .Registry import register_inference_provider


@register_inference_provider("ultralytics_e2e")
class UltralyticsE2eInferenceProvider(OnnxYoloInferenceProvider):
    """Ultralytics NMS-free ("end-to-end") exports: YOLOv10 and YOLO26.

    Confirmed empirically (both installed via `ultralytics` 8.4.87, exported
    fresh from `YOLO("yolov10n.yaml")` / `YOLO("yolo26n.yaml")` and run through
    onnxruntime — see MORE_YOLO_PLAN.md Section 2 research notes): both
    architectures export an identical raw output tensor of shape
    ``(1, 300, 6)`` — a fixed top-300 list of ALREADY-PICKED detections, sorted
    by confidence descending, one row per detection:

        [x1, y1, x2, y2, confidence, class_id]

    Boxes are already xyxy in canvas-pixel (letterboxed input) space,
    confidence is already sigmoid-activated (values observed in [0, 1], not
    raw logits), and there is exactly one class_id per row — no per-class
    score matrix to argmax over, and no overlapping candidates left for
    `cv2.dnn.NMSBoxes` to suppress (the graph's own top-k/dedup already did
    that). Both families share this exact layout, so one provider covers both;
    unlike `UltralyticsYoloV8InferenceProvider`'s cxcywh/logits layout, there is
    no distinct decode per architecture here.
    """

    SKIP_NMS = True

    def _extract(self, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split raw preds (num_preds, 6) into (boxes_xyxy, scores).

        `scores` here is (num_preds, 2): [confidence, class_id] — NOT a
        per-class score matrix. `decode_detections(..., skip_nms=True)` reads
        column 0 as confidence and column 1 as class_id directly, skipping the
        argmax + NMS path used by every other (non-end-to-end) family.
        """
        if preds.shape[1] < 6:
            raise ValueError(
                "Prediction tensor must have at least 6 elements per row " "(x1, y1, x2, y2, confidence, class_id)."
            )
        boxes_xyxy = preds[:, :4].copy()
        scores = preds[:, 4:6].copy()  # [confidence, class_id], already picked/sorted by the exported graph
        return boxes_xyxy, scores
