from typing import Tuple

import numpy as np

from .onnxYoloInferenceProvider import OnnxYoloInferenceProvider
from .Registry import register_inference_provider


@register_inference_provider("ultralytics_yolov8")
@register_inference_provider("ultralytics_detect")
class UltralyticsYoloV8InferenceProvider(OnnxYoloInferenceProvider):
    """Ultralytics YOLOv8/YOLO11 exports: boxes are cxcywh, scores are already
    sigmoid-activated by the exported graph (Ultralytics' Detect head applies
    `cls.sigmoid()` before returning in export mode — confirmed empirically:
    raw class-score channels of yolov8x.onnx/Yolo11S.onnx are already in
    [0, 1]). (1, 4+nc, num_preds) transposed to (1, num_preds, 4+nc) by the
    caller — 4 box channels + nc class-score channels, no separate objectness
    channel. Re-applying sigmoid here double-activates already-activated
    scores, squashing every detection toward ~0.5 confidence — that was a
    real regression (see tests/helpers/test_yolo_utils.py).

    Registered under two family keys: "ultralytics_yolov8" (legacy, kept so
    existing on-disk sidecars still resolve — do NOT remove) and
    "ultralytics_detect" (new, family-neutral name written by
    ultralytics_detection_trainer.py for v8/Ultralytics-v9/v11/v12 — they all
    share this exact tensor layout, so no new decode class is needed)."""

    def _extract(self, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split raw preds (num_preds, 4+nc) into (boxes_xyxy, scores)."""
        if preds.shape[1] < 5:
            raise ValueError("Prediction tensor must have at least 5 elements per box (x, y, w, h, class_scores...).")
        boxes = preds[:, :4].copy()
        scores = preds[:, 4:]  # already sigmoid-activated by the exported graph

        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = boxes.copy()
        boxes_xyxy[:, 0] -= boxes[:, 2] / 2  # x1 = cx - w / 2
        boxes_xyxy[:, 1] -= boxes[:, 3] / 2  # y1 = cy - h / 2
        boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes[:, 2]  # x2 = x1 + w
        boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes[:, 3]  # y2 = y1 + h

        return boxes_xyxy, scores
