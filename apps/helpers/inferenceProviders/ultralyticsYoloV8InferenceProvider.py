from typing import Tuple

import numpy as np

import apps.helpers.yoloUtils as YoloUtils

from .onnxYoloInferenceProvider import OnnxYoloInferenceProvider
from .Registry import register_inference_provider


@register_inference_provider("ultralytics_yolov8")
class UltralyticsYoloV8InferenceProvider(OnnxYoloInferenceProvider):
    """Ultralytics YOLOv8 exports: boxes are cxcywh, scores are raw logits.
    (1, 4+nc, num_preds) transposed to (1, num_preds, 4+nc) by the caller —
    4 box channels + nc class-score channels, no separate objectness channel."""

    def _extract(self, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split raw preds (num_preds, 4+nc) into (boxes_xyxy, scores)."""
        if preds.shape[1] < 5:
            raise ValueError("Prediction tensor must have at least 5 elements per box (x, y, w, h, class_scores...).")
        boxes = preds[:, :4].copy()
        scores = YoloUtils.sigmoid(preds[:, 4:])

        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = boxes.copy()
        boxes_xyxy[:, 0] -= boxes[:, 2] / 2  # x1 = cx - w / 2
        boxes_xyxy[:, 1] -= boxes[:, 3] / 2  # y1 = cy - h / 2
        boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes[:, 2]  # x2 = x1 + w
        boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes[:, 3]  # y2 = y1 + h

        return boxes_xyxy, scores
