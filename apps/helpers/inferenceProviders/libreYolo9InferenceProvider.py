from typing import Tuple

import numpy as np

from .onnxYoloInferenceProvider import OnnxYoloInferenceProvider
from .Registry import register_inference_provider


@register_inference_provider("libreyolo9")
class LibreYolo9InferenceProvider(OnnxYoloInferenceProvider):
    """LibreYOLO9 exports: boxes already xyxy (canvas-pixel space), scores already
    sigmoid-activated inside the exported graph. (1, 4+nc, num_preds) transposed
    to (1, num_preds, 4+nc) by the caller — 4 box channels + nc class-score
    channels, no separate objectness channel."""

    def _extract(self, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split raw preds (num_preds, 4+nc) into (boxes_xyxy, scores)."""
        if preds.shape[1] < 5:
            raise ValueError("Prediction tensor must have at least 5 elements per box (x, y, w, h, class_scores...).")
        boxes_xyxy = preds[:, :4].copy()
        scores = preds[:, 4:]  # already sigmoid-activated
        return boxes_xyxy, scores
