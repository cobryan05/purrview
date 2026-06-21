"""Shared preprocessing utilities for ONNX classifier inference.

Used by both RunClassifierTask (batch) and the inference_server classify
endpoint (single-image interactive). Keeps preprocessing logic in one place.
"""

from __future__ import annotations

import cv2
import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along *axis*."""
    e = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def preprocess_image_bytes(img_bytes: bytes, input_size: tuple[int, int]) -> np.ndarray | None:
    """Decode raw image bytes → normalised (3, H, W) float32 tensor, or None on failure."""
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return preprocess_bgr_array(img_bgr, input_size)


def preprocess_bgr_array(img_bgr: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    """Normalise a BGR numpy array → (3, H, W) float32 tensor.

    *input_size* is passed directly to cv2.resize, i.e. (width, height).
    Square inputs (224, 224) are the norm so order doesn't matter in practice.
    """
    img = cv2.resize(img_bgr, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return img.transpose(2, 0, 1)  # HWC → CHW
