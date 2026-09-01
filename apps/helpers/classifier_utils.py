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


#: Fraction of a box's size added as context on each side before classifying it.
#: Written into the crop classifier's sidecar at train time and read back at
#: inference time, so a model trained with one padding is never served with another.
DEFAULT_CROP_PADDING = 0.1


def crop_region(
    img_bgr: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    padding: float = DEFAULT_CROP_PADDING,
) -> np.ndarray:
    """Cut a *relative* (x, y, width, height) box out of a BGR array, padded by
    ``padding`` of the box size on each side and clamped to the image.

    Shared by the crop-classifier trainer and every caller that runs one, because
    train/serve crop geometry that drifts apart degrades accuracy silently — there
    is no error, the model just sees a differently-framed animal than it learned on.

    Always returns at least a 1x1 region: a box that clamps to nothing would
    otherwise reach cv2.resize as an empty array and raise from inside a
    DataLoader worker.
    """
    img_h, img_w = img_bgr.shape[:2]
    pad_x = width * padding
    pad_y = height * padding

    x1 = int(round(max(0.0, x - pad_x) * img_w))
    y1 = int(round(max(0.0, y - pad_y) * img_h))
    x2 = int(round(min(1.0, x + width + pad_x) * img_w))
    y2 = int(round(min(1.0, y + height + pad_y) * img_h))

    x1 = min(x1, max(0, img_w - 1))
    y1 = min(y1, max(0, img_h - 1))
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    return img_bgr[y1:y2, x1:x2]


def preprocess_crop(
    img_bgr: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    input_size: tuple[int, int],
    padding: float = DEFAULT_CROP_PADDING,
) -> np.ndarray:
    """``crop_region`` + ``preprocess_bgr_array`` — one box → one (3, H, W) tensor."""
    return preprocess_bgr_array(crop_region(img_bgr, x, y, width, height, padding), input_size)


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
