"""Utility functions for YOLO models"""

import logging
import os
import sys
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import onnxruntime as ort

import numpy as np

import apps.helpers.metadataUtils as metadataUtils
from apps.helpers.inferenceProviders.inferenceProvider import InferenceProvider
from apps.helpers.inferenceProviders.Registry import inference_provider_registry

if os.getenv("GENERATING_OPENAPI_CLIENTS") != "1":
    import cv2
    import onnxruntime as ort  # noqa: F811

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_input_shape(session: "ort.InferenceSession") -> Tuple[int, int]:
    input_shape = session.get_inputs()[0].shape  # e.g. [1, 3, 640, 640]
    return int(input_shape[2]), int(input_shape[3])  # height, width


def load_yolo_onnx(model_path: str) -> InferenceProvider:
    # Importing these registers them (via @register_inference_provider) into
    # inference_provider_registry as a side effect. Deferred to call time: the
    # provider modules import this module for the shared pipeline helpers, so a
    # module-level import here would be circular.
    import apps.helpers.inferenceProviders.libreYolo9InferenceProvider  # noqa: F401
    import apps.helpers.inferenceProviders.ultralyticsYoloV8InferenceProvider  # noqa: F401

    metadata = metadataUtils.get_model_metadata(model_path, create_json_if_missing=True)
    providers = metadataUtils.providers_for_device(metadata.get("device", metadataUtils.DEFAULT_MODEL_DEVICE))

    model_kind: str = metadata.get("model_kind", "")
    if model_kind == "image_classifier":
        from apps.helpers.inferenceProviders.imageClassifierInferenceProvider import (
            ImageClassifierInferenceProvider,
        )

        label_uuid: str = metadata.get("label_uuid", "")
        class_names: list[str] = metadata.get("class_names", ["absent", "present"])
        return ImageClassifierInferenceProvider(model_path, label_uuid, class_names, providers=providers)

    if model_kind == "multi_label_classifier":
        from apps.helpers.inferenceProviders.multiLabelClassifierInferenceProvider import (
            MultiLabelClassifierInferenceProvider,
        )

        label_uuids: list[str] = metadata.get("label_uuids") or []
        return MultiLabelClassifierInferenceProvider(model_path, label_uuids, providers=providers)

    classes: list[str] = metadata.get("classes", None)
    model_family: str = metadata.get("model_family") or metadataUtils.resolve_legacy_model_family(metadata)
    provider_cls = inference_provider_registry.get(model_family)
    if provider_cls is None:
        raise Exception(f"Unhandled model family {model_family!r}. Registered: {sorted(inference_provider_registry)}")
    return provider_cls(model_path, classes, providers=providers)


def preprocess_image(image: np.ndarray, input_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Preprocess an image for YOLO model inference by resizing, padding, and normalizing.

    Args:
        image (np.ndarray): Input image as a NumPy array (HWC format, BGR color space).
        input_shape (Tuple[int, int]): Target input shape (height, width) for the model.

    Returns:
        Tuple[np.ndarray, float, Tuple[int, int]]:
            - Preprocessed image as a NumPy array (NCHW format, normalized to [0, 1]).
            - Scaling factor used to resize the image.
            - Padding added to the image (dw, dh) to maintain aspect ratio.
    """
    # Original image dimensions
    h0, w0 = image.shape[:2]
    input_h, input_w = input_shape

    # Compute the scaling factor to fit the image within the target dimensions
    r = min(input_w / w0, input_h / h0)  # Scale factor to maintain aspect ratio
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))  # New dimensions after scaling

    # Compute padding to center the image within the target dimensions
    dw = (input_w - new_unpad[0]) / 2  # Padding width (left and right)
    dh = (input_h - new_unpad[1]) / 2  # Padding height (top and bottom)

    # Resize the image to the new dimensions
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add padding to the resized image
    padded = cv2.copyMakeBorder(
        resized,
        top=int(dh),
        bottom=int(dh + 0.1),  # Add a small offset to handle rounding issues
        left=int(dw),
        right=int(dw + 0.1),  # Add a small offset to handle rounding issues
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),  # Use a gray color for padding
    )

    # Normalize the image to [0, 1] and convert to NCHW format
    img = padded.astype(np.float32) / 255.0  # Normalize pixel values
    img = np.transpose(img, (2, 0, 1))  # Convert from HWC to CHW format
    img = np.expand_dims(img, 0)  # Add batch dimension (NCHW format)

    # Return the preprocessed image, scaling factor, and padding
    return img, r, (dw, dh)


def decode_detections(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    scale: float,
    pad: Tuple[int, int],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Shared NMS + rescale/clip tail for YOLO-family raw outputs.

    Family-agnostic: takes already-decoded xyxy boxes (canvas-pixel space, pre-NMS)
    and already-activated per-class scores (no objectness column). Each
    InferenceProvider is responsible for extracting `boxes_xyxy`/`scores` from its
    own raw tensor layout (box encoding, logit vs. sigmoid, etc.) before calling
    this.

    Args:
        boxes_xyxy (np.ndarray): Boxes as (num_preds, 4) in [x1, y1, x2, y2], canvas-pixel space.
        scores (np.ndarray): Per-class scores, shape (num_preds, nc), already activated (no raw logits).
        conf_thresh (float): Confidence threshold for filtering detections.
        iou_thresh (float): IoU threshold for NMS.
        scale (float): Scaling factor used during preprocessing.
        pad (Tuple[int, int]): Padding added during preprocessing (dw, dh).
        original_shape (Tuple[int, int]): Original image dimensions (height, width).

    Returns:
        np.ndarray: Array of detections with shape (num_detections, 6) where each row contains
                    [x1, y1, x2, y2, confidence, class_id].
    """
    boxes = boxes_xyxy.copy()

    # Compute final scores and class IDs
    class_ids = np.argmax(scores, axis=1)  # Get class with highest score
    confidences = scores[np.arange(len(scores)), class_ids]  # Confidence for the selected class

    # Filter out low-confidence detections
    mask = confidences > conf_thresh
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # Apply Non-Maximum Suppression (NMS). cv2.dnn.NMSBoxes expects boxes as
    # (x, y, w, h) rects, not (x1, y1, x2, y2) corners — convert here; `boxes`
    # itself (used for the returned detections) stays in xyxy.
    nms_boxes = boxes.copy()
    nms_boxes[:, 2] -= nms_boxes[:, 0]  # width = x2 - x1
    nms_boxes[:, 3] -= nms_boxes[:, 1]  # height = y2 - y1
    indices = cv2.dnn.NMSBoxes(
        bboxes=nms_boxes.tolist(),
        scores=confidences.tolist(),
        score_threshold=conf_thresh,
        nms_threshold=iou_thresh,
    )

    if len(indices) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # Keep only the selected boxes, confidences, and class IDs
    indices = np.array(indices).flatten()
    boxes = boxes[indices]
    confidences = confidences[indices]
    class_ids = class_ids[indices]

    # Undo padding and scaling to restore original image coordinates
    dw, dh = pad
    boxes[:, [0, 2]] -= dw  # Adjust x-coordinates
    boxes[:, [1, 3]] -= dh  # Adjust y-coordinates
    boxes /= scale  # Rescale to original dimensions

    # Clip boxes to image bounds
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, original_shape[1])  # Clip x-coordinates
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, original_shape[0])  # Clip y-coordinates

    # Combine results into a single array
    return np.hstack((boxes, confidences[:, None], class_ids[:, None].astype(np.float32)))
