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

if os.getenv("GENERATING_OPENAPI_CLIENTS") != "1":
    import cv2
    import onnxruntime as ort  # noqa: F811

    from apps.helpers.inferenceProviders.yolov8OnnxInferenceProvider import (
        YOLOv8ONNXInferenceProvider,
    )

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_input_shape(session: "ort.InferenceSession") -> Tuple[int, int]:
    input_shape = session.get_inputs()[0].shape  # e.g. [1, 3, 640, 640]
    return int(input_shape[2]), int(input_shape[3])  # height, width


def load_yolo_onnx(model_path: str) -> InferenceProvider:
    metadata = metadataUtils.get_model_metadata(model_path, create_json_if_missing=True)

    model_kind: str = metadata.get("model_kind", "")
    if model_kind == "image_classifier":
        from apps.helpers.inferenceProviders.imageClassifierInferenceProvider import (
            ImageClassifierInferenceProvider,
        )
        label_uuid: str = metadata.get("label_uuid", "")
        class_names: list[str] = metadata.get("class_names", ["absent", "present"])
        return ImageClassifierInferenceProvider(model_path, label_uuid, class_names)

    if model_kind == "multi_label_classifier":
        from apps.helpers.inferenceProviders.multiLabelClassifierInferenceProvider import (
            MultiLabelClassifierInferenceProvider,
        )
        label_uuids: list[str] = metadata.get("label_uuids") or []
        return MultiLabelClassifierInferenceProvider(model_path, label_uuids)

    classes: list[str] = metadata.get("classes", None)
    yolo_version: int = metadata.get("yolo", 8)
    if yolo_version in (8, 9):
        # Both export (1, 4+nc, num_preds) transposed to (1, num_preds, 4+nc) by
        # the provider — 4 box channels + nc class-score channels, no separate
        # objectness channel. The two families differ in box encoding and
        # whether scores are raw logits:
        #   yolo 8 (ultralytics YOLOv8): boxes are cxcywh, scores are raw logits.
        #   yolo 9 (libreyolo9):         boxes are already xyxy (canvas-pixel
        #                                 space), scores are already sigmoid
        #                                 activated inside the exported graph.
        box_format = "cxcywh" if yolo_version == 8 else "xyxy"
        scores_are_logits = yolo_version == 8
        return YOLOv8ONNXInferenceProvider(
            model_path, classes, box_format=box_format, scores_are_logits=scores_are_logits
        )
    raise Exception(f"Unhandled YOLO version: {yolo_version}")


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


def postprocess(
    preds: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    scale: float,
    pad: Tuple[int, int],
    original_shape: Tuple[int, int],
    box_format: str = "cxcywh",
    scores_are_logits: bool = True,
) -> np.ndarray:
    """
    Postprocess predictions: apply Non-Maximum Suppression (NMS), rescale boxes, and clip to image bounds.

    Args:
        preds (np.ndarray): Model predictions of shape (1, num_preds, num_features), where
            num_features = 4 (box) + nc (class scores). There is no separate objectness channel —
            every YOLOv8/v9-family export used here has already folded objectness into the
            per-class scores (or never had one).
        conf_thresh (float): Confidence threshold for filtering detections.
        iou_thresh (float): IoU threshold for NMS.
        scale (float): Scaling factor used during preprocessing.
        pad (Tuple[int, int]): Padding added during preprocessing (dw, dh).
        original_shape (Tuple[int, int]): Original image dimensions (height, width).
        box_format (str): "cxcywh" (needs conversion to corners) or "xyxy" (already corners).
        scores_are_logits (bool): True if class scores are raw logits needing sigmoid();
            False if the exported graph already applied sigmoid (e.g. libreyolo9).

    Returns:
        np.ndarray: Array of detections with shape (num_detections, 6) where each row contains
                    [x1, y1, x2, y2, confidence, class_id].
    """
    # Extract predictions for the first batch
    preds = preds[0]  # shape: (num_preds, num_features)
    if preds.shape[1] < 5:
        raise ValueError("Prediction tensor must have at least 5 elements per box (x, y, w, h, class_scores...).")

    # Split predictions into boxes and per-class scores (no objectness column).
    boxes = preds[:, :4].copy()
    scores = sigmoid(preds[:, 4:]) if scores_are_logits else preds[:, 4:]

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

    if box_format == "cxcywh":
        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
        boxes[:, 0] -= boxes[:, 2] / 2  # x1 = cx - w / 2
        boxes[:, 1] -= boxes[:, 3] / 2  # y1 = cy - h / 2
        boxes[:, 2] += boxes[:, 0]  # x2 = x1 + w
        boxes[:, 3] += boxes[:, 1]  # y2 = y1 + h
    elif box_format != "xyxy":
        raise ValueError(f"Unknown box_format: {box_format!r}. Expected 'cxcywh' or 'xyxy'.")

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
