"""Image utilities"""

import base64
import json
from typing import List

import cv2
import lz4.frame
import numpy as np

from apps.helpers.types import DetectionResult


def base64_encode_png(image: np.ndarray) -> str:
    """Encodes an image to PNG format and packages it into a unified base64 packet."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode PNG image")

    b64_payload = base64.b64encode(buffer).decode("utf-8")

    packet = {"format": "png", "metadata": {}, "data": b64_payload}
    return json.dumps(packet)


def base64_encode_jpg(image: np.ndarray, quality: int = 85) -> str:
    """Encodes an image to JPEG format and packages it into a unified base64 packet."""
    success, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not success:
        raise RuntimeError("Failed to encode JPEG image")

    b64_payload = base64.b64encode(buffer).decode("utf-8")

    packet = {"format": "jpeg", "metadata": {}, "data": b64_payload}
    return json.dumps(packet)


def base64_encode_lz4(image: np.ndarray) -> str:
    """Compresses raw pixel bytes via LZ4 and packages it with necessary array shape metadata."""
    raw_bytes = image.tobytes()
    metadata = {
        "shape": image.shape,
        "dtype": str(image.dtype),
    }
    compressed_bytes = lz4.frame.compress(raw_bytes)
    b64_payload = base64.b64encode(compressed_bytes).decode("utf-8")

    packet = {"format": "lz4", "metadata": metadata, "data": b64_payload}
    return json.dumps(packet)

def base64_decode_image(json_packet_str: str) -> np.ndarray:
    """Universal reader that decodes an image packet from png, jpeg, or lz4 format

    back into a numpy array (OpenCV image format).
    """
    try:
        packet = json.loads(json_packet_str)
    except json.JSONDecodeError:
        # Assume it's a raw base64 string for PNG
        b64_data = json_packet_str
        packet = {"format": "png", "data": b64_data, "metadata": {}}

    fmt = packet.get("format")
    b64_data = packet.get("data")
    metadata = packet.get("metadata", {})

    if not fmt or not b64_data:
        raise ValueError("Invalid packet structure: missing 'format' or 'data'.")

    binary_data = base64.b64decode(b64_data.encode("utf-8"))

    if fmt in ("jpeg", "png"):
        np_arr = np.frombuffer(binary_data, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cv2.imdecode failed to parse {fmt} buffer.")
        return image

    elif fmt == "lz4":
        if "shape" not in metadata or "dtype" not in metadata:
            raise ValueError("LZ4 packets require 'shape' and 'dtype' metadata.")

        raw_bytes = lz4.frame.decompress(binary_data)
        flat_array = np.frombuffer(raw_bytes, dtype=metadata["dtype"])
        return flat_array.reshape(metadata["shape"])

    else:
        raise ValueError(f"Unsupported decoding format: '{fmt}'")


def letterbox(image, size=640, color=(114, 114, 114)):
    """
    Resizes and pads an image to fit within a square of the specified size while maintaining aspect ratio.

    Args:
        image (np.array): Input image as a NumPy array.
        size (int): Target size for the square (default is 640).
        color (Tuple[int, int, int]): Padding color in RGB format (default is (114, 114, 114)).

    Returns:
        Tuple[np.array, float, Tuple[float, float]]:
            - Resized and padded image.
            - Scaling factor used for resizing.
            - Padding applied (width and height).
    """
    shape = image.shape[:2]  # current shape [height, width]
    r = min(size / shape[0], size / shape[1])  # scaling factor
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # new dimensions
    dw, dh = size - new_unpad[0], size - new_unpad[1]  # width and height padding
    dw /= 2  # divide padding into left/right
    dh /= 2  # divide padding into top/bottom

    # Resize the image
    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Apply padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, r, (dw, dh)


def compute_iou(box, boxes):
    """
    Computes the Intersection over Union (IoU) between a single box and multiple boxes.

    Args:
        box (Tuple[int, int, int, int]): A single bounding box (x_min, y_min, x_max, y_max).
        boxes (np.array): Array of bounding boxes with shape (N, 4).

    Returns:
        np.ndarray: Array of IoU values for the input box against each box in `boxes`.
    """
    x1 = np.maximum(box[0], boxes[:, 0])  # Intersection top-left x
    y1 = np.maximum(box[1], boxes[:, 1])  # Intersection top-left y
    x2 = np.minimum(box[2], boxes[:, 2])  # Intersection bottom-right x
    y2 = np.minimum(box[3], boxes[:, 3])  # Intersection bottom-right y

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)  # Intersection area
    box_area = (box[2] - box[0]) * (box[3] - box[1])  # Area of the single box
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # Areas of all boxes

    union_area = box_area + boxes_area - inter_area  # Union area
    return inter_area / np.maximum(union_area, 1e-6)  # IoU


def nms(boxes, scores, iou_threshold):
    """
    Performs Non-Maximum Suppression (NMS) to filter overlapping bounding boxes based on their scores.

    Args:
        boxes (np.array): Array of bounding boxes with shape (N, 4).
        scores (np.array): Array of confidence scores for each bounding box.
        iou_threshold (float): IoU threshold for suppression (e.g., 0.5).

    Returns:
        List[int]: Indices of the bounding boxes to keep after NMS.
    """
    idxs = np.argsort(scores)[::-1]  # Sort indices by scores in descending order
    keep = []  # List to store indices of boxes to keep

    while idxs.size > 0:
        current = idxs[0]  # Index of the box with the highest score
        keep.append(current)  # Keep the current box
        if idxs.size == 1:
            break

        # Compute IoU between the current box and the remaining boxes
        ious = compute_iou(boxes[current], boxes[idxs[1:]])
        idxs = idxs[1:][ious <= iou_threshold]  # Suppress boxes with IoU > threshold

    return keep


def annotate_image(image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
    """
    Annotate the image with bounding boxes and labels.

    Args:
        image (np.ndarray): The original image in BGR format.
        detections (List[DetectionResult]): The detected bounding boxes.

    Returns:
        np.ndarray: The annotated image.
    """
    imgH, imgW = image.shape[:2]
    for detection in detections:
        # DetectionResult now uses a BBox object for bounding_box
        bbox = detection.bounding_box
        x1, y1, w, h = bbox.asX1Y1WH(imgW, imgH)
        x2, y2 = x1 + w, y1 + h
        label = detection.class_str if detection.class_str is not None else str(detection.class_id)
        confidence = detection.confidence

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put the label and confidence
        text = f"{label}: {confidence:.2f}"
        cv2.putText(
            image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return image
