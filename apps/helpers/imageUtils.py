""" Image utilities """

import cv2
import numpy as np

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
        np.array: Array of IoU values for the input box against each box in `boxes`.
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
