"""Inference Provider Interface Class"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from apps.helpers.bboxUtils import BBox


@dataclass
class DetectionResult:
    """Represents a single detection result."""

    bounding_box: BBox
    confidence: float  # Confidence score (0.0 to 1.0)
    class_id: int  # Class ID of the detected object
    class_name: Optional[str] = None  # Human-readable class name (optional)
    label_uuid: Optional[str] = None  # UUID of label that this class is associated with (optional)


    def serialize(self) -> dict:
        return {
            "bounding_box": self.bounding_box.asRX1Y1WH(),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "label_uuid": self.label_uuid,
        }


@dataclass
class InferenceResult:
    """Represents the inference result for an image."""

    detections: List[DetectionResult]  # List of detection results
    annotated_image: Optional[np.ndarray] = None  # Annotated image (optional)
    inference_time: Optional[float] = None  # Time taken for inference (in seconds)
    source_image: Optional[np.ndarray] = None  # Original source image (optional)

    def serialize(self, include_annotated: bool = False) -> dict:
        result = {
            "detections": [det.serialize() for det in self.detections],
        }
        if self.inference_time is not None:
            result["inference_time"] = self.inference_time
        if include_annotated and self.annotated_image is not None:
            import cv2, base64
            _, buffer = cv2.imencode(".png", self.annotated_image)
            result["annotated_image"] = base64.b64encode(buffer).decode("utf-8")
        return result


class InferenceProvider:
    async def processImage(self, image: np.ndarray, **kwargs) -> InferenceResult:
        """Retrieves the next image for processing from the image source"""
        raise NotImplementedError()
