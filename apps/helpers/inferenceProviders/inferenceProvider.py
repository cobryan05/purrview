"""Inference Provider Interface Class"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DetectionResult:
    """Represents a single detection result."""

    bounding_box: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    confidence: float  # Confidence score (0.0 to 1.0)
    class_id: int  # Class ID of the detected object
    class_name: Optional[str] = None  # Human-readable class name (optional)


@dataclass
class InferenceResult:
    """Represents the inference result for an image."""

    detections: List[DetectionResult]  # List of detection results
    annotated_image: Optional[np.array] = None  # Annotated image (optional)
    inference_time: Optional[float] = None  # Time taken for inference (in seconds)
    source_image: Optional[np.array] = None  # Original source image (optional)


class InferenceProvider:
    async def processImage(self, image: np.array) -> InferenceResult:
        """Retrieves the next image for processing from the image source"""
        raise NotImplementedError()
