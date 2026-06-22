"""Inference provider for binary (present/absent) image classifiers.

Loads a timm-trained ONNX model whose output is [batch, num_classes] raw logits.
Applies softmax and returns a single DetectionResult for the configured label UUID
covering the full image extent.  Always emits the prediction regardless of
conf_thresh so the /classify endpoint can decide the is_present threshold itself.
"""

from __future__ import annotations

import time
import uuid
from typing import List

import numpy as np

from apps.helpers.inferenceProviders.inferenceProvider import InferenceProvider
from apps.helpers.types import BoundingBoxMetadata, DetectionResult, InferenceResult


class ImageClassifierInferenceProvider(InferenceProvider):
    """ONNX-backed binary image classifier (present / absent).

    Args:
        model_path:   Path to the ``.onnx`` file.
        label_uuid:   UUID of the label this model classifies.
        class_names:  Ordered list of class names matching the model's output indices.
                      Must contain ``"present"``; defaults to ``["absent", "present"]``.
    """

    _INPUT_SIZE = (224, 224)

    def __init__(self, model_path: str, label_uuid: str, class_names: List[str]):
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._label_uuid = label_uuid
        self._present_idx: int = class_names.index("present") if "present" in class_names else 1

    async def processImage(self, image: np.ndarray, **kwargs) -> InferenceResult:
        from apps.helpers.classifier_utils import preprocess_bgr_array, softmax

        conf_thresh: float = float(kwargs.get("conf_thresh", 0.5))
        src_h, src_w = image.shape[:2]

        t0 = time.perf_counter()
        inp = preprocess_bgr_array(image, self._INPUT_SIZE)[np.newaxis]  # [1,3,H,W]
        logits = self._session.run(None, {self._input_name: inp})[0][0]  # [num_classes]
        probs = softmax(logits)
        confidence = float(probs[self._present_idx])
        t1 = time.perf_counter()

        detections = []
        if confidence >= conf_thresh:
            detections.append(
                DetectionResult(
                    bbox=BoundingBoxMetadata(
                        uuid=str(uuid.uuid4()),
                        x=0.0,
                        y=0.0,
                        width=float(src_w),
                        height=float(src_h),
                        label_uuid=self._label_uuid,
                        class_str=self._label_uuid,
                    ),
                    confidence=confidence,
                )
            )

        return InferenceResult(
            source_width=src_w,
            source_height=src_h,
            detections=detections,
            inference_time=(t1 - t0) * 1000,
        )
