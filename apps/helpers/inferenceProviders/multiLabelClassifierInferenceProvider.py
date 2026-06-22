"""Inference provider for multi-label image classifiers.

Loads a timm-trained ONNX model whose output is [batch, N] raw logits, one per
label.  Applies sigmoid and returns one DetectionResult per label whose probability
exceeds conf_thresh, covering the full image extent.
"""

from __future__ import annotations

import time
import uuid
from typing import List

import numpy as np

from apps.helpers.inferenceProviders.inferenceProvider import InferenceProvider
from apps.helpers.types import BoundingBoxMetadata, DetectionResult, InferenceResult


class MultiLabelClassifierInferenceProvider(InferenceProvider):
    """ONNX-backed multi-label image classifier.

    Args:
        model_path:  Path to the ``.onnx`` file.
        label_uuids: Ordered list of label UUIDs matching the model's output indices.
    """

    _INPUT_SIZE = (224, 224)

    def __init__(self, model_path: str, label_uuids: List[str]):
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._label_uuids = label_uuids

    async def processImage(self, image: np.ndarray, **kwargs) -> InferenceResult:
        from apps.helpers.classifier_utils import preprocess_bgr_array, sigmoid

        conf_thresh: float = float(kwargs.get("conf_thresh", 0.5))
        src_h, src_w = image.shape[:2]

        t0 = time.perf_counter()
        inp = preprocess_bgr_array(image, self._INPUT_SIZE)[np.newaxis]  # [1,3,H,W]
        logits = self._session.run(None, {self._input_name: inp})[0][0]  # [N]
        probs = sigmoid(logits)
        t1 = time.perf_counter()

        detections = []
        for prob, label_uuid in zip(probs.tolist(), self._label_uuids):
            if prob >= conf_thresh:
                detections.append(
                    DetectionResult(
                        bbox=BoundingBoxMetadata(
                            uuid=str(uuid.uuid4()),
                            x=0.0,
                            y=0.0,
                            width=float(src_w),
                            height=float(src_h),
                            label_uuid=label_uuid,
                            class_str=label_uuid,
                        ),
                        confidence=float(prob),
                    )
                )

        return InferenceResult(
            source_width=src_w,
            source_height=src_h,
            detections=detections,
            inference_time=(t1 - t0) * 1000,
        )
