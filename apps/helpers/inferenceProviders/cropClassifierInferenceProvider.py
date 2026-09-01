"""Inference provider for crop classifiers.

Loads a timm-trained ONNX model whose output is [batch, N] raw logits, one per
child label of a sub-model boundary. Applies **softmax** and returns one
DetectionResult per class above conf_thresh, covering the full extent of what it
was given.

The caller is expected to hand it an already-cropped region — this is the second
pass of a hierarchical config, so the "image" it sees is one detection from the
top model, not a whole frame. Feeding it a full frame is not an error and will
not raise; it will just answer which class the *whole frame* most resembles,
which is meaningless. Crop first, with ``classifier_utils.crop_region`` and the
sidecar's ``crop_padding``, so serve-time framing matches training.

The sibling multi-label provider is deliberately not reused: its sigmoid scores
each label independently, which is right for "which of these things are in this
picture" and wrong here. A crop holds exactly one cat, so the classes compete —
softmax, and the scores sum to 1.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import List, Optional, Tuple

import numpy as np

from apps.helpers.inferenceProviders.inferenceProvider import InferenceProvider
from apps.helpers.types import BoundingBoxMetadata, DetectionResult, InferenceResult


class CropClassifierInferenceProvider(InferenceProvider):
    """ONNX-backed single-label crop classifier.

    Args:
        model_path:  Path to the ``.onnx`` file.
        label_uuids: Ordered list of label UUIDs matching the model's output indices.
    """

    _INPUT_SIZE = (224, 224)

    def __init__(self, model_path: str, label_uuids: List[str], providers: Optional[List[str]] = None):
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            model_path,
            providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._label_uuids = label_uuids

    async def processCrops(
        self,
        image: np.ndarray,
        boxes: List[dict],
        padding: float,
    ) -> List[List[Tuple[str, float]]]:
        """Classify several *relative* boxes of one image in a single session run.

        Returns one ``[(label_uuid, confidence), ...]`` list per box, highest
        first, in the caller's box order — order is the contract, since that is
        how the caller pairs a result back to the detection it came from.

        One batched run rather than a loop over processImage: a watched frame
        with three cats is three crops, and three round trips through
        onnxruntime cost measurably more than one batch of three.
        """
        from apps.helpers.classifier_utils import crop_region, preprocess_bgr_array, softmax

        if not boxes:
            return []

        tensors = [
            preprocess_bgr_array(
                crop_region(image, b["x"], b["y"], b["width"], b["height"], padding),
                self._INPUT_SIZE,
            )
            for b in boxes
        ]
        batch = np.stack(tensors)
        logits = await asyncio.to_thread(self._session.run, None, {self._input_name: batch})
        probs = softmax(logits[0], axis=-1)

        results: List[List[Tuple[str, float]]] = []
        for row in probs:
            scored = list(zip(self._label_uuids, (float(p) for p in row)))
            scored.sort(key=lambda pair: pair[1], reverse=True)
            results.append(scored)
        return results

    async def processImage(self, image: np.ndarray, **kwargs) -> InferenceResult:
        from apps.helpers.classifier_utils import preprocess_bgr_array, softmax

        conf_thresh: float = float(kwargs.get("conf_thresh", 0.5))
        src_h, src_w = image.shape[:2]

        t0 = time.perf_counter()
        inp = preprocess_bgr_array(image, self._INPUT_SIZE)[np.newaxis]  # [1,3,H,W]
        logits = self._session.run(None, {self._input_name: inp})[0][0]  # [N]
        probs = softmax(logits)
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
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return InferenceResult(
            source_width=src_w,
            source_height=src_h,
            detections=detections,
            inference_time=(t1 - t0) * 1000,
        )
