from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

import apps.helpers.yoloUtils as YoloUtils

from .inferenceProvider import InferenceProvider

class InferenceResult:
    def __init__(
        self, boxes: np.ndarray, confidences: np.ndarray, class_ids: np.ndarray, class_names: Optional[List[str]] = None
    ):
        self.boxes = boxes
        self.confidences = confidences
        self.class_ids = class_ids
        self.class_names = class_names


class YOLOv8ONNXInferenceProvider(InferenceProvider):
    def __init__(
        self,
        model_path: str,
        class_names: Optional[List[str]] = None,
        conf_thresh: float = 0.251,
        iou_thresh: float = 0.45,
    ):
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        try:
            self.input_shape = YoloUtils.get_input_shape(self.session)  # (h, w)
        except:
            self.input_shape = [640, 640]

        self.class_names = class_names
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    async def processImage(self, image: np.ndarray, **kwargs) -> InferenceResult:
        original_shape = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img, scale, pad = YoloUtils.preprocess_image(image, self.input_shape)
        preds = self.session.run(None, {self.input_name: img})[0]

        # Handle (batch, classes, boxes) -> (batch, boxes, classes)
        if preds.shape[1] < preds.shape[2]:
            preds = np.transpose(preds, (0, 2, 1))

        detections = YoloUtils.postprocess(
            preds,
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh,
            scale=scale,
            pad=pad,
            original_shape=original_shape,
        )

        return InferenceResult(
            boxes=detections[:, :4],
            confidences=detections[:, 4],
            class_ids=detections[:, 5].astype(int),
            class_names=[self.class_names[i+1] for i in detections[:, 5].astype(int)] if self.class_names else None,
        )
