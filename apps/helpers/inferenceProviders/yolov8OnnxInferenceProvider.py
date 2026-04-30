from typing import List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np
import onnxruntime as ort

import apps.helpers.yoloUtils as YoloUtils
from apps.helpers.bboxUtils import BBox
from apps.helpers.types import BoundingBoxMetadata, DetectionResult, InferenceResult

from .inferenceProvider import InferenceProvider


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
        imgH, imgW = original_shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img, scale, pad = YoloUtils.preprocess_image(image_rgb, self.input_shape)
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

        detection_results: List[DetectionResult] = []
        for det in detections:
            x_min, y_min, x_max, y_max = map(float, det[:4])
            confidence = float(det[4])
            class_id = int(det[5])
            class_name = self.class_names[class_id + 1] if self.class_names else None

            # Convert absolute xyxy to relative x1y1wh for BBox
            rel_bbox = BBox.fromX1Y1X2Y2(x_min, y_min, x_max, y_max, imgW, imgH)
            x, y, width, height = rel_bbox.asRX1Y1WH()

            bbox = BoundingBoxMetadata(uuid=str(uuid4()), class_str=class_name, x=x, y=y, width=width, height=height)
            res = DetectionResult(bbox=bbox, confidence=confidence)
            detection_results.append(res)

        return InferenceResult(
            source_width=imgW,
            source_height=imgH,
            detections=detection_results,
            inference_time=None,
            source_image=image,
        )
