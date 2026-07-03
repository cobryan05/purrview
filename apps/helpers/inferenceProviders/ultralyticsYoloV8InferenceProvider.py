import asyncio
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np
import onnxruntime as ort

import apps.helpers.yoloUtils as YoloUtils
from apps.helpers.bboxUtils import BBox
from apps.helpers.types import BoundingBoxMetadata, DetectionResult, InferenceResult

from .inferenceProvider import InferenceProvider
from .Registry import register_inference_provider

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@register_inference_provider("ultralytics_yolov8")
class UltralyticsYoloV8InferenceProvider(InferenceProvider):
    """Ultralytics YOLOv8 exports: boxes are cxcywh, scores are raw logits.
    (1, 4+nc, num_preds) transposed to (1, num_preds, 4+nc) by the caller —
    4 box channels + nc class-score channels, no separate objectness channel."""

    def __init__(
        self,
        model_path: str,
        class_names: Optional[List[str]] = None,
        conf_thresh: float = 0.251,
        iou_thresh: float = 0.45,
    ):
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        active_providers = self.session.get_providers()
        logger.info(f"ONNX providers active: {active_providers}")

        use_cuda = "CUDAExecutionProvider" in active_providers

        # If using CUDA, spin up a dedicated single-threaded executor
        # to prevent CUDA memory/context errors from shifting threads.
        if use_cuda:
            logger.info("Initializing dedicated background thread executor for CUDA.")
            self.cuda_executor = ThreadPoolExecutor(max_workers=1)
        else:
            self.cuda_executor = None

        self.input_name = self.session.get_inputs()[0].name
        try:
            self.input_shape = YoloUtils.get_input_shape(self.session)  # (h, w)
        except:
            self.input_shape = [640, 640]

        self.class_names = class_names
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def _extract(self, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split raw preds (num_preds, 4+nc) into (boxes_xyxy, scores)."""
        if preds.shape[1] < 5:
            raise ValueError("Prediction tensor must have at least 5 elements per box (x, y, w, h, class_scores...).")
        boxes = preds[:, :4].copy()
        scores = YoloUtils.sigmoid(preds[:, 4:])

        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = boxes.copy()
        boxes_xyxy[:, 0] -= boxes[:, 2] / 2  # x1 = cx - w / 2
        boxes_xyxy[:, 1] -= boxes[:, 3] / 2  # y1 = cy - h / 2
        boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes[:, 2]  # x2 = x1 + w
        boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes[:, 3]  # y2 = y1 + h

        return boxes_xyxy, scores

    async def processImage(self, image: np.ndarray, **kwargs) -> InferenceResult:
        original_shape = image.shape[:2]
        imgH, imgW = original_shape

        t0 = time.perf_counter()  # PERF
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img, scale, pad = YoloUtils.preprocess_image(image_rgb, self.input_shape)
        t_pre = time.perf_counter()  # PERF

        if self.cuda_executor:
            loop = asyncio.get_running_loop()
            preds_list = await loop.run_in_executor(self.cuda_executor, self.session.run, None, {self.input_name: img})
        else:
            preds_list = await asyncio.to_thread(self.session.run, None, {self.input_name: img})

        preds = preds_list[0]
        # -----------------------------------

        t_onnx = time.perf_counter()  # PERF

        # Handle (batch, classes, boxes) -> (batch, boxes, classes)
        if preds.shape[1] < preds.shape[2]:
            preds = np.transpose(preds, (0, 2, 1))

        conf_thresh = float(kwargs.get("conf_thresh", self.conf_thresh))
        boxes_xyxy, scores = self._extract(preds[0])
        detections = YoloUtils.decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=conf_thresh,
            iou_thresh=self.iou_thresh,
            scale=scale,
            pad=pad,
            original_shape=original_shape,
        )
        t_post = time.perf_counter()  # PERF
        # PERF BEGIN
        logger.debug(
            "processImage: preprocess=%.1fms onnx=%.1fms postprocess=%.1fms total=%.1fms (%d raw dets)",
            1000 * (t_pre - t0),
            1000 * (t_onnx - t_pre),
            1000 * (t_post - t_onnx),
            1000 * (t_post - t0),
            len(detections),
        )
        # PERF END

        detection_results: List[DetectionResult] = []
        for det in detections:
            x_min, y_min, x_max, y_max = map(float, det[:4])
            confidence = float(det[4])
            class_id = int(det[5])
            class_name = self.class_names[class_id] if self.class_names and class_id < len(self.class_names) else None

            # Convert absolute xyxy to relative x1y1wh for BBox
            rel_bbox = BBox.fromX1Y1X2Y2(x_min, y_min, x_max, y_max, imgW, imgH)
            x, y, width, height = rel_bbox.asRX1Y1WH()

            bbox = BoundingBoxMetadata(
                uuid=str(uuid4()),
                class_str=class_name,
                x=x,
                y=y,
                width=width,
                height=height,
            )
            res = DetectionResult(bbox=bbox, confidence=confidence)
            detection_results.append(res)

        return InferenceResult(
            source_width=imgW,
            source_height=imgH,
            detections=detection_results,
            inference_time=None,
            source_image=image,
        )

    def __del__(self):
        """Clean up background threads safely when provider object is destroyed."""
        if hasattr(self, "cuda_executor") and self.cuda_executor is not None:
            self.cuda_executor.shutdown(wait=False)
