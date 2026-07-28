"""YOLOX (Megvii) ONNX inference provider.

Pure numpy/cv2 — NO torch/yolox import here (this module runs in the
pure-ONNX inference container; see NOTES.md's "Importing any server main.py
runs its full import graph" guard, enforced by
tests/docker/test_schema_gen_torch_free.py).

YOLOX is the one YOLO family in this codebase with a genuinely different raw
ONNX output layout from every Ultralytics/LibreYOLO9 family already
supported:

- Raw output tensor: ``(1, n_anchors, 5 + nc)`` =
  ``[cx, cy, w, h, objectness, class_scores...]`` — confirmed against YOLOX's
  own ``yolox/models/yolo_head.py`` (inference-time forward concatenates
  ``[reg_output, obj_output.sigmoid(), cls_output.sigmoid()]``) and
  ``yolox/utils/demo_utils.py::demo_postprocess`` (the reference decode for
  a raw, non-``decode_in_inference`` export).
- Box channels are GRID-RELATIVE, not yet in pixel space: ``cx, cy`` are
  offsets from each anchor's grid cell (needs ``(cx + grid_x) * stride``),
  and ``w, h`` are log-scale (needs ``exp(w) * stride``). This provider
  targets the classic multi-stride (8/16/32) FPN export — the RAW (not
  ``--decode_in_inference``) export variant, matching what
  ``yolox_object_detection_trainer.py`` produces (``model.head
  .decode_in_inference = False`` before ``torch.onnx.export``).
- Objectness and class scores are ALREADY sigmoid-activated by the graph
  itself (same "don't re-sigmoid" trap as the Ultralytics providers — see
  ``ultralyticsYoloV8InferenceProvider.py``'s docstring) — but unlike every
  other family here, YOLOX keeps objectness as a SEPARATE channel: final
  per-class confidence is ``objectness * class_score``, not the class score
  alone.
- NMS is still required (YOLOX is not an NMS-free/end-to-end export family
  like YOLOv10/YOLO26) — ``SKIP_NMS`` stays ``False`` (the base-class
  default) so ``decode_detections`` runs its normal argmax + ``cv2.dnn
  .NMSBoxes`` path with the objectness-scaled scores this provider returns.
- Letterbox: YOLOX's own preprocessing (``yolox/data/data_augment.py``'s
  ``preproc``/``ValTransform``) anchors the resized image at the TOP-LEFT of
  the padded canvas (all padding pushed to bottom/right), same convention as
  LibreYOLO9 — NOT centered like Ultralytics/YOLOv5. Also: YOLOX's own
  preprocessing does NOT divide by 255 (raw 0-255 float32 pixel values feed
  the network) in its standard, non-"legacy" path. The shared
  ``YoloUtils.preprocess_image`` always normalizes to [0, 1] and has no
  no-normalize option, so this provider overrides preprocessing (see
  ``processImage`` override below) rather than trying to express the
  no-normalize behavior through the shared helper's parameters.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

import apps.helpers.yoloUtils as YoloUtils
from apps.helpers.bboxUtils import BBox
from apps.helpers.types import BoundingBoxMetadata, DetectionResult, InferenceResult

from .onnxYoloInferenceProvider import OnnxYoloInferenceProvider
from .Registry import register_inference_provider

# Classic YOLOX FPN strides (P3/P4/P5). p6-variant (YOLOX-x with an extra P6
# head) is out of scope here — all arches in yolox_object_detection_trainer's
# _ARCH_MAP (nano/tiny/s/m/l/x) use the standard 3-stride head.
_STRIDES = (8, 16, 32)


def _build_grids_and_strides(input_h: int, input_w: int, strides=_STRIDES) -> Tuple[np.ndarray, np.ndarray]:
    """Build the (n_anchors, 2) grid-cell coordinate array and matching
    (n_anchors, 1) per-anchor stride array, concatenated across all strides in
    the same order the model's head flattens/concatenates them (P3 first,
    i.e. smallest stride / largest feature map first) — mirrors YOLOX's own
    ``demo_postprocess``.
    """
    grids = []
    expanded_strides = []
    for stride in strides:
        h_feat = input_h // stride
        w_feat = input_w // stride
        xv, yv = np.meshgrid(np.arange(w_feat), np.arange(h_feat))
        grid = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((grid.shape[0], 1), stride))
    return np.concatenate(grids, axis=0).astype(np.float32), np.concatenate(expanded_strides, axis=0).astype(np.float32)


@register_inference_provider("yolox")
class YoloxInferenceProvider(OnnxYoloInferenceProvider):
    """Decodes YOLOX's grid-relative raw output + separate objectness channel.

    Overrides ``processImage`` (not just ``_extract``) because YOLOX's own
    preprocessing convention (no /255 normalize) differs from the shared
    ``YoloUtils.preprocess_image`` helper in a way that helper's parameters
    can't express — see module docstring.
    """

    LETTERBOX_ANCHOR = "top_left"
    SKIP_NMS = False

    def _extract(self, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split raw preds (num_preds, 5+nc) into (boxes_xyxy, scores).

        ``preds`` columns: ``[cx, cy, w, h, objectness, class_scores...]``,
        boxes grid-relative. Returns ``boxes_xyxy`` already decoded to
        letterboxed-canvas pixel space, and ``scores`` = per-class score
        already multiplied by objectness (so the shared ``decode_detections``
        argmax + threshold path works unmodified — it expects a single
        already-combined per-class score matrix with no separate objectness
        column).
        """
        if preds.shape[1] < 6:
            raise ValueError(
                "Prediction tensor must have at least 6 elements per box "
                "(cx, cy, w, h, objectness, class_scores...)."
            )
        input_h, input_w = self.input_shape
        grids, strides = _build_grids_and_strides(input_h, input_w)
        if grids.shape[0] != preds.shape[0]:
            raise ValueError(
                f"Grid/anchor count mismatch: built {grids.shape[0]} grid cells for "
                f"input_shape={self.input_shape} but preds has {preds.shape[0]} rows. "
                "This provider only supports the standard 3-stride (8/16/32) YOLOX head."
            )

        boxes = preds[:, :4].copy()
        objectness = preds[:, 4:5]
        class_scores = preds[:, 5:]

        # Stride-grid decode (yolox/utils/demo_utils.py::demo_postprocess):
        # cx,cy are grid-cell-relative offsets; w,h are log-scale.
        cx = (boxes[:, 0] + grids[:, 0]) * strides[:, 0]
        cy = (boxes[:, 1] + grids[:, 1]) * strides[:, 0]
        w = np.exp(boxes[:, 2]) * strides[:, 0]
        h = np.exp(boxes[:, 3]) * strides[:, 0]

        boxes_xyxy = np.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            axis=1,
        ).astype(np.float32)

        # Separate objectness channel (unlike every other family here) — final
        # confidence is objectness * class_score, both already sigmoid'd by
        # the exported graph (do not re-sigmoid).
        scores = (objectness * class_scores).astype(np.float32)

        return boxes_xyxy, scores

    async def processImage(self, image: np.ndarray, **kwargs) -> InferenceResult:
        """Overrides the base pipeline's preprocessing step only: YOLOX's own
        preprocessing does not normalize to [0, 1] (raw 0-255 float32 pixel
        values feed the network) — the shared ``YoloUtils.preprocess_image``
        always divides by 255 with no opt-out, so this duplicates that
        function's resize/letterbox/pad logic without the normalize step
        rather than adding a provider-specific flag to the shared helper that
        every other (normalize-expecting) family would have to ignore.
        """
        import asyncio
        import time
        from uuid import uuid4

        original_shape = image.shape[:2]
        imgH, imgW = original_shape

        t0 = time.perf_counter()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img, scale, pad = self._preprocess_no_normalize(image_rgb, self.input_shape)
        t_pre = time.perf_counter()

        if self.cuda_executor:
            loop = asyncio.get_running_loop()
            preds_list = await loop.run_in_executor(self.cuda_executor, self.session.run, None, {self.input_name: img})
        else:
            preds_list = await asyncio.to_thread(self.session.run, None, {self.input_name: img})

        preds = preds_list[0]
        t_onnx = time.perf_counter()

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
            skip_nms=self.SKIP_NMS,
        )
        t_post = time.perf_counter()

        detection_results: List[DetectionResult] = []
        for det in detections:
            x_min, y_min, x_max, y_max = map(float, det[:4])
            confidence = float(det[4])
            class_id = int(det[5])
            class_name = self.class_names[class_id] if self.class_names and class_id < len(self.class_names) else None

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
            detection_results.append(DetectionResult(bbox=bbox, confidence=confidence))

        return InferenceResult(
            source_width=imgW,
            source_height=imgH,
            detections=detection_results,
            inference_time=None,
            source_image=image,
        )

    @staticmethod
    def _preprocess_no_normalize(
        image: np.ndarray, input_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Same resize/letterbox/pad math as ``YoloUtils.preprocess_image``
        (top-left anchored, value-114 padding) but WITHOUT the final /255
        normalize step — matches YOLOX's own ``preproc``/``ValTransform``.
        """
        h0, w0 = image.shape[:2]
        input_h, input_w = input_shape

        r = min(input_w / w0, input_h / h0)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))

        dw, dh = 0.0, 0.0  # top-left anchor: all padding goes right/bottom

        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top = int(dh)
        left = int(dw)
        bottom = input_h - new_unpad[1] - top
        right = input_w - new_unpad[0] - left
        padded = cv2.copyMakeBorder(
            resized,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        img = padded.astype(np.float32)  # NO /255 — YOLOX feeds raw 0-255 values
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)

        return img, r, (dw, dh)

    def __del__(self):
        if hasattr(self, "cuda_executor") and self.cuda_executor is not None:
            self.cuda_executor.shutdown(wait=False)
