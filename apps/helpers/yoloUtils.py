"""Utility functions for YOLO models"""

import logging
import os
import sys
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import onnxruntime as ort

import numpy as np
import onnx

import apps.helpers.metadataUtils as metadataUtils
from apps.helpers.inferenceProviders.inferenceProvider import InferenceProvider
from apps.helpers.inferenceProviders.Registry import inference_provider_registry
from apps.helpers.logUtils import LOG_LEVEL

if os.getenv("GENERATING_OPENAPI_CLIENTS") != "1":
    import cv2
    import onnxruntime as ort  # noqa: F811

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_input_shape(session: "ort.InferenceSession") -> Tuple[int, int]:
    input_shape = session.get_inputs()[0].shape  # e.g. [1, 3, 640, 640]
    return int(input_shape[2]), int(input_shape[3])  # height, width


def get_input_shape_from_file(model_path: str) -> Optional[Tuple[int, int]]:
    """(height, width) read from the ONNX graph, without loading a session.

    Mirrors get_input_shape() but works on a model that isn't pinned — no GPU
    context, no provider load. Returns None rather than guessing if the graph
    declares a dynamic height/width (no static dim_value on those axes).

    Blocking (onnx.load reads the whole file); call under asyncio.to_thread
    from async code, per apps.helpers.metadataUtils.get_onnx_metadata.

    Never raises: a model file that's missing, unreadable, or not valid ONNX
    (a test fixture's placeholder bytes, a mid-write file caught by a stat
    race) is not a reason to fail a model-info request that also carries
    device/label metadata that IS available — same "degrade to None, don't
    guess" contract as the dynamic-shape case.
    """
    try:
        model = onnx.load(model_path)
        dims = model.graph.input[0].type.tensor_type.shape.dim
        if len(dims) < 4:
            return None
        h_dim, w_dim = dims[2], dims[3]
        if not h_dim.HasField("dim_value") or not w_dim.HasField("dim_value"):
            return None
        return int(h_dim.dim_value), int(w_dim.dim_value)
    except Exception:
        logger.debug("Could not read input_shape from %s", model_path, exc_info=True)
        return None


def load_yolo_onnx(model_path: str) -> InferenceProvider:
    # Importing these registers them (via @register_inference_provider) into
    # inference_provider_registry as a side effect. Deferred to call time: the
    # provider modules import this module for the shared pipeline helpers, so a
    # module-level import here would be circular.
    import apps.helpers.inferenceProviders.libreYolo9InferenceProvider  # noqa: F401
    import apps.helpers.inferenceProviders.ultralyticsE2eInferenceProvider  # noqa: F401
    import apps.helpers.inferenceProviders.ultralyticsYoloV8InferenceProvider  # noqa: F401
    import apps.helpers.inferenceProviders.yoloxInferenceProvider  # noqa: F401

    metadata = metadataUtils.get_model_metadata(model_path, create_json_if_missing=True)
    providers = metadataUtils.providers_for_device(metadata.get("device", metadataUtils.DEFAULT_MODEL_DEVICE))

    model_kind: str = metadata.get("model_kind", "")
    if model_kind == "image_classifier":
        from apps.helpers.inferenceProviders.imageClassifierInferenceProvider import (
            ImageClassifierInferenceProvider,
        )

        label_uuid: str = metadata.get("label_uuid", "")
        class_names: list[str] = metadata.get("class_names", ["absent", "present"])
        return ImageClassifierInferenceProvider(model_path, label_uuid, class_names, providers=providers)

    if model_kind == "multi_label_classifier":
        from apps.helpers.inferenceProviders.multiLabelClassifierInferenceProvider import (
            MultiLabelClassifierInferenceProvider,
        )

        label_uuids: list[str] = metadata.get("label_uuids") or []
        return MultiLabelClassifierInferenceProvider(model_path, label_uuids, providers=providers)

    if model_kind == "crop_classifier":
        from apps.helpers.inferenceProviders.cropClassifierInferenceProvider import (
            CropClassifierInferenceProvider,
        )

        # The caller crops before calling; this provider only classifies what it is
        # handed. See the provider's docstring.
        crop_label_uuids: list[str] = metadata.get("label_uuids") or []
        return CropClassifierInferenceProvider(model_path, crop_label_uuids, providers=providers)

    classes: list[str] = metadata.get("classes", None)
    model_family: str = metadata.get("model_family") or metadataUtils.resolve_legacy_model_family(metadata)
    provider_cls = inference_provider_registry.get(model_family)
    if provider_cls is None:
        raise Exception(f"Unhandled model family {model_family!r}. Registered: {sorted(inference_provider_registry)}")
    return provider_cls(model_path, classes, providers=providers)


def preprocess_image(
    image: np.ndarray, input_shape: Tuple[int, int], anchor: str = "center"
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Preprocess an image for YOLO model inference by resizing, padding, and normalizing.

    Args:
        image (np.ndarray): Input image as a NumPy array (HWC format, BGR color space).
        input_shape (Tuple[int, int]): Target input shape (height, width) for the model.
        anchor (str): Where the resized image is placed within the padded canvas.
            "center" (Ultralytics/YOLOv5-style, default) splits padding evenly on
            both sides. "top_left" (LibreYOLO-style) places the resized image
            flush against the top-left corner, with all padding on the
            bottom/right — must match whatever convention the model was
            trained/validated with, or decoded boxes will be offset.

    Returns:
        Tuple[np.ndarray, float, Tuple[int, int]]:
            - Preprocessed image as a NumPy array (NCHW format, normalized to [0, 1]).
            - Scaling factor used to resize the image.
            - Padding added to the image (dw, dh) to maintain aspect ratio.
    """
    h0, w0 = image.shape[:2]
    input_h, input_w = input_shape

    r = min(input_w / w0, input_h / h0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))

    if anchor == "top_left":
        dw = 0.0
        dh = 0.0
    else:
        dw = (input_w - new_unpad[0]) / 2
        dh = (input_h - new_unpad[1]) / 2

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Pad to the target canvas regardless of anchor; top/left absorb all the
    # padding for "top_left", split evenly for "center".
    top = int(dh)
    left = int(dw)
    bottom = input_h - new_unpad[1] - top
    right = input_w - new_unpad[0] - left
    padded = cv2.copyMakeBorder(
        resized, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    img = padded.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, 0)  # -> NCHW

    return img, r, (dw, dh)


def decode_detections(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    scale: float,
    pad: Tuple[int, int],
    original_shape: Tuple[int, int],
    skip_nms: bool = False,
    top_k: int = 0,
) -> np.ndarray:
    """
    Shared NMS + rescale/clip tail for YOLO-family raw outputs.

    Family-agnostic: takes already-decoded xyxy boxes (canvas-pixel space, pre-NMS)
    and already-activated per-class scores (no objectness column). Each
    InferenceProvider is responsible for extracting `boxes_xyxy`/`scores` from its
    own raw tensor layout (box encoding, logit vs. sigmoid, etc.) before calling
    this.

    Args:
        boxes_xyxy (np.ndarray): Boxes as (num_preds, 4) in [x1, y1, x2, y2], canvas-pixel space.
        scores (np.ndarray): Per-class scores, shape (num_preds, nc), already activated (no raw logits).
            When `skip_nms` is True, this is instead a single already-picked
            confidence column, shape (num_preds, 1) — see below.
        conf_thresh (float): Confidence threshold for filtering detections.
        iou_thresh (float): IoU threshold for NMS. Unused when `skip_nms` is True.
        scale (float): Scaling factor used during preprocessing.
        pad (Tuple[int, int]): Padding added during preprocessing (dw, dh).
        original_shape (Tuple[int, int]): Original image dimensions (height, width).
        skip_nms (bool): NMS-free ("end-to-end") models (e.g. YOLOv10, YOLO26)
            already return picked rows — one row per detection, `scores` is a
            single confidence column (num_preds, 1), and there is no per-class
            score matrix to argmax over. When True, skip both the
            confidence-argmax class selection and the `cv2.dnn.NMSBoxes` call;
            only confidence thresholding + coordinate rescale/clip still apply.
            `class_ids` for these rows come from `boxes_xyxy`'s companion
            `scores` column-mate — callers must instead pass the already-picked
            class id via `scores[:, 0]` for confidence and a matching class-id
            array through the same row ordering (see
            `ultralyticsE2eInferenceProvider.py`'s `_extract`, which returns
            `scores` as (num_preds, 2): [confidence, class_id]).
        top_k (int): When > 0, also return the per-detection **n-best runners-up**
            — the classes that scored next-highest on the same prediction row.
            This is the "confident but confused" signal (AMBIGUITY_REVIEW_PLAN.md):
            the network already emits a score for every class on every prediction
            and this function otherwise discards all but the argmax, so the data
            costs nothing extra to produce.

    Returns:
        np.ndarray: when ``top_k <= 0`` (the default), detections with shape
            (num_detections, 6): [x1, y1, x2, y2, confidence, class_id].
        Tuple[np.ndarray, np.ndarray]: when ``top_k > 0``, that same array plus an
            ``alternates`` array of shape (num_detections, n_alt, 2) whose last
            axis is [class_id, confidence], ordered best-first and **excluding the
            winning class**. ``n_alt`` is ``min(top_k - 1, num_classes - 1)``.

            ⚠️ The conditional return type follows the numpy ``return_index`` /
            ``return_counts`` idiom deliberately: every existing caller and test
            passes no ``top_k`` and must keep receiving a bare array. Do not
            "clean this up" into an always-tuple return without updating all of
            them.

            ⚠️ Always empty (n_alt = 0) when ``skip_nms`` is True — NMS-free/E2E
            exports bake the argmax into the graph and never emit a per-class
            score matrix, so there is nothing to rank. Callers must surface that
            as "unavailable", not as "no confusion found".
    """
    boxes = boxes_xyxy.copy()
    n_alt = 0
    # Bound here, not only in the argmax branch below: the skip_nms path never
    # assigns it and the tail must still be able to read it.
    scores_masked: "np.ndarray | None" = None
    if top_k > 0 and not skip_nms:
        # scores is (num_preds, nc) here; one column can have no runner-up.
        n_alt = max(0, min(int(top_k) - 1, int(scores.shape[1]) - 1))

    def _empty_alternates(n_dets: int) -> np.ndarray:
        return np.zeros((n_dets, n_alt, 2), dtype=np.float32)

    if skip_nms:
        # Rows are already picked (one detection per row, no multi-class
        # score matrix to argmax over, no overlapping candidates left for
        # NMS to suppress). `scores` is (num_preds, 2): [confidence, class_id].
        confidences = scores[:, 0]
        class_ids = scores[:, 1].astype(np.float32)

        mask = confidences > conf_thresh
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            empty = np.zeros((0, 6), dtype=np.float32)
            return (empty, _empty_alternates(0)) if top_k > 0 else empty
    else:
        # Compute final scores and class IDs
        class_ids_int = np.argmax(scores, axis=1)  # Get class with highest score
        confidences = scores[np.arange(len(scores)), class_ids_int]  # Confidence for the selected class

        # Filter out low-confidence detections
        mask = confidences > conf_thresh
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids_int = class_ids_int[mask]
        # The surviving rows' FULL per-class score vectors, carried alongside the
        # argmax so the runners-up are still reachable after NMS. Only retained
        # when asked for — otherwise this is a wasted (num_preds, nc) copy on the
        # hot path of every inference call in the app.
        scores_masked = scores[mask] if n_alt > 0 else None

        if len(boxes) == 0:
            empty = np.zeros((0, 6), dtype=np.float32)
            return (empty, _empty_alternates(0)) if top_k > 0 else empty

        # Apply Non-Maximum Suppression (NMS). cv2.dnn.NMSBoxes expects boxes as
        # (x, y, w, h) rects, not (x1, y1, x2, y2) corners — convert here; `boxes`
        # itself (used for the returned detections) stays in xyxy.
        #
        # Class-aware: NMSBoxesBatched groups suppression by class_ids, so two
        # different-class boxes that nearly coincide (e.g. a cat detection
        # sharing almost the same rect as the litter_box it's standing in)
        # don't suppress each other. Only same-class overlaps get suppressed.
        nms_boxes = boxes.copy()
        nms_boxes[:, 2] -= nms_boxes[:, 0]  # width = x2 - x1
        nms_boxes[:, 3] -= nms_boxes[:, 1]  # height = y2 - y1
        indices = cv2.dnn.NMSBoxesBatched(
            bboxes=nms_boxes.tolist(),
            scores=confidences.tolist(),
            class_ids=class_ids_int.tolist(),
            score_threshold=conf_thresh,
            nms_threshold=iou_thresh,
        )

        if len(indices) == 0:
            empty = np.zeros((0, 6), dtype=np.float32)
            return (empty, _empty_alternates(0)) if top_k > 0 else empty

        # Keep only the selected boxes, confidences, and class IDs
        indices = np.array(indices).flatten()
        boxes = boxes[indices]
        confidences = confidences[indices]
        class_ids = class_ids_int[indices].astype(np.float32)
        if scores_masked is not None:
            scores_masked = scores_masked[indices]

    # Undo padding and scaling to restore original image coordinates
    dw, dh = pad
    boxes[:, [0, 2]] -= dw  # Adjust x-coordinates
    boxes[:, [1, 3]] -= dh  # Adjust y-coordinates
    boxes /= scale  # Rescale to original dimensions

    # Clip boxes to image bounds
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, original_shape[1])  # Clip x-coordinates
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, original_shape[0])  # Clip y-coordinates

    # Combine results into a single array
    dets = np.hstack((boxes, confidences[:, None], class_ids[:, None]))
    if top_k <= 0:
        return dets

    if n_alt == 0 or skip_nms or scores_masked is None:
        # Either nothing was asked for beyond the winner, or the provider's
        # architecture never gave us a score matrix to rank (skip_nms).
        return dets, _empty_alternates(len(dets))

    # Mask out the winner so it can't reappear as its own runner-up, then take
    # the next n_alt classes. argpartition is O(nc) per row rather than a full
    # sort; only the small n_alt slice is then ordered.
    masked = scores_masked.astype(np.float32, copy=True)
    masked[np.arange(len(masked)), class_ids.astype(np.int64)] = -np.inf
    part = np.argpartition(-masked, n_alt - 1, axis=1)[:, :n_alt]
    part_scores = np.take_along_axis(masked, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    alt_ids = np.take_along_axis(part, order, axis=1).astype(np.float32)
    alt_scores = np.take_along_axis(part_scores, order, axis=1).astype(np.float32)
    alternates = np.stack((alt_ids, alt_scores), axis=2)
    return dets, alternates
