"""Unit tests for YOLO ONNX decode.

Pure numpy: no I/O, no ONNX session, no network.

Covers:
  - `decode_detections` (apps/helpers/yoloUtils.py) — the family-agnostic NMS +
    rescale/clip tail shared by every provider.
  - Each provider's `_extract` — the family-specific raw-tensor decode:
    - `UltralyticsYoloV8InferenceProvider._extract` — cxcywh boxes, scores
      already sigmoid-activated by the exported graph (real ultralytics
      YOLOv8/YOLO11 exports, yolo: 8).
    - `LibreYolo9InferenceProvider._extract` — xyxy boxes, pre-sigmoided scores
      (libreyolo9 exports, yolo: 9).

Both layouts are (num_preds, 4 + nc): 4 box channels + nc class-score channels,
with NO separate objectness channel, and both are already sigmoid-activated —
confirmed against real yolov8x.onnx/Yolo11S.onnx/libreyolo9 exports (see
tests/helpers/test_yolo_sample_models.py, which runs the real ONNX fixtures
end to end).
"""

import numpy as np
import pytest

from apps.helpers.inferenceProviders.libreYolo9InferenceProvider import (
    LibreYolo9InferenceProvider,
)
from apps.helpers.inferenceProviders.ultralyticsE2eInferenceProvider import (
    UltralyticsE2eInferenceProvider,
)
from apps.helpers.inferenceProviders.ultralyticsYoloV8InferenceProvider import (
    UltralyticsYoloV8InferenceProvider,
)
from apps.helpers.yoloUtils import decode_detections


def _extract_libreyolo9(preds: np.ndarray):
    return LibreYolo9InferenceProvider._extract(None, preds)


def _extract_yolov8(preds: np.ndarray):
    return UltralyticsYoloV8InferenceProvider._extract(None, preds)


def _extract_e2e(preds: np.ndarray):
    return UltralyticsE2eInferenceProvider._extract(None, preds)


class TestExtractLibreYolo9Format:
    """LibreYolo9InferenceProvider._extract — xyxy boxes, pre-sigmoided scores."""

    def test_decodes_correct_box_and_class(self):
        # One box for class 1 (of 2 classes), already xyxy in canvas-pixel space,
        # scores already sigmoid-activated (as libreyolo9's own decode assumes).
        preds = np.array(
            [[100.0, 100.0, 200.0, 200.0, 0.05, 0.92]],  # [x1, y1, x2, y2, class0, class1]
            dtype=np.float32,
        )
        boxes_xyxy, scores = _extract_libreyolo9(preds)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
        )
        assert dets.shape == (1, 6)
        x1, y1, x2, y2, confidence, class_id = dets[0]
        assert (x1, y1, x2, y2) == pytest.approx((100.0, 100.0, 200.0, 200.0), abs=1e-3)
        assert class_id == 1
        # Confidence should stay close to the input score (0.92), not be squashed
        # toward 0.5 by a spurious second sigmoid.
        assert confidence == pytest.approx(0.92, abs=1e-6)

    def test_double_sigmoid_regression_would_fail(self):
        """Guards against re-introducing the old objectness-column split + double sigmoid.

        The old code read preds[:, 4:5] as "objectness" and preds[:, 5:] as
        "class_probs", sigmoid'ing both again. For this fixture that collapses
        the two already-distinct scores toward ~0.5-0.6, well under a 0.7
        threshold — reproducing the "boxes look random" symptom. The fixed
        decode must clear a 0.7 threshold on the true 0.92 score.
        """
        preds = np.array(
            [[100.0, 100.0, 200.0, 200.0, 0.05, 0.92]],
            dtype=np.float32,
        )
        boxes_xyxy, scores = _extract_libreyolo9(preds)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.7,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
        )
        assert len(dets) == 1
        assert dets[0][4] == pytest.approx(0.92, abs=1e-6)

    def test_undoes_padding_and_scale(self):
        preds = np.array(
            [[110.0, 60.0, 210.0, 160.0, 0.0, 0.9]],
            dtype=np.float32,
        )
        boxes_xyxy, scores = _extract_libreyolo9(preds)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=2.0,
            pad=(10.0, 20.0),
            original_shape=(200, 200),
        )
        assert len(dets) == 1
        x1, y1, x2, y2 = dets[0][:4]
        # (110-10)/2=50, (60-20)/2=20, (210-10)/2=100, (160-20)/2=70
        assert (x1, y1, x2, y2) == pytest.approx((50.0, 20.0, 100.0, 70.0), abs=1e-3)

    def test_low_score_filtered_out(self):
        preds = np.array(
            [[100.0, 100.0, 200.0, 200.0, 0.1, 0.15]],
            dtype=np.float32,
        )
        boxes_xyxy, scores = _extract_libreyolo9(preds)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
        )
        assert dets.shape == (0, 6)

    def test_rejects_too_few_channels(self):
        preds = np.zeros((1, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            _extract_libreyolo9(preds)


class TestExtractYolov8Format:
    """UltralyticsYoloV8InferenceProvider._extract — cxcywh boxes, scores
    already sigmoid-activated by the exported graph."""

    def test_decodes_correct_box_and_class(self):
        # cx=150, cy=150, w=100, h=100 -> x1,y1,x2,y2 = 100,100,200,200
        preds = np.array(
            [[150.0, 150.0, 100.0, 100.0, 0.05, 0.9]],
            dtype=np.float32,
        )
        boxes_xyxy, scores = _extract_yolov8(preds)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
        )
        assert dets.shape == (1, 6)
        x1, y1, x2, y2, confidence, class_id = dets[0]
        assert (x1, y1, x2, y2) == pytest.approx((100.0, 100.0, 200.0, 200.0), abs=1e-3)
        assert class_id == 1
        assert confidence == pytest.approx(0.9, abs=1e-6)

    def test_class_zero_is_reachable(self):
        """class_id 0 must be selectable directly — no more '+1' shift downstream."""
        preds = np.array(
            [[150.0, 150.0, 100.0, 100.0, 0.9, 0.05]],
            dtype=np.float32,
        )
        boxes_xyxy, scores = _extract_yolov8(preds)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
        )
        assert len(dets) == 1
        assert dets[0][5] == 0

    def test_rejects_too_few_channels(self):
        preds = np.zeros((1, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            _extract_yolov8(preds)

    def test_double_sigmoid_regression_would_fail(self):
        """Guards against re-introducing a spurious extra sigmoid() call on

        already-activated scores — that regression squashed every detection
        toward ~0.5 confidence regardless of the model's actual prediction
        (the "tons of boxes at 50% confidence" symptom for real YOLOv8/YOLO11
        ONNX exports, whose Detect head already sigmoids class scores before
        the graph returns).
        """
        preds = np.array(
            [[150.0, 150.0, 100.0, 100.0, 0.02, 0.9]],
            dtype=np.float32,
        )
        boxes_xyxy, scores = _extract_yolov8(preds)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.7,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
        )
        assert len(dets) == 1
        assert dets[0][4] == pytest.approx(0.9, abs=1e-6)


class TestDecodeDetectionsNms:
    def test_distinct_non_overlapping_boxes_both_survive_nms(self):
        """cv2.dnn.NMSBoxes expects (x, y, w, h) rects, not (x1, y1, x2, y2) corners.

        These two boxes are far enough from the image origin — and genuinely
        non-overlapping (real IoU = 0) — that feeding raw xyxy corners
        straight into NMSBoxes (misread as x,y,w,h) produces oversized phantom
        rects that spuriously overlap and get one box wrongly suppressed.
        """
        boxes_xyxy = np.array([[300.0, 300.0, 340.0, 340.0], [400.0, 400.0, 440.0, 440.0]], dtype=np.float32)
        scores = np.array([[0.9], [0.8]], dtype=np.float32)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.25,
            iou_thresh=0.1,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
        )
        assert len(dets) == 2

    def test_truly_overlapping_boxes_are_still_suppressed(self):
        """Sanity check that the xywh fix doesn't just disable NMS entirely."""
        boxes_xyxy = np.array([[100.0, 100.0, 200.0, 200.0], [110.0, 110.0, 210.0, 210.0]], dtype=np.float32)
        scores = np.array([[0.9], [0.8]], dtype=np.float32)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
        )
        assert len(dets) == 1
        assert dets[0][4] == pytest.approx(0.9, abs=1e-6)


class TestExtractUltralyticsE2eFormat:
    """UltralyticsE2eInferenceProvider._extract — YOLOv10/YOLO26 NMS-free
    export: (num_preds, 6) rows of already-picked [x1, y1, x2, y2, conf, cls],
    confirmed against real yolov10n.onnx/yolo26n.onnx exports (see
    MORE_YOLO_PLAN.md Section 2 research notes)."""

    def test_decodes_correct_box_and_class(self):
        preds = np.array(
            [[100.0, 100.0, 200.0, 200.0, 0.92, 1.0]],  # [x1, y1, x2, y2, conf, cls]
            dtype=np.float32,
        )
        boxes_xyxy, scores = _extract_e2e(preds)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
            skip_nms=True,
        )
        assert dets.shape == (1, 6)
        x1, y1, x2, y2, confidence, class_id = dets[0]
        assert (x1, y1, x2, y2) == pytest.approx((100.0, 100.0, 200.0, 200.0), abs=1e-3)
        assert class_id == 1
        assert confidence == pytest.approx(0.92, abs=1e-6)

    def test_rejects_too_few_channels(self):
        preds = np.zeros((1, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            _extract_e2e(preds)

    def test_low_score_filtered_out(self):
        preds = np.array(
            [[100.0, 100.0, 200.0, 200.0, 0.1, 0.0]],
            dtype=np.float32,
        )
        boxes_xyxy, scores = _extract_e2e(preds)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
            skip_nms=True,
        )
        assert dets.shape == (0, 6)

    def test_undoes_padding_and_scale(self):
        preds = np.array(
            [[110.0, 60.0, 210.0, 160.0, 0.9, 0.0]],
            dtype=np.float32,
        )
        boxes_xyxy, scores = _extract_e2e(preds)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=2.0,
            pad=(10.0, 20.0),
            original_shape=(200, 200),
            skip_nms=True,
        )
        assert len(dets) == 1
        x1, y1, x2, y2 = dets[0][:4]
        # (110-10)/2=50, (60-20)/2=20, (210-10)/2=100, (160-20)/2=70
        assert (x1, y1, x2, y2) == pytest.approx((50.0, 20.0, 100.0, 70.0), abs=1e-3)

    def test_skip_nms_does_not_drop_overlapping_boxes(self):
        """NMS-free models can legitimately return two heavily overlapping
        boxes if the graph's own dedup considered them distinct detections
        (e.g. two different classes at the same location) — decode_detections
        must NOT run cv2.dnn.NMSBoxes on skip_nms=True input and silently
        suppress one of them."""
        preds = np.array(
            [
                [100.0, 100.0, 200.0, 200.0, 0.9, 0.0],
                [100.0, 100.0, 200.0, 200.0, 0.8, 1.0],
            ],
            dtype=np.float32,
        )
        boxes_xyxy, scores = _extract_e2e(preds)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.25,
            iou_thresh=0.1,  # would aggressively suppress if NMS ran
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
            skip_nms=True,
        )
        assert len(dets) == 2

    def test_multiple_rows_preserve_ordering_and_class_ids(self):
        preds = np.array(
            [
                [10.0, 10.0, 20.0, 20.0, 0.95, 3.0],
                [30.0, 30.0, 40.0, 40.0, 0.5, 7.0],
            ],
            dtype=np.float32,
        )
        boxes_xyxy, scores = _extract_e2e(preds)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
            skip_nms=True,
        )
        assert len(dets) == 2
        assert dets[0][5] == 3
        assert dets[1][5] == 7


class TestDecodeDetectionsTopK:
    """`top_k` returns the runner-up classes the network already scored and this
    function otherwise discards at the argmax (AMBIGUITY_REVIEW_PLAN.md).

    The parity test is the important one: ~40 call sites and every other test in
    this file pass no `top_k` and must keep receiving a bare array.
    """

    def _one_box(self, scores):
        return (
            np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32),
            np.array([scores], dtype=np.float32),
        )

    def _decode(self, boxes, scores, **kw):
        return decode_detections(
            boxes,
            scores,
            conf_thresh=0.1,
            iou_thresh=0.5,
            scale=1.0,
            pad=(0, 0),
            original_shape=(100, 100),
            **kw,
        )

    def test_top_k_zero_returns_a_bare_array_identical_to_before(self):
        boxes, scores = self._one_box([0.4, 0.9, 0.1, 0.85])
        plain = self._decode(boxes, scores)
        with_k = self._decode(boxes, scores, top_k=3)[0]
        assert isinstance(plain, np.ndarray)
        assert np.array_equal(plain, with_k)

    def test_alternates_are_ranked_best_first_and_exclude_the_winner(self):
        # winner = class 1 (0.9); runners-up = class 3 (0.85) then class 0 (0.4)
        boxes, scores = self._one_box([0.4, 0.9, 0.1, 0.85])
        dets, alternates = self._decode(boxes, scores, top_k=3)

        assert int(dets[0][5]) == 1
        assert alternates.shape == (1, 2, 2)
        assert [int(c) for c in alternates[0][:, 0]] == [3, 0]
        assert alternates[0][:, 1] == pytest.approx([0.85, 0.4])

    def test_alternate_count_is_capped_by_the_number_of_classes(self):
        """Asking for more classes than the model has must not fabricate rows."""
        boxes, scores = self._one_box([0.9, 0.8])
        _, alternates = self._decode(boxes, scores, top_k=8)
        assert alternates.shape == (1, 1, 2)

    def test_single_class_model_yields_no_alternates(self):
        boxes, scores = self._one_box([0.9])
        dets, alternates = self._decode(boxes, scores, top_k=3)
        assert len(dets) == 1
        assert alternates.shape == (1, 0, 2)

    def test_no_detections_still_returns_a_pair(self):
        """Callers unpack two values whenever they asked for top_k; the empty
        path must not break that."""
        boxes, scores = self._one_box([0.01, 0.02])
        dets, alternates = self._decode(boxes, scores, top_k=3)
        assert len(dets) == 0
        assert alternates.shape[0] == 0

    def test_nms_free_models_report_no_alternates_rather_than_failing(self):
        """An E2E export's `scores` is [confidence, class_id] — already argmaxed
        inside the graph, so there is no per-class matrix to rank. It must return
        empty alternates, not raise and not misread the class_id column."""
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
        scores = np.array([[0.9, 3.0]], dtype=np.float32)  # confidence, class_id
        dets, alternates = self._decode(boxes, scores, skip_nms=True, top_k=3)
        assert int(dets[0][5]) == 3
        assert alternates.shape == (1, 0, 2)

    def test_alternates_align_row_for_row_with_surviving_detections(self):
        """After NMS drops boxes, each alternates row must still belong to the
        detection at the same index — a misalignment here would silently attribute
        one box's confusion to another."""
        boxes = np.array(
            [[10.0, 10.0, 50.0, 50.0], [11.0, 11.0, 51.0, 51.0], [80.0, 80.0, 95.0, 95.0]],
            dtype=np.float32,
        )
        # rows 0/1 overlap (NMS keeps the stronger); row 2 is far away.
        scores = np.array([[0.9, 0.85, 0.0], [0.5, 0.4, 0.0], [0.0, 0.2, 0.95]], dtype=np.float32)
        dets, alternates = self._decode(boxes, scores, top_k=2)

        assert len(dets) == len(alternates)
        for det, alt in zip(dets, alternates):
            winner_class = int(det[5])
            # The winner never appears among its own runners-up.
            assert winner_class not in [int(c) for c in alt[:, 0]]
            # And every runner-up scored no higher than the winner.
            assert all(s <= det[4] + 1e-6 for s in alt[:, 1])
