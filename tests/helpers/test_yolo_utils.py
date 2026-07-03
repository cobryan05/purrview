"""Unit tests for YOLO ONNX decode — apps/helpers/yoloUtils.py::postprocess.

Pure numpy: no I/O, no ONNX session, no network.

Covers the two raw-output layouts actually produced by models this app loads:
  - "cxcywh" + raw logits  — real ultralytics YOLOv8 exports (yolo: 8)
  - "xyxy" + pre-sigmoided — libreyolo9 exports (yolo: 9)

Both layouts are (num_preds, 4 + nc): 4 box channels + nc class-score channels,
with NO separate objectness channel.
"""

import numpy as np
import pytest

from apps.helpers.yoloUtils import postprocess


def _logit(p: float) -> float:
    """Inverse sigmoid — turn a desired post-sigmoid probability into a raw logit."""
    return float(np.log(p / (1 - p)))


class TestPostprocessLibreYolo9Format:
    """box_format='xyxy', scores_are_logits=False — libreyolo9 (yolo: 9) exports."""

    def test_decodes_correct_box_and_class(self):
        # One box for class 1 (of 2 classes), already xyxy in canvas-pixel space,
        # scores already sigmoid-activated (as libreyolo9's own decode assumes).
        preds = np.array(
            [[[100.0, 100.0, 200.0, 200.0, 0.05, 0.92]]],  # [x1, y1, x2, y2, class0, class1]
            dtype=np.float32,
        )
        dets = postprocess(
            preds,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
            box_format="xyxy",
            scores_are_logits=False,
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
            [[[100.0, 100.0, 200.0, 200.0, 0.05, 0.92]]],
            dtype=np.float32,
        )
        dets = postprocess(
            preds,
            conf_thresh=0.7,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
            box_format="xyxy",
            scores_are_logits=False,
        )
        assert len(dets) == 1
        assert dets[0][4] == pytest.approx(0.92, abs=1e-6)

    def test_undoes_padding_and_scale(self):
        preds = np.array(
            [[[110.0, 60.0, 210.0, 160.0, 0.0, 0.9]]],
            dtype=np.float32,
        )
        dets = postprocess(
            preds,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=2.0,
            pad=(10.0, 20.0),
            original_shape=(200, 200),
            box_format="xyxy",
            scores_are_logits=False,
        )
        assert len(dets) == 1
        x1, y1, x2, y2 = dets[0][:4]
        # (110-10)/2=50, (60-20)/2=20, (210-10)/2=100, (160-20)/2=70
        assert (x1, y1, x2, y2) == pytest.approx((50.0, 20.0, 100.0, 70.0), abs=1e-3)

    def test_low_score_filtered_out(self):
        preds = np.array(
            [[[100.0, 100.0, 200.0, 200.0, 0.1, 0.15]]],
            dtype=np.float32,
        )
        dets = postprocess(
            preds,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
            box_format="xyxy",
            scores_are_logits=False,
        )
        assert dets.shape == (0, 6)


class TestPostprocessYolov8Format:
    """box_format='cxcywh', scores_are_logits=True — real ultralytics YOLOv8 exports."""

    def test_decodes_correct_box_and_class(self):
        # cx=150, cy=150, w=100, h=100 -> x1,y1,x2,y2 = 100,100,200,200
        preds = np.array(
            [[[150.0, 150.0, 100.0, 100.0, _logit(0.05), _logit(0.9)]]],
            dtype=np.float32,
        )
        dets = postprocess(
            preds,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
            box_format="cxcywh",
            scores_are_logits=True,
        )
        assert dets.shape == (1, 6)
        x1, y1, x2, y2, confidence, class_id = dets[0]
        assert (x1, y1, x2, y2) == pytest.approx((100.0, 100.0, 200.0, 200.0), abs=1e-3)
        assert class_id == 1
        assert confidence == pytest.approx(0.9, abs=1e-3)

    def test_class_zero_is_reachable(self):
        """class_id 0 must be selectable directly — no more '+1' shift downstream."""
        preds = np.array(
            [[[150.0, 150.0, 100.0, 100.0, _logit(0.9), _logit(0.05)]]],
            dtype=np.float32,
        )
        dets = postprocess(
            preds,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
            box_format="cxcywh",
            scores_are_logits=True,
        )
        assert len(dets) == 1
        assert dets[0][5] == 0


class TestPostprocessNms:
    def test_distinct_non_overlapping_boxes_both_survive_nms(self):
        """cv2.dnn.NMSBoxes expects (x, y, w, h) rects, not (x1, y1, x2, y2) corners.

        These two boxes are far enough from the image origin — and genuinely
        non-overlapping (real IoU = 0) — that feeding raw xyxy corners
        straight into NMSBoxes (misread as x,y,w,h) produces oversized phantom
        rects that spuriously overlap and get one box wrongly suppressed.
        """
        preds = np.array(
            [[[300.0, 300.0, 340.0, 340.0, 0.9], [400.0, 400.0, 440.0, 440.0, 0.8]]],
            dtype=np.float32,
        )
        dets = postprocess(
            preds,
            conf_thresh=0.25,
            iou_thresh=0.1,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
            box_format="xyxy",
            scores_are_logits=False,
        )
        assert len(dets) == 2

    def test_truly_overlapping_boxes_are_still_suppressed(self):
        """Sanity check that the xywh fix doesn't just disable NMS entirely."""
        preds = np.array(
            [[[100.0, 100.0, 200.0, 200.0, 0.9], [110.0, 110.0, 210.0, 210.0, 0.8]]],
            dtype=np.float32,
        )
        dets = postprocess(
            preds,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
            box_format="xyxy",
            scores_are_logits=False,
        )
        assert len(dets) == 1
        assert dets[0][4] == pytest.approx(0.9, abs=1e-6)


class TestPostprocessValidation:
    def test_rejects_unknown_box_format(self):
        # Score above conf_thresh so the box survives filtering and reaches
        # the box_format branch.
        preds = np.array([[[100.0, 100.0, 200.0, 200.0, 0.0, 0.9]]], dtype=np.float32)
        with pytest.raises(ValueError):
            postprocess(
                preds,
                conf_thresh=0.25,
                iou_thresh=0.45,
                scale=1.0,
                pad=(0.0, 0.0),
                original_shape=(640, 640),
                box_format="bogus",
                scores_are_logits=False,
            )

    def test_rejects_too_few_channels(self):
        preds = np.zeros((1, 1, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            postprocess(
                preds,
                conf_thresh=0.25,
                iou_thresh=0.45,
                scale=1.0,
                pad=(0.0, 0.0),
                original_shape=(640, 640),
            )
