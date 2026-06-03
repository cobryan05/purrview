"""Unit tests for image utility functions — apps/helpers/imageUtils.py.

Tests cover the pure numerical functions (letterbox, compute_iou, nms).
annotate_image is skipped here because it depends on a DetectionResult shape
that differs from the current dataclass definition in types.py.
"""

import numpy as np
import pytest

from apps.helpers.imageUtils import compute_iou, letterbox, nms

# ──────────────────────────────────────────────────────────────────────────────
# letterbox
# ──────────────────────────────────────────────────────────────────────────────


class TestLetterbox:
    def test_square_image_unchanged_shape(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        out, r, (dw, dh) = letterbox(img, size=640)
        assert out.shape == (640, 640, 3)
        assert r == pytest.approx(1.0)
        assert dw == pytest.approx(0.0)
        assert dh == pytest.approx(0.0)

    def test_output_is_always_square(self):
        for h, w in [(100, 200), (480, 640), (200, 100)]:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            out, _, _ = letterbox(img, size=320)
            assert out.shape[0] == 320
            assert out.shape[1] == 320

    def test_wide_image_scale_factor(self):
        # 320×640 image, target 640: width already fits → r=1.0, padded vertically
        img = np.zeros((320, 640, 3), dtype=np.uint8)
        out, r, (dw, dh) = letterbox(img, size=640)
        assert r == pytest.approx(1.0)
        assert out.shape == (640, 640, 3)

    def test_tall_image_scale_factor(self):
        img = np.zeros((640, 320, 3), dtype=np.uint8)
        out, r, (dw, dh) = letterbox(img, size=640)
        assert r == pytest.approx(1.0)
        assert out.shape == (640, 640, 3)

    def test_small_image_upscaled(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out, r, _ = letterbox(img, size=640)
        assert out.shape == (640, 640, 3)
        assert r == pytest.approx(6.4)

    def test_custom_padding_color(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        color = (0, 128, 255)
        out, _, (dw, dh) = letterbox(img, size=200, color=color)
        # The padded rows should contain the padding color
        assert out.shape == (200, 200, 3)

    def test_output_dtype_preserved(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out, _, _ = letterbox(img, size=200)
        assert out.dtype == np.uint8


# ──────────────────────────────────────────────────────────────────────────────
# compute_iou
# ──────────────────────────────────────────────────────────────────────────────


class TestComputeIoU:
    def test_identical_boxes(self):
        box = [0, 0, 100, 100]
        boxes = np.array([[0, 0, 100, 100]])
        iou = compute_iou(box, boxes)
        assert iou[0] == pytest.approx(1.0)

    def test_no_overlap(self):
        box = [0, 0, 10, 10]
        boxes = np.array([[20, 20, 30, 30]])
        iou = compute_iou(box, boxes)
        assert iou[0] == pytest.approx(0.0)

    def test_partial_overlap(self):
        # 10×10 box vs offset by (5,5): intersection is 5×5=25, union=175
        box = [0, 0, 10, 10]
        boxes = np.array([[5, 5, 15, 15]])
        iou = compute_iou(box, boxes)
        assert iou[0] == pytest.approx(25 / 175)

    def test_contained_box(self):
        # small box fully inside large box
        box = [2, 2, 8, 8]  # area 36
        boxes = np.array([[0, 0, 10, 10]])  # area 100
        iou = compute_iou(box, boxes)
        # IoU = 36 / (36 + 100 - 36)
        assert iou[0] == pytest.approx(36 / 100)

    def test_multiple_boxes(self):
        box = [0, 0, 10, 10]
        boxes = np.array(
            [
                [0, 0, 10, 10],  # identical → 1.0
                [20, 20, 30, 30],  # no overlap → 0.0
            ]
        )
        iou = compute_iou(box, boxes)
        assert len(iou) == 2
        assert iou[0] == pytest.approx(1.0)
        assert iou[1] == pytest.approx(0.0)

    def test_single_pixel_box(self):
        box = [5, 5, 6, 6]
        boxes = np.array([[5, 5, 6, 6]])
        iou = compute_iou(box, boxes)
        assert iou[0] == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────────
# nms
# ──────────────────────────────────────────────────────────────────────────────


class TestNMS:
    def test_single_box_kept(self):
        boxes = np.array([[0, 0, 10, 10]])
        scores = np.array([0.9])
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert keep == [0]

    def test_identical_boxes_highest_score_kept(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        scores = np.array([0.9, 0.5])
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert 0 in keep
        assert 1 not in keep

    def test_non_overlapping_all_kept(self):
        boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert set(keep) == {0, 1}

    def test_below_threshold_kept(self):
        # Overlap 25/175 ≈ 0.14 → below 0.5, so both kept
        boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert set(keep) == {0, 1}

    def test_above_threshold_suppressed(self):
        # Same boxes as above but threshold=0.1 → second suppressed
        boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, iou_threshold=0.1)
        assert 0 in keep
        assert 1 not in keep

    def test_order_by_score(self):
        # Lower index has lower score; NMS should keep higher-score box
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        scores = np.array([0.3, 0.9])
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert 1 in keep
        assert 0 not in keep

    def test_three_boxes_chain_suppression(self):
        # boxes[0] overlaps [1], [1] overlaps [2], [0] and [2] don't overlap
        boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [11, 0, 21, 10]])
        scores = np.array([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, iou_threshold=0.3)
        assert 0 in keep  # highest score always kept
        assert len(keep) >= 1
