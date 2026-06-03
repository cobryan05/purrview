"""Unit tests for BBox utility class — apps/helpers/bboxUtils.py.

All tests are pure arithmetic: no I/O, no network, no database.
"""

import pytest

from apps.helpers.bboxUtils import BBox

# ──────────────────────────────────────────────────────────────────────────────
# Factory constructors
# ──────────────────────────────────────────────────────────────────────────────


class TestFromX1Y1X2Y2:
    def test_basic(self):
        b = BBox.fromX1Y1X2Y2(100, 0, 200, 50, 1000, 500)
        x1, y1, w, h = b.asRX1Y1WH()
        assert pytest.approx(x1) == 0.1
        assert pytest.approx(y1) == 0.0
        assert pytest.approx(w) == 0.1
        assert pytest.approx(h) == 0.1

    def test_full_image(self):
        b = BBox.fromX1Y1X2Y2(0, 0, 640, 480, 640, 480)
        assert pytest.approx(b.area) == 1.0

    def test_single_pixel(self):
        b = BBox.fromX1Y1X2Y2(10, 10, 11, 11, 100, 100)
        assert pytest.approx(b.area) == 0.0001


class TestFromX1Y1WH:
    def test_basic(self):
        b = BBox.fromX1Y1WH(50, 25, 100, 50, 500, 500)
        x1, y1, w, h = b.asRX1Y1WH()
        assert pytest.approx(x1) == 0.1
        assert pytest.approx(y1) == 0.05
        assert pytest.approx(w) == 0.2
        assert pytest.approx(h) == 0.1


class TestFromYolo:
    def test_basic(self):
        # center (0.5, 0.5), size (0.2, 0.1) → x1=0.4, y1=0.45
        b = BBox.fromYolo(0.5, 0.5, 0.2, 0.1)
        x1, y1, w, h = b.asRX1Y1WH()
        assert pytest.approx(x1) == 0.4
        assert pytest.approx(y1) == 0.45
        assert pytest.approx(w) == 0.2
        assert pytest.approx(h) == 0.1

    def test_center_is_reconstructed(self):
        cX, cY, w, h = 0.3, 0.7, 0.4, 0.2
        b = BBox.fromYolo(cX, cY, w, h)
        back_cX, back_cY, back_w, back_h = b.asYolo()
        assert pytest.approx(back_cX) == cX
        assert pytest.approx(back_cY) == cY


class TestFromRX1Y1WH:
    def test_passthrough(self):
        b = BBox.fromRX1Y1WH(0.1, 0.2, 0.3, 0.4)
        x1, y1, w, h = b.asRX1Y1WH()
        assert pytest.approx(x1) == 0.1
        assert pytest.approx(y1) == 0.2
        assert pytest.approx(w) == 0.3
        assert pytest.approx(h) == 0.4


class TestFromRX1Y1X2Y2:
    def test_basic(self):
        b = BBox.fromRX1Y1X2Y2(0.1, 0.2, 0.5, 0.7)
        x1, y1, w, h = b.asRX1Y1WH()
        assert pytest.approx(x1) == 0.1
        assert pytest.approx(y1) == 0.2
        assert pytest.approx(w) == 0.4
        assert pytest.approx(h) == 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Negative-size normalisation (constructor flips negative w/h)
# ──────────────────────────────────────────────────────────────────────────────


class TestNegativeDimensions:
    def test_negative_width_flipped(self):
        b = BBox((0.5, 0.1, -0.2, 0.3))
        x1, y1, w, h = b.asRX1Y1WH()
        assert pytest.approx(x1) == 0.3
        assert pytest.approx(w) == 0.2
        assert w >= 0

    def test_negative_height_flipped(self):
        b = BBox((0.1, 0.5, 0.2, -0.3))
        x1, y1, w, h = b.asRX1Y1WH()
        assert pytest.approx(y1) == 0.2
        assert pytest.approx(h) == 0.3
        assert h >= 0

    def test_both_negative(self):
        b = BBox((0.6, 0.8, -0.2, -0.3))
        x1, y1, w, h = b.asRX1Y1WH()
        assert w >= 0
        assert h >= 0


# ──────────────────────────────────────────────────────────────────────────────
# Round-trip conversions
# ──────────────────────────────────────────────────────────────────────────────


class TestConversionRoundTrips:
    def test_x1y1x2y2_roundtrip(self):
        b = BBox.fromX1Y1X2Y2(10, 20, 110, 70, 200, 100)
        x1, y1, x2, y2 = b.asX1Y1X2Y2(200, 100)
        assert x1 == 10
        assert y1 == 20
        assert x2 == 110
        assert y2 == 70

    def test_x1y1wh_roundtrip(self):
        b = BBox.fromX1Y1WH(10, 20, 100, 50, 200, 200)
        x1, y1, w, h = b.asX1Y1WH(200, 200)
        assert x1 == 10
        assert y1 == 20
        assert w == 100
        assert h == 50

    def test_yolo_roundtrip(self):
        b = BBox.fromYolo(0.5, 0.5, 0.2, 0.4)
        cX, cY, w, h = b.asYolo()
        assert pytest.approx(cX) == 0.5
        assert pytest.approx(cY) == 0.5
        assert pytest.approx(w) == 0.2
        assert pytest.approx(h) == 0.4

    def test_rx1y1x2y2(self):
        b = BBox.fromRX1Y1WH(0.1, 0.2, 0.3, 0.4)
        x1, y1, x2, y2 = b.asRX1Y1X2Y2()
        assert pytest.approx(x2) == 0.4
        assert pytest.approx(y2) == 0.6


# ──────────────────────────────────────────────────────────────────────────────
# Area
# ──────────────────────────────────────────────────────────────────────────────


class TestArea:
    def test_normal(self):
        b = BBox.fromRX1Y1WH(0.0, 0.0, 0.5, 0.4)
        assert pytest.approx(b.area) == 0.2

    def test_zero_width(self):
        b = BBox.fromRX1Y1WH(0.0, 0.0, 0.0, 0.5)
        assert b.area == 0.0

    def test_zero_height(self):
        b = BBox.fromRX1Y1WH(0.0, 0.0, 0.5, 0.0)
        assert b.area == 0.0

    def test_point(self):
        b = BBox.fromRX1Y1WH(0.5, 0.5, 0.0, 0.0)
        assert b.area == 0.0

    def test_full(self):
        b = BBox.fromRX1Y1WH(0.0, 0.0, 1.0, 1.0)
        assert pytest.approx(b.area) == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Distance
# ──────────────────────────────────────────────────────────────────────────────


class TestDist:
    def test_same_origin(self):
        b = BBox.fromRX1Y1WH(0.0, 0.0, 0.1, 0.1)
        assert pytest.approx(b.dist(b)) == 0.0

    def test_known_distance(self):
        # (0,0) vs (0.3, 0.4) → Pythagorean 0.5
        b1 = BBox.fromRX1Y1WH(0.0, 0.0, 0.0, 0.0)
        b2 = BBox.fromRX1Y1WH(0.3, 0.4, 0.0, 0.0)
        assert pytest.approx(b1.dist(b2)) == 0.5

    def test_symmetry(self):
        b1 = BBox.fromRX1Y1WH(0.1, 0.1, 0.2, 0.2)
        b2 = BBox.fromRX1Y1WH(0.5, 0.6, 0.1, 0.1)
        assert pytest.approx(b1.dist(b2)) == pytest.approx(b2.dist(b1))


# ──────────────────────────────────────────────────────────────────────────────
# Similarity / equality
# ──────────────────────────────────────────────────────────────────────────────


class TestSimilar:
    def test_identical_is_similar(self):
        b1 = BBox.fromRX1Y1WH(0.1, 0.1, 0.3, 0.3)
        b2 = BBox.fromRX1Y1WH(0.1, 0.1, 0.3, 0.3)
        assert b1.similar(b2)

    def test_far_apart_not_similar(self):
        b1 = BBox.fromRX1Y1WH(0.0, 0.0, 0.1, 0.1)
        b2 = BBox.fromRX1Y1WH(0.9, 0.9, 0.1, 0.1)
        assert not b1.similar(b2)

    def test_eq_operator(self):
        b1 = BBox.fromRX1Y1WH(0.1, 0.1, 0.2, 0.2)
        b2 = BBox.fromRX1Y1WH(0.1, 0.1, 0.2, 0.2)
        assert b1 == b2

    def test_same_object_similar(self):
        b = BBox.fromRX1Y1WH(0.1, 0.2, 0.3, 0.4)
        assert b.similar(b)


# ──────────────────────────────────────────────────────────────────────────────
# Copy
# ──────────────────────────────────────────────────────────────────────────────


class TestCopy:
    def test_copy_is_equal(self):
        b1 = BBox.fromRX1Y1WH(0.1, 0.2, 0.3, 0.4)
        b2 = b1.copy()
        assert b1 == b2

    def test_copy_is_independent(self):
        b1 = BBox.fromRX1Y1WH(0.1, 0.2, 0.3, 0.4)
        b2 = b1.copy()
        b2.bbox[0] = 0.9
        assert b1.bbox[0] != b2.bbox[0]
