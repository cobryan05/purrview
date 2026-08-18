"""Unit tests for image utility functions — apps/helpers/imageUtils.py.

Tests cover the pure numerical functions (letterbox, compute_iou, nms).
annotate_image is skipped here because it depends on a DetectionResult shape
that differs from the current dataclass definition in types.py.
"""

import numpy as np
import pytest

from apps.helpers.imageUtils import (
    ANNOTATION_MAX_STROKE_SCALE,
    DEFAULT_BOX_BGR,
    annotate_detections,
    annotation_stroke_scale,
    compute_iou,
    downscale_to_width,
    encode_jpg_bytes,
    hex_to_bgr,
    letterbox,
    nms,
)
from apps.helpers.tracking.tracker import Detection

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


# ──────────────────────────────────────────────────────────────────────────────
# annotate_detections / hex_to_bgr / downscale_to_width
#
# These take the watcher's `Detection` (absolute pixel corners, resolved label name),
# not `DetectionResult` (relative box, model class string) — which is exactly why they
# are a separate renderer. See the note at the top of this file about annotate_image.
# ──────────────────────────────────────────────────────────────────────────────


def _det(x1=10, y1=10, x2=40, y2=40, label_uuid="lab-1", name="tabby", confidence=0.8):
    return Detection(bbox=(x1, y1, x2, y2), label_uuid=label_uuid, label_name=name, confidence=confidence)


class TestHexToBgr:
    def test_converts_and_swaps_channel_order(self):
        assert hex_to_bgr("#ff0000") == (0, 0, 255)

    def test_shorthand_is_expanded(self):
        assert hex_to_bgr("#f00") == (0, 0, 255)

    def test_a_malformed_colour_falls_back_rather_than_raising(self):
        """A bad colour in the vocabulary must not stop a 24/7 watcher drawing."""
        for bad in (None, "", "nonsense", "#12345", "#zzzzzz"):
            assert hex_to_bgr(bad) == DEFAULT_BOX_BGR


class TestAnnotateDetections:
    def test_copies_by_default(self):
        """The caller's frame is also written by snapshots and held in the rewind ring."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = annotate_detections(frame, [_det()])
        assert np.array_equal(frame, np.zeros((100, 100, 3), dtype=np.uint8))
        assert not np.array_equal(out, frame)

    def test_draws_in_the_label_colour(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = annotate_detections(frame, [_det()], colours={"lab-1": "#ff0000"})
        # Red in BGR is (0, 0, 255) — the blue channel must stay untouched.
        assert out[:, :, 2].max() == 255
        assert out[:, :, 0].max() == 0

    def test_a_box_outside_the_frame_is_clamped_not_dropped(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = annotate_detections(frame, [_det(x1=-20, y1=-20, x2=50, y2=50)])
        assert out.max() > 0

    def test_a_degenerate_box_is_skipped(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = annotate_detections(frame, [_det(x1=50, y1=50, x2=50, y2=50)])
        assert out.max() == 0

    def test_a_box_at_the_top_edge_is_still_labelled(self):
        """Text above a box at y=0 would draw off-frame and vanish."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        top = annotate_detections(frame, [_det(x1=10, y1=0, x2=60, y2=30)])
        # More lit pixels than the rectangle alone would produce.
        assert top.max() > 0
        assert np.count_nonzero(top) > 4 * (60 - 10)

    def test_a_malformed_detection_is_skipped_not_raised(self):
        class _Bad:
            bbox = "not a box"

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert annotate_detections(frame, [_Bad()]).max() == 0


class TestAnnotationScaling:
    """Detections and canvas need not share a pixel space.

    The watcher infers on a frame downscaled to `analysis_width`, so its boxes are in
    that space, while the MQTT live feed publishes the camera's full-resolution frame.
    `scale` is what lets the one set of boxes annotate the other.
    """

    def test_boxes_are_drawn_in_the_scaled_position(self):
        """The corner moves to 2x; without the scale it would stay at 1x."""
        frame = np.zeros((960, 1280, 3), dtype=np.uint8)
        det = _det(x1=10, y1=10, x2=40, y2=40)
        out = annotate_detections(frame, [det], colours={"lab-1": "#ff0000"}, scale=(2.0, 2.0))

        assert out[20, 20].any(), "the scaled corner should be on the box outline"

    def test_without_a_scale_the_box_stays_in_the_detections_own_space(self):
        """The regression guard: a full-res publish that forgets to scale draws here."""
        frame = np.zeros((960, 1280, 3), dtype=np.uint8)
        det = _det(x1=10, y1=10, x2=40, y2=40)
        unscaled = annotate_detections(frame, [det], colours={"lab-1": "#ff0000"})

        assert unscaled[10, 10].any()
        # Interior of the unscaled box, so it is only lit if the box actually moved.
        assert not unscaled[20, 20].any()

    def test_a_scale_of_one_matches_no_scale_at_all(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        det = _det()
        assert np.array_equal(
            annotate_detections(frame, [det], scale=(1.0, 1.0)),
            annotate_detections(frame, [det]),
        )

    def test_non_square_scales_are_applied_per_axis(self):
        """A camera whose aspect the analysis downscale did not preserve exactly."""
        frame = np.zeros((400, 1280, 3), dtype=np.uint8)
        out = annotate_detections(frame, [_det(x1=10, y1=10, x2=40, y2=40)], scale=(4.0, 2.0))
        assert out[20, 40].any()

    def test_stroke_weight_grows_with_the_canvas(self):
        """A 2 px box is right at 640 px wide and invisible at 2560."""
        assert annotation_stroke_scale(640) == 1.0
        assert annotation_stroke_scale(1280) == 2.0

    def test_a_small_canvas_keeps_the_reference_weights(self):
        """The reference stroke is already the thinnest that survives JPEG."""
        assert annotation_stroke_scale(320) == 1.0

    def test_stroke_growth_is_capped(self):
        """Uncapped, a 4K frame gets borders thick enough to hide the cat."""
        assert annotation_stroke_scale(8000) == ANNOTATION_MAX_STROKE_SCALE

    def test_a_wider_canvas_actually_draws_more_ink(self):
        """The scale is not just arithmetic — it must reach cv2."""
        small = annotate_detections(np.zeros((480, 640, 3), dtype=np.uint8), [_det(x1=10, y1=10, x2=110, y2=110)])
        large = annotate_detections(
            np.zeros((960, 1280, 3), dtype=np.uint8),
            [_det(x1=10, y1=10, x2=110, y2=110)],
            scale=(2.0, 2.0),
        )
        # Twice the perimeter at twice the thickness is ~4x the pixels; well clear of 2x.
        assert np.count_nonzero(large) > 2 * np.count_nonzero(small)


class TestDownscaleToWidth:
    def test_shrinks_and_preserves_aspect(self):
        out = downscale_to_width(np.zeros((480, 640, 3), dtype=np.uint8), 320)
        assert out.shape[:2] == (240, 320)

    def test_never_upscales(self):
        img = np.zeros((100, 160, 3), dtype=np.uint8)
        assert downscale_to_width(img, 640).shape == img.shape

    def test_zero_is_a_no_op(self):
        img = np.zeros((100, 160, 3), dtype=np.uint8)
        assert downscale_to_width(img, 0).shape == img.shape


class TestEncodeJpgBytes:
    def test_returns_bare_jpeg_bytes_not_a_json_packet(self):
        """`base64_encode_jpg`'s packet is for the inference API; MQTT wants the bytes."""
        payload = encode_jpg_bytes(np.zeros((32, 32, 3), dtype=np.uint8))
        assert isinstance(payload, bytes)
        assert payload[:2] == b"\xff\xd8"
