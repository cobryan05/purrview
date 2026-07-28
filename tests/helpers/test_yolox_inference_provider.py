"""Unit tests for YoloxInferenceProvider (apps/helpers/inferenceProviders/
yoloxInferenceProvider.py).

Pure numpy: no ONNX session, no torch/yolox import — this is exactly the
decode math that runs in the pure-ONNX inference container at runtime.

Covers:
  - `_build_grids_and_strides` — the stride-grid layout (P3/P4/P5, 8/16/32),
    confirmed to match yolox/utils/demo_utils.py::demo_postprocess's grid
    construction and concatenation order (smallest stride/largest feature
    map first).
  - `_extract` — grid-relative box decode (`(cx+grid)*stride`,
    `exp(w)*stride`) + separate-objectness scoring (`objectness *
    class_score`), the one genuinely distinct YOLOX layout in this codebase.
  - Registration under the "yolox" family key and `load_yolo_onnx` dispatch.
"""

import json

import numpy as np
import pytest

# Importing this triggers @register_inference_provider registration.
import apps.helpers.inferenceProviders.yoloxInferenceProvider  # noqa: F401
from apps.helpers.inferenceProviders.Registry import inference_provider_registry
from apps.helpers.inferenceProviders.yoloxInferenceProvider import (
    YoloxInferenceProvider,
    _build_grids_and_strides,
)
from apps.helpers.yoloUtils import decode_detections


class _FakeProvider:
    """Stand-in exposing only what `_extract` reads (`self.input_shape`) —
    avoids constructing a real ONNX session in these pure-decode-math tests."""

    def __init__(self, input_shape=(640, 640)):
        self.input_shape = input_shape


def _extract(preds: np.ndarray, input_shape=(640, 640)):
    return YoloxInferenceProvider._extract(_FakeProvider(input_shape), preds)


class TestBuildGridsAndStrides:
    def test_total_anchor_count_matches_standard_640_head(self):
        # 640/8=80 -> 6400, 640/16=40 -> 1600, 640/32=20 -> 400. Total 8400 —
        # the well-known YOLOX-640 anchor count (matches every public YOLOX
        # ONNX export's raw output shape).
        grids, strides = _build_grids_and_strides(640, 640)
        assert grids.shape == (8400, 2)
        assert strides.shape == (8400, 1)

    def test_concatenation_order_is_p3_then_p4_then_p5(self):
        grids, strides = _build_grids_and_strides(640, 640)
        # First 6400 rows belong to stride-8 (P3), next 1600 to stride-16 (P4),
        # last 400 to stride-32 (P5) — must match the model's own head
        # flatten/concat order for the decode to land on the right cells.
        assert np.all(strides[:6400] == 8)
        assert np.all(strides[6400:8000] == 16)
        assert np.all(strides[8000:8400] == 32)
        # First row of each block is grid cell (0, 0).
        assert tuple(grids[0]) == (0.0, 0.0)
        assert tuple(grids[6400]) == (0.0, 0.0)
        assert tuple(grids[8000]) == (0.0, 0.0)


class TestExtractYoloxFormat:
    """`_extract` — grid-relative boxes decoded via stride grids, plus
    objectness * class_score scoring (YOLOX's separate-objectness layout)."""

    def test_decodes_known_box_at_known_grid_cell_stride16(self):
        n_anchors = 8400
        nc = 3
        preds = np.zeros((n_anchors, 5 + nc), dtype=np.float32)

        # Pick a cell in the stride-16 (P4) block: feature map is 40x40, row
        # index within that block for grid cell (gx=5, gy=3) is gy*40+gx = 125,
        # offset by the preceding 6400 stride-8 rows -> absolute row 6525.
        stride = 16
        gx, gy = 5, 3
        row = 6400 + gy * 40 + gx

        # Raw cx,cy offsets within the cell (0.5, 0.5 = cell center), raw w,h
        # in log-scale such that exp(w)*stride = 32 (i.e. w = log(32/16)).
        cx_off, cy_off = 0.5, 0.5
        target_w_px, target_h_px = 32.0, 48.0
        preds[row, 0] = cx_off
        preds[row, 1] = cy_off
        preds[row, 2] = np.log(target_w_px / stride)
        preds[row, 3] = np.log(target_h_px / stride)
        objectness = 0.8
        class_score = 0.5
        preds[row, 4] = objectness
        preds[row, 5 + 1] = class_score  # class index 1

        boxes_xyxy, scores = _extract(preds)

        expected_cx = (gx + cx_off) * stride  # (5.5)*16 = 88
        expected_cy = (gy + cy_off) * stride  # (3.5)*16 = 56
        expected_x1 = expected_cx - target_w_px / 2
        expected_y1 = expected_cy - target_h_px / 2
        expected_x2 = expected_cx + target_w_px / 2
        expected_y2 = expected_cy + target_h_px / 2

        assert boxes_xyxy[row] == pytest.approx([expected_x1, expected_y1, expected_x2, expected_y2], abs=1e-3)
        # Final score = objectness * class_score (separate-objectness layout).
        assert scores[row, 1] == pytest.approx(objectness * class_score, abs=1e-6)
        # Every other class column for this row stays at objectness * 0 = 0.
        assert scores[row, 0] == pytest.approx(0.0, abs=1e-6)
        assert scores[row, 2] == pytest.approx(0.0, abs=1e-6)

    def test_full_pipeline_through_decode_detections_lands_on_right_pixel_and_score(self):
        """End-to-end: _extract -> decode_detections (NMS + rescale/clip)
        with scale=1.0/no padding, asserting the returned detection's xyxy
        and confidence match objectness*class_score exactly."""
        n_anchors = 8400
        nc = 2
        preds = np.zeros((n_anchors, 5 + nc), dtype=np.float32)

        stride = 32
        gx, gy = 2, 1
        row = 8000 + gy * 20 + gx  # P5 block starts at row 8000 (20x20 feature map)

        preds[row, 0] = 0.0  # cx offset = 0 -> cell's top-left corner
        preds[row, 1] = 0.0
        preds[row, 2] = np.log(64.0 / stride)
        preds[row, 3] = np.log(64.0 / stride)
        objectness = 0.9
        class_score = 0.7
        preds[row, 4] = objectness
        preds[row, 5] = class_score  # class 0

        boxes_xyxy, scores = _extract(preds)
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
        expected_cx = gx * stride
        expected_cy = gy * stride
        assert (x1, y1, x2, y2) == pytest.approx(
            (expected_cx - 32.0, expected_cy - 32.0, expected_cx + 32.0, expected_cy + 32.0), abs=1e-3
        )
        assert class_id == 0
        assert confidence == pytest.approx(objectness * class_score, abs=1e-6)

    def test_rejects_too_few_channels(self):
        preds = np.zeros((1, 5), dtype=np.float32)
        with pytest.raises(ValueError):
            _extract(preds)

    def test_grid_anchor_count_mismatch_raises(self):
        # 100 rows can't correspond to any valid 3-stride 640x640 head.
        preds = np.zeros((100, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="Grid/anchor count mismatch"):
            _extract(preds)

    def test_low_score_filtered_out_by_conf_thresh(self):
        n_anchors = 8400
        nc = 1
        preds = np.zeros((n_anchors, 5 + nc), dtype=np.float32)
        row = 0
        preds[row, 2] = np.log(32.0 / 8)
        preds[row, 3] = np.log(32.0 / 8)
        preds[row, 4] = 0.1  # low objectness
        preds[row, 5] = 0.1  # low class score -> combined 0.01

        boxes_xyxy, scores = _extract(preds)
        dets = decode_detections(
            boxes_xyxy,
            scores,
            conf_thresh=0.25,
            iou_thresh=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            original_shape=(640, 640),
        )
        assert len(dets) == 0

    def test_no_double_sigmoid_objectness_and_class_scores_used_as_is(self):
        """Guards against re-applying sigmoid() to already-activated
        objectness/class-score channels — YOLOX's head sigmoids both before
        the graph returns (yolox/models/yolo_head.py inference path)."""
        n_anchors = 8400
        nc = 1
        preds = np.zeros((n_anchors, 5 + nc), dtype=np.float32)
        row = 0
        preds[row, 2] = np.log(32.0 / 8)
        preds[row, 3] = np.log(32.0 / 8)
        objectness = 0.95
        class_score = 0.9
        preds[row, 4] = objectness
        preds[row, 5] = class_score

        _, scores = _extract(preds)
        assert scores[row, 0] == pytest.approx(objectness * class_score, abs=1e-6)


class TestRegistration:
    def test_yolox_registered(self):
        assert inference_provider_registry["yolox"] is YoloxInferenceProvider

    def test_yolox_skip_nms_is_false(self):
        """YOLOX is not an NMS-free/end-to-end export family — NMS must
        still run (unlike ultralytics_e2e's YOLOv10/YOLO26)."""
        assert YoloxInferenceProvider.SKIP_NMS is False

    def test_letterbox_anchor_is_top_left(self):
        assert YoloxInferenceProvider.LETTERBOX_ANCHOR == "top_left"


class TestLoadYoloOnnxDispatchesYolox:
    def test_yolox_family_resolves_without_raising(self, tmp_path):
        """Sidecar-driven dispatch: model_family="yolox" must resolve to
        YoloxInferenceProvider via load_yolo_onnx's registry lookup, without
        needing a real ONNX session (constructing a real one is exercised
        elsewhere; this only proves the family string resolves to the right
        class, catching the classic 'forgot to import the provider module in
        load_yolo_onnx' regression)."""
        from apps.helpers.yoloUtils import load_yolo_onnx

        model_path = tmp_path / "m.onnx"
        model_path.write_bytes(b"not a real onnx file")
        (tmp_path / "m.json").write_text(
            json.dumps({"model_family": "yolox", "classes": ["a", "b"], "model_kind": "object_detection"})
        )
        # A garbage ONNX file fails at ort.InferenceSession construction
        # (inside YoloxInferenceProvider.__init__), not at the registry
        # lookup — which is the thing this test actually verifies. Any
        # exception other than the "Unhandled model family" one confirms
        # dispatch succeeded and got as far as trying to build the provider.
        with pytest.raises(Exception) as exc_info:
            load_yolo_onnx(str(model_path))
        assert "Unhandled model family" not in str(exc_info.value)
