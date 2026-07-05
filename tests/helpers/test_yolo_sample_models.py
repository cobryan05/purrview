"""End-to-end regression tests for the YOLO ONNX inference pipeline.

Unlike test_yolo_utils.py (which calls `_extract`/`decode_detections` directly
with hand-built numpy arrays), these tests build tiny *real* ONNX models with
`onnx.helper` — a fixed-output graph that ignores its input and always returns
one deterministic prediction tensor, shaped exactly like a real ultralytics
YOLOv8/YOLO11 or LibreYOLO9 export (already sigmoid-activated class scores,
transposed (batch, 4+nc, num_preds) layout) — and run them through the real
provider's `processImage()` against a real image array via onnxruntime. This
exercises the whole pipeline: preprocessing, the ONNX session, the
(classes, boxes) transpose, `_extract`, and `decode_detections` together, so
it would have caught the "tons of boxes at 50% confidence" double-sigmoid
regression that a mocked/no-ONNX test could miss.
"""

import asyncio

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from apps.helpers.inferenceProviders.libreYolo9InferenceProvider import (
    LibreYolo9InferenceProvider,
)
from apps.helpers.inferenceProviders.ultralyticsYoloV8InferenceProvider import (
    UltralyticsYoloV8InferenceProvider,
)

_INPUT_SIZE = 32


def _build_fixed_output_onnx(path, output_data: np.ndarray) -> None:
    """Write an ONNX model that ignores its input and always returns `output_data`.

    `output_data` must be shaped (1, 4+nc, num_preds), with num_preds > 4+nc —
    the raw, pre-transpose layout a real ultralytics/libreyolo9 export
    produces, where `onnxYoloInferenceProvider.processImage` decides whether
    to transpose by comparing those two axis sizes (real exports have
    thousands of anchor boxes, so this always holds true in practice).
    """
    input_tensor = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, _INPUT_SIZE, _INPUT_SIZE]
    )
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(output_data.shape))
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output"],
        value=helper.make_tensor(
            "const_val", TensorProto.FLOAT, list(output_data.shape), output_data.flatten().tolist()
        ),
    )
    graph = helper.make_graph([const_node], "fixed_output", [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _dummy_image() -> np.ndarray:
    """A real (H, W, 3) uint8 array — no cv2 preprocessing edge case skipped."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


class TestUltralyticsYoloV8SampleModel:
    """Real ultralytics YOLOv8/YOLO11 exports already sigmoid-activate class
    scores in-graph (confirmed against real yolov8x.onnx/Yolo11S.onnx: raw
    score channels are already in [0, 1]) — this fixture reproduces that."""

    def _build(self, tmp_path):
        # 10 candidate anchors so shape[1] (=6) < shape[2] (=10), matching a
        # real export's (batch, 4+nc, num_anchors) layout and triggering the
        # provider's transpose-detection the same way real models do.
        num_preds = 10
        data = np.zeros((1, 6, num_preds), dtype=np.float32)
        # Anchor 0: cx=16, cy=16, w=8, h=8 (canvas-pixel, 32x32 input) ->
        # xyxy (12,12,20,20). Scores already sigmoid-activated: class 0 =
        # 0.05, class 1 = 0.9. The other 9 anchors are all-zero boxes with
        # near-zero scores (background), like a real model's low-confidence
        # anchors, and must be filtered out rather than surviving as
        # phantom detections.
        data[0, :, 0] = [16.0, 16.0, 8.0, 8.0, 0.05, 0.9]
        path = tmp_path / "fixed_yolov8.onnx"
        _build_fixed_output_onnx(path, data)
        return str(path)

    def test_confidence_is_not_squashed_toward_half(self, tmp_path):
        model_path = self._build(tmp_path)
        provider = UltralyticsYoloV8InferenceProvider(
            model_path, class_names=["cat", "dog"], conf_thresh=0.25, providers=["CPUExecutionProvider"]
        )
        result = asyncio.run(provider.processImage(_dummy_image()))

        assert len(result.detections) == 1
        det = result.detections[0]
        # The regression symptom was every detection landing at ~0.5 no matter
        # the model's real prediction — assert we recover the true 0.9 score.
        assert det.confidence == pytest.approx(0.9, abs=1e-3)
        assert det.bbox.class_str == "dog"

    def test_low_score_class_is_filtered(self, tmp_path):
        model_path = self._build(tmp_path)
        provider = UltralyticsYoloV8InferenceProvider(
            model_path, class_names=["cat", "dog"], conf_thresh=0.7, providers=["CPUExecutionProvider"]
        )
        result = asyncio.run(provider.processImage(_dummy_image()))
        # Only one detection (the argmax'd class per box) should ever be emitted,
        # never phantom secondary boxes from the low-scoring class.
        assert len(result.detections) == 1


class TestLibreYolo9SampleModel:
    """Real LibreYOLO9 exports (verified against a live model.export('onnx') of
    LibreYOLO9t.pt) already return xyxy canvas-pixel boxes and pre-sigmoided
    scores — no further activation should be applied."""

    def _build(self, tmp_path):
        num_preds = 10
        data = np.zeros((1, 6, num_preds), dtype=np.float32)
        # Already xyxy canvas-pixel space; scores already sigmoid-activated.
        data[0, :, 0] = [12.0, 12.0, 20.0, 20.0, 0.05, 0.9]
        path = tmp_path / "fixed_libreyolo9.onnx"
        _build_fixed_output_onnx(path, data)
        return str(path)

    def test_confidence_is_not_squashed_toward_half(self, tmp_path):
        model_path = self._build(tmp_path)
        provider = LibreYolo9InferenceProvider(
            model_path, class_names=["cat", "dog"], conf_thresh=0.25, providers=["CPUExecutionProvider"]
        )
        result = asyncio.run(provider.processImage(_dummy_image()))

        assert len(result.detections) == 1
        det = result.detections[0]
        assert det.confidence == pytest.approx(0.9, abs=1e-3)
        assert det.bbox.class_str == "dog"
