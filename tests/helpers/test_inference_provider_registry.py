"""Unit tests for the inference provider registry + legacy-sidecar resolution.

Covers apps/helpers/metadataUtils.py::resolve_legacy_model_family and
apps/helpers/yoloUtils.py::load_yolo_onnx's family-lookup dispatch, against
the four sidecar shapes that actually exist on disk (see
INFERENCE_PROVIDER_REGISTRY_PLAN.md):
  - yolov8x.json-style: {"yolo": 8}, no model_family key
  - freshly-trained object_detection_trainer style: {"yolo": 9, "model_family": "libreyolo9"}
  - yolov5.json-style: {"yolo": 5}, no model_family key
  - yolov7.json-style: {"yolo": 7}, no model_family key

No ONNX I/O: this only exercises metadata resolution and registry lookup, not
provider construction (which would require a real ONNX session).
"""

import json

import pytest

# Importing these triggers @register_inference_provider registration.
import apps.helpers.inferenceProviders.libreYolo9InferenceProvider  # noqa: F401
import apps.helpers.inferenceProviders.ultralyticsYoloV8InferenceProvider  # noqa: F401
from apps.helpers.inferenceProviders.libreYolo9InferenceProvider import (
    LibreYolo9InferenceProvider,
)
from apps.helpers.inferenceProviders.Registry import inference_provider_registry
from apps.helpers.inferenceProviders.ultralyticsYoloV8InferenceProvider import (
    UltralyticsYoloV8InferenceProvider,
)
from apps.helpers.metadataUtils import resolve_legacy_model_family
from apps.helpers.yoloUtils import load_yolo_onnx


class TestResolveLegacyModelFamily:
    def test_yolo_8_no_model_family_key(self):
        """yolov8x.json / cats_yolov8.json shape."""
        assert resolve_legacy_model_family({"yolo": 8}) == "ultralytics_yolov8"

    def test_yolo_9_maps_to_libreyolo9(self):
        assert resolve_legacy_model_family({"yolo": 9}) == "libreyolo9"

    def test_yolo_5_maps_to_dead_family(self):
        """yolov5.json shape — stays unloadable per plan choice (a)."""
        assert resolve_legacy_model_family({"yolo": 5}) == "ultralytics_yolov5"

    def test_yolo_7_maps_to_dead_family(self):
        """yolov7.json shape — stays unloadable per plan choice (a)."""
        assert resolve_legacy_model_family({"yolo": 7}) == "ultralytics_yolov5"

    def test_unhandled_version_raises(self):
        with pytest.raises(Exception):
            resolve_legacy_model_family({"yolo": 42})


class TestRegistryResolvesExpectedProvider:
    def test_ultralytics_yolov8_registered(self):
        assert inference_provider_registry["ultralytics_yolov8"] is UltralyticsYoloV8InferenceProvider

    def test_libreyolo9_registered(self):
        assert inference_provider_registry["libreyolo9"] is LibreYolo9InferenceProvider

    def test_ultralytics_yolov5_not_registered(self):
        """Choice (a): yolo 5/7 resolve to a family string with no provider."""
        assert "ultralytics_yolov5" not in inference_provider_registry


class TestLoadYoloOnnxDispatch:
    """End-to-end through load_yolo_onnx for the cases that don't require an
    actual ONNX session — i.e. the unhandled-family error path."""

    def _write_sidecar(self, tmp_path, name, metadata):
        model_path = tmp_path / f"{name}.onnx"
        model_path.write_bytes(b"")
        (tmp_path / f"{name}.json").write_text(json.dumps(metadata))
        return str(model_path)

    def test_yolov5_sidecar_raises_unhandled_family(self, tmp_path):
        model_path = self._write_sidecar(tmp_path, "yolov5", {"yolo": 5, "classes": ["a"]})
        with pytest.raises(Exception, match="Unhandled model family 'ultralytics_yolov5'"):
            load_yolo_onnx(model_path)

    def test_yolov7_sidecar_raises_unhandled_family(self, tmp_path):
        model_path = self._write_sidecar(tmp_path, "yolov7", {"yolo": 7, "classes": ["a"]})
        with pytest.raises(Exception, match="Unhandled model family 'ultralytics_yolov5'"):
            load_yolo_onnx(model_path)
