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

import onnx
import pytest
from onnx import TensorProto, helper

# Importing these triggers @register_inference_provider registration.
import apps.helpers.inferenceProviders.libreYolo9InferenceProvider  # noqa: F401
import apps.helpers.inferenceProviders.ultralyticsE2eInferenceProvider  # noqa: F401
import apps.helpers.inferenceProviders.ultralyticsYoloV8InferenceProvider  # noqa: F401
from apps.helpers import metadataUtils
from apps.helpers.inferenceProviders.libreYolo9InferenceProvider import (
    LibreYolo9InferenceProvider,
)
from apps.helpers.inferenceProviders.Registry import inference_provider_registry
from apps.helpers.inferenceProviders.ultralyticsE2eInferenceProvider import (
    UltralyticsE2eInferenceProvider,
)
from apps.helpers.inferenceProviders.ultralyticsYoloV8InferenceProvider import (
    UltralyticsYoloV8InferenceProvider,
)
from apps.helpers.metadataUtils import (
    get_model_metadata,
    providers_for_device,
    resolve_legacy_model_family,
)
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

    def test_ultralytics_detect_registered_same_class_as_yolov8(self):
        """ultralytics_detection_trainer.py writes model_family="ultralytics_detect"
        for v8/Ultralytics-v9/v11/v12 — all share UltralyticsYoloV8InferenceProvider's
        decode (cxcywh boxes, pre-sigmoid'd scores, no objectness). Stacked
        @register_inference_provider decorators must register the SAME class
        under both the legacy "ultralytics_yolov8" key (existing on-disk
        sidecars) and the new "ultralytics_detect" key (new sidecars) — no
        second class, no data migration needed."""
        assert inference_provider_registry["ultralytics_detect"] is UltralyticsYoloV8InferenceProvider
        assert inference_provider_registry["ultralytics_detect"] is inference_provider_registry["ultralytics_yolov8"]

    def test_ultralytics_e2e_registered(self):
        """YOLOv10/YOLO26 NMS-free exports resolve to their own decode class,
        distinct from the classic v8-shape UltralyticsYoloV8InferenceProvider."""
        assert inference_provider_registry["ultralytics_e2e"] is UltralyticsE2eInferenceProvider
        assert UltralyticsE2eInferenceProvider.SKIP_NMS is True


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


def _write_onnx_with_description(path, description: str) -> None:
    """A minimal valid ONNX model whose only purpose is carrying a metadata
    `description` string — mirrors what real ultralytics exports embed
    (e.g. "Ultralytics YOLOv8x model trained on coco.yaml" /
    "Ultralytics YOLO11s model trained on ...")."""
    input_tensor = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output_tensor = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph([node], "g", [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    entry = model.metadata_props.add()
    entry.key = "description"
    entry.value = description
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


class TestGetModelMetadataYoloVersionSniffing:
    """metadataUtils.get_model_metadata()'s description-string sniffing, run
    against the real description format ultralytics embeds in its ONNX
    exports (confirmed against actual yolov8x.onnx/Yolo11S.onnx files)."""

    def test_yolov8_description_detected(self, tmp_path):
        model_path = tmp_path / "m.onnx"
        _write_onnx_with_description(model_path, "Ultralytics YOLOv8x model trained on coco.yaml")
        metadata = get_model_metadata(str(model_path), create_json_if_missing=False)
        assert metadata["yolo"] == 8

    def test_yolo11_description_is_recognized_not_silently_misdetected(self, tmp_path):
        """YOLO11 shares YOLOv8's Detect head/ONNX layout, so it's mapped to
        yolo version 8 (same provider) rather than falling through to the
        YoloV5 default — that default previously left YOLO11 models
        undetected and only "working" by accident when a caller hardcoded
        model_family: ultralytics_yolov8 in the sidecar."""
        model_path = tmp_path / "m.onnx"
        _write_onnx_with_description(model_path, "Ultralytics YOLO11s model trained on coco.yaml")
        metadata = get_model_metadata(str(model_path), create_json_if_missing=False)
        assert metadata["yolo"] == 8

    def test_unrecognized_description_falls_back_to_yolov5(self, tmp_path):
        model_path = tmp_path / "m.onnx"
        _write_onnx_with_description(model_path, "some other export")
        metadata = get_model_metadata(str(model_path), create_json_if_missing=False)
        assert metadata["yolo"] == 5


class TestProvidersForDevice:
    """The provider list is filtered against what this build actually has.

    Both builds are pinned explicitly rather than read off the host. These assertions
    used to depend on whichever onnxruntime the test container happened to carry, so the
    CPU-only case — the one with a bug in it — was never exercised anywhere.
    """

    @pytest.fixture
    def gpu_build(self, monkeypatch):
        monkeypatch.setattr(
            metadataUtils,
            "available_providers",
            lambda: ("CUDAExecutionProvider", "CPUExecutionProvider"),
        )

    @pytest.fixture
    def cpu_build(self, monkeypatch):
        """What `requirements-cpu.txt` gives you: plain onnxruntime, no CUDA at all."""
        monkeypatch.setattr(metadataUtils, "available_providers", lambda: ("CPUExecutionProvider",))

    def test_auto_prefers_cuda_with_cpu_fallback(self, gpu_build):
        assert providers_for_device("auto") == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_gpu_prefers_cuda_with_cpu_fallback(self, gpu_build):
        assert providers_for_device("gpu") == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_cpu_is_cpu_only(self, gpu_build):
        assert providers_for_device("cpu") == ["CPUExecutionProvider"]

    def test_unknown_device_falls_back_to_auto_behavior(self, gpu_build):
        assert providers_for_device("tpu") == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @pytest.mark.parametrize("device", ["auto", "gpu", "cpu", "tpu"])
    def test_a_cpu_only_build_never_names_cuda(self, cpu_build, device):
        """Naming a provider that was never compiled in makes onnxruntime log an
        error-level complaint on every single model load, which reads like a broken GPU
        on a machine that was never meant to have one."""
        assert providers_for_device(device) == ["CPUExecutionProvider"]

    def test_an_unreadable_onnxruntime_still_yields_a_usable_list(self, monkeypatch):
        """Returning [] would fail the session with a far less obvious error."""
        monkeypatch.setattr(metadataUtils, "available_providers", tuple)
        assert providers_for_device("auto") == ["CPUExecutionProvider"]
