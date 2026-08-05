"""Unit tests for UltralyticsDetectionTrainer (apps/training_server/trainers/
ultralytics_detection_trainer.py).

Mocks the `ultralytics` package entirely (via sys.modules injection) so this
covers: arch-map dispatch/validation, hyperparameter defaults, dataset-builder
wiring, and sidecar JSON contents — without requiring torch/ultralytics to be
installed. Mirrors tests/training_server/test_training.py's patterns.

Per NOTES.md: `libreyolo`/`ultralytics`-adjacent training code should be run
in the training dev container, since real torch/albumentations imports happen
transitively via apps/training_server/dataset.py (ImageLabelDataset). This
test avoids exercising that path directly (it mocks build_yolo_detection_dataset),
but lives here so it runs alongside the rest of the trainer suite.
"""

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from apps.training_server.trainers import TRAINER_REGISTRY
from apps.training_server.trainers.ultralytics_detection_trainer import (
    _ARCH_MAP,
    UltralyticsDetectionTrainer,
    _model_family_for_arch,
)


class TestTrainerRegistration:
    def test_registered_under_ultralytics_object_detection(self):
        assert isinstance(TRAINER_REGISTRY["ultralytics_object_detection"], UltralyticsDetectionTrainer)


class TestArchMap:
    @pytest.mark.parametrize(
        "arch",
        [
            "yolov8n",
            "yolov8s",
            "yolov8m",
            "yolov8l",
            "yolov8x",
            "yolov9t",
            "yolov9s",
            "yolov9m",
            "yolov9c",
            "yolov9e",
            "yolo11n",
            "yolo11s",
            "yolo11m",
            "yolo11l",
            "yolo11x",
            "yolo12n",
            "yolo12s",
            "yolo12m",
            "yolo12l",
            "yolo12x",
            "yolov10n",
            "yolov10s",
            "yolov10m",
            "yolov10b",
            "yolov10l",
            "yolov10x",
            "yolo26n",
            "yolo26s",
            "yolo26m",
            "yolo26l",
            "yolo26x",
        ],
    )
    def test_all_planned_architectures_present(self, arch):
        assert arch in _ARCH_MAP

    def test_arch_map_values_are_the_weight_stem(self):
        # Weight filename stem should match the arch name for all entries here
        # (Ultralytics resolves "<stem>.pt" directly).
        for arch, stem in _ARCH_MAP.items():
            assert stem == arch


class TestModelFamilyForArch:
    """model_family is chosen per-architecture, not hardcoded to one family:
    classic v8-shape archs (v8/Ultralytics-v9/v11/v12) write "ultralytics_detect"
    (decoded by UltralyticsYoloV8InferenceProvider); NMS-free v10/YOLO26 write
    "ultralytics_e2e" (decoded by UltralyticsE2eInferenceProvider) — see
    MORE_YOLO_PLAN.md Section 2 research notes for the confirmed export shapes."""

    @pytest.mark.parametrize(
        "arch",
        [
            "yolov8n",
            "yolov8x",
            "yolov9t",
            "yolov9e",
            "yolo11n",
            "yolo11x",
            "yolo12n",
            "yolo12x",
        ],
    )
    def test_classic_architectures_write_ultralytics_detect(self, arch):
        assert _model_family_for_arch(arch) == "ultralytics_detect"

    @pytest.mark.parametrize(
        "arch",
        [
            "yolov10n",
            "yolov10s",
            "yolov10m",
            "yolov10b",
            "yolov10l",
            "yolov10x",
            "yolo26n",
            "yolo26s",
            "yolo26m",
            "yolo26l",
            "yolo26x",
        ],
    )
    def test_nms_free_architectures_write_ultralytics_e2e(self, arch):
        assert _model_family_for_arch(arch) == "ultralytics_e2e"


class TestHyperparamDefaults:
    def test_matches_object_detection_trainer_shape(self):
        d = UltralyticsDetectionTrainer.HYPERPARAM_DEFAULTS
        assert set(d.keys()) == {"max_epochs", "imgsz", "batch_size", "patience"}

    def test_reads_patience_not_early_stopping_patience(self):
        """See NOTES.md: hyperparams_json key-name mismatch gotcha — this
        trainer must read "patience" like object_detection_trainer, not
        "early_stopping_patience" like the classifier trainers."""
        d = UltralyticsDetectionTrainer.HYPERPARAM_DEFAULTS
        assert "patience" in d
        assert "early_stopping_patience" not in d


def _make_fake_ultralytics_module(fake_results_dict, exported_path):
    """Build a fake `ultralytics` module exposing a YOLO class whose .train()/
    .export() behave like the real API just enough for the trainer to run."""
    fake_module = MagicMock()

    class FakeYOLO:
        def __init__(self, weights):
            self.weights = weights
            self._callbacks = {}

        def add_callback(self, event, fn):
            self._callbacks[event] = fn

        def train(self, **kwargs):
            # Simulate one epoch completing.
            cb = self._callbacks.get("on_train_epoch_end")
            if cb is not None:
                fake_trainer = SimpleNamespace(
                    epoch=0,
                    metrics={"metrics/mAP50(B)": fake_results_dict.get("metrics/mAP50(B)", 0.0)},
                    loss_items=None,
                    stop_training=False,
                )
                cb(fake_trainer)
            return SimpleNamespace(results_dict=fake_results_dict)

        def export(self, format="onnx"):
            assert format == "onnx"
            return exported_path

    fake_module.YOLO = FakeYOLO
    return fake_module


@pytest.fixture
def trainer():
    return UltralyticsDetectionTrainer()


@pytest.fixture
def base_config(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    # Populate a couple of fake cached "images" so build_yolo_detection_dataset
    # (which we still exercise for real — it's pure filesystem, no torch) has
    # something to copy.
    manifest = []
    for i in range(3):
        uuid = f"uuid-{i}"
        (cache_dir / uuid).write_bytes(b"\xff\xd8\xff\xd9")  # minimal jpeg-ish bytes
        manifest.append({"image_uuid": uuid, "filename": f"{uuid}.jpg", "status": "present"})

    return {
        "architecture": "yolov8n",
        "class_names": ["cat", "dog"],
        "annotations": {},
        "model_name": "test_model",
        "cache_dir": str(cache_dir),
        "artifact_dir": str(artifact_dir),
        "manifest": manifest,
        "role": "top",
        "model_config_uuid": "mc-uuid",
        "model_config_node_uuid": "node-uuid",
        "label_uuids": ["label-1", "label-2"],
        "max_epochs": 3,
        "imgsz": 320,
        "batch_size": 4,
        "patience": 5,
    }


class TestTrainValidation:
    async def test_raises_without_class_names(self, trainer, base_config):
        base_config["class_names"] = []
        with pytest.raises(ValueError, match="class_names"):
            await trainer.train(base_config, progress_cb=lambda e: None, cancel_event=asyncio.Event())

    async def test_raises_without_manifest(self, trainer, base_config):
        base_config["manifest"] = []
        with pytest.raises(ValueError, match="empty manifest"):
            await trainer.train(base_config, progress_cb=lambda e: None, cancel_event=asyncio.Event())

    async def test_raises_without_cache_dir(self, trainer, base_config):
        base_config["cache_dir"] = ""
        with pytest.raises(ValueError, match="cache_dir"):
            await trainer.train(base_config, progress_cb=lambda e: None, cancel_event=asyncio.Event())

    async def test_raises_without_artifact_dir(self, trainer, base_config):
        base_config["artifact_dir"] = ""
        with pytest.raises(ValueError, match="artifact_dir"):
            await trainer.train(base_config, progress_cb=lambda e: None, cancel_event=asyncio.Event())

    async def test_raises_on_unsupported_architecture(self, trainer, base_config):
        base_config["architecture"] = "not_a_real_arch"
        with pytest.raises(ValueError, match="Unsupported architecture"):
            await trainer.train(base_config, progress_cb=lambda e: None, cancel_event=asyncio.Event())


class TestTrainHappyPath:
    async def test_writes_onnx_and_sidecar_with_ultralytics_detect_family(self, trainer, base_config):
        artifact_dir = base_config["artifact_dir"]
        fake_exported = f"{artifact_dir}/somewhere_else.onnx"
        # Ultralytics' export() call needs to actually produce bytes for the
        # trainer's shutil.copy2 fallback path.
        with open(fake_exported, "wb") as f:
            f.write(b"fake onnx bytes")

        fake_ultralytics = _make_fake_ultralytics_module(
            fake_results_dict={"metrics/mAP50(B)": 0.85},
            exported_path=fake_exported,
        )

        events = []

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            result = await trainer.train(
                base_config,
                progress_cb=events.append,
                cancel_event=asyncio.Event(),
            )

        assert result["model_name"] == "test_model"
        assert result["val_accuracy"] == pytest.approx(0.85)

        onnx_path = f"{artifact_dir}/test_model.onnx"
        assert result["model_path"] == onnx_path
        with open(onnx_path, "rb") as f:
            assert f.read() == b"fake onnx bytes"

        sidecar_path = f"{artifact_dir}/test_model.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert sidecar["model_family"] == "ultralytics_detect"
        assert sidecar["architecture"] == "yolov8n"
        assert sidecar["classes"] == ["cat", "dog"]
        assert sidecar["trainer"] == "ultralytics_object_detection"
        assert sidecar["val_accuracy"] == pytest.approx(0.85)
        assert sidecar["role"] == "top"
        assert sidecar["model_config_uuid"] == "mc-uuid"
        assert sidecar["model_config_node_uuid"] == "node-uuid"
        assert sidecar["label_uuids"] == ["label-1", "label-2"]
        assert set(sidecar["train_image_uuids"] + sidecar["val_image_uuids"]) == {
            e["image_uuid"] for e in base_config["manifest"]
        }

        # Progress callback fired at least once with an epoch update.
        assert any("epoch" in e for e in events)

    async def test_yolov10_arch_writes_ultralytics_e2e_family(self, trainer, base_config):
        """NMS-free architectures must write model_family="ultralytics_e2e",
        not the classic "ultralytics_detect" — the two decode differently
        (see UltralyticsE2eInferenceProvider vs UltralyticsYoloV8InferenceProvider)."""
        base_config["architecture"] = "yolov10n"
        artifact_dir = base_config["artifact_dir"]
        fake_exported = f"{artifact_dir}/exported_v10.onnx"
        with open(fake_exported, "wb") as f:
            f.write(b"fake onnx bytes")

        fake_ultralytics = _make_fake_ultralytics_module(
            fake_results_dict={"metrics/mAP50(B)": 0.7},
            exported_path=fake_exported,
        )
        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            await trainer.train(base_config, progress_cb=lambda e: None, cancel_event=asyncio.Event())

        sidecar_path = f"{artifact_dir}/test_model.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert sidecar["model_family"] == "ultralytics_e2e"
        assert sidecar["architecture"] == "yolov10n"

    async def test_yolo26_arch_writes_ultralytics_e2e_family(self, trainer, base_config):
        base_config["architecture"] = "yolo26n"
        artifact_dir = base_config["artifact_dir"]
        fake_exported = f"{artifact_dir}/exported_26.onnx"
        with open(fake_exported, "wb") as f:
            f.write(b"fake onnx bytes")

        fake_ultralytics = _make_fake_ultralytics_module(
            fake_results_dict={"metrics/mAP50(B)": 0.7},
            exported_path=fake_exported,
        )
        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            await trainer.train(base_config, progress_cb=lambda e: None, cancel_event=asyncio.Event())

        sidecar_path = f"{artifact_dir}/test_model.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert sidecar["model_family"] == "ultralytics_e2e"
        assert sidecar["architecture"] == "yolo26n"

    async def test_record_image_manifest_false_omits_the_uuid_lists(self, trainer, base_config):
        """The per-training-run opt-out reaches the sidecar. Keys are omitted,
        not emptied, so "not recorded" stays distinguishable from "empty"."""
        base_config["record_image_manifest"] = False
        artifact_dir = base_config["artifact_dir"]
        fake_exported = f"{artifact_dir}/exported_no_manifest.onnx"
        with open(fake_exported, "wb") as f:
            f.write(b"fake onnx bytes")

        fake_ultralytics = _make_fake_ultralytics_module(
            fake_results_dict={"metrics/mAP50(B)": 0.7},
            exported_path=fake_exported,
        )
        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            await trainer.train(base_config, progress_cb=lambda e: None, cancel_event=asyncio.Event())

        with open(f"{artifact_dir}/test_model.json") as f:
            sidecar = json.load(f)
        assert "train_image_uuids" not in sidecar
        assert "val_image_uuids" not in sidecar
        # Everything else the sidecar carries is untouched by the opt-out.
        assert sidecar["label_uuids"] == ["label-1", "label-2"]
        assert sidecar["architecture"] == "yolov8n"

    async def test_sidecar_has_no_libreyolo_only_keys(self, trainer, base_config):
        """Sanity guard: this trainer's sidecar must not carry the LibreYOLO9-
        specific "yolo": 9 legacy key — model_family alone must resolve the
        provider (see metadataUtils.resolve_legacy_model_family, unchanged)."""
        artifact_dir = base_config["artifact_dir"]
        fake_exported = f"{artifact_dir}/exported.onnx"
        with open(fake_exported, "wb") as f:
            f.write(b"x")

        fake_ultralytics = _make_fake_ultralytics_module(
            fake_results_dict={"metrics/mAP50(B)": 0.5},
            exported_path=fake_exported,
        )
        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            await trainer.train(base_config, progress_cb=lambda e: None, cancel_event=asyncio.Event())

        sidecar_path = f"{artifact_dir}/test_model.json"
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        assert "yolo" not in sidecar
