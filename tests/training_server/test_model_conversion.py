"""Tests for the model-conversion trainer and source-model ingest endpoint.

The real conversion path (loading an actual libreyolo/ultralytics checkpoint and
exporting to ONNX) needs real pretrained weights and is out of scope for a fast
unit suite — it's exercised manually / in integration testing. Here we cover:
registration, the source-model ingest endpoint (success/404/409), and the
trainer's "unsupported checkpoint" failure path with a stub progress_cb and a
junk .pt file.
"""

import asyncio
import json

import onnx
import pytest
from onnx import TensorProto, helper

from apps.training_server.manager import Manager
from apps.training_server.trainers import TRAINER_REGISTRY
from apps.training_server.trainers.model_converter import (
    ModelConverter,
    _detect_ultralytics_model_family,
)


async def test_convert_model_trainer_is_registered(manager):
    """Starting the manager must register the convert_model trainer."""
    manager.start()
    assert "convert_model" in TRAINER_REGISTRY
    assert isinstance(TRAINER_REGISTRY["convert_model"], ModelConverter)


# ---------------------------------------------------------------------------
# Manager — ingest_source_model
# ---------------------------------------------------------------------------


async def test_ingest_source_model_sets_config_and_writes_file(manager):
    manager.start()
    job = await manager.submit_job({"trainer_type": "convert_model", "model_name": "mymodel"})

    data = b"fake-pt-bytes"
    result = manager.ingest_source_model(job.job_id, data)

    assert result is not None
    assert result["received"] == len(data)

    assert job.config["source_model_path"] == result["source_model_path"]
    source_path = job.artifact_dir / "_input" / "source_model.bin"
    assert source_path.exists()
    assert source_path.read_bytes() == data
    assert str(source_path) == result["source_model_path"]


async def test_ingest_source_model_unknown_job(manager):
    assert manager.ingest_source_model("no-such-job", b"data") is None


# ---------------------------------------------------------------------------
# Web endpoint — POST /jobs/{id}/source-model
# ---------------------------------------------------------------------------


async def test_ingest_source_model_endpoint(http_client, manager):
    manager.start()
    submit_resp = await http_client.post(
        "/api/training/jobs",
        json={"trainer_type": "convert_model", "model_name": "mymodel"},
    )
    job_id = submit_resp.json()["job_id"]

    resp = await http_client.post(
        f"/api/training/jobs/{job_id}/source-model",
        content=b"fake-pt-bytes",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["received"] == len(b"fake-pt-bytes")

    job = manager.get_job(job_id)
    assert job.config["source_model_path"].endswith("source_model.bin")


async def test_ingest_source_model_endpoint_unknown_job(http_client, manager):
    manager.start()
    resp = await http_client.post(
        "/api/training/jobs/no-such-job/source-model",
        content=b"data",
    )
    assert resp.status_code == 404


async def test_ingest_source_model_endpoint_rejects_wrong_status(http_client, manager):
    """409 once the job is no longer awaiting_data/queued (e.g. done)."""
    manager.start()
    submit_resp = await http_client.post(
        "/api/training/jobs",
        json={"trainer_type": "convert_model", "model_name": "mymodel"},
    )
    job_id = submit_resp.json()["job_id"]
    job = manager.get_job(job_id)
    job.status = "done"

    resp = await http_client.post(
        f"/api/training/jobs/{job_id}/source-model",
        content=b"data",
    )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Trainer — unsupported checkpoint path
# ---------------------------------------------------------------------------


async def test_convert_model_rejects_unsupported_checkpoint(tmp_path):
    """Feeding junk bytes should raise a clear ValueError (neither libreyolo
    nor ultralytics can load it), not some opaque internal exception."""
    junk_path = tmp_path / "junk.pt"
    junk_path.write_bytes(b"not a real checkpoint")

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    trainer = ModelConverter()
    logs = []

    def progress_cb(data):
        logs.append(data)

    config = {
        "source_model_path": str(junk_path),
        "model_name": "bad_model",
        "artifact_dir": str(artifact_dir),
    }

    with pytest.raises(ValueError):
        await trainer.train(config, progress_cb, asyncio.Event())


async def test_convert_model_requires_source_model_path(tmp_path):
    trainer = ModelConverter()
    config = {"model_name": "x", "artifact_dir": str(tmp_path)}
    with pytest.raises(ValueError):
        await trainer.train(config, lambda d: None, asyncio.Event())


async def test_convert_model_requires_existing_file(tmp_path):
    trainer = ModelConverter()
    config = {
        "source_model_path": str(tmp_path / "does_not_exist.pt"),
        "model_name": "x",
        "artifact_dir": str(tmp_path / "artifacts"),
    }
    with pytest.raises(ValueError):
        await trainer.train(config, lambda d: None, asyncio.Event())


# ---------------------------------------------------------------------------
# _detect_ultralytics_model_family — output-shape -> family detection
# ---------------------------------------------------------------------------


def _make_onnx_with_output_shape(tmp_path, dims: list[int]):
    """Build a minimal real ONNX file whose single output has the given shape,
    to drive _detect_ultralytics_model_family without needing a real model export."""
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, dims)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, dims)],
    )
    model = helper.make_model(graph)
    path = tmp_path / "shape_test.onnx"
    onnx.save(model, str(path))
    return path


class TestDetectUltralyticsModelFamily:
    def test_picked_rows_shape_detected_as_e2e(self, tmp_path):
        """(1, 300, 6) — YOLOv10/YOLO26's NMS-free fixed top-k picked-rows layout."""
        path = _make_onnx_with_output_shape(tmp_path, [1, 300, 6])
        assert _detect_ultralytics_model_family(path) == "ultralytics_e2e"

    def test_wide_multiclass_scores_shape_detected_as_classic(self, tmp_path):
        """(1, 84, 8400) — v8/v9/v11/v12's classic wide per-class-score layout."""
        path = _make_onnx_with_output_shape(tmp_path, [1, 84, 8400])
        assert _detect_ultralytics_model_family(path) == "ultralytics_detect"

    def test_transposed_wide_shape_detected_as_classic(self, tmp_path):
        """(1, 8400, 84) — transposed variant, last dim != 6, still classic."""
        path = _make_onnx_with_output_shape(tmp_path, [1, 8400, 84])
        assert _detect_ultralytics_model_family(path) == "ultralytics_detect"

    def test_unexpected_shape_falls_back_to_classic(self, tmp_path):
        """Any shape that doesn't match the picked-rows pattern exactly falls back
        to the classic decode path (safer default than mis-reading picked rows as
        a per-class score matrix)."""
        path = _make_onnx_with_output_shape(tmp_path, [1, 5])
        assert _detect_ultralytics_model_family(path) == "ultralytics_detect"


# ---------------------------------------------------------------------------
# Real end-to-end conversion (slow ~100s each; needs the training container's
# real ultralytics/libreyolo installs — see NOTES.md "libreyolo isn't
# installed in the inference container").
# ---------------------------------------------------------------------------


class TestRealUltralyticsConversion:
    """Exercises the actual bug this section fixed: model_converter.py's
    top-level `lm.load_untrusted_torch_file(source_path)` call used to raise
    unconditionally on any genuine ultralytics checkpoint (torch's
    weights_only=True safe loader correctly rejects pickled
    `ultralytics.nn.tasks.DetectionModel`), making the documented "ultralytics
    fallback" branch dead code for every real .pt file. These tests build
    real (randomly-initialized, untrained) ultralytics checkpoints via
    `YOLO(...).save(...)` and run them through the ACTUAL ModelConverter.train
    — not mocked — asserting both that the conversion completes and that the
    written sidecar's model_family matches the real exported ONNX shape.
    """

    def _save_checkpoint(self, tmp_path, yaml_name: str, filename: str):
        from ultralytics import YOLO

        path = tmp_path / filename
        YOLO(yaml_name).save(str(path))
        return path

    async def test_yolov8_checkpoint_converts_to_classic_family(self, tmp_path):
        checkpoint_path = self._save_checkpoint(tmp_path, "yolov8n.yaml", "fake_v8.pt")
        artifact_dir = tmp_path / "artifacts_v8"

        trainer = ModelConverter()
        logs = []
        result = await trainer.train(
            {
                "source_model_path": str(checkpoint_path),
                "model_name": "conv_v8",
                "artifact_dir": str(artifact_dir),
            },
            logs.append,
            asyncio.Event(),
        )

        onnx_path = artifact_dir / "conv_v8.onnx"
        sidecar_path = artifact_dir / "conv_v8.json"
        assert onnx_path.exists()
        assert result["model_path"] == str(onnx_path)

        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["model_family"] == "ultralytics_detect"
        assert len(sidecar["classes"]) == 80

    async def test_yolov10_checkpoint_converts_to_e2e_family(self, tmp_path):
        """The family-correctness case called out in MORE_YOLO_PLAN.md Section 6:
        a v10 checkpoint MUST land as ultralytics_e2e, not the hardcoded
        ultralytics_yolov8/ultralytics_detect the pre-fix code always wrote."""
        checkpoint_path = self._save_checkpoint(tmp_path, "yolov10n.yaml", "fake_v10.pt")
        artifact_dir = tmp_path / "artifacts_v10"

        trainer = ModelConverter()
        logs = []
        result = await trainer.train(
            {
                "source_model_path": str(checkpoint_path),
                "model_name": "conv_v10",
                "artifact_dir": str(artifact_dir),
            },
            logs.append,
            asyncio.Event(),
        )

        onnx_path = artifact_dir / "conv_v10.onnx"
        sidecar_path = artifact_dir / "conv_v10.json"
        assert onnx_path.exists()
        assert result["model_path"] == str(onnx_path)

        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["model_family"] == "ultralytics_e2e"

    async def test_converted_onnx_loads_through_load_yolo_onnx(self, tmp_path):
        """Full round-trip: convert a real checkpoint, then load the produced
        ONNX through the actual inference-side loader and run one inference
        pass — proving load_yolo_onnx picks the provider that matches the
        sidecar family this trainer wrote."""
        from apps.helpers.yoloUtils import load_yolo_onnx

        checkpoint_path = self._save_checkpoint(tmp_path, "yolov10n.yaml", "fake_v10_roundtrip.pt")
        artifact_dir = tmp_path / "artifacts_roundtrip"

        trainer = ModelConverter()
        await trainer.train(
            {
                "source_model_path": str(checkpoint_path),
                "model_name": "roundtrip",
                "artifact_dir": str(artifact_dir),
            },
            lambda d: None,
            asyncio.Event(),
        )

        onnx_path = artifact_dir / "roundtrip.onnx"
        provider = load_yolo_onnx(str(onnx_path))
        assert type(provider).__name__ == "UltralyticsE2eInferenceProvider"
        assert provider.SKIP_NMS is True

        import numpy as np

        img = (np.random.rand(480, 640, 3) * 255).astype("uint8")
        result = await provider.processImage(img, conf_thresh=0.01)
        assert isinstance(result.detections, list)
