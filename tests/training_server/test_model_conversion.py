"""Tests for the model-conversion trainer and source-model ingest endpoint.

The real conversion path (loading an actual libreyolo/ultralytics checkpoint and
exporting to ONNX) needs real pretrained weights and is out of scope for a fast
unit suite — it's exercised manually / in integration testing. Here we cover:
registration, the source-model ingest endpoint (success/404/409), and the
trainer's "unsupported checkpoint" failure path with a stub progress_cb and a
junk .pt file.
"""

import asyncio

import pytest

from apps.training_server.manager import Manager
from apps.training_server.trainers import TRAINER_REGISTRY
from apps.training_server.trainers.model_converter import ModelConverter


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
