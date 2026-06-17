import asyncio

import pytest

from apps import APPS
from apps.training_server.manager import Manager


# ---------------------------------------------------------------------------
# Web endpoints
# ---------------------------------------------------------------------------


async def test_server_config_returns_service_name(http_client):
    resp = await http_client.get("/server-config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["config"]["service"] == APPS.TRAINING_SERVER.value


async def test_server_config_failure_returns_error_status(http_client, manager, monkeypatch):
    async def boom():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(manager, "get_server_config", boom)

    resp = await http_client.get("/server-config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failure"
    assert "config unavailable" in data["message"]


async def test_training_status_returns_running_message(http_client):
    resp = await http_client.get("/api/training/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["message"] == "Training server running."


# ---------------------------------------------------------------------------
# Manager — server config
# ---------------------------------------------------------------------------


async def test_get_server_config_includes_service(manager):
    """Server config must always expose the service name."""
    config = await manager.get_server_config()
    assert config["service"] == APPS.TRAINING_SERVER


async def test_get_server_config_is_a_copy(manager):
    """Mutating the returned dict must not affect internal state."""
    config = await manager.get_server_config()
    original_service = config["service"]
    config["service"] = "tampered"
    config2 = await manager.get_server_config()
    assert config2["service"] == original_service


async def test_get_server_config_includes_fs_fields(manager):
    """Config includes shared_fs_available, fs_root, and default_image_source."""
    config = await manager.get_server_config()
    assert "shared_fs_available" in config
    assert "fs_root" in config
    assert "default_image_source" in config
    assert config["default_image_source"] in ("filesystem", "http")


# ---------------------------------------------------------------------------
# Manager — job submission and status
# ---------------------------------------------------------------------------


async def test_submit_job_returns_queued_job(manager):
    manager.start()
    job = await manager.submit_job({"trainer_type": "binary_image_classifier", "model_name": "test"})
    assert job.job_id
    assert job.status in ("queued", "running", "error")  # may advance quickly


async def test_list_jobs_empty_initially(manager):
    assert manager.list_jobs() == []


async def test_list_jobs_after_submit(manager):
    manager.start()
    await manager.submit_job({"trainer_type": "binary_image_classifier", "model_name": "m1"})
    jobs = manager.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].config["model_name"] == "m1"


async def test_get_job_returns_none_for_unknown(manager):
    assert manager.get_job("does-not-exist") is None


async def test_cancel_unknown_job_returns_false(manager):
    assert manager.cancel_job("does-not-exist") is False


async def test_cancel_queued_job(manager):
    # Use no workers so the submitted job stays queued.
    manager._queue._worker_count = 0
    manager.start()
    job = await manager.submit_job({"trainer_type": "binary_image_classifier", "model_name": "x"})
    assert job.status == "queued"
    result = manager.cancel_job(job.job_id)
    assert result is True
    assert job.status == "cancelled"


# ---------------------------------------------------------------------------
# Web endpoints — job lifecycle
# ---------------------------------------------------------------------------


async def test_submit_job_endpoint(http_client, manager):
    manager.start()
    resp = await http_client.post(
        "/api/training/jobs",
        json={"trainer_type": "binary_image_classifier", "model_name": "test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] in ("queued", "running", "error")


async def test_list_jobs_endpoint_empty(http_client, manager):
    manager.start()
    resp = await http_client.get("/api/training/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["jobs"] == []


async def test_list_jobs_endpoint_after_submit(http_client, manager):
    manager.start()
    await http_client.post(
        "/api/training/jobs",
        json={"trainer_type": "binary_image_classifier", "model_name": "m1"},
    )
    resp = await http_client.get("/api/training/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["jobs"]) == 1
    assert data["jobs"][0]["model_name"] == "m1"


async def test_get_job_endpoint_not_found(http_client, manager):
    manager.start()
    resp = await http_client.get("/api/training/jobs/no-such-job")
    assert resp.status_code == 404


async def test_cancel_job_endpoint_not_found(http_client, manager):
    manager.start()
    resp = await http_client.delete("/api/training/jobs/no-such-job")
    assert resp.status_code == 404


async def test_cancel_queued_job_endpoint(http_client, manager):
    # Disable workers so the job stays queued.
    manager._queue._worker_count = 0
    manager.start()
    submit_resp = await http_client.post(
        "/api/training/jobs",
        json={"trainer_type": "binary_image_classifier", "model_name": "cancel_me"},
    )
    job_id = submit_resp.json()["job_id"]

    cancel_resp = await http_client.delete(f"/api/training/jobs/{job_id}")
    assert cancel_resp.status_code == 200
    data = cancel_resp.json()
    assert data["cancelled"] is True


async def test_get_job_endpoint_returns_job_info(http_client, manager):
    manager._queue._worker_count = 0
    manager.start()
    submit_resp = await http_client.post(
        "/api/training/jobs",
        json={"trainer_type": "binary_image_classifier", "model_name": "detail_test"},
    )
    job_id = submit_resp.json()["job_id"]

    get_resp = await http_client.get(f"/api/training/jobs/{job_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["job_id"] == job_id
    assert data["trainer_type"] == "binary_image_classifier"
    assert data["model_name"] == "detail_test"
