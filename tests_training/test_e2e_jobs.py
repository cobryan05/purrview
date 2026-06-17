"""E2E tests against the live training server HTTP API.

Requires the training container to be running on port 8005.
Set TRAINING_SERVER_URL env var to override (default: http://localhost:8005).
"""

import os
import time

import httpx
import pytest

BASE_URL = os.environ.get("TRAINING_SERVER_URL", "http://localhost:8005")


@pytest.fixture(scope="module")
def client():
    return httpx.Client(base_url=BASE_URL, timeout=10.0)


def test_training_status(client):
    r = client.get("/api/training/status")
    assert r.status_code == 200
    assert r.json()["status"] == "success"


def test_server_config(client):
    r = client.get("/server-config")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert "shared_fs_available" in data["config"]


def test_submit_and_list_job(client):
    r = client.post(
        "/api/training/jobs",
        json={
            "trainer_type": "binary_image_classifier",
            "label_uuid": "00000000-0000-0000-0000-000000000000",
            "model_name": "e2e_test_model",
            "max_epochs": 1,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "job_id" in data
    assert data["status"] in ("queued", "running", "error")
    job_id = data["job_id"]

    # List — should contain our job
    r2 = client.get("/api/training/jobs")
    assert r2.status_code == 200
    ids = [j["job_id"] for j in r2.json()["jobs"]]
    assert job_id in ids

    # Get single job
    r3 = client.get(f"/api/training/jobs/{job_id}")
    assert r3.status_code == 200
    assert r3.json()["job_id"] == job_id


def test_cancel_queued_job(client):
    """Submit a job and immediately cancel it before it starts."""
    r = client.post(
        "/api/training/jobs",
        json={
            "trainer_type": "binary_image_classifier",
            "label_uuid": "00000000-0000-0000-0000-000000000000",
            "model_name": "cancel_e2e_test",
            "max_epochs": 100,
        },
    )
    job_id = r.json()["job_id"]

    cancel_r = client.delete(f"/api/training/jobs/{job_id}")
    assert cancel_r.status_code == 200
    data = cancel_r.json()
    # cancelled=True if it was still queued; cancelled=False + message if running
    assert isinstance(data["cancelled"], bool)


def test_get_nonexistent_job(client):
    r = client.get("/api/training/jobs/does-not-exist")
    assert r.status_code == 404


def test_delete_nonexistent_job(client):
    r = client.delete("/api/training/jobs/does-not-exist")
    assert r.status_code == 404
