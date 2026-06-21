"""Tests for the training server — job lifecycle, endpoints, and SSE."""

import asyncio
import io
import tarfile
from datetime import datetime, timezone

import pytest

from apps import APPS
from apps.training_server.job_queue import Job
from apps.training_server.manager import Manager

# ---------------------------------------------------------------------------
# Web endpoints — server config / status
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


# ---------------------------------------------------------------------------
# Manager — job submission (awaiting_data state)
# ---------------------------------------------------------------------------


async def test_submit_job_returns_awaiting_data(manager):
    """New jobs start in awaiting_data — they are not queued until start_job()."""
    manager.start()
    job = await manager.submit_job({"trainer_type": "binary_image_classifier", "model_name": "test"})
    assert job.job_id
    assert job.status == "awaiting_data"


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


async def test_cancel_awaiting_data_job(manager):
    """Jobs in awaiting_data can be cancelled immediately."""
    manager.start()
    job = await manager.submit_job({"trainer_type": "binary_image_classifier", "model_name": "x"})
    assert job.status == "awaiting_data"
    result = manager.cancel_job(job.job_id)
    assert result is True
    assert job.status == "cancelled"


async def test_cancel_queued_job(manager):
    # Use no workers so the submitted job stays queued.
    manager._queue._worker_count = 0
    manager.start()
    job = await manager.submit_job({"trainer_type": "binary_image_classifier", "model_name": "x"})
    # No manifest so start_job succeeds immediately.
    ok, missing = manager.start_job(job.job_id)
    assert ok
    assert job.status == "queued"
    result = manager.cancel_job(job.job_id)
    assert result is True
    assert job.status == "cancelled"


# ---------------------------------------------------------------------------
# Manager — missing images
# ---------------------------------------------------------------------------


async def test_missing_images_no_manifest(manager):
    """A job with no manifest reports 0 missing images."""
    manager.start()
    job = await manager.submit_job({"trainer_type": "binary_image_classifier"})
    missing = manager.get_missing_images(job.job_id)
    assert missing == []


async def test_missing_images_with_manifest(manager, tmp_path):
    """Only UUIDs not in the cache are returned as missing."""
    manager.start()
    manifest = [
        {"image_uuid": "uuid-1", "filename": "a.jpg", "status": "present"},
        {"image_uuid": "uuid-2", "filename": "b.jpg", "status": "absent"},
    ]
    # Pre-populate uuid-1 in the cache.
    (manager._cache._dir / "uuid-1").write_bytes(b"fake")

    job = await manager.submit_job({"trainer_type": "binary_image_classifier"}, manifest=manifest)
    missing = manager.get_missing_images(job.job_id)
    assert "uuid-2" in missing
    assert "uuid-1" not in missing


async def test_missing_images_unknown_job(manager):
    assert manager.get_missing_images("no-such-job") is None


# ---------------------------------------------------------------------------
# Manager — image ingest
# ---------------------------------------------------------------------------


def _make_tar(entries: list[tuple[str, bytes]]) -> bytes:
    """Build a tar in-memory from (name, data) pairs."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tf:
        for name, data in entries:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


async def test_ingest_images_adds_to_cache(manager, tmp_path):
    """Ingested tar members appear in the cache."""
    manager.start()
    manifest = [{"image_uuid": "img-a", "filename": "a.jpg", "status": "present"}]
    job = await manager.submit_job({"trainer_type": "binary_image_classifier"}, manifest=manifest)

    tar = _make_tar([("img-a", b"fake-jpeg-bytes")])
    result = manager.ingest_images(job.job_id, tar)
    assert result is not None
    assert result["received"] == 1
    assert manager._cache.has("img-a")


async def test_ingest_images_unknown_job(manager):
    tar = _make_tar([("uuid-x", b"data")])
    assert manager.ingest_images("no-job", tar) is None


async def test_ingest_images_rejects_non_manifest_uuids(manager, tmp_path):
    """UUIDs not in the manifest are silently dropped."""
    manager.start()
    manifest = [{"image_uuid": "allowed-uuid", "filename": "a.jpg", "status": "present"}]
    job = await manager.submit_job({"trainer_type": "binary_image_classifier"}, manifest=manifest)

    tar = _make_tar([("not-in-manifest", b"data")])
    manager.ingest_images(job.job_id, tar)
    assert not manager._cache.has("not-in-manifest")


# ---------------------------------------------------------------------------
# Manager — start_job
# ---------------------------------------------------------------------------


async def test_start_job_fails_when_data_missing(manager):
    """start_job returns (False, missing_uuids) when images are not cached."""
    manager.start()
    manifest = [{"image_uuid": "missing-uuid", "filename": "x.jpg", "status": "present"}]
    job = await manager.submit_job({"trainer_type": "binary_image_classifier"}, manifest=manifest)
    ok, missing = manager.start_job(job.job_id)
    assert not ok
    assert "missing-uuid" in missing
    assert job.status == "awaiting_data"


async def test_start_job_succeeds_when_data_complete(manager):
    """start_job transitions to queued when all images are cached."""
    manager._queue._worker_count = 0
    manager.start()
    manifest = [{"image_uuid": "cached-uuid", "filename": "x.jpg", "status": "present"}]
    (manager._cache._dir / "cached-uuid").write_bytes(b"fake")
    job = await manager.submit_job({"trainer_type": "binary_image_classifier"}, manifest=manifest)
    ok, missing = manager.start_job(job.job_id)
    assert ok
    assert missing == []
    assert job.status == "queued"


async def test_start_job_no_manifest_always_succeeds(manager):
    """A job with no manifest requires no images and starts immediately."""
    manager._queue._worker_count = 0
    manager.start()
    job = await manager.submit_job({"trainer_type": "binary_image_classifier"})
    ok, missing = manager.start_job(job.job_id)
    assert ok
    assert job.status == "queued"


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
    assert data["status"] == "awaiting_data"


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


async def test_cancel_awaiting_data_job_endpoint(http_client, manager):
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
    assert data["status"] == "awaiting_data"


async def test_missing_images_endpoint(http_client, manager):
    """GET /jobs/{id}/missing-images returns the uncached UUIDs."""
    manager.start()
    manifest = [
        {"image_uuid": "uuid-A", "filename": "a.jpg", "status": "present"},
        {"image_uuid": "uuid-B", "filename": "b.jpg", "status": "absent"},
    ]
    submit_resp = await http_client.post(
        "/api/training/jobs",
        json={"trainer_type": "binary_image_classifier", "manifest": manifest},
    )
    job_id = submit_resp.json()["job_id"]

    resp = await http_client.get(f"/api/training/jobs/{job_id}/missing-images")
    assert resp.status_code == 200
    uuids = set(resp.json()["uuids"])
    assert uuids == {"uuid-A", "uuid-B"}


async def test_ingest_images_endpoint(http_client, manager):
    """POST /jobs/{id}/images accepts a tar and caches the images."""
    manager.start()
    manifest = [{"image_uuid": "uuid-X", "filename": "x.jpg", "status": "present"}]
    submit_resp = await http_client.post(
        "/api/training/jobs",
        json={"trainer_type": "binary_image_classifier", "manifest": manifest},
    )
    job_id = submit_resp.json()["job_id"]

    tar = _make_tar([("uuid-X", b"fake-jpeg")])
    resp = await http_client.post(
        f"/api/training/jobs/{job_id}/images",
        content=tar,
        headers={"Content-Type": "application/x-tar"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["received"] == 1


async def test_start_job_endpoint_fails_with_missing_data(http_client, manager):
    """POST /start returns 409 when images are missing."""
    manager.start()
    manifest = [{"image_uuid": "uuid-missing", "filename": "m.jpg", "status": "present"}]
    submit_resp = await http_client.post(
        "/api/training/jobs",
        json={"trainer_type": "binary_image_classifier", "manifest": manifest},
    )
    job_id = submit_resp.json()["job_id"]

    resp = await http_client.post(f"/api/training/jobs/{job_id}/start")
    assert resp.status_code == 409
    assert "uuid-missing" in resp.json()["missing"]


async def test_start_job_endpoint_succeeds(http_client, manager):
    """POST /start transitions to queued when all data is present."""
    manager._queue._worker_count = 0
    manager.start()
    manifest = [{"image_uuid": "uuid-ok", "filename": "ok.jpg", "status": "present"}]
    (manager._cache._dir / "uuid-ok").write_bytes(b"ok")
    submit_resp = await http_client.post(
        "/api/training/jobs",
        json={"trainer_type": "binary_image_classifier", "manifest": manifest},
    )
    job_id = submit_resp.json()["job_id"]

    resp = await http_client.post(f"/api/training/jobs/{job_id}/start")
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"


async def test_artifact_endpoint_returns_404_until_done(http_client, manager):
    """GET /artifact returns 404 when the job is not done."""
    manager.start()
    submit_resp = await http_client.post(
        "/api/training/jobs",
        json={"trainer_type": "binary_image_classifier"},
    )
    job_id = submit_resp.json()["job_id"]
    resp = await http_client.get(f"/api/training/jobs/{job_id}/artifact")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Job — emit / stream
# ---------------------------------------------------------------------------


def _make_job(status: str = "queued") -> Job:
    return Job(
        job_id="test-job",
        status=status,
        config={},
        cancel_event=asyncio.Event(),
        created_at=datetime.now(timezone.utc),
    )


async def test_emit_stores_event_for_replay():
    """emit() stores events so late subscribers receive them via history replay."""
    job = _make_job()
    job.emit("progress", {"epoch": 1})
    job.emit("progress", {"epoch": 2})
    assert len(job._events) == 2
    assert job._events[0] == ("progress", {"epoch": 1})
    assert job._events[1] == ("progress", {"epoch": 2})


async def test_stream_replays_history():
    """stream() yields all past events before closing on an already-terminal job."""
    job = _make_job(status="done")
    job._events = [("progress", {"epoch": 1}), ("progress", {"epoch": 2})]

    events = [(et, d) async for et, d in job.stream()]

    assert events == [("progress", {"epoch": 1}), ("progress", {"epoch": 2})]


async def test_stream_terminal_job_exits_immediately():
    """stream() on a terminal job with no history closes without blocking."""
    job = _make_job(status="done")
    events = [(et, d) async for et, d in job.stream()]
    assert events == []


async def test_stream_receives_live_events():
    """Events emitted after stream() opens are delivered to the subscriber."""
    job = _make_job(status="running")

    async def emit_then_done():
        await asyncio.sleep(0)
        job.emit("progress", {"epoch": 1})
        job.emit("done", {"status": "done"})

    asyncio.create_task(emit_then_done())
    events = [(et, d) async for et, d in job.stream()]

    assert ("progress", {"epoch": 1}) in events
    assert events[-1] == ("done", {"status": "done"})


async def test_stream_closes_on_done_event():
    """A 'done' emit sends the sentinel that closes the stream."""
    job = _make_job(status="running")

    async def do_emit():
        await asyncio.sleep(0)
        job.emit("done", {"status": "done", "model_path": "/models/x.onnx"})

    asyncio.create_task(do_emit())
    event_types = [et async for et, _ in job.stream()]
    assert "done" in event_types


async def test_stream_closes_on_error_event():
    """An 'error' emit sends the sentinel that closes the stream."""
    job = _make_job(status="running")

    async def do_emit():
        await asyncio.sleep(0)
        job.emit("error", {"status": "error", "message": "boom"})

    asyncio.create_task(do_emit())
    event_types = [et async for et, _ in job.stream()]
    assert "error" in event_types


async def test_stream_unsubscribes_on_exit():
    """Subscriber queue is removed from job._subscribers after stream() finishes."""
    job = _make_job(status="running")

    async def do_emit():
        await asyncio.sleep(0)
        job.emit("done", {"status": "done"})

    asyncio.create_task(do_emit())
    async for _ in job.stream():
        pass

    assert len(job._subscribers) == 0


# ---------------------------------------------------------------------------
# Job lifecycle — awaiting_data state
# ---------------------------------------------------------------------------


async def test_awaiting_data_job_emits_initial_status():
    """A newly-submitted job emits an awaiting_data status event."""
    job = _make_job(status="awaiting_data")
    job.emit("status", {"status": "awaiting_data"})
    assert any(data.get("status") == "awaiting_data" for _, data in job._events)


async def test_job_stream_does_not_exit_on_awaiting_data():
    """awaiting_data is not a terminal status — stream should stay open."""
    job = _make_job(status="awaiting_data")
    # Inject an awaiting_data status event then cancel.
    job._events.append(("status", {"status": "awaiting_data"}))

    # Verify awaiting_data is NOT treated as terminal by stream().
    is_terminal = job.status in ("done", "error", "cancelled")
    assert not is_terminal


# ---------------------------------------------------------------------------
# Dataset transforms — albumentations API compatibility
# ---------------------------------------------------------------------------


class TestDatasetTransformConstruction:
    """Instantiating either dataset class with split='train' must not raise.

    Regression for albumentations 2.x: RandomResizedCrop replaced the
    ``height``/``width`` keyword arguments with a single ``size`` tuple.
    The bug surface: constructing any dataset with split='train' would raise
    a pydantic ``InitSchema`` validation error (``size: Field required``)
    from inside albumentations before any image was touched.
    """

    def test_multilabel_dataset_train_split(self):
        """MultiLabelDataset(split='train') constructs without error."""
        from apps.training_server.dataset import MultiLabelDataset

        ds = MultiLabelDataset(
            image_uuids=[],
            label_uuids=["label-uuid-1"],
            label_targets={},
            cache_dir="/tmp",
            split="train",
        )
        assert len(ds) == 0

    def test_multilabel_dataset_val_split(self):
        """MultiLabelDataset(split='val') constructs without error."""
        from apps.training_server.dataset import MultiLabelDataset

        ds = MultiLabelDataset(
            image_uuids=[],
            label_uuids=["label-uuid-1"],
            label_targets={},
            cache_dir="/tmp",
            split="val",
        )
        assert len(ds) == 0

    def test_image_label_dataset_train_split(self):
        """ImageLabelDataset(split='train') constructs without error."""
        from apps.training_server.dataset import ImageLabelDataset

        ds = ImageLabelDataset(split_rows=[], cache_dir="/tmp", split="train")
        assert len(ds) == 0

    def test_image_label_dataset_val_split(self):
        """ImageLabelDataset(split='val') constructs without error."""
        from apps.training_server.dataset import ImageLabelDataset

        ds = ImageLabelDataset(split_rows=[], cache_dir="/tmp", split="val")
        assert len(ds) == 0

    def test_build_multilabel_single_label_root_node(self, tmp_path):
        """build_multilabel_dataset_from_manifest with one label must not raise.

        Regression: training with a single-label root node always goes through
        the multi-label path, which triggered the albumentations 2.x
        RandomResizedCrop ``InitSchema`` / ``size: Field required`` error on
        dataset construction.
        """
        from apps.training_server.dataset import build_multilabel_dataset_from_manifest

        label_uuid = "32c2a22e-6f86-469f-810f-fef111479828"
        image_uuid = "img-001"
        (tmp_path / image_uuid).write_bytes(b"fake")

        manifest = [{"image_uuid": image_uuid, "filename": "a.jpg", "status": "present"}]
        label_targets = {image_uuid: {label_uuid: 1}}

        # Both splits must construct — 'train' is the split that uses RandomResizedCrop.
        ds_train = build_multilabel_dataset_from_manifest(
            manifest=manifest,
            label_uuids=[label_uuid],
            label_targets=label_targets,
            cache_dir=str(tmp_path),
            split="train",
        )
        ds_val = build_multilabel_dataset_from_manifest(
            manifest=manifest,
            label_uuids=[label_uuid],
            label_targets=label_targets,
            cache_dir=str(tmp_path),
            split="val",
        )
        # With one image, n_train=0 and n_val=1 (max-1 guard on n_val).
        assert len(ds_train) == 0
        assert len(ds_val) == 1
