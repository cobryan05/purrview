import pytest


# --- liveness ---

async def test_db_server_alive(db):
    r = await db.get("/server-config")
    assert r.status_code == 200
    assert r.json()["status"] == "success"


async def test_inference_server_alive(inference):
    r = await inference.get("/server-config")
    assert r.status_code == 200
    assert r.json()["status"] == "success"


async def test_tasks_server_alive(tasks):
    r = await tasks.get("/server-config")
    assert r.status_code == 200
    assert r.json()["status"] == "success"


async def test_relay_server_alive(relay):
    r = await relay.get("/server-config")
    assert r.status_code == 200
    assert r.json()["status"] == "success"


async def test_tagging_server_alive(tagging):
    r = await tagging.get("/server-config")
    assert r.status_code == 200
    assert r.json()["status"] == "success"


# --- db_server round-trip ---

async def test_label_round_trip(db):
    r = await db.post("/api/labels", json={"name": "_smoke", "color": "#123456"})
    assert r.json()["status"] == "success"
    uuid = r.json()["label"]["uuid"]

    r = await db.get("/api/labels")
    assert any(lbl["uuid"] == uuid for lbl in r.json()["labels"])

    await db.delete(f"/api/labels/{uuid}")
    r = await db.get("/api/labels")
    assert not any(lbl["uuid"] == uuid for lbl in r.json()["labels"])


async def test_image_list(db):
    r = await db.get("/api/images/list")
    assert r.status_code == 200
    assert r.json()["status"] == "success"


# --- inference_server ---

async def test_models_list(inference):
    r = await inference.get("/api/models")
    assert r.status_code == 200
    assert isinstance(r.json()["models"], list)


# --- tasks_server ---

async def test_task_types_list(tasks):
    r = await tasks.get("/api/tasks/types/list")
    assert r.status_code == 200
    assert r.json()["status"] == "success"


async def test_started_task_reports_running_status(tasks):
    """A task must immediately appear as 'running' (not the DB-initial 'new').

    This is the live-server regression guard for the bug where the in-memory
    running status was overwritten by the stale DB record on every poll.

    Uses a UUID-suffixed name to avoid conflicts with leftovers from previous
    smoke runs.  Skips gracefully if the tasks/db server is not fully operational
    (e.g. another smoke test has already broken the session).
    """
    import uuid as _uuid

    unique_name = f"_smoke_lifecycle_{_uuid.uuid4().hex[:8]}"
    config_uuid = None
    task_uuid = None

    try:
        # Create a minimal task config.  Empty source_uuids means the task finds
        # no providers and completes immediately with 0 saves — lightweight but real.
        r = await tasks.post(
            "/api/tasks/configs/create",
            json={
                "name": unique_name,
                "typename": "CollectTrainingImagesTask",
                "params": {
                    "source_uuids": [],
                    "output_dir": "_smoke_test_output",
                    "model_label_config": [],
                },
                "persistent": False,
            },
        )
        if r.json().get("status") != "success":
            pytest.skip(f"Task config creation failed — environment may be broken: {r.text}")
        config_uuid = r.json()["configUuid"]

        r = await tasks.post(
            "/api/tasks/instances/start",
            json={"configUuid": config_uuid},
        )
        assert r.json()["status"] == "success", r.text
        task_uuid = r.json()["taskUuid"]

        # Query all instances immediately. Status must be 'running' or a terminal
        # state — never the DB-initial 'new'.
        r = await tasks.post("/api/tasks/instances/get", json={})
        assert r.json()["status"] == "success"
        task_data = r.json()["tasks"].get(task_uuid, {})
        instance_status = task_data.get("instance", {}).get("status", "")
        assert instance_status != "new", (
            f"Task status must not be the stale DB initial value 'new'; got {instance_status!r}"
        )
        assert instance_status in ("running", "completed", "error"), (
            f"Unexpected task status: {instance_status!r}"
        )

    finally:
        # Best-effort cleanup — cancel if still running, then delete.
        if task_uuid:
            await tasks.post("/api/tasks/instances/cancel", json={"taskUuids": [task_uuid]})
            await tasks.post("/api/tasks/instances/delete", json={"taskUuids": [task_uuid]})
        if config_uuid:
            await tasks.post("/api/tasks/configs/delete", json={"configUuids": [config_uuid]})


# --- relay_buffer_server ---

async def test_streams_list(relay):
    r = await relay.get("/api/streams/list")
    assert r.status_code == 200
    assert r.json()["success"] is True


# --- end-to-end proxy (image_tagging_server -> downstream) ---

async def test_proxy_to_db(tagging):
    r = await tagging.get("/api/labels/")
    assert r.status_code == 200


async def test_proxy_to_inference(tagging):
    r = await tagging.get("/api/models/")
    assert r.status_code == 200


async def test_proxy_to_tasks(tagging):
    r = await tagging.get("/api/tasks/types/list")
    assert r.status_code == 200
