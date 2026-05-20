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
