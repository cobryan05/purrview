"""Tests for the model configurations feature — CRUD, validation, and cascade behavior."""

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _create_label(http_client, name: str, parent_uuid: str | None = None) -> str:
    """Create a label and return its uuid."""
    payload: dict = {"name": name, "color": "#ff0000"}
    if parent_uuid is not None:
        payload["parentUuid"] = parent_uuid
    r = await http_client.post("/api/labels", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["label"]["uuid"]


async def _create_config(
    http_client,
    name: str,
    nodes: list | None = None,
    description: str | None = None,
) -> dict:
    """Create a model config and return the config dict."""
    payload: dict = {"name": name, "nodes": nodes or []}
    if description is not None:
        payload["description"] = description
    r = await http_client.post("/api/model-configs", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success", body
    return body["config"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


async def test_typenames_endpoint(http_client):
    r = await http_client.get("/api/model-configs/typenames")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    typenames = body["typenames"]
    assert "YoloV8" in typenames
    assert "TestModel" in typenames


async def test_create_and_list(http_client):
    cfg = await _create_config(http_client, "MyConfig")
    assert cfg["name"] == "MyConfig"
    assert cfg["uuid"]

    r = await http_client.get("/api/model-configs")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    uuids = [c["uuid"] for c in body["configs"]]
    assert cfg["uuid"] in uuids


async def test_get_single(http_client):
    cfg = await _create_config(http_client, "SingleConfig", description="desc")
    r = await http_client.get(f"/api/model-configs/{cfg['uuid']}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["config"]["uuid"] == cfg["uuid"]
    assert body["config"]["description"] == "desc"


async def test_create_with_nodes(http_client):
    label_uuid = await _create_label(http_client, "Cat")
    nodes = [
        {"label_uuid": None, "model_typename": "YoloV8"},
        {"label_uuid": label_uuid, "model_typename": "TestModel"},
    ]
    cfg = await _create_config(http_client, "WithNodes", nodes=nodes)
    assert len(cfg["nodes"]) == 2
    typenames = {n["model_typename"] for n in cfg["nodes"]}
    assert typenames == {"YoloV8", "TestModel"}
    root_node = next(n for n in cfg["nodes"] if n["label_uuid"] is None)
    assert root_node["model_typename"] == "YoloV8"


async def test_update_scalar_fields(http_client):
    cfg = await _create_config(http_client, "OldName", description="old")
    r = await http_client.patch(
        f"/api/model-configs/{cfg['uuid']}",
        json={"name": "NewName", "description": "new"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["config"]["name"] == "NewName"
    assert body["config"]["description"] == "new"
    # Nodes unchanged (none were set)
    assert body["config"]["nodes"] == cfg["nodes"]


async def test_update_nodes_wholesale(http_client):
    label_uuid = await _create_label(http_client, "Dog")
    cfg = await _create_config(
        http_client,
        "NodeReplace",
        nodes=[{"label_uuid": None, "model_typename": "YoloV8"}],
    )
    assert len(cfg["nodes"]) == 1

    new_nodes = [
        {"label_uuid": None, "model_typename": "TestModel"},
        {"label_uuid": label_uuid, "model_typename": "YoloV8"},
    ]
    r = await http_client.patch(
        f"/api/model-configs/{cfg['uuid']}",
        json={"nodes": new_nodes},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert len(body["config"]["nodes"]) == 2
    typenames = {n["model_typename"] for n in body["config"]["nodes"]}
    assert typenames == {"YoloV8", "TestModel"}


async def test_update_nodes_clear(http_client):
    cfg = await _create_config(
        http_client,
        "ClearNodes",
        nodes=[{"label_uuid": None, "model_typename": "YoloV8"}],
    )
    assert len(cfg["nodes"]) == 1

    r = await http_client.patch(
        f"/api/model-configs/{cfg['uuid']}",
        json={"nodes": []},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["config"]["nodes"] == []


async def test_delete(http_client):
    cfg = await _create_config(http_client, "ToDelete")
    r = await http_client.delete(f"/api/model-configs/{cfg['uuid']}")
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    r2 = await http_client.get("/api/model-configs")
    uuids = [c["uuid"] for c in r2.json()["configs"]]
    assert cfg["uuid"] not in uuids


async def test_nodes_cascade_on_delete(http_client):
    label_uuid = await _create_label(http_client, "Bird")
    cfg = await _create_config(
        http_client,
        "CascadeTest",
        nodes=[
            {"label_uuid": None, "model_typename": "YoloV8"},
            {"label_uuid": label_uuid, "model_typename": "TestModel"},
        ],
    )
    assert len(cfg["nodes"]) == 2

    r = await http_client.delete(f"/api/model-configs/{cfg['uuid']}")
    assert r.json()["status"] == "success"

    # Config is gone; no 404-type error but config field is null
    r2 = await http_client.get(f"/api/model-configs/{cfg['uuid']}")
    body = r2.json()
    assert body["config"] is None


async def test_label_delete_cascades_node(http_client):
    """Deleting a label removes its node row; the config itself survives."""
    label_uuid = await _create_label(http_client, "Fish")
    cfg = await _create_config(
        http_client,
        "LabelCascade",
        nodes=[
            {"label_uuid": None, "model_typename": "YoloV8"},
            {"label_uuid": label_uuid, "model_typename": "TestModel"},
        ],
    )
    assert len(cfg["nodes"]) == 2

    # Delete the label
    r = await http_client.delete(f"/api/labels/{label_uuid}")
    assert r.json()["status"] == "success"

    # Config should still exist with only the root node remaining
    r2 = await http_client.get(f"/api/model-configs/{cfg['uuid']}")
    body = r2.json()
    assert body["status"] == "success"
    assert body["config"] is not None
    remaining_nodes = body["config"]["nodes"]
    assert len(remaining_nodes) == 1
    assert remaining_nodes[0]["label_uuid"] is None


async def test_invalid_typename_rejected(http_client):
    r = await http_client.post(
        "/api/model-configs",
        json={"name": "BadTypename", "nodes": [{"label_uuid": None, "model_typename": "NonExistentModel"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failure"
    assert "NonExistentModel" in body.get("message", "")


async def test_duplicate_label_uuid_rejected(http_client):
    label_uuid = await _create_label(http_client, "Rabbit")
    r = await http_client.post(
        "/api/model-configs",
        json={
            "name": "DupLabel",
            "nodes": [
                {"label_uuid": label_uuid, "model_typename": "YoloV8"},
                {"label_uuid": label_uuid, "model_typename": "TestModel"},
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failure"


async def test_root_node_label_uuid_none(http_client):
    """label_uuid=null is valid as the root node."""
    cfg = await _create_config(
        http_client,
        "RootNull",
        nodes=[{"label_uuid": None, "model_typename": "YoloV8"}],
    )
    assert len(cfg["nodes"]) == 1
    assert cfg["nodes"][0]["label_uuid"] is None
    assert cfg["nodes"][0]["model_typename"] == "YoloV8"
