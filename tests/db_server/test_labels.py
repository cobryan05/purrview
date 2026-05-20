import pytest


async def test_list_labels_empty(http_client):
    r = await http_client.get("/api/labels")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["labels"] == []


async def test_create_label(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["label"]["name"] == "cat"
    assert body["label"]["color"] == "#FF0000"
    assert "uuid" in body["label"]


async def test_create_and_list_label(http_client):
    await http_client.post("/api/labels", json={"name": "dog", "color": "#00FF00"})
    r = await http_client.get("/api/labels")
    body = r.json()
    assert body["status"] == "success"
    names = [lbl["name"] for lbl in body["labels"]]
    assert "dog" in names


async def test_update_label(http_client):
    create_r = await http_client.post("/api/labels", json={"name": "bird", "color": "#0000FF"})
    uuid = create_r.json()["label"]["uuid"]

    r = await http_client.patch(f"/api/labels/{uuid}", json={"name": "parrot", "color": "#FFFF00"})
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    list_r = await http_client.get("/api/labels")
    names = [lbl["name"] for lbl in list_r.json()["labels"]]
    assert "parrot" in names
    assert "bird" not in names


async def test_delete_label(http_client):
    create_r = await http_client.post("/api/labels", json={"name": "fish", "color": "#AABBCC"})
    uuid = create_r.json()["label"]["uuid"]

    r = await http_client.delete(f"/api/labels/{uuid}")
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    list_r = await http_client.get("/api/labels")
    names = [lbl["name"] for lbl in list_r.json()["labels"]]
    assert "fish" not in names


async def test_delete_label_with_children_fails(http_client):
    parent_r = await http_client.post("/api/labels", json={"name": "animal", "color": "#111111"})
    parent_uuid = parent_r.json()["label"]["uuid"]

    await http_client.post("/api/labels", json={"name": "dog", "color": "#222222", "parent_uuid": parent_uuid})

    r = await http_client.delete(f"/api/labels/{parent_uuid}")
    assert r.status_code == 200
    assert r.json()["status"] == "failure"


async def test_create_child_label(http_client):
    parent_r = await http_client.post("/api/labels", json={"name": "vehicle", "color": "#333333"})
    parent_uuid = parent_r.json()["label"]["uuid"]

    child_r = await http_client.post(
        "/api/labels", json={"name": "car", "color": "#444444", "parent_uuid": parent_uuid}
    )
    assert child_r.status_code == 200
    body = child_r.json()
    assert body["status"] == "success"
    assert body["label"]["parent_uuid"] == parent_uuid
