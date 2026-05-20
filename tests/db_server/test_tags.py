import pytest


async def test_list_tags_empty(http_client):
    r = await http_client.get("/api/tags/list")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["tags"] == []


async def test_create_tag(http_client):
    r = await http_client.post("/api/tags/add", json={"name": "occluded", "color": "#FF0000"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["tag"]["name"] == "occluded"
    assert body["tag"]["color"] == "#FF0000"
    assert "uuid" in body["tag"]


async def test_create_and_list_tag(http_client):
    await http_client.post("/api/tags/add", json={"name": "truncated", "color": "#00FF00"})
    r = await http_client.get("/api/tags/list")
    body = r.json()
    assert body["status"] == "success"
    names = [t["name"] for t in body["tags"]]
    assert "truncated" in names


async def test_update_tag(http_client):
    create_r = await http_client.post("/api/tags/add", json={"name": "blurry", "color": "#0000FF"})
    uuid = create_r.json()["tag"]["uuid"]

    r = await http_client.post("/api/tags/update", json={"tag_uuid": uuid, "data": {"name": "out-of-focus"}})
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    list_r = await http_client.get("/api/tags/list")
    names = [t["name"] for t in list_r.json()["tags"]]
    assert "out-of-focus" in names
    assert "blurry" not in names


async def test_delete_tag(http_client):
    create_r = await http_client.post("/api/tags/add", json={"name": "small", "color": "#AABBCC"})
    uuid = create_r.json()["tag"]["uuid"]

    r = await http_client.post("/api/tags/delete", json={"tag_uuid": uuid})
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    list_r = await http_client.get("/api/tags/list")
    names = [t["name"] for t in list_r.json()["tags"]]
    assert "small" not in names


async def test_create_tag_duplicate_name_fails(http_client):
    await http_client.post("/api/tags/add", json={"name": "rare", "color": "#123456"})
    r = await http_client.post("/api/tags/add", json={"name": "rare", "color": "#654321"})
    assert r.status_code == 200
    assert r.json()["status"] == "failure"
