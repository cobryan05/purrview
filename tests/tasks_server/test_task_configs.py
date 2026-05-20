import pytest

from tests.tasks_server.conftest import CONFIG_UUID


async def test_list_task_types(http_client):
    resp = await http_client.get("/api/tasks/types/list")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "TestTask" in data["types"]


async def test_get_task_type_schema(http_client):
    resp = await http_client.post(
        "/api/tasks/types/schema",
        json={"taskTypenames": ["TestTask"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "TestTask" in data["schemas"]


async def test_get_task_type_schema_unknown_raises_failure(http_client):
    resp = await http_client.post(
        "/api/tasks/types/schema",
        json={"taskTypenames": ["NonExistentTask"]},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failure"


async def test_create_task_config(http_client):
    resp = await http_client.post(
        "/api/tasks/configs/create",
        json={"name": "my-config", "typename": "TestTask", "params": {}, "persistent": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["configUuid"] == CONFIG_UUID


async def test_create_task_config_unknown_type_returns_failure(http_client):
    resp = await http_client.post(
        "/api/tasks/configs/create",
        json={"name": "bad", "typename": "UnknownTask", "params": {}, "persistent": True},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failure"


async def test_get_task_configs(http_client):
    resp = await http_client.post(
        "/api/tasks/configs/get",
        json={"configUuids": [CONFIG_UUID]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert CONFIG_UUID in data["configs"]
    assert data["configs"][CONFIG_UUID]["typename"] == "TestTask"


async def test_get_all_task_configs(http_client):
    resp = await http_client.post("/api/tasks/configs/get", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"


async def test_delete_task_configs(http_client):
    resp = await http_client.post(
        "/api/tasks/configs/delete",
        json={"configUuids": [CONFIG_UUID]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert CONFIG_UUID in data["configUuids"]


async def test_update_task_config(http_client):
    resp = await http_client.post(
        "/api/tasks/configs/update",
        json={"configUuid": CONFIG_UUID, "name": "updated-name"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


async def test_update_task_config_unknown_type_returns_failure(http_client):
    resp = await http_client.post(
        "/api/tasks/configs/update",
        json={"configUuid": CONFIG_UUID, "typename": "UnknownTask"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failure"


async def test_server_config(http_client):
    resp = await http_client.get("/server-config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "db_server" in data["config"]
    assert "inference_server" in data["config"]
