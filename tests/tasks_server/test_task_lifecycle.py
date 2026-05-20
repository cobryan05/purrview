import asyncio

import pytest

from tests.tasks_server.conftest import CONFIG_UUID, TASK_UUID


async def test_start_task_returns_task_uuid(http_client):
    resp = await http_client.post(
        "/api/tasks/instances/start",
        json={"configUuid": CONFIG_UUID},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["taskUuid"] == TASK_UUID


async def test_start_task_unknown_config_returns_failure(http_client, legacy_db):
    legacy_db.get_task_configs.return_value = {}
    resp = await http_client.post(
        "/api/tasks/instances/start",
        json={"configUuid": "nonexistent-uuid"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failure"


async def test_get_task_info_empty(http_client):
    resp = await http_client.post("/api/tasks/instances/get", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["tasks"] == {}


async def test_get_task_info_after_start(http_client):
    await http_client.post(
        "/api/tasks/instances/start",
        json={"configUuid": CONFIG_UUID},
    )
    resp = await http_client.post(
        "/api/tasks/instances/get",
        json={"taskUuids": [TASK_UUID]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert TASK_UUID in data["tasks"]
    assert data["tasks"][TASK_UUID]["configMetadata"]["typename"] == "TestTask"


async def test_cancel_running_task(http_client):
    await http_client.post(
        "/api/tasks/instances/start",
        json={"configUuid": CONFIG_UUID},
    )
    # Give the event loop a tick so the task can finish (TestTask is instant)
    await asyncio.sleep(0)

    resp = await http_client.post(
        "/api/tasks/instances/cancel",
        json={"taskUuids": [TASK_UUID]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert TASK_UUID in data["taskUuids"]


async def test_cancel_unknown_task_returns_empty_list(http_client):
    resp = await http_client.post(
        "/api/tasks/instances/cancel",
        json={"taskUuids": ["non-existent-uuid"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["taskUuids"] == []


async def test_delete_task(http_client):
    await http_client.post(
        "/api/tasks/instances/start",
        json={"configUuid": CONFIG_UUID},
    )
    resp = await http_client.post(
        "/api/tasks/instances/delete",
        json={"taskUuids": [TASK_UUID]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert TASK_UUID in data["taskUuids"]
