import asyncio
from unittest.mock import MagicMock

import db_client.models as db_models
import pytest

from apps.helpers.consts import TaskStatus
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


async def test_start_task_unknown_config_returns_failure(http_client, tasks_api_mock):
    tasks_api_mock.list_task_configs.return_value = MagicMock(configs={})
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
    assert data["tasks"][TASK_UUID]["config"]["typename"] == "TestTask"


async def test_cancel_running_task(http_client, tasks_api_mock):
    # Make the configured TestTask run for a short time so cancel has effect
    tasks_api_mock.list_task_configs.return_value.configs[CONFIG_UUID].params_json = {"duration": 5}

    await http_client.post(
        "/api/tasks/instances/start",
        json={"configUuid": CONFIG_UUID},
    )
    # Give the event loop a tick so the task has started
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


# ─── Tests for changes made to _start_task and get_tasks_instance_info ────────


async def test_start_task_writes_running_status_to_db(manager, tasks_api_mock):
    """_start_task must persist status='running' to the DB immediately on start."""
    # Use a long-running task so completion doesn't race with our assertion.
    tasks_api_mock.list_task_configs.return_value = MagicMock(
        configs={
            CONFIG_UUID: db_models.TaskConfigRead(
                uuid=CONFIG_UUID,
                name="test-config",
                typename="TestTask",
                params_json={"duration": 60.0},
                marked_for_delete=False,
                description=None,
            )
        }
    )

    await manager.start_new_task(CONFIG_UUID)

    running_writes = [
        c for c in tasks_api_mock.set_task_instance_status.call_args_list if c.args[1].status == "running"
    ]
    assert running_writes, "set_task_instance_status('running') must be called on task start"

    # Clean up so the background task doesn't outlive the test.
    await manager.cancel_tasks([TASK_UUID])


async def test_get_task_info_uses_in_memory_status_not_stale_db(manager, tasks_api_mock):
    """In-memory status ('running') must win over a stale DB record ('new').

    This is the regression guard for the bug where list_task_instances results
    unconditionally overwrote the running-tasks cache on a 'get all' query.
    """
    # Use a long-running task so it's still alive in _running_tasks when we query.
    tasks_api_mock.list_task_configs.return_value = MagicMock(
        configs={
            CONFIG_UUID: db_models.TaskConfigRead(
                uuid=CONFIG_UUID,
                name="test-config",
                typename="TestTask",
                params_json={"duration": 60.0},
                marked_for_delete=False,
                description=None,
            )
        }
    )
    # Configure the DB to return the instance with its initial 'new' status —
    # this simulates the period before the task's first DB status write arrives.
    stale_instance = db_models.TaskInstanceRead(
        uuid=TASK_UUID,
        config_uuid=CONFIG_UUID,
        status=TaskStatus.NEW,
        resume_data_json=None,
        result_json=None,
        error_message=None,
    )
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[stale_instance])

    await manager.start_new_task(CONFIG_UUID)
    await asyncio.sleep(0)  # yield so the task coroutine starts

    info = await manager.get_tasks_instance_info(None)

    assert TASK_UUID in info, "Running task must appear in results"
    assert (
        info[TASK_UUID].instance.status == TaskStatus.RUNNING
    ), "In-memory 'running' status must not be overwritten by stale DB 'new' status"

    await manager.cancel_tasks([TASK_UUID])
