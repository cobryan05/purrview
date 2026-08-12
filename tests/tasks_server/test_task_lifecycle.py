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
    tasks_api_mock.list_task_configs.return_value.configs[CONFIG_UUID].params = {"duration": 5}

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


async def test_cancel_tasks_survives_asyncio_timeout(manager, tasks_api_mock):
    """cancel_tasks must catch asyncio.TimeoutError from wait_for_task_done.

    On Python 3.10 asyncio.TimeoutError is a distinct class from the builtin
    TimeoutError, so `except TimeoutError` fails to catch it: the exception
    escapes the loop, 500s the cancel endpoint, and every task after the slow
    one never gets its final status persisted.
    """
    tasks_api_mock.list_task_configs.return_value.configs[CONFIG_UUID].params = {"duration": 60}
    await manager.start_new_task(CONFIG_UUID)
    await asyncio.sleep(0)

    task_info = manager._running_tasks[TASK_UUID]

    async def _raise_asyncio_timeout(timeout=None):
        raise asyncio.TimeoutError("simulated slow-to-cancel task")

    task_info.task.wait_for_task_done = _raise_asyncio_timeout

    # Must not raise: a task that times out is skipped, not fatal.
    result = await manager.cancel_tasks([TASK_UUID])
    assert result == []  # timed-out task is not reported as cleanly stopped

    task_info.task.cancel_task()


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
                params={"duration": 60.0},
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
                params={"duration": 60.0},
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
        resume_data=None,
        result=None,
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


# ─── progress: Optional[float] wire round-trip ─────────────────────────────────
#
# Task._progress / TaskInfoRead.progress must distinguish "indeterminate" (None,
# which the frontend renders as a striped bar) from "determinate 0%" (0.0, a
# normal progress bar at the start). Both must round-trip through the actual
# JSON API response, not just the Python object, since a naive `progress or 0.0`
# anywhere in the serialization path would silently collapse the distinction.


async def test_indeterminate_progress_serializes_as_null(http_client, tasks_api_mock):
    """A task that reports None progress must serialize as JSON null, not 0."""
    tasks_api_mock.list_task_configs.return_value.configs[CONFIG_UUID].typename = "IndeterminateProgressTask"

    await http_client.post("/api/tasks/instances/start", json={"configUuid": CONFIG_UUID})
    await asyncio.sleep(0)  # let the task run far enough to call _update_progress(None)

    resp = await http_client.post("/api/tasks/instances/get", json={"taskUuids": [TASK_UUID]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["tasks"][TASK_UUID]["progress"] is None

    await http_client.post("/api/tasks/instances/cancel", json={"taskUuids": [TASK_UUID]})


async def test_determinate_zero_progress_serializes_as_zero_not_null(http_client, tasks_api_mock):
    """A task that never touches progress (base Task default) must serialize as
    0.0/0, not null — the base default must stay determinate so every pre-existing
    batch task keeps rendering a normal (non-striped) progress bar."""
    tasks_api_mock.list_task_configs.return_value.configs[CONFIG_UUID].params = {"duration": 60}

    await http_client.post("/api/tasks/instances/start", json={"configUuid": CONFIG_UUID})
    await asyncio.sleep(0)

    resp = await http_client.post("/api/tasks/instances/get", json={"taskUuids": [TASK_UUID]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["tasks"][TASK_UUID]["progress"] == 0.0
    assert data["tasks"][TASK_UUID]["progress"] is not None

    await http_client.post("/api/tasks/instances/cancel", json={"taskUuids": [TASK_UUID]})


async def test_failed_task_persists_its_error_message(manager, tasks_api_mock):
    """End-to-end: a task that raises must have the reason written to the DB
    alongside status='error'. The result payload carrying the same text is
    one-shot — it's deleted on first read — so this is what makes the failure
    still explainable on the card afterwards."""
    # Registered in test_real_task_execution; importing it here would be circular,
    # so reuse the registry the way the manager itself does.
    import tests.tasks_server.test_real_task_execution  # noqa: F401

    tasks_api_mock.list_task_configs.return_value.configs[CONFIG_UUID].typename = "ErrorTask"
    tasks_api_mock.list_task_configs.return_value.configs[CONFIG_UUID].params = {}

    await manager.start_new_task(CONFIG_UUID)
    # Let _execute run to completion and the manager's completion path persist.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if any(c.args[1].status == "error" for c in tasks_api_mock.set_task_instance_status.call_args_list):
            break

    error_writes = [c for c in tasks_api_mock.set_task_instance_status.call_args_list if c.args[1].status == "error"]
    assert error_writes, "a raising task must write status='error'"
    assert error_writes[-1].args[1].error_message == "ValueError: deliberate error"


async def test_start_task_persists_running_before_starting_execution(manager, tasks_api_mock, monkeypatch):
    """_start_task must persist status='running' to the DB (and register the
    completion handler) BEFORE calling task.start().

    A task whose _init()/_run() fails entirely synchronously (no real await —
    e.g. ErrorTask) can run to completion, and have _on_task_complete write its
    terminal ERROR status, before this RUNNING write's own thread-pool call
    would otherwise have landed. If task.start() ran first (the old order),
    the late RUNNING write can then clobber the terminal status: the task is
    popped from the in-memory running-tasks registry (so Stop/Suspend become
    silent no-ops and the active-task count excludes it) while the DB/UI is
    left showing "running" forever.
    """
    import tests.tasks_server.test_real_task_execution  # noqa: F401 (registers ErrorTask)

    tasks_api_mock.list_task_configs.return_value.configs[CONFIG_UUID].typename = "ErrorTask"
    tasks_api_mock.list_task_configs.return_value.configs[CONFIG_UUID].params = {}

    order = []

    def _record_status(uuid, payload):
        if payload.status == "running":
            order.append("running_write")
        return MagicMock()

    tasks_api_mock.set_task_instance_status.side_effect = _record_status

    from apps.tasks_server.tasks.Task import Task

    orig_start = Task.start

    async def _tracked_start(self):
        order.append("task_start")
        return await orig_start(self)

    monkeypatch.setattr(Task, "start", _tracked_start)

    await manager.start_new_task(CONFIG_UUID)
    for _ in range(50):
        if "task_start" in order:
            break
        await asyncio.sleep(0.01)

    assert order[:2] == [
        "running_write",
        "task_start",
    ], f"the RUNNING status write must be persisted before task.start() is called, got: {order}"
