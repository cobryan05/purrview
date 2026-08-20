"""Tests for Phase 2B — cooperative in-place pause / resume.

These tests verify that:
- pause-in-place sets the _pause_flag without removing the task from _running_tasks
- resume-in-place clears the _pause_flag
- get_tasks_info reflects is_paused_in_place correctly
- _check_pause actually blocks a PausableTestTask loop until resumed
"""

import asyncio
from unittest.mock import MagicMock

import db_client.models as db_models
import pytest

from apps.helpers.consts import TaskStatus
from tests.tasks_server.conftest import CONFIG_UUID, TASK_UUID
from tests.waits import await_condition

# ── Helpers ─────────────────────────────────────────────────────────────────────


async def _start_pausable_task(manager, tasks_api_mock, iterations=50):
    tasks_api_mock.list_task_configs.return_value = MagicMock(
        configs={
            CONFIG_UUID: db_models.TaskConfigRead(
                uuid=CONFIG_UUID,
                name="test-config",
                typename="PausableTestTask",
                params={"iterations": iterations},
                marked_for_delete=False,
                description=None,
            )
        }
    )
    return await manager.start_new_task(CONFIG_UUID)


# ── Tests ────────────────────────────────────────────────────────────────────────


async def test_pause_in_place_sets_flag(manager, tasks_api_mock):
    """pause-in-place sets _pause_flag without removing task from _running_tasks."""
    await _start_pausable_task(manager, tasks_api_mock)
    await asyncio.sleep(0)  # one loop tick so task starts

    paused = await manager.pause_tasks_in_place([TASK_UUID])
    assert TASK_UUID in paused

    # Task still in _running_tasks
    assert TASK_UUID in manager._running_tasks

    task = manager._running_tasks[TASK_UUID].task
    assert task.is_paused_in_place() is True

    await manager.cancel_tasks([TASK_UUID])


async def test_resume_in_place_clears_flag(manager, tasks_api_mock):
    """resume-in-place clears _pause_flag."""
    await _start_pausable_task(manager, tasks_api_mock)
    await asyncio.sleep(0)

    await manager.pause_tasks_in_place([TASK_UUID])
    task = manager._running_tasks[TASK_UUID].task
    assert task.is_paused_in_place() is True

    resumed = await manager.resume_tasks_in_place([TASK_UUID])
    assert TASK_UUID in resumed
    assert task.is_paused_in_place() is False

    await manager.cancel_tasks([TASK_UUID])


async def test_get_tasks_info_reports_is_paused_in_place(manager, tasks_api_mock, http_client):
    """get_tasks_info endpoint includes is_paused_in_place=true after pause-in-place."""
    await _start_pausable_task(manager, tasks_api_mock)
    await asyncio.sleep(0)

    await manager.pause_tasks_in_place([TASK_UUID])

    resp = await http_client.post(
        "/api/tasks/instances/get",
        json={"taskUuids": [TASK_UUID]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["tasks"][TASK_UUID]["isPausedInPlace"] is True

    await manager.cancel_tasks([TASK_UUID])


async def test_get_tasks_info_reports_is_paused_in_place_false_after_resume(manager, tasks_api_mock, http_client):
    """is_paused_in_place is false after resume-in-place."""
    await _start_pausable_task(manager, tasks_api_mock)
    await asyncio.sleep(0)

    await manager.pause_tasks_in_place([TASK_UUID])
    await manager.resume_tasks_in_place([TASK_UUID])

    resp = await http_client.post(
        "/api/tasks/instances/get",
        json={"taskUuids": [TASK_UUID]},
    )
    data = resp.json()
    assert data["tasks"][TASK_UUID]["isPausedInPlace"] is False

    await manager.cancel_tasks([TASK_UUID])


async def test_check_pause_blocks_loop_until_resumed(manager, tasks_api_mock):
    """A PausableTestTask loop blocks while paused and completes after resume."""
    await _start_pausable_task(manager, tasks_api_mock, iterations=5)
    await await_condition(
        lambda: TASK_UUID in manager._running_tasks,
        "the task never reached the running set",
    )

    await manager.pause_tasks_in_place([TASK_UUID])
    task = manager._running_tasks[TASK_UUID].task
    assert task.is_paused_in_place() is True

    # Task should NOT be done yet (blocked). Still a sleep, and correctly so: this
    # asserts the loop *stays* blocked, and a non-event cannot be polled for. A slower
    # box only makes the window more generous to the thing under test.
    await asyncio.sleep(0.15)
    assert not task.is_task_done(), "Task should be blocked in _check_pause"

    # Resume — task should be able to finish
    await manager.resume_tasks_in_place([TASK_UUID])
    await task.wait_for_task_done(timeout=3.0)
    assert task.is_task_done()


async def test_pause_in_place_ignores_nonrunning_task(manager, tasks_api_mock):
    """pause-in-place silently skips tasks not in _running_tasks."""
    paused = await manager.pause_tasks_in_place(["nonexistent-uuid"])
    assert paused == []


async def test_resume_in_place_ignores_nonrunning_task(manager, tasks_api_mock):
    """resume-in-place silently skips tasks not in _running_tasks."""
    resumed = await manager.resume_tasks_in_place(["nonexistent-uuid"])
    assert resumed == []


async def test_pause_in_place_endpoint(http_client, manager, tasks_api_mock):
    """POST /pause-in-place returns the paused task UUID."""
    await _start_pausable_task(manager, tasks_api_mock)
    await asyncio.sleep(0)

    resp = await http_client.post(
        "/api/tasks/instances/pause-in-place",
        json={"taskUuids": [TASK_UUID]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert TASK_UUID in data["taskUuids"]

    await manager.cancel_tasks([TASK_UUID])


async def test_resume_in_place_endpoint(http_client, manager, tasks_api_mock):
    """POST /resume-in-place clears the pause flag and returns the task UUID."""
    await _start_pausable_task(manager, tasks_api_mock)
    await asyncio.sleep(0)

    await manager.pause_tasks_in_place([TASK_UUID])

    resp = await http_client.post(
        "/api/tasks/instances/resume-in-place",
        json={"taskUuids": [TASK_UUID]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert TASK_UUID in data["taskUuids"]

    task = manager._running_tasks[TASK_UUID].task
    assert task.is_paused_in_place() is False

    await manager.cancel_tasks([TASK_UUID])
