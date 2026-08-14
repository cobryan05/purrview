"""Tests for Phase 2A — persisted start/end times and logs via tasks_server manager.

These tests verify that _start_task records started_at and _save_task_data
records ended_at + log_json by inspecting the mock TasksApi call args.
"""

import asyncio
from unittest.mock import MagicMock, call

import db_client.models as db_models
import pytest

from apps.helpers.consts import TaskStatus
from tests.tasks_server.conftest import CONFIG_UUID, TASK_UUID


async def _start_long_task(manager, tasks_api_mock, duration=60.0):
    tasks_api_mock.list_task_configs.return_value = MagicMock(
        configs={
            CONFIG_UUID: db_models.TaskConfigRead(
                uuid=CONFIG_UUID,
                name="test-config",
                typename="TestTask",
                params={"duration": duration},
                marked_for_delete=False,
                description=None,
            )
        }
    )
    return await manager.start_new_task(CONFIG_UUID)


async def test_start_task_records_started_at(manager, tasks_api_mock):
    """_start_task must call set_task_instance_timestamps with started_at immediately."""
    await _start_long_task(manager, tasks_api_mock)

    timestamp_calls = tasks_api_mock.set_task_instance_timestamps.call_args_list
    assert len(timestamp_calls) >= 1, "set_task_instance_timestamps must be called on start"

    # At least one call should have a non-None started_at
    started_at_calls = [c for c in timestamp_calls if c.args[1].started_at is not None]
    assert started_at_calls, "started_at must be set on task start"

    await manager.cancel_tasks([TASK_UUID])


async def test_completed_task_records_ended_at_and_logs(manager, tasks_api_mock):
    """When a task completes, set_task_instance_timestamps is called with ended_at + log_json."""
    tasks_api_mock.list_task_configs.return_value = MagicMock(
        configs={
            CONFIG_UUID: db_models.TaskConfigRead(
                uuid=CONFIG_UUID,
                name="test-config",
                typename="TestTask",
                params={"duration": 0.0},
                marked_for_delete=False,
                description=None,
            )
        }
    )
    task_info = await manager.start_new_task(CONFIG_UUID)
    # Wait for completion
    await task_info.task.wait_for_task_done(timeout=5.0)
    await asyncio.sleep(0.05)  # Give _on_task_complete a tick to run

    timestamp_calls = tasks_api_mock.set_task_instance_timestamps.call_args_list
    ended_calls = [c for c in timestamp_calls if c.args[1].ended_at is not None]
    assert ended_calls, "ended_at must be set after task completes"

    # log_json should be a list (possibly empty for this trivial task)
    log_call = ended_calls[-1]
    assert log_call.args[1].log_json is not None
    assert isinstance(log_call.args[1].log_json, list)


async def test_is_paused_in_place_false_for_running_task(manager, tasks_api_mock):
    """A freshly started task should NOT be paused in place."""
    await _start_long_task(manager, tasks_api_mock)

    resp = await manager.get_tasks_instance_info([TASK_UUID])
    task_info = resp.get(TASK_UUID)
    assert task_info is not None
    assert task_info.task.is_paused_in_place() is False

    await manager.cancel_tasks([TASK_UUID])


class TestResumeClearsPreviousRunResult:
    """Resuming reuses the instance row, so the previous run's one-shot result
    must not survive into the new run — otherwise the instance reports status
    "running" while carrying a "completed" result that is never consumed, and
    every task poll ships that dead payload to the UI forever.
    """

    async def test_resume_clears_the_previous_runs_result(self, manager, tasks_api_mock):
        stale_result = {"status": "completed", "data": {"images_saved": 561}}
        paused = db_models.TaskInstanceRead(
            uuid=TASK_UUID,
            config_uuid=CONFIG_UUID,
            status=TaskStatus.PAUSED,
            resume_data={},
            result=stale_result,
            error_message=None,
        )
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[paused])

        resumed = await manager.resume_tasks([TASK_UUID])
        assert resumed == [TASK_UUID]

        cleared = [c for c in tasks_api_mock.set_task_instance_result.call_args_list if not c.args[1].result]
        assert cleared, "resuming must clear the previous run's result"
        assert not paused.result, "the in-memory instance must not keep serving the stale result"

        await manager.cancel_tasks([TASK_UUID])
