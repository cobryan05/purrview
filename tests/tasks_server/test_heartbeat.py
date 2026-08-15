"""Tests for the task heartbeat mechanism.

Design: heartbeat() is rate-limited and called automatically by _check_pause().
Tasks also call heartbeat(force=True) before long single-await operations so
the timestamp is fresh going into them.

Covers:
  - heartbeat() fires when a looping task calls _check_pause()
  - heartbeat() is rate-limited (multiple _check_pause calls → one DB write per interval)
  - heartbeat(force=True) bypasses the rate limit
  - heartbeat() is a no-op until set_instance_uuid() has been called
  - set_instance_uuid is called before task.start() in _start_task
  - _check_stale_tasks marks orphaned tasks with stale heartbeats as error
  - _check_stale_tasks leaves live tasks (in _running_tasks) alone
  - _check_stale_tasks ignores tasks whose heartbeat_at is None (pre-heartbeat)
  - _check_stale_tasks ignores tasks whose heartbeat is still fresh
  - _check_stale_tasks handles multiple stale tasks in one pass
  - _check_stale_tasks handles timezone-naive datetimes from SQLite
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import db_client.models as db_models
import pytest

from apps.helpers.consts import TaskStatus
from apps.tasks_server.tasks.Registry import register_task
from apps.tasks_server.tasks.Task import Task
from tests.tasks_server.conftest import CONFIG_UUID, TASK_UUID

# ---------------------------------------------------------------------------
# A looping test task that calls _check_pause() every iteration.
# This is the integration point: _check_pause() → heartbeat().
# ---------------------------------------------------------------------------


@register_task("LoopingHeartbeatTask")
class LoopingHeartbeatTask(Task):
    async def _init(self, params: dict, resume_data: Optional[dict]) -> None:
        self._duration = float(params.get("duration", 0.2))

    async def _run(self) -> dict[str, Any]:
        deadline = asyncio.get_event_loop().time() + self._duration
        while asyncio.get_event_loop().time() < deadline and not self._cancel_flag.is_set():
            await self._check_pause()
            await asyncio.sleep(0.01)
        return {}

    async def _deinit(self) -> None:
        pass


def _looping_config(duration: float = 0.2) -> db_models.TaskConfigRead:
    return db_models.TaskConfigRead(
        uuid=CONFIG_UUID,
        name="looping-config",
        typename="LoopingHeartbeatTask",
        params={"duration": duration},
        marked_for_delete=False,
        description=None,
    )


# ---------------------------------------------------------------------------
# Helpers for _check_stale_tasks tests
# ---------------------------------------------------------------------------


def _make_instance(
    uuid: str,
    status: str,
    heartbeat_at: datetime | None = None,
) -> db_models.TaskInstanceRead:
    return db_models.TaskInstanceRead(
        uuid=uuid,
        config_uuid="cfg-uuid",
        status=status,
        resume_data=None,
        result=None,
        error_message=None,
        heartbeat_at=heartbeat_at,
    )


def _fresh_hb() -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=5)


def _stale_hb(interval: int = 30) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=interval * 5)


# ---------------------------------------------------------------------------
# heartbeat() integration via _check_pause()
# ---------------------------------------------------------------------------


async def test_heartbeat_fires_via_check_pause(manager, tasks_api_mock):
    """heartbeat() must be called (and reach the DB) when _check_pause() is invoked."""
    tasks_api_mock.set_task_instance_heartbeat.return_value = MagicMock()

    with patch("apps.tasks_server.tasks.Task.Task.HEARTBEAT_INTERVAL", 0.05):
        tasks_api_mock.list_task_configs.return_value = MagicMock(configs={CONFIG_UUID: _looping_config(duration=0.3)})
        await manager.start_new_task(CONFIG_UUID)
        await asyncio.sleep(0.2)

    assert tasks_api_mock.set_task_instance_heartbeat.call_count >= 1
    call_uuid = tasks_api_mock.set_task_instance_heartbeat.call_args.args[0]
    assert call_uuid == TASK_UUID


async def test_heartbeat_is_rate_limited(manager, tasks_api_mock):
    """Many _check_pause() calls within one interval should produce only one DB write."""
    tasks_api_mock.set_task_instance_heartbeat.return_value = MagicMock()

    # Long interval — heartbeat should fire at most once however many iterations run.
    with patch("apps.tasks_server.tasks.Task.Task.HEARTBEAT_INTERVAL", 60):
        tasks_api_mock.list_task_configs.return_value = MagicMock(configs={CONFIG_UUID: _looping_config(duration=0.2)})
        await manager.start_new_task(CONFIG_UUID)
        await asyncio.sleep(0.25)

    assert tasks_api_mock.set_task_instance_heartbeat.call_count == 1


async def test_heartbeat_force_bypasses_rate_limit(manager, tasks_api_mock):
    """heartbeat(force=True) must write to the DB even if the interval hasn't elapsed."""
    tasks_api_mock.set_task_instance_heartbeat.return_value = MagicMock()

    tasks_api_mock.list_task_configs.return_value = MagicMock(configs={CONFIG_UUID: _looping_config(duration=0.05)})
    task_info = await manager.start_new_task(CONFIG_UUID)
    await asyncio.sleep(0.1)  # let it finish

    # Manually call heartbeat(force=True) on the completed task — it should always write.
    task = task_info.task
    task._last_heartbeat_time = 1e18  # simulate "just sent"
    initial_count = tasks_api_mock.set_task_instance_heartbeat.call_count
    await task.heartbeat(force=True)
    assert tasks_api_mock.set_task_instance_heartbeat.call_count == initial_count + 1


async def test_heartbeat_noop_without_instance_uuid(manager, tasks_api_mock):
    """heartbeat() must be a no-op if set_instance_uuid() was never called."""
    tasks_api_mock.set_task_instance_heartbeat.return_value = MagicMock()

    tasks_api_mock.list_task_configs.return_value = MagicMock(configs={CONFIG_UUID: _looping_config(duration=0.05)})
    task_info = await manager.start_new_task(CONFIG_UUID)
    await asyncio.sleep(0.1)

    task = task_info.task
    task._instance_uuid = None  # clear it
    task._last_heartbeat_time = 0.0  # ensure rate limit won't block
    count_before = tasks_api_mock.set_task_instance_heartbeat.call_count
    await task.heartbeat()
    assert tasks_api_mock.set_task_instance_heartbeat.call_count == count_before


# ---------------------------------------------------------------------------
# set_instance_uuid wired up in _start_task
# ---------------------------------------------------------------------------


async def test_start_task_sets_instance_uuid(manager, tasks_api_mock):
    """_start_task must call task.set_instance_uuid() before task.start()."""
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
    await asyncio.sleep(0.05)

    assert task_info is not None
    assert task_info.task._instance_uuid == TASK_UUID


# ---------------------------------------------------------------------------
# _check_stale_tasks
#
# These set _recovery_done directly: the stale net is gated on the startup
# recovery pass having run (see test_restart_recovery.py), and what is under
# test here is the staleness logic itself, not the gate.
# ---------------------------------------------------------------------------


async def test_check_stale_tasks_marks_orphan_with_stale_hb_as_error(manager, tasks_api_mock):
    inst = _make_instance("orphan-uuid", TaskStatus.RUNNING, heartbeat_at=_stale_hb())
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
    manager._recovery_done = True

    await manager._check_stale_tasks()

    tasks_api_mock.set_task_instance_status.assert_called_once()
    uuid_arg, payload_arg = tasks_api_mock.set_task_instance_status.call_args.args
    assert uuid_arg == "orphan-uuid"
    assert payload_arg.status == TaskStatus.ERROR


async def test_check_stale_tasks_ignores_live_running_task(manager, tasks_api_mock):
    """A task present in _running_tasks must never be touched."""
    inst = _make_instance(TASK_UUID, TaskStatus.RUNNING, heartbeat_at=_stale_hb())
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
    manager._running_tasks[TASK_UUID] = MagicMock()
    manager._recovery_done = True

    await manager._check_stale_tasks()

    tasks_api_mock.set_task_instance_status.assert_not_called()
    manager._running_tasks.pop(TASK_UUID, None)


async def test_check_stale_tasks_ignores_no_heartbeat(manager, tasks_api_mock):
    """heartbeat_at=None means the task pre-dates heartbeat support — leave it alone."""
    inst = _make_instance("old-task-uuid", TaskStatus.RUNNING, heartbeat_at=None)
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
    manager._recovery_done = True

    await manager._check_stale_tasks()

    tasks_api_mock.set_task_instance_status.assert_not_called()


async def test_check_stale_tasks_ignores_fresh_heartbeat(manager, tasks_api_mock):
    inst = _make_instance("fresh-uuid", TaskStatus.RUNNING, heartbeat_at=_fresh_hb())
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
    manager._recovery_done = True

    await manager._check_stale_tasks()

    tasks_api_mock.set_task_instance_status.assert_not_called()


async def test_check_stale_tasks_ignores_non_running_statuses(manager, tasks_api_mock):
    instances = [
        _make_instance("done-uuid", TaskStatus.COMPLETED, heartbeat_at=_stale_hb()),
        _make_instance("err-uuid", TaskStatus.ERROR, heartbeat_at=_stale_hb()),
        _make_instance("paused-uuid", TaskStatus.PAUSED, heartbeat_at=_stale_hb()),
    ]
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=instances)
    manager._recovery_done = True

    await manager._check_stale_tasks()

    tasks_api_mock.set_task_instance_status.assert_not_called()


async def test_check_stale_tasks_handles_multiple_stale_instances(manager, tasks_api_mock):
    instances = [
        _make_instance("stale-1", TaskStatus.RUNNING, heartbeat_at=_stale_hb()),
        _make_instance("stale-2", TaskStatus.RUNNING, heartbeat_at=_stale_hb()),
        _make_instance("fresh-1", TaskStatus.RUNNING, heartbeat_at=_fresh_hb()),
    ]
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=instances)
    manager._recovery_done = True

    await manager._check_stale_tasks()

    assert tasks_api_mock.set_task_instance_status.call_count == 2
    marked = {c.args[0] for c in tasks_api_mock.set_task_instance_status.call_args_list}
    assert marked == {"stale-1", "stale-2"}


async def test_check_stale_tasks_tolerates_naive_datetime(manager, tasks_api_mock):
    """SQLite may return timezone-naive datetimes; the stale check must handle them."""
    naive_stale = datetime.utcnow() - timedelta(seconds=30 * 5)
    inst = _make_instance("naive-uuid", TaskStatus.RUNNING, heartbeat_at=naive_stale)
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
    manager._recovery_done = True

    await manager._check_stale_tasks()

    tasks_api_mock.set_task_instance_status.assert_called_once()
    uuid_arg, payload_arg = tasks_api_mock.set_task_instance_status.call_args.args
    assert uuid_arg == "naive-uuid"
    assert payload_arg.status == TaskStatus.ERROR


# ---------------------------------------------------------------------------
# _worker_task must keep running its periodic maintenance loop even when the
# one-time _init() recovery pass fails (e.g. db_server isn't reachable yet
# during a container boot race). _check_stale_tasks is the safety net for
# orphaned "running" rows that _init() failed to reconcile — if a failed
# _init() takes the whole loop down with it, that safety net never runs
# either, and a task instance orphaned by a crash/restart is stuck showing
# "running" forever with a frozen heartbeat, immune to Stop (which only
# checks the in-memory _running_tasks dict).
# ---------------------------------------------------------------------------


async def test_worker_task_runs_periodic_loop_even_when_init_fails(manager, tasks_api_mock):
    tasks_api_mock.list_task_instances.side_effect = ConnectionError("db_server not reachable yet")

    stale_check_ran = False

    async def _fake_check_stale_tasks():
        nonlocal stale_check_ran
        stale_check_ran = True

    manager._check_stale_tasks = _fake_check_stale_tasks
    manager._checkpoint_running_tasks = AsyncMock()
    manager._prune_finished_task_instances = AsyncMock()

    # Let the `while True` loop run exactly one pass, then stop it the same way
    # a real cancellation would (_worker_task already catches CancelledError
    # internally and returns cleanly) — avoids racing a real task's cancellation
    # against the event loop.
    sleep_calls = 0

    async def _sleep_once_then_stop(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError()

    with patch("apps.tasks_server.manager.asyncio.sleep", new=_sleep_once_then_stop):
        await manager._worker_task()

    assert stale_check_ran
