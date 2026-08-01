"""Tests for Phase 2 backend durability: periodic checkpointing and auto-restart
of PERSISTENT task configs across a simulated server restart.

Covers:
  - Manager._checkpoint_running_tasks() writes resume data via the NARROW
    set_task_instance_resume_data path only — never touches result/status/
    ended_at, which would clobber a task that is still running.
  - CHECKPOINT_INTERVAL is respected: a task is not asked to checkpoint more
    often than its own interval.
  - Manager._init() auto-resumes a PERSISTENT config's interrupted-but-
    checkpointed instance; a non-persistent config is left PAUSED instead.
  - The crash-loop guard stops auto-restarting after N restarts within its
    window, and decays back to a clean slate once the window has elapsed.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import db_client.models as db_models
import pytest

from apps.helpers.consts import TaskStatus
from apps.tasks_server.manager import Manager
from apps.tasks_server.tasks.Task import Task
from apps.tasks_server.types import TaskInfo

# ─── Test helpers ─────────────────────────────────────────────────────────────


def _make_instance(
    uuid: str, status: str, resume_data=None, config_uuid: str = "cfg-uuid"
) -> db_models.TaskInstanceRead:
    return db_models.TaskInstanceRead(
        uuid=uuid,
        config_uuid=config_uuid,
        status=status,
        resume_data=resume_data,
        result=None,
        error_message=None,
    )


def _make_config(uuid: str, params: dict, typename: str = "TestTask") -> db_models.TaskConfigRead:
    return db_models.TaskConfigRead(
        uuid=uuid,
        name="live-config",
        typename=typename,
        params=params,
        marked_for_delete=False,
        description=None,
    )


class _FakeRunningTask:
    """Minimal stand-in for a running Task, giving the checkpoint pass full
    control over the is_data_ready()/get_resume_data()/request_data_update()
    handshake without actually running a task loop."""

    def __init__(self, resume_data=None, data_ready: bool = False, checkpoint_interval=None):
        self._resume_data = resume_data or {}
        self._ready = data_ready
        self.request_data_update = MagicMock(side_effect=self._on_request)
        self.clear_data_ready = MagicMock(side_effect=self._on_clear)
        if checkpoint_interval is not None:
            self.CHECKPOINT_INTERVAL = checkpoint_interval

    def _on_request(self):
        pass

    def _on_clear(self):
        self._ready = False

    def is_data_ready(self) -> bool:
        return self._ready

    def get_resume_data(self) -> dict:
        return dict(self._resume_data)

    def set_ready(self, resume_data: dict) -> None:
        self._resume_data = resume_data
        self._ready = True


def _put_running(manager, uuid: str, task, status: str = TaskStatus.RUNNING) -> None:
    manager._running_tasks[uuid] = TaskInfo(
        config=_make_config(f"cfg-for-{uuid}", params={}),
        instance=_make_instance(uuid, status),
        task=task,
    )


# ─── _checkpoint_running_tasks: narrow write path ──────────────────────────────


async def test_checkpoint_writes_resume_data_without_touching_status_result_or_ended_at(manager, tasks_api_mock):
    task = _FakeRunningTask(checkpoint_interval=300)
    task.set_ready({"images_saved": 7})
    _put_running(manager, "live-uuid", task)

    await manager._checkpoint_running_tasks()

    tasks_api_mock.set_task_instance_resume_data.assert_called_once()
    uuid_arg, payload = tasks_api_mock.set_task_instance_resume_data.call_args.args
    assert uuid_arg == "live-uuid"
    assert payload.resume_data == {"images_saved": 7}

    # The narrow path must not touch result, status, or timestamps — those belong
    # to _save_task_data, which is only used when a task truly completes/pauses.
    tasks_api_mock.set_task_instance_result.assert_not_called()
    tasks_api_mock.set_task_instance_status.assert_not_called()
    tasks_api_mock.set_task_instance_timestamps.assert_not_called()

    task.clear_data_ready.assert_called_once()


async def test_checkpoint_pass_ignores_tasks_with_no_data_ready(manager, tasks_api_mock):
    task = _FakeRunningTask(checkpoint_interval=300)  # never marked ready
    _put_running(manager, "live-uuid", task)

    await manager._checkpoint_running_tasks()

    tasks_api_mock.set_task_instance_resume_data.assert_not_called()


async def test_checkpoint_pass_skips_non_running_instances(manager, tasks_api_mock):
    task = _FakeRunningTask(checkpoint_interval=300)
    task.set_ready({"x": 1})
    _put_running(manager, "paused-uuid", task, status=TaskStatus.PAUSED)

    await manager._checkpoint_running_tasks()

    tasks_api_mock.set_task_instance_resume_data.assert_not_called()
    task.request_data_update.assert_not_called()


# ─── CHECKPOINT_INTERVAL is respected ──────────────────────────────────────────


async def test_checkpoint_interval_not_exceeded(manager, tasks_api_mock, monkeypatch):
    task = _FakeRunningTask(checkpoint_interval=1000)  # effectively "never" within this test
    _put_running(manager, "live-uuid", task)

    fake_now = [1000.0]
    monkeypatch.setattr("apps.tasks_server.manager.time.monotonic", lambda: fake_now[0])

    await manager._checkpoint_running_tasks()
    assert task.request_data_update.call_count == 1

    # A second pass immediately after must not re-request — interval hasn't elapsed.
    await manager._checkpoint_running_tasks()
    assert task.request_data_update.call_count == 1


async def test_checkpoint_interval_elapsed_triggers_new_request(manager, tasks_api_mock, monkeypatch):
    task = _FakeRunningTask(checkpoint_interval=60)
    _put_running(manager, "live-uuid", task)

    fake_now = [1000.0]
    monkeypatch.setattr("apps.tasks_server.manager.time.monotonic", lambda: fake_now[0])

    await manager._checkpoint_running_tasks()
    assert task.request_data_update.call_count == 1

    fake_now[0] += 61  # interval elapsed
    await manager._checkpoint_running_tasks()
    assert task.request_data_update.call_count == 2


# ─── Auto-restart of PERSISTENT configs ────────────────────────────────────────


async def test_persistent_config_auto_resumes_after_restart(manager, tasks_api_mock, monkeypatch):
    cfg_uuid = "persist-cfg"
    inst = _make_instance("run-chk", TaskStatus.RUNNING, resume_data={"images_saved": 3}, config_uuid=cfg_uuid)
    config = _make_config(cfg_uuid, params={Task.InternalKeys.PERSISTENT: True})
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
    tasks_api_mock.list_task_configs.return_value = MagicMock(configs={cfg_uuid: config})

    started = []

    async def fake_start_task(cfg, resume_instance=None):
        started.append((cfg.uuid, resume_instance.uuid if resume_instance else None))

    monkeypatch.setattr(manager, "_start_task", fake_start_task)

    await manager._init()

    assert started == [(cfg_uuid, "run-chk")]
    # The interrupted instance was first promoted to paused (recoverable)...
    statuses = [c.args[1].status for c in tasks_api_mock.set_task_instance_status.call_args_list]
    assert "paused" in statuses
    # ...and never left in error by the guard, since it auto-restarted cleanly.
    assert "error" not in statuses


async def test_non_persistent_config_does_not_auto_resume(manager, tasks_api_mock, monkeypatch):
    cfg_uuid = "non-persist-cfg"
    inst = _make_instance("run-chk", TaskStatus.RUNNING, resume_data={"images_saved": 3}, config_uuid=cfg_uuid)
    config = _make_config(cfg_uuid, params={})  # no _persistent flag at all
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
    tasks_api_mock.list_task_configs.return_value = MagicMock(configs={cfg_uuid: config})

    started = []

    async def fake_start_task(cfg, resume_instance=None):
        started.append(cfg.uuid)

    monkeypatch.setattr(manager, "_start_task", fake_start_task)

    await manager._init()

    assert started == []
    # Left paused for the user to resume manually, per the recovery pass.
    statuses = [c.args[1].status for c in tasks_api_mock.set_task_instance_status.call_args_list]
    assert statuses == ["paused"]


# ─── Crash-loop guard ───────────────────────────────────────────────────────────


async def test_crash_loop_guard_stops_after_n_restarts_within_window(manager, tasks_api_mock, monkeypatch):
    cfg_uuid = "flaky-cfg"
    config = _make_config(cfg_uuid, params={Task.InternalKeys.PERSISTENT: True})

    def update_config_side_effect(_uuid, payload):
        # Simulate persistence: the next _init() pass reads the updated params back.
        config.params.update(payload.params)
        return MagicMock()

    tasks_api_mock.update_task_config.side_effect = update_config_side_effect
    tasks_api_mock.list_task_configs.return_value = MagicMock(configs={cfg_uuid: config})

    started = []

    async def fake_start_task(cfg, resume_instance=None):
        started.append(cfg.uuid)

    monkeypatch.setattr(manager, "_start_task", fake_start_task)

    total_attempts = Manager._CRASH_LOOP_MAX_RESTARTS + 2
    for i in range(total_attempts):
        inst = _make_instance(f"run-{i}", TaskStatus.RUNNING, resume_data={"x": 1}, config_uuid=cfg_uuid)
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
        await manager._init()

    # Only the first MAX_RESTARTS attempts actually got auto-restarted.
    assert len(started) == Manager._CRASH_LOOP_MAX_RESTARTS

    error_statuses = [
        c.args[1].status for c in tasks_api_mock.set_task_instance_status.call_args_list if c.args[1].status == "error"
    ]
    # The guard trips and marks ERROR for every attempt beyond the threshold.
    assert len(error_statuses) == total_attempts - Manager._CRASH_LOOP_MAX_RESTARTS

    # Loudly logged and visibly ERROR, not silently dead — the final instance
    # of this run must be in the error set (it was the one that tripped the guard).
    final_restart_info = config.params[Task.InternalKeys.RESTART_INFO]
    assert final_restart_info["count"] == total_attempts


async def test_crash_loop_guard_decays_after_window_elapses(manager, tasks_api_mock, monkeypatch):
    cfg_uuid = "decayed-cfg"
    old_window_start = (datetime.now(timezone.utc) - Manager._CRASH_LOOP_WINDOW - timedelta(minutes=1)).isoformat()
    config = _make_config(
        cfg_uuid,
        params={
            Task.InternalKeys.PERSISTENT: True,
            Task.InternalKeys.RESTART_INFO: {
                "count": Manager._CRASH_LOOP_MAX_RESTARTS,  # already at the threshold
                "window_start": old_window_start,
            },
        },
    )
    inst = _make_instance("run-x", TaskStatus.RUNNING, resume_data={"x": 1}, config_uuid=cfg_uuid)
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
    tasks_api_mock.list_task_configs.return_value = MagicMock(configs={cfg_uuid: config})

    started = []

    async def fake_start_task(cfg, resume_instance=None):
        started.append(cfg.uuid)

    monkeypatch.setattr(manager, "_start_task", fake_start_task)

    await manager._init()

    # The old window had fully elapsed, so this restart starts a fresh window
    # (count -> 1) instead of tripping the guard — a task that only crashes
    # occasionally is not permanently barred.
    assert started == [cfg_uuid]
    _, payload = tasks_api_mock.update_task_config.call_args.args
    assert payload.params[Task.InternalKeys.RESTART_INFO]["count"] == 1
