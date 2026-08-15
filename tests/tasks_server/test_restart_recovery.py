"""Recovery when tasks_server restarts alongside (or without) its neighbours.

Three separate holes this covers, all of which showed up as "the training run
is stuck in the UI but the service says it is running 0 tasks":

  1. ``_init()``'s one-shot recovery pass is skipped when db_server isn't
     reachable yet — exactly what happens when every service restarts at once.
     Nothing retried it, so orphaned "running" rows were never reconciled.
  2. ``_check_stale_tasks`` marked every orphan ERROR regardless of whether it
     had a checkpoint, throwing away a resumable job (and orphaning whatever
     remote work it was driving).
  3. ``get_status`` counted only in-memory tasks, so an orphaned row was
     invisible in the status panel while the Runs list still showed it —
     two views of the same task that could not both be right.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import db_client.models as db_models

from apps.helpers.consts import TaskStatus
from apps.tasks_server.tasks.Task import Task


def _make_instance(uuid: str, status: str, resume_data=None, heartbeat_at=None) -> db_models.TaskInstanceRead:
    return db_models.TaskInstanceRead(
        uuid=uuid,
        config_uuid="cfg-uuid",
        status=status,
        resume_data=resume_data,
        result=None,
        error_message=None,
        heartbeat_at=heartbeat_at,
    )


def _stale_hb() -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=Task.HEARTBEAT_INTERVAL * 5)


def _config(persistent: bool) -> db_models.TaskConfigRead:
    return db_models.TaskConfigRead(
        uuid="cfg-uuid",
        name="train-config",
        typename="TestTask",
        params={"duration": 60.0, Task.InternalKeys.PERSISTENT: persistent},
        marked_for_delete=False,
        description=None,
    )


# ─── Recovery is retried until it actually runs ───────────────────────────────


class TestRecoveryRetry:
    async def test_worker_retries_recovery_after_db_server_is_reachable(self, manager, tasks_api_mock):
        """A boot race with db_server must delay recovery, not cancel it.

        First pass raises (db_server still starting); the loop has to try again
        rather than leave every interrupted instance stuck "running" forever.
        """
        inst = _make_instance("orphan", TaskStatus.RUNNING, resume_data={"job_id": "j1"})
        calls = {"n": 0}

        def _list_instances(*_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("db_server not reachable yet")
            return MagicMock(instances=[inst])

        tasks_api_mock.list_task_instances.side_effect = _list_instances
        tasks_api_mock.list_task_configs.return_value = MagicMock(configs={})

        manager._check_stale_tasks = AsyncMock()
        manager._checkpoint_running_tasks = AsyncMock()
        manager._prune_finished_task_instances = AsyncMock()

        sleep_calls = 0

        async def _sleep_once_then_stop(_seconds):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls > 1:
                raise asyncio.CancelledError()

        with patch("apps.tasks_server.manager.asyncio.sleep", new=_sleep_once_then_stop):
            await manager._worker_task()

        tasks_api_mock.set_task_instance_status.assert_called_once()
        uuid_arg, payload_arg = tasks_api_mock.set_task_instance_status.call_args.args
        assert uuid_arg == "orphan"
        assert payload_arg.status == TaskStatus.PAUSED

    async def test_recovery_is_not_repeated_once_it_has_succeeded(self, manager, tasks_api_mock):
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[])

        await manager._run_startup_recovery()
        first_count = tasks_api_mock.list_task_instances.call_count
        await manager._run_startup_recovery()

        assert tasks_api_mock.list_task_instances.call_count == first_count

    async def test_stale_check_is_gated_until_recovery_has_run(self, manager, tasks_api_mock):
        """The stale net must not fire before recovery has had its chance.

        Recovery is the pass that knows how to keep a checkpointed task
        resumable; letting the (blunter) stale check reach an orphan first
        would decide its fate on a coin toss between two 60s timers.
        """
        inst = _make_instance("orphan", TaskStatus.RUNNING, resume_data={"job_id": "j1"}, heartbeat_at=_stale_hb())
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])

        assert manager._recovery_done is False
        await manager._check_stale_tasks()

        tasks_api_mock.set_task_instance_status.assert_not_called()


# ─── The stale net respects checkpoints ───────────────────────────────────────


class TestStaleCheckRespectsCheckpoints:
    async def test_stale_orphan_with_checkpoint_becomes_paused(self, manager, tasks_api_mock):
        inst = _make_instance("orphan", TaskStatus.RUNNING, resume_data={"job_id": "j1"}, heartbeat_at=_stale_hb())
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
        tasks_api_mock.list_task_configs.return_value = MagicMock(configs={"cfg-uuid": _config(persistent=False)})
        manager._recovery_done = True

        await manager._check_stale_tasks()

        tasks_api_mock.set_task_instance_status.assert_called_once()
        uuid_arg, payload_arg = tasks_api_mock.set_task_instance_status.call_args.args
        assert uuid_arg == "orphan"
        assert payload_arg.status == TaskStatus.PAUSED
        assert payload_arg.error_message is None

    async def test_stale_orphan_without_checkpoint_still_errors(self, manager, tasks_api_mock):
        inst = _make_instance("orphan", TaskStatus.RUNNING, resume_data=None, heartbeat_at=_stale_hb())
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
        tasks_api_mock.list_task_configs.return_value = MagicMock(configs={"cfg-uuid": _config(persistent=False)})
        manager._recovery_done = True

        await manager._check_stale_tasks()

        _, payload_arg = tasks_api_mock.set_task_instance_status.call_args.args
        assert payload_arg.status == TaskStatus.ERROR
        assert payload_arg.error_message

    async def test_stale_orphan_of_a_persistent_config_is_auto_restarted(self, manager, tasks_api_mock):
        inst = _make_instance("orphan", TaskStatus.RUNNING, resume_data={"job_id": "j1"}, heartbeat_at=_stale_hb())
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
        tasks_api_mock.list_task_configs.return_value = MagicMock(configs={"cfg-uuid": _config(persistent=True)})
        manager._recovery_done = True
        manager._auto_restart_with_guard = AsyncMock()

        await manager._check_stale_tasks()

        manager._auto_restart_with_guard.assert_awaited_once()
        restarted_inst = manager._auto_restart_with_guard.await_args.args[1]
        assert restarted_inst.uuid == "orphan"

    async def test_stale_orphan_of_a_non_persistent_config_is_left_for_the_user(self, manager, tasks_api_mock):
        """Auto-resume stays opt-in — a paused task waits for a click."""
        inst = _make_instance("orphan", TaskStatus.RUNNING, resume_data={"job_id": "j1"}, heartbeat_at=_stale_hb())
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
        tasks_api_mock.list_task_configs.return_value = MagicMock(configs={"cfg-uuid": _config(persistent=False)})
        manager._recovery_done = True
        manager._auto_restart_with_guard = AsyncMock()

        await manager._check_stale_tasks()

        manager._auto_restart_with_guard.assert_not_awaited()

    async def test_stale_check_reads_resume_data(self, manager, tasks_api_mock):
        """The default listing omits resume_data — asking without it would make
        every orphan look uncheckpointed and unresumable."""
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[])
        manager._recovery_done = True

        await manager._check_stale_tasks()

        assert tasks_api_mock.list_task_instances.call_args.kwargs.get("include_resume_data") is True


# ─── The status panel and the task list agree ─────────────────────────────────


class TestStatusCountsUnattached:
    async def test_status_reports_instances_this_process_is_not_driving(self, manager, tasks_api_mock):
        """An orphan the UI still lists must not read as "running: 0" here."""
        inst = _make_instance("orphan", TaskStatus.RUNNING, resume_data={"job_id": "j1"})
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
        manager._recovery_done = True

        await manager._refresh_unattached_count()
        status = manager.get_status()

        assert status["tasks"]["unattached"] == 1
        assert "unattached" in status["summary"]
        assert any(n["level"] in ("warning", "error") for n in status["notices"])

    async def test_no_unattached_instances_keeps_the_summary_clean(self, manager, tasks_api_mock):
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[])
        manager._recovery_done = True

        await manager._refresh_unattached_count()
        status = manager.get_status()

        assert status["tasks"]["unattached"] == 0
        assert "unattached" not in status["summary"]
