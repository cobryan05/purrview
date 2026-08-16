"""Tests for Manager._init() startup recovery logic.

On startup the manager inspects persisted task instances and decides what to
do with any that weren't cleanly shut down:

  "new"     → task was created but the server crashed before it could start;
               delete it (nothing to recover).
  "running" + resume_data  → server was killed while task was executing and
               had already written a checkpoint; promote to "paused".
  "running" + no resume_data → server was killed before any checkpoint;
               mark as "error" so it's clearly visible but not misleadingly
               resumable.
  "paused" / "completed" / "error" → terminal or intentional states; leave
               them alone.

**"paused" is not the end of the story for a PERSISTENT config.** After the
statuses above are applied, `Manager._recover_interrupted_instances` makes a
second pass over the instances that recovered *with* a checkpoint, and any whose
config carries `Task.InternalKeys.PERSISTENT` is auto-restarted via
`_auto_restart_with_guard` — no user action at all. So a persistent task comes
back running on its own after a `tasks_server` restart, and only a
non-persistent one waits, paused, for someone to press resume.

The auto-restart is subject to a crash-loop guard: more than
`_CRASH_LOOP_MAX_RESTARTS` (5) restarts inside `_CRASH_LOOP_WINDOW` (1 hour) and
the instance is marked "error" instead. A restart after the window has elapsed
opens a fresh window, which is how the guard decays for a task that only fails
occasionally.

Read the list above as "what the status pass does", not as "what the user will
see" — reading it as the latter leads directly to telling someone their
long-running collect run needs a manual resume when it does not.

The auto-restart half is tested in `test_checkpoint_and_restart.py`
(`test_persistent_config_auto_resumes_after_restart`,
`test_non_persistent_config_does_not_auto_resume`, and the two crash-loop guard
tests) and in `test_restart_recovery.py`
(`test_stale_orphan_of_a_persistent_config_is_auto_restarted`). This file covers
only the status pass, which is why its docstring described only half the story.
"""

from unittest.mock import MagicMock, call

import db_client.models as db_models
import pytest

from apps.helpers.consts import TaskStatus
from apps.tasks_server.tasks.Task import Task


def _make_instance(uuid: str, status: str, resume_data=None) -> db_models.TaskInstanceRead:
    return db_models.TaskInstanceRead(
        uuid=uuid,
        config_uuid="cfg-uuid",
        status=status,
        resume_data=resume_data,
        result=None,
        error_message=None,
    )


# ─── "new" tasks ──────────────────────────────────────────────────────────────


async def test_init_deletes_never_started_new_tasks(manager, tasks_api_mock):
    inst = _make_instance("new-task-uuid", TaskStatus.NEW)
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])

    await manager._init()

    tasks_api_mock.delete_task_instances.assert_called_once()
    deleted = tasks_api_mock.delete_task_instances.call_args.args[0]
    assert "new-task-uuid" in deleted.task_uuids


async def test_init_deletes_multiple_new_tasks_in_one_call(manager, tasks_api_mock):
    instances = [_make_instance(f"new-{i}", TaskStatus.NEW) for i in range(3)]
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=instances)

    await manager._init()

    tasks_api_mock.delete_task_instances.assert_called_once()
    deleted = tasks_api_mock.delete_task_instances.call_args.args[0]
    assert set(deleted.task_uuids) == {"new-0", "new-1", "new-2"}


# ─── "running" tasks ──────────────────────────────────────────────────────────


async def test_init_promotes_running_with_checkpoint_to_paused(manager, tasks_api_mock):
    inst = _make_instance("run-chk-uuid", TaskStatus.RUNNING, resume_data={"images_saved": 10})
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])

    await manager._init()

    tasks_api_mock.set_task_instance_status.assert_called_once()
    uuid_arg, payload_arg = tasks_api_mock.set_task_instance_status.call_args.args
    assert uuid_arg == "run-chk-uuid"
    assert payload_arg.status == "paused"


async def test_init_marks_running_without_checkpoint_as_error(manager, tasks_api_mock):
    inst = _make_instance("run-no-chk-uuid", TaskStatus.RUNNING, resume_data=None)
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])

    await manager._init()

    tasks_api_mock.set_task_instance_status.assert_called_once()
    uuid_arg, payload_arg = tasks_api_mock.set_task_instance_status.call_args.args
    assert uuid_arg == "run-no-chk-uuid"
    assert payload_arg.status == "error"
    # There is no Task object left to explain this one, so the recovery pass has
    # to supply the reason itself — otherwise the card just says "error".
    assert payload_arg.error_message is not None
    assert "no checkpoint" in payload_arg.error_message


async def test_init_does_not_attach_an_error_message_to_a_recovered_paused_task(manager, tasks_api_mock):
    """A resumable task isn't a failure; it must not carry failure text."""
    inst = _make_instance("run-chk-uuid", TaskStatus.RUNNING, resume_data={"images_saved": 10})
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])

    await manager._init()

    _, payload_arg = tasks_api_mock.set_task_instance_status.call_args.args
    assert payload_arg.status == "paused"
    assert payload_arg.error_message is None


async def test_init_handles_mixed_running_instances(manager, tasks_api_mock):
    """Two interrupted running tasks: one with checkpoint, one without."""
    with_chk = _make_instance("run-chk", TaskStatus.RUNNING, resume_data={"x": 1})
    without_chk = _make_instance("run-no", TaskStatus.RUNNING, resume_data=None)
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[with_chk, without_chk])

    await manager._init()

    assert tasks_api_mock.set_task_instance_status.call_count == 2
    calls_by_uuid = {c.args[0]: c.args[1].status for c in tasks_api_mock.set_task_instance_status.call_args_list}
    assert calls_by_uuid["run-chk"] == "paused"
    assert calls_by_uuid["run-no"] == "error"


# ─── Tasks this process is actually driving ───────────────────────────────────
#
# Recovery is retried on every worker pass until it succeeds, so it no longer
# runs only before any task can exist. A pass that lands while a task is running
# in-process must leave that task alone: it is not an orphan, it has something
# driving it.


async def test_recovery_leaves_a_task_this_process_is_running(manager, tasks_api_mock):
    """db_server was slow to boot, the user started a run, and the retry then
    found that run's own RUNNING row. Reconciling it would pause or error a task
    that never stopped."""
    inst = _make_instance("live-uuid", TaskStatus.RUNNING, resume_data={"images_saved": 3})
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
    manager._running_tasks["live-uuid"] = MagicMock()

    await manager._init()

    tasks_api_mock.set_task_instance_status.assert_not_called()


async def test_recovery_does_not_delete_an_instance_still_being_started(manager, tasks_api_mock):
    """_start_task registers the instance in _running_tasks before the RUNNING
    write lands, so "new" is a live task's transient state, not only a crash
    leftover."""
    inst = _make_instance("starting-uuid", TaskStatus.NEW)
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
    manager._running_tasks["starting-uuid"] = MagicMock()

    await manager._init()

    tasks_api_mock.delete_task_instances.assert_not_called()


async def test_recovery_still_reconciles_the_orphans_beside_a_live_task(manager, tasks_api_mock):
    """Filtering out what we are driving must not cost us the genuine orphans in
    the same listing."""
    live = _make_instance("live-uuid", TaskStatus.RUNNING, resume_data={"x": 1})
    orphan = _make_instance("orphan-uuid", TaskStatus.RUNNING, resume_data={"x": 1})
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[live, orphan])
    manager._running_tasks["live-uuid"] = MagicMock()

    await manager._init()

    tasks_api_mock.set_task_instance_status.assert_called_once()
    uuid_arg, payload_arg = tasks_api_mock.set_task_instance_status.call_args.args
    assert uuid_arg == "orphan-uuid"
    assert payload_arg.status == "paused"


async def test_a_live_persistent_task_is_not_auto_restarted_a_second_time(manager, tasks_api_mock):
    """The auto-restart path is the expensive half of the mistake: a second
    driver for a job that is still running."""
    persistent_config = db_models.TaskConfigRead(
        uuid="cfg-uuid",
        name="watcher",
        typename="TestTask",
        params={Task.InternalKeys.PERSISTENT: True},
        marked_for_delete=False,
        description=None,
    )
    tasks_api_mock.list_task_configs.return_value = MagicMock(configs={"cfg-uuid": persistent_config})
    inst = _make_instance("live-uuid", TaskStatus.RUNNING, resume_data={"x": 1})
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])
    manager._running_tasks["live-uuid"] = MagicMock()

    restarts = []

    async def _record(config, instance):
        restarts.append(instance.uuid)

    manager._auto_restart_with_guard = _record

    await manager._init()

    assert restarts == []


# ─── Terminal / intentional states left untouched ─────────────────────────────


async def test_init_does_not_touch_completed_tasks(manager, tasks_api_mock):
    inst = _make_instance("done-uuid", TaskStatus.COMPLETED)
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])

    await manager._init()

    tasks_api_mock.delete_task_instances.assert_not_called()
    tasks_api_mock.set_task_instance_status.assert_not_called()


async def test_init_does_not_touch_paused_tasks(manager, tasks_api_mock):
    inst = _make_instance("paused-uuid", TaskStatus.PAUSED, resume_data={"y": 2})
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])

    await manager._init()

    tasks_api_mock.delete_task_instances.assert_not_called()
    tasks_api_mock.set_task_instance_status.assert_not_called()


async def test_init_does_not_touch_errored_tasks(manager, tasks_api_mock):
    inst = _make_instance("err-uuid", TaskStatus.ERROR)
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[inst])

    await manager._init()

    tasks_api_mock.delete_task_instances.assert_not_called()
    tasks_api_mock.set_task_instance_status.assert_not_called()


async def test_init_with_empty_instance_list_is_a_no_op(manager, tasks_api_mock):
    tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[])

    await manager._init()

    tasks_api_mock.delete_task_instances.assert_not_called()
    tasks_api_mock.set_task_instance_status.assert_not_called()
