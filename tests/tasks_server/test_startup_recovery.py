"""Tests for Manager._init() startup recovery logic.

On startup the manager inspects persisted task instances and decides what to
do with any that weren't cleanly shut down:

  "new"     → task was created but the server crashed before it could start;
               delete it (nothing to recover).
  "running" + resume_data  → server was killed while task was executing and
               had already written a checkpoint; promote to "paused" so the
               user can resume.
  "running" + no resume_data → server was killed before any checkpoint;
               mark as "error" so it's clearly visible but not misleadingly
               resumable.
  "paused" / "completed" / "error" → terminal or intentional states; leave
               them alone.
"""

import pytest
from unittest.mock import MagicMock, call

import db_client.models as db_models

from apps.helpers.consts import TaskStatus


def _make_instance(uuid: str, status: str, resume_data=None) -> db_models.TaskInstanceRead:
    return db_models.TaskInstanceRead(
        uuid=uuid,
        config_uuid="cfg-uuid",
        status=status,
        resume_data_json=resume_data,
        result_json=None,
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
    inst = _make_instance(
        "run-chk-uuid", TaskStatus.RUNNING, resume_data={"images_saved": 10}
    )
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


async def test_init_handles_mixed_running_instances(manager, tasks_api_mock):
    """Two interrupted running tasks: one with checkpoint, one without."""
    with_chk = _make_instance("run-chk", TaskStatus.RUNNING, resume_data={"x": 1})
    without_chk = _make_instance("run-no", TaskStatus.RUNNING, resume_data=None)
    tasks_api_mock.list_task_instances.return_value = MagicMock(
        instances=[with_chk, without_chk]
    )

    await manager._init()

    assert tasks_api_mock.set_task_instance_status.call_count == 2
    calls_by_uuid = {
        c.args[0]: c.args[1].status
        for c in tasks_api_mock.set_task_instance_status.call_args_list
    }
    assert calls_by_uuid["run-chk"] == "paused"
    assert calls_by_uuid["run-no"] == "error"


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
