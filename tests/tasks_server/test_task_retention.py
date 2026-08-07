"""Tests for Manager._prune_finished_task_instances().

Finished task instances are history, and history now carries a persisted status
snapshot and log buffer alongside it, so it has to be bounded. Two bounds apply,
whichever removes more:

  age   — anything that ended longer ago than _RETENTION_DAYS
  count — anything beyond the _RETENTION_MAX_COMPLETED most recent

Only COMPLETED/ERROR instances are candidates: running, pending and suspended
tasks are live state, never history, no matter how old.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import db_client.models as db_models
import pytest

from apps.helpers.consts import TaskStatus
from apps.tasks_server.manager import Manager

NOW = datetime.now(timezone.utc)


def _instance(uuid: str, status: str, ended_days_ago: float | None = 0.0) -> db_models.TaskInstanceRead:
    ended = None if ended_days_ago is None else NOW - timedelta(days=ended_days_ago)
    return db_models.TaskInstanceRead(
        uuid=uuid,
        config_uuid="cfg-uuid",
        status=status,
        resume_data=None,
        result=None,
        error_message=None,
        ended_at=ended,
    )


def _deleted_uuids(tasks_api_mock) -> list[str]:
    calls = tasks_api_mock.delete_task_instances.call_args_list
    out: list[str] = []
    for c in calls:
        out.extend(c.args[0].task_uuids)
    return out


class TestAgeBound:
    async def test_instances_older_than_the_window_are_pruned(self, manager, tasks_api_mock):
        old = _instance("old", TaskStatus.COMPLETED, ended_days_ago=20)
        recent = _instance("recent", TaskStatus.COMPLETED, ended_days_ago=2)
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[old, recent])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Manager, "_RETENTION_DAYS", 15)
            mp.setattr(Manager, "_RETENTION_MAX_COMPLETED", 0)
            deleted = await manager._prune_finished_task_instances()

        assert deleted == ["old"]
        assert _deleted_uuids(tasks_api_mock) == ["old"]

    async def test_undated_finished_instances_are_treated_as_oldest(self, manager, tasks_api_mock):
        """Rows written before ended_at was persisted would otherwise be immortal."""
        undated = _instance("undated", TaskStatus.ERROR, ended_days_ago=None)
        recent = _instance("recent", TaskStatus.COMPLETED, ended_days_ago=1)
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=[undated, recent])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Manager, "_RETENTION_DAYS", 15)
            mp.setattr(Manager, "_RETENTION_MAX_COMPLETED", 0)
            deleted = await manager._prune_finished_task_instances()

        assert deleted == ["undated"]


class TestCountBound:
    async def test_only_the_most_recent_n_are_kept(self, manager, tasks_api_mock):
        # All well inside the age window, so only the count bound can bite.
        instances = [_instance(f"i{n}", TaskStatus.COMPLETED, ended_days_ago=n) for n in range(5)]
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=instances)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Manager, "_RETENTION_DAYS", 0)
            mp.setattr(Manager, "_RETENTION_MAX_COMPLETED", 2)
            deleted = await manager._prune_finished_task_instances()

        # i0 (newest) and i1 survive; the three older ones go.
        assert deleted == ["i2", "i3", "i4"]

    async def test_both_bounds_apply_together(self, manager, tasks_api_mock):
        instances = [
            _instance("newest", TaskStatus.COMPLETED, ended_days_ago=0),
            _instance("middle", TaskStatus.COMPLETED, ended_days_ago=1),
            _instance("ancient", TaskStatus.ERROR, ended_days_ago=99),
        ]
        tasks_api_mock.list_task_instances.return_value = MagicMock(instances=instances)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Manager, "_RETENTION_DAYS", 15)  # removes "ancient"
            mp.setattr(Manager, "_RETENTION_MAX_COMPLETED", 2)  # also removes "ancient"
            deleted = await manager._prune_finished_task_instances()

        assert deleted == ["ancient"]


class TestWhatIsNeverPruned:
    @pytest.mark.parametrize("status", [TaskStatus.RUNNING, TaskStatus.PENDING, TaskStatus.PAUSED, TaskStatus.NEW])
    async def test_live_instances_are_never_pruned_however_old(self, manager, tasks_api_mock, status):
        """A suspended task is resumable state, not history — age must not touch it."""
        tasks_api_mock.list_task_instances.return_value = MagicMock(
            instances=[_instance("live", status, ended_days_ago=999)]
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Manager, "_RETENTION_DAYS", 1)
            mp.setattr(Manager, "_RETENTION_MAX_COMPLETED", 1)
            deleted = await manager._prune_finished_task_instances()

        assert deleted == []
        tasks_api_mock.delete_task_instances.assert_not_called()

    async def test_both_bounds_disabled_is_a_no_op(self, manager, tasks_api_mock):
        tasks_api_mock.list_task_instances.return_value = MagicMock(
            instances=[_instance("ancient", TaskStatus.COMPLETED, ended_days_ago=999)]
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Manager, "_RETENTION_DAYS", 0)
            mp.setattr(Manager, "_RETENTION_MAX_COMPLETED", 0)
            deleted = await manager._prune_finished_task_instances()

        assert deleted == []
        tasks_api_mock.list_task_instances.assert_not_called()

    async def test_nothing_to_prune_makes_no_delete_call(self, manager, tasks_api_mock):
        tasks_api_mock.list_task_instances.return_value = MagicMock(
            instances=[_instance("recent", TaskStatus.COMPLETED, ended_days_ago=1)]
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Manager, "_RETENTION_DAYS", 15)
            mp.setattr(Manager, "_RETENTION_MAX_COMPLETED", 200)
            deleted = await manager._prune_finished_task_instances()

        assert deleted == []
        tasks_api_mock.delete_task_instances.assert_not_called()


class TestFailureHandling:
    async def test_a_failing_delete_does_not_raise_into_the_worker_loop(self, manager, tasks_api_mock):
        """The prune runs on the shared 60s worker loop — an exception here would
        take the heartbeat and checkpoint passes down with it."""
        tasks_api_mock.list_task_instances.return_value = MagicMock(
            instances=[_instance("old", TaskStatus.COMPLETED, ended_days_ago=99)]
        )
        tasks_api_mock.delete_task_instances.side_effect = RuntimeError("db down")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Manager, "_RETENTION_DAYS", 15)
            mp.setattr(Manager, "_RETENTION_MAX_COMPLETED", 0)
            deleted = await manager._prune_finished_task_instances()

        assert deleted == []

    async def test_a_failing_list_does_not_raise(self, manager, tasks_api_mock):
        tasks_api_mock.list_task_instances.side_effect = RuntimeError("db down")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Manager, "_RETENTION_DAYS", 15)
            mp.setattr(Manager, "_RETENTION_MAX_COMPLETED", 0)
            deleted = await manager._prune_finished_task_instances()

        assert deleted == []
