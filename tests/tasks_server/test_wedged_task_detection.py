"""A task that is stuck but still attached has to be visible.

The blind spot: `_check_stale_tasks` skips anything in `_running_tasks`, and that skip
is load-bearing — its finding goes to `_reconcile_interrupted`, which marks the instance
paused-or-error and auto-restarts persistent configs. Doing that to a merely-stuck task
starts a second driver beside a first that still holds its subprocesses and relay
leases, which is precisely the state a wedged watcher sat in for 45 minutes.

So the same signal means opposite things either side of that line: an *unregistered*
stale row means no driver exists, a *registered* one means a driver exists and is
stuck. These tests pin the second case being detected and reported, and — just as
importantly — pin that detecting it changes no task state.
"""

import time
from types import SimpleNamespace

import pytest

from apps.helpers.consts import TaskStatus
from apps.tasks_server.tasks.Task import Task


def _task_double(*, silent_for, name="Watch Test", uuid="task-1", paused=False):
    """A TaskInfo-shaped stand-in whose heartbeat age is dictated by the test."""
    task = SimpleNamespace(
        seconds_since_heartbeat=lambda now=None: silent_for,
        is_paused_in_place=lambda: paused,
        get_progress=lambda: None,
        get_status_message=lambda: "Watching",
        get_notices=lambda: [],
    )
    return SimpleNamespace(
        task=task,
        config=SimpleNamespace(name=name, typename="StreamWatcherTask", uuid="cfg-1"),
        instance=SimpleNamespace(uuid=uuid, status=TaskStatus.RUNNING),
    )


def _threshold(manager):
    return Task.HEARTBEAT_INTERVAL * manager._WEDGED_HEARTBEAT_INTERVALS


def _concrete_task():
    """An un-run instance of the suite's registered TestTask.

    `Task` is abstract, so seconds_since_heartbeat() is exercised through the same
    concrete subclass the rest of these tests drive, rather than a bespoke stub that
    could drift from it.
    """
    from apps.tasks_server.tasks.Registry import task_registry

    return task_registry["TestTask"].__new__(task_registry["TestTask"])


class TestDetection:
    def test_a_progressing_task_is_not_flagged(self, manager):
        manager._running_tasks = {"task-1": _task_double(silent_for=1.0)}

        manager._check_wedged_tasks()

        assert manager._wedged_tasks == {}

    def test_a_silent_task_is_flagged(self, manager):
        silent = _threshold(manager) + 60
        manager._running_tasks = {"task-1": _task_double(silent_for=silent)}

        manager._check_wedged_tasks()

        assert "task-1" in manager._wedged_tasks

    def test_a_task_just_under_the_threshold_is_not_flagged(self):
        """A slow operation between beats must not read as a fault."""
        from apps.tasks_server.manager import Manager

        assert Manager._WEDGED_HEARTBEAT_INTERVALS >= 2, "one missed beat cannot be enough"

    def test_a_task_that_has_not_started_is_not_flagged(self, manager):
        """seconds_since_heartbeat() is None before _execute begins."""
        manager._running_tasks = {"task-1": _task_double(silent_for=None)}

        manager._check_wedged_tasks()

        assert manager._wedged_tasks == {}

    def test_recovery_clears_the_flag(self, manager):
        info = _task_double(silent_for=_threshold(manager) + 60)
        manager._running_tasks = {"task-1": info}
        manager._check_wedged_tasks()
        assert manager._wedged_tasks

        info.task.seconds_since_heartbeat = lambda now=None: 0.5
        manager._check_wedged_tasks()

        assert manager._wedged_tasks == {}

    def test_a_finished_task_leaving_the_registry_clears_the_flag(self, manager):
        manager._running_tasks = {"task-1": _task_double(silent_for=_threshold(manager) + 60)}
        manager._check_wedged_tasks()
        assert manager._wedged_tasks

        manager._running_tasks = {}
        manager._check_wedged_tasks()

        assert manager._wedged_tasks == {}


class TestItOnlyReports:
    """Detection must not become remediation by accident — see the module docstring."""

    def test_the_task_is_left_registered_and_untouched(self, manager, tasks_api_mock):
        info = _task_double(silent_for=_threshold(manager) + 600)
        manager._running_tasks = {"task-1": info}

        manager._check_wedged_tasks()

        assert "task-1" in manager._running_tasks, "a wedged task must not be de-registered"
        assert info.instance.status == TaskStatus.RUNNING, "its status must not be rewritten"
        tasks_api_mock.set_task_instance_status.assert_not_called()

    def test_no_restart_is_triggered(self, manager, tasks_api_mock):
        """Auto-restart beside a still-running driver is the state being avoided."""
        manager._running_tasks = {"task-1": _task_double(silent_for=_threshold(manager) + 600)}

        manager._check_wedged_tasks()

        tasks_api_mock.update_task_config.assert_not_called()

    def test_it_needs_no_database(self, manager, tasks_api_mock):
        """In-memory by design: it must still report when the DB is what has jammed."""
        manager._running_tasks = {"task-1": _task_double(silent_for=_threshold(manager) + 600)}
        tasks_api_mock.list_task_instances.side_effect = AssertionError("must not hit the DB")

        manager._check_wedged_tasks()

        assert manager._wedged_tasks


class TestReporting:
    def test_the_status_panel_counts_and_warns(self, manager):
        manager._running_tasks = {"task-1": _task_double(silent_for=_threshold(manager) + 600)}
        manager._check_wedged_tasks()

        status = manager.get_status()

        assert status["tasks"]["wedged"] == 1
        assert "not responding" in status["summary"]
        notice = next(n for n in status["notices"] if n["key"] == "tasks_server:wedged")
        assert notice["level"] == "warning"
        # The user's next question is "so what do I do" — the notice has to answer it.
        assert "restarting tasks_server" in notice["message"]

    def test_the_row_says_which_task_is_stuck(self, manager):
        """A stuck task's own status message is whatever it last set, which reads as
        though it were still doing that."""
        manager._running_tasks = {"task-1": _task_double(silent_for=_threshold(manager) + 600)}
        manager._check_wedged_tasks()

        rows = manager.get_status()["details"][0]["rows"]

        assert any("NOT RESPONDING" in r["value"] for r in rows)
        assert any("Watch Test" in r["label"] for r in rows)

    def test_a_healthy_server_says_nothing_about_it(self, manager):
        manager._running_tasks = {"task-1": _task_double(silent_for=1.0)}
        manager._check_wedged_tasks()

        status = manager.get_status()

        assert status["tasks"]["wedged"] == 0
        assert "not responding" not in status["summary"]
        assert not [n for n in status["notices"] if n["key"] == "tasks_server:wedged"]

    def test_the_transition_is_logged_once_not_every_pass(self, manager, caplog):
        """This runs every worker pass; a task wedged for a day would otherwise write
        the same line hundreds of times into the log that explains it."""
        manager._running_tasks = {"task-1": _task_double(silent_for=_threshold(manager) + 600)}

        with caplog.at_level("WARNING"):
            for _ in range(5):
                manager._check_wedged_tasks()

        warnings = [r for r in caplog.records if "has not reported progress" in r.getMessage()]
        assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"


class TestSecondsSinceHeartbeat:
    """The signal itself. Read raw, `_last_heartbeat_time` is 0.0 until the first beat,
    so a task one second old would look silent since the epoch."""

    def test_none_before_the_task_starts(self):
        task = _concrete_task()
        task._started_monotonic = None
        task._last_heartbeat_time = 0.0

        assert task.seconds_since_heartbeat() is None

    def test_measured_from_start_before_the_first_beat(self):
        task = _concrete_task()
        now = time.monotonic()
        task._started_monotonic = now - 5.0
        task._last_heartbeat_time = 0.0

        assert task.seconds_since_heartbeat(now) == pytest.approx(5.0)

    def test_measured_from_the_last_beat_once_there_is_one(self):
        task = _concrete_task()
        now = time.monotonic()
        task._started_monotonic = now - 500.0
        task._last_heartbeat_time = now - 3.0

        assert task.seconds_since_heartbeat(now) == pytest.approx(3.0)
