"""When asking a task to stop doesn't work, interrupt it.

`cancel_task()` only sets a flag a task's own loop has to poll. A task parked
somewhere that never looks at it again does not stop, and nothing followed up — the
user's Stop was the end of the story either way. `force_stop()` is the backup: it
cancels the asyncio task itself.

It is explicitly **not** a guaranteed kill, and these tests pin that honestly. It
raises CancelledError at the task's current await, so it recovers a task parked at an
ordinary await and does nothing to one blocked inside `asyncio.to_thread`, whose thread
cannot be interrupted. That case is what bounding thread-bound calls
(apps/helpers/apiCall.py) and reporting wedged tasks are for.
"""

import asyncio
import contextlib
import threading

import pytest

from apps.helpers.consts import TaskStatus
from tests.tasks_server.conftest import CONFIG_UUID, TASK_UUID


async def _start_long_task(manager, tasks_api_mock, duration=60):
    tasks_api_mock.list_task_configs.return_value.configs[CONFIG_UUID].params = {"duration": duration}
    await manager.start_new_task(CONFIG_UUID)
    await asyncio.sleep(0)
    return manager._running_tasks[TASK_UUID]


class TestForceStop:
    async def test_it_ends_a_task_that_ignores_the_cancel_flag(self, manager, tasks_api_mock):
        info = await _start_long_task(manager, tasks_api_mock)

        assert info.task.force_stop() is True
        await asyncio.wait_for(info.task.wait_for_task_done(5.0), timeout=5.0)

        assert info.task.is_task_done()

    async def test_a_force_stopped_task_ends_in_a_terminal_status(self, manager, tasks_api_mock):
        """The trap: CancelledError is a BaseException, so `except Exception` never
        sees it. Unhandled, _status stays RUNNING and _on_task_complete persists
        "running" as the *final* status — a finished task the DB calls running
        forever."""
        info = await _start_long_task(manager, tasks_api_mock)

        info.task.force_stop()
        await asyncio.wait_for(info.task.wait_for_task_done(5.0), timeout=5.0)

        assert info.task.get_status() == TaskStatus.ERROR
        assert info.task.get_status() != TaskStatus.RUNNING
        assert "did not respond" in (info.task.get_error_message() or "")

    async def test_deinit_still_runs_so_resources_are_released(self, manager, tasks_api_mock):
        """An interrupted task is exactly when cleanup matters most — it is holding
        subprocesses, leases and file handles at the moment it is cut off."""
        info = await _start_long_task(manager, tasks_api_mock)
        ran = []
        original = info.task._deinit

        async def _tracking_deinit():
            ran.append(True)
            await original()

        info.task._deinit = _tracking_deinit

        info.task.force_stop()
        await asyncio.wait_for(info.task.wait_for_task_done(5.0), timeout=5.0)

        assert ran == [True]

    async def test_a_raising_deinit_does_not_strand_the_task(self, manager, tasks_api_mock):
        """_task_done is what wait_for_task_done — and through it _on_task_complete,
        which removes the task and writes its status — keys off. A _deinit that raised
        used to skip it and strand all of that, and a task torn down mid-operation is
        when _deinit is most likely to raise."""
        info = await _start_long_task(manager, tasks_api_mock)

        async def _exploding_deinit():
            raise RuntimeError("cleanup failed")

        info.task._deinit = _exploding_deinit

        info.task.force_stop()
        await asyncio.wait_for(info.task.wait_for_task_done(5.0), timeout=5.0)

        assert info.task.is_task_done(), "the task must still be marked done"

    async def test_force_stopping_a_finished_task_is_a_no_op(self, manager, tasks_api_mock):
        info = await _start_long_task(manager, tasks_api_mock, duration=0)
        await asyncio.wait_for(info.task.wait_for_task_done(5.0), timeout=5.0)

        assert info.task.force_stop() is False

    async def test_an_unrelated_cancellation_still_propagates(self, manager, tasks_api_mock):
        """Loop shutdown cancels everything. Swallowing that would hang the shutdown
        on a task refusing to end, so only *our* cancellation is absorbed."""
        info = await _start_long_task(manager, tasks_api_mock)

        info.task._task.cancel()  # not via force_stop — nobody set _force_stopping

        with pytest.raises(asyncio.CancelledError):
            await info.task._task

        assert not info.task.is_force_stopping()


class TestCancelIsStillCooperativeFirst:
    async def test_cancel_task_does_not_interrupt(self, manager, tasks_api_mock):
        """Asking stays asking. A task that honours the flag gets to shut down
        properly, which is the difference between a clean stop and a severed one."""
        info = await _start_long_task(manager, tasks_api_mock)

        info.task.cancel_task()

        assert not info.task.is_force_stopping()
        assert not info.task._task.cancelled()

    async def test_the_request_time_is_recorded(self, manager, tasks_api_mock):
        info = await _start_long_task(manager, tasks_api_mock)
        assert info.task.seconds_since_cancel_requested() is None

        info.task.cancel_task()

        assert info.task.seconds_since_cancel_requested() >= 0
        info.task.force_stop()

    async def test_repeated_stops_do_not_reset_the_clock(self, manager, tasks_api_mock):
        """Otherwise a user clicking Stop every minute would hold off escalation
        forever — the exact thing they are clicking *because* it is not working."""
        info = await _start_long_task(manager, tasks_api_mock)

        info.task.cancel_task()
        first = info.task._cancel_requested_at
        info.task.cancel_task()

        assert info.task._cancel_requested_at == first
        info.task.force_stop()


class TestEscalation:
    async def test_a_task_stopping_normally_is_left_alone(self, manager, tasks_api_mock):
        """Interrupting costs whatever the task was midway through, so a slow but
        honest shutdown must not be cut short."""
        info = await _start_long_task(manager, tasks_api_mock)
        info.task.cancel_task()

        manager._escalate_stuck_cancels()

        assert not info.task.is_force_stopping()
        info.task.force_stop()

    async def test_a_task_that_ignores_the_request_is_interrupted(self, manager, tasks_api_mock):
        info = await _start_long_task(manager, tasks_api_mock)
        info.task.cancel_task()
        # Backdate the request past the escalation threshold.
        info.task._cancel_requested_at -= manager._FORCE_STOP_AFTER_S + 1

        manager._escalate_stuck_cancels()

        assert info.task.is_force_stopping()
        await asyncio.wait_for(info.task.wait_for_task_done(5.0), timeout=5.0)

    async def test_a_task_never_asked_to_stop_is_never_interrupted(self, manager, tasks_api_mock):
        info = await _start_long_task(manager, tasks_api_mock)

        manager._escalate_stuck_cancels()

        assert not info.task.is_force_stopping()
        info.task.cancel_task()
        info.task.force_stop()

    async def test_escalation_happens_once(self, manager, tasks_api_mock):
        info = await _start_long_task(manager, tasks_api_mock)
        info.task.cancel_task()
        info.task._cancel_requested_at -= manager._FORCE_STOP_AFTER_S + 1
        calls = []
        original = info.task.force_stop
        info.task.force_stop = lambda: calls.append(True) or original()

        for _ in range(4):
            manager._escalate_stuck_cancels()

        assert len(calls) == 1, "an already-interrupted task must not be re-interrupted"
        await asyncio.wait_for(info.task.wait_for_task_done(5.0), timeout=5.0)

    async def test_it_needs_no_database(self, manager, tasks_api_mock):
        """In-memory, so it still runs when the DB is what has jammed."""
        info = await _start_long_task(manager, tasks_api_mock)
        info.task.cancel_task()
        info.task._cancel_requested_at -= manager._FORCE_STOP_AFTER_S + 1
        tasks_api_mock.list_task_instances.side_effect = AssertionError("must not hit the DB")

        manager._escalate_stuck_cancels()

        assert info.task.is_force_stopping()
        await asyncio.wait_for(info.task.wait_for_task_done(5.0), timeout=5.0)


class TestItIsNotAGuaranteedKill:
    async def test_cancelling_a_task_blocked_in_a_thread_does_nothing(self):
        """The honest limit, pinned so nobody writes this up as a kill switch.

        `asyncio.to_thread` cannot be cancelled — the thread runs to completion no
        matter what, and `cancel()` cheerfully reports success while the work carries
        on. This is the exact shape of the hang that started all of this (an ffmpeg
        `process.wait()` in a thread), and the reason bounding thread-bound calls
        (apps/helpers/apiCall.py) matters far more than force_stop does. A task that
        survives an interrupt is reported by _check_wedged_tasks rather than assumed
        dead.

        Deliberately tests asyncio's behaviour rather than Task's: the point is the
        platform limit that force_stop is built on, and a test that went through Task
        would be asserting the same thing less directly.
        """
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def _blocking_work():
            started.set()
            # Bounded so a regression here fails the test instead of hanging the suite.
            release.wait(timeout=10)
            finished.set()

        task = asyncio.ensure_future(asyncio.to_thread(_blocking_work))
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set(), "the thread never got going"

        assert task.cancel() is True
        await asyncio.sleep(0.05)

        # This is the part worth pinning, and it is worse than "cancelling does not
        # work": asyncio reports the task cancelled *immediately* while the thread
        # carries on holding its executor slot. The caller is told the work stopped
        # and it has not. Enough of these and the shared pool — min(32, cpu + 4)
        # threads, used by every to_thread in the process — has nothing left to give,
        # which is how heartbeats stop while everything still looks healthy.
        assert task.cancelled(), "asyncio calls it cancelled..."
        assert not finished.is_set(), "...while the thread is demonstrably still running"

        release.set()
        assert finished.wait(5), "the thread runs to completion regardless"
        with contextlib.suppress(asyncio.CancelledError):
            await task
