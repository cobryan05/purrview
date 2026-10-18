"""Tests for the task SSE stream's polling cadence.

/api/tasks/stream is a poll loop wearing a push-shaped API: every connected
browser costs a full image_tagging -> tasks -> db round trip per tick, forever,
whether or not anything is happening. The loop ticks at 2 Hz only while a task is
actually live *and* its payload is still changing, and backs off otherwise.

The second condition is what makes the idle cadence reachable at all on an NVR:
a StreamWatcher is "running" for weeks, so liveness alone pinned every connected
browser at 2 Hz for the life of the process.
"""

from apps.image_tagging_server.web import (
    TASKS_STREAM_ACTIVE_POLL_S,
    TASKS_STREAM_IDLE_POLL_S,
    TASKS_STREAM_SETTLE_S,
    _tasks_poll_interval,
)


def _payload(*statuses: str) -> dict:
    return {
        "status": "success",
        "tasks": {f"uuid-{n}": {"instance": {"status": s}} for n, s in enumerate(statuses)},
    }


class TestTasksPollInterval:
    def test_running_task_polls_fast(self):
        assert _tasks_poll_interval(_payload("running"), 0.0) == TASKS_STREAM_ACTIVE_POLL_S

    def test_pending_and_new_count_as_live(self):
        assert _tasks_poll_interval(_payload("pending"), 0.0) == TASKS_STREAM_ACTIVE_POLL_S
        assert _tasks_poll_interval(_payload("new"), 0.0) == TASKS_STREAM_ACTIVE_POLL_S

    def test_finished_tasks_back_off(self):
        assert _tasks_poll_interval(_payload("completed", "error"), 0.0) == TASKS_STREAM_IDLE_POLL_S

    def test_paused_is_not_live(self):
        """A paused task holds still until the user acts, so it needs no animation."""
        assert _tasks_poll_interval(_payload("paused"), 0.0) == TASKS_STREAM_IDLE_POLL_S

    def test_one_running_task_among_many_finished_polls_fast(self):
        payload = _payload("completed", "error", "paused", "running")
        assert _tasks_poll_interval(payload, 0.0) == TASKS_STREAM_ACTIVE_POLL_S

    def test_empty_and_malformed_payloads_back_off(self):
        """The loop feeds {} in on a failed fetch; that must not pin it at 2 Hz."""
        for payload in ({}, {"tasks": {}}, {"tasks": None}, {"tasks": {"u": None}}, {"tasks": {"u": {}}}):
            assert _tasks_poll_interval(payload, 0.0) == TASKS_STREAM_IDLE_POLL_S

    def test_idle_is_slower_than_active(self):
        assert TASKS_STREAM_IDLE_POLL_S > TASKS_STREAM_ACTIVE_POLL_S


class TestALiveButUnchangingTaskBacksOff:
    """Liveness alone must not buy the fast cadence forever.

    A watcher task reports `running` for as long as the deployment is up. Keying the
    2 Hz cadence on that status pinned it on permanently, so the idle path was dead
    code on exactly the deployment that runs a task continuously — and the cost was
    paid once per connected browser, per tick, for a payload that had not changed.
    """

    def test_a_running_task_whose_payload_has_settled_backs_off(self):
        payload = _payload("running")
        assert _tasks_poll_interval(payload, TASKS_STREAM_SETTLE_S + 1) == TASKS_STREAM_IDLE_POLL_S

    def test_it_speeds_back_up_as_soon_as_the_payload_moves(self):
        """Backing off must not be a latch — the next change has to restore 2 Hz."""
        payload = _payload("running")
        assert _tasks_poll_interval(payload, TASKS_STREAM_SETTLE_S + 1) == TASKS_STREAM_IDLE_POLL_S
        assert _tasks_poll_interval(payload, 0.0) == TASKS_STREAM_ACTIVE_POLL_S

    def test_a_payload_still_moving_keeps_the_fast_cadence(self):
        payload = _payload("running")
        assert _tasks_poll_interval(payload, TASKS_STREAM_SETTLE_S / 2) == TASKS_STREAM_ACTIVE_POLL_S

    def test_a_settled_idle_list_is_still_idle(self):
        """Both conditions point the same way here; neither must flip the result."""
        payload = _payload("completed")
        assert _tasks_poll_interval(payload, TASKS_STREAM_SETTLE_S + 1) == TASKS_STREAM_IDLE_POLL_S
