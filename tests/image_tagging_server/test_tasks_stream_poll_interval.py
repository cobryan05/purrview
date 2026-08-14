"""Tests for the task SSE stream's polling cadence.

/api/tasks/stream is a poll loop wearing a push-shaped API: every connected
browser costs a full image_tagging -> tasks -> db round trip per tick, forever,
whether or not anything is happening. The loop now ticks at 2 Hz only while a
task is actually live and backs off when the list is at rest.
"""

from apps.image_tagging_server.web import (
    TASKS_STREAM_ACTIVE_POLL_S,
    TASKS_STREAM_IDLE_POLL_S,
    _tasks_poll_interval,
)


def _payload(*statuses: str) -> dict:
    return {
        "status": "success",
        "tasks": {f"uuid-{n}": {"instance": {"status": s}} for n, s in enumerate(statuses)},
    }


class TestTasksPollInterval:
    def test_running_task_polls_fast(self):
        assert _tasks_poll_interval(_payload("running")) == TASKS_STREAM_ACTIVE_POLL_S

    def test_pending_and_new_count_as_live(self):
        assert _tasks_poll_interval(_payload("pending")) == TASKS_STREAM_ACTIVE_POLL_S
        assert _tasks_poll_interval(_payload("new")) == TASKS_STREAM_ACTIVE_POLL_S

    def test_finished_tasks_back_off(self):
        assert _tasks_poll_interval(_payload("completed", "error")) == TASKS_STREAM_IDLE_POLL_S

    def test_paused_is_not_live(self):
        """A paused task holds still until the user acts, so it needs no animation."""
        assert _tasks_poll_interval(_payload("paused")) == TASKS_STREAM_IDLE_POLL_S

    def test_one_running_task_among_many_finished_polls_fast(self):
        payload = _payload("completed", "error", "paused", "running")
        assert _tasks_poll_interval(payload) == TASKS_STREAM_ACTIVE_POLL_S

    def test_empty_and_malformed_payloads_back_off(self):
        """The loop feeds {} in on a failed fetch; that must not pin it at 2 Hz."""
        for payload in ({}, {"tasks": {}}, {"tasks": None}, {"tasks": {"u": None}}, {"tasks": {"u": {}}}):
            assert _tasks_poll_interval(payload) == TASKS_STREAM_IDLE_POLL_S

    def test_idle_is_slower_than_active(self):
        assert TASKS_STREAM_IDLE_POLL_S > TASKS_STREAM_ACTIVE_POLL_S
