"""A watcher's checkpoint actually reaches the database.

Checkpointing is a two-part handshake with `Manager._checkpoint_running_tasks`:
the task fills `_resume_data` AND raises `_data_ready_flag`, and the manager
writes nothing until `is_data_ready()` says there is something to write.

StreamWatcherTask did only the first half. It answered every checkpoint request
by building resume data and dropping it on the floor, so its instance reached a
restart with `resume_data=None` and startup recovery marked it
*"interrupted without checkpoint — marking error"* rather than resuming it.
Observed live: a watcher that had been running for 29 minutes, with
CHECKPOINT_INTERVAL=60, came back from a restart as an errored instance.

That failed silently in both directions — nothing logged, nothing raised — and
the only visible symptom was a watcher that did not come back. It also quietly
defeats `DEFAULT_PERSISTENT`: auto-resume is gated on `resume_data`, so a task
that never checkpoints cannot be persistent no matter what it declares.

These drive `_maybe_checkpoint` — the real method the run loop calls — rather
than a copy of the loop body, because a copy would have passed against the
broken code.
"""

import asyncio

import pytest

from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask


@pytest.fixture
def watcher():
    """A StreamWatcherTask with only the checkpoint plumbing wired.

    Built with ``object.__new__`` rather than the constructor: the real one
    wants a manager, a config uuid and a DB, and none of that participates in
    the handshake under test.
    """
    w = object.__new__(StreamWatcherTask)
    w._data_req_flag = asyncio.Event()
    w._data_ready_flag = asyncio.Event()
    w._resume_data = {}  # matches Task.__init__
    w._pipelines = []
    w._open_event_uuids = {}
    # _build_watch_resume_data snapshots these counters too; set them here
    # rather than stubbing the builder, so the test exercises the real one.
    w._events_started = 0
    w._events_ended = 0
    return w


class TestCheckpointPrecondition:
    def test_a_persistent_default_needs_a_checkpoint_interval(self):
        """A persistent-by-default task that cannot checkpoint is a promise it
        cannot keep — auto-resume is gated on resume_data."""
        assert StreamWatcherTask.DEFAULT_PERSISTENT is True
        assert StreamWatcherTask.CHECKPOINT_INTERVAL > 0


class TestMaybeCheckpoint:
    def test_answering_a_request_marks_the_data_ready(self, watcher):
        """The assertion that fails against the old code: resume data was built,
        but is_data_ready() stayed False, so the manager wrote nothing."""
        watcher.request_data_update()

        answered = watcher._maybe_checkpoint()

        assert answered is True
        assert watcher.is_data_ready() is True, "manager would write nothing for this watcher"
        assert watcher.get_resume_data() != {}

    def test_the_request_is_acknowledged(self, watcher):
        # Left set, the task would rebuild resume data on every single loop
        # iteration instead of once per checkpoint interval.
        watcher.request_data_update()

        watcher._maybe_checkpoint()

        assert watcher._data_req_flag.is_set() is False

    def test_no_request_means_no_work(self, watcher):
        """The loop calls this every iteration — many times a second — so the
        no-request path must do nothing at all."""
        answered = watcher._maybe_checkpoint()

        assert answered is False
        assert watcher.is_data_ready() is False
        assert watcher.get_resume_data() == {}

    def test_open_events_survive_the_checkpoint(self, watcher):
        """What the checkpoint is *for*: an event still open at restart should
        resume against the same event row rather than opening a second one."""
        watcher._open_event_uuids = {"rule-1": "event-1"}
        watcher.request_data_update()

        watcher._maybe_checkpoint()

        assert watcher.get_resume_data()["open_event_uuids"] == {"rule-1": "event-1"}

    def test_repeated_requests_each_get_answered(self, watcher):
        """A watcher runs for weeks; the handshake has to keep working, not just
        fire once."""
        for i in range(3):
            watcher.clear_data_ready()
            watcher._open_event_uuids = {"rule": f"event-{i}"}
            watcher.request_data_update()

            assert watcher._maybe_checkpoint() is True
            assert watcher.is_data_ready() is True
            assert watcher.get_resume_data()["open_event_uuids"] == {"rule": f"event-{i}"}


class TestSetResumeDataAlone:
    def test_filling_the_field_does_not_mark_it_ready(self, watcher):
        """Pins the sharp edge that caused this: set_resume_data fills the field
        and nothing else, so any task using it without raising the flag
        checkpoints into a void."""
        watcher.set_resume_data({"open_event_uuids": {}})

        assert watcher.is_data_ready() is False
