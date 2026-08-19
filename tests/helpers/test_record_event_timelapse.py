"""Tests for `record_event_timelapse` — the action that spans an event's whole life.

Two things make this action unlike every other one, and both are where the bugs are:

* **It is stateful across three triggers.** A frame arriving between the open and the
  close belongs to a session it has to find for itself, because the `frame_analysed`
  signal carries no rule. Every test here drives the real trigger sequence rather than
  calling the private methods, so a change to how `run` dispatches on the signal is
  caught here rather than at 3am on a real camera.

* **It writes to disk for the duration of an event.** The frame buffer is a temp
  directory, and a session that is dropped without being compiled leaks one. The
  clean-up tests are as important as the compile ones.

The compile is exercised against the real ffmpeg in the container. Mocking it would test
that we build a command line, not that the command line produces a video a browser can
play, and the pixel format is exactly the sort of thing that only fails for real.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from apps.helpers.actions.action import ActionContext
from apps.helpers.actions.recordEventTimelapseAction import DEFAULT_TIMELAPSE_ROOT, RecordEventTimelapseAction
from apps.helpers.actions.Registry import action_registry, create_action
from apps.helpers.watch.events import EventPhase, WatchEventSignal
from apps.helpers.watch.scoping import EVENT_ENDED, EVENT_STARTED, FRAME_ANALYSED, action_applies


def _frame(width=320, height=240, value=40):
    return np.full((height, width, 3), value, dtype=np.uint8)


def _context(**overrides) -> ActionContext:
    kwargs = dict(
        profile_uuid="prof-1",
        profile_name="House cams",
        source_uuid="src-1",
        source_name="Window cam",
        wallclock_offset=1_700_000_000.0,
        frame=_frame(),
    )
    kwargs.update(overrides)
    return ActionContext(**kwargs)


def _event_signal(phase, started_at=100.0, at=None, **overrides) -> WatchEventSignal:
    kwargs = dict(
        rule_uuid="rule-1",
        rule_name="cat in window",
        phase=phase,
        at=at if at is not None else started_at,
        started_at=started_at,
        source_uuid="src-1",
        track_ids=[1],
        peak_confidence=0.9,
    )
    kwargs.update(overrides)
    return WatchEventSignal(**kwargs)


def _frame_signal(at, source_uuid="src-1") -> WatchEventSignal:
    """What `_dispatch_frame` builds: no rule, no phase that means anything.

    The empty `rule_uuid` is the *only* thing that tells the action this is a frame
    rather than an event transition, so it is spelled out here rather than defaulted.
    """
    return WatchEventSignal(
        rule_uuid="",
        rule_name="",
        phase=EventPhase.UPDATE,
        at=at,
        started_at=at,
        source_uuid=source_uuid,
    )


async def _open(action, context, started_at=100.0, **overrides):
    return await action.run(_event_signal(EventPhase.START, started_at=started_at, **overrides), context)


async def _feed(action, context, times, source_uuid="src-1"):
    """Offer frames at the given monotonic instants."""
    for at in times:
        await action.run(_frame_signal(at, source_uuid=source_uuid), context)


async def _close(action, context, started_at=100.0, at=200.0, **overrides):
    return await action.run(_event_signal(EventPhase.END, started_at=started_at, at=at, **overrides), context)


@pytest.fixture
def action(tmp_path):
    return RecordEventTimelapseAction(interval_s=10.0, min_frames=2, output_dir=str(tmp_path / "timelapses"))


class TestRegistration:
    def test_it_is_registered_and_constructible(self):
        assert "record_event_timelapse" in action_registry
        assert isinstance(create_action("record_event_timelapse"), RecordEventTimelapseAction)

    def test_it_declares_all_three_triggers(self):
        row = type("Row", (), {"trigger": EVENT_ENDED, "enabled": True})()
        assert RecordEventTimelapseAction.triggers(row) == {EVENT_STARTED, FRAME_ANALYSED, EVENT_ENDED}

    def test_the_dispatcher_honours_the_declared_triggers(self):
        """The row sits on `event_ended`, so without `fires_on` the frame trigger — the
        one it actually collects on — would never reach it, and every event would
        compile from nothing."""
        row = type(
            "Row", (), {"trigger": EVENT_ENDED, "enabled": True, "rule_uuids_json": [], "source_uuids_json": []}
        )()
        fires_on = RecordEventTimelapseAction.triggers(row)
        assert action_applies(row, FRAME_ANALYSED, source_uuid="src-1", fires_on=fires_on)
        assert not action_applies(row, FRAME_ANALYSED, source_uuid="src-1")

    def test_an_ordinary_action_still_fires_only_on_its_own_trigger(self):
        """The default must not have widened for everything else."""
        row = type(
            "Row", (), {"trigger": EVENT_ENDED, "enabled": True, "rule_uuids_json": [], "source_uuids_json": []}
        )()
        from apps.helpers.actions.saveSnapshotAction import SaveSnapshotAction

        fires_on = SaveSnapshotAction.triggers(row)
        assert fires_on == {EVENT_ENDED}
        assert not action_applies(row, EVENT_STARTED, rule_uuid="rule-1", fires_on=fires_on)


class TestSampling:
    async def test_frames_are_kept_one_per_interval(self, action):
        """The watcher analyses several frames a second; the interval is what makes this
        a timelapse rather than a re-encode of the whole event."""
        context = _context()
        await _open(action, context)
        # 30 seconds of frames at 2 fps — 60 offered, 4 buckets touched (0, 10, 20, 30).
        await _feed(action, context, [100.0 + i * 0.5 for i in range(61)])
        session = next(iter(action._sessions.values()))
        assert session.frames == 4

    async def test_nothing_is_kept_while_no_event_is_open(self, action):
        """The whole point: between visits this action costs a dict scan and no disk."""
        context = _context()
        await _feed(action, context, [100.0, 200.0, 300.0])
        assert action._sessions == {}

    async def test_frames_stop_at_the_ceiling(self, tmp_path):
        """A stuck detection must not be able to fill the disk."""
        # `min_frames` has to come down with it: the ceiling is clamped up to the floor,
        # so leaving the default 10 here would silently test a ceiling of 10.
        action = RecordEventTimelapseAction(interval_s=1.0, min_frames=1, max_frames=5, output_dir=str(tmp_path))
        context = _context()
        await _open(action, context)
        await _feed(action, context, [100.0 + i for i in range(50)])
        assert next(iter(action._sessions.values())).frames == 5

    async def test_two_rules_on_one_camera_get_their_own_videos(self, action):
        """Each session buffers independently, and one frame is encoded once for both."""
        context = _context()
        await _open(action, context, rule_uuid="rule-1", rule_name="in window")
        await _open(action, context, rule_uuid="rule-2", rule_name="on sill")
        await _feed(action, context, [100.0, 110.0, 120.0])
        assert len(action._sessions) == 2
        assert [s.frames for s in action._sessions.values()] == [3, 3]

    async def test_a_frame_from_another_camera_is_ignored(self, action):
        """Sessions are keyed by source; a busy hallway camera must not fill the window
        camera's timelapse."""
        context = _context()
        await _open(action, context)
        await _feed(action, _context(source_uuid="src-2"), [110.0, 120.0], source_uuid="src-2")
        assert next(iter(action._sessions.values())).frames == 0

    async def test_a_missing_frame_does_not_consume_the_interval(self, action):
        """A source that briefly has nothing decoded should resume on the next frame,
        not skip a whole interval it never got a picture for."""
        context = _context()
        await _open(action, context)
        result = await action.run(_frame_signal(100.0), _context(frame=None))
        assert not result.ok
        await _feed(action, context, [100.5])
        assert next(iter(action._sessions.values())).frames == 1


class TestCompile:
    async def test_an_event_produces_a_playable_mp4(self, action, tmp_path):
        context = _context()
        await _open(action, context)
        await _feed(action, context, [100.0 + i * 10 for i in range(6)])
        result = await _close(action, context)

        assert result.ok, result.detail
        path = Path(result.data["path"])
        assert path.is_file() and path.suffix == ".mp4"
        assert result.data["frames"] == 6
        # Read back rather than trusted: a file ffmpeg wrote but no decoder will open is
        # exactly the failure a mocked subprocess would have hidden.
        capture = cv2.VideoCapture(str(path))
        try:
            assert capture.isOpened()
            assert capture.read()[0]
        finally:
            capture.release()

    async def test_the_duration_is_the_videos_not_the_events(self, action):
        """A two-hour event compiles to seconds. Reporting the event's span would leave
        the player's scrubber wrong by two orders of magnitude."""
        context = _context()
        await _open(action, context)
        await _feed(action, context, [100.0 + i * 10 for i in range(12)])
        result = await _close(action, context, at=100.0 + 7200)

        assert result.data["durationS"] == pytest.approx(12 / action.output_fps)
        assert result.data["sourceSpanS"] < 60  # wallclock across the test, not the event

    async def test_a_short_event_writes_nothing(self, tmp_path):
        """Below `min_frames` there is no file, no row and no badge — but the event and
        its snapshot are untouched."""
        action = RecordEventTimelapseAction(interval_s=10.0, min_frames=10, output_dir=str(tmp_path))
        context = _context()
        await _open(action, context)
        await _feed(action, context, [100.0, 110.0, 120.0])
        result = await _close(action, context)

        assert result.ok
        assert "below the 10-frame minimum" in result.detail
        assert result.data == {}
        assert list(Path(tmp_path).rglob("*.mp4")) == []

    async def test_the_frame_buffer_is_removed_after_compiling(self, action):
        context = _context()
        await _open(action, context)
        directory = next(iter(action._sessions.values())).directory
        await _feed(action, context, [100.0 + i * 10 for i in range(4)])
        await _close(action, context)
        assert not directory.exists()

    async def test_the_frame_buffer_is_removed_when_nothing_is_compiled(self, tmp_path):
        """The short-event path returns early, and an early return that skipped the
        clean-up would leak a temp directory per brief visit — which is most of them."""
        action = RecordEventTimelapseAction(interval_s=10.0, min_frames=99, output_dir=str(tmp_path))
        context = _context()
        await _open(action, context)
        directory = next(iter(action._sessions.values())).directory
        await _feed(action, context, [100.0, 110.0])
        await _close(action, context)
        assert not directory.exists()

    async def test_closing_an_event_that_never_opened_is_not_an_error(self, action):
        """A watcher resumed mid-event has no session to compile, and must not report a
        failure for it — the counter drives the status panel's error badge."""
        result = await _close(action, _context())
        assert result.ok
        assert result.data == {}

    async def test_a_second_start_discards_the_first_buffer(self, action):
        """A lost close would otherwise splice two visits into one video."""
        context = _context()
        await _open(action, context)
        first = next(iter(action._sessions.values())).directory
        await _feed(action, context, [100.0, 110.0])
        await _open(action, context)
        assert not first.exists()
        assert next(iter(action._sessions.values())).frames == 0

    async def test_close_drops_every_open_buffer(self, action):
        """`_deinit` calls this. Anything still open is an event that will never compile,
        holding a directory that nothing will ever read."""
        context = _context()
        await _open(action, context, rule_uuid="rule-1")
        await _open(action, context, rule_uuid="rule-2")
        directories = [s.directory for s in action._sessions.values()]
        await action.close()
        assert action._sessions == {}
        assert not any(d.exists() for d in directories)


class TestParams:
    def test_an_unset_interval_falls_back_to_the_default(self):
        """0 and None both mean "not configured" through the UI's form machinery, and
        take the default — the same reading `save_timelapse` gives them."""
        assert RecordEventTimelapseAction(interval_s=0).interval_s == 10.0
        assert RecordEventTimelapseAction(interval_s=None).interval_s == 10.0

    def test_a_negative_interval_is_clamped(self):
        """Negative would make every frame land in a new bucket — several a second, for
        the length of the event, which is never what "timelapse" means."""
        assert RecordEventTimelapseAction(interval_s=-5).interval_s == 1.0

    def test_max_frames_cannot_sit_below_min_frames(self):
        """That combination can never compile anything, and would silently produce no
        video for every event forever."""
        action = RecordEventTimelapseAction(min_frames=20, max_frames=5)
        assert action.max_frames >= action.min_frames

    def test_the_schema_covers_every_constructor_option(self):
        """The Watchers pane renders the schema; a parameter missing from it is one the
        user cannot set and will never discover."""
        schema = RecordEventTimelapseAction.params_schema()
        assert {"interval_s", "output_fps", "min_frames", "max_frames", "annotate"} <= set(schema)

    def test_an_unset_output_dir_lands_under_the_recordings_root(self):
        assert RecordEventTimelapseAction().output_dir == DEFAULT_TIMELAPSE_ROOT

    def test_a_relative_output_dir_is_resolved_under_the_recordings_root(self):
        """The Watchers pane had no form for this action, so every row saved through it
        fell through to the snapshot branch and stored `output_dir: "snapshots"` — a
        dataset-relative spelling. Taken literally it is relative to the process's
        working directory, which puts the mp4 outside the tree the retention sweep walks
        and outside the tree the file endpoint serves: the recording row points at a
        file the UI cannot play and nothing will ever delete."""
        action = RecordEventTimelapseAction(output_dir="snapshots")
        assert action.output_dir == DEFAULT_TIMELAPSE_ROOT / "snapshots"

    def test_an_absolute_output_dir_is_left_alone(self):
        """Someone bind-mounting a NAS path meant it."""
        action = RecordEventTimelapseAction(output_dir="/mnt/nas/timelapses")
        assert action.output_dir == Path("/mnt/nas/timelapses")

    def test_a_blank_output_dir_falls_back_to_the_default(self):
        assert RecordEventTimelapseAction(output_dir="").output_dir == DEFAULT_TIMELAPSE_ROOT
        assert RecordEventTimelapseAction(output_dir=None).output_dir == DEFAULT_TIMELAPSE_ROOT
