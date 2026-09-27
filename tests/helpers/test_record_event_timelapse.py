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

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from apps.helpers.actions.action import ActionContext
from apps.helpers.actions.recordEventTimelapseAction import (
    BUFFER_OWNER_FILENAME,
    DEFAULT_TIMELAPSE_ROOT,
    RecordEventTimelapseAction,
    sweep_frame_buffers,
)
from apps.helpers.actions.Registry import action_registry, create_action
from apps.helpers.watch.events import EventPhase, WatchEventSignal
from apps.helpers.watch.scoping import EVENT_ENDED, EVENT_STARTED, FRAME_ANALYSED, action_applies


def _frame(width=320, height=240, value=40):
    return np.full((height, width, 3), value, dtype=np.uint8)


def _unused_pid() -> int:
    """A pid that is certainly not a live user process.

    0 is not usable — `os.kill(0, 0)` signals the whole process group — so this walks
    up from a high number until it finds one nothing is using.
    """
    import os

    for pid in range(4_194_303, 4_190_000, -1):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
        except OSError:
            continue
    raise RuntimeError("no free pid to test with")


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


@pytest.fixture(autouse=True)
def _isolate_frame_buffers(tmp_path, monkeypatch):
    """Keep every frame buffer this module opens inside the test's own `tmp_path`.

    A session buffers frames into `tempfile.mkdtemp()`, and most tests here open one
    *without* closing it — deliberately, because the open state is the thing under
    test. Only `_on_end` discards, so every one of those left a directory behind in the
    real temp directory, for as long as this file has existed. 113 of them were sitting
    in the container's `/tmp` when this was noticed, every frame 320x240 solid grey:
    `_frame()`, not a camera.

    They were also mistaken for evidence of the leak the *watcher* has — they have the
    same `wwtl-` prefix — which is worth knowing before reading a buffer count as a
    production symptom again.

    Isolating here rather than discarding in each test keeps the leak impossible rather
    than merely tidied, and it is what lets the preview test below assert on a bare
    glob instead of diffing a process-global directory.
    """
    buffers = tmp_path / "frame-buffers"
    buffers.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(buffers))
    return buffers


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
        assert {"interval_s", "output_fps", "min_frames", "max_frames", "preroll_frames", "save_overlay"} <= set(schema)
        # Removed rather than defaulted off: the boxes are a sidecar the player toggles,
        # so a burn-them-in checkbox can now only spoil a recording permanently. Offering
        # it would be offering a way to ruin every frame for training.
        assert "annotate" not in schema

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


class TestTheDetectionOverlay:
    """Boxes beside the video instead of burnt into it.

    The point is not cosmetic. `annotate` paints rectangles into the JPEGs before they
    are compiled, which makes every frame of the finished timelapse useless as a
    training image — and the modal's "save frame for annotation" is aimed straight at a
    dataset. A sidecar leaves the pixels alone, so the frame that reaches the dataset
    has never had a box on it, and the player can still show what the watcher saw.
    """

    def _det(self, x1=32, y1=24, x2=96, y2=120):
        """Stands in for `apps.helpers.tracking.Detection`, read by attribute."""
        from types import SimpleNamespace

        return SimpleNamespace(bbox=(x1, y1, x2, y2), label_name="cat", label_uuid="lbl-1", confidence=0.8)

    def _sidecar(self, result) -> dict:
        import json

        path = Path(result.data["overlayPath"])
        return json.loads(path.read_text())

    async def test_the_sidecar_lands_beside_the_video(self, action):
        context = _context(detections=[self._det()])
        await _open(action, context)
        await _feed(action, context, [100.0 + i * 10 for i in range(4)])
        result = await _close(action, context)

        sidecar = Path(result.data["overlayPath"])
        assert sidecar.is_file()
        # Named after the video, so the two sort together and the sidecar is obviously
        # derived from that exact file — the retention sweep leans on this too.
        assert sidecar.name == Path(result.data["path"]).name + ".boxes.json"

    async def test_there_is_one_entry_per_kept_frame_not_per_analysed_frame(self, action):
        """The alignment that makes the overlay correct. Frames arrive far faster than
        the interval keeps them; an entry per *analysed* frame would shift every box
        later in the video by a growing amount."""
        context = _context(detections=[self._det()])
        await _open(action, context)
        # Twelve analysed frames 2.5 s apart span 27.5 s, which is three whole 10 s
        # buckets — so three kept frames out of twelve offered.
        await _feed(action, context, [100.0 + i * 2.5 for i in range(12)])
        result = await _close(action, context)

        doc = self._sidecar(result)
        assert result.data["frames"] == len(doc["frames"]) == 3
        assert [f["t"] for f in doc["frames"]] == pytest.approx([i / action.output_fps for i in range(3)], abs=1e-3)

    async def test_the_boxes_are_normalised_against_the_analysis_frame(self, action):
        """Detections are in analysis pixels and the video is not, so a box stored in
        pixels would sit wrong by the downscale ratio — and would rot the day someone
        changed the analysis width."""
        context = _context(frame=_frame(width=320, height=240), detections=[self._det(32, 24, 96, 120)])
        await _open(action, context)
        await _feed(action, context, [100.0 + i * 10 for i in range(4)])
        result = await _close(action, context)

        box = self._sidecar(result)["frames"][0]["boxes"][0]
        assert (box["x"], box["y"], box["w"], box["h"]) == pytest.approx((0.1, 0.1, 0.2, 0.4))
        assert box["labelUuid"] == "lbl-1"

    async def test_a_frame_where_nothing_was_seen_is_recorded_as_empty(self, action):
        """An empty list is a fact, not a gap: without it the player holds the previous
        frame's boxes and draws a cat that had already left."""
        context = _context(detections=[self._det()])
        blank = _context(detections=[])
        await _open(action, context)
        await _feed(action, context, [100.0, 110.0])
        await _feed(action, blank, [120.0, 130.0])
        result = await _close(action, context)

        doc = self._sidecar(result)
        assert [len(f["boxes"]) for f in doc["frames"]] == [1, 1, 0, 0]

    async def test_the_video_itself_is_never_touched(self, action):
        """The whole reason for the sidecar: the pixels stay as the camera sent them,
        whatever the detector saw, so any frame of the video is a usable training image.

        There is no longer an option that could make this false — the burn-in was
        removed with the sidecar — so this is the assertion that keeps it that way.
        """
        context = _context(detections=[self._det()])
        await _open(action, context)
        await _feed(action, context, [100.0 + i * 10 for i in range(4)])
        result = await _close(action, context)

        assert result.data["overlayPath"] is not None
        # Pixel-level proof that nothing was drawn: the fixture frame is a flat fill, so
        # any burnt-in rectangle would introduce a second colour.
        capture = cv2.VideoCapture(result.data["path"])
        try:
            ok, decoded = capture.read()
            assert ok
        finally:
            capture.release()
        assert decoded.std() < 8.0, "the compiled frames are not flat — something was drawn on them"

    async def test_switching_the_sidecar_off_writes_none(self, tmp_path):
        action = RecordEventTimelapseAction(interval_s=10.0, min_frames=2, output_dir=str(tmp_path), save_overlay=False)
        context = _context(detections=[self._det()])
        await _open(action, context)
        await _feed(action, context, [100.0 + i * 10 for i in range(4)])
        result = await _close(action, context)

        assert result.data["overlayPath"] is None
        assert list(Path(tmp_path).rglob("*.boxes.json")) == []

    async def test_an_event_with_no_detections_at_all_gets_no_sidecar(self, action):
        """An all-empty document claims the detector saw nothing for the whole event,
        which is a stronger statement than "no record" — and it would put an overlay
        toggle on a video that can never show one."""
        context = _context(detections=[])
        await _open(action, context)
        await _feed(action, context, [100.0 + i * 10 for i in range(4)])
        result = await _close(action, context)

        assert result.data["overlayPath"] is None


class TestPreRoll:
    """Frames from before the event opened, so the video shows the approach.

    Without it a timelapse opens on a subject that is already there — `min_duration_s`
    means the rule only fires once the predicate has held for a couple of seconds, and
    the interesting part is what happened on the way in.
    """

    def _action(self, tmp_path, **kw):
        kw.setdefault("interval_s", 10.0)
        kw.setdefault("min_frames", 1)
        return RecordEventTimelapseAction(output_dir=str(tmp_path), **kw)

    async def test_frames_from_before_the_event_are_put_at_the_front(self, tmp_path):
        action = self._action(tmp_path, preroll_frames=3)
        context = _context()
        # Idle camera: nothing is open, so these can only reach the ring.
        await _feed(action, context, [0.0, 10.0, 20.0, 30.0])
        await _open(action, context, started_at=40.0)
        await _feed(action, context, [40.0, 50.0])
        result = await _close(action, context, started_at=40.0, at=60.0)

        # Three pre-roll frames (the ring's cap, not all four offered) plus the two live.
        assert result.data["frames"] == 5

    async def test_the_ring_never_grows_past_its_cap(self, tmp_path):
        """It is held JPEGs, so an unbounded ring is a memory leak on every idle
        camera — the one place this action would otherwise cost something all day."""
        action = self._action(tmp_path, preroll_frames=2)
        context = _context()
        await _feed(action, context, [i * 10.0 for i in range(50)])

        assert len(action._preroll["src-1"]) == 2

    async def test_an_idle_camera_costs_nothing_when_preroll_is_off(self, tmp_path):
        """The default. Sampling an idle camera all day to serve a feature nobody asked
        for is exactly the regression this guards."""
        action = self._action(tmp_path, preroll_frames=0)
        context = _context()
        await _feed(action, context, [0.0, 10.0, 20.0])

        assert action._preroll == {}

    async def test_the_ring_is_only_sampled_once_per_interval(self, tmp_path):
        """A watcher analysing at 5 fps must not put fifty frames in a ten-second
        bucket."""
        action = self._action(tmp_path, preroll_frames=5)
        context = _context()
        await _feed(action, context, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        assert len(action._preroll["src-1"]) == 1

    async def test_the_reported_start_is_the_first_frame_shown_not_the_events(self, tmp_path):
        """The modal measures the scrubber against this. Reporting the event's start
        over a video that begins thirty seconds earlier puts every timestamp out."""
        action = self._action(tmp_path, preroll_frames=2)
        context = _context()
        await _feed(action, context, [0.0, 10.0])
        first_ring_at = action._preroll["src-1"][0][2]
        await _open(action, context, started_at=20.0)
        await _feed(action, context, [20.0])
        result = await _close(action, context, started_at=20.0, at=30.0)

        expected = datetime.fromtimestamp(first_ring_at, tz=timezone.utc)
        assert result.data["startedAt"].startswith(expected.isoformat()[:19])

    async def test_the_overlay_stays_aligned_across_the_join(self, tmp_path):
        """Pre-roll frames carry their own boxes. One missing entry there shifts every
        annotation in the rest of the video by a frame."""
        from types import SimpleNamespace

        det = SimpleNamespace(bbox=(32, 24, 96, 120), label_name="cat", label_uuid="lbl-1", confidence=0.8)
        action = self._action(tmp_path, preroll_frames=2)
        empty, seen = _context(detections=[]), _context(detections=[det])
        await _feed(action, empty, [0.0, 10.0])
        await _open(action, seen, started_at=20.0)
        await _feed(action, seen, [20.0, 30.0])
        result = await _close(action, seen, started_at=20.0, at=40.0)

        import json

        doc = json.loads(Path(result.data["overlayPath"]).read_text())
        assert len(doc["frames"]) == result.data["frames"] == 4
        # Empty through the lead-in, boxed once the event opens — the join is where an
        # off-by-one would show.
        assert [len(f["boxes"]) for f in doc["frames"]] == [0, 0, 1, 1]

    async def test_a_second_rule_opening_later_gets_a_current_lead_in(self, tmp_path):
        """The ring keeps being fed while an event is open, so the second rule's
        pre-roll is the last 20 seconds — not whatever preceded the first event."""
        action = self._action(tmp_path, preroll_frames=2)
        context = _context()
        await _feed(action, context, [0.0, 10.0])
        await _open(action, context, started_at=20.0)
        await _feed(action, context, [20.0, 30.0, 40.0])
        await _open(action, context, started_at=50.0, rule_uuid="rule-2", rule_name="second rule")
        await _feed(action, context, [50.0])
        second = await _close(action, context, started_at=50.0, at=60.0, rule_uuid="rule-2", rule_name="second rule")

        # Two rung frames + the live one. That it is non-empty at all is the point: the
        # ring did not go stale while the first event held it.
        assert second.data["frames"] == 3


class TestMidEventPreview:
    """Watching a long event before it ends.

    The action's whole reason to exist is events too long to watch in real time, and
    those are exactly the ones that hold their video hostage until the subject leaves.
    The preview compiles what is on disk without disturbing the session — the failure to
    guard against is a preview that consumes, truncates or renames the frames the real
    compile still needs.
    """

    async def test_a_preview_plays_before_the_event_ends(self, action, tmp_path):
        context = _context()
        await _open(action, context)
        await _feed(action, context, [100.0, 110.0, 120.0, 130.0])

        preview = await action.preview(("src-1", "rule-1", ""))
        assert preview is not None
        # A real mp4, not merely bytes: the point of compiling server-side is that a
        # browser can play the result.
        out = tmp_path / "preview.mp4"
        out.write_bytes(preview["video"])
        capture = cv2.VideoCapture(str(out))
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) >= 1
        capture.release()

    async def test_the_event_still_compiles_everything_afterwards(self, action):
        """The preview must not consume the buffer. This is the regression that would
        cost the user the real video in exchange for a glimpse of it."""
        context = _context()
        await _open(action, context)
        await _feed(action, context, [100.0, 110.0, 120.0, 130.0])

        await action.preview(("src-1", "rule-1", ""))
        await _feed(action, context, [140.0, 150.0])
        result = await _close(action, context, at=160.0)

        assert result.ok
        # Every frame sampled, before and after the preview — nothing was eaten.
        assert result.data["frames"] == 6

    async def test_the_newest_frame_is_left_out(self, action):
        """It may be half-written — `write_bytes` is not atomic and the sampler could be
        inside that very file — and a torn JPEG at the end corrupts the preview."""
        context = _context()
        await _open(action, context)
        await _feed(action, context, [100.0, 110.0, 120.0])

        preview = await action.preview(("src-1", "rule-1", ""))
        assert preview is not None and preview["frames"] == 2

    async def test_a_session_with_almost_nothing_previews_nothing(self, action):
        """One usable frame is not a video. Returning None is what lets the caller say
        "not yet" rather than handing back an unplayable file."""
        context = _context()
        await _open(action, context)
        await _feed(action, context, [100.0, 110.0])
        assert await action.preview(("src-1", "rule-1", "")) is None

    async def test_an_unknown_event_previews_nothing(self, action):
        context = _context()
        await _open(action, context)
        await _feed(action, context, [100.0, 110.0, 120.0])
        assert await action.preview(("src-1", "another-rule", "")) is None

    async def test_two_rules_on_one_camera_preview_their_own_buffers(self, action):
        """Same reason each gets its own video: the key is the event, not the camera."""
        context = _context()
        await _open(action, context)
        await _open(action, context, rule_uuid="rule-2", started_at=100.0)
        await _feed(action, context, [100.0, 110.0, 120.0, 130.0])

        first = await action.preview(("src-1", "rule-1", ""))
        second = await action.preview(("src-1", "rule-2", ""))
        assert first is not None and second is not None
        assert first["frames"] == second["frames"]

    async def test_a_long_event_previews_only_its_most_recent_frames(self, tmp_path, monkeypatch):
        """An unbounded re-encode on every click would cost more than the watcher does.
        The tail is the part anyone previewing a running event is looking for."""
        action = RecordEventTimelapseAction(interval_s=10.0, min_frames=2, output_dir=str(tmp_path / "t"))
        monkeypatch.setattr(action, "PREVIEW_MAX_FRAMES", 4)
        context = _context()
        await _open(action, context)
        await _feed(action, context, [100.0 + 10 * i for i in range(12)])

        preview = await action.preview(("src-1", "rule-1", ""))
        assert preview is not None
        assert preview["frames"] == 4
        # And it says so, so the modal does not present a tail as the whole event.
        assert preview["partial"] is True

    async def test_two_events_of_one_rule_preview_their_own_buffers(self, action):
        """Two cats at one bowl are two events on the same rule, and two videos.

        The session key used to be the rule plus the matched labels, so two concurrent
        events that agreed on both — which is what two cats the model cannot tell apart
        looks like — shared one buffer: whichever closed first compiled both cats' frames
        and the other found its directory gone.
        """
        context = _context()
        await action.run(_event_signal(EventPhase.START, instance_id="rule-1:1"), context)
        await action.run(_event_signal(EventPhase.START, instance_id="rule-1:2"), context)

        assert sorted(action.open_session_keys()) == [
            ("src-1", "rule-1", "rule-1:1"),
            ("src-1", "rule-1", "rule-1:2"),
        ]

    async def test_a_preview_leaves_no_scratch_behind(self, action):
        """It compiles into a temp directory of its own; leaking one per click on a
        watcher that runs for weeks is a disk leak nothing would ever clean up."""
        context = _context()
        await _open(action, context)
        await _feed(action, context, [100.0, 110.0, 120.0, 130.0])

        session = action._sessions[("src-1", "rule-1", "")]

        # A bare glob is safe again: `_isolate_frame_buffers` gives this test its own
        # temp root, so nothing else can have a scratch dir in flight here. It used to
        # have to diff before-and-after against the process-global directory, which
        # failed whenever a second xdist worker, a concurrent suite or another agent in
        # the same worktree had one open — and that is routine on this box.
        tmp_root = Path(tempfile.gettempdir())
        await action.preview(("src-1", "rule-1", ""))

        assert not (session.directory / "preview.txt").exists()
        leaked = sorted(tmp_root.glob("wwtl-preview-*"))
        assert not leaked, f"preview left scratch directories behind: {leaked}"


class TestOrphanedFrameBuffersAreSwept:
    """A killed watcher leaves its frame buffers behind, and nothing ever reclaimed them.

    The ordinary paths are covered: `_on_end` discards in a `finally`, and a watcher
    that stops cleanly force-closes everything open on the way out. What was missing is
    the case where no Python runs at all — an OOM kill, `docker kill`, a power cut.
    Clip chunks already get `sweep_chunk_root()` at watcher start; these got nothing, and
    sixteen orphaned buffers were sitting in the container's temp directory against zero
    open events when this was written.

    The sweep keys on the pid that opened each buffer rather than on its age, because
    every watcher shares the tasks_server process: "left behind by the process the OOM
    killer took" and "being filled right now by the other profile's watcher" are
    otherwise indistinguishable, and guessing wrong deletes a live event's frames.
    """

    def _buffer(self, root: Path, name: str, owner: str | None) -> Path:
        directory = root / name
        directory.mkdir()
        (directory / "000000.jpg").write_bytes(b"not really a jpeg")
        if owner is not None:
            (directory / BUFFER_OWNER_FILENAME).write_text(owner)
        return directory

    def test_a_buffer_from_a_dead_process_is_removed(self, tmp_path):
        # A pid that cannot be running: 0 is never a live user process.
        orphan = self._buffer(tmp_path, "wwtl-dead", owner=str(_unused_pid()))

        assert sweep_frame_buffers(tmp_path) == 1
        assert not orphan.exists()

    def test_a_buffer_this_process_still_owns_is_left_alone(self):
        """The one that matters: a second watcher starting must not rob the first."""
        import os

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = self._buffer(root, "wwtl-live", owner=str(os.getpid()))
            dead = self._buffer(root, "wwtl-dead", owner=str(_unused_pid()))

            assert sweep_frame_buffers(root) == 1
            assert live.exists(), "a buffer owned by this very process is in use"
            assert not dead.exists()

    def test_an_unmarked_buffer_is_treated_as_an_orphan(self):
        """Buffers written before the marker existed can only be from an earlier run."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = self._buffer(root, "wwtl-legacy", owner=None)

            assert sweep_frame_buffers(root) == 1
            assert not legacy.exists()

    def test_it_touches_nothing_that_is_not_a_frame_buffer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            innocent = root / "someone-elses-tempdir"
            innocent.mkdir()
            (innocent / "important.txt").write_text("keep me")
            loose = root / "wwtl-not-a-directory"
            loose.write_text("a file, not a buffer")

            assert sweep_frame_buffers(root) == 0
            assert (innocent / "important.txt").read_text() == "keep me"
            assert loose.exists()

    def test_a_missing_root_is_not_an_error(self, tmp_path):
        assert sweep_frame_buffers(tmp_path / "nope") == 0

    async def test_a_live_session_marks_its_own_buffer(self, _isolate_frame_buffers):
        """The sweep is only safe because `_on_start` stamps the buffer as it opens.

        Runs against the module's isolated temp root, so the sweep under test cannot
        reach a buffer another test — or the live watcher sharing this container — is
        filling.
        """
        import os

        root = _isolate_frame_buffers
        action = RecordEventTimelapseAction(interval_s=1.0)
        await action.run(_event_signal(EventPhase.START), _context())
        directory = next(iter(action._sessions.values())).directory
        assert directory.parent == root, "the session must have landed in the isolated root"

        orphan = self._buffer(root, "wwtl-dead", owner=str(_unused_pid()))
        assert (directory / BUFFER_OWNER_FILENAME).read_text().strip() == str(os.getpid())
        # The live one survives a sweep that removes the dead one beside it.
        assert sweep_frame_buffers(root) == 1
        assert directory.exists()
        assert not orphan.exists()
