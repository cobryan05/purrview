"""Which clip backend a watcher uses, per source, and the frame ring behind the fallback.

Two things were wrong before this, both silent:

  * **One backend for the whole profile.** The task built `PlaybackClipBackend` once and
    handed it to every `record_clip` action, so in a profile mixing an RTSP camera with a
    snapshot poller the poller's events failed every clip with "Source has no relay path".
  * **The frame ring was never fed.** `FrameRing`, `RingClipBackend` and the
    `ring_seconds` / `ring_max_mb` / `buffer_mode` columns all existed, and nothing
    constructed, wrote to or read any of them — so a source with no rewind buffer had no
    clip path at all, while the config implied it did.

The ring is only for sources that cannot be rewound. A profile of RTSP cameras must
allocate none, which is what most of these tests are guarding.
"""

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from apps.helpers.actions.action import ActionContext
from apps.helpers.actions.recordClipAction import (
    LAST_FRAME_HOLD_S,
    FrameRing,
    FrameSpool,
    PlaybackClipBackend,
    RecordClipAction,
    RingClipBackend,
    encode_timeline,
)
from apps.helpers.watch.engine import WatchEngine
from apps.helpers.watch.events import EventPhase, WatchEventSignal
from apps.helpers.watch.rules import RuleSpec
from apps.tasks_server.tasks._watch_pipeline import WatchPipeline
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask


def _pipeline(source_uuid="src-1", *, buffered=False, buffer_mode="auto", ring_seconds=20.0, ring_max_mb=32.0):
    return WatchPipeline(
        source_uuid=source_uuid,
        source_name=f"cam-{source_uuid}",
        provider=SimpleNamespace(),
        tracker=SimpleNamespace(update=lambda dets, now: []),
        engine=WatchEngine(rules=[], zones=[], source_uuid=source_uuid),
        buffered=buffered,
        buffer_mode=buffer_mode,
        ring_seconds=ring_seconds,
        ring_max_mb=ring_max_mb,
    )


def _task(pipelines, *, relay_client=object()):
    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._pipelines = list(pipelines)
    task._frame_rings = {}
    task._wallclock_offset = 1_700_000_000.0
    task._manager = SimpleNamespace(get_relay_api_client=lambda: relay_client)
    task._actions = []
    task.log = lambda *a, **k: None
    return task


def _packet(payload: bytes = b"\xff\xd8jpegbytes") -> str:
    """A frame as `base64_encode_jpg` produces it — a JSON packet, not bare base64."""
    return json.dumps({"format": "jpeg", "metadata": {}, "data": base64.b64encode(payload).decode()})


def _spill(tmp_path, index, blob: bytes) -> Path:
    """The timeline encoder reads frames from disk; tests hold theirs as bytes."""
    path = Path(tmp_path) / f"spill{index:05d}.jpg"
    path.write_bytes(blob)
    return path


def _context(source_uuid="src-1"):
    return ActionContext(profile_uuid="p1", profile_name="P", source_uuid=source_uuid, source_name="cam")


class TestBackendSelection:
    def test_a_buffered_source_is_cut_from_the_relay(self):
        task = _task([_pipeline("src-1", buffered=True)])

        backend = task._clip_backend_for(_context("src-1"))

        assert isinstance(backend, PlaybackClipBackend)

    def test_an_unbuffered_source_falls_back_to_its_ring(self):
        pipeline = _pipeline("src-1", buffered=False)
        task = _task([pipeline])
        task._push_to_ring(pipeline, _packet(), now=100.0)

        backend = task._clip_backend_for(_context("src-1"))

        assert isinstance(backend, RingClipBackend)

    def test_each_source_in_a_mixed_profile_gets_its_own_backend(self):
        """The bug: one backend for the profile meant the snapshot source used the relay."""
        relayed = _pipeline("rtsp", buffered=True)
        snapshot = _pipeline("snap", buffered=False)
        task = _task([relayed, snapshot])
        task._push_to_ring(snapshot, _packet(), now=100.0)

        assert isinstance(task._clip_backend_for(_context("rtsp")), PlaybackClipBackend)
        assert isinstance(task._clip_backend_for(_context("snap")), RingClipBackend)

    def test_an_unbuffered_source_with_no_frames_yet_has_no_backend(self):
        """Better than a backend that raises: the action reports it as a clean failure."""
        task = _task([_pipeline("src-1", buffered=False)])

        assert task._clip_backend_for(_context("src-1")) is None

    def test_an_unknown_source_has_no_backend(self):
        task = _task([_pipeline("src-1")])

        assert task._clip_backend_for(_context("nope")) is None

    def test_a_manager_without_a_relay_client_fails_loudly(self):
        """This used to return None, which is how a missing `get_relay_api_client`
        on the real Manager went unnoticed: every clip on a relayed camera failed
        with "no clip backend available" and nothing said why. A manager that cannot
        supply a relay client is a wiring bug, so it must raise.
        """
        task = _task([_pipeline("src-1", buffered=True)])
        task._manager = SimpleNamespace()

        with pytest.raises(AttributeError):
            task._clip_backend_for(_context("src-1"))


class TestFrameRingWiring:
    def test_a_buffered_source_never_allocates_a_ring(self):
        """A profile of RTSP cameras must not hold tens of megabytes it will never read."""
        pipeline = _pipeline("src-1", buffered=True)
        task = _task([pipeline])

        task._push_to_ring(pipeline, _packet(), now=100.0)

        assert task._frame_rings == {}

    def test_buffer_mode_off_suppresses_the_ring(self):
        """ "Off" has to mean off — otherwise it still costs RAM on every frame."""
        pipeline = _pipeline("src-1", buffered=False, buffer_mode="off")
        task = _task([pipeline])

        task._push_to_ring(pipeline, _packet(), now=100.0)

        assert task._frame_rings == {}

    def test_the_ring_honours_the_source_s_configured_bounds(self):
        """`ring_seconds`/`ring_max_mb` were stored and read by nothing until now."""
        pipeline = _pipeline("src-1", ring_seconds=5.0, ring_max_mb=1.0)
        task = _task([pipeline])

        task._push_to_ring(pipeline, _packet(), now=100.0)

        ring = task._frame_rings["src-1"]
        assert ring._max_seconds == 5.0
        assert ring._max_bytes == 1024 * 1024

    def test_frames_are_stored_in_wallclock_so_a_clip_window_can_find_them(self):
        """Rule timing is monotonic; clip windows are wallclock. Mixing them finds nothing."""
        pipeline = _pipeline("src-1")
        task = _task([pipeline])

        task._push_to_ring(pipeline, _packet(b"\xff\xd8one"), now=100.0)

        stamp, jpeg = task._frame_rings["src-1"]._frames[0]
        assert stamp == 100.0 + task._wallclock_offset
        assert jpeg == b"\xff\xd8one"

    def test_the_ring_unwraps_the_json_packet_rather_than_storing_it(self):
        """`base64_encode_jpg` returns a packet; storing that verbatim yields an
        unplayable "mp4" assembled from JSON."""
        pipeline = _pipeline("src-1")
        task = _task([pipeline])

        task._push_to_ring(pipeline, _packet(b"\xff\xd8realjpeg"), now=100.0)

        assert task._frame_rings["src-1"]._frames[0][1] == b"\xff\xd8realjpeg"

    def test_a_malformed_frame_does_not_kill_the_frame_loop(self):
        pipeline = _pipeline("src-1")
        task = _task([pipeline])

        task._push_to_ring(pipeline, "not json at all", now=100.0)

        assert task._frame_rings["src-1"].frame_count == 0

    def test_old_frames_are_evicted_by_age(self):
        pipeline = _pipeline("src-1", ring_seconds=5.0)
        task = _task([pipeline])

        task._push_to_ring(pipeline, _packet(b"\xff\xd8old"), now=100.0)
        task._push_to_ring(pipeline, _packet(b"\xff\xd8new"), now=110.0)

        ring = task._frame_rings["src-1"]
        assert ring.frame_count == 1
        assert ring._frames[0][1] == b"\xff\xd8new"


class TestPushToRingIdleTagging:
    """`_push_to_ring` has to know whether a frame came off the live tier or the slow
    idle poll to feed `FrameRing`'s sticky lead-in cache correctly — see
    `RingMqttImageProvider.is_live`."""

    def test_a_live_provider_frame_is_not_tagged_idle(self):
        pipeline = _pipeline("src-1")
        pipeline.provider = SimpleNamespace(is_live=True)
        task = _task([pipeline])

        task._push_to_ring(pipeline, _packet(), now=100.0)

        assert task._frame_rings["src-1"].lead_in_frames(before=1e12) == []

    def test_an_idle_provider_frame_is_tagged_idle(self):
        pipeline = _pipeline("src-1")
        pipeline.provider = SimpleNamespace(is_live=False)
        task = _task([pipeline])

        task._push_to_ring(pipeline, _packet(b"\xff\xd8snap"), now=100.0)

        pushed_at = 100.0 + task._wallclock_offset
        assert task._frame_rings["src-1"].lead_in_frames(before=pushed_at + 60.0) == [(pushed_at, b"\xff\xd8snap")]

    def test_a_provider_with_no_tier_concept_defaults_to_not_idle(self):
        """Every non-ring provider (RTSP, etc.) has no `is_live` at all — the sticky
        cache must simply never engage for them, not crash."""
        pipeline = _pipeline("src-1")
        task = _task([pipeline])

        task._push_to_ring(pipeline, _packet(), now=100.0)

        assert task._frame_rings["src-1"].lead_in_frames(before=1e12) == []


class TestFrameRingClosest:
    """`closest()` is what an on_exit event's still-image preview reads instead of the
    live frame — see `StreamWatcherTask._historical_frame_from_ring`."""

    def test_picks_the_nearer_of_the_two_straddling_frames(self):
        ring = FrameRing(max_seconds=30.0)
        ring.push(100.0, b"a")
        ring.push(104.0, b"b")
        assert ring.closest(101.9, tolerance=10.0) == b"a"
        assert ring.closest(102.1, tolerance=10.0) == b"b"

    def test_outside_tolerance_returns_none_even_if_the_ring_has_frames(self):
        ring = FrameRing(max_seconds=30.0)
        ring.push(100.0, b"a")
        assert ring.closest(150.0, tolerance=5.0) is None

    def test_an_empty_ring_returns_none(self):
        assert FrameRing().closest(100.0, tolerance=10.0) is None


class TestFrameRingHoldFrom:
    """The bug this fixes: `_evict`'s age cutoff climbs with every push, so a visit
    longer than `max_seconds` evicted its own arrival frame before `RecordClipAction`
    ever read the ring at `EventPhase.END` — losing pre-roll on any long-enough event.
    `hold_from` pins the cutoff while any event is open; `hold_from(None)` lifts it.
    The floor itself is re-derived from the engine's open events once a pass — see
    `TestOverlappingEventsDoNotEvictEachOthersFrames` for why it cannot be taken and
    released on the event edges."""

    def test_a_long_open_event_loses_its_preroll_without_a_hold(self):
        ring = FrameRing(max_seconds=20.0)
        ring.push(0.0, b"arrival")
        for t in range(1, 31):
            ring.push(float(t), b"frame")
        assert ring.frames_between(0.0, 1.0) == []

    def test_hold_from_keeps_the_prerolled_frame_alive_through_a_long_event(self):
        ring = FrameRing(max_seconds=20.0)
        ring.push(0.0, b"arrival")
        ring.hold_from(0.0)
        for t in range(1, 31):
            ring.push(float(t), b"frame")
        assert ring.frames_between(0.0, 0.5) == [(0.0, b"arrival")]

    def test_clearing_the_hold_resumes_normal_eviction(self):
        ring = FrameRing(max_seconds=20.0)
        ring.push(0.0, b"arrival")
        ring.hold_from(0.0)
        ring.push(30.0, b"frame")
        ring.hold_from(None)
        assert ring.frames_between(0.0, 1.0) == []

    def test_first_at_or_after_returns_the_earliest_matching_frame(self):
        ring = FrameRing(max_seconds=30.0)
        ring.push(10.0, b"a")
        ring.push(20.0, b"b")
        assert ring.first_at_or_after(15.0) == (20.0, b"b")
        assert ring.first_at_or_after(5.0) == (10.0, b"a")
        assert ring.first_at_or_after(25.0) is None


class TestDetectionTrackAllocation:
    """Who gets a detection ring, and who pays nothing for not wanting one.

    Same discipline as the frame ring above: this is on the per-frame path of a process
    that runs 24/7, so a watcher with no clip actions must not walk the action list once
    per analysed frame to be told "no" every time.
    """

    def _task(self, pipelines, actions):
        task = _task(pipelines)
        task._detection_tracks = {}
        task._actions = actions
        return task

    def _clip_action(self, *, save_overlay=True, sources=None):
        row = SimpleNamespace(kind="record_clip", uuid="a1", source_uuids_json=sources)
        return (row, SimpleNamespace(save_overlay=save_overlay, preroll_s=10.0))

    def test_a_clip_action_asking_for_an_overlay_gets_a_ring(self):
        pipeline = _pipeline("src-1", buffered=True)
        task = self._task([pipeline], [self._clip_action()])

        assert task._detection_track_for(pipeline) is not None

    def test_the_checkbox_being_off_allocates_nothing(self):
        pipeline = _pipeline("src-1", buffered=True)
        task = self._task([pipeline], [self._clip_action(save_overlay=False)])

        assert task._detection_track_for(pipeline) is None

    def test_a_profile_with_no_clip_action_allocates_nothing(self):
        pipeline = _pipeline("src-1", buffered=True)
        task = self._task([pipeline], [])

        assert task._detection_track_for(pipeline) is None

    def test_an_out_of_scope_clip_action_does_not_arm_another_camera(self):
        """Actions scope by source list; a ring on every camera would cost memory for
        cameras whose clips can never carry an overlay."""
        pipeline = _pipeline("src-2", buffered=True)
        task = self._task([pipeline], [self._clip_action(sources=["src-1"])])

        assert task._detection_track_for(pipeline) is None

    def test_the_same_ring_is_reused_across_frames(self):
        pipeline = _pipeline("src-1", buffered=True)
        task = self._task([pipeline], [self._clip_action()])

        assert task._detection_track_for(pipeline) is task._detection_track_for(pipeline)

    def test_a_negative_answer_is_cached_too(self):
        """Otherwise every frame of a watcher that wants no overlay re-scans the action
        list to reach the same conclusion."""
        pipeline = _pipeline("src-1", buffered=True)
        task = self._task([pipeline], [])

        task._detection_track_for(pipeline)

        assert "src-1" in task._detection_tracks
        assert task._detection_tracks["src-1"] is None

    def test_the_ring_outreaches_the_sources_rewind_window(self):
        """Pre-roll is cut from the buffer, so a ring shorter than the retention would
        leave the lead-in — the approach — unannotated."""
        pipeline = _pipeline("src-1", buffered=True)
        pipeline.retention_s = 900.0
        task = self._task([pipeline], [self._clip_action()])

        track = task._detection_track_for(pipeline)

        assert track._max_seconds >= 900.0

    def test_pushing_a_frame_is_a_no_op_without_a_ring(self):
        """The push happens on every analysed frame of every source, wanted or not."""
        pipeline = _pipeline("src-1", buffered=True)
        task = self._task([pipeline], [])

        task._push_detection_track(pipeline, 100.0, (640, 360))

        assert task._detection_tracks["src-1"] is None

    def test_a_pushed_frame_is_timestamped_in_wallclock(self):
        """The ring is sliced by a clip window, which is addressed in wallclock; the
        frame loop's clock is monotonic. Storing the monotonic reading would put every
        sample decades away from its clip."""
        pipeline = _pipeline("src-1", buffered=True)
        task = self._task([pipeline], [self._clip_action()])

        task._push_detection_track(pipeline, 100.0, (640, 360))

        track = task._detection_tracks["src-1"]
        assert track.samples_between(0, 1e12)[0].at == pytest.approx(task._wallclock_offset + 100.0)


class TestRecordClipActionResolver:
    async def test_the_resolver_wins_over_a_statically_injected_backend(self):
        """Both are accepted so existing callers keep working; the resolver is preferred."""
        static = PlaybackClipBackend(object())
        chosen = RingClipBackend(FrameRing())
        action = RecordClipAction(backend=static, backend_for=lambda ctx: chosen)

        assert action._resolve_backend(_context()) is chosen

    async def test_a_resolver_returning_none_is_reported_not_raised(self):
        from apps.helpers.watch.events import EventPhase

        action = RecordClipAction(backend_for=lambda ctx: None)
        signal = SimpleNamespace(
            phase=EventPhase.END,
            rule_name="r",
            started_at=1.0,
            at=2.0,
            duration_s=1.0,
            truncated=False,
            to_dict=lambda: {},
        )

        result = await action.run(signal, _context())

        assert result.ok is False
        assert "No clip backend" in result.detail


class TestRingPrefersTheCamerasOwnFrame:
    """A ring clip should be the camera's full-resolution frame where one exists.

    The analysis frame is downscaled to `analysis_width`, so ringing it would cap a
    snapshot camera's clips at 640 px for no reason: the camera's own JPEG was already
    downloaded, and using it avoids an encode rather than costing one.
    """

    def test_the_full_resolution_original_wins_over_the_analysis_frame(self):
        pipeline = _pipeline("src-1")
        task = _task([pipeline])

        task._push_to_ring(
            pipeline,
            _packet(b"\xff\xd8downscaled"),
            now=100.0,
            full_res=_packet(b"\xff\xd8fullsize"),
        )

        assert task._frame_rings["src-1"]._frames[0][1] == b"\xff\xd8fullsize"

    def test_it_falls_back_to_the_analysis_frame_when_there_is_no_original(self):
        """Not every provider hands one over; the ring must still fill."""
        pipeline = _pipeline("src-1")
        task = _task([pipeline])

        task._push_to_ring(pipeline, _packet(b"\xff\xd8downscaled"), now=100.0, full_res=None)

        assert task._frame_rings["src-1"]._frames[0][1] == b"\xff\xd8downscaled"


class _RecordingRelay:
    """Relay double that writes a file, so a drained chunk looks real on disk."""

    def __init__(self):
        self.calls = []

    async def extract_clip(self, path, start, duration_s, dest):
        self.calls.append((path, start, duration_s, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mp4")
        return {"path": str(dest), "sizeBytes": 3}


def _drain_task(pipelines, tmp_path, relay=None, actions=()):
    task = _task(pipelines, relay_client=relay or _RecordingRelay())
    task._clip_drains = {}
    task._actions = list(actions)
    task.CHUNK_DIRNAME = ".drain"
    task._chunk_root = lambda: tmp_path
    return task


def _open_event_pipeline(source_uuid="src-1", started_at=100.0, **kwargs):
    """A pipeline whose engine reports one open event, without driving a real rule."""
    pipeline = _pipeline(source_uuid, **kwargs)
    pipeline.provider = SimpleNamespace(_relayed_name="relay_master_abc")
    pipeline.engine.open_events = lambda: [{"ruleUuid": "r1", "startedAt": started_at, "sourceUuid": source_uuid}]
    return pipeline


class TestClipDrainIsDriven:
    """The drain is what lets `max_duration_s` exceed the rewind window.

    Wiring it wrong fails silently in the worst way: long events keep producing short
    clips exactly as they did before, with no error anywhere.
    """

    async def test_an_idle_source_never_allocates_a_drain(self, tmp_path):
        """Most events close inside the buffer; they must not touch this path at all."""
        pipeline = _pipeline("src-1", buffered=True)
        pipeline.retention_s = 45.0
        task = _drain_task([pipeline], tmp_path)

        await task._service_drain(pipeline)

        assert task._clip_drains == {}

    async def test_an_open_event_arms_a_drain_on_a_buffered_source(self, tmp_path):
        pipeline = _open_event_pipeline(buffered=True)
        pipeline.retention_s = 45.0
        task = _drain_task([pipeline], tmp_path)

        await task._service_drain(pipeline)

        assert "src-1" in task._clip_drains
        assert task._clip_drains["src-1"].active is True

    async def test_an_unbuffered_source_gets_no_drain(self, tmp_path):
        """A ring-backed source has no timeline to drain — its history is the RAM ring."""
        pipeline = _open_event_pipeline(buffered=False)
        task = _drain_task([pipeline], tmp_path)

        await task._service_drain(pipeline)

        assert task._clip_drains == {}

    async def test_the_drain_starts_ahead_of_the_event_by_the_largest_preroll(self, tmp_path):
        """Pre-roll sits in front of the event and cannot be re-derived once evicted."""
        pipeline = _open_event_pipeline(buffered=True)
        pipeline.retention_s = 45.0
        rows = [
            (SimpleNamespace(kind="record_clip", source_uuids_json=[]), RecordClipAction(preroll_s=10)),
            (SimpleNamespace(kind="record_clip", source_uuids_json=[]), RecordClipAction(preroll_s=30)),
        ]
        task = _drain_task([pipeline], tmp_path, actions=rows)

        await task._service_drain(pipeline)

        assert task._clip_drains["src-1"].lead_s == 30.0

    async def test_a_clip_action_scoped_to_another_source_does_not_set_the_lead(self, tmp_path):
        pipeline = _open_event_pipeline(buffered=True)
        pipeline.retention_s = 45.0
        rows = [(SimpleNamespace(kind="record_clip", source_uuids_json=["other"]), RecordClipAction(preroll_s=30))]
        task = _drain_task([pipeline], tmp_path, actions=rows)

        await task._service_drain(pipeline)

        assert task._clip_drains["src-1"].lead_s == 0.0

    async def test_the_event_closing_discards_the_chunks(self, tmp_path):
        """Chunks are full-rate video; a drain that outlived its event is a disk leak."""
        pipeline = _open_event_pipeline(buffered=True)
        pipeline.retention_s = 45.0
        task = _drain_task([pipeline], tmp_path)
        await task._service_drain(pipeline)

        pipeline.engine.open_events = lambda: []
        await task._service_drain(pipeline)

        assert task._clip_drains["src-1"].active is False

    async def test_chunks_are_offered_to_the_action_for_the_events_window(self, tmp_path):
        pipeline = _open_event_pipeline(buffered=True, started_at=0.0)
        pipeline.retention_s = 45.0
        task = _drain_task([pipeline], tmp_path)
        drain = task._drain_for(pipeline)
        drain.note_open(_dt(task, 0.0))
        await drain.drain(_dt(task, 40.0))

        signal = SimpleNamespace(source_uuid="src-1", started_at=0.0, at=60.0)
        assert len(task._chunks_for_signal(signal)) == 1

    async def test_no_drain_means_no_chunks(self, tmp_path):
        """The ordinary single-cut path must stay reachable."""
        pipeline = _pipeline("src-1", buffered=True)
        task = _drain_task([pipeline], tmp_path)
        signal = SimpleNamespace(source_uuid="src-1", started_at=0.0, at=60.0)
        assert task._chunks_for_signal(signal) == []


def _dt(task, monotonic_s):
    from datetime import datetime, timezone

    return datetime.fromtimestamp(monotonic_s + task._wallclock_offset, tz=timezone.utc)


def _solid_jpeg(fill: int) -> bytes:
    frame = np.full((4, 4, 3), fill, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def _engine_pipeline(trigger: str, source_uuid: str = "src-1") -> WatchPipeline:
    engine = WatchEngine(
        rules=[
            RuleSpec(
                uuid="rule-1",
                name="cat present",
                subject_a_kind="label",
                subject_a_refs=["label-cat"],
                trigger=trigger,
            )
        ],
        zones=[],
        source_uuid=source_uuid,
    )
    return WatchPipeline(
        source_uuid=source_uuid,
        source_name="cam",
        provider=SimpleNamespace(),
        tracker=SimpleNamespace(update=lambda dets, now: []),
        engine=engine,
        buffered=False,
        buffer_mode="auto",
        ring_seconds=20.0,
        ring_max_mb=32.0,
    )


def _start_signal(rule_uuid="rule-1", source_uuid="src-1", started_at=60.0):
    return WatchEventSignal(
        rule_uuid=rule_uuid,
        rule_name="cat left the scene",
        phase=EventPhase.START,
        at=started_at,
        started_at=started_at,
        source_uuid=source_uuid,
        track_ids=[1],
    )


class TestCardArtContextForOnExit:
    """The bug this fixes: an on_exit ("left the scene") event's still-image preview
    showed the empty room. `_hold_event` encodes card art from the pipeline's *live*
    frame at dispatch time, but on_exit dispatches `rearm_delay_s` after the subject
    actually left (see TestImpulseTriggers in test_watch_engine.py) — so the live
    frame is of the now-empty scene, not the departure. `_card_art_context` is what
    swaps it for a frame near the true departure moment, and only for on_exit.
    """

    async def test_on_exit_swaps_the_live_frame_for_the_departure_frame(self):
        pipeline = _engine_pipeline("on_exit")
        task = _task([pipeline])
        task._wallclock_offset = 0.0
        ring = FrameRing(max_seconds=30.0)
        ring.push(60.0, _solid_jpeg(10))  # the departure moment
        ring.push(70.0, _solid_jpeg(200))  # dispatch time: the room is now empty
        task._frame_rings["src-1"] = ring

        live_frame = np.full((4, 4, 3), 200, dtype=np.uint8)
        context = ActionContext(
            profile_uuid="p", profile_name="P", source_uuid="src-1", frame=live_frame, full_frame=live_frame
        )

        result = await task._card_art_context(_start_signal(started_at=60.0), context, pipeline)

        assert result is not context
        assert not np.array_equal(result.frame, live_frame)
        assert int(result.frame[0, 0, 0]) < 50  # the departure frame, not the empty one

    async def test_while_trigger_keeps_the_live_frame(self):
        """A while rule's dwell leaves the same gap between started_at and dispatch —
        confirming the swap is gated on the rule's trigger, not on that gap."""
        pipeline = _engine_pipeline("while")
        task = _task([pipeline])
        task._wallclock_offset = 0.0
        ring = FrameRing(max_seconds=30.0)
        ring.push(60.0, _solid_jpeg(10))
        task._frame_rings["src-1"] = ring

        live_frame = np.full((4, 4, 3), 200, dtype=np.uint8)
        context = ActionContext(
            profile_uuid="p", profile_name="P", source_uuid="src-1", frame=live_frame, full_frame=live_frame
        )

        result = await task._card_art_context(_start_signal(started_at=60.0), context, pipeline)

        assert result is context

    async def test_no_pipeline_leaves_context_untouched(self):
        task = _task([])
        context = ActionContext(profile_uuid="p", profile_name="P")

        result = await task._card_art_context(_start_signal(), context, None)

        assert result is context

    async def test_nothing_in_the_ring_falls_back_to_the_live_frame(self):
        """A ring too young to reach the departure (e.g. a just-armed watcher) must not
        leave the card art blank — the live frame, empty as it is, still beats nothing."""
        pipeline = _engine_pipeline("on_exit")
        task = _task([pipeline])
        task._wallclock_offset = 0.0
        task._frame_rings["src-1"] = FrameRing(max_seconds=30.0)

        live_frame = np.full((4, 4, 3), 200, dtype=np.uint8)
        context = ActionContext(
            profile_uuid="p", profile_name="P", source_uuid="src-1", frame=live_frame, full_frame=live_frame
        )

        result = await task._card_art_context(_start_signal(started_at=60.0), context, pipeline)

        assert result is context

    async def test_buffered_source_pulls_a_still_from_the_relay(self):
        """The other rewind tier: no frame ring at all, so the departure frame has to
        come from a short cut of the relay's own recorded segments instead."""
        pipeline = _engine_pipeline("on_exit")
        pipeline.buffered = True
        pipeline.provider = SimpleNamespace(_relayed_name="cam/relay")

        calls = []

        class _Relay:
            async def extract_clip(self, *, path, start, duration_s, dest):
                calls.append((path, start, duration_s))
                dest.write_bytes(b"stand-in mp4 bytes; _first_frame is stubbed below")
                return {"partialPreroll": False}

        task = _task([pipeline], relay_client=_Relay())
        task._wallclock_offset = 0.0
        historical = np.full((4, 4, 3), 10, dtype=np.uint8)
        task._first_frame = lambda path: historical

        live_frame = np.full((4, 4, 3), 200, dtype=np.uint8)
        context = ActionContext(
            profile_uuid="p", profile_name="P", source_uuid="src-1", frame=live_frame, full_frame=live_frame
        )

        result = await task._card_art_context(_start_signal(started_at=60.0), context, pipeline)

        assert calls and calls[0][0] == "cam/relay" and calls[0][2] == 1.0
        assert np.array_equal(result.frame, historical)

    async def test_a_relay_extraction_failure_falls_back_to_the_live_frame(self):
        pipeline = _engine_pipeline("on_exit")
        pipeline.buffered = True
        pipeline.provider = SimpleNamespace(_relayed_name="cam/relay")

        class _FailingRelay:
            async def extract_clip(self, **kwargs):
                raise RuntimeError("buffer aged out")

        task = _task([pipeline], relay_client=_FailingRelay())
        task._wallclock_offset = 0.0

        live_frame = np.full((4, 4, 3), 200, dtype=np.uint8)
        context = ActionContext(
            profile_uuid="p", profile_name="P", source_uuid="src-1", frame=live_frame, full_frame=live_frame
        )

        result = await task._card_art_context(_start_signal(started_at=60.0), context, pipeline)

        assert result is context


class TestARingClipIsAsLongAsTheEventItRecords:
    """A ring clip's playback length must match the span of frames it holds.

    A snapshot camera delivering a frame every three seconds is 0.32 fps, and the
    encoder floored its derived rate at 1.0 — so an 88-second visit was written as a
    28-second file played at 3x, while the recordings row, the sidecar and the overlay
    all went on believing it was 88 seconds long. Seeking past the real end of a video
    does not fail: the playhead clamps to the final frame. A correction harvesting eight
    moments across the event therefore captured the same last picture seven times, each
    saved with the corrected boxes of a moment the file did not contain.

    These are integration tests against the real encoder, because the bug was in what
    ffmpeg was told, not in the arithmetic around it.
    """

    @staticmethod
    def _frames(count: int, spacing_s: float, start: float = 1000.0):
        return [(start + i * spacing_s, _solid_jpeg(i * 7 % 255)) for i in range(count)]

    @staticmethod
    def _duration(path) -> float:
        import subprocess

        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(out.stdout.strip())

    @pytest.mark.parametrize("spacing_s", [3.2, 1.0, 0.25])
    def test_playback_matches_the_frames_wall_span(self, tmp_path, spacing_s):
        """Sub-1 fps included — that is the rate real snapshot cameras deliver."""
        frames = self._frames(28, spacing_s)
        wall_span = frames[-1][0] - frames[0][0]
        dest = tmp_path / f"ring-{spacing_s}.mp4"

        encode_timeline([(t, _spill(tmp_path, i, blob)) for i, (t, blob) in enumerate(frames)], dest)

        played = self._duration(dest)
        assert played == pytest.approx(
            wall_span, rel=0.1
        ), f"{spacing_s}s spacing: {wall_span:.1f}s of frames became a {played:.1f}s file"

    async def test_the_reported_duration_describes_the_file_not_the_frames(self, tmp_path, monkeypatch):
        """Whatever the encoder does, `durationS` is what a player can seek within.

        Measured from the written file rather than assumed from the frames, because a
        clip that claims to be longer than it is sends every seek past its end — which
        is how a harvest of eight moments came back as the same picture seven times.
        """
        ring = FrameRing()
        lead = self._frames(6, 6.0)
        dense = self._frames(20, 0.5, start=lead[-1][0] + 6.0)
        monkeypatch.setattr(ring, "frames_between", lambda *a, **k: list(dense))
        monkeypatch.setattr(ring, "lead_in_frames", lambda **k: list(lead))
        backend = RingClipBackend(ring)
        dest = tmp_path / "lead-in.mp4"

        result = await backend.extract(
            _context("snap"), datetime.fromtimestamp(lead[0][0], tz=timezone.utc), 120.0, dest
        )

        played = self._duration(dest)
        assert result["durationS"] == pytest.approx(played, rel=0.05), result
        # The lead-in still is clamped to the window start, so the span the clip covers
        # is the window itself rather than the age of the oldest sticky still.
        assert result["sourceSpanS"] == pytest.approx(dense[-1][0] - lead[0][0], rel=0.01)
        # And the video is that span — no longer compressed against it, which is what
        # made boxes after the still seam point at the wrong moment.
        assert result["durationS"] == pytest.approx(result["sourceSpanS"], abs=LAST_FRAME_HOLD_S + 0.1)


class TestOverlappingEventsDoNotEvictEachOthersFrames:
    """One frame ring serves every rule watching a source, and the engine runs a state
    machine per `(rule, subject, target)` — so several events are routinely open over one
    camera at once. Two cats in the yard is the ordinary case, not the exotic one.

    The hold protecting open events' frames from age-based eviction used to be taken and
    released on the event edges. A second event's START overwrote the first's floor, and
    whichever event closed first released the hold for *all* of them: the next push
    evicted everything older than `max_seconds`, so a neighbour still recording had its
    clip cut from a ring holding only the last few seconds. An 88-second visit produced
    26 seconds of video, the rest replaced by two stills.

    The floor is now re-derived from the engine's open events once a pass, which also
    means a missed END cannot leak it and pin the ring in RAM.
    """

    @staticmethod
    def _fill(ring: FrameRing, start: float, count: int, step: float = 0.5) -> float:
        for i in range(count):
            ring.push(start + i * step, _solid_jpeg(i % 255))
        return start + (count - 1) * step

    @staticmethod
    def _task_with_ring(ring, open_events, *, ring_seconds=20.0, preroll=5.0):
        pipeline = _pipeline("src-1", ring_seconds=ring_seconds)
        pipeline.engine.open_events = lambda: list(open_events)
        task = _task([pipeline])
        task._frame_rings["src-1"] = ring
        task._wallclock_offset = 0.0
        task._max_preroll_for = lambda _uuid: preroll
        return task, pipeline

    def test_the_floor_follows_the_earliest_open_event(self):
        ring = FrameRing(max_seconds=20.0, max_mb=32.0)
        # Two events, one per cat, both matched by the same rule.
        events = [
            {"startedAt": 100.0, "ruleUuid": "r1", "subjectRef": "cat-a"},
            {"startedAt": 140.0, "ruleUuid": "r1", "subjectRef": "cat-b"},
        ]
        task, pipeline = self._task_with_ring(ring, events)
        # Serviced once a pass, so the hold is in place while the frames arrive — not
        # applied retroactively to frames the age cap has already dropped.
        task._service_ring_hold(pipeline)
        last = self._fill(ring, 100.0, 180)
        task._service_ring_hold(pipeline)
        ring.push(last + 0.5, _solid_jpeg(1))

        held = ring.frames_between(0.0, last + 1.0)
        assert held[0][0] <= 100.5, (
            f"the earliest open event needs its own pre-roll; earliest held was "
            f"{held[0][0]} and the age cap alone would have left {last + 0.5 - 20.0}"
        )
        assert len(held) > 170, f"only {len(held)} frames survived"

    def test_a_closing_event_keeps_its_neighbours_frames(self):
        """The bug, stated directly: the survivor's frames must outlive its neighbour."""
        ring = FrameRing(max_seconds=20.0, max_mb=32.0)
        open_events = [
            {"startedAt": 100.0, "ruleUuid": "r1", "subjectRef": "cat-a"},
            {"startedAt": 140.0, "ruleUuid": "r1", "subjectRef": "cat-b"},
        ]
        task, pipeline = self._task_with_ring(ring, open_events)
        task._service_ring_hold(pipeline)
        last = self._fill(ring, 100.0, 180)

        # The first cat leaves; the second is still in frame.
        open_events.pop(0)
        task._service_ring_hold(pipeline)
        ring.push(last + 0.5, _solid_jpeg(1))

        held = ring.frames_between(0.0, last + 1.0)
        assert held[0][0] <= 135.5, f"the surviving event still needs its pre-roll; earliest held was {held[0][0]}"
        # Frames older than the survivor's floor are correctly gone — the hold is the
        # open event's, not the departed one's.
        assert held[0][0] >= 135.0, held[0][0]

    def test_the_last_event_closing_resumes_normal_eviction(self):
        """The hold is a reprieve, not a leak: once nothing is open the ring goes back to
        its age cap, or a busy camera would keep every frame it ever saw."""
        ring = FrameRing(max_seconds=20.0, max_mb=32.0)
        open_events = [{"startedAt": 100.0, "ruleUuid": "r1", "subjectRef": "cat-a"}]
        task, pipeline = self._task_with_ring(ring, open_events)
        task._service_ring_hold(pipeline)
        last = self._fill(ring, 100.0, 180)

        open_events.clear()
        task._service_ring_hold(pipeline)

        held = ring.frames_between(0.0, last + 1.0)
        assert held[0][0] >= last - 20.0, "age eviction must resume once nothing is open"

    def test_a_source_with_no_ring_is_left_alone(self):
        """Buffered cameras cut from the relay and never allocate a ring; servicing one
        must not invent state for them."""
        pipeline = _pipeline("src-1", buffered=True)
        pipeline.engine.open_events = lambda: [{"startedAt": 100.0}]
        task = _task([pipeline])
        task._max_preroll_for = lambda _uuid: 5.0

        task._service_ring_hold(pipeline)

        assert task._frame_rings == {}


class TestATruncatedHeadIsReported:
    """The ring's byte cap evicts under the hold, so a visit longer than the ring can
    afford loses its head however long `ring_seconds` is.

    That stretch is not missing from the clip, which is worse: `lead_in_frames` fills it
    with a couple of held stills, so it plays as footage in which nothing moves. The old
    `partialPreroll` was computed across the spliced stills *and* the dense frames, so
    the presence of a still made it read "complete" in exactly the case a reviewer needs
    told about.
    """

    @staticmethod
    def _ring_with(lead, dense):
        ring = FrameRing()
        ring.frames_between = lambda *a, **k: list(dense)
        ring.lead_in_frames = lambda **k: list(lead)
        return RingClipBackend(ring)

    async def test_stills_standing_in_for_evicted_video_are_reported(self, tmp_path):
        t0 = 1000.0
        # The event ran from t0, but the ring only still holds its last 20 seconds.
        lead = [(t0 + i * 6.0, _solid_jpeg(i)) for i in range(2)]
        dense = [(t0 + 60.0 + i * 0.4, _solid_jpeg(100 + i)) for i in range(50)]
        backend = self._ring_with(lead, dense)

        result = await backend.extract(
            _context("snap"), datetime.fromtimestamp(t0, tz=timezone.utc), 120.0, tmp_path / "c.mp4"
        )

        assert result["stillsOnlyS"] == pytest.approx(60.0, abs=1.0), result
        assert result["partialPreroll"] is True, "a minute of stills is not a complete clip"

    async def test_a_late_start_with_no_stills_is_not_reported_as_stills(self, tmp_path):
        """The player's warning reads "the first N seconds are stills, not video". With no
        sticky still to splice — none held, or the newest one past `LEAD_IN_MAX_AGE_S` —
        the clip simply begins late and holds no stills at all, so the figure has to be
        zero or the warning describes footage the file does not contain. The gap itself is
        still reported, through `partialPreroll`."""
        t0 = 1000.0
        dense = [(t0 + 60.0 + i * 0.4, _solid_jpeg(i)) for i in range(50)]
        backend = self._ring_with([], dense)

        result = await backend.extract(
            _context("snap"), datetime.fromtimestamp(t0, tz=timezone.utc), 120.0, tmp_path / "c.mp4"
        )

        assert result["leadInFrames"] == 0
        assert result["stillsOnlyS"] == 0.0, result
        assert result["partialPreroll"] is True, "a clip starting a minute late is still partial"

    async def test_a_clip_the_ring_fully_covered_is_not_flagged(self):
        """The negative half: flagging every ring clip would make the warning noise."""
        t0 = 1000.0
        dense = [(t0 + i * 0.4, _solid_jpeg(i)) for i in range(50)]
        backend = self._ring_with([], dense)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            result = await backend.extract(
                _context("snap"), datetime.fromtimestamp(t0, tz=timezone.utc), 20.0, Path(tmp) / "c.mp4"
            )

        assert result["stillsOnlyS"] == pytest.approx(0.0, abs=0.5), result
        assert result["partialPreroll"] is False


class TestAnOpenEventSpoolsToDisk:
    """Why a clip's length is a question about disk, not RAM.

    The ring exists to answer a retroactive question — "what did the seconds before this
    event look like?" — so it must hold frames nobody has asked for, which is why it is
    bounded and why it evicts. Once an event opens that justification is gone: every
    frame from then on is known to be wanted. Before spooling, the byte cap evicted
    those frames *while the event was still recording*, and the clip was cut from
    whatever survived: an 88-second visit whose video was 26 seconds, the hole papered
    over with two held stills.

    The pre-roll must survive the change. It is the one thing spooling cannot
    reconstruct, so opening an event seeds the spool from the RAM the ring already has.
    """

    @staticmethod
    def _ring(tmp_path, **kwargs):
        return FrameRing(spool=FrameSpool(Path(tmp_path) / "spool"), **kwargs)

    def test_frames_outlive_the_byte_cap_once_an_event_opens(self, tmp_path):
        # A cap far too small to hold the event, which is the case that used to truncate.
        ring = self._ring(tmp_path, max_seconds=20.0, max_mb=0.05)
        ring.hold_from(100.0)

        for i in range(200):
            ring.push(100.0 + i * 0.4, _solid_jpeg(i % 255))

        assert ring.frame_count < 200, "the RAM ring is still bounded — that is the point"
        assert ring._spool.frame_count == 200, "every frame of an open event must reach disk"

    def test_opening_an_event_keeps_the_preroll_already_in_ram(self, tmp_path):
        """Retroactive pre-roll is exactly what a spool cannot go back and capture."""
        ring = self._ring(tmp_path, max_seconds=20.0)
        ring.push(95.0, _solid_jpeg(1))  # before anything was open
        ring.push(97.0, _solid_jpeg(2))

        ring.hold_from(96.0)  # an event opens, asking for pre-roll from 96s

        spooled = [t for t, _ in ring._spool.frames_between(0.0, 200.0)]
        assert 97.0 in spooled, "pre-roll inside the requested window must be kept"
        assert 95.0 not in spooled, "frames older than the window are not this event's"

    def test_the_spool_is_dropped_once_nothing_is_open(self, tmp_path):
        """A spool that outlived its event would fill the disk a watcher runs on."""
        ring = self._ring(tmp_path, max_seconds=20.0)
        ring.hold_from(100.0)
        for i in range(10):
            ring.push(100.0 + i * 0.4, _solid_jpeg(i))
        spool_dir = ring._spool._dir
        assert spool_dir is not None and spool_dir.exists()

        ring.hold_from(None)

        assert not spool_dir.exists(), "the spooled frames must go with the event"
        assert ring._spool.frame_count == 0

    def test_a_ring_with_no_spool_still_works(self, tmp_path):
        """The spool is optional: a ring built without one behaves exactly as before."""
        ring = FrameRing(max_seconds=20.0)

        ring.hold_from(100.0)
        ring.push(100.0, _solid_jpeg(1))
        ring.hold_from(None)

        assert ring.frame_count == 1

    async def test_the_clip_is_cut_from_the_spool_not_the_surviving_ram(self, tmp_path):
        """End to end, against real ffmpeg: the finished video covers the whole event."""
        ring = self._ring(tmp_path, max_seconds=20.0, max_mb=0.05)
        t0 = 1000.0
        ring.push(t0, _solid_jpeg(0))
        ring.hold_from(t0)
        for i in range(1, 200):
            ring.push(t0 + i * 0.4, _solid_jpeg(i % 255))
        wall_span = 199 * 0.4

        result = await RingClipBackend(ring).extract(
            _context("snap"), datetime.fromtimestamp(t0, tz=timezone.utc), 120.0, tmp_path / "out.mp4"
        )

        assert result["durationS"] == pytest.approx(
            wall_span, rel=0.1
        ), f"{wall_span:.1f}s of event became a {result['durationS']:.1f}s clip"
        assert result["stillsOnlyS"] == pytest.approx(0.0, abs=0.5)
        assert result["partialPreroll"] is False


class TestVideoTimeEqualsEventTime:
    """The contract every consumer of a clip assumes: a box the sidecar records at t=41s
    is at t=41s in the video.

    It was broken in two places at once. Each segment was encoded at one constant rate,
    so a 0.32fps camera played back at 1fps — three times too fast. And a spliced still
    lead-in was encoded at a *different* constant rate and stream-copied on, so the two
    halves ran on different clocks: 61s of wall time became 2s of video, and every box
    after the seam pointed at the wrong moment.
    """

    async def test_the_seam_after_a_still_lead_in_does_not_shift_the_timeline(self, tmp_path):
        t0 = 1000.0
        ring = FrameRing(max_seconds=20.0)
        # A sticky still from well before the window — but inside `LEAD_IN_MAX_AGE_S`,
        # since a still older than that is refused outright and there would be no seam
        # left to measure. Dense frames start 40s into the window.
        ring.push(t0 - 120.0, _solid_jpeg(5), idle=True)
        for i in range(40):
            ring.push(t0 + 40.0 + i * 0.4, _solid_jpeg(i % 255))

        result = await RingClipBackend(ring).extract(
            _context("snap"), datetime.fromtimestamp(t0, tz=timezone.utc), 120.0, tmp_path / "seam.mp4"
        )

        assert result["leadInFrames"] == 1, "one still, not a parade of them"
        # The clip starts where it was asked to start, not ten minutes earlier.
        started = datetime.fromisoformat(result["startedAt"]).timestamp()
        assert started == pytest.approx(t0, abs=0.5), result["startedAt"]
        # And it runs the full requested span: 40s of still plus ~16s of dense frames.
        # A capped or re-rated lead-in would land far short of that.
        assert result["durationS"] == pytest.approx(40.0 + 39 * 0.4, rel=0.1), result["durationS"]
        # The gap is reported rather than presented as footage.
        assert result["stillsOnlyS"] == pytest.approx(40.0, abs=1.0)
        assert result["partialPreroll"] is True

    async def test_a_dense_clip_runs_at_wall_speed(self, tmp_path):
        """No stills involved: the plain case must be exact, not merely close."""
        t0 = 1000.0
        ring = FrameRing(max_seconds=200.0, max_mb=64.0)
        for i in range(100):
            ring.push(t0 + i * 0.4, _solid_jpeg(i % 255))

        result = await RingClipBackend(ring).extract(
            _context("snap"), datetime.fromtimestamp(t0, tz=timezone.utc), 120.0, tmp_path / "dense.mp4"
        )

        assert result["durationS"] == pytest.approx(99 * 0.4, rel=0.05), result["durationS"]
        assert result["sourceSpanS"] == pytest.approx(99 * 0.4, rel=0.01)


def _sized_jpeg(fill: int, width: int, height: int) -> bytes:
    frame = np.full((height, width, 3), fill, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


class TestAMixedResolutionTimeline:
    """One clip's frames are not all one size, and ffmpeg does not say so.

    A Ring source's sticky idle snapshot and its live frames come off different tiers at
    different resolutions and are spliced into the same timeline. ffmpeg configures the
    filter graph from the *first* frame and silently stretches every later one into it,
    so a snapshot leading the clip decided the geometry of the whole event body — a
    640x480 clip of 1920x1080 footage, squashed from 16:9 to 4:3, no error and no warning.

    Integration tests against the real encoder, because the bug is in what ffmpeg is told.
    """

    @staticmethod
    def _geometry(path) -> tuple:
        import subprocess

        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        width, height = out.stdout.strip().split(",")
        return int(width), int(height)

    @pytest.mark.parametrize(
        "lead_size, dense_size",
        [
            ((64, 48), (128, 72)),  # a smaller 4:3 snapshot in front of 16:9 live frames
            ((128, 72), (64, 48)),  # and the other way round: the snapshot is the larger
        ],
    )
    def test_the_event_body_decides_the_geometry_not_the_frame_that_happens_to_be_first(
        self, tmp_path, lead_size, dense_size
    ):
        """State-dependent by construction: the answer has to follow the dense frames in
        both directions, so it cannot be satisfied by always taking the first frame or
        always taking the larger one."""
        timeline = [(1000.0, _spill(tmp_path, 0, _sized_jpeg(30, *lead_size)))]
        timeline += [(1001.0 + i * 0.4, _spill(tmp_path, i + 1, _sized_jpeg(i % 255, *dense_size))) for i in range(12)]
        dest = tmp_path / "mixed.mp4"

        encode_timeline(timeline, dest)

        assert self._geometry(dest) == dense_size, (
            f"the 12 dense frames are {dense_size}; a clip encoded at "
            f"{self._geometry(dest)} has stretched the event body into the lead-in still"
        )

    def test_an_odd_sized_frame_is_still_padded_to_even(self, tmp_path):
        """yuv420p cannot represent odd dimensions, and a snapshot camera is under no
        obligation to deliver even ones. The canvas has to round up rather than hand
        ffmpeg a size it will refuse."""
        timeline = [(1000.0 + i * 0.4, _spill(tmp_path, i, _sized_jpeg(i % 255, 65, 49))) for i in range(4)]
        dest = tmp_path / "odd.mp4"

        encode_timeline(timeline, dest)

        assert self._geometry(dest) == (66, 50)


class TestTheSpoolStopsAtTheLongestOpenClip:
    """Spooling is bounded by what a clip can still read, not by how long a rule stays open.

    A clip's window ends at `min(event duration + roll, max_length_s)`, so frames past the
    last open event's cap are written to disk, counted against the recordings size cap,
    and then read by nothing. A rule that stays open for hours — a lit driveway, a rule
    whose exit condition never fires — turned that into gigabytes of jpgs whose only
    effect was to evict real event clips.
    """

    @staticmethod
    def _task_with_ring(ring, open_events, *, preroll=5.0, max_length=60.0):
        pipeline = _pipeline("src-1")
        pipeline.engine.open_events = lambda: list(open_events)
        task = _task([pipeline])
        task._frame_rings["src-1"] = ring
        task._wallclock_offset = 0.0
        task._max_preroll_for = lambda _uuid: preroll
        task._max_clip_length_for = lambda _uuid: max_length
        return task, pipeline

    def test_frames_past_the_cap_are_not_spooled(self, tmp_path):
        ring = FrameRing(max_seconds=20.0, spool=FrameSpool(Path(tmp_path) / "spool"))
        events = [{"startedAt": 100.0, "ruleUuid": "r1", "subjectRef": "cat-a"}]
        task, pipeline = self._task_with_ring(ring, events, preroll=5.0, max_length=60.0)
        task._service_ring_hold(pipeline)

        # The window runs 95s -> 155s; push well past it, as an event that never closes does.
        for i in range(600):
            ring.push(100.0 + i * 0.5, _solid_jpeg(i % 255))

        spooled = [t for t, _ in ring._spool.frames_between(0.0, 1e9)]
        assert spooled, "the event's own window must still be spooled"
        assert max(spooled) <= 160.0, f"spooled up to {max(spooled)}s for a window ending at 155s"
        assert ring._spool.dropped_count > 0, "the frames past the cap must be refused, not merely unread"

    def test_the_latest_open_event_sets_the_cap_not_the_earliest(self, tmp_path):
        """One ring serves every rule watching a camera, and the event that opened *last*
        is the one whose window reaches furthest forward. Sizing the bound from the
        earliest event would stop spooling while a neighbour was still recording."""
        ring = FrameRing(max_seconds=20.0, spool=FrameSpool(Path(tmp_path) / "spool"))
        open_events = [{"startedAt": 100.0, "ruleUuid": "r1", "subjectRef": "cat-a"}]
        task, pipeline = self._task_with_ring(ring, open_events, preroll=5.0, max_length=60.0)
        task._service_ring_hold(pipeline)

        # A second cat arrives 100s in; its own window runs to 255s, long past the first's.
        open_events.append({"startedAt": 200.0, "ruleUuid": "r1", "subjectRef": "cat-b"})
        task._service_ring_hold(pipeline)
        for i in range(500):
            ring.push(100.0 + i * 0.5, _solid_jpeg(i % 255))

        spooled = [t for t, _ in ring._spool.frames_between(0.0, 1e9)]
        assert max(spooled) > 250.0, (
            f"the second event records until 255s; spooling stopped at {max(spooled)}s, "
            "so its clip would be cut short"
        )

    def test_no_configured_clip_length_leaves_the_spool_unbounded(self, tmp_path):
        """A source with no clip action has no cap to derive, and inventing one would
        silently truncate whatever else reads the spool."""
        ring = FrameRing(max_seconds=20.0, spool=FrameSpool(Path(tmp_path) / "spool"))
        events = [{"startedAt": 100.0, "ruleUuid": "r1", "subjectRef": "cat-a"}]
        task, pipeline = self._task_with_ring(ring, events, max_length=0.0)
        task._service_ring_hold(pipeline)

        for i in range(400):
            ring.push(100.0 + i * 0.5, _solid_jpeg(i % 255))

        assert ring._spool.frame_count == 400
        assert ring._spool.dropped_count == 0

    def test_the_cap_is_forgotten_with_the_event(self, tmp_path):
        """A deadline that outlived its event would refuse the next event's frames."""
        ring = FrameRing(max_seconds=20.0, spool=FrameSpool(Path(tmp_path) / "spool"))
        events = [{"startedAt": 100.0, "ruleUuid": "r1", "subjectRef": "cat-a"}]
        task, pipeline = self._task_with_ring(ring, events, max_length=60.0)
        task._service_ring_hold(pipeline)
        for i in range(600):
            ring.push(100.0 + i * 0.5, _solid_jpeg(i % 255))
        assert ring._spool.dropped_count > 0

        events.clear()
        task._service_ring_hold(pipeline)
        events.append({"startedAt": 900.0, "ruleUuid": "r1", "subjectRef": "cat-b"})
        task._service_ring_hold(pipeline)
        ring.push(900.0, _solid_jpeg(1))

        assert ring._spool.dropped_count == 0, "the previous event's deadline must not bind the next one"
        assert ring._spool.frame_count == 1
