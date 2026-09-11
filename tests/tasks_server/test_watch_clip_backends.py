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
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from apps.helpers.actions.action import ActionContext
from apps.helpers.actions.recordClipAction import (
    FrameRing,
    PlaybackClipBackend,
    RecordClipAction,
    RingClipBackend,
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
    task.log = lambda *a, **k: None
    return task


def _packet(payload: bytes = b"\xff\xd8jpegbytes") -> str:
    """A frame as `base64_encode_jpg` produces it — a JSON packet, not bare base64."""
    return json.dumps({"format": "jpeg", "metadata": {}, "data": base64.b64encode(payload).decode()})


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
