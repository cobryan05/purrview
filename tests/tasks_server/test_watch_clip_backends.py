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

import pytest

from apps.helpers.actions.action import ActionContext
from apps.helpers.actions.recordClipAction import (
    FrameRing,
    PlaybackClipBackend,
    RecordClipAction,
    RingClipBackend,
)
from apps.helpers.watch.engine import WatchEngine
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
