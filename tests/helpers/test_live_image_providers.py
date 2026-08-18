"""Tests for the live image providers (UrlImageProvider, FfmpegImageProvider).

Covers the Phase 1 live-collection fixes:

  - min_capture_interval decimates capture rate (at most one frame per interval),
    tested with a fake clock injected via the provider's `_sleep`/`_now` seams
    so the suite stays fast and doesn't touch the real, process-wide asyncio
    module.
  - Every returned frame carries a unique (source, frame_idx) metadata pair, so
    downstream filenames are unique instead of always "unknown_0.jpg".
  - FfmpegImageProvider.getNextImage() drains to the *newest* queued frame
    rather than returning a stale one.
  - A stream that stops yielding data causes getNextImage() to give up after
    _MAX_STREAM_RESTARTS attempts and return None, rather than looping forever.
  - A provider whose start() fails does not have that error masked by an
    AttributeError out of stop().

No real ffmpeg subprocess or network I/O is used anywhere in this file. Tests
never monkeypatch the shared `asyncio`/`time` modules — both providers expose
injectable `_sleep`/`_now` seams for exactly this purpose, since patching the
real module objects would be process-wide and unsafe alongside concurrently
running tests.
"""

import asyncio
import ssl

import numpy as np
import pytest

from apps.helpers.imageProviders import ffmpegImageProvider as ffmpeg_provider_module
from apps.helpers.imageProviders import urlImageProvider as url_provider_module
from apps.helpers.imageProviders.ffmpegImageProvider import FfmpegImageProvider
from apps.helpers.imageProviders.urlImageProvider import UrlImageProvider
from apps.helpers.streams.ffmpegStreamerIn import FFmpegStreamerIn

# ─── Helpers ──────────────────────────────────────────────────────────────────


class FakeClock:
    """Deterministic stand-in for time.monotonic()/asyncio.sleep() so interval
    tests don't need to actually wait in real time. Injected via a provider's
    `_now`/`_sleep` seams rather than by patching the real modules."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += seconds


class FakeFfmpegStreamer:
    """Minimal stand-in for FFmpegStreamerIn that always reports frame-aligned
    queue items, so FfmpegImageProvider.getNextImage() can be exercised without
    a real ffmpeg process."""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = list(frames)
        self.astart_calls = 0
        self.astop_calls = 0

    def is_frame_aligned(self) -> bool:
        return True

    def read_nowait(self):
        if self._frames:
            return self._frames.pop(0)
        return None

    async def read_async(self, timeout=None):
        if self._frames:
            return self._frames.pop(0)
        raise asyncio.TimeoutError()

    async def wait_for_metadata(self) -> None:
        return None

    def metadata_ready(self) -> bool:
        return True

    def get_frame_size(self):
        return (2, 2)

    async def astart(self) -> None:
        self.astart_calls += 1

    async def astop(self) -> None:
        self.astop_calls += 1


class _NotYetReadyStreamer(FakeFfmpegStreamer):
    """A reader that has been spawned but whose source has not come up yet.

    The state a slow RTSP camera sits in for minutes: the process exists, ffmpeg is
    retrying, and no geometry has been reported. Indistinguishable from a dead source
    without the connect grace, which is the whole point.
    """

    def metadata_ready(self) -> bool:
        return False

    async def wait_for_metadata(self) -> None:
        await asyncio.sleep(3600)


def _frame_bytes(fill: int, size: int = 12) -> bytes:
    return bytes([fill]) * size


def _fast_sleep(_seconds: float):
    """A `_sleep` replacement that yields control without any real delay."""
    return asyncio.sleep(0)


# ─── UrlImageProvider ───────────────────────────────────────────────────────────


class TestUrlImageProviderInterval:
    async def test_min_capture_interval_enforced_with_fake_clock(self, monkeypatch):
        clock = FakeClock()
        provider = UrlImageProvider(url="http://example.invalid/img.jpg", min_capture_interval=1.0)
        monkeypatch.setattr(provider, "_now", clock.monotonic)
        monkeypatch.setattr(provider, "_sleep", clock.sleep)
        monkeypatch.setattr(provider, "_downloadImage", lambda: (np.zeros((1, 1, 3), dtype=np.uint8), b""))

        timestamps = []
        for _ in range(3):
            await provider.getNextImage()
            timestamps.append(clock.t)

        # First capture is immediate; each subsequent capture waits a full interval.
        assert timestamps[0] == 0.0
        assert timestamps[1] >= 1.0
        assert timestamps[2] >= 2.0

    async def test_zero_interval_never_sleeps(self, monkeypatch):
        clock = FakeClock()
        provider = UrlImageProvider(url="http://example.invalid/img.jpg")  # default min_capture_interval=0.0
        monkeypatch.setattr(provider, "_now", clock.monotonic)

        async def _sleep_should_not_be_called(seconds):
            raise AssertionError("sleep() should not be called when min_capture_interval is 0")

        monkeypatch.setattr(provider, "_sleep", _sleep_should_not_be_called)
        monkeypatch.setattr(provider, "_downloadImage", lambda: (np.zeros((1, 1, 3), dtype=np.uint8), b""))

        for _ in range(3):
            await provider.getNextImage()


class TestUrlImageProviderMetadata:
    async def test_metadata_has_unique_source_and_frame_idx(self, monkeypatch):
        provider = UrlImageProvider(url="http://example.invalid/img.jpg")
        provider.set_source_identity("cam1")
        monkeypatch.setattr(provider, "_downloadImage", lambda: (np.zeros((1, 1, 3), dtype=np.uint8), b""))

        pairs = set()
        for _ in range(5):
            result = await provider.getNextImage()
            pairs.add((result.metadata.source, result.metadata.frame_idx))

        assert len(pairs) == 5
        assert all(source == "cam1" for source, _ in pairs)


class TestUrlImageProviderDownloadFailure:
    # Regression test: a UrlImageProvider whose target URL hangs or errors used
    # to give no diagnostic signal at all — no timeout on urlopen() and no
    # logger in the module, unlike every other image provider (ffmpeg/mqtt/cv2),
    # which log extensively via their own per-module logger. That combination
    # left preview requests silently stuck until the frontend's own 30s SSE
    # timeout fired, with nothing in server logs pointing at the cause.

    def test_downloadImage_passes_a_timeout_to_urlopen(self, monkeypatch):
        captured = {}

        def fake_urlopen(request, *args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            raise TimeoutError("simulated hang")

        monkeypatch.setattr(url_provider_module.urllib.request, "urlopen", fake_urlopen)

        provider = UrlImageProvider(url="http://example.invalid/img.jpg", ignore_ssl_errors=False)
        with pytest.raises(TimeoutError):
            provider._downloadImage()

        assert captured["timeout"] is not None

    def test_downloadImage_failure_is_logged(self, monkeypatch, caplog):
        def fake_urlopen(request, *args, **kwargs):
            raise OSError("connection refused")

        monkeypatch.setattr(url_provider_module.urllib.request, "urlopen", fake_urlopen)

        provider = UrlImageProvider(url="http://example.invalid/img.jpg", ignore_ssl_errors=False)
        with caplog.at_level("ERROR", logger=url_provider_module.logger.name):
            with pytest.raises(OSError):
                provider._downloadImage()

        assert any("example.invalid" in record.message for record in caplog.records)


class TestUrlImageProviderSslVerification:
    """Certificate verification is disabled only by explicit opt-in.

    IP cameras ship self-signed certs for their LAN addresses, so
    `ignore_ssl_errors` is a real need — but it is the kind of switch that must
    stay exactly as narrow as it looks, hence asserting both directions.
    """

    def _captured_context(self, monkeypatch, **provider_kwargs):
        captured = {}

        def fake_urlopen(request, *args, **kwargs):
            captured["context"] = kwargs.get("context")
            raise OSError("stop here — the context is all this test wants")

        monkeypatch.setattr(url_provider_module.urllib.request, "urlopen", fake_urlopen)
        provider = UrlImageProvider(url="https://example.invalid/img.jpg", **provider_kwargs)
        with pytest.raises(OSError):
            provider._downloadImage()
        return captured["context"]

    def test_verification_stays_on_by_default(self, monkeypatch):
        """No context passed means urllib's default, which verifies."""
        assert self._captured_context(monkeypatch, ignore_ssl_errors=False) is None

    def test_opting_out_disables_both_checks(self, monkeypatch):
        """Hostname checking and cert verification both have to go: leaving
        check_hostname on would still reject a self-signed LAN cert.
        """
        context = self._captured_context(monkeypatch, ignore_ssl_errors=True)

        assert context is not None
        assert context.check_hostname is False
        assert context.verify_mode is ssl.CERT_NONE

    def test_built_from_the_public_api(self, monkeypatch):
        """Regression guard: this used to call the private
        `ssl._create_unverified_context()`. The public path is
        `create_default_context()` plus the two explicit relaxations, and its
        assignment order is load-bearing — setting verify_mode before
        check_hostname raises ValueError, which would break image collection
        for every camera on a self-signed cert.
        """
        context = url_provider_module._unverified_ssl_context()

        assert isinstance(context, ssl.SSLContext)
        assert (context.check_hostname, context.verify_mode) == (False, ssl.CERT_NONE)


# ─── FfmpegImageProvider ────────────────────────────────────────────────────────


class TestFfmpegImageProviderMetadata:
    async def test_metadata_has_unique_source_and_frame_idx(self, monkeypatch):
        # Feed exactly one frame per call rather than pre-queueing all of them —
        # getNextImage() drains to the newest frame, so pre-queueing 5 frames
        # would have the first call consume all of them and calls 2-5 would find
        # an empty queue (a different, separately-tested scenario).
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        provider.set_source_identity("cam2")
        streamer = FakeFfmpegStreamer([])
        provider._stream = streamer
        provider._frame_size = (2, 2)

        pairs = []
        for i in range(5):
            streamer._frames.append(_frame_bytes(i))
            result = await asyncio.wait_for(provider.getNextImage(), timeout=5.0)
            pairs.append((result.metadata.source, result.metadata.frame_idx))

        assert [p[0] for p in pairs] == ["cam2"] * 5
        assert [p[1] for p in pairs] == [0, 1, 2, 3, 4]


class TestFfmpegImageProviderDrain:
    async def test_getNextImage_returns_newest_queued_frame(self):
        # Three frames are already queued (oldest first, as a real bounded queue
        # would yield them via get_nowait()); getNextImage() must drain to the
        # newest rather than returning the stale first one.
        frames = [_frame_bytes(1), _frame_bytes(2), _frame_bytes(3)]
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        provider._stream = FakeFfmpegStreamer(frames)
        provider._frame_size = (2, 2)

        result = await asyncio.wait_for(provider.getNextImage(), timeout=5.0)

        expected = np.frombuffer(_frame_bytes(3), dtype=np.uint8).reshape((2, 2, 3))
        assert np.array_equal(result.image, expected)

    async def test_min_capture_interval_enforced_with_fake_clock(self, monkeypatch):
        clock = FakeClock()
        provider = FfmpegImageProvider(
            video_path="rtsp://cam.invalid/stream", rtsp_relay=False, min_capture_interval=1.0
        )
        monkeypatch.setattr(provider, "_now", clock.monotonic)
        monkeypatch.setattr(provider, "_sleep", clock.sleep)
        # Plenty of frames queued so draining never blocks on read_async().
        provider._stream = FakeFfmpegStreamer([_frame_bytes(i) for i in range(10)])
        provider._frame_size = (2, 2)

        timestamps = []
        for _ in range(3):
            await asyncio.wait_for(provider.getNextImage(), timeout=5.0)
            timestamps.append(clock.t)

        assert timestamps[0] == 0.0
        assert timestamps[1] >= 1.0
        assert timestamps[2] >= 2.0

    async def test_min_capture_interval_adds_vf_output_filter(self):
        provider = FfmpegImageProvider(
            video_path="rtsp://cam.invalid/stream", rtsp_relay=False, min_capture_interval=2.0
        )
        assert provider._ffmpeg_args.get("vf") == "fps=1/2.0"

    async def test_zero_interval_omits_vf_output_filter(self):
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        assert "vf" not in provider._ffmpeg_args


class TestFfmpegImageProviderRestartBound:
    async def test_stream_with_no_data_returns_none_after_max_restarts(self, monkeypatch):
        # A stream that never yields data (dead source, EOF, etc.) must not spin
        # astop()/astart() forever — getNextImage() must give up and return None.
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        streamer = FakeFfmpegStreamer([])  # never produces a frame
        provider._stream = streamer
        provider._frame_size = (2, 2)
        monkeypatch.setattr(provider, "_sleep", _fast_sleep)  # skip the real backoff delay

        result = await asyncio.wait_for(provider.getNextImage(), timeout=5.0)

        assert result is None
        assert streamer.astop_calls == provider._MAX_STREAM_RESTARTS
        assert streamer.astart_calls == provider._MAX_STREAM_RESTARTS

    async def test_restart_count_resets_after_a_successful_frame(self, monkeypatch):
        # A long-lived stream with occasional hiccups shouldn't be permanently
        # killed by restarts accumulated across many prior getNextImage() calls.
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        streamer = FakeFfmpegStreamer([])
        provider._stream = streamer
        provider._frame_size = (2, 2)
        monkeypatch.setattr(provider, "_sleep", _fast_sleep)

        # A successful frame arrives immediately (no restarts needed), which should
        # reset the restart counter to 0.
        streamer._frames.append(_frame_bytes(1))
        result = await asyncio.wait_for(provider.getNextImage(), timeout=5.0)
        assert result is not None
        assert provider._restart_count == 0

        # Now simulate MAX_STREAM_RESTARTS - 1 hiccups (still under the cap
        # because the counter was reset by the successful frame above).
        for _ in range(provider._MAX_STREAM_RESTARTS - 1):
            streamer._frames.append(_frame_bytes(2))
            result = await asyncio.wait_for(provider.getNextImage(), timeout=5.0)
            assert result is not None


class TestFfmpegImageProviderStopGuards:
    async def test_stop_before_stream_assigned_does_not_raise(self):
        # Simulates start() raising before self._stream is ever assigned (e.g. an
        # RTSP relay timeout) — stop() must not turn that into an AttributeError
        # that masks the original failure.
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        assert provider._stream is None
        await provider.stop()  # must not raise

    async def test_start_failure_does_not_mask_error_and_stop_is_safe(self, monkeypatch):
        async def _raise_astart(self):
            raise RuntimeError("ffmpeg failed to start")

        monkeypatch.setattr(FFmpegStreamerIn, "astart", _raise_astart)

        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        with pytest.raises(RuntimeError, match="ffmpeg failed to start"):
            await provider.start()

        # The original error must propagate uncorrupted, and stop() afterwards
        # (e.g. from a task's `finally`) must not itself raise.
        await provider.stop()

    async def test_relay_destroy_failure_does_not_raise(self, monkeypatch):
        """If destroying the relay stream fails, stop() should log, not raise."""
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        provider._stream = FakeFfmpegStreamer([])

        class ExplodingRelayApi:
            def destroy_stream(self, name):
                raise ConnectionError("relay server unreachable")

        provider._rtsp_relay_api = object()
        monkeypatch.setattr(
            ffmpeg_provider_module.stream_client,
            "StreamsApi",
            lambda api_client: ExplodingRelayApi(),
        )

        await provider.stop()  # must not raise


class TestUrlImageProviderKeepsTheCamerasOwnEncoding:
    """The downloaded JPEG is handed on, not thrown away and re-encoded later.

    It is the camera's full-resolution original, which is what lets a snapshot camera's
    event clip be full-res rather than a 640 px re-encode of the analysis frame — and
    keeping it *removes* an encode rather than adding one.
    """

    def _provider_returning(self, monkeypatch, payload: bytes):
        provider = UrlImageProvider(url="http://example.invalid/img.jpg")
        decoded = np.zeros((4, 4, 3), dtype=np.uint8)
        monkeypatch.setattr(provider, "_downloadImage", lambda: (decoded, payload))
        return provider

    async def test_a_jpeg_payload_is_carried_through_verbatim(self, monkeypatch):
        import base64
        import json

        import cv2

        ok, buffer = cv2.imencode(".jpg", np.zeros((4, 4, 3), dtype=np.uint8))
        assert ok
        payload = buffer.tobytes()
        provider = self._provider_returning(monkeypatch, payload)

        provided = await provider.getNextImage()

        packet = json.loads(provided.encoded_b64)
        assert packet["format"] == "jpeg"
        assert base64.b64decode(packet["data"]) == payload

    async def test_a_non_jpeg_payload_is_not_announced_as_jpeg(self, monkeypatch):
        """A PNG declared as JPEG is a lie that only surfaces in whatever decodes it."""
        import cv2

        ok, buffer = cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint8))
        assert ok
        provider = self._provider_returning(monkeypatch, buffer.tobytes())

        provided = await provider.getNextImage()

        assert provided.encoded_b64 is None


class TestFfmpegImageProviderReconnect:
    """Recovering a dropped camera without disturbing what other readers share.

    The provider gives up on its own stream after `_MAX_STREAM_RESTARTS` frameless
    reads and returns None from then on. That is the right call for one getNextImage()
    — but nothing re-armed it, so the source stayed dark for the rest of the run no
    matter how patiently its consumer retried. reconnect() is what re-arms it.
    """

    async def test_reconnect_rebuilds_the_reader(self, monkeypatch):
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        old = FakeFfmpegStreamer([])
        provider._stream = old
        provider._stream_url = "rtsp://cam.invalid/stream"

        built = []
        monkeypatch.setattr(
            ffmpeg_provider_module,
            "FFmpegStreamerIn",
            lambda url, **kw: built.append(url) or FakeFfmpegStreamer([]),
        )

        await provider.reconnect()

        assert old.astop_calls == 1, "the dead reader must be torn down"
        assert built == ["rtsp://cam.invalid/stream"], "a fresh reader must be built"
        assert provider._stream is not old

    async def test_reconnect_clears_the_giving_up_state(self, monkeypatch):
        """Without this the rebuilt stream inherits a spent restart budget and a
        cached frame size, and gives up again on its first read."""
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        provider._stream = FakeFfmpegStreamer([])
        provider._stream_url = "rtsp://cam.invalid/stream"
        provider._restart_count = provider._MAX_STREAM_RESTARTS + 5
        provider._frame_size = (2, 2)
        provider._buffer = b"stale"
        monkeypatch.setattr(ffmpeg_provider_module, "FFmpegStreamerIn", lambda url, **kw: FakeFfmpegStreamer([]))

        await provider.reconnect()

        assert provider._restart_count == 0
        # Re-read rather than trusted: a camera can come back at a different
        # resolution, and a stale frame size misaligns every read off the pipe.
        assert provider._frame_size is None
        assert provider._buffer == b""

    async def test_reconnect_keeps_the_relay_lease(self, monkeypatch):
        """The path is shared and refcounted. stop()/start() on every retry would
        churn a lease other watchers on the same camera are also holding — and drop
        the rewind buffer's continuity each time a flapping camera bounced."""
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        provider._stream = FakeFfmpegStreamer([])
        provider._stream_url = "rtsp://relay/path"
        provider._lease_id = "lease-123"
        released = []
        monkeypatch.setattr(provider, "_release_lease_quietly", lambda: released.append(True) or asyncio.sleep(0))
        monkeypatch.setattr(ffmpeg_provider_module, "FFmpegStreamerIn", lambda url, **kw: FakeFfmpegStreamer([]))

        await provider.reconnect()

        assert released == [], "reconnect must not hand the lease back"
        assert provider._lease_id == "lease-123"

    async def test_reconnect_does_not_wait_for_the_camera_to_come_up(self, monkeypatch):
        """It rebuilds and returns; readiness is asked separately.

        Blocking here for a camera that is not coming back would cost every *other*
        camera its frame rate, since one loop services them all.
        """

        class _NeverReady(FakeFfmpegStreamer):
            async def wait_for_metadata(self):
                await asyncio.sleep(3600)

            def metadata_ready(self) -> bool:
                return False

        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        provider._stream = FakeFfmpegStreamer([])
        provider._stream_url = "rtsp://cam.invalid/stream"
        monkeypatch.setattr(ffmpeg_provider_module, "FFmpegStreamerIn", lambda url, **kw: _NeverReady([]))

        await asyncio.wait_for(provider.reconnect(), timeout=5.0)

        assert provider.is_ready() is False, "a source that has not come up is not ready"

    async def test_a_provider_that_never_started_falls_back_to_a_full_start(self, monkeypatch):
        """start() failing is the ordinary way a slow camera begins life — it misses
        the relay-readiness window. If reconnect could not recover that, the source
        would be permanently dead for the rest of the run."""
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        started = []
        monkeypatch.setattr(
            ffmpeg_provider_module,
            "FFmpegStreamerIn",
            lambda url, **kw: started.append(url) or FakeFfmpegStreamer([]),
        )

        await provider.reconnect()

        assert started == ["rtsp://cam.invalid/stream"], "a full start() must be attempted"

    async def test_is_ready_is_false_without_a_stream(self):
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        assert provider.is_ready() is False


class TestFfmpegImageProviderConnectGrace:
    """A camera that is slow to come up must not be torn down mid-handshake.

    An RTSP camera behind a relay can take two or three minutes from cold, and
    MediaMTX retries the source in the background throughout. Before the grace
    existed, the watcher saw "not ready", called it broken, and rebuilt the reader —
    restarting the clock. A camera slower than the retry interval could therefore
    reconnect forever without ever once connecting.
    """

    async def test_a_freshly_started_reader_is_connecting_not_broken(self, monkeypatch):
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        provider._stream = _NotYetReadyStreamer([])
        provider._stream_url = "rtsp://cam.invalid/stream"
        provider._stream_started_at = 100.0
        monkeypatch.setattr(provider, "_now", lambda: 101.0)

        assert provider.is_ready() is False
        assert provider.is_connecting() is True

    async def test_reconnect_leaves_a_connecting_reader_alone(self, monkeypatch):
        """The core regression: rebuilding does not hurry a handshake, it restarts it."""
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        streamer = _NotYetReadyStreamer([])
        provider._stream = streamer
        provider._stream_url = "rtsp://cam.invalid/stream"
        provider._stream_started_at = 100.0
        # Well inside the grace, and far beyond any failure backoff.
        monkeypatch.setattr(provider, "_now", lambda: 100.0 + 120.0)
        monkeypatch.setattr(ffmpeg_provider_module, "FFmpegStreamerIn", lambda url, **kw: FakeFfmpegStreamer([]))

        await provider.reconnect()

        assert streamer.astop_calls == 0, "a connecting reader must not be torn down"
        assert provider._stream is streamer, "and must not be replaced"

    async def test_a_reader_past_the_grace_is_rebuilt(self, monkeypatch):
        """Patience is bounded — a stream that never arrives is eventually broken."""
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        streamer = _NotYetReadyStreamer([])
        provider._stream = streamer
        provider._stream_url = "rtsp://cam.invalid/stream"
        provider._stream_started_at = 100.0
        monkeypatch.setattr(provider, "_now", lambda: 100.0 + provider.CONNECT_GRACE_S + 1)
        monkeypatch.setattr(ffmpeg_provider_module, "FFmpegStreamerIn", lambda url, **kw: FakeFfmpegStreamer([]))

        assert provider.is_connecting() is False
        await provider.reconnect()

        assert streamer.astop_calls == 1
        assert provider._stream is not streamer

    async def test_a_connected_reader_is_not_connecting(self):
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        provider._stream = FakeFfmpegStreamer([])  # metadata_ready() is True
        provider._stream_started_at = 0.0

        assert provider.is_ready() is True
        assert provider.is_connecting() is False

    async def test_the_grace_is_long_enough_for_a_real_camera(self):
        """Two to three minutes is normal from cold; the window has to clear that
        with room, or the fix only moves the cliff."""
        assert FfmpegImageProvider.CONNECT_GRACE_S >= 180.0

    async def test_each_rebuild_gets_a_full_window(self, monkeypatch):
        """Otherwise the second attempt inherits the first one's elapsed time and is
        declared broken before it has had a chance."""
        provider = FfmpegImageProvider(video_path="rtsp://cam.invalid/stream", rtsp_relay=False)
        provider._stream = _NotYetReadyStreamer([])
        provider._stream_url = "rtsp://cam.invalid/stream"
        provider._stream_started_at = 0.0
        monkeypatch.setattr(provider, "_now", lambda: 1000.0)
        monkeypatch.setattr(ffmpeg_provider_module, "FFmpegStreamerIn", lambda url, **kw: _NotYetReadyStreamer([]))

        await provider.reconnect()

        assert provider._stream_started_at == 1000.0
        assert provider.is_connecting() is True
