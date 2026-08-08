"""Tests for FFmpegStreamerIn — bounded, frame-aligned queue and async start/stop.

Covers the Phase 1 live-collection fixes:

  - The frame queue is bounded (max_queued_frames) and drops the *oldest* item
    when full, so read_nowait()/read_async() always surface the freshest data.
  - When output is rawvideo, the worker reads whole frames via readexactly()
    and each queue item is exactly one frame (is_frame_aligned() is True); an
    IncompleteReadError is treated as end-of-stream.
  - read_nowait() is non-blocking and returns None on an empty queue.
  - stop() is idempotent and None-guarded (no AttributeError after a failed
    start()).
  - astart()/astop() work without blocking, and astop() is safe to call when
    nothing was ever started.

No real ffmpeg subprocess or network I/O is used anywhere in this file.
"""

import asyncio

import pytest

from apps.helpers.streams.ffmpegStreamerIn import FFmpegStreamerIn

# ─── Helpers ──────────────────────────────────────────────────────────────────


class FakeReader:
    """Stand-in for asyncio.StreamReader that only implements readexactly()."""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = list(frames)

    async def readexactly(self, n: int) -> bytes:
        if not self._frames:
            raise asyncio.IncompleteReadError(partial=b"", expected=n)
        return self._frames.pop(0)


def _drain_all(streamer: FFmpegStreamerIn) -> list[bytes]:
    collected = []
    item = streamer.read_nowait()
    while item is not None:
        collected.append(item)
        item = streamer.read_nowait()
    return collected


# ─── Bounded queue / frame alignment ───────────────────────────────────────────


class TestBoundedQueue:
    async def test_drops_oldest_when_full_and_stays_frame_aligned(self):
        frame_size = 10
        streamer = FFmpegStreamerIn("rtsp://fake", output_args={"format": "rawvideo"}, max_queued_frames=2)
        streamer._frame_size = frame_size
        streamer._is_frame_aligned = True

        frames = [bytes([i]) * frame_size for i in range(5)]
        for f in frames:
            await streamer._put_dropping_oldest(f)

        collected = _drain_all(streamer)

        # Only the last max_queued_frames survive, and each is exactly frame-sized.
        assert collected == frames[-2:]
        assert all(len(item) == frame_size for item in collected)
        assert streamer.is_frame_aligned() is True

    async def test_put_dropping_oldest_never_blocks(self):
        streamer = FFmpegStreamerIn("rtsp://fake", max_queued_frames=1)
        # Two puts on a maxsize=1 queue would deadlock with put_nowait/put; verify neither blocks.
        await asyncio.wait_for(streamer._put_dropping_oldest(b"a"), timeout=1.0)
        await asyncio.wait_for(streamer._put_dropping_oldest(b"b"), timeout=1.0)
        assert _drain_all(streamer) == [b"b"]

    async def test_read_nowait_returns_none_when_empty(self):
        streamer = FFmpegStreamerIn("rtsp://fake", max_queued_frames=2)
        assert streamer.read_nowait() is None

    async def test_is_frame_aligned_defaults_false(self):
        streamer = FFmpegStreamerIn("rtsp://fake")
        assert streamer.is_frame_aligned() is False


class TestFrameAlignedWorkerLoop:
    async def test_reads_whole_frames_and_stops_on_incomplete_read(self):
        frame_size = 4
        frames = [bytes([i]) * frame_size for i in range(3)]
        streamer = FFmpegStreamerIn("rtsp://fake", output_args={"format": "rawvideo"}, max_queued_frames=10)
        streamer._frame_size = frame_size

        reader = FakeReader(frames)
        await streamer._worker_loop_frame_aligned(reader)

        assert _drain_all(streamer) == frames

    async def test_incomplete_read_error_is_treated_as_eof_not_raised(self):
        frame_size = 4
        streamer = FFmpegStreamerIn("rtsp://fake", output_args={"format": "rawvideo"}, max_queued_frames=10)
        streamer._frame_size = frame_size

        reader = FakeReader([])  # immediately raises IncompleteReadError
        # Should return normally (EOF), not propagate the exception.
        await streamer._worker_loop_frame_aligned(reader)
        assert _drain_all(streamer) == []


# ─── stop()/astop() guards and idempotency ─────────────────────────────────────


class TestStopGuards:
    async def test_sync_stop_guards_none_process_and_is_idempotent(self):
        streamer = FFmpegStreamerIn("rtsp://fake")
        assert streamer._process is None
        # Must not raise AttributeError even though start() was never called.
        streamer.stop()
        streamer.stop()

    async def test_astop_guards_none_process_and_is_idempotent(self):
        streamer = FFmpegStreamerIn("rtsp://fake")
        assert streamer._process is None
        await streamer.astop()
        await streamer.astop()

    async def test_astop_cancels_worker_task(self):
        streamer = FFmpegStreamerIn("rtsp://fake")

        async def _never_ending():
            await asyncio.Event().wait()

        streamer._task = asyncio.get_running_loop().create_task(_never_ending())
        await streamer.astop()
        assert streamer._task is None


class TestReadTransportCleanup:
    """The stdout pipe transport from connect_read_pipe() is created on every
    start()/astart(). It used to be dropped without closing, so each restart leaked
    the pipe fd and the StreamReader's buffer — and FfmpegImageProvider resets its
    restart counter on every successful frame, so a flapping source restarts
    unboundedly over a long run."""

    class _FakeTransport:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    async def test_sync_stop_closes_the_read_transport(self):
        streamer = FFmpegStreamerIn("rtsp://fake")
        transport = self._FakeTransport()
        streamer._read_transport = transport

        streamer.stop()

        assert transport.closed
        assert streamer._read_transport is None

    async def test_astop_closes_the_read_transport(self):
        streamer = FFmpegStreamerIn("rtsp://fake")
        transport = self._FakeTransport()
        streamer._read_transport = transport

        await streamer.astop()

        assert transport.closed
        assert streamer._read_transport is None

    async def test_close_is_idempotent(self):
        streamer = FFmpegStreamerIn("rtsp://fake")
        transport = self._FakeTransport()
        streamer._read_transport = transport

        streamer.stop()
        streamer.stop()
        await streamer.astop()

        assert transport.closed

    async def test_a_failing_close_does_not_propagate(self):
        """stop() is called from cleanup paths; a bad transport must not mask them."""

        class _Boom:
            def close(self):
                raise OSError("already gone")

        streamer = FFmpegStreamerIn("rtsp://fake")
        streamer._read_transport = _Boom()

        streamer.stop()

        assert streamer._read_transport is None

    async def test_restart_cycles_do_not_accumulate_transports(self):
        streamer = FFmpegStreamerIn("rtsp://fake")
        transports = []

        for _ in range(5):
            transport = self._FakeTransport()
            transports.append(transport)
            streamer._read_transport = transport  # as the worker would set it
            await streamer.astop()

        assert all(t.closed for t in transports)
        assert streamer._read_transport is None
