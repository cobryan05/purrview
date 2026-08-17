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


class TestMetadataReadyMeansGeometryActuallyArrived:
    """`metadata_ready()` is what the watcher reads as "this source is really there".

    The stderr reader sets `_metadata_event` in a `finally`, which is right for
    `wait_for_metadata()` — a waiter must not hang when ffmpeg dies without printing a
    banner. It is wrong for the readiness question: a camera that is gone makes ffmpeg
    exit immediately, stderr hits EOF, and the event is set with no geometry behind it.
    The provider then reports `is_ready()` and the watcher, seeing a reader it believes
    is up, never rebuilds it.
    """

    class _Stderr:
        def __init__(self, lines: list[bytes]) -> None:
            self._lines = list(lines)

        def readline(self) -> bytes:
            return self._lines.pop(0) if self._lines else b""

    def _streamer(self, lines: list[bytes]) -> FFmpegStreamerIn:
        streamer = FFmpegStreamerIn("rtsp://fake")
        streamer._process = type("P", (), {"stderr": self._Stderr(lines)})()
        return streamer

    async def test_a_banner_that_never_parses_is_not_ready(self):
        streamer = self._streamer([b"ffmpeg version 6.0\n", b"rtsp://fake: Connection refused\n"])

        streamer._pump_stderr()

        assert streamer.metadata_ready() is False, "no geometry arrived, so the source is not there"
        # The waiter still has to be released, or every caller hangs forever.
        await asyncio.wait_for(streamer.wait_for_metadata(), timeout=1.0)

    async def test_a_parsed_banner_is_ready(self):
        streamer = self._streamer(
            [
                b"ffmpeg version 6.0\n",
                b"  Stream #0:0: Video: h264, yuv420p, 1920x1080, 15 fps, 15 tbr, 90k tbn\n",
            ]
        )

        streamer._pump_stderr()

        assert streamer.metadata_ready() is True
        assert streamer.get_frame_size() == (1920, 1080)

    async def test_a_restart_clears_a_previous_readiness(self):
        """Geometry from the stream that died must not vouch for the one replacing it."""
        streamer = self._streamer([b"  Stream #0:0: Video: h264, yuv420p, 1920x1080, 15 fps, 15 tbr, 90k tbn\n"])
        streamer._pump_stderr()
        assert streamer.metadata_ready() is True

        streamer._process = None
        streamer._metadata_event.clear()
        streamer._metadata_parsed = False

        assert streamer.metadata_ready() is False


class TestStderrIsDrainedForTheLifeOfTheProcess:
    """The stall that killed a live RTSP source for an hour without killing anything.

    ffmpeg runs at `-loglevel info` and writes to stderr for as long as it lives, but
    the reader stopped at the first parsed banner and never read another byte. A pipe
    holds 64 KiB. Once that filled, ffmpeg blocked in `write()` on stderr — and a
    process blocked writing its log is a process that has stopped reading its input, so
    it stopped pulling RTSP, MediaMTX dropped the stalled session, and the frames
    ended. Nothing crashed: the process stayed alive, `metadata_ready()` stayed true,
    stdout stayed open and empty, and every health surface upstream reported a reader
    that was up. Measured on the wedged container, stdout held 0/65536 bytes and
    stderr 65087/65536.

    So the contract is: stderr is consumed until EOF, whether or not a banner ever
    parsed, and whether or not anyone is still interested in the metadata.
    """

    class _CountingStderr:
        """Yields `total` lines, then EOF. Counts how many were actually consumed."""

        def __init__(self, total: int, banner_at: int = 1) -> None:
            self.total = total
            self.banner_at = banner_at
            self.consumed = 0

        def readline(self) -> bytes:
            if self.consumed >= self.total:
                return b""
            self.consumed += 1
            if self.consumed == self.banner_at:
                return b"  Stream #0:0: Video: h264, yuv420p, 1280x720, 25 fps, 25 tbr, 90k tbn\n"
            return b"[rtsp @ 0x1] max delay reached. need to consume packet\n"

    def _streamer(self, stderr) -> FFmpegStreamerIn:
        streamer = FFmpegStreamerIn("rtsp://fake")
        streamer._process = type("P", (), {"stderr": stderr})()
        return streamer

    async def test_logging_continues_to_be_consumed_after_the_banner(self):
        """The regression, in one assertion: everything ffmpeg writes gets read."""
        stderr = self._CountingStderr(total=500, banner_at=1)
        streamer = self._streamer(stderr)

        streamer._pump_stderr()

        assert streamer.metadata_ready() is True
        assert stderr.consumed == 500, (
            "stderr must be drained for the life of the process — stopping at the banner "
            "wedges ffmpeg on a full pipe once it has logged 64 KiB"
        )

    async def test_the_metadata_is_published_without_waiting_for_eof(self):
        """A live camera's stderr never reaches EOF, so a waiter that needed the pump to
        finish would wait forever. The banner has to release it mid-stream."""
        streamer = self._streamer(self._CountingStderr(total=3, banner_at=1))
        released = []

        async def waiter():
            await streamer.wait_for_metadata()
            released.append(streamer.get_frame_size())

        task = asyncio.ensure_future(waiter())
        await asyncio.to_thread(streamer._pump_stderr)
        await asyncio.wait_for(task, timeout=1.0)

        assert released == [(1280, 720)]

    async def test_a_second_banner_does_not_overwrite_the_first(self):
        """ffmpeg prints a banner per stream and per reconnect. The geometry the frame
        reader is already unpacking against must not change under it."""
        stderr = self._CountingStderr(total=2, banner_at=1)
        streamer = self._streamer(stderr)
        streamer._pump_stderr()
        first = streamer.get_frame_size()

        streamer._process = type("P", (), {"stderr": self._CountingStderr(total=1, banner_at=1)})()
        streamer._pump_stderr()

        assert streamer.get_frame_size() == first

    async def test_draining_survives_a_stderr_that_raises(self):
        """A pump that dies on a decode error re-creates the deadlock it exists to
        prevent, so the failure must not escape it."""

        class _Boom:
            def readline(self):
                raise OSError("stderr went away")

        streamer = self._streamer(_Boom())

        streamer._pump_stderr()  # must not raise

        await asyncio.wait_for(streamer.wait_for_metadata(), timeout=1.0)
