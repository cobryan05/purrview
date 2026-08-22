import asyncio
import logging
import re
from dataclasses import dataclass
from subprocess import Popen
from typing import Any, Dict, Optional, Tuple

import ffmpeg

from apps.helpers.logUtils import LOG_LEVEL

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

READ_CHUNK_SIZE = 16 * 1024

# A `Stream #0:0: Video: ...` banner line, which is where ffmpeg reports geometry.
_VIDEO_BANNER_RE = re.compile(r"Stream #\d+:\d+.*?:\s*Video:", re.IGNORECASE)
# WxH as a standalone token. The word boundaries matter: without them `0x18424752`
# (the rawvideo fourcc) parses as a resolution.
_RESOLUTION_RE = re.compile(r"(?<![\w.])(\d{2,5})x(\d{2,5})(?![\w.])")
# fps is what we want; tbr is the fallback, since plenty of IP cameras report only
# that and waiting for an fps figure that never arrives hangs the stream forever.
_FPS_RE = re.compile(r"([\d.]+)\s*fps")
_TBR_RE = re.compile(r"([\d.]+)\s*tbr")
# Pixel format token, e.g. `yuv420p(progressive)` or `rgb24(pc, gbr/...)`.
_PIX_FMT_RE = re.compile(r"\b((?:yuv|yuvj|rgb|bgr|gray|nv)\w*)\b", re.IGNORECASE)


@dataclass(frozen=True)
class VideoStreamMetadata:
    width: int
    height: int
    fps: float
    pixel_format: str

    @property
    def raw24_frame_size(self) -> int:
        """Bytes per frame on the rawvideo output pipe, whatever the source's own format.

        Packed 24-bit either way, so the number is the same for bgr24 and rgb24 — the
        name deliberately does not say which, because callers request `bgr24` (see
        `FfmpegImageProvider`) and a name asserting otherwise is how the wrong channel
        order gets re-introduced.
        """
        return self.width * self.height * 3


def parse_video_stream_banner(line: str) -> Optional[VideoStreamMetadata]:
    """Parse one ffmpeg stderr line into video metadata, or None if it isn't a
    complete video-stream banner.

    Deliberately tolerant, because the caller cannot recover from a miss: for a live
    source stderr never reaches EOF, so failing to parse means `wait_for_metadata()`
    blocks forever instead of raising. Handles the real-world shapes that broke a
    naive comma-splitting parser — `640x480 [SAR 1:1 DAR 4:3]` (a space inside the
    resolution field), `rawvideo (RGB[24] / 0x18424752)` (an 'x' that is not a
    resolution), and cameras that report `tbr` but no `fps`.

    Returns None rather than partial data: a wrong frame size silently misaligns every
    subsequent read off the rawvideo pipe, which is far worse than reading one more
    line of stderr.
    """
    if not _VIDEO_BANNER_RE.search(line):
        return None

    resolution = _RESOLUTION_RE.search(line)
    if not resolution:
        return None

    fps_match = _FPS_RE.search(line) or _TBR_RE.search(line)
    if not fps_match:
        return None

    candidates = [m.group(1).lower() for m in _PIX_FMT_RE.finditer(line)]
    if not candidates:
        return None
    # Prefer a token carrying a bit depth ("rgb24", "yuv420p") over a bare one: the
    # codec parenthetical `rawvideo (RGB[24] / 0x…)` yields a spurious "rgb" that
    # appears *before* the real pixel format in the line.
    pixel_format = next((c for c in candidates if any(ch.isdigit() for ch in c)), candidates[0])

    return VideoStreamMetadata(
        width=int(resolution.group(1)),
        height=int(resolution.group(2)),
        fps=float(fps_match.group(1)),
        pixel_format=pixel_format,
    )


class FFmpegStreamerIn:
    # Shutdown budget. Every stop path is bounded by these, because the caller is a
    # task being cancelled by a user clicking Stop: unbounded teardown reads to that
    # user as a button that does nothing. Class attributes, not module constants, so
    # a test can shrink them without patching the module.
    TERMINATE_TIMEOUT_S: float = 5.0
    KILL_TIMEOUT_S: float = 5.0
    WORKER_JOIN_TIMEOUT_S: float = 5.0

    def __init__(
        self,
        source: str,
        input_args: Optional[Dict[str, str]] = None,
        output_args: Optional[Dict[str, str]] = None,
        max_queued_frames: int = 2,
    ) -> None:

        self._extra_args: Dict[str, Any] = {}
        if source.startswith("http://") or source.startswith("rtsp://") or source.startswith("https://"):
            self._extra_args = {
                "rtsp_transport": "tcp",
                "format": "rtsp",
                "use_wallclock_as_timestamps": 1,
                # Read at the live edge. Without these ffmpeg holds roughly three
                # seconds of the stream before it hands over the first frame, and —
                # because it then delivers in real time rather than catching up — that
                # head start never closes: every frame reaches the reader ~3.2 s after
                # the camera sent it, for the whole life of the process.
                #
                # Measured against the relay, matching a live frame back to the same
                # picture in the recording buffer: 3.20 s and 3.40 s on the two
                # cameras with no flags, 0.41 s and 0.21 s with both of these.
                #
                # That lag is what desynchronised clip overlays. A detection is
                # timestamped when the watcher analyses the frame, while the clip it
                # is drawn over is timestamped when MediaMTX recorded that same
                # picture, so the reader's latency is subtracted from one side only
                # and every box lands seconds late.
                #
                # Both are needed and neither alone does anything: `nobuffer` on its
                # own measured 3.20 s, unchanged. `probesize`/`analyzeduration` reach
                # the same latency but cost more than they are worth here — they
                # suppress ffmpeg's frame-rate estimate ("not enough frames to
                # estimate rate"), and the banner is where this class reads geometry
                # and fps from.
                "fflags": "nobuffer",
                "flags": "low_delay",
            }

        self._source: str = source
        self._input_args: Dict[str, str] = input_args or {}
        self._output_args: Dict[str, str] = output_args or {}
        self._max_queued_frames: int = max_queued_frames

        self._pixel_format: Optional[str] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._fps: Optional[float] = None
        self._frame_size: Optional[int] = None
        self._process: Optional[Popen] = None
        self._frame_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max_queued_frames)
        self._stop_event: asyncio.Event = asyncio.Event()
        self._metadata_event: asyncio.Event = asyncio.Event()
        #: Whether a banner actually parsed, as opposed to the stderr reader merely
        #: finishing. Kept apart from `_metadata_event` because that event doubles as
        #: "stop waiting", and "stop waiting" is not "the camera is there".
        self._metadata_parsed: bool = False
        self._task: Optional[asyncio.Task] = None
        #: The stderr pump. Held only so a caller can tell it apart from a leak; it is
        #: never cancelled, because it is blocked in a thread on a blocking readline
        #: that cancellation cannot interrupt. Reaping the process is what ends it.
        self._stderr_task: Optional[asyncio.Future] = None
        self._event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._is_frame_aligned: bool = False
        # Transport returned by connect_read_pipe() in the worker. A new one is created
        # on every start()/astart(); without closing it on stop the pipe fd and the
        # StreamReader's buffer stay alive for the life of the process, and a source
        # that flaps restarts unboundedly (FfmpegImageProvider resets its restart
        # counter on every successful frame, so the cap is only on *consecutive* ones).
        self._read_transport: Optional[asyncio.ReadTransport] = None

    def __repr__(self):
        return f"FFmpegStreamerIn [{self._source}]"

    def get_fps(self) -> float:
        """Returns the FPS of the source stream, if available."""
        if self._fps is None:
            raise RuntimeError("FPS not available yet.")
        return self._fps

    def get_frame_size(self) -> Tuple[int, int]:
        """Returns the frame size (width, height) of the source stream, if available"""
        if self._width is None or self._height is None:
            raise RuntimeError("Frame size not available yet.")
        return self._width, self._height

    def get_pixel_format(self) -> str:
        """Returns the pixel format of the stream, if available"""
        if self._pixel_format is None:
            raise RuntimeError("Frame size not available yet.")
        return self._pixel_format

    def is_frame_aligned(self) -> bool:
        """Returns True if each item in the frame queue is exactly one whole raw frame."""
        return self._is_frame_aligned

    def metadata_ready(self) -> bool:
        """Non-blocking form of wait_for_metadata(): has ffmpeg reported geometry yet?

        For a live source that never connects, stderr never reaches EOF and the banner
        never parses, so waiting is indistinguishable from waiting forever. A caller
        polling a reconnect needs to ask without committing to that.

        Answers from the parsed flag rather than the event, and the two genuinely differ:
        the event is also set when the reader gives up, so that `wait_for_metadata()`
        releases its waiters instead of hanging on a camera that is gone. Reading the
        event here would turn that release into a claim that the source is up — which is
        the reading callers act on when they decide not to rebuild it.
        """
        return self._metadata_parsed

    async def wait_for_metadata(self) -> None:
        """Waits for metadata to become available"""
        await self._metadata_event.wait()

    async def read_async(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """Returns the next frame from the queue"""
        item = await asyncio.wait_for(self._frame_queue.get(), timeout=timeout)
        self._frame_queue.task_done()
        return item

    def read(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """Returns the next frame from the queue, or None if no frame is available within the timeout."""
        try:
            assert self._event_loop is not None
            future = asyncio.run_coroutine_threadsafe(self.read_async(timeout=timeout), self._event_loop)
            return future.result()
        except asyncio.TimeoutError:
            return None

    def read_nowait(self) -> Optional[bytes]:
        """Non-blocking read of the next queued item; returns None if the queue is empty."""
        try:
            item = self._frame_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        self._frame_queue.task_done()
        return item

    async def _put_dropping_oldest(self, item: bytes) -> None:
        """Enqueue item, dropping the oldest queued item if full. Never blocks."""
        while True:
            try:
                self._frame_queue.put_nowait(item)
                return
            except asyncio.QueueFull:
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.task_done()
                except asyncio.QueueEmpty:
                    pass

    def _close_read_transport(self) -> None:
        """Close the stdout pipe transport, if one is open. Safe to call repeatedly."""
        if self._read_transport is None:
            return
        try:
            self._read_transport.close()
        except Exception as e:
            logger.warning(f"Failed to close read transport for {self._source}: {e}")
        self._read_transport = None

    def _reap_process(self, process: Popen) -> None:
        """Terminate *process*, escalating to SIGKILL, and never block indefinitely.

        A bare `terminate()` + unbounded `wait()` is what wedged a watcher forever:
        ffmpeg blocked on a read from an RTSP source that stopped delivering does not
        act on SIGTERM, so the wait never returns. Off the event loop that is worse
        than slow — `asyncio.to_thread` is not cancellable, so the task awaiting it
        can never be cancelled either, and the whole Stop path hangs behind one dead
        camera.

        The escalation therefore lives in a single synchronous function: whichever
        thread runs it is guaranteed back within TERMINATE + KILL seconds, so the one
        `to_thread` its async caller awaits is bounded too. Giving up (both waits
        expiring) is logged and returns rather than raising — a leaked ffmpeg is a
        far smaller problem than an un-stoppable task, and the caller still has
        teardown left to do.
        """
        process.terminate()
        try:
            process.wait(timeout=self.TERMINATE_TIMEOUT_S)
            return
        except Exception:  # subprocess.TimeoutExpired
            logger.warning(f"ffmpeg for {self._source} ignored SIGTERM after {self.TERMINATE_TIMEOUT_S}s — killing")
        process.kill()
        try:
            process.wait(timeout=self.KILL_TIMEOUT_S)
        except Exception:
            logger.error(
                f"ffmpeg for {self._source} (pid {process.pid}) survived SIGKILL for "
                f"{self.KILL_TIMEOUT_S}s — abandoning it rather than blocking shutdown"
            )

    def stop(self) -> None:
        """Stops the streamer and terminates the process. Safe to call multiple times."""
        self._stop_event.set()
        self._close_read_transport()
        if self._process is None:
            return
        process = self._process
        self._process = None
        self._reap_process(process)

    def _build_process(self) -> Popen:
        """Builds (spawns) the ffmpeg subprocess. Blocking."""
        output_args = self._output_args or {"codec": "copy", "format": "nut"}
        return (
            ffmpeg.input(self._source, **self._input_args, **self._extra_args)
            .output("pipe:1", **output_args)
            .global_args("-nostats", "-loglevel", "info")  # Set loglevel to info to ensure dimensions and fps present
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

    def start(self) -> None:
        """Starts the FFmpeg process and the background task."""
        if self._process:
            raise RuntimeError("Can't start new process while old process is running!")

        self._process = self._build_process()

        self._stop_event.clear()
        self._metadata_event.clear()
        # Geometry from the stream that just died must not vouch for the one replacing
        # it: a camera can come back at a different resolution, and a caller polling
        # readiness would otherwise be told the new reader was up before it had read a
        # single byte of it.
        self._metadata_parsed = False
        self._is_frame_aligned = False

        self._task = self._event_loop.create_task(self._worker_task())

    async def astart(self) -> None:
        """Async, non-blocking equivalent of start(). Spawns the process off the event loop
        and creates the worker task on the running loop (task creation is not thread-safe)."""
        if self._process:
            raise RuntimeError("Can't start new process while old process is running!")

        self._process = await asyncio.to_thread(self._build_process)

        self._stop_event.clear()
        self._metadata_event.clear()
        # See start(): a restart must not inherit the old stream's readiness.
        self._metadata_parsed = False
        self._is_frame_aligned = False

        self._task = self._event_loop.create_task(self._worker_task())

    async def astop(self) -> None:
        """Async, non-blocking equivalent of stop(). Terminates the process and cancels/awaits
        the worker task without blocking the event loop."""
        self._stop_event.set()
        self._close_read_transport()

        if self._process is not None:
            process = self._process
            self._process = None
            await asyncio.to_thread(self._reap_process, process)

        if self._task is not None:
            task = self._task
            self._task = None
            task.cancel()
            # asyncio.wait, not wait_for: on timeout wait_for cancels the inner task
            # and then *awaits* it, so a worker that swallows cancellation (one stuck
            # in a blocking call) hangs wait_for exactly as hard as the bare `await
            # task` this replaced. asyncio.wait just stops watching.
            done, pending = await asyncio.wait({task}, timeout=self.WORKER_JOIN_TIMEOUT_S)
            if pending:
                logger.error(
                    f"Worker task for {self._source} did not stop within "
                    f"{self.WORKER_JOIN_TIMEOUT_S}s — leaving it detached"
                )
            else:
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Worker task for {self._source} raised during astop: {e}")

    def _pump_stderr(self) -> None:
        """Consume ffmpeg's stderr until EOF, publishing the geometry banner on the way.

        Draining is the point, and it is not optional bookkeeping. ffmpeg runs at
        `-loglevel info` and writes to stderr for as long as it lives; a pipe holds
        64 KiB. This used to stop reading at the first parsed banner, so on a long-lived
        camera the pipe eventually filled and ffmpeg blocked in `write()` — and a
        process blocked writing its log has stopped reading its input, so it stopped
        pulling RTSP, MediaMTX dropped the stalled session, and the frames ended.

        Nothing about that failure looks like a failure from outside: the process stays
        alive, stdout stays open (and empty), the geometry is still known, so every
        readiness check upstream keeps answering yes. It was measured on a wedged
        container as stdout 0/65536 bytes and stderr 65087/65536.

        The banner is published mid-stream rather than by returning, because for a live
        source EOF never comes — see `wait_for_metadata()`. Only the first banner counts:
        ffmpeg prints one per stream and per internal reconnect, and the frame reader is
        already unpacking fixed-size frames against the first one.
        """
        assert self._process is not None and self._process.stderr is not None
        try:
            for line in iter(self._process.stderr.readline, b""):
                line = line.decode("utf-8", "replace").strip()
                logger.debug(f"FFmpeg stderr: {line}")

                if self._metadata_parsed:
                    continue
                meta = parse_video_stream_banner(line)
                if meta is None:
                    continue
                self._width = meta.width
                self._height = meta.height
                self._fps = meta.fps
                self._pixel_format = meta.pixel_format
                self._frame_size = meta.raw24_frame_size
                self._metadata_parsed = True
                self._metadata_event.set()
        except Exception as e:
            # Swallowed on purpose: a pump that dies re-creates the exact deadlock it
            # exists to prevent, and it dying is invisible from anywhere else.
            logger.error(f"Error while reading ffmpeg stderr for {self._source}: {e}")
        finally:
            # Releases `wait_for_metadata()` even when no banner ever parsed — waiting
            # on a camera that is gone is otherwise indistinguishable from waiting
            # forever. `metadata_ready()` deliberately does not read this event.
            self._metadata_event.set()

    async def _worker_task(self) -> None:
        """Reads frames from the source and puts them into the frame queue."""
        assert (
            self._process is not None
            and self._process.stdout is not None
            and self._process.stderr is not None
            and self._event_loop is not None
        )

        # Started, not awaited: the pump now runs for the life of the process, so
        # awaiting it would mean never reading a frame. Waiting on the event instead
        # keeps the old ordering guarantee — geometry is known before the frame reader
        # below sizes itself against it — without tying the two lifetimes together.
        # The thread ends on its own when the process is reaped and stderr hits EOF.
        self._stderr_task = asyncio.ensure_future(asyncio.to_thread(self._pump_stderr))
        await self._metadata_event.wait()

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        transport, _ = await self._event_loop.connect_read_pipe(lambda: protocol, self._process.stdout)
        self._read_transport = transport

        logger.info(f"Starting stream [{self._source}] {self._width}x{self._height}@{self._fps}fps")

        self._is_frame_aligned = bool(self._output_args.get("format") == "rawvideo" and self._frame_size)

        if self._is_frame_aligned:
            await self._worker_loop_frame_aligned(reader)
        else:
            await self._worker_loop_chunked(reader)

        logger.info(f"Worker for {self._source} stopped")

    async def _worker_loop_frame_aligned(self, reader: asyncio.StreamReader) -> None:
        """Reads whole raw frames from stdout, one queue item per frame."""
        assert self._frame_size is not None
        while not self._stop_event.is_set():
            try:
                frame_bytes = await reader.readexactly(self._frame_size)
            except asyncio.IncompleteReadError as e:
                if e.partial:
                    logger.warning(
                        f"Incomplete final frame ({len(e.partial)}/{self._frame_size} bytes) for {self._source}, "
                        "treating as end-of-stream"
                    )
                logger.error(f"No more full frames available, exiting worker for {self._source}")
                break
            except Exception as e:
                logger.error(e)
                raise
            await self._put_dropping_oldest(frame_bytes)

    async def _worker_loop_chunked(self, reader: asyncio.StreamReader) -> None:
        """Reads arbitrary-size chunks from stdout (non-rawvideo outputs, e.g. codec=copy)."""
        while not self._stop_event.is_set():
            try:
                in_bytes = await reader.read(READ_CHUNK_SIZE)
            except Exception as e:
                logger.error(e)
                raise
            if not in_bytes:
                logger.error(f"No bytes returned, exiting worker for {self._source}")
                break
            await self._put_dropping_oldest(in_bytes)
