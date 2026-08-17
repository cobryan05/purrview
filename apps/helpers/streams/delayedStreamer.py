import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

from apps.helpers.logUtils import LOG_LEVEL

from .ffmpegStreamerIn import FFmpegStreamerIn
from .ffmpegStreamerOut import FFmpegStreamerOut

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


STREAM_START_TIMEOUT = 60.0  # Timeout for initial stream read
READ_TIMEOUT = 10.0  # Timeout for subsequent stream reads
MAX_NO_FRAME_TIME = 5.0  # Max time between frames even during initial delay
DELAY_STEP_SIZE = 0.1  # Increment step for increasing delay up to target delay

# Bound on buffered chunks. Chunks are ~16 KB, so this is roughly 32 MB per stream —
# enough to ride out an output stall, small enough that a permanently stalled output
# can't consume the box. Previously unbounded, which meant a stalled output pipe grew
# RAM without limit.
MAX_QUEUED_CHUNKS = 2048


class DelayedStreamer:
    """Re-publishes a stream on a wallclock delay.

    **Not part of the NVR recording path.** Event clips come from MediaMTX's recorded
    rolling buffer via the playback server, which gives retroactive, per-event pre-roll;
    a delayed stream only ever offers "the live feed, permanently N seconds late", can't
    produce a clip for an event that has already finished, and costs two long-lived
    ffmpeg processes per camera. This class remains solely to serve a live *delayed
    view* in a browser. Do not build recording on it.
    """

    @dataclass
    class QueueItem:
        data: bytes
        target_time: float

    def __init__(self, source: FFmpegStreamerIn, dest: FFmpegStreamerOut, delay: float) -> None:
        self._source: FFmpegStreamerIn = source
        self._dest: FFmpegStreamerOut = dest
        self._delay: float = delay
        self._data_queue: asyncio.Queue[DelayedStreamer.QueueItem] = asyncio.Queue(maxsize=MAX_QUEUED_CHUNKS)
        self._stop_event: asyncio.Event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._thread_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        # Counters a consumer can read to tell whether the delay guarantee is intact.
        # A silently-reset delay is the failure mode that made this class unusable as a
        # pre-roll mechanism: it stopped being delayed exactly when it mattered.
        self.dropped_chunks: int = 0
        self.delay_resets: int = 0

    @property
    def delay_intact(self) -> bool:
        """False once an output stall has forced the delay back to zero.

        Consumers must check this rather than assuming the configured delay still
        holds; the ramp-back-up takes DELAY_STEP_SIZE per written chunk.
        """
        return self.delay_resets == 0

    def start(self) -> None:
        """Starts the background task to manage frame reading and buffering."""
        self._event_loop = asyncio.get_running_loop()
        self._task = self._event_loop.create_task(self._worker_task())  # Schedule a new task

    def stop(self) -> None:
        """Stops the background task"""
        self._stop_event.set()

    async def _await_in_thread(self, func: Callable, *args, **kwargs):
        """Run a blocking function on the background thread"""
        return await self._event_loop.run_in_executor(self._thread_executor, func, *args, **kwargs)

    async def _worker_task(self) -> None:
        """Reads frames from the source and pushes them into the frame queue."""
        self._source.start()
        self._dest.start()

        self._data_queue = asyncio.Queue(maxsize=MAX_QUEUED_CHUNKS)

        next_item: Optional[DelayedStreamer.QueueItem] = None
        last_write_timestamp: float = 0.0

        # Slowly increase the delay to the target delay
        current_delay: float = 0.0

        write_error_cnt: int = 0
        read_error_cnt: int = 0
        read_stream_timeout: Optional[float] = time.time() + STREAM_START_TIMEOUT
        data: Optional[bytes] = None

        while not self._stop_event.is_set():
            try:
                try:
                    if read_stream_timeout is None:
                        read_stream_timeout = time.time() + READ_TIMEOUT
                    data = await self._source.read_async(timeout=min(MAX_NO_FRAME_TIME, MAX_NO_FRAME_TIME))
                    read_stream_timeout = None
                except asyncio.TimeoutError:
                    data = None
                    now = time.time()
                    if now > read_stream_timeout:
                        read_error_cnt += 1
                        logger.warning(f"Read stream failed (cnt: {read_error_cnt}).")
                        self._source.stop()
                        self._source.start()
                        read_stream_timeout = now + STREAM_START_TIMEOUT

                if data is not None:
                    item = DelayedStreamer.QueueItem(data=data, target_time=time.time() + current_delay)
                    if next_item is None:
                        next_item = item
                    else:
                        # Drop-oldest rather than await put(): blocking here would stall
                        # the read side too, so a stalled output would back pressure into
                        # the camera connection instead of just degrading this view.
                        while True:
                            try:
                                self._data_queue.put_nowait(item)
                                break
                            except asyncio.QueueFull:
                                try:
                                    self._data_queue.get_nowait()
                                    self._data_queue.task_done()
                                    self.dropped_chunks += 1
                                except asyncio.QueueEmpty:
                                    break

                now = time.time()
                if next_item and (now >= next_item.target_time or now > last_write_timestamp + MAX_NO_FRAME_TIME):
                    try:
                        await self._await_in_thread(self._dest.write, next_item.data)
                    except BrokenPipeError:
                        write_error_cnt += 1
                        self._dest.stop()
                        self._dest.start()
                        # The stream is briefly NOT delayed after this, and ramps back
                        # up only DELAY_STEP_SIZE per written chunk. Surface it loudly
                        # and record it on `delay_resets` — a caller that assumed the
                        # delay still held would be silently wrong.
                        current_delay = 0.0
                        self.delay_resets += 1
                        logger.warning(
                            f"Publish stream failed (cnt: {write_error_cnt}); delay reset to 0 "
                            f"and ramping back to {self._delay}s — the delay guarantee is not "
                            f"held until it recovers (resets: {self.delay_resets})."
                        )

                    now = time.time()
                    last_write_timestamp = now

                    # Slowly ramp up delay so that there are some initial frames
                    if current_delay < self._delay:
                        current_delay = min(self._delay, current_delay + DELAY_STEP_SIZE)

                    try:
                        next_item = self._data_queue.get_nowait()
                        self._data_queue.task_done()
                    except asyncio.QueueEmpty:
                        next_item = None
            except Exception as e:
                # TODO: Streamer reset API for manager to reset?
                logger.error(f"Error  {e}")

        self._source.stop()
        self._dest.stop()
