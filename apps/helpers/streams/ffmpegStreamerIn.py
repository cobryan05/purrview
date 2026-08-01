import asyncio
import logging
from subprocess import Popen
from typing import Any, Dict, Optional, Tuple

import ffmpeg

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

READ_CHUNK_SIZE = 16 * 1024


class FFmpegStreamerIn:
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
        self._task: Optional[asyncio.Task] = None
        self._event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._is_frame_aligned: bool = False

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

    def stop(self) -> None:
        """Stops the streamer and terminates the process. Safe to call multiple times."""
        self._stop_event.set()
        if self._process is None:
            return
        self._process.terminate()
        self._process.wait()
        self._process = None

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
        self._is_frame_aligned = False

        self._task = self._event_loop.create_task(self._worker_task())

    async def astop(self) -> None:
        """Async, non-blocking equivalent of stop(). Terminates the process and cancels/awaits
        the worker task without blocking the event loop."""
        self._stop_event.set()

        if self._process is not None:
            process = self._process
            self._process = None
            process.terminate()
            await asyncio.to_thread(process.wait)

        if self._task is not None:
            task = self._task
            self._task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Worker task for {self._source} raised during astop: {e}")

    def _extract_metadata_from_stderr(self) -> None:
        """Extracts metadata from the FFmpeg process's stderr output."""
        assert self._process is not None and self._process.stderr is not None
        try:
            for line in iter(self._process.stderr.readline, b""):
                line = line.decode("utf-8").strip()
                logger.debug(f"FFmpeg stderr: {line}")

                # Parse resolution
                if "Stream #" in line and "Video:" in line:
                    parts = line.split(",")
                    for part in parts:
                        if "x" in part and "fps" not in part and " " not in part.strip():
                            resolution = part.strip().split(" ")[0]
                            self._width, self._height = map(int, resolution.split("x"))
                        elif "fps" in part:
                            self._fps = float(part.strip().split(" ")[0])
                        elif "rgb" in part or "yuv" in part:
                            self._pixel_format = part.strip()

                    # If all metadata is found, set the event
                    if self._width and self._height and self._fps and self._pixel_format:
                        self._frame_size = self._width * self._height * 3  # Assuming rgb24
                        self._metadata_event.set()
                        break
        except Exception as e:
            logger.error(f"Error while extracting metadata: {e}")
        finally:
            self._metadata_event.set()

    async def _worker_task(self) -> None:
        """Reads frames from the source and puts them into the frame queue."""
        assert (
            self._process is not None
            and self._process.stdout is not None
            and self._process.stderr is not None
            and self._event_loop is not None
        )

        # Extract metadata from stderr
        await asyncio.to_thread(self._extract_metadata_from_stderr)

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await self._event_loop.connect_read_pipe(lambda: protocol, self._process.stdout)

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
