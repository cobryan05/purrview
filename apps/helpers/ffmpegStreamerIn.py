import ffmpeg
import threading
import queue
import numpy as np
import subprocess
import logging
from fractions import Fraction
from typing import Any, Dict, Optional, Tuple, List

logging.basicConfig()
logger = logging.getLogger(__file__)

READ_CHUNK_SIZE = 16*1024

class FFmpegStreamerIn:
    def __init__(
        self,
        source: str,
        input_args: Optional[List[str]] = None,
    ) -> None:

        self._extra_args: Dict[str, Any] = {}
        if source.startswith("http://") or source.startswith("rtsp://") or source.startswith("https://"):
            self._extra_args = {"rtsp_transport": "tcp", "format": "rtsp", "use_wallclock_as_timestamps": 1}

        self._source: str = source
        self._input_args: List[str] = input_args or []

        self._pixel_format: Optional[str] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._fps: Optional[float] = None
        self._frame_size: Optional[int] = None

        self._process: Optional[subprocess.Popen] = None
        self._frame_queue: queue.Queue[bytes] = queue.Queue()
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread = threading.Thread(target=self._reader_thread, daemon=True)

        self._metadata_event = threading.Event()

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
                        if "x" in part and "fps" not in part:
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

    def get_fps(self) -> float:
        """Returns the FPS of the source stream (waits until it's available)."""
        self._metadata_event.wait()
        if self._fps is None:
            raise RuntimeError("FPS not available yet.")
        return self._fps

    def get_frame_size(self) -> Tuple[int, int]:
        """Returns the frame size (width, height) of the source stream (waits until it's available)."""
        self._metadata_event.wait()
        if self._width is None or self._height is None:
            raise RuntimeError("Frame size not available yet.")
        return self._width, self._height

    def get_pixel_format(self) -> str:
        self._metadata_event.wait()
        if self._pixel_format is None:
            raise RuntimeError("Frame size not available yet.")
        return self._pixel_format

    def start(self) -> None:
        """Starts the FFmpeg process and the background thread."""
        input_kwargs = dict()
        for i in range(0, len(self._input_args), 2):
            key = self._input_args[i].lstrip("-")
            value = self._input_args[i + 1] if i + 1 < len(self._input_args) else None
            input_kwargs[key] = value

        self._process = (
            ffmpeg.input(self._source, **input_kwargs, **self._extra_args)
            .output("pipe:1", codec="copy", format="nut")
            .global_args("-nostats", "-loglevel", "info")  # Set loglevel to info to ensure dimensions and fps present
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        # Start frame reading thread
        self._thread.start()

    def _reader_thread(self) -> None:
        """Reads frames from the source and puts them into the frame queue."""
        assert self._process is not None and self._process.stdout is not None and self._process.stderr is not None

        # Extract metadata from stderr
        self._extract_metadata_from_stderr()

        logger.info(f"Starting stream {self._width}x{self._height}@{self._fps}fps")

        # Read frames from stdout
        while not self._stop_event.is_set():
            in_bytes = self._process.stdout.read(READ_CHUNK_SIZE)
            if not in_bytes:
                break
            self._frame_queue.put(in_bytes)

    def read(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """Returns the next frame from the queue, or None if no frame is available within the timeout."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stops the streamer and terminates the process."""
        self._stop_event.set()
        if self._process:
            self._process.terminate()
        self._thread.join()
