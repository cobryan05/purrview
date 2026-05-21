import logging
import select
import subprocess
from threading import Event
from typing import Dict, List, Optional, Union

import ffmpeg

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class FFmpegStreamerOut:
    def __init__(self, dest: Union[str, int]):
        """
        :param dest: The output destination. This can be:
            - A URL (e.g., 'rtsp://...')
            - A local file path (e.g., '/path/to/output.mp4')
            - A pipe (int value representing file descriptor, e.g., sys.stdout)
        """
        self._dest = dest
        self._ffmpeg_out: Optional[ffmpeg.nodes.OutputStream] = None
        self._stop_event: Event = Event()
        self._extra_args: Dict[str, str] = {}
        self._error_cnt: int = 0
        self._process: Optional[subprocess.Popen] = None

        if isinstance(dest, str):
            # If destination is a string, check if it's a URL or file path
            if dest.startswith("http://") or dest.startswith("rtsp://") or dest.startswith("https://"):
                self._extra_args = {"rtsp_transport": "tcp", "format": "rtsp"}

    def start(self) -> None:
        """Start the output stream"""
        if self._process:
            raise RuntimeError("Can't start new process while old process is running!")

        self._ffmpeg_out = (
            ffmpeg.input("pipe:")
            .output(self._dest, codec="copy", **self._extra_args)
            .global_args("-nostats")  # , "-loglevel", "debug")
        )

        self._stop_event.clear()
        # Using ffmpeg-python to run the process asynchronously
        self._process = self._ffmpeg_out.run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)

    def stop(self) -> None:
        """Stop the output stream"""
        self._stop_event.set()
        if self._process:
            if self._process.stdin:
                self._process.stdin.close()
            self._process.wait()
            self._process = None

    def write(self, data: bytes) -> None:
        """Write data to the output stream"""
        if not self._stop_event.is_set():
            try:
                # Periodically read from stderr to prevent the pipe from filling up
                output = self._read_stderr()
                if output:
                    logger.debug(output)
                self._process.stdin.write(data)
            except BrokenPipeError:
                logger.warning(self._read_stderr())
                self.stop()
                raise

    def _read_stderr(self) -> str:
        """Non-blocking read from ffmpeg's stderr"""
        output: List[str] = []
        rlist, _, _ = select.select([self._process.stderr], [], [], 0)  # Non-blocking check
        while rlist:
            chunk = self._process.stderr.read(256)
            if not chunk:
                break
            output.append(chunk)
            rlist, _, _ = select.select([self._process.stderr], [], [], 0)  # Check again
        return b"".join(output).decode("utf-8", errors="ignore")
