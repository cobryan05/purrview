import ffmpeg
import logging
import time
import subprocess
from collections import deque
from threading import Event
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

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
        self._out_proc: Optional[subprocess.Popen] = None

        if isinstance(dest, str):
            # If destination is a string, check if it's a URL or file path
            if dest.startswith("http://") or dest.startswith("rtsp://") or dest.startswith("https://"):
                self._extra_args = {"rtsp_transport": "tcp", "format": "rtsp"}

    def start(self):
        """
        Start the output stream
        """
        self._ffmpeg_out = (
            ffmpeg.input("pipe:")
            .output(self._dest, codec="copy", **self._extra_args)
            .global_args("-nostats")  # , "-loglevel", "debug")
        )

        # Using ffmpeg-python to run the process asynchronously
        self._out_proc = self._ffmpeg_out.run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)

    def stop(self):
        """Stop the output stream"""
        self._stop_event.set()
        if self._out_proc.stdin:
            self._out_proc.stdin.close()
        self._out_proc.wait()

    def write(self, data: bytes):
        """Write data to the output stream"""
        if not self._stop_event.is_set():
            try:
                self._out_proc.stdin.write(data)
            except BrokenPipeError:
                raise Exception(self._out_proc.stderr.read())
