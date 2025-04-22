from collections import deque
import numpy as np
import threading
from typing import Deque, Optional
from .ffmpegStreamerIn import FFmpegStreamerIn
from .ffmpegStreamerOut import FFmpegStreamerOut


class DelayedStreamer:
    def __init__(
        self, source: FFmpegStreamerIn, dest: FFmpegStreamerOut, delay: float
    ) -> None:
        self._source: FFmpegStreamerIn = source
        self._dest: FFmpegStreamerOut = dest
        self._delay: float = delay
        self._frame_queue: Optional[Deque[bytes]] = None
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread = threading.Thread(
            target=self._delay_thread, daemon=True
        )

    def start(self) -> None:
        """Starts the background thread to manage frame reading and buffering."""
        self._thread.start()

    def _delay_thread(self) -> None:
        """Reads frames from the source and pushes them into the frame queue."""
        self._source.start()
        self._dest.start()
        fps = self._source.get_fps()
        delay_frame_cnt = int(
            self._delay * fps
        )  # Calculate buffer size based on FPS and delay
        self._frame_queue = deque(maxlen=2 * delay_frame_cnt)

        while not self._stop_event.is_set():
            frame = self._source.read()
            if frame is not None:
                self._frame_queue.append(frame)

            # Wait until the buffer is full or we've accumulated enough frames
            if len(self._frame_queue) >= delay_frame_cnt:
                self._write_to_dest()

        self._source.stop()
        self._dest.stop()

    def _write_to_dest(self) -> None:
        """Writes frames from the buffer to the destination stream."""
        while len(self._frame_queue) > 0:
            frame: bytes = self._frame_queue.popleft()  # Pop the oldest frame
            self._dest.write(frame)  # Write frame to the destination

    def stop(self) -> None:
        """Stops the background thread and stops the process."""
        self._stop_event.set()
        self._thread.join()
