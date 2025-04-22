import os
from PIL import Image
import numpy as np
from helpers.ffmpegStreamerIn import FFmpegStreamerIn
from helpers.ffmpegStreamerOut import FFmpegStreamerOut
from helpers.delayedStreamer import DelayedStreamer
from typing import List, Optional
import time
import os


class FFmpegStreamerWithSave(FFmpegStreamerIn):
    def __init__(self, source: str, input_args: Optional[List[str]] = None, pixel_format: str = "rgb24") -> None:
        super().__init__(source, input_args, pixel_format)
        self.frame_counter: int = 0
        self.save_dir: str = "/app/logs"  # Directory to save frames
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_frame(self, frame: np.ndarray) -> None:
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame)
        frame_filename = os.path.join(self.save_dir, f"frame_{self.frame_counter}.png")
        image.save(frame_filename)
        self.frame_counter += 1


# Usage
src = FFmpegStreamerIn("rtsp://192.168.3.24:8554/unicast")

dest_file = "/app/logs/out.mkv"
if os.path.exists(dest_file):
    os.remove(dest_file)
dest = FFmpegStreamerOut(dest_file)
delayer = DelayedStreamer(src, dest, 10)
delayer.start()

time.sleep(100)
