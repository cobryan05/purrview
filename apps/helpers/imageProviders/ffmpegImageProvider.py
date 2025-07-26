"""FFMpeg-backed Image Provider class"""

import logging
from typing import Any, Optional, Tuple

import numpy as np

from apps.helpers.streams.ffmpegStreamerIn import FFmpegStreamerIn

from .imageProvider import ImageProvider
from .Registry import register_image_provider
import asyncio

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

@register_image_provider()
class FfmpegImageProvider(ImageProvider):
    def __init__(self, video_path: str, loop: bool = True, rtsp_relay: bool = True):
        output_args = {"format": "rawvideo", "pix_fmt": "rgb24", "codec": "rawvideo"}  # decode raw frames
        self._stream: FFmpegStreamerIn = FFmpegStreamerIn(video_path, output_args=output_args)
        self._loop: bool = loop
        self._rtsp_relay: bool = rtsp_relay
        self._stream.start()
        self._frame_size: Optional[Tuple[int, int]] = None
        self._buffer: bytes = b""

    def __repr__(self):
        return f"FfmpegImageProvider [{self._stream}]"

    async def stop(self) -> None:
        await asyncio.to_thread(self._stream.stop)

    async def getNextImage(self) -> np.array:
        if self._frame_size is None:
            await self._stream.wait_for_metadata()
            self._frame_size = self._stream.get_frame_size()
        width, height = self._frame_size
        frame_size_bytes = width * height * 3
        while True:
            chunk: bytes = await self._stream.read_async()
            if not chunk and self._loop:  # If no data is returned, restart the stream
                self._stream.stop()
                self._stream.start()
                self._buffer = b""  # Clear the buffer
                logger.debug(f"Looping stream {self._stream}")
                continue
            self._buffer += chunk

            if len(self._buffer) >= frame_size_bytes:
                # Extract one full frame from the buffer
                frame_bytes = self._buffer[:frame_size_bytes]
                self._buffer = self._buffer[frame_size_bytes:]  # Keep remaining data in the buffer

                image = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
                return image


    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        """
        Return a schema describing the parameters for this Task.
        Each key is a parameter name, value is a dict with:
            - type: str
            - required: bool
            - default: Any (optional)
            - help: str (optional)
            - options: list (optional, for enums)
            - schema: dict (optional, for nested objects)
        """
        return {
            "video_path": {
                "type": "string",
                "required": True,
                "description": "Path to pass to ffmpeg (eg, rtsp://, local url. etc)"
            },
            "rtsp_relay": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Proxy RTSP streams through MediaMTX"
            },
            "loop": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Whether to loop the video"
            }
        }
