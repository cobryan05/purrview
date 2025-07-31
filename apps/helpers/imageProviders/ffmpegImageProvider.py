"""FFMpeg-backed Image Provider class"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Optional, Tuple

import numpy as np

from apps import APPS_CONFIG
from apps.helpers.streams.ffmpegStreamerIn import FFmpegStreamerIn

from .imageProvider import ImageProvider
from .Registry import register_image_provider

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

if os.getenv("GENERATING_OPENAPI_CLIENTS") != "1":
    import relay_buffer_client


@register_image_provider()
class FfmpegImageProvider(ImageProvider):
    def __init__(self, video_path: str, loop: bool = True, rtsp_relay: bool = True):
        output_args = {"format": "rawvideo", "pix_fmt": "rgb24", "codec": "rawvideo"}  # decode raw frames
        self._video_path: str = video_path.strip()
        self._loop: bool = loop
        self._frame_size: Optional[Tuple[int, int]] = None
        self._buffer: bytes = b""
        self._rtsp_relay_api: Optional[relay_buffer_client.ApiClient] = None
        self._relayed_name: str = ""
        self._stream: Optional[FFmpegStreamerIn] = None
        self._video_path: str = video_path.strip()
        self._ffmpeg_args: dict = output_args
        if rtsp_relay and video_path.lower().startswith("rtsp:"):
            relay_host, relay_port = APPS_CONFIG["relay_buffer_server"].host, APPS_CONFIG["relay_buffer_server"].port
            config = relay_buffer_client.Configuration(f"http://{relay_host}:{relay_port}")
            self._rtsp_relay_api = relay_buffer_client.ApiClient(config)
            self._relayed_name = f"stream-{uuid.uuid4()}"

    def __repr__(self):
        return f"FfmpegImageProvider [{self._stream}]"

    async def start(self) -> None:
        if self._rtsp_relay_api:
            api: relay_buffer_client.StreamsApi = relay_buffer_client.StreamsApi(self._rtsp_relay_api)
            await asyncio.to_thread(api.create_stream, self._video_path, self._relayed_name)

            async def wait_for_stream(
                api, stream_name: str, timeout_ms: int = 10000, poll_interval_ms: int = 250
            ) -> bool:
                deadline = time.monotonic() + timeout_ms / 1000
                while time.monotonic() < deadline:
                    streams = await asyncio.to_thread(api.list_streams)
                    for stream in streams.get("results", []):
                        if stream.get("name") == stream_name:
                            return True
                    await asyncio.sleep(poll_interval_ms / 1000)
                return False

            stream_found = await wait_for_stream(api, self._relayed_name, timeout_ms=10000)
            if not stream_found:
                raise TimeoutError(f"Timed out waiting for '{self._relayed_name}' to appear in relay list")
            await asyncio.sleep(10) # TODO: Wait until bytes received
            stream_url = f"rtsp://{APPS_CONFIG['media_mtx_rtsp'].host}:{APPS_CONFIG['media_mtx_rtsp'].port}/{self._relayed_name}"
        else:
            stream_url = self._video_path

        self._stream = FFmpegStreamerIn(stream_url, output_args=self._ffmpeg_args)
        await asyncio.to_thread(self._stream.start)

    async def stop(self) -> None:
        await asyncio.to_thread(self._stream.stop)
        if self._rtsp_relay_api:
            api: relay_buffer_client.StreamsApi = relay_buffer_client.StreamsApi(self._rtsp_relay_api)
            await asyncio.to_thread(api.destroy_stream, self._relayed_name)

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
                "description": "Path to pass to ffmpeg (eg, rtsp://, local url. etc)",
            },
            "rtsp_relay": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Proxy RTSP streams through MediaMTX",
            },
            "loop": {"type": "boolean", "required": False, "default": True, "description": "Whether to loop the video"},
        }
