"""FFMpeg-backed Image Provider class"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Optional, Tuple

import numpy as np

from apps import APPS, APPS_CONFIG
from apps.helpers.streams.ffmpegStreamerIn import FFmpegStreamerIn

from .imageProvider import (
    ImageProvider,
    ImageProviderMetadata,
    ImageProviderType,
    ImageWithProviderMetadata,
)
from .Registry import register_image_provider

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

if os.getenv("GENERATING_OPENAPI_CLIENTS") != "1":
    import relay_buffer_client

STREAM_START_TIMEOUT = 10.0
STREAM_READ_TIMEOUT = 10.0
FFMPEG_START_TIMEOUT = 30.0


@register_image_provider()
class FfmpegImageProvider(ImageProvider):
    # Bound on consecutive stream-restart attempts inside getNextImage() so a stream that
    # stops yielding data (EOF, dead source, etc.) can't spin astop()/astart() forever.
    _MAX_STREAM_RESTARTS = 3
    _RESTART_BACKOFF = 0.5

    # Injectable seams for tests (patch on the class/instance instead of the global
    # asyncio/time modules, which are shared process-wide and unsafe to monkeypatch
    # while other tests may be running concurrently).
    _sleep = staticmethod(asyncio.sleep)
    _now = staticmethod(time.monotonic)

    @classmethod
    def provider_type(cls) -> ImageProviderType:
        return ImageProviderType.LIVE_STREAM

    def __init__(
        self,
        video_path: str,
        loop: bool = True,
        rtsp_relay: bool = True,
        min_capture_interval: float = 0.0,
    ):
        output_args = {
            "format": "rawvideo",
            "pix_fmt": "rgb24",
            "codec": "rawvideo",
        }  # decode raw frames
        if min_capture_interval > 0:
            # Let ffmpeg itself decimate frames rather than decoding+discarding in Python.
            output_args["vf"] = f"fps=1/{min_capture_interval}"
        self._video_path: str = video_path.strip()
        self._loop: bool = loop
        self._frame_size: Optional[Tuple[int, int]] = None
        self._buffer: bytes = b""
        self._rtsp_relay_api: Optional[relay_buffer_client.ApiClient] = None
        self._relayed_name: str = ""
        self._stream: Optional[FFmpegStreamerIn] = None
        self._video_path: str = video_path.strip()
        self._ffmpeg_args: dict = output_args
        self._min_capture_interval: float = min_capture_interval
        self.min_capture_interval: float = min_capture_interval
        self._next_capture_time: float = 0.0
        self._frame_idx: int = 0
        self._restart_count: int = 0
        if rtsp_relay and video_path.lower().startswith("rtsp:"):
            relay_host, relay_port = (
                APPS_CONFIG[APPS.RELAY_BUFFER_SERVER].host,
                APPS_CONFIG[APPS.RELAY_BUFFER_SERVER].port,
            )
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
                api,
                stream_name: str,
                timeout: float = STREAM_START_TIMEOUT,
                poll_interval: float = 0.5,
            ) -> bool:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    streams = await asyncio.to_thread(api.list_streams)
                    for stream in streams.get("results", []):
                        if stream.get("name") == stream_name:
                            if stream["info"]["mtx_path"]["bytes_received"] > 0:
                                return True
                            break
                    await asyncio.sleep(poll_interval)
                return False

            stream_found = await wait_for_stream(api, self._relayed_name, STREAM_START_TIMEOUT)
            if not stream_found:
                raise TimeoutError(f"Timed out waiting for '{self._relayed_name}' to appear in relay list")
            stream_url = f"rtsp://{APPS_CONFIG[APPS.MEDIA_MTX_RTSP].host}:{APPS_CONFIG[APPS.MEDIA_MTX_RTSP].port}/{self._relayed_name}"
        else:
            stream_url = self._video_path

        self._stream = FFmpegStreamerIn(stream_url, output_args=self._ffmpeg_args)
        await self._stream.astart()

    async def stop(self) -> None:
        if self._stream is not None:
            await self._stream.astop()
        if self._rtsp_relay_api:
            try:
                api: relay_buffer_client.StreamsApi = relay_buffer_client.StreamsApi(self._rtsp_relay_api)
                await asyncio.to_thread(api.destroy_stream, self._relayed_name)
            except Exception as e:
                logger.error(f"Failed to destroy relay stream {self._relayed_name}: {e}")

    async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
        if self._frame_size is None:
            try:
                await asyncio.wait_for(self._stream.wait_for_metadata(), timeout=FFMPEG_START_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error(f"FFmpeg metadata wait timed out after {FFMPEG_START_TIMEOUT} seconds")
                raise
            self._frame_size = self._stream.get_frame_size()
        width, height = self._frame_size
        frame_size_bytes = width * height * 3

        if self._min_capture_interval > 0:
            now = self._now()
            if now < self._next_capture_time:
                await self._sleep(self._next_capture_time - now)

        while True:
            frame_bytes = await self._read_latest_frame(frame_size_bytes)

            if frame_bytes:
                self._restart_count = 0  # a successful frame clears cumulative restart history

                if self._min_capture_interval > 0:
                    self._next_capture_time = self._now() + self._min_capture_interval

                image = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
                frame_idx = self._frame_idx
                self._frame_idx += 1
                return ImageWithProviderMetadata(
                    image, metadata=ImageProviderMetadata(source=self.source_name, frame_idx=frame_idx)
                )

            if not self._loop:
                return None

            self._restart_count += 1
            if self._restart_count > self._MAX_STREAM_RESTARTS:
                logger.error(
                    f"Giving up on stream {self._stream} after {self._MAX_STREAM_RESTARTS} consecutive "
                    "restart attempts with no data"
                )
                return None

            logger.debug(
                f"Looping stream {self._stream} (restart attempt {self._restart_count}/{self._MAX_STREAM_RESTARTS})"
            )
            await self._stream.astop()
            await self._stream.astart()
            self._buffer = b""  # Clear the buffer
            await self._sleep(self._RESTART_BACKOFF)

    async def _read_latest_frame(self, frame_size_bytes: int) -> Optional[bytes]:
        """Returns the freshest full frame available, draining stale queued frames first.

        When the underlying streamer's queue items are whole frames (the normal case for this
        provider, since output is always rawvideo), drains via read_nowait() to the newest item
        and only blocks on read_async() if nothing was queued. Falls back to chunk accumulation
        if the streamer somehow isn't frame-aligned yet.
        """
        if self._stream.is_frame_aligned():
            latest: Optional[bytes] = None
            drained = self._stream.read_nowait()
            while drained is not None:
                latest = drained
                drained = self._stream.read_nowait()

            if latest is not None:
                return latest

            try:
                return await asyncio.wait_for(self._stream.read_async(), timeout=STREAM_READ_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error(f"FFmpeg stream read timed out after {STREAM_READ_TIMEOUT} seconds")
                return None

        # Not frame-aligned yet (e.g. metadata just arrived) — accumulate chunks the old way.
        while len(self._buffer) < frame_size_bytes:
            try:
                chunk: Optional[bytes] = await asyncio.wait_for(self._stream.read_async(), timeout=STREAM_READ_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error(f"FFmpeg stream read timed out after {STREAM_READ_TIMEOUT} seconds")
                chunk = None
            if not chunk:
                return None
            self._buffer += chunk

        frame_bytes = self._buffer[:frame_size_bytes]
        self._buffer = self._buffer[frame_size_bytes:]
        return frame_bytes

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        """
        Return a schema describing the parameters for this Provider
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
                "label": "Video Source Path",
                "type": "string",
                "required": True,
                "description": "Path to pass to ffmpeg (eg, rtsp://, server-local url path, etc)",
            },
            "rtsp_relay": {
                "label": "Use RTSP Proxy Server",
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Proxy RTSP streams through MediaMTX",
            },
            "loop": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Whether to loop the video",
            },
            "min_capture_interval": {
                "label": "Min Capture Interval (s)",
                "type": "float",
                "required": False,
                "default": 0.0,
                "description": "Minimum seconds between frames returned by this source. 0 = no source-level limit (the task's own interval applies).",
            },
        }
