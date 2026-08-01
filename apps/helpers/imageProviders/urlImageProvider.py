"""URL-backed image provider"""

import asyncio
import base64
import random
import ssl
import time
import urllib.request
from typing import Any, Optional

import cv2
import numpy as np

from .imageProvider import (
    ImageProvider,
    ImageProviderMetadata,
    ImageProviderType,
    ImageWithProviderMetadata,
)
from .Registry import register_image_provider


@register_image_provider()
class UrlImageProvider(ImageProvider):
    # Injectable seams for tests (patch on the class/instance instead of the global
    # asyncio/time modules, which are shared process-wide and unsafe to monkeypatch
    # while other tests may be running concurrently).
    _sleep = staticmethod(asyncio.sleep)
    _now = staticmethod(time.monotonic)

    @classmethod
    def provider_type(cls) -> ImageProviderType:
        return ImageProviderType.LIVE_SNAPSHOT

    def __init__(
        self,
        url: str,
        user: str = None,
        password: str = None,
        bust_cache: bool = False,
        ignore_ssl_errors: bool = True,
        min_capture_interval: float = 0.0,
    ):
        self._url: str = url
        self._request = urllib.request.Request(url)
        self._ignore_ssl_errors: bool = ignore_ssl_errors
        self._min_capture_interval: float = min_capture_interval
        self._next_capture_time: float = 0.0
        self._frame_idx: int = 0
        if bust_cache:
            separator = "&" if "?" in url else "?"
            self._url += f"{separator}{random.randint(100000, 999999)}"
        if user:
            base64String = base64.b64encode(bytes(f"{user}:{password}", encoding="utf8"))
            self._request.add_header("Authorization", f"Basic {base64String.decode()}")

    def __repr__(self):
        return f"UrlImageProvider [{self._url}]"

    async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
        if self._min_capture_interval > 0:
            now = self._now()
            if now < self._next_capture_time:
                await self._sleep(self._next_capture_time - now)

        frame = await asyncio.to_thread(self._downloadImage)

        if self._min_capture_interval > 0:
            self._next_capture_time = self._now() + self._min_capture_interval

        frame_idx = self._frame_idx
        self._frame_idx += 1
        return ImageWithProviderMetadata(
            frame, metadata=ImageProviderMetadata(source=self.source_name, frame_idx=frame_idx)
        )

    def _downloadImage(self):
        if self._ignore_ssl_errors:
            ssl_context = ssl._create_unverified_context()
            req = urllib.request.urlopen(self._request, context=ssl_context)
        else:
            req = urllib.request.urlopen(self._request)

        buffer = np.array(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        """
        Return a schema describing the parameters for this Provider.
        Each key is a parameter name, value is a dict with:
            - type: str
            - required: bool
            - default: Any (optional)
            - help: str (optional)
            - options: list (optional, for enums)
            - schema: dict (optional, for nested objects)
        """
        return {
            "url": {
                "type": "string",
                "label": "URL",
                "required": True,
                "description": "URL of the image",
            },
            "user": {
                "type": "string",
                "label": "Username",
                "required": False,
                "description": "Username for authentication",
            },
            "password": {
                "type": "string",
                "label": "Password",
                "required": False,
                "description": "Password for authentication",
            },
            "bust_cache": {
                "type": "boolean",
                "label": "Bust Cache",
                "required": False,
                "description": "Append random number to url to avoid caching",
            },
            "ignore_ssl_errors": {
                "type": "boolean",
                "label": "Ignore SSL Errors",
                "required": False,
                "description": "Ignore SSL certificate errors",
                "default": True,
            },
            "min_capture_interval": {
                "label": "Min Capture Interval (s)",
                "type": "float",
                "required": False,
                "default": 0.0,
                "description": "Minimum seconds between frames returned by this source. 0 = no source-level limit (the task's own interval applies).",
            },
        }
