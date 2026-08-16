"""URL-backed image provider"""

import asyncio
import base64
import json
import logging
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

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

DOWNLOAD_TIMEOUT = 10.0


def _unverified_ssl_context() -> ssl.SSLContext:
    """An SSL context that accepts any certificate.

    Only reachable through a source's explicit ``ignore_ssl_errors`` opt-in,
    which exists because IP cameras ship self-signed certs for their LAN
    addresses and there is nothing to validate them against.

    Built from the public ``create_default_context()`` rather than
    ``ssl._create_unverified_context()``: that helper is private, and the docs
    describe it as being for stdlib modules' own use. This is the documented
    way to say the same thing, and it says it in the open — the two disabling
    lines are visible at the call site instead of hidden behind a name.

    Assignment order is load-bearing: ``check_hostname`` must go False *before*
    ``verify_mode`` is relaxed, or CPython raises ValueError.
    """
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


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
        self.min_capture_interval: float = min_capture_interval
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

        frame, raw = await asyncio.to_thread(self._downloadImage)

        if self._min_capture_interval > 0:
            self._next_capture_time = self._now() + self._min_capture_interval

        frame_idx = self._frame_idx
        self._frame_idx += 1
        provided = ImageWithProviderMetadata(
            frame, metadata=ImageProviderMetadata(source=self.source_name, frame_idx=frame_idx)
        )
        # Hand on the bytes the camera actually sent, rather than discarding them and
        # re-encoding the decoded array later. This is the camera's own full-resolution
        # JPEG: it is what makes an event clip from a snapshot camera full-res, and it
        # saves an encode wherever a consumer wanted a JPEG anyway. Only JPEG payloads
        # qualify — the packet declares its format, and a PNG announced as JPEG would be
        # a lie that only surfaces in whatever tries to decode it.
        if raw[:2] == b"\xff\xd8":
            provided.encoded_b64 = json.dumps(
                {"format": "jpeg", "metadata": {}, "data": base64.b64encode(raw).decode("ascii")}
            )
        return provided

    def _downloadImage(self):
        """Fetch and decode one frame, returning `(decoded_bgr, raw_bytes)`.

        The raw bytes are returned rather than dropped so a caller can keep the
        camera's own encoding — see `getNextImage`.
        """
        logger.debug(f"Downloading image from {self._url}")
        try:
            if self._ignore_ssl_errors:
                req = urllib.request.urlopen(self._request, timeout=DOWNLOAD_TIMEOUT, context=_unverified_ssl_context())
            else:
                req = urllib.request.urlopen(self._request, timeout=DOWNLOAD_TIMEOUT)

            raw = req.read()
            buffer = np.frombuffer(raw, dtype=np.uint8)
        except Exception:
            logger.exception(f"Failed to download image from {self._url}")
            raise

        frame = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
        if frame is None:
            logger.error(f"Failed to decode image from {self._url} (response was not a valid image)")
        return frame, raw

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
