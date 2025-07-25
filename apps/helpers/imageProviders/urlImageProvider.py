"""URL-backed image provider"""

import asyncio
import base64
import logging
import random
import ssl
import urllib.request
from typing import Any

import cv2
import numpy as np

from .imageProvider import ImageProvider
from .Registry import register_image_provider


@register_image_provider()
class UrlImageProvider(ImageProvider):
    def __init__(self, url: str, user: str = None, password: str = None, bust_cache: bool = False):
        self._url: str = url
        self._request = urllib.request.Request(url)
        if bust_cache:
            separator = '&' if '?' in url else '?'
            self._url += f"{separator}{random.randint(100000, 999999)}"
        if user:
            base64String = base64.b64encode(bytes(f"{user}:{password}", encoding="utf8"))
            self._request.add_header("Authorization", f"Basic {base64String.decode()}")

    def __repr__(self):
        return f"UrlImageProvider [{self._url}]"

    async def getNextImage(self) -> np.array:
        return await asyncio.to_thread(self._downloadImage)

    def _downloadImage(self):
        req = urllib.request.urlopen(self._request, context=ssl._create_unverified_context())
        buffer = np.array(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)


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
            "url": {
                "type": "string",
                "required": True,
                "description": "URL of the image"
            },
            "user": {
                "type": "string",
                "required": False,
                "description": "Username for authentication"
            },
            "password": {
                "type": "string",
                "required": False,
                "description": "Password for authentication"
            },
            "bust_cache": {
                "type": "boolean",
                "required": False,
                "description": "Append random number to url to avoid caching"
            }
        }
