"""URL-backed image provider"""

from .imageProvider import ImageProvider
import asyncio
import base64
import cv2
import logging
import numpy as np
import ssl
import urllib.request


class UrlImageProvider(ImageProvider):
    def __init__(self, url: str, user: str = None, password: str = None):
        self._url: str = url
        self._request = urllib.request.Request(url)
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
