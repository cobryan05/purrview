"""CV2 VideoCapture ImageProvider"""

from .imageProvider import ImageProvider
from collections.abc import Iterator
from threading import Event
import asyncio
import cv2
import logging
import numpy as np

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Cv2VideoImageProvider(ImageProvider):

    def __init__(self, paths: list[str]):
        if isinstance(paths, str):
            paths = [paths]

        self._paths: list[str] = paths
        self._iter: Iterator = iter(self._paths)
        self._vidPath: str = None
        self._vid: cv2.VideoCapture = None
        self._nextVideo()

    def __repr__(self):
        return f"Cv2VideoImageProvider [{self._vidPath}]"

    def _nextVideo(self):
        if self._vid:
            self._vid.release()
        self._vidPath = next(self._iter, None)

        # Reset iterator when we get to the end
        if not self._vidPath:
            self._iter = iter(self._paths)
            self._vidPath = next(self._iter)
        self._vid = cv2.VideoCapture(self._vidPath)
        logger.debug(f"Next video: {self._vidPath}")

    async def getNextImage(self) -> np.array:
        while True:
            ret, frame = await asyncio.to_thread(self._vid.read)
            if not ret:
                await asyncio.to_thread(self._nextVideo)
                ret, frame = await asyncio.to_thread(self._vid.read)
                if not ret:
                    return None
            return frame
