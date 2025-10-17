"""CV2 VideoCapture ImageProvider"""

import asyncio
import glob
import logging
from collections.abc import Iterator
from threading import Event
from typing import Any, Optional

import cv2
import numpy as np

from .imageProvider import ImageProvider, ImageWithMetadata
from .Registry import register_image_provider

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

@register_image_provider()
class Cv2VideoImageProvider(ImageProvider):

    def __init__(self, paths: list[str]):
        if isinstance(paths, str):
            paths = [paths]

        paths_expanded: list[str] = []
        for pattern in paths:
            paths_expanded.extend(glob.glob(pattern, recursive=True))

        self._paths: list[str] = paths_expanded
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

    async def getNextImage(self) -> Optional[ImageWithMetadata]:
        while True:
            ret, frame = await asyncio.to_thread(self._vid.read)
            if not ret:
                await asyncio.to_thread(self._nextVideo)
                ret, frame = await asyncio.to_thread(self._vid.read)
                if not ret:
                    return None
            return ImageWithMetadata(frame)


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
            "paths": {
                "type": "array",
                "label": "Media Paths",
                "required": True,
                "items": {
                    "type": "string",
                    "description": "File paths to image or video files, accepts glob patterns."
                }
            }
        }
