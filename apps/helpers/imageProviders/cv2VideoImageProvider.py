"""CV2 VideoCapture ImageProvider"""

import asyncio
import glob
import logging
from collections.abc import Iterator
from typing import Any, Optional

import cv2

from .imageProvider import ImageProvider, ImageWithProviderMetadata
from .Registry import register_image_provider

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@register_image_provider()
class Cv2VideoImageProvider(ImageProvider):
    PREFETCH_QUEUE_SIZE = 2

    def __init__(self, paths: list[str], loop: bool = False):
        if isinstance(paths, str):
            paths = [paths]

        paths_expanded: list[str] = []
        for pattern in paths:
            paths_expanded.extend(glob.glob(pattern, recursive=True))

        self._paths: list[str] = paths_expanded
        self._iter: Iterator = iter(self._paths)
        self._vidPath: str = None
        self._vid: Optional[cv2.VideoCapture]= None
        self._vidFrameCnt: int = 0
        self._loop: bool = loop
        self._nextVideo()

    def __repr__(self):
        return f"Cv2VideoImageProvider [{self._vidPath}]"

    def _nextVideo(self):
        if self._vid:
            self._vid.release()
        self._vidPath = next(self._iter, None)

        # Reset iterator when we get to the end
        if not self._vidPath:
            if self._loop:
                self._iter = iter(self._paths)
                self._vidPath = next(self._iter)

        if self._vidPath:
            self._vid = cv2.VideoCapture(self._vidPath)
            logger.debug(f"Next video: {self._vidPath}")
        else:
            self._vid = None

    async def start(self) -> None:
        self._prefetch_queue: asyncio.Queue = asyncio.Queue(maxsize=Cv2VideoImageProvider.PREFETCH_QUEUE_SIZE)
        self._prefetch_task: asyncio.Task = asyncio.create_task(self._prefetch_worker())

    async def stop(self) -> None:
        task = getattr(self, "_prefetch_task", None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._prefetch_task = None

    async def _prefetch_worker(self) -> None:
        """Reads frames off the event loop and buffers them for getNextImage()."""
        try:
            while self._vid is not None:
                ret, frame = await asyncio.to_thread(self._vid.read)
                if not ret:
                    await asyncio.to_thread(self._nextVideo)
                    if self._vid is None:
                        break
                    ret, frame = await asyncio.to_thread(self._vid.read)
                    if not ret:
                        break
                    self._vidFrameCnt = 0

                img = ImageWithProviderMetadata(frame)
                img.metadata.source = self._vidPath
                img.metadata.frame_idx = self._vidFrameCnt
                self._vidFrameCnt += 1
                await self._prefetch_queue.put(img)
        except asyncio.CancelledError:
            raise
        finally:
            # Sentinel so any waiting getNextImage() call unblocks cleanly.
            # put_nowait is fine here: if the queue is full the caller has already exited.
            try:
                self._prefetch_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
        if not hasattr(self, "_prefetch_queue"):
            # start() was not called — fall back to direct (unbuffered) read
            return await self._direct_read()
        return await self._prefetch_queue.get()

    async def _direct_read(self) -> Optional[ImageWithProviderMetadata]:
        while self._vid is not None:
            ret, frame = await asyncio.to_thread(self._vid.read)
            if not ret:
                await asyncio.to_thread(self._nextVideo)
                if self._vid is None:
                    return None
                ret, frame = await asyncio.to_thread(self._vid.read)
                if not ret:
                    return None
                self._vidFrameCnt = 0

            ret_image = ImageWithProviderMetadata(frame)
            ret_image.metadata.source = self._vidPath
            ret_image.metadata.frame_idx = self._vidFrameCnt
            self._vidFrameCnt += 1
            return ret_image

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
            },
            "loop": {
                "label": "Loop playlist",
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "Whether to loop the video playlist or not.",
            },
        }
