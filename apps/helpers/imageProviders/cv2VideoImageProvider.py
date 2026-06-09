"""CV2 VideoCapture ImageProvider"""

import asyncio
import glob
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import cv2

from apps.helpers.imageUtils import base64_encode_png

from .imageProvider import ImageProvider, ImageProviderType, ImageWithProviderMetadata
from .Registry import register_image_provider

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@register_image_provider()
class Cv2VideoImageProvider(ImageProvider):
    PREFETCH_QUEUE_SIZE = 16

    def provider_type(self) -> ImageProviderType:
        return ImageProviderType.LIVE_STREAM if self._loop else ImageProviderType.FIXED

    def __init__(
        self,
        rootdir: str = "",
        loop: bool = False,
        frame_stride: int = 1,
        recursive: bool = True,
        globs: list[str] = None,
    ):
        # The file browser returns paths relative to FILES_ROOT (e.g. "videos/cats").
        # Absolute-looking paths (e.g. "/videos" at root) are also FILES_ROOT-relative,
        # not filesystem root — strip the anchor and resolve under FILES_ROOT.
        _files_root = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))
        rootdir_path = Path(rootdir)
        if not rootdir_path.is_absolute():
            rootdir_path = _files_root / rootdir_path
        else:
            rootdir_path = _files_root / rootdir_path.relative_to(rootdir_path.anchor)
        paths = [str(rootdir_path / g) for g in (globs or ["**/*"])]

        paths_expanded: list[str] = []
        for pattern in paths:
            paths_expanded.extend(glob.glob(pattern, recursive=recursive))

        self._paths: list[str] = paths_expanded
        self._iter: Iterator = iter(self._paths)
        self._vidPath: str = None
        self._vid: Optional[cv2.VideoCapture] = None
        self._vidFrameCnt: int = 0
        self._loop: bool = loop
        self._frame_stride: int = max(1, int(frame_stride))
        # Set to True before start() to have the prefetch worker base64-encode each frame
        # so that encode cost is pipelined with the previous frame's inference.
        self.pre_encode: bool = False
        # Start with 1-frame-per-file estimate; _refine_total_frames() updates it lazily
        # as the prefetch worker opens each file and discovers its real frame count.
        self._total_frames: int = len(self._paths)
        self._nextVideo()

    def __repr__(self):
        return f"Cv2VideoImageProvider [{self._vidPath}]"

    def get_total_frames(self) -> Optional[int]:
        """Current best estimate of total frames to be emitted. May grow as video files are opened.
        Returns None for looping providers (unbounded) or when no paths were matched."""
        if self._loop or not self._paths:
            return None
        return self._total_frames

    def _refine_total_frames(self) -> None:
        """Replace the 1-frame placeholder for the current file with its real strided frame count.
        Called by the prefetch worker each time a new file is opened — O(1), no seek needed."""
        if self._vid is None:
            return
        raw = int(self._vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if raw > 1:  # genuine video file; image files report 0 or 1
            strided = (raw + self._frame_stride - 1) // self._frame_stride
            self._total_frames += strided - 1  # replace the 1-frame placeholder

    def _advance_stride(self) -> None:
        """Seek to the next strided frame after a read. _vidFrameCnt was just incremented
        by _frame_stride so it already equals the target capture position."""
        if self._frame_stride > 1 and self._vid is not None:
            self._vid.set(cv2.CAP_PROP_POS_FRAMES, self._vidFrameCnt)

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
        """Reads frames off the event loop, applying frame_stride, and buffers them for getNextImage()."""
        # Refine total frame count for the first file (already opened by __init__)
        self._refine_total_frames()
        try:
            while self._vid is not None:
                ret, frame = await asyncio.to_thread(self._vid.read)
                if not ret:
                    await asyncio.to_thread(self._nextVideo)
                    if self._vid is None:
                        break
                    self._refine_total_frames()
                    ret, frame = await asyncio.to_thread(self._vid.read)
                    if not ret:
                        break
                    self._vidFrameCnt = 0

                img = ImageWithProviderMetadata(frame)
                img.metadata.source = self._vidPath
                img.metadata.frame_idx = self._vidFrameCnt
                self._vidFrameCnt += self._frame_stride
                if self.pre_encode:
                    img.encoded_b64 = await asyncio.to_thread(base64_encode_png, frame)
                await self._prefetch_queue.put(img)

                self._advance_stride()

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
            self._vidFrameCnt += self._frame_stride

            self._advance_stride()
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
            "rootdir": {
                "type": "directory",
                "label": "Root Directory",
                "required": True,
                "description": "Root directory to search for media files.",
            },
            "globs": {
                "type": "array",
                "label": "Glob Patterns",
                "required": False,
                "default": ["**/*"],
                "items": {
                    "type": "string",
                    "description": "Pattern relative to root directory (e.g. **/*.mp4, *.jpg).",
                },
            },
            "loop": {
                "label": "Loop playlist",
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "Whether to loop the video playlist or not.",
            },
            "frame_stride": {
                "label": "Frame Stride",
                "type": "int",
                "required": False,
                "default": 1,
                "description": "Collect every Nth frame from video files (e.g. 30 ≈ 1 fps at 30 fps). Image files in the glob are always collected.",
            },
            "recursive": {
                "label": "Recursive Glob",
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Whether glob patterns recurse into subdirectories (e.g. **/*.mp4).",
            },
        }
