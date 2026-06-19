"""CV2 VideoCapture & Direct File ImageProvider"""

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from apps.helpers.hashUtils import phash_from_numpy
from apps.helpers.imageUtils import base64_encode_jpg

from .imageProvider import ImageProvider, ImageProviderType, ImageWithProviderMetadata
from .Registry import register_image_provider

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@register_image_provider()
class Cv2VideoImageProvider(ImageProvider):
    PREFETCH_QUEUE_SIZE = 16
    PARALLEL_READERS = 4  # concurrent image reads in the prefetch worker

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    # Non-media extensions to skip during discovery (e.g. JSON sidecars, annotation files).
    SKIP_EXTENSIONS = frozenset(
        {
            ".json",
            ".txt",
            ".xml",
            ".csv",
            ".yaml",
            ".yml",
            ".md",
            ".db",
            ".sqlite",
            ".log",
        }
    )

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
        # Resolve paths relative to FILES_ROOT
        _files_root = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))
        rootdir_path = Path(rootdir)
        if not rootdir_path.is_absolute():
            rootdir_path = _files_root / rootdir_path
        else:
            rootdir_path = _files_root / rootdir_path.relative_to(rootdir_path.anchor)

        # Async pipeline initializations
        self._paths: list[str] = []
        self._index: int = 0

        self._vidPath: str = None
        self._vid: Optional[cv2.VideoCapture] = None
        self._is_static_image: bool = False
        self._vidFrameCnt: int = 0
        self._loop: bool = loop
        self._frame_stride: int = max(1, int(frame_stride))
        self.pre_encode: bool = False
        self.pre_phash: bool = False
        self.image_hash_size: int = 8

        # Config variables preserved for discovery worker
        self._rootdir_path = rootdir_path
        self._globs = globs or ["**/*"]
        self._recursive = recursive

        self._total_frames: int = 0
        self._discovery_complete: bool = False
        self._discovery_task: Optional[asyncio.Task] = None

    def __repr__(self):
        state = "Static Image" if self._is_static_image else "Video Stream"
        return f"Cv2VideoImageProvider [{state}: {self._vidPath}]"

    def get_total_frames(self) -> Optional[int]:
        if self._loop or not self._paths:
            return None
        return self._total_frames

    def _refine_total_frames(self) -> None:
        """Refines frames only for true video streams."""
        if self._is_static_image or self._vid is None:
            return
        raw = int(self._vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if raw > 1:
            strided = (raw + self._frame_stride - 1) // self._frame_stride
            self._total_frames += strided - 1

    def _advance_stride(self) -> None:
        if not self._is_static_image and self._frame_stride > 1 and self._vid is not None:
            self._vid.set(cv2.CAP_PROP_POS_FRAMES, self._vidFrameCnt)

    async def _discover_files_async(self) -> None:
        """Async directory scan — os.walk in a thread, batches streamed via a queue.

        os.walk is used instead of aiopath.rglob("*") + is_file() because:
        - Its filenames list contains only files; no per-entry stat() needed.
        - Directory/file separation uses d_type from readdir on Linux/macOS — free.
        - A single pass is made regardless of how many glob patterns are configured;
          pattern matching is done in userspace as each file name is encountered.

        Batches are pushed onto an asyncio.Queue as they fill, so the prefetch
        worker can start reading images before the walk finishes.
        """
        from pathlib import PurePath

        BATCH_SIZE = 64
        root = self._rootdir_path
        if not root.is_dir():
            logger.warning("Root directory path does not exist: %s", root)
            return

        catch_all = any(g in {"**/*", "*"} for g in self._globs)
        # Strip leading "**/" before matching: PurePath.match() is right-anchored, so
        # "*.png" already matches any .png at any depth without the "**/" prefix.
        # Keeping "**/" causes files directly in root to fail (no extra component to absorb it).
        patterns = None if catch_all else [g[3:] if g.startswith("**/") else g for g in self._globs]
        use_recursive = self._recursive or any("**" in g for g in self._globs)

        # threading.Event is used (not asyncio.Event) so is_set() is safe to call
        # from the worker thread without GIL caveats.
        stop_event = threading.Event()
        loop = asyncio.get_running_loop()
        # Unbounded so the walk thread never blocks on put; path strings are tiny.
        result_queue: asyncio.Queue = asyncio.Queue()

        def _walk() -> None:
            batch: list[str] = []

            def _flush() -> None:
                if batch:
                    loop.call_soon_threadsafe(result_queue.put_nowait, list(batch))
                    batch.clear()

            def _accept(path_str: str) -> None:
                _, ext = os.path.splitext(path_str.lower())
                if ext in self.SKIP_EXTENSIONS:
                    return
                if patterns is not None:
                    try:
                        rel = PurePath(path_str).relative_to(root)
                    except ValueError:
                        return
                    if not any(rel.match(g) for g in patterns):
                        return
                batch.append(path_str)
                if len(batch) >= BATCH_SIZE:
                    _flush()

            try:
                if use_recursive:
                    for dirpath, dirnames, filenames in os.walk(str(root)):
                        if stop_event.is_set():
                            dirnames.clear()  # prune os.walk — stops further recursion
                            break
                        for fname in filenames:
                            _accept(os.path.join(dirpath, fname))
                else:
                    with os.scandir(str(root)) as it:
                        for entry in it:
                            if stop_event.is_set():
                                break
                            if entry.is_file(follow_symlinks=True):
                                _accept(entry.path)

                if not stop_event.is_set():
                    _flush()
            finally:
                # Always send the sentinel so the consumer never hangs, even on exception.
                loop.call_soon_threadsafe(result_queue.put_nowait, None)

        loop.run_in_executor(None, _walk)

        unique_seen: set[str] = set()
        try:
            while True:
                batch = await result_queue.get()
                if batch is None:
                    break
                for path_str in batch:
                    if path_str not in unique_seen:
                        unique_seen.add(path_str)
                        self._paths.append(path_str)
                        self._total_frames += 1
        except asyncio.CancelledError:
            stop_event.set()
            raise

    async def _background_discovery_worker(self) -> None:
        try:
            await self._discover_files_async()
        except asyncio.CancelledError:
            logger.debug("File discovery aborted.")
            raise
        finally:
            self._discovery_complete = True
            logger.debug(f"Discovery complete. Found {len(self._paths)} unique files.")

    def _nextVideo(self):
        """Release current handle and select the next path. Does not open a new file handle."""
        if self._vid:
            self._vid.release()
            self._vid = None
        self._is_static_image = False

        paths_len = len(self._paths)
        if self._index < paths_len:
            self._vidPath = self._paths[self._index]
            self._index += 1
        elif self._loop and paths_len > 0:
            self._index = 0
            self._vidPath = self._paths[self._index]
            self._index += 1
        else:
            self._vidPath = None

    def _open_current_path(self) -> None:
        """Open the VideoCapture handle for self._vidPath. Blocking — dispatch via asyncio.to_thread."""
        if not self._vidPath:
            return
        _, ext = os.path.splitext(self._vidPath.lower())
        if ext in self.IMAGE_EXTENSIONS:
            self._is_static_image = True
            self._vid = None
            logger.debug(f"Next image (Direct imread): {self._vidPath}")
        else:
            self._is_static_image = False
            self._vid = cv2.VideoCapture(self._vidPath)
            logger.debug(f"Next video (VideoCapture): {self._vidPath}")

    async def _advance_to_next(self) -> None:
        """Select the next path and open its file handle, offloading the blocking cv2.VideoCapture to a thread."""
        self._nextVideo()
        await asyncio.to_thread(self._open_current_path)

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Internal frame access layer (blocking CV2 calls)."""
        if self._is_static_image:
            frame = cv2.imread(self._vidPath)
            return (frame is not None), frame
        elif self._vid is not None:
            return self._vid.read()
        return False, None

    def _session_active(self) -> bool:
        """Determines if the provider still has active files to process."""
        has_more_paths = self._index < len(self._paths)
        return self._is_static_image or (self._vid is not None) or has_more_paths or not self._discovery_complete

    async def start(self) -> None:
        self._discovery_task = asyncio.create_task(self._background_discovery_worker())
        self._prefetch_queue: asyncio.Queue = asyncio.Queue(maxsize=Cv2VideoImageProvider.PREFETCH_QUEUE_SIZE)
        self._prefetch_task: asyncio.Task = asyncio.create_task(self._prefetch_worker())

    async def stop(self) -> None:
        if self._discovery_task and not self._discovery_task.done():
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass

        if hasattr(self, "_prefetch_task") and self._prefetch_task and not self._prefetch_task.done():
            self._prefetch_task.cancel()
            try:
                await self._prefetch_task
            except asyncio.CancelledError:
                pass
        self._prefetch_task = None

    async def _read_image_file(self, path: str) -> Optional[ImageWithProviderMetadata]:
        """Read one static image file off the event loop. Safe to call concurrently."""
        # PERF
        t_read = time.perf_counter()
        # /PERF
        frame = await asyncio.to_thread(cv2.imread, path)
        # PERF
        logger.debug("prefetch: read=%.1fms path=%s", 1000 * (time.perf_counter() - t_read), path)
        # /PERF
        if frame is None:
            logger.warning("cv2.imread returned None: %s", path)
            return None
        img = ImageWithProviderMetadata(frame)
        img.metadata.source = path
        img.metadata.frame_idx = 0
        if self.pre_encode or self.pre_phash:
            # PERF
            t_enc = time.perf_counter()
            # /PERF
            if self.pre_encode and self.pre_phash:
                enc_b64, phash_val = await asyncio.gather(
                    asyncio.to_thread(base64_encode_jpg, frame),
                    asyncio.to_thread(phash_from_numpy, frame, self.image_hash_size),
                )
                img.encoded_b64 = enc_b64
                img.image_phash = phash_val
            elif self.pre_encode:
                img.encoded_b64 = await asyncio.to_thread(base64_encode_jpg, frame)
            else:
                img.image_phash = await asyncio.to_thread(phash_from_numpy, frame, self.image_hash_size)
            # PERF
            logger.debug("prefetch: encode/phash=%.1fms", 1000 * (time.perf_counter() - t_enc))
            # /PERF
        return img

    async def _read_video_sequential(self, path: str) -> None:
        """Read all strided frames from a single video file into the prefetch queue."""
        self._vidPath = path
        await asyncio.to_thread(self._open_current_path)
        if self._vid is None:
            return
        self._refine_total_frames()
        self._vidFrameCnt = 0
        while True:
            ret, frame = await asyncio.to_thread(self._read_frame)
            if not ret:
                break
            img = ImageWithProviderMetadata(frame)
            img.metadata.source = self._vidPath
            img.metadata.frame_idx = self._vidFrameCnt
            self._vidFrameCnt += self._frame_stride
            self._advance_stride()
            if self.pre_encode:
                img.encoded_b64 = await asyncio.to_thread(base64_encode_jpg, frame)
            await self._prefetch_queue.put(img)
        if self._vid:
            self._vid.release()
            self._vid = None

    async def _prefetch_worker(self) -> None:
        """Reads frames and buffers them asynchronously.

        Static image files are read with up to PARALLEL_READERS concurrent threads.
        Video files are processed sequentially between image batches.
        """
        self._refine_total_frames()
        in_flight: set[asyncio.Task] = set()
        path_index: int = 0

        # PERF
        _perf = {"frames_produced": 0, "queue_full_count": 0}
        _SUMMARY_INTERVAL = 100
        # /PERF

        async def _enqueue_done(done: set) -> None:
            for task in done:
                try:
                    img = task.result()
                except Exception as exc:
                    logger.warning("Image read task failed: %s", exc)
                    continue
                if img is None:
                    continue
                # PERF
                if self._prefetch_queue.full():
                    _perf["queue_full_count"] += 1
                    logger.debug(
                        "prefetch: queue full — inference is the bottleneck (total=%d)",
                        _perf["queue_full_count"],
                    )
                # /PERF
                await self._prefetch_queue.put(img)
                # PERF
                _perf["frames_produced"] += 1
                if _perf["frames_produced"] % _SUMMARY_INTERVAL == 0:
                    logger.info(
                        "[prefetch summary] frames_produced=%d queue=%d/%d queue_full_events=%d",
                        _perf["frames_produced"],
                        self._prefetch_queue.qsize(),
                        self._prefetch_queue.maxsize,
                        _perf["queue_full_count"],
                    )
                # /PERF

        try:
            while True:
                # Wait for discovery to produce paths when we've caught up.
                while path_index >= len(self._paths) and not self._discovery_complete:
                    if in_flight:
                        break  # drain in-flight while waiting for more paths
                    await asyncio.sleep(0.01)

                # Fill the in-flight pool up to PARALLEL_READERS concurrent image reads.
                while len(in_flight) < self.PARALLEL_READERS and path_index < len(self._paths):
                    path = self._paths[path_index]
                    path_index += 1
                    _, ext = os.path.splitext(path.lower())
                    if ext in self.IMAGE_EXTENSIONS:
                        in_flight.add(asyncio.create_task(self._read_image_file(path)))
                    else:
                        # Video file: drain in-flight image reads first, then read frames sequentially.
                        if in_flight:
                            all_done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.ALL_COMPLETED)
                            await _enqueue_done(all_done)
                        await self._read_video_sequential(path)

                # Collect at least one completed result.
                if in_flight:
                    done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                    await _enqueue_done(done)
                elif path_index >= len(self._paths):
                    if not self._discovery_complete:
                        await asyncio.sleep(0.01)  # discovery still running, wait for more paths
                        continue
                    if self._loop and self._paths:
                        path_index = 0  # restart playlist for looping providers
                        continue
                    break  # all paths exhausted

        except asyncio.CancelledError:
            for task in list(in_flight):
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            raise
        finally:
            # PERF
            if _perf["frames_produced"]:
                logger.info(
                    "[prefetch final] frames_produced=%d queue_full_events=%d",
                    _perf["frames_produced"],
                    _perf["queue_full_count"],
                )
            # /PERF
            try:
                self._prefetch_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
        if not hasattr(self, "_prefetch_queue"):
            return await self._direct_read()
        return await self._prefetch_queue.get()

    async def _direct_read(self) -> Optional[ImageWithProviderMetadata]:
        """Direct (unbuffered) fallback read path."""
        # Discovery only runs when start() has been called; without it there is nothing to read.
        if self._discovery_task is None:
            return None
        while self._session_active():
            if not self._is_static_image and self._vid is None:
                if self._index >= len(self._paths) and not self._discovery_complete:
                    await asyncio.sleep(0.01)
                    await self._advance_to_next()
                    continue

            ret, frame = await asyncio.to_thread(self._read_frame)
            if not ret:
                await self._advance_to_next()
                if not self._session_active():
                    return None
                ret, frame = await asyncio.to_thread(self._read_frame)
                if not ret:
                    if not self._discovery_complete:
                        await asyncio.sleep(0.01)
                        continue
                    return None
                self._vidFrameCnt = 0

            ret_image = ImageWithProviderMetadata(frame)
            ret_image.metadata.source = self._vidPath
            ret_image.metadata.frame_idx = self._vidFrameCnt

            if self._is_static_image:
                self._vidFrameCnt = 0
                await self._advance_to_next()
            else:
                self._vidFrameCnt += self._frame_stride
                self._advance_stride()

            return ret_image

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
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
                "description": "Collect every Nth frame from video files. Image files are always collected immediately.",
            },
            "recursive": {
                "label": "Recursive Glob",
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Whether glob patterns recurse into subdirectories (e.g. **/*.mp4).",
            },
        }
