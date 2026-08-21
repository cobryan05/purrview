"""Image Provider Interface Class"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class ImageProviderType(Enum):
    FIXED = "fixed"  # Known frame count; process at full speed, report progress
    LIVE_SNAPSHOT = "live_snapshot"  # Poll on a slow interval (e.g. URL cameras)
    LIVE_STREAM = "live_stream"  # Continuous stream; rate-limit collection (e.g. RTSP)


@dataclass
class ImageProviderMetadata:
    source: Optional[str] = None
    frame_idx: Optional[int] = None
    additional_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageWithProviderMetadata:
    image: np.ndarray
    metadata: ImageProviderMetadata = field(default_factory=ImageProviderMetadata)
    encoded_b64: Optional[str] = None
    image_phash: Optional[str] = None  # pre-computed by prefetch worker when pre_phash=True


class ImageProvider:
    source_name: Optional[str] = None
    # Minimum seconds between frames this provider will return, self-paced inside
    # getNextImage() (snapshots sleep, streams drain to the newest frame). 0.0 = unset;
    # the task-level fallback interval (keyed by ImageProviderType) applies instead.
    # Subclasses that support source-level pacing assign this in __init__.
    min_capture_interval: float = 0.0

    def set_source_fps_ceiling(self, fps: float) -> None:
        """Tell the source not to send more than *fps* frames per second, if it can.

        A hint about the *source*, not a promise to the caller: it never makes a
        provider yield frames more slowly than it otherwise would, so a caller that
        already paces itself can set it freely. Distinct from `min_capture_interval`,
        which does pace the caller, by sleeping inside getNextImage().

        A no-op by default. Frame-at-a-time providers have nothing to throttle — the
        source sends what it sends — and only ones that pull a continuous stream they
        must then drain (see FfmpegImageProvider) can act on it.
        """

    def set_source_identity(self, name: str) -> None:
        """Sets the human-readable identity of the source this provider reads from,
        used to name notices, status rows and output filenames."""
        self.source_name = name

    @classmethod
    def provider_type(cls) -> ImageProviderType:
        """Return the behavioral type of this provider. Default is LIVE_STREAM (safest fallback)."""
        return ImageProviderType.LIVE_STREAM

    @classmethod
    def ingest_tier(cls) -> str:
        """How this provider's sources can be got into a MediaMTX path, and therefore
        whether they can have a rolling rewind buffer.

        One of ``native``/``ffmpeg``/``ring``/``none`` — see
        ``apps.helpers.streams.ingest``. The default is ``ring`` (the safe answer: a
        bounded in-RAM JPEG buffer) because a provider that hasn't thought about it is
        almost certainly frame-at-a-time. Providers backed by a real continuous stream
        override this; the URL is only inspected for those.
        """
        return "ring"

    def get_total_frames(self) -> Optional[int]:
        """For FIXED providers: best current estimate of total frames to be emitted.
        May grow as new files are opened. Returns None for non-FIXED providers."""
        return None

    async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
        """Retrieves the next image for processing from the image source"""
        raise NotImplementedError()

    async def start(self) -> None:
        """Starts the image provider and prepares it for use"""

    async def stop(self) -> None:
        """Stops the image provider and releases any resources"""

    async def reconnect(self) -> None:
        """Re-establish this provider's connection to its source after a failure.

        Called by a long-running consumer that has decided this source is down and is
        due another attempt; the consumer owns the retry schedule, so this must do one
        attempt and return rather than loop or sleep. It may raise — a caller treats
        that as one more failed attempt, not as fatal.

        The default is stop-then-start, which is correct for a provider holding no
        shared state. One that does (a relay lease, a device handle refcounted with
        other readers) should override to rebuild only its own connection: going
        through stop()/start() would hand that shared thing back and re-take it on
        every attempt, and a camera that is down for an hour makes that a lot of churn
        on something other readers are also holding.
        """
        await self.stop()
        await self.start()

    def is_ready(self) -> bool:
        """Whether a getNextImage() call can be expected to reach the source.

        Exists so a consumer can decline to call getNextImage() on a provider that is
        still connecting, rather than discovering it by blocking for the fetch timeout.
        That distinction matters to a watcher servicing several cameras from one loop:
        the stall would be paid by every *other* camera too.

        Default True — a provider with nothing to connect to is always ready.
        """
        return True

    def is_connecting(self) -> bool:
        """Whether this provider is still coming up and should be left alone.

        Splits "not ready" into its two cases, which want opposite treatment: a
        provider still connecting needs patience, a broken one needs rebuilding. Left
        undivided, a source that is merely slow to connect gets torn down on every
        retry and never completes a connection at all — which for an RTSP camera
        behind a relay, where coming up cold can take minutes, is the difference
        between working and never working.

        Default False — a provider that is not ready and claims no connect window is
        simply not ready.
        """
        return False

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
        return {}
