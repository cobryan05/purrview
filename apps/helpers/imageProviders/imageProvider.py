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


class ImageProvider:
    @classmethod
    def provider_type(cls) -> ImageProviderType:
        """Return the behavioral type of this provider. Default is LIVE_STREAM (safest fallback)."""
        return ImageProviderType.LIVE_STREAM

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
