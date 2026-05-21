"""Image Provider Interface Class"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ImageProviderMetadata:
    source: Optional[str] = None
    frame_idx: Optional[int ] = None
    additional_info: dict[str, Any] = field(default_factory=dict)

@dataclass
class ImageWithProviderMetadata:
    image: np.ndarray
    metadata: ImageProviderMetadata = field(default_factory=ImageProviderMetadata)

class ImageProvider:
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
