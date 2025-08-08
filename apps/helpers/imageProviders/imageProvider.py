"""Image Provider Interface Class"""

import numpy as np
from typing import Any, Optional
from dataclasses import dataclass, field

@dataclass
class ImageMetadata:
    source: Optional[str] = None
    additional_info: dict[str, Any] = field(default_factory=dict)

@dataclass
class ImageWithMetadata:
    image: np.ndarray
    metadata: ImageMetadata = field(default_factory=ImageMetadata)

class ImageProvider:
    async def getNextImage(self) -> Optional[ImageWithMetadata]:
        """Retrieves the next image for processing from the image source"""
        raise NotImplementedError()

    async def start(self) -> None:
        """Starts the image provider and prepares it for use"""
        pass

    async def stop(self) -> None:
        """Stops the image provider and releases any resources"""
        pass

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
        return {}
