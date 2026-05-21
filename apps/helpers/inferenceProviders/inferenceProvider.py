"""Inference Provider Interface Class"""

from abc import ABC, abstractmethod

import numpy as np

from apps.helpers.types import InferenceResult


class InferenceProvider(ABC):
    @abstractmethod
    async def processImage(self, image: np.ndarray, **kwargs) -> InferenceResult:
        """Retrieves the next image for processing from the image source"""
        raise NotImplementedError()
