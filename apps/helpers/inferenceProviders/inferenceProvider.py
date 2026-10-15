"""Inference Provider Interface Class"""

from abc import ABC, abstractmethod
from typing import Any, Hashable, Optional

import numpy as np

from apps.helpers.types import InferenceResult


class InferenceProvider(ABC):
    # Set by a provider whose session ended up on the CPU because CUDA could not
    # initialize — a one-line, user-facing reason (see
    # apps.helpers.onnxSession.create_session). None on the normal path.
    #
    # Declared here so a caller can read it off any provider without knowing
    # which family it is: inference_server.manager surfaces it as a detect-unit
    # note, which is the channel that reaches the UI.
    device_fallback: Optional[str] = None

    @abstractmethod
    async def processImage(self, image: np.ndarray, **kwargs) -> InferenceResult:
        """Retrieves the next image for processing from the image source"""
        raise NotImplementedError()

    def preprocess_key(self) -> Optional[Hashable]:
        """Identity of this provider's preprocessing, for sharing it across models.

        Running several models over one frame is the normal case (a watcher's
        model stack, the collector's model_label_config), and each of them
        would otherwise letterbox the same image to the same tensor
        independently. Providers that produce a byte-identical tensor for a
        given input return equal keys, so the work can be done once and handed
        to all of them.

        Returning ``None`` means "do not share" and is the safe default: a
        provider whose preprocessing depends on anything not captured by a key
        must opt out rather than risk being fed a tensor prepared for a
        different model. Getting this wrong is silent — the model still returns
        detections, they are just decoded against the wrong letterbox — so the
        default has to be the conservative one.
        """
        return None

    def preprocess_shared(self, image: np.ndarray) -> Any:
        """Compute the shareable preprocessing result for *image*.

        Only called when :meth:`preprocess_key` returns non-None. Whatever this
        returns is passed back to :meth:`processImage` as ``pre=`` for every
        provider with an equal key.
        """
        raise NotImplementedError()
