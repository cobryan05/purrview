"""Shared ONNX classifier session + batched inference for tasks_server tasks.

Both RunClassifierTask and PreviewClassifierMatchTask load a classifier once
per task run and batch many images through a single `session.run` call,
instead of round-tripping through inference_server's `/classify` endpoint —
that endpoint auto-pins/unpins per call (a full model load+unload cycle) and
processes one image at a time, which is fine for interactive single-image use
but far too slow across thousands of images.
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from apps.helpers.classifier_utils import preprocess_image_bytes, sigmoid, softmax
from apps.helpers.metadataUtils import DEFAULT_MODEL_DEVICE, providers_for_device

_MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/models"))


def load_classifier_config(model_name: str) -> Tuple[Path, Dict[str, Any]]:
    """Read a classifier's ONNX path + sidecar JSON config from the models root.

    Assumes the caller has already validated model_name (bare name, no path
    separators) and confirmed its model_kind is a classifier.
    """
    model_path = _MODELS_ROOT / f"{model_name}.onnx"
    config_path = _MODELS_ROOT / f"{model_name}.json"
    with open(config_path) as f:
        config: Dict[str, Any] = json.load(f)
    return model_path, config


class ClassifierRuntime:
    """Loads an ONNX classifier once and runs batched inference for the
    lifetime of the owning task — an implicit, in-process pin (no
    inference_server load/unload cycling per image)."""

    def __init__(self, model_path: Path, config: Dict[str, Any]):
        self._model_path = model_path
        self.config = config
        self.model_kind: str = config.get("model_kind", "image_classifier")
        self.input_size: Tuple[int, int] = tuple(config.get("input_size", [224, 224]))  # type: ignore
        self._session: Any = None
        self._input_name: Optional[str] = None
        self._cuda_executor: Optional[ThreadPoolExecutor] = None

    async def load(self) -> None:
        import onnxruntime as ort

        providers = providers_for_device(self.config.get("device", DEFAULT_MODEL_DEVICE))
        self._session = await asyncio.to_thread(ort.InferenceSession, str(self._model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

        # If CUDA ended up active, pin session.run to a dedicated single-threaded
        # executor for the lifetime of this runtime — shifting CUDA calls across
        # asyncio.to_thread's default-pool threads causes context/memory errors
        # (see OnnxYoloInferenceProvider, which applies the same mitigation).
        if "CUDAExecutionProvider" in self._session.get_providers():
            self._cuda_executor = ThreadPoolExecutor(max_workers=1)

    def preprocess(self, img_bytes: bytes) -> Optional[np.ndarray]:
        return preprocess_image_bytes(img_bytes, self.input_size)

    async def run_batch(self, tensors: List[np.ndarray]) -> np.ndarray:
        """Run one batched `session.run` call over *tensors* ((3, H, W) arrays).

        Returns raw logits, shape (N, num_classes).
        """
        batch_arr = np.stack(tensors)
        loop = asyncio.get_running_loop()
        if self._cuda_executor:
            raw_outputs = await loop.run_in_executor(
                self._cuda_executor, lambda: self._session.run(None, {self._input_name: batch_arr})
            )
        else:
            raw_outputs = await asyncio.to_thread(self._session.run, None, {self._input_name: batch_arr})
        return raw_outputs[0]

    async def predict_batch(self, tensors: List[np.ndarray]) -> List[List[Tuple[str, float]]]:
        """Run inference and map logits -> [(label_uuid, confidence), ...] per image,
        matching what inference_server's /classify endpoint returns per image.

        `image_classifier` models are binary present/absent (softmax, single
        configured label_uuid); `multi_label_classifier` models score each
        label independently (sigmoid, one label_uuid per output unit);
        `crop_classifier` models pick one of N competing classes for the crop
        they were handed (softmax across label_uuids).
        """
        logits = await self.run_batch(tensors)

        if self.model_kind == "crop_classifier":
            # Softmax, not sigmoid: the classes compete for a single crop. Falling
            # through to the multi-label branch below would score each class
            # independently and quietly return numbers that don't sum to 1.
            crop_label_uuids: List[str] = self.config.get("label_uuids") or []
            crop_probs = softmax(logits, axis=-1)
            return [list(zip(crop_label_uuids, (float(p) for p in row))) for row in crop_probs]

        if self.model_kind == "image_classifier":
            class_names = self.config.get("class_names") or ["absent", "present"]
            present_idx = class_names.index("present") if "present" in class_names else 1
            label_uuid = self.config.get("label_uuid")
            probs = softmax(logits, axis=-1)
            return [[(label_uuid, float(row[present_idx]))] for row in probs]

        label_uuids: List[str] = self.config.get("label_uuids") or []
        probs = sigmoid(logits)
        return [list(zip(label_uuids, (float(p) for p in row))) for row in probs]

    async def close(self) -> None:
        if self._cuda_executor is not None:
            self._cuda_executor.shutdown(wait=False)
