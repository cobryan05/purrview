"""Manager models on Inference Server"""

import asyncio
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from apps import APPS, CACHE_DIR
from apps.helpers.imageUtils import annotate_image
from apps.helpers.inferenceProviders.inferenceProvider import (
    InferenceProvider,
    InferenceResult,
)
from apps.helpers.metadataUtils import (
    VALID_MODEL_DEVICES,
    get_model_metadata,
    save_model_json_metadata,
    validate_bare_model_name,
    write_onnx_with_sidecar,
)
from apps.helpers.modelFormat import (
    ModelFormat,
    detect_model_format,
    is_conversion_candidate,
)
from apps.helpers.modelStaging import STAGING_SUBDIR
from apps.helpers.yoloUtils import load_yolo_onnx

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

# Subdirectory (under CACHE_DIR) where raw uploaded model files are staged pending
# an asynchronous conversion task (e.g. pytorch -> onnx) run by the tasks_server.
# The subdirectory name + staging-id validation are centralized in
# apps.helpers.modelStaging (also used directly by ConvertModelTask); this module
# keeps resolving the directory from its OWN module-level CACHE_DIR (rather than
# calling into the helper's staging_dir()) so tests that monkeypatch
# apps.inference_server.manager.CACHE_DIR keep working unmodified.
_STAGING_SUBDIR = STAGING_SUBDIR


@dataclass
class Pin:
    """Represents a pin for a model with its unique ID and expiration time."""

    pin_id: str
    expiry: float
    timeout: float


@dataclass
class InferenceModel:
    """Represents a model that is pinned in memory."""

    model_name: str
    path: str
    inference: Optional[InferenceProvider] = None
    pins: Dict[str, Pin] = field(default_factory=dict)  # {pin_id: Pin}


class Manager:
    """Manages inference models, including loading, pinning, and unpinning."""

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks
    MODEL_PIN_TIME: float = 10.0 * 60  # Default timeout in seconds before unloading model from GPU

    def __init__(self, models_path: str):
        """
        Initialize the Manager with a models directory path.

        Args:
            models_path (str): Path to the directory containing model files.
        """
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations
        self._models_path: str = models_path  # Store the models directory path
        self._avail_models: Dict[str, str] = {}  # {basename (no .onnx): full_path}
        self._pinned_models: Dict[str, InferenceModel] = {}  # {model_name: PinnedModel}
        self._loading_models: Dict[str, asyncio.Future] = {}  # in-progress loads; awaited by concurrent pin requests
        self._config = {"service": APPS.INFERENCE_SERVER, "models_path": models_path}

    async def get_server_config(self) -> Dict[str, Any]:
        return self._config.copy()

    def get_models_path(self) -> str:
        return self._models_path

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Read the companion JSON for *model_name*; returns None if missing."""
        import json
        from pathlib import Path

        config_path = Path(self._models_path) / f"{model_name}.json"
        if not config_path.exists():
            return None
        with open(config_path) as f:
            return json.load(f)

    async def list_models(self, pattern: Optional[str] = "*.onnx") -> List[str]:
        """
        List all files in the models directory, optionally filtered by a glob pattern.
        Updates the available models cache.

        Args:
            pattern (Optional[str]): A glob pattern to filter files (default: "*.onnx").

        Returns:
            List[str]: A list of filenames in the models directory.
        """
        if not self._models_path:
            logger.warning("Models path is not set.")
            return []

        try:
            search_path = os.path.join(self._models_path, pattern)
            model_files = [f for f in glob(search_path) if os.path.isfile(f)]

            # Update the available models cache
            self._avail_models = {os.path.splitext(os.path.basename(f))[0]: f for f in model_files}
            return list(self._avail_models.keys())
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []

    async def get_models_classes(self, model_names: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Get the classes for a specific model.

        Args:
            model_name (str): The name of the model to get classes for.

        Returns:
            Dict[str, str]: A dictionary mapping class names to class IDs.
        """

        model_paths = {name: self._avail_models[name] for name in model_names if name in self._avail_models}

        # Create metadata but don't entirely fail on get_model_metadata failrule just leave that one out
        ret = {}
        for model_name, model_path in model_paths.items():
            try:
                metadata = get_model_metadata(model_path)
                model_classes = metadata.get("classes", [])
                class_map = metadata.get("class_map", {})
                # Models trained in-app already know their class -> label
                # mapping (written positionally into the sidecar by the
                # trainer); use it to pre-fill classes that don't yet have
                # an explicit class_map entry, instead of making the user
                # re-enter an assignment we already have. Evaluated per
                # class so an explicit clear on one class (which persists
                # as class_map[name] = None) doesn't suppress the
                # positional fallback for every other class.
                label_uuids = metadata.get("label_uuids") or []
                positional_mappings = (
                    dict(zip(model_classes, label_uuids)) if len(label_uuids) == len(model_classes) else {}
                )
                class_uuids: Dict[str, Optional[str]] = {
                    name: class_map[name] if name in class_map else positional_mappings.get(name)
                    for name in model_classes
                }
                ret[model_name] = class_uuids
            except FileNotFoundError:
                continue
        return ret

    async def set_model_class_label_uuid(self, model_name: str, model_class: str, label_uuid: Optional[str]) -> bool:
        """
        Associate a model's class label with a label uuid

        Args:
            model_name (str): The name of the model to set the association on
            model_class (str): The class label to associate with the UUID
            label_uuid (Optional[str]): The UUID of the label to associate, or to clear if None

        Returns:
            bool: True if the association was successful, False otherwise.
        """
        if model_name in self._avail_models:
            model_path = self._avail_models[model_name]
        else:
            raise FileNotFoundError(f"Model '{model_name}' not found in available models")

        try:
            metadata = get_model_metadata(model_path)
        except FileNotFoundError:
            metadata = {}
        if model_class not in metadata.get("classes", {}):
            raise ValueError(f"Class '{model_class}' not found in metadata for model '{model_name}'")
        class_map = metadata.get("class_map", {})
        if label_uuid is None and model_class in class_map:
            del class_map[model_class]
        else:
            class_map[model_class] = label_uuid
        metadata["class_map"] = class_map
        save_model_json_metadata(model_path, metadata=metadata)
        return True

    async def set_model_device(self, model_name: str, device: str) -> bool:
        """
        Set a model's device preference (auto/gpu/cpu), persisted to its sidecar JSON.

        Args:
            model_name (str): The name of the model to set the preference on
            device (str): One of "auto", "gpu", "cpu"

        Returns:
            bool: True if the preference was persisted.
        """
        if device not in VALID_MODEL_DEVICES:
            raise ValueError(f"Invalid device {device!r}; must be one of {VALID_MODEL_DEVICES}")

        if model_name in self._avail_models:
            model_path = self._avail_models[model_name]
        else:
            raise FileNotFoundError(f"Model '{model_name}' not found in available models")

        try:
            metadata = get_model_metadata(model_path)
        except FileNotFoundError:
            metadata = {}
        metadata["device"] = device
        save_model_json_metadata(model_path, metadata=metadata)
        return True

    async def import_model_upload(self, model_name: str, onnx_bytes: bytes, overwrite: bool = False) -> Dict[str, Any]:
        """Write an uploaded ONNX file to the models dir and auto-generate its sidecar
        (classes parsed from the ONNX itself).

        Args:
            model_name (str): Bare model name (no extension, no path separators).
            onnx_bytes (bytes): Raw ONNX file content.
            overwrite (bool): If True, replace an existing model + sidecar. If False and
                the model already exists, raise FileExistsError.

        Returns:
            Dict[str, Any]: {"model_name": ..., "classes": [...]}
        """
        validate_bare_model_name(model_name)

        model_path = Path(self._models_path) / f"{model_name}.onnx"

        if model_path.exists() and not overwrite:
            raise FileExistsError(f"Model '{model_name}' already exists; pass overwrite=True to replace it")

        metadata = await asyncio.to_thread(write_onnx_with_sidecar, str(model_path), onnx_bytes, overwrite=overwrite)

        await self.list_models()

        return {"model_name": model_name, "classes": metadata.get("classes", [])}

    async def delete_model(self, model_name: str) -> None:
        """Delete a model's ONNX weights and sidecar JSON from the models dir.

        Args:
            model_name (str): Bare model name (no extension) to delete.

        Raises:
            FileNotFoundError: If the model is not in the available-models list.
            ValueError: If the model is currently pinned (in use) and can't be
                safely removed.
        """
        if model_name not in self._avail_models:
            raise FileNotFoundError(f"Model '{model_name}' not found in available models")
        if model_name in self._pinned_models:
            raise ValueError(f"Model '{model_name}' is currently pinned/in use and cannot be deleted")

        model_path = Path(self._avail_models[model_name])
        sidecar_path = model_path.with_suffix(".json")

        await asyncio.to_thread(model_path.unlink, True)
        await asyncio.to_thread(sidecar_path.unlink, True)

        del self._avail_models[model_name]

    def detect_uploaded_model(self, filename: str, header: bytes) -> Dict[str, Any]:
        """Detect the format of an about-to-be-uploaded model file from its filename and
        leading bytes (torch-free — see apps/helpers/modelFormat.py). Pure/sync, no I/O.

        Args:
            filename (str): The original filename of the uploaded file.
            header (bytes): The first chunk of the file's bytes (used for magic-byte sniffing).

        Returns:
            Dict[str, Any]: {"format": ..., "is_onnx": ..., "conversion_supported": ...}
        """
        fmt = detect_model_format(filename, header)
        return {
            "format": fmt,
            "is_onnx": fmt == ModelFormat.ONNX,
            "conversion_supported": is_conversion_candidate(fmt),
        }

    def _staging_dir(self) -> Path:
        """Path to the shared staging directory (under CACHE_DIR) used to hand raw
        uploaded model bytes off to a later conversion task run by another service
        (same container/filesystem). Computed only — no filesystem side effects."""
        return CACHE_DIR / _STAGING_SUBDIR

    async def stage_source_model(self, data: bytes) -> str:
        """Write raw uploaded model bytes to the shared staging dir under a fresh id.

        Args:
            data (bytes): Raw file content to stage.

        Returns:
            str: The generated staging id (bare hex token, no extension); the caller
                (a conversion task) can resolve it back to a path via `staged_model_path`.
        """
        if not data:
            raise ValueError("Cannot stage empty model data")

        staging_id = uuid.uuid4().hex
        staging_dir = self._staging_dir()
        staging_path = staging_dir / f"{staging_id}.bin"

        await asyncio.to_thread(staging_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(staging_path.write_bytes, data)

        return staging_id

    def staged_model_path(self, staging_id: str) -> Path:
        """Resolve a staging id (as returned by `stage_source_model`) to its file path.

        Args:
            staging_id (str): Bare staging id — must not contain path separators or '..'.

        Returns:
            Path: The path to the staged file (may not exist).
        """
        validate_bare_model_name(staging_id)
        return self._staging_dir() / f"{staging_id}.bin"

    def delete_staged_model(self, staging_id: str) -> None:
        """Best-effort removal of a staged model file. The conversion task owns cleanup
        after a successful convert; this is provided for defensive/manual cleanup."""
        path = self.staged_model_path(staging_id)
        path.unlink(missing_ok=True)

    async def pin_model(self, model_name: str, timeout: float = MODEL_PIN_TIME) -> str:
        """
        Pin a model into memory, keeping it loaded until unpinned.

        Args:
            model_name (str): Name of the model to pin.
            timeout (float): Timeout in seconds for the pin.

        Returns:
            str: A unique pin ID that can be used to unpin the model.
        """
        # Determine the model path
        if model_name in self._avail_models:
            model_path = self._avail_models[model_name]
        else:
            raise FileNotFoundError(f"Model '{model_name}' not found in available models")

        # If a concurrent pin_model call is already loading this model, wait for it.
        if model_name in self._loading_models:
            await self._loading_models[model_name]

        if model_name not in self._pinned_models:
            load_future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._loading_models[model_name] = load_future
            try:
                new_model = InferenceModel(model_name=model_name, path=model_path)
                new_model.inference = await asyncio.to_thread(load_yolo_onnx, model_path)
                self._pinned_models[model_name] = new_model
                logger.info(f"Model '{model_name}' loaded into memory.")
                load_future.set_result(None)
            except Exception as exc:
                load_future.set_exception(exc)
                raise
            finally:
                self._loading_models.pop(model_name, None)

        pin_id = str(uuid.uuid4())
        expiry = time.time() + timeout
        self._pinned_models[model_name].pins[pin_id] = Pin(pin_id=pin_id, expiry=expiry, timeout=timeout)
        logger.debug(f"Model '{model_name}' pinned with ID '{pin_id}' and timeout {timeout}s.")

        return pin_id

    async def unpin_model(self, model_name: str, pin_id: str) -> None:
        """
        Unpin a model using the provided pin ID.

        Args:
            model_name (str): Name of the model to unpin.
            pin_id (str): The unique pin ID returned by `pin_model`.

        Raises:
            ValueError: If the model or pin ID is not found.
        """
        if model_name not in self._pinned_models or pin_id not in self._pinned_models[model_name].pins:
            raise ValueError(f"Pin ID '{pin_id}' for model '{model_name}' not found.")

        # Remove the pin
        del self._pinned_models[model_name].pins[pin_id]
        logger.debug(f"Model '{model_name}' unpinned with ID '{pin_id}'.")

        # If no more pins exist for the model, unload it from memory
        if not self._pinned_models[model_name].pins:
            del self._pinned_models[model_name]
            logger.info(f"Model '{model_name}' unloaded from memory.")

    async def recognize(
        self,
        model_name: str,
        image: np.ndarray,
        conf_thresh: float,
        return_annotated_img: bool = False,
        pin_id: Optional[str] = None,
        **kwargs,
    ) -> InferenceResult:
        """
        Recognize objects in an image using a specified model.

        Args:
            model_name (str): Name of the model to use for recognition.
            image (np.ndarray): The input image in BGR format.
            conf_thresh (float): Confidence threshold for detections.
            pin_id (str): pin_id to refresh if using a pinned model
            return_annotated_img (bool): Whether to return the annotated image.
            **kwargs: Additional parameters to pass to the inference backend.

        Returns:
            Tuple[InferenceResult, Optional[np.ndarray]]:
                - An InferenceResult with information from the recognition
        """
        if model_name not in self._avail_models:
            raise ValueError(f"Model '{model_name}' is not available. Please check the model name.")

        # If the caller holds a valid long-lived pin and the model is already loaded,
        # skip creating a temp pin — the caller's pin guarantees the model stays alive.
        # Refresh the pin's expiry first so it can't expire mid-inference.
        pinned = self._pinned_models.get(model_name)
        caller_pin_valid = pin_id is not None and pinned is not None and pin_id in pinned.pins
        if caller_pin_valid:
            pinned.pins[pin_id].expiry = time.time() + pinned.pins[pin_id].timeout
            tmp_pin_id = None
        else:
            tmp_pin_id = await self.pin_model(model_name)

        try:
            model = self._pinned_models.get(model_name)
            if not model or not model.inference:
                raise ValueError(f"Model '{model_name}' is not properly loaded or initialized.")

            t0 = time.perf_counter()
            inference_result: InferenceResult = await model.inference.processImage(
                image, conf_thresh=conf_thresh, **kwargs
            )
            t_infer = time.perf_counter()

            if return_annotated_img:
                inference_result.annotated_image = annotate_image(image, inference_result.detections)

            if len(inference_result.detections) > 0:
                t_cls = time.perf_counter()
                class_uuid_maps = await self.get_models_classes([model_name])
                class_uuid_map = class_uuid_maps.get(model_name, {})
                for det in inference_result.detections:
                    if not det.bbox.class_str:
                        continue
                    det.bbox.label_uuid = class_uuid_map.get(det.bbox.class_str, det.bbox.label_uuid)
                logger.debug(
                    f"recognize '{model_name}': inference={1000*(t_infer-t0):.1f}ms "
                    f"class_lookup={1000*(time.perf_counter()-t_cls):.1f}ms "
                    f"({len(inference_result.detections)} dets)"
                )
            else:
                logger.debug(f"recognize '{model_name}': inference={1000*(t_infer-t0):.1f}ms (no detections)")

            return inference_result

        finally:
            if tmp_pin_id:
                await self.unpin_model(model_name, tmp_pin_id)

    def _cleanup_expired_pins(self):
        """
        Periodically clean up expired pins.
        """
        current_time = time.time()
        for model_name, pinned_model in list(self._pinned_models.items()):
            for pin_id, pin in list(pinned_model.pins.items()):
                if pin.expiry < current_time:
                    logger.info(f"Pin ID '{pin_id}' for model '{model_name}' expired.")
                    del pinned_model.pins[pin_id]

            # If no more pins exist for the model, unload it
            if not pinned_model.pins:
                del self._pinned_models[model_name]
                logger.info(f"Model '{model_name}' unloaded from memory due to expired pins.")

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()  # Cancel the existing task if it's still running
        self._task = asyncio.get_running_loop().create_task(self._worker_task())  # Schedule a new task

    async def _init(self):
        """Initialization that should run on event loop"""
        await self.list_models()

    async def _worker_task(self):
        """
        Periodic worker task that runs at regular intervals.
        """
        try:
            await self._init()
            while True:
                self._cleanup_expired_pins()
                await asyncio.sleep(Manager.POLLING_INTERVAL)  # Wait for the polling interval
        except asyncio.CancelledError:
            logger.info("Periodic task was cancelled.")  # Handle task cancellation
        finally:
            logger.info("Periodic task cleanup.")  # Perform cleanup when the task is stopped

    def get_status(self) -> dict:
        """Return service health and diagnostic info."""
        return {
            "service": "inference_server",
            "status": "ok",
            "models": {
                "loaded": len(self._pinned_models),
                "available": len(self._avail_models),
            },
        }
