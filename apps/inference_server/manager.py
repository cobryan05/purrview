"""Manager models on Inference Server"""

import asyncio
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from glob import glob
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from apps.helpers.inferenceProviders.inferenceProvider import InferenceProvider, InferenceResult
from apps.helpers.yoloUtils import load_yolo_onnx
from apps.helpers.metadataUtils import get_model_metadata, save_model_json_metadata
from apps.helpers.imageUtils import annotate_image

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


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
        self._config = { "models_path": models_path }


    async def get_server_config(self) -> Dict[str, Any]:
        return self._config.copy()

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
                model_class_mappings = metadata.get("class_map", {})
                class_uuids: Dict[str, Optional[str]] = {name: None for name in model_classes}
                for k, v in model_class_mappings.items():
                    class_uuids[k] = v
                ret[model_name] = class_uuids
            except FileNotFoundError:
                continue
        return ret

    async def set_model_class_uuid(self, model_name: str, model_class: str, class_uuid: Optional[str]) -> bool:
        """
        Associate a model's class label with a class uuid

        Args:
            model_name (str): The name of the model to set the association on
            model_class (str): The class label to associate with the UUID
            class_uuid (Optional[str]): The UUID of the class to associate, or to clear if None

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
        if class_uuid is None and model_class in class_map:
            del class_map[model_class]
        else:
            class_map[model_class] = class_uuid
        metadata["class_map"] = class_map
        save_model_json_metadata(model_path, metadata=metadata)
        return True

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

        # Load the model into memory if not already loaded
        if model_name not in self._pinned_models:
            new_model = InferenceModel(model_name=model_name, path=model_path)
            new_model.inference = await asyncio.to_thread(load_yolo_onnx, model_path)
            self._pinned_models[model_name] = new_model
            logger.info(f"Model '{model_name}' loaded into memory.")

        pin_id = str(uuid.uuid4())
        expiry = time.time() + timeout
        self._pinned_models[model_name].pins[pin_id] = Pin(pin_id=pin_id, expiry=expiry, timeout=timeout)
        logger.info(f"Model '{model_name}' pinned with ID '{pin_id}' and timeout {timeout}s.")

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
        logger.info(f"Model '{model_name}' unpinned with ID '{pin_id}'.")

        # If no more pins exist for the model, unload it from memory
        if not self._pinned_models[model_name].pins:
            del self._pinned_models[model_name]
            logger.info(f"Model '{model_name}' unloaded from memory.")

    async def recognize(
        self,
        model_name: str,
        image: np.ndarray,
        conf_thresh: float,
        return_annotated: bool = False,
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
            return_annotated (bool): Whether to return the annotated image.
            **kwargs: Additional parameters to pass to the inference backend.

        Returns:
            Tuple[InferenceResult, Optional[np.ndarray]]:
                - An InferenceResult with information from the recognition
        """
        if model_name not in self._avail_models:
            raise ValueError(f"Model '{model_name}' is not available. Please check the model name.")

        # Temporarily pin the model
        tmp_pin_id = await self.pin_model(model_name)
        try:
            # Ensure the model is loaded
            model = self._pinned_models.get(model_name)
            if not model or not model.inference:
                raise ValueError(f"Model '{model_name}' is not properly loaded or initialized.")

            inference_result: InferenceResult = await model.inference.processImage(
                image, conf_thresh=conf_thresh, **kwargs
            )

            if return_annotated:
                inference_result.annotated_image = annotate_image(image, inference_result.detections)

            if len(inference_result.detections) > 0:
                class_uuid_maps = await self.get_models_classes([model_name])
                class_uuid_map = class_uuid_maps.get(model_name, {})
                for det in inference_result.detections:
                    if not det.class_name:
                        continue
                    det.class_uuid = class_uuid_map.get(det.class_name, det.class_uuid)

            if pin_id in model.pins:
                logger.info(f"Refreshing timeout for pin {pin_id}")
                model.pins[pin_id].expiry = time.time() + model.pins[pin_id].timeout

            return inference_result

        finally:
            # Unpin the model after inference
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
