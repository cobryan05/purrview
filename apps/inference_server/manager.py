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
import cv2

from apps.helpers.inferenceProviders.inferenceProvider import InferenceProvider, InferenceResult
from apps.helpers.yoloUtils import load_yolo_onnx

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@dataclass
class Pin:
    """Represents a pin for a model with its unique ID and expiration time."""

    pin_id: str
    expiry: float


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

    async def list_models(self, glob_pattern: Optional[str] = "*.onnx") -> List[str]:
        """
        List all files in the models directory, optionally filtered by a glob pattern.
        Updates the available models cache.

        Args:
            glob_pattern (Optional[str]): A glob pattern to filter files (default: "*.onnx").

        Returns:
            List[str]: A list of filenames in the models directory.
        """
        if not self._models_path:
            logger.warning("Models path is not set.")
            return []

        try:
            search_path = os.path.join(self._models_path, glob_pattern)
            model_files = [f for f in glob(search_path) if os.path.isfile(f)]

            # Update the available models cache
            self._avail_models = {os.path.splitext(os.path.basename(f))[0]: f for f in model_files}
            return list(self._avail_models.keys())
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []

    async def pin_model(self, model_name: str, timeout: float, refresh: bool = False) -> str:
        """
        Pin a model into memory, keeping it loaded until unpinned.

        Args:
            model_name (str): Name of the model to pin.
            timeout (float): Timeout in seconds for the pin.
            refresh (bool): If True, refresh the timeout for an existing pin.

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

        # Generate a unique pin ID
        pin_id = str(uuid.uuid4())
        expiry = time.time() + timeout

        # Refresh timeout if requested
        if refresh:
            for pin in self._pinned_models[model_name].pins.values():
                pin.expiry = expiry
            logger.info(f"Timeout for model '{model_name}' refreshed.")
        else:
            # Add a new pin
            self._pinned_models[model_name].pins[pin_id] = Pin(pin_id=pin_id, expiry=expiry)
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
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Recognize objects in an image using a specified model.

        Args:
            model_name (str): Name of the model to use for recognition.
            image (np.ndarray): The input image in BGR format.
            conf_thresh (float): Confidence threshold for detections.
            return_annotated (bool): Whether to return the annotated image.
            **kwargs: Additional parameters to pass to the inference backend.

        Returns:
            Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
                - A list of labeled bounding boxes detected in the image.
                  Each bounding box is represented as a dictionary with keys:
                  'label', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max'.
                - Optionally, the annotated image if `return_annotated` is True.
        """
        if model_name not in self._avail_models:
            raise ValueError(f"Model '{model_name}' is not available. Please check the model name.")

        # Temporarily pin the model
        pin_id = await self.pin_model(model_name, timeout=60)  # Pin for 60 seconds
        try:
            # Ensure the model is loaded
            model = self._pinned_models.get(model_name)
            if not model or not model.inference:
                raise ValueError(f"Model '{model_name}' is not properly loaded or initialized.")

            # Run inference
            inference_result: InferenceResult = await model.inference.processImage(image, conf_thresh=conf_thresh, **kwargs)

            # Prepare detections
            detections = [
                {
                    "label": class_name,
                    "confidence": float(confidence),
                    "x_min": int(box[0]),
                    "y_min": int(box[1]),
                    "x_max": int(box[2]),
                    "y_max": int(box[3]),
                }
                for box, confidence, class_name in zip(
                    inference_result.boxes,
                    inference_result.confidences,
                    inference_result.class_names or inference_result.class_ids,
                )
            ]

            # Annotate the image if requested
            annotated_image = None
            if return_annotated:
                annotated_image = self._annotate_image(image, detections)

            return detections, annotated_image

        finally:
            # Unpin the model after inference
            await self.unpin_model(model_name, pin_id)

    # TODO: Move this
    def _annotate_image(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Annotate the image with bounding boxes and labels.

        Args:
            image (np.ndarray): The original image in BGR format.
            detections (List[Dict[str, Any]]): The detected bounding boxes.

        Returns:
            np.ndarray: The annotated image.
        """
        for detection in detections:
            x_min, y_min, x_max, y_max = (
                detection["x_min"],
                detection["y_min"],
                detection["x_max"],
                detection["y_max"],
            )
            label = detection["label"]
            confidence = detection["confidence"]

            # Draw the bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Put the label and confidence
            text = f"{label}: {confidence:.2f}"
            cv2.putText(
                image,
                text,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return image

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
