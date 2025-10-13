"""Manager for Image Tagging Server"""

import asyncio
import base64

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


import cv2
import db_client
import inference_client
import numpy as np
import tasks_client
from db_client.api.labels_api import LabelsApi
from db_client.api.sources_api import SourcesApi
from db_client.api.images_api import ImagesApi
from inference_client.api.inference_api import InferenceApi
from inference_client.api.models_api import ModelsApi
from inference_client.models.associate_label_with_model_class_payload import (
    AssociateLabelWithModelClassPayload,
)
from inference_client.models.body_pin_model import BodyPinModel
from inference_client.models.recognize_payload import RecognizePayload
from tasks_client.api.tasks_api import TasksApi

from apps.helpers.db.db_client import (
    BoundingBoxMetadata,
    DbClient,
    LabelMetadata,
)

from apps.helpers.imageUtils import base64_encode_png
from apps.helpers.webUtils import api_forward_request

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Manager:
    """Manages image tagging tasks."""

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks
    PIN_DURATION: float = 30 * 60  # Timeout before unloading model

    def __init__(
        self,
        inference_api_client: inference_client.ApiClient,
        tasks_api_client: tasks_client.ApiClient,
        db_api_client: db_client.ApiClient,
        legacy_db_client: DbClient,
    ):
        """Initialize the Manager"""
        self._inference_api_client: inference_client.ApiClient = inference_api_client
        self._tasks_api_client: tasks_client.ApiClient = tasks_api_client
        self._db_api_client: db_client.ApiClient = db_api_client
        self._legacy_db_client: DbClient = legacy_db_client
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations

        self._model_pins: Dict[str, str] = {}
        self._config = {
            "inference_server": inference_api_client.configuration.host,
            "tasks_server": tasks_api_client.configuration.host,
            "db_server": db_api_client.configuration.host,
        }

    async def get_server_config(self) -> Dict[str, Any]:
        return self._config.copy()

    def images_api_request(self, api_method_name: str):
        """
        Decorator to forward request to the Images API
        """
        return api_forward_request(ImagesApi(self._db_api_client), api_method_name)

    def inference_api_request(self, api_method_name: str):
        """
        Decorator to forward request to the Inference API
        """
        return api_forward_request(InferenceApi(self._inference_api_client), api_method_name)

    def labels_api_request(self, api_method_name: str):
        """
        Decorator to forward request to the Labels API
        """
        return api_forward_request(LabelsApi(self._db_api_client), api_method_name)

    def model_api_request(self, api_method_name: str):
        """
        Decorator to forward request to the Model API
        """
        return api_forward_request(ModelsApi(self._inference_api_client), api_method_name)

    def sources_api_request(self, api_method_name: str):
        """
        Decorator to forward request to the Sources API
        """
        return api_forward_request(SourcesApi(self._db_api_client), api_method_name)

    def task_api_request(self, api_method_name: str):
        """
        Decorator to forward request to the Task API
        """
        return api_forward_request(TasksApi(self._tasks_api_client), api_method_name)

    async def set_model_label_uuid(self, model_name: str, model_class: str, label_uuid: Optional[str]) -> bool:
        """
        Sets the label UUID that a model's class name should link to

        Params:
            model_name (str): The name of the model.
            model_class (str): The name of the class label within the model.
            label_uuid (str): The UUID of the label to associate, or None to disassociate.

        Returns:
            Dict[str, Any] dict of label info
        """
        api = ModelsApi(self._inference_api_client)
        request = AssociateLabelWithModelClassPayload(
            model_name=model_name, model_class=model_class, label_uuid=label_uuid
        )
        response: Dict[str, Any] = await asyncio.to_thread(api.associate_label_with_model_class, request)
        return response.get("label_set", False)

    async def create_box(
        self,
        image_rel_path: str,
        x: float,
        y: float,
        width: float,
        height: float,
        labels: List[LabelMetadata] = None,
        extra: dict = None,
    ) -> None:
        """
        Add a bounding box to image metadata and save.

        Args:
            image_rel_path (str): Relative image path.
            x, y, width, height: Bounding box coordinates.
            labels: Optional list of Label objects.
            extra: Optional dict of extra metadata.
        """
        image_meta = await self.get_image_metadata(image_rel_path)
        if image_meta is None:
            raise ValueError(f"Image '{image_rel_path}' not found")

        if labels is None:
            labels = []
        if extra is None:
            extra = {}

        # TODO labels
        new_box = BoundingBoxMetadata(
            id=None,
            x=x,
            y=y,
            width=width,
            height=height,
            extra=extra,
        )
        image_meta.boxes.append(new_box)

        await self.update_image_metadata(image_rel_path, image_meta)

    async def update_box(
        self,
        image_rel_path: str,
        box_id: int,
        x: float,
        y: float,
        width: float,
        height: float,
        labels: List[LabelMetadata] = None,
        extra: dict = None,
    ) -> None:
        """
        Update a bounding box inside image metadata.

        Args:
            image_rel_path (str): Relative image path.
            box_id (int): Bounding box ID to update.
            x, y, width, height: New box coordinates.
            labels: New label list.
            extra: New extra dict.
        """
        image_meta = await self.get_image_metadata(image_rel_path)
        if image_meta is None:
            raise ValueError(f"Image '{image_rel_path}' not found")

        updated = False
        for box in image_meta.boxes:
            if box.id == box_id:
                box.x = x
                box.y = y
                box.width = width
                box.height = height
                if labels is not None:
                    box.labels = labels
                if extra is not None:
                    box.extra = extra
                updated = True
                break

        if not updated:
            raise ValueError(f"Box ID {box_id} not found for image '{image_rel_path}'")

        await self.update_image_metadata(image_rel_path, image_meta)

    async def delete_box(self, image_rel_path: str, box_id: int) -> None:
        """
        Delete a bounding box from image metadata.

        Args:
            image_rel_path (str): Relative image path.
            box_id (int): Bounding box ID to delete.
        """
        image_meta = await self.get_image_metadata(image_rel_path)
        if image_meta is None:
            raise ValueError(f"Image '{image_rel_path}' not found")

        image_meta.boxes = [box for box in image_meta.boxes if box.id != box_id]

        await self.update_image_metadata(image_rel_path, image_meta)

    async def add_label_to_box(
        self,
        image_rel_path: str,
        box_id: int,
        label: LabelMetadata,
    ) -> None:
        """
        Add a label to a bounding box in the image metadata.

        Args:
            image_rel_path (str): Relative image path.
            box_id (int): Bounding box ID.
            label (Label): Label to add.
        """
        image_meta = await self.get_image_metadata(image_rel_path)
        if image_meta is None:
            raise ValueError(f"Image '{image_rel_path}' not found")

        for box in image_meta.boxes:
            if box.id == box_id:
                if all(l.id != label.id for l in box.labels):
                    box.labels.append(label)
                break
        else:
            raise ValueError(f"Box ID {box_id} not found for image '{image_rel_path}'")

        await self.update_image_metadata(image_rel_path, image_meta)

    async def remove_label_from_box(
        self,
        image_rel_path: str,
        box_id: int,
        label_uuid: str,
    ) -> None:
        """
        Remove a label from a bounding box in the image metadata.

        Args:
            image_rel_path (str): Relative image path.
            box_id (int): Bounding box ID.
            label_uuid (str): Label uuid to remove.
        """
        image_meta = await self.get_image_metadata(image_rel_path)
        if image_meta is None:
            raise ValueError(f"Image '{image_rel_path}' not found")

        for box in image_meta.boxes:
            if box.id == box_id:
                box.labels = [l for l in box.labels if l.uuid != label_uuid]
                break
        else:
            raise ValueError(f"Box ID {box_id} not found for image '{image_rel_path}'")

        await self.update_image_metadata(image_rel_path, image_meta)

    async def recognize(
        self,
        model_name: str,
        image: np.ndarray,
        conf_thresh: float = 0.25,
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Recognize objects in an image using a specified model.

        Args:
            model_name (str): Name of the model to use for recognition.
            image (np.ndarray): The input image in BGR format.
            conf_thresh (float): Confidence threshold for detections.

        Returns:
            Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
                List of detections and optionally the annotated image
        """
        # Ensure model is pinned
        model_api = ModelsApi(self._inference_api_client)
        post_info = BodyPinModel(model_name=model_name, duration=self.PIN_DURATION)
        pin_id = self._model_pins.get(model_name)

        if pin_id is None:
            model_api = ModelsApi(self._inference_api_client)
            post_info = BodyPinModel(model_name=model_name, duration=self.PIN_DURATION)
            pin_response: Dict[str, Any] = await asyncio.to_thread(model_api.pin_model, post_info)
            if pin_response.get("status") != "success":
                raise RuntimeError(f"Failed to pin model '{model_name}': {pin_response}")
            self._model_pins[model_name] = pin_response.get("pin_id")

        image_base64 = base64_encode_png(image)

        # Build the RecognizePayload Pydantic model
        return_annotated = kwargs.get("return_annotated", False)
        payload = RecognizePayload(
            model_name=model_name,
            conf_thresh=conf_thresh,
            return_annotated=return_annotated,
            pin_id=pin_id,
            image_base64=image_base64,
        )

        # Call the /api/recognize-json endpoint, operationId: "recognize_json"
        inference_api = InferenceApi(self._inference_api_client)
        response: Dict[str, Any] = await asyncio.to_thread(inference_api.recognize_json, recognize_payload=payload)

        if "detections" not in response:
            raise RuntimeError(f"Recognition failed: {response}")

        detections = response["detections"]
        annotated_image = None

        if return_annotated and "annotated_image" in response:
            annotated_bytes = base64.b64decode(response["annotated_image"])
            nparr = np.frombuffer(annotated_bytes, np.uint8)
            annotated_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        return detections, annotated_image

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()  # Cancel the existing task if it's still running
        self._task = asyncio.get_running_loop().create_task(self._worker_task())  # Schedule a new task

    async def _init(self):
        """Initialization that should run on event loop"""
        try:
            await self._legacy_db_client.init_db()
        except Exception as e:
            logger.exception(e)
            raise

    async def _worker_task(self):
        """
        Periodic worker task that runs at regular intervals.
        """
        try:
            await self._init()
            while True:
                await asyncio.sleep(Manager.POLLING_INTERVAL)
        except asyncio.CancelledError:
            logger.info("Periodic task was cancelled.")
        finally:
            logger.info("Periodic task cleanup.")
