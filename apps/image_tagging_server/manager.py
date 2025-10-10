"""Manager for Image Tagging Server"""

import asyncio
import base64
import fnmatch
import glob
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import cv2
import db_client
import inference_client
import numpy as np
import tasks_client
from db_client.api.labels_api import LabelsApi
from db_client.api.sources_api import SourcesApi
from inference_client.api.inference_api import InferenceApi
from inference_client.api.models_api import ModelsApi
from inference_client.models.associate_label_with_model_class_payload import (
    AssociateLabelWithModelClassPayload,
)
from inference_client.models.body_pin_model import BodyPinModel
from inference_client.models.recognize_payload import RecognizePayload
from pydantic import BaseModel
from tasks_client.api.tasks_api import TasksApi

from apps.helpers.db.db_client import (
    BoundingBoxMetadata,
    DbClient,
    ImageMetadata,
    LabelMetadata,
)
from apps.helpers.fileUtils import get_safe_path
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
        files_root: Path,
    ):
        """Initialize the Manager"""
        self._inference_api_client: inference_client.ApiClient = inference_api_client
        self._tasks_api_client: tasks_client.ApiClient = tasks_api_client
        self._db_api_client: db_client.ApiClient = db_api_client
        self._legacy_db_client: DbClient = legacy_db_client
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations
        self._files_root: Path = files_root
        self._model_pins: Dict[str, str] = {}
        self._config = {
            "files_root": str(files_root),
            "inference_server": inference_api_client.configuration.host,
            "tasks_server": tasks_api_client.configuration.host,
            "db_server": db_api_client.configuration.host,
        }

    async def get_server_config(self) -> Dict[str, Any]:
        return self._config.copy()

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

    @dataclass
    class FileEntry:
        name: str
        type: str  # "file" or "dir"
        path: str  # relative path from root

    async def list_files(
        self,
        rel_path: str = "/",
        patterns: str | List[str] = "*",
        exclude_patterns: str | List[str] = "*.json",
        recursive: bool = False,
    ) -> List[FileEntry]:
        """
        List all files and directories in the given relative path under the root directory.

        Args:
            rel_path (str): Relative path from the root directory (default: "/").
            include_patterns (str or List[str]): Glob pattern(s) to include (default: "*").
            exclude_patterns (str or List[str]): Glob pattern(s) to exclude (default: "").
            recursive (bool): Whether to list files recursively (default: False).

        Returns:
            List[FileEntry]: List of FileEntry instances.
        """
        if not self._files_root or not os.path.isdir(self._files_root):
            logger.warning("Root directory is not set or does not exist.")
            return []

        safe_path = get_safe_path(self._files_root, rel_path)

        if not os.path.exists(safe_path) or not os.path.isdir(safe_path):
            logger.warning(f"Directory not found: {safe_path}")
            return []

        # Normalize patterns
        if isinstance(patterns, str):
            patterns = [patterns]
        if isinstance(exclude_patterns, str):
            exclude_patterns = [exclude_patterns]

        # Get matching files
        glob_path = os.path.join(safe_path, "**", "*") if recursive else os.path.join(safe_path, "*")
        matched_paths = glob.glob(glob_path, recursive=recursive)

        entries: List[Manager.FileEntry] = []
        for full_path in sorted(matched_paths):
            basename = os.path.basename(full_path)

            # Skip hidden files/folders
            if basename.startswith("."):
                continue

            # Skip if outside root
            if not os.path.commonpath([self._files_root, full_path]).startswith(str(self._files_root)):
                continue

            # Apply include/exclude pattern matching
            rel_entry_path = os.path.relpath(full_path, self._files_root)
            matched = any(fnmatch.fnmatch(rel_entry_path, pat) for pat in patterns)
            excluded = any(fnmatch.fnmatch(rel_entry_path, pat) for pat in exclude_patterns)

            if matched and not excluded:
                entry_type = "dir" if os.path.isdir(full_path) else "file"
                entries.append(Manager.FileEntry(name=basename, type=entry_type, path=rel_entry_path))

        return entries

    async def open_file(self, rel_path: str) -> Optional[Tuple[aiofiles.threadpool.binary.AsyncBufferedReader, str]]:
        """
        Securely open a file under the root and return an aiofiles stream and filename.

        Args:
            rel_path (str): Relative path under the image root.

        Returns:
            Tuple[aiofiles.AsyncBufferedReader, str] or None
        """
        if not self._files_root or not os.path.isdir(self._files_root):
            logger.warning("Root directory not set or invalid")
            return None

        try:
            abs_path = os.path.normpath(os.path.join(self._files_root, rel_path.lstrip("/")))

            # Prevent directory traversal
            if not abs_path.startswith(str(self._files_root)):
                logger.warning(f"Blocked directory traversal attempt: {rel_path}")
                return None

            if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
                logger.warning(f"File not found: {abs_path}")
                return None

            f = await aiofiles.open(abs_path, mode="rb")
            filename = os.path.basename(abs_path)
            return f, filename

        except Exception as e:
            logger.error(f"Failed to open file {rel_path}: {e}")
            return None

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

    async def get_image_metadata(self, image_rel_path: str) -> Optional[ImageMetadata]:
        """
        Get metadata for image by resolving image ID from filename.

        Args:
            image_rel_path (str): Relative image path (filename).

        Returns:
            Optional[ImageMetadata]: Full metadata object or None.
        """
        safe_path = get_safe_path(self._files_root, image_rel_path)
        if not safe_path or not safe_path.exists():
            return None
        img_path = str(safe_path)
        image_id = await self._legacy_db_client.get_image_id_by_filename(img_path)
        if image_id is None:
            image_id = await self._legacy_db_client.add_image(img_path)
            await self._legacy_db_client.read_image_metadata_json_to_db(img_path)
        return await self._legacy_db_client.read_image_metadata_from_db(image_id)

    async def update_image_metadata(self, image_rel_path: str, metadata: ImageMetadata) -> None:
        """
        Update metadata for image identified by filename.

        Args:
            image_rel_path (str): Relative image path (filename).
            metadata (ImageMetadata): New metadata to write.
        """
        safe_path = get_safe_path(self._files_root, image_rel_path)
        if safe_path is None or not safe_path.exists():
            logger.warning(f"Path not found or inaccessible: {image_rel_path}")
            return
        img_path = str(safe_path)
        image_id = await self._legacy_db_client.add_image(img_path)
        await self._legacy_db_client.write_image_metadata_to_db(image_id, metadata)
        await self._legacy_db_client.save_image_metadata_db_to_json(img_path)

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
