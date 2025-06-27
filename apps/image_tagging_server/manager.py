"""Manager for Image Tagging Server"""

import asyncio
import base64
import glob
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import aiofiles

import cv2
import numpy as np
from inference_client.api.inference_api import InferenceApi
from inference_client.api.models_api import ModelsApi
from inference_client.api_client import ApiClient
from inference_client.models.body_pin_model_api import (
    BodyPinModelApi,  # from updated schema
)
from inference_client.models.recognize_request import RecognizeRequest

from apps.helpers.db.db_client import DbClient

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@dataclass
class LabelInfo:
    id: int
    name: str
    color: str
    uuid: str


class Manager:
    """Manages image tagging tasks."""

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks
    PIN_DURATION: float = 30 * 60  # Timeout before unloading model

    def __init__(self, api_client: ApiClient, db_client: DbClient, files_root: Path):
        """Initialize the Manager"""
        self._api_client: ApiClient = api_client
        self._db_client: DbClient = db_client
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations
        self._files_root: Path = files_root
        self._model_pins: Dict[str, str] = {}

    @dataclass
    class FileEntry:
        name: str
        type: str  # "file" or "dir"
        path: str  # relative path from root

    async def list_files(
        self, rel_path: str = "/", glob_pattern: str = "*", recursive: bool = False
    ) -> List[FileEntry]:
        """
        List all files and directories in the given relative path under the root directory.

        Args:
            rel_path (str): Relative path from the root directory (default: "/").
            glob_pattern (str): Glob pattern to filter files (default: "*").
            recursive (bool): Whether to list files recursively (default: False).

        Returns:
            List[FileEntry]: List of FileEntry instances.
        """
        if not self._files_root or not os.path.isdir(self._files_root):
            logger.warning("Root directory is not set or does not exist.")
            return []

        try:
            # Normalize and join with root, prevent directory traversal
            safe_path = os.path.normpath(os.path.join(self._files_root, rel_path.lstrip("/")))
            if not safe_path.startswith(str(self._files_root)):
                logger.warning(f"Directory traversal attempt: {rel_path}")
                return []

            if not os.path.exists(safe_path) or not os.path.isdir(safe_path):
                logger.warning(f"Directory not found: {safe_path}")
                return []

            # Construct glob pattern path
            glob_path = (
                os.path.join(safe_path, "**", glob_pattern) if recursive else os.path.join(safe_path, glob_pattern)
            )
            matched_paths = glob.glob(glob_path, recursive=recursive)

            entries: List[Manager.FileEntry] = []
            for full_path in sorted(matched_paths):
                # Skip hidden files/folders
                basename = os.path.basename(full_path)
                if basename.startswith("."):
                    continue

                # Must be under the root
                if not os.path.commonpath([self._files_root, full_path]).startswith(str(self._files_root)):
                    continue

                entry_type = "dir" if os.path.isdir(full_path) else "file"
                rel_entry_path = os.path.relpath(full_path, self._files_root)
                entries.append(Manager.FileEntry(name=basename, type=entry_type, path=rel_entry_path))

            return entries

        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []

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

    async def list_models(self) -> List[str]:
        """
        List models available from the inference server

        Returns:
            List[str]: list of models
        """
        api = ModelsApi(self._api_client)
        # Use the new method matching operationId "list_models_api"
        response: Dict[str, Any] = await asyncio.to_thread(api.list_models_api)
        return response.get("models", [])

    async def add_label(self, name: str, color: str) -> LabelInfo:
        """
        Adds a new label to the database

        Returns:
            LabelInfo: Label metadata added to database
        """
        row = await self._db_client.add_label(name, color)
        return LabelInfo(**row)

    async def list_labels(self) -> List[LabelInfo]:
        """
        List all labels from the database.

        Returns:
            List[LabelInfo]: List of label metadata (id, name)
        """
        rows = await self._db_client.list_labels()
        return [LabelInfo(id=row["id"], name=row["name"], color=row["color"], uuid=row["uuid"]) for row in rows]

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
        model_api = ModelsApi(self._api_client)
        post_info = BodyPinModelApi(model_name=model_name, duration=self.PIN_DURATION)
        pin_id = self._model_pins.get(model_name)

        if pin_id is None:
            model_api = ModelsApi(self._api_client)
            post_info = BodyPinModelApi(model_name=model_name, duration=self.PIN_DURATION)
            pin_response: Dict[str, Any] = await asyncio.to_thread(
                model_api.pin_model_api, post_info  # matches operationId "pin_model_api"
            )
            if pin_response.get("status") != "success":
                raise RuntimeError(f"Failed to pin model '{model_name}': {pin_response}")
            self._model_pins[model_name] = pin_response.get("pin_id")

        # Encode image to PNG and base64 encode
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Failed to encode image")

        image_base64 = base64.b64encode(buffer).decode("utf-8")

        # Build the RecognizeRequest Pydantic model
        return_annotated = kwargs.get("return_annotated", False)
        request = RecognizeRequest(
            model_name=model_name,
            conf_thresh=conf_thresh,
            return_annotated=return_annotated,
            pin_id=pin_id,
            image_base64=image_base64,
        )

        # Call the /api/recognize-json endpoint, operationId: "recognize_json"
        inference_api = InferenceApi(self._api_client)
        response: Dict[str, Any] = await asyncio.to_thread(inference_api.recognize_json, recognize_request=request)

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
            await self._db_client.init_db()
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
