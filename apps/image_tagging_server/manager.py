"""Manager for Image Tagging Server"""

import asyncio
import logging
import sys
from typing import Any, Dict, Optional

import db_client
import inference_client
import tasks_client
from db_client.api.classes_api import ClassesApi
from db_client.api.images_api import ImagesApi
from db_client.api.sources_api import SourcesApi
from db_client.api.tags_api import TagsApi
from inference_client.api.inference_api import InferenceApi
from inference_client.api.models_api import ModelsApi
from tasks_client.api.tasks_api import TasksApi

from apps.helpers.db.db_client import DbClient
from apps.helpers.webUtils import api_forward_request

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Manager:
    """Manages image tagging frontend"""

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

    def classes_api_request(self, api_method_name: str):
        """
        Decorator to forward request to the Classes API
        """
        return api_forward_request(ClassesApi(self._db_api_client), api_method_name)

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

    def tags_api_request(self, api_method_name: str):
        """
        Decorator to forward request to the Tags API
        """
        return api_forward_request(TagsApi(self._db_api_client), api_method_name)

    def task_api_request(self, api_method_name: str):
        """
        Decorator to forward request to the Task API
        """
        return api_forward_request(TasksApi(self._tasks_api_client), api_method_name)

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
            logger.error(e, exc_info=True)
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
