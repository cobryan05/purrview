"""Manager for Image Tagging Server"""

import logging
import sys
from typing import Any, Dict

import db_client
import inference_client
import tasks_client

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Manager:
    """Manages image tagging frontend"""

    def __init__(
        self,
        inference_api_client: inference_client.ApiClient,
        tasks_api_client: tasks_client.ApiClient,
        db_api_client: db_client.ApiClient,
    ):
        """Initialize the Manager"""
        self._inference_api_client: inference_client.ApiClient = inference_api_client
        self._tasks_api_client: tasks_client.ApiClient = tasks_api_client
        self._db_api_client: db_client.ApiClient = db_api_client
        self._config = {
            "inference_server": inference_api_client.configuration.host,
            "tasks_server": tasks_api_client.configuration.host,
            "db_server": db_api_client.configuration.host,
        }

    async def get_server_config(self) -> Dict[str, Any]:
        return self._config.copy()
