"""Manager for Training Server"""

import asyncio
import logging
import sys
from typing import Any, Dict, Optional

from apps import APPS

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Manager:
    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._config: Dict[str, Any] = {"service": APPS.TRAINING_SERVER}

    async def get_server_config(self) -> Dict[str, Any]:
        return self._config.copy()

    def start(self):
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._worker_task())

    async def _worker_task(self):
        try:
            logger.info("Training server worker started.")
        except asyncio.CancelledError:
            logger.info("Training worker cancelled.")
        finally:
            logger.info("Training worker cleanup.")
