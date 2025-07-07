"""Manager models on Tasks Server"""

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from apps.helpers.db.db_client import DbClient
from apps.helpers.tasks.Task import Task
from apps.helpers.tasks.Registry import task_registry
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@dataclass
class TaskInfo:
    id: str
    status: str
    task: Task


class Manager:
    """Manages inference models, including loading, pinning, and unpinning."""

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks

    def __init__(self, db_client: DbClient):
        """
        Initialize the Manager.
        """
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations
        self._db_client: DbClient = db_client

    async def list_avail_tasks(self) -> List[str]:
        """
        List all tasks managed by the Manager.
        """
        return list(task_registry.keys())

    async def list_running_tasks(self) -> List[TaskInfo]:
        """
        List all tasks managed by the Manager.
        """
        return []

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()  # Cancel the existing task if it's still running
        self._task = asyncio.get_running_loop().create_task(self._worker_task())  # Schedule a new task

    async def _init(self):
        """Initialization that should run on event loop"""
        await self._db_client.init_db()

    async def _worker_task(self):
        """
        Periodic worker task that runs at regular intervals.
        """
        try:
            await self._init()
            while True:
                await asyncio.sleep(Manager.POLLING_INTERVAL)  # Wait for the polling interval
        except asyncio.CancelledError:
            logger.info("Periodic task was cancelled.")  # Handle task cancellation
        finally:
            logger.info("Periodic task cleanup.")  # Perform cleanup when the task is stopped
