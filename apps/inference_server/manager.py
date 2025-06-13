"""Manages streams on the MediaMTX server"""
from dataclasses import dataclass
from typing import Any, Optional, Dict
import asyncio
import logging
import sys

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Manager:

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks

    def __init__(self):
        """Initialize the Manager with an API client"""
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()  # Cancel the existing task if it's still running
        self._task = asyncio.get_running_loop().create_task(self._worker_task())  # Schedule a new task

    async def _init(self):
        """Initialize on worker thread"""
        pass

    async def _worker_task(self):
        """Periodic worker task that runs at regular intervals"""
        try:
            await self._init()
            while True:
                await asyncio.sleep(Manager.POLLING_INTERVAL)  # Wait for the polling interval
        except asyncio.CancelledError:
            print("Periodic task was cancelled.")  # Handle task cancellation
        finally:
            print("Periodic task cleanup.")  # Perform cleanup when the task is stopped
