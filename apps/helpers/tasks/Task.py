from abc import ABC, abstractmethod
from typing import Any
import asyncio
import logging
import sys

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Task(ABC):
    def __init__(self, task_id: int, params: dict[str, Any]):
        self._task_id = task_id
        self._params = params
        self._task: asyncio.Task | None = None
        self._progress: float = 0.0
        self._results: dict[str, Any] = {"status": "pending"}

    @abstractmethod
    async def _init(self) -> None:
        """Run any initialization logic for the task."""
        pass

    @abstractmethod
    async def _run(self) -> dict[str, Any]:
        """Run the main logic of the task."""
        pass

    @abstractmethod
    async def _deinit(self) -> None:
        """Run any cleanup logic for the task."""
        pass

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        """
        Return a schema describing the parameters for this Task.
        Each key is a parameter name, value is a dict with:
            - type: str
            - required: bool
            - default: Any (optional)
            - help: str (optional)
            - options: list (optional, for enums)
            - schema: dict (optional, for nested objects)
        """
        return {}

    async def _execute(self) -> None:
        await self._init()
        try:
            self._results = await self._run()
            self._results = {"status": "success", "data": self._results}
        except Exception as e:
            logger.exception(e)
            self._results = {"status": "error", "message": str(e)}
        finally:
            self._progress = 100.0
            await self._deinit()

    def _update_progress(self, percent: float):
        """Store progress"""
        self._progress = percent

    async def start(self) -> None:
        """Execute the task. Should periodically update progress and save resume data."""
        if self._task is None:
            self._task = asyncio.create_task(self._execute())
            return
        raise RuntimeError("Task is already running")

    async def stop(self) -> None:
        """Interrupt the task."""
        if self._task:
            self._task.cancel()

    def get_progress(self) -> float:
        """Return the current progress of the task."""
        return self._progress

    def get_results(self) -> dict[str, Any]:
        return self._results

    def typename(self) -> str:
        """Return the unique task type name"""
        return self.__class__.__name__
