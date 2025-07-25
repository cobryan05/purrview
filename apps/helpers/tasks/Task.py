import asyncio
import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Optional
from enum import Enum

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

class Task(ABC):
    class Status(str, Enum):
        PENDING = "pending"
        RUNNING = "running"
        PAUSED = "paused"
        COMPLETED = "completed"
        ERROR = "error"

    def __init__(self, task_id: int, params: dict[str, Any]):
        self._task_id = task_id
        self._params: dict[str, Any] = params
        self._task: asyncio.Task | None = None
        self._progress: float = 0.0
        self._results: dict[str, Any] = {}
        self._resume_data: dict[str, Any] = {}
        self._task_done: asyncio.Event = asyncio.Event()
        self._cancel_flag: asyncio.Event = asyncio.Event()
        self._pause_flag: asyncio.Event = asyncio.Event()
        self._data_req_flag: asyncio.Event = asyncio.Event()
        self._data_ready_flag: asyncio.Event = asyncio.Event()

    @abstractmethod
    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        """
        Run any initialization logic for the task.

        Args:
            params (dict[str, Any]): The parameters for this task.

            resume_data (Optional[dict[str, Any]]): If the task is being resumed from a saved state,
                this will contain the serialized resume information. If None, this is a fresh task.

        This method is called once before the task starts running, and should perform any
        setup required to begin or resume execution.
        """
        pass

    @abstractmethod
    async def _run(self) -> dict[str, Any]:
        """
        Run the main logic of the task.

        Long-running tasks should check self._cancel_flag and
        self._pause_flag for stop conditions, and self._data_req_flag
        to check if a database update is requested.

        The _cancel_flag and _data_req_flag may be set at the same time. The
        _data_flag should take priority.

        Tasks can save data to the db by setting putting it in the
        self._resume_data dictionary. Tasks should set self._data_ready_flag
        to request the data be written to the database. Tasks should check
        the _data_req_flag to check if a _resume_data update is being requested.

        Tasks should periodically call self._update_progress(float perc).
        """
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
        await self._init(params=self._params, resume_data=self._resume_data)
        try:
            results = await self._run()
            self._results = {"status": Task.Status.COMPLETED, "data": results}
        except Exception as e:
            logger.exception(e)
            self._results = {"status": Task.Status.ERROR, "message": str(e)}
        finally:
            self._progress = 100.0
            await self._deinit()
            self._task_done.set()

    def _update_progress(self, percent: float):
        """Store progress"""
        self._progress = percent

    async def start(self) -> None:
        """Execute the task. Should periodically update progress and save resume data."""
        if self._task is None:
            self._task = asyncio.create_task(self._execute())
            return
        raise RuntimeError("Task is already running")

    async def pause(self) -> None:
        """Pause a task in a resumable way"""
        pass

    async def resume(self) -> None:
        """Resume a paused task"""
        pass

    async def stop(self) -> None:
        """Request a task to stop"""
        self._cancel_flag.set()

    async def kill(self) -> None:
        """Force a task to stop"""
        if self._task:
            self._task.cancel()

    def set_resume_data(self, data: dict[str, Any]) -> None:
        """Return the resume data for the task."""
        self._resume_data = data

    def get_resume_data(self) -> dict[str, Any]:
        """Return the resume data for the task."""
        return dict(self._resume_data)

    def get_status_message(self) -> str:
        """Returns any message to return with task status"""
        return ""

    def get_progress(self) -> float:
        """Return the current progress of the task."""
        return self._progress

    def get_results(self) -> dict[str, Any]:
        return dict(self._results)

    def typename(self) -> str:
        """Return the unique task type name"""
        return self.__class__.__name__

    def is_task_done(self) -> bool:
        """Return whether the task is done."""
        return self._task_done.is_set()

    def is_data_ready(self) -> bool:
        """Return whether the task has data ready for saving."""
        return self._data_ready_flag.is_set()

    def clear_data_ready(self) -> None:
        """Clear the data ready flag."""
        self._data_ready_flag.clear()

    def request_data_update(self) -> None:
        """Request a data update for the task."""
        self._data_req_flag.set()

    def clear_data_request(self) -> None:
        """Clear the data request flag."""
        self._data_req_flag.clear()

    async def wait_for_data_ready(self, timeout: float) -> None:
        """
        Wait until either data is ready to save

        Raises:
            asyncio.TimeoutError: if the timeout expires
            StopIteration: if the task is completed
        """
        done, _ = await asyncio.wait(
            [self._data_ready_flag.wait(), self._task_done.wait()],
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if self._data_ready_flag.is_set():
            return
        elif self._task_done.is_set():
            raise StopIteration("Task has completed.")
        else:
            raise asyncio.TimeoutError("Timed out waiting for task or data.")

    def cancel_task(self) -> None:
        """Request a data update for the task."""
        self._cancel_flag.set()

    async def wait_for_task_done(self, timeout: float) -> None:
        """
        Wait until the task is done.

        Raises:
            asyncio.TimeoutError: if the timeout expires
        """
        try:
            await asyncio.wait_for(self._task_done.wait(), timeout)
        except TimeoutError:
            raise asyncio.TimeoutError("Timed out waiting for task to complete.")
