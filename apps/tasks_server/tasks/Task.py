from __future__ import annotations

import asyncio
import logging
import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from apps.helpers.consts import TaskStatus

if TYPE_CHECKING:
    from ..manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Task(ABC):
    class InternalKeys(str, Enum):
        PERSISTENT = "_persistent"
        ONESHOT_RESULT = "_oneshot_result"  # delete the task when the result is fetched
        # TODO: Task *configs* need a way to delete or else Preview tasks can build up

    # How often (seconds) a running task pings the DB to confirm it is alive.
    # The stale-check threshold in the manager uses a multiple of this value.
    HEARTBEAT_INTERVAL: int = 30

    def __init__(self, task_config_uuid: str, params: dict[str, Any], manager: Manager):
        self._config_uuid: str = task_config_uuid
        self._instance_uuid: Optional[str] = None
        self._status: TaskStatus = TaskStatus.PENDING
        self._params: dict[str, Any] = params
        self._task: asyncio.Task | None = None
        self._progress: float = 0.0
        self._results: dict[str, Any] = {}
        self._resume_data: dict[str, Any] = {}
        self._log_lines: deque[str] = deque(maxlen=100)
        self._task_done: asyncio.Event = asyncio.Event()
        self._cancel_flag: asyncio.Event = asyncio.Event()
        # Starts set (running). pause() clears it; resume() sets it.
        # _check_pause awaits it — blocks while clear, passes through while set.
        self._resume_event: asyncio.Event = asyncio.Event()
        self._resume_event.set()
        self._data_req_flag: asyncio.Event = asyncio.Event()
        self._data_ready_flag: asyncio.Event = asyncio.Event()
        self._manager: Manager = manager
        self._last_heartbeat_time: float = 0.0

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

    @abstractmethod
    async def _run(self) -> dict[str, Any]:
        """
        Run the main logic of the task.

        Long-running tasks should check self._cancel_flag and
        self._resume_event for stop/pause conditions, and self._data_req_flag
        to check if a database update is requested.

        The _cancel_flag and _data_req_flag may be set at the same time. The
        _data_flag should take priority.

        Tasks can save data to the db by setting putting it in the
        self._resume_data dictionary. Tasks should set self._data_ready_flag
        to request the data be written to the database. Tasks should check
        the _data_req_flag to check if a _resume_data update is being requested.

        Tasks should periodically call self._update_progress(float perc).
        """

    @abstractmethod
    async def _deinit(self) -> None:
        """Run any cleanup logic for the task."""

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
        try:
            self.log("Initializing")
            await self._init(params=self._params, resume_data=self._resume_data)
            self._status = TaskStatus.RUNNING
            self.log("Running")
            results = await self._run()
            self._results = {"status": TaskStatus.COMPLETED, "data": results}
            self._status = TaskStatus.COMPLETED
            self.log("Completed")
        except Exception as e:
            self._status = TaskStatus.ERROR
            logger.error(e, exc_info=True)
            self._results = {"status": TaskStatus.ERROR, "message": str(e)}
            self.log(f"Error: {e}")
        finally:
            self._progress = 100.0
            await self._deinit()
            self._task_done.set()

    def _update_progress(self, percent: float):
        """Store progress"""
        self._progress = percent

    def log(self, msg: str) -> None:
        """Append a message to the in-memory log ring buffer and emit to logger."""
        logger.info(msg)
        self._log_lines.append(msg)

    def get_log_lines(self) -> list[str]:
        """Return a copy of recent log lines."""
        return list(self._log_lines)

    async def start(self) -> None:
        """Execute the task. Should periodically update progress and save resume data."""
        if self._task is None:
            self._task = asyncio.create_task(self._execute())
            return
        raise RuntimeError("Task is already running")

    async def pause(self) -> None:
        """Pause a task cooperatively in-place. Task stays in _running_tasks; DB status unchanged."""
        self._resume_event.clear()

    async def resume(self) -> None:
        """Resume a task that was paused in-place."""
        self._resume_event.set()

    def is_paused_in_place(self) -> bool:
        """Return True if the task is cooperatively paused in-place."""
        return not self._resume_event.is_set()

    async def _check_pause(self) -> None:
        """
        Cooperative pause checkpoint. Call this from task run loops.
        Blocks while paused; returns immediately when resumed or if cancel is requested.
        Uses asyncio.Event.wait() — zero polling overhead.

        Also sends a rate-limited heartbeat so the manager can detect tasks
        that have silently stopped iterating.
        """
        await self.heartbeat()
        if self._resume_event.is_set():
            return  # fast path: not paused
        cancel_task = asyncio.create_task(self._cancel_flag.wait())
        resume_task = asyncio.create_task(self._resume_event.wait())
        try:
            _, pending = await asyncio.wait({cancel_task, resume_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
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

    def set_instance_uuid(self, instance_uuid: str) -> None:
        """Set the DB instance UUID so the heartbeat loop can update it."""
        self._instance_uuid = instance_uuid

    async def heartbeat(self, force: bool = False) -> None:
        """Update the DB heartbeat timestamp to confirm this task is still progressing.

        Rate-limited: the DB write only fires once per HEARTBEAT_INTERVAL seconds
        unless *force* is True.  Pass ``force=True`` before a single long-running
        ``await`` so the timestamp is fresh going into it — otherwise a slow
        operation would cause the stale-heartbeat checker to incorrectly mark
        the task as dead.

        Called automatically by ``_check_pause()``; call explicitly from ``_run()``
        before any await that may take longer than HEARTBEAT_INTERVAL to complete.
        """
        if not self._instance_uuid:
            return
        now = time.monotonic()
        if not force and (now - self._last_heartbeat_time) < self.HEARTBEAT_INTERVAL:
            return
        self._last_heartbeat_time = now
        try:
            await self._manager.update_task_heartbeat(self._instance_uuid)
        except Exception:
            logger.warning("Heartbeat update failed for task %s", self._instance_uuid)

    def get_resume_data(self) -> dict[str, Any]:
        """Return the resume data for the task."""
        return dict(self._resume_data)

    def get_status(self) -> TaskStatus:
        """Returns the current status"""
        return self._status

    def get_status_message(self) -> str:
        """Returns any message to return with task status"""
        return ""

    def get_status_actions(self) -> list[dict]:
        """Returns action buttons to show in the live task status UI while the task is running."""
        return []

    def get_notices(self) -> list[dict]:
        """Return active advisory notices for this task (shown in the UI notices bar).

        Each notice is a dict with keys:
            key     — stable identifier (str)
            level   — "info" | "warning" | "error"
            message — human-readable string
        Override in subclasses to surface task-specific status to the frontend.
        """
        return []

    def get_progress(self) -> float:
        """Return the current progress of the task."""
        return self._progress

    def get_results(self) -> Optional[dict[str, Any]]:
        """Return results from this task"""
        return self._results.copy() if self._results else None

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
            [
                asyncio.create_task(self._data_ready_flag.wait()),
                asyncio.create_task(self._task_done.wait()),
            ],
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

    async def wait_for_task_done(self, timeout: Optional[float] = None) -> None:
        """
        Wait until the task is done.

        Raises:
            asyncio.TimeoutError: if the timeout expires
        """
        try:
            await asyncio.wait_for(self._task_done.wait(), timeout)
        except (asyncio.TimeoutError, TimeoutError):
            # asyncio.wait_for raises asyncio.TimeoutError, which on Python 3.10
            # is not the builtin TimeoutError — catch both to be version-safe.
            raise asyncio.TimeoutError("Timed out waiting for task to complete.")

    @staticmethod
    async def _read_int_setting(settings_api: Any, key: str, default: int) -> int:
        """Read an integer setting from the DB settings API; return default on any failure."""
        try:
            resp = await asyncio.to_thread(settings_api.get_setting, key)
            return int(resp.value)
        except Exception:
            return default
