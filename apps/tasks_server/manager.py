"""Manager models on Tasks Server"""

import asyncio
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from apps.helpers.db.db_client import DbClient, TaskRecord
from apps.helpers.tasks.Registry import task_registry
from apps.helpers.tasks.Task import Task

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@dataclass
class TaskMetadata:
    id: int
    typename: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = ""


@dataclass
class TaskInfo:
    task: Task
    metadata: TaskMetadata


class Manager:
    """Manages inference models, including loading, pinning, and unpinning."""

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks

    def __init__(self, db_client: DbClient):
        """
        Initialize the Manager.
        """
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations
        self._db_client: DbClient = db_client
        self._running_tasks: Dict[int, TaskInfo] = {}
        self._config = { "db_path": db_client.get_path() }


    async def get_server_config(self) -> Dict[str, Any]:
        return self._config.copy()

    async def list_avail_tasks(self) -> List[str]:
        """
        List all tasks managed by the Manager.
        """
        return list(task_registry.keys())

    async def get_task_result(self, task_id: int) -> Optional[dict[str, Any]]:
        """
        Get the result of a specific task.

        Args:
            task_id (int): The ID of the task.

        Returns:
            Optional[dict[str, Any]]: The result of the task, or None if not found.
        """
        task_info = self._running_tasks.get(task_id)
        if task_info:
            return task_info.task.get_results()

        # If not running, check the database
        return await self._db_client.get_task_result(task_id)

    async def get_tasks_status(self, task_ids: Optional[Union[List[int], int]] = None) -> Dict[int, dict[str, Any]]:
        """
        Returns status about a specified task, or all tasks.
        Combines DB and running tasks, with running tasks taking precedence.
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]

        running_tasks = (
            self._running_tasks
            if task_ids is None
            else {tid: self._running_tasks.get(tid) for tid in task_ids if tid in self._running_tasks}
        )

        missing_ids = None if task_ids is None else [tid for tid in task_ids if tid not in running_tasks]
        info_from_db = {}
        if missing_ids is None or len(missing_ids) > 0:
            db_tasks = await self._db_client.get_tasks(task_id=missing_ids)
            info_from_db = {
                task.id: {
                    "id": task.id,
                    "typename": task.typename,
                    "description": "",
                    "parameters": task.params_json,
                    "progress": None,
                    "status": task.status,
                }
                for task in db_tasks
            }

        running_info = {
            tid: {
                **asdict(task_info.metadata),
                "status": task_info.task.get_status(),
                "progress": task_info.task.get_progress(),
                "message": task_info.task.get_status_message(),
            }
            for tid, task_info in running_tasks.items()
            if task_info is not None
        }

        # Combine with running tasks overriding DB
        return {**info_from_db, **running_info}

    async def get_task_from_db(self, task_id: int) -> Optional[TaskInfo]:
        """
        Retrieve a task's information from the database.
        """
        task_data = await self._db_client.get_tasks([task_id])
        if not task_data or not len(task_data) > 0:
            return None
        task_data = task_data[0]

        params = json.loads(task_data.params_json) if task_data.params_json else {}
        task_instance = task_registry[task_data.typename](task_id=task_data.id, params=params)
        task_metadata = TaskMetadata(
            id=task_data.id,
            typename=task_data.typename,
            description="",
            parameters=params,
            status=task_data.status,
        )
        resume_data = json.loads(task_data.resume_data_json) if task_data.resume_data_json else {}
        task_instance.set_resume_data(resume_data)
        return TaskInfo(task=task_instance, metadata=task_metadata)

    async def create_new_task(self, typename: str, params: Dict[str, Any]) -> int:
        """
        Start a new task.

        Returns new task id
        """
        if typename not in task_registry:
            raise ValueError(f"Unknown task: {typename}")

        record: TaskRecord = await self._db_client.add_task(typename=typename, params=params)
        task_instance = task_registry[typename](task_id=record.id, params=params)

        task_metadata: TaskMetadata = TaskMetadata(
            id=record.id, typename=typename, parameters=params, status=Task.Status.PENDING
        )
        task_info = TaskInfo(task=task_instance, metadata=task_metadata)
        await self._start_task(task_info)
        return record.id

    async def delete_tasks(self, task_ids: Union[List[int], int]) -> None:
        """
        Delete a task by its ID.

        Args:
            task_id (int): The ID of the task to delete.
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]

        for task_id in task_ids:
            if task_id in self._running_tasks:
                # TODO: How does a task sync to DB?
                await self._running_tasks[task_id].task.stop()
                del self._running_tasks[task_id]
        await self._db_client.delete_tasks(task_ids)

    async def pause_tasks(self, task_ids: Union[List[int], int]) -> None:
        """
        Pause a task by its ID.

        Args:
            task_id (int): The ID of the task to pause.
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]

        task_infos: list[TaskInfo] = [
            self._running_tasks[task_id]
            for task_id in task_ids
            if task_id in self._running_tasks and self._running_tasks[task_id].metadata.status == Task.Status.RUNNING
        ]

        for task_info in task_infos:
            task_info.task.request_data_update()
            task_info.task.cancel_task()

        for task_info in task_infos:
            try:
                await task_info.task.wait_for_task_done(timeout=10.0)
                task_info.metadata.status = Task.Status.PAUSED
            except asyncio.TimeoutError as e:
                logger.warning(f"Timeout while waiting task pause: {e}")
                task_info.metadata.status = Task.Status.ERROR
            except StopIteration as e:
                task_info.metadata.status = Task.Status.COMPLETED
                logger.info(f"Task finished while waiting for data ready: {e}")

            try:
                del self._running_tasks[task_info.metadata.id]
                await self._save_task_data(task_info)
            except Exception as e:
                logger.exception(f"Error while saving task data: {e}")

    async def resume_tasks(self, task_ids: Union[List[int], int]) -> None:
        """
        Resume a paused task by its ID.

        Args:
            task_id (int): The ID of the task to resume.
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]

        for task_id in task_ids:
            if task_id in self._running_tasks:
                logger.warning(f"Can't resume task {task_id}: already in running tasks")
            else:
                task_info = await self.get_task_from_db(task_id)
                if task_info.metadata.status == Task.Status.PAUSED:
                    await self._start_task(task_info)
                else:
                    logger.warning(f"Can't resume task {task_id} from {task_info.metadata.status}")

    async def get_task_schema(self, typename: str) -> dict[str, dict[str, Any]]:
        """Get the schema for a specific task type."""
        if typename not in task_registry:
            raise ValueError(f"Unknown task: {typename}")
        return task_registry[typename].params_schema()

    async def _save_task_data(self, task_info: TaskInfo) -> None:
        """Save the resume data for a task."""
        if task_info.task.is_data_ready():
            await self._db_client.set_task_resume_data(task_info.metadata.id, task_info.task.get_resume_data())
            task_info.task.clear_data_ready()
        await self._db_client.set_task_result(task_info.metadata.id, task_info.task.get_results())
        await self._db_client.set_task_status(task_info.metadata.id, task_info.metadata.status)

    async def _start_task(self, task_info: TaskInfo) -> None:
        """Start a task."""
        self._running_tasks[task_info.metadata.id] = task_info
        await task_info.task.start()
        task_info.metadata.status = Task.Status.RUNNING

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()  # Cancel the existing task if it's still running
        self._task = asyncio.get_running_loop().create_task(self._worker_task())  # Schedule a new task

    async def _init(self):
        """Initialization that should run on event loop"""
        await self._db_client.init_db()

        # Clear any tasks left in DB that can't be resumed
        stale_tasks: List[TaskRecord] = await self._db_client.get_tasks(resumable=False)
        if stale_tasks:
            stale_ids = [info.id for info in stale_tasks]
            await self._db_client.delete_tasks(stale_ids)

    async def _worker_task(self):
        """
        Periodic worker task that runs at regular intervals.
        """
        try:
            await self._init()
            while True:
                await asyncio.sleep(Manager.POLLING_INTERVAL)  # TODO: Drive some of this with events?

                # Check if any data should be persisted to DB
                tasks_to_remove: set[int] = set()
                for task_id, task_info in dict(self._running_tasks).items():
                    task_done = task_info.task.is_task_done()
                    if task_done:
                        task_info.metadata.status = Task.Status.COMPLETED
                    await self._save_task_data(task_info)
                    if task_done:
                        tasks_to_remove.add(task_id)
                for task_id in tasks_to_remove:
                    del self._running_tasks[task_id]

        except asyncio.CancelledError:
            logger.info("Periodic task was cancelled.")  # Handle task cancellation
        finally:
            logger.info("Periodic task cleanup.")  # Perform cleanup when the task is stopped
