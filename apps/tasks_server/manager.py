"""Manager models on Tasks Server"""

import asyncio
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from apps.helpers.db.db_client import ActiveTaskMetadata, DbClient, TaskConfigMetadata
from apps.helpers.tasks.Registry import task_registry
from apps.helpers.tasks.Task import Task

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@dataclass
class RunningTaskInfo:
    task: Task
    config_metadata: TaskConfigMetadata
    task_metadata: ActiveTaskMetadata


class Manager:
    """Manages inference models, including loading, pinning, and unpinning."""

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks

    def __init__(self, db_client: DbClient):
        """
        Initialize the Manager.
        """
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations
        self._db_client: DbClient = db_client
        self._running_tasks: Dict[int, RunningTaskInfo] = {}
        self._config = {"db_path": db_client.get_path()}

    async def get_server_config(self) -> Dict[str, Any]:
        return self._config.copy()

    async def list_task_types(self) -> List[str]:
        """
        List all tasks managed by the Manager.
        """
        return list(task_registry.keys())

    async def get_task_result(self, task_ids: List[int]) -> Optional[dict[str, Any]]:
        """
        Get the result of specified tasks.

        Args:
            task_ids (List[int]): The IDs of the tasks.

        Returns:
            Optional[dict[str, Any]]: The result of the tasks, or None if not found.
        """
        tasks_info = {tid: self._running_tasks.get(tid) for tid in task_ids}
        tasks_to_delete = []

        results = {}
        for tid, task_info in tasks_info.items():
            if task_info:
                result = task_info.task.get_results()
                if result:
                    results[tid] = result
                    delete_on_success = task_info.config_metadata.params.get(Task.InternalKeys.ONESHOT_RESULT, False)
                    if delete_on_success:
                        tasks_to_delete.append(tid)
            else:
                # If not running, check the database
                results = await self._db_client.get_task_results(task_ids)
        if results and tasks_to_delete:
            asyncio.create_task(self.delete_tasks(task_ids=tasks_to_delete))

        return results

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
            db_tasks = await self._db_client.get_active_tasks(task_id=missing_ids)
            config_uuids = [task.config_uuid for task in db_tasks]
            config_metadata = await self._db_client.get_task_configs(config_uuids=config_uuids)
            info_from_db = {
                task.id: {
                    "id": task.id,
                    "typename": task.typename,
                    "description": "",
                    "config_metadata": config,
                    "message": "Not Running",
                    "progress": None,
                    "status": task.status,
                }
                for task, config in zip(db_tasks, config_metadata)
            }

        running_info = {
            tid: {
                **asdict(task_info.config_metadata),
                "status": task_info.task.get_status(),
                "progress": task_info.task.get_progress(),
                "message": task_info.task.get_status_message(),
            }
            for tid, task_info in running_tasks.items()
            if task_info is not None
        }

        # Combine with running tasks overriding DB
        return {**info_from_db, **running_info}

    async def get_task_config_from_db(self, task_config_uuid: str) -> Optional[TaskConfigMetadata]:
        """
        Retrieve a task's information from the database.
        """
        configs_metadata = await self._db_client.get_task_configs(config_uuids=task_config_uuid)
        if not configs_metadata or not len(configs_metadata) > 0:
            return None
        return configs_metadata[0]

    async def create_new_task_config(
        self, typename: str, params: Dict[str, Any], persistent: bool
    ) -> TaskConfigMetadata:
        """
        Start a new task.

        Returns new task metadata
        """
        if typename not in task_registry:
            raise ValueError(f"Unknown task: {typename}")

        params[Task.InternalKeys.PERSISTENT] = persistent
        task_config_metadata: TaskConfigMetadata = await self._db_client.add_task_config(
            name=f"{typename}_task", typename=typename, params=params
        )

        return task_config_metadata

    async def delete_task_configs(self, config_uuids: Union[List[str], str]) -> None:
        """
        Delete task configs by their UUIDs.

        Args:
            config_uuids (Union[List[str], str]): The UUIDs of the task configs to delete.
        """
        if isinstance(config_uuids, str):
            config_uuids = [config_uuids]
        await self._db_client.delete_task_config(config_uuids)

    async def delete_tasks(self, task_ids: Union[List[int], int]) -> None:
        """
        Delete tasks by their IDs.
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]

        for task_id in task_ids:
            if task_id in self._running_tasks:
                await self._running_tasks[task_id].task.stop()
                del self._running_tasks[task_id]
        await self._db_client.delete_active_tasks(task_ids)

    async def pause_tasks(self, task_ids: Union[List[int], int]) -> None:
        """
        Pause a task by its ID.

        Args:
            task_id (int): The ID of the task to pause.
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]

        task_infos: list[RunningTaskInfo] = [
            self._running_tasks[task_id]
            for task_id in task_ids
            if task_id in self._running_tasks
            and self._running_tasks[task_id].task_metadata.status == Task.Status.RUNNING
        ]

        for task_info in task_infos:
            task_info.task.request_data_update()
            task_info.task.cancel_task()

        for task_info in task_infos:
            try:
                await task_info.task.wait_for_task_done(timeout=10.0)
                task_info.config_metadata.status = Task.Status.PAUSED
            except asyncio.TimeoutError as e:
                logger.warning(f"Timeout while waiting task pause: {e}")
                task_info.config_metadata.status = Task.Status.ERROR
            except StopIteration as e:
                task_info.config_metadata.status = Task.Status.COMPLETED
                logger.info(f"Task finished while waiting for data ready: {e}")

            try:
                del self._running_tasks[task_info.task_metadata.id]
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
                task_info = await self.get_task_config_from_db(task_id)
                if task_info.config_metadata.status == Task.Status.PAUSED:
                    await self._start_task(task_info)
                else:
                    logger.warning(f"Can't resume task {task_id} from {task_info.config_metadata.status}")

    async def start_new_task(self, task_config_uuid: str) -> Optional[RunningTaskInfo]:
        """
        Start a configured task by its config uuid

        Args:
            task_config_uuid (str): The UUID of the task configuration to start.
        """
        task_config_metadata: TaskConfigMetadata = await self.get_task_config_from_db(task_config_uuid)
        if task_config_metadata:
            return await self._start_task(task_config_metadata)
        else:
            logger.warning(f"Unknown task configuration: {task_config_uuid}")

    async def cancel_tasks(self, task_ids: Union[List[int], int]) -> None:
        """
        Stop tasks by id, syncing to db

        Args:
            task_ids (Union[List[int], int]): The ID(s) of the task(s) to stop.
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]

        canceled_tasks = []
        for task_id in task_ids:
            if task_id not in self._running_tasks:
                logger.warning(f"Can't stop task {task_id}: not found in running tasks")
            else:
                task_info: RunningTaskInfo = self._running_tasks.pop(task_id)
                task_info.task.cancel_task()
                canceled_tasks.append(task_info)
                logger.info(f"Task {task_id} cancel request sent")

        for task_info in canceled_tasks:
            try:
                await task_info.task.wait_for_task_done(5.0)
                await self._db_client.set_task_status(task_info.task_metadata.id, task_info.task.get_status())
            except TimeoutError:
                logger.warning(f"Task {task_info.task_metadata.id} did not stop in time")

    async def get_task_schema(self, typename: str) -> dict[str, dict[str, Any]]:
        """Get the schema for a specific task type."""
        if typename not in task_registry:
            raise ValueError(f"Unknown task: {typename}")
        return task_registry[typename].params_schema()

    async def _save_task_data(self, task_info: RunningTaskInfo) -> None:
        """Save the resume data for a task."""
        if task_info.task.is_data_ready():
            await self._db_client.set_task_resume_data(task_info.task_metadata.id, task_info.task.get_resume_data())
            task_info.task.clear_data_ready()
        # TODO: Sync metadata and DB?
        await self._db_client.set_task_result(task_info.task_metadata.id, task_info.task.get_results())
        await self._db_client.set_task_status(task_info.task_metadata.id, task_info.task_metadata.status)

    async def _start_task(self, config_metadata: TaskConfigMetadata) -> Optional[RunningTaskInfo]:
        """Start a task."""
        task_instance = task_registry[config_metadata.typename](task_config_uuid=config_metadata.uuid, params=config_metadata.params)
        task_metadata: ActiveTaskMetadata = await self._db_client.insert_new_active_task(config_metadata.uuid)
        task_info: RunningTaskInfo = RunningTaskInfo(task=task_instance, config_metadata=config_metadata, task_metadata=task_metadata)

        self._running_tasks[task_metadata.id] = task_info
        await task_info.task.start()
        task_info.task_metadata.status = Task.Status.RUNNING
        return task_info

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()  # Cancel the existing task if it's still running
        self._task = asyncio.get_running_loop().create_task(self._worker_task())  # Schedule a new task

    async def _init(self):
        """Initialization that should run on event loop"""
        await self._db_client.init_db()

        # Clear any tasks left in DB that can't be resumed
        stale_tasks: List[TaskConfigMetadata] = await self._db_client.get_active_tasks(resumable=False)
        if stale_tasks:
            stale_ids = [info.id for info in stale_tasks]
            await self._db_client.delete_task_config(stale_ids)

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
                        task_info.config_metadata.status = Task.Status.COMPLETED
                    await self._save_task_data(task_info)
                    if task_done:
                        tasks_to_remove.add(task_id)
                for task_id in tasks_to_remove:
                    del self._running_tasks[task_id]

        except asyncio.CancelledError:
            logger.info("Periodic task was cancelled.")  # Handle task cancellation
        finally:
            logger.info("Periodic task cleanup.")  # Perform cleanup when the task is stopped
