"""Manager models on Tasks Server"""

import asyncio
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import db_client
import inference_client

from apps.helpers.consts import TaskStatus
from apps.helpers.db.db_client import ActiveTaskMetadata, DbClient, TaskConfigMetadata

from .tasks.Registry import task_registry
from .tasks.Task import Task

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

    def __init__(
        self,
        db_api_client: db_client.ApiClient,
        legacy_db_client: DbClient,
        inference_api_client: inference_client.ApiClient,
    ):
        """
        Initialize the Manager.
        """
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations
        self._legacy_db_client: DbClient = legacy_db_client
        self._inference_api_client: inference_client.ApiClient = inference_api_client
        self._db_api_client: db_client.ApiClient = db_api_client
        self._running_tasks: Dict[int, RunningTaskInfo] = {}
        self._config = {
            "db_path": legacy_db_client.get_path(),
            "db_server": db_api_client.configuration.host,
            "inference_server": inference_api_client.configuration.host,
        }

    def get_db_api_client(self) -> db_client.ApiClient:
        return self._db_api_client

    def get_inference_api_client(self) -> inference_client.ApiClient:
        return self._inference_api_client

    async def get_server_config(self) -> Dict[str, Any]:
        return self._config.copy()

    async def list_task_types(self) -> List[str]:
        """
        List all tasks managed by the Manager.
        """
        return list(task_registry.keys())

    async def get_tasks_result(self, task_ids: Union[List[int], int]) -> Optional[dict[str, Any]]:
        """
        Get the result of specified tasks.

        Args:
            task_ids (List[int]): The IDs of the tasks.

        Returns:
            Optional[dict[str, Any]]: The result of the tasks, or None if not found.
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]

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
                results = await self._legacy_db_client.get_task_results(task_ids)
        if results and tasks_to_delete:
            asyncio.create_task(self.delete_tasks(task_ids=tasks_to_delete))

        return results

    async def get_tasks_instance_info(
        self, task_ids: Optional[Union[List[int], int]] = None
    ) -> Dict[int, dict[str, Any]]:
        """
        Returns status about a specified task, or all tasks.
        Combines DB and running tasks, with running tasks taking precedence.
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]

        # TODO: Ensure this is all merged well
        running_tasks = (
            self._running_tasks
            if task_ids is None
            else {tid: self._running_tasks.get(tid) for tid in task_ids if tid in self._running_tasks}
        )

        missing_ids = None if task_ids is None else [tid for tid in task_ids if tid not in running_tasks]
        info_from_db = {}
        if missing_ids is None or len(missing_ids) > 0:
            db_tasks = await self._legacy_db_client.get_active_tasks(task_id=missing_ids)
            config_uuids = list(set(task.config_uuid for task in db_tasks))
            config_metadata = await self._legacy_db_client.get_task_configs(config_uuids=config_uuids)
            config_map = {config.uuid: config for config in config_metadata}
            info_from_db = {
                task.id: {
                    "id": task.id,
                    "typename": task.typename,
                    "description": "",
                    "config_metadata": config_map.get(task.config_uuid),
                    "message": "Not Running",
                    "progress": None,
                    "status": task.status,
                }
                for task in db_tasks
            }

        running_info = {
            tid: {
                **task_info.config_metadata.dict(),
                "status": task_info.task.get_status(),
                "progress": task_info.task.get_progress(),
                "message": task_info.task.get_status_message(),
            }
            for tid, task_info in running_tasks.items()
            if task_info is not None
        }

        # Combine with running tasks overriding DB
        return {**info_from_db, **running_info}

    async def create_new_task_config(
        self, name: str, typename: str, params: Dict[str, Any], persistent: bool
    ) -> TaskConfigMetadata:
        """
        Start a new task.

        Returns new task metadata
        """
        if typename not in task_registry:
            raise ValueError(f"Unknown task: {typename}")

        params[Task.InternalKeys.PERSISTENT] = persistent
        task_config_metadata: TaskConfigMetadata = await self._legacy_db_client.add_task_config(
            name=name, typename=typename, params=params
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
        await self._legacy_db_client.delete_task_config(config_uuids)

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
        await self._legacy_db_client.delete_active_tasks(task_ids)

    async def update_task_config(
        self,
        config_uuid: str,
        name: Optional[str] = None,
        typename: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        marked_for_delete: Optional[bool] = None,
    ) -> None:
        """
        Update an existing task configuration.

        Args:
            config_uuid (str): UUID of the task config to update.
            name (Optional[str]): New name.
            typename (Optional[str]): New typename.
            params (Optional[Dict[str, Any]]): New parameters.
            description (Optional[str]): New description.
            marked_for_delete (Optional[bool]): Flag to mark for deletion.
        """
        if typename is not None and typename not in task_registry:
            raise ValueError(f"Unknown task type: {typename}")

        # Update in database
        await self._legacy_db_client.update_task_config(
            config_uuid=config_uuid,
            name=name,
            typename=typename,
            params=params,
            description=description,
            marked_for_delete=marked_for_delete,
        )

    async def get_task_configs(
        self, task_config_uuids: Optional[Union[List[str], str]]
    ) -> dict[str, TaskConfigMetadata]:
        """
        Get task configurations by their IDs.
        """
        """
        Retrieve a task's information from the database.
        """
        if isinstance(task_config_uuids, str):
            task_config_uuids = [task_config_uuids]

        configs_metadata = await self._legacy_db_client.get_task_configs(config_uuids=task_config_uuids)
        return {cfg.uuid: cfg for cfg in configs_metadata}

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
            and self._running_tasks[task_id].task_metadata.status == TaskStatus.RUNNING
        ]

        for task_info in task_infos:
            task_info.task.request_data_update()
            task_info.task.cancel_task()

        for task_info in task_infos:
            try:
                await task_info.task.wait_for_task_done(timeout=10.0)
                task_info.config_metadata.status = TaskStatus.PAUSED
            except asyncio.TimeoutError as e:
                logger.warning(f"Timeout while waiting task pause: {e}")
                task_info.config_metadata.status = TaskStatus.ERROR
            except StopIteration as e:
                task_info.config_metadata.status = TaskStatus.COMPLETED
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
                task_info = (await self.get_task_configs(task_id)).get(task_id)
                if task_info and task_info.config_metadata.status == TaskStatus.PAUSED:
                    await self._start_task(task_info)
                else:
                    logger.warning(f"Can't resume task {task_id} from {task_info.config_metadata.status}")

    async def start_new_task(self, task_config_uuid: str) -> Optional[RunningTaskInfo]:
        """
        Start a configured task by its config uuid

        Args:
            task_config_uuid (str): The UUID of the task configuration to start.
        """
        task_config_metadata: dict[str, TaskConfigMetadata] = await self.get_task_configs([task_config_uuid])
        metadata = task_config_metadata.get(task_config_uuid)
        if metadata:
            return await self._start_task(metadata)
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
                await self._legacy_db_client.set_task_status(task_info.task_metadata.id, task_info.task.get_status())
            except TimeoutError:
                logger.warning(f"Task {task_info.task_metadata.id} did not stop in time")

    async def get_tasks_type_schema(self, typenames: Union[List[str], str]) -> dict[str, dict[str, Any]]:
        """Get the schema for a specific task type."""
        if isinstance(typenames, str):
            typenames = [typenames]

        schemas = {}
        for typename in typenames:
            if typename not in task_registry:
                raise ValueError(f"Unknown task: {typename}")
            schemas[typename] = task_registry[typename].params_schema()
        return schemas

    async def _save_task_data(self, task_info: RunningTaskInfo) -> None:
        """Save the resume data for a task."""
        if task_info.task.is_data_ready():
            await self._legacy_db_client.set_task_resume_data(
                task_info.task_metadata.id, task_info.task.get_resume_data()
            )
            task_info.task.clear_data_ready()
        # TODO: Sync metadata and DB?
        await self._legacy_db_client.set_task_result(task_info.task_metadata.id, task_info.task.get_results())
        await self._legacy_db_client.set_task_status(task_info.task_metadata.id, task_info.task_metadata.status)

    async def _start_task(self, config_metadata: TaskConfigMetadata) -> Optional[RunningTaskInfo]:
        """Start a task."""
        task_instance = task_registry[config_metadata.typename](
            task_config_uuid=config_metadata.uuid, params=config_metadata.params, manager=self
        )
        task_metadata: ActiveTaskMetadata = await self._legacy_db_client.insert_new_active_task(config_metadata.uuid)
        task_info: RunningTaskInfo = RunningTaskInfo(
            task=task_instance, config_metadata=config_metadata, task_metadata=task_metadata
        )

        self._running_tasks[task_metadata.id] = task_info
        await task_info.task.start()
        task_info.task_metadata.status = TaskStatus.RUNNING
        return task_info

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()  # Cancel the existing task if it's still running
        self._task = asyncio.get_running_loop().create_task(self._worker_task())  # Schedule a new task

    async def _init(self):
        """Initialization that should run on event loop"""
        await self._legacy_db_client.init_db()

        # Clear any tasks left in DB that can't be resumed
        stale_tasks: List[TaskConfigMetadata] = await self._legacy_db_client.get_active_tasks(resumable=False)
        if stale_tasks:
            stale_ids = [info.id for info in stale_tasks]
            await self._legacy_db_client.delete_task_config(stale_ids)

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
                        task_info.config_metadata.status = TaskStatus.COMPLETED
                    await self._save_task_data(task_info)
                    if task_done:
                        tasks_to_remove.add(task_id)
                for task_id in tasks_to_remove:
                    del self._running_tasks[task_id]

        except asyncio.CancelledError:
            logger.info("Periodic task was cancelled.")  # Handle task cancellation
        finally:
            logger.info("Periodic task cleanup.")  # Perform cleanup when the task is stopped
