"""Manager models on Tasks Server"""

import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional, Union

import db_client
import inference_client

from apps.helpers.consts import TaskStatus
from apps.helpers.db.db_client import DbClient
from apps.helpers.types import TaskConfigMetadata, TaskInstanceMetadata, TaskResult
from apps.tasks_server.types import TaskInfo

from .tasks.Registry import task_registry
from .tasks.Task import Task

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


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
        self._running_tasks: Dict[str, TaskInfo] = {}
        self._config = {
            "db_path": None, #legacy_db_client.get_path(),
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

    async def get_tasks_result(self, task_ids: Union[List[str], str]) -> dict[str, TaskResult]:
        """
        Get the result of specified tasks.

        Args:
            task_ids (List[str]): The UUIDs of the tasks.

        Returns:
            Optional[dict[str, Any]]: The result of the tasks, or None if not found.
        """
        if isinstance(task_ids, str):
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
            asyncio.create_task(self.delete_tasks(task_uuids=tasks_to_delete))

        return results

    async def get_tasks_instance_info(
        self, task_uuids: Union[List[str], str, None] = None
    ) -> Dict[str, TaskInfo]:
        """
        Returns status about a specified task, or all tasks.
        Combines DB and running tasks, with running tasks taking precedence.
        """
        if isinstance(task_uuids, str):
            task_uuids = [task_uuids]

        uuids_left = task_uuids.copy() if task_uuids is not None else None
        ret_info: Dict[str, TaskInfo] = {}
        for uuid, task in self._running_tasks.items():
            if uuids_left is None or uuid in uuids_left:
                ret_info[uuid] = task
                if uuids_left is not None:
                    uuids_left.remove(uuid)

        if uuids_left is None or len(uuids_left) > 0:
            db_task_inst_metadata = await self._legacy_db_client.get_tasks(task_uuid=uuids_left)
            config_uuids = list(set(task_meta.config_uuid for task_meta in db_task_inst_metadata))
            config_metadata = await self._legacy_db_client.get_task_configs(config_uuids=config_uuids)
            for inst_metadata in db_task_inst_metadata:
                config_meta = config_metadata[inst_metadata.config_uuid]
                running_task = self._running_tasks.get(inst_metadata.uuid)
                task_info = TaskInfo(config_metadata=config_meta, task_metadata=inst_metadata, task=running_task)
                ret_info[inst_metadata.uuid] = task_info

        return ret_info

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

    async def delete_tasks(self, task_uuids: Union[List[str], str]) -> List[str]:
        """
        Delete tasks by their UUIDs.

        Returns:
            List[str]: The UUIDs of the deleted tasks.
        """
        if isinstance(task_uuids, str):
            task_uuids = [task_uuids]

        deleted_uuids = []
        for task_id in task_uuids:
            if task_id in self._running_tasks:
                await self._running_tasks[task_id].task.stop()
                del self._running_tasks[task_id]
                deleted_uuids.append(task_id)
        await self._legacy_db_client.delete_active_tasks(task_uuids)
        return deleted_uuids

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
        self, task_config_uuids: Union[List[str], str, None]
    ) -> dict[str, TaskConfigMetadata]:
        """
        Get task configurations by their IDs.
        """
        if isinstance(task_config_uuids, str):
            task_config_uuids = [task_config_uuids]

        configs_metadata = await self._legacy_db_client.get_task_configs(config_uuids=task_config_uuids)
        return configs_metadata

    async def pause_tasks(self, task_uuids: Union[List[str], str]) -> List[str]:
        """
        Pause tasks by their UUID.

        Args:
            task_uuids (list[str]): The UUIDs of the tasks to pause.
        """
        if isinstance(task_uuids, str):
            task_uuids = [task_uuids]

        task_infos: list[TaskInfo] = [
            self._running_tasks[task_id]
            for task_id in task_uuids
            if task_id in self._running_tasks
            and self._running_tasks[task_id].task_metadata.status == TaskStatus.RUNNING
        ]

        for task_info in task_infos:
            task_info.task.request_data_update()
            task_info.task.cancel_task()

        paused_tasks = []
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
            paused_tasks.append(task_info.task_metadata.uuid)

        return paused_tasks

    async def resume_tasks(self, task_uuids: Union[List[str], str]) -> List[str]:
        """
        Resume a paused task by its UUID.

        Args:
            task_uuid (str): The UUID of the task to resume.
        """
        if isinstance(task_uuids, str):
            task_uuids = [task_uuids]

        resumed_tasks = []
        for task_uuid in task_uuids:
            if task_uuid in self._running_tasks:
                logger.warning(f"Can't resume task {task_uuid}: already in running tasks")
            else:
                task_info = (await self.get_task_configs(task_uuid)).get(task_uuid)
                if task_info and task_info.config_metadata.status == TaskStatus.PAUSED:
                    await self._start_task(task_info)
                else:
                    logger.warning(f"Can't resume task {task_uuid} from {task_info.config_metadata.status}")

        return resumed_tasks

    async def start_new_task(self, task_config_uuid: str) -> Optional[TaskInfo]:
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

    async def cancel_tasks(self, task_uuids: Union[List[str], str]) -> List[str]:
        """
        Stop tasks by id, syncing to db

        Args:
            task_uuids (Union[List[str], str]): The UUID(s) of the task(s) to stop.
        """
        if isinstance(task_uuids, str):
            task_uuids = [task_uuids]

        canceled_tasks = []
        for task_uuid in task_uuids:
            if task_uuid not in self._running_tasks:
                logger.warning(f"Can't stop task {task_uuid}: not found in running tasks")
            else:
                task_info: TaskInfo = self._running_tasks.pop(task_uuid)
                task_info.task.cancel_task()
                canceled_tasks.append(task_info)
                logger.info(f"Task {task_uuid} cancel request sent")

        stopped_tasks = []
        for task_info in canceled_tasks:
            try:
                await task_info.task.wait_for_task_done(5.0)
                await self._legacy_db_client.set_task_status(task_info.task_metadata.uuid, task_info.task.get_status())
                stopped_tasks.append(task_info.task_metadata.uuid)
            except TimeoutError:
                logger.warning(f"Task {task_info.task_metadata.uuid} did not stop in time")

        return stopped_tasks

    async def get_tasks_type_schema(self, typenames: Union[List[str], str]) -> dict[str, dict]:
        """Get the schema for a specific task type."""
        if isinstance(typenames, str):
            typenames = [typenames]

        schemas = {}
        for typename in typenames:
            if typename not in task_registry:
                raise ValueError(f"Unknown task: {typename}")
            schemas[typename] = task_registry[typename].params_schema()
        return schemas

    async def _save_task_data(self, task_info: TaskInfo) -> None:
        """Save the resume data for a task."""
        if task_info.task.is_data_ready():
            await self._legacy_db_client.set_task_resume_data(
                task_info.task_metadata.uuid, task_info.task.get_resume_data()
            )
            task_info.task.clear_data_ready()
        # TODO: Sync metadata and DB?
        await self._legacy_db_client.set_task_result(task_info.task_metadata.uuid, task_info.task.get_results())
        await self._legacy_db_client.set_task_status(task_info.task_metadata.uuid, task_info.task_metadata.status)

    async def _start_task(self, config_metadata: TaskConfigMetadata) -> Optional[TaskInfo]:
        """Start a task."""
        task_instance = task_registry[config_metadata.typename](
            task_config_uuid=config_metadata.uuid, params=config_metadata.params, manager=self
        )
        task_metadata: TaskInstanceMetadata = await self._legacy_db_client.insert_new_active_task(config_metadata.uuid)
        task_info: TaskInfo = TaskInfo(task=task_instance, config_metadata=config_metadata, task_metadata=task_metadata)

        self._running_tasks[task_metadata.uuid] = task_info
        await task_instance.start()
        task_metadata.status = TaskStatus.RUNNING
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
        stale_tasks: List[TaskInstanceMetadata] = await self._legacy_db_client.get_tasks(resumable=False)
        if stale_tasks:
            stale_ids = [info.uuid for info in stale_tasks]
            await self._legacy_db_client.delete_active_tasks(stale_ids)

    async def _worker_task(self):
        """
        Periodic worker task that runs at regular intervals.
        """
        return
        try:
            await self._init()
            while True:
                await asyncio.sleep(Manager.POLLING_INTERVAL)  # TODO: Drive some of this with events?

                # Check if any data should be persisted to DB
                tasks_to_remove: set[int] = set()
                for task_uuid, task_info in dict(self._running_tasks).items():
                    task_done = task_info.task.is_task_done()
                    if task_done:
                        task_info.task_metadata.status = TaskStatus.COMPLETED
                    await self._save_task_data(task_info)
                    if task_done:
                        tasks_to_remove.add(task_uuid)
                for task_uuid in tasks_to_remove:
                    if task_uuid in self._running_tasks:
                        del self._running_tasks[task_uuid]
                    else:
                        logger.warning(f"Task {task_uuid} not found in running tasks.")

        except asyncio.CancelledError:
            logger.info("Periodic task was cancelled.")  # Handle task cancellation
        except Exception as e:
            logger.exception(e)
        finally:
            logger.info("Periodic task cleanup.")  # Perform cleanup when the task is stopped
