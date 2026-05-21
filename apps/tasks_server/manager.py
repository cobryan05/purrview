"""Manager models on Tasks Server"""

import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional, Union

import db_client
import inference_client

from apps.helpers.consts import TaskStatus
from apps.helpers.db.db_client import DbClient
from apps.helpers.db.types import TaskConfig, TaskInstance
from apps.tasks_server.types import TaskInfo

from .tasks.Registry import task_registry
from .tasks.Task import Task

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Manager:
    """Manages inference models, including loading, pinning, and unpinning."""

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
        self._completion_tasks: set[asyncio.Task] = set()
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

    async def get_tasks_result(self, task_ids: Union[List[str], str]) -> dict[str, dict]:
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
                    delete_on_success = task_info.config.params_json.get(Task.InternalKeys.ONESHOT_RESULT, False)
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
            db_instances = await self._legacy_db_client.get_tasks(task_uuid=uuids_left)
            config_uuids = list(set(inst.config_uuid for inst in db_instances))
            configs = await self._legacy_db_client.get_task_configs(config_uuids=config_uuids)
            for instance in db_instances:
                config = configs[instance.config_uuid]
                running_task = self._running_tasks.get(instance.uuid)
                task_info = TaskInfo(config=config, instance=instance, task=running_task.task if running_task else None)
                ret_info[instance.uuid] = task_info

        return ret_info

    async def create_new_task_config(
        self, name: str, typename: str, params: Dict[str, Any], persistent: bool
    ) -> TaskConfig:
        """
        Start a new task.

        Returns new task config
        """
        if typename not in task_registry:
            raise ValueError(f"Unknown task: {typename}")

        params[Task.InternalKeys.PERSISTENT] = persistent
        return await self._legacy_db_client.add_task_config(name=name, typename=typename, params=params)

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
    ) -> dict[str, TaskConfig]:
        """
        Get task configurations by their IDs.
        """
        if isinstance(task_config_uuids, str):
            task_config_uuids = [task_config_uuids]

        return await self._legacy_db_client.get_task_configs(config_uuids=task_config_uuids)

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
            and self._running_tasks[task_id].instance is not None
            and self._running_tasks[task_id].instance.status == TaskStatus.RUNNING
        ]

        for task_info in task_infos:
            task_info.task.request_data_update()
            task_info.task.cancel_task()

        paused_tasks = []
        for task_info in task_infos:
            try:
                await task_info.task.wait_for_task_done(timeout=10.0)
                task_info.instance.status = TaskStatus.PAUSED
            except asyncio.TimeoutError as e:
                logger.warning(f"Timeout while waiting task pause: {e}")
                task_info.instance.status = TaskStatus.ERROR
            except StopIteration as e:
                task_info.instance.status = TaskStatus.COMPLETED
                logger.info(f"Task finished while waiting for data ready: {e}")

            try:
                del self._running_tasks[task_info.instance.uuid]
                await self._save_task_data(task_info)
            except Exception as e:
                logger.exception(f"Error while saving task data: {e}")
            paused_tasks.append(task_info.instance.uuid)

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
                instances = await self._legacy_db_client.get_tasks(task_uuid=task_uuid)
                instance = instances[0] if instances else None
                if instance and instance.status == TaskStatus.PAUSED:
                    config = (await self._legacy_db_client.get_task_configs(config_uuids=[instance.config_uuid])).get(instance.config_uuid)
                    if config:
                        await self._start_task(config, resume_instance=instance)
                else:
                    logger.warning(f"Can't resume task {task_uuid}: status={instance.status if instance else 'not found'}")

        return resumed_tasks

    async def start_new_task(self, task_config_uuid: str) -> Optional[TaskInfo]:
        """
        Start a configured task by its config uuid

        Args:
            task_config_uuid (str): The UUID of the task configuration to start.
        """
        configs = await self.get_task_configs([task_config_uuid])
        config = configs.get(task_config_uuid)
        if config:
            return await self._start_task(config)
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
                await self._legacy_db_client.set_task_status(task_info.instance.uuid, task_info.task.get_status())
                stopped_tasks.append(task_info.instance.uuid)
            except TimeoutError:
                logger.warning(f"Task {task_info.instance.uuid} did not stop in time")

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

    async def _on_task_complete(self, task_info: TaskInfo) -> None:
        if task_info.task is None:
            return
        await task_info.task.wait_for_task_done()
        task_info.instance.status = task_info.task.get_status()
        await self._save_task_data(task_info)
        self._running_tasks.pop(task_info.instance.uuid, None)

    async def _save_task_data(self, task_info: TaskInfo) -> None:
        """Save the resume data for a task."""
        if task_info.task.is_data_ready():
            await self._legacy_db_client.set_task_resume_data(
                task_info.instance.uuid, task_info.task.get_resume_data()
            )
            task_info.task.clear_data_ready()
        # TODO: Sync metadata and DB?
        await self._legacy_db_client.set_task_result(task_info.instance.uuid, task_info.task.get_results())
        await self._legacy_db_client.set_task_status(task_info.instance.uuid, task_info.instance.status)

    async def _start_task(self, config: TaskConfig, resume_instance: Optional[TaskInstance] = None) -> Optional[TaskInfo]:
        """Start a task."""
        task = task_registry[config.typename](
            task_config_uuid=config.uuid, params=config.params_json, manager=self
        )
        instance: TaskInstance = resume_instance or await self._legacy_db_client.insert_new_active_task(config.uuid)
        task_info: TaskInfo = TaskInfo(task=task, config=config, instance=instance)

        self._running_tasks[instance.uuid] = task_info
        await task.start()
        instance.status = TaskStatus.RUNNING
        t = asyncio.create_task(self._on_task_complete(task_info))
        self._completion_tasks.add(t)
        t.add_done_callback(self._completion_tasks.discard)
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
        stale_tasks: List[TaskInstance] = await self._legacy_db_client.get_tasks(resumable=False)
        if stale_tasks:
            stale_ids = [inst.uuid for inst in stale_tasks]
            await self._legacy_db_client.delete_active_tasks(stale_ids)

    async def _worker_task(self):
        try:
            await self._init()
        except asyncio.CancelledError:
            logger.info("Periodic task was cancelled.")
        except Exception as e:
            logger.exception(e)
        finally:
            logger.info("Periodic task cleanup.")
