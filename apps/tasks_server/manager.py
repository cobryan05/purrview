"""Manager models on Tasks Server"""

import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional, Union

import db_client
import db_client.models as db_models
import inference_client
from db_client.api.tasks_api import TasksApi

from apps.helpers.consts import TaskStatus
from apps.tasks_server.types import TaskInfo

from .tasks.Registry import task_registry
from .tasks.Task import Task

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Manager:
    """Manages running tasks and coordinates with db_server and inference_server."""

    def __init__(
        self,
        db_api_client: db_client.ApiClient,
        inference_api_client: inference_client.ApiClient,
    ):
        self._task: Optional[asyncio.Task] = None
        self._inference_api_client: inference_client.ApiClient = inference_api_client
        self._db_api_client: db_client.ApiClient = db_api_client
        self._running_tasks: Dict[str, TaskInfo] = {}
        self._completion_tasks: set[asyncio.Task] = set()
        self._pending_oneshot_deletes: set[str] = set()
        self._config = {
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
        return list(task_registry.keys())

    async def get_tasks_result(self, task_ids: Union[List[str], str]) -> dict[str, dict]:
        if isinstance(task_ids, str):
            task_ids = [task_ids]

        tasks_to_delete = []
        results = {}
        not_running = []

        for tid in task_ids:
            task_info = self._running_tasks.get(tid)
            if task_info:
                result = task_info.task.get_results()
                if result:
                    results[tid] = result
                    if task_info.config.params_json.get(Task.InternalKeys.ONESHOT_RESULT, False):
                        tasks_to_delete.append(tid)
            else:
                not_running.append(tid)

        if not_running:
            resp = await asyncio.to_thread(
                TasksApi(self._db_api_client).get_task_results,
                db_models.GetTaskResultsPayload(task_uuids=not_running),
            )
            results.update(resp.results or {})
            for tid in not_running:
                if tid in results and tid in self._pending_oneshot_deletes:
                    self._pending_oneshot_deletes.discard(tid)
                    tasks_to_delete.append(tid)

        if results and tasks_to_delete:
            asyncio.create_task(self.delete_tasks(task_uuids=tasks_to_delete))

        return results

    async def get_tasks_instance_info(
        self, task_uuids: Union[List[str], str, None] = None
    ) -> Dict[str, TaskInfo]:
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
            instances_resp = await asyncio.to_thread(
                TasksApi(self._db_api_client).list_task_instances,
                uuids=uuids_left,
            )
            db_instances = instances_resp.instances or []
            config_uuids = list(set(inst.config_uuid for inst in db_instances))
            configs_resp = await asyncio.to_thread(
                TasksApi(self._db_api_client).list_task_configs,
                uuids=config_uuids,
            )
            configs = configs_resp.configs or {}
            for instance in db_instances:
                config = configs.get(instance.config_uuid)
                running_task = self._running_tasks.get(instance.uuid)
                task_info = TaskInfo(config=config, instance=instance, task=running_task.task if running_task else None)
                ret_info[instance.uuid] = task_info

        return ret_info

    async def create_new_task_config(
        self, name: str, typename: str, params: Dict[str, Any], persistent: bool
    ) -> db_models.TaskConfigRead:
        if typename not in task_registry:
            raise ValueError(f"Unknown task: {typename}")

        params[Task.InternalKeys.PERSISTENT] = persistent
        resp = await asyncio.to_thread(
            TasksApi(self._db_api_client).create_task_config,
            db_models.CreateTaskConfigPayload(name=name, typename=typename, params=params),
        )
        return resp.config

    async def delete_task_configs(self, config_uuids: Union[List[str], str]) -> None:
        if isinstance(config_uuids, str):
            config_uuids = [config_uuids]
        await asyncio.to_thread(
            TasksApi(self._db_api_client).delete_task_configs,
            db_models.DeleteTaskConfigsPayload(config_uuids=config_uuids),
        )

    async def delete_tasks(self, task_uuids: Union[List[str], str]) -> List[str]:
        if isinstance(task_uuids, str):
            task_uuids = [task_uuids]

        deleted_uuids = []
        for task_id in task_uuids:
            if task_id in self._running_tasks:
                await self._running_tasks[task_id].task.stop()
                del self._running_tasks[task_id]
                deleted_uuids.append(task_id)
        await asyncio.to_thread(
            TasksApi(self._db_api_client).delete_task_instances,
            db_models.DeleteTaskInstancesPayload(task_uuids=task_uuids),
        )
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
        if typename is not None and typename not in task_registry:
            raise ValueError(f"Unknown task type: {typename}")

        await asyncio.to_thread(
            TasksApi(self._db_api_client).update_task_config,
            config_uuid,
            db_models.UpdateTaskConfigPayload(
                name=name,
                typename=typename,
                params=params,
                description=description,
                marked_for_delete=marked_for_delete,
            ),
        )

    async def get_task_configs(
        self, task_config_uuids: Union[List[str], str, None]
    ) -> dict[str, db_models.TaskConfigRead]:
        if isinstance(task_config_uuids, str):
            task_config_uuids = [task_config_uuids]

        resp = await asyncio.to_thread(
            TasksApi(self._db_api_client).list_task_configs,
            uuids=task_config_uuids,
        )
        return resp.configs or {}

    async def pause_tasks(self, task_uuids: Union[List[str], str]) -> List[str]:
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
                if self._running_tasks.pop(task_info.instance.uuid, None) is not None:
                    await self._save_task_data(task_info)
            except Exception as e:
                logger.exception(f"Error while saving task data: {e}")
            paused_tasks.append(task_info.instance.uuid)

        return paused_tasks

    async def resume_tasks(self, task_uuids: Union[List[str], str]) -> List[str]:
        if isinstance(task_uuids, str):
            task_uuids = [task_uuids]

        uuids_to_check = []
        for tid in task_uuids:
            if tid in self._running_tasks:
                logger.warning(f"Can't resume task {tid}: already in running tasks")
            else:
                uuids_to_check.append(tid)

        if not uuids_to_check:
            return []

        instances_resp = await asyncio.to_thread(
            TasksApi(self._db_api_client).list_task_instances,
            uuids=uuids_to_check,
        )
        instances = instances_resp.instances or []

        found_uuids = {inst.uuid for inst in instances}
        for tid in uuids_to_check:
            if tid not in found_uuids:
                logger.warning(f"Can't resume task {tid}: status=not found")

        paused = [inst for inst in instances if inst.status == TaskStatus.PAUSED]
        for inst in instances:
            if inst.status != TaskStatus.PAUSED:
                logger.warning(f"Can't resume task {inst.uuid}: status={inst.status}")

        if not paused:
            return []

        config_uuids = list(set(inst.config_uuid for inst in paused))
        configs_resp = await asyncio.to_thread(
            TasksApi(self._db_api_client).list_task_configs,
            uuids=config_uuids,
        )
        configs = configs_resp.configs or {}

        resumed_tasks = []
        for instance in paused:
            config = configs.get(instance.config_uuid)
            if config:
                await self._start_task(config, resume_instance=instance)
                resumed_tasks.append(instance.uuid)

        return resumed_tasks

    async def start_new_task(self, task_config_uuid: str) -> Optional[TaskInfo]:
        configs = await self.get_task_configs([task_config_uuid])
        config = configs.get(task_config_uuid)
        if config:
            return await self._start_task(config)
        else:
            logger.warning(f"Unknown task configuration: {task_config_uuid}")

    async def cancel_tasks(self, task_uuids: Union[List[str], str]) -> List[str]:
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
                await asyncio.to_thread(
                    TasksApi(self._db_api_client).set_task_instance_status,
                    task_info.instance.uuid,
                    db_models.SetTaskStatusPayload(status=task_info.task.get_status()),
                )
                stopped_tasks.append(task_info.instance.uuid)
            except TimeoutError:
                logger.warning(f"Task {task_info.instance.uuid} did not stop in time")

        return stopped_tasks

    async def get_tasks_type_schema(self, typenames: Union[List[str], str]) -> dict[str, dict]:
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
        # Use pop to atomically claim save responsibility; pause_tasks uses the same pattern.
        # Whichever coroutine wins the pop is responsible for persisting final status.
        if self._running_tasks.pop(task_info.instance.uuid, None) is None:
            return
        task_info.instance.status = task_info.task.get_status()
        await self._save_task_data(task_info)
        if task_info.config.params_json.get(Task.InternalKeys.ONESHOT_RESULT, False):
            self._pending_oneshot_deletes.add(task_info.instance.uuid)

    async def _save_task_data(self, task_info: TaskInfo) -> None:
        if task_info.task.is_data_ready():
            await asyncio.to_thread(
                TasksApi(self._db_api_client).set_task_instance_resume_data,
                task_info.instance.uuid,
                db_models.SetTaskResumeDataPayload(resume_data=task_info.task.get_resume_data()),
            )
            task_info.task.clear_data_ready()
        await asyncio.gather(
            asyncio.to_thread(
                TasksApi(self._db_api_client).set_task_instance_result,
                task_info.instance.uuid,
                db_models.SetTaskResultPayload(result=task_info.task.get_results() or {}),
            ),
            asyncio.to_thread(
                TasksApi(self._db_api_client).set_task_instance_status,
                task_info.instance.uuid,
                db_models.SetTaskStatusPayload(status=task_info.instance.status),
            ),
        )

    async def _start_task(
        self,
        config: db_models.TaskConfigRead,
        resume_instance: Optional[db_models.TaskInstanceRead] = None,
    ) -> Optional[TaskInfo]:
        task = task_registry[config.typename](
            task_config_uuid=config.uuid, params=config.params_json, manager=self
        )
        if resume_instance:
            instance = resume_instance
            task.set_resume_data(resume_instance.resume_data or {})
        else:
            resp = await asyncio.to_thread(
                TasksApi(self._db_api_client).create_task_instance,
                db_models.CreateTaskInstancePayload(config_uuid=config.uuid),
            )
            instance = resp.instance
        task_info: TaskInfo = TaskInfo(task=task, config=config, instance=instance)

        self._running_tasks[instance.uuid] = task_info
        await task.start()
        instance.status = TaskStatus.RUNNING
        t = asyncio.create_task(self._on_task_complete(task_info))
        self._completion_tasks.add(t)
        t.add_done_callback(self._completion_tasks.discard)
        return task_info

    def start(self):
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._worker_task())

    async def _init(self):
        resp = await asyncio.to_thread(
            TasksApi(self._db_api_client).list_task_instances,
            resumable=False,
        )
        stale_tasks = resp.instances or []
        if stale_tasks:
            stale_ids = [inst.uuid for inst in stale_tasks]
            await asyncio.to_thread(
                TasksApi(self._db_api_client).delete_task_instances,
                db_models.DeleteTaskInstancesPayload(task_uuids=stale_ids),
            )

    async def _worker_task(self):
        try:
            await self._init()
        except asyncio.CancelledError:
            logger.info("Periodic task was cancelled.")
        except Exception as e:
            logger.exception(e)
        finally:
            logger.info("Periodic task cleanup.")
