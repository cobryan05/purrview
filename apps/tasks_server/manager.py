"""Manager models on Tasks Server"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

import db_client
import db_client.models as db_models
import inference_client
from db_client.api.tasks_api import TasksApi

from apps import TASK_RETENTION_DAYS, TASK_RETENTION_MAX_COMPLETED
from apps.helpers.consts import TaskStatus
from apps.tasks_server.types import TaskInfo

from .tasks.Registry import task_registry
from .tasks.Task import Task

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class Manager:
    """Manages running tasks and coordinates with db_server and inference_server."""

    # Crash-loop guard for PERSISTENT auto-restart (see _auto_restart_with_guard):
    # stop auto-restarting an instance once it has crashed and been restarted this
    # many times inside a rolling window. The window resets (count -> 1) the next
    # time a crash is seen after the window has elapsed since it started — this is
    # also how a task that crashes only occasionally (e.g. once a month) decays back
    # to a clean slate instead of being permanently barred.
    _CRASH_LOOP_MAX_RESTARTS: int = 5
    _CRASH_LOOP_WINDOW: timedelta = timedelta(hours=1)

    # Retention bounds for finished instances (see apps/__init__.py). Read from the
    # module-level config at class definition rather than per-call so tests can
    # patch.object these directly, the same way they do the crash-loop knobs.
    _RETENTION_DAYS: int = TASK_RETENTION_DAYS
    _RETENTION_MAX_COMPLETED: int = TASK_RETENTION_MAX_COMPLETED
    # Statuses a task never leaves on its own — only these are prunable.
    _TERMINAL_STATUSES = (TaskStatus.COMPLETED, TaskStatus.ERROR)

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
        # monotonic time.monotonic() of the last time we asked each running task
        # (keyed by instance uuid) to checkpoint. See _checkpoint_running_tasks.
        self._last_checkpoint_request: Dict[str, float] = {}
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
                    if task_info.config.params.get(Task.InternalKeys.ONESHOT_RESULT, False):
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

    async def get_tasks_instance_info(self, task_uuids: Union[List[str], str, None] = None) -> Dict[str, TaskInfo]:
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
                if instance.uuid in ret_info:
                    # Already populated from _running_tasks; the in-memory entry has
                    # the correct live status ("running") whereas the DB record still
                    # has the initial status ("new") until the task completes or pauses.
                    continue
                config = configs.get(instance.config_uuid)
                running_task = self._running_tasks.get(instance.uuid)
                task_info = TaskInfo(
                    config=config,
                    instance=instance,
                    task=running_task.task if running_task else None,
                )
                ret_info[instance.uuid] = task_info

        return ret_info

    async def create_new_task_config(
        self, name: str, typename: str, params: Dict[str, Any], persistent: bool, is_temporary: bool = False
    ) -> db_models.TaskConfigRead:
        if typename not in task_registry:
            raise ValueError(f"Unknown task: {typename}")

        params[Task.InternalKeys.PERSISTENT] = persistent
        resp = await asyncio.to_thread(
            TasksApi(self._db_api_client).create_task_config,
            db_models.CreateTaskConfigPayload(name=name, typename=typename, params=params, is_temporary=is_temporary),
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
        self, task_config_uuids: Union[List[str], str, None], is_temporary: Optional[bool] = None
    ) -> dict[str, db_models.TaskConfigRead]:
        if isinstance(task_config_uuids, str):
            task_config_uuids = [task_config_uuids]

        resp = await asyncio.to_thread(
            TasksApi(self._db_api_client).list_task_configs,
            uuids=task_config_uuids,
            is_temporary=is_temporary,
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
                task_info.task.set_error_message("Timed out waiting for the task to reach a suspendable point.")
            except StopIteration as e:
                task_info.instance.status = TaskStatus.COMPLETED
                logger.info(f"Task finished while waiting for data ready: {e}")

            try:
                if self._running_tasks.pop(task_info.instance.uuid, None) is not None:
                    # Suspended tasks are resumable; don't write ended_at.
                    await self._save_task_data(task_info, ended_at=None)
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

    async def pause_tasks_in_place(self, task_uuids: Union[List[str], str]) -> List[str]:
        """Pause tasks cooperatively in-place. Task stays in _running_tasks; DB status unchanged."""
        if isinstance(task_uuids, str):
            task_uuids = [task_uuids]
        paused = []
        for tid in task_uuids:
            task_info = self._running_tasks.get(tid)
            if task_info and task_info.task and task_info.instance.status == TaskStatus.RUNNING:
                await task_info.task.pause()
                paused.append(tid)
            else:
                logger.warning(f"Can't pause in place {tid}: not a running task")
        return paused

    async def resume_tasks_in_place(self, task_uuids: Union[List[str], str]) -> List[str]:
        """Resume tasks that were paused in-place."""
        if isinstance(task_uuids, str):
            task_uuids = [task_uuids]
        resumed = []
        for tid in task_uuids:
            task_info = self._running_tasks.get(tid)
            if task_info and task_info.task:
                await task_info.task.resume()
                resumed.append(tid)
            else:
                logger.warning(f"Can't resume in place {tid}: not in running tasks")
        return resumed

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
            except (asyncio.TimeoutError, TimeoutError):
                # On Python 3.10 asyncio.TimeoutError (what wait_for_task_done
                # raises) is a distinct class from the builtin TimeoutError; catch
                # both so one slow task doesn't abort the whole cancel loop and
                # leave the remaining tasks' statuses unpersisted.
                logger.warning(f"Task {task_info.instance.uuid} did not stop in time")

        return stopped_tasks

    async def update_task_heartbeat(self, task_uuid: str) -> None:
        """Forward a heartbeat ping to the DB server for the given task instance."""
        await asyncio.to_thread(
            TasksApi(self._db_api_client).set_task_instance_heartbeat,
            task_uuid,
        )

    async def get_tasks_type_schema(self, typenames: Union[List[str], str]) -> dict[str, dict]:
        if isinstance(typenames, str):
            typenames = [typenames]

        schemas = {}
        for typename in typenames:
            if typename not in task_registry:
                raise ValueError(f"Unknown task: {typename}")
            schemas[typename] = task_registry[typename].params_schema()
        return schemas

    async def get_tasks_status_schema(self, typenames: Union[List[str], str]) -> dict[str, list[dict]]:
        if isinstance(typenames, str):
            typenames = [typenames]

        schemas = {}
        for typename in typenames:
            if typename not in task_registry:
                raise ValueError(f"Unknown task: {typename}")
            schemas[typename] = task_registry[typename].status_schema()
        return schemas

    async def get_tasks_status_data(self, task_uuids: Union[List[str], str]) -> dict[str, dict]:
        """Return status_data for the given task instance uuids.

        Live values win for a running task; for one that has finished (or was
        suspended) the last snapshot persisted at save time is served instead, so a
        completed card still shows the histograms and stats the run produced.

        Uuids with neither are simply omitted, not an error — the task may have been
        deleted, or never existed.
        """
        if isinstance(task_uuids, str):
            task_uuids = [task_uuids]

        status_data = {}
        not_running = []
        for uuid in task_uuids:
            task_info = self._running_tasks.get(uuid)
            if task_info is None or task_info.task is None:
                not_running.append(uuid)
                continue
            status_data[uuid] = task_info.task.get_status_data()

        if not_running:
            try:
                resp = await asyncio.to_thread(
                    TasksApi(self._db_api_client).list_task_instances,
                    uuids=not_running,
                )
                for inst in resp.instances or []:
                    if inst.status_data:
                        status_data[inst.uuid] = inst.status_data
            except Exception:
                # A finished card losing its widgets is not worth failing the request
                # for — the rest of the card (status, logs, error) still renders.
                logger.exception("Failed to load persisted status data for %s", not_running)
        return status_data

    async def send_task_command(self, task_uuid: str, command: str, params: dict[str, Any]) -> None:
        """Forward a UI-issued command to the running task instance's handle_command().

        Raises ValueError if the task isn't currently running; propagates any
        exception raised by handle_command() (e.g. NotImplementedError) to the caller.
        """
        task_info = self._running_tasks.get(task_uuid)
        if task_info is None or task_info.task is None:
            raise ValueError(f"Task {task_uuid} is not currently running")
        await task_info.task.handle_command(command, params)

    async def _on_task_complete(self, task_info: TaskInfo) -> None:
        if task_info.task is None:
            return
        await task_info.task.wait_for_task_done()
        # Use pop to atomically claim save responsibility; pause_tasks uses the same pattern.
        # Whichever coroutine wins the pop is responsible for persisting final status.
        if self._running_tasks.pop(task_info.instance.uuid, None) is None:
            return
        try:
            task_info.instance.status = task_info.task.get_status()
            await self._save_task_data(task_info, ended_at=datetime.now(timezone.utc))
            if task_info.config.params.get(Task.InternalKeys.ONESHOT_RESULT, False):
                self._pending_oneshot_deletes.add(task_info.instance.uuid)
        except Exception as e:
            logger.exception(f"Error completing task {task_info.instance.uuid}; forcing ERROR status")
            task_info.instance.status = TaskStatus.ERROR
            # Don't clobber a message the task set itself — the original failure
            # explains more than the bookkeeping error that followed it.
            if task_info.task.get_error_message() is None:
                task_info.task.set_error_message(f"Failed while finalising the task: {type(e).__name__}: {e}")
            try:
                await self._save_task_data(task_info, ended_at=datetime.now(timezone.utc))
            except Exception:
                logger.exception(f"Failed to save ERROR status for task {task_info.instance.uuid}")

    async def _save_task_data(self, task_info: TaskInfo, ended_at: Optional[datetime] = None) -> None:
        if task_info.task.is_data_ready():
            await asyncio.to_thread(
                TasksApi(self._db_api_client).set_task_instance_resume_data,
                task_info.instance.uuid,
                db_models.SetTaskResumeDataPayload(resume_data=task_info.task.get_resume_data()),
            )
            task_info.task.clear_data_ready()
        # Keep the in-memory copy in step with what we're about to persist: polls
        # answered from _running_tasks would otherwise show "error" with no reason
        # for the moment between the task failing and it being popped.
        task_info.instance.error_message = task_info.task.get_error_message()
        # Status and result are critical — a failure here must propagate so the
        # caller's retry logic can attempt recovery.
        await asyncio.gather(
            asyncio.to_thread(
                TasksApi(self._db_api_client).set_task_instance_result,
                task_info.instance.uuid,
                db_models.SetTaskResultPayload(result=task_info.task.get_results() or {}),
            ),
            asyncio.to_thread(
                TasksApi(self._db_api_client).set_task_instance_status,
                task_info.instance.uuid,
                db_models.SetTaskStatusPayload(
                    status=task_info.instance.status,
                    error_message=task_info.task.get_error_message(),
                ),
            ),
        )
        # Timestamps and logs are best-effort: a failure must not block status/result
        # from being written, and must not prevent the task from being marked done.
        # ended_at is None for suspended tasks (they can be resumed) and set by
        # callers that know the task has truly terminated.
        # The status widgets' data lives in the task object, which is discarded the
        # moment the run ends — so snapshot it here, or a finished card shows empty
        # histograms and stat grids. get_status_data() is contractually cheap and
        # non-blocking, but a buggy one must not cost us the logs, hence its own try.
        status_data = None
        try:
            status_data = task_info.task.get_status_data()
        except Exception:
            logger.exception(f"get_status_data() failed for task {task_info.instance.uuid} (non-fatal)")
        try:
            await asyncio.to_thread(
                TasksApi(self._db_api_client).set_task_instance_timestamps,
                task_info.instance.uuid,
                db_models.SetTaskTimestampsPayload(
                    ended_at=ended_at,
                    log_json=list(task_info.task.get_log_lines()),
                    status_data_json=status_data,
                ),
            )
        except Exception:
            logger.exception(f"Failed to persist timestamps/logs for task {task_info.instance.uuid} (non-fatal)")

    async def _start_task(
        self,
        config: db_models.TaskConfigRead,
        resume_instance: Optional[db_models.TaskInstanceRead] = None,
    ) -> Optional[TaskInfo]:
        # Inject the config display name so tasks can use it for generated artifact names
        # without needing a separate API call.  Use a copy to avoid mutating the config.
        task_params = {**config.params, "_config_name": config.name}
        task_cls = task_registry.get(config.typename)
        if task_cls is None:
            # A bare task_registry[...] raises KeyError("'OldName'") — useless
            # when a config was saved under a since-renamed/removed task type.
            raise ValueError(
                f"Unknown task type {config.typename!r} for config {config.name!r} "
                f"(registered: {sorted(task_registry)})"
            )
        task = task_cls(task_config_uuid=config.uuid, params=task_params, manager=self)
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

        task.set_instance_uuid(instance.uuid)
        self._running_tasks[instance.uuid] = task_info
        await task.start()
        instance.status = TaskStatus.RUNNING
        started_at = datetime.now(timezone.utc)
        # Register the completion handler before the DB write so that a DB
        # failure cannot leave the task running with no completion handler.
        t = asyncio.create_task(self._on_task_complete(task_info))
        self._completion_tasks.add(t)
        t.add_done_callback(self._completion_tasks.discard)
        # Persist the running status and start timestamp immediately.
        await asyncio.gather(
            asyncio.to_thread(
                TasksApi(self._db_api_client).set_task_instance_status,
                instance.uuid,
                db_models.SetTaskStatusPayload(status=TaskStatus.RUNNING),
            ),
            asyncio.to_thread(
                TasksApi(self._db_api_client).set_task_instance_timestamps,
                instance.uuid,
                db_models.SetTaskTimestampsPayload(started_at=started_at),
            ),
        )
        return task_info

    def start(self):
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._worker_task())

    async def _init(self):
        resp = await asyncio.to_thread(
            TasksApi(self._db_api_client).list_task_instances,
        )
        all_instances = resp.instances or []

        # "new" = created but server crashed before task.start() was reached.
        # Nothing useful to recover; remove them.
        new_tasks = [inst for inst in all_instances if inst.status == TaskStatus.NEW]
        if new_tasks:
            logger.info(f"Removing {len(new_tasks)} never-started task instance(s)")
            await asyncio.to_thread(
                TasksApi(self._db_api_client).delete_task_instances,
                db_models.DeleteTaskInstancesPayload(task_uuids=[inst.uuid for inst in new_tasks]),
            )

        # "running" = server was killed while the task was executing.
        # If the task managed to save a checkpoint, treat it as paused so the user
        # can resume it.  If there is no checkpoint there is nothing to resume from,
        # so mark it as an error so it's clearly visible but not misleadingly resumable.
        running_tasks = [inst for inst in all_instances if inst.status == TaskStatus.RUNNING]
        resumable_instances: List[db_models.TaskInstanceRead] = []
        if running_tasks:
            coros = []
            for inst in running_tasks:
                recovery_message = None
                if inst.resume_data:
                    recovered_status = db_models.TaskStatus.PAUSED
                    logger.info(f"Task {inst.uuid} interrupted with checkpoint — marking paused (resumable)")
                    resumable_instances.append(inst)
                else:
                    recovered_status = db_models.TaskStatus.ERROR
                    recovery_message = (
                        "The server stopped while this task was running, and it had no checkpoint to resume from."
                    )
                    logger.info(f"Task {inst.uuid} interrupted without checkpoint — marking error")
                coros.append(
                    asyncio.to_thread(
                        TasksApi(self._db_api_client).set_task_instance_status,
                        inst.uuid,
                        db_models.SetTaskStatusPayload(status=recovered_status, error_message=recovery_message),
                    )
                )
            await asyncio.gather(*coros)

        # Auto-restart PERSISTENT configs whose interrupted instance recovered with a
        # checkpoint (resumable). This is the only path that restarts a task without
        # explicit user action, so it stays strictly opt-in per config (a crashed
        # TrainModelTask, or any config that never set _persistent, is left PAUSED for
        # the user to resume manually) and is subject to the crash-loop guard.
        if resumable_instances:
            config_uuids = list({inst.config_uuid for inst in resumable_instances})
            configs_resp = await asyncio.to_thread(
                TasksApi(self._db_api_client).list_task_configs,
                uuids=config_uuids,
            )
            configs = configs_resp.configs or {}
            for inst in resumable_instances:
                config = configs.get(inst.config_uuid)
                if config is None:
                    continue
                if not config.params.get(Task.InternalKeys.PERSISTENT, False):
                    continue
                try:
                    await self._auto_restart_with_guard(config, inst)
                except Exception:
                    logger.exception(f"Failed to auto-restart persistent task instance {inst.uuid}")

    async def _check_stale_tasks(self) -> None:
        """Mark running tasks as error when their heartbeat has gone stale.

        A task's heartbeat goes stale if the asyncio Task died without updating
        its DB status (e.g. the _on_task_complete DB write failed). Tasks that
        are still alive in _running_tasks are excluded from the check.
        Threshold = 4 × HEARTBEAT_INTERVAL so a single missed beat doesn't
        trigger a false positive.
        """
        stale_threshold = timedelta(seconds=Task.HEARTBEAT_INTERVAL * 4)
        now = datetime.now(timezone.utc)
        try:
            resp = await asyncio.to_thread(TasksApi(self._db_api_client).list_task_instances)
            all_instances = resp.instances or []
            stale = []
            for inst in all_instances:
                if inst.status != TaskStatus.RUNNING:
                    continue
                if inst.uuid in self._running_tasks:
                    continue  # alive in this process
                if inst.heartbeat_at is None:
                    continue  # no heartbeat yet — started before heartbeat support
                hb = inst.heartbeat_at
                if hb.tzinfo is None:
                    hb = hb.replace(tzinfo=timezone.utc)
                if (now - hb) > stale_threshold:
                    stale.append(inst)
            if stale:
                coros = []
                for inst in stale:
                    logger.warning("Task %s has stale heartbeat — marking error", inst.uuid)
                    coros.append(
                        asyncio.to_thread(
                            TasksApi(self._db_api_client).set_task_instance_status,
                            inst.uuid,
                            db_models.SetTaskStatusPayload(
                                status=TaskStatus.ERROR,
                                error_message=(
                                    "No heartbeat from this task for over "
                                    f"{stale_threshold}; assuming it died without reporting an error."
                                ),
                            ),
                        )
                    )
                await asyncio.gather(*coros)
        except Exception:
            logger.exception("Error checking stale tasks")

    async def _checkpoint_running_tasks(self) -> None:
        """Periodic checkpoint pass, run from the 60s worker loop.

        Handshake with each task (mirrors Task's own docstring on _data_req_flag /
        _data_ready_flag): request and collect are deliberately two separate,
        non-blocking steps run every pass, so one slow task can never stall this
        loop or delay checkpointing of any other task.

          1. Collect: for every running task, check is_data_ready(). A task notices
             _data_req_flag at its own next convenient point (ideally promptly — see
             the hoisted check at the top of CollectTrainingImagesTask's main loop),
             builds resume data, and sets _data_ready_flag. That may happen on this
             pass or several passes later for a slow task; whenever it does, we pick
             it up here and write it via the NARROW set_task_instance_resume_data
             path only — never _save_task_data, which also writes result/status/
             ended_at and would clobber a task that is still running.
          2. Request: if this task's CHECKPOINT_INTERVAL has elapsed since we last
             asked it to checkpoint, call request_data_update(). This only sets an
             asyncio.Event — it never blocks and never waits for the task to react.

        Because request and collect are decoupled, a task with CHECKPOINT_INTERVAL
        shorter than 60s (this loop's tick) still only actually writes to the DB
        once it responds — it is not force-checkpointed on every tick.
        """
        now = time.monotonic()
        # Prune stale entries for tasks that are no longer running so this dict
        # doesn't grow unboundedly across a long server uptime.
        self._last_checkpoint_request = {
            uuid: t for uuid, t in self._last_checkpoint_request.items() if uuid in self._running_tasks
        }
        for uuid, task_info in list(self._running_tasks.items()):
            task = task_info.task
            if task is None or task_info.instance is None:
                continue
            if task_info.instance.status != TaskStatus.RUNNING:
                continue

            if task.is_data_ready():
                try:
                    await asyncio.to_thread(
                        TasksApi(self._db_api_client).set_task_instance_resume_data,
                        uuid,
                        db_models.SetTaskResumeDataPayload(resume_data=task.get_resume_data()),
                    )
                    task.clear_data_ready()
                except Exception:
                    logger.exception(f"Failed to write periodic checkpoint for task {uuid}")

            interval = getattr(task, "CHECKPOINT_INTERVAL", Task.CHECKPOINT_INTERVAL)
            last_request = self._last_checkpoint_request.get(uuid, 0.0)
            if (now - last_request) >= interval:
                self._last_checkpoint_request[uuid] = now
                task.request_data_update()

    async def _auto_restart_with_guard(
        self, config: db_models.TaskConfigRead, inst: db_models.TaskInstanceRead
    ) -> None:
        """Auto-resume a PERSISTENT task instance recovered (with a checkpoint) after
        an unexpected restart, subject to a crash-loop guard.

        The restart counter + window-start timestamp live on the task CONFIG's params
        (Task.InternalKeys.RESTART_INFO) rather than the instance's resume_data: that
        blob is owned by the task itself and gets wholesale-replaced the next time it
        checkpoints (Task._build_resume_data), which would silently erase this
        bookkeeping. Policy: stop auto-restarting once an instance has been restarted
        more than _CRASH_LOOP_MAX_RESTARTS times inside _CRASH_LOOP_WINDOW; a restart
        seen after the window has elapsed starts a fresh window (count -> 1), which is
        also how the guard decays for a task that only crashes occasionally.
        """
        now = datetime.now(timezone.utc)
        restart_info = dict(config.params.get(Task.InternalKeys.RESTART_INFO) or {})
        window_start: Optional[datetime] = None
        raw_window_start = restart_info.get("window_start")
        if raw_window_start:
            try:
                window_start = datetime.fromisoformat(raw_window_start)
            except ValueError:
                window_start = None

        if window_start is None or (now - window_start) > self._CRASH_LOOP_WINDOW:
            count = 0
            window_start = now
        else:
            count = restart_info.get("count", 0)
        count += 1

        new_params = {
            **config.params,
            Task.InternalKeys.RESTART_INFO: {"count": count, "window_start": window_start.isoformat()},
        }

        if count > self._CRASH_LOOP_MAX_RESTARTS:
            logger.error(
                "Crash-loop guard tripped for task config %r (%s): %d restarts within %s — "
                "NOT auto-restarting instance %s; marking it ERROR",
                config.name,
                config.uuid,
                count,
                self._CRASH_LOOP_WINDOW,
                inst.uuid,
            )
            await asyncio.gather(
                asyncio.to_thread(
                    TasksApi(self._db_api_client).set_task_instance_status,
                    inst.uuid,
                    db_models.SetTaskStatusPayload(
                        status=TaskStatus.ERROR,
                        error_message=(
                            f"Auto-restart stopped: this task restarted {count} times within "
                            f"{self._CRASH_LOOP_WINDOW}, which trips the crash-loop guard."
                        ),
                    ),
                ),
                asyncio.to_thread(
                    TasksApi(self._db_api_client).update_task_config,
                    config.uuid,
                    db_models.UpdateTaskConfigPayload(params=new_params),
                ),
            )
            return

        await asyncio.to_thread(
            TasksApi(self._db_api_client).update_task_config,
            config.uuid,
            db_models.UpdateTaskConfigPayload(params=new_params),
        )
        logger.info(
            "Auto-restarting persistent task instance %s (config %r): restart %d/%d in window",
            inst.uuid,
            config.name,
            count,
            self._CRASH_LOOP_MAX_RESTARTS,
        )
        restarted_config = config.model_copy(update={"params": new_params})
        await self._start_task(restarted_config, resume_instance=inst)

    async def _prune_finished_task_instances(self) -> List[str]:
        """Delete finished task instances past either retention bound.

        Both bounds apply, whichever removes more: instances that ended longer ago
        than _RETENTION_DAYS, plus anything beyond the _RETENTION_MAX_COMPLETED most
        recent. Age alone doesn't bound storage (a persistent config can produce
        instances far faster than they age out) and a count alone gives no sense of
        how far back the history reaches.

        Only COMPLETED/ERROR instances are candidates — a running, pending or
        suspended task is live state, never history, regardless of its age.
        Returns the uuids deleted, for logging and tests.
        """
        if self._RETENTION_DAYS <= 0 and self._RETENTION_MAX_COMPLETED <= 0:
            return []
        try:
            resp = await asyncio.to_thread(TasksApi(self._db_api_client).list_task_instances)
        except Exception:
            logger.exception("Failed to list task instances for retention pruning")
            return []

        finished = [i for i in (resp.instances or []) if i.status in self._TERMINAL_STATUSES]
        if not finished:
            return []

        def _finished_at(inst) -> datetime:
            # ended_at is the real signal; created_at covers rows written before
            # ended_at was persisted. The epoch fallback sorts undated rows oldest,
            # so they are pruned first rather than pinned at the top forever.
            stamp = inst.ended_at or inst.created_at
            if stamp is None:
                return datetime.min.replace(tzinfo=timezone.utc)
            return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)

        finished.sort(key=_finished_at, reverse=True)  # newest first

        victims: set[str] = set()
        if self._RETENTION_MAX_COMPLETED > 0:
            victims.update(i.uuid for i in finished[self._RETENTION_MAX_COMPLETED :])
        if self._RETENTION_DAYS > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self._RETENTION_DAYS)
            victims.update(i.uuid for i in finished if _finished_at(i) < cutoff)

        if not victims:
            return []
        uuids = sorted(victims)
        try:
            await asyncio.to_thread(
                TasksApi(self._db_api_client).delete_task_instances,
                db_models.DeleteTaskInstancesPayload(task_uuids=uuids),
            )
        except Exception:
            logger.exception("Failed to delete %d expired task instances", len(uuids))
            return []
        logger.info(
            "Pruned %d finished task instance(s) (keeping %d days / %d most recent)",
            len(uuids),
            self._RETENTION_DAYS,
            self._RETENTION_MAX_COMPLETED,
        )
        return uuids

    async def _worker_task(self):
        try:
            await self._init()
            while True:
                await asyncio.sleep(60)
                await self._check_stale_tasks()
                await self._checkpoint_running_tasks()
                await self._prune_finished_task_instances()
        except asyncio.CancelledError:
            logger.info("Periodic task was cancelled.")
        except Exception as e:
            logger.exception(e)
        finally:
            logger.info("Periodic task cleanup.")

    def get_status(self) -> dict:
        """Return service health and diagnostic info."""
        notices: list[dict] = []
        for ti in self._running_tasks.values():
            if ti.task:
                notices.extend(ti.task.get_notices())
        return {
            "service": "tasks_server",
            "status": "ok",
            "tasks": {"running": len(self._running_tasks)},
            "notices": notices,
        }
