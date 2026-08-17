"""Manager models on Tasks Server"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

import db_client
import db_client.models as db_models
import httpx
import inference_client
import relay_buffer_client
from db_client.api.tasks_api import TasksApi

from apps import APPS, APPS_CONFIG, TASK_RETENTION_DAYS, TASK_RETENTION_MAX_COMPLETED
from apps.helpers.apiCall import LONG_TIMEOUT, bind_default_timeout
from apps.helpers.consts import TaskStatus
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.recordings.relayClipClient import RelayClipClient
from apps.helpers.serverUrls import read_url_overrides
from apps.helpers.serviceStatus import row, section
from apps.tasks_server.types import TaskInfo

from .tasks.Registry import task_registry
from .tasks.Task import Task

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


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
    # How many missed heartbeats before a task this process is driving is called
    # wedged. Matches the orphan check's threshold on purpose — the two ask the same
    # question of the same signal and differ only in what they conclude (see
    # _check_wedged_tasks) — and is loose enough that an ordinary slow operation
    # between beats does not trip it.
    _WEDGED_HEARTBEAT_INTERVALS: int = 4

    _RETENTION_DAYS: int = TASK_RETENTION_DAYS
    _RETENTION_MAX_COMPLETED: int = TASK_RETENTION_MAX_COMPLETED
    # Statuses a task never leaves on its own — only these are prunable.
    _TERMINAL_STATUSES = (TaskStatus.COMPLETED, TaskStatus.ERROR)

    def __init__(
        self,
        db_api_client: db_client.ApiClient,
        inference_api_client: inference_client.ApiClient,
        default_training_url: Optional[str] = None,
    ):
        self._task: Optional[asyncio.Task] = None
        self._inference_api_client: inference_client.ApiClient = inference_api_client
        self._db_api_client: db_client.ApiClient = db_api_client

        # Settings → Servers can point inference/training at another host (both
        # services are network-isolated by design — see apps/helpers/serverUrls.py).
        # The injected client and APPS_CONFIG supply the local defaults; overrides
        # are resolved from db_server on every worker pass and on demand via
        # reload_url_overrides(), because a task started an hour from now must use
        # whatever URL is current then, not whatever was set at import time.
        if default_training_url is None:
            training_cfg = APPS_CONFIG[APPS.TRAINING_SERVER]
            default_training_url = f"http://{training_cfg.host}:{training_cfg.port}"
        self._default_inference_url: str = inference_api_client.configuration.host
        self._default_training_url: str = default_training_url
        self._inference_url_override: Optional[str] = None
        self._training_url_override: Optional[str] = None
        self._training_base_url: str = default_training_url
        self._url_settings_client = httpx.AsyncClient(timeout=10.0)
        # Built lazily by get_relay_api_client — see there for why it is not injected.
        self._relay_clip_client: Optional[RelayClipClient] = None
        self._running_tasks: Dict[str, TaskInfo] = {}
        self._completion_tasks: set[asyncio.Task] = set()
        self._pending_oneshot_deletes: set[str] = set()
        # monotonic time.monotonic() of the last time we asked each running task
        # (keyed by instance uuid) to checkpoint. See _checkpoint_running_tasks.
        self._last_checkpoint_request: Dict[str, float] = {}
        # Has the startup recovery pass completed? It is retried every worker pass
        # until it does (db_server may not be up yet), and the stale-heartbeat net
        # waits on it — see _run_startup_recovery.
        self._recovery_done: bool = False
        # DB instances marked "running" that this process is not driving. Sampled on
        # the worker pass so the sync get_status() stays an in-memory read.
        self._unattached_count: int = 0
        # instance uuid → seconds since its last heartbeat, for tasks this process IS
        # driving that have stopped progressing. See _check_wedged_tasks: reported,
        # never acted on.
        self._wedged_tasks: Dict[str, float] = {}

    def get_db_api_client(self) -> db_client.ApiClient:
        return self._db_api_client

    def get_inference_api_client(self) -> inference_client.ApiClient:
        """The inference client to use *right now*.

        Tasks must call this at the point of use rather than caching it across a
        long run: a URL change swaps the client out, and a task holding the old
        one would keep talking to the previous host.
        """
        return self._inference_api_client

    def get_relay_api_client(self) -> RelayClipClient:
        """The relay client used to cut event clips out of the rolling buffer.

        Built on first use rather than injected: only a watcher with a `record_clip`
        action on a buffered source ever needs it.

        No Settings → Servers override, unlike inference and training. Those two are
        stateless compute and are the only services the deployment model allows onto
        hosts of their own; the relay ships alongside db_server and
        image_tagging_server, and a clip is cut into a `dest` path on the relay's
        filesystem, which is the same filesystem this process can read. The host
        still comes from APPS_CONFIG, so RELAY_BUFFER_SERVER_HOST moves it within
        that group — but do not add a UI override here expecting a remote relay to
        work: the clip path would be written somewhere nothing else can see.
        """
        if self._relay_clip_client is None:
            relay_cfg = APPS_CONFIG[APPS.RELAY_BUFFER_SERVER]
            # Bound long as a whole: this client exists solely to cut clips, and
            # remuxing a minute of video out of the rolling buffer is real work. The
            # relay's own fast calls (leases, readiness) go through a different client.
            api_client = bind_default_timeout(
                relay_buffer_client.ApiClient(
                    relay_buffer_client.Configuration(f"http://{relay_cfg.host}:{relay_cfg.port}")
                ),
                timeout=LONG_TIMEOUT,
            )
            self._relay_clip_client = RelayClipClient(api_client)
        return self._relay_clip_client

    def get_training_base_url(self) -> str:
        """Base URL of the training server, honouring the Settings override."""
        return self._training_base_url

    async def reload_url_overrides(self) -> None:
        """Re-read the inference/training URL overrides and apply any change.

        Cheap and idempotent — the inference client is only rebuilt when the
        effective URL actually moves, so calling this on every worker pass does
        not churn urllib3 connection pools.
        """
        inference_override, training_override = await read_url_overrides(
            self._url_settings_client, self._db_api_client.configuration.host
        )

        # The Settings input happily accepts a trailing slash; the generated
        # client concatenates host + path, so leaving it on yields "//api/...".
        def _clean(url: Optional[str]) -> Optional[str]:
            return url.rstrip("/") if url else None

        self._inference_url_override = _clean(inference_override)
        self._training_url_override = _clean(training_override)

        effective_inference = self._inference_url_override or self._default_inference_url
        effective_training = self._training_url_override or self._default_training_url

        if effective_inference != self._inference_api_client.configuration.host:
            logger.info(f"Inference server URL -> {effective_inference}")
            previous = self._inference_api_client
            self._inference_api_client = bind_default_timeout(
                inference_client.ApiClient(inference_client.Configuration(effective_inference))
            )
            self._release_api_client(previous)

        if effective_training != self._training_base_url:
            logger.info(f"Training server URL -> {effective_training}")
            self._training_base_url = effective_training

    @staticmethod
    def _release_api_client(client: Any) -> None:
        """Release the connection pool held by a discarded generated ApiClient.

        The generated client has no close(): its urllib3 PoolManager is only
        reachable through rest_client, and nothing finalises it. Dropping the
        reference therefore leaves the sockets to the old host open, so each
        Settings → Servers change strands a pool for the life of the process.

        Tasks are told to fetch the client at the point of use rather than cache
        it (see get_inference_api_client), so by the time a swap happens nothing
        should still be holding this one.
        """
        pool_manager = getattr(getattr(client, "rest_client", None), "pool_manager", None)
        if pool_manager is None:
            return
        try:
            pool_manager.clear()
        except Exception as e:
            logger.warning(f"Could not release the previous API client's connection pool: {e}")

    async def stop(self) -> None:
        """Stop the worker loop and release the clients this manager owns.

        Called from the app lifespan on shutdown. Without it the settings
        httpx client is never closed, which leaks in tests that build many
        managers and logs an unclosed-session warning on teardown.
        """
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Worker task raised during shutdown: {e}")
        await self._url_settings_client.aclose()
        self._release_api_client(self._inference_api_client)

    async def _reload_url_overrides_safe(self) -> None:
        """reload_url_overrides() wrapped for the worker loop, where an exception
        would kill the pass that also does recovery, heartbeats and pruning."""
        try:
            await self.reload_url_overrides()
        except Exception as e:
            logger.warning(f"Could not refresh server URL overrides: {e}")

    async def get_server_config(self) -> Dict[str, Any]:
        return {
            "db_server": self._db_api_client.configuration.host,
            "inference_server": self._inference_api_client.configuration.host,
            "training_server": self._training_base_url,
            "inference_server_url_override": self._inference_url_override,
            "training_server_url_override": self._training_url_override,
        }

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
        self,
        name: str,
        typename: str,
        params: Dict[str, Any],
        persistent: Optional[bool] = None,
        is_temporary: bool = False,
    ) -> db_models.TaskConfigRead:
        if typename not in task_registry:
            raise ValueError(f"Unknown task: {typename}")

        # None means "the caller has no opinion" — ask the task class rather than
        # letting whichever UI launched it decide. Resolved here, the single place
        # every creation path funnels through, so a task type cannot end up
        # persistent from one button and not from another.
        if persistent is None:
            persistent = task_registry[typename].DEFAULT_PERSISTENT
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

        # One of the two paths that actually restarts a task, so it needs the
        # checkpoint — the listing omits resume_data unless asked (see below).
        instances_resp = await asyncio.to_thread(
            TasksApi(self._db_api_client).list_task_instances,
            uuids=uuids_to_check,
            include_resume_data=True,
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

    # How long cancel_tasks lingers before answering "still stopping". Purely a
    # presentation choice: a task that shuts down inside this window lets the UI say
    # "Stopped", and one that doesn't gets an honest "Stopping…" instead of a lie.
    _CANCEL_GRACE_S: float = 5.0

    async def cancel_tasks(self, task_uuids: Union[List[str], str]) -> Dict[str, List[str]]:
        """Ask running tasks to stop, and report per-task what actually happened.

        Returns {"stopped": [...], "stopping": [...], "notRunning": [...]}. The three
        outcomes exist because the caller is a button: "I asked and it stopped", "I
        asked and it is still winding down" and "there was nothing there to ask" are
        different things to show a user, and collapsing them into one list is how a
        Stop click that did nothing came back as success.

        Registration is deliberately NOT dropped here. This method used to pop the
        task out of _running_tasks up front and then wait 5s; a task that overran that
        wait was left orphaned — unreachable by a second Stop click (which then
        reported success while logging "not found in running tasks"), never given a
        final DB status, and eventually mistaken by the stale-heartbeat net for a
        crashed instance and auto-restarted, undoing the user's stop. _on_task_complete
        is already watching every running task and owns the pop-and-persist; letting it
        do that job is what makes a slow stop merely slow instead of unrecoverable.
        """
        if isinstance(task_uuids, str):
            task_uuids = [task_uuids]

        canceled: List[TaskInfo] = []
        not_running: List[str] = []
        for task_uuid in task_uuids:
            task_info = self._running_tasks.get(task_uuid)
            if task_info is None or task_info.task is None:
                logger.warning(f"Can't stop task {task_uuid}: not found in running tasks")
                not_running.append(task_uuid)
                continue
            task_info.task.cancel_task()
            canceled.append(task_info)
            logger.info(f"Task {task_uuid} cancel request sent")

        stopped: List[str] = []
        stopping: List[str] = []
        # Waited on together, not one after another: serial waits made the grace
        # period cumulative, so stopping four tasks at once could block the request
        # for four times the budget.
        results = await asyncio.gather(
            *(self._await_cancelled(info) for info in canceled),
            return_exceptions=True,
        )
        for task_info, done in zip(canceled, results):
            if done is True:
                stopped.append(task_info.instance.uuid)
            else:
                logger.warning(f"Task {task_info.instance.uuid} did not stop within {self._CANCEL_GRACE_S}s")
                stopping.append(task_info.instance.uuid)

        return {"stopped": stopped, "stopping": stopping, "notRunning": not_running}

    async def _await_cancelled(self, task_info: TaskInfo) -> bool:
        """Wait out the grace period for one cancelled task. True if it finished.

        Persisting the final status is _on_task_complete's job, not this one's — two
        writers for one instance is how a status write can land after the row has
        already been finalised.
        """
        try:
            await task_info.task.wait_for_task_done(self._CANCEL_GRACE_S)
            return True
        except (asyncio.TimeoutError, TimeoutError):
            # On Python 3.10 asyncio.TimeoutError (what wait_for_task_done raises) is
            # a distinct class from the builtin TimeoutError; catch both.
            return False

    async def save_task_resume_data(self, task_uuid: str, resume_data: Dict[str, Any]) -> None:
        """Write one task instance's checkpoint straight through to the DB.

        The narrow write only — never _save_task_data, which also writes
        result/status/ended_at and would clobber a task that is still running.
        Used by Task.save_checkpoint_now() for state that must not wait for the
        periodic pass.
        """
        await asyncio.to_thread(
            TasksApi(self._db_api_client).set_task_instance_resume_data,
            task_uuid,
            db_models.SetTaskResumeDataPayload(resume_data=resume_data),
        )

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

    async def get_task_live_preview(
        self, task_uuid: str, source_uuid: Optional[str] = None, **options: Any
    ) -> Optional[dict[str, Any]]:
        """Ask a running task for its current frame and what it sees in it.

        Only ever served from the live task — unlike status_data there is no persisted
        fallback, because a frame from a finished run is not a live view of anything.

        Raises ValueError if the task isn't running; returns None if the task type has
        no live view.
        """
        task_info = self._running_tasks.get(task_uuid)
        if task_info is None or task_info.task is None:
            raise ValueError(f"Task {task_uuid} is not currently running")
        return await task_info.task.get_live_preview(source_uuid=source_uuid, **options)

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
            # The row is reused across runs, so the previous run's one-shot result
            # would otherwise ride along into this one — leaving a "running" task
            # advertising a "completed" result nothing ever consumes, and shipping
            # that dead payload on every poll. Cleared in memory too, so polls
            # answered from _running_tasks don't keep serving it.
            instance.result = None
        else:
            resp = await asyncio.to_thread(
                TasksApi(self._db_api_client).create_task_instance,
                db_models.CreateTaskInstancePayload(config_uuid=config.uuid),
            )
            instance = resp.instance
        task_info: TaskInfo = TaskInfo(task=task, config=config, instance=instance)

        task.set_instance_uuid(instance.uuid)
        self._running_tasks[instance.uuid] = task_info
        instance.status = TaskStatus.RUNNING
        started_at = datetime.now(timezone.utc)
        # Persist the running status and start timestamp, and register the
        # completion handler, BEFORE actually starting the task. A task whose
        # _init()/_run() fails entirely synchronously can finish — and have
        # _on_task_complete write its terminal ERROR status — before this
        # RUNNING write's thread-pool call would otherwise have landed; if
        # task.start() ran first, the late RUNNING write could then clobber
        # the terminal status, leaving the task stuck as "running" forever
        # with no in-memory entry left for Stop/Suspend/Cancel to act on.
        start_writes = [
            asyncio.to_thread(
                TasksApi(self._db_api_client).set_task_instance_status,
                instance.uuid,
                db_models.SetTaskStatusPayload(status=TaskStatus.RUNNING),
            ),
            # This write also retires the previous run's ended_at (see
            # set_task_timestamps_and_logs) so a resumed row can't read as
            # having ended before it started.
            asyncio.to_thread(
                TasksApi(self._db_api_client).set_task_instance_timestamps,
                instance.uuid,
                db_models.SetTaskTimestampsPayload(started_at=started_at),
            ),
        ]
        if resume_instance:
            start_writes.append(
                asyncio.to_thread(
                    TasksApi(self._db_api_client).set_task_instance_result,
                    instance.uuid,
                    db_models.SetTaskResultPayload(result={}),
                )
            )
        await asyncio.gather(*start_writes)
        t = asyncio.create_task(self._on_task_complete(task_info))
        self._completion_tasks.add(t)
        t.add_done_callback(self._completion_tasks.discard)
        await task.start()
        return task_info

    def start(self):
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._worker_task())

    async def _init(self):
        # Startup recovery inspects each interrupted instance's checkpoint to decide
        # paused-and-resumable vs. error, and hands it to the auto-restart path, so
        # this is the other listing that must opt into resume_data.
        resp = await asyncio.to_thread(
            TasksApi(self._db_api_client).list_task_instances,
            include_resume_data=True,
        )
        all_instances = resp.instances or []

        # "new" = created but server crashed before task.start() was reached.
        # Nothing useful to recover; remove them — unless this process is the one
        # starting it right now. _start_task registers the instance in
        # _running_tasks before its RUNNING write lands, so between those two
        # points a perfectly healthy task is sitting in the DB as "new". That
        # window only became reachable when recovery started retrying on later
        # worker passes instead of running once before any task could exist.
        new_tasks = [
            inst for inst in all_instances if inst.status == TaskStatus.NEW and inst.uuid not in self._running_tasks
        ]
        if new_tasks:
            logger.info(f"Removing {len(new_tasks)} never-started task instance(s)")
            await asyncio.to_thread(
                TasksApi(self._db_api_client).delete_task_instances,
                db_models.DeleteTaskInstancesPayload(task_uuids=[inst.uuid for inst in new_tasks]),
            )

        # "running" = server was killed while the task was executing.
        running_tasks = [inst for inst in all_instances if inst.status == TaskStatus.RUNNING]
        await self._reconcile_interrupted(
            running_tasks,
            no_checkpoint_message=(
                "The server stopped while this task was running, and it had no checkpoint to resume from."
            ),
        )

    async def _reconcile_interrupted(
        self,
        instances: List[db_models.TaskInstanceRead],
        no_checkpoint_message: str,
    ) -> None:
        """Decide the fate of instances left "running" with nothing driving them.

        If the task managed to save a checkpoint, treat it as paused so it can be
        resumed. If there is no checkpoint there is nothing to resume from, so mark
        it as an error — clearly visible, and not misleadingly resumable.

        Shared by both paths that can find an orphan: the startup recovery pass and
        the stale-heartbeat net. They differ only in how they noticed (hence
        *no_checkpoint_message*); what to do about it must not differ, or which of
        the two timers happened to fire first would decide whether a resumable job
        stays resumable.

        An instance this process is driving is not an orphan, so the filter lives
        here rather than in either caller. Startup recovery used to run exactly
        once, before any task could exist, and so needed no filter; now that it
        retries until db_server answers, a pass that succeeds late would otherwise
        pause (or error) a task that never stopped — and, for a persistent config,
        hand it to _auto_restart_with_guard for a second driver.
        """
        instances = [inst for inst in instances if inst.uuid not in self._running_tasks]
        if not instances:
            return

        resumable_instances: List[db_models.TaskInstanceRead] = []
        coros = []
        for inst in instances:
            recovery_message = None
            if inst.resume_data:
                recovered_status = db_models.TaskStatus.PAUSED
                logger.info(f"Task {inst.uuid} interrupted with checkpoint — marking paused (resumable)")
                resumable_instances.append(inst)
            else:
                recovered_status = db_models.TaskStatus.ERROR
                recovery_message = no_checkpoint_message
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
        # explicit user action, so it stays strictly opt-in per config (any config that
        # never set _persistent is left PAUSED for the user to resume manually) and is
        # subject to the crash-loop guard.
        if not resumable_instances:
            return
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

    async def _run_startup_recovery(self) -> bool:
        """Run the one-time recovery pass, retrying on later worker passes if it fails.

        _init() reads from db_server, which during a whole-stack restart is very
        likely not up yet — services start concurrently. Losing the recovery pass to
        that race used to be permanent for the boot: every instance orphaned by the
        restart stayed "running" forever with a frozen heartbeat, which is exactly
        the state where the UI lists a task the service says it isn't running.

        Returns True once recovery has completed (now or on an earlier call), so the
        stale-heartbeat net can wait for it rather than racing it.
        """
        if self._recovery_done:
            return True
        try:
            await self._init()
        except Exception:
            logger.exception("Manager startup recovery failed; will retry on the next pass")
            return False
        self._recovery_done = True
        return True

    async def _check_stale_tasks(self) -> None:
        """Reconcile running tasks whose heartbeat has gone stale.

        A task's heartbeat goes stale if the asyncio Task died without updating
        its DB status (e.g. the _on_task_complete DB write failed), or if the
        process that was driving it went away. Tasks that are still alive in
        _running_tasks are excluded from the check. Threshold = 4 ×
        HEARTBEAT_INTERVAL so a single missed beat doesn't trigger a false
        positive.

        What happens to a stale task is _reconcile_interrupted's call, not this
        method's: an orphan with a checkpoint is resumable however it was
        noticed. Skipped entirely until the startup recovery pass has run, so
        the two never race for the same orphan.
        """
        if not self._recovery_done:
            return
        stale_threshold = timedelta(seconds=Task.HEARTBEAT_INTERVAL * 4)
        now = datetime.now(timezone.utc)
        try:
            # resume_data is what decides resumable-vs-error below, and the
            # default listing omits it — without it every orphan looks
            # uncheckpointed.
            resp = await asyncio.to_thread(
                TasksApi(self._db_api_client).list_task_instances,
                include_resume_data=True,
            )
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
                    logger.warning("Task %s has stale heartbeat — reconciling", inst.uuid)
                    stale.append(inst)
            await self._reconcile_interrupted(
                stale,
                no_checkpoint_message=(
                    "No heartbeat from this task for over "
                    f"{stale_threshold}, and it had no checkpoint to resume from."
                ),
            )
        except Exception:
            logger.exception("Error checking stale tasks")

    def _check_wedged_tasks(self) -> None:
        """Notice running tasks this process *is* driving that have stopped progressing.

        The blind spot this fills: _check_stale_tasks deliberately skips anything in
        _running_tasks, because its finding is routed to _reconcile_interrupted, which
        marks the instance paused-or-error and — for a persistent config — auto-restarts
        it. Doing that to a task that is merely stuck would start a second driver
        alongside a first that still holds its subprocesses and relay leases. So a
        registered task that stops beating is invisible to every safety net, which is
        exactly the state a wedged watcher sat in.

        The same signal therefore means opposite things either side of that line. An
        *unregistered* stale row means no driver exists and one may be needed;
        a *registered* one means a driver exists and is stuck, where starting another is
        the worst available move. Hence detection only: this counts and reports, and
        deliberately changes no task state.

        It cannot be escalated to action on this evidence anyway. A task doing a long
        operation without calling `heartbeat(force=True)` first is indistinguishable
        from a wedged one — see Task.heartbeat's own warning — and that ambiguity is
        fine for telling a person and not fine for killing something.

        In-memory and synchronous: it reads each task's monotonic heartbeat rather than
        the DB's, so it still works when the DB write is what has jammed.
        """
        threshold = Task.HEARTBEAT_INTERVAL * self._WEDGED_HEARTBEAT_INTERVALS
        now = time.monotonic()
        wedged: Dict[str, float] = {}
        for uuid, task_info in self._running_tasks.items():
            if task_info.task is None:
                continue
            silent_for = task_info.task.seconds_since_heartbeat(now)
            if silent_for is not None and silent_for > threshold:
                wedged[uuid] = silent_for

        # Logged on the way in and out only. This runs every worker pass, and a task
        # wedged for a day would otherwise write the same line hundreds of times.
        for uuid, silent_for in wedged.items():
            if uuid not in self._wedged_tasks:
                task_info = self._running_tasks.get(uuid)
                name = task_info.config.name if task_info else uuid
                logger.warning(
                    "Task %r (%s) is running but has not reported progress for %.0fs. "
                    "It is still registered here, so nothing will recover it "
                    "automatically — if it never returns, restarting tasks_server is "
                    "the only way to clear it.",
                    name,
                    uuid,
                    silent_for,
                )
        for uuid in self._wedged_tasks:
            if uuid not in wedged and uuid in self._running_tasks:
                logger.info("Task %s is reporting progress again", uuid)

        self._wedged_tasks = wedged

    async def _refresh_unattached_count(self) -> None:
        """Count DB instances that say "running" but that this process isn't driving.

        get_status() is polled every few seconds and must stay an in-memory read
        (see apps/helpers/serviceStatus.py), so the DB side of the count is
        sampled here on the 60s worker pass and cached.

        Without it the status panel counts only _running_tasks while the Runs list
        renders the DB rows, so an orphaned task shows as "running" in one view and
        "0 running" in the other, with nothing to say which is right.
        """
        try:
            resp = await asyncio.to_thread(TasksApi(self._db_api_client).list_task_instances)
        except Exception:
            logger.exception("Error refreshing unattached task count")
            return
        self._unattached_count = sum(
            1
            for inst in (resp.instances or [])
            if inst.status == TaskStatus.RUNNING and inst.uuid not in self._running_tasks
        )

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
            # Recovery can fail on a boot-time race (db_server isn't reachable yet
            # while services start concurrently). It is retried on every pass below
            # until it runs — losing it for the whole boot leaves any instance
            # orphaned by the restart stuck "running" with a frozen heartbeat.
            # Resolve server URL overrides before recovery so a recovered task
            # starts against the configured host, not the compiled-in default.
            # Re-read every pass (not once) for the same reason recovery retries:
            # db_server may not be up yet at boot, and the setting can change at
            # any time from Settings → Servers.
            await self._reload_url_overrides_safe()
            await self._run_startup_recovery()
            while True:
                await asyncio.sleep(60)
                await self._reload_url_overrides_safe()
                await self._run_startup_recovery()
                await self._check_stale_tasks()
                await self._checkpoint_running_tasks()
                await self._prune_finished_task_instances()
                await self._refresh_unattached_count()
                # Last, and synchronous: it reads in-memory state only, so it still
                # reports when the DB calls above are what has jammed.
                self._check_wedged_tasks()
        except asyncio.CancelledError:
            logger.info("Periodic task was cancelled.")
        except Exception as e:
            logger.exception(e)
        finally:
            logger.info("Periodic task cleanup.")

    def get_status(self) -> dict:
        """Return service health and diagnostic info.

        ``details`` names the running tasks. That is redundant with the task list
        on the same Operations page, but it is still the answer to "what is this
        service busy with" — which is what the panel is read for.
        """
        notices: list[dict] = []
        for ti in self._running_tasks.values():
            if ti.task:
                notices.extend(ti.task.get_notices())

        rows = []
        paused = 0
        for ti in sorted(self._running_tasks.values(), key=lambda t: t.config.name.lower()):
            parts = []
            if ti.task and ti.task.is_paused_in_place():
                parts.append("paused")
                paused += 1
            elif ti.instance is not None:
                parts.append(ti.instance.status)
            progress = ti.task.get_progress() if ti.task else None
            # None is indeterminate progress, not 0% — don't render it as a number.
            if progress is not None:
                parts.append(f"{progress * 100:.0f}%")
            message = ti.task.get_status_message() if ti.task else ""
            if message:
                parts.append(message)
            # A stuck task's own status message is whatever it last managed to set,
            # which reads as though it were still doing that. Say otherwise.
            silent_for = self._wedged_tasks.get(ti.instance.uuid) if ti.instance else None
            if silent_for is not None:
                parts.append(f"NOT RESPONDING for {silent_for / 60:.0f}m")
            rows.append(row(f"{ti.config.name} ({ti.config.typename})", "  ·  ".join(p for p in parts if p)))

        running = len(self._running_tasks)
        summary = f"running: {running}"
        if paused:
            summary += f" ({paused} paused)"

        # Instances the DB still calls "running" that nothing here is driving. They
        # are what the Runs list shows while this panel would otherwise say zero, so
        # the disagreement is reported rather than left for the reader to discover.
        unattached = self._unattached_count
        if unattached:
            summary += f" · {unattached} unattached"
            notices.append(
                {
                    "key": "tasks_server:unattached",
                    "level": "warning",
                    "message": (
                        f"{unattached} task(s) are marked running in the database but are not attached to "
                        "this server. They are being recovered — resumable ones will be paused for you to "
                        "resume, or auto-resumed if the task is persistent."
                    ),
                }
            )

        # Tasks this server is driving that have stopped reporting progress. Unlike
        # `unattached` above, nothing is recovering these — that is the point of
        # saying so, since a stuck task is otherwise indistinguishable from a busy one
        # in every view the app has.
        wedged = len(self._wedged_tasks)
        if wedged:
            summary += f" · {wedged} not responding"
            longest = max(self._wedged_tasks.values())
            notices.append(
                {
                    "key": "tasks_server:wedged",
                    "level": "warning",
                    "message": (
                        f"{wedged} running task(s) have not reported progress for up to "
                        f"{longest / 60:.0f} minute(s). They are still attached to this server, so "
                        "they are not being recovered automatically and Stop may not reach them. "
                        "If they do not resume, restarting tasks_server will clear them."
                    ),
                }
            )

        return {
            "service": "tasks_server",
            "status": "ok",
            "tasks": {
                "running": running,
                "paused": paused,
                "unattached": unattached,
                "wedged": wedged,
            },
            "summary": summary,
            "notices": notices,
            "details": [
                section(f"Running tasks ({running})", rows, empty="No tasks running"),
            ],
        }
