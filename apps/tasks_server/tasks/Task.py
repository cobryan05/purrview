from __future__ import annotations

import asyncio
import logging
import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from apps.helpers.consts import TaskAccent, TaskStatus
from apps.helpers.logUtils import LOG_LEVEL

if TYPE_CHECKING:
    from ..manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


class Task(ABC):
    class InternalKeys(str, Enum):
        PERSISTENT = "_persistent"
        ONESHOT_RESULT = "_oneshot_result"  # delete the task when the result is fetched
        # Crash-loop guard bookkeeping for PERSISTENT auto-restart, stored on the task
        # CONFIG's params (see Manager._auto_restart_with_guard). Deliberately NOT stored
        # in a task instance's resume_data: that blob is owned by the task itself and is
        # wholesale-replaced on every checkpoint (Task._build_resume_data), which would
        # silently erase this bookkeeping the moment the task checkpoints again.
        RESTART_INFO = "_restart_info"
        # TODO: Task *configs* need a way to delete or else Preview tasks can build up

    # How often (seconds) a running task pings the DB to confirm it is alive.
    # The stale-check threshold in the manager uses a multiple of this value.
    HEARTBEAT_INTERVAL: int = 30

    # How often (seconds) the manager's periodic checkpoint pass should request this
    # task write its resume data to the DB (Manager._checkpoint_running_tasks). Long-lived
    # live-collection tasks should override this with a shorter value; batch tasks that
    # complete quickly on their own can leave the default alone.
    CHECKPOINT_INTERVAL: int = 300

    # Does this task run until something stops it, rather than to completion?
    #
    # A training runs a fixed number of epochs and finishes; a stream watcher is
    # `while not self._cancel_flag.is_set()` and never does. They need different
    # controls (arm/disarm and uptime, not a progress bar toward a 100% that
    # never arrives) and they must be counted differently: a permanently running
    # watcher folded into a "1 running" badge makes that badge meaningless.
    #
    # Declared here rather than inferred, because nothing observable distinguishes
    # them. Indeterminate progress does not: CollectTrainingImagesTask reports
    # None the whole way and still completes. PERSISTENT does not either -- that
    # is about auto-restart, and the watcher is continuous *without* being
    # persistent.
    CONTINUOUS: bool = False

    # Should a config for this task be persistent when the caller does not say?
    #
    # "Persistent" means the instance auto-resumes after a server restart (see
    # Manager._reconcile_interrupted). Callers that care still pass an explicit
    # value; this only decides the default, and it belongs to the task type
    # rather than to whichever UI happened to launch it — a watcher started
    # from the Runs launcher and one started from an NVR card should not differ
    # in whether they survive a restart.
    #
    # False for batch work, deliberately: a one-off that silently resumed hours
    # later would be a surprise. True for the always-on kinds, where the
    # surprise runs the other way — a watcher that quietly stopped watching
    # because a service restarted is exactly the failure an NVR must not have.
    #
    # Note this only has teeth for tasks that checkpoint: auto-resume requires
    # resume_data, so a task that never calls set_resume_data is marked ERROR on
    # an interrupted restart no matter what this says.
    DEFAULT_PERSISTENT: bool = False

    # Optional card tint for this task type, in the Jobs pane's Active list.
    #
    # The pane is a grid of visually identical cards, and the question it has to
    # answer at a glance — "is that a training or the collection feeding it?" —
    # was only answerable by reading each card's typename badge. Colour is the
    # cheapest way to make the kinds separable without reading anything.
    #
    # Owned by the task type rather than by the pane so a new task ships its own
    # colour with it, instead of the Jobs UI carrying a hardcoded list of every
    # typename that exists. None means no tint (the plain card) — that is the
    # right answer for a task with no family to be confused with.
    #
    # A shared accent between two typenames is fine and intended: the useful
    # grouping is by *kind of work* (everything that trains, everything that
    # collects), not one hue per class.
    CARD_ACCENT: Optional[TaskAccent] = None

    # Is this task type machinery, rather than something a user builds a saved
    # config for?
    #
    # Most registered types are started by a screen that already knows every
    # parameter: the model import forms build an ImportModelTask, Review builds
    # a preview run, the NVR blips panel builds a harvest, the Watchers editor
    # builds a StreamWatcherTask. Offering those in the "new task config"
    # typename dropdown lists a dozen types of which only a few can be filled in
    # by hand at all — and picking one of the others gets you a config that
    # either refuses to start or needs uuids you would have to copy out of the
    # database.
    #
    # This hides them from that dropdown. It does *not* restrict what may be
    # launched: the API is unchanged, the dedicated screens still start these,
    # and the Task Configs screen's existing "Show system" toggle reveals them
    # for anyone who does want to hand-build one.
    #
    # Declared per type rather than inferred, for the reason CONTINUOUS is:
    # nothing observable separates "needs a uuid only Review can supply" from
    # "has a form a person can fill in". Default False so a newly written task
    # is offered — a task nobody can find is a worse failure than one listed
    # that need not have been.
    SYSTEM_TASK: bool = False

    def __init__(self, task_config_uuid: str, params: dict[str, Any], manager: Manager):
        self._config_uuid: str = task_config_uuid
        self._instance_uuid: Optional[str] = None
        self._status: TaskStatus = TaskStatus.PENDING
        self._params: dict[str, Any] = params
        self._task: asyncio.Task | None = None
        # None means "indeterminate" (frontend renders a striped progress bar).
        # The base default MUST stay 0.0, not None — tasks opt into indeterminate
        # progress explicitly (see CollectTrainingImagesTask), otherwise every
        # existing batch task would suddenly render as indeterminate.
        self._progress: Optional[float] = 0.0
        self._results: dict[str, Any] = {}
        # Why this is separate from _results: results are one-shot (fetched once,
        # then deleted), so the message explaining a failure used to vanish from
        # the UI the first time anything read it. This is persisted to
        # task_instances.error_message and survives for as long as the instance does.
        self._error_message: Optional[str] = None
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
        # Deliberately 0.0 rather than "now": `heartbeat()` rate-limits on this, and
        # seeding it with the clock would delay the first beat by a whole interval.
        # seconds_since_heartbeat() handles the not-yet-beaten case instead.
        self._last_heartbeat_time: float = 0.0
        # When _execute() began, so a task that has not had time to beat yet is
        # measured from its own start rather than from the epoch.
        self._started_monotonic: Optional[float] = None
        # When the cooperative stop was first asked for, so the manager can tell a
        # task that is shutting down from one that has ignored the request.
        self._cancel_requested_at: Optional[float] = None
        # Set only by force_stop(). Tells _execute that an incoming CancelledError is
        # ours and should end the task, rather than someone else's (loop shutdown)
        # which must be allowed to propagate.
        self._force_stopping: bool = False

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

    @classmethod
    def status_schema(cls) -> list[dict[str, Any]]:
        """
        Return an ordered list of widget descriptors describing the rich task-status
        card for this task type. Static per typename — fetched once by the frontend
        and cached (mirrors params_schema()/the existing get_task_type_schema path).

        Each descriptor is a dict:
            - id     — stable key. For "status"-sourced widgets this is the key
                       get_status_data() returns its value under.
            - type   — widget type name, resolved by a frontend registry.
            - source — "summary" (read from the already-polled TaskInfoRead payload,
                       e.g. "progress"/"statusMessage"/"logLines" — costs nothing extra)
                       or "status" (read from the per-instance status-data endpoint,
                       fetched only while a card is expanded). Defaults to "status"
                       when absent.
            - plus arbitrary widget-specific config keys (label, lines, columns, ...).

        One widget type is structural rather than a data view:

            {"id": ..., "type": "group", "label": ..., "collapsed": True,
             "widgets": [ ...nested descriptors, groups included... ]}

        A group holds no data and never appears in get_status_data() — its child
        ids remain the payload keys. Use it to fold a long diagnostic block down
        to one line: `collapsed: True` starts it closed, and the frontend builds
        a closed group's children only when it is first opened (so nothing inside
        one costs anything until asked for). The user's open/closed choice
        survives the card rebuilds that a running task triggers constantly.

        The base implementation reproduces today's card exactly (progress bar +
        status message + log tail) so every existing task is unaffected unless it
        overrides this.
        """
        return [
            {"id": "progress", "type": "progress_bar", "source": "summary"},
            {"id": "statusMessage", "type": "text", "source": "summary"},
            {"id": "log", "type": "log_tail", "source": "summary", "lines": 5},
        ]

    def get_status_data(self) -> dict[str, Any]:
        """
        Return live values for this task's "status"-sourced status_schema() widgets,
        keyed by widget id.

        MUST be cheap and non-blocking — no I/O. This is called synchronously from a
        poll handler while a task card is expanded; anything that awaits I/O here
        blocks that handler.
        """
        return {}

    async def get_live_preview(self, source_uuid: Optional[str] = None, **options: Any) -> Optional[dict[str, Any]]:
        """
        Return the most recent frame this task is working on, plus whatever it currently
        believes about that frame, for a live drill-down view.

        Unlike get_status_data() this MAY do CPU work (JPEG encoding) and so is async —
        but it must remain *pull*-driven: encode on request, never on every frame. A
        task with no viewer must pay nothing, which is why nothing here is precomputed
        in the hot loop.

        Returns None when this task type has no live view (the default), which the web
        layer reports as "not available" rather than an error.
        """
        return None

    async def handle_command(self, command: str, params: dict[str, Any]) -> None:
        """
        Handle a UI-issued command sent via POST /api/tasks/instances/command
        (Manager.send_task_command()).

        MUST ONLY set a flag or push to a queue for the running _run() loop to pick
        up later — exactly like the existing _data_req_flag/_cancel_flag pattern.
        NEVER perform I/O directly here: this is invoked from a request handler, and
        any blocking or awaited I/O in this method stalls that handler.

        The base implementation raises NotImplementedError; the web layer translates
        that into a clean failure response rather than a 500. Override in subclasses
        that advertise a "task_command" status action.
        """
        raise NotImplementedError(f"{self.typename()} does not implement any commands")

    async def _execute(self) -> None:
        self._started_monotonic = time.monotonic()
        try:
            self.log("Initializing")
            await self._init(params=self._params, resume_data=self._resume_data)
            self._status = TaskStatus.RUNNING
            self.log("Running")
            results = await self._run()
            self._results = {"status": TaskStatus.COMPLETED, "data": results}
            self._status = TaskStatus.COMPLETED
            self.log("Completed")
        except asyncio.CancelledError:
            # Handled explicitly because CancelledError is a BaseException, so the
            # `except Exception` below never sees it. Left unhandled it would skip
            # both branches with _status still RUNNING, and _on_task_complete would
            # then persist "running" as the task's *final* status — a finished task
            # that the DB calls running forever, which is the shape of bug this
            # whole area keeps producing.
            if not self._force_stopping:
                # Not our doing — a loop shutdown, most likely. Cancellation has to
                # keep travelling, or shutdown hangs on a task refusing to end.
                raise
            self._status = TaskStatus.ERROR
            self._error_message = "Force-stopped: the task did not respond to the stop request."
            self._results = {"status": TaskStatus.ERROR, "message": self._error_message}
            self.log("Force-stopped")
        except Exception as e:
            self._status = TaskStatus.ERROR
            logger.error(e, exc_info=True)
            # The exception type carries real diagnostic weight when str(e) is
            # empty or terse (a bare TimeoutError says nothing on its own).
            self._error_message = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            self._results = {"status": TaskStatus.ERROR, "message": str(e)}
            self.log(f"Error: {e}")
        finally:
            # Completion always ends at a determinate 100% — even a task that ran with
            # indeterminate (None) progress the whole way must not leave the frontend's
            # striped bar showing after the task is actually done.
            self._progress = 100.0
            try:
                await self._deinit()
            except Exception:
                # _task_done is what everything waiting on this task keys off:
                # wait_for_task_done, and through it _on_task_complete, which owns
                # removing the task and writing its final status. A _deinit that
                # raised used to skip it and strand all of that — and a task being
                # torn down mid-operation, which is exactly what a force-stop
                # produces, is when _deinit is most likely to raise.
                logger.exception("Error during _deinit for task %s", self._instance_uuid)
            finally:
                self._task_done.set()

    def _update_progress(self, percent: Optional[float]):
        """Store progress. Pass None for indeterminate progress (striped bar in the UI)."""
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

    def seconds_since_heartbeat(self, now: Optional[float] = None) -> Optional[float]:
        """How long since this task last reported progress. None before it starts.

        In-memory and monotonic, unlike the DB `heartbeat_at` the orphan check reads:
        the question here is whether *this* process's task is still iterating, which
        needs no round trip and must stay answerable when the DB write is the thing
        that has stopped working.

        Measured from the task's start until the first beat lands, because
        `_last_heartbeat_time` sits at 0.0 until then — read raw, a task one second
        old would look like it had been silent since the epoch.
        """
        if self._started_monotonic is None:
            return None
        reference = self._last_heartbeat_time or self._started_monotonic
        return (time.monotonic() if now is None else now) - reference

    def get_resume_data(self) -> dict[str, Any]:
        """Return the resume data for the task."""
        return dict(self._resume_data)

    def get_status(self) -> TaskStatus:
        """Returns the current status"""
        return self._status

    def get_status_message(self) -> str:
        """Returns any message to return with task status"""
        return ""

    def get_error_message(self) -> Optional[str]:
        """The message explaining why this task failed, or None if it hasn't.

        Persisted alongside the ERROR status so the reason outlives the one-shot
        result payload and stays visible on the task's card.
        """
        return self._error_message

    def set_error_message(self, message: str) -> None:
        """Record a failure the task itself didn't raise.

        For the Manager, which marks tasks ERROR for reasons outside _execute
        (a pause that timed out, a stale heartbeat) and would otherwise leave the
        card saying "error" with no reason at all.
        """
        self._error_message = message

    def get_status_actions(self) -> list[dict]:
        """Returns action buttons to show in the live task status UI while the task is running.

        Each action is a dict with at minimum a "type" key naming the action kind and
        a "label" for the button text; other keys are kind-specific. Existing kinds:
            - "open_review" — {"type": "open_review", "label": ..., "payload": {"tag_uuids": [...]}}
            - "open_preview_run" — {"type": "open_preview_run", "label": ...,
                                 "payload": {"run_uuid": <str>}}. Opens that persisted
                                 preview run's triage panel directly. For a task whose
                                 output *is* a run of proposals: sending the user to
                                 Review and asking them to find it again in Saved
                                 Previews is how those runs got lost.
            - "task_command" — {"type": "task_command", "label": ..., "command": <str>,
                                 "params": {...}}. The frontend posts {taskUuid, command, params}
                                 to POST /api/tasks/instances/command, which forwards to this
                                 task's handle_command(command, params). Only meaningful for a
                                 task that overrides handle_command() (the base raises
                                 NotImplementedError).
        """
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

    def get_progress(self) -> Optional[float]:
        """Return the current progress of the task, or None if indeterminate."""
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

    async def save_checkpoint_now(self) -> bool:
        """Write the current _resume_data to the DB immediately.

        The manager's periodic pass (Manager._checkpoint_running_tasks) collects
        checkpoints roughly once a minute, which is fine for progress but not for
        the fact that a task has taken ownership of something outside this process
        — a remote training job, say. Losing that to a restart in the collection
        window leaves the remote work running with nothing pointing at it, so the
        task can force the write at the moment it acquires such a handle.

        Falls back to the periodic pass (by leaving _data_ready_flag set) if the
        write fails, so a db_server blip costs a delay rather than the task.
        Returns whether the direct write succeeded.
        """
        self._data_ready_flag.set()
        if not self._instance_uuid:
            return False
        try:
            await self._manager.save_task_resume_data(self._instance_uuid, self.get_resume_data())
        except Exception:
            logger.warning(
                "Immediate checkpoint failed for task %s; leaving it to the periodic pass", self._instance_uuid
            )
            return False
        self._data_ready_flag.clear()
        return True

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
        """Ask the task to stop cooperatively.

        Sets the flag a task's own loop polls; it does not interrupt anything. A task
        that never checks the flag, or is parked in a call that cannot be interrupted,
        will not stop — which is what force_stop() is the backup for.
        """
        if self._cancel_requested_at is None:
            self._cancel_requested_at = time.monotonic()
        self._cancel_flag.set()

    def seconds_since_cancel_requested(self, now: Optional[float] = None) -> Optional[float]:
        """How long this task has been asked to stop. None if it never was."""
        if self._cancel_requested_at is None:
            return None
        return (time.monotonic() if now is None else now) - self._cancel_requested_at

    def force_stop(self) -> bool:
        """Escalate from asking to interrupting: cancel the asyncio task itself.

        The backup for a cooperative stop that never lands. Returns True if a
        cancellation was delivered — False if there was nothing to cancel, or the task
        had already finished on its own between the decision and this call.

        **This is not a guaranteed kill, and must not be described as one.** It raises
        CancelledError at the task's current await point, so it works on a task parked
        at an ordinary await and does nothing whatsoever to one blocked inside
        `asyncio.to_thread` — that thread cannot be interrupted, which is why bounding
        the calls that run in threads (see apps/helpers/apiCall.py) matters more than
        this does. Treat it as recovering the cases that are recoverable, not as a
        general answer to a stuck task.
        """
        if self._task is None or self._task.done():
            return False
        self._force_stopping = True
        self.log("Stop request ignored — interrupting the task")
        return self._task.cancel()

    def is_force_stopping(self) -> bool:
        """Whether this task has been interrupted rather than merely asked to stop."""
        return self._force_stopping

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
