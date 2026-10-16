"""Job queue and job state for the training server."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Statuses from which no further transitions are possible.
# "interrupted" means the server restarted mid-job; it can be restarted via restart_job().
TERMINAL_STATUSES = frozenset({"done", "error", "cancelled", "interrupted"})


@dataclass
class Job:
    job_id: str
    status: str  # awaiting_data | queued | running | done | error | cancelled | interrupted
    config: dict[str, Any]
    cancel_event: asyncio.Event
    created_at: datetime
    # Ordered list of {image_uuid, filename, status} dicts from the manifest.
    manifest: list[dict[str, str]] = field(default_factory=list)
    # Set of image UUIDs that must be in the cache before the job can start.
    required_uuids: set[str] = field(default_factory=set)
    # Directory where the trainer writes its output artifacts.
    artifact_dir: Optional[Path] = None
    queue_position: int = 0
    result: Optional[dict[str, Any]] = None
    # All events emitted so far (event_type, data); used for replay on reconnect.
    _events: list = field(default_factory=list)
    # One asyncio.Queue per active SSE subscriber.
    _subscribers: list = field(default_factory=list)
    # Path to the job state JSON file on disk (set by Manager after artifact_dir is known).
    state_path: Optional[Path] = None
    # Latest payload passed to the trainer's progress_cb (epoch, losses, val_accuracy,
    # …) and the wall-clock moment the worker picked the job up. Both feed the
    # Service Status detail view, which needs "how is training going right now"
    # without replaying the whole SSE event log. In-memory only: a restored
    # (interrupted) job has no live progress to report.
    last_progress: Optional[dict[str, Any]] = None
    started_at: Optional[datetime] = None
    # Latched from whichever progress event carried it. `warn_if_cpu` emits the CPU-fallback
    # warning once at the top of train(), and `last_progress` only ever holds the newest
    # event, so the app-wide notice needs a field that outlives the next epoch tick.
    device_warning: str = ""
    # A stop has been asked for but the trainer has not unwound yet. Cancellation is
    # cooperative and polled between epochs, so a running job stays `running` for up to
    # a whole epoch after the request — and on CPU that is the 10-50x figure the device
    # warning quotes. Without this the app has no way to tell "training" from "stopping",
    # and the two look identical for minutes.
    cancel_requested: bool = False
    # The last epoch a progress event reported, latched the way `device_warning` is and
    # persisted alongside the status. `last_progress` cannot answer this after a restart
    # — it is in-memory only — and this is what lets an interrupted job's resume offer
    # name the epoch it would continue from without opening the 300 MB checkpoint that
    # holds the authoritative number.
    #
    # Approximate by at most one: a server killed between the epoch event and
    # libreyolo's checkpoint write leaves this one ahead of the file. Only the label is
    # affected — the resume itself reads the epoch out of the checkpoint.
    last_epoch: Optional[int] = None

    def _persist(self) -> None:
        """Atomically write job state to disk for crash recovery."""
        if self.state_path is None:
            return
        try:
            data = {
                "job_id": self.job_id,
                "status": self.status,
                "config": self.config,
                "created_at": self.created_at.isoformat(),
                "manifest": self.manifest,
                "required_uuids": list(self.required_uuids),
                "artifact_dir": str(self.artifact_dir) if self.artifact_dir else None,
                "result": self.result,
                "last_epoch": self.last_epoch,
            }
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(self.state_path)
        except Exception:
            logger.exception("Failed to persist job state for %s", self.job_id)

    def emit(self, event_type: str, data: dict) -> None:
        """Broadcast an SSE event to all subscribers and store it for replay."""
        entry = (event_type, data)
        self._events.append(entry)
        for q in list(self._subscribers):
            q.put_nowait(entry)
        # Notify subscribers the stream is ending on any terminal event.
        is_terminal = event_type in ("done", "error", "interrupted") or (
            event_type == "status" and data.get("status") in TERMINAL_STATUSES
        )
        if is_terminal:
            for q in list(self._subscribers):
                q.put_nowait(None)  # sentinel → close
        # Persist state after status transitions and terminal events.
        if event_type in ("status", "done", "error", "interrupted"):
            self._persist()

    def to_dict(self) -> dict[str, Any]:
        config = self.config or {}
        return {
            "job_id": self.job_id,
            "status": self.status,
            "trainer_type": config.get("trainer_type", ""),
            "model_name": config.get("model_name", ""),
            "created_at": self.created_at.isoformat(),
            "queue_position": self.queue_position,
            "result": self.result,
            # Only an interrupted job can be resumed, so the offer is scoped here rather
            # than leaving every caller to pair the two fields correctly.
            "resumable_epoch": (self.last_epoch + 1) if (self.status == "interrupted" and self.last_epoch) else None,
        }

    async def stream(self) -> AsyncIterator[tuple[str, dict]]:
        """Yield (event_type, data) pairs: replay history then follow live events."""
        queue: asyncio.Queue = asyncio.Queue()
        past_events = list(self._events)
        is_terminal = self.status in TERMINAL_STATUSES

        if not is_terminal:
            self._subscribers.append(queue)

        try:
            for event_type, data in past_events:
                yield event_type, data

            if is_terminal:
                return

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item[0], item[1]
        finally:
            try:
                self._subscribers.remove(queue)
            except ValueError:
                pass


class JobQueue:
    """Single-worker (extensible to N via TRAINING_WORKER_COUNT) async job queue.

    Jobs start in ``awaiting_data`` state and are not queued for the worker until
    ``start_job()`` is called.  This allows the caller to push image data first.
    """

    def __init__(self, worker_count: int = 1):
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_count = worker_count
        self._workers: list[asyncio.Task] = []
        self._trainers: dict[str, Any] = {}

    def start(self) -> None:
        for _ in range(self._worker_count):
            task = asyncio.create_task(self._worker())
            self._workers.append(task)

    def register_trainer(self, name: str, trainer: Any) -> None:
        self._trainers[name] = trainer

    async def submit(
        self,
        config: dict[str, Any],
        manifest: Optional[list[dict[str, str]]] = None,
        artifact_dir: Optional[Path] = None,
    ) -> Job:
        """Create a new job in ``awaiting_data`` state.

        The job is NOT queued for the worker until :meth:`start_job` is called.
        """
        job_id = str(uuid4())
        m = manifest or []
        required = {entry["image_uuid"] for entry in m}
        job = Job(
            job_id=job_id,
            status="awaiting_data",
            config=config,
            cancel_event=asyncio.Event(),
            created_at=datetime.now(timezone.utc),
            manifest=m,
            required_uuids=required,
            artifact_dir=artifact_dir,
            queue_position=0,
        )
        self._jobs[job_id] = job
        job.emit("status", {"status": "awaiting_data"})
        return job

    def start_job(self, job_id: str, missing_uuids: list[str]) -> tuple[bool, list[str]]:
        """Transition a job from ``awaiting_data`` to ``queued``.

        Validates that all required image UUIDs are present in the cache.
        Returns ``(success, missing_uuids)``; ``success`` is False when data is
        still incomplete and the job is not queued.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False, []
        if job.status != "awaiting_data":
            # Already queued/running/done — idempotent.
            return job.status not in TERMINAL_STATUSES, []
        if missing_uuids:
            return False, missing_uuids
        position = self._queue.qsize()
        job.status = "queued"
        job.queue_position = position
        job.emit("status", {"status": "queued"})
        self._queue.put_nowait(job_id)
        return True, []

    def cancel(self, job_id: str) -> bool:
        """Cancel a job regardless of whether it is awaiting_data, queued, running, or interrupted."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status in ("awaiting_data", "queued", "interrupted"):
            job.status = "cancelled"
            job.emit("status", {"status": "cancelled"})
            return True
        if job.status == "running":
            job.cancel_event.set()
            job.cancel_requested = True
            # Not a status event: `running` is still the truth until the trainer returns,
            # and emitting a status here would trip the terminal-event detection in
            # `emit` and close every SSE subscriber on a job that is still going.
            job.emit(
                "progress",
                {"log": "Stop requested — training will end after the current epoch finishes."},
            )
            return True
        return False

    def restart_job(self, job_id: str, resume: bool = True) -> bool:
        """Re-queue a job that was interrupted by a server restart.

        `resume` carries into the trainer's config, which is what makes the run continue
        from `last.pt` instead of starting over. It defaults to true because that is
        what this method has always claimed to do and what every caller assumed: the
        trainer ignored the checkpoint entirely, so an interrupted 20-epoch run silently
        went back to epoch 1 and the only symptom was the loss climbing back to its
        opening value.

        Passed false by "start over", which is a real thing to want — a resume is
        refused outright on an architecture or config mismatch, and a run whose dataset
        has changed underneath it should not continue.
        """
        job = self._jobs.get(job_id)
        if not job or job.status != "interrupted":
            return False
        # On the config, not on the Job, because the config is what reaches the trainer
        # and what `_persist` already writes — a second field would have to be threaded
        # through both and kept in step with this one.
        job.config = {**(job.config or {}), "resume": bool(resume)}
        position = self._queue.qsize()
        job.status = "queued"
        job.queue_position = position
        job.cancel_event.clear()
        job.cancel_requested = False
        job.emit("status", {"status": "queued"})
        self._queue.put_nowait(job_id)
        return True

    def list_jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def forget(self, job_id: str) -> bool:
        """Drop a finished job from the queue's memory.

        Only terminal jobs, and never an ``interrupted`` one: that status means
        the server restarted mid-run and the job can still be restarted from its
        checkpoint, so forgetting it would strand work that is still recoverable.
        Used by artifact eviction, which has just deleted the directory this job's
        record points at — leaving the record would offer a restart that cannot
        read its own inputs.
        """
        job = self._jobs.get(job_id)
        if job is None or job.status not in TERMINAL_STATUSES or job.status == "interrupted":
            return False
        del self._jobs[job_id]
        return True

    async def shutdown(self) -> None:
        """Cancel all worker tasks and wait for them to persist their state."""
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                job = self._jobs.get(job_id)
                if job and job.status == "queued":
                    await self._run_job(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Worker error processing job %s", job_id)
            finally:
                self._queue.task_done()

    async def _run_job(self, job: Job) -> None:
        job.status = "running"
        job.queue_position = 0
        job.started_at = datetime.now(timezone.utc)
        job.emit("status", {"status": "running"})

        trainer_type = job.config.get("trainer_type", "")
        trainer = self._trainers.get(trainer_type)
        if trainer is None:
            msg = f"Unknown trainer type: {trainer_type!r}"
            logger.error(msg)
            job.status = "error"
            job.result = {"message": msg}
            job.emit("error", {"status": "error", "message": msg})
            return

        def progress_cb(data: dict) -> None:
            job.last_progress = data
            if data.get("device_warning"):
                job.device_warning = data["device_warning"]
            if isinstance(data.get("epoch"), int) and data["epoch"] != job.last_epoch:
                job.last_epoch = data["epoch"]
                # Written here because `emit` persists only on status transitions, and a
                # crash is precisely the case where no transition follows: without this
                # the state file still says whatever it did when the job went `running`,
                # and the resume offer could never name an epoch. One small JSON write
                # per epoch — minutes apart — and only when the number actually moves,
                # so a trainer reporting progress per batch does not write per batch.
                job._persist()
            job.emit("progress", data)

        try:
            result = await trainer.train(
                {**job.config, "manifest": job.manifest},
                progress_cb,
                job.cancel_event,
            )
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.emit("status", {"status": "cancelled"})
            else:
                job.status = "done"
                job.result = result
                job.emit("done", {"status": "done", **result})
        except asyncio.CancelledError:
            # Distinguish user-initiated cancel (cancel_event set) from SIGTERM/shutdown.
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.emit("status", {"status": "cancelled"})
            else:
                job.status = "interrupted"
                job.emit("status", {"status": "interrupted"})
            raise
        except Exception as e:
            logger.exception("Trainer error for job %s", job.job_id)
            job.status = "error"
            job.result = {"message": str(e)}
            job.emit("error", {"status": "error", "message": str(e)})
