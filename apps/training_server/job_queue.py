"""Job queue and job state for the training server."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"done", "error", "cancelled"})


@dataclass
class Job:
    job_id: str
    status: str  # queued | running | done | error | cancelled
    config: dict[str, Any]
    cancel_event: asyncio.Event
    created_at: datetime
    queue_position: int = 0
    result: Optional[dict[str, Any]] = None
    # All events emitted so far (event_type, data); used for replay on reconnect.
    _events: list = field(default_factory=list)
    # One asyncio.Queue per active SSE subscriber.
    _subscribers: list = field(default_factory=list)

    def emit(self, event_type: str, data: dict) -> None:
        """Broadcast an SSE event to all subscribers and store it for replay."""
        entry = (event_type, data)
        self._events.append(entry)
        for q in list(self._subscribers):
            q.put_nowait(entry)
        # Notify subscribers the stream is ending on any terminal event.
        is_terminal = event_type in ("done", "error") or (
            event_type == "status" and data.get("status") in TERMINAL_STATUSES
        )
        if is_terminal:
            for q in list(self._subscribers):
                q.put_nowait(None)  # sentinel → close

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
    """Single-worker (extensible to N via TRAINING_WORKER_COUNT) async job queue."""

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

    async def submit(self, config: dict[str, Any]) -> Job:
        job_id = str(uuid4())
        position = self._queue.qsize()
        job = Job(
            job_id=job_id,
            status="queued",
            config=config,
            cancel_event=asyncio.Event(),
            created_at=datetime.now(timezone.utc),
            queue_position=position,
        )
        self._jobs[job_id] = job
        await self._queue.put(job_id)
        return job

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job. Returns True if the cancel was accepted."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status == "queued":
            job.status = "cancelled"
            job.emit("status", {"status": "cancelled"})
            return True
        if job.status == "running":
            job.cancel_event.set()
            return True
        return False

    def list_jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                job = self._jobs.get(job_id)
                if job and job.status != "cancelled":
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
            job.emit("progress", data)

        try:
            result = await trainer.train(job.config, progress_cb, job.cancel_event)
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.emit("status", {"status": "cancelled"})
            else:
                job.status = "done"
                job.result = result
                job.emit("done", {"status": "done", **result})
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.emit("status", {"status": "cancelled"})
            raise
        except Exception as e:
            logger.exception("Trainer error for job %s", job.job_id)
            job.status = "error"
            job.result = {"message": str(e)}
            job.emit("error", {"status": "error", "message": str(e)})
