"""Manager for Training Server"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from apps import APPS, FILES_ROOT
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.model_type_map import resolve_trainer_type
from apps.helpers.serviceStatus import fmt_bytes, fmt_duration, row, section

from .image_cache import ImageCache
from .job_queue import Job, JobQueue

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

_DEFAULT_ARTIFACTS_DIR = "/training-artifacts"
_DEFAULT_EVICTION_SWEEP_SEC = 3600  # background age/size sweep interval


def _fmt_counts(counts: Dict[str, int]) -> str:
    """``{"done": 2, "running": 1}`` → ``"1 running, 2 done"`` (busiest first)."""
    return ", ".join(f"{n} {status}" for status, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


class Manager:
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        artifacts_dir: Optional[str] = None,
    ):
        worker_count = int(os.environ.get("TRAINING_WORKER_COUNT", "1"))
        self._queue = JobQueue(worker_count=worker_count)
        self._cache = ImageCache(cache_dir=cache_dir)
        self._artifacts_dir = Path(artifacts_dir or os.environ.get("TRAINING_ARTIFACTS_DIR", _DEFAULT_ARTIFACTS_DIR))
        self._config: Dict[str, Any] = {"service": APPS.TRAINING_SERVER}
        self._eviction_sweep_sec = int(os.environ.get("TRAINING_CACHE_SWEEP_INTERVAL_SEC", _DEFAULT_EVICTION_SWEEP_SEC))
        self._eviction_task: Optional[asyncio.Task] = None
        # Serialises artifact sweeps: tasks_server evicts after every training
        # run, and two runs finishing together would otherwise both select and
        # both try to delete the same directories.
        self._artifact_eviction_lock = asyncio.Lock()
        self._started_at = time.time()

    def start(self) -> None:
        self._cache.ensure_dir()
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Reload persisted job state from the last run before starting workers.
        self._restore_jobs()

        # Import trainer modules to trigger @register_trainer decorators.
        from .trainers import TRAINER_REGISTRY  # noqa: F401
        from .trainers import binary_image_classifier  # noqa: F401
        from .trainers import crop_classifier_trainer  # noqa: F401
        from .trainers import model_converter  # noqa: F401
        from .trainers import multi_label_classifier  # noqa: F401
        from .trainers import object_detection_trainer  # noqa: F401
        from .trainers import ultralytics_detection_trainer  # noqa: F401

        for name, trainer in TRAINER_REGISTRY.items():
            self._queue.register_trainer(name, trainer)
            logger.info("Registered trainer: %s", name)

        self._queue.start()
        if self._eviction_sweep_sec > 0:
            self._eviction_task = asyncio.get_event_loop().create_task(self._periodic_eviction())
        logger.info("Training server started (worker_count=%d)", self._queue._worker_count)

    async def _periodic_eviction(self) -> None:
        """Sweep the image cache for age/size eviction on a fixed interval,
        independent of ingest activity, so a cache that stops receiving
        pushes still shrinks back down over time."""
        try:
            while True:
                await asyncio.sleep(self._eviction_sweep_sec)
                try:
                    await asyncio.to_thread(self._cache.evict)
                except Exception:
                    logger.exception("Periodic cache eviction sweep failed")
        except asyncio.CancelledError:
            pass

    def _restore_jobs(self) -> None:
        """Reload job state from disk after a restart."""
        for state_file in sorted(self._artifacts_dir.glob("*/job_state.json")):
            try:
                data = json.loads(state_file.read_text())
                job_id = data["job_id"]
                status = data["status"]

                # Jobs that were active when the server crashed are interrupted.
                if status in ("running", "queued"):
                    status = "interrupted"

                artifact_dir = Path(data["artifact_dir"]) if data.get("artifact_dir") else None

                job = Job(
                    job_id=job_id,
                    status=status,
                    config=data.get("config", {}),
                    cancel_event=asyncio.Event(),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    manifest=data.get("manifest", []),
                    required_uuids=set(data.get("required_uuids", [])),
                    artifact_dir=artifact_dir,
                    queue_position=0,
                    result=data.get("result"),
                    state_path=state_file,
                )
                # Seed the event history so SSE replay shows the current status.
                job._events = [("status", {"status": status})]

                self._queue._jobs[job_id] = job

                # Re-pin images for jobs that still need them.
                if status == "awaiting_data" and job.required_uuids:
                    self._cache.pin(job.required_uuids)

                logger.info("Restored job %s (status=%s)", job_id, status)
            except Exception:
                logger.exception("Failed to restore job from %s", state_file)

    async def shutdown(self) -> None:
        """Cancel workers and let them persist interrupted state before exit."""
        if self._eviction_task is not None:
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass
        await self._queue.shutdown()

    async def get_server_config(self) -> Dict[str, Any]:
        return dict(self._config)

    async def submit_job(
        self,
        config: Dict[str, Any],
        manifest: Optional[List[Dict[str, str]]] = None,
    ) -> Job:
        """Create a new job in ``awaiting_data`` state.

        ``trainer_type`` is derived from ``model_type`` via the shared map when
        not provided explicitly.
        """
        config = dict(config)

        # Derive trainer_type from model_type (+ architecture) when absent.
        if "trainer_type" not in config:
            model_type = config.get("model_type", "")
            architecture = config.get("architecture", "")
            config["trainer_type"] = resolve_trainer_type(model_type, architecture)

        m = manifest or []
        # Pin all required UUIDs in the cache so they won't be evicted.
        required_uuids = {entry["image_uuid"] for entry in m}
        if required_uuids:
            self._cache.pin(required_uuids)

        job = await self._queue.submit(config, manifest=m)

        # Now that we have the real job_id, set the artifact dir and update config.
        artifact_dir = self._artifacts_dir / job.job_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        job.artifact_dir = artifact_dir
        job.state_path = artifact_dir / "job_state.json"

        # Inject cache_dir and artifact_dir into config so the trainer can use them.
        self._cache.ensure_dir()
        job.config = {
            **config,
            "cache_dir": self._cache.cache_dir,
            "artifact_dir": str(artifact_dir),
        }
        # Persist initial state now that all fields are set.
        job._persist()
        return job

    def get_missing_images(self, job_id: str) -> Optional[List[str]]:
        """Return image UUIDs from the job's manifest that are not yet cached.

        Returns None if the job doesn't exist.
        """
        job = self._queue.get_job(job_id)
        if job is None:
            return None
        return self._cache.missing(list(job.required_uuids))

    def ingest_images(self, job_id: str, tar_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Ingest a tar of images for a job.

        Only images whose UUIDs are in the job's required set are accepted.
        Returns None if the job doesn't exist.
        """
        job = self._queue.get_job(job_id)
        if job is None:
            return None
        return self._cache.ingest_tar(tar_bytes, required_uuids=job.required_uuids)

    def ingest_local(self, job_id: str, entries: List[tuple[str, str]]) -> Optional[Dict[str, Any]]:
        """Zero-copy ingest: populate the cache directly from filenames under
        ``FILES_ROOT``, without receiving any bytes over HTTP.

        Only works when this process can see ``FILES_ROOT`` locally (e.g. the
        pusher and training_server share a container/bind mount). UUIDs whose
        source file isn't reachable here come back in ``not_found`` — the
        caller should push those through the ordinary tar-upload path.
        Returns None if the job doesn't exist.
        """
        job = self._queue.get_job(job_id)
        if job is None:
            return None
        pairs = [(uuid, FILES_ROOT / filename) for uuid, filename in entries]
        return self._cache.ingest_local(pairs, required_uuids=job.required_uuids)

    def ingest_source_model(self, job_id: str, data: bytes) -> Optional[Dict[str, Any]]:
        """Ingest the raw bytes of a source ``.pt`` checkpoint for a convert_model job.

        Written under an ``_input`` subdirectory of the job's artifact_dir so it is
        never swept up by ``get_artifact`` (which only tars top-level files in
        artifact_dir). Returns None if the job doesn't exist.
        """
        job = self._queue.get_job(job_id)
        if job is None:
            return None
        input_dir = job.artifact_dir / "_input"
        input_dir.mkdir(parents=True, exist_ok=True)
        source_path = input_dir / "source_model.bin"
        source_path.write_bytes(data)

        job.config["source_model_path"] = str(source_path)
        job._persist()

        return {"received": len(data), "source_model_path": str(source_path)}

    def start_job(self, job_id: str) -> tuple[bool, list[str]]:
        """Transition a job from ``awaiting_data`` → ``queued`` if data is complete."""
        missing = self.get_missing_images(job_id) or []
        ok, still_missing = self._queue.start_job(job_id, missing)
        return ok, still_missing

    def get_artifact_dir(self, job_id: str) -> Optional[Path]:
        """Return the artifact directory for a completed job, or None."""
        job = self._queue.get_job(job_id)
        if job is None or job.status != "done":
            return None
        return job.artifact_dir

    def cancel_job(self, job_id: str) -> bool:
        job = self._queue.get_job(job_id)
        if job:
            # Release cache pins when a job is cancelled.
            self._cache.unpin(job.required_uuids)
        return self._queue.cancel(job_id)

    def restart_job(self, job_id: str) -> bool:
        """Re-queue an interrupted job. Images remain cached; training resumes from checkpoint."""
        return self._queue.restart_job(job_id)

    def list_jobs(self) -> list[Job]:
        return self._queue.list_jobs()

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._queue.get_job(job_id)

    # ── Cache maintenance (landing page / ops) ───────────────────────────────

    def get_cache_stats(self) -> Dict[str, Any]:
        """Occupancy and bounds of the image cache. Does filesystem I/O."""
        return self._cache.stats()

    def clear_cache(self, include_pinned: bool = False) -> Dict[str, Any]:
        """Delete cached images, keeping images pinned by live jobs unless asked."""
        return self._cache.clear(include_pinned=include_pinned)

    def evict_cache(self) -> Dict[str, Any]:
        """Run the age+size eviction sweep now, and report what it changed."""
        before = self._cache.stats()
        self._cache.evict()
        after = self._cache.stats()
        return {
            "removed": before["files"] - after["files"],
            "freed_bytes": before["bytes"] - after["bytes"],
            "files": after["files"],
            "bytes": after["bytes"],
        }

    #: Finished jobs whose artifact directories are kept when the caller names no
    #: number. Every completed run used to be kept forever — a single object-detection
    #: job's directory here measured 1.1 GB — so this is a bound where there was none,
    #: not a tightening of an existing one.
    DEFAULT_ARTIFACTS_KEEP = 3

    async def evict_artifacts(self, keep: Optional[int] = None) -> Dict[str, Any]:
        """Delete the artifact directories of all but the ``keep`` newest finished jobs.

        Only ``done``/``error``/``cancelled`` jobs are candidates. A queued, running
        or awaiting job is still going to write there; an ``interrupted`` one can be
        restarted from the checkpoint in that directory, and deleting it would throw
        away recoverable work. Newest-first by creation time, so the run whose weights
        were just fetched is never the one removed.

        The job record is dropped along with its directory — a job whose artifacts are
        gone can neither be restarted nor downloaded, and leaving it listed offers both.

        Deleting a gigabyte of checkpoints is slow enough that it must not block the
        event loop, so the filesystem work — and only that — runs in a worker thread.
        Selecting the victims and forgetting them stay on the loop, because
        :class:`JobQueue` carries no lock: it is an event-loop structure whose
        callers are expected to be the loop, and this sweep was the only one that
        was not. (Under CPython that was survivable rather than broken: ``sorted``
        materialises the dict view in a C loop that never drops the GIL. It is not a
        property worth depending on.)

        The lock is the part that fixes real behaviour. tasks_server evicts after
        every training run, so two runs finishing together produce two sweeps that
        select the same victims; both then walk and delete the same directories, and
        each reports the bytes the other freed as its own.
        """
        keep = self.DEFAULT_ARTIFACTS_KEEP if keep is None else max(0, int(keep))

        async with self._artifact_eviction_lock:
            # On the loop: snapshot the victims. Nothing can be added to or removed
            # from the queue between these two lines.
            evictable = [j for j in self._queue.list_jobs() if j.status in ("done", "error", "cancelled")]
            # list_jobs() is already newest-first.
            doomed = [(job.job_id, job.artifact_dir) for job in evictable[keep:]]

            forgettable, removed, freed = await asyncio.to_thread(self._delete_artifact_dirs, doomed)

            # Back on the loop: drop the records whose directories are now gone.
            for job_id in forgettable:
                self._queue.forget(job_id)

            if removed:
                logger.info("Evicted %d artifact director(ies), freeing %d bytes", len(removed), freed)
            after = await asyncio.to_thread(self._artifacts_stats)

        return {
            "removed": removed,
            "freed_bytes": freed,
            "kept": min(len(evictable), keep),
            "jobs": after["jobs"],
            "bytes": after["bytes"],
        }

    def _delete_artifact_dirs(self, doomed: List[tuple]) -> tuple[List[str], List[str], int]:
        """Filesystem half of :meth:`evict_artifacts`; runs in a worker thread.

        Touches no queue state — it is handed plain ``(job_id, dir)`` pairs and
        reports back which ids the caller should forget, which it actually deleted
        (a job whose directory was already gone is forgotten but not "removed"),
        and how many bytes that freed.
        """
        forgettable: List[str] = []
        removed: List[str] = []
        freed = 0
        for job_id, job_dir in doomed:
            if job_dir is None or not Path(job_dir).is_dir():
                forgettable.append(job_id)
                continue
            try:
                for path in Path(job_dir).rglob("*"):
                    if path.is_file():
                        try:
                            freed += path.stat().st_size
                        except OSError:
                            pass
                shutil.rmtree(job_dir)
                forgettable.append(job_id)
                removed.append(job_id)
            except OSError:
                logger.exception("Could not remove artifact directory %s", job_dir)
        return forgettable, removed, freed

    def _artifacts_stats(self) -> Dict[str, Any]:
        """Disk used by job artifact directories (trained weights, logs)."""
        jobs = 0
        total_bytes = 0
        if self._artifacts_dir.exists():
            for job_dir in self._artifacts_dir.iterdir():
                if not job_dir.is_dir():
                    continue
                jobs += 1
                for path in job_dir.rglob("*"):
                    try:
                        if path.is_file():
                            total_bytes += path.stat().st_size
                    except OSError:
                        continue
        return {"dir": str(self._artifacts_dir), "jobs": jobs, "bytes": total_bytes}

    def get_info(self) -> Dict[str, Any]:
        """Heavier, on-demand detail for the landing page.

        Deliberately *not* folded into :meth:`get_status`: this walks the cache
        and artifact directories, and ``/status`` is polled every few seconds by
        the notices loop. Same ``summary``/``details`` shape, so the landing
        page renders both with one code path.
        """
        from .trainers import TRAINER_REGISTRY

        cache = self.get_cache_stats()
        artifacts = self._artifacts_stats()

        pct = (cache["bytes"] / cache["max_bytes"] * 100) if cache["max_bytes"] else 0
        max_age = cache["max_age_seconds"]
        now = time.time()

        cache_rows = [
            row("Directory", cache["dir"]),
            row("Cached images", f"{cache['files']}  ·  {fmt_bytes(cache['bytes'])}"),
            row("Size limit", f"{fmt_bytes(cache['max_bytes'])}  ·  {pct:.0f}% used"),
            row("Age limit", fmt_duration(max_age) if max_age else "disabled"),
            row("Pinned by live jobs", cache["pinned"]),
        ]
        if cache["oldest_mtime"] is not None:
            cache_rows.append(row("Oldest entry", f"{fmt_duration(now - cache['oldest_mtime'])} old"))
        if cache["newest_mtime"] is not None:
            cache_rows.append(row("Newest entry", f"{fmt_duration(now - cache['newest_mtime'])} old"))

        jobs = self._queue.list_jobs()
        by_status: Dict[str, int] = {}
        for job in jobs:
            by_status[job.status] = by_status.get(job.status, 0) + 1

        return {
            "service": "training_server",
            "status": "ok",
            "cache": cache,
            "artifacts": artifacts,
            "summary": f"cache {fmt_bytes(cache['bytes'])} / {fmt_bytes(cache['max_bytes'])}",
            "details": [
                section("Image cache", cache_rows),
                section(
                    "Artifacts",
                    [
                        row("Directory", artifacts["dir"]),
                        row("Job directories", artifacts["jobs"]),
                        row("Disk used", fmt_bytes(artifacts["bytes"])),
                    ],
                ),
                section(
                    "Server",
                    [
                        row("Uptime", fmt_duration(now - self._started_at)),
                        row("Workers", self._queue._worker_count),
                        row("Jobs known", f"{len(jobs)}" + (f"  ({_fmt_counts(by_status)})" if by_status else "")),
                        row("Trainers", ", ".join(sorted(TRAINER_REGISTRY)) or "none registered"),
                        row(
                            "Eviction sweep",
                            fmt_duration(self._eviction_sweep_sec) if self._eviction_sweep_sec else "disabled",
                        ),
                    ],
                ),
            ],
        }

    def get_status(self) -> Dict[str, Any]:
        """Service health plus per-job training stats for the Service Status panel.

        Reads the in-memory job queue only — no artifact-dir or cache I/O, since
        /status is polled by the notices loop every few seconds.
        """
        jobs = self._queue.list_jobs()
        running = [j for j in jobs if j.status == "running"]
        queued = [j for j in jobs if j.status == "queued"]
        awaiting = [j for j in jobs if j.status == "awaiting_data"]

        summary_parts = [f"training: {len(running)}"]
        if queued:
            summary_parts.append(f"{len(queued)} queued")
        if awaiting:
            summary_parts.append(f"{len(awaiting)} awaiting data")
        if not running and not queued and not awaiting:
            summary_parts = ["idle"]

        now = datetime.now(timezone.utc)

        def _job_label(job: Job) -> str:
            config = job.config or {}
            return config.get("model_name") or job.job_id[:8]

        running_rows = []
        for job in running:
            progress = job.last_progress or {}
            parts = []
            epoch, total_epochs = progress.get("epoch"), progress.get("total_epochs")
            if epoch is not None and total_epochs:
                parts.append(f"epoch {epoch}/{total_epochs}")
            frac = progress.get("progress")
            if isinstance(frac, (int, float)):
                parts.append(f"{frac * 100:.0f}%")
            if progress.get("train_loss") is not None:
                parts.append(f"loss {progress['train_loss']}")
            # val_loss is 0.0 for the detection trainers, which don't expose it
            # separately — reporting a hard zero as a metric would be a lie.
            if progress.get("val_loss"):
                parts.append(f"val_loss {progress['val_loss']}")
            if progress.get("val_accuracy") is not None:
                parts.append(f"val_acc {progress['val_accuracy']}")
            if job.started_at is not None:
                parts.append(f"elapsed {fmt_duration((now - job.started_at).total_seconds())}")
            if not parts:
                parts.append("starting…")
            trainer_type = (job.config or {}).get("trainer_type")
            if trainer_type:
                parts.append(trainer_type)
            running_rows.append(row(_job_label(job), "  ·  ".join(parts)))

        details = [
            section(f"Training now ({len(running)})", running_rows, empty="No training running"),
        ]
        pending = queued + awaiting
        if pending:
            pending_rows = [
                row(
                    _job_label(job),
                    (f"#{job.queue_position} queued" if job.status == "queued" else "awaiting data"),
                )
                for job in pending
            ]
            details.append(section(f"Waiting ({len(pending)})", pending_rows))

        return {
            "service": "training_server",
            "status": "ok",
            "training": {
                "running": len(running),
                "queued": len(queued),
                "awaiting_data": len(awaiting),
            },
            "summary": "  ·  ".join(summary_parts),
            "details": details,
            "notices": self._cpu_training_notices(running),
        }

    @staticmethod
    def _expects_gpu() -> bool:
        """Whether this deployment was set up expecting a training GPU.

        Written by ``setup.sh``: 0 only where the operator answered "continue
        anyway" on a host with no GPU, which makes CPU training the thing they
        asked for rather than a fault. Anything else — including the variable
        being absent, as it is for a container built before this existed —
        counts as expecting one, so the warning fails loud rather than silent.
        """
        return os.environ.get("TRAINING_EXPECTS_GPU", "1").strip() not in ("0", "false", "False")

    def _cpu_training_notices(self, running: List[Job]) -> List[Dict[str, Any]]:
        """A standing notice for a job that fell back to the CPU.

        The trainer already logs this and reports it as the task's status
        message, but both are only visible to someone who opens that job — and
        the failure this catches (a container that quietly lost GPU access) is
        one you otherwise notice by a job taking hours instead of minutes. The
        notices bar is app-wide, so it is seen without going to look.

        Keyed per job so two CPU jobs do not collapse into one line, and drawn from
        the running jobs only, so it clears when the job ends.

        "Ends" is not "was asked to stop": cancellation is cooperative and polled
        between epochs, so a stopped job keeps this notice until the trainer actually
        unwinds. The job list the user is looking at drops the task as soon as the stop
        is sent, which left the notice apparently outliving its own job. A requested
        stop therefore says so rather than still claiming to be training — the wait is
        real, and on CPU it is as slow as the figure quoted here.
        """
        if not self._expects_gpu():
            return []
        notices: List[Dict[str, Any]] = []
        for job in running:
            if not job.device_warning:
                continue
            label = (job.config or {}).get("model_name") or job.job_id[:8]
            if job.cancel_requested:
                message = (
                    f"Training '{label}' is stopping — it finishes the epoch in progress first. "
                    "That epoch is running on the CPU, roughly 10-50x slower than GPU, so the "
                    "stop can take a while to land."
                )
            else:
                message = (
                    f"Training '{label}' is running on the CPU — roughly 10-50x slower than "
                    "GPU. This deployment expects a GPU, so check that the container still "
                    "has access to it."
                )
            notices.append(
                {
                    "key": f"training_cpu_only_{job.job_id}",
                    "level": "warning",
                    "message": message,
                }
            )
        return notices
