"""Manager for Training Server"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from apps import APPS, FILES_ROOT
from apps.helpers.model_type_map import resolve_trainer_type

from .image_cache import ImageCache
from .job_queue import Job, JobQueue

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

_DEFAULT_ARTIFACTS_DIR = "/training-artifacts"
_DEFAULT_EVICTION_SWEEP_SEC = 3600  # background age/size sweep interval


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
