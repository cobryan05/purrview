"""Manager for Training Server"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from apps import APPS
from apps.helpers.model_type_map import MODEL_TYPE_TO_TRAINER_TYPE

from .image_cache import ImageCache
from .job_queue import Job, JobQueue

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

_DEFAULT_ARTIFACTS_DIR = "/training-artifacts"


class Manager:
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        artifacts_dir: Optional[str] = None,
    ):
        worker_count = int(os.environ.get("TRAINING_WORKER_COUNT", "1"))
        self._queue = JobQueue(worker_count=worker_count)
        self._cache = ImageCache(cache_dir=cache_dir)
        self._artifacts_dir = Path(
            artifacts_dir or os.environ.get("TRAINING_ARTIFACTS_DIR", _DEFAULT_ARTIFACTS_DIR)
        )
        self._config: Dict[str, Any] = {"service": APPS.TRAINING_SERVER}

    def start(self) -> None:
        self._cache.ensure_dir()
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Import trainer modules to trigger @register_trainer decorators.
        from .trainers import TRAINER_REGISTRY  # noqa: F401
        from .trainers import binary_image_classifier  # noqa: F401
        from .trainers import crop_classifier_trainer  # noqa: F401
        from .trainers import object_detection_trainer  # noqa: F401

        for name, trainer in TRAINER_REGISTRY.items():
            self._queue.register_trainer(name, trainer)
            logger.info("Registered trainer: %s", name)

        self._queue.start()
        logger.info("Training server started (worker_count=%d)", self._queue._worker_count)

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

        # Derive trainer_type from model_type when absent.
        if "trainer_type" not in config:
            model_type = config.get("model_type", "")
            trainer_type = MODEL_TYPE_TO_TRAINER_TYPE.get(model_type, model_type)
            config["trainer_type"] = trainer_type

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

        # Inject cache_dir and artifact_dir into config so the trainer can use them.
        self._cache.ensure_dir()
        job.config = {
            **config,
            "cache_dir": self._cache.cache_dir,
            "artifact_dir": str(artifact_dir),
        }
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

    def list_jobs(self) -> list[Job]:
        return self._queue.list_jobs()

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._queue.get_job(job_id)
