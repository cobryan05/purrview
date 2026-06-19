"""Manager for Training Server"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from apps import APPS

from .job_queue import Job, JobQueue

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

_DEFAULT_FS_ROOT = "/app/image_datasets"


class Manager:
    def __init__(self):
        worker_count = int(os.environ.get("TRAINING_WORKER_COUNT", "1"))
        self._queue = JobQueue(worker_count=worker_count)
        self._fs_root: str = os.environ.get("TRAINING_FS_ROOT", _DEFAULT_FS_ROOT)
        self._shared_fs_available: bool = False
        self._config: Dict[str, Any] = {"service": APPS.TRAINING_SERVER}

    def start(self) -> None:
        # Auto-detect shared filesystem availability.
        self._shared_fs_available = Path(self._fs_root).exists()
        if self._shared_fs_available:
            logger.info("Shared filesystem detected at %s", self._fs_root)
        else:
            logger.info(
                "Shared filesystem not found at %s — HTTP image fetching will be default",
                self._fs_root,
            )

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
        return {
            **self._config,
            "shared_fs_available": self._shared_fs_available,
            "fs_root": self._fs_root,
            "default_image_source": "filesystem" if self._shared_fs_available else "http",
        }

    async def submit_job(self, config: Dict[str, Any]) -> Job:
        if "image_source" not in config:
            config = {**config, "image_source": "filesystem" if self._shared_fs_available else "http"}
        # Derive trainer_type from model_type if not explicitly set.
        if "trainer_type" not in config and "model_type" in config:
            config = {**config, "trainer_type": config["model_type"]}
        return await self._queue.submit(config)

    def cancel_job(self, job_id: str) -> bool:
        return self._queue.cancel(job_id)

    def list_jobs(self) -> list[Job]:
        return self._queue.list_jobs()

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._queue.get_job(job_id)
