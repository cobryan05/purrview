"""ConvertModelTask — drive a model-conversion job on the training server.

A raw (non-ONNX) model file is uploaded synchronously to inference_server and
staged to disk (see apps.helpers.modelStaging / inference_server.manager.stage_source_model).
This task then, in the background:
  1. Submits a "convert_model" job to the training server (no manifest needed —
     conversion doesn't touch training images).
  2. Pushes the staged source-model bytes to the job (raw POST, like
     _training_job_runner._push_images uses httpx directly for binary bodies).
  3. Starts the job and relays its SSE stream, reusing
     TrainingJobRunnerMixin._consume_sse/_handle_event/_save_artifact to land the
     resulting ONNX (+ sidecar) in $MODELS_ROOT — the same "done" handling used by
     TrainModelTask.

Unlike TrainModelTask, this task never calls _run_one_job() (that method also
pushes training images, which conversion doesn't need); it drives the
submit -> push-source-model -> start -> SSE sequence itself, delegating only the
SSE relay + artifact save to the mixin.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import db_client
import httpx
import training_client
from training_client.api.training_api import TrainingApi
from training_client.models.submit_job_payload import SubmitJobPayload

from apps import APPS, APPS_CONFIG
from apps.helpers.metadataUtils import validate_bare_model_name
from apps.helpers.modelStaging import staged_model_path

from ._training_job_runner import TrainingJobRunnerMixin
from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

_MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/models"))


@register_task()
class ConvertModelTask(Task, TrainingJobRunnerMixin):
    # Conversion is generally quick, but reuse TrainModelTask's more tolerant
    # heartbeat interval since it shares the same SSE-relay code path.
    HEARTBEAT_INTERVAL = 60

    async def _init(self, params: Dict[str, Any], resume_data: Optional[Dict[str, Any]]) -> None:
        # ── API client setup (mirrors TrainModelTask._init) ─────────────────
        db_cfg = APPS_CONFIG[APPS.DB_SERVER]
        db_base_url = f"http://{db_cfg.host}:{db_cfg.port}"
        self._db_api_client = db_client.ApiClient(db_client.Configuration(host=db_base_url))

        training_cfg = APPS_CONFIG[APPS.TRAINING_SERVER]
        self._training_base_url: str = f"http://{training_cfg.host}:{training_cfg.port}"
        tc_config = training_client.Configuration(host=self._training_base_url)
        self._api_client = training_client.ApiClient(tc_config)

        # ── Resume from a previous run ───────────────────────────────────────
        self._job_id: Optional[str] = resume_data.get("job_id") if resume_data else None
        if self._job_id:
            self.log(f"Resuming conversion job {self._job_id}")
            self._model_name = str(resume_data.get("model_name", "model"))
            self._staging_id = str(resume_data.get("staging_id", ""))
            self._manifest: list[dict] = []
            self._source_path = staged_model_path(self._staging_id) if self._staging_id else None
            return

        # ── Fresh run: params are raw/unvalidated — validate everything here ─
        self._staging_id = str(params.get("staging_id") or "")
        if not self._staging_id:
            raise ValueError("staging_id is required")
        self._source_path = staged_model_path(self._staging_id)
        if not self._source_path.exists():
            raise ValueError(f"Staged source model not found for staging_id {self._staging_id!r}")

        self._model_name = str(params.get("model_name") or "")
        validate_bare_model_name(self._model_name)

        self._overwrite: bool = bool(params.get("overwrite", False))

        target_path = _MODELS_ROOT / f"{self._model_name}.onnx"
        if target_path.exists() and not self._overwrite:
            raise FileExistsError(f"Model '{self._model_name}' already exists; pass overwrite=True to replace it")

        self._job_id = None
        self._manifest = []

    async def _run(self) -> Dict[str, Any]:
        if self._job_id:
            status = await self._check_remote_job_status(self._job_id)
            self.log(f"Resuming conversion job {self._job_id} (remote status: {status})")
            await self.heartbeat(force=True)

            if status == "done":
                await self._save_artifact(self._job_id)
                return {"job_id": self._job_id, "model_name": self._model_name}

            if status in ("running", "queued", "awaiting_data"):
                return await self._consume_sse(self._job_id)

            if status in ("error", "cancelled"):
                raise RuntimeError(f"Conversion job {self._job_id} is in terminal state: {status}")

            # "not_found" (404) or unknown — server restarted without persisted
            # state. Conversions are short/cheap; simplest correct behavior is
            # to re-submit from scratch.
            self.log(f"Job {self._job_id} not found on training server; resubmitting from scratch")
            self._job_id = None

        # ── Step 1: Submit job ────────────────────────────────────────────────
        payload = SubmitJobPayload(
            trainer_type="convert_model",
            model_name=self._model_name,
            manifest=[],
            hyperparams={"overwrite": self._overwrite},
        )
        resp = await asyncio.to_thread(
            TrainingApi(self._api_client).submit_training_job,
            payload,
        )
        self._job_id = resp.job_id
        self.log(f"Submitted conversion job {self._job_id} (model_name={self._model_name})")

        # Persist immediately so cancel can reach the remote job.
        self._resume_data["job_id"] = self._job_id
        self._resume_data["model_name"] = self._model_name
        self._resume_data["staging_id"] = self._staging_id
        self._data_ready_flag.set()

        if self._cancel_flag.is_set():
            await self._cancel_remote_job(self._job_id)
            return {"cancelled": True, "job_id": self._job_id}

        # ── Step 2: Push the staged source-model bytes ───────────────────────
        source_bytes = await asyncio.to_thread(self._source_path.read_bytes)
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(
                f"{self._training_base_url}/api/training/jobs/{self._job_id}/source-model",
                content=source_bytes,
                headers={"Content-Type": "application/octet-stream"},
            )
            r.raise_for_status()
        self.log(f"Pushed {len(source_bytes)} source-model bytes to job {self._job_id}")

        if self._cancel_flag.is_set():
            await self._cancel_remote_job(self._job_id)
            return {"cancelled": True, "job_id": self._job_id}

        # ── Step 3: Start the job ─────────────────────────────────────────────
        await asyncio.to_thread(
            TrainingApi(self._api_client).start_training_job,
            self._job_id,
        )
        self.log("Conversion job started, waiting for completion...")

        # ── Step 4: Relay SSE (mixin's "done" handler calls _save_artifact) ──
        await self.heartbeat(force=True)
        return await self._consume_sse(self._job_id)

    async def _deinit(self) -> None:
        # The staged upload is single-use — clean it up whether the run
        # completed, errored, or was cancelled.
        source_path = getattr(self, "_source_path", None)
        if source_path is None:
            return
        try:
            await asyncio.to_thread(Path(source_path).unlink, True)
        except Exception:
            logger.warning("Could not delete staged source model %s", source_path, exc_info=True)

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "staging_id": {
                "type": "str",
                "required": True,
                "label": "Staging ID",
                "help": "ID returned by the staging upload of the raw source model file",
            },
            "model_name": {
                "type": "str",
                "required": True,
                "label": "Model Name",
                "help": "Target filename to save the converted ONNX model as (without extension) in /models/",
            },
            "overwrite": {
                "type": "boolean",
                "required": False,
                "default": False,
                "label": "Overwrite",
                "help": "Replace an existing model + sidecar of the same name",
            },
        }
