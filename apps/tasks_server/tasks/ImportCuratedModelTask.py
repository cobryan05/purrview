"""ImportCuratedModelTask — download a curated `requires_conversion` .pt model
and convert it to ONNX via the training server's `convert_model` trainer.

Sibling to ImportModelTask (direct-ONNX curated downloads): this task handles
the other kind of curated entry — one that only has a public `.pt` URL (e.g.
an Ultralytics GitHub release asset), no maintainer-published ONNX export.

Flow:
  1. Download the `.pt` to a local temp file (same httpx pattern ImportModelTask
     already uses for its ONNX downloads).
  2. Submit a "convert_model" job to the training server and push the temp
     file's bytes as the job's source-model, reusing
     TrainingJobRunnerMixin._consume_sse/_handle_event for the submit -> push ->
     start -> SSE-poll sequence (same machinery ConvertModelTask drives).
  3. Override _save_artifact so the converted ONNX is validated IN MEMORY
     (validate_onnx_bytes, via write_onnx_with_sidecar) before ever touching
     `/models` — the same landing path used by every other import path
     (upload, curated direct-ONNX, ConvertModelTask). The mixin's default
     _save_artifact writes tar members straight to disk with no validation,
     which is fine for ConvertModelTask (arbitrary user upload, no
     pre-existing "curated" trust claim) but NOT acceptable here, where the
     target filename is explicitly the user's chosen model_name and an
     unparseable or truncated artifact must not clobber an existing model.

Unlike ConvertModelTask (whose source file is a pre-staged synchronous
upload), this task owns the temp source file's entire lifecycle: download it
in _run, always delete it in _deinit whether the run completed, errored, or
was cancelled.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import db_client
import httpx
import training_client
from training_client.api.training_api import TrainingApi
from training_client.models.submit_job_payload import SubmitJobPayload

from apps import APPS, APPS_CONFIG
from apps.helpers.apiCall import LONG_TIMEOUT, bind_default_timeout
from apps.helpers.consts import TaskAccent
from apps.helpers.curatedModels import get_curated_model
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.metadataUtils import validate_bare_model_name, write_onnx_with_sidecar

from ._inference_push import build_artifact_tar, push_artifact_tar_to_inference_server
from ._training_job_runner import TrainingJobRunnerMixin
from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

_MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/models"))
_DOWNLOAD_TIMEOUT = 300.0  # seconds; matches ImportModelTask's curated-ONNX download


@register_task()
class ImportCuratedModelTask(Task, TrainingJobRunnerMixin):
    # Model production: this is the run that makes a model.
    CARD_ACCENT = TaskAccent.VIOLET

    # Started from Configuration → Models → Import model. See ImportModelTask.
    SYSTEM_TASK = True

    # Conversion is generally quick, but reuse ConvertModelTask/TrainModelTask's
    # more tolerant heartbeat interval since it shares the same SSE-relay code path.
    HEARTBEAT_INTERVAL = 60

    async def _init(self, params: Dict[str, Any], resume_data: Optional[Dict[str, Any]]) -> None:
        # ── API client setup (mirrors ConvertModelTask._init) ────────────────
        db_cfg = APPS_CONFIG[APPS.DB_SERVER]
        db_base_url = f"http://{db_cfg.host}:{db_cfg.port}"
        self._db_api_client = bind_default_timeout(db_client.ApiClient(db_client.Configuration(host=db_base_url)))

        # Resolved through the manager, not APPS_CONFIG: Settings → Servers can
        # point training at another host, and only the manager tracks that override.
        self._training_base_url: str = self._manager.get_training_base_url()
        tc_config = training_client.Configuration(host=self._training_base_url)
        # Every call this client makes is to the training server, and the training

        # server does slow work — submitting a job with a large manifest, ingesting

        # images, converting a model. Bound as a whole rather than per call, so no

        # caller has to know about the client library's timeout parameter.

        self._api_client = bind_default_timeout(training_client.ApiClient(tc_config), timeout=LONG_TIMEOUT)

        # ── Resume from a previous run ────────────────────────────────────────
        self._job_id: Optional[str] = resume_data.get("job_id") if resume_data else None
        if self._job_id:
            self.log(f"Resuming conversion job {self._job_id}")
            self._model_name = str(resume_data.get("model_name", "model"))
            self._overwrite = bool(resume_data.get("overwrite", False))
            source_path_str = resume_data.get("source_path")
            self._source_path: Optional[Path] = Path(source_path_str) if source_path_str else None
            self._manifest: list[dict] = []
            return

        # ── Fresh run: params are raw/unvalidated — validate everything here ─
        curated_key = str(params.get("curated_key") or "")
        try:
            self._curated_entry = get_curated_model(curated_key)
        except KeyError as exc:
            raise ValueError(f"Unknown curated_key {curated_key!r}: {exc}") from exc

        if not self._curated_entry.get("requires_conversion"):
            raise ValueError(f"curated_key {curated_key!r} does not require conversion; use ImportModelTask instead")

        self._pt_url: str = self._curated_entry["pt_url"]
        self._display_name: str = self._curated_entry["display_name"]

        self._model_name = str(params.get("model_name") or "")
        validate_bare_model_name(self._model_name)

        self._overwrite: bool = bool(params.get("overwrite", False))

        target_path = _MODELS_ROOT / f"{self._model_name}.onnx"
        if target_path.exists() and not self._overwrite:
            raise FileExistsError(f"Model '{self._model_name}' already exists; pass overwrite=True to replace it")

        self._job_id = None
        self._manifest = []
        self._source_path = None

    async def _run(self) -> Dict[str, Any]:
        if self._cancel_flag.is_set():
            return {"cancelled": True}

        # ── Step 0: download the .pt to a local temp file (fresh runs only;
        # a resumed run already has _source_path from resume_data, and the
        # source-model bytes are already pushed to the remote job) ──────────
        if self._source_path is None:
            self.log(f"Downloading {self._display_name} from {self._pt_url}...")
            fd, tmp_name = tempfile.mkstemp(suffix=".pt", prefix="curated_import_")
            os.close(fd)
            self._source_path = Path(tmp_name)
            try:
                async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
                    resp = await client.get(self._pt_url)
                    resp.raise_for_status()
                    pt_bytes = resp.content
            except Exception:
                # Fresh download failed — don't leave an empty temp file around.
                self._source_path.unlink(missing_ok=True)
                self._source_path = None
                raise
            self._source_path.write_bytes(pt_bytes)
            self.log(f"Downloaded {len(pt_bytes)} bytes to {self._source_path}")

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
            # state. Conversions are cheap; simplest correct behavior is to
            # re-submit from scratch (the .pt is still on local disk).
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

        # Persist immediately so cancel/resume can reach the remote job and the
        # local temp source file survives a task restart.
        self._resume_data["job_id"] = self._job_id
        self._resume_data["model_name"] = self._model_name
        self._resume_data["overwrite"] = self._overwrite
        self._resume_data["source_path"] = str(self._source_path)
        self._data_ready_flag.set()

        if self._cancel_flag.is_set():
            await self._cancel_remote_job(self._job_id)
            return {"cancelled": True, "job_id": self._job_id}

        # ── Step 2: Push the downloaded source-model bytes ───────────────────
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

        # ── Step 4: Relay SSE (mixin's "done" handler calls our overridden
        # _save_artifact, which validates in memory before writing) ──────────
        await self.heartbeat(force=True)
        return await self._consume_sse(self._job_id)

    async def _save_artifact(self, job_id: str) -> None:
        """Download the artifact tar, validate the ONNX bytes IN MEMORY, and land
        it via the same write_onnx_with_sidecar path every other import uses.

        Overrides TrainingJobRunnerMixin._save_artifact (which writes tar members
        straight to disk with no validation) — a bad/truncated conversion result
        must not leave an unparseable orphan or clobber an existing good model.
        """
        artifact_url = f"{self._training_base_url}/api/training/jobs/{job_id}/artifact"
        self.log("Downloading converted model artifact...")

        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.get(artifact_url)
            r.raise_for_status()
            tar_bytes = r.content

        onnx_bytes: Optional[bytes] = None
        sidecar_extra: Dict[str, Any] = {}
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                fobj = tf.extractfile(member)
                if not fobj:
                    continue
                if name.endswith(".onnx"):
                    onnx_bytes = fobj.read()
                elif name.endswith(".json"):
                    import json as _json

                    try:
                        sidecar_extra = _json.loads(fobj.read())
                    except Exception:
                        logger.warning("Could not parse conversion sidecar %s; ignoring.", name, exc_info=True)

        if onnx_bytes is None:
            raise RuntimeError(f"Conversion job {job_id} artifact did not contain an .onnx file")

        model_path = _MODELS_ROOT / f"{self._model_name}.onnx"
        metadata = await asyncio.to_thread(
            write_onnx_with_sidecar,
            str(model_path),
            onnx_bytes,
            overwrite=self._overwrite,
            extra_metadata=sidecar_extra,
        )
        self.log(f"Saved converted model artifact: {model_path} ({len(metadata.get('classes', []))} classes)")

        await self._push_to_inference_server(model_path, onnx_bytes)

    async def _push_to_inference_server(self, model_path: Path, onnx_bytes: bytes) -> None:
        """Push the same .onnx + sidecar to inference_server (see _inference_push).

        Without this, a split-container inference_server would never learn about a
        curated model imported here.
        """
        self.log("Pushing model to inference server...")
        sidecar_bytes = await asyncio.to_thread(model_path.with_suffix(".json").read_bytes)
        tar_bytes = build_artifact_tar(
            {
                f"{self._model_name}.onnx": onnx_bytes,
                f"{self._model_name}.json": sidecar_bytes,
            }
        )
        files = await push_artifact_tar_to_inference_server(self._manager.get_inference_api_client(), tar_bytes)
        self.log(f"Inference server updated: {files}")

    async def _deinit(self) -> None:
        # Always clean up the local temp .pt file — whether the run completed,
        # errored, or was cancelled. Unlike ConvertModelTask's staged upload
        # (which lives under CACHE_DIR and is single-use by convention), this
        # is a raw tempfile we created ourselves in _run.
        source_path = getattr(self, "_source_path", None)
        if source_path is None:
            return
        try:
            await asyncio.to_thread(Path(source_path).unlink, True)
        except Exception:
            logger.warning("Could not delete temp source model %s", source_path, exc_info=True)

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        from apps.helpers.curatedModels import CURATED_BASE_MODELS

        conversion_entries = [e for e in CURATED_BASE_MODELS if e.get("requires_conversion")]
        return {
            "curated_key": {
                "type": "select",
                "required": True,
                "label": "Curated Model (via training server)",
                "options": [{"value": e["key"], "label": e["display_name"]} for e in conversion_entries],
                "help": "Which curated pretrained base model to download and convert",
            },
            "model_name": {
                "type": "str",
                "required": True,
                "label": "Model Name",
                "help": "Target filename to save as (without extension) in /models/",
            },
            "overwrite": {
                "type": "boolean",
                "required": False,
                "default": False,
                "label": "Overwrite",
                "help": "Replace an existing model + sidecar of the same name",
            },
        }
