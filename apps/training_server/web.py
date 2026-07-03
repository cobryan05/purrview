"""Web API for Training Server"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import tarfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from .manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


# ─── Request / Response models ───────────────────────────────────────────────


class ManifestEntry(BaseModel):
    image_uuid: str
    filename: str
    status: str  # "present" | "absent"


class SubmitJobPayload(BaseModel):
    """Payload for POST /api/training/jobs.

    ``trainer_type`` is required; all other fields are optional.
    Extra fields are forwarded to the trainer config.
    """

    model_config = ConfigDict(extra="allow")

    trainer_type: str
    label_uuid: str = ""
    model_name: str = ""
    architecture: str = ""
    class_names: List[str] = Field(default_factory=lambda: ["absent", "present"])
    hyperparams: Dict[str, Any] = Field(default_factory=dict)
    # Flattened hyperparams also accepted at top level for backward compat.
    max_epochs: int = 50
    batch_size: int = 32
    early_stopping_patience: int = 10
    manifest: List[ManifestEntry] = Field(default_factory=list)


class JobInfo(BaseModel):
    job_id: str
    status: str
    trainer_type: str
    model_name: str
    created_at: str
    queue_position: int
    result: Optional[Dict[str, Any]] = None


class SubmitJobResponse(BaseModel):
    job_id: str
    status: str
    queue_position: int


class ListJobsResponse(BaseModel):
    jobs: List[JobInfo]


class CancelJobResponse(BaseModel):
    cancelled: bool
    message: str


class MissingImagesResponse(BaseModel):
    uuids: List[str]


class IngestImagesResponse(BaseModel):
    received: int
    cached_total: int


class StartJobResponse(BaseModel):
    status: str
    missing: List[str] = Field(default_factory=list)


class RestartJobResponse(BaseModel):
    job_id: str
    status: str


def _job_to_info(job) -> JobInfo:
    d = job.to_dict()
    return JobInfo(**d)


class WebApp:
    def __init__(self, app_name: str, manager: Manager):
        self._manager = manager
        self._app_name = app_name
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)
        self._register_routes()

    def app(self) -> FastAPI:
        return self._app

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        self._manager.start()
        try:
            yield
        finally:
            await self._manager.shutdown()

    def _register_routes(self) -> None:
        @self._app.post(
            "/api/training/jobs",
            operation_id="submit_training_job",
            response_model=SubmitJobResponse,
            tags=["training"],
        )
        async def submit_job(payload: SubmitJobPayload) -> SubmitJobResponse:
            manifest = [e.model_dump() for e in payload.manifest]
            # Build config dict from payload, merging nested hyperparams with
            # top-level hyperparams fields (nested take precedence).
            # Load-bearing: task runners smuggle bundle-target identity keys
            # (role, model_config_uuid, model_config_node_uuid, label_uuids)
            # inside hyperparams — this update() must keep running after
            # model_dump(), or sidecar.json loses bundle identity. Pinned by
            # test_submit_job_endpoint_merges_hyperparams_into_trainer_config.
            config = payload.model_dump(exclude={"manifest", "hyperparams"})
            config.update(payload.hyperparams)
            job = await self._manager.submit_job(config, manifest=manifest)
            return SubmitJobResponse(
                job_id=job.job_id,
                status=job.status,
                queue_position=job.queue_position,
            )

        @self._app.get(
            "/api/training/jobs",
            operation_id="list_training_jobs",
            response_model=ListJobsResponse,
            tags=["training"],
        )
        async def list_jobs() -> ListJobsResponse:
            jobs = self._manager.list_jobs()
            return ListJobsResponse(jobs=[_job_to_info(j) for j in jobs])

        @self._app.get(
            "/api/training/jobs/{job_id}",
            operation_id="get_training_job",
            response_model=JobInfo,
            tags=["training"],
        )
        async def get_job(job_id: str):
            job = self._manager.get_job(job_id)
            if not job:
                return JSONResponse({"detail": "Job not found"}, status_code=404)
            return _job_to_info(job)

        @self._app.get(
            "/api/training/jobs/{job_id}/missing-images",
            operation_id="get_missing_images",
            response_model=MissingImagesResponse,
            tags=["training"],
        )
        async def get_missing_images(job_id: str):
            """Return image UUIDs from the job's manifest that are not yet cached."""
            missing = self._manager.get_missing_images(job_id)
            if missing is None:
                return JSONResponse({"detail": "Job not found"}, status_code=404)
            return MissingImagesResponse(uuids=missing)

        @self._app.post(
            "/api/training/jobs/{job_id}/images",
            operation_id="ingest_images",
            response_model=IngestImagesResponse,
            tags=["training"],
        )
        async def ingest_images(job_id: str, request: Request):
            """Accept a tar stream of images and extract them into the cache."""
            job = self._manager.get_job(job_id)
            if not job:
                return JSONResponse({"detail": "Job not found"}, status_code=404)
            if job.status not in ("awaiting_data", "queued"):
                return JSONResponse(
                    {"detail": f"Cannot ingest images for job in status: {job.status}"},
                    status_code=409,
                )
            tar_bytes = await request.body()
            result = self._manager.ingest_images(job_id, tar_bytes)
            if result is None:
                return JSONResponse({"detail": "Job not found"}, status_code=404)
            return IngestImagesResponse(**result)

        @self._app.post(
            "/api/training/jobs/{job_id}/start",
            operation_id="start_training_job",
            response_model=StartJobResponse,
            tags=["training"],
        )
        async def start_job(job_id: str):
            """Validate data completeness and transition job to 'queued'."""
            job = self._manager.get_job(job_id)
            if not job:
                return JSONResponse({"detail": "Job not found"}, status_code=404)
            ok, missing = self._manager.start_job(job_id)
            if not ok:
                return JSONResponse(
                    StartJobResponse(status=job.status, missing=missing).model_dump(),
                    status_code=409,
                )
            return StartJobResponse(status=job.status, missing=[])

        @self._app.get(
            "/api/training/jobs/{job_id}/artifact",
            operation_id="get_training_artifact",
            tags=["training"],
        )
        async def get_artifact(job_id: str):
            """Stream a tar of the trained model artifacts (ONNX + JSON).

            Returns 404 until the job is in 'done' status.
            """
            artifact_dir = self._manager.get_artifact_dir(job_id)
            if artifact_dir is None:
                job = self._manager.get_job(job_id)
                if job is None:
                    return JSONResponse({"detail": "Job not found"}, status_code=404)
                return JSONResponse(
                    {"detail": f"Artifact not available for job in status: {job.status}"},
                    status_code=404,
                )

            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:") as tf:
                for p in sorted(artifact_dir.iterdir()):
                    if p.is_file():
                        tf.add(str(p), arcname=p.name)
            return Response(
                content=buf.getvalue(),
                media_type="application/x-tar",
                headers={"Content-Disposition": f'attachment; filename="artifacts_{job_id}.tar"'},
            )

        @self._app.post(
            "/api/training/jobs/{job_id}/restart",
            operation_id="restart_training_job",
            response_model=RestartJobResponse,
            tags=["training"],
        )
        async def restart_job(job_id: str):
            """Re-queue an interrupted job. Only valid for jobs in 'interrupted' status."""
            job = self._manager.get_job(job_id)
            if not job:
                return JSONResponse({"detail": "Job not found"}, status_code=404)
            ok = self._manager.restart_job(job_id)
            if not ok:
                return JSONResponse(
                    {"detail": f"Cannot restart job in status: {job.status}"},
                    status_code=409,
                )
            return RestartJobResponse(job_id=job_id, status="queued")

        @self._app.delete(
            "/api/training/jobs/{job_id}",
            operation_id="cancel_training_job",
            response_model=CancelJobResponse,
            tags=["training"],
        )
        async def cancel_job(job_id: str):
            job = self._manager.get_job(job_id)
            if not job:
                return JSONResponse({"detail": "Job not found"}, status_code=404)
            cancelled = self._manager.cancel_job(job_id)
            if cancelled:
                return CancelJobResponse(cancelled=True, message="Job cancelled")
            return CancelJobResponse(
                cancelled=False,
                message=f"Cannot cancel job in status: {job.status}",
            )

        @self._app.get(
            "/api/training/jobs/{job_id}/stream",
            operation_id="stream_training_job",
            tags=["training"],
        )
        async def stream_job(job_id: str, request: Request):
            """SSE stream of progress events for a training job.

            Replays all past events on connect so clients can reconnect mid-job
            and still receive the full history. Closes automatically when the
            job reaches a terminal state (done / error / cancelled).
            """
            job = self._manager.get_job(job_id)
            if not job:
                return JSONResponse({"detail": "Job not found"}, status_code=404)

            async def event_generator():
                try:
                    async for event_type, data in job.stream():
                        if await request.is_disconnected():
                            break
                        yield {"event": event_type, "data": json.dumps(data)}
                except asyncio.CancelledError:
                    pass

            return EventSourceResponse(event_generator())

        @self._app.get(
            "/server-config",
            operation_id="server_config",
            response_class=JSONResponse,
        )
        async def server_config(request: Request):
            try:
                config = await self._manager.get_server_config()
                return JSONResponse({"status": "success", "config": config})
            except Exception as e:
                return JSONResponse({"status": "failure", "message": str(e)})

        @self._app.get(
            "/api/training/status",
            operation_id="training_status",
            response_class=JSONResponse,
        )
        async def training_status(request: Request):
            return JSONResponse({"status": "success", "message": "Training server running."})

        @self._app.get(
            "/status",
            operation_id="training_server_status",
            response_class=JSONResponse,
        )
        async def status(request: Request):
            return JSONResponse({"status": "ok", "service": "training_server"})
