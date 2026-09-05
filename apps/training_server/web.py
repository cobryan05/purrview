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
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from apps.helpers.landingPage import action, render_landing_page
from apps.helpers.logUtils import LOG_LEVEL

from .manager import Manager
from .trainers import TRAINER_REGISTRY

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


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
    # These stay Optional/None-default so an omitted field falls through to
    # each trainer's own config.get(key, default) — trainers disagree on the
    # right default (e.g. object_detection batch_size=16 vs classifier=32),
    # and a fixed Pydantic default here would silently override all of them.
    max_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    early_stopping_patience: Optional[int] = None
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


class HyperparamDefaultsResponse(BaseModel):
    trainer_type: str
    defaults: Dict[str, Any]


class CancelJobResponse(BaseModel):
    cancelled: bool
    message: str


class MissingImagesResponse(BaseModel):
    uuids: List[str]


class IngestImagesResponse(BaseModel):
    received: int
    cached_total: int


class LocalManifestEntry(BaseModel):
    image_uuid: str
    filename: str


class LocalIngestPayload(BaseModel):
    entries: List[LocalManifestEntry]


class LocalIngestResponse(BaseModel):
    received: int
    cached_total: int
    linked: int
    copied: int
    not_found: List[str] = Field(default_factory=list)


class IngestSourceModelResponse(BaseModel):
    received: int


class StartJobResponse(BaseModel):
    status: str
    missing: List[str] = Field(default_factory=list)


class RestartJobResponse(BaseModel):
    job_id: str
    status: str


class CacheClearResponse(BaseModel):
    removed: int
    freed_bytes: int
    kept_pinned: int


class CacheEvictResponse(BaseModel):
    removed: int
    freed_bytes: int
    files: int
    bytes: int


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
            # Load-bearing: task runners smuggle training-target identity
            # keys (role, model_config_uuid, model_config_node_uuid,
            # label_uuids) inside hyperparams — this update() must keep running
            # after model_dump(), or sidecar.json loses that identity. Pinned by
            # test_submit_job_endpoint_merges_hyperparams_into_trainer_config.
            config = payload.model_dump(exclude={"manifest", "hyperparams"}, exclude_none=True)
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
            "/api/training/trainers/{trainer_type}/hyperparams",
            operation_id="get_trainer_hyperparam_defaults",
            response_model=HyperparamDefaultsResponse,
            tags=["training"],
        )
        async def get_trainer_hyperparam_defaults(trainer_type: str):
            trainer = TRAINER_REGISTRY.get(trainer_type)
            if trainer is None:
                return JSONResponse({"detail": f"Unknown trainer_type {trainer_type!r}"}, status_code=404)
            return HyperparamDefaultsResponse(trainer_type=trainer_type, defaults=trainer.HYPERPARAM_DEFAULTS)

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
            # Tar extraction does synchronous disk I/O (write + fsync per file) —
            # run it off the event loop so a large push doesn't stall every other
            # request this process is serving.
            result = await asyncio.to_thread(self._manager.ingest_images, job_id, tar_bytes)
            if result is None:
                return JSONResponse({"detail": "Job not found"}, status_code=404)
            return IngestImagesResponse(**result)

        @self._app.post(
            "/api/training/jobs/{job_id}/images/local",
            operation_id="ingest_local_images",
            response_model=LocalIngestResponse,
            tags=["training"],
        )
        async def ingest_local_images(job_id: str, payload: LocalIngestPayload):
            """Zero-copy ingest: populate the cache from local filenames under
            FILES_ROOT instead of receiving image bytes over HTTP.

            Only succeeds for UUIDs whose source file is actually reachable from
            this process (e.g. pusher and training_server share a filesystem).
            Anything else comes back in ``not_found`` for the caller to push via
            the tar-upload endpoint instead.
            """
            job = self._manager.get_job(job_id)
            if not job:
                return JSONResponse({"detail": "Job not found"}, status_code=404)
            if job.status not in ("awaiting_data", "queued"):
                return JSONResponse(
                    {"detail": f"Cannot ingest images for job in status: {job.status}"},
                    status_code=409,
                )
            entries = [(e.image_uuid, e.filename) for e in payload.entries]
            result = await asyncio.to_thread(self._manager.ingest_local, job_id, entries)
            if result is None:
                return JSONResponse({"detail": "Job not found"}, status_code=404)
            return LocalIngestResponse(**result)

        @self._app.post(
            "/api/training/jobs/{job_id}/source-model",
            operation_id="ingest_source_model",
            response_model=IngestSourceModelResponse,
            tags=["training"],
        )
        async def ingest_source_model(job_id: str, request: Request):
            """Accept the raw bytes of a source .pt checkpoint for a convert_model job."""
            job = self._manager.get_job(job_id)
            if not job:
                return JSONResponse({"detail": "Job not found"}, status_code=404)
            if job.status not in ("awaiting_data", "queued"):
                return JSONResponse(
                    {"detail": f"Cannot ingest source model for job in status: {job.status}"},
                    status_code=409,
                )
            data = await request.body()
            result = await asyncio.to_thread(self._manager.ingest_source_model, job_id, data)
            if result is None:
                return JSONResponse({"detail": "Job not found"}, status_code=404)
            return IngestSourceModelResponse(received=result["received"])

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
            return JSONResponse(self._manager.get_status())

        @self._app.get(
            "/api/training/info",
            operation_id="training_server_info",
            response_class=JSONResponse,
            tags=["training"],
        )
        async def info(request: Request):
            """Cache/artifact/server detail for the landing page.

            Separate from /status because it walks the cache and artifact
            directories — off the event loop, and never on the status poll path.
            """
            return JSONResponse(await asyncio.to_thread(self._manager.get_info))

        @self._app.post(
            "/api/training/cache/evict",
            operation_id="evict_training_cache",
            response_model=CacheEvictResponse,
            tags=["training"],
        )
        async def evict_cache() -> CacheEvictResponse:
            """Run the age/size eviction sweep now instead of waiting for the timer."""
            return CacheEvictResponse(**await asyncio.to_thread(self._manager.evict_cache))

        @self._app.delete(
            "/api/training/cache",
            operation_id="clear_training_cache",
            response_model=CacheClearResponse,
            tags=["training"],
        )
        async def clear_cache(include_pinned: bool = False) -> CacheClearResponse:
            """Delete cached images.

            Images pinned by live jobs are kept unless ``include_pinned`` is
            set — dropping them leaves a queued job unable to run without a
            fresh upload of every image it needs.
            """
            result = await asyncio.to_thread(self._manager.clear_cache, include_pinned)
            return CacheClearResponse(**result)

        @self._app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def index() -> HTMLResponse:
            """Operator landing page — this service is often the only thing on its host."""
            return HTMLResponse(
                render_landing_page(
                    title="PurrView Training Server",
                    subtitle="model training jobs + image cache",
                    status_url="/status",
                    info_url="/api/training/info",
                    actions=[
                        action(
                            "Run eviction sweep",
                            "/api/training/cache/evict",
                            description="Apply the age and size limits now.",
                        ),
                        action(
                            "Clear unpinned cache",
                            "/api/training/cache",
                            method="DELETE",
                            danger=True,
                            confirm="Delete all cached images that no live job needs?",
                            description="Keeps images pinned by awaiting/queued/running jobs.",
                        ),
                        action(
                            "Clear entire cache",
                            "/api/training/cache?include_pinned=true",
                            method="DELETE",
                            danger=True,
                            confirm=(
                                "Delete EVERY cached image, including images pinned by live jobs? "
                                "Those jobs will need their images uploaded again."
                            ),
                            description="Includes pinned images — live jobs will need a re-upload.",
                        ),
                    ],
                    links=[
                        ("API docs", "/docs"),
                        ("Status JSON", "/status"),
                        ("Jobs JSON", "/api/training/jobs"),
                    ],
                )
            )
