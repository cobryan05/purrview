"""Web API for Training Server"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from sse_starlette.sse import EventSourceResponse

from .manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


# ─── Request / Response models ───────────────────────────────────────────────


class SubmitJobPayload(BaseModel):
    """Payload for POST /api/training/jobs.

    ``trainer_type`` is required; all other fields are optional (trainer-specific
    extras are accepted via ``extra="allow"``).
    """

    model_config = ConfigDict(extra="allow")

    trainer_type: str
    label_uuid: str = ""
    model_name: str = ""
    max_epochs: int = 50
    image_source: Optional[str] = None


class JobInfo(BaseModel):
    job_id: str
    status: str
    trainer_type: str
    model_name: str
    created_at: str
    queue_position: int
    result: Optional[dict[str, Any]] = None


class SubmitJobResponse(BaseModel):
    job_id: str
    status: str
    queue_position: int


class ListJobsResponse(BaseModel):
    jobs: list[JobInfo]


class CancelJobResponse(BaseModel):
    cancelled: bool
    message: str


# ─── WebApp ───────────────────────────────────────────────────────────────────


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
        yield

    def _register_routes(self) -> None:
        @self._app.post(
            "/api/training/jobs",
            operation_id="submit_training_job",
            response_model=SubmitJobResponse,
            tags=["training"],
        )
        async def submit_job(payload: SubmitJobPayload) -> SubmitJobResponse:
            config = payload.model_dump()
            job = await self._manager.submit_job(config)
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


def _job_to_info(job) -> JobInfo:
    d = job.to_dict()
    return JobInfo(
        job_id=d["job_id"],
        status=d["status"],
        trainer_type=d["trainer_type"],
        model_name=d["model_name"],
        created_at=d["created_at"],
        queue_position=d["queue_position"],
        result=d["result"],
    )
