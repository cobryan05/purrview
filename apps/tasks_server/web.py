"""Web API for Tasks Server"""

import asyncio
import html
import json
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, Form, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import Field

from apps.helpers.consts import JsonValues
from apps.helpers.db.types import TaskConfigRead, TaskResult
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.types import Payload, StatusResponse
from apps.tasks_server.manager import Manager
from apps.tasks_server.types import TaskInfo, TaskInfoRead

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


class CreateTaskConfigPayload(Payload):
    """Request model for creating a new task configuration."""

    name: str
    typename: str
    params: dict
    # None = "no opinion, use the task type's DEFAULT_PERSISTENT". A caller that
    # cares still sends true/false and is obeyed; omitting it is how a client
    # says the task type should decide (a watcher defaults persistent, batch
    # work does not).
    persistent: Optional[bool] = None
    is_temporary: bool = False


class CreateTaskConfigResponse(StatusResponse):
    config_uuid: Optional[str] = None


class DeleteTasksPayload(Payload):
    task_uuids: List[str]


class DeleteTasksResponse(StatusResponse):
    task_uuids: List[str] = Field(default_factory=list)


class DeleteTaskConfigsPayload(Payload):
    config_uuids: List[str]


class DeleteTaskConfigsResponse(StatusResponse):
    config_uuids: List[str] = Field(default_factory=list)


class GetTaskConfigsPayload(Payload):
    config_uuids: Optional[Union[List[str], str]] = None
    exclude_temporary: bool = False


class GetTaskConfigResponse(StatusResponse):
    configs: Dict[str, TaskConfigRead] = Field(default_factory=dict)


class ListTaskTypesResponse(StatusResponse):
    types: List[str] = Field(default_factory=list)
    # typename -> TaskAccent value, only for types that claim one. Rides along
    # with the type list because it is the same static, already-cached-by-the-
    # frontend fact about a task type; a second endpoint would mean a second
    # request on every page that draws a task card.
    accents: dict[str, str] = Field(default_factory=dict)
    # Subset of `types` that dedicated UI starts and the "new task config" form
    # therefore hides by default. Rides along for the same reason `accents` does.
    system_types: List[str] = Field(default_factory=list)


class PauseTasksPayload(Payload):
    task_uuids: List[str]


class PauseTasksResponse(StatusResponse):
    task_uuids: List[str] = Field(default_factory=list)


class ResumeTasksPayload(Payload):
    task_uuids: List[str]


class ResumeTasksResponse(StatusResponse):
    task_uuids: List[str] = Field(default_factory=list)


class CancelTasksPayload(Payload):
    task_uuids: List[str]


class CancelTasksResponse(StatusResponse):
    # `task_uuids` keeps its original meaning — the instances confirmed stopped — so
    # existing callers are unaffected. The other two are what let the UI tell a stop
    # that is still winding down apart from one that had nothing to stop; both used to
    # be indistinguishable from success.
    task_uuids: List[str] = Field(default_factory=list)
    stopping: List[str] = Field(default_factory=list)
    not_running: List[str] = Field(default_factory=list)


class PauseInPlacePayload(Payload):
    task_uuids: List[str]


class PauseInPlaceResponse(StatusResponse):
    task_uuids: List[str] = Field(default_factory=list)


class ResumeInPlacePayload(Payload):
    task_uuids: List[str]


class ResumeInPlaceResponse(StatusResponse):
    task_uuids: List[str] = Field(default_factory=list)


class StartTasksPayload(Payload):
    config_uuid: str


class StartTasksResponse(StatusResponse):
    task_uuid: Optional[str] = None


class TasksTypeSchemaPayload(Payload):
    task_typenames: List[str]


class TasksTypeSchemaResponse(StatusResponse):
    schemas: Dict[str, dict] = Field(default_factory=dict)


class TasksStatusSchemaPayload(Payload):
    task_typenames: List[str]


class TasksStatusSchemaResponse(StatusResponse):
    schemas: Dict[str, List[dict]] = Field(default_factory=dict)


class TasksStatusDataPayload(Payload):
    task_uuids: List[str]


class TasksStatusDataResponse(StatusResponse):
    status_data: Dict[str, dict] = Field(default_factory=dict)


class TaskLivePreviewResponse(StatusResponse):
    #: Free-form because its shape is the task type's business, not the transport's —
    #: the same reason status_data is a bare dict. Absent when the task type has no
    #: live view.
    preview: Optional[dict] = None


class TaskCollectorDiagnosticsResponse(StatusResponse):
    #: `{stats, calibration, notices}` — same free-form-dict reasoning as
    #: `preview` above. Absent when the task isn't running, has no such
    #: collector attached, or is a task type with nothing per-collector to show.
    diagnostics: Optional[dict] = None


class GrabSourceFrameResponse(StatusResponse):
    #: Bare base64 JPEG — the bytes an <img> src wants, not the inference API's envelope.
    frame: Optional[str] = None
    width: int = 0
    height: int = 0
    #: "watcher" when reused from a running watcher's latest analysed frame, "provider"
    #: when the camera was opened for this request. Surfaced because the two differ in a
    #: way that matters: a watcher's frame is downscaled to its analysis width, so the
    #: still is lower resolution than the camera's own.
    source: Optional[str] = None


class TaskCommandPayload(Payload):
    task_uuid: str
    command: str
    params: dict = Field(default_factory=dict)


class TaskCommandResponse(StatusResponse):
    pass


class TasksResultPayload(Payload):
    task_uuids: List[str]


class TasksResultResponse(StatusResponse):
    results: dict[str, TaskResult] = Field(default_factory=dict)


class TasksInfoPayload(Payload):
    task_uuids: Optional[List[str]] = None


class TasksInfoResponse(StatusResponse):
    tasks: dict[str, TaskInfoRead] = Field(default_factory=dict)


class UpdateTaskConfigPayload(Payload):
    """Request model for updating a task configuration."""

    config_uuid: str
    name: Optional[str] = None
    typename: Optional[str] = None
    params: Optional[dict] = None
    description: Optional[str] = None
    marked_for_delete: Optional[bool] = None


class UpdateTaskConfigResponse(StatusResponse):
    pass


class WebApp:
    """Web application for managing inference models and running recognitions"""

    SUCCESS_KEY = "success"
    FAILURE_KEY = "failure"
    ERROR_KEY = "error"
    RESULT_KEY = "results"

    TASKS_API_TAG_NAME = "tasks"

    def __init__(self, app_name: str, manager: Manager):
        self._manager: Manager = manager
        self._app_name: str = app_name
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)
        self._templates: Jinja2Templates = Jinja2Templates(directory="/app/templates")
        self._dflt_args: dict[str, str] = {"app_name": self._app_name}

        self._app.mount("/static", StaticFiles(directory="/app/static", check_dir=False), name="static")

        self._register_routes()
        self._register_exception()

    def _register_exception(self):
        @self._app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
            # logger.error(msg, *args) does printf-style formatting: passing the
            # Request as the message and exc_str as an arg made logging try
            # str(request) % (exc_str,) and raise "not all arguments converted",
            # so the validation error never reached the log at all.
            logger.error("Validation error on %s %s: %s", request.method, request.url.path, exc_str)
            content = {"status_code": 10422, "message": exc_str, "data": None}
            return JSONResponse(content=content, status_code=422)

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        logging.info("Application is starting up...")

        # Register SIGUSR1 as a live task-stack dumper.
        # Usage: kill -USR1 $(pgrep -f tasks_server)
        # Output goes to stderr (check logs/tasks_server.log or supervisord stderr).
        _loop = asyncio.get_event_loop()

        def _dump_asyncio_tasks(sig, frame):
            tasks = asyncio.all_tasks(_loop)
            print(
                f"\n=== asyncio task dump: {len(tasks)} task(s) ===",
                file=sys.stderr,
                flush=True,
            )
            for t in tasks:
                print(f"\n--- {t.get_name()}  done={t.done()} ---", file=sys.stderr)
                t.print_stack(file=sys.stderr)
            print("=== end task dump ===\n", file=sys.stderr, flush=True)

        signal.signal(signal.SIGUSR1, _dump_asyncio_tasks)
        logging.info("SIGUSR1 handler registered — send 'kill -USR1 <pid>' to dump asyncio tasks to stderr")

        self._manager.start()
        yield

        logging.info("Application is shutting down...")
        await self._manager.stop()

    def app(self) -> FastAPI:
        return self._app

    def _error_response(self, request: Request, message: str):
        """Render an error response using the dynamic response template."""
        return self._templates.TemplateResponse(
            "dynamic_response.html",
            {
                "request": request,
                "title": "Error",
                "response_data": {self.ERROR_KEY: message},
                **self._dflt_args,
            },
        )

    def _register_routes(self):
        """Register all routes for the application."""

        @self._app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            """Render the home page with navigation buttons."""
            buttons = []
            return self._templates.TemplateResponse(
                "dynamic_index.html",
                {
                    "request": request,
                    "app_name": "Tasks Server",
                    "buttons": buttons,
                },
            )

        @self._app.get(
            "/server-config",
            operation_id="server_config",
            response_class=JSONResponse,
        )
        @self._app.get("/server-config", response_class=JSONResponse)
        async def server_config(request: Request):
            """Get server configuration params."""
            try:
                server_config = await self._manager.get_server_config()
                response_data = {"status": WebApp.SUCCESS_KEY, "config": server_config}
            except Exception as e:
                response_data = {"status": WebApp.FAILURE_KEY, "message": str(e)}
            return JSONResponse(content=response_data)

        @self._app.get("/status", operation_id="tasks_server_status", response_class=JSONResponse)
        async def tasks_server_status():
            return JSONResponse(content=self._manager.get_status())

        @self._app.post(
            "/api/reload-server-config",
            operation_id="reload_server_config",
            response_class=JSONResponse,
        )
        async def reload_server_config() -> JSONResponse:
            """Re-read the inference/training URL overrides from db_server settings.

            The worker loop already re-reads these once a minute; this endpoint is
            what image_tagging_server calls the moment Settings → Servers is saved,
            so a new URL takes effect immediately instead of on the next pass.
            """
            try:
                await self._manager.reload_url_overrides()
                config = await self._manager.get_server_config()
                return JSONResponse(content={"status": WebApp.SUCCESS_KEY, "config": config})
            except Exception as e:
                logger.error(e, exc_info=True)
                # 500, not 200: image_tagging_server decides whether the reload
                # took by looking at the status code alone, so a 200 here reads
                # as success and hides the split brain (UI on the new URL, tasks
                # still on the old one) until the next worker pass.
                return JSONResponse(
                    status_code=500,
                    content={"status": WebApp.FAILURE_KEY, "message": str(e)},
                )

        @self._app.get(
            "/api/tasks/types/list",
            response_model=ListTaskTypesResponse,
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="list_task_types",
        )
        async def list_task_types_api(request: Request) -> ListTaskTypesResponse:
            """Return the list of task types."""
            try:
                types_list = await self._manager.list_task_types()
                accents = await self._manager.list_task_accents()
                system_types = await self._manager.list_system_task_types()
                return ListTaskTypesResponse(
                    status=WebApp.SUCCESS_KEY,
                    types=types_list,
                    accents=accents,
                    system_types=system_types,
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListTaskTypesResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/types/schema",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="get_task_type_schema",
            response_model=TasksTypeSchemaResponse,
        )
        async def get_task_type_schema_api(
            payload: TasksTypeSchemaPayload,
        ) -> TasksTypeSchemaResponse:
            """Return the config schema for the given task type(s)."""
            try:
                schemas = await self._manager.get_tasks_type_schema(typenames=payload.task_typenames)
                return TasksTypeSchemaResponse(status=WebApp.SUCCESS_KEY, schemas=schemas)
            except Exception as e:
                return TasksTypeSchemaResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/types/status-schema",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="get_task_type_status_schema",
            response_model=TasksStatusSchemaResponse,
        )
        async def get_task_type_status_schema_api(
            payload: TasksStatusSchemaPayload,
        ) -> TasksStatusSchemaResponse:
            """
            API endpoint for getting the rich-status-card widget schema for one or more
            task types. Static per typename — safe for the frontend to fetch once and cache.
            """
            try:
                schemas = await self._manager.get_tasks_status_schema(typenames=payload.task_typenames)
                return TasksStatusSchemaResponse(status=WebApp.SUCCESS_KEY, schemas=schemas)
            except Exception as e:
                return TasksStatusSchemaResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/instances/status-data",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="get_tasks_status_data",
            response_model=TasksStatusDataResponse,
        )
        async def get_tasks_status_data_api(
            payload: TasksStatusDataPayload,
        ) -> TasksStatusDataResponse:
            """
            API endpoint for getting live status_data for one or more running task
            instances (fetched only while a task's status card is expanded). Unknown or
            not-currently-running uuids are simply omitted from the response, not an error.
            """
            try:
                status_data = await self._manager.get_tasks_status_data(payload.task_uuids)
                return TasksStatusDataResponse(status=WebApp.SUCCESS_KEY, status_data=status_data)
            except Exception as e:
                return TasksStatusDataResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.get(
            "/api/tasks/instances/{task_uuid}/live-preview",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="get_task_live_preview",
            response_model=TaskLivePreviewResponse,
        )
        async def get_task_live_preview_api(
            task_uuid: str,
            source_uuid: Optional[str] = None,
            quality: Optional[int] = None,
            since_seq: Optional[int] = None,
            full: bool = False,
        ) -> TaskLivePreviewResponse:
            """
            The frame a running task is currently working on, plus what it sees in it.

            A GET rather than the surrounding POST-with-payload convention because it is
            polled several times a second by whoever has the view open, and because it
            reads state without touching it.

            The frame is encoded per request, so a task nobody is watching pays nothing
            for this endpoint existing. Pass `since_seq` with the last `frameSeq` you
            received and a poll that outruns the camera returns fresh metadata with no
            image attached, rather than re-encoding and re-sending pixels you have.

            `full=true` serves the camera's own resolution rather than the downscaled
            frame inference ran on. Detections stay in the analysis coordinate space
            either way — see `analysisFrameWidth`/`analysisFrameHeight` in the payload.
            """
            try:
                preview = await self._manager.get_task_live_preview(
                    task_uuid, source_uuid=source_uuid, quality=quality, since_seq=since_seq, full=full
                )
            except ValueError as e:
                # Not running is the ordinary case (the watcher was stopped while the
                # view was open), not a server fault.
                return TaskLivePreviewResponse(status=WebApp.FAILURE_KEY, message=str(e))
            except Exception as e:
                logger.error(e, exc_info=True)
                return TaskLivePreviewResponse(status=WebApp.FAILURE_KEY, message=str(e))
            if preview is None:
                return TaskLivePreviewResponse(status=WebApp.FAILURE_KEY, message="This task type has no live view")
            return TaskLivePreviewResponse(status=WebApp.SUCCESS_KEY, preview=preview)

        @self._app.get(
            "/api/tasks/instances/{task_uuid}/collectors/{collector_uuid}/diagnostics",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="get_task_collector_diagnostics",
            response_model=TaskCollectorDiagnosticsResponse,
        )
        async def get_task_collector_diagnostics_api(
            task_uuid: str, collector_uuid: str
        ) -> TaskCollectorDiagnosticsResponse:
            """Live tuning stats for one collector attached to a running watcher —
            the data half of the collector diagnostics modal. A GET, polled every
            few seconds while the modal is open, for the same reason live-preview
            is: it reads state without touching it.
            """
            try:
                diagnostics = await self._manager.get_task_collector_diagnostics(task_uuid, collector_uuid)
            except ValueError as e:
                # Not running is the ordinary case (the watcher was stopped while
                # the modal was open), not a server fault.
                return TaskCollectorDiagnosticsResponse(status=WebApp.FAILURE_KEY, message=str(e))
            except Exception as e:
                logger.error(e, exc_info=True)
                return TaskCollectorDiagnosticsResponse(status=WebApp.FAILURE_KEY, message=str(e))
            return TaskCollectorDiagnosticsResponse(status=WebApp.SUCCESS_KEY, diagnostics=diagnostics)

        @self._app.get(
            "/api/tasks/instances/{task_uuid}/timelapse-preview",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="get_task_timelapse_preview",
        )
        async def get_task_timelapse_preview_api(
            task_uuid: str,
            source_uuid: str,
            rule_uuid: str,
            subject_ref: str = "",
            target_ref: str = "",
        ) -> Response:
            """The video an in-progress event timelapse has collected so far.

            Returns the mp4 itself rather than a JSON envelope: it is played by a
            `<video>` element, and base64 in a JSON field would cost a third more bytes
            for a file that is already megabytes.

            Facts about the preview ride along as headers, because the body is the
            video. `X-Timelapse-Partial` says the video is only the most recent frames
            of a longer event, which the player has to disclose.

            A GET, and it reads without mutating: the session keeps sampling throughout
            and its frames are untouched. Not cached — the whole point is that the
            answer is different a minute later.
            """
            try:
                preview = await self._manager.get_task_timelapse_preview(
                    task_uuid,
                    source_uuid=source_uuid,
                    rule_uuid=rule_uuid,
                    subject_ref=subject_ref,
                    target_ref=target_ref,
                )
            except ValueError as e:
                # The watcher stopped while the modal was open — ordinary, not a fault.
                return JSONResponse(status_code=404, content={"status": WebApp.FAILURE_KEY, "message": str(e)})
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(status_code=500, content={"status": WebApp.FAILURE_KEY, "message": str(e)})
            if preview is None:
                return JSONResponse(
                    status_code=404,
                    content={
                        "status": WebApp.FAILURE_KEY,
                        "message": "Nothing has been buffered for this event yet",
                    },
                )
            return Response(
                content=preview["video"],
                media_type="video/mp4",
                headers={
                    "Cache-Control": "no-store",
                    "X-Timelapse-Frames": str(preview["frames"]),
                    "X-Timelapse-Duration-S": f"{preview['durationS']:.3f}",
                    "X-Timelapse-Source-Span-S": f"{preview['sourceSpanS']:.3f}",
                    "X-Timelapse-Partial": "true" if preview["partial"] else "false",
                },
            )

        @self._app.get(
            "/api/tasks/sources/{source_uuid}/grab-frame",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="grab_source_frame",
            response_model=GrabSourceFrameResponse,
        )
        async def grab_source_frame_api(source_uuid: str, quality: int = 85) -> GrabSourceFrameResponse:
            """One still from a source, for drawing zones and masks against.

            Lives on tasks_server rather than db_server because opening an image provider
            is this service's job — db_server owns the source *rows*, not the cameras.

            Served from a running watcher's latest frame when there is one. That avoids a
            second RTSP session against a camera that may cap concurrent connections,
            which is a real failure mode rather than a micro-optimisation.
            """
            try:
                result = await self._manager.grab_source_frame(source_uuid, quality=quality)
            except ValueError as e:
                # An unreachable camera is an ordinary outcome here, not a server fault.
                return GrabSourceFrameResponse(status=WebApp.FAILURE_KEY, message=str(e))
            except Exception as e:
                logger.error(e, exc_info=True)
                return GrabSourceFrameResponse(status=WebApp.FAILURE_KEY, message=str(e))
            return GrabSourceFrameResponse(status=WebApp.SUCCESS_KEY, **result)

        @self._app.post(
            "/api/tasks/instances/command",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="send_task_command",
            response_model=TaskCommandResponse,
        )
        async def send_task_command_api(payload: TaskCommandPayload) -> TaskCommandResponse:
            """
            API endpoint for sending a UI-issued command to a running task instance
            (e.g. "capture now", "skip this source"). Forwarded to
            Task.handle_command(), which must only set a flag or push to a queue —
            never perform I/O. A task that doesn't implement handle_command() (the
            base implementation raises NotImplementedError) is translated into a
            clean failure response here rather than an unhandled 500.
            """
            try:
                await self._manager.send_task_command(
                    task_uuid=payload.task_uuid, command=payload.command, params=payload.params
                )
                return TaskCommandResponse(status=WebApp.SUCCESS_KEY)
            except Exception as e:
                logger.error(e, exc_info=True)
                return TaskCommandResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/instances/start",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="start_task",
            response_model=StartTasksResponse,
        )
        async def start_task_api(payload: StartTasksPayload) -> StartTasksResponse:
            """Start a new task instance from a saved task config."""
            msg = None
            try:
                task_info: TaskInfo | None = await self._manager.start_new_task(task_config_uuid=payload.config_uuid)
                if task_info and task_info.instance:
                    return StartTasksResponse(status=WebApp.SUCCESS_KEY, task_uuid=task_info.instance.uuid)
            except Exception as e:
                msg = str(e)
            return StartTasksResponse(status=WebApp.FAILURE_KEY, message=msg)

        @self._app.post(
            "/api/tasks/instances/get",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="get_tasks_info",
            response_model=TasksInfoResponse,
        )
        async def task_tasks_info_api(payload: TasksInfoPayload) -> TasksInfoResponse:
            """Return status info for one or more task instances."""
            try:
                tasks_info = await self._manager.get_tasks_instance_info(payload.task_uuids)
                models = {uuid: TaskInfoRead.from_task_info(task) for uuid, task in tasks_info.items()}
                return TasksInfoResponse(tasks=models)
            except Exception as e:
                return TasksInfoResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/tasks/instances/result",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="get_tasks_result",
            response_model=TasksResultResponse,
        )
        async def get_tasks_result_api(
            payload: TasksResultPayload,
        ) -> TasksResultResponse:
            """Return the one-shot result for a specific task (auto-deleted after fetch)."""
            try:
                tasks_result = await self._manager.get_tasks_result(payload.task_uuids)
                models = {uuid: TaskResult.model_validate(result) for uuid, result in tasks_result.items()}
                return TasksResultResponse(results=models)
            except Exception as e:
                return TasksResultResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/instances/pause",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="pause_tasks",
            response_model=PauseTasksResponse,
        )
        async def pause_tasks_api(payload: PauseTasksPayload) -> PauseTasksResponse:
            """Pause the given task instances."""
            try:
                paused_tasks = await self._manager.pause_tasks(payload.task_uuids)
                return PauseTasksResponse(task_uuids=paused_tasks)
            except Exception as e:
                logger.error(e, exc_info=True)
                return PauseTasksResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/instances/delete",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="delete_tasks",
            response_model=DeleteTasksResponse,
        )
        async def delete_tasks_api(payload: DeleteTasksPayload) -> DeleteTasksResponse:
            """Delete the given task instances."""
            try:
                deleted_uuids = await self._manager.delete_tasks(payload.task_uuids)
                return DeleteTasksResponse(task_uuids=deleted_uuids)
            except Exception as e:
                logger.error(e, exc_info=True)
                return DeleteTasksResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/instances/resume",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="resume_tasks",
            response_model=ResumeTasksResponse,
        )
        async def resume_tasks_api(payload: ResumeTasksPayload) -> ResumeTasksResponse:
            """Resume the given task instances."""
            try:
                resumed_uuids = await self._manager.resume_tasks(payload.task_uuids)
                return ResumeTasksResponse(task_uuids=resumed_uuids)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ResumeTasksResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/instances/pause-in-place",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="pause_tasks_in_place",
            response_model=PauseInPlaceResponse,
        )
        async def pause_tasks_in_place_api(
            payload: PauseInPlacePayload,
        ) -> PauseInPlaceResponse:
            """Pause tasks cooperatively in-place (task stays running, loop blocks)."""
            try:
                paused = await self._manager.pause_tasks_in_place(payload.task_uuids)
                return PauseInPlaceResponse(task_uuids=paused)
            except Exception as e:
                logger.error(e, exc_info=True)
                return PauseInPlaceResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/instances/resume-in-place",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="resume_tasks_in_place",
            response_model=ResumeInPlaceResponse,
        )
        async def resume_tasks_in_place_api(
            payload: ResumeInPlacePayload,
        ) -> ResumeInPlaceResponse:
            """Resume tasks that were paused in-place."""
            try:
                resumed = await self._manager.resume_tasks_in_place(payload.task_uuids)
                return ResumeInPlaceResponse(task_uuids=resumed)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ResumeInPlaceResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/instances/cancel",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="cancel_tasks",
            response_model=CancelTasksResponse,
        )
        async def cancel_tasks_api(payload: CancelTasksPayload) -> CancelTasksResponse:
            """Cancel the given task instances."""
            try:
                outcome = await self._manager.cancel_tasks(payload.task_uuids)
                return CancelTasksResponse(
                    task_uuids=outcome["stopped"],
                    stopping=outcome["stopping"],
                    not_running=outcome["notRunning"],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return CancelTasksResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/configs/create",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="create_task_config",
            response_model=CreateTaskConfigResponse,
        )
        async def create_task_config_api(
            payload: CreateTaskConfigPayload,
        ) -> CreateTaskConfigResponse:
            """Create a new task configuration."""
            try:
                task_config = await self._manager.create_new_task_config(
                    name=payload.name,
                    typename=payload.typename,
                    params=payload.params,
                    persistent=payload.persistent,
                    is_temporary=payload.is_temporary,
                )
                return CreateTaskConfigResponse(status=WebApp.SUCCESS_KEY, config_uuid=task_config.uuid)
            except Exception as e:
                return CreateTaskConfigResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/configs/delete",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="delete_task_configs",
            response_model=DeleteTaskConfigsResponse,
        )
        async def delete_task_configs_api(
            payload: DeleteTaskConfigsPayload,
        ) -> DeleteTaskConfigsResponse:
            """Delete the given task configs."""
            try:
                await self._manager.delete_task_configs(payload.config_uuids)
                return DeleteTaskConfigsResponse(config_uuids=payload.config_uuids)
            except Exception as e:
                logger.error(e, exc_info=True)
                return DeleteTaskConfigsResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/configs/get",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="get_task_configs",
            response_model=GetTaskConfigResponse,
        )
        async def get_task_configs_api(
            payload: GetTaskConfigsPayload,
        ) -> GetTaskConfigResponse:
            """Get task configs by uuid."""
            try:
                task_configs = await self._manager.get_task_configs(
                    payload.config_uuids,
                    is_temporary=False if payload.exclude_temporary else None,
                )
                models = {k: TaskConfigRead.model_validate(v) for k, v in task_configs.items()}
                return GetTaskConfigResponse(configs=models)
            except Exception as e:
                logger.error(e, exc_info=True)
                return GetTaskConfigResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/configs/update",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="update_task_config",
            response_model=UpdateTaskConfigResponse,
        )
        async def update_task_config_api(
            payload: UpdateTaskConfigPayload,
        ) -> UpdateTaskConfigResponse:
            """Update an existing task configuration."""
            try:
                await self._manager.update_task_config(
                    config_uuid=payload.config_uuid,
                    name=payload.name,
                    typename=payload.typename,
                    params=payload.params,
                    description=payload.description,
                    marked_for_delete=payload.marked_for_delete,
                )
                return UpdateTaskConfigResponse()
            except Exception as e:
                logger.error(e, exc_info=True)
                return UpdateTaskConfigResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/task_schema",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="schema",
            response_class=HTMLResponse,
        )
        async def get_task_schema(typename: str = Form(...)) -> HTMLResponse:
            """Retrieve and render the schema for a given task type as HTML."""
            try:
                schema = await self._manager.get_task_schema(typename=typename)
                pretty = json.dumps(schema, indent=2)
                escaped = html.escape(pretty)
                html_content = f"<pre>{escaped}</pre>"
            except Exception as e:
                html_content = f"<pre>Error: {html.escape(str(e))}</pre>"
            return HTMLResponse(content=html_content)
