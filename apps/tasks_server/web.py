"""Web API for Tasks Server"""

import html
import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from apps.helpers.consts import JsonValues
from apps.helpers.types import (
    StatusResponse,
    TaskConfigMetadata,
    TaskConfigMetadataModel,
    TaskResultModel,
)
from apps.tasks_server.manager import Manager, TaskInfo
from apps.tasks_server.types import TaskInfoModel

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class CreateTaskConfigPayload(BaseModel):
    """Request model for creating a new task configuration."""

    name: str
    typename: str
    params: dict
    persistent: bool = True


class CreateTaskConfigResponse(StatusResponse):
    config_uuid: Optional[str] = None


class DeleteTasksPayload(BaseModel):
    task_uuids: List[str]


class DeleteTasksResponse(StatusResponse):
    task_uuids: List[str] = Field(default_factory=list)


class DeleteTaskConfigsPayload(BaseModel):
    config_uuids: List[str]


class DeleteTaskConfigsResponse(StatusResponse):
    config_uuids: List[str] = Field(default_factory=list)


class GetTaskConfigsPayload(BaseModel):
    config_uuids: Optional[Union[List[str], str]] = None


class GetTaskConfigResponse(StatusResponse):
    configs: Dict[str, TaskConfigMetadataModel] = Field(default_factory=dict)


class ListTaskTypesResponse(StatusResponse):
    types: List[str] = Field(default_factory=list)


class PauseTasksPayload(BaseModel):
    task_uuids: List[str]

class PauseTasksResponse(StatusResponse):
    task_uuids: List[str] = Field(default_factory=list)


class ResumeTasksPayload(BaseModel):
    task_uuids: List[str]


class ResumeTasksResponse(StatusResponse):
    task_uuids: List[str] = Field(default_factory=list)


class CancelTasksPayload(BaseModel):
    task_uuids: List[str]


class CancelTasksResponse(StatusResponse):
    task_uuids: List[str] = Field(default_factory=list)


class StartTasksPayload(BaseModel):
    config_uuid: str


class StartTasksResponse(StatusResponse):
    task_uuid: Optional[str] = None


class TasksTypeSchemaPayload(BaseModel):
    task_typenames: List[str]

class TasksTypeSchemaResponse(StatusResponse):
    schemas: Dict[str, dict] = Field(default_factory=dict)


class TasksResultPayload(BaseModel):
    task_uuids: List[str]


class TasksResultResponse(StatusResponse):
    results: dict[str, TaskResultModel] = Field(default_factory=dict)


class TasksInfoPayload(BaseModel):
    task_uuids: Optional[List[str]] = None


class TasksInfoResponse(StatusResponse):
    tasks: dict[str, TaskInfoModel] = Field(default_factory=dict)


class UpdateTaskConfigPayload(BaseModel):
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
        """
        Initialize the WebApp with the application name and manager.

        Args:
            app_name (str): Name of the application.
            manager (Manager): Instance of the Manager class
        """
        self._manager: Manager = manager  # Manager instance
        self._app_name: str = app_name  # Name of the application
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)  # FastAPI instance with lifespan events
        self._templates: Jinja2Templates = Jinja2Templates(directory="/app/templates")  # Template engine for rendering HTML
        self._dflt_args: dict[str, str] = {"app_name": self._app_name}  # Default arguments for templates

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="/app/static"), name="static")

        # Register routes
        self._register_routes()
        self._register_exception()

    def _register_exception(self):
        @self._app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            """
            Custom exception handler for validation errors.

            Args:
                request (Request): The incoming request.
                exc (RequestValidationError): The validation exception.

            Returns:
                JSONResponse: A JSON-formatted error response.
            """
            exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
            logger.error(request, exc_str)
            content = {"status_code": 10422, "message": exc_str, "data": None}
            return JSONResponse(content=content, status_code=422)

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """
        Lifespan event handler for startup and shutdown logic.

        Args:
            app (FastAPI): The FastAPI application instance.
        """
        logging.info("Application is starting up...")

        self._manager.start()  # Start the manager's periodic task
        yield  # This allows the app to run

        logging.info("Application is shutting down...")

    def app(self) -> FastAPI:
        """
        Return the FastAPI application instance.

        Returns:
            FastAPI: The FastAPI application instance.
        """
        return self._app

    def _error_response(self, request: Request, message: str):
        """
        Render an error response using the dynamic response template.

        Args:
            request (Request): The request object.
            message (str): The error message to display.

        Returns:
            TemplateResponse: Rendered HTML error response.
        """
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
            """
            Render the home page with navigation buttons.

            Args:
                request (Request): The incoming request.

            Returns:
                HTMLResponse: Rendered homepage with buttons.
            """
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
            """
            Get server configuration params

            Args:
                request (Request): The incoming request.

            Returns:
                JSONResponse: Server configuration parameters.
            """
            try:
                server_config = await self._manager.get_server_config()
                response_data = {"status": WebApp.SUCCESS_KEY, "config": server_config}
            except Exception as e:
                response_data = {"status": WebApp.FAILURE_KEY, "message": str(e)}
            return JSONResponse(content=response_data)

        @self._app.get(
            "/api/tasks/types/list",
            response_model=ListTaskTypesResponse,
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="list_task_types",
        )
        async def list_task_types_api(request: Request) -> ListTaskTypesResponse:
            """
            API endpoint to return a list of task types.

            Args:
                request (Request): The FastAPI request object.

            Returns:
                ListTaskTypesResponse: A response containing the list of task types.
            """
            try:
                types_list = await self._manager.list_task_types()
                return ListTaskTypesResponse(status=WebApp.SUCCESS_KEY, types=types_list)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListTaskTypesResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/types/schema",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="get_task_type_schema",
            response_model=TasksTypeSchemaResponse,
        )
        async def get_task_type_schema_api(payload: TasksTypeSchemaPayload) -> TasksTypeSchemaResponse:
            """
            API endpoint for getting the schema of a specific task type.

            Args:
                payload (TasksSchemaPayload): Request payload containing the task type.

            Returns:
                TasksTypeSchemaResult: Response with task type schema or error.
            """
            try:
                schemas = await self._manager.get_tasks_type_schema(typenames=payload.task_typenames)
                return TasksTypeSchemaResponse(status=WebApp.SUCCESS_KEY, schemas=schemas)
            except Exception as e:
                return TasksTypeSchemaResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/instances/start",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="start_task",
            response_model=StartTasksResponse,
        )
        async def start_task_api(payload: StartTasksPayload) -> StartTasksResponse:
            """
            API endpoint for starting a new task.

            Args:
                req (StartTaskRequest): Request object with task parameters.

            Returns:
                StartTasksResponse: Response with new task ID or error.
            """
            msg = None
            try:
                task_info: TaskInfo | None = await self._manager.start_new_task(task_config_uuid=payload.config_uuid)
                if task_info and task_info.task_metadata:
                    return StartTasksResponse(status=WebApp.SUCCESS_KEY, task_uuid=task_info.task_metadata.uuid)
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
            """
            API endpoint for getting status of one or more tasks.

            Args:
                req (TaskStatusRequest): Request object with task IDs.

            Returns:
                TasksInfoResponse: Response with task information or error.
            """
            try:
                tasks_info = await self._manager.get_tasks_instance_info(payload.task_uuids)
                models = {uuid: TaskInfoModel.from_dataclass(task) for uuid, task in tasks_info.items()}
                return TasksInfoResponse(tasks=models)
            except Exception as e:
                return TasksInfoResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/tasks/instances/result",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="get_tasks_result",
            response_model=TasksResultResponse,
        )
        async def get_tasks_result_api(payload: TasksResultPayload) -> TasksResultResponse:
            """
            API endpoint for getting result of a specific task.

            Args:
                req (TaskResultRequest): Request object with task ID.

            Returns:
                TasksResultResponse: A response containing task result.
            """
            try:
                tasks_result = await self._manager.get_tasks_result(payload.task_uuids)
                models = {uuid: TaskResultModel.from_dataclass(result) for uuid, result in tasks_result.items()}
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
            """
            API endpoint to pause tasks.

            Args:
                req (PauseTasksRequest): Request object with task IDs to pause.

            Returns:
                JSONResponse: JSON response with pause result.
            """
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
            """
            API endpoint to delete tasks.

            Args:
                req (DeleteTasksRequest): Request object with task IDs to delete.

            Returns:
                JSONResponse: JSON response with delete result.
            """
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
            """
            API endpoint to resume tasks.

            Args:
                req (ResumeTasksRequest): Request object with task IDs to resume.

            Returns:
                ResumeTasksResponse: A response containing the UUIDs of the resumed tasks.
            """
            try:
                resumed_uuids = await self._manager.resume_tasks(payload.task_uuids)
                return ResumeTasksResponse(task_uuids=resumed_uuids)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ResumeTasksResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/instances/cancel",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="cancel_tasks",
            response_model=CancelTasksResponse,
        )
        async def cancel_tasks_api(payload: CancelTasksPayload) -> CancelTasksResponse:
            """
            API endpoint to cancel tasks.

            Args:
                req (ResumeTasksRequest): Request object with task IDs to cancel.

            Returns:
                JSONResponse: JSON response with resume result.
            """
            try:
                uuids = await self._manager.cancel_tasks(payload.task_uuids)
                return CancelTasksResponse(task_uuids=uuids)
            except Exception as e:
                logger.error(e, exc_info=True)
                return CancelTasksResponse(status=WebApp.FAILURE_KEY, message=str(e))

        @self._app.post(
            "/api/tasks/configs/create",
            tags=[WebApp.TASKS_API_TAG_NAME],
            operation_id="create_task_config",
            response_model=CreateTaskConfigResponse,
        )
        async def create_task_config_api(payload: CreateTaskConfigPayload) -> CreateTaskConfigResponse:
            """
            API endpoint for creating a new task configuration.

            Args:
                req (CreateTaskRequest): Request object with task type and parameters.

            Returns:
                JSONResponse: Response with new task ID or error.
            """
            try:
                task_config: TaskConfigMetadata = await self._manager.create_new_task_config(
                    name=payload.name, typename=payload.typename, params=payload.params, persistent=payload.persistent
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
        async def delete_task_configs_api(payload: DeleteTaskConfigsPayload) -> DeleteTaskConfigsResponse:
            """
            API endpoint to delete tasks configs

            Args:
                req (DeleteTasksRequest): Request object with task IDs to delete.

            Returns:
                JSONResponse: JSON response with deletion result.
            """
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
        async def get_task_configs_api(payload: GetTaskConfigsPayload) -> GetTaskConfigResponse:
            """
            API endpoint to get tasks configs

            Args:
                req (GetTaskConfigsPayload): Request object with task IDs to get

            Returns:
                JSONResponse: JSON response with task configs.
            """
            try:
                task_configs = await self._manager.get_task_configs(payload.config_uuids)
                models = {k: TaskConfigMetadataModel.from_dataclass(v) for k, v in task_configs.items()}
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
        async def update_task_config_api(payload: UpdateTaskConfigPayload) -> UpdateTaskConfigResponse:
            """
            API endpoint for updating an existing task configuration.

            Args:
                payload (UpdateTaskConfigPayload): Update data.
            """
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
            """
            Endpoint to retrieve and display the schema for a given task type.

            Args:
                typename (str): The name of the task type to retrieve the schema for.

            Returns:
                HTMLResponse: Rendered HTML content of the JSON schema or error message.
            """
            try:
                schema = await self._manager.get_task_schema(typename=typename)
                pretty = json.dumps(schema, indent=2)
                escaped = html.escape(pretty)
                html_content = f"<pre>{escaped}</pre>"
            except Exception as e:
                html_content = f"<pre>Error: {html.escape(str(e))}</pre>"
            return HTMLResponse(content=html_content)
