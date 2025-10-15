import base64
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from apps.db_server.web import (
    AddLabelPayload,
    CreateSourcePayload,
    DeleteLabelPayload,
    DeleteSourcePayload,
    GetFilePayload,
    GetImageMetadataPayload,
    GetImageProviderSchemaPayload,
    ListFilesPayload,
    UpdateLabelPayload,
    UpdateMetadataPayload,
    UpdateSourcePayload,
)
from apps.helpers.consts import ApiTags
from apps.inference_server.web import (
    AssociateLabelWithModelClassPayload,
    GetModelLabelsPayload,
    RecognizePayload,
)
from apps.tasks_server.web import (
    CancelTasksPayload,
    CreateTaskConfigPayload,
    DeleteTaskConfigsPayload,
    DeleteTasksPayload,
    GetTaskConfigsPayload,
    StartTasksPayload,
    TasksInfoPayload,
    TasksResultPayload,
    TasksTypeSchemaPayload,
    UpdateTaskConfigPayload,
)

from .manager import Manager

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class WebApp:
    SUCCESS_KEY = "success"
    FAILURE_KEY = "failure"

    def __init__(self, app_name: str, manager: Manager):
        """Initialize the WebApp with the application name"""
        self._app_name: str = app_name
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)
        self._templates: Jinja2Templates = Jinja2Templates(directory=os.path.join(SCRIPT_DIR, "templates"))
        self._manager: Manager = manager

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="static"), name="static")
        self._app.mount("/app-static", StaticFiles(directory=os.path.join(SCRIPT_DIR, "static")), name="app-static")

        # Register routes
        self._register_routes()

    def app(self) -> FastAPI:
        """Return the FastAPI application instance"""
        return self._app

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

    def _register_routes(self):
        """Register all routes for the application"""

        @self._app.get("/", response_class=HTMLResponse)
        async def get_home(request: Request):
            """Render the home page"""
            return self._templates.TemplateResponse("index.html", {"request": request})

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

        ################################################################################
        # MODELS API
        ################################################################################

        @self._app.get("/api/models/list", response_class=JSONResponse, tags=[ApiTags.MODELS])
        @self._manager.model_api_request("list_models")
        async def list_models_api(request: Request) -> JSONResponse: ...

        @self._app.post(
            "/api/models/labels/associate",
            tags=[ApiTags.MODELS],
            operation_id="associate_label_with_model_class",
            response_class=JSONResponse,
        )
        @self._manager.model_api_request("associate_label_with_model_class")
        async def associate_label_with_model_class_api(
            payload: AssociateLabelWithModelClassPayload,
        ) -> JSONResponse: ...

        @self._app.post(
            "/api/models/labels/get",
            tags=[ApiTags.MODELS],
            operation_id="get_labels",
            response_class=JSONResponse,
        )
        @self._manager.model_api_request("get_model_labels")
        async def get_model_labels_api(payload: GetModelLabelsPayload) -> JSONResponse: ...


        @self._app.post(
            "/api/models/labels/get",
            tags=[ApiTags.MODELS],
            operation_id="get_labels",
            response_class=JSONResponse,
        )
        @self._manager.model_api_request("get_model_labels")
        async def get_model_labels_api(payload: GetModelLabelsPayload) -> JSONResponse: ...

        ################################################################################
        # INFERENCE API
        ################################################################################

        @self._app.post(
            "/api/recognize",
            tags=[ApiTags.INFERENCE],
            operation_id="recognize",
            response_class=JSONResponse,
        )
        @self._manager.inference_api_request("recognize")
        async def recognize_api(payload: RecognizePayload) -> JSONResponse: ...

        ################################################################################
        # LABELS API
        ################################################################################

        @self._app.post(
            "/api/labels/add",
            tags=[ApiTags.LABELS],
            operation_id="add_label",
            response_class=JSONResponse,
        )
        @self._manager.labels_api_request("add_label")
        async def add_label_api(payload: AddLabelPayload) -> JSONResponse: ...

        @self._app.post(
            "/api/labels/delete",
            tags=[ApiTags.LABELS],
            operation_id="delete_label",
            response_class=JSONResponse,
        )
        @self._manager.labels_api_request("delete_label")
        async def delete_label_api(payload: DeleteLabelPayload) -> JSONResponse: ...

        @self._app.get(
            "/api/labels/list",
            tags=[ApiTags.LABELS],
            operation_id="list_labels",
            response_class=JSONResponse,
        )
        @self._manager.labels_api_request("list_labels")
        async def list_labels_api(request: Request) -> JSONResponse: ...

        @self._app.post(
            "/api/labels/update",
            tags=[ApiTags.LABELS],
            operation_id="update_label",
            response_class=JSONResponse,
        )
        @self._manager.labels_api_request("update_label")
        async def update_label_api(payload: UpdateLabelPayload) -> JSONResponse: ...

        ################################################################################
        # TASKS API
        ################################################################################

        @self._app.get(
            "/api/tasks/types/list",
            response_class=JSONResponse,
            tags=[ApiTags.TASKS],
            operation_id="list_task_types",
        )
        @self._manager.task_api_request("list_task_types")
        async def list_task_types_api(request: Request) -> JSONResponse: ...

        @self._app.post(
            "/api/tasks/types/schema",
            tags=[ApiTags.TASKS],
            operation_id="get_task_type_schema",
            response_class=JSONResponse,
        )
        @self._manager.task_api_request("get_task_type_schema")
        async def get_task_type_schema_api(payload: TasksTypeSchemaPayload) -> JSONResponse: ...

        @self._app.post(
            "/api/tasks/configs/create",
            tags=[ApiTags.TASKS],
            operation_id="create_task_config",
            response_class=JSONResponse,
        )
        @self._manager.task_api_request("create_task_config")
        async def create_task_config_api(payload: CreateTaskConfigPayload) -> JSONResponse: ...

        @self._app.post(
            "/api/tasks/configs/update",
            tags=[ApiTags.TASKS],
            operation_id="update_task_config",
            response_class=JSONResponse,
        )
        @self._manager.task_api_request("update_task_config")
        async def update_task_config_api(payload: UpdateTaskConfigPayload) -> JSONResponse: ...

        @self._app.post(
            "/api/tasks/configs/delete",
            response_class=JSONResponse,
            tags=[ApiTags.TASKS],
            operation_id="delete_task_configs",
        )
        @self._manager.task_api_request("delete_task_configs")
        async def delete_task_configs_api(payload: DeleteTaskConfigsPayload) -> JSONResponse: ...

        @self._app.post(
            "/api/tasks/configs/get",
            tags=[ApiTags.TASKS],
            operation_id="get_task_configs",
            response_class=JSONResponse,
        )
        @self._manager.task_api_request("get_task_configs")
        async def get_task_configs_api(payload: GetTaskConfigsPayload) -> JSONResponse: ...

        @self._app.post(
            "/api/tasks/instances/start",
            tags=[ApiTags.TASKS],
            operation_id="start_task",
            response_class=JSONResponse,
        )
        @self._manager.task_api_request("start_task")
        async def start_task_api(payload: StartTasksPayload) -> JSONResponse: ...

        @self._app.post(
            "/api/tasks/instances/cancel",
            tags=[ApiTags.TASKS],
            operation_id="cancel_tasks",
            response_class=JSONResponse,
        )
        @self._manager.task_api_request("cancel_tasks")
        async def cancel_task_api(payload: CancelTasksPayload) -> JSONResponse: ...

        @self._app.post(
            "/api/tasks/instances/get",
            tags=[ApiTags.TASKS],
            operation_id="get_tasks_info",
            response_class=JSONResponse,
        )
        @self._manager.task_api_request("get_tasks_info")
        async def get_tasks_info_api(payload: TasksInfoPayload) -> JSONResponse: ...

        @self._app.post(
            "/api/tasks/instances/delete",
            tags=[ApiTags.TASKS],
            operation_id="delete_tasks",
            response_class=JSONResponse,
        )
        @self._manager.task_api_request("delete_tasks")
        async def delete_tasks_api(payload: DeleteTasksPayload) -> JSONResponse: ...

        @self._app.post(
            "/api/tasks/instances/result",
            tags=[ApiTags.TASKS],
            operation_id="get_tasks_result",
            response_class=JSONResponse,
        )
        @self._manager.task_api_request("get_tasks_result")
        async def task_result_api(payload: TasksResultPayload) -> JSONResponse: ...

        ################################################################################
        # SOURCES API
        ################################################################################

        @self._app.get(
            "/api/sources/image-providers/list",
            response_class=JSONResponse,
            tags=[ApiTags.SOURCES],
            operation_id="list_image_providers",
        )
        @self._manager.sources_api_request("list_image_providers")
        async def list_image_providers_api(request: Request) -> JSONResponse: ...

        @self._app.post(
            "/api/sources/image-providers/schema",
            tags=[ApiTags.SOURCES],
            operation_id="get_image_provider_schema",
            response_class=JSONResponse,
        )
        @self._manager.sources_api_request("get_image_provider_schema")
        async def get_image_provider_schema_api(payload: GetImageProviderSchemaPayload) -> JSONResponse: ...

        @self._app.post(
            "/api/sources/update",
            tags=[ApiTags.SOURCES],
            operation_id="update_source",
            response_class=JSONResponse,
        )
        @self._manager.sources_api_request("update_source")
        async def update_source_api(payload: UpdateSourcePayload) -> JSONResponse: ...

        @self._app.post(
            "/api/sources/create",
            tags=[ApiTags.SOURCES],
            operation_id="create_source",
            response_class=JSONResponse,
        )
        @self._manager.sources_api_request("create_source")
        async def create_source_api(payload: CreateSourcePayload) -> JSONResponse: ...

        @self._app.post(
            "/api/sources/delete",
            tags=[ApiTags.SOURCES],
            operation_id="delete_sources",
            response_class=JSONResponse,
        )
        @self._manager.sources_api_request("delete_sources")
        async def delete_sources_api(payload: DeleteSourcePayload) -> JSONResponse: ...

        @self._app.get(
            "/api/sources/get",
            response_class=JSONResponse,
            tags=[ApiTags.SOURCES],
            operation_id="get_sources",
        )
        @self._manager.sources_api_request("get_sources")
        async def get_sources(payload: Request) -> JSONResponse: ...

        ################################################################################
        # Images API
        ################################################################################

        @self._app.post(
            "/api/images/metadata/get",
            response_class=JSONResponse,
            tags=[ApiTags.IMAGES],
            operation_id="get_image_metadata",
        )
        @self._manager.images_api_request("get_image_metadata")
        async def get_image_metadata(payload: GetImageMetadataPayload) -> JSONResponse: ...

        @self._app.post(
            "/api/images/metadata/update",
            response_class=JSONResponse,
            tags=[ApiTags.IMAGES],
            operation_id="update_image_metadata",
        )
        @self._manager.images_api_request("update_image_metadata")
        async def update_image_metadata(payload: UpdateMetadataPayload) -> JSONResponse: ...

        @self._app.post(
            "/api/images/list", response_class=JSONResponse, tags=[ApiTags.IMAGES], operation_id="list_images"
        )
        @self._manager.images_api_request("list_images")
        async def list_files(payload: ListFilesPayload) -> JSONResponse: ...

        @self._app.post("/api/images/get", response_class=JSONResponse, tags=[ApiTags.IMAGES], operation_id="get_image")
        @self._manager.images_api_request("get_image")
        async def get_file(payload: GetFilePayload) -> JSONResponse: ...
