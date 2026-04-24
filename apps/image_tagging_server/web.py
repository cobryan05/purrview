import base64
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import apps.db_server.web as db_web
from apps.helpers.consts import ApiTags
from apps.helpers.types import StatusResponse
from apps.inference_server.web import (
    AssociateClassWithModelClassPayload,
    GetModelClassesPayload,
    RecognizePayload,
)
from apps.tasks_server.web import (
    CancelTasksPayload,
    CancelTasksResponse,
    CreateTaskConfigPayload,
    CreateTaskConfigResponse,
    DeleteTaskConfigsPayload,
    DeleteTaskConfigsResponse,
    DeleteTasksPayload,
    DeleteTasksResponse,
    GetTaskConfigResponse,
    GetTaskConfigsPayload,
    ListTaskTypesResponse,
    StartTasksPayload,
    StartTasksResponse,
    TasksResultResponse,
    TasksInfoPayload,
    TasksInfoResponse,
    TasksResultPayload,
    TasksTypeSchemaPayload,
    TasksTypeSchemaResponse,
    UpdateTaskConfigPayload,
    UpdateTaskConfigResponse,
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
            "/api/models/classes/associate",
            tags=[ApiTags.MODELS],
            operation_id="associate_class_with_model_class",
            response_class=JSONResponse,
        )
        @self._manager.model_api_request("associate_class_with_model_class")
        async def associate_class_with_model_class_api(
            payload: AssociateClassWithModelClassPayload,
        ) -> JSONResponse: ...

        @self._app.post(
            "/api/models/classes/get",
            tags=[ApiTags.MODELS],
            operation_id="get_classes",
            response_class=JSONResponse,
        )
        @self._manager.model_api_request("get_model_classes")
        async def get_model_classes_api(_payload: GetModelClassesPayload) -> JSONResponse: ...


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
        async def recognize_api(_payload: RecognizePayload) -> JSONResponse: ...

        ################################################################################
        # CLASSES API
        ################################################################################

        @self._app.post(
            "/api/classes/add",
            tags=[ApiTags.CLASSES],
            operation_id="add_class",
            response_model=db_web.AddClassResponse,
        )
        @self._manager.classes_api_request("add_class")
        async def add_class_api(_payload: db_web.AddClassPayload) -> db_web.AddClassResponse: ...

        @self._app.post(
            "/api/classes/delete",
            tags=[ApiTags.CLASSES],
            operation_id="delete_class",
            response_model=db_web.DeleteClassResponse,
            )
        @self._manager.classes_api_request("delete_class")
        async def delete_class_api(_payload: db_web.DeleteClassPayload) -> db_web.DeleteClassResponse: ...

        @self._app.get(
            "/api/classes/list",
            tags=[ApiTags.CLASSES],
            operation_id="list_classes",
            response_model=db_web.ListClassesResponse,
        )
        @self._manager.classes_api_request("list_classes")
        async def list_classes_api() -> db_web.ListClassesResponse: ...

        @self._app.post(
            "/api/classes/update",
            tags=[ApiTags.CLASSES],
            operation_id="update_class",
            response_model=db_web.UpdateClassResponse,
        )
        @self._manager.classes_api_request("update_class")
        async def update_class_api(_payload: db_web.UpdateClassPayload) -> db_web.UpdateClassResponse: ...

        ################################################################################
        # TAGS API
        ################################################################################

        @self._app.post(
            "/api/tags/add",
            tags=[ApiTags.TAGS],
            operation_id="add_tag",
            response_model=db_web.AddTagResponse,
        )
        @self._manager.tags_api_request("add_tag")
        async def add_tag_api(_payload: db_web.AddTagPayload) -> db_web.AddTagResponse: ...
        @self._app.post(
            "/api/tags/delete",
            tags=[ApiTags.TAGS],
            operation_id="delete_tag",
            response_model=db_web.DeleteTagResponse,
        )
        @self._manager.tags_api_request("delete_tag")
        async def delete_tag_api(_payload: db_web.DeleteTagPayload) -> db_web.DeleteTagResponse: ...

        @self._app.get(
            "/api/tags/list",
            tags=[ApiTags.TAGS],
            operation_id="list_tags",
            response_model=db_web.ListTagsResponse,
        )
        @self._manager.tags_api_request("list_tags")
        async def list_tags_api() -> db_web.ListTagsResponse: ...

        @self._app.post(
            "/api/tags/update",
            tags=[ApiTags.TAGS],
            operation_id="update_tag",
            response_model=db_web.UpdateTagResponse,
        )
        @self._manager.tags_api_request("update_tag")
        async def update_tag_api(_payload: db_web.UpdateTagPayload) -> db_web.UpdateTagResponse: ...
        ################################################################################
        # TASKS API
        ################################################################################

        @self._app.get(
            "/api/tasks/types/list",
            response_model=ListTaskTypesResponse,
            tags=[ApiTags.TASKS],
            operation_id="list_task_types",
        )
        @self._manager.task_api_request("list_task_types")
        async def list_task_types_api(request: Request) -> ListTaskTypesResponse: ...

        @self._app.post(
            "/api/tasks/types/schema",
            tags=[ApiTags.TASKS],
            operation_id="get_task_type_schema",
            response_model=TasksTypeSchemaResponse,
        )
        @self._manager.task_api_request("get_task_type_schema")
        async def get_task_type_schema_api(_payload: TasksTypeSchemaPayload) -> TasksTypeSchemaResponse: ...

        @self._app.post(
            "/api/tasks/configs/create",
            tags=[ApiTags.TASKS],
            operation_id="create_task_config",
            response_model=CreateTaskConfigResponse
        )
        @self._manager.task_api_request("create_task_config")
        async def create_task_config_api(_payload: CreateTaskConfigPayload) -> CreateTaskConfigResponse: ...

        @self._app.post(
            "/api/tasks/configs/update",
            tags=[ApiTags.TASKS],
            operation_id="update_task_config",
            response_model=UpdateTaskConfigResponse,
        )
        @self._manager.task_api_request("update_task_config")
        async def update_task_config_api(_payload: UpdateTaskConfigPayload) -> UpdateTaskConfigResponse: ...

        @self._app.post(
            "/api/tasks/configs/delete",
            response_model=DeleteTaskConfigsResponse,
            tags=[ApiTags.TASKS],
            operation_id="delete_task_configs",
        )
        @self._manager.task_api_request("delete_task_configs")
        async def delete_task_configs_api(_payload: DeleteTaskConfigsPayload) -> DeleteTaskConfigsResponse: ...

        @self._app.post(
            "/api/tasks/configs/get",
            tags=[ApiTags.TASKS],
            operation_id="get_task_configs",
            response_model=GetTaskConfigResponse,
        )
        @self._manager.task_api_request("get_task_configs")
        async def get_task_configs_api(_payload: GetTaskConfigsPayload) -> GetTaskConfigResponse: ...

        @self._app.post(
            "/api/tasks/instances/start",
            tags=[ApiTags.TASKS],
            operation_id="start_task",
            response_model=StartTasksResponse,
        )
        @self._manager.task_api_request("start_task")
        async def start_task_api(_payload: StartTasksPayload) -> StartTasksResponse: ...

        @self._app.post(
            "/api/tasks/instances/cancel",
            tags=[ApiTags.TASKS],
            operation_id="cancel_tasks",
            response_model=CancelTasksResponse,
        )
        @self._manager.task_api_request("cancel_tasks")
        async def cancel_task_api(_payload: CancelTasksPayload) -> CancelTasksResponse: ...

        @self._app.post(
            "/api/tasks/instances/get",
            tags=[ApiTags.TASKS],
            operation_id="get_tasks_info",
            response_model=TasksInfoResponse
        )
        @self._manager.task_api_request("get_tasks_info")
        async def get_tasks_info_api(_payload: TasksInfoPayload) -> TasksInfoResponse: ...

        @self._app.post(
            "/api/tasks/instances/delete",
            tags=[ApiTags.TASKS],
            operation_id="delete_tasks",
            response_model=DeleteTasksResponse,
        )
        @self._manager.task_api_request("delete_tasks")
        async def delete_tasks_api(_payload: DeleteTasksPayload) -> DeleteTasksResponse: ...

        @self._app.post(
            "/api/tasks/instances/result",
            tags=[ApiTags.TASKS],
            operation_id="get_tasks_result",
            response_model=TasksResultResponse,
        )
        @self._manager.task_api_request("get_tasks_result")
        async def task_result_api(_payload: TasksResultPayload) -> TasksResultResponse: ...

        ################################################################################
        # SOURCES API
        ################################################################################

        @self._app.get(
            "/api/sources/image-providers/list",
            response_model=db_web.ListImageProvidersResponse,
            tags=[ApiTags.SOURCES],
            operation_id="list_image_providers",
        )
        @self._manager.sources_api_request("list_image_providers")
        async def list_image_providers_api() -> db_web.ListImageProvidersResponse: ...

        @self._app.post(
            "/api/sources/image-providers/schema",
            tags=[ApiTags.SOURCES],
            operation_id="get_image_provider_schema",
            response_model=db_web.GetImageProviderSchemaResponse,
        )
        @self._manager.sources_api_request("get_image_provider_schema")
        async def get_image_provider_schema_api(_payload: db_web.GetImageProviderSchemaPayload) -> db_web.GetImageProviderSchemaResponse: ...

        @self._app.post(
            "/api/sources/update",
            tags=[ApiTags.SOURCES],
            operation_id="update_source",
            response_model=db_web.UpdateSourceResponse,
        )
        @self._manager.sources_api_request("update_source")
        async def update_source_api(_payload: db_web.UpdateSourcePayload) -> db_web.UpdateSourceResponse: ...

        @self._app.post(
            "/api/sources/create",
            tags=[ApiTags.SOURCES],
            operation_id="create_source",
            response_model=db_web.CreateSourceResponse,
        )
        @self._manager.sources_api_request("create_source")
        async def create_source_api(_payload: db_web.CreateSourcePayload) -> db_web.CreateSourceResponse: ...

        @self._app.post(
            "/api/sources/delete",
            tags=[ApiTags.SOURCES],
            operation_id="delete_sources",
            response_model=db_web.DeleteSourceResponse,
        )
        @self._manager.sources_api_request("delete_sources")
        async def delete_sources_api(_payload: db_web.DeleteSourcePayload) -> db_web.DeleteSourceResponse: ...

        @self._app.get(
            "/api/sources/get",
            response_model=db_web.GetSourcesResponse,
            tags=[ApiTags.SOURCES],
            operation_id="get_sources",
        )
        @self._manager.sources_api_request("get_sources")
        async def get_sources() -> db_web.GetSourcesResponse: ...

        ################################################################################
        # Images API
        ################################################################################

        @self._app.post(
            "/api/images/metadata/get",
            response_model=db_web.GetImageMetadataResponse,
            tags=[ApiTags.IMAGES],
            operation_id="get_image_metadata",
        )
        @self._manager.images_api_request("get_image_metadata")
        async def get_image_metadata(_payload: db_web.GetImageMetadataPayload) -> db_web.GetImageMetadataResponse: ...

        @self._app.post(
            "/api/images/metadata/update",
            response_model=db_web.UpdateMetadataResponse,
            tags=[ApiTags.IMAGES],
            operation_id="update_image_metadata",
        )
        @self._manager.images_api_request("update_image_metadata")
        async def update_image_metadata(_payload: db_web.UpdateMetadataPayload) -> db_web.UpdateMetadataResponse: ...

        @self._app.post(
            "/api/images/list", response_model=db_web.ListFilesResponse, tags=[ApiTags.IMAGES], operation_id="list_images"
        )
        @self._manager.images_api_request("list_images")
        async def list_files(_payload: db_web.ListFilesPayload) -> db_web.ListFilesResponse: ...

        @self._app.post("/api/images/get", response_model=db_web.GetFileResponse, tags=[ApiTags.IMAGES], operation_id="get_image")
        @self._manager.images_api_request("get_image")
        async def get_file(_payload: db_web.GetFilePayload) -> db_web.GetFileResponse: ...