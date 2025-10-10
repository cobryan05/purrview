import base64
import json
import logging
import mimetypes
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import List, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from apps.inference_server.web import (
    AssociateLabelWithModelClassPayload,
    GetModelLabelsPayload,
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
from apps.db_server.web import (
    AddLabelPayload,
    DeleteLabelPayload,
    UpdateLabelPayload,
    GetImageProviderSchemaPayload,
    CreateSourcePayload,
    DeleteSourcePayload,
    UpdateSourcePayload,
)
from apps.helpers.consts import ApiTags


from .manager import BoundingBoxMetadata, ImageMetadata, Manager


class BoundingBoxInput(BaseModel):
    id: Optional[int]
    label_uuid: str
    x: float
    y: float
    width: float
    height: float
    tags: Optional[List[str]] = []  # List of tag uuids
    extra: Optional[dict] = {}


class UpdateMetadataPayload(BaseModel):
    image_path: str
    boxes: List[BoundingBoxInput]
    extra: Optional[dict] = {}


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
        # Images API
        ################################################################################

        @self._app.get("/api/images/metadata/get", response_class=JSONResponse, tags=[ApiTags.IMAGES])
        async def get_metadata(rel_path: str = Query("/", alias="path")):
            metadata: Optional[ImageMetadata] = await self._manager.get_image_metadata(rel_path)
            if metadata is None:
                raise HTTPException(status_code=404, detail="Image not found")
            # convert to dict for JSONResponse
            return JSONResponse(content=asdict(metadata))

        @self._app.post("/api/images/metadata/update", response_class=JSONResponse, tags=[ApiTags.IMAGES])
        async def update_metadata(payload: UpdateMetadataPayload) -> JSONResponse:
            image_path = payload.image_path
            metadata = await self._manager.get_image_metadata(image_path)
            label_uuid_map = await self._manager.get_label_uuid_map()

            if metadata is None:
                raise HTTPException(status_code=404, detail="Image not found")

            # Build new bounding box list from input
            new_boxes = []
            for b in payload.boxes:
                label_data = label_uuid_map.get(b.label_uuid, None)
                label_text = label_data.metadata.name if label_data else "Unknown"
                # Resolve label info for each label ID
                # tags = []
                # for label_id in b.tags or []:
                #     # Here, you might want to fetch label info by ID from DB or cache
                #     # Let's assume manager._db_client has get_label_by_id
                #     label_row = await self._manager._db_client.get_label_by_id(label_id)
                #     if label_row:
                #         labels.append(Label(id=label_row["id"], name=label_row["name"], color=label_row["color"]))
                # TODO TAGS
                new_boxes.append(
                    BoundingBoxMetadata(
                        id=b.id,
                        label_uuid=b.label_uuid,
                        label_text=label_text,
                        x=b.x,
                        y=b.y,
                        width=b.width,
                        height=b.height,
                        extra=b.extra or {},
                    )
                )

            metadata.boxes = new_boxes
            metadata.extra = payload.extra or {}

            await self._manager.update_image_metadata(image_path, metadata)
            return JSONResponse(content={"status": WebApp.SUCCESS_KEY})

        @self._app.get("/api/images/list", response_class=JSONResponse, tags=[ApiTags.IMAGES])
        async def list_files(
            request: Request,
            rel_path: str = Query("/", alias="path"),
            pattern: str = Query("*", alias="pattern"),
            recursive: bool = Query(False, alias="recursive"),
        ) -> JSONResponse:
            """
            API endpoint to return a list of files.

            Query Parameters:
                rel_path (str): Relative path under the root directory (default: "/").
                glob (str): Glob pattern to filter files (default: "*").
                recursive (bool): Whether to search directories recursively (default: False).

            Returns:
                JSONResponse: A JSON response containing the list of files.
            """
            try:
                file_list: List[Manager.FileEntry] = await self._manager.list_files(
                    rel_path=rel_path, patterns=pattern, recursive=recursive
                )
                response_data = {"status": WebApp.SUCCESS_KEY, "files": [entry.__dict__ for entry in file_list]}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.exception(e)
                return JSONResponse(
                    content={"status": WebApp.FAILURE_KEY, "message": str(e)},
                    status_code=500,
                )

        @self._app.get("/api/images/get", tags=[ApiTags.IMAGES])
        async def get_file(
            rel_path: str = Query(..., alias="path", description="Relative path to the file under root"),
            raw: bool = Query(False, description="If true, return as raw file response instead of base64 JSON"),
        ):
            """
            Retrieve a file either as base64 JSON (default) or as a browser-friendly image file (raw=true).

            Args:
                rel_path (str): Relative file path.
                raw (bool): If true, return as image; otherwise return base64 JSON.

            Returns:
                JSONResponse or Response
            """
            try:
                result = await self._manager.open_file(rel_path)
                if not result:
                    if raw:
                        raise HTTPException(status_code=404, detail="File not found")
                    else:
                        return JSONResponse({"status": WebApp.FAILURE_KEY, "message": f"File {rel_path} not found"})

                file_obj, filename = result
                content = await file_obj.read()
                await file_obj.close()

                mime_type, _ = mimetypes.guess_type(filename)
                mime_type = mime_type or "application/octet-stream"

                if raw:
                    return Response(
                        content=content,
                        media_type=mime_type,
                        headers={"Content-Disposition": f'inline; filename="{filename}"'},
                    )

                metadata: ImageMetadata = await self._manager.get_image_metadata(rel_path)
                encoded = base64.b64encode(content).decode("utf-8")
                return JSONResponse(
                    {
                        "status": WebApp.SUCCESS_KEY,
                        "filename": filename,
                        "mime_type": mime_type,
                        "content": encoded,
                        **asdict(metadata),
                    }
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.exception("Error retrieving file")
                return JSONResponse(
                    status_code=500,
                    content={"status": WebApp.FAILURE_KEY, "message": str(e)},
                )

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

        ################################################################################
        # RECOGNIZE API
        ################################################################################

        @self._app.post("/api/recognize", response_class=JSONResponse, tags=[ApiTags.INFERENCE])
        async def recognize_api(
            model_name: str = Form(..., description="Name of the model to use for recognition"),
            conf_thresh: float = Form(..., description="Confidence threshold for detections"),
            return_annotated: bool = Form(False, description="Whether to return the annotated image"),
            image: UploadFile = File(..., description="Image file to process"),
        ):
            """
            API endpoint for recognizing an image with a specific model.

            Args:
                model_name (str): Name of the model to use for recognition.
                conf_thresh (float): Confidence threshold for detections.
                return_annotated (bool): Whether to return the annotated image.
                image (UploadFile): Image file to process.

            Returns:
                JSONResponse: Response with new task ID or error.
            """
            try:
                image_bytes = await image.read()

                import cv2
                import numpy as np

                nparr = np.frombuffer(image_bytes, np.uint8)
                image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                results, annotated_image = await self._manager.recognize(
                    model_name=model_name,
                    image=image_array,
                    conf_thresh=conf_thresh,
                    return_annotated=return_annotated,
                )

                response_content = {
                    "status": WebApp.SUCCESS_KEY,
                    "results": results,
                }

                if return_annotated and annotated_image is not None:
                    # Encode annotated_image as base64 JPEG string
                    success, buffer = cv2.imencode(".png", annotated_image)
                    if success:
                        annotated_base64 = base64.b64encode(buffer).decode("utf-8")
                        response_content["annotated_image"] = annotated_base64

                return JSONResponse(content=response_content)

            except Exception as e:
                logger.exception("Recognition failed")
                return JSONResponse(
                    status_code=500,
                    content={"status": WebApp.FAILURE_KEY, "message": str(e)},
                )

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
