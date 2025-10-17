"""Web API for DB Server"""

import base64
import logging
import mimetypes
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from apps.helpers.consts import ApiTags, JsonKeys, JsonValues

from .manager import (
    BoundingBoxMetadata,
    FileEntry,
    ImageMetadata,
    Manager,
    SourceMetadata,
)

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class AddLabelPayload(BaseModel):
    name: str
    color: str
    parent_uuid: Optional[str] = None


class DeleteLabelPayload(BaseModel):
    label_uuid: str


class UpdateLabelPayload(BaseModel):
    label_uuid: str
    name: Optional[str] = None
    color: Optional[str] = None


class GetImageProviderSchemaPayload(BaseModel):
    image_provider: str


class CreateSourcePayload(BaseModel):
    source_name: str
    image_provider: str
    provider_params: dict


class DeleteSourcePayload(BaseModel):
    source_uuids: List[str]


class UpdateSourcePayload(BaseModel):
    source_name: str
    source_uuid: str
    image_provider: str
    provider_params: dict


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


class ListFilesPayload(BaseModel):
    path: str = "/"
    pattern: str = "*"
    recursive: bool = False


class GetImageMetadataPayload(BaseModel):
    image_path: str


class GetFilePayload(BaseModel):
    path: str


class WebApp:
    """Web application for managing the database"""

    def __init__(self, app_name: str, manager: Manager):
        """
        Initialize the WebApp with the application name and manager.

        Args:
            app_name (str): Name of the application.
            manager (Manager): Instance of the Manager class for handling models.
        """
        self._manager: Manager = manager
        self._app_name: str = app_name
        self._app: FastAPI = FastAPI(lifespan=self._lifespan)
        self._dflt_args: dict[str, str] = {"app_name": self._app_name}  # Default arguments for templates

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="static"), name="static")

        # Register routes
        self._register_routes()
        self._register_exception()

    def _register_exception(self):
        @self._app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):

            exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
            logger.error(request, exc_str)
            content = {"status_code": 10422, JsonKeys.MESSAGE: exc_str, "data": None}
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

    def _register_routes(self):
        """Register all routes for the application."""

        ################################################################################
        # Labels API
        ################################################################################

        @self._app.post("/api/labels/add", response_class=JSONResponse, tags=[ApiTags.LABELS], operation_id="add_label")
        async def add_label_api(request: AddLabelPayload) -> JSONResponse:
            """
            API endpoint to add a new label.
            """
            try:
                label = await self._manager.create_new_label(
                    request.name, request.color, parent_uuid=request.parent_uuid
                )
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.SUCCESS, "label": asdict(label)})
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)})

        @self._app.post(
            "/api/labels/delete", response_class=JSONResponse, tags=[ApiTags.LABELS], operation_id="delete_label"
        )
        async def delete_label_api(request: DeleteLabelPayload) -> JSONResponse:
            """
            API endpoint to delete an existing abel.
            """
            try:
                await self._manager.delete_label(request.label_uuid)
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.SUCCESS})
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)})

        @self._app.get(
            "/api/labels/list", response_class=JSONResponse, tags=[ApiTags.LABELS], operation_id="list_labels"
        )
        async def list_labels_api(request: Request) -> JSONResponse:
            """
            API endpoint to return a list of labels with metadata.

            Args:
                request (Request): The FastAPI request object.

            Returns:
                JSONResponse: A JSON response containing the list of labels.
            """
            try:
                label_list = await self._manager.get_labels()
                labels = [asdict(label) for label in label_list]
                response_data = {JsonKeys.STATUS: JsonValues.SUCCESS, "labels": labels}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                )

        @self._app.post(
            "/api/labels/update", response_class=JSONResponse, tags=[ApiTags.LABELS], operation_id="update_label"
        )
        async def update_label_api(request: UpdateLabelPayload) -> JSONResponse:
            """
            API endpoint to update an existing label's name and/or color.
            """
            try:
                await self._manager.update_label(label_uuid=request.label_uuid, name=request.name, color=request.color)
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.SUCCESS})
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                )

        ################################################################################
        # Sources API
        ################################################################################

        @self._app.get(
            "/api/sources/image-providers/list",
            response_class=JSONResponse,
            tags=[ApiTags.SOURCES],
            operation_id="list_image_providers",
        )
        async def list_image_providers_api(request: Request) -> JSONResponse:
            """
            API endpoint to return a list of available image provider types

            Args:
                request (Request): The FastAPI request object.

            Returns:
                JSONResponse: A JSON response containing the list of models.
            """
            try:
                providers_list = await self._manager.list_avail_image_providers()
                response_data = {JsonKeys.STATUS: JsonValues.SUCCESS, "providers": providers_list}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                )

        @self._app.post(
            "/api/sources/image-providers/schema",
            tags=[ApiTags.SOURCES],
            operation_id="get_image_provider_schema",
            response_class=JSONResponse,
        )
        async def get_image_provider_schema_api(payload: GetImageProviderSchemaPayload) -> JSONResponse:
            """
            API endpoint to get the schema for a specific image provider.
            """
            try:
                schema = await self._manager.get_image_provider_schema(payload.image_provider)
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.SUCCESS, "schema": schema})
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                )

        @self._app.post(
            "/api/sources/update",
            tags=[ApiTags.SOURCES],
            operation_id="update_source",
            response_class=JSONResponse,
        )
        async def update_source_api(payload: UpdateSourcePayload) -> JSONResponse:
            """
            API endpoint for updating an existing source.
            """
            try:
                source_metadata: SourceMetadata = await self._manager.update_source(
                    payload.source_uuid,
                    image_provider=payload.image_provider,
                    provider_params=payload.provider_params,
                    source_name=payload.source_name,
                )
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.SUCCESS, "source": asdict(source_metadata)})
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                )

        @self._app.post(
            "/api/sources/create",
            tags=[ApiTags.SOURCES],
            operation_id="create_source",
            response_class=JSONResponse,
        )
        async def create_source_api(payload: CreateSourcePayload) -> JSONResponse:
            """
            API endpoint for creating a new source.

            Args:
                req (CreateSourceRequest): Request object with source type and parameters.

            Returns:
                JSONResponse: Response with new source ID or error.
            """
            try:
                source_metadata: SourceMetadata = await self._manager.create_new_source(
                    image_provider=payload.image_provider,
                    provider_params=payload.provider_params,
                    source_name=payload.source_name,
                )
                response_data = {JsonKeys.STATUS: JsonValues.SUCCESS, "source": asdict(source_metadata)}
            except Exception as e:
                response_data = {JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)}
            return JSONResponse(content=response_data)

        @self._app.post(
            "/api/sources/delete",
            tags=[ApiTags.SOURCES],
            operation_id="delete_sources",
            response_class=JSONResponse,
        )
        async def delete_sources_api(payload: DeleteSourcePayload) -> JSONResponse:
            """
            API endpoint to delete sources.
            """
            try:
                await self._manager.delete_sources(payload.source_uuids)
                return JSONResponse(content={JsonKeys.STATUS: JsonValues.SUCCESS})
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                )

        @self._app.get(
            "/api/sources/get",
            response_class=JSONResponse,
            tags=[ApiTags.SOURCES],
            operation_id="get_sources",
        )
        async def get_sources_api(request: Request) -> JSONResponse:
            """
            API endpoint to return a list of available sources
            Args:
                request (Request): The FastAPI request object.

            Returns:
                JSONResponse: A JSON response containing the list of sources
            """
            try:
                sources: List[SourceMetadata] = await self._manager.get_avail_sources()
                sources_dict = {source.uuid: asdict(source) for source in sources}
                response_data = {JsonKeys.STATUS: JsonValues.SUCCESS, "sources": sources_dict}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                )

        ################################################################################
        # Images API
        ################################################################################

        @self._app.get(
            "/api/images/metadata/get",
            response_class=JSONResponse,
            tags=[ApiTags.IMAGES],
            operation_id="get_image_metadata",
        )
        async def get_image_metadata(payload: GetImageMetadataPayload) -> JSONResponse:
            metadata: Optional[ImageMetadata] = await self._manager.get_image_metadata(payload.image_path)
            if metadata is None:
                raise HTTPException(status_code=404, detail="Image not found")
            # convert to dict for JSONResponse
            return JSONResponse(content=asdict(metadata))

        @self._app.post(
            "/api/images/metadata/update",
            response_class=JSONResponse,
            tags=[ApiTags.IMAGES],
            operation_id="update_image_metadata",
        )
        async def update_image_metadata(payload: UpdateMetadataPayload) -> JSONResponse:
            image_path = payload.image_path
            metadata = await self._manager.get_image_metadata(image_path)
            label_uuid_map = await self._manager.get_label_uuid_map()

            if metadata is None:
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: "Image not found"},
                )

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
            return JSONResponse(content={JsonKeys.STATUS: JsonValues.SUCCESS})

        @self._app.post(
            "/api/images/list",
            response_class=JSONResponse,
            tags=[ApiTags.IMAGES],
            operation_id="list_images",
        )
        async def list_images(payload: ListFilesPayload) -> JSONResponse:
            """
            API endpoint to return a list of files.

            Query Parameters:
                path (str): Relative path under the root directory (default: "/").
                glob (str): Glob pattern to filter files (default: "*").
                recursive (bool): Whether to search directories recursively (default: False).

            Returns:
                JSONResponse: A JSON response containing the list of files.
            """
            try:
                file_list: List[FileEntry] = await self._manager.list_files(
                    rel_path=payload.path, patterns=payload.pattern, recursive=payload.recursive
                )
                response_data = {JsonKeys.STATUS: JsonValues.SUCCESS, "files": [entry.__dict__ for entry in file_list]}
                return JSONResponse(content=response_data)
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse(
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                )

        @self._app.post(
            "/api/images/get",
            response_class=JSONResponse,
            tags=[ApiTags.IMAGES],
            operation_id="get_image",
        )
        async def get_file(
            payload: GetFilePayload,
        ) -> JSONResponse:
            """
            Retrieve a file either as base64 JSON

            Args:
                path (str): Relative file path.

            Returns:
                JSONResponse or Response
            """
            try:
                result = await self._manager.open_file(payload.path)
                if not result:
                    return JSONResponse(
                        {
                            JsonKeys.STATUS: JsonValues.FAILURE,
                            JsonKeys.MESSAGE: f"File {payload.path} not found",
                        }
                    )

                file_obj, filename = result
                content = await file_obj.read()
                await file_obj.close()

                mime_type, _ = mimetypes.guess_type(filename)
                mime_type = mime_type or "application/octet-stream"

                metadata: ImageMetadata = await self._manager.get_image_metadata(payload.path)
                encoded = base64.b64encode(content).decode("utf-8")
                return JSONResponse(
                    {
                        JsonKeys.STATUS: JsonValues.SUCCESS,
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
                    content={JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)},
                )
