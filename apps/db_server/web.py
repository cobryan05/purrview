"""Web API for DB Server"""

import base64
import logging
import mimetypes
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from apps.db_server.manager import Manager
from apps.helpers.consts import ApiTags, JsonKeys, JsonValues
from apps.helpers.types import (
    BoundingBoxMetadata,
    BoundingBoxMetadataModel,
    ClassDataModel,
    ClassMetadataModel,
    FileEntryModel,
    ImageMetadata,
    ImageMetadataModel,
    SourceMetadata,
    SourceMetadataModel,
    StatusResponse,
    TagKinds,
    TagMetadataModel,
)

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


########
# Classes
########
class AddClassPayload(BaseModel):
    name: str
    color: str
    parent_uuid: Optional[str] = None


class AddClassResponse(StatusResponse):
    metadata: Optional[ClassMetadataModel] = None


class DeleteClassPayload(BaseModel):
    class_uuid: str


class DeleteClassResponse(StatusResponse):
    pass


class ListClassesPayload(BaseModel):
    pass


class ListClassesResponse(StatusResponse):
    classes: List[ClassDataModel] = Field(default_factory=list)


class UpdateClassPayload(BaseModel):
    class_uuid: str
    name: Optional[str] = None
    color: Optional[str] = None


class UpdateClassResponse(StatusResponse):
    pass


########
# Tags
########
class AddTagPayload(BaseModel):
    name: str
    color: str
    protected: bool = False
    kind: str = TagKinds.GENERIC
    exclusive_group: Optional[str] = None


class AddTagResponse(StatusResponse):
    tag: Optional[TagMetadataModel] = None


class DeleteTagPayload(BaseModel):
    tag_uuid: str


class DeleteTagResponse(StatusResponse):
    pass


class ListTagsPayload(BaseModel):
    pass


class ListTagsResponse(StatusResponse):
    tags: List[TagMetadataModel] = Field(default_factory=list)


class UpdateTagPayload(BaseModel):
    tag_uuid: str
    name: Optional[str] = None
    color: Optional[str] = None
    protected: Optional[bool] = None
    kind: Optional[str] = None
    exclusive_group: Optional[str] = None


class UpdateTagResponse(StatusResponse):
    pass


########
# Image Providers
########
class GetImageProviderSchemaPayload(BaseModel):
    image_provider: str


class GetImageProviderSchemaResponse(StatusResponse):
    image_provider_schema: Optional[dict] = None


class ListImageProvidersPayload(BaseModel):
    pass


class ListImageProvidersResponse(StatusResponse):
    image_providers: List[str] = Field(default_factory=list)


########
# Sources
########
class CreateSourcePayload(BaseModel):
    source_name: str
    image_provider: str
    provider_params: dict


class CreateSourceResponse(StatusResponse):
    source: Optional[SourceMetadataModel] = None


class DeleteSourcePayload(BaseModel):
    source_uuids: List[str]


class DeleteSourceResponse(StatusResponse):
    pass


class GetSourcesPayload(BaseModel):
    pass


class GetSourcesResponse(StatusResponse):
    sources: dict[str, SourceMetadataModel] = Field(default_factory=dict)


class UpdateSourcePayload(BaseModel):
    source_name: str
    source_uuid: str
    image_provider: str
    provider_params: dict


class UpdateSourceResponse(StatusResponse):
    source: Optional[SourceMetadataModel] = None


########
# Bounding Boxes
########
class BoundingBoxInput(BaseModel):
    class_uuid: str
    x: float
    y: float
    width: float
    height: float
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    tag_uuids: List[str] = Field(default_factory=list)
    extra: Optional[dict] = {}


class GetBoundingBoxInfoPayload(BaseModel):
    uuids: List[str]


class GetBoundingBoxInfoResponse(StatusResponse):
    metadata: dict[str, BoundingBoxMetadataModel] = Field(default_factory=dict)


class UpdateMetadataPayload(BaseModel):
    image_path: str
    boxes: List[BoundingBoxInput]
    extra: Optional[dict] = {}


class UpdateMetadataResponse(StatusResponse):
    metadata: Optional[ImageMetadataModel] = None


class ListFilesPayload(BaseModel):
    path: str = "/"
    pattern: Optional[str] = "*"
    recursive: Optional[bool] = False


class ListFilesResponse(StatusResponse):
    files: List[FileEntryModel] = Field(default_factory=list)


class GetImageMetadataPayload(BaseModel):
    image_path: str


class GetImageMetadataResponse(StatusResponse):
    metadata: Optional[ImageMetadataModel] = None


class GetFilePayload(BaseModel):
    path: str


class GetFileResponse(StatusResponse):
    filename: str
    mime_type: Optional[str] = None
    image_base64: Optional[str] = None
    metadata: Optional[ImageMetadataModel] = None


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
        # Classes API
        ################################################################################

        @self._app.post(
            "/api/classes/add", response_model=AddClassResponse, tags=[ApiTags.CLASSES], operation_id="add_class"
        )
        async def add_class_api(request: AddClassPayload) -> AddClassResponse:
            """
            API endpoint to add a new class.
            """
            try:
                new_metadata = await self._manager.create_new_class(
                    request.name, request.color, parent_uuid=request.parent_uuid
                )

                return AddClassResponse(status=JsonValues.SUCCESS, metadata=ClassMetadataModel.from_dataclass(new_metadata))
            except Exception as e:
                logger.error(e, exc_info=True)
                return AddClassResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/classes/delete",
            response_model=DeleteClassResponse,
            tags=[ApiTags.CLASSES],
            operation_id="delete_class",
        )
        async def delete_class_api(request: DeleteClassPayload) -> DeleteClassResponse:
            """
            API endpoint to delete an existing class.
            """
            try:
                await self._manager.delete_class(request.class_uuid)
                return DeleteClassResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return DeleteClassResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/classes/list", response_model=ListClassesResponse, tags=[ApiTags.CLASSES], operation_id="list_classes"
        )
        async def list_classes_api() -> ListClassesResponse:
            """
            API endpoint to return a list of classes with metadata.

            Args:
                request (Request): The FastAPI request object.

            Returns:
                ListClassesResponse: A response containing the list of classes.
            """
            try:
                class_list = await self._manager.get_classes()
                class_models = [ClassDataModel.from_dataclass(cls) for cls in class_list]
                response_data = ListClassesResponse(status=JsonValues.SUCCESS, classes=class_models)
                return response_data
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListClassesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/classes/update",
            response_model=UpdateClassResponse,
            tags=[ApiTags.CLASSES],
            operation_id="update_class",
        )
        async def update_class_api(request: UpdateClassPayload) -> UpdateClassResponse:
            """
            API endpoint to update an existing class's name and/or color.
            """
            try:
                await self._manager.update_class(class_uuid=request.class_uuid, name=request.name, color=request.color)
                return UpdateClassResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return UpdateClassResponse(status=JsonValues.FAILURE, message=str(e))

        ################################################################################
        # Sources API
        ################################################################################

        @self._app.get(
            "/api/sources/image-providers/list",
            response_model=ListImageProvidersResponse,
            tags=[ApiTags.SOURCES],
            operation_id="list_image_providers",
        )
        async def list_image_providers_api() -> ListImageProvidersResponse:
            """
            API endpoint to return a list of available image provider types

            Args:
                request (Request): The FastAPI request object.

            Returns:
                ListImageProvidersResponse: A response containing the list of models.
            """
            try:
                providers_list = await self._manager.list_avail_image_providers()
                return ListImageProvidersResponse(status=JsonValues.SUCCESS, image_providers=providers_list)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListImageProvidersResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/sources/image-providers/schema",
            tags=[ApiTags.SOURCES],
            operation_id="get_image_provider_schema",
            response_model=GetImageProviderSchemaResponse,
        )
        async def get_image_provider_schema_api(
            payload: GetImageProviderSchemaPayload,
        ) -> GetImageProviderSchemaResponse:
            """
            API endpoint to get the schema for a specific image provider.
            """
            try:
                schema = await self._manager.get_image_provider_schema(payload.image_provider)
                return GetImageProviderSchemaResponse(status=JsonValues.SUCCESS, image_provider_schema=schema)
            except Exception as e:
                logger.error(e, exc_info=True)
                return GetImageProviderSchemaResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/sources/update",
            tags=[ApiTags.SOURCES],
            operation_id="update_source",
            response_model=UpdateSourceResponse,
        )
        async def update_source_api(payload: UpdateSourcePayload) -> UpdateSourceResponse:
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
                source_model = SourceMetadataModel.from_dataclass(source_metadata)
                return UpdateSourceResponse(status=JsonValues.SUCCESS, source=source_model)
            except Exception as e:
                logger.error(e, exc_info=True)
                return UpdateSourceResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/sources/create",
            tags=[ApiTags.SOURCES],
            operation_id="create_source",
            response_model=CreateSourceResponse,
        )
        async def create_source_api(payload: CreateSourcePayload) -> CreateSourceResponse:
            """
            API endpoint for creating a new source.

            Args:
                req (CreateSourceRequest): Request object with source type and parameters.

            Returns:
                CreateSourceResponse: Response with new source ID or error.
            """
            try:
                source_metadata: SourceMetadata = await self._manager.create_new_source(
                    image_provider=payload.image_provider,
                    provider_params=payload.provider_params,
                    source_name=payload.source_name,
                )
                return CreateSourceResponse(status=JsonValues.SUCCESS, source=source_metadata)
            except Exception as e:
                return CreateSourceResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/sources/delete",
            tags=[ApiTags.SOURCES],
            operation_id="delete_sources",
            response_model=DeleteSourceResponse,
        )
        async def delete_sources_api(payload: DeleteSourcePayload) -> DeleteSourceResponse:
            """
            API endpoint to delete sources.
            """
            try:
                await self._manager.delete_sources(payload.source_uuids)
                return DeleteSourceResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return DeleteSourceResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/sources/get",
            response_model=GetSourcesResponse,
            tags=[ApiTags.SOURCES],
            operation_id="get_sources",
        )
        async def get_sources_api() -> GetSourcesResponse:
            """
            API endpoint to return a list of available sources
            Args:
                request (Request): The FastAPI request object.

            Returns:
                GetSourcesResponse: A JSON response containing the list of sources
            """
            try:
                sources: list[SourceMetadata] = await self._manager.get_avail_sources()
                sources_dict: dict[str, SourceMetadata] = {source.uuid: source for source in sources}
                return GetSourcesResponse(status=JsonValues.SUCCESS, sources=sources_dict)
            except Exception as e:
                logger.error(e, exc_info=True)
                return GetSourcesResponse(status=JsonValues.FAILURE, message=str(e), sources={})

        ################################################################################
        # Tags API
        ################################################################################

        @self._app.post("/api/tags/add", response_model=AddTagResponse, tags=[ApiTags.TAGS], operation_id="add_tag")
        async def add_tag_api(request: AddTagPayload) -> AddTagResponse:
            """
            API endpoint to add a new tag.
            """
            try:
                tag = await self._manager.create_new_tag(
                    request.name, request.color, kind=request.kind, exclusive_group=request.exclusive_group
                )
                return AddTagResponse(status=JsonValues.SUCCESS, tag=tag)
            except Exception as e:
                logger.error(e, exc_info=True)
                return AddTagResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/tags/delete", response_model=DeleteTagResponse, tags=[ApiTags.TAGS], operation_id="delete_tag"
        )
        async def delete_tag_api(request: DeleteTagPayload) -> DeleteTagResponse:
            """
            API endpoint to delete an existing tag.
            """
            try:
                await self._manager.delete_tag(request.tag_uuid)
                return DeleteTagResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return DeleteTagResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get("/api/tags/list", response_model=ListTagsResponse, tags=[ApiTags.TAGS], operation_id="list_tags")
        async def list_tags_api() -> ListTagsResponse:
            """
            API endpoint to return a list of tags with metadata.
            Args:
                request (Request): The FastAPI request object.

            Returns:
                ListTagsResponse: A response containing the list of tags.
            """
            try:
                tags = await self._manager.get_tags()
                tag_models = [TagMetadataModel.from_dataclass(tag) for tag in tags]
                return ListTagsResponse(status=JsonValues.SUCCESS, tags=tag_models)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListTagsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/tags/update", response_model=UpdateTagResponse, tags=[ApiTags.TAGS], operation_id="update_tag"
        )
        async def update_tag_api(request: UpdateTagPayload) -> UpdateTagResponse:
            """
            API endpoint to update an existing tag's name and/or color.
            """
            try:
                await self._manager.update_tag(tag_uuid=request.tag_uuid, name=request.name, color=request.color)
                return UpdateTagResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return UpdateTagResponse(status=JsonValues.FAILURE, message=str(e))

        ################################################################################
        # BBox API
        ################################################################################

        @self._app.get(
            "/api/bboxes/get",
            response_model=GetBoundingBoxInfoResponse,
            tags=[ApiTags.IMAGES],
            operation_id="get_bbox_info",
        )
        async def get_bbox_info(payload: GetBoundingBoxInfoPayload) -> GetBoundingBoxInfoResponse:
            metadata: List[BoundingBoxMetadata] = await self._manager.get_bboxes_info(payload.uuids)
            metadata_map = {m.uuid: m for m in metadata}
            if not metadata:
                raise HTTPException(status_code=404, detail="Bounding box not found")
            return GetBoundingBoxInfoResponse(status=JsonValues.SUCCESS, metadata=metadata_map)

        ################################################################################
        # Images API
        ################################################################################

        @self._app.post(
            "/api/images/metadata/get",
            response_model=GetImageMetadataResponse,
            tags=[ApiTags.IMAGES],
            operation_id="get_image_metadata",
        )
        async def get_image_metadata(payload: GetImageMetadataPayload) -> GetImageMetadataResponse:
            metadata = await self._manager.get_image_metadata(payload.image_path)
            if metadata is None:
                return GetImageMetadataResponse(status=JsonValues.FAILURE, message="Image not found")

            return GetImageMetadataResponse(metadata=metadata)

        @self._app.post(
            "/api/images/metadata/update",
            response_model=UpdateMetadataResponse,
            tags=[ApiTags.IMAGES],
            operation_id="update_image_metadata",
        )
        async def update_image_metadata(payload: UpdateMetadataPayload) -> UpdateMetadataResponse:
            image_path = payload.image_path
            metadata = await self._manager.get_image_metadata(image_path)

            if metadata is None:
                return UpdateMetadataResponse(status=JsonValues.FAILURE, message="Image not found")

            # Build new bounding box list from input
            new_boxes: List[BoundingBoxMetadata] = []
            for b in payload.boxes:
                new_boxes.append(
                    BoundingBoxMetadata(
                        uuid=b.uuid,
                        class_uuid=b.class_uuid,
                        tag_uuids=b.tag_uuids,
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
            model = ImageMetadataModel.from_dataclass(metadata)
            return UpdateMetadataResponse(metadata=model)

        @self._app.post(
            "/api/images/list",
            response_model=ListFilesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="list_images",
        )
        async def list_images(payload: ListFilesPayload) -> ListFilesResponse:
            """
            API endpoint to return a list of files.

            Query Parameters:
                path (str): Relative path under the root directory (default: "/").
                glob (str): Glob pattern to filter files (default: "*").
                recursive (bool): Whether to search directories recursively (default: False).

            Returns:
                ListFilesResponse: A response containing the list of files.
            """
            try:
                file_list: List[FileEntry] = await self._manager.list_files(
                    rel_path=payload.path, patterns=payload.pattern, recursive=payload.recursive
                )
                return ListFilesResponse(files=file_list)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListFilesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/get",
            response_model=GetFileResponse,
            tags=[ApiTags.IMAGES],
            operation_id="get_image",
        )
        async def get_file(
            payload: GetFilePayload,
        ) -> GetFileResponse:
            """
            Retrieve a file either as base64 JSON

            Args:
                path (str): Relative file path.

            Returns:
                GetFileResponse: A response containing the file data.
            """
            try:
                result = await self._manager.open_file(payload.path)
                if not result:
                    return GetFileResponse(
                        status=JsonValues.FAILURE, message=f"File {payload.path} not found", filename=payload.path
                    )

                file_obj, filename = result
                content = await file_obj.read()
                await file_obj.close()

                mime_type, _ = mimetypes.guess_type(filename)
                mime_type = mime_type or "application/octet-stream"

                metadata: ImageMetadata | None = await self._manager.get_image_metadata(payload.path)
                model = ImageMetadataModel.from_dataclass(metadata) if metadata else None
                encoded = base64.b64encode(content).decode("utf-8")
                return GetFileResponse(
                    status=JsonValues.SUCCESS,
                    filename=filename,
                    mime_type=mime_type,
                    image_base64=encoded,
                    metadata=model,
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.exception("Error retrieving file")
                return GetFileResponse(status=JsonValues.FAILURE, message=str(e), filename=payload.path)
