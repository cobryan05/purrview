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
from pydantic import ConfigDict, Field
from pydantic.alias_generators import to_camel

from apps.db_server.manager import Manager
from apps.helpers.consts import ApiTags, JsonKeys, JsonValues
from apps.helpers.db.types import ImageRecordRead, LabelUpdate, Tag, TagBase, TagUpdate, Label, LabelUpdate
from apps.helpers.db.types import BBoxUpdate, ImageRecordUpdate
from apps.helpers.types import (
    Base64Image,
    BoundingBoxMetadataModel,
    FileEntry,
    Payload,
    SourceMetadata,
    SourceMetadataModel,
    StatusResponse,
)


class ApiPayload(Payload):
    pass


class UpdateMetadataPayload(ApiPayload):
    image_path: str
    boxes: List[BoundingBoxMetadataModel] = Field(default_factory=list)
    extra: Optional[Any] = None


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


########
# Labels
########
class AddLabelPayload(ApiPayload):
    name: str
    color: str
    parent_uuid: Optional[str] = None


class AddLabelResponse(StatusResponse):
    label: Optional[Label] = None


class DeleteLabelPayload(ApiPayload):
    label_uuid: str


class DeleteLabelResponse(StatusResponse):
    pass


class ListLabelsPayload(ApiPayload):
    pass


class ListLabelsResponse(StatusResponse):
    labels: List[Label] = Field(default_factory=list)


class UpdateLabelPayload(ApiPayload):
    label_uuid: str
    name: Optional[str] = None
    color: Optional[str] = None


class UpdateLabelResponse(StatusResponse):
    pass


########
# Tags
########
class AddTagPayload(TagBase):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)  # type: ignore[assignment]


class AddTagResponse(StatusResponse):
    tag: Optional[Tag] = None


class DeleteTagPayload(ApiPayload):
    tag_uuid: str


class DeleteTagResponse(StatusResponse):
    pass


class ListTagsPayload(ApiPayload):
    pass


class ListTagsResponse(StatusResponse):
    tags: List[Tag] = Field(default_factory=list)


class UpdateTagPayload(ApiPayload):
    tag_uuid: str
    data: TagUpdate


class UpdateTagResponse(StatusResponse):
    pass


########
# Image Providers
########
class GetImageProviderSchemaPayload(ApiPayload):
    image_provider: str


class GetImageProviderSchemaResponse(StatusResponse):
    image_provider_schema: Optional[dict] = None


class ListImageProvidersPayload(ApiPayload):
    pass


class ListImageProvidersResponse(StatusResponse):
    image_providers: List[str] = Field(default_factory=list)


########
# Sources
########
class CreateSourcePayload(ApiPayload):
    source_name: str
    image_provider: str
    provider_params: dict


class CreateSourceResponse(StatusResponse):
    source: Optional[SourceMetadataModel] = None


class DeleteSourcePayload(ApiPayload):
    source_uuids: List[str]


class DeleteSourceResponse(StatusResponse):
    pass


class GetSourcesPayload(ApiPayload):
    pass


class GetSourcesResponse(StatusResponse):
    sources: dict[str, SourceMetadataModel] = Field(default_factory=dict)


class UpdateSourcePayload(ApiPayload):
    source_name: str
    source_uuid: str
    image_provider: str
    provider_params: dict


class UpdateSourceResponse(StatusResponse):
    source: Optional[SourceMetadataModel] = None


class ListFilesPayload(ApiPayload):
    path: str = "/"
    pattern: Optional[str] = "*"
    recursive: Optional[bool] = False


class ListFilesResponse(StatusResponse):
    files: List[FileEntry] = []


class GetImageMetadataPayload(ApiPayload):
    image_path: str


class GetImageMetadataResponse(StatusResponse):
    image: Optional[ImageRecordRead] = None


class UpdateMetadataResponse(StatusResponse):
    image: Optional[ImageRecordRead] = None


class GetFilePayload(ApiPayload):
    path: str


class GetFileResponse(StatusResponse):
    filename: str
    mime_type: Optional[str] = None
    image_base64: Base64Image = None
    image: Optional[ImageRecordRead] = None


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
        self._app.mount("/static", StaticFiles(directory="/app/static"), name="static")

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
        @self._app.post(
            "/api/labels",
            response_model=AddLabelResponse,
            tags=[ApiTags.LABELS],
            operation_id="add_label",
        )
        async def add_label_api(request: AddLabelPayload) -> AddLabelResponse:
            """
            API endpoint to add a new label.
            """
            try:
                label = await self._manager.create_new_label(
                    request.name, request.color, parent_uuid=request.parent_uuid
                )

                return AddLabelResponse(status=JsonValues.SUCCESS, label=label)
            except Exception as e:
                logger.error(e, exc_info=True)
                return AddLabelResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/labels/{label_uuid}",
            response_model=DeleteLabelResponse,
            tags=[ApiTags.LABELS],
            operation_id="delete_label",
        )
        async def delete_labels_api(label_uuid: str) -> DeleteLabelResponse:
            """
            API endpoint to delete an existing label.
            """
            try:
                await self._manager.delete_label(label_uuid)
                return DeleteLabelResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return DeleteLabelResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/labels",
            response_model=ListLabelsResponse,
            tags=[ApiTags.LABELS],
            operation_id="list_labels",
        )
        async def list_labels_api() -> ListLabelsResponse:
            """
            API endpoint to return a list of labels with metadata.
            """
            try:
                label_list = await self._manager.get_labels()
                return ListLabelsResponse(status=JsonValues.SUCCESS, labels=label_list)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListLabelsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/labels/{label_uuid}",
            response_model=UpdateLabelResponse,
            tags=[ApiTags.LABELS],
            operation_id="update_label",
        )
        async def update_label_api(label_uuid: str, request: LabelUpdate) -> UpdateLabelResponse:
            """
            API endpoint to update an existing label's name and/or color.
            """
            try:
                await self._manager.update_label(
                    label_uuid=label_uuid,
                    name=request.name,
                    color=request.color,
                )
                return UpdateLabelResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return UpdateLabelResponse(status=JsonValues.FAILURE, message=str(e))

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
                return ListTagsResponse(status=JsonValues.SUCCESS, tags=tags)
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
                await self._manager.update_tag(tag_uuid=request.tag_uuid, update_data=request.data)
                return UpdateTagResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return UpdateTagResponse(status=JsonValues.FAILURE, message=str(e))

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

            return GetImageMetadataResponse(image=metadata)

        @self._app.post(
            "/api/images/metadata/update",
            response_model=UpdateMetadataResponse,
            tags=[ApiTags.IMAGES],
            operation_id="update_image_metadata",
        )
        async def update_image_metadata(payload: UpdateMetadataPayload) -> UpdateMetadataResponse:
            image_path = payload.image_path
            existing = await self._manager.get_image_metadata(image_path)
            if existing is None:
                return UpdateMetadataResponse(status=JsonValues.FAILURE, message="Image not found")

            update = ImageRecordUpdate(
                bboxes=[
                    BBoxUpdate(
                        uuid=b.uuid,
                        label_uuid=b.label_uuid,
                        x=b.x,
                        y=b.y,
                        width=b.width,
                        height=b.height,
                        tag_uuids=b.tag_uuids or [],
                    )
                    for b in payload.boxes
                    if b.label_uuid
                ]
            )

            updated = await self._manager.update_image_metadata(image_path, update)
            return UpdateMetadataResponse(image=updated)

        @self._app.get(
            "/api/images/list",
            response_model=ListFilesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="list_images",
        )
        async def list_images(path: str = "/", pattern: str = "*", recursive: bool = False) -> ListFilesResponse:
            """
            List files and directories.
            Usage: /api/images/list?path=/data&pattern=*.jpg&recursive=true
            """
            try:
                file_list = await self._manager.list_files(rel_path=path, patterns=[pattern], recursive=recursive)
                return ListFilesResponse(status=JsonValues.SUCCESS, files=file_list)
            except Exception as e:
                logger.exception("Failed to list files")
                return ListFilesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/images/{image_path:path}/file",
            response_model=GetFileResponse,
            tags=[ApiTags.IMAGES],
            operation_id="get_image_file",
        )
        async def get_file(image_path: str, include_binary: bool = True) -> GetFileResponse:
            """
            Retrieve a file and its associated metadata in a single request.
            The image data is automatically encoded to base64 by the response model.
            """
            try:
                result = await self._manager.open_file(image_path)
                if not result:
                    return GetFileResponse(
                        status=JsonValues.FAILURE, message=f"File {image_path} not found", filename=image_path
                    )

                file_obj, filename = result
                if include_binary:
                    content = await file_obj.read()  # Raw bytes
                    await file_obj.close()
                else:
                    content = None

                mime_type, _ = mimetypes.guess_type(filename)
                mime_type = mime_type or "application/octet-stream"

                image_record = await self._manager.get_image_metadata(image_path)

                return GetFileResponse(
                    status=JsonValues.SUCCESS,
                    filename=filename,
                    mime_type=mime_type,
                    image_base64=content,
                    image=image_record,
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"Error retrieving file: {image_path}")
                return GetFileResponse(status=JsonValues.FAILURE, message=str(e), filename=image_path)
