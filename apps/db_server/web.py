"""Web API for DB Server"""

import logging
import mimetypes
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from sqlalchemy.exc import IntegrityError

from apps.db_server.manager import Manager
from apps.helpers.consts import ApiTags, JsonKeys, JsonValues, TaskStatus
from apps.helpers.db.types import (
    ApiModel,
    BBoxUpdate,
    DatasetCreate,
    DatasetFilterCriteria,
    DatasetRead,
    DatasetUpdate,
    ImageLabelRead,
    ImageLabelStatus,
    ImageRecordRead,
    ImageRecordUpdate,
    ImageSearchParams,
    Label,
    LabelRead,
    LabelType,
    LabelUpdate,
    ModelConfigCreate,
    ModelConfigRead,
    ModelConfigUpdate,
    Source,
    SourceRead,
    Tag,
    TagBase,
    TagRead,
    TagUpdate,
    TaskConfigRead,
    TaskInstanceRead,
    TaskParams,
    TaskResultData,
    TaskResumeData,
)
from apps.helpers.types import (
    Base64Image,
    BoundingBoxMetadataModel,
    FileEntryRead,
    Payload,
    StatusResponse,
)


class ApiPayload(Payload):
    pass


class UpdateMetadataPayload(ApiPayload):
    image_path: str
    boxes: List[BoundingBoxMetadataModel] = Field(default_factory=list)
    extra: Optional[Any] = None


class BboxPatchPayload(ApiPayload):
    label_uuid: Optional[str] = None
    tag_uuids: List[str] = Field(default_factory=list)


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
    label_type: LabelType = LabelType.STANDARD


class AddLabelResponse(StatusResponse):
    label: Optional[LabelRead] = None


class DeleteLabelPayload(ApiPayload):
    label_uuid: str


class DeleteLabelResponse(StatusResponse):
    pass


class ListLabelsPayload(ApiPayload):
    pass


class ListLabelsResponse(StatusResponse):
    labels: List[LabelRead] = Field(default_factory=list)


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
    uuid: Optional[str] = None


class AddTagResponse(StatusResponse):
    tag: Optional[TagRead] = None


class DeleteTagPayload(ApiPayload):
    tag_uuid: str


class DeleteTagResponse(StatusResponse):
    pass


class ListTagsPayload(ApiPayload):
    pass


class ListTagsResponse(StatusResponse):
    tags: List[TagRead] = Field(default_factory=list)


class UpdateTagPayload(ApiPayload):
    tag_uuid: str
    data: TagUpdate


class UpdateTagResponse(StatusResponse):
    pass


class ClearOrphanGeneratedTagsResponse(StatusResponse):
    deleted_count: int = 0


########
# User Settings
########
class SettingResponse(StatusResponse):
    key: str
    value: Optional[str] = None


class SetSettingPayload(ApiPayload):
    value: str


########
# Image Reconciliation
########
class ImageStub(BaseModel):
    uuid: str
    filename: str


class ListImageStubsResponse(StatusResponse):
    stubs: List[ImageStub] = Field(default_factory=list)


class ReconcileImagePayload(ApiPayload):
    rel_path: str
    sync_mode: str = "merge"


class ReconcileImageResponse(StatusResponse):
    action: str = ""
    rel_path: str = ""
    reason: Optional[str] = None


class PruneImagesPayload(ApiPayload):
    known_disk_uuids: List[str] = Field(default_factory=list)


class PruneImagesResponse(StatusResponse):
    deleted_count: int = 0


class DeleteImagesByDirectoryPayload(ApiPayload):
    directories: List[str] = Field(default_factory=list)


class DeleteImagesByDirectoryResponse(StatusResponse):
    deleted_count: int = 0


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
    source: Optional[SourceRead] = None


class DeleteSourcePayload(ApiPayload):
    source_uuids: List[str]


class DeleteSourceResponse(StatusResponse):
    pass


class GetSourcesPayload(ApiPayload):
    pass


class GetSourcesResponse(StatusResponse):
    sources: dict[str, SourceRead] = Field(default_factory=dict)


class UpdateSourcePayload(ApiPayload):
    source_name: str
    source_uuid: str
    image_provider: str
    provider_params: dict


class UpdateSourceResponse(StatusResponse):
    source: Optional[SourceRead] = None


class ListFilesPayload(ApiPayload):
    path: str = "/"
    pattern: Optional[str] = "*"
    recursive: Optional[bool] = False


class ListFilesResponse(StatusResponse):
    files: List[FileEntryRead] = []
    has_more: bool = False


class CreateDirectoryPayload(ApiPayload):
    path: str


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


class SetImageLabelPayload(ApiPayload):
    image_path: str
    label_uuid: str
    status: ImageLabelStatus


class SetImageLabelResponse(StatusResponse):
    pass


class GetImageLabelsResponse(StatusResponse):
    labels: List[ImageLabelRead] = Field(default_factory=list)


class ReviewQueueResponse(StatusResponse):
    images: List[ImageRecordRead] = Field(default_factory=list)
    has_more: bool = False
    total_count: int = 0


class InvalidateThumbnailsPayload(ApiPayload):
    bbox_uuids: List[str] = Field(default_factory=list)


########
# Task Configs
########
class CreateTaskConfigPayload(ApiPayload):
    name: str
    typename: str
    params: TaskParams = Field(default_factory=dict)
    description: Optional[str] = None


class TaskConfigReadResponse(StatusResponse):
    config: Optional[TaskConfigRead] = None


class ListTaskConfigsResponse(StatusResponse):
    configs: dict[str, TaskConfigRead] = Field(default_factory=dict)


class UpdateTaskConfigPayload(ApiPayload):
    name: Optional[str] = None
    typename: Optional[str] = None
    params: Optional[TaskParams] = None
    description: Optional[str] = None
    marked_for_delete: Optional[bool] = None


class DeleteTaskConfigsPayload(ApiPayload):
    config_uuids: List[str]


########
# Task Instances
########
class CreateTaskInstancePayload(ApiPayload):
    config_uuid: str


class TaskInstanceReadResponse(StatusResponse):
    instance: Optional[TaskInstanceRead] = None


class ListTaskInstancesResponse(StatusResponse):
    instances: List[TaskInstanceRead] = Field(default_factory=list)


class DeleteTaskInstancesPayload(ApiPayload):
    task_uuids: List[str]


class SetTaskStatusPayload(ApiPayload):
    status: TaskStatus


class SetTaskResultPayload(ApiPayload):
    result: TaskResultData = Field(default_factory=dict)


class SetTaskResumeDataPayload(ApiPayload):
    resume_data: TaskResumeData = Field(default_factory=dict)


class SetTaskTimestampsPayload(ApiPayload):
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    log_json: Optional[List[str]] = None


class GetTaskResultsPayload(ApiPayload):
    task_uuids: List[str]


class GetTaskResultsResponse(StatusResponse):
    results: Dict[str, TaskResultData] = Field(default_factory=dict)


class ListDatasetsResponse(StatusResponse):
    datasets: List[DatasetRead] = Field(default_factory=list)


class DatasetResponse(StatusResponse):
    dataset: Optional[DatasetRead] = None


class DatasetImagesResponse(StatusResponse):
    images: List[ImageRecordRead] = Field(default_factory=list)
    has_more: bool = False


class DatasetLabelsResponse(StatusResponse):
    labels: List[LabelRead] = Field(default_factory=list)


class DatasetPreviewPayload(ApiModel):
    """Request body for the unsaved-preview endpoint."""

    filter_json: DatasetFilterCriteria = Field(default_factory=DatasetFilterCriteria)
    label_uuids: List[str] = Field(default_factory=list)


class ListModelTypenamesResponse(StatusResponse):
    typenames: List[str] = Field(default_factory=list)


class ListModelConfigsResponse(StatusResponse):
    configs: List[ModelConfigRead] = Field(default_factory=list)


class ModelConfigResponse(StatusResponse):
    config: Optional[ModelConfigRead] = None


########
# Perceptual Hash
########
class FindSimilarImagesPayload(ApiPayload):
    phash: str
    threshold: int = 5


class SimilarImageMatch(ApiModel):
    image_uuid: str
    distance: int


class FindSimilarImagesResponse(StatusResponse):
    matches: List[SimilarImageMatch] = Field(default_factory=list)


class FindSimilarBboxesPayload(ApiPayload):
    phash: str
    threshold: int = 8
    label_uuid: Optional[str] = None


class SimilarBboxMatch(ApiModel):
    bbox_uuid: str
    image_uuid: Optional[str] = None
    label_uuid: str
    distance: int


class FindSimilarBboxesResponse(StatusResponse):
    matches: List[SimilarBboxMatch] = Field(default_factory=list)


class FindSimilarImagesByUuidPayload(ApiPayload):
    image_uuid: str
    threshold: int = 5


class ImageStubsNeedingPhashPayload(ApiPayload):
    image_hex_len: int
    bbox_hex_len: int


class FindSimilarBboxesByUuidPayload(ApiPayload):
    bbox_uuid: str
    threshold: int = 8
    label_uuid: Optional[str] = None


class FindSimilarBboxesByCoordsPayload(ApiPayload):
    image_uuid: str
    x: float
    y: float
    w: float
    h: float
    threshold: int = 8
    label_uuid: Optional[str] = None


class RebuildHashIndexResponse(StatusResponse):
    image_count: int = 0
    bbox_count: int = 0
    duration_ms: float = 0.0


class BatchByUuidsPayload(ApiPayload):
    image_uuids: List[str]


class BatchImagesResponse(StatusResponse):
    images: List[dict] = Field(default_factory=list)


class DeleteImageResponse(StatusResponse):
    image_uuid: Optional[str] = None
    filename: Optional[str] = None


class BboxDedupItem(ApiModel):
    phash: str
    label_uuid: Optional[str] = None
    threshold: int = 8


class CheckDedupBulkPayload(ApiPayload):
    image_phash: str
    image_threshold: int = 5
    bboxes: List[BboxDedupItem] = Field(default_factory=list)


class CheckDedupBulkResponse(StatusResponse):
    image_is_dup: bool = False
    bbox_flags: List[bool] = Field(default_factory=list)


class RecomputePhashPayload(ApiPayload):
    image_hash_size: Optional[int] = None
    bbox_hash_size: Optional[int] = None


class RecomputePhashResponse(StatusResponse):
    image_phash: Optional[str] = None
    bbox_count: int = 0


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
        self._register_model_config_routes()
        self._register_exception()

    @staticmethod
    def _unique_constraint_msg(e: IntegrityError, entity: str) -> str:
        """Return a friendly message for unique-constraint violations, raw error otherwise."""
        return f'A {entity} with that name already exists.' if 'unique' in str(e).lower() else str(e)

    def _register_exception(self):
        @self._app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):

            exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
            logger.error("%s %s", request, exc_str)
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

        @self._app.get("/server-config", operation_id="server_config", response_class=JSONResponse)
        async def server_config(request: Request):
            try:
                server_config = await self._manager.get_server_config()
                response_data = {JsonKeys.STATUS: JsonValues.SUCCESS, JsonKeys.CONFIG: server_config}
            except Exception as e:
                response_data = {JsonKeys.STATUS: JsonValues.FAILURE, JsonKeys.MESSAGE: str(e)}
            return JSONResponse(content=response_data)

        @self._app.get("/status", operation_id="db_server_status", response_class=JSONResponse)
        async def db_server_status():
            try:
                status = await self._manager.get_status()
                return JSONResponse(content=status)
            except Exception as e:
                return JSONResponse(content={"service": "db_server", "status": "error", "message": str(e)})

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
                    request.name, request.color, parent_uuid=request.parent_uuid, label_type=request.label_type
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
                    label_type=request.label_type,
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
                source: Source = await self._manager.update_source(
                    payload.source_uuid,
                    image_provider=payload.image_provider,
                    provider_params=payload.provider_params,
                    source_name=payload.source_name,
                )
                return UpdateSourceResponse(status=JsonValues.SUCCESS, source=source)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return UpdateSourceResponse(status=JsonValues.FAILURE, message=self._unique_constraint_msg(e, 'source'))
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
                source: Source = await self._manager.create_new_source(
                    image_provider=payload.image_provider,
                    provider_params=payload.provider_params,
                    source_name=payload.source_name,
                )
                return CreateSourceResponse(status=JsonValues.SUCCESS, source=source)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return CreateSourceResponse(status=JsonValues.FAILURE, message=self._unique_constraint_msg(e, 'source'))
            except Exception as e:
                logger.error(e, exc_info=True)
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
                sources: list[Source] = await self._manager.get_avail_sources()
                sources_dict: dict[str, Source] = {source.uuid: source for source in sources}
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
                    request.name,
                    request.color,
                    icon=request.icon,
                    kind=request.kind,
                    exclusive_group=request.exclusive_group,
                    uuid=request.uuid,
                )
                return AddTagResponse(status=JsonValues.SUCCESS, tag=tag)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return AddTagResponse(status=JsonValues.FAILURE, message=self._unique_constraint_msg(e, 'tag'))
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
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return UpdateTagResponse(status=JsonValues.FAILURE, message=self._unique_constraint_msg(e, 'tag'))
            except Exception as e:
                logger.error(e, exc_info=True)
                return UpdateTagResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/tags/clear-orphan-generated",
            response_model=ClearOrphanGeneratedTagsResponse,
            tags=[ApiTags.TAGS],
            operation_id="clear_orphan_generated_tags",
        )
        async def clear_orphan_generated_tags_api() -> ClearOrphanGeneratedTagsResponse:
            """Delete TASK_GENERATED tags that have no associated bounding boxes."""
            try:
                count = await self._manager.clear_orphan_generated_tags()
                return ClearOrphanGeneratedTagsResponse(status=JsonValues.SUCCESS, deleted_count=count)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ClearOrphanGeneratedTagsResponse(status=JsonValues.FAILURE, message=str(e))

        ################################################################################
        # User Settings API
        ################################################################################

        @self._app.get(
            "/api/settings/{key}", response_model=SettingResponse, tags=[ApiTags.SETTINGS], operation_id="get_setting"
        )
        async def get_setting_api(key: str) -> SettingResponse:
            """Return the stored value for a user-settings key."""
            try:
                value = await self._manager.get_setting(key)
                if value is None:
                    raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
                return SettingResponse(status=JsonValues.SUCCESS, key=key, value=value)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(e, exc_info=True)
                return SettingResponse(status=JsonValues.FAILURE, key=key, message=str(e))

        @self._app.put(
            "/api/settings/{key}", response_model=SettingResponse, tags=[ApiTags.SETTINGS], operation_id="set_setting"
        )
        async def set_setting_api(key: str, request: SetSettingPayload) -> SettingResponse:
            """Upsert a user-settings key-value pair."""
            try:
                await self._manager.set_setting(key, request.value)
                return SettingResponse(status=JsonValues.SUCCESS, key=key, value=request.value)
            except Exception as e:
                logger.error(e, exc_info=True)
                return SettingResponse(status=JsonValues.FAILURE, key=key, message=str(e))

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
                        metadata_json=b.extra or {},
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
        async def list_images(
            path: str = "/",
            pattern: str = "*",
            recursive: bool = False,
            include_db_status: bool = False,
            offset: int = Query(default=0, ge=0),
            limit: int = Query(default=200, ge=1, le=2000),
        ) -> ListFilesResponse:
            """
            List files and directories.
            Usage: /api/images/list?path=/data&pattern=*.jpg&recursive=true&include_db_status=true&offset=0&limit=200
            """
            try:
                file_list, has_more = await self._manager.list_files(
                    rel_path=path, patterns=[pattern], recursive=recursive,
                    include_db_status=include_db_status, offset=offset, limit=limit,
                )
                return ListFilesResponse(status=JsonValues.SUCCESS, files=file_list, has_more=has_more)
            except Exception as e:
                logger.exception("Failed to list files")
                return ListFilesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/mkdir",
            response_model=StatusResponse,
            tags=[ApiTags.IMAGES],
            operation_id="create_directory",
        )
        async def create_directory(payload: CreateDirectoryPayload) -> StatusResponse:
            """Create a directory under FILES_ROOT."""
            try:
                await self._manager.create_directory(payload.path)
                return StatusResponse(status=JsonValues.SUCCESS)
            except ValueError as e:
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))
            except Exception as e:
                logger.exception("Failed to create directory")
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

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
                try:
                    content = await file_obj.read() if include_binary else None
                finally:
                    await file_obj.close()

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

        ################################################################################
        # Image Labels API
        ################################################################################

        @self._app.post(
            "/api/images/labels/set",
            response_model=SetImageLabelResponse,
            tags=[ApiTags.IMAGES],
            operation_id="set_image_label",
        )
        async def set_image_label(payload: SetImageLabelPayload) -> SetImageLabelResponse:
            if payload.status == ImageLabelStatus.PRESENT:
                return SetImageLabelResponse(
                    status=JsonValues.FAILURE,
                    message="PRESENT status cannot be set directly; it is derived from verified bboxes",
                )
            ok = await self._manager.set_image_label(payload.image_path, payload.label_uuid, payload.status)
            if not ok:
                return SetImageLabelResponse(status=JsonValues.FAILURE, message="Image not found")
            return SetImageLabelResponse()

        @self._app.get(
            "/api/images/{image_path:path}/labels",
            response_model=GetImageLabelsResponse,
            tags=[ApiTags.IMAGES],
            operation_id="get_image_labels",
        )
        async def get_image_labels(image_path: str) -> GetImageLabelsResponse:
            labels = await self._manager.get_image_labels(image_path)
            if labels is None:
                return GetImageLabelsResponse(status=JsonValues.FAILURE, message="Image not found")
            return GetImageLabelsResponse(labels=labels)

        @self._app.post(
            "/api/images/review",
            response_model=ReviewQueueResponse,
            tags=[ApiTags.IMAGES],
            operation_id="get_review_queue",
        )
        async def get_review_queue(params: ImageSearchParams) -> ReviewQueueResponse:
            """Search the image/bbox store with flexible filters and cursor-based pagination.

            Defaults to showing images with unverified bboxes, newest first.
            Pass show_all_bboxes=true to bypass the tag filter entirely.
            """
            try:
                images, has_more, total_count = await self._manager.get_review_queue(params)
                return ReviewQueueResponse(images=images, has_more=has_more, total_count=total_count)
            except Exception as e:
                logger.exception("Failed to fetch review queue")
                return ReviewQueueResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/images/bbox/{bbox_uuid}",
            tags=[ApiTags.IMAGES],
            operation_id="update_bbox",
        )
        async def update_bbox(bbox_uuid: str, payload: BboxPatchPayload) -> Response:
            ok = await self._manager.update_single_bbox(bbox_uuid, payload.label_uuid, payload.tag_uuids)
            if not ok:
                raise HTTPException(status_code=404, detail="Bbox not found")
            return Response(status_code=204)

        @self._app.post(
            "/api/images/bbox/thumbnails/invalidate",
            tags=[ApiTags.IMAGES],
            operation_id="invalidate_bbox_thumbnails",
        )
        async def invalidate_bbox_thumbnails(payload: InvalidateThumbnailsPayload) -> StatusResponse:
            await self._manager.invalidate_bbox_thumbnails(payload.bbox_uuids)
            return StatusResponse()

        @self._app.delete(
            "/api/images/bbox/{bbox_uuid}",
            tags=[ApiTags.IMAGES],
            operation_id="delete_bbox",
        )
        async def delete_bbox(bbox_uuid: str) -> Response:
            ok = await self._manager.delete_single_bbox(bbox_uuid)
            if not ok:
                raise HTTPException(status_code=404, detail="Bbox not found")
            return Response(status_code=204)

        # NOTE: bbox thumbnail route must come BEFORE the generic image thumbnail route
        # because {path:path} is greedy and would otherwise capture the /bbox/uuid/thumbnail suffix.
        @self._app.get(
            "/api/images/{image_path:path}/bbox/{bbox_uuid}/thumbnail",
            tags=[ApiTags.IMAGES],
            operation_id="get_bbox_thumbnail",
        )
        async def get_bbox_thumbnail(image_path: str, bbox_uuid: str, width: int = 200) -> Response:
            png_bytes = await self._manager.get_bbox_thumbnail(bbox_uuid, width=width)
            if png_bytes is None:
                raise HTTPException(status_code=404, detail="Thumbnail not available")
            return Response(content=png_bytes, media_type="image/png")

        @self._app.get(
            "/api/images/by-uuid/{image_uuid}/thumbnail",
            tags=[ApiTags.IMAGES],
            operation_id="get_image_thumbnail_by_uuid",
        )
        async def get_image_thumbnail_by_uuid(image_uuid: str, width: int = 640) -> Response:
            filename = await self._manager.get_image_filename_by_uuid(image_uuid)
            if filename is None:
                raise HTTPException(status_code=404, detail="Image not found")
            png_bytes = await self._manager.get_image_thumbnail(filename, width=width)
            if png_bytes is None:
                raise HTTPException(status_code=404, detail="Thumbnail not available")
            return Response(content=png_bytes, media_type="image/png")

        @self._app.get(
            "/api/images/bboxes/{bbox_uuid}/thumbnail",
            tags=[ApiTags.IMAGES],
            operation_id="get_bbox_thumbnail_by_uuid",
        )
        async def get_bbox_thumbnail_by_uuid(bbox_uuid: str, width: int = 200) -> Response:
            png_bytes = await self._manager.get_bbox_thumbnail(bbox_uuid, width=width)
            if png_bytes is None:
                raise HTTPException(status_code=404, detail="Thumbnail not available")
            return Response(content=png_bytes, media_type="image/png")

        @self._app.get(
            "/api/images/{image_path:path}/thumbnail",
            tags=[ApiTags.IMAGES],
            operation_id="get_image_thumbnail",
        )
        async def get_image_thumbnail(image_path: str, width: int = 640) -> Response:
            png_bytes = await self._manager.get_image_thumbnail(image_path, width=width)
            if png_bytes is None:
                raise HTTPException(status_code=404, detail="Image not found")
            return Response(content=png_bytes, media_type="image/png")

        ################################################################################
        # Image Reconciliation API
        ################################################################################

        @self._app.get(
            "/api/images/by-uuid/{image_uuid}/meta",
            tags=[ApiTags.IMAGES],
            operation_id="get_image_meta_by_uuid",
        )
        async def get_image_meta_by_uuid(image_uuid: str) -> JSONResponse:
            """Return {uuid, filename} for a single image by UUID."""
            filename = await self._manager.get_image_filename_by_uuid(image_uuid)
            if filename is None:
                raise HTTPException(status_code=404, detail="Image not found")
            return JSONResponse(content={"uuid": image_uuid, "filename": filename})

        @self._app.get(
            "/api/images/stubs",
            response_model=ListImageStubsResponse,
            tags=[ApiTags.IMAGES],
            operation_id="list_image_stubs",
        )
        async def list_image_stubs() -> ListImageStubsResponse:
            """Return uuid+filename for every image record (no pagination)."""
            try:
                raw = await self._manager.list_all_image_stubs()
                stubs = [ImageStub(uuid=s["uuid"], filename=s["filename"]) for s in raw]
                return ListImageStubsResponse(status=JsonValues.SUCCESS, stubs=stubs)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListImageStubsResponse(status=JsonValues.FAILURE, message=str(e))


        @self._app.post(
            "/api/images/stubs/needing-phash",
            response_model=ListImageStubsResponse,
            tags=[ApiTags.IMAGES],
            operation_id="list_image_stubs_needing_phash",
        )
        async def list_image_stubs_needing_phash(payload: ImageStubsNeedingPhashPayload) -> ListImageStubsResponse:
            """Return uuid+filename for images whose phash is absent or computed at a different size."""
            try:
                raw = await self._manager.list_image_stubs_needing_phash(payload.image_hex_len, payload.bbox_hex_len)
                stubs = [ImageStub(uuid=s["uuid"], filename=s["filename"]) for s in raw]
                return ListImageStubsResponse(status=JsonValues.SUCCESS, stubs=stubs)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListImageStubsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/reconcile",
            response_model=ReconcileImageResponse,
            tags=[ApiTags.IMAGES],
            operation_id="reconcile_image",
        )
        async def reconcile_image(payload: ReconcileImagePayload) -> ReconcileImageResponse:
            """Move-aware reconcile of a single image file against the DB."""
            try:
                result = await self._manager.reconcile_image(payload.rel_path, payload.sync_mode)
                return ReconcileImageResponse(
                    status=JsonValues.SUCCESS,
                    action=result.get("action", ""),
                    rel_path=result.get("rel_path", payload.rel_path),
                    reason=result.get("reason"),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return ReconcileImageResponse(
                    status=JsonValues.FAILURE,
                    message=str(e),
                    rel_path=payload.rel_path,
                )

        @self._app.post(
            "/api/images/prune",
            response_model=PruneImagesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="prune_missing_images",
        )
        async def prune_missing_images(payload: PruneImagesPayload) -> PruneImagesResponse:
            """Delete DB records for images whose file is missing and not accounted for by known_disk_uuids."""
            try:
                count = await self._manager.prune_missing_images(payload.known_disk_uuids)
                return PruneImagesResponse(status=JsonValues.SUCCESS, deleted_count=count)
            except Exception as e:
                logger.error(e, exc_info=True)
                return PruneImagesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/delete-by-directory",
            response_model=DeleteImagesByDirectoryResponse,
            tags=[ApiTags.IMAGES],
            operation_id="delete_images_by_directory",
        )
        async def delete_images_by_directory(
            payload: DeleteImagesByDirectoryPayload,
        ) -> DeleteImagesByDirectoryResponse:
            """Delete all DB records (and their bboxes) for images under the given directory paths."""
            try:
                count = await self._manager.delete_images_by_directory(payload.directories)
                return DeleteImagesByDirectoryResponse(status=JsonValues.SUCCESS, deleted_count=count)
            except Exception as e:
                logger.error(e, exc_info=True)
                return DeleteImagesByDirectoryResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/images/{image_uuid}",
            response_model=DeleteImageResponse,
            tags=[ApiTags.IMAGES],
            operation_id="delete_image",
        )
        async def delete_image_api(image_uuid: str) -> DeleteImageResponse:
            """Permanently delete an image: removes the DB record, image file, and JSON sidecar."""
            try:
                rel_path = await self._manager.delete_image_permanently(image_uuid)
                if rel_path is None:
                    return DeleteImageResponse(status=JsonValues.FAILURE, message="Image not found")
                return DeleteImageResponse(status=JsonValues.SUCCESS, image_uuid=image_uuid, filename=rel_path)
            except Exception as e:
                logger.error(e, exc_info=True)
                return DeleteImageResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/images/{image_uuid}/db-only",
            response_model=DeleteImageResponse,
            tags=[ApiTags.IMAGES],
            operation_id="delete_image_db_only",
        )
        async def delete_image_db_only_api(image_uuid: str) -> DeleteImageResponse:
            """Remove an image from the DB and hash index without deleting files from disk."""
            try:
                rel_path = await self._manager.delete_image_db_only(image_uuid)
                if rel_path is None:
                    return DeleteImageResponse(status=JsonValues.FAILURE, message="Image not found")
                return DeleteImageResponse(status=JsonValues.SUCCESS, image_uuid=image_uuid, filename=rel_path)
            except Exception as e:
                logger.error(e, exc_info=True)
                return DeleteImageResponse(status=JsonValues.FAILURE, message=str(e))

        ################################################################################
        # Perceptual Hash API
        ################################################################################

        @self._app.post(
            "/api/images/find-similar",
            response_model=FindSimilarImagesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="find_similar_images",
        )
        async def find_similar_images(payload: FindSimilarImagesPayload) -> FindSimilarImagesResponse:
            try:
                matches = self._manager.find_similar_images(payload.phash, payload.threshold)
                return FindSimilarImagesResponse(
                    status=JsonValues.SUCCESS,
                    matches=[SimilarImageMatch(image_uuid=uuid, distance=dist) for uuid, dist in matches],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return FindSimilarImagesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/bbox/find-similar",
            response_model=FindSimilarBboxesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="find_similar_bboxes",
        )
        async def find_similar_bboxes(payload: FindSimilarBboxesPayload) -> FindSimilarBboxesResponse:
            try:
                matches = await self._manager.find_similar_bboxes_enriched(
                    payload.phash, payload.label_uuid, payload.threshold
                )
                return FindSimilarBboxesResponse(
                    status=JsonValues.SUCCESS,
                    matches=[
                        SimilarBboxMatch(bbox_uuid=bbox_uuid, image_uuid=image_uuid, label_uuid=lbl, distance=dist)
                        for bbox_uuid, image_uuid, dist, lbl in matches
                    ],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return FindSimilarBboxesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/find-similar-by-image-uuid",
            response_model=FindSimilarImagesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="find_similar_images_by_uuid",
        )
        async def find_similar_images_by_uuid(payload: FindSimilarImagesByUuidPayload) -> FindSimilarImagesResponse:
            try:
                matches = await self._manager.find_similar_images_by_uuid(payload.image_uuid, payload.threshold)
                return FindSimilarImagesResponse(
                    status=JsonValues.SUCCESS,
                    matches=[SimilarImageMatch(image_uuid=uuid, distance=dist) for uuid, dist in matches],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return FindSimilarImagesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/bbox/find-similar-by-bbox-uuid",
            response_model=FindSimilarBboxesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="find_similar_bboxes_by_uuid",
        )
        async def find_similar_bboxes_by_uuid(payload: FindSimilarBboxesByUuidPayload) -> FindSimilarBboxesResponse:
            try:
                matches = await self._manager.find_similar_bboxes_by_uuid(
                    payload.bbox_uuid, payload.label_uuid, payload.threshold
                )
                return FindSimilarBboxesResponse(
                    status=JsonValues.SUCCESS,
                    matches=[
                        SimilarBboxMatch(bbox_uuid=bbox_uuid, image_uuid=image_uuid, label_uuid=lbl, distance=dist)
                        for bbox_uuid, image_uuid, dist, lbl in matches
                    ],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return FindSimilarBboxesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/bbox/find-similar-by-coords",
            response_model=FindSimilarBboxesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="find_similar_bboxes_by_coords",
        )
        async def find_similar_bboxes_by_coords(
            payload: FindSimilarBboxesByCoordsPayload,
        ) -> FindSimilarBboxesResponse:
            try:
                matches = await self._manager.find_similar_bboxes_by_coords(
                    payload.image_uuid,
                    payload.x,
                    payload.y,
                    payload.w,
                    payload.h,
                    payload.label_uuid,
                    payload.threshold,
                )
                return FindSimilarBboxesResponse(
                    status=JsonValues.SUCCESS,
                    matches=[
                        SimilarBboxMatch(bbox_uuid=bbox_uuid, image_uuid=image_uuid, label_uuid=lbl, distance=dist)
                        for bbox_uuid, image_uuid, dist, lbl in matches
                    ],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return FindSimilarBboxesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/batch-by-uuids",
            response_model=BatchImagesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="batch_images_by_uuids",
        )
        async def batch_images_by_uuids(payload: BatchByUuidsPayload) -> BatchImagesResponse:
            try:
                images = await self._manager.get_images_by_uuids(payload.image_uuids)
                return BatchImagesResponse(
                    status=JsonValues.SUCCESS,
                    images=[img.model_dump(by_alias=True) for img in images],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return BatchImagesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/dedup-check",
            response_model=CheckDedupBulkResponse,
            tags=[ApiTags.IMAGES],
            operation_id="check_dedup_bulk",
        )
        async def check_dedup_bulk(payload: CheckDedupBulkPayload) -> CheckDedupBulkResponse:
            try:
                result = self._manager.check_dedup_bulk(
                    payload.image_phash,
                    payload.image_threshold,
                    [
                        {"phash": b.phash, "label_uuid": b.label_uuid, "threshold": b.threshold}
                        for b in payload.bboxes
                    ],
                )
                return CheckDedupBulkResponse(
                    status=JsonValues.SUCCESS,
                    image_is_dup=result["image_is_dup"],
                    bbox_flags=result["bbox_flags"],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return CheckDedupBulkResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/phash/rebuild-index",
            response_model=RebuildHashIndexResponse,
            tags=[ApiTags.IMAGES],
            operation_id="rebuild_hash_index",
        )
        async def rebuild_hash_index() -> RebuildHashIndexResponse:
            try:
                stats = await self._manager.rebuild_hash_index()
                return RebuildHashIndexResponse(
                    status=JsonValues.SUCCESS,
                    image_count=stats["image_count"],
                    bbox_count=stats["bbox_count"],
                    duration_ms=stats["duration_ms"],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return RebuildHashIndexResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/{image_uuid}/phash/recompute",
            response_model=RecomputePhashResponse,
            tags=[ApiTags.IMAGES],
            operation_id="recompute_image_phash",
        )
        async def recompute_image_phash(
            image_uuid: str,
            payload: RecomputePhashPayload,
        ) -> RecomputePhashResponse:
            try:
                result = await self._manager.recompute_image_phash(
                    image_uuid,
                    image_hash_size=payload.image_hash_size,
                    bbox_hash_size=payload.bbox_hash_size,
                )
                if result is None:
                    return RecomputePhashResponse(status=JsonValues.FAILURE, message="Image not found")
                return RecomputePhashResponse(
                    status=JsonValues.SUCCESS,
                    image_phash=result["image_phash"],
                    bbox_count=result["bbox_count"],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return RecomputePhashResponse(status=JsonValues.FAILURE, message=str(e))

        ################################################################################
        # Task Configs API
        ################################################################################

        @self._app.post(
            "/api/task-configs",
            response_model=TaskConfigReadResponse,
            tags=[ApiTags.TASKS],
            operation_id="create_task_config",
        )
        async def create_task_config(payload: CreateTaskConfigPayload) -> TaskConfigReadResponse:
            try:
                config = await self._manager.create_task_config(
                    name=payload.name,
                    typename=payload.typename,
                    params=payload.params,
                    description=payload.description,
                )
                return TaskConfigReadResponse(status=JsonValues.SUCCESS, config=TaskConfigRead.model_validate(config))
            except Exception as e:
                logger.error(e, exc_info=True)
                return TaskConfigReadResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/task-configs",
            response_model=ListTaskConfigsResponse,
            tags=[ApiTags.TASKS],
            operation_id="list_task_configs",
        )
        async def list_task_configs(
            uuids: List[str] = Query(default=[]),
        ) -> ListTaskConfigsResponse:
            try:
                configs = await self._manager.list_task_configs(config_uuids=uuids or None)
                return ListTaskConfigsResponse(
                    status=JsonValues.SUCCESS,
                    configs={uuid: TaskConfigRead.model_validate(c) for uuid, c in configs.items()},
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListTaskConfigsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/task-configs/{config_uuid}",
            response_model=StatusResponse,
            tags=[ApiTags.TASKS],
            operation_id="update_task_config",
        )
        async def update_task_config(config_uuid: str, payload: UpdateTaskConfigPayload) -> StatusResponse:
            try:
                await self._manager.update_task_config(
                    config_uuid=config_uuid,
                    name=payload.name,
                    typename=payload.typename,
                    params=payload.params,
                    description=payload.description,
                    marked_for_delete=payload.marked_for_delete,
                )
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/task-configs/delete",
            response_model=StatusResponse,
            tags=[ApiTags.TASKS],
            operation_id="delete_task_configs",
        )
        async def delete_task_configs(payload: DeleteTaskConfigsPayload) -> StatusResponse:
            try:
                await self._manager.delete_task_configs(payload.config_uuids)
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        ################################################################################
        # Task Instances API
        ################################################################################

        @self._app.post(
            "/api/task-instances",
            response_model=TaskInstanceReadResponse,
            tags=[ApiTags.TASKS],
            operation_id="create_task_instance",
        )
        async def create_task_instance(payload: CreateTaskInstancePayload) -> TaskInstanceReadResponse:
            try:
                instance = await self._manager.create_task_instance(payload.config_uuid)
                return TaskInstanceReadResponse(
                    status=JsonValues.SUCCESS,
                    instance=TaskInstanceRead.model_validate(instance),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return TaskInstanceReadResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/task-instances",
            response_model=ListTaskInstancesResponse,
            tags=[ApiTags.TASKS],
            operation_id="list_task_instances",
        )
        async def list_task_instances(
            uuids: List[str] = Query(default=[]),
            resumable: Optional[bool] = None,
        ) -> ListTaskInstancesResponse:
            try:
                instances = await self._manager.list_task_instances(
                    task_uuids=uuids or None,
                    resumable=resumable,
                )
                return ListTaskInstancesResponse(
                    status=JsonValues.SUCCESS,
                    instances=[TaskInstanceRead.model_validate(i) for i in instances],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListTaskInstancesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/task-instances/delete",
            response_model=StatusResponse,
            tags=[ApiTags.TASKS],
            operation_id="delete_task_instances",
        )
        async def delete_task_instances(payload: DeleteTaskInstancesPayload) -> StatusResponse:
            try:
                await self._manager.delete_task_instances(payload.task_uuids)
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/task-instances/{instance_uuid}/status",
            response_model=StatusResponse,
            tags=[ApiTags.TASKS],
            operation_id="set_task_instance_status",
        )
        async def set_task_instance_status(instance_uuid: str, payload: SetTaskStatusPayload) -> StatusResponse:
            try:
                await self._manager.set_task_status(instance_uuid, payload.status)
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/task-instances/{instance_uuid}/result",
            response_model=StatusResponse,
            tags=[ApiTags.TASKS],
            operation_id="set_task_instance_result",
        )
        async def set_task_instance_result(instance_uuid: str, payload: SetTaskResultPayload) -> StatusResponse:
            try:
                await self._manager.set_task_result(instance_uuid, payload.result)
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/task-instances/{instance_uuid}/resume-data",
            response_model=StatusResponse,
            tags=[ApiTags.TASKS],
            operation_id="set_task_instance_resume_data",
        )
        async def set_task_instance_resume_data(
            instance_uuid: str, payload: SetTaskResumeDataPayload
        ) -> StatusResponse:
            try:
                await self._manager.set_task_resume_data(instance_uuid, payload.resume_data)
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/task-instances/{instance_uuid}/timestamps",
            response_model=StatusResponse,
            tags=[ApiTags.TASKS],
            operation_id="set_task_instance_timestamps",
        )
        async def set_task_instance_timestamps(instance_uuid: str, payload: SetTaskTimestampsPayload) -> StatusResponse:
            try:
                await self._manager.set_task_timestamps_and_logs(
                    instance_uuid,
                    started_at=payload.started_at,
                    ended_at=payload.ended_at,
                    log_json=payload.log_json,
                )
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/task-instances/{instance_uuid}/heartbeat",
            response_model=StatusResponse,
            tags=[ApiTags.TASKS],
            operation_id="set_task_instance_heartbeat",
        )
        async def set_task_instance_heartbeat(instance_uuid: str) -> StatusResponse:
            try:
                await self._manager.update_task_heartbeat(instance_uuid)
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/task-instances/results",
            response_model=GetTaskResultsResponse,
            tags=[ApiTags.TASKS],
            operation_id="get_task_results",
        )
        async def get_task_results(payload: GetTaskResultsPayload) -> GetTaskResultsResponse:
            try:
                results = await self._manager.get_task_results(payload.task_uuids)
                return GetTaskResultsResponse(status=JsonValues.SUCCESS, results=results)
            except Exception as e:
                logger.error(e, exc_info=True)
                return GetTaskResultsResponse(status=JsonValues.FAILURE, message=str(e))

        # ─────────────────────────────────────────────────────────────────────
        # Dataset endpoints
        # ─────────────────────────────────────────────────────────────────────

        @self._app.get(
            "/api/datasets",
            response_model=ListDatasetsResponse,
            tags=[ApiTags.DATASETS],
            operation_id="list_datasets",
        )
        async def list_datasets() -> ListDatasetsResponse:
            try:
                datasets = await self._manager.list_datasets()
                return ListDatasetsResponse(status=JsonValues.SUCCESS, datasets=datasets)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListDatasetsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/datasets",
            response_model=DatasetResponse,
            tags=[ApiTags.DATASETS],
            operation_id="create_dataset",
        )
        async def create_dataset(payload: DatasetCreate) -> DatasetResponse:
            try:
                dataset = await self._manager.create_dataset(payload)
                return DatasetResponse(status=JsonValues.SUCCESS, dataset=dataset)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return DatasetResponse(status=JsonValues.FAILURE, message=self._unique_constraint_msg(e, 'dataset'))
            except Exception as e:
                logger.error(e, exc_info=True)
                return DatasetResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/datasets/{uuid}",
            response_model=DatasetResponse,
            tags=[ApiTags.DATASETS],
            operation_id="update_dataset",
        )
        async def update_dataset(uuid: str, payload: DatasetUpdate) -> DatasetResponse:
            try:
                dataset = await self._manager.update_dataset(uuid, payload)
                if dataset is None:
                    return DatasetResponse(status=JsonValues.FAILURE, message="Dataset not found")
                return DatasetResponse(status=JsonValues.SUCCESS, dataset=dataset)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return DatasetResponse(status=JsonValues.FAILURE, message=self._unique_constraint_msg(e, 'dataset'))
            except Exception as e:
                logger.error(e, exc_info=True)
                return DatasetResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/datasets/{uuid}",
            response_model=StatusResponse,
            tags=[ApiTags.DATASETS],
            operation_id="delete_dataset",
        )
        async def delete_dataset(uuid: str) -> StatusResponse:
            try:
                ok = await self._manager.delete_dataset(uuid)
                if not ok:
                    return StatusResponse(status=JsonValues.FAILURE, message="Dataset not found")
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/datasets/preview",
            response_model=DatasetImagesResponse,
            tags=[ApiTags.DATASETS],
            operation_id="preview_dataset_images",
        )
        async def preview_dataset_images(
            payload: DatasetPreviewPayload,
            limit: int = Query(default=500, ge=1, le=500),
        ) -> DatasetImagesResponse:
            try:
                images, has_more = await self._manager.preview_dataset_images(
                    payload.filter_json.model_dump(mode="json"),
                    payload.label_uuids,
                    limit=limit,
                )
                return DatasetImagesResponse(
                    status=JsonValues.SUCCESS,
                    images=images,
                    has_more=has_more,
                )
            except Exception as e:
                logger.exception("Failed to preview dataset images")
                return DatasetImagesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/datasets/{uuid}/images",
            response_model=DatasetImagesResponse,
            tags=[ApiTags.DATASETS],
            operation_id="get_dataset_images",
        )
        async def get_dataset_images(
            uuid: str,
            limit: int = Query(default=50, ge=1, le=500),
            sort_by: str = Query(default="updated_at"),
            sort_order: str = Query(default="desc"),
            after_uuid: Optional[str] = Query(default=None),
            after_sort_value: Optional[datetime] = Query(default=None),
        ) -> DatasetImagesResponse:
            try:
                result = await self._manager.get_dataset_images(
                    uuid,
                    limit=limit,
                    sort_by=sort_by,
                    sort_order=sort_order,
                    after_uuid=after_uuid,
                    after_sort_value=after_sort_value,
                )
                if result is None:
                    return DatasetImagesResponse(status=JsonValues.FAILURE, message="Dataset not found")
                images, has_more = result
                return DatasetImagesResponse(
                    status=JsonValues.SUCCESS,
                    images=images,
                    has_more=has_more,
                )
            except Exception as e:
                logger.exception("Failed to fetch dataset images")
                return DatasetImagesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/datasets/{uuid}/labels",
            response_model=DatasetLabelsResponse,
            tags=[ApiTags.DATASETS],
            operation_id="get_dataset_labels",
        )
        async def get_dataset_labels(uuid: str) -> DatasetLabelsResponse:
            try:
                labels = await self._manager.get_dataset_labels(uuid)
                if labels is None:
                    return DatasetLabelsResponse(status=JsonValues.FAILURE, message="Dataset not found")
                return DatasetLabelsResponse(status=JsonValues.SUCCESS, labels=labels)
            except Exception as e:
                logger.error(e, exc_info=True)
                return DatasetLabelsResponse(status=JsonValues.FAILURE, message=str(e))

    # ─── Model Configurations ─────────────────────────────────────────────────
    # NOTE: /typenames must be registered before /{uuid} to prevent FastAPI
    # routing the literal string "typenames" as a UUID path parameter.

    def _register_model_config_routes(self) -> None:
        @self._app.get(
            "/api/model-configs/typenames",
            response_model=ListModelTypenamesResponse,
            tags=[ApiTags.MODEL_CONFIGS],
            operation_id="list_model_typenames",
        )
        async def list_model_typenames() -> ListModelTypenamesResponse:
            try:
                typenames = self._manager.get_available_model_typenames()
                return ListModelTypenamesResponse(status=JsonValues.SUCCESS, typenames=typenames)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListModelTypenamesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/model-configs",
            response_model=ListModelConfigsResponse,
            tags=[ApiTags.MODEL_CONFIGS],
            operation_id="list_model_configs",
        )
        async def list_model_configs() -> ListModelConfigsResponse:
            try:
                configs = await self._manager.list_model_configs()
                return ListModelConfigsResponse(status=JsonValues.SUCCESS, configs=configs)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListModelConfigsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/model-configs",
            response_model=ModelConfigResponse,
            tags=[ApiTags.MODEL_CONFIGS],
            operation_id="create_model_config",
        )
        async def create_model_config(payload: ModelConfigCreate) -> ModelConfigResponse:
            try:
                config = await self._manager.create_model_config(payload)
                return ModelConfigResponse(status=JsonValues.SUCCESS, config=config)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return ModelConfigResponse(status=JsonValues.FAILURE, message=self._unique_constraint_msg(e, 'model configuration'))
            except Exception as e:
                logger.error(e, exc_info=True)
                return ModelConfigResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/model-configs/{uuid}",
            response_model=ModelConfigResponse,
            tags=[ApiTags.MODEL_CONFIGS],
            operation_id="get_model_config",
        )
        async def get_model_config(uuid: str) -> ModelConfigResponse:
            try:
                config = await self._manager.get_model_config(uuid)
                if config is None:
                    return ModelConfigResponse(status=JsonValues.FAILURE, message="Model config not found")
                return ModelConfigResponse(status=JsonValues.SUCCESS, config=config)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ModelConfigResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/model-configs/{uuid}",
            response_model=ModelConfigResponse,
            tags=[ApiTags.MODEL_CONFIGS],
            operation_id="update_model_config",
        )
        async def update_model_config(uuid: str, payload: ModelConfigUpdate) -> ModelConfigResponse:
            try:
                config = await self._manager.update_model_config(uuid, payload)
                if config is None:
                    return ModelConfigResponse(status=JsonValues.FAILURE, message="Model config not found")
                return ModelConfigResponse(status=JsonValues.SUCCESS, config=config)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return ModelConfigResponse(status=JsonValues.FAILURE, message=self._unique_constraint_msg(e, 'model configuration'))
            except Exception as e:
                logger.error(e, exc_info=True)
                return ModelConfigResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/model-configs/{uuid}",
            response_model=StatusResponse,
            tags=[ApiTags.MODEL_CONFIGS],
            operation_id="delete_model_config",
        )
        async def delete_model_config(uuid: str) -> StatusResponse:
            try:
                ok = await self._manager.delete_model_config(uuid)
                if not ok:
                    return StatusResponse(status=JsonValues.FAILURE, message="Model config not found")
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))
