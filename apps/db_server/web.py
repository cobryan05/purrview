"""Web API for DB Server"""

import logging
import mimetypes
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel
from sqlalchemy.exc import IntegrityError

from apps.db_server.manager import Manager
from apps.helpers.consts import ApiTags, JsonKeys, JsonValues, TaskStatus
from apps.helpers.db.types import (
    DEFAULT_SESSION_GAP_S,
    ApiModel,
    BBoxConfidenceRange,
    BBoxUpdate,
    BlipState,
    DatasetCreate,
    DatasetFilterCriteria,
    DatasetRead,
    DatasetUpdate,
    EventBlipRead,
    FilterChainCreate,
    FilterChainRead,
    FilterChainUpdate,
    ImageItemStatus,
    ImageLabelRead,
    ImageLabelWithFilename,
    ImageRecordRead,
    ImageRecordUpdate,
    ImageSearchParams,
    ImageTagStatusRead,
    Label,
    LabelRead,
    LabelScope,
    LabelType,
    LabelUpdate,
    ModelConfigCreate,
    ModelConfigNodeRead,
    ModelConfigRead,
    ModelConfigUpdate,
    ModelGroupCreate,
    ModelGroupRead,
    ModelGroupUpdate,
    PreviewApplyResult,
    PreviewCandidateBulkUpdate,
    PreviewCandidateCreate,
    PreviewCandidateQuery,
    PreviewCandidateRead,
    PreviewConflictPolicy,
    PreviewRunCreate,
    PreviewRunRead,
    PreviewRunUpdate,
    Recording,
    RecordingKind,
    RecordingRead,
    RecordingStatus,
    Source,
    SourceRead,
    SourceRegionRead,
    SourceRegionWrite,
    Tag,
    TagBase,
    TagRead,
    TagUpdate,
    TaskConfigRead,
    TaskInstanceRead,
    TaskParams,
    TaskResultData,
    TaskResumeData,
    TrainingCollectorCreate,
    TrainingCollectorRead,
    TrainingCollectorUpdate,
    TrainingConfigCreate,
    TrainingConfigRead,
    TrainingConfigUpdate,
    WatchEvent,
    WatchEventRead,
    WatchEventReviewState,
    WatchProfileCreate,
    WatchProfileRead,
    WatchProfileUpdate,
)
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.training.collectorSpec import collector_params_schema
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
    # Which source produced this image. Only meaningful on the save that first
    # registers the image (a collecting task); the canvas omits it, and an
    # omission never clears a source already recorded — see DbClient.add_image.
    source_uuid: Optional[str] = None


class BboxPatchPayload(ApiPayload):
    label_uuid: Optional[str] = None
    tag_uuids: List[str] = Field(default_factory=list)
    metadata_json: Optional[Dict[str, Any]] = None


class AddBboxPayload(ApiPayload):
    image_path: str
    label_uuid: Optional[str] = None
    x: float
    y: float
    width: float
    height: float
    tag_uuids: List[str] = Field(default_factory=list)
    metadata_json: Optional[Dict[str, Any]] = None


class AddBboxResponse(StatusResponse):
    bbox_uuid: Optional[str] = None


class BulkApplyAllPayload(ApiPayload):
    filter: ImageSearchParams
    action: str  # relabel | delete | addTags | removeTags | revertRelabel
    label_uuid: Optional[str] = None
    tag_uuids: List[str] = Field(default_factory=list)
    # Bbox-level subset filters (mirror the client-side render logic).
    bbox_tag_filter: List[str] = Field(default_factory=list)
    bbox_label_filter: List[str] = Field(default_factory=list)
    show_all_bboxes: bool = False
    # The review grid's bbox-verification tri-state, as a bbox-level predicate.
    # `filter` may also carry a BBoxVerificationTerm, but that one only selects
    # *images* having such a bbox — it cannot stop the action from also hitting
    # the other-state bboxes on those images, which is what the grid hides.
    # None means "no narrowing"; 'any' is the UI's no-op value and is accepted
    # as the same thing.
    bbox_verification: Optional[str] = None  # verified | unverified
    # Same image-vs-bbox split as bbox_verification above, for the confidence
    # band: the filter's BBoxConfidenceTerm picks images that HAVE an in-band
    # bbox; this narrows the action to the in-band bboxes themselves.
    bbox_confidence: Optional[BBoxConfidenceRange] = None


class CountBboxesPayload(ApiPayload):
    """Same filter surface as BulkApplyAllPayload, minus the action fields."""

    filter: ImageSearchParams
    bbox_tag_filter: List[str] = Field(default_factory=list)
    bbox_label_filter: List[str] = Field(default_factory=list)
    show_all_bboxes: bool = False
    bbox_verification: Optional[str] = None  # verified | unverified
    # Same image-vs-bbox split as bbox_verification above, for the confidence
    # band: the filter's BBoxConfidenceTerm picks images that HAVE an in-band
    # bbox; this narrows the action to the in-band bboxes themselves.
    bbox_confidence: Optional[BBoxConfidenceRange] = None


class CountBboxesResponse(StatusResponse):
    count: int = 0


class BulkApplyAllResponse(StatusResponse):
    modified_count: int = 0
    fail_count: int = 0
    skipped_count: int = 0


class BulkApplyBboxesByUuidsPayload(ApiPayload):
    bbox_uuids: List[str]
    action: str  # relabel | delete | addTags | removeTags | revertRelabel
    label_uuid: Optional[str] = None
    tag_uuids: List[str] = Field(default_factory=list)
    # Optional bbox_uuid -> {"x","y","width","height"} overrides, applied on relabel
    # (e.g. "Run Model on Results" replacing a bbox with a refinement model's box).
    bbox_geometry: Optional[Dict[str, Dict[str, float]]] = None
    # Whether a relabel promotes the box to VERIFIED. Defaults True (a relabel is
    # normally a human decision); "Run Model on Results" sends False so accepting
    # a model's suggestion leaves the box reviewable.
    mark_verified: bool = True


class BulkApplyImagesByUuidsPayload(ApiPayload):
    image_uuids: List[str]
    action: (
        # setPresent | setAbsent | setUncertain | delete | removeFromDb
        # | addTags | removeTags | setTagAbsent | setTagUncertain
        str
    )
    # A label action may target several labels in one pass.
    label_uuids: List[str] = Field(default_factory=list)
    tag_uuids: List[str] = Field(default_factory=list)


class BulkApplyImagesPayload(ApiPayload):
    filter: ImageSearchParams
    action: (
        # setPresent | setAbsent | setUncertain | delete | removeFromDb
        # | addTags | removeTags | setTagAbsent | setTagUncertain
        str
    )
    label_uuids: List[str] = Field(default_factory=list)
    tag_uuids: List[str] = Field(default_factory=list)


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


########
# Labels
########
class AddLabelPayload(ApiPayload):
    name: str
    color: str
    parent_uuid: Optional[str] = None
    label_type: LabelType = LabelType.STANDARD
    scope: LabelScope = LabelScope.OBJECT
    #: Both default off, matching `LabelBase`. Accepted at creation because the client
    #: has always sent them (`shared/api/labels.js`) — the payload simply had no field to
    #: receive them, so they were dropped. Harmless while the default was on; now it is
    #: the difference between creating an opted-in label and having to patch it after.
    publish_to_ha: bool = False
    publish_per_source: bool = False


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


class ListImagesByDirectoryPayload(ApiPayload):
    directories: List[str] = Field(default_factory=list)


class ImageLocationRead(ApiPayload):
    uuid: str
    path: str


class ListImagesByDirectoryResponse(StatusResponse):
    images: List[ImageLocationRead] = Field(default_factory=list)


class ImageRelocation(ApiPayload):
    uuid: str
    new_path: str


class RelocateImagesPayload(ApiPayload):
    moves: List[ImageRelocation] = Field(default_factory=list)


class RelocateImageError(ApiPayload):
    uuid: str
    reason: str


class RelocateImagesResponse(StatusResponse):
    updated: int = 0
    errors: List[RelocateImageError] = Field(default_factory=list)


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


class SourceIngestTiersResponse(StatusResponse):
    #: source uuid → IngestPlan.to_dict() (tier, reason, buffered, transcoding).
    tiers: Dict[str, dict] = Field(default_factory=dict)


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


class GetSourceRegionsPayload(ApiPayload):
    #: None lists every source's regions in one call, which is what the rule editor needs
    #: — it offers zones across all of a watcher's sources and would otherwise fan out
    #: one request per source.
    source_uuid: Optional[str] = None
    role: Optional[str] = None


class GetSourceRegionsResponse(StatusResponse):
    regions: List[SourceRegionRead] = Field(default_factory=list)


class SetSourceRegionsPayload(ApiPayload):
    source_uuid: str
    regions: List[SourceRegionWrite] = Field(default_factory=list)


class SetSourceRegionsResponse(StatusResponse):
    regions: List[SourceRegionRead] = Field(default_factory=list)
    #: Rules that referenced a region this write removed. Not an error — the user may
    #: well mean it — but silently breaking a rule is the failure mode zones-by-uuid
    #: invites, so the caller gets told and can warn.
    orphaned_rules: List[str] = Field(default_factory=list)


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


class UploadedImageRead(ApiModel):
    """One file's outcome. Reported per-file rather than as a single overall
    status, because a mixed result is the normal case for a multi-file drop and
    the user needs to know *which* ones landed."""

    filename: str
    ok: bool
    path: Optional[str] = None
    uuid: Optional[str] = None
    message: Optional[str] = None
    # Set when the file resembles an image already in the DB. Advisory only —
    # the upload still happened. See Manager.save_uploaded_image.
    duplicate_of: Optional[str] = None
    duplicate_distance: Optional[int] = None
    # Set when a name collision forced a new name, so the UI can say so rather
    # than leaving the user to wonder where "snapshot.jpg" went.
    renamed_from: Optional[str] = None


class UploadImagesResponse(StatusResponse):
    results: List[UploadedImageRead] = []


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
    status: ImageItemStatus


class SetImageLabelResponse(StatusResponse):
    # True when the write was refused as a legitimate business-rule conflict
    # (ABSENT blocked by an existing verified bbox for the label) rather than a
    # real error. status stays "success" — the stored state is exactly what it
    # should be — so callers can tell "nothing to worry about" apart from an
    # actual failure (e.g. image not found).
    skipped: bool = False


class GetImageLabelsResponse(StatusResponse):
    labels: List[ImageLabelRead] = Field(default_factory=list)


class BatchImageLabelsPayload(Payload):
    image_uuids: List[str] = Field(default_factory=list)


class BatchImageLabelsResponse(StatusResponse):
    # image_uuid → label rows; images with no rows are omitted.
    image_labels: Dict[str, List[ImageLabelRead]] = Field(default_factory=dict)


class ListImagesForLabelResponse(StatusResponse):
    images: List[ImageLabelWithFilename] = Field(default_factory=list)


class SetImageTagPayload(ApiPayload):
    image_path: str
    tag_uuid: str
    action: str  # add | remove


class SetImageTagResponse(StatusResponse):
    pass


class GetImageTagsResponse(StatusResponse):
    tags: List[TagRead] = Field(default_factory=list)


class SetImageTagStatusPayload(ApiPayload):
    image_path: str
    tag_uuid: str
    status: ImageItemStatus  # present | absent | uncertain


class SetImageTagStatusResponse(StatusResponse):
    pass


class GetImageTagStatusesResponse(StatusResponse):
    tag_statuses: List[ImageTagStatusRead] = Field(default_factory=list)


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
    is_temporary: Optional[bool] = None


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
    # Only meaningful with status=ERROR. Any other status clears whatever message
    # was stored, so a resumed/rerun instance doesn't keep displaying a stale one.
    error_message: Optional[str] = None


class SetTaskResultPayload(ApiPayload):
    result: TaskResultData = Field(default_factory=dict)


class SetTaskResumeDataPayload(ApiPayload):
    resume_data: TaskResumeData = Field(default_factory=dict)


class SetTaskTimestampsPayload(ApiPayload):
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    log_json: Optional[List[str]] = None
    # Snapshot of the task's status widgets, persisted next to the logs for the same
    # reason and on the same best-effort footing (see TaskInstanceBase.status_data_json).
    status_data_json: Optional[Dict[str, Any]] = None


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


class DatasetImageUuidsResponse(StatusResponse):
    image_uuids: List[str] = Field(default_factory=list)


class DatasetLabelsResponse(StatusResponse):
    labels: List[LabelRead] = Field(default_factory=list)


class DatasetPreviewPayload(ApiModel):
    """Request body for the unsaved-preview endpoint."""

    filter_json: DatasetFilterCriteria = Field(default_factory=DatasetFilterCriteria)
    label_uuids: List[str] = Field(default_factory=list)


class ListTrainingConfigsResponse(StatusResponse):
    configs: List[TrainingConfigRead] = Field(default_factory=list)


class TrainingConfigResponse(StatusResponse):
    config: Optional[TrainingConfigRead] = None


class TrainingConfigReadinessResponse(StatusResponse):
    ready_count: int = 0
    # Split of ready_count by whether the image actually contains an in-scope label:
    # positive = at least one label verified PRESENT, negative = fully verified but all
    # ABSENT (background). The two always sum to ready_count.
    ready_positive_count: int = 0
    ready_negative_count: int = 0
    needs_review_count: int = 0
    not_started_count: int = 0
    dataset_uuids: List[str] = Field(default_factory=list)
    label_uuids: List[str] = Field(default_factory=list)
    roll_up_active: bool = False


class DetectionImageEntry(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    image_uuid: str
    filename: str
    annotations: List[List[float]] = Field(default_factory=list)


class DetectionDatasetResponse(StatusResponse):
    class_names: List[str] = Field(default_factory=list)
    images: List[DetectionImageEntry] = Field(default_factory=list)


class CropEntry(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    bbox_uuid: str
    class_idx: int
    x: float
    y: float
    width: float
    height: float


class CropImageEntry(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    image_uuid: str
    filename: str
    crops: List[CropEntry] = Field(default_factory=list)


class CropDatasetResponse(StatusResponse):
    class_names: List[str] = Field(default_factory=list)
    class_label_uuids: List[str] = Field(default_factory=list)
    images: List[CropImageEntry] = Field(default_factory=list)


class CoLabelGapBreakdownEntry(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    label_uuid: str
    label_name: str
    count: int = 0


class LabelStatEntry(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    label_uuid: str
    label_scope: str  # "image" or "object"
    verified_image_count: int = 0
    unverified_image_count: int = 0
    present_image_count: int = 0
    absent_image_count: int = 0
    verified_bbox_count: int = 0
    unverified_bbox_count: int = 0
    co_labels_covered_count: int = 0
    co_label_gaps_count: int = 0
    co_label_gap_breakdown: List[CoLabelGapBreakdownEntry] = Field(default_factory=list)
    has_rollup: bool = False
    roll_up_parent_uuid: Optional[str] = None


class TrainingConfigLabelStatsResponse(StatusResponse):
    label_stats: List[LabelStatEntry] = Field(default_factory=list)
    total_images: int = 0


class TrainingConfigLabelStatsForImagesPayload(Payload):
    # A previously-trained model's recorded image manifest. POST rather than a GET
    # query param purely because this list runs to thousands of UUIDs.
    image_uuids: List[str] = Field(default_factory=list)


class TrainingConfigImageSubsetPayload(Payload):
    # Same shape as the payload above, kept as its own model rather than shared:
    # the generated API clients name their request models after the endpoint's
    # payload class, so collapsing the two would rename an existing client model.
    image_uuids: List[str] = Field(default_factory=list)


class NeedsReviewImagesPayload(Payload):
    label_uuids: List[str] = Field(default_factory=list)
    after_uuid: Optional[str] = None
    after_sort_value: Optional[datetime] = None
    sort_by: Literal["uuid", "created_at", "updated_at"] = "created_at"
    sort_order: Literal["asc", "desc"] = "desc"
    limit: int = 50


class ReadyImagesPayload(Payload):
    after_uuid: Optional[str] = None
    after_sort_value: Optional[datetime] = None
    sort_by: Literal["uuid", "created_at", "updated_at"] = "created_at"
    sort_order: Literal["asc", "desc"] = "desc"
    limit: int = 50


class TrainingTargetDetectionDatasetPayload(Payload):
    dataset_uuids: List[str] = Field(default_factory=list)
    # Optional pre-resolved target scope — a training run snapshots its targets from
    # /training-targets at _init, so passing that snapshot here keeps every target
    # of the run consistent with it (a config edit mid-run can't shift the class
    # set) and skips re-deriving all targets on each per-target request.
    label_uuids: Optional[List[str]] = None
    class_remap: Dict[str, str] = Field(default_factory=dict)
    # None = Manager's defaults (see Manager.NEGATIVE_RATIO_DEFAULT / _FLOOR_DEFAULT).
    negative_ratio: Optional[float] = None
    negative_ratio_floor: Optional[int] = None


class TrainingTargetCropDatasetPayload(Payload):
    dataset_uuids: List[str] = Field(default_factory=list)
    # Same pre-resolved-scope contract as TrainingTargetDetectionDatasetPayload.
    label_uuids: Optional[List[str]] = None
    class_remap: Dict[str, str] = Field(default_factory=dict)


class ImageGapEntry(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    label_uuid: str
    label_name: str
    label_scope: str  # "image" or "object"


class ImageGapsPayload(Payload):
    image_uuids: List[str] = Field(default_factory=list)


class TrainingConfigImageGapsResponse(StatusResponse):
    # Keyed by image uuid; images with no gaps are omitted rather than mapped to [].
    gaps: Dict[str, List[ImageGapEntry]] = Field(default_factory=dict)


class ImagesByLabelStatusPayload(Payload):
    label_uuid: str
    rollup_label_uuids: Optional[List[str]] = None
    status: Optional[
        Literal["present", "absent", "uncertain", "all", "has_verified_bbox", "co_labels_covered", "co_label_gaps"]
    ] = None
    # Only used when status == "co_label_gaps": narrows to images where this specific other
    # top-level label is the one unresolved, matching one row of the co-label gap breakdown.
    missing_label_uuid: Optional[str] = None
    after_uuid: Optional[str] = None
    after_sort_value: Optional[datetime] = None
    sort_by: Literal["uuid", "created_at", "updated_at"] = "created_at"
    sort_order: Literal["asc", "desc"] = "desc"
    limit: int = 50


class ListModelConfigsResponse(StatusResponse):
    configs: List[ModelConfigRead] = Field(default_factory=list)


class ModelConfigResponse(StatusResponse):
    config: Optional[ModelConfigRead] = None


class ListFilterChainsResponse(StatusResponse):
    chains: List[FilterChainRead] = Field(default_factory=list)


class FilterChainResponse(StatusResponse):
    chain: Optional[FilterChainRead] = None


class ListTrainingCollectorsResponse(StatusResponse):
    collectors: List[TrainingCollectorRead] = Field(default_factory=list)


class TrainingCollectorResponse(StatusResponse):
    collector: Optional[TrainingCollectorRead] = None


class TrainingCollectorSchemaResponse(StatusResponse):
    #: `collector_params_schema()` verbatim — field shapes vary per knob, so this
    #: stays an untyped dict rather than a model that would have to restate them.
    params_schema: Dict[str, Any] = Field(default_factory=dict)


class AddPreviewCandidatesPayload(ApiPayload):
    candidates: List[PreviewCandidateCreate] = Field(default_factory=list)
    # Which model wins when two of them propose on the same bbox. `last` is
    # what the browser's merge loop always did, silently; it stays the default.
    conflict_policy: PreviewConflictPolicy = PreviewConflictPolicy.LAST


class ApplyPreviewRunPayload(ApiPayload):
    # Off by default: a model's suggestion is not a human verification.
    mark_verified: bool = False


class ListPreviewRunsResponse(StatusResponse):
    runs: List[PreviewRunRead] = Field(default_factory=list)


class PreviewRunResponse(StatusResponse):
    run: Optional[PreviewRunRead] = None


class ListPreviewCandidatesResponse(StatusResponse):
    candidates: List[PreviewCandidateRead] = Field(default_factory=list)
    # The count *after* filtering, not the run's total — the triage panel's
    # pagination and its "N of M" both read off this.
    total_count: int = 0


class PreviewCandidatesWriteResponse(StatusResponse):
    updated: int = 0


class PreviewApplyResponse(StatusResponse):
    result: PreviewApplyResult = Field(default_factory=PreviewApplyResult)


class TrackerInfo(Payload):
    typename: str
    display_name: str
    #: `Tracker.params_schema()` verbatim — field shapes vary per tracker, so this stays
    #: an open dict and is rendered by the same machinery that renders provider schemas.
    params_schema: Dict[str, Any] = Field(default_factory=dict)


class ListTrackersResponse(StatusResponse):
    trackers: List[TrackerInfo] = Field(default_factory=list)


class ActionInfo(Payload):
    #: The registry key, which is what a `watch_actions` row stores in `kind`.
    kind: str
    display_name: str
    #: `Action.params_schema()` verbatim, for the same reason `TrackerInfo` keeps it
    #: open: the field shapes are the action's to declare, not this layer's to model.
    params_schema: Dict[str, Any] = Field(default_factory=dict)


class ListActionsResponse(StatusResponse):
    actions: List[ActionInfo] = Field(default_factory=list)


class ListWatchProfilesResponse(StatusResponse):
    profiles: List[WatchProfileRead] = Field(default_factory=list)


class WatchProfileResponse(StatusResponse):
    profile: Optional[WatchProfileRead] = None


class WatchProfileValidationResponse(StatusResponse):
    valid: bool = False
    # Errors mean "this watcher cannot work"; warnings mean "this will run but probably
    # not do what you meant". Both matter because every one of them fails silently.
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ListWatchEventsResponse(StatusResponse):
    events: List[WatchEventRead] = Field(default_factory=list)
    #: Matches before paging, so the page can say "N events" rather than "N so far".
    total: int = 0


class WatchEventResponse(StatusResponse):
    event: Optional[WatchEventRead] = None


class EventBlipPayload(Payload):
    """One incident, as the analysis task reports it.

    Mirrors `apps.helpers.recordings.blips.Blip.to_dict()`. `harvestable` is sent by the
    analyser rather than recomputed here on purpose: the rule that decides it lives with
    the detectors that set `fault`, and re-deriving it at the API boundary would put the
    same rule in two places to drift apart.
    """

    kind: str
    track_id: int = 0
    start_s: float = 0.0
    end_s: float = 0.0
    severity: float = 0.0
    fault: str = ""
    harvestable: bool = False
    detail: Dict[str, Any] = Field(default_factory=dict)


class ReplaceEventBlipsRequest(Payload):
    """A whole analysis result for one event, replacing any previous one.

    Whole-list replace rather than append, because analysis is deterministic over a
    sidecar that does not change — re-running it must converge rather than accumulate.
    """

    event_uuid: str
    recording_uuid: Optional[str] = None
    blips: List[EventBlipPayload] = Field(default_factory=list)


class SetEventBlipStatesRequest(Payload):
    uuids: List[str] = Field(default_factory=list)
    state: str = ""


class ListEventBlipsResponse(StatusResponse):
    blips: List[EventBlipRead] = Field(default_factory=list)
    #: Matches before paging, so the panel can say "N blips" rather than "N so far".
    total: int = 0


class EventBlipsResponse(StatusResponse):
    blips: List[EventBlipRead] = Field(default_factory=list)


class UpdatedCountResponse(StatusResponse):
    #: How many rows actually moved. Lower than requested when a uuid was already gone
    #: or already in the target state — a normal outcome, not an error.
    updated: int = 0


class DeleteWatchEventsRequest(Payload):
    uuids: List[str] = Field(default_factory=list)


class DeleteWatchEventsResponse(StatusResponse):
    #: How many of the requested events actually went away. Lower than the request when
    #: a uuid was already gone — which is a normal outcome for a selection deleted from
    #: two tabs, not an error.
    deleted: int = 0


class CloseStrandedEventsRequest(Payload):
    profile_uuid: str


class CloseStrandedEventsResponse(StatusResponse):
    #: How many events were still open. Zero is the normal answer.
    closed: int = 0


class WatchEventFacet(ApiModel):
    ref: str
    name: str


class WatchEventFacetsResponse(StatusResponse):
    """What each filter dropdown can offer — the values actually present in history."""

    rules: List[WatchEventFacet] = Field(default_factory=list)
    subjects: List[WatchEventFacet] = Field(default_factory=list)
    targets: List[WatchEventFacet] = Field(default_factory=list)
    sources: List[WatchEventFacet] = Field(default_factory=list)


class WatchEventCount(ApiModel):
    source_uuid: str
    rule_uuid: str
    #: None for a rule with no subject slot. Kept rather than filtered so rule-level
    #: totals stay complete — the bridge sums these rows to get every coarser tier.
    subject_ref: Optional[str] = None
    count: int
    #: ISO 8601, the most recent event in this group. Seeds the "last seen" sensors,
    #: which would otherwise read `unknown` after a restart until the next event —
    #: for a rule that fires twice a day, most of the day.
    last_seen: Optional[str] = None


class WatchEventCountsResponse(StatusResponse):
    counts: List[WatchEventCount] = Field(default_factory=list)


class WatchEventQuery:
    """The `watch_events` search parameters, shared by every endpoint that asks them.

    Three endpoints — the grid, the grouped view and the timeline — must answer the
    *same* question about the *same* rows and differ only in what they return. Spelled
    out three times, the first filter added to one and not the others would show a group
    count that disagreed with the grid total it drills into, with nothing to make the
    divergence audible. FastAPI flattens a dependency's parameters into the operation's
    query string, so this reads identically from the outside and generates the same
    client.

    The db layer has the matching half of this contract in
    `DbClient._apply_watch_event_filters`; `as_kwargs()` is the join between them.
    """

    def __init__(
        self,
        profile_uuid: Optional[str] = None,
        source_uuid: Optional[str] = None,
        rule_uuid: Optional[str] = None,
        subject_ref: Optional[str] = None,
        target_ref: Optional[str] = None,
        source_uuids: Optional[List[str]] = Query(None),
        rule_uuids: Optional[List[str]] = Query(None),
        subject_refs: Optional[List[str]] = Query(None),
        target_refs: Optional[List[str]] = Query(None),
        exclude_source_uuids: Optional[List[str]] = Query(None),
        exclude_rule_uuids: Optional[List[str]] = Query(None),
        exclude_subject_refs: Optional[List[str]] = Query(None),
        exclude_target_refs: Optional[List[str]] = Query(None),
        start_from: Optional[datetime] = None,
        start_to: Optional[datetime] = None,
        has_recording: Optional[bool] = None,
        recording_kind: Optional[str] = None,
        has_blips: Optional[bool] = None,
        review_state: Optional[str] = None,
    ):
        self.profile_uuid = profile_uuid
        self.source_uuid = source_uuid
        self.rule_uuid = rule_uuid
        self.subject_ref = subject_ref
        self.target_ref = target_ref
        self.source_uuids = source_uuids
        self.rule_uuids = rule_uuids
        self.subject_refs = subject_refs
        self.target_refs = target_refs
        self.exclude_source_uuids = exclude_source_uuids
        self.exclude_rule_uuids = exclude_rule_uuids
        self.exclude_subject_refs = exclude_subject_refs
        self.exclude_target_refs = exclude_target_refs
        self.start_from = start_from
        self.start_to = start_to
        self.has_recording = has_recording
        self.recording_kind = recording_kind
        self.has_blips = has_blips
        self.review_state = review_state

    def as_kwargs(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class WatchEventGroup(ApiModel):
    """One identity — a rule, an actor and a thing acted on — and its whole history."""

    rule_uuid: Optional[str] = None
    rule_name: Optional[str] = None
    subject_ref: Optional[str] = None
    subject_name: Optional[str] = None
    target_ref: Optional[str] = None
    target_name: Optional[str] = None
    #: Grouped at the finest grain always; the UI folds the camera away unless asked to
    #: split by it, summing these rows the way the MQTT bridge sums `/counts`.
    source_uuid: Optional[str] = None
    count: int = 0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    #: Summed over closed events only. Open ones contribute nothing rather than a guess.
    total_duration_s: float = 0.0
    #: The event whose snapshot is the card art — newest *with* a snapshot, so a group
    #: cannot render as an empty tile because its newest event happened to lack one.
    representative_uuid: Optional[str] = None
    with_recording: int = 0
    with_blips: int = 0


class WatchEventGroupsResponse(StatusResponse):
    groups: List[WatchEventGroup] = Field(default_factory=list)
    #: Events behind the groups, so the view switcher can show the same total the grid
    #: does without a second round trip.
    total: int = 0


class WatchEventHistogramBucket(ApiModel):
    #: ISO 8601 without an offset, in **UTC** — see `watch_event_histogram`. The client
    #: composes local days from hour buckets rather than this endpoint taking a timezone.
    bucket_start: str
    rule_uuid: Optional[str] = None
    subject_ref: Optional[str] = None
    target_ref: Optional[str] = None
    source_uuid: Optional[str] = None
    count: int = 0


class SetWatchEventReviewStatesRequest(Payload):
    uuids: List[str] = Field(default_factory=list)
    state: str = ""


class WatchEventSession(ApiModel):
    """One visit — consecutive events of an identity with no long gap between them.

    A single trip to the water bowl re-fires its rule several times as `min_duration_s`
    and `rearm_delay_s` cycle, so the flat grid shows seven tiles for one drink.
    """

    rule_uuid: Optional[str] = None
    rule_name: Optional[str] = None
    subject_ref: Optional[str] = None
    subject_name: Optional[str] = None
    target_ref: Optional[str] = None
    target_name: Optional[str] = None
    source_uuid: Optional[str] = None
    #: The session's ordinal within its identity. Stable for a given gap and filter set,
    #: and only meaningful alongside the identity — not a global id.
    session_ix: int = 0
    count: int = 0
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    representative_uuid: Optional[str] = None


class ListWatchEventSessionsResponse(StatusResponse):
    sessions: List[WatchEventSession] = Field(default_factory=list)
    total: int = 0
    #: Echoed back so the UI can label the grouping it is actually showing rather than
    #: the one it asked for.
    gap_s: float = 0.0


class WatchEventHistogramResponse(StatusResponse):
    #: The resolution actually used — `day` when an `hour` request would have overflowed.
    #: Reported rather than applied silently: a histogram that quietly coarsened itself
    #: would misrepresent when things happened.
    bucket: str = "hour"
    buckets: List[WatchEventHistogramBucket] = Field(default_factory=list)


class WatchEventCreate(Payload):
    profile_uuid: str
    source_uuid: Optional[str] = None
    rule_uuid: Optional[str] = None
    rule_name: Optional[str] = None
    event_label: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    peak_confidence: float = 0.0
    track_ids_json: List[Any] = Field(default_factory=list)
    subject_ref: Optional[str] = None
    subject_name: Optional[str] = None
    target_ref: Optional[str] = None
    target_name: Optional[str] = None
    truncated: bool = False
    close_reason: Optional[str] = None
    snapshot_path: Optional[str] = None
    metadata_json: Dict[str, Any] = Field(default_factory=dict)


class WatchEventUpdate(Payload):
    """Only the fields an event's *closing* edge changes.

    A closed event is history: the rule, source and start moment are not editable, so
    they are absent here rather than optional-and-ignored.
    """

    ended_at: Optional[datetime] = None
    peak_confidence: Optional[float] = None
    truncated: Optional[bool] = None
    close_reason: Optional[str] = None
    review_state: Optional[str] = None
    snapshot_path: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None

    @field_validator("review_state")
    @classmethod
    def _known_review_state(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {s.value for s in WatchEventReviewState}:
            raise ValueError(f"unknown review_state {v!r}")
        return v


class StorageBucketRead(BaseModel):
    """One independently-bounded pile of files on disk.

    Carries the alias generator itself: it is nested inside a StatusResponse but
    does not inherit from one, and without this its multi-word fields would go out
    snake_case while every field around them is camel.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    bytes: int = 0
    files: int = 0
    # 0 means "no size limit", matching how the settings themselves read a 0.
    max_bytes: int = 0
    retention_days: Optional[int] = None
    roots: List[str] = Field(default_factory=list)
    dir: Optional[str] = None


class StorageUsageResponse(StatusResponse):
    recordings: Optional[StorageBucketRead] = None
    timelapse: Optional[StorageBucketRead] = None
    thumbnail_cache: Optional[StorageBucketRead] = None


class StoragePurgeResponse(StatusResponse):
    """What a purge or a sweep actually removed. Counts, not a bare success —
    "0 files" and "4,000 files" are very different answers to the same click."""

    files: int = 0
    bytes: int = 0
    events: int = 0
    recordings: int = 0


class ListRecordingsResponse(StatusResponse):
    recordings: List[RecordingRead] = Field(default_factory=list)


class RecordingResponse(StatusResponse):
    recording: Optional[RecordingRead] = None


class RecordingCreate(Payload):
    """A media file that has already been written to disk.

    For a clip, `started_at` is its *actual* first frame, not the event's start:
    pre-roll is cut retroactively out of the buffer and may be clamped by eviction, so
    the two differ by design. The review page subtracts them to mark where in the clip
    the rule fired, which only works if this really is the clip's own start.

    For a timelapse the same field is the first *sampled* frame, and `duration_s` is the
    length of the finished video rather than of the event it summarises — a two-hour
    event compiles to seconds, and the player's scrubber measures the file.
    """

    source_uuid: Optional[str] = None
    profile_uuid: Optional[str] = None
    event_uuid: Optional[str] = None
    kind: RecordingKind = RecordingKind.CLIP
    file_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    overlay_path: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_s: float = 0.0
    size_bytes: int = 0
    status: str = RecordingStatus.COMPLETE.value
    error: Optional[str] = None
    partial_preroll: bool = False
    metadata_json: Dict[str, Any] = Field(default_factory=dict)


class ListModelGroupsResponse(StatusResponse):
    groups: List[ModelGroupRead] = Field(default_factory=list)


class ModelGroupResponse(StatusResponse):
    group: Optional[ModelGroupRead] = None


class TrainingTarget(ApiModel):
    """One artifact a full training run must produce (see
    DbClient.derive_training_targets for the derivation semantics)."""

    node_uuid: str
    role: Literal["top", "submodel"]
    label_uuid: Optional[str] = None
    model_type: str
    architecture: str
    label_uuids: List[str] = Field(default_factory=list)
    class_names: List[str] = Field(default_factory=list)
    class_remap: Dict[str, str] = Field(default_factory=dict)


class TrainingTargetsResponse(StatusResponse):
    targets: List[TrainingTarget] = Field(default_factory=list)


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
    label_uuids: List[str] = []
    tag_uuids: List[str] = []
    exclude_tag_uuids: List[str] = []
    tag_match_mode: Literal["any", "all"] = "any"


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
    label_uuids: List[str] = []
    tag_uuids: List[str] = []
    exclude_tag_uuids: List[str] = []
    tag_match_mode: Literal["any", "all"] = "any"


class FindSimilarBboxesByCoordsPayload(ApiPayload):
    image_uuid: str
    x: float
    y: float
    w: float
    h: float
    threshold: int = 8
    label_uuids: List[str] = []
    tag_uuids: List[str] = []
    exclude_tag_uuids: List[str] = []
    tag_match_mode: Literal["any", "all"] = "any"


class FindDissimilarImagesByUuidPayload(ApiPayload):
    image_uuid: str
    count: int = 100
    candidate_uuids: Optional[List[str]] = None


class DissimilarImageMatch(ApiModel):
    image_uuid: str
    rank: int


class FindDissimilarImagesResponse(StatusResponse):
    matches: List[DissimilarImageMatch] = Field(default_factory=list)


class FindDissimilarBboxesByUuidPayload(ApiPayload):
    bbox_uuid: str
    count: int = 100
    label_uuids: List[str] = []


class DissimilarBboxMatch(ApiModel):
    bbox_uuid: str
    image_uuid: Optional[str] = None
    label_uuid: str
    rank: int


class FindDissimilarBboxesResponse(StatusResponse):
    matches: List[DissimilarBboxMatch] = Field(default_factory=list)


class FindDissimilarGlobalPayload(ApiPayload):
    count: int = 100
    seed_image_uuid: Optional[str] = None
    # When set, scopes the walk to the caller's filtered image set instead of the
    # default unverified-bbox scope. An empty list means "nothing matched".
    candidate_uuids: Optional[List[str]] = None


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
    # Threshold-independent nearest-neighbour distances, for threshold calibration.
    # None = nothing to compare against (empty index / unlabelled bbox), which is
    # distinct from "far away" and must not be treated as a large distance.
    image_min_distance: Optional[int] = None
    bbox_min_distances: List[Optional[int]] = Field(default_factory=list)


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

        # Block all non-status requests while the DB is still initializing (alembic
        # migration may be in progress).  /status is always allowed so the notices
        # SSE stream can surface the migration phase to the frontend.
        _mgr = manager

        @self._app.middleware("http")
        async def _db_ready_gate(request: Request, call_next):
            if not _mgr._db_ready and request.url.path != "/status":
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Database is initializing, please retry shortly"},
                )
            return await call_next(request)

        # Mount static files
        self._app.mount("/static", StaticFiles(directory="/app/static", check_dir=False), name="static")

        # Register routes
        self._register_routes()
        self._register_model_config_routes()
        self._register_filter_chain_routes()
        self._register_training_collector_routes()
        self._register_preview_run_routes()
        self._register_watch_profile_routes()
        self._register_recording_routes()
        self._register_model_group_routes()
        self._register_exception()

    @staticmethod
    def _unique_constraint_msg(e: IntegrityError, entity: str) -> str:
        """Return a friendly message for unique-constraint violations, raw error otherwise."""
        return f"A {entity} with that name already exists." if "unique" in str(e).lower() else str(e)

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
                response_data = {
                    JsonKeys.STATUS: JsonValues.SUCCESS,
                    JsonKeys.CONFIG: server_config,
                }
            except Exception as e:
                response_data = {
                    JsonKeys.STATUS: JsonValues.FAILURE,
                    JsonKeys.MESSAGE: str(e),
                }
            return JSONResponse(content=response_data)

        @self._app.get("/status", operation_id="db_server_status", response_class=JSONResponse)
        async def db_server_status():
            try:
                status = await self._manager.get_status()
                return JSONResponse(content=status)
            except Exception as e:
                return JSONResponse(
                    content={
                        "service": "db_server",
                        "status": "error",
                        "message": str(e),
                    }
                )

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
                    request.name,
                    request.color,
                    parent_uuid=request.parent_uuid,
                    label_type=request.label_type,
                    scope=request.scope,
                    publish_to_ha=request.publish_to_ha,
                    publish_per_source=request.publish_per_source,
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
            API endpoint to update an existing label's name, color, type, scope,
            and/or parent (re-parent / "move" it in the hierarchy).
            """
            try:
                # parentUuid is only forwarded when the caller explicitly set it —
                # `None` (promote to root) must stay distinguishable from "field
                # absent from the request body" (leave the existing parent alone).
                parent_uuid_kwargs = {}
                if "parent_uuid" in request.model_fields_set:
                    parent_uuid_kwargs["parent_uuid"] = request.parent_uuid

                await self._manager.update_label(
                    label_uuid=label_uuid,
                    name=request.name,
                    color=request.color,
                    label_type=request.label_type,
                    scope=request.scope,
                    publish_to_ha=request.publish_to_ha,
                    publish_per_source=request.publish_per_source,
                    **parent_uuid_kwargs,
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

        @self._app.get(
            "/api/sources/ingest-tiers",
            response_model=SourceIngestTiersResponse,
            tags=[ApiTags.SOURCES],
            operation_id="get_source_ingest_tiers",
        )
        async def get_source_ingest_tiers_api() -> SourceIngestTiersResponse:
            """
            Whether each source can be rewound, and why — the four-state buffer answer.

            Exists so the Watchers editor can warn *at configuration time* that a
            pre-roll being typed into a snapshot camera cannot be honoured. Without it
            the mistake is only discoverable at event time, as a clip that silently
            never arrives.
            """
            try:
                tiers = await self._manager.get_source_ingest_tiers()
                return SourceIngestTiersResponse(status=JsonValues.SUCCESS, tiers=tiers)
            except Exception as e:
                logger.error(e, exc_info=True)
                return SourceIngestTiersResponse(status=JsonValues.FAILURE, message=str(e))

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
        async def update_source_api(
            payload: UpdateSourcePayload,
        ) -> UpdateSourceResponse:
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
                return UpdateSourceResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "source"),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return UpdateSourceResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/sources/create",
            tags=[ApiTags.SOURCES],
            operation_id="create_source",
            response_model=CreateSourceResponse,
        )
        async def create_source_api(
            payload: CreateSourcePayload,
        ) -> CreateSourceResponse:
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
                return CreateSourceResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "source"),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return CreateSourceResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/sources/delete",
            tags=[ApiTags.SOURCES],
            operation_id="delete_sources",
            response_model=DeleteSourceResponse,
        )
        async def delete_sources_api(
            payload: DeleteSourcePayload,
        ) -> DeleteSourceResponse:
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

        @self._app.post(
            "/api/sources/regions/get",
            response_model=GetSourceRegionsResponse,
            tags=[ApiTags.SOURCES],
            operation_id="get_source_regions",
        )
        async def get_source_regions_api(payload: GetSourceRegionsPayload) -> GetSourceRegionsResponse:
            """Zones and masks drawn on a source's frame.

            Regions belong to the source rather than to a watcher's use of it: the
            geometry describes the camera's view, so it is drawn once and every watcher
            on that camera sees it.
            """
            try:
                regions = await self._manager.list_source_regions(payload.source_uuid, payload.role)
                return GetSourceRegionsResponse(status=JsonValues.SUCCESS, regions=regions)
            except Exception as e:
                logger.error(e, exc_info=True)
                return GetSourceRegionsResponse(status=JsonValues.FAILURE, message=str(e), regions=[])

        @self._app.post(
            "/api/sources/regions/set",
            response_model=SetSourceRegionsResponse,
            tags=[ApiTags.SOURCES],
            operation_id="set_source_regions",
        )
        async def set_source_regions_api(payload: SetSourceRegionsPayload) -> SetSourceRegionsResponse:
            """Whole-list replace of one source's regions.

            Uuids supplied by the caller are preserved, because rules point at zones by
            uuid from another table entirely — regenerating them here would break every
            zone rule at once with no error anywhere.
            """
            try:
                regions, orphaned = await self._manager.set_source_regions(payload.source_uuid, payload.regions)
                return SetSourceRegionsResponse(status=JsonValues.SUCCESS, regions=regions, orphaned_rules=orphaned)
            except Exception as e:
                logger.error(e, exc_info=True)
                return SetSourceRegionsResponse(status=JsonValues.FAILURE, message=str(e), regions=[])

        ################################################################################
        # Tags API
        ################################################################################

        @self._app.post(
            "/api/tags/add",
            response_model=AddTagResponse,
            tags=[ApiTags.TAGS],
            operation_id="add_tag",
        )
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
                return AddTagResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "tag"),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return AddTagResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/tags/delete",
            response_model=DeleteTagResponse,
            tags=[ApiTags.TAGS],
            operation_id="delete_tag",
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

        @self._app.get(
            "/api/tags/list",
            response_model=ListTagsResponse,
            tags=[ApiTags.TAGS],
            operation_id="list_tags",
        )
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
            "/api/tags/update",
            response_model=UpdateTagResponse,
            tags=[ApiTags.TAGS],
            operation_id="update_tag",
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
                return UpdateTagResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "tag"),
                )
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
            "/api/settings/{key}",
            response_model=SettingResponse,
            tags=[ApiTags.SETTINGS],
            operation_id="get_setting",
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
            "/api/settings/{key}",
            response_model=SettingResponse,
            tags=[ApiTags.SETTINGS],
            operation_id="set_setting",
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
        async def get_image_metadata(
            payload: GetImageMetadataPayload,
        ) -> GetImageMetadataResponse:
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
        async def update_image_metadata(
            payload: UpdateMetadataPayload,
        ) -> UpdateMetadataResponse:
            image_path = payload.image_path
            existing = await self._manager.get_image_metadata(image_path)
            if existing is None:
                return UpdateMetadataResponse(status=JsonValues.FAILURE, message="Image not found")

            update = ImageRecordUpdate(
                source_uuid=payload.source_uuid,
                bboxes=[
                    BBoxUpdate(
                        uuid=b.uuid,
                        label_uuid=b.label_uuid,
                        x=b.x,
                        y=b.y,
                        width=b.width,
                        height=b.height,
                        tag_uuids=b.tag_uuids or [],
                        # None is passed through, not coerced: the manager reads
                        # it as "preserve what's stored for this bbox".
                        metadata_json=b.extra,
                    )
                    for b in payload.boxes
                    if b.label_uuid
                ],
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
                    rel_path=path,
                    patterns=[pattern],
                    recursive=recursive,
                    include_db_status=include_db_status,
                    offset=offset,
                    limit=limit,
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

        @self._app.post(
            "/api/images/upload",
            response_model=UploadImagesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="upload_images",
        )
        async def upload_images(
            files: List[UploadFile] = File(...),
            path: str = Form(""),
            overwrite: bool = Form(False),
            source_uuid: str = Form(""),
        ) -> UploadImagesResponse:
            """Import images from the user's machine into FILES_ROOT.

            Registered on arrival so they are searchable immediately, rather than
            only once something annotates them.

            One bad file does not fail the batch: dropping twenty images and
            getting nothing because the nineteenth was a PDF would be worse than
            useless, so each is reported on its own.
            """
            results: List[UploadedImageRead] = []
            for upload in files:
                name = upload.filename or "(unnamed)"
                try:
                    data = await upload.read()
                    saved = await self._manager.save_uploaded_image(
                        rel_dir=path,
                        filename=name,
                        data=data,
                        overwrite=overwrite,
                        source_uuid=source_uuid or None,
                    )
                    results.append(
                        UploadedImageRead(
                            filename=name,
                            ok=True,
                            path=saved["path"],
                            uuid=saved["uuid"],
                            duplicate_of=saved["duplicate_of"],
                            duplicate_distance=saved["duplicate_distance"],
                            renamed_from=saved["renamed_from"],
                        )
                    )
                except ValueError as e:
                    # Rejected input (bad name, wrong type, undecodable): the
                    # message is written for the user, so pass it through.
                    results.append(UploadedImageRead(filename=name, ok=False, message=str(e)))
                except Exception as e:
                    logger.exception("Failed to save uploaded image %r", name)
                    results.append(UploadedImageRead(filename=name, ok=False, message=str(e)))
                finally:
                    await upload.close()

            # SUCCESS as long as the request was handled; per-file outcomes carry
            # the detail. A partial failure is not a failed request.
            return UploadImagesResponse(status=JsonValues.SUCCESS, results=results)

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
                        status=JsonValues.FAILURE,
                        message=f"File {image_path} not found",
                        filename=image_path,
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
            operation_id="set_image_label_status",
        )
        async def set_image_label_status(
            payload: SetImageLabelPayload,
        ) -> SetImageLabelResponse:
            outcome = await self._manager.set_image_label_status(payload.image_path, payload.label_uuid, payload.status)
            if outcome == "not_found":
                return SetImageLabelResponse(status=JsonValues.FAILURE, message="Image not found")
            if outcome == "skipped":
                # Deliberate business-rule rejection, not a failure: the stored
                # state is exactly correct. Keep status=success so the client
                # doesn't treat this as something going wrong.
                return SetImageLabelResponse(
                    status=JsonValues.SUCCESS,
                    skipped=True,
                    message="Cannot mark label absent: the image has a verified bbox with this label",
                )
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
            "/api/images/labels/batch",
            response_model=BatchImageLabelsResponse,
            tags=[ApiTags.IMAGES],
            operation_id="get_image_labels_batch",
        )
        async def get_image_labels_batch(payload: BatchImageLabelsPayload) -> BatchImageLabelsResponse:
            """Return image-label rows for a batch of image UUIDs in one query.

            Images with no rows (and unknown UUIDs) are omitted from the map.
            """
            image_labels = await self._manager.get_image_labels_bulk(payload.image_uuids)
            return BatchImageLabelsResponse(image_labels=image_labels)

        @self._app.get(
            "/api/image-labels",
            response_model=ListImagesForLabelResponse,
            tags=[ApiTags.IMAGES],
            operation_id="list_images_for_label",
        )
        async def list_images_for_label(
            label_uuid: str,
            status: Optional[List[ImageItemStatus]] = Query(default=None),
        ) -> ListImagesForLabelResponse:
            """List all images labelled with a given label, filtered by status(es).

            Used by the training dataset loader to enumerate present/absent examples.
            """
            statuses = status if status is not None else list(ImageItemStatus)
            rows = await self._manager.list_images_for_label(label_uuid, statuses)
            return ListImagesForLabelResponse(images=rows)

        @self._app.post(
            "/api/images/tags/set",
            response_model=SetImageTagResponse,
            tags=[ApiTags.IMAGES],
            operation_id="set_image_tag",
        )
        async def set_image_tag(payload: SetImageTagPayload) -> SetImageTagResponse:
            ok = await self._manager.set_image_tag(payload.image_path, payload.tag_uuid, payload.action)
            if not ok:
                return SetImageTagResponse(status=JsonValues.FAILURE, message="Image not found")
            return SetImageTagResponse()

        @self._app.get(
            "/api/images/{image_path:path}/tags",
            response_model=GetImageTagsResponse,
            tags=[ApiTags.IMAGES],
            operation_id="get_image_tags",
        )
        async def get_image_tags(image_path: str) -> GetImageTagsResponse:
            tags = await self._manager.get_image_tags(image_path)
            if tags is None:
                return GetImageTagsResponse(status=JsonValues.FAILURE, message="Image not found")
            return GetImageTagsResponse(tags=tags)

        @self._app.post(
            "/api/images/tags/set-status",
            response_model=SetImageTagStatusResponse,
            tags=[ApiTags.IMAGES],
            operation_id="set_image_tag_status",
        )
        async def set_image_tag_status(
            payload: SetImageTagStatusPayload,
        ) -> SetImageTagStatusResponse:
            ok = await self._manager.set_image_tag_status(payload.image_path, payload.tag_uuid, payload.status)
            if not ok:
                return SetImageTagStatusResponse(status=JsonValues.FAILURE, message="Image not found")
            return SetImageTagStatusResponse()

        @self._app.get(
            "/api/images/{image_path:path}/tags/statuses",
            response_model=GetImageTagStatusesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="get_image_tag_statuses",
        )
        async def get_image_tag_statuses(
            image_path: str,
        ) -> GetImageTagStatusesResponse:
            statuses = await self._manager.get_image_tag_statuses(image_path)
            if statuses is None:
                return GetImageTagStatusesResponse(status=JsonValues.FAILURE, message="Image not found")
            return GetImageTagStatusesResponse(tag_statuses=statuses)

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
            ok = await self._manager.update_single_bbox(
                bbox_uuid, payload.label_uuid, payload.tag_uuids, metadata_json=payload.metadata_json
            )
            if not ok:
                raise HTTPException(status_code=404, detail="Bbox not found")
            return Response(status_code=204)

        @self._app.post(
            "/api/images/bbox",
            response_model=AddBboxResponse,
            tags=[ApiTags.IMAGES],
            operation_id="add_bbox",
        )
        async def add_bbox(payload: AddBboxPayload) -> AddBboxResponse:
            """Insert a single new bbox on an image without disturbing its existing
            bboxes (the filter-chain add_bbox action uses this)."""
            bbox_uuid = await self._manager.add_single_bbox(
                payload.image_path,
                payload.label_uuid,
                payload.x,
                payload.y,
                payload.width,
                payload.height,
                payload.tag_uuids,
                metadata_json=payload.metadata_json,
            )
            if bbox_uuid is None:
                raise HTTPException(status_code=404, detail="Image not found")
            return AddBboxResponse(bbox_uuid=bbox_uuid)

        @self._app.post(
            "/api/images/bbox/thumbnails/invalidate",
            tags=[ApiTags.IMAGES],
            operation_id="invalidate_bbox_thumbnails",
        )
        async def invalidate_bbox_thumbnails(
            payload: InvalidateThumbnailsPayload,
        ) -> StatusResponse:
            await self._manager.invalidate_bbox_thumbnails(payload.bbox_uuids)
            return StatusResponse()

        @self._app.post(
            "/api/images/bbox/count",
            response_model=CountBboxesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="count_bboxes",
        )
        async def count_bboxes_endpoint(
            payload: CountBboxesPayload,
        ) -> CountBboxesResponse:
            """Count the bboxes a bulk-apply with this same filter would modify."""
            try:
                count = await self._manager.count_bboxes_matching(
                    search_params=payload.filter,
                    bbox_tag_filter=payload.bbox_tag_filter,
                    bbox_label_filter=payload.bbox_label_filter,
                    show_all_bboxes=payload.show_all_bboxes,
                    bbox_verification=payload.bbox_verification,
                    bbox_confidence=payload.bbox_confidence,
                )
                return CountBboxesResponse(count=count)
            except Exception as e:
                logger.exception("count_bboxes failed")
                return CountBboxesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/bbox/bulk-apply",
            response_model=BulkApplyAllResponse,
            tags=[ApiTags.IMAGES],
            operation_id="bulk_apply_bboxes",
        )
        async def bulk_apply_bboxes_endpoint(
            payload: BulkApplyAllPayload,
        ) -> BulkApplyAllResponse:
            """Apply an action to all bboxes matching the given filter across all pages."""
            try:
                modified_count, fail_count = await self._manager.bulk_apply_bboxes(
                    search_params=payload.filter,
                    action=payload.action,
                    label_uuid=payload.label_uuid,
                    tag_uuids=payload.tag_uuids,
                    bbox_tag_filter=payload.bbox_tag_filter,
                    bbox_label_filter=payload.bbox_label_filter,
                    show_all_bboxes=payload.show_all_bboxes,
                    bbox_verification=payload.bbox_verification,
                    bbox_confidence=payload.bbox_confidence,
                )
                return BulkApplyAllResponse(modified_count=modified_count, fail_count=fail_count)
            except Exception as e:
                logger.exception("bulk_apply_bboxes failed")
                return BulkApplyAllResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/bbox/bulk-apply-by-uuids",
            response_model=BulkApplyAllResponse,
            tags=[ApiTags.IMAGES],
            operation_id="bulk_apply_bboxes_by_uuids",
        )
        async def bulk_apply_bboxes_by_uuids_endpoint(
            payload: BulkApplyBboxesByUuidsPayload,
        ) -> BulkApplyAllResponse:
            """Apply an action to an explicit list of bboxes (no filter equivalent,
            e.g. similarity/diversity search results)."""
            try:
                modified_count, fail_count = await self._manager.bulk_apply_bboxes_by_uuids(
                    bbox_uuids=payload.bbox_uuids,
                    action=payload.action,
                    label_uuid=payload.label_uuid,
                    tag_uuids=payload.tag_uuids,
                    bbox_geometry=payload.bbox_geometry,
                    mark_verified=payload.mark_verified,
                )
                return BulkApplyAllResponse(modified_count=modified_count, fail_count=fail_count)
            except Exception as e:
                logger.exception("bulk_apply_bboxes_by_uuids failed")
                return BulkApplyAllResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/bulk-apply-by-uuids",
            response_model=BulkApplyAllResponse,
            tags=[ApiTags.IMAGES],
            operation_id="bulk_apply_images_by_uuids",
        )
        async def bulk_apply_images_by_uuids_endpoint(
            payload: BulkApplyImagesByUuidsPayload,
        ) -> BulkApplyAllResponse:
            """Apply an image-level action to an explicit list of images (no filter
            equivalent, e.g. similarity/diversity search results)."""
            try:
                modified_count, fail_count, skipped_count = await self._manager.bulk_apply_images_by_uuids(
                    image_uuids=payload.image_uuids,
                    action=payload.action,
                    label_uuids=payload.label_uuids,
                    tag_uuids=payload.tag_uuids,
                )
                return BulkApplyAllResponse(
                    modified_count=modified_count, fail_count=fail_count, skipped_count=skipped_count
                )
            except Exception as e:
                logger.exception("bulk_apply_images_by_uuids failed")
                return BulkApplyAllResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/bulk-apply",
            response_model=BulkApplyAllResponse,
            tags=[ApiTags.IMAGES],
            operation_id="bulk_apply_images",
        )
        async def bulk_apply_images_endpoint(
            payload: BulkApplyImagesPayload,
        ) -> BulkApplyAllResponse:
            """Apply an image-level action to every image matching the given filter,
            across all pages — the image-kind counterpart of bulk_apply_bboxes_endpoint."""
            try:
                modified_count, fail_count, skipped_count = await self._manager.bulk_apply_images(
                    search_params=payload.filter,
                    action=payload.action,
                    label_uuids=payload.label_uuids,
                    tag_uuids=payload.tag_uuids,
                )
                return BulkApplyAllResponse(
                    modified_count=modified_count, fail_count=fail_count, skipped_count=skipped_count
                )
            except Exception as e:
                logger.exception("bulk_apply_images failed")
                return BulkApplyAllResponse(status=JsonValues.FAILURE, message=str(e))

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

        # Same greedy-path caveat as the bbox route above: this must precede the
        # generic image thumbnail route or {image_path:path} eats "/region/thumbnail".
        @self._app.get(
            "/api/images/{image_path:path}/region/thumbnail",
            tags=[ApiTags.IMAGES],
            operation_id="get_region_thumbnail",
        )
        async def get_region_thumbnail(
            image_path: str,
            x: float,
            y: float,
            width_frac: float,
            height_frac: float,
            width: int = 200,
            color: Optional[str] = None,
            pad: float = 0.2,
            outline: bool = True,
        ) -> Response:
            """Cropped thumbnail of an arbitrary normalized region — the bbox
            thumbnail's sibling for boxes that have no bbox row yet (a preview
            run's proposed *new* boxes).

            `pad=0&outline=false` returns the bare contents of the region, for a
            caller whose own frame is already the box."""
            png_bytes = await self._manager.get_region_thumbnail(
                image_path,
                x=x,
                y=y,
                width_frac=width_frac,
                height_frac=height_frac,
                width=width,
                color=color,
                pad_frac=pad,
                outline=outline,
            )
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
        async def list_image_stubs_needing_phash(
            payload: ImageStubsNeedingPhashPayload,
        ) -> ListImageStubsResponse:
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
        async def reconcile_image(
            payload: ReconcileImagePayload,
        ) -> ReconcileImageResponse:
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
        async def prune_missing_images(
            payload: PruneImagesPayload,
        ) -> PruneImagesResponse:
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

        @self._app.post(
            "/api/images/list-by-directory",
            response_model=ListImagesByDirectoryResponse,
            tags=[ApiTags.IMAGES],
            operation_id="list_images_by_directory",
        )
        async def list_images_by_directory(
            payload: ListImagesByDirectoryPayload,
        ) -> ListImagesByDirectoryResponse:
            """Image records (uuid + FILES_ROOT-relative path) under the given directories.

            Reads the DB only — unlike `/api/images/list`, which walks the disk. That is
            the point: a directory holding a million untracked frames costs the same here
            as an empty one, because only records are counted.
            """
            try:
                images = await self._manager.list_images_by_directory(payload.directories)
                return ListImagesByDirectoryResponse(
                    status=JsonValues.SUCCESS,
                    images=[ImageLocationRead(**img) for img in images],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListImagesByDirectoryResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/relocate",
            response_model=RelocateImagesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="relocate_images",
        )
        async def relocate_images(payload: RelocateImagesPayload) -> RelocateImagesResponse:
            """Re-point image records at new paths. Does NOT move or copy any file.

            The caller writes the bytes and then tells the DB where they went, so a
            failed copy never produces a record pointing at a file that was never made.
            """
            try:
                result = await self._manager.relocate_images(
                    [{"uuid": m.uuid, "new_path": m.new_path} for m in payload.moves]
                )
                return RelocateImagesResponse(
                    status=JsonValues.SUCCESS,
                    updated=result["updated"],
                    errors=[RelocateImageError(**e) for e in result["errors"]],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return RelocateImagesResponse(status=JsonValues.FAILURE, message=str(e))

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
        async def find_similar_images(
            payload: FindSimilarImagesPayload,
        ) -> FindSimilarImagesResponse:
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
        async def find_similar_bboxes(
            payload: FindSimilarBboxesPayload,
        ) -> FindSimilarBboxesResponse:
            try:
                matches = await self._manager.find_similar_bboxes_enriched(
                    payload.phash,
                    payload.label_uuids,
                    payload.threshold,
                    payload.tag_uuids,
                    payload.exclude_tag_uuids,
                    payload.tag_match_mode,
                )
                return FindSimilarBboxesResponse(
                    status=JsonValues.SUCCESS,
                    matches=[
                        SimilarBboxMatch(
                            bbox_uuid=bbox_uuid,
                            image_uuid=image_uuid,
                            label_uuid=lbl,
                            distance=dist,
                        )
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
        async def find_similar_images_by_uuid(
            payload: FindSimilarImagesByUuidPayload,
        ) -> FindSimilarImagesResponse:
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
        async def find_similar_bboxes_by_uuid(
            payload: FindSimilarBboxesByUuidPayload,
        ) -> FindSimilarBboxesResponse:
            try:
                matches = await self._manager.find_similar_bboxes_by_uuid(
                    payload.bbox_uuid,
                    payload.label_uuids,
                    payload.threshold,
                    payload.tag_uuids,
                    payload.exclude_tag_uuids,
                    payload.tag_match_mode,
                )
                return FindSimilarBboxesResponse(
                    status=JsonValues.SUCCESS,
                    matches=[
                        SimilarBboxMatch(
                            bbox_uuid=bbox_uuid,
                            image_uuid=image_uuid,
                            label_uuid=lbl,
                            distance=dist,
                        )
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
                    payload.label_uuids,
                    payload.threshold,
                    payload.tag_uuids,
                    payload.exclude_tag_uuids,
                    payload.tag_match_mode,
                )
                return FindSimilarBboxesResponse(
                    status=JsonValues.SUCCESS,
                    matches=[
                        SimilarBboxMatch(
                            bbox_uuid=bbox_uuid,
                            image_uuid=image_uuid,
                            label_uuid=lbl,
                            distance=dist,
                        )
                        for bbox_uuid, image_uuid, dist, lbl in matches
                    ],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return FindSimilarBboxesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/find-dissimilar-by-image-uuid",
            response_model=FindDissimilarImagesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="find_dissimilar_images_by_uuid",
        )
        async def find_dissimilar_images_by_uuid(
            payload: FindDissimilarImagesByUuidPayload,
        ) -> FindDissimilarImagesResponse:
            try:
                candidate_set = set(payload.candidate_uuids) if payload.candidate_uuids else None
                ranked = await self._manager.rank_images_by_diversity(payload.image_uuid, candidate_set, payload.count)
                return FindDissimilarImagesResponse(
                    status=JsonValues.SUCCESS,
                    matches=[DissimilarImageMatch(image_uuid=uuid, rank=i) for i, uuid in enumerate(ranked)],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return FindDissimilarImagesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/bbox/find-dissimilar-by-bbox-uuid",
            response_model=FindDissimilarBboxesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="find_dissimilar_bboxes_by_uuid",
        )
        async def find_dissimilar_bboxes_by_uuid(
            payload: FindDissimilarBboxesByUuidPayload,
        ) -> FindDissimilarBboxesResponse:
            try:
                ranked = await self._manager.rank_bboxes_by_diversity(
                    payload.bbox_uuid, payload.label_uuids or None, payload.count
                )
                return FindDissimilarBboxesResponse(
                    status=JsonValues.SUCCESS,
                    matches=[
                        DissimilarBboxMatch(
                            bbox_uuid=b_uuid,
                            image_uuid=img_uuid,
                            label_uuid=lbl,
                            rank=i,
                        )
                        for i, (b_uuid, img_uuid, lbl) in enumerate(ranked)
                    ],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return FindDissimilarBboxesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/find-dissimilar-global",
            response_model=FindDissimilarImagesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="find_dissimilar_images_global",
        )
        async def find_dissimilar_images_global(
            payload: FindDissimilarGlobalPayload,
        ) -> FindDissimilarImagesResponse:
            try:
                ranked = await self._manager.get_most_unique_images(
                    payload.count, payload.seed_image_uuid, payload.candidate_uuids
                )
                return FindDissimilarImagesResponse(
                    status=JsonValues.SUCCESS,
                    matches=[DissimilarImageMatch(image_uuid=uuid, rank=i) for i, uuid in enumerate(ranked)],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return FindDissimilarImagesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/images/batch-by-uuids",
            response_model=BatchImagesResponse,
            tags=[ApiTags.IMAGES],
            operation_id="batch_images_by_uuids",
        )
        async def batch_images_by_uuids(
            payload: BatchByUuidsPayload,
        ) -> BatchImagesResponse:
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
        async def check_dedup_bulk(
            payload: CheckDedupBulkPayload,
        ) -> CheckDedupBulkResponse:
            try:
                result = self._manager.check_dedup_bulk(
                    payload.image_phash,
                    payload.image_threshold,
                    [
                        {
                            "phash": b.phash,
                            "label_uuid": b.label_uuid,
                            "threshold": b.threshold,
                        }
                        for b in payload.bboxes
                    ],
                )
                return CheckDedupBulkResponse(
                    status=JsonValues.SUCCESS,
                    image_is_dup=result["image_is_dup"],
                    bbox_flags=result["bbox_flags"],
                    image_min_distance=result.get("image_min_distance"),
                    bbox_min_distances=result.get("bbox_min_distances", []),
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
            "/api/images/phash/clear-and-rebuild",
            response_model=RebuildHashIndexResponse,
            tags=[ApiTags.IMAGES],
            operation_id="clear_and_rebuild_hash_index",
        )
        async def clear_and_rebuild_hash_index() -> RebuildHashIndexResponse:
            try:
                stats = await self._manager.rebuild_hash_index(clear_first=True)
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
        async def create_task_config(
            payload: CreateTaskConfigPayload,
        ) -> TaskConfigReadResponse:
            try:
                config = await self._manager.create_task_config(
                    name=payload.name,
                    typename=payload.typename,
                    params=payload.params,
                    description=payload.description,
                    is_temporary=bool(payload.is_temporary),
                )
                return TaskConfigReadResponse(
                    status=JsonValues.SUCCESS,
                    config=TaskConfigRead.model_validate(config),
                )
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
            is_temporary: Optional[bool] = Query(default=None),
        ) -> ListTaskConfigsResponse:
            try:
                configs = await self._manager.list_task_configs(config_uuids=uuids or None, is_temporary=is_temporary)
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
        async def delete_task_configs(
            payload: DeleteTaskConfigsPayload,
        ) -> StatusResponse:
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
        async def create_task_instance(
            payload: CreateTaskInstancePayload,
        ) -> TaskInstanceReadResponse:
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
            include_resume_data: bool = False,
        ) -> ListTaskInstancesResponse:
            try:
                instances = await self._manager.list_task_instances(
                    task_uuids=uuids or None,
                    resumable=resumable,
                )
                reads = [TaskInstanceRead.model_validate(i) for i in instances]
                if not include_resume_data:
                    # A checkpoint is internal state — only the tasks_server paths that
                    # actually restart a task read it, and it is the heaviest field on an
                    # instance (~500 KB for a finished TrainModelTask). This listing is
                    # polled twice a second by the task SSE stream, so shipping checkpoints
                    # by default cost megabytes of JSON per frame across three services.
                    # Dropping it from the response only; the column itself is untouched.
                    reads = [r.model_copy(update={"resume_data": None}) for r in reads]
                return ListTaskInstancesResponse(
                    status=JsonValues.SUCCESS,
                    instances=reads,
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
        async def delete_task_instances(
            payload: DeleteTaskInstancesPayload,
        ) -> StatusResponse:
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
                await self._manager.set_task_status(instance_uuid, payload.status, payload.error_message)
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
                    status_data_json=payload.status_data_json,
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
        async def get_task_results(
            payload: GetTaskResultsPayload,
        ) -> GetTaskResultsResponse:
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
        async def list_datasets(include_views: bool = False) -> ListDatasetsResponse:
            """List Datasets. Saved Views are excluded unless include_views=true.

            Defaulting to promoted rows keeps every existing caller correct: the
            training-config picker, the model-config picker and the dataset
            checklist all mean "a trainable Dataset", and would silently start
            offering Views if this defaulted the other way.
            """
            try:
                datasets = await self._manager.list_datasets(include_views=include_views)
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
                return DatasetResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "dataset"),
                )
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
                return DatasetResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "dataset"),
                )
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
            except ValueError as e:
                raise HTTPException(status_code=409, detail=str(e))
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/datasets/image-uuids",
            response_model=DatasetImageUuidsResponse,
            tags=[ApiTags.DATASETS],
            operation_id="get_dataset_image_uuids",
        )
        async def get_dataset_image_uuids(
            dataset_uuid: List[str] = Query(default=[]),
        ) -> DatasetImageUuidsResponse:
            try:
                uuids = await self._manager.get_dataset_image_uuids(dataset_uuid)
                return DatasetImageUuidsResponse(status=JsonValues.SUCCESS, image_uuids=uuids)
            except Exception as e:
                logger.exception("Failed to fetch dataset image UUIDs")
                return DatasetImageUuidsResponse(status=JsonValues.FAILURE, message=str(e))

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
        async def create_model_config(
            payload: ModelConfigCreate,
        ) -> ModelConfigResponse:
            try:
                config = await self._manager.create_model_config(payload)
                return ModelConfigResponse(status=JsonValues.SUCCESS, config=config)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return ModelConfigResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "model configuration"),
                )
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

        @self._app.get(
            "/api/model-config-nodes/{node_uuid}",
            response_model=ModelConfigNodeRead,
            tags=[ApiTags.MODEL_CONFIGS],
            operation_id="get_model_config_node",
        )
        async def get_model_config_node(node_uuid: str):
            try:
                node = await self._manager.get_model_config_node(node_uuid)
                if node is None:
                    return JSONResponse({"detail": "Model config node not found"}, status_code=404)
                return node
            except Exception as e:
                logger.error(e, exc_info=True)
                return JSONResponse({"detail": str(e)}, status_code=500)

        @self._app.get(
            "/api/model-configs/{uuid}/training-targets",
            response_model=TrainingTargetsResponse,
            tags=[ApiTags.MODEL_CONFIGS],
            operation_id="get_model_config_training_targets",
        )
        async def get_model_config_training_targets(uuid: str) -> TrainingTargetsResponse:
            try:
                targets = await self._manager.derive_training_targets(uuid)
                if targets is None:
                    return TrainingTargetsResponse(status=JsonValues.FAILURE, message="Model config not found")
                return TrainingTargetsResponse(
                    status=JsonValues.SUCCESS,
                    targets=[TrainingTarget(**t) for t in targets],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return TrainingTargetsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/model-configs/{uuid}/training-targets/{node_uuid}/detection-dataset",
            response_model=DetectionDatasetResponse,
            tags=[ApiTags.MODEL_CONFIGS],
            operation_id="get_training_target_detection_dataset",
        )
        async def get_training_target_detection_dataset(
            uuid: str,
            node_uuid: str,
            payload: TrainingTargetDetectionDatasetPayload,
        ) -> DetectionDatasetResponse:
            try:
                result = await self._manager.get_training_target_detection_dataset(
                    uuid,
                    node_uuid,
                    payload.dataset_uuids,
                    label_uuids=payload.label_uuids,
                    class_remap=payload.class_remap,
                    negative_ratio=payload.negative_ratio,
                    negative_ratio_floor=payload.negative_ratio_floor,
                )
                if result is None:
                    return DetectionDatasetResponse(
                        status=JsonValues.FAILURE, message="Model config or training target not found"
                    )
                return DetectionDatasetResponse(
                    status=JsonValues.SUCCESS,
                    class_names=result["class_names"],
                    images=[
                        DetectionImageEntry(
                            image_uuid=img["image_uuid"],
                            filename=img["filename"],
                            annotations=img["annotations"],
                        )
                        for img in result["images"]
                    ],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return DetectionDatasetResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/model-configs/{uuid}/training-targets/{node_uuid}/crop-dataset",
            response_model=CropDatasetResponse,
            tags=[ApiTags.MODEL_CONFIGS],
            operation_id="get_training_target_crop_dataset",
        )
        async def get_training_target_crop_dataset(
            uuid: str,
            node_uuid: str,
            payload: TrainingTargetCropDatasetPayload,
        ) -> CropDatasetResponse:
            try:
                result = await self._manager.get_training_target_crop_dataset(
                    uuid,
                    node_uuid,
                    payload.dataset_uuids,
                    label_uuids=payload.label_uuids,
                    class_remap=payload.class_remap,
                )
                if result is None:
                    return CropDatasetResponse(
                        status=JsonValues.FAILURE, message="Model config or training target not found"
                    )
                return CropDatasetResponse(
                    status=JsonValues.SUCCESS,
                    class_names=result["class_names"],
                    class_label_uuids=result["class_label_uuids"],
                    images=[
                        CropImageEntry(
                            image_uuid=img["image_uuid"],
                            filename=img["filename"],
                            crops=[CropEntry(**c) for c in img["crops"]],
                        )
                        for img in result["images"]
                    ],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return CropDatasetResponse(status=JsonValues.FAILURE, message=str(e))

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
                return ModelConfigResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "model configuration"),
                )
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

        # ── Training configs ──────────────────────────────────────────────────

        @self._app.get(
            "/api/training-configs",
            response_model=ListTrainingConfigsResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="list_training_configs",
        )
        async def list_training_configs() -> ListTrainingConfigsResponse:
            try:
                configs = await self._manager.list_training_configs()
                return ListTrainingConfigsResponse(status=JsonValues.SUCCESS, configs=configs)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListTrainingConfigsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/training-configs",
            response_model=TrainingConfigResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="create_training_config",
        )
        async def create_training_config(
            payload: TrainingConfigCreate,
        ) -> TrainingConfigResponse:
            try:
                config = await self._manager.create_training_config(payload)
                return TrainingConfigResponse(status=JsonValues.SUCCESS, config=config)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return TrainingConfigResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "training config"),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return TrainingConfigResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/training-configs/{uuid}",
            response_model=TrainingConfigResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="get_training_config",
        )
        async def get_training_config(uuid: str) -> TrainingConfigResponse:
            try:
                config = await self._manager.get_training_config(uuid)
                if config is None:
                    return TrainingConfigResponse(status=JsonValues.FAILURE, message="Training config not found")
                return TrainingConfigResponse(status=JsonValues.SUCCESS, config=config)
            except Exception as e:
                logger.error(e, exc_info=True)
                return TrainingConfigResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.put(
            "/api/training-configs/{uuid}",
            response_model=TrainingConfigResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="update_training_config",
        )
        async def update_training_config(uuid: str, payload: TrainingConfigUpdate) -> TrainingConfigResponse:
            try:
                config = await self._manager.update_training_config(uuid, payload)
                if config is None:
                    return TrainingConfigResponse(status=JsonValues.FAILURE, message="Training config not found")
                return TrainingConfigResponse(status=JsonValues.SUCCESS, config=config)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return TrainingConfigResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "training config"),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return TrainingConfigResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/training-configs/{uuid}",
            response_model=StatusResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="delete_training_config",
        )
        async def delete_training_config(uuid: str) -> StatusResponse:
            try:
                ok = await self._manager.delete_training_config(uuid)
                if not ok:
                    return StatusResponse(status=JsonValues.FAILURE, message="Training config not found")
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/training-configs/{uuid}/readiness",
            response_model=TrainingConfigReadinessResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="get_training_config_readiness",
        )
        async def get_training_config_readiness(
            uuid: str,
        ) -> TrainingConfigReadinessResponse:
            try:
                result = await self._manager.get_training_config_readiness(uuid)
                if result is None:
                    return TrainingConfigReadinessResponse(
                        status=JsonValues.FAILURE, message="Training config not found"
                    )
                return TrainingConfigReadinessResponse(status=JsonValues.SUCCESS, **result)
            except Exception as e:
                logger.error(e, exc_info=True)
                return TrainingConfigReadinessResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/training-configs/{uuid}/readiness-for-images",
            response_model=TrainingConfigReadinessResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="get_training_config_readiness_for_images",
        )
        async def get_training_config_readiness_for_images(
            uuid: str,
            payload: TrainingConfigImageSubsetPayload,
        ) -> TrainingConfigReadinessResponse:
            """Same counts as GET /readiness, over only the supplied images.

            The Runs launcher replays a previous run's recorded manifest through
            this to say how many *positive* images have appeared since — a delta
            in ready_count alone is dominated by confirmed-absence negatives and
            reads as far more new training signal than there is."""
            try:
                result = await self._manager.get_training_config_readiness(
                    uuid, restrict_to_image_uuids=payload.image_uuids
                )
                if result is None:
                    return TrainingConfigReadinessResponse(
                        status=JsonValues.FAILURE, message="Training config not found"
                    )
                return TrainingConfigReadinessResponse(status=JsonValues.SUCCESS, **result)
            except Exception as e:
                logger.error(e, exc_info=True)
                return TrainingConfigReadinessResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/training-configs/{uuid}/detection-dataset",
            response_model=DetectionDatasetResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="get_training_config_detection_dataset",
        )
        async def get_training_config_detection_dataset(
            uuid: str,
        ) -> DetectionDatasetResponse:
            try:
                result = await self._manager.get_training_config_detection_dataset(uuid)
                if result is None:
                    return DetectionDatasetResponse(status=JsonValues.FAILURE, message="Training config not found")
                return DetectionDatasetResponse(
                    status=JsonValues.SUCCESS,
                    class_names=result["class_names"],
                    images=[
                        DetectionImageEntry(
                            image_uuid=img["image_uuid"],
                            filename=img["filename"],
                            annotations=img["annotations"],
                        )
                        for img in result["images"]
                    ],
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return DetectionDatasetResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/training-configs/{uuid}/label-stats",
            response_model=TrainingConfigLabelStatsResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="get_training_config_label_stats",
        )
        async def get_training_config_label_stats(
            uuid: str,
        ) -> TrainingConfigLabelStatsResponse:
            try:
                result = await self._manager.get_training_config_label_stats(uuid)
                if result is None:
                    return TrainingConfigLabelStatsResponse(
                        status=JsonValues.FAILURE, message="Training config not found"
                    )
                return TrainingConfigLabelStatsResponse(status=JsonValues.SUCCESS, **result)
            except Exception as e:
                logger.error(e, exc_info=True)
                return TrainingConfigLabelStatsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/training-configs/{uuid}/label-stats-for-images",
            response_model=TrainingConfigLabelStatsResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="get_training_config_label_stats_for_images",
        )
        async def get_training_config_label_stats_for_images(
            uuid: str,
            payload: TrainingConfigLabelStatsForImagesPayload,
        ) -> TrainingConfigLabelStatsResponse:
            """Same per-label stats as GET /label-stats, but counted over only the
            supplied images — the training page replays a previous run's recorded
            manifest through it to show what went in last time."""
            try:
                result = await self._manager.get_training_config_label_stats(
                    uuid, restrict_to_image_uuids=payload.image_uuids
                )
                if result is None:
                    return TrainingConfigLabelStatsResponse(
                        status=JsonValues.FAILURE, message="Training config not found"
                    )
                return TrainingConfigLabelStatsResponse(status=JsonValues.SUCCESS, **result)
            except Exception as e:
                logger.error(e, exc_info=True)
                return TrainingConfigLabelStatsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/training-configs/{uuid}/needs-review-images",
            response_model=ReviewQueueResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="get_training_config_needs_review_images",
        )
        async def get_training_config_needs_review_images(
            uuid: str, payload: NeedsReviewImagesPayload
        ) -> ReviewQueueResponse:
            """Return images in this training config's dataset union that have ≥1 unverified target label.

            Uses the same response shape as the review queue so the existing review
            grid can render the results unchanged.
            """
            try:
                images, has_more = await self._manager.get_training_config_needs_review_images(uuid, payload)
                return ReviewQueueResponse(images=images, has_more=has_more, total_count=len(images))
            except Exception as e:
                logger.exception("Failed to fetch needs-review images for training config %s", uuid)
                return ReviewQueueResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/training-configs/{uuid}/ready-images",
            response_model=ReviewQueueResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="get_training_config_ready_images",
        )
        async def get_training_config_ready_images(uuid: str, payload: ReadyImagesPayload) -> ReviewQueueResponse:
            """Return images in this training config's dataset union that are ready for training.

            Uses the same response shape as the review queue so the existing review
            grid can render the results unchanged.
            """
            try:
                images, has_more = await self._manager.get_training_config_ready_images(uuid, payload)
                return ReviewQueueResponse(images=images, has_more=has_more, total_count=len(images))
            except Exception as e:
                logger.exception("Failed to fetch ready images for training config %s", uuid)
                return ReviewQueueResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/training-configs/{uuid}/ready-positive-images",
            response_model=ReviewQueueResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="get_training_config_ready_positive_images",
        )
        async def get_training_config_ready_positive_images(
            uuid: str, payload: ReadyImagesPayload
        ) -> ReviewQueueResponse:
            """Return the READY images that are POSITIVE samples — fully verified with at
            least one in-scope label verified PRESENT.

            Uses the same response shape as the review queue so the existing review
            grid can render the results unchanged.
            """
            try:
                images, has_more = await self._manager.get_training_config_ready_positive_images(uuid, payload)
                return ReviewQueueResponse(images=images, has_more=has_more, total_count=len(images))
            except Exception as e:
                logger.exception("Failed to fetch ready-positive images for training config %s", uuid)
                return ReviewQueueResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/training-configs/{uuid}/ready-negative-images",
            response_model=ReviewQueueResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="get_training_config_ready_negative_images",
        )
        async def get_training_config_ready_negative_images(
            uuid: str, payload: ReadyImagesPayload
        ) -> ReviewQueueResponse:
            """Return the READY images that are NEGATIVE (background) samples — fully verified
            with every in-scope label verified ABSENT.

            Uses the same response shape as the review queue so the existing review
            grid can render the results unchanged.
            """
            try:
                images, has_more = await self._manager.get_training_config_ready_negative_images(uuid, payload)
                return ReviewQueueResponse(images=images, has_more=has_more, total_count=len(images))
            except Exception as e:
                logger.exception("Failed to fetch ready-negative images for training config %s", uuid)
                return ReviewQueueResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/training-configs/{uuid}/partial-review-images",
            response_model=ReviewQueueResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="get_training_config_partial_review_images",
        )
        async def get_training_config_partial_review_images(
            uuid: str, payload: ReadyImagesPayload
        ) -> ReviewQueueResponse:
            """Return images with PARTIAL label coverage (0 < verified_count < readiness_threshold) —
            the readiness panel's "needs review" bucket. Distinct from the bbox-scoped
            needs-review-images endpoint used by the label-stats panel.

            Uses the same response shape as the review queue so the existing review
            grid can render the results unchanged.
            """
            try:
                images, has_more = await self._manager.get_training_config_partial_review_images(uuid, payload)
                return ReviewQueueResponse(images=images, has_more=has_more, total_count=len(images))
            except Exception as e:
                logger.exception("Failed to fetch partial-review images for training config %s", uuid)
                return ReviewQueueResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/training-configs/{uuid}/not-started-images",
            response_model=ReviewQueueResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="get_training_config_not_started_images",
        )
        async def get_training_config_not_started_images(uuid: str, payload: ReadyImagesPayload) -> ReviewQueueResponse:
            """Return images with ZERO label coverage (verified_count == 0) — images that have
            never been reviewed at all, as opposed to partially reviewed.

            Uses the same response shape as the review queue so the existing review
            grid can render the results unchanged.
            """
            try:
                images, has_more = await self._manager.get_training_config_not_started_images(uuid, payload)
                return ReviewQueueResponse(images=images, has_more=has_more, total_count=len(images))
            except Exception as e:
                logger.exception("Failed to fetch not-started images for training config %s", uuid)
                return ReviewQueueResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/training-configs/{uuid}/image-gaps",
            response_model=TrainingConfigImageGapsResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="get_training_config_image_gaps",
        )
        async def get_training_config_image_gaps(
            uuid: str, payload: ImageGapsPayload
        ) -> TrainingConfigImageGapsResponse:
            """Per-image list of this training config's labels that are still unresolved.

            Batched deliberately: callers pass one rendered page of image uuids, not one
            image at a time.
            """
            try:
                gaps = await self._manager.get_training_config_image_gaps(uuid, payload.image_uuids)
                return TrainingConfigImageGapsResponse(
                    status=JsonValues.SUCCESS,
                    gaps={k: [ImageGapEntry(**entry) for entry in v] for k, v in gaps.items()},
                )
            except Exception as e:
                logger.exception("Failed to fetch image gaps for training config %s", uuid)
                return TrainingConfigImageGapsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/training-configs/{uuid}/images-by-label-status",
            response_model=ReviewQueueResponse,
            tags=[ApiTags.TRAINING_CONFIGS],
            operation_id="get_training_config_images_by_label_status",
        )
        async def get_training_config_images_by_label_status(
            uuid: str, payload: ImagesByLabelStatusPayload
        ) -> ReviewQueueResponse:
            """Return images from a training config's datasets filtered by image-label status for a given label."""
            try:
                images, has_more = await self._manager.get_training_config_images_by_label_status(uuid, payload)
                return ReviewQueueResponse(images=images, has_more=has_more, total_count=len(images))
            except Exception as e:
                logger.exception("Failed to fetch images by label status for training config %s", uuid)
                return ReviewQueueResponse(status=JsonValues.FAILURE, message=str(e))

    def _register_filter_chain_routes(self) -> None:
        @self._app.get(
            "/api/filter-chains",
            response_model=ListFilterChainsResponse,
            tags=[ApiTags.FILTER_CHAINS],
            operation_id="list_filter_chains",
        )
        async def list_filter_chains() -> ListFilterChainsResponse:
            try:
                chains = await self._manager.list_filter_chains()
                return ListFilterChainsResponse(status=JsonValues.SUCCESS, chains=chains)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListFilterChainsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/filter-chains",
            response_model=FilterChainResponse,
            tags=[ApiTags.FILTER_CHAINS],
            operation_id="create_filter_chain",
        )
        async def create_filter_chain(payload: FilterChainCreate) -> FilterChainResponse:
            try:
                chain = await self._manager.create_filter_chain(payload)
                return FilterChainResponse(status=JsonValues.SUCCESS, chain=chain)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return FilterChainResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "filter chain"),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return FilterChainResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/filter-chains/{uuid}",
            response_model=FilterChainResponse,
            tags=[ApiTags.FILTER_CHAINS],
            operation_id="get_filter_chain",
        )
        async def get_filter_chain(uuid: str) -> FilterChainResponse:
            try:
                chain = await self._manager.get_filter_chain(uuid)
                if chain is None:
                    return FilterChainResponse(status=JsonValues.FAILURE, message="Filter chain not found")
                return FilterChainResponse(status=JsonValues.SUCCESS, chain=chain)
            except Exception as e:
                logger.error(e, exc_info=True)
                return FilterChainResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/filter-chains/{uuid}",
            response_model=FilterChainResponse,
            tags=[ApiTags.FILTER_CHAINS],
            operation_id="update_filter_chain",
        )
        async def update_filter_chain(uuid: str, payload: FilterChainUpdate) -> FilterChainResponse:
            try:
                chain = await self._manager.update_filter_chain(uuid, payload)
                if chain is None:
                    return FilterChainResponse(status=JsonValues.FAILURE, message="Filter chain not found")
                return FilterChainResponse(status=JsonValues.SUCCESS, chain=chain)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return FilterChainResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "filter chain"),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return FilterChainResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/filter-chains/{uuid}",
            response_model=StatusResponse,
            tags=[ApiTags.FILTER_CHAINS],
            operation_id="delete_filter_chain",
        )
        async def delete_filter_chain(uuid: str) -> StatusResponse:
            try:
                ok = await self._manager.delete_filter_chain(uuid)
                if not ok:
                    return StatusResponse(status=JsonValues.FAILURE, message="Filter chain not found")
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

    def _register_training_collector_routes(self) -> None:
        @self._app.get(
            "/api/training-collectors",
            response_model=ListTrainingCollectorsResponse,
            tags=[ApiTags.TRAINING_COLLECTORS],
            operation_id="list_training_collectors",
        )
        async def list_training_collectors() -> ListTrainingCollectorsResponse:
            try:
                collectors = await self._manager.list_training_collectors()
                return ListTrainingCollectorsResponse(status=JsonValues.SUCCESS, collectors=collectors)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListTrainingCollectorsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/training-collectors",
            response_model=TrainingCollectorResponse,
            tags=[ApiTags.TRAINING_COLLECTORS],
            operation_id="create_training_collector",
        )
        async def create_training_collector(payload: TrainingCollectorCreate) -> TrainingCollectorResponse:
            try:
                collector = await self._manager.create_training_collector(payload)
                return TrainingCollectorResponse(status=JsonValues.SUCCESS, collector=collector)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return TrainingCollectorResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "collector"),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return TrainingCollectorResponse(status=JsonValues.FAILURE, message=str(e))

        # Before /{uuid}: FastAPI matches in registration order, and "schema" would
        # otherwise be read as a collector uuid.
        @self._app.get(
            "/api/training-collectors/schema",
            response_model=TrainingCollectorSchemaResponse,
            tags=[ApiTags.TRAINING_COLLECTORS],
            operation_id="get_training_collector_schema",
        )
        async def get_training_collector_schema() -> TrainingCollectorSchemaResponse:
            # The one definition of the collect knobs (collectorSpec.py), served so
            # the editor renders it rather than restating it in JS.
            return TrainingCollectorSchemaResponse(status=JsonValues.SUCCESS, params_schema=collector_params_schema())

        @self._app.get(
            "/api/training-collectors/{uuid}",
            response_model=TrainingCollectorResponse,
            tags=[ApiTags.TRAINING_COLLECTORS],
            operation_id="get_training_collector",
        )
        async def get_training_collector(uuid: str) -> TrainingCollectorResponse:
            try:
                collector = await self._manager.get_training_collector(uuid)
                if collector is None:
                    return TrainingCollectorResponse(status=JsonValues.FAILURE, message="Collector not found")
                return TrainingCollectorResponse(status=JsonValues.SUCCESS, collector=collector)
            except Exception as e:
                logger.error(e, exc_info=True)
                return TrainingCollectorResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/training-collectors/{uuid}",
            response_model=TrainingCollectorResponse,
            tags=[ApiTags.TRAINING_COLLECTORS],
            operation_id="update_training_collector",
        )
        async def update_training_collector(uuid: str, payload: TrainingCollectorUpdate) -> TrainingCollectorResponse:
            try:
                collector = await self._manager.update_training_collector(uuid, payload)
                if collector is None:
                    return TrainingCollectorResponse(status=JsonValues.FAILURE, message="Collector not found")
                return TrainingCollectorResponse(status=JsonValues.SUCCESS, collector=collector)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return TrainingCollectorResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "collector"),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return TrainingCollectorResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/training-collectors/{uuid}",
            response_model=StatusResponse,
            tags=[ApiTags.TRAINING_COLLECTORS],
            operation_id="delete_training_collector",
        )
        async def delete_training_collector(uuid: str) -> StatusResponse:
            try:
                ok = await self._manager.delete_training_collector(uuid)
                if not ok:
                    return StatusResponse(status=JsonValues.FAILURE, message="Collector not found")
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

    def _register_preview_run_routes(self) -> None:
        """Persisted "Run Model on Results" previews.

        These are scratch tables, not annotations — nothing here writes to
        bboxes/labels/tags except `apply`, which goes through the ordinary bulk
        bbox path.
        """

        @self._app.post(
            "/api/preview-runs",
            response_model=PreviewRunResponse,
            tags=[ApiTags.PREVIEW_RUNS],
            operation_id="create_preview_run",
        )
        async def create_preview_run(payload: PreviewRunCreate) -> PreviewRunResponse:
            try:
                run = await self._manager.create_preview_run(payload)
                return PreviewRunResponse(status=JsonValues.SUCCESS, run=run)
            except Exception as e:
                logger.error(e, exc_info=True)
                return PreviewRunResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/preview-runs",
            response_model=ListPreviewRunsResponse,
            tags=[ApiTags.PREVIEW_RUNS],
            operation_id="list_preview_runs",
        )
        async def list_preview_runs() -> ListPreviewRunsResponse:
            try:
                runs = await self._manager.list_preview_runs()
                return ListPreviewRunsResponse(status=JsonValues.SUCCESS, runs=runs)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListPreviewRunsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/preview-runs/{uuid}",
            response_model=PreviewRunResponse,
            tags=[ApiTags.PREVIEW_RUNS],
            operation_id="get_preview_run",
        )
        async def get_preview_run(uuid: str) -> PreviewRunResponse:
            try:
                run = await self._manager.get_preview_run(uuid)
                if run is None:
                    return PreviewRunResponse(status=JsonValues.FAILURE, message="Preview run not found")
                return PreviewRunResponse(status=JsonValues.SUCCESS, run=run)
            except Exception as e:
                logger.error(e, exc_info=True)
                return PreviewRunResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/preview-runs/{uuid}",
            response_model=PreviewRunResponse,
            tags=[ApiTags.PREVIEW_RUNS],
            operation_id="update_preview_run",
        )
        async def update_preview_run(uuid: str, payload: PreviewRunUpdate) -> PreviewRunResponse:
            try:
                run = await self._manager.update_preview_run(uuid, payload)
                if run is None:
                    return PreviewRunResponse(status=JsonValues.FAILURE, message="Preview run not found")
                return PreviewRunResponse(status=JsonValues.SUCCESS, run=run)
            except Exception as e:
                logger.error(e, exc_info=True)
                return PreviewRunResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/preview-runs/{uuid}",
            response_model=StatusResponse,
            tags=[ApiTags.PREVIEW_RUNS],
            operation_id="delete_preview_run",
        )
        async def delete_preview_run(uuid: str) -> StatusResponse:
            try:
                ok = await self._manager.delete_preview_run(uuid)
                if not ok:
                    return StatusResponse(status=JsonValues.FAILURE, message="Preview run not found")
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/preview-runs/{uuid}/candidates",
            response_model=PreviewCandidatesWriteResponse,
            tags=[ApiTags.PREVIEW_RUNS],
            operation_id="add_preview_candidates",
        )
        async def add_preview_candidates(
            uuid: str, payload: AddPreviewCandidatesPayload
        ) -> PreviewCandidatesWriteResponse:
            try:
                n = await self._manager.add_preview_candidates(uuid, payload.candidates, payload.conflict_policy)
                return PreviewCandidatesWriteResponse(status=JsonValues.SUCCESS, updated=n)
            except Exception as e:
                logger.error(e, exc_info=True)
                return PreviewCandidatesWriteResponse(status=JsonValues.FAILURE, message=str(e))

        # POST, not GET: the filter set (label multi-select, confidence band,
        # decision set) is a body, exactly as /api/images/review takes one.
        @self._app.post(
            "/api/preview-runs/{uuid}/candidates/query",
            response_model=ListPreviewCandidatesResponse,
            tags=[ApiTags.PREVIEW_RUNS],
            operation_id="query_preview_candidates",
        )
        async def query_preview_candidates(uuid: str, payload: PreviewCandidateQuery) -> ListPreviewCandidatesResponse:
            try:
                candidates, total = await self._manager.query_preview_candidates(uuid, payload)
                return ListPreviewCandidatesResponse(
                    status=JsonValues.SUCCESS, candidates=candidates, total_count=total
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListPreviewCandidatesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/preview-runs/{uuid}/candidates",
            response_model=PreviewCandidatesWriteResponse,
            tags=[ApiTags.PREVIEW_RUNS],
            operation_id="update_preview_candidates",
        )
        async def update_preview_candidates(
            uuid: str, payload: PreviewCandidateBulkUpdate
        ) -> PreviewCandidatesWriteResponse:
            try:
                n = await self._manager.update_preview_candidates(uuid, payload)
                return PreviewCandidatesWriteResponse(status=JsonValues.SUCCESS, updated=n)
            except Exception as e:
                logger.error(e, exc_info=True)
                return PreviewCandidatesWriteResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/preview-runs/{uuid}/apply",
            response_model=PreviewApplyResponse,
            tags=[ApiTags.PREVIEW_RUNS],
            operation_id="apply_preview_run",
        )
        async def apply_preview_run(uuid: str, payload: ApplyPreviewRunPayload) -> PreviewApplyResponse:
            try:
                result = await self._manager.apply_preview_run(uuid, mark_verified=payload.mark_verified)
                return PreviewApplyResponse(status=JsonValues.SUCCESS, result=result)
            except Exception as e:
                logger.error(e, exc_info=True)
                return PreviewApplyResponse(status=JsonValues.FAILURE, message=str(e))

    def _register_watch_profile_routes(self) -> None:
        # Registered before "/api/watch-profiles/{uuid}": FastAPI matches routes in
        # registration order, so a literal path declared after the parameterised one
        # would be shadowed by it and "trackers" would arrive as a profile uuid.
        @self._app.get(
            "/api/watch-profiles/trackers",
            response_model=ListTrackersResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="list_trackers",
        )
        async def list_trackers() -> ListTrackersResponse:
            """Available trackers and their parameter schemas, for the Watchers pane."""
            try:
                trackers = await self._manager.list_avail_trackers()
                return ListTrackersResponse(status=JsonValues.SUCCESS, trackers=trackers)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListTrackersResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/watch-profiles/actions",
            response_model=ListActionsResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="list_watch_actions",
        )
        async def list_watch_actions() -> ListActionsResponse:
            """Available actions and their parameter schemas, for the Watchers pane.

            Registered before `/{uuid}`-shaped routes for the same reason `facets` is —
            a literal path declared after a path parameter of the same shape is
            unreachable.
            """
            try:
                actions = await self._manager.list_avail_actions()
                return ListActionsResponse(status=JsonValues.SUCCESS, actions=actions)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListActionsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/watch-profiles",
            response_model=ListWatchProfilesResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="list_watch_profiles",
        )
        async def list_watch_profiles() -> ListWatchProfilesResponse:
            try:
                profiles = await self._manager.list_watch_profiles()
                return ListWatchProfilesResponse(status=JsonValues.SUCCESS, profiles=profiles)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListWatchProfilesResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/watch-profiles",
            response_model=WatchProfileResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="create_watch_profile",
        )
        async def create_watch_profile(payload: WatchProfileCreate) -> WatchProfileResponse:
            try:
                profile = await self._manager.create_watch_profile(payload)
                return WatchProfileResponse(status=JsonValues.SUCCESS, profile=profile)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return WatchProfileResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "watch profile"),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return WatchProfileResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/watch-profiles/{uuid}",
            response_model=WatchProfileResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="get_watch_profile",
        )
        async def get_watch_profile(uuid: str) -> WatchProfileResponse:
            try:
                profile = await self._manager.get_watch_profile(uuid)
                if profile is None:
                    return WatchProfileResponse(status=JsonValues.FAILURE, message="Watch profile not found")
                return WatchProfileResponse(status=JsonValues.SUCCESS, profile=profile)
            except Exception as e:
                logger.error(e, exc_info=True)
                return WatchProfileResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/watch-profiles/{uuid}",
            response_model=WatchProfileResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="update_watch_profile",
        )
        async def update_watch_profile(uuid: str, payload: WatchProfileUpdate) -> WatchProfileResponse:
            try:
                profile = await self._manager.update_watch_profile(uuid, payload)
                if profile is None:
                    return WatchProfileResponse(status=JsonValues.FAILURE, message="Watch profile not found")
                return WatchProfileResponse(status=JsonValues.SUCCESS, profile=profile)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return WatchProfileResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "watch profile"),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return WatchProfileResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/watch-profiles/{uuid}",
            response_model=StatusResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="delete_watch_profile",
        )
        async def delete_watch_profile(uuid: str) -> StatusResponse:
            try:
                ok = await self._manager.delete_watch_profile(uuid)
                if not ok:
                    return StatusResponse(status=JsonValues.FAILURE, message="Watch profile not found")
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/watch-profiles/{uuid}/validate",
            response_model=WatchProfileValidationResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="validate_watch_profile",
        )
        async def validate_watch_profile(uuid: str) -> WatchProfileValidationResponse:
            """Surface misconfigurations that would otherwise fail silently at run time."""
            try:
                report = await self._manager.validate_watch_profile(uuid)
                if report is None:
                    return WatchProfileValidationResponse(status=JsonValues.FAILURE, message="Watch profile not found")
                return WatchProfileValidationResponse(status=JsonValues.SUCCESS, **report)
            except Exception as e:
                logger.error(e, exc_info=True)
                return WatchProfileValidationResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/watch-events",
            response_model=ListWatchEventsResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="list_watch_events",
        )
        async def list_watch_events(
            query: WatchEventQuery = Depends(),
            limit: int = 200,
            offset: int = 0,
        ) -> ListWatchEventsResponse:
            try:
                events, total = await self._manager.list_watch_events(
                    limit=limit,
                    offset=offset,
                    **query.as_kwargs(),
                )
                return ListWatchEventsResponse(status=JsonValues.SUCCESS, events=events, total=total)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListWatchEventsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/watch-events",
            response_model=WatchEventResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="create_watch_event",
        )
        async def create_watch_event(payload: WatchEventCreate) -> WatchEventResponse:
            try:
                fields = payload.model_dump(exclude_none=True)
                event = await self._manager.create_watch_event(WatchEvent(**fields))
                return WatchEventResponse(status=JsonValues.SUCCESS, event=event)
            except Exception as e:
                logger.error(e, exc_info=True)
                return WatchEventResponse(status=JsonValues.FAILURE, message=str(e))

        # ⚠️ Registered before `/{uuid}`. FastAPI matches in declaration order, so a
        # literal path that comes after a path parameter of the same shape is
        # unreachable — "facets" would be read as an event uuid and 404.
        @self._app.get(
            "/api/watch-events/facets",
            response_model=WatchEventFacetsResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="get_watch_event_facets",
        )
        async def get_watch_event_facets(profile_uuid: Optional[str] = None) -> WatchEventFacetsResponse:
            try:
                facets = await self._manager.watch_event_facets(profile_uuid=profile_uuid)
                return WatchEventFacetsResponse(status=JsonValues.SUCCESS, **facets)
            except Exception as e:
                logger.error(e, exc_info=True)
                return WatchEventFacetsResponse(status=JsonValues.FAILURE, message=str(e))

        # ⚠️ Same ordering reason as `facets` above — must precede `/{uuid}`.
        @self._app.get(
            "/api/watch-events/counts",
            response_model=WatchEventCountsResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="get_watch_event_counts",
        )
        async def get_watch_event_counts() -> WatchEventCountsResponse:
            """Lifetime event counts grouped by (source, rule) — MQTT bridge counter seeding."""
            try:
                counts = await self._manager.watch_event_counts()
                return WatchEventCountsResponse(status=JsonValues.SUCCESS, counts=counts)
            except Exception as e:
                logger.error(e, exc_info=True)
                return WatchEventCountsResponse(status=JsonValues.FAILURE, message=str(e))

        # ⚠️ Same ordering reason as `facets` and `counts` above — must precede `/{uuid}`.
        @self._app.get(
            "/api/watch-events/groups",
            response_model=WatchEventGroupsResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="get_watch_event_groups",
        )
        async def get_watch_event_groups(query: WatchEventQuery = Depends()) -> WatchEventGroupsResponse:
            """The filtered events collapsed to one row per identity, biggest first.

            Deliberately *not* `/counts`, which looks superficially similar: that one
            ignores every filter, groups by a shape the MQTT bridge chose, and exists to
            seed counters. Pointing the review UI at it would couple what the user sees
            to what the bridge publishes.
            """
            try:
                groups = await self._manager.watch_event_groups(**query.as_kwargs())
                return WatchEventGroupsResponse(
                    status=JsonValues.SUCCESS,
                    groups=groups,
                    total=sum(g["count"] for g in groups),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return WatchEventGroupsResponse(status=JsonValues.FAILURE, message=str(e))

        # ⚠️ Same ordering reason as `facets` and `counts` above — must precede `/{uuid}`.
        @self._app.get(
            "/api/watch-events/sessions",
            response_model=ListWatchEventSessionsResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="list_watch_event_sessions",
        )
        async def list_watch_event_sessions(
            query: WatchEventQuery = Depends(),
            gap_s: float = DEFAULT_SESSION_GAP_S,
            limit: int = 60,
            offset: int = 0,
        ) -> ListWatchEventSessionsResponse:
            """The flat grid's events, collapsed into the visits they describe.

            Paged over *sessions*, not events, so a session can never be split across two
            pages — which is also why this cannot be done in the browser.
            """
            try:
                sessions, total = await self._manager.watch_event_sessions(
                    gap_s=gap_s,
                    limit=limit,
                    offset=offset,
                    **query.as_kwargs(),
                )
                return ListWatchEventSessionsResponse(
                    status=JsonValues.SUCCESS, sessions=sessions, total=total, gap_s=gap_s
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListWatchEventSessionsResponse(status=JsonValues.FAILURE, message=str(e))

        # ⚠️ Same ordering reason as `facets` and `counts` above — must precede `/{uuid}`.
        @self._app.get(
            "/api/watch-events/histogram",
            response_model=WatchEventHistogramResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="get_watch_event_histogram",
        )
        async def get_watch_event_histogram(
            query: WatchEventQuery = Depends(),
            bucket: str = "hour",
        ) -> WatchEventHistogramResponse:
            """Event counts per UTC time bucket per identity — the timeline's data.

            Ask for `hour` and compose local days in the browser; see
            `DbClient.watch_event_histogram` for why this endpoint has no timezone
            parameter.
            """
            try:
                result = await self._manager.watch_event_histogram(bucket=bucket, **query.as_kwargs())
                return WatchEventHistogramResponse(status=JsonValues.SUCCESS, **result)
            except Exception as e:
                logger.error(e, exc_info=True)
                return WatchEventHistogramResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/watch-events/{uuid}",
            response_model=WatchEventResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="get_watch_event",
        )
        async def get_watch_event(uuid: str) -> WatchEventResponse:
            try:
                event = await self._manager.get_watch_event(uuid)
                if event is None:
                    return WatchEventResponse(status=JsonValues.FAILURE, message="Event not found")
                return WatchEventResponse(status=JsonValues.SUCCESS, event=event)
            except Exception as e:
                logger.error(e, exc_info=True)
                return WatchEventResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/watch-events/{uuid}",
            response_model=WatchEventResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="update_watch_event",
        )
        async def update_watch_event(uuid: str, payload: WatchEventUpdate) -> WatchEventResponse:
            try:
                # exclude_unset, not exclude_none: an explicit null is how a caller
                # clears a link, and an omitted field must not be confused with one.
                event = await self._manager.update_watch_event(uuid, **payload.model_dump(exclude_unset=True))
                if event is None:
                    return WatchEventResponse(status=JsonValues.FAILURE, message="Event not found")
                return WatchEventResponse(status=JsonValues.SUCCESS, event=event)
            except Exception as e:
                logger.error(e, exc_info=True)
                return WatchEventResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/watch-events/close-stranded",
            response_model=CloseStrandedEventsResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="close_stranded_watch_events",
        )
        async def close_stranded_watch_events(payload: CloseStrandedEventsRequest) -> CloseStrandedEventsResponse:
            """Close events a killed watcher left open. Called by that watcher on start.

            One request rather than a list followed by a patch each: the caller has no
            interest in the rows, only in the invariant, and doing it in one transaction
            means a watcher that dies mid-cleanup has not half-cleaned its own history.
            """
            try:
                closed = await self._manager.close_stranded_watch_events(payload.profile_uuid)
                return CloseStrandedEventsResponse(status=JsonValues.SUCCESS, closed=closed)
            except Exception as e:
                logger.error(e, exc_info=True)
                return CloseStrandedEventsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/watch-events/delete",
            response_model=DeleteWatchEventsResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="delete_watch_events",
        )
        async def delete_watch_events(payload: DeleteWatchEventsRequest) -> DeleteWatchEventsResponse:
            """Delete a selection of events in one request.

            POST rather than DELETE because the uuids are a body, and a DELETE with a
            body is refused or silently stripped by enough proxies to be a bad bet.
            Registered before `/{uuid}` for the reason noted above `facets`, even though
            no `/{uuid}` route takes POST today — the next one would break silently.
            """
            try:
                deleted = await self._manager.delete_watch_events(payload.uuids)
                return DeleteWatchEventsResponse(status=JsonValues.SUCCESS, deleted=deleted)
            except Exception as e:
                logger.error(e, exc_info=True)
                return DeleteWatchEventsResponse(status=JsonValues.FAILURE, message=str(e))

        # ── event blips ──────────────────────────────────────────────────────
        # Their own `/api/event-blips` prefix rather than nesting under
        # `/api/watch-events/{uuid}/blips`: blips are queried *across* events for the
        # cross-event review surface, so an event-scoped path would be the wrong shape
        # for the main consumer — and it sidesteps the `/{uuid}` ordering trap above.

        @self._app.get(
            "/api/event-blips",
            response_model=ListEventBlipsResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="list_event_blips",
        )
        async def list_event_blips(
            uuids: Optional[List[str]] = Query(default=None),
            event_uuid: Optional[str] = None,
            recording_uuid: Optional[str] = None,
            kinds: Optional[List[str]] = Query(default=None),
            states: Optional[List[str]] = Query(default=None),
            harvestable: Optional[bool] = None,
            limit: int = 200,
            offset: int = 0,
        ) -> ListEventBlipsResponse:
            try:
                blips, total = await self._manager.list_event_blips(
                    uuids=uuids,
                    event_uuid=event_uuid,
                    recording_uuid=recording_uuid,
                    kinds=kinds,
                    states=states,
                    harvestable=harvestable,
                    limit=limit,
                    offset=offset,
                )
                return ListEventBlipsResponse(status=JsonValues.SUCCESS, blips=blips, total=total)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListEventBlipsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/event-blips/replace",
            response_model=EventBlipsResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="replace_event_blips",
        )
        async def replace_event_blips(payload: ReplaceEventBlipsRequest) -> EventBlipsResponse:
            """File one event's analysis result, discarding any previous one.

            POST rather than PUT because the body carries the scope as well as the
            contents — the caller does not address a resource by path.
            """
            try:
                blips = await self._manager.replace_event_blips(
                    payload.event_uuid,
                    [b.model_dump() for b in payload.blips],
                    recording_uuid=payload.recording_uuid,
                )
                return EventBlipsResponse(status=JsonValues.SUCCESS, blips=blips)
            except Exception as e:
                logger.error(e, exc_info=True)
                return EventBlipsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/event-blips/states",
            response_model=UpdatedCountResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="set_event_blip_states",
        )
        async def set_event_blip_states(payload: SetEventBlipStatesRequest) -> UpdatedCountResponse:
            """Move a selection of blips to one triage state."""
            try:
                if payload.state not in {s.value for s in BlipState}:
                    raise ValueError(f"unknown blip state {payload.state!r}")
                updated = await self._manager.set_event_blip_states(payload.uuids, payload.state)
                return UpdatedCountResponse(status=JsonValues.SUCCESS, updated=updated)
            except Exception as e:
                logger.error(e, exc_info=True)
                return UpdatedCountResponse(status=JsonValues.FAILURE, message=str(e))

        # ⚠️ Before `/{uuid}`, same reason as `facets`, `counts`, `groups` and the rest:
        # a literal segment declared after a path parameter of the same shape is
        # unreachable, and "review-states" would be read as an event uuid and 404.
        @self._app.post(
            "/api/watch-events/review-states",
            response_model=UpdatedCountResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="set_watch_event_review_states",
        )
        async def set_watch_event_review_states(payload: SetWatchEventReviewStatesRequest) -> UpdatedCountResponse:
            """Move a selection of events to one triage state.

            POST with a body for the same reason the bulk delete is one: the uuids are a
            list, and a GET or DELETE carrying one is not reliably forwarded through the
            proxy.
            """
            try:
                if payload.state not in {s.value for s in WatchEventReviewState}:
                    raise ValueError(f"unknown review state {payload.state!r}")
                updated = await self._manager.set_watch_event_review_states(payload.uuids, payload.state)
                return UpdatedCountResponse(status=JsonValues.SUCCESS, updated=updated)
            except Exception as e:
                logger.error(e, exc_info=True)
                return UpdatedCountResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/watch-events/{uuid}",
            response_model=StatusResponse,
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="delete_watch_event",
        )
        async def delete_watch_event(uuid: str) -> StatusResponse:
            try:
                ok = await self._manager.delete_watch_event(uuid)
                if not ok:
                    return StatusResponse(status=JsonValues.FAILURE, message="Event not found")
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/watch-events/{uuid}/snapshot",
            tags=[ApiTags.WATCH_PROFILES],
            operation_id="get_watch_event_snapshot",
        )
        async def get_watch_event_snapshot(uuid: str):
            """Serve the event's card art.

            The stored path is never exposed to the browser; this endpoint is the only
            way to reach the file, so a snapshot cannot be used to probe the filesystem.
            """
            path = await self._manager.watch_event_snapshot_file(uuid)
            if path is None:
                raise HTTPException(status_code=404, detail="No snapshot for this event")
            return FileResponse(path, media_type="image/jpeg")

    def _register_recording_routes(self) -> None:
        @self._app.get(
            "/api/storage/usage",
            response_model=StorageUsageResponse,
            tags=[ApiTags.RECORDINGS],
            operation_id="get_storage_usage",
        )
        async def get_storage_usage() -> StorageUsageResponse:
            """How much disk the recordings, timelapses and thumbnail cache hold.

            Walks directories, so it is deliberately its own endpoint rather than a
            field on /status — the notices loop polls that every few seconds.
            """
            try:
                usage = await self._manager.get_storage_usage()
                return StorageUsageResponse(
                    status=JsonValues.SUCCESS,
                    recordings=StorageBucketRead(**usage["recordings"]),
                    timelapse=StorageBucketRead(**usage["timelapse"]),
                    thumbnail_cache=StorageBucketRead(**usage["thumbnailCache"]),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return StorageUsageResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/storage/sweep",
            response_model=StoragePurgeResponse,
            tags=[ApiTags.RECORDINGS],
            operation_id="run_storage_sweep",
        )
        async def run_storage_sweep() -> StoragePurgeResponse:
            """Apply the configured retention bounds right now.

            Not a purge: it removes exactly what the hourly sweep would have removed
            on its own. That is what makes it the safe button — it changes *when*,
            never *what*.
            """
            try:
                recordings = await self._manager.prune_expired_recordings()
                timelapse = await self._manager.prune_expired_timelapse()
                return StoragePurgeResponse(
                    status=JsonValues.SUCCESS,
                    events=recordings.get("events", 0),
                    recordings=recordings.get("recordings", 0),
                    files=timelapse.get("files", 0) + recordings.get("orphanFiles", 0),
                    bytes=timelapse.get("bytes", 0) + recordings.get("bytes", 0),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return StoragePurgeResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/storage/timelapse",
            response_model=StoragePurgeResponse,
            tags=[ApiTags.RECORDINGS],
            operation_id="purge_timelapse_frames",
        )
        async def purge_timelapse_frames() -> StoragePurgeResponse:
            """Delete every timelapse frame, ignoring the retention bounds.

            Scoped to exactly what the timelapse sweep considers deletable — day
            folders holding frames the action wrote, never a file with a JSON
            sidecar — so it cannot reach an annotated image sharing the tree.
            """
            try:
                result = await self._manager.purge_all_timelapse_frames()
                return StoragePurgeResponse(status=JsonValues.SUCCESS, files=result["files"], bytes=result["bytes"])
            except Exception as e:
                logger.error(e, exc_info=True)
                return StoragePurgeResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/storage/thumbnail-cache",
            response_model=StoragePurgeResponse,
            tags=[ApiTags.RECORDINGS],
            operation_id="clear_thumbnail_cache",
        )
        async def clear_thumbnail_cache() -> StoragePurgeResponse:
            """Drop every cached thumbnail. They regenerate on the next request."""
            try:
                result = await self._manager.clear_thumbnail_cache()
                return StoragePurgeResponse(status=JsonValues.SUCCESS, files=result["files"], bytes=result["bytes"])
            except Exception as e:
                logger.error(e, exc_info=True)
                return StoragePurgeResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/recordings",
            response_model=ListRecordingsResponse,
            tags=[ApiTags.RECORDINGS],
            operation_id="list_recordings",
        )
        async def list_recordings(
            profile_uuid: Optional[str] = None,
            source_uuid: Optional[str] = None,
            event_uuid: Optional[str] = None,
            kind: Optional[RecordingKind] = None,
            limit: int = 200,
        ) -> ListRecordingsResponse:
            try:
                recordings = await self._manager.list_recordings(
                    profile_uuid=profile_uuid,
                    source_uuid=source_uuid,
                    event_uuid=event_uuid,
                    kind=kind.value if kind else None,
                    limit=limit,
                )
                return ListRecordingsResponse(status=JsonValues.SUCCESS, recordings=recordings)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListRecordingsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/recordings",
            response_model=RecordingResponse,
            tags=[ApiTags.RECORDINGS],
            operation_id="create_recording",
        )
        async def create_recording(payload: RecordingCreate) -> RecordingResponse:
            try:
                recording = await self._manager.create_recording(Recording(**payload.model_dump(exclude_none=True)))
                return RecordingResponse(status=JsonValues.SUCCESS, recording=recording)
            except Exception as e:
                logger.error(e, exc_info=True)
                return RecordingResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/recordings/{uuid}",
            response_model=RecordingResponse,
            tags=[ApiTags.RECORDINGS],
            operation_id="get_recording",
        )
        async def get_recording(uuid: str) -> RecordingResponse:
            try:
                recording = await self._manager.get_recording(uuid)
                if recording is None:
                    return RecordingResponse(status=JsonValues.FAILURE, message="Recording not found")
                return RecordingResponse(status=JsonValues.SUCCESS, recording=recording)
            except Exception as e:
                logger.error(e, exc_info=True)
                return RecordingResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/recordings/{uuid}",
            response_model=StatusResponse,
            tags=[ApiTags.RECORDINGS],
            operation_id="delete_recording",
        )
        async def delete_recording(uuid: str) -> StatusResponse:
            try:
                ok = await self._manager.delete_recording(uuid)
                if not ok:
                    return StatusResponse(status=JsonValues.FAILURE, message="Recording not found")
                return StatusResponse(status=JsonValues.SUCCESS)
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/recordings/{uuid}/file",
            tags=[ApiTags.RECORDINGS],
            operation_id="get_recording_file",
        )
        async def get_recording_file(uuid: str, request: Request):
            """Serve a clip with HTTP range support so a <video> element can seek."""
            recording = await self._manager.get_recording(uuid)
            if recording is None or not recording.file_path:
                raise HTTPException(status_code=404, detail="Recording not found")
            path = Path(recording.file_path)
            if not path.is_file():
                raise HTTPException(status_code=404, detail="Recording file is missing on disk")
            # FileResponse handles Range requests itself, which is what makes seeking
            # work; note this must NOT be fetched through ReverseProxy, which buffers
            # whole responses.
            return FileResponse(path, media_type="video/mp4", filename=path.name)

        @self._app.get(
            "/api/recordings/{uuid}/overlay",
            tags=[ApiTags.RECORDINGS],
            operation_id="get_recording_overlay",
        )
        async def get_recording_overlay(uuid: str):
            """Serve a clip's detection-overlay sidecar.

            404 for "this clip has no overlay" as well as for "no such clip": the player
            asks for one on every clip it opens and treats either answer the same way —
            draw the video, offer no overlay toggle. Distinguishing them would give the
            caller a difference it has no use for.
            """
            recording = await self._manager.get_recording(uuid)
            if recording is None or not recording.overlay_path:
                raise HTTPException(status_code=404, detail="Recording has no overlay")
            path = Path(recording.overlay_path)
            if not path.is_file():
                raise HTTPException(status_code=404, detail="Overlay file is missing on disk")
            return FileResponse(path, media_type="application/json", filename=path.name)

    def _register_model_group_routes(self) -> None:
        @self._app.get(
            "/api/model-groups",
            response_model=ListModelGroupsResponse,
            tags=[ApiTags.MODEL_GROUPS],
            operation_id="list_model_groups",
        )
        async def list_model_groups() -> ListModelGroupsResponse:
            try:
                groups = await self._manager.list_model_groups()
                return ListModelGroupsResponse(status=JsonValues.SUCCESS, groups=groups)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ListModelGroupsResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.post(
            "/api/model-groups",
            response_model=ModelGroupResponse,
            tags=[ApiTags.MODEL_GROUPS],
            operation_id="create_model_group",
        )
        async def create_model_group(payload: ModelGroupCreate) -> ModelGroupResponse:
            try:
                group = await self._manager.create_model_group(payload)
                return ModelGroupResponse(status=JsonValues.SUCCESS, group=group)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return ModelGroupResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "model group"),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return ModelGroupResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.get(
            "/api/model-groups/{uuid}",
            response_model=ModelGroupResponse,
            tags=[ApiTags.MODEL_GROUPS],
            operation_id="get_model_group",
        )
        async def get_model_group(uuid: str) -> ModelGroupResponse:
            try:
                group = await self._manager.get_model_group(uuid)
                if group is None:
                    return ModelGroupResponse(status=JsonValues.FAILURE, message="Model group not found")
                return ModelGroupResponse(status=JsonValues.SUCCESS, group=group)
            except Exception as e:
                logger.error(e, exc_info=True)
                return ModelGroupResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.patch(
            "/api/model-groups/{uuid}",
            response_model=ModelGroupResponse,
            tags=[ApiTags.MODEL_GROUPS],
            operation_id="update_model_group",
        )
        async def update_model_group(uuid: str, payload: ModelGroupUpdate) -> ModelGroupResponse:
            try:
                group = await self._manager.update_model_group(uuid, payload)
                if group is None:
                    return ModelGroupResponse(status=JsonValues.FAILURE, message="Model group not found")
                return ModelGroupResponse(status=JsonValues.SUCCESS, group=group)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return ModelGroupResponse(
                    status=JsonValues.FAILURE,
                    message=self._unique_constraint_msg(e, "model group"),
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return ModelGroupResponse(status=JsonValues.FAILURE, message=str(e))

        @self._app.delete(
            "/api/model-groups/{uuid}",
            response_model=StatusResponse,
            tags=[ApiTags.MODEL_GROUPS],
            operation_id="delete_model_group",
        )
        async def delete_model_group(uuid: str) -> StatusResponse:
            try:
                ok = await self._manager.delete_model_group(uuid)
                if not ok:
                    return StatusResponse(status=JsonValues.FAILURE, message="Model group not found")
                return StatusResponse(status=JsonValues.SUCCESS)
            except IntegrityError as e:
                logger.error(e, exc_info=True)
                return StatusResponse(
                    status=JsonValues.FAILURE,
                    message="Cannot delete: this model group is referenced as a member of another group.",
                )
            except Exception as e:
                logger.error(e, exc_info=True)
                return StatusResponse(status=JsonValues.FAILURE, message=str(e))
