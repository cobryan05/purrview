from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, TypeAlias, Union
from uuid import uuid4

TaskParams: TypeAlias = Dict[str, Any]
TaskResumeData: TypeAlias = Dict[str, Any]
TaskResultData: TypeAlias = Dict[str, Any]


class ImageLabelStatus(str, Enum):
    UNCERTAIN = "uncertain"
    PRESENT = "present"
    ABSENT = "absent"


# System tag UUIDs — keep in sync with static/js/shared/api/review.js
UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"
DUPLICATE_TAG_UUID = "00000000-0000-0000-0000-000000000002"

import jstyleson
import numpy as np
import sqlalchemy as sa
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
)
from pydantic.alias_generators import to_camel
from sqlmodel import JSON, Column, DateTime, Field, Relationship, SQLModel
from typing_extensions import Annotated


def validate_raw_json_dict(v: Any) -> Dict[str, str]:
    if isinstance(v, str):
        try:
            return jstyleson.loads(v)
        except (jstyleson.json.JSONDecodeError, TypeError):
            return {}
    return v if isinstance(v, dict) else {}


RawJsonDict = Annotated[Dict[str, Any], BeforeValidator(validate_raw_json_dict)]


def validate_base64_img(v: Any) -> str:
    if isinstance(v, np.ndarray):
        try:
            import base64

            import cv2

            _, buffer = cv2.imencode(".png", v)
            return base64.b64encode(buffer).decode("utf-8")
        except Exception:
            pass
    return v if isinstance(v, str) else None


Base64Image = Annotated[Optional[str], BeforeValidator(validate_base64_img)]


# ─────────────────────────────────────────────────────────────────────────────
# Shared API model bases — centralise alias_generator=to_camel so every new
# model automatically gets camelCase serialisation at the API boundary.
# ─────────────────────────────────────────────────────────────────────────────


class ApiModel(BaseModel):
    """Base for camelCase API models (create / update / filter payloads)."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class OrmModel(BaseModel):
    """Base for ORM-backed read models. Adds from_attributes=True to ApiModel config."""
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)


class TagKind(str, Enum):
    GENERIC = "generic"
    SYSTEM = "system"
    TASK_GENERATED = "task_generated"


class TimestampModel(SQLModel):
    """
    An abstract mixin for tracking record lifecycle using timezone-aware UTC timestamps.
    """

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={
            "onupdate": lambda: datetime.now(timezone.utc),
        },
    )


class UserSetting(SQLModel, table=True):
    """Simple key-value store for per-user (per-browser) persistent settings."""
    __tablename__ = "user_settings"
    key: str = Field(primary_key=True)
    value: str = Field(default="", sa_column=Column(sa.Text))


class LabelBase(SQLModel):
    """
    Base schema for object classification labels.

    Attributes:
        name: The display name of the label (e.g., 'Car', 'Pedestrian').
        color: A hex or CSS color string used for UI visualization.
        parent_uuid: Reference to a parent label uuid. Deletion is restricted if children exist.
    """

    name: str
    color: Optional[str] = None
    parent_uuid: Optional[str] = Field(default=None, foreign_key="labels.uuid", ondelete="RESTRICT")


class Label(LabelBase, TimestampModel, table=True):
    """
    The database representation of an annotation label.
    """

    __tablename__ = "labels"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    parent: Optional["Label"] = Relationship(
        back_populates="children", sa_relationship_kwargs={"remote_side": "Label.uuid"}
    )
    children: List["Label"] = Relationship(back_populates="parent")
    bboxes: List["BBox"] = Relationship(back_populates="label")


class LabelRead(OrmModel):
    """
    The public read-only representation of an annotation label.
    """

    uuid: str
    name: str
    color: Optional[str] = None
    parent_uuid: Optional[str] = None


class LabelUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    parent_uuid: Optional[str] = None

class ImageRecordBase(SQLModel):
    """
    Base schema for an image and its metadata
    """

    filename: str = Field(unique=True, index=True)
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    phash: Optional[str] = None


class ImageRecord(ImageRecordBase, TimestampModel, table=True):
    """
    The database representation of an image
    """

    __tablename__ = "images"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    bboxes: List["BBox"] = Relationship(
        back_populates="image", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class ImageRecordRead(OrmModel):
    uuid: str
    filename: str
    bboxes: List["BBoxRead"]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ImageLabelFilter(ApiModel):
    """A single condition on the image_labels table used when searching images."""

    label_uuid: str
    status: ImageLabelStatus
    exclude: bool = False
    require_verified: bool = False  # if True: EXISTS (status IN PRESENT, ABSENT); status/exclude ignored


class TagPresenceFilter(ApiModel):
    """Require or exclude images based on bbox-level tag presence."""

    tag_uuid: str
    exclude: bool = False  # False = require at least one bbox has this tag; True = require none do


class ImageSearchParams(ApiModel):
    show_all_bboxes: bool = False
    tag_uuids: List[str] = Field(default_factory=list)
    label_uuid: Optional[str] = None
    image_label_filters: List["ImageLabelFilter"] = Field(default_factory=list)
    limit: int = 50
    sort_by: Literal["uuid", "created_at", "updated_at", "filename"] = "created_at"
    sort_order: Literal["asc", "desc"] = "desc"
    filename_search: Optional[str] = None
    filename_searches: List[str] = Field(default_factory=list)  # AND of all patterns
    # Date range filters on images.created_at.
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    # Date range filters on images.updated_at.
    updated_date_from: Optional[datetime] = None
    updated_date_to: Optional[datetime] = None
    # Tag presence/absence filters (complement to tag_uuids positive-only filter).
    tag_presence_filters: List["TagPresenceFilter"] = Field(default_factory=list)
    # Dataset / review label filter: images must have at least one bbox with one of these
    # label uuids ("any") or all of them ("all").  Empty list = no label filter.
    label_uuids: List[str] = Field(default_factory=list)
    # "any" = image has a bbox with ANY of the selected labels (OR logic, default)
    # "all" = image has a bbox with EACH of the selected labels (AND logic)
    label_match_mode: Literal["any", "all"] = "any"
    # When True, label_uuids is expanded to include all descendant labels in the
    # hierarchy before filtering (e.g. "cat" also matches "tabby", "siamese", …).
    label_include_children: bool = False
    # "any" = image has a bbox with ANY of the selected tags (OR logic, default)
    # "all" = image has a bbox with EACH of the selected tags (AND logic, one per bbox ok)
    tag_match_mode: Literal["any", "all"] = "any"
    # Keyset pagination cursor.
    after_uuid: Optional[str] = None
    after_sort_value: Optional[datetime] = None  # populated when sort_by != "uuid"


class BBoxTagLink(SQLModel, table=True):
    """
    The database representation of the many-to-many relationship between bounding boxes and tags.
    """

    __tablename__ = "bbox_tags"
    bbox_uuid: str = Field(foreign_key="bboxes.uuid", primary_key=True, ondelete="CASCADE")
    tag_uuid: str = Field(foreign_key="tags.uuid", primary_key=True, ondelete="CASCADE")


class BBoxBase(SQLModel):
    """
    Base schema for a bounding box and its metadata.

    Coordinates are normalized floats in the range [0.0, 1.0], relative to the
    associated image dimensions:
        - (x, y) is the top-left corner
        - width and height are proportions of the full image size

    This means values are resolution-independent and must be multiplied by the
    image width/height to obtain pixel coordinates.
    """

    x: float
    y: float
    width: float
    height: float
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    phash: Optional[str] = None

    @field_validator("x", "y", "width", "height")
    @classmethod
    def _validate_normalized(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Bounding box values must be between 0.0 and 1.0")
        return v


class BBox(BBoxBase, TimestampModel, table=True):
    __tablename__ = "bboxes"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    image_uuid: str = Field(foreign_key="images.uuid", index=True, ondelete="CASCADE")
    label_uuid: str = Field(foreign_key="labels.uuid", index=True)

    image: ImageRecord = Relationship(back_populates="bboxes")
    label: Label = Relationship(back_populates="bboxes")
    tags: List["Tag"] = Relationship(back_populates="bboxes", link_model=BBoxTagLink)


class BBoxRead(OrmModel):
    """The public read-only representation of a bounding box."""

    uuid: str
    label_uuid: Optional[str] = None
    x: float
    y: float
    width: float
    height: float
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    label: Optional[LabelRead] = None
    tags: List["TagRead"] = []


class BBoxUpdate(ApiModel):
    """Input data for creating or replacing a bounding box."""

    uuid: str
    label_uuid: str
    x: float
    y: float
    width: float
    height: float
    tag_uuids: List[str] = []
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    # Sidecar rebuild fallback: names are stored alongside UUIDs so that a DB
    # rebuild can resolve references even when UUIDs differ after re-generation.
    label_name: Optional[str] = None
    tag_names: List[str] = []
    phash: Optional[str] = None


class ImageRecordUpdate(BaseModel):
    """Input data for replacing all bboxes on an image."""

    bboxes: List[BBoxUpdate] = []


class TagBase(SQLModel):
    name: str = Field(unique=True)
    color: Optional[str] = None
    icon: Optional[str] = None
    kind: TagKind = TagKind.GENERIC
    protected: bool = Field(default=False)
    exclusive_group: Optional[str] = None


class TagUpdate(SQLModel):
    name: Optional[str] = None
    color: Optional[str] = None
    icon: Optional[str] = None


class Tag(TagBase, TimestampModel, table=True):
    __tablename__ = "tags"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    bboxes: List[BBox] = Relationship(back_populates="tags", link_model=BBoxTagLink)


class TagRead(OrmModel):
    uuid: str
    name: str
    color: Optional[str] = None
    icon: Optional[str] = None
    kind: TagKind = TagKind.GENERIC


class SourceBase(SQLModel):
    name: str = Field(unique=True, index=True)
    typename: str
    params: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class Source(SourceBase, TimestampModel, table=True):
    __tablename__ = "sources"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)


class SourceRead(OrmModel):
    uuid: str
    name: str
    typename: str
    params: Dict[str, Any]


class SourceUpdate(BaseModel):
    name: Optional[str] = None
    typename: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class TaskConfigBase(SQLModel):
    name: str
    typename: str
    description: Optional[str] = None
    params_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class TaskConfig(TaskConfigBase, TimestampModel, table=True):
    __tablename__ = "task_configs"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    marked_for_delete: bool = Field(default=False)
    instances: List["TaskInstance"] = Relationship(back_populates="config")


class TaskInstanceBase(SQLModel):
    status: str
    resume_data_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    result_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    error_message: Optional[str] = None
    expires_at: Optional[datetime] = None
    started_at: Optional[datetime] = Field(default=None, sa_type=DateTime(timezone=True))
    ended_at: Optional[datetime] = Field(default=None, sa_type=DateTime(timezone=True))
    log_json: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))


class TaskInstance(TaskInstanceBase, TimestampModel, table=True):
    __tablename__ = "task_instances"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    config_uuid: str = Field(foreign_key="task_configs.uuid", ondelete="RESTRICT")
    config: TaskConfig = Relationship(back_populates="instances")


class TaskConfigRead(OrmModel):
    uuid: str
    name: str
    typename: str
    description: Optional[str] = None
    params: Dict[str, Any] = Field(validation_alias="params_json")
    marked_for_delete: bool = False


class TaskInstanceRead(OrmModel):
    uuid: str
    config_uuid: str
    status: str
    resume_data: Optional[Dict[str, Any]] = Field(default=None, validation_alias="resume_data_json")
    result: Optional[Dict[str, Any]] = Field(default=None, validation_alias="result_json")
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    log_lines: Optional[List[str]] = Field(default=None, validation_alias="log_json")


class TaskResult(BaseModel):
    data: Dict[str, Any] = Field(default_factory=dict)


class ImageLabel(SQLModel, table=True):
    __tablename__ = "image_labels"
    image_uuid: str = Field(foreign_key="images.uuid", primary_key=True, ondelete="CASCADE")
    label_uuid: str = Field(foreign_key="labels.uuid", primary_key=True, ondelete="CASCADE")
    status: ImageLabelStatus = Field(default=ImageLabelStatus.UNCERTAIN)
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=DateTime(timezone=True),
    )


class ImageLabelRead(OrmModel):
    image_uuid: str
    label_uuid: str
    status: ImageLabelStatus


# ─────────────────────────────────────────────────────────────────────────────
# Dataset filter condition types
# ─────────────────────────────────────────────────────────────────────────────


class ImageAgeCondition(ApiModel):
    """Filter by image created_at or updated_at."""

    type: Literal["image_age"] = "image_age"
    field: Literal["created_at", "updated_at"] = "updated_at"
    within_days: Optional[int] = None  # relative: last N days (takes priority over after/before)
    after: Optional[datetime] = None  # absolute lower bound
    before: Optional[datetime] = None  # absolute upper bound


class BboxAgeCondition(ApiModel):
    """Filter images by whether any of their bboxes were recently created/updated.
    Requires bbox.created_at / bbox.updated_at columns (added in add_timestamps_to_bboxes migration)."""

    type: Literal["bbox_age"] = "bbox_age"
    field: Literal["created_at", "updated_at"] = "updated_at"
    within_days: Optional[int] = None
    after: Optional[datetime] = None
    before: Optional[datetime] = None


class TagCondition(ApiModel):
    """Require or exclude images based on any bbox carrying a specific tag."""

    type: Literal["tag"] = "tag"
    tag_uuid: str
    exclude: bool = False  # False = require at least one bbox has this tag; True = require none do


class DatasetLabelStatusCondition(ApiModel):
    """Filter on the image_labels status for this dataset's own relevant labels."""

    type: Literal["dataset_label_status"] = "dataset_label_status"
    check: Literal["all_verified", "exclude_uncertain"]
    # "all_verified"      → for each dataset label: EXISTS row with status PRESENT or ABSENT
    # "exclude_uncertain" → same semantics (kept separate for UI clarity)


class FilenameCondition(ApiModel):
    """Substring match on image filename."""

    type: Literal["filename"] = "filename"
    pattern: str


FilterCondition = Annotated[
    Union[
        ImageAgeCondition,
        BboxAgeCondition,
        TagCondition,
        DatasetLabelStatusCondition,
        FilenameCondition,
    ],
    Field(discriminator="type"),
]


class DatasetFilterCriteria(ApiModel):
    """Stored as JSON in datasets.filter_json. AND-combined list of conditions.
    Conditions of type DatasetLabelStatusCondition operate over the dataset's own label_uuids."""

    conditions: List[FilterCondition] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset ORM models
# ─────────────────────────────────────────────────────────────────────────────


class DatasetBase(SQLModel):
    name: str = Field(unique=True, index=True)
    description: Optional[str] = None
    filter_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class Dataset(DatasetBase, TimestampModel, table=True):
    __tablename__ = "datasets"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    label_links: List["DatasetLabelLink"] = Relationship(
        back_populates="dataset",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class DatasetLabelLink(SQLModel, table=True):
    __tablename__ = "dataset_label_links"
    dataset_uuid: str = Field(foreign_key="datasets.uuid", primary_key=True, ondelete="CASCADE")
    label_uuid: str = Field(foreign_key="labels.uuid", primary_key=True, ondelete="CASCADE")
    dataset: Optional[Dataset] = Relationship(back_populates="label_links")


class DatasetRead(OrmModel):
    uuid: str
    name: str
    description: Optional[str] = None
    filter_json: DatasetFilterCriteria = Field(default_factory=DatasetFilterCriteria)
    label_uuids: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DatasetCreate(ApiModel):
    name: str
    description: Optional[str] = None
    filter_json: DatasetFilterCriteria = Field(default_factory=DatasetFilterCriteria)
    label_uuids: List[str] = Field(default_factory=list)


class DatasetUpdate(ApiModel):
    name: Optional[str] = None
    description: Optional[str] = None
    filter_json: Optional[DatasetFilterCriteria] = None
    label_uuids: Optional[List[str]] = None  # None = no change; [] = remove all


# ─── Model Configurations ────────────────────────────────────────────────────

AVAILABLE_MODEL_TYPENAMES = ["YoloV8", "TestModel"]


class ModelConfigNode(SQLModel, table=True):
    __tablename__ = "model_config_nodes"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    model_config_uuid: str = Field(foreign_key="model_configs.uuid", index=True, ondelete="CASCADE")
    label_uuid: Optional[str] = Field(default=None, foreign_key="labels.uuid", ondelete="CASCADE")
    model_typename: str
    config: "ModelConfig" = Relationship(back_populates="nodes")


class ModelConfig(TimestampModel, table=True):
    __tablename__ = "model_configs"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    name: str = Field(unique=True, index=True)
    description: Optional[str] = None
    nodes: List[ModelConfigNode] = Relationship(
        back_populates="config",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class ModelConfigNodeRead(OrmModel):
    uuid: str
    label_uuid: Optional[str] = None
    model_typename: str


class ModelConfigRead(OrmModel):
    uuid: str
    name: str
    description: Optional[str] = None
    nodes: List[ModelConfigNodeRead] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ModelConfigNodeWrite(ApiModel):
    label_uuid: Optional[str] = None  # None = root node
    model_typename: str


class ModelConfigCreate(ApiModel):
    name: str
    description: Optional[str] = None
    nodes: List[ModelConfigNodeWrite] = Field(default_factory=list)


class ModelConfigUpdate(ApiModel):
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: Optional[List[ModelConfigNodeWrite]] = None  # None=no change; []=remove all
