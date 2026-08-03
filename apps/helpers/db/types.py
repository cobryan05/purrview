from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, TypeAlias, Union
from uuid import uuid4

TaskParams: TypeAlias = Dict[str, Any]
TaskResumeData: TypeAlias = Dict[str, Any]
TaskResultData: TypeAlias = Dict[str, Any]


class ImageItemStatus(str, Enum):
    """Tri-state verification status shared by per-image labels and whole-image tags."""

    UNCERTAIN = "uncertain"
    PRESENT = "present"
    ABSENT = "absent"


# System tag UUIDs — keep in sync with static/js/shared/api/review.js
UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"
FALSE_POS_TAG_UUID = "00000000-0000-0000-0000-000000000003"
REJECTION_TAG_UUID = "00000000-0000-0000-0000-000000000004"
VERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000005"

# System dataset UUIDs
WHOLE_DATABASE_DATASET_UUID = "00000000-0000-0000-0000-000000000002"

import jstyleson
import numpy as np
import sqlalchemy as sa
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
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


class LabelType(str, Enum):
    STANDARD = "standard"
    DYNAMIC = "dynamic"


class LabelScope(str, Enum):
    OBJECT = "object"
    IMAGE = "image"


class TimestampModel(SQLModel):
    """
    An abstract mixin for tracking record lifecycle using timezone-aware UTC timestamps.
    """

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=DateTime(timezone=True),
    )

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
        label_type: Whether this is a standard dataset label or a dynamic user-created label.
        scope: Whether this label applies to whole images (image) or individual objects (object).
    """

    name: str
    color: Optional[str] = None
    parent_uuid: Optional[str] = Field(default=None, foreign_key="labels.uuid", ondelete="RESTRICT")
    label_type: LabelType = Field(default=LabelType.STANDARD)
    scope: LabelScope = Field(default=LabelScope.OBJECT)


class Label(LabelBase, TimestampModel, table=True):
    """
    The database representation of an annotation label.
    """

    __tablename__ = "labels"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    # Override to store enum values ("standard", "dynamic") not names ("STANDARD", "DYNAMIC").
    # sa.Enum(LabelType) defaults to names; values_callable forces value-based storage,
    # which matches the migration's server_default and frontend expectations.
    label_type: LabelType = Field(
        default=LabelType.STANDARD,
        sa_column=Column(
            sa.Enum(LabelType, values_callable=lambda x: [e.value for e in x]),
            nullable=False,
            server_default="standard",
        ),
    )
    scope: LabelScope = Field(
        default=LabelScope.OBJECT,
        sa_column=Column(
            sa.Enum(LabelScope, values_callable=lambda x: [e.value for e in x]),
            nullable=False,
            server_default="object",
        ),
    )
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
    label_type: LabelType = LabelType.STANDARD
    scope: LabelScope = LabelScope.OBJECT


class LabelUpdate(ApiModel):
    name: Optional[str] = None
    color: Optional[str] = None
    parent_uuid: Optional[str] = None
    label_type: Optional[LabelType] = None
    scope: Optional[LabelScope] = None


class ImageTagLink(SQLModel, table=True):
    """
    The database representation of the many-to-many relationship between whole images and tags.

    Tri-state, mirroring ImageLabel: a row only exists for PRESENT or ABSENT status;
    UNCERTAIN (never tagged, never verified absent) means no row. Unlike labels, PRESENT
    here is not derived from anything — it's just "the user applied this tag" — so all
    three statuses are directly user-settable.
    """

    __tablename__ = "image_tags"
    image_uuid: str = Field(foreign_key="images.uuid", primary_key=True, ondelete="CASCADE")
    tag_uuid: str = Field(foreign_key="tags.uuid", primary_key=True, ondelete="CASCADE", index=True)
    status: ImageItemStatus = Field(default=ImageItemStatus.PRESENT)
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=DateTime(timezone=True),
    )


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
    # Sort-order indexes for the image-search keyset pagination. When the
    # bbox tag/label filters are dispatched to correlated EXISTS (common
    # predicates), SQLite scans one of these indexes in ORDER BY direction and
    # stops at LIMIT — without them the EXISTS strategy degrades to a full
    # scan + temp b-tree sort and is never a win. uuid is included so the
    # composite fully covers the (sort_col, uuid) keyset ordering.
    __table_args__ = (
        sa.Index("ix_images_created_at_uuid", "created_at", "uuid"),
        sa.Index("ix_images_updated_at_uuid", "updated_at", "uuid"),
    )
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    bboxes: List["BBox"] = Relationship(
        back_populates="image", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    tags: List["Tag"] = Relationship(
        back_populates="images",
        link_model=ImageTagLink,
        sa_relationship_kwargs={
            "primaryjoin": "ImageRecord.uuid==ImageTagLink.image_uuid",
            "secondaryjoin": f"(ImageTagLink.tag_uuid==Tag.uuid) & (ImageTagLink.status=='{ImageItemStatus.PRESENT.value}')",
            "viewonly": True,
        },
    )


class ImageRecordRead(OrmModel):
    uuid: str
    filename: str
    bboxes: List["BBoxRead"]
    tags: List["TagRead"] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# Discriminated-union filter-term model (PR 1.4a; sole filter surface as of 1.4b).
#
# `ImageSearchParams.terms` is the query surface for all image-search
# filtering. All terms in the list are ANDed together; OR lives *inside* a
# term via multi-select + match_mode. See db_client._apply_filter_terms for
# the query-building side.
# ─────────────────────────────────────────────────────────────────────────────


class ImageLabelState(str, Enum):
    """Discriminant state for imageLabel / imageTag filter terms.

    present / absent   — the existing verified-status match.
    unreviewed         — existing require_unset semantics (NOT EXISTS any row).
    any_reviewed       — existing require_verified semantics (EXISTS a
                          PRESENT-or-ABSENT row).
    """

    PRESENT = "present"
    ABSENT = "absent"
    UNREVIEWED = "unreviewed"
    ANY_REVIEWED = "any_reviewed"


class ObjectLabelTerm(ApiModel):
    """Matches images having a BBOX with one of the given labels."""

    kind: Literal["objectLabel"] = "objectLabel"
    label_uuids: List[str]
    match_mode: Literal["any", "all"] = "any"
    # Replaces the old flat label_include_children for this term.
    roll_up: bool = True
    negate: bool = False


class ImageLabelTerm(ApiModel):
    """Matches on the image_labels table (whole-image label assessment)."""

    kind: Literal["imageLabel"] = "imageLabel"
    label_uuid: str
    state: ImageLabelState
    # If True (default), also matches on any DESCENDANT of label_uuid, not
    # just label_uuid itself — e.g. a "Cat present" filter also matches images
    # that only have "Tabby" (a child of Cat) recorded PRESENT.
    #
    # Applies to present / unreviewed / any_reviewed. ignored for absent:
    # roll-up direction is asymmetric for absence — if the parent is ABSENT
    # that implies all descendants are absent, but a descendant being ABSENT
    # says nothing about the parent (other descendants, or the parent itself,
    # may still be PRESENT). Expanding ABSENT over descendants with EXISTS
    # would match the wrong set, so absent terms always behave as if
    # roll_up=False server-side.
    roll_up: bool = True
    negate: bool = False


class BBoxTagTerm(ApiModel):
    """Matches images having a BBOX carrying one of the given tags."""

    kind: Literal["bboxTag"] = "bboxTag"
    tag_uuids: List[str]
    match_mode: Literal["any", "all"] = "any"
    negate: bool = False


class ImageTagTerm(ApiModel):
    """Matches on the image_tags table (whole-image tag assessment)."""

    kind: Literal["imageTag"] = "imageTag"
    tag_uuid: str
    state: ImageLabelState
    negate: bool = False


class BBoxVerificationTerm(ApiModel):
    """Matches images having a BBOX in the given verification state.

    Backed by the system verification tags (UNVERIFIED / FALSE_POS / VERIFIED),
    which are mutually exclusive per bbox via exclusive_group="verification".
    """

    kind: Literal["bboxVerification"] = "bboxVerification"
    state: Literal["verified", "unverified", "false_positive"]
    negate: bool = False


class LabelVerificationTerm(ApiModel):
    """The "Config Ready" predicate: has every target label been assessed?

    label_uuids must be supplied explicitly by the caller — as of PR 1.4b
    there is no flat ImageSearchParams field to fall back to. An empty list
    is vacuously true/false (see _apply_filter_terms).
    """

    kind: Literal["labelVerification"] = "labelVerification"
    check: Literal["all_verified", "has_unverified"]
    label_uuids: List[str] = Field(default_factory=list)
    negate: bool = False


class DateTerm(ApiModel):
    """Date range filter on images.created_at or images.updated_at.

    `from` is a Python keyword, so the field is named `from_` internally but
    aliased to "from" at the API boundary (overriding the alias_generator,
    which would otherwise leave a trailing-underscore "from_" key — see
    pydantic.alias_generators.to_camel). populate_by_name (inherited from
    ApiModel) still accepts the Python name `from_` too.
    """

    kind: Literal["date"] = "date"
    field: Literal["created", "updated"]
    from_: Optional[datetime] = Field(default=None, alias="from")
    to: Optional[datetime] = None
    negate: bool = False


class FilenameTerm(ApiModel):
    """AND of substring patterns against images.filename (like filename_searches)."""

    kind: Literal["filename"] = "filename"
    patterns: List[str] = Field(default_factory=list)
    negate: bool = False


FilterTerm = Annotated[
    Union[
        ObjectLabelTerm,
        ImageLabelTerm,
        BBoxTagTerm,
        ImageTagTerm,
        BBoxVerificationTerm,
        LabelVerificationTerm,
        DateTerm,
        FilenameTerm,
    ],
    Field(discriminator="kind"),
]


class ImageSearchParams(ApiModel):
    # Not a filter: display-mode flag consumed by callers outside the term
    # pipeline (e.g. bulk-apply's own bbox subset logic). Has no effect on
    # `terms` evaluation — a future slice may wire it into term evaluation.
    show_all_bboxes: bool = False
    limit: int = 50
    sort_by: Literal["uuid", "created_at", "updated_at"] = "created_at"
    sort_order: Literal["asc", "desc"] = "desc"
    # When True, only images that have at least one bbox are returned.
    # Useful for bbox-view pagination so empty-bbox images don't dilute pages.
    require_bboxes: bool = False
    # Keyset pagination cursor.
    after_uuid: Optional[str] = None
    after_sort_value: Optional[datetime] = None  # populated when sort_by != "uuid"
    # Discriminated-union filter-term model (PR 1.4a; sole filter surface as
    # of 1.4b — the old flat filter fields were deleted). All terms are
    # ANDed together; see apps.helpers.db.db_client._apply_filter_terms.
    terms: List[FilterTerm] = Field(default_factory=list)


class BBoxTagLink(SQLModel, table=True):
    """
    The database representation of the many-to-many relationship between bounding boxes and tags.
    """

    __tablename__ = "bbox_tags"
    bbox_uuid: str = Field(foreign_key="bboxes.uuid", primary_key=True, ondelete="CASCADE")
    tag_uuid: str = Field(foreign_key="tags.uuid", primary_key=True, ondelete="CASCADE", index=True)


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
    # Two composites, one per lookup direction, each replacing the single-column
    # index its leading column would otherwise need. Kept on the model (not
    # migration-only) so fresh create_all installs get them too, not just
    # migration-upgraded DBs.
    #
    # (label_uuid, image_uuid): serves the reverse label→image_uuid lookups used
    # by the dataset/label filters as a COVERING index seek and satisfies their
    # DISTINCT without a temp b-tree.
    #
    # (image_uuid, label_uuid): serves the correlated-EXISTS per-image probes
    # (image_uuid = images.uuid AND label_uuid IN (...)) with a single range
    # seek. Originally added by migration b1d4e8f2a093, silently dropped by the
    # phash batch_alter_table migration, restored by a6c3f8d1e2b7.
    __table_args__ = (
        sa.Index("ix_bboxes_label_uuid_image_uuid", "label_uuid", "image_uuid"),
        sa.Index("ix_bboxes_image_uuid_label_uuid", "image_uuid", "label_uuid"),
    )
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    image_uuid: str = Field(foreign_key="images.uuid", ondelete="CASCADE")
    label_uuid: str = Field(foreign_key="labels.uuid")

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
    images: List[ImageRecord] = Relationship(
        back_populates="tags",
        link_model=ImageTagLink,
        sa_relationship_kwargs={
            "primaryjoin": "Tag.uuid==ImageTagLink.tag_uuid",
            "secondaryjoin": f"(ImageTagLink.image_uuid==ImageRecord.uuid) & (ImageTagLink.status=='{ImageItemStatus.PRESENT.value}')",
            "viewonly": True,
        },
    )


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
    is_temporary: bool = Field(default=False)


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
    heartbeat_at: Optional[datetime] = Field(default=None, sa_type=DateTime(timezone=True))
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
    is_temporary: bool = False


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
    heartbeat_at: Optional[datetime] = None
    log_lines: Optional[List[str]] = Field(default=None, validation_alias="log_json")


class TaskResult(BaseModel):
    data: Dict[str, Any] = Field(default_factory=dict)


class ImageLabel(SQLModel, table=True):
    __tablename__ = "image_labels"
    # The composite PK leads with image_uuid, so reverse lookups by label_uuid
    # alone (list_images_for_label / dataset assembly) would full-scan without
    # this index.
    __table_args__ = (sa.Index("ix_image_labels_label_uuid", "label_uuid"),)
    image_uuid: str = Field(foreign_key="images.uuid", primary_key=True, ondelete="CASCADE")
    label_uuid: str = Field(foreign_key="labels.uuid", primary_key=True, ondelete="CASCADE")
    status: ImageItemStatus = Field(default=ImageItemStatus.UNCERTAIN)
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=DateTime(timezone=True),
    )


class ImageLabelRead(OrmModel):
    image_uuid: str
    label_uuid: str
    status: ImageItemStatus


class ImageLabelWithFilename(OrmModel):
    """ImageLabel row joined with the image filename — used by the dataset loader."""

    image_uuid: str
    label_uuid: str
    status: ImageItemStatus
    filename: str  # relative path (same as ImageRecord.filename)


class ImageTagStatusRead(OrmModel):
    image_uuid: str
    tag_uuid: str
    status: ImageItemStatus


# ─────────────────────────────────────────────────────────────────────────────
# Dataset filter criteria
# ─────────────────────────────────────────────────────────────────────────────


class DatasetFilterCriteria(ApiModel):
    """Stored as JSON in datasets.filter_json.

    As of PR 1.4b, self-contained criteria (tags, dates, filename, whole-image
    tag/label conditions) live in `terms` — the same discriminated-union shape
    as ImageSearchParams.terms (see db_client.build_dataset_search_params for
    how the two are combined). The remaining flat fields here are modifiers of
    Dataset.label_uuids, a structurally separate column (dataset_label_links,
    for FK integrity) that cannot itself live inside this JSON blob.
    """

    # Label modifiers (label_uuids lives on Dataset.label_uuids / dataset_label_links)
    label_exclude_uuids: List[str] = Field(default_factory=list)
    label_match_mode: Literal["any", "all"] = "any"
    label_include_children: bool = False
    # Verification status check (operates on dataset.label_uuids)
    # "all_verified"   = every dataset label has a verified image_label row (PRESENT or ABSENT)
    # "has_unverified" = at least one dataset label is uncertain or missing
    verification_check: Optional[Literal["all_verified", "has_unverified"]] = None
    # Discriminated-union filter terms — tags, dates, filename, whole-image
    # tag/label conditions. ANDed together, same as ImageSearchParams.terms.
    terms: List[FilterTerm] = Field(default_factory=list)

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        # Unknown fields must be rejected, not silently dropped — extra="ignore"
        # here previously meant a client sending a not-yet-recognised filter
        # field would have it "save" successfully then silently vanish from
        # every subsequent GET (data loss by design). See test_datasets.py
        # TestDatasetFilterCriteriaExtraFields.
        extra="forbid",
    )


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

MODEL_TYPE_ARCHITECTURES: Dict[str, List[str]] = {
    "object_detection": [
        "yolo_v9_t",
        "yolo_v9_s",
        "yolo_v9_m",
        "yolo_v9_c",
        "yolov8n",
        "yolov8s",
        "yolov8m",
        "yolov8l",
        "yolov8x",
        "yolov9t",
        "yolov9s",
        "yolov9m",
        "yolov9c",
        "yolov9e",
        "yolo11n",
        "yolo11s",
        "yolo11m",
        "yolo11l",
        "yolo11x",
        "yolo12n",
        "yolo12s",
        "yolo12m",
        "yolo12l",
        "yolo12x",
        "yolov10n",
        "yolov10s",
        "yolov10m",
        "yolov10b",
        "yolov10l",
        "yolov10x",
        "yolo26n",
        "yolo26s",
        "yolo26m",
        "yolo26l",
        "yolo26x",
    ],
    "crop_classifier": [
        "efficientnet_b0",
        "efficientnet_b2",
        "efficientnet_b4",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "resnet_18",
        "resnet_50",
    ],
    "image_classifier": [
        "efficientnet_b0",
        "efficientnet_b2",
        "efficientnet_b4",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "resnet_18",
        "resnet_50",
    ],
}


class ModelConfigNode(SQLModel, table=True):
    __tablename__ = "model_config_nodes"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    model_config_uuid: str = Field(foreign_key="model_configs.uuid", index=True, ondelete="CASCADE")
    label_uuid: Optional[str] = Field(default=None, foreign_key="labels.uuid", ondelete="CASCADE")
    model_type: str
    architecture: str
    roll_up_unchecked: bool = Field(default=False)
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
    model_config_uuid: str
    label_uuid: Optional[str] = None
    model_type: str
    architecture: str
    roll_up_unchecked: bool = False


class ModelConfigRead(OrmModel):
    uuid: str
    name: str
    description: Optional[str] = None
    nodes: List[ModelConfigNodeRead] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ModelConfigNodeWrite(ApiModel):
    label_uuid: Optional[str] = None  # None = root node
    model_type: str
    architecture: str
    roll_up_unchecked: bool = False


class ModelConfigCreate(ApiModel):
    name: str
    description: Optional[str] = None
    nodes: List[ModelConfigNodeWrite] = Field(default_factory=list)


class ModelConfigUpdate(ApiModel):
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: Optional[List[ModelConfigNodeWrite]] = None  # None=no change; []=remove all


# ─── Filter Chains ───────────────────────────────────────────────────────────


class ActionKind(str, Enum):
    DISCARD_IMAGE = "discard_image"
    TAG_IMAGE = "tag_image"
    STRIP_UNVERIFIED_BBOXES = "strip_unverified_bboxes"
    RELABEL_BBOX = "relabel_bbox"
    TAG_BBOX = "tag_bbox"
    ADD_BBOX = "add_bbox"


class FilterChainStep(SQLModel, table=True):
    __tablename__ = "filter_chain_steps"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    filter_chain_uuid: str = Field(foreign_key="filter_chains.uuid", index=True, ondelete="CASCADE")
    order: int
    model_name: str
    confidence: float = Field(default=0.5)
    iou_threshold: float = Field(default=0.5)  # detector-overlap steps only
    action: str  # validated against ActionKind in the write models
    action_params: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    enabled: bool = Field(default=True)
    chain: "FilterChain" = Relationship(back_populates="steps")
    # match_mode is derived at run time from the model sidecar's model_kind; not stored.
    # action_params holds: tag_uuid, target_label_uuid, only_if_label_uuid (all optional per action).


class FilterChain(TimestampModel, table=True):
    __tablename__ = "filter_chains"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    name: str = Field(unique=True, index=True)
    description: Optional[str] = None
    steps: List[FilterChainStep] = Relationship(
        back_populates="chain",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "order_by": "FilterChainStep.order"},
    )


class FilterChainStepRead(OrmModel):
    uuid: str
    filter_chain_uuid: str
    order: int
    model_name: str
    confidence: float = 0.5
    iou_threshold: float = 0.5
    action: str
    action_params: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class FilterChainRead(OrmModel):
    uuid: str
    name: str
    description: Optional[str] = None
    steps: List[FilterChainStepRead] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class FilterChainStepWrite(ApiModel):
    order: int
    model_name: str
    confidence: float = 0.5
    iou_threshold: float = 0.5
    action: ActionKind
    action_params: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    @model_validator(mode="after")
    def _validate_action_params(self) -> "FilterChainStepWrite":
        if self.action in (ActionKind.TAG_IMAGE, ActionKind.TAG_BBOX):
            if not self.action_params.get("tag_uuid"):
                raise ValueError(f"action {self.action.value!r} requires action_params.tag_uuid")
        elif self.action == ActionKind.RELABEL_BBOX:
            if not self.action_params.get("target_label_uuid"):
                raise ValueError("action 'relabel_bbox' requires action_params.target_label_uuid")
        return self


class FilterChainCreate(ApiModel):
    name: str
    description: Optional[str] = None
    steps: List[FilterChainStepWrite] = Field(default_factory=list)


class FilterChainUpdate(ApiModel):
    name: Optional[str] = None
    description: Optional[str] = None
    steps: Optional[List[FilterChainStepWrite]] = None  # None=no change; []=remove all


# ─── Model Groups ────────────────────────────────────────────────────────────


class ModelGroupMember(SQLModel, table=True):
    __tablename__ = "model_group_members"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    model_group_uuid: str = Field(foreign_key="model_groups.uuid", index=True, ondelete="CASCADE")
    order: int
    # Exactly one of these two is set — enforced by ModelGroupMemberWrite, not the DB row.
    model_name: Optional[str] = None
    # RESTRICT: deleting a group referenced as a nested member elsewhere must fail loudly
    # rather than silently orphaning the parent's member row.
    child_group_uuid: Optional[str] = Field(default=None, foreign_key="model_groups.uuid", ondelete="RESTRICT")
    group: "ModelGroup" = Relationship(
        back_populates="members",
        sa_relationship_kwargs={"foreign_keys": "ModelGroupMember.model_group_uuid"},
    )


class ModelGroup(TimestampModel, table=True):
    __tablename__ = "model_groups"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    name: str = Field(unique=True, index=True)
    description: Optional[str] = None
    # Cross-model detection merge policy for CollectTrainingImagesTask (Phase 5A,
    # LIVE_COLLECTION_PLAN.md): which model's detection wins a same-label collision.
    # Defaulted to "first_wins" so existing rows keep today's behaviour unchanged.
    merge_policy: str = Field(default="first_wins")
    members: List[ModelGroupMember] = Relationship(
        back_populates="group",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "order_by": "ModelGroupMember.order",
            "foreign_keys": "ModelGroupMember.model_group_uuid",
        },
    )


class ModelGroupMemberRead(OrmModel):
    uuid: str
    model_group_uuid: str
    order: int
    model_name: Optional[str] = None
    child_group_uuid: Optional[str] = None


class ModelGroupRead(OrmModel):
    uuid: str
    name: str
    description: Optional[str] = None
    merge_policy: Literal["first_wins", "last_wins"] = "first_wins"
    members: List[ModelGroupMemberRead] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ModelGroupMemberWrite(ApiModel):
    order: int
    model_name: Optional[str] = None
    child_group_uuid: Optional[str] = None

    @model_validator(mode="after")
    def _validate_exactly_one_member_kind(self) -> "ModelGroupMemberWrite":
        if bool(self.model_name) == bool(self.child_group_uuid):
            raise ValueError("Exactly one of model_name or child_group_uuid must be set")
        return self


class ModelGroupCreate(ApiModel):
    name: str
    description: Optional[str] = None
    merge_policy: Literal["first_wins", "last_wins"] = "first_wins"
    members: List[ModelGroupMemberWrite] = Field(default_factory=list)


class ModelGroupUpdate(ApiModel):
    name: Optional[str] = None
    description: Optional[str] = None
    merge_policy: Optional[Literal["first_wins", "last_wins"]] = None
    members: Optional[List[ModelGroupMemberWrite]] = None  # None=no change; []=remove all


class TrainingConfigDatasetLink(SQLModel, table=True):
    __tablename__ = "training_config_dataset_links"
    training_config_uuid: str = Field(foreign_key="training_configs.uuid", primary_key=True, ondelete="CASCADE")
    dataset_uuid: str = Field(foreign_key="datasets.uuid", primary_key=True, ondelete="RESTRICT")


class TrainingConfig(TimestampModel, table=True):
    __tablename__ = "training_configs"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    name: str = Field(unique=True, index=True)
    description: Optional[str] = None
    model_config_node_uuid: Optional[str] = Field(
        default=None, foreign_key="model_config_nodes.uuid", ondelete="SET NULL"
    )
    hyperparams_json: Dict[str, Any] = Field(default_factory=dict, sa_column=sa.Column(sa.JSON))
    dataset_links: List[TrainingConfigDatasetLink] = Relationship(
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class TrainingConfigRead(OrmModel):
    uuid: str
    name: str
    description: Optional[str]
    model_config_node_uuid: Optional[str]
    dataset_uuids: List[str]
    hyperparams_json: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class TrainingConfigCreate(ApiModel):
    name: str
    description: Optional[str] = None
    model_config_node_uuid: Optional[str] = None
    dataset_uuids: List[str] = Field(default_factory=list)
    hyperparams_json: Dict[str, Any] = Field(default_factory=dict)


class TrainingConfigUpdate(ApiModel):
    name: Optional[str] = None
    description: Optional[str] = None
    model_config_node_uuid: Optional[str] = None
    dataset_uuids: Optional[List[str]] = None
    hyperparams_json: Optional[Dict[str, Any]] = None  # None = no change; [] = remove all


# ─── Model Bundles ────────────────────────────────────────────────────────────


class ModelBundle(TimestampModel, table=True):
    __tablename__ = "model_bundles"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    name: str = Field(unique=True, index=True)
    description: Optional[str] = None
    model_config_uuid: str = Field(foreign_key="model_configs.uuid", ondelete="RESTRICT")
    bundle_path: str
    manifest_json: Dict[str, Any] = Field(default_factory=dict, sa_column=sa.Column(sa.JSON))
    training_config_uuids: List[str] = Field(default_factory=list, sa_column=sa.Column(sa.JSON))


class ModelBundleRead(OrmModel):
    uuid: str
    name: str
    description: Optional[str] = None
    model_config_uuid: str
    bundle_path: str
    manifest_json: Dict[str, Any] = Field(default_factory=dict)
    training_config_uuids: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ModelBundleCreate(ApiModel):
    name: str
    description: Optional[str] = None
    model_config_uuid: str
    bundle_path: str
    manifest_json: Dict[str, Any] = Field(default_factory=dict)
    training_config_uuids: List[str] = Field(default_factory=list)


class ModelBundleUpdate(ApiModel):
    name: Optional[str] = None
    description: Optional[str] = None
