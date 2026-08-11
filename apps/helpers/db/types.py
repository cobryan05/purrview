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
REJECTION_TAG_UUID = "00000000-0000-0000-0000-000000000004"
VERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000005"

# System tags that used to exist and must never come back. A tags.json dumped
# before the removing migration ran still lists them, and sync_tags_from_json()
# runs on exactly the rebuild/restore path those sidecars exist for — without
# this denylist the tag is re-inserted by the "no match → insert new" branch.
# 000...003 is the old False Positive verification tag (migration c1e5a83f7d92).
RETIRED_TAG_UUIDS = frozenset({"00000000-0000-0000-0000-000000000003"})

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

    Backed by the system verification tags (UNVERIFIED / VERIFIED), which are
    mutually exclusive per bbox via exclusive_group="verification".
    """

    kind: Literal["bboxVerification"] = "bboxVerification"
    state: Literal["verified", "unverified"]
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
    # None = "preserve the stored metadata for this bbox"; {} = "clear it".
    # See BoundingBoxMetadataModel.extra and Manager.update_image_metadata.
    metadata_json: Optional[Dict[str, Any]] = Field(default=None)
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
    # Last snapshot of the task's get_status_data() — the histograms, stat grids and
    # source tables its status_schema() widgets render from. Persisted for the same
    # reason as log_json: the live values live in the task object, which is gone once
    # the run ends, so a finished card could previously only show empty widgets.
    # Measured at ~13 KB for the heaviest task type; most are far smaller.
    status_data_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))


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
    status_data: Optional[Dict[str, Any]] = Field(default=None, validation_alias="status_data_json")


class TaskResult(BaseModel):
    data: Dict[str, Any] = Field(default_factory=dict)


# ── Preview runs ────────────────────────────────────────────────────────────
# Persisted "Run Model on Results" previews. A detector run over a review scope
# used to live only in a JS array inside the confirmation modal: closing the
# modal threw away every candidate, and there was no per-candidate state, so a
# 3,000-candidate run had to be triaged in one sitting or not at all.
#
# ⚠️ These are **scratch** tables, not annotation tables. Nothing here is a
# fact about an image until the user applies it; the preview tasks still never
# write to bboxes/labels/tags (see PreviewModelRelabelTask's docstring).


class PreviewRunTTL:
    """How long a run survives before the lazy sweep on GET /api/preview-runs.

    Fully-triaged runs go early: their candidates are all `applied`/`rejected`,
    so keeping them costs rows for a list nobody will open again.
    """

    DEFAULT_DAYS = 7
    TRIAGED_HOURS = 24


class PreviewDecision(str, Enum):
    """Per-candidate triage state — this is what makes a run resumable."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    APPLIED = "applied"
    STALE = "stale"


class PreviewDecisionAction(str, Enum):
    """What accepting a candidate does. Per candidate, not per run — the old
    global Relabel/Delete radio could only be one or the other for a whole run."""

    RELABEL = "relabel"
    DELETE = "delete"


class PreviewConflictPolicy(str, Enum):
    """Which model wins when several models propose on the same bbox.

    `LAST` reproduces the behavior that used to be hard-coded (and invisible) in
    runModelOnResults.js's merge loop.
    """

    LAST = "last"
    FIRST = "first"


class PreviewRunBase(SQLModel):
    name: str
    model_name: str
    task_typename: str
    params_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    scope_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    stats_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    pinned: bool = Field(default=False)
    expires_at: Optional[datetime] = Field(default=None, sa_type=DateTime(timezone=True))


class PreviewRun(PreviewRunBase, TimestampModel, table=True):
    __tablename__ = "preview_runs"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    candidates: List["PreviewCandidate"] = Relationship(
        back_populates="run", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class PreviewCandidateBase(SQLModel):
    image_uuid: str
    # None for a proposed *new* box: there is no bbox row to point at until apply.
    bbox_uuid: Optional[str] = None
    is_new: bool = Field(default=False)
    # Snapshot of the bbox's label at run time. Compared against the live row on
    # read to spot "changed since the run" — see the staleness handling in manager.
    current_label_uuid: Optional[str] = None
    proposed_label_uuid: str
    confidence: float = Field(default=0.0)
    geometry_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    decision: PreviewDecision = Field(default=PreviewDecision.PENDING)
    decision_action: Optional[PreviewDecisionAction] = Field(default=None)
    source_model: str = Field(default="")
    applied_at: Optional[datetime] = Field(default=None, sa_type=DateTime(timezone=True))


class PreviewCandidate(PreviewCandidateBase, TimestampModel, table=True):
    __tablename__ = "preview_candidates"
    # The two default triage views are "unreviewed first" and
    # "highest confidence first", both scoped to one run.
    __table_args__ = (
        sa.Index("ix_preview_candidates_run_decision", "run_uuid", "decision"),
        sa.Index("ix_preview_candidates_run_confidence", "run_uuid", "confidence"),
        # Multi-model conflict resolution upserts on this pair.
        sa.Index("ix_preview_candidates_run_bbox", "run_uuid", "bbox_uuid"),
    )
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    run_uuid: str = Field(foreign_key="preview_runs.uuid", ondelete="CASCADE", index=True)
    run: PreviewRun = Relationship(back_populates="candidates")


class PreviewRunRead(OrmModel):
    uuid: str
    name: str
    model_name: str
    task_typename: str
    params: Dict[str, Any] = Field(default_factory=dict, validation_alias="params_json")
    scope: Dict[str, Any] = Field(default_factory=dict, validation_alias="scope_json")
    stats: Dict[str, Any] = Field(default_factory=dict, validation_alias="stats_json")
    pinned: bool = False
    expires_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Filled in by the manager, not the ORM: counts per PreviewDecision value.
    decision_counts: Dict[str, int] = Field(default_factory=dict)
    candidate_count: int = 0


class PreviewCandidateRead(OrmModel):
    uuid: str
    run_uuid: str
    image_uuid: str
    # Joined in at read time, FILES_ROOT-relative. The candidate row stores only
    # the image uuid, but every thumbnail URL the panel builds is keyed on the
    # path — without this the client would need a second round-trip per page.
    image_filename: str = ""
    bbox_uuid: Optional[str] = None
    is_new: bool = False
    current_label_uuid: Optional[str] = None
    proposed_label_uuid: str
    confidence: float = 0.0
    geometry: Optional[Dict[str, Any]] = Field(default=None, validation_alias="geometry_json")
    decision: PreviewDecision = PreviewDecision.PENDING
    decision_action: Optional[PreviewDecisionAction] = None
    source_model: str = ""
    applied_at: Optional[datetime] = None
    # Derived at read time by joining to the live rows — never stored, because
    # it is a statement about *now*, not about when the run happened.
    label_changed_since_run: bool = False


class PreviewRunCreate(ApiModel):
    name: Optional[str] = None
    model_name: str = ""
    task_typename: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)
    scope: Dict[str, Any] = Field(default_factory=dict)


class PreviewRunUpdate(ApiModel):
    name: Optional[str] = None
    pinned: Optional[bool] = None
    stats: Optional[Dict[str, Any]] = None


class PreviewCandidateCreate(ApiModel):
    image_uuid: str
    bbox_uuid: Optional[str] = None
    is_new: bool = False
    current_label_uuid: Optional[str] = None
    proposed_label_uuid: str
    confidence: float = 0.0
    geometry: Optional[Dict[str, Any]] = None
    source_model: str = ""


class PreviewCandidateBulkUpdate(ApiModel):
    """Bulk PATCH over a list of candidates. Every field is optional — the
    triage panel sends only what changed, debounced, on each decision edit."""

    candidate_uuids: List[str] = Field(default_factory=list)
    decision: Optional[PreviewDecision] = None
    decision_action: Optional[PreviewDecisionAction] = None
    proposed_label_uuid: Optional[str] = None
    geometry: Optional[Dict[str, Any]] = None


class PreviewCandidateQuery(ApiModel):
    """Filter/paginate params for the candidates endpoint.

    ⚠️ Offset-paginated, unlike ImageSearchParams' keyset cursor. A preview run
    is a bounded, closed set (already fully written before triage begins,
    indexed by run_uuid), and the default `image_confidence` ordering is a
    four-part composite with no single scalar cursor value — a keyset over it
    would mean threading the whole tuple through the API for no win at this size.
    """

    decisions: List[PreviewDecision] = Field(default_factory=list)
    proposed_label_uuids: List[str] = Field(default_factory=list)
    confidence_min: Optional[float] = None
    confidence_max: Optional[float] = None
    is_new: Optional[bool] = None
    source_models: List[str] = Field(default_factory=list)
    # image_confidence groups every candidate from one image together, ranked by
    # that image's best candidate — the modal did this client-side, which does
    # not survive pagination.
    sort_by: Literal["confidence", "image_confidence", "created_at"] = "image_confidence"
    sort_order: Literal["asc", "desc"] = "desc"
    limit: int = 100
    offset: int = 0


class PreviewApplyResult(BaseModel):
    modified: int = 0
    added: int = 0
    deleted: int = 0
    failed: int = 0
    stale: int = 0


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
    # Views and Datasets share this table (decision register #10: "Dataset =
    # promoted View, one record + flag"). A View is a saved search; promoting it
    # makes a Dataset that training runs consume.
    #
    # ⚠️ Rows default to promoted=False, i.e. a View. list_datasets() therefore
    # filters to promoted rows *by default* — five callers of fetchDatasets()
    # (the dataset manager, training configs, model configs, dataset checklist)
    # all mean "a real, trainable Dataset", and would silently start offering
    # Views if the default leaked. Pass include_views=True to opt in.
    promoted: bool = Field(default=False, index=True)


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
    promoted: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DatasetCreate(ApiModel):
    name: str
    description: Optional[str] = None
    filter_json: DatasetFilterCriteria = Field(default_factory=DatasetFilterCriteria)
    label_uuids: List[str] = Field(default_factory=list)
    # Deliberately True here while the *table* default is False. A bare row is a
    # View, but this endpoint is /api/datasets and its existing callers (the
    # Datasets manager, the review tab's "Open in Datasets Tab" export) all mean
    # to create a real Dataset. The Views UI is the one caller that passes False.
    promoted: bool = True


class DatasetUpdate(ApiModel):
    name: Optional[str] = None
    description: Optional[str] = None
    filter_json: Optional[DatasetFilterCriteria] = None
    label_uuids: Optional[List[str]] = None  # None = no change; [] = remove all
    promoted: Optional[bool] = None  # None = no change; promotion is a flag flip


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


# ─── Watch Profiles (StreamWatcher / NVR) ────────────────────────────────────
#
# A Watch Profile is the curated "what to watch, and what counts as an event"
# entity; a StreamWatcher task is merely an execution of one. That split mirrors
# Datasets, Filter Chains, Model Stacks and Training Configs, and it is what makes
# "which watchers use camera 3?" and "show me every event for this rule" answerable
# in SQL rather than by grepping a JSON blob.
#
# `watch_events` and `recordings` are written from the start even though the media
# browser is a separate plan: they are its foundation, and adding them now costs
# almost nothing versus a migration later.


class RuleSubjectKind(str, Enum):
    """What one side of an interaction rule refers to."""

    LABEL = "label"
    ZONE = "zone"
    ANY = "any"
    NONE = "none"  # arity-1 rules ("a cat is tracked at all")


class RuleRelation(str, Enum):
    """How the two subjects of a rule are compared.

    `CONTAINS` is the default for "cat in litter box" and similar: a cat's box is far
    smaller than the litter box's, so IoU stays low even at perfect containment.

    `OVERLAPS` covers the other half of that problem — two boxes that merely lap over
    each other's edges, like a cat drinking from a water bowl. There the cat is neither
    inside the bowl nor the same size as it, so `CONTAINS` and `IOU` both stay near
    zero while the interaction is plainly happening.
    """

    PRESENT = "present"  # arity 1: the subject is tracked at all
    IOU = "iou"  # intersection over union — right for same-size objects
    CONTAINS = "contains"  # intersection / area(a) — "how much of A is inside B"
    OVERLAPS = "overlaps"  # intersection / min(area) — partial edge overlap, symmetric
    CENTROID_IN = "centroid_in"  # A's centre is inside B — cheapest and very robust
    DISTANCE = "distance"  # normalised centre distance — "cat near door"


class ZoneKind(str, Enum):
    BOX = "box"
    POLYGON = "polygon"


class WatchActionTrigger(str, Enum):
    OBJECT_DETECTED = "object_detected"
    TRACK_STARTED = "track_started"
    EVENT_STARTED = "event_started"
    EVENT_ENDED = "event_ended"


class BufferMode(str, Enum):
    AUTO = "auto"  # use whatever ingest tier the source resolves to
    OFF = "off"  # never arm a rolling buffer for this source


class RecordingStatus(str, Enum):
    PENDING = "pending"
    WRITING = "writing"
    COMPLETE = "complete"
    FAILED = "failed"
    # The window aged out of the RAM buffer before extraction ran. Terminal: retrying
    # can never succeed, so it must not sit in the work queue forever.
    EXPIRED = "expired"


class WatchZone(SQLModel, table=True):
    """A user-drawn region on one source's frame.

    Zones are the highest value-per-effort part of the whole feature: a box named
    "litter box area" is more reliable than a litter-box detector, needs no training
    data, and makes `cat ∩ zone` work today.
    """

    __tablename__ = "watch_zones"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    profile_source_uuid: str = Field(foreign_key="watch_profile_sources.uuid", index=True, ondelete="CASCADE")
    name: str
    kind: str = Field(default=ZoneKind.BOX.value)  # validated against ZoneKind in write models
    points_json: List[Any] = Field(default_factory=list, sa_column=Column(JSON))
    # Coordinates are stored normalised (0..1), but the resolution they were drawn
    # against is kept so a camera resolution change is detectable rather than silently
    # shifting every zone.
    reference_width: int = Field(default=0)
    reference_height: int = Field(default=0)
    profile_source: "WatchProfileSource" = Relationship(back_populates="zones")


class WatchProfileSource(SQLModel, table=True):
    """The junction that *is* the source↔watcher link."""

    __tablename__ = "watch_profile_sources"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    profile_uuid: str = Field(foreign_key="watch_profiles.uuid", index=True, ondelete="CASCADE")
    # RESTRICT, not CASCADE: deleting a source that a watcher uses must fail loudly
    # rather than quietly emptying the watcher. CASCADE on both sides of a junction
    # turns a hard constraint into a silent unlink.
    source_uuid: str = Field(foreign_key="sources.uuid", index=True, ondelete="RESTRICT")
    order: int = Field(default=0)
    enabled: bool = Field(default=True)
    # NULL means "use the profile default" for each of these.
    model_group_uuid: Optional[str] = Field(default=None, foreign_key="model_groups.uuid", ondelete="SET NULL")
    analysis_fps: Optional[float] = Field(default=None)
    analysis_width: Optional[int] = Field(default=None)
    record_retention_s: Optional[float] = Field(default=None)
    tracker_params_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    buffer_mode: str = Field(default=BufferMode.AUTO.value)
    # Tier C (snapshot/MQTT sources) only: the in-RAM JPEG frame ring's bounds.
    ring_seconds: float = Field(default=20.0)
    ring_max_mb: float = Field(default=32.0)
    profile: "WatchProfile" = Relationship(back_populates="sources")
    zones: List[WatchZone] = Relationship(
        back_populates="profile_source",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class InteractionRule(SQLModel, table=True):
    """A stateful predicate over tracks that opens and closes events.

    Every rule carries hysteresis because a raw threshold *will* flap: detection
    confidence and box jitter cross it many times per second, which unguarded means an
    MQTT storm and a dozen three-second clips for one litter-box visit.
    """

    __tablename__ = "interaction_rules"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    profile_uuid: str = Field(foreign_key="watch_profiles.uuid", index=True, ondelete="CASCADE")
    order: int = Field(default=0)
    name: str
    enabled: bool = Field(default=True)
    subject_a_kind: str = Field(default=RuleSubjectKind.LABEL.value)
    subject_a_ref: Optional[str] = Field(default=None)  # label uuid or zone uuid
    relation: str = Field(default=RuleRelation.CONTAINS.value)
    subject_b_kind: str = Field(default=RuleSubjectKind.NONE.value)
    subject_b_ref: Optional[str] = Field(default=None)
    threshold: float = Field(default=0.5)
    # The predicate must hold continuously this long before the event STARTS.
    min_duration_s: float = Field(default=2.0)
    # ...and be continuously false this long before it ENDS, which is what stops one
    # visit becoming twelve events.
    rearm_delay_s: float = Field(default=10.0)
    # Hard cap so a stuck detection cannot hold an event (and a recording) open forever.
    max_duration_s: float = Field(default=300.0)
    event_label: Optional[str] = Field(default=None)  # human name; also the MQTT topic leaf
    profile: "WatchProfile" = Relationship(back_populates="rules")


class LabelLifetime(SQLModel, table=True):
    """How long a label's track survives without a detection.

    In **seconds, not frames**: a frame count silently means 1 s on a 5 fps camera and
    0.2 s on a 25 fps one, and drifts whenever a source stutters — which live streams
    do constantly.
    """

    __tablename__ = "label_lifetimes"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    profile_uuid: str = Field(foreign_key="watch_profiles.uuid", index=True, ondelete="CASCADE")
    label_uuid: str = Field(foreign_key="labels.uuid", index=True, ondelete="CASCADE")
    ttl_s: float = Field(default=2.0)
    # Anchors latch their box once seen with high confidence and hold it for a long
    # TTL, rather than trusting a detector that intermittently misses a stationary
    # object like a litter box. Cheaper and far more robust than re-detecting it.
    is_anchor: bool = Field(default=False)
    profile: "WatchProfile" = Relationship(back_populates="label_lifetimes")


class WatchAction(SQLModel, table=True):
    """A side effect to run at some point in an event's life."""

    __tablename__ = "watch_actions"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    profile_uuid: str = Field(foreign_key="watch_profiles.uuid", index=True, ondelete="CASCADE")
    # NULL means "every rule in this profile".
    rule_uuid: Optional[str] = Field(default=None, foreign_key="interaction_rules.uuid", ondelete="CASCADE")
    order: int = Field(default=0)
    enabled: bool = Field(default=True)
    kind: str  # registry key, e.g. "publish_mqtt" / "record_clip" / "save_snapshot"
    params_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    trigger: str = Field(default=WatchActionTrigger.EVENT_ENDED.value)
    profile: "WatchProfile" = Relationship(back_populates="actions")


class WatchProfile(TimestampModel, table=True):
    __tablename__ = "watch_profiles"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    name: str = Field(unique=True, index=True)
    description: Optional[str] = None
    enabled: bool = Field(default=True)
    tracker_typename: str = Field(default="IouTracker")
    tracker_params_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    default_model_group_uuid: Optional[str] = Field(default=None, foreign_key="model_groups.uuid", ondelete="SET NULL")
    # Analysis rate and width are first-class because per-frame inference over
    # base64 HTTP is the throughput ceiling: 2–5 fps at 640 px tracks cats fine and
    # cuts payloads ~10x. Recording is unaffected — MediaMTX records the full-rate,
    # full-resolution stream regardless of what we analyse.
    default_analysis_fps: float = Field(default=5.0)
    default_analysis_width: int = Field(default=640)
    sources: List[WatchProfileSource] = Relationship(
        back_populates="profile",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "order_by": "WatchProfileSource.order"},
    )
    rules: List[InteractionRule] = Relationship(
        back_populates="profile",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "order_by": "InteractionRule.order"},
    )
    actions: List[WatchAction] = Relationship(
        back_populates="profile",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "order_by": "WatchAction.order"},
    )
    label_lifetimes: List[LabelLifetime] = Relationship(
        back_populates="profile",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class Recording(TimestampModel, table=True):
    """One extracted clip on disk.

    Independent of the rolling buffer: the buffer is RAM and evaporates, these are the
    only durable media. There is deliberately no 24/7 timeline.
    """

    __tablename__ = "recordings"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    source_uuid: Optional[str] = Field(default=None, foreign_key="sources.uuid", index=True, ondelete="SET NULL")
    profile_uuid: Optional[str] = Field(default=None, foreign_key="watch_profiles.uuid", ondelete="SET NULL")
    event_uuid: Optional[str] = Field(default=None, index=True)
    file_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    started_at: Optional[datetime] = Field(default=None, sa_type=DateTime(timezone=True))
    ended_at: Optional[datetime] = Field(default=None, sa_type=DateTime(timezone=True))
    duration_s: float = Field(default=0.0)
    size_bytes: int = Field(default=0)
    status: str = Field(default=RecordingStatus.PENDING.value)
    error: Optional[str] = None
    # True when the requested pre-roll had already aged out of the RAM buffer. A
    # truncated clip must never be presented as though its pre-roll were intact.
    partial_preroll: bool = Field(default=False)


class WatchEvent(SQLModel, table=True):
    """One opening and closing of a rule.

    Rows with `ended_at` set and no `recording_uuid` are a work queue — but a
    short-deadline one, because the source window ages out of the RAM buffer in under
    a minute.
    """

    __tablename__ = "watch_events"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    profile_uuid: str = Field(foreign_key="watch_profiles.uuid", index=True, ondelete="CASCADE")
    source_uuid: Optional[str] = Field(default=None, foreign_key="sources.uuid", index=True, ondelete="SET NULL")
    rule_uuid: Optional[str] = Field(default=None, foreign_key="interaction_rules.uuid", ondelete="SET NULL")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))
    ended_at: Optional[datetime] = Field(default=None, sa_type=DateTime(timezone=True))
    peak_confidence: float = Field(default=0.0)
    track_ids_json: List[Any] = Field(default_factory=list, sa_column=Column(JSON))
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    recording_uuid: Optional[str] = Field(default=None, foreign_key="recordings.uuid", ondelete="SET NULL")


# ─── Watch Profile read/write models ─────────────────────────────────────────
# Nested writes with whole-list replace, following the Filter Chains precedent.
# Enum-typed fields appear on the *write* models only: `sa.Enum(StrEnum)` stores
# names rather than values, so the columns stay plain `str` and validation lives here.


class WatchZoneRead(OrmModel):
    uuid: str
    profile_source_uuid: str
    name: str
    kind: str
    points_json: List[Any] = Field(default_factory=list)
    reference_width: int = 0
    reference_height: int = 0


class WatchZoneWrite(ApiModel):
    uuid: Optional[str] = None  # preserved on update so rules keep referring to it
    name: str
    kind: ZoneKind = ZoneKind.BOX
    points_json: List[Any] = Field(default_factory=list)
    reference_width: int = 0
    reference_height: int = 0

    @model_validator(mode="after")
    def _validate_points(self) -> "WatchZoneWrite":
        if self.kind == ZoneKind.BOX and len(self.points_json) != 2:
            raise ValueError("a box zone needs exactly 2 points ([x1,y1],[x2,y2])")
        if self.kind == ZoneKind.POLYGON and len(self.points_json) < 3:
            raise ValueError("a polygon zone needs at least 3 points")
        for point in self.points_json:
            if not (isinstance(point, (list, tuple)) and len(point) == 2):
                raise ValueError(f"zone points must be [x, y] pairs, got {point!r}")
            # Normalised coordinates survive a camera resolution change; pixel values
            # would silently point somewhere else after one.
            if not all(0.0 <= float(c) <= 1.0 for c in point):
                raise ValueError(f"zone points must be normalised to 0..1, got {point!r}")
        return self


class WatchProfileSourceRead(OrmModel):
    uuid: str
    profile_uuid: str
    source_uuid: str
    order: int = 0
    enabled: bool = True
    model_group_uuid: Optional[str] = None
    analysis_fps: Optional[float] = None
    analysis_width: Optional[int] = None
    record_retention_s: Optional[float] = None
    tracker_params_json: Optional[Dict[str, Any]] = None
    buffer_mode: str = BufferMode.AUTO.value
    ring_seconds: float = 20.0
    ring_max_mb: float = 32.0
    zones: List[WatchZoneRead] = Field(default_factory=list)


class WatchProfileSourceWrite(ApiModel):
    source_uuid: str
    order: int = 0
    enabled: bool = True
    model_group_uuid: Optional[str] = None
    analysis_fps: Optional[float] = None
    analysis_width: Optional[int] = None
    record_retention_s: Optional[float] = None
    tracker_params_json: Optional[Dict[str, Any]] = None
    buffer_mode: BufferMode = BufferMode.AUTO
    ring_seconds: float = 20.0
    ring_max_mb: float = 32.0
    zones: List[WatchZoneWrite] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_rates(self) -> "WatchProfileSourceWrite":
        if self.analysis_fps is not None and self.analysis_fps <= 0:
            raise ValueError("analysis_fps must be positive when set")
        if self.analysis_width is not None and self.analysis_width <= 0:
            raise ValueError("analysis_width must be positive when set")
        return self


class InteractionRuleRead(OrmModel):
    uuid: str
    profile_uuid: str
    order: int = 0
    name: str
    enabled: bool = True
    subject_a_kind: str
    subject_a_ref: Optional[str] = None
    relation: str
    subject_b_kind: str
    subject_b_ref: Optional[str] = None
    threshold: float = 0.5
    min_duration_s: float = 2.0
    rearm_delay_s: float = 10.0
    max_duration_s: float = 300.0
    event_label: Optional[str] = None


class InteractionRuleWrite(ApiModel):
    uuid: Optional[str] = None
    order: int = 0
    name: str
    enabled: bool = True
    subject_a_kind: RuleSubjectKind = RuleSubjectKind.LABEL
    subject_a_ref: Optional[str] = None
    relation: RuleRelation = RuleRelation.CONTAINS
    subject_b_kind: RuleSubjectKind = RuleSubjectKind.NONE
    subject_b_ref: Optional[str] = None
    threshold: float = 0.5
    min_duration_s: float = 2.0
    rearm_delay_s: float = 10.0
    max_duration_s: float = 300.0
    event_label: Optional[str] = None

    @model_validator(mode="after")
    def _validate_arity(self) -> "InteractionRuleWrite":
        # `present` is the arity-1 relation; every other relation compares two things,
        # so a missing second subject would silently never fire.
        if self.relation == RuleRelation.PRESENT:
            if self.subject_b_kind != RuleSubjectKind.NONE:
                raise ValueError("relation 'present' takes one subject; subject_b_kind must be 'none'")
        elif self.subject_b_kind == RuleSubjectKind.NONE:
            raise ValueError(f"relation {self.relation.value!r} needs a second subject")

        for kind, ref, side in (
            (self.subject_a_kind, self.subject_a_ref, "a"),
            (self.subject_b_kind, self.subject_b_ref, "b"),
        ):
            if kind in (RuleSubjectKind.LABEL, RuleSubjectKind.ZONE) and not ref:
                raise ValueError(f"subject_{side}_kind {kind.value!r} requires subject_{side}_ref")

        if self.min_duration_s < 0 or self.rearm_delay_s < 0:
            raise ValueError("min_duration_s and rearm_delay_s must not be negative")
        if self.max_duration_s <= 0:
            raise ValueError("max_duration_s must be positive — it is the runaway-event cap")
        return self


class LabelLifetimeRead(OrmModel):
    uuid: str
    profile_uuid: str
    label_uuid: str
    ttl_s: float = 2.0
    is_anchor: bool = False


class LabelLifetimeWrite(ApiModel):
    label_uuid: str
    ttl_s: float = 2.0
    is_anchor: bool = False

    @field_validator("ttl_s")
    @classmethod
    def _positive_ttl(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("ttl_s is in seconds and must be positive")
        return v


class WatchActionRead(OrmModel):
    uuid: str
    profile_uuid: str
    rule_uuid: Optional[str] = None
    order: int = 0
    enabled: bool = True
    kind: str
    params_json: Dict[str, Any] = Field(default_factory=dict)
    trigger: str


class WatchActionWrite(ApiModel):
    uuid: Optional[str] = None
    rule_uuid: Optional[str] = None  # None = all rules in the profile
    order: int = 0
    enabled: bool = True
    kind: str
    params_json: Dict[str, Any] = Field(default_factory=dict)
    trigger: WatchActionTrigger = WatchActionTrigger.EVENT_ENDED


class WatchProfileRead(OrmModel):
    uuid: str
    name: str
    description: Optional[str] = None
    enabled: bool = True
    tracker_typename: str
    tracker_params_json: Dict[str, Any] = Field(default_factory=dict)
    default_model_group_uuid: Optional[str] = None
    default_analysis_fps: float = 5.0
    default_analysis_width: int = 640
    sources: List[WatchProfileSourceRead] = Field(default_factory=list)
    rules: List[InteractionRuleRead] = Field(default_factory=list)
    actions: List[WatchActionRead] = Field(default_factory=list)
    label_lifetimes: List[LabelLifetimeRead] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class WatchProfileCreate(ApiModel):
    name: str
    description: Optional[str] = None
    enabled: bool = True
    tracker_typename: str = "IouTracker"
    tracker_params_json: Dict[str, Any] = Field(default_factory=dict)
    default_model_group_uuid: Optional[str] = None
    default_analysis_fps: float = 5.0
    default_analysis_width: int = 640
    sources: List[WatchProfileSourceWrite] = Field(default_factory=list)
    rules: List[InteractionRuleWrite] = Field(default_factory=list)
    actions: List[WatchActionWrite] = Field(default_factory=list)
    label_lifetimes: List[LabelLifetimeWrite] = Field(default_factory=list)


class WatchProfileUpdate(ApiModel):
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    tracker_typename: Optional[str] = None
    tracker_params_json: Optional[Dict[str, Any]] = None
    default_model_group_uuid: Optional[str] = None
    default_analysis_fps: Optional[float] = None
    default_analysis_width: Optional[int] = None
    # None = no change; [] = remove all.
    sources: Optional[List[WatchProfileSourceWrite]] = None
    rules: Optional[List[InteractionRuleWrite]] = None
    actions: Optional[List[WatchActionWrite]] = None
    label_lifetimes: Optional[List[LabelLifetimeWrite]] = None


class WatchEventRead(OrmModel):
    uuid: str
    profile_uuid: str
    source_uuid: Optional[str] = None
    rule_uuid: Optional[str] = None
    started_at: datetime
    ended_at: Optional[datetime] = None
    peak_confidence: float = 0.0
    track_ids_json: List[Any] = Field(default_factory=list)
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    recording_uuid: Optional[str] = None


class RecordingRead(OrmModel):
    uuid: str
    source_uuid: Optional[str] = None
    profile_uuid: Optional[str] = None
    event_uuid: Optional[str] = None
    file_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_s: float = 0.0
    size_bytes: int = 0
    status: str
    error: Optional[str] = None
    partial_preroll: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
