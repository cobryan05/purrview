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
# Marks a bbox whose runner-up class scored close enough to the winner that the
# model is effectively undecided — "confident but confused" (see
# AMBIGUITY_REVIEW_PLAN.md). Deliberately NOT in the "verification" exclusive
# group: a box is independently confusable *and* unverified/verified.
CONFUSABLE_TAG_UUID = "00000000-0000-0000-0000-000000000006"
# Marks an *image* pulled out of a clip by HarvestBlipFramesTask.
#
# Image-scope, unlike every tag above it: the frame is the thing that came from a
# harvest, and it arrives with no bbox at all — the proposal lives on a preview
# candidate until someone applies the run. A bbox tag would have nothing to
# attach to.
#
# It exists because the preview run is not a permanent address. Applying a run,
# or deleting it, takes the harvested frames out of Saved Previews and leaves
# them indistinguishable from any other image in the dataset — which is exactly
# the "I harvest frames and then lose them" complaint. This tag survives all of
# that, so "show me everything I have ever harvested" stays answerable.
HARVESTED_TAG_UUID = "00000000-0000-0000-0000-000000000007"
# Marks a bbox whose position came from the tracker, not the detector — a
# confirmed track coasted for a frame or two, the detector missed it, and the
# collector seeded the tracker's interpolated box instead of dropping the
# frame (COLLECTOR_UNIFICATION_PLAN.md §4A). Bbox-scope like Confusable, and
# for the same reason: it is a statement about one box, and NOT in the
# "verification" group — a detector-miss box is unverified like any other
# fresh detection, this tag only marks *why* it exists so it can be found and
# prioritised in review.
DETECTOR_MISS_TAG_UUID = "00000000-0000-0000-0000-000000000008"

# System tags that used to exist and must never come back. A tags.json dumped
# before the removing migration ran still lists them, and sync_tags_from_json()
# runs on exactly the rebuild/restore path those sidecars exist for — without
# this denylist the tag is re-inserted by the "no match → insert new" branch.
# 000...003 is the old False Positive verification tag (migration c1e5a83f7d92).
RETIRED_TAG_UUIDS = frozenset({"00000000-0000-0000-0000-000000000003"})

# System dataset UUIDs
WHOLE_DATABASE_DATASET_UUID = "00000000-0000-0000-0000-000000000002"

# Every bboxes.metadata_json key a model confidence has ever been written under,
# newest-preferred first. The mirror of CONFIDENCE_KEYS in
# static/js/shared/domain/image/confidence.js — keep the two in sync.
#   - "confidence"        — CollectTrainingImagesTask, canvas inference, apply-preview-run
#   - "source_confidence" — ApplyFilterChainTask's add_bbox step
# Reading only one silently blanks the confidence on whole classes of bbox.
BBOX_CONFIDENCE_METADATA_KEYS = ("confidence", "source_confidence")


def strip_bbox_confidence(metadata_json: "Optional[Dict[str, Any]]") -> "Dict[str, Any]":
    """Return *metadata_json* without any model-confidence key.

    A confidence describes what a MODEL guessed. Once a human has verified the
    box that guess is settled history, not live information, so a verified box
    stores none — see `db_client._apply_verified_confidence_rule`, the single
    enforcement point. Every other metadata key is preserved.
    """
    return {k: v for k, v in (metadata_json or {}).items() if k not in BBOX_CONFIDENCE_METADATA_KEYS}


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

from apps.helpers.timeUtils import ensure_utc


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
# Timestamp column type
# ─────────────────────────────────────────────────────────────────────────────


class UtcDateTime(sa.types.TypeDecorator):
    """A `DateTime` that is guaranteed to be timezone-aware UTC on the way out.

    SQLite has no offset-bearing storage type. `DateTime(timezone=True)` is
    accepted but not honoured: the driver writes the naive wall-clock digits and
    reads them back with `tzinfo=None`, silently converting "3pm UTC" into
    "3pm, zone unknown".

    That loss is invisible in Python — naive datetimes compare and format fine —
    and only surfaces at the API boundary, where a naive datetime serialises to
    `"2026-08-20T15:47:55"`. Per ES2015 a browser parses an offset-less
    *date-time* string as **local time**, so every timestamp rendered in the UI
    was shifted by the viewer's UTC offset with nothing raising an error.

    Fixing it here rather than per-field means every timestamp column, and every
    consumer of one, gets the guarantee for free — including the ORM-level
    comparisons that were previously re-stamping `tzinfo` by hand at each call
    site.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], dialect) -> Optional[datetime]:
        """Normalise to UTC before storing, so the stored digits are always UTC."""
        if value is None:
            return None
        if value.tzinfo is None:
            # Naive input is assumed UTC: every writer in this codebase uses
            # `datetime.now(timezone.utc)`, so this is a no-op in practice and a
            # safe reading for anything hand-rolled.
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: Optional[datetime], dialect) -> Optional[datetime]:
        """Re-attach the UTC offset the storage layer discarded."""
        return ensure_utc(value)


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
        sa_type=UtcDateTime(),
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_type=UtcDateTime(),
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
    # ── Home Assistant publishing ────────────────────────────────────────────────
    #
    # On the label rather than on a watch profile because the question is global: a
    # label either is or is not something the user wants represented in their house's
    # automation system. The same label watched by two profiles should not appear in HA
    # under one and not the other.
    #
    # Off means the label gets no subject device at all — no entities, and its existing
    # discovery is retracted. Its events are still recorded and still drive rule-level
    # entities; this suppresses the *per-label* device only.
    #
    # Default **off**, and it is an opt-in for the same reason `publish_per_source` is.
    # A label device is justified by the label rather than by a rule mentioning it, which
    # is what makes `Detected now` useful without inventing a rule — but it also means
    # the vocabulary is what bounds the device count, and a vocabulary is a working list
    # that grows with every class a model can name. Defaulting on turned a 43-label tree
    # into 43 Home Assistant devices, most of them things nobody automates on. Home
    # Assistant charges per *device* in its UI, so the cost lands where it is most
    # visible. Tick the handful you actually want in the house.
    publish_to_ha: bool = Field(default=False)
    # The finest tier: one counter per (label, rule, source). Default off because this
    # is the only term in the entity count that multiplies all three axes together —
    # every other tier is additive. A label tree of any size makes it the dominant cost,
    # and it answers a question ("how often at *that* camera specifically") that the
    # rule-level event entity's attributes already answer for automations. Opt in per
    # label where the per-camera breakdown is genuinely wanted as long-term statistics.
    publish_per_source: bool = Field(default=False)
    # When True, this label's children are alternate guesses for the SAME physical
    # object (e.g. breed/color variants), not different objects that can coexist. NMS
    # groups them together at inference time instead of by raw class_id, so an
    # ambiguous detection collapses to its single best guess instead of drawing one
    # box per candidate class. Field-level default is off (matches a bare `Label(...)`
    # construction with nothing said either way); the label-creation API applies a
    # smarter parent-aware default instead — see `AddLabelPayload.exclusive_variants`
    # in apps/db_server/web.py.
    exclusive_variants: bool = Field(default=False)
    # Only meaningful when exclusive_variants is True. Deliberately stricter than the
    # model's general NMS iou_thresh (0.45): two genuinely distinct objects of the same
    # exclusive-variant group can sit close together with moderate box overlap, and a
    # low threshold would wrongly collapse them. None means "use the default" (set in
    # apps/helpers/yoloUtils.py) rather than baking a magic number into every row.
    exclusive_variants_iou_threshold: Optional[float] = Field(default=None)


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
    # Overridden for the same reason as the two enums above: without an explicit
    # `server_default` the table SQLModel creates has NOT NULL and no DEFAULT, while the
    # migration's has both — so a row inserted without these columns succeeds against a
    # migrated database and fails against a freshly created one. That divergence shows
    # up only as an IntegrityError in whichever of the two a given environment built.
    publish_to_ha: bool = Field(
        default=False,
        sa_column=Column(sa.Boolean(), nullable=False, server_default=sa.text("0")),
    )
    publish_per_source: bool = Field(
        default=False,
        sa_column=Column(sa.Boolean(), nullable=False, server_default=sa.text("0")),
    )
    exclusive_variants: bool = Field(
        default=False,
        sa_column=Column(sa.Boolean(), nullable=False, server_default=sa.text("0")),
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
    #: All three opt-in. A read model that defaulted on would make a row serialised
    #: before the column existed arrive at the UI ticked.
    publish_to_ha: bool = False
    publish_per_source: bool = False
    exclusive_variants: bool = False
    exclusive_variants_iou_threshold: Optional[float] = None


class LabelUpdate(ApiModel):
    name: Optional[str] = None
    color: Optional[str] = None
    parent_uuid: Optional[str] = None
    label_type: Optional[LabelType] = None
    scope: Optional[LabelScope] = None
    publish_to_ha: Optional[bool] = None
    publish_per_source: Optional[bool] = None
    exclusive_variants: Optional[bool] = None
    exclusive_variants_iou_threshold: Optional[float] = None


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
        sa_type=UtcDateTime(),
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
    # Which source produced this image ("the camera that took it"). NULL for
    # manual uploads, for anything collected before this column existed, and for
    # frames whose producer had no sources row. Deliberately a real FK rather
    # than a per-source tag: sources get renamed, and a tag named after a camera
    # drifts from it. ondelete="SET NULL" so deleting a camera does not delete
    # the training data it collected — it only forgets where it came from.
    source_uuid: Optional[str] = Field(
        default=None, foreign_key="sources.uuid", ondelete="SET NULL", index=True, nullable=True
    )
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
    source_uuid: Optional[str] = None
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


def confidence_in_band(
    confidence: "Optional[float]",
    minimum: "Optional[float]" = None,
    maximum: "Optional[float]" = None,
    include_missing: bool = False,
) -> bool:
    """Does this confidence fall in [minimum, maximum]?

    Bounds are inclusive and independently optional; a band with neither bound
    matches everything (the caller decides whether to apply it at all). None —
    a bbox with no confidence — matches only under `include_missing`, the same
    rule as BBoxConfidenceTerm and the client-side `bboxMatchesConfidence`.

    Takes the value, not the bbox, so callers with the generated
    `BBox.confidence` column loaded (the bulk-apply loop) can pass it straight
    in instead of re-parsing the JSON it was generated from.
    """
    if confidence is None:
        return include_missing
    if minimum is not None and confidence < minimum:
        return False
    if maximum is not None and confidence > maximum:
        return False
    return True


class BBoxConfidenceRange(ApiModel):
    """The confidence band as a **bbox-level** predicate for bulk-apply/count.

    Same three fields as BBoxConfidenceTerm, deliberately as a separate model:
    the term selects *images* that have a bbox in the band, which cannot stop a
    bulk action from also hitting the out-of-band bboxes on those images — the
    exact split that already exists for bbox_verification (see
    BulkApplyAllPayload in db_server/web.py). No `negate`: the grid has no
    negated-confidence view to mirror.
    """

    min: Optional[float] = None
    max: Optional[float] = None
    include_missing: bool = False

    def is_active(self) -> bool:
        """True when a bound is set. `include_missing` alone matches everything,
        so it is not on its own a reason to narrow anything."""
        return self.min is not None or self.max is not None


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


class ImageSourceTerm(ApiModel):
    """Matches on images.source_uuid — "which camera took this".

    Multi-select like BBoxTagTerm, but "all" would be vacuous (an image has
    exactly one source), so there is no match_mode: the listed sources are
    OR'd. `include_unknown` additionally matches images with no source
    recorded — manual uploads, and everything collected before the column
    existed — which is otherwise unreachable, since NULL never satisfies IN.
    An empty selection with include_unknown=False is vacuously true.
    """

    kind: Literal["imageSource"] = "imageSource"
    source_uuids: List[str] = Field(default_factory=list)
    include_unknown: bool = False
    negate: bool = False


class BBoxVerificationTerm(ApiModel):
    """Matches images having a BBOX in the given verification state.

    Backed by the system verification tags (UNVERIFIED / VERIFIED), which are
    mutually exclusive per bbox via exclusive_group="verification".

    `no_boxes` is the odd one out and is deliberately here rather than in a term
    of its own: it answers the same question the user is asking ("what still
    needs looking at?") and belongs in the same one-of control. It matches
    images with NO bbox at all — which is what "unverified because there is
    nothing on it" actually looks like in the data, and which every tag-based
    state necessarily misses, since a tag needs a box to hang off.
    """

    kind: Literal["bboxVerification"] = "bboxVerification"
    state: Literal["verified", "unverified", "no_boxes"]
    negate: bool = False


class BBoxConfidenceTerm(ApiModel):
    """Matches images having a BBOX whose model confidence falls in [min, max].

    Confidence is not a column — it lives in `bboxes.metadata_json` under one of
    `BBOX_CONFIDENCE_METADATA_KEYS`, written by whichever producer created the
    box, and (for some producers) as a *string* rather than a number. The
    predicate therefore coalesces across the keys and casts to REAL; see
    `_bbox_confidence_expr` in db_client.py.

    Bounds are inclusive and each is independently optional (min-only = "at
    least", max-only = "at most"). Boxes carrying no confidence at all match
    only when `include_missing` is set — an unset confidence is not 0.0.
    """

    kind: Literal["bboxConfidence"] = "bboxConfidence"
    min: Optional[float] = None
    max: Optional[float] = None
    include_missing: bool = False
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
    """Substring patterns matched against images.filename.

    match_mode combines the patterns *within* this term: "all" (the default,
    and the historical behaviour) requires every pattern to appear in the
    path; "any" requires at least one. Separate filename terms are still
    ANDed like every other term, so an exclusion is expressed as its own
    negated term rather than as a pattern in this list.
    """

    kind: Literal["filename"] = "filename"
    patterns: List[str] = Field(default_factory=list)
    match_mode: Literal["any", "all"] = "all"
    negate: bool = False


FilterTerm = Annotated[
    Union[
        ObjectLabelTerm,
        ImageLabelTerm,
        BBoxTagTerm,
        ImageTagTerm,
        ImageSourceTerm,
        BBoxVerificationTerm,
        BBoxConfidenceTerm,
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


def _bbox_confidence_sql() -> str:
    """SQL for the `bboxes.confidence` generated column.

    Built from BBOX_CONFIDENCE_METADATA_KEYS so the column and the JS reader
    (`readConfidence`, which still parses the raw metadata client-side to draw
    the badge) can't drift apart on which keys count or in what order. Python
    has no reader of its own any more — `BBox.confidence` IS the read path.

    Why a CASE rather than a bare CAST: SQLite's CAST of a non-numeric string
    yields **0.0**, so a junk value ('n/a', '') would materialise as the lowest
    possible confidence and be swept into every low-confidence band — the exact
    boxes this filter exists to find. Junk lands as NULL instead, matching both
    readers (`float()` raises → None; `parseFloat` → NaN → undefined). A
    genuine "0"/"0.0" string still reads as 0.0 via the GLOB branch.
    """
    raw = "CASE"
    for key in BBOX_CONFIDENCE_METADATA_KEYS[:-1]:
        raw += (
            f" WHEN json_extract(metadata_json, '$.{key}') IS NOT NULL" f" THEN json_extract(metadata_json, '$.{key}')"
        )
    raw += f" ELSE json_extract(metadata_json, '$.{BBOX_CONFIDENCE_METADATA_KEYS[-1]}') END"
    return (
        "CASE"
        f" WHEN {raw} IS NULL THEN NULL"
        f" WHEN typeof({raw}) IN ('integer', 'real') THEN {raw}"
        f" WHEN CAST({raw} AS REAL) != 0.0 THEN CAST({raw} AS REAL)"
        f" WHEN trim({raw}) GLOB '[+-]*' OR trim({raw}) GLOB '[0.]*' THEN CAST({raw} AS REAL)"
        " ELSE NULL END"
    )


# The generated-column expression, as it must appear in both create_all and the
# migration that adds it to an existing DB.
BBOX_CONFIDENCE_SQL = _bbox_confidence_sql()


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
    #
    # (confidence, image_uuid): serves the confidence-band filter
    # (BBoxConfidenceTerm) as a covering range seek — the band predicate reads
    # confidence and returns image_uuid, so neither column needs a row lookup.
    __table_args__ = (
        sa.Index("ix_bboxes_label_uuid_image_uuid", "label_uuid", "image_uuid"),
        sa.Index("ix_bboxes_image_uuid_label_uuid", "image_uuid", "label_uuid"),
        sa.Index("ix_bboxes_confidence_image_uuid", "confidence", "image_uuid"),
    )
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    image_uuid: str = Field(foreign_key="images.uuid", ondelete="CASCADE")
    label_uuid: str = Field(foreign_key="labels.uuid")

    # Model confidence, denormalised out of metadata_json so it can be INDEXED.
    #
    # A **generated** column, deliberately, rather than a plain column the
    # writers maintain: confidence is written by half a dozen producers
    # (CollectTrainingImagesTask, the canvas save, ApplyFilterChainTask,
    # preview-run apply, …) under two different keys and two different JSON
    # types, and any one of them forgetting to also set a plain column would
    # silently drop boxes out of the filter. SQLite recomputes this on every
    # INSERT/UPDATE of metadata_json, so it cannot drift and no writer changes.
    #
    # VIRTUAL, not STORED: SQLite only allows ALTER TABLE ADD COLUMN for
    # virtual generated columns, and a virtual column can still be indexed (the
    # index materialises the value), which is all the query planner needs.
    confidence: Optional[float] = Field(
        default=None,
        sa_column=Column("confidence", sa.Float, sa.Computed(BBOX_CONFIDENCE_SQL, persisted=False), nullable=True),
    )

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
    # Fills in images.source_uuid when it is still blank; never clears it.
    source_uuid: Optional[str] = None


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
    started_at: Optional[datetime] = Field(default=None, sa_type=UtcDateTime())
    ended_at: Optional[datetime] = Field(default=None, sa_type=UtcDateTime())
    heartbeat_at: Optional[datetime] = Field(default=None, sa_type=UtcDateTime())
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
    expires_at: Optional[datetime] = Field(default=None, sa_type=UtcDateTime())


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
    applied_at: Optional[datetime] = Field(default=None, sa_type=UtcDateTime())


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
    #: The "≠" half of the same two dropdowns. Separate lists rather than a mode flag,
    #: because include and exclude are routinely used *together* — "every cat except
    #: Willow" is one filter, not two passes — and that is the contract every other
    #: multi-select in the app already offers (`createMultiSelectDropdown`'s
    #: `allowExclude`). Exclusion wins where the two overlap, which is what the widget
    #: enforces on its own side too.
    exclude_decisions: List[PreviewDecision] = Field(default_factory=list)
    exclude_proposed_label_uuids: List[str] = Field(default_factory=list)
    confidence_min: Optional[float] = None
    confidence_max: Optional[float] = None
    is_new: Optional[bool] = None
    source_models: List[str] = Field(default_factory=list)
    # image_confidence groups every candidate from one image together, ranked by
    # that image's best candidate — the modal did this client-side, which does
    # not survive pagination. label_confidence is the same trick keyed on the
    # proposed label, so a whole class can be triaged as a block.
    sort_by: Literal["confidence", "image_confidence", "label_confidence", "created_at"] = "image_confidence"
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
        sa_type=UtcDateTime(),
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
    #: True when a separate sub-model starts at this label: the top model rolls this
    #: label's descendants up into it, and a second-pass model trained here tells them
    #: apart. Explicit since it used to be *inferred* from "spec differs from the
    #: inherited spec", which silently tied nesting depth to architecture choice —
    #: efficientnet_b0 under efficientnet_b0 read as one model, not two, so a tree
    #: could not repeat an architecture down a chain. Backfilled from that old rule
    #: (migration 995286dce2d9), so existing configs keep the targets they had.
    is_boundary: bool = Field(default=False)
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
    is_boundary: bool = False


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
    # Where a sub-model starts. Ignored on the root node, which is the top model by
    # definition. None (the default) means "not stated", and is resolved by the old
    # spec-differs rule so a client written before this field keeps producing the tree
    # it always did — a plain `False` default would silently flatten every such config
    # to a single top model.
    is_boundary: Optional[bool] = None


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


# ─── Training collectors ─────────────────────────────────────────────────────


class TrainingCollectorRow(TimestampModel, table=True):
    """The collect-training-images parameter set, stored as a reusable entity.

    Deliberately not a `task_configs` row: a task config is *runnable*, and a
    collector has no sources of its own by construction, so it would appear in
    Jobs offering a start button that cannot work.

    `params_json` holds exactly what `collector_params_schema()` describes —
    never the loop params (`source_uuids`, `snapshot_interval`,
    `stream_interval`), which belong to whoever drives the frames.
    """

    __tablename__ = "training_collectors"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    name: str = Field(unique=True, index=True)
    description: Optional[str] = None
    enabled: bool = Field(default=True)
    params_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class TrainingCollectorRead(OrmModel):
    uuid: str
    name: str
    description: Optional[str] = None
    enabled: bool = True
    params: Dict[str, Any] = Field(default_factory=dict, validation_alias="params_json")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TrainingCollectorCreate(ApiModel):
    name: str
    description: Optional[str] = None
    enabled: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)


class TrainingCollectorUpdate(ApiModel):
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None  # None = no change


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
    # Arity 0: pixels changed, no detection involved. Subject A names *where* to
    # measure (zones, or nothing for the whole frame) rather than what to look for,
    # which is why it is the one relation that fires on a frame with no tracks at all.
    MOTION = "motion"


class RuleTrigger(str, Enum):
    """*When* a rule fires, as opposed to `RuleRelation`, which is *what it compares*.

    `WHILE` is the original and stays the default: the event spans the predicate, from
    the moment it has held for `min_duration_s` to the moment it has been clear for
    `rearm_delay_s`. Everything the watcher recorded before this enum existed is this.

    The other two are *impulses* — a zero-length event, START and END at the same
    instant — and they exist because the interesting moment is often an edge, not a
    span. "Record the last 15 seconds before the car leaves the garage, but only if it
    was there at least 15 minutes" has no duration to record: it is one instant, and the
    footage that matters is on either side of it, which is exactly what an action's
    `preroll_s`/`postroll_s` already express.

    Both edges reuse the guards a `while` rule already has rather than inventing their
    own, which is what keeps them cheap to reason about:

    * `min_duration_s` is the *dwell*: how long the predicate must have held before
      either edge counts. On `ON_EXIT` this is the "only if it was there at least 15
      minutes" clause, for free.
    * `rearm_delay_s` is the *debounce*: on `ON_EXIT` how long the subject must be gone
      before its departure is believed (a tracker dropping a car for two frames must not
      fire "car left"), and on `ON_ENTER` how long it must stay gone before an arrival
      can fire again.
    * `max_duration_s` does not apply at all. There is no span to cap, and applying it
      would break the headline case outright — a 15-minute dwell under the default
      300 s cap would be truncated before it could ever fire.
    """

    WHILE = "while"
    ON_ENTER = "on_enter"
    ON_EXIT = "on_exit"


class ZoneKind(str, Enum):
    BOX = "box"
    POLYGON = "polygon"


class RegionRole(str, Enum):
    """What a drawn region *does*, which is the whole reason zones and masks differ.

    A `ZONE` is inert until a rule names it by uuid: it is a noun the rule sentence can
    point at, and an unreferenced one costs nothing. A `MASK` is the opposite — it acts
    on every detection whether or not anything references it, suppressing anything whose
    centre falls inside. That asymmetry is why they can share one table and one editor
    but not one mental model.
    """

    ZONE = "zone"
    MASK = "mask"


class WatchActionTrigger(str, Enum):
    OBJECT_DETECTED = "object_detected"
    TRACK_STARTED = "track_started"
    EVENT_STARTED = "event_started"
    EVENT_ENDED = "event_ended"
    # Fires once per analysed frame per source, whether or not any rule is open. Every
    # other trigger is an *event* transition, which is the wrong shape for a live feed:
    # a feed that only publishes while an event is open shows a frozen frame for the
    # 99% of the day when nothing is happening.
    FRAME_ANALYSED = "frame_analysed"


class BufferMode(str, Enum):
    AUTO = "auto"  # use whatever ingest tier the source resolves to
    OFF = "off"  # never arm a rolling buffer for this source


class RecordingKind(str, Enum):
    """What sort of media a `recordings` row points at.

    One table rather than two, because everything that is *about* the file is identical:
    the retention bounds, the size cap, the orphan sweep, the file endpoint and the
    `<video>` that plays it. Only the way it was produced differs, and that is exactly
    one column's worth of difference.
    """

    #: Cut out of the rolling buffer at event close — real time, camera frame rate.
    CLIP = "clip"
    #: Sampled at an interval while the event was open, compiled at close. Minutes or
    #: hours of an event compressed into seconds.
    TIMELAPSE = "timelapse"


class RecordingStatus(str, Enum):
    PENDING = "pending"
    WRITING = "writing"
    COMPLETE = "complete"
    FAILED = "failed"
    # The window aged out of the RAM buffer before extraction ran. Terminal: retrying
    # can never succeed, so it must not sit in the work queue forever.
    EXPIRED = "expired"


class SourceRegion(SQLModel, table=True):
    """A user-drawn region on one source's frame — a zone or a mask.

    Zones are the highest value-per-effort part of the whole feature: a box named
    "litter box area" is more reliable than a litter-box detector, needs no training
    data, and makes `cat ∩ zone` work today.

    Regions hang off the **source**, not off a watcher's use of it, because the geometry
    describes the camera's view rather than anyone's opinion of it. Drawing the litter
    box once per watcher meant redrawing it in every watcher when the camera got bumped,
    and it is what forced the rule editor to hot-patch its own zone dropdowns from an
    unsaved modal. A watcher that wants a mask *not* to apply says so in
    `WatchProfileSource.disabled_mask_uuids_json`.
    """

    __tablename__ = "source_regions"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    source_uuid: str = Field(foreign_key="sources.uuid", index=True, ondelete="CASCADE")
    name: str
    role: str = Field(default=RegionRole.ZONE.value)  # validated against RegionRole in write models
    kind: str = Field(default=ZoneKind.BOX.value)  # validated against ZoneKind in write models
    points_json: List[Any] = Field(default_factory=list, sa_column=Column(JSON))
    # Coordinates are stored normalised (0..1), but the resolution they were drawn
    # against is kept so a camera resolution change is detectable rather than silently
    # shifting every zone.
    reference_width: int = Field(default=0)
    reference_height: int = Field(default=0)
    order: int = Field(default=0)


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
    # As on InteractionRule: `enabled` stops the camera being watched, this stops only
    # its Home Assistant representation. Watching a camera without publishing it is the
    # case this exists for — recording and event history continue, the device and its
    # entities go away.
    #
    # `server_default` so the created and migrated schemas agree — see `Label`.
    publish_to_ha: bool = Field(
        default=True,
        sa_column=Column(sa.Boolean(), nullable=False, server_default=sa.text("1")),
    )
    # What `{source}` renders as in this watcher's MQTT topics. NULL or blank falls back
    # to the source's own name.
    #
    # It lives on the junction, not on the Source, because the two names answer to
    # different audiences: the source's name is for the person picking a camera out of a
    # list ("Back room — Reolink, upstairs"), while this one is a topic level that
    # someone's Home Assistant automation is written against and must stay short and
    # stable. Renaming the camera in the Sources pane would otherwise silently re-route
    # every topic it publishes to.
    mqtt_name: Optional[str] = Field(default=None)
    # NULL means "use the profile default" for each of these.
    model_group_uuid: Optional[str] = Field(default=None, foreign_key="model_groups.uuid", ondelete="SET NULL")
    # A single model, for the common case where a stack of one is pure ceremony. Mutually
    # exclusive with model_group_uuid — enforced by WatchProfileSourceWrite, not the DB
    # row, the same way ModelGroupMember handles its own model_name/child_group_uuid pair.
    # Not an FK: models are files under MODELS_ROOT, not rows.
    model_name: Optional[str] = Field(default=None)
    analysis_fps: Optional[float] = Field(default=None)
    analysis_width: Optional[int] = Field(default=None)
    record_retention_s: Optional[float] = Field(default=None)
    tracker_params_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    buffer_mode: str = Field(default=BufferMode.AUTO.value)
    # Tier C (snapshot/MQTT sources) only: the in-RAM JPEG frame ring's bounds.
    ring_seconds: float = Field(default=20.0)
    ring_max_mb: float = Field(default=32.0)
    # Masks act on every detection, so the source owning them is right for the common
    # case ("ignore the neighbour's driveway, always") and wrong for the rare one (a
    # people-watcher masking the street a car-watcher needs). Opting out per watcher
    # keeps the common case zero-configuration — an empty list means every mask applies,
    # so a mask added to a camera protects watchers written before it existed.
    disabled_mask_uuids_json: List[Any] = Field(default_factory=list, sa_column=Column(JSON))
    profile: "WatchProfile" = Relationship(back_populates="sources")


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
    # Distinct from `enabled`, and the distinction is the point: `enabled` stops the
    # rule being evaluated at all, so no event is raised, nothing is recorded and no
    # clip is kept. This stops only the Home Assistant *representation* — the rule still
    # runs, still files events, still shows in the event browser, it simply has no
    # entities. Off retracts whatever discovery it previously had, on every device it
    # appeared on (camera, subject and the rules device alike).
    #
    # `server_default` so the created and migrated schemas agree — see `Label`.
    publish_to_ha: bool = Field(
        default=True,
        sa_column=Column(sa.Boolean(), nullable=False, server_default=sa.text("1")),
    )
    # Independent of publish_to_ha, and off by default unlike it: a snapshot is a
    # meaningfully bigger MQTT payload than the sensors/counters publish_to_ha alone
    # already gates, so this is opt-in per rule rather than inherited "free" the way
    # the rest of a rule's entities are. Meaningless when publish_to_ha is off — there
    # is no device for the snapshot entity to hang off then — but stored independently
    # rather than implied, so re-enabling publish_to_ha doesn't silently resurrect it.
    publish_snapshot_to_ha: bool = Field(
        default=False,
        sa_column=Column(sa.Boolean(), nullable=False, server_default=sa.text("0")),
    )
    # Which sources this rule is evaluated on. **Empty means every source in the
    # profile**, matching `WatchAction`'s two scope lists, so a rule written before this
    # column existed — or by someone who never opens the picker — still runs everywhere.
    #
    # This is the sharper instrument of the two scopes, and the reason both exist:
    # narrowing the *action* leaves the event raised, logged and visible on every camera
    # and only silences the side effect, while narrowing the rule suppresses the event
    # itself. "A cat is present" is true in every room and is only worth recording in
    # one, so scoping the action there would fill the event browser with events the user
    # explicitly does not care about.
    #
    # Not a foreign key, for the same reason as `WatchAction`'s lists: the nested write
    # replaces every child row of a profile wholesale, so an FK would fire against
    # sources not yet inserted. `_validate_watch_profile` enforces it at the API
    # boundary, where it can name the rule and the uuid.
    source_uuids_json: List[Any] = Field(default_factory=list, sa_column=Column(JSON))
    subject_a_kind: str = Field(default=RuleSubjectKind.LABEL.value)
    # A *list* of label (or zone) uuids: "when any of these three cats is inside the
    # litter box" is one rule, not three. Which member actually matched is reported on
    # the event so an action's template can name it.
    subject_a_refs_json: List[Any] = Field(default_factory=list, sa_column=Column(JSON))
    # "…and everything under it in the label tree", resolved when the watcher starts
    # rather than stored expanded. Storing the descendants would freeze the rule at the
    # moment it was written: a label added to the tree later would silently not match,
    # which is exactly the maintenance this flag exists to avoid. Label slots only —
    # zones have no hierarchy.
    subject_a_include_descendants: bool = Field(default=False)
    relation: str = Field(default=RuleRelation.CONTAINS.value)
    subject_b_kind: str = Field(default=RuleSubjectKind.NONE.value)
    subject_b_refs_json: List[Any] = Field(default_factory=list, sa_column=Column(JSON))
    subject_b_include_descendants: bool = Field(default=False)
    threshold: float = Field(default=0.5)
    # Widens the match before the relation is measured, so "the cat got *near* the water
    # bowl" is the same rule as "in it" with wider boxes, rather than a relation of its
    # own. Measured in units of the smaller box's size — 1.0 closes a gap of about one
    # subject-width — so it means the same thing at any distance from the camera, which
    # a fraction of the frame would not. Negative shrinks, demanding a deeper overlap
    # than the raw boxes show. See `apps.helpers.watch.geometry.margin_pad`.
    margin: float = Field(default=0.0)
    # `motion` only: how big a per-pixel luminance change counts as movement (0..1).
    # The independent other half of `threshold`, which says how much of the region must
    # change — a noisy IR camera needs a higher sensitivity *and* the same area, and one
    # number cannot say both. 0 means "use the default" (see `watch.motion`).
    motion_sensitivity: float = Field(default=0.0)
    # Whether the event spans the predicate or is an impulse at one of its edges. See
    # `RuleTrigger`; `server_default` so the created and migrated schemas agree, and so
    # every rule written before this column existed keeps the only behaviour it had.
    trigger: str = Field(
        default=RuleTrigger.WHILE.value,
        sa_column=Column(sa.String(), nullable=False, server_default=sa.text("'while'")),
    )
    # The predicate must hold continuously this long before the event STARTS — or, for
    # an impulse trigger, the dwell it must have served before either edge counts.
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
    # A *list*, exactly as a rule's subject slot is: "these three cats survive 1 s" is one
    # row, not three, and the picker is the same hierarchy-aware multi-select.
    #
    # Not a foreign key any more, for the same reason as `InteractionRule`'s ref lists:
    # a JSON list cannot carry one. `_validate_watch_profile` checks the labels exist at
    # the API boundary, where it can name the offending uuid.
    label_uuids_json: List[Any] = Field(default_factory=list, sa_column=Column(JSON))
    # "…and everything under it in the label tree", resolved when the watcher starts
    # rather than stored expanded — see `InteractionRule.subject_a_include_descendants`
    # for why the flag is stored and not its closure.
    include_descendants: bool = Field(default=False)
    ttl_s: float = Field(default=2.0)
    # Anchors latch their box once seen with high confidence and hold it for a long
    # TTL, rather than trusting a detector that intermittently misses a stationary
    # object like a litter box. Cheaper and far more robust than re-detecting it.
    is_anchor: bool = Field(default=False)
    profile: "WatchProfile" = Relationship(back_populates="label_lifetimes")


class WatchAction(SQLModel, table=True):
    """A side effect to run at some point in an event's life.

    Scoped on two axes, both **empty-means-all** — see `apps.helpers.watch.scoping` for
    why the scope lives here rather than on the rule. Lists rather than the single
    nullable FK this started as: "record on these two cameras" is one action, and
    spelling it as two rows means two sets of parameters to keep in step.

    Neither list is a foreign key, which is a deliberate trade. The nested write path
    replaces every child row of a profile wholesale, so an FK would have to be
    satisfied mid-transaction against rules that have not been inserted yet.
    `_validate_watch_children` enforces the same integrity at the API boundary, where
    it can say which action and which rule rather than raising an integrity error.
    """

    __tablename__ = "watch_actions"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    profile_uuid: str = Field(foreign_key="watch_profiles.uuid", index=True, ondelete="CASCADE")
    # Empty means "every rule in this profile".
    rule_uuids_json: List[Any] = Field(default_factory=list, sa_column=Column(JSON))
    # Empty means "every source in this profile". This is the axis that removes the
    # reason to duplicate a watcher per camera.
    source_uuids_json: List[Any] = Field(default_factory=list, sa_column=Column(JSON))
    order: int = Field(default=0)
    enabled: bool = Field(default=True)
    # A label for humans, never read by the engine. A watcher's actions are otherwise
    # identified only by kind, so three "Save snapshot" rows scoped to different cameras
    # are indistinguishable in the list — and the parameters that tell them apart are
    # spread across a dense row. Optional: an unnamed action still says what kind it is.
    name: Optional[str] = Field(default=None)
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
    # The single-model counterpart of default_model_group_uuid; see WatchProfileSource.
    default_model_name: Optional[str] = Field(default=None)
    # Analysis rate and width are first-class because per-frame inference over
    # base64 HTTP is the throughput ceiling: 2–5 fps at 640 px tracks cats fine and
    # cuts payloads ~10x. Recording is unaffected — MediaMTX records the full-rate,
    # full-resolution stream regardless of what we analyse.
    default_analysis_fps: float = Field(default=5.0)
    default_analysis_width: int = Field(default=640)
    # Whether this watcher's clips get analysed for tracking blips when they land. Per
    # watcher rather than global because watchers differ in what they are for: a doorbell
    # profile watching for any motion has nothing to learn from identity flips, while a
    # litter-box profile distinguishing two similar cats is the whole reason this exists.
    blip_analysis_enabled: bool = Field(default=True)
    # Overrides for `apps.helpers.recordings.blips.BlipThresholds`. Only the keys the
    # user has actually changed; anything absent falls back to the conservative default.
    # JSON rather than a column each: ten tuning knobs would be ten columns that nothing
    # ever filters or sorts on, and they are read as one bundle or not at all.
    blip_thresholds_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
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
    """One piece of durable media on disk — a clip or an event timelapse.

    Independent of the rolling buffer: the buffer is RAM and evaporates, these are the
    only durable media. There is deliberately no 24/7 timeline.

    An event may own several of these, one per `kind`, which is why the link lives here
    on `event_uuid` and not as a single pointer on the event. A watcher recording both
    the clip and the timelapse of a cat's visit produces two rows for one event, and the
    review card shows a badge for each.
    """

    __tablename__ = "recordings"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    source_uuid: Optional[str] = Field(default=None, foreign_key="sources.uuid", index=True, ondelete="SET NULL")
    profile_uuid: Optional[str] = Field(default=None, foreign_key="watch_profiles.uuid", ondelete="SET NULL")
    event_uuid: Optional[str] = Field(default=None, index=True)
    # Validated against RecordingKind on the write models, stored as a plain string for
    # the same reason every other enum-ish column here is — see the WatchProfile note.
    kind: str = Field(default=RecordingKind.CLIP.value, index=True)
    file_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    #: Absolute path to the clip's detection-overlay sidecar, when one was written. The
    #: boxes the watcher saw, in clip-relative time — see
    #: `apps.helpers.recordings.detectionTrack`. Its own column rather than a derived
    #: name because the orphan sweep deletes any file under the recordings roots that no
    #: row names, so an unrecorded sidecar would be swept away an hour after it landed.
    overlay_path: Optional[str] = None
    started_at: Optional[datetime] = Field(default=None, sa_type=UtcDateTime())
    ended_at: Optional[datetime] = Field(default=None, sa_type=UtcDateTime())
    duration_s: float = Field(default=0.0)
    size_bytes: int = Field(default=0)
    status: str = Field(default=RecordingStatus.PENDING.value)
    error: Optional[str] = None
    # True when the requested pre-roll had already aged out of the RAM buffer. A
    # truncated clip must never be presented as though its pre-roll were intact.
    partial_preroll: bool = Field(default=False)
    # Whatever the producing action wants said about the file, for the player to show.
    # A timelapse puts its frame count, sample interval and the real span it compressed
    # here, because "18 seconds" on its own is a lie about what you are watching. JSON
    # rather than columns: it is display text on one kind of row, never a filter.
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class WatchEventReviewState(str, Enum):
    """Where an event stands in blip review."""

    #: Analysed or not, nobody has looked.
    UNREVIEWED = "unreviewed"
    #: A human has settled every blip on it, one way or the other.
    CONFIRMED = "confirmed"
    #: Looked at and deliberately set aside — the blips were not worth acting on.
    DISMISSED = "dismissed"
    #: Someone outside the app (Home Assistant) says the labels on this event are
    #: wrong. Orthogonal to blip triage — a labelling fault, not a tracking one; see
    #: `EventBlip` for the latter.
    FLAGGED = "flagged"


class BlipState(str, Enum):
    """Per-blip triage state.

    `EXPIRED` is not a triage outcome but a fact about the evidence: retention deleted
    the clip, so the blip can no longer be reviewed. It is kept rather than deleted so
    the record of what went wrong survives the video it was found in.
    """

    NEW = "new"
    CONFIRMED = "confirmed"
    HARVESTED = "harvested"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


#: Seconds of quiet that end an event *session* — one visit, however many times the rule
#: re-fired during it.
#:
#: A single trip to the water bowl re-fires its rule several times as `min_duration_s`
#: and `rearm_delay_s` cycle. Measured on live history, 32% of consecutive same-identity
#: events start within 30s of the previous one ending, and this default — the observed
#: median gap — turns 1277 events into 653 sessions.
#:
#: Lives here rather than on `DbClient` so the query layer and the API layer that
#: defaults the parameter share one definition; two copies would drift and the endpoint
#: would document a grouping it does not perform.
DEFAULT_SESSION_GAP_S = 120.0


class WatchEvent(SQLModel, table=True):
    """One opening and closing of a rule.

    Rows with `ended_at` set and no `recordings` row naming them are a work queue — but
    a short-deadline one, because the source window ages out of the RAM buffer in under
    a minute.

    The link to media is one-way, from `recordings.event_uuid` back to here. It used to
    be two-way, with a `recording_uuid` column answering "does this card have a video"
    without a join — which stopped being answerable by one pointer the moment an event
    could own both a clip and a timelapse.
    """

    __tablename__ = "watch_events"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    profile_uuid: str = Field(foreign_key="watch_profiles.uuid", index=True, ondelete="CASCADE")
    source_uuid: Optional[str] = Field(default=None, foreign_key="sources.uuid", index=True, ondelete="SET NULL")
    # ⚠️ Deliberately *not* a foreign key, unlike the two above. It used to be one with
    # `ondelete="SET NULL"`, and that silently destroyed the event history's link to its
    # rule: saving a watch profile replaces its nested lists wholesale
    # (`update_watch_profile` deletes every `InteractionRule` for the profile and
    # re-inserts them with their original uuids), so the delete fired SET NULL across
    # every event that rule had ever produced. The rule came back with the same uuid;
    # the events did not get their pointer back. The visible symptom was the Event type
    # filter matching only events fired since the last time anyone saved the profile.
    #
    # SET NULL was wrong even without that: the facets endpoint builds the dropdown from
    # `rule_uuid` values present in history precisely so a deleted rule keeps its month
    # of events filterable, which NULLing the column defeats.
    rule_uuid: Optional[str] = Field(default=None, index=True)
    # Denormalised from the rule at event time, on purpose, for the same reason as the
    # uuid above: a rule can be renamed or deleted long after the events it fired, and
    # an event review page whose rows go blank when someone tidies up a profile is
    # worthless. This is the history of what happened, not a live view of the config.
    rule_name: Optional[str] = Field(default=None)
    event_label: Optional[str] = Field(default=None)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=UtcDateTime())
    ended_at: Optional[datetime] = Field(default=None, sa_type=UtcDateTime())
    peak_confidence: float = Field(default=0.0)
    track_ids_json: List[Any] = Field(default_factory=list, sa_column=Column(JSON))
    # Which member of each multi-ref slot actually matched — the "acting" thing and the
    # thing it acted on. Columns rather than `metadata_json` keys because these are the
    # two filters the review page is built around, and a JSON extract cannot use an index.
    subject_ref: Optional[str] = Field(default=None, index=True)
    subject_name: Optional[str] = Field(default=None)
    target_ref: Optional[str] = Field(default=None, index=True)
    target_name: Optional[str] = Field(default=None)
    # The event's span stops short of what actually happened — see `close_reason` for
    # which of the two ways. Kept as its own column rather than derived in a query
    # because the events grid filters on it and rows written before `close_reason`
    # existed have nothing to derive it from.
    truncated: bool = Field(default=False)
    # Why the event ended: `predicate` (the ordinary ending), `cap` (`max_duration_s`
    # elapsed) or `forced` (the watcher stopped, paused or restarted with the event
    # open). One boolean could not tell the last two apart, so saving a watcher's config
    # — which restarts the task — reported every interrupted event as capped, including
    # for rules whose cap is disabled. Nullable: rows written before this column cannot
    # be back-filled, since the information was never recorded.
    close_reason: Optional[str] = Field(default=None)
    # Where this event stands in blip review. A column rather than a `metadata_json`
    # key for the same reason `subject_ref` is one: the events grid filters on it, and
    # a JSON extract cannot use an index.
    review_state: str = Field(default=WatchEventReviewState.UNREVIEWED.value, index=True)
    # A pinned event (and every recording it owns) is excluded from both the age-based
    # retention sweep and the size-cap eviction — see `DbClient.watch_event_uuids_before`,
    # the single query both sweeps page through. Also refused by `delete_watch_event`
    # itself, so a pin protects against an accidental manual delete as well as the
    # automatic ones; unpin first. Indexed because the eviction sweep filters on it every
    # pass, not just because it happens to be a column someone might search.
    pinned: bool = Field(default=False, index=True)
    # Absolute path to the event's own thumbnail, written by the task at START. Distinct
    # from whatever a `save_snapshot` action wrote: that one is a training image the user
    # curates, this one is the review page's card art and is deleted with the event.
    snapshot_path: Optional[str] = Field(default=None)
    # Absolute path to the *cropped* card art: the same moment as `snapshot_path`, cut
    # down to the boxes the event actually fired on and taken from the camera's own
    # frame rather than the analysis downscale.
    #
    # A second file rather than a replacement for the snapshot, because the two answer
    # different questions. The card asks "what was this event?", which a wide shot of a
    # room cannot answer — every card on a camera looks the same. The detail modal (for
    # an event with no clip), the timeline hover and the group/session cards ask "where
    # in the room?", which the crop cannot answer. Neither is a downscale of the other.
    #
    # Deliberately un-annotated, unlike the snapshot: at this zoom a drawn box covers
    # the animal it is pointing at. That also means it never needs the label palette.
    #
    # Null for an event whose boxes were never recorded, whose frame was unavailable, or
    # that predates this column — `has_crop` is what the UI checks, and the wide
    # snapshot is the fallback.
    crop_path: Optional[str] = Field(default=None)
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    __table_args__ = (
        # The grid's ORDER BY. `started_at` is the only column every reader of this
        # table sorts on and it had no index of its own — tolerable while the log was
        # small, but it is append-only and grows forever.
        sa.Index("ix_watch_events_started_at", "started_at"),
        # The identity grouping, in the exact column order `DbClient._IDENTITY_COLUMNS`
        # groups and partitions by, with `started_at` last so the representative
        # `ROW_NUMBER()` and the session window function can walk each partition in
        # order instead of sorting it.
        sa.Index(
            "ix_watch_events_identity_started_at",
            "rule_uuid",
            "subject_ref",
            "target_ref",
            "source_uuid",
            "started_at",
        ),
    )


class EventBlip(TimestampModel, table=True):
    """One stretch of a clip where tracking went wrong.

    Written by the analysis task from the clip's detection sidecar — see
    `apps.helpers.recordings.blips` for what each `kind` means and how fault is
    decided. An incident, not a frame: a four-second dropout is one row, not a hundred
    and twenty.

    **A blip outlives the clip it was found in.** Retention deletes recordings on its
    own schedule, and a blip whose clip has gone still records that something went
    wrong even though it can no longer be reviewed — hence `BlipState.EXPIRED` rather
    than a delete, and hence enough denormalised detail here to stand alone. That is
    also why `recording_uuid` carries no foreign key: a hard FK would either block the
    retention sweep or take the blip with it, and the same reasoning already keeps
    `recordings.event_uuid` FK-free in the other direction.

    The event is different — deleting an event should take its blips, so that link is
    a real cascade.
    """

    __tablename__ = "event_blips"
    uuid: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    event_uuid: str = Field(foreign_key="watch_events.uuid", index=True, ondelete="CASCADE")
    #: Which recording it was found in. Deliberately no FK — see the class docstring.
    recording_uuid: Optional[str] = Field(default=None, index=True)
    #: One of the `KIND_*` constants in `apps.helpers.recordings.blips`. A plain string
    #: for the same reason every other enum-ish column here is one.
    kind: str = Field(index=True)
    #: The tracker's id for the object involved. Only unique within its own clip.
    track_id: int = Field(default=0)
    #: Clip-relative seconds, matching the sidecar's time base so the player can seek
    #: straight to it with no clock arithmetic.
    start_s: float = Field(default=0.0)
    end_s: float = Field(default=0.0)
    #: How sure the detector was about the thing it then got wrong, 0..1. Confidently
    #: wrong is the most valuable training data, so this sorts the way attention should.
    severity: float = Field(default=0.0, index=True)
    #: `detector` or `tracker`.
    fault: str = Field(default="")
    #: Whether these frames may become training data. Derived from kind + fault at write
    #: time and stored, because it is the primary filter of the review surface and
    #: recomputing it in SQL would mean encoding the rule in two places.
    harvestable: bool = Field(default=False, index=True)
    state: str = Field(default=BlipState.NEW.value, index=True)
    #: Kind-specific supporting detail — the modal label, what was seen instead, how
    #: many samples, which track absorbed which. Display and provenance, never a filter.
    detail_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


# ─── Watch Profile read/write models ─────────────────────────────────────────
# Nested writes with whole-list replace, following the Filter Chains precedent.
# Enum-typed fields appear on the *write* models only: `sa.Enum(StrEnum)` stores
# names rather than values, so the columns stay plain `str` and validation lives here.


class SourceRegionRead(OrmModel):
    uuid: str
    source_uuid: str
    name: str
    role: str
    kind: str
    points_json: List[Any] = Field(default_factory=list)
    reference_width: int = 0
    reference_height: int = 0
    order: int = 0


class SourceRegionWrite(ApiModel):
    uuid: Optional[str] = None  # preserved on update so rules keep referring to it
    name: str
    role: RegionRole = RegionRole.ZONE
    kind: ZoneKind = ZoneKind.BOX
    points_json: List[Any] = Field(default_factory=list)
    reference_width: int = 0
    reference_height: int = 0
    order: int = 0

    @model_validator(mode="after")
    def _validate_points(self) -> "SourceRegionWrite":
        if self.kind == ZoneKind.BOX and len(self.points_json) != 2:
            raise ValueError("a box region needs exactly 2 points ([x1,y1],[x2,y2])")
        if self.kind == ZoneKind.POLYGON and len(self.points_json) < 3:
            raise ValueError("a polygon region needs at least 3 points")
        for point in self.points_json:
            if not (isinstance(point, (list, tuple)) and len(point) == 2):
                raise ValueError(f"region points must be [x, y] pairs, got {point!r}")
            # Normalised coordinates survive a camera resolution change; pixel values
            # would silently point somewhere else after one.
            if not all(0.0 <= float(c) <= 1.0 for c in point):
                raise ValueError(f"region points must be normalised to 0..1, got {point!r}")
        return self


class WatchProfileSourceRead(OrmModel):
    uuid: str
    profile_uuid: str
    source_uuid: str
    order: int = 0
    enabled: bool = True
    publish_to_ha: bool = True
    mqtt_name: Optional[str] = None
    model_group_uuid: Optional[str] = None
    model_name: Optional[str] = None
    analysis_fps: Optional[float] = None
    analysis_width: Optional[int] = None
    record_retention_s: Optional[float] = None
    tracker_params_json: Optional[Dict[str, Any]] = None
    buffer_mode: str = BufferMode.AUTO.value
    ring_seconds: float = 20.0
    ring_max_mb: float = 32.0
    disabled_mask_uuids_json: List[Any] = Field(default_factory=list)


class WatchProfileSourceWrite(ApiModel):
    source_uuid: str
    order: int = 0
    enabled: bool = True
    publish_to_ha: bool = True
    # Blank is stored as NULL by `_normalise_mqtt_name`, so "cleared" and "never set"
    # are the same state rather than two that behave alike but compare differently.
    mqtt_name: Optional[str] = None
    model_group_uuid: Optional[str] = None
    model_name: Optional[str] = None
    analysis_fps: Optional[float] = None
    analysis_width: Optional[int] = None
    record_retention_s: Optional[float] = None
    tracker_params_json: Optional[Dict[str, Any]] = None
    buffer_mode: BufferMode = BufferMode.AUTO
    ring_seconds: float = 20.0
    ring_max_mb: float = 32.0
    disabled_mask_uuids_json: List[Any] = Field(default_factory=list)

    @field_validator("mqtt_name")
    @classmethod
    def _normalise_mqtt_name(cls, v: Optional[str]) -> Optional[str]:
        """Empty and whitespace-only both mean "no override".

        A stored `""` would be indistinguishable from a real override at every call
        site that tests it with `or`, and a stored `"  "` would slugify to the topic
        level "unnamed" — a name nobody typed, published to a topic nobody expects.
        """
        if v is None:
            return None
        return v.strip() or None

    @model_validator(mode="after")
    def _validate_rates(self) -> "WatchProfileSourceWrite":
        if self.analysis_fps is not None and self.analysis_fps <= 0:
            raise ValueError("analysis_fps must be positive when set")
        if self.analysis_width is not None and self.analysis_width <= 0:
            raise ValueError("analysis_width must be positive when set")
        # Both set has no sensible reading — silently preferring one would make the
        # UI's other dropdown a control that does nothing.
        if self.model_group_uuid and self.model_name:
            raise ValueError("a source selects either a model stack or a single model, not both")
        return self


class InteractionRuleRead(OrmModel):
    uuid: str
    profile_uuid: str
    order: int = 0
    name: str
    enabled: bool = True
    publish_to_ha: bool = True
    publish_snapshot_to_ha: bool = False
    source_uuids_json: List[Any] = Field(default_factory=list)
    subject_a_kind: str
    subject_a_refs_json: List[Any] = Field(default_factory=list)
    subject_a_include_descendants: bool = False
    relation: str
    subject_b_kind: str
    subject_b_refs_json: List[Any] = Field(default_factory=list)
    subject_b_include_descendants: bool = False
    threshold: float = 0.5
    margin: float = 0.0
    motion_sensitivity: float = 0.0
    trigger: str = RuleTrigger.WHILE.value
    min_duration_s: float = 2.0
    rearm_delay_s: float = 10.0
    max_duration_s: float = 300.0
    event_label: Optional[str] = None


class InteractionRuleWrite(ApiModel):
    uuid: Optional[str] = None
    order: int = 0
    name: str
    enabled: bool = True
    publish_to_ha: bool = True
    publish_snapshot_to_ha: bool = False
    # Empty means "every source in this profile" — see `InteractionRule`.
    source_uuids_json: List[Any] = Field(default_factory=list)
    subject_a_kind: RuleSubjectKind = RuleSubjectKind.LABEL
    subject_a_refs_json: List[Any] = Field(default_factory=list)
    subject_a_include_descendants: bool = False
    relation: RuleRelation = RuleRelation.CONTAINS
    subject_b_kind: RuleSubjectKind = RuleSubjectKind.NONE
    subject_b_refs_json: List[Any] = Field(default_factory=list)
    subject_b_include_descendants: bool = False
    threshold: float = 0.5
    # Bounded because the control is a slider and the ends have to mean something: 2.0
    # already fires on objects two subject-widths apart, which is most of a small room,
    # and -1.0 erodes a box to nothing — a rule that can never fire.
    margin: float = Field(default=0.0, ge=-1.0, le=2.0)
    # Bounded like `margin`, and for the same reason: the ends of the slider have to
    # mean something. 1.0 demands a full black-to-white swing on a pixel and can never
    # fire; 0 is not "most sensitive" but "use the default", so the floor is above it.
    motion_sensitivity: float = Field(default=0.0, ge=0.0, le=1.0)
    trigger: RuleTrigger = RuleTrigger.WHILE
    min_duration_s: float = 2.0
    rearm_delay_s: float = 10.0
    max_duration_s: float = 300.0
    event_label: Optional[str] = None

    @model_validator(mode="after")
    def _validate_arity(self) -> "InteractionRuleWrite":
        # `motion` is arity 0 — it reads pixels, not tracks. Subject A is optional and
        # means "where to measure": zones, or `any`/`none` for the whole frame. There is
        # nothing for a second subject to be, so naming one is a configuration error.
        if self.relation == RuleRelation.MOTION:
            if self.subject_b_kind != RuleSubjectKind.NONE:
                raise ValueError("relation 'motion' takes no second subject; subject_b_kind must be 'none'")
            if self.subject_a_kind == RuleSubjectKind.LABEL:
                raise ValueError(
                    "relation 'motion' does not look at labels; subject_a_kind must be 'zone' "
                    "(measure inside those zones) or 'any' (measure the whole frame)"
                )
        # `present` is the arity-1 relation; every other relation compares two things,
        # so a missing second subject would silently never fire.
        elif self.relation == RuleRelation.PRESENT:
            if self.subject_b_kind != RuleSubjectKind.NONE:
                raise ValueError("relation 'present' takes one subject; subject_b_kind must be 'none'")
        elif self.subject_b_kind == RuleSubjectKind.NONE:
            raise ValueError(f"relation {self.relation.value!r} needs a second subject")

        for kind, refs, side in (
            (self.subject_a_kind, self.subject_a_refs_json, "a"),
            (self.subject_b_kind, self.subject_b_refs_json, "b"),
        ):
            if kind in (RuleSubjectKind.LABEL, RuleSubjectKind.ZONE):
                if not refs:
                    raise ValueError(f"subject_{side}_kind {kind.value!r} requires at least one subject_{side} ref")
                if len(set(refs)) != len(refs):
                    # A duplicate would evaluate the same target twice and could report
                    # a matched slot value that looks arbitrary.
                    raise ValueError(f"subject_{side} refs contain duplicates")
            # A zone slot naming several zones is "inside any of these", which is a
            # coherent thing to want; a *subject* slot of zones is not, since zones
            # aren't tracked. That is already prevented by kind, not by count.

        # Only labels form a tree, so "include descendants" against a zone slot is a
        # control that silently does nothing. Rejected rather than ignored — the UI
        # hides the toggle for zones, so a payload carrying it means something is wrong.
        for kind, expand, side in (
            (self.subject_a_kind, self.subject_a_include_descendants, "a"),
            (self.subject_b_kind, self.subject_b_include_descendants, "b"),
        ):
            if expand and kind != RuleSubjectKind.LABEL:
                raise ValueError(
                    f"subject_{side}_include_descendants needs subject_{side}_kind 'label'; "
                    f"{kind.value!r} has no hierarchy to descend"
                )

        if len(set(map(str, self.source_uuids_json))) != len(self.source_uuids_json):
            raise ValueError("source_uuids_json contains duplicates")

        # A margin only means something to a relation that measures boxes. `present`
        # has no second box at all and `distance` compares centre points, which grow
        # with nothing — so a margin there is a control that silently does nothing, and
        # is rejected for the same reason as the descendants flag above.
        if self.margin and self.relation in (RuleRelation.PRESENT, RuleRelation.DISTANCE):
            raise ValueError(
                f"relation {self.relation.value!r} ignores margin; "
                "it compares no boxes, so send margin 0 (or use 'overlaps' to widen a match)"
            )

        if self.min_duration_s < 0 or self.rearm_delay_s < 0:
            raise ValueError("min_duration_s and rearm_delay_s must not be negative")
        # 0 is "no cap", the same reading the recording retention settings give it, and
        # the way to say "this event should last exactly as long as its subject is
        # there" — a cat asleep in a window for two hours is one visit, and a positive cap
        # would cut the recording short of it. A positive cap now bounds the *recording*
        # rather than slicing the visit: it closes the event and the rule stays shut until
        # the subject has actually gone (see `RuleState.SPENT`). Negative is
        # still refused: it is not a spelling of anything, and accepting it would leave
        # two values meaning "off" with nothing to choose between them.
        if self.max_duration_s < 0:
            raise ValueError("max_duration_s must not be negative — use 0 to disable the runaway-event cap")
        return self


class LabelLifetimeRead(OrmModel):
    uuid: str
    profile_uuid: str
    label_uuids_json: List[Any] = Field(default_factory=list)
    include_descendants: bool = False
    ttl_s: float = 2.0
    is_anchor: bool = False


class LabelLifetimeWrite(ApiModel):
    label_uuids_json: List[Any] = Field(default_factory=list)
    include_descendants: bool = False
    ttl_s: float = 2.0
    is_anchor: bool = False

    @field_validator("ttl_s")
    @classmethod
    def _positive_ttl(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("ttl_s is in seconds and must be positive")
        return v

    @model_validator(mode="after")
    def _validate_labels(self) -> "LabelLifetimeWrite":
        # A row naming no label configures nothing; the UI drops empty rows on save, so
        # one arriving here means the payload is wrong rather than merely unfinished.
        if not self.label_uuids_json:
            raise ValueError("a label lifetime needs at least one label")
        if len(set(map(str, self.label_uuids_json))) != len(self.label_uuids_json):
            raise ValueError("label_uuids_json contains duplicates")
        return self


class WatchActionRead(OrmModel):
    uuid: str
    profile_uuid: str
    rule_uuids_json: List[Any] = Field(default_factory=list)
    source_uuids_json: List[Any] = Field(default_factory=list)
    order: int = 0
    enabled: bool = True
    name: Optional[str] = None
    kind: str
    params_json: Dict[str, Any] = Field(default_factory=dict)
    trigger: str


class WatchActionWrite(ApiModel):
    uuid: Optional[str] = None
    # Both empty = every rule / every source in the profile. See helpers.watch.scoping.
    rule_uuids_json: List[Any] = Field(default_factory=list)
    source_uuids_json: List[Any] = Field(default_factory=list)
    order: int = 0
    enabled: bool = True
    name: Optional[str] = None
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
    default_model_name: Optional[str] = None
    default_analysis_fps: float = 5.0
    default_analysis_width: int = 640
    blip_analysis_enabled: bool = True
    blip_thresholds_json: Dict[str, Any] = Field(default_factory=dict)
    sources: List[WatchProfileSourceRead] = Field(default_factory=list)
    rules: List[InteractionRuleRead] = Field(default_factory=list)
    actions: List[WatchActionRead] = Field(default_factory=list)
    label_lifetimes: List[LabelLifetimeRead] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("tracker_params_json", "blip_thresholds_json", mode="before")
    @classmethod
    def _null_json_is_empty(cls, v: Any) -> Any:
        """A row that predates a JSON column holds SQL NULL, not `{}`.

        `RecordingRead._null_metadata_is_empty` documents where this bites: typing a
        nullable JSON column as a plain dict makes validating an older row raise, and
        the endpoint returns null for the *whole profile* rather than for the one field.
        Every watcher in an existing install predates `blip_thresholds_json`.
        """
        return {} if v is None else v


class WatchProfileCreate(ApiModel):
    name: str
    description: Optional[str] = None
    enabled: bool = True
    tracker_typename: str = "IouTracker"
    tracker_params_json: Dict[str, Any] = Field(default_factory=dict)
    default_model_group_uuid: Optional[str] = None
    default_model_name: Optional[str] = None
    default_analysis_fps: float = 5.0
    default_analysis_width: int = 640
    blip_analysis_enabled: bool = True
    blip_thresholds_json: Dict[str, Any] = Field(default_factory=dict)
    sources: List[WatchProfileSourceWrite] = Field(default_factory=list)
    rules: List[InteractionRuleWrite] = Field(default_factory=list)
    actions: List[WatchActionWrite] = Field(default_factory=list)
    label_lifetimes: List[LabelLifetimeWrite] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_default_models(self) -> "WatchProfileCreate":
        if self.default_model_group_uuid and self.default_model_name:
            raise ValueError("a watcher's default is either a model stack or a single model, not both")
        return self


class WatchProfileUpdate(ApiModel):
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    tracker_typename: Optional[str] = None
    tracker_params_json: Optional[Dict[str, Any]] = None
    default_model_group_uuid: Optional[str] = None
    default_model_name: Optional[str] = None
    default_analysis_fps: Optional[float] = None
    default_analysis_width: Optional[int] = None
    blip_analysis_enabled: Optional[bool] = None
    blip_thresholds_json: Optional[Dict[str, Any]] = None
    # None = no change; [] = remove all.
    sources: Optional[List[WatchProfileSourceWrite]] = None
    rules: Optional[List[InteractionRuleWrite]] = None
    actions: Optional[List[WatchActionWrite]] = None
    label_lifetimes: Optional[List[LabelLifetimeWrite]] = None

    @model_validator(mode="after")
    def _validate_default_models(self) -> "WatchProfileUpdate":
        # Only fires when the caller explicitly sends both as non-null; a PATCH that
        # sets one and omits the other is the ordinary way to switch between them.
        if self.default_model_group_uuid and self.default_model_name:
            raise ValueError("a watcher's default is either a model stack or a single model, not both")
        return self


class EventRecordingRead(OrmModel):
    """One piece of media hanging off an event, as the review list needs it.

    A summary, not the whole `RecordingRead`: the events list fetches a page of these
    per request and the card only decides which badges to draw. The modal fetches the
    full row by uuid once the user opens it, which is the only place `file_path` and the
    overlay actually matter.
    """

    uuid: str
    kind: str
    duration_s: float = 0.0
    status: str = RecordingStatus.PENDING.value


class WatchEventRead(OrmModel):
    uuid: str
    profile_uuid: str
    source_uuid: Optional[str] = None
    rule_uuid: Optional[str] = None
    rule_name: Optional[str] = None
    event_label: Optional[str] = None
    started_at: datetime
    ended_at: Optional[datetime] = None
    peak_confidence: float = 0.0
    track_ids_json: List[Any] = Field(default_factory=list)
    subject_ref: Optional[str] = None
    subject_name: Optional[str] = None
    target_ref: Optional[str] = None
    target_name: Optional[str] = None
    truncated: bool = False
    #: `predicate` | `cap` | `forced`, or null for rows written before it was recorded.
    #: The UI must not spell out a cause when this is null — that guess is the bug this
    #: field exists to end.
    close_reason: Optional[str] = None
    review_state: str = WatchEventReviewState.UNREVIEWED.value
    #: Protected from the retention sweep, the size-cap eviction, and manual delete.
    #: See `WatchEvent.pinned`.
    pinned: bool = False
    # Whether the event has card art on disk. The path itself is deliberately not
    # exposed: it is a server filesystem path, and the browser fetches the image from
    # `/api/watch-events/{uuid}/snapshot` instead.
    has_snapshot: bool = False
    # Whether the event has a *cropped* card too, served from
    # `/api/watch-events/{uuid}/crop`. Separate from `has_snapshot` rather than implied
    # by it: an event can have the wide shot and no crop (no boxes were recorded, or it
    # predates the crop), and a card that assumed otherwise would show a broken image
    # instead of falling back to the picture it does have.
    has_crop: bool = False
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    #: How many blips the analyser found, by kind — `{"identity_flip": 2, ...}`. Filled
    #: in by the db client alongside `recordings`, so the events grid can badge a card
    #: without a second request per event. Empty both when an event is clean and when it
    #: has not been analysed; `review_state` is what tells those apart.
    blip_counts: Dict[str, int] = Field(default_factory=dict)
    #: Every recording naming this event, clip and timelapse alike. Filled in by the db
    #: client in one batched query per page — not a relationship, because `Recording`
    #: has no FK back to `watch_events` (an event can be deleted while its extraction is
    #: still in flight, and a hard FK would either block that or take the row with it).
    recordings: List[EventRecordingRead] = Field(default_factory=list)


class EventBlipRead(OrmModel):
    uuid: str
    event_uuid: str
    recording_uuid: Optional[str] = None
    kind: str
    track_id: int = 0
    start_s: float = 0.0
    end_s: float = 0.0
    severity: float = 0.0
    fault: str = ""
    harvestable: bool = False
    state: str = BlipState.NEW.value
    detail_json: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("detail_json", mode="before")
    @classmethod
    def _null_detail_is_empty(cls, v: Any) -> Any:
        """A NULL JSON column must validate as `{}`, not blow up the whole row.

        The same failure `RecordingRead._null_metadata_is_empty` documents: typing a
        nullable JSON column as a plain dict makes validating a row that predates it
        raise, and the endpoint then returns null for the entire object rather than for
        the one field.
        """
        return {} if v is None else v


class RecordingRead(OrmModel):
    uuid: str
    source_uuid: Optional[str] = None
    profile_uuid: Optional[str] = None
    event_uuid: Optional[str] = None
    kind: str = RecordingKind.CLIP.value
    file_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    overlay_path: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_s: float = 0.0
    size_bytes: int = 0
    status: str
    error: Optional[str] = None
    partial_preroll: bool = False
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("metadata_json", mode="before")
    @classmethod
    def _null_metadata_is_empty(cls, v: Any) -> Any:
        """A row that predates the column holds SQL NULL, not `{}`.

        Adding `metadata_json` as a nullable column left every existing recording with
        NULL, and this model typed it as a plain dict — so validating one raised, the
        endpoint returned `recording: null`, and the events modal, unable to tell that
        apart from a missing file, reported an install's entire clip history as "no
        longer on disk" while every mp4 sat untouched on the volume.

        A backfill migration fixes the rows that exist (see `9c2f3a1b7d40`). This is the
        rule that stops the next nullable column doing it again: an absent annotation is
        not a reason to refuse to show someone their video.
        """
        return {} if v is None else v
