import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, TypeVar, Union
from uuid import uuid4

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig

ALEMBIC_INI_PATH = Path(__file__).parent / "alembic.ini"

import jstyleson
from alembic.runtime.migration import MigrationContext
from pydantic import TypeAdapter
from sqlalchemy import Select, and_, case, delete, distinct, event, exists, func, or_, text, true, union
from sqlalchemy import update as sa_update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import defer, selectinload, sessionmaker
from sqlalchemy.orm.attributes import set_committed_value
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from apps import FILES_ROOT
from apps.helpers.consts import JsonValues, TaskStatus
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.recordings.blips import KIND_IDENTITY_SWITCH
from apps.helpers.suppressionGroups import suppression_group_for
from apps.helpers.watch.events import EventCloseReason

from .types import (
    BBOX_CONFIDENCE_METADATA_KEYS,
    CONFUSABLE_TAG_UUID,
    DEFAULT_SESSION_GAP_S,
    DETECTOR_MISS_TAG_UUID,
    HARVESTED_TAG_UUID,
    MODEL_TYPE_ARCHITECTURES,
    REJECTION_TAG_UUID,
    RETIRED_TAG_UUIDS,
    UNVERIFIED_TAG_UUID,
    VERIFIED_TAG_UUID,
    WHOLE_DATABASE_DATASET_UUID,
    BBox,
    BBoxConfidenceRange,
    BBoxConfidenceTerm,
    BBoxTagLink,
    BBoxTagTerm,
    BBoxVerificationTerm,
    BlipState,
    Dataset,
    DatasetCreate,
    DatasetFilterCriteria,
    DatasetLabelLink,
    DatasetRead,
    DatasetUpdate,
    DateTerm,
    EventBlip,
    EventBlipRead,
    EventRecordingRead,
    FilenameTerm,
    FilterChain,
    FilterChainCreate,
    FilterChainRead,
    FilterChainStep,
    FilterChainUpdate,
    FilterTerm,
    ImageItemStatus,
    ImageLabel,
    ImageLabelState,
    ImageLabelTerm,
    ImageLabelWithFilename,
    ImageRecord,
    ImageRecordRead,
    ImageRecordUpdate,
    ImageSearchParams,
    ImageSourceTerm,
    ImageTagLink,
    ImageTagTerm,
    InteractionRule,
    InteractionRuleWrite,
    Label,
    LabelLifetime,
    LabelLifetimeWrite,
    LabelScope,
    LabelType,
    LabelVerificationTerm,
    ModelConfig,
    ModelConfigCreate,
    ModelConfigNode,
    ModelConfigNodeRead,
    ModelConfigRead,
    ModelConfigUpdate,
    ModelGroup,
    ModelGroupCreate,
    ModelGroupMember,
    ModelGroupRead,
    ModelGroupUpdate,
    ObjectLabelTerm,
    PreviewCandidate,
    PreviewCandidateCreate,
    PreviewCandidateQuery,
    PreviewConflictPolicy,
    PreviewDecision,
    PreviewDecisionAction,
    PreviewRun,
    PreviewRunTTL,
    Recording,
    RecordingRead,
    RegionRole,
    RuleSubjectKind,
    Source,
    SourceRead,
    SourceRegion,
    SourceRegionRead,
    SourceRegionWrite,
    Tag,
    TagKind,
    TaskConfig,
    TaskInstance,
    TrainingCollectorCreate,
    TrainingCollectorRead,
    TrainingCollectorRow,
    TrainingCollectorUpdate,
    TrainingConfig,
    TrainingConfigCreate,
    TrainingConfigDatasetLink,
    TrainingConfigRead,
    TrainingConfigUpdate,
    UserSetting,
    WatchAction,
    WatchActionWrite,
    WatchEvent,
    WatchEventCounter,
    WatchEventRead,
    WatchProfile,
    WatchProfileCreate,
    WatchProfileRead,
    WatchProfileSource,
    WatchProfileSourceWrite,
    WatchProfileUpdate,
    strip_bbox_confidence,
)

_SYSTEM_TAGS = [
    Tag(
        uuid=UNVERIFIED_TAG_UUID,
        name="Unverified",
        color="#f0a030",
        kind=TagKind.SYSTEM,
        exclusive_group="verification",
        protected=True,
        icon="?",
    ),
    Tag(
        uuid=VERIFIED_TAG_UUID,
        name="Verified",
        color="#27ae60",
        kind=TagKind.SYSTEM,
        exclusive_group="verification",
        protected=True,
        icon="✓",
    ),
    Tag(
        uuid=REJECTION_TAG_UUID,
        name="Rejection",
        color="#8e44ad",
        kind=TagKind.SYSTEM,
        exclusive_group=None,
        protected=True,
        icon=None,
    ),
    Tag(
        uuid=CONFUSABLE_TAG_UUID,
        name="Confusable",
        color="#c0392b",
        kind=TagKind.SYSTEM,
        # Not a verification state — a box can be confusable and verified at once.
        exclusive_group=None,
        protected=True,
        # The icon is load-bearing, not decoration: BboxView._computeIconText
        # renders any tag's icon on the canvas, so this single field is what
        # marks a confusable box in the UI without a line of JS.
        icon="⇄",
    ),
    Tag(
        uuid=HARVESTED_TAG_UUID,
        name="Harvested",
        color="#2980b9",
        kind=TagKind.SYSTEM,
        exclusive_group=None,
        protected=True,
        # No icon: this one is applied to the *image*, and the canvas icon
        # renderer draws bbox tags. An icon here would never be seen.
        icon=None,
    ),
    Tag(
        uuid=DETECTOR_MISS_TAG_UUID,
        name="Detector Miss",
        color="#d4a017",
        kind=TagKind.SYSTEM,
        exclusive_group=None,
        protected=True,
        icon="~",
    ),
]

# Default image-scope labels seeded on startup. These are ordinary labels (not
# addressable system objects) — seeding only creates them if no label with the
# same name+scope already exists; it never assigns or relies on a fixed UUID.
_DEFAULT_IMAGE_LABEL_NAMES = ["Corrupt", "Blurry"]

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)

# Guards the read-merge-write of JSON sidecars (see DbClient._sidecar_lock).
# Sidecar writes run on the default thread pool via asyncio.to_thread, so this
# is thread contention within one process, not cross-process locking — db_server
# is the only service that writes sidecars.
_SIDECAR_LOCKS = tuple(threading.Lock() for _ in range(64))
logger.setLevel(LOG_LEVEL)


# ── Write epoch ───────────────────────────────────────────────────────────────
#
# The review queue's `total_count` is expensive, so `Manager` caches it. That
# cache had a TTL and nothing else: delete every image matching a filter and the
# grid emptied while the toolbar went on saying "Page 1 of 2" and "Apply to
# Filtered Images (500)" for up to a minute, because the cached count knew
# nothing about the delete.
#
# Rather than ask a dozen mutating methods to remember to invalidate it — a list
# that is wrong the moment someone adds the thirteenth — the epoch is bumped
# from the one place every write already passes through, the session flush.
_WRITE_EPOCH_INFO_KEY = "purrview_write_epoch"

# Only writes to tables a review count is derived from count. Task heartbeats
# and recording rows churn constantly and would otherwise keep the cache
# permanently cold, which is the same as not having one.
# Derived from the models rather than written out: the table names are not the
# class names ("images", not "imagerecord"), and a hand-written list that is
# quietly wrong disables the invalidation without failing anything.
_COUNT_RELEVANT_TABLES = frozenset(
    model.__table__.name for model in (ImageRecord, BBox, BBoxTagLink, ImageTagLink, ImageLabel, Label, Tag)
)

# The listener goes on the sync Session class, which is shared by every client
# in the process, so it is registered once and scoped per client by `info`.
_write_epoch_listener_registered = False


def _register_write_epoch_listener(sync_session_class: type) -> None:
    global _write_epoch_listener_registered
    if _write_epoch_listener_registered:
        return
    _write_epoch_listener_registered = True

    @event.listens_for(sync_session_class, "after_flush")
    def _bump_write_epoch(session, flush_context):  # noqa: ANN001 - SQLAlchemy signature
        epoch = session.info.get(_WRITE_EPOCH_INFO_KEY)
        if epoch is None:
            return  # a session from somewhere else in the process
        for obj in (*session.new, *session.deleted, *session.dirty):
            table = getattr(getattr(obj, "__table__", None), "name", None)
            if table in _COUNT_RELEVANT_TABLES:
                # Fires before commit, so a rolled-back flush over-invalidates.
                # That costs one recount and never serves a wrong number.
                epoch[0] += 1
                return


class SidecarParseError(Exception):
    """Raised when a present JSON sidecar cannot be parsed and proceeding would
    destroy DB state (e.g. an overwrite sync from a torn/corrupt sidecar)."""


class WatchEventPinnedError(Exception):
    """Raised by `delete_watch_event` when the target event is pinned.

    A pin is meant to protect against every delete path, not just the automatic
    sweeps — see `WatchEvent.pinned`. The caller must unpin before deleting; there is
    deliberately no force-delete parameter here, so that guarantee cannot be bypassed
    by one careless call site.
    """


T = TypeVar("T", bound=SQLModel)


LABELS_JSON = "labels.json"
TAGS_JSON = "tags.json"
SOURCES_JSON = "sources.json"
DATASETS_JSON = "datasets.json"
TASK_CONFIGS_JSON = "task_configs.json"
MODEL_CONFIGS_JSON = "model_configs.json"
DB_FILE = "db.sqlite"

# IN-vs-EXISTS dispatch for the image-search bbox filters (see
# _apply_image_search_filters / DbClient._should_use_exists_dispatch).
# Estimated matched-image fraction above which every positive tag/label
# predicate is emitted as a correlated EXISTS instead of IN (subquery).
# Measured crossover on a 100k-image DB is ~1-2%; 20% leaves a wide margin for
# the estimate's overestimation error (link counts vs distinct images).
_EXISTS_DISPATCH_MIN_FRACTION = 0.20
# TTL for the cached selectivity probe counts. Staleness only affects which
# query shape is chosen, never correctness, so this can be generous.
_SELECTIVITY_CACHE_TTL = 60.0
_SELECTIVITY_CACHE_MAX_ENTRIES = 512


def _escape_like_pattern(pattern: str) -> str:
    """Escape SQL LIKE metacharacters so user input is treated as literal text."""
    return pattern.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def _relative_filename_expr(files_root: Optional[Union[str, Path]] = None):
    """SQL expression yielding each image's FILES_ROOT-relative filename.

    The `images.filename` column stores absolute paths, but every record handed
    to a client is relativised first (`Manager._image_rel_path`), so filename
    *filtering* has to work on the relative form too — otherwise a pattern can
    match a prefix the user is never shown, and the server-side filter
    disagrees with the identical client-side one used in the review modals
    (`filterPredicate.js`'s `imageMatchesFilename`).

    Mirrors `_image_rel_path` exactly, fallback included: a row whose filename
    is not under FILES_ROOT (legacy imports, relative paths written before the
    root was enforced) is matched on its stored value unchanged, because that
    is also what the client displays for it.
    """
    root = str(Path(files_root) if files_root is not None else FILES_ROOT).rstrip("/")
    prefix = f"{root}/"
    # +1 to move past the last character of the prefix, +1 again because
    # SQLite's substr() is 1-indexed.
    return case(
        (
            ImageRecord.filename.like(f"{_escape_like_pattern(prefix)}%", escape="!"),
            func.substr(ImageRecord.filename, len(prefix) + 1),
        ),
        else_=ImageRecord.filename,
    )


def _label_uuids_with_descendants(label_uuids: List[str], cte_name: str = "label_desc") -> Select:
    """Return a SELECT of all label UUIDs that are the given labels OR any of
    their descendants (recursive).  Uses a recursive CTE so no separate query
    is needed; SQLite has supported recursive CTEs since 3.8.3 (2014).

    The returned statement is meant to be used in an IN() or EXISTS context,
    e.g. ``BBox.label_uuid.in_(_label_uuids_with_descendants(uuids))``.

    ``cte_name`` must be unique within the enclosing SQLAlchemy statement.
    When this function is called multiple times in the same statement (e.g.
    once per label in a loop), pass a distinct name each time to avoid a
    SQLAlchemy ``CompileError: Multiple, unrelated CTEs found with the same
    name``.
    """
    # Anchor: the explicitly requested labels
    anchor = select(Label.uuid).where(Label.uuid.in_(label_uuids))
    cte = anchor.cte(cte_name, recursive=True)
    # Recursive: children of any label already in the CTE
    recursive_part = select(Label.uuid).where(Label.parent_uuid == cte.c.uuid)
    cte = cte.union_all(recursive_part)
    return select(cte.c.uuid)


def _apply_image_search_filters(
    stmt: Select,
    params: ImageSearchParams,
    *,
    use_exists: bool = False,
    files_root: Optional[Union[str, Path]] = None,
) -> Select:
    """Apply all ImageSearchParams filter predicates to *stmt* and return it.

    Does not add ORDER BY, LIMIT, or keyset cursors — those are caller concerns.
    Works for any Select target (ImageRecord, ImageRecord.uuid, etc.).

    As of PR 1.4b, `terms` is the only filter surface (the old flat fields —
    tag_uuids, label_uuids, image_label_filters, date_from, etc. — were
    deleted; see _apply_filter_terms for the per-kind predicate building).

    Performance note: the bbox tag/label term predicates have two shapes,
    selected by *use_exists* (see DbClient._should_use_exists_dispatch):

    - use_exists=False (default): non-correlated ``IN (subquery)``. SQLite
      materialises the inner SELECT once as a hash table; right choice for
      rare predicates, and the only safe choice for zero-match ones.
    - use_exists=True: correlated ``EXISTS`` per image. When every predicate
      is common AND the caller orders by an indexed (sort_col, uuid) prefix
      with a LIMIT, SQLite scans the sort index and stops after LIMIT matches
      instead of materialising + sorting the whole matched set (measured
      ~916ms → ~0.3ms on a 100k-image DB for the default review view). On a
      rare predicate this degrades to a full-table scan, so callers must only
      set it when the *conjunction* of all predicates is common.
    """
    # Track whether any filter that already guarantees at least one bbox has
    # been applied, so we can skip the redundant require_bboxes subquery.
    _bbox_implied = False

    if params.terms:
        stmt, _bbox_implied = _apply_filter_terms(stmt, params, use_exists=use_exists, files_root=files_root)

    # ── Require at least one bbox ─────────────────────────────────────────────
    # Skip when any earlier filter already guarantees bbox existence.
    if params.require_bboxes and not _bbox_implied:
        image_uuids = select(BBox.image_uuid)
        stmt = stmt.where(ImageRecord.uuid.in_(image_uuids))

    return stmt


# Maps BBoxVerificationTerm.state to the system verification tag UUIDs. These
# tags are mutually exclusive per bbox (exclusive_group="verification").
BBOX_VERIFICATION_STATE_TO_TAG_UUID = {
    "verified": VERIFIED_TAG_UUID,
    "unverified": UNVERIFIED_TAG_UUID,
}


def _bbox_confidence_expr():
    """A bbox's model confidence, or NULL when it has none.

    This is the indexed `bboxes.confidence` column, not a json_extract over
    metadata_json. The column is GENERATED from that same JSON (see
    BBOX_CONFIDENCE_SQL in types.py), so it cannot drift from it, and it handles
    the two wrinkles a naive `json_extract(metadata_json, '$.confidence')`
    misses — the two keys producers disagree about, and the value being a string
    on rows the canvas wrote and a float on rows the tasks wrote.

    Reading the column rather than recomputing the extraction is what makes the
    band filter scale: `ix_bboxes_confidence_image_uuid` turns it into a
    covering range seek instead of a full scan of `bboxes`.
    """
    return BBox.confidence


def _bbox_confidence_condition(minimum, maximum, include_missing: bool):
    """Per-bbox WHERE condition for a confidence band.

    Shared by the image-selecting BBoxConfidenceTerm and the bbox-level narrowing
    in count_bboxes_search, so the count on the Apply-to-All button cannot mean
    something different from the filter that produced it. The in-memory twin is
    `confidence_in_band` (types.py), used by the bulk-apply loop.
    """
    conf = _bbox_confidence_expr()
    bounds = []
    if minimum is not None:
        bounds.append(conf >= minimum)
    if maximum is not None:
        bounds.append(conf <= maximum)
    if bounds:
        condition = and_(*bounds)
        # "Has no confidence at all" is a separate disjunct rather than a relaxed
        # bound: NULL fails every comparison, so without this an unscored box
        # could never satisfy the band.
        return or_(condition, conf.is_(None)) if include_missing else condition
    # No bounds: everything qualifies when unscored boxes are allowed, otherwise
    # this degenerates to "has a confidence at all".
    return true() if include_missing else conf.is_not(None)


def _apply_filter_terms(
    stmt: Select,
    params: ImageSearchParams,
    *,
    use_exists: bool = False,
    files_root: Optional[Union[str, Path]] = None,
) -> "tuple[Select, bool]":
    """Apply params.terms (the new discriminated-union filter model) to *stmt*.

    Returns (stmt, bbox_implied) — bbox_implied mirrors _apply_image_search_filters'
    _bbox_implied tracking: True if any (non-negated) term here already
    guarantees the image has at least one bbox, so callers can skip a redundant
    require_bboxes subquery.

    Each term contributes exactly one AND'd predicate to *stmt*. A term with
    match_mode="all" (objectLabel / bboxTag) or multiple date bounds is built
    as a single and_()/or_() expression first so that negate can invert the
    WHOLE term correctly (rather than negating each sub-condition
    independently, which would change the boolean structure).

    CTE naming: recursive descendant CTEs built here use the prefix
    "term_desc_{term_index}" (or "term_desc_{term_index}_{label_index}" for
    per-label expansion under match_mode="all"), keyed off the term's position
    in params.terms.
    """

    def _tag_predicate(tag_condition):
        if use_exists:
            return (
                select(BBox.uuid)
                .join(BBoxTagLink, BBoxTagLink.bbox_uuid == BBox.uuid)
                .where(BBox.image_uuid == ImageRecord.uuid, tag_condition)
                .exists()
                .correlate(ImageRecord)
            )
        return ImageRecord.uuid.in_(
            select(BBox.image_uuid).join(BBoxTagLink, BBoxTagLink.bbox_uuid == BBox.uuid).where(tag_condition)
        )

    def _label_predicate(label_condition):
        if use_exists:
            return (
                select(BBox.uuid)
                .where(BBox.image_uuid == ImageRecord.uuid, label_condition)
                .exists()
                .correlate(ImageRecord)
            )
        return ImageRecord.uuid.in_(select(BBox.image_uuid).where(label_condition))

    bbox_implied = False

    for i, term in enumerate(params.terms):
        expr = None
        term_implies_bbox = False

        if isinstance(term, ObjectLabelTerm):
            term_implies_bbox = bool(term.label_uuids)
            if term.match_mode == "all":
                sub_exprs = []
                for j, label_uuid in enumerate(term.label_uuids):
                    if term.roll_up:
                        label_set = _label_uuids_with_descendants([label_uuid], cte_name=f"term_desc_{i}_{j}")
                        condition = BBox.label_uuid.in_(label_set)
                    else:
                        condition = BBox.label_uuid == label_uuid
                    sub_exprs.append(_label_predicate(condition))
                expr = and_(*sub_exprs) if sub_exprs else true()
            else:
                if term.roll_up:
                    label_set = _label_uuids_with_descendants(term.label_uuids, cte_name=f"term_desc_{i}")
                    condition = BBox.label_uuid.in_(label_set)
                else:
                    condition = BBox.label_uuid.in_(term.label_uuids)
                expr = _label_predicate(condition)

        elif isinstance(term, ImageLabelTerm):
            # Roll-up applies to present / unreviewed / any_reviewed; ignored
            # for absent (see ImageLabelTerm.roll_up docstring in types.py for
            # why absent can't roll up).
            rolls_up = term.roll_up and term.state != ImageLabelState.ABSENT
            if rolls_up:
                label_condition = ImageLabel.label_uuid.in_(
                    _label_uuids_with_descendants([term.label_uuid], cte_name=f"term_desc_{i}")
                )
            else:
                label_condition = ImageLabel.label_uuid == term.label_uuid

            if term.state == ImageLabelState.UNREVIEWED:
                expr = ~(select(ImageLabel).where(ImageLabel.image_uuid == ImageRecord.uuid, label_condition).exists())
            elif term.state == ImageLabelState.ANY_REVIEWED:
                expr = (
                    select(ImageLabel)
                    .where(
                        ImageLabel.image_uuid == ImageRecord.uuid,
                        label_condition,
                        ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                    )
                    .exists()
                )
            else:
                target_status = (
                    ImageItemStatus.PRESENT if term.state == ImageLabelState.PRESENT else ImageItemStatus.ABSENT
                )
                expr = (
                    select(ImageLabel)
                    .where(
                        ImageLabel.image_uuid == ImageRecord.uuid,
                        label_condition,
                        ImageLabel.status == target_status,
                    )
                    .exists()
                )

        elif isinstance(term, BBoxTagTerm):
            term_implies_bbox = bool(term.tag_uuids)
            if term.match_mode == "all":
                sub_exprs = [_tag_predicate(BBoxTagLink.tag_uuid == tag_uuid) for tag_uuid in term.tag_uuids]
                expr = and_(*sub_exprs) if sub_exprs else true()
            else:
                expr = _tag_predicate(BBoxTagLink.tag_uuid.in_(term.tag_uuids))

        elif isinstance(term, ImageTagTerm):
            if term.state == ImageLabelState.UNREVIEWED:
                expr = ~(
                    select(ImageTagLink)
                    .where(ImageTagLink.image_uuid == ImageRecord.uuid, ImageTagLink.tag_uuid == term.tag_uuid)
                    .exists()
                )
            elif term.state == ImageLabelState.ANY_REVIEWED:
                expr = (
                    select(ImageTagLink)
                    .where(
                        ImageTagLink.image_uuid == ImageRecord.uuid,
                        ImageTagLink.tag_uuid == term.tag_uuid,
                        ImageTagLink.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                    )
                    .exists()
                )
            else:
                target_status = (
                    ImageItemStatus.PRESENT if term.state == ImageLabelState.PRESENT else ImageItemStatus.ABSENT
                )
                expr = (
                    select(ImageTagLink)
                    .where(
                        ImageTagLink.image_uuid == ImageRecord.uuid,
                        ImageTagLink.tag_uuid == term.tag_uuid,
                        ImageTagLink.status == target_status,
                    )
                    .exists()
                )

        elif isinstance(term, ImageSourceTerm):
            # A column predicate, not an EXISTS — images.source_uuid is indexed.
            sub_exprs = []
            if term.source_uuids:
                # The IS NOT NULL guard is not redundant. `NULL IN ('a')` is
                # NULL, and negation (line ~620 applies `~expr`) turns NULL into
                # NULL, not TRUE — so without it "not camera A" would silently
                # drop every image that has no source at all, which is the
                # opposite of what it says. `FALSE AND NULL` is FALSE, so the
                # guard collapses those rows to a definite FALSE that negates.
                sub_exprs.append(
                    and_(
                        ImageRecord.source_uuid.is_not(None),
                        ImageRecord.source_uuid.in_(term.source_uuids),
                    )
                )
            if term.include_unknown:
                sub_exprs.append(ImageRecord.source_uuid.is_(None))
            expr = or_(*sub_exprs) if sub_exprs else true()

        elif isinstance(term, BBoxVerificationTerm):
            if term.state == "no_boxes":
                # The inverse of every other state here, so it must NOT set
                # term_implies_bbox — that flag exists to let the query narrow to
                # images that have a box, and this term asks for exactly the ones
                # that do not. Setting it would make the predicate unsatisfiable.
                expr = ~(select(BBox).where(BBox.image_uuid == ImageRecord.uuid).exists())
            else:
                term_implies_bbox = True
                tag_uuid = BBOX_VERIFICATION_STATE_TO_TAG_UUID[term.state]
                expr = _tag_predicate(BBoxTagLink.tag_uuid == tag_uuid)

        elif isinstance(term, BBoxConfidenceTerm):
            term_implies_bbox = True
            expr = _label_predicate(_bbox_confidence_condition(term.min, term.max, term.include_missing))

        elif isinstance(term, LabelVerificationTerm):
            # No flat ImageSearchParams.label_uuids to fall back to as of 1.4b —
            # callers (build_dataset_search_params, get_training_config_needs_
            # review_images) must supply label_uuids explicitly on the term.
            targets = term.label_uuids
            if not targets:
                expr = true()
            elif term.check == "all_verified":
                sub_exprs = [
                    select(ImageLabel)
                    .where(
                        ImageLabel.image_uuid == ImageRecord.uuid,
                        ImageLabel.label_uuid == lu,
                        ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                    )
                    .exists()
                    for lu in targets
                ]
                expr = and_(*sub_exprs)
            else:  # has_unverified
                sub_exprs = [
                    ~(
                        select(ImageLabel)
                        .where(
                            ImageLabel.image_uuid == ImageRecord.uuid,
                            ImageLabel.label_uuid == lu,
                            ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                        )
                        .exists()
                    )
                    for lu in targets
                ]
                expr = or_(*sub_exprs)

        elif isinstance(term, DateTerm):
            col = ImageRecord.created_at if term.field == "created" else ImageRecord.updated_at
            sub_exprs = []
            if term.from_ is not None:
                sub_exprs.append(col >= term.from_)
            if term.to is not None:
                sub_exprs.append(col <= term.to)
            expr = and_(*sub_exprs) if sub_exprs else true()

        elif isinstance(term, FilenameTerm):
            # Matched against the FILES_ROOT-*relative* path, not the absolute
            # one the column stores. The column is absolute, but every record
            # that reaches a client has already been relativised by
            # Manager._image_rel_path, so matching the raw column let a pattern
            # hit on a prefix the user can never see ("image_datasets" matched
            # everything) and made this filter disagree with the identical
            # client-side one in the review modals (filterPredicate.js's
            # imageMatchesFilename). `_relative_filename_expr` mirrors
            # _image_rel_path exactly, including its fallback for rows that
            # aren't under FILES_ROOT at all.
            rel_filename = _relative_filename_expr(files_root)
            sub_exprs = []
            for pattern in term.patterns:
                escaped = _escape_like_pattern(pattern)
                sub_exprs.append(rel_filename.like(f"%{escaped}%", escape="!"))
            if not sub_exprs:
                expr = true()
            elif term.match_mode == "any":
                expr = or_(*sub_exprs)
            else:
                expr = and_(*sub_exprs)

        else:  # pragma: no cover - unreachable, pydantic discriminator rejects unknown kinds
            raise ValueError(f"Unknown filter term kind: {term!r}")

        stmt = stmt.where(~expr if term.negate else expr)
        # A negated term only proves the ABSENCE of a matching bbox — it does
        # not guarantee the image has any bbox at all, so it must never set
        # bbox_implied (an all-absent image with zero bboxes still passes).
        if term_implies_bbox and not term.negate:
            bbox_implied = True

    return stmt, bbox_implied


# Eager-load options shared by all training-config queries.
_TC_LOAD_OPTS = (selectinload(TrainingConfig.dataset_links),)

# Page size for image pagination in readiness checks.
_READINESS_PAGE_SIZE = 1000

# Max uuids per IN (?, ?, ...) list. SQLite hard-errors ("too many SQL
# variables") past SQLITE_MAX_VARIABLE_NUMBER — 250k on this build, as low as
# 999 on older ones — so any query filtering by an unbounded uuid list must be
# chunked. 10k keeps a wide margin on every build at negligible query overhead.
_UUID_IN_CHUNK = 10_000


def _crop_blip_frame(
    clip_path: str,
    document: Dict[str, Any],
    track_id: int,
    label_uuid: str,
    at_s: float,
) -> Optional[Dict[str, Any]]:
    """One before/after crop for `get_blip_crop_preview`, as a data URL.

    Run off the event loop (see its `asyncio.to_thread` callers) — both the video seek
    and the JPEG encode are blocking CPU/IO work. Cropped to the track's own box when the
    sidecar has one for this label at this instant (`proposed_box_at` — exact if the
    track was seen, interpolated and still usable otherwise); the whole frame is
    returned uncropped rather than nothing when it has none, since seeing the room is
    still more useful than no picture at all.
    """
    import base64

    import cv2

    from apps.helpers.recordings.blips import proposed_box_at
    from apps.helpers.recordings.eventCrop import CROP_WIDTH, crop_to_rect, focus_rect
    from apps.helpers.recordings.frameGrab import grab_frame_array

    frame = grab_frame_array(clip_path, at_s)
    if frame is None:
        return None

    height, width = frame.shape[:2]
    box = proposed_box_at(document, track_id, label_uuid, at_s) if label_uuid else None
    rect = focus_rect([(box.x, box.y, box.w, box.h)], width / height) if box and height else None
    cropped = crop_to_rect(frame, rect, CROP_WIDTH) if rect else frame

    ok, buffer = cv2.imencode(".jpg", cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return {
        "atS": at_s,
        "label": label_uuid or None,
        "dataUrl": f"data:image/jpeg;base64,{base64.b64encode(buffer.tobytes()).decode('ascii')}",
    }


def _chunked(seq: List, size: Optional[int] = None):
    """Yield consecutive slices of *seq* of at most *size* elements.

    size defaults to _UUID_IN_CHUNK, read at call time so tests can shrink it.
    """
    if size is None:
        size = _UUID_IN_CHUNK
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _keyset_start_index(
    images: Sequence["ImageRecord"],
    *,
    after_uuid: str,
    after_sort_value: Optional[datetime],
    sort_by: str,
    sort_order: str,
) -> Optional[int]:
    """Index of the first row *after* the cursor, in an already-sorted list.

    Returns ``None`` when the cursor cannot be located — the caller should then
    return an empty page rather than restarting from the top.

    ⚠️ A cursor with no ``after_sort_value`` cannot be honoured by comparing sort
    values, and comparing ``uuid`` instead is only correct when ``sort_by`` is
    ``"uuid"``. Under a ``created_at`` sort, uuids are random with respect to the
    row order, so a uuid comparison matches within the first row or two: the page
    barely advances and can settle on re-serving the *same* page forever, with
    ``has_more`` stuck true. That is an unbounded request loop for any client that
    pages until exhaustion — which is exactly what the review modals do when a
    client-side filter rejects a whole chunk (`app/review/main.js` sends
    ``afterSortValue: null``). Since the list is already materialised here, locate
    the cursor row itself and resume after it; no sort value is needed.
    """
    if sort_by != "uuid" and after_sort_value is None:
        for i, img in enumerate(images):
            if img.uuid == after_uuid:
                return i + 1
        # The cursor row is gone (deleted, or no longer matches the filter). There
        # is no sound resume point, so end the walk instead of starting over.
        return None

    for i, img in enumerate(images):
        if sort_by == "uuid":
            past = img.uuid > after_uuid if sort_order == "asc" else img.uuid < after_uuid
        else:
            attr = getattr(img, sort_by)
            if sort_order == "asc":
                past = attr > after_sort_value or (attr == after_sort_value and img.uuid > after_uuid)
            else:
                past = attr < after_sort_value or (attr == after_sort_value and img.uuid < after_uuid)
        if past:
            return i
    return None


class DbClient:
    """Helper class for SQLModel interactions for image annotations."""

    # Below this, a delete's freed pages are left for the next big cleanup to
    # reclaim rather than paying incremental_vacuum's I/O on every request.
    _INCREMENTAL_VACUUM_THRESHOLD = 500

    def __init__(
        self,
        db_dir: Union[str, Path],
        *,
        memory: bool = False,
        files_root: Optional[Union[str, Path]] = None,
    ):
        # `files_root` is only used to make FilenameTerm match the same
        # FILES_ROOT-relative path the client sees (Manager._image_rel_path
        # rewrites every outgoing record), rather than the absolute path the
        # column actually stores. Defaults to the process-wide FILES_ROOT so
        # existing callers (and every test fixture) keep working untouched.
        self._files_root: Path = Path(files_root) if files_root is not None else FILES_ROOT
        self._db_dir = Path(db_dir)
        self._db_path: Path = self._db_dir / DB_FILE
        self._labels_json_path: Path = self._db_dir / LABELS_JSON
        self._tags_json_path: Path = self._db_dir / TAGS_JSON
        self._sources_json_path: Path = self._db_dir / SOURCES_JSON
        self._datasets_json_path: Path = self._db_dir / DATASETS_JSON
        self._task_configs_json_path: Path = self._db_dir / TASK_CONFIGS_JSON
        self._model_configs_json_path: Path = self._db_dir / MODEL_CONFIGS_JSON
        self._memory: bool = memory

        if memory:
            self._url = "sqlite+aiosqlite:///:memory:"
            self._engine = create_async_engine(
                self._url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            self._url = f"sqlite+aiosqlite:///{self._db_path}"
            self._engine = create_async_engine(self._url, echo=False)

        @event.listens_for(self._engine.sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            # WAL mode allows concurrent reads during writes, preventing long
            # image-search queries from blocking all other db_server requests.
            cursor.execute("PRAGMA journal_mode=WAL")
            # NORMAL is safe with WAL and avoids the fsync overhead of FULL.
            cursor.execute("PRAGMA synchronous=NORMAL")
            # 64 MB page cache per connection; reduces repeated disk I/O on
            # image/bbox index scans (default is only ~2 MB).
            cursor.execute("PRAGMA cache_size=-65536")
            # Only takes effect on a brand-new (empty) DB file; an existing DB
            # created before this needs a one-time `PRAGMA auto_vacuum=INCREMENTAL;
            # VACUUM;` to convert. Without it, deleted rows just sit on a freelist
            # forever instead of being reclaimed by incremental_vacuum() below.
            cursor.execute("PRAGMA auto_vacuum=INCREMENTAL")
            cursor.close()

        # Every session this client makes carries the epoch box, which is how a
        # listener on the (shared, process-wide) sync Session class tells one
        # DbClient's writes from another's — tests build many clients at once.
        self._write_epoch = [0]
        self._async_session_maker = sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            info={_WRITE_EPOCH_INFO_KEY: self._write_epoch},
        )
        _register_write_epoch_listener(self._async_session_maker.class_.sync_session_class)
        # Cached selectivity probe counts for the IN-vs-EXISTS search dispatch.
        # Key → (count, monotonic timestamp); see _selectivity_count().
        self._selectivity_cache: Dict[str, tuple] = {}

    @property
    def write_epoch(self) -> int:
        """Bumped whenever this client flushes a change to a table the review
        counts are derived from. A cached count taken at one epoch is stale at
        the next — see `Manager.get_review_queue`, which caches counts and had
        no way to know a bulk delete had happened underneath it.
        """
        return self._write_epoch[0]

    def _bump_write_epoch(self) -> None:
        """Retire the cached review counts explicitly.

        The `after_flush` listener above only sees ORM unit-of-work changes.
        A Core statement handed to `session.exec()` — `delete(ImageRecord)`,
        `sqlite_insert(ImageLabel)` and friends — never populates
        `session.new/deleted/dirty`, so `flush()` short-circuits and the event
        never fires. `delete_images_bulk` is exactly that shape, and it is the
        "Apply to filtered images" path the epoch exists for, so every Core
        write to a count-relevant table calls this by hand.
        """
        self._write_epoch[0] += 1

    async def init_db(self) -> None:
        """Initialize database and run any pending Alembic migrations.

        Three cases:
          - In-memory DB  — always fresh; create_all only (alembic CLI can't reach
                            a :memory: engine, and migration history is irrelevant).
          - Fresh file DB — create_all + alembic stamp (fast, no migration replay).
          - Existing DB   — alembic upgrade head (incremental migrations).
        """
        self._db_dir.mkdir(parents=True, exist_ok=True)

        if self._memory:
            # In-memory: always empty on creation; skip alembic entirely.
            async with self._engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
        else:
            loop = asyncio.get_running_loop()
            alembic_cfg = AlembicConfig(ALEMBIC_INI_PATH)
            alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{self._db_path}")

            is_fresh = not self._db_path.exists() or self._db_path.stat().st_size == 0
            if is_fresh:
                # Fresh file DB: create schema in one shot, stamp as head.
                async with self._engine.begin() as conn:
                    await conn.run_sync(SQLModel.metadata.create_all)
                await loop.run_in_executor(None, lambda: alembic_command.stamp(alembic_cfg, "head"))
            else:
                # Existing DB — but guard against a crash between create_all and
                # alembic stamp on a previous startup.  If the schema was created
                # but never stamped, `upgrade head` would fail ("table already exists").
                # Use alembic's own MigrationContext to check: an empty heads tuple
                # means the DB is unstamped; stamp now so upgrade head can proceed.
                async with self._engine.connect() as conn:
                    is_stamped = await conn.run_sync(
                        lambda sync_conn: bool(MigrationContext.configure(sync_conn).get_current_heads())
                    )
                if not is_stamped:
                    await loop.run_in_executor(None, lambda: alembic_command.stamp(alembic_cfg, "head"))
                else:
                    await loop.run_in_executor(None, lambda: alembic_command.upgrade(alembic_cfg, "head"))

        # Count labels and tags before seeding system tags so the zero-check
        # reflects only user-defined data (system tag seeding would inflate tag_count).
        async with self._async_session_maker() as session:
            count_statement = select(func.count()).select_from(Label)
            result = await session.exec(count_statement)
            label_count = result.one() or 0

            count_statement = select(func.count()).select_from(Tag)
            result = await session.exec(count_statement)
            tag_count = result.one() or 0

        await self._seed_system_tags()
        await self._seed_system_datasets()
        await self._seed_system_labels()

        if label_count == 0:
            await self.sync_labels_from_json()
        if tag_count == 0:
            await self.sync_tags_from_json()

        async with self._async_session_maker() as session:
            count_statement = select(func.count()).select_from(TaskConfig)
            result = await session.exec(count_statement)
            task_config_count = result.one() or 0

        if task_config_count == 0:
            await self.sync_task_configs_from_json()

        async with self._async_session_maker() as session:
            count_statement = select(func.count()).select_from(Source)
            result = await session.exec(count_statement)
            source_count = result.one() or 0

        if source_count == 0:
            await self.sync_sources_from_json()

        async with self._async_session_maker() as session:
            # Exclude system datasets from the count so seeding "Whole Database"
            # doesn't prevent sync_datasets_from_json from running on a fresh DB.
            count_statement = (
                select(func.count()).select_from(Dataset).where(Dataset.uuid != WHOLE_DATABASE_DATASET_UUID)
            )
            result = await session.exec(count_statement)
            dataset_count = result.one() or 0

        if dataset_count == 0:
            await self.sync_datasets_from_json()

        async with self._async_session_maker() as session:
            count_statement = select(func.count()).select_from(ModelConfig)
            result = await session.exec(count_statement)
            model_config_count = result.one() or 0

        if model_config_count == 0:
            await self.sync_model_configs_from_json()
        elif not self._memory and not self._model_configs_json_path.exists():
            # Configs predating this dump have never been written out, and the
            # dump only runs on mutation — so without this they stay DB-only
            # until someone happens to edit one. Backfill on first startup.
            await self._sync_model_configs_to_json()

        # Refresh query-planner statistics so SQLite picks optimal indexes for
        # the current data distribution.  PRAGMA optimize is incremental — it
        # only runs ANALYZE on tables/indexes whose stats are stale, so it is
        # cheap on subsequent startups once the stats are fresh.
        if not self._memory:
            async with self._engine.connect() as conn:
                await conn.execute(text("PRAGMA optimize"))

    async def _seed_system_tags(self) -> None:
        """Ensure system tags always exist (idempotent).

        _SYSTEM_TAGS are module-level objects shared across DbClient instances.
        After the first seed they become SQLAlchemy "detached" objects; re-using
        them in a second session would make SQLAlchemy treat them as already-
        persisted (no INSERT emitted).  Always construct fresh Tag instances.
        """
        async with self._async_session_maker() as session:
            for tmpl in _SYSTEM_TAGS:
                existing = await session.get(Tag, tmpl.uuid)
                if existing is None:
                    session.add(
                        Tag(
                            uuid=tmpl.uuid,
                            name=tmpl.name,
                            color=tmpl.color,
                            icon=tmpl.icon,
                            kind=tmpl.kind,
                            exclusive_group=tmpl.exclusive_group,
                            protected=tmpl.protected,
                        )
                    )
            await session.commit()

    async def _seed_system_datasets(self) -> None:
        """Ensure system datasets always exist (idempotent)."""
        async with self._async_session_maker() as session:
            existing = await session.get(Dataset, WHOLE_DATABASE_DATASET_UUID)
            if existing is None:
                session.add(
                    Dataset(
                        uuid=WHOLE_DATABASE_DATASET_UUID,
                        name="Whole Database",
                        description="All images in the database (no filters applied).",
                        filter_json=DatasetFilterCriteria().model_dump(mode="json"),
                        # Explicit: the table defaults to a View, and a system
                        # dataset that trainers cannot see is useless.
                        promoted=True,
                    )
                )
                await session.commit()

    async def _seed_system_labels(self) -> None:
        """Ensure default image-scope labels always exist (idempotent).

        Unlike _seed_system_tags/_seed_system_datasets, these are NOT looked up by a
        fixed UUID. bboxes/image_labels reference labels by UUID, and a label with a
        matching name may already exist (pre-created by a user, or from an older DB)
        with annotation rows already pointing at its UUID. Inventing a fixed UUID here
        and matching on it would create a duplicate label and orphan those rows.
        Instead, match on (name, scope) case-insensitively — case-insensitive so a
        pre-existing "corrupt" or "blurry" is recognized as the same label rather than
        producing a near-duplicate that differs only in case. If a match exists, it is
        left completely untouched (no UUID/color/type/parent changes). Only missing
        labels are inserted, with a freshly generated UUID.
        """
        async with self._async_session_maker() as session:
            statement = select(Label).where(Label.scope == LabelScope.IMAGE)
            result = await session.exec(statement)
            existing_names = {label.name.lower() for label in result.all()}

            for name in _DEFAULT_IMAGE_LABEL_NAMES:
                if name.lower() in existing_names:
                    continue
                session.add(
                    Label(
                        uuid=str(uuid4()),
                        name=name,
                        color=None,
                        parent_uuid=None,
                        label_type=LabelType.STANDARD,
                        scope=LabelScope.IMAGE,
                    )
                )
            await session.commit()

    async def get_dataset_image_uuids(self, dataset_uuids: list[str]) -> list[str]:
        """Return deduplicated image UUIDs from the union of the given datasets."""
        stubs = await self._enumerate_dataset_union_stubs(dataset_uuids)
        return [s.uuid for s in stubs]

    async def close(self):
        """Dispose of the engine connection pool."""
        await self._engine.dispose()

    ################################################################################
    # Images
    ################################################################################
    async def sync_image_from_json(self, image_path: Path, overwrite: bool = False) -> ImageRecord:
        """
        Syncs a single image's JSON sidecar into the DB.

        Args:
            image_path: Path to the image file (sidecar is image_path.with_suffix('.json')).
            overwrite: If True, replace the DB record's bboxes and metadata_json with
                       the JSON contents.  If False (default), merge: add only the JSON
                       bboxes whose UUIDs are not already in the DB, and leave metadata
                       unchanged if the DB already has any.

        Returns:
            The inserted or updated ImageRecord.

        Label and tag UUIDs are resolved UUID-first; if the UUID is not found in the
        DB, the stored name is used as a fallback (warns on mismatch).  Bboxes whose
        label cannot be resolved at all are skipped with a warning.
        """
        # --- Parse JSON sidecar ---
        json_path = image_path.with_suffix(".json")
        metadata_dict = {}
        json_uuid = None
        json_phash: Optional[str] = None
        json_source_uuid: Optional[str] = None
        json_bboxes: List[BBox] = []
        # Per-bbox supplementary data extracted before constructing BBox objects.
        box_tag_uuids: Dict[str, List[str]] = {}  # bbox_uuid → tag UUID list
        box_tag_names: Dict[str, List[str]] = {}  # bbox_uuid → tag name list (fallback)
        box_label_names: Dict[str, str] = {}  # bbox_uuid → label name (fallback)
        # Whole-image label assessments: (label_uuid, label_name, status) triples,
        # resolved to real label UUIDs further down alongside the bbox labels.
        json_image_labels: List[Tuple[Optional[str], Optional[str], ImageItemStatus]] = []
        sidecar_has_image_labels = False

        if json_path.exists():
            try:
                metadata_dict = jstyleson.loads(json_path.read_text())
                json_uuid = metadata_dict.get("uuid")
                json_phash = metadata_dict.get("phash")
                json_source_uuid = metadata_dict.get("sourceUuid")
                for box_data in metadata_dict.get("boxes", []):
                    raw = dict(box_data)
                    tag_uuids_list = raw.pop("tag_uuids", [])
                    tag_names_list = raw.pop("tag_names", [])
                    label_name = raw.pop("label_name", None)
                    bbox = BBox(**raw)
                    json_bboxes.append(bbox)
                    box_tag_uuids[bbox.uuid] = tag_uuids_list
                    box_tag_names[bbox.uuid] = tag_names_list
                    if label_name:
                        box_label_names[bbox.uuid] = label_name
                # Distinguish "sidecar predates this feature" (key absent) from
                # "user cleared every assessment" (key present but empty). Only
                # the latter may delete DB rows during an overwrite sync.
                sidecar_has_image_labels = "image_labels" in metadata_dict
                for row in metadata_dict.get("image_labels", []):
                    raw_status = str(row.get("status", "")).lower()
                    try:
                        status = ImageItemStatus(raw_status)
                    except ValueError:
                        logger.warning("Skipping image_label with unknown status %r in %s", raw_status, json_path)
                        continue
                    # UNCERTAIN is the absence of a row, so a sidecar carrying one
                    # has nothing to restore — skip rather than insert a row that
                    # every reader would have to special-case back out.
                    if status == ImageItemStatus.UNCERTAIN:
                        continue
                    json_image_labels.append((row.get("label_uuid"), row.get("label_name"), status))
            except Exception as e:
                logger.error(f"Failed to parse {json_path}: {e}")
                # A present-but-unparseable sidecar must never be treated as an
                # empty set of annotations. In overwrite mode that would replace
                # the DB's bboxes/metadata with nothing — silent data loss of the
                # user's annotations (the sidecar is often their last copy). Refuse.
                if overwrite:
                    raise SidecarParseError(
                        f"Refusing to overwrite DB from unparseable sidecar {json_path}: {e}"
                    ) from e
                # In merge mode a parse failure is non-destructive (nothing is
                # added, existing bboxes are kept), so fall through with empty state.
                json_bboxes = []
                box_tag_uuids = {}
                box_tag_names = {}
                box_label_names = {}
                json_image_labels = []
                sidecar_has_image_labels = False

        async with self._async_session_maker() as session:
            # A source can be deleted between the writer recording its UUID and this
            # sync running; provenance is not worth failing (or corrupting) an image
            # sync over, so an unresolvable UUID is treated the same as none at all.
            if json_source_uuid is not None:
                src = await session.get(Source, json_source_uuid)
                if src is None:
                    logger.warning(
                        "Sidecar %s references source %s, which no longer exists; leaving unattributed.",
                        json_path,
                        json_source_uuid,
                    )
                    json_source_uuid = None

            # --- Resolve label/tag UUIDs (UUID-first, name-fallback) ---
            all_label_uuids = {b.label_uuid for b in json_bboxes if b.label_uuid}
            all_label_uuids |= {u for u, _n, _s in json_image_labels if u}
            all_tag_uuids = {u for uuids in box_tag_uuids.values() for u in uuids}
            all_label_names = set(box_label_names.values())
            all_label_names |= {n for _u, n, _s in json_image_labels if n}
            all_tag_names = {n for names in box_tag_names.values() for n in names}

            labels_by_uuid: Dict[str, str] = {}
            labels_by_name: Dict[str, str] = {}
            if all_label_uuids or all_label_names:
                lbl_stmt = select(Label).where(or_(Label.uuid.in_(all_label_uuids), Label.name.in_(all_label_names)))
                for lbl in (await session.exec(lbl_stmt)).all():
                    labels_by_uuid[lbl.uuid] = lbl.uuid
                    labels_by_name[lbl.name] = lbl.uuid

            tags_by_uuid: Dict[str, str] = {}
            tags_by_name: Dict[str, str] = {}
            tags_obj_by_uuid: Dict[str, "Tag"] = {}  # full objects for exclusive-group enforcement
            if all_tag_uuids or all_tag_names:
                tag_stmt = select(Tag).where(or_(Tag.uuid.in_(all_tag_uuids), Tag.name.in_(all_tag_names)))
                for tag in (await session.exec(tag_stmt)).all():
                    tags_by_uuid[tag.uuid] = tag.uuid
                    tags_by_name[tag.name] = tag.uuid
                    tags_obj_by_uuid[tag.uuid] = tag

            # Resolve each bbox's label_uuid; skip bboxes with no resolvable label.
            invalid_bbox_uuids: set = set()
            for bbox in json_bboxes:
                if bbox.label_uuid and bbox.label_uuid in labels_by_uuid:
                    pass  # UUID valid as-is
                else:
                    fallback_name = box_label_names.get(bbox.uuid)
                    if fallback_name and fallback_name in labels_by_name:
                        resolved = labels_by_name[fallback_name]
                        logger.warning(
                            "BBox %s: label_uuid %s not in DB; resolved by name '%s' → %s",
                            bbox.uuid,
                            bbox.label_uuid,
                            fallback_name,
                            resolved,
                        )
                        bbox.label_uuid = resolved
                    else:
                        logger.warning(
                            "BBox %s: label_uuid %s unresolvable (name fallback '%s' also failed); skipping.",
                            bbox.uuid,
                            bbox.label_uuid,
                            fallback_name,
                        )
                        invalid_bbox_uuids.add(bbox.uuid)

            # Resolve each bbox's tag UUIDs; drop individual unresolvable tags.
            for bbox in json_bboxes:
                if bbox.uuid in invalid_bbox_uuids:
                    continue
                raw_uuids = box_tag_uuids.get(bbox.uuid, [])
                raw_names = box_tag_names.get(bbox.uuid, [])
                uuid_to_name = dict(zip(raw_uuids, raw_names))
                resolved: List[str] = []
                for tag_uuid in raw_uuids:
                    if tag_uuid in tags_by_uuid:
                        resolved.append(tag_uuid)
                    else:
                        name = uuid_to_name.get(tag_uuid)
                        if name and name in tags_by_name:
                            remapped = tags_by_name[name]
                            logger.warning(
                                "BBox %s: tag_uuid %s not in DB; resolved by name '%s' → %s",
                                bbox.uuid,
                                tag_uuid,
                                name,
                                remapped,
                            )
                            resolved.append(remapped)
                        else:
                            logger.warning(
                                "BBox %s: tag_uuid %s unresolvable (name '%s' also failed); dropping tag.",
                                bbox.uuid,
                                tag_uuid,
                                name,
                            )
                box_tag_uuids[bbox.uuid] = resolved

            # Filter out bboxes with unresolvable labels.
            json_bboxes = [b for b in json_bboxes if b.uuid not in invalid_bbox_uuids]

            # --- Look up any existing image records ---
            path_stmt = (
                select(ImageRecord)
                .where(ImageRecord.filename == str(image_path))
                .options(selectinload(ImageRecord.bboxes))
            )
            existing_by_path = (await session.exec(path_stmt)).one_or_none()

            existing_by_uuid = None
            if json_uuid:
                uuid_stmt = (
                    select(ImageRecord).where(ImageRecord.uuid == json_uuid).options(selectinload(ImageRecord.bboxes))
                )
                existing_by_uuid = (await session.exec(uuid_stmt)).one_or_none()

            # --- Warn about conflicts ---
            if existing_by_path and json_uuid and existing_by_path.uuid != json_uuid:
                logger.warning(
                    "Conflict: Path '%s' exists in DB with UUID %s, " "but JSON sidecar specifies UUID %s.",
                    image_path,
                    existing_by_path.uuid,
                    json_uuid,
                )

            if existing_by_uuid and existing_by_uuid.filename != str(image_path):
                logger.warning(
                    "Conflict: UUID %s already belongs to file '%s'. "
                    "The sidecar at '%s' is claiming a UUID used elsewhere.",
                    json_uuid,
                    existing_by_uuid.filename,
                    image_path,
                )

            # --- Insert or update ---
            db_record = existing_by_path or existing_by_uuid
            bboxes_needing_links: List[str] = []

            if db_record is None:
                db_record = ImageRecord(
                    uuid=json_uuid or str(uuid4()),
                    filename=str(image_path),
                    bboxes=json_bboxes,
                    metadata_json=metadata_dict,
                    phash=json_phash,
                    source_uuid=json_source_uuid,
                )
                session.add(db_record)
                bboxes_needing_links = [b.uuid for b in json_bboxes if box_tag_uuids.get(b.uuid)]
            elif overwrite:
                # Old BBoxTagLink rows are cascade-deleted when the old BBox rows are removed.
                #
                # Drop the old rows and flush *before* attaching the sidecar's,
                # so a bbox whose UUID appears in both is INSERTed fresh rather
                # than UPDATEd in place. An UPDATE spans every column including
                # `confidence`, which is GENERATED — SQLite rejects the write and
                # the whole restore fails. That is the common case, not an edge
                # one: re-syncing a sidecar over a DB that still holds the same
                # bboxes is exactly what reconcile does.
                db_record.bboxes = []
                await session.flush()
                db_record.bboxes = json_bboxes
                db_record.metadata_json = metadata_dict
                if json_phash:
                    db_record.phash = json_phash
                if json_source_uuid and db_record.source_uuid is None:
                    db_record.source_uuid = json_source_uuid
                session.add(db_record)
                bboxes_needing_links = [b.uuid for b in json_bboxes if box_tag_uuids.get(b.uuid)]
            else:
                existing_uuids = {b.uuid for b in db_record.bboxes}
                new_bboxes = [b for b in json_bboxes if b.uuid not in existing_uuids]
                db_record.bboxes = db_record.bboxes + new_bboxes
                if not db_record.metadata_json:
                    db_record.metadata_json = metadata_dict
                if json_source_uuid and db_record.source_uuid is None:
                    db_record.source_uuid = json_source_uuid
                session.add(db_record)
                bboxes_needing_links = [b.uuid for b in new_bboxes if box_tag_uuids.get(b.uuid)]

            # Flush so BBox PKs are in DB before BBoxTagLink FK references them.
            await session.flush()

            for bbox_uuid in bboxes_needing_links:
                raw_tags = box_tag_uuids.get(bbox_uuid, [])
                # _resolve_exclusive_groups enforces mutual-exclusion constraints and
                # deduplicates the list, preventing composite-PK IntegrityErrors.
                for tag_uuid in self._resolve_exclusive_groups(raw_tags, tags_obj_by_uuid):
                    session.add(BBoxTagLink(bbox_uuid=bbox_uuid, tag_uuid=tag_uuid))

            # --- Restore whole-image label assessments ---
            # These are the only annotations a user can make that leave no trace
            # in `boxes`: an ABSENT row means "I looked, and this label is not
            # here". Nothing else in the sidecar can reconstruct them.
            if sidecar_has_image_labels:
                resolved_statuses: Dict[str, ImageItemStatus] = {}
                for lbl_uuid, lbl_name, status in json_image_labels:
                    if lbl_uuid and lbl_uuid in labels_by_uuid:
                        resolved = lbl_uuid
                    elif lbl_name and lbl_name in labels_by_name:
                        resolved = labels_by_name[lbl_name]
                        logger.warning(
                            "image_label for '%s': uuid %s not found, resolved by name to %s.",
                            lbl_name,
                            lbl_uuid,
                            resolved,
                        )
                    else:
                        logger.warning(
                            "image_label (uuid=%s, name=%s) unresolvable in %s; dropping.",
                            lbl_uuid,
                            lbl_name,
                            json_path,
                        )
                        continue
                    resolved_statuses[resolved] = status

                # no_autoflush: the pending BBox state is mid-write here, and
                # letting this unrelated SELECT trigger a flush makes SQLAlchemy
                # emit an UPDATE that includes `confidence` — a generated column
                # SQLite refuses to update. The bboxes flush on commit as before.
                with session.no_autoflush:
                    existing_rows = (
                        await session.exec(select(ImageLabel).where(ImageLabel.image_uuid == db_record.uuid))
                    ).all()
                existing_by_label = {r.label_uuid: r for r in existing_rows}
                now = datetime.now(timezone.utc)

                for label_uuid, status in resolved_statuses.items():
                    row = existing_by_label.get(label_uuid)
                    if row is None:
                        session.add(
                            ImageLabel(
                                image_uuid=db_record.uuid,
                                label_uuid=label_uuid,
                                status=status,
                                updated_at=now,
                            )
                        )
                    elif overwrite:
                        row.status = status
                        row.updated_at = now
                        session.add(row)
                    # else: merge mode leaves the DB's own assessment alone.

                if overwrite:
                    # Symmetry with the bbox overwrite branch: the sidecar is the
                    # authority, so an assessment it no longer lists is one the
                    # user cleared. Guarded by sidecar_has_image_labels above so a
                    # pre-feature sidecar can never trigger this.
                    for label_uuid, row in existing_by_label.items():
                        if label_uuid not in resolved_statuses:
                            await session.delete(row)

            try:
                await session.commit()
            except Exception:
                logger.exception("Commit failed while writing image metadata")
                raise

            await session.refresh(db_record)
            return db_record

    async def read_image_metadata_from_json(self, json_path: Path) -> Optional[ImageRecordRead]:
        """Reads image metadata from a side JSON file"""
        if not json_path.exists():
            return None

        try:
            data = jstyleson.loads(json_path.read_text())
            return ImageRecordRead.model_validate(data)
        except Exception as e:
            logger.error(f"Error parsing sidefile {json_path}: {e}")
            return None

    async def get_image_by_uuid(self, uuid: str) -> Optional[ImageRecord]:
        """Retrieve a full Image object by its UUID."""
        async with self._async_session_maker() as session:
            stmt = (
                select(ImageRecord)
                .where(ImageRecord.uuid == uuid)
                .options(
                    selectinload(ImageRecord.bboxes).options(
                        selectinload(BBox.label),
                        selectinload(BBox.tags),
                    ),
                    selectinload(ImageRecord.tags),
                )
            )
            result = await session.exec(stmt)
            return result.one_or_none()

    async def get_image_uuid_by_filename(self, filename: Path) -> Optional[str]:
        """Return only the UUID for an image by filename.  Cheaper than a full load."""
        async with self._async_session_maker() as session:
            stmt = select(ImageRecord.uuid).where(ImageRecord.filename == str(filename))
            result = await session.exec(stmt)
            return result.one_or_none()

    async def get_image_by_filename(self, filename: Path) -> Optional[ImageRecord]:
        """Retrieve the full Image object by its filename."""
        async with self._async_session_maker() as session:
            statement = (
                select(ImageRecord)
                .where(ImageRecord.filename == str(filename))
                .options(
                    selectinload(ImageRecord.bboxes).options(
                        selectinload(BBox.label),
                        selectinload(BBox.tags),
                    ),
                    selectinload(ImageRecord.tags),
                )
            )
            result = await session.exec(statement)
            return result.one_or_none()

    async def get_filenames_in_db(self, filenames: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping of absolute filename → {uuid, bbox_count} for each filename that
        exists in the images table.  Filenames not in the DB are absent from the result.
        """
        if not filenames:
            return {}
        async with self._async_session_maker() as session:
            statement = (
                select(
                    ImageRecord.filename,
                    ImageRecord.uuid,
                    func.count(BBox.uuid).label("bbox_count"),
                )
                .outerjoin(BBox, BBox.image_uuid == ImageRecord.uuid)
                .where(ImageRecord.filename.in_(filenames))
                .group_by(ImageRecord.filename, ImageRecord.uuid)
            )
            result = await session.exec(statement)
            return {row[0]: {"uuid": row[1], "bbox_count": row[2]} for row in result.all()}

    async def add_image(
        self,
        filename: Path,
        metadata_json: Optional[Dict[str, Any]] = None,
        source_uuid: Optional[str] = None,
    ) -> ImageRecord:
        """
        Add a new image record or return existing object if filename exists.

        `source_uuid` records which source produced the image. It only ever
        *fills in* a blank: every annotation save re-enters through here with no
        source, so overwriting would erase the provenance on the first edit.
        Backfilling an existing blank row is intentional though — it is how an
        image registered by a plain path save (which runs before the collecting
        task reports its source) acquires one.

        Returns:
            Image: The fully populated Image SQLModel instance.
        """
        async with self._async_session_maker() as session:
            # Check for existing image first (fast path)
            statement = select(ImageRecord).where(ImageRecord.filename == str(filename))
            result = await session.exec(statement)
            existing_image = result.one_or_none()

            if existing_image:
                if source_uuid and existing_image.source_uuid is None:
                    existing_image.source_uuid = source_uuid
                    session.add(existing_image)
                    try:
                        await session.commit()
                        await session.refresh(existing_image)
                    except IntegrityError:
                        # The source was deleted between the caller reading it and
                        # this write. Provenance is not worth failing an image save
                        # over — see the same trade-off on the INSERT path below.
                        logger.warning("Source %s no longer exists; leaving %s unattributed", source_uuid, filename)
                        await session.rollback()
                        result = await session.exec(statement)
                        return result.one()
                return existing_image

            # Create new record; catch the race where two concurrent callers
            # both pass the check above before either INSERT commits.
            new_image = ImageRecord(
                uuid=str(uuid4()),
                filename=str(filename),
                metadata_json=metadata_json or {},
                source_uuid=source_uuid,
            )
            session.add(new_image)
            try:
                await session.commit()
                await session.refresh(new_image)
                return new_image
            except IntegrityError:
                await session.rollback()
                result = await session.exec(statement)
                existing_image = result.one_or_none()
                if existing_image is not None:
                    return existing_image
                # Not the filename race, so it was the source FK: the source row
                # went away mid-run. Losing the image because we can no longer say
                # which camera took it would be far worse than losing the
                # attribution, so register it unattributed.
                if source_uuid is None:
                    raise
                logger.warning("Source %s no longer exists; registering %s unattributed", source_uuid, filename)
                new_image = ImageRecord(
                    uuid=str(uuid4()),
                    filename=str(filename),
                    metadata_json=metadata_json or {},
                )
                session.add(new_image)
                await session.commit()
                await session.refresh(new_image)
                return new_image

    async def list_images(self, limit: int = 100, offset: int = 0) -> List[ImageRecord]:
        """List images with basic pagination."""
        async with self._async_session_maker() as session:
            statement = select(ImageRecord).offset(offset).limit(limit)
            result = await session.exec(statement)
            return result.all()

    async def _selectivity_count(self, key: str, count_stmt: Select) -> int:
        """Run (or return the cached result of) a cheap indexed COUNT probe."""
        now = time.monotonic()
        cached = self._selectivity_cache.get(key)
        if cached is not None and (now - cached[1]) < _SELECTIVITY_CACHE_TTL:
            return cached[0]
        async with self._async_session_maker() as session:
            value = (await session.exec(count_stmt)).one()
        now = time.monotonic()
        self._selectivity_cache[key] = (value, now)
        # Keys are derived from the filter predicates, so a browsing session that keeps
        # varying its filters leaves an entry behind for each combination; expired ones
        # are never otherwise removed. Sweep once past a soft ceiling.
        if len(self._selectivity_cache) > _SELECTIVITY_CACHE_MAX_ENTRIES:
            stale_keys = [k for k, (_, ts) in self._selectivity_cache.items() if now - ts >= _SELECTIVITY_CACHE_TTL]
            for stale_key in stale_keys:
                del self._selectivity_cache[stale_key]
            if len(self._selectivity_cache) > _SELECTIVITY_CACHE_MAX_ENTRIES:
                # Everything is still live (a burst of distinct filters inside one TTL
                # window). Drop the oldest to keep the ceiling meaningful.
                excess = len(self._selectivity_cache) - _SELECTIVITY_CACHE_MAX_ENTRIES
                oldest_keys = sorted(self._selectivity_cache, key=lambda k: self._selectivity_cache[k][1])[:excess]
                for stale_key in oldest_keys:
                    del self._selectivity_cache[stale_key]
        return value

    async def _should_use_exists_dispatch(self, params: ImageSearchParams) -> bool:
        """Decide IN vs correlated-EXISTS for the bbox tag/label search filters.

        The EXISTS shape only wins when SQLite can scan the (sort_col, uuid)
        index and stop after LIMIT rows pass the filter — which requires the
        CONJUNCTION of every restrictive predicate to be common. A single rare
        (or unestimatable) predicate turns the ordered scan into a full-table
        scan, so this is deliberately all-or-nothing:

        - Bail out on any term we cannot cheaply estimate, and on any negated
          term (a negated common predicate is not a common predicate).
        - Probe each remaining term with an indexed COUNT (cached, see
          _selectivity_count) and require *every* term's estimated
          matched-image fraction to clear the threshold.

        Only three term kinds have an EXISTS branch at all
        (`_apply_filter_terms`'s `_tag_predicate`/`_label_predicate`:
        objectLabel, bboxTag, bboxVerification; bboxConfidence reuses
        `_label_predicate` and so has one too, but is never probed) — but the
        bail-out list is
        wider than that on purpose. imageLabel/imageTag/labelVerification are
        already correlated EXISTS regardless of this flag, and date/filename
        are plain column predicates; none of them changes shape here, yet a
        rare one still destroys the ordered scan the flag exists to enable. So
        their mere presence keeps the IN shape.

        The probes count bbox/link rows rather than distinct images (a COUNT on
        a covering index, ~1ms, vs an expensive DISTINCT join), which
        overestimates the fraction — the safe error direction: wrongly picking
        EXISTS near the threshold costs 10s of ms, while a zero/near-zero count
        always keeps IN and avoids the catastrophic full-scan case.

        Restores the dispatch PR 1.4b had to disable when the flat fields it
        probed were deleted (follow-up C). `test_exists_in_parity.py` proves
        the two shapes return identical result sets; this only picks between
        them, so a wrong answer is slow, never incorrect.
        """
        estimates: List[int] = []

        for term in params.terms:
            # Negation inverts selectivity: "not the common tag" is a
            # different, un-probed predicate, so don't reason about it.
            if getattr(term, "negate", False):
                return False

            if isinstance(term, BBoxTagTerm):
                if not term.tag_uuids:
                    # any → matches nothing; all → matches everything. Neither
                    # is worth probing, and guessing wrong is the full-scan case.
                    return False
                counts = await asyncio.gather(
                    *(
                        self._selectivity_count(
                            f"tag:{tag_uuid}",
                            select(func.count()).select_from(BBoxTagLink).where(BBoxTagLink.tag_uuid == tag_uuid),
                        )
                        for tag_uuid in term.tag_uuids
                    )
                )
                # "any" matches the union (bounded below by the biggest member);
                # "all" matches at most the smallest.
                estimates.append(max(counts) if term.match_mode == "any" else min(counts))

            elif isinstance(term, BBoxVerificationTerm):
                if term.state == "no_boxes":
                    # A NOT EXISTS over bboxes has no COUNT probe that means
                    # anything — counting bbox rows says nothing about how many
                    # images have none. Fall back to the non-EXISTS dispatch.
                    return False
                tag_uuid = BBOX_VERIFICATION_STATE_TO_TAG_UUID[term.state]
                estimates.append(
                    await self._selectivity_count(
                        f"tag:{tag_uuid}",
                        select(func.count()).select_from(BBoxTagLink).where(BBoxTagLink.tag_uuid == tag_uuid),
                    )
                )

            elif isinstance(term, ObjectLabelTerm):
                if not term.label_uuids:
                    return False
                if term.match_mode == "all":
                    counts = await asyncio.gather(
                        *(self._object_label_count([lu], term.roll_up) for lu in term.label_uuids)
                    )
                    estimates.append(min(counts))
                else:
                    estimates.append(await self._object_label_count(term.label_uuids, term.roll_up))

            else:
                # imageLabel / imageTag / labelVerification / date / filename /
                # bboxConfidence — no cheap COUNT probe for these (estimating a
                # confidence band would itself need the full json_extract scan
                # the probe exists to avoid). See the docstring.
                return False

        if not estimates or min(estimates) == 0:
            return False
        total_images = await self._selectivity_count("images:total", select(func.count()).select_from(ImageRecord))
        if total_images == 0:
            return False
        return (min(estimates) / total_images) >= _EXISTS_DISPATCH_MIN_FRACTION

    async def _object_label_count(self, label_uuids: List[str], roll_up: bool) -> int:
        """Cached COUNT of bboxes carrying any of *label_uuids*.

        Must expand descendants exactly when the query does: a parent label
        with no bboxes of its own is rare on its own and common rolled up, and
        probing the unexpanded set would keep IN for the most common filter in
        the app.
        """
        key = "label:" + ",".join(sorted(label_uuids)) + (":tree" if roll_up else "")
        if roll_up:
            condition = BBox.label_uuid.in_(_label_uuids_with_descendants(label_uuids, cte_name="label_desc_probe"))
        elif len(label_uuids) == 1:
            condition = BBox.label_uuid == label_uuids[0]
        else:
            condition = BBox.label_uuid.in_(label_uuids)
        return await self._selectivity_count(key, select(func.count()).select_from(BBox).where(condition))

    async def list_images_search(self, params: ImageSearchParams) -> List[ImageRecord]:
        """Flexible image search used by the review queue.

        Supports tag filtering (OR across selected tags), label filtering, filename
        substring search, date range, image-label status filters, and stable keyset
        pagination for both UUID and datetime sort orders.
        """
        use_exists = await self._should_use_exists_dispatch(params)
        async with self._async_session_maker() as session:
            stmt = select(ImageRecord).options(
                selectinload(ImageRecord.bboxes).options(
                    selectinload(BBox.label),
                    selectinload(BBox.tags),
                ),
                selectinload(ImageRecord.tags),
            )

            stmt = _apply_image_search_filters(stmt, params, use_exists=use_exists, files_root=self._files_root)

            # ── Sort column ────────────────────────────────────────────────────
            # NOTE: filename is intentionally NOT a sort option — the keyset
            # cursor (after_sort_value) is a datetime and cannot carry a filename,
            # so filename-sorted pagination would silently break past page 1.
            sort_col_map = {
                "created_at": ImageRecord.created_at,
                "updated_at": ImageRecord.updated_at,
                "uuid": ImageRecord.uuid,
            }
            sort_col = sort_col_map[params.sort_by]

            # ── Keyset pagination ──────────────────────────────────────────────
            # For UUID sort, a simple scalar comparison is sufficient (UUIDs are unique).
            # For datetime sorts, use a composite (sort_field, uuid) keyset to handle
            # ties deterministically without gaps or duplicates.
            if params.after_uuid is not None:
                sv = params.after_sort_value
                if params.sort_by != "uuid" and sv is None:
                    # A uuid-only cursor cannot drive a (sort_col, uuid) keyset, and
                    # falling back to a bare uuid comparison is wrong here for the same
                    # reason it is in `_keyset_start_index`: uuids are unordered with
                    # respect to created_at/updated_at, so the page barely advances and
                    # a client paging to exhaustion can loop forever. Recover the sort
                    # value from the cursor row itself (PK lookup) instead.
                    sv = await session.scalar(select(sort_col).where(ImageRecord.uuid == params.after_uuid))

                if params.sort_by == "uuid":
                    if params.sort_order == "asc":
                        stmt = stmt.where(ImageRecord.uuid > params.after_uuid)
                    else:
                        stmt = stmt.where(ImageRecord.uuid < params.after_uuid)
                elif sv is None:
                    # The cursor row is gone; there is no sound resume point, so return
                    # an empty page rather than restarting from the top.
                    return []
                else:
                    if params.sort_order == "asc":
                        stmt = stmt.where(
                            or_(
                                sort_col > sv,
                                and_(sort_col == sv, ImageRecord.uuid > params.after_uuid),
                            )
                        )
                    else:
                        stmt = stmt.where(
                            or_(
                                sort_col < sv,
                                and_(sort_col == sv, ImageRecord.uuid < params.after_uuid),
                            )
                        )

            # ── Order + limit ──────────────────────────────────────────────────
            if params.sort_order == "asc":
                stmt = stmt.order_by(sort_col.asc(), ImageRecord.uuid.asc())
            else:
                stmt = stmt.order_by(sort_col.desc(), ImageRecord.uuid.desc())
            stmt = stmt.limit(params.limit)

            result = await session.exec(stmt)
            return result.all()

    async def count_images_search(self, params: ImageSearchParams) -> int:
        """Return the total count of images matching `params`, ignoring pagination.

        Applies the same filter predicates as list_images_search but without ORDER BY,
        LIMIT, or keyset cursors so the result reflects the full matched set.

        Always uses the IN (subquery) filter shape: a count visits every
        matching row, so the EXISTS dispatch's early-termination advantage
        does not apply and IN's one-time materialisation is cheapest.
        """
        async with self._async_session_maker() as session:
            inner = _apply_image_search_filters(select(ImageRecord.uuid), params, files_root=self._files_root)
            stmt = select(func.count()).select_from(inner.subquery())
            result = await session.exec(stmt)
            return result.one()

    async def count_bboxes_search(
        self,
        params: ImageSearchParams,
        bbox_tag_filter: Optional[List[str]] = None,
        bbox_label_filter: Optional[List[str]] = None,
        show_all_bboxes: bool = False,
        bbox_verification_tag_uuid: Optional[str] = None,
        bbox_confidence: Optional[BBoxConfidenceRange] = None,
    ) -> int:
        """Count the bboxes that bulk_apply_bboxes would touch for this filter.

        The image-level predicates select the same images as count_images_search;
        the bbox-level arguments then narrow to the subset the review grid shows,
        mirroring bulk_apply_bboxes's per-bbox filter exactly (an image count is
        NOT a usable stand-in — images carry many bboxes, and the subset filter
        can drop most of them).
        """
        async with self._async_session_maker() as session:
            image_uuids = _apply_image_search_filters(
                select(ImageRecord.uuid), params, files_root=self._files_root
            ).subquery()
            stmt = select(func.count()).select_from(BBox).where(BBox.image_uuid.in_(select(image_uuids.c.uuid)))

            tag_filter = None if show_all_bboxes else (bbox_tag_filter or None)
            if tag_filter:
                stmt = stmt.where(
                    BBox.uuid.in_(select(BBoxTagLink.bbox_uuid).where(BBoxTagLink.tag_uuid.in_(tag_filter)))
                )
            if bbox_label_filter:
                stmt = stmt.where(BBox.label_uuid.in_(bbox_label_filter))
            if bbox_verification_tag_uuid is not None:
                # ANDed rather than folded into tag_filter — that list is OR
                # semantics, so merging would widen the count, not narrow it.
                stmt = stmt.where(
                    BBox.uuid.in_(
                        select(BBoxTagLink.bbox_uuid).where(BBoxTagLink.tag_uuid == bbox_verification_tag_uuid)
                    )
                )
            if bbox_confidence is not None and bbox_confidence.is_active():
                stmt = stmt.where(
                    _bbox_confidence_condition(
                        bbox_confidence.min, bbox_confidence.max, bbox_confidence.include_missing
                    )
                )

            result = await session.exec(stmt)
            return result.one()

    async def get_bbox_with_image(self, bbox_uuid: str) -> Optional[tuple[BBox, ImageRecord]]:
        """Return a bbox (with tags and label eagerly loaded) and its parent image, or None."""
        async with self._async_session_maker() as session:
            stmt = select(BBox).where(BBox.uuid == bbox_uuid).options(selectinload(BBox.tags), selectinload(BBox.label))
            result = await session.exec(stmt)
            bbox = result.first()
            if bbox is None:
                return None
            image = await session.get(ImageRecord, bbox.image_uuid)
            return bbox, image

    async def get_image_uuids_for_bboxes(self, bbox_uuids: List[str]) -> Dict[str, str]:
        """Return {bbox_uuid: image_uuid} for a batch of bbox UUIDs."""
        if not bbox_uuids:
            return {}
        async with self._async_session_maker() as session:
            stmt = select(BBox.uuid, BBox.image_uuid).where(BBox.uuid.in_(bbox_uuids))
            result = await session.exec(stmt)
            return {b_uuid: img_uuid for b_uuid, img_uuid in result.all()}

    async def get_image_uuids_by_bbox_tag(self, tag_uuid: str) -> Set[str]:
        """Return all image UUIDs that have at least one bbox carrying the given tag."""
        async with self._async_session_maker() as session:
            stmt = (
                select(BBox.image_uuid)
                .join(BBoxTagLink, BBoxTagLink.bbox_uuid == BBox.uuid)
                .where(BBoxTagLink.tag_uuid == tag_uuid)
                .distinct()
            )
            result = await session.exec(stmt)
            return set(result.all())

    async def filter_bboxes_by_tags(
        self,
        bbox_uuids: List[str],
        require_tag_uuids: List[str],
        exclude_tag_uuids: List[str],
        tag_match_mode: Literal["any", "all"] = "any",
    ) -> Set[str]:
        """Return the subset of bbox_uuids that pass the tag filter.

        require_tag_uuids + tag_match_mode control which bboxes are *included*:
          "any" → bbox must have at least one of the required tags (OR logic)
          "all" → bbox must have every required tag (AND logic)

        exclude_tag_uuids removes bboxes that carry any of those tags regardless
        of the include result.
        """
        if not bbox_uuids:
            return set()
        if not require_tag_uuids and not exclude_tag_uuids:
            return set(bbox_uuids)

        valid: Set[str] = set(bbox_uuids)

        async with self._async_session_maker() as session:
            if require_tag_uuids:
                if tag_match_mode == "all":
                    # Bbox must carry EACH required tag; intersect per-tag.
                    for tag_uuid in require_tag_uuids:
                        stmt = select(BBoxTagLink.bbox_uuid).where(
                            BBoxTagLink.bbox_uuid.in_(valid),
                            BBoxTagLink.tag_uuid == tag_uuid,
                        )
                        result = await session.exec(stmt)
                        valid &= set(result.all())
                        if not valid:
                            return set()
                else:
                    # "any" — bbox has at least one of the required tags.
                    stmt = (
                        select(BBoxTagLink.bbox_uuid)
                        .where(
                            BBoxTagLink.bbox_uuid.in_(valid),
                            BBoxTagLink.tag_uuid.in_(require_tag_uuids),
                        )
                        .distinct()
                    )
                    result = await session.exec(stmt)
                    valid &= set(result.all())

            if exclude_tag_uuids and valid:
                stmt = (
                    select(BBoxTagLink.bbox_uuid)
                    .where(
                        BBoxTagLink.bbox_uuid.in_(valid),
                        BBoxTagLink.tag_uuid.in_(exclude_tag_uuids),
                    )
                    .distinct()
                )
                result = await session.exec(stmt)
                valid -= set(result.all())

        return valid

    async def get_images_by_uuids(self, uuids: List[str]) -> List[ImageRecord]:
        """Return full ImageRecord rows (with bboxes + tags) for a list of image UUIDs."""
        if not uuids:
            return []
        async with self._async_session_maker() as session:
            stmt = (
                select(ImageRecord)
                .where(ImageRecord.uuid.in_(uuids))
                .options(
                    selectinload(ImageRecord.bboxes).options(
                        selectinload(BBox.label),
                        selectinload(BBox.tags),
                    ),
                    selectinload(ImageRecord.tags),
                )
            )
            result = await session.exec(stmt)
            return result.all()

    async def update_single_bbox(
        self,
        bbox_uuid: str,
        label_uuid: Optional[str],
        tag_uuids: List[str],
        metadata_json: Optional[Dict[str, Any]] = None,
        geometry: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update one bbox's label and replace its tag links in place.

        Exclusive-group constraints are enforced: if multiple tags in tag_uuids share
        an exclusive_group, only the *last* one in the list is kept.  Callers can
        therefore pass [*existing_tags, *new_tags] and the new tags will win any
        exclusive-group conflict.

        `metadata_json`, if given (not None), fully replaces the bbox's metadata_json.
        `geometry`, if given (not None), must have "x"/"y"/"width"/"height" keys and
        replaces the bbox's coordinates (e.g. a refinement model's suggested box).
        """
        async with self._async_session_maker() as session:
            bbox = await session.get(BBox, bbox_uuid)
            if bbox is None:
                return
            if label_uuid is not None:
                bbox.label_uuid = label_uuid
            if metadata_json is not None:
                bbox.metadata_json = metadata_json
            if geometry is not None:
                bbox.x = geometry["x"]
                bbox.y = geometry["y"]
                bbox.width = geometry["width"]
                bbox.height = geometry["height"]

            # Resolve exclusive-group conflicts before writing.
            if tag_uuids:
                tag_rows = await session.exec(select(Tag).where(Tag.uuid.in_(tag_uuids)))
                tags_by_uuid = {t.uuid: t for t in tag_rows.all()}
            else:
                tags_by_uuid = {}
            final_uuids = self._resolve_exclusive_groups(tag_uuids, tags_by_uuid)

            # After the tag list is final, and after any metadata replacement
            # above: a box that ends this write Verified keeps no confidence,
            # whether it was verified by this call or already was.
            bbox.metadata_json = self._apply_verified_confidence_rule(bbox.metadata_json, final_uuids)

            await session.exec(delete(BBoxTagLink).where(BBoxTagLink.bbox_uuid == bbox_uuid))
            for tag_uuid in final_uuids:
                session.add(BBoxTagLink(bbox_uuid=bbox_uuid, tag_uuid=tag_uuid))
            session.add(bbox)
            # Touch the parent image so updated_at reflects bbox-level changes.
            await self._touch_image(session, bbox.image_uuid)
            await session.commit()

    @staticmethod
    async def _touch_image(session: AsyncSession, image_uuid: str) -> None:
        """Set updated_at on an image to now, so bbox-level changes are reflected in sorts."""
        image = await session.get(ImageRecord, image_uuid)
        if image is not None:
            image.updated_at = datetime.now(timezone.utc)
            session.add(image)

    @staticmethod
    def _resolve_exclusive_groups(tag_uuids: List[str], tags_by_uuid: dict) -> List[str]:
        """Drop earlier members of an exclusive_group so only the last one survives.

        Tags with no exclusive_group are kept as-is.  The order of the remaining
        tags in the output matches their order in the input.
        """
        # First pass: find which uuid wins each exclusive group (last occurrence wins).
        exclusive_winner: dict[str, str] = {}
        for uuid in tag_uuids:
            tag = tags_by_uuid.get(uuid)
            if tag and tag.exclusive_group:
                exclusive_winner[tag.exclusive_group] = uuid

        # Second pass: keep a tag if it has no exclusive_group, or if it is the winner.
        seen: set[str] = set()
        resolved: List[str] = []
        for uuid in tag_uuids:
            if uuid in seen:
                continue
            seen.add(uuid)
            tag = tags_by_uuid.get(uuid)
            if tag and tag.exclusive_group and exclusive_winner.get(tag.exclusive_group) != uuid:
                continue  # lost the exclusive-group contest
            resolved.append(uuid)
        return resolved

    @staticmethod
    def _apply_verified_confidence_rule(
        metadata_json: Optional[Dict[str, Any]],
        final_tag_uuids: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        """Enforce "a verified bbox stores no model confidence" on one bbox's metadata.

        Returns the metadata to persist: the input unchanged when the box is not
        verified (or carries no confidence anyway), a new dict without the
        confidence keys when it is. Returning a NEW dict matters — SQLModel's
        JSON column only detects a reassignment, not an in-place mutation.

        Called at every point that writes a bbox's tags, since the trigger is a
        tag change as often as a metadata change: marking an existing inferred
        box Verified must drop the confidence it was created with, not just
        hide it. `BBox.confidence` is a generated column over these same keys,
        so removing them is what actually takes the box out of the confidence
        band filter and off the cards.

        NOTE: this is a one-way door by design. Un-verifying a box later does
        not bring the old confidence back — it is gone from the row.
        """
        if VERIFIED_TAG_UUID not in final_tag_uuids:
            return metadata_json
        if not any(key in (metadata_json or {}) for key in BBOX_CONFIDENCE_METADATA_KEYS):
            return metadata_json
        return strip_bbox_confidence(metadata_json)

    async def add_single_bbox(
        self,
        image_uuid: str,
        bbox_uuid: str,
        label_uuid: Optional[str],
        x: float,
        y: float,
        width: float,
        height: float,
        tag_uuids: List[str],
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert one new bbox into an image *without* touching its siblings.

        Unlike write_image_metadata_to_db (a full DELETE+INSERT replace that
        re-writes and re-phashes every existing bbox on the image), this is a
        targeted single-row insert — the right primitive for programmatic
        add-a-box callers such as the filter-chain add_bbox action. Exclusive-group
        conflicts among tag_uuids are resolved the same way update_single_bbox does.
        """
        async with self._async_session_maker() as session:
            final_uuids: List[str] = []
            if tag_uuids:
                tag_rows = await session.exec(select(Tag).where(Tag.uuid.in_(tag_uuids)))
                tags_by_uuid = {t.uuid: t for t in tag_rows.all()}
                final_uuids = self._resolve_exclusive_groups(tag_uuids, tags_by_uuid)
            # Resolve the tags BEFORE building the row: a box inserted already
            # Verified must not carry a confidence in the first place.
            session.add(
                BBox(
                    uuid=bbox_uuid,
                    image_uuid=image_uuid,
                    label_uuid=label_uuid,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    metadata_json=self._apply_verified_confidence_rule(metadata_json, final_uuids) or {},
                )
            )
            for tag_uuid in final_uuids:
                session.add(BBoxTagLink(bbox_uuid=bbox_uuid, tag_uuid=tag_uuid))
            await self._touch_image(session, image_uuid)
            await session.commit()

    async def delete_single_bbox(self, bbox_uuid: str) -> Optional[str]:
        """Delete one bbox and return its image_uuid, or None if not found."""
        async with self._async_session_maker() as session:
            bbox = await session.get(BBox, bbox_uuid)
            if bbox is None:
                return None
            image_uuid = bbox.image_uuid
            await session.delete(bbox)
            await session.commit()
            return image_uuid

    async def get_bboxes_for_image(self, image_uuid: str) -> List[BBox]:
        """Return all BBox rows for an image with their tags and label loaded."""
        async with self._async_session_maker() as session:
            stmt = (
                select(BBox)
                .where(BBox.image_uuid == image_uuid)
                .options(
                    selectinload(BBox.tags),
                    selectinload(BBox.label),
                )
            )
            result = await session.exec(stmt)
            return list(result.all())

    async def get_bboxes_for_images(self, image_uuids: List[str]) -> Dict[str, List[BBox]]:
        """{image_uuid: [BBox]} for a batch, tags and label loaded.

        Exists for the sidecar-creation path: an image whose sidecar is missing
        needs its real boxes written alongside the assessments, and doing that
        per image would put a query per image on a bulk action. Images with no
        boxes are omitted — callers default the lookup themselves.
        """
        if not image_uuids:
            return {}
        out: Dict[str, List[BBox]] = {}
        async with self._async_session_maker() as session:
            for chunk in _chunked(image_uuids):
                stmt = (
                    select(BBox)
                    .where(BBox.image_uuid.in_(chunk))  # type: ignore[attr-defined]
                    .options(
                        selectinload(BBox.tags),
                        selectinload(BBox.label),
                    )
                )
                for bbox in (await session.exec(stmt)).all():
                    out.setdefault(bbox.image_uuid, []).append(bbox)
        return out

    async def delete_image(self, image_uuid: str) -> bool:
        """Delete an image by UUID. BBoxes will be deleted via CASCADE."""
        async with self._async_session_maker() as session:
            db_image = await session.get(ImageRecord, image_uuid)
            if not db_image:
                return False

            await session.delete(db_image)
            await session.commit()
            return True

    async def get_image(self, image_uuid: str) -> Optional[ImageRecord]:
        """Return the ImageRecord for the given UUID, or None if not found."""
        async with self._async_session_maker() as session:
            return await session.get(ImageRecord, image_uuid)

    async def get_all_image_stubs(self) -> List[Dict[str, str]]:
        """Return [{uuid, filename}] for every image record, with no pagination limit."""
        async with self._async_session_maker() as session:
            result = await session.exec(select(ImageRecord.uuid, ImageRecord.filename))
            return [{"uuid": uuid, "filename": filename} for uuid, filename in result.all()]

    async def get_image_hash_data_by_prefixes(self, prefixes: List[str]) -> tuple:
        """Return hash data for all images whose filename falls under any of the given prefixes.

        *prefixes* must be absolute paths (already resolved).  A filename matches when it equals
        a prefix exactly or starts with ``prefix + "/"``.

        Returns ``(image_rows, bbox_rows)`` where:
          - ``image_rows`` is a list of ``(uuid, phash)`` tuples
          - ``bbox_rows``  is a list of ``(uuid, phash, label_uuid)`` tuples for every bbox
            that belongs to a matched image
        """
        if not prefixes:
            return [], []
        async with self._async_session_maker() as session:
            conditions = or_(
                *[
                    or_(
                        ImageRecord.filename == p,
                        ImageRecord.filename.like(p + "/%"),
                    )
                    for p in prefixes
                ]
            )
            img_result = await session.exec(select(ImageRecord.uuid, ImageRecord.phash).where(conditions))
            image_rows = img_result.all()
            if not image_rows:
                return [], []

            image_uuids = [r[0] for r in image_rows]
            bbox_rows = []
            for chunk in _chunked(image_uuids):
                bbox_result = await session.exec(
                    select(BBox.uuid, BBox.phash, BBox.label_uuid).where(BBox.image_uuid.in_(chunk))
                )
                bbox_rows.extend(bbox_result.all())

        return image_rows, bbox_rows

    async def get_images_by_prefixes(self, prefixes: List[str]) -> List[Tuple[str, str]]:
        """Return ``(uuid, filename)`` for every image whose filename falls under a prefix.

        Same matching rule as ``get_image_hash_data_by_prefixes`` — equal to a prefix, or
        starting with ``prefix + "/"`` — so a relocation sees exactly the set a
        delete-by-directory would have removed. Sorted by filename to give the caller a
        stable order: a relocation that is interrupted and re-run then resumes over the
        same sequence rather than a fresh arbitrary one.
        """
        if not prefixes:
            return []
        async with self._async_session_maker() as session:
            conditions = or_(
                *[
                    or_(
                        ImageRecord.filename == p,
                        ImageRecord.filename.like(p + "/%"),
                    )
                    for p in prefixes
                ]
            )
            result = await session.exec(
                select(ImageRecord.uuid, ImageRecord.filename).where(conditions).order_by(ImageRecord.filename)
            )
            return list(result.all())

    async def get_image_uuids_by_filenames(self, filenames: List[str]) -> Dict[str, str]:
        """Map absolute filename → image uuid for the given filenames.

        Used to detect collisions before a relocation writes: ``images.filename`` is
        UNIQUE, so a destination already held by another record must be rejected up
        front rather than surfacing as an IntegrityError halfway through a batch.
        """
        if not filenames:
            return {}
        out: Dict[str, str] = {}
        async with self._async_session_maker() as session:
            for chunk in _chunked(filenames):
                result = await session.exec(
                    select(ImageRecord.filename, ImageRecord.uuid).where(ImageRecord.filename.in_(chunk))
                )
                for fname, uuid in result.all():
                    out[fname] = uuid
        return out

    async def delete_images_bulk(self, uuids: List[str]) -> int:
        """Delete image records in bulk.  BBoxes cascade-delete automatically.

        Returns the number of image records deleted.
        """
        if not uuids:
            return 0
        async with self._async_session_maker() as session:
            result = await session.exec(delete(ImageRecord).where(ImageRecord.uuid.in_(uuids)))
            await session.commit()
            deleted_count = result.rowcount or 0
        self._bump_write_epoch()

        # A large delete leaves that many pages on the freelist; auto_vacuum=INCREMENTAL
        # (set in _set_sqlite_pragma) doesn't reclaim them on its own, only on request.
        # Reclaiming inline keeps the file from silently ballooning after big cleanups
        # like an accidental mass-ingest, without paying VACUUM's full-file-rewrite cost
        # on every small delete.
        if not self._memory and deleted_count >= self._INCREMENTAL_VACUUM_THRESHOLD:
            async with self._engine.connect() as conn:
                await conn.execute(text("PRAGMA incremental_vacuum"))

        return deleted_count

    #: Checkpoint the WAL once it exceeds this. SQLite's own wal_autocheckpoint
    #: fires at 1000 pages (~4 MB) but only ever runs a PASSIVE checkpoint, which
    #: copies pages into the main DB without resetting the file — so a WAL that
    #: never finds a reader-free moment keeps its high-water mark forever. 16 MB
    #: leaves plenty of room above the 4 MB passive threshold, so this only fires
    #: when the passive path has genuinely failed to keep up rather than racing it.
    _WAL_CHECKPOINT_THRESHOLD_BYTES = 16 * 1024 * 1024

    async def checkpoint_wal(self) -> Optional[int]:
        """Truncate the WAL back to empty if it has grown past the threshold.

        Observed in production on 2026-08-19: a 57 MB WAL against a 38 MB
        database, 14x past SQLite's own 4 MB autocheckpoint threshold and not
        shrinking. That is the expected outcome of a busy server, not a bug in
        SQLite: a PASSIVE checkpoint (the only kind autocheckpoint runs) needs a
        moment with no active readers to reset the file, and db_server rarely has
        one. Readers then pay to search an ever-larger WAL, and crash recovery has
        that much more to replay.

        TRUNCATE is the mode that actually returns the space. It is also the one
        that can block, so this is deliberately called from the periodic worker
        rather than from a request path, and a busy result is left for the next
        pass instead of retried — the WAL being large is a slow-burn cost, never
        an emergency.

        Returns:
            Optional[int]: Bytes the WAL shrank by, 0 if it was under the
            threshold or the checkpoint was blocked, or None for an in-memory DB
            (which has no WAL at all).
        """
        if self._memory:
            return None

        wal_path = self._db_path.with_name(self._db_path.name + "-wal")
        try:
            before = wal_path.stat().st_size
        except OSError:
            # No WAL file yet, or it vanished under us — nothing to reclaim.
            return 0
        if before < self._WAL_CHECKPOINT_THRESHOLD_BYTES:
            return 0

        async with self._engine.connect() as conn:
            result = await conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
            row = result.first()

        # Row is (busy, log_pages, checkpointed_pages); busy=1 means a reader or
        # writer held it off. Nothing to do but come back next pass.
        if row is not None and row[0]:
            logger.info("WAL checkpoint blocked by an active reader/writer; will retry (wal=%d bytes)", before)
            return 0

        try:
            after = wal_path.stat().st_size
        except OSError:
            after = 0
        reclaimed = max(before - after, 0)
        if reclaimed:
            logger.info("WAL checkpoint reclaimed %.1f MB (%d -> %d bytes)", reclaimed / 1048576, before, after)
        return reclaimed

    async def get_image_stubs_needing_phash(self, image_hex_len: int, bbox_hex_len: int) -> List[Dict[str, str]]:
        """Return [{uuid, filename}] for images that are missing a phash or have one at the wrong size.

        Covers two cases:
          1. The image's own phash is NULL or has a different hex length than *image_hex_len*.
          2. Any bbox on the image has a NULL phash or a different hex length than *bbox_hex_len*.
        """
        bbox_missing = (
            select(BBox.uuid)
            .where(
                BBox.image_uuid == ImageRecord.uuid,
                or_(BBox.phash.is_(None), func.length(BBox.phash) != bbox_hex_len),
            )
            .exists()
        )
        stmt = select(ImageRecord.uuid, ImageRecord.filename).where(
            or_(
                ImageRecord.phash.is_(None),
                func.length(ImageRecord.phash) != image_hex_len,
                bbox_missing,
            )
        )
        async with self._async_session_maker() as session:
            result = await session.exec(stmt)
            return [{"uuid": uuid, "filename": filename} for uuid, filename in result.all()]

    async def update_image_filename(self, image_uuid: str, new_filename: str) -> bool:
        """Update the filename of an existing image record. Returns True if found and updated."""
        async with self._async_session_maker() as session:
            db_image = await session.get(ImageRecord, image_uuid)
            if not db_image:
                return False
            db_image.filename = new_filename
            session.add(db_image)
            await session.commit()
            return True

    async def clear_orphan_generated_tags(self) -> int:
        """Delete TASK_GENERATED tags that have no associated bbox_tags rows.

        Returns the number of tags deleted.
        """
        async with self._async_session_maker() as session:
            # Find TASK_GENERATED tag UUIDs that have at least one bbox_tags row.
            referenced = select(BBoxTagLink.tag_uuid).distinct()
            stmt = select(Tag).where(
                Tag.kind == TagKind.TASK_GENERATED,
                ~Tag.uuid.in_(referenced),
            )
            result = await session.exec(stmt)
            orphans = result.all()
            count = len(orphans)
            for tag in orphans:
                await session.delete(tag)
            if count:
                await session.commit()
                await self._sync_tags_to_json()
        return count

    async def write_image_metadata_to_db(
        self,
        image_uuid: str,
        update: ImageRecordUpdate,
        image_phash: Optional[str] = None,
    ) -> None:
        """Replace all bboxes for an image directly in the database.

        Each bbox's `phash` is written with the row it belongs to, and
        `image_phash` (when given) in the same transaction. SQLite takes one
        writer at a time, so a save's cost against a busy database is the number
        of times it asks for that lock — a separate `update_phashes` round trip
        doubled it for hashes the caller already had in hand.
        """
        async with self._async_session_maker() as session:
            await session.exec(delete(BBox).where(BBox.image_uuid == image_uuid))

            # Batch-fetch all tag metadata needed for exclusive-group enforcement.
            all_tag_uuids = {u for b in update.bboxes for u in b.tag_uuids}
            if all_tag_uuids:
                tag_rows = await session.exec(select(Tag).where(Tag.uuid.in_(all_tag_uuids)))
                tags_by_uuid = {t.uuid: t for t in tag_rows.all()}
            else:
                tags_by_uuid = {}

            for b in update.bboxes:
                final_uuids = self._resolve_exclusive_groups(b.tag_uuids, tags_by_uuid)
                session.add(
                    BBox(
                        uuid=b.uuid,
                        image_uuid=image_uuid,
                        label_uuid=b.label_uuid,
                        x=b.x,
                        y=b.y,
                        width=b.width,
                        height=b.height,
                        metadata_json=self._apply_verified_confidence_rule(b.metadata_json, final_uuids) or {},
                        phash=b.phash,
                    )
                )
                for tag_uuid in final_uuids:
                    session.add(BBoxTagLink(bbox_uuid=b.uuid, tag_uuid=tag_uuid))
            # Touch the image so updated_at reflects the full bbox replacement.
            await self._touch_image(session, image_uuid)
            if image_phash is not None:
                image = await session.get(ImageRecord, image_uuid)
                if image is not None:
                    image.phash = image_phash
                    session.add(image)
            await session.commit()

    @staticmethod
    def _sidecar_lock(json_path: Path) -> threading.Lock:
        """Return the lock guarding read-modify-write access to `json_path`.

        Striped rather than one-lock-per-path: a per-path registry would grow a
        lock object for every image ever written and never release them, and the
        writes under the lock are a couple of milliseconds of file I/O, so the
        occasional two-unrelated-sidecars collision costs nothing measurable.
        """
        return _SIDECAR_LOCKS[hash(str(json_path)) % len(_SIDECAR_LOCKS)]

    @staticmethod
    def _merge_write_sidecar(json_path: Path, new_data: Dict[str, Any]) -> None:
        """Merge `new_data` into the sidecar at `json_path` and write it atomically.

        Only the top-level keys present in `new_data` are replaced; every other
        key already in the file survives. That is what lets the bbox writer and
        the image_labels writer each own their own slice of the same file
        without either having to know what the other stores.

        The read and the write are one critical section, not two independent
        steps. Saving an annotation issues one write per changed label, all of
        them concurrent on the thread pool and all of them aimed at the same
        sidecar; without the lock two writers both read the pre-merge file and
        the second one writes its slice over a stale copy, dropping the first
        one's key from disk while the database kept both.
        """
        with DbClient._sidecar_lock(json_path):
            existing: Dict[str, Any] = {}
            if json_path.exists():
                try:
                    existing = jstyleson.loads(json_path.read_text())
                except Exception as e:
                    logger.warning(f"Could not parse sidecar {json_path}, overwriting: {e}")
            existing.update(new_data)
            DbClient._atomic_write_text(json_path, json.dumps(existing, indent=2))

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        """Write `text` to `path` via a sibling temp file and os.replace.

        A crash mid-write can never leave a torn/half-written file (which a
        later overwrite-sync would then refuse — or, historically, wipe from).
        Every on-disk copy that exists to survive a lost database goes through
        here; a truncated one defeats the point of keeping it.

        The temp file name is unique per call. A fixed `<name>.tmp` is shared by
        every concurrent writer of the same file: the first `os.replace` renames
        the temp file away, and the next writer's replace then fails with
        FileNotFoundError even though its own data was written fine — which the
        UI reported as "N label state(s) failed to save" after the row had
        already landed in the database.
        """
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.replace(tmp_path, path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def write_image_labels_to_json(self, json_path: Path, image_labels: List[Dict[str, Any]]) -> None:
        """Write only the whole-image label assessments into an existing sidecar.

        Kept separate from write_image_metadata_to_json because the two are
        driven by different events: bbox edits rewrite `boxes`, and
        present/absent/uncertain actions rewrite `image_labels`. Neither should
        have to re-derive the other's data just to persist its own.

        An empty list is written as an empty list, not omitted — "the user
        cleared every assessment" has to be distinguishable from "this sidecar
        predates the feature", or a restore would resurrect assessments the
        user deliberately removed.
        """
        self._merge_write_sidecar(json_path, {"image_labels": image_labels})

    def write_image_metadata_to_json(
        self,
        json_path: Path,
        image_uuid: str,
        filename: str,
        update: ImageRecordUpdate,
        image_phash: Optional[str] = None,
        image_labels: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Write image metadata to the JSON sidecar file.

        `image_labels` is optional: when None the sidecar's existing
        `image_labels` key is left untouched rather than cleared, so a caller
        that only knows about bboxes cannot destroy the label assessments.
        """
        new_data: Dict[str, Any] = {
            "uuid": image_uuid,
            "filename": filename,
            "boxes": [
                {
                    "uuid": b.uuid,
                    "label_uuid": b.label_uuid,
                    "label_name": b.label_name,
                    "x": b.x,
                    "y": b.y,
                    "width": b.width,
                    "height": b.height,
                    "tag_uuids": b.tag_uuids,
                    "tag_names": b.tag_names,
                    **({"phash": b.phash} if b.phash else {}),
                }
                for b in update.bboxes
            ],
        }
        if image_phash:
            new_data["phash"] = image_phash
        if image_labels is not None:
            new_data["image_labels"] = image_labels
        self._merge_write_sidecar(json_path, new_data)

    async def get_all_image_phashes(self) -> List[Dict[str, str]]:
        """Return [{uuid, phash}] for all images with a non-null phash."""
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(ImageRecord.uuid, ImageRecord.phash).where(ImageRecord.phash.is_not(None))  # type: ignore[arg-type]
            )
            return [{"uuid": row[0], "phash": row[1]} for row in result.all()]

    async def get_all_bbox_phashes(self) -> List[Dict[str, str]]:
        """Return [{uuid, label_uuid, phash}] for all bboxes with a non-null phash."""
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(BBox.uuid, BBox.label_uuid, BBox.phash).where(BBox.phash.is_not(None))  # type: ignore[arg-type]
            )
            return [{"uuid": row[0], "label_uuid": row[1], "phash": row[2]} for row in result.all()]

    async def get_image_phashes_page(self, offset: int, limit: int) -> List[Dict[str, str]]:
        """Return up to `limit` [{uuid, phash}] rows starting at `offset`."""
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(ImageRecord.uuid, ImageRecord.phash)
                .where(ImageRecord.phash.is_not(None))  # type: ignore[arg-type]
                .offset(offset)
                .limit(limit)
            )
            return [{"uuid": row[0], "phash": row[1]} for row in result.all()]

    async def get_bbox_phashes_page(self, offset: int, limit: int) -> List[Dict[str, str]]:
        """Return up to `limit` [{uuid, label_uuid, phash}] rows starting at `offset`."""
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(BBox.uuid, BBox.label_uuid, BBox.phash)
                .where(BBox.phash.is_not(None))  # type: ignore[arg-type]
                .offset(offset)
                .limit(limit)
            )
            return [{"uuid": row[0], "label_uuid": row[1], "phash": row[2]} for row in result.all()]

    async def count_image_phashes(self) -> int:
        """Return count of images that have a non-null phash."""
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(func.count()).select_from(ImageRecord).where(ImageRecord.phash.is_not(None))  # type: ignore[arg-type]
            )
            return result.one() or 0

    async def count_bbox_phashes(self) -> int:
        """Return count of bboxes that have a non-null phash."""
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(func.count()).select_from(BBox).where(BBox.phash.is_not(None))  # type: ignore[arg-type]
            )
            return result.one() or 0

    async def clear_all_phashes(self) -> None:
        """NULL out every image and bbox phash in a single transaction.

        Called when the global hash size changes — old hashes are incompatible
        with the new size and must be removed so fresh ones can be computed.
        """
        import sqlalchemy as _sa

        async with self._async_session_maker() as session:
            await session.exec(_sa.update(ImageRecord).values(phash=None))  # type: ignore[call-overload]
            await session.exec(_sa.update(BBox).values(phash=None))  # type: ignore[call-overload]
            await session.commit()

    async def update_phashes(
        self,
        image_uuid: str,
        image_phash: str,
        bbox_phashes: List[Dict[str, str]],
    ) -> None:
        """Write image phash and bbox phashes to the DB in a single transaction.

        bbox_phashes is a list of {"uuid": ..., "phash": ...} dicts.
        """
        import sqlalchemy as _sa

        async with self._async_session_maker() as session:
            await session.exec(  # type: ignore[call-overload]
                _sa.update(ImageRecord).where(ImageRecord.uuid == image_uuid).values(phash=image_phash)
            )
            for bp in bbox_phashes:
                await session.exec(  # type: ignore[call-overload]
                    _sa.update(BBox).where(BBox.uuid == bp["uuid"]).values(phash=bp["phash"])
                )
            await session.commit()

    ################################################################################
    # Labels
    ################################################################################
    async def _sync_labels_to_json(self):
        """Internal helper to dump the current labels table to json."""
        labels = await self.list_labels()
        adapter = TypeAdapter(List[Label])
        json_str = adapter.dump_json(
            labels,
            exclude={
                "__all__": {
                    "bboxes": True,
                    "children": True,
                    "parent": True,
                    "created_at": True,
                    "updated_at": True,
                }
            },
            indent=2,
        )
        self._labels_json_path.write_bytes(json_str)

    async def sync_labels_from_json(self):
        """Reads labels from JSON to db. Does not remove existing labels from database.

        Resolution order per label:
        1. UUID match  → update the existing DB record in-place.
        2. Name match  → reuse the existing DB record (log UUID mismatch), remap
                         any child parent_uuid references to the DB UUID.
        3. No match    → insert a new record with the UUID from the JSON.

        The multi-pass loop handles the label hierarchy (parents before children).
        """
        if not self._labels_json_path.exists():
            return

        json_data = jstyleson.loads(self._labels_json_path.read_text())

        async with self._async_session_maker() as session:
            # Pre-fetch all existing labels for O(1) lookups.
            existing = (await session.exec(select(Label))).all()
            labels_by_uuid: Dict[str, Label] = {lbl.uuid: lbl for lbl in existing}
            labels_by_name: Dict[str, Label] = {lbl.name: lbl for lbl in existing}

            # Seed with all pre-existing DB label UUIDs so that JSON children whose
            # parent already lives in the DB (but is absent from this JSON batch) are
            # not incorrectly deferred as orphans.
            processed_uuids: set = set(labels_by_uuid.keys())
            # Maps a JSON uuid → the actual DB uuid used (identity or name-fallback remap).
            # Used to rewrite child parent_uuid fields so the hierarchy stays consistent.
            uuid_remapping: Dict[str, str] = {}

            pending = list(json_data)
            while pending:
                batch_processed = False
                remaining = []

                for item in pending:
                    json_uuid = item.get("uuid")
                    parent_uuid = item.get("parent_uuid")

                    # Resolve parent through any prior remapping so the hierarchy check works.
                    resolved_parent = uuid_remapping.get(parent_uuid, parent_uuid)
                    if resolved_parent is not None and resolved_parent not in processed_uuids:
                        remaining.append(item)
                        continue

                    # The parent_uuid stored in DB should use the remapped (DB) uuid.
                    effective_parent = uuid_remapping.get(parent_uuid, parent_uuid)

                    if json_uuid and json_uuid in labels_by_uuid:
                        # 1. UUID match — update fields in-place.
                        db_label = labels_by_uuid[json_uuid]
                        for key, value in item.items():
                            if key != "uuid" and hasattr(db_label, key):
                                setattr(db_label, key, value)
                        db_label.parent_uuid = effective_parent
                        session.add(db_label)
                        processed_uuids.add(json_uuid)
                        uuid_remapping[json_uuid] = json_uuid

                    elif item.get("name") and item["name"] in labels_by_name:
                        # 2. Name fallback — reuse existing DB record's UUID.
                        db_label = labels_by_name[item["name"]]
                        logger.warning(
                            "Label '%s' from JSON (UUID %s) matched by name to existing "
                            "DB label (UUID %s). Using DB UUID.",
                            item["name"],
                            json_uuid,
                            db_label.uuid,
                        )
                        for key, value in item.items():
                            if key != "uuid" and hasattr(db_label, key):
                                setattr(db_label, key, value)
                        db_label.parent_uuid = effective_parent
                        session.add(db_label)
                        processed_uuids.add(db_label.uuid)
                        if json_uuid:
                            uuid_remapping[json_uuid] = db_label.uuid
                            processed_uuids.add(json_uuid)

                    else:
                        # 3. No match — insert new with the JSON's UUID.
                        effective_item = dict(item)
                        effective_item["parent_uuid"] = effective_parent
                        new_label = Label.model_validate(effective_item)
                        session.add(new_label)
                        labels_by_uuid[new_label.uuid] = new_label
                        labels_by_name[new_label.name] = new_label
                        processed_uuids.add(new_label.uuid)
                        if json_uuid:
                            uuid_remapping[json_uuid] = new_label.uuid

                    batch_processed = True

                if not batch_processed and remaining:
                    logger.warning(
                        "Orphaned labels in JSON (unresolved parents): %s",
                        [x.get("name") for x in remaining],
                    )
                    break

                pending = remaining

            await session.commit()
        logger.info("Imported %d labels from JSON", len(processed_uuids))

    async def list_labels(self) -> List[Label]:
        """List all labels using SQLModel select."""
        async with self._async_session_maker() as session:
            statement = select(Label)
            result = await session.exec(statement)
            return result.all()

    async def add_label(
        self,
        name: str,
        color: Optional[str] = None,
        uuid: Optional[str] = None,
        parent_uuid: Optional[str] = None,
        label_type: LabelType = LabelType.STANDARD,
        scope: LabelScope = LabelScope.OBJECT,
        publish_to_ha: bool = False,
        publish_per_source: bool = False,
        exclusive_variants: bool = False,
        exclusive_variants_iou_threshold: Optional[float] = None,
    ) -> Label:
        """Insert a label and return the created object.

        The two publish flags default off to match `LabelBase`, so a caller that does not
        mention Home Assistant does not create a device in one. `exclusive_variants` also
        defaults off here — the parent-aware smart default (a parentless label defaults
        on) lives in `AddLabelPayload` (apps/db_server/web.py), not this lower layer.
        """
        new_label = Label(
            uuid=uuid or str(uuid4()),
            name=name,
            color=color,
            parent_uuid=parent_uuid,
            label_type=label_type,
            scope=scope,
            publish_to_ha=publish_to_ha,
            publish_per_source=publish_per_source,
            exclusive_variants=exclusive_variants,
            exclusive_variants_iou_threshold=exclusive_variants_iou_threshold,
        )

        async with self._async_session_maker() as session:
            session.add(new_label)
            await session.commit()
            await session.refresh(new_label)
        await self._sync_labels_to_json()
        return new_label

    async def get_label_by_uuid(self, uuid: str) -> Optional[Label]:
        """Retrieve a label by its primary key UUID."""
        async with self._async_session_maker() as session:
            return await session.get(Label, uuid)

    async def get_labels_by_uuids(self, uuids: List[str]) -> Dict[str, Label]:
        """Retrieve multiple labels by their UUIDs in a single batch."""
        if not uuids:
            return {}

        async with self._async_session_maker() as session:
            statement = select(Label).where(Label.uuid.in_(uuids))
            result = await session.exec(statement)
            labels = result.all()
            return {lbl.uuid: lbl for lbl in labels}

    async def get_label_children(self, parent_uuid: str) -> List[str]:
        """Retrieve the UUIDs of labels that have the given UUID as their parent."""
        async with self._async_session_maker() as session:
            statement = select(Label.uuid).where(Label.parent_uuid == parent_uuid)
            result = await session.exec(statement)
            return list(result.all())

    async def update_label(self, label_uuid: str, **kwargs) -> Optional[Label]:
        """
        Update label fields dynamically.
        Usage: await client.update_label("some-uuid", name="New Name", parent_uuid=None)
        """
        async with self._async_session_maker() as session:
            db_label = await session.get(Label, label_uuid)
            if not db_label:
                return None

            # Apply changes from kwargs
            for key, value in kwargs.items():
                # For parents, we allow explicit None, so we only skip if key doesn't exist
                if hasattr(db_label, key):
                    setattr(db_label, key, value)

            session.add(db_label)
            await session.commit()
            await session.refresh(db_label)
        await self._sync_labels_to_json()
        return db_label

    async def delete_label(self, label_uuid: str) -> bool:
        """Delete a label by UUID."""
        async with self._async_session_maker() as session:
            db_label = await session.get(Label, label_uuid)
            if not db_label:
                return False

            await session.delete(db_label)
            await session.commit()
        await self._sync_labels_to_json()
        return True

    ################################################################################
    # Tags
    ################################################################################
    async def _sync_tags_to_json(self):
        """Internal helper to dump the current tags table to tags.json."""
        tags = await self.list_tags()
        adapter = TypeAdapter(List[Tag])
        json_str = adapter.dump_json(
            tags,
            exclude={"__all__": {"images": True, "created_at": True, "updated_at": True}},
            indent=2,
        )
        self._tags_json_path.write_bytes(json_str)

    async def sync_tags_from_json(self):
        """Reads tags.json and syncs to DB. Does not remove existing tags.

        Resolution order per tag:
        1. UUID match  → update the existing DB record in-place.
        2. Name match  → reuse the existing DB record (avoids unique-name IntegrityError).
        3. No match    → insert a new record with the UUID from the JSON.

        Entries whose UUID is in RETIRED_TAG_UUIDS are dropped outright — a
        sidecar dumped before the removing migration ran still lists them, and
        this is exactly the rebuild path that would resurrect one.
        """
        if not self._tags_json_path.exists():
            return

        try:
            json_data = jstyleson.loads(self._tags_json_path.read_text())
        except Exception as e:
            logger.error(f"Failed to parse tags.json: {e}")
            return

        async with self._async_session_maker() as session:
            existing = (await session.exec(select(Tag))).all()
            tags_by_uuid: Dict[str, Tag] = {t.uuid: t for t in existing}
            tags_by_name: Dict[str, Tag] = {t.name: t for t in existing}

            count = 0
            for item in json_data:
                json_uuid = item.get("uuid")
                name = item.get("name")

                if json_uuid in RETIRED_TAG_UUIDS:
                    logger.info("Skipping retired tag '%s' (UUID %s) from tags.json", name, json_uuid)
                    continue

                if json_uuid and json_uuid in tags_by_uuid:
                    # 1. UUID match — update in-place.
                    db_tag = tags_by_uuid[json_uuid]
                    for key, value in item.items():
                        if key != "uuid" and hasattr(db_tag, key):
                            setattr(db_tag, key, value)
                    session.add(db_tag)
                elif name and name in tags_by_name:
                    # 2. Name fallback — reuse DB record to avoid unique-name violation.
                    db_tag = tags_by_name[name]
                    logger.warning(
                        "Tag '%s' from JSON (UUID %s) matched by name to existing DB tag " "(UUID %s). Using DB UUID.",
                        name,
                        json_uuid,
                        db_tag.uuid,
                    )
                    for key, value in item.items():
                        if key != "uuid" and hasattr(db_tag, key):
                            setattr(db_tag, key, value)
                    session.add(db_tag)
                else:
                    # 3. No match — insert new.
                    new_tag = Tag.model_validate(item)
                    session.add(new_tag)
                    tags_by_uuid[new_tag.uuid] = new_tag
                    tags_by_name[new_tag.name] = new_tag
                count += 1

            await session.commit()
        logger.info("Imported %d tags from JSON", count)

    async def list_tags(self) -> List[Tag]:
        """List all tags using SQLModel select."""
        async with self._async_session_maker() as session:
            statement = select(Tag)
            result = await session.exec(statement)
            return result.all()

    async def add_tag(
        self,
        name: str,
        color: Optional[str] = None,
        icon: Optional[str] = None,
        uuid: Optional[str] = None,
        protected: bool = False,
        kind: TagKind = TagKind.GENERIC,
        exclusive_group: Optional[str] = None,
    ) -> Tag:
        """Insert a tag and return the created object."""
        new_tag = Tag(
            uuid=uuid or str(uuid4()),
            name=name,
            color=color,
            icon=icon,
            protected=protected,
            kind=kind,
            exclusive_group=exclusive_group,
        )

        async with self._async_session_maker() as session:
            session.add(new_tag)
            await session.commit()
            await session.refresh(new_tag)
        await self._sync_tags_to_json()
        return new_tag

    async def get_tag_by_uuid(self, uuid: str) -> Optional[Tag]:
        """Retrieve a tag by its primary key UUID."""
        async with self._async_session_maker() as session:
            return await session.get(Tag, uuid)

    async def update_tag(self, tag_uuid: str, **kwargs) -> Optional[Tag]:
        """
        Update tag fields dynamically.
        Usage: await client.update_tag("some-uuid", name="New Name", color="#FFF")
        """
        async with self._async_session_maker() as session:
            db_tag = await session.get(Tag, tag_uuid)
            if not db_tag:
                return None

            # Apply changes from kwargs
            for key, value in kwargs.items():
                if value is not None and hasattr(db_tag, key):
                    setattr(db_tag, key, value)

            session.add(db_tag)
            await session.commit()
            await session.refresh(db_tag)
        await self._sync_tags_to_json()
        return db_tag

    async def delete_tag(self, tag_uuid: str) -> bool:
        """Delete a tag by UUID."""
        async with self._async_session_maker() as session:
            db_tag = await session.get(Tag, tag_uuid)
            if not db_tag:
                return False

            await session.delete(db_tag)
            await session.commit()
        await self._sync_tags_to_json()
        return True

    ################################################################################
    # User Settings
    ################################################################################

    async def get_setting(self, key: str) -> Optional[str]:
        """Return the stored value for key, or None if not set."""
        async with self._async_session_maker() as session:
            obj = await session.get(UserSetting, key)
            return obj.value if obj else None

    async def set_setting(self, key: str, value: str) -> None:
        """Upsert a key-value pair in user_settings."""
        async with self._async_session_maker() as session:
            await session.merge(UserSetting(key=key, value=value))
            await session.commit()

    ################################################################################
    # Sources
    ################################################################################
    async def _sync_sources_to_json(self):
        """Internal helper to dump the current sources table to sources.json."""
        sources = await self.get_sources()
        adapter = TypeAdapter(List[SourceRead])
        json_str = adapter.dump_json(
            [SourceRead.model_validate(s) for s in sources],
            indent=2,
        )
        self._sources_json_path.write_bytes(json_str)

    async def sync_sources_from_json(self):
        """Reads sources.json and syncs to DB. Does not remove existing sources.

        Resolution order per source:
        1. UUID match  → update the existing DB record in-place.
        2. Name match  → reuse the existing DB record (avoids unique-name IntegrityError).
        3. No match    → insert a new record with the UUID from the JSON.
        """
        if not self._sources_json_path.exists():
            return

        try:
            json_data = jstyleson.loads(self._sources_json_path.read_text())
        except Exception as e:
            logger.error(f"Failed to parse sources.json: {e}")
            return

        async with self._async_session_maker() as session:
            existing = (await session.exec(select(Source))).all()
            sources_by_uuid: Dict[str, Source] = {s.uuid: s for s in existing}
            sources_by_name: Dict[str, Source] = {s.name: s for s in existing}

            count = 0
            for item in json_data:
                json_uuid = item.get("uuid")
                name = item.get("name")

                if json_uuid and json_uuid in sources_by_uuid:
                    # 1. UUID match — update in-place.
                    db_src = sources_by_uuid[json_uuid]
                    for key, value in item.items():
                        if key != "uuid" and hasattr(db_src, key):
                            setattr(db_src, key, value)
                    session.add(db_src)
                elif name and name in sources_by_name:
                    # 2. Name fallback — reuse DB record to avoid unique-name violation.
                    db_src = sources_by_name[name]
                    logger.warning(
                        "Source '%s' from JSON (UUID %s) matched by name to existing DB source "
                        "(UUID %s). Using DB UUID.",
                        name,
                        json_uuid,
                        db_src.uuid,
                    )
                    for key, value in item.items():
                        if key != "uuid" and hasattr(db_src, key):
                            setattr(db_src, key, value)
                    session.add(db_src)
                else:
                    # 3. No match — insert new.
                    new_src = Source.model_validate(item)
                    session.add(new_src)
                    sources_by_uuid[new_src.uuid] = new_src
                    sources_by_name[new_src.name] = new_src
                count += 1

            await session.commit()
        logger.info("Imported %d sources from JSON", count)

    async def get_sources(self) -> List[Source]:
        """List all sources."""
        async with self._async_session_maker() as session:
            result = await session.exec(select(Source))
            return result.all()

    async def add_source(self, name: str, typename: str, params: Dict[str, Any]) -> Source:
        """Insert a source and return the created object."""
        new_source = Source(uuid=str(uuid4()), name=name, typename=typename, params=params)
        async with self._async_session_maker() as session:
            session.add(new_source)
            await session.commit()
            await session.refresh(new_source)
        await self._sync_sources_to_json()
        return new_source

    async def delete_sources(self, uuids: List[str]) -> None:
        """Delete sources by UUID list."""
        async with self._async_session_maker() as session:
            await session.exec(delete(Source).where(Source.uuid.in_(uuids)))
            await session.commit()
        await self._sync_sources_to_json()

    async def update_source(
        self, source_uuid: str, name: str, typename: str, params: Dict[str, Any]
    ) -> Optional[Source]:
        """Update a source by UUID and return the updated object."""
        async with self._async_session_maker() as session:
            source = await session.get(Source, source_uuid)
            if not source:
                return None
            source.name = name
            source.typename = typename
            source.params = params
            session.add(source)
            await session.commit()
            await session.refresh(source)
        await self._sync_sources_to_json()
        return source

    ################################################################################
    # Source Regions (zones and masks)
    ################################################################################
    #
    # Regions belong to the source rather than to a watcher's use of it: the geometry
    # describes the camera's view, so drawing it once per watcher only ever meant
    # redrawing it in every watcher when a camera moved.

    async def list_source_regions(
        self, source_uuid: Optional[str] = None, role: Optional[str] = None
    ) -> List[SourceRegionRead]:
        """Regions for one source, or for every source when `source_uuid` is None."""
        async with self._async_session_maker() as session:
            stmt = select(SourceRegion)
            if source_uuid is not None:
                stmt = stmt.where(SourceRegion.source_uuid == source_uuid)
            if role is not None:
                stmt = stmt.where(SourceRegion.role == role)
            stmt = stmt.order_by(SourceRegion.source_uuid, SourceRegion.order, SourceRegion.name)
            result = await session.exec(stmt)
            return [SourceRegionRead.model_validate(r) for r in result.all()]

    async def replace_source_regions(
        self, source_uuid: str, regions: List[SourceRegionWrite]
    ) -> List[SourceRegionRead]:
        """Whole-list replace of one source's regions, following the nested-write precedent.

        Caller-supplied uuids are preserved. That matters more here than anywhere else in
        the codebase: a rule points at a zone by uuid from a *different* table, so
        regenerating uuids on save would break every zone rule in every watcher at once,
        with no error and no obvious cause.
        """
        if not await self.source_exists(source_uuid):
            raise ValueError(f"Source {source_uuid!r} does not exist")

        seen_names: set = set()
        for region in regions:
            key = (region.role.value, region.name.strip().lower())
            if key in seen_names:
                raise ValueError(f"Two {region.role.value}s on this source are both named {region.name!r}")
            seen_names.add(key)

        async with self._async_session_maker() as session:
            await session.exec(delete(SourceRegion).where(SourceRegion.source_uuid == source_uuid))
            await session.flush()
            for index, region in enumerate(regions):
                session.add(
                    SourceRegion(
                        **({"uuid": region.uuid} if region.uuid else {}),
                        source_uuid=source_uuid,
                        name=region.name,
                        role=region.role.value,
                        kind=region.kind.value,
                        points_json=region.points_json,
                        reference_width=region.reference_width,
                        reference_height=region.reference_height,
                        order=region.order or index,
                    )
                )
            await session.commit()
        return await self.list_source_regions(source_uuid)

    async def source_exists(self, source_uuid: str) -> bool:
        async with self._async_session_maker() as session:
            return await session.get(Source, source_uuid) is not None

    async def find_rules_referencing_regions(self, region_uuids: Sequence[str]) -> List[str]:
        """Names of rules pointing at any of *region_uuids*.

        So deleting a region can warn instead of silently breaking rules.

        Takes the whole set rather than one uuid at a time: the refs live in JSON
        columns, so the membership test happens in Python either way, and asking once
        per region turned a single scan of `interaction_rules` into one scan per deleted
        region. Only zone-kind rules can name a region at all, so the rest never leave
        the database.
        """
        wanted = {str(u) for u in region_uuids}
        if not wanted:
            return []
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(InteractionRule).where(
                    or_(
                        InteractionRule.subject_a_kind == RuleSubjectKind.ZONE.value,
                        InteractionRule.subject_b_kind == RuleSubjectKind.ZONE.value,
                    )
                )
            )
            return [
                rule.name
                for rule in result.all()
                if wanted.intersection(rule.subject_a_refs_json or [])
                or wanted.intersection(rule.subject_b_refs_json or [])
            ]

    ################################################################################
    # Image Labels
    ################################################################################

    async def set_image_label_status(self, image_uuid: str, label_uuid: str, status: ImageItemStatus) -> None:
        """Upsert an image_label record, overwriting any existing status.

        Uses a single atomic INSERT ... ON CONFLICT DO UPDATE rather than a
        get-then-insert round trip. Two concurrent calls for the same
        (image_uuid, label_uuid) would otherwise both miss the SELECT and both
        attempt an INSERT — the loser hits a PK violation and surfaces as a
        failure to its caller even though the winner's write landed correctly.
        """
        now = datetime.now(timezone.utc)
        async with self._async_session_maker() as session:
            stmt = sqlite_insert(ImageLabel).values(
                image_uuid=image_uuid,
                label_uuid=label_uuid,
                status=status,
                updated_at=now,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[ImageLabel.image_uuid, ImageLabel.label_uuid],
                set_={"status": stmt.excluded.status, "updated_at": stmt.excluded.updated_at},
            )
            await session.exec(stmt)
            await session.commit()
        self._bump_write_epoch()

    async def set_image_labels_status(self, image_uuid: str, label_uuids: List[str], status: ImageItemStatus) -> None:
        """Batch upsert image_label rows for several labels on one image in a single
        session/statement, rather than one set_image_label_status() round trip per
        label — used by the bulk setAbsent/setPresent actions."""
        if not label_uuids:
            return
        now = datetime.now(timezone.utc)
        async with self._async_session_maker() as session:
            stmt = sqlite_insert(ImageLabel).values(
                [
                    {"image_uuid": image_uuid, "label_uuid": label_uuid, "status": status, "updated_at": now}
                    for label_uuid in label_uuids
                ]
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[ImageLabel.image_uuid, ImageLabel.label_uuid],
                set_={"status": stmt.excluded.status, "updated_at": stmt.excluded.updated_at},
            )
            await session.exec(stmt)
            await session.commit()
        self._bump_write_epoch()

    async def sync_image_present_labels(
        self,
        image_uuid: str,
        present_label_uuids: List[str],
        stale_label_uuids: List[str],
    ) -> None:
        """Set PRESENT rows and drop the ones that no longer apply, in one write.

        The two halves are one decision — "these labels are present now, those
        are not any more" — and were costing one transaction per present label
        plus another for the delete. On a busy database that is the difference
        between a save that feels instant and one that takes seconds, because
        each transaction queues separately for SQLite's single writer.

        ABSENT rows are never touched here; they are user-asserted.
        """
        if not present_label_uuids and not stale_label_uuids:
            return
        now = datetime.now(timezone.utc)
        async with self._async_session_maker() as session:
            if present_label_uuids:
                stmt = sqlite_insert(ImageLabel).values(
                    [
                        {
                            "image_uuid": image_uuid,
                            "label_uuid": label_uuid,
                            "status": ImageItemStatus.PRESENT,
                            "updated_at": now,
                        }
                        for label_uuid in present_label_uuids
                    ]
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[ImageLabel.image_uuid, ImageLabel.label_uuid],
                    set_={"status": stmt.excluded.status, "updated_at": stmt.excluded.updated_at},
                )
                await session.exec(stmt)
            if stale_label_uuids:
                await session.exec(
                    delete(ImageLabel).where(
                        ImageLabel.image_uuid == image_uuid,
                        ImageLabel.label_uuid.in_(stale_label_uuids),
                    )
                )
            await session.commit()
        self._bump_write_epoch()

    async def get_image_labels(self, image_uuid: str) -> List[ImageLabel]:
        """Return all image_label rows for a given image."""
        async with self._async_session_maker() as session:
            result = await session.exec(select(ImageLabel).where(ImageLabel.image_uuid == image_uuid))
            return result.all()

    async def get_image_uuids_with_labels(self) -> List[str]:
        """Every image uuid that has at least one image_labels row.

        Used by the sidecar backfill, which has no reason to walk images that
        carry no assessments.
        """
        async with self._async_session_maker() as session:
            result = await session.exec(select(ImageLabel.image_uuid).distinct())
            return list(result.all())

    async def get_image_labels_for_sidecar(self, image_uuid: str) -> List[Dict[str, Any]]:
        """Return this image's label assessments in sidecar form.

        Each row carries label_name alongside label_uuid for the same reason
        bboxes do: on a rebuild that mints new label UUIDs, the name is the only
        key left that can resolve the row.
        """
        rows = await self.get_image_labels_for_sidecar_bulk([image_uuid])
        return rows.get(image_uuid, [])

    async def get_image_labels_for_sidecar_bulk(self, image_uuids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Batched get_image_labels_for_sidecar: {image_uuid: [sidecar rows]}.

        Images with no assessments are omitted — callers that need an explicit
        empty list should default the lookup themselves.
        """
        if not image_uuids:
            return {}
        out: Dict[str, List[Dict[str, Any]]] = {}
        async with self._async_session_maker() as session:
            for chunk in _chunked(image_uuids):
                stmt = (
                    select(ImageLabel, Label.name)
                    .join(Label, Label.uuid == ImageLabel.label_uuid, isouter=True)
                    .where(ImageLabel.image_uuid.in_(chunk))
                )
                for row, label_name in (await session.exec(stmt)).all():
                    status = row.status.value if isinstance(row.status, ImageItemStatus) else str(row.status).lower()
                    # UNCERTAIN is the absence of a row by convention, and the
                    # reconcile reader skips it — so serialising one would write
                    # a row that a later overwrite sync deletes rather than
                    # restores. Keep writer and reader on the same rule.
                    if status == ImageItemStatus.UNCERTAIN.value:
                        continue
                    out.setdefault(row.image_uuid, []).append(
                        {
                            "label_uuid": row.label_uuid,
                            "label_name": label_name,
                            "status": status,
                        }
                    )
        # Stable ordering keeps the sidecar byte-identical across rewrites that
        # changed nothing, so backups and diffs stay quiet.
        for rows in out.values():
            rows.sort(key=lambda r: (r["label_name"] or "", r["label_uuid"]))
        return out

    async def get_image_label_sidecar_batch(
        self, image_uuids: List[str]
    ) -> Dict[str, Tuple[str, Optional[str], List[Dict[str, Any]]]]:
        """{image_uuid: (filename, phash, [sidecar rows])} for a batch, in one query.

        Deliberately one query rather than "fetch filenames, fetch rows": the
        bulk-apply paths run this after every page, and an image whose
        assessments were just cleared has zero rows but still needs its
        filename — so it is an outer join from images, not a lookup keyed on
        the rows. Images with no assessments come back with an empty list,
        which is the instruction to write `[]`.

        phash comes along because a sidecar that has to be *created* (an image
        with no boxes never got one from the bbox writer) would otherwise be
        written without it, and a rebuild from that sidecar would lose it.
        """
        if not image_uuids:
            return {}
        out: Dict[str, Tuple[str, Optional[str], List[Dict[str, Any]]]] = {}
        async with self._async_session_maker() as session:
            for chunk in _chunked(image_uuids):
                stmt = (
                    select(
                        ImageRecord.uuid,
                        ImageRecord.filename,
                        ImageRecord.phash,
                        ImageLabel.label_uuid,
                        ImageLabel.status,
                        Label.name,
                    )
                    .join(ImageLabel, ImageLabel.image_uuid == ImageRecord.uuid, isouter=True)
                    .join(Label, Label.uuid == ImageLabel.label_uuid, isouter=True)
                    .where(ImageRecord.uuid.in_(chunk))
                )
                for image_uuid, filename, phash, label_uuid, status, label_name in (await session.exec(stmt)).all():
                    entry = out.setdefault(image_uuid, (filename, phash, []))
                    if label_uuid is None:
                        continue
                    status_value = status.value if isinstance(status, ImageItemStatus) else str(status).lower()
                    # Same rule as get_image_labels_for_sidecar_bulk: UNCERTAIN is
                    # the absence of a row, and the reader skips it.
                    if status_value == ImageItemStatus.UNCERTAIN.value:
                        continue
                    entry[2].append({"label_uuid": label_uuid, "label_name": label_name, "status": status_value})
        for _filename, _phash, rows in out.values():
            rows.sort(key=lambda r: (r["label_name"] or "", r["label_uuid"]))
        return out

    async def get_image_labels_bulk(self, image_uuids: List[str]) -> List[ImageLabel]:
        """Return all image_label rows for a batch of images (chunked IN)."""
        if not image_uuids:
            return []
        rows: List[ImageLabel] = []
        async with self._async_session_maker() as session:
            for chunk in _chunked(image_uuids):
                result = await session.exec(select(ImageLabel).where(ImageLabel.image_uuid.in_(chunk)))
                rows.extend(result.all())
        return rows

    async def delete_image_labels(self, image_uuid: str, label_uuids: List[str]) -> None:
        """Delete image_label rows for the given image/label pairs."""
        if not label_uuids:
            return
        async with self._async_session_maker() as session:
            await session.exec(
                delete(ImageLabel).where(
                    ImageLabel.image_uuid == image_uuid,
                    ImageLabel.label_uuid.in_(label_uuids),
                )
            )
            await session.commit()
        # Core statement, not ORM — the after_flush listener never sees it. Same
        # reason `delete_images_bulk` bumps by hand; see the write-epoch comment above.
        self._bump_write_epoch()

    async def image_has_verified_bbox_for_label(self, image_uuid: str, label_uuid: str) -> bool:
        """Return True if the image has at least one bbox with the given label that carries the VERIFIED tag."""
        async with self._async_session_maker() as session:
            stmt = (
                select(BBox.uuid)
                .join(BBoxTagLink, BBoxTagLink.bbox_uuid == BBox.uuid)
                .where(
                    BBox.image_uuid == image_uuid,
                    BBox.label_uuid == label_uuid,
                    BBoxTagLink.tag_uuid == VERIFIED_TAG_UUID,
                )
                .limit(1)
            )
            result = await session.exec(stmt)
            return result.first() is not None

    async def labels_with_verified_bbox(self, image_uuid: str, label_uuids: List[str]) -> Set[str]:
        """Return the subset of label_uuids that have >=1 verified bbox on this image,
        in a single query — the batch counterpart to image_has_verified_bbox_for_label,
        used by the bulk setAbsent action to avoid one query per label."""
        if not label_uuids:
            return set()
        async with self._async_session_maker() as session:
            stmt = (
                select(BBox.label_uuid)
                .join(BBoxTagLink, BBoxTagLink.bbox_uuid == BBox.uuid)
                .where(
                    BBox.image_uuid == image_uuid,
                    BBox.label_uuid.in_(label_uuids),
                    BBoxTagLink.tag_uuid == VERIFIED_TAG_UUID,
                )
                .distinct()
            )
            result = await session.exec(stmt)
            return set(result.all())

    async def list_images_for_label(
        self,
        label_uuid: str,
        statuses: List[ImageItemStatus],
    ) -> List[ImageLabelWithFilename]:
        """Return all (image_uuid, filename, label_uuid, status) rows for a label+statuses filter.

        Two-step: query image_labels with the full filter, then batch-fetch filenames
        from images.  Avoids a cross-table column-projection JOIN that silently drops
        WHERE clauses in some SQLModel versions.
        Used by the training dataset loader to enumerate training/validation images.
        """
        if not statuses:
            return []
        async with self._async_session_maker() as session:
            label_rows: List[ImageLabel] = (
                await session.exec(
                    select(ImageLabel).where(
                        ImageLabel.label_uuid == label_uuid,
                        ImageLabel.status.in_(statuses),
                    )
                )
            ).all()

            if not label_rows:
                return []

            image_uuids = [r.image_uuid for r in label_rows]
            uuid_to_filename: Dict[str, str] = {}
            for chunk in _chunked(image_uuids):
                img_records = (await session.exec(select(ImageRecord).where(ImageRecord.uuid.in_(chunk)))).all()
                uuid_to_filename.update({img.uuid: img.filename for img in img_records})

        return [
            ImageLabelWithFilename(
                image_uuid=r.image_uuid,
                filename=uuid_to_filename.get(r.image_uuid, ""),
                label_uuid=r.label_uuid,
                status=r.status,
            )
            for r in label_rows
        ]

    async def add_image_tag(self, image_uuid: str, tag_uuid: str) -> None:
        """Link a tag to a whole image. No-op if the link already exists."""
        await self.add_image_tags(image_uuid, [tag_uuid])

    async def add_image_tags(self, image_uuid: str, tag_uuids: List[str]) -> None:
        """Set multiple whole-image tags to PRESENT in one transaction.

        Inserts a new PRESENT row for tags with no row, and flips any existing
        ABSENT row to PRESENT. No-op for tags already PRESENT.

        Two-pass design: existing-row flips are committed first so that an
        IntegrityError on the insert pass cannot roll them back.
        """
        if not tag_uuids:
            return

        # Pass 1: flip any existing non-PRESENT rows to PRESENT, committed
        # independently so a concurrent-insert failure in pass 2 cannot undo these.
        existing_tag_uuids: set = set()
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(ImageTagLink).where(
                    ImageTagLink.image_uuid == image_uuid,
                    ImageTagLink.tag_uuid.in_(tag_uuids),
                )
            )
            for row in result.all():
                existing_tag_uuids.add(row.tag_uuid)
                if row.status != ImageItemStatus.PRESENT:
                    row.status = ImageItemStatus.PRESENT
                    row.updated_at = datetime.now(timezone.utc)
            await session.commit()

        # Pass 2: insert rows that had no entry at all.
        to_add = [t for t in tag_uuids if t not in existing_tag_uuids]
        if not to_add:
            return
        async with self._async_session_maker() as session:
            for tag_uuid in to_add:
                session.add(
                    ImageTagLink(
                        image_uuid=image_uuid,
                        tag_uuid=tag_uuid,
                        status=ImageItemStatus.PRESENT,
                    )
                )
            try:
                await session.commit()
            except IntegrityError:
                # A concurrent caller inserted one of these rows between our SELECT and
                # INSERT.  Any status satisfies the PK constraint — only re-raise if the
                # row is still completely absent (which indicates an FK violation on
                # tag_uuid or image_uuid, not a harmless race).
                await session.rollback()
                result = await session.exec(
                    select(ImageTagLink.tag_uuid).where(
                        ImageTagLink.image_uuid == image_uuid,
                        ImageTagLink.tag_uuid.in_(to_add),
                    )
                )
                still_missing = set(to_add) - set(result.all())
                if still_missing:
                    raise

    async def remove_image_tag(self, image_uuid: str, tag_uuid: str) -> None:
        """Unlink a tag from a whole image. No-op if the link doesn't exist."""
        await self.remove_image_tags(image_uuid, [tag_uuid])

    async def remove_image_tags(self, image_uuid: str, tag_uuids: List[str]) -> None:
        """Delete the image_tags rows for the given tags (UNCERTAIN = no row). No-op for tags not linked."""
        if not tag_uuids:
            return
        async with self._async_session_maker() as session:
            await session.exec(
                delete(ImageTagLink).where(
                    ImageTagLink.image_uuid == image_uuid,
                    ImageTagLink.tag_uuid.in_(tag_uuids),
                )
            )
            await session.commit()
        # Core statement, not ORM — the after_flush listener never sees it. Same
        # reason `delete_images_bulk` bumps by hand; see the write-epoch comment above.
        self._bump_write_epoch()

    async def get_image_tag_uuids(self, image_uuid: str) -> List[str]:
        """Return all PRESENT tag UUIDs linked to a whole image."""
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(ImageTagLink.tag_uuid).where(
                    ImageTagLink.image_uuid == image_uuid,
                    ImageTagLink.status == ImageItemStatus.PRESENT,
                )
            )
            return list(result.all())

    async def get_image_uuids_by_tag(self, tag_uuid: str) -> Set[str]:
        """Return all image UUIDs carrying the given whole-image tag as PRESENT."""
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(ImageTagLink.image_uuid).where(
                    ImageTagLink.tag_uuid == tag_uuid,
                    ImageTagLink.status == ImageItemStatus.PRESENT,
                )
            )
            return set(result.all())

    async def set_image_tag_status(self, image_uuid: str, tag_uuid: str, status: ImageItemStatus) -> None:
        """Explicitly set PRESENT, ABSENT, or UNCERTAIN for a whole-image tag.

        Unlike image labels, all three states are user-settable here — whole-image
        tags have no derived-PRESENT concept to protect.
        """
        async with self._async_session_maker() as session:
            existing = await session.get(ImageTagLink, (image_uuid, tag_uuid))
            if status == ImageItemStatus.UNCERTAIN:
                if existing:
                    await session.delete(existing)
            elif existing:
                existing.status = status
                existing.updated_at = datetime.now(timezone.utc)
            else:
                session.add(ImageTagLink(image_uuid=image_uuid, tag_uuid=tag_uuid, status=status))
            await session.commit()

    async def get_image_tag_statuses(self, image_uuid: str) -> List[ImageTagLink]:
        """Return all image_tags rows (PRESENT and ABSENT) for a given image."""
        async with self._async_session_maker() as session:
            result = await session.exec(select(ImageTagLink).where(ImageTagLink.image_uuid == image_uuid))
            return result.all()

    ################################################################################
    # Task Configs
    ################################################################################

    async def _sync_task_configs_to_json(self):
        """Internal helper to dump the current task_configs table to task_configs.json.

        Configs marked_for_delete are excluded — they are logically deleted and
        restoring them from JSON would create confusion in the UI.
        """
        all_configs = await self.get_task_configs()
        active_configs = [c for c in all_configs.values() if not c.marked_for_delete]
        adapter = TypeAdapter(List[TaskConfig])
        json_str = adapter.dump_json(
            active_configs,
            exclude={"__all__": {"instances": True, "created_at": True, "updated_at": True}},
            indent=2,
        )
        self._task_configs_json_path.write_bytes(json_str)

    async def sync_task_configs_from_json(self):
        """Reads task_configs.json and syncs to DB. Does not remove existing configs.

        Resolution order per config:
        1. UUID match  → update the existing DB record in-place.
        2. Name match  → reuse the existing DB record (avoids unique-name IntegrityError).
        3. No match    → insert a new record with the UUID from the JSON.
        """
        if not self._task_configs_json_path.exists():
            return

        try:
            json_data = jstyleson.loads(self._task_configs_json_path.read_text())
        except Exception as e:
            logger.error(f"Failed to parse task_configs.json: {e}")
            return

        async with self._async_session_maker() as session:
            existing = (await session.exec(select(TaskConfig))).all()
            configs_by_uuid: Dict[str, TaskConfig] = {c.uuid: c for c in existing}
            configs_by_name: Dict[str, TaskConfig] = {c.name: c for c in existing}

            count = 0
            for item in json_data:
                json_uuid = item.get("uuid")
                name = item.get("name")

                if json_uuid and json_uuid in configs_by_uuid:
                    # 1. UUID match — update in-place.
                    db_config = configs_by_uuid[json_uuid]
                elif name and name in configs_by_name:
                    # 2. Name fallback — reuse DB record to avoid unique-name violation.
                    db_config = configs_by_name[name]
                    logger.warning(
                        "TaskConfig '%s' from JSON (UUID %s) matched by name to existing DB config "
                        "(UUID %s). Using DB UUID.",
                        name,
                        json_uuid,
                        db_config.uuid,
                    )
                else:
                    db_config = None

                if db_config is not None:
                    old_name = db_config.name
                    for key, value in item.items():
                        if key != "uuid" and hasattr(db_config, key):
                            setattr(db_config, key, value)
                    session.add(db_config)
                    # Keep the name index consistent if the JSON renamed this config.
                    if old_name != db_config.name:
                        configs_by_name.pop(old_name, None)
                        configs_by_name[db_config.name] = db_config
                else:
                    # 3. No match — insert new.
                    new_config = TaskConfig.model_validate(item)
                    session.add(new_config)
                    configs_by_uuid[new_config.uuid] = new_config
                    configs_by_name[new_config.name] = new_config
                count += 1

            await session.commit()
        logger.info("Imported %d task configs from JSON", count)

    async def add_task_config(
        self,
        name: str,
        typename: str,
        params: Dict[str, Any],
        description: Optional[str] = None,
        is_temporary: bool = False,
    ) -> TaskConfig:
        config = TaskConfig(
            name=name, typename=typename, params_json=params, description=description, is_temporary=is_temporary
        )
        async with self._async_session_maker() as session:
            session.add(config)
            await session.commit()
            await session.refresh(config)
        await self._sync_task_configs_to_json()
        return config

    async def update_task_config(
        self,
        config_uuid: str,
        name: Optional[str] = None,
        typename: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        marked_for_delete: Optional[bool] = None,
    ) -> None:
        async with self._async_session_maker() as session:
            config = await session.get(TaskConfig, config_uuid)
            if not config:
                return
            if name is not None:
                config.name = name
            if typename is not None:
                config.typename = typename
            if params is not None:
                config.params_json = params
            if description is not None:
                config.description = description
            if marked_for_delete is not None:
                config.marked_for_delete = marked_for_delete
            session.add(config)
            await session.commit()
        await self._sync_task_configs_to_json()

    async def delete_task_config(self, config_uuids: Union[str, List[str]]) -> None:
        if isinstance(config_uuids, str):
            config_uuids = [config_uuids]
        if not config_uuids:
            return
        async with self._async_session_maker() as session:
            # Delete configs with no task instances referencing them
            configs_with_instances = select(TaskInstance.config_uuid).where(TaskInstance.config_uuid.in_(config_uuids))
            await session.exec(
                delete(TaskConfig).where(
                    TaskConfig.uuid.in_(config_uuids),
                    ~TaskConfig.uuid.in_(configs_with_instances),
                )
            )
            # Mark the rest for deletion
            result = await session.exec(select(TaskConfig).where(TaskConfig.uuid.in_(config_uuids)))
            for config in result.all():
                config.marked_for_delete = True
                session.add(config)
            await session.commit()
        await self._sync_task_configs_to_json()

    async def get_task_configs(
        self,
        config_uuids: Optional[Union[str, List[str]]] = None,
        typename: Optional[Union[str, List[str]]] = None,
        name: Optional[Union[str, List[str]]] = None,
        is_temporary: Optional[bool] = None,
    ) -> Dict[str, TaskConfig]:
        stmt = select(TaskConfig)
        if config_uuids is not None:
            uuids = [config_uuids] if isinstance(config_uuids, str) else config_uuids
            stmt = stmt.where(TaskConfig.uuid.in_(uuids))
        if typename is not None:
            typenames = [typename] if isinstance(typename, str) else typename
            stmt = stmt.where(TaskConfig.typename.in_(typenames))
        if name is not None:
            names = [name] if isinstance(name, str) else name
            stmt = stmt.where(TaskConfig.name.in_(names))
        if is_temporary is not None:
            stmt = stmt.where(TaskConfig.is_temporary == is_temporary)
        stmt = stmt.order_by(TaskConfig.created_at.desc())
        async with self._async_session_maker() as session:
            result = await session.exec(stmt)
            configs = result.all()
        return {c.uuid: c for c in configs}

    async def get_tasks(
        self,
        task_uuid: Optional[Union[str, List[str]]] = None,
        config_uuid: Optional[Union[str, List[str]]] = None,
        typename: Optional[Union[str, List[str]]] = None,
        status: Optional[Union[str, List[str]]] = None,
        resumable: Optional[bool] = None,
        include_resume_data: bool = False,
    ) -> List[TaskInstance]:
        """Task instances, without their checkpoints unless asked for.

        ``resume_data_json`` is by far the heaviest column on the table — ~1 MB on a
        finished TrainModelTask — and this listing is polled twice a second by the task
        SSE stream. Omitting it from the *response* (see ``list_task_instances`` in
        db_server/web.py) was never enough on its own: the ORM still loaded every blob
        and ran ``json.loads`` over it on the event loop before the response model threw
        it away. Measured at ~60 ms of pure event-loop CPU per call against six finished
        training runs, which pegged db_server's loop at ~98% and made every other
        request queue behind it.

        Deferring it here is what actually stops the read. The filter below still works:
        a WHERE clause on a deferred column is evaluated by SQLite, not by the ORM.
        """
        stmt = select(TaskInstance).options(selectinload(TaskInstance.config))
        if not include_resume_data:
            stmt = stmt.options(defer(TaskInstance.resume_data_json))
        if task_uuid is not None:
            uuids = [task_uuid] if isinstance(task_uuid, str) else task_uuid
            stmt = stmt.where(TaskInstance.uuid.in_(uuids))
        if config_uuid is not None:
            cuuids = [config_uuid] if isinstance(config_uuid, str) else config_uuid
            stmt = stmt.where(TaskInstance.config_uuid.in_(cuuids))
        if typename is not None:
            typenames = [typename] if isinstance(typename, str) else typename
            stmt = stmt.join(TaskConfig).where(TaskConfig.typename.in_(typenames))
        if status is not None:
            statuses = [status] if isinstance(status, str) else status
            stmt = stmt.where(TaskInstance.status.in_(statuses))
        if resumable is True:
            stmt = stmt.where(TaskInstance.resume_data_json.is_not(None))
        elif resumable is False:
            stmt = stmt.where(TaskInstance.resume_data_json.is_(None))
        async with self._async_session_maker() as session:
            result = await session.exec(stmt)
            instances = result.all()
            if not include_resume_data:
                # These rows detach when the session closes, and reading a deferred
                # column on a detached instance raises DetachedInstanceError — which
                # TaskInstanceRead would walk straight into, since it reads
                # resume_data by its `resume_data_json` alias. Seed the attribute as
                # an already-loaded None so the read model sees the same shape it did
                # before, without a per-row SELECT to fetch what we just refused to load.
                for instance in instances:
                    set_committed_value(instance, "resume_data_json", None)
        return list(instances)

    async def get_task_results(self, task_uuids: Union[List[str], str]) -> Optional[Dict[str, Any]]:
        if isinstance(task_uuids, str):
            task_uuids = [task_uuids]
        stmt = select(TaskInstance).where(TaskInstance.uuid.in_(task_uuids))
        async with self._async_session_maker() as session:
            result = await session.exec(stmt)
            instances = result.all()
        if not instances:
            return None
        return {i.uuid: i.result_json for i in instances if i.result_json is not None}

    async def set_task_result(self, task_uuid: str, results: Dict[str, Any]) -> None:
        async with self._async_session_maker() as session:
            instance = await session.get(TaskInstance, task_uuid)
            if instance:
                instance.result_json = results
                session.add(instance)
                await session.commit()

    async def set_task_resume_data(self, task_uuid: str, resume_data: Dict[str, Any]) -> None:
        async with self._async_session_maker() as session:
            instance = await session.get(TaskInstance, task_uuid)
            if instance:
                instance.resume_data_json = resume_data
                session.add(instance)
                await session.commit()

    async def set_task_timestamps_and_logs(
        self,
        task_uuid: str,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
        log_json: Optional[List[str]] = None,
        status_data_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Each field is written only when supplied, so a caller that sends just one
        # (e.g. the start-of-run started_at write) can't blank the others. Built as a
        # values dict rather than mutating a loaded row for the same reason
        # update_task_heartbeat does: the row carries the checkpoint, and none of these
        # writes need to read it.
        values: Dict[str, Any] = {}
        if started_at is not None:
            values["started_at"] = started_at
            # A resume/auto-restart reuses the same row, and only the start of a run
            # writes started_at. Without this, the previous run's ended_at outlives it
            # and the instance reads as running while claiming to have ended *before*
            # it started. Callers never send both in one call, so this can't discard an
            # end being recorded — and if one ever does, the explicit ended_at below
            # still wins.
            values["ended_at"] = None
        if ended_at is not None:
            values["ended_at"] = ended_at
        if log_json is not None:
            values["log_json"] = log_json
        if status_data_json is not None:
            values["status_data_json"] = status_data_json
        if not values:
            return
        async with self._async_session_maker() as session:
            await session.exec(sa_update(TaskInstance).where(TaskInstance.uuid == task_uuid).values(**values))
            await session.commit()

    async def update_task_heartbeat(self, task_uuid: str) -> None:
        """Write the liveness timestamp without reading the row.

        session.get() loads the whole TaskInstance to write one column — including
        resume_data_json, which on a running training task is the largest value in the
        schema and grows with every checkpoint. This fires once per HEARTBEAT_INTERVAL
        per running task, so it was re-decoding a megabyte of checkpoint on a timer to
        record that the task was still alive. An UPDATE reads nothing; a missing row is
        a no-op either way, and updated_at still moves because its onupdate lives on the
        column rather than in the ORM layer.
        """
        async with self._async_session_maker() as session:
            await session.exec(
                sa_update(TaskInstance)
                .where(TaskInstance.uuid == task_uuid)
                .values(heartbeat_at=datetime.now(timezone.utc))
            )
            await session.commit()

    async def set_task_status(self, task_uuid: str, new_status: str, error_message: Optional[str] = None) -> None:
        """Set an instance's status, and the message explaining an ERROR.

        error_message is written on every call, not just when one is supplied:
        leaving a previous failure's text behind on a task that has since been
        resumed and is running again would misreport the live state.

        Completing a task drops its checkpoint. A checkpoint exists to restart an
        interrupted run, and only two statuses ever consult it: an interrupted RUNNING
        instance (its presence is what makes recovery choose paused-and-resumable over
        error) and a PAUSED one (resume_tasks only ever resumes those). A COMPLETED run
        has nothing left to resume, so keeping ~1 MB of dead checkpoint per finished
        training run only costs disk and decode time.

        Deliberately *not* cleared on ERROR: recovery marks an interrupted task error
        precisely when it had no checkpoint, and an errored row that does carry one is
        the evidence for why — clearing it there would destroy state the resumability
        logic reasons about, which is the failure mode NOTES.md warns about.
        """
        values: Dict[str, Any] = {
            "status": new_status,
            "error_message": error_message if new_status == TaskStatus.ERROR else None,
        }
        if new_status == TaskStatus.COMPLETED:
            values["resume_data_json"] = None
        async with self._async_session_maker() as session:
            await session.exec(sa_update(TaskInstance).where(TaskInstance.uuid == task_uuid).values(**values))
            await session.commit()

    async def insert_new_active_task(self, config_uuid: str) -> TaskInstance:
        instance = TaskInstance(config_uuid=config_uuid, status=TaskStatus.NEW)
        async with self._async_session_maker() as session:
            session.add(instance)
            await session.commit()
            await session.refresh(instance)
            stmt = (
                select(TaskInstance)
                .options(selectinload(TaskInstance.config))
                .where(TaskInstance.uuid == instance.uuid)
            )
            result = await session.exec(stmt)
            instance = result.one()
        return instance

    async def delete_active_tasks(self, task_uuids: Union[str, List[str]]) -> None:
        if isinstance(task_uuids, str):
            task_uuids = [task_uuids]
        if not task_uuids:
            return
        async with self._async_session_maker() as session:
            # Capture config UUIDs before deleting so we can GC marked_for_delete configs
            result = await session.exec(select(TaskInstance.config_uuid).where(TaskInstance.uuid.in_(task_uuids)))
            affected_config_uuids = list(set(result.all()))

            await session.exec(delete(TaskInstance).where(TaskInstance.uuid.in_(task_uuids)))

            # Delete any configs that were marked_for_delete or is_temporary and now have no remaining instances
            if affected_config_uuids:
                still_referenced = select(TaskInstance.config_uuid).where(
                    TaskInstance.config_uuid.in_(affected_config_uuids)
                )
                await session.exec(
                    delete(TaskConfig).where(
                        TaskConfig.uuid.in_(affected_config_uuids),
                        (TaskConfig.marked_for_delete == True) | (TaskConfig.is_temporary == True),  # type: ignore[arg-type]  # noqa: E712
                        ~TaskConfig.uuid.in_(still_referenced),
                    )
                )

            await session.commit()

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset CRUD
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dataset_to_read(d: Dataset) -> DatasetRead:
        return DatasetRead(
            uuid=d.uuid,
            name=d.name,
            description=d.description,
            filter_json=d.filter_json or {},
            label_uuids=[link.label_uuid for link in (d.label_links or [])],
            promoted=d.promoted,
            created_at=d.created_at,
            updated_at=d.updated_at,
        )

    async def list_datasets(self, include_views: bool = False) -> List[DatasetRead]:
        """List Datasets, excluding Views unless asked.

        Views and Datasets share this table (decision register #10). Callers
        overwhelmingly mean "a real, trainable Dataset" — the training-config
        picker, the model-config picker and the dataset checklist all do — so
        unpromoted rows are excluded by default. Only the Views UI passes
        include_views=True.

        A row whose filter_json no longer validates is **skipped, not raised**.
        filter_json is validated on read, and this listing feeds every dataset
        picker in the app, so one unconvertible row would otherwise 500 the
        whole response and take training configs, model configs and the dataset
        checklist down with it — which is exactly what a pre-1.4b `filter_json`
        did before migration b7d31c904ae2. Skipping loses one row loudly rather
        than every row silently. It is deliberately not coerced to a default
        filter: that would present a row as editable and overwrite the user's
        real (recoverable) filter on the next save.
        """
        async with self._async_session_maker() as session:
            stmt = select(Dataset).options(selectinload(Dataset.label_links))
            if not include_views:
                stmt = stmt.where(Dataset.promoted.is_(True))
            result = await session.exec(stmt)
            out: List[DatasetRead] = []
            for d in result.all():
                try:
                    out.append(self._dataset_to_read(d))
                except Exception as e:
                    logger.error(
                        "list_datasets: skipping dataset %s (%s) — filter_json does not validate: %s",
                        d.uuid,
                        d.name,
                        e,
                    )
            return out

    async def _sync_datasets_to_json(self) -> None:
        """Dump the current datasets to datasets.json — Views included.

        Views share this table with Datasets (see NOTES.md) and are just as
        much user-authored data: they replaced the per-browser filter presets.
        This file is the restore path for the table, not a "trainable Datasets"
        export, so excluding unpromoted rows would silently lose every saved
        View on a rebuild. `promoted` is written per row and honoured on load.

        Reads **raw rows**, not list_datasets(): that skips rows whose
        filter_json no longer validates, so dumping through it would drop such
        a row from the restore file and lose it on the next rebuild — trading
        a loud outage for silent data loss. filter_json is written back
        verbatim, so a row the reader can't yet understand survives until
        something migrates it.
        """
        async with self._async_session_maker() as session:
            stmt = select(Dataset).options(selectinload(Dataset.label_links))
            datasets = (await session.exec(stmt)).all()
            rows = [
                {
                    "uuid": d.uuid,
                    "name": d.name,
                    "description": d.description,
                    "filter_json": d.filter_json or {},
                    "label_uuids": [link.label_uuid for link in (d.label_links or [])],
                    "promoted": d.promoted,
                }
                for d in datasets
            ]
        self._datasets_json_path.write_text(json.dumps(rows, indent=2))

    async def sync_datasets_from_json(self) -> None:
        """Reads datasets.json and syncs to DB. Does not remove existing datasets.

        Resolution order per dataset:
        1. UUID match  → update the existing DB record in-place.
        2. Name match  → reuse the existing DB record (log UUID mismatch).
        3. No match    → insert a new record with the UUID from the JSON.

        Label links are replaced wholesale for any updated/inserted dataset.
        Label UUIDs that no longer exist in the labels table are silently skipped.
        """
        if not self._datasets_json_path.exists():
            return

        try:
            json_data = json.loads(self._datasets_json_path.read_text())
        except Exception as e:
            logger.error(f"Failed to parse datasets.json: {e}")
            return

        async with self._async_session_maker() as session:
            existing = (await session.exec(select(Dataset).options(selectinload(Dataset.label_links)))).all()
            by_uuid: Dict[str, Dataset] = {d.uuid: d for d in existing}
            by_name: Dict[str, Dataset] = {d.name: d for d in existing}

            # Collect valid label UUIDs so we can skip broken FK references.
            valid_labels = set((await session.exec(select(Label.uuid))).all())  # type: ignore[attr-defined]

            count = 0
            for item in json_data:
                json_uuid = item.get("uuid")
                name = item.get("name")
                label_uuids = [lu for lu in item.get("label_uuids", []) if lu in valid_labels]

                if json_uuid and json_uuid in by_uuid:
                    # 1. UUID match — update in-place.
                    db_ds = by_uuid[json_uuid]
                    for key, value in item.items():
                        if key not in ("uuid", "label_uuids") and hasattr(db_ds, key):
                            setattr(db_ds, key, value)
                    session.add(db_ds)
                    await session.exec(delete(DatasetLabelLink).where(DatasetLabelLink.dataset_uuid == db_ds.uuid))
                    for lu in label_uuids:
                        session.add(DatasetLabelLink(dataset_uuid=db_ds.uuid, label_uuid=lu))

                elif name and name in by_name:
                    # 2. Name fallback — reuse DB record to avoid unique-name violation.
                    db_ds = by_name[name]
                    logger.warning(
                        "Dataset '%s' from JSON (UUID %s) matched by name to existing "
                        "DB dataset (UUID %s). Using DB UUID.",
                        name,
                        json_uuid,
                        db_ds.uuid,
                    )
                    for key, value in item.items():
                        if key not in ("uuid", "label_uuids") and hasattr(db_ds, key):
                            setattr(db_ds, key, value)
                    session.add(db_ds)
                    await session.exec(delete(DatasetLabelLink).where(DatasetLabelLink.dataset_uuid == db_ds.uuid))
                    for lu in label_uuids:
                        session.add(DatasetLabelLink(dataset_uuid=db_ds.uuid, label_uuid=lu))

                else:
                    # 3. No match — insert new.
                    effective_uuid = json_uuid or str(uuid4())
                    ds = Dataset(
                        uuid=effective_uuid,
                        name=name,
                        description=item.get("description"),
                        filter_json=item.get("filter_json") or {},
                        # The dump writes `promoted` per row (Views included), so
                        # normally this key is present and authoritative. Sidecar
                        # files written before this column existed carry no key —
                        # default True, or a restore would silently demote every
                        # dataset to an invisible View.
                        promoted=bool(item.get("promoted", True)),
                    )
                    session.add(ds)
                    await session.flush()
                    for lu in label_uuids:
                        session.add(DatasetLabelLink(dataset_uuid=ds.uuid, label_uuid=lu))
                    by_uuid[ds.uuid] = ds
                    if ds.name:
                        by_name[ds.name] = ds

                count += 1

            await session.commit()
        logger.info("Imported %d datasets from JSON", count)

    async def get_dataset(self, uuid: str) -> Optional[DatasetRead]:
        async with self._async_session_maker() as session:
            stmt = select(Dataset).where(Dataset.uuid == uuid).options(selectinload(Dataset.label_links))
            result = await session.exec(stmt)
            d = result.one_or_none()
            return self._dataset_to_read(d) if d else None

    async def create_dataset(self, data: DatasetCreate) -> DatasetRead:
        async with self._async_session_maker() as session:
            dataset = Dataset(
                name=data.name,
                description=data.description,
                filter_json=data.filter_json.model_dump(mode="json"),
                promoted=data.promoted,
            )
            session.add(dataset)
            await session.flush()
            for label_uuid in data.label_uuids:
                session.add(
                    DatasetLabelLink(
                        dataset_uuid=dataset.uuid,
                        label_uuid=label_uuid,
                    )
                )
            await session.commit()
            stmt = select(Dataset).where(Dataset.uuid == dataset.uuid).options(selectinload(Dataset.label_links))
            result = await session.exec(stmt)
            read = self._dataset_to_read(result.one())
        await self._sync_datasets_to_json()
        return read

    async def update_dataset(self, uuid: str, data: DatasetUpdate) -> Optional[DatasetRead]:
        async with self._async_session_maker() as session:
            # Fetch without label_links — we'll replace them via bulk DELETE/INSERT,
            # so loading them into the session identity map first would cause
            # SQLAlchemy to complain about deleted-but-still-tracked objects.
            result = await session.exec(select(Dataset).where(Dataset.uuid == uuid))
            dataset = result.one_or_none()
            if dataset is None:
                return None

            changes = data.model_dump(exclude_unset=True, mode="json")
            label_uuids = changes.pop("label_uuids", None)

            # Refuse to demote a Dataset a training config still depends on.
            # The link row is ON DELETE RESTRICT and survives the flip, but an
            # unpromoted row drops out of the default listing — so the Dataset
            # would silently disappear from every training/model picker and the
            # next save of that training config would drop the link. Mirrors the
            # same check in delete_dataset().
            if changes.get("promoted") is False and dataset.promoted:
                link_result = await session.exec(
                    select(TrainingConfigDatasetLink).where(TrainingConfigDatasetLink.dataset_uuid == uuid)
                )
                if link_result.first() is not None:
                    raise ValueError(
                        f'Dataset "{dataset.name}" is referenced by one or more training configs '
                        f"and cannot be demoted to a View"
                    )

            for key, value in changes.items():
                if hasattr(dataset, key):
                    setattr(dataset, key, value)

            if label_uuids is not None:
                await session.exec(delete(DatasetLabelLink).where(DatasetLabelLink.dataset_uuid == uuid))
                for label_uuid in label_uuids:
                    session.add(
                        DatasetLabelLink(
                            dataset_uuid=uuid,
                            label_uuid=label_uuid,
                        )
                    )

            dataset.updated_at = datetime.now(timezone.utc)
            session.add(dataset)
            await session.commit()

            result = await session.exec(
                select(Dataset).where(Dataset.uuid == uuid).options(selectinload(Dataset.label_links))
            )
            read = self._dataset_to_read(result.one())
        await self._sync_datasets_to_json()
        return read

    async def delete_dataset(self, uuid: str) -> bool:
        async with self._async_session_maker() as session:
            dataset = await session.get(Dataset, uuid)
            if dataset is None:
                return False
            # Reject if any training config references this dataset (preserve old RESTRICT behavior).
            link_result = await session.exec(
                select(TrainingConfigDatasetLink).where(TrainingConfigDatasetLink.dataset_uuid == uuid)
            )
            if link_result.first() is not None:
                raise ValueError(f"Dataset {uuid} is referenced by one or more training configs")
            await session.delete(dataset)
            await session.commit()
        await self._sync_datasets_to_json()
        return True

    @staticmethod
    def build_dataset_search_params(
        dataset: DatasetRead,
        *,
        limit: int = 50,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
        after_uuid: Optional[str] = None,
        after_sort_value: Optional[datetime] = None,
    ) -> ImageSearchParams:
        """Expand a DatasetRead's filter criteria into an ImageSearchParams.

        f.terms (self-contained criteria: tags, dates, filename, whole-image
        tag/label conditions) are carried straight through. The label_uuids
        modifiers (label_match_mode / label_include_children / verification_check)
        parametrize the structurally separate dataset.label_uuids column
        (dataset_label_links), so they're expanded here into ObjectLabelTerm /
        LabelVerificationTerm entries appended to the term list.
        """
        f = dataset.filter_json
        terms: List[FilterTerm] = list(f.terms)

        if dataset.label_uuids:
            terms.append(
                ObjectLabelTerm(
                    label_uuids=dataset.label_uuids,
                    match_mode=f.label_match_mode,
                    roll_up=f.label_include_children,
                )
            )
        for excluded_uuid in f.label_exclude_uuids:
            terms.append(ObjectLabelTerm(label_uuids=[excluded_uuid], negate=True, roll_up=f.label_include_children))
        if f.verification_check:
            terms.append(LabelVerificationTerm(check=f.verification_check, label_uuids=dataset.label_uuids))

        return ImageSearchParams(
            terms=terms,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
            after_uuid=after_uuid,
            after_sort_value=after_sort_value,
        )

    async def _list_images_for_dataset(
        self,
        dataset: DatasetRead,
        *,
        limit: int = 50,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
        after_uuid: Optional[str] = None,
        after_sort_value: Optional[datetime] = None,
    ) -> tuple[list[ImageRecord], bool]:
        """Execute a DatasetRead's filter criteria and return (images, has_more).

        has_more is True if there may be additional results beyond this page.
        """
        # Fetch limit+1 rows so we can distinguish "exactly limit" from "more exist".
        params = self.build_dataset_search_params(
            dataset,
            limit=limit + 1,
            sort_by=sort_by,
            sort_order=sort_order,
            after_uuid=after_uuid,
            after_sort_value=after_sort_value,
        )
        results = await self.list_images_search(params)
        has_more = len(results) > limit
        return results[:limit], has_more

    async def _enumerate_dataset_union_stubs(
        self,
        dataset_uuids: List[str],
        *,
        mutate_params=None,
    ) -> list:
        """Enumerate the de-duplicated union of images across datasets as
        lightweight (uuid, filename, created_at, updated_at) rows.

        Selects only the columns the readiness/bucket endpoints actually use —
        no ORM entities, no selectinload of bboxes/labels/tags — so paging a
        large union costs plain row fetches instead of full object graphs
        (~5x per page on a 100k-image union). Callers hydrate just the final
        page via _get_images_with_relations().

        mutate_params, when given, is applied to each page's ImageSearchParams
        before execution (e.g. to add verification predicates).
        """
        seen: set[str] = set()
        stubs: list = []
        for ds_uuid in dataset_uuids:
            dataset = await self.get_dataset(ds_uuid)
            if dataset is None:
                continue
            after_uuid: Optional[str] = None
            while True:
                params = self.build_dataset_search_params(
                    dataset,
                    limit=_READINESS_PAGE_SIZE,
                    sort_by="uuid",
                    sort_order="asc",
                    after_uuid=after_uuid,
                    after_sort_value=None,
                )
                if mutate_params is not None:
                    mutate_params(params)
                use_exists = await self._should_use_exists_dispatch(params)
                async with self._async_session_maker() as session:
                    stmt = select(
                        ImageRecord.uuid,
                        ImageRecord.filename,
                        ImageRecord.created_at,
                        ImageRecord.updated_at,
                    )
                    stmt = _apply_image_search_filters(stmt, params, use_exists=use_exists, files_root=self._files_root)
                    if after_uuid is not None:
                        stmt = stmt.where(ImageRecord.uuid > after_uuid)
                    stmt = stmt.order_by(ImageRecord.uuid.asc()).limit(_READINESS_PAGE_SIZE)
                    page = (await session.exec(stmt)).all()
                for row in page:
                    if row.uuid not in seen:
                        seen.add(row.uuid)
                        stubs.append(row)
                if len(page) < _READINESS_PAGE_SIZE:
                    break
                after_uuid = page[-1].uuid
                # Yield between pages so long enumerations don't starve the loop.
                await asyncio.sleep(0)
        return stubs

    async def _get_images_with_relations(self, image_uuids: List[str]) -> list[ImageRecord]:
        """Fetch full ImageRecords (bboxes, labels, tags eagerly loaded) for a
        page of uuids, preserving the input order.

        Intended for page-sized lists (the final page of a stub enumeration);
        images deleted since enumeration are silently dropped.
        """
        if not image_uuids:
            return []
        async with self._async_session_maker() as session:
            stmt = (
                select(ImageRecord)
                .options(
                    selectinload(ImageRecord.bboxes).options(
                        selectinload(BBox.label),
                        selectinload(BBox.tags),
                    ),
                    selectinload(ImageRecord.tags),
                )
                .where(ImageRecord.uuid.in_(image_uuids))
            )
            rows = (await session.exec(stmt)).all()
        by_uuid = {img.uuid: img for img in rows}
        return [by_uuid[u] for u in image_uuids if u in by_uuid]

    async def _get_image_label_verification_counts(
        self, image_uuids: List[str], label_uuids: List[str]
    ) -> tuple[Dict[str, int], Dict[str, int]]:
        """Return ({image: verified_count}, {image: present_count}) over the
        given images and labels.

        verified counts PRESENT+ABSENT image_label rows; present counts
        PRESENT only. The uuid list is chunked (_UUID_IN_CHUNK) so unbounded
        dataset unions stay under the SQLite bind-variable cap; chunks are
        disjoint, so merging the per-chunk GROUP BY results is a plain update.
        """
        verified_by_image: Dict[str, int] = {}
        present_by_image: Dict[str, int] = {}
        async with self._async_session_maker() as session:
            for chunk in _chunked(image_uuids):
                result = await session.exec(
                    select(ImageLabel.image_uuid, func.count(ImageLabel.label_uuid))
                    .where(
                        ImageLabel.image_uuid.in_(chunk),
                        ImageLabel.label_uuid.in_(label_uuids),
                        ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                    )
                    .group_by(ImageLabel.image_uuid)
                )
                verified_by_image.update(dict(result.all()))
                result = await session.exec(
                    select(ImageLabel.image_uuid, func.count(ImageLabel.label_uuid))
                    .where(
                        ImageLabel.image_uuid.in_(chunk),
                        ImageLabel.label_uuid.in_(label_uuids),
                        ImageLabel.status == ImageItemStatus.PRESENT,
                    )
                    .group_by(ImageLabel.image_uuid)
                )
                present_by_image.update(dict(result.all()))
        return verified_by_image, present_by_image

    async def list_dataset_images(
        self,
        uuid: str,
        *,
        limit: int = 50,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
        after_uuid: Optional[str] = None,
        after_sort_value: Optional[datetime] = None,
    ) -> Optional[tuple[list[ImageRecord], bool]]:
        """Execute a dataset's filter criteria and return (images, has_more).

        Returns None if the dataset doesn't exist.
        """
        dataset = await self.get_dataset(uuid)
        if dataset is None:
            return None
        return await self._list_images_for_dataset(
            dataset,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
            after_uuid=after_uuid,
            after_sort_value=after_sort_value,
        )

    # ── Training configs ──────────────────────────────────────────────────────

    @staticmethod
    def _training_config_to_read(tc: TrainingConfig) -> TrainingConfigRead:
        return TrainingConfigRead(
            uuid=tc.uuid,
            name=tc.name,
            description=tc.description,
            model_config_node_uuid=tc.model_config_node_uuid,
            dataset_uuids=[link.dataset_uuid for link in (tc.dataset_links or [])],
            hyperparams_json=tc.hyperparams_json or {},
            created_at=tc.created_at,
            updated_at=tc.updated_at,
        )

    async def create_training_config(self, data: TrainingConfigCreate) -> TrainingConfigRead:
        async with self._async_session_maker() as session:
            tc = TrainingConfig(
                name=data.name,
                description=data.description,
                model_config_node_uuid=data.model_config_node_uuid,
                hyperparams_json=data.hyperparams_json or {},
            )
            session.add(tc)
            await session.flush()  # assign uuid before adding links
            for dataset_uuid in data.dataset_uuids or []:
                session.add(TrainingConfigDatasetLink(training_config_uuid=tc.uuid, dataset_uuid=dataset_uuid))
            await session.commit()
            await session.refresh(tc, ["dataset_links"])
            return self._training_config_to_read(tc)

    async def list_training_configs(self) -> list[TrainingConfigRead]:
        async with self._async_session_maker() as session:
            stmt = select(TrainingConfig).options(*_TC_LOAD_OPTS)
            result = await session.exec(stmt)
            return [self._training_config_to_read(tc) for tc in result.all()]

    async def get_training_config(self, uuid: str) -> Optional[TrainingConfigRead]:
        async with self._async_session_maker() as session:
            stmt = select(TrainingConfig).where(TrainingConfig.uuid == uuid).options(*_TC_LOAD_OPTS)
            result = await session.exec(stmt)
            tc = result.first()
            return self._training_config_to_read(tc) if tc else None

    async def update_training_config(self, uuid: str, data: TrainingConfigUpdate) -> Optional[TrainingConfigRead]:
        async with self._async_session_maker() as session:
            stmt = select(TrainingConfig).where(TrainingConfig.uuid == uuid).options(*_TC_LOAD_OPTS)
            result = await session.exec(stmt)
            tc = result.first()
            if tc is None:
                return None
            if data.name is not None:
                tc.name = data.name
            if data.description is not None:
                tc.description = data.description
            if data.model_config_node_uuid is not None:
                tc.model_config_node_uuid = data.model_config_node_uuid
            if data.hyperparams_json is not None:
                tc.hyperparams_json = data.hyperparams_json
            if data.dataset_uuids is not None:
                for link in list(tc.dataset_links or []):
                    await session.delete(link)
            await session.flush()
            if data.dataset_uuids is not None:
                for dataset_uuid in data.dataset_uuids:
                    session.add(TrainingConfigDatasetLink(training_config_uuid=tc.uuid, dataset_uuid=dataset_uuid))
            tc.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(tc, ["dataset_links"])
            return self._training_config_to_read(tc)

    async def delete_training_config(self, uuid: str) -> bool:
        async with self._async_session_maker() as session:
            tc = await session.get(TrainingConfig, uuid)
            if tc is None:
                return False
            await session.delete(tc)
            await session.commit()
        return True

    @staticmethod
    def _sort_labels_hierarchically(uuids: list[str], labels_map: dict) -> list[str]:
        """Return label UUIDs sorted depth-first: parents before children, siblings by name.

        Used to give consistent hierarchical display when the order is otherwise arbitrary
        (e.g. the fallback path that unions label UUIDs from datasets).
        """
        uuid_set = set(uuids)
        children_of: dict[Optional[str], list[str]] = {}
        for uuid in uuids:
            label = labels_map.get(uuid)
            parent = label.parent_uuid if label else None
            # If the parent is not in our set, treat as a top-level entry.
            effective_parent = parent if (parent in uuid_set) else None
            children_of.setdefault(effective_parent, []).append(uuid)

        # Sort siblings alphabetically by name for deterministic output.
        for sibling_list in children_of.values():
            sibling_list.sort(key=lambda u: (labels_map[u].name.lower() if u in labels_map else u))

        result: list[str] = []

        def _visit(uuid: str) -> None:
            result.append(uuid)
            for child in children_of.get(uuid, []):
                _visit(child)

        for root_uuid in children_of.get(None, []):
            _visit(root_uuid)

        return result

    async def _derive_labels_for_node(
        self, model_config_node_uuid: Optional[str]
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Return (label_uuids, rollup_map).

        label_uuids: all relevant labels in depth-first hierarchical order
            (parents appear before their rollup children).
        rollup_map: maps each rollup-parent label UUID to the list of child label
            UUIDs that aggregate into it.  Empty dict = no rollup active.

        - model_config_node_uuid is None  → ([], {})
        - Root node (label_uuid=None): returns all non-root node label UUIDs.
          For nodes with roll_up_unchecked=True, appends that label's hierarchy
          descendants (those without their own model config node) right after the
          parent and records them in rollup_map.
        - Specific node: returns that label + all descendants in DFS order.
          If roll_up_unchecked=True, records all descendants in rollup_map under
          that label.
        """
        if model_config_node_uuid is None:
            return [], {}

        async with self._async_session_maker() as session:
            node = await session.get(ModelConfigNode, model_config_node_uuid)
            if node is None:
                return [], {}

            if node.label_uuid is None:
                # ── Root node ────────────────────────────────────────────────
                result = await session.exec(
                    select(ModelConfigNode).where(
                        ModelConfigNode.model_config_uuid == node.model_config_uuid,
                        ModelConfigNode.label_uuid.isnot(None),
                    )
                )
                non_root_nodes = list(result.all())
                base_label_uuids = [n.label_uuid for n in non_root_nodes]
                rollup_nodes = [n for n in non_root_nodes if n.roll_up_unchecked]

                # Load full label hierarchy: needed to find rollup children, and to
                # order the result depth-first. The query above returns rows in
                # arbitrary order, and callers (the coverage table) indent by label
                # depth, so unsorted output renders children detached from parents.
                all_labels_result = await session.exec(select(Label))
                all_labels = {lbl.uuid: lbl for lbl in all_labels_result.all()}

                if not rollup_nodes:
                    return DbClient._sort_labels_hierarchically(base_label_uuids, all_labels), {}

            else:
                # ── Specific node ─────────────────────────────────────────────
                is_rollup = node.roll_up_unchecked
                all_labels_result = await session.exec(select(Label))
                all_labels = {lbl.uuid: lbl for lbl in all_labels_result.all()}

        # ── Outside the session ───────────────────────────────────────────────

        # Index children by parent once so descendant collection is O(descendants)
        # instead of re-scanning every label at each recursion level.
        children_by_parent: dict[Optional[str], list[Label]] = {}
        for lbl in all_labels.values():
            children_by_parent.setdefault(lbl.parent_uuid, []).append(lbl)
        # Siblings alphabetical, matching _sort_labels_hierarchically, so descendant
        # order is deterministic rather than whatever the label table returned.
        for siblings in children_by_parent.values():
            siblings.sort(key=lambda lbl: lbl.name.lower())

        def _collect_label_descendants(parent_uuid: str, exclude_set: set) -> list[str]:
            """Recursively collect label-hierarchy descendants, skipping those in exclude_set."""
            result: list[str] = []
            for lbl in children_by_parent.get(parent_uuid, []):
                if lbl.uuid not in exclude_set:
                    result.append(lbl.uuid)
                # Always recurse so we pick up grandchildren of excluded nodes too.
                result.extend(_collect_label_descendants(lbl.uuid, exclude_set))
            return result

        if node.label_uuid is None:
            # Root case: build expanded list; rollup children appear right after parent.
            base_label_set = set(base_label_uuids)
            rollup_label_set = {n.label_uuid for n in rollup_nodes}

            def _label_depth(uuid: str) -> int:
                depth = 0
                current = all_labels.get(uuid)
                while current is not None and current.parent_uuid is not None:
                    depth += 1
                    current = all_labels.get(current.parent_uuid)
                return depth

            # Precompute depths for rollup labels once rather than calling the
            # traversal O(n log n) times during sort.
            rollup_depths = {uuid: _label_depth(uuid) for uuid in rollup_label_set}

            # Assign each rollup descendant to its NEAREST rollup ancestor by processing
            # rollup parents deepest-first: a label shared between nested rollup parents
            # (e.g. A→B→C with both A and B roll_up_unchecked) is claimed by the closest
            # one (B) and never double-counted. The source query order is unspecified, so
            # depth — not row order — drives the assignment.
            rollup_map: dict[str, list[str]] = {}
            claimed_as_rollup: set[str] = set()
            for parent_uuid in sorted(rollup_label_set, key=rollup_depths.__getitem__, reverse=True):
                children = _collect_label_descendants(parent_uuid, base_label_set)
                fresh_children = [u for u in children if u not in claimed_as_rollup]
                if fresh_children:
                    rollup_map[parent_uuid] = fresh_children
                    claimed_as_rollup.update(fresh_children)

            # Display order: base labels depth-first, each rollup parent's claimed
            # children inserted right after it.
            expanded: list[str] = []
            for lbl_uuid in DbClient._sort_labels_hierarchically(base_label_uuids, all_labels):
                expanded.append(lbl_uuid)
                expanded.extend(rollup_map.get(lbl_uuid, []))

            return expanded, rollup_map

        else:
            # Specific-node case: this label + all descendants (reuse helper, no exclusions).
            all_uuids = [node.label_uuid] + _collect_label_descendants(node.label_uuid, set())
            if not is_rollup:
                return all_uuids, {}
            rollup_map = {node.label_uuid: [u for u in all_uuids if u != node.label_uuid]}
            return all_uuids, rollup_map

    async def derive_training_targets(self, model_config_uuid: str) -> Optional[list[dict]]:
        """Derive the ordered list of training targets a full training run must produce.

        Single authority on sub-model boundaries and per-target label scope /
        class remapping (the frontend heuristic in ``ui/modelSpec.js`` mirrors the
        boundary rule for display only).

        A non-root node is a sub-model *boundary* when it says so: ``is_boundary``.
        That used to be inferred from "(model_type, architecture) differs from the
        inherited spec", which tied nesting depth to architecture choice — a sub-model
        could not reuse its parent's architecture without dissolving back into it, so
        `Animal(efficientnet_b0) → Cat(efficientnet_b0)` silently produced one model
        rather than two. The column is backfilled from that rule, so configs predating
        it derive exactly the targets they always did.

        Returns None if the model config doesn't exist. Otherwise a list of targets
        (top model first, then sub-models in hierarchical label order), each:
          - node_uuid, role ("top" | "submodel"), label_uuid (None for top)
          - model_type, architecture (the node's own spec)
          - label_uuids: ordered class label UUIDs this artifact predicts
          - class_names: label names aligned with label_uuids
          - class_remap: {in-scope label uuid -> class label uuid} for labels whose
            bboxes train as another class. A label in neither label_uuids nor
            class_remap is out of scope for that target.
          - suppression_group_uuids / suppression_group_iou_thresholds: aligned with
            label_uuids. A class's group is its nearest exclusive_variants=True
            ancestor-or-self; group == the class's own label_uuid means ungrouped.
            Consumed by the trainer's sidecar export so inference-time NMS can suppress
            same-group classes together instead of by raw class_id.

        Class-set semantics:
          - Top model: assigned labels with no boundary ancestor; every other
            assigned label chases up to its *outermost* boundary (nested boundaries
            collapse all the way, so one visual concept never trains under two
            different top classes).
          - Sub-model at boundary B: assigned labels whose nearest proper boundary
            ancestor is B (what B's model distinguishes). B itself is not a class of
            its own sub-model; a childless boundary yields an empty class list.
          - roll_up_unchecked descendants use the root-scope rollup assignment from
            _derive_labels_for_node (nearest rollup parent claims them) and remap to
            wherever their rollup parent resolves in each target.

        A config with no root node yields no targets ([]).
        """
        async with self._async_session_maker() as session:
            config = await session.get(ModelConfig, model_config_uuid)
            if config is None:
                return None
            result = await session.exec(
                select(ModelConfigNode).where(ModelConfigNode.model_config_uuid == model_config_uuid)
            )
            nodes = result.all()
            labels_result = await session.exec(select(Label))
            all_labels = {lbl.uuid: lbl for lbl in labels_result.all()}

        root = next((n for n in nodes if n.label_uuid is None), None)
        if root is None:
            return []
        non_root = [n for n in nodes if n.label_uuid is not None]

        def _parent(uuid: str) -> Optional[str]:
            lbl = all_labels.get(uuid)
            return lbl.parent_uuid if lbl is not None else None

        def _depth(uuid: str) -> int:
            depth, p = 0, _parent(uuid)
            while p is not None:
                depth += 1
                p = _parent(p)
            return depth

        # Boundaries are declared, not inferred. `is_boundary` is backfilled from the
        # old "spec differs from inherited" rule, so this is the same set for every
        # config that predates the column — but architecture is now free to repeat down
        # a chain, which that rule made impossible (a sub-model could never use its
        # parent's architecture without dissolving into it).
        boundaries = {n.label_uuid for n in non_root if n.is_boundary}
        # Sorted for the depth walk below, which needs ancestors before descendants.
        non_root.sort(key=lambda n: _depth(n.label_uuid))

        def _boundary_ancestor(uuid: str) -> Optional[str]:
            p = _parent(uuid)
            while p is not None:
                if p in boundaries:
                    return p
                p = _parent(p)
            return None

        assigned = [n.label_uuid for n in non_root]
        node_by_label = {n.label_uuid: n for n in non_root}

        _, rollup_map = await self._derive_labels_for_node(root.uuid)

        def _resolve(uuid: str, class_set: set) -> Optional[str]:
            """Nearest ancestor-or-self of `uuid` that is a class of the target."""
            current: Optional[str] = uuid
            while current is not None:
                if current in class_set:
                    return current
                current = _parent(current)
            return None

        # Built once for the shared suppression_group_for() — see
        # apps/helpers/suppressionGroups.py for why this computation is factored out
        # rather than inlined here.
        _parent_of = {uuid: lbl.parent_uuid for uuid, lbl in all_labels.items()}
        _exclusive_variants_of = {uuid: lbl.exclusive_variants for uuid, lbl in all_labels.items()}
        _threshold_of = {uuid: lbl.exclusive_variants_iou_threshold for uuid, lbl in all_labels.items()}

        def _suppression_group(uuid: str) -> tuple[str, Optional[float]]:
            return suppression_group_for(uuid, _parent_of, _exclusive_variants_of, _threshold_of)

        def _build_target(node: ModelConfigNode, role: str, class_uuids: list[str]) -> dict:
            class_set = set(class_uuids)
            ordered = self._sort_labels_hierarchically(class_uuids, all_labels)
            class_remap: dict[str, str] = {}
            for a in assigned:
                cls = _resolve(a, class_set)
                if cls is not None and cls != a:
                    class_remap[a] = cls
            for parent_uuid, children in rollup_map.items():
                cls = _resolve(parent_uuid, class_set)
                if cls is None:
                    continue
                for child in children:
                    class_remap[child] = cls
            suppression_groups = [_suppression_group(lu) for lu in ordered]
            return {
                "node_uuid": node.uuid,
                "role": role,
                "label_uuid": node.label_uuid,
                "model_type": node.model_type,
                "architecture": node.architecture,
                "label_uuids": ordered,
                "class_names": [all_labels[lu].name if lu in all_labels else lu for lu in ordered],
                "class_remap": class_remap,
                # Parallel to label_uuids/class_names. group_uuid == the class's own
                # label_uuid means "not grouped" (today's per-class_id NMS behavior).
                "suppression_group_uuids": [grp for grp, _thr in suppression_groups],
                "suppression_group_iou_thresholds": [thr for _grp, thr in suppression_groups],
            }

        top_classes = [a for a in assigned if _boundary_ancestor(a) is None]
        targets = [_build_target(root, "top", top_classes)]
        for b_uuid in self._sort_labels_hierarchically(list(boundaries), all_labels):
            sub_classes = [a for a in assigned if a != b_uuid and _boundary_ancestor(a) == b_uuid]
            targets.append(_build_target(node_by_label[b_uuid], "submodel", sub_classes))
        return targets

    async def get_training_config_readiness(
        self, uuid: str, restrict_to_image_uuids: Optional[list[str]] = None
    ) -> Optional[dict]:
        """Return ready/needs-review/not-started counts for a training config's datasets (union).

        An image is 'ready' when every *class* is accounted for — a verified bbox or
        an explicit ABSENT for an object-scope class, any explicit PRESENT/ABSENT for
        an image-scope one. 'needs_review' is partial coverage plus something
        positive; 'not_started' is everything else, including images whose only
        decisions are ABSENT, since they haven't surfaced any in-scope class yet.

        ready_count is further split into positive/negative samples: an image is a
        POSITIVE sample when at least one class is positive on it, and a NEGATIVE
        (background) sample when it is fully accounted for and none are. The split
        matters because a dataset can be overwhelmingly background — 'ready' alone
        reads as far more training signal than actually exists.
        ready_positive_count + ready_negative_count == ready_count.

        Rollup children are not classes: they annotate their parent, so verifying one
        accounts for the parent and the parent alone. This is emphatically NOT "any
        one verified label makes the image ready", which is what rollup used to mean
        here — under that rule an image could be ready with most of its classes never
        looked at, and the images went to the trainer as though those classes were
        absent. See _class_coverage.

        *restrict_to_image_uuids* narrows every count to that subset of the dataset
        union (None = the whole union). Mirrors get_training_config_label_stats:
        the launcher and the training page replay these counts over a previously
        trained model's recorded manifest, so "what went in last time" and "what is
        here now" come out of the same code. Intersected with the union rather than
        used raw — an image the run trained on but that has since left the dataset
        is not part of this config's baseline any more.

        Returns None if the training config doesn't exist.
        """
        tc = await self.get_training_config(uuid)
        if tc is None:
            return None
        dataset_uuids = tc.dataset_uuids

        label_uuids, class_uuids, effective_for_class, labels_map, is_rollup = await self._readiness_class_shape(
            tc.model_config_node_uuid
        )

        if not label_uuids:
            return {
                "ready_count": 0,
                "ready_positive_count": 0,
                "ready_negative_count": 0,
                "needs_review_count": 0,
                "not_started_count": 0,
                "dataset_uuids": dataset_uuids,
                "label_uuids": [],
                "roll_up_active": False,
            }

        counts = await self._readiness_counts(
            dataset_uuids,
            class_uuids,
            effective_for_class,
            labels_map,
            restrict_to_image_uuids,
        )
        return {
            **counts,
            "dataset_uuids": dataset_uuids,
            "label_uuids": label_uuids,
            "roll_up_active": is_rollup,
        }

    async def _readiness_class_shape(
        self, model_config_node_uuid: Optional[str]
    ) -> tuple[List[str], List[str], Dict[str, List[str]], Dict[str, Any], bool]:
        """(label_uuids, class_uuids, effective_for_class, labels_map, is_rollup).

        A rollup child is not a class of its own — it annotates its parent — so the
        classes are the label list minus every rollup child, and each class carries
        the raw label UUIDs that can speak for it.
        """
        label_uuids, rollup_map = await self._derive_labels_for_node(model_config_node_uuid)
        if not label_uuids:
            return [], [], {}, {}, False

        rollup_children = {child for children in rollup_map.values() for child in children}
        class_uuids = [lu for lu in label_uuids if lu not in rollup_children]
        effective_for_class = {cu: [cu, *rollup_map.get(cu, [])] for cu in class_uuids}

        async with self._async_session_maker() as session:
            rows = (await session.exec(select(Label).where(Label.uuid.in_(class_uuids)))).all()
        return label_uuids, class_uuids, effective_for_class, {lbl.uuid: lbl for lbl in rows}, bool(rollup_map)

    async def _class_coverage(
        self,
        image_uuids: List[str],
        class_uuids: List[str],
        effective_for_class: Dict[str, List[str]],
        labels_map: Dict[str, Any],
    ) -> tuple[Dict[str, int], set]:
        """(image -> how many classes are accounted for, images carrying a positive).

        The per-image form of ``_class_coverage_sources``, for the callers that hold a
        list of image UUIDs rather than a dataset-union select. Same two subqueries, so
        this cannot drift from what readiness counts.
        """
        if not class_uuids or not image_uuids:
            return {}, set()

        coverage_sel, positives_sel = self._class_coverage_sources(class_uuids, effective_for_class, labels_map)
        coverage = coverage_sel.subquery()
        positives_sq = positives_sel.subquery()

        covered: Dict[str, int] = {}
        positives: set = set()
        async with self._async_session_maker() as session:
            for chunk in _chunked(image_uuids):
                rows = (
                    await session.exec(
                        select(coverage.c.image_uuid, coverage.c.covered).where(coverage.c.image_uuid.in_(chunk))
                    )
                ).all()
                covered.update(dict(rows))
                positives.update(
                    (
                        await session.exec(
                            select(positives_sq.c.image_uuid).where(positives_sq.c.image_uuid.in_(chunk))
                        )
                    ).all()
                )
        return covered, positives

    def _readiness_bucket(self, covered: int, total_classes: int, positive: bool) -> str:
        """The bucket an image falls in. One definition, so the counts, the browse
        lists and the trainer's manifest cannot disagree about the same image.

        An image whose only decisions are ABSENT has surfaced no in-scope class, so
        it reads as not_started rather than needs_review — partial coverage alone is
        not enough to claim someone started reviewing it.
        """
        if covered >= total_classes:
            return "ready_positive" if positive else "ready_negative"
        if covered > 0 and positive:
            return "needs_review"
        return "not_started"

    async def _readiness_scope_selects(self, dataset_uuids: List[str]) -> List[Select]:
        """One `SELECT images.uuid` per dataset, carrying that dataset's filters.

        A dataset is a saved search rather than a membership table, so its images
        can only be named by re-running its filter — which is why this cannot be a
        join against a link table.
        """
        selects: List[Select] = []
        for ds_uuid in dataset_uuids:
            dataset = await self.get_dataset(ds_uuid)
            if dataset is None:
                continue
            params = self.build_dataset_search_params(dataset, sort_by="uuid", sort_order="asc")
            # use_exists=False deliberately. The correlated-EXISTS shape exists so an
            # ORDER BY + LIMIT scan can stop early, and an aggregate over the whole
            # union has no LIMIT to stop at — the IN form is the safe one here.
            selects.append(
                _apply_image_search_filters(
                    select(ImageRecord.uuid),
                    params,
                    use_exists=False,
                    files_root=self._files_root,
                )
            )
        return selects

    @staticmethod
    def _class_coverage_sources(
        class_uuids: List[str],
        effective_for_class: Dict[str, List[str]],
        labels_map: Dict[str, Any],
    ) -> tuple[Select, Select]:
        """(coverage, positives) as `SELECT image_uuid, covered` / `SELECT image_uuid`.

        The same rule ``_class_coverage`` applies in Python, expressed as SQL so
        readiness can stay a single aggregate over the dataset union rather than
        enumerating it (see TestReadinessIsAnAggregateNotAnEnumeration — the
        enumeration cost ~0.6s of event-loop CPU per call, paid twice per config).

        Both halves are keyed on the *class* a raw label speaks for, not the label,
        which is the whole correction: rollup means more labels than classes, and a
        count of verified labels cannot tell which classes they landed on.

        ``coverage`` is pre-aggregated to one row per image (COUNT DISTINCT class_idx)
        rather than left as the raw per-class rows. Left as raw rows, joining it against
        the (much larger) dataset scope gives SQLite a many-rows-per-image side it
        refuses to build an automatic index for — a full nested-loop scan, O(images ×
        coverage rows). On a real config that was 3.1k images × 10.3k coverage rows =
        32M comparisons for a 2.5s query. Pre-aggregating first makes both sides of the
        join one-row-per-image, which SQLite auto-indexes, at ~30ms.
        """
        class_idx = {cu: i for i, cu in enumerate(class_uuids)}
        raw_to_idx: Dict[str, int] = {}
        for class_uuid in class_uuids:
            if labels_map.get(class_uuid) is None:
                continue
            for raw in effective_for_class[class_uuid]:
                raw_to_idx[raw] = class_idx[class_uuid]

        all_raw = set(raw_to_idx)
        verified_boxes = select(BBoxTagLink.bbox_uuid).where(BBoxTagLink.tag_uuid == VERIFIED_TAG_UUID)

        il_case = case(raw_to_idx, value=ImageLabel.label_uuid)
        bbox_case = case(raw_to_idx, value=BBox.label_uuid)

        coverage_raw = union(
            select(ImageLabel.image_uuid.label("image_uuid"), il_case.label("class_idx")).where(
                ImageLabel.label_uuid.in_(all_raw),
                ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
            ),
            select(BBox.image_uuid.label("image_uuid"), bbox_case.label("class_idx")).where(
                BBox.label_uuid.in_(all_raw),
                BBox.uuid.in_(verified_boxes),
            ),
        ).subquery()
        coverage = select(
            coverage_raw.c.image_uuid.label("image_uuid"),
            func.count(distinct(coverage_raw.c.class_idx)).label("covered"),
        ).group_by(coverage_raw.c.image_uuid)

        # Positive = the image has surfaced a class, by either medium. Scope is not the
        # discriminator it looks like: object-scope labels are routinely trained by an
        # image_classifier node with no boxes drawn at all, so requiring a box here
        # would read those whole configs as background.
        positives = union(
            select(ImageLabel.image_uuid.label("image_uuid")).where(
                ImageLabel.label_uuid.in_(all_raw),
                ImageLabel.status == ImageItemStatus.PRESENT,
            ),
            select(BBox.image_uuid.label("image_uuid")).where(
                BBox.label_uuid.in_(all_raw),
                BBox.uuid.in_(verified_boxes),
            ),
        )

        return coverage, positives

    async def _readiness_counts(
        self,
        dataset_uuids: List[str],
        class_uuids: List[str],
        effective_for_class: Dict[str, List[str]],
        labels_map: Dict[str, Any],
        restrict_to_image_uuids: Optional[list[str]] = None,
    ) -> Dict[str, int]:
        """The five readiness buckets, counted by SQLite rather than in Python.

        The bucket definitions mirror ``_readiness_bucket`` (which the list endpoints
        use) one for one:

        - ready         every class accounted for, split by whether anything is positive
        - needs_review  partially covered AND something positive (an image whose only
                        decisions are ABSENT has surfaced no class yet, so it is
                        not started)
        - not_started   everything else, including images with no rows at all — hence
                        the OUTER joins

        `not_started` is derived as the remainder rather than counted, so the buckets
        cannot drift apart from the total no matter what the predicates do.
        """
        empty = {
            "ready_count": 0,
            "ready_positive_count": 0,
            "ready_negative_count": 0,
            "needs_review_count": 0,
            "not_started_count": 0,
        }
        if not class_uuids:
            return empty

        scope_selects = await self._readiness_scope_selects(dataset_uuids)
        if not scope_selects:
            return empty
        if restrict_to_image_uuids is not None and not restrict_to_image_uuids:
            return empty

        # UNION (not UNION ALL) de-duplicates across datasets; a single dataset's
        # filter can still fan out over a join, so that case needs DISTINCT itself.
        scope = (scope_selects[0].distinct() if len(scope_selects) == 1 else union(*scope_selects)).subquery()

        coverage_sel, positives_sel = self._class_coverage_sources(class_uuids, effective_for_class, labels_map)
        coverage = coverage_sel.subquery()
        positives = positives_sel.subquery()

        # coverage and positives are both pre-aggregated to one row per image, so this
        # join is 1:1:1 — no GROUP BY needed, and (critically) each side is small enough
        # for SQLite to auto-index instead of falling back to a nested-loop scan of the
        # dataset scope against the raw per-class rows.
        per_image = select(
            scope.c.uuid.label("uuid"),
            func.coalesce(coverage.c.covered, 0).label("covered"),
            case((positives.c.image_uuid.isnot(None), 1), else_=0).label("positive"),
        ).select_from(
            scope.outerjoin(coverage, coverage.c.image_uuid == scope.c.uuid).outerjoin(
                positives, positives.c.image_uuid == scope.c.uuid
            )
        )

        # Chunked because a restrict list can exceed SQLite's bind-variable cap.
        # Chunks are disjoint and every image lands in exactly one bucket, so the
        # per-chunk totals add.
        chunks: List[Optional[Sequence[str]]] = (
            [None] if restrict_to_image_uuids is None else list(_chunked(list(restrict_to_image_uuids)))
        )

        n_classes = len(class_uuids)
        totals = dict(empty)
        for chunk in chunks:
            stmt = per_image if chunk is None else per_image.where(scope.c.uuid.in_(chunk))
            sq = stmt.subquery()
            is_ready = sq.c.covered >= n_classes
            is_positive = sq.c.positive > 0
            agg = select(
                func.count().label("total"),
                func.sum(case((is_ready, 1), else_=0)).label("ready"),
                func.sum(case((and_(is_ready, is_positive), 1), else_=0)).label("ready_positive"),
                func.sum(case((and_(is_ready, ~is_positive), 1), else_=0)).label("ready_negative"),
                func.sum(case((and_(sq.c.covered < n_classes, sq.c.covered > 0, is_positive), 1), else_=0)).label(
                    "needs_review"
                ),
            ).select_from(sq)

            async with self._async_session_maker() as session:
                row = (await session.exec(agg)).one()

            total = row.total or 0
            ready = row.ready or 0
            needs_review = row.needs_review or 0
            totals["ready_count"] += ready
            totals["ready_positive_count"] += row.ready_positive or 0
            totals["ready_negative_count"] += row.ready_negative or 0
            totals["needs_review_count"] += needs_review
            totals["not_started_count"] += total - ready - needs_review

        return totals

    async def get_training_config_detection_dataset(self, uuid: str) -> Optional[dict]:
        """Assemble a YOLO-format detection dataset for an object_detection training config.

        Returns a dict with:
          - class_names: list[str]  — ordered class names (parents before rollup children)
          - images: list[dict] each with image_uuid, filename, and
              annotations: list of [class_idx, cx, cy, w, h] (YOLO-normalized floats)

        Only verified bboxes (carrying VERIFIED_TAG_UUID) whose label is in scope are
        emitted. Images with no eligible bboxes appear with annotations=[] (negative
        samples). Returns None if the training config doesn't exist.
        """
        tc = await self.get_training_config(uuid)
        if tc is None:
            return None

        label_uuids, rollup_map = await self._derive_labels_for_node(tc.model_config_node_uuid)
        if not label_uuids:
            return {"class_names": [], "images": []}

        # Build reverse map: every label_uuid → class index.
        # Rollup children share their parent's index.
        child_to_parent: dict[str, str] = {}
        for parent_uuid, children in rollup_map.items():
            for child_uuid in children:
                child_to_parent[child_uuid] = parent_uuid

        # Build class list in label_uuids order (parents before rollup children).
        # Rollup children are NOT separate classes; they map to their parent's class.
        class_label_uuids: list[str] = []
        seen_class: set[str] = set()
        for lu in label_uuids:
            effective = child_to_parent.get(lu, lu)
            if effective not in seen_class:
                class_label_uuids.append(effective)
                seen_class.add(effective)

        return await self._assemble_detection_dataset(class_label_uuids, child_to_parent, tc.dataset_uuids)

    async def get_training_target_detection_dataset(
        self,
        model_config_uuid: str,
        node_uuid: str,
        dataset_uuids: list[str],
        label_uuids: Optional[list[str]] = None,
        class_remap: Optional[dict[str, str]] = None,
    ) -> Optional[dict]:
        """Assemble a YOLO-format detection dataset for one training target.

        Same output shape as ``get_training_config_detection_dataset``, but scoped by
        ``derive_training_targets`` (the single authority on sub-model boundaries
        and class remapping) rather than a TrainingConfig's own node lookup — so a
        training run's object-detection sub-models get exactly the classes/remap shown
        in the model tree preview, including the outermost-boundary top-model
        collapsing that per-TrainingConfig detection datasets don't apply.

        When ``label_uuids`` is given (a caller-held snapshot of that same authority,
        e.g. a caller's targets resolved at _init), derivation is skipped and the
        snapshot scope is used directly — so a multi-target run trains against one
        consistent snapshot even if the config is edited mid-run.

        Returns None if the config doesn't exist or no target matches node_uuid.
        """
        if label_uuids is not None:
            return await self._assemble_detection_dataset(label_uuids, dict(class_remap or {}), dataset_uuids)

        targets = await self.derive_training_targets(model_config_uuid)
        if targets is None:
            return None
        target = next((t for t in targets if t["node_uuid"] == node_uuid), None)
        if target is None:
            return None

        return await self._assemble_detection_dataset(target["label_uuids"], target["class_remap"], dataset_uuids)

    async def get_dataset_detection_dataset(
        self,
        dataset_uuid: str,
        *,
        require_verification: bool = True,
    ) -> Optional[dict]:
        """Assemble a detection dataset for one Dataset row, for export.

        Classes are the Dataset's own labels, in hierarchical order. A box carrying
        a label that is a *descendant* of a selected label but is not itself selected
        rolls up into its nearest selected ancestor's class — the same collapsing the
        model tree does, and the only reading that doesn't silently drop boxes when a
        Dataset selects "Cat" and the annotations say "Cat > Tabby".

        Deliberately independent of the Dataset filter's ``labelIncludeChildren``:
        that flag decides which *images* the Dataset contains, while this decides
        what the boxes on an included image are called. A descendant box left
        unannotated on an image that ships anyway is a false negative — the one
        outcome worth ruling out.

        Same output shape as ``get_training_config_detection_dataset``. Returns None
        if the dataset doesn't exist.
        """
        dataset = await self.get_dataset(dataset_uuid)
        if dataset is None:
            return None

        selected = list(dataset.label_uuids or [])
        if not selected:
            return {"class_names": [], "images": []}

        async with self._async_session_maker() as session:
            result = await session.exec(select(Label))
            all_labels = {lbl.uuid: lbl for lbl in result.all()}

        class_label_uuids = DbClient._sort_labels_hierarchically(selected, all_labels)
        selected_set = set(class_label_uuids)

        # Walk each unselected label up to its nearest selected ancestor.
        remap: dict[str, str] = {}
        for uuid, lbl in all_labels.items():
            if uuid in selected_set:
                continue
            parent = lbl.parent_uuid
            while parent is not None:
                if parent in selected_set:
                    remap[uuid] = parent
                    break
                parent_label = all_labels.get(parent)
                parent = parent_label.parent_uuid if parent_label is not None else None

        return await self._assemble_detection_dataset(
            class_label_uuids, remap, [dataset_uuid], require_verification=require_verification
        )

    async def _assemble_detection_dataset(
        self,
        class_label_uuids: list[str],
        remap: dict[str, str],
        dataset_uuids: list[str],
        *,
        require_verification: bool = True,
    ) -> dict:
        """Shared core: given an ordered class list + {in-scope label -> class label}
        remap, assemble YOLO-format annotations from verified bboxes across a dataset
        union. See ``get_training_config_detection_dataset`` for the output shape.

        ``require_verification=False`` drops the "every in-scope class must carry a
        PRESENT/ABSENT decision on this image" gate and emits every image in the
        union instead. Training must never do that — an unreviewed image with no
        boxes is indistinguishable from a true negative and teaches the model to
        miss — but an *export* is allowed to hand over the raw contents on request.
        """
        # class_names are fetched from the DB; we need the label names in order.
        async with self._async_session_maker() as session:
            result = await session.exec(select(Label).where(Label.uuid.in_(class_label_uuids)))
            labels_by_uuid = {lbl.uuid: lbl for lbl in result.all()}

        label_uuid_to_class_idx: dict[str, int] = {lu: i for i, lu in enumerate(class_label_uuids)}
        # Also map remapped (e.g. rollup child) labels to their class's index.
        for child_uuid, parent_uuid in remap.items():
            label_uuid_to_class_idx[child_uuid] = label_uuid_to_class_idx[parent_uuid]

        class_names = [labels_by_uuid[lu].name if lu in labels_by_uuid else lu for lu in class_label_uuids]

        in_scope_label_uuids: set[str] = set(class_label_uuids) | set(remap.keys())

        all_images = await self._enumerate_dataset_union_stubs(dataset_uuids)

        if not all_images:
            return {"class_names": class_names, "images": []}

        image_uuids = [img.uuid for img in all_images]

        if require_verification:
            # Every class must be accounted for individually. Counting verified rows
            # instead — which is what this did — passes an image whose rows all sit on
            # a couple of classes while the rest have never been looked at, and those
            # images then train as though the unreviewed classes were absent.
            effective_for_class: dict[str, list[str]] = {cu: [cu] for cu in class_label_uuids}
            for child_uuid, parent_uuid in remap.items():
                effective_for_class[parent_uuid].append(child_uuid)

            covered, _ = await self._class_coverage(image_uuids, class_label_uuids, effective_for_class, labels_by_uuid)
            eligible_images = [img for img in all_images if covered.get(img.uuid, 0) >= len(class_label_uuids)]
        else:
            eligible_images = list(all_images)

        # Batch-fetch all verified in-scope bboxes for eligible images (chunked
        # to stay under the SQLite bind-variable cap).
        eligible_image_uuids = [img.uuid for img in eligible_images]
        raw_bboxes: list = []
        async with self._async_session_maker() as session:
            for chunk in _chunked(eligible_image_uuids):
                result = await session.exec(
                    select(
                        BBox.image_uuid,
                        BBox.label_uuid,
                        BBox.x,
                        BBox.y,
                        BBox.width,
                        BBox.height,
                    ).where(
                        BBox.image_uuid.in_(chunk),
                        BBox.label_uuid.in_(list(in_scope_label_uuids)),
                        BBox.uuid.in_(select(BBoxTagLink.bbox_uuid).where(BBoxTagLink.tag_uuid == VERIFIED_TAG_UUID)),
                    )
                )
                raw_bboxes.extend(result.all())

        # Group by image_uuid for O(1) per-image lookup.
        bboxes_by_image: dict[str, list] = {}
        for row in raw_bboxes:
            bboxes_by_image.setdefault(row[0], []).append(row)

        image_entries = []
        for img in eligible_images:
            annotations = []
            for row in bboxes_by_image.get(img.uuid, []):
                _, lbl_uuid, x, y, w, h = row
                cls_idx = label_uuid_to_class_idx[lbl_uuid]
                # Convert (x1, y1, w, h) relative coords → YOLO (cx, cy, w, h).
                annotations.append([cls_idx, x + w / 2, y + h / 2, w, h])
            image_entries.append(
                {
                    "image_uuid": img.uuid,
                    "filename": img.filename,
                    "annotations": annotations,
                }
            )

        return {"class_names": class_names, "images": image_entries}

    async def get_training_target_crop_dataset(
        self,
        model_config_uuid: str,
        node_uuid: str,
        dataset_uuids: list[str],
        label_uuids: Optional[list[str]] = None,
        class_remap: Optional[dict[str, str]] = None,
    ) -> Optional[dict]:
        """Assemble a crop-classification dataset for one training target.

        The crop_classifier counterpart to ``get_training_target_detection_dataset``:
        scoped by ``derive_training_targets`` (or a caller-held snapshot of it, same
        contract as the detection variant) and returning one entry per image with the
        in-scope verified bboxes to cut out of it.

        Returns None if the config doesn't exist or no target matches node_uuid.
        """
        if label_uuids is not None:
            return await self._assemble_crop_dataset(label_uuids, dict(class_remap or {}), dataset_uuids)

        targets = await self.derive_training_targets(model_config_uuid)
        if targets is None:
            return None
        target = next((t for t in targets if t["node_uuid"] == node_uuid), None)
        if target is None:
            return None

        return await self._assemble_crop_dataset(target["label_uuids"], target["class_remap"], dataset_uuids)

    async def _assemble_crop_dataset(
        self, class_label_uuids: list[str], remap: dict[str, str], dataset_uuids: list[str]
    ) -> dict:
        """Shared core: given an ordered class list + {in-scope label -> class label}
        remap, collect the verified bboxes to train a crop classifier on.

        Returns:
            class_names:       ordered class names (index = the model's output unit)
            class_label_uuids: the same classes as label UUIDs
            images:            [{image_uuid, filename, crops: [{bbox_uuid, class_idx,
                               x, y, width, height}]}] — relative coords, as stored.

        Unlike ``_assemble_detection_dataset`` this applies **no image-level readiness
        gate**. A detection model has to know that everything unboxed in an image is
        background, so it can only train on images where every in-scope label was
        explicitly decided. A crop classifier never sees the rest of the image: each
        sample is one already-verified box, and its class is that box's own label. The
        readiness threshold would only discard usable crops — on the live DB it drops
        ~1500 verified cat boxes to a few dozen, because nobody ever marks the rare
        sibling labels explicitly absent (see the readiness note in NOTES.md).

        Images with no in-scope verified bbox are omitted entirely rather than emitted
        as negatives: there is no "background" class to assign them to.
        """
        async with self._async_session_maker() as session:
            result = await session.exec(select(Label).where(Label.uuid.in_(class_label_uuids)))
            labels_by_uuid = {lbl.uuid: lbl for lbl in result.all()}

        label_uuid_to_class_idx: dict[str, int] = {lu: i for i, lu in enumerate(class_label_uuids)}
        for child_uuid, parent_uuid in remap.items():
            if parent_uuid in label_uuid_to_class_idx:
                label_uuid_to_class_idx[child_uuid] = label_uuid_to_class_idx[parent_uuid]

        class_names = [labels_by_uuid[lu].name if lu in labels_by_uuid else lu for lu in class_label_uuids]
        empty = {"class_names": class_names, "class_label_uuids": list(class_label_uuids), "images": []}

        if not class_label_uuids:
            return empty

        # Only labels that actually resolve to a class index are in scope. A remap
        # entry pointing outside this target's class set (the top model's collapse
        # map, handed to a sub-model) must not drag its labels in.
        in_scope_label_uuids = list(label_uuid_to_class_idx)

        all_images = await self._enumerate_dataset_union_stubs(dataset_uuids)
        if not all_images:
            return empty

        raw_bboxes: list = []
        async with self._async_session_maker() as session:
            for chunk in _chunked([img.uuid for img in all_images]):
                result = await session.exec(
                    select(
                        BBox.image_uuid,
                        BBox.uuid,
                        BBox.label_uuid,
                        BBox.x,
                        BBox.y,
                        BBox.width,
                        BBox.height,
                    ).where(
                        BBox.image_uuid.in_(chunk),
                        BBox.label_uuid.in_(in_scope_label_uuids),
                        BBox.uuid.in_(select(BBoxTagLink.bbox_uuid).where(BBoxTagLink.tag_uuid == VERIFIED_TAG_UUID)),
                    )
                )
                raw_bboxes.extend(result.all())

        crops_by_image: dict[str, list] = {}
        for image_uuid, bbox_uuid, lbl_uuid, x, y, w, h in raw_bboxes:
            crops_by_image.setdefault(image_uuid, []).append(
                {
                    "bbox_uuid": bbox_uuid,
                    "class_idx": label_uuid_to_class_idx[lbl_uuid],
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                }
            )

        image_entries = [
            {"image_uuid": img.uuid, "filename": img.filename, "crops": crops_by_image[img.uuid]}
            for img in all_images
            if img.uuid in crops_by_image
        ]
        return {
            "class_names": class_names,
            "class_label_uuids": list(class_label_uuids),
            "images": image_entries,
        }

    async def _derive_co_label_shape(self, model_config_node_uuid: Optional[str]):
        """Resolve a model-config node into the rollup-aware "co-label shape" used by
        every co-label computation.

        Returns (labels_map, top_level_uuids, effective_for_tl, child_to_parent) where:
            labels_map        {label_uuid: Label} for every label in the node's shape
            top_level_uuids   labels that are not a rollup child — the units a gap is counted against
            effective_for_tl  {top_level_uuid: [top_level_uuid, *rollup_children]}
            child_to_parent   {rollup_child_uuid: top_level_uuid}

        Returns empty structures when the node resolves to no labels.
        """
        node_label_uuids, rollup_map = await self._derive_labels_for_node(model_config_node_uuid)
        if not node_label_uuids:
            return {}, [], {}, {}

        labels_map = await self.get_labels_by_uuids(node_label_uuids)

        child_to_parent: dict[str, str] = {}
        for _parent, _children in rollup_map.items():
            for _child in _children:
                child_to_parent[_child] = _parent

        top_level_uuids = [u for u in node_label_uuids if u not in child_to_parent and u in labels_map]
        effective_for_tl = {tl: [tl] + rollup_map.get(tl, []) for tl in top_level_uuids}
        return labels_map, top_level_uuids, effective_for_tl, child_to_parent

    async def get_training_config_image_gaps(
        self, uuid: str, image_uuids: list[str]
    ) -> dict[str, list[dict[str, str]]]:
        """Per-image list of the training config's labels that are still unresolved.

        For each requested image, returns the top-level labels of the config's model shape
        that are NOT yet "accounted for" (see _co_label_accounted_for: a verified bbox or an
        explicit Absent for object-scope labels; an explicit Present or Absent for
        image-scope ones). This is the same notion of "gap" the coverage table counts, but
        sliced per image instead of per label, so a review card can show what that one image
        still needs.

        Scoped to the passed image uuids — callers batch one call per rendered page rather
        than per card, and the underlying queries only ever touch those images.

        Images with no gaps are omitted from the result entirely.
        """
        tc = await self.get_training_config(uuid)
        if tc is None or not image_uuids:
            return {}

        labels_map, top_level_uuids, effective_for_tl, _ = await self._derive_co_label_shape(tc.model_config_node_uuid)
        if not top_level_uuids:
            return {}

        async with self._async_session_maker() as session:
            accounted_for = await self._co_label_accounted_for(
                session, list(image_uuids), top_level_uuids, effective_for_tl, labels_map
            )

        gaps: dict[str, list[dict[str, str]]] = {}
        for image_uuid in image_uuids:
            missing = []
            # Iterate top_level_uuids (not the dict) so the order is the model shape's own
            # order — the card renders these in a capped row, and a stable order keeps the
            # same labels from shuffling between page renders.
            for tl_uuid in top_level_uuids:
                if image_uuid in accounted_for.get(tl_uuid, set()):
                    continue
                label = labels_map.get(tl_uuid)
                if label is None:
                    continue
                missing.append(
                    {
                        "label_uuid": tl_uuid,
                        "label_name": label.name,
                        "label_scope": label.scope.value,
                    }
                )
            if missing:
                gaps[image_uuid] = missing
        return gaps

    async def _co_label_accounted_for(
        self,
        session,
        all_image_uuids: list[str],
        top_level_uuids: list[str],
        effective_for_tl: dict[str, list[str]],
        tl_label_lookup: dict[str, Any],
        verified_bbox_sets: Optional[dict[str, set]] = None,
    ) -> dict[str, set]:
        """Images "accounted for" per top-level label, shared by the co-label-gap
        filter (get_training_config_images_by_label_status) and the co-label stats
        aggregation (get_training_config_label_stats):

        object-scope: has a verified bbox OR an explicit absent ImageLabel (for the
            label or its rollup children).
        image-scope:  has any explicit PRESENT or ABSENT ImageLabel (ditto).

        verified_bbox_sets: optional precomputed {label_uuid: verified-image-set} for
        object-scope labels, so a caller that already queried those sets (the stats
        aggregation, which needs them for other purposes too) doesn't re-query them.

        Batches its queries across all of top_level_uuids (grouped by what they
        actually need — ABSENT-only ImageLabel rows for object-scope labels,
        PRESENT-or-ABSENT rows for everything else, plus a verified-bbox query for
        any object-scope label not already covered by verified_bbox_sets) instead of
        querying once per label: with dozens of top-level labels, a per-label loop
        turns into dozens of sequential round trips for no reason, since a single
        query with label_uuid IN (...) returns exactly the same rows.
        """
        object_tls = [
            tl
            for tl in top_level_uuids
            if tl_label_lookup.get(tl) is not None and tl_label_lookup[tl].scope == LabelScope.OBJECT
        ]
        other_tls = [tl for tl in top_level_uuids if tl_label_lookup.get(tl) is not None and tl not in object_tls]

        # Raw label UUIDs needed per ImageLabel status: ABSENT-only backs the
        # object-scope "accounted for" check, PRESENT-or-ABSENT backs everything else.
        absent_only_raw = {u for tl in object_tls for u in effective_for_tl[tl]}
        present_or_absent_raw = {u for tl in other_tls for u in effective_for_tl[tl]}
        all_il_raw = absent_only_raw | present_or_absent_raw

        raw_absent: dict[str, set] = defaultdict(set)
        raw_present_or_absent: dict[str, set] = defaultdict(set)
        if all_il_raw:
            for chunk in _chunked(all_image_uuids):
                rows = (
                    await session.exec(
                        select(ImageLabel.label_uuid, ImageLabel.image_uuid, ImageLabel.status).where(
                            ImageLabel.image_uuid.in_(chunk),
                            ImageLabel.label_uuid.in_(all_il_raw),
                            ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                        )
                    )
                ).all()
                for raw_lbl_uuid, img_uuid, status in rows:
                    raw_present_or_absent[raw_lbl_uuid].add(img_uuid)
                    if status == ImageItemStatus.ABSENT:
                        raw_absent[raw_lbl_uuid].add(img_uuid)

        # Verified-bbox rows for object-scope top-levels not already supplied.
        need_bbox_query = [tl for tl in object_tls if not (verified_bbox_sets is not None and tl in verified_bbox_sets)]
        raw_verified_bbox: dict[str, set] = defaultdict(set)
        if need_bbox_query:
            needed_raw = {u for tl in need_bbox_query for u in effective_for_tl[tl]}
            for chunk in _chunked(all_image_uuids):
                rows = (
                    await session.exec(
                        select(BBox.label_uuid, BBox.image_uuid)
                        .distinct()
                        .where(
                            BBox.image_uuid.in_(chunk),
                            BBox.label_uuid.in_(needed_raw),
                            BBox.uuid.in_(
                                select(BBoxTagLink.bbox_uuid).where(BBoxTagLink.tag_uuid == VERIFIED_TAG_UUID)
                            ),
                        )
                    )
                ).all()
                for raw_lbl_uuid, img_uuid in rows:
                    raw_verified_bbox[raw_lbl_uuid].add(img_uuid)

        accounted_for: dict[str, set] = {}
        for tl_uuid in top_level_uuids:
            eff = effective_for_tl[tl_uuid]
            tl_label = tl_label_lookup.get(tl_uuid)
            if tl_label is None:
                accounted_for[tl_uuid] = set()
                continue
            if tl_label.scope == LabelScope.OBJECT:
                if verified_bbox_sets is not None and tl_uuid in verified_bbox_sets:
                    verified_bboxes = verified_bbox_sets[tl_uuid]
                else:
                    verified_bboxes = set().union(*(raw_verified_bbox.get(u, set()) for u in eff))
                verified_absent = set().union(*(raw_absent.get(u, set()) for u in eff))
                accounted_for[tl_uuid] = verified_bboxes | verified_absent
            else:
                accounted_for[tl_uuid] = set().union(*(raw_present_or_absent.get(u, set()) for u in eff))
        return accounted_for

    async def get_training_config_images_by_label_status(
        self,
        uuid: str,
        label_uuid: str,
        status: Optional[str],
        *,
        rollup_label_uuids: Optional[List[str]] = None,
        missing_label_uuid: Optional[str] = None,
        limit: int = 50,
        after_uuid: Optional[str] = None,
        after_sort_value: Optional[datetime] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> tuple[list["ImageRecord"], bool]:
        """Return paginated images from a training config's datasets filtered by image-label status.

        status: 'present', 'absent', 'uncertain' (no image_label row), or None/'all' for all images.
        rollup_label_uuids: when the label is a rollup parent, pass its children's UUIDs here so the
            query spans the full rollup group (parent + children), matching the aggregated counts shown
            in the stats panel.
        missing_label_uuid: only used when status == 'co_label_gaps'. Narrows the gap set to images
            where this specific other top-level label is the one left unaccounted for, matching a single
            row of the co-label gap breakdown instead of the full (union of all) gap set.
        """
        tc = await self.get_training_config(uuid)
        if tc is None:
            return [], False

        # Collect all images from the training config's datasets (union, de-duped).
        all_images = await self._enumerate_dataset_union_stubs(tc.dataset_uuids)

        if not all_images:
            return [], False

        all_image_uuids = [img.uuid for img in all_images]

        # Effective label UUIDs: when rollup children are provided, query across
        # the parent + all children so the results match the rolled-up stat counts.
        effective_label_uuids = [label_uuid] + (rollup_label_uuids or [])

        # Filter by status.
        if status and status != "all":
            async with self._async_session_maker() as session:

                async def _collect(build_stmt) -> set:
                    """Union results across uuid chunks (SQLite bind-variable cap)."""
                    out: set[str] = set()
                    for chunk in _chunked(all_image_uuids):
                        out.update((await session.exec(build_stmt(chunk))).all())
                    return out

                if status == "uncertain":
                    # Images with no PRESENT/ABSENT row for any of the effective labels.
                    verified = await _collect(
                        lambda chunk: select(ImageLabel.image_uuid).where(
                            ImageLabel.image_uuid.in_(chunk),
                            ImageLabel.label_uuid.in_(effective_label_uuids),
                            ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                        )
                    )
                    matching = set(all_image_uuids) - verified
                elif status == "has_verified_bbox":
                    # Images with at least one verified bbox for any of the effective labels.
                    matching = await _collect(
                        lambda chunk: select(BBox.image_uuid)
                        .distinct()
                        .where(
                            BBox.image_uuid.in_(chunk),
                            BBox.label_uuid.in_(effective_label_uuids),
                            BBox.uuid.in_(
                                select(BBoxTagLink.bbox_uuid).where(BBoxTagLink.tag_uuid == VERIFIED_TAG_UUID)
                            ),
                        )
                    )
                elif status in ("co_labels_covered", "co_label_gaps"):
                    # Derive all labels in the training config to build the co-label picture.
                    (
                        co_labels_map,
                        top_level_uuids_co,
                        effective_for_tl_co,
                        child_to_parent_co,
                    ) = await self._derive_co_label_shape(tc.model_config_node_uuid)
                    if not top_level_uuids_co:
                        matching = set()
                    else:
                        # Base: images with ≥1 verified bbox for the focus label (effective set).
                        focus_eff = effective_for_tl_co.get(label_uuid, effective_label_uuids)
                        base = await _collect(
                            lambda chunk: select(BBox.image_uuid)
                            .distinct()
                            .where(
                                BBox.image_uuid.in_(chunk),
                                BBox.label_uuid.in_(focus_eff),
                                BBox.uuid.in_(
                                    select(BBoxTagLink.bbox_uuid).where(BBoxTagLink.tag_uuid == VERIFIED_TAG_UUID)
                                ),
                            )
                        )

                        # Compute "accounted for" per other top-level label, then intersect.
                        focus_tl_ancestor = child_to_parent_co.get(label_uuid, label_uuid)
                        other_tls_co = [tl for tl in top_level_uuids_co if tl != focus_tl_ancestor]

                        accounted_for_co = await self._co_label_accounted_for(
                            session, all_image_uuids, other_tls_co, effective_for_tl_co, co_labels_map
                        )

                        co_covered = set(base)
                        for accounted in accounted_for_co.values():
                            co_covered &= accounted

                        if status == "co_labels_covered":
                            matching = co_covered
                        elif missing_label_uuid:
                            if missing_label_uuid not in accounted_for_co:
                                # The caller asked to narrow to a specific other top-level
                                # label, but it isn't one of this label's current co-label
                                # breakdown rows (stale UI state, or a label config change
                                # since the breakdown was computed). Fail loudly instead of
                                # silently widening back out to the full (unfiltered) gap set.
                                raise ValueError(
                                    f"missing_label_uuid {missing_label_uuid!r} is not a co-label of {label_uuid!r}"
                                )
                            # Narrow the gap to images where this specific other label is unresolved.
                            matching = base - accounted_for_co[missing_label_uuid]
                        else:
                            matching = base - co_covered
                else:
                    target = ImageItemStatus.PRESENT if status == "present" else ImageItemStatus.ABSENT
                    matching = await _collect(
                        lambda chunk: select(ImageLabel.image_uuid).where(
                            ImageLabel.image_uuid.in_(chunk),
                            ImageLabel.label_uuid.in_(effective_label_uuids),
                            ImageLabel.status == target,
                        )
                    )
            all_images = [img for img in all_images if img.uuid in matching]

        if not all_images:
            return [], False

        def _sort_key(img: "ImageRecord"):
            return (getattr(img, sort_by), img.uuid)

        all_images.sort(key=_sort_key, reverse=(sort_order == "desc"))

        # Apply keyset cursor.
        if after_uuid is not None:
            start_idx = _keyset_start_index(
                all_images,
                after_uuid=after_uuid,
                after_sort_value=after_sort_value,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            if start_idx is None:
                return [], False
            all_images = all_images[start_idx:]

        has_more = len(all_images) > limit
        page = await self._get_images_with_relations([img.uuid for img in all_images[:limit]])
        return page, has_more

    async def _images_needing_label_review(
        self,
        image_uuids: List[str],
        image_label_uuids: List[str],
        object_label_uuids: List[str],
    ) -> set[str]:
        """Return the subset of image_uuids that still need review for any of the given
        target labels — image-scope labels checked against image_labels rows, object-scope
        labels checked against bbox UNVERIFIED tags. See get_training_config_needs_review_images.
        """
        needs_review: set[str] = set()
        if not image_uuids:
            return needs_review

        async with self._async_session_maker() as session:
            if image_label_uuids:
                # Count present/absent image_labels rows per image, across the target
                # image-scope labels only. An image needs review if that count is short
                # of the full target-label count (i.e. at least one is unassessed).
                verified_counts: dict[str, int] = {}
                for chunk in _chunked(image_uuids):
                    rows = (
                        await session.exec(
                            select(ImageLabel.image_uuid, func.count().label("cnt"))
                            .where(
                                ImageLabel.image_uuid.in_(chunk),
                                ImageLabel.label_uuid.in_(image_label_uuids),
                                ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                            )
                            .group_by(ImageLabel.image_uuid)
                        )
                    ).all()
                    for img_uuid, cnt in rows:
                        verified_counts[img_uuid] = cnt
                target_count = len(image_label_uuids)
                for img_uuid in image_uuids:
                    if verified_counts.get(img_uuid, 0) < target_count:
                        needs_review.add(img_uuid)

            if object_label_uuids:
                # Images with at least one bbox for a target object-scope label that
                # still carries the UNVERIFIED tag.
                for chunk in _chunked(image_uuids):
                    rows = (
                        await session.exec(
                            select(BBox.image_uuid)
                            .join(BBoxTagLink, BBox.uuid == BBoxTagLink.bbox_uuid)
                            .where(
                                BBox.image_uuid.in_(chunk),
                                BBox.label_uuid.in_(object_label_uuids),
                                BBoxTagLink.tag_uuid == UNVERIFIED_TAG_UUID,
                            )
                            .distinct()
                        )
                    ).all()
                    needs_review.update(rows)

        return needs_review

    # Mirrors NO_SOURCE_KEY in shared/api/review.js — the client sends this sentinel
    # inside sourceUuids/sourceExcludeUuids for "no source recorded" rather than a real
    # source uuid, since NULL never satisfies an IN clause on its own.
    _NO_SOURCE_KEY = "__no_source__"

    @classmethod
    def _build_scope_filter_terms(
        cls,
        source_uuids: Optional[List[str]] = None,
        source_exclude_uuids: Optional[List[str]] = None,
        filename_conditions: Optional[List[Dict[str, Any]]] = None,
        filename_match_mode: str = "any",
    ) -> List[Any]:
        """Extra ImageSourceTerm/FilenameTerm terms for source and filename filtering,
        for callers (the training-config readiness/needs-review image lists) that
        enumerate a dataset union themselves rather than going through the main
        Review page's search. Mirrors buildImageSourceTerms/buildFilenameTerms in
        shared/api/review.js term for term, so the same curated filter state behaves
        identically whether it lands in a training-config modal or the Review tab.
        """
        terms: List[Any] = []

        inc = source_uuids or []
        inc_uuids = [u for u in inc if u and u != cls._NO_SOURCE_KEY]
        inc_unknown = cls._NO_SOURCE_KEY in inc
        if inc_uuids or inc_unknown:
            terms.append(ImageSourceTerm(source_uuids=inc_uuids, include_unknown=inc_unknown))
        for u in source_exclude_uuids or []:
            if u == cls._NO_SOURCE_KEY:
                terms.append(ImageSourceTerm(source_uuids=[], include_unknown=True, negate=True))
            elif u:
                terms.append(ImageSourceTerm(source_uuids=[u], include_unknown=False, negate=True))

        conditions = [c for c in (filename_conditions or []) if (c.get("pattern") or "").strip()]
        includes = [c["pattern"] for c in conditions if not c.get("negate")]
        if includes:
            term = FilenameTerm(patterns=includes)
            if len(includes) > 1:
                term.match_mode = filename_match_mode if filename_match_mode in ("any", "all") else "all"
            terms.append(term)
        for c in conditions:
            if c.get("negate"):
                terms.append(FilenameTerm(patterns=[c["pattern"]], negate=True))

        return terms

    async def get_training_config_needs_review_images(
        self,
        uuid: str,
        *,
        limit: int = 50,
        after_uuid: Optional[str] = None,
        after_sort_value: Optional[datetime] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        extra_label_uuids: Optional[List[str]] = None,
        source_uuids: Optional[List[str]] = None,
        source_exclude_uuids: Optional[List[str]] = None,
        filename_conditions: Optional[List[Dict[str, Any]]] = None,
        filename_match_mode: str = "any",
    ) -> tuple[list["ImageRecord"], bool]:
        """Return (images, has_more) for images that need verification in a training config.

        Images are those within the config's linked dataset union that have at least one
        target label (derived from the model config node) not yet verified. What "verified"
        means depends on the target label's scope, mirroring get_training_config_label_stats:
        - image-scope labels: a present/absent image_labels row.
        - object-scope labels: at least one bbox for that label carries the UNVERIFIED tag
          (an image can be fully triaged at the whole-image level — e.g. marked "absent" —
          while an individual bbox for that label still needs a human look).

        extra_label_uuids — when provided, additionally narrows the pool to images that
        also have bboxes with these labels (applied on top of each dataset's own filters).

        source_uuids/source_exclude_uuids/filename_conditions/filename_match_mode — see
        _build_scope_filter_terms; applied per-dataset in SQL via _narrow_pool below,
        same as extra_label_uuids.

        v1: in-memory merge across datasets. Push to SQL union if datasets grow large.
        """
        tc = await self.get_training_config(uuid)
        if tc is None:
            return [], False

        label_uuids, _ = await self._derive_labels_for_node(tc.model_config_node_uuid)
        if not label_uuids:
            return [], False

        async with self._async_session_maker() as session:
            labels_result = await session.exec(select(Label).where(Label.uuid.in_(label_uuids)))
            labels_by_uuid = {lbl.uuid: lbl for lbl in labels_result.all()}
        image_label_uuids = [
            u for u in label_uuids if labels_by_uuid.get(u) and labels_by_uuid[u].scope == LabelScope.IMAGE
        ]
        object_label_uuids = [
            u for u in label_uuids if labels_by_uuid.get(u) and labels_by_uuid[u].scope == LabelScope.OBJECT
        ]

        # Collect the candidate pool from each linked dataset (union, de-duped by UUID).
        # Verification itself is checked separately below (per-scope), not folded into the
        # dataset search params, since object-scope verification isn't expressible as a
        # LabelVerificationTerm (that term only reads image_labels rows).
        def _narrow_pool(params: ImageSearchParams) -> None:
            if extra_label_uuids:
                # Narrow pool: image must also have bboxes with the extra labels.
                # Merge into the dataset's own pool ObjectLabelTerm (built by
                # build_dataset_search_params) when there is one, so the union
                # stays a single "any of these labels" predicate rather than an
                # AND of two independent (and possibly disjoint) pool filters.
                # build_dataset_search_params appends that term (from
                # dataset.label_uuids) *after* the user-supplied f.terms, so it
                # is the LAST non-negated ObjectLabelTerm, not the first — a
                # user-supplied objectLabel term earlier in f.terms is an
                # unrelated filter and must not be widened here.
                pool_term = next(
                    (t for t in reversed(params.terms) if isinstance(t, ObjectLabelTerm) and not t.negate),
                    None,
                )
                if pool_term is not None:
                    pool_term.label_uuids = list(set(pool_term.label_uuids) | set(extra_label_uuids))
                else:
                    params.terms.append(ObjectLabelTerm(label_uuids=list(extra_label_uuids), roll_up=False))
            params.terms.extend(
                self._build_scope_filter_terms(
                    source_uuids, source_exclude_uuids, filename_conditions, filename_match_mode
                )
            )

        all_images = await self._enumerate_dataset_union_stubs(tc.dataset_uuids, mutate_params=_narrow_pool)

        if not all_images:
            return [], False

        needs_review_uuids = await self._images_needing_label_review(
            [img.uuid for img in all_images], image_label_uuids, object_label_uuids
        )
        all_images = [img for img in all_images if img.uuid in needs_review_uuids]

        if not all_images:
            return [], False

        # Sort the merged list by (sort_by, uuid) for deterministic ordering.
        def _sort_key(img: "ImageRecord"):
            return (getattr(img, sort_by), img.uuid)

        all_images.sort(key=_sort_key, reverse=(sort_order == "desc"))

        # Apply keyset cursor to resume from the previous page.
        if after_uuid is not None:
            start_idx = _keyset_start_index(
                all_images,
                after_uuid=after_uuid,
                after_sort_value=after_sort_value,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            if start_idx is None:
                return [], False
            all_images = all_images[start_idx:]

        has_more = len(all_images) > limit
        page = await self._get_images_with_relations([img.uuid for img in all_images[:limit]])
        return page, has_more

    async def _get_training_config_images_by_readiness_bucket(
        self,
        uuid: str,
        bucket: str,
        *,
        limit: int = 50,
        after_uuid: Optional[str] = None,
        after_sort_value: Optional[datetime] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        source_uuids: Optional[List[str]] = None,
        source_exclude_uuids: Optional[List[str]] = None,
        filename_conditions: Optional[List[Dict[str, Any]]] = None,
        filename_match_mode: str = "any",
    ) -> tuple[list["ImageRecord"], bool]:
        """Shared implementation backing the readiness-bucket image list methods.

        ``bucket`` selects which readiness bucket to return. The buckets themselves are
        ``_readiness_bucket``'s, the same predicate ``get_training_config_readiness``
        counts with, so a count and the list it is clickable through to cannot disagree
        about an image; "ready" is the union of ready_positive and ready_negative.

        source_uuids/source_exclude_uuids/filename_conditions/filename_match_mode — see
        _build_scope_filter_terms; applied per-dataset in SQL, same as the dataset's own
        filters, before the readiness bucket is computed over the (already narrowed) pool.

        v1: in-memory merge across datasets, same approach as
        ``get_training_config_needs_review_images``. Push to SQL union if datasets grow large.
        """
        tc = await self.get_training_config(uuid)
        if tc is None:
            return [], False

        label_uuids, class_uuids, effective_for_class, labels_map, _is_rollup = await self._readiness_class_shape(
            tc.model_config_node_uuid
        )
        if not label_uuids:
            return [], False

        if bucket not in ("ready", "ready_positive", "ready_negative", "needs_review", "not_started"):
            raise ValueError(f"Unknown readiness bucket: {bucket!r}")

        def _apply_scope_filters(params: ImageSearchParams) -> None:
            params.terms.extend(
                self._build_scope_filter_terms(
                    source_uuids, source_exclude_uuids, filename_conditions, filename_match_mode
                )
            )

        # Collect all images across all linked datasets (union, de-duped by uuid).
        all_images = await self._enumerate_dataset_union_stubs(tc.dataset_uuids, mutate_params=_apply_scope_filters)

        if not all_images:
            return [], False

        image_uuids = [img.uuid for img in all_images]
        covered, positives = await self._class_coverage(image_uuids, class_uuids, effective_for_class, labels_map)

        def _matches(image_uuid: str) -> bool:
            actual = self._readiness_bucket(covered.get(image_uuid, 0), len(class_uuids), image_uuid in positives)
            # "ready" is the union of its two halves; every other bucket is exact.
            return actual == bucket or (bucket == "ready" and actual.startswith("ready_"))

        bucket_images = [img for img in all_images if _matches(img.uuid)]

        if not bucket_images:
            return [], False

        # Sort the merged list by (sort_by, uuid) for deterministic ordering.
        def _sort_key(img: "ImageRecord"):
            return (getattr(img, sort_by), img.uuid)

        bucket_images.sort(key=_sort_key, reverse=(sort_order == "desc"))

        # Apply keyset cursor to resume from the previous page.
        if after_uuid is not None:
            start_idx = _keyset_start_index(
                bucket_images,
                after_uuid=after_uuid,
                after_sort_value=after_sort_value,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            if start_idx is None:
                return [], False
            bucket_images = bucket_images[start_idx:]

        has_more = len(bucket_images) > limit
        page = await self._get_images_with_relations([img.uuid for img in bucket_images[:limit]])
        return page, has_more

    async def get_training_config_ready_images(
        self,
        uuid: str,
        *,
        limit: int = 50,
        after_uuid: Optional[str] = None,
        after_sort_value: Optional[datetime] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        source_uuids: Optional[List[str]] = None,
        source_exclude_uuids: Optional[List[str]] = None,
        filename_conditions: Optional[List[Dict[str, Any]]] = None,
        filename_match_mode: str = "any",
    ) -> tuple[list["ImageRecord"], bool]:
        """Return (images, has_more) for images that are ready for training in a training config.

        Mirrors the eligibility gate used by ``get_training_config_readiness`` /
        ``get_training_config_detection_dataset``: an image is ready when every in-scope
        target label (derived from the model config node) has an explicit verified
        PRESENT/ABSENT image_label row. When rollup is active, an image is ready as soon
        as ANY ONE in-scope label is verified.
        """
        return await self._get_training_config_images_by_readiness_bucket(
            uuid,
            "ready",
            limit=limit,
            after_uuid=after_uuid,
            after_sort_value=after_sort_value,
            sort_by=sort_by,
            sort_order=sort_order,
            source_uuids=source_uuids,
            source_exclude_uuids=source_exclude_uuids,
            filename_conditions=filename_conditions,
            filename_match_mode=filename_match_mode,
        )

    async def get_training_config_ready_positive_images(
        self,
        uuid: str,
        *,
        limit: int = 50,
        after_uuid: Optional[str] = None,
        after_sort_value: Optional[datetime] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        source_uuids: Optional[List[str]] = None,
        source_exclude_uuids: Optional[List[str]] = None,
        filename_conditions: Optional[List[Dict[str, Any]]] = None,
        filename_match_mode: str = "any",
    ) -> tuple[list["ImageRecord"], bool]:
        """Return (images, has_more) for READY images that are POSITIVE samples — fully
        verified and carrying at least one in-scope label verified PRESENT.

        This is the subset of ``get_training_config_ready_images`` that actually contributes
        object annotations; the complement is ``get_training_config_ready_negative_images``.
        """
        return await self._get_training_config_images_by_readiness_bucket(
            uuid,
            "ready_positive",
            limit=limit,
            after_uuid=after_uuid,
            after_sort_value=after_sort_value,
            sort_by=sort_by,
            sort_order=sort_order,
            source_uuids=source_uuids,
            source_exclude_uuids=source_exclude_uuids,
            filename_conditions=filename_conditions,
            filename_match_mode=filename_match_mode,
        )

    async def get_training_config_ready_negative_images(
        self,
        uuid: str,
        *,
        limit: int = 50,
        after_uuid: Optional[str] = None,
        after_sort_value: Optional[datetime] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        source_uuids: Optional[List[str]] = None,
        source_exclude_uuids: Optional[List[str]] = None,
        filename_conditions: Optional[List[Dict[str, Any]]] = None,
        filename_match_mode: str = "any",
    ) -> tuple[list["ImageRecord"], bool]:
        """Return (images, has_more) for READY images that are NEGATIVE (background) samples —
        fully verified with every in-scope label verified ABSENT.
        """
        return await self._get_training_config_images_by_readiness_bucket(
            uuid,
            "ready_negative",
            limit=limit,
            after_uuid=after_uuid,
            after_sort_value=after_sort_value,
            sort_by=sort_by,
            sort_order=sort_order,
            source_uuids=source_uuids,
            source_exclude_uuids=source_exclude_uuids,
            filename_conditions=filename_conditions,
            filename_match_mode=filename_match_mode,
        )

    async def get_training_config_partial_review_images(
        self,
        uuid: str,
        *,
        limit: int = 50,
        after_uuid: Optional[str] = None,
        after_sort_value: Optional[datetime] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        source_uuids: Optional[List[str]] = None,
        source_exclude_uuids: Optional[List[str]] = None,
        filename_conditions: Optional[List[Dict[str, Any]]] = None,
        filename_match_mode: str = "any",
    ) -> tuple[list["ImageRecord"], bool]:
        """Return (images, has_more) for images with PARTIAL label coverage (the readiness
        panel's "needs review" bucket): 0 < verified_count < readiness_threshold.

        Distinct from ``get_training_config_needs_review_images``, which answers a different
        question (per-label bbox verification status, with pool-narrowing support) used by the
        label-stats panel's bbox-scoped browsing. Do not conflate the two.
        """
        return await self._get_training_config_images_by_readiness_bucket(
            uuid,
            "needs_review",
            limit=limit,
            after_uuid=after_uuid,
            after_sort_value=after_sort_value,
            sort_by=sort_by,
            sort_order=sort_order,
            source_uuids=source_uuids,
            source_exclude_uuids=source_exclude_uuids,
            filename_conditions=filename_conditions,
            filename_match_mode=filename_match_mode,
        )

    async def get_training_config_not_started_images(
        self,
        uuid: str,
        *,
        limit: int = 50,
        after_uuid: Optional[str] = None,
        after_sort_value: Optional[datetime] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        source_uuids: Optional[List[str]] = None,
        source_exclude_uuids: Optional[List[str]] = None,
        filename_conditions: Optional[List[Dict[str, Any]]] = None,
        filename_match_mode: str = "any",
    ) -> tuple[list["ImageRecord"], bool]:
        """Return (images, has_more) for images with ZERO label coverage (verified_count == 0) —
        i.e. images that have never been reviewed at all, as opposed to partially reviewed.
        """
        return await self._get_training_config_images_by_readiness_bucket(
            uuid,
            "not_started",
            limit=limit,
            after_uuid=after_uuid,
            after_sort_value=after_sort_value,
            sort_by=sort_by,
            sort_order=sort_order,
            source_uuids=source_uuids,
            source_exclude_uuids=source_exclude_uuids,
            filename_conditions=filename_conditions,
            filename_match_mode=filename_match_mode,
        )

    async def get_training_config_label_stats(
        self, uuid: str, restrict_to_image_uuids: Optional[list[str]] = None
    ) -> Optional[dict]:
        """Return per-label dataset stats for a training config.

        Relevant labels are derived from the model config node referenced by
        model_config_node_uuid. If no node is assigned, falls back to using
        all labels across all linked datasets.

        *restrict_to_image_uuids* narrows every count to that subset of the
        dataset union (None = the whole union, today's behaviour). The training
        page uses it to replay these stats over a previously-trained model's
        recorded image manifest, so "what went in last time" and "what is here
        now" are produced by the same code rather than two counting rules that
        can drift apart.

        Returns None if the training config doesn't exist.
        """
        tc = await self.get_training_config(uuid)
        if tc is None:
            return None

        dataset_uuids = tc.dataset_uuids

        # Fetch all datasets once and cache — both loops below need them.
        datasets_by_uuid: dict[str, Any] = {}
        for ds_uuid in dataset_uuids:
            dataset = await self.get_dataset(ds_uuid)
            if dataset is not None:
                datasets_by_uuid[ds_uuid] = dataset

        # Derive relevant label UUIDs from the node hierarchy (includes rollup children).
        node_label_uuids, rollup_map = await self._derive_labels_for_node(tc.model_config_node_uuid)

        if node_label_uuids:
            relevant_label_uuids = node_label_uuids
        else:
            # Fallback: union of all label UUIDs across selected datasets.
            dataset_label_uuids: set[str] = set()
            for dataset in datasets_by_uuid.values():
                dataset_label_uuids.update(dataset.label_uuids)
            relevant_label_uuids = list(dataset_label_uuids)

        if not relevant_label_uuids:
            return {"label_stats": [], "total_images": 0}

        labels_map = await self.get_labels_by_uuids(relevant_label_uuids)

        # For the fallback path (no model config node), sort in hierarchical order now
        # that we have the label metadata.  The node-derived path is already DFS-ordered.
        if not node_label_uuids:
            relevant_label_uuids = DbClient._sort_labels_hierarchically(relevant_label_uuids, labels_map)

        # Collect all image UUIDs from the datasets (union, de-duped).
        _union_stubs = await self._enumerate_dataset_union_stubs(list(datasets_by_uuid.keys()))
        all_image_uuids: list[str] = [s.uuid for s in _union_stubs]

        # Optional restriction to an explicit image set — used to re-run these exact
        # stats over a previous training run's recorded manifest, so "then" and "now"
        # are counted by identical rules. Intersected with the dataset union rather
        # than used raw: an image the run trained on but that has since left the
        # dataset can't be counted against labels this config no longer covers.
        # `total_images` then reports how much of the manifest survived, which the
        # caller compares against the manifest length it already holds.
        if restrict_to_image_uuids is not None:
            _restrict = set(restrict_to_image_uuids)
            all_image_uuids = [u for u in all_image_uuids if u in _restrict]

        total_images = len(all_image_uuids)

        image_label_uuids = [
            u for u in relevant_label_uuids if u in labels_map and labels_map[u].scope == LabelScope.IMAGE
        ]
        object_label_uuids = [
            u for u in relevant_label_uuids if u in labels_map and labels_map[u].scope == LabelScope.OBJECT
        ]

        # Seed result dict.
        label_stats: dict[str, dict] = {}
        for lbl_uuid in relevant_label_uuids:
            label = labels_map.get(lbl_uuid)
            if label is None:
                continue
            label_stats[lbl_uuid] = {
                "label_uuid": lbl_uuid,
                "label_scope": label.scope.value,
                "verified_image_count": 0,
                "unverified_image_count": total_images,
                "present_image_count": 0,
                "absent_image_count": 0,
                "verified_bbox_count": 0,
                "unverified_bbox_count": 0,
                "co_labels_covered_count": 0,
                "co_label_gaps_count": 0,
            }

        if not all_image_uuids:
            return {"label_stats": list(label_stats.values()), "total_images": 0}

        async with self._async_session_maker() as session:
            # NOTE: every query below filters on image_uuid IN (chunk) — the
            # image list is chunked (_UUID_IN_CHUNK) to stay under the SQLite
            # bind-variable cap. Chunks partition the images, so per-chunk
            # GROUP BY partial counts are summed into the zero-seeded stats.

            async def _rows_chunked(build_stmt) -> list:
                rows: list = []
                for chunk in _chunked(all_image_uuids):
                    rows.extend((await session.exec(build_stmt(chunk))).all())
                return rows

            # ── Image-scope labels ──────────────────────────────────────────
            if image_label_uuids:
                # Count verified (PRESENT/ABSENT) rows per label.
                for lbl_uuid, cnt in await _rows_chunked(
                    lambda chunk: select(ImageLabel.label_uuid, func.count().label("cnt"))
                    .where(
                        ImageLabel.image_uuid.in_(chunk),
                        ImageLabel.label_uuid.in_(image_label_uuids),
                        ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                    )
                    .group_by(ImageLabel.label_uuid)
                ):
                    if lbl_uuid in label_stats:
                        label_stats[lbl_uuid]["verified_image_count"] += cnt
                for lbl_uuid in image_label_uuids:
                    if lbl_uuid in label_stats:
                        label_stats[lbl_uuid]["unverified_image_count"] = (
                            total_images - label_stats[lbl_uuid]["verified_image_count"]
                        )

                # Count PRESENT and ABSENT rows per label in a single query.
                for lbl_uuid, status, cnt in await _rows_chunked(
                    lambda chunk: select(ImageLabel.label_uuid, ImageLabel.status, func.count().label("cnt"))
                    .where(
                        ImageLabel.image_uuid.in_(chunk),
                        ImageLabel.label_uuid.in_(image_label_uuids),
                        ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                    )
                    .group_by(ImageLabel.label_uuid, ImageLabel.status)
                ):
                    if lbl_uuid in label_stats:
                        key = "present_image_count" if status == ImageItemStatus.PRESENT else "absent_image_count"
                        label_stats[lbl_uuid][key] += cnt

            # ── Object-scope labels ─────────────────────────────────────────
            if object_label_uuids:

                def _bbox_count_stmt(chunk, tag_uuid):
                    inner = (
                        select(BBox.label_uuid, BBox.uuid.label("bbox_uuid"))
                        .join(BBoxTagLink, BBox.uuid == BBoxTagLink.bbox_uuid)
                        .where(
                            BBox.image_uuid.in_(chunk),
                            BBox.label_uuid.in_(object_label_uuids),
                            BBoxTagLink.tag_uuid == tag_uuid,
                        )
                        .subquery()
                    )
                    return select(inner.c.label_uuid, func.count().label("cnt")).group_by(inner.c.label_uuid)

                # Total verified bboxes per label.
                for lbl_uuid, cnt in await _rows_chunked(lambda chunk: _bbox_count_stmt(chunk, VERIFIED_TAG_UUID)):
                    if lbl_uuid in label_stats:
                        label_stats[lbl_uuid]["verified_bbox_count"] += cnt

                # Total unverified bboxes per label.
                for lbl_uuid, cnt in await _rows_chunked(lambda chunk: _bbox_count_stmt(chunk, UNVERIFIED_TAG_UUID)):
                    if lbl_uuid in label_stats:
                        label_stats[lbl_uuid]["unverified_bbox_count"] += cnt

        # ── Rollup aggregation ──────────────────────────────────────────────────
        # For each rollup parent, aggregate its children's stats into the parent row
        # and mark every entry with rollup metadata.
        if rollup_map:
            async with self._async_session_maker() as session:
                for rollup_parent_uuid, child_uuids in rollup_map.items():
                    if rollup_parent_uuid not in label_stats:
                        continue
                    parent_stat = label_stats[rollup_parent_uuid]
                    all_rollup_uuids = [rollup_parent_uuid] + [u for u in child_uuids if u in label_stats]
                    parent_scope = parent_stat["label_scope"]

                    if parent_scope == LabelScope.IMAGE.value:
                        # DISTINCT counts so a multi-labeled image isn't double-counted.
                        # Chunks partition the images, so per-chunk DISTINCT
                        # counts sum without double-counting.
                        for status_val, key in [
                            (ImageItemStatus.PRESENT, "present_image_count"),
                            (ImageItemStatus.ABSENT, "absent_image_count"),
                        ]:
                            total_cnt = 0
                            for chunk in _chunked(all_image_uuids):
                                distinct_sub = (
                                    select(ImageLabel.image_uuid)
                                    .where(
                                        ImageLabel.image_uuid.in_(chunk),
                                        ImageLabel.label_uuid.in_(all_rollup_uuids),
                                        ImageLabel.status == status_val,
                                    )
                                    .distinct()
                                    .subquery()
                                )
                                r = await session.exec(select(func.count()).select_from(distinct_sub))
                                total_cnt += r.one() or 0
                            parent_stat[key] = total_cnt

                        parent_stat["verified_image_count"] = (
                            parent_stat["present_image_count"] + parent_stat["absent_image_count"]
                        )
                        parent_stat["unverified_image_count"] = max(
                            total_images - parent_stat["verified_image_count"], 0
                        )

                    elif parent_scope == LabelScope.OBJECT.value:
                        # Bbox counts: each box has exactly one label → simple sum, no double-count.
                        parent_stat["verified_bbox_count"] = sum(
                            label_stats.get(u, {}).get("verified_bbox_count", 0) for u in all_rollup_uuids
                        )
                        parent_stat["unverified_bbox_count"] = sum(
                            label_stats.get(u, {}).get("unverified_bbox_count", 0) for u in all_rollup_uuids
                        )

                    parent_stat["has_rollup"] = True
                    for child_uuid in child_uuids:
                        if child_uuid in label_stats:
                            label_stats[child_uuid]["roll_up_parent_uuid"] = rollup_parent_uuid

        # ── Co-label coverage (object-scope labels) ─────────────────────────────
        # For each object label L, compute:
        #   base           = images with ≥1 verified bbox for L
        #   co_covered     = base ∩ ⋂{ accounted_for[M] : M is another top-level label }
        #   co_label_gaps  = base − co_covered
        #
        # "accounted for" a top-level label M:
        #   object-scope: has verified bbox OR explicit absent ImageLabel (for M or its rollup children)
        #   image-scope:  has any explicit PRESENT or ABSENT ImageLabel (for M or its rollup children)
        if object_label_uuids and all_image_uuids:
            # Determine rollup hierarchy so we can group children under their parent.
            child_to_parent: dict[str, str] = {}
            for _parent, _children in rollup_map.items():
                for _child in _children:
                    child_to_parent[_child] = _parent
            all_rollup_children_set = set(child_to_parent.keys())
            top_level_uuids = [u for u in relevant_label_uuids if u not in all_rollup_children_set and u in labels_map]

            # Effective UUIDs per top-level label: self + all rollup children (if any).
            effective_for_tl: dict[str, list[str]] = {tl: [tl] + rollup_map.get(tl, []) for tl in top_level_uuids}

            async with self._async_session_maker() as session:

                # Verified bbox image sets for every object label, batched into one
                # query per image-uuid chunk across *all* labels' raw UUIDs instead of
                # one query per label — with dozens of object labels this was the
                # dominant cost of this endpoint (one sequential round trip per label).
                # Rollup parents use their full effective set; children and standalone use just themselves.
                raw_needed = sorted(
                    {u for lbl_uuid in object_label_uuids for u in effective_for_tl.get(lbl_uuid, [lbl_uuid])}
                )
                raw_verified_by_label: dict[str, set[str]] = defaultdict(set)
                for chunk in _chunked(all_image_uuids):
                    rows = (
                        await session.exec(
                            select(BBox.label_uuid, BBox.image_uuid)
                            .distinct()
                            .where(
                                BBox.image_uuid.in_(chunk),
                                BBox.label_uuid.in_(raw_needed),
                                BBox.uuid.in_(
                                    select(BBoxTagLink.bbox_uuid).where(BBoxTagLink.tag_uuid == VERIFIED_TAG_UUID)
                                ),
                            )
                        )
                    ).all()
                    for raw_lbl_uuid, img_uuid in rows:
                        raw_verified_by_label[raw_lbl_uuid].add(img_uuid)

                obj_verified_imgs: dict[str, set[str]] = {
                    lbl_uuid: set().union(
                        *(raw_verified_by_label.get(u, set()) for u in effective_for_tl.get(lbl_uuid, [lbl_uuid]))
                    )
                    for lbl_uuid in object_label_uuids
                }

                # "Accounted for" image sets per top-level label.
                accounted_for_tl = await self._co_label_accounted_for(
                    session,
                    all_image_uuids,
                    top_level_uuids,
                    effective_for_tl,
                    labels_map,
                    verified_bbox_sets=obj_verified_imgs,
                )

                # Compute co-label stats for each object label.
                for lbl_uuid in object_label_uuids:
                    if lbl_uuid not in label_stats:
                        continue
                    base = obj_verified_imgs.get(lbl_uuid, set())
                    # Exclude this label's top-level ancestor from the co-label check.
                    tl_ancestor = child_to_parent.get(lbl_uuid, lbl_uuid)
                    other_tls = [tl for tl in top_level_uuids if tl != tl_ancestor]

                    co_covered = set(base)
                    for tl in other_tls:
                        co_covered &= accounted_for_tl.get(tl, set())

                    gap = base - co_covered
                    breakdown = []
                    for tl in other_tls:
                        # Equivalent to `gap - accounted_for_tl[tl]`: co_covered (which gap
                        # already excludes) is always a subset of accounted_for_tl[tl].
                        missing = base - accounted_for_tl.get(tl, set())
                        if not missing:
                            continue
                        tl_label = labels_map.get(tl)
                        breakdown.append(
                            {
                                "label_uuid": tl,
                                "label_name": tl_label.name if tl_label is not None else tl,
                                "count": len(missing),
                            }
                        )
                    breakdown.sort(key=lambda b: b["count"], reverse=True)

                    label_stats[lbl_uuid]["co_labels_covered_count"] = len(co_covered)
                    label_stats[lbl_uuid]["co_label_gaps_count"] = len(gap)
                    label_stats[lbl_uuid]["co_label_gap_breakdown"] = breakdown

        return {"label_stats": list(label_stats.values()), "total_images": total_images}

    async def preview_dataset_images(
        self,
        filter_json: Dict[str, Any],
        label_uuids: List[str],
        *,
        limit: int = 500,
    ) -> tuple[list[ImageRecord], bool]:
        """Execute filter criteria without a stored dataset (unsaved preview)."""
        fake_dataset = DatasetRead(
            uuid="__preview__",
            name="__preview__",
            filter_json=filter_json,
            label_uuids=label_uuids,
        )
        return await self._list_images_for_dataset(fake_dataset, limit=limit)

    # ─── Model Configurations ─────────────────────────────────────────────────

    @staticmethod
    def _mc_to_read(mc: ModelConfig) -> ModelConfigRead:
        return ModelConfigRead.model_validate(mc)

    async def _sync_model_configs_to_json(self) -> None:
        """Dump model_configs (+ their nodes) to model_configs.json.

        Each node stores label_name beside label_uuid, matching the sidecar
        convention: on a rebuild that mints new label UUIDs the name is the
        only key that can still resolve the node to a label.
        """
        async with self._async_session_maker() as session:
            stmt = select(ModelConfig).options(selectinload(ModelConfig.nodes))
            configs = (await session.exec(stmt)).all()
            label_names: Dict[str, str] = {
                uuid: name for uuid, name in (await session.exec(select(Label.uuid, Label.name))).all()
            }
            rows = [
                {
                    "uuid": mc.uuid,
                    "name": mc.name,
                    "description": mc.description,
                    "nodes": [
                        {
                            # The node UUID is a real reference target, not an
                            # internal detail: training_configs.model_config_node_uuid
                            # is an FK to it and every trained model records it in
                            # its metadata. Minting a fresh one on restore would
                            # leave all of those dangling.
                            "uuid": n.uuid,
                            "label_uuid": n.label_uuid,
                            "label_name": label_names.get(n.label_uuid) if n.label_uuid else None,
                            "model_type": n.model_type,
                            "architecture": n.architecture,
                            "roll_up_unchecked": n.roll_up_unchecked,
                        }
                        for n in (mc.nodes or [])
                    ],
                }
                for mc in configs
            ]
        self._atomic_write_text(self._model_configs_json_path, json.dumps(rows, indent=2))

    async def sync_model_configs_from_json(self) -> None:
        """Read model_configs.json into the DB. Does not remove existing configs.

        A config is matched by UUID, then by name (names are unique, so a name
        collision must reuse the DB row or the insert violates the constraint).
        Nodes are replaced wholesale for any config it touches.

        A node whose label can be resolved by neither UUID nor name is dropped
        with a warning rather than failing the whole config — the FK would
        reject it anyway, and losing one node beats losing the config.
        """
        if not self._model_configs_json_path.exists():
            return

        try:
            json_data = json.loads(self._model_configs_json_path.read_text())
        except Exception as e:
            logger.error(f"Failed to parse model_configs.json: {e}")
            return

        async with self._async_session_maker() as session:
            existing = (await session.exec(select(ModelConfig).options(selectinload(ModelConfig.nodes)))).all()
            by_uuid: Dict[str, ModelConfig] = {mc.uuid: mc for mc in existing}
            by_name: Dict[str, ModelConfig] = {mc.name: mc for mc in existing}

            labels = (await session.exec(select(Label.uuid, Label.name))).all()
            valid_label_uuids = {uuid for uuid, _ in labels}
            label_uuid_by_name = {name: uuid for uuid, name in labels}

            count = 0
            for item in json_data:
                json_uuid = item.get("uuid")
                name = item.get("name")
                if not name:
                    continue

                mc = by_uuid.get(json_uuid) if json_uuid else None
                if mc is None and name in by_name:
                    mc = by_name[name]
                    logger.warning(
                        "Model config '%s' from JSON (UUID %s) matched by name to existing DB row (UUID %s).",
                        name,
                        json_uuid,
                        mc.uuid,
                    )
                if mc is None:
                    mc = ModelConfig(uuid=json_uuid or str(uuid4()), name=name)
                    session.add(mc)
                    await session.flush()

                mc.name = name
                mc.description = item.get("description")
                session.add(mc)

                await session.exec(delete(ModelConfigNode).where(ModelConfigNode.model_config_uuid == mc.uuid))
                seen_labels: Set[Optional[str]] = set()
                for node in item.get("nodes", []):
                    label_uuid = node.get("label_uuid")
                    if label_uuid and label_uuid not in valid_label_uuids:
                        fallback = label_uuid_by_name.get(node.get("label_name") or "")
                        if fallback is None:
                            logger.warning(
                                "Model config '%s': node label (uuid=%s, name=%s) unresolvable; dropping node.",
                                name,
                                label_uuid,
                                node.get("label_name"),
                            )
                            continue
                        label_uuid = fallback
                    # A duplicate label within one config violates the same rule
                    # create_model_config enforces; keep the first and warn.
                    if label_uuid in seen_labels:
                        logger.warning("Model config '%s': duplicate node label %s; dropping.", name, label_uuid)
                        continue
                    # model_configs.json sits beside labels.json, which is meant to
                    # be hand-edited, so a node missing a required key is a normal
                    # accident. Drop it like an unresolvable label rather than
                    # raising: this runs inside init_db(), and a KeyError here takes
                    # db_server down at boot with nothing pointing at the cause.
                    model_type = node.get("model_type")
                    architecture = node.get("architecture")
                    if not model_type or not architecture:
                        logger.warning(
                            "Model config '%s': node (label=%s) missing model_type/architecture; dropping node.",
                            name,
                            label_uuid,
                        )
                        continue
                    seen_labels.add(label_uuid)
                    session.add(
                        ModelConfigNode(
                            # Preserve the dumped UUID so training configs and
                            # trained models keep resolving to this node.
                            uuid=node.get("uuid") or str(uuid4()),
                            model_config_uuid=mc.uuid,
                            label_uuid=label_uuid,
                            model_type=model_type,
                            architecture=architecture,
                            roll_up_unchecked=node.get("roll_up_unchecked", False),
                            is_boundary=node.get("is_boundary", False),
                        )
                    )
                count += 1

            await session.commit()
            logger.info("Synced %d model configs from %s", count, self._model_configs_json_path)

    async def list_model_configs(self) -> List[ModelConfigRead]:
        async with self._async_session_maker() as session:
            stmt = select(ModelConfig).options(selectinload(ModelConfig.nodes))
            result = await session.exec(stmt)
            return [self._mc_to_read(mc) for mc in result.all()]

    async def get_model_config(self, uuid: str) -> Optional[ModelConfigRead]:
        async with self._async_session_maker() as session:
            stmt = select(ModelConfig).where(ModelConfig.uuid == uuid).options(selectinload(ModelConfig.nodes))
            result = await session.exec(stmt)
            mc = result.one_or_none()
            if mc is None:
                return None
            return self._mc_to_read(mc)

    async def get_model_config_node(self, node_uuid: str) -> Optional[ModelConfigNodeRead]:
        """Look up a ModelConfigNode by its UUID and return a read model."""
        async with self._async_session_maker() as session:
            node = await session.get(ModelConfigNode, node_uuid)
            if node is None:
                return None
            return ModelConfigNodeRead.model_validate(node)

    @staticmethod
    def _resolve_node_boundaries(nodes, parent_of: dict) -> dict:
        """Decide `is_boundary` for each written node: stated wins, unstated is inferred.

        *nodes* is an iterable of ``(label_uuid, model_type, architecture, is_boundary)``
        — a tuple rather than the write model because the create path holds Pydantic
        objects and the update path holds dumped dicts. Returns {label_uuid: bool} for
        the non-root nodes of one written config.

        Unstated (None) nodes fall back to the rule this column replaced — spec differs
        from the nearest boundary ancestor's spec, else the root's. That keeps a client
        written before the field, or a config dump restored from before it, producing
        the tree it always produced; defaulting them to False instead would silently
        collapse every sub-model into its top model, which is a data-shaped bug that
        only shows up much later as detections stopping at a parent label.

        Nodes are walked shallowest-first because a node's inherited spec depends on
        which of its ancestors are boundaries.
        """
        nodes = list(nodes)
        non_root = [n for n in nodes if n[0] is not None]
        if not non_root:
            return {}
        root = next((n for n in nodes if n[0] is None), None)

        def depth(label_uuid: str) -> int:
            d, p = 0, parent_of.get(label_uuid)
            while p is not None:
                d += 1
                p = parent_of.get(p)
            return d

        root_spec = (root[1], root[2]) if root is not None else None
        boundary_spec: dict = {}
        resolved: dict = {}
        for label_uuid, model_type, architecture, stated in sorted(non_root, key=lambda n: depth(n[0])):
            if stated is not None:
                is_boundary = bool(stated)
            elif root_spec is None:
                # No root in the payload, so there is no inherited spec to differ from.
                is_boundary = False
            else:
                ancestor, p = None, parent_of.get(label_uuid)
                while p is not None:
                    if p in boundary_spec:
                        ancestor = p
                        break
                    p = parent_of.get(p)
                inherited = boundary_spec[ancestor] if ancestor is not None else root_spec
                is_boundary = (model_type, architecture) != inherited
            if is_boundary:
                boundary_spec[label_uuid] = (model_type, architecture)
            resolved[label_uuid] = is_boundary
        return resolved

    async def create_model_config(self, data: ModelConfigCreate) -> ModelConfigRead:
        # Validate uniqueness of label_uuid within the node list
        seen: set = set()
        for node in data.nodes:
            key = node.label_uuid  # None is valid once (root)
            if key in seen:
                raise ValueError(f"Duplicate label_uuid in nodes: {key!r}")
            seen.add(key)

        async with self._async_session_maker() as session:
            labels = (await session.exec(select(Label))).all()
        boundaries = self._resolve_node_boundaries(
            [(n.label_uuid, n.model_type, n.architecture, n.is_boundary) for n in data.nodes],
            {lbl.uuid: lbl.parent_uuid for lbl in labels},
        )

        async with self._async_session_maker() as session:
            mc = ModelConfig(name=data.name, description=data.description)
            session.add(mc)
            await session.flush()
            for node in data.nodes:
                session.add(
                    ModelConfigNode(
                        model_config_uuid=mc.uuid,
                        label_uuid=node.label_uuid,
                        model_type=node.model_type,
                        architecture=node.architecture,
                        roll_up_unchecked=node.roll_up_unchecked,
                        # Root is the top model by definition, never a sub-model boundary.
                        is_boundary=boundaries.get(node.label_uuid, False),
                    )
                )
            await session.commit()
            stmt = select(ModelConfig).where(ModelConfig.uuid == mc.uuid).options(selectinload(ModelConfig.nodes))
            result = await session.exec(stmt)
            out = self._mc_to_read(result.one())
        await self._sync_model_configs_to_json()
        return out

    async def update_model_config(self, uuid: str, data: ModelConfigUpdate) -> Optional[ModelConfigRead]:
        async with self._async_session_maker() as session:
            result = await session.exec(select(ModelConfig).where(ModelConfig.uuid == uuid))
            mc = result.one_or_none()
            if mc is None:
                return None

            changes = data.model_dump(exclude_unset=True, mode="json")
            nodes_data = changes.pop("nodes", None)

            for key, value in changes.items():
                if hasattr(mc, key):
                    setattr(mc, key, value)

            if isinstance(nodes_data, list):
                # Reconcile by label_uuid rather than replacing the node rows
                # wholesale. A node UUID is a reference target, not an internal
                # detail — training_configs.model_config_node_uuid is an FK to it
                # (ON DELETE SET NULL) and every trained model stamps it into its
                # metadata. Deleting and re-inserting would mint fresh UUIDs and
                # silently unassign every training config using this shape, even
                # when the edit changed nothing about their node.
                #
                # label_uuid is the right identity: it is unique within a config
                # (enforced just below, as create_model_config does) and it is what
                # the editor UI keys its rows on.
                seen: set = set()
                for node in nodes_data:
                    key_lbl = node.get("label_uuid")
                    if key_lbl in seen:
                        raise ValueError(f"Duplicate label_uuid in nodes: {key_lbl!r}")
                    seen.add(key_lbl)

                existing_nodes = (
                    await session.exec(select(ModelConfigNode).where(ModelConfigNode.model_config_uuid == uuid))
                ).all()
                by_label: Dict[Optional[str], ModelConfigNode] = {n.label_uuid: n for n in existing_nodes}

                labels = (await session.exec(select(Label))).all()
                boundaries = self._resolve_node_boundaries(
                    [
                        (n.get("label_uuid"), n["model_type"], n["architecture"], n.get("is_boundary"))
                        for n in nodes_data
                    ],
                    {lbl.uuid: lbl.parent_uuid for lbl in labels},
                )

                for node in nodes_data:
                    label_uuid = node.get("label_uuid")
                    row = by_label.pop(label_uuid, None)
                    if row is None:
                        row = ModelConfigNode(model_config_uuid=uuid, label_uuid=label_uuid)
                    row.model_type = node["model_type"]
                    row.architecture = node["architecture"]
                    row.roll_up_unchecked = node.get("roll_up_unchecked", False)
                    # Root is the top model by definition, never a sub-model boundary.
                    row.is_boundary = boundaries.get(label_uuid, False)
                    session.add(row)

                # Whatever is left was dropped from the shape. These genuinely go
                # away, and the FK's SET NULL correctly unassigns anything pointing
                # at them — that is a real change, not incidental churn.
                for stale in by_label.values():
                    await session.delete(stale)

            mc.updated_at = datetime.now(timezone.utc)
            session.add(mc)
            await session.commit()

            result2 = await session.exec(
                select(ModelConfig).where(ModelConfig.uuid == uuid).options(selectinload(ModelConfig.nodes))
            )
            out = self._mc_to_read(result2.one())
        await self._sync_model_configs_to_json()
        return out

    async def delete_model_config(self, uuid: str) -> bool:
        async with self._async_session_maker() as session:
            mc = await session.get(ModelConfig, uuid)
            if mc is None:
                return False
            await session.delete(mc)
            await session.commit()
        await self._sync_model_configs_to_json()
        return True

    # ─── Filter Chains ────────────────────────────────────────────────────────

    @staticmethod
    def _fc_to_read(fc: FilterChain) -> FilterChainRead:
        return FilterChainRead.model_validate(fc)

    @staticmethod
    def _validate_filter_chain_step_orders(orders) -> None:
        seen: set = set()
        for order in orders:
            if order in seen:
                raise ValueError(f"Duplicate step order in filter chain: {order!r}")
            seen.add(order)

    async def list_filter_chains(self) -> List[FilterChainRead]:
        async with self._async_session_maker() as session:
            stmt = select(FilterChain).options(selectinload(FilterChain.steps))
            result = await session.exec(stmt)
            return [self._fc_to_read(fc) for fc in result.all()]

    async def get_filter_chain(self, uuid: str) -> Optional[FilterChainRead]:
        async with self._async_session_maker() as session:
            stmt = select(FilterChain).where(FilterChain.uuid == uuid).options(selectinload(FilterChain.steps))
            result = await session.exec(stmt)
            fc = result.one_or_none()
            if fc is None:
                return None
            return self._fc_to_read(fc)

    async def create_filter_chain(self, data: FilterChainCreate) -> FilterChainRead:
        self._validate_filter_chain_step_orders(step.order for step in data.steps)

        async with self._async_session_maker() as session:
            fc = FilterChain(name=data.name, description=data.description)
            session.add(fc)
            await session.flush()
            for step in data.steps:
                session.add(
                    FilterChainStep(
                        filter_chain_uuid=fc.uuid,
                        order=step.order,
                        model_name=step.model_name,
                        confidence=step.confidence,
                        iou_threshold=step.iou_threshold,
                        action=step.action.value,
                        action_params=step.action_params,
                        enabled=step.enabled,
                    )
                )
            await session.commit()
            stmt = select(FilterChain).where(FilterChain.uuid == fc.uuid).options(selectinload(FilterChain.steps))
            result = await session.exec(stmt)
            return self._fc_to_read(result.one())

    async def update_filter_chain(self, uuid: str, data: FilterChainUpdate) -> Optional[FilterChainRead]:
        if data.steps is not None:
            self._validate_filter_chain_step_orders(step.order for step in data.steps)

        async with self._async_session_maker() as session:
            result = await session.exec(select(FilterChain).where(FilterChain.uuid == uuid))
            fc = result.one_or_none()
            if fc is None:
                return None

            changes = data.model_dump(exclude_unset=True, mode="json")
            steps_data = changes.pop("steps", None)

            for key, value in changes.items():
                if hasattr(fc, key):
                    setattr(fc, key, value)

            if isinstance(steps_data, list):
                await session.exec(delete(FilterChainStep).where(FilterChainStep.filter_chain_uuid == uuid))
                for step in steps_data:
                    session.add(
                        FilterChainStep(
                            filter_chain_uuid=uuid,
                            order=step["order"],
                            model_name=step["model_name"],
                            confidence=step.get("confidence", 0.5),
                            iou_threshold=step.get("iou_threshold", 0.5),
                            action=step["action"],
                            action_params=step.get("action_params", {}),
                            enabled=step.get("enabled", True),
                        )
                    )

            fc.updated_at = datetime.now(timezone.utc)
            session.add(fc)
            await session.commit()

            result2 = await session.exec(
                select(FilterChain).where(FilterChain.uuid == uuid).options(selectinload(FilterChain.steps))
            )
            return self._fc_to_read(result2.one())

    async def delete_filter_chain(self, uuid: str) -> bool:
        async with self._async_session_maker() as session:
            fc = await session.get(FilterChain, uuid)
            if fc is None:
                return False
            await session.delete(fc)
            await session.commit()
        return True

    # ─── Training Collectors ──────────────────────────────────────────────────

    async def list_training_collectors(self) -> List[TrainingCollectorRead]:
        async with self._async_session_maker() as session:
            result = await session.exec(select(TrainingCollectorRow).order_by(TrainingCollectorRow.name))
            return [TrainingCollectorRead.model_validate(row) for row in result.all()]

    async def get_training_collector(self, uuid: str) -> Optional[TrainingCollectorRead]:
        async with self._async_session_maker() as session:
            row = await session.get(TrainingCollectorRow, uuid)
            return None if row is None else TrainingCollectorRead.model_validate(row)

    async def create_training_collector(self, data: TrainingCollectorCreate) -> TrainingCollectorRead:
        async with self._async_session_maker() as session:
            row = TrainingCollectorRow(
                name=data.name,
                description=data.description,
                enabled=data.enabled,
                params_json=data.params,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return TrainingCollectorRead.model_validate(row)

    async def update_training_collector(
        self, uuid: str, data: TrainingCollectorUpdate
    ) -> Optional[TrainingCollectorRead]:
        async with self._async_session_maker() as session:
            row = await session.get(TrainingCollectorRow, uuid)
            if row is None:
                return None
            changes = data.model_dump(exclude_unset=True, mode="json")
            # params: None means "no change" (see TrainingCollectorUpdate) — an explicit
            # null must not clear a non-nullable column, so only apply it when non-null.
            if changes.pop("params", None) is not None:
                row.params_json = data.params
            for key, value in changes.items():
                setattr(row, key, value)
            row.updated_at = datetime.now(timezone.utc)
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return TrainingCollectorRead.model_validate(row)

    async def delete_training_collector(self, uuid: str) -> bool:
        async with self._async_session_maker() as session:
            row = await session.get(TrainingCollectorRow, uuid)
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
        return True

    # ─── Model Groups ───────────────────────────────────────────────────────

    @staticmethod
    def _mg_to_read(mg: ModelGroup) -> ModelGroupRead:
        return ModelGroupRead.model_validate(mg)

    @staticmethod
    def _validate_model_group_member_orders(orders) -> None:
        seen: set = set()
        for order in orders:
            if order in seen:
                raise ValueError(f"Duplicate member order in model group: {order!r}")
            seen.add(order)

    async def _collect_reachable_group_uuids(self, session, start_uuids: List[str]) -> set:
        """BFS over child_group_uuid references. Returns every group uuid reachable
        (directly or transitively) from `start_uuids`, including `start_uuids` themselves."""
        seen: set = set()
        frontier = list(start_uuids)
        while frontier:
            group_uuid = frontier.pop()
            if group_uuid in seen:
                continue
            seen.add(group_uuid)
            result = await session.exec(
                select(ModelGroupMember.child_group_uuid).where(
                    ModelGroupMember.model_group_uuid == group_uuid,
                    ModelGroupMember.child_group_uuid.is_not(None),
                )
            )
            for child_uuid in result.all():
                if child_uuid not in seen:
                    frontier.append(child_uuid)
        return seen

    async def _validate_no_model_group_cycle(
        self, session, group_uuid: Optional[str], member_child_uuids: List[str]
    ) -> None:
        """Raise if saving `member_child_uuids` as members of `group_uuid` would create a
        cycle — i.e. one of those child groups (directly or transitively) already contains
        `group_uuid`. A brand-new group (`group_uuid` is None) can't yet be referenced by
        anything, so it's trivially safe."""
        if group_uuid is None or not member_child_uuids:
            return
        reachable = await self._collect_reachable_group_uuids(session, member_child_uuids)
        if group_uuid in reachable:
            raise ValueError(
                "Model group members would create a cycle " "(a group cannot contain itself, directly or transitively)"
            )

    async def list_model_groups(self) -> List[ModelGroupRead]:
        async with self._async_session_maker() as session:
            stmt = select(ModelGroup).options(selectinload(ModelGroup.members))
            result = await session.exec(stmt)
            return [self._mg_to_read(mg) for mg in result.all()]

    async def get_model_group(self, uuid: str) -> Optional[ModelGroupRead]:
        async with self._async_session_maker() as session:
            stmt = select(ModelGroup).where(ModelGroup.uuid == uuid).options(selectinload(ModelGroup.members))
            result = await session.exec(stmt)
            mg = result.one_or_none()
            if mg is None:
                return None
            return self._mg_to_read(mg)

    async def create_model_group(self, data: ModelGroupCreate) -> ModelGroupRead:
        self._validate_model_group_member_orders(m.order for m in data.members)

        async with self._async_session_maker() as session:
            mg = ModelGroup(name=data.name, description=data.description, merge_policy=data.merge_policy)
            session.add(mg)
            await session.flush()
            for member in data.members:
                session.add(
                    ModelGroupMember(
                        model_group_uuid=mg.uuid,
                        order=member.order,
                        model_name=member.model_name,
                        child_group_uuid=member.child_group_uuid,
                    )
                )
            await session.commit()
            stmt = select(ModelGroup).where(ModelGroup.uuid == mg.uuid).options(selectinload(ModelGroup.members))
            result = await session.exec(stmt)
            return self._mg_to_read(result.one())

    async def update_model_group(self, uuid: str, data: ModelGroupUpdate) -> Optional[ModelGroupRead]:
        if data.members is not None:
            self._validate_model_group_member_orders(m.order for m in data.members)

        async with self._async_session_maker() as session:
            result = await session.exec(select(ModelGroup).where(ModelGroup.uuid == uuid))
            mg = result.one_or_none()
            if mg is None:
                return None

            changes = data.model_dump(exclude_unset=True, mode="json")
            members_data = changes.pop("members", None)

            if isinstance(members_data, list):
                child_uuids = [m["child_group_uuid"] for m in members_data if m.get("child_group_uuid")]
                await self._validate_no_model_group_cycle(session, uuid, child_uuids)

            for key, value in changes.items():
                if hasattr(mg, key):
                    setattr(mg, key, value)

            if isinstance(members_data, list):
                await session.exec(delete(ModelGroupMember).where(ModelGroupMember.model_group_uuid == uuid))
                for member in members_data:
                    session.add(
                        ModelGroupMember(
                            model_group_uuid=uuid,
                            order=member["order"],
                            model_name=member.get("model_name"),
                            child_group_uuid=member.get("child_group_uuid"),
                        )
                    )

            mg.updated_at = datetime.now(timezone.utc)
            session.add(mg)
            await session.commit()

            result2 = await session.exec(
                select(ModelGroup).where(ModelGroup.uuid == uuid).options(selectinload(ModelGroup.members))
            )
            return self._mg_to_read(result2.one())

    async def delete_model_group(self, uuid: str) -> bool:
        async with self._async_session_maker() as session:
            mg = await session.get(ModelGroup, uuid)
            if mg is None:
                return False
            await session.delete(mg)
            await session.commit()
        return True

    # ─── Watch Profiles (StreamWatcher / NVR) ─────────────────────────────────
    #
    # Nested writes with whole-list replace, following the Filter Chains precedent —
    # with one deliberate difference: rules and zones may carry a `uuid` on write and
    # it is preserved. A rule refers to a zone by uuid, so regenerating uuids on every
    # save would silently break every zone-based rule in the profile.

    @staticmethod
    def _wp_to_read(profile: WatchProfile) -> WatchProfileRead:
        return WatchProfileRead.model_validate(profile)

    @staticmethod
    def _watch_profile_loaders():
        """Eager-load options for a fully-populated profile.

        Note `selectinload` + `limit` interact badly in general (LIMIT applies only to
        the main query), but there is no pagination here — profiles are few and always
        fetched whole.
        """
        return (
            selectinload(WatchProfile.sources),
            selectinload(WatchProfile.rules),
            selectinload(WatchProfile.actions),
            selectinload(WatchProfile.label_lifetimes),
        )

    async def _validate_watch_profile(self, data) -> None:
        """Cross-entity checks the per-model validators cannot see.

        Zone references are the important one: a rule naming a zone that isn't drawn on
        any of the profile's sources would simply never fire, which is invisible at
        runtime and maddening to debug.

        This became a DB query when regions moved to `source_regions`: the zone uuids a
        rule may legally name are no longer anywhere in the payload, they belong to the
        sources the payload merely *references*. Validating against the payload alone
        would have meant accepting every zone uuid, silently.
        """
        sources = data.sources if data.sources is not None else []
        rules = data.rules if data.rules is not None else []
        actions = data.actions if data.actions is not None else []
        lifetimes = data.label_lifetimes if data.label_lifetimes is not None else []

        # Label lifetimes used to hold a single `label_uuid` with a real foreign key, and
        # the FK is what rejected a made-up uuid. The column is a JSON list now — a list
        # cannot carry an FK — so the same check happens here, in one query rather than
        # one per row.
        lifetime_refs = {str(u) for row in lifetimes for u in (row.label_uuids_json or [])}
        if lifetime_refs:
            async with self._async_session_maker() as session:
                result = await session.exec(select(Label.uuid).where(Label.uuid.in_(lifetime_refs)))
                known_labels = set(result.all())
            missing = sorted(lifetime_refs - known_labels)
            if missing:
                raise ValueError(f"Label lifetime references label {missing[0]!r}, which does not exist")

        seen_sources: set = set()
        for src in sources:
            if src.source_uuid in seen_sources:
                raise ValueError(f"Source {src.source_uuid!r} is listed twice in this watch profile")
            seen_sources.add(src.source_uuid)

        # Only the zone-kind refs need resolving, so skip the query when no rule uses one.
        needs_zones = any(
            getattr(kind, "value", kind) == "zone"
            for rule in rules
            for kind in (rule.subject_a_kind, rule.subject_b_kind)
        )
        zone_uuids: set = set()
        if needs_zones and seen_sources:
            async with self._async_session_maker() as session:
                result = await session.exec(
                    select(SourceRegion.uuid).where(
                        SourceRegion.source_uuid.in_(seen_sources),
                        SourceRegion.role == RegionRole.ZONE.value,
                    )
                )
                zone_uuids = set(result.all())

        rule_uuids = {rule.uuid for rule in rules if rule.uuid}

        for rule in rules:
            for kind, refs, side in (
                (rule.subject_a_kind, rule.subject_a_refs_json, "a"),
                (rule.subject_b_kind, rule.subject_b_refs_json, "b"),
            ):
                if getattr(kind, "value", kind) != "zone":
                    continue
                for ref in refs or []:
                    if ref not in zone_uuids:
                        raise ValueError(
                            f"Rule {rule.name!r} references zone {ref!r} (subject {side}), "
                            "which is not drawn on any source in this profile"
                        )

        # A mask this watcher disables must actually exist on the source it is listed
        # under, or the opt-out is a no-op that reads as though it were doing something.
        #
        # The *check* is per source, but the fetch is not: one query over every source
        # with an opt-out, keyed by source on the way out, exactly as the zone lookup
        # above does. Asking per source opened a session per iteration on the save path.
        opted_out = {src.source_uuid for src in sources if src.disabled_mask_uuids_json}
        masks_by_source: Dict[str, set] = {}
        if opted_out:
            async with self._async_session_maker() as session:
                result = await session.exec(
                    select(SourceRegion.source_uuid, SourceRegion.uuid).where(
                        SourceRegion.source_uuid.in_(opted_out),
                        SourceRegion.role == RegionRole.MASK.value,
                    )
                )
                for source_uuid, mask_uuid in result.all():
                    masks_by_source.setdefault(source_uuid, set()).add(mask_uuid)

        for src in sources:
            mask_uuids = masks_by_source.get(src.source_uuid, set())
            for ref in (str(u) for u in (src.disabled_mask_uuids_json or [])):
                if ref not in mask_uuids:
                    raise ValueError(
                        f"Source {src.source_uuid!r} disables mask {ref!r}, which is not drawn on that source"
                    )

        # A rule's source scope is validated the same way, and for the same reason: a
        # rule naming a source this watcher does not have would simply never be
        # evaluated, which looks identical to a rule that is broken.
        for rule in rules:
            for ref in rule.source_uuids_json or []:
                if str(ref) not in seen_sources:
                    raise ValueError(f"Rule {rule.name!r} targets source {ref!r}, which is not in this profile")

        # Both scope lists are validated here rather than by a foreign key: the nested
        # write replaces every child row wholesale, so an FK would fire against rules
        # that have not been inserted yet. Saying which action and which uuid is also
        # far more useful than an integrity error.
        for action in actions:
            for ref in action.rule_uuids_json or []:
                if str(ref) not in rule_uuids:
                    raise ValueError(f"Action {action.kind!r} targets rule {ref!r}, which is not in this profile")
            for ref in action.source_uuids_json or []:
                if str(ref) not in seen_sources:
                    raise ValueError(f"Action {action.kind!r} targets source {ref!r}, which is not in this profile")

    @staticmethod
    def _add_watch_children(
        session,
        profile_uuid: str,
        sources: Optional[List[WatchProfileSourceWrite]],
        rules: Optional[List[InteractionRuleWrite]],
        actions: Optional[List[WatchActionWrite]],
        lifetimes: Optional[List[LabelLifetimeWrite]],
    ) -> None:
        """Insert the nested children of a profile. Callers delete the old rows first."""
        for src in sources or []:
            row = WatchProfileSource(
                profile_uuid=profile_uuid,
                source_uuid=src.source_uuid,
                order=src.order,
                enabled=src.enabled,
                publish_to_ha=src.publish_to_ha,
                mqtt_name=src.mqtt_name,
                model_group_uuid=src.model_group_uuid,
                model_name=src.model_name,
                analysis_fps=src.analysis_fps,
                analysis_width=src.analysis_width,
                record_retention_s=src.record_retention_s,
                tracker_params_json=src.tracker_params_json,
                buffer_mode=src.buffer_mode.value,
                ring_seconds=src.ring_seconds,
                ring_max_mb=src.ring_max_mb,
                disabled_mask_uuids_json=[str(u) for u in (src.disabled_mask_uuids_json or [])],
            )
            session.add(row)

        for rule in rules or []:
            session.add(
                InteractionRule(
                    **({"uuid": rule.uuid} if rule.uuid else {}),
                    profile_uuid=profile_uuid,
                    order=rule.order,
                    name=rule.name,
                    enabled=rule.enabled,
                    publish_to_ha=rule.publish_to_ha,
                    publish_snapshot_to_ha=rule.publish_snapshot_to_ha,
                    source_uuids_json=[str(s) for s in (rule.source_uuids_json or [])],
                    subject_a_kind=rule.subject_a_kind.value,
                    subject_a_refs_json=[str(r) for r in (rule.subject_a_refs_json or [])],
                    subject_a_include_descendants=rule.subject_a_include_descendants,
                    relation=rule.relation.value,
                    subject_b_kind=rule.subject_b_kind.value,
                    subject_b_refs_json=[str(r) for r in (rule.subject_b_refs_json or [])],
                    subject_b_include_descendants=rule.subject_b_include_descendants,
                    threshold=rule.threshold,
                    margin=rule.margin,
                    motion_sensitivity=rule.motion_sensitivity,
                    trigger=rule.trigger.value,
                    min_duration_s=rule.min_duration_s,
                    rearm_delay_s=rule.rearm_delay_s,
                    max_duration_s=rule.max_duration_s,
                    event_label=rule.event_label,
                )
            )

        for action in actions or []:
            session.add(
                WatchAction(
                    **({"uuid": action.uuid} if action.uuid else {}),
                    profile_uuid=profile_uuid,
                    rule_uuids_json=[str(r) for r in (action.rule_uuids_json or [])],
                    source_uuids_json=[str(s) for s in (action.source_uuids_json or [])],
                    order=action.order,
                    enabled=action.enabled,
                    name=(action.name or "").strip() or None,
                    kind=action.kind,
                    params_json=action.params_json,
                    trigger=action.trigger.value,
                )
            )

        for lifetime in lifetimes or []:
            session.add(
                LabelLifetime(
                    profile_uuid=profile_uuid,
                    label_uuids_json=[str(u) for u in (lifetime.label_uuids_json or [])],
                    include_descendants=lifetime.include_descendants,
                    ttl_s=lifetime.ttl_s,
                    is_anchor=lifetime.is_anchor,
                )
            )

    async def list_watch_profiles(self) -> List[WatchProfileRead]:
        async with self._async_session_maker() as session:
            stmt = select(WatchProfile).options(*self._watch_profile_loaders())
            result = await session.exec(stmt)
            return [self._wp_to_read(p) for p in result.all()]

    async def get_watch_profile(self, uuid: str) -> Optional[WatchProfileRead]:
        async with self._async_session_maker() as session:
            stmt = select(WatchProfile).where(WatchProfile.uuid == uuid).options(*self._watch_profile_loaders())
            result = await session.exec(stmt)
            profile = result.one_or_none()
            return None if profile is None else self._wp_to_read(profile)

    async def create_watch_profile(self, data: WatchProfileCreate) -> WatchProfileRead:
        await self._validate_watch_profile(data)
        async with self._async_session_maker() as session:
            profile = WatchProfile(
                name=data.name,
                description=data.description,
                enabled=data.enabled,
                tracker_typename=data.tracker_typename,
                tracker_params_json=data.tracker_params_json,
                default_model_group_uuid=data.default_model_group_uuid,
                default_model_name=data.default_model_name,
                default_analysis_fps=data.default_analysis_fps,
                default_analysis_width=data.default_analysis_width,
            )
            session.add(profile)
            await session.flush()
            self._add_watch_children(
                session, profile.uuid, data.sources, data.rules, data.actions, data.label_lifetimes
            )
            await session.commit()

            stmt = select(WatchProfile).where(WatchProfile.uuid == profile.uuid).options(*self._watch_profile_loaders())
            result = await session.exec(stmt)
            return self._wp_to_read(result.one())

    async def update_watch_profile(self, uuid: str, data: WatchProfileUpdate) -> Optional[WatchProfileRead]:
        await self._validate_watch_profile(data)
        async with self._async_session_maker() as session:
            result = await session.exec(select(WatchProfile).where(WatchProfile.uuid == uuid))
            profile = result.one_or_none()
            if profile is None:
                return None

            scalar_fields = data.model_dump(
                exclude_unset=True,
                exclude={"sources", "rules", "actions", "label_lifetimes"},
                mode="json",
            )
            for key, value in scalar_fields.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)

            # Whole-list replace, but only for lists the caller actually sent:
            # None means "no change", [] means "remove all".
            if data.sources is not None:
                await session.exec(delete(WatchProfileSource).where(WatchProfileSource.profile_uuid == uuid))
            if data.rules is not None:
                await session.exec(delete(InteractionRule).where(InteractionRule.profile_uuid == uuid))
            if data.actions is not None:
                await session.exec(delete(WatchAction).where(WatchAction.profile_uuid == uuid))
            if data.label_lifetimes is not None:
                await session.exec(delete(LabelLifetime).where(LabelLifetime.profile_uuid == uuid))
            await session.flush()

            self._add_watch_children(session, uuid, data.sources, data.rules, data.actions, data.label_lifetimes)

            profile.updated_at = datetime.now(timezone.utc)
            session.add(profile)
            await session.commit()

            stmt = select(WatchProfile).where(WatchProfile.uuid == uuid).options(*self._watch_profile_loaders())
            result2 = await session.exec(stmt)
            return self._wp_to_read(result2.one())

    async def delete_watch_profile(self, uuid: str) -> bool:
        async with self._async_session_maker() as session:
            profile = await session.get(WatchProfile, uuid)
            if profile is None:
                return False
            await session.delete(profile)
            await session.commit()
        return True

    # ─── Watch events and recordings ──────────────────────────────────────────

    @staticmethod
    def _watch_event_to_read(event: WatchEvent) -> WatchEventRead:
        """`has_snapshot` is derived, not stored.

        The column holds a server filesystem path, which the browser can do nothing
        with and should not see; what the card needs to know is only whether there is
        an image to fetch from the snapshot endpoint.

        `recordings` is left empty here and filled in by the caller — see
        `_attach_event_recordings`. Doing it per event would be a query per card.
        """
        read = WatchEventRead.model_validate(event)
        read.has_snapshot = bool(event.snapshot_path)
        read.has_crop = bool(event.crop_path)
        return read

    async def _attach_event_recordings(self, session: Any, events: List[WatchEventRead]) -> List[WatchEventRead]:
        """Fill in each event's media list with one query for the whole page.

        The alternative — a relationship with `selectinload` — needs a foreign key from
        `recordings.event_uuid` back to `watch_events`, and that FK is deliberately
        absent: extraction runs after the event closes and outlives a user deleting it,
        so the constraint would either block the delete or take the in-flight row with
        it. One `IN` per page costs the same as the `selectinload` would have.
        """
        if not events:
            return events
        uuids = [e.uuid for e in events]
        rows = (
            await session.exec(
                select(
                    Recording.event_uuid, Recording.uuid, Recording.kind, Recording.duration_s, Recording.status
                ).where(Recording.event_uuid.in_(uuids))
                # Stable order, so the card's badges and the modal's players do not
                # reshuffle between two polls of a list that refreshes on a timer.
                .order_by(Recording.kind, Recording.created_at)
            )
        ).all()
        by_event: Dict[str, List[EventRecordingRead]] = {}
        for event_uuid, uuid, kind, duration_s, status in rows:
            by_event.setdefault(event_uuid, []).append(
                EventRecordingRead(uuid=uuid, kind=kind, duration_s=duration_s or 0.0, status=status)
            )
        for event in events:
            event.recordings = by_event.get(event.uuid, [])
        return events

    @staticmethod
    def _apply_watch_event_filters(
        stmt,
        profile_uuid: Optional[str] = None,
        source_uuid: Optional[str] = None,
        rule_uuid: Optional[str] = None,
        subject_ref: Optional[str] = None,
        target_ref: Optional[str] = None,
        source_uuids: Optional[List[str]] = None,
        rule_uuids: Optional[List[str]] = None,
        subject_refs: Optional[List[str]] = None,
        target_refs: Optional[List[str]] = None,
        exclude_source_uuids: Optional[List[str]] = None,
        exclude_rule_uuids: Optional[List[str]] = None,
        exclude_subject_refs: Optional[List[str]] = None,
        exclude_target_refs: Optional[List[str]] = None,
        start_from: Optional[datetime] = None,
        start_to: Optional[datetime] = None,
        has_recording: Optional[bool] = None,
        recording_kind: Optional[str] = None,
        has_blips: Optional[bool] = None,
        review_state: Optional[str] = None,
        pinned: Optional[bool] = None,
    ):
        """Every `watch_events` search predicate, in one place, for every reader.

        Four endpoints ask the same question of this table and differ only in what they
        do with the matching rows: `list_watch_events` pages them, `watch_event_groups`
        collapses them by identity, `watch_event_histogram` bins them by time, and
        `watch_event_sessions` walks them in order. If each assembled its own `WHERE`,
        the group counts and the grid totals would drift the first time a filter was
        added to one and not the others — the same failure mode
        `_apply_image_search_filters` exists to prevent on the image side.

        Adds no ORDER BY, LIMIT or cursor: those are the caller's, and they are the only
        thing the callers legitimately disagree about.

        Each of the four facet filters takes a list of values to include and a list to
        exclude. The singular forms are the same filter spelled for one value, folded
        into the lists below rather than applied separately, so asking for both cannot
        produce two contradictory conditions on one column.
        """

        def _include(column, singular: Optional[str], values: Optional[List[str]]):
            wanted = [v for v in ([singular] if singular else []) + list(values or []) if v]
            # Within one column the values are alternatives: an event has exactly one
            # camera, so "cameras A and B" can only mean "A or B".
            return column.in_(wanted) if wanted else None

        def _exclude(column, values: Optional[List[str]]):
            unwanted = [v for v in (values or []) if v]
            if not unwanted:
                return None
            # `NOT IN` alone drops rows where the column is NULL, because SQL compares
            # NULL to the list and gets NULL rather than true. Those are exactly the
            # events with no subject at all, which "not the cat" plainly includes — so
            # the NULL case has to be spelled out.
            return or_(column == None, column.notin_(unwanted))  # noqa: E711

        if profile_uuid:
            stmt = stmt.where(WatchEvent.profile_uuid == profile_uuid)
        for clause in (
            _include(WatchEvent.source_uuid, source_uuid, source_uuids),
            _include(WatchEvent.rule_uuid, rule_uuid, rule_uuids),
            _include(WatchEvent.subject_ref, subject_ref, subject_refs),
            _include(WatchEvent.target_ref, target_ref, target_refs),
            _exclude(WatchEvent.source_uuid, exclude_source_uuids),
            _exclude(WatchEvent.rule_uuid, exclude_rule_uuids),
            _exclude(WatchEvent.subject_ref, exclude_subject_refs),
            _exclude(WatchEvent.target_ref, exclude_target_refs),
        ):
            if clause is not None:
                stmt = stmt.where(clause)
        if start_from:
            stmt = stmt.where(WatchEvent.started_at >= start_from)
        if start_to:
            stmt = stmt.where(WatchEvent.started_at <= start_to)
        if has_recording is not None:
            # EXISTS rather than the reverse pointer this used to test, because an
            # event can own several recordings and no single column can answer it.
            # `recordings.event_uuid` is indexed, so this stays a lookup per row.
            owns_media = select(Recording.uuid).where(Recording.event_uuid == WatchEvent.uuid).exists()
            stmt = stmt.where(owns_media if has_recording else ~owns_media)
        if recording_kind:
            # Narrower than `has_recording`, and deliberately a separate parameter
            # rather than a third state of it: "owns a timelapse" says nothing about
            # whether it also owns a clip, and the two questions compose. Same
            # indexed `event_uuid` lookup, with `kind` — also indexed — alongside.
            owns_kind = (
                select(Recording.uuid)
                .where(Recording.event_uuid == WatchEvent.uuid, Recording.kind == recording_kind)
                .exists()
            )
            stmt = stmt.where(owns_kind)
        if has_blips is not None:
            # EXISTS for the same reason as recordings above: an event owns many
            # blips and no column on the event can answer it. `event_blips.event_uuid`
            # is indexed, so this stays a lookup per row rather than a scan.
            flagged = select(EventBlip.uuid).where(EventBlip.event_uuid == WatchEvent.uuid).exists()
            stmt = stmt.where(flagged if has_blips else ~flagged)
        if review_state:
            stmt = stmt.where(WatchEvent.review_state == review_state)
        if pinned is not None:
            stmt = stmt.where(WatchEvent.pinned == pinned)
        return stmt

    async def list_watch_events(
        self,
        limit: int = 200,
        offset: int = 0,
        **filters,
    ) -> Tuple[List[WatchEventRead], int]:
        """Newest first, offset paginated, with the unpaginated total.

        Offset rather than the keyset cursor the review queue uses: events are a
        bounded, append-only log ordered by a single scalar the user also filters on,
        and the page shows "N events" — which needs the count anyway.

        See `_apply_watch_event_filters` for what `**filters` accepts.
        """
        async with self._async_session_maker() as session:
            stmt = self._apply_watch_event_filters(select(WatchEvent), **filters)

            count_stmt = select(func.count()).select_from(stmt.subquery())
            total = (await session.exec(count_stmt)).one()

            stmt = stmt.order_by(WatchEvent.started_at.desc()).offset(offset).limit(limit)
            result = await session.exec(stmt)
            events = [self._watch_event_to_read(e) for e in result.all()]
            events = await self._attach_event_recordings(session, events)
            return await self._attach_event_blip_counts(session, events), int(total)

    #: The identity of an event for grouping: the rule that fired, the thing that acted,
    #: and the thing it acted on — plus the camera, which the UI folds away by default.
    #:
    #: Measured on live history: 22 distinct `(rule, subject, target)` identities cover
    #: 1277 events, and 7 of those 22 span more than one camera. Including the camera in
    #: the key takes 22 rows to 29 and renders as visibly duplicated headlines, so the
    #: grouping is always computed at the finer grain here and summed to the coarser one
    #: in the caller — one query serving both, the same trick `watch_event_counts` uses.
    _IDENTITY_COLUMNS = (
        WatchEvent.rule_uuid,
        WatchEvent.subject_ref,
        WatchEvent.target_ref,
        WatchEvent.source_uuid,
    )

    async def watch_event_groups(self, **filters) -> List[Dict[str, Any]]:
        """Collapse the filtered events to one row per identity.

        This is the whole point of the grouped review view: 1277 events are 22 things
        that happened, and a grid of 1277 tiles is the wrong unit rather than too much
        detail. Aggregating here rather than in the browser is not an optimisation — the
        counts are properties of the *whole* filtered set, and the grid only ever holds
        a page of it.

        Two queries, both `GROUP BY`s over tens of output rows:

        1. the aggregates (count, span, duration, media/blip tallies), and
        2. the representative event whose snapshot becomes the card's art.

        The representative cannot be folded into (1). SQLite's bare-column rule would
        hand it over for free with a single `max()`, but this query has several
        aggregates, and with more than one min/max the value taken is explicitly
        undefined. A `ROW_NUMBER()` window says what is actually wanted instead.
        """
        async with self._async_session_maker() as session:
            # Correlated EXISTS in a SUM, not a JOIN: an event owning both a clip and a
            # timelapse would be counted twice by a join, which turns "377 of 383 have
            # video" into a number larger than the group.
            owns_media = select(Recording.uuid).where(Recording.event_uuid == WatchEvent.uuid).exists()
            owns_blips = select(EventBlip.uuid).where(EventBlip.event_uuid == WatchEvent.uuid).exists()
            stmt = self._apply_watch_event_filters(
                select(
                    *self._IDENTITY_COLUMNS,
                    # max() over the display names rather than grouping by them: a slot
                    # renamed mid-history would otherwise split one identity in two,
                    # exactly as `watch_event_facets` guards against.
                    func.max(WatchEvent.rule_name),
                    func.max(WatchEvent.subject_name),
                    func.max(WatchEvent.target_name),
                    func.count(),
                    func.min(WatchEvent.started_at),
                    func.max(WatchEvent.started_at),
                    # Open events contribute nothing rather than a guess — the same
                    # refusal `formatDuration` makes on the card. julianday() is in days.
                    func.sum((func.julianday(WatchEvent.ended_at) - func.julianday(WatchEvent.started_at)) * 86400.0),
                    func.sum(case((owns_media, 1), else_=0)),
                    func.sum(case((owns_blips, 1), else_=0)),
                ),
                **filters,
            ).group_by(*self._IDENTITY_COLUMNS)
            rows = (await session.exec(stmt)).all()

            representatives = await self._watch_event_group_representatives(session, **filters)
            groups = []
            for (
                rule_uuid,
                subject_ref,
                target_ref,
                source_uuid,
                rule_name,
                subject_name,
                target_name,
                count,
                first_seen,
                last_seen,
                total_duration_s,
                with_recording,
                with_blips,
            ) in rows:
                key = (rule_uuid, subject_ref, target_ref, source_uuid)
                groups.append(
                    {
                        "rule_uuid": rule_uuid,
                        "rule_name": rule_name,
                        "subject_ref": subject_ref,
                        "subject_name": subject_name,
                        "target_ref": target_ref,
                        "target_name": target_name,
                        "source_uuid": source_uuid,
                        "count": int(count),
                        "first_seen": first_seen.isoformat() if first_seen is not None else None,
                        "last_seen": last_seen.isoformat() if last_seen is not None else None,
                        "total_duration_s": float(total_duration_s or 0.0),
                        "representative_uuid": representatives.get(key),
                        "with_recording": int(with_recording or 0),
                        "with_blips": int(with_blips or 0),
                    }
                )
            # Biggest first: the grouped view exists to put "383 of these" at the top of
            # the page. Ties broken by recency so the order is total and therefore stable
            # between two renders of an unchanged set — *most* recent first, which is the
            # reverse of the order `last_seen` strings sort in naturally. Two stable
            # passes rather than one clever key: `sort` is stable, so the count pass
            # preserves the recency order within each tie, and neither pass has to invent
            # a reversible encoding of a timestamp. A missing `last_seen` lands last, for
            # the same reason it does in the client's `sortGroups` — a row that has never
            # fired is not the most recent thing that happened.
            groups.sort(key=lambda g: (g["last_seen"] or "", g["rule_uuid"] or ""), reverse=True)
            groups.sort(key=lambda g: -g["count"])
            return groups

    async def _watch_event_group_representatives(self, session: Any, **filters) -> Dict[Tuple, str]:
        """Per identity, the uuid of the event whose snapshot should be the card art.

        Newest event *that has a snapshot*, falling back to the newest overall when none
        of them do. The fallback matters more than it looks: 1 event in 1277 has no card
        art, and a group whose representative happened to be that one would render as an
        empty tile for no cause the user could see.

        A croppable event outranks a merely snapshotted one for the same reason, one
        step further on: the group card shows the crop, so a representative picked
        without regard to it can leave a whole group rendering the wide shot while a
        sibling event two rows down has the close-up.
        """
        has_crop = case((WatchEvent.crop_path != None, 1), else_=0)  # noqa: E711
        has_snapshot = case((WatchEvent.snapshot_path != None, 1), else_=0)  # noqa: E711
        ranked = self._apply_watch_event_filters(
            select(
                WatchEvent.uuid,
                *self._IDENTITY_COLUMNS,
                func.row_number()
                .over(
                    partition_by=list(self._IDENTITY_COLUMNS),
                    order_by=[has_crop.desc(), has_snapshot.desc(), WatchEvent.started_at.desc()],
                )
                .label("rn"),
            ),
            **filters,
        ).subquery()
        # Columns named explicitly rather than `select(ranked)`: selecting a subquery
        # whole hands back its entity as well as its columns, and the shape of the row
        # is then something the caller has to guess at.
        rows = (
            await session.exec(
                select(
                    ranked.c.uuid,
                    ranked.c.rule_uuid,
                    ranked.c.subject_ref,
                    ranked.c.target_ref,
                    ranked.c.source_uuid,
                ).where(ranked.c.rn == 1)
            )
        ).all()
        return {
            (rule_uuid, subject_ref, target_ref, source_uuid): uuid
            for uuid, rule_uuid, subject_ref, target_ref, source_uuid in rows
        }

    async def watch_event_sessions(
        self,
        gap_s: float = DEFAULT_SESSION_GAP_S,
        limit: int = 200,
        offset: int = 0,
        **filters,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Consecutive events of one identity, collapsed into the visit they describe.

        ⚠️ **Computed here, never in the browser.** The grid is offset-paged, so a client
        that grouped what it received would split every session straddling a page
        boundary — and would group differently depending on which page you happened to
        land on. The window function sees the whole filtered set, so a session has one
        identity and one extent no matter how it is paged.

        Expect roughly a halving, not the order-of-magnitude the grouped view gives:
        measured on live history, a 120s gap turns 1277 events into 653 sessions. The
        big reduction is identity grouping (`watch_event_groups`); this is what stops one
        drink at the bowl *inside* that group looking like seven separate visits.

        SQLite has had `LAG` and windowed `SUM` since 3.25 and the container ships 3.37.
        """
        async with self._async_session_maker() as session:
            # The event's own end, or its start when it is still open — an open event
            # cannot have left a gap behind it yet.
            prev_end = func.lag(func.coalesce(WatchEvent.ended_at, WatchEvent.started_at)).over(
                partition_by=list(self._IDENTITY_COLUMNS),
                order_by=WatchEvent.started_at,
            )
            # julianday() is in days; the first event of a partition has no predecessor
            # and always opens a session.
            starts_session = case(
                (prev_end == None, 1),  # noqa: E711
                ((func.julianday(WatchEvent.started_at) - func.julianday(prev_end)) * 86400.0 > gap_s, 1),
                else_=0,
            )
            marked = self._apply_watch_event_filters(
                select(
                    WatchEvent.uuid,
                    *self._IDENTITY_COLUMNS,
                    WatchEvent.rule_name,
                    WatchEvent.subject_name,
                    WatchEvent.target_name,
                    WatchEvent.started_at,
                    WatchEvent.ended_at,
                    WatchEvent.snapshot_path,
                    WatchEvent.crop_path,
                    starts_session.label("starts_session"),
                ),
                **filters,
            ).subquery()

            # A running count of session starts *is* the session number: it increments
            # exactly where a new one begins and holds steady through the rest of it.
            sessioned = select(
                marked,
                func.sum(marked.c.starts_session)
                .over(
                    partition_by=[
                        marked.c.rule_uuid,
                        marked.c.subject_ref,
                        marked.c.target_ref,
                        marked.c.source_uuid,
                    ],
                    order_by=marked.c.started_at,
                    rows=(None, 0),
                )
                .label("session_ix"),
            ).subquery()

            # ⚠️ A separate level, not another column alongside `session_ix`. SQLite
            # rejects a window function inside another window's PARTITION BY outright
            # ("misuse of window function sum()"), and this one has to partition by the
            # session number the level below computes.
            ranked = select(
                sessioned,
                func.row_number()
                .over(
                    # Ranked within the session rather than the identity, so each session
                    # card gets art from its own visit rather than from the newest one.
                    partition_by=[
                        sessioned.c.rule_uuid,
                        sessioned.c.subject_ref,
                        sessioned.c.target_ref,
                        sessioned.c.source_uuid,
                        sessioned.c.session_ix,
                    ],
                    order_by=[
                        case((sessioned.c.crop_path != None, 1), else_=0).desc(),  # noqa: E711
                        case((sessioned.c.snapshot_path != None, 1), else_=0).desc(),  # noqa: E711
                        sessioned.c.started_at.desc(),
                    ],
                )
                .label("rn"),
            ).subquery()
            sessioned = ranked

            group_cols = [
                sessioned.c.rule_uuid,
                sessioned.c.subject_ref,
                sessioned.c.target_ref,
                sessioned.c.source_uuid,
                sessioned.c.session_ix,
            ]
            agg = (
                select(
                    *group_cols,
                    func.max(sessioned.c.rule_name).label("rule_name"),
                    func.max(sessioned.c.subject_name).label("subject_name"),
                    func.max(sessioned.c.target_name).label("target_name"),
                    func.count().label("count"),
                    func.min(sessioned.c.started_at).label("started_at"),
                    func.max(func.coalesce(sessioned.c.ended_at, sessioned.c.started_at)).label("ended_at"),
                    # `ended_at` above is coalesced, so an in-progress visit is
                    # indistinguishable from one that ended the instant it began — which
                    # is exactly what a one-event open session looks like. The card needs
                    # to say "in progress" rather than "0s", and the drill-down needs an
                    # open-ended window rather than a zero-width one, so the openness is
                    # reported separately instead of inferred from the two timestamps.
                    func.max(case((sessioned.c.ended_at == None, 1), else_=0)).label("is_open"),  # noqa: E711
                    # `rn` is 1 on exactly one row per session, so a max() over this
                    # picks that row's uuid without a second pass over the table.
                    func.max(case((sessioned.c.rn == 1, sessioned.c.uuid))).label("representative_uuid"),
                )
                .group_by(*group_cols)
                .subquery()
            )

            total = (await session.exec(select(func.count()).select_from(agg))).one()
            # Newest session first, matching the grid this replaces. Ordered by the
            # session's *start*, which is the moment the card is captioned with.
            ordered = (
                select(
                    agg.c.rule_uuid,
                    agg.c.subject_ref,
                    agg.c.target_ref,
                    agg.c.source_uuid,
                    agg.c.session_ix,
                    agg.c.rule_name,
                    agg.c.subject_name,
                    agg.c.target_name,
                    agg.c.count,
                    agg.c.started_at,
                    agg.c.ended_at,
                    agg.c.is_open,
                    agg.c.representative_uuid,
                )
                .order_by(agg.c.started_at.desc())
                .offset(offset)
                .limit(limit)
            )
            rows = (await session.exec(ordered)).all()

            sessions = []
            for (
                rule_uuid,
                subject_ref,
                target_ref,
                source_uuid,
                session_ix,
                rule_name,
                subject_name,
                target_name,
                count,
                started_at,
                ended_at,
                is_open,
                representative_uuid,
            ) in rows:
                sessions.append(
                    {
                        "rule_uuid": rule_uuid,
                        "rule_name": rule_name,
                        "subject_ref": subject_ref,
                        "subject_name": subject_name,
                        "target_ref": target_ref,
                        "target_name": target_name,
                        "source_uuid": source_uuid,
                        "session_ix": int(session_ix or 0),
                        "count": int(count),
                        "started_at": started_at.isoformat() if hasattr(started_at, "isoformat") else started_at,
                        "ended_at": ended_at.isoformat() if hasattr(ended_at, "isoformat") else ended_at,
                        "is_open": bool(is_open),
                        "representative_uuid": representative_uuid,
                    }
                )
            return sessions, int(total)

    #: Above this many histogram rows the request is answered at day resolution instead.
    #: The output of the histogram is `buckets × identities`, which is unbounded in
    #: principle — a year at hour resolution across 22 identities is ~190k rows — even
    #: though real history is sparse enough to return a few hundred.
    HISTOGRAM_ROW_LIMIT = 5000

    async def watch_event_histogram(self, bucket: str = "hour", **filters) -> Dict[str, Any]:
        """Event counts per time bucket per identity — the timeline's whole data source.

        ⚠️ **Buckets are UTC, and `hour` is the resolution the UI should ask for.**
        `started_at` carries no offset, so bucketing by *day* in SQL buckets by UTC day,
        and "today" then begins at the wrong hour for every user who is not on UTC. Hour
        buckets are safe to re-aggregate into local days in the browser for every
        whole-hour timezone, which is why the client composes local days itself rather
        than this endpoint growing a timezone parameter.

        `day` exists only as the overflow answer for very wide ranges, where a few hours
        of skew at the edges cannot change what the picture says. It is reported back in
        the response rather than applied silently: a histogram that quietly changed
        resolution — or worse, was truncated — would be a lie about when things happened.
        """
        bucket = "day" if bucket == "day" else "hour"
        async with self._async_session_maker() as session:
            for attempt in ("hour", "day") if bucket == "hour" else ("day",):
                fmt = "%Y-%m-%dT%H:00:00" if attempt == "hour" else "%Y-%m-%dT00:00:00"
                bucket_col = func.strftime(fmt, WatchEvent.started_at).label("bucket_start")
                stmt = self._apply_watch_event_filters(
                    select(bucket_col, *self._IDENTITY_COLUMNS, func.count()),
                    **filters,
                ).group_by(bucket_col, *self._IDENTITY_COLUMNS)
                rows = (await session.exec(stmt)).all()
                if len(rows) <= self.HISTOGRAM_ROW_LIMIT or attempt == "day":
                    return {
                        "bucket": attempt,
                        # Truncation is never the answer; when even day buckets overflow
                        # the caller gets the whole honest result and the flag saying the
                        # resolution was coarsened.
                        "buckets": [
                            {
                                "bucket_start": bucket_start,
                                "rule_uuid": rule_uuid,
                                "subject_ref": subject_ref,
                                "target_ref": target_ref,
                                "source_uuid": source_uuid,
                                "count": int(count),
                            }
                            for bucket_start, rule_uuid, subject_ref, target_ref, source_uuid, count in rows
                        ],
                    }
            return {"bucket": bucket, "buckets": []}

    async def watch_event_facets(self, profile_uuid: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
        """The distinct values each filter dropdown can offer, with display names.

        Derived from the events themselves rather than from the watcher config, because
        the two diverge: a rule deleted last month still has a month of history behind
        it, and a rule added this morning has none. Offering config values would show
        filters that match nothing and hide the ones that match everything.
        """

        async def _distinct(ref_col, name_col) -> List[Dict[str, str]]:
            stmt = select(ref_col, func.max(name_col)).where(ref_col != None)  # noqa: E711
            if profile_uuid:
                stmt = stmt.where(WatchEvent.profile_uuid == profile_uuid)
            # max() over the name, not group-by both: a slot renamed mid-history would
            # otherwise appear as two entries for one thing.
            stmt = stmt.group_by(ref_col)
            rows = (await session.exec(stmt)).all()
            return [{"ref": ref, "name": name or ref} for ref, name in rows if ref]

        async def _sources() -> List[Dict[str, str]]:
            # Joined, not denormalised like the other three: a camera is a live entity
            # the user renames in Sources and expects to see renamed here, whereas a
            # rule name is a historical fact about an event that already happened.
            stmt = (
                select(WatchEvent.source_uuid, func.max(Source.name))
                .join(Source, Source.uuid == WatchEvent.source_uuid)
                .where(WatchEvent.source_uuid != None)  # noqa: E711
            )
            if profile_uuid:
                stmt = stmt.where(WatchEvent.profile_uuid == profile_uuid)
            rows = (await session.exec(stmt.group_by(WatchEvent.source_uuid))).all()
            return [{"ref": ref, "name": name or ref} for ref, name in rows if ref]

        async with self._async_session_maker() as session:
            return {
                "rules": await _distinct(WatchEvent.rule_uuid, WatchEvent.rule_name),
                "subjects": await _distinct(WatchEvent.subject_ref, WatchEvent.subject_name),
                "targets": await _distinct(WatchEvent.target_ref, WatchEvent.target_name),
                "sources": await _sources(),
            }

    async def watch_event_counts(self) -> List[Dict[str, Any]]:
        """Lifetime event counts and last-seen times grouped by (source, rule, subject).

        For MQTT bridge counter seeding — a bridge that just (re)connected needs each
        rule's lifetime total, not zero, so its first published count doesn't read as
        a reset.

        Read off `watch_event_counters`, never as a `COUNT(*)` over `watch_events`:
        that table is deleted from by the retention sweep, the size-cap eviction and a
        profile-save cascade, and a count that falls is read by Home Assistant as a
        counter reset and added on top of the running total. See `WatchEventCounter`.

        These are the finest grain the bridge publishes; it sums them itself into the
        per-subject and per-(subject, rule) tiers. Storing one row per tier instead
        would be three tallies free to disagree.

        `subject_ref` comes back `None` for the rule-level tally, matching the nullable
        column on `watch_events`; the table stores `""` there, for the primary-key
        reason in `WatchEventCounter`.
        """
        async with self._async_session_maker() as session:
            rows = (await session.exec(select(WatchEventCounter))).all()
            return [
                {
                    "source_uuid": row.source_uuid,
                    "rule_uuid": row.rule_uuid,
                    "subject_ref": row.subject_ref or None,
                    "count": row.count,
                    "last_seen": row.last_seen.isoformat() if row.last_seen is not None else None,
                }
                for row in rows
            ]

    async def get_watch_event(self, uuid: str) -> Optional[WatchEventRead]:
        async with self._async_session_maker() as session:
            event = await session.get(WatchEvent, uuid)
            if event is None:
                return None
            read = self._watch_event_to_read(event)
            attached = await self._attach_event_recordings(session, [read])
            return (await self._attach_event_blip_counts(session, attached))[0]

    # ─── event blips ─────────────────────────────────────────────────────────

    async def _attach_event_blip_counts(self, session: Any, events: List[WatchEventRead]) -> List[WatchEventRead]:
        """Fill in each event's blip tally with one grouped query for the whole page.

        Same shape and the same reasoning as `_attach_event_recordings`: the grid draws
        a badge per event, and a query per card would make the list cost scale with the
        page size.

        Counts harvestable (detector-fault) blips, plus `identity_switch` — the one
        tracker-fault kind with a real review action of its own (the crop-compare
        prompt in `blipsPanel.js`, confirm-or-dismiss) — still in `new`/`confirmed`.
        Every other tracker-fault blip carries no actionable content (the detector
        already got those frames right) so it never belongs in a badge that means
        "something to review here", and a harvested/dismissed/resolved blip is a
        problem the user already handled.
        """
        if not events:
            return events
        uuids = [e.uuid for e in events]
        rows = (
            await session.exec(
                select(EventBlip.event_uuid, EventBlip.kind, func.count())
                .where(EventBlip.event_uuid.in_(uuids))
                .where(or_(EventBlip.harvestable == True, EventBlip.kind == KIND_IDENTITY_SWITCH))  # noqa: E712
                .where(EventBlip.state.not_in([BlipState.HARVESTED.value, BlipState.DISMISSED.value]))
                .group_by(EventBlip.event_uuid, EventBlip.kind)
            )
        ).all()
        by_event: Dict[str, Dict[str, int]] = {}
        for event_uuid, kind, count in rows:
            by_event.setdefault(event_uuid, {})[kind] = int(count)
        for event in events:
            event.blip_counts = by_event.get(event.uuid, {})
        return events

    async def replace_event_blips(
        self,
        event_uuid: str,
        blips: Sequence[Dict[str, Any]],
        recording_uuid: Optional[str] = None,
    ) -> List[EventBlipRead]:
        """Write an analysis result, replacing any previous one for the same recording.

        Replace rather than append, because analysis is deterministic over a sidecar
        that does not change: re-running it must converge rather than accumulate. The
        delete is scoped to `recording_uuid` when one is given so that re-analysing an
        event's clip does not discard the blips found in its timelapse.

        Triage state is *not* preserved across a replace. That is deliberate and worth
        stating: if the analyser now says something different about a span, the earlier
        decision was made about a different claim, and silently carrying "dismissed"
        onto a new finding would hide it.
        """
        async with self._async_session_maker() as session:
            stmt = delete(EventBlip).where(EventBlip.event_uuid == event_uuid)
            if recording_uuid:
                stmt = stmt.where(EventBlip.recording_uuid == recording_uuid)
            await session.exec(stmt)

            rows = [
                EventBlip(
                    event_uuid=event_uuid,
                    recording_uuid=recording_uuid,
                    kind=str(blip.get("kind") or ""),
                    track_id=int(blip.get("trackId", blip.get("track_id", 0)) or 0),
                    start_s=float(blip.get("startS", blip.get("start_s", 0.0)) or 0.0),
                    end_s=float(blip.get("endS", blip.get("end_s", 0.0)) or 0.0),
                    severity=float(blip.get("severity", 0.0) or 0.0),
                    fault=str(blip.get("fault") or ""),
                    harvestable=bool(blip.get("harvestable", False)),
                    detail_json=dict(blip.get("detail") or {}),
                )
                for blip in blips
            ]
            for row in rows:
                session.add(row)
            await session.commit()
            for row in rows:
                await session.refresh(row)
            return [EventBlipRead.model_validate(row) for row in rows]

    async def list_event_blips(
        self,
        uuids: Optional[Sequence[str]] = None,
        event_uuid: Optional[str] = None,
        recording_uuid: Optional[str] = None,
        kinds: Optional[Sequence[str]] = None,
        states: Optional[Sequence[str]] = None,
        harvestable: Optional[bool] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> Tuple[List[EventBlipRead], int]:
        """Worst first, with the unpaginated total.

        Ordered by severity descending rather than by time, because severity here means
        "how sure the detector was about what it got wrong" — so the most valuable
        training data sorts to the top of the queue rather than merely the earliest.
        """
        async with self._async_session_maker() as session:
            stmt = select(EventBlip)
            if uuids:
                stmt = stmt.where(EventBlip.uuid.in_(list(uuids)))
            if event_uuid:
                stmt = stmt.where(EventBlip.event_uuid == event_uuid)
            if recording_uuid:
                stmt = stmt.where(EventBlip.recording_uuid == recording_uuid)
            if kinds:
                stmt = stmt.where(EventBlip.kind.in_(list(kinds)))
            if states:
                stmt = stmt.where(EventBlip.state.in_(list(states)))
            if harvestable is not None:
                stmt = stmt.where(EventBlip.harvestable == harvestable)

            count_stmt = select(func.count()).select_from(stmt.subquery())
            total = (await session.exec(count_stmt)).one()

            stmt = stmt.order_by(EventBlip.severity.desc(), EventBlip.start_s).offset(offset).limit(limit)
            rows = (await session.exec(stmt)).all()
            return [EventBlipRead.model_validate(row) for row in rows], int(total)

    async def set_event_blip_states(self, uuids: Sequence[str], state: str) -> int:
        """Bulk triage. Returns how many rows actually moved."""
        if not uuids:
            return 0
        async with self._async_session_maker() as session:
            result = await session.exec(sa_update(EventBlip).where(EventBlip.uuid.in_(list(uuids))).values(state=state))
            await session.commit()
            return int(result.rowcount or 0)

    async def get_blip_crop_preview(self, blip_uuid: str) -> Dict[str, Any]:
        """Before/after crops around an `identity_switch` blip's own instant.

        Lets a reviewer see the two without opening the clip at all: is this really the
        same animal carrying a wrong label for a stretch (a tracker mistake — dismiss it),
        or a genuine identity change worth correcting (relabel it)? "Before" is half a
        second ahead of `start_s` under the track's `modalLabel`; "after" is `start_s`
        itself under `sawLabel` — the two labels `detail_json` already recorded for this
        blip, from `_identity_blips` in `apps.helpers.recordings.blips`.

        Returns `{"status": "failure", "message": ...}` for any reason a picture can't be
        produced (blip gone, clip swept, sidecar missing, frame undecodable) rather than
        raising — this backs a preview a user is looking at, not a durable artifact.
        """
        blips, _ = await self.list_event_blips(uuids=[blip_uuid], limit=1)
        if not blips:
            return {"status": JsonValues.FAILURE, "message": "that finding no longer exists"}
        blip = blips[0]
        if not blip.recording_uuid:
            return {"status": JsonValues.FAILURE, "message": "this finding has no clip attached"}

        recording = await self.get_recording(blip.recording_uuid)
        if recording is None or not recording.file_path or not recording.overlay_path:
            return {"status": JsonValues.FAILURE, "message": "the clip is no longer available"}

        try:
            document = json.loads(Path(recording.overlay_path).read_text())
        except OSError:
            return {"status": JsonValues.FAILURE, "message": "the clip's detection sidecar is missing"}

        detail = blip.detail_json or {}
        modal_label = str(detail.get("modalLabel") or "")
        saw_label = str(detail.get("sawLabel") or "")
        # Half a second back, not the track's first frame: the switch is a boundary, and
        # what matters is the pose right up against it, not the start of the whole visit.
        before_s = max(0.0, blip.start_s - 0.5)

        before, after = await asyncio.gather(
            asyncio.to_thread(_crop_blip_frame, recording.file_path, document, blip.track_id, modal_label, before_s),
            asyncio.to_thread(_crop_blip_frame, recording.file_path, document, blip.track_id, saw_label, blip.start_s),
        )
        if before is None and after is None:
            return {"status": JsonValues.FAILURE, "message": "could not read any frames from the clip"}
        return {"status": JsonValues.SUCCESS, "before": before, "after": after}

    async def set_watch_event_review_states(self, uuids: Sequence[str], state: str) -> int:
        """Bulk triage for events, mirroring `set_event_blip_states` for blips.

        Turns the events page from a log into a queue. The column, the read model and the
        `review_state` search filter all predate this — every one of the 1277 rows in
        live history reads `unreviewed` because nothing could ever write it.

        Returns how many rows actually moved; lower than requested is normal (an event
        already in the target state, or deleted from another tab), not an error.
        """
        if not uuids:
            return 0
        async with self._async_session_maker() as session:
            result = await session.exec(
                sa_update(WatchEvent).where(WatchEvent.uuid.in_(list(uuids))).values(review_state=state)
            )
            await session.commit()
            return int(result.rowcount or 0)

    async def set_watch_event_pinned(self, uuids: Sequence[str], pinned: bool) -> int:
        """Bulk pin/unpin, mirroring `set_watch_event_review_states`.

        Returns how many rows actually moved; lower than requested is normal (a uuid
        already in the target state, or deleted from another tab), not an error.
        """
        if not uuids:
            return 0
        async with self._async_session_maker() as session:
            result = await session.exec(
                sa_update(WatchEvent).where(WatchEvent.uuid.in_(list(uuids))).values(pinned=pinned)
            )
            await session.commit()
            return int(result.rowcount or 0)

    async def expire_blips_for_recording(self, recording_uuid: str) -> int:
        """Mark a swept clip's blips unreviewable, without deleting them.

        Retention deletes the video on its own schedule; the finding outlives it. The
        row stays so the record of what went wrong survives, but nothing will offer to
        review or harvest it, because the evidence is gone.

        Only untriaged blips move. One already harvested or dismissed has a real
        outcome, and overwriting that with "expired" would lose it.
        """
        async with self._async_session_maker() as session:
            result = await session.exec(
                sa_update(EventBlip)
                .where(EventBlip.recording_uuid == recording_uuid)
                .where(EventBlip.state.in_([BlipState.NEW.value, BlipState.CONFIRMED.value]))
                .values(state=BlipState.EXPIRED.value)
            )
            await session.commit()
            return int(result.rowcount or 0)

    async def get_watch_event_snapshot_path(self, uuid: str) -> Optional[str]:
        """The raw path, for the endpoint that streams the file. Deliberately its own
        method so the path never rides out on the read model by accident."""
        async with self._async_session_maker() as session:
            event = await session.get(WatchEvent, uuid)
            return None if event is None else event.snapshot_path

    async def get_watch_event_crop_path(self, uuid: str) -> Optional[str]:
        """The cropped card art's path. Same contract as the snapshot's."""
        async with self._async_session_maker() as session:
            event = await session.get(WatchEvent, uuid)
            return None if event is None else event.crop_path

    async def create_watch_event(self, event: WatchEvent) -> WatchEventRead:
        """Insert the event. The lifetime tally is bumped when it *closes*.

        Not here, even though this is where the row appears. An open event's subject is
        still the watcher's running opinion, and it is revised at the closing edge to
        whichever label the visit actually weighed most — so a tally taken now would
        count a `Person` for an event whose card ends up reading `Alice`, and the two would
        disagree permanently, because a `total_increasing` counter cannot be walked back.

        See `_close_watch_event_counter`.
        """
        async with self._async_session_maker() as session:
            session.add(event)
            await session.commit()
            await session.refresh(event)
            return self._watch_event_to_read(event)

    async def _close_watch_event_counter(self, session: Any, event: WatchEvent) -> None:
        """`+1` on the closing event's counter row, creating it at 1 if this pair is new.

        Called once per event, at the moment it closes, so the tally counts the subject
        the event was finally decided to be. Events missing either uuid are counted by
        nothing, matching the filter `watch_event_counts` has always applied: a tally
        keyed on a NULL has no Home Assistant entity to publish to.

        An event that never closes is never counted — a watcher killed hard enough that
        its shutdown sweep does not run leaves the row open, and one uncounted visit is a
        far smaller lie than a count that disagrees with every card under it.

        An upsert rather than read-modify-write because two watchers can close an event
        on the same pair concurrently, and `count = count + 1` evaluated in SQLite is
        atomic where a Python round-trip would lose one of them.
        """
        if event.source_uuid is None or event.rule_uuid is None:
            return
        stmt = sqlite_insert(WatchEventCounter).values(
            source_uuid=event.source_uuid,
            rule_uuid=event.rule_uuid,
            subject_ref=event.subject_ref or "",
            count=1,
            last_seen=event.started_at,
        )
        counter = WatchEventCounter.__table__.c
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                WatchEventCounter.source_uuid,
                WatchEventCounter.rule_uuid,
                WatchEventCounter.subject_ref,
            ],
            set_={
                "count": counter.count + 1,
                # Events do not always arrive in start order — a long event closes after
                # a short one that began later — and `last_seen` must never walk back.
                # The coalesce is load-bearing: SQLite's `max()` returns NULL if either
                # argument is NULL, so a first-ever row would erase its own timestamp.
                "last_seen": func.max(
                    func.coalesce(counter.last_seen, stmt.excluded.last_seen),
                    stmt.excluded.last_seen,
                ),
            },
        )
        await session.exec(stmt)

    async def update_watch_event(self, uuid: str, **fields) -> Optional[WatchEventRead]:
        async with self._async_session_maker() as session:
            event = await session.get(WatchEvent, uuid)
            if event is None:
                return None
            # Read before the fields are applied: the tally is owed on the *transition*
            # from open to closed, and applying first would make every later correction
            # to an already-closed event look like another close and count it again.
            was_open = event.ended_at is None
            for key, value in fields.items():
                if hasattr(event, key):
                    setattr(event, key, value)
            session.add(event)
            if was_open and event.ended_at is not None:
                await self._close_watch_event_counter(session, event)
            await session.commit()
            await session.refresh(event)
            read = self._watch_event_to_read(event)
            attached = await self._attach_event_recordings(session, [read])
            return (await self._attach_event_blip_counts(session, attached))[0]

    async def delete_watch_event(self, uuid: str) -> bool:
        async with self._async_session_maker() as session:
            event = await session.get(WatchEvent, uuid)
            if event is None:
                return False
            if event.pinned:
                raise WatchEventPinnedError(f"Event {uuid} is pinned — unpin it before deleting")
            await session.delete(event)
            await session.commit()
        return True

    async def watch_event_uuids_before(self, cutoff: datetime, limit: int = 500) -> List[str]:
        """Uuids of events that started before *cutoff*, oldest first.

        Uuids rather than rows: the caller deletes them one at a time so it can unlink
        each event's files, and hydrating full read models to throw them away is waste.
        Batched because a watcher left running over a long holiday can accumulate more
        expired events than anyone wants to delete inside one worker tick.

        Excludes pinned events. Both the age-based retention sweep and the size-cap
        eviction page through this one query, so filtering here is the single choke
        point that keeps a pin out of every automatic deletion path at once — see
        `WatchEvent.pinned`.
        """
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(WatchEvent.uuid)
                .where(WatchEvent.started_at < cutoff, WatchEvent.pinned == False)  # noqa: E712
                .order_by(WatchEvent.started_at)
                .limit(limit)
            )
            return list(result.all())

    async def recording_uuids_before(self, cutoff: datetime, limit: int = 500) -> List[str]:
        """Uuids of recordings that started before *cutoff*, oldest first.

        Falls back to `created_at` when `started_at` is NULL: a clip whose extraction
        failed before the relay reported a start time still occupies disk, and a NULL
        compares false against any cutoff, so it would otherwise never expire.

        Excludes recordings that belong to a pinned event — a clip is that event's
        video, so the size cap's "no events left to shed, evict orphan recordings
        instead" fallback must not reach a clip the event phase was made to skip.
        Recordings with no event (`event_uuid` NULL) or an already-deleted one are
        unaffected: the NOT EXISTS is true for both.
        """
        started = func.coalesce(Recording.started_at, Recording.created_at)
        not_pinned = ~(
            select(WatchEvent.uuid)
            .where(WatchEvent.uuid == Recording.event_uuid, WatchEvent.pinned == True)  # noqa: E712
            .exists()
        )
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(Recording.uuid).where(started < cutoff, not_pinned).order_by(started).limit(limit)
            )
            return list(result.all())

    async def referenced_recording_paths(self) -> Set[str]:
        """Every file path any surviving row points at.

        The orphan sweep's safety rail: a file under the recordings roots that is not in
        this set has no row and nothing will ever show it, so deleting it loses nothing.
        Fetched as one set rather than asked per file, because the question is asked once
        per file on disk and the answer does not change during a sweep.

        Deliberately covers every column that can name a file — a clip, its thumbnail,
        its detection-overlay sidecar, and an event's snapshot and crop. Missing one
        would make the sweep delete live files, which is the failure mode that makes
        directory-walking sweeps dangerous.
        """
        paths: Set[str] = set()
        async with self._async_session_maker() as session:
            result = await session.exec(select(Recording.file_path, Recording.thumbnail_path, Recording.overlay_path))
            for file_path, thumbnail_path, overlay_path in result.all():
                paths.update(p for p in (file_path, thumbnail_path, overlay_path) if p)
            result = await session.exec(select(WatchEvent.snapshot_path, WatchEvent.crop_path))
            for snapshot_path, crop_path in result.all():
                paths.update(p for p in (snapshot_path, crop_path) if p)
        return paths

    async def rows_naming_recording_files(self) -> List[Dict[str, Any]]:
        """Every row that names a media file, for the reverse of the orphan sweep.

        The orphan sweep asks "is there a row for this file?"; this answers "is there a
        file for this row?", which is the question a user deleting clips in a file
        manager creates. Both are needed: neither direction repairs the other's damage.

        Returned as plain dicts rather than read models because the caller only stats
        paths and deletes rows, and the read models deliberately omit `snapshot_path`.
        `status` and `created_at` come along because a row is *expected* to have no file
        while it is still being written — without them the reconciler would race clip
        extraction and delete the row of a clip being cut right now.

        `overlay_path` is deliberately *not* listed, unlike in `referenced_recording_paths`.
        The caller keeps a row when any of its paths still exists, so a sidecar surviving
        a hand-deleted mp4 would hold the row open and leave a card whose video 404s. A
        sidecar is derived from its clip and never worth keeping the clip's row alive for;
        once the row goes the orphan sweep collects it.
        """
        rows: List[Dict[str, Any]] = []
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(
                    Recording.uuid,
                    Recording.file_path,
                    Recording.thumbnail_path,
                    Recording.status,
                    Recording.created_at,
                )
            )
            for uuid, file_path, thumbnail_path, status, created_at in result.all():
                if file_path or thumbnail_path:
                    rows.append(
                        {
                            "kind": "recording",
                            "uuid": uuid,
                            "paths": [p for p in (file_path, thumbnail_path) if p],
                            "status": status,
                            "at": created_at,
                        }
                    )
            # ⚠️ An event's two images get a row each, never one row naming both. The
            # caller keeps a row when *any* of its paths still exists, so pairing them
            # would make a surviving snapshot hold a deleted crop's column open forever
            # — the card would then keep asking for a file nobody can produce again.
            result = await session.exec(
                select(WatchEvent.uuid, WatchEvent.snapshot_path, WatchEvent.crop_path, WatchEvent.started_at).where(
                    or_(WatchEvent.snapshot_path.is_not(None), WatchEvent.crop_path.is_not(None))
                )
            )
            for uuid, snapshot_path, crop_path, started_at in result.all():
                for kind, path in (("event_snapshot", snapshot_path), ("event_crop", crop_path)):
                    if path:
                        rows.append(
                            {
                                "kind": kind,
                                "uuid": uuid,
                                "paths": [path],
                                # Both images are written synchronously when the event
                                # opens, so there is no in-progress state to exempt the
                                # way a clip has.
                                "status": None,
                                "at": started_at,
                            }
                        )
        return rows

    async def close_watch_event(
        self,
        uuid: str,
        ended_at: datetime,
        peak_confidence: Optional[float] = None,
    ) -> Optional[WatchEventRead]:
        """Close an event. Media is attached separately, via `recordings.event_uuid`.

        This used to take the clip's uuid too, from when an event pointed at one
        recording. It never had a caller that passed it — the recorder always patched
        the event in its own request — and an event can now own several, so the
        parameter had nothing left to mean.
        """
        async with self._async_session_maker() as session:
            event = await session.get(WatchEvent, uuid)
            if event is None:
                return None
            event.ended_at = ended_at
            if peak_confidence is not None:
                event.peak_confidence = peak_confidence
            session.add(event)
            await session.commit()
            await session.refresh(event)
            read = self._watch_event_to_read(event)
            attached = await self._attach_event_recordings(session, [read])
            return (await self._attach_event_blip_counts(session, attached))[0]

    async def list_open_watch_events(self, profile_uuid: Optional[str] = None) -> List[WatchEventRead]:
        """Events with no `ended_at` — what resume has to close out after a restart."""
        async with self._async_session_maker() as session:
            stmt = select(WatchEvent).where(WatchEvent.ended_at == None)  # noqa: E711
            if profile_uuid:
                stmt = stmt.where(WatchEvent.profile_uuid == profile_uuid)
            result = await session.exec(stmt)
            return [self._watch_event_to_read(e) for e in result.all()]

    async def close_stranded_watch_events(self, profile_uuid: str) -> int:
        """Close every still-open event of a profile. Returns how many there were.

        Called by a watcher as it starts, which is the moment the claim "this event is in
        progress" is provably false: this process owns the profile's sources and has just
        begun, so nothing else is holding an event open. Only a killed watcher leaves one
        — a clean stop closes its own — and until this existed nothing ever closed them,
        so a row could sit in the Events list saying "in progress" indefinitely.

        `ended_at` is set to `started_at`, which reads as a zero-length event. That is
        deliberate: the true end is somewhere between the start and now, and there is no
        record of where. Guessing `now` would be worse — it claims a visit that lasted
        however long the watcher was down, which is exactly the kind of invented history
        the events list is supposed to be trustworthy against. The metadata flag is what a
        card can use to say "the watcher stopped before this event closed" rather than
        presenting the duration as measured.
        """
        async with self._async_session_maker() as session:
            stmt = select(WatchEvent).where(
                WatchEvent.profile_uuid == profile_uuid,
                WatchEvent.ended_at == None,  # noqa: E711
            )
            rows = (await session.exec(stmt)).all()
            for event in rows:
                event.ended_at = event.started_at
                # Recorded in the same column as every other forced ending, so anything
                # asking "why did this end?" gets one answer in one place. `truncated`
                # is deliberately *not* set: it means "the span is real but cut short",
                # and this row's span is not real at all — `ended_at` is a placeholder.
                # The card has its own, blunter treatment for that, and both badges at
                # once would say the same thing twice.
                event.close_reason = EventCloseReason.FORCED.value
                # Reassigned, not mutated in place: a JSON column tracks changes by
                # identity, so mutating the dict leaves the row unchanged on commit.
                event.metadata_json = {**(event.metadata_json or {}), "closedBy": "restart"}
                session.add(event)
            if rows:
                await session.commit()
            return len(rows)

    async def list_recordings(
        self,
        profile_uuid: Optional[str] = None,
        source_uuid: Optional[str] = None,
        event_uuid: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 200,
    ) -> List[RecordingRead]:
        async with self._async_session_maker() as session:
            stmt = select(Recording)
            if profile_uuid:
                stmt = stmt.where(Recording.profile_uuid == profile_uuid)
            if source_uuid:
                stmt = stmt.where(Recording.source_uuid == source_uuid)
            if event_uuid:
                stmt = stmt.where(Recording.event_uuid == event_uuid)
            if kind:
                stmt = stmt.where(Recording.kind == kind)
            stmt = stmt.order_by(Recording.started_at.desc()).limit(limit)
            result = await session.exec(stmt)
            return [RecordingRead.model_validate(r) for r in result.all()]

    async def recording_uuids_for_event(self, event_uuid: str) -> List[str]:
        """Every recording naming this event.

        Uuids rather than rows: the two callers both delete or stat them one at a time,
        and hydrating read models to throw away is the same waste
        `watch_event_uuids_before` avoids.
        """
        async with self._async_session_maker() as session:
            result = await session.exec(select(Recording.uuid).where(Recording.event_uuid == event_uuid))
            return list(result.all())

    async def get_recording(self, uuid: str) -> Optional[RecordingRead]:
        async with self._async_session_maker() as session:
            rec = await session.get(Recording, uuid)
            return None if rec is None else RecordingRead.model_validate(rec)

    async def create_recording(self, recording: Recording) -> RecordingRead:
        async with self._async_session_maker() as session:
            session.add(recording)
            await session.commit()
            await session.refresh(recording)
            return RecordingRead.model_validate(recording)

    async def update_recording(self, uuid: str, **fields) -> Optional[RecordingRead]:
        async with self._async_session_maker() as session:
            rec = await session.get(Recording, uuid)
            if rec is None:
                return None
            for key, value in fields.items():
                if hasattr(rec, key):
                    setattr(rec, key, value)
            rec.updated_at = datetime.now(timezone.utc)
            session.add(rec)
            await session.commit()
            await session.refresh(rec)
            return RecordingRead.model_validate(rec)

    async def delete_recording(self, uuid: str) -> bool:
        """Delete one recording row.

        Refuses a recording that belongs to a pinned event — a clip *is* that event's
        video, so exempting the event's own row while leaving its clip reachable
        through this path would make the pin meaningless. Recordings with no event, or
        whose event has since been deleted, have nothing to check and are unaffected.
        """
        async with self._async_session_maker() as session:
            rec = await session.get(Recording, uuid)
            if rec is None:
                return False
            if rec.event_uuid:
                event = await session.get(WatchEvent, rec.event_uuid)
                if event is not None and event.pinned:
                    raise WatchEventPinnedError(
                        f"Recording {uuid} belongs to pinned event {rec.event_uuid} — unpin it before deleting"
                    )
            await session.delete(rec)
            await session.commit()
        return True

    # ── Preview runs & candidates ──────────────────────────────────────────
    # Scratch state for persisted "Run Model on Results" previews. Nothing here
    # is an annotation: a candidate becomes a fact only when the apply path
    # writes it through the normal bbox methods.
    #
    # ⚠️ Pagination here is **offset**, not the keyset cursor the review queue
    # uses. Two reasons: a run is a bounded, closed set (thousands of rows,
    # already fully written before triage starts, indexed by run_uuid), and the
    # default sort — "group every candidate from one image together, images
    # ranked by their best candidate" — is a composite
    # (image_max_conf, image_uuid, confidence, uuid) with no single scalar cursor
    # value. A keyset over that would need the whole tuple threaded through the
    # API for no measurable win at this size.

    async def create_preview_run(
        self,
        name: str,
        model_name: str,
        task_typename: str,
        params: Optional[Dict[str, Any]] = None,
        scope: Optional[Dict[str, Any]] = None,
        ttl_days: int = PreviewRunTTL.DEFAULT_DAYS,
    ) -> PreviewRun:
        run = PreviewRun(
            name=name,
            model_name=model_name,
            task_typename=task_typename,
            params_json=params or {},
            scope_json=scope or {},
            expires_at=datetime.now(timezone.utc) + timedelta(days=ttl_days),
        )
        async with self._async_session_maker() as session:
            session.add(run)
            await session.commit()
            await session.refresh(run)
        return run

    async def get_preview_run(self, run_uuid: str) -> Optional[PreviewRun]:
        async with self._async_session_maker() as session:
            return await session.get(PreviewRun, run_uuid)

    async def get_preview_runs(self) -> List[PreviewRun]:
        """Newest first — the saved-runs list is a recency list."""
        async with self._async_session_maker() as session:
            result = await session.exec(select(PreviewRun).order_by(PreviewRun.created_at.desc()))
            return list(result.all())

    async def update_preview_run(
        self,
        run_uuid: str,
        name: Optional[str] = None,
        pinned: Optional[bool] = None,
        stats: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[PreviewRun]:
        async with self._async_session_maker() as session:
            run = await session.get(PreviewRun, run_uuid)
            if run is None:
                return None
            if name is not None:
                run.name = name
            if pinned is not None:
                run.pinned = pinned
            if stats is not None:
                run.stats_json = stats
            if expires_at is not None:
                run.expires_at = expires_at
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return run

    async def delete_preview_runs(self, run_uuids: Union[str, List[str]]) -> int:
        if isinstance(run_uuids, str):
            run_uuids = [run_uuids]
        if not run_uuids:
            return 0
        async with self._async_session_maker() as session:
            # Explicit candidate delete first: SQLite only honours ON DELETE
            # CASCADE when foreign_keys pragma is on, and a core DELETE bypasses
            # the ORM's delete-orphan cascade either way.
            await session.exec(delete(PreviewCandidate).where(PreviewCandidate.run_uuid.in_(run_uuids)))
            result = await session.exec(delete(PreviewRun).where(PreviewRun.uuid.in_(run_uuids)))
            await session.commit()
            return result.rowcount or 0

    async def sweep_expired_preview_runs(self) -> int:
        """Delete unpinned runs past their expiry. Called lazily from the list
        endpoint — a scheduler for something this cheap would be its own bug source.

        A run whose candidates are all `applied`/`rejected` expires early
        (PreviewRunTTL.TRIAGED_HOURS): there is nothing left to come back to.
        """
        now = datetime.now(timezone.utc)
        triaged_cutoff = now - timedelta(hours=PreviewRunTTL.TRIAGED_HOURS)
        async with self._async_session_maker() as session:
            expired = await session.exec(
                select(PreviewRun.uuid).where(
                    PreviewRun.pinned == False,  # noqa: E712 — SQL expression, not a truth test
                    PreviewRun.expires_at.is_not(None),
                    PreviewRun.expires_at <= now,
                )
            )
            doomed = set(expired.all())

            # Fully-triaged runs older than the short TTL. "Fully triaged" is
            # expressed as NOT EXISTS an untriaged candidate, so a run with zero
            # candidates (a model that proposed nothing) is swept too.
            untriaged = (
                select(PreviewCandidate.uuid)
                .where(
                    PreviewCandidate.run_uuid == PreviewRun.uuid,
                    PreviewCandidate.decision.notin_([PreviewDecision.APPLIED, PreviewDecision.REJECTED]),
                )
                .exists()
            )
            stale = await session.exec(
                select(PreviewRun.uuid).where(
                    PreviewRun.pinned == False,  # noqa: E712
                    PreviewRun.updated_at <= triaged_cutoff,
                    ~untriaged,
                )
            )
            doomed.update(stale.all())

        return await self.delete_preview_runs(list(doomed)) if doomed else 0

    async def add_preview_candidates(
        self,
        run_uuid: str,
        candidates: Sequence[PreviewCandidateCreate],
        conflict_policy: PreviewConflictPolicy = PreviewConflictPolicy.LAST,
    ) -> int:
        """Insert candidates for a run, resolving multi-model conflicts.

        Several models running over the same scope can each propose on the same
        bbox. `LAST` (the default, and what the client's merge loop always did
        silently) overwrites the earlier proposal; `FIRST` keeps it. Conflicts are
        keyed on (run_uuid, bbox_uuid) and only apply to *existing* boxes — two
        models proposing new boxes are two separate detections, not a conflict.

        Returns the number of rows written (updates included).
        """
        if not candidates:
            return 0

        async with self._async_session_maker() as session:
            # One read of this run's existing bbox-keyed rows, rather than a
            # lookup per candidate: a run writes candidates in batches of
            # thousands and per-row SELECTs dominated the task's runtime.
            existing_result = await session.exec(
                select(PreviewCandidate).where(
                    PreviewCandidate.run_uuid == run_uuid,
                    PreviewCandidate.bbox_uuid.is_not(None),
                )
            )
            by_bbox: Dict[str, PreviewCandidate] = {c.bbox_uuid: c for c in existing_result.all()}

            written = 0
            for c in candidates:
                prior = by_bbox.get(c.bbox_uuid) if c.bbox_uuid else None
                if prior is not None:
                    if conflict_policy == PreviewConflictPolicy.FIRST:
                        continue
                    prior.proposed_label_uuid = c.proposed_label_uuid
                    prior.confidence = c.confidence
                    prior.geometry_json = c.geometry
                    prior.source_model = c.source_model
                    prior.current_label_uuid = c.current_label_uuid
                    session.add(prior)
                    written += 1
                    continue

                row = PreviewCandidate(
                    run_uuid=run_uuid,
                    image_uuid=c.image_uuid,
                    bbox_uuid=c.bbox_uuid,
                    is_new=c.is_new,
                    current_label_uuid=c.current_label_uuid,
                    proposed_label_uuid=c.proposed_label_uuid,
                    confidence=c.confidence,
                    geometry_json=c.geometry,
                    source_model=c.source_model,
                )
                session.add(row)
                if c.bbox_uuid:
                    by_bbox[c.bbox_uuid] = row
                written += 1

            await session.commit()
            return written

    def _preview_candidate_filters(self, run_uuid: str, q: PreviewCandidateQuery) -> List[Any]:
        clauses: List[Any] = [PreviewCandidate.run_uuid == run_uuid]
        if q.decisions:
            clauses.append(PreviewCandidate.decision.in_(q.decisions))
        if q.exclude_decisions:
            clauses.append(PreviewCandidate.decision.notin_(q.exclude_decisions))
        if q.proposed_label_uuids:
            clauses.append(PreviewCandidate.proposed_label_uuid.in_(q.proposed_label_uuids))
        if q.exclude_proposed_label_uuids:
            # NULL-safe: a candidate with no proposed label is not "excluded by" a label
            # it does not have, and a bare NOT IN would drop it — SQL's three-valued
            # logic makes `NULL NOT IN (...)` unknown, which filters as false.
            clauses.append(
                or_(
                    PreviewCandidate.proposed_label_uuid.is_(None),
                    PreviewCandidate.proposed_label_uuid.notin_(q.exclude_proposed_label_uuids),
                )
            )
        if q.confidence_min is not None:
            clauses.append(PreviewCandidate.confidence >= q.confidence_min)
        if q.confidence_max is not None:
            clauses.append(PreviewCandidate.confidence <= q.confidence_max)
        if q.is_new is not None:
            clauses.append(PreviewCandidate.is_new == q.is_new)
        if q.source_models:
            clauses.append(PreviewCandidate.source_model.in_(q.source_models))
        return clauses

    async def query_preview_candidates(
        self, run_uuid: str, q: PreviewCandidateQuery
    ) -> tuple[List[PreviewCandidate], int]:
        """Filtered, sorted, offset-paginated candidates + the filtered total.

        `image_confidence` (the default) reproduces the modal's client-side
        grouping — every candidate from one image adjacent, images ranked by
        their best candidate — which is exactly the part that did not survive
        being paginated. `label_confidence` is the same grouping keyed on the
        proposed label instead, so every candidate of one class is contiguous
        (and stays contiguous across page boundaries).
        """
        clauses = self._preview_candidate_filters(run_uuid, q)
        desc = q.sort_order == "desc"

        def _dir(col):
            return col.desc() if desc else col.asc()

        async with self._async_session_maker() as session:
            total = await session.exec(select(func.count()).select_from(PreviewCandidate).where(*clauses))
            total_count = total.one()

            stmt = select(PreviewCandidate).where(*clauses)
            group_col = {
                "image_confidence": PreviewCandidate.image_uuid,
                "label_confidence": PreviewCandidate.proposed_label_uuid,
            }.get(q.sort_by)
            if group_col is not None:
                # Rank groups by their best candidate *within the current filter*,
                # so filtering to one label doesn't leave images ordered by a
                # candidate the user filtered away.
                rank = (
                    select(
                        group_col.label("group_key"),
                        func.max(PreviewCandidate.confidence).label("best"),
                    )
                    .where(*clauses)
                    .group_by(group_col)
                    .subquery()
                )
                stmt = stmt.join(rank, rank.c.group_key == group_col).order_by(
                    _dir(rank.c.best),
                    group_col.asc(),
                    _dir(PreviewCandidate.confidence),
                    PreviewCandidate.uuid.asc(),
                )
            elif q.sort_by == "created_at":
                stmt = stmt.order_by(_dir(PreviewCandidate.created_at), PreviewCandidate.uuid.asc())
            else:
                stmt = stmt.order_by(_dir(PreviewCandidate.confidence), PreviewCandidate.uuid.asc())

            stmt = stmt.offset(max(0, q.offset)).limit(max(1, q.limit))
            result = await session.exec(stmt)
            return list(result.all()), total_count

    async def get_preview_candidates(
        self,
        run_uuid: str,
        decisions: Optional[Sequence[PreviewDecision]] = None,
        candidate_uuids: Optional[Sequence[str]] = None,
    ) -> List[PreviewCandidate]:
        """Unpaginated fetch — used by the apply path, which needs the whole
        accepted set, not a page of it."""
        clauses: List[Any] = [PreviewCandidate.run_uuid == run_uuid]
        if decisions:
            clauses.append(PreviewCandidate.decision.in_(list(decisions)))
        if candidate_uuids is not None:
            clauses.append(PreviewCandidate.uuid.in_(list(candidate_uuids)))
        async with self._async_session_maker() as session:
            result = await session.exec(select(PreviewCandidate).where(*clauses))
            return list(result.all())

    async def count_preview_decisions(self, run_uuids: Sequence[str]) -> Dict[str, Dict[str, int]]:
        """{run_uuid: {decision: count}} for the saved-runs list, in one query
        rather than one per run."""
        if not run_uuids:
            return {}
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(PreviewCandidate.run_uuid, PreviewCandidate.decision, func.count())
                .where(PreviewCandidate.run_uuid.in_(list(run_uuids)))
                .group_by(PreviewCandidate.run_uuid, PreviewCandidate.decision)
            )
            counts: Dict[str, Dict[str, int]] = {}
            for run_uuid, decision, n in result.all():
                value = decision.value if isinstance(decision, PreviewDecision) else str(decision)
                counts.setdefault(run_uuid, {})[value] = n
            return counts

    async def update_preview_candidates(
        self,
        run_uuid: str,
        candidate_uuids: Sequence[str],
        decision: Optional[PreviewDecision] = None,
        decision_action: Optional[PreviewDecisionAction] = None,
        proposed_label_uuid: Optional[str] = None,
        geometry: Optional[Dict[str, Any]] = None,
        applied_at: Optional[datetime] = None,
    ) -> int:
        """Bulk-set fields on a list of candidates. Every field is optional so the
        triage panel can PATCH only what changed on each debounced flush.

        Scoped by run_uuid as well as uuid so a stale panel can never write into
        a different run's candidates.
        """
        if not candidate_uuids:
            return 0
        values: Dict[str, Any] = {}
        if decision is not None:
            values["decision"] = decision
        if decision_action is not None:
            values["decision_action"] = decision_action
        if proposed_label_uuid is not None:
            values["proposed_label_uuid"] = proposed_label_uuid
        if geometry is not None:
            values["geometry_json"] = geometry
        if applied_at is not None:
            values["applied_at"] = applied_at
        if not values:
            return 0
        values["updated_at"] = datetime.now(timezone.utc)

        async with self._async_session_maker() as session:
            result = await session.exec(
                sa_update(PreviewCandidate)
                .where(
                    PreviewCandidate.run_uuid == run_uuid,
                    PreviewCandidate.uuid.in_(list(candidate_uuids)),
                )
                .values(**values)
            )
            await session.commit()
            return result.rowcount or 0

    async def get_bbox_label_uuids(self, bbox_uuids: Sequence[str]) -> Dict[str, Optional[str]]:
        """{bbox_uuid: label_uuid} for the bboxes that still exist.

        A missing key means the row is gone — which is exactly what preview
        staleness needs, so this deliberately returns only live rows rather than
        padding absent ones with None.
        """
        if not bbox_uuids:
            return {}
        async with self._async_session_maker() as session:
            result = await session.exec(select(BBox.uuid, BBox.label_uuid).where(BBox.uuid.in_(list(bbox_uuids))))
            return {uuid: label_uuid for uuid, label_uuid in result.all()}

    async def get_image_filenames(self, image_uuids: Sequence[str]) -> Dict[str, str]:
        """{image_uuid: filename} for the images that still exist.

        Absent keys mean the row is gone — the same "only live rows" contract as
        get_bbox_label_uuids, which is what preview staleness checks against.
        """
        if not image_uuids:
            return {}
        async with self._async_session_maker() as session:
            result = await session.exec(
                select(ImageRecord.uuid, ImageRecord.filename).where(ImageRecord.uuid.in_(list(image_uuids)))
            )
            return {uuid: filename for uuid, filename in result.all()}
