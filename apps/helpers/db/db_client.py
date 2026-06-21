import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, TypeVar, Union
from uuid import uuid4

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig

ALEMBIC_INI_PATH = Path(__file__).parent / "alembic.ini"

import jstyleson
from alembic.runtime.migration import MigrationContext
from pydantic import TypeAdapter
from sqlalchemy import Select, and_, delete, event, exists, func, or_, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import selectinload, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from apps.helpers.consts import TaskStatus

from .types import (
    FALSE_POS_TAG_UUID,
    MODEL_TYPE_ARCHITECTURES,
    REJECTION_TAG_UUID,
    UNVERIFIED_TAG_UUID,
    VERIFIED_TAG_UUID,
    WHOLE_DATABASE_DATASET_UUID,
    BBox,
    BBoxTagLink,
    Dataset,
    DatasetCreate,
    DatasetFilterCriteria,
    DatasetLabelLink,
    DatasetRead,
    DatasetUpdate,
    ImageItemStatus,
    ImageLabel,
    ImageLabelFilter,
    ImageLabelWithFilename,
    ImageRecord,
    ImageRecordRead,
    ImageRecordUpdate,
    ImageSearchParams,
    ImageTagFilter,
    ImageTagLink,
    Label,
    LabelScope,
    LabelType,
    ModelBundle,
    ModelBundleCreate,
    ModelBundleRead,
    ModelBundleUpdate,
    ModelConfig,
    ModelConfigCreate,
    ModelConfigNode,
    ModelConfigNodeRead,
    ModelConfigRead,
    ModelConfigUpdate,
    Source,
    SourceRead,
    Tag,
    TagKind,
    TagPresenceFilter,
    TaskConfig,
    TaskInstance,
    TrainingConfig,
    TrainingConfigCreate,
    TrainingConfigDatasetLink,
    TrainingConfigRead,
    TrainingConfigUpdate,
    UserSetting,
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
        uuid=FALSE_POS_TAG_UUID,
        name="False Positive",
        color="#c0392b",
        kind=TagKind.SYSTEM,
        exclusive_group="verification",
        protected=True,
        icon="✗",
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
]

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


T = TypeVar("T", bound=SQLModel)


LABELS_JSON = "labels.json"
TAGS_JSON = "tags.json"
SOURCES_JSON = "sources.json"
DATASETS_JSON = "datasets.json"
TASK_CONFIGS_JSON = "task_configs.json"
DB_FILE = "db.sqlite"


def _escape_like_pattern(pattern: str) -> str:
    """Escape SQL LIKE metacharacters so user input is treated as literal text."""
    return pattern.replace("!", "!!").replace("%", "!%").replace("_", "!_")


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


def _apply_image_search_filters(stmt: Select, params: ImageSearchParams) -> Select:
    """Apply all ImageSearchParams filter predicates to *stmt* and return it.

    Does not add ORDER BY, LIMIT, or keyset cursors — those are caller concerns.
    Works for any Select target (ImageRecord, ImageRecord.uuid, etc.).

    Performance note: bbox-related filters (tag, label, require_bboxes,
    label_exclusions) are expressed as non-correlated IN / NOT IN subqueries
    rather than correlated EXISTS.  SQLite materialises the inner SELECT once
    as a hash table and then the outer scan does O(1) membership probes, which
    is dramatically faster than re-evaluating a correlated EXISTS for every row
    in a large images table.
    """
    # Track whether any filter that already guarantees at least one bbox has
    # been applied, so we can skip the redundant require_bboxes subquery.
    _bbox_implied = False

    # ── Tag filter ────────────────────────────────────────────────────────────
    # When show_all_bboxes is True or tag_uuids is empty, skip so all images
    # are candidates regardless of bbox tags.
    if not params.show_all_bboxes and params.tag_uuids:
        if params.tag_match_mode == "all":
            # Image must have at least one bbox carrying EACH selected tag
            # (the bboxes can be different).
            for tag_uuid in params.tag_uuids:
                image_uuids = (
                    select(BBox.image_uuid)
                    .join(BBoxTagLink, BBoxTagLink.bbox_uuid == BBox.uuid)
                    .where(BBoxTagLink.tag_uuid == tag_uuid)
                    .distinct()
                )
                stmt = stmt.where(ImageRecord.uuid.in_(image_uuids))
        else:
            # "any of" — image has a bbox tagged with ANY of the selected tags.
            image_uuids = (
                select(BBox.image_uuid)
                .join(BBoxTagLink, BBoxTagLink.bbox_uuid == BBox.uuid)
                .where(BBoxTagLink.tag_uuid.in_(params.tag_uuids))
                .distinct()
            )
            stmt = stmt.where(ImageRecord.uuid.in_(image_uuids))
        _bbox_implied = True

    # ── Single-label backward-compat filter ───────────────────────────────────
    # Kept for API callers that still send label_uuid.  New callers should use
    # label_uuids instead.
    if params.label_uuid and params.label_uuid not in params.label_uuids:
        image_uuids = select(BBox.image_uuid).where(BBox.label_uuid == params.label_uuid).distinct()
        stmt = stmt.where(ImageRecord.uuid.in_(image_uuids))
        _bbox_implied = True

    # ── Filename search ────────────────────────────────────────────────────────
    if params.filename_search:
        escaped = _escape_like_pattern(params.filename_search)
        stmt = stmt.where(ImageRecord.filename.like(f"%{escaped}%", escape="!"))
    for pattern in params.filename_searches:
        escaped = _escape_like_pattern(pattern)
        stmt = stmt.where(ImageRecord.filename.like(f"%{escaped}%", escape="!"))

    # ── Date range on created_at ───────────────────────────────────────────────
    if params.date_from:
        stmt = stmt.where(ImageRecord.created_at >= params.date_from)
    if params.date_to:
        stmt = stmt.where(ImageRecord.created_at <= params.date_to)

    # ── Date range on updated_at ───────────────────────────────────────────────
    if params.updated_date_from:
        stmt = stmt.where(ImageRecord.updated_at >= params.updated_date_from)
    if params.updated_date_to:
        stmt = stmt.where(ImageRecord.updated_at <= params.updated_date_to)

    # ── Tag presence/absence filters ──────────────────────────────────────────
    for tf in params.tag_presence_filters:
        image_uuids = (
            select(BBox.image_uuid)
            .join(BBoxTagLink, BBoxTagLink.bbox_uuid == BBox.uuid)
            .where(BBoxTagLink.tag_uuid == tf.tag_uuid)
            .distinct()
        )
        if tf.exclude:
            stmt = stmt.where(ImageRecord.uuid.not_in(image_uuids))
        else:
            stmt = stmt.where(ImageRecord.uuid.in_(image_uuids))
            _bbox_implied = True

    # ── Image-label filters (EXISTS / NOT EXISTS subqueries) ──────────────────
    # These filter on the image_labels table (not bboxes), which is small and
    # accessed by PK — correlated EXISTS is fine here.
    for f in params.image_label_filters:
        if f.require_verified:
            # require EXISTS row with PRESENT or ABSENT (not uncertain)
            subq = (
                select(ImageLabel)
                .where(
                    ImageLabel.image_uuid == ImageRecord.uuid,
                    ImageLabel.label_uuid == f.label_uuid,
                    ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                )
                .exists()
            )
            stmt = stmt.where(subq)
        else:
            subq = (
                select(ImageLabel)
                .where(
                    ImageLabel.image_uuid == ImageRecord.uuid,
                    ImageLabel.label_uuid == f.label_uuid,
                    ImageLabel.status == f.status,
                )
                .exists()
            )
            stmt = stmt.where(~subq if f.exclude else subq)

    # ── Whole-image tag filters (EXISTS / NOT EXISTS subqueries) ──────────────
    # These filter on the image_tags table (not bboxes), which is small and
    # accessed by PK — correlated EXISTS is fine here.
    for tf in params.image_tag_filters:
        if tf.require_unset:
            # NOT EXISTS any row for this image+tag (tag was never assessed)
            subq = (
                select(ImageTagLink)
                .where(
                    ImageTagLink.image_uuid == ImageRecord.uuid,
                    ImageTagLink.tag_uuid == tf.tag_uuid,
                )
                .exists()
            )
            stmt = stmt.where(~subq)
        elif tf.require_verified:
            # require EXISTS row with PRESENT or ABSENT (not uncertain)
            subq = (
                select(ImageTagLink)
                .where(
                    ImageTagLink.image_uuid == ImageRecord.uuid,
                    ImageTagLink.tag_uuid == tf.tag_uuid,
                    ImageTagLink.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                )
                .exists()
            )
            stmt = stmt.where(subq)
        else:
            subq = (
                select(ImageTagLink)
                .where(
                    ImageTagLink.image_uuid == ImageRecord.uuid,
                    ImageTagLink.tag_uuid == tf.tag_uuid,
                    ImageTagLink.status == tf.status,
                )
                .exists()
            )
            stmt = stmt.where(~subq if tf.exclude else subq)

    # ── Dataset / review label filter ─────────────────────────────────────────
    if params.label_uuids:
        if params.label_match_mode == "all":
            # Image must have at least one bbox with EACH selected label (or any
            # of its descendants when label_include_children is True).
            for i, label_uuid in enumerate(params.label_uuids):
                if params.label_include_children:
                    label_set = _label_uuids_with_descendants([label_uuid], cte_name=f"label_desc_{i}")
                    condition = BBox.label_uuid.in_(label_set)
                else:
                    condition = BBox.label_uuid == label_uuid
                image_uuids = select(BBox.image_uuid).where(condition).distinct()
                stmt = stmt.where(ImageRecord.uuid.in_(image_uuids))
        else:
            # "any of" — image has a bbox with ANY of the selected labels (or
            # their descendants when label_include_children is True).
            if params.label_include_children:
                label_condition = BBox.label_uuid.in_(_label_uuids_with_descendants(params.label_uuids))
            else:
                label_condition = BBox.label_uuid.in_(params.label_uuids)
            image_uuids = select(BBox.image_uuid).where(label_condition).distinct()
            stmt = stmt.where(ImageRecord.uuid.in_(image_uuids))
        _bbox_implied = True

    # ── Verification check ────────────────────────────────────────────────────
    # Uses verification_label_uuids when set, falling back to label_uuids.
    # This lets callers decouple the pool filter from the verification target.
    _verification_targets = params.verification_label_uuids or params.label_uuids
    if params.label_verification_check and _verification_targets:
        if params.label_verification_check == "all_verified":
            for lu in _verification_targets:
                subq = (
                    select(ImageLabel)
                    .where(
                        ImageLabel.image_uuid == ImageRecord.uuid,
                        ImageLabel.label_uuid == lu,
                        ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                    )
                    .exists()
                )
                stmt = stmt.where(subq)
        elif params.label_verification_check == "has_unverified":
            # At least one label must NOT have a verified row
            not_verified = [
                ~(
                    select(ImageLabel)
                    .where(
                        ImageLabel.image_uuid == ImageRecord.uuid,
                        ImageLabel.label_uuid == lu,
                        ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                    )
                    .exists()
                )
                for lu in _verification_targets
            ]
            stmt = stmt.where(or_(*not_verified))

    # ── Require at least one bbox ─────────────────────────────────────────────
    # Skip when any earlier filter already guarantees bbox existence.
    if params.require_bboxes and not _bbox_implied:
        image_uuids = select(BBox.image_uuid).distinct()
        stmt = stmt.where(ImageRecord.uuid.in_(image_uuids))

    # ── Label exclusions (NOT IN per label UUID) ───────────────────────────────
    for label_uuid in params.label_exclusions:
        image_uuids = select(BBox.image_uuid).where(BBox.label_uuid == label_uuid).distinct()
        stmt = stmt.where(ImageRecord.uuid.not_in(image_uuids))

    return stmt


# Eager-load options shared by all training-config queries.
_TC_LOAD_OPTS = (selectinload(TrainingConfig.dataset_links),)

# Page size for image pagination in readiness checks.
_READINESS_PAGE_SIZE = 1000


class DbClient:
    """Helper class for SQLModel interactions for image annotations."""

    def __init__(self, db_dir: Union[str, Path], *, memory: bool = False):
        self._db_dir = Path(db_dir)
        self._db_path: Path = self._db_dir / DB_FILE
        self._labels_json_path: Path = self._db_dir / LABELS_JSON
        self._tags_json_path: Path = self._db_dir / TAGS_JSON
        self._sources_json_path: Path = self._db_dir / SOURCES_JSON
        self._datasets_json_path: Path = self._db_dir / DATASETS_JSON
        self._task_configs_json_path: Path = self._db_dir / TASK_CONFIGS_JSON
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
            cursor.close()

        self._async_session_maker = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)

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
                    )
                )
                await session.commit()

    async def get_dataset_image_uuids(self, dataset_uuids: list[str]) -> list[str]:
        """Return deduplicated image UUIDs from the union of the given datasets."""
        seen: set[str] = set()
        result: list[str] = []
        for ds_uuid in dataset_uuids:
            dataset = await self.get_dataset(ds_uuid)
            if dataset is None:
                continue
            after_uuid: Optional[str] = None
            after_sort_value = None
            while True:
                page, has_more = await self._list_images_for_dataset(
                    dataset,
                    limit=_READINESS_PAGE_SIZE,
                    sort_by="uuid",
                    sort_order="asc",
                    after_uuid=after_uuid,
                    after_sort_value=after_sort_value,
                )
                for img in page:
                    if img.uuid not in seen:
                        seen.add(img.uuid)
                        result.append(img.uuid)
                if not has_more or not page:
                    break
                after_uuid = page[-1].uuid
                after_sort_value = page[-1].uuid
        return result

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
        json_bboxes: List[BBox] = []
        # Per-bbox supplementary data extracted before constructing BBox objects.
        box_tag_uuids: Dict[str, List[str]] = {}  # bbox_uuid → tag UUID list
        box_tag_names: Dict[str, List[str]] = {}  # bbox_uuid → tag name list (fallback)
        box_label_names: Dict[str, str] = {}  # bbox_uuid → label name (fallback)

        if json_path.exists():
            try:
                metadata_dict = jstyleson.loads(json_path.read_text())
                json_uuid = metadata_dict.get("uuid")
                json_phash = metadata_dict.get("phash")
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
            except Exception as e:
                logger.error(f"Failed to parse {json_path}: {e}")
                # Reset to empty — partial state from a mid-loop failure is worse than no state.
                json_bboxes = []
                box_tag_uuids = {}
                box_tag_names = {}
                box_label_names = {}

        async with self._async_session_maker() as session:
            # --- Resolve label/tag UUIDs (UUID-first, name-fallback) ---
            all_label_uuids = {b.label_uuid for b in json_bboxes if b.label_uuid}
            all_tag_uuids = {u for uuids in box_tag_uuids.values() for u in uuids}
            all_label_names = set(box_label_names.values())
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
                )
                session.add(db_record)
                bboxes_needing_links = [b.uuid for b in json_bboxes if box_tag_uuids.get(b.uuid)]
            elif overwrite:
                # Old BBoxTagLink rows are cascade-deleted when the old BBox rows are removed.
                db_record.bboxes = json_bboxes
                db_record.metadata_json = metadata_dict
                if json_phash:
                    db_record.phash = json_phash
                session.add(db_record)
                bboxes_needing_links = [b.uuid for b in json_bboxes if box_tag_uuids.get(b.uuid)]
            else:
                existing_uuids = {b.uuid for b in db_record.bboxes}
                new_bboxes = [b for b in json_bboxes if b.uuid not in existing_uuids]
                db_record.bboxes = db_record.bboxes + new_bboxes
                if not db_record.metadata_json:
                    db_record.metadata_json = metadata_dict
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

            try:
                await session.commit()
            except Exception as e:
                logger.exception(e)
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

    async def add_image(self, filename: Path, metadata_json: Optional[Dict[str, Any]] = None) -> ImageRecord:
        """
        Add a new image record or return existing object if filename exists.

        Returns:
            Image: The fully populated Image SQLModel instance.
        """
        async with self._async_session_maker() as session:
            # Check for existing image first (fast path)
            statement = select(ImageRecord).where(ImageRecord.filename == str(filename))
            result = await session.exec(statement)
            existing_image = result.one_or_none()

            if existing_image:
                return existing_image

            # Create new record; catch the race where two concurrent callers
            # both pass the check above before either INSERT commits.
            new_image = ImageRecord(
                uuid=str(uuid4()),
                filename=str(filename),
                metadata_json=metadata_json or {},
            )
            session.add(new_image)
            try:
                await session.commit()
                await session.refresh(new_image)
                return new_image
            except IntegrityError:
                await session.rollback()
                result = await session.exec(statement)
                return result.one()

    async def list_images(self, limit: int = 100, offset: int = 0) -> List[ImageRecord]:
        """List images with basic pagination."""
        async with self._async_session_maker() as session:
            statement = select(ImageRecord).offset(offset).limit(limit)
            result = await session.exec(statement)
            return result.all()

    async def list_images_search(self, params: ImageSearchParams) -> List[ImageRecord]:
        """Flexible image search used by the review queue.

        Supports tag filtering (OR across selected tags), label filtering, filename
        substring search, date range, image-label status filters, and stable keyset
        pagination for both UUID and datetime sort orders.
        """
        async with self._async_session_maker() as session:
            stmt = select(ImageRecord).options(
                selectinload(ImageRecord.bboxes).options(
                    selectinload(BBox.label),
                    selectinload(BBox.tags),
                ),
                selectinload(ImageRecord.tags),
            )

            stmt = _apply_image_search_filters(stmt, params)

            # ── Sort column ────────────────────────────────────────────────────
            sort_col_map = {
                "created_at": ImageRecord.created_at,
                "updated_at": ImageRecord.updated_at,
                "uuid": ImageRecord.uuid,
                "filename": ImageRecord.filename,
            }
            sort_col = sort_col_map[params.sort_by]

            # ── Keyset pagination ──────────────────────────────────────────────
            # For UUID sort, a simple scalar comparison is sufficient (UUIDs are unique).
            # For datetime sorts, use a composite (sort_field, uuid) keyset to handle
            # ties deterministically without gaps or duplicates.
            if params.after_uuid is not None:
                if params.sort_by == "uuid" or params.after_sort_value is None:
                    if params.sort_order == "asc":
                        stmt = stmt.where(ImageRecord.uuid > params.after_uuid)
                    else:
                        stmt = stmt.where(ImageRecord.uuid < params.after_uuid)
                else:
                    sv = params.after_sort_value
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
        """
        async with self._async_session_maker() as session:
            inner = _apply_image_search_filters(select(ImageRecord.uuid), params)
            stmt = select(func.count()).select_from(inner.subquery())
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

    async def update_single_bbox(self, bbox_uuid: str, label_uuid: Optional[str], tag_uuids: List[str]) -> None:
        """Update one bbox's label and replace its tag links in place.

        Exclusive-group constraints are enforced: if multiple tags in tag_uuids share
        an exclusive_group, only the *last* one in the list is kept.  Callers can
        therefore pass [*existing_tags, *new_tags] and the new tags will win any
        exclusive-group conflict.
        """
        async with self._async_session_maker() as session:
            bbox = await session.get(BBox, bbox_uuid)
            if bbox is None:
                return
            if label_uuid is not None:
                bbox.label_uuid = label_uuid

            # Resolve exclusive-group conflicts before writing.
            if tag_uuids:
                tag_rows = await session.exec(select(Tag).where(Tag.uuid.in_(tag_uuids)))
                tags_by_uuid = {t.uuid: t for t in tag_rows.all()}
            else:
                tags_by_uuid = {}
            final_uuids = self._resolve_exclusive_groups(tag_uuids, tags_by_uuid)

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
            bbox_result = await session.exec(
                select(BBox.uuid, BBox.phash, BBox.label_uuid).where(BBox.image_uuid.in_(image_uuids))
            )
            bbox_rows = bbox_result.all()

        return image_rows, bbox_rows

    async def delete_images_bulk(self, uuids: List[str]) -> int:
        """Delete image records in bulk.  BBoxes cascade-delete automatically.

        Returns the number of image records deleted.
        """
        if not uuids:
            return 0
        async with self._async_session_maker() as session:
            result = await session.exec(delete(ImageRecord).where(ImageRecord.uuid.in_(uuids)))
            await session.commit()
            return result.rowcount or 0

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

    async def write_image_metadata_to_db(self, image_uuid: str, update: ImageRecordUpdate) -> None:
        """Replace all bboxes for an image directly in the database."""
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
                session.add(
                    BBox(
                        uuid=b.uuid,
                        image_uuid=image_uuid,
                        label_uuid=b.label_uuid,
                        x=b.x,
                        y=b.y,
                        width=b.width,
                        height=b.height,
                        metadata_json=b.metadata_json or {},
                    )
                )
                for tag_uuid in self._resolve_exclusive_groups(b.tag_uuids, tags_by_uuid):
                    session.add(BBoxTagLink(bbox_uuid=b.uuid, tag_uuid=tag_uuid))
            # Touch the image so updated_at reflects the full bbox replacement.
            await self._touch_image(session, image_uuid)
            await session.commit()

    def write_image_metadata_to_json(
        self,
        json_path: Path,
        image_uuid: str,
        filename: str,
        update: ImageRecordUpdate,
        image_phash: Optional[str] = None,
    ) -> None:
        """Write image metadata to the JSON sidecar file."""
        existing: Dict[str, Any] = {}
        if json_path.exists():
            try:
                existing = jstyleson.loads(json_path.read_text())
            except Exception as e:
                logger.warning(f"Could not parse sidecar {json_path}, overwriting: {e}")
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
        existing.update(new_data)
        json_path.write_text(json.dumps(existing, indent=2))

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
    ) -> Label:
        """Insert a label and return the created object."""
        new_label = Label(
            uuid=uuid or str(uuid4()),
            name=name,
            color=color,
            parent_uuid=parent_uuid,
            label_type=label_type,
            scope=scope,
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
    # Image Labels
    ################################################################################

    async def set_image_label_status(self, image_uuid: str, label_uuid: str, status: ImageItemStatus) -> None:
        """Upsert an image_label record, overwriting any existing status."""
        async with self._async_session_maker() as session:
            existing = await session.get(ImageLabel, (image_uuid, label_uuid))
            if existing:
                existing.status = status
                existing.updated_at = datetime.now(timezone.utc)
            else:
                session.add(ImageLabel(image_uuid=image_uuid, label_uuid=label_uuid, status=status))
            await session.commit()

    async def get_image_labels(self, image_uuid: str) -> List[ImageLabel]:
        """Return all image_label rows for a given image."""
        async with self._async_session_maker() as session:
            result = await session.exec(select(ImageLabel).where(ImageLabel.image_uuid == image_uuid))
            return result.all()

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
            img_records: List[ImageRecord] = (
                await session.exec(select(ImageRecord).where(ImageRecord.uuid.in_(image_uuids)))
            ).all()
            uuid_to_filename = {img.uuid: img.filename for img in img_records}

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
    ) -> TaskConfig:
        config = TaskConfig(name=name, typename=typename, params_json=params, description=description)
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
    ) -> List[TaskInstance]:
        stmt = select(TaskInstance).options(selectinload(TaskInstance.config))
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
    ) -> None:
        async with self._async_session_maker() as session:
            instance = await session.get(TaskInstance, task_uuid)
            if instance:
                if started_at is not None:
                    instance.started_at = started_at
                if ended_at is not None:
                    instance.ended_at = ended_at
                if log_json is not None:
                    instance.log_json = log_json
                session.add(instance)
                await session.commit()

    async def update_task_heartbeat(self, task_uuid: str) -> None:
        async with self._async_session_maker() as session:
            instance = await session.get(TaskInstance, task_uuid)
            if instance:
                instance.heartbeat_at = datetime.now(timezone.utc)
                session.add(instance)
                await session.commit()

    async def set_task_status(self, task_uuid: str, new_status: str) -> None:
        async with self._async_session_maker() as session:
            instance = await session.get(TaskInstance, task_uuid)
            if instance:
                instance.status = new_status
                session.add(instance)
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

            # Delete any configs that were marked_for_delete and now have no remaining instances
            if affected_config_uuids:
                still_referenced = select(TaskInstance.config_uuid).where(
                    TaskInstance.config_uuid.in_(affected_config_uuids)
                )
                await session.exec(
                    delete(TaskConfig).where(
                        TaskConfig.uuid.in_(affected_config_uuids),
                        TaskConfig.marked_for_delete == True,  # type: ignore[arg-type]  # noqa: E712
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
            created_at=d.created_at,
            updated_at=d.updated_at,
        )

    async def list_datasets(self) -> List[DatasetRead]:
        async with self._async_session_maker() as session:
            stmt = select(Dataset).options(selectinload(Dataset.label_links))
            result = await session.exec(stmt)
            return [self._dataset_to_read(d) for d in result.all()]

    async def _sync_datasets_to_json(self) -> None:
        """Dump the current datasets table to datasets.json."""
        datasets = await self.list_datasets()
        rows = [
            {
                "uuid": d.uuid,
                "name": d.name,
                "description": d.description,
                "filter_json": d.filter_json.model_dump(mode="json"),
                "label_uuids": d.label_uuids,
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
        """Expand a DatasetRead's flat filter criteria into an ImageSearchParams."""
        f = dataset.filter_json
        tag_presence_filters: List[TagPresenceFilter] = []
        # Tag excludes go through tag_presence_filters (work regardless of show_all_bboxes)
        for u in f.tag_exclude_uuids:
            tag_presence_filters.append(TagPresenceFilter(tag_uuid=u, exclude=True))

        # show_all_bboxes=True means the tag_uuids include-filter is skipped.
        # Mirror the review panel: show_all_bboxes=True only when no tag includes.
        show_all_bboxes = len(f.tag_uuids) == 0

        image_tag_filters: List[ImageTagFilter] = []
        for cond in f.image_tag_conditions:
            if cond.state == "present":
                image_tag_filters.append(ImageTagFilter(tag_uuid=cond.tag_uuid, status=ImageItemStatus.PRESENT))
            elif cond.state == "absent":
                image_tag_filters.append(ImageTagFilter(tag_uuid=cond.tag_uuid, status=ImageItemStatus.ABSENT))
            elif cond.state == "unset":
                image_tag_filters.append(ImageTagFilter(tag_uuid=cond.tag_uuid, require_unset=True))

        image_label_filters: List[ImageLabelFilter] = []
        for cond in f.image_label_conditions:
            image_label_filters.append(ImageLabelFilter(label_uuid=cond.label_uuid, status=ImageItemStatus(cond.state)))

        return ImageSearchParams(
            show_all_bboxes=show_all_bboxes,
            tag_uuids=f.tag_uuids,
            tag_match_mode=f.tag_match_mode,
            tag_presence_filters=tag_presence_filters,
            label_uuids=dataset.label_uuids,
            label_match_mode=f.label_match_mode,
            label_include_children=f.label_include_children,
            label_exclusions=f.label_exclude_uuids,
            date_from=f.date_from,
            date_to=f.date_to,
            filename_search=f.filename_search or None,
            label_verification_check=f.verification_check,
            image_tag_filters=image_tag_filters,
            image_label_filters=image_label_filters,
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

                if not rollup_nodes:
                    return base_label_uuids, {}

                # Load full label hierarchy to find rollup children.
                all_labels_result = await session.exec(select(Label))
                all_labels = {lbl.uuid: lbl for lbl in all_labels_result.all()}

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

            # Assign each rollup descendant to its NEAREST rollup ancestor by processing
            # rollup parents deepest-first: a label shared between nested rollup parents
            # (e.g. A→B→C with both A and B roll_up_unchecked) is claimed by the closest
            # one (B) and never double-counted. The source query order is unspecified, so
            # depth — not row order — drives the assignment.
            rollup_map: dict[str, list[str]] = {}
            claimed_as_rollup: set[str] = set()
            for parent_uuid in sorted(rollup_label_set, key=_label_depth, reverse=True):
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

    async def get_training_config_readiness(self, uuid: str) -> Optional[dict]:
        """Return ready/needs-review counts for a training config's datasets (union).

        An image is 'ready' when every target label has an explicit PRESENT or ABSENT
        row in image_labels; 'needs_review' otherwise.

        When rollup is active (any node has roll_up_unchecked=True) an image is
        considered ready as soon as ANY ONE of the in-scope labels is verified,
        because child annotations all roll up to their parent label.

        Returns None if the training config doesn't exist.
        """
        tc = await self.get_training_config(uuid)
        if tc is None:
            return None
        dataset_uuids = tc.dataset_uuids

        label_uuids, rollup_map = await self._derive_labels_for_node(tc.model_config_node_uuid)
        is_rollup = bool(rollup_map)

        if not label_uuids:
            return {
                "ready_count": 0,
                "needs_review_count": 0,
                "dataset_uuids": dataset_uuids,
                "label_uuids": [],
                "roll_up_active": False,
            }

        # Collect all images across all linked datasets (union, de-duped by uuid).
        seen_uuids: set[str] = set()
        all_images: list[ImageRecord] = []
        for ds_uuid in dataset_uuids:
            dataset = await self.get_dataset(ds_uuid)
            if dataset is None:
                continue
            after_uuid: Optional[str] = None
            after_sort_value: Optional[datetime] = None
            while True:
                page, has_more = await self._list_images_for_dataset(
                    dataset,
                    limit=_READINESS_PAGE_SIZE,
                    sort_by="uuid",
                    sort_order="asc",
                    after_uuid=after_uuid,
                    after_sort_value=after_sort_value,
                )
                for img in page:
                    if img.uuid not in seen_uuids:
                        seen_uuids.add(img.uuid)
                        all_images.append(img)
                if not has_more or not page:
                    break
                last = page[-1]
                after_uuid = last.uuid
                # after_sort_value stays None when sort_by="uuid"; pagination uses after_uuid only

        if not all_images:
            return {
                "ready_count": 0,
                "needs_review_count": 0,
                "dataset_uuids": dataset_uuids,
                "label_uuids": label_uuids,
                "roll_up_active": is_rollup,
            }

        image_uuids = [img.uuid for img in all_images]
        n_labels = len(label_uuids)
        # With rollup, any one verified label means the image is ready.
        readiness_threshold = 1 if is_rollup else n_labels

        async with self._async_session_maker() as session:
            verified_count_col = func.count(ImageLabel.label_uuid).label("verified_count")
            result = await session.exec(
                select(ImageLabel.image_uuid, verified_count_col)
                .where(
                    ImageLabel.image_uuid.in_(image_uuids),
                    ImageLabel.label_uuid.in_(label_uuids),
                    ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                )
                .group_by(ImageLabel.image_uuid)
            )
            rows = result.all()
            verified_by_image = {row[0]: row[1] for row in rows}

            ready_count = 0
            needs_review_count = 0
            for img_uuid in image_uuids:
                if verified_by_image.get(img_uuid, 0) >= readiness_threshold:
                    ready_count += 1
                else:
                    needs_review_count += 1

        return {
            "ready_count": ready_count,
            "needs_review_count": needs_review_count,
            "dataset_uuids": dataset_uuids,
            "label_uuids": label_uuids,
            "roll_up_active": is_rollup,
        }

    async def get_training_config_images_by_label_status(
        self,
        uuid: str,
        label_uuid: str,
        status: Optional[str],
        *,
        rollup_label_uuids: Optional[List[str]] = None,
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
        """
        tc = await self.get_training_config(uuid)
        if tc is None:
            return [], False

        # Collect all images from the training config's datasets (union, de-duped).
        seen_uuids: set[str] = set()
        all_images: list[ImageRecord] = []

        for ds_uuid in tc.dataset_uuids:
            dataset = await self.get_dataset(ds_uuid)
            if dataset is None:
                continue
            ds_after_uuid: Optional[str] = None
            while True:
                page, has_more = await self._list_images_for_dataset(
                    dataset,
                    limit=_READINESS_PAGE_SIZE,
                    sort_by="uuid",
                    sort_order="asc",
                    after_uuid=ds_after_uuid,
                )
                for img in page:
                    if img.uuid not in seen_uuids:
                        seen_uuids.add(img.uuid)
                        all_images.append(img)
                if not has_more or not page:
                    break
                ds_after_uuid = page[-1].uuid

        if not all_images:
            return [], False

        all_image_uuids = [img.uuid for img in all_images]

        # Effective label UUIDs: when rollup children are provided, query across
        # the parent + all children so the results match the rolled-up stat counts.
        effective_label_uuids = [label_uuid] + (rollup_label_uuids or [])

        # Filter by status.
        if status and status != "all":
            async with self._async_session_maker() as session:
                if status == "uncertain":
                    # Images with no PRESENT/ABSENT row for any of the effective labels.
                    result = await session.exec(
                        select(ImageLabel.image_uuid).where(
                            ImageLabel.image_uuid.in_(all_image_uuids),
                            ImageLabel.label_uuid.in_(effective_label_uuids),
                            ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                        )
                    )
                    verified = set(result.all())
                    matching = set(all_image_uuids) - verified
                else:
                    target = ImageItemStatus.PRESENT if status == "present" else ImageItemStatus.ABSENT
                    result = await session.exec(
                        select(ImageLabel.image_uuid).where(
                            ImageLabel.image_uuid.in_(all_image_uuids),
                            ImageLabel.label_uuid.in_(effective_label_uuids),
                            ImageLabel.status == target,
                        )
                    )
                    matching = set(result.all())
            all_images = [img for img in all_images if img.uuid in matching]

        if not all_images:
            return [], False

        def _sort_key(img: "ImageRecord"):
            return (getattr(img, sort_by), img.uuid)

        all_images.sort(key=_sort_key, reverse=(sort_order == "desc"))

        # Apply keyset cursor.
        if after_uuid is not None:
            start_idx = None
            for i, img in enumerate(all_images):
                attr = getattr(img, sort_by)
                if sort_by == "uuid" or after_sort_value is None:
                    past = img.uuid > after_uuid if sort_order == "asc" else img.uuid < after_uuid
                else:
                    sv = after_sort_value
                    if sort_order == "asc":
                        past = attr > sv or (attr == sv and img.uuid > after_uuid)
                    else:
                        past = attr < sv or (attr == sv and img.uuid < after_uuid)
                if past:
                    start_idx = i
                    break
            if start_idx is None:
                return [], False
            all_images = all_images[start_idx:]

        has_more = len(all_images) > limit
        return all_images[:limit], has_more

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
    ) -> tuple[list["ImageRecord"], bool]:
        """Return (images, has_more) for images that need verification in a training config.

        Images are those within the config's linked dataset union that have at least one
        target label (derived from the model config node) without a verified image_label row.

        extra_label_uuids — when provided, additionally narrows the pool to images that
        also have bboxes with these labels (applied on top of each dataset's own filters).

        v1: in-memory merge across datasets. Push to SQL union if datasets grow large.
        """
        tc = await self.get_training_config(uuid)
        if tc is None:
            return [], False

        label_uuids, _ = await self._derive_labels_for_node(tc.model_config_node_uuid)
        if not label_uuids:
            return [], False

        # Collect all needs-review images from each linked dataset (union, de-duped by UUID).
        # Each dataset is iterated in pages (sort_by="uuid" for stable pagination).
        seen_uuids: set[str] = set()
        all_images: list[ImageRecord] = []

        for ds_uuid in tc.dataset_uuids:
            dataset = await self.get_dataset(ds_uuid)
            if dataset is None:
                continue

            ds_after_uuid: Optional[str] = None
            while True:
                params = self.build_dataset_search_params(
                    dataset,
                    limit=_READINESS_PAGE_SIZE,
                    sort_by="uuid",
                    sort_order="asc",
                    after_uuid=ds_after_uuid,
                    after_sort_value=None,
                )
                # Decouple verification targets from the pool label filter.
                params.verification_label_uuids = label_uuids
                params.label_verification_check = "has_unverified"
                if extra_label_uuids:
                    # Narrow pool: image must also have bboxes with the extra labels.
                    params.label_uuids = list(set(params.label_uuids) | set(extra_label_uuids))

                page = await self.list_images_search(params)
                for img in page:
                    if img.uuid not in seen_uuids:
                        seen_uuids.add(img.uuid)
                        all_images.append(img)

                if len(page) < _READINESS_PAGE_SIZE:
                    break
                ds_after_uuid = page[-1].uuid

        if not all_images:
            return [], False

        # Sort the merged list by (sort_by, uuid) for deterministic ordering.
        def _sort_key(img: "ImageRecord"):
            return (getattr(img, sort_by), img.uuid)

        all_images.sort(key=_sort_key, reverse=(sort_order == "desc"))

        # Apply keyset cursor to resume from the previous page.
        start_idx = 0
        if after_uuid is not None:
            for i, img in enumerate(all_images):
                attr = getattr(img, sort_by)
                if sort_by == "uuid" or after_sort_value is None:
                    past = img.uuid > after_uuid if sort_order == "asc" else img.uuid < after_uuid
                else:
                    sv = after_sort_value
                    if sort_order == "asc":
                        past = attr > sv or (attr == sv and img.uuid > after_uuid)
                    else:
                        past = attr < sv or (attr == sv and img.uuid < after_uuid)
                if past:
                    start_idx = i
                    break
            else:
                return [], False
            all_images = all_images[start_idx:]

        has_more = len(all_images) > limit
        return all_images[:limit], has_more

    async def get_training_config_label_stats(self, uuid: str) -> Optional[dict]:
        """Return per-label dataset stats for a training config.

        Relevant labels are derived from the model config node referenced by
        model_config_node_uuid. If no node is assigned, falls back to using
        all labels across all linked datasets.

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
        seen_uuids: set[str] = set()
        all_image_uuids: list[str] = []
        for dataset in datasets_by_uuid.values():
            after_uuid: Optional[str] = None
            after_sort_value: Optional[datetime] = None
            while True:
                page, has_more = await self._list_images_for_dataset(
                    dataset,
                    limit=_READINESS_PAGE_SIZE,
                    sort_by="uuid",
                    sort_order="asc",
                    after_uuid=after_uuid,
                    after_sort_value=after_sort_value,
                )
                for img in page:
                    if img.uuid not in seen_uuids:
                        seen_uuids.add(img.uuid)
                        all_image_uuids.append(img.uuid)
                if not has_more or not page:
                    break
                last = page[-1]
                after_uuid = last.uuid
                # after_sort_value stays None when sort_by="uuid"; pagination uses after_uuid only

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
                "fully_verified_image_count": 0,
                "images_missing_verification_count": 0,
            }

        if not all_image_uuids:
            return {"label_stats": list(label_stats.values()), "total_images": 0}

        async with self._async_session_maker() as session:
            # ── Image-scope labels ──────────────────────────────────────────
            if image_label_uuids:
                # Count verified (PRESENT/ABSENT) rows per label.
                result = await session.exec(
                    select(ImageLabel.label_uuid, func.count().label("cnt"))
                    .where(
                        ImageLabel.image_uuid.in_(all_image_uuids),
                        ImageLabel.label_uuid.in_(image_label_uuids),
                        ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                    )
                    .group_by(ImageLabel.label_uuid)
                )
                for lbl_uuid, cnt in result.all():
                    if lbl_uuid in label_stats:
                        label_stats[lbl_uuid]["verified_image_count"] = cnt
                        label_stats[lbl_uuid]["unverified_image_count"] = total_images - cnt

                # Count PRESENT and ABSENT rows per label in a single query.
                result = await session.exec(
                    select(ImageLabel.label_uuid, ImageLabel.status, func.count().label("cnt"))
                    .where(
                        ImageLabel.image_uuid.in_(all_image_uuids),
                        ImageLabel.label_uuid.in_(image_label_uuids),
                        ImageLabel.status.in_([ImageItemStatus.PRESENT, ImageItemStatus.ABSENT]),
                    )
                    .group_by(ImageLabel.label_uuid, ImageLabel.status)
                )
                for lbl_uuid, status, cnt in result.all():
                    if lbl_uuid in label_stats:
                        key = "present_image_count" if status == ImageItemStatus.PRESENT else "absent_image_count"
                        label_stats[lbl_uuid][key] = cnt

            # ── Object-scope labels ─────────────────────────────────────────
            if object_label_uuids:
                # Total verified bboxes per label.
                verified_inner = (
                    select(BBox.label_uuid, BBox.uuid.label("bbox_uuid"))
                    .join(BBoxTagLink, BBox.uuid == BBoxTagLink.bbox_uuid)
                    .where(
                        BBox.image_uuid.in_(all_image_uuids),
                        BBox.label_uuid.in_(object_label_uuids),
                        BBoxTagLink.tag_uuid == VERIFIED_TAG_UUID,
                    )
                    .subquery()
                )
                result = await session.exec(
                    select(verified_inner.c.label_uuid, func.count().label("cnt")).group_by(verified_inner.c.label_uuid)
                )
                for lbl_uuid, cnt in result.all():
                    if lbl_uuid in label_stats:
                        label_stats[lbl_uuid]["verified_bbox_count"] = cnt

                # Total unverified bboxes per label.
                unverified_inner = (
                    select(BBox.label_uuid, BBox.uuid.label("bbox_uuid"))
                    .join(BBoxTagLink, BBox.uuid == BBoxTagLink.bbox_uuid)
                    .where(
                        BBox.image_uuid.in_(all_image_uuids),
                        BBox.label_uuid.in_(object_label_uuids),
                        BBoxTagLink.tag_uuid == UNVERIFIED_TAG_UUID,
                    )
                    .subquery()
                )
                result = await session.exec(
                    select(unverified_inner.c.label_uuid, func.count().label("cnt")).group_by(
                        unverified_inner.c.label_uuid
                    )
                )
                for lbl_uuid, cnt in result.all():
                    if lbl_uuid in label_stats:
                        label_stats[lbl_uuid]["unverified_bbox_count"] = cnt

                # Images with at least one unverified bbox per label (distinct images).
                imgs_missing_inner = (
                    select(BBox.label_uuid, BBox.image_uuid)
                    .join(BBoxTagLink, BBox.uuid == BBoxTagLink.bbox_uuid)
                    .where(
                        BBox.image_uuid.in_(all_image_uuids),
                        BBox.label_uuid.in_(object_label_uuids),
                        BBoxTagLink.tag_uuid == UNVERIFIED_TAG_UUID,
                    )
                    .distinct()
                    .subquery()
                )
                result = await session.exec(
                    select(imgs_missing_inner.c.label_uuid, func.count().label("cnt")).group_by(
                        imgs_missing_inner.c.label_uuid
                    )
                )
                imgs_missing_by_label: dict[str, int] = {}
                for lbl_uuid, cnt in result.all():
                    imgs_missing_by_label[lbl_uuid] = cnt
                    if lbl_uuid in label_stats:
                        label_stats[lbl_uuid]["images_missing_verification_count"] = cnt

                # Images with any bbox for this label (to derive fully_verified count).
                imgs_with_bbox_inner = (
                    select(BBox.label_uuid, BBox.image_uuid)
                    .where(
                        BBox.image_uuid.in_(all_image_uuids),
                        BBox.label_uuid.in_(object_label_uuids),
                    )
                    .distinct()
                    .subquery()
                )
                result = await session.exec(
                    select(imgs_with_bbox_inner.c.label_uuid, func.count().label("cnt")).group_by(
                        imgs_with_bbox_inner.c.label_uuid
                    )
                )
                for lbl_uuid, cnt in result.all():
                    if lbl_uuid in label_stats:
                        missing = imgs_missing_by_label.get(lbl_uuid, 0)
                        label_stats[lbl_uuid]["fully_verified_image_count"] = cnt - missing

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
                        for status_val, key in [
                            (ImageItemStatus.PRESENT, "present_image_count"),
                            (ImageItemStatus.ABSENT, "absent_image_count"),
                        ]:
                            distinct_sub = (
                                select(ImageLabel.image_uuid)
                                .where(
                                    ImageLabel.image_uuid.in_(all_image_uuids),
                                    ImageLabel.label_uuid.in_(all_rollup_uuids),
                                    ImageLabel.status == status_val,
                                )
                                .distinct()
                                .subquery()
                            )
                            r = await session.exec(select(func.count()).select_from(distinct_sub))
                            parent_stat[key] = r.one() or 0

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

                        # Image-level counts require distinct queries across all child labels.
                        imgs_missing_sub = (
                            select(BBox.image_uuid)
                            .join(BBoxTagLink, BBox.uuid == BBoxTagLink.bbox_uuid)
                            .where(
                                BBox.image_uuid.in_(all_image_uuids),
                                BBox.label_uuid.in_(all_rollup_uuids),
                                BBoxTagLink.tag_uuid == UNVERIFIED_TAG_UUID,
                            )
                            .distinct()
                            .subquery()
                        )
                        r = await session.exec(select(func.count()).select_from(imgs_missing_sub))
                        imgs_missing = r.one() or 0
                        parent_stat["images_missing_verification_count"] = imgs_missing

                        imgs_with_bbox_sub = (
                            select(BBox.image_uuid)
                            .where(
                                BBox.image_uuid.in_(all_image_uuids),
                                BBox.label_uuid.in_(all_rollup_uuids),
                            )
                            .distinct()
                            .subquery()
                        )
                        r = await session.exec(select(func.count()).select_from(imgs_with_bbox_sub))
                        imgs_with_bbox = r.one() or 0
                        parent_stat["fully_verified_image_count"] = max(imgs_with_bbox - imgs_missing, 0)

                    parent_stat["has_rollup"] = True
                    for child_uuid in child_uuids:
                        if child_uuid in label_stats:
                            label_stats[child_uuid]["roll_up_parent_uuid"] = rollup_parent_uuid

        return {"label_stats": list(label_stats.values()), "total_images": total_images}

    # ── Model Bundles ─────────────────────────────────────────────────────────

    async def create_bundle(self, data: ModelBundleCreate) -> ModelBundle:
        async with self._async_session_maker() as session:
            bundle = ModelBundle(
                name=data.name,
                description=data.description,
                model_config_uuid=data.model_config_uuid,
                bundle_path=data.bundle_path,
                manifest_json=data.manifest_json or {},
                training_config_uuids=data.training_config_uuids or [],
            )
            session.add(bundle)
            await session.commit()
            await session.refresh(bundle)
            return bundle

    async def list_bundles(self) -> list[ModelBundleRead]:
        async with self._async_session_maker() as session:
            result = await session.exec(select(ModelBundle))
            return [ModelBundleRead.model_validate(b) for b in result.all()]

    async def get_bundle(self, uuid: str) -> Optional[ModelBundleRead]:
        async with self._async_session_maker() as session:
            bundle = await session.get(ModelBundle, uuid)
            if bundle is None:
                return None
            return ModelBundleRead.model_validate(bundle)

    async def update_bundle(self, uuid: str, data: ModelBundleUpdate) -> Optional[ModelBundleRead]:
        async with self._async_session_maker() as session:
            bundle = await session.get(ModelBundle, uuid)
            if bundle is None:
                return None
            if data.name is not None:
                bundle.name = data.name
            if data.description is not None:
                bundle.description = data.description
            bundle.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(bundle)
            return ModelBundleRead.model_validate(bundle)

    async def delete_bundle(self, uuid: str) -> bool:
        async with self._async_session_maker() as session:
            bundle = await session.get(ModelBundle, uuid)
            if bundle is None:
                return False
            await session.delete(bundle)
            await session.commit()
        return True

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

    async def create_model_config(self, data: ModelConfigCreate) -> ModelConfigRead:
        # Validate uniqueness of label_uuid within the node list
        seen: set = set()
        for node in data.nodes:
            key = node.label_uuid  # None is valid once (root)
            if key in seen:
                raise ValueError(f"Duplicate label_uuid in nodes: {key!r}")
            seen.add(key)

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
                    )
                )
            await session.commit()
            stmt = select(ModelConfig).where(ModelConfig.uuid == mc.uuid).options(selectinload(ModelConfig.nodes))
            result = await session.exec(stmt)
            return self._mc_to_read(result.one())

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
                await session.exec(delete(ModelConfigNode).where(ModelConfigNode.model_config_uuid == uuid))
                if nodes_data:
                    seen: set = set()
                    for node in nodes_data:
                        key_lbl = node.get("label_uuid")
                        if key_lbl in seen:
                            raise ValueError(f"Duplicate label_uuid in nodes: {key_lbl!r}")
                        seen.add(key_lbl)
                        session.add(
                            ModelConfigNode(
                                model_config_uuid=uuid,
                                label_uuid=node.get("label_uuid"),
                                model_type=node["model_type"],
                                architecture=node["architecture"],
                                roll_up_unchecked=node.get("roll_up_unchecked", False),
                            )
                        )

            mc.updated_at = datetime.now(timezone.utc)
            session.add(mc)
            await session.commit()

            result2 = await session.exec(
                select(ModelConfig).where(ModelConfig.uuid == uuid).options(selectinload(ModelConfig.nodes))
            )
            return self._mc_to_read(result2.one())

    async def delete_model_config(self, uuid: str) -> bool:
        async with self._async_session_maker() as session:
            mc = await session.get(ModelConfig, uuid)
            if mc is None:
                return False
            await session.delete(mc)
            await session.commit()
        return True
