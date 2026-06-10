import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union
from uuid import uuid4

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig

ALEMBIC_INI_PATH = Path(__file__).parent / "alembic.ini"

import jstyleson
from alembic.runtime.migration import MigrationContext
from pydantic import TypeAdapter
from sqlalchemy import Select, and_, delete, event, func, or_, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import selectinload, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from apps.helpers.consts import TaskStatus

from .types import (
    AVAILABLE_MODEL_TYPENAMES,
    DUPLICATE_TAG_UUID,
    FALSE_POS_TAG_UUID,
    REJECTION_TAG_UUID,
    UNVERIFIED_TAG_UUID,
    BBox,
    BboxAgeCondition,
    BBoxTagLink,
    Dataset,
    DatasetCreate,
    DatasetFilterCriteria,
    DatasetLabelLink,
    DatasetLabelStatusCondition,
    DatasetRead,
    DatasetUpdate,
    FilenameCondition,
    ImageAgeCondition,
    ImageLabel,
    ImageLabelFilter,
    ImageLabelStatus,
    ImageRecord,
    ImageRecordRead,
    ImageRecordUpdate,
    ImageSearchParams,
    Label,
    LabelType,
    ModelConfig,
    ModelConfigCreate,
    ModelConfigNode,
    ModelConfigRead,
    ModelConfigUpdate,
    Source,
    SourceRead,
    Tag,
    TagCondition,
    TagKind,
    TagPresenceFilter,
    TaskConfig,
    TaskInstance,
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
        uuid=DUPLICATE_TAG_UUID,
        name="Duplicate",
        color="#e05c5c",
        kind=TagKind.SYSTEM,
        exclusive_group=None,
        protected=True,
        icon="⊘",
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
        uuid=REJECTION_TAG_UUID,
        name="Rejection",
        color="#8e44ad",
        kind=TagKind.SYSTEM,
        exclusive_group=None,
        protected=True,
        icon="⊗",
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
        image_uuids = (
            select(BBox.image_uuid)
            .where(BBox.label_uuid == params.label_uuid)
            .distinct()
        )
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
                    ImageLabel.status.in_([ImageLabelStatus.PRESENT, ImageLabelStatus.ABSENT]),
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
                image_uuids = (
                    select(BBox.image_uuid)
                    .where(condition)
                    .distinct()
                )
                stmt = stmt.where(ImageRecord.uuid.in_(image_uuids))
        else:
            # "any of" — image has a bbox with ANY of the selected labels (or
            # their descendants when label_include_children is True).
            if params.label_include_children:
                label_condition = BBox.label_uuid.in_(_label_uuids_with_descendants(params.label_uuids))
            else:
                label_condition = BBox.label_uuid.in_(params.label_uuids)
            image_uuids = (
                select(BBox.image_uuid)
                .where(label_condition)
                .distinct()
            )
            stmt = stmt.where(ImageRecord.uuid.in_(image_uuids))
        _bbox_implied = True

    # ── Require at least one bbox ─────────────────────────────────────────────
    # Skip when any earlier filter already guarantees bbox existence.
    if params.require_bboxes and not _bbox_implied:
        image_uuids = select(BBox.image_uuid).distinct()
        stmt = stmt.where(ImageRecord.uuid.in_(image_uuids))

    # ── Label exclusions (NOT IN per label UUID) ───────────────────────────────
    for label_uuid in params.label_exclusions:
        image_uuids = (
            select(BBox.image_uuid)
            .where(BBox.label_uuid == label_uuid)
            .distinct()
        )
        stmt = stmt.where(ImageRecord.uuid.not_in(image_uuids))

    return stmt


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
            count_statement = select(func.count()).select_from(Dataset)
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
                    )
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
                    )
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
            new_image = ImageRecord(uuid=str(uuid4()), filename=str(filename), metadata_json=metadata_json or {})
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
            stmt = (
                select(ImageRecord)
                .options(
                    selectinload(ImageRecord.bboxes).options(
                        selectinload(BBox.label),
                        selectinload(BBox.tags),
                    )
                )
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
        """Return a bbox and its parent image record, or None if not found."""
        async with self._async_session_maker() as session:
            stmt = select(BBox).where(BBox.uuid == bbox_uuid)
            result = await session.exec(stmt)
            bbox = result.first()
            if bbox is None:
                return None
            image = await session.get(ImageRecord, bbox.image_uuid)
            return bbox, image

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
        self, json_path: Path, image_uuid: str, filename: str, update: ImageRecordUpdate,
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
                "__all__": {"bboxes": True, "children": True, "parent": True, "created_at": True, "updated_at": True}
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
    ) -> Label:
        """Insert a label and return the created object."""
        new_label = Label(uuid=uuid or str(uuid4()), name=name, color=color, parent_uuid=parent_uuid, label_type=label_type)

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

    async def set_image_label(self, image_uuid: str, label_uuid: str, status: ImageLabelStatus) -> None:
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
            await session.delete(dataset)
            await session.commit()
        await self._sync_datasets_to_json()
        return True

    @staticmethod
    def build_dataset_search_params(
        dataset: DatasetRead,
        criteria: DatasetFilterCriteria,
        *,
        limit: int = 50,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
        after_uuid: Optional[str] = None,
        after_sort_value: Optional[datetime] = None,
    ) -> ImageSearchParams:
        """Expand a DatasetFilterCriteria into an ImageSearchParams.

        All conditions in the criteria are AND-combined.
        DatasetLabelStatusCondition entries are expanded into one ImageLabelFilter
        per label in dataset.label_uuids.
        """
        now = datetime.now(timezone.utc)

        params = ImageSearchParams(
            show_all_bboxes=True,
            tag_uuids=[],
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
            after_uuid=after_uuid,
            after_sort_value=after_sort_value,
        )

        image_label_filters: List[ImageLabelFilter] = []
        tag_presence_filters: List[TagPresenceFilter] = []
        filename_searches: List[str] = []

        for cond in criteria.conditions:
            if isinstance(cond, ImageAgeCondition):
                if cond.within_days is not None:
                    cutoff = now - timedelta(days=cond.within_days)
                    if cond.field == "created_at":
                        params.date_from = cutoff
                    else:
                        params.updated_date_from = cutoff
                else:
                    if cond.field == "created_at":
                        if cond.after:
                            params.date_from = cond.after
                        if cond.before:
                            params.date_to = cond.before
                    else:
                        if cond.after:
                            params.updated_date_from = cond.after
                        if cond.before:
                            params.updated_date_to = cond.before

            elif isinstance(cond, BboxAgeCondition):
                # Filtered via EXISTS on bboxes.{field}. Handled separately in
                # list_dataset_images since ImageSearchParams has no bbox-age field.
                pass  # stored in criteria, executed in list_dataset_images

            elif isinstance(cond, TagCondition):
                tag_presence_filters.append(TagPresenceFilter(tag_uuid=cond.tag_uuid, exclude=cond.exclude))

            elif isinstance(cond, DatasetLabelStatusCondition):
                if not dataset.label_uuids:
                    # Dataset has no labels — condition can never be satisfied.
                    # Inject an impossible filter so the query returns zero rows.
                    image_label_filters.append(
                        ImageLabelFilter(
                            label_uuid="00000000-0000-0000-0000-000000000000",
                            status=ImageLabelStatus.PRESENT,
                            require_verified=True,
                        )
                    )
                else:
                    # Expand: one filter per dataset label
                    for label_uuid in dataset.label_uuids:
                        image_label_filters.append(
                            ImageLabelFilter(
                                label_uuid=label_uuid,
                                status=ImageLabelStatus.PRESENT,  # ignored when require_verified=True
                                require_verified=True,
                            )
                        )

            elif isinstance(cond, FilenameCondition):
                filename_searches.append(cond.pattern)

        params.image_label_filters = image_label_filters
        params.tag_presence_filters = tag_presence_filters
        params.filename_searches = filename_searches
        # When the dataset has label_uuids, restrict to images that have at least
        # one bbox annotated with one of those labels.
        if dataset.label_uuids:
            params.label_uuids = dataset.label_uuids
        return params

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
        BboxAgeCondition entries are applied as an additional in-DB filter after
        the base query.
        """
        criteria = dataset.filter_json

        # Fetch limit+1 rows so we can distinguish "exactly limit" from "more exist".
        params = self.build_dataset_search_params(
            dataset,
            criteria,
            limit=limit + 1,
            sort_by=sort_by,
            sort_order=sort_order,
            after_uuid=after_uuid,
            after_sort_value=after_sort_value,
        )

        # For BboxAgeCondition, apply extra WHERE clauses directly.
        bbox_age_conditions = [c for c in criteria.conditions if isinstance(c, BboxAgeCondition)]

        if not bbox_age_conditions:
            results = await self.list_images_search(params)
            has_more = len(results) > limit
            return results[:limit], has_more

        # Apply bbox-age filters with an additional DB query against the matched UUID set.
        # has_more is derived from the base page so we never terminate pagination
        # early when the in-DB filter reduces the page below `limit`.
        now = datetime.now(timezone.utc)
        results = await self.list_images_search(params)
        has_more = len(results) > limit
        results = results[:limit]  # trim the sentinel row before further filtering

        if not results:
            return results, False

        filtered_uuids = {img.uuid for img in results}
        async with self._async_session_maker() as session:
            for cond in bbox_age_conditions:
                cutoff: Optional[datetime] = None
                if cond.within_days is not None:
                    cutoff = now - timedelta(days=cond.within_days)
                elif cond.after:
                    cutoff = cond.after

                if cutoff is None and cond.before is None:
                    continue

                col = BBox.created_at if cond.field == "created_at" else BBox.updated_at
                stmt = select(BBox.image_uuid).where(BBox.image_uuid.in_(list(filtered_uuids))).distinct()
                if cutoff:
                    stmt = stmt.where(col >= cutoff)
                if cond.before:
                    stmt = stmt.where(col <= cond.before)
                result = await session.exec(stmt)
                matching = set(result.all())
                filtered_uuids = filtered_uuids & matching

        return [img for img in results if img.uuid in filtered_uuids], has_more

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
                        model_typename=node.model_typename,
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
                                model_typename=node["model_typename"],
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
