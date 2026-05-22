import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union
from uuid import uuid4

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig

ALEMBIC_INI_PATH = Path(__file__).parent / "alembic.ini"

import jstyleson
from pydantic import TypeAdapter
from sqlalchemy import delete, event, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import selectinload, sessionmaker
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from apps.helpers.consts import TaskStatus

from .types import (
    BBox,
    BBoxTagLink,
    ImageLabel,
    ImageLabelStatus,
    ImageRecord,
    ImageRecordRead,
    ImageRecordUpdate,
    Label,
    Source,
    Tag,
    TagKind,
    TaskConfig,
    TaskInstance,
)

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"
VERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000002"
FALSE_POS_TAG_UUID = "00000000-0000-0000-0000-000000000003"

_SYSTEM_TAGS = [
    Tag(
        uuid=UNVERIFIED_TAG_UUID,
        name="Unverified",
        color="#f0a030",
        kind=TagKind.SYSTEM,
        exclusive_group="verification",
        protected=True,
    ),
    Tag(
        uuid=VERIFIED_TAG_UUID,
        name="Verified",
        color="#30c060",
        kind=TagKind.SYSTEM,
        exclusive_group="verification",
        protected=True,
    ),
    Tag(
        uuid=FALSE_POS_TAG_UUID,
        name="FalsePositive",
        color="#e04040",
        kind=TagKind.SYSTEM,
        exclusive_group="verification",
        protected=True,
    ),
]

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


T = TypeVar("T", bound=SQLModel)



LABELS_JSON = "labels.json"
TAGS_JSON = "tags.json"
DB_FILE = "db.sqlite"


class DbClient:
    """Helper class for SQLModel interactions for image annotations."""

    def __init__(self, db_dir: Union[str, Path]):
        self._db_dir = Path(db_dir)
        self._db_path: Path = self._db_dir / DB_FILE
        self._labels_json_path: Path = self._db_dir / LABELS_JSON
        self._tags_json_path: Path = self._db_dir / TAGS_JSON

        self._url: str = f"sqlite+aiosqlite:///{self._db_path}"

        self._engine = create_async_engine(self._url, echo=False)

        @event.listens_for(self._engine.sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self._async_session_maker = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)

    async def init_db(self) -> None:
        """Initialize database and run any pending Alembic migrations."""
        self._db_dir.mkdir(parents=True, exist_ok=True)

        alembic_cfg = AlembicConfig(ALEMBIC_INI_PATH)
        alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{self._db_path}")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: alembic_command.upgrade(alembic_cfg, "head"))

        await self._seed_system_tags()

        # Sync labels from json if table is currently empty
        async with self._async_session_maker() as session:
            count_statement = select(func.count()).select_from(Label)
            result = await session.exec(count_statement)
            label_count = result.one() or 0

            count_statement = select(func.count()).select_from(Tag)
            result = await session.exec(count_statement)
            tag_count = result.one() or 0

        if label_count == 0:
            await self.sync_labels_from_json()
        if tag_count == 0:
            await self.sync_tags_from_json()

    async def _seed_system_tags(self) -> None:
        """Ensure system tags always exist (idempotent)."""
        async with self._async_session_maker() as session:
            for tag in _SYSTEM_TAGS:
                existing = await session.get(Tag, tag.uuid)
                if existing is None:
                    session.add(tag)
            await session.commit()

    async def close(self):
        """Dispose of the engine connection pool."""
        await self._engine.dispose()

    ################################################################################
    # Images
    ################################################################################
    async def sync_image_from_json(self, image_path: Path) -> ImageRecord:
        """
        Syncs a single image's JSON sidecar into the DB.
        Warns if UUIDs or Paths conflict with existing data.
        """
        json_path = image_path.with_suffix(".json")
        metadata_dict = {}
        json_uuid = None
        bboxes: List[BBox] = []

        if json_path.exists():
            try:
                metadata_dict = jstyleson.loads(json_path.read_text())
                json_uuid = metadata_dict.get("uuid")
                json_bboxes = metadata_dict.get("boxes", [])
                bboxes = [BBox(**box_data) for box_data in json_bboxes]
            except Exception as e:
                logger.error(f"Failed to parse {json_path}: {e}")

        async with self._async_session_maker() as session:
            path_stmt = select(ImageRecord).where(ImageRecord.filename == str(image_path))
            existing_by_path = (await session.exec(path_stmt)).one_or_none()

            existing_by_uuid = None
            if json_uuid:
                uuid_stmt = select(ImageRecord).where(ImageRecord.uuid == json_uuid)
                existing_by_uuid = (await session.exec(uuid_stmt)).one_or_none()

            if existing_by_path and json_uuid and existing_by_path.uuid != json_uuid:
                logger.warning(
                    f"Conflict: Path '{image_path}' exists in DB with UUID {existing_by_path.uuid}, "
                    f"but JSON sidecar specifies UUID {json_uuid}."
                )

            if existing_by_uuid and existing_by_uuid.filename != str(image_path):
                logger.warning(
                    f"Conflict: UUID {json_uuid} already belongs to file '{existing_by_uuid.filename}'. "
                    f"The sidecar at '{image_path}' is claiming a UUID used elsewhere."
                )

            temp_image = ImageRecord(
                uuid=json_uuid or str(uuid4()),
                filename=str(image_path),
                bboxes=bboxes,
                metadata_json=metadata_dict,
            )

            db_image = await session.merge(temp_image)
            await session.commit()
            await session.refresh(db_image)
            return db_image

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
                    selectinload(ImageRecord.bboxes).selectinload(BBox.label),
                    selectinload(ImageRecord.bboxes).selectinload(BBox.tags),
                )
            )
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

    async def add_image(self, filename: Path, metadata_json: Optional[Dict[str, Any]] = None) -> ImageRecord:
        """
        Add a new image record or return existing object if filename exists.

        Returns:
            Image: The fully populated Image SQLModel instance.
        """
        async with self._async_session_maker() as session:
            # Check for existing image to prevent UniqueConstraint errors
            statement = select(ImageRecord).where(ImageRecord.filename == str(filename))
            result = await session.exec(statement)
            existing_image = result.one_or_none()

            if existing_image:
                return existing_image

            # Create new record
            new_image = ImageRecord(uuid=str(uuid4()), filename=str(filename), metadata_json=metadata_json or {})

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

    async def list_images_with_bbox_tag(
        self,
        tag_uuid: str,
        label_uuid: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ImageRecord]:
        """Return images that have at least one bbox tagged with tag_uuid, optionally filtered by label."""
        async with self._async_session_maker() as session:
            stmt = (
                select(ImageRecord)
                .join(BBox, BBox.image_uuid == ImageRecord.uuid)
                .join(BBoxTagLink, BBoxTagLink.bbox_uuid == BBox.uuid)
                .where(BBoxTagLink.tag_uuid == tag_uuid)
                .options(
                    selectinload(ImageRecord.bboxes).selectinload(BBox.label),
                    selectinload(ImageRecord.bboxes).selectinload(BBox.tags),
                )
                .distinct()
                .offset(offset)
                .limit(limit)
            )
            if label_uuid:
                stmt = stmt.where(BBox.label_uuid == label_uuid)
            result = await session.exec(stmt)
            return result.all()

    async def delete_image(self, image_uuid: str) -> bool:
        """Delete an image by UUID. BBoxes will be deleted via CASCADE."""
        async with self._async_session_maker() as session:
            db_image = await session.get(ImageRecord, image_uuid)
            if not db_image:
                return False

            await session.delete(db_image)
            await session.commit()
            return True

    async def write_image_metadata_to_db(self, image_uuid: str, update: ImageRecordUpdate) -> None:
        """Replace all bboxes for an image directly in the database."""
        async with self._async_session_maker() as session:
            await session.exec(delete(BBox).where(BBox.image_uuid == image_uuid))
            for b in update.bboxes:
                session.add(BBox(
                    uuid=b.uuid,
                    image_uuid=image_uuid,
                    label_uuid=b.label_uuid,
                    x=b.x,
                    y=b.y,
                    width=b.width,
                    height=b.height,
                    metadata_json=b.metadata_json or {},
                ))
                for tag_uuid in b.tag_uuids:
                    session.add(BBoxTagLink(bbox_uuid=b.uuid, tag_uuid=tag_uuid))
            await session.commit()

    def write_image_metadata_to_json(self, json_path: Path, image_uuid: str, filename: str, update: ImageRecordUpdate) -> None:
        """Write image metadata to the JSON sidecar file."""
        existing: Dict[str, Any] = {}
        if json_path.exists():
            try:
                existing = jstyleson.loads(json_path.read_text())
            except Exception:
                pass
        existing.update({
            "uuid": image_uuid,
            "filename": filename,
            "boxes": [
                {"uuid": b.uuid, "label_uuid": b.label_uuid,
                 "x": b.x, "y": b.y, "width": b.width, "height": b.height}
                for b in update.bboxes
            ],
        })
        json_path.write_text(json.dumps(existing, indent=2))

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
        """Reads labels from JSON to db. Does not remove existing labels from database."""
        if not self._labels_json_path.exists():
            return

        # Read the raw bytes and load into list of dicts
        json_data = jstyleson.loads(self._labels_json_path.read_text())

        async with self._async_session_maker() as session:
            processed_uuids = set()

            # Multi-pass loop to handle the hierarchy
            pending = json_data
            while pending:
                batch_processed = False
                remaining = []

                for item in pending:
                    parent_uuid = item.get("parent_uuid")

                    # Can we process this now?
                    # Yes, if no parent OR parent already exists/processed
                    if parent_uuid is None or parent_uuid in processed_uuids:
                        # Use model_validate to leverage Pydantic's parsing
                        # then session.merge to handle potential existing records
                        label_obj = Label.model_validate(item)
                        await session.merge(label_obj)

                        processed_uuids.add(label_obj.uuid)
                        batch_processed = True
                    else:
                        remaining.append(item)

                if not batch_processed and remaining:
                    print(f"Warning: Orphaned labels found in JSON: {[x['name'] for x in remaining]}")
                    break

                pending = remaining

            await session.commit()
        logger.info(f"Imported {len(processed_uuids)} labels from JSON")

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
    ) -> Label:
        """Insert a label and return the created object."""
        new_label = Label(uuid=uuid or str(uuid4()), name=name, color=color, parent_uuid=parent_uuid)

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
        """Reads tags.json and syncs to DB. Does not remove existing tags."""
        if not self._tags_json_path.exists():
            return

        try:
            json_data = jstyleson.loads(self._tags_json_path.read_text())
        except Exception as e:
            logger.error(f"Failed to parse tags.json: {e}")
            return

        async with self._async_session_maker() as session:
            for item in json_data:
                # Use merge so we update existing tags if the UUID matches
                tag_obj = Tag.model_validate(item)
                await session.merge(tag_obj)

            await session.commit()
        logger.info(f"Imported {len(json_data)} tags from JSON")

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
    # Sources
    ################################################################################
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
        return new_source

    async def delete_sources(self, uuids: List[str]) -> None:
        """Delete sources by UUID list."""
        async with self._async_session_maker() as session:
            await session.exec(delete(Source).where(Source.uuid.in_(uuids)))
            await session.commit()

    async def update_source(self, source_uuid: str, name: str, typename: str, params: Dict[str, Any]) -> Optional[Source]:
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

    async def delete_task_config(self, config_uuids: Union[str, List[str]]) -> None:
        if isinstance(config_uuids, str):
            config_uuids = [config_uuids]
        if not config_uuids:
            return
        async with self._async_session_maker() as session:
            # Delete configs with no task instances referencing them
            configs_with_instances = select(TaskInstance.config_uuid).where(
                TaskInstance.config_uuid.in_(config_uuids)
            )
            await session.exec(
                delete(TaskConfig).where(
                    TaskConfig.uuid.in_(config_uuids),
                    ~TaskConfig.uuid.in_(configs_with_instances),
                )
            )
            # Mark the rest for deletion
            result = await session.exec(
                select(TaskConfig).where(TaskConfig.uuid.in_(config_uuids))
            )
            for config in result.all():
                config.marked_for_delete = True
                session.add(config)
            await session.commit()

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
            stmt = select(TaskInstance).options(selectinload(TaskInstance.config)).where(
                TaskInstance.uuid == instance.uuid
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
