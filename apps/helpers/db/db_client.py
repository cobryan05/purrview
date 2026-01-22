from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from uuid import uuid4

import aiofiles
import aiosqlite
from dacite import Config, from_dict
from pydantic import BaseModel, Field

from apps.helpers.consts import TaskStatus

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


CLASSES_JSON = "classes.json"


class TagKinds:
    GENERIC = "generic"
    SYSTEM = "system"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT UNIQUE NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS classes (
    uuid TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    color TEXT,
    parent_uuid TEXT REFERENCES classes(uuid)
);

CREATE TABLE IF NOT EXISTS tags (
    uuid TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    color TEXT,
    kind TEXT NOT NULL DEFAULT 'generic',
    protected BOOLEAN NOT NULL DEFAULT FALSE,
    description TEXT,
    exclusive_group TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bboxes (
    uuid TEXT PRIMARY KEY,
    class_uuid TEXT NOT NULL,
    image_id INTEGER NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    width REAL NOT NULL,
    height REAL NOT NULL,
    metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY(class_uuid) REFERENCES classes(uuid)
);

CREATE TABLE IF NOT EXISTS bbox_tags (
    bbox_uuid TEXT NOT NULL,
    tag_uuid TEXT NOT NULL,
    PRIMARY KEY (bbox_uuid, tag_uuid),
    FOREIGN KEY(bbox_uuid) REFERENCES bboxes(uuid) ON DELETE CASCADE,
    FOREIGN KEY(tag_uuid) REFERENCES tags(uuid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS task_configs (
    uuid TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    typename TEXT NOT NULL,
    description TEXT,
    params_json TEXT CHECK (params_json IS NULL OR json_valid(params_json)) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    marked_for_delete BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tasks_active (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_uuid TEXT NOT NULL,
    status TEXT NOT NULL,
    resume_data_json TEXT CHECK (resume_data_json IS NULL OR json_valid(resume_data_json)),
    result_json TEXT CHECK (result_json IS NULL OR json_valid(result_json)),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    marked_for_delete BOOLEAN DEFAULT FALSE,
    FOREIGN KEY(config_uuid) REFERENCES task_configs(uuid) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS sources (
    uuid TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    typename TEXT NOT NULL,
    params_json TEXT CHECK (params_json IS NULL OR json_valid(params_json)) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Update Timestamp triggers
CREATE TRIGGER IF NOT EXISTS trg_images_updated_at
AFTER UPDATE ON images
FOR EACH ROW
WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE images
    SET updated_at = CURRENT_TIMESTAMP
    WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_task_configs_updated_at
AFTER UPDATE ON task_configs
FOR EACH ROW
WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE task_configs
    SET updated_at = CURRENT_TIMESTAMP
    WHERE uuid = OLD.uuid;
END;

CREATE TRIGGER IF NOT EXISTS trg_tasks_active_updated_at
AFTER UPDATE ON tasks_active
FOR EACH ROW
WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE tasks_active
    SET updated_at = CURRENT_TIMESTAMP
    WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_sources_updated_at
AFTER UPDATE ON sources
FOR EACH ROW
WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE sources
    SET updated_at = CURRENT_TIMESTAMP
    WHERE uuid = OLD.uuid;
END;

CREATE TRIGGER IF NOT EXISTS trg_tags_updated_at
AFTER UPDATE ON tags
FOR EACH ROW
WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE tags SET updated_at = CURRENT_TIMESTAMP WHERE uuid = OLD.uuid;
END;

CREATE TRIGGER IF NOT EXISTS trg_classes_updated_at
AFTER UPDATE ON classes
FOR EACH ROW
BEGIN
    UPDATE classes SET parent_uuid = NEW.parent_uuid WHERE uuid = OLD.uuid;
END;

CREATE TRIGGER IF NOT EXISTS trg_bboxes_updated_at
AFTER UPDATE ON bboxes
FOR EACH ROW
BEGIN
    UPDATE bboxes SET metadata_json = metadata_json WHERE uuid = OLD.uuid;
END;

-- Cleanup task configs when marked and unreferenced
CREATE TRIGGER IF NOT EXISTS cleanup_task_config_after_task_delete
AFTER DELETE ON tasks_active
FOR EACH ROW
BEGIN
    DELETE FROM task_configs
    WHERE marked_for_delete = 1
      AND uuid = OLD.config_uuid
      AND NOT EXISTS (
          SELECT 1 FROM tasks_active WHERE config_uuid = OLD.config_uuid
      );
END;

CREATE TRIGGER IF NOT EXISTS prevent_delete_parent_class
BEFORE DELETE ON classes
WHEN EXISTS (
    SELECT 1 FROM classes c WHERE c.parent_uuid = OLD.uuid
)
BEGIN
    SELECT RAISE(ABORT, 'Cannot delete class with children');
END;
"""


class TagMetadata(BaseModel):
    uuid: str
    name: str
    color: str
    protected: bool = False
    kind: Optional[str] = None
    exclusive_group: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ClassMetadata(BaseModel):
    uuid: str
    name: str
    color: str
    parent_uuid: Optional[str] = None


class ClassData(BaseModel):
    metadata: ClassMetadata
    children: List[ClassData] = Field(default_factory=list)


class BoundingBoxMetadata(BaseModel):
    class_uuid: str
    x: float
    y: float
    width: float
    height: float
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    tag_uuids: List[str] = Field(default_factory=list)
    extra: Dict[str, str] = Field(default_factory=dict)


class ImageMetadata(BaseModel):
    id: int
    filename: str
    last_updated: Optional[str] = None
    extra: Dict[str, str] = Field(default_factory=dict)
    boxes: List[BoundingBoxMetadata] = Field(default_factory=list)


class SourceMetadata(BaseModel):
    name: str
    typename: str
    params: dict[str, Any]
    uuid: str


class TaskConfigMetadata(BaseModel):
    uuid: str
    typename: str
    params: dict[str, Any]
    name: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ActiveTaskMetadata(BaseModel):
    id: int
    config_uuid: str
    typename: str
    status: str
    resume_data: Optional[dict[str, str]] = None
    result_json: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    expires_at: Optional[str] = None


class DbClient:
    """Helper class for SQLite database interactions for image annotations."""

    @staticmethod
    def _cols_from_basemodel(model_cls: Type) -> str:
        return ", ".join(model_cls.__fields__.keys())

    TAG_COLUMNS = _cols_from_basemodel(TagMetadata)

    def __init__(self, db_dir: Union[str, Path]):
        """
        Initialize DbClient with a path to the SQLite database file.

        Args:
            db_dir (str): Filesystem path to the directory containing the SQLite DB.
        """
        self._db_dir: Path = Path(db_dir)
        self._db_path: Path = self._db_dir / "db.sqlite"
        self._classes_json_path: Path = self._db_dir / CLASSES_JSON

    def db_exists(self) -> bool:
        """Checks if the database file exists"""
        return self._db_path.exists()

    def get_path(self) -> Path:
        """Gets the current db_path"""
        return self._db_path

    async def init_db(self) -> None:
        """Initialize DB schema; import classes.json only if DB was just created."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self._db_path) as db:
            # Acquire exclusive lock so multiple processes don't race the init
            await db.execute("BEGIN EXCLUSIVE")

            # Create tables if not exist
            await db.executescript(SCHEMA_SQL)

            await self._migrate_db(db)

            # Check if classes table is empty
            cursor = await db.execute("SELECT COUNT(*) FROM classes")
            (count,) = await cursor.fetchone()

            if count == 0:
                await self._import_classes_from_json(db)

            await db.commit()

    ################################################################################
    # Utils
    ################################################################################
    async def _migrate_db(self, db: aiosqlite.Connection) -> None:
        """
        Run simple migrations by adding new columns if they do not exist.
        """
        # Define desired columns per table
        migrations: dict[str, dict[str, str]] = {
            "tasks_active": {
                "expires_at": "TIMESTAMP",
                "marked_for_delete": "BOOLEAN DEFAULT FALSE",
            },
            "task_configs": {
                "marked_for_delete": "BOOLEAN DEFAULT FALSE",
            },
        }

        for table, new_columns in migrations.items():
            cursor = await db.execute(f"PRAGMA table_info({table})")
            rows = await cursor.fetchall()
            existing_cols = {row[1] for row in rows}

            for col, col_def in new_columns.items():
                if col not in existing_cols:
                    print(f"Adding column '{col}' to {table}")
                    await db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")

        await db.commit()

    ################################################################################
    # Images
    ################################################################################

    async def get_image_id_by_filename(self, filename: str) -> Optional[int]:
        """
        Retrieve image ID by filename.

        Args:
            filename (str): Filename of the image.

        Returns:
            Optional[int]: Image ID if found, otherwise None.
        """
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT id FROM images WHERE filename = ?", (filename,))
            row = await cursor.fetchone()
            await cursor.close()
            return row["id"] if row else None

    async def add_image(self, filename: str) -> Optional[int]:
        """
        Add a new image record or return existing ID.

        Args:
            filename (str): Filename of the image.

        Returns:
            int: Image ID.
        """
        existing_id = await self.get_image_id_by_filename(filename)
        if existing_id:
            return existing_id

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("INSERT INTO images (filename) VALUES (?)", (filename,))
            await db.commit()
            return cursor.lastrowid

    ################################################################################
    # Tasks
    ################################################################################

    async def add_task_config(
        self,
        name: str,
        typename: str,
        params: dict,
        description: Optional[str] = None,
    ) -> TaskConfigMetadata:
        config_uuid = str(uuid4())
        params_json = json.dumps(params)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO task_configs (uuid, name, typename, description, params_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (config_uuid, name, typename, description, params_json),
            )
            await db.commit()

            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM task_configs WHERE uuid = ?", (config_uuid,))
            row = await cursor.fetchone()
            await cursor.close()

        if not row:
            raise Exception(f"Failed to retrieve inserted task config with UUID {config_uuid}")

        data = dict(row)
        if "params_json" in data:
            data["params"] = json.loads(data.pop("params_json"))

        return TaskConfigMetadata(**data)

    async def update_task_config(
        self,
        config_uuid: str,
        name: Optional[str] = None,
        typename: Optional[str] = None,
        params: Optional[dict] = None,
        description: Optional[str] = None,
        marked_for_delete: Optional[bool] = None,
    ) -> None:
        """
        Update properties of a task configuration.

        Args:
            config_uuid (str): UUID of the task config.
            name (Optional[str]): New name.
            typename (Optional[str]): New typename.
            params (Optional[dict]): New parameters.
            description (Optional[str]): New description.
            marked_for_delete (Optional[bool]): Mark this config for deletion.
        """
        if not any([name, typename, params, description, marked_for_delete is not None]):
            return  # Nothing to update

        query_parts: list[str] = []
        query_params: list = []

        if name is not None:
            query_parts.append("name = ?")
            query_params.append(name)
        if typename is not None:
            query_parts.append("typename = ?")
            query_params.append(typename)
        if params is not None:
            query_parts.append("params_json = ?")
            query_params.append(json.dumps(params))
        if description is not None:
            query_parts.append("description = ?")
            query_params.append(description)
        if marked_for_delete is not None:
            query_parts.append("marked_for_delete = ?")
            query_params.append(1 if marked_for_delete else 0)

        query_params.append(config_uuid)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                f"UPDATE task_configs SET {', '.join(query_parts)} WHERE uuid = ?",
                tuple(query_params),
            )
            await db.commit()

    async def insert_new_active_task(
        self,
        config_uuid: str,
        expiry: Optional[Union[datetime, timedelta]] = None,
        resume_data: Optional[dict] = None,
    ) -> ActiveTaskMetadata:
        """
        Insert a new active task (runtime task) into the database.

        Args:
            config_uuid: UUID of the task configuration this task is based on.
            expiry: Optional expiration datetime (absolute) or timedelta (relative).
            resume_data: Optional dict containing serialized resume data. Stored as JSON in the DB.

        Returns:
            ActiveTaskMetadata: A dataclass instance representing the newly inserted active task,
            including its auto-generated ID, timestamps, and any provided metadata.
        """
        resume_data_json = json.dumps(resume_data) if resume_data is not None else None

        # Handle relative vs absolute expiry
        if isinstance(expiry, timedelta):
            expiry_value = datetime.utcnow() + expiry
        else:
            expiry_value = expiry

        # Store as ISO8601 string so SQLite's datetime functions work
        expiry_sql = expiry_value.isoformat(" ") if expiry_value else None

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO tasks_active (
                    config_uuid,
                    status,
                    resume_data_json,
                    error_message,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (config_uuid, TaskStatus.NEW, resume_data_json, None, expiry_sql),
            )
            await db.commit()

            cursor = await db.execute(
                """
                SELECT *
                FROM tasks_active ta
                JOIN task_configs tc ON ta.config_uuid = tc.uuid
                WHERE ta.rowid = last_insert_rowid()
                """
            )
            row = await cursor.fetchone()
            if not row:
                raise Exception("Failed to retrieve inserted active task")

            return DbClient.row_to_basemodel(cursor, row, ActiveTaskMetadata)

    async def delete_active_tasks(self, task_ids: Union[int, List[int]]) -> None:
        """
        Delete one or more tasks by their IDs.

        Args:
            task_ids (Union[int, List[int]]): A single task ID or a list of task IDs to delete.

        Raises:
            ValueError: If task_ids is an empty list.
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        elif isinstance(task_ids, list):
            if not task_ids:
                raise ValueError("No task IDs provided for deletion.")
        else:
            raise TypeError("task_ids must be an int or list of ints.")

        placeholders = ",".join("?" for _ in task_ids)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(f"DELETE FROM tasks_active WHERE id IN ({placeholders})", tuple(task_ids))
            await db.commit()

    async def delete_task_config(self, config_uuids: Union[str, List[str]]) -> None:
        """
        Delete one or more task configs by their UUID(s). If a config has
        active tasks, it will be marked for deletion instead.

        Args:
            config_uuids (Union[str, List[str]]): A single config UUID or a list of UUIDs.

        Raises:
            ValueError: If config_uuids is an empty list.
        """
        if isinstance(config_uuids, str):
            config_uuids = [config_uuids]
        elif isinstance(config_uuids, list):
            if not config_uuids:
                raise ValueError("No config UUIDs provided for deletion.")
        else:
            raise TypeError("config_uuids must be a str or list of strs.")

        placeholders = ",".join("?" for _ in config_uuids)

        async with aiosqlite.connect(self._db_path) as db:
            # Delete configs with no tasks referencing them
            cursor = await db.execute(
                f"""
                DELETE FROM task_configs
                WHERE uuid IN ({placeholders})
                AND NOT EXISTS (
                    SELECT 1 FROM tasks_active ta WHERE ta.config_uuid = task_configs.uuid
                )
                """,
                tuple(config_uuids),
            )

            # Mark the rest for deletion
            await db.execute(
                f"""
                UPDATE task_configs
                SET marked_for_delete = 1
                WHERE uuid IN ({placeholders})
            """,
                tuple(config_uuids),
            )

            await db.commit()

    async def get_task_configs(
        self,
        config_uuids: Optional[Union[str, List[str]]] = None,
        typename: Optional[Union[str, List[str]]] = None,
        name: Optional[Union[str, List[str]]] = None,
    ) -> List[TaskConfigMetadata]:
        """
        Retrieve task configs from the database with optional filtering by uuid, typename, and/or name.
        Each filter can be a single value or a list of values.

        Args:
            uuid (Optional[str or List[str]]): Filter by task config UUID(s).
            typename (Optional[str or List[str]]): Filter by typename(s).
            name (Optional[str or List[str]]): Filter by name(s).

        Returns:
            List[TaskConfigMetadata]: List of matching task configs.
        """
        query = """
            SELECT *
            FROM task_configs
            WHERE 1=1
        """
        params = []

        def add_filter(field, value):
            nonlocal query, params
            if value is None:
                return
            if isinstance(value, list):
                if not value:
                    query += " AND 1=0"  # No matches if empty list
                else:
                    placeholders = ",".join("?" for _ in value)
                    query += f" AND {field} IN ({placeholders})"
                    params.extend(value)
            else:
                query += f" AND {field} = ?"
                params.append(value)

        add_filter("uuid", config_uuids)
        add_filter("typename", typename)
        add_filter("name", name)

        query += " ORDER BY created_at DESC"

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, tuple(params))
            rows = await cursor.fetchall()
            await cursor.close()

        results = []
        for r in rows:
            data = dict(r)
            # convert JSON string to dict for Pydantic
            if "params_json" in data:
                data["params"] = json.loads(data.pop("params_json"))
            results.append(TaskConfigMetadata(**data))

        return results

    async def get_active_tasks(
        self,
        task_id: Optional[Union[int, List[int]]] = None,
        config_uuid: Optional[Union[str, List[str]]] = None,
        typename: Optional[Union[str, List[str]]] = None,
        status: Optional[Union[str, List[str]]] = None,
        resumable: Optional[bool] = None,
    ) -> List[ActiveTaskMetadata]:
        """
        Retrieve active tasks with optional filtering by id, config_uuid, typename (via task_configs), and/or status.
        Each filter can be a single value or a list of values.

        Args:
            task_id (Optional[int or List[int]]): Filter by tasks_active.id(s).
            config_uuid (Optional[str or List[str]]): Filter by tasks_active.config_uuid(s).
            typename (Optional[str or List[str]]): Filter by task_configs.typename(s).
            status (Optional[str or List[str]]): Filter by tasks_active.status(es).
            resumable (Optional[bool]): Filter tasks with or without resume data.

        Returns:
            List[ActiveTaskMetadata]: List of matching active tasks.
        """
        query = """
            SELECT *
            FROM tasks_active ta
            JOIN task_configs tc ON ta.config_uuid = tc.uuid
            WHERE 1=1
        """
        params = []

        def add_filter(field, value):
            nonlocal query, params
            if value is None:
                return
            if isinstance(value, list):
                if not value:
                    query += " AND 1=0"  # No match if list empty
                else:
                    placeholders = ",".join("?" for _ in value)
                    query += f" AND {field} IN ({placeholders})"
                    params.extend(value)
            else:
                query += f" AND {field} = ?"
                params.append(value)

        add_filter("ta.id", task_id)
        add_filter("ta.config_uuid", config_uuid)  # <-- added filter here
        add_filter("tc.typename", typename)
        add_filter("ta.status", status)

        if resumable is True:
            query += " AND ta.resume_data_json IS NOT NULL"
        elif resumable is False:
            query += " AND (ta.resume_data_json IS NULL OR TRIM(ta.resume_data_json) = '')"

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(query, tuple(params))
            rows = await cursor.fetchall()
            await cursor.close()

        return [DbClient.row_to_basemodel(cursor, r, ActiveTaskMetadata) for r in rows]

    async def get_task_has_result(self, task_id: int) -> bool:
        """
        Check whether a task has a result stored in the database.

        Args:
            task_id (int): Task ID

        Returns:
            bool: True if result_json is non-null and exists, False otherwise
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM tasks_active
                    WHERE id = ? AND result_json IS NOT NULL
                )
                """,
                (task_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()

            return bool(row[0])

    async def get_task_results(self, task_ids: Union[List[int], int]) -> Optional[dict[str, Any]]:
        """
        Retrieve result data for given tasks.

        Args:
            task_ids (List[int]): List of Task IDs

        Returns:
            dict[str, Any] or None if not found or empty
        """
        if isinstance(task_ids, int):
            task_ids = [task_ids]

        async with aiosqlite.connect(self._db_path) as db:
            clause, params = DbClient._make_in_clause("id", task_ids)
            cursor = await db.execute(f"SELECT result_json FROM tasks_active WHERE {clause}", params)
            rows = await cursor.fetchall()
            await cursor.close()

            if not rows:
                return None

            result = {}
            for task_id, row in zip(task_ids, rows):
                if not row[0]:
                    continue
                try:
                    result[task_id] = json.loads(row[0])
                except Exception as e:
                    logger.warning("Failed to parse JSON for task_id=%s: %s", task_id, e)
                    continue
            return result

    async def get_task_resume_data(self, task_id: int) -> Optional[dict[str, Any]]:
        """
        Retrieve resume data for a given task.

        Args:
            task_id (int): Task ID

        Returns:
            dict[str, Any] or None if not found or empty
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT resume_data_json FROM tasks_active WHERE id = ?", (task_id,))
            row = await cursor.fetchone()
            await cursor.close()

            if not row or not row[0]:
                return None

            try:
                return json.loads(row[0])
            except Exception as e:
                logger.warning(f"Invalid JSON in resume_data_json for task {task_id}: {e}")
                return None

    async def set_task_result(
        self,
        task_id: int,
        result: Optional[dict[str, Any]],
        status: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Set the result of a task, optionally updating its status and error message.

        Args:
            task_id (int): ID of the task.
            result (dict): Result dictionary to store as JSON.
            status (Optional[str]): New status to set (e.g., "completed", "error").
            error_message (Optional[str]): Optional error message.
        """
        result_json = json.dumps(result) if result else None
        parts = ["result_json = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: list[Any] = [result_json]

        if status is not None:
            parts.append("status = ?")
            params.append(status)

        if error_message is not None:
            parts.append("error_message = ?")
            params.append(error_message)

        params.append(task_id)

        query = f"UPDATE tasks_active SET {', '.join(parts)} WHERE id = ?"

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(query, tuple(params))
            await db.commit()

    async def set_task_resume_data(self, task_id: int, resume_data: dict[str, Any]) -> None:
        """
        Update the resume_data_json field for a task.

        Args:
            task_id (int): Task ID
            resume_data (dict): Resume state data
        """
        resume_data_json = json.dumps(resume_data)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE tasks_active
                SET resume_data_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (resume_data_json, task_id),
            )
            await db.commit()

    async def set_task_status(self, task_id: int, new_status: str) -> None:
        """
        Update the status of a task.

        Args:
            task_id (int): The ID of the task to update.
            new_status (str): The new status string (e.g., "pending", "running", "completed", "error").
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE tasks_active
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (new_status, task_id),
            )
            await db.commit()

    ################################################################################
    # Sources
    ################################################################################

    async def get_sources(self) -> List[SourceMetadata]:
        """
        List all sources.

        Returns:
            List[SourceMetaData]: All source entries.
        """
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM sources")
            rows = await cursor.fetchall()
            await cursor.close()
            results = []
            for r in rows:
                data = dict(r)
                if "params_json" in data:
                    data["params"] = json.loads(data.pop("params_json"))
                results.append(SourceMetadata(**data))
            return results

    async def add_source(self, name: str, typename: str, params: dict, uuid: Optional[str] = None) -> SourceMetadata:
        """
        Add a source or return existing one by name.

        Args:
            name (str): Source name.
            typename (str): Type of the image provider.
            params (dict): Parameters
            uuid (Optional[str]): Optional UUID.

        Returns:
            SourceMetaData: Source data.
        """
        uuid = uuid or str(uuid4())
        params_json = json.dumps(params)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO sources (name, typename, params_json, uuid) VALUES (?, ?, ?, ?)",
                (name, typename, params_json, uuid),
            )
            await db.commit()

            cursor = await db.execute(
                """
                SELECT * FROM sources
                WHERE uuid = ?
                """,
                (uuid,),
            )
            row = await cursor.fetchone()
            await cursor.close()

        if not row:
            raise ValueError("Failed to retrieve inserted source")
        return DbClient.row_to_basemodel(cursor, row, SourceMetadata)

    async def update_source(
        self,
        source_uuid: str,
        name: Optional[str] = None,
        typename: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> SourceMetadata:
        """
        Update source properties.

        Args:
            source_uuid (str): UUID of the source.
            name (Optional[str]): New name.
            typename (Optional[str]): New typename.
            params (Optional[dict]): New parameters
        """
        query_parts = []
        query_params = []

        if name:
            query_parts.append("name = ?")
            query_params.append(name)
        if typename:
            query_parts.append("typename = ?")
            query_params.append(typename)
            params = params or {}
        if params:
            query_parts.append("params_json = ?")
            query_params.append(json.dumps(params))

        query_params.append(source_uuid)

        async with aiosqlite.connect(self._db_path) as db:
            if query_parts:
                await db.execute(
                    f"UPDATE sources SET {', '.join(query_parts)} WHERE uuid = ?",
                    tuple(query_params),
                )
                await db.commit()

            cursor = await db.execute(
                """
                SELECT * FROM sources WHERE uuid = ?
                """,
                (source_uuid,),
            )
            row = await cursor.fetchone()
            await cursor.close()

        if not row:
            raise ValueError("Failed to retrieve updated source")
        return DbClient.row_to_basemodel(cursor, row, SourceMetadata)

    async def delete_sources(self, source_uuids: list[str]) -> None:
        """
        Delete sources.

        Args:
            source_uuids (list[str]): UUIDs of the sources to delete.
        """
        async with aiosqlite.connect(self._db_path) as db:
            for source_uuid in source_uuids:
                await db.execute("DELETE FROM sources WHERE uuid = ?", (source_uuid,))
            await db.commit()

    ################################################################################
    # Classes
    ################################################################################

    async def list_classes(self) -> List[ClassMetadata]:
        """
        List all classes.

        Returns:
            List[Dict]: List of classes dictionaries with keys: id, name, color, uuid.
        """
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM classes")
            rows = await cursor.fetchall()
            await cursor.close()
            return [ClassMetadata(**dict(r)) for r in rows]

    async def add_class(
        self, name: str, color: str, uuid: Optional[str] = None, parent_uuid: Optional[str] = None
    ) -> ClassMetadata:
        """
        Add a class or return existing one by name.

        Args:
            name (str): Class name.
            color (str): Class color hex string.
            uuid (Optional[str]): Optional UUID.

        Returns:
            Dict: Class data with id, name, color, uuid.
        """
        uuid = uuid or str(uuid4())
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM classes WHERE uuid = ?", (uuid,))
            row = await cursor.fetchone()
            if row:
                return ClassMetadata(**dict(row))

            await db.execute(
                "INSERT INTO classes (uuid, name, color, parent_uuid) VALUES (?, ?, ?, ?)",
                (uuid, name, color, parent_uuid),
            )
            await db.commit()

            cursor = await db.execute("SELECT * FROM classes WHERE uuid = ?", (uuid,))
            row = await cursor.fetchone()
            await cursor.close()

        if not row:
            raise ValueError("Failed to retrieve inserted class")
        return ClassMetadata(**dict(row))

    async def get_class_by_uuid(self, uuid: str) -> Optional[ClassMetadata]:
        """
        Retrieve a class by its UUID.

        Args:
            uuid (str): Class UUID.

        Returns:
            Optional[Dict]: Class data or None.
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT * FROM classes WHERE uuid = ?", (uuid,))
            row = await cursor.fetchone()
            await cursor.close()
            if row:
                return DbClient.row_to_basemodel(cursor, row, ClassMetadata)
            return None

    async def get_class_children(self, parent_uuid: str) -> List[str]:
        """
        Retrieve the UUIDs of classes that have the given UUID as their parent.

        Args:
            parent_uuid (str): The UUID of the parent class.

        Returns:
            List[str]: List of child class UUIDs.
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT uuid FROM classes WHERE parent_uuid = ?", (parent_uuid,))
            rows = await cursor.fetchall()
            await cursor.close()

            return [row[0] for row in rows]

    async def update_class(self, class_uuid: str, name: Optional[str] = None, color: Optional[str] = None) -> None:
        """
        Update class properties.

        Args:
            class_uuid (str): Class uuid.
            name (Optional[str]): New name.
            color (Optional[str]): New color.
        """
        if name is None and color is None:
            return
        query_parts = []
        params = []

        if name:
            query_parts.append("name = ?")
            params.append(name)
        if color:
            query_parts.append("color = ?")
            params.append(color)

        params.append(class_uuid)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(f"UPDATE classes SET {', '.join(query_parts)} WHERE uuid = ?", tuple(params))
            await db.commit()

    async def delete_class(self, class_uuid: str) -> None:
        """
        Delete a class and related entries.

        Args:
            class_uuid (int): Class ID to delete.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM classes WHERE uuid = ?", (class_uuid,))
            await db.commit()

    ################################################################################
    # Tags
    ################################################################################

    async def list_tags(self) -> List[TagMetadata]:
        """
        List all tags.

        Returns:
            List[Dict]: List of tags dictionaries with keys: id, name, color, uuid.
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(f"SELECT * FROM tags")
            rows = await cursor.fetchall()
            await cursor.close()
            return [DbClient.row_to_basemodel(cursor, r, TagMetadata) for r in rows]

    async def add_tag(
        self,
        name: str,
        color: str,
        uuid: Optional[str] = None,
        protected: bool = False,
        kind: str = TagKinds.GENERIC,
        exclusive_group: Optional[str] = None,
    ) -> TagMetadata:
        """
        Insert a tag, ignoring duplicates by UUID.
        Always returns the fully populated TagMetadata with DB defaults.
        """
        uuid = uuid or str(uuid4())

        async with aiosqlite.connect(self._db_path) as db:
            # Insert with conflict handling
            await db.execute(
                """
                INSERT INTO tags (uuid, name, color, protected, kind, exclusive_group)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (uuid, name, color, protected, kind, exclusive_group),
            )
            await db.commit()

            # Fetch the full row including defaults
            cursor = await db.execute(f"SELECT * FROM tags WHERE uuid = ?", (uuid,))
            row = await cursor.fetchone()
            await cursor.close()

            if not row:
                raise RuntimeError("Tag inserted but not found!")

            return DbClient.row_to_basemodel(cursor, row, TagMetadata)

    async def get_tag_by_uuid(self, uuid: str) -> Optional[TagMetadata]:
        """
        Retrieve a tag by its UUID.

        Args:
            uuid (str): Tag UUID.

        Returns:
            Optional[Dict]: Tag data or None.
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(f"SELECT * FROM tags WHERE uuid = ?", (uuid,))
            row = await cursor.fetchone()
            await cursor.close()
            if row:
                return self.row_to_basemodel(cursor, row, TagMetadata)
            return None

    async def update_tag(
        self,
        tag_uuid: str,
        name: Optional[str] = None,
        color: Optional[str] = None,
        protected: Optional[bool] = None,
        kind: Optional[str] = None,
        exclusive_group: Optional[str] = None,
    ) -> None:
        query_parts = []
        params = []

        if name is not None:
            query_parts.append("name = ?")
            params.append(name)
        if color is not None:
            query_parts.append("color = ?")
            params.append(color)
        if protected is not None:
            query_parts.append("protected = ?")
            params.append(protected)
        if kind is not None:
            query_parts.append("kind = ?")
            params.append(kind)
        if exclusive_group is not None:
            query_parts.append("exclusive_group = ?")
            params.append(exclusive_group)

        if not query_parts:
            return

        params.append(tag_uuid)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(f"UPDATE tags SET {', '.join(query_parts)} WHERE uuid = ?", tuple(params))
            await db.commit()

    async def delete_tag(self, tag_uuid: str) -> None:
        """
        Delete a tag and related entries.

        Args:
            tag_uuid (int): Tag ID to delete.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM tags WHERE uuid = ?", (tag_uuid,))
            await db.commit()

    ################################################################################
    # BBoxes
    ################################################################################

    async def add_bbox_for_image(
        self,
        image_filename: str,
        x: float,
        y: float,
        width: float,
        height: float,
        metadata: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Add bounding box to image, creating image record if needed.

        Args:
            image_filename (str): Filename of the image.
            x (float), y (float), width (float), height (float): Bounding box coordinates.
            metadata (Optional[Dict[str, str]]): Extra metadata for the box.

        Returns:
            int: New bounding box ID.
        """
        image_id = await self.add_image(image_filename)
        meta_json = json.dumps(metadata or {})
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "INSERT INTO bboxes (image_id, x, y, width, height, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
                (image_id, x, y, width, height, meta_json),
            )
            await db.commit()
            return cursor.lastrowid

    async def get_bboxes_info(self, bbox_uuids: List[str]) -> List[BoundingBoxMetadata]:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM bboxes WHERE uuid = ?",
                (bbox_uuid,),
            )

            rows = await cursor.fetchall()
            await cursor.close()
            if rows:
                return [self.row_to_basemodel(cursor, row, BoundingBoxMetadata) for row in rows]
            return []

    async def update_bbox(
        self,
        box_id: int,
        x: float,
        y: float,
        width: float,
        height: float,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Update bounding box properties.

        Args:
            box_id (int): Bounding box ID.
            x, y, width, height (float): New bounding box coords.
            metadata (Optional[Dict]): New extra metadata.
        """
        meta_json = json.dumps(metadata or {})
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE bboxes SET x = ?, y = ?, width = ?, height = ?, metadata_json = ? WHERE id = ?",
                (x, y, width, height, meta_json, box_id),
            )
            await db.commit()

    async def delete_bbox(self, box_id: int) -> None:
        """
        Delete bounding box and associated classs.

        Args:
            box_id (int): Bounding box ID.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM bbox_tags WHERE bbox_uuid = ?", (box_id,))
            await db.execute("DELETE FROM bboxes WHERE id = ?", (box_id,))
            await db.commit()

    async def assign_class_to_bbox(self, bbox_uuid: int, class_uuid: int) -> None:
        """
        Assign a class to a bounding box.

        Args:
            bbox_uuid (int): Bounding box ID.
            class_uuid (int): Class ID.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO bbox_tags (bbox_uuid, class_uuid) VALUES (?, ?)",
                (bbox_uuid, class_uuid),
            )
            await db.commit()

    async def remove_class_from_box(self, bbox_uuid: int, class_uuid: int) -> None:
        """
        Remove a class from a bounding box.

        Args:
            bbox_uuid (int): Bounding box ID.
            class_uuid (int): Class ID.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "DELETE FROM bbox_tags WHERE bbox_uuid = ? AND classuuid = ?",
                (bbox_uuid, class_uuid),
            )
            await db.commit()

    ################################################################################
    # Unsorted
    ################################################################################

    async def read_image_metadata_from_db(self, image_id: int) -> Optional[ImageMetadata]:
        """
        Read image metadata including bounding boxes and classes.

        Returns:
            ImageMetadata or None if image not found.
        """
        async with aiosqlite.connect(self._db_path) as db:
            # Read image base info + metadata JSON
            cursor = await db.execute(
                "SELECT id, filename, last_updated, metadata_json FROM images WHERE id = ?",
                (image_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if not row:
                return None

            img_id, filename, last_updated, meta_json = row
            image_extra = {}
            if meta_json:
                try:
                    image_extra = json.loads(meta_json)
                except Exception:
                    image_extra = {}

            # Read bounding boxes
            cursor = await db.execute(
                """
                SELECT
                    b.uuid,
                    b.class_uuid,
                    b.x,
                    b.y,
                    b.width,
                    b.height,
                    b.metadata_json
                FROM bboxes b
                WHERE b.image_id = ?
                """,
                (image_id,),
            )
            bbox_rows = await cursor.fetchall()
            await cursor.close()

            if not bbox_rows:
                return ImageMetadata(
                    id=img_id,
                    filename=filename,
                    last_updated=last_updated,
                    extra=image_extra,
                    boxes=[],
                )

            bbox_uuids = [row[0] for row in bbox_rows]

            # Read tags for all bboxes in one query
            placeholders = ",".join("?" for _ in bbox_uuids)
            cursor = await db.execute(
                f"""
                SELECT
                    bt.bbox_uuid,
                    bt.tag_uuid
                FROM bbox_tags bt
                WHERE bt.bbox_uuid IN ({placeholders})
                """,
                tuple(bbox_uuids),
            )
            tag_rows = await cursor.fetchall()
            await cursor.close()

            # Map bbox_uuid -> [tag_uuid, ...]
            tags_by_bbox: dict[str, list[str]] = {}
            for bbox_uuid, tag_uuid in tag_rows:
                tags_by_bbox.setdefault(bbox_uuid, []).append(tag_uuid)

            boxes = []
            for bbox_row in bbox_rows:
                bbox_uuid, bbox_class_uuid, x, y, w, h, bbox_meta_json = bbox_row
                bbox_extra = {}
                if bbox_meta_json:
                    try:
                        bbox_extra = json.loads(bbox_meta_json)
                    except Exception:
                        bbox_extra = {}

                boxes.append(
                    BoundingBoxMetadata(
                        uuid=bbox_uuid,
                        class_uuid=bbox_class_uuid,
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        extra=bbox_extra,
                        tag_uuids=tags_by_bbox.get(bbox_uuid, []),
                    )
                )

            return ImageMetadata(
                id=img_id,
                filename=filename,
                last_updated=last_updated,
                extra=image_extra,
                boxes=boxes,
            )

    async def write_image_metadata_to_db(self, image_id: int, metadata: ImageMetadata) -> None:
        """
        Write image metadata including bounding boxes and classes atomically.

        Args:
            image_id: ID of the image to update.
            metadata: ImageMetadata object containing image-level extra data and boxes.
        """
        image_meta_json = json.dumps(metadata.extra)

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("BEGIN"):
                # Update image metadata + last_updated
                await db.execute(
                    "UPDATE images SET metadata_json = ?, last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                    (image_meta_json, image_id),
                )

                # Insert / update bounding boxes and sync tags
                for box in metadata.boxes:
                    bbox_meta_json = json.dumps(box.extra) if box.extra else None

                    await db.execute(
                        """
                        INSERT INTO bboxes (uuid, image_id, class_uuid, x, y, width, height, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(uuid) DO UPDATE SET
                            image_id = excluded.image_id,
                            class_uuid = excluded.class_uuid,
                            x = excluded.x,
                            y = excluded.y,
                            width = excluded.width,
                            height = excluded.height,
                            metadata_json = excluded.metadata_json
                        """,
                        (
                            box.uuid,
                            image_id,
                            box.class_uuid,
                            box.x,
                            box.y,
                            box.width,
                            box.height,
                            bbox_meta_json,
                        ),
                    )

                    # Sync tags for this bbox
                    cursor = await db.execute(
                        "SELECT tag_uuid FROM bbox_tags WHERE bbox_uuid = ?",
                        (box.uuid,),
                    )
                    existing_tags = {row[0] for row in await cursor.fetchall()}
                    await cursor.close()

                    incoming_tags = set(box.tag_uuids)

                    # Add missing tags
                    for tag_uuid in incoming_tags - existing_tags:
                        await db.execute(
                            """
                            INSERT INTO bbox_tags (bbox_uuid, tag_uuid)
                            VALUES (?, ?)
                            """,
                            (box.uuid, tag_uuid),
                        )

                    # Remove stale tags
                    for tag_uuid in existing_tags - incoming_tags:
                        await db.execute(
                            """
                            DELETE FROM bbox_tags
                            WHERE bbox_uuid = ? AND tag_uuid = ?
                            """,
                            (box.uuid, tag_uuid),
                        )

                # Delete bboxes that exist in DB but not in incoming list
                incoming_uuids = [box.uuid for box in metadata.boxes]

                if incoming_uuids:
                    placeholders = ",".join("?" for _ in incoming_uuids)
                    cursor = await db.execute(
                        f"""
                        SELECT uuid FROM bboxes
                        WHERE image_id = ? AND uuid NOT IN ({placeholders})
                        """,
                        (image_id, *incoming_uuids),
                    )
                else:
                    cursor = await db.execute(
                        "SELECT uuid FROM bboxes WHERE image_id = ?",
                        (image_id,),
                    )

                bbox_uuids_to_del = [row[0] for row in await cursor.fetchall()]
                await cursor.close()

                if bbox_uuids_to_del:
                    placeholders = ",".join("?" for _ in bbox_uuids_to_del)
                    await db.execute(
                        f"DELETE FROM bbox_tags WHERE bbox_uuid IN ({placeholders})",
                        tuple(bbox_uuids_to_del),
                    )
                    await db.execute(
                        f"DELETE FROM bboxes WHERE uuid IN ({placeholders})",
                        tuple(bbox_uuids_to_del),
                    )

                await db.commit()

    async def export_classes_from_db_to_json(self, json_path: Optional[Path | str] = None) -> None:
        """
        Save all classes from the database into a JSON file.

        Args:
            json_path (str): Optional override path. Default: <db_dir>/classes`.json
        """
        json_path = Path(json_path) if json_path else self._db_path.parent / CLASSES_JSON
        classes = await self.list_classes()
        data = [asdict(cls) for cls in classes]

        async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2))

    async def _import_classes_from_json(self, db: aiosqlite.Connection) -> None:
        json_path = self._db_path.parent / CLASSES_JSON
        if json_path.exists():
            async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
                raw = await f.read()
                class_list = json.loads(raw)

            for cls in class_list:
                await db.execute(
                    """
                    INSERT INTO classes (uuid, name, color, parent_uuid)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        cls.get("uuid", str(uuid4())),
                        cls["name"],
                        cls.get("color"),
                        cls.get("parent_uuid"),
                    ),
                )
            logger.info(f"Imported {len(class_list)} classes from JSON.")

    async def save_image_metadata_db_to_json(self, abs_path: str) -> None:
        """
        Sync metadata from the database into the image's .json side file.

        Args:
            abs_path (str):Absolute path to the json file to write
        """

        # TODO: Check if json newer?
        image_id = await self.get_image_id_by_filename(abs_path)
        if image_id is None:
            logger.warning(f"No database entry found for image: {abs_path}")
            return

        metadata: Optional[ImageMetadata] = await self.read_image_metadata_from_db(image_id)
        if metadata is None:
            logger.warning(f"No metadata found for image: {abs_path}")
            return

        json_path = Path(abs_path).with_suffix(".json")
        os.makedirs(json_path.parent, exist_ok=True)

        async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(metadata.model_dump(), indent=2))

    async def read_image_metadata_json_to_db(self, abs_path: str) -> Optional[ImageMetadata]:
        """
        Load metadata from an image's .json side file and update the database.

        Args:
            abs_path (str): Absolute path to the image
        """
        try:
            json_path = Path(abs_path).with_suffix(".json")
            async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
                raw = await f.read()
                parsed = json.loads(raw)
            image_id = await self.add_image(abs_path)

            config = Config(strict=False)  # ignore unknown fields
            # Convert parsed dict into ImageMetadata dataclass
            image_meta: ImageMetadata = from_dict(data_class=ImageMetadata, data=parsed, config=config)
            image_meta.id = image_id
            await self.write_image_metadata_to_db(image_id, image_meta)
            return image_meta
        except FileNotFoundError:
            logger.warning(f"JSON file not found: {json_path}")
            return None

    def basemodel_to_row(model: BaseModel, json_fields: list[str] = None) -> dict[str, Any]:
        """
        Convert a BaseModel to a dict suitable for DB insertion,
        serializing any fields listed in json_fields.
        """
        row = model.model_dump()
        if json_fields:
            for field in json_fields:
                if field in row:
                    row[field] = json.dumps(row[field])
        return row

    T = TypeVar("T", bound=BaseModel)

    def row_to_basemodel(
        cursor,
        row: Tuple[Any, ...],
        cls_type: Type[T],
    ) -> T:
        """
        Convert a SQLite row into a Pydantic BaseModel instance.
        Handles columns ending in `_json`, which are JSON-decoded
        and mapped to the field without the `_json` suffix.
        """
        columns: List[str] = [col[0] for col in cursor.description]
        row_dict: dict[str, Any] = dict(zip(columns, row))

        model_fields = cls_type.model_fields  # pydantic v2 field set

        transformed: dict[str, Any] = {}

        for key, value in row_dict.items():

            # Handle JSON columns
            if key.endswith("_json"):
                new_key = key[:-5]
                if new_key not in model_fields:
                    continue
                try:
                    transformed[new_key] = json.loads(value) if value is not None else None
                except json.JSONDecodeError:
                    transformed[new_key] = value
                continue

            # Skip existing columns
            if key not in model_fields:
                continue

            # SQLite boolean coercion
            field_type = model_fields[key].annotation
            if field_type is bool and value is not None:
                transformed[key] = bool(value)
            else:
                transformed[key] = value

        return cls_type(**transformed)

    def _make_in_clause(column: str, values: list[int] | list[str]) -> tuple[str, tuple]:
        """
        Build a safe SQL IN clause with placeholders.

        Args:
            column: The column name for the IN clause.
            values: A non-empty list of values (ints or strs).

        Returns:
            (clause, params) where:
            clause = "column IN (?,?,?)"
            params = tuple(values)

        Raises:
            ValueError if values is empty.
        """
        if not values:
            raise ValueError("Values for IN clause cannot be empty")

        placeholders = ",".join("?" for _ in values)
        clause = f"{column} IN ({placeholders})"
        return clause, tuple(values)
