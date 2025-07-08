import json
import logging
import os
import sys
from uuid import uuid4
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import aiosqlite
from dacite import from_dict

from apps.helpers.fileUtils import get_safe_path

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@dataclass
class LabelMetaData:
    name: str
    color: str
    uuid: str
    parent_uuid: Optional[int]


@dataclass
class BoundingBoxMetadata:
    id: Optional[int]  # may be None for new boxes
    label_uuid: str
    label_text: str
    x: float
    y: float
    width: float
    height: float
    tags: List[LabelMetaData] = field(default_factory=list)
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass
class ImageMetadata:
    id: int
    filename: str
    last_updated: Optional[str]
    extra: Dict[str, str] = field(default_factory=dict)
    boxes: List[BoundingBoxMetadata] = field(default_factory=list)


@dataclass
class TaskRecord:
    id: int
    name: str
    status: str
    params_json: str
    resume_data_json: Optional[str]
    result_json: Optional[str]
    error_message: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class DbClient:
    """Helper class for SQLite database interactions for image annotations."""

    def __init__(self, db_path: str):
        """
        Initialize DbClient with a path to the SQLite database file.

        Args:
            db_path (str): Filesystem path to the SQLite DB.
        """
        self._db_path = db_path

    def db_exists(self) -> bool:
        """Checks if the database file exists"""
        return os.path.exists(self._db_path)

    async def init_db(self) -> None:
        """
        Initiaize the database schema if it does not exist.
        """
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                );
                CREATE TABLE IF NOT EXISTS labels (
                    uuid TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    color TEXT,
                    parent_id TEXT REFERENCES labels(uuid)
                );
                CREATE TABLE IF NOT EXISTS bounding_boxes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label_uuid TEXT NOT NULL,
                    image_id INTEGER NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    width REAL NOT NULL,
                    height REAL NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS bbox_tags (
                    bbox_id INTEGER NOT NULL,
                    label_uuid TEXT NOT NULL,
                    PRIMARY KEY (bbox_id, label_uuid),
                    FOREIGN KEY(bbox_id) REFERENCES bounding_boxes(id) ON DELETE CASCADE,
                    FOREIGN KEY(label_uuid) REFERENCES labels(uuid) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    resume_data_json TEXT,
                    result_json TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            await db.commit()

    async def get_image_id_by_filename(self, filename: str) -> Optional[int]:
        """
        Retrieve image ID by filename.

        Args:
            filename (str): Filename of the image.

        Returns:
            Optional[int]: Image ID if found, otherwise None.
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT id FROM images WHERE filename = ?", (filename,))
            row = await cursor.fetchone()
            await cursor.close()
            return row[0] if row else None

    async def add_image(self, filename: str) -> int:
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

    async def add_task(self, typename: str, params: dict) -> TaskRecord:
        """
        Insert a new task into the database with an auto-incrementing ID.

        Args:
            name (str): Task name (from registry).
            params (dict): Task parameters.
            resume_data (Optional[dict]): Optional resume state.

        Returns:
            TaskRecord: The full task record, including auto-generated ID.
        """
        params_json = json.dumps(params)
        status = "pending"
        resume_data_json = None
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO tasks (name, status, params_json, resume_data_json)
                VALUES (?, ?, ?, ?)
                """,
                (typename, status, params_json, resume_data_json),
            )
            await db.commit()

            task_id = cursor.lastrowid

            cursor = await db.execute(
                """
                SELECT id, name, status, params_json, resume_data_json,
                       result_json, error_message, created_at, updated_at
                FROM tasks WHERE id = ?
                """,
                (task_id,),
            )
            row = await cursor.fetchone()
            if not row:
                raise Exception(f"Failed to retrieve inserted task with ID {task_id}")

            return TaskRecord(*row)

    async def list_labels(self) -> List[LabelMetaData]:
        """
        List all labels.

        Returns:
            List[Dict]: List of label dictionaries with keys: id, name, color, uuid.
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT name, color, uuid, parent_uuid FROM labels")
            rows = await cursor.fetchall()
            await cursor.close()
            return [LabelMetaData(name=r[0], color=r[1], uuid=r[2], parent_uuid=r[3]) for r in rows]

    async def add_label(
        self, name: str, color: str, uuid: Optional[str] = None, parent_uuid: Optional[str] = None
    ) -> LabelMetaData:
        """
        Add a label or return existing one by name.

        Args:
            name (str): Label name.
            color (str): Label color hex string.
            uuid (Optional[str]): Optional UUID.

        Returns:
            Dict: Label data with id, name, color, uuid.
        """
        uuid = uuid or str(uuid4())
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT name, color, uuid, parent_uuid FROM labels WHERE name = ?", (name,))
            row = await cursor.fetchone()
            if row:
                return LabelMetaData(name=row[0], color=row[1], uuid=row[2], parent_uuid=row[3])

            cursor = await db.execute(
                "INSERT INTO labels (name, color, uuid, parent_uuid) VALUES (?, ?, ?, ?)",
                (name, color, uuid, parent_uuid),
            )
            await db.commit()
            label_uuid = cursor.lastrowid
            if label_uuid is None:
                raise Exception(f"Failed to insert label: {name}")
            return LabelMetaData(name=name, color=color, uuid=uuid, parent_uuid=parent_uuid)

    async def get_label_by_name(self, name: str) -> Optional[LabelMetaData]:
        """
        Retrieve a label by its name.

        Args:
            name (str): Label name.

        Returns:
            Optional[Dict]: Label data or None.
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT name, color, uuid FROM labels WHERE name = ?", (name,))
            row = await cursor.fetchone()
            await cursor.close()
            if row:
                return LabelMetaData(name=row[0], color=row[1], uuid=row[2])
            return None

    async def update_label(self, label_uuid: str, name: Optional[str] = None, color: Optional[str] = None) -> None:
        """
        Update label properties.

        Args:
            label_uuid (str): Label uuid.
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

        params.append(label_uuid)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(f"UPDATE labels SET {', '.join(query_parts)} WHERE uuid = ?", tuple(params))
            await db.commit()

    async def delete_label(self, label_uuid: str) -> None:
        """
        Delete a label and related entries.

        Args:
            label_uuid (int): Label ID to delete.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM bbox_tags WHERE label_uuid = ?", (label_uuid,))
            await db.execute("DELETE FROM labels WHERE uuid = ?", (label_uuid,))
            await db.commit()

    async def add_box_for_image(
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
                "INSERT INTO bounding_boxes (image_id, x, y, width, height, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (image_id, x, y, width, height, meta_json),
            )
            await db.commit()
            return cursor.lastrowid

    async def update_box(
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
                "UPDATE bounding_boxes SET x = ?, y = ?, width = ?, height = ?, metadata = ? WHERE id = ?",
                (x, y, width, height, meta_json, box_id),
            )
            await db.commit()

    async def delete_box(self, box_id: int) -> None:
        """
        Delete bounding box and associated labels.

        Args:
            box_id (int): Bounding box ID.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM bbox_tags WHERE bbox_id = ?", (box_id,))
            await db.execute("DELETE FROM bounding_boxes WHERE id = ?", (box_id,))
            await db.commit()

    async def assign_label_to_box(self, bbox_id: int, label_uuid: int) -> None:
        """
        Assign a label to a bounding box.

        Args:
            bbox_id (int): Bounding box ID.
            label_uuid (int): Label ID.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO bbox_tags (bbox_id, label_uuid) VALUES (?, ?)",
                (bbox_id, label_uuid),
            )
            await db.commit()

    async def remove_label_from_box(self, bbox_id: int, label_uuid: int) -> None:
        """
        Remove a label from a bounding box.

        Args:
            bbox_id (int): Bounding box ID.
            label_uuid (int): Label ID.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "DELETE FROM bbox_tags WHERE bbox_id = ? AND label_uuid = ?",
                (bbox_id, label_uuid),
            )
            await db.commit()

    async def read_metadata(self, image_id: int) -> Optional[ImageMetadata]:
        """
        Read image metadata including bounding boxes and labels.

        Returns:
            ImageMetadata or None if image not found.
        """
        async with aiosqlite.connect(self._db_path) as db:
            # Read image base info + metadata JSON
            cursor = await db.execute(
                "SELECT id, filename, last_updated, metadata FROM images WHERE id = ?",
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
                    b.id,
                    b.label_uuid,
                    l.name AS label_name,
                    b.x,
                    b.y,
                    b.width,
                    b.height,
                    b.metadata
                FROM bounding_boxes b
                JOIN labels l ON b.label_uuid = l.uuid
                WHERE b.image_id = ?
                """,
                (image_id,),
            )
            bbox_rows = await cursor.fetchall()
            await cursor.close()

            boxes = []
            for bbox_row in bbox_rows:
                bbox_id, bbox_label_uuid, bbox_label_text, x, y, w, h, bbox_meta_json = bbox_row
                bbox_extra = {}
                if bbox_meta_json:
                    try:
                        bbox_extra = json.loads(bbox_meta_json)
                    except Exception:
                        bbox_extra = {}

                # TODO TAGS
                # Read labels for this box
                tag_cursor = await db.execute(
                    """
                    SELECT labels.uuid, labels.name, labels.color
                    FROM labels
                    JOIN bbox_tags ON labels.uuid = bbox_tags.label_uuid
                    WHERE bbox_tags.bbox_id = ?
                    """,
                    (bbox_id,),
                )
                tag_rows = await tag_cursor.fetchall()
                await tag_cursor.close()

                tags = [LabelMetaData(uuid=l[0], name=l[1], color=l[2]) for l in tag_rows]

                boxes.append(
                    BoundingBoxMetadata(
                        id=bbox_id,
                        label_uuid=bbox_label_uuid,
                        label_text=bbox_label_text,
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        extra=bbox_extra,
                    )
                )

            return ImageMetadata(
                id=img_id,
                filename=filename,
                last_updated=last_updated,
                extra=image_extra,
                boxes=boxes,
            )

    async def write_metadata_to_db(self, image_id: int, metadata: ImageMetadata) -> None:
        """
        Write image metadata including bounding boxes and labels atomically.

        Args:
            image_id: ID of the image to update.
            metadata: ImageMetadata object containing image-level extra data and boxes.
        """
        image_meta_json = json.dumps(metadata.extra)

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("BEGIN"):
                # Update image metadata + last_updated
                await db.execute(
                    "UPDATE images SET metadata = ?, last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                    (image_meta_json, image_id),
                )

                # Delete existing bounding boxes and bbox_tags for image
                await db.execute(
                    "DELETE FROM bbox_tags WHERE bbox_id IN (SELECT id FROM bounding_boxes WHERE image_id = ?)",
                    (image_id,),
                )
                await db.execute("DELETE FROM bounding_boxes WHERE image_id = ?", (image_id,))

                # Insert bounding boxes
                for box in metadata.boxes:
                    bbox_meta_json = json.dumps(box.extra) if box.extra else None
                    cursor = await db.execute(
                        "INSERT INTO bounding_boxes (image_id, label_uuid, x, y, width, height, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (image_id, box.label_uuid, box.x, box.y, box.width, box.height, bbox_meta_json),
                    )
                    new_bbox_id = cursor.lastrowid

                await db.commit()

    async def save_labels_metadata_db_to_json(self, json_path: Optional[str] = None) -> None:
        """
        Save all labels from the database into a JSON file.

        Args:
            json_path (str): Optional override path. Default: <db_dir>/labels.json
        """
        json_path = json_path or os.path.join(os.path.dirname(self._db_path), "labels.json")
        labels = await self.list_labels()
        data = [asdict(label) for label in labels]

        async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2))

    async def read_labels_metadata_json_to_db(
        self, json_path: Optional[str] = None, overwrite_existing: bool = False
    ) -> None:
        """
        Load labels from a JSON file and insert into database if they don't exist.

        Args:
            json_path (str): Optional override path. Default: <db_dir>/labels.json
            overwrite_existing (bool): If True, delete existing labels first.
        """
        json_path = json_path or os.path.join(os.path.dirname(self._db_path), "labels.json")

        try:
            async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
                raw = await f.read()
                label_list = json.loads(raw)

            async with aiosqlite.connect(self._db_path) as db:
                if overwrite_existing:
                    await db.execute("DELETE FROM bbox_tags")
                    await db.execute("DELETE FROM labels")

                for label in label_list:
                    # Insert or replace based on uuid
                    await db.execute(
                        """
                        INSERT OR REPLACE INTO labels (uuid, name, color)
                        VALUES (?, ?, ?)
                        """,
                        (label.get("uuid", str(uuid4()), label["name"], label["color"])),
                    )

                await db.commit()

        except FileNotFoundError:
            logger.warning(f"Label JSON file not found: {json_path}")
        except Exception as e:
            logger.error(f"Failed to load labels from JSON: {e}")

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

        metadata: Optional[ImageMetadata] = await self.read_metadata(image_id)
        if metadata is None:
            logger.warning(f"No metadata found for image: {abs_path}")
            return

        json_path = Path(abs_path).with_suffix(".json")
        os.makedirs(json_path.parent, exist_ok=True)

        async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(asdict(metadata), indent=2))

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

            # Convert parsed dict into ImageMetadata dataclass
            image_meta: ImageMetadata = from_dict(data_class=ImageMetadata, data=parsed)
            image_meta.id = image_id
            await self.write_metadata_to_db(image_id, image_meta)
            return image_meta
        except FileNotFoundError:
            logger.warning(f"JSON file not found: {json_path}")
            return None
