import json
import uuid
from pathlib import Path
from typing import List, Optional, Union

import aiosqlite


class DbClient:
    def __init__(self, db_path: Union[str, Path]):
        self._db_path = str(Path(db_path))

    async def init_db(self) -> None:
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.executescript(
                """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS image_labels (
                image_id INTEGER NOT NULL,
                label_id INTEGER NOT NULL,
                type TEXT CHECK(type IN ('present', 'absent')) DEFAULT 'present',
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY(label_id) REFERENCES labels(id) ON DELETE CASCADE,
                PRIMARY KEY(image_id, label_id, type)
            );
            CREATE TABLE IF NOT EXISTS boxes (
                id INTEGER PRIMARY KEY,
                image_id INTEGER NOT NULL,
                x REAL, y REAL, width REAL, height REAL,
                metadata TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS box_labels (
                box_id INTEGER NOT NULL,
                label_id INTEGER NOT NULL,
                FOREIGN KEY(box_id) REFERENCES boxes(id) ON DELETE CASCADE,
                FOREIGN KEY(label_id) REFERENCES labels(id) ON DELETE CASCADE,
                PRIMARY KEY(box_id, label_id)
            );
            """
            )
            # Check for 'color' column
            cur = await conn.execute("PRAGMA table_info(labels)")
            columns = [row[1] for row in await cur.fetchall()]
            if "color" not in columns:
                await conn.execute("ALTER TABLE labels ADD COLUMN color TEXT DEFAULT '#ff0033'")
            if "uuid" not in columns:
                await conn.execute("ALTER TABLE labels ADD COLUMN uuid TEXT DEFAULT ''")
            await conn.commit()

    async def add_image(self, filename: str) -> int:
        async with aiosqlite.connect(self._db_path) as conn:
            cur = await conn.execute("INSERT OR IGNORE INTO images (filename) VALUES (?)", (filename,))
            await conn.commit()
            cur = await conn.execute("SELECT id FROM images WHERE filename = ?", (filename,))
            row = await cur.fetchone()
            return row["id"]

    async def add_label(self, name: str, color: str) -> dict:
        label_uuid = str(uuid.uuid4())
        async with aiosqlite.connect(self._db_path) as conn:
            try:
                await conn.execute(
                    "INSERT INTO labels (name, color, uuid) VALUES (?, ?, ?)", (name, color, label_uuid)
                )
                await conn.commit()
            except Exception:
                pass  # Ignore if already exists
            conn.row_factory = aiosqlite.Row
            cur = await conn.execute("SELECT id, name, color, uuid FROM labels WHERE name = ?", (name,))
            row = await cur.fetchone()
            return {k: row[k] for k in row.keys()} if row else None

    async def list_labels(self) -> List[aiosqlite.Row]:
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = aiosqlite.Row  # <-- Add this line
            cur = await conn.execute("SELECT id, name, color, uuid FROM labels ORDER BY name ASC")
            rows = await cur.fetchall()
            return rows

    async def label_image(self, image_id: int, label: str, label_type: str = "present") -> None:
        label_id = await self.add_label(label)
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO image_labels (image_id, label_id, type) VALUES (?, ?, ?)",
                (image_id, label_id, label_type),
            )
            await conn.commit()

    async def add_box(
        self, image_id: int, x: float, y: float, width: float, height: float, metadata: Optional[dict] = None
    ) -> int:
        metadata_json = json.dumps(metadata or {})
        async with aiosqlite.connect(self._db_path) as conn:
            cur = await conn.execute(
                """
                INSERT INTO boxes (image_id, x, y, width, height, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (image_id, x, y, width, height, metadata_json),
            )
            await conn.commit()
            return cur.lastrowid

    async def update_box(
        self, box_id: int, x: float, y: float, width: float, height: float, metadata: Optional[dict] = None
    ) -> None:
        metadata_json = json.dumps(metadata or {})
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute(
                """
                UPDATE boxes
                SET x = ?, y = ?, width = ?, height = ?, metadata = ?, last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (x, y, width, height, metadata_json, box_id),
            )
            await conn.commit()

    async def delete_box(self, box_id: int) -> None:
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute("DELETE FROM boxes WHERE id = ?", (box_id,))
            await conn.commit()

    async def label_box(self, box_id: int, label: str) -> None:
        label_id = await self.add_label(label)
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute("INSERT OR IGNORE INTO box_labels (box_id, label_id) VALUES (?, ?)", (box_id, label_id))
            await conn.commit()

    async def get_images_by_label(self, label: str, label_type: str = "present") -> List[aiosqlite.Row]:
        async with aiosqlite.connect(self._db_path) as conn:
            cur = await conn.execute(
                """
                SELECT i.* FROM images i
                JOIN image_labels il ON il.image_id = i.id
                JOIN labels l ON l.id = il.label_id
                WHERE l.name = ? AND il.type = ?
                """,
                (label, label_type),
            )
            rows = await cur.fetchall()
            return rows

    async def get_boxes_by_label(self, label: str) -> List[aiosqlite.Row]:
        async with aiosqlite.connect(self._db_path) as conn:
            cur = await conn.execute(
                """
                SELECT b.* FROM boxes b
                JOIN box_labels bl ON bl.box_id = b.id
                JOIN labels l ON l.id = bl.label_id
                WHERE l.name = ?
                """,
                (label,),
            )
            rows = await cur.fetchall()
            return rows
