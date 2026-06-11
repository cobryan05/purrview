"""In-memory BK-tree index for perceptual-hash similarity search.

All hash entries are stored as (phash_hex, uuid) tuples so that individual
image/bbox records are uniquely addressable even when two entries share the same
pHash.  Deleted entries are handled via a tombstone set (fast filter-at-query-
time) with a lazy rebuild triggered when the dead ratio crosses REBUILD_THRESHOLD.

Rebuild scheduling note
-----------------------
When the tombstone ratio exceeds the threshold we set ``_needs_rebuild = True``
instead of spawning a concurrent asyncio task.  Spawning a task during a
destructive DB operation (e.g. delete) would open a second SQLAlchemy session
while the first is still active; with the StaticPool used in tests (and
potentially under high concurrency in production) the two sessions share one
underlying connection, causing the first session's transaction to be clobbered.
Instead, callers that want to opportunistically clean up dead entries should
call ``rebuild_if_needed()`` from a safe context (e.g. the periodic worker or
the explicit rebuild endpoint).
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pybktree

from apps.helpers.db.db_client import DbClient
from apps.helpers.hashUtils import hamming

logger = logging.getLogger(__name__)


def _item_distance(a: Tuple[str, str], b: Tuple[str, str]) -> int:
    """Hamming distance between two (phash_hex, uuid) items — compares phash only."""
    return hamming(a[0], b[0])


@dataclass
class BuildStats:
    image_count: int = 0
    bbox_count: int = 0
    duration_ms: float = 0.0


@dataclass
class RebuildStats:
    image_count: int = 0
    bbox_count: int = 0
    duration_ms: float = 0.0


class HashIndex:
    """BK-tree index for whole-image and per-label bbox perceptual-hash lookup."""

    REBUILD_THRESHOLD = 0.15  # rebuild when dead entries exceed 15% of tree size
    CHUNK_SIZE = 500  # rows per page when loading phashes from the DB

    def __init__(
        self,
        db_client: DbClient,
        image_hash_size: int = 8,
        bbox_hash_size: int = 16,
    ) -> None:
        self._db_client = db_client
        self._image_hash_size = image_hash_size
        self._bbox_hash_size = bbox_hash_size

        # Image tree: items are (phash_hex, image_uuid) tuples.
        self._image_tree: Optional[pybktree.BKTree] = None
        self._valid_images: Set[str] = set()
        self._image_tombstones: int = 0

        # Per-label bbox trees: label_uuid → BKTree of (phash_hex, bbox_uuid) items.
        self._bbox_trees: Dict[str, pybktree.BKTree] = {}
        self._valid_bboxes: Dict[str, Set[str]] = {}   # label_uuid → set of bbox uuids
        self._bbox_tombstones: Dict[str, int] = {}

        # Set to True when tombstone ratio crosses the threshold.  Callers should
        # invoke rebuild_if_needed() from a safe context rather than relying on
        # an automatic concurrent task.
        self._needs_rebuild: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Build / rebuild
    # ─────────────────────────────────────────────────────────────────────────

    async def _paginate(self, fetch_fn) -> List[Dict[str, str]]:
        """Page through a db fetch function, yielding to the event loop between full pages."""
        rows: List[Dict[str, str]] = []
        offset = 0
        while True:
            chunk = await fetch_fn(offset, self.CHUNK_SIZE)
            rows.extend(chunk)
            if len(chunk) < self.CHUNK_SIZE:
                break
            offset += self.CHUNK_SIZE
            await asyncio.sleep(0)
        return rows

    async def _load_phashes(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Fetch all image and bbox phash rows in pages, yielding to the event loop
        between chunks so other requests are not starved during startup or rebuild."""
        image_rows, bbox_rows = await asyncio.gather(
            self._paginate(self._db_client.get_image_phashes_page),
            self._paginate(self._db_client.get_bbox_phashes_page),
        )
        return image_rows, bbox_rows

    async def build(self, on_phase=None) -> BuildStats:
        """Load all non-null phash rows from the DB and build BK-trees."""
        t0 = time.perf_counter()
        if on_phase:
            on_phase("loading")
        image_rows, bbox_rows = await self._load_phashes()
        if on_phase:
            on_phase("constructing")
        stats = await asyncio.to_thread(self._construct_trees, image_rows, bbox_rows)
        stats.duration_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "HashIndex built: %d images, %d bboxes in %.1f ms",
            stats.image_count,
            stats.bbox_count,
            stats.duration_ms,
        )
        return stats

    def _construct_trees(
        self,
        image_rows: List[Dict[str, str]],
        bbox_rows: List[Dict[str, str]],
    ) -> BuildStats:
        """Build internal trees from raw DB row dicts."""
        # Image tree
        image_items = [(row["phash"], row["uuid"]) for row in image_rows]
        if image_items:
            self._image_tree = pybktree.BKTree(_item_distance, image_items)
        else:
            self._image_tree = None
        self._valid_images = {row["uuid"] for row in image_rows}
        self._image_tombstones = 0

        # Per-label bbox trees
        by_label: Dict[str, List[Tuple[str, str]]] = {}
        for row in bbox_rows:
            by_label.setdefault(row["label_uuid"], []).append((row["phash"], row["uuid"]))

        self._bbox_trees = {}
        self._valid_bboxes = {}
        self._bbox_tombstones = {}
        for label_uuid, items in by_label.items():
            self._bbox_trees[label_uuid] = pybktree.BKTree(_item_distance, items)
            self._valid_bboxes[label_uuid] = {item[1] for item in items}
            self._bbox_tombstones[label_uuid] = 0

        self._needs_rebuild = False
        return BuildStats(image_count=len(image_rows), bbox_count=len(bbox_rows))

    async def rebuild(self) -> RebuildStats:
        """Full reload from DB — resets all tombstone counters."""
        t0 = time.perf_counter()
        image_rows, bbox_rows = await self._load_phashes()
        stats_build = await asyncio.to_thread(self._construct_trees, image_rows, bbox_rows)
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "HashIndex rebuilt: %d images, %d bboxes in %.1f ms",
            stats_build.image_count,
            stats_build.bbox_count,
            duration_ms,
        )
        return RebuildStats(
            image_count=stats_build.image_count,
            bbox_count=stats_build.bbox_count,
            duration_ms=duration_ms,
        )

    async def rebuild_if_needed(self) -> bool:
        """Rebuild the index if the tombstone ratio has exceeded the threshold.

        Returns True if a rebuild was performed.  Call this from a context where
        no other DB session is open (e.g. the periodic worker task or immediately
        after an explicit user request).
        """
        if not self._needs_rebuild:
            return False
        await self.rebuild()
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Mutations
    # ─────────────────────────────────────────────────────────────────────────

    def add_image(self, uuid: str, phash: str) -> None:
        """Insert an image entry into the index."""
        item = (phash, uuid)
        if self._image_tree is None:
            self._image_tree = pybktree.BKTree(_item_distance, [item])
        else:
            self._image_tree.add(item)
        self._valid_images.add(uuid)

    def add_bbox(self, uuid: str, label_uuid: str, phash: str) -> None:
        """Insert a bbox entry into the per-label index."""
        item = (phash, uuid)
        if label_uuid not in self._bbox_trees:
            self._bbox_trees[label_uuid] = pybktree.BKTree(_item_distance, [item])
            self._valid_bboxes[label_uuid] = {uuid}
            self._bbox_tombstones[label_uuid] = 0
        else:
            self._bbox_trees[label_uuid].add(item)
            self._valid_bboxes[label_uuid].add(uuid)

    def remove_image(self, uuid: str) -> None:
        """Tombstone an image entry (lazy eviction from the tree)."""
        self._valid_images.discard(uuid)
        self._image_tombstones += 1
        self._check_image_tombstone_ratio()

    def remove_bbox(self, uuid: str, label_uuid: str) -> None:
        """Tombstone a bbox entry in the per-label tree."""
        valid = self._valid_bboxes.get(label_uuid)
        if valid is not None:
            valid.discard(uuid)
        self._bbox_tombstones[label_uuid] = self._bbox_tombstones.get(label_uuid, 0) + 1
        self._check_bbox_tombstone_ratio(label_uuid)

    def move_bbox(self, uuid: str, old_label_uuid: str, new_label_uuid: str, phash: str) -> None:
        """Move a bbox entry from one label tree to another (called on label change)."""
        self.remove_bbox(uuid, old_label_uuid)
        self.add_bbox(uuid, new_label_uuid, phash)

    # ─────────────────────────────────────────────────────────────────────────
    # Queries
    # ─────────────────────────────────────────────────────────────────────────

    def find_similar_images(self, phash: str, threshold: int) -> List[Tuple[str, int]]:
        """Return [(image_uuid, distance)] for images within *threshold* of *phash*.

        Dead (tombstoned) entries are filtered via *_valid_images*.
        """
        if self._image_tree is None:
            return []
        query = (phash, "")
        try:
            raw = self._image_tree.find(query, threshold)
        except Exception:
            logger.exception("HashIndex.find_similar_images error")
            return []
        return [
            (item[1], dist)
            for dist, item in raw
            if item[1] in self._valid_images
        ]

    def find_similar_bboxes(
        self,
        phash: str,
        label_uuid: Optional[str],
        threshold: int,
    ) -> List[Tuple[str, int, str]]:
        """Return [(bbox_uuid, distance, label_uuid)] within *threshold* of *phash*.

        If *label_uuid* is None, searches all per-label trees.
        """
        query = (phash, "")
        results: List[Tuple[str, int, str]] = []

        target_labels = [label_uuid] if label_uuid else list(self._bbox_trees.keys())
        for lbl in target_labels:
            tree = self._bbox_trees.get(lbl)
            if tree is None:
                continue
            valid = self._valid_bboxes.get(lbl, set())
            try:
                raw = tree.find(query, threshold)
            except Exception:
                logger.exception("HashIndex.find_similar_bboxes error for label %s", lbl)
                continue
            for dist, item in raw:
                if item[1] in valid:
                    results.append((item[1], dist, lbl))
        return results

    def check_dedup_bulk(
        self,
        image_phash: str,
        image_threshold: int,
        bboxes: List[Dict],
    ) -> Dict:
        """Check image + all bbox phashes in one call.

        Args:
            bboxes: list of dicts with keys {phash, label_uuid, threshold}

        Returns:
            {image_is_dup: bool, bbox_flags: [bool]}
            bbox_flags[i] is True when BOTH the image AND bbox i are duplicates.
        """
        image_is_dup = bool(self.find_similar_images(image_phash, image_threshold))
        bbox_flags = [
            image_is_dup and bool(
                self.find_similar_bboxes(b["phash"], b["label_uuid"], b["threshold"])
            )
            for b in bboxes
        ]
        return {"image_is_dup": image_is_dup, "bbox_flags": bbox_flags}

    # ─────────────────────────────────────────────────────────────────────────
    # Stats & tombstone checks
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return diagnostic stats for the rebuild endpoint response."""
        image_tree_size = len(self._valid_images) + self._image_tombstones
        bbox_sizes = {
            lbl: len(self._valid_bboxes.get(lbl, set())) + self._bbox_tombstones.get(lbl, 0)
            for lbl in self._bbox_trees
        }
        return {
            "image_tree_size": image_tree_size,
            "image_valid": len(self._valid_images),
            "image_tombstones": self._image_tombstones,
            "bbox_tree_sizes_by_label": bbox_sizes,
            "bbox_tombstones_by_label": dict(self._bbox_tombstones),
            "needs_rebuild": self._needs_rebuild,
        }

    def _check_image_tombstone_ratio(self) -> None:
        total = len(self._valid_images) + self._image_tombstones
        if total > 0 and self._image_tombstones / total > self.REBUILD_THRESHOLD:
            logger.info(
                "HashIndex image tombstone ratio %.2f exceeded threshold — rebuild deferred",
                self._image_tombstones / total,
            )
            self._needs_rebuild = True

    def _check_bbox_tombstone_ratio(self, label_uuid: str) -> None:
        dead = self._bbox_tombstones.get(label_uuid, 0)
        alive = len(self._valid_bboxes.get(label_uuid, set()))
        total = alive + dead
        if total > 0 and dead / total > self.REBUILD_THRESHOLD:
            logger.info(
                "HashIndex bbox tombstone ratio %.2f for label %s — rebuild deferred",
                dead / total,
                label_uuid,
            )
            self._needs_rebuild = True
