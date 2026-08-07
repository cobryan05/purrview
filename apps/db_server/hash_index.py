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

import numpy as np
import pybktree

from apps.helpers.db.db_client import DbClient
from apps.helpers.hashUtils import hamming

# ── Byte-level popcount LUT (always available) ────────────────────────────────
# Used directly for arbitrary-width hash arrays in rank_images_by_diversity,
# and as the fallback inside _popcount for older numpy.
_LUT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# ── Vectorised popcount for uint64 numpy arrays ────────────────────────────────
# np.bitwise_count was added in numpy 2.0; fall back to the byte-LUT for older
# versions so we don't require a pinned numpy version.
try:
    _popcount = np.bitwise_count  # type: ignore[attr-defined]  # numpy >= 2.0
except AttributeError:

    def _popcount(arr_u64: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        """Return per-element popcount for a uint64 array using a byte look-up table."""
        return _LUT8[arr_u64.view(np.uint8)].reshape(arr_u64.shape + (8,)).sum(axis=-1).astype(np.int32)


logger = logging.getLogger(__name__)


def _item_distance(a: Tuple[str, str], b: Tuple[str, str]) -> int:
    """Hamming distance between two (phash_hex, uuid) items — compares phash only."""
    return hamming(a[0], b[0])


class _FlatHashTable:
    """Flat numpy mirror of one BK-tree, used only for *nearest-distance* queries.

    The BK-tree answers "is anything within R?" quickly for small R, but the
    calibration stats need the distance to the nearest neighbour whatever it is,
    and a BK-tree search degenerates towards a full traversal as R grows — with
    far worse constants than a straight vectorised scan.  So nearest-distance
    queries run against a contiguous ``(n, byte_len)`` uint8 array: one XOR plus
    one LUT popcount over every row, no Python-level loop, sub-millisecond for
    100k rows.

    Kept deliberately dumb: it mirrors the tree's *membership*, so every mutation
    point on HashIndex must update both.  Deletions tombstone via ``_alive``
    exactly like the tree's valid-set, so the two structures stay in agreement.

    Re-adding a uuid with a different phash leaves the old row alive, which
    matches the BK-tree's behaviour (it also keeps the superseded item, and the
    uuid stays in the valid-set).  Both structures therefore report the stale
    entry; they do not disagree with each other.
    """

    _INITIAL_CAPACITY = 256

    def __init__(self, byte_len: int) -> None:
        self._byte_len = byte_len
        self._rows = np.zeros((self._INITIAL_CAPACITY, byte_len), dtype=np.uint8)
        self._alive = np.zeros(self._INITIAL_CAPACITY, dtype=bool)
        self._uuids: List[str] = []
        self._rows_of: Dict[str, List[int]] = {}
        self._n = 0

    @classmethod
    def from_items(cls, items: List[Tuple[str, str]], byte_len: int) -> "_FlatHashTable":
        """Build from [(phash_hex, uuid)] in one shot — used by _build_structures.

        Rows whose hash width doesn't match *byte_len* are skipped rather than
        raising: a DB carrying phashes from a previous hash-size setting must not
        take the whole index build down with it (the BK-tree tolerates them too,
        comparing hex-as-int). Such rows simply don't participate in
        nearest-distance queries until the phashes are recomputed.
        """
        table = cls(byte_len)
        usable = [(phash, uuid) for phash, uuid in items if len(phash) == byte_len * 2]
        if len(usable) != len(items):
            logger.warning(
                "HashIndex: skipping %d phash row(s) with unexpected width (expected %d hex chars) — "
                "recompute phashes to include them in duplicate-distance queries",
                len(items) - len(usable),
                byte_len * 2,
            )
        items = usable
        if not items:
            return table
        n = len(items)
        table._rows = np.frombuffer(
            b"".join(bytes.fromhex(phash) for phash, _ in items),
            dtype=np.uint8,
        ).reshape(n, byte_len)
        # np.frombuffer returns a read-only view over the joined bytes; _grow()
        # and per-row writes need a writable array.
        table._rows = table._rows.copy()
        table._alive = np.ones(n, dtype=bool)
        table._uuids = [uuid for _, uuid in items]
        for i, (_, uuid) in enumerate(items):
            table._rows_of.setdefault(uuid, []).append(i)
        table._n = n
        return table

    def _grow(self) -> None:
        capacity = max(self._INITIAL_CAPACITY, self._rows.shape[0] * 2)
        rows = np.zeros((capacity, self._byte_len), dtype=np.uint8)
        rows[: self._n] = self._rows[: self._n]
        alive = np.zeros(capacity, dtype=bool)
        alive[: self._n] = self._alive[: self._n]
        self._rows = rows
        self._alive = alive

    def add(self, uuid: str, phash_hex: str) -> None:
        if self._n >= self._rows.shape[0]:
            self._grow()
        self._rows[self._n] = np.frombuffer(bytes.fromhex(phash_hex), dtype=np.uint8)
        self._alive[self._n] = True
        self._uuids.append(uuid)
        self._rows_of.setdefault(uuid, []).append(self._n)
        self._n += 1

    def discard(self, uuid: str) -> None:
        """Tombstone every row for *uuid* (mirrors valid-set ``discard``)."""
        for row in self._rows_of.pop(uuid, ()):
            self._alive[row] = False

    def nearest(self, phash_hex: str) -> Optional[Tuple[int, str]]:
        """Return ``(distance, uuid)`` for the closest live row, or None if empty."""
        if self._n == 0 or not self._alive[: self._n].any():
            return None
        query = np.frombuffer(bytes.fromhex(phash_hex), dtype=np.uint8)
        dist = _LUT8[self._rows[: self._n] ^ query].sum(axis=1).astype(np.int32)
        # Dead rows must never win the argmin; push them past any achievable
        # distance (byte_len * 8 bits is the maximum).
        dist[~self._alive[: self._n]] = self._byte_len * 8 + 1
        best = int(np.argmin(dist))
        return int(dist[best]), self._uuids[best]


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


@dataclass
class _IndexData:
    """Fully-built index structures, produced off the event loop by
    _build_structures and swapped into the live HashIndex atomically by
    _install. Kept separate from the live fields so a worker thread never
    mutates state that event-loop queries are concurrently reading."""

    image_tree: Optional["pybktree.BKTree"]
    valid_images: Set[str]
    bbox_trees: Dict[str, "pybktree.BKTree"]
    valid_bboxes: Dict[str, Set[str]]
    bbox_tombstones: Dict[str, int]
    image_count: int
    bbox_count: int
    image_flat: "_FlatHashTable"
    bbox_flat: Dict[str, "_FlatHashTable"]


class HashIndex:
    """BK-tree index for whole-image and per-label bbox perceptual-hash lookup."""

    REBUILD_THRESHOLD = 0.15  # rebuild when dead entries exceed 15% of tree size
    CHUNK_SIZE = 500  # rows per page when loading phashes from the DB
    MAX_DIVERSITY_CANDIDATES = 100_000  # _greedy_diversity_order is documented fast up to this size

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
        self._valid_bboxes: Dict[str, Set[str]] = {}  # label_uuid → set of bbox uuids
        self._bbox_tombstones: Dict[str, int] = {}

        # Flat numpy mirrors of the trees above, for nearest-distance queries
        # (see _FlatHashTable). Every mutation below updates tree and mirror.
        self._image_flat = _FlatHashTable(self._image_byte_len)
        self._bbox_flat: Dict[str, _FlatHashTable] = {}

        # Set to True when tombstone ratio crosses the threshold.  Callers should
        # invoke rebuild_if_needed() from a safe context rather than relying on
        # an automatic concurrent task.
        self._needs_rebuild: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Build / rebuild
    # ─────────────────────────────────────────────────────────────────────────

    async def _paginate(self, fetch_fn, on_chunk=None) -> List[Dict[str, str]]:
        """Page through a db fetch function, yielding to the event loop between full pages."""
        rows: List[Dict[str, str]] = []
        offset = 0
        while True:
            chunk = await fetch_fn(offset, self.CHUNK_SIZE)
            rows.extend(chunk)
            if on_chunk:
                on_chunk(len(rows))
            if len(chunk) < self.CHUNK_SIZE:
                break
            offset += self.CHUNK_SIZE
            await asyncio.sleep(0)
        return rows

    async def _load_phashes(
        self, on_img_chunk=None, on_bbox_chunk=None
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Fetch all image and bbox phash rows in pages, yielding to the event loop
        between chunks so other requests are not starved during startup or rebuild."""
        image_rows, bbox_rows = await asyncio.gather(
            self._paginate(self._db_client.get_image_phashes_page, on_chunk=on_img_chunk),
            self._paginate(self._db_client.get_bbox_phashes_page, on_chunk=on_bbox_chunk),
        )
        return image_rows, bbox_rows

    async def build(self, on_phase=None, on_progress=None) -> BuildStats:
        """Load all non-null phash rows from the DB and build BK-trees.

        on_progress, if provided, is called as on_progress(images_loaded, bboxes_loaded)
        after each chunk arrives so callers can report live counts.
        """
        t0 = time.perf_counter()

        if on_phase:
            on_phase("loading")

        loaded_images = loaded_bboxes = 0

        def _img_chunk(n: int) -> None:
            nonlocal loaded_images
            loaded_images = n
            if on_progress:
                on_progress(loaded_images, loaded_bboxes)

        def _bbox_chunk(n: int) -> None:
            nonlocal loaded_bboxes
            loaded_bboxes = n
            if on_progress:
                on_progress(loaded_images, loaded_bboxes)

        if on_progress:
            image_rows, bbox_rows = await self._load_phashes(
                on_img_chunk=_img_chunk,
                on_bbox_chunk=_bbox_chunk,
            )
        else:
            image_rows, bbox_rows = await self._load_phashes()
        if on_phase:
            on_phase("constructing")
        # Build off the event loop, then swap in on the loop (no await between
        # the to_thread result and _install → atomic from every reader's view).
        data = await asyncio.to_thread(self._build_structures, image_rows, bbox_rows)
        self._install(data)
        stats = BuildStats(image_count=data.image_count, bbox_count=data.bbox_count)
        stats.duration_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "HashIndex built: %d images, %d bboxes in %.1f ms",
            stats.image_count,
            stats.bbox_count,
            stats.duration_ms,
        )
        return stats

    def _build_structures(
        self,
        image_rows: List[Dict[str, str]],
        bbox_rows: List[Dict[str, str]],
    ) -> _IndexData:
        """Build BK-trees from raw DB row dicts into a fresh _IndexData.

        Pure with respect to ``self`` — it must NOT assign to any live field.
        Runs in a worker thread (via asyncio.to_thread); the live trees stay
        readable by event-loop queries until _install swaps in the result.
        """
        # Image tree
        image_items = [(row["phash"], row["uuid"]) for row in image_rows]
        image_tree = pybktree.BKTree(_item_distance, image_items) if image_items else None
        valid_images = {row["uuid"] for row in image_rows}

        # Per-label bbox trees
        by_label: Dict[str, List[Tuple[str, str]]] = {}
        for row in bbox_rows:
            by_label.setdefault(row["label_uuid"], []).append((row["phash"], row["uuid"]))

        bbox_trees: Dict[str, pybktree.BKTree] = {}
        valid_bboxes: Dict[str, Set[str]] = {}
        bbox_tombstones: Dict[str, int] = {}
        bbox_flat: Dict[str, _FlatHashTable] = {}
        for label_uuid, items in by_label.items():
            bbox_trees[label_uuid] = pybktree.BKTree(_item_distance, items)
            valid_bboxes[label_uuid] = {item[1] for item in items}
            bbox_tombstones[label_uuid] = 0
            bbox_flat[label_uuid] = _FlatHashTable.from_items(items, self._bbox_byte_len)

        return _IndexData(
            image_tree=image_tree,
            valid_images=valid_images,
            bbox_trees=bbox_trees,
            valid_bboxes=valid_bboxes,
            bbox_tombstones=bbox_tombstones,
            image_count=len(image_rows),
            bbox_count=len(bbox_rows),
            image_flat=_FlatHashTable.from_items(image_items, self._image_byte_len),
            bbox_flat=bbox_flat,
        )

    def _install(self, data: _IndexData) -> None:
        """Atomically swap freshly-built structures into the live index.

        Called on the event loop with NO intervening await, so from the point of
        view of any other coroutine the whole index flips over in one step —
        readers never observe a partially-rebuilt tree. Resets tombstone
        counters since the new trees contain only live entries.
        """
        self._image_tree = data.image_tree
        self._valid_images = data.valid_images
        self._image_tombstones = 0
        self._bbox_trees = data.bbox_trees
        self._valid_bboxes = data.valid_bboxes
        self._bbox_tombstones = data.bbox_tombstones
        self._image_flat = data.image_flat
        self._bbox_flat = data.bbox_flat
        self._needs_rebuild = False

    async def rebuild(self) -> RebuildStats:
        """Full reload from DB — resets all tombstone counters."""
        t0 = time.perf_counter()
        image_rows, bbox_rows = await self._load_phashes()
        # Build off the loop, swap in atomically on the loop (see build()).
        data = await asyncio.to_thread(self._build_structures, image_rows, bbox_rows)
        self._install(data)
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "HashIndex rebuilt: %d images, %d bboxes in %.1f ms",
            data.image_count,
            data.bbox_count,
            duration_ms,
        )
        return RebuildStats(
            image_count=data.image_count,
            bbox_count=data.bbox_count,
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
        self._validate_image_phash(phash, context=f"uuid={uuid}")
        item = (phash, uuid)
        if self._image_tree is None:
            self._image_tree = pybktree.BKTree(_item_distance, [item])
        else:
            self._image_tree.add(item)
        self._valid_images.add(uuid)
        self._image_flat.add(uuid, phash)

    def add_bbox(self, uuid: str, label_uuid: str, phash: str) -> None:
        """Insert a bbox entry into the per-label index."""
        self._validate_bbox_phash(phash, context=f"uuid={uuid}")
        item = (phash, uuid)
        if label_uuid not in self._bbox_trees:
            self._bbox_trees[label_uuid] = pybktree.BKTree(_item_distance, [item])
            self._valid_bboxes[label_uuid] = {uuid}
            self._bbox_tombstones[label_uuid] = 0
        else:
            self._bbox_trees[label_uuid].add(item)
            self._valid_bboxes[label_uuid].add(uuid)
        self._bbox_flat.setdefault(label_uuid, _FlatHashTable(self._bbox_byte_len)).add(uuid, phash)

    def remove_image(self, uuid: str) -> None:
        """Tombstone an image entry (lazy eviction from the tree)."""
        self._valid_images.discard(uuid)
        self._image_flat.discard(uuid)
        self._image_tombstones += 1
        self._check_image_tombstone_ratio()

    def remove_bbox(self, uuid: str, label_uuid: str) -> None:
        """Tombstone a bbox entry in the per-label tree."""
        valid = self._valid_bboxes.get(label_uuid)
        if valid is not None:
            valid.discard(uuid)
        flat = self._bbox_flat.get(label_uuid)
        if flat is not None:
            flat.discard(uuid)
        self._bbox_tombstones[label_uuid] = self._bbox_tombstones.get(label_uuid, 0) + 1
        self._check_bbox_tombstone_ratio(label_uuid)

    def move_bbox(self, uuid: str, old_label_uuid: str, new_label_uuid: str, phash: str) -> None:
        """Move a bbox entry from one label tree to another (called on label change)."""
        self.remove_bbox(uuid, old_label_uuid)
        self.add_bbox(uuid, new_label_uuid, phash)

    def pick_any_image_phash(self, candidate_uuids: Optional[Set[str]] = None) -> Optional[str]:
        """Return the phash of any valid (candidate-filtered) image entry, or None if empty."""
        if self._image_tree is None:
            return None
        for phash_hex, uuid in self._image_tree:
            if uuid not in self._valid_images:
                continue
            if candidate_uuids is not None and uuid not in candidate_uuids:
                continue
            return phash_hex
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Queries
    # ─────────────────────────────────────────────────────────────────────────

    def find_similar_images(self, phash: str, threshold: int) -> List[Tuple[str, int]]:
        """Return [(image_uuid, distance)] for images within *threshold* of *phash*, sorted by distance ascending.

        Dead (tombstoned) entries are filtered via *_valid_images*.
        Raises ValueError if *phash* doesn't match the current image hash size.
        """
        self._validate_image_phash(phash)
        if self._image_tree is None:
            return []
        query = (phash, "")
        try:
            raw = self._image_tree.find(query, threshold)
        except Exception:
            logger.exception("HashIndex.find_similar_images error")
            return []
        results = [(item[1], dist) for dist, item in raw if item[1] in self._valid_images]
        results.sort(key=lambda x: x[1])
        return results

    def find_similar_bboxes(
        self,
        phash: str,
        label_uuids: Optional[List[str]],
        threshold: int,
    ) -> List[Tuple[str, int, str]]:
        """Return [(bbox_uuid, distance, label_uuid)] within *threshold* of *phash*, sorted by distance ascending.

        If *label_uuids* is None or empty, searches all per-label trees.
        Raises ValueError if *phash* doesn't match the current bbox hash size.
        """
        self._validate_bbox_phash(phash)
        query = (phash, "")
        results: List[Tuple[str, int, str]] = []

        target_labels = label_uuids if label_uuids else list(self._bbox_trees.keys())
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
        results.sort(key=lambda x: x[1])
        return results

    def nearest_image_distance(self, phash: str) -> Optional[int]:
        """Hamming distance to the closest stored image, or None if the index is empty.

        Unbounded — unlike find_similar_images this has no threshold, which is
        exactly what threshold calibration needs: the distance is a property of
        the data, independent of whatever threshold happens to be configured.
        """
        self._validate_image_phash(phash)
        result = self._image_flat.nearest(phash)
        return result[0] if result else None

    def nearest_bbox_distance(self, phash: str, label_uuid: Optional[str]) -> Optional[int]:
        """Hamming distance to the closest stored bbox with the same label, or None.

        A bbox crop is only meaningfully comparable to crops of the same label, so
        unlike find_similar_bboxes this never falls back to scanning every label
        tree — an unlabelled query has nothing to compare against and returns None.
        """
        self._validate_bbox_phash(phash)
        if not label_uuid:
            return None
        flat = self._bbox_flat.get(label_uuid)
        if flat is None:
            return None
        result = flat.nearest(phash)
        return result[0] if result else None

    @staticmethod
    def _greedy_diversity_order(phash_hexes: List[str], seed_phash: str, count: int) -> List[int]:
        """Return indices into *phash_hexes*, ordered most-diverse-first.

        Greedy farthest-point sampling: the first pick is the entry most unlike
        *seed_phash*, and each subsequent pick is the entry most unlike *all
        already-selected* entries. This avoids returning a cluster of
        near-duplicates at the far end of a single distance sort.

        Hashes are XORed as raw bytes and popcounted via a byte-level lookup
        table, so this works for any hash width — image hashes and bbox
        hashes share this one implementation rather than diverging because one
        happens to fit in a uint64 and the other doesn't.
        Complexity: O(N × count) vectorised XOR + popcount; fast for N ≤ ~100k.
        """
        if not phash_hexes:
            return []

        byte_len = len(seed_phash) // 2
        phash_bytes = np.frombuffer(
            b"".join(bytes.fromhex(ph) for ph in phash_hexes),
            dtype=np.uint8,
        ).reshape(len(phash_hexes), byte_len)
        seed_bytes = np.frombuffer(bytes.fromhex(seed_phash), dtype=np.uint8)

        # min_dist[i] = Hamming distance from candidate i to the nearest already-selected point.
        # Initialised to distance from the seed so the first argmax picks the farthest from it.
        min_dist: np.ndarray = _LUT8[phash_bytes ^ seed_bytes].sum(axis=1).astype(np.int32)

        selected: List[int] = []
        limit = min(count, len(phash_hexes))
        for _ in range(limit):
            j = int(np.argmax(min_dist))
            if min_dist[j] < 0:
                break  # all candidates already selected
            selected.append(j)
            min_dist[j] = -1  # mark as selected so it can't be re-picked
            # Update min_dist: each remaining candidate's distance shrinks if the
            # new pivot is closer than any previously selected point.
            d = _LUT8[phash_bytes ^ phash_bytes[j]].sum(axis=1).astype(np.int32)
            min_dist = np.where(min_dist >= 0, np.minimum(min_dist, d), min_dist)

        return selected

    async def rank_images_by_diversity(
        self,
        seed_phash: str,
        candidate_uuids: Optional[Set[str]],
        count: int,
    ) -> List[str]:
        """Return up to *count* image UUIDs ordered most-diverse first.

        See `_greedy_diversity_order` for the ranking algorithm.
        If *candidate_uuids* is given, only those images participate.
        Raises ValueError if *seed_phash* doesn't match the current image hash size.
        """
        self._validate_image_phash(seed_phash)
        if self._image_tree is None:
            return []

        # Materialise phash hex strings from the tree ON THE EVENT LOOP (this
        # method is a coroutine; the loop below has no await, so no concurrent
        # add_image/add_bbox can mutate the tree mid-iteration). Only the
        # CPU-bound numpy ranking is offloaded to a worker thread — it operates
        # on the local snapshot, never on the live tree.
        uuids: List[str] = []
        phash_hexes: List[str] = []
        for phash_hex, uuid in self._image_tree:
            if uuid not in self._valid_images:
                continue
            if candidate_uuids is not None and uuid not in candidate_uuids:
                continue
            uuids.append(uuid)
            phash_hexes.append(phash_hex)

        order = await asyncio.to_thread(self._greedy_diversity_order, phash_hexes, seed_phash, count)
        return [uuids[i] for i in order]

    async def rank_bboxes_by_diversity(
        self,
        seed_phash: str,
        label_uuids: Optional[List[str]],
        count: int,
    ) -> List[Tuple[str, str]]:
        """Return up to *count* (bbox_uuid, label_uuid) tuples ordered most-diverse first.

        See `_greedy_diversity_order` for the ranking algorithm — shared with
        `rank_images_by_diversity` since the byte-level LUT approach works for
        any hash width, including 256-bit (hash_size 16) bbox phashes.

        If *label_uuids* is None or empty, all per-label trees are searched.
        Raises ValueError if *seed_phash* doesn't match the current bbox hash size.
        """
        self._validate_bbox_phash(seed_phash)

        # Materialise (bbox_uuid, label_uuid, phash_hex) across relevant trees ON
        # THE EVENT LOOP (no await in the loop below), so the CPU-bound thread
        # never iterates a live tree while a concurrent add_bbox mutates it.
        items: List[Tuple[str, str, str]] = []
        target_labels = label_uuids if label_uuids else list(self._bbox_trees.keys())
        for lbl in target_labels:
            tree = self._bbox_trees.get(lbl)
            if tree is None:
                continue
            valid = self._valid_bboxes.get(lbl, set())
            for phash_hex, bbox_uuid in tree:
                if bbox_uuid in valid:
                    items.append((bbox_uuid, lbl, phash_hex))

        order = await asyncio.to_thread(self._greedy_diversity_order, [p for _, _, p in items], seed_phash, count)
        return [(items[i][0], items[i][1]) for i in order]

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
            {image_is_dup: bool, bbox_flags: [bool],
             image_min_distance: int|None, bbox_min_distances: [int|None]}
            image_is_dup: True when the image phash is within image_threshold of any stored image.
            bbox_flags[i]: True when bbox i's phash is within its threshold of any stored bbox with
            the same label — independently of image_is_dup.  Callers that need a combined signal
            (e.g. "tag only when both image and bbox are dups") should AND with image_is_dup themselves.
            *_min_distance(s): threshold-independent nearest-neighbour distances, for threshold
            calibration.  None means "nothing to compare against" (empty index, or an unlabelled
            bbox), which is not the same as "far away" and must not be bucketed as a distance.

        The flags are derived from the same distances rather than from a separate
        BK-tree query, so "is_dup" and "min_distance" can never disagree.
        """
        image_min = self.nearest_image_distance(image_phash)
        image_is_dup = image_min is not None and image_min <= image_threshold

        bbox_flags: List[bool] = []
        bbox_min_distances: List[Optional[int]] = []
        for b in bboxes:
            dist = self.nearest_bbox_distance(b["phash"], b.get("label_uuid"))
            bbox_min_distances.append(dist)
            bbox_flags.append(dist is not None and dist <= b["threshold"])

        return {
            "image_is_dup": image_is_dup,
            "bbox_flags": bbox_flags,
            "image_min_distance": image_min,
            "bbox_min_distances": bbox_min_distances,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Stats & tombstone checks
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return diagnostic stats for the rebuild endpoint response."""
        image_tree_size = len(self._valid_images) + self._image_tombstones
        bbox_sizes = {
            lbl: len(self._valid_bboxes.get(lbl, set())) + self._bbox_tombstones.get(lbl, 0) for lbl in self._bbox_trees
        }
        return {
            "image_tree_size": image_tree_size,
            "image_valid": len(self._valid_images),
            "image_tombstones": self._image_tombstones,
            "bbox_tree_sizes_by_label": bbox_sizes,
            "bbox_tombstones_by_label": dict(self._bbox_tombstones),
            "needs_rebuild": self._needs_rebuild,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Hash-size validation
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def _image_byte_len(self) -> int:
        """Byte width of an image phash — hex length halved."""
        return self._expected_image_hex_len // 2

    @property
    def _bbox_byte_len(self) -> int:
        """Byte width of a bbox phash — hex length halved."""
        return self._expected_bbox_hex_len // 2

    @property
    def _expected_image_hex_len(self) -> int:
        """Expected hex-string length for image phashes: hash_size² bits → hash_size²/4 hex chars."""
        return (self._image_hash_size**2) // 4

    @property
    def _expected_bbox_hex_len(self) -> int:
        """Expected hex-string length for bbox phashes: hash_size² bits → hash_size²/4 hex chars."""
        return (self._bbox_hash_size**2) // 4

    def _validate_image_phash(self, phash: str, context: str = "") -> None:
        """Raise ValueError if *phash* length doesn't match the current image hash size."""
        expected = self._expected_image_hex_len
        if len(phash) != expected:
            raise ValueError(
                f"Image phash length mismatch{' (' + context + ')' if context else ''}: "
                f"got {len(phash)} hex chars, expected {expected} "
                f"(image_hash_size={self._image_hash_size}). "
                "Rebuild the index after changing hash size settings."
            )

    def _validate_bbox_phash(self, phash: str, context: str = "") -> None:
        """Raise ValueError if *phash* length doesn't match the current bbox hash size."""
        expected = self._expected_bbox_hex_len
        if len(phash) != expected:
            raise ValueError(
                f"Bbox phash length mismatch{' (' + context + ')' if context else ''}: "
                f"got {len(phash)} hex chars, expected {expected} "
                f"(bbox_hash_size={self._bbox_hash_size}). "
                "Rebuild the index after changing hash size settings."
            )

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
