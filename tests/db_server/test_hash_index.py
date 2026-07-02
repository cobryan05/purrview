"""Tests for the in-memory BK-tree hash index (apps/db_server/hash_index.py).

Most tests exercise the pure in-memory operations (add/find/remove/move) and
the tombstone / _needs_rebuild flag.  Build-from-DB tests use the real
in-memory db_client fixture from conftest.py.
"""

import numpy as np
import pytest

from apps.db_server.hash_index import HashIndex
from apps.helpers.hashUtils import hamming, phash_from_numpy

# ─────────────────────────────────────────────────────────────────────────────
# Test phash values — computed once at module import time for speed.
# Using distinct colours produces meaningfully different hashes.
# ─────────────────────────────────────────────────────────────────────────────


def _solid(b: int, g: int, r: int, size: int = 64, hash_size: int = 8) -> str:
    arr = np.full((size, size, 3), [b, g, r], dtype=np.uint8)
    return phash_from_numpy(arr, hash_size=hash_size)


IMG_BLACK = _solid(0, 0, 0)
IMG_WHITE = _solid(255, 255, 255)
IMG_GREY = _solid(128, 128, 128)

BBOX_RED = _solid(0, 0, 255, hash_size=16)  # BGR: red


def _checkerboard_phash(hash_size: int = 16) -> str:
    """pHash of a checkerboard — maximum high-frequency content."""
    mask = np.indices((64, 64)).sum(axis=0) % 2
    bgr = (mask[:, :, None] * 255).astype(np.uint8).repeat(3, axis=2)
    return phash_from_numpy(bgr, hash_size=hash_size)


def _gradient_phash(hash_size: int = 16) -> str:
    """pHash of a horizontal gradient — smooth, mostly low-frequency."""
    row = np.linspace(0, 255, 64).astype(np.uint8)
    bgr = np.tile(row, (64, 1))[:, :, None].repeat(3, axis=2)
    return phash_from_numpy(bgr, hash_size=hash_size)


BBOX_CHECKER = _checkerboard_phash()
BBOX_GRAD = _gradient_phash()

# All three image hashes must differ (sanity check)
assert hamming(IMG_BLACK, IMG_WHITE) > 0


@pytest.fixture
def idx(db_client):
    """Empty HashIndex not yet built from DB."""
    return HashIndex(db_client)


# ─────────────────────────────────────────────────────────────────────────────
# Image: add / find / remove
# ─────────────────────────────────────────────────────────────────────────────


class TestImageOperations:
    def test_empty_index_returns_no_matches(self, idx):
        assert idx.find_similar_images(IMG_BLACK, threshold=5) == []

    def test_added_image_found_at_threshold_zero(self, idx):
        idx.add_image("img-1", IMG_BLACK)
        results = idx.find_similar_images(IMG_BLACK, threshold=0)
        assert any(uuid == "img-1" for uuid, _ in results)

    def test_result_contains_uuid_and_distance(self, idx):
        idx.add_image("img-1", IMG_BLACK)
        ((uuid, dist),) = idx.find_similar_images(IMG_BLACK, threshold=0)
        assert uuid == "img-1"
        assert dist == 0

    def test_dissimilar_image_excluded_at_low_threshold(self, idx):
        idx.add_image("img-black", IMG_BLACK)
        idx.add_image("img-white", IMG_WHITE)
        results = idx.find_similar_images(IMG_BLACK, threshold=0)
        uuids = {u for u, _ in results}
        assert "img-black" in uuids
        assert "img-white" not in uuids

    def test_removed_image_not_returned(self, idx):
        idx.add_image("img-1", IMG_BLACK)
        idx.remove_image("img-1")
        results = idx.find_similar_images(IMG_BLACK, threshold=20)
        assert all(uuid != "img-1" for uuid, _ in results)

    def test_remove_nonexistent_image_does_not_raise(self, idx):
        idx.remove_image("ghost-uuid")  # should not raise

    def test_add_multiple_images_all_findable(self, idx):
        idx.add_image("img-black", IMG_BLACK)
        idx.add_image("img-grey", IMG_GREY)
        # Broad threshold to capture both
        results = idx.find_similar_images(IMG_BLACK, threshold=64)
        uuids = {u for u, _ in results}
        assert "img-black" in uuids
        assert "img-grey" in uuids


# ─────────────────────────────────────────────────────────────────────────────
# Bbox: add / find / remove / move
# ─────────────────────────────────────────────────────────────────────────────


class TestBboxOperations:
    def test_empty_index_returns_no_matches(self, idx):
        assert idx.find_similar_bboxes(BBOX_RED, label_uuids=["lbl-1"], threshold=5) == []

    def test_added_bbox_found_at_threshold_zero(self, idx):
        idx.add_bbox("bbox-1", "lbl-1", BBOX_RED)
        results = idx.find_similar_bboxes(BBOX_RED, label_uuids=["lbl-1"], threshold=0)
        assert any(u == "bbox-1" for u, _, _ in results)

    def test_result_contains_uuid_distance_label(self, idx):
        idx.add_bbox("bbox-1", "lbl-1", BBOX_RED)
        ((bbox_uuid, dist, label_uuid),) = idx.find_similar_bboxes(BBOX_RED, label_uuids=["lbl-1"], threshold=0)
        assert bbox_uuid == "bbox-1"
        assert dist == 0
        assert label_uuid == "lbl-1"

    def test_search_restricted_to_specified_label(self, idx):
        idx.add_bbox("bbox-lbl1", "lbl-1", BBOX_RED)
        idx.add_bbox("bbox-lbl2", "lbl-2", BBOX_RED)
        results = idx.find_similar_bboxes(BBOX_RED, label_uuids=["lbl-1"], threshold=0)
        uuids = {u for u, _, _ in results}
        assert "bbox-lbl1" in uuids
        assert "bbox-lbl2" not in uuids

    def test_search_all_labels_when_label_uuid_is_none(self, idx):
        idx.add_bbox("bbox-a", "lbl-1", BBOX_RED)
        idx.add_bbox("bbox-b", "lbl-2", BBOX_RED)
        results = idx.find_similar_bboxes(BBOX_RED, label_uuids=None, threshold=0)
        uuids = {u for u, _, _ in results}
        assert "bbox-a" in uuids
        assert "bbox-b" in uuids

    def test_removed_bbox_not_returned(self, idx):
        idx.add_bbox("bbox-1", "lbl-1", BBOX_RED)
        idx.remove_bbox("bbox-1", "lbl-1")
        results = idx.find_similar_bboxes(BBOX_RED, label_uuids=["lbl-1"], threshold=20)
        assert all(u != "bbox-1" for u, _, _ in results)

    def test_remove_nonexistent_bbox_does_not_raise(self, idx):
        idx.remove_bbox("ghost-uuid", "lbl-1")

    def test_move_bbox_removes_from_old_label(self, idx):
        idx.add_bbox("bbox-1", "lbl-1", BBOX_RED)
        idx.move_bbox("bbox-1", "lbl-1", "lbl-2", BBOX_RED)
        results = idx.find_similar_bboxes(BBOX_RED, label_uuids=["lbl-1"], threshold=0)
        assert all(u != "bbox-1" for u, _, _ in results)

    def test_move_bbox_adds_to_new_label(self, idx):
        idx.add_bbox("bbox-1", "lbl-1", BBOX_RED)
        idx.move_bbox("bbox-1", "lbl-1", "lbl-2", BBOX_RED)
        results = idx.find_similar_bboxes(BBOX_RED, label_uuids=["lbl-2"], threshold=0)
        assert any(u == "bbox-1" for u, _, _ in results)

    def test_dissimilar_bbox_excluded_at_low_threshold(self, idx):
        # Use images with very different frequency content so they have a large
        # pHash distance (solid colours are not enough — they share the same DCT
        # pattern and can hash to within 1 bit of each other).
        idx.add_bbox("bbox-checker", "lbl-1", BBOX_CHECKER)
        idx.add_bbox("bbox-grad", "lbl-1", BBOX_GRAD)
        results = idx.find_similar_bboxes(BBOX_CHECKER, label_uuids=["lbl-1"], threshold=0)
        uuids = {u for u, _, _ in results}
        assert "bbox-checker" in uuids
        assert "bbox-grad" not in uuids


# ─────────────────────────────────────────────────────────────────────────────
# Tombstone → _needs_rebuild flag
# ─────────────────────────────────────────────────────────────────────────────


class TestTombstoneFlag:
    def test_needs_rebuild_false_initially(self, idx):
        assert idx._needs_rebuild is False

    def test_removing_image_above_threshold_sets_flag(self, idx):
        idx.add_image("img-1", IMG_BLACK)
        idx.remove_image("img-1")  # 1/1 = 100% > 15% threshold
        assert idx._needs_rebuild is True

    def test_removing_bbox_above_threshold_sets_flag(self, idx):
        idx.add_bbox("bbox-1", "lbl-1", BBOX_RED)
        idx.remove_bbox("bbox-1", "lbl-1")
        assert idx._needs_rebuild is True

    def test_flag_not_set_when_ratio_below_threshold(self, idx):
        # Add 10 images, remove 1 → 10% dead < 15% threshold
        for i in range(10):
            idx.add_image(f"img-{i}", IMG_GREY)
        idx.remove_image("img-0")
        assert idx._needs_rebuild is False

    async def test_rebuild_if_needed_performs_rebuild_and_clears_flag(self, idx):
        idx.add_image("img-1", IMG_BLACK)
        idx.remove_image("img-1")
        assert idx._needs_rebuild is True
        did = await idx.rebuild_if_needed()
        assert did is True
        assert idx._needs_rebuild is False

    async def test_rebuild_if_needed_no_op_when_flag_clear(self, idx):
        did = await idx.rebuild_if_needed()
        assert did is False


# ─────────────────────────────────────────────────────────────────────────────
# Build / rebuild from DB
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildFromDB:
    async def test_build_on_empty_db_creates_empty_index(self, idx):
        stats = await idx.build()
        assert stats.image_count == 0
        assert stats.bbox_count == 0
        assert idx._image_tree is None
        assert idx._valid_images == set()

    async def test_rebuild_on_empty_db_returns_zero_counts(self, idx):
        stats = await idx.rebuild()
        assert stats.image_count == 0
        assert stats.bbox_count == 0
        assert idx._needs_rebuild is False

    async def test_stats_reflects_current_state(self, idx):
        idx.add_image("img-1", IMG_BLACK)
        s = idx.stats()
        assert s["image_valid"] == 1
        assert s["image_tombstones"] == 0
        assert s["needs_rebuild"] is False

    def test_build_structures_does_not_mutate_live_index(self, idx):
        """_build_structures (run in a worker thread) must build into a returned
        container WITHOUT touching the live trees, so concurrent queries on the
        event loop keep seeing the old, fully-populated index until _install
        performs the atomic swap. Otherwise a rebuild exposes half-built state.
        """
        idx.add_image("img-1", IMG_BLACK)

        data = idx._build_structures([{"uuid": "img-2", "phash": IMG_WHITE}], [])

        # Live index is untouched: img-1 still findable, img-2 not yet visible.
        assert any(u == "img-1" for u, _ in idx.find_similar_images(IMG_BLACK, 0))
        assert idx.find_similar_images(IMG_WHITE, 0) == []

        # The atomic swap makes the new dataset visible in one step.
        idx._install(data)
        assert any(u == "img-2" for u, _ in idx.find_similar_images(IMG_WHITE, 0))
        assert idx.find_similar_images(IMG_BLACK, 0) == []

    async def test_build_does_not_block_event_loop(self, idx):
        """_construct_trees must run off the event loop via asyncio.to_thread.

        If called synchronously, BK-tree construction on a large dataset blocks
        all concurrent requests until the build completes.
        """
        import threading
        from unittest.mock import patch

        ran_in_thread = False
        original = idx._build_structures

        def tracked_construct(image_rows, bbox_rows):
            nonlocal ran_in_thread
            ran_in_thread = threading.current_thread() is not threading.main_thread()
            return original(image_rows, bbox_rows)

        with patch.object(idx, "_build_structures", tracked_construct):
            await idx.build()

        assert ran_in_thread, "_build_structures must run in a worker thread, not on the event loop"

    async def test_rebuild_does_not_block_event_loop(self, idx):
        """rebuild() has the same non-blocking requirement as build()."""
        import threading
        from unittest.mock import patch

        ran_in_thread = False
        original = idx._build_structures

        def tracked_construct(image_rows, bbox_rows):
            nonlocal ran_in_thread
            ran_in_thread = threading.current_thread() is not threading.main_thread()
            return original(image_rows, bbox_rows)

        with patch.object(idx, "_build_structures", tracked_construct):
            await idx.rebuild()

        assert ran_in_thread, "_build_structures must run in a worker thread, not on the event loop"

    async def test_load_phashes_returns_all_rows(self, idx, db_client):
        """_load_phashes must accumulate every row regardless of CHUNK_SIZE."""
        from pathlib import Path

        # Seed 3 images with phashes
        for i in range(3):
            img = await db_client.add_image(Path(f"img_{i}.jpg"))
            await db_client.update_phashes(img.uuid, "a" * 16, [])

        # Force chunking: process 2 rows per page so we need 2 pages for 3 images
        idx.CHUNK_SIZE = 2
        image_rows, bbox_rows = await idx._load_phashes()

        assert len(image_rows) == 3
        assert all("uuid" in r and "phash" in r for r in image_rows)
        assert bbox_rows == []  # no bboxes seeded

    async def test_load_phashes_yields_between_full_pages(self, idx, db_client):
        """_load_phashes must call asyncio.sleep(0) after each full page so the
        event loop can process other requests during startup / rebuild."""
        import asyncio
        from pathlib import Path
        from unittest.mock import patch

        # Seed 3 images: with CHUNK_SIZE=2 the first page is full → one sleep expected
        for i in range(3):
            img = await db_client.add_image(Path(f"chunk_{i}.jpg"))
            await db_client.update_phashes(img.uuid, "b" * 16, [])

        idx.CHUNK_SIZE = 2
        sleep_calls = []

        async def _track_sleep(delay):
            sleep_calls.append(delay)

        with patch.object(asyncio, "sleep", _track_sleep):
            await idx._load_phashes()

        assert 0 in sleep_calls, "asyncio.sleep(0) must be called between full pages"

    async def test_load_phashes_no_sleep_when_single_page(self, idx, db_client):
        """_load_phashes must NOT yield when all rows fit in the first page."""
        import asyncio
        from pathlib import Path
        from unittest.mock import patch

        # 2 images with CHUNK_SIZE=500 → single page, no sleep
        for i in range(2):
            img = await db_client.add_image(Path(f"small_{i}.jpg"))
            await db_client.update_phashes(img.uuid, "c" * 16, [])

        sleep_calls = []

        async def _track_sleep(delay):
            sleep_calls.append(delay)

        with patch.object(asyncio, "sleep", _track_sleep):
            await idx._load_phashes()

        assert sleep_calls == [], "No sleep expected when all rows fit in one page"


# ─────────────────────────────────────────────────────────────────────────────
# on_phase callback
# ─────────────────────────────────────────────────────────────────────────────


class TestOnPhaseCallback:
    """build() invokes the on_phase callback at the correct points."""

    async def test_phases_reported_in_order(self, idx):
        phases = []
        await idx.build(on_phase=phases.append)
        assert phases == ["loading", "constructing"]

    async def test_loading_fires_before_db_query(self, idx):
        """'loading' must be reported before _load_phashes runs."""
        from unittest.mock import AsyncMock, patch

        captured = []

        async def _spy_load():
            captured.append(list(phases))
            return [], []

        phases = []
        with patch.object(idx, "_load_phashes", _spy_load):
            await idx.build(on_phase=phases.append)

        assert captured[0] == ["loading"], "on_phase('loading') must fire before _load_phashes"

    async def test_constructing_fires_before_tree_build(self, idx):
        """'constructing' must be reported before _construct_trees runs."""
        from unittest.mock import patch

        captured = []
        original = idx._build_structures

        def _spy_construct(image_rows, bbox_rows):
            captured.append(list(phases))
            return original(image_rows, bbox_rows)

        phases = []
        with patch.object(idx, "_build_structures", _spy_construct):
            await idx.build(on_phase=phases.append)

        assert captured[0] == [
            "loading",
            "constructing",
        ], "on_phase('constructing') must fire before _build_structures"

    async def test_build_without_on_phase_does_not_raise(self, idx):
        """on_phase is optional; omitting it must not raise."""
        await idx.build()  # no on_phase argument
