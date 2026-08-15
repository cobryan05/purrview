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


class TestReplaceImage:
    """An image uuid can outlive its contents — an overwriting upload writes new
    bytes over a registered path and keeps the row. The index has to follow, and
    remove+add cannot do it: the tree keeps the superseded item and its liveness
    filter is per-uuid, so re-adding the uuid makes the old hash answerable again.
    """

    def test_the_old_content_no_longer_matches(self, idx):
        idx.add_image("img-1", IMG_BLACK)
        idx.replace_image("img-1", IMG_WHITE)

        results = idx.find_similar_images(IMG_BLACK, threshold=0)
        assert all(uuid != "img-1" for uuid, _ in results)

    def test_the_new_content_matches(self, idx):
        idx.add_image("img-1", IMG_BLACK)
        idx.replace_image("img-1", IMG_WHITE)

        ((uuid, dist),) = idx.find_similar_images(IMG_WHITE, threshold=0)
        assert uuid == "img-1"
        assert dist == 0

    def test_other_images_are_untouched(self, idx):
        idx.add_image("img-1", IMG_BLACK)
        idx.add_image("img-2", IMG_BLACK)
        idx.replace_image("img-1", IMG_WHITE)

        uuids = {u for u, _ in idx.find_similar_images(IMG_BLACK, threshold=0)}
        assert uuids == {"img-2"}

    def test_replacing_with_the_same_hash_keeps_the_image_findable(self, idx):
        """Re-uploading identical bytes is a normal thing to do, and must not
        tombstone the entry it just re-added."""
        idx.add_image("img-1", IMG_BLACK)
        idx.replace_image("img-1", IMG_BLACK)

        assert any(uuid == "img-1" for uuid, _ in idx.find_similar_images(IMG_BLACK, threshold=0))

    def test_replacing_an_unknown_image_just_adds_it(self, idx):
        idx.replace_image("img-new", IMG_BLACK)

        assert any(uuid == "img-new" for uuid, _ in idx.find_similar_images(IMG_BLACK, threshold=0))

    def test_reverting_to_an_earlier_hash_makes_it_findable_again(self, idx):
        """Undoing an overwrite must clear the tombstone the overwrite left, or
        the restored image is filtered out on the strength of its own hash."""
        idx.add_image("img-1", IMG_BLACK)
        idx.replace_image("img-1", IMG_WHITE)
        idx.replace_image("img-1", IMG_BLACK)

        assert any(uuid == "img-1" for uuid, _ in idx.find_similar_images(IMG_BLACK, threshold=0))


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


# ─────────────────────────────────────────────────────────────────────────────
# Nearest-neighbour distance (threshold calibration)
# ─────────────────────────────────────────────────────────────────────────────


class TestNearestDistance:
    def test_empty_index_returns_none(self, idx):
        """None means 'nothing to compare against' — distinct from 'far away'."""
        assert idx.nearest_image_distance(IMG_BLACK) is None

    def test_identical_image_is_distance_zero(self, idx):
        idx.add_image("img-1", IMG_BLACK)
        assert idx.nearest_image_distance(IMG_BLACK) == 0

    def test_matches_hamming_of_closest_entry(self, idx):
        idx.add_image("img-white", IMG_WHITE)
        idx.add_image("img-grey", IMG_GREY)
        expected = min(hamming(IMG_BLACK, IMG_WHITE), hamming(IMG_BLACK, IMG_GREY))
        assert idx.nearest_image_distance(IMG_BLACK) == expected

    def test_reports_distance_beyond_the_search_threshold(self, idx):
        """The whole point: a distance is reported even when nothing is 'similar'.

        find_similar_images can only ever answer "within R?"; the calibration
        stats need the number itself, precisely for frames that fall outside R.
        """
        idx.add_image("img-checker", _checkerboard_phash(hash_size=8))
        far = hamming(IMG_BLACK, _checkerboard_phash(hash_size=8))
        assert far > 0, "test fixture must supply two genuinely different hashes"
        assert idx.find_similar_images(IMG_BLACK, threshold=far - 1) == []
        assert idx.nearest_image_distance(IMG_BLACK) == far

    def test_removed_image_no_longer_counts(self, idx):
        idx.add_image("img-1", IMG_BLACK)
        idx.add_image("img-white", IMG_WHITE)
        idx.remove_image("img-1")
        assert idx.nearest_image_distance(IMG_BLACK) == hamming(IMG_BLACK, IMG_WHITE)

    def test_removing_every_image_returns_none_again(self, idx):
        idx.add_image("img-1", IMG_BLACK)
        idx.remove_image("img-1")
        assert idx.nearest_image_distance(IMG_BLACK) is None

    def test_wrong_hash_size_raises(self, idx):
        with pytest.raises(ValueError):
            idx.nearest_image_distance("abcd")

    def test_bbox_distance_is_per_label(self, idx):
        idx.add_bbox("bb-1", "label-a", BBOX_CHECKER)
        assert idx.nearest_bbox_distance(BBOX_CHECKER, "label-a") == 0
        assert idx.nearest_bbox_distance(BBOX_CHECKER, "label-b") is None

    def test_unlabelled_bbox_returns_none(self, idx):
        """A crop is only comparable to crops of the same label."""
        idx.add_bbox("bb-1", "label-a", BBOX_CHECKER)
        assert idx.nearest_bbox_distance(BBOX_CHECKER, None) is None

    def test_removed_bbox_no_longer_counts(self, idx):
        idx.add_bbox("bb-1", "label-a", BBOX_CHECKER)
        idx.add_bbox("bb-2", "label-a", BBOX_GRAD)
        idx.remove_bbox("bb-1", "label-a")
        assert idx.nearest_bbox_distance(BBOX_CHECKER, "label-a") == hamming(BBOX_CHECKER, BBOX_GRAD)

    def test_moved_bbox_follows_its_new_label(self, idx):
        idx.add_bbox("bb-1", "label-a", BBOX_CHECKER)
        idx.move_bbox("bb-1", "label-a", "label-b", BBOX_CHECKER)
        assert idx.nearest_bbox_distance(BBOX_CHECKER, "label-a") is None
        assert idx.nearest_bbox_distance(BBOX_CHECKER, "label-b") == 0

    async def test_survives_rebuild_from_db(self, idx, db_client):
        """The flat mirror is rebuilt by _install, not only by incremental adds."""
        idx.add_image("img-1", IMG_BLACK)
        await idx.rebuild()
        # rebuild() reloads from the DB, which has no phash rows in this fixture —
        # the mirror must be reset in step with the tree, not left holding stale rows.
        assert idx.nearest_image_distance(IMG_BLACK) is None

    def test_grows_past_initial_capacity(self, idx):
        """The flat table starts at a fixed capacity and must grow correctly."""
        from apps.db_server.hash_index import _FlatHashTable

        for i in range(_FlatHashTable._INITIAL_CAPACITY + 10):
            idx.add_image(f"img-{i}", IMG_WHITE)
        idx.add_image("needle", IMG_BLACK)
        assert idx.nearest_image_distance(IMG_BLACK) == 0


class TestCheckDedupBulkDistances:
    def test_reports_distances_alongside_flags(self, idx):
        idx.add_image("img-1", IMG_BLACK)
        idx.add_bbox("bb-1", "label-a", BBOX_CHECKER)
        result = idx.check_dedup_bulk(
            IMG_BLACK,
            image_threshold=5,
            bboxes=[{"phash": BBOX_CHECKER, "label_uuid": "label-a", "threshold": 8}],
        )
        assert result["image_is_dup"] is True
        assert result["bbox_flags"] == [True]
        assert result["image_min_distance"] == 0
        assert result["bbox_min_distances"] == [0]

    def test_empty_index_gives_none_distances_and_false_flags(self, idx):
        result = idx.check_dedup_bulk(
            IMG_BLACK,
            image_threshold=5,
            bboxes=[{"phash": BBOX_CHECKER, "label_uuid": "label-a", "threshold": 8}],
        )
        assert result["image_is_dup"] is False
        assert result["bbox_flags"] == [False]
        assert result["image_min_distance"] is None
        assert result["bbox_min_distances"] == [None]

    def test_flag_and_distance_never_disagree(self, idx):
        """is_dup must be exactly 'min_distance <= threshold' at every threshold."""
        idx.add_image("img-white", IMG_WHITE)
        actual = hamming(IMG_BLACK, IMG_WHITE)
        for threshold in range(0, 65):
            result = idx.check_dedup_bulk(IMG_BLACK, image_threshold=threshold, bboxes=[])
            assert result["image_min_distance"] == actual
            assert result["image_is_dup"] is (actual <= threshold)


class TestMixedWidthPhashesDoNotBreakBuild:
    """A DB still holding phashes from a previous hash-size setting must not take
    the index build down — the BK-tree tolerates them, so the flat mirror must too."""

    def test_from_items_skips_wrong_width_rows(self):
        from apps.db_server.hash_index import _FlatHashTable

        items = [(IMG_BLACK, "ok-1"), ("dead", "stale-1"), (IMG_WHITE, "ok-2")]
        table = _FlatHashTable.from_items(items, byte_len=len(IMG_BLACK) // 2)
        assert table.nearest(IMG_BLACK) == (0, "ok-1")

    async def test_build_survives_a_stale_width_row(self, idx, db_client):
        from unittest.mock import patch

        rows = ([{"uuid": "ok-1", "phash": IMG_BLACK}, {"uuid": "stale", "phash": "dead"}], [])

        async def _load(*args, **kwargs):
            return rows

        with patch.object(idx, "_load_phashes", _load):
            await idx.build()
        assert idx.nearest_image_distance(IMG_BLACK) == 0
