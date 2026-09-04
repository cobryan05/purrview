"""Tests for the bounded-growth guards on db_server's caches and thumbnail work.

Each of these covers a structure that previously grew (or spawned work) once per
browsing request with nothing ever reclaiming it.
"""

import asyncio
import os

import pytest

from apps.db_server.manager import Manager


@pytest.fixture
def manager(db_client, files_root):
    mgr = Manager(db_client=db_client, files_root=files_root)
    mgr._db_ready = True
    return mgr


class TestCountCacheEviction:
    def test_expired_entries_are_dropped_once_past_the_ceiling(self, manager):
        now = 10_000.0
        ttl = manager._COUNT_CACHE_TTL
        ceiling = manager._COUNT_CACHE_MAX_ENTRIES

        # Fill past the ceiling with entries that have already timed out.
        for i in range(ceiling + 50):
            manager._count_cache[f"stale-{i}"] = (i, now - ttl - 1)

        manager._evict_expired_counts(now)

        assert manager._count_cache == {}

    def test_under_the_ceiling_nothing_is_touched(self, manager):
        """The sweep is a safety valve, not a TTL enforcer — reads still check the ts."""
        now = 10_000.0
        manager._count_cache["a"] = (1, now - manager._COUNT_CACHE_TTL - 1)

        manager._evict_expired_counts(now)

        assert "a" in manager._count_cache

    def test_live_entries_are_trimmed_oldest_first_when_all_are_fresh(self, manager):
        now = 10_000.0
        ceiling = manager._COUNT_CACHE_MAX_ENTRIES

        # All still within TTL, so the expiry sweep can't free anything.
        for i in range(ceiling + 10):
            manager._count_cache[f"fresh-{i}"] = (i, now - (ceiling + 10 - i) * 0.001)

        manager._evict_expired_counts(now)

        assert len(manager._count_cache) == ceiling
        # The ten oldest went first.
        assert "fresh-0" not in manager._count_cache
        assert f"fresh-{ceiling + 9}" in manager._count_cache

    def test_cache_does_not_grow_without_bound_over_many_distinct_filters(self, manager):
        now = 10_000.0
        for i in range(5_000):
            manager._count_cache[f"filter-{i}"] = (i, now - manager._COUNT_CACHE_TTL - 1)
            manager._evict_expired_counts(now)

        assert len(manager._count_cache) <= manager._COUNT_CACHE_MAX_ENTRIES


class TestThumbCacheTrimCoalescing:
    async def test_second_schedule_is_a_noop_while_one_is_in_flight(self, manager, monkeypatch):
        """Each trim globs the whole cache dir; one per thumbnail miss is pure waste."""
        started = asyncio.Event()
        release = asyncio.Event()
        runs = []

        async def _blocking_trim():
            runs.append(1)
            started.set()
            await release.wait()

        monkeypatch.setattr("apps.db_server.manager.THUMB_CACHE_MAX_MB", 100)
        monkeypatch.setattr(manager, "_trim_thumb_cache_sync", lambda mb: None)
        # Replace the scheduled coroutine with one we control the lifetime of.
        monkeypatch.setattr(
            "apps.db_server.manager.asyncio.to_thread",
            lambda fn, *a, **kw: _blocking_trim(),
        )

        manager._schedule_thumb_cache_trim()
        await started.wait()
        first_task = manager._thumb_trim_task

        for _ in range(10):
            manager._schedule_thumb_cache_trim()

        assert manager._thumb_trim_task is first_task
        assert len(runs) == 1

        release.set()
        await first_task

    async def test_a_new_trim_can_be_scheduled_after_the_previous_finished(self, manager, monkeypatch):
        monkeypatch.setattr("apps.db_server.manager.THUMB_CACHE_MAX_MB", 100)
        calls = []
        monkeypatch.setattr(manager, "_trim_thumb_cache_sync", lambda mb: calls.append(mb))

        manager._schedule_thumb_cache_trim()
        await manager._thumb_trim_task
        manager._schedule_thumb_cache_trim()
        await manager._thumb_trim_task

        assert len(calls) == 2

    def test_disabled_limit_schedules_nothing(self, manager, monkeypatch):
        monkeypatch.setattr("apps.db_server.manager.THUMB_CACHE_MAX_MB", 0)

        manager._schedule_thumb_cache_trim()

        assert manager._thumb_trim_task is None

    async def test_trim_task_reference_is_retained(self, manager, monkeypatch):
        """asyncio only holds weak refs to tasks; a bare create_task can be GC'd mid-run."""
        monkeypatch.setattr("apps.db_server.manager.THUMB_CACHE_MAX_MB", 100)
        monkeypatch.setattr(manager, "_trim_thumb_cache_sync", lambda mb: None)

        manager._schedule_thumb_cache_trim()

        assert manager._thumb_trim_task is not None
        await manager._thumb_trim_task


class TestThumbnailDecodeConcurrency:
    async def test_decode_concurrency_is_bounded(self, manager):
        """Full-res decodes must not all run at once — a 4K source is ~25 MB each."""
        limit = manager._thumb_decode_sem._value
        assert limit >= 1

        in_flight = 0
        peak = 0

        async def _fake_decode():
            nonlocal in_flight, peak
            async with manager._thumb_decode_sem:
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0)
                in_flight -= 1

        await asyncio.gather(*(_fake_decode() for _ in range(limit * 5)))

        assert peak <= limit


class TestSelectivityCacheEviction:
    async def test_expired_entries_swept_once_past_ceiling(self, db_client):
        from apps.helpers.db import db_client as db_client_mod

        ttl = db_client_mod._SELECTIVITY_CACHE_TTL
        ceiling = db_client_mod._SELECTIVITY_CACHE_MAX_ENTRIES

        import time

        now = time.monotonic()
        for i in range(ceiling + 5):
            db_client._selectivity_cache[f"stale-{i}"] = (i, now - ttl - 1)

        # Any real probe triggers the sweep on write.
        from sqlmodel import func, select

        from apps.helpers.db.types import ImageRecord

        await db_client._selectivity_count("probe", select(func.count()).select_from(ImageRecord))

        assert len(db_client._selectivity_cache) <= ceiling + 1

    async def test_live_entries_are_trimmed_oldest_first_when_none_have_expired(self, db_client):
        """A burst of distinct filter probes inside one TTL window leaves every
        entry still "live" — the expiry sweep alone can't free anything, so it
        must fall back to dropping the oldest entries to enforce the ceiling."""
        from apps.helpers.db import db_client as db_client_mod

        ceiling = db_client_mod._SELECTIVITY_CACHE_MAX_ENTRIES

        import time

        now = time.monotonic()
        # All within TTL (recent timestamps), so the expiry sweep can't evict any.
        for i in range(ceiling + 10):
            db_client._selectivity_cache[f"fresh-{i}"] = (i, now - (ceiling + 10 - i) * 0.001)

        from sqlmodel import func, select

        from apps.helpers.db.types import ImageRecord

        await db_client._selectivity_count("probe", select(func.count()).select_from(ImageRecord))

        assert len(db_client._selectivity_cache) <= ceiling + 1
        # The oldest of the "fresh" entries went first.
        assert "fresh-0" not in db_client._selectivity_cache
        assert f"fresh-{ceiling + 9}" in db_client._selectivity_cache


class TestThumbCacheTrimHysteresis:
    """The trim triggers at the cap but runs down past it.

    Evicting to exactly the cap leaves the cache full, so the very next write trims
    again — and the file it just evicted is usually the one the next request
    regenerates. The live cache sat at 99.98% of its 100 MB limit doing precisely
    that, evicting and re-encoding continuously while never being over its bound.
    """

    @staticmethod
    def _fill(cache_dir, count: int, size: int, suffix: str = ".jpg"):
        cache_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            p = cache_dir / f"img_fill{i:03d}_200{suffix}"
            p.write_bytes(b"x" * size)
            os.utime(p, (1_000_000 + i, 1_000_000 + i))  # deterministic LRU order

    def test_trim_leaves_headroom_below_the_cap(self, manager, monkeypatch, tmp_path):
        monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)
        cache_dir = tmp_path / "thumb_cache"
        self._fill(cache_dir, 20, 1000)

        cap = 15_000
        manager._trim_thumb_cache_sync(cap)

        remaining = sum(p.stat().st_size for p in cache_dir.glob("*.jpg"))
        assert remaining <= cap * 0.85, "trim stopped at the cap instead of trimming past it"
        # ...but it is a trim, not a purge.
        assert remaining > 0

    def test_a_cache_under_the_cap_is_left_alone(self, manager, monkeypatch, tmp_path):
        monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)
        cache_dir = tmp_path / "thumb_cache"
        self._fill(cache_dir, 5, 1000)

        manager._trim_thumb_cache_sync(100_000)

        assert len(list(cache_dir.glob("*.jpg"))) == 5

    def test_oldest_are_evicted_first(self, manager, monkeypatch, tmp_path):
        monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)
        cache_dir = tmp_path / "thumb_cache"
        self._fill(cache_dir, 10, 1000)

        manager._trim_thumb_cache_sync(5_000)

        survivors = sorted(p.name for p in cache_dir.glob("*.jpg"))
        assert survivors and all(name >= "img_fill005" for name in survivors)


class TestLegacyPngThumbnailsStillCount:
    """Thumbnails are JPEG now, but the cache is full of PNGs written before the switch.

    Globbing only the current suffix would make that set invisible: not counted by the
    storage readout, and — the part that matters — never evicted, so it would sit on
    disk forever while the cache "stayed under" its cap by ignoring it.
    """

    def test_usage_counts_both_formats(self, manager, monkeypatch, tmp_path):
        monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)
        cache_dir = tmp_path / "thumb_cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "img_old_200.png").write_bytes(b"x" * 700)
        (cache_dir / "img_new_200.jpg").write_bytes(b"x" * 300)

        files, total = manager._thumb_cache_usage()

        assert (files, total) == (2, 1000)

    def test_trim_evicts_legacy_files_too(self, manager, monkeypatch, tmp_path):
        monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)
        cache_dir = tmp_path / "thumb_cache"
        cache_dir.mkdir(parents=True)
        old = cache_dir / "img_old_200.png"
        old.write_bytes(b"x" * 5000)
        os.utime(old, (1_000_000, 1_000_000))
        new = cache_dir / "img_new_200.jpg"
        new.write_bytes(b"x" * 1000)
        os.utime(new, (2_000_000, 2_000_000))

        manager._trim_thumb_cache_sync(2_000)

        assert not old.exists(), "a pre-switch PNG was never a candidate for eviction"
        assert new.exists()

    async def test_clearing_the_cache_removes_both_formats(self, manager, monkeypatch, tmp_path):
        monkeypatch.setattr("apps.db_server.manager.CACHE_DIR", tmp_path)
        cache_dir = tmp_path / "thumb_cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "img_old_200.png").write_bytes(b"x" * 700)
        (cache_dir / "img_new_200.jpg").write_bytes(b"x" * 300)

        result = await manager.clear_thumbnail_cache()

        assert result == {"files": 2, "bytes": 1000}
        assert list(cache_dir.iterdir()) == []
