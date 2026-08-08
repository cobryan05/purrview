"""Tests for the bounded-growth guards on db_server's caches and thumbnail work.

Each of these covers a structure that previously grew (or spawned work) once per
browsing request with nothing ever reclaiming it.
"""

import asyncio

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
