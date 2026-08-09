"""Tests for ImageCache eviction (size + age) and local (no-copy) ingest."""

import errno
import os
import time

from apps.training_server.image_cache import ImageCache

# ---------------------------------------------------------------------------
# Size-based eviction
# ---------------------------------------------------------------------------


def test_evict_removes_oldest_when_over_size_limit(tmp_path):
    cache = ImageCache(cache_dir=str(tmp_path), max_mb=0, max_age_days=0)
    cache._max_bytes = 10  # force eviction regardless of max_mb rounding
    cache.ensure_dir()

    old = tmp_path / "old-uuid"
    old.write_bytes(b"x" * 6)
    os.utime(old, (time.time() - 100, time.time() - 100))

    new = tmp_path / "new-uuid"
    new.write_bytes(b"x" * 6)

    cache.evict()

    assert not old.exists()
    assert new.exists()


def test_evict_never_removes_pinned_uuid_under_size_pressure(tmp_path):
    cache = ImageCache(cache_dir=str(tmp_path), max_mb=0, max_age_days=0)
    cache._max_bytes = 1
    cache.ensure_dir()
    cache.pin({"pinned-uuid"})

    p = tmp_path / "pinned-uuid"
    p.write_bytes(b"x" * 100)

    cache.evict()

    assert p.exists()


# ---------------------------------------------------------------------------
# Age-based (TTL) eviction
# ---------------------------------------------------------------------------


def test_evict_removes_files_older_than_ttl_even_under_size_limit(tmp_path):
    """An old file is evicted purely on age, even though the cache is nowhere near its size cap."""
    cache = ImageCache(cache_dir=str(tmp_path), max_mb=5000, max_age_days=1)
    cache.ensure_dir()

    stale = tmp_path / "stale-uuid"
    stale.write_bytes(b"x")
    two_days_ago = time.time() - 2 * 86400
    os.utime(stale, (two_days_ago, two_days_ago))

    fresh = tmp_path / "fresh-uuid"
    fresh.write_bytes(b"x")

    cache.evict()

    assert not stale.exists()
    assert fresh.exists()


def test_evict_ttl_disabled_when_max_age_days_is_zero(tmp_path):
    cache = ImageCache(cache_dir=str(tmp_path), max_mb=5000, max_age_days=0)
    cache.ensure_dir()

    stale = tmp_path / "stale-uuid"
    stale.write_bytes(b"x")
    long_ago = time.time() - 365 * 86400
    os.utime(stale, (long_ago, long_ago))

    cache.evict()

    assert stale.exists()


def test_evict_ttl_never_removes_pinned_uuid(tmp_path):
    cache = ImageCache(cache_dir=str(tmp_path), max_mb=5000, max_age_days=1)
    cache.ensure_dir()
    cache.pin({"pinned-uuid"})

    p = tmp_path / "pinned-uuid"
    p.write_bytes(b"x")
    two_days_ago = time.time() - 2 * 86400
    os.utime(p, (two_days_ago, two_days_ago))

    cache.evict()

    assert p.exists()


# ---------------------------------------------------------------------------
# Local (no-copy) ingest
# ---------------------------------------------------------------------------


def test_ingest_local_hardlinks_same_device(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache = ImageCache(cache_dir=str(cache_dir))

    src = src_dir / "a.jpg"
    src.write_bytes(b"fake-jpeg")

    result = cache.ingest_local([("uuid-a", src)])

    assert result["received"] == 1
    assert result["linked"] == 1
    assert result["copied"] == 0
    assert result["not_found"] == []
    dest = cache_dir / "uuid-a"
    assert dest.read_bytes() == b"fake-jpeg"
    # A true hardlink shares the inode with the source.
    assert dest.stat().st_ino == src.stat().st_ino


def test_ingest_local_falls_back_to_copy_across_devices(tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache = ImageCache(cache_dir=str(cache_dir))

    src = src_dir / "a.jpg"
    src.write_bytes(b"fake-jpeg")

    def fake_link(_src, _dst):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", fake_link)

    result = cache.ingest_local([("uuid-a", src)])

    assert result["received"] == 1
    assert result["linked"] == 0
    assert result["copied"] == 1
    dest = cache_dir / "uuid-a"
    assert dest.read_bytes() == b"fake-jpeg"
    assert dest.stat().st_ino != src.stat().st_ino


def test_ingest_local_reports_missing_source_as_not_found(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = ImageCache(cache_dir=str(cache_dir))

    result = cache.ingest_local([("uuid-missing", tmp_path / "does-not-exist.jpg")])

    assert result["received"] == 0
    assert result["not_found"] == ["uuid-missing"]


def test_ingest_local_rejects_uuids_outside_required_set(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache = ImageCache(cache_dir=str(cache_dir))

    src = src_dir / "a.jpg"
    src.write_bytes(b"data")

    result = cache.ingest_local([("uuid-a", src)], required_uuids={"some-other-uuid"})

    assert not cache.has("uuid-a")
    assert result["received"] == 0


def test_ingest_local_skips_already_cached_uuid(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache = ImageCache(cache_dir=str(cache_dir))
    cache.ensure_dir()
    (cache_dir / "uuid-a").write_bytes(b"already-here")

    src = src_dir / "a.jpg"
    src.write_bytes(b"different-bytes")

    result = cache.ingest_local([("uuid-a", src)])

    assert result["received"] == 1
    assert result["linked"] == 0
    assert result["copied"] == 0
    # Existing cached content is untouched, not overwritten from source.
    assert (cache_dir / "uuid-a").read_bytes() == b"already-here"
