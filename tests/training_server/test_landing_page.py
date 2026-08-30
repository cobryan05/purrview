"""The training server's landing page, info endpoint, and cache controls.

The training server is meant to be runnable on its own host, where GET / was a
404 and there was no way to see — or reclaim — the image cache without a shell
on the box.
"""

import pytest

from apps.training_server.image_cache import ImageCache


def _fill_cache(cache_dir: str, *uuids: str, size: int = 1024) -> None:
    from pathlib import Path

    for uuid in uuids:
        (Path(cache_dir) / uuid).write_bytes(b"x" * size)


class TestLandingPage:
    async def test_root_serves_a_page(self, http_client):
        resp = await http_client.get("/")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        body = resp.text
        assert "PurrView Training Server" in body
        # The page drives itself off these two endpoints; a rename that breaks
        # them leaves a page that renders but shows nothing.
        assert '"/status"' in body
        assert '"/api/training/info"' in body

    async def test_page_is_self_contained(self, http_client):
        """A service alone on a host has no /static and no CDN reachable."""
        body = (await http_client.get("/")).text

        assert "/static/" not in body
        assert "http://" not in body.replace("http://www.w3.org", "")
        assert "https://" not in body


class TestInfoEndpoint:
    async def test_reports_cache_occupancy(self, http_client, tmp_training_dirs):
        cache_dir, _ = tmp_training_dirs
        _fill_cache(cache_dir, "uuid-a", "uuid-b", size=2048)

        resp = await http_client.get("/api/training/info")

        assert resp.status_code == 200
        body = resp.json()
        assert body["cache"]["files"] == 2
        assert body["cache"]["bytes"] == 4096
        assert body["cache"]["dir"] == cache_dir
        assert [s["title"] for s in body["details"]] == ["Image cache", "Artifacts", "Server"]

    async def test_reports_artifact_usage(self, http_client, manager, tmp_training_dirs):
        _, artifacts_dir = tmp_training_dirs
        from pathlib import Path

        job_dir = Path(artifacts_dir) / "job-1"
        job_dir.mkdir()
        (job_dir / "model.onnx").write_bytes(b"y" * 512)

        body = (await http_client.get("/api/training/info")).json()

        assert body["artifacts"]["jobs"] == 1
        assert body["artifacts"]["bytes"] == 512


class TestCacheControls:
    async def test_clear_keeps_images_pinned_by_live_jobs(self, http_client, manager, tmp_training_dirs):
        cache_dir, _ = tmp_training_dirs
        _fill_cache(cache_dir, "keep-me", "drop-me")
        manager._cache.pin({"keep-me"})

        resp = await http_client.delete("/api/training/cache")

        assert resp.status_code == 200
        assert resp.json() == {"removed": 1, "freed_bytes": 1024, "kept_pinned": 1}
        assert manager._cache.has("keep-me")
        assert not manager._cache.has("drop-me")

    async def test_clear_include_pinned_takes_everything(self, http_client, manager, tmp_training_dirs):
        cache_dir, _ = tmp_training_dirs
        _fill_cache(cache_dir, "keep-me", "drop-me")
        manager._cache.pin({"keep-me"})

        resp = await http_client.delete("/api/training/cache?include_pinned=true")

        assert resp.json()["removed"] == 2
        assert resp.json()["kept_pinned"] == 0
        assert not manager._cache.has("keep-me")

    async def test_evict_applies_the_size_bound_now(self, http_client, manager, tmp_training_dirs):
        cache_dir, _ = tmp_training_dirs
        # 3 KB of images against a 2 KB bound: the sweep must drop the oldest.
        manager._cache = ImageCache(cache_dir=cache_dir, max_mb=0)
        manager._cache._max_bytes = 2048
        _fill_cache(cache_dir, "a", "b", "c")

        resp = await http_client.post("/api/training/cache/evict")

        assert resp.status_code == 200
        assert resp.json()["removed"] >= 1
        assert resp.json()["bytes"] <= 2048


class TestCacheStats:
    def test_stats_does_not_touch_mtimes(self, tmp_path):
        """Reading occupancy must not make every file look freshly used."""
        cache = ImageCache(cache_dir=str(tmp_path))
        target = tmp_path / "uuid-a"
        target.write_bytes(b"x" * 10)
        import os

        os.utime(target, (1_000_000, 1_000_000))

        stats = cache.stats()

        assert stats["files"] == 1
        assert target.stat().st_mtime == pytest.approx(1_000_000)
        assert stats["oldest_mtime"] == pytest.approx(1_000_000)

    def test_stats_ignores_partial_writes(self, tmp_path):
        cache = ImageCache(cache_dir=str(tmp_path))
        (tmp_path / "uuid-a").write_bytes(b"x" * 10)
        (tmp_path / ".tmp_uuid-b").write_bytes(b"x" * 999)

        assert cache.stats() == {
            **cache.stats(),
            "files": 1,
            "bytes": 10,
        }

    def test_clear_ignores_partial_writes(self, tmp_path):
        cache = ImageCache(cache_dir=str(tmp_path))
        (tmp_path / ".tmp_uuid-b").write_bytes(b"x" * 999)

        assert cache.clear()["removed"] == 0
        assert (tmp_path / ".tmp_uuid-b").exists()
