"""Disk accounting for the recording side, and the purges that act on it.

The settings screens let you cap how much disk clips and timelapses may take, but
nothing said where you actually were against those caps — so the caps were being
set blind. `get_storage_usage()` answers that, and three actions reclaim space.

Two properties matter enough to pin, and both are about the *relationship* between
this readout and the sweeps rather than about any particular number:

  - Usage must be measured by the same walk the size caps enforce with. A figure
    from a different walk would disagree with eviction in exactly the cases that
    matter, and a readout that disagrees with enforcement is worse than none.
  - The unbounded purge must have exactly the timelapse sweep's scope. It is the
    one button here that ignores the limits, and it operates in a directory tree
    that also holds a user's annotated images.
"""

from datetime import datetime, timedelta

import pytest

from apps.db_server.manager import Manager
from apps.helpers.actions import outputDir
from apps.helpers.db.types import WatchAction, WatchProfile


@pytest.fixture
def timelapse_root(tmp_path, monkeypatch):
    """A FILES_ROOT the test owns — a purge that unlinks must never be able to run
    against the developer's own image_datasets just because a test called it."""
    root = tmp_path / "files"
    (root / "timelapse").mkdir(parents=True)
    monkeypatch.setattr(outputDir, "FILES_ROOT", root)
    return root / "timelapse"


@pytest.fixture
def manager(db_client, files_root, tmp_path, monkeypatch):
    m = Manager(db_client=db_client, files_root=files_root)
    m._db_ready = True
    # Same reasoning as timelapse_root: this walks and deletes.
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    m._recording_roots = (recordings,)
    monkeypatch.setattr(m, "_thumb_cache_dir", lambda: tmp_path / "thumbs")
    return m


@pytest.fixture
async def timelapse_action(db_client):
    async with db_client._async_session_maker() as session:
        profile = WatchProfile(name="House cams")
        session.add(profile)
        await session.commit()
        await session.refresh(profile)
        session.add(
            WatchAction(
                profile_uuid=profile.uuid,
                kind="save_timelapse",
                trigger="frame_analysed",
                params_json={"output_dir": "timelapse"},
            )
        )
        await session.commit()
        return profile.uuid


def _frame(root, days_ago: int, name="back-room_12-00-00.jpg", size=1024):
    day = (datetime.now().astimezone() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    day_dir = root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / name
    path.write_bytes(b"x" * size)
    return path


class TestStorageUsage:
    async def test_each_bucket_reports_what_is_actually_on_disk(
        self, manager, timelapse_root, timelapse_action, tmp_path
    ):
        (manager._recording_roots[0] / "clip.mp4").write_bytes(b"c" * 4096)
        _frame(timelapse_root, days_ago=1, size=1024)
        _frame(timelapse_root, days_ago=2, size=2048)
        thumbs = tmp_path / "thumbs"
        thumbs.mkdir()
        (thumbs / "abc_120.png").write_bytes(b"t" * 512)

        usage = await manager.get_storage_usage()

        assert usage["recordings"]["bytes"] == 4096
        assert usage["recordings"]["files"] == 1
        assert usage["timelapse"]["bytes"] == 3072
        assert usage["timelapse"]["files"] == 2
        assert usage["thumbnailCache"]["bytes"] == 512
        assert usage["thumbnailCache"]["files"] == 1

    async def test_usage_agrees_with_what_the_size_cap_measures(self, manager, timelapse_root, timelapse_action):
        """The load-bearing property. If these two walks ever diverge, the panel
        reports one number and eviction acts on another — and the symptom is a cap
        that fires 'early' or never, with a readout that says it shouldn't have."""
        _frame(timelapse_root, days_ago=1, size=1500)
        _frame(timelapse_root, days_ago=1, name="back-room_12-00-30.jpg", size=2500)

        usage = await manager.get_storage_usage()
        roots = await manager.timelapse_roots()
        by_the_sweep = sum(size for _p, size, _d in manager._scan_timelapse_frames(roots))

        assert usage["timelapse"]["bytes"] == by_the_sweep

    async def test_empty_directories_report_zero_rather_than_failing(self, manager, timelapse_root):
        """No watchers configured, nothing recorded, no cache directory yet — the
        state a fresh install is in, where a crash here would break Settings."""
        usage = await manager.get_storage_usage()
        assert usage["timelapse"]["bytes"] == 0
        assert usage["thumbnailCache"]["files"] == 0


class TestPurgeAllTimelapseFrames:
    async def test_it_ignores_the_retention_window(self, manager, timelapse_root, timelapse_action):
        """The difference from the sweep: a frame filmed an hour ago goes too.
        A purge that only removed expired frames would be the sweep with a
        different name, and would free nothing on the disk that prompted it."""
        today = _frame(timelapse_root, days_ago=0)
        old = _frame(timelapse_root, days_ago=99)

        result = await manager.purge_all_timelapse_frames()

        assert result["files"] == 2
        assert not today.exists()
        assert not old.exists()

    async def test_it_will_not_touch_an_image_that_has_a_sidecar(self, manager, timelapse_root, timelapse_action):
        """The rail that makes this safe to put behind a button. A curated image
        living in the same day folder — which happens the moment someone annotates
        a frame in place — must survive a purge of everything around it."""
        frame = _frame(timelapse_root, days_ago=1)
        curated = frame.parent / "back-room_13-00-00.jpg"
        curated.write_bytes(b"y" * 900)
        curated.with_suffix(".jpg.json").write_text("{}")

        result = await manager.purge_all_timelapse_frames()

        assert not frame.exists()
        assert curated.exists(), "an annotated image must survive an unbounded purge"
        assert result["files"] == 1

    async def test_it_will_not_touch_a_file_outside_a_day_folder(self, manager, timelapse_root, timelapse_action):
        """Only files filed under a YYYY-MM-DD folder are ones this app wrote.
        Anything else in the directory the user pointed the action at is theirs."""
        stray = timelapse_root / "back-room_12-00-00.jpg"
        stray.write_bytes(b"z" * 100)
        _frame(timelapse_root, days_ago=1)

        result = await manager.purge_all_timelapse_frames()

        assert stray.exists()
        assert result["files"] == 1

    async def test_nothing_to_purge_is_reported_rather_than_raised(self, manager, timelapse_root):
        result = await manager.purge_all_timelapse_frames()
        assert result == {"files": 0, "bytes": 0}


class TestClearThumbnailCache:
    async def test_it_removes_cached_thumbnails_and_reports_what_it_freed(self, manager, tmp_path):
        thumbs = tmp_path / "thumbs"
        thumbs.mkdir()
        (thumbs / "a_120.png").write_bytes(b"a" * 300)
        (thumbs / "b_240.png").write_bytes(b"b" * 700)

        result = await manager.clear_thumbnail_cache()

        assert result == {"files": 2, "bytes": 1000}
        assert list(thumbs.glob("*.png")) == []

    async def test_it_leaves_non_thumbnail_files_alone(self, manager, tmp_path):
        """The cache directory is under CACHE_DIR, which is not exclusively ours.
        Globbing for what we write is the difference between clearing a cache and
        emptying a directory."""
        thumbs = tmp_path / "thumbs"
        thumbs.mkdir()
        (thumbs / "a_120.png").write_bytes(b"a" * 300)
        keeper = thumbs / "notes.txt"
        keeper.write_text("not ours")

        await manager.clear_thumbnail_cache()

        assert keeper.exists()

    async def test_a_missing_cache_directory_is_not_an_error(self, manager):
        assert await manager.clear_thumbnail_cache() == {"files": 0, "bytes": 0}


class TestStorageApi:
    async def test_usage_endpoint_serves_every_bucket_in_camel_case(self, http_client):
        resp = await http_client.get("/api/storage/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        for bucket in ("recordings", "timelapse", "thumbnailCache"):
            assert bucket in data, data
            # The nested model does not inherit StatusResponse, so its alias
            # generator has to be declared on it — snake_case here means the
            # frontend reads undefined and renders "0 B" for everything.
            assert "maxBytes" in data[bucket], data[bucket]
