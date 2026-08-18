"""Timelapse frames are swept once they pass their own bounds.

A timelapse is the app's most prolific writer by a wide margin — a camera shooting every
30 seconds writes ~2,900 files a day, where a busy watcher might produce a few dozen
clips — so it gets bounds of its own rather than sharing the recording ones.

The edges these tests pin down are all variations of one worry. Unlike clips, which live
in a folder the app owns, timelapse frames live under FILES_ROOT: the same tree as the
images a user has spent months annotating, in a directory the user chose. So most of
this file is about what the sweep must **not** touch.
"""

from datetime import datetime, timedelta

import pytest

from apps.db_server.manager import (
    DEFAULT_TIMELAPSE_RETENTION_DAYS,
    TIMELAPSE_MAX_TOTAL_MB_SETTING_KEY,
    TIMELAPSE_RETENTION_SETTING_KEY,
    Manager,
)
from apps.helpers.actions import outputDir
from apps.helpers.db.types import WatchAction, WatchProfile


@pytest.fixture
def timelapse_root(tmp_path, monkeypatch):
    """A FILES_ROOT the test owns, since the action resolves its directory against it.

    A sweep that walks directories and unlinks must never be able to run against the
    developer's own image_datasets just because a test called it.
    """
    root = tmp_path / "files"
    (root / "timelapse").mkdir(parents=True)
    monkeypatch.setattr(outputDir, "FILES_ROOT", root)
    return root / "timelapse"


@pytest.fixture
def manager(db_client, files_root):
    m = Manager(db_client=db_client, files_root=files_root)
    m._db_ready = True
    return m


@pytest.fixture
async def timelapse_action(db_client):
    """A profile with one `save_timelapse` action — the sweep takes its roots from these."""
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


def _frame(root, days_ago: int, name="back-room_12-00-00.jpg", size=1024) -> "object":
    day = (datetime.now().astimezone() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    day_dir = root / day
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / name
    path.write_bytes(b"x" * size)
    return path


class TestTimelapseRoots:
    async def test_roots_come_from_the_actions(self, manager, timelapse_root, timelapse_action):
        assert await manager.timelapse_roots() == [timelapse_root]

    async def test_a_disabled_action_still_has_its_frames_managed(
        self, manager, db_client, timelapse_root, timelapse_action
    ):
        """Disabling stops new frames; it does not make the old ones unmanaged."""
        async with db_client._async_session_maker() as session:
            action = (await session.exec(WatchAction.__table__.select())).first()
            await session.exec(
                WatchAction.__table__.update().where(WatchAction.uuid == action.uuid).values(enabled=False)
            )
            await session.commit()
        assert await manager.timelapse_roots() == [timelapse_root]

    async def test_an_unresolvable_root_does_not_stop_the_sweep(self, manager, db_client, timelapse_root):
        async with db_client._async_session_maker() as session:
            profile = WatchProfile(name="Bad watcher")
            session.add(profile)
            await session.commit()
            await session.refresh(profile)
            session.add(
                WatchAction(
                    profile_uuid=profile.uuid,
                    kind="save_timelapse",
                    trigger="frame_analysed",
                    params_json={"output_dir": "../../etc"},
                )
            )
            await session.commit()
        # Reported, skipped, and the pass still completes.
        assert await manager.timelapse_roots() == []
        assert await manager.prune_expired_timelapse() == {"files": 0, "bytes": 0}

    async def test_no_timelapse_actions_means_no_work(self, manager, timelapse_root):
        assert await manager.prune_expired_timelapse() == {"files": 0, "bytes": 0}


class TestTimelapseAgeBound:
    async def test_frames_past_the_window_are_deleted(self, manager, timelapse_root, timelapse_action):
        old = _frame(timelapse_root, days_ago=40)
        recent = _frame(timelapse_root, days_ago=1)
        result = await manager.prune_timelapse_by_age()
        assert result["files"] == 1
        assert not old.exists()
        assert recent.exists()

    async def test_an_emptied_day_directory_is_removed(self, manager, timelapse_root, timelapse_action):
        """A year of retention would otherwise leave 365 empty folders behind."""
        old = _frame(timelapse_root, days_ago=40)
        await manager.prune_timelapse_by_age()
        assert not old.parent.exists()

    async def test_a_day_directory_holding_something_else_survives(self, manager, timelapse_root, timelapse_action):
        old = _frame(timelapse_root, days_ago=40)
        keeper = old.parent / "notes.txt"
        keeper.write_text("mine")
        await manager.prune_timelapse_by_age()
        assert not old.exists()
        assert keeper.exists()

    async def test_zero_days_keeps_everything(self, manager, db_client, timelapse_root, timelapse_action):
        old = _frame(timelapse_root, days_ago=400)
        await db_client.set_setting(TIMELAPSE_RETENTION_SETTING_KEY, "0")
        assert (await manager.prune_timelapse_by_age())["files"] == 0
        assert old.exists()

    async def test_an_unreadable_setting_falls_back_rather_than_wedging(self, manager, db_client):
        """A stuck sweep is a disk that fills silently, which is the failure this prevents."""
        await db_client.set_setting(TIMELAPSE_RETENTION_SETTING_KEY, "soon")
        assert await manager.get_timelapse_retention_days() == DEFAULT_TIMELAPSE_RETENTION_DAYS


class TestTimelapseSweepSafety:
    """The rails that let this run inside a folder a user also keeps images in."""

    async def test_a_file_that_is_not_a_frame_is_left_alone(self, manager, timelapse_root, timelapse_action):
        old = _frame(timelapse_root, days_ago=40)
        stranger = old.parent / "holiday-photo.jpg"
        stranger.write_bytes(b"mine")
        await manager.prune_timelapse_by_age()
        assert stranger.exists()

    async def test_a_frame_with_a_sidecar_is_never_deleted(self, manager, timelapse_root, timelapse_action):
        """A JSON sidecar is this app's marker for a curated image; a frame has none."""
        curated = _frame(timelapse_root, days_ago=40, name="back-room_09-00-00.jpg")
        curated.with_suffix(".jpg.json").write_text("{}")
        await manager.prune_timelapse_by_age()
        assert curated.exists()

    async def test_files_outside_a_day_directory_are_left_alone(self, manager, timelapse_root, timelapse_action):
        loose = timelapse_root / "back-room_12-00-00.jpg"
        loose.write_bytes(b"x" * 32)
        nested = timelapse_root / "archive" / "back-room_12-00-00.jpg"
        nested.parent.mkdir(parents=True)
        nested.write_bytes(b"x" * 32)
        await manager.prune_expired_timelapse()
        assert loose.exists()
        assert nested.exists()


class TestTimelapseSizeBound:
    async def test_the_oldest_frames_go_first(self, manager, db_client, timelapse_root, timelapse_action):
        oldest = _frame(timelapse_root, days_ago=3, size=400_000)
        middle = _frame(timelapse_root, days_ago=2, size=400_000)
        newest = _frame(timelapse_root, days_ago=1, size=400_000)
        # 1 MB cap against 1.2 MB on disk: eviction must stop as soon as it fits, so
        # exactly one frame goes rather than the whole oldest day.
        await db_client.set_setting(TIMELAPSE_MAX_TOTAL_MB_SETTING_KEY, "1")
        result = await manager.enforce_timelapse_size_cap()
        assert result["files"] == 1
        assert not oldest.exists()
        assert middle.exists() and newest.exists()

    async def test_within_the_cap_nothing_is_touched(self, manager, db_client, timelapse_root, timelapse_action):
        frame = _frame(timelapse_root, days_ago=1, size=1024)
        await db_client.set_setting(TIMELAPSE_MAX_TOTAL_MB_SETTING_KEY, "100")
        assert (await manager.enforce_timelapse_size_cap())["files"] == 0
        assert frame.exists()

    async def test_zero_mb_disables_the_cap(self, manager, db_client, timelapse_root, timelapse_action):
        frame = _frame(timelapse_root, days_ago=1, size=600_000)
        await db_client.set_setting(TIMELAPSE_MAX_TOTAL_MB_SETTING_KEY, "0")
        assert (await manager.enforce_timelapse_size_cap())["files"] == 0
        assert frame.exists()

    async def test_the_bounds_are_independent(self, manager, db_client, timelapse_root, timelapse_action):
        """Switching the age bound off must leave the size bound working, and vice versa."""
        old = _frame(timelapse_root, days_ago=40, size=600_000)
        newer = _frame(timelapse_root, days_ago=1, size=600_000)
        await db_client.set_setting(TIMELAPSE_RETENTION_SETTING_KEY, "0")
        await db_client.set_setting(TIMELAPSE_MAX_TOTAL_MB_SETTING_KEY, "1")
        await manager.prune_expired_timelapse()
        assert not old.exists()
        assert newer.exists()

    async def test_age_runs_before_the_cap(self, manager, db_client, timelapse_root, timelapse_action):
        """Otherwise the cap evicts frames a user still wants to reclaim bytes that the
        age bound was about to release anyway."""
        expired = _frame(timelapse_root, days_ago=40, size=1_200_000)
        keeper = _frame(timelapse_root, days_ago=1, size=600_000)
        await db_client.set_setting(TIMELAPSE_MAX_TOTAL_MB_SETTING_KEY, "1")
        result = await manager.prune_expired_timelapse()
        assert result["files"] == 1
        assert not expired.exists()
        assert keeper.exists()


class TestTemplatedTimelapseRoots:
    """A templated `output_dir` is what lets one action file four cameras separately, so
    the sweep has to find day folders that are no longer immediate children of the root.

    The root it bounds itself to is the template's *literal prefix*, not the expansion.
    Expanding against the current source list would look more precise and be worse: a
    camera renamed last month leaves a folder no expansion names, and its frames would
    then never be swept again.
    """

    @pytest.fixture
    async def templated_action(self, db_client):
        async with db_client._async_session_maker() as session:
            profile = WatchProfile(name="Templated")
            session.add(profile)
            await session.commit()
            await session.refresh(profile)
            session.add(
                WatchAction(
                    profile_uuid=profile.uuid,
                    kind="save_timelapse",
                    trigger="frame_analysed",
                    params_json={"output_dir": "timelapse/{source}"},
                )
            )
            await session.commit()
            return profile.uuid

    def _nested(self, root, camera, days_ago, name="cam_12-00-00.jpg", size=1024):
        day = (datetime.now().astimezone() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        path = root / camera / day / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        return path

    async def test_the_root_is_the_literal_prefix(self, manager, timelapse_root, templated_action):
        assert await manager.timelapse_roots() == [timelapse_root]

    async def test_frames_one_level_down_are_swept(self, manager, timelapse_root, templated_action):
        old = self._nested(timelapse_root, "back-room", days_ago=40)
        recent = self._nested(timelapse_root, "back-room", days_ago=1)
        result = await manager.prune_timelapse_by_age()
        assert result["files"] == 1
        assert not old.exists()
        assert recent.exists()

    async def test_a_renamed_cameras_old_folder_is_still_swept(self, manager, timelapse_root, templated_action):
        """The reason the prefix is used rather than the expansion."""
        orphaned = self._nested(timelapse_root, "the-old-name", days_ago=40)
        await manager.prune_timelapse_by_age()
        assert not orphaned.exists()

    async def test_a_camera_folder_left_empty_is_removed(self, manager, timelapse_root, templated_action):
        """Otherwise a retired camera keeps an empty shell forever."""
        old = self._nested(timelapse_root, "retired-cam", days_ago=40)
        await manager.prune_timelapse_by_age()
        assert not (timelapse_root / "retired-cam").exists()
        # Never the root itself: the action needs somewhere to write when it comes back.
        assert timelapse_root.exists()

    async def test_a_camera_folder_still_holding_frames_survives(self, manager, timelapse_root, templated_action):
        old = self._nested(timelapse_root, "busy-cam", days_ago=40)
        keep = self._nested(timelapse_root, "busy-cam", days_ago=1)
        await manager.prune_timelapse_by_age()
        assert not old.exists()
        assert keep.exists()
        assert (timelapse_root / "busy-cam").exists()

    async def test_a_date_first_layout_is_swept_too(self, manager, db_client, timelapse_root):
        """`timelapse/{date}/{source}` puts the day folder *above* the camera folder."""
        async with db_client._async_session_maker() as session:
            profile = WatchProfile(name="Date first")
            session.add(profile)
            await session.commit()
            await session.refresh(profile)
            session.add(
                WatchAction(
                    profile_uuid=profile.uuid,
                    kind="save_timelapse",
                    trigger="frame_analysed",
                    params_json={"output_dir": "timelapse/{date}/{source}"},
                )
            )
            await session.commit()

        day = (datetime.now().astimezone() - timedelta(days=40)).strftime("%Y-%m-%d")
        old = timelapse_root / day / "back-room" / "cam_12-00-00.jpg"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"x" * 512)

        assert (await manager.prune_timelapse_by_age())["files"] == 1
        assert not old.exists()

    async def test_a_file_with_no_day_folder_above_it_is_never_touched(self, manager, timelapse_root, templated_action):
        """The wider walk must not make the shape rails weaker."""
        stray = timelapse_root / "back-room" / "cam_12-00-00.jpg"
        stray.parent.mkdir(parents=True)
        stray.write_bytes(b"x" * 32)
        await manager.prune_expired_timelapse()
        assert stray.exists()

    async def test_a_template_with_no_fixed_folder_is_skipped(self, manager, db_client, timelapse_root):
        """Its only bound would be the dataset root, which is not a thing to sweep."""
        async with db_client._async_session_maker() as session:
            profile = WatchProfile(name="Rootless")
            session.add(profile)
            await session.commit()
            await session.refresh(profile)
            session.add(
                WatchAction(
                    profile_uuid=profile.uuid,
                    kind="save_timelapse",
                    trigger="frame_analysed",
                    params_json={"output_dir": "{source}/frames"},
                )
            )
            await session.commit()
        assert await manager.timelapse_roots() == []
