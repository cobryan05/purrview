"""Event clips and card art are swept once they pass the retention window.

The NVR writes two kinds of file nobody ever deletes by hand: a clip per event and a
small snapshot per event. Everything comparable in this codebase is bounded — the
thumbnail cache by bytes, the frame ring by bytes, the rewind buffer by seconds — so
without this the only unbounded writer in the app is the one pointed at a camera 24
hours a day.

What these tests pin down is mostly the *edges*, because the happy path is one
comparison: that a bad setting cannot wedge the sweep (a stuck sweep is a full disk),
that "keep everything" is expressible, that files go with their rows, and that a clip
with no event still expires.
"""

from datetime import datetime, timedelta, timezone

import pytest

from apps.db_server.manager import (
    DEFAULT_RECORDING_RETENTION_DAYS,
    RECORDING_RETENTION_SETTING_KEY,
    Manager,
)
from apps.helpers.db.types import Recording, WatchEvent, WatchProfile


@pytest.fixture
def manager(db_client, files_root):
    m = Manager(db_client=db_client, files_root=files_root)
    m._db_ready = True
    return m


@pytest.fixture
async def profile_uuid(db_client):
    """`watch_events.profile_uuid` is a real FK, so events need a profile to hang off."""
    async with db_client._async_session_maker() as session:
        profile = WatchProfile(name="House cams")
        session.add(profile)
        await session.commit()
        await session.refresh(profile)
        return profile.uuid


def _ago(days: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


async def _event(db_client, tmp_path, profile_uuid, *, age_days: float, name: str) -> tuple[str, object]:
    """A watch event with a snapshot file on disk. Returns (uuid, snapshot path)."""
    snapshot = tmp_path / f"{name}.jpg"
    snapshot.write_bytes(b"\xff\xd8\xff")
    event = await db_client.create_watch_event(
        WatchEvent(
            profile_uuid=profile_uuid,
            rule_name=name,
            started_at=_ago(age_days),
            snapshot_path=str(snapshot),
        )
    )
    return event.uuid, snapshot


async def _recording(db_client, tmp_path, *, age_days: float, name: str, event_uuid=None):
    clip = tmp_path / f"{name}.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftyp")
    recording = await db_client.create_recording(
        Recording(
            file_path=str(clip),
            started_at=_ago(age_days),
            event_uuid=event_uuid,
            status="complete",
        )
    )
    return recording.uuid, clip


class TestTheRetentionSetting:
    async def test_defaults_to_thirty_days_when_unset(self, manager):
        assert await manager.get_recording_retention_days() == DEFAULT_RECORDING_RETENTION_DAYS
        assert DEFAULT_RECORDING_RETENTION_DAYS == 30

    async def test_reads_the_stored_value(self, manager, db_client):
        await db_client.set_setting(RECORDING_RETENTION_SETTING_KEY, "7")
        assert await manager.get_recording_retention_days() == 7

    @pytest.mark.parametrize("stored", ["", "   ", "not a number", "thirty"])
    async def test_an_unreadable_value_falls_back_rather_than_raising(self, manager, db_client, stored):
        """A hand-edited row must not be able to stop the sweep — that fills a disk."""
        await db_client.set_setting(RECORDING_RETENTION_SETTING_KEY, stored)
        assert await manager.get_recording_retention_days() == DEFAULT_RECORDING_RETENTION_DAYS

    async def test_a_float_is_truncated_not_rejected(self, manager, db_client):
        await db_client.set_setting(RECORDING_RETENTION_SETTING_KEY, "7.9")
        assert await manager.get_recording_retention_days() == 7


class TestTheSweep:
    async def test_an_expired_event_and_its_snapshot_are_removed(self, manager, db_client, tmp_path, profile_uuid):
        uuid, snapshot = await _event(db_client, tmp_path, profile_uuid, age_days=40, name="old")

        result = await manager.prune_expired_recordings()

        assert result["events"] == 1
        assert await db_client.get_watch_event(uuid) is None
        assert not snapshot.exists()

    async def test_a_recent_event_is_left_alone(self, manager, db_client, tmp_path, profile_uuid):
        uuid, snapshot = await _event(db_client, tmp_path, profile_uuid, age_days=3, name="recent")

        await manager.prune_expired_recordings()

        assert await db_client.get_watch_event(uuid) is not None
        assert snapshot.exists()

    async def test_the_boundary_is_the_configured_window(self, manager, db_client, tmp_path, profile_uuid):
        await db_client.set_setting(RECORDING_RETENTION_SETTING_KEY, "10")
        old, _ = await _event(db_client, tmp_path, profile_uuid, age_days=11, name="just-past")
        young, _ = await _event(db_client, tmp_path, profile_uuid, age_days=9, name="just-inside")

        await manager.prune_expired_recordings()

        assert await db_client.get_watch_event(old) is None
        assert await db_client.get_watch_event(young) is not None

    async def test_zero_keeps_everything(self, manager, db_client, tmp_path, profile_uuid):
        """There has to be a way to opt out that isn't deleting the setting."""
        await db_client.set_setting(RECORDING_RETENTION_SETTING_KEY, "0")
        uuid, snapshot = await _event(db_client, tmp_path, profile_uuid, age_days=9999, name="ancient")

        result = await manager.prune_expired_recordings()

        assert result == {"events": 0, "recordings": 0}
        assert await db_client.get_watch_event(uuid) is not None
        assert snapshot.exists()

    async def test_an_expired_event_takes_its_clip_with_it(self, manager, db_client, tmp_path, profile_uuid):
        """`delete_watch_event` already cascades — the sweep must not bypass that."""
        event_uuid, snapshot = await _event(db_client, tmp_path, profile_uuid, age_days=40, name="old")
        rec_uuid, clip = await _recording(db_client, tmp_path, age_days=40, name="old-clip", event_uuid=event_uuid)
        await db_client.update_watch_event(event_uuid, recording_uuid=rec_uuid)

        await manager.prune_expired_recordings()

        assert await db_client.get_recording(rec_uuid) is None
        assert not clip.exists()
        assert not snapshot.exists()

    async def test_a_clip_with_no_event_still_expires(self, manager, db_client, tmp_path):
        """Otherwise a hand-deleted event strands its clip on disk forever."""
        rec_uuid, clip = await _recording(db_client, tmp_path, age_days=40, name="orphan")

        result = await manager.prune_expired_recordings()

        assert result["recordings"] == 1
        assert await db_client.get_recording(rec_uuid) is None
        assert not clip.exists()

    async def test_a_recording_with_no_start_time_expires_on_its_created_at(self, manager, db_client, tmp_path):
        """NULL compares false against any cutoff, so it would otherwise never expire."""
        clip = tmp_path / "no-start.mp4"
        clip.write_bytes(b"\x00")
        recording = await db_client.create_recording(Recording(file_path=str(clip), started_at=None, status="error"))

        # created_at is now, so it survives a 30-day window but not a zero-length one.
        await manager.prune_expired_recordings()
        assert await db_client.get_recording(recording.uuid) is not None

        await manager.prune_expired_recordings(now=datetime.now(timezone.utc) + timedelta(days=31))
        assert await db_client.get_recording(recording.uuid) is None

    async def test_a_missing_file_does_not_stop_the_row_being_deleted(self, manager, db_client, tmp_path, profile_uuid):
        """The row is the index; a file already gone by other means is not an error."""
        uuid, snapshot = await _event(db_client, tmp_path, profile_uuid, age_days=40, name="gone")
        snapshot.unlink()

        result = await manager.prune_expired_recordings()

        assert result["events"] == 1
        assert await db_client.get_watch_event(uuid) is None


class TestTheSweepSchedule:
    async def test_the_first_call_runs_and_the_next_is_throttled(self, manager, db_client, tmp_path, profile_uuid):
        """Hourly, not per worker tick — the window is measured in days."""
        first, _ = await _event(db_client, tmp_path, profile_uuid, age_days=40, name="first")
        await manager._maybe_prune_expired_recordings()
        assert await db_client.get_watch_event(first) is None

        second, snapshot = await _event(db_client, tmp_path, profile_uuid, age_days=40, name="second")
        await manager._maybe_prune_expired_recordings()
        assert await db_client.get_watch_event(second) is not None
        assert snapshot.exists()

    async def test_a_failing_sweep_does_not_escape_the_worker_loop(self, manager, monkeypatch):
        """The same loop rebuilds the hash index; retention must not take it down."""

        async def boom():
            raise RuntimeError("db is on fire")

        monkeypatch.setattr(manager, "prune_expired_recordings", boom)
        await manager._maybe_prune_expired_recordings()  # must not raise
