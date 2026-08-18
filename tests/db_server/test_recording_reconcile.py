"""Rows whose media file was deleted outside the app are cleared.

The orphan sweep answers "is there a row for this file?". This is the other direction —
"is there a file for this row?" — which is what a user tidying up in a file manager
creates. Neither direction repairs the other's damage: an orphan file wastes disk, a row
with no file is an event card whose player never loads.

Most of this file is about what the reconciler must *not* do, because the failure mode
is deleting a month of event history over a volume that mounted a second late.
"""

import time
from datetime import datetime, timedelta, timezone

import pytest

from apps.db_server.manager import RECONCILE_GUARD_MIN_ROWS, Manager
from apps.helpers.db.types import Recording, WatchEvent, WatchProfile

#: Comfortably past `RECORDINGS_ORPHAN_GRACE_S`, so a freshly written row is eligible.
LATER = time.time() + 86_400


@pytest.fixture
def recording_roots(tmp_path):
    clips = tmp_path / "recordings" / "clips"
    snapshots = tmp_path / "recordings" / "event_snapshots"
    clips.mkdir(parents=True)
    snapshots.mkdir(parents=True)
    return clips, snapshots


@pytest.fixture
def manager(db_client, files_root, recording_roots):
    m = Manager(db_client=db_client, files_root=files_root)
    m._db_ready = True
    # Never let this loose on the real /data/recordings because a test called it.
    m._recording_roots = recording_roots
    return m


@pytest.fixture
async def profile_uuid(db_client):
    async with db_client._async_session_maker() as session:
        profile = WatchProfile(name="House cams")
        session.add(profile)
        await session.commit()
        await session.refresh(profile)
        return profile.uuid


async def _recording(db_client, clips, *, name="clip", status="complete", age_days=1.0):
    path = clips / f"{name}.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftyp")
    recording = await db_client.create_recording(
        Recording(
            file_path=str(path),
            status=status,
            started_at=datetime.now(timezone.utc) - timedelta(days=age_days),
        )
    )
    return recording.uuid, path


async def _event_with_snapshot(db_client, snapshots, profile_uuid, *, name="event"):
    path = snapshots / f"{name}.jpg"
    path.write_bytes(b"\xff\xd8\xff")
    event = await db_client.create_watch_event(
        WatchEvent(
            profile_uuid=profile_uuid,
            rule_name=name,
            started_at=datetime.now(timezone.utc) - timedelta(days=1),
            snapshot_path=str(path),
        )
    )
    return event.uuid, path


class TestReconcilingDeletedMedia:
    async def test_a_row_whose_clip_was_deleted_is_dropped(self, manager, db_client, recording_roots):
        clips, _snapshots = recording_roots
        uuid, path = await _recording(db_client, clips)
        path.unlink()

        result = await manager.reconcile_missing_recording_files(now_ts=LATER)

        assert result["recordings"] == 1
        assert await db_client.get_recording(uuid) is None

    async def test_a_row_whose_clip_is_present_is_untouched(self, manager, db_client, recording_roots):
        clips, _snapshots = recording_roots
        uuid, _path = await _recording(db_client, clips)

        assert (await manager.reconcile_missing_recording_files(now_ts=LATER))["recordings"] == 0
        assert await db_client.get_recording(uuid) is not None

    async def test_a_deleted_snapshot_clears_the_path_but_keeps_the_event(
        self, manager, db_client, recording_roots, profile_uuid
    ):
        """The event is the record that something happened, which stays true once its
        picture is gone. Deleting it would lose history to a tidy-up."""
        _clips, snapshots = recording_roots
        uuid, path = await _event_with_snapshot(db_client, snapshots, profile_uuid)
        path.unlink()

        result = await manager.reconcile_missing_recording_files(now_ts=LATER)

        assert result["snapshots"] == 1
        assert await db_client.get_watch_event(uuid) is not None
        assert await db_client.get_watch_event_snapshot_path(uuid) is None


class TestReconcileSafety:
    async def test_an_in_progress_clip_is_exempt(self, manager, db_client, recording_roots):
        """A `writing` row is supposed to have no file yet — reaping it would race the
        extraction that is about to produce one."""
        clips, _snapshots = recording_roots
        uuid, path = await _recording(db_client, clips, status="writing")
        path.unlink()

        assert (await manager.reconcile_missing_recording_files(now_ts=LATER))["recordings"] == 0
        assert await db_client.get_recording(uuid) is not None

    async def test_a_fresh_row_is_inside_the_grace_period(self, manager, db_client, recording_roots):
        """A row is written around the time its file appears, and the order varies."""
        clips, _snapshots = recording_roots
        uuid, path = await _recording(db_client, clips)
        path.unlink()

        # Real "now", so the row is younger than the grace period.
        assert (await manager.reconcile_missing_recording_files())["recordings"] == 0
        assert await db_client.get_recording(uuid) is not None

    async def test_a_clip_outside_the_managed_roots_is_left_alone(self, manager, db_client, tmp_path):
        """A custom `output_dir` on a drive that happens to be unplugged must not cost
        the row — the same rule the rest of retention follows."""
        elsewhere = tmp_path / "elsewhere" / "clip.mp4"
        elsewhere.parent.mkdir(parents=True)
        recording = await db_client.create_recording(Recording(file_path=str(elsewhere), status="complete"))

        assert (await manager.reconcile_missing_recording_files(now_ts=LATER))["recordings"] == 0
        assert await db_client.get_recording(recording.uuid) is not None

    async def test_nothing_happens_when_no_managed_root_exists(self, manager, db_client, recording_roots, tmp_path):
        """The unmounted case at its clearest: every row would look missing."""
        clips, _snapshots = recording_roots
        uuid, _path = await _recording(db_client, clips)
        manager._recording_roots = (tmp_path / "gone",)

        assert (await manager.reconcile_missing_recording_files(now_ts=LATER))["recordings"] == 0
        assert await db_client.get_recording(uuid) is not None

    async def test_a_mass_disappearance_trips_the_circuit_breaker(self, manager, db_client, recording_roots):
        """Most rows losing their file at once is an unavailable volume, not a user who
        deleted almost everything. Stale rows are recoverable; deleted history is not."""
        clips, _snapshots = recording_roots
        uuids = []
        for i in range(RECONCILE_GUARD_MIN_ROWS + 2):
            uuid, path = await _recording(db_client, clips, name=f"clip-{i}")
            path.unlink()
            uuids.append(uuid)

        result = await manager.reconcile_missing_recording_files(now_ts=LATER)

        assert result == {"recordings": 0, "snapshots": 0}
        for uuid in uuids:
            assert await db_client.get_recording(uuid) is not None

    async def test_a_few_deletions_among_many_still_reconcile(self, manager, db_client, recording_roots):
        """The breaker must not make ordinary tidying-up a no-op."""
        clips, _snapshots = recording_roots
        kept = [await _recording(db_client, clips, name=f"keep-{i}") for i in range(RECONCILE_GUARD_MIN_ROWS)]
        gone_uuid, gone_path = await _recording(db_client, clips, name="gone")
        gone_path.unlink()

        result = await manager.reconcile_missing_recording_files(now_ts=LATER)

        assert result["recordings"] == 1
        assert await db_client.get_recording(gone_uuid) is None
        for uuid, _path in kept:
            assert await db_client.get_recording(uuid) is not None
