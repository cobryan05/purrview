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

import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from apps.db_server.manager import (
    DEFAULT_RECORDING_RETENTION_DAYS,
    RECORDING_MAX_TOTAL_MB_SETTING_KEY,
    RECORDING_RETENTION_SETTING_KEY,
    Manager,
)
from apps.helpers.db.types import Recording, WatchEvent, WatchProfile


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
    # Never let a directory-walking deleter loose on the real /data/recordings.
    m._recording_roots = recording_roots
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

    async def test_zero_disables_the_age_bound_only(self, manager, db_client, tmp_path, profile_uuid):
        """Both bounds are independent — switching one off must not switch off the other.

        Asserted through `prune_recordings_by_age` rather than the full pass, because the
        full pass still runs the size cap and the orphan sweep, which is the point.
        """
        await db_client.set_setting(RECORDING_RETENTION_SETTING_KEY, "0")
        uuid, snapshot = await _event(db_client, tmp_path, profile_uuid, age_days=9999, name="ancient")

        result = await manager.prune_recordings_by_age()

        assert result == {"events": 0, "recordings": 0}
        assert await db_client.get_watch_event(uuid) is not None
        assert snapshot.exists()

    async def test_an_expired_event_takes_its_clip_with_it(self, manager, db_client, tmp_path, profile_uuid):
        """`delete_watch_event` already cascades — the sweep must not bypass that."""
        event_uuid, snapshot = await _event(db_client, tmp_path, profile_uuid, age_days=40, name="old")
        rec_uuid, clip = await _recording(db_client, tmp_path, age_days=40, name="old-clip", event_uuid=event_uuid)

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


class TestTheOrphanSweep:
    """Files under the recordings roots that no row points at.

    Rows normally take their files with them, but a row can vanish without passing
    through that path — deleting a watch profile cascades to its events in the database
    and leaves their snapshots behind, and a crashed ffmpeg leaves a `.part` nothing ever
    referenced. Nothing looks at those files again, so without a sweep they are permanent.
    """

    def _orphan(self, root, name="stray.mp4", age_s=10_000):
        path = root / name
        path.write_bytes(b"x" * 2048)
        os.utime(path, (time.time() - age_s, time.time() - age_s))
        return path

    async def test_a_file_no_row_points_at_is_removed(self, manager, recording_roots):
        clips, _snapshots = recording_roots
        stray = self._orphan(clips)

        result = await manager.sweep_orphan_recording_files()

        assert result["files"] == 1
        assert result["bytes"] == 2048
        assert not stray.exists()

    async def test_a_referenced_file_is_never_touched(self, manager, db_client, recording_roots):
        """The safety rail. Deleting a live clip is the failure mode here."""
        clips, _snapshots = recording_roots
        live = self._orphan(clips, "live.mp4")
        await db_client.create_recording(Recording(file_path=str(live), started_at=_ago(0.1), status="complete"))

        result = await manager.sweep_orphan_recording_files()

        assert result["files"] == 0
        assert live.exists()

    async def test_a_referenced_snapshot_is_never_touched(self, manager, db_client, recording_roots, profile_uuid):
        _clips, snapshots = recording_roots
        live = self._orphan(snapshots, "card.jpg")
        await db_client.create_watch_event(
            WatchEvent(profile_uuid=profile_uuid, started_at=_ago(0.1), snapshot_path=str(live))
        )

        await manager.sweep_orphan_recording_files()

        assert live.exists()

    async def test_a_freshly_written_file_is_left_alone(self, manager, recording_roots):
        """A clip exists on disk before the row referencing it does — without the grace
        period the sweep races clip extraction and deletes its output mid-write."""
        clips, _snapshots = recording_roots
        fresh = self._orphan(clips, "in-flight.mp4", age_s=0)

        result = await manager.sweep_orphan_recording_files()

        assert result["files"] == 0
        assert fresh.exists()

    async def test_nested_day_folders_are_swept(self, manager, recording_roots):
        """Clips are written under a per-day folder, so a flat listdir would miss them."""
        clips, _snapshots = recording_roots
        day = clips / "2026-01-01"
        day.mkdir()
        stray = self._orphan(day, "old.mp4")

        result = await manager.sweep_orphan_recording_files()

        assert result["files"] == 1
        assert not stray.exists()

    async def test_files_outside_the_managed_roots_are_untouched(self, manager, tmp_path):
        """A hand-set `output_dir` is the user's own space, not the sweep's."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        mine = elsewhere / "mine.mp4"
        mine.write_bytes(b"x")
        os.utime(mine, (0, 0))

        result = await manager.sweep_orphan_recording_files()

        assert result["files"] == 0
        assert mine.exists()

    async def test_a_missing_root_is_not_an_error(self, manager, tmp_path):
        manager._recording_roots = (tmp_path / "nope", tmp_path / "also-nope")
        assert await manager.sweep_orphan_recording_files() == {"files": 0, "bytes": 0}


class TestTheSizeCap:
    async def test_the_oldest_events_are_evicted_until_under_the_cap(
        self, manager, db_client, recording_roots, profile_uuid
    ):
        clips, _snapshots = recording_roots
        await db_client.set_setting(RECORDING_MAX_TOTAL_MB_SETTING_KEY, "1")

        # Six events, ~512 KB of clip each: comfortably over a 1 MB cap.
        uuids = []
        for i in range(6):
            clip = clips / f"clip-{i}.mp4"
            clip.write_bytes(b"x" * (512 * 1024))
            event = await db_client.create_watch_event(
                WatchEvent(profile_uuid=profile_uuid, started_at=_ago(30 - i), rule_name=f"e{i}")
            )
            recording = await db_client.create_recording(
                Recording(file_path=str(clip), started_at=_ago(30 - i), event_uuid=event.uuid, status="complete")
            )
            uuids.append(event.uuid)

        result = await manager.enforce_recording_size_cap()

        assert result["events"] > 0
        # Oldest first: index 0 is the oldest, so it must go before the newest does.
        assert await db_client.get_watch_event(uuids[0]) is None
        assert await db_client.get_watch_event(uuids[-1]) is not None
        remaining = sum(p.stat().st_size for p in clips.rglob("*") if p.is_file())
        assert remaining <= 1024 * 1024

    async def test_a_cap_of_zero_evicts_nothing(self, manager, db_client, recording_roots, profile_uuid):
        clips, _snapshots = recording_roots
        await db_client.set_setting(RECORDING_MAX_TOTAL_MB_SETTING_KEY, "0")
        big = clips / "huge.mp4"
        big.write_bytes(b"x" * (2 * 1024 * 1024))
        event = await db_client.create_watch_event(WatchEvent(profile_uuid=profile_uuid, started_at=_ago(1)))
        await db_client.create_recording(
            Recording(file_path=str(big), started_at=_ago(1), event_uuid=event.uuid, status="complete")
        )

        result = await manager.enforce_recording_size_cap()

        assert result == {"events": 0, "recordings": 0, "bytes": 0}
        assert await db_client.get_watch_event(event.uuid) is not None

    async def test_under_the_cap_nothing_is_evicted(self, manager, db_client, recording_roots, profile_uuid):
        clips, _snapshots = recording_roots
        await db_client.set_setting(RECORDING_MAX_TOTAL_MB_SETTING_KEY, "100")
        (clips / "small.mp4").write_bytes(b"x" * 1024)
        event = await db_client.create_watch_event(WatchEvent(profile_uuid=profile_uuid, started_at=_ago(1)))

        result = await manager.enforce_recording_size_cap()

        assert result["events"] == 0
        assert await db_client.get_watch_event(event.uuid) is not None

    async def test_being_over_cap_with_nothing_deletable_terminates(self, manager, db_client, recording_roots):
        """A fresh orphan is over the cap but exempt from both sweeps — this must report
        and stop rather than spin the worker."""
        clips, _snapshots = recording_roots
        await db_client.set_setting(RECORDING_MAX_TOTAL_MB_SETTING_KEY, "1")
        (clips / "fresh-and-huge.mp4").write_bytes(b"x" * (2 * 1024 * 1024))

        result = await manager.enforce_recording_size_cap()

        assert result["events"] == 0
        assert result["recordings"] == 0

    async def test_orphan_recordings_are_evicted_when_no_events_remain(self, manager, db_client, recording_roots):
        clips, _snapshots = recording_roots
        await db_client.set_setting(RECORDING_MAX_TOTAL_MB_SETTING_KEY, "1")
        rec_uuids = []
        for i in range(4):
            clip = clips / f"orphan-{i}.mp4"
            clip.write_bytes(b"x" * (512 * 1024))
            rec = await db_client.create_recording(
                Recording(file_path=str(clip), started_at=_ago(10 - i), status="complete")
            )
            rec_uuids.append(rec.uuid)

        result = await manager.enforce_recording_size_cap()

        assert result["recordings"] > 0
        assert await db_client.get_recording(rec_uuids[0]) is None

    async def test_a_recording_outside_the_roots_counts_toward_the_cap(
        self, manager, db_client, tmp_path, profile_uuid
    ):
        """Deletion scope and accounting scope have to agree.

        An action pointed at an absolute `output_dir` writes outside the managed roots.
        Age pruning already deletes those files — it goes through the row, which unlinks
        whatever `file_path` names. The size cap measured by walking the roots only, so
        the same file was invisible to it: it could never push the total over the cap,
        and a camera writing there filled the disk while the sweep reported it was
        comfortably under. Counting a file the sweep is already willing to delete costs
        no new authority.
        """
        external = tmp_path / "nas" / "timelapses"
        external.mkdir(parents=True)
        clip = external / "visit.mp4"
        clip.write_bytes(b"x" * (2 * 1024 * 1024))
        await db_client.set_setting(RECORDING_MAX_TOTAL_MB_SETTING_KEY, "1")
        event = await db_client.create_watch_event(
            WatchEvent(profile_uuid=profile_uuid, started_at=_ago(5), rule_name="outside")
        )
        await db_client.create_recording(
            Recording(file_path=str(clip), started_at=_ago(5), event_uuid=event.uuid, status="complete")
        )

        result = await manager.enforce_recording_size_cap()

        assert result["events"] == 1
        assert await db_client.get_watch_event(event.uuid) is None
        assert not clip.exists()

    async def test_an_unreferenced_file_beside_it_is_neither_counted_nor_deleted(self, manager, db_client, tmp_path):
        """The rail that makes counting external files safe: rows, never directories.

        The reason the sweep refuses to walk a hand-set `output_dir` is that the user
        chose it for their own reasons and it holds their own files. Reaching those by
        walking would be the failure that makes directory sweeps dangerous. Reaching our
        own by following a row is not the same act, and this is the test that keeps the
        two apart.
        """
        external = tmp_path / "nas" / "timelapses"
        external.mkdir(parents=True)
        theirs = external / "holiday-video.mp4"
        theirs.write_bytes(b"x" * (4 * 1024 * 1024))
        await db_client.set_setting(RECORDING_MAX_TOTAL_MB_SETTING_KEY, "1")

        result = await manager.enforce_recording_size_cap()

        assert result == {"events": 0, "recordings": 0, "bytes": 0}
        assert theirs.exists()


class TestTheWholePass:
    async def test_all_three_phases_report_together(self, manager, db_client, tmp_path, recording_roots, profile_uuid):
        clips, _snapshots = recording_roots
        stray = clips / "stray.mp4"
        stray.write_bytes(b"x" * 128)
        os.utime(stray, (0, 0))
        old, _ = await _event(db_client, tmp_path, profile_uuid, age_days=40, name="old")

        result = await manager.prune_expired_recordings()

        assert result["events"] == 1
        assert result["orphanFiles"] == 1
        assert result["bytes"] >= 128
        assert await db_client.get_watch_event(old) is None
        assert not stray.exists()

    async def test_the_orphan_sweep_runs_even_with_the_age_bound_off(self, manager, db_client, recording_roots):
        """Orphans are wasted bytes regardless of how long anything is kept for."""
        clips, _snapshots = recording_roots
        await db_client.set_setting(RECORDING_RETENTION_SETTING_KEY, "0")
        stray = clips / "stray.mp4"
        stray.write_bytes(b"x" * 64)
        os.utime(stray, (0, 0))

        result = await manager.prune_expired_recordings()

        assert result["orphanFiles"] == 1
        assert not stray.exists()


class TestTheOverlaySidecar:
    """A clip's detection sidecar must live and die with the clip.

    It is a file under the recordings roots, which makes it the orphan sweep's business
    by default: without `overlay_path` on the row it would be deleted about an hour after
    it was written, and the failure would look like an overlay feature that works right
    after an event and mysteriously stops later. And in the other direction, a sidecar
    left behind by a deleted clip is dead weight nothing will ever read.
    """

    async def test_a_sidecar_the_row_names_survives_the_orphan_sweep(self, manager, db_client, recording_roots):
        clips, _snapshots = recording_roots
        clip = clips / "live.mp4"
        clip.write_bytes(b"\x00" * 64)
        sidecar = clips / "live.mp4.boxes.json"
        sidecar.write_text('{"version": 1, "frames": []}')
        os.utime(sidecar, (0, 0))
        await db_client.create_recording(
            Recording(
                file_path=str(clip),
                overlay_path=str(sidecar),
                started_at=_ago(0.1),
                status="complete",
            )
        )

        result = await manager.sweep_orphan_recording_files()

        assert result["files"] == 0
        assert sidecar.exists()

    async def test_an_unreferenced_sidecar_is_swept(self, manager, recording_roots):
        clips, _snapshots = recording_roots
        stray = clips / "gone.mp4.boxes.json"
        stray.write_text('{"version": 1, "frames": []}')
        os.utime(stray, (0, 0))

        result = await manager.sweep_orphan_recording_files()

        assert result["files"] == 1
        assert not stray.exists()

    async def test_deleting_a_recording_deletes_its_sidecar(self, manager, db_client, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00" * 16)
        sidecar = tmp_path / "clip.mp4.boxes.json"
        sidecar.write_text("{}")
        recording = await db_client.create_recording(
            Recording(file_path=str(clip), overlay_path=str(sidecar), started_at=_ago(0.1), status="complete")
        )

        assert await manager.delete_recording(recording.uuid)

        assert not clip.exists()
        assert not sidecar.exists()

    async def test_expiry_takes_the_sidecar_with_the_clip(self, manager, db_client, tmp_path, profile_uuid):
        """The end-to-end shape: retention deletes the event, which deletes the
        recording, which must delete all three files."""
        event_uuid, snapshot = await _event(db_client, tmp_path, profile_uuid, age_days=40, name="old")
        clip = tmp_path / "old.mp4"
        clip.write_bytes(b"\x00" * 16)
        sidecar = tmp_path / "old.mp4.boxes.json"
        sidecar.write_text("{}")
        recording = await db_client.create_recording(
            Recording(
                file_path=str(clip),
                overlay_path=str(sidecar),
                started_at=_ago(40),
                event_uuid=event_uuid,
                status="complete",
            )
        )
        await manager.prune_expired_recordings()

        assert not snapshot.exists()
        assert not clip.exists()
        assert not sidecar.exists()

    async def test_a_surviving_sidecar_does_not_keep_a_fileless_row_alive(self, manager, db_client, recording_roots):
        """The reverse reconcile deliberately ignores `overlay_path`. A sidecar outliving
        a hand-deleted mp4 must not hold the row open — that leaves a card whose video
        404s, which is worse than losing the row."""
        clips, _snapshots = recording_roots
        sidecar = clips / "vanished.mp4.boxes.json"
        sidecar.write_text("{}")
        recording = await db_client.create_recording(
            Recording(
                file_path=str(clips / "vanished.mp4"),
                overlay_path=str(sidecar),
                started_at=_ago(0.5),
                created_at=_ago(0.5),
                status="complete",
            )
        )

        await manager.reconcile_missing_recording_files()

        assert await db_client.get_recording(recording.uuid) is None


class TestBlipsOutliveTheirClip:
    """A blip is a finding about a clip, and it is kept after the clip is swept.

    That is only tolerable if it stops *presenting itself as reviewable* at the same
    moment. A blip still reading `new` after its video has gone offers a harvest that
    can never find frames and a "play the clip" button that opens a dead player.
    """

    async def _blip(self, db_client, event_uuid, recording_uuid, **over):
        blips = await db_client.replace_event_blips(
            event_uuid,
            [{"kind": "detector_dropout", "fault": "detector", "harvestable": True, **over}],
            recording_uuid=recording_uuid,
        )
        return blips[0].uuid

    async def test_deleting_a_clip_expires_the_blips_found_in_it(self, manager, db_client, tmp_path, profile_uuid):
        event_uuid, _snapshot = await _event(db_client, tmp_path, profile_uuid, age_days=1, name="visit")
        recording_uuid, _clip = await _recording(db_client, tmp_path, age_days=1, name="visit", event_uuid=event_uuid)
        blip_uuid = await self._blip(db_client, event_uuid, recording_uuid)

        assert await manager.delete_recording(recording_uuid)

        blips, _total = await db_client.list_event_blips(uuids=[blip_uuid])
        assert blips[0].state == "expired"

    async def test_a_triaged_blip_keeps_its_outcome(self, manager, db_client, tmp_path, profile_uuid):
        """`harvested` and `dismissed` are decisions a human made. Losing the video does
        not un-make them, and overwriting them with `expired` would."""
        event_uuid, _snapshot = await _event(db_client, tmp_path, profile_uuid, age_days=1, name="visit")
        recording_uuid, _clip = await _recording(db_client, tmp_path, age_days=1, name="visit", event_uuid=event_uuid)
        blip_uuid = await self._blip(db_client, event_uuid, recording_uuid)
        await db_client.set_event_blip_states([blip_uuid], "harvested")

        await manager.delete_recording(recording_uuid)

        blips, _total = await db_client.list_event_blips(uuids=[blip_uuid])
        assert blips[0].state == "harvested"

    async def test_a_blip_on_another_clip_of_the_same_event_is_untouched(
        self, manager, db_client, tmp_path, profile_uuid
    ):
        """An event owns a clip *and* a timelapse. Sweeping one must not expire findings
        whose own evidence is still on disk."""
        event_uuid, _snapshot = await _event(db_client, tmp_path, profile_uuid, age_days=1, name="visit")
        swept_uuid, _a = await _recording(db_client, tmp_path, age_days=1, name="clip", event_uuid=event_uuid)
        kept_uuid, _b = await _recording(db_client, tmp_path, age_days=1, name="lapse", event_uuid=event_uuid)
        await self._blip(db_client, event_uuid, swept_uuid)
        kept_blip = await self._blip(db_client, event_uuid, kept_uuid)

        await manager.delete_recording(swept_uuid)

        blips, _total = await db_client.list_event_blips(uuids=[kept_blip])
        assert blips[0].state == "new"
