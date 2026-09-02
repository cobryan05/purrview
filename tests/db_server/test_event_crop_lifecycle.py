"""An event's crop is a second file, and every path that handles the first must handle it.

The crop was added beside `snapshot_path` rather than replacing it, which means four
existing mechanisms each had a chance to know about only one of the two. Each failure is
silent and each is expensive:

* the **orphan sweep** deletes any file under a managed root that no row names, so a
  crop missing from `referenced_recording_paths` is deleted an hour after it is written
  — on every card, for ever, with nothing in the logs;
* **deleting an event** that only unlinks the snapshot leaves the crop behind, where
  only the orphan sweep will eventually find it;
* the **reconcile pass** clears the path of a file a user deleted by hand. Pairing the
  two images in one row would let a surviving snapshot hold a deleted crop's column
  open, leaving a card asking for a file nobody can produce again — and clearing both
  together would blank a card that still has a perfectly good picture;
* the **size cap** decrements a running total by what each deletion frees, so an event
  measured without its crop makes the sweep believe it freed less than it did and evict
  further than it needed to.
"""

import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from apps.db_server.manager import Manager
from apps.helpers.db.types import WatchEvent, WatchProfile

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
    # Never let a directory-walking deleter loose on the real /data/recordings.
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


async def _event_with_both(db_client, snapshots, profile_uuid, *, name="event", age_days=1.0):
    """An event carrying a wide snapshot and a crop, both on disk.

    ⚠️ **Both files are back-dated.** The orphan sweep exempts anything written inside
    `RECORDINGS_ORPHAN_GRACE_S`, because a file exists on disk before the row naming it
    does. A freshly written fixture is therefore skipped whatever the sweep believes
    about it — which silently made the "a referenced crop is never deleted" test pass
    against a `referenced_recording_paths` that had never heard of `crop_path`. Aging
    them is what makes that test able to fail.
    """
    snapshot = snapshots / f"{name}.jpg"
    crop = snapshots / f"{name}_crop.jpg"
    snapshot.write_bytes(b"\xff\xd8\xff" + b"\x00" * 400)
    crop.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
    aged = time.time() - 86_400
    for path in (snapshot, crop):
        os.utime(path, (aged, aged))
    event = await db_client.create_watch_event(
        WatchEvent(
            profile_uuid=profile_uuid,
            rule_name=name,
            started_at=datetime.now(timezone.utc) - timedelta(days=age_days),
            snapshot_path=str(snapshot),
            crop_path=str(crop),
        )
    )
    return event.uuid, snapshot, crop


class TestTheOrphanSweepSparesCrops:
    async def test_a_referenced_crop_is_never_deleted(self, manager, db_client, recording_roots, profile_uuid):
        """The sweep asks "does a row name this file?" and unlinks when the answer is no.
        A crop absent from that set is deleted on the next sweep, on every event."""
        _clips, snapshots = recording_roots
        _uuid, snapshot, crop = await _event_with_both(db_client, snapshots, profile_uuid)

        await manager.sweep_orphan_recording_files()

        assert crop.exists(), "the orphan sweep deleted card art a row points at"
        assert snapshot.exists()

    async def test_a_crop_no_row_names_is_still_swept(self, manager, recording_roots):
        """The other half of the contract — the sweep must not be neutered into
        skipping anything that merely looks like a crop."""
        _clips, snapshots = recording_roots
        stray = snapshots / "left-behind_crop.jpg"
        stray.write_bytes(b"\xff\xd8\xff")
        # Aged past the grace period, which exists because a file appears on disk before
        # the row naming it does.
        old = time.time() - 86_400
        os.utime(stray, (old, old))

        await manager.sweep_orphan_recording_files()

        assert not stray.exists()


class TestDeletingAnEvent:
    async def test_both_images_go_with_the_row(self, manager, db_client, recording_roots, profile_uuid):
        _clips, snapshots = recording_roots
        uuid, snapshot, crop = await _event_with_both(db_client, snapshots, profile_uuid)

        assert await manager.delete_watch_event(uuid) is True

        assert not snapshot.exists()
        assert not crop.exists(), "the crop outlived its event"

    async def test_an_event_with_no_crop_still_deletes(self, manager, db_client, recording_roots, profile_uuid):
        """Most of the archive has no crop and never will — the ones with no clip to
        backfill from. Deleting one must not depend on the second file existing."""
        _clips, snapshots = recording_roots
        snapshot = snapshots / "lone.jpg"
        snapshot.write_bytes(b"\xff\xd8\xff")
        event = await db_client.create_watch_event(
            WatchEvent(profile_uuid=profile_uuid, started_at=datetime.now(timezone.utc), snapshot_path=str(snapshot))
        )

        assert await manager.delete_watch_event(event.uuid) is True
        assert not snapshot.exists()


class TestReconcilingAHandDeletedImage:
    async def test_a_deleted_crop_clears_only_its_own_column(self, manager, db_client, recording_roots, profile_uuid):
        """The card falls back to the wide shot, which is still there. Clearing both
        would turn a recoverable state into an empty tile."""
        _clips, snapshots = recording_roots
        uuid, _snapshot, crop = await _event_with_both(db_client, snapshots, profile_uuid)
        crop.unlink()

        await manager.reconcile_missing_recording_files(now_ts=LATER)

        assert await db_client.get_watch_event_crop_path(uuid) is None
        assert await db_client.get_watch_event_snapshot_path(uuid) is not None
        assert await db_client.get_watch_event(uuid) is not None

    async def test_a_deleted_snapshot_leaves_the_crop_alone(self, manager, db_client, recording_roots, profile_uuid):
        """The mirror image, and the one that catches the two paths being paired into a
        single row: a surviving crop must not keep the snapshot's column pointing at a
        file that is gone."""
        _clips, snapshots = recording_roots
        uuid, snapshot, _crop = await _event_with_both(db_client, snapshots, profile_uuid)
        snapshot.unlink()

        await manager.reconcile_missing_recording_files(now_ts=LATER)

        assert await db_client.get_watch_event_snapshot_path(uuid) is None
        assert await db_client.get_watch_event_crop_path(uuid) is not None

    async def test_both_present_changes_nothing(self, manager, db_client, recording_roots, profile_uuid):
        _clips, snapshots = recording_roots
        uuid, _snapshot, _crop = await _event_with_both(db_client, snapshots, profile_uuid)

        result = await manager.reconcile_missing_recording_files(now_ts=LATER)

        assert result["snapshots"] == 0
        assert await db_client.get_watch_event_crop_path(uuid) is not None


class TestServingTheCrop:
    async def test_the_endpoint_serves_the_file(self, http_client, db_client, recording_roots, profile_uuid):
        _clips, snapshots = recording_roots
        uuid, _snapshot, crop = await _event_with_both(db_client, snapshots, profile_uuid)

        response = await http_client.get(f"/api/watch-events/{uuid}/crop")

        assert response.status_code == 200
        assert response.content == crop.read_bytes()

    async def test_an_event_with_no_crop_404s(self, http_client, db_client, profile_uuid):
        """Ordinary, not an error — the card falls back. Asserted so the endpoint cannot
        start answering 200 with the wide shot, which would defeat the fallback by
        making "no crop" indistinguishable from "here is a crop"."""
        event = await db_client.create_watch_event(
            WatchEvent(profile_uuid=profile_uuid, started_at=datetime.now(timezone.utc))
        )

        assert (await http_client.get(f"/api/watch-events/{event.uuid}/crop")).status_code == 404

    async def test_a_stale_path_404s_rather_than_500ing(self, http_client, db_client, recording_roots, profile_uuid):
        """Retention deletes files on its own schedule; a card must degrade, not error."""
        _clips, snapshots = recording_roots
        uuid, _snapshot, crop = await _event_with_both(db_client, snapshots, profile_uuid)
        crop.unlink()

        assert (await http_client.get(f"/api/watch-events/{uuid}/crop")).status_code == 404

    async def test_the_read_model_reports_whether_there_is_one(
        self, http_client, db_client, recording_roots, profile_uuid
    ):
        """`hasCrop` is what stops every card in a no-crop grid making a doomed request.
        The path itself must stay off the wire — it is a server filesystem path."""
        _clips, snapshots = recording_roots
        with_crop, _s, _c = await _event_with_both(db_client, snapshots, profile_uuid, name="cropped")
        bare = await db_client.create_watch_event(
            WatchEvent(profile_uuid=profile_uuid, started_at=datetime.now(timezone.utc))
        )

        response = await http_client.get("/api/watch-events", params={"limit": 50})
        events = {e["uuid"]: e for e in response.json()["events"]}

        assert events[with_crop]["hasCrop"] is True
        assert events[bare.uuid]["hasCrop"] is False
        assert "cropPath" not in events[with_crop]


class TestTheSizeCap:
    async def test_an_events_measured_size_includes_its_crop(self, manager, db_client, recording_roots, profile_uuid):
        """Eviction decrements a running total by this figure and stops when it is back
        under the cap, so undercounting evicts more history than it needed to."""
        _clips, snapshots = recording_roots
        uuid, snapshot, crop = await _event_with_both(db_client, snapshots, profile_uuid)
        sizes = {
            str(snapshot.resolve()): snapshot.stat().st_size,
            str(crop.resolve()): crop.stat().st_size,
        }

        measured = await manager._event_disk_usage(uuid, sizes)

        assert measured == snapshot.stat().st_size + crop.stat().st_size
