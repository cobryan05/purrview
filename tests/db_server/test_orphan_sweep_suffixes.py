"""The orphan sweep deletes by walking a directory, so what it will *not* touch matters.

It unlinks anything under a managed root that no row names and that is past the grace
period. Both of those rails are about the app's own files — a row it has not written
yet, a clip still being written — and neither says anything about files the app could
not have produced. A recordings root pointed somewhere it should not be therefore used
to cost the user every unrelated file under it, an hour later, silently.

Restricting the sweep to the suffixes this app actually writes does not make a
misconfigured root safe: `.jpg` and `.mp4` are exactly what a photo library holds, so
the documented warning still stands. It does mean a root aimed at a home directory or a
document folder no longer takes everything with it, which is the difference between a
bad hour and an unrecoverable one.
"""

import os
import time

import pytest

from apps.db_server.manager import RECORDING_SWEEPABLE_SUFFIXES, Manager


@pytest.fixture
def recording_roots(tmp_path):
    clips = tmp_path / "recordings" / "clips"
    clips.mkdir(parents=True)
    return (clips,)


@pytest.fixture
def manager(db_client, files_root, recording_roots):
    m = Manager(db_client=db_client, files_root=files_root)
    m._db_ready = True
    # Never let a directory-walking deleter loose on the real /data/recordings.
    m._recording_roots = recording_roots
    return m


def _aged(path, body=b"x"):
    """Write *path* and backdate it comfortably past the grace period."""
    path.write_bytes(body)
    old = time.time() - 86_400
    os.utime(path, (old, old))
    return path


class TestTheSweepOnlyDeletesWhatThisAppWrites:
    async def test_a_foreign_file_survives_a_sweep_that_takes_the_orphan_beside_it(self, manager, recording_roots):
        """The pair is the point: both are unreferenced, both are past the grace period,
        and they sit in the same directory. Only the suffix separates them, so a sweep
        that takes both has lost its only protection for a user's own data."""
        (clips,) = recording_roots
        orphan = _aged(clips / "left-behind.mp4")
        document = _aged(clips / "taxes.docx")
        database = _aged(clips / "notes.sqlite")

        result = await manager.sweep_orphan_recording_files()

        assert not orphan.exists(), "the sweep stopped collecting its own orphans"
        assert document.exists(), "the sweep deleted a file this app could never write"
        assert database.exists(), "the sweep deleted a file this app could never write"
        assert result["files"] == 1

    @pytest.mark.parametrize("suffix", sorted(RECORDING_SWEEPABLE_SUFFIXES))
    async def test_every_suffix_the_app_writes_is_still_collected(self, manager, recording_roots, suffix):
        """The allowlist is a leak the moment it falls behind what the writers emit: an
        orphan whose suffix is missing is never collected and the disk fills quietly.
        Parametrised over the constant so adding a suffix without a writer, or a writer
        without a suffix, is visible here rather than in a full disk."""
        (clips,) = recording_roots
        orphan = _aged(clips / f"left-behind{suffix}")

        await manager.sweep_orphan_recording_files()

        assert not orphan.exists(), f"an orphaned {suffix} file is never reclaimed"

    async def test_suffix_matching_ignores_case(self, manager, recording_roots):
        """A camera or an external tool writing `.MP4` must not strand a file forever."""
        (clips,) = recording_roots
        orphan = _aged(clips / "LEFT-BEHIND.MP4")

        await manager.sweep_orphan_recording_files()

        assert not orphan.exists()

    async def test_an_extensionless_file_is_left_alone(self, manager, recording_roots):
        """Nothing this app writes is extensionless, so one found here came from
        somewhere else — a lock file, a socket, something a user dropped in."""
        (clips,) = recording_roots
        stray = _aged(clips / "LICENSE")

        await manager.sweep_orphan_recording_files()

        assert stray.exists()
