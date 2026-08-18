"""Unit tests for the clip drain — the thing that makes an event longer than the buffer work.

The failure this exists to prevent is silent: a rule's `max_duration_s` is a free number
in the editor, the rewind buffer is a fixed RAM window, and cutting only at the closing
edge quietly keeps whatever survived. A ten-minute event against a 45 s buffer produced a
45 s clip and no error.

Everything here is pure — the extract call is a stub — because the real bugs are cursor
arithmetic and eviction bookkeeping, and those are only cheap to find without a relay.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from apps.helpers.recordings.clipDrain import (
    MIN_DRAIN_INTERVAL_S,
    SEGMENT_LAG_S,
    ClipDrain,
    DrainedChunk,
    sweep_chunk_root,
)

T0 = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)


def _at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


class _FakeExtract:
    """Records what was asked for, and writes a file so the chunk looks real."""

    def __init__(self, fail_with: Exception = None) -> None:
        self.calls = []
        self.fail_with = fail_with

    async def __call__(self, start: datetime, duration_s: float, dest: Path) -> dict:
        self.calls.append((start, duration_s, dest))
        if self.fail_with is not None:
            raise self.fail_with
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"video")
        return {"path": str(dest), "sizeBytes": 5}


def _drain(tmp_path, retention_s=45.0, **kwargs) -> tuple:
    extract = kwargs.pop("extract", None) or _FakeExtract()
    drain = ClipDrain(
        source_uuid="src-1",
        retention_s=retention_s,
        extract=extract,
        chunk_root=tmp_path,
        **kwargs,
    )
    return drain, extract


class TestIdleCostsNothing:
    """Most events close inside the buffer and must never touch this path."""

    def test_a_drain_with_no_open_event_is_never_due(self, tmp_path):
        drain, _ = _drain(tmp_path)
        assert drain.active is False
        assert drain.due(_at(600)) is False

    async def test_draining_while_idle_does_nothing(self, tmp_path):
        drain, extract = _drain(tmp_path)
        assert await drain.drain(_at(600)) is None
        assert extract.calls == []

    def test_note_idle_on_an_idle_drain_is_a_no_op(self, tmp_path):
        drain, _ = _drain(tmp_path)
        drain.note_idle()
        assert drain.active is False


class TestCursor:
    def test_the_cursor_starts_ahead_of_the_event_to_cover_pre_roll(self, tmp_path):
        """Pre-roll sits in front of the event, so starting at the event's own start
        would let exactly the pre-roll age out — the part that cannot be re-derived."""
        drain, extract = _drain(tmp_path, lead_s=10.0)
        drain.note_open(_at(100))
        assert drain.active

        # Due once a full interval of material has accumulated behind the lag.
        assert drain.due(_at(100)) is False

    async def test_the_first_chunk_begins_at_event_start_minus_lead(self, tmp_path):
        drain, extract = _drain(tmp_path, lead_s=10.0)
        drain.note_open(_at(100))
        await drain.drain(_at(130))
        start, _duration, _dest = extract.calls[0]
        assert start == _at(90)

    async def test_chunks_are_contiguous(self, tmp_path):
        """Any hole between chunks is lost video, so the cursor must resume exactly
        where the previous chunk ended."""
        drain, extract = _drain(tmp_path, lead_s=0.0)
        drain.note_open(_at(0))
        first = await drain.drain(_at(30))
        second = await drain.drain(_at(60))
        assert first.end == second.start
        assert first.gap_before is False and second.gap_before is False

    async def test_a_drain_stays_one_segment_behind_now(self, tmp_path):
        """The current segment is still being written; asking for it returns a partial."""
        drain, extract = _drain(tmp_path, lead_s=0.0)
        drain.note_open(_at(0))
        chunk = await drain.drain(_at(30))
        assert chunk.end == _at(30 - SEGMENT_LAG_S)

    async def test_a_drain_with_nothing_new_yet_writes_no_chunk(self, tmp_path):
        drain, extract = _drain(tmp_path, lead_s=0.0)
        drain.note_open(_at(0))
        assert await drain.drain(_at(SEGMENT_LAG_S - 1)) is None
        assert extract.calls == []


class TestDueness:
    def test_the_interval_is_derived_from_the_retention_window(self, tmp_path):
        """Tied to the window rather than fixed, so it stays correct if retention changes."""
        drain, _ = _drain(tmp_path, retention_s=45.0)
        assert drain.interval_s == pytest.approx(15.0)

    def test_a_tiny_retention_cannot_produce_a_busy_loop(self, tmp_path):
        drain, _ = _drain(tmp_path, retention_s=3.0)
        assert drain.interval_s == MIN_DRAIN_INTERVAL_S

    def test_becomes_due_once_an_interval_of_material_has_built_up(self, tmp_path):
        drain, _ = _drain(tmp_path, retention_s=45.0, lead_s=0.0)
        drain.note_open(_at(0))
        assert drain.due(_at(10)) is False
        assert drain.due(_at(15 + SEGMENT_LAG_S)) is True

    async def test_dueness_is_measured_from_the_cursor_not_the_last_call(self, tmp_path):
        """What matters is how much un-rescued material is standing in the buffer."""
        drain, _ = _drain(tmp_path, retention_s=45.0, lead_s=0.0)
        drain.note_open(_at(0))
        await drain.drain(_at(30))
        # Cursor is now at 24s; a further interval must pass from *there*.
        assert drain.due(_at(32)) is False
        assert drain.due(_at(24 + 15 + SEGMENT_LAG_S)) is True


class TestEviction:
    async def test_material_older_than_the_window_is_skipped_and_flagged(self, tmp_path):
        """A stalled loop loses video. The clip must declare the hole, not hide it."""
        drain, extract = _drain(tmp_path, retention_s=45.0, lead_s=0.0)
        drain.note_open(_at(0))
        # Nothing ran for five minutes: everything before now-45s is gone.
        chunk = await drain.drain(_at(300))
        assert chunk.gap_before is True
        assert chunk.start == _at(300 - 45)

    async def test_a_later_chunk_does_not_inherit_an_earlier_gap(self, tmp_path):
        drain, extract = _drain(tmp_path, retention_s=45.0, lead_s=0.0)
        drain.note_open(_at(0))
        await drain.drain(_at(300))
        following = await drain.drain(_at(330))
        assert following.gap_before is False

    async def test_a_failed_extract_does_not_advance_the_cursor(self, tmp_path):
        """The next pass must retry the same window; only real eviction may lose it."""
        extract = _FakeExtract(fail_with=RuntimeError("playback server down"))
        drain, _ = _drain(tmp_path, lead_s=0.0, extract=extract)
        drain.note_open(_at(0))
        assert await drain.drain(_at(30)) is None
        assert drain.chunks == []

        # Recovered inside the window: the retry covers everything since the event
        # started, not just since the failure, and nothing is flagged as lost.
        drain._extract = _FakeExtract()
        chunk = await drain.drain(_at(40))
        assert chunk.start == _at(0)
        assert chunk.gap_before is False


class TestSharedAcrossEvents:
    """Two rules open on one camera share one drain — the whole reason this beats
    recording each event live."""

    def test_a_second_event_does_not_move_the_cursor_forward(self, tmp_path):
        drain, _ = _drain(tmp_path, lead_s=0.0)
        drain.note_open(_at(100))
        drain.note_open(_at(200))
        assert drain._cursor == _at(100)

    async def test_both_events_are_served_from_the_same_chunks(self, tmp_path):
        drain, extract = _drain(tmp_path, lead_s=0.0)
        drain.note_open(_at(0))
        await drain.drain(_at(60))
        await drain.drain(_at(120))
        assert len(extract.calls) == 2, "one drain per source, not per event"
        assert len(drain.chunks_for(_at(0), _at(120))) == 2
        assert len(drain.chunks_for(_at(80), _at(120))) == 1


class TestChunkSelection:
    async def test_only_chunks_overlapping_the_window_are_returned(self, tmp_path):
        drain, _ = _drain(tmp_path, lead_s=0.0)
        drain.note_open(_at(0))
        await drain.drain(_at(60))
        await drain.drain(_at(120))
        assert drain.chunks_for(_at(500), _at(600)) == []


class TestCleanup:
    async def test_discard_removes_the_files(self, tmp_path):
        drain, _ = _drain(tmp_path, lead_s=0.0)
        drain.note_open(_at(0))
        chunk = await drain.drain(_at(30))
        assert chunk.path.exists()

        drain.discard()
        assert not chunk.path.exists()
        assert drain.active is False

    async def test_going_idle_discards_everything(self, tmp_path):
        drain, _ = _drain(tmp_path, lead_s=0.0)
        drain.note_open(_at(0))
        chunk = await drain.drain(_at(30))
        drain.note_idle()
        assert not chunk.path.exists()
        assert drain.chunks == []

    def test_sweep_removes_chunks_left_by_a_previous_process(self, tmp_path):
        """Full-rate video, worthless across a restart — a slow disk leak otherwise."""
        stale = tmp_path / "src-1" / "20260818-120000-000000.mp4"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"video")

        assert sweep_chunk_root(tmp_path) == 1
        assert not stale.exists()

    def test_sweeping_a_missing_root_is_not_an_error(self, tmp_path):
        assert sweep_chunk_root(tmp_path / "never-created") == 0


class TestDrainedChunk:
    def test_duration_is_derived_from_the_bounds(self):
        chunk = DrainedChunk(path=Path("/x.mp4"), start=_at(0), end=_at(30))
        assert chunk.duration_s == 30.0
