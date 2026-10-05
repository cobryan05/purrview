"""Unit tests for the clip drain — the thing that makes an event longer than the buffer work.

The failure this exists to prevent is silent: a rule's `max_duration_s` is a free number
in the editor, the rewind buffer is a fixed RAM window, and cutting only at the closing
edge quietly keeps whatever survived. A ten-minute event against a 45 s buffer produced a
45 s clip and no error.

Everything here is pure — the extract call is a stub — because the real bugs are cursor
arithmetic and eviction bookkeeping, and those are only cheap to find without a relay.
"""

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import pytest

from apps.helpers.recordings.clipDrain import (
    MIN_DRAIN_INTERVAL_S,
    SEGMENT_LAG_S,
    ClipDrain,
    DrainedChunk,
    concat_parts,
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


class TestTheExtractorsActualWindow:
    """A drain records what was CUT, not what it asked for.

    `StreamManager.extract_clip` clamps the window to what the buffer holds at both ends
    and reports the real `startedAt`/`durationS`. Discarding that was one silent bug with
    three faces: the chunk overstated its own length, so the assembled clip's summed
    `durationS` ran past the end of its own file (a real one declared 246.4 s against
    231.2 s, putting its last box 14 s past the video); `hasGap` stayed False because the
    eviction check only covers material lost *before* the window, so the overlay was
    never suppressed; and the cursor advanced past video that was never copied, making
    the hole permanent.
    """

    def _clamping(self, *, start_delay=0.0, duration=None):
        """An extractor that returns a window narrower than the one requested."""

        class _Clamping(_FakeExtract):
            async def __call__(self, start, duration_s, dest):
                await super().__call__(start, duration_s, dest)
                actual_start = start + timedelta(seconds=start_delay)
                actual = duration_s - start_delay if duration is None else duration
                return {
                    "path": str(dest),
                    "sizeBytes": 5,
                    "startedAt": actual_start.isoformat(),
                    "durationS": actual,
                }

        return _Clamping()

    async def test_a_short_cut_is_not_recorded_as_a_full_one(self, tmp_path):
        """The chunk's own length is what `_assemble_from_chunks` sums into the clip's
        `durationS`, so an overstatement here is what puts boxes past the end."""
        drain, _ = _drain(tmp_path, lead_s=0.0, extract=self._clamping(duration=20.0))
        drain.note_open(_at(0))

        chunk = await drain.drain(_at(30))

        assert chunk.duration_s == pytest.approx(20.0), "recorded the requested span, not the cut"

    async def test_material_the_buffer_had_not_flushed_yet_is_asked_for_again(self, tmp_path):
        """The far-end clamp is a flush lag, not eviction: the video exists moments
        later. Advancing past it is what turned a transient lag into a lost stretch."""
        drain, extract = _drain(tmp_path, lead_s=0.0, extract=self._clamping(duration=20.0))
        drain.note_open(_at(0))

        await drain.drain(_at(30))
        await drain.drain(_at(60))

        second_start = extract.calls[1][0]
        assert second_start == _at(20), f"resumed at {second_start} instead of the end of what was cut"

    async def test_a_clamped_head_is_declared_as_a_gap(self, tmp_path):
        """Material that existed when the window was chosen and was gone when it was read
        is eviction by another route, and the finished clip has a hole in it. `hasGap` is
        what suppresses the overlay, so an unflagged chunk draws boxes that slide."""
        drain, _ = _drain(tmp_path, lead_s=0.0, extract=self._clamping(start_delay=5.0))
        drain.note_open(_at(0))

        chunk = await drain.drain(_at(30))

        assert chunk.gap_before is True
        assert chunk.start == _at(5)

    async def test_a_sub_second_clamp_is_rounding_not_a_hole(self, tmp_path):
        """Segment boundaries never land exactly on the request. Flagging that would mark
        every ordinary clip as gapped and silence every overlay."""
        drain, _ = _drain(tmp_path, lead_s=0.0, extract=self._clamping(start_delay=0.2))
        drain.note_open(_at(0))

        chunk = await drain.drain(_at(30))

        assert chunk.gap_before is False

    async def test_a_backend_reporting_nothing_keeps_the_requested_window(self, tmp_path):
        """The ring backend and the test doubles return no window at all; falling back to
        the request is the previous behaviour and must stay."""
        drain, _ = _drain(tmp_path, lead_s=0.0)
        drain.note_open(_at(0))

        chunk = await drain.drain(_at(30))

        assert chunk.start == _at(0)
        # 24, not 30: a drain never reads closer to now than SEGMENT_LAG_S, because
        # MediaMTX is still writing the current segment.
        assert chunk.duration_s == pytest.approx(30.0 - SEGMENT_LAG_S)
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


class TestConcatParts:
    """The join itself, against the real ffmpeg.

    Everything else in this file stubs the extract, because the bugs worth hunting there
    are cursor arithmetic. That left `concat_parts` — the function that actually produces
    the finished mp4 for every event that outlives the buffer — with no coverage at all,
    and it never worked: the output is written to `<name>.mp4.part` so the orphan sweep
    never sees a half-written file, but ffmpeg picks its muxer from the extension and
    `.part` matches none, so every assembled clip died with "Unable to find a suitable
    output format" reported as `Invalid argument`.

    Asserting on the argv would not have caught it. These tests run ffmpeg and read the
    result back.
    """

    @staticmethod
    def _part(path: Path, seconds: int = 1) -> Path:
        """A real, playable mp4 — parts must be codec-identical for a stream copy."""
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size=160x120:rate=10:duration={seconds}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            capture_output=True,
            check=True,
            timeout=60,
        )
        return path

    def test_two_parts_join_into_a_playable_clip(self, tmp_path):
        parts = [self._part(tmp_path / "a.mp4"), self._part(tmp_path / "b.mp4")]
        dest = tmp_path / "out" / "joined.mp4"

        size = concat_parts(parts, dest)

        assert dest.is_file() and size > 0
        # The `.part` is renamed away on success, never left beside the finished file.
        assert not dest.with_suffix(".mp4.part").exists()
        capture = cv2.VideoCapture(str(dest))
        try:
            assert capture.isOpened(), "ffmpeg wrote a file no decoder will open"
            assert capture.read()[0]
        finally:
            capture.release()

    def test_a_single_part_still_produces_a_clip(self, tmp_path):
        """An event that closed inside one drain interval has exactly one chunk."""
        dest = tmp_path / "one.mp4"
        assert concat_parts([self._part(tmp_path / "a.mp4")], dest) > 0
        assert dest.is_file()

    def test_empty_and_missing_parts_are_skipped(self, tmp_path):
        """A chunk whose extract failed is a zero-byte file, not a reason to lose the
        rest of the clip."""
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        parts = [empty, tmp_path / "never-written.mp4", self._part(tmp_path / "a.mp4")]

        assert concat_parts(parts, tmp_path / "out.mp4") > 0

    def test_nothing_usable_raises_rather_than_writing_an_empty_file(self, tmp_path):
        with pytest.raises(ValueError):
            concat_parts([tmp_path / "missing.mp4"], tmp_path / "out.mp4")
        assert not (tmp_path / "out.mp4").exists()

    def test_a_length_cap_trims_the_tail(self, tmp_path):
        """`max_length_s` keeps the head, the same end the single-cut path keeps."""
        parts = [self._part(tmp_path / "a.mp4", seconds=2), self._part(tmp_path / "b.mp4", seconds=2)]
        dest = tmp_path / "capped.mp4"

        concat_parts(parts, dest, max_length_s=1.0)

        capture = cv2.VideoCapture(str(dest))
        try:
            frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
        finally:
            capture.release()
        assert 0 < frames / fps <= 1.5, f"expected about a second, got {frames / fps:.2f}s"


class TestAssembleReportsWhatTheFileHolds:
    """`assemble`'s two declarations about the finished clip: how long it is, and
    whether wallclock still maps linearly onto it.

    Both drive the detection overlay. `build_overlay` places a box by subtracting
    `startedAt` from its wallclock, which is only valid while the video is one unbroken
    run of time — so a clip that declares more seconds than it holds draws every box
    after the missing stretch late by its length, and reads as a detector fault.
    """

    @staticmethod
    def _chunk(path: Path, start_s: float, span_s: float, gap_before: bool = False) -> DrainedChunk:
        TestConcatParts._part(path, seconds=max(1, int(round(span_s))))
        return DrainedChunk(path=path, start=_at(start_s), end=_at(start_s + span_s), gap_before=gap_before)

    async def test_the_measured_length_is_reported(self, tmp_path):
        from apps.helpers.recordings.clipDrain import assemble

        chunks = [self._chunk(tmp_path / "a.mp4", 0, 2), self._chunk(tmp_path / "b.mp4", 2, 2)]

        result = await assemble(chunks, None, tmp_path / "out.mp4")

        measured = result["measuredDurationS"]
        assert measured is not None
        assert abs(measured - 4.0) < 0.5, f"two 2s parts should measure about 4s, got {measured}"

    async def test_a_tail_cut_later_than_asked_is_a_gap(self, tmp_path):
        """The chunk-to-tail seam. `gap_before` lives on chunks, and the tail is not
        one, so a tail clamped forward opens a hole nothing else reports."""
        from apps.helpers.recordings.clipDrain import assemble

        chunks = [self._chunk(tmp_path / "a.mp4", 0, 2)]
        tail = TestConcatParts._part(tmp_path / "tail.mp4", seconds=1)

        result = await assemble(
            chunks,
            tail,
            tmp_path / "out.mp4",
            tail_start=_at(2),
            tail_actual_start=_at(9),
        )

        assert result["hasGap"] is True

    async def test_a_tail_cut_where_it_was_asked_is_not_a_gap(self, tmp_path):
        """The negative half — without it the test above passes on a function that
        hardcodes True, and every assembled clip silently loses its overlay."""
        from apps.helpers.recordings.clipDrain import assemble

        chunks = [self._chunk(tmp_path / "a.mp4", 0, 2)]
        tail = TestConcatParts._part(tmp_path / "tail.mp4", seconds=1)

        result = await assemble(
            chunks,
            tail,
            tmp_path / "out.mp4",
            tail_start=_at(2),
            tail_actual_start=_at(2),
        )

        assert result["hasGap"] is False

    async def test_a_sub_second_tail_clamp_is_rounding_not_a_hole(self, tmp_path):
        """Segments are keyframe-aligned, so a fraction of a second of movement at the
        seam is ordinary. Flagging it would cost every clip its overlay."""
        from apps.helpers.recordings.clipDrain import assemble

        chunks = [self._chunk(tmp_path / "a.mp4", 0, 2)]
        tail = TestConcatParts._part(tmp_path / "tail.mp4", seconds=1)

        result = await assemble(
            chunks,
            tail,
            tmp_path / "out.mp4",
            tail_start=_at(2),
            tail_actual_start=_at(2.2),
        )

        assert result["hasGap"] is False
