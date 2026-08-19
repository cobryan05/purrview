"""Pull a source's buffered video to disk while an event is still open.

The rewind buffer is RAM and holds one fixed window per camera — 45 s by default. A
rule's ``max_duration_s`` is a free number in the editor, so an event can legitimately
run far longer than that window. Cutting only at the closing edge then keeps whatever
survived: a ten-minute visit against a 45 s buffer yields 45 s, silently, with the front
of the event evicted long before anything asked for it.

This closes that gap without giving up the retroactive cut. While an event is open, the
material behind it is copied out to disk **before it ages out**, in chunks; at the
closing edge the chunks and the still-buffered tail are concatenated into the finished
clip. The buffer stays the same size, events become effectively unbounded, and nothing
has to be predicted at event start.

Three properties are deliberate:

* **Per source, not per event.** Two rules open at once on one camera share one drain
  and one set of chunks. Draining per event would copy the same video twice, which is
  precisely the advantage this design has over recording each event live.
* **Idle costs nothing.** No events open means no cursor, no chunks and no calls. The
  overwhelming majority of events close inside the buffer and never touch this path.
* **A gap is reported, never hidden.** If material is evicted before a drain reaches it
  the chunk list carries the discontinuity, and the finished recording says so rather
  than presenting a clip with a silent hole.
"""

import asyncio
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

#: How close to "now" a drain is willing to read. MediaMTX is still writing the current
#: segment, so asking for it returns a partial or nothing at all; staying one segment
#: behind is what makes every drained chunk whole.
SEGMENT_LAG_S = 6.0

#: Fraction of the retention window a drain is allowed to fall behind before it must
#: run. A third leaves two thirds of the window as margin for a stalled loop, a slow
#: inference call or a paused task — all of which delay the next drain.
DRAIN_FRACTION = 1.0 / 3.0

#: Never drain more often than this regardless of how short the retention is, so a
#: misconfigured 3 s buffer cannot turn into a busy loop against the playback server.
MIN_DRAIN_INTERVAL_S = 5.0


class ExtractFn(Protocol):
    """Cut ``[start, start+duration_s)`` from the buffer into ``dest``."""

    def __call__(self, start: datetime, duration_s: float, dest: Path) -> Awaitable[dict]: ...


@dataclass(frozen=True)
class DrainedChunk:
    """One stretch of video already rescued from the buffer."""

    path: Path
    start: datetime
    end: datetime
    #: True when material was evicted between the previous chunk and this one. Carried
    #: on the chunk rather than the drain so it survives into assembly, where the
    #: finished recording needs to declare it.
    gap_before: bool = False

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()


class ClipDrain:
    """Per-source drain state. One instance per source with an open event."""

    def __init__(
        self,
        source_uuid: str,
        retention_s: float,
        extract: ExtractFn,
        chunk_root: Path,
        lead_s: float = 10.0,
        interval_s: Optional[float] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.source_uuid = source_uuid
        self.retention_s = float(retention_s)
        self._extract = extract
        self._chunk_root = Path(chunk_root)
        # How far back of the event's own start to begin. Pre-roll lives in front of the
        # event, so a drain that started at the event's start would let exactly the
        # pre-roll age out — the one part of the clip that cannot be re-derived.
        self.lead_s = float(lead_s)
        self.interval_s = float(interval_s) if interval_s else self._default_interval(retention_s)
        self._log = log or (lambda _msg: None)
        #: Everything up to here has been copied out. None means no event is open.
        self._cursor: Optional[datetime] = None
        self._chunks: List[DrainedChunk] = []
        self._pending_gap = False

    @staticmethod
    def _default_interval(retention_s: float) -> float:
        """Derived from the window rather than fixed, so it stays correct if retention changes."""
        return max(MIN_DRAIN_INTERVAL_S, float(retention_s) * DRAIN_FRACTION)

    # ------------------------------------------------------------------ state

    @property
    def active(self) -> bool:
        return self._cursor is not None

    @property
    def chunks(self) -> List[DrainedChunk]:
        return list(self._chunks)

    def note_open(self, event_start: datetime) -> None:
        """Arm the drain for an event that opened at `event_start` (wallclock).

        Idempotent, and it never moves the cursor forward: with two rules open on one
        camera the earliest start wins, because the later event's material is a subset
        of what the earlier one already needs.
        """
        if self._cursor is None:
            self._cursor = event_start - timedelta(seconds=self.lead_s)

    def note_idle(self) -> None:
        """No events open: drop the chunks and disarm.

        Called every pass rather than only on transitions, so a drain cannot survive an
        event that ended through a path that forgot to tell it.
        """
        if self._cursor is None and not self._chunks:
            return
        self.discard()

    def due(self, now: datetime) -> bool:
        if self._cursor is None:
            return False
        # Measured from the cursor, not from the last call: what matters is how much
        # un-rescued material is standing in the buffer, not how long ago we ran.
        behind = (now - self._cursor).total_seconds() - SEGMENT_LAG_S
        return behind >= self.interval_s

    # ------------------------------------------------------------------ work

    async def drain(self, now: datetime) -> Optional[DrainedChunk]:
        """Copy everything between the cursor and one segment ago out to disk."""
        if self._cursor is None:
            return None
        end = now - timedelta(seconds=SEGMENT_LAG_S)
        start = self._cursor
        duration = (end - start).total_seconds()
        if duration <= 0:
            return None

        # Material older than the window is already gone. Skip forward to what still
        # exists rather than asking for it and failing the whole drain — and remember
        # that the result has a hole in it.
        oldest = now - timedelta(seconds=self.retention_s)
        if start < oldest:
            lost = (oldest - start).total_seconds()
            self._pending_gap = True
            self._log(
                f"Clip drain on {self.source_uuid} fell behind by {lost:.0f}s; that much of "
                "the event was evicted from the buffer before it could be copied out"
            )
            start = oldest
            duration = (end - start).total_seconds()
            if duration <= 0:
                self._cursor = end
                return None

        dest = self._chunk_path(start)
        try:
            await self._extract(start=start, duration_s=duration, dest=dest)
        except Exception as e:
            # A failed drain must not advance the cursor: the next pass retries the same
            # window, and only genuine eviction (handled above) is allowed to lose it.
            logger.warning("Clip drain failed for %s: %s", self.source_uuid, e)
            self._log(f"Clip drain on {self.source_uuid} failed: {e}")
            return None

        chunk = DrainedChunk(path=dest, start=start, end=end, gap_before=self._pending_gap)
        self._pending_gap = False
        self._chunks.append(chunk)
        self._cursor = end
        return chunk

    def chunks_for(self, start: datetime, end: datetime) -> List[DrainedChunk]:
        """Chunks overlapping `[start, end)`, in order."""
        return [c for c in self._chunks if c.end > start and c.start < end]

    def discard(self) -> None:
        """Delete every chunk and disarm. Safe to call repeatedly."""
        for chunk in self._chunks:
            try:
                chunk.path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not remove drained chunk %s", chunk.path, exc_info=True)
        self._chunks = []
        self._cursor = None
        self._pending_gap = False

    def _chunk_path(self, start: datetime) -> Path:
        stamp = start.strftime("%Y%m%d-%H%M%S-%f")
        return self._chunk_root / self.source_uuid / f"{stamp}.mp4"


def sweep_chunk_root(chunk_root: Path) -> int:
    """Delete chunks left behind by a previous process. Returns how many were removed.

    Chunks are worthless across a restart — the events they belonged to were force-closed
    and their clips already cut or lost — but they are full-rate video, so leaving them
    is a slow disk leak that nothing else would ever clean up.
    """
    root = Path(chunk_root)
    if not root.exists():
        return 0
    removed = 0
    for path in root.rglob("*.mp4"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            logger.debug("Could not sweep drained chunk %s", path, exc_info=True)
    return removed


def concat_parts(parts: Sequence[Path], dest: Path, max_length_s: Optional[float] = None) -> int:
    """Join `parts` into `dest` with the concat demuxer, returning the byte count.

    Stream copy, never a re-encode: every part came out of the same camera's recording
    through the same playback server, so the codec parameters are identical by
    construction and joining is a container-level operation.

    `-fflags +genpts` is not optional here. Each part carries its own timeline starting
    near zero, so concatenating them without regenerating timestamps produces
    non-monotonic DTS — which players show as a clip that freezes or jumps at every
    boundary rather than as an error.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    existing = [p for p in parts if p.exists() and p.stat().st_size > 0]
    if not existing:
        raise ValueError("No parts to concatenate")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        for path in existing:
            # The concat demuxer's own quoting rule: single quotes are escaped as '\''.
            escaped = str(path).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
        list_path = Path(handle.name)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-fflags",
        "+genpts",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
    ]
    if max_length_s and max_length_s > 0:
        # Trims the tail, keeping the head — the same end the single-cut path keeps, so
        # a capped clip means the same thing however it was produced.
        cmd += ["-t", f"{max_length_s:.3f}"]
    # Named explicitly because the output goes to `<name>.mp4.part` — written under a
    # temporary name so the orphan sweep and the file endpoint never see a half-written
    # clip — and ffmpeg picks its muxer from the *extension*. `.part` matches none, so
    # without this every assembled clip died with "Unable to find a suitable output
    # format", surfacing as a bare `Invalid argument` on the output path that reads like
    # a disk or permissions fault.
    cmd += ["-f", "mp4"]
    cmd.append(str(partial))

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0 or not partial.exists():
            raise RuntimeError(f"ffmpeg concat failed: {proc.stderr[-300:].decode('utf-8', 'replace')}")
        size = partial.stat().st_size
        partial.replace(dest)
        return size
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        list_path.unlink(missing_ok=True)


async def assemble(
    chunks: Sequence[DrainedChunk],
    tail: Optional[Path],
    dest: Path,
    max_length_s: Optional[float] = None,
) -> dict:
    """Concatenate drained chunks plus the still-buffered tail into one clip."""
    parts = [c.path for c in chunks] + ([tail] if tail is not None else [])
    size = await asyncio.to_thread(concat_parts, parts, dest, max_length_s)
    started_at = chunks[0].start if chunks else None
    return {
        "path": str(dest),
        "sizeBytes": size,
        "startedAt": started_at.isoformat() if started_at else None,
        "partCount": len(parts),
        # Declared rather than smoothed over: a clip missing a stretch in the middle is
        # a different artifact from a complete one, and only this layer knows.
        "hasGap": any(c.gap_before for c in chunks),
    }
