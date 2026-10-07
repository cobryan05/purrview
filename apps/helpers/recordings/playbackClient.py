"""Client for MediaMTX's playback server — the thing that turns the rolling RAM
buffer into retroactive clips.

Two endpoints matter:

* ``GET /list?path=<path>`` — the time ranges currently held for a path. This is how
  we find out how much rewind actually survives, which is *not* the same as the
  retention we asked for: the buffer is RAM-backed, so a relay restart silently
  shortens it, and segments are keyframe-aligned so they overshoot.
* ``GET /get?path=&start=&duration=&format=mp4`` — one concatenated clip for an
  arbitrary window, streamed.

Because extraction happens after the event, pre-roll is chosen per event rather than
fixed when the stream was created. The deadline is short, though: a window that ages
out of the buffer is gone permanently, so callers must treat a failed extraction as
terminal past the retention window rather than retrying forever.
"""

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx

from apps.helpers.timeUtils import ensure_utc

logger = logging.getLogger(__name__)

DEFAULT_PLAYBACK_PORT = 9996
# Extraction is a short-deadline operation (the source window is evaporating), but the
# response is streamed, so this bounds the whole transfer rather than a single read.
DEFAULT_TIMEOUT_S = 60.0


class PlaybackError(RuntimeError):
    """Raised when the playback server refuses or fails a request."""


@dataclass(frozen=True)
class BufferedRange:
    """One contiguous span of recording the playback server still holds."""

    start: datetime
    duration_s: float

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_s)

    def covers(self, start: datetime, end: datetime) -> bool:
        return self.start <= start and self.end >= end

    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "durationS": self.duration_s,
            "end": self.end.isoformat(),
        }


def piece_windows(ranges: "list[BufferedRange]", start: datetime, end: datetime) -> "list[tuple[datetime, datetime]]":
    """``[start, end)`` split into one window per advertised range, in order.

    Adjacent and overlapping ranges are kept **separate** rather than merged into one
    span. That distinction is the whole point: a merged span asserts that every instant
    inside it is present, and ``/list`` is not entitled to assert that — it advertises
    the segment MediaMTX is still writing, segments its own retention has deleted, and
    after a re-arm ranges that overlap each other. A cut taken per range can be measured
    per range, so a range that delivers less than it claimed is found out on its own
    instead of passing the shortfall to its neighbours, where it has no position and
    nothing downstream can place a box.

    Overlaps are resolved by clipping, never by dropping: each window starts where the
    previous one ended, so the same wallclock is never served twice and nothing inside
    the envelope is skipped.
    """
    out: "list[tuple[datetime, datetime]]" = []
    cursor = start
    for r in sorted(ranges, key=lambda r: r.start):
        lo, hi = max(r.start, cursor), min(r.end, end)
        if (hi - lo).total_seconds() <= 0:
            continue
        out.append((lo, hi))
        cursor = hi
        if cursor >= end:
            break
    return out


async def probe_duration_s(path: Path) -> Optional[float]:
    """Playing length of ``path`` per ffprobe, or None if it cannot be read.

    None and 0.0 mean different things to every caller here — "no measurement" against
    "a file holding nothing" — so an unreadable file must not come back as a number.
    """

    def _run() -> Optional[float]:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True,
            timeout=30,
        )
        try:
            return float(proc.stdout.decode().strip())
        except ValueError:
            return None

    try:
        return await asyncio.to_thread(_run)
    except Exception:
        logger.debug("Could not probe %s", path, exc_info=True)
        return None


def _parse_rfc3339(value: str) -> datetime:
    """Parse MediaMTX's RFC3339 timestamps, including the 'Z' suffix Python < 3.11
    won't take and the >6-digit fractional seconds Go emits."""
    text = value.strip().replace("Z", "+00:00")
    if "." in text:
        head, _, tail = text.partition(".")
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                tail = tail[len(digits) :]
                break
        else:
            tail = ""
        text = f"{head}.{digits[:6]:0<6}{tail}"
    return ensure_utc(datetime.fromisoformat(text))


class PlaybackClient:
    """Async wrapper over the MediaMTX playback HTTP server."""

    def __init__(self, host: str = "localhost", port: int = DEFAULT_PLAYBACK_PORT, timeout: float = DEFAULT_TIMEOUT_S):
        self._base_url = f"http://{host}:{port}"
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        return self._base_url

    async def list_ranges(self, path: str) -> list[BufferedRange]:
        """Time ranges currently buffered for ``path``, oldest first.

        An unknown path is not an error — a camera that has only just been armed has
        no segments yet — so that returns an empty list rather than raising.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.get(f"{self._base_url}/list", params={"path": path})
            except httpx.HTTPError as exc:
                raise PlaybackError(f"playback /list failed for {path!r}: {exc}") from exc

        if resp.status_code == 404:
            return []
        if resp.status_code >= 400:
            raise PlaybackError(f"playback /list returned {resp.status_code} for {path!r}: {resp.text[:200]}")

        ranges: list[BufferedRange] = []
        for entry in resp.json() or []:
            try:
                ranges.append(BufferedRange(start=_parse_rfc3339(entry["start"]), duration_s=float(entry["duration"])))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(f"Skipping unparseable playback range {entry!r}: {exc}")
        ranges.sort(key=lambda r: r.start)
        return ranges

    async def coverage_seconds(self, path: str, now: Optional[datetime] = None) -> float:
        """How many seconds of rewind are actually available for ``path`` right now.

        Measured back from the newest buffered instant rather than from wallclock, so a
        stream that stalled a moment ago still reports the depth it really has.
        """
        ranges = await self.list_ranges(path)
        if not ranges:
            return 0.0
        newest_end = max(r.end for r in ranges)
        oldest_start = min(r.start for r in ranges)
        reference = now or newest_end
        return max(0.0, (min(reference, newest_end) - oldest_start).total_seconds())

    async def stream_clip(
        self, path: str, start: datetime, duration_s: float, fmt: str = "mp4"
    ) -> AsyncIterator[bytes]:
        """Yield the bytes of one concatenated clip for ``[start, start+duration_s)``."""
        params = {
            "path": path,
            "start": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "duration": f"{duration_s:.3f}",
            "format": fmt,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("GET", f"{self._base_url}/get", params=params) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread())[:200].decode("utf-8", "replace")
                    raise PlaybackError(f"playback /get returned {resp.status_code} for {path!r}: {body}")
                async for chunk in resp.aiter_bytes():
                    yield chunk

    async def write_clip(self, path: str, start: datetime, duration_s: float, dest: Path, fmt: str = "mp4") -> int:
        """Stream a clip straight to ``dest`` and return the byte count written.

        Writes to a ``.part`` sibling and renames on success, so a failed or truncated
        extraction never leaves a file that looks like a complete recording.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(dest.suffix + ".part")
        written = 0
        try:
            with partial.open("wb") as handle:
                async for chunk in self.stream_clip(path, start, duration_s, fmt=fmt):
                    handle.write(chunk)
                    written += len(chunk)
            if written == 0:
                raise PlaybackError(f"playback /get produced no data for {path!r} at {start.isoformat()}")
            partial.replace(dest)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return written
