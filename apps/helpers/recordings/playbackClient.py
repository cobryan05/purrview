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

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx

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
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


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
