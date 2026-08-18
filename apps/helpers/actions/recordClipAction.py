"""Cut an event clip out of the rolling buffer.

Extraction happens *after* the event ends, which is the entire point of recording into
a rewindable buffer rather than into a delayed stream: pre-roll is chosen per event,
per rule, at cut time, and nothing has to be predicted in advance.

Two backends behind one interface, chosen per source at event time:

* ``PlaybackClipBackend`` — Tiers A/B. MediaMTX's playback server concatenates the
  requested window out of the fMP4 segments in ``/buffer``.
* ``RingClipBackend`` — Tier C (snapshot pollers, MQTT feeds). There is no continuous
  stream to record, so a bounded in-RAM JPEG ring is flushed to mp4 instead. Choppy,
  labelled as such, and far better than nothing.

The action's parameters mean the same thing in both.

The work queue this feeds has a **short deadline**. The buffer is RAM and ages out in
under a minute, so a failed extraction past the window is permanent — it is recorded
as `expired` rather than retried forever.
"""

import asyncio
import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from apps import RECORDINGS_CLIP_DIR

from ..watch.events import EventPhase, WatchEventSignal
from .action import Action, ActionContext, ActionResult
from .Registry import register_action

logger = logging.getLogger(__name__)

#: Shared with db_server's retention sweep, which has to find these files to delete
#: them — see `apps/__init__.py`. A clip written anywhere else (a hand-set
#: `output_dir`) is outside the managed area and is never swept.
DEFAULT_CLIP_ROOT = RECORDINGS_CLIP_DIR


class ClipBackend(ABC):
    """Produces an mp4 for a time window."""

    @abstractmethod
    async def extract(self, context: ActionContext, start: datetime, duration_s: float, dest: Path) -> Dict[str, Any]:
        """Write a clip covering `[start, start+duration_s)` to `dest`."""


class PlaybackClipBackend(ClipBackend):
    """Tiers A/B — cut from MediaMTX's recorded segments via the playback server."""

    def __init__(self, relay_client: Any) -> None:
        self._relay = relay_client

    async def extract(self, context: ActionContext, start: datetime, duration_s: float, dest: Path) -> Dict[str, Any]:
        if not context.relay_path:
            raise ValueError("Source has no relay path, so there is no buffer to cut from")
        return await self._relay.extract_clip(path=context.relay_path, start=start, duration_s=duration_s, dest=dest)


class RingClipBackend(ClipBackend):
    """Tier C — flush a bounded in-RAM JPEG ring to an mp4.

    Real pre-roll for snapshot cameras: at 1–2 fps a 20 s ring is a few MB of RAM and
    produces a genuinely useful, if choppy, lead-in. The output fps is written from the
    frames' own timestamps so the clip plays back at wall speed rather than as a
    slideshow at an invented rate.
    """

    def __init__(self, ring: "FrameRing") -> None:
        self._ring = ring

    async def extract(self, context: ActionContext, start: datetime, duration_s: float, dest: Path) -> Dict[str, Any]:
        frames = self._ring.frames_between(start.timestamp(), start.timestamp() + duration_s)
        if not frames:
            raise ValueError("Frame ring holds nothing for the requested window")
        size = await asyncio.to_thread(self._write_mp4, frames, dest)
        actual_start = datetime.fromtimestamp(frames[0][0], tz=timezone.utc)
        actual_duration = max(0.1, frames[-1][0] - frames[0][0])
        return {
            "path": str(dest),
            "sizeBytes": size,
            "startedAt": actual_start.isoformat(),
            "durationS": actual_duration,
            "requestedStart": start.isoformat(),
            "requestedDurationS": duration_s,
            # A ring clip is inherently partial pre-roll relative to a real timeline;
            # only flag it when the ring genuinely did not reach back far enough.
            "partialPreroll": frames[0][0] > start.timestamp() + 0.5,
            "lowFps": True,
        }

    @staticmethod
    def _write_mp4(frames: List[Tuple[float, bytes]], dest: Path) -> int:
        """Encode JPEG frames to mp4 via image2pipe. Runs in a worker thread."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        span = max(0.1, frames[-1][0] - frames[0][0])
        fps = max(1.0, min(30.0, len(frames) / span))
        partial = dest.with_suffix(dest.suffix + ".part")
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "image2pipe",
            "-framerate",
            f"{fps:.3f}",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            # yuv420p, not the JPEG-native 4:4:4: anything else is unplayable in a
            # browser <video> element, which is the whole point of writing a clip.
            "-pix_fmt",
            "yuv420p",
            str(partial),
        ]
        try:
            proc = subprocess.run(cmd, input=b"".join(f[1] for f in frames), capture_output=True, timeout=120)
            if proc.returncode != 0 or not partial.exists():
                raise RuntimeError(f"ffmpeg failed writing ring clip: {proc.stderr[-300:].decode('utf-8', 'replace')}")
            size = partial.stat().st_size
            partial.replace(dest)
            return size
        except BaseException:
            partial.unlink(missing_ok=True)
            raise


class FrameRing:
    """Bounded in-RAM JPEG ring for sources with no continuous stream.

    Bounded by **bytes, not frame count**: frame sizes vary by an order of magnitude
    between cameras, so a count-based bound gives wildly different memory footprints
    for the same configuration.
    """

    def __init__(self, max_seconds: float = 20.0, max_mb: float = 32.0) -> None:
        self._max_seconds = max_seconds
        self._max_bytes = int(max_mb * 1024 * 1024)
        self._frames: List[Tuple[float, bytes]] = []
        self._bytes = 0

    def push(self, timestamp: float, jpeg: bytes) -> None:
        self._frames.append((timestamp, jpeg))
        self._bytes += len(jpeg)
        self._evict(timestamp)

    def _evict(self, now: float) -> None:
        cutoff = now - self._max_seconds
        while self._frames and (self._frames[0][0] < cutoff or self._bytes > self._max_bytes):
            _, data = self._frames.pop(0)
            self._bytes -= len(data)

    def frames_between(self, start: float, end: float) -> List[Tuple[float, bytes]]:
        return [f for f in self._frames if start <= f[0] <= end]

    @property
    def size_bytes(self) -> int:
        return self._bytes

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def coverage_s(self) -> float:
        if len(self._frames) < 2:
            return 0.0
        return self._frames[-1][0] - self._frames[0][0]


@register_action("record_clip")
class RecordClipAction(Action):
    """Writes an mp4 for an event, with pre-roll taken from the rolling buffer."""

    def __init__(
        self,
        preroll_s: float = 10.0,
        postroll_s: float = 5.0,
        max_length_s: float = 180.0,
        output_dir: str = str(DEFAULT_CLIP_ROOT),
        backend: Optional[ClipBackend] = None,
        backend_for: Optional[Callable[[ActionContext], Optional[ClipBackend]]] = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.preroll_s = float(preroll_s)
        self.postroll_s = float(postroll_s)
        self.max_length_s = float(max_length_s)
        self.output_dir = Path(output_dir)
        self._backend = backend
        # Resolved per event rather than per action, because the right backend is a
        # property of the *source*: one profile can watch an RTSP camera (playback out
        # of MediaMTX) and a snapshot poller (frame ring) at once. A single injected
        # backend meant the whole profile got whichever one the task built, so a
        # snapshot source in a mixed profile failed every clip with "no relay path".
        self._backend_for = backend_for

    def _resolve_backend(self, context: ActionContext) -> Optional[ClipBackend]:
        if self._backend_for is not None:
            return self._backend_for(context)
        return self._backend

    @classmethod
    def display_name(cls) -> str:
        return "Record clip"

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "preroll_s": {
                "type": "float",
                "required": False,
                "default": 10.0,
                "help": (
                    "Seconds of context before the event. Clamped to what the source's "
                    "buffer actually holds — asking for more than the retention is a "
                    "misconfiguration, not a runtime surprise."
                ),
            },
            "postroll_s": {"type": "float", "required": False, "default": 5.0, "help": "Seconds after the event ends."},
            "max_length_s": {
                "type": "float",
                "required": False,
                "default": 180.0,
                "help": "Hard cap on clip length regardless of event duration.",
            },
            "output_dir": {
                "type": "str",
                "required": False,
                "default": str(DEFAULT_CLIP_ROOT),
                "help": "Directory clips are written to. Must be on a bind-mounted path to survive a restart.",
            },
        }

    def plan_window(self, signal: WatchEventSignal, context: ActionContext) -> Tuple[datetime, float]:
        """Resolve the wallclock window to cut, with pre-roll clamped to the buffer.

        Clamping happens here as well as at config-save time because the buffer is RAM:
        a relay restart shortens it without warning, so the save-time check alone is not
        enough.
        """
        effective_preroll = self.preroll_s
        if context.retention_s > 0:
            effective_preroll = min(effective_preroll, context.retention_s)

        # Rule timing is monotonic; the buffer is addressed by wallclock.
        start_wall = signal.started_at + context.wallclock_offset
        end_wall = signal.at + context.wallclock_offset

        start = datetime.fromtimestamp(start_wall - effective_preroll, tz=timezone.utc)
        duration = (end_wall - start_wall) + effective_preroll + self.postroll_s
        return start, min(duration, self.max_length_s)

    def clip_path(self, signal: WatchEventSignal, context: ActionContext, start: datetime) -> Path:
        from .mqttPublishAction import slugify_topic

        day = start.strftime("%Y-%m-%d")
        stamp = start.strftime("%H-%M-%S")
        source = slugify_topic(context.source_name or context.source_uuid or "source")
        rule = slugify_topic(signal.rule_name)
        return self.output_dir / day / f"{stamp}_{source}_{rule}.mp4"

    async def _assemble_from_chunks(
        self, signal: WatchEventSignal, context: ActionContext, dest: Path
    ) -> Dict[str, Any]:
        """Build the clip from drained chunks plus whatever is still in the buffer.

        Taken only when the event outlived the rewind window. The chunks cover from the
        pre-roll up to the last drain; everything after that is still buffered, so the
        tail is cut the ordinary way and concatenated on. Without the tail the clip would
        end at the last drain rather than at the event, losing up to one drain interval
        of the ending — which is the part the user is most likely to be watching for.
        """
        from ..recordings.clipDrain import assemble

        chunks = list(context.clip_chunks)
        tail_start = chunks[-1].end
        end_wall = signal.at + context.wallclock_offset
        tail_duration = (datetime.fromtimestamp(end_wall, tz=timezone.utc) - tail_start).total_seconds()
        tail_duration += self.postroll_s

        tail: Optional[Path] = None
        if tail_duration > 0:
            backend = self._resolve_backend(context)
            candidate = dest.with_suffix(".tail.mp4")
            try:
                await backend.extract(context, tail_start, tail_duration, candidate)
                tail = candidate
            except Exception as e:
                # The chunks are the bulk of the clip and are already on disk; losing the
                # last few seconds is far better than losing the whole event.
                logger.warning("Tail extraction failed for %s, writing chunks only: %s", signal.rule_name, e)

        try:
            result = await assemble(chunks, tail, dest, max_length_s=self.max_length_s)
        finally:
            if tail is not None:
                tail.unlink(missing_ok=True)
        result["requestedStart"] = chunks[0].start.isoformat()
        # A clip built this way always reached back past the buffer, so its pre-roll is
        # whole unless a drain was actually missed — which `hasGap` reports.
        result["partialPreroll"] = bool(result.get("hasGap"))
        return result

    async def run(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        # Only the closing edge can be cut: the window isn't known until the event ends,
        # which is precisely what a retroactive buffer buys us.
        if signal.phase is not EventPhase.END:
            return ActionResult(ok=True, detail="not an event end; nothing to cut")
        backend = self._resolve_backend(context)
        if backend is None:
            return ActionResult(ok=False, detail="No clip backend available for this source")

        start, duration = self.plan_window(signal, context)
        dest = self.clip_path(signal, context, start)
        try:
            if context.clip_chunks:
                result = await self._assemble_from_chunks(signal, context, dest)
                return ActionResult(
                    ok=True, detail=f"wrote {dest.name} from {result['partCount']} part(s)", data=result
                )
            result = await backend.extract(context, start, duration, dest)
            return ActionResult(ok=True, detail=f"wrote {dest.name}", data=result)
        except Exception as e:
            # Past the buffer window this is permanent, not transient — the caller marks
            # the recording row accordingly rather than queueing an endless retry.
            logger.warning("Clip extraction failed for %s: %s", signal.rule_name, e)
            return ActionResult(ok=False, detail=str(e), data={"expired": "aged out" in str(e).lower()})
