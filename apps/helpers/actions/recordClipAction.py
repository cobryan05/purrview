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
import bisect
import logging
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from apps import RECORDINGS_CLIP_DIR

from ..timeUtils import parse_iso_utc
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
        # The spool holds every frame since the event opened, including the pre-roll the
        # ring was seeded from; the ring holds only what its caps still allow. Preferring
        # the spool is what makes a clip's length a question about disk rather than RAM.
        frames = self._spooled_or_ring_frames(start.timestamp(), start.timestamp() + duration_s)
        lead_in: List[Tuple[float, bytes]] = []
        if not frames or frames[0][0] > start.timestamp() + 0.5:
            # The dense ring didn't reach back to the requested pre-roll start — see
            # `FrameRing._sticky_recent`. Splice in whatever sticky stills predate
            # it, so the clip opens on an honest lead-in instead of either nothing or a
            # still that never appears in the video (the bug this replaces).
            lead_in = self._montage(self._ring.lead_in_frames(before=start.timestamp()), start.timestamp(), frames)
        if not frames and not lead_in:
            raise ValueError("Frame ring holds nothing for the requested window")
        size = await asyncio.to_thread(self._write_timeline, lead_in, frames, dest)
        all_frames = lead_in + frames
        actual_start = datetime.fromtimestamp(all_frames[0][0], tz=timezone.utc)
        wall_span = max(0.1, all_frames[-1][0] - all_frames[0][0])
        # Measured from the file, never assumed from the frames' wall span. The two
        # differ whenever the encoder cannot play the frames back at wall speed — a
        # compressed lead-in, or a rate clamp — and every consumer of `durationS` treats
        # it as the length of the video it can actually seek within. A clip that claims
        # to be three times longer than it is sends every seek past its end.
        played = await asyncio.to_thread(self._probe_duration, dest)
        # Seconds at the head of the requested window with no dense frames behind them.
        # `frames` is what the ring still held; anything earlier the caller asked for
        # was evicted. When there are no dense frames at all, the lead-in still's
        # timestamp is clamped to `start` (above), so it can't be used to measure the
        # gap — the whole requested window is missing.
        head_gap_s = duration_s if not frames else max(0.0, frames[0][0] - start.timestamp())
        # The montage's own length, not the gap it stands in for: the stills are clamped
        # into a short run and the rest of the wake gap is simply absent from the file,
        # so calling that stretch "stills" would make the player describe footage the
        # clip does not contain. `partialPreroll` reports the gap either way.
        stills_only_s = 0.0
        if lead_in:
            body = frames[0][0] if frames else lead_in[-1][0] + LAST_FRAME_HOLD_S
            stills_only_s = max(0.0, body - lead_in[0][0])
        return {
            "path": str(dest),
            "sizeBytes": size,
            "startedAt": actual_start.isoformat(),
            "durationS": played if played else wall_span,
            # What the frames actually covered, kept for the gap this exposes: a clip
            # whose span exceeds its duration is time-compressed, and a reader comparing
            # the two is the only way to notice.
            "sourceSpanS": wall_span,
            "requestedStart": start.isoformat(),
            "requestedDurationS": duration_s,
            # Measured against the *dense* frames, not the spliced stills. The stills
            # are two frames standing in for however long the ring could not hold, so
            # counting them as coverage reported a complete clip for an event whose
            # body had been evicted — which is exactly the case a reviewer needs told.
            "partialPreroll": head_gap_s > 0.5,
            # How much of the requested window has stills instead of video, so the
            # event can say so rather than presenting the gap as a lead-in. The byte
            # cap evicts under the hold (see `_evict`), so a visit longer than the ring
            # can afford loses its head however long `ring_seconds` is.
            "stillsOnlyS": round(stills_only_s, 2),
            "lowFps": True,
            "leadInFrames": len(lead_in),
        }

    @staticmethod
    def _montage(
        stills: List[Tuple[float, bytes]],
        window_start: float,
        frames: List[Tuple[float, Any]],
    ) -> List[Tuple[float, bytes]]:
        """Sticky stills as a quick run ending where the dense frames begin.

        A sticky still can be many minutes old and the wake gap it stands in for is many
        seconds, so giving the head its real duration opens the clip on a freeze long
        enough to read as a broken video rather than as context. The stills are clamped
        into `LEAD_IN_MONTAGE_S` instead and the clip simply starts later — the stretch
        it no longer covers held nothing but that same motionless frame.

        The *frames* are clamped, never the timeline. Each still still sits at a real
        offset from the clip's first frame, so one video second stays one world second
        and `build_overlay`'s wallclock subtraction stays exact. Compressing the head
        instead would shift every box in the body by the amount removed.
        """
        if not stills:
            return []
        anchor = frames[0][0] if frames else window_start + LEAD_IN_MONTAGE_S
        span = min(LEAD_IN_MONTAGE_S, max(0.0, anchor - window_start))
        if span <= 0:
            return []
        step = span / len(stills)
        first = anchor - span
        return [(first + index * step, blob) for index, (_stamp, blob) in enumerate(stills)]

    def _spooled_or_ring_frames(self, start: float, end: float) -> List[Tuple[float, bytes]]:
        """This event's frames for the window, from the spool where there is one.

        Spooled frames are returned as `(timestamp, path)` — the caller only ever hands
        them to the encoder, which wants paths anyway, and reading a long event's frames
        back into RAM to encode them would give back exactly what spooling bought.
        """
        spool = getattr(self._ring, "_spool", None)
        if spool is not None and spool.active:
            spooled = spool.frames_between(start, end)
            if spooled:
                return spooled  # type: ignore[return-value]
            if spool.frame_count:
                # The spool has frames, just none in this window — every write for it
                # failed (see `FrameSpool.push`'s dropped-OSError path), or the window
                # predates spooling. Either way the RAM-ring fallback below is degraded
                # coverage silently standing in for what should have come from disk.
                logger.warning(
                    "Spool held %d frame(s) but none for the requested window; falling back to the RAM ring",
                    spool.frame_count,
                )
        return self._ring.frames_between(start, end)

    @staticmethod
    def _write_timeline(lead_in: List[Tuple[float, bytes]], frames: List[Tuple[float, Any]], dest: Path) -> int:
        """Encode the lead-in stills and the event's frames as one honest timeline.

        One encode over one list, rather than two constant-rate segments stream-copied
        together: that seam is what put 61 seconds of wall time into 2 seconds of video
        and left every box in the clip pointing at the wrong moment. See
        `encode_timeline` for why a frame rate is the wrong concept here at all.

        Bytes still arrive for the sticky stills (they live in RAM, outside the spool),
        so they are written out beside the list and cleaned up with it.
        """
        with tempfile.TemporaryDirectory(prefix="ring-stills-") as tmp:
            tmp_dir = Path(tmp)
            timeline: List[Tuple[float, Path]] = []
            for index, (stamp, blob) in enumerate(lead_in):
                still = tmp_dir / f"lead{index:04d}.jpg"
                still.write_bytes(blob)
                timeline.append((stamp, still))
            for index, (stamp, source) in enumerate(frames):
                if isinstance(source, Path):
                    timeline.append((stamp, source))
                else:
                    spilled = tmp_dir / f"frame{index:06d}.jpg"
                    spilled.write_bytes(source)
                    timeline.append((stamp, spilled))
            return encode_timeline(timeline, dest)

    @staticmethod
    def _probe_duration(path: Path) -> Optional[float]:
        """The encoded file's own playback length, or None when it cannot be read.

        None rather than a raised error: a clip that was written is worth keeping even
        if ffprobe is unavailable, and the caller falls back to the frames' wall span.
        """
        try:
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                timeout=30,
            )
            value = float(proc.stdout.decode("utf-8", "replace").strip())
            return value if value > 0 else None
        except (subprocess.SubprocessError, ValueError, OSError):
            return None


#: What the final frame is held for, having no successor to measure against.
LAST_FRAME_HOLD_S = 0.5

#: Total video time the spliced idle stills are given, however long a gap they cover.
#: Short enough to read as a lead-in rather than as a stalled player, long enough that
#: each of the two or three stills is legible.
LEAD_IN_MONTAGE_S = 1.2

#: How far an assembled clip's measured length may fall short of its parts' before the
#: shortfall counts as missing video rather than keyframe alignment and muxer rounding.
_ASSEMBLY_TOLERANCE_S = 0.5


def _windows_map(covered_ranges: Any, measured_s: float) -> Optional[List[Dict[str, Any]]]:
    """A segment map from a backend's `coveredRanges`, or None when it cannot be trusted."""
    from ..recordings.clipDrain import map_from_windows

    windows = [
        (parse_iso_utc(w.get("startedAt")), float(w.get("durationS") or 0.0))
        for w in (covered_ranges or [])
        if parse_iso_utc(w.get("startedAt")) is not None
    ]
    return map_from_windows(windows, measured_s) if windows else None


def _canvas_for(frames: Sequence[Tuple[float, Path]]) -> Optional[Tuple[int, int]]:
    """The geometry to encode a mixed-size timeline into: whichever size most frames are.

    One clip's frames are not all one size. A Ring source's sticky idle snapshot and its
    live frames come off different tiers at different resolutions, and both are spliced
    into the same timeline. ffmpeg does not complain about that — it configures the filter
    graph from the *first* frame and silently squeezes every later one into it, so a
    snapshot leading 1920x1080 footage produced a 640x480 clip with the event body
    stretched to 4:3 (measured: PSNR 32.5 against 46.3 for the same frames encoded alone).

    Most frames wins rather than the largest, because the event body is what the clip is
    for and it outnumbers a lead-in still by two orders of magnitude; sizing to the
    largest would upscale the whole body to match one stale snapshot. A tie goes to the
    larger frame. Returns even dimensions, which yuv420p requires.
    """
    counts: Dict[Tuple[int, int], int] = {}
    for _stamp, path in frames:
        try:
            with Image.open(path) as im:
                size = im.size
        except (OSError, ValueError):
            # A frame ffmpeg may still manage to read; it just gets no say in the canvas.
            continue
        counts[size] = counts.get(size, 0) + 1
    if not counts:
        return None
    width, height = max(counts, key=lambda size: (counts[size], size[0] * size[1]))
    return width + width % 2, height + height % 2


def encode_timeline(frames: Sequence[Tuple[float, Path]], dest: Path) -> int:
    """Encode frames so each is shown for the time that actually passed after it.

    The clip's timeline then *is* the event's: a box the sidecar records at t=41s is at
    t=41s in the video. Everything downstream assumes that — the overlay, seeking, the
    correction harvest — and the previous encoder broke it in two ways at once. It chose
    one constant frame rate for a whole segment, so a camera delivering 0.32 fps was
    played back at 1 fps (three times too fast); and where sticky stills were spliced in
    front, they were encoded at a *different* constant rate and stream-copied on, so the
    two halves ran on different clocks.

    Variable frame durations remove the concept of a rate altogether. Implemented with
    the concat demuxer rather than `image2pipe`, because a pipe carries no per-frame
    timing and `-framerate` is exactly the constant this exists to get rid of.

    @returns the encoded size in bytes.
    """
    if not frames:
        raise ValueError("No frames to encode")
    dest.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(frames, key=lambda f: f[0])

    lines: List[str] = []
    for index, (stamp, path) in enumerate(ordered):
        lines.append(f"file '{path.as_posix()}'")
        if index + 1 < len(ordered):
            # The true gap, never a capped one. Shortening any frame's hold shifts every
            # frame after it earlier in video time while the sidecar goes on quoting the
            # original moment — which is the timeline break this function exists to end.
            # A long freeze is honest; a silently re-timed clip is not.
            lines.append(f"duration {max(ordered[index + 1][0] - stamp, 0.001):.3f}")
        else:
            lines.append(f"duration {LAST_FRAME_HOLD_S:.3f}")
    # The demuxer applies a `duration` to the file *before* it, and drops the last one
    # unless a file follows it — so the final entry is repeated to give it its hold.
    lines.append(f"file '{ordered[-1][1].as_posix()}'")

    canvas = _canvas_for(ordered)
    if canvas is None:
        vf = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    else:
        width, height = canvas
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        )

    with tempfile.TemporaryDirectory(prefix="ring-timeline-") as tmp:
        list_path = Path(tmp) / "frames.txt"
        list_path.write_text("\n".join(lines) + "\n")
        target = dest.with_suffix(dest.suffix + ".part")
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "concat",
            # The list names absolute paths outside the list's own directory.
            "-safe",
            "0",
            "-i",
            str(list_path),
            # Keep the per-frame timing the list describes rather than resampling it to
            # a constant rate, which is the whole point.
            "-vsync",
            "vfr",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            # yuv420p, not the JPEG-native 4:4:4: anything else is unplayable in a
            # browser <video> element, which is the whole point of writing a clip.
            "-pix_fmt",
            "yuv420p",
            # Every frame into one geometry, so a mixed-resolution timeline is letterboxed
            # rather than stretched — see `_canvas_for`. Doubles as the even-dimension
            # guarantee yuv420p needs, which a snapshot camera is under no obligation to
            # deliver on its own.
            "-vf",
            vf,
            "-f",
            "mp4",
            str(target),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=300)
            if proc.returncode != 0 or not target.exists():
                raise RuntimeError(f"ffmpeg failed writing ring clip: {proc.stderr[-300:].decode('utf-8', 'replace')}")
            size = target.stat().st_size
            target.replace(dest)
            return size
        except BaseException:
            target.unlink(missing_ok=True)
            raise


class FrameSpool:
    """An open event's frames, on disk instead of in RAM.

    The ring exists to answer a question that can only be answered retroactively — "what
    did the ten seconds before this event look like?" — and for that it has to hold
    frames nobody has asked for yet, which is why it is bounded and why it evicts. Once
    an event *opens*, that justification is gone: every frame from here on is known to be
    wanted, so it can go straight to disk.

    That is the whole point of this class. The ring's byte cap evicts under the hold, so
    a visit longer than the ring could afford lost its head while it was still being
    recorded — an 88-second event whose video was 26 seconds, the gap papered over with
    stills. Spooling makes a clip's length a question about disk, not memory, which is
    the same trade `clipDrain` already makes for cameras that have a real rewind buffer.

    Frames are written under a per-event directory and removed with it. Nothing here
    encodes: `encode_timeline` reads the spool back at the closing edge.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._dir: Optional[Path] = None
        self._frames: List[Tuple[float, Path]] = []
        self._seq = 0
        #: Latest frame timestamp any open event can still read, or None for no bound.
        #: See `limit_to` — without it the spool writes for as long as an event stays
        #: open, and a rule that never closes fills the disk with frames nothing reads.
        self._deadline: Optional[float] = None
        self._dropped = 0

    @property
    def active(self) -> bool:
        return self._dir is not None

    @property
    def dropped_count(self) -> int:
        """Frames refused because they fall past `limit_to`'s deadline."""
        return self._dropped

    def limit_to(self, deadline: Optional[float]) -> None:
        """Stop spooling frames later than `deadline`; `None` lifts the bound.

        A clip's window ends at `min(event duration + roll, max_length_s)`, so frames
        past the last open event's cap are written, counted against the recordings size
        cap by `_scan_recording_files`, and then never read by anything. A rule that
        stays open for hours — a lit driveway, a rule whose exit condition never fires —
        turned that into gigabytes of jpgs whose only effect was to evict real clips.

        The deadline belongs to the *source*, not to one event, for exactly the reason
        `FrameRing.hold_from` documents: several events are routinely open over one ring,
        and the one that opened last needs frames the first one does not. It must
        therefore be the latest deadline any currently open event needs, recomputed from
        live state each pass — `StreamWatcherTask._service_ring_hold` does both together.
        """
        self._deadline = deadline

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def start(self) -> None:
        """Begin spooling. Idempotent, so it can be called from a per-pass service."""
        if self._dir is not None:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        self._dir = Path(tempfile.mkdtemp(prefix="event-", dir=str(self._root)))

    def push(self, timestamp: float, jpeg: bytes) -> None:
        """Write one frame. A failed write is dropped, never raised.

        This sits on the per-frame path of a watcher that runs for weeks: a full disk
        must cost the clip its frames, not take down the analysis loop behind it.
        """
        if self._dir is None:
            return
        if self._deadline is not None and timestamp > self._deadline:
            # Past every open event's cap, so no clip will ever read it. Logged once per
            # event rather than per frame: at 2 fps a long event would emit thousands.
            self._dropped += 1
            if self._dropped == 1:
                logger.info("Spool reached the longest open clip's cap; later frames are not being kept")
            return
        path = self._dir / f"{self._seq:08d}.jpg"
        try:
            path.write_bytes(jpeg)
        except OSError:
            logger.warning("Could not spool a frame for an open event", exc_info=True)
            return
        self._seq += 1
        self._frames.append((float(timestamp), path))

    def frames_between(self, start: float, end: float) -> List[Tuple[float, Path]]:
        return [f for f in self._frames if start <= f[0] <= end]

    def discard(self) -> None:
        """Drop every spooled frame. Safe to call when nothing was ever spooled."""
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
        self._dir = None
        self._frames = []
        self._seq = 0
        self._deadline = None
        self._dropped = 0


def sweep_spool_root(spool_root: Path) -> int:
    """Delete spooled events left behind by a previous process. Returns how many.

    Same reasoning as `clipDrain.sweep_chunk_root`: a killed watcher never reaches the
    closing edge that calls `FrameSpool.discard()`, so its per-event directory is a
    disk leak nothing else would ever clean up.
    """
    root = Path(spool_root)
    if not root.exists():
        return 0
    removed = 0
    for path in root.glob("*/event-*"):
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError:
            logger.debug("Could not sweep spooled event %s", path, exc_info=True)
    return removed


#: Oldest a sticky still may be, relative to a clip's pre-roll start, to open that clip.
#:
#: The lead-in exists to show what the scene looked like just before the event, and a
#: still only speaks for the moment it was taken. A snapshot camera gives no way to tell
#: a static scene from a stale one, so the older the still, the more likely the clip
#: opens on something the world has already moved past.
#:
#: 300s deliberately matches `RingMqttImageProvider.IDLE_FRAME_TIMEOUT` — the window
#: beyond which the provider itself stops expecting a frame. ring-mqtt polls every 90s
#: by default, so a healthy camera's newest still is never close to this bound; it fires
#: only once polling has actually stalled, which is exactly when a still stops being
#: evidence. Past it the clip opens on its own first frame instead, and `stillsOnlyS`
#: already records the gap that leaves.
LEAD_IN_MAX_AGE_S = 300.0


class FrameRing:
    """Bounded in-RAM JPEG ring for sources with no continuous stream.

    Bounded by **bytes, not frame count**: frame sizes vary by an order of magnitude
    between cameras, so a count-based bound gives wildly different memory footprints
    for the same configuration.
    """

    def __init__(
        self,
        max_seconds: float = 20.0,
        max_mb: float = 32.0,
        sticky_idle_count: int = 2,
        lead_in_max_age_s: float = LEAD_IN_MAX_AGE_S,
        spool: Optional[FrameSpool] = None,
    ) -> None:
        self._max_seconds = max_seconds
        self._max_bytes = int(max_mb * 1024 * 1024)
        self._lead_in_max_age_s = float(lead_in_max_age_s)
        self._frames: List[Tuple[float, bytes]] = []
        self._bytes = 0
        self._hold_from: Optional[float] = None
        #: Where frames go once an event is open — see `FrameSpool`. Without one the
        #: ring behaves exactly as it did: bounded RAM, and a long event loses its head.
        self._spool = spool
        #: The most recent frames from a source that polls, kept outside the age/byte-
        #: bounded ring above. Ring cameras only push idle snapshots every 30-90s (the
        #: ring-mqtt bridge's own poll interval), far slower than `max_seconds` (default
        #: 20s), so every idle snapshot is age-evicted from `_frames` before the next one
        #: even arrives — the ring holds real pre-roll only for events lucky enough to
        #: start within seconds of a poll. This is a separate, small, deliberately
        #: unbounded-by-age cache so `lead_in_frames` can still offer *something* — a
        #: couple of stale-but-honest stills — as a clip's lead-in when the dense ring
        #: has nothing.
        #:
        self._sticky_recent: List[Tuple[float, bytes]] = []
        self._sticky_idle_count = int(sticky_idle_count)
        #: The last frame seen on the *live* tier, offered alongside the idle stills.
        #:
        #: A Ring camera stops polling idle the moment it goes live, so an idle-only
        #: cache is blind to everything a live session saw and keeps offering the last
        #: poll from before it. That is a clip opening on a scene its next frame
        #: contradicts — a garage with both cars parked, a hundred seconds after one of
        #: them drove away.
        #:
        #: Kept as one slot of its own rather than pushed into `_sticky_recent`: at 1-2
        #: fps a live session would flood a two-slot cache with consecutive frames from
        #: *inside* the next event's window, which `lead_in_frames` then filters out —
        #: trading a stale lead-in for none at all.
        self._last_live: Optional[Tuple[float, bytes]] = None
        #: Whether this source has ever pushed an idle frame. Continuous sources (RTSP)
        #: never do: their dense ring is the only pre-roll there is, and the sticky cache
        #: must stay empty rather than start splicing stills into clips that never
        #: needed them.
        self._has_idle_tier = False

    def hold_from(self, floor: Optional[float], until: Optional[float] = None) -> None:
        """Suspend time-based eviction for frames at/after `floor`; `None` resumes it.

        `until` is passed straight to the spool as the latest frame any open event can
        still read (see `FrameSpool.limit_to`). It bounds only the spool: the RAM ring is
        already bounded by its own caps, and its later frames still answer historical-
        still lookups.

        `_evict`'s `max_seconds` cutoff climbs with every push, so a visit longer than
        `max_seconds` has its own pre-roll evicted out from under it before
        `RecordClipAction` ever reads the ring at `EventPhase.END`. The byte cap still
        applies underneath this — only the age-based cap is suspended.

        ⚠️ **The floor belongs to the source, not to one event.** One ring serves every
        rule watching a camera, and the engine runs a state machine per
        `(rule, subject, target)`, so several events are routinely open over it at once.
        Callers must therefore pass the earliest floor any *currently open* event needs,
        recomputed from live state — `StreamWatcherTask._service_ring_hold` does this
        once a pass, the same way `_service_drain` arms the chunk drain.

        It is deliberately not a hold/release pair taken on the event edges. That is what
        this was, and it failed twice over: a second event's START overwrote the first
        event's floor, and whichever event closed first released the hold for all of
        them — so a neighbour still recording had its clip cut from a ring holding only
        the last few seconds. An edge-driven hold also leaks its floor forever whenever
        an END never arrives, pinning every frame the ring has in RAM.
        """
        was_holding = self._hold_from is not None
        self._hold_from = floor
        if floor is None:
            # Nothing is open, so the spooled event is finished with — the clip was cut
            # at the closing edge, before this ran (see `_service_ring_hold`).
            if self._spool is not None:
                self._spool.discard()
            if self._frames:
                self._evict(self._frames[-1][0])
            return
        if self._spool is not None:
            self._spool.limit_to(until)
        if self._spool is not None and not was_holding:
            # An event just opened. The pre-roll it wants is the RAM the ring is already
            # holding, so seed the spool with it — retroactive pre-roll is the one thing
            # spooling cannot reconstruct, and losing it would trade one gap for another.
            self._spool.start()
            for stamp, jpeg in self._frames:
                if stamp >= floor:
                    self._spool.push(stamp, jpeg)

    def push(self, timestamp: float, jpeg: bytes, idle: bool = False) -> None:
        """`idle=True` marks a slow-poll snapshot rather than a dense live frame —
        callers pass it through from the provider's own tier. It is what tells the ring
        this source *has* an idle tier; after that every frame feeds the sticky lead-in
        cache regardless of tier, so a live session cannot leave it stale. See
        `_sticky_recent`."""
        self._frames.append((timestamp, jpeg))
        self._bytes += len(jpeg)
        # To disk as well while an event is open. The RAM copy still ages out under the
        # caps below; the spool is what the clip is actually cut from, and it is the
        # reason a long visit no longer loses its head to the byte cap.
        if self._spool is not None and self._spool.active:
            self._spool.push(timestamp, jpeg)
        self._evict(timestamp)
        if idle:
            self._has_idle_tier = True
            self._sticky_recent.append((timestamp, jpeg))
            if len(self._sticky_recent) > self._sticky_idle_count:
                self._sticky_recent.pop(0)
        elif self._has_idle_tier:
            self._last_live = (timestamp, jpeg)

    def _evict(self, now: float) -> None:
        cutoff = now - self._max_seconds
        if self._hold_from is not None:
            cutoff = min(cutoff, self._hold_from)
        while self._frames and (self._frames[0][0] < cutoff or self._bytes > self._max_bytes):
            _, data = self._frames.pop(0)
            self._bytes -= len(data)

    def frames_between(self, start: float, end: float) -> List[Tuple[float, bytes]]:
        return [f for f in self._frames if start <= f[0] <= end]

    def first_at_or_after(self, ts: float) -> Optional[Tuple[float, bytes]]:
        """The earliest held frame at or after `ts`, or None.

        Used to draw a still from the same pre-roll instant a clip's own window starts
        from, rather than a separately-tolerant nearest-frame search that can land on a
        different moment than the video ever contains.
        """
        for f in self._frames:
            if f[0] >= ts:
                return f
        return None

    def lead_in_frames(self, before: float) -> List[Tuple[float, bytes]]:
        """Sticky stills strictly before `before`, oldest first.

        For a Ring-style source, `before` is a clip's requested pre-roll start. The
        dense ring frequently has nothing that far back (see `_sticky_recent`'s docstring
        above) — this is what `RingClipBackend` prepends instead, so the clip opens on
        a couple of honest, if stale, stills rather than jumping straight into the event
        with no lead-in at all.
        """
        held = list(self._sticky_recent)
        if self._last_live is not None:
            held.append(self._last_live)
        held.sort(key=lambda f: f[0])
        # The newest few, because `RingClipBackend` splices only the last of them: what
        # the camera saw most recently before the window is the honest lead-in, whether
        # that was an idle poll or the tail of a live session.
        fresh = [f for f in held if f[0] < before and before - f[0] <= self._lead_in_max_age_s]
        return fresh[-self._sticky_idle_count :]

    def closest(self, ts: float, tolerance: float) -> Optional[bytes]:
        """The frame nearest `ts`, if the ring holds one within `tolerance` seconds.

        For a source with no rewind buffer, this is the same ring `RecordClipAction`
        already cuts pre-roll from — reused here so an on_exit event's still-image
        preview can show the subject instead of the empty frame the ring has moved on
        to by the time the debounced departure signal is dispatched.
        """
        if not self._frames:
            return None
        times = [f[0] for f in self._frames]
        i = bisect.bisect_left(times, ts)
        candidates = [self._frames[j] for j in (i - 1, i) if 0 <= j < len(self._frames)]
        if not candidates:
            return None
        best = min(candidates, key=lambda f: abs(f[0] - ts))
        return best[1] if abs(best[0] - ts) <= tolerance else None

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
        save_overlay: bool = True,
        backend: Optional[ClipBackend] = None,
        backend_for: Optional[Callable[[ActionContext], Optional[ClipBackend]]] = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.preroll_s = float(preroll_s)
        self.postroll_s = float(postroll_s)
        self.max_length_s = float(max_length_s)
        self.output_dir = Path(output_dir)
        self.save_overlay = bool(save_overlay)
        # Claimed unconditionally: whether a clip is repaired or merely scanned is a
        # property of its source, which is not known until an event fires. The worker
        # builds nothing until the first submit, so an action that never submits pays
        # only the refcount.
        from ..recordings.repairClip import retain_repairs

        retain_repairs()
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
    def logs_event(cls) -> bool:
        # The clip is attached to the event row, so the event has to exist to hold it.
        return True

    def wrote_to_event(self, result) -> bool:
        # No mp4, no event. `run` returns ok for several outcomes that produce no file —
        # a phase that is not END, a source with no backend, a window with nothing in it.
        return bool(result.ok and result.data.get("path"))

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "preroll_s": {
                "type": "float",
                "required": False,
                "default": 10.0,
                "help": (
                    "Seconds of video to include from before the event began, so the clip shows "
                    "the approach rather than opening with the subject already in shot. This is "
                    'the place to choose the lead-in; the camera\'s own "rewind s" setting is '
                    "only the ceiling on how far back it is able to reach. Asking for more than "
                    "the camera holds quietly gives you as much as it has."
                ),
            },
            "postroll_s": {
                "type": "float",
                "required": False,
                "default": 5.0,
                "help": (
                    "Seconds of video to keep after the event ends, so the clip does not cut "
                    "away the moment the subject stops being detected. Recording continues for "
                    "this long after the event closes, so a large value delays the clip "
                    "appearing in the events list by the same amount."
                ),
            },
            "max_length_s": {
                "type": "float",
                "required": False,
                "default": 180.0,
                "help": (
                    "Longest a single clip may be, in seconds, however long the event runs. "
                    "Reaching it ends the video early — the event itself is unaffected, and "
                    "nothing is recorded for the remainder of it. A safety net against one "
                    "stuck detection filling the disk."
                ),
            },
            "output_dir": {
                "type": "str",
                "required": False,
                "default": str(DEFAULT_CLIP_ROOT),
                "help": (
                    "Folder clips are written to. Leave this alone unless you know you want "
                    "them elsewhere: at the default they are stored with the rest of the "
                    "recordings, played from the events list, and cleaned up on the schedule in "
                    "Settings → Recording retention. A folder outside that is not tidied up for "
                    "you, and one that is not on storage mounted into the container is lost "
                    "whenever it restarts."
                ),
            },
            "save_overlay": {
                "type": "bool",
                "required": False,
                "default": True,
                "help": (
                    "Save the detection boxes alongside the clip, so the player can draw "
                    "what the watcher saw and mark when the rule was firing. Costs a few "
                    "tens of KB and no re-encoding; the clip itself stays unannotated, so "
                    "frames saved for annotation never have boxes burnt into them."
                ),
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
        tail_played = 0.0
        tail_actual_start: Optional[datetime] = None
        tail_covered: tuple = ()
        if tail_duration > 0:
            backend = self._resolve_backend(context)
            candidate = dest.with_suffix(".tail.mp4")
            try:
                cut = await backend.extract(context, tail_start, tail_duration, candidate)
                tail = candidate
                # What the backend actually cut, not what was asked for: the buffer can
                # hold less than the request, and this figure is what the clip's playing
                # time is built from below.
                tail_played = float(cut.get("durationS") or 0.0)
                # The instant the tail really starts at. A tail clamped forward of
                # `tail_start` leaves a hole between it and the last chunk, and this is
                # the only report of it.
                tail_actual_start = parse_iso_utc(cut.get("startedAt")) or tail_start
                tail_covered = tuple(
                    (parse_iso_utc(w.get("startedAt")), float(w.get("durationS") or 0.0))
                    for w in (cut.get("coveredRanges") or [])
                    if parse_iso_utc(w.get("startedAt")) is not None
                )
            except Exception as e:
                # The chunks are the bulk of the clip and are already on disk; losing the
                # last few seconds is far better than losing the whole event.
                logger.warning("Tail extraction failed for %s, writing chunks only: %s", signal.rule_name, e)

        try:
            result = await assemble(
                chunks,
                tail,
                dest,
                max_length_s=self.max_length_s,
                tail_start=tail_start,
                tail_actual_start=tail_actual_start,
                tail_covered=tail_covered,
            )
        finally:
            if tail is not None:
                tail.unlink(missing_ok=True)
        result["requestedStart"] = chunks[0].start.isoformat()
        # The sum of what the parts hold, capped the same way `concat_parts` caps the
        # muxer. Reporting nothing here is not a neutral omission: the recording row
        # stores 0, `ended_at` lands empty, and the overlay writer — which needs a
        # window to slice the detection ring over — silently declines every assembled
        # clip.
        played = sum(c.duration_s for c in chunks) + tail_played
        if self.max_length_s and self.max_length_s > 0:
            played = min(played, self.max_length_s)
        # The file wins wherever the two disagree. `durationS` is what a player seeks
        # within and what the overlay's box offsets are bounded by, so a figure longer
        # than the video sends both past its end.
        measured = result.pop("measuredDurationS", None)
        if measured is not None and measured > 0:
            if played - measured > _ASSEMBLY_TOLERANCE_S:
                # Short by more than segment rounding: video the parts claimed is not in
                # the file. A validated segment map still places every box correctly;
                # without one, wallclock does not map onto this file at all.
                result["hasGap"] = True
                logger.warning(
                    "Assembled clip %s holds %.2fs against %.2fs of parts; overlay %s",
                    dest.name,
                    measured,
                    played,
                    "mapped per segment" if result.get("segments") else "withheld",
                )
            played = measured
        result["durationS"] = max(0.0, played)
        # Must follow the measured-duration check above: a shortfall against what the
        # parts hold sets `hasGap` too.
        #
        # A clip built this way always reached back past the buffer, so its pre-roll is
        # whole unless a drain was missed (`hasGap`) or the first cut started later than
        # it was asked to (`headClamped`). The two are separate because only the former
        # invalidates the overlay; both mean the pre-roll is short.
        result["partialPreroll"] = bool(result.get("hasGap")) or bool(result.get("headClamped"))
        return result

    async def close(self) -> None:
        """Let go of the shared repair worker, which outlives any one action."""
        from ..recordings.repairClip import release_repairs

        await release_repairs()

    def _conceal_damage(self, result: Dict[str, Any], dest: Path, context: ActionContext) -> None:
        """Repair this clip if its camera opted in, or scan it to find out if it should.

        Queued either way rather than awaited: the watcher awaits its actions inside the
        frame loop, and a decode scan plus a re-encode is minutes during which the
        profile analyses none of its cameras. Safe to run after the row is written
        because concealment keeps the frame count and is discarded if the duration
        moved, so it cannot move a box; a ring-cut clip is whole JPEGs with no damaged
        slices, so it is skipped by both paths.
        """
        if result.get("lowFps"):
            return
        from ..recordings import clipRepairAdvice
        from ..recordings.repairClip import submit_probe, submit_repair

        if context.repair_corruption:
            submit_repair(dest)
        elif clipRepairAdvice.wants_probe(context.source_uuid):
            submit_probe(dest, source_uuid=context.source_uuid, source_name=context.source_name or "")

    def _check_video(self, result: Dict[str, Any], dest: Path, context: ActionContext) -> None:
        """Queue a check that the clip actually holds continuous video.

        Skipped for ring-cut clips: a snapshot-cadence source is gaps between frames by
        design, and would read as a camera that keeps losing its picture.
        """
        if result.get("lowFps"):
            return
        from ..recordings import clipVideoHealth

        clipVideoHealth.submit(dest, source_uuid=context.source_uuid, source_name=context.source_name or "")

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
                # An assembled clip does get an overlay, but only while its parts are
                # contiguous. The sidecar's contract is that a box's `t` is where it
                # lands in the video, and `build_overlay` gets there by subtracting
                # wallclocks — exact for a run of back-to-back drains, but wrong by the
                # length of any missing stretch for everything after it. `hasGap` is the
                # only place that is known, so it is the gate: a gapped clip goes without
                # rather than drawing boxes that slide further off the picture the longer
                # the event ran, which reads as a detector fault.
                # A gap only costs the overlay when nothing can place the boxes. The
                # segment map carries each surviving stretch's own video offset, so a
                # mapped clip is annotated across the hole; `segment_map` returns None
                # unless its own arithmetic matches the file, which is what keeps a bad
                # map from drawing boxes in the wrong place.
                result["overlay"] = self.save_overlay and (not result.get("hasGap") or bool(result.get("segments")))
                self._conceal_damage(result, dest, context)
                self._check_video(result, dest, context)
                return ActionResult(
                    ok=True, detail=f"wrote {dest.name} from {result['partCount']} part(s)", data=result
                )
            result = await backend.extract(context, start, duration, dest)
            # A single cut can come back short too — `/list` advertises the segment
            # MediaMTX is still writing, and ranges that leave holes. The window is one
            # file here, so a hole inside it breaks the overlay's wallclock subtraction
            # exactly as an inter-chunk gap does; the coverage report is what makes the
            # boxes placeable anyway.
            measured = float(result.get("durationS") or 0.0)
            charged = float(result.get("clampedDurationS") or measured)
            if charged - measured > _ASSEMBLY_TOLERANCE_S:
                result["hasGap"] = True
                result["segments"] = _windows_map(result.get("coveredRanges"), measured)
                logger.warning(
                    "Cut %s holds %.2fs against %.2fs charged; overlay %s",
                    dest.name,
                    measured,
                    charged,
                    "mapped per segment" if result.get("segments") else "withheld",
                )
            result["overlay"] = self.save_overlay and (not result.get("hasGap") or bool(result.get("segments")))
            self._conceal_damage(result, dest, context)
            self._check_video(result, dest, context)
            return ActionResult(ok=True, detail=f"wrote {dest.name}", data=result)
        except Exception as e:
            # Past the buffer window this is permanent, not transient — the caller marks
            # the recording row accordingly rather than queueing an endless retry.
            logger.warning("Clip extraction failed for %s: %s", signal.rule_name, e)
            return ActionResult(ok=False, detail=str(e), data={"expired": "aged out" in str(e).lower()})
