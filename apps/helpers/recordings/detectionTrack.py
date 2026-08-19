"""What the watcher saw, kept beside the clip it saw it in.

A clip on its own answers "what happened"; it cannot answer "what did the detector
think was happening", which is the question you ask when a rule fired on a shadow or
failed to fire on a cat. The frames the rules were evaluated against are gone the
moment the next one arrives — the watcher keeps pixels in a ring, but nothing kept the
*boxes*.

This keeps them: a bounded per-source ring of timestamped, normalised detections, and
a writer that slices the ring to a clip's window and drops it beside the mp4 as a JSON
sidecar. The player draws them over the video.

**Why a sidecar and not a second, annotated mp4.** Burning boxes into video costs a
full decode/re-encode of every clip on a machine already spending its CPU on
inference, doubles clip storage, and — worst — bakes the alignment in. Any drift
between the boxes and the picture becomes permanent and unfixable without regenerating
every file. A sidecar costs a few tens of KB, re-encodes nothing, can be toggled off in
the player, and leaves the clip itself pristine, which is what keeps "save this frame
for annotation" honest: the frame that reaches the dataset has never had a box drawn on
it.

**Coordinates are normalised (0..1).** Detections live in the *analysis* pixel space —
a downscale of the camera's frame whose width is a per-source setting — while the clip
is at the camera's own resolution. Storing pixels would make every box wrong by the
downscale ratio, and would silently rot if someone changed `analysis_width` later.

**Times are seconds from the clip's own first frame**, not wallclock, so the player can
use them against `video.currentTime` with no clock arithmetic. The conversion happens
here, once, against the clip's *actual* start — the buffer clamps a request that
reaches back further than it still holds, so the requested start is the wrong anchor
and using it would offset every box by the clamp.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Bumped when the shape changes incompatibly. The player refuses a version it does not
#: know rather than drawing boxes it has misread — a silently misplaced box is worse
#: than no overlay, because it looks like a detector bug.
OVERLAY_VERSION = 1

#: Sidecar extension. Appended to the clip's full name rather than replacing its suffix
#: (`clip.mp4.boxes.json`, not `clip.boxes.json`) so the two sort together and the
#: sidecar is obviously derived from that exact file.
OVERLAY_SUFFIX = ".boxes.json"

#: How far back the ring holds samples. Comfortably past the default rewind retention,
#: because pre-roll is cut from the buffer and a sidecar that starts where the event
#: started would leave the approach — the interesting part — unannotated.
DEFAULT_TRACK_SECONDS = 400.0

#: Hard bound on samples, independent of the time bound. A source analysed at 30 fps
#: would otherwise hold 12,000 samples for the same 400 s a 1 fps source holds 400 in.
DEFAULT_TRACK_SAMPLES = 6000

#: A sample is only used for a frame within this many seconds of it. Beyond that the
#: overlay reports nothing rather than holding the last known boxes indefinitely: a
#: watcher that stalls for ten seconds should show a gap, not a frozen cat.
DEFAULT_MAX_STALENESS_S = 1.5


def _round(value: float) -> float:
    """Four decimals — sub-pixel on any sane frame, and it halves the file size."""
    return round(float(value), 4)


def _normalise(bbox: Sequence[float], width: int, height: int) -> Optional[Dict[str, float]]:
    """Pixel corners in the analysis frame → normalised x/y/w/h, or None if degenerate.

    Clamped rather than dropped when a box runs off the edge: a subject walking out of
    shot is routine and its partial box is exactly what you want to see. Only a box with
    no area left after clamping is dropped, because it would draw as an invisible
    artifact someone would have to puzzle over.
    """
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    left = max(0.0, min(1.0, min(x1, x2) / width))
    top = max(0.0, min(1.0, min(y1, y2) / height))
    right = max(0.0, min(1.0, max(x1, x2) / width))
    bottom = max(0.0, min(1.0, max(y1, y2) / height))
    if right - left <= 0 or bottom - top <= 0:
        return None
    return {"x": _round(left), "y": _round(top), "w": _round(right - left), "h": _round(bottom - top)}


@dataclass
class TrackSample:
    """One analysed frame's worth of boxes, in normalised coordinates.

    `boxes` are the mask-filtered detections the tracker was fed — model output, so a
    flicker here is a detector problem. `tracks` are what the tracker made of them, so
    a box present in `boxes` and absent from `tracks` localises the fault to tracking
    rather than detection. Keeping both is the whole diagnostic value; keeping only the
    tracks would hide exactly the failures this exists to expose.
    """

    #: Wallclock epoch seconds. The clip window is addressed in wallclock, the frame
    #: loop runs on a monotonic clock, and the caller supplies the offset.
    at: float
    boxes: List[Dict[str, Any]] = field(default_factory=list)
    tracks: List[Dict[str, Any]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.boxes and not self.tracks


class DetectionTrack:
    """Bounded ring of recent `TrackSample`s for one source.

    Bounded by both age and count — see the two module constants. Memory is a few
    hundred KB at the defaults, versus the tens of MB the *frame* ring costs, because
    this holds numbers rather than pixels.
    """

    def __init__(
        self,
        max_seconds: float = DEFAULT_TRACK_SECONDS,
        max_samples: int = DEFAULT_TRACK_SAMPLES,
    ) -> None:
        self._max_seconds = float(max_seconds)
        self._max_samples = int(max_samples)
        self._samples: List[TrackSample] = []

    def push(
        self,
        at: float,
        detections: Sequence[Any],
        tracks: Sequence[Any],
        frame_size: Tuple[int, int],
    ) -> None:
        """Record one analysed frame. `frame_size` is the analysis frame's (w, h).

        Empty frames are recorded too, and deliberately: "the detector saw nothing here"
        is a fact the overlay needs. Without it the player cannot tell an empty frame
        from a gap in the record, so a detection that dropped out for two seconds would
        render as a box that simply persisted.
        """
        width, height = frame_size
        sample = TrackSample(
            at=float(at),
            boxes=[b for b in (self._detection_dict(d, width, height) for d in detections) if b],
            tracks=[t for t in (self._track_dict(t, width, height) for t in tracks) if t],
        )
        self._samples.append(sample)
        self._evict(sample.at)

    @staticmethod
    def _detection_dict(detection: Any, width: int, height: int) -> Optional[Dict[str, Any]]:
        box = _normalise(getattr(detection, "bbox", ()) or (), width, height)
        if box is None:
            return None
        # `label` is the resolved name and `labelUuid` the identity: the name is what a
        # human reads off the overlay, the uuid is what matches the colour the label has
        # everywhere else in the app, and a renamed label must not restyle old clips.
        box["label"] = getattr(detection, "label_name", None) or getattr(detection, "label_uuid", None) or ""
        box["labelUuid"] = getattr(detection, "label_uuid", None) or ""
        box["conf"] = round(float(getattr(detection, "confidence", 0.0) or 0.0), 3)
        return box

    @staticmethod
    def _track_dict(track: Any, width: int, height: int) -> Optional[Dict[str, Any]]:
        box = _normalise(getattr(track, "bbox", ()) or (), width, height)
        if box is None:
            return None
        box["id"] = int(getattr(track, "track_id", 0) or 0)
        box["label"] = getattr(track, "label_name", None) or getattr(track, "label_uuid", None) or ""
        box["labelUuid"] = getattr(track, "label_uuid", None) or ""
        box["conf"] = round(float(getattr(track, "confidence", 0.0) or 0.0), 3)
        return box

    def _evict(self, now: float) -> None:
        cutoff = now - self._max_seconds
        while self._samples and (self._samples[0].at < cutoff or len(self._samples) > self._max_samples):
            self._samples.pop(0)

    def samples_between(self, start: float, end: float) -> List[TrackSample]:
        """Samples inside `[start, end]`, oldest first."""
        return [s for s in self._samples if start <= s.at <= end]

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def coverage_s(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1].at - self._samples[0].at


def overlay_path_for(clip_path: Path) -> Path:
    """Where a clip's sidecar lives. Beside the clip, named after it."""
    return clip_path.with_name(clip_path.name + OVERLAY_SUFFIX)


def build_overlay(
    samples: Sequence[TrackSample],
    clip_start: datetime,
    duration_s: float,
    *,
    event: Optional[Dict[str, Any]] = None,
    max_staleness_s: float = DEFAULT_MAX_STALENESS_S,
) -> Dict[str, Any]:
    """Slice samples into a clip-relative overlay document.

    `clip_start` must be the clip's **actual** first frame, which for a buffer-cut clip
    is the clamped start the extractor reports and not the start that was requested.

    Samples outside `[0, duration_s]` are dropped rather than clamped to the ends:
    pinning an out-of-window sample to t=0 would draw a box at a moment the detector had
    not yet seen it, which reads as a false positive in the exact tool someone is using
    to judge false positives.
    """
    origin = clip_start.timestamp()
    frames: List[Dict[str, Any]] = []
    for sample in samples:
        offset = sample.at - origin
        if offset < 0 or offset > duration_s:
            continue
        frames.append({"t": _round(offset), "boxes": sample.boxes, "tracks": sample.tracks})

    document: Dict[str, Any] = {
        "version": OVERLAY_VERSION,
        "startedAt": clip_start.isoformat(),
        "durationS": _round(duration_s),
        # The player holds each sample's boxes until the next one, which is right for a
        # 5 fps overlay on 30 fps video but wrong across a gap. It needs to know how long
        # a hold is legitimate, and only this side knows the analysis cadence.
        "maxStalenessS": _round(max_staleness_s),
        "frames": frames,
    }
    if event:
        document["event"] = event
    return document


def event_window(
    clip_start: datetime,
    duration_s: float,
    event_start: Optional[datetime],
    event_end: Optional[datetime],
    rule_name: Optional[str] = None,
    event_uuid: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The rule's own span, in clip-relative seconds, for the player to mark.

    This is what turns the sidecar from "boxes" into "boxes plus when the rule believed
    it was looking at something": everything before `startS` is pre-roll and everything
    after `endS` is post-roll, and the two are indistinguishable from the video alone.

    Clamped to the clip rather than dropped when it overruns, because a clip capped by
    `max_length_s` legitimately ends before its event does — reporting no window there
    would lose the marking on precisely the longest, most interesting events.
    """
    if event_start is None:
        return None
    origin = clip_start.timestamp()
    start_s = max(0.0, event_start.timestamp() - origin)
    if start_s > duration_s:
        return None
    end_s = duration_s if event_end is None else min(duration_s, max(start_s, event_end.timestamp() - origin))
    return {
        "startS": _round(start_s),
        "endS": _round(end_s),
        "ruleName": rule_name or "",
        "eventUuid": event_uuid or "",
    }


def write_overlay(document: Dict[str, Any], dest: Path) -> int:
    """Write a sidecar atomically. Returns bytes written.

    Atomic because the player fetches it by URL the moment the card appears, and a
    half-written JSON is a parse error the user sees as "the overlay is broken" rather
    than as the transient it is.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    partial = dest.with_suffix(dest.suffix + ".part")
    try:
        partial.write_bytes(payload)
        partial.replace(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return len(payload)
