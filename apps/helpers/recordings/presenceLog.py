"""What each camera could see, kept long enough to answer for a clip window.

An event's row records *why* it fired — the rule, the thing acting, the thing acted on.
It says nothing about what else was in frame, and that is the question a catalogue has to
answer: "show me every clip a person appears in", where the person is incidental to the
rule that saved the clip.

The window that has to be answered for is **the clip's**, not the event's. A clip carries
pre-roll — seconds cut from the buffer *before* the rule fired — so a subject that walked
through the lead-in is plainly in the video while being nowhere in the event's own span.
The recorder therefore cannot accumulate per open event: at the moment those frames were
analysed the event did not exist yet.

So this keeps a per-source interval log instead, and both windows are slices of it: the
event span at the closing edge, the clip window when a recording is attached.

**Intervals, not per-frame samples.** A label that is visible for ten minutes at 10 fps is
one interval here and six thousand samples in `DetectionTrack`. Presence is a step
function — the only questions asked of it are "was this label up at any point inside
[a, b]" — so the transitions are the whole content, and storing frames would be storing
the same answer thousands of times.

`gap_s` is what separates two visits from one: sightings closer than that extend the
interval, further apart start a new one. It exists so a subject that flickers out for one
frame is not recorded as two visits, which matters because every interval is a row's worth
of overlap arithmetic later.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

#: Sightings this far apart or closer are one visit. Comfortably longer than any
#: analysis interval a source runs at, so an ordinary frame-to-frame gap never splits an
#: interval, and short enough that two genuine visits an hour apart stay two intervals.
DEFAULT_GAP_S = 15.0

#: How far back the log is kept. Must exceed the longest pre-roll plus the longest event
#: a rule can hold open, because a clip window is addressed relative to the *clip's*
#: start and an event capped at an hour still needs its lead-in answerable at its close.
DEFAULT_HORIZON_S = 7200.0

#: Hard bound per source, independent of the time bound. A detector that flickers between
#: two labels every frame on a noisy camera would otherwise mint intervals without limit.
DEFAULT_MAX_INTERVALS = 4000


@dataclass
class PresenceInterval:
    """One unbroken stretch during which a label was present on one source."""

    label_uuid: str
    start: float
    end: float
    #: The highest confidence reported for this label during the stretch. Carried so a
    #: catalogue entry can say how sure the detector was without a second lookup; a
    #: filter that wants "confidently present" has it in the row.
    peak_confidence: float = 0.0


@dataclass
class PresenceLog:
    """Per-source presence intervals, bounded by age and count.

    Times are wallclock epoch seconds — the same clock a clip window is addressed in, and
    the reason the caller supplies the offset rather than this measuring anything itself.
    """

    gap_s: float = DEFAULT_GAP_S
    horizon_s: float = DEFAULT_HORIZON_S
    max_intervals: int = DEFAULT_MAX_INTERVALS
    #: source_uuid → intervals, oldest first.
    _by_source: Dict[str, List[PresenceInterval]] = field(default_factory=dict)

    def note(self, source_uuid: str, at: float, labels: Mapping[str, float]) -> None:
        """Record one analysed frame's verdict for one source.

        `labels` maps label uuid → the confidence behind it on this frame. An empty
        mapping is a legitimate frame — nothing was visible — and records nothing, which
        is what lets the gap rule close the intervals that were open.
        """
        if not labels:
            return
        intervals = self._by_source.setdefault(source_uuid or "", [])
        # Only the newest interval per label can still be open: they are appended in time
        # order and `at` never goes backwards on a source's frame loop.
        open_by_label: Dict[str, PresenceInterval] = {}
        for interval in reversed(intervals):
            if interval.label_uuid in labels and interval.label_uuid not in open_by_label:
                open_by_label[interval.label_uuid] = interval
            if len(open_by_label) == len(labels):
                break
        for label_uuid, confidence in labels.items():
            if not label_uuid:
                continue
            conf = float(confidence or 0.0)
            current = open_by_label.get(label_uuid)
            if current is not None and at - current.end <= self.gap_s:
                current.end = max(current.end, at)
                current.peak_confidence = max(current.peak_confidence, conf)
                continue
            intervals.append(PresenceInterval(label_uuid=label_uuid, start=at, end=at, peak_confidence=conf))
        self._prune(source_uuid or "", at)

    def labels_between(self, source_uuid: str, start: float, end: float) -> Dict[str, float]:
        """Labels present at any point inside `[start, end]` → peak confidence there.

        Overlap, not containment: a label that was up before the window opened and stayed
        up through it is present in it, and requiring the interval to sit inside the
        window would report nothing for exactly the long visits the window was cut from.

        The confidence is the interval's peak, which may have been reached just outside
        the window. That is the honest reading of a per-visit peak, and the alternative —
        a per-frame maximum — is the frame storage this class exists not to keep.
        """
        if end < start:
            start, end = end, start
        found: Dict[str, float] = {}
        for interval in self._by_source.get(source_uuid or "", ()):
            if interval.end < start or interval.start > end:
                continue
            previous = found.get(interval.label_uuid)
            if previous is None or interval.peak_confidence > previous:
                found[interval.label_uuid] = interval.peak_confidence
        return found

    def forget(self, source_uuid: str) -> None:
        self._by_source.pop(source_uuid or "", None)

    def clear(self) -> None:
        self._by_source.clear()

    def interval_count(self, source_uuid: Optional[str] = None) -> int:
        if source_uuid is None:
            return sum(len(v) for v in self._by_source.values())
        return len(self._by_source.get(source_uuid or "", ()))

    def _prune(self, source_uuid: str, now: float) -> None:
        intervals = self._by_source.get(source_uuid)
        if not intervals:
            return
        cutoff = now - self.horizon_s
        # An interval still open at the cutoff is kept: its `end` is what ages, not its
        # start, so a subject present for longer than the horizon does not vanish from a
        # window it is still inside.
        if intervals[0].end < cutoff:
            intervals = [i for i in intervals if i.end >= cutoff]
        if len(intervals) > self.max_intervals:
            intervals = intervals[len(intervals) - self.max_intervals :]
        self._by_source[source_uuid] = intervals


def labels_from_overlay(document: Mapping[str, object]) -> Dict[str, float]:
    """Distinct track labels in an overlay document → peak confidence.

    Reads `frames[].tracks[]` and never `frames[].boxes[]`. Tracks are what the tracker
    confirmed — `Tracker.update` returns `active_tracks()`, which is confirmed only — while
    boxes are raw model output at the deliberately low floor ByteTrack needs. Cataloguing
    boxes would mark nearly every clip as containing nearly every label.

    Used for two things that must agree: the clip-window slice written beside a new clip,
    and the one-time backfill over sidecars that already exist.
    """
    found: Dict[str, float] = {}
    frames = document.get("frames") if isinstance(document, Mapping) else None
    for frame in frames or ():
        if not isinstance(frame, Mapping):
            continue
        for track in frame.get("tracks") or ():
            if not isinstance(track, Mapping):
                continue
            label_uuid = str(track.get("labelUuid") or "")
            if not label_uuid:
                continue
            conf = float(track.get("conf") or 0.0)
            if conf > found.get(label_uuid, -1.0):
                found[label_uuid] = conf
    return found


def merge_label_confidences(*sources: Iterable[Tuple[str, float]]) -> Dict[str, float]:
    """Union several label→confidence mappings, keeping the highest confidence each."""
    merged: Dict[str, float] = {}
    for source in sources:
        for label_uuid, confidence in source:
            if not label_uuid:
                continue
            conf = float(confidence or 0.0)
            if conf > merged.get(label_uuid, -1.0):
                merged[label_uuid] = conf
    return merged
