"""Find where tracking went wrong, from the record the watcher already left behind.

A clip's `.boxes.json` sidecar (see `detectionTrack`) holds every analysed frame's
detections and the tracks built from them. Watching a clip back, certain failures are
obvious to a person and invisible to any single-frame tool: an identity that flips and
flips back, a subject the detector loses for a second while the tracker coasts on, two
objects that overlap and merge into one. Those frames are the most valuable training
data available — by construction they are the cases the current model fails on, in a
context where the right answer is unambiguous.

This module finds them. It reads numbers, not pixels: no inference, no video decode,
no database. That is the whole reason the feature is cheap enough to run on every clip.

**No ground truth is needed as input.** An earlier design had the user confirm each
event's identities before analysis, which front-loads work onto every event including
the ones that are fine. Instead the detectors look for *self-inconsistency*: within one
track, the label carried for the large majority of its life is almost certainly right,
so a track labelled one thing 112 times and another 9 times is contradictory regardless
of which is true. Confirmation moves downstream, onto a proposal that is already
correct, and only for events that flagged.

**Fault attribution is the load-bearing part.** `TrackSample` keeps raw detections and
tracker output apart precisely so "the detector missed it" can be told from "the tracker
lost it", and the two have opposite consequences here: a frame the *detector* got right
and the *tracker* dropped is not training data, and harvesting it would pad the set with
cases the model already handles. Every blip therefore carries a `fault`, and only
detector-fault blips are eligible to become training frames.

Output is an *incident* — a span with a start and an end — not a frame. A four-second
dropout is one blip, not a hundred and twenty.
"""

import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from dataclasses import fields as dataclasses_fields
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from apps.helpers.bboxUtils import iou_xywh

logger = logging.getLogger(__name__)

# ─── blip kinds ──────────────────────────────────────────────────────────────
# Strings rather than an Enum because these travel to the database and out through
# the generated API client, where every field is already a plain string — the same
# reasoning `_watch_event_recorder` gives for spelling out RECORDING_KIND_CLIP.

#: The track's label left its modal value and came back. Detector fault, high value:
#: the model contradicted itself about one object inside one clip.
KIND_IDENTITY_FLIP = "identity_flip"

#: The label changed and never changed back. Usually the tracker reusing an id across
#: two different objects rather than the detector misreading one, so it is attributed
#: to tracking and is not harvested.
KIND_IDENTITY_SWITCH = "identity_switch"

#: The track survived (coasting on its TTL) but no raw detection backed it. The
#: detector went blind while the tracker carried on — exactly the frames worth training.
KIND_DETECTOR_DROPOUT = "detector_dropout"

#: Neither a track nor a detection, bracketed by both. The detector lost the object
#: outright.
KIND_FULL_DROPOUT = "full_dropout"

#: The track vanished while a detection was still sitting on it. The detector was fine;
#: the tracker dropped the thread. Reported for diagnosis, never harvested.
KIND_TRACKER_DROPOUT = "tracker_dropout"

#: Two tracks overlapped and one of them stopped existing while the overlap held. The
#: cat-approaches-the-litter-box case.
KIND_OCCLUSION_MERGE = "occlusion_merge"

#: A track that lived a sample or two and never returned — most likely a false positive.
#: Off by default; see `BlipThresholds.enabled_kinds`.
KIND_GHOST = "ghost"

#: Present and correctly labelled, but the detector was barely sure. A near miss at the
#: decision boundary. Off by default.
KIND_CONFIDENCE_DIP = "confidence_dip"

FAULT_DETECTOR = "detector"
FAULT_TRACKER = "tracker"

#: Kinds whose frames may become training data. The others are diagnostic only — see
#: the module docstring on why a tracker fault must never be harvested.
HARVESTABLE_KINDS: FrozenSet[str] = frozenset(
    {
        KIND_IDENTITY_FLIP,
        KIND_DETECTOR_DROPOUT,
        KIND_FULL_DROPOUT,
        KIND_OCCLUSION_MERGE,
        KIND_GHOST,
        KIND_CONFIDENCE_DIP,
    }
)

#: On under the conservative default. `ghost` and `confidence_dip` are the two noisiest
#: and are left off pending threshold tuning against real events — not for want of a
#: mechanism; both have a working path all the way through to training.
DEFAULT_ENABLED_KINDS: FrozenSet[str] = frozenset(
    {
        KIND_IDENTITY_FLIP,
        KIND_IDENTITY_SWITCH,
        KIND_DETECTOR_DROPOUT,
        KIND_FULL_DROPOUT,
        KIND_TRACKER_DROPOUT,
        KIND_OCCLUSION_MERGE,
    }
)


@dataclass(frozen=True)
class BlipThresholds:
    """What counts as a blip. Config rather than constants, deliberately.

    Every number here is a guess until there is a night of real events to look at, and
    the shipping values are meant to be tuned from measured data rather than argued
    about. The defaults are the *conservative* envelope: what they surface should be
    real, at the known cost of missing single-sample glitches.
    """

    #: A departure from the modal label must last this many samples to count. At 2, a
    #: single-sample blink is ignored — those are common and usually uninteresting.
    min_flip_samples: int = 2

    #: A gap must last this many samples to count. Below it, the tracker's own TTL
    #: smoothing makes the distinction meaningless.
    min_dropout_samples: int = 3

    #: How much two boxes must overlap before they are treated as the same object —
    #: used both to match a track to the raw detection backing it, and to decide that
    #: a dropped track was really absorbed by its neighbour.
    match_iou: float = 0.5

    #: Overlap above which two *different* tracks are considered mutually occluding.
    occlusion_iou: float = 0.5

    #: A track this short that never returns is treated as a false positive.
    ghost_max_samples: int = 2

    #: Correct-but-unconvinced. Only consulted when `confidence_dip` is enabled.
    dip_below: float = 0.35
    min_dip_samples: int = 3

    #: A track must be seen this many times before its modal label means anything. Two
    #: samples split one-one have no majority, and calling either one "the truth" would
    #: manufacture a flip out of nothing.
    min_track_samples: int = 6

    #: Sample spacing past which the temporal detectors are refused outright. A
    #: timelapse plays frames minutes apart, where "the detection disappeared between
    #: these two frames" carries no information — the subject simply left. Guards
    #: against a timelapse overlay being fed in and producing confident nonsense.
    max_sample_gap_s: float = 2.0

    enabled_kinds: FrozenSet[str] = DEFAULT_ENABLED_KINDS

    def enabled(self, kind: str) -> bool:
        return kind in self.enabled_kinds

    @classmethod
    def from_overrides(cls, overrides: Optional[Dict[str, Any]]) -> "BlipThresholds":
        """Build from a watcher's stored tuning bundle, defaulting anything absent.

        Tolerant on purpose. This reads a JSON column a user can edit, and the failure
        mode to avoid is a watcher that refuses to analyse — or worse, refuses to
        *record* — because one key was misspelt or typed as a string. Unknown keys are
        ignored and unusable values fall back to the default for that field, so the
        analysis degrades toward conservative rather than toward broken.

        `enabled_kinds` is accepted as a list, since JSON has no set.
        """
        if not overrides:
            return cls()

        fields = {f.name for f in dataclasses_fields(cls)}
        kwargs: Dict[str, Any] = {}
        for key, value in overrides.items():
            if key not in fields or value is None:
                continue
            default = getattr(cls(), key)
            try:
                if key == "enabled_kinds":
                    kinds = frozenset(str(k) for k in value)
                    # An empty set would silently disable every detector, which is what
                    # `blip_analysis_enabled` is for and is almost certainly not what an
                    # empty list in a tuning bundle meant.
                    kwargs[key] = kinds if kinds else default
                elif isinstance(default, bool):
                    kwargs[key] = bool(value)
                elif isinstance(default, int):
                    kwargs[key] = int(value)
                elif isinstance(default, float):
                    kwargs[key] = float(value)
            except (TypeError, ValueError):
                logger.warning("Ignoring unusable blip threshold %s=%r", key, value)
        return cls(**kwargs)


@dataclass(frozen=True)
class Blip:
    """One incident: a span of clip-relative time where something went wrong."""

    kind: str
    track_id: int
    start_s: float
    end_s: float
    #: How confident the detector was about the thing it then got wrong, 0..1. One
    #: meaning across every kind: for a flip it is the mean confidence of the wrong
    #: label, for a dropout the mean confidence of the samples bracketing the gap.
    #: A detector that was *sure* and wrong is the most valuable case, so this sorts
    #: the way attention should flow.
    severity: float
    fault: str
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    @property
    def harvestable(self) -> bool:
        """Whether this blip's frames may become training data."""
        return self.fault == FAULT_DETECTOR and self.kind in HARVESTABLE_KINDS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "trackId": self.track_id,
            "startS": round(self.start_s, 4),
            "endS": round(self.end_s, 4),
            "severity": round(self.severity, 4),
            "fault": self.fault,
            "harvestable": self.harvestable,
            "detail": dict(self.detail),
        }


# ─── geometry helpers ────────────────────────────────────────────────────────


def _as_wh(box: Dict[str, Any]) -> Dict[str, float]:
    """Overlay boxes are `x/y/w/h`; `iou_xywh` reads `x/y/width/height`.

    An adapter rather than a second IoU implementation: a divergent copy of this
    arithmetic is exactly the kind of silent drift `detectionMerger` warns about.
    """
    return {
        "x": float(box.get("x", 0.0)),
        "y": float(box.get("y", 0.0)),
        "width": float(box.get("w", 0.0)),
        "height": float(box.get("h", 0.0)),
    }


def _overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return iou_xywh(_as_wh(a), _as_wh(b))


def _best_box_for(track: Dict[str, Any], boxes: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The raw detection backing this track on one frame, if any.

    Matched on overlap alone, not on label. Matching by label too would make an
    identity flip look like a detection dropout — the box is right there, it just says
    the wrong thing — and those are different faults with different fixes.
    """
    best: Optional[Dict[str, Any]] = None
    best_iou = 0.0
    for box in boxes:
        score = _overlap(track, box)
        if score > best_iou:
            best, best_iou = box, score
    return best if best is not None and best_iou > 0.0 else None


# ─── frame / track indexing ──────────────────────────────────────────────────


@dataclass
class _TrackTimeline:
    """One track's presence across the clip, indexed by frame position."""

    track_id: int
    #: frame index -> the track dict on that frame
    present: Dict[int, Dict[str, Any]]
    modal_label: str
    #: Whether one label actually holds a majority. A track split evenly between two
    #: labels has no truth to infer, and picking the first-seen one would invent a flip
    #: out of a genuine ambiguity — so identity analysis is skipped for those.
    modal_is_clear: bool = True

    @property
    def first(self) -> int:
        return min(self.present)

    @property
    def last(self) -> int:
        return max(self.present)

    def label_at(self, index: int) -> str:
        return str(self.present[index].get("labelUuid") or self.present[index].get("label") or "")

    def conf_at(self, index: int) -> float:
        return float(self.present[index].get("conf", 0.0) or 0.0)


def _frames_of(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Frames in time order. Sorted defensively — the builder appends in order, but a
    hand-edited or merged document should not silently produce reversed spans."""
    frames = list(document.get("frames") or [])
    return sorted(frames, key=lambda f: float(f.get("t", 0.0)))


def _median_gap(frames: Sequence[Dict[str, Any]]) -> float:
    gaps = [float(b.get("t", 0.0)) - float(a.get("t", 0.0)) for a, b in zip(frames, frames[1:])]
    positive = [g for g in gaps if g > 0]
    return statistics.median(positive) if positive else 0.0


def _build_timelines(frames: Sequence[Dict[str, Any]], min_samples: int) -> List[_TrackTimeline]:
    """Group every frame's tracks by id, and settle each one's modal label.

    Tracks seen fewer than `min_samples` times are dropped here rather than analysed:
    with too few samples the modal label is not a majority at all, and treating a
    coin-flip as ground truth manufactures flips that were never there. `ghost` catches
    the short ones it cares about separately, before this filter.
    """
    by_id: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for index, frame in enumerate(frames):
        for track in frame.get("tracks") or []:
            track_id = int(track.get("id", 0) or 0)
            if not track_id:
                continue
            by_id.setdefault(track_id, {})[index] = track

    timelines: List[_TrackTimeline] = []
    for track_id, present in by_id.items():
        if len(present) < min_samples:
            continue
        labels = [str(t.get("labelUuid") or t.get("label") or "") for t in present.values()]
        ranked = Counter(labels).most_common()
        modal = ranked[0][0] if ranked else ""
        clear = len(ranked) < 2 or ranked[0][1] > ranked[1][1]
        timelines.append(_TrackTimeline(track_id=track_id, present=present, modal_label=modal, modal_is_clear=clear))
    return sorted(timelines, key=lambda t: t.track_id)


def _runs(indices: Sequence[int]) -> List[Tuple[int, int]]:
    """Collapse a sorted index list into inclusive (start, end) runs of consecutive values."""
    runs: List[Tuple[int, int]] = []
    for index in sorted(indices):
        if runs and index == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return runs


def _t(frames: Sequence[Dict[str, Any]], index: int) -> float:
    return float(frames[index].get("t", 0.0))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ─── detectors ───────────────────────────────────────────────────────────────


def _identity_blips(
    frames: Sequence[Dict[str, Any]],
    line: _TrackTimeline,
    thresholds: BlipThresholds,
) -> List[Blip]:
    """Departures from the track's modal label.

    A departure bounded by the modal label on both sides is a *flip* — the detector
    contradicted itself and corrected itself, which is a detector fault and prime
    training material. A departure that runs to the end of the track is a *switch*:
    far more likely the tracker handing one id to a second object, so it is attributed
    to tracking and never harvested.
    """
    if not line.modal_is_clear:
        return []

    wrong = [i for i in sorted(line.present) if line.label_at(i) != line.modal_label]
    if not wrong:
        return []

    blips: List[Blip] = []
    for start, end in _runs(wrong):
        length = end - start + 1
        if length < thresholds.min_flip_samples:
            continue

        returns = end < line.last
        kind = KIND_IDENTITY_FLIP if returns else KIND_IDENTITY_SWITCH
        if not thresholds.enabled(kind):
            continue

        confidences = [line.conf_at(i) for i in range(start, end + 1) if i in line.present]
        blips.append(
            Blip(
                kind=kind,
                track_id=line.track_id,
                start_s=_t(frames, start),
                end_s=_t(frames, end),
                severity=_mean(confidences),
                fault=FAULT_DETECTOR if returns else FAULT_TRACKER,
                detail={
                    "modalLabel": line.modal_label,
                    "sawLabel": line.label_at(start),
                    "samples": length,
                    "trackSamples": len(line.present),
                },
            )
        )
    return blips


def _dropout_blips(
    frames: Sequence[Dict[str, Any]],
    line: _TrackTimeline,
    thresholds: BlipThresholds,
) -> List[Blip]:
    """Gaps in the record, split three ways by who actually failed.

    * track present, no detection backing it → the detector went blind and the tracker
      coasted. `detector_dropout`.
    * track absent, but a detection still sitting where it was → the tracker dropped a
      thread the detector was still holding. `tracker_dropout`, diagnostic only.
    * neither → `full_dropout`.

    The three are indistinguishable from the video and have entirely different fixes,
    which is why the sidecar keeps detections and tracks apart in the first place.
    """
    blips: List[Blip] = []

    # ── track present but unbacked: the detector went quiet under a coasting track.
    unbacked = [i for i in sorted(line.present) if _best_box_for(line.present[i], frames[i].get("boxes") or []) is None]
    for start, end in _runs(unbacked):
        if end - start + 1 < thresholds.min_dropout_samples:
            continue
        if not thresholds.enabled(KIND_DETECTOR_DROPOUT):
            continue
        blips.append(
            Blip(
                kind=KIND_DETECTOR_DROPOUT,
                track_id=line.track_id,
                start_s=_t(frames, start),
                end_s=_t(frames, end),
                severity=_bracketing_confidence(line, start, end),
                fault=FAULT_DETECTOR,
                detail={"samples": end - start + 1, "modalLabel": line.modal_label},
            )
        )

    # ── track absent entirely, between its own first and last sighting.
    absent = [i for i in range(line.first, line.last + 1) if i not in line.present]
    for start, end in _runs(absent):
        if end - start + 1 < thresholds.min_dropout_samples:
            continue

        # Was the detector still seeing it? Compare against the last box the track held
        # before it vanished — if something is still sitting there, the tracker is at
        # fault and these frames are worthless as training data.
        anchor = line.present[max(i for i in line.present if i < start)]
        still_detected = any(
            _best_and_over(anchor, frames[i].get("boxes") or [], thresholds.match_iou) for i in range(start, end + 1)
        )
        kind = KIND_TRACKER_DROPOUT if still_detected else KIND_FULL_DROPOUT
        if not thresholds.enabled(kind):
            continue

        blips.append(
            Blip(
                kind=kind,
                track_id=line.track_id,
                start_s=_t(frames, start),
                end_s=_t(frames, end),
                severity=_bracketing_confidence(line, start, end),
                fault=FAULT_TRACKER if still_detected else FAULT_DETECTOR,
                detail={"samples": end - start + 1, "modalLabel": line.modal_label},
            )
        )

    return blips


def _best_and_over(anchor: Dict[str, Any], boxes: Sequence[Dict[str, Any]], threshold: float) -> bool:
    match = _best_box_for(anchor, boxes)
    return match is not None and _overlap(anchor, match) >= threshold


def _bracketing_confidence(line: _TrackTimeline, start: int, end: int) -> float:
    """How sure the detector was either side of a gap.

    A subject held at 0.9 and then lost completely is a worse miss than one held at
    0.3, so this is the honest severity for a span that contains no detection of its
    own to measure.
    """
    before = [i for i in line.present if i < start]
    after = [i for i in line.present if i > end]
    samples: List[float] = []
    if before:
        samples.append(line.conf_at(max(before)))
    if after:
        samples.append(line.conf_at(min(after)))
    return _mean(samples)


def _occlusion_blips(
    frames: Sequence[Dict[str, Any]],
    lines: Sequence[_TrackTimeline],
    thresholds: BlipThresholds,
) -> List[Blip]:
    """One track absorbed by another while the two overlapped.

    The litter-box case: a subject approaches an object, their boxes overlap heavily,
    and one of the two stops being reported for the duration. Distinct from a plain
    dropout because the fix is different — this wants training data of the two objects
    overlapping, not of the subject alone.
    """
    if not thresholds.enabled(KIND_OCCLUSION_MERGE):
        return []

    blips: List[Blip] = []
    for i, line in enumerate(lines):
        for other in lines[i + 1 :]:
            shared = sorted(set(line.present) & set(other.present))
            overlapping = [
                index
                for index in shared
                if _overlap(line.present[index], other.present[index]) >= thresholds.occlusion_iou
            ]
            if not overlapping:
                continue

            # Whichever of the two stops first, while the other carries on.
            for vanished, survivor in ((line, other), (other, line)):
                gap_start = vanished.last + 1
                if gap_start > survivor.last:
                    continue
                if vanished.last not in overlapping:
                    continue
                carried_on = [i2 for i2 in survivor.present if i2 >= gap_start]
                if len(carried_on) < thresholds.min_dropout_samples:
                    continue

                end_index = max(carried_on)
                blips.append(
                    Blip(
                        kind=KIND_OCCLUSION_MERGE,
                        track_id=vanished.track_id,
                        start_s=_t(frames, vanished.last),
                        end_s=_t(frames, end_index),
                        severity=vanished.conf_at(vanished.last),
                        fault=FAULT_DETECTOR,
                        detail={
                            "modalLabel": vanished.modal_label,
                            "absorbedByTrackId": survivor.track_id,
                            "absorbedByLabel": survivor.modal_label,
                            "overlapSamples": len(overlapping),
                        },
                    )
                )
    return blips


def _ghost_blips(
    frames: Sequence[Dict[str, Any]],
    all_tracks: Dict[int, Dict[int, Dict[str, Any]]],
    thresholds: BlipThresholds,
) -> List[Blip]:
    """Tracks that flickered into existence and never came back.

    Reads the *unfiltered* track index, because `_build_timelines` drops short tracks —
    and a ghost is short by definition.
    """
    if not thresholds.enabled(KIND_GHOST):
        return []

    blips: List[Blip] = []
    for track_id, present in sorted(all_tracks.items()):
        if len(present) > thresholds.ghost_max_samples:
            continue
        indices = sorted(present)
        confidences = [float(present[i].get("conf", 0.0) or 0.0) for i in indices]
        blips.append(
            Blip(
                kind=KIND_GHOST,
                track_id=track_id,
                start_s=_t(frames, indices[0]),
                end_s=_t(frames, indices[-1]),
                severity=_mean(confidences),
                fault=FAULT_DETECTOR,
                detail={
                    "samples": len(indices),
                    "sawLabel": str(present[indices[0]].get("labelUuid") or ""),
                },
            )
        )
    return blips


def _confidence_dip_blips(
    frames: Sequence[Dict[str, Any]],
    line: _TrackTimeline,
    thresholds: BlipThresholds,
) -> List[Blip]:
    """Correctly labelled, but the detector was barely sure.

    Off by default, and the reason is calibration rather than noise: plenty of frames
    are *legitimately* low-confidence — motion blur, subject half out of shot — and
    stamping those with a confident ground-truth box teaches the model to be certain
    about genuinely ambiguous input.
    """
    if not thresholds.enabled(KIND_CONFIDENCE_DIP):
        return []

    dipped = [
        i
        for i in sorted(line.present)
        if line.label_at(i) == line.modal_label and line.conf_at(i) < thresholds.dip_below
    ]
    blips: List[Blip] = []
    for start, end in _runs(dipped):
        if end - start + 1 < thresholds.min_dip_samples:
            continue
        blips.append(
            Blip(
                kind=KIND_CONFIDENCE_DIP,
                track_id=line.track_id,
                start_s=_t(frames, start),
                end_s=_t(frames, end),
                severity=_mean([line.conf_at(i) for i in range(start, end + 1) if i in line.present]),
                fault=FAULT_DETECTOR,
                detail={"samples": end - start + 1, "modalLabel": line.modal_label},
            )
        )
    return blips


# ─── entry point ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProposedBox:
    """A corrected box to offer for one harvested frame, in normalised coordinates."""

    x: float
    y: float
    w: float
    h: float
    label_uuid: str
    #: True when no detection existed at this moment and the geometry was interpolated
    #: between the samples either side. The reviewer has to be told: a box twenty pixels
    #: out is worse for training than no box at all, so an estimated one needs looking at
    #: on the canvas rather than accepting from a thumbnail.
    estimated: bool

    def as_geometry(self) -> Dict[str, float]:
        return {"x": _round(self.x), "y": _round(self.y), "width": _round(self.w), "height": _round(self.h)}


def _round(value: float) -> float:
    return round(float(value), 4)


def harvest_times(start_s: float, end_s: float, count: int = 3) -> List[float]:
    """Which moments of an incident to pull frames from.

    Onset, middle and recovery by default — the boundaries are where the detector's
    decision actually flips, so they carry more than the middle does. Not every frame:
    a four-second dropout at 30 fps is a hundred and twenty near-identical images that
    add annotation work and no information.

    Collapses to fewer moments for a short incident rather than returning the same
    instant several times.
    """
    if count <= 1 or end_s <= start_s:
        return [start_s]
    span = end_s - start_s
    times = [start_s + span * i / (count - 1) for i in range(count)]
    # Distinct to 3 dp: two "different" moments inside one frame interval would upload
    # the same picture twice and rely on dedup to notice.
    seen: List[float] = []
    for t in times:
        if not seen or abs(t - seen[-1]) > 0.001:
            seen.append(t)
    return seen


def proposed_box_at(
    document: Dict[str, Any],
    track_id: int,
    label_uuid: str,
    at_s: float,
    *,
    max_estimate_gap_s: float = 3.0,
) -> Optional[ProposedBox]:
    """Where the corrected box belongs at one moment of the clip.

    One function for every blip kind, because the distinction that matters is not the
    kind but whether the track was *seen* at that moment:

    * seen — the geometry is the detector's own box and only the label is wrong. That is
      the identity-flip case, and the proposal is exact.
    * not seen — the detector had lost it, so the box is interpolated between the
      samples either side and flagged `estimated`. That is the dropout case.

    Returns None when there is nothing honest to propose: no track either side, or a gap
    so long that interpolating across it would be invention rather than estimation. A
    missing proposal costs one frame; a confidently wrong box costs training data.
    """
    frames = _frames_of(document)
    before: Optional[Tuple[float, Dict[str, Any]]] = None
    after: Optional[Tuple[float, Dict[str, Any]]] = None

    for frame in frames:
        t = float(frame.get("t", 0.0))
        match = next((tr for tr in (frame.get("tracks") or []) if int(tr.get("id", 0) or 0) == track_id), None)
        if match is None:
            continue
        if t <= at_s:
            before = (t, match)
        elif after is None:
            after = (t, match)
            break

    if before is not None and abs(before[0] - at_s) < 1e-6:
        box = before[1]
        return ProposedBox(
            x=float(box.get("x", 0.0)),
            y=float(box.get("y", 0.0)),
            w=float(box.get("w", 0.0)),
            h=float(box.get("h", 0.0)),
            label_uuid=label_uuid,
            estimated=False,
        )

    if before is None or after is None:
        # Nothing to interpolate between. Extrapolating from one side would put a box
        # where the subject might not be at all.
        return None
    if after[0] - before[0] > max_estimate_gap_s:
        return None

    span = after[0] - before[0]
    ratio = 0.0 if span <= 0 else (at_s - before[0]) / span
    first, second = before[1], after[1]

    def _lerp(key: str) -> float:
        start = float(first.get(key, 0.0))
        end = float(second.get(key, 0.0))
        return start + (end - start) * ratio

    return ProposedBox(
        x=_lerp("x"),
        y=_lerp("y"),
        w=_lerp("w"),
        h=_lerp("h"),
        label_uuid=label_uuid,
        estimated=True,
    )


def analyse_overlay(
    document: Dict[str, Any],
    thresholds: Optional[BlipThresholds] = None,
) -> List[Blip]:
    """Every blip in one clip's overlay document, ordered by when it starts.

    Returns an empty list — rather than raising — for a document that cannot support
    the analysis: too few frames, or samples spaced so far apart that temporal
    reasoning is meaningless. A watcher that stops recording events because one clip
    was odd is a worse failure than one that reports nothing for it.
    """
    thresholds = thresholds or BlipThresholds()
    frames = _frames_of(document)
    if len(frames) < 2:
        return []

    # A timelapse plays frames minutes apart. "The detection vanished between these two"
    # then means the subject left, not that anything failed, and every temporal detector
    # below would report confident nonsense.
    if _median_gap(frames) > thresholds.max_sample_gap_s:
        return []

    raw_tracks: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for index, frame in enumerate(frames):
        for track in frame.get("tracks") or []:
            track_id = int(track.get("id", 0) or 0)
            if track_id:
                raw_tracks.setdefault(track_id, {})[index] = track

    lines = _build_timelines(frames, thresholds.min_track_samples)

    blips: List[Blip] = []
    for line in lines:
        blips.extend(_identity_blips(frames, line, thresholds))
        blips.extend(_dropout_blips(frames, line, thresholds))
        blips.extend(_confidence_dip_blips(frames, line, thresholds))
    blips.extend(_occlusion_blips(frames, lines, thresholds))
    blips.extend(_ghost_blips(frames, raw_tracks, thresholds))

    return sorted(blips, key=lambda b: (b.start_s, b.track_id, b.kind))
