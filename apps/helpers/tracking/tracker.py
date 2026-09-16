"""Tracker interface and the Track value type.

Tracking exists to turn per-frame detections into *objects with identity over time*,
which is what interaction rules need: "a cat entered the litter box" is a statement
about one cat across many frames, not about a detection in one frame.

Two design points that the rest of the NVR depends on:

* **Ages and lifetimes are in seconds, never frames.** A frame count silently means
  1 s on a 5 fps camera and 0.2 s on a 25 fps one, and drifts whenever a source
  stutters or backs off — which live streams do constantly. Every method here takes an
  explicit timestamp rather than assuming a fixed cadence.
* **Anchors latch.** A stationary object like a litter box is intermittently missed by
  a detector; re-detecting it every frame is both expensive and unreliable. An anchor
  track keeps its last confident box for a long TTL instead.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# A detection as it arrives from inference, before it has any identity.
BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass
class Detection:
    """One model output for one frame."""

    bbox: BBox
    label_uuid: Optional[str] = None
    label_name: Optional[str] = None
    confidence: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Track:
    """One object followed across frames."""

    track_id: int
    bbox: BBox
    label_uuid: Optional[str] = None
    label_name: Optional[str] = None
    confidence: float = 0.0
    #: Wallclock seconds when this track was first confirmed.
    first_seen: float = 0.0
    #: Wallclock seconds of the most recent *detection* (not the most recent frame).
    last_seen: float = 0.0
    #: Detections matched so far. Used for the min-hits confirmation gate.
    hits: int = 0
    #: True once the track has enough hits to be reported. Unconfirmed tracks exist so
    #: a single spurious detection doesn't immediately open an event.
    confirmed: bool = False
    #: True when this track's label is configured as an anchor and its box is latched.
    is_anchor: bool = False
    #: Whether THIS `update()` call matched a real detection to this track, as
    #: opposed to the track merely surviving its TTL while coasting. Not
    #: recoverable from `last_seen` alone (that's the wallclock of the last
    #: match, not a fact about the current call) — a caller that needs "did the
    #: detector miss this track just now" (e.g. detector-miss collection,
    #: COLLECTOR_UNIFICATION_PLAN.md §4A) needs this flag, not a `last_seen`
    #: comparison against a float `now` it may not have to hand.
    matched_this_update: bool = True

    def age_s(self, now: float) -> float:
        """Seconds since this track last matched a detection."""
        return max(0.0, now - self.last_seen)

    def duration_s(self, now: float) -> float:
        return max(0.0, now - self.first_seen)

    def to_dict(self) -> dict:
        return {
            "trackId": self.track_id,
            "bbox": list(self.bbox),
            "labelUuid": self.label_uuid,
            "labelName": self.label_name,
            "confidence": round(self.confidence, 4),
            "firstSeen": self.first_seen,
            "lastSeen": self.last_seen,
            "hits": self.hits,
            "isAnchor": self.is_anchor,
        }


@dataclass(frozen=True)
class LabelLifetimeSpec:
    """Per-label tracking lifetime, resolved from the profile's `label_lifetimes`."""

    ttl_s: float
    is_anchor: bool = False


class Tracker(ABC):
    """Base class for trackers.

    Registered subclasses get their `params_schema()` rendered into the Watchers UI
    automatically, the same way image providers do — so adding ByteTrack later is a
    registry entry plus a class, with no UI work.
    """

    def __init__(self, **params: Any) -> None:
        self._params = params

    @classmethod
    def display_name(cls) -> str:
        return cls.__name__

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Schema describing this tracker's parameters, in the same shape
        `ImageProvider.params_schema()` uses so the existing form machinery renders it."""
        return {}

    @abstractmethod
    def update(self, detections: List[Detection], now: float) -> List[Track]:
        """Advance the tracker by one frame and return the currently live tracks.

        `now` is wallclock seconds. It is passed rather than measured so that replay and
        tests are deterministic, and so a stuttering source ages tracks by real elapsed
        time rather than by frame count.
        """

    @abstractmethod
    def reset(self) -> None:
        """Forget all tracks. Track ids do not survive a restart and must not pretend to."""

    def active_tracks(self) -> List[Track]:
        """Currently live, confirmed tracks. Default implementation returns nothing."""
        return []


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


#: Same rationale as `DEFAULT_EXCLUSIVE_VARIANTS_IOU_THRESH` in yoloUtils.py: high
#: enough that two genuinely distinct animals standing close together essentially
#: never clear it, but well below 1.0 because near-duplicate tracks drift apart
#: slightly frame to frame rather than sitting pixel-identical.
DUPLICATE_TRACK_IOU_THRESH = 0.85


def merge_duplicate_tracks(tracks: List[Track], iou_threshold: float = DUPLICATE_TRACK_IOU_THRESH) -> List[Track]:
    """Collapse tracks that are almost certainly the same physical object.

    A tracker occasionally spawns two persistent tracks for one animal — a brief
    detector dropout breaks association and a new id starts beside the old one
    instead of continuing it. The two then coast independently, each drawn and
    classified every frame (sometimes under different guessed identities), until —
    if ever — a delayed, offline occlusion-merge heuristic notices, sometimes many
    seconds later and never retroactively. This runs every frame instead, right
    after `Tracker.update()`, so the duplicate never reaches the rule engine, the
    overlay, or a saved training frame in the first place.

    Safe to drop a track id outright rather than reconciling it into the survivor:
    rule/event state is keyed by label+zone, not by track id, and track ids are
    already documented as not surviving even a watcher restart (see
    `Tracker.reset()` and STREAM_WATCHER_PLAN.md) — nothing downstream depends on
    which of two duplicate ids kept living.

    Greedy NMS by confidence, independent of label: two boxes this close together
    are the same object regardless of whether they currently agree on *who* it is
    — that disagreement is exactly the case a same-label-only check would miss.
    """
    if len(tracks) < 2:
        return tracks
    ordered = sorted(tracks, key=lambda t: t.confidence, reverse=True)
    kept: List[Track] = []
    for t in ordered:
        if any(_iou(t.bbox, k.bbox) >= iou_threshold for k in kept):
            continue
        kept.append(t)
    kept_ids = {t.track_id for t in kept}
    return [t for t in tracks if t.track_id in kept_ids]
