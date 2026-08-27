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
