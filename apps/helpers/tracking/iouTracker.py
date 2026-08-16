"""SORT-lite: greedy IoU association, no Kalman filter, no extra dependencies.

This is the only tracker in v1, and it is enough for the target case. Cats in a room
move slowly relative to a 2–5 fps analysis rate, so frame-to-frame boxes overlap
heavily and greedy IoU matching is both sufficient and cheap. ByteTrack can be added
later as another registry entry without touching anything that consumes tracks.

What this tracker deliberately does *not* do: predict motion. A Kalman filter earns
its keep at high frame rates with occlusion; here it would mostly add tuning surface
and a dependency for a case that IoU already handles.
"""

from typing import Any, Dict, List, Optional, Tuple

from .Registry import register_tracker
from .tracker import BBox, Detection, LabelLifetimeSpec, Track, Tracker


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@register_tracker()
class IouTracker(Tracker):
    """Greedy IoU tracker with per-label lifetimes and anchor latching."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        min_hits: int = 3,
        max_age_s: float = 2.0,
        anchor_confidence: float = 0.6,
        min_confidence: float = 0.25,
        label_lifetimes: Optional[Dict[str, LabelLifetimeSpec]] = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._min_confidence = min_confidence
        self._iou_threshold = iou_threshold
        self._min_hits = min_hits
        self._max_age_s = max_age_s
        self._anchor_confidence = anchor_confidence
        self._label_lifetimes: Dict[str, LabelLifetimeSpec] = label_lifetimes or {}
        self._tracks: List[Track] = []
        self._next_id = 1

    @classmethod
    def display_name(cls) -> str:
        return "IoU / SORT-lite"

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "min_confidence": {
                "type": "float",
                "required": False,
                "default": 0.25,
                "help": (
                    "Detections below this confidence are ignored entirely. Inference is asked for a "
                    "lower floor than this so other trackers can use the weak boxes; this tracker "
                    "has no use for them."
                ),
            },
            "iou_threshold": {
                "type": "float",
                "required": False,
                "default": 0.3,
                "help": "Minimum box overlap to consider a detection the same object as an existing track.",
            },
            "min_hits": {
                "type": "int",
                "required": False,
                "default": 3,
                "help": (
                    "Detections required before a track is reported. Guards against a single "
                    "spurious detection immediately opening an event."
                ),
            },
            "max_age_s": {
                "type": "float",
                "required": False,
                "default": 2.0,
                "help": (
                    "Seconds a track survives without a detection, unless its label overrides it. "
                    "Seconds rather than frames so the behaviour is identical on a 2 fps and a "
                    "25 fps camera."
                ),
            },
            "anchor_confidence": {
                "type": "float",
                "required": False,
                "default": 0.6,
                "help": "Confidence at which an anchor label's box is latched and held.",
            },
        }

    # ------------------------------------------------------------------ internals

    def _lifetime_for(self, label_uuid: Optional[str]) -> LabelLifetimeSpec:
        """The TTL and anchor flag for a label, falling back to the tracker default."""
        if label_uuid and label_uuid in self._label_lifetimes:
            return self._label_lifetimes[label_uuid]
        return LabelLifetimeSpec(ttl_s=self._max_age_s, is_anchor=False)

    def _match(self, detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedily pair tracks with detections by descending IoU.

        Greedy rather than Hungarian: with a handful of objects per frame the optimal
        assignment and the greedy one almost always agree, and greedy has no dependency.
        Pairs are only formed between matching labels — a cat detection must never
        inherit the litter box's track id.
        """
        candidates: List[Tuple[float, int, int]] = []
        for t_idx, track in enumerate(self._tracks):
            for d_idx, det in enumerate(detections):
                if track.label_uuid != det.label_uuid:
                    continue
                score = iou(track.bbox, det.bbox)
                if score >= self._iou_threshold:
                    candidates.append((score, t_idx, d_idx))

        candidates.sort(key=lambda c: c[0], reverse=True)
        used_tracks: set = set()
        used_dets: set = set()
        pairs: List[Tuple[int, int]] = []
        for _, t_idx, d_idx in candidates:
            if t_idx in used_tracks or d_idx in used_dets:
                continue
            used_tracks.add(t_idx)
            used_dets.add(d_idx)
            pairs.append((t_idx, d_idx))

        unmatched_tracks = [i for i in range(len(self._tracks)) if i not in used_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        return pairs, unmatched_tracks, unmatched_dets

    # ------------------------------------------------------------------ API

    def update(self, detections: List[Detection], now: float) -> List[Track]:
        # The task asks inference for a deliberately low floor so that ByteTrack can use
        # the low-scoring boxes. This tracker has no use for them — it has no second
        # association pass — so it applies its own threshold here rather than inheriting
        # whatever floor the task happened to request.
        detections = [d for d in detections if d.confidence >= self._min_confidence]
        pairs, _, unmatched_dets = self._match(detections)

        for t_idx, d_idx in pairs:
            track = self._tracks[t_idx]
            det = detections[d_idx]
            spec = self._lifetime_for(det.label_uuid)
            # A latched anchor keeps the box it was confident about. Re-detections of a
            # stationary object jitter, and letting that jitter through would make a
            # containment rule flicker at the zone boundary.
            if not (track.is_anchor and det.confidence < self._anchor_confidence):
                track.bbox = det.bbox
            track.confidence = det.confidence
            track.label_name = det.label_name or track.label_name
            track.last_seen = now
            track.hits += 1
            if track.hits >= self._min_hits:
                track.confirmed = True
            if spec.is_anchor and det.confidence >= self._anchor_confidence:
                track.is_anchor = True

        for d_idx in unmatched_dets:
            det = detections[d_idx]
            spec = self._lifetime_for(det.label_uuid)
            self._tracks.append(
                Track(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    label_uuid=det.label_uuid,
                    label_name=det.label_name,
                    confidence=det.confidence,
                    first_seen=now,
                    last_seen=now,
                    hits=1,
                    confirmed=self._min_hits <= 1,
                    is_anchor=spec.is_anchor and det.confidence >= self._anchor_confidence,
                )
            )
            self._next_id += 1

        # Expire by wallclock age against the label's own TTL. This is what makes
        # "litter box missing 5 frames = still there, cat missing 5 frames = gone"
        # expressible without either camera's frame rate entering into it.
        self._tracks = [t for t in self._tracks if t.age_s(now) <= self._lifetime_for(t.label_uuid).ttl_s]

        return self.active_tracks()

    def active_tracks(self) -> List[Track]:
        return [t for t in self._tracks if t.confirmed]

    def all_tracks(self) -> List[Track]:
        """Including unconfirmed ones — useful for the status panel's diagnostics."""
        return list(self._tracks)

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1
