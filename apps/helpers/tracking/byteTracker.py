"""ByteTrack: two-stage association that recovers low-confidence detections.

The idea BYTE contributes (Zhang et al., 2021) is not a better motion model — it is
refusing to throw away weak boxes. A conventional tracker keeps detections above one
threshold and discards the rest; but an object that becomes briefly occluded, blurred
or oddly lit does not vanish, it merely *scores lower*. Discarding those boxes is what
produces the classic failure: a track dies mid-occlusion and the object reappears with
a new id.

So association happens twice per frame:

1. **High boxes** (``>= track_thresh``) are matched against all live tracks. Anything
   left over starts a new track.
2. **Low boxes** (``>= low_thresh``, below ``track_thresh``) are matched against only
   the tracks that stage 1 failed to match. These *never* start a track — a weak box
   with no history is far more likely to be noise than a new object. They exist purely
   to keep an existing track alive through a bad patch.

Three deliberate departures from the reference implementation, each because this
codebase already made the opposite call for good reasons:

* **Class-aware.** Canonical ByteTrack is class-agnostic: it pools every box and
  associates on geometry alone. Here every rule matches on ``label_uuid``, so pooling
  would let a cat track inherit the litter box's id and quietly corrupt every
  interaction rule. Pairs are only ever formed within one ``exclusive_variants`` group
  — see ``Tracker.can_associate``, which is deliberately looser than label equality so
  that a detector alternating between two names for one object does not fork the track.
* **Lifetimes in seconds, not frames.** The reference expires tracks after
  ``track_buffer`` *frames*, which silently means 1 s at 30 fps and 15 s at 2 fps —
  and drifts whenever a live source stutters. This tracker expires against each
  label's own TTL in wallclock seconds, so "a litter box survives 30 s of misses, a
  cat survives 1 s" is expressible and means the same thing on every camera.
* **No Kalman filter.** SORT's motion model earns its keep at high frame rates with
  fast subjects. At the 2–5 fps this analyses, a constant-velocity prediction over a
  200–500 ms gap is mostly noise, and it would add a tuning surface and a dependency
  for little gain. The BYTE association strategy is orthogonal to it and is the part
  that matters here. If fast subjects ever become the problem, prediction is the
  extension point — not a reason to avoid BYTE.

Association is Hungarian (``scipy.optimize.linear_sum_assignment``) rather than the
greedy pass `IouTracker` uses. Greedy is fine when objects are few and well separated;
BYTE's second stage deliberately matches weak, overlapping boxes where greedy's
first-come-first-served ordering starts to cost real assignments. scipy is already a
dependency, so this needs no new package.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .iouTracker import iou
from .Registry import register_tracker
from .tracker import Detection, LabelLifetimeSpec, Track, Tracker


@register_tracker()
class ByteTracker(Tracker):
    """Two-stage (BYTE) association with per-label lifetimes and anchor latching."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.3,
        low_match_thresh: float = 0.2,
        min_hits: int = 3,
        max_age_s: float = 2.0,
        anchor_confidence: float = 0.6,
        label_lifetimes: Optional[Dict[str, LabelLifetimeSpec]] = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._track_thresh = track_thresh
        self._low_thresh = low_thresh
        self._match_thresh = match_thresh
        self._low_match_thresh = low_match_thresh
        self._min_hits = min_hits
        self._max_age_s = max_age_s
        self._anchor_confidence = anchor_confidence
        self._label_lifetimes: Dict[str, LabelLifetimeSpec] = label_lifetimes or {}
        self._tracks: List[Track] = []
        self._next_id = 1

    @property
    def track_thresh(self) -> float:
        """The confidence at or above which a box may *start* a track.

        Public because a consumer that cannot use tracks at all — a snapshot source,
        where frames arrive minutes apart and no track can ever confirm — still needs a
        confidence floor, and it should be this one rather than a second constant kept
        in sync by hand. See `WatchPipeline.presence_labels`.
        """
        return self._track_thresh

    @classmethod
    def display_name(cls) -> str:
        return "ByteTrack (two-stage)"

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "track_thresh": {
                "type": "float",
                "required": False,
                "default": 0.5,
                "help": (
                    "Detections at or above this confidence can start a new track. Below it, a "
                    "detection can only continue a track that already exists."
                ),
            },
            "low_thresh": {
                "type": "float",
                "required": False,
                "default": 0.1,
                "help": (
                    "Weakest detection considered at all. The whole point of this tracker is that "
                    "boxes between this and the track threshold keep an occluded object alive "
                    "rather than being discarded — raising this to the track threshold turns it "
                    "back into an ordinary single-stage tracker."
                ),
            },
            "match_thresh": {
                "type": "float",
                "required": False,
                "default": 0.3,
                "help": "Minimum box overlap to associate a confident detection with an existing track.",
            },
            "low_match_thresh": {
                "type": "float",
                "required": False,
                "default": 0.2,
                "help": (
                    "Overlap required in the second pass. Lower than the first, because a box that "
                    "scored badly is usually also positioned badly."
                ),
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
            "label_vote_half_life_s": {
                "type": "float",
                "required": False,
                "default": 6.0,
                "help": (
                    "How long a track remembers what the detector called it, in seconds. A track's "
                    "reported identity is a decaying, confidence-weighted vote over the labels "
                    "declared as alternate guesses for one object, not whatever the latest frame "
                    "said. Raise it if an identity still flickers; lower it if a track is slow to "
                    "accept that it is now looking at someone else."
                ),
            },
            "label_vote_margin": {
                "type": "float",
                "required": False,
                "default": 1.3,
                "help": (
                    "How much better a rival must score before the reported identity changes, and "
                    "before a specific identity is named at all rather than the general one it sits "
                    "under. 1.0 disables both and restores per-frame flicker."
                ),
            },
        }

    # ------------------------------------------------------------------ internals

    def _lifetime_for(self, label_uuid: Optional[str]) -> LabelLifetimeSpec:
        """The TTL and anchor flag for a label, falling back to the tracker default."""
        if label_uuid and label_uuid in self._label_lifetimes:
            return self._label_lifetimes[label_uuid]
        return LabelLifetimeSpec(ttl_s=self._max_age_s, is_anchor=False)

    def _associate(
        self,
        track_idxs: Sequence[int],
        detections: Sequence[Detection],
        det_idxs: Sequence[int],
        threshold: float,
    ) -> List[Tuple[int, int]]:
        """Optimally pair the given tracks and detections by IoU.

        Returns (track_index, detection_index) pairs, both indexing into the caller's
        own lists. Pairs below `threshold`, and pairs whose labels belong to different
        `exclusive_variants` groups, are never formed — the cost matrix makes them
        infeasible rather than merely expensive.
        """
        if not track_idxs or not det_idxs:
            return []

        # linear_sum_assignment minimises, so cost is 1 - IoU. Mismatched labels and
        # sub-threshold overlaps are given a cost above any feasible pair, then filtered
        # out of the result: scipy requires a complete matrix, so infeasible pairs must
        # be representable rather than absent.
        infeasible = 10.0
        cost = np.full((len(track_idxs), len(det_idxs)), infeasible, dtype=float)
        for row, t_idx in enumerate(track_idxs):
            track = self._tracks[t_idx]
            for col, d_idx in enumerate(det_idxs):
                det = detections[d_idx]
                if not self.can_associate(track.label_uuid, det.label_uuid):
                    continue
                score = iou(track.bbox, det.bbox)
                if score >= threshold:
                    cost[row, col] = 1.0 - score

        rows, cols = linear_sum_assignment(cost)
        return [
            (track_idxs[r], det_idxs[c])
            for r, c in zip(rows, cols)
            if cost[r, c] < infeasible  # an infeasible pair means "no match", not a match
        ]

    def _absorb(self, track: Track, det: Detection, now: float) -> None:
        """Fold a matched detection into an existing track."""
        # The TRACK's label, not the detection's: association is group-wide, so a
        # `Person` track can absorb a `Alice` box, and reading the spec off the detection
        # would let one frame's guess swap the track's TTL and anchor rules out from
        # under it. Expiry below reads `t.label_uuid`, and these two must agree.
        spec = self._lifetime_for(track.label_uuid)
        # A latched anchor keeps the box it was confident about. Re-detections of a
        # stationary object jitter, and letting that jitter through would make a
        # containment rule flicker at the zone boundary.
        if not (track.is_anchor and det.confidence < self._anchor_confidence):
            track.bbox = det.bbox
        track.confidence = det.confidence
        # Identity is the vote's to decide, not this frame's. `label_uuid` and
        # `label_name` are both written here.
        self.record_observation(track, det, now)
        track.last_seen = now
        track.hits += 1
        track.matched_this_update = True
        if track.hits >= self._min_hits:
            track.confirmed = True
        if spec.is_anchor and det.confidence >= self._anchor_confidence:
            track.is_anchor = True

    # ------------------------------------------------------------------ API

    def update(self, detections: List[Detection], now: float) -> List[Track]:
        # Cleared before either association pass, and set again by `_absorb` for
        # whichever tracks a pass explains. A track still False afterwards is
        # exactly one that survived on its TTL while coasting, which is what
        # detector-miss collection reads (see Track.matched_this_update). Tracks
        # created below default to True — a brand-new track did match a detection.
        for track in self._tracks:
            track.matched_this_update = False
            track.forget_observation()

        high_idxs = [i for i, d in enumerate(detections) if d.confidence >= self._track_thresh]
        low_idxs = [i for i, d in enumerate(detections) if self._low_thresh <= d.confidence < self._track_thresh]

        # Stage 1 — confident detections against every live track.
        all_track_idxs = list(range(len(self._tracks)))
        first_pass = self._associate(all_track_idxs, detections, high_idxs, self._match_thresh)
        matched_tracks = {t_idx for t_idx, _ in first_pass}
        matched_dets = {d_idx for _, d_idx in first_pass}

        for t_idx, d_idx in first_pass:
            self._absorb(self._tracks[t_idx], detections[d_idx], now)

        # Stage 2 — the whole point. Tracks stage 1 could not explain get a second
        # chance against the weak boxes, which is what carries an object through an
        # occlusion instead of killing its id.
        leftover_tracks = [i for i in all_track_idxs if i not in matched_tracks]
        second_pass = self._associate(leftover_tracks, detections, low_idxs, self._low_match_thresh)
        for t_idx, d_idx in second_pass:
            self._absorb(self._tracks[t_idx], detections[d_idx], now)

        # New tracks come only from confident, unmatched detections. A weak box with no
        # history is far likelier to be noise than a new object, so low detections that
        # matched nothing are dropped rather than promoted.
        for d_idx in high_idxs:
            if d_idx in matched_dets:
                continue
            det = detections[d_idx]
            spec = self._lifetime_for(det.label_uuid)
            track = Track(
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
                vote=self.new_vote(),
            )
            self.record_observation(track, det, now)
            self._tracks.append(track)
            self._next_id += 1

        # Expire by wallclock age against the label's own TTL, never by frame count.
        # This is what makes "litter box missing for 30 s = still there, cat missing for
        # 1 s = gone" expressible without either camera's frame rate entering into it.
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
