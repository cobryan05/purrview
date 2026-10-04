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
from typing import Any, Dict, List, Mapping, Optional, Tuple

# A detection as it arrives from inference, before it has any identity.
BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

#: One class's score on one box: (label_uuid, confidence).
LabelScore = Tuple[Optional[str], float]


@dataclass
class Detection:
    """One model output for one frame."""

    bbox: BBox
    label_uuid: Optional[str] = None
    label_name: Optional[str] = None
    confidence: float = 0.0
    #: Runner-up classes the network scored on this same box, best-first, excluding the
    #: winner. Empty unless the caller asked inference for them, and always empty for
    #: NMS-free/E2E models, which bake the argmax into the graph. A label vote degrades
    #: to counting winners without these rather than failing — worse, not broken.
    alternates: List[LabelScore] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


#: How fast a track's accumulated label evidence fades, in seconds. Long enough to
#: outlast a multi-second run of the wrong label — the observed bursts were 2.4 s and
#: 7.5 s — and short enough that a track which genuinely changes hands settles onto the
#: new identity within a couple of seconds rather than carrying a dead vote for a minute.
LABEL_VOTE_HALF_LIFE_S = 6.0

#: How much better a challenger must be before the reported label moves. One number
#: serves both directions of the same question: how much better than the incumbent
#: before the track renames itself, and how much better than its nearest rival before a
#: *variant* is named at all rather than the group root. At 1.0 both tests collapse to
#: a bare argmax and the flicker comes straight back.
LABEL_VOTE_MARGIN = 1.3


#: Help text for the settings every tracker shares, so the two implementations cannot
#: drift into describing the same field differently. These are read by a person tuning a
#: camera in the Watchers editor, not by anyone who knows what a track is — each one says
#: what the setting does, then what to do when the symptom you are looking at appears.
SHARED_PARAM_HELP = {
    "min_hits": (
        "How many times in a row an object must be seen before the watcher will act on it. "
        "This is the false-alarm guard: at 1, a single bad frame is enough to open an event "
        "and record a clip. At a few frames a second, 3 costs well under a second of delay."
    ),
    "max_age_s": (
        "How long an object is remembered after the detector stops seeing it, in seconds. "
        "It covers brief misses — a cat behind a chair leg — so one is not counted as having "
        "left and come back. Too high and something that really did leave lingers; too low "
        "and a single missed frame splits one visit into two. Set in seconds rather than "
        "frames, so it means the same thing on a slow camera and a fast one. Individual "
        "labels can be given their own value in the lifetime rows above."
    ),
    "anchor_confidence": (
        "How sure the detector has to be before it is allowed to move an anchored object. "
        "Only affects labels you have ticked as Anchor above, which is how a fixed thing "
        "like a bin or a litter box stops drifting in and out of a zone: once it has been "
        "seen this confidently, weaker sightings keep it alive but no longer nudge its "
        "position."
    ),
    "label_vote_half_life_s": (
        "How long an object remembers what it has been called, in seconds. Rather than "
        "trusting whichever name the newest frame produced, the watcher keeps a fading "
        "tally of every name that object has been given and reports the winner — so one odd "
        "frame naming the wrong cat neither renames it nor splits it into a second object. "
        "Raise it if an identity still flickers; lower it if an object is slow to accept "
        "that it is now looking at someone else."
    ),
    "label_vote_margin": (
        "How far ahead a name has to be before the watcher uses it. Two jobs, same number: "
        "a rival name must beat the current one by this much before the object is renamed, "
        "and a specific name must beat its rivals by this much before it is used in place of "
        "the general one it belongs to — so a cat the model cannot quite pick out is "
        'reported as "Cat" rather than as a coin toss between two of them. Raise it to be '
        "more cautious and fall back to the general name more often. 1.0 turns both off and "
        "lets the name change every frame."
    ),
}


def shared_param(name: str, default: Any) -> Dict[str, Any]:
    """One tracker parameter whose meaning is identical in every tracker."""
    return {
        "type": "float" if isinstance(default, float) else "int",
        "required": False,
        "default": default,
        "help": SHARED_PARAM_HELP[name],
    }


@dataclass
class LabelVote:
    """A track's accumulated evidence for each label in its suppression group.

    Exists because a per-frame argmax between two correlated, near-tied class scores is
    not a measurement of identity — it is a coin toss the detector re-runs every frame,
    and every layer downstream used to treat each toss as a fact about the world.

    Three properties earn their keep:

    * **Every class's score counts, not just the winner's.** A frame that says
      `Person 0.72 / Alice 0.68` is evidence for *both*, and mostly for Person. Counting
      only winners discards the margin, which is the part that actually distinguishes a
      steady identity from an alternating one.
    * **Evidence decays in seconds**, never in frames — the same invariant the rest of
      this module holds to, and for the same reason: a frame count means different
      things on a 2 fps and a 25 fps camera.
    * **The group root is the honest answer to a tie.** `Person` in this vocabulary is
      not an abstention above `Alice`; it is the positive claim "a person I cannot name",
      which is exactly what a tie between its variants means.
    """

    half_life_s: float = LABEL_VOTE_HALF_LIFE_S
    #: label_uuid -> decayed sum of that class's scores.
    mass: Dict[str, float] = field(default_factory=dict)
    #: Last wallclock second the mass was decayed to. None until the first observation.
    updated_at: Optional[float] = None

    def observe(self, det: "Detection", now: float) -> None:
        """Fold one frame's scores for this object into the vote."""
        self._decay_to(now)
        if det.label_uuid:
            self.mass[det.label_uuid] = self.mass.get(det.label_uuid, 0.0) + max(0.0, det.confidence)
        for label_uuid, confidence in det.alternates:
            if label_uuid:
                self.mass[label_uuid] = self.mass.get(label_uuid, 0.0) + max(0.0, confidence)

    def _decay_to(self, now: float) -> None:
        if self.updated_at is not None and now > self.updated_at and self.half_life_s > 0:
            factor = 0.5 ** ((now - self.updated_at) / self.half_life_s)
            self.mass = {label: m * factor for label, m in self.mass.items()}
        self.updated_at = now

    def settled(self, incumbent: Optional[str], group_root: str, margin: float = LABEL_VOTE_MARGIN) -> Optional[str]:
        """The label this track should report, given what it reports now.

        `incumbent` keeps the seat unless a challenger clears `margin` — without that
        hysteresis the vote oscillates at the boundary exactly where the raw argmax
        does, only more slowly.
        """
        if not self.mass:
            return incumbent
        ranked = sorted(self.mass.items(), key=lambda kv: kv[1], reverse=True)
        leader, leader_mass = ranked[0]
        runner_up_mass = ranked[1][1] if len(ranked) > 1 else 0.0

        if incumbent and leader != incumbent and leader_mass < margin * self.mass.get(incumbent, 0.0):
            return incumbent
        # A variant must out-score its rivals to be named. Below the margin the group
        # root is the truthful answer, not the variant that happened to be ahead.
        if leader != group_root and leader_mass < margin * runner_up_mass:
            return group_root
        return leader


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
    #: Consecutive detections matched, reset by `miss()`. Used for the min-hits gate.
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
    #: Accumulated per-label evidence. `label_uuid` is this vote's verdict, not the
    #: label of whichever detection happened to start the track.
    vote: LabelVote = field(default_factory=LabelVote)
    #: What the detection matched on THIS update actually said, or None when the track
    #: coasted. The raw observation the settled label is smoothing over — kept so the
    #: clip sidecar can record the disagreement and the blip analysis can harvest it.
    observed_label_uuid: Optional[str] = None
    observed_confidence: float = 0.0
    #: Winner confidence minus best in-group runner-up on this frame. Small means the
    #: detector was confused between two names for one object, which is the calibrated
    #: "worth a human's time" signal — see AMBIGUITY_REVIEW_PLAN.md.
    observed_margin: Optional[float] = None

    def forget_observation(self) -> None:
        """Clear last frame's raw observation, before this frame's matching runs.

        A track still cleared afterwards is one that coasted: the detector said nothing
        about it at all, which is a different record from "the detector disagreed" and
        must not be confused with it downstream.
        """
        self.observed_label_uuid = None
        self.observed_confidence = 0.0
        self.observed_margin = None

    def miss(self) -> None:
        """Record that this update had no detection for this track.

        `hits` restarts rather than accumulating. The gate is "seen this many times in a
        row": a counter that only grows lets unrelated flickers, each one reviving the
        same track inside its TTL, add up to a confirmation the detector never supported.
        A confirmed track keeps its count — de-confirming on a dropout is `max_age_s`'s
        job, and `merge_rank` reads the history.
        """
        if not self.confirmed:
            self.hits = 0

    def age_s(self, now: float) -> float:
        """Seconds since this track last matched a detection."""
        return max(0.0, now - self.last_seen)

    def duration_s(self, now: float) -> float:
        return max(0.0, now - self.first_seen)

    def merge_rank(self, now: float) -> Tuple[float, int]:
        """How strong a claim this track has to survive a duplicate merge.

        Confidence decayed by how long the track has been coasting, then `hits` to
        break a tie — between two tracks the detector likes equally, the one with more
        history behind it is the one whose id is worth keeping.
        """
        decayed = self.confidence * (0.5 ** (self.age_s(now) / COAST_CONFIDENCE_HALF_LIFE_S))
        return (decayed, self.hits)

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

    def __init__(
        self,
        label_groups: Optional[Mapping[str, str]] = None,
        label_names: Optional[Mapping[str, str]] = None,
        label_vote_half_life_s: float = LABEL_VOTE_HALF_LIFE_S,
        label_vote_margin: float = LABEL_VOTE_MARGIN,
        **params: Any,
    ) -> None:
        self._params = params
        self._label_groups: Dict[str, str] = dict(label_groups or {})
        self._label_names: Dict[str, str] = dict(label_names or {})
        self._label_vote_half_life_s = label_vote_half_life_s
        self._label_vote_margin = label_vote_margin

    @classmethod
    def display_name(cls) -> str:
        return cls.__name__

    def group_of(self, label_uuid: Optional[str]) -> str:
        """The `exclusive_variants` group a label belongs to, or the label itself.

        Keyed the same way `suppression_group_for` keys it, because the map is built
        from that function — an ungrouped label is its own group, so the equality test
        below degrades exactly to the label comparison it replaced.
        """
        key = label_uuid or ""
        return self._label_groups.get(key, key)

    def can_associate(self, track_label: Optional[str], det_label: Optional[str]) -> bool:
        """Whether a detection may continue a track, by label.

        Labels in one `exclusive_variants` group are *alternate guesses about one
        physical object* — that is precisely what the flag declares, and it is the same
        set NMS already collapses to a single box per frame. So a group-mate must be
        allowed to continue the track: refusing it is what made one person walking
        through a yard produce nine tracks and three clips, as the argmax between two
        near-tied class scores alternated frame to frame and each spelling coasted and
        died on its own TTL.

        `Cat` against `Dog` is still refused — they share no group.
        """
        return self.group_of(track_label) == self.group_of(det_label)

    def new_vote(self) -> LabelVote:
        """A vote configured with this tracker's decay, for a track being born."""
        return LabelVote(half_life_s=self._label_vote_half_life_s)

    def record_observation(self, track: Track, det: Detection, now: float) -> None:
        """Fold a matched detection into a track's vote and re-settle its label.

        Called wherever a tracker absorbs a detection. The reported label is the vote's
        verdict from this point on, not the label of whichever detection started the
        track — that birth-assignment is what let a single mislabelled first frame name
        an object for its whole life.

        Safe to re-settle on every frame even before the track is confirmed: an
        unconfirmed track is withheld from every consumer by `active_tracks()`, so the
        vote has its warm-up for free and needs no separate gate.
        """
        track.vote.observe(det, now)
        track.observed_label_uuid = det.label_uuid
        track.observed_confidence = det.confidence
        best_rival = max((c for _label, c in det.alternates), default=None)
        track.observed_margin = None if best_rival is None else det.confidence - best_rival

        previous = track.label_uuid
        settled = track.vote.settled(previous, self.group_of(previous), self._label_vote_margin)
        track.label_uuid = settled
        track.label_name = self._settled_name(track.label_name, previous, settled, det)

    def _settled_name(
        self,
        current_name: Optional[str],
        previous: Optional[str],
        settled: Optional[str],
        det: Detection,
    ) -> Optional[str]:
        """The caption for a label the vote just settled on.

        The label tree answers first, because the vote can settle on a group root that
        no detection this track ever matched was named with — and a caption left over
        from the previous verdict would read `Alice` beside the uuid every rule matches
        `Person` on.

        A model class with no label row falls back to the detector's own word for it,
        which is what makes that misconfiguration visible rather than showing a uuid.
        Left as None when the verdict changed and nothing can name it: a bare uuid on
        the overlay is a better bug report than a confident wrong caption.
        """
        from_tree = self._label_names.get(settled or "")
        if from_tree:
            return from_tree
        if settled and det.label_uuid == settled and det.label_name:
            return det.label_name
        return current_name if settled == previous else None

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

#: How fast a coasting track's *ranking* confidence falls away, in seconds. Roughly two
#: analysed frames at 5 fps, and chosen so a frozen 0.90 loses to a live 0.54 after
#: about a third of a second — the observed failure was a stale box winning the merge
#: against the live detection that had replaced it. Ranking only: the reported
#: `confidence` is left alone, because that is a fact about the detector.
COAST_CONFIDENCE_HALF_LIFE_S = 0.5


def merge_duplicate_tracks(
    tracks: List[Track],
    now: float,
    iou_threshold: float = DUPLICATE_TRACK_IOU_THRESH,
) -> List[Track]:
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

    Greedy NMS, independent of label: two boxes this close together are the same
    object regardless of whether they currently agree on *who* it is — that
    disagreement is exactly the case a same-label-only check would miss.

    Ranked by `merge_rank`, not by raw confidence. A coasting track holds the
    confidence of whatever frame it last matched, so ranking on that made the merge
    actively prefer the stale box: a frozen `Person:0.90` beat the live `Alice:0.54`
    standing where the subject had actually walked to. Since the loser's id is dropped
    outright, that also discarded the accumulated history of the track that was still
    being fed.
    """
    if len(tracks) < 2:
        return tracks
    ordered = sorted(tracks, key=lambda t: t.merge_rank(now), reverse=True)
    kept: List[Track] = []
    for t in ordered:
        if any(_iou(t.bbox, k.bbox) >= iou_threshold for k in kept):
            continue
        kept.append(t)
    kept_ids = {t.track_id for t in kept}
    return [t for t in tracks if t.track_id in kept_ids]
