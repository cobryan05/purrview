"""Interaction rules and their per-rule state machine.

The state machine is the whole reason this module exists. A raw predicate like
`containment(cat, litterbox) > 0.6` crosses its threshold many times per second
because detection confidence and box geometry jitter. Dispatching on the raw
predicate means an MQTT storm and a dozen three-second clips for one litter-box
visit. Three guards fix that, and every rule carries all three:

* ``min_duration_s`` — the predicate must hold *continuously* before the event starts,
* ``rearm_delay_s`` — it must be *continuously false* before the event ends,
* ``max_duration_s`` — a hard cap, so a stuck detection can't record forever. ``0``
  disables the cap, for an event that should last exactly as long as its subject does.

States::

    IDLE ──predicate true──> PENDING ──held min_duration──> ACTIVE
      ^                         │                              │
      │                    predicate false                predicate false
      │                         v                              v
      ├───────────────────── IDLE                          CLOSING
      │                                                        │
      │                                   ┌────false for rearm_delay────> IDLE (END)
      │                                   └────true again────> ACTIVE (no new event)
      │                                                        │
      └──false for rearm_delay── SPENT <───hit max_duration────┘ (END, cap)

The `CLOSING` state is what coalesces a flapping predicate into one event: a
re-fire inside the rearm window rejoins the event in progress rather than opening a
new one.

`SPENT` is what stops the cap from becoming a metronome. A cap that reset straight to
IDLE re-opened the moment it was checked again, because the predicate that tripped it
is still true — so a car parked in the garage under a 30 s cap did not produce one
30 s clip, it produced one every thirty seconds until the car left, which is the exact
runaway the cap exists to prevent. After a capped close the subject must genuinely go
away — predicate false for `rearm_delay_s`, the same window the normal close uses —
before anything can open again.

`RuleSpec.trigger` moves *where* on that path the event is. The states and the three
guards are unchanged; what differs is which transition emits:

* ``while`` (default, and everything written before triggers existed) — the event spans
  the predicate, exactly as drawn above.
* ``on_enter`` — a zero-length event at the moment the subject appeared, emitted when
  PENDING clears `min_duration_s`, then straight to SPENT. One per arrival.
* ``on_exit`` — nothing is emitted on the way in; PENDING→ACTIVE only *arms* the rule,
  and the zero-length event fires at the moment the subject stopped matching, once
  `rearm_delay_s` has confirmed it really left.

An impulse carries no duration, so the footage around it comes from an action's
`preroll_s`/`postroll_s`: "the last 15 seconds before the car leaves the garage, but
only if it was there 15 minutes" is `on_exit` + `min_duration_s=900` + `preroll_s=15`.
The cap does not apply to either impulse — see the guard in `step`.
"""

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..tracking.tracker import BBox, Track
from .events import EventCloseReason, EventPhase, WatchEventSignal
from .geometry import (
    Zone,
    centroid,
    containment,
    inflate,
    iou,
    margin_pad,
    normalised_distance,
    overlap_ratio,
    zone_area_terms,
)
from .motion import DEFAULT_SENSITIVITY as MOTION_DEFAULT_SENSITIVITY
from .motion import MotionField

#: Trigger values, mirrored from `db.types.RuleTrigger`. Strings rather than that enum
#: because this module is pure — it holds no DB types, and a spec can be built from a
#: plain dict in a test with no database anywhere near it.
TRIGGER_WHILE = "while"
TRIGGER_ON_ENTER = "on_enter"
TRIGGER_ON_EXIT = "on_exit"

#: `event_granularity` values, mirrored from `db.types.EventGranularity` — strings, like
#: the triggers above, because this module holds no DB types.
#:
#: Neither is what the engine used to do. An event was per *label*, which is a property
#: of the detector's guess rather than of anything in the world: it split one person into
#: two events when the model changed its mind, and merged two cats into one when it did
#: not. So there is no "as before" option here, and a rule has to say which it meant.
GRANULARITY_SUBJECT = "subject"  # one event per distinct object
GRANULARITY_RULE = "rule"  # one event while the predicate holds for anyone


class RuleState(str, Enum):
    IDLE = "idle"
    PENDING = "pending"  # predicate true, waiting out min_duration_s
    ACTIVE = "active"  # event open
    CLOSING = "closing"  # predicate false, waiting out rearm_delay_s
    #: Closed by the cap while the predicate was still true. Nothing is open and nothing
    #: can open until the subject goes away for `rearm_delay_s`.
    SPENT = "spent"


@dataclass
class RuleSpec:
    """A rule's configuration, decoupled from its DB row so the engine stays pure."""

    uuid: str
    name: str
    subject_a_kind: str  # "label" | "zone" | "any"
    #: Every label (or zone) uuid this slot accepts. "any of these three cats" is one
    #: rule rather than three, and the member that actually matched is reported back on
    #: the event so an action's template can name it.
    subject_a_refs: List[str] = field(default_factory=list)
    relation: str = "contains"  # "present" | "iou" | "contains" | "overlaps" | "centroid_in" | "distance" | "motion"
    subject_b_kind: str = "none"  # "label" | "zone" | "none"
    subject_b_refs: List[str] = field(default_factory=list)
    #: Whether each slot's refs were widened to include their label descendants. Recorded
    #: for reporting only — `expand_descendants` has already rewritten the ref lists by
    #: the time the engine sees a spec, so matching never consults these. The engine is
    #: pure and has no label tree to consult even if it wanted to.
    subject_a_include_descendants: bool = False
    subject_b_include_descendants: bool = False
    threshold: float = 0.5
    #: Widen the match by this much *of the smaller box's size* before the relation is
    #: measured, so a rule can fire on "nearly touching" without a relation of its own.
    #: 1.0 closes a gap of about one subject-width; negative shrinks instead, demanding
    #: a deeper overlap than the raw boxes show. Sized against the object rather than
    #: the frame so it survives perspective — see `geometry.margin_pad`.
    #: Ignored by `present` (no geometry) and `distance` (already a proximity measure).
    margin: float = 0.0
    #: `motion` only: how big a per-pixel luminance change counts as movement, normalised
    #: to 0..1. Separate from `threshold`, which is *how much of the region* must change:
    #: they are the two independent halves of "did anything move", and one value cannot
    #: express both. 0 means "use the module default" — see `motion.DEFAULT_SENSITIVITY`.
    motion_sensitivity: float = 0.0
    #: When the rule fires: `while` (the event spans the predicate), `on_enter` or
    #: `on_exit` (a zero-length impulse at that edge). The guards are shared rather than
    #: duplicated — see `RuleTrigger` in `db.types` for what each one means per trigger.
    #: A plain string, like `relation`, because the engine is pure and holds no enums
    #: from the DB layer.
    trigger: str = "while"
    #: Whether the rule opens one event per object or one event for the whole rule. See
    #: the `GRANULARITY_*` constants; `subject` is the default because "one event per
    #: thing that happened" is the more useful card.
    event_granularity: str = GRANULARITY_SUBJECT
    min_duration_s: float = 2.0
    rearm_delay_s: float = 10.0
    max_duration_s: float = 300.0
    enabled: bool = True
    event_label: Optional[str] = None
    #: Sources this rule is evaluated on; empty means all of them. Applied by
    #: `WatchEngine`, which is already per-source, rather than inside the predicate — a
    #: rule that does not apply to a camera must have no *state* on that camera either,
    #: or its machine would sit in CLOSING and emit a stray END later.
    #: Declared last so the field order the rest of the dataclass exposes is unchanged.
    source_uuids: List[str] = field(default_factory=list)
    #: Whether the rule is armed *right now*, as opposed to `enabled`, which is whether
    #: the user configured it to run at all.
    #:
    #: The two are deliberately separate. `enabled` is authored intent, lives in the DB
    #: and is what the Watchers editor shows; `armed` is runtime state driven from Home
    #: Assistant ("record people only while the house is empty") and is never persisted —
    #: it is restored from its own retained MQTT topic instead. Folding the two together
    #: would mean every arrival and departure wrote to the config DB and silently moved a
    #: checkbox the user had set by hand.
    #:
    #: A rule that is `enabled=False` has no HA representation at all, so it can never be
    #: armed or disarmed from there; arming only ever narrows what an enabled rule does.
    armed: bool = True

    @property
    def active(self) -> bool:
        """Whether this rule should be evaluated at all on this frame.

        The one place the two flags are combined, so no caller has to remember that
        there are two.
        """
        return self.enabled and self.armed

    @classmethod
    def from_row(cls, row) -> "RuleSpec":
        get = row.get if isinstance(row, dict) else lambda k, d=None: getattr(row, k, d)
        return cls(
            uuid=get("uuid"),
            name=get("name"),
            subject_a_kind=get("subject_a_kind", "label"),
            subject_a_refs=[str(r) for r in (get("subject_a_refs_json") or [])],
            relation=get("relation", "contains"),
            subject_b_kind=get("subject_b_kind", "none"),
            subject_b_refs=[str(r) for r in (get("subject_b_refs_json") or [])],
            subject_a_include_descendants=bool(get("subject_a_include_descendants", False)),
            subject_b_include_descendants=bool(get("subject_b_include_descendants", False)),
            threshold=get("threshold", 0.5),
            margin=get("margin", 0.0) or 0.0,
            motion_sensitivity=get("motion_sensitivity", 0.0) or 0.0,
            trigger=get("trigger", "while") or "while",
            event_granularity=get("event_granularity", GRANULARITY_SUBJECT) or GRANULARITY_SUBJECT,
            min_duration_s=get("min_duration_s", 2.0),
            rearm_delay_s=get("rearm_delay_s", 10.0),
            max_duration_s=get("max_duration_s", 300.0),
            enabled=get("enabled", True),
            event_label=get("event_label"),
            source_uuids=[str(s) for s in (get("source_uuids_json") or [])],
        )

    def narrowed(self, subject_a_ref: Optional[str], subject_b_ref: Optional[str]) -> "RuleSpec":
        """This rule restricted to one member of each slot.

        Used for the two slots that still deserve a machine of their own, and only those:
        a `motion` rule's subject (which names a *place* to measure, not an object) and an
        impulse rule's target (entering the second bowl is a second arrival). Everything
        else keys on the rule alone and separates objects inside the machine, because a
        label is a guess about the subject rather than an identity for it.

        `None` leaves a slot alone.
        """
        return replace(
            self,
            subject_a_refs=[subject_a_ref] if subject_a_ref is not None else list(self.subject_a_refs),
            subject_b_refs=[subject_b_ref] if subject_b_ref is not None else list(self.subject_b_refs),
        )


def descendants_of(label_uuid: str, ancestry: Dict[str, Sequence[str]]) -> List[str]:
    """Every label under `label_uuid`, from an {uuid: ancestors} map.

    Inverted from the ancestry map the watcher already builds rather than fetching a
    second parent→children view of the same tree.
    """
    return sorted(uuid for uuid, ancestors in ancestry.items() if label_uuid in set(ancestors or ()))


def expand_descendants(spec: RuleSpec, ancestry: Dict[str, Sequence[str]]) -> RuleSpec:
    """Widen a rule's label slots to cover their descendants, returning a new spec.

    Done here, once at watcher start, rather than inside the matching loop: the engine is
    pure and has no label tree, and re-deriving the closure per frame would be pointless
    work at 5 fps per camera.

    Storing the expansion instead of the flag was the alternative and is worse — it
    freezes the rule at the moment it was written, so a label added to the tree later
    silently stops being matched. The cost of doing it here is that a *newly added* label
    is picked up when the watcher restarts, not the instant it is created.

    Only label slots expand. Zones have no hierarchy, and a zone uuid appears in no
    ancestry map, so a zone slot would silently widen to nothing.
    """
    if not ancestry:
        return spec

    def widen(kind: str, refs: List[str], enabled: bool) -> List[str]:
        if not enabled or kind != "label":
            return refs
        widened = list(refs)
        seen = set(refs)
        for ref in refs:
            for child in descendants_of(ref, ancestry):
                if child not in seen:
                    seen.add(child)
                    widened.append(child)
        return widened

    return replace(
        spec,
        subject_a_refs=widen(spec.subject_a_kind, spec.subject_a_refs, spec.subject_a_include_descendants),
        subject_b_refs=widen(spec.subject_b_kind, spec.subject_b_refs, spec.subject_b_include_descendants),
    )


@dataclass(frozen=True)
class PredicateMatch:
    """One (subject, target) pair that fired on one frame.

    The aggregate fields on `PredicateResult` answer "did anything match"; these answer
    "what, individually". An event is per *object*, so the machine has to see the pairs
    separately — a result that reports only the strongest pair cannot tell one cat at two
    bowls from two cats at one bowl, and those are one event and two.

    `bbox` rides along because associating this frame's matches to the events already
    open is a geometric question, and the machine holds no tracks of its own to look it
    up in.
    """

    track_id: int
    bbox: BBox
    subject_ref: Optional[str]
    target_ref: Optional[str]
    score: float
    confidence: float
    #: Whether the tracker matched this track to a detection on this frame, as opposed
    #: to carrying it while it coasts. Unknown trackers report True.
    matched: bool = True
    #: The frame time the tracker created this track on, in the engine's clock.
    first_seen: float = 0.0


@dataclass
class PredicateResult:
    """Whether a rule's condition holds right now, and which tracks satisfied it."""

    holds: bool
    track_ids: List[int] = field(default_factory=list)
    score: float = 0.0
    confidence: float = 0.0
    #: Which member of each slot the strongest matching pair used. With multi-ref slots
    #: "a cat is in a litter box" is no longer enough to build an MQTT topic from — the
    #: caller needs to know *which* cat and *which* litter box, and only the evaluation
    #: knows that.
    subject_ref: Optional[str] = None
    target_ref: Optional[str] = None
    #: Every pair that fired, not just the strongest. Empty for a `motion` rule, which
    #: measures pixels and has no tracks to report by construction.
    matches: List[PredicateMatch] = field(default_factory=list)


def _matching_tracks(tracks: Sequence[Track], kind: str, refs: Sequence[str]) -> List[Track]:
    if kind == "any":
        return list(tracks)
    if kind == "label":
        wanted = set(refs)
        return [t for t in tracks if t.label_uuid in wanted]
    return []


def evaluate_predicate(
    spec: RuleSpec,
    tracks: Sequence[Track],
    zones: Dict[str, Zone],
    frame_width: float,
    frame_height: float,
    motion: Optional[MotionField] = None,
) -> PredicateResult:
    """Evaluate a rule against one frame's tracks. Pure — no state, no I/O."""
    # Arity 0: motion is the one relation that reads pixels rather than tracks, so it
    # has to be answered before the subject lookup below — which would return an empty
    # list on a frame with no detections and report "no motion" for exactly the frames
    # a motion rule exists to catch.
    if spec.relation == "motion":
        if motion is None or not motion.measured:
            return PredicateResult(holds=False)
        # Subject A names *where* to measure, not what to look for: zones scope the
        # measurement to a region, anything else means the whole (unmasked) frame.
        zone_refs = spec.subject_a_refs if spec.subject_a_kind == "zone" else ()
        sensitivity = spec.motion_sensitivity or MOTION_DEFAULT_SENSITIVITY
        score, zone_uuid = motion.best(sensitivity, zone_refs)
        holds = score >= spec.threshold
        return PredicateResult(
            holds=holds,
            # Motion has no tracks by construction. An event opened by it still names
            # its zone, which is what an action's template needs.
            track_ids=[],
            score=score,
            confidence=score,
            subject_ref=zone_uuid if holds else None,
        )

    subjects_a = _matching_tracks(tracks, spec.subject_a_kind, spec.subject_a_refs)
    if not subjects_a:
        return PredicateResult(holds=False)

    # Arity 1: "a cat is tracked at all". Same machinery as the two-subject relations,
    # which is why single-object rules need no special case downstream.
    if spec.relation == "present":
        best = max(subjects_a, key=lambda t: t.confidence)
        holds = best.confidence >= spec.threshold
        firing = [t for t in subjects_a if t.confidence >= spec.threshold]
        return PredicateResult(
            holds=holds,
            track_ids=[t.track_id for t in firing],
            score=best.confidence,
            confidence=best.confidence,
            subject_ref=best.label_uuid if holds else None,
            # Each track's own label, not the best one's: three cats in shot are three
            # matches under three names, and collapsing them to the clearest is exactly
            # what loses the other two.
            matches=[
                PredicateMatch(
                    track_id=t.track_id,
                    bbox=t.bbox,
                    subject_ref=t.label_uuid,
                    target_ref=None,
                    score=t.confidence,
                    confidence=t.confidence,
                    # Explicit because the field defaults to True: the event layer
                    # reads it to tell one body under two ids from two bodies, and a
                    # coasting track must not look like a fresh detection.
                    matched=bool(getattr(t, "matched_this_update", True)),
                    first_seen=float(getattr(t, "first_seen", 0.0) or 0.0),
                )
                for t in firing
            ],
        )

    # Subject B is either one or more zones (the common, most reliable case) or another
    # label. Multiple zones read as "inside any of these", which is what a rule spanning
    # two litter boxes needs.
    if spec.subject_b_kind == "zone":
        targets: List[Any] = [zones[ref] for ref in spec.subject_b_refs if ref in zones]
        if not targets:
            # A dangling zone reference can only ever be false. The validate endpoint
            # surfaces this at config time; here it must simply not fire.
            return PredicateResult(holds=False)
    else:
        targets = _matching_tracks(tracks, spec.subject_b_kind, spec.subject_b_refs)
        if not targets:
            return PredicateResult(holds=False)

    # `distance` is excluded from the margin: its threshold *is* a proximity measure,
    # and it compares centre points, which do not grow.
    margin = 0.0 if spec.relation == "distance" else spec.margin

    best_score = 0.0
    best_conf = 0.0
    matched: List[int] = []
    matches: List[PredicateMatch] = []
    # The slot values reported belong to the best-scoring *matching* pair, tracked
    # separately from `best_score` — which also moves for pairs that never fired, and
    # would otherwise name a cat that failed the threshold.
    best_match_score = -1.0
    subject_ref: Optional[str] = None
    target_ref: Optional[str] = None

    for track_a in subjects_a:
        for target in targets:
            # The margin is measured against the smaller of the pair, so it has to be
            # computed per pair rather than once for the rule — see `margin_pad` for
            # why the object, and not the frame, is the ruler.
            target_box = target.bounds(frame_width, frame_height) if isinstance(target, Zone) else target.bbox
            pad = margin_pad(margin, track_a.bbox, target_box) if margin else 0.0
            box_a = inflate(track_a.bbox, pad) if pad else track_a.bbox
            box_b = inflate(target_box, pad) if pad else target_box

            if isinstance(target, Zone):
                if spec.relation == "centroid_in":
                    # The subject contributes a point, which cannot be inflated, so the
                    # whole margin widens the zone — the same total slack as inflating
                    # both, and it keeps "the cat's centre is within N of the zone".
                    score = (
                        1.0
                        if target.contains_point_within(centroid(track_a.bbox), 2 * pad, frame_width, frame_height)
                        else 0.0
                    )
                elif spec.relation == "distance":
                    # Distance is an inverted predicate: closer is "more true", so the
                    # threshold is an upper bound and the score is inverted to keep
                    # "higher score = closer to firing" true everywhere.
                    dist = normalised_distance(track_a.bbox, target_box, frame_width, frame_height)
                    score = 1.0 - dist if dist <= spec.threshold else 0.0
                else:
                    # Measured against the polygon itself, not its bounding box. These
                    # three used to share the box relations above via `Zone.bounds`,
                    # which for a traced yard is most of the frame — a person on the
                    # pavement scored a full 1.0 `overlaps` and fired the rule. See
                    # `zone_area_terms` for where the margin lands, which differs by kind.
                    inter, subject_area, zone_area = zone_area_terms(
                        target, track_a.bbox, margin, frame_width, frame_height
                    )
                    if spec.relation == "iou":
                        union = subject_area + zone_area - inter
                        score = inter / union if union > 0 else 0.0
                    elif spec.relation == "overlaps":
                        smaller = min(subject_area, zone_area)
                        score = inter / smaller if smaller > 0 else 0.0
                    else:  # contains
                        score = inter / subject_area if subject_area > 0 else 0.0
            else:
                if track_a.track_id == target.track_id:
                    continue  # an object never interacts with itself
                if spec.relation == "centroid_in":
                    cx, cy = centroid(track_a.bbox)
                    # Whole margin on the target, as in the zone branch above.
                    bx1, by1, bx2, by2 = inflate(target_box, 2 * pad)
                    score = 1.0 if (bx1 <= cx <= bx2 and by1 <= cy <= by2) else 0.0
                elif spec.relation == "distance":
                    dist = normalised_distance(track_a.bbox, target_box, frame_width, frame_height)
                    score = 1.0 - dist if dist <= spec.threshold else 0.0
                elif spec.relation == "iou":
                    score = iou(box_a, box_b)
                elif spec.relation == "overlaps":
                    score = overlap_ratio(box_a, box_b)
                else:  # contains
                    score = containment(box_a, box_b)

            # centroid_in and distance already encode their own thresholding, so a
            # positive score means the predicate held for that pair.
            fired = score > 0.0 if spec.relation in ("centroid_in", "distance") else score >= spec.threshold
            if fired:
                matched.append(track_a.track_id)
                matches.append(
                    PredicateMatch(
                        track_id=track_a.track_id,
                        bbox=track_a.bbox,
                        subject_ref=track_a.label_uuid,
                        target_ref=target.uuid if isinstance(target, Zone) else target.label_uuid,
                        score=score,
                        confidence=track_a.confidence,
                        matched=bool(getattr(track_a, "matched_this_update", True)),
                        first_seen=float(getattr(track_a, "first_seen", 0.0) or 0.0),
                    )
                )
                if score > best_match_score:
                    best_match_score = score
                    subject_ref = track_a.label_uuid
                    target_ref = target.uuid if isinstance(target, Zone) else target.label_uuid
            if score > best_score:
                best_score = score
                best_conf = track_a.confidence

    return PredicateResult(
        holds=bool(matched),
        track_ids=sorted(set(matched)),
        score=best_score,
        confidence=best_conf,
        subject_ref=subject_ref,
        target_ref=target_ref,
        matches=matches,
    )


#: Minimum IoU at which `cluster_matches` joins two tracks into one object. Looser than
#: the tracker's `DUPLICATE_TRACK_IOU_THRESH` (0.85), and applied only to pairs where at
#: most one side was matched to a detection on this frame.
EVENT_CLUSTER_IOU = 0.3

#: Ceiling on concurrent events one machine will hold. A machine is no longer allocated
#: once at load time — it grows an instance per distinct object — so it needs a bound.
#: Past the cap a new cluster joins its nearest neighbour instead of opening an event of
#: its own: over-merging costs a shared card, whereas refusing outright would silently
#: drop a real event.
MAX_EVENT_INSTANCES = 16

#: How many track ids one machine remembers coexistence for. Bounded because track ids
#: are minted forever while the map only has to survive from "both subjects were seen
#: together" to the frame one of them is missed on — seconds, not days.
MAX_COEXIST_TRACKS = 512

#: Shared empty set for the coexistence lookups, so a miss allocates nothing.
_NO_TRACKS: Set[int] = set()


def _bbox_iou(a: BBox, b: BBox) -> float:
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


def _best_iou(left: Sequence[BBox], right: Sequence[BBox]) -> float:
    return max((_bbox_iou(a, b) for a in left for b in right), default=0.0)


@dataclass
class MatchCluster:
    """One frame's matches, grouped into what is probably a single physical object.

    The unit an event is per. A cluster is not a track: one body routinely arrives as
    several tracks (a detector dropout breaks association and a new id starts beside the
    old one), and the whole reason this type exists is that those must be one event.
    """

    track_ids: Set[int] = field(default_factory=set)
    boxes: List[BBox] = field(default_factory=list)
    subject_ref: Optional[str] = None
    target_ref: Optional[str] = None
    score: float = 0.0
    confidence: float = 0.0
    #: True when any track in the cluster was matched to a detection on this frame.
    matched: bool = False
    #: Earliest `first_seen` across the cluster's tracks; +inf for a trackless cluster.
    first_seen: float = float("inf")


def cluster_matches(
    matches: Sequence[PredicateMatch],
    iou_threshold: float = EVENT_CLUSTER_IOU,
    collapse: bool = False,
) -> List[MatchCluster]:
    """Group a frame's firing pairs by physical object.

    Two passes. First by track id, because one track matching two targets is one object
    at two bowls, not two objects. Then by overlap, because one object is routinely
    carried by more than one track id — but never two tracks the tracker matched to
    separate detections on this frame. Those are two objects by the tracker's own
    verdict, whatever their boxes do; the id-beside-id case this pass exists for has the
    old id coasting while the new one is the one being matched.

    `collapse` is `rule` granularity: every match becomes one cluster, so the rule holds
    a single event however many objects satisfy it. That is the right answer for
    "something is in the yard" — where the recording is the whole frame either way — and
    the wrong one for "a cat drank water", which is per cat.
    """
    if not matches:
        return []
    if collapse:
        merged = MatchCluster()
        for m in matches:
            _absorb(merged, m)
        return [merged]

    by_track: Dict[int, MatchCluster] = {}
    for m in matches:
        _absorb(by_track.setdefault(m.track_id, MatchCluster()), m)

    clusters: List[MatchCluster] = []
    for candidate in by_track.values():
        for existing in clusters:
            if candidate.matched and existing.matched:
                continue
            if _best_iou(candidate.boxes, existing.boxes) >= iou_threshold:
                _join(existing, candidate)
                break
        else:
            clusters.append(candidate)
    return clusters


def _absorb(cluster: MatchCluster, m: PredicateMatch) -> None:
    """Fold one match into a cluster, keeping the strongest pair's slot values."""
    cluster.track_ids.add(m.track_id)
    if m.bbox not in cluster.boxes:
        cluster.boxes.append(m.bbox)
    if m.score >= cluster.score or cluster.subject_ref is None:
        cluster.score = max(cluster.score, m.score)
        if m.subject_ref is not None:
            cluster.subject_ref = m.subject_ref
        if m.target_ref is not None:
            cluster.target_ref = m.target_ref
    cluster.confidence = max(cluster.confidence, m.confidence)
    cluster.matched = cluster.matched or m.matched
    cluster.first_seen = min(cluster.first_seen, m.first_seen)


def _join(into: MatchCluster, other: MatchCluster) -> None:
    into.track_ids |= other.track_ids
    into.boxes.extend(b for b in other.boxes if b not in into.boxes)
    into.confidence = max(into.confidence, other.confidence)
    into.matched = into.matched or other.matched
    into.first_seen = min(into.first_seen, other.first_seen)
    if other.score > into.score:
        into.score = other.score
        into.subject_ref = other.subject_ref or into.subject_ref
        into.target_ref = other.target_ref or into.target_ref


@dataclass
class _EventInstance:
    """One event's hysteresis, anchored to a cluster rather than to a label.

    Everything `RuleMachine` used to hold as fields of itself. The move to a set of these
    is what takes the subject's *label* out of an event's identity: a relabel inside an
    exclusive-variants group leaves the same object matching the same rule, so it lands
    on the same instance and the open event simply carries on.
    """

    instance_id: str
    state: RuleState = RuleState.IDLE
    true_since: Optional[float] = None
    false_since: Optional[float] = None
    started_at: Optional[float] = None
    #: Every track that has ever been part of this event. Unioned, never replaced, so an
    #: obstruction that drops the subject and hands it back under fresh ids still reports
    #: one event covering all of them.
    track_ids: Set[int] = field(default_factory=set)
    #: Boxes from the most recent frame this instance matched on — what an unassociated
    #: cluster is compared against.
    boxes: List[BBox] = field(default_factory=list)
    peak_confidence: float = 0.0
    target_ref: Optional[str] = None
    #: Accumulated confidence per subject label over the last `max_duration_s`. The
    #: reported subject is this pile's leader rather than the latest frame's answer: an
    #: event is named by the weight of what it saw, so a single odd frame at the closing
    #: edge cannot rename the whole visit.
    #:
    #: The window is what keeps that true in both directions. An instance outlives its
    #: event — a SPENT latch keeps observing, and an `on_exit` instance is exempt from
    #: the runaway cap and so sits ACTIVE for the whole dwell, days if the subject stays
    #: put. An unwindowed pile there is not "the weight of what this event saw" but the
    #: weight of everything the instance ever saw, and a departure cannot outvote the
    #: parked hours that preceded it — least of all when the instance changed object
    #: mid-dwell.
    label_mass: Dict[str, float] = field(default_factory=dict)
    #: `label_mass` split into whole-second buckets so the window can be enforced by
    #: subtraction. Seconds, not frames: the pile only has to be accurate enough to pick
    #: a winner, and per-frame entries cost 16 instances × the frame rate × the window.
    mass_log: Deque[Tuple[int, Dict[str, float]]] = field(default_factory=deque)
    #: True when the object this instance opened on already existed when the machine
    #: started watching. An `on_enter` instance so marked latches without firing.
    preexisting: bool = False

    @property
    def subject_ref(self) -> Optional[str]:
        if not self.label_mass:
            return None
        return max(self.label_mass.items(), key=lambda kv: kv[1])[0]

    def observe(self, cluster: MatchCluster, now: float, window_s: float) -> None:
        self.track_ids |= cluster.track_ids
        self.boxes = list(cluster.boxes)
        self.peak_confidence = max(self.peak_confidence, cluster.confidence)
        if cluster.subject_ref is not None:
            # Floored so a zero-confidence match still records *which* label matched.
            # Without it an event whose every observation scored 0 could not be named.
            self._add_mass(cluster.subject_ref, max(0.0, cluster.confidence) or 1e-9, now)
        if cluster.target_ref is not None:
            self.target_ref = cluster.target_ref
        self._expire_mass(now, window_s)

    def _add_mass(self, label: str, weight: float, now: float) -> None:
        self.label_mass[label] = self.label_mass.get(label, 0.0) + weight
        second = int(now)
        if self.mass_log and self.mass_log[-1][0] == second:
            bucket = self.mass_log[-1][1]
            bucket[label] = bucket.get(label, 0.0) + weight
        else:
            self.mass_log.append((second, {label: weight}))

    def _expire_mass(self, now: float, window_s: float) -> None:
        """Drop buckets older than the window, and their contribution to the pile.

        A window of zero or less is `max_duration_s`'s "no cap" reading, the same one the
        runaway cap gives it: the caller has asked for an event that lasts exactly as
        long as its subject does, and a pile that quietly forgot the start of it would
        not be that.
        """
        if window_s <= 0:
            return
        cutoff = now - window_s
        while self.mass_log and self.mass_log[0][0] < cutoff:
            _, bucket = self.mass_log.popleft()
            for label, weight in bucket.items():
                # Popped rather than left at ~0: `subject_ref` takes the leader of
                # whatever is in the pile, so a label whose every observation has aged
                # out must not still be a candidate for naming the event.
                remaining = self.label_mass.get(label, 0.0) - weight
                if remaining > 1e-12:
                    self.label_mass[label] = remaining
                else:
                    self.label_mass.pop(label, None)

    def reset(self) -> None:
        self.state = RuleState.IDLE
        self.true_since = None
        self.false_since = None
        self.started_at = None
        self.track_ids = set()
        self.boxes = []
        self.peak_confidence = 0.0
        self.target_ref = None
        self.label_mass = {}
        self.mass_log.clear()
        self.preexisting = False

    @property
    def spent(self) -> bool:
        return self.state is RuleState.SPENT

    @property
    def retired(self) -> bool:
        """Nothing left to remember — safe to drop from the machine's table."""
        return self.state is RuleState.IDLE


class RuleMachine:
    """One rule's state on one source: a set of concurrently open events.

    Was one event's worth of hysteresis per `(rule, subject label, target)` triple. Both
    of those refs were mutable facts about the subject rather than identities, so a
    person the detector renamed mid-crossing, and a cat that moved from one bowl to the
    next, each ended one event and opened another. The machine now keys on nothing but
    the rule and holds an instance per object — see `_EventInstance`.
    """

    def __init__(
        self,
        spec: RuleSpec,
        source_uuid: Optional[str] = None,
        label_groups: Optional[Mapping[str, str]] = None,
        cluster_iou: float = EVENT_CLUSTER_IOU,
    ) -> None:
        self.spec = spec
        self.source_uuid = source_uuid
        self._label_groups: Dict[str, str] = dict(label_groups or {})
        self._cluster_iou = cluster_iou
        self._instances: Dict[str, _EventInstance] = {}
        self._next_instance = 0
        #: Track ids that have been seen in the same frame as each other, in *different*
        #: clusters. Two tracks that have ever been apart in one frame are two objects,
        #: whatever the detector called them, and this is the only thing in a frame that
        #: says so — a dropout's replacement id never shares a frame with the id it
        #: replaced, while two cars in one garage share hundreds.
        self._coexist: Dict[int, Set[int]] = {}
        #: Frame time of the first `step` since the machine was built or last re-armed;
        #: None while inactive. Tracks created at or before it were already there.
        self._watching_since: Optional[float] = None
        #: Totalled on the machine, not the instance, so a tuning hint survives the
        #: eviction of the instance that flapped.
        self.flap_count: int = 0

    # ------------------------------------------------------------------ helpers

    def _record_coexistence(self, clusters: Sequence[MatchCluster]) -> None:
        """Note which of this frame's tracks are provably about different objects.

        Across clusters only. Two tracks the clusterer put in *one* cluster are its
        verdict that they are one body carried by two ids, which is the case adoption is
        for; recording those as rivals would make `_disown_rival_tracks` throw away the
        very association that found them.

        Matched clusters only. Two tracks are two objects by the tracker's own verdict
        only where it had a separate detection for each; an unmatched cluster is coasting
        on a frozen box where the body used to be, so pairing it with the id that
        replaced it files one body as two. The record is permanent, so that split never
        heals.
        """
        clusters = [cluster for cluster in clusters if cluster.matched]
        if len(clusters) < 2:
            return
        for index, cluster in enumerate(clusters):
            rivals = {t for other in clusters[index + 1 :] for t in other.track_ids}
            if not rivals:
                continue
            for track_id in cluster.track_ids:
                self._coexist.setdefault(track_id, set()).update(rivals)
                for rival in rivals:
                    self._coexist.setdefault(rival, set()).add(track_id)
        while len(self._coexist) > MAX_COEXIST_TRACKS:
            # Insertion order, so the oldest track id goes first. Its id lingers in other
            # entries' sets, which is harmless: a stale id never appears in a frame again.
            self._coexist.pop(next(iter(self._coexist)))

    def _coexisted(self, left: Set[int], right: Set[int]) -> bool:
        return any(self._coexist.get(track_id, _NO_TRACKS) & right for track_id in left)

    def _disown_rival_tracks(self, instance: _EventInstance, cluster: MatchCluster) -> None:
        """Drop tracks this instance cannot still be about.

        `track_ids` is unioned and never replaced, so one wrong adoption leaves an
        instance permanently ambiguous: both objects' clusters then match it on a shared
        id and which one owns it comes down to cluster ordering. A track that has been in
        the frame alongside this cluster is a different object, so it is not this one.
        """
        rivals = {
            t for t in instance.track_ids - cluster.track_ids if self._coexist.get(t, _NO_TRACKS) & cluster.track_ids
        }
        instance.track_ids -= rivals

    def _group_of(self, label_uuid: Optional[str]) -> str:
        key = label_uuid or ""
        return self._label_groups.get(key, key)

    def _new_instance(self) -> _EventInstance:
        self._next_instance += 1
        instance = _EventInstance(instance_id=f"{self.spec.uuid}:{self._next_instance}")
        self._instances[instance.instance_id] = instance
        return instance

    def _signal(
        self,
        instance: _EventInstance,
        phase: EventPhase,
        now: float,
        close_reason: EventCloseReason = EventCloseReason.PREDICATE,
    ) -> WatchEventSignal:
        return WatchEventSignal(
            rule_uuid=self.spec.uuid,
            rule_name=self.spec.name,
            phase=phase,
            at=now,
            started_at=instance.started_at if instance.started_at is not None else now,
            source_uuid=self.source_uuid,
            track_ids=sorted(instance.track_ids),
            peak_confidence=instance.peak_confidence,
            close_reason=close_reason,
            subject_ref=instance.subject_ref,
            target_ref=instance.target_ref,
            instance_id=instance.instance_id,
            metadata={"eventLabel": self.spec.event_label} if self.spec.event_label else {},
        )

    def _impulse(self, instance: _EventInstance, at: Optional[float]) -> List[WatchEventSignal]:
        """One edge as a zero-length event: START and END at the same instant.

        Both phases rather than a phase of its own, because every consumer downstream —
        the recorder's open map, the MQTT publisher, the actions' `event_started` /
        `event_ended` triggers — is built on the pair, and a third phase would mean
        teaching all of them a case that carries no information they do not already
        have.

        `started_at` is set to the anchor first so both signals carry it, and so
        `duration_s` on the END is exactly 0 rather than the dwell that preceded it.
        """
        instance.started_at = at
        moment = at if at is not None else 0.0
        return [
            self._signal(instance, EventPhase.START, moment),
            self._signal(instance, EventPhase.END, moment, EventCloseReason.PREDICATE),
        ]

    def _spend(self, instance: _EventInstance, false_since: Optional[float]) -> None:
        """Drop the event, keep the identity: nothing may open until the subject is gone.

        `track_ids`, the boxes and the label pile deliberately survive. They are how the
        latch recognises the subject it just capped — clear them and the next frame looks
        like a brand new arrival, which turns the cap straight back into the metronome
        SPENT exists to prevent.
        """
        instance.state = RuleState.SPENT
        instance.true_since = None
        instance.started_at = None
        instance.peak_confidence = 0.0
        instance.false_since = false_since

    @staticmethod
    def _is_open(spec: RuleSpec, instance: _EventInstance) -> bool:
        return spec.trigger == TRIGGER_WHILE and instance.state in (RuleState.ACTIVE, RuleState.CLOSING)

    def instance_is_open(self, instance: _EventInstance) -> bool:
        return self._is_open(self.spec, instance)

    # ------------------------------------------------------------------ association

    def _clusters(self, result: PredicateResult) -> List[MatchCluster]:
        """This frame's matches as clusters, one per object the event layer can see.

        A predicate that holds with no matches at all is `motion`, which measures pixels
        and reports a zone. It still deserves an event, so it becomes one trackless
        cluster — which `_associate` then routes to the machine's single instance,
        because a trackless rule can never distinguish a second object anyway.
        """
        if result.matches:
            return cluster_matches(
                result.matches,
                self._cluster_iou,
                collapse=self.spec.event_granularity != GRANULARITY_SUBJECT,
            )
        if result.holds:
            return [
                MatchCluster(
                    subject_ref=result.subject_ref,
                    target_ref=result.target_ref,
                    score=result.score,
                    confidence=result.confidence,
                )
            ]
        return []

    def _associate(self, clusters: Sequence[MatchCluster]) -> List[Tuple[MatchCluster, Optional[_EventInstance]]]:
        """Pair each cluster with the event it belongs to, or with None for a new one.

        Four rules, in order of how much they claim:

        1. **A shared track id.** Unambiguous: the same track is the same object.
        2. **Overlap with the instance's last boxes.** Covers the ordinary case of a body
           carried by successive track ids.
        3. **Adoption.** An open event none of whose own tracks appear in this frame takes
           the next compatible cluster wherever it appears. This is what the rearm window
           is *for* — an obstruction that hides the subject and hands it back somewhere
           else must extend the open event, not start a second one — and rules 1 and 2
           both miss it, because the returning object shares neither ids nor position.

           The test is "its tracks are absent from this frame", not "it is CLOSING". An
           instance does not learn its subject is gone until after this frame's clusters
           have been placed, so on the very frame a dropout replaces one track id with
           another it is still ACTIVE — and keying on CLOSING let exactly that mint a
           second event, observed live as three overlapping `Maple Present` cards.

           Absent counts only *matched* tracks. The tracker does not predict motion, so a
           dropped track coasts on a frozen box for its whole TTL; counting that as the
           subject being present withholds the one instance the returning body could have
           adopted, for as long as the ghost outlives the gap.

           Bounded by coexistence as well as by group: a cluster whose tracks have shared
           a frame with the instance's is a second object, not its subject returning.
        4. **Nothing** — a genuinely new object, unless the machine is at its cap, in
           which case it joins its nearest neighbour rather than being dropped.

        An unmatched cluster carrying track ids is exempt from rule 4: it may feed an
        event it already belongs to, so a brief dropout does not start the rearm
        countdown, but it may never open one. A ghost whose event has moved on is the
        residue of a body already accounted for elsewhere in the frame, and minting an
        instance for it turns one dropout into a second card.

        A *trackless* cluster is not a ghost and keeps rule 4. It is what a `motion` rule
        produces — pixels and a zone, no track to be unmatched — and `MatchCluster.matched`
        is False there only because nothing ever absorbed a match into it.

        One instance per cluster per frame: a claimed instance is withdrawn from the pool
        so two objects can never collapse into one event by both matching it.
        """
        available = list(self._instances.values())
        present = {track_id for cluster in clusters if cluster.matched for track_id in cluster.track_ids}
        pairs: List[Tuple[MatchCluster, Optional[_EventInstance]]] = []
        # Matched clusters choose first. A coasting cluster still carries the open event's
        # track id, so left to frame order it claims that event by rule 1 and the re-found
        # body has nothing left to adopt.
        for cluster in sorted(clusters, key=lambda cluster: not cluster.matched):
            match = self._claim(cluster, available, present)
            if match is None and cluster.track_ids and not cluster.matched:
                continue
            if match is not None:
                available.remove(match)
            pairs.append((cluster, match))
        return pairs

    def _claim(
        self,
        cluster: MatchCluster,
        available: List[_EventInstance],
        present: Set[int],
    ) -> Optional[_EventInstance]:
        if not cluster.track_ids or self.spec.event_granularity != GRANULARITY_SUBJECT:
            # One cluster per frame by construction — a trackless `motion` rule has no
            # object to distinguish, and `rule` granularity collapses every match into
            # one — so the machine holds exactly one instance and every cluster is it.
            # Including a SPENT one: the cap latch must keep the seat, or capping an event
            # would immediately open its replacement.
            return available[0] if available else None

        for instance in available:
            if cluster.track_ids & instance.track_ids:
                return instance

        overlapping = [
            (_best_iou(cluster.boxes, i.boxes), i)
            for i in available
            if i.boxes and i.state is not RuleState.IDLE and not self._coexisted(cluster.track_ids, i.track_ids)
        ]
        best = max(overlapping, key=lambda pair: pair[0], default=(0.0, None))
        if best[0] >= self._cluster_iou and best[1] is not None:
            return best[1]

        # Adoption and the cap fold below are for live events only. A SPENT instance is
        # a latch, not an event: it may recognise its own subject (above) but must never
        # absorb a different one, or one animal's cap would swallow the next animal's
        # event entirely.
        live = [i for i in available if not i.spent]
        adoptable = [
            i
            for i in live
            if i.state in (RuleState.ACTIVE, RuleState.CLOSING)
            and not (i.track_ids & present)
            and self._compatible(cluster, i)
            and not self._coexisted(cluster.track_ids, i.track_ids)
        ]
        if adoptable:
            # Longest-waiting first, and an instance that lost its subject this very frame
            # (still ACTIVE, so no `false_since` yet) has waited least of all.
            return min(adoptable, key=lambda i: i.false_since if i.false_since is not None else float("inf"))

        if len(self._instances) >= MAX_EVENT_INSTANCES and live:
            return max(live, key=lambda i: _best_iou(cluster.boxes, i.boxes) if i.boxes else 0.0)
        return None

    def _compatible(self, cluster: MatchCluster, instance: _EventInstance) -> bool:
        """Whether a cluster may be adopted into an event opened on something else.

        Same `exclusive_variants` group only. The group means "alternate guesses about
        one object", so a cluster inside it is a plausible continuation of the same
        visit; anything outside it is a different animal, and a rearm window is not a
        licence to file it under the first one. An unknown label on either side is
        permitted rather than refused — refusing would make an unlabelled `any` rule
        unable to extend any event at all.

        Necessary, not sufficient. The group is a statement about one frame's labels, so
        two siblings that are genuinely two bodies — a Hatchback and a Sedan in one garage —
        both pass it; `_coexisted` is what separates those.
        """
        if cluster.subject_ref is None or instance.subject_ref is None:
            return True
        return self._group_of(cluster.subject_ref) == self._group_of(instance.subject_ref)

    # ------------------------------------------------------------------ step

    def step(self, result: PredicateResult, now: float) -> List[WatchEventSignal]:
        """Advance every instance by one frame; return any transitions to dispatch."""
        if not self.spec.active:
            self._watching_since = None
            return self._deactivate(now)
        if self._watching_since is None:
            self._watching_since = now

        signals: List[WatchEventSignal] = []
        matched: Set[str] = set()
        clusters = self._clusters(result)
        # Before association, so this frame's own verdict about what is distinct is
        # already on the record when adoption asks.
        self._record_coexistence(clusters)
        for cluster, instance in self._associate(clusters):
            if instance is None:
                instance = self._new_instance()
                instance.preexisting = cluster.first_seen <= self._watching_since
            matched.add(instance.instance_id)
            signals.extend(self._advance(instance, cluster, now))

        # Every instance the frame said nothing about sees a false predicate — which is
        # what starts, and eventually finishes, its rearm countdown.
        for instance in list(self._instances.values()):
            if instance.instance_id not in matched:
                signals.extend(self._advance(instance, None, now))

        self._evict()
        return signals

    def _deactivate(self, now: float) -> List[WatchEventSignal]:
        """Disabled or disarmed: close what is open, forget the rest.

        Forced, not predicate: the subject may still be right there, and the span stops
        short of whatever it was actually doing. Nothing open means nothing to close —
        that covers SPENT (already capped, and a second closing edge would be filed
        against a row that has one), PENDING (which never had an event) and an impulse
        machine mid-dwell (whose event does not exist yet; disarming a rule is not the
        subject leaving).
        """
        signals: List[WatchEventSignal] = []
        for instance in self._instances.values():
            if self.instance_is_open(instance):
                signals.append(self._signal(instance, EventPhase.END, now, EventCloseReason.FORCED))
            instance.reset()
        self._instances.clear()
        return signals

    def _advance(
        self,
        instance: _EventInstance,
        cluster: Optional[MatchCluster],
        now: float,
    ) -> List[WatchEventSignal]:
        signals: List[WatchEventSignal] = []
        spec = self.spec
        holds = cluster is not None

        # Waiting out a capped event. The subject is still there as far as this rule is
        # concerned, so nothing is measured and nothing is latched — the only question
        # is whether it has been gone long enough to allow a new event.
        if instance.state is RuleState.SPENT:
            if holds:
                assert cluster is not None
                # Keep following the subject while latched. A latch tracking a stale box
                # loses the object as soon as it moves, and then a new instance opens
                # beside it — the cap defeated by the very motion it was watching.
                self._disown_rival_tracks(instance, cluster)
                instance.observe(cluster, now, spec.max_duration_s)
                instance.false_since = None
                return signals
            if instance.false_since is None:
                instance.false_since = now
            # Not `elif`: a rule with no rearm delay is asking to be free again as soon
            # as the subject is gone, not one frame later.
            if now - instance.false_since >= spec.rearm_delay_s:
                instance.reset()
            return signals

        if holds:
            assert cluster is not None
            self._disown_rival_tracks(instance, cluster)
            instance.observe(cluster, now, spec.max_duration_s)
            instance.false_since = None

            if instance.state is RuleState.IDLE:
                instance.state = RuleState.PENDING
                # The event's start is when the predicate *first* held, not when
                # hysteresis confirmed it — pre-roll is measured back from here.
                instance.true_since = now
            elif instance.state is RuleState.PENDING:
                assert instance.true_since is not None
                if now - instance.true_since >= spec.min_duration_s:
                    instance.state = RuleState.ACTIVE
                    instance.started_at = instance.true_since
                    if spec.trigger == TRIGGER_ON_ENTER:
                        # The arrival itself, anchored at the moment the subject appeared
                        # rather than at the moment the dwell confirmed it — a clip's
                        # preroll reaches back from the anchor, and anchoring at
                        # confirmation would cut `min_duration` off the front. A subject
                        # that was already there when watching began did not appear, so
                        # it only takes the latch.
                        if not instance.preexisting:
                            signals.extend(self._impulse(instance, instance.true_since))
                        # Nothing is open, and the subject is still there. SPENT is
                        # exactly the right latch: no second arrival until it has gone.
                        self._spend(instance, false_since=None)
                    elif spec.trigger == TRIGGER_ON_EXIT:
                        # Armed, not open: the dwell is being served and the event, if
                        # there is ever one, happens when the subject leaves. No START is
                        # emitted, so nothing downstream sees an event it will never get
                        # a matching END for.
                        pass
                    else:
                        signals.append(self._signal(instance, EventPhase.START, now))
            elif instance.state is RuleState.CLOSING:
                # Re-fired inside the rearm window: rejoin the open event rather than
                # opening a second one. This is what stops one visit becoming twelve, and
                # with adoption it holds even when the subject comes back under new track
                # ids in a different part of the frame.
                instance.state = RuleState.ACTIVE
                self.flap_count += 1
        else:
            if instance.state is RuleState.PENDING:
                # Never reached min_duration — no event ever existed, so nothing to emit.
                instance.reset()
            elif instance.state is RuleState.ACTIVE:
                instance.state = RuleState.CLOSING
                instance.false_since = now
            elif instance.state is RuleState.CLOSING:
                assert instance.false_since is not None
                if now - instance.false_since >= spec.rearm_delay_s:
                    if spec.trigger == TRIGGER_ON_EXIT:
                        # The departure, anchored at the moment the subject stopped
                        # matching rather than at the end of the debounce. The delay
                        # decides whether it *really* left — a tracker dropping a car for
                        # two frames must not fire "car left" — it does not move when it
                        # left, and a preroll is measured back from the anchor.
                        signals.extend(self._impulse(instance, instance.false_since))
                    else:
                        signals.append(self._signal(instance, EventPhase.END, now))
                    instance.reset()

        # The runaway cap applies regardless of the predicate: a stuck detection would
        # otherwise hold an event — and a recording — open indefinitely.
        if (
            # ...but only to a rule that has a span to cap. An impulse has none, and an
            # `on_exit` instance sitting ACTIVE is serving a *dwell*, not holding an event
            # open: capping it would emit an END for an event that never started, and
            # would break the headline case outright — "in the garage for 15 minutes"
            # under the default 300 s cap could never fire.
            spec.trigger == TRIGGER_WHILE
            and instance.state in (RuleState.ACTIVE, RuleState.CLOSING)
            and instance.started_at is not None
            # Zero (or negative) means "no cap", the same reading the recording retention
            # settings give it. Without this the comparison is trivially true the instant
            # the event opens: `started_at` is `min_duration_s` in the past, so a zero cap
            # emits END in the *same step* as START, forever.
            and spec.max_duration_s > 0
            and now - instance.started_at >= spec.max_duration_s
        ):
            signals.append(self._signal(instance, EventPhase.END, now, EventCloseReason.CAP))
            # Deliberately *not* a reset to IDLE. The predicate that tripped the cap is
            # usually still true — that is what "stuck" means — and an IDLE instance would
            # re-open on the next frame and cap again one max_duration later, forever.
            # Capping out of CLOSING carries the countdown over rather than restarting it,
            # so a subject that has in fact left is released on the very next step.
            self._spend(instance, false_since=instance.false_since)

        return signals

    def _evict(self) -> None:
        """Forget instances with nothing left to remember.

        The old machine table was allocated once at load time and never grew, which is
        what made cardinality a non-question. An instance per object brings the question
        back, and this is the answer: an instance is dropped the moment it holds no
        state, so the table is bounded by what is actually in front of the camera. SPENT
        instances are deliberately kept — their whole purpose is a latch outliving the
        event.
        """
        for instance_id in [i.instance_id for i in self._instances.values() if i.retired]:
            del self._instances[instance_id]

    # ------------------------------------------------------------------ views

    @property
    def open_instances(self) -> List[_EventInstance]:
        return [i for i in self._instances.values() if self.instance_is_open(i)]

    @property
    def is_open(self) -> bool:
        """Whether *any* event is open right now — a START emitted with no END yet.

        Only a `while` rule can be. An impulse emits both edges in one step and is never
        between them, and an `on_exit` instance in ACTIVE is serving its dwell with no
        event to show for it yet. Getting this wrong would put a phantom row in the
        status panel's "Open events" table and, worse, make `force_close` emit an END on
        shutdown for an event that was never started.
        """
        return bool(self.open_instances)

    @property
    def state(self) -> RuleState:
        """The machine's most advanced instance state, for the status panel.

        A machine no longer *has* one state. Reported as the furthest along any instance
        has got, because that is what a reader of "is this rule doing anything" means.
        """
        order = [RuleState.IDLE, RuleState.SPENT, RuleState.PENDING, RuleState.CLOSING, RuleState.ACTIVE]
        return max((i.state for i in self._instances.values()), key=order.index, default=RuleState.IDLE)

    def open_signals(self, now: float) -> List[WatchEventSignal]:
        """Every open event as an UPDATE signal. Advances nothing.

        Each carries the same identity its START and END do — rule, source and instance —
        so a caller can find the event's row, or the buffer an action is filling for it,
        while it is still running.
        """
        return [self._signal(i, EventPhase.UPDATE, now) for i in self.open_instances]

    def force_close(
        self,
        now: float,
        reason: EventCloseReason = EventCloseReason.FORCED,
    ) -> List[WatchEventSignal]:
        """Close every open event without waiting for hysteresis.

        Used on shutdown and on resume: an event that was open when the task stopped must
        be closed and its clip enqueued, not left dangling.

        `reason` is the caller's, because the machine cannot tell the difference. The
        state it sees is identical whether the watcher is stopping or the camera has gone
        dark, and those read very differently on an event card.

        Instances that hold nothing open are dropped rather than closed: an impulse rule
        mid-dwell has no event, and a watcher stopping is not the subject leaving — an
        `on_exit` rule must not fire "the car left" because someone pressed Stop.
        """
        signals = [self._signal(i, EventPhase.END, now, reason) for i in self.open_instances]
        for instance in self._instances.values():
            instance.reset()
        self._instances.clear()
        return signals
