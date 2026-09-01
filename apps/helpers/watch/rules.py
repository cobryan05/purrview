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
"""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from ..tracking.tracker import Track
from .events import EventCloseReason, EventPhase, WatchEventSignal
from .geometry import Zone, centroid, containment, inflate, iou, margin_pad, normalised_distance, overlap_ratio
from .motion import DEFAULT_SENSITIVITY as MOTION_DEFAULT_SENSITIVITY
from .motion import MotionField


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
            min_duration_s=get("min_duration_s", 2.0),
            rearm_delay_s=get("rearm_delay_s", 10.0),
            max_duration_s=get("max_duration_s", 300.0),
            enabled=get("enabled", True),
            event_label=get("event_label"),
            source_uuids=[str(s) for s in (get("source_uuids_json") or [])],
        )

    def narrowed(self, subject_a_ref: Optional[str], subject_b_ref: Optional[str]) -> "RuleSpec":
        """This rule restricted to one member of each slot — the unit an event is per.

        A multi-ref rule is a convenience for the person writing it ("any of these three
        cats"), not a statement that the three should share one event. They should not:
        Two different cats are two visits, with their own spans, their own cards and
        their own recordings, and collapsing them into whichever was seen most clearly
        silently drops the other.

        So the engine gives every member its own machine, built from a spec narrowed to
        that member, and `evaluate_predicate` needs no knowledge of any of this — it is
        simply asked about one cat at a time.

        What it deliberately does *not* do is split by track. "Any Cat is present" stays
        one event however many cats are in shot and however often each comes and goes,
        because the recording is the whole frame either way: three tracks sharing a label
        would mean three near-identical videos of one scene. One event per *label* is the
        line, and it is also what makes a brief relabel harmless — the machine matches on
        `label_uuid` and never on track identity, so a box that flickers to another label
        and back rejoins its own event inside `rearm_delay_s`, while the label it briefly
        borrowed never clears `min_duration_s` and so never opens anything.

        `None` leaves a slot alone, which is how a rule with nothing to enumerate — a
        subject-less `present`, an `any` subject, a `motion` rule with no zones — keeps
        the single machine it has always had.
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
        return PredicateResult(
            holds=holds,
            track_ids=[t.track_id for t in subjects_a if t.confidence >= spec.threshold],
            score=best.confidence,
            confidence=best.confidence,
            subject_ref=best.label_uuid if holds else None,
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
                elif spec.relation == "iou":
                    # Bounding-box approximate for polygon zones, exactly as
                    # `Zone.bounds` documents — the margin does not change
                    # that, it only moves the bounds.
                    score = iou(box_a, box_b)
                elif spec.relation == "overlaps":
                    score = overlap_ratio(box_a, box_b)
                else:  # contains
                    score = containment(box_a, box_b)
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
    )


class RuleMachine:
    """Per-rule hysteresis state. One instance per (rule, source)."""

    def __init__(self, spec: RuleSpec, source_uuid: Optional[str] = None) -> None:
        self.spec = spec
        self.source_uuid = source_uuid
        self.state = RuleState.IDLE
        #: When the predicate most recently became true (the event's real start).
        self._true_since: Optional[float] = None
        #: When the predicate most recently became false (the rearm countdown's start).
        self._false_since: Optional[float] = None
        self._event_started_at: Optional[float] = None
        self._track_ids: List[int] = []
        self._peak_confidence: float = 0.0
        #: Slot values from the strongest observation seen *before* this event opened,
        #: and fixed from that moment on. Latched against `_peak_confidence` rather than
        #: taken from the first frame, so a rule spanning several cats still names the
        #: one it saw most clearly during the lead-in rather than whichever tripped
        #: `min_duration_s` — but pinned at the opening edge, because from there on
        #: these two values are the event's identity. See `step` for what relatching
        #: them mid-event cost.
        self._subject_ref: Optional[str] = None
        self._target_ref: Optional[str] = None
        #: How many times the predicate flapped while an event was open. Surfaced as a
        #: tuning hint — a rule that flaps a lot wants a longer rearm delay.
        self.flap_count: int = 0

    # ------------------------------------------------------------------ helpers

    def _signal(
        self,
        phase: EventPhase,
        now: float,
        close_reason: EventCloseReason = EventCloseReason.PREDICATE,
    ) -> WatchEventSignal:
        return WatchEventSignal(
            rule_uuid=self.spec.uuid,
            rule_name=self.spec.name,
            phase=phase,
            at=now,
            started_at=self._event_started_at if self._event_started_at is not None else now,
            source_uuid=self.source_uuid,
            track_ids=list(self._track_ids),
            peak_confidence=self._peak_confidence,
            close_reason=close_reason,
            subject_ref=self._subject_ref,
            target_ref=self._target_ref,
            metadata={"eventLabel": self.spec.event_label} if self.spec.event_label else {},
        )

    def _reset(self) -> None:
        self.state = RuleState.IDLE
        self._true_since = None
        self._false_since = None
        self._event_started_at = None
        self._track_ids = []
        self._peak_confidence = 0.0
        self._subject_ref = None
        self._target_ref = None

    # ------------------------------------------------------------------ step

    def step(self, result: PredicateResult, now: float) -> List[WatchEventSignal]:
        """Advance the machine one frame; return any transitions to dispatch."""
        if not self.spec.active:
            # SPENT holds no event, so there is nothing to close — emitting an END for it
            # would file a second closing edge against an event already capped.
            if self.state is RuleState.SPENT:
                self._reset()
                return []
            if self.state != RuleState.IDLE:
                # Disabling *or disarming* a rule mid-event must close it, not strand it
                # open forever. Forced, not predicate: the subject may still be right
                # there, and the span stops short of whatever it was actually doing.
                signals = [self._signal(EventPhase.END, now, EventCloseReason.FORCED)]
                self._reset()
                return signals
            return []

        signals: List[WatchEventSignal] = []

        # Waiting out a capped event. The subject is still there as far as this rule is
        # concerned, so nothing is measured and nothing is latched — the only question
        # is whether it has been gone long enough to allow a new event.
        if self.state is RuleState.SPENT:
            if result.holds:
                self._false_since = None
                return signals
            if self._false_since is None:
                self._false_since = now
            # Not `elif`: a rule with no rearm delay is asking to be free again as soon
            # as the subject is gone, not one frame later.
            if now - self._false_since >= self.spec.rearm_delay_s:
                self._reset()
            return signals

        if result.holds:
            self._track_ids = sorted(set(self._track_ids) | set(result.track_ids))
            # Slot values follow the peak only until the event opens. Once it is open
            # they are its *identity* and must not move under it: the recorder's
            # `_event_key`, the timelapse action's `_session_key`, the in-progress
            # preview endpoint and the MQTT template all hash an event on
            # (source, rule, subject, target).
            #
            # Nothing is lost by pinning them. The row records the slots once, at the
            # opening edge, and no later write revises them — so relatching mid-event
            # could never change what was stored, and only ever cost the closing edge
            # its key. A `present` rule spanning several cats would open on one of
            # them, meet a second more confident cat, and emit an END that hashed to a
            # tuple nothing had been filed under: the row never received `ended_at`
            # and sat "in progress" until a restart swept it, while the timelapse
            # action never found its buffer, so it neither compiled the video nor
            # released the frames — it kept sampling into a directory no closing edge
            # could reach.
            #
            # `is_open` covers CLOSING as well as ACTIVE, which is what keeps a flap
            # back inside the rearm window rejoining the same event rather than
            # quietly renaming it.
            if not self.is_open and (result.confidence >= self._peak_confidence or self._subject_ref is None):
                if result.subject_ref is not None:
                    self._subject_ref = result.subject_ref
                if result.target_ref is not None:
                    self._target_ref = result.target_ref
            # Not pinned: the reported confidence is a measurement of the event, not a
            # key into it, and it should still report the best look it ever got.
            self._peak_confidence = max(self._peak_confidence, result.confidence)
            self._false_since = None

            if self.state is RuleState.IDLE:
                self.state = RuleState.PENDING
                # The event's start is when the predicate *first* held, not when
                # hysteresis confirmed it — pre-roll is measured back from here.
                self._true_since = now
            elif self.state is RuleState.PENDING:
                assert self._true_since is not None
                if now - self._true_since >= self.spec.min_duration_s:
                    self.state = RuleState.ACTIVE
                    self._event_started_at = self._true_since
                    signals.append(self._signal(EventPhase.START, now))
            elif self.state is RuleState.CLOSING:
                # Re-fired inside the rearm window: rejoin the open event rather than
                # opening a second one. This is what stops one visit becoming twelve.
                self.state = RuleState.ACTIVE
                self.flap_count += 1
        else:
            if self.state is RuleState.PENDING:
                # Never reached min_duration — no event ever existed, so nothing to emit.
                self._reset()
            elif self.state is RuleState.ACTIVE:
                self.state = RuleState.CLOSING
                self._false_since = now
            elif self.state is RuleState.CLOSING:
                assert self._false_since is not None
                if now - self._false_since >= self.spec.rearm_delay_s:
                    signals.append(self._signal(EventPhase.END, now))
                    self._reset()

        # The runaway cap applies regardless of the predicate: a stuck detection would
        # otherwise hold an event — and a recording — open indefinitely.
        if (
            self.state in (RuleState.ACTIVE, RuleState.CLOSING)
            and self._event_started_at is not None
            # Zero (or negative) means "no cap", the same reading the recording retention
            # settings give it. Without this the comparison is trivially true the instant
            # the event opens: `_event_started_at` is `min_duration_s` in the past, so a
            # zero cap emits END in the *same step* as START, forever. That is the exact
            # opposite of what anyone typing 0 into a "max" field intends, and it fails
            # silently — an endless churn of zero-length events, and a timelapse that
            # compiles nothing because no event ever holds a frame.
            and self.spec.max_duration_s > 0
            and now - self._event_started_at >= self.spec.max_duration_s
        ):
            signals.append(self._signal(EventPhase.END, now, EventCloseReason.CAP))
            # Deliberately *not* a reset to IDLE. The predicate that tripped the cap is
            # usually still true — that is what "stuck" means — and an IDLE machine would
            # re-open on the next frame and cap again one max_duration later, forever.
            # Capping out of CLOSING goes the same way rather than straight to IDLE: the
            # countdown it was already serving carries over, so a subject that has in
            # fact left is released on the very next step and nothing waits twice.
            false_since = self._false_since
            self._reset()
            self.state = RuleState.SPENT
            self._false_since = false_since

        return signals

    def open_signal(self, now: float) -> Optional[WatchEventSignal]:
        """This machine's open event as an UPDATE signal, or None when nothing is open.

        Carries the same identity the START and END signals do — rule, source and both
        matched slots — so a caller can find the event's row, or the buffer an action is
        filling for it, while it is still running. Emits nothing itself: it is a view of
        the current state, not a transition, and `step` remains the only thing that
        advances the machine.
        """
        if not self.is_open:
            return None
        return self._signal(EventPhase.UPDATE, now)

    def force_close(
        self,
        now: float,
        reason: EventCloseReason = EventCloseReason.FORCED,
    ) -> Optional[WatchEventSignal]:
        """Close an open event without waiting for hysteresis.

        Used on shutdown and on resume: an event that was open when the task stopped
        must be closed and its clip enqueued, not left dangling.

        `reason` is the caller's, because the machine cannot tell the difference. The
        state it sees is identical whether the watcher is stopping or the camera has
        gone dark, and those read very differently on an event card — see
        `EventCloseReason`.
        """
        if self.state in (RuleState.ACTIVE, RuleState.CLOSING):
            signal = self._signal(EventPhase.END, now, reason)
            self._reset()
            return signal
        self._reset()
        return None

    @property
    def is_open(self) -> bool:
        return self.state in (RuleState.ACTIVE, RuleState.CLOSING)
