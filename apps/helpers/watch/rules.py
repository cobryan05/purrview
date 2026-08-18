"""Interaction rules and their per-rule state machine.

The state machine is the whole reason this module exists. A raw predicate like
`containment(cat, litterbox) > 0.6` crosses its threshold many times per second
because detection confidence and box geometry jitter. Dispatching on the raw
predicate means an MQTT storm and a dozen three-second clips for one litter-box
visit. Three guards fix that, and every rule carries all three:

* ``min_duration_s`` — the predicate must hold *continuously* before the event starts,
* ``rearm_delay_s`` — it must be *continuously false* before the event ends,
* ``max_duration_s`` — a hard cap, so a stuck detection can't record forever.

States::

    IDLE ──predicate true──> PENDING ──held min_duration──> ACTIVE
      ^                         │                              │
      │                    predicate false                predicate false
      │                         v                              v
      └───────────────────── IDLE                          CLOSING
                                                               │
                                          ┌────false for rearm_delay────> IDLE (END)
                                          └────true again────> ACTIVE (no new event)

The `CLOSING` state is what coalesces a flapping predicate into one event: a
re-fire inside the rearm window rejoins the event in progress rather than opening a
new one.
"""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from ..tracking.tracker import Track
from .events import EventPhase, WatchEventSignal
from .geometry import Zone, centroid, containment, iou, normalised_distance, overlap_ratio


class RuleState(str, Enum):
    IDLE = "idle"
    PENDING = "pending"  # predicate true, waiting out min_duration_s
    ACTIVE = "active"  # event open
    CLOSING = "closing"  # predicate false, waiting out rearm_delay_s


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
    relation: str = "contains"  # "present" | "iou" | "contains" | "overlaps" | "centroid_in" | "distance"
    subject_b_kind: str = "none"  # "label" | "zone" | "none"
    subject_b_refs: List[str] = field(default_factory=list)
    #: Whether each slot's refs were widened to include their label descendants. Recorded
    #: for reporting only — `expand_descendants` has already rewritten the ref lists by
    #: the time the engine sees a spec, so matching never consults these. The engine is
    #: pure and has no label tree to consult even if it wanted to.
    subject_a_include_descendants: bool = False
    subject_b_include_descendants: bool = False
    threshold: float = 0.5
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
            min_duration_s=get("min_duration_s", 2.0),
            rearm_delay_s=get("rearm_delay_s", 10.0),
            max_duration_s=get("max_duration_s", 300.0),
            enabled=get("enabled", True),
            event_label=get("event_label"),
            source_uuids=[str(s) for s in (get("source_uuids_json") or [])],
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
) -> PredicateResult:
    """Evaluate a rule against one frame's tracks. Pure — no state, no I/O."""
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
            if isinstance(target, Zone):
                if spec.relation == "centroid_in":
                    score = 1.0 if target.contains_point(centroid(track_a.bbox), frame_width, frame_height) else 0.0
                elif spec.relation == "distance":
                    # Distance is an inverted predicate: closer is "more true", so the
                    # threshold is an upper bound and the score is inverted to keep
                    # "higher score = closer to firing" true everywhere.
                    dist = normalised_distance(
                        track_a.bbox, target.bounds(frame_width, frame_height), frame_width, frame_height
                    )
                    score = 1.0 - dist if dist <= spec.threshold else 0.0
                elif spec.relation == "iou":
                    score = iou(track_a.bbox, target.bounds(frame_width, frame_height))
                elif spec.relation == "overlaps":
                    score = target.overlap_ratio(track_a.bbox, frame_width, frame_height)
                else:  # contains
                    score = target.overlap_fraction(track_a.bbox, frame_width, frame_height)
            else:
                if track_a.track_id == target.track_id:
                    continue  # an object never interacts with itself
                if spec.relation == "centroid_in":
                    cx, cy = centroid(track_a.bbox)
                    bx1, by1, bx2, by2 = target.bbox
                    score = 1.0 if (bx1 <= cx <= bx2 and by1 <= cy <= by2) else 0.0
                elif spec.relation == "distance":
                    dist = normalised_distance(track_a.bbox, target.bbox, frame_width, frame_height)
                    score = 1.0 - dist if dist <= spec.threshold else 0.0
                elif spec.relation == "iou":
                    score = iou(track_a.bbox, target.bbox)
                elif spec.relation == "overlaps":
                    score = overlap_ratio(track_a.bbox, target.bbox)
                else:  # contains
                    score = containment(track_a.bbox, target.bbox)

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
        #: Slot values from the strongest observation seen while this event was open.
        #: Latched alongside `_peak_confidence` rather than taken from the first or last
        #: frame: a rule spanning several cats should name the one it saw most clearly,
        #: not whichever happened to be in the frame when hysteresis expired.
        self._subject_ref: Optional[str] = None
        self._target_ref: Optional[str] = None
        #: How many times the predicate flapped while an event was open. Surfaced as a
        #: tuning hint — a rule that flaps a lot wants a longer rearm delay.
        self.flap_count: int = 0

    # ------------------------------------------------------------------ helpers

    def _signal(self, phase: EventPhase, now: float, truncated: bool = False) -> WatchEventSignal:
        return WatchEventSignal(
            rule_uuid=self.spec.uuid,
            rule_name=self.spec.name,
            phase=phase,
            at=now,
            started_at=self._event_started_at if self._event_started_at is not None else now,
            source_uuid=self.source_uuid,
            track_ids=list(self._track_ids),
            peak_confidence=self._peak_confidence,
            truncated=truncated,
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
        if not self.spec.enabled:
            if self.state != RuleState.IDLE:
                # Disabling a rule mid-event must close it, not strand it open forever.
                signals = [self._signal(EventPhase.END, now)]
                self._reset()
                return signals
            return []

        signals: List[WatchEventSignal] = []

        if result.holds:
            self._track_ids = sorted(set(self._track_ids) | set(result.track_ids))
            # Slot values move with the peak, so they always describe the same
            # observation the reported confidence came from.
            if result.confidence >= self._peak_confidence or self._subject_ref is None:
                if result.subject_ref is not None:
                    self._subject_ref = result.subject_ref
                if result.target_ref is not None:
                    self._target_ref = result.target_ref
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
            and now - self._event_started_at >= self.spec.max_duration_s
        ):
            signals.append(self._signal(EventPhase.END, now, truncated=True))
            self._reset()

        return signals

    def force_close(self, now: float) -> Optional[WatchEventSignal]:
        """Close an open event without waiting for hysteresis.

        Used on shutdown and on resume: an event that was open when the task stopped
        must be closed and its clip enqueued, not left dangling.
        """
        if self.state in (RuleState.ACTIVE, RuleState.CLOSING):
            signal = self._signal(EventPhase.END, now, truncated=True)
            self._reset()
            return signal
        self._reset()
        return None

    @property
    def is_open(self) -> bool:
        return self.state in (RuleState.ACTIVE, RuleState.CLOSING)
