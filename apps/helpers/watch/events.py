"""Event value types emitted by the rule engine."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventPhase(str, Enum):
    START = "start"
    UPDATE = "update"
    END = "end"


class EventCloseReason(str, Enum):
    """Why an event ended. Only meaningful on an END signal.

    These used to be one boolean (`truncated`), which read as "the runaway cap fired".
    It did not distinguish the cap from a watcher that was simply stopped — so saving a
    watcher's config, which restarts the task and force-closes everything open, reported
    every interrupted event as capped. That is a diagnosis pointing at a setting the
    rule may not even have enabled (`max_duration_s = 0` disables the cap outright), and
    it cost a real debugging session.
    """

    #: The predicate went false and the rearm delay elapsed — the ordinary ending.
    PREDICATE = "predicate"
    #: `max_duration_s` elapsed while the predicate still held.
    CAP = "cap"
    #: The watcher stopped, paused or restarted with the event open. Says nothing about
    #: the subject: it may well still have been there.
    FORCED = "forced"


@dataclass
class WatchEventSignal:
    """One transition in a rule's life, as the engine reports it.

    Deliberately not a DB row: the engine is pure and knows nothing about persistence.
    The task turns these into `watch_events` rows and action dispatches.
    """

    rule_uuid: str
    rule_name: str
    phase: EventPhase
    #: Wallclock seconds of the transition.
    at: float
    #: When the *event* began — equal to `at` on START, earlier on UPDATE/END. This is
    #: what a clip's pre-roll is measured back from, so it must be the moment the
    #: predicate first held continuously, not the moment hysteresis confirmed it.
    started_at: float
    source_uuid: Optional[str] = None
    track_ids: List[int] = field(default_factory=list)
    peak_confidence: float = 0.0
    #: Why the event ended. `truncated` is derived from this rather than stored beside
    #: it, so the flag and the reason cannot disagree.
    close_reason: EventCloseReason = EventCloseReason.PREDICATE
    #: Which member of each multi-ref slot actually matched — a label uuid for the
    #: subject, a zone uuid (or label uuid) for the target. Uuids, not names: the engine
    #: is pure and has no vocabulary to resolve them with. The task that owns the DB
    #: turns these into the human names an action's template interpolates.
    subject_ref: Optional[str] = None
    target_ref: Optional[str] = None
    #: Resolved display names for the two slots, filled in above the engine. Kept beside
    #: the uuids rather than replacing them so a consumer can key on a stable id while a
    #: template prints something a human recognises.
    subject_name: Optional[str] = None
    target_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.at - self.started_at)

    @property
    def truncated(self) -> bool:
        """The event was cut short — by the cap or by the watcher stopping.

        Kept because everything downstream asks "is this span the whole story?" and the
        answer is no in both cases; `close_reason` is what says *which*.
        """
        return self.close_reason is not EventCloseReason.PREDICATE

    def to_dict(self) -> dict:
        return {
            "ruleUuid": self.rule_uuid,
            "ruleName": self.rule_name,
            "phase": self.phase.value,
            "at": self.at,
            "startedAt": self.started_at,
            "durationS": round(self.duration_s, 3),
            "sourceUuid": self.source_uuid,
            "trackIds": list(self.track_ids),
            "peakConfidence": round(self.peak_confidence, 4),
            "truncated": self.truncated,
            "closeReason": self.close_reason.value,
            "subjectRef": self.subject_ref,
            "targetRef": self.target_ref,
            "subjectName": self.subject_name,
            "targetName": self.target_name,
            "metadata": dict(self.metadata),
        }
