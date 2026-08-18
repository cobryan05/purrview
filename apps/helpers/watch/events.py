"""Event value types emitted by the rule engine."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventPhase(str, Enum):
    START = "start"
    UPDATE = "update"
    END = "end"


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
    #: True when the event was closed by `max_duration_s` rather than by the predicate
    #: going false. A stuck detection looks exactly like a very long event otherwise.
    truncated: bool = False
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
            "subjectRef": self.subject_ref,
            "targetRef": self.target_ref,
            "subjectName": self.subject_name,
            "targetName": self.target_name,
            "metadata": dict(self.metadata),
        }
