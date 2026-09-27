"""The rule engine: a frame of tracks in, a stream of event transitions out.

Pure by construction — no DB, no HTTP, no clock of its own. Everything that touches
the outside world (inference, persistence, MQTT, clip extraction) lives above this in
`StreamWatcherTask`. That split is deliberate: the real bugs in this feature are
association and hysteresis mistakes, and they are only cheap to find when the logic
under test needs no services to run.

One engine per source. Rules are shared across a profile, but their *state* is
per-source: two cameras watching two litter boxes must open two independent events for
the same rule. A rule may also opt out of a source entirely (`RuleSpec.source_uuids`),
in which case this engine simply has no machine for it.
"""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..tracking.tracker import Track
from .events import EventCloseReason, WatchEventSignal
from .geometry import Zone
from .motion import MotionField
from .rules import TRIGGER_ON_ENTER, TRIGGER_ON_EXIT, RuleMachine, RuleSpec, evaluate_predicate
from .scoping import rule_applies_to_source

#: Which machine a transition belongs to: the rule, plus any slot member the rule was
#: narrowed to. `None` in a position means that slot is not split on — which, for the
#: subject slot, is now almost always the case: see `_machine_keys`.
#:
#: No longer the shape an event is keyed on. Events key on `instance_id`, because a
#: machine holds several at once.
SlotKey = Tuple[str, Optional[str], Optional[str]]


def _machine_keys(rule: RuleSpec) -> List[SlotKey]:
    """Which machines one rule needs.

    Almost always exactly one. A machine separates concurrent events *inside* itself now,
    by object, so splitting it by slot member would only re-create the bug that motivated
    all of this: a subject whose label changed, or that moved to the next target, landed
    in a different machine and so became a second event.

    Two slots still earn a machine each, for reasons that are not about identity:

    * **A `motion` rule's subject.** It names a region to measure, not a thing to look
      for. Two zones are two independent measurements and there is no object to cluster.
    * **An impulse rule's target.** Walking from the first bowl to the second genuinely
      *is* a second arrival, so `on_enter` must fire again. Only `while` rules want the
      target out of the key, where a subject moving between targets is one continuous
      thing happening.
    """
    if rule.relation == "motion":
        return [(rule.uuid, a_ref, None) for a_ref in _slot_refs(rule.subject_a_kind, rule.subject_a_refs)]
    if rule.trigger in (TRIGGER_ON_ENTER, TRIGGER_ON_EXIT):
        return [(rule.uuid, None, b_ref) for b_ref in _slot_refs(rule.subject_b_kind, rule.subject_b_refs)]
    return [(rule.uuid, None, None)]


def _slot_refs(kind: str, refs: Sequence[str]) -> List[Optional[str]]:
    """The members of one subject slot that each deserve their own machine.

    A slot with nothing to enumerate yields a single `None`, which `narrowed` reads as
    "leave this slot as it is". That covers `subject_b_kind="none"` (a `present` rule has
    no target), `subject_a_kind="any"` (no ref list exists to split on, so it keeps the
    one machine it has always had) and an empty list (a rule that matches nothing, which
    must still hold a machine so it can be reported rather than vanishing).
    """
    if kind not in ("label", "zone") or not refs:
        return [None]
    return list(refs)


class WatchEngine:
    """Evaluates a profile's rules against one source's tracks."""

    def __init__(
        self,
        rules: Sequence[RuleSpec],
        zones: Sequence[Zone] = (),
        source_uuid: Optional[str] = None,
        frame_width: float = 640.0,
        frame_height: float = 480.0,
        label_groups: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.source_uuid = source_uuid
        self._zones: Dict[str, Zone] = {z.uuid: z for z in zones}
        # Machines are still enumerated once, here — see `_machine_keys` for which slots
        # earn one. What is no longer fixed at load time is how many *events* a machine
        # can hold: it grows one per object in front of the camera and drops it again when
        # the object is gone, so cardinality is bounded by the scene rather than by the
        # config. `RuleMachine._evict` is where that bound is enforced.
        #
        # A rule scoped to other sources gets no machine here at all, rather than a
        # machine that is asked never to fire. That is the difference between "this
        # camera does not run this rule" and "this rule happens to be false" — the
        # status panel, the tuning hints and `close_all` all read the machine map, and a
        # dormant machine would show up in every one of them.
        self._machines: Dict[SlotKey, RuleMachine] = {}
        for rule in rules:
            if not rule_applies_to_source(rule.source_uuids, source_uuid):
                continue
            for key in _machine_keys(rule):
                _rule_uuid, a_ref, b_ref = key
                self._machines[key] = RuleMachine(
                    rule.narrowed(a_ref, b_ref),
                    source_uuid=source_uuid,
                    label_groups=label_groups,
                )
        self.frame_width = frame_width
        self.frame_height = frame_height

    # ------------------------------------------------------------------ config

    def set_frame_size(self, width: float, height: float) -> None:
        """Update the analysis frame size zones are scaled against.

        Called when the first frame arrives, since zones are stored normalised and the
        analysis resolution isn't known until then.
        """
        if width > 0 and height > 0:
            self.frame_width = float(width)
            self.frame_height = float(height)

    @property
    def zones(self) -> Dict[str, Zone]:
        return dict(self._zones)

    @property
    def rule_uuids(self) -> List[str]:
        """The rules this engine actually evaluates — the profile's, minus those scoped away.

        De-duplicated: a multi-ref rule owns several machines, and callers here are
        counting *rules* ("how many were scoped away?", "are they all motion rules?"),
        not slots.
        """
        return list(dict.fromkeys(rule_uuid for rule_uuid, _a, _b in self._machines))

    def trigger_for(self, rule_uuid: str) -> Optional[str]:
        """The `RuleTrigger` a rule was configured with, or None if it isn't scoped here.

        Every slot-key machine for a rule shares one spec, so the first match answers
        for all of them. Lets a caller downstream of a signal (which carries no trigger
        of its own — see `WatchEventSignal`) tell an `on_exit` impulse apart from an
        `on_enter`/`while` one without threading the trigger through every signal.
        """
        for (machine_rule_uuid, _a, _b), machine in self._machines.items():
            if machine_rule_uuid == rule_uuid:
                return machine.spec.trigger
        return None

    # ------------------------------------------------------------------ step

    def step(
        self,
        tracks: Sequence[Track],
        now: float,
        motion: Optional[MotionField] = None,
    ) -> List[WatchEventSignal]:
        """Advance every rule by one frame and collect the transitions.

        `motion` is this frame's pixel-difference measurement, or None for a caller
        that computes none. Only `motion` rules read it; passing None simply means
        those rules never hold, which is the right answer for a source that is not
        being differenced.
        """
        signals: List[WatchEventSignal] = []
        for machine in self._machines.values():
            result = evaluate_predicate(
                machine.spec,
                tracks,
                self._zones,
                self.frame_width,
                self.frame_height,
                motion=motion,
            )
            signals.extend(machine.step(result, now))
        return signals

    def open_signals(self, now: float) -> List[WatchEventSignal]:
        """One UPDATE signal per currently-open event. Advances nothing."""
        signals: List[WatchEventSignal] = []
        for machine in self._machines.values():
            signals.extend(machine.open_signals(now))
        return signals

    def close_all(
        self,
        now: float,
        reason: EventCloseReason = EventCloseReason.FORCED,
    ) -> List[WatchEventSignal]:
        """Close every open event. Used on shutdown, pause, resume and a lost source."""
        signals: List[WatchEventSignal] = []
        for machine in self._machines.values():
            signals.extend(machine.force_close(now, reason))
        return signals

    def set_rule_armed(
        self,
        rule_uuid: str,
        armed: bool,
        now: float,
        reason: EventCloseReason = EventCloseReason.FORCED,
    ) -> List[WatchEventSignal]:
        """Arm or disarm one rule at runtime, returning any events the change closed.

        The signals are returned rather than swallowed because disarming closes an open
        event *immediately* rather than on the rule's next frame. Waiting for the frame
        would be wrong twice over: a source that is down or simply not due is not stepped
        for minutes at a time, so "stop recording when everyone comes home" would keep
        recording until the camera next produced a frame — and a source that never comes
        back would strand the event entirely, which is the bug `close_all` exists to
        avoid on the shutdown path.

        Idempotent: re-arming an armed rule closes nothing, so a retained MQTT message
        replayed on every reconnect is free.

        Returns an empty list when arming, which cannot close anything by definition.
        """
        signals: List[WatchEventSignal] = []
        for (machine_rule_uuid, _a, _b), machine in self._machines.items():
            if machine_rule_uuid != rule_uuid or machine.spec.armed == armed:
                continue
            machine.spec.armed = armed
            if not armed:
                signals.extend(machine.force_close(now, reason))
        return signals

    def rule_arming(self) -> List[Tuple[str, str, bool]]:
        """`(uuid, name, armed)` per rule this engine runs, for the status panel.

        Read off the machines rather than off the profile rows, so the panel reports the
        state the watcher is actually running under rather than what something else
        believes. The two can differ for a pass — arming is polled, not pushed — and in
        that window the machines are the truthful answer.

        De-duplicated per rule like `rule_uuids`: a multi-ref rule owns a machine per
        slot, and they arm and disarm together.
        """
        rows: Dict[str, Tuple[str, str, bool]] = {}
        for (rule_uuid, _a, _b), machine in self._machines.items():
            rows.setdefault(rule_uuid, (rule_uuid, machine.spec.name, machine.spec.armed))
        return list(rows.values())

    # ------------------------------------------------------------------ status

    def open_events(self) -> List[Dict[str, object]]:
        """Open events, shaped for the status panel's "Open events" table.

        One row per open event, not per machine: a rule holds an event per object, and
        `instanceId` is what tells two of this rule's rows apart. The slot refs ride along
        for display only — they move as the detector changes its mind, which is exactly
        why they are no longer the identity.
        """
        return [
            {
                "ruleUuid": machine.spec.uuid,
                "ruleName": machine.spec.name,
                "sourceUuid": self.source_uuid,
                "instanceId": instance.instance_id,
                "subjectRef": instance.subject_ref,
                "targetRef": instance.target_ref,
                "state": instance.state.value,
                "startedAt": instance.started_at,
                "trackIds": sorted(instance.track_ids),
            }
            for machine in self._machines.values()
            for instance in machine.open_instances
        ]

    def tuning_hints(self, min_flaps: int = 5) -> List[str]:
        """Human-readable suggestions derived from observed flapping.

        This feature is almost entirely threshold tuning, and a rule that repeatedly
        re-fires inside its rearm window is the clearest signal that its guards are too
        tight — but it is invisible unless something says so.

        Totalled per rule rather than per machine: a multi-ref rule has one machine per
        cat, and a threshold that is too tight flaps on all of them. Reporting each
        separately would turn one piece of advice into three copies of itself, each with
        a count too small to cross the bar on its own.
        """
        flaps: Dict[str, int] = {}
        specs = {}
        for machine in self._machines.values():
            flaps[machine.spec.uuid] = flaps.get(machine.spec.uuid, 0) + machine.flap_count
            specs.setdefault(machine.spec.uuid, machine.spec)
        hints = []
        for rule_uuid, count in flaps.items():
            if count >= min_flaps:
                spec = specs[rule_uuid]
                hints.append(
                    f"Rule {spec.name!r} re-fired {count}× inside its "
                    f"rearm window ({spec.rearm_delay_s:g}s). Raising the rearm delay "
                    "or lowering the threshold would make it steadier."
                )
        return hints
