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

from typing import Dict, List, Optional, Sequence, Tuple

from ..tracking.tracker import Track
from .events import EventCloseReason, WatchEventSignal
from .geometry import Zone
from .motion import MotionField
from .rules import RuleMachine, RuleSpec, evaluate_predicate
from .scoping import rule_applies_to_source

#: Which machine a transition belongs to: the rule, plus the member of each subject slot
#: it was narrowed to. `None` in either position means that slot had nothing to
#: enumerate. Deliberately the same shape the recorder and the timelapse action key an
#: event on — `(source, rule, subject_ref, target_ref)` — minus the source, which an
#: engine already is.
SlotKey = Tuple[str, Optional[str], Optional[str]]


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
    ) -> None:
        self.source_uuid = source_uuid
        self._zones: Dict[str, Zone] = {z.uuid: z for z in zones}
        # One machine per *slot* — per member of each multi-ref subject — not per rule.
        # "Any of these three cats" is three independent events with their own hysteresis
        # and their own recordings; see `RuleSpec.narrowed` for why that line falls
        # between labels rather than between tracks.
        #
        # Enumerated once, here, because the ref lists are known at load time (descendants
        # were already widened by `expand_descendants` before the spec arrived). Nothing
        # is ever allocated per frame, so there is no cardinality to cap and no eviction
        # to get wrong: the table is exactly as big as the rules say it is.
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
            for a_ref in _slot_refs(rule.subject_a_kind, rule.subject_a_refs):
                for b_ref in _slot_refs(rule.subject_b_kind, rule.subject_b_refs):
                    self._machines[(rule.uuid, a_ref, b_ref)] = RuleMachine(
                        rule.narrowed(a_ref, b_ref), source_uuid=source_uuid
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
        signals = []
        for machine in self._machines.values():
            signal = machine.open_signal(now)
            if signal is not None:
                signals.append(signal)
        return signals

    def close_all(
        self,
        now: float,
        reason: EventCloseReason = EventCloseReason.FORCED,
    ) -> List[WatchEventSignal]:
        """Close every open event. Used on shutdown, pause, resume and a lost source."""
        signals = []
        for machine in self._machines.values():
            signal = machine.force_close(now, reason)
            if signal is not None:
                signals.append(signal)
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
                signal = machine.force_close(now, reason)
                if signal is not None:
                    signals.append(signal)
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

        The slot refs are carried because one rule can now hold several events at once —
        without them two of this rule's rows would be indistinguishable, and the panel
        would look like it was double-reporting rather than showing two cats.
        """
        return [
            {
                "ruleUuid": machine.spec.uuid,
                "ruleName": machine.spec.name,
                "sourceUuid": self.source_uuid,
                "subjectRef": subject_ref,
                "targetRef": target_ref,
                "state": machine.state.value,
                "startedAt": machine._event_started_at,
                "trackIds": list(machine._track_ids),
            }
            for (_rule_uuid, subject_ref, target_ref), machine in self._machines.items()
            if machine.is_open
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
