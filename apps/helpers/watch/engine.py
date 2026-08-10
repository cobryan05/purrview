"""The rule engine: a frame of tracks in, a stream of event transitions out.

Pure by construction — no DB, no HTTP, no clock of its own. Everything that touches
the outside world (inference, persistence, MQTT, clip extraction) lives above this in
`StreamWatcherTask`. That split is deliberate: the real bugs in this feature are
association and hysteresis mistakes, and they are only cheap to find when the logic
under test needs no services to run.

One engine per source. Rules are shared across a profile, but their *state* is
per-source: two cameras watching two litter boxes must open two independent events for
the same rule.
"""

from typing import Dict, List, Optional, Sequence

from ..tracking.tracker import Track
from .events import WatchEventSignal
from .geometry import Zone
from .rules import RuleMachine, RuleSpec, evaluate_predicate


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
        self._machines: Dict[str, RuleMachine] = {
            rule.uuid: RuleMachine(rule, source_uuid=source_uuid) for rule in rules
        }
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

    # ------------------------------------------------------------------ step

    def step(self, tracks: Sequence[Track], now: float) -> List[WatchEventSignal]:
        """Advance every rule by one frame and collect the transitions."""
        signals: List[WatchEventSignal] = []
        for machine in self._machines.values():
            result = evaluate_predicate(
                machine.spec,
                tracks,
                self._zones,
                self.frame_width,
                self.frame_height,
            )
            signals.extend(machine.step(result, now))
        return signals

    def close_all(self, now: float) -> List[WatchEventSignal]:
        """Close every open event. Used on shutdown, pause and resume."""
        signals = []
        for machine in self._machines.values():
            signal = machine.force_close(now)
            if signal is not None:
                signals.append(signal)
        return signals

    # ------------------------------------------------------------------ status

    def open_events(self) -> List[Dict[str, object]]:
        """Open events, shaped for the status panel's "Open events" table."""
        return [
            {
                "ruleUuid": machine.spec.uuid,
                "ruleName": machine.spec.name,
                "sourceUuid": self.source_uuid,
                "state": machine.state.value,
                "startedAt": machine._event_started_at,
                "trackIds": list(machine._track_ids),
            }
            for machine in self._machines.values()
            if machine.is_open
        ]

    def tuning_hints(self, min_flaps: int = 5) -> List[str]:
        """Human-readable suggestions derived from observed flapping.

        This feature is almost entirely threshold tuning, and a rule that repeatedly
        re-fires inside its rearm window is the clearest signal that its guards are too
        tight — but it is invisible unless something says so.
        """
        hints = []
        for machine in self._machines.values():
            if machine.flap_count >= min_flaps:
                hints.append(
                    f"Rule {machine.spec.name!r} re-fired {machine.flap_count}× inside its "
                    f"rearm window ({machine.spec.rearm_delay_s:g}s). Raising the rearm delay "
                    "or lowering the threshold would make it steadier."
                )
        return hints
