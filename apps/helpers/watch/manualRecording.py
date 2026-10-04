"""A recording the user asked for directly, with no rule behind it.

Modelled as an ordinary event rather than as a parallel recording path. The watcher
already knows how to open an event, hold the frame ring and the drain open across it,
cut a clip at its closing edge, burn the overlay and let the retention sweep reclaim
it; a manual recording wants every one of those and differs only in what opens and
closes it. So it emits the same `WatchEventSignal` the engine does, under a sentinel
rule uuid, and `_dispatch` cannot tell the difference.

That is also what makes a manual recording survive the things that are easy to forget:
a watcher restart force-closes it like any open event, `close_open_events` cuts
whatever it had rather than losing the file, and the Events view lists it beside the
rule-driven ones with no new query.

The sentinel uuid is not a rule that exists. Nothing may look it up — `rule_uuid` is
denormalised onto `watch_events` precisely so a row outlives the rule it names, and the
Events filter builds its dropdown from the uuids present in history, so the manual
recordings group themselves under one entry for free.

**The cap is a stop, not a ceiling on a prediction.** Unlike a rule's `max_duration_s`,
which bounds an event whose end nobody chose, this one exists because a manual
recording has no predicate to go false: without it a user who closes the tab records
until the disk fills. It is therefore enforced on the loop pass, not inside `step()` —
a source that stops delivering frames must still stop recording.
"""

from dataclasses import dataclass, field
from typing import Optional

#: Stands in for `rule_uuid` on a manually started recording. Deliberately not a uuid:
#: it must never collide with a real rule, and it reads as itself in a log line.
MANUAL_RULE_UUID = "manual"

#: Denormalised onto the event row as `rule_name`, which is what the Events view and
#: the clip's filename show. This is the "manual recording" tag.
MANUAL_RULE_NAME = "Manual recording"

#: Used when a profile has no `default_manual_recording_s` of its own.
DEFAULT_MAX_DURATION_S = 300.0

#: Floor on the cap. A recording shorter than this cannot produce a clip worth cutting —
#: the drain stays a segment behind what MediaMTX is still writing (`SEGMENT_LAG_S`) —
#: so accepting one would hand back an empty file and call it success.
MIN_MAX_DURATION_S = 10.0


@dataclass
class ManualRecording:
    """One in-flight manual recording on one source.

    `started_at` is monotonic, like `WatchEventSignal.at`, because it is compared
    against `time.monotonic()` on every loop pass and a wallclock jump must not end a
    recording early or hold one open forever.
    """

    source_uuid: str
    instance_id: str
    started_at: float
    max_duration_s: float
    #: Set by `stop_recording`; drained on the next loop pass for the same reason
    #: `_live_requests` is — `handle_command` may not do I/O.
    stop_requested: bool = field(default=False)

    def elapsed_s(self, now: float) -> float:
        return max(0.0, now - self.started_at)

    def remaining_s(self, now: float) -> float:
        return max(0.0, self.max_duration_s - self.elapsed_s(now))

    def expired(self, now: float) -> bool:
        return self.elapsed_s(now) >= self.max_duration_s


def clamp_max_duration(requested: Optional[float], default_s: float) -> float:
    """The cap to enforce, given what a caller asked for and the profile's default.

    A caller may omit the duration entirely (the UI button, the MQTT switch) or name one
    (the `purrview.start_recording` service), and an out-of-range number is clamped
    rather than refused: a Home Assistant automation that asks for five seconds wants a
    short recording, not an error it will never see.
    """
    value = float(requested) if requested else float(default_s)
    if value <= 0:
        value = float(default_s)
    return max(MIN_MAX_DURATION_S, value)
