"""Action interface — the side effects an event triggers.

Actions are deliberately separate from filter-chain actions. Filter chain steps are
stateless, per-frame, per-detection, and their actions mutate an image's annotations.
These operate on *events* (stateful across time, opened and closed by rules) and their
effects are on the outside world: MQTT, files, recordings. Overloading one registry
with both would force every filter-chain consumer to handle actions it cannot execute.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..watch.events import WatchEventSignal


@dataclass
class ActionContext:
    """Everything an action needs that it must not fetch for itself.

    Injected rather than looked up so actions stay unit-testable and so the task keeps
    sole ownership of its API clients and DB writes.
    """

    profile_uuid: str
    profile_name: str
    source_uuid: Optional[str] = None
    source_name: Optional[str] = None
    #: This watcher's override for the name templates render `{source}` as. None means
    #: "use `source_name`". Deliberately does not replace `source_name`: logs, the
    #: status table, snapshot filenames and clip filenames should keep naming the camera
    #: the way the Sources pane does, so that a topic renamed for Home Assistant's
    #: benefit does not also rename the files on disk.
    source_mqtt_name: Optional[str] = None
    #: Relay master path for this source, when it has one. None for ring/none tiers.
    relay_path: Optional[str] = None
    #: Whether this source actually has a MediaMTX-backed rewind buffer.
    buffered: bool = False
    #: Seconds of rewind actually configured, for clamping pre-roll at cut time.
    retention_s: float = 0.0
    #: Latest analysed frame, for snapshot actions. Numpy array or None. This is the
    #: *downscaled* frame — the one inference ran on, and the one `detections` are in
    #: the pixel space of.
    frame: Any = None
    #: The same frame at the camera's own resolution, before the analysis downscale.
    #: None when the source never provided one. May be the identical array as `frame`
    #: when no downscale was needed, so an action must not write into it in place.
    #:
    #: ⚠️ `detections` are NOT in this frame's pixel space. Scale them by
    #: `full_frame.width / frame.width` before drawing, or every box lands in the
    #: top-left quadrant.
    full_frame: Any = None
    #: The detections on that frame, in its pixel space — mask-filtered, so this is what
    #: the tracker actually saw rather than raw model output. Carried beside the frame
    #: because an action that wants to draw them has no other way to get them: the event
    #: signal reports track ids, not boxes, and the pipeline is not reachable from here.
    detections: List[Any] = field(default_factory=list)
    #: label uuid → `#rrggbb`, so a drawn box matches the colour the label has
    #: everywhere else in the app. Resolved by the task, which owns the vocabulary.
    label_colours: Dict[str, str] = field(default_factory=dict)
    #: Wallclock (epoch seconds) matching the event's monotonic timestamps. The buffer
    #: is addressed by wallclock, but rule timing uses a monotonic clock, so the task
    #: supplies the offset rather than every action re-deriving it.
    wallclock_offset: float = 0.0
    #: `DrainedChunk`s already copied out of the rolling buffer for this event, oldest
    #: first. Non-empty only when the event outlived the buffer, which is what lets a
    #: rule's `max_duration_s` exceed the rewind window without silently losing the front
    #: of the clip. Supplied by the task, which owns the drain — see
    #: `apps.helpers.recordings.clipDrain`.
    clip_chunks: List[Any] = field(default_factory=list)
    #: This frame's ordinal in the pipeline's own counter
    #: (`WatchPipeline.frame_seq`). A training-collector action drives this into
    #: its saved image's provenance instead of a collector-local counter, so a
    #: watcher restart can't collide with a previous run's filenames.
    frame_seq: Optional[int] = None
    #: The exact base64 JSON packet (`{"format","data","metadata"}`) already sent
    #: to inference for this frame — what a `frame_analysed` action's own
    #: inference call, if it makes one, should reuse rather than re-encoding
    #: `full_frame` from scratch.
    analysis_base64: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    ok: bool
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


class Action(ABC):
    """Base class for event actions."""

    def __init__(self, **params: Any) -> None:
        self._params = params

    @classmethod
    def display_name(cls) -> str:
        return cls.__name__

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Rendered directly into the Watchers pane by the existing form machinery, so
        a new action kind needs no UI change."""
        return {}

    @classmethod
    def triggers(cls, row: Any) -> Set[str]:
        """Every trigger this action must be dispatched on, given its configured row.

        Almost every action is one side effect at one moment, and the default — the
        single trigger the user picked — is exactly right for those.

        An action that spans an event's life is not. `record_event_timelapse` has to
        open a buffer when the event starts, take a frame periodically for as long as it
        stays open, and compile at the close; three triggers, one action, one row of
        configuration. Expressing that as three rows would mean three sets of parameters
        the user has to keep in step, and nothing linking them.

        Takes the row rather than being a bare constant so an action can still *use* the
        configured trigger — an action might fire on whichever edge the user chose plus
        `frame_analysed`. `row.trigger` stays meaningful either way: it is the moment the
        action is understood to do its job, and it is what the Watchers pane shows.
        """
        return {getattr(row, "trigger", None)}

    @classmethod
    def logs_event(cls) -> bool:
        """Whether this action's presence should put the event in the Events view.

        A matched rule is not by itself a reason to write a `watch_events` row. Most
        rules match constantly — "a cat is present" is true for hours a day — and a
        watcher whose only action is an MQTT publish wants those matches *broadcast*,
        not filed. Filing them anyway is what filled the Events list with rows that had
        no clip, no timelapse and nothing to look at, burying the handful that did.

        So the row follows the artifact: an event is recorded when some action in scope
        for it puts something *in* it. Today that means a recording — see
        `WatchEventRecorder.attach_recording` — or the explicit `save_event`, which
        exists precisely so a user can say "note this one" for an event that produces no
        media of its own.

        Default False. Overriding it to True is a promise that the action is worth a row
        even when it fails: a clip action whose extraction failed still leaves the event
        recorded, because "the watcher tried to record this and could not" is exactly the
        thing a silent skip hid.

        Classmethod, not a row-dependent one like `triggers`: nothing an action row can
        configure changes whether its kind writes to the event log.

        This is the *configuration-time* answer — "can this kind file an event", which is
        what the editor's hint and the profile validator need. Whether one particular run
        actually filed one is `wrote_to_event`.
        """
        return False

    def holds_open_event(self, signal: "WatchEventSignal") -> bool:
        """Whether this action has already committed to filing media for a *still-open* event.

        The third question in the set, and the only one asked mid-event. `logs_event` is
        configuration ("can this kind file a row"), `wrote_to_event` is history ("did this
        run file one"), and this is a promise: the action is holding material it will
        certainly turn into a recording when the event closes.

        It exists because an event whose only action is a timelapse has no row until it
        ends — and an event timelapse of a cat asleep in a window runs for hours, which is
        exactly the event a user most wants to see listed while it is happening. Answering
        True brings the row forward to now, where the Events grid renders it "In progress".

        Deliberately *not* the same as "a session is open". A timelapse below its frame
        minimum compiles nothing, so a session that has just started is not yet a promise
        of anything; only one past that minimum is. Answering True too eagerly re-creates
        the empty-card bug `wrote_to_event` exists to prevent.

        Default False: an action that produces its artifact at one instant has nothing to
        promise beforehand, and must not bring a row forward.
        """
        return False

    def wrote_to_event(self, result: "ActionResult") -> bool:
        """Whether this run actually put something in the event.

        `logs_event` says a kind is capable of filing an event; this says a given run did
        it. The two are not the same, and treating them as the same is what produced
        event cards with nothing behind them: `record_event_timelapse` returns **ok** with
        no file when the visit was shorter than its frame minimum — a legitimate, expected
        outcome — and a clip can fail to cut. Both filed a card that pointed at no media.

        The default is "ran without failing", which is right for an action whose only
        product *is* the row. Anything that writes a file overrides this to require the
        file, because "it worked" and "it produced something" diverge for all of them.
        """
        return self.logs_event() and result.ok

    @abstractmethod
    async def run(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        """Perform the side effect for one event transition.

        Must never raise: an action failure is reported and counted, but it must not be
        able to kill the frame loop that a 24/7 watcher depends on.
        """

    async def close(self) -> None:
        """Release any resources. Called from the task's `_deinit`."""
