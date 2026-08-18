"""Action interface — the side effects an event triggers.

Actions are deliberately separate from filter-chain actions. Filter chain steps are
stateless, per-frame, per-detection, and their actions mutate an image's annotations.
These operate on *events* (stateful across time, opened and closed by rules) and their
effects are on the outside world: MQTT, files, recordings. Overloading one registry
with both would force every filter-chain consumer to handle actions it cannot execute.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

    @abstractmethod
    async def run(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        """Perform the side effect for one event transition.

        Must never raise: an action failure is reported and counted, but it must not be
        able to kill the frame loop that a 24/7 watcher depends on.
        """

    async def close(self) -> None:
        """Release any resources. Called from the task's `_deinit`."""
