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
    #: Relay master path for this source, when it has one. None for ring/none tiers.
    relay_path: Optional[str] = None
    #: Whether this source actually has a MediaMTX-backed rewind buffer.
    buffered: bool = False
    #: Seconds of rewind actually configured, for clamping pre-roll at cut time.
    retention_s: float = 0.0
    #: Latest analysed frame, for snapshot actions. Numpy array or None.
    frame: Any = None
    #: Wallclock (epoch seconds) matching the event's monotonic timestamps. The buffer
    #: is addressed by wallclock, but rule timing uses a monotonic clock, so the task
    #: supplies the offset rather than every action re-deriving it.
    wallclock_offset: float = 0.0
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
