"""The action that does nothing except make an event worth keeping.

Every other action is a side effect on the outside world and the `watch_events` row is
a consequence of it. This one is the row: it has no output, no parameters and no
failure mode. Its whole job is to answer "should this show up in the Events view?" with
yes, for an event whose other actions produce nothing to attach — or that has no other
actions at all.

Why an action rather than a checkbox on the rule: the answer is not a property of the
rule. "A cat is at the litter box" is one rule watched on five cameras, and it is worth
filing on the one pointed at the box and pure noise on the other four. That is exactly
the axis a watch action is already scoped on — rules × sources, empty-means-all — so
expressing it as an action gets the scoping for free and puts it in the same list as
the recording actions it substitutes for. A per-rule flag would need its own source
scope invented beside it.

The engine never needs `run()` to have been called to record the event: the decision is
read off the *configuration* at the opening edge (see `Action.logs_event` and
`StreamWatcherTask._event_is_logged`), because the row has to exist before an artifact
can hang off it. `run()` is therefore a no-op that reports what already happened, and
is what keeps the action visible in the watcher's action counters and status panel like
any other.
"""

import logging
from typing import Any, Dict

from ..watch.events import WatchEventSignal
from .action import Action, ActionContext, ActionResult
from .Registry import register_action

logger = logging.getLogger(__name__)


@register_action("save_event")
class SaveEventAction(Action):
    """Records the event in the Events view, with no other side effect."""

    @classmethod
    def display_name(cls) -> str:
        return "Save event"

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {}

    @classmethod
    def logs_event(cls) -> bool:
        return True

    async def run(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        return ActionResult(ok=True, detail="event logged")
