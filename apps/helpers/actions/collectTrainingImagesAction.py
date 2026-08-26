"""Feed a watcher's own frames into a `TrainingCollector`.

Phases 4-5 of COLLECTOR_UNIFICATION_PLAN.md (repo root). The collector is
driver-agnostic by construction (`CollectorHost` — see
`apps.helpers.training.collector`), so attaching it here needs no change to
the collector itself: this action just calls `offer()` at watcher cadence
instead of the standalone task's fetch loop calling it at provider cadence.

The collector instance is memoised by `collector_uuid` on the driving task
(`StreamWatcherTask._collector_for`) — two action rows naming the same
collector, scoped to different sources, share one dedup index and one output
directory rather than deduplicating against nothing.

`min_interval_s` is the design's primary rate control, not a safety valve: a
collector is meant to sample every 5-10s, not see every analysed frame. The
floor is enforced here (clamped to 1.0), not merely defaulted — an unbounded
watcher offering every frame would spend its own CPU on phashes and JPEG
writes for training data that is ~95% near-duplicate anyway.

## Two modes, one action kind

**Continuous** (row's own trigger is `frame_analysed`): offers whenever the
interval has elapsed, unconditionally — phase 4's behaviour, unchanged.

**Event-scoped** (row's trigger is `event_started`/`event_ended`): "collect
training data only while this rule's event is open" — impossible to express
by running the standalone task alongside the watcher, and the strongest
argument for attaching rather than co-running (plan §2.4). Mirrors
`RecordEventTimelapseAction`'s shape (an action whose configured trigger is
one event edge but which also needs every analysed frame while a session is
open): `triggers()` adds `frame_analysed` on top of the row's own edge, and
`run()` branches on whether the dispatched signal carries a rule at all.

A **refcount**, not a boolean, tracks "is a session open" per source: a
source can have more than one in-scope rule open at once (the row's own
`rule_uuids_json` already filters which `event_started`/`event_ended`
transitions reach this action), and collection must continue until every
one of them has closed, not just the first.

## Provenance

An event-scoped offer stamps `watch_rule_uuid` / `watch_event_started_at`
into `ImageProviderMetadata.additional_info` — an existing pass-through
field `TrainingCollector._save_image` folds into every saved box's
`bbox_meta`. This is deliberately NOT the real `watch_events.uuid`: that row
(if one ever gets written at all — this action doesn't `logs_event()`) isn't
resolved until after every action for a transition has run
(`StreamWatcherTask._dispatch`), so it can't be known at collection time.
Rule + start time is what's actually reachable, and is enough to correlate
a collected image back to the window it came from in review.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..imageProviders.imageProvider import ImageProviderMetadata, ImageWithProviderMetadata
from ..training.collector import FrameOffer, TrainingCollector
from ..watch.events import EventPhase, WatchEventSignal
from ..watch.scoping import EVENT_ENDED, EVENT_STARTED, FRAME_ANALYSED
from .action import Action, ActionContext, ActionResult
from .Registry import register_action

logger = logging.getLogger(__file__)


@register_action("collect_training_images")
class CollectTrainingImagesAction(Action):
    """Offers analysed frames to an attached training collector, rate-limited per source."""

    #: Floor on min_interval_s. Sub-second sampling is what the standalone task's
    #: own snapshot/stream intervals are for; this action exists for occasional
    #: sampling out of a live feed, not a second collection loop running at
    #: analysis fps.
    MIN_INTERVAL_FLOOR_S: float = 1.0

    def __init__(
        self,
        collector_for: Callable[[str], Optional[TrainingCollector]],
        collector_uuid: str = "",
        min_interval_s: float = 10.0,
        trigger: str = FRAME_ANALYSED,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._collector_for = collector_for
        self._collector_uuid = collector_uuid
        self._min_interval_s = max(float(min_interval_s), self.MIN_INTERVAL_FLOOR_S)
        # Decided once, from the row's own configured trigger, rather than asked
        # per frame — see the module docstring's "Two modes" section.
        self._continuous = trigger == FRAME_ANALYSED
        #: source_uuid -> monotonic time of the last offer, so the interval is
        #: per camera — a doorbell and a litter-box camera on the same watcher
        #: must not share one clock.
        self._last_offered: Dict[str, float] = {}
        #: source_uuid -> stack of (rule_uuid, event started_at) for the
        #: currently-open in-scope rule events, oldest first. A stack, not a
        #: bool: two in-scope rules opening close together on one source must
        #: not let the first one's END switch collection off while the second
        #: is still open. The top of the stack supplies provenance metadata —
        #: last-opened wins while several overlap, but popping the ended rule
        #: (rather than only decrementing a count) means a closed event never
        #: keeps stamping its uuid onto a still-open one's frames.
        self._open_sessions: Dict[str, List[Tuple[str, float]]] = {}

    @classmethod
    def triggers(cls, row: Any) -> Set[str]:
        if getattr(row, "trigger", None) == FRAME_ANALYSED:
            return {FRAME_ANALYSED}
        # Event-scoped: the row's own edge is when the Watchers pane says this
        # action "runs", but frame_analysed is how it actually samples while
        # that edge's session stays open.
        return {EVENT_STARTED, FRAME_ANALYSED, EVENT_ENDED}

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "collector_uuid": {
                "type": "str",
                "required": True,
                "help": "Collector to feed — see Configuration → Collectors for the uuid.",
            },
            "min_interval_s": {
                "type": "float",
                "required": False,
                "default": 10.0,
                "help": "Minimum seconds between offers, per source. Enforced floor: 1.0.",
            },
        }

    async def run(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        # A frame_analysed dispatch carries no rule — see _dispatch_frame's own
        # docstring for why an empty rule_uuid is how a frame signal is told
        # apart from a real event transition.
        if not signal.rule_uuid:
            return await self._on_frame(context)
        if signal.phase is EventPhase.START:
            return self._on_event_start(signal)
        if signal.phase is EventPhase.END:
            return self._on_event_end(signal)
        # UPDATE on a real rule re-fires event_started for actions that want a
        # repeated nudge; there is nothing to reopen, the session already is.
        return ActionResult(ok=True, detail="already collecting")

    def _on_event_start(self, signal: WatchEventSignal) -> ActionResult:
        key = signal.source_uuid or ""
        self._open_sessions.setdefault(key, []).append((signal.rule_uuid, signal.started_at))
        return ActionResult(ok=True)

    def _on_event_end(self, signal: WatchEventSignal) -> ActionResult:
        key = signal.source_uuid or ""
        stack = self._open_sessions.get(key) or []
        # Pop the entry this END belongs to, not simply the top: rule B can end
        # while rule A is still open, and A's frames must keep A's provenance.
        for idx in range(len(stack) - 1, -1, -1):
            if stack[idx][0] == signal.rule_uuid:
                stack.pop(idx)
                break
        else:
            if stack:
                stack.pop()
        return ActionResult(ok=True)

    async def _on_frame(self, context: ActionContext) -> ActionResult:
        key = context.source_uuid or ""
        if not self._continuous and not self._open_sessions.get(key):
            return ActionResult(ok=True, detail="no session open")

        collector = self._collector_for(self._collector_uuid)
        if collector is None:
            return ActionResult(ok=False, detail=f"Collector {self._collector_uuid!r} is not attached")
        if context.full_frame is None or context.analysis_base64 is None:
            return ActionResult(ok=False, detail="No frame available")

        now = time.monotonic()
        if now - self._last_offered.get(key, 0.0) < self._min_interval_s:
            return ActionResult(ok=True, detail="waiting for interval")
        self._last_offered[key] = now

        additional_info: Dict[str, Any] = {}
        stack = self._open_sessions.get(key) or []
        if not self._continuous and stack:
            rule_uuid, started_at = stack[-1]
            additional_info = {"watch_rule_uuid": rule_uuid, "watch_event_started_at": str(started_at)}

        frame = FrameOffer(
            provided=ImageWithProviderMetadata(
                image=context.full_frame,
                metadata=ImageProviderMetadata(
                    source=context.source_name,
                    source_uuid=context.source_uuid,
                    frame_idx=context.frame_seq,
                    additional_info=additional_info,
                ),
            ),
            base64=context.analysis_base64,
        )
        try:
            await collector.offer(frame)
        except Exception as e:
            logger.exception("collect_training_images offer failed for collector %s", self._collector_uuid)
            return ActionResult(ok=False, detail=f"Offer failed: {e}")
        return ActionResult(ok=True)
