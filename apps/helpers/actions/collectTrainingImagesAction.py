"""Feed a watcher's own frames into a `TrainingCollector`.

Phase 4 of COLLECTOR_UNIFICATION_PLAN.md (repo root). The collector is
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
"""

import logging
import time
from typing import Any, Callable, Dict, Optional

from ..imageProviders.imageProvider import ImageProviderMetadata, ImageWithProviderMetadata
from ..training.collector import FrameOffer, TrainingCollector
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
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._collector_for = collector_for
        self._collector_uuid = collector_uuid
        self._min_interval_s = max(float(min_interval_s), self.MIN_INTERVAL_FLOOR_S)
        #: source_uuid -> monotonic time of the last offer, so the interval is
        #: per camera — a doorbell and a litter-box camera on the same watcher
        #: must not share one clock.
        self._last_offered: Dict[str, float] = {}

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

    async def run(self, signal, context: ActionContext) -> ActionResult:
        collector = self._collector_for(self._collector_uuid)
        if collector is None:
            return ActionResult(ok=False, detail=f"Collector {self._collector_uuid!r} is not attached")
        if context.full_frame is None or context.analysis_base64 is None:
            return ActionResult(ok=False, detail="No frame available")

        now = time.monotonic()
        key = context.source_uuid or ""
        if now - self._last_offered.get(key, 0.0) < self._min_interval_s:
            return ActionResult(ok=True, detail="waiting for interval")
        self._last_offered[key] = now

        frame = FrameOffer(
            provided=ImageWithProviderMetadata(
                image=context.full_frame,
                metadata=ImageProviderMetadata(
                    source=context.source_name,
                    source_uuid=context.source_uuid,
                    frame_idx=context.frame_seq,
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
