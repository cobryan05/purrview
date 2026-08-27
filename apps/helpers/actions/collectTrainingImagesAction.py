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

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from ..db.db_client import DETECTOR_MISS_TAG_UUID
from ..imageProviders.imageProvider import ImageProviderMetadata, ImageWithProviderMetadata
from ..imageUtils import base64_encode_jpg, downscale_to_width
from ..watch.events import EventPhase, WatchEventSignal
from ..watch.scoping import EVENT_ENDED, EVENT_STARTED, FRAME_ANALYSED
from .action import Action, ActionContext, ActionResult
from .Registry import register_action

if TYPE_CHECKING:
    # Deferred: `db_server` eagerly imports every `*Action.py` (see
    # `actions/__init__.py`), and `..training.collector` imports the generated
    # `db_client`/`inference_client` packages at module scope — which don't
    # exist yet while those very clients are being generated from db_server's
    # own OpenAPI spec (`generate_openapi_modules.sh`). Real imports happen
    # lazily, at the two call sites that actually construct these types.
    from ..training.collector import FrameOffer, TrackedCrop, TrainingCollector

logger = logging.getLogger(__file__)


@dataclass
class _GapState:
    """One track's currently-open detector-miss gap (plan §4A).

    `held_frame` is the frame from the FIRST coasting step of the gap — the
    plan's "cap at one candidate per gap episode" — kept in memory only until
    the gap resolves (a few frames, at most `MAX_GAP_FRAMES`). `before` is that
    same step's `TrackedCrop`, i.e. the track's last real position, which is
    one endpoint of the eventual interpolation.
    """

    held_frame: FrameOffer
    before: TrackedCrop
    frame_count: int = 1
    ambiguous: bool = False


@dataclass
class _DetectorMissBBox:
    """Duck-types the `.bbox` shape `TrainingCollector._save_image` reads off a
    real detection (`x/y/width/height/label_uuid/class_str/tag_uuids/uuid`) —
    see `_run_inference`'s detections for the real thing this stands in for."""

    x: float
    y: float
    width: float
    height: float
    label_uuid: Optional[str]
    class_str: Optional[str] = None
    tag_uuids: List[str] = field(default_factory=list)
    uuid: str = ""


@dataclass
class _DetectorMissDetection:
    """Duck-types a real detection well enough for `_save_image`: `.bbox` plus
    `.confidence` (deliberately `None` — plan §5: no model score backs this
    box, and a synthetic one would let it sort/filter alongside real
    detections while lying about where it came from)."""

    bbox: _DetectorMissBBox
    confidence: Optional[float] = None


@register_action("collect_training_images")
class CollectTrainingImagesAction(Action):
    """Offers analysed frames to an attached training collector, rate-limited per source."""

    #: Floor on min_interval_s. Sub-second sampling is what the standalone task's
    #: own snapshot/stream intervals are for; this action exists for occasional
    #: sampling out of a live feed, not a second collection loop running at
    #: analysis fps.
    MIN_INTERVAL_FLOOR_S: float = 1.0

    #: Past this many consecutive coasting frames, the tracker's box has
    #: drifted too far from the truth to be worth a human's time (plan §4A) —
    #: the gap is abandoned rather than committed on eventual re-acquisition.
    MAX_GAP_FRAMES: int = 3
    #: "Another live track within ~2 box-widths" — the plan's own "cheap and
    #: nearly-sufficient first cut" for association ambiguity (§5). Widths, not
    #: an absolute distance, so it scales with the object's own size in frame.
    ASSOCIATION_AMBIGUITY_BOX_WIDTHS: float = 2.0

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
        #: source_uuid -> track_id -> currently-open detector-miss gap (plan
        #: §4A). Keyed per source because track_ids are only unique within one
        #: source's own tracker instance, not across a multi-camera watcher.
        self._track_gaps: Dict[str, Dict[int, _GapState]] = {}
        #: source_uuid -> track_ids that died since the last offer actually went
        #: out. `context.dead_track_ids` is only valid for the single step it's
        #: read on (WatchPipeline reports each death exactly once), but offers to
        #: the collector are throttled by `_min_interval_s` — far coarser than the
        #: tracker's own cadence. Without accumulating here, a track that dies
        #: between two throttled offers is never reported to the collector at
        #: all, silently breaking `TrainingCollector`'s "commit the track's worst
        #: crop at track end" guarantee (plan §4B) for the common case.
        #: A list, not a set: `dead_track_ids`' consumers (and the tests
        #: exercising them) treat it as ordered, and a set's iteration order is
        #: not something to rely on even for ints.
        self._pending_dead_track_ids: Dict[str, List[int]] = {}

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
                # Its own type, not `str`: the watchers form renders a picker for it
                # (`collectorPickerWidget`). A pasted uuid naming nothing fails when the
                # WATCHER builds, far from the form that accepted it.
                "type": "training_collector_uuid",
                "required": True,
                "help": "Collector to feed. Edit collectors in Configuration → Collectors.",
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

    @staticmethod
    def _normalize_tracks(tracks: List[Any], frame: Any) -> List[TrackedCrop]:
        """`Track.bbox` is pixel coordinates in `frame`'s space (the analysed,
        downscaled frame — never `full_frame`); `TrackedCrop` is normalized so the
        collector can compare it against detections, which it always sees in
        normalized form regardless of which resolution inference actually ran at.
        Only the driver knows which frame `Track.bbox` is relative to, so this
        conversion happens here rather than inside the collector.
        """
        if frame is None:
            return []
        height, width = frame.shape[:2]
        if not width or not height:
            return []
        from ..training.collector import TrackedCrop

        out: List[TrackedCrop] = []
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            out.append(
                TrackedCrop(
                    track_id=t.track_id,
                    x=max(0.0, x1 / width),
                    y=max(0.0, y1 / height),
                    width=max(0.0, (x2 - x1) / width),
                    height=max(0.0, (y2 - y1) / height),
                    label_uuid=t.label_uuid,
                    matched_this_update=t.matched_this_update,
                    confirmed=t.confirmed,
                )
            )
        return out

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
        # No matching entry — e.g. this action's in-memory state was reset (task
        # restart) after the matching START had already landed. There is nothing
        # of ours to close, and popping an unrelated-but-still-open rule's entry
        # would end that rule's session early; leave the stack untouched instead.
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

        # Accumulated every in-scope frame, not just the ones that end up
        # offered — see `_pending_dead_track_ids`'s own docstring for why a
        # per-frame read here (rather than reading `context.dead_track_ids`
        # only inside the throttled offer below) is required.
        if context.dead_track_ids:
            bucket = self._pending_dead_track_ids.setdefault(key, [])
            for track_id in context.dead_track_ids:
                if track_id not in bucket:
                    bucket.append(track_id)

        # Gap tracking runs on EVERY in-scope frame, unlike the periodic offer
        # below — a gap can open and resolve entirely between two throttled
        # samples, and missing it is exactly what this feature exists to catch
        # (plan §4A). Never raises past this call: a broken gap-tracking pass
        # must not cost the watcher its ordinary periodic sampling.
        if collector.spec.enable_detector_miss:
            try:
                await self._process_track_gaps(key, collector, context)
            except Exception:
                logger.exception(
                    "collect_training_images detector-miss gap tracking failed for %s", self._collector_uuid
                )

        now = time.monotonic()
        if now - self._last_offered.get(key, 0.0) < self._min_interval_s:
            return ActionResult(ok=True, detail="waiting for interval")
        self._last_offered[key] = now

        # The offer actually going out discharges every death accumulated since
        # the last one — see `_pending_dead_track_ids`.
        dead_track_ids = self._pending_dead_track_ids.pop(key, None)
        frame = self._build_frame_offer(
            context,
            key,
            dead_track_ids=dead_track_ids,
            base64_override=await self._wider_inference_base64(collector, context),
        )
        try:
            await collector.offer(frame)
        except Exception as e:
            logger.exception("collect_training_images offer failed for collector %s", self._collector_uuid)
            return ActionResult(ok=False, detail=f"Offer failed: {e}")
        return ActionResult(ok=True)

    async def _wider_inference_base64(self, collector: TrainingCollector, context: ActionContext) -> Optional[str]:
        """Re-encode the full-res frame when the collector's models can use more
        pixels than the watcher's analysis frame carries (plan §2.3b).

        `None` means "reuse `context.analysis_base64`", and that is the common
        case by a wide margin: for a 640-input model the watcher's own
        `analysis_width=640` is *already* `min(native, model_input)`, so its boxes
        are not a degraded approximation of a full-res pass — they are the same
        boxes, and sharing them is free. This only spends an encode where 640
        genuinely throws away detail the model could have used.

        Never sends MORE than the model's input either: above it you pay to ship
        pixels the server resizes away, which is the mistake the deleted
        "full_res inference mode" would have made on every frame.

        Only the periodic offer takes this path. A detector-miss commit
        (`_commit_gap`) bypasses inference entirely, so re-encoding for it would
        be pure waste.
        """
        full = getattr(context, "full_frame", None)
        analysis = getattr(context, "frame", None)
        if full is None or analysis is None:
            return None
        try:
            native_width = int(full.shape[1])
            analysis_width = int(analysis.shape[1])
            send_width = collector.send_width(native_width)
            # isinstance, not truthiness: only a real width is actionable, and a
            # test double's auto-vivified attribute is truthy but not comparable.
            if not isinstance(send_width, int) or send_width <= analysis_width:
                return None
            resized = downscale_to_width(full, send_width)
            return await asyncio.to_thread(base64_encode_jpg, resized)
        except Exception:
            # The analysis frame is always a valid input — losing the offer over a
            # resolution optimisation would trade a collected frame for nothing.
            logger.exception("collect_training_images could not re-encode at send_width; using the analysis frame")
            return None

    def _build_frame_offer(
        self,
        context: ActionContext,
        key: str,
        dead_track_ids: Optional[List[int]] = None,
        base64_override: Optional[str] = None,
    ) -> FrameOffer:
        additional_info: Dict[str, Any] = {}
        stack = self._open_sessions.get(key) or []
        if not self._continuous and stack:
            rule_uuid, started_at = stack[-1]
            additional_info = {"watch_rule_uuid": rule_uuid, "watch_event_started_at": str(started_at)}
        # `dead_track_ids=None` means "caller doesn't care" (e.g. the gap-tracking
        # snapshot below, which never reads this field) — fall back to the raw,
        # single-step context value rather than silently swallowing it.
        resolved_dead_ids = context.dead_track_ids if dead_track_ids is None else dead_track_ids
        from ..training.collector import FrameOffer

        return FrameOffer(
            provided=ImageWithProviderMetadata(
                image=context.full_frame,
                metadata=ImageProviderMetadata(
                    source=context.source_name,
                    source_uuid=context.source_uuid,
                    frame_idx=context.frame_seq,
                    additional_info=additional_info,
                ),
            ),
            base64=base64_override or context.analysis_base64,
            tracks=tuple(self._normalize_tracks(context.tracks, context.frame)),
            dead_track_ids=tuple(resolved_dead_ids),
        )

    async def _process_track_gaps(self, key: str, collector: TrainingCollector, context: ActionContext) -> None:
        """Open, extend, resolve or abandon detector-miss gaps for every live
        track on this source (plan §4A) — called every analysed frame,
        independent of the periodic offer's own interval gate.
        """
        live = self._normalize_tracks(context.tracks, context.frame)
        target_labels = collector.spec.target_label_bands
        gaps = self._track_gaps.setdefault(key, {})

        for track in live:
            # Only for labels the collector is actually configured to collect —
            # mirrors `_filter_model`'s own target-label restriction, so this
            # never seeds a box for a label nobody asked for.
            if track.label_uuid not in target_labels:
                continue
            # Only a CONFIRMED track (plan §4A) — a still-provisional, single-hit
            # track is exactly as likely to be tracker noise as a real object, and
            # seeding a "detector miss" box for one would put unverified noise
            # into the training set under the same tag real gaps use.
            if not track.confirmed:
                continue
            pending = gaps.get(track.track_id)
            if track.matched_this_update:
                if pending is not None:
                    del gaps[track.track_id]
                    await self._commit_gap(collector, pending, track)
                continue
            # Coasting this frame — the detector missed it.
            if pending is None:
                ambiguous = self._is_association_ambiguous(track, live)
                held_frame = self._build_frame_offer(context, key)
                # Computed at collection time, not recoverable later — the
                # same provenance pass-through event-scoped offers already
                # use, read by the canvas for the two-press Verify-all warning
                # (plan §5).
                held_frame.provided.metadata.additional_info["detector_miss_ambiguous"] = ambiguous
                gaps[track.track_id] = _GapState(held_frame=held_frame, before=track, ambiguous=ambiguous)
            else:
                pending.frame_count += 1
                if pending.frame_count > self.MAX_GAP_FRAMES:
                    # Past this many frames the tracker's box has drifted too
                    # far from the truth to be worth a human's time — drop it
                    # silently rather than committing a stale interpolation.
                    del gaps[track.track_id]

        for track_id in context.dead_track_ids:
            # The object left; it did not merely blink. A gap that never
            # re-acquires must never be committed (plan §4A: "a coasting track
            # is a prediction, not ground truth").
            gaps.pop(track_id, None)

        if not gaps:
            self._track_gaps.pop(key, None)

    @classmethod
    def _is_association_ambiguous(cls, track: TrackedCrop, live: List[TrackedCrop]) -> bool:
        """ "Is there another live track close enough to this one that the
        interpolated box's identity could be wrong" (plan §5) — the cheap,
        nearly-sufficient first cut: distance between box centres against a
        multiple of the track's own box width. Hierarchy-aware weighting
        (favour flagging two overlapping animals over an animal and a parked
        car) is a documented future refinement, not implemented here.
        """
        if track.width <= 0:
            return False
        cx, cy = track.x + track.width / 2, track.y + track.height / 2
        threshold = cls.ASSOCIATION_AMBIGUITY_BOX_WIDTHS * track.width
        for other in live:
            if other.track_id == track.track_id:
                continue
            ocx, ocy = other.x + other.width / 2, other.y + other.height / 2
            if math.hypot(cx - ocx, cy - ocy) <= threshold:
                return True
        return False

    async def _commit_gap(self, collector: TrainingCollector, pending: _GapState, reacquired: TrackedCrop) -> None:
        """A gap resolved by re-acquisition — seed the interpolated box onto
        the held (first-coasting-frame) image and save it.

        Interpolate, don't extrapolate (plan §5): both real detections
        bracketing the gap are in hand, so the seeded box is the position
        between them, not the tracker's forward-only live prediction.
        `frame_count` is how many coasting steps the gap actually lasted, so
        the held frame sits at fraction `1 / (frame_count + 1)` of the way from
        `before` to `reacquired`.
        """
        before = pending.before
        frac = 1.0 / (pending.frame_count + 1)
        det = _DetectorMissDetection(
            bbox=_DetectorMissBBox(
                x=before.x + frac * (reacquired.x - before.x),
                y=before.y + frac * (reacquired.y - before.y),
                width=before.width + frac * (reacquired.width - before.width),
                height=before.height + frac * (reacquired.height - before.height),
                label_uuid=before.label_uuid,
                tag_uuids=[DETECTOR_MISS_TAG_UUID],
            ),
        )
        try:
            await collector.offer_detector_miss(pending.held_frame, det)
        except Exception:
            logger.exception("collect_training_images detector-miss commit failed for %s", self._collector_uuid)
