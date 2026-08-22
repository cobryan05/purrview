"""Per-source pipeline for StreamWatcherTask: provider → tracker → engine.

Holds everything that is *per camera* — the image provider, its pacing, its tracker
state, its zones and its rule machines — so the task above stays orchestration and
nothing else. `CollectTrainingImagesTask` at 2400 lines is the warning this split
exists to avoid.

Nothing here does inference or persistence; both are injected by the task. That keeps
the pipeline testable against a fake provider and a stub inference callable.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from apps.helpers.imageProviders.imageProvider import ImageProvider
from apps.helpers.tracking.tracker import Detection, LabelLifetimeSpec, Track, Tracker
from apps.helpers.watch.engine import WatchEngine
from apps.helpers.watch.events import WatchEventSignal
from apps.helpers.watch.geometry import Zone, is_masked
from apps.helpers.watch.motion import DEFAULT_SENSITIVITY as MOTION_DEFAULT_SENSITIVITY
from apps.helpers.watch.motion import NO_MOTION, MotionDetector, MotionField
from apps.helpers.watch.rules import descendants_of

logger = logging.getLogger(__name__)


def downscale_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    """Shrink a frame to `target_width`, preserving aspect ratio.

    Analysis resolution is a first-class knob because per-frame inference over
    base64 HTTP is the throughput ceiling: 640 px is plenty to track a cat and cuts
    the payload roughly tenfold. Recording is untouched — MediaMTX keeps writing the
    full-rate, full-resolution stream regardless of what we analyse.

    Never upscales: making a 320 px camera into a 640 px payload costs bandwidth and
    inference time while adding no information.
    """
    height, width = image.shape[:2]
    if target_width <= 0 or width <= target_width:
        return image
    scale = target_width / float(width)
    new_height = max(1, int(round(height * scale)))
    # cv2 is already a dependency via the image providers; INTER_AREA is the correct
    # filter for downscaling (box-averaging), unlike the default bilinear.
    import cv2

    return cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)


def _format_duration(seconds: float) -> str:
    """Coarse, readable elapsed time — "8s", "4m", "2h 15m"."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


@dataclass
class SourceHealth:
    """Per-source liveness, mirroring CollectTrainingImagesTask's backoff behaviour.

    A camera that drops for a minute is a normal event on a 24/7 run; losing that
    source silently for the rest of the run is not acceptable, and neither is killing
    the whole watcher.
    """

    consecutive_failures: int = 0
    backoff_s: float = 0.0
    next_attempt_at: float = 0.0
    last_frame_at: Optional[float] = None
    last_error: Optional[str] = None
    frames_analysed: int = 0
    #: When the current run of failures began, or None while healthy. Downtime is
    #: measured from here rather than counted in failures, because the retry interval
    #: grows: ten failures is 30 seconds early on and over half an hour later, so a
    #: failure count says almost nothing about how long a camera has been away.
    failing_since: Optional[float] = None
    #: True while the source is coming up rather than broken. Kept apart from the
    #: failure count because the two need opposite handling: a connect is polled
    #: often and briefly, a failure is retried on a widening backoff.
    connecting: bool = False

    #: How often to look in on a source that is still connecting. Short, because the
    #: cost is one cheap readiness check and the alternative is leaving a camera that
    #: came up two seconds ago unread for the length of a failure backoff.
    CONNECT_POLL_S = 2.0

    BACKOFF_BASE_S = 5.0
    BACKOFF_MAX_S = 300.0
    #: Downtime past which the source stops being "a camera that dropped" and starts
    #: being a fault worth reporting. Retries do not stop here — they continue at
    #: BACKOFF_MAX_S forever, because a camera that has been unplugged for a day is
    #: still expected to come back when it is plugged in again.
    FAILED_AFTER_S = 120.0

    def record_success(self, now: float) -> None:
        self.consecutive_failures = 0
        self.backoff_s = 0.0
        self.next_attempt_at = 0.0
        self.last_frame_at = now
        self.last_error = None
        self.failing_since = None
        self.connecting = False
        self.frames_analysed += 1

    def record_connecting(self, now: float) -> None:
        """Note that the source is still coming up. Not a failure.

        Deliberately does not touch the failure count or the backoff. A camera that
        takes three minutes to connect would otherwise spend that time accruing
        failures, widening its own retry interval and being declared failed before it
        ever had the chance to succeed — and, worse, being torn down and restarted on
        each of those retries so the three minutes never elapsed uninterrupted.

        Downtime still runs from here, because the source genuinely is not delivering;
        what changes is that `is_failed` will not call it a fault while it holds.
        """
        if self.failing_since is None:
            self.failing_since = now
        self.connecting = True
        self.last_error = None
        self.next_attempt_at = now + self.CONNECT_POLL_S

    def record_failure(self, now: float, error: str) -> None:
        if self.failing_since is None:
            self.failing_since = now
        # A real failure supersedes a connect in progress: whatever window it had, it
        # has stopped being a source that is merely slow to arrive.
        self.connecting = False
        self.consecutive_failures += 1
        self.last_error = error
        # The exponent is capped, not the failure count: retries continue indefinitely,
        # they just settle at BACKOFF_MAX_S apart rather than growing until the next
        # attempt is scheduled after the heat death of the universe.
        self.backoff_s = min(
            self.BACKOFF_MAX_S,
            self.BACKOFF_BASE_S * (2 ** min(self.consecutive_failures - 1, 8)),
        )
        self.next_attempt_at = now + self.backoff_s

    @property
    def healthy(self) -> bool:
        return self.consecutive_failures == 0

    def downtime_s(self, now: float) -> float:
        """How long this source has been failing. 0.0 while healthy."""
        if self.failing_since is None:
            return 0.0
        return max(0.0, now - self.failing_since)

    def is_failed(self, now: float) -> bool:
        """Whether this source has been down long enough to call it failed.

        A reporting distinction only — nothing stops retrying on the strength of it.

        A source still connecting is never failed however long it has taken. Coming up
        can legitimately outlast FAILED_AFTER_S, and reporting a camera as failed
        while it is in the middle of successfully connecting is how a health signal
        teaches people to ignore it.
        """
        if self.connecting:
            return False
        return self.failing_since is not None and self.downtime_s(now) >= self.FAILED_AFTER_S

    def state(self, now: float) -> str:
        """One of ``ok`` / ``connecting`` / ``reconnecting`` / ``failed``, for display."""
        if self.healthy and not self.connecting:
            return "ok"
        if self.connecting:
            return "connecting"
        return "failed" if self.is_failed(now) else "reconnecting"


@dataclass
class WatchPipeline:
    """One camera's full analysis path."""

    source_uuid: str
    source_name: str
    provider: ImageProvider
    tracker: Tracker
    engine: WatchEngine
    #: This watcher's override for the name MQTT topics publish this source under.
    #: None means "use `source_name`". Kept separate from `source_name` rather than
    #: replacing it, because logs, the status table and snapshot filenames should keep
    #: naming the camera the way the Sources pane does — only the topic changes.
    mqtt_name: Optional[str] = None
    model_names: List[str] = field(default_factory=list)
    analysis_fps: float = 5.0
    analysis_width: int = 640
    #: Reported by the relay's ensure response — drives the UI's buffer badge and the
    #: choice of clip backend. Never inferred locally, so the two cannot disagree.
    ingest_tier: str = "native"
    buffered: bool = False
    retention_s: float = 0.0
    #: Frame-ring bounds for a source with no continuous stream, from its watch-source
    #: row. Unused when the source has a real rewind buffer — the ring exists precisely
    #: because there is no timeline to rewind.
    ring_seconds: float = 20.0
    ring_max_mb: float = 32.0
    #: BufferMode.OFF means the user has asked for no buffering at all on this source,
    #: which must also suppress the ring — otherwise "off" would still cost RAM.
    buffer_mode: str = "auto"
    #: Regions that suppress detections, already filtered to the ones this watcher has
    #: not opted out of. Unlike zones these are not addressable by any rule — a mask acts
    #: on everything or not at all, which is exactly why a watcher can disable one.
    masks: List[Zone] = field(default_factory=list)
    #: Frame differencer for `motion` rules, or None when this watcher has no motion
    #: rule that could run on this source. None rather than an idle detector so a
    #: watcher that never asked for motion pays nothing per frame for it.
    motion_detector: Optional[MotionDetector] = None
    health: SourceHealth = field(default_factory=SourceHealth)
    _next_frame_at: float = 0.0
    _last_tracks: List[Track] = field(default_factory=list)
    #: Detections from the last analysed frame, *before* the tracker saw them. Kept
    #: only for the live view, and kept separately from tracks on purpose: "inference
    #: returned nothing" and "the tracker discarded everything inference returned" look
    #: identical from the tracks alone, and they have opposite fixes.
    _last_detections: List[Detection] = field(default_factory=list)
    _last_frame_shape: Optional[tuple] = None
    #: How many detections the masks suppressed on the last analysed frame. Surfaced in
    #: the status row because a mask that is silently eating every detection and a model
    #: that has stopped detecting look identical otherwise — and they have opposite fixes.
    _last_masked_count: int = 0
    #: This frame's motion, produced by `prepare_frame` and consumed by `step`. Held on
    #: the pipeline rather than passed between them because the frame is only in scope
    #: in the first and the rules are only evaluated in the second.
    _last_motion: MotionField = field(default=NO_MOTION)
    #: Whole-frame unmasked motion at the default sensitivity, for the status panel.
    #: A number here is the difference between "the rule's threshold is too high" and
    #: "this camera reports no motion at all", which look identical from the outside.
    _last_motion_ratio: Optional[float] = None
    #: Most recent analysed frame, kept for snapshot actions. One frame per source,
    #: replaced each time — deliberately not a history.
    last_frame: Optional[np.ndarray] = None
    #: The same frame *before* the analysis downscale — i.e. at the resolution the
    #: camera actually sent. Kept so the live MQTT feed can publish a full-resolution
    #: image while inference keeps running on the cheap one; the two are different
    #: questions and `analysis_width` should only answer the first.
    #:
    #: ⚠️ Detections are in `last_frame`'s pixel space, never this one. Anything drawing
    #: onto this frame must scale the boxes (see `annotate_detections(scale=...)`).
    #:
    #: When no downscale happened this is the *same array object* as `last_frame`, not a
    #: copy — so the extra cost for an already-small camera is one reference.
    last_full_frame: Optional[np.ndarray] = None
    #: Increments once per analysed frame. Lets the live view's encode cache and its
    #: caller both ask "is this the frame I already have?" without comparing pixels —
    #: which is what keeps a 5 Hz poll of a 1 fps camera free.
    frame_seq: int = 0

    @property
    def frame_interval_s(self) -> float:
        return 1.0 / self.analysis_fps if self.analysis_fps > 0 else 0.0

    def due(self, now: float) -> bool:
        """Whether this source should be serviced now.

        Pacing is per source rather than global so a 1 fps doorbell camera doesn't
        drag a 5 fps litter-box camera down to its rate.
        """
        if now < self.health.next_attempt_at:
            return False
        return now >= self._next_frame_at

    def schedule_next(self, now: float) -> None:
        self._next_frame_at = now + self.frame_interval_s

    def prepare_frame(self, image: np.ndarray) -> np.ndarray:
        """Downscale and remember the analysis geometry the zones scale against."""
        analysed = downscale_to_width(image, self.analysis_width)
        self.last_frame = analysed
        # Held before the downscale, so an action that wants the camera's own resolution
        # has it. `downscale_to_width` returns its input unchanged when nothing needs
        # shrinking, in which case these two names point at one array.
        self.last_full_frame = image
        self.frame_seq += 1
        height, width = analysed.shape[:2]
        if self._last_frame_shape != (height, width):
            self._last_frame_shape = (height, width)
            # Zones are stored normalised, so they only become pixels once we know the
            # analysis size — which isn't known until the first frame arrives.
            self.engine.set_frame_size(width, height)

        # Differenced here rather than in `step`, because this is the only place the
        # picture itself is in scope — and against the *analysed* frame, so a motion
        # ratio means the same thing as a zone, which is scaled to this same geometry.
        if self.motion_detector is not None:
            self._last_motion = self.motion_detector.update(analysed)
            self._last_motion_ratio = (
                round(self._last_motion.ratio(MOTION_DEFAULT_SENSITIVITY), 4) if self._last_motion.measured else None
            )
        return analysed

    def apply_masks(self, detections: Sequence[Detection]) -> List[Detection]:
        """Drop detections whose centre falls inside a mask.

        Applied *before* the tracker, not after, so a masked object never forms a track
        at all. Filtering tracks instead would let one build up history, satisfy a rule's
        `min_duration_s`, and then vanish — and a track that flickers in and out of
        existence is far worse for hysteresis than one that never existed.
        """
        if not self.masks:
            self._last_masked_count = 0
            return list(detections)

        width, height = self.engine.frame_width, self.engine.frame_height
        kept = [d for d in detections if not is_masked(d.bbox, self.masks, width, height)]
        self._last_masked_count = len(detections) - len(kept)
        return kept

    def step(self, detections: Sequence[Detection], now: float) -> List[WatchEventSignal]:
        """Advance the tracker and rule engine for one analysed frame."""
        surviving = self.apply_masks(detections)
        # `_last_detections` is what the tracker actually saw, so the live view shows the
        # same picture the rules are reasoning about.
        self._last_detections = list(surviving)
        self._last_tracks = self.tracker.update(list(surviving), now)
        return self.engine.step(self._last_tracks, now, motion=self._last_motion)

    def close_open_events(self, now: float) -> List[WatchEventSignal]:
        return self.engine.close_all(now)

    def open_event_signals(self, now: float) -> List[WatchEventSignal]:
        return self.engine.open_signals(now)

    @property
    def tracks(self) -> List[Track]:
        return list(self._last_tracks)

    @property
    def detections(self) -> List[Detection]:
        return list(self._last_detections)

    def adopt_buffer_capabilities(self, provider: Any) -> None:
        """Take the source's buffer facts from the provider, which got them from the relay.

        All three move together on purpose. They were copied field-by-field at the call
        site and `retention_s` was simply forgotten, which is invisible until something
        reads it: the badge showed "0s rewind" for a camera with a 45 s window, and
        `RecordClipAction.plan_window` guards its pre-roll clamp on `retention_s > 0`, so
        the clamp never ran. Neither failure raises anything.

        Defaults are the unbuffered ones, so a provider that reports nothing (a snapshot
        poller, a test double) is described as having no rewind rather than inheriting
        whatever the previous source had.
        """
        self.buffered = bool(getattr(provider, "buffered", False))
        self.ingest_tier = getattr(provider, "ingest_tier_name", "none")
        self.retention_s = float(getattr(provider, "retention_s", 0.0) or 0.0)

    def status_row(self, now: float) -> Dict[str, Any]:
        """One row of the status panel's Sources table."""
        last_frame = self.health.last_frame_at
        return {
            "source": self.source_name,
            "sourceUuid": self.source_uuid,
            "analysisFps": self.analysis_fps,
            "analysisWidth": self.analysis_width,
            "framesAnalysed": self.health.frames_analysed,
            "lastFrameAgoS": round(now - last_frame, 1) if last_frame else None,
            # A source that has been down for an hour must not read the same as one
            # that missed a frame two seconds ago. The duration is what says which.
            "health": self._health_label(now),
            # Four-state, not yes/no: this is what stops a user configuring 10 s of
            # pre-roll on a snapshot camera and silently getting three choppy frames.
            "buffer": self._buffer_label(),
            "tracks": len(self._last_tracks),
            "masks": len(self.masks),
            "maskedLastFrame": self._last_masked_count,
            # None means this source is not being differenced at all (no motion rule
            # applies to it), which is a different thing from "nothing moved".
            "motionRatio": self._last_motion_ratio,
            "lastError": self.health.last_error,
        }

    def _health_label(self, now: float) -> str:
        """Human-readable health, leading with how long rather than how many.

        "failing (37)" invited the reading that something had gone wrong 37 times;
        what it meant was that one camera had been away for an hour and a half.
        """
        down = _format_duration(self.health.downtime_s(now))
        if self.health.connecting:
            # Named separately from reconnecting so a camera that is simply slow to
            # come up does not read as one that keeps dropping.
            return f"connecting — waiting for the stream ({down})"
        if self.health.healthy:
            return "ok"
        if self.health.is_failed(now):
            return f"failed — no frames for {down}, still retrying"
        return f"reconnecting — no frames for {down}"

    def _buffer_label(self) -> str:
        if not self.buffered:
            if self.ingest_tier == "ring":
                return "◐ frame ring (snapshots, low fps)"
            return "⚠ no buffer"
        seconds = int(self.retention_s)
        if self.ingest_tier == "ffmpeg":
            return f"● {seconds}s rewind (ffmpeg ingest)"
        return f"● {seconds}s rewind (native)"


def build_label_lifetimes(
    rows: Sequence[Any],
    ancestry: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, LabelLifetimeSpec]:
    """Turn `label_lifetimes` rows into the tracker's lookup table.

    A row names a *list* of labels and may opt into covering their descendants, which is
    expanded here — once at watcher start, against the ancestry map the task already
    fetched — for the same reason rules are: storing the closure would freeze the row at
    the moment it was written.

    A label can be reached by two rows: named outright by one and inherited from an
    ancestor by another. **Named beats inherited**, regardless of row order, because the
    specific statement is the one the user wrote about *that* label. Two rows naming the
    same label outright is a genuine ambiguity the last one wins, as it always did.
    """
    inherited: Dict[str, LabelLifetimeSpec] = {}
    named: Dict[str, LabelLifetimeSpec] = {}
    for row in rows or []:
        get = row.get if isinstance(row, dict) else lambda k, d=None: getattr(row, k, d)
        refs = get("label_uuids_json", get("labelUuidsJson", None)) or []
        refs = [str(u) for u in refs if u]
        if not refs:
            continue
        spec = LabelLifetimeSpec(
            ttl_s=float(get("ttl_s", get("ttlS", 2.0)) or 2.0),
            is_anchor=bool(get("is_anchor", get("isAnchor", False))),
        )
        for ref in refs:
            named[ref] = spec
        if ancestry and bool(get("include_descendants", get("includeDescendants", False))):
            for ref in refs:
                for child in descendants_of(ref, ancestry):
                    inherited[child] = spec

    return {**inherited, **named}


def expand_model_group(group_uuid: Optional[str], groups_by_uuid: Dict[str, Any]) -> List[str]:
    """Flatten a Model Stack into an ordered, de-duplicated list of model names.

    Groups may nest, so this recurses; a visited set guards against a cycle that
    slipped past the DB-level check, because an infinite recursion here would hang the
    watcher rather than reporting a bad config.
    """
    names: List[str] = []
    visited: set = set()

    def _walk(uuid: Optional[str]) -> None:
        if not uuid or uuid in visited:
            return
        visited.add(uuid)
        group = groups_by_uuid.get(uuid)
        if group is None:
            logger.warning("Model group %s referenced but not found", uuid)
            return
        members = group.members if hasattr(group, "members") else group.get("members", [])
        for member in sorted(
            members, key=lambda m: getattr(m, "order", 0) if not isinstance(m, dict) else m.get("order", 0)
        ):
            get = member.get if isinstance(member, dict) else lambda k, d=None: getattr(member, k, d)
            model_name = get("model_name") or get("modelName")
            child = get("child_group_uuid") or get("childGroupUuid")
            if model_name:
                if model_name not in names:
                    names.append(model_name)
            elif child:
                _walk(child)

    _walk(group_uuid)
    return names


def resolve_model_names(
    group_uuid: Optional[str],
    model_name: Optional[str],
    groups_by_uuid: Dict[str, Any],
) -> List[str]:
    """Resolve a (stack, single model) pair into the models to run.

    A watch source picks *either* a stack or one bare model — wrapping a single model
    in a one-member stack just to run it is ceremony the config should not demand. The
    two are mutually exclusive at the API layer, but a row written before that check
    existed could hold both, so the stack wins here rather than raising: a watcher that
    detects with too much is recoverable, one that refuses to start is not.
    """
    if group_uuid:
        return expand_model_group(group_uuid, groups_by_uuid)
    return [model_name] if model_name else []
