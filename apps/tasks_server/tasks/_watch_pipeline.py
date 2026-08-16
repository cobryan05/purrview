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
from apps.helpers.watch.geometry import Zone

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

    BACKOFF_BASE_S = 5.0
    BACKOFF_MAX_S = 300.0

    def record_success(self, now: float) -> None:
        self.consecutive_failures = 0
        self.backoff_s = 0.0
        self.next_attempt_at = 0.0
        self.last_frame_at = now
        self.last_error = None
        self.frames_analysed += 1

    def record_failure(self, now: float, error: str) -> None:
        self.consecutive_failures += 1
        self.last_error = error
        self.backoff_s = min(
            self.BACKOFF_MAX_S,
            self.BACKOFF_BASE_S * (2 ** min(self.consecutive_failures - 1, 8)),
        )
        self.next_attempt_at = now + self.backoff_s

    @property
    def healthy(self) -> bool:
        return self.consecutive_failures == 0


@dataclass
class WatchPipeline:
    """One camera's full analysis path."""

    source_uuid: str
    source_name: str
    provider: ImageProvider
    tracker: Tracker
    engine: WatchEngine
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
    health: SourceHealth = field(default_factory=SourceHealth)
    _next_frame_at: float = 0.0
    _last_tracks: List[Track] = field(default_factory=list)
    #: Detections from the last analysed frame, *before* the tracker saw them. Kept
    #: only for the live view, and kept separately from tracks on purpose: "inference
    #: returned nothing" and "the tracker discarded everything inference returned" look
    #: identical from the tracks alone, and they have opposite fixes.
    _last_detections: List[Detection] = field(default_factory=list)
    _last_frame_shape: Optional[tuple] = None
    #: Most recent analysed frame, kept for snapshot actions. One frame per source,
    #: replaced each time — deliberately not a history.
    last_frame: Optional[np.ndarray] = None
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
        self.frame_seq += 1
        height, width = analysed.shape[:2]
        if self._last_frame_shape != (height, width):
            self._last_frame_shape = (height, width)
            # Zones are stored normalised, so they only become pixels once we know the
            # analysis size — which isn't known until the first frame arrives.
            self.engine.set_frame_size(width, height)
        return analysed

    def step(self, detections: Sequence[Detection], now: float) -> List[WatchEventSignal]:
        """Advance the tracker and rule engine for one analysed frame."""
        self._last_detections = list(detections)
        self._last_tracks = self.tracker.update(list(detections), now)
        return self.engine.step(self._last_tracks, now)

    def close_open_events(self, now: float) -> List[WatchEventSignal]:
        return self.engine.close_all(now)

    @property
    def tracks(self) -> List[Track]:
        return list(self._last_tracks)

    @property
    def detections(self) -> List[Detection]:
        return list(self._last_detections)

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
            "health": "ok" if self.health.healthy else f"failing ({self.health.consecutive_failures})",
            # Four-state, not yes/no: this is what stops a user configuring 10 s of
            # pre-roll on a snapshot camera and silently getting three choppy frames.
            "buffer": self._buffer_label(),
            "tracks": len(self._last_tracks),
            "lastError": self.health.last_error,
        }

    def _buffer_label(self) -> str:
        if not self.buffered:
            if self.ingest_tier == "ring":
                return "◐ frame ring (snapshots, low fps)"
            return "⚠ no buffer"
        seconds = int(self.retention_s)
        if self.ingest_tier == "ffmpeg":
            return f"● {seconds}s rewind (ffmpeg ingest)"
        return f"● {seconds}s rewind (native)"


def build_label_lifetimes(rows: Sequence[Any]) -> Dict[str, LabelLifetimeSpec]:
    """Turn `label_lifetimes` rows into the tracker's lookup table."""
    lifetimes: Dict[str, LabelLifetimeSpec] = {}
    for row in rows or []:
        get = row.get if isinstance(row, dict) else lambda k, d=None: getattr(row, k, d)
        label_uuid = get("label_uuid") or get("labelUuid")
        if not label_uuid:
            continue
        lifetimes[label_uuid] = LabelLifetimeSpec(
            ttl_s=float(get("ttl_s", get("ttlS", 2.0)) or 2.0),
            is_anchor=bool(get("is_anchor", get("isAnchor", False))),
        )
    return lifetimes


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
