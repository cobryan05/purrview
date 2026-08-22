"""Frame-difference motion, for the `motion` rule relation.

Motion is the one predicate that does not come from the model. Every other relation
reads tracks, which means nothing fires until inference has named an object — and a
camera pointed at a back gate should be able to record *something moved* without
first being taught what a delivery van looks like.

The measure is deliberately the plainest one that works: blur, absolute difference
against the previous analysed frame, threshold per pixel, count what changed. No
background model. A learned background (MOG2 et al.) is better at ignoring a swaying
branch, but it has to be *trained*, and it re-learns a parked object into the
background after a minute — which for "did anything move at the gate" is the failure
mode that matters, and it is silent. Frame differencing has one honest weakness (a
perfectly still object is invisible) and the rule's `min_duration_s` already assumes
the predicate is instantaneous.

Masks are honoured here rather than after the fact, and this is the whole reason
this module knows about geometry at all: masked pixels are excluded from both the
numerator *and* the denominator, so a mask over a busy road does not merely fail to
trigger — it stops diluting the ratio for the rest of the frame. Suppressing the
event afterwards would not do that: a camera whose right half is a motorway would
read as "3% of the frame changed" all night.

Cost is one blur, one absdiff and one bitwise-and per analysed frame per camera, at
the analysis resolution (640px wide by default) — well under a millisecond. The
per-sensitivity threshold-and-count is done lazily, once per distinct
(sensitivity, region) pair a rule actually asks for, and memoised for the frame; two
rules watching the same zone at the same sensitivity cost one count between them.
"""

from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import Zone

#: Per-pixel luminance change, normalised to 0..1, that counts as "this pixel changed".
#: 0.10 (≈25/255) is the usual starting point for frame differencing: high enough to
#: ignore sensor noise and JPEG ringing, low enough to catch a dark animal against a
#: dark floor. Rules may override it — an IR camera at night needs a higher value.
DEFAULT_SENSITIVITY = 0.10

#: Gaussian kernel applied before differencing. Its whole job is to stop per-pixel
#: sensor noise from being counted as motion; without it a noisy night feed reads as a
#: steady 1–2% of the frame changing with nothing in front of the camera.
_BLUR_KERNEL = (5, 5)

#: Key under which the whole (unmasked) frame is rasterised. Not a valid uuid, so it
#: cannot collide with a zone.
_WHOLE_FRAME = ""


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return cv2.GaussianBlur(gray, _BLUR_KERNEL, 0)


def _zone_raster(zone: Zone, width: int, height: int) -> np.ndarray:
    """A uint8 0/1 raster of one zone at the analysis resolution.

    Polygons are rasterised exactly, unlike the area *relations*, which measure a zone
    by its bounding box (see `Zone.bounds`). There is no reason to approximate here —
    filling a polygon is a single cheap call and the result is cached per frame size,
    so the concave zones people actually draw around a driveway measure correctly.
    """
    raster = np.zeros((height, width), dtype=np.uint8)
    points = zone.to_pixels(float(width), float(height))
    if zone.kind == "box":
        if len(points) < 2:
            return raster
        (x1, y1), (x2, y2) = points[0], points[1]
        lo_x, hi_x = sorted((int(round(x1)), int(round(x2))))
        lo_y, hi_y = sorted((int(round(y1)), int(round(y2))))
        cv2.rectangle(raster, (lo_x, lo_y), (hi_x, hi_y), 1, thickness=-1)
        return raster
    if len(points) < 3:
        return raster
    poly = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
    cv2.fillPoly(raster, [poly], 1)
    return raster


class MotionField:
    """One frame's motion, queryable at whatever sensitivity each rule asked for.

    Holds the raw per-pixel delta rather than a decision, because "how much moved" is
    only answerable once a rule says how big a change it counts and which region it
    cares about. Thresholding eagerly at one sensitivity would force every rule on the
    camera to share it.
    """

    def __init__(
        self,
        delta: Optional[np.ndarray],
        rasters: Dict[str, np.ndarray],
        areas: Dict[str, int],
    ) -> None:
        self._delta = delta
        self._rasters = rasters
        self._areas = areas
        self._cache: Dict[Tuple[float, Tuple[str, ...]], Tuple[float, Optional[str]]] = {}

    @property
    def measured(self) -> bool:
        """False on the first frame from a camera, when there is nothing to difference."""
        return self._delta is not None

    def _region_ratio(self, key: str, threshold_px: int) -> float:
        raster = self._rasters.get(key)
        area = self._areas.get(key, 0)
        if raster is None or area <= 0 or self._delta is None:
            return 0.0
        changed = cv2.threshold(self._delta, threshold_px, 1, cv2.THRESH_BINARY)[1]
        # The whole-frame raster is already mask-subtracted and every zone raster is
        # intersected with it, so this one AND applies the masks in every case.
        return float(np.count_nonzero(cv2.bitwise_and(changed, raster))) / float(area)

    def best(self, sensitivity: float, zone_uuids: Sequence[str] = ()) -> Tuple[float, Optional[str]]:
        """Strongest motion ratio over the requested regions, and which zone produced it.

        No zones means the whole unmasked frame, and no zone to name. Several zones read
        as "motion in any of these", matching how a rule's other multi-ref slots behave.
        """
        key = (float(sensitivity), tuple(zone_uuids))
        if key in self._cache:
            return self._cache[key]

        threshold_px = int(round(max(0.0, min(1.0, sensitivity)) * 255))
        # A threshold of 0 would count every pixel whose value moved at all, i.e. all of
        # them on any real sensor. cv2.threshold is `> thresh`, so 0 still admits a
        # delta of 1 — clamp to at least that.
        threshold_px = max(1, threshold_px)

        if not zone_uuids:
            result = (self._region_ratio(_WHOLE_FRAME, threshold_px), None)
        else:
            best_ratio = 0.0
            best_zone: Optional[str] = None
            for uuid in zone_uuids:
                ratio = self._region_ratio(uuid, threshold_px)
                if ratio > best_ratio:
                    best_ratio, best_zone = ratio, uuid
            result = (best_ratio, best_zone)

        self._cache[key] = result
        return result

    def ratio(self, sensitivity: float, zone_uuids: Sequence[str] = ()) -> float:
        return self.best(sensitivity, zone_uuids)[0]


#: What a camera reports before it has two frames to compare, and what a pipeline with
#: no motion rules hands the engine. Shared because it is immutable in practice.
NO_MOTION = MotionField(None, {}, {})


class MotionDetector:
    """Per-source frame differencer. Stateful (it remembers the previous frame).

    Rasters are rebuilt only when the analysis resolution changes, which in practice
    means once, on the first frame. Zones and masks are fixed for a watcher's run — a
    change to either restarts the watcher — so nothing else can invalidate them.
    """

    def __init__(self, masks: Sequence[Zone] = (), zones: Sequence[Zone] = ()) -> None:
        self._masks = list(masks)
        self._zones = list(zones)
        self._prev: Optional[np.ndarray] = None
        self._shape: Optional[Tuple[int, int]] = None
        self._rasters: Dict[str, np.ndarray] = {}
        self._areas: Dict[str, int] = {}

    def _rebuild_rasters(self, width: int, height: int) -> None:
        allowed = np.ones((height, width), dtype=np.uint8)
        for mask in self._masks:
            raster = _zone_raster(mask, width, height)
            # Masked pixels leave the measurable area entirely — see the module
            # docstring for why this is not the same as ignoring motion found there.
            allowed[raster > 0] = 0

        rasters: Dict[str, np.ndarray] = {_WHOLE_FRAME: allowed}
        areas: Dict[str, int] = {_WHOLE_FRAME: int(np.count_nonzero(allowed))}
        for zone in self._zones:
            raster = cv2.bitwise_and(_zone_raster(zone, width, height), allowed)
            rasters[zone.uuid] = raster
            areas[zone.uuid] = int(np.count_nonzero(raster))
        self._rasters = rasters
        self._areas = areas

    def update(self, frame: np.ndarray) -> MotionField:
        """Difference `frame` against the previous one and return this frame's motion.

        Always advances the previous frame, including on the first call and on a
        resolution change — a detector that skipped the update would compare every
        later frame against one stale image and report the scene as permanently moving.
        """
        gray = _to_gray(frame)
        height, width = gray.shape[:2]

        if self._shape != (height, width):
            self._shape = (height, width)
            self._rebuild_rasters(width, height)
            # A resolution change makes the stored frame incomparable; treat the next
            # frame as a first frame rather than differencing against a resized ghost.
            self._prev = None

        prev = self._prev
        self._prev = gray
        if prev is None:
            return MotionField(None, self._rasters, self._areas)
        return MotionField(cv2.absdiff(gray, prev), self._rasters, self._areas)

    def reset(self) -> None:
        """Forget the previous frame, so the next one measures no motion.

        Used when a source reconnects: the frame before the outage and the frame after
        it may be minutes apart, and differencing them reports the whole scene as
        having moved at the exact moment a watcher comes back.
        """
        self._prev = None
