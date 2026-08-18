"""Geometric relations between tracked boxes and user-drawn zones.

Pure functions, no I/O. This is where the plan's pushback about IoU lives: for the
canonical "cat in litter box" case IoU is simply the wrong measure, because a cat's
box is far smaller than the litter box's and IoU stays low even at perfect
containment. `containment` and `centroid_in` are the relations that actually work.

Zones are stored normalised (0..1) so they survive a camera resolution change; they
are scaled to pixels here, at the one point where they meet a detection box.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2
Point = Tuple[float, float]


def area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    w = min(ax2, bx2) - max(ax1, bx1)
    h = min(ay2, by2) - max(ay1, by1)
    return w * h if w > 0 and h > 0 else 0.0


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union. Correct for objects of *similar* size only."""
    inter = intersection_area(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(a: BBox, b: BBox) -> float:
    """How much of A lies inside B, as a fraction of A's own area.

    This is the right relation for "is the cat in the litter box": it reaches 1.0 when
    A is fully inside B regardless of how much bigger B is, which is exactly where IoU
    fails.
    """
    a_area = area(a)
    return intersection_area(a, b) / a_area if a_area > 0 else 0.0


def overlap_ratio(a: BBox, b: BBox) -> float:
    """How much the boxes overlap, as a fraction of the *smaller* box's area.

    The relation for "is the cat drinking from the water bowl": the cat's box merely
    laps over the bowl's edge, so neither of the other two measures registers it. IoU
    is dominated by the union of two differently-sized boxes and stays near zero;
    `containment` asks how much of the cat is inside the bowl and answers "almost
    none". Neither is wrong — they answer questions this case isn't asking.

    Dividing by the smaller area asks the question that actually matters — how much of
    the smaller object is covered — so a cat's muzzle over the rim of a bowl scores on
    the bowl's scale rather than the cat's. Symmetric (unlike `containment`), and
    equal to `max(containment(a, b), containment(b, a))`, so it still reaches 1.0 when
    either box fully contains the other.
    """
    inter = intersection_area(a, b)
    smaller = min(area(a), area(b))
    return inter / smaller if smaller > 0 else 0.0


def centroid(box: BBox) -> Point:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def centroid_in_box(a: BBox, b: BBox) -> bool:
    """Whether A's centre point falls inside B. The cheapest robust containment test."""
    cx, cy = centroid(a)
    bx1, by1, bx2, by2 = b
    return bx1 <= cx <= bx2 and by1 <= cy <= by2


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test.

    Handles concave polygons, which matters because a hand-drawn "litter box area" is
    rarely convex.
    """
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Does the horizontal ray from (x, y) cross edge j->i?
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def normalised_distance(a: BBox, b: BBox, frame_width: float, frame_height: float) -> float:
    """Centre-to-centre distance as a fraction of the frame diagonal.

    Normalised so a "cat near door" threshold means the same thing on a 640px analysis
    frame and a 1080p one.
    """
    ax, ay = centroid(a)
    bx, by = centroid(b)
    diagonal = math.hypot(frame_width, frame_height)
    if diagonal <= 0:
        return 0.0
    return math.hypot(ax - bx, ay - by) / diagonal


@dataclass(frozen=True)
class Zone:
    """A user-drawn region, in normalised coordinates."""

    uuid: str
    name: str
    kind: str  # "box" | "polygon"
    points: List[Point]

    def to_pixels(self, frame_width: float, frame_height: float) -> List[Point]:
        return [(x * frame_width, y * frame_height) for x, y in self.points]

    def bounds(self, frame_width: float, frame_height: float) -> BBox:
        """Axis-aligned bounding box of the zone in pixels."""
        pts = self.to_pixels(frame_width, frame_height)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def contains_point(self, point: Point, frame_width: float, frame_height: float) -> bool:
        pts = self.to_pixels(frame_width, frame_height)
        if self.kind == "box":
            (x1, y1), (x2, y2) = pts[0], pts[1]
            lo_x, hi_x = min(x1, x2), max(x1, x2)
            lo_y, hi_y = min(y1, y2), max(y1, y2)
            return lo_x <= point[0] <= hi_x and lo_y <= point[1] <= hi_y
        return point_in_polygon(point, pts)

    def overlap_fraction(self, box: BBox, frame_width: float, frame_height: float) -> float:
        """Fraction of `box` inside this zone.

        Exact for box zones. For polygons this uses the zone's bounding box, which
        over-estimates on a concave shape — deliberately, because the alternative is a
        polygon-clipping dependency for a gain that a `centroid_in` rule already
        provides more cheaply and more predictably.
        """
        return containment(box, self.bounds(frame_width, frame_height))

    def overlap_ratio(self, box: BBox, frame_width: float, frame_height: float) -> float:
        """Overlap between `box` and this zone, over the smaller of the two areas.

        Same caveat as `overlap_fraction`: exact for box zones, bounding-box
        approximate for polygons.
        """
        return overlap_ratio(box, self.bounds(frame_width, frame_height))


def zone_from_row(row) -> Zone:
    """Build a Zone from a `SourceRegionRead`-shaped object or plain dict.

    Shared by zones and masks: the geometry is identical, only what the caller does with
    it differs.
    """
    if isinstance(row, dict):
        return Zone(
            uuid=row.get("uuid", ""),
            name=row.get("name", ""),
            kind=row.get("kind", "box"),
            points=[tuple(p) for p in row.get("points_json") or row.get("pointsJson") or []],
        )
    return Zone(
        uuid=row.uuid,
        name=row.name,
        kind=row.kind,
        points=[tuple(p) for p in row.points_json],
    )


def is_masked(box: BBox, masks: Sequence[Zone], frame_width: float, frame_height: float) -> bool:
    """Whether a detection should be suppressed by any of `masks`.

    Centroid-based, deliberately. Asking whether the box *overlaps* a mask would
    suppress an object merely standing next to a masked region, and for polygons the
    overlap test falls back to the mask's bounding box (see `Zone.overlap_fraction`),
    which over-suppresses on exactly the concave shapes hand-drawn masks tend to be.
    The centre point is exact for both kinds and matches the intuition that an object is
    "in" the masked area when its middle is.
    """
    if not masks:
        return False
    point = centroid(box)
    return any(mask.contains_point(point, frame_width, frame_height) for mask in masks)
