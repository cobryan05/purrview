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


def box_scale(box: BBox) -> float:
    """A single length standing in for a box's size: the mean of its two sides.

    Mean side rather than `sqrt(area)` because a detection box is often far from
    square — a standing cat is tall and narrow, and its root-area would read as much
    smaller than the animal a person sees.
    """
    x1, y1, x2, y2 = box
    return (max(0.0, x2 - x1) + max(0.0, y2 - y1)) / 2.0


def margin_pad(margin: float, a: BBox, b: BBox) -> float:
    """Pixels to grow (or shrink) every edge by, for a rule's `margin` fraction.

    Measured against the **smaller** of the two boxes, which is what makes the margin
    survive the thing we cannot control: neither box's scale is known in advance, and
    the same interaction is a hundred pixels wide at the camera and twenty at the far
    wall.

    A frame-relative pad (a fraction of the frame's short side) was the obvious first
    answer and is wrong for exactly that reason: under perspective both boxes *and the
    gap between them* shrink by the same 1/z, so a fixed pixel pad is a generous margin
    for a distant cat and a tight one for a near cat — the rule means something
    different in every part of the image. A pad proportional to the object cancels the
    1/z out, so "within half a cat of the bowl" holds at any distance.

    The *smaller* box is the ruler, not each box's own size, so a rule pairing a cat
    with a whole litter box is measured in cats: sizing each pad to its own box would
    let the larger object's halo dominate, and a room-sized zone would swallow the
    frame. This is the same choice `overlap_ratio` already makes for the same reason.

    Each box grows by *half* the margin, so the number reads as the whole gap it
    closes: margin 1.0 means "fires when they are within about one subject-width of
    each other", which is the sentence the UI puts next to the slider.
    """
    return margin * min(box_scale(a), box_scale(b)) / 2.0


def inflate(box: BBox, pad: float) -> BBox:
    """Grow `box` by `pad` on every edge; a negative `pad` shrinks it.

    This is what makes a rule fire on "nearly touching": grow both boxes and the
    ordinary overlap relations answer a proximity question with no new relation to
    reason about. A shrink deep enough to invert the box collapses it to its centre
    point instead, so a large negative margin means "essentially never" rather than a
    box with negative area that scores nonsense.
    """
    x1, y1, x2, y2 = box
    nx1, nx2 = x1 - pad, x2 + pad
    ny1, ny2 = y1 - pad, y2 + pad
    if nx1 > nx2:
        nx1 = nx2 = (x1 + x2) / 2.0
    if ny1 > ny2:
        ny1 = ny2 = (y1 + y2) / 2.0
    return (nx1, ny1, nx2, ny2)


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


def polygon_area(polygon: Sequence[Point]) -> float:
    """Absolute area of a simple polygon, by the shoelace formula.

    Sign-free on purpose: zones are drawn by hand and arrive in whichever winding the
    user happened to draw, and nothing here cares which way round they go.
    """
    n = len(polygon)
    if n < 3:
        return 0.0
    total = 0.0
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        total += (xj + xi) * (yj - yi)
        j = i
    return abs(total) / 2.0


def _axis_crossing(a: Point, b: Point, axis: int, limit: float) -> Point:
    """Where segment a→b crosses the line `axis == limit`."""
    if axis == 0:
        denom = b[0] - a[0]
        t = 0.0 if denom == 0 else (limit - a[0]) / denom
        return (limit, a[1] + t * (b[1] - a[1]))
    denom = b[1] - a[1]
    t = 0.0 if denom == 0 else (limit - a[1]) / denom
    return (a[0] + t * (b[0] - a[0]), limit)


def clip_polygon_to_box(polygon: Sequence[Point], box: BBox) -> List[Point]:
    """The part of `polygon` lying inside the axis-aligned `box`, by Sutherland–Hodgman.

    This is what makes the area relations exact for a polygon zone. They used to measure
    a polygon by `Zone.bounds`, and for the shapes people actually draw — a yard traced
    under a roofline, an L-shaped run of floor — the bounding box is most of the frame.
    A person on the pavement well outside the traced yard scored a full 1.0 `overlaps`
    against it and fired the rule, which is the bug this replaces.

    Clipping a *concave* subject against a convex window can emit degenerate edges that
    run back along the window's boundary, joining what are really separate pieces. That
    is harmless here: those edges enclose no area, so `polygon_area` of the result is
    still the sum of the pieces. Only the area is claimed to be meaningful — the returned
    ring is not guaranteed to be a simple polygon.
    """
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1 or len(polygon) < 3:
        return []
    points = list(polygon)
    # (axis, limit, keep points on the >= side)
    for axis, limit, keep_greater in ((0, x1, True), (0, x2, False), (1, y1, True), (1, y2, False)):
        if not points:
            return []
        clipped: List[Point] = []
        previous = points[-1]
        previous_in = previous[axis] >= limit if keep_greater else previous[axis] <= limit
        for current in points:
            current_in = current[axis] >= limit if keep_greater else current[axis] <= limit
            if current_in != previous_in:
                clipped.append(_axis_crossing(previous, current, axis, limit))
            if current_in:
                clipped.append(current)
            previous, previous_in = current, current_in
        points = clipped
    return points


def point_to_segment_distance(point: Point, a: Point, b: Point) -> float:
    """Shortest distance from `point` to the segment a→b."""
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def point_to_polygon_distance(point: Point, polygon: Sequence[Point]) -> float:
    """Distance from `point` to a polygon's boundary, or 0.0 if it is inside.

    Exists so a margin can widen `centroid_in` on a hand-drawn polygon. The alternative
    — offsetting the polygon's vertices — distorts exactly the concave shapes those
    zones tend to be, while "is the centre within `pad` of the zone" is the same
    question, answered exactly.
    """
    if len(polygon) < 2:
        return math.inf
    if point_in_polygon(point, polygon):
        return 0.0
    n = len(polygon)
    return min(point_to_segment_distance(point, polygon[i], polygon[(i + 1) % n]) for i in range(n))


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
        """Axis-aligned bounding box of the zone in pixels.

        A *ruler*, not a stand-in for the zone. The area relations no longer measure a
        polygon by this — they used to, and a traced yard whose bounding box is most of
        the frame fired on anything anywhere below the roofline (see
        `clip_polygon_to_box`). What still legitimately uses it is `margin_pad`, which
        only needs the zone's rough scale, and the `distance` relation, which compares
        centre points.
        """
        pts = self.to_pixels(frame_width, frame_height)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def pixel_area(self, frame_width: float, frame_height: float) -> float:
        """The zone's own area in pixels — the polygon's, not its bounding box's."""
        if self.kind == "box":
            return area(self.bounds(frame_width, frame_height))
        return polygon_area(self.to_pixels(frame_width, frame_height))

    def intersection_area_with_box(self, box: BBox, frame_width: float, frame_height: float) -> float:
        """Exact area of `box` ∩ this zone, in pixels, for both kinds."""
        if self.kind == "box":
            return intersection_area(box, self.bounds(frame_width, frame_height))
        return polygon_area(clip_polygon_to_box(self.to_pixels(frame_width, frame_height), box))

    def contains_point(self, point: Point, frame_width: float, frame_height: float) -> bool:
        pts = self.to_pixels(frame_width, frame_height)
        if self.kind == "box":
            (x1, y1), (x2, y2) = pts[0], pts[1]
            lo_x, hi_x = min(x1, x2), max(x1, x2)
            lo_y, hi_y = min(y1, y2), max(y1, y2)
            return lo_x <= point[0] <= hi_x and lo_y <= point[1] <= hi_y
        return point_in_polygon(point, pts)

    def contains_point_within(self, point: Point, pad: float, frame_width: float, frame_height: float) -> bool:
        """Whether `point` is inside this zone, or within `pad` pixels of its edge.

        `pad` may be negative, which erodes the zone: the point must then be that far
        *inside* it. Exact for both kinds — see `point_to_polygon_distance` for why a
        distance test is used rather than an offset polygon.
        """
        if pad == 0:
            return self.contains_point(point, frame_width, frame_height)
        pts = self.to_pixels(frame_width, frame_height)
        if self.kind == "box":
            (x1, y1), (x2, y2) = pts[0], pts[1]
            box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            bx1, by1, bx2, by2 = inflate(box, pad)
            return bx1 <= point[0] <= bx2 and by1 <= point[1] <= by2
        distance = point_to_polygon_distance(point, pts)
        if pad > 0:
            return distance <= pad
        # Eroding: inside already, and no closer than |pad| to any edge.
        if distance > 0:
            return False
        n = len(pts)
        if n < 2:
            return False
        edge_distance = min(point_to_segment_distance(point, pts[i], pts[(i + 1) % n]) for i in range(n))
        return edge_distance >= -pad


def zone_area_terms(
    zone: Zone,
    box: BBox,
    margin: float,
    frame_width: float,
    frame_height: float,
) -> Tuple[float, float, float]:
    """`(intersection, subject area, zone area)` in pixels, with `margin` applied.

    The three area relations — `contains`, `overlaps`, `iou` — are all ratios of these
    same three numbers against a zone, so they are computed once here rather than three
    times at the call site with three chances to disagree about the margin.

    The margin lands asymmetrically, by kind, and that is deliberate:

    * A **box** zone can be grown exactly, so subject and zone each take half, exactly as
      a pair of detection boxes do. Nothing about box zones changes.
    * A **polygon** cannot be offset without distorting exactly the concave shapes these
      zones tend to be — the same reason `contains_point_within` measures a distance
      rather than moving the vertices — so the whole margin goes on the subject box
      instead. It closes the same total gap, and it is the choice `centroid_in` already
      makes for a polygon zone.

    With `margin == 0`, which is the default and what nearly every rule uses, the two
    paths are identical anyway.
    """
    bounds = zone.bounds(frame_width, frame_height)
    pad = margin_pad(margin, box, bounds) if margin else 0.0
    if zone.kind == "box":
        subject = inflate(box, pad) if pad else box
        target = inflate(bounds, pad) if pad else bounds
        return intersection_area(subject, target), area(subject), area(target)
    subject = inflate(box, 2 * pad) if pad else box
    return (
        zone.intersection_area_with_box(subject, frame_width, frame_height),
        area(subject),
        zone.pixel_area(frame_width, frame_height),
    )


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
    overlap test falls back to the mask's bounding box (see `Zone.bounds`),
    which over-suppresses on exactly the concave shapes hand-drawn masks tend to be.
    The centre point is exact for both kinds and matches the intuition that an object is
    "in" the masked area when its middle is.
    """
    if not masks:
        return False
    point = centroid(box)
    return any(mask.contains_point(point, frame_width, frame_height) for mask in masks)
