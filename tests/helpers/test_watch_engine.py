"""Unit tests for watch geometry, rule hysteresis and the engine.

Everything here is pure, which is the point: this is where the feature's real bugs
live, and a synthetic detection sequence pins them down in milliseconds.

The hysteresis tests are the important ones. Without min-duration / rearm-delay /
max-duration a raw threshold flaps many times per second, and the visible symptom is
an MQTT storm plus a dozen three-second clips for one litter-box visit.
"""

from dataclasses import replace

import pytest

from apps.helpers.tracking.iouTracker import IouTracker
from apps.helpers.tracking.tracker import Detection, Track, merge_duplicate_tracks
from apps.helpers.watch.engine import WatchEngine
from apps.helpers.watch.events import EventCloseReason, EventPhase
from apps.helpers.watch.geometry import (
    Zone,
    area,
    box_scale,
    centroid,
    centroid_in_box,
    containment,
    inflate,
    intersection_area,
    iou,
    margin_pad,
    normalised_distance,
    overlap_ratio,
    point_in_polygon,
    point_to_polygon_distance,
    polygon_area,
    zone_area_terms,
)
from apps.helpers.watch.rules import (
    GRANULARITY_RULE,
    RuleMachine,
    RuleSpec,
    RuleState,
    evaluate_predicate,
)

CAT = "label-cat"
BOX = "label-litterbox"
ZONE_UUID = "zone-litterbox"

# A zone covering the middle of the frame, normalised.
LITTER_ZONE = Zone(uuid=ZONE_UUID, name="litter box area", kind="box", points=[(0.4, 0.4), (0.8, 0.8)])
FRAME_W, FRAME_H = 1000.0, 1000.0


def _track(track_id=1, bbox=(0, 0, 20, 20), label=CAT, conf=0.9) -> Track:
    return Track(track_id=track_id, bbox=bbox, label_uuid=label, label_name=label, confidence=conf)


def _rule(**overrides) -> RuleSpec:
    """Build a RuleSpec, accepting either a single ref or a list for each subject slot.

    Slots are lists now ("any of these cats"), but most tests only care about one ref and
    read far better naming it directly, so a bare string is wrapped here.
    """
    spec = dict(
        uuid="rule-1",
        name="cat using litter box",
        subject_a_kind="label",
        subject_a_refs=[CAT],
        relation="contains",
        subject_b_kind="zone",
        subject_b_refs=[ZONE_UUID],
        threshold=0.6,
        min_duration_s=2.0,
        rearm_delay_s=10.0,
        max_duration_s=300.0,
    )
    for side in ("a", "b"):
        if f"subject_{side}_ref" in overrides:
            ref = overrides.pop(f"subject_{side}_ref")
            overrides[f"subject_{side}_refs"] = [] if ref is None else [ref]
    spec.update(overrides)
    return RuleSpec(**spec)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────


class TestGeometry:
    def test_containment_beats_iou_for_a_small_box_in_a_large_one(self):
        """The core argument for `contains` being the default relation."""
        cat = (450.0, 450.0, 500.0, 500.0)
        litterbox = (400.0, 400.0, 800.0, 800.0)
        assert containment(cat, litterbox) == 1.0
        assert iou(cat, litterbox) < 0.05

    def test_containment_is_partial_when_half_outside(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 0.0, 15.0, 10.0)
        assert containment(a, b) == pytest.approx(0.5)

    def test_containment_of_a_zero_area_box_is_zero(self):
        assert containment((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0

    def test_intersection_of_disjoint_boxes_is_zero(self):
        assert intersection_area((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0

    def test_centroid_is_the_box_middle(self):
        assert centroid((0.0, 0.0, 10.0, 20.0)) == (5.0, 10.0)

    def test_centroid_in_box(self):
        assert centroid_in_box((4, 4, 6, 6), (0, 0, 10, 10)) is True
        assert centroid_in_box((40, 40, 60, 60), (0, 0, 10, 10)) is False

    def test_normalised_distance_is_resolution_independent(self):
        """A 'cat near door' threshold must mean the same on 640px and 1080p."""
        small = normalised_distance((0, 0, 10, 10), (100, 0, 110, 10), 640, 480)
        large = normalised_distance((0, 0, 20, 20), (200, 0, 220, 20), 1280, 960)
        assert small == pytest.approx(large, rel=1e-6)

    def test_distance_of_zero_sized_frame_does_not_divide_by_zero(self):
        assert normalised_distance((0, 0, 1, 1), (2, 2, 3, 3), 0, 0) == 0.0


class TestOverlapRatio:
    """The 'cat drinking from the water bowl' case, which the other relations miss."""

    # A cat lapping at a bowl: the cat's box laps over the bowl's left edge. It is not
    # inside the bowl, and it is several times the bowl's size.
    CAT_AT_BOWL = (100.0, 100.0, 300.0, 300.0)
    BOWL = (280.0, 200.0, 360.0, 280.0)

    def test_the_other_relations_miss_a_cat_drinking_from_a_bowl(self):
        """Why this relation had to exist: both existing measures read as 'no'."""
        assert iou(self.CAT_AT_BOWL, self.BOWL) < 0.05
        assert containment(self.CAT_AT_BOWL, self.BOWL) < 0.05

    def test_overlap_ratio_registers_the_same_interaction(self):
        # 20x80 of overlap against the bowl's 80x80 — a quarter of the bowl covered.
        assert overlap_ratio(self.CAT_AT_BOWL, self.BOWL) == pytest.approx(0.25)

    def test_it_is_symmetric(self):
        """Unlike containment, which side you name first must not change the answer."""
        assert overlap_ratio(self.CAT_AT_BOWL, self.BOWL) == pytest.approx(overlap_ratio(self.BOWL, self.CAT_AT_BOWL))

    def test_it_equals_the_better_of_the_two_containments(self):
        a, b = self.CAT_AT_BOWL, self.BOWL
        assert overlap_ratio(a, b) == pytest.approx(max(containment(a, b), containment(b, a)))

    def test_full_containment_still_scores_one(self):
        """It must not *lose* the containment case it generalises."""
        cat = (450.0, 450.0, 500.0, 500.0)
        litterbox = (400.0, 400.0, 800.0, 800.0)
        assert overlap_ratio(cat, litterbox) == 1.0

    def test_disjoint_boxes_score_zero(self):
        assert overlap_ratio((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0

    def test_touching_edges_do_not_count_as_overlap(self):
        """Boxes that merely abut share no area, so nothing should fire."""
        assert overlap_ratio((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_zero_area_box_does_not_divide_by_zero(self):
        assert overlap_ratio((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


class TestPointInPolygon:
    TRIANGLE = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]

    def test_inside(self):
        assert point_in_polygon((5.0, 2.0), self.TRIANGLE) is True

    def test_outside(self):
        assert point_in_polygon((0.0, 9.0), self.TRIANGLE) is False

    def test_concave_polygon_excludes_the_notch(self):
        # A 'C' shape: the notch on the right must not count as inside.
        c_shape = [(0, 0), (10, 0), (10, 3), (3, 3), (3, 7), (10, 7), (10, 10), (0, 10)]
        assert point_in_polygon((1.0, 5.0), c_shape) is True
        assert point_in_polygon((7.0, 5.0), c_shape) is False

    def test_degenerate_polygon_is_never_inside(self):
        assert point_in_polygon((0.0, 0.0), [(0, 0), (1, 1)]) is False


class TestZone:
    def test_normalised_points_scale_to_pixels(self):
        """Zones are stored normalised so they survive a resolution change."""
        assert LITTER_ZONE.bounds(1000, 1000) == (400.0, 400.0, 800.0, 800.0)
        assert LITTER_ZONE.bounds(500, 500) == (200.0, 200.0, 400.0, 400.0)

    def test_contains_point_for_a_box_zone(self):
        assert LITTER_ZONE.contains_point((500, 500), FRAME_W, FRAME_H) is True
        assert LITTER_ZONE.contains_point((100, 100), FRAME_W, FRAME_H) is False

    def test_box_zone_handles_reversed_corner_order(self):
        # A user dragging bottom-right to top-left produces reversed points.
        reversed_zone = Zone("z", "z", "box", [(0.8, 0.8), (0.4, 0.4)])
        assert reversed_zone.contains_point((500, 500), FRAME_W, FRAME_H) is True

    def test_polygon_zone_contains_point(self):
        poly = Zone("z", "z", "polygon", [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)])
        assert poly.contains_point((500, 200), FRAME_W, FRAME_H) is True
        assert poly.contains_point((50, 900), FRAME_W, FRAME_H) is False


# A yard traced under a roofline: its bounding box is the WHOLE frame, but the top-left
# corner — the pavement beyond the yard — is outside it. This is the shape that made the
# area relations fire on anything anywhere in the picture.
YARD_ZONE = Zone(
    uuid=ZONE_UUID,
    name="yard",
    kind="polygon",
    points=[(0.0, 0.5), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
)

# A 'C' opening to the right: the notch is inside the bounding box and outside the zone.
C_ZONE = Zone(
    uuid=ZONE_UUID,
    name="hallway",
    kind="polygon",
    points=[(0, 0), (1, 0), (1, 0.3), (0.3, 0.3), (0.3, 0.7), (1, 0.7), (1, 1), (0, 1)],
)


def _zone_score(zone, box, relation, margin=0.0):
    """The area relations' score for one box against one zone, as the engine computes it."""
    inter, subject_area, zone_area = zone_area_terms(zone, box, margin, FRAME_W, FRAME_H)
    if relation == "iou":
        union = subject_area + zone_area - inter
        return inter / union if union > 0 else 0.0
    if relation == "overlaps":
        smaller = min(subject_area, zone_area)
        return inter / smaller if smaller > 0 else 0.0
    return inter / subject_area if subject_area > 0 else 0.0


class TestPolygonZoneAreas:
    """The area relations must measure the polygon, not its bounding box.

    They used to measure `Zone.bounds`, and every test here fails against that: a yard
    zone's bounding box is the whole frame, so a person on the pavement scored a perfect
    1.0 and fired the rule.
    """

    PAVEMENT = (50.0, 50.0, 150.0, 150.0)  # inside the zone's bounds, outside the zone
    IN_YARD = (400.0, 700.0, 600.0, 900.0)

    def test_bounding_box_would_have_swallowed_the_pavement(self):
        """The premise: the bounds *do* contain the box that must not fire."""
        assert overlap_ratio(self.PAVEMENT, YARD_ZONE.bounds(FRAME_W, FRAME_H)) == 1.0

    @pytest.mark.parametrize("relation", ["contains", "overlaps", "iou"])
    def test_outside_the_polygon_scores_zero(self, relation):
        assert _zone_score(YARD_ZONE, self.PAVEMENT, relation) == 0.0

    @pytest.mark.parametrize("relation", ["contains", "overlaps"])
    def test_inside_the_polygon_still_scores_one(self, relation):
        assert _zone_score(YARD_ZONE, self.IN_YARD, relation) == pytest.approx(1.0)

    def test_partial_overlap_is_the_exact_clipped_area(self):
        # The zone's edge runs (0, 500) → (200, 400) across this box, cutting off a
        # 10 000 px triangle of its 40 000 px area.
        straddling = (0.0, 400.0, 200.0, 600.0)
        assert _zone_score(YARD_ZONE, straddling, "contains") == pytest.approx(0.75)

    def test_a_box_in_a_concave_notch_scores_zero(self):
        assert _zone_score(C_ZONE, (500.0, 400.0, 700.0, 600.0), "contains") == 0.0

    def test_two_disjoint_pieces_sum_rather_than_cancel(self):
        """Clipping a concave zone can emit degenerate joining edges; they enclose no area."""
        # Spans both arms of the C and the notch between them: 2 × (500 × 100) of
        # 500 × 600.
        across_both_arms = (500.0, 200.0, 1000.0, 800.0)
        assert _zone_score(C_ZONE, across_both_arms, "contains") == pytest.approx(1 / 3)

    def test_zone_area_is_the_polygon_not_the_bounds(self):
        assert YARD_ZONE.pixel_area(FRAME_W, FRAME_H) == pytest.approx(750_000.0)
        assert area(YARD_ZONE.bounds(FRAME_W, FRAME_H)) == pytest.approx(1_000_000.0)

    def test_margin_reaches_across_the_polygon_edge(self):
        """A subject just short of the zone fires only once the margin closes the gap."""
        # The zone's edge runs y = 500 - x/2, so it sits at y = 200 by the time this box's
        # right side is reached — 20 px below the box's bottom.
        just_short = (400.0, 60.0, 600.0, 180.0)
        assert _zone_score(YARD_ZONE, just_short, "contains", margin=0.0) == 0.0
        assert _zone_score(YARD_ZONE, just_short, "contains", margin=1.0) > 0.0

    def test_box_zones_are_unaffected(self):
        """A box zone is its own bounds, so nothing about it changes."""
        box = (450.0, 450.0, 550.0, 550.0)
        assert _zone_score(LITTER_ZONE, box, "contains") == pytest.approx(
            containment(box, LITTER_ZONE.bounds(FRAME_W, FRAME_H))
        )
        pad = margin_pad(0.5, box, LITTER_ZONE.bounds(FRAME_W, FRAME_H))
        assert _zone_score(LITTER_ZONE, box, "iou", margin=0.5) == pytest.approx(
            iou(inflate(box, pad), inflate(LITTER_ZONE.bounds(FRAME_W, FRAME_H), pad))
        )

    def test_the_rule_that_shipped_this_bug(self):
        """End to end: "person overlaps the yard" must not fire on the pavement."""
        spec = _rule(relation="overlaps", threshold=0.6)
        zones = {ZONE_UUID: YARD_ZONE}
        assert not evaluate_predicate(spec, [_track(bbox=self.PAVEMENT)], zones, FRAME_W, FRAME_H).holds
        assert evaluate_predicate(spec, [_track(bbox=self.IN_YARD)], zones, FRAME_W, FRAME_H).holds


class TestPolygonArea:
    def test_shoelace_matches_a_known_rectangle(self):
        assert polygon_area([(0, 0), (10, 0), (10, 4), (0, 4)]) == pytest.approx(40.0)

    def test_winding_direction_does_not_matter(self):
        clockwise = [(0, 0), (0, 4), (10, 4), (10, 0)]
        assert polygon_area(clockwise) == pytest.approx(40.0)

    def test_degenerate_polygon_has_no_area(self):
        assert polygon_area([(0, 0), (1, 1)]) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Predicates
# ─────────────────────────────────────────────────────────────────────────────


class TestPredicates:
    def test_cat_inside_the_zone_fires(self):
        track = _track(bbox=(450, 450, 550, 550))
        result = evaluate_predicate(_rule(), [track], {ZONE_UUID: LITTER_ZONE}, FRAME_W, FRAME_H)
        assert result.holds is True
        assert result.track_ids == [1]

    def test_cat_outside_the_zone_does_not_fire(self):
        track = _track(bbox=(10, 10, 60, 60))
        result = evaluate_predicate(_rule(), [track], {ZONE_UUID: LITTER_ZONE}, FRAME_W, FRAME_H)
        assert result.holds is False

    def test_no_tracks_does_not_fire(self):
        assert evaluate_predicate(_rule(), [], {ZONE_UUID: LITTER_ZONE}, FRAME_W, FRAME_H).holds is False

    def test_a_dangling_zone_reference_never_fires(self):
        """Config-time validation catches this; at run time it must simply be false."""
        track = _track(bbox=(450, 450, 550, 550))
        result = evaluate_predicate(_rule(), [track], {}, FRAME_W, FRAME_H)
        assert result.holds is False

    def test_wrong_label_does_not_fire(self):
        track = _track(bbox=(450, 450, 550, 550), label="label-dog")
        assert evaluate_predicate(_rule(), [track], {ZONE_UUID: LITTER_ZONE}, FRAME_W, FRAME_H).holds is False

    def test_present_is_arity_one(self):
        spec = _rule(relation="present", subject_b_kind="none", subject_b_ref=None, threshold=0.5)
        assert evaluate_predicate(spec, [_track(conf=0.9)], {}, FRAME_W, FRAME_H).holds is True
        assert evaluate_predicate(spec, [_track(conf=0.2)], {}, FRAME_W, FRAME_H).holds is False

    def test_centroid_in_fires_only_when_the_centre_is_inside(self):
        spec = _rule(relation="centroid_in")
        inside = _track(bbox=(590, 590, 610, 610))  # centre at 600,600 — inside
        outside = _track(bbox=(780, 780, 900, 900))  # centre at 840,840 — outside
        zones = {ZONE_UUID: LITTER_ZONE}
        assert evaluate_predicate(spec, [inside], zones, FRAME_W, FRAME_H).holds is True
        assert evaluate_predicate(spec, [outside], zones, FRAME_W, FRAME_H).holds is False

    def test_overlaps_fires_on_a_cat_lapping_over_a_zone_edge(self):
        """The zone-target branch of the water-bowl case.

        The zone spans 400..800; the cat spans 100..450, so only a sliver is inside.
        `contains` reads that as a tiny fraction of a big cat and stays silent.
        """
        cat = _track(bbox=(100, 400, 450, 800))
        zones = {ZONE_UUID: LITTER_ZONE}
        assert evaluate_predicate(_rule(relation="contains", threshold=0.6), [cat], zones, FRAME_W, FRAME_H).holds is (
            False
        )
        assert evaluate_predicate(_rule(relation="overlaps", threshold=0.1), [cat], zones, FRAME_W, FRAME_H).holds is (
            True
        )

    def test_overlaps_fires_between_two_tracked_labels(self):
        """The track-vs-track branch: a cat drinking from a detected water bowl."""
        cat = _track(track_id=1, bbox=(100, 100, 300, 300), label=CAT)
        bowl = _track(track_id=2, bbox=(280, 200, 360, 280), label=BOX)
        spec = _rule(relation="overlaps", subject_b_kind="label", subject_b_ref=BOX, threshold=0.2)
        result = evaluate_predicate(spec, [cat, bowl], {}, FRAME_W, FRAME_H)
        assert result.holds is True
        assert result.track_ids == [1]

    def test_overlaps_respects_its_threshold(self):
        """A cat merely brushing the bowl's corner should not read as drinking."""
        cat = _track(track_id=1, bbox=(100, 100, 290, 290), label=CAT)
        bowl = _track(track_id=2, bbox=(280, 280, 360, 360), label=BOX)
        spec = _rule(relation="overlaps", subject_b_kind="label", subject_b_ref=BOX, threshold=0.2)
        assert evaluate_predicate(spec, [cat, bowl], {}, FRAME_W, FRAME_H).holds is False

    def test_overlaps_does_not_match_a_track_against_itself(self):
        cat = _track(track_id=1, bbox=(100, 100, 300, 300), label=CAT)
        spec = _rule(relation="overlaps", subject_b_kind="label", subject_b_ref=CAT, threshold=0.1)
        assert evaluate_predicate(spec, [cat], {}, FRAME_W, FRAME_H).holds is False

    def test_centroid_in_fires_even_when_containment_would_not(self):
        """A cat much larger than the zone still counts as 'in' it by centroid."""
        big_cat = _track(bbox=(300, 300, 900, 900))
        zones = {ZONE_UUID: LITTER_ZONE}
        assert evaluate_predicate(_rule(relation="centroid_in"), [big_cat], zones, FRAME_W, FRAME_H).holds is True
        assert evaluate_predicate(_rule(relation="contains"), [big_cat], zones, FRAME_W, FRAME_H).holds is False

    def test_distance_fires_when_close_enough(self):
        spec = _rule(relation="distance", threshold=0.2)
        near = _track(bbox=(560, 560, 580, 580))
        far = _track(bbox=(0, 0, 20, 20))
        zones = {ZONE_UUID: LITTER_ZONE}
        assert evaluate_predicate(spec, [near], zones, FRAME_W, FRAME_H).holds is True
        assert evaluate_predicate(spec, [far], zones, FRAME_W, FRAME_H).holds is False

    def test_label_to_label_interaction(self):
        spec = _rule(subject_b_kind="label", subject_b_ref=BOX, relation="contains")
        cat = _track(track_id=1, bbox=(450, 450, 500, 500), label=CAT)
        litterbox = _track(track_id=2, bbox=(400, 400, 800, 800), label=BOX)
        assert evaluate_predicate(spec, [cat, litterbox], {}, FRAME_W, FRAME_H).holds is True

    def test_an_object_never_interacts_with_itself(self):
        spec = _rule(subject_a_kind="any", subject_a_ref=None, subject_b_kind="label", subject_b_ref=CAT)
        cat = _track(track_id=1, bbox=(450, 450, 500, 500), label=CAT)
        assert evaluate_predicate(spec, [cat], {}, FRAME_W, FRAME_H).holds is False

    def test_iou_relation_still_available_for_same_size_objects(self):
        spec = _rule(subject_b_kind="label", subject_b_ref=BOX, relation="iou", threshold=0.4)
        a = _track(track_id=1, bbox=(0, 0, 100, 100), label=CAT)
        b = _track(track_id=2, bbox=(10, 10, 110, 110), label=BOX)
        assert evaluate_predicate(spec, [a, b], {}, FRAME_W, FRAME_H).holds is True


# ─────────────────────────────────────────────────────────────────────────────
# Hysteresis
# ─────────────────────────────────────────────────────────────────────────────


class TestRuleMachine:
    @staticmethod
    def _run(machine, sequence):
        """Feed (holds, now) pairs; return every emitted signal."""
        from apps.helpers.watch.rules import PredicateResult

        emitted = []
        for holds, now in sequence:
            emitted.extend(machine.step(PredicateResult(holds=holds, track_ids=[1] if holds else []), now))
        return emitted

    def test_no_event_before_min_duration(self):
        machine = RuleMachine(_rule(min_duration_s=2.0))
        signals = self._run(machine, [(True, 0.0), (True, 1.0), (True, 1.9)])
        assert signals == []
        assert machine.state is RuleState.PENDING

    def test_event_starts_after_min_duration(self):
        machine = RuleMachine(_rule(min_duration_s=2.0))
        signals = self._run(machine, [(True, 0.0), (True, 1.0), (True, 2.5)])
        assert [s.phase for s in signals] == [EventPhase.START]
        assert machine.state is RuleState.ACTIVE

    def test_a_brief_blip_never_creates_an_event(self):
        """The single most important guard: one flicker must not fire anything."""
        machine = RuleMachine(_rule(min_duration_s=2.0))
        signals = self._run(machine, [(True, 0.0), (False, 0.2), (True, 0.4), (False, 0.6)])
        assert signals == []
        assert machine.state is RuleState.IDLE

    def test_event_start_is_backdated_to_when_the_predicate_first_held(self):
        """Pre-roll is measured from here, so it must not be the confirmation moment."""
        machine = RuleMachine(_rule(min_duration_s=2.0))
        signals = self._run(machine, [(True, 100.0), (True, 102.5)])
        assert signals[0].started_at == 100.0
        assert signals[0].at == 102.5

    def test_event_does_not_end_until_the_rearm_delay_elapses(self):
        machine = RuleMachine(_rule(min_duration_s=1.0, rearm_delay_s=10.0))
        signals = self._run(machine, [(True, 0.0), (True, 2.0), (False, 3.0), (False, 8.0)])
        assert [s.phase for s in signals] == [EventPhase.START]
        assert machine.state is RuleState.CLOSING

    def test_event_ends_after_the_rearm_delay(self):
        machine = RuleMachine(_rule(min_duration_s=1.0, rearm_delay_s=10.0))
        signals = self._run(machine, [(True, 0.0), (True, 2.0), (False, 3.0), (False, 14.0)])
        assert [s.phase for s in signals] == [EventPhase.START, EventPhase.END]
        assert machine.state is RuleState.IDLE

    def test_a_flapping_predicate_produces_exactly_one_event(self):
        """Twelve clips of one litter-box visit is the failure this prevents."""
        machine = RuleMachine(_rule(min_duration_s=1.0, rearm_delay_s=10.0))
        sequence = [(True, 0.0), (True, 2.0)]
        # 20 flips over the next 10 seconds, all inside the rearm window.
        for i in range(20):
            sequence.append((i % 2 == 0, 2.5 + i * 0.5))
        sequence.append((False, 40.0))
        signals = self._run(machine, sequence)
        assert [s.phase for s in signals] == [EventPhase.START, EventPhase.END]

    def test_flapping_is_counted_for_the_tuning_hint(self):
        machine = RuleMachine(_rule(min_duration_s=1.0, rearm_delay_s=10.0))
        sequence = [(True, 0.0), (True, 2.0)]
        for i in range(10):
            sequence.append((i % 2 == 0, 2.5 + i * 0.2))
        self._run(machine, sequence)
        assert machine.flap_count > 0

    def test_a_second_event_can_start_after_the_first_fully_closes(self):
        machine = RuleMachine(_rule(min_duration_s=1.0, rearm_delay_s=5.0))
        signals = self._run(
            machine,
            [(True, 0.0), (True, 2.0), (False, 3.0), (False, 10.0), (True, 11.0), (True, 13.0)],
        )
        assert [s.phase for s in signals] == [EventPhase.START, EventPhase.END, EventPhase.START]

    def test_max_duration_closes_a_stuck_event(self):
        """A stuck detection must not hold a recording open forever."""
        machine = RuleMachine(_rule(min_duration_s=1.0, max_duration_s=30.0))
        signals = self._run(machine, [(True, 0.0), (True, 2.0), (True, 40.0)])
        assert [s.phase for s in signals] == [EventPhase.START, EventPhase.END]
        assert signals[-1].truncated is True

    def test_a_capped_event_does_not_reopen_while_the_subject_is_still_there(self):
        """The cap must not become a metronome.

        Observed on a real "Car Arrives" rule (present, 30 s cap): every event closed
        with reason `cap` and another opened seconds later, so a car parked in the
        garage produced a clip every half minute all evening. Resetting to IDLE put the
        machine back in front of the same still-true predicate, which is the runaway the
        cap exists to prevent rather than a way of serving it.
        """
        machine = RuleMachine(_rule(min_duration_s=1.0, rearm_delay_s=10.0, max_duration_s=30.0))
        # The subject never leaves, for ten minutes after the cap fires.
        sequence = [(True, 0.0), (True, 2.0), (True, 40.0)] + [(True, t) for t in range(41, 600, 5)]
        signals = self._run(machine, sequence)

        assert [s.phase for s in signals] == [EventPhase.START, EventPhase.END]
        assert signals[-1].close_reason is EventCloseReason.CAP

    def test_a_capped_rule_fires_again_once_the_subject_leaves_and_returns(self):
        """The other half: latching after a cap must not disable the rule for good."""
        machine = RuleMachine(_rule(min_duration_s=1.0, rearm_delay_s=10.0, max_duration_s=30.0))
        signals = self._run(
            machine,
            [
                (True, 0.0),
                (True, 2.0),
                (True, 40.0),  # cap closes it while the car is still there
                (True, 50.0),  # still there — nothing may open
                (False, 60.0),  # gone
                (False, 75.0),  # gone long enough to rearm
                (True, 80.0),  # comes back
                (True, 82.0),
            ],
        )

        assert [s.phase for s in signals] == [EventPhase.START, EventPhase.END, EventPhase.START]

    def test_a_cap_that_fires_while_closing_carries_the_countdown_over(self):
        """A subject that has already left must not serve the rearm delay twice.

        Capping out of CLOSING is the one case where the wait is already part-served, and
        restarting it would hold the rule shut for a second window after the garage is
        empty.
        """
        machine = RuleMachine(_rule(min_duration_s=1.0, rearm_delay_s=10.0, max_duration_s=30.0))
        signals = self._run(
            machine,
            [
                (True, 0.0),
                (True, 2.0),
                (False, 25.0),  # left; rearm countdown starts here
                (False, 31.0),  # cap fires first, mid-countdown
                (False, 36.0),  # 11s since it left — the carried countdown has elapsed
                (True, 40.0),
                (True, 42.0),
            ],
        )

        assert [s.phase for s in signals] == [EventPhase.START, EventPhase.END, EventPhase.START]

    def test_a_zero_max_duration_disables_the_cap(self):
        """0 means "never expire", for a rule whose event should last as long as its
        subject is there — a timelapse of a cat asleep in the window, say.

        Read literally, 0 was catastrophic rather than merely wrong: `_event_started_at`
        is `min_duration_s` in the past by the time the cap is checked, so
        `now - started >= 0` held on the very first step and END was emitted in the same
        step as START, forever. The event never lasted a frame, so nothing recorded and
        nothing explained why.
        """
        machine = RuleMachine(_rule(min_duration_s=1.0, max_duration_s=0.0))
        signals = self._run(machine, [(True, 0.0), (True, 2.0), (True, 5000.0), (True, 100000.0)])
        assert [s.phase for s in signals] == [EventPhase.START]

    def test_a_negative_max_duration_also_disables_the_cap(self):
        """Same reading the recording retention settings give a non-positive bound, so a
        hand-edited row cannot produce the instant-close behaviour by another spelling."""
        machine = RuleMachine(_rule(min_duration_s=1.0, max_duration_s=-1.0))
        signals = self._run(machine, [(True, 0.0), (True, 2.0), (True, 9000.0)])
        assert [s.phase for s in signals] == [EventPhase.START]

    def test_a_zero_cap_still_ends_when_the_predicate_clears(self):
        """ "No cap" must not mean "never closes" — the hysteresis still owns the end."""
        machine = RuleMachine(_rule(min_duration_s=1.0, rearm_delay_s=2.0, max_duration_s=0.0))
        signals = self._run(machine, [(True, 0.0), (True, 2.0), (False, 500.0), (False, 503.0)])
        assert [s.phase for s in signals] == [EventPhase.START, EventPhase.END]
        assert signals[-1].truncated is False

    def test_a_normal_end_is_not_marked_truncated(self):
        machine = RuleMachine(_rule(min_duration_s=1.0, rearm_delay_s=2.0))
        signals = self._run(machine, [(True, 0.0), (True, 2.0), (False, 3.0), (False, 6.0)])
        assert signals[-1].truncated is False

    def test_a_disabled_rule_emits_nothing(self):
        machine = RuleMachine(_rule(enabled=False, min_duration_s=1.0))
        assert self._run(machine, [(True, 0.0), (True, 5.0)]) == []

    def test_disabling_a_rule_mid_event_closes_it(self):
        """Otherwise the event is stranded open forever."""
        machine = RuleMachine(_rule(min_duration_s=1.0))
        self._run(machine, [(True, 0.0), (True, 2.0)])
        machine.spec.enabled = False
        signals = self._run(machine, [(True, 3.0)])
        assert [s.phase for s in signals] == [EventPhase.END]

    def test_force_close_closes_an_open_event(self):
        machine = RuleMachine(_rule(min_duration_s=1.0))
        self._run(machine, [(True, 0.0), (True, 2.0)])
        (signal,) = machine.force_close(5.0)
        assert signal.phase is EventPhase.END
        assert signal.truncated is True

    def test_a_forced_close_is_not_reported_as_the_cap(self):
        """The two cut-short endings must stay tellable apart.

        They were one boolean, so a watcher restarting — which is what saving its own
        config does — closed every open event and each was reported as "capped by max
        duration", on rules whose cap was `0`, i.e. disabled. The user goes looking at a
        setting that cannot have caused it. Asserting the *difference* rather than one
        value is the point: collapsing them back into one flag fails this.
        """
        machine = RuleMachine(_rule(min_duration_s=1.0, max_duration_s=0.0))
        self._run(machine, [(True, 0.0), (True, 2.0)])
        (forced,) = machine.force_close(5.0)
        assert forced.close_reason is EventCloseReason.FORCED

        capped = RuleMachine(_rule(min_duration_s=1.0, max_duration_s=30.0))
        signals = self._run(capped, [(True, 0.0), (True, 2.0), (True, 40.0)])
        assert signals[-1].close_reason is EventCloseReason.CAP

    def test_an_ordinary_end_reports_the_predicate(self):
        """And is therefore not truncated at all — `truncated` is derived from the
        reason, so the flag and the reason cannot drift apart."""
        machine = RuleMachine(_rule(min_duration_s=1.0, rearm_delay_s=2.0))
        signals = self._run(machine, [(True, 0.0), (True, 2.0), (False, 3.0), (False, 6.0)])
        assert signals[-1].close_reason is EventCloseReason.PREDICATE
        assert signals[-1].truncated is False

    def test_disabling_a_rule_mid_event_is_a_forced_close(self):
        """The subject may still be right there; the rule simply stopped watching."""
        machine = RuleMachine(_rule(min_duration_s=1.0))
        self._run(machine, [(True, 0.0), (True, 2.0)])
        machine.spec.enabled = False
        signals = self._run(machine, [(True, 3.0)])
        assert signals[-1].close_reason is EventCloseReason.FORCED

    def test_force_close_on_an_idle_rule_emits_nothing(self):
        assert RuleMachine(_rule()).force_close(1.0) == []

    def test_peak_confidence_is_retained_across_the_event(self):
        from apps.helpers.watch.rules import PredicateResult

        machine = RuleMachine(_rule(min_duration_s=1.0, rearm_delay_s=1.0))
        machine.step(PredicateResult(holds=True, track_ids=[1], confidence=0.7), 0.0)
        machine.step(PredicateResult(holds=True, track_ids=[1], confidence=0.95), 1.0)
        machine.step(PredicateResult(holds=True, track_ids=[1], confidence=0.6), 2.0)
        signals = machine.step(PredicateResult(holds=False), 3.0)
        signals += machine.step(PredicateResult(holds=False), 5.0)
        assert signals[-1].peak_confidence == pytest.approx(0.95)


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────


class TestWatchEngine:
    def _engine(self, **rule_overrides) -> WatchEngine:
        return WatchEngine(
            rules=[_rule(min_duration_s=1.0, rearm_delay_s=5.0, **rule_overrides)],
            zones=[LITTER_ZONE],
            source_uuid="src-1",
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )

    def test_a_cat_entering_the_zone_opens_an_event(self):
        """The end-to-end slice the plan targets, minus every service."""
        engine = self._engine()
        inside = [_track(bbox=(450, 450, 550, 550))]
        assert engine.step(inside, now=0.0) == []
        signals = engine.step(inside, now=1.5)
        assert [s.phase for s in signals] == [EventPhase.START]
        assert signals[0].source_uuid == "src-1"
        assert signals[0].track_ids == [1]

    def test_the_cat_leaving_closes_the_event_after_the_rearm_delay(self):
        engine = self._engine()
        inside = [_track(bbox=(450, 450, 550, 550))]
        engine.step(inside, now=0.0)
        engine.step(inside, now=1.5)
        engine.step([], now=2.0)
        signals = engine.step([], now=8.0)
        assert [s.phase for s in signals] == [EventPhase.END]

    def test_open_events_are_reported_for_the_status_panel(self):
        engine = self._engine()
        inside = [_track(bbox=(450, 450, 550, 550))]
        engine.step(inside, now=0.0)
        engine.step(inside, now=1.5)
        open_events = engine.open_events()
        assert len(open_events) == 1
        assert open_events[0]["ruleName"] == "cat using litter box"
        assert open_events[0]["startedAt"] == 0.0

    def test_close_all_closes_open_events(self):
        engine = self._engine()
        inside = [_track(bbox=(450, 450, 550, 550))]
        engine.step(inside, now=0.0)
        engine.step(inside, now=1.5)
        signals = engine.close_all(now=3.0)
        assert [s.phase for s in signals] == [EventPhase.END]
        assert engine.open_events() == []

    def test_frame_size_change_rescales_zones(self):
        """Zones are normalised, so the same rule must fire at any analysis width."""
        engine = WatchEngine(
            rules=[_rule(min_duration_s=0.0)],
            zones=[LITTER_ZONE],
            frame_width=1000,
            frame_height=1000,
        )
        engine.set_frame_size(500, 500)
        # 0.4..0.8 of a 500px frame is 200..400.
        inside = [_track(bbox=(250, 250, 300, 300))]
        engine.step(inside, now=0.0)
        signals = engine.step(inside, now=0.1)
        assert [s.phase for s in signals] == [EventPhase.START]

    def test_a_zero_frame_size_is_ignored_rather_than_breaking_every_zone(self):
        engine = self._engine()
        engine.set_frame_size(0, 0)
        assert engine.frame_width == FRAME_W

    def test_two_rules_track_independently(self):
        engine = WatchEngine(
            rules=[
                _rule(uuid="r1", name="in box", min_duration_s=1.0),
                _rule(uuid="r2", name="near box", relation="distance", threshold=0.5, min_duration_s=1.0),
            ],
            zones=[LITTER_ZONE],
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )
        inside = [_track(bbox=(450, 450, 550, 550))]
        engine.step(inside, now=0.0)
        signals = engine.step(inside, now=1.5)
        assert {s.rule_uuid for s in signals} == {"r1", "r2"}

    def test_tuning_hint_appears_for_a_flapping_rule(self):
        engine = self._engine()
        inside = [_track(bbox=(450, 450, 550, 550))]
        engine.step(inside, now=0.0)
        engine.step(inside, now=1.5)
        for i in range(12):
            engine.step(inside if i % 2 == 0 else [], now=2.0 + i * 0.3)
        hints = engine.tuning_hints(min_flaps=3)
        assert hints and "rearm" in hints[0]

    def test_no_hints_for_a_steady_rule(self):
        engine = self._engine()
        inside = [_track(bbox=(450, 450, 550, 550))]
        for i in range(10):
            engine.step(inside, now=i * 0.5)
        assert engine.tuning_hints() == []

    def test_trigger_for_reports_the_rules_configured_trigger(self):
        # StreamWatcherTask reads this to tell an on_exit impulse apart from a while
        # rule's START, which a signal alone cannot: both leave the same gap between
        # `started_at` and dispatch time (see TestImpulseTriggers), so the trigger has
        # to come from the rule, not be guessed from timing.
        engine = WatchEngine(
            rules=[
                _rule(uuid="r-while", trigger="while"),
                _rule(uuid="r-exit", trigger="on_exit"),
            ],
            zones=[LITTER_ZONE],
            source_uuid="src-1",
        )
        assert engine.trigger_for("r-while") == "while"
        assert engine.trigger_for("r-exit") == "on_exit"

    def test_trigger_for_an_unscoped_rule_is_none(self):
        engine = self._engine()
        assert engine.trigger_for("some-other-rule") is None


# ─────────────────────────────────────────────────────────────────────────────
# Multi-ref slots and the slot values they report
# ─────────────────────────────────────────────────────────────────────────────

CAT_A = "label-cat-a"
CAT_B = "label-cat-b"
ZONE_B_UUID = "zone-upstairs"
UPSTAIRS_ZONE = Zone(uuid=ZONE_B_UUID, name="upstairs box", kind="box", points=[(0.0, 0.0), (0.2, 0.2)])
ALL_ZONES = {ZONE_UUID: LITTER_ZONE, ZONE_B_UUID: UPSTAIRS_ZONE}


class TestMultiRefSubjects:
    """One rule covering several cats and several litter boxes, rather than N*M rules."""

    def test_any_listed_label_satisfies_the_subject(self):
        spec = _rule(subject_a_refs=[CAT_A, CAT_B])
        for label in (CAT_A, CAT_B):
            track = _track(bbox=(450, 450, 550, 550), label=label)
            assert evaluate_predicate(spec, [track], ALL_ZONES, FRAME_W, FRAME_H).holds

    def test_an_unlisted_label_does_not(self):
        spec = _rule(subject_a_refs=[CAT_A, CAT_B])
        track = _track(bbox=(450, 450, 550, 550), label="label-dog")
        assert not evaluate_predicate(spec, [track], ALL_ZONES, FRAME_W, FRAME_H).holds

    def test_any_listed_zone_satisfies_the_target(self):
        spec = _rule(subject_b_refs=[ZONE_UUID, ZONE_B_UUID])
        # Inside the *second* zone only.
        track = _track(bbox=(10, 10, 100, 100))
        assert evaluate_predicate(spec, [track], ALL_ZONES, FRAME_W, FRAME_H).holds

    def test_a_dangling_zone_ref_cannot_fire(self):
        spec = _rule(subject_b_refs=["zone-that-was-deleted"])
        track = _track(bbox=(450, 450, 550, 550))
        assert not evaluate_predicate(spec, [track], ALL_ZONES, FRAME_W, FRAME_H).holds

    def test_an_empty_subject_slot_cannot_fire(self):
        spec = _rule(subject_a_refs=[])
        track = _track(bbox=(450, 450, 550, 550))
        assert not evaluate_predicate(spec, [track], ALL_ZONES, FRAME_W, FRAME_H).holds


class TestReportedSlotValues:
    """Which cat and which litter box — the values a topic template interpolates."""

    def test_the_matching_label_and_zone_are_reported(self):
        spec = _rule(subject_a_refs=[CAT_A, CAT_B], subject_b_refs=[ZONE_UUID, ZONE_B_UUID])
        track = _track(bbox=(450, 450, 550, 550), label=CAT_B)
        result = evaluate_predicate(spec, [track], ALL_ZONES, FRAME_W, FRAME_H)
        assert result.subject_ref == CAT_B
        assert result.target_ref == ZONE_UUID

    def test_a_present_rule_reports_its_subject(self):
        spec = _rule(relation="present", subject_b_kind="none", subject_b_refs=[], subject_a_refs=[CAT_A])
        track = _track(label=CAT_A, conf=0.9)
        assert evaluate_predicate(spec, [track], ALL_ZONES, FRAME_W, FRAME_H).subject_ref == CAT_A

    def test_nothing_is_reported_when_the_predicate_fails(self):
        spec = _rule(subject_a_refs=[CAT_A])
        track = _track(bbox=(0, 0, 10, 10), label=CAT_A)  # far outside the zone
        result = evaluate_predicate(spec, [track], ALL_ZONES, FRAME_W, FRAME_H)
        assert not result.holds
        assert result.subject_ref is None

    def test_a_failing_pair_does_not_name_the_subject(self):
        """`best_score` moves for pairs that never fired; the reported slot must not."""
        spec = _rule(subject_a_refs=[CAT_A, CAT_B], threshold=0.9)
        fully_inside = _track(track_id=1, bbox=(450, 450, 550, 550), label=CAT_B)
        barely_touching = _track(track_id=2, bbox=(395, 395, 405, 405), label=CAT_A)
        result = evaluate_predicate(spec, [fully_inside, barely_touching], ALL_ZONES, FRAME_W, FRAME_H)
        assert result.holds
        assert result.subject_ref == CAT_B

    def test_the_signal_carries_the_slot_values(self):
        spec = _rule(subject_a_refs=[CAT_A, CAT_B], min_duration_s=1.0)
        machine = RuleMachine(spec, source_uuid="src-1")
        inside = _track(bbox=(450, 450, 550, 550), label=CAT_B)
        machine.step(evaluate_predicate(spec, [inside], ALL_ZONES, FRAME_W, FRAME_H), now=0.0)
        signals = machine.step(evaluate_predicate(spec, [inside], ALL_ZONES, FRAME_W, FRAME_H), now=1.5)
        assert signals[0].phase is EventPhase.START
        assert signals[0].subject_ref == CAT_B
        assert signals[0].target_ref == ZONE_UUID

    def test_slot_values_are_cleared_between_events(self):
        """A stale subject would name last visit's cat on this visit's event."""
        spec = _rule(subject_a_refs=[CAT_A, CAT_B], min_duration_s=0.0, rearm_delay_s=0.0)
        machine = RuleMachine(spec, source_uuid="src-1")
        inside_b = _track(bbox=(450, 450, 550, 550), label=CAT_B)
        machine.step(evaluate_predicate(spec, [inside_b], ALL_ZONES, FRAME_W, FRAME_H), now=0.0)
        machine.step(evaluate_predicate(spec, [inside_b], ALL_ZONES, FRAME_W, FRAME_H), now=0.1)
        machine.step(evaluate_predicate(spec, [], ALL_ZONES, FRAME_W, FRAME_H), now=1.0)
        machine.step(evaluate_predicate(spec, [], ALL_ZONES, FRAME_W, FRAME_H), now=2.0)
        assert machine.state is RuleState.IDLE

        inside_a = _track(bbox=(450, 450, 550, 550), label=CAT_A)
        machine.step(evaluate_predicate(spec, [inside_a], ALL_ZONES, FRAME_W, FRAME_H), now=3.0)
        signals = machine.step(evaluate_predicate(spec, [inside_a], ALL_ZONES, FRAME_W, FRAME_H), now=3.1)
        assert [s.subject_ref for s in signals if s.phase is EventPhase.START] == [CAT_A]


class TestEventIdentityIsStableWhileAnEventIsOpen:
    """An open event keeps one identity from START to END, so its row can be found.

    Four consumers have to find an event again while it runs: the recorder's open map,
    the timelapse action's frame buffer, the in-progress preview endpoint and the MQTT
    template. If the closing edge hashes to something the opening edge was not filed
    under, the row never receives `ended_at` and sits "in progress" until a restart
    sweeps it, while the timelapse action goes on sampling into a directory no closing
    edge can reach.

    `instance_id` is that identity. The slot refs are not, and deliberately move — an
    event is named by the weight of what it saw, so that a subject the detector renames
    mid-visit is still one event, correctly labelled at the end.
    """

    def _spec(self, **overrides):
        kwargs = dict(
            relation="present",
            subject_a_refs=[CAT_A, CAT_B],
            subject_b_kind="none",
            subject_b_refs=[],
            threshold=0.6,
            min_duration_s=1.0,
            rearm_delay_s=1.0,
        )
        kwargs.update(overrides)
        return _rule(**kwargs)

    def _drive(self, spec, frames):
        machine = RuleMachine(spec, source_uuid="src-1")
        emitted = []
        for now, tracks in frames:
            emitted.extend(machine.step(evaluate_predicate(spec, tracks, ALL_ZONES, FRAME_W, FRAME_H), now))
        return machine, emitted

    def test_a_more_confident_second_subject_does_not_rekey_the_close(self):
        spec = self._spec()
        weak_a = _track(track_id=1, label=CAT_A, conf=0.7)
        strong_b = _track(track_id=2, label=CAT_B, conf=0.98)
        _machine, emitted = self._drive(
            spec,
            [
                (0.0, [weak_a]),  # PENDING
                (1.5, [weak_a]),  # START, on CAT_A
                (2.0, [strong_b]),  # a *different* cat, seen more clearly
                (3.0, []),  # CLOSING
                (4.5, []),  # END
            ],
        )
        start = [s for s in emitted if s.phase is EventPhase.START]
        end = [s for s in emitted if s.phase is EventPhase.END]
        assert len(start) == 1 and len(end) == 1
        # The whole point: the key END is filed under is the one START was filed under.
        assert end[0].instance_id == start[0].instance_id
        assert end[0].instance_id is not None
        # And the name still follows the evidence — two frames of CAT_A outweigh one of
        # CAT_B, so the event is not renamed by the single clearer look.
        assert end[0].subject_ref == CAT_A

    def test_the_in_progress_signal_keeps_the_opening_identity(self):
        """`open_signals` is how a running event's row and frame buffer are found."""
        spec = self._spec()
        weak_a = _track(track_id=1, label=CAT_A, conf=0.7)
        strong_b = _track(track_id=2, label=CAT_B, conf=0.98)
        machine, emitted = self._drive(spec, [(0.0, [weak_a]), (1.5, [weak_a]), (2.0, [strong_b])])
        start = [s for s in emitted if s.phase is EventPhase.START][0]
        (update,) = machine.open_signals(2.5)
        assert update.instance_id == start.instance_id

    def test_the_peak_confidence_still_follows_the_strongest_observation(self):
        """The reported confidence is a measurement of the event, not a key into it, so it
        must still report the best look it ever got."""
        spec = self._spec()
        weak_a = _track(track_id=1, label=CAT_A, conf=0.7)
        strong_b = _track(track_id=2, label=CAT_B, conf=0.98)
        _machine, emitted = self._drive(
            spec,
            [(0.0, [weak_a]), (1.5, [weak_a]), (2.0, [strong_b]), (3.0, []), (4.5, [])],
        )
        end = [s for s in emitted if s.phase is EventPhase.END][0]
        assert end.peak_confidence == pytest.approx(0.98)

    def test_the_strongest_subject_before_the_event_opens_still_wins(self):
        """The lead-in counts too: the pile is accumulated from the first matching frame,
        so an event that spent its whole dwell looking like CAT_B opens as CAT_B."""
        spec = self._spec(min_duration_s=2.0)
        weak_a = _track(track_id=1, label=CAT_A, conf=0.7)
        strong_b = _track(track_id=2, label=CAT_B, conf=0.98)
        _machine, emitted = self._drive(
            spec,
            [
                (0.0, [weak_a]),  # PENDING, on CAT_A
                (1.0, [strong_b]),  # still PENDING — the event has not opened yet
                (2.5, [strong_b]),  # START
            ],
        )
        start = [s for s in emitted if s.phase is EventPhase.START]
        assert len(start) == 1
        assert start[0].subject_ref == CAT_B

    def test_a_flap_back_open_does_not_rekey_the_event(self):
        """Re-firing inside the rearm window rejoins the event, so it keeps its key."""
        spec = self._spec()
        weak_a = _track(track_id=1, label=CAT_A, conf=0.7)
        strong_b = _track(track_id=2, label=CAT_B, conf=0.98)
        _machine, emitted = self._drive(
            spec,
            [
                (0.0, [weak_a]),
                (1.5, [weak_a]),  # START, on CAT_A
                (2.0, []),  # CLOSING
                (2.5, [strong_b]),  # back inside the rearm window — same event
                (3.5, []),  # CLOSING again
                (5.0, []),  # END
            ],
        )
        start = [s for s in emitted if s.phase is EventPhase.START]
        end = [s for s in emitted if s.phase is EventPhase.END]
        assert len(start) == 1 and len(end) == 1
        assert end[0].instance_id == start[0].instance_id


class TestOneEventPerSubject:
    """An event is per *object*, and neither per label nor per track.

    Per label was the old answer and it was wrong in both directions at once. It split
    one subject into two events whenever the detector changed its mind about the name —
    one person crossing a yard produced two overlapping events, two clips and two cards —
    and it merged two subjects into one whenever they happened to share a name, which in
    a three-cat house is the common case rather than the corner.

    Per track is not the answer either: one body routinely arrives under several track
    ids, because a brief dropout breaks association and a new id starts beside the old
    one. Keying on track id would have turned that into an event per id.

    So the engine groups a frame's matches into clusters and keeps one event per cluster.
    `event_granularity` says whether the user wanted that at all: `rule` collapses every
    match into a single event, which is the right answer when the recording is the whole
    frame either way.
    """

    DOG = "label-dog"
    #: Two cats and a dog, as `suppression_group_for` would report them: the cats are
    #: alternate guesses about one animal, the dog is its own thing.
    GROUPS = {CAT_A: "group-cat", CAT_B: "group-cat", DOG: "group-dog"}

    def _present(self, refs, **overrides):
        kwargs = dict(
            relation="present",
            subject_a_refs=list(refs),
            subject_b_kind="none",
            subject_b_refs=[],
            threshold=0.6,
            min_duration_s=1.0,
            rearm_delay_s=10.0,
            max_duration_s=0.0,
        )
        kwargs.update(overrides)
        return _rule(**kwargs)

    def _drive(self, engine, frames):
        emitted = []
        for now, tracks in frames:
            emitted.extend(engine.step(tracks, now))
        return emitted

    def _drive_machine(self, spec, frames, label_groups=None):
        machine = RuleMachine(spec, source_uuid="src-1", label_groups=label_groups)
        emitted = []
        for now, tracks in frames:
            emitted.extend(machine.step(evaluate_predicate(spec, tracks, ALL_ZONES, FRAME_W, FRAME_H), now))
        return machine, emitted

    @staticmethod
    def _phases(emitted, phase):
        return [s for s in emitted if s.phase is phase]

    # ── two objects are two events ───────────────────────────────────────────

    def test_two_cats_in_different_places_open_two_events(self):
        spec = self._present([CAT_A, CAT_B])
        engine = WatchEngine([spec], source_uuid="src-1")
        cat_a = _track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)
        cat_b = _track(track_id=2, bbox=(400, 400, 420, 420), label=CAT_B, conf=0.9)

        emitted = self._drive(engine, [(0.0, [cat_a, cat_b]), (1.5, [cat_a, cat_b])])

        starts = self._phases(emitted, EventPhase.START)
        assert len(starts) == 2, "one event per cat, not one event naming the best cat"
        assert len({s.instance_id for s in starts}) == 2
        assert sorted(s.subject_ref for s in starts) == sorted([CAT_A, CAT_B])

    def test_two_cats_sharing_one_label_are_still_two_events(self):
        """The case that decided the design: "unique cats" means unique *objects*.

        Two cats the model cannot tell apart both settle on the group root, so keying an
        event on the label — or on the group — files two separate visits under one card
        and records only one of them.
        """
        spec = self._present([CAT_A])
        engine = WatchEngine([spec], source_uuid="src-1")
        one = _track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)
        two = _track(track_id=2, bbox=(400, 400, 420, 420), label=CAT_A, conf=0.9)

        emitted = self._drive(engine, [(0.0, [one, two]), (1.5, [one, two])])

        assert len(self._phases(emitted, EventPhase.START)) == 2

    def test_each_subject_keeps_its_own_hysteresis(self):
        """Two cats arriving at different times get their own clocks, not a shared one."""
        spec = self._present([CAT_A, CAT_B], min_duration_s=2.0)
        engine = WatchEngine([spec], source_uuid="src-1")
        cat_a = _track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)
        cat_b = _track(track_id=2, bbox=(400, 400, 420, 420), label=CAT_B, conf=0.9)

        emitted = self._drive(
            engine,
            [
                (0.0, [cat_a]),
                (3.0, [cat_a]),  # CAT_A opens
                (3.5, [cat_a, cat_b]),  # CAT_B only just arrived
                (4.0, [cat_a, cat_b]),  # still inside its own min_duration
                (6.0, [cat_a, cat_b]),  # now CAT_B opens too
            ],
        )

        starts = self._phases(emitted, EventPhase.START)
        assert [s.subject_ref for s in starts] == [CAT_A, CAT_B]
        assert starts[0].started_at < starts[1].started_at

    # ── one object is one event, whatever it is called ───────────────────────

    def test_a_sustained_relabel_inside_the_group_keeps_one_event(self):
        """The incident this design exists for.

        A person crossed a yard, the detector's argmax moved from the group root to a
        named variant and stayed there for tens of seconds, and the per-label machines
        turned one crossing into two overlapping events with two clips. `min_duration_s`
        cannot help — the wrong-label run is far longer than it — so nothing but taking
        the label out of the event's identity fixes this.
        """
        spec = self._present([CAT_A, CAT_B], min_duration_s=2.0)
        engine = WatchEngine([spec], source_uuid="src-1", label_groups=self.GROUPS)
        as_a = _track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)
        as_b = _track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_B, conf=0.9)

        emitted = self._drive(
            engine,
            [(0.0, [as_a]), (3.0, [as_a])] + [(t, [as_b]) for t in (4.0, 8.0, 12.0, 20.0, 30.0)],
        )

        assert len(self._phases(emitted, EventPhase.START)) == 1
        assert self._phases(emitted, EventPhase.END) == []

    def test_a_subject_moving_between_targets_stays_one_event(self):
        """One cat, two litter boxes, one visit.

        The target was in the event key too, so a cat that walked from one box to the
        other ended one event and opened another. Nobody had reported it; it is the same
        bug wearing the other slot.
        """
        spec = _rule(
            subject_a_refs=[CAT_A],
            subject_b_refs=[ZONE_UUID, ZONE_B_UUID],
            min_duration_s=1.0,
            rearm_delay_s=10.0,
            max_duration_s=0.0,
        )
        engine = WatchEngine([spec], zones=list(ALL_ZONES.values()), source_uuid="src-1")
        engine.set_frame_size(FRAME_W, FRAME_H)
        upstairs = _track(track_id=1, bbox=(20, 20, 120, 120), label=CAT_A, conf=0.9)
        downstairs = _track(track_id=1, bbox=(450, 450, 750, 750), label=CAT_A, conf=0.9)

        emitted = self._drive(
            engine,
            [
                (0.0, [upstairs]),
                (2.0, [upstairs]),  # START, in the upstairs box
                (3.0, [downstairs]),  # walked to the other box
                (5.0, [downstairs]),
                # Far past the rearm delay: if the move had closed the first event, the
                # END would have landed by now.
                (30.0, [downstairs]),
            ],
        )

        assert len(self._phases(emitted, EventPhase.START)) == 1
        assert self._phases(emitted, EventPhase.END) == []

    def test_an_obstruction_hiding_both_subjects_extends_the_open_event(self):
        """Everything drops out, comes back under new ids somewhere else, one event.

        The whole reason the rearm window exists. Neither shared track ids nor overlap
        with the last known boxes can recognise the returning subject, so an open event
        in its rearm window adopts the next compatible cluster wherever it appears —
        otherwise a passing obstruction costs a second card and a clip that duplicates
        the first one's tail.
        """
        spec = _rule(
            subject_a_refs=[CAT_A],
            subject_b_refs=[ZONE_UUID],
            min_duration_s=1.0,
            rearm_delay_s=10.0,
            max_duration_s=0.0,
        )
        engine = WatchEngine([spec], zones=list(ALL_ZONES.values()), source_uuid="src-1")
        engine.set_frame_size(FRAME_W, FRAME_H)
        before = _track(track_id=1, bbox=(450, 450, 600, 600), label=CAT_A, conf=0.9)
        after = _track(track_id=77, bbox=(650, 650, 790, 790), label=CAT_A, conf=0.9)

        emitted = self._drive(
            engine,
            [
                (0.0, [before]),
                (2.0, [before]),  # START
                (3.0, []),  # obstruction: both tracks gone
                (4.0, []),
                (6.0, [after]),  # back, new id, elsewhere in the zone
                (8.0, [after]),
            ],
        )

        assert len(self._phases(emitted, EventPhase.START)) == 1
        assert self._phases(emitted, EventPhase.END) == []

    def test_a_different_group_does_not_adopt_an_open_event(self):
        """A rearm window is not a licence to file the next animal under the first.

        Adoption is bounded by the `exclusive_variants` group, which is what declares two
        labels to be alternate guesses about one object. A dog is not a cat's alias, so
        it opens its own event and the cat's closes on time.
        """
        spec = self._present([CAT_A, self.DOG], min_duration_s=1.0, rearm_delay_s=6.0)
        machine, emitted = self._drive_machine(
            spec,
            [
                (0.0, [_track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)]),
                (2.0, [_track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)]),  # START, cat
                (3.0, []),  # cat gone — CLOSING
                (4.0, [_track(track_id=2, bbox=(500, 500, 520, 520), label=self.DOG, conf=0.9)]),
                (6.0, [_track(track_id=2, bbox=(500, 500, 520, 520), label=self.DOG, conf=0.9)]),
                (14.0, []),
                (21.0, []),
            ],
            label_groups=self.GROUPS,
        )

        starts = self._phases(emitted, EventPhase.START)
        assert [s.subject_ref for s in starts] == [CAT_A, self.DOG]
        assert len({s.instance_id for s in starts}) == 2
        assert machine.is_open is False

    # ── rule granularity ────────────────────────────────────────────────────

    def test_rule_granularity_gives_one_event_for_the_whole_scene(self):
        """Three unidentified cats must not become three videos of the same scene.

        `present` records the whole frame, so per-subject events would be near-identical
        clips. That is what `rule` granularity is for, and it is the only way to get it —
        per-label happened to give it only while every cat shared a name.
        """
        spec = self._present([CAT_A], event_granularity=GRANULARITY_RULE)
        engine = WatchEngine([spec], source_uuid="src-1")
        one = _track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)
        two = _track(track_id=2, bbox=(400, 400, 420, 420), label=CAT_A, conf=0.9)
        three = _track(track_id=3, bbox=(800, 800, 820, 820), label=CAT_A, conf=0.9)

        emitted = self._drive(engine, [(0.0, [one, two, three]), (1.5, [one, two, three])])

        assert len(self._phases(emitted, EventPhase.START)) == 1

    def test_rule_granularity_holds_while_any_subject_remains(self):
        """The headline case: one clip for as long as *a* cat is there, whoever it is."""
        spec = self._present([CAT_A], event_granularity=GRANULARITY_RULE)
        engine = WatchEngine([spec], source_uuid="src-1")
        one = _track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)
        two = _track(track_id=2, bbox=(400, 400, 420, 420), label=CAT_A, conf=0.9)
        three = _track(track_id=3, bbox=(800, 800, 820, 820), label=CAT_A, conf=0.9)

        emitted = self._drive(
            engine,
            [
                (0.0, [one, two]),
                (1.5, [one, two]),  # START
                (2.0, [two]),  # one leaves — still a cat present
                (3.0, [three]),  # a different cat entirely, new track id
                (4.0, [two, three]),
                (5.0, []),  # now none — CLOSING begins
                (16.0, []),  # rearm elapsed — END
            ],
        )

        assert len(self._phases(emitted, EventPhase.START)) == 1
        assert len(self._phases(emitted, EventPhase.END)) == 1

    def test_rule_granularity_never_holds_two_events_at_once(self):
        """One event "while the predicate holds for anyone" means exactly one, always.

        Observed live on 2026-09-27: three overlapping `Maple Present` events on one
        camera, a rule that migrated to `rule` granularity. A detector dropout replaced
        the track id in the same frame the cat reappeared a little further off, so no
        rule could place the new cluster on the open event and a second one was minted —
        under a setting whose whole definition is that there is only ever one.
        """
        spec = self._present([CAT_A], event_granularity=GRANULARITY_RULE)
        machine, emitted = self._drive_machine(
            spec,
            [
                (0.0, [_track(track_id=368, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)]),
                (2.0, [_track(track_id=368, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)]),  # START
                # New id, elsewhere, with no empty frame in between — the event is still
                # ACTIVE at the instant the replacement asks for a home.
                (3.0, [_track(track_id=411, bbox=(400, 400, 420, 420), label=CAT_A, conf=0.9)]),
                (5.0, [_track(track_id=411, bbox=(400, 400, 420, 420), label=CAT_A, conf=0.9)]),
            ],
        )

        assert len(self._phases(emitted, EventPhase.START)) == 1
        assert self._phases(emitted, EventPhase.END) == []
        assert len(machine._instances) == 1

    def test_a_capped_rule_event_stays_capped_when_the_track_id_changes(self):
        """The cap latch must survive a dropout, or the cap becomes the metronome again.

        Adoption deliberately refuses SPENT instances — one animal's cap must not swallow
        the next animal's event — so under `rule` granularity nothing else would hand the
        replacement track to the latch, and the capped event would reopen at once under a
        new id. That is what keeps every cluster on the single instance there.
        """
        spec = self._present(
            [CAT_A], event_granularity=GRANULARITY_RULE, min_duration_s=1.0, rearm_delay_s=5.0, max_duration_s=10.0
        )
        _machine, emitted = self._drive_machine(
            spec,
            [
                (0.0, [_track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)]),
                (2.0, [_track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)]),  # START
                (12.0, [_track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)]),  # capped
                # Same cat, new id, well clear of the old box — the latch must still hold.
                (14.0, [_track(track_id=77, bbox=(400, 400, 420, 420), label=CAT_A, conf=0.9)]),
                (20.0, [_track(track_id=77, bbox=(400, 400, 420, 420), label=CAT_A, conf=0.9)]),
            ],
        )

        assert len(self._phases(emitted, EventPhase.START)) == 1
        assert [e.close_reason for e in self._phases(emitted, EventPhase.END)] == [EventCloseReason.CAP]

    def test_a_track_replaced_in_one_frame_does_not_open_a_second_event(self):
        """The same handoff under `subject` granularity, which is the default.

        Adoption keyed on CLOSING alone missed this: the instance does not learn its
        subject is gone until after the frame's clusters have been placed, so on the
        transition frame it is still ACTIVE and refuses to take the replacement.
        """
        spec = self._present([CAT_A], min_duration_s=1.0, rearm_delay_s=10.0)
        machine, emitted = self._drive_machine(
            spec,
            [
                (0.0, [_track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)]),
                (2.0, [_track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)]),  # START
                (3.0, [_track(track_id=99, bbox=(400, 400, 420, 420), label=CAT_A, conf=0.9)]),
                (5.0, [_track(track_id=99, bbox=(400, 400, 420, 420), label=CAT_A, conf=0.9)]),
            ],
        )

        assert len(self._phases(emitted, EventPhase.START)) == 1
        assert self._phases(emitted, EventPhase.END) == []
        # Both ids belong to the one event, which is what the clip's track list reports.
        (open_instance,) = machine.open_instances
        assert open_instance.track_ids == {1, 99}

    def test_two_subjects_present_together_still_open_two_events(self):
        """The guard on the fix: a cluster may only be handed to an instance whose own
        tracks are absent from this frame. A second cat arriving beside a first, whose
        track is still right there, is a second event and must stay one."""
        spec = self._present([CAT_A], min_duration_s=1.0, rearm_delay_s=10.0)
        first = _track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)
        second = _track(track_id=2, bbox=(400, 400, 420, 420), label=CAT_A, conf=0.9)
        _machine, emitted = self._drive_machine(
            spec,
            [(0.0, [first]), (2.0, [first]), (3.0, [first, second]), (5.0, [first, second])],
        )

        starts = self._phases(emitted, EventPhase.START)
        assert len(starts) == 2
        assert len({s.instance_id for s in starts}) == 2

    # ── flicker, which the tracker's vote and min_duration already cover ─────

    def test_a_brief_relabel_does_not_open_an_event_for_the_borrowed_label(self):
        """`min_duration_s` is what makes a flicker free — no extra bookkeeping needed."""
        spec = self._present([CAT_A, CAT_B], min_duration_s=2.0)
        engine = WatchEngine([spec], source_uuid="src-1", label_groups=self.GROUPS)
        as_a = _track(track_id=1, label=CAT_A, conf=0.9)
        as_b = _track(track_id=1, label=CAT_B, conf=0.9)  # same track, other label

        emitted = self._drive(
            engine,
            [
                (0.0, [as_a]),
                (3.0, [as_a]),  # START on CAT_A
                (4.0, [as_b]),  # flickers to CAT_B for well under min_duration_s
                (4.5, [as_a]),  # and back
                (6.0, [as_a]),
            ],
        )

        starts = self._phases(emitted, EventPhase.START)
        assert [s.subject_ref for s in starts] == [CAT_A], "the borrowed label must open nothing"

    def test_a_brief_relabel_does_not_close_the_original_event(self):
        """The other half: one object, so there is no gap for a close to happen in."""
        spec = self._present([CAT_A, CAT_B], min_duration_s=2.0, rearm_delay_s=10.0)
        engine = WatchEngine([spec], source_uuid="src-1", label_groups=self.GROUPS)
        as_a = _track(track_id=1, label=CAT_A, conf=0.9)
        as_b = _track(track_id=1, label=CAT_B, conf=0.9)

        emitted = self._drive(
            engine,
            [(0.0, [as_a]), (3.0, [as_a]), (4.0, [as_b]), (4.5, [as_a]), (6.0, [as_a])],
        )

        assert self._phases(emitted, EventPhase.END) == []

    def test_a_label_reappearing_on_a_new_track_rejoins_its_open_event(self):
        """Left the frame and came back inside the rearm window — one event, not two."""
        spec = self._present([CAT_A])
        engine = WatchEngine([spec], source_uuid="src-1")
        first = _track(track_id=1, label=CAT_A, conf=0.9)
        # A genuinely new track: re-detection after an occlusion never reuses the id.
        second = _track(track_id=99, bbox=(300, 300, 320, 320), label=CAT_A, conf=0.9)

        emitted = self._drive(
            engine,
            [
                (0.0, [first]),
                (1.5, [first]),  # START
                (2.0, []),  # gone — CLOSING
                (5.0, [second]),  # back inside the 10s rearm window
                (7.0, [second]),
            ],
        )

        assert len(self._phases(emitted, EventPhase.START)) == 1
        assert self._phases(emitted, EventPhase.END) == []

    # ── the machine table stays bounded ─────────────────────────────────────

    def test_a_finished_event_is_forgotten(self):
        """Instances are allocated per object now, so they have to be reclaimed.

        The old machine table was fixed at load time, which is what made cardinality a
        non-question. A watcher runs for weeks, so an instance that outlived its event
        would be a slow leak with no symptom until the box ran out of memory.
        """
        spec = self._present([CAT_A], min_duration_s=1.0, rearm_delay_s=2.0)
        machine, emitted = self._drive_machine(
            spec,
            [
                (0.0, [_track(track_id=1, label=CAT_A, conf=0.9)]),
                (2.0, [_track(track_id=1, label=CAT_A, conf=0.9)]),
                (3.0, []),
                (6.0, []),
            ],
        )

        assert len(self._phases(emitted, EventPhase.END)) == 1
        # Private on purpose: there is no public view of retired state, and the invariant
        # being asserted is that there is none left to view.
        assert machine._instances == {}

    def test_a_capped_event_does_not_reopen_while_its_subject_is_still_there(self):
        """SPENT has to recognise the subject it capped, which means keeping its identity.

        A cap that forgot what it capped would see a brand new arrival on the very next
        frame and open again one `max_duration_s` later, forever — the metronome SPENT
        exists to prevent.
        """
        spec = self._present([CAT_A], min_duration_s=1.0, rearm_delay_s=5.0, max_duration_s=10.0)
        cat = _track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)
        # The subject drifts across the frame while latched, so a stale box cannot be
        # what recognises it.
        _machine, emitted = self._drive_machine(
            spec,
            [(0.0, [cat])]
            + [
                (t, [_track(track_id=1, bbox=(t * 10, 0, t * 10 + 20, 20), label=CAT_A, conf=0.9)])
                for t in (2.0, 12.0, 20.0, 30.0, 40.0)
            ],
        )

        assert len(self._phases(emitted, EventPhase.START)) == 1
        ends = self._phases(emitted, EventPhase.END)
        assert [e.close_reason for e in ends] == [EventCloseReason.CAP]

    # ── machine counts: one rule is one machine again ───────────────────────

    def test_a_single_ref_rule_is_unchanged(self):
        spec = self._present([CAT_A])
        engine = WatchEngine([spec], source_uuid="src-1")
        assert len(engine._machines) == 1

    def test_a_subjectless_or_unrestricted_rule_keeps_one_machine(self):
        """`any` has no ref list to split on, so it must not lose its machine."""
        anything = _rule(relation="present", subject_a_kind="any", subject_a_refs=[], subject_b_kind="none")
        engine = WatchEngine([anything], source_uuid="src-1")
        assert len(engine._machines) == 1

    def test_the_rule_list_still_counts_rules_not_slots(self):
        """`rule_uuids` answers "which rules run here", and callers count it."""
        spec = self._present([CAT_A, CAT_B])
        engine = WatchEngine([spec], source_uuid="src-1")
        assert engine.rule_uuids == [spec.uuid]

    def test_one_flapping_rule_gives_one_hint_however_many_subjects_it_has(self):
        spec = self._present([CAT_A, CAT_B], min_duration_s=0.0, rearm_delay_s=5.0)
        engine = WatchEngine([spec], source_uuid="src-1")
        cat_a = _track(track_id=1, bbox=(0, 0, 20, 20), label=CAT_A, conf=0.9)
        cat_b = _track(track_id=2, bbox=(400, 400, 420, 420), label=CAT_B, conf=0.9)
        now = 0.0
        # Two holding frames to get past PENDING and actually open the events.
        for _ in range(2):
            engine.step([cat_a, cat_b], now)
            now += 0.1
        for _ in range(8):
            engine.step([], now)  # drop out, but stay inside the rearm window
            now += 0.1
            engine.step([cat_a, cat_b], now)  # ...and back: CLOSING -> ACTIVE is a flap
            now += 0.1

        hints = engine.tuning_hints()
        assert len(hints) == 1, "one rule, one piece of advice — not one per cat"


class TestMasking:
    """Masks suppress by centroid — see `is_masked` for why not by overlap."""

    def test_a_detection_centred_in_a_mask_is_suppressed(self):
        from apps.helpers.watch.geometry import is_masked

        assert is_masked((450, 450, 550, 550), [LITTER_ZONE], FRAME_W, FRAME_H)

    def test_a_detection_centred_outside_is_kept(self):
        from apps.helpers.watch.geometry import is_masked

        assert not is_masked((0, 0, 100, 100), [LITTER_ZONE], FRAME_W, FRAME_H)

    def test_merely_overlapping_a_mask_is_not_enough(self):
        """A cat half in a masked doorway still tracks; only its centre decides."""
        from apps.helpers.watch.geometry import is_masked

        # Spans 300..450 so it genuinely overlaps the zone's 400..800, but its centre
        # (375) sits outside. Note 400 exactly would count as inside — containment is
        # boundary-inclusive.
        assert intersection_area((300, 300, 450, 450), LITTER_ZONE.bounds(FRAME_W, FRAME_H)) > 0
        assert not is_masked((300, 300, 450, 450), [LITTER_ZONE], FRAME_W, FRAME_H)

    def test_no_masks_suppresses_nothing(self):
        from apps.helpers.watch.geometry import is_masked

        assert not is_masked((450, 450, 550, 550), [], FRAME_W, FRAME_H)

    def test_a_concave_polygon_mask_uses_exact_containment(self):
        """The reason centroid beats overlap: overlap falls back to the bounding box."""
        from apps.helpers.watch.geometry import is_masked

        # An L-shape; the notch at (0.8, 0.8) is inside the bounding box but outside
        # the polygon itself.
        l_shape = Zone(
            uuid="m1",
            name="L",
            kind="polygon",
            points=[(0.0, 0.0), (0.6, 0.0), (0.6, 0.6), (0.0, 0.6)],
        )
        assert is_masked((250, 250, 350, 350), [l_shape], FRAME_W, FRAME_H)
        assert not is_masked((750, 750, 850, 850), [l_shape], FRAME_W, FRAME_H)


class TestIncludeDescendants:
    """ "Include child labels" is stored as a flag and expanded when the watcher starts.

    Storing the expansion instead would freeze the rule at the moment it was written, so
    a label added to the tree later silently stops matching — which is the maintenance
    this flag exists to remove.
    """

    # Cat
    #  ├── Tabby
    #  │    └── Kitten
    #  └── Ginger
    ANCESTRY = {
        "cat": set(),
        "tabby": {"cat"},
        "kitten": {"tabby", "cat"},
        "ginger": {"cat"},
        "dog": set(),
    }

    def test_descendants_are_found_transitively(self):
        from apps.helpers.watch.rules import descendants_of

        assert descendants_of("cat", self.ANCESTRY) == ["ginger", "kitten", "tabby"]

    def test_a_leaf_has_no_descendants(self):
        from apps.helpers.watch.rules import descendants_of

        assert descendants_of("kitten", self.ANCESTRY) == []

    def test_the_slot_widens_to_the_whole_subtree(self):
        from apps.helpers.watch.rules import expand_descendants

        spec = _rule(subject_a_refs=["cat"], subject_a_include_descendants=True)
        widened = expand_descendants(spec, self.ANCESTRY)
        assert set(widened.subject_a_refs) == {"cat", "tabby", "kitten", "ginger"}

    def test_the_flag_off_changes_nothing(self):
        from apps.helpers.watch.rules import expand_descendants

        spec = _rule(subject_a_refs=["cat"], subject_a_include_descendants=False)
        assert expand_descendants(spec, self.ANCESTRY).subject_a_refs == ["cat"]

    def test_the_original_pick_is_kept_and_comes_first(self):
        """The rule still matches the parent itself, not only its children."""
        from apps.helpers.watch.rules import expand_descendants

        spec = _rule(subject_a_refs=["cat"], subject_a_include_descendants=True)
        assert expand_descendants(spec, self.ANCESTRY).subject_a_refs[0] == "cat"

    def test_overlapping_subtrees_do_not_duplicate(self):
        """Picking a parent and its child must not evaluate the child twice."""
        from apps.helpers.watch.rules import expand_descendants

        spec = _rule(subject_a_refs=["cat", "tabby"], subject_a_include_descendants=True)
        refs = expand_descendants(spec, self.ANCESTRY).subject_a_refs
        assert len(refs) == len(set(refs))

    def test_a_zone_slot_never_widens(self):
        """Zones have no hierarchy; a zone uuid is in no ancestry map, so widening it
        would quietly produce nothing."""
        from apps.helpers.watch.rules import expand_descendants

        spec = _rule(
            subject_b_kind="zone",
            subject_b_refs=[ZONE_UUID],
            subject_b_include_descendants=True,
        )
        assert expand_descendants(spec, self.ANCESTRY).subject_b_refs == [ZONE_UUID]

    def test_both_slots_widen_independently(self):
        from apps.helpers.watch.rules import expand_descendants

        spec = _rule(
            subject_a_refs=["cat"],
            subject_a_include_descendants=True,
            subject_b_kind="label",
            subject_b_refs=["dog"],
            subject_b_include_descendants=False,
        )
        widened = expand_descendants(spec, self.ANCESTRY)
        assert len(widened.subject_a_refs) == 4
        assert widened.subject_b_refs == ["dog"]

    def test_an_empty_ancestry_map_is_a_no_op(self):
        """A watcher that could not load labels must not silently empty its rules."""
        from apps.helpers.watch.rules import expand_descendants

        spec = _rule(subject_a_refs=["cat"], subject_a_include_descendants=True)
        assert expand_descendants(spec, {}).subject_a_refs == ["cat"]

    def test_expansion_returns_a_new_spec(self):
        """The stored spec is shared across sources; widening in place would compound."""
        from apps.helpers.watch.rules import expand_descendants

        spec = _rule(subject_a_refs=["cat"], subject_a_include_descendants=True)
        expand_descendants(spec, self.ANCESTRY)
        assert spec.subject_a_refs == ["cat"]

    def test_a_widened_rule_matches_a_child_label(self):
        """End to end: the point of the flag."""
        from apps.helpers.watch.rules import expand_descendants

        spec = expand_descendants(_rule(subject_a_refs=["cat"], subject_a_include_descendants=True), self.ANCESTRY)
        kitten = _track(bbox=(450, 450, 550, 550), label="kitten")
        result = evaluate_predicate(spec, [kitten], {ZONE_UUID: LITTER_ZONE}, FRAME_W, FRAME_H)
        assert result.holds
        # The reported slot names the *specific* label that matched, not the parent that
        # was picked — which is what makes a per-cat MQTT topic work.
        assert result.subject_ref == "kitten"

    def test_an_unrelated_label_still_does_not_match(self):
        from apps.helpers.watch.rules import expand_descendants

        spec = expand_descendants(_rule(subject_a_refs=["cat"], subject_a_include_descendants=True), self.ANCESTRY)
        dog = _track(bbox=(450, 450, 550, 550), label="dog")
        assert not evaluate_predicate(spec, [dog], {ZONE_UUID: LITTER_ZONE}, FRAME_W, FRAME_H).holds


class TestRuleSourceScope:
    """A rule can be restricted to particular sources; empty means every source.

    The motivating case is `present`: "a cat is in the scene" is true in every room all
    day, and is only worth an event in one of them. Scoping the *action* would leave the
    event raised on every camera — this suppresses the event itself.
    """

    def _engine(self, source_uuid, **rule_overrides):
        return WatchEngine(
            rules=[_rule(min_duration_s=1.0, rearm_delay_s=5.0, **rule_overrides)],
            zones=[LITTER_ZONE],
            source_uuid=source_uuid,
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )

    def test_an_unscoped_rule_runs_on_every_source(self):
        engine = self._engine("src-2")
        inside = [_track(bbox=(450, 450, 550, 550))]
        engine.step(inside, now=0.0)
        assert [s.phase for s in engine.step(inside, now=1.5)] == [EventPhase.START]

    def test_a_rule_scoped_to_this_source_still_fires(self):
        engine = self._engine("src-1", source_uuids=["src-1"])
        inside = [_track(bbox=(450, 450, 550, 550))]
        engine.step(inside, now=0.0)
        assert [s.phase for s in engine.step(inside, now=1.5)] == [EventPhase.START]

    def test_a_rule_scoped_to_another_source_never_fires(self):
        """The regression this feature is: the same tracks, a different camera."""
        engine = self._engine("src-2", source_uuids=["src-1"])
        inside = [_track(bbox=(450, 450, 550, 550))]
        engine.step(inside, now=0.0)
        assert engine.step(inside, now=1.5) == []
        assert engine.step(inside, now=99.0) == []

    def test_a_scoped_out_rule_holds_no_state_at_all(self):
        """Not merely silenced: no machine, so nothing to show, close or hint about.

        A dormant machine would appear in the status panel's open-events table and in
        `close_all`, and could emit a stray END if the scope ever changed under it.
        """
        engine = self._engine("src-2", source_uuids=["src-1"])
        assert engine.rule_uuids == []
        inside = [_track(bbox=(450, 450, 550, 550))]
        engine.step(inside, now=0.0)
        engine.step(inside, now=1.5)
        assert engine.open_events() == []
        assert engine.close_all(now=2.0) == []

    def test_a_multi_source_scope_covers_each_of_them(self):
        inside = [_track(bbox=(450, 450, 550, 550))]
        for source in ("src-1", "src-3"):
            engine = self._engine(source, source_uuids=["src-1", "src-3"])
            engine.step(inside, now=0.0)
            assert [s.phase for s in engine.step(inside, now=1.5)] == [EventPhase.START]
        excluded = self._engine("src-2", source_uuids=["src-1", "src-3"])
        excluded.step(inside, now=0.0)
        assert excluded.step(inside, now=1.5) == []

    def test_an_engine_with_no_source_identity_evaluates_every_rule(self):
        """A preview has no source to judge against; dropping every scoped rule there
        would look exactly like a predicate that is never true."""
        engine = self._engine(None, source_uuids=["src-1"])
        inside = [_track(bbox=(450, 450, 550, 550))]
        engine.step(inside, now=0.0)
        assert [s.phase for s in engine.step(inside, now=1.5)] == [EventPhase.START]

    def test_the_scope_is_read_off_a_row(self):
        """`from_row` is what the watcher actually builds specs with."""
        row = {
            "uuid": "rule-1",
            "name": "cat present",
            "subject_a_kind": "label",
            "subject_a_refs_json": [CAT],
            "relation": "present",
            "subject_b_kind": "none",
            "source_uuids_json": ["src-1"],
        }
        assert RuleSpec.from_row(row).source_uuids == ["src-1"]

    def test_a_row_without_the_column_scopes_to_nothing(self):
        """Empty means all — a rule stored before the column existed runs everywhere."""
        row = {"uuid": "r", "name": "n", "subject_a_kind": "label", "subject_a_refs_json": [CAT]}
        assert RuleSpec.from_row(row).source_uuids == []

    def test_expanding_descendants_preserves_the_scope(self):
        """`expand_descendants` rebuilds the spec, and a dropped field would silently
        widen the rule back to every camera."""
        from apps.helpers.watch.rules import expand_descendants

        spec = _rule(subject_a_refs=["cat"], subject_a_include_descendants=True, source_uuids=["src-1"])
        assert expand_descendants(spec, {"cat": set(), "tabby": {"cat"}}).source_uuids == ["src-1"]


# ─────────────────────────────────────────────────────────────────────────────
# Margin — "fire when they are merely *near* each other"
# ─────────────────────────────────────────────────────────────────────────────
class TestMarginGeometry:
    def test_inflate_grows_every_edge(self):
        assert inflate((100, 100, 200, 200), 10) == (90, 90, 210, 210)

    def test_a_negative_pad_shrinks(self):
        assert inflate((100, 100, 200, 200), -10) == (110, 110, 190, 190)

    def test_over_shrinking_collapses_to_the_centre_rather_than_inverting(self):
        """A box with negative width would score nonsense; a point scores zero."""
        assert inflate((100, 100, 200, 200), -80) == (150, 150, 150, 150)
        assert area(inflate((100, 100, 200, 200), -80)) == 0.0

    def test_box_scale_is_the_mean_of_the_two_sides(self):
        assert box_scale((0, 0, 100, 50)) == 75.0

    def test_the_pad_is_measured_against_the_smaller_box(self):
        """A cat paired with a whole litter box is measured in cats, not litter boxes."""
        cat, litterbox = (0, 0, 100, 100), (0, 0, 500, 500)
        assert margin_pad(1.0, cat, litterbox) == pytest.approx(50.0)
        assert margin_pad(1.0, litterbox, cat) == pytest.approx(50.0)

    def test_each_box_grows_by_half_the_margin(self):
        """So the number reads as the whole gap it closes — one subject-width at 1.0."""
        box = (0, 0, 100, 100)
        assert 2 * margin_pad(1.0, box, box) == pytest.approx(box_scale(box))

    def test_point_inside_a_polygon_is_zero_distance(self):
        square = [(0, 0), (100, 0), (100, 100), (0, 100)]
        assert point_to_polygon_distance((50, 50), square) == 0.0

    def test_point_outside_measures_to_the_nearest_edge(self):
        square = [(0, 0), (100, 0), (100, 100), (0, 100)]
        assert point_to_polygon_distance((130, 50), square) == pytest.approx(30.0)

    def test_a_widened_zone_admits_a_point_just_outside_it(self):
        point = (0.85 * FRAME_W, 0.6 * FRAME_H)  # 50px right of the zone's 0.8 edge
        assert LITTER_ZONE.contains_point(point, FRAME_W, FRAME_H) is False
        assert LITTER_ZONE.contains_point_within(point, 60, FRAME_W, FRAME_H) is True
        assert LITTER_ZONE.contains_point_within(point, 40, FRAME_W, FRAME_H) is False

    def test_an_eroded_zone_rejects_a_point_near_its_edge(self):
        point = (0.78 * FRAME_W, 0.6 * FRAME_H)  # 20px inside the zone
        assert LITTER_ZONE.contains_point(point, FRAME_W, FRAME_H) is True
        assert LITTER_ZONE.contains_point_within(point, -50, FRAME_W, FRAME_H) is False

    def test_a_polygon_zone_widens_by_distance_to_its_edge(self):
        triangle = Zone(uuid="z", name="wedge", kind="polygon", points=[(0.0, 0.0), (0.4, 0.0), (0.0, 0.4)])
        point = (0.0, 0.45 * FRAME_H)  # 50px below the triangle's lowest vertex
        assert triangle.contains_point(point, FRAME_W, FRAME_H) is False
        assert triangle.contains_point_within(point, 60, FRAME_W, FRAME_H) is True
        assert triangle.contains_point_within(point, 40, FRAME_W, FRAME_H) is False


class TestMarginPredicates:
    """The "cat is *near* the water bowl" case: no new relation, just wider boxes."""

    def _pair(self, scale=1.0):
        """A cat and a bowl 40px apart — or the same scene at another distance."""

        def box(x1, y1, x2, y2):
            return (x1 * scale, y1 * scale, x2 * scale, y2 * scale)

        return [
            _track(track_id=1, bbox=box(100, 100, 300, 300), label=CAT),
            _track(track_id=2, bbox=box(340, 100, 440, 200), label=BOX),
        ]

    def _spec(self, margin):
        return _rule(relation="overlaps", subject_b_kind="label", subject_b_ref=BOX, threshold=0.1, margin=margin)

    def test_disjoint_boxes_do_not_fire_without_a_margin(self):
        assert evaluate_predicate(self._spec(0.0), self._pair(), {}, FRAME_W, FRAME_H).holds is False

    def test_a_margin_makes_nearby_boxes_fire(self):
        result = evaluate_predicate(self._spec(1.0), self._pair(), {}, FRAME_W, FRAME_H)
        assert result.holds is True
        assert result.track_ids == [1]

    def test_a_margin_too_small_to_close_the_gap_does_not_fire(self):
        """The gap is 40px and the smaller box is 100 across, so 0.2 buys 20px total."""
        assert evaluate_predicate(self._spec(0.2), self._pair(), {}, FRAME_W, FRAME_H).holds is False

    def test_the_same_scene_at_half_the_size_scores_identically(self):
        """The whole point of sizing the margin to the object: a cat at the far wall is
        half as wide, but so is its bowl and so is the gap, and the rule must not
        quietly become stricter."""
        near = evaluate_predicate(self._spec(1.0), self._pair(), {}, FRAME_W, FRAME_H)
        far = evaluate_predicate(self._spec(1.0), self._pair(scale=0.5), {}, FRAME_W, FRAME_H)
        assert far.holds == near.holds is True
        assert far.score == pytest.approx(near.score)

    def test_a_negative_margin_demands_a_deeper_overlap(self):
        """Boxes that plainly overlap stop matching once both are eroded past it."""
        cat = _track(track_id=1, bbox=(100, 100, 300, 300), label=CAT)
        bowl = _track(track_id=2, bbox=(280, 100, 480, 300), label=BOX)  # 20px of overlap
        fires = _rule(relation="overlaps", subject_b_kind="label", subject_b_ref=BOX, threshold=0.05)
        assert evaluate_predicate(fires, [cat, bowl], {}, FRAME_W, FRAME_H).holds is True
        eroded = _rule(
            relation="overlaps", subject_b_kind="label", subject_b_ref=BOX, threshold=0.05, margin=-0.2
        )  # 20px off each box closes a 20px overlap
        assert evaluate_predicate(eroded, [cat, bowl], {}, FRAME_W, FRAME_H).holds is False

    def test_a_margin_widens_a_zone_rule_too(self):
        """The zone branch: a cat 10px shy of the litter box area."""
        cat = _track(bbox=(300, 500, 390, 600))  # zone starts at x=400
        zones = {ZONE_UUID: LITTER_ZONE}
        assert evaluate_predicate(_rule(relation="overlaps", threshold=0.1), [cat], zones, FRAME_W, FRAME_H).holds is (
            False
        )
        near = _rule(relation="overlaps", threshold=0.1, margin=0.5)
        assert evaluate_predicate(near, [cat], zones, FRAME_W, FRAME_H).holds is True

    def test_a_margin_widens_centroid_in(self):
        """The centroid cannot grow, so the whole margin goes on the target."""
        cat = _track(bbox=(820, 500, 900, 600))  # centre at x=860, zone ends at 800
        zones = {ZONE_UUID: LITTER_ZONE}
        assert evaluate_predicate(_rule(relation="centroid_in"), [cat], zones, FRAME_W, FRAME_H).holds is False
        # The cat's mean side is 90, so 1.0 puts 90px on the zone and 0.5 puts 45.
        assert evaluate_predicate(_rule(relation="centroid_in", margin=0.5), [cat], zones, FRAME_W, FRAME_H).holds is (
            False
        )
        assert evaluate_predicate(_rule(relation="centroid_in", margin=1.0), [cat], zones, FRAME_W, FRAME_H).holds is (
            True
        )

    def test_distance_ignores_a_margin(self):
        """Its threshold is already a proximity measure, and centres do not inflate."""
        far = _track(bbox=(0, 0, 20, 20))
        zones = {ZONE_UUID: LITTER_ZONE}
        spec = _rule(relation="distance", threshold=0.2, margin=2.0)
        assert evaluate_predicate(spec, [far], zones, FRAME_W, FRAME_H).holds is False

    def test_a_rule_row_without_the_column_has_no_margin(self):
        row = {"uuid": "r", "name": "n", "subject_a_kind": "label", "subject_a_refs_json": [CAT]}
        assert RuleSpec.from_row(row).margin == 0.0

    def test_a_null_margin_column_reads_as_zero(self):
        """SQLite hands back None for a column added before its default was applied."""
        row = {"uuid": "r", "name": "n", "subject_a_kind": "label", "subject_a_refs_json": [CAT], "margin": None}
        assert RuleSpec.from_row(row).margin == 0.0


class TestRuntimeArming:
    """`armed` is the runtime half of "is this rule running", driven from Home Assistant
    and ANDed with the configured `enabled`. The two are separate so that arming never
    rewrites the user's saved intent — see `RuleSpec.armed`.
    """

    def _engine(self, **rule_overrides) -> WatchEngine:
        return WatchEngine(
            rules=[_rule(min_duration_s=1.0, rearm_delay_s=5.0, **rule_overrides)],
            zones=[LITTER_ZONE],
            source_uuid="src-1",
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )

    def _open(self, engine) -> list:
        inside = [_track(bbox=(450, 450, 550, 550))]
        engine.step(inside, now=0.0)
        return engine.step(inside, now=1.5)

    def test_disarming_closes_the_open_event_without_waiting_for_a_frame(self):
        """The reason `set_rule_armed` returns signals at all. A source that is down, or
        simply not due, is not stepped for minutes — so leaving the close to the next
        `step()` would keep recording well after the condition stopped holding, and a
        source that never came back would strand the event outright.
        """
        engine = self._engine()
        assert [s.phase for s in self._open(engine)] == [EventPhase.START]

        signals = engine.set_rule_armed("rule-1", False, now=3.0)

        assert [s.phase for s in signals] == [EventPhase.END]
        assert engine.open_events() == []

    def test_disarming_twice_closes_nothing_the_second_time(self):
        """The retained command is replayed on every reconnect, so a no-op disarm has to
        be genuinely free rather than emitting a second END for an event already closed.
        """
        engine = self._engine()
        self._open(engine)
        engine.set_rule_armed("rule-1", False, now=3.0)

        assert engine.set_rule_armed("rule-1", False, now=4.0) == []

    def test_arming_closes_nothing(self):
        engine = self._engine()
        self._open(engine)

        assert engine.set_rule_armed("rule-1", True, now=3.0) == []
        assert len(engine.open_events()) == 1, "an armed rule keeps the event it holds"

    def test_a_disarmed_rule_opens_nothing_and_reopens_once_rearmed(self):
        """The gate itself. Asserted as a conditional — that the same input produces no
        event while disarmed and an event once armed — because a test that only checked
        the disarmed half would pass against a rule that was simply broken.
        """
        engine = self._engine()
        engine.set_rule_armed("rule-1", False, now=0.0)
        inside = [_track(bbox=(450, 450, 550, 550))]

        engine.step(inside, now=0.0)
        assert engine.step(inside, now=1.5) == []

        engine.set_rule_armed("rule-1", True, now=2.0)
        engine.step(inside, now=2.0)
        assert [s.phase for s in engine.step(inside, now=3.5)] == [EventPhase.START]

    def test_arming_cannot_revive_a_rule_the_user_disabled(self):
        """`enabled` is authored intent and stays authoritative: a switch in HA narrows
        what an enabled rule does, it never turns a disabled one back on."""
        engine = self._engine(enabled=False)
        engine.set_rule_armed("rule-1", True, now=0.0)
        inside = [_track(bbox=(450, 450, 550, 550))]

        engine.step(inside, now=0.0)
        assert engine.step(inside, now=1.5) == []

    def test_arming_one_rule_leaves_the_others_alone(self):
        engine = WatchEngine(
            rules=[
                _rule(uuid="rule-1", min_duration_s=1.0),
                _rule(uuid="rule-2", min_duration_s=1.0),
            ],
            zones=[LITTER_ZONE],
            source_uuid="src-1",
            frame_width=FRAME_W,
            frame_height=FRAME_H,
        )
        engine.set_rule_armed("rule-1", False, now=0.0)

        assert [(uuid, armed) for uuid, _name, armed in engine.rule_arming()] == [
            ("rule-1", False),
            ("rule-2", True),
        ]
        inside = [_track(bbox=(450, 450, 550, 550))]
        engine.step(inside, now=0.0)
        signals = engine.step(inside, now=1.5)
        assert [s.rule_uuid for s in signals] == ["rule-2"]


# ─────────────────────────────────────────────────────────────────────────────
# Impulse triggers
# ─────────────────────────────────────────────────────────────────────────────
class TestImpulseTriggers:
    """`on_enter` / `on_exit`: a zero-length event at one edge of the predicate.

    The interesting moment is often an edge rather than a span — "the car left the
    garage" has no duration, and the footage worth keeping is on either side of it,
    which is what an action's preroll and postroll already express. What must hold is
    that the impulse is anchored at the *real* edge (a preroll is measured back from
    it), that the existing guards keep their meaning, and that nothing downstream is
    ever shown half an event.
    """

    @staticmethod
    def _run(machine, sequence):
        from apps.helpers.watch.rules import PredicateResult

        emitted = []
        for holds, now in sequence:
            emitted.extend(machine.step(PredicateResult(holds=holds, track_ids=[1] if holds else []), now))
        return emitted

    # ---------------------------------------------------------------- on_enter

    def test_on_enter_fires_one_zero_length_event_at_the_arrival(self):
        machine = RuleMachine(_rule(trigger="on_enter", min_duration_s=2.0, rearm_delay_s=10.0))
        signals = self._run(machine, [(True, 100.0), (True, 101.0), (True, 102.5)])

        assert [s.phase for s in signals] == [EventPhase.START, EventPhase.END]
        assert signals[-1].duration_s == 0
        # Anchored at when the subject appeared, not at when the dwell confirmed it —
        # otherwise a 15s preroll starts 2s late and misses the approach.
        assert signals[0].started_at == 100.0
        assert signals[0].at == 100.0

    def test_on_enter_does_not_fire_again_while_the_subject_stays(self):
        """The failure this shares with the cap: a still-true predicate must not be read
        as a fresh arrival on every frame."""
        machine = RuleMachine(_rule(trigger="on_enter", min_duration_s=1.0, rearm_delay_s=10.0))
        signals = self._run(machine, [(True, 0.0), (True, 2.0)] + [(True, t) for t in range(3, 400, 5)])

        assert [s.phase for s in signals] == [EventPhase.START, EventPhase.END]

    def test_on_enter_fires_again_after_the_subject_leaves_and_returns(self):
        machine = RuleMachine(_rule(trigger="on_enter", min_duration_s=1.0, rearm_delay_s=10.0))
        signals = self._run(
            machine,
            [(True, 0.0), (True, 2.0), (False, 10.0), (False, 25.0), (True, 30.0), (True, 32.0)],
        )

        assert [s.phase for s in signals] == [EventPhase.START, EventPhase.END, EventPhase.START, EventPhase.END]
        assert signals[2].started_at == 30.0

    def test_on_enter_ignores_a_blip_shorter_than_the_dwell(self):
        machine = RuleMachine(_rule(trigger="on_enter", min_duration_s=5.0))
        assert self._run(machine, [(True, 0.0), (True, 2.0), (False, 3.0)]) == []

    # ----------------------------------------------------------------- on_exit

    def test_on_exit_emits_nothing_on_the_way_in(self):
        """Half the point: an arrival is not an exit, and a rule that opened on arrival
        would leave a row nothing ever closes."""
        machine = RuleMachine(_rule(trigger="on_exit", min_duration_s=1.0, rearm_delay_s=10.0))
        signals = self._run(machine, [(True, 0.0), (True, 5.0), (True, 500.0)])

        assert signals == []
        # And nothing downstream may believe an event is open — the status panel's
        # "Open events" table and `force_close` on shutdown both read this.
        assert machine.is_open is False

    def test_on_exit_fires_at_the_departure_once_it_is_confirmed(self):
        machine = RuleMachine(_rule(trigger="on_exit", min_duration_s=1.0, rearm_delay_s=10.0))
        signals = self._run(machine, [(True, 0.0), (True, 5.0), (False, 60.0), (False, 65.0), (False, 71.0)])

        assert [s.phase for s in signals] == [EventPhase.START, EventPhase.END]
        assert signals[-1].duration_s == 0
        # 60.0, when it stopped matching — not 71.0, when the debounce agreed. The delay
        # decides *whether* it left, not *when*, and a 15s preroll anchored at 71 would
        # record the empty garage instead of the departure.
        assert signals[0].started_at == 60.0
        assert signals[0].at == 60.0

    def test_on_exit_ignores_a_dropout_inside_the_debounce(self):
        """A tracker losing a car for two frames is not the car leaving."""
        machine = RuleMachine(_rule(trigger="on_exit", min_duration_s=1.0, rearm_delay_s=10.0))
        signals = self._run(
            machine,
            [(True, 0.0), (True, 5.0), (False, 20.0), (False, 21.0), (True, 22.0), (True, 30.0)],
        )

        assert signals == []

    def test_on_exit_requires_the_dwell_before_a_departure_counts(self):
        """ "only if the car was in the garage for at least 15 minutes" — a car that
        turned around in the driveway must fire nothing."""
        machine = RuleMachine(_rule(trigger="on_exit", min_duration_s=900.0, rearm_delay_s=10.0))
        brief = self._run(machine, [(True, 0.0), (True, 60.0), (False, 61.0), (False, 80.0)])
        assert brief == []

        stayed = self._run(
            machine,
            [(True, 100.0), (True, 1100.0), (False, 1200.0), (False, 1215.0)],
        )
        assert [s.phase for s in stayed] == [EventPhase.START, EventPhase.END]

    def test_the_cap_does_not_truncate_a_long_dwell(self):
        """The regression that would gut the headline case: a 15-minute dwell under the
        default 300s cap. The cap bounds a *recording*, and an impulse has none."""
        machine = RuleMachine(_rule(trigger="on_exit", min_duration_s=900.0, rearm_delay_s=10.0, max_duration_s=300.0))
        signals = self._run(
            machine,
            [(True, 0.0), (True, 950.0), (True, 2000.0), (False, 3000.0), (False, 3011.0)],
        )

        assert [s.phase for s in signals] == [EventPhase.START, EventPhase.END]
        assert signals[0].started_at == 3000.0
        assert signals[-1].truncated is False

    def test_stopping_the_watcher_does_not_fire_an_exit(self):
        """Shutdown is not the subject leaving. `force_close` exists to close what is
        open, and a rule mid-dwell has nothing open."""
        machine = RuleMachine(_rule(trigger="on_exit", min_duration_s=1.0))
        self._run(machine, [(True, 0.0), (True, 5.0)])

        assert machine.force_close(now=6.0) == []

    def test_disarming_mid_dwell_fires_nothing(self):
        """Same reasoning as shutdown, by the other route into the machine."""
        spec = _rule(trigger="on_exit", min_duration_s=1.0)
        machine = RuleMachine(spec)
        self._run(machine, [(True, 0.0), (True, 5.0)])

        machine.spec = replace(spec, armed=False)
        assert self._run(machine, [(True, 6.0)]) == []


PERSON = "label-person"
ALICE = "label-alice"
PERSON_GROUP = {PERSON: PERSON, ALICE: PERSON}


class TestSustainedRelabelStillOpensOneEvent:
    """The incident this whole subsystem was reworked for.

    One person crossed a yard once and produced three events and three clips. The flat
    model scored `Person` and `Alice` on the same box, stage-2 NMS emitted whichever won
    that frame, and the wrong-label runs lasted 2.4 s and 7.5 s — comfortably past the
    2 s `min_duration_s` that `test_a_brief_relabel_does_not_open_an_event_for_the_
    borrowed_label` relies on. Hysteresis alone cannot fix this: by the time the rule
    engine sees two labels the damage is done, so the fix is upstream, in the tracker.

    Driven through a real tracker rather than hand-built tracks, because the whole claim
    under test is that the tracker no longer hands the engine two identities.
    """

    def _present(self, refs, **overrides):
        kwargs = dict(
            relation="present",
            subject_a_refs=list(refs),
            subject_b_kind="none",
            subject_b_refs=[],
            threshold=0.6,
            min_duration_s=2.0,
            rearm_delay_s=10.0,
            max_duration_s=0.0,
        )
        kwargs.update(overrides)
        return _rule(**kwargs)

    def _run(self, tracker, engine, frames=60, dt=0.25):
        """A subject walking across frame whose label alternates every sample."""
        emitted = []
        for i in range(frames):
            label, rival = (PERSON, ALICE) if i % 2 == 0 else (ALICE, PERSON)
            det = Detection(
                bbox=(10.0 + i * 3, 400.0, 110.0 + i * 3, 600.0),
                label_uuid=label,
                label_name=label,
                confidence=0.9,
                alternates=[(rival, 0.7)],
            )
            now = i * dt
            tracks = merge_duplicate_tracks(tracker.update([det], now), now)
            emitted.extend(engine.step(tracks, now))
        return emitted

    def test_one_subject_one_event(self):
        engine = WatchEngine([self._present([PERSON, ALICE])], source_uuid="src-1")
        tracker = IouTracker(min_hits=3, recovery_margin=1.0, label_groups=PERSON_GROUP)

        emitted = self._run(tracker, engine)

        starts = [s for s in emitted if s.phase is EventPhase.START]
        assert len(starts) == 1, "one person walking through must not open an event per spelling of their name"
        assert starts[0].subject_ref == PERSON

    def test_the_subject_is_tracked_as_one_object_throughout(self):
        engine = WatchEngine([self._present([PERSON, ALICE])], source_uuid="src-1")
        tracker = IouTracker(min_hits=3, recovery_margin=1.0, label_groups=PERSON_GROUP)

        self._run(tracker, engine)

        assert len({t.track_id for t in tracker.all_tracks()}) == 1
