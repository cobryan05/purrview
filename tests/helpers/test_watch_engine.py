"""Unit tests for watch geometry, rule hysteresis and the engine.

Everything here is pure, which is the point: this is where the feature's real bugs
live, and a synthetic detection sequence pins them down in milliseconds.

The hysteresis tests are the important ones. Without min-duration / rearm-delay /
max-duration a raw threshold flaps many times per second, and the visible symptom is
an MQTT storm plus a dozen three-second clips for one litter-box visit.
"""

import pytest

from apps.helpers.tracking.tracker import Track
from apps.helpers.watch.engine import WatchEngine
from apps.helpers.watch.events import EventPhase
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
)
from apps.helpers.watch.rules import RuleMachine, RuleSpec, RuleState, evaluate_predicate

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
        signal = machine.force_close(5.0)
        assert signal is not None and signal.phase is EventPhase.END
        assert signal.truncated is True

    def test_force_close_on_an_idle_rule_emits_nothing(self):
        assert RuleMachine(_rule()).force_close(1.0) is None

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
