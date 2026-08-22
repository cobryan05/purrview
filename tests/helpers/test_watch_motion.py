"""Unit tests for the `motion` relation — frame differencing and its rule wiring.

Synthetic frames, so every assertion is about the measurement rather than about a
camera. The properties pinned here are the ones whose failure is silent: a masked
region that still contributes to the ratio, a first frame that reads as whole-scene
movement, and a motion rule that needs a detection before it can fire.
"""

import numpy as np
import pytest

from apps.helpers.watch.geometry import Zone
from apps.helpers.watch.motion import DEFAULT_SENSITIVITY, MotionDetector
from apps.helpers.watch.rules import RuleSpec, evaluate_predicate

FRAME_H, FRAME_W = 100, 100

# Left half, right half, and a small square inside the left half.
LEFT_MASK = Zone(uuid="mask-left", name="left", kind="box", points=[(0.0, 0.0), (0.5, 1.0)])
LEFT_ZONE = Zone(uuid="zone-left", name="left", kind="box", points=[(0.0, 0.0), (0.5, 1.0)])
RIGHT_ZONE = Zone(uuid="zone-right", name="right", kind="box", points=[(0.5, 0.0), (1.0, 1.0)])


def _frame(value: int = 0) -> np.ndarray:
    return np.full((FRAME_H, FRAME_W, 3), value, dtype=np.uint8)


def _with_patch(base: int, x1: int, y1: int, x2: int, y2: int, value: int) -> np.ndarray:
    """A frame of `base` with one rectangle painted `value`."""
    frame = _frame(base)
    frame[y1:y2, x1:x2] = value
    return frame


def _motion_rule(**overrides) -> RuleSpec:
    spec = dict(
        uuid="rule-motion",
        name="something moved",
        subject_a_kind="any",
        subject_a_refs=[],
        relation="motion",
        subject_b_kind="none",
        subject_b_refs=[],
        threshold=0.02,
        min_duration_s=0.0,
        rearm_delay_s=0.0,
    )
    spec.update(overrides)
    return RuleSpec(**spec)


class TestMotionMeasurement:
    def test_the_first_frame_reports_no_motion(self):
        """A watcher must not raise an event the instant it starts.

        There is nothing to difference against on frame one, and treating the absent
        previous frame as black would read the whole scene as having just appeared.
        """
        detector = MotionDetector()
        field = detector.update(_frame(0))
        assert field.measured is False
        assert field.ratio(DEFAULT_SENSITIVITY) == 0.0

    def test_a_still_scene_reports_no_motion(self):
        detector = MotionDetector()
        detector.update(_frame(30))
        field = detector.update(_frame(30))
        assert field.measured is True
        assert field.ratio(DEFAULT_SENSITIVITY) == 0.0

    def test_the_ratio_is_the_changed_fraction_of_the_frame(self):
        detector = MotionDetector()
        detector.update(_frame(0))
        # A 20x100 stripe down the left edge: a fifth of the frame.
        field = detector.update(_with_patch(0, 0, 0, 20, FRAME_H, 255))
        assert field.ratio(DEFAULT_SENSITIVITY) == pytest.approx(0.2, abs=0.03)

    def test_a_reconnect_reset_makes_the_next_frame_measure_nothing(self):
        """The frames either side of an outage can be minutes apart; differencing them
        reports the whole scene as moving at the exact moment a camera comes back."""
        detector = MotionDetector()
        detector.update(_frame(0))
        detector.reset()
        field = detector.update(_frame(255))
        assert field.measured is False
        assert field.ratio(DEFAULT_SENSITIVITY) == 0.0


class TestMasks:
    def test_motion_inside_a_mask_is_not_reported(self):
        detector = MotionDetector(masks=[LEFT_MASK])
        detector.update(_frame(0))
        # Change confined to the masked left half.
        field = detector.update(_with_patch(0, 0, 0, 40, FRAME_H, 255))
        assert field.ratio(DEFAULT_SENSITIVITY) == pytest.approx(0.0, abs=0.01)

    def test_a_mask_shrinks_the_measured_area_rather_than_only_the_numerator(self):
        """The property masks exist for: a mask over half the view must not dilute the
        ratio for the other half.

        The same 20-pixel stripe on the right is a fifth of the whole frame but two
        fifths of what is left once the left half is masked. Suppressing masked motion
        after the fact would report 0.2 in both cases, and a camera whose right half is
        a motorway would sit permanently just under its threshold.
        """
        change = _with_patch(0, 60, 0, 80, FRAME_H, 255)

        unmasked = MotionDetector()
        unmasked.update(_frame(0))
        without_mask = unmasked.update(change).ratio(DEFAULT_SENSITIVITY)

        masked = MotionDetector(masks=[LEFT_MASK])
        masked.update(_frame(0))
        with_mask = masked.update(change).ratio(DEFAULT_SENSITIVITY)

        assert without_mask == pytest.approx(0.2, abs=0.03)
        assert with_mask == pytest.approx(2 * without_mask, rel=0.1)


class TestZones:
    def test_motion_is_attributed_to_the_zone_it_happened_in(self):
        detector = MotionDetector(zones=[LEFT_ZONE, RIGHT_ZONE])
        detector.update(_frame(0))
        field = detector.update(_with_patch(0, 60, 0, 80, FRAME_H, 255))

        assert field.ratio(DEFAULT_SENSITIVITY, [RIGHT_ZONE.uuid]) > 0.3
        assert field.ratio(DEFAULT_SENSITIVITY, [LEFT_ZONE.uuid]) == pytest.approx(0.0, abs=0.01)

    def test_several_zones_read_as_any_of_them_and_name_the_strongest(self):
        detector = MotionDetector(zones=[LEFT_ZONE, RIGHT_ZONE])
        detector.update(_frame(0))
        field = detector.update(_with_patch(0, 60, 0, 80, FRAME_H, 255))

        ratio, zone_uuid = field.best(DEFAULT_SENSITIVITY, [LEFT_ZONE.uuid, RIGHT_ZONE.uuid])
        assert zone_uuid == RIGHT_ZONE.uuid
        assert ratio > 0.3

    def test_a_zone_fully_inside_a_mask_can_never_report_motion(self):
        """Zone rasters are intersected with the unmasked area, so a mask wins over a
        zone drawn on top of it rather than the two disagreeing."""
        detector = MotionDetector(masks=[LEFT_MASK], zones=[LEFT_ZONE])
        detector.update(_frame(0))
        field = detector.update(_with_patch(0, 0, 0, 40, FRAME_H, 255))
        assert field.ratio(DEFAULT_SENSITIVITY, [LEFT_ZONE.uuid]) == 0.0


class TestSensitivity:
    def test_a_faint_change_counts_only_below_its_sensitivity(self):
        """Sensitivity is what separates a dark animal from sensor noise; if it did not
        gate the count, the field would report the same ratio at every setting."""
        detector = MotionDetector()
        detector.update(_frame(100))
        # A 25% -> 112 step: about 12/255, so it clears a 0.02 sensitivity and not a 0.3.
        field = detector.update(_with_patch(100, 0, 0, 50, FRAME_H, 112))

        assert field.ratio(0.02) > 0.3
        assert field.ratio(0.3) == pytest.approx(0.0, abs=0.01)


class TestMotionPredicate:
    def test_motion_fires_with_no_tracks_at_all(self):
        """The whole point of the relation: it must not need a model to have detected
        something first, so it cannot go through the subject lookup."""
        detector = MotionDetector()
        detector.update(_frame(0))
        field = detector.update(_with_patch(0, 0, 0, 40, FRAME_H, 255))

        result = evaluate_predicate(
            _motion_rule(), tracks=[], zones={}, frame_width=FRAME_W, frame_height=FRAME_H, motion=field
        )
        assert result.holds is True
        assert result.track_ids == []

    def test_motion_below_the_area_threshold_does_not_hold(self):
        detector = MotionDetector()
        detector.update(_frame(0))
        # A 5x5 patch — 0.25% of the frame, well under a 2% threshold.
        field = detector.update(_with_patch(0, 0, 0, 5, 5, 255))

        result = evaluate_predicate(
            _motion_rule(threshold=0.02), tracks=[], zones={}, frame_width=FRAME_W, frame_height=FRAME_H, motion=field
        )
        assert result.holds is False

    def test_a_motion_rule_never_holds_when_nothing_is_differencing(self):
        """A pipeline that built no detector passes None. That must read as "not
        measured", not as "no motion is a match for a threshold of zero"."""
        result = evaluate_predicate(
            _motion_rule(threshold=0.0), tracks=[], zones={}, frame_width=FRAME_W, frame_height=FRAME_H, motion=None
        )
        assert result.holds is False

    def test_a_zone_scoped_motion_rule_reports_which_zone_matched(self):
        detector = MotionDetector(zones=[LEFT_ZONE, RIGHT_ZONE])
        detector.update(_frame(0))
        field = detector.update(_with_patch(0, 60, 0, 80, FRAME_H, 255))

        spec = _motion_rule(subject_a_kind="zone", subject_a_refs=[LEFT_ZONE.uuid, RIGHT_ZONE.uuid])
        result = evaluate_predicate(spec, tracks=[], zones={}, frame_width=FRAME_W, frame_height=FRAME_H, motion=field)
        assert result.holds is True
        assert result.subject_ref == RIGHT_ZONE.uuid
