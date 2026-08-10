"""Unit tests for the tracker registry and IouTracker.

Pure and synthetic by design: the real bugs in a tracker are association mistakes and
lifetime arithmetic, both of which are far easier to pin down against a hand-built
detection sequence than against a live camera.
"""

import pytest

from apps.helpers.tracking.iouTracker import IouTracker, iou
from apps.helpers.tracking.Registry import (
    create_tracker,
    register_tracker,
    tracker_registry,
)
from apps.helpers.tracking.tracker import Detection, LabelLifetimeSpec, Track, Tracker

CAT = "label-cat"
BOX = "label-litterbox"


def _det(x1, y1, x2, y2, label=CAT, conf=0.9) -> Detection:
    return Detection(bbox=(x1, y1, x2, y2), label_uuid=label, label_name=label, confidence=conf)


def _feed(tracker: IouTracker, det_lists, start=0.0, dt=0.2):
    """Feed successive frames, returning the tracks after the last one."""
    tracks = []
    for i, dets in enumerate(det_lists):
        tracks = tracker.update(dets, now=start + i * dt)
    return tracks


class TestIou:
    def test_identical_boxes(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_touching_edges_do_not_overlap(self):
        assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_half_overlap(self):
        # Two 10x10 boxes offset by 5 in x: intersection 50, union 150.
        assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)

    def test_small_box_inside_large_one_scores_low(self):
        """The reason `contains` exists as a relation and IoU is the wrong default.

        A cat inside a litter box is perfectly contained, yet IoU is tiny.
        """
        assert iou((40, 40, 50, 50), (0, 0, 100, 100)) < 0.02

    def test_zero_area_box_does_not_divide_by_zero(self):
        assert iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


class TestRegistry:
    def test_iou_tracker_is_registered(self):
        assert "IouTracker" in tracker_registry

    def test_create_by_name(self):
        assert isinstance(create_tracker("IouTracker"), IouTracker)

    def test_params_are_passed_through(self):
        tracker = create_tracker("IouTracker", iou_threshold=0.7)
        assert tracker._iou_threshold == 0.7

    def test_unknown_tracker_raises_rather_than_defaulting(self):
        # Silently substituting a different tracker would change detection behaviour
        # with no signal to the user.
        with pytest.raises(KeyError):
            create_tracker("ByteTrack")

    def test_registration_decorator_adds_to_registry(self):
        @register_tracker("_TestOnlyTracker")
        class _TestOnlyTracker(Tracker):
            def update(self, detections, now):
                return []

            def reset(self):
                pass

        try:
            assert create_tracker("_TestOnlyTracker") is not None
        finally:
            tracker_registry.pop("_TestOnlyTracker", None)

    def test_params_schema_is_exposed_for_the_ui(self):
        schema = IouTracker.params_schema()
        assert "iou_threshold" in schema and "max_age_s" in schema
        # The UI renders these directly, so they need types and defaults.
        assert all("type" in field and "default" in field for field in schema.values())


class TestConfirmation:
    def test_a_single_detection_does_not_produce_a_track(self):
        """min_hits guards against one spurious detection opening an event."""
        tracker = IouTracker(min_hits=3)
        assert tracker.update([_det(0, 0, 10, 10)], now=0.0) == []

    def test_a_track_is_confirmed_after_min_hits(self):
        tracker = IouTracker(min_hits=3)
        tracks = _feed(tracker, [[_det(0, 0, 10, 10)]] * 3)
        assert len(tracks) == 1
        assert tracks[0].hits == 3

    def test_min_hits_of_one_confirms_immediately(self):
        tracker = IouTracker(min_hits=1)
        assert len(tracker.update([_det(0, 0, 10, 10)], now=0.0)) == 1


class TestAssociation:
    def test_a_moving_object_keeps_its_track_id(self):
        tracker = IouTracker(min_hits=1, iou_threshold=0.3)
        frames = [[_det(0, 0, 20, 20)], [_det(2, 0, 22, 20)], [_det(4, 0, 24, 20)]]
        ids = [tracker.update(f, now=i * 0.2)[0].track_id for i, f in enumerate(frames)]
        assert len(set(ids)) == 1

    def test_a_jump_beyond_the_threshold_starts_a_new_track(self):
        tracker = IouTracker(min_hits=1, iou_threshold=0.3, max_age_s=10.0)
        tracker.update([_det(0, 0, 20, 20)], now=0.0)
        tracks = tracker.update([_det(500, 500, 520, 520)], now=0.2)
        assert len({t.track_id for t in tracks}) == 2

    def test_labels_never_swap_track_ids(self):
        """A cat detection must not inherit the litter box's identity."""
        tracker = IouTracker(min_hits=1)
        tracker.update([_det(0, 0, 100, 100, label=BOX)], now=0.0)
        tracks = tracker.update([_det(0, 0, 100, 100, label=CAT)], now=0.2)
        by_label = {t.label_uuid: t.track_id for t in tracks}
        assert by_label[CAT] != by_label[BOX]

    def test_two_objects_get_two_tracks(self):
        tracker = IouTracker(min_hits=1)
        tracks = tracker.update([_det(0, 0, 20, 20), _det(200, 200, 220, 220)], now=0.0)
        assert len({t.track_id for t in tracks}) == 2

    def test_the_best_overlap_wins_when_two_detections_compete(self):
        tracker = IouTracker(min_hits=1, iou_threshold=0.1)
        tracker.update([_det(0, 0, 20, 20)], now=0.0)
        # The near-identical box should claim the existing track; the far one is new.
        tracks = tracker.update([_det(1, 1, 21, 21), _det(15, 0, 35, 20)], now=0.2)
        matched = next(t for t in tracks if t.track_id == 1)
        assert matched.bbox == (1, 1, 21, 21)


class TestLifetimes:
    def test_a_track_expires_after_its_ttl_in_seconds(self):
        tracker = IouTracker(min_hits=1, max_age_s=2.0)
        tracker.update([_det(0, 0, 20, 20)], now=0.0)
        assert tracker.update([], now=1.5) != []
        assert tracker.update([], now=2.5) == []

    def test_ttl_is_seconds_not_frames(self):
        """The same wallclock gap expires a track regardless of how many frames pass.

        A frame count would mean 1 s on a 5 fps camera and 0.2 s on a 25 fps one.
        """
        slow = IouTracker(min_hits=1, max_age_s=1.0)
        fast = IouTracker(min_hits=1, max_age_s=1.0)
        slow.update([_det(0, 0, 20, 20)], now=0.0)
        fast.update([_det(0, 0, 20, 20)], now=0.0)

        # 2 empty frames over 1.5 s vs 30 empty frames over the same 1.5 s.
        slow.update([], now=0.75)
        assert slow.update([], now=1.5) == []
        for i in range(1, 31):
            fast.update([], now=i * 0.05)
        assert fast.update([], now=1.5) == []

    def test_per_label_ttl_overrides_the_default(self):
        """The litter-box-vs-cat case: same missed frames, different outcome."""
        tracker = IouTracker(
            min_hits=1,
            max_age_s=1.0,
            label_lifetimes={
                CAT: LabelLifetimeSpec(ttl_s=1.0),
                BOX: LabelLifetimeSpec(ttl_s=300.0, is_anchor=True),
            },
        )
        tracker.update([_det(0, 0, 20, 20, label=CAT), _det(50, 50, 150, 150, label=BOX)], now=0.0)
        tracks = tracker.update([], now=5.0)
        assert [t.label_uuid for t in tracks] == [BOX]

    def test_a_redetection_resets_the_age(self):
        tracker = IouTracker(min_hits=1, max_age_s=1.0)
        tracker.update([_det(0, 0, 20, 20)], now=0.0)
        tracker.update([_det(0, 0, 20, 20)], now=0.9)
        assert tracker.update([], now=1.5) != []


class TestAnchors:
    def test_an_anchor_latches_its_box(self):
        """A low-confidence re-detection must not move a latched anchor.

        Box jitter on a stationary object would otherwise make a containment rule
        flicker at the zone boundary.
        """
        tracker = IouTracker(
            min_hits=1,
            anchor_confidence=0.6,
            label_lifetimes={BOX: LabelLifetimeSpec(ttl_s=300.0, is_anchor=True)},
        )
        tracker.update([_det(0, 0, 100, 100, label=BOX, conf=0.9)], now=0.0)
        tracks = tracker.update([_det(5, 5, 105, 105, label=BOX, conf=0.2)], now=0.2)
        assert tracks[0].bbox == (0, 0, 100, 100)
        assert tracks[0].is_anchor is True

    def test_a_confident_redetection_does_move_the_anchor(self):
        tracker = IouTracker(
            min_hits=1,
            anchor_confidence=0.6,
            label_lifetimes={BOX: LabelLifetimeSpec(ttl_s=300.0, is_anchor=True)},
        )
        tracker.update([_det(0, 0, 100, 100, label=BOX, conf=0.9)], now=0.0)
        tracks = tracker.update([_det(5, 5, 105, 105, label=BOX, conf=0.95)], now=0.2)
        assert tracks[0].bbox == (5, 5, 105, 105)

    def test_a_low_confidence_first_sighting_does_not_latch(self):
        tracker = IouTracker(
            min_hits=1,
            anchor_confidence=0.6,
            label_lifetimes={BOX: LabelLifetimeSpec(ttl_s=300.0, is_anchor=True)},
        )
        tracks = tracker.update([_det(0, 0, 100, 100, label=BOX, conf=0.3)], now=0.0)
        assert tracks[0].is_anchor is False

    def test_non_anchor_labels_never_latch(self):
        tracker = IouTracker(min_hits=1, label_lifetimes={CAT: LabelLifetimeSpec(ttl_s=2.0)})
        tracker.update([_det(0, 0, 20, 20, conf=0.9)], now=0.0)
        tracks = tracker.update([_det(2, 2, 22, 22, conf=0.2)], now=0.2)
        assert tracks[0].bbox == (2, 2, 22, 22)


class TestReset:
    def test_reset_forgets_tracks_and_ids(self):
        """Track ids do not survive a restart and must not pretend to."""
        tracker = IouTracker(min_hits=1)
        tracker.update([_det(0, 0, 20, 20)], now=0.0)
        tracker.reset()
        assert tracker.active_tracks() == []
        tracks = tracker.update([_det(0, 0, 20, 20)], now=1.0)
        assert tracks[0].track_id == 1


class TestTrackModel:
    def test_age_and_duration_are_measured_in_seconds(self):
        track = Track(track_id=1, bbox=(0, 0, 1, 1), first_seen=10.0, last_seen=12.0)
        assert track.age_s(15.0) == 3.0
        assert track.duration_s(15.0) == 5.0

    def test_ages_never_go_negative_on_clock_wobble(self):
        track = Track(track_id=1, bbox=(0, 0, 1, 1), first_seen=10.0, last_seen=12.0)
        assert track.age_s(11.0) == 0.0

    def test_to_dict_is_camel_cased_for_the_status_panel(self):
        payload = Track(track_id=7, bbox=(0, 0, 1, 1)).to_dict()
        assert payload["trackId"] == 7
        assert "labelUuid" in payload and "isAnchor" in payload
