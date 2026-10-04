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
from apps.helpers.tracking.tracker import (
    Detection,
    LabelLifetimeSpec,
    Track,
    Tracker,
    merge_duplicate_tracks,
)

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

    def test_blips_separated_by_misses_never_confirm(self):
        """`min_hits` is "in a row", not a lifetime total.

        A counter that only ever grows is not a false-alarm guard: each sighting revives
        the same track inside its TTL, so three unrelated flickers minutes apart confirm
        an object that was never there for two consecutive frames. Observed as a floating
        spider web opening a hundred garage events overnight.
        """
        tracker = IouTracker(min_hits=3, max_age_s=6.0)
        for i in range(6):
            # One detection, then a miss, repeatedly. The track survives its TTL
            # throughout, so nothing ever expires it between blips.
            assert tracker.update([_det(0, 0, 10, 10)], now=i * 2.0) == []
            assert tracker.update([], now=i * 2.0 + 1.0) == []

    def test_a_confirmed_track_stays_confirmed_through_a_miss(self):
        """Dropout tolerance is `max_age_s`'s job — a miss must not de-confirm."""
        tracker = IouTracker(min_hits=2, max_age_s=6.0)
        _feed(tracker, [[_det(0, 0, 10, 10)]] * 2)
        assert len(tracker.update([], now=1.0)) == 1

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
        """A non-anchor track always takes the new box, however weakly it was detected.

        0.4 is deliberately below `anchor_confidence` (0.6) — which is the contrast this
        test exists to draw — but above `min_confidence` (0.25), below which the tracker
        discards a detection outright and there is no box to take.
        """
        tracker = IouTracker(min_hits=1, label_lifetimes={CAT: LabelLifetimeSpec(ttl_s=2.0)})
        tracker.update([_det(0, 0, 20, 20, conf=0.9)], now=0.0)
        tracks = tracker.update([_det(2, 2, 22, 22, conf=0.4)], now=0.2)
        assert tracks[0].bbox == (2, 2, 22, 22)


class TestMinConfidence:
    """The task asks inference for a low floor so ByteTrack can use weak boxes; this
    tracker has no second association pass, so it applies its own threshold."""

    def test_detections_below_min_confidence_are_ignored(self):
        tracker = IouTracker(min_hits=1, min_confidence=0.25)

        assert tracker.update([_det(0, 0, 20, 20, conf=0.1)], now=0.0) == []

    def test_a_weak_detection_does_not_refresh_a_track(self):
        tracker = IouTracker(min_hits=1, min_confidence=0.25)
        tracker.update([_det(0, 0, 20, 20, conf=0.9)], now=0.0)

        tracker.update([_det(0, 0, 20, 20, conf=0.1)], now=0.5)

        # Coasting on its TTL, not refreshed — the weak box was never seen.
        assert tracker.all_tracks()[0].last_seen == 0.0

    def test_the_threshold_is_configurable(self):
        tracker = IouTracker(min_hits=1, min_confidence=0.05)

        assert len(tracker.update([_det(0, 0, 20, 20, conf=0.1)], now=0.0)) == 1


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


class TestMatchedThisUpdate:
    """`Track.matched_this_update` is the only reliable "did the detector miss
    this specific track just now" signal — `last_seen` alone can't answer that
    without the caller separately tracking `now` (COLLECTOR_UNIFICATION_PLAN.md
    §4A, detector-miss collection)."""

    def test_a_freshly_created_track_is_matched(self):
        tracker = IouTracker(min_hits=1)
        (track,) = tracker.update([_det(0, 0, 10, 10)], now=0.0)
        assert track.matched_this_update is True

    def test_a_track_re_matched_next_frame_stays_matched(self):
        tracker = IouTracker(min_hits=1)
        tracker.update([_det(0, 0, 10, 10)], now=0.0)
        (track,) = tracker.update([_det(1, 1, 11, 11)], now=0.2)
        assert track.matched_this_update is True

    def test_a_coasting_track_is_reported_unmatched(self):
        tracker = IouTracker(min_hits=1, max_age_s=5.0)
        tracker.update([_det(0, 0, 10, 10)], now=0.0)
        (track,) = tracker.update([], now=0.2)
        assert track.matched_this_update is False

    def test_re_acquisition_flips_it_back_to_matched(self):
        tracker = IouTracker(min_hits=1, max_age_s=5.0)
        tracker.update([_det(0, 0, 10, 10)], now=0.0)
        (track,) = tracker.update([], now=0.2)
        assert track.matched_this_update is False
        (track,) = tracker.update([_det(1, 1, 11, 11)], now=0.4)
        assert track.matched_this_update is True


class TestRecoveryMargin:
    """The second, forgiving association pass.

    Requiring overlap is what stops a track jumping across the image, and it is also
    what loses a subject that merely moved: below a few fps a cat can travel more than
    its own width between samples, so consecutive detections of it fail to overlap
    *each other* and each starts a track that never confirms.
    """

    @staticmethod
    def _walk(tracker, step, frames=6, size=10.0):
        """Move one box `step` px per frame and return the distinct confirmed ids."""
        ids = set()
        for i in range(frames):
            x = i * step
            for track in tracker.update([_det(x, 0, x + size, size)], now=i * 1.0):
                ids.add(track.track_id)
        return ids

    def test_off_by_default(self):
        """Nothing changes until it is asked for."""
        assert IouTracker()._recovery_margin == 0.0

    def test_a_subject_moving_its_own_width_is_lost_without_it(self):
        # 10px box moving 12px per frame: consecutive boxes never overlap, so every
        # frame starts a fresh track and min_hits is never reached.
        assert self._walk(IouTracker(min_hits=3), step=12.0) == set()

    def test_the_same_subject_is_held_by_one_track_with_it(self):
        ids = self._walk(IouTracker(min_hits=3, recovery_margin=1.5), step=12.0)
        assert len(ids) == 1

    def test_two_stationary_objects_of_one_label_stay_apart(self):
        """The case a loosened rule would be most likely to ruin.

        Both match exactly in the strict pass, so the forgiving pass never sees them —
        that ordering is the whole safety argument, and a margin far wider than the gap
        between them must not merge them.
        """
        tracker = IouTracker(min_hits=1, recovery_margin=3.0)
        tracks = []
        for i in range(4):
            tracks = tracker.update([_det(0, 0, 10, 10), _det(14, 0, 24, 10)], now=i * 1.0)
        assert len({t.track_id for t in tracks}) == 2

    def test_a_vanished_object_does_not_steal_its_neighbour(self):
        """One of two same-label boxes disappearing must not hand its track the other.

        The survivor is already claimed by its own track in the strict pass, so the
        orphaned track has nothing left to match. It stays where it was and ages out —
        what it must never do is jump onto the box that is still there.
        """
        tracker = IouTracker(min_hits=1, recovery_margin=3.0)
        left, right = (0, 0, 10, 10), (14, 0, 24, 10)
        for i in range(3):
            tracks = tracker.update([_det(*left), _det(*right)], now=i * 1.0)
        before = {t.bbox: t.track_id for t in tracks}

        after = {t.track_id: t.bbox for t in tracker.update([_det(*right)], now=3.0)}

        # The survivor keeps the id it already had...
        assert after[before[right]] == right
        # ...and the orphan is still sitting where it was, not on top of the survivor.
        assert after[before[left]] == left

    def test_margin_cannot_reach_across_the_image(self):
        """Bounded tolerance, not unbounded. A box the far side of the frame is never
        the same object, however forgiving the margin."""
        tracker = IouTracker(min_hits=1, recovery_margin=3.0)
        near = tracker.update([_det(0, 0, 10, 10)], now=0.0)
        near_id = near[0].track_id

        far = (900, 0, 910, 10)
        moved = tracker.update([_det(*far)], now=1.0)

        owner = next(t.track_id for t in moved if t.bbox == far)
        assert owner != near_id


def _track(track_id, x1, y1, x2, y2, conf=0.9, label=CAT) -> Track:
    return Track(track_id=track_id, bbox=(x1, y1, x2, y2), label_uuid=label, confidence=conf, confirmed=True)


class TestMergeDuplicateTracks:
    """Regression coverage for the ghost-track bug: a tracker spawning two live ids for
    one physical cat, each drawn and classified independently every frame until (if
    ever) a slow, offline occlusion-merge heuristic notices. See PR discussion — real
    saved training frames were found with two ~0.9+ IOU 'Luna' boxes stacked on one cat,
    traced back to two simultaneously-alive tracks, one of which briefly disagreed on
    the cat's identity before correction.
    """

    def test_drops_the_lower_confidence_near_duplicate(self):
        tracks = [_track(1, 0, 0, 100, 100, conf=0.73), _track(2, 2, 2, 98, 99, conf=0.57)]
        kept = merge_duplicate_tracks(tracks, now=0.0)
        assert [t.track_id for t in kept] == [1]

    def test_merges_across_disagreeing_labels(self):
        """The exact bug found in production: one track guessed Maple, the other Willow,
        both boxes on the same physical cat. A same-label-only check would miss this."""
        tracks = [
            _track(1, 0, 0, 100, 100, conf=0.47, label="label-maple"),
            _track(2, 3, 3, 97, 98, conf=0.46, label="label-willow"),
        ]
        kept = merge_duplicate_tracks(tracks, now=0.0)
        assert len(kept) == 1
        assert kept[0].track_id == 1

    def test_distinct_animals_side_by_side_both_survive(self):
        """Low overlap must never be treated as the same object."""
        tracks = [_track(1, 0, 0, 10, 10), _track(2, 14, 0, 24, 10)]
        kept = merge_duplicate_tracks(tracks, now=0.0)
        assert {t.track_id for t in kept} == {1, 2}

    def test_three_way_duplicate_keeps_only_the_strongest(self):
        tracks = [
            _track(1, 0, 0, 100, 100, conf=0.40),
            _track(2, 1, 1, 99, 99, conf=0.90),
            _track(3, 2, 2, 98, 98, conf=0.60),
        ]
        kept = merge_duplicate_tracks(tracks, now=0.0)
        assert [t.track_id for t in kept] == [2]


PERSON = "label-person"
ALICE = "label-alice"
DOG = "label-dog"

#: `Person` is an exclusive-variants group root and `Alice` one of its variants, the way
#: the live label tree has it. `Cat` is its own group and shares nothing with either.
PERSON_GROUP = {PERSON: PERSON, ALICE: PERSON, CAT: CAT, DOG: DOG}


class TestGroupAwareAssociation:
    """One physical object whose label alternates must stay one track.

    The production failure: a flat model scores `Person` and `Alice` on the same anchor,
    stage-2 NMS emits whichever won that frame, and a label-gated tracker read the
    alternation as two objects — nine tracks and three clips for one person crossing a
    yard.
    """

    def test_alternating_group_labels_stay_one_track(self):
        tracker = IouTracker(min_hits=1, label_groups=PERSON_GROUP)
        labels = [PERSON, ALICE, PERSON, ALICE, PERSON]
        for i, label in enumerate(labels):
            # Drifting, as a walking subject does: this is what defeated the duplicate
            # merge, which needs 0.85 IoU and cannot collapse a frozen box against a
            # moving one.
            tracks = tracker.update([_det(i * 2, 0, 100 + i * 2, 100, label=label)], now=i * 0.2)
        assert len(tracks) == 1

    def test_ungrouped_labels_still_never_share_a_track(self):
        """The gate loosened only within a group. `Cat` against `Dog` is still refused."""
        tracker = IouTracker(min_hits=1, label_groups=PERSON_GROUP)
        tracker.update([_det(0, 0, 100, 100, label=CAT)], now=0.0)
        tracks = tracker.update([_det(0, 0, 100, 100, label=DOG)], now=0.2)
        assert len({t.track_id for t in tracks}) == 2

    def test_without_a_group_map_association_is_label_equality(self):
        """No `exclusive_variants` anywhere in the tree must behave exactly as before."""
        tracker = IouTracker(min_hits=1)
        tracker.update([_det(0, 0, 100, 100, label=PERSON)], now=0.0)
        tracks = tracker.update([_det(0, 0, 100, 100, label=ALICE)], now=0.2)
        assert len({t.track_id for t in tracks}) == 2

    def test_a_group_mate_does_not_rename_the_track(self):
        """The caption must not contradict the uuid every rule matches on."""
        tracker = IouTracker(min_hits=1, label_groups=PERSON_GROUP)
        tracker.update([Detection((0, 0, 100, 100), PERSON, "Person", 0.9)], now=0.0)
        tracks = tracker.update([Detection((0, 0, 100, 100), ALICE, "Alice", 0.9)], now=0.2)
        assert len(tracks) == 1
        assert tracks[0].label_name == "Person"


class TestCoastingConfidenceDecay:
    """A track that stopped matching must stop out-ranking the one that replaced it."""

    def test_stale_high_confidence_loses_to_a_live_lower_one(self):
        stale = _track(1, 0, 0, 100, 100, conf=0.90)
        stale.last_seen = 0.0
        live = _track(2, 2, 2, 98, 99, conf=0.54)
        live.last_seen = 1.0

        assert [t.track_id for t in merge_duplicate_tracks([stale, live], now=0.0)] == [1]
        assert [t.track_id for t in merge_duplicate_tracks([stale, live], now=1.0)] == [2]

    def test_equally_fresh_tracks_are_still_ranked_by_confidence(self):
        weak = _track(1, 0, 0, 100, 100, conf=0.40)
        strong = _track(2, 1, 1, 99, 99, conf=0.90)
        for t in (weak, strong):
            t.last_seen = 5.0
        assert [t.track_id for t in merge_duplicate_tracks([weak, strong], now=5.0)] == [2]

    def test_a_tie_on_confidence_keeps_the_better_established_track(self):
        established = _track(1, 0, 0, 100, 100, conf=0.7)
        established.hits = 40
        newcomer = _track(2, 1, 1, 99, 99, conf=0.7)
        newcomer.hits = 1
        assert [t.track_id for t in merge_duplicate_tracks([newcomer, established], now=0.0)] == [1]


def _group_det(x1, label, alternates=(), conf=0.9):
    return Detection(
        bbox=(x1, 0, x1 + 100, 100),
        label_uuid=label,
        label_name=label,
        confidence=conf,
        alternates=list(alternates),
    )


class TestLabelVote:
    """A track's identity is a decaying vote, not the label of its first frame.

    The originating question: if a track is called `Luna` fifty times and `Cat` once,
    does it switch? It must not — and one `Cat` frame must not fork a second track
    either, which is what `TestGroupAwareAssociation` covers.
    """

    def _alternating(self, tracker, first, second, frames=24, dt=0.2):
        """A subject whose label alternates every frame, as a tied argmax does."""
        tracks = []
        for i in range(frames):
            winner, loser = (first, second) if i % 2 == 0 else (second, first)
            det = _group_det(i * 2, winner, alternates=[(loser, 0.70)], conf=0.90)
            tracks = tracker.update([det], now=i * dt)
        return tracks

    def test_alternation_resolves_to_the_group_root(self):
        """Neither variant out-scores the other, so the honest answer is `Person`."""
        tracker = IouTracker(min_hits=1, label_groups=PERSON_GROUP)
        tracks = self._alternating(tracker, PERSON, ALICE)
        assert len(tracks) == 1
        assert tracks[0].label_uuid == PERSON

    def test_a_track_born_on_the_wrong_variant_still_resolves_to_the_root(self):
        """The first frame must not get to name the object for the rest of its life."""
        tracker = IouTracker(min_hits=1, label_groups=PERSON_GROUP)
        tracks = self._alternating(tracker, ALICE, PERSON)
        assert tracks[0].label_uuid == PERSON

    def test_one_stray_frame_does_not_unseat_a_settled_identity(self):
        """Fifty `Alice` frames and one `Person` frame is still Alice."""
        tracker = IouTracker(min_hits=1, label_groups=PERSON_GROUP)
        tracks = []
        for i in range(50):
            tracks = tracker.update([_group_det(i, ALICE, alternates=[(PERSON, 0.15)], conf=0.95)], now=i * 0.2)
        assert tracks[0].label_uuid == ALICE

        tracks = tracker.update([_group_det(50, PERSON, alternates=[(ALICE, 0.60)], conf=0.7)], now=10.0)
        assert tracks[0].label_uuid == ALICE

    def test_a_sustained_identity_change_is_accepted(self):
        """The negative of the test above: smoothing must not become deafness.

        Asserted as a pair with `test_alternation_resolves_to_the_group_root` — either
        assertion alone is satisfiable by an implementation that never moves, or by one
        that moves on every frame.
        """
        tracker = IouTracker(min_hits=1, label_groups=PERSON_GROUP)
        for i in range(30):
            tracker.update([_group_det(i, PERSON, alternates=[(ALICE, 0.2)], conf=0.9)], now=i * 0.2)

        tracks = []
        for i in range(30, 150):
            tracks = tracker.update([_group_det(i, ALICE, alternates=[(PERSON, 0.15)], conf=0.95)], now=i * 0.2)
        assert tracks[0].label_uuid == ALICE

    def test_the_settled_label_carries_a_matching_name(self):
        """The caption must never contradict the uuid the rules match on."""
        tracker = IouTracker(min_hits=1, label_groups=PERSON_GROUP, label_names={PERSON: "Person", ALICE: "Alice"})
        tracks = self._alternating(tracker, ALICE, PERSON)
        assert (tracks[0].label_uuid, tracks[0].label_name) == (PERSON, "Person")

    def test_the_raw_observation_is_kept_beside_the_verdict(self):
        """Smoothing the report must not erase what the detector actually said —
        that disagreement is the training signal the whole feature exists to harvest."""
        tracker = IouTracker(min_hits=1, label_groups=PERSON_GROUP)
        # An even frame count ends on the second label, so the last thing the detector
        # said disagrees with the verdict — which is the case being asserted.
        tracks = self._alternating(tracker, PERSON, ALICE, frames=24)
        track = tracks[0]
        assert track.label_uuid == PERSON
        assert track.observed_label_uuid == ALICE
        assert track.observed_margin == pytest.approx(0.20)

    def test_a_coasting_track_records_no_observation(self):
        """ "The detector disagreed" and "the detector said nothing" are different
        faults with different fixes, and must not be confused downstream."""
        tracker = IouTracker(min_hits=1, max_age_s=5.0, label_groups=PERSON_GROUP)
        tracker.update([_group_det(0, PERSON)], now=0.0)
        tracks = tracker.update([], now=0.2)
        assert tracks[0].observed_label_uuid is None
        assert tracks[0].observed_margin is None

    def test_an_ungrouped_label_is_unaffected(self):
        tracker = IouTracker(min_hits=1, label_groups=PERSON_GROUP)
        tracks = []
        for i in range(5):
            tracks = tracker.update([_group_det(i, DOG, conf=0.8)], now=i * 0.2)
        assert tracks[0].label_uuid == DOG
