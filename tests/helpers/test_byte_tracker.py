"""ByteTracker — the two-stage association tracker.

The tests worth having here are the ones that distinguish it from `IouTracker`; a
tracker that merely follows a confident box across frames would pass an IoU tracker's
suite unchanged and prove nothing. So: low-confidence recovery, the refusal to promote
weak boxes into new tracks, class-awareness, and per-label lifetimes in seconds.
"""

import pytest

from apps.helpers.tracking.byteTracker import ByteTracker
from apps.helpers.tracking.tracker import Detection, LabelLifetimeSpec

CAT = "label-cat"
LITTERBOX = "label-litterbox"


def _det(x: float, y: float, conf: float, label: str = CAT, size: float = 10.0) -> Detection:
    return Detection(bbox=(x, y, x + size, y + size), label_uuid=label, label_name=label, confidence=conf)


def _tracker(**overrides) -> ByteTracker:
    params = dict(min_hits=1, max_age_s=2.0)
    params.update(overrides)
    return ByteTracker(**params)


class TestTwoStageAssociation:
    def test_low_confidence_detection_keeps_a_track_alive(self):
        """The reason this tracker exists.

        A confident cat, then a bad frame where it only scores 0.3. A single-stage
        tracker discards that box; here the track must survive with the same id.
        """
        tracker = _tracker()
        tracker.update([_det(0, 0, 0.9)], now=0.0)
        original_id = tracker.active_tracks()[0].track_id

        tracks = tracker.update([_det(1, 1, 0.3)], now=0.5)

        assert len(tracks) == 1
        assert tracks[0].track_id == original_id, "a weak detection must continue the track, not orphan it"
        assert tracks[0].last_seen == pytest.approx(0.5), "the weak match must refresh the track's age"

    def test_low_confidence_detection_does_not_start_a_track(self):
        """A weak box with no history is noise until proven otherwise."""
        tracker = _tracker()

        assert tracker.update([_det(0, 0, 0.3)], now=0.0) == []
        assert tracker.all_tracks() == []

    def test_below_low_thresh_is_ignored_entirely(self):
        tracker = _tracker()
        tracker.update([_det(0, 0, 0.9)], now=0.0)

        tracker.update([_det(1, 1, 0.02)], now=0.5)

        # Nothing matched, so the track is coasting on its TTL rather than being refreshed.
        assert tracker.all_tracks()[0].last_seen == pytest.approx(0.0)

    def test_weak_box_cannot_steal_a_track_from_a_confident_one(self):
        """Stage 1 runs first precisely so the confident box wins the association.

        Both detections overlap the existing track. The confident one must claim it;
        the weak one then has no unmatched track left to join and is discarded rather
        than promoted, so exactly one track survives.
        """
        tracker = _tracker()
        tracker.update([_det(0, 0, 0.9)], now=0.0)
        original_id = tracker.active_tracks()[0].track_id

        tracker.update([_det(1, 1, 0.95), _det(2, 2, 0.2)], now=0.5)

        tracks = tracker.active_tracks()
        assert len(tracks) == 1
        assert tracks[0].track_id == original_id
        assert tracks[0].confidence == pytest.approx(0.95), "the track took the weak box over the confident one"


class TestClassAwareness:
    def test_a_cat_never_inherits_a_litterbox_track(self):
        """Canonical ByteTrack pools classes; here that would corrupt every rule."""
        tracker = _tracker()
        tracker.update([_det(0, 0, 0.9, label=LITTERBOX)], now=0.0)
        box_id = tracker.active_tracks()[0].track_id

        # A cat detection in exactly the same place — perfect IoU, wrong label.
        tracker.update([_det(0, 0, 0.9, label=CAT)], now=0.5)

        by_label = {t.label_uuid: t for t in tracker.active_tracks()}
        assert set(by_label) == {CAT, LITTERBOX}
        assert by_label[LITTERBOX].track_id == box_id
        assert by_label[CAT].track_id != box_id

    def test_low_pass_is_class_aware_too(self):
        tracker = _tracker()
        tracker.update([_det(0, 0, 0.9, label=LITTERBOX)], now=0.0)

        tracker.update([_det(0, 0, 0.3, label=CAT)], now=0.5)

        # The weak cat box matches nothing (wrong label) and must not start a track.
        assert [t.label_uuid for t in tracker.all_tracks()] == [LITTERBOX]


class TestPerLabelLifetimes:
    """The headline case: a litter box outlives a cat by 30x, on any camera."""

    def _lifetimes(self):
        return {
            CAT: LabelLifetimeSpec(ttl_s=1.0),
            LITTERBOX: LabelLifetimeSpec(ttl_s=30.0),
        }

    def test_cat_expires_while_litterbox_survives(self):
        tracker = _tracker(label_lifetimes=self._lifetimes())
        tracker.update([_det(0, 0, 0.9, label=CAT), _det(50, 50, 0.9, label=LITTERBOX)], now=0.0)
        assert len(tracker.active_tracks()) == 2

        # 5 s later, with nothing detected at all.
        tracks = tracker.update([], now=5.0)

        assert [t.label_uuid for t in tracks] == [LITTERBOX], "the cat's 1 s TTL should have expired it"

    def test_litterbox_expires_eventually(self):
        tracker = _tracker(label_lifetimes=self._lifetimes())
        tracker.update([_det(50, 50, 0.9, label=LITTERBOX)], now=0.0)

        assert tracker.update([], now=31.0) == []

    def test_ttl_is_seconds_not_frames(self):
        """Same number of empty frames, different elapsed time, different outcome."""
        lifetimes = self._lifetimes()

        fast = _tracker(label_lifetimes=lifetimes)
        fast.update([_det(0, 0, 0.9, label=CAT)], now=0.0)
        for i in range(1, 6):  # 5 frames at 25 fps = 0.2 s
            survivors = fast.update([], now=i * 0.04)
        assert len(survivors) == 1, "0.2 s of misses must not expire a 1 s TTL"

        slow = _tracker(label_lifetimes=lifetimes)
        slow.update([_det(0, 0, 0.9, label=CAT)], now=0.0)
        for i in range(1, 6):  # the same 5 frames at 2 fps = 2.5 s
            survivors = slow.update([], now=i * 0.5)
        assert survivors == [], "2.5 s of misses must expire a 1 s TTL"

    def test_unknown_label_falls_back_to_max_age(self):
        tracker = _tracker(max_age_s=2.0, label_lifetimes=self._lifetimes())
        tracker.update([_det(0, 0, 0.9, label="label-unconfigured")], now=0.0)

        assert tracker.update([], now=1.0) != []
        assert tracker.update([], now=3.0) == []


class TestAnchors:
    def test_anchor_latches_its_box_against_a_weak_redetection(self):
        """A stationary object's box must not jitter, or containment rules flicker."""
        lifetimes = {LITTERBOX: LabelLifetimeSpec(ttl_s=30.0, is_anchor=True)}
        tracker = _tracker(label_lifetimes=lifetimes, anchor_confidence=0.6)
        tracker.update([_det(0, 0, 0.9, label=LITTERBOX)], now=0.0)
        latched = tracker.active_tracks()[0].bbox

        # Re-detected weakly and slightly offset — the latched box must win.
        tracker.update([_det(3, 3, 0.3, label=LITTERBOX)], now=0.5)

        assert tracker.active_tracks()[0].bbox == latched
        assert tracker.active_tracks()[0].is_anchor

    def test_confident_redetection_still_moves_an_anchor(self):
        lifetimes = {LITTERBOX: LabelLifetimeSpec(ttl_s=30.0, is_anchor=True)}
        tracker = _tracker(label_lifetimes=lifetimes, anchor_confidence=0.6)
        tracker.update([_det(0, 0, 0.9, label=LITTERBOX)], now=0.0)

        tracker.update([_det(3, 3, 0.95, label=LITTERBOX)], now=0.5)

        assert tracker.active_tracks()[0].bbox == (3, 3, 13, 13)


class TestConfirmation:
    def test_min_hits_gates_reporting(self):
        tracker = _tracker(min_hits=3)

        assert tracker.update([_det(0, 0, 0.9)], now=0.0) == []
        assert tracker.update([_det(0, 0, 0.9)], now=0.2) == []
        assert len(tracker.update([_det(0, 0, 0.9)], now=0.4)) == 1

    def test_blips_separated_by_misses_never_confirm(self):
        """`min_hits` counts consecutive sightings, not a running total over the TTL."""
        tracker = _tracker(min_hits=3, max_age_s=6.0)
        for i in range(6):
            assert tracker.update([_det(0, 0, 0.9)], now=i * 2.0) == []
            assert tracker.update([], now=i * 2.0 + 1.0) == []

    def test_low_confidence_hits_count_toward_confirmation(self):
        tracker = _tracker(min_hits=2)
        tracker.update([_det(0, 0, 0.9)], now=0.0)

        assert len(tracker.update([_det(0, 0, 0.3)], now=0.2)) == 1


class TestRegistry:
    def test_registers_itself_from_a_bare_registry_import(self):
        """Production only ever imports the Registry — see test_plugin_registry_autodiscovery."""
        from apps.helpers.tracking.Registry import create_tracker, tracker_registry

        assert "ByteTracker" in tracker_registry
        assert isinstance(create_tracker("ByteTracker"), ByteTracker)

    def test_params_schema_is_renderable(self):
        """The Watchers pane builds its form straight from this, with no UI change."""
        schema = ByteTracker.params_schema()

        assert {"track_thresh", "low_thresh", "match_thresh", "min_hits", "max_age_s"} <= set(schema)
        for name, field in schema.items():
            assert field["type"] in ("float", "int", "bool", "str"), name
            assert field.get("help"), f"{name} needs help text — it is user-facing"

    def test_reset_forgets_everything(self):
        tracker = _tracker()
        tracker.update([_det(0, 0, 0.9)], now=0.0)

        tracker.reset()

        assert tracker.all_tracks() == []
        # Ids restart: they are per-session and must not pretend to survive.
        tracker.update([_det(0, 0, 0.9)], now=1.0)
        assert tracker.active_tracks()[0].track_id == 1


class TestMatchedThisUpdate:
    """Detector-miss collection (COLLECTOR_UNIFICATION_PLAN.md §4A) asks each
    track "did the detector explain you on THIS call, or are you coasting?".
    `IouTracker` answers it; this tracker used to leave the field at its `True`
    default forever, which silently turned the whole feature off for the tracker
    most likely to be chosen for it — no gap ever opened, and nothing said why.
    """

    def test_a_coasting_track_is_reported_unmatched(self):
        tracker = _tracker()
        tracker.update([_det(0, 0, 0.9)], now=0.0)

        (track,) = tracker.update([], now=0.5)

        assert track.matched_this_update is False

    def test_a_weak_second_stage_match_still_counts_as_matched(self):
        """Stage 2 is this tracker's whole point: a low-confidence box IS the
        detector explaining the track, so it must not read as a miss."""
        tracker = _tracker()
        tracker.update([_det(0, 0, 0.9)], now=0.0)

        (track,) = tracker.update([_det(1, 1, 0.3)], now=0.5)

        assert track.matched_this_update is True

    def test_re_acquisition_flips_it_back(self):
        tracker = _tracker()
        tracker.update([_det(0, 0, 0.9)], now=0.0)
        assert tracker.update([], now=0.5)[0].matched_this_update is False

        (track,) = tracker.update([_det(1, 1, 0.9)], now=1.0)

        assert track.matched_this_update is True

    def test_only_the_missed_track_is_marked(self):
        """Two tracks, one detection — the flag is per track, not per call."""
        tracker = _tracker()
        tracker.update([_det(0, 0, 0.9), _det(100, 100, 0.9)], now=0.0)

        tracks = {t.track_id: t for t in tracker.update([_det(0, 0, 0.9)], now=0.5)}

        assert tracks[1].matched_this_update is True
        assert tracks[2].matched_this_update is False
