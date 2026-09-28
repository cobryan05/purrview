"""Tests for the presence log behind the events' "Present" catalogue.

The properties worth protecting are the ones whose failure is a plausible-looking wrong
answer rather than an error: a window that reports nothing for a subject that was visible
throughout it, a flicker recorded as two visits, or a catalogue built from raw detections
rather than confirmed tracks — the last of which would mark nearly every clip as
containing nearly every label, and would look like the detector being generous.
"""

from apps.helpers.recordings.presenceLog import (
    PresenceLog,
    labels_from_overlay,
    merge_label_confidences,
)

CAM = "cam-1"
CAT = "label-cat"
DOG = "label-dog"


class TestPresenceWindows:
    def test_a_label_still_visible_when_the_window_opens_is_present_in_it(self):
        """Overlap, not containment. A subject that was already there when the clip's
        window began is in the video; requiring the interval to sit inside the window
        would report nothing for exactly the long visits a clip is cut from."""
        log = PresenceLog()
        for at in range(100, 141, 5):
            log.note(CAM, float(at), {CAT: 0.8})

        assert set(log.labels_between(CAM, 120.0, 200.0)) == {CAT}
        assert set(log.labels_between(CAM, 0.0, 110.0)) == {CAT}

    def test_a_window_before_the_sighting_reports_nothing(self):
        log = PresenceLog()
        log.note(CAM, 100.0, {CAT: 0.8})
        assert log.labels_between(CAM, 0.0, 50.0) == {}

    def test_presence_is_per_source(self):
        log = PresenceLog()
        log.note(CAM, 100.0, {CAT: 0.8})
        log.note("cam-2", 100.0, {DOG: 0.8})
        assert set(log.labels_between(CAM, 90.0, 110.0)) == {CAT}
        assert set(log.labels_between("cam-2", 90.0, 110.0)) == {DOG}

    def test_the_peak_confidence_of_a_visit_is_reported_not_the_last(self):
        log = PresenceLog()
        log.note(CAM, 100.0, {CAT: 0.4})
        log.note(CAM, 101.0, {CAT: 0.9})
        log.note(CAM, 102.0, {CAT: 0.5})
        assert log.labels_between(CAM, 99.0, 103.0)[CAT] == 0.9


class TestIntervalMerging:
    def test_a_one_frame_dropout_stays_one_visit(self):
        """The gap rule exists for this: a detector that misses a frame must not turn one
        visit into two, because every interval is overlap arithmetic later."""
        log = PresenceLog(gap_s=15.0)
        log.note(CAM, 100.0, {CAT: 0.8})
        log.note(CAM, 102.0, {CAT: 0.8})  # one frame missing between these
        assert log.interval_count(CAM) == 1

    def test_two_sightings_further_apart_than_the_gap_are_two_visits(self):
        log = PresenceLog(gap_s=15.0)
        log.note(CAM, 100.0, {CAT: 0.8})
        log.note(CAM, 200.0, {CAT: 0.8})
        assert log.interval_count(CAM) == 2

    def test_a_second_visit_does_not_extend_the_first_over_the_quiet_gap(self):
        """The failure this catches: merging regardless of the gap would report the cat as
        present during the hundred seconds it was gone, so a clip cut from that quiet
        stretch would be catalogued as containing it."""
        log = PresenceLog(gap_s=15.0)
        log.note(CAM, 100.0, {CAT: 0.8})
        log.note(CAM, 200.0, {CAT: 0.8})
        assert log.labels_between(CAM, 130.0, 170.0) == {}

    def test_two_labels_on_one_frame_are_tracked_independently(self):
        log = PresenceLog(gap_s=15.0)
        log.note(CAM, 100.0, {CAT: 0.8, DOG: 0.3})
        log.note(CAM, 105.0, {CAT: 0.8})
        log.note(CAM, 200.0, {DOG: 0.9})
        assert set(log.labels_between(CAM, 100.0, 106.0)) == {CAT, DOG}
        assert set(log.labels_between(CAM, 195.0, 205.0)) == {DOG}

    def test_an_empty_frame_records_nothing_and_does_not_close_an_interval_early(self):
        log = PresenceLog(gap_s=15.0)
        log.note(CAM, 100.0, {CAT: 0.8})
        log.note(CAM, 101.0, {})
        log.note(CAM, 102.0, {CAT: 0.8})
        assert log.interval_count(CAM) == 1


class TestPruning:
    def test_intervals_past_the_horizon_are_dropped(self):
        log = PresenceLog(horizon_s=100.0)
        log.note(CAM, 100.0, {CAT: 0.8})
        log.note(CAM, 500.0, {DOG: 0.8})
        assert set(log.labels_between(CAM, 90.0, 110.0)) == set()
        assert set(log.labels_between(CAM, 490.0, 510.0)) == {DOG}

    def test_an_interval_still_open_at_the_horizon_survives(self):
        """`end` ages, not `start`. A subject present for longer than the horizon must not
        vanish from a window it is still inside."""
        log = PresenceLog(horizon_s=100.0, gap_s=60.0)
        for at in range(100, 501, 50):
            log.note(CAM, float(at), {CAT: 0.8})
        assert set(log.labels_between(CAM, 480.0, 500.0)) == {CAT}

    def test_the_interval_count_is_capped(self):
        log = PresenceLog(gap_s=0.0, max_intervals=5)
        for at in range(100, 200):
            log.note(CAM, float(at), {CAT: 0.8})
        assert log.interval_count(CAM) <= 5

    def test_a_long_lived_interval_does_not_shield_stale_ones_behind_it(self):
        """Intervals are ordered by `start`, but `end` is what ages. A subject that stays
        visible sits at the head forever, so no property of the head interval can stand in
        for the sweep: an expired sighting behind it must still be dropped."""
        log = PresenceLog(horizon_s=100.0, gap_s=60.0)
        log.note(CAM, 0.0, {CAT: 0.8})
        log.note(CAM, 5.0, {DOG: 0.8})
        for at in range(50, 2001, 50):
            log.note(CAM, float(at), {CAT: 0.8})
        assert set(log.labels_between(CAM, 0.0, 10.0)) == {CAT}
        assert set(log.labels_between(CAM, 1980.0, 2000.0)) == {CAT}

    def test_the_count_cap_evicts_the_stale_intervals_not_the_open_one(self):
        """A still-open interval sits at the head for as long as it lasts, so a cap that
        trimmed by position would discard exactly the label most reliably in the clip and
        keep the per-frame flicker that crowded it out."""
        log = PresenceLog(gap_s=60.0, max_intervals=5)
        for at in range(0, 21):
            log.note(CAM, float(at), {CAT: 0.8, f"flicker-{at}": 0.5})
        assert log.interval_count(CAM) <= 5
        assert CAT in log.labels_between(CAM, 0.0, 2.0)


class TestOverlayLabels:
    def test_only_confirmed_tracks_are_catalogued_never_raw_boxes(self):
        """The whole reason the catalogue is usable. The watcher asks inference for a 0.1
        floor because ByteTrack needs weak boxes, so a catalogue built from `boxes` would
        contain every label the detector ever guessed at."""
        document = {
            "version": 2,
            "frames": [
                {
                    "t": 0.0,
                    "boxes": [{"labelUuid": DOG, "conf": 0.12}],
                    "tracks": [{"id": 1, "labelUuid": CAT, "conf": 0.81}],
                }
            ],
        }
        assert labels_from_overlay(document) == {CAT: 0.81}

    def test_the_peak_across_frames_wins(self):
        document = {
            "frames": [
                {"tracks": [{"id": 1, "labelUuid": CAT, "conf": 0.4}]},
                {"tracks": [{"id": 1, "labelUuid": CAT, "conf": 0.93}]},
                {"tracks": [{"id": 1, "labelUuid": CAT, "conf": 0.5}]},
            ]
        }
        assert labels_from_overlay(document) == {CAT: 0.93}

    def test_a_track_with_no_label_is_skipped(self):
        document = {"frames": [{"tracks": [{"id": 1, "labelUuid": "", "conf": 0.9}]}]}
        assert labels_from_overlay(document) == {}

    def test_a_document_with_no_frames_yields_nothing(self):
        assert labels_from_overlay({}) == {}
        assert labels_from_overlay({"frames": []}) == {}


class TestMerge:
    def test_the_higher_confidence_survives_a_union(self):
        merged = merge_label_confidences({CAT: 0.4, DOG: 0.9}.items(), {CAT: 0.7}.items())
        assert merged == {CAT: 0.7, DOG: 0.9}
