"""The detection sidecar — what the watcher saw, kept beside the clip it saw it in.

The bugs worth catching here are all arithmetic, and all silent. A box normalised
against the wrong frame size, an offset taken from the *requested* clip start instead of
the actual one, a sample held across a gap it should not span — none of them raise, and
every one of them renders as boxes that sit slightly wrong over the video, which reads
as a detector fault rather than a player fault. That is the expensive kind of wrong: the
overlay exists to diagnose detection, so an overlay that lies about detection is worse
than no overlay at all.

Pure unit tests. The ring holds numbers, the builder does date arithmetic, and neither
needs a camera.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from apps.helpers.recordings.detectionTrack import (
    OVERLAY_SUFFIX,
    OVERLAY_VERSION,
    DetectionTrack,
    build_overlay,
    event_window,
    overlay_path_for,
    write_overlay,
)

T0 = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)


class _Det:
    """Stands in for `apps.helpers.tracking.Detection` — the real one is a dataclass and
    the module reads it by attribute, so a namespace is a faithful double."""

    def __init__(self, bbox, label_name="cat", label_uuid="lbl-1", confidence=0.9):
        self.bbox = bbox
        self.label_name = label_name
        self.label_uuid = label_uuid
        self.confidence = confidence


class _Trk(_Det):
    def __init__(self, track_id, bbox, **kw):
        super().__init__(bbox, **kw)
        self.track_id = track_id


class TestNormalising:
    def test_a_box_is_stored_as_a_fraction_of_the_analysis_frame(self):
        track = DetectionTrack()
        track.push(100.0, [_Det((160, 90, 320, 180))], [], (640, 360))

        box = track.samples_between(100.0, 100.0)[0].boxes[0]

        assert box["x"] == pytest.approx(0.25)
        assert box["y"] == pytest.approx(0.25)
        assert box["w"] == pytest.approx(0.25)
        assert box["h"] == pytest.approx(0.25)

    def test_a_box_running_off_the_edge_is_clamped_not_dropped(self):
        """A subject walking out of shot is routine, and its partial box is the
        interesting one."""
        track = DetectionTrack()
        track.push(100.0, [_Det((-50, 10, 700, 200))], [], (640, 360))

        box = track.samples_between(100.0, 100.0)[0].boxes[0]

        assert box["x"] == 0.0
        assert box["x"] + box["w"] == pytest.approx(1.0)

    def test_a_box_entirely_outside_the_frame_is_dropped(self):
        track = DetectionTrack()
        track.push(100.0, [_Det((700, 10, 900, 200))], [], (640, 360))

        assert track.samples_between(100.0, 100.0)[0].boxes == []

    def test_corners_in_either_order_normalise_the_same(self):
        track = DetectionTrack()
        track.push(100.0, [_Det((320, 180, 160, 90))], [], (640, 360))

        box = track.samples_between(100.0, 100.0)[0].boxes[0]

        assert box["x"] == pytest.approx(0.25)
        assert box["w"] == pytest.approx(0.25)

    def test_a_zero_sized_frame_yields_no_boxes_rather_than_dividing_by_zero(self):
        track = DetectionTrack()
        track.push(100.0, [_Det((10, 10, 20, 20))], [], (0, 0))

        assert track.samples_between(100.0, 100.0)[0].boxes == []

    def test_a_track_keeps_its_id(self):
        track = DetectionTrack()
        track.push(100.0, [], [_Trk(7, (0, 0, 320, 180))], (640, 360))

        assert track.samples_between(100.0, 100.0)[0].tracks[0]["id"] == 7

    def test_detections_and_tracks_are_kept_apart(self):
        """A box the model saw and the tracker dropped is exactly the diagnosis this
        exists for, so collapsing the two would destroy the point."""
        track = DetectionTrack()
        track.push(100.0, [_Det((0, 0, 100, 100))], [], (640, 360))

        sample = track.samples_between(100.0, 100.0)[0]

        assert len(sample.boxes) == 1
        assert sample.tracks == []


class TestTheRing:
    def test_samples_older_than_the_window_are_evicted(self):
        track = DetectionTrack(max_seconds=10.0)
        for t in range(0, 30):
            track.push(float(t), [], [], (640, 360))

        assert track.samples_between(0.0, 100.0)[0].at >= 19.0

    def test_the_sample_count_is_bounded_independently_of_time(self):
        """A 30 fps source must not hold thirty times what a 1 fps source holds for the
        same window."""
        track = DetectionTrack(max_seconds=10_000.0, max_samples=5)
        for t in range(0, 50):
            track.push(float(t), [], [], (640, 360))

        assert track.sample_count <= 6

    def test_an_empty_frame_is_still_recorded(self):
        """ "The detector saw nothing here" is a fact the overlay needs — without it a
        detection that dropped out renders as a box that simply persisted."""
        track = DetectionTrack()
        track.push(100.0, [], [], (640, 360))

        assert track.sample_count == 1

    def test_samples_between_is_inclusive_at_both_ends(self):
        track = DetectionTrack()
        for t in (10.0, 20.0, 30.0):
            track.push(t, [], [], (640, 360))

        assert [s.at for s in track.samples_between(10.0, 30.0)] == [10.0, 20.0, 30.0]


class TestBuildingTheOverlay:
    def _track(self):
        track = DetectionTrack()
        base = T0.timestamp()
        for offset in range(0, 20):
            track.push(base + offset, [_Det((160, 90, 320, 180))], [], (640, 360))
        return track

    def test_times_are_relative_to_the_clips_own_start(self):
        track = self._track()
        clip_start = T0 + timedelta(seconds=5)
        samples = track.samples_between(clip_start.timestamp(), clip_start.timestamp() + 10)

        doc = build_overlay(samples, clip_start, 10.0)

        assert doc["frames"][0]["t"] == 0.0
        assert doc["frames"][-1]["t"] == 10.0

    def test_samples_outside_the_window_are_dropped_not_clamped(self):
        """Pinning an out-of-window sample to t=0 would draw a box at a moment the
        detector had not yet seen it — a fabricated false positive in the very tool used
        to judge false positives."""
        track = self._track()
        clip_start = T0 + timedelta(seconds=5)
        # Deliberately hand it everything, including samples before the clip starts.
        doc = build_overlay(track.samples_between(0, 1e12), clip_start, 5.0)

        assert all(0.0 <= f["t"] <= 5.0 for f in doc["frames"])

    def test_the_document_declares_its_version(self):
        doc = build_overlay([], T0, 10.0)

        assert doc["version"] == OVERLAY_VERSION

    def test_the_staleness_bound_travels_with_the_document(self):
        """The player cannot know the watcher's analysis cadence; only this side does."""
        doc = build_overlay([], T0, 10.0, max_staleness_s=4.0)

        assert doc["maxStalenessS"] == 4.0


class TestTheEventWindow:
    def test_the_rules_span_is_reported_relative_to_the_clip(self):
        clip_start = T0
        window = event_window(clip_start, 30.0, T0 + timedelta(seconds=10), T0 + timedelta(seconds=20), "Cat at door")

        assert window["startS"] == 10.0
        assert window["endS"] == 20.0
        assert window["ruleName"] == "Cat at door"

    def test_an_event_overrunning_the_clip_is_clamped_rather_than_dropped(self):
        """A clip capped by `max_length_s` legitimately ends before its event does, and
        those are the longest — most interesting — events."""
        window = event_window(T0, 30.0, T0 + timedelta(seconds=10), T0 + timedelta(seconds=300))

        assert window["endS"] == 30.0

    def test_an_event_starting_after_the_clip_ends_has_no_window(self):
        assert event_window(T0, 30.0, T0 + timedelta(seconds=100), None) is None

    def test_an_open_ended_event_runs_to_the_end_of_the_clip(self):
        window = event_window(T0, 30.0, T0 + timedelta(seconds=10), None)

        assert window["endS"] == 30.0

    def test_no_start_means_no_window(self):
        assert event_window(T0, 30.0, None, None) is None


class TestWriting:
    def test_the_sidecar_sits_beside_the_clip_and_names_it(self, tmp_path):
        assert overlay_path_for(tmp_path / "clip.mp4").name == f"clip.mp4{OVERLAY_SUFFIX}"

    def test_a_written_sidecar_parses_back(self, tmp_path):
        dest = overlay_path_for(tmp_path / "clip.mp4")

        write_overlay(build_overlay([], T0, 10.0), dest)

        assert json.loads(dest.read_text())["version"] == OVERLAY_VERSION

    def test_no_partial_file_is_left_behind(self, tmp_path):
        """The player fetches this the moment the card opens; half a JSON document reads
        as a broken feature rather than as the transient it is."""
        dest = overlay_path_for(tmp_path / "clip.mp4")

        write_overlay(build_overlay([], T0, 10.0), dest)

        assert list(tmp_path.glob("*.part")) == []
