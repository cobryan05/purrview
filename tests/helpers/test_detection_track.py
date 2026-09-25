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
    build_timelapse_overlay,
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


class TestTheIngestLagCompensation:
    """The boxes and the picture are measured on two different clocks.

    A buffer-cut clip's timeline is the relay's record clock; a sample's `at` is the
    watcher's fetch clock, one ffmpeg pipe and jitter buffer later. Uncompensated, every
    box renders that much late — measured at +0.4 s to +1.6 s on a real install.
    """

    def _track(self):
        track = DetectionTrack()
        base = T0.timestamp()
        for offset in range(0, 20):
            track.push(base + offset, [_Det((160, 90, 320, 180))], [], (640, 360))
        return track

    def test_the_lag_moves_every_box_earlier_in_the_clip(self):
        track = self._track()
        clip_start = T0 + timedelta(seconds=5)
        samples = track.samples_between(clip_start.timestamp(), clip_start.timestamp() + 10)

        plain = build_overlay(samples, clip_start, 10.0)
        shifted = build_overlay(samples, clip_start, 10.0, lag_s=1.0)

        # Same samples, each one second earlier in the clip. The first now falls before
        # t=0 and is dropped rather than pinned there, exactly as an out-of-window
        # sample is.
        assert [f["t"] for f in shifted["frames"]] == [t - 1.0 for t in [f["t"] for f in plain["frames"]][1:]]

    def test_no_lag_is_byte_for_byte_the_old_behaviour(self):
        """0.0 is the default and must be a true no-op — an install that has not
        calibrated must not have its overlays quietly moved."""
        track = self._track()
        clip_start = T0 + timedelta(seconds=5)
        samples = track.samples_between(clip_start.timestamp(), clip_start.timestamp() + 10)

        assert build_overlay(samples, clip_start, 10.0, lag_s=0.0) == build_overlay(samples, clip_start, 10.0)

    def test_the_applied_lag_is_recorded_in_the_document(self):
        """A sidecar written under one calibration and read after it changed is
        otherwise indistinguishable from one written under the new value."""
        assert build_overlay([], T0, 10.0, lag_s=1.5)["lagS"] == 1.5

    def test_the_event_marker_shifts_with_the_boxes(self):
        """The rule fired on the watcher's copy of the frame, so its span is on the
        watcher's clock too. Shifting only the boxes would make the red frame and the
        boxes disagree — worse than both being late together."""
        window = event_window(T0, 30.0, T0 + timedelta(seconds=10), T0 + timedelta(seconds=20), lag_s=1.0)

        assert window["startS"] == 9.0
        assert window["endS"] == 19.0

    def test_a_lag_cannot_push_the_event_marker_before_the_clip_starts(self):
        window = event_window(T0, 30.0, T0 + timedelta(seconds=0.5), T0 + timedelta(seconds=20), lag_s=1.0)

        assert window["startS"] == 0.0


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


class TestTheTimelapseOverlay:
    """A timelapse's sidecar cannot be built the way a clip's is.

    `build_overlay` maps a sample onto the video by subtracting wallclocks, which is
    only valid while one second of video is one second of world. A timelapse plays two
    hours in eighteen seconds, so that subtraction puts every box thousands of seconds
    past the end of the video and the whole sidecar drops out as out-of-window — a
    silent failure that looks exactly like "this timelapse has no detections".
    """

    def test_each_kept_frame_lands_on_its_own_frame_of_video(self):
        doc = build_timelapse_overlay([[{"x": 0.1}], [{"x": 0.2}], [{"x": 0.3}]], T0, output_fps=12.0)

        assert [f["t"] for f in doc["frames"]] == [
            0.0,
            pytest.approx(0.0833, abs=1e-4),
            pytest.approx(0.1667, abs=1e-4),
        ]
        assert doc["durationS"] == pytest.approx(0.25, abs=1e-4)

    def test_the_wallclock_span_of_the_event_does_not_leak_into_the_timings(self):
        """The regression that motivates the separate builder: frames sampled ten
        seconds apart in the world are one frame apart in the video."""
        # Two hours of event at one frame per 10s is 720 frames; at 12 fps that is a
        # 60-second video, not a 7200-second one.
        doc = build_timelapse_overlay([[] for _ in range(720)], T0, output_fps=12.0)

        assert doc["durationS"] == pytest.approx(60.0)
        assert doc["frames"][-1]["t"] == pytest.approx(59.9167, abs=1e-3)

    def test_the_hold_is_exactly_one_frame(self):
        """The player holds a sample until the next one. Here the next one is the very
        next frame of video, so a generous hold would smear a departed cat's box across
        the frames after it left."""
        doc = build_timelapse_overlay([[], []], T0, output_fps=10.0)

        assert doc["maxStalenessS"] == pytest.approx(0.1)

    def test_no_event_window_is_marked(self):
        """Every frame of a timelapse is inside its event, so the red frame would cover
        the whole video — an annotation that marks everything marks nothing."""
        doc = build_timelapse_overlay([[{"x": 0.1}]], T0, output_fps=12.0)

        assert "event" not in doc

    def test_the_player_recognises_the_version_and_shape(self):
        """Same document the clip player already reads — a second shape would mean a
        second renderer."""
        doc = build_timelapse_overlay([[]], T0, output_fps=12.0)

        assert doc["version"] == OVERLAY_VERSION
        assert doc["startedAt"] == T0.isoformat()
        assert all({"t", "boxes", "tracks"} <= set(f) for f in doc["frames"])


class TestTheRawObservationBesideTheVerdict:
    """A track's reported label is a vote across frames; the sidecar must still say what
    the detector claimed on each one.

    Losing that would cost the most valuable training data a clip contains — the frames
    where the model was wrong about an object whose identity the rest of the clip
    establishes. `boxes` records raw model output and `tracks` the smoothed verdict, and
    neither on its own can express "these two disagreed here".
    """

    def _push(self, track):
        ring = DetectionTrack()
        ring.push(at=1000.0, detections=[], tracks=[track], frame_size=(100, 100))
        return ring.samples_between(999.0, 1001.0)[0].tracks[0]

    def test_a_disagreeing_frame_records_both_labels(self):
        track = _Trk(1, (0, 0, 50, 50), label_uuid="lbl-person", label_name="Person")
        track.observed_label_uuid = "lbl-alice"
        track.observed_confidence = 0.88
        track.observed_margin = 0.05

        box = self._push(track)

        assert box["labelUuid"] == "lbl-person"
        assert box["sawLabelUuid"] == "lbl-alice"
        assert box["sawConf"] == pytest.approx(0.88)
        assert box["margin"] == pytest.approx(0.05)

    def test_a_coasting_frame_records_no_observation(self):
        """Absent, not equal to the verdict: "the detector said nothing" and "the
        detector agreed" are different facts, and the blip analysis splits on it."""
        track = _Trk(1, (0, 0, 50, 50), label_uuid="lbl-person", label_name="Person")
        track.observed_label_uuid = None
        track.observed_margin = None

        box = self._push(track)

        assert "sawLabelUuid" not in box
        assert "margin" not in box

    def test_a_model_with_no_runner_ups_still_records_the_observation(self):
        """An NMS-free export cannot produce a margin. The label it saw is still a fact."""
        track = _Trk(1, (0, 0, 50, 50), label_uuid="lbl-person")
        track.observed_label_uuid = "lbl-person"
        track.observed_confidence = 0.9
        track.observed_margin = None

        box = self._push(track)

        assert box["sawLabelUuid"] == "lbl-person"
        assert "margin" not in box
