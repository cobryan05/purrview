"""Presence reporting picks its signal from how the source is actually behaving.

The rule under test: a source producing video gets its presence from confirmed tracks,
a source producing stills gets it from recent detections. Both failure directions are
silent and neither throws, which is what makes them worth a test:

* tracks on a snapshot source report *nothing* — at one frame per 30-90 s no track can
  reach `min_hits`, and it would expire between frames anyway;
* detections on a streaming source report *everything*, including the single-frame
  noise the tracker exists to absorb.

A Ring camera is both at different moments — full-rate RTSP while motion is escalating
it, ~90 s cloud snapshots when nothing is moving — so this is derived from the measured
frame gap rather than configured per camera.
"""

from types import SimpleNamespace

from apps.helpers.tracking.tracker import Detection, Track
from apps.tasks_server.tasks._watch_pipeline import (
    SNAPSHOT_MODE_FRAME_GAP_S,
    WatchPipeline,
)

CAT = "label-cat"
DOG = "label-dog"


def _pipeline(**overrides) -> WatchPipeline:
    """A pipeline carrying only what presence reporting reads."""
    pipeline = WatchPipeline.__new__(WatchPipeline)
    pipeline.tracker = SimpleNamespace(track_thresh=0.5)
    pipeline._last_tracks = []
    pipeline._last_detections = []
    pipeline._last_analysis_at = 1000.0
    pipeline._last_analysis_gap_s = 0.2
    for key, value in overrides.items():
        setattr(pipeline, key, value)
    return pipeline


def _track(label: str, *, confirmed: bool = True) -> Track:
    return Track(track_id=1, bbox=(0.0, 0.0, 1.0, 1.0), label_uuid=label, confirmed=confirmed)


def _detection(label: str, confidence: float) -> Detection:
    return Detection(bbox=(0.0, 0.0, 1.0, 1.0), label_uuid=label, confidence=confidence)


class TestPresenceModeFollowsTheSource:
    def test_a_streaming_source_uses_tracks(self):
        pipeline = _pipeline(_last_analysis_gap_s=0.2)
        assert pipeline.presence_mode() == "tracks"

    def test_a_snapshot_source_uses_detections(self):
        pipeline = _pipeline(_last_analysis_gap_s=90.0)
        assert pipeline.presence_mode() == "snapshots"

    def test_one_source_moves_between_modes_as_its_cadence_changes(self):
        """The reason this is derived rather than configured: a Ring camera escalates
        to full frame rate on motion and falls back to snapshots when it clears, with
        no change to its ingest tier."""
        pipeline = _pipeline(_last_analysis_gap_s=SNAPSHOT_MODE_FRAME_GAP_S * 30)
        assert pipeline.presence_mode() == "snapshots"
        pipeline._last_analysis_gap_s = 0.2  # motion escalated it to live RTSP
        assert pipeline.presence_mode() == "tracks"


class TestStreamingPresenceUsesTracks:
    def test_an_unconfirmed_track_is_not_presence(self):
        """`min_hits` is the tracker's anti-flicker gate; presence must not undercut
        it by reporting a track the tracker itself would not report."""
        pipeline = _pipeline(_last_tracks=[_track(CAT, confirmed=False)])
        assert pipeline.presence_labels(now=1000.0) == set()

    def test_a_confirmed_track_is_presence(self):
        pipeline = _pipeline(_last_tracks=[_track(CAT)])
        assert pipeline.presence_labels(now=1000.0) == {CAT}

    def test_detections_alone_are_not_presence_while_streaming(self):
        """The single-frame detection the tracker declined to promote is exactly the
        flicker this mode exists to absorb — a union of both signals would let it
        through."""
        pipeline = _pipeline(_last_tracks=[], _last_detections=[_detection(CAT, 0.9)])
        assert pipeline.presence_labels(now=1000.0) == set()


class TestSnapshotPresenceUsesDetections:
    def test_a_recent_detection_is_presence(self):
        """The case tracks cannot serve: a cat that walked in during a motion
        escalation, settled down, and is now only visible in still frames."""
        pipeline = _pipeline(
            _last_analysis_gap_s=90.0,
            _last_detections=[_detection(CAT, 0.9)],
            _last_tracks=[],
        )
        assert pipeline.presence_labels(now=1000.0) == {CAT}

    def test_a_weak_detection_is_not_presence(self):
        pipeline = _pipeline(_last_analysis_gap_s=90.0, _last_detections=[_detection(CAT, 0.2)])
        assert pipeline.presence_labels(now=1000.0) == set()

    def test_the_floor_is_the_trackers_own_threshold_not_a_second_constant(self):
        """Whatever the tracker is configured to start a track at is what the snapshot
        path admits — the two must not drift apart."""
        pipeline = _pipeline(_last_analysis_gap_s=90.0, _last_detections=[_detection(CAT, 0.6)])
        pipeline.tracker = SimpleNamespace(track_thresh=0.9)
        assert pipeline.presence_labels(now=1000.0) == set()
        pipeline.tracker = SimpleNamespace(track_thresh=0.5)
        assert pipeline.presence_labels(now=1000.0) == {CAT}

    def test_a_stale_snapshot_stops_asserting_presence(self):
        """A still is a point sample. Once the next one is overdue the honest answer is
        no longer 'still there'."""
        pipeline = _pipeline(_last_analysis_gap_s=30.0, _last_detections=[_detection(CAT, 0.9)])
        # Within the cadence slack the last verdict stands…
        assert pipeline.presence_labels(now=1000.0 + 30.0) == {CAT}
        # …well past it, it does not.
        assert pipeline.presence_labels(now=1000.0 + 300.0) == set()

    def test_only_the_labels_actually_seen_are_reported(self):
        pipeline = _pipeline(
            _last_analysis_gap_s=90.0,
            _last_detections=[_detection(CAT, 0.9), _detection(DOG, 0.1)],
        )
        assert pipeline.presence_labels(now=1000.0) == {CAT}


class TestBeforeAnyFrame:
    def test_a_source_that_has_never_produced_a_frame_reports_nothing(self):
        pipeline = _pipeline(_last_analysis_at=None)
        assert pipeline.presence_labels(now=1000.0) == set()
