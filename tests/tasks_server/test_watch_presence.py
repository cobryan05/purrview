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
    PRESENCE_CADENCE_BUCKET_S,
    PRESENCE_CADENCE_WINDOW_S,
    SNAPSHOT_MODE_FRAME_GAP_S,
    WatchPipeline,
)

CAT = "label-cat"
DOG = "label-dog"


def _pipeline(**overrides) -> WatchPipeline:
    """A pipeline carrying only what presence reporting reads."""
    pipeline = WatchPipeline.__new__(WatchPipeline)
    # max_age_s as well as the floor: the handover carry is bounded by it, and a stub
    # without one would exercise a fallback rather than the real configuration.
    pipeline.tracker = SimpleNamespace(track_thresh=0.5, max_age_s=6.0)
    pipeline._last_tracks = []
    pipeline._last_detections = []
    pipeline._last_analysis_at = 1000.0
    pipeline._last_analysis_gap_s = 0.2
    pipeline._cadence_buckets = []
    pipeline._presence_held_until = {}
    pipeline._presence_last_mode = None
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


class TestPresenceStalenessSurvivesAChangeOfPace:
    """A camera that speeds up must not be called stale when it slows back down.

    The hold was scaled off the *latest* frame gap. A Ring camera escalating to live
    video drove that to a fraction of a second, so the first cloud snapshot after the
    burst — thirty to ninety seconds later, exactly on time — arrived long after the
    verdict had expired. The per-camera presence entities went unavailable and back
    again on every motion event, which is what a user sees as constant flapping.
    """

    @staticmethod
    def _feed(pipeline, gaps, start=1000.0):
        """Replay a series of frame gaps through the cadence estimator."""
        now = start
        for gap in gaps:
            now += gap
            pipeline._note_analysis_gap(gap, now)
            pipeline._last_analysis_gap_s = gap
            pipeline._last_analysis_at = now
        return now

    def test_a_burst_of_live_frames_does_not_shorten_the_hold(self):
        pipeline = _pipeline()
        # Two snapshots 90s apart, then a minute of 20fps live video.
        now = self._feed(pipeline, [90.0, 90.0])
        now = self._feed(pipeline, [0.05] * 1200, start=now)

        # The next snapshot is 90s out — on time for this camera, and the moment the
        # old rule had already declared the verdict stale.
        assert pipeline.presence_is_known(now + 90.0) is True

    def test_a_camera_that_actually_stops_still_goes_unknown(self):
        """The other half. A hold that never expired would pin a parked car in the
        garage forever off a feed that died — which is the failure the staleness rule
        exists for, and it must survive the fix for the flap."""
        pipeline = _pipeline()
        now = self._feed(pipeline, [90.0, 90.0])

        # Well past two cadences with nothing arriving.
        assert pipeline.presence_is_known(now + 400.0) is False

    def test_the_estimate_forgets_a_gap_that_has_aged_out(self):
        """Bounded, or one stalled morning would keep a camera's hold enormous for the
        rest of the run."""
        pipeline = _pipeline()
        # From the frame's own arrival, not the start of the replay: the bucket is
        # stamped when the gap *ended*.
        now = self._feed(pipeline, [300.0])
        far_later = now + PRESENCE_CADENCE_WINDOW_S + PRESENCE_CADENCE_BUCKET_S * 2
        pipeline._note_analysis_gap(1.0, far_later)

        assert pipeline._observed_cadence_s() == 1.0


class TestTheHandoverBetweenModes:
    """The gap the either/or leaves: neither mode's guarantee spans the switch.

    Observed on a Ring camera escalating on motion. The first frame of the escalation
    still carries the idle gap, so it is judged in snapshot mode and one detection turns
    presence on. The next frame arrives sub-second later, so the mode flips to tracks —
    where that same object has one hit of the three it needs, and the set is empty. The
    sensor read Detected / Clear / Detected inside four seconds, with the last two in
    the same second, for a cat that had simply walked in.

    Each mode is right on its own; the switch between them is what has no rule.
    """

    def _escalating(self) -> WatchPipeline:
        """Idle at 90s snapshots, with the cat detected but no track yet confirmed."""
        return _pipeline(
            _last_analysis_gap_s=90.0,
            _cadence_buckets=[(0.0, 90.0)],
            _last_detections=[_detection(CAT, 0.8)],
            _last_tracks=[_track(CAT, confirmed=False)],
            _last_analysis_at=1000.0,
        )

    def test_presence_survives_the_escalation_into_tracks_mode(self):
        pipeline = self._escalating()
        assert pipeline.presence_labels(now=1000.0) == {CAT}, "the snapshot sees the cat"

        # Motion escalates the camera: frames now arrive sub-second apart, so the mode
        # flips — but the track still has one hit of three and reports nothing.
        pipeline._last_analysis_gap_s = 0.2
        pipeline._last_analysis_at = 1004.0
        assert pipeline.presence_mode() == "tracks"

        assert pipeline.presence_labels(now=1004.0) == {CAT}, "must not clear mid-handover"

    def test_the_track_then_confirms_without_a_gap(self):
        """The third reading in the bounce. Once the track confirms, presence continues
        from the tracker with no interruption to paper over."""
        pipeline = self._escalating()
        pipeline.presence_labels(now=1000.0)
        pipeline._last_analysis_gap_s = 0.2
        pipeline.presence_labels(now=1004.0)

        pipeline._last_tracks = [_track(CAT, confirmed=True)]
        assert pipeline.presence_labels(now=1004.1) == {CAT}

    def test_the_hold_expires_so_a_departure_is_still_reported(self):
        """The hold must not become a latch. Once it lapses with nothing seeing the cat,
        the answer is absence — otherwise a subject that left is pinned present."""
        pipeline = self._escalating()
        assert pipeline.presence_labels(now=1000.0) == {CAT}

        pipeline._last_detections = []
        pipeline._last_tracks = []
        pipeline._last_analysis_at = 1400.0
        assert pipeline.presence_labels(now=1400.0) == set()

    def test_a_streaming_source_is_not_given_an_extra_tail(self):
        """A confirmed track that ends has already served `max_age_s` inside the tracker.
        Holding it again here would double the tail on every departure a camera reports
        properly, to fix a handover that never happens on a source that only streams.
        """
        pipeline = _pipeline(
            _last_analysis_gap_s=0.2,
            _cadence_buckets=[(0.0, 0.2)],
            _last_tracks=[_track(CAT)],
            _last_analysis_at=1000.0,
        )
        assert pipeline.presence_labels(now=1000.0) == {CAT}

        pipeline._last_tracks = []
        pipeline._last_analysis_at = 1000.5
        assert pipeline.presence_labels(now=1000.5) == set()

    def test_a_stale_snapshot_is_capped_once_frames_start_arriving(self):
        """The bound that stops the carry becoming a long false positive.

        A snapshot's answer stands for a cadence — up to three minutes on these cameras
        — which is right when it is the only evidence, and wrong the moment twenty
        frames a second are contradicting it. So entering tracks mode caps the carry at
        the tracker's `max_age_s`: a cat that left *before* the escalation clears in
        seconds rather than being pinned present for the rest of the cadence.
        """
        pipeline = self._escalating()
        assert pipeline.presence_labels(now=1000.0) == {CAT}

        # Escalated, and the cat is not there — it left before the camera woke up.
        pipeline._last_detections = []
        pipeline._last_tracks = []
        pipeline._last_analysis_gap_s = 0.2
        pipeline._last_analysis_at = 1002.0
        assert pipeline.presence_labels(now=1002.0) == {CAT}, "inside max_age_s, still standing"

        pipeline._last_analysis_at = 1009.0
        assert pipeline.presence_labels(now=1009.0) == set(), "past max_age_s, not a whole cadence"


class TestTheFirstFrameAfterAStart:
    """Before a *second* frame there is no measured cadence, and assuming one is wrong.

    A gap is only recorded between two analysed frames, so every source spends its first
    stretch after a restart with no cadence at all. The fallback was
    `SNAPSHOT_MODE_FRAME_GAP_S` — the threshold that *defines* streaming — which put the
    hold at six seconds. A Ring camera on a 30-90s cadence therefore went stale six
    seconds after its first frame, and its presence entities dropped to `unavailable`
    until the second snapshot arrived, once per restart per camera. It was visible in
    Home Assistant as an `unavailable` landing 6.2-6.6s into every restart.

    The provider already knows better: `next_frame_expected_by` is its own deadline for
    the next frame, and is what the live view and card previews already use to decide
    whether a source is overdue.
    """

    @staticmethod
    def _first_frame(deadline):
        """One analysed frame at t=1000, with the provider's next-frame deadline."""
        pipeline = _pipeline(_last_analysis_at=1000.0, _last_analysis_gap_s=0.0, _cadence_buckets=[])
        pipeline.provider = SimpleNamespace(next_frame_expected_by=lambda: deadline)
        return pipeline

    def test_a_snapshot_camera_is_not_stale_six_seconds_in(self):
        """The regression, at the exact timing observed: the next snapshot is 90s out,
        and the camera must still be believed a few seconds after its first frame."""
        pipeline = self._first_frame(deadline=1090.0)

        assert pipeline.presence_is_known(now=1006.5) is True
        assert pipeline.presence_is_known(now=1080.0) is True

    def test_a_provider_with_no_opinion_keeps_the_old_fallback(self):
        """`None` means "no opinion", and those providers must behave exactly as before
        rather than inheriting a snapshot camera's patience."""
        pipeline = self._first_frame(deadline=None)

        assert pipeline.presence_is_known(now=1005.0) is True
        assert pipeline.presence_is_known(now=1007.0) is False

    def test_a_deadline_already_past_does_not_collapse_the_hold(self):
        """A provider can report a deadline in the past — it means "overdue", not "zero
        patience", and a zero hold would call the source stale on the very next check."""
        pipeline = self._first_frame(deadline=995.0)

        assert pipeline.presence_is_known(now=1005.0) is True

    def test_a_provider_that_raises_does_not_break_presence(self):
        """This runs on the frame path. A provider bug must not take presence with it."""
        pipeline = self._first_frame(deadline=None)

        def _boom():
            raise RuntimeError("provider is confused")

        pipeline.provider = SimpleNamespace(next_frame_expected_by=_boom)
        assert pipeline.presence_is_known(now=1005.0) is True

    def test_a_measured_cadence_still_wins_once_there_is_one(self):
        """The deadline is a stand-in until the source has shown its own pace, not a
        replacement for it — otherwise a provider's estimate would override ten minutes
        of observed behaviour."""
        pipeline = self._first_frame(deadline=1090.0)
        pipeline._note_analysis_gap(0.05, 1000.05)
        pipeline._last_analysis_gap_s = 0.05
        pipeline._last_analysis_at = 1000.05

        # Measured at 0.05s, so the hold is a fraction of a second — not the provider's
        # ninety. A source that is genuinely streaming and stops must be noticed quickly.
        assert pipeline.presence_is_known(now=1010.0) is False
