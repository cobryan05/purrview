"""Unit tests for the StreamWatcher's per-source pipeline helpers.

These cover the parts of the watcher that are pure enough to test without services:
frame downscaling, per-source pacing, backoff, model-stack flattening and the status
row the UI reads. The rule logic itself is tested in `tests/helpers/test_watch_engine.py`.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from apps.helpers.tracking.iouTracker import IouTracker
from apps.helpers.tracking.tracker import Detection, LabelLifetimeSpec
from apps.helpers.watch.engine import WatchEngine
from apps.helpers.watch.events import EventPhase
from apps.helpers.watch.geometry import Zone
from apps.helpers.watch.rules import RuleSpec
from apps.tasks_server.tasks._watch_pipeline import (
    SourceHealth,
    WatchPipeline,
    build_label_lifetimes,
    downscale_to_width,
    expand_model_group,
    resolve_model_names,
)

CAT = "label-cat"
ZONE_UUID = "zone-1"
LITTER_ZONE = Zone(uuid=ZONE_UUID, name="litter box", kind="box", points=[(0.4, 0.4), (0.8, 0.8)])


class _FakeProvider:
    source_name = "Litter cam"

    async def start(self):
        pass

    async def stop(self):
        pass

    async def getNextImage(self):
        return None


def _pipeline(**overrides) -> WatchPipeline:
    rule = RuleSpec(
        uuid="rule-1",
        name="cat in box",
        subject_a_kind="label",
        subject_a_refs=[CAT],
        relation="contains",
        subject_b_kind="zone",
        subject_b_refs=[ZONE_UUID],
        threshold=0.6,
        min_duration_s=1.0,
        rearm_delay_s=5.0,
    )
    kwargs = dict(
        source_uuid="src-1",
        source_name="Litter cam",
        provider=_FakeProvider(),
        tracker=IouTracker(min_hits=1),
        engine=WatchEngine(rules=[rule], zones=[LITTER_ZONE], source_uuid="src-1"),
        model_names=["cat_detector"],
        analysis_fps=5.0,
        analysis_width=640,
    )
    kwargs.update(overrides)
    return WatchPipeline(**kwargs)


class TestDownscale:
    def test_shrinks_to_the_target_width_preserving_aspect(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out = downscale_to_width(image, 640)
        assert out.shape[1] == 640
        assert out.shape[0] == 360

    def test_never_upscales(self):
        """Making a 320px camera into a 640px payload costs bandwidth for no gain."""
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        assert downscale_to_width(image, 640).shape == (240, 320, 3)

    def test_zero_target_is_a_no_op(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        assert downscale_to_width(image, 0).shape == (240, 320, 3)

    def test_odd_aspect_ratios_keep_at_least_one_row(self):
        image = np.zeros((3, 4000, 3), dtype=np.uint8)
        assert downscale_to_width(image, 100).shape[0] >= 1


class TestPacing:
    def test_a_source_is_due_immediately_at_first(self):
        assert _pipeline().due(now=0.0) is True

    def test_a_source_is_not_due_again_until_its_interval_elapses(self):
        pipeline = _pipeline(analysis_fps=5.0)  # 0.2s interval
        pipeline.schedule_next(now=10.0)
        assert pipeline.due(now=10.1) is False
        assert pipeline.due(now=10.25) is True

    def test_pacing_is_per_source(self):
        """A 1 fps doorbell must not drag a 5 fps litter-box camera to its rate."""
        slow = _pipeline(analysis_fps=1.0)
        fast = _pipeline(analysis_fps=5.0)
        slow.schedule_next(now=0.0)
        fast.schedule_next(now=0.0)
        assert fast.due(now=0.3) is True
        assert slow.due(now=0.3) is False

    def test_zero_fps_means_no_pacing_delay(self):
        pipeline = _pipeline(analysis_fps=0.0)
        pipeline.schedule_next(now=5.0)
        assert pipeline.due(now=5.0) is True

    def test_a_backed_off_source_is_not_due(self):
        pipeline = _pipeline()
        pipeline.health.record_failure(now=10.0, error="camera unreachable")
        assert pipeline.due(now=11.0) is False


class TestSourceHealth:
    def test_backoff_grows_as_the_outage_lengthens(self):
        """Growth comes from elapsed downtime, so time has to pass for it to widen."""
        health = SourceHealth()
        health.record_failure(0.0, "boom")
        first = health.backoff_s
        health.record_failure(first, "boom")
        assert health.backoff_s > first

    def test_backoff_is_capped(self):
        health = SourceHealth()
        now = 0.0
        for _ in range(50):
            health.record_failure(now, "boom")
            now = health.next_attempt_at
        assert health.backoff_s == SourceHealth.BACKOFF_MAX_S

    def test_repeated_instant_failures_do_not_widen_the_backoff(self):
        """Retries that fail immediately say nothing about how long the camera is away.

        A relay path that is momentarily not-ready fails a reconnect in milliseconds;
        counting those as rungs pinned the wait at the ceiling while the source had
        been gone under a second, and left it unread for five minutes after it
        returned.
        """
        health = SourceHealth()
        for _ in range(50):
            health.record_failure(0.0, "no bytes returned")
        assert health.backoff_s == SourceHealth.BACKOFF_BASE_S

    def test_a_success_clears_the_backoff(self):
        """A camera that drops for a minute is normal; it must fully recover."""
        health = SourceHealth()
        health.record_failure(0.0, "boom")
        health.record_success(10.0)
        assert health.healthy is True
        assert health.backoff_s == 0.0
        assert health.last_error is None

    def test_successes_count_frames(self):
        health = SourceHealth()
        health.record_success(1.0)
        health.record_success(2.0)
        assert health.frames_analysed == 2


class TestPipelineStep:
    def test_a_cat_in_the_zone_opens_an_event(self):
        # Detections are in ANALYSIS-frame coordinates (inference runs on the
        # downscaled image), so a 640-wide frame puts the 0.4..0.8 zone at 256..512.
        pipeline = _pipeline(analysis_width=640)
        pipeline.prepare_frame(np.zeros((640, 640, 3), dtype=np.uint8))
        det = Detection(bbox=(300, 300, 400, 400), label_uuid=CAT, confidence=0.9)
        assert pipeline.step([det], now=0.0) == []
        signals = pipeline.step([det], now=1.5)
        assert [s.phase for s in signals] == [EventPhase.START]

    def test_prepare_frame_rescales_zones_to_the_analysis_size(self):
        """Zones are normalised; they only become pixels once the frame size is known."""
        pipeline = _pipeline(analysis_width=500)
        pipeline.prepare_frame(np.zeros((1000, 1000, 3), dtype=np.uint8))
        assert pipeline.engine.frame_width == 500
        assert pipeline.engine.frame_height == 500

    def test_prepare_frame_keeps_the_pre_downscale_frame(self):
        """The MQTT live feed publishes at the camera's resolution, not the model's."""
        pipeline = _pipeline(analysis_width=500)
        pipeline.prepare_frame(np.zeros((1000, 1000, 3), dtype=np.uint8))

        assert pipeline.last_frame.shape[:2] == (500, 500)
        assert pipeline.last_full_frame.shape[:2] == (1000, 1000)

    def test_an_undownscaled_frame_costs_a_reference_not_a_copy(self):
        """A camera already at or below the analysis width must not double its RAM."""
        pipeline = _pipeline(analysis_width=640)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.prepare_frame(frame)

        assert pipeline.last_full_frame is pipeline.last_frame

    def test_closing_open_events_emits_an_end(self):
        pipeline = _pipeline(analysis_width=640)
        pipeline.prepare_frame(np.zeros((640, 640, 3), dtype=np.uint8))
        det = Detection(bbox=(300, 300, 400, 400), label_uuid=CAT, confidence=0.9)
        pipeline.step([det], now=0.0)
        pipeline.step([det], now=1.5)
        signals = pipeline.close_open_events(now=3.0)
        assert [s.phase for s in signals] == [EventPhase.END]


class TestDeadTrackIds:
    """A vanished track_id is the only "this track just ended" signal the tracker
    exposes — the training-collector's per-track novelty sampling (§4B of
    COLLECTOR_UNIFICATION_PLAN.md) commits its held worst-case crop on this signal."""

    def _pipeline_with_tracker(self, **tracker_overrides) -> WatchPipeline:
        tracker = IouTracker(min_hits=1, max_age_s=0.5, **tracker_overrides)
        return _pipeline(analysis_width=640, tracker=tracker)

    def test_no_deaths_on_the_first_frame(self):
        pipeline = self._pipeline_with_tracker()
        pipeline.prepare_frame(np.zeros((640, 640, 3), dtype=np.uint8))
        det = Detection(bbox=(300, 300, 400, 400), label_uuid=CAT, confidence=0.9)
        pipeline.step([det], now=0.0)
        assert pipeline.dead_track_ids == []

    def test_a_track_that_ages_out_is_reported_dead_exactly_once(self):
        pipeline = self._pipeline_with_tracker()
        pipeline.prepare_frame(np.zeros((640, 640, 3), dtype=np.uint8))
        det = Detection(bbox=(300, 300, 400, 400), label_uuid=CAT, confidence=0.9)
        pipeline.step([det], now=0.0)
        (track,) = pipeline.tracks
        track_id = track.track_id

        # No detection this frame, and the tracker's max_age_s=0.5 has elapsed.
        pipeline.step([], now=1.0)
        assert pipeline.dead_track_ids == [track_id]

        # Reading it again without another step must not repeat the death.
        pipeline.step([], now=1.1)
        assert pipeline.dead_track_ids == []

    def test_a_still_live_track_is_never_reported_dead(self):
        pipeline = self._pipeline_with_tracker()
        pipeline.prepare_frame(np.zeros((640, 640, 3), dtype=np.uint8))
        det = Detection(bbox=(300, 300, 400, 400), label_uuid=CAT, confidence=0.9)
        pipeline.step([det], now=0.0)
        pipeline.step([det], now=0.2)
        assert pipeline.dead_track_ids == []


class TestStatusRow:
    def test_reports_the_fields_the_ui_table_needs(self):
        pipeline = _pipeline()
        pipeline.health.record_success(100.0)
        row = pipeline.status_row(now=102.0)
        assert row["source"] == "Litter cam"
        assert row["framesAnalysed"] == 1
        assert row["lastFrameAgoS"] == 2.0
        assert row["health"] == "ok"

    def test_buffer_capabilities_are_adopted_from_the_provider(self):
        """The relay is the only authority on all three, so they must travel together.

        `buffered` and `ingest_tier` were copied off the provider and `retention_s` was
        not, so every buffered camera reported "0s rewind" and `plan_window`'s pre-roll
        clamp — guarded on `retention_s > 0` — silently never ran.
        """
        pipeline = _pipeline()
        provider = SimpleNamespace(buffered=True, ingest_tier_name="native", retention_s=45.0)

        pipeline.adopt_buffer_capabilities(provider)

        assert pipeline.buffered is True
        assert pipeline.ingest_tier == "native"
        assert pipeline.retention_s == 45.0
        assert "45s rewind" in pipeline.status_row(0.0)["buffer"]

    def test_a_provider_reporting_nothing_leaves_the_source_unbuffered(self):
        """A provider with none of these attributes must not fake a rewind window."""
        pipeline = _pipeline()

        pipeline.adopt_buffer_capabilities(SimpleNamespace())

        assert pipeline.buffered is False
        assert pipeline.retention_s == 0.0
        assert "no buffer" in pipeline.status_row(0.0)["buffer"]

    def test_buffer_label_is_four_state_not_yes_no(self):
        """The honesty affordance: a snapshot camera must not look like it has rewind."""
        native = _pipeline(buffered=True, ingest_tier="native", retention_s=45)
        ffmpeg = _pipeline(buffered=True, ingest_tier="ffmpeg", retention_s=45)
        ring = _pipeline(buffered=False, ingest_tier="ring")
        none = _pipeline(buffered=False, ingest_tier="none")
        assert "native" in native.status_row(0.0)["buffer"]
        assert "ffmpeg" in ffmpeg.status_row(0.0)["buffer"]
        assert "ring" in ring.status_row(0.0)["buffer"]
        assert "no buffer" in none.status_row(0.0)["buffer"]

    def test_a_failing_source_reports_its_error(self):
        pipeline = _pipeline()
        pipeline.health.record_failure(now=1.0, error="connection refused")
        row = pipeline.status_row(now=2.0)
        assert row["health"] != "ok"
        assert row["lastError"] == "connection refused"

    def test_health_says_how_long_a_source_has_been_down(self):
        """The label leads with duration, not with a failure count.

        "failing (37)" invited the reading that something had gone wrong 37 times;
        what it meant was that one camera had been away for an hour and a half. With
        a backoff that grows, the count says almost nothing about the downtime.
        """
        pipeline = _pipeline()
        pipeline.health.record_failure(now=0.0, error="connection refused")

        recent = pipeline.status_row(now=30.0)["health"]
        assert "reconnecting" in recent
        assert "30s" in recent

        long_gone = pipeline.status_row(now=5400.0)["health"]
        assert "failed" in long_gone
        assert "1h 30m" in long_gone
        # Reporting it failed must not imply it has been abandoned.
        assert "still retrying" in long_gone


class TestLabelLifetimes:
    def test_builds_the_tracker_lookup_from_rows(self):
        rows = [
            {"label_uuids_json": [CAT], "ttl_s": 2.0, "is_anchor": False},
            {"label_uuids_json": ["label-box"], "ttl_s": 300.0, "is_anchor": True},
        ]
        lifetimes = build_label_lifetimes(rows)
        assert lifetimes[CAT] == LabelLifetimeSpec(ttl_s=2.0, is_anchor=False)
        assert lifetimes["label-box"].is_anchor is True

    def test_one_row_covers_every_label_it_names(self):
        lifetimes = build_label_lifetimes([{"label_uuids_json": [CAT, "label-box"], "ttl_s": 7.0}])
        assert lifetimes[CAT].ttl_s == 7.0
        assert lifetimes["label-box"].ttl_s == 7.0

    def test_accepts_camel_cased_rows(self):
        lifetimes = build_label_lifetimes([{"labelUuidsJson": [CAT], "ttlS": 5.0, "isAnchor": True}])
        assert lifetimes[CAT].ttl_s == 5.0

    def test_rows_without_a_label_are_skipped(self):
        assert build_label_lifetimes([{"ttl_s": 1.0}]) == {}

    def test_empty_input_is_fine(self):
        assert build_label_lifetimes(None) == {}

    def test_descendants_inherit_the_row_that_opted_in(self):
        ancestry = {"animal": set(), CAT: {"animal"}, "kitten": {CAT, "animal"}}
        lifetimes = build_label_lifetimes(
            [{"label_uuids_json": ["animal"], "ttl_s": 9.0, "include_descendants": True}],
            ancestry,
        )
        assert lifetimes["animal"].ttl_s == 9.0
        assert lifetimes[CAT].ttl_s == 9.0
        assert lifetimes["kitten"].ttl_s == 9.0

    def test_descendants_are_left_alone_without_the_flag(self):
        ancestry = {"animal": set(), CAT: {"animal"}}
        lifetimes = build_label_lifetimes([{"label_uuids_json": ["animal"], "ttl_s": 9.0}], ancestry)
        assert CAT not in lifetimes

    def test_a_label_named_outright_beats_one_inherited_from_a_parent(self):
        # Row order must not decide this: the specific statement about the cat is the one
        # the user wrote about the cat, whichever row it sits in.
        ancestry = {"animal": set(), CAT: {"animal"}}
        wide = {"label_uuids_json": ["animal"], "ttl_s": 30.0, "include_descendants": True}
        narrow = {"label_uuids_json": [CAT], "ttl_s": 1.0}
        assert build_label_lifetimes([wide, narrow], ancestry)[CAT].ttl_s == 1.0
        assert build_label_lifetimes([narrow, wide], ancestry)[CAT].ttl_s == 1.0
        assert build_label_lifetimes([narrow, wide], ancestry)["animal"].ttl_s == 30.0


class TestExpandModelGroup:
    def test_flattens_members_in_order(self):
        groups = {
            "g1": {
                "members": [
                    {"order": 1, "model_name": "second"},
                    {"order": 0, "model_name": "first"},
                ]
            }
        }
        assert expand_model_group("g1", groups) == ["first", "second"]

    def test_recurses_into_nested_groups(self):
        groups = {
            "g1": {"members": [{"order": 0, "child_group_uuid": "g2"}, {"order": 1, "model_name": "outer"}]},
            "g2": {"members": [{"order": 0, "model_name": "inner"}]},
        }
        assert expand_model_group("g1", groups) == ["inner", "outer"]

    def test_deduplicates_models_reachable_by_two_paths(self):
        groups = {
            "g1": {"members": [{"order": 0, "child_group_uuid": "g2"}, {"order": 1, "model_name": "shared"}]},
            "g2": {"members": [{"order": 0, "model_name": "shared"}]},
        }
        assert expand_model_group("g1", groups) == ["shared"]

    def test_a_cycle_does_not_hang(self):
        """An infinite recursion here would hang the watcher rather than report a bug."""
        groups = {
            "g1": {"members": [{"order": 0, "child_group_uuid": "g2"}]},
            "g2": {"members": [{"order": 0, "child_group_uuid": "g1"}, {"order": 1, "model_name": "m"}]},
        }
        assert expand_model_group("g1", groups) == ["m"]

    def test_missing_group_yields_nothing_rather_than_raising(self):
        assert expand_model_group("nope", {}) == []

    def test_none_group_yields_nothing(self):
        assert expand_model_group(None, {}) == []


class TestResolveModelNames:
    """A watch source picks either a Model Stack or one bare model."""

    def test_a_bare_model_needs_no_stack(self):
        assert resolve_model_names(None, "cat_detector", {}) == ["cat_detector"]

    def test_a_stack_is_still_expanded(self):
        groups = {"g1": {"members": [{"order": 0, "model_name": "first"}, {"order": 1, "model_name": "second"}]}}
        assert resolve_model_names("g1", None, groups) == ["first", "second"]

    def test_neither_yields_nothing(self):
        assert resolve_model_names(None, None, {}) == []

    def test_a_row_holding_both_prefers_the_stack_rather_than_raising(self):
        """The API rejects the pair, but a pre-existing row must still start the watcher."""
        groups = {"g1": {"members": [{"order": 0, "model_name": "from_stack"}]}}
        assert resolve_model_names("g1", "bare", groups) == ["from_stack"]


class TestMasks:
    """Masks suppress detections before the tracker sees them.

    Filtering tracks instead would let a masked object build up history, satisfy a
    rule's `min_duration_s`, and then vanish — a track flickering in and out of
    existence is worse for hysteresis than one that never existed.
    """

    MASK = Zone(uuid="mask-1", name="driveway", kind="box", points=[(0.0, 0.0), (0.3, 0.3)])

    def _detection(self, bbox):
        return Detection(bbox=bbox, label_uuid=CAT, label_name="cat", confidence=0.9)

    def test_a_detection_inside_a_mask_never_reaches_the_tracker(self):
        pipeline = _pipeline(masks=[self.MASK])
        pipeline.prepare_frame(np.zeros((640, 640, 3), dtype=np.uint8))
        w, h = pipeline.engine.frame_width, pipeline.engine.frame_height
        inside = self._detection((0.05 * w, 0.05 * h, 0.15 * w, 0.15 * h))
        pipeline.step([inside], now=0.0)
        assert pipeline.detections == []
        assert pipeline.tracks == []

    def test_a_detection_outside_a_mask_is_untouched(self):
        pipeline = _pipeline(masks=[self.MASK])
        pipeline.prepare_frame(np.zeros((640, 640, 3), dtype=np.uint8))
        w, h = pipeline.engine.frame_width, pipeline.engine.frame_height
        outside = self._detection((0.5 * w, 0.5 * h, 0.6 * w, 0.6 * h))
        pipeline.step([outside], now=0.0)
        assert len(pipeline.detections) == 1

    def test_with_no_masks_everything_passes(self):
        pipeline = _pipeline(masks=[])
        pipeline.prepare_frame(np.zeros((640, 640, 3), dtype=np.uint8))
        pipeline.step([self._detection((10, 10, 30, 30))], now=0.0)
        assert len(pipeline.detections) == 1

    def test_the_suppressed_count_is_reported(self):
        """A mask eating every detection and a dead model look identical otherwise."""
        pipeline = _pipeline(masks=[self.MASK])
        pipeline.prepare_frame(np.zeros((640, 640, 3), dtype=np.uint8))
        w, h = pipeline.engine.frame_width, pipeline.engine.frame_height
        pipeline.step(
            [
                self._detection((0.05 * w, 0.05 * h, 0.15 * w, 0.15 * h)),
                self._detection((0.5 * w, 0.5 * h, 0.6 * w, 0.6 * h)),
            ],
            now=0.0,
        )
        row = pipeline.status_row(now=1.0)
        assert row["masks"] == 1
        assert row["maskedLastFrame"] == 1

    def test_a_masked_cat_never_opens_an_event(self):
        pipeline = _pipeline(masks=[LITTER_ZONE])
        pipeline.prepare_frame(np.zeros((640, 640, 3), dtype=np.uint8))
        w, h = pipeline.engine.frame_width, pipeline.engine.frame_height
        in_the_zone = self._detection((0.5 * w, 0.5 * h, 0.7 * w, 0.7 * h))
        assert pipeline.step([in_the_zone], now=0.0) == []
        assert pipeline.step([in_the_zone], now=2.0) == []
        assert pipeline.step([in_the_zone], now=4.0) == []
