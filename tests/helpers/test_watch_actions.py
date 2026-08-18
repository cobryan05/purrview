"""Tests for the event action layer.

The recurring theme: an action must never be able to take the watcher down. A dead
broker, a full disk or a window that aged out of the RAM buffer are all normal events
on a 24/7 run, and every one of them has to degrade to a reported failure rather than
an exception on the frame loop.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from apps.helpers.actions import outputDir, saveTimelapseAction
from apps.helpers.actions.action import ActionContext, ActionResult
from apps.helpers.actions.mqttPublishAction import MqttPublishAction, slugify_topic
from apps.helpers.actions.mqttPublisher import AsyncMqttPublisher, MqttSettings
from apps.helpers.actions.mqttPublishImageAction import MqttPublishImageAction
from apps.helpers.actions.recordClipAction import (
    FrameRing,
    PlaybackClipBackend,
    RecordClipAction,
)
from apps.helpers.actions.Registry import action_registry, create_action
from apps.helpers.actions.saveSnapshotAction import SaveSnapshotAction
from apps.helpers.actions.saveTimelapseAction import SaveTimelapseAction
from apps.helpers.tracking.tracker import Detection
from apps.helpers.watch.events import EventPhase, WatchEventSignal


def _signal(phase=EventPhase.END, started_at=100.0, at=130.0, **overrides) -> WatchEventSignal:
    kwargs = dict(
        rule_uuid="rule-1",
        rule_name="cat using litter box",
        phase=phase,
        at=at,
        started_at=started_at,
        source_uuid="src-1",
        track_ids=[1],
        peak_confidence=0.9,
    )
    kwargs.update(overrides)
    return WatchEventSignal(**kwargs)


def _context(**overrides) -> ActionContext:
    kwargs = dict(
        profile_uuid="prof-1",
        profile_name="House cams",
        source_uuid="src-1",
        source_name="Litter cam",
        relay_path="relay_master_abc",
        buffered=True,
        retention_s=45.0,
        wallclock_offset=1_700_000_000.0,
    )
    kwargs.update(overrides)
    return ActionContext(**kwargs)


class TestRegistry:
    def test_the_actions_are_registered(self):
        assert {
            "publish_mqtt",
            "publish_mqtt_image",
            "record_clip",
            "save_snapshot",
            "save_timelapse",
        } <= set(action_registry)

    def test_create_by_kind(self):
        assert isinstance(create_action("publish_mqtt"), MqttPublishAction)

    def test_unknown_kind_raises(self):
        with pytest.raises(KeyError):
            create_action("send_carrier_pigeon")

    def test_every_action_exposes_a_ui_schema(self):
        for kind, cls in action_registry.items():
            schema = cls.params_schema()
            assert isinstance(schema, dict), kind
            for name, field in schema.items():
                assert "type" in field, f"{kind}.{name} needs a type for the UI form"


class TestSlugifyTopic:
    def test_spaces_and_case_are_normalised(self):
        assert slugify_topic("Litter Cam") == "litter-cam"

    def test_topic_separators_are_collapsed(self):
        """A '/' in a name would silently reshape the topic tree."""
        assert "/" not in slugify_topic("kitchen/cam")

    def test_wildcards_are_collapsed(self):
        assert slugify_topic("cam+1#") == "cam-1"

    def test_an_empty_name_still_yields_a_level(self):
        assert slugify_topic("") == "unnamed"


class _FakePublisher:
    def __init__(self, fail=False):
        self.messages = []
        self.coalesced = []
        self.fail = fail

    def publish(self, topic, payload, retain=False, qos=0, coalesce=False):
        if self.fail:
            raise RuntimeError("broker exploded")
        self.messages.append((topic, payload, retain, qos))
        self.coalesced.append(coalesce)


class TestMqttPublishAction:
    async def test_publishes_to_the_documented_topic_shape(self):
        publisher = _FakePublisher()
        action = MqttPublishAction(publisher=publisher)
        result = await action.run(_signal(), _context())
        assert result.ok is True
        topic = publisher.messages[0][0]
        assert topic == "house-cams/litter-cam/event/cat-using-litter-box"

    async def test_the_event_label_overrides_the_rule_name_in_the_topic(self):
        publisher = _FakePublisher()
        action = MqttPublishAction(publisher=publisher)
        await action.run(_signal(metadata={"eventLabel": "litterbox"}), _context())
        assert publisher.messages[0][0].endswith("/event/litterbox")

    async def test_payload_carries_the_event_details(self):
        publisher = _FakePublisher()
        await MqttPublishAction(publisher=publisher).run(_signal(), _context())
        payload = publisher.messages[0][1]
        assert payload["phase"] == "end"
        assert payload["profileUuid"] == "prof-1"
        assert payload["trackIds"] == [1]

    async def test_retain_is_off_by_default(self):
        """A retained event edge would replay to every new subscriber as if live."""
        publisher = _FakePublisher()
        await MqttPublishAction(publisher=publisher).run(_signal(), _context())
        assert publisher.messages[0][2] is False

    async def test_no_broker_configured_is_reported_not_raised(self):
        result = await MqttPublishAction(publisher=None).run(_signal(), _context())
        assert result.ok is False
        assert "Settings" in result.detail

    async def test_a_failing_broker_never_raises(self):
        result = await MqttPublishAction(publisher=_FakePublisher(fail=True)).run(_signal(), _context())
        assert result.ok is False


def _det(x1=10, y1=10, x2=40, y2=40, label_uuid="lab-1", name="tabby", confidence=0.8):
    return Detection(bbox=(x1, y1, x2, y2), label_uuid=label_uuid, label_name=name, confidence=confidence)


class TestMqttPublishImageAction:
    def _frame(self):
        return np.zeros((120, 160, 3), dtype=np.uint8)

    async def test_publishes_raw_jpeg_bytes_to_a_source_keyed_topic(self):
        """Not base64 in JSON: an MQTT camera wants the bytes, and base64 costs a third."""
        publisher = _FakePublisher()
        action = MqttPublishImageAction(publisher=publisher)
        result = await action.run(_signal(), _context(frame=self._frame(), detections=[_det()]))

        assert result.ok is True
        topic, payload, retain, _ = publisher.messages[0]
        assert topic == "house-cams/litter-cam/image"
        assert isinstance(payload, bytes)
        assert payload[:2] == b"\xff\xd8", "should be a JPEG"

    async def test_publishes_coalescing_so_a_backlog_collapses(self):
        publisher = _FakePublisher()
        action = MqttPublishImageAction(publisher=publisher)
        await action.run(_signal(), _context(frame=self._frame()))
        assert publisher.coalesced == [True]

    async def test_retains_by_default_unlike_the_event_action(self):
        """An image is state; retaining it is what makes a subscriber show one at once."""
        publisher = _FakePublisher()
        await MqttPublishImageAction(publisher=publisher).run(_signal(), _context(frame=self._frame()))
        assert publisher.messages[0][2] is True

    async def test_annotating_does_not_mutate_the_pipeline_frame(self):
        """The same array is written by snapshots and held in the rewind ring.

        Burning boxes into it would put annotations in the saved training images.
        """
        frame = self._frame()
        before = frame.copy()
        action = MqttPublishImageAction(publisher=_FakePublisher())
        await action.run(_signal(), _context(frame=frame, detections=[_det()]))
        assert np.array_equal(frame, before)

    async def test_annotation_actually_draws(self):
        action = MqttPublishImageAction(publisher=_FakePublisher())
        context = _context(frame=self._frame(), detections=[_det()], label_colours={"lab-1": "#ff0000"})
        drawn = action.render_jpeg(context)
        plain = MqttPublishImageAction(annotate=False, publisher=_FakePublisher()).render_jpeg(context)
        assert drawn != plain

    async def test_max_width_shrinks_the_payload(self):
        big = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        context = _context(frame=big)
        small = MqttPublishImageAction(max_width=160, publisher=_FakePublisher()).render_jpeg(context)
        full = MqttPublishImageAction(publisher=_FakePublisher()).render_jpeg(context)
        assert len(small) < len(full)

    async def test_max_fps_throttles_per_source_and_reports_ok(self):
        """Throttling is the ceiling working, so it must not count as a failed action."""
        publisher = _FakePublisher()
        action = MqttPublishImageAction(max_fps=1.0, publisher=publisher)
        first = await action.run(_signal(), _context(frame=self._frame()))
        second = await action.run(_signal(), _context(frame=self._frame()))

        assert first.ok is True and second.ok is True
        assert second.detail == "throttled"
        assert len(publisher.messages) == 1

    async def test_one_busy_source_does_not_spend_another_sources_budget(self):
        publisher = _FakePublisher()
        action = MqttPublishImageAction(max_fps=1.0, publisher=publisher)
        await action.run(_signal(), _context(frame=self._frame(), source_uuid="src-1"))
        await action.run(_signal(), _context(frame=self._frame(), source_uuid="src-2"))
        assert len(publisher.messages) == 2

    async def test_zero_max_fps_publishes_every_frame(self):
        publisher = _FakePublisher()
        action = MqttPublishImageAction(publisher=publisher)
        for _ in range(5):
            await action.run(_signal(), _context(frame=self._frame()))
        assert len(publisher.messages) == 5

    async def test_no_frame_is_reported_not_raised(self):
        result = await MqttPublishImageAction(publisher=_FakePublisher()).run(_signal(), _context(frame=None))
        assert result.ok is False and "frame" in result.detail.lower()

    async def test_no_broker_configured_is_reported_not_raised(self):
        result = await MqttPublishImageAction(publisher=None).run(_signal(), _context(frame=self._frame()))
        assert result.ok is False and "Settings" in result.detail

    async def test_a_failing_broker_never_raises(self):
        result = await MqttPublishImageAction(publisher=_FakePublisher(fail=True)).run(
            _signal(), _context(frame=self._frame())
        )
        assert result.ok is False


class TestMqttPublishImageResolution:
    """The published frame is the camera's, not the watcher's analysis downscale.

    `analysis_width` exists to make inference cheap. Letting it also decide what the
    live feed looks like is what made the feed soft, so the action prefers
    `context.full_frame` and scales the detections — which are in the *analysis*
    frame's pixel space — up to match.
    """

    def _frames(self):
        """An analysis frame and the 4x-wider frame it was downscaled from."""
        analysis = np.zeros((120, 160, 3), dtype=np.uint8)
        full = np.zeros((480, 640, 3), dtype=np.uint8)
        return analysis, full

    def _decode(self, payload: bytes) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)

    async def test_publishes_at_the_full_frames_resolution_by_default(self):
        analysis, full = self._frames()
        publisher = _FakePublisher()
        await MqttPublishImageAction(publisher=publisher).run(_signal(), _context(frame=analysis, full_frame=full))
        assert self._decode(publisher.messages[0][1]).shape[:2] == (480, 640)

    async def test_existing_configs_get_full_resolution_without_being_edited(self):
        """The stored params_json predates this option, so the default is the migration."""
        assert MqttPublishImageAction.params_schema()["full_resolution"]["default"] is True

    async def test_opting_out_publishes_the_analysis_frame(self):
        analysis, full = self._frames()
        publisher = _FakePublisher()
        await MqttPublishImageAction(full_resolution=False, publisher=publisher).run(
            _signal(), _context(frame=analysis, full_frame=full)
        )
        assert self._decode(publisher.messages[0][1]).shape[:2] == (120, 160)

    async def test_boxes_are_scaled_into_the_published_frames_space(self):
        """Unscaled, every box would land in the top-left quadrant of a 4x frame."""
        analysis, full = self._frames()
        det = _det(x1=10, y1=10, x2=40, y2=40)
        action = MqttPublishImageAction(publisher=_FakePublisher(), quality=100)
        image = self._decode(
            action.render_jpeg(
                _context(frame=analysis, full_frame=full, detections=[det], label_colours={"lab-1": "#ff0000"})
            )
        )

        # The 4x-scaled box spans 40..160; its far corner only has ink if it was scaled.
        assert image[160, 160].any()

    async def test_a_source_that_never_downscaled_still_publishes(self):
        """`full_frame` is the same array as `frame` when no downscale happened."""
        analysis = np.zeros((120, 160, 3), dtype=np.uint8)
        publisher = _FakePublisher()
        result = await MqttPublishImageAction(publisher=publisher).run(
            _signal(), _context(frame=analysis, full_frame=analysis, detections=[_det()])
        )
        assert result.ok is True
        assert self._decode(publisher.messages[0][1]).shape[:2] == (120, 160)

    async def test_a_missing_full_frame_falls_back_to_the_analysis_frame(self):
        analysis, _ = self._frames()
        publisher = _FakePublisher()
        result = await MqttPublishImageAction(publisher=publisher).run(
            _signal(), _context(frame=analysis, full_frame=None, detections=[_det()])
        )
        assert result.ok is True
        assert self._decode(publisher.messages[0][1]).shape[:2] == (120, 160)

    async def test_max_width_caps_the_full_resolution_frame(self):
        """The bandwidth knob still applies once the resolution ceiling is lifted."""
        analysis, full = self._frames()
        publisher = _FakePublisher()
        await MqttPublishImageAction(max_width=320, publisher=publisher).run(
            _signal(), _context(frame=analysis, full_frame=full)
        )
        assert self._decode(publisher.messages[0][1]).shape[:2] == (240, 320)

    async def test_boxes_are_scaled_to_the_capped_size_not_the_full_one(self):
        """max_width applies before annotation, so the scale must follow it."""
        analysis, full = self._frames()
        det = _det(x1=10, y1=10, x2=40, y2=40)
        action = MqttPublishImageAction(max_width=320, publisher=_FakePublisher(), quality=100)
        image = self._decode(
            action.render_jpeg(
                _context(frame=analysis, full_frame=full, detections=[det], label_colours={"lab-1": "#ff0000"})
            )
        )

        # 320/160 = 2x, so the far corner is at 80 — not the 160 the full frame implies.
        assert image[80, 80].any()
        assert not image[160, 160].any()

    async def test_neither_pipeline_frame_is_mutated(self):
        """Both are live pipeline arrays: the ring holds one and snapshots write it."""
        analysis, full = self._frames()
        before_analysis, before_full = analysis.copy(), full.copy()
        action = MqttPublishImageAction(publisher=_FakePublisher())
        await action.run(_signal(), _context(frame=analysis, full_frame=full, detections=[_det()]))

        assert np.array_equal(analysis, before_analysis)
        assert np.array_equal(full, before_full)


class TestAsyncMqttPublisher:
    def test_publish_is_bounded_and_drops_oldest(self):
        """Unbounded growth against a dead broker is the failure this prevents."""
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"), queue_size=5)
        for i in range(20):
            publisher.publish("topic", {"i": i})
        assert len(publisher._pending) == 5
        assert publisher.dropped == 15

    def test_publish_never_raises_without_a_connection(self):
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"))
        publisher.publish("topic", {"a": 1})  # must not raise

    def test_non_string_payloads_are_json_encoded(self):
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"))
        publisher.publish("topic", {"a": 1})
        assert publisher._pending[0].payload == '{"a": 1}'

    def test_stats_expose_counters_for_the_status_panel(self):
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"), queue_size=1)
        publisher.publish("t", "a")
        publisher.publish("t", "b")
        stats = publisher.stats()
        assert stats["dropped"] == 1 and stats["queued"] == 1

    def test_publish_is_bounded_by_bytes_not_only_by_count(self):
        """A message count alone was only ever right for small JSON events.

        An annotated-frame feed publishes JPEGs, where a 1000-message allowance is
        ~100 MB of RAM held for a broker that is not reading.
        """
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"), queue_size=1000, queue_bytes=1000)
        for i in range(20):
            publisher.publish(f"topic/{i}", "x" * 300)
        assert publisher._queued_bytes <= 1000
        assert len(publisher._pending) == 3
        assert publisher.dropped == 17

    def test_an_over_budget_message_is_still_sent_once(self):
        """Silently discarding every oversized payload would read as broker refusal."""
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"), queue_bytes=10)
        publisher.publish("topic", "x" * 500)
        assert len(publisher._pending) == 1

    def test_coalescing_replaces_a_queued_payload_for_the_same_topic(self):
        """A live feed's backlog must be one frame per topic, not a run of stale ones."""
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"))
        for i in range(10):
            publisher.publish("cam/image", f"frame-{i}", coalesce=True)
        assert len(publisher._pending) == 1
        assert publisher._pending[0].payload == "frame-9"
        # Superseded, not lost: counted apart from `dropped` so a working live feed
        # does not read as a failing broker.
        assert publisher.coalesced == 9
        assert publisher.dropped == 0

    def test_coalescing_is_per_topic(self):
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"))
        publisher.publish("cam/a", "a1", coalesce=True)
        publisher.publish("cam/b", "b1", coalesce=True)
        publisher.publish("cam/a", "a2", coalesce=True)
        assert [m.payload for m in publisher._pending] == ["a2", "b1"]

    def test_coalescing_tracks_bytes_when_it_replaces(self):
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"))
        publisher.publish("cam/image", "x" * 100, coalesce=True)
        publisher.publish("cam/image", "x" * 10, coalesce=True)
        assert publisher._queued_bytes == 10

    def test_event_edges_are_not_coalesced(self):
        """Every transition is its own fact; collapsing two loses information."""
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"))
        publisher.publish("evt", "started")
        publisher.publish("evt", "ended")
        assert [m.payload for m in publisher._pending] == ["started", "ended"]

    def test_a_sent_message_is_not_coalesced_onto(self):
        """The queued entry a topic coalesces onto must be forgotten once it leaves."""
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"))
        publisher.publish("cam/image", "first", coalesce=True)
        sent = publisher._pending.popleft()
        publisher._forget(sent)
        publisher.publish("cam/image", "second", coalesce=True)
        assert [m.payload for m in publisher._pending] == ["second"]
        assert publisher._queued_bytes == len("second")

    def test_settings_from_dict_applies_defaults(self):
        settings = MqttSettings.from_dict({"host": "broker.lan"})
        assert settings.port == 1883 and settings.prefix == "whiskerwatcher"

    def test_settings_strip_a_trailing_slash_from_the_prefix(self):
        assert MqttSettings.from_dict({"host": "h", "prefix": "home/"}).prefix == "home"


class _FakeRelay:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    async def extract_clip(self, path, start, duration_s, dest):
        self.calls.append((path, start, duration_s, dest))
        if self.fail:
            raise RuntimeError(self.fail)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"mp4")
        return {"path": str(dest), "sizeBytes": 3, "partialPreroll": False}


class TestRecordClipAction:
    async def test_only_cuts_on_the_event_end(self):
        """The window isn't known until the event closes — that's the point of a
        retroactive buffer."""
        relay = _FakeRelay()
        action = RecordClipAction(backend=PlaybackClipBackend(relay))
        result = await action.run(_signal(phase=EventPhase.START), _context())
        assert result.ok is True
        assert relay.calls == []

    async def test_writes_a_clip_on_the_event_end(self, tmp_path):
        relay = _FakeRelay()
        action = RecordClipAction(backend=PlaybackClipBackend(relay), output_dir=str(tmp_path))
        result = await action.run(_signal(), _context())
        assert result.ok is True
        assert relay.calls[0][0] == "relay_master_abc"

    def test_window_includes_preroll_and_postroll(self):
        action = RecordClipAction(preroll_s=10, postroll_s=5)
        start, duration = action.plan_window(_signal(started_at=100, at=130), _context())
        # 30s event + 10s pre + 5s post.
        assert duration == pytest.approx(45.0)

    def test_preroll_is_clamped_to_what_the_buffer_holds(self):
        """The buffer is RAM and shrinks across a relay restart, so the save-time
        check alone is not enough."""
        action = RecordClipAction(preroll_s=60, postroll_s=0)
        start, duration = action.plan_window(_signal(started_at=100, at=130), _context(retention_s=20.0))
        assert duration == pytest.approx(50.0)  # 30s event + 20s clamped pre-roll

    def test_max_length_caps_a_long_event(self):
        action = RecordClipAction(preroll_s=10, postroll_s=5, max_length_s=60)
        _, duration = action.plan_window(_signal(started_at=0, at=3600), _context())
        assert duration == 60

    def test_window_start_is_in_wallclock_not_monotonic(self):
        """Rule timing is monotonic; the buffer is addressed by wallclock."""
        action = RecordClipAction(preroll_s=10)
        start, _ = action.plan_window(_signal(started_at=100, at=130), _context(wallclock_offset=1_700_000_000.0))
        assert start == datetime.fromtimestamp(1_700_000_090.0, tz=timezone.utc)

    def test_clip_path_is_organised_by_day(self):
        action = RecordClipAction(output_dir="/clips")
        start = datetime(2026, 8, 10, 14, 30, 5, tzinfo=timezone.utc)
        path = action.clip_path(_signal(), _context(), start)
        assert path.parent.name == "2026-08-10"
        assert "litter-cam" in path.name and path.suffix == ".mp4"

    async def test_a_failed_extraction_is_reported_not_raised(self):
        action = RecordClipAction(backend=PlaybackClipBackend(_FakeRelay(fail="playback down")))
        result = await action.run(_signal(), _context())
        assert result.ok is False

    async def test_an_aged_out_window_is_flagged_as_terminal(self):
        """Past the retention window a retry can never succeed, so it must not queue
        forever."""
        action = RecordClipAction(backend=PlaybackClipBackend(_FakeRelay(fail="window has aged out of the buffer")))
        result = await action.run(_signal(), _context())
        assert result.ok is False
        assert result.data.get("expired") is True

    async def test_no_backend_is_reported(self):
        result = await RecordClipAction(backend=None).run(_signal(), _context())
        assert result.ok is False

    async def test_a_source_without_a_relay_path_fails_cleanly(self):
        action = RecordClipAction(backend=PlaybackClipBackend(_FakeRelay()))
        result = await action.run(_signal(), _context(relay_path=None))
        assert result.ok is False


class TestFrameRing:
    def test_frames_are_evicted_by_age(self):
        ring = FrameRing(max_seconds=10, max_mb=100)
        for i in range(20):
            ring.push(float(i), b"x" * 100)
        # Only the last ~10 seconds survive.
        assert ring.frame_count <= 11
        assert ring.coverage_s() <= 10

    def test_frames_are_evicted_by_bytes_not_count(self):
        """Frame sizes vary by an order of magnitude between cameras, so a count-based
        bound gives wildly different memory for the same config."""
        ring = FrameRing(max_seconds=3600, max_mb=0.001)  # ~1 KB
        for i in range(50):
            ring.push(float(i), b"x" * 500)
        assert ring.size_bytes <= 1024 + 500

    def test_frames_between_selects_the_window(self):
        ring = FrameRing(max_seconds=100, max_mb=100)
        for i in range(10):
            ring.push(float(i), bytes([i]))
        selected = ring.frames_between(3.0, 6.0)
        assert [f[0] for f in selected] == [3.0, 4.0, 5.0, 6.0]

    def test_an_empty_ring_reports_no_coverage(self):
        assert FrameRing().coverage_s() == 0.0


class TestSaveSnapshotAction:
    async def test_writes_a_jpg_and_a_sidecar(self, tmp_path):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        action = SaveSnapshotAction(output_dir=str(tmp_path))
        result = await action.run(_signal(phase=EventPhase.START), _context(frame=frame))
        assert result.ok is True
        written = Path(result.data["path"])
        assert written.exists()
        assert written.with_suffix(".jpg.json").exists()

    async def test_skips_the_event_end(self):
        """An END snapshot would capture the empty scene after the subject left."""
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        action = SaveSnapshotAction()
        result = await action.run(_signal(phase=EventPhase.END), _context(frame=frame))
        assert result.ok is True
        assert "skipped" in result.detail

    async def test_no_frame_is_reported_not_raised(self):
        result = await SaveSnapshotAction().run(_signal(phase=EventPhase.START), _context(frame=None))
        assert result.ok is False

    async def test_an_unwritable_directory_is_reported_not_raised(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        # The dataset root itself is what has to be unwritable: a path outside it is now
        # re-rooted rather than honoured, so "/proc/..." as an `output_dir` would land
        # somewhere perfectly writable and stop testing anything.
        with patch.object(outputDir, "FILES_ROOT", Path("/proc")):
            action = SaveSnapshotAction(output_dir="definitely-not-writable")
            result = await action.run(_signal(phase=EventPhase.START), _context(frame=frame))
        assert result.ok is False


class TestSaveTimelapseAction:
    """The interval gate is the whole action: everything else is a jpg write.

    A timelapse runs on `frame_analysed`, which fires several times a second per camera,
    so "not due" is the outcome for ~99.5% of calls at the default 30 s. Getting that
    wrong does not fail loudly — it silently fills a disk.
    """

    @pytest.fixture(autouse=True)
    def _dataset_root(self, tmp_path):
        """A FILES_ROOT the test owns, for the whole test.

        Not merely for construction: the directory is a template now, so it is resolved
        again on every frame — a patch that ended at construction let the writes land in
        the container's real image_datasets.
        """
        with patch.object(outputDir, "FILES_ROOT", Path(tmp_path)):
            yield

    def _action(self, tmp_path, **params):
        return SaveTimelapseAction(**params)

    async def test_writes_one_frame_into_a_per_day_directory(self, tmp_path):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        action = self._action(tmp_path, output_dir="timelapse")
        result = await action.run(_signal(), _context(frame=frame))
        assert result.ok is True
        written = Path(result.data["path"])
        assert written.exists()
        # output_dir / YYYY-MM-DD / <source>_HH-MM-SS.jpg
        assert written.parent.parent == tmp_path / "timelapse"
        datetime.strptime(written.parent.name, "%Y-%m-%d")
        assert written.name.startswith("litter-cam_")
        # No sidecar: 2880 of them a day describe nothing anyone will query.
        assert not written.with_suffix(".jpg.json").exists()

    async def test_a_second_frame_in_the_same_interval_is_skipped(self, tmp_path):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        action = self._action(tmp_path, output_dir="timelapse", interval_s=3600)
        first = await action.run(_signal(), _context(frame=frame))
        second = await action.run(_signal(), _context(frame=frame))
        assert first.ok is True and "wrote" in first.detail
        # Skipping is a success, not a failure — a failure would be counted against the
        # watcher and logged on every frame.
        assert second.ok is True and second.detail == "not due"
        assert len(list((tmp_path / "timelapse").rglob("*.jpg"))) == 1

    async def test_the_next_interval_writes_again(self, tmp_path):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        action = self._action(tmp_path, output_dir="timelapse", interval_s=30)
        # 1_699_999_980 is a multiple of 30, so the bucket it opens runs to +30.
        start = 1_699_999_980.0
        assert action._due("src-1", start) is not None
        action._written_bucket["src-1"] = int(start // 30)
        assert action._due("src-1", start + 29.9) is None
        assert action._due("src-1", start + 30.0) is not None

    async def test_the_interval_is_per_source(self, tmp_path):
        """One busy camera must not consume another camera's interval budget."""
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        action = self._action(tmp_path, output_dir="timelapse", interval_s=3600)
        await action.run(_signal(), _context(frame=frame, source_uuid="src-1", source_name="one"))
        other = await action.run(_signal(), _context(frame=frame, source_uuid="src-2", source_name="two"))
        assert other.ok is True and "wrote" in other.detail

    async def test_a_zero_interval_is_clamped(self, tmp_path):
        """Honouring it would write every analysed frame, forever, to a disk."""
        assert self._action(tmp_path, interval_s=0).interval_s >= 1.0

    async def test_prefers_the_full_resolution_frame(self, tmp_path):
        analysed = np.zeros((32, 32, 3), dtype=np.uint8)
        full = np.zeros((64, 64, 3), dtype=np.uint8)
        action = self._action(tmp_path, output_dir="timelapse")
        result = await action.run(_signal(), _context(frame=analysed, full_frame=full))
        assert cv2.imread(result.data["path"]).shape[:2] == (64, 64)

    async def test_falls_back_to_the_analysis_frame(self, tmp_path):
        analysed = np.zeros((32, 32, 3), dtype=np.uint8)
        action = self._action(tmp_path, output_dir="timelapse")
        result = await action.run(_signal(), _context(frame=analysed, full_frame=None))
        assert cv2.imread(result.data["path"]).shape[:2] == (32, 32)

    async def test_no_frame_is_reported_not_raised(self, tmp_path):
        action = self._action(tmp_path, output_dir="timelapse")
        result = await action.run(_signal(), _context(frame=None, full_frame=None))
        assert result.ok is False

    async def test_an_unwritable_directory_does_not_retry_every_frame(self, tmp_path):
        """A misconfiguration must not become a hot loop of encode attempts."""
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        action = self._action(tmp_path, output_dir="timelapse", interval_s=3600)
        # The dataset root is what has to be unwritable: an output_dir outside it is
        # re-rooted rather than honoured, so it would land somewhere perfectly writable.
        with patch.object(outputDir, "FILES_ROOT", Path("/proc")):
            first = await action.run(_signal(), _context(frame=frame))
            second = await action.run(_signal(), _context(frame=frame))
        assert first.ok is False
        assert second.detail == "not due"


class TestTimelapseOutputDirResolution:
    """Three spellings reach `output_dir`, and one of them used to mean the container's
    own root — the picker's `/timelapse` is rooted at FILES_ROOT, not at `/`."""

    def _resolve(self, root, value):
        with patch.object(outputDir, "FILES_ROOT", Path(root)):
            return outputDir.resolve_output_dir(value, saveTimelapseAction.DEFAULT_TIMELAPSE_DIR)

    def test_a_relative_path_is_joined_onto_the_dataset_root(self, tmp_path):
        assert self._resolve(tmp_path, "timelapse/front") == tmp_path / "timelapse/front"

    def test_the_pickers_leading_slash_is_the_dataset_root(self, tmp_path):
        assert self._resolve(tmp_path, "/timelapse") == tmp_path / "timelapse"

    def test_an_absolute_path_already_inside_the_root_is_kept(self, tmp_path):
        assert self._resolve(tmp_path, str(tmp_path / "snaps")) == tmp_path / "snaps"

    def test_an_empty_value_falls_back_to_the_default(self, tmp_path):
        assert self._resolve(tmp_path, "  ") == tmp_path / saveTimelapseAction.DEFAULT_TIMELAPSE_DIR

    def test_climbing_out_of_the_root_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            self._resolve(tmp_path, "../../etc/cron.d")


class TestSnapshotDirectoryTemplates:
    """The snapshot folder is a template over the same vocabulary the topics use.

    That is what makes "a folder per camera" or "a folder per day" a thing you can write
    rather than a feature request — and it is also a path built from user input, so the
    substituted values must not be able to change the *shape* of the path.
    """

    def _path(self, tmp_path, output_dir, **ctx):
        with patch.object(outputDir, "FILES_ROOT", Path(tmp_path)):
            action = SaveSnapshotAction(output_dir=output_dir)
            when = datetime(2026, 8, 18, 14, 30, 0).astimezone()
            return action.snapshot_path(_signal(phase=EventPhase.START), _context(**ctx), when)

    def test_a_plain_folder_still_gets_the_day_level(self, tmp_path):
        path = self._path(tmp_path, "watch_snapshots")
        assert path.parent == tmp_path / "watch_snapshots" / "2026-08-18"

    def test_source_and_rule_can_name_folders(self, tmp_path):
        path = self._path(tmp_path, "snaps/{source}/{rule}")
        assert path.parent == tmp_path / "snaps/litter-cam/cat-using-litter-box/2026-08-18"

    def test_placing_the_date_suppresses_the_automatic_day_level(self, tmp_path):
        """Otherwise saying where the date goes silently gets you a second one."""
        path = self._path(tmp_path, "snaps/{date}/{source}")
        assert path.parent == tmp_path / "snaps/2026-08-18/litter-cam"

    def test_the_time_token_is_filename_safe(self, tmp_path):
        path = self._path(tmp_path, "snaps/{date}/{hour}")
        assert path.parent == tmp_path / "snaps/2026-08-18/14"

    def test_a_slash_in_a_value_cannot_add_a_folder_level(self, tmp_path):
        """A camera named "Kitchen / Upstairs" must not fan one folder out into two."""
        path = self._path(tmp_path, "snaps/{source}", source_name="Kitchen / Upstairs")
        assert path.parent == tmp_path / "snaps/kitchen-upstairs/2026-08-18"

    def test_a_value_cannot_climb_out_of_the_root(self, tmp_path):
        path = self._path(tmp_path, "snaps/{source}", source_name="../../etc")
        assert tmp_path in path.parents

    def test_an_absolute_legacy_path_inside_the_root_still_works(self, tmp_path):
        """A watcher configured before the picker existed must keep writing where it did."""
        path = self._path(tmp_path, str(tmp_path / "watch_snapshots"))
        assert path.parent == tmp_path / "watch_snapshots" / "2026-08-18"

    def test_the_pickers_leading_slash_is_the_dataset_root(self, tmp_path):
        path = self._path(tmp_path, "/snaps")
        assert path.parent == tmp_path / "snaps" / "2026-08-18"

    async def test_a_templated_folder_is_actually_written(self, tmp_path):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        with patch.object(outputDir, "FILES_ROOT", Path(tmp_path)):
            action = SaveSnapshotAction(output_dir="snaps/{source}")
            result = await action.run(_signal(phase=EventPhase.START), _context(frame=frame))
        assert result.ok is True
        written = Path(result.data["path"])
        assert written.exists()
        assert written.parent.parent == tmp_path / "snaps" / "litter-cam"


class TestTimelapseDirectoryTemplates:
    """One action covers every camera in the watcher, so the folder has to be able to
    split them. Without `{source}` four cameras interleave in one directory — the
    filenames still say which camera, but nothing sorts them."""

    @pytest.fixture(autouse=True)
    def _dataset_root(self, tmp_path):
        with patch.object(outputDir, "FILES_ROOT", Path(tmp_path)):
            yield

    def _path(self, output_dir, **ctx):
        action = SaveTimelapseAction(output_dir=output_dir)
        when = datetime(2026, 8, 18, 14, 30, 0).astimezone()
        return action.frame_path(_signal(), _context(**ctx), when)

    def test_source_gives_each_camera_its_own_folder(self, tmp_path):
        one = self._path("timelapse/{source}", source_name="Back room")
        two = self._path("timelapse/{source}", source_name="Front door")
        assert one.parent == tmp_path / "timelapse/back-room/2026-08-18"
        assert two.parent == tmp_path / "timelapse/front-door/2026-08-18"

    def test_without_a_token_every_camera_shares_one_folder(self, tmp_path):
        one = self._path("timelapse", source_name="Back room")
        two = self._path("timelapse", source_name="Front door")
        assert one.parent == two.parent
        # The filename still identifies the camera, which is what makes the shared
        # folder survivable rather than actively wrong.
        assert one.name != two.name

    def test_placing_the_date_suppresses_the_automatic_day_level(self, tmp_path):
        path = self._path("timelapse/{date}/{source}", source_name="Back room")
        assert path.parent == tmp_path / "timelapse/2026-08-18/back-room"

    def test_the_camera_stays_in_the_filename_even_when_it_names_a_folder(self, tmp_path):
        path = self._path("timelapse/{source}", source_name="Back room")
        assert path.name.startswith("back-room_")

    def test_a_slash_in_a_camera_name_cannot_add_a_level(self, tmp_path):
        path = self._path("timelapse/{source}", source_name="Kitchen / Upstairs")
        assert path.parent == tmp_path / "timelapse/kitchen-upstairs/2026-08-18"

    def test_a_template_with_no_fixed_folder_is_refused(self, tmp_path):
        """The retention sweep bounds itself to the literal part of the path. With none,
        the only directory that would bound it is the dataset root itself."""
        with pytest.raises(ValueError):
            SaveTimelapseAction(output_dir="{source}/frames")

    async def test_two_sources_actually_write_to_two_folders(self, tmp_path):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        action = SaveTimelapseAction(output_dir="timelapse/{source}", interval_s=3600)
        a = await action.run(_signal(), _context(frame=frame, source_uuid="s1", source_name="Back room"))
        b = await action.run(_signal(), _context(frame=frame, source_uuid="s2", source_name="Front door"))
        assert Path(a.data["path"]).exists() and Path(b.data["path"]).exists()
        assert Path(a.data["path"]).parent != Path(b.data["path"]).parent
