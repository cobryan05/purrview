"""Tests for the event action layer.

The recurring theme: an action must never be able to take the watcher down. A dead
broker, a full disk or a window that aged out of the RAM buffer are all normal events
on a 24/7 run, and every one of them has to degrade to a reported failure rather than
an exception on the frame loop.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

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
        assert {"publish_mqtt", "publish_mqtt_image", "record_clip", "save_snapshot"} <= set(action_registry)

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
        action = SaveSnapshotAction(output_dir="/proc/definitely-not-writable")
        result = await action.run(_signal(phase=EventPhase.START), _context(frame=frame))
        assert result.ok is False
