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
from apps.helpers.actions.recordClipAction import (
    FrameRing,
    PlaybackClipBackend,
    RecordClipAction,
)
from apps.helpers.actions.Registry import action_registry, create_action
from apps.helpers.actions.saveSnapshotAction import SaveSnapshotAction
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
    def test_the_three_actions_are_registered(self):
        assert {"publish_mqtt", "record_clip", "save_snapshot"} <= set(action_registry)

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
        self.fail = fail

    def publish(self, topic, payload, retain=False, qos=0):
        if self.fail:
            raise RuntimeError("broker exploded")
        self.messages.append((topic, payload, retain, qos))


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


class TestAsyncMqttPublisher:
    def test_publish_is_bounded_and_drops_oldest(self):
        """Unbounded growth against a dead broker is the failure this prevents."""
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"), queue_size=5)
        for i in range(20):
            publisher.publish("topic", {"i": i})
        assert publisher._queue.qsize() == 5
        assert publisher.dropped == 15

    def test_publish_never_raises_without_a_connection(self):
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"))
        publisher.publish("topic", {"a": 1})  # must not raise

    def test_non_string_payloads_are_json_encoded(self):
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"))
        publisher.publish("topic", {"a": 1})
        _, payload, _, _ = publisher._queue.get_nowait()
        assert payload == '{"a": 1}'

    def test_stats_expose_counters_for_the_status_panel(self):
        publisher = AsyncMqttPublisher(MqttSettings(host="localhost"), queue_size=1)
        publisher.publish("t", "a")
        publisher.publish("t", "b")
        stats = publisher.stats()
        assert stats["dropped"] == 1 and stats["queued"] == 1

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
