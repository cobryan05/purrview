"""Tests for the event action layer.

The recurring theme: an action must never be able to take the watcher down. A dead
broker, a full disk or a window that aged out of the RAM buffer are all normal events
on a 24/7 run, and every one of them has to degrade to a reported failure rather than
an exception on the frame loop.
"""

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest

from apps.helpers.actions import outputDir, saveTimelapseAction
from apps.helpers.actions.action import ActionContext, ActionResult
from apps.helpers.actions.collectTrainingImagesAction import CollectTrainingImagesAction
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
from apps.helpers.db.db_client import DETECTOR_MISS_TAG_UUID
from apps.helpers.tracking.tracker import Detection, Track
from apps.helpers.training.collector import TrackedCrop
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
        # `durationS` because the real playback server reports what it actually cut,
        # which is what the assembled clip's playing time is summed from.
        return {"path": str(dest), "sizeBytes": 3, "durationS": duration_s, "partialPreroll": False}


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

    async def test_the_overlay_checkbox_reaches_the_result(self, tmp_path):
        """The action does not write the sidecar — the task does, because only it holds
        the detection ring. This flag is how the choice gets there."""
        relay = _FakeRelay()
        action = RecordClipAction(backend=PlaybackClipBackend(relay), output_dir=str(tmp_path), save_overlay=False)

        result = await action.run(_signal(), _context())

        assert result.data["overlay"] is False

    async def test_the_overlay_is_on_by_default(self, tmp_path):
        relay = _FakeRelay()
        action = RecordClipAction(backend=PlaybackClipBackend(relay), output_dir=str(tmp_path))

        result = await action.run(_signal(), _context())

        assert result.data["overlay"] is True

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


def _fake_concat(parts, dest, max_length_s=None):
    """Stand-in for `concat_parts`, matching the one behaviour callers depend on.

    The real function creates its own parent directory. A double that does not looks
    identical until the *only* other thing that would have created it — the tail
    extraction — is the thing under test for failing.
    """
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_bytes(b"joined")
    return 6


class TestRecordClipFromDrainedChunks:
    """An event that outlived the buffer is assembled from chunks plus the live tail.

    Without this, cutting only at the closing edge kept whatever survived: a ten-minute
    event against a 45 s buffer produced a 45 s clip, silently, with the front evicted
    long before anything asked for it.
    """

    def _chunks(self, tmp_path, count=2, gap_before=False):
        from apps.helpers.recordings.clipDrain import DrainedChunk

        base = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
        chunks = []
        for i in range(count):
            path = tmp_path / f"chunk-{i}.mp4"
            path.write_bytes(b"mp4")
            chunks.append(
                DrainedChunk(
                    path=path,
                    start=base + timedelta(seconds=60 * i),
                    end=base + timedelta(seconds=60 * (i + 1)),
                    gap_before=gap_before and i == 1,
                )
            )
        return chunks

    async def test_chunks_are_concatenated_with_the_tail(self, tmp_path, monkeypatch):
        joined = {}

        def fake_concat(parts, dest, max_length_s=None):
            joined["parts"] = list(parts)
            joined["maxLength"] = max_length_s
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"joined")
            return 6

        monkeypatch.setattr("apps.helpers.recordings.clipDrain.concat_parts", fake_concat)
        relay = _FakeRelay()
        action = RecordClipAction(backend=PlaybackClipBackend(relay), output_dir=str(tmp_path / "clips"))
        chunks = self._chunks(tmp_path)

        result = await action.run(_signal(), _context(clip_chunks=chunks))

        assert result.ok is True
        # Two chunks plus the tail cut from what is still buffered.
        assert len(joined["parts"]) == 3
        assert joined["parts"][:2] == [c.path for c in chunks]

    async def test_an_assembled_clip_reports_its_playing_time(self, tmp_path, monkeypatch):
        """Reporting nothing here is not a harmless omission: the recording row stores a
        duration of 0, so `ended_at` lands empty and the overlay writer — which needs a
        window to slice the detection ring over — declines every assembled clip. That is
        how long events silently lost "show what the watcher saw"."""
        monkeypatch.setattr("apps.helpers.recordings.clipDrain.concat_parts", _fake_concat)
        action = RecordClipAction(
            backend=PlaybackClipBackend(_FakeRelay()), postroll_s=5, output_dir=str(tmp_path / "clips")
        )

        result = await action.run(_signal(), _context(clip_chunks=self._chunks(tmp_path)))

        # Two 60 s chunks, plus a tail of the 10 s from the last drain to the event end
        # and 5 s of post-roll.
        assert result.data["durationS"] == pytest.approx(135.0)

    async def test_the_playing_time_is_capped_the_way_the_muxer_caps_it(self, tmp_path, monkeypatch):
        """`concat_parts` trims the output to `max_length_s`, so a reported duration
        longer than the cap would place the overlay's last boxes past the end of the
        video."""
        monkeypatch.setattr("apps.helpers.recordings.clipDrain.concat_parts", _fake_concat)
        action = RecordClipAction(
            backend=PlaybackClipBackend(_FakeRelay()), max_length_s=90.0, output_dir=str(tmp_path / "clips")
        )

        result = await action.run(_signal(), _context(clip_chunks=self._chunks(tmp_path)))

        assert result.data["durationS"] == pytest.approx(90.0)

    async def test_an_assembled_clip_still_gets_an_overlay(self, tmp_path, monkeypatch):
        """Outliving the buffer is not a reason to lose the boxes — a long event is
        exactly the one you want to inspect."""
        monkeypatch.setattr("apps.helpers.recordings.clipDrain.concat_parts", _fake_concat)
        action = RecordClipAction(
            backend=PlaybackClipBackend(_FakeRelay()), output_dir=str(tmp_path / "clips"), save_overlay=True
        )

        result = await action.run(_signal(), _context(clip_chunks=self._chunks(tmp_path)))

        assert result.data["overlay"] is True

    async def test_a_gapped_assembly_gets_no_overlay(self, tmp_path, monkeypatch):
        """`build_overlay` places a box by subtracting wallclocks, which is wrong by the
        length of the missing stretch for everything after a gap. Boxes that slide
        further off the picture read as a detector fault rather than a timing one."""
        monkeypatch.setattr("apps.helpers.recordings.clipDrain.concat_parts", _fake_concat)
        action = RecordClipAction(
            backend=PlaybackClipBackend(_FakeRelay()), output_dir=str(tmp_path / "clips"), save_overlay=True
        )

        result = await action.run(_signal(), _context(clip_chunks=self._chunks(tmp_path, gap_before=True)))

        assert result.data["overlay"] is False

    async def test_the_tail_covers_from_the_last_chunk_to_the_event_end(self, tmp_path, monkeypatch):
        """Without the tail the clip would stop at the last drain, losing the ending —
        the part the user is most likely watching for."""
        monkeypatch.setattr("apps.helpers.recordings.clipDrain.concat_parts", _fake_concat)
        relay = _FakeRelay()
        action = RecordClipAction(backend=PlaybackClipBackend(relay), postroll_s=5, output_dir=str(tmp_path / "clips"))
        chunks = self._chunks(tmp_path)

        await action.run(_signal(), _context(clip_chunks=chunks))

        _path, tail_start, tail_duration, _dest = relay.calls[0]
        assert tail_start == chunks[-1].end

    async def test_a_failed_tail_still_writes_the_chunks(self, tmp_path, monkeypatch):
        """The chunks are the bulk of the clip and already on disk — losing the last few
        seconds beats losing the whole event."""
        monkeypatch.setattr("apps.helpers.recordings.clipDrain.concat_parts", _fake_concat)
        action = RecordClipAction(
            backend=PlaybackClipBackend(_FakeRelay(fail="buffer gone")), output_dir=str(tmp_path / "clips")
        )

        result = await action.run(_signal(), _context(clip_chunks=self._chunks(tmp_path)))
        assert result.ok is True

    async def test_a_missed_drain_is_reported_as_a_partial_clip(self, tmp_path, monkeypatch):
        """A clip with a hole is a different artifact from a complete one."""
        monkeypatch.setattr("apps.helpers.recordings.clipDrain.concat_parts", _fake_concat)
        action = RecordClipAction(backend=PlaybackClipBackend(_FakeRelay()), output_dir=str(tmp_path / "clips"))

        result = await action.run(_signal(), _context(clip_chunks=self._chunks(tmp_path, gap_before=True)))
        assert result.data["hasGap"] is True
        assert result.data["partialPreroll"] is True

    async def test_a_clean_assembly_is_not_flagged_partial(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.helpers.recordings.clipDrain.concat_parts", _fake_concat)
        action = RecordClipAction(backend=PlaybackClipBackend(_FakeRelay()), output_dir=str(tmp_path / "clips"))

        result = await action.run(_signal(), _context(clip_chunks=self._chunks(tmp_path)))
        assert result.data["hasGap"] is False
        assert result.data["partialPreroll"] is False

    async def test_no_chunks_takes_the_ordinary_single_cut_path(self, tmp_path):
        """Most events close inside the buffer and must not touch any of this."""
        relay = _FakeRelay()
        action = RecordClipAction(backend=PlaybackClipBackend(relay), output_dir=str(tmp_path / "clips"))

        result = await action.run(_signal(), _context())
        assert result.ok is True
        assert len(relay.calls) == 1, "a single cut, not a tail plus an assembly"
        assert "partCount" not in (result.data or {})


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

    def test_the_ring_really_encodes_an_mp4(self, tmp_path):
        """Against the real ffmpeg, writing to the `.part` name the backend uses.

        This is the whole of Tier C's clip support — a snapshot poller or MQTT feed has
        no continuous stream to cut from, so a failure here means those sources never
        produce a clip at all. It failed for exactly that reason: the output is written
        to `<name>.mp4.part` and ffmpeg picks its muxer from the extension, so `.part`
        left it with "Unable to find a suitable output format" on every attempt. Nothing
        caught it because the only coverage was of the ring's eviction arithmetic.
        """
        from apps.helpers.actions.recordClipAction import RingClipBackend

        frames = []
        for i in range(4):
            ok, buffer = cv2.imencode(".jpg", np.full((240, 320, 3), 40 + i * 30, dtype=np.uint8))
            assert ok
            frames.append((float(i), buffer.tobytes()))

        dest = tmp_path / "ring.mp4"
        size = RingClipBackend._write_mp4(frames, dest)

        assert dest.is_file() and size > 0
        # No `.part` left behind, and the file is one a browser's <video> can open.
        assert not dest.with_suffix(".mp4.part").exists()
        capture = cv2.VideoCapture(str(dest))
        try:
            assert capture.isOpened()
            assert capture.read()[0]
        finally:
            capture.release()


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


def _frame_signal(**overrides) -> WatchEventSignal:
    """A `frame_analysed` dispatch, as `_dispatch_frame` actually builds one:
    empty rule, phase UPDATE. `rule_uuid=""` is what `CollectTrainingImagesAction.run()`
    (and RecordEventTimelapseAction's, which this mirrors) uses to tell a frame
    apart from a real event transition — see StreamWatcherTask._dispatch_frame."""
    return _signal(rule_uuid="", rule_name="", phase=EventPhase.UPDATE, track_ids=[], peak_confidence=0.0, **overrides)


class TestCollectTrainingImagesAction:
    """The watcher-side half of collector attachment (phases 4-5,
    COLLECTOR_UNIFICATION_PLAN.md). The collector runtime itself
    (`TrainingCollector`) is exercised elsewhere — this only covers what the
    action owns: resolving the collector, continuous vs event-scoped mode,
    the per-source interval gate, provenance, and building a correct
    `FrameOffer` out of an `ActionContext`.
    """

    def _action(self, collector=None, **overrides):
        return CollectTrainingImagesAction(collector_for=lambda uuid: collector, **overrides)

    def _full_frame_context(self, **overrides):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        return _context(
            full_frame=frame,
            analysis_base64='{"format": "jpg", "data": "AA==", "metadata": {}}',
            frame_seq=7,
            **overrides,
        )

    async def test_registered_kind(self):
        assert "collect_training_images" in action_registry
        assert isinstance(
            create_action("collect_training_images", collector_for=lambda uuid: None), CollectTrainingImagesAction
        )

    async def test_unresolvable_collector_fails_without_raising(self):
        action = self._action(collector=None, collector_uuid="missing")
        result = await action.run(_frame_signal(), self._full_frame_context())
        assert result.ok is False

    async def test_no_frame_available_fails_without_raising(self):
        collector = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1")
        result = await action.run(_frame_signal(), _context(full_frame=None, analysis_base64=None))
        assert result.ok is False
        collector.offer.assert_not_called()

    async def test_first_offer_on_a_source_always_goes_through(self):
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=10.0)

        result = await action.run(_frame_signal(), self._full_frame_context(source_uuid="s1"))

        assert result.ok is True
        collector.offer.assert_awaited_once()

    async def test_a_second_offer_inside_the_interval_is_skipped(self):
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1000.0)

        await action.run(_frame_signal(), self._full_frame_context(source_uuid="s1"))
        await action.run(_frame_signal(), self._full_frame_context(source_uuid="s1"))

        collector.offer.assert_awaited_once()

    async def test_the_interval_is_tracked_per_source(self):
        """A doorbell camera and a litter-box camera on the same watcher must not
        share one clock — one being due must not suppress the other."""
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1000.0)

        await action.run(_frame_signal(), self._full_frame_context(source_uuid="s1"))
        await action.run(_frame_signal(), self._full_frame_context(source_uuid="s2"))

        assert collector.offer.await_count == 2

    async def test_min_interval_is_floored_not_merely_defaulted(self):
        """plan §2.4: a collector offered every analysed frame would spend the
        watcher's CPU on phashes and JPEG writes for ~95% near-duplicate data."""
        action = self._action(collector=None, collector_uuid="c1", min_interval_s=0.0)
        assert action._min_interval_s >= CollectTrainingImagesAction.MIN_INTERVAL_FLOOR_S

    async def test_offer_carries_the_full_resolution_frame_and_provenance(self):
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1.0)
        context = self._full_frame_context(source_uuid="s1", source_name="Back room")

        await action.run(_frame_signal(), context)

        offered = collector.offer.await_args.args[0]
        assert offered.provided.image is context.full_frame
        assert offered.provided.metadata.source == "Back room"
        assert offered.provided.metadata.source_uuid == "s1"
        assert offered.provided.metadata.frame_idx == 7
        assert offered.base64 == context.analysis_base64

    async def test_the_offer_declares_itself_a_live_stream(self):
        """`frame_seq` is a per-run counter that restarts at 0, and the default
        (FIXED) provider type names files `{source}_{frame_idx}.jpg` — so a
        watcher restart would overwrite the previous run's collected images one
        for one. The live schemes in `_save_image` are timestamped; declaring
        the type is what selects one."""
        from apps.helpers.imageProviders.imageProvider import ImageProviderType

        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1.0)

        await action.run(_frame_signal(), self._full_frame_context())

        assert collector.offer.await_args.args[0].provider_type is not ImageProviderType.FIXED

    async def test_a_raising_collector_fails_without_propagating(self):
        """Action.run() must never raise — a broken collector is a reported
        failure, not a reason to kill the frame loop (test_watch_actions.py's
        own module docstring states this rule for every action)."""
        collector = MagicMock()
        collector.offer = AsyncMock(side_effect=RuntimeError("disk full"))
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1.0)

        result = await action.run(_frame_signal(), self._full_frame_context())

        assert result.ok is False

    async def test_tracks_are_normalized_to_the_analysis_frame(self):
        """`Track.bbox` is pixel coords in `context.frame`'s space; `TrackedCrop`
        on the FrameOffer must be normalized so the collector can compare it
        against detections, which it always sees normalized (plan §4B)."""
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1.0)
        analysis_frame = np.zeros((100, 200, 3), dtype=np.uint8)  # height=100, width=200
        track = Track(track_id=42, bbox=(20.0, 10.0, 60.0, 40.0), confidence=0.8)
        context = self._full_frame_context(frame=analysis_frame, tracks=[track])

        await action.run(_frame_signal(), context)

        offered = collector.offer.await_args.args[0]
        assert len(offered.tracks) == 1
        crop = offered.tracks[0]
        assert crop.track_id == 42
        assert crop.x == pytest.approx(20.0 / 200)
        assert crop.y == pytest.approx(10.0 / 100)
        assert crop.width == pytest.approx(40.0 / 200)
        assert crop.height == pytest.approx(30.0 / 100)

    async def test_no_tracks_without_an_analysis_frame(self):
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1.0)
        track = Track(track_id=1, bbox=(0.0, 0.0, 10.0, 10.0), confidence=0.5)
        context = self._full_frame_context(frame=None, tracks=[track])

        await action.run(_frame_signal(), context)

        assert collector.offer.await_args.args[0].tracks == ()

    async def test_dead_track_ids_pass_through_onto_the_offer(self):
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1.0)
        context = self._full_frame_context(dead_track_ids=[7, 9])

        await action.run(_frame_signal(), context)

        assert collector.offer.await_args.args[0].dead_track_ids == (7, 9)

    async def test_a_death_between_two_throttled_offers_is_not_lost(self):
        """`dead_track_ids` is only valid for the single step it's read on, but
        offers are throttled far coarser than the tracker's own cadence — a
        death on a skipped-over frame must still reach the collector on the
        next offer that actually goes out, not vanish."""
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1000.0)

        # First call passes the interval gate (no prior offer) and reports no
        # deaths yet.
        await action.run(_frame_signal(), self._full_frame_context(source_uuid="s1", dead_track_ids=[]))
        # A track dies on a frame that the interval gate throttles away —
        # collector.offer is not called for this one.
        await action.run(_frame_signal(), self._full_frame_context(source_uuid="s1", dead_track_ids=[3]))
        collector.offer.assert_called_once()

        # Force the next call through the gate; it must still carry the death
        # from the skipped frame above.
        action._last_offered["s1"] = 0.0
        await action.run(_frame_signal(), self._full_frame_context(source_uuid="s1", dead_track_ids=[]))

        assert collector.offer.await_args.args[0].dead_track_ids == (3,)


class TestCollectTrainingImagesActionTriggers:
    """`triggers()` decides continuous vs event-scoped mode from the row's own
    configured trigger — see the module docstring's "Two modes" section."""

    def _row(self, trigger):
        return SimpleNamespace(trigger=trigger)

    def test_frame_analysed_trigger_stays_frame_only(self):
        assert CollectTrainingImagesAction.triggers(self._row("frame_analysed")) == {"frame_analysed"}

    def test_event_started_trigger_adds_frame_analysed(self):
        assert CollectTrainingImagesAction.triggers(self._row("event_started")) == {
            "event_started",
            "frame_analysed",
            "event_ended",
        }

    def test_event_ended_trigger_adds_frame_analysed(self):
        assert CollectTrainingImagesAction.triggers(self._row("event_ended")) == {
            "event_started",
            "frame_analysed",
            "event_ended",
        }


class TestCollectTrainingImagesActionEventScoped:
    """Rule-scoped collection (phase 5): 'collect training data only while
    this rule's event is open' — the capability plan §2.4 calls out as
    impossible to express by co-running the standalone task."""

    def _action(self, collector=None, **overrides):
        overrides.setdefault("trigger", "event_started")
        return CollectTrainingImagesAction(collector_for=lambda uuid: collector, **overrides)

    def _full_frame_context(self, **overrides):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        return _context(
            full_frame=frame,
            analysis_base64='{"format": "jpg", "data": "AA==", "metadata": {}}',
            frame_seq=7,
            **overrides,
        )

    async def test_a_frame_before_any_event_is_a_noop(self):
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1.0)

        result = await action.run(_frame_signal(source_uuid="s1"), self._full_frame_context(source_uuid="s1"))

        assert result.ok is True
        collector.offer.assert_not_called()

    async def test_a_frame_while_the_event_is_open_is_offered(self):
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1.0)

        await action.run(_signal(phase=EventPhase.START, source_uuid="s1"), _context(source_uuid="s1"))
        result = await action.run(_frame_signal(source_uuid="s1"), self._full_frame_context(source_uuid="s1"))

        assert result.ok is True
        collector.offer.assert_awaited_once()

    async def test_a_frame_after_the_event_ends_is_a_noop_again(self):
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1.0)

        await action.run(_signal(phase=EventPhase.START, source_uuid="s1"), _context(source_uuid="s1"))
        await action.run(_signal(phase=EventPhase.END, source_uuid="s1"), _context(source_uuid="s1"))
        await action.run(_frame_signal(source_uuid="s1"), self._full_frame_context(source_uuid="s1"))

        collector.offer.assert_not_called()

    async def test_a_track_dying_after_the_event_closed_is_still_discharged(self):
        """`context.dead_track_ids` is valid for exactly one frame, and out of
        session there will be no offer to carry it. The collector holds a whole
        full-resolution frame per track in `_track_worst`, so a death observed
        and then dropped strands that frame for the life of the watcher — days,
        on a watcher that never opens another event."""
        collector = MagicMock()
        collector.offer = AsyncMock()
        collector.commit_dead_tracks = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1.0)

        await action.run(_signal(phase=EventPhase.START, source_uuid="s1"), _context(source_uuid="s1"))
        await action.run(_signal(phase=EventPhase.END, source_uuid="s1"), _context(source_uuid="s1"))
        await action.run(
            _frame_signal(source_uuid="s1"), self._full_frame_context(source_uuid="s1", dead_track_ids=[4])
        )

        collector.offer.assert_not_called()
        collector.commit_dead_tracks.assert_awaited_once_with((4,))

    async def test_held_gaps_are_dropped_when_the_event_closes(self):
        """A gap can only be extended or resolved while frames are in scope, and
        each one pins the full-resolution frame it is holding."""
        collector = MagicMock()
        collector.offer = AsyncMock()
        collector.commit_dead_tracks = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1.0)
        action._track_gaps["s1"] = {4: object()}

        await action.run(_frame_signal(source_uuid="s1"), self._full_frame_context(source_uuid="s1"))

        assert "s1" not in action._track_gaps

    async def test_two_overlapping_rules_on_one_source_need_both_to_end(self):
        """A refcount, not a boolean — two in-scope rules opening near-simultaneously
        on one camera must not let the first one's END switch collection off
        while the second is still open."""
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1.0)

        await action.run(
            _signal(phase=EventPhase.START, rule_uuid="rule-a", source_uuid="s1"), _context(source_uuid="s1")
        )
        await action.run(
            _signal(phase=EventPhase.START, rule_uuid="rule-b", source_uuid="s1"), _context(source_uuid="s1")
        )
        await action.run(
            _signal(phase=EventPhase.END, rule_uuid="rule-a", source_uuid="s1"), _context(source_uuid="s1")
        )

        await action.run(_frame_signal(source_uuid="s1"), self._full_frame_context(source_uuid="s1"))
        collector.offer.assert_awaited_once()  # rule-b is still open

        await action.run(
            _signal(phase=EventPhase.END, rule_uuid="rule-b", source_uuid="s1"), _context(source_uuid="s1")
        )
        await action.run(_frame_signal(source_uuid="s1"), self._full_frame_context(source_uuid="s1"))
        collector.offer.assert_awaited_once()  # still just the one — both rules now closed

    async def test_sessions_are_tracked_per_source(self):
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1.0)

        await action.run(_signal(phase=EventPhase.START, source_uuid="s1"), _context(source_uuid="s1"))
        result = await action.run(_frame_signal(source_uuid="s2"), self._full_frame_context(source_uuid="s2"))

        assert result.ok is True
        collector.offer.assert_not_called()  # s2 has no open session of its own

    async def test_provenance_is_stamped_while_a_session_is_open(self):
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=1.0)

        await action.run(
            _signal(phase=EventPhase.START, rule_uuid="rule-a", started_at=123.0, source_uuid="s1"),
            _context(source_uuid="s1"),
        )
        await action.run(_frame_signal(source_uuid="s1"), self._full_frame_context(source_uuid="s1"))

        offered = collector.offer.await_args.args[0]
        assert offered.provided.metadata.additional_info == {
            "watch_rule_uuid": "rule-a",
            "watch_event_started_at": "123.0",
        }

    async def test_a_closed_rule_stops_claiming_frames_of_a_still_open_one(self):
        """Ending rule-b must hand provenance back to rule-a, which is still open.

        Provenance used to be a single last-opened-wins slot that END only
        decremented a count against, never restored — so every frame for the
        rest of rule-a's session was stamped with rule-b's uuid and rule-b's
        start time, pointing at an event that had already closed.
        """
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=0.0)

        await action.run(
            _signal(phase=EventPhase.START, rule_uuid="rule-a", started_at=100.0, source_uuid="s1"),
            _context(source_uuid="s1"),
        )
        await action.run(
            _signal(phase=EventPhase.START, rule_uuid="rule-b", started_at=200.0, source_uuid="s1"),
            _context(source_uuid="s1"),
        )
        await action.run(
            _signal(phase=EventPhase.END, rule_uuid="rule-b", source_uuid="s1"), _context(source_uuid="s1")
        )
        await action.run(_frame_signal(source_uuid="s1"), self._full_frame_context(source_uuid="s1"))

        offered = collector.offer.await_args.args[0]
        assert offered.provided.metadata.additional_info == {
            "watch_rule_uuid": "rule-a",
            "watch_event_started_at": "100.0",
        }

    async def test_an_orphan_end_does_not_close_an_unrelated_open_session(self):
        """An END for a rule this action never saw START for (e.g. its
        in-memory state was reset by a restart after that START had already
        landed) must not be treated as 'close whatever is on top of the
        stack' — that would end a genuinely-still-open, unrelated rule's
        session early."""
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=0.0)

        await action.run(
            _signal(phase=EventPhase.START, rule_uuid="rule-a", started_at=100.0, source_uuid="s1"),
            _context(source_uuid="s1"),
        )
        # An END for a rule that never opened a session here.
        await action.run(
            _signal(phase=EventPhase.END, rule_uuid="rule-unrelated", source_uuid="s1"), _context(source_uuid="s1")
        )
        await action.run(_frame_signal(source_uuid="s1"), self._full_frame_context(source_uuid="s1"))

        collector.offer.assert_awaited_once()  # rule-a's session must still be open
        offered = collector.offer.await_args.args[0]
        assert offered.provided.metadata.additional_info["watch_rule_uuid"] == "rule-a"

    async def test_the_newest_open_rule_wins_while_two_overlap(self):
        """The flip side of the above: while both are genuinely open, the
        most-recently-opened rule supplies provenance."""
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = self._action(collector=collector, collector_uuid="c1", min_interval_s=0.0)

        await action.run(
            _signal(phase=EventPhase.START, rule_uuid="rule-a", started_at=100.0, source_uuid="s1"),
            _context(source_uuid="s1"),
        )
        await action.run(
            _signal(phase=EventPhase.START, rule_uuid="rule-b", started_at=200.0, source_uuid="s1"),
            _context(source_uuid="s1"),
        )
        await action.run(_frame_signal(source_uuid="s1"), self._full_frame_context(source_uuid="s1"))

        offered = collector.offer.await_args.args[0]
        assert offered.provided.metadata.additional_info["watch_rule_uuid"] == "rule-b"

    async def test_continuous_mode_carries_no_provenance(self):
        """Regression guard: continuous mode (phase 4) has no session to
        describe, and must not stamp stale/empty provenance."""
        collector = MagicMock()
        collector.offer = AsyncMock()
        action = CollectTrainingImagesAction(
            collector_for=lambda uuid: collector, collector_uuid="c1", min_interval_s=1.0, trigger="frame_analysed"
        )

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        await action.run(
            _frame_signal(source_uuid="s1"),
            _context(
                source_uuid="s1",
                full_frame=frame,
                analysis_base64='{"format": "jpg", "data": "AA==", "metadata": {}}',
            ),
        )

        offered = collector.offer.await_args.args[0]
        assert offered.provided.metadata.additional_info == {}


def _detector_miss_collector(target_label="cat"):
    collector = MagicMock()
    collector.spec.enable_detector_miss = True
    collector.spec.target_label_bands = {target_label: object()}
    collector.offer_detector_miss = AsyncMock()
    return collector


def _track(track_id, bbox, label_uuid="cat", matched=True, confirmed=True):
    return Track(track_id=track_id, bbox=bbox, label_uuid=label_uuid, matched_this_update=matched, confirmed=confirmed)


class TestDetectorMissGapTracking:
    """Plan §4A: a confirmed track the detector missed for a frame or two is a
    candidate for collection, if it re-acquires within a few frames. This
    exercises `CollectTrainingImagesAction._process_track_gaps` directly —
    the state machine that opens/extends/resolves/abandons a gap per track,
    independent of the periodic offer's own interval gate.
    """

    _FRAME = np.zeros((100, 100, 3), dtype=np.uint8)  # 100x100, so pixel/100 = normalized

    def _action(self):
        return CollectTrainingImagesAction(collector_for=lambda uuid: None, collector_uuid="c1")

    def _ctx(self, tracks, dead_track_ids=()):
        return _context(source_uuid="s1", frame=self._FRAME, tracks=list(tracks), dead_track_ids=list(dead_track_ids))

    async def test_a_coasting_track_opens_a_gap_without_offering(self):
        action = self._action()
        collector = _detector_miss_collector()
        track = _track(1, (10, 10, 30, 30), matched=False)

        await action._process_track_gaps("s1", collector, self._ctx([track]))

        assert 1 in action._track_gaps["s1"]
        collector.offer_detector_miss.assert_not_called()

    async def test_reacquisition_commits_an_interpolated_box(self):
        action = self._action()
        collector = _detector_miss_collector()
        before = _track(1, (10, 10, 30, 30), matched=False)
        after = _track(1, (20, 20, 40, 40), matched=True)

        await action._process_track_gaps("s1", collector, self._ctx([before]))
        await action._process_track_gaps("s1", collector, self._ctx([after]))

        collector.offer_detector_miss.assert_awaited_once()
        frame, det = collector.offer_detector_miss.await_args.args
        assert det.confidence is None
        assert det.bbox.label_uuid == "cat"
        assert DETECTOR_MISS_TAG_UUID in det.bbox.tag_uuids
        # frame_count=1 (one coasting step) -> fraction 1/2 -> midpoint.
        assert det.bbox.x == pytest.approx(0.15)
        assert det.bbox.y == pytest.approx(0.15)
        assert det.bbox.width == pytest.approx(0.2)
        assert det.bbox.height == pytest.approx(0.2)
        # The gap is fully forgotten once resolved.
        assert "s1" not in action._track_gaps or 1 not in action._track_gaps["s1"]

    async def test_a_gap_past_the_frame_cap_is_abandoned(self):
        action = self._action()
        collector = _detector_miss_collector()
        coasting = _track(1, (10, 10, 30, 30), matched=False)

        # One more coasting frame than MAX_GAP_FRAMES tolerates.
        for _ in range(action.MAX_GAP_FRAMES + 1):
            await action._process_track_gaps("s1", collector, self._ctx([coasting]))
        await action._process_track_gaps("s1", collector, self._ctx([_track(1, (20, 20, 40, 40), matched=True)]))

        collector.offer_detector_miss.assert_not_called()

    async def test_track_death_drops_the_gap_without_committing(self):
        action = self._action()
        collector = _detector_miss_collector()
        coasting = _track(1, (10, 10, 30, 30), matched=False)

        await action._process_track_gaps("s1", collector, self._ctx([coasting]))
        # The track is gone next frame — it left, it didn't blink.
        await action._process_track_gaps("s1", collector, self._ctx([], dead_track_ids=[1]))

        collector.offer_detector_miss.assert_not_called()
        assert "s1" not in action._track_gaps or 1 not in action._track_gaps.get("s1", {})

    async def test_a_track_outside_the_collectors_target_labels_is_ignored(self):
        action = self._action()
        collector = _detector_miss_collector(target_label="dog")
        coasting = _track(1, (10, 10, 30, 30), label_uuid="cat", matched=False)

        await action._process_track_gaps("s1", collector, self._ctx([coasting]))

        assert not action._track_gaps.get("s1")

    async def test_an_unconfirmed_track_never_opens_a_gap(self):
        """Plan §4A calls for a CONFIRMED track's gaps specifically — a
        still-provisional, single-hit track is exactly as likely to be tracker
        noise as a real object, and must not seed a 'detector miss' box."""
        action = self._action()
        collector = _detector_miss_collector()
        coasting = _track(1, (10, 10, 30, 30), matched=False, confirmed=False)

        await action._process_track_gaps("s1", collector, self._ctx([coasting]))

        assert not action._track_gaps.get("s1")
        collector.offer_detector_miss.assert_not_called()

    async def test_track_ids_are_scoped_per_source(self):
        """Two cameras' trackers both mint track_id=1 independently — a gap on
        one source must not resolve or interfere with the other's."""
        action = self._action()
        collector = _detector_miss_collector()
        coasting = _track(1, (10, 10, 30, 30), matched=False)

        await action._process_track_gaps("cam-a", collector, self._ctx([coasting]))

        assert 1 in action._track_gaps["cam-a"]
        assert "cam-b" not in action._track_gaps


class TestAssociationAmbiguity:
    """Plan §5's 'cheap and nearly-sufficient first cut': another live track
    within ~2 box-widths of the coasting one's centre."""

    def test_no_other_tracks_is_unambiguous(self):
        action = CollectTrainingImagesAction(collector_for=lambda uuid: None, collector_uuid="c1")
        crop = TrackedCrop(track_id=1, x=0.1, y=0.1, width=0.1, height=0.1)
        assert action._is_association_ambiguous(crop, [crop]) is False

    def test_a_nearby_track_is_ambiguous(self):
        action = CollectTrainingImagesAction(collector_for=lambda uuid: None, collector_uuid="c1")
        a = TrackedCrop(track_id=1, x=0.1, y=0.1, width=0.1, height=0.1)
        b = TrackedCrop(track_id=2, x=0.15, y=0.1, width=0.1, height=0.1)  # centres 0.05 apart, threshold 0.2
        assert action._is_association_ambiguous(a, [a, b]) is True

    def test_a_distant_track_is_unambiguous(self):
        action = CollectTrainingImagesAction(collector_for=lambda uuid: None, collector_uuid="c1")
        a = TrackedCrop(track_id=1, x=0.1, y=0.1, width=0.1, height=0.1)
        b = TrackedCrop(track_id=2, x=0.9, y=0.9, width=0.1, height=0.1)
        assert action._is_association_ambiguous(a, [a, b]) is False

    async def test_ambiguity_rides_onto_the_held_frames_provenance(self):
        action = CollectTrainingImagesAction(collector_for=lambda uuid: None, collector_uuid="c1")
        collector = _detector_miss_collector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        near = _track(1, (10, 10, 30, 30), matched=False)
        other = _track(2, (12, 10, 32, 30), matched=True)  # nearly on top of track 1

        ctx = _context(source_uuid="s1", frame=frame, tracks=[near, other])
        await action._process_track_gaps("s1", collector, ctx)

        held = action._track_gaps["s1"][1].held_frame
        assert held.provided.metadata.additional_info["detector_miss_ambiguous"] is True


class TestCollectorSendWidth:
    """Which pixels the collector's own inference call gets (plan §2.3b).

    The saved image is always the full-res frame — that is `_save_image`'s
    contract and is never in tension with this. Only the INFERENCE input is at
    stake here: at or above the model's input width, the watcher's analysis frame
    already gives the same boxes a full-res pass would, so reusing it is free.
    """

    def _action(self, collector, **overrides):
        return CollectTrainingImagesAction(collector_for=lambda uuid: collector, collector_uuid="c1", **overrides)

    def _context(self, native_width=1920, analysis_width=640):
        return _context(
            full_frame=np.zeros((native_width // 2, native_width, 3), dtype=np.uint8),
            frame=np.zeros((analysis_width // 2, analysis_width, 3), dtype=np.uint8),
            analysis_base64='{"format": "jpg", "data": "AA==", "metadata": {}}',
            frame_seq=7,
        )

    @staticmethod
    def _collector(send_width):
        collector = MagicMock()
        collector.offer = AsyncMock()
        collector.spec.enable_detector_miss = False
        collector.send_width = lambda native: send_width
        return collector

    async def test_a_640_model_reuses_the_watchers_own_analysis_frame(self):
        """The common case, and the whole reason sharing is free: the watcher's
        640px frame IS min(native, model_input), so a re-encode would recompute
        boxes it already has at full cost."""
        context = self._context()
        collector = self._collector(640)

        await self._action(collector).run(_frame_signal(), context)

        offer = collector.offer.await_args.args[0]
        assert offer.base64 == context.analysis_base64

    async def test_a_wider_model_gets_a_re_encoded_frame(self):
        context = self._context(native_width=1920, analysis_width=640)
        collector = self._collector(1280)

        await self._action(collector).run(_frame_signal(), context)

        offer = collector.offer.await_args.args[0]
        assert offer.base64 != context.analysis_base64
        packet = json.loads(offer.base64)
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(packet["data"]), np.uint8), cv2.IMREAD_COLOR)
        # min(native=1920, model_input=1280) — not the native width, and not 640.
        assert decoded.shape[1] == 1280

    async def test_the_saved_image_stays_full_resolution_either_way(self):
        context = self._context(native_width=1920, analysis_width=640)
        collector = self._collector(1280)

        await self._action(collector).run(_frame_signal(), context)

        offer = collector.offer.await_args.args[0]
        assert offer.provided.image.shape[1] == 1920

    async def test_an_unknown_model_width_changes_nothing(self):
        context = self._context()
        collector = self._collector(None)

        await self._action(collector).run(_frame_signal(), context)

        assert collector.offer.await_args.args[0].base64 == context.analysis_base64
