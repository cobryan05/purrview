"""RingMqttImageProvider's two-tier behaviour, and the reconnect race underneath it.

A Ring camera is asleep almost all of the time, so this provider idles on ring-mqtt's
free snapshot topic and only wakes the camera — an operation with a real battery and
cloud cost — when something asks for video. The tests here are about *when* it decides
to do that and when it decides to stop, because getting either wrong is expensive in a
way that no assertion about the frames themselves would catch.

No broker and no camera are contacted: `MqttClient` connects inside its own
constructor, and the live tier's `FfmpegImageProvider` spawns ffmpeg, so both are
replaced with fakes that record what they were asked to do.
"""

import asyncio
import io

import numpy as np
import pytest
from PIL import Image

from apps.helpers.imageProviders import ringMqttImageProvider as ring_module
from apps.helpers.imageProviders.imageProvider import ImageWithProviderMetadata
from apps.helpers.imageProviders.ringMqttImageProvider import RingMqttImageProvider
from apps.helpers.mqttClient import MqttClient

LOCATION = "loc-uuid"
CAMERA = "abc123"
BASE = f"ring/{LOCATION}/camera/{CAMERA}"


class FakeMqttClient:
    """Stand-in for MqttClient that never touches a network."""

    def __init__(self, broker_address, prefix="myhome/", broker_port=1883):
        self.callbacks: dict[str, object] = {}
        self.published: list[tuple[str, str]] = []
        self.unsubscribed: list[str] = []
        self.disconnected = False

    def subscribe(self, topic, callback, absoluteTopic=False):
        self.callbacks[topic] = callback

    def unsubscribe(self, topic):
        self.unsubscribed.append(topic)
        self.callbacks.pop(topic, None)

    def publish(self, topic, value, retain=False, absoluteTopic=False):
        self.published.append((topic, value))

    def disconnect(self):
        self.disconnected = True

    def deliver(self, topic: str, payload: bytes) -> None:
        self.callbacks[topic](FakeMessage(payload))


class FakeMessage:
    def __init__(self, payload: bytes):
        self.payload = payload


class FakeLiveProvider:
    """Stand-in for the inner FfmpegImageProvider, which would spawn ffmpeg."""

    instances: list["FakeLiveProvider"] = []
    fail_on_start = False

    def __init__(self, url, loop=True, rtsp_relay=True):
        self.url = url
        self.started = False
        self.stopped = False
        FakeLiveProvider.instances.append(self)

    def set_source_identity(self, name):
        self.name = name

    def set_source_fps_ceiling(self, fps):
        self.fps = fps

    def is_connecting(self):
        return False

    async def start(self):
        if FakeLiveProvider.fail_on_start:
            raise RuntimeError("relay refused")
        self.started = True

    async def stop(self):
        self.stopped = True

    async def getNextImage(self):
        return ImageWithProviderMetadata(np.zeros((8, 8, 3), dtype=np.uint8))


def _jpeg_bytes(rgb: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((4, 4, 3), rgb, dtype=np.uint8)).save(buf, format="JPEG", quality=100)
    return buf.getvalue()


@pytest.fixture
def fakes(monkeypatch):
    made: list[FakeMqttClient] = []
    FakeLiveProvider.instances = []
    FakeLiveProvider.fail_on_start = False

    def factory(*args, **kwargs):
        client = FakeMqttClient(*args, **kwargs)
        made.append(client)
        return client

    monkeypatch.setattr(ring_module, "MqttClient", factory)
    monkeypatch.setattr(ring_module, "FfmpegImageProvider", FakeLiveProvider)
    return made


async def _started_provider(**kwargs) -> tuple[RingMqttImageProvider, FakeMqttClient]:
    provider = RingMqttImageProvider(
        broker_address="broker.invalid",
        location_id=LOCATION,
        camera_id=CAMERA,
        **kwargs,
    )
    await provider.start()
    return provider, provider._mqtt_client


class TestEscalationTriggers:
    """What is allowed to wake the camera."""

    async def test_motion_escalates_only_when_enabled(self, fakes):
        """`live_on_motion` is the whole point of the setting: with it off, a motion
        event must not wake the camera, however loudly ring-mqtt reports it."""
        for enabled in (True, False):
            provider, client = await _started_provider(live_on_motion=enabled)
            client.deliver(f"{BASE}/motion/state", b"ON")
            assert provider._escalate_requested is enabled

    async def test_motion_off_does_not_escalate(self, fakes):
        """Motion clearing is not a reason to start streaming — only motion starting is."""
        provider, client = await _started_provider(live_on_motion=True)
        client.deliver(f"{BASE}/motion/state", b"OFF")
        assert provider._escalate_requested is False

    async def test_a_deliberate_stream_request_escalates(self, fakes):
        """A stream someone else asked for (the UI button, Home Assistant, the Ring
        app) is joined rather than ignored — this is the external-trigger path, and it
        is the reason the trigger is a broker message rather than an RPC."""
        provider, client = await _started_provider(live_on_motion=False)
        client.deliver(f"{BASE}/stream/command", b"on")
        assert provider._escalate_requested is True

    async def test_on_demand_churn_does_not_escalate(self, fakes):
        """The whole point of keying on the command topic rather than the state topic.

        ring-mqtt's RTSP server cycles `stream/state` OFF→ON constantly on its own —
        measured at roughly every 20 seconds per camera on a live system, with no MQTT
        command involved. A provider that treated that as demand would hold the camera
        awake and run inference on full-rate 1080p forever without anyone asking.
        """
        provider, client = await _started_provider(live_on_motion=False)
        for _ in range(5):
            client.deliver(f"{BASE}/stream/state", b"OFF")
            client.deliver(f"{BASE}/stream/state", b"ON")
        assert provider._escalate_requested is False

    async def test_a_trigger_interrupts_an_idle_wait(self, fakes):
        """ "Go live" must not wait out the cloud snapshot cadence.

        An idle camera spends nearly all its time blocked inside `getNextImage`
        waiting on the snapshot queue, and ring-mqtt publishes one every 30-90 s. A
        trigger that was only read at the top of that call therefore did nothing until
        the snapshot it was meant to replace had already arrived.
        """
        provider, client = await _started_provider(live_on_motion=False)
        pending = asyncio.ensure_future(provider.getNextImage())
        # Let it reach the idle wait with an empty queue — no snapshot is ever delivered.
        for _ in range(3):
            await asyncio.sleep(0)
        assert provider._live_provider is None

        await provider.request_live()
        frame = await asyncio.wait_for(pending, timeout=2.0)

        assert provider._live_provider is not None
        assert frame.image.shape == (8, 8, 3)

    async def test_a_broker_thread_trigger_interrupts_an_idle_wait(self, fakes):
        """Same, for the triggers that arrive on paho's thread rather than the loop."""
        provider, client = await _started_provider(live_on_motion=True)
        pending = asyncio.ensure_future(provider.getNextImage())
        for _ in range(3):
            await asyncio.sleep(0)

        client.deliver(f"{BASE}/motion/state", b"ON")
        frame = await asyncio.wait_for(pending, timeout=2.0)

        assert provider._live_provider is not None
        assert frame.image.shape == (8, 8, 3)

    async def test_someone_elses_off_does_not_end_our_live_window(self, fakes):
        """One client's `off` must not cut short a stream another is mid-way through
        reading. The hold timer is what ends our window."""
        provider, client = await _started_provider(live_on_motion=False)
        client.deliver(f"{BASE}/stream/command", b"on")
        await provider.getNextImage()
        assert provider._live_provider is not None

        client.deliver(f"{BASE}/stream/command", b"off")

        assert provider._live_provider is not None


class TestTierTransitions:
    async def test_live_window_expiry_returns_to_snapshots(self, fakes):
        """The camera must go back to sleep on its own once nothing re-triggers it."""
        provider, client = await _started_provider(live_hold_s=10.0)
        now = [1000.0]
        provider._now = lambda: now[0]

        await provider.request_live()
        await provider.getNextImage()
        assert provider._live_provider is not None

        # Past the hold window with no further trigger.
        now[0] += 11.0
        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((1, 2, 3)))
        await provider.getNextImage()

        assert provider._live_provider is None
        assert FakeLiveProvider.instances[0].stopped is True

    async def test_motion_outlasting_the_hold_is_not_cut_short(self, fakes):
        """Ring decides how long an event lasts, not us.

        `motion/state` stays ON for the bridge's `motion_duration` — 180s by default,
        against a `live_hold_s` default of 120s. Expiring on our own timer would end
        the capture a minute before Ring considers the event over, losing the tail of
        every single motion event.
        """
        provider, client = await _started_provider(live_hold_s=10.0, live_on_motion=True)
        now = [1000.0]
        provider._now = lambda: now[0]

        # A snapshot is kept queued before every read. Without one, a regression that
        # de-escalates early would leave getNextImage() blocked on the real 300s
        # IDLE_FRAME_TIMEOUT (which `_now` does not fake) and the test would fail five
        # minutes later on a timeout instead of immediately on the assertion below.
        def read():
            client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((7, 8, 9)))
            return provider.getNextImage()

        client.deliver(f"{BASE}/motion/state", b"ON")
        await read()
        assert provider._live_provider is not None

        # Well past live_hold_s, but Ring still reports the event as running.
        now[0] += 100.0
        await read()
        assert provider._live_provider is not None, "must stay live while motion is active"

        # Motion clears: the grace period starts now, not when the event began.
        client.deliver(f"{BASE}/motion/state", b"OFF")
        now[0] += 5.0
        await read()
        assert provider._live_provider is not None, "grace period must run from the clear"

        now[0] += 11.0
        await read()
        assert provider._live_provider is None

    async def test_motion_reasserting_during_the_grace_keeps_the_same_stream(self, fakes):
        """A second event inside the grace period must not cost a reconnect.

        Ring re-publishes `ON` for fresh motion inside an open window, and events
        cluster (someone walking back past the camera). Tearing the stream down and
        rebuilding it would cost ~8.5s to first frame, and — worse — briefly let the
        camera sleep, so the reconnect is not merely slow but drops the very footage
        the re-trigger was about.
        """
        provider, client = await _started_provider(live_hold_s=10.0, live_on_motion=True)
        now = [1000.0]
        provider._now = lambda: now[0]

        def read():
            client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((7, 8, 9)))
            return provider.getNextImage()

        client.deliver(f"{BASE}/motion/state", b"ON")
        await read()
        first = provider._live_provider
        assert first is not None

        # Motion clears; part-way through the grace, a new event arrives.
        client.deliver(f"{BASE}/motion/state", b"OFF")
        now[0] += 5.0
        await read()
        client.deliver(f"{BASE}/motion/state", b"ON")
        now[0] += 2.0
        await read()

        assert provider._live_provider is first, "must be the same reader, not a rebuilt one"
        assert first.stopped is False, "the stream must never have been torn down"
        assert len(FakeLiveProvider.instances) == 1, "no second reader may be constructed"

        # And the grace now runs from the *new* clear, not the original one.
        client.deliver(f"{BASE}/motion/state", b"OFF")
        now[0] += 9.0
        await read()
        assert provider._live_provider is first, "grace must restart from the later clear"
        now[0] += 2.0
        await read()
        assert provider._live_provider is None

    async def test_failed_live_start_falls_back_to_snapshots(self, fakes):
        """A camera that will not stream is not a dead source — the free snapshot tier
        is still perfectly good, and reporting failure would take the whole source
        down over an optional upgrade."""
        FakeLiveProvider.fail_on_start = True
        provider, client = await _started_provider()

        await provider.request_live()
        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((4, 5, 6)))
        image = await provider.getNextImage()

        assert provider._live_provider is None
        assert image.metadata.additional_info["tier"] == "snapshot"


class TestFrameExpectations:
    """`next_frame_expected_by` — what stops a healthy idle camera being called failing.

    A Ring source publishes a cloud snapshot every ~30s while idle. A consumer that
    times out a fetch at 30s marks it *failing*, not merely stale, which is why an
    interval derived from a rate limit is not good enough: it has to reflect the tier
    the provider is actually in.
    """

    async def test_idle_expectation_covers_the_measured_snapshot_cadence(self, fakes):
        provider, client = await _started_provider()
        now = [1000.0]
        provider._now = lambda: now[0]

        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((1, 2, 3)))
        now[0] += 30.0
        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((1, 2, 3)))

        horizon = provider.next_frame_expected_by() - now[0]
        assert horizon >= 30.0, "must give at least one whole cadence before complaining"

    async def test_a_motion_snapshot_does_not_shrink_the_learned_cadence(self, fakes):
        """Motion injects an out-of-band snapshot seconds after the interval one.

        Learning from the newest gap would teach a ~2s cadence and then report a
        perfectly healthy idle camera as overdue on every read.
        """
        provider, client = await _started_provider()
        now = [1000.0]
        provider._now = lambda: now[0]

        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((1, 2, 3)))
        now[0] += 30.0
        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((1, 2, 3)))
        now[0] += 2.0  # a motion-triggered snapshot lands right behind an interval one
        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((1, 2, 3)))

        assert provider._snapshot_cadence_s == 30.0
        assert provider.next_frame_expected_by() - now[0] >= 30.0

    async def test_live_tier_expects_frames_promptly(self, fakes):
        """The point of a deadline over an interval: the same source, a different answer."""
        provider, client = await _started_provider()
        now = [1000.0]
        provider._now = lambda: now[0]
        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((1, 2, 3)))
        now[0] += 30.0
        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((1, 2, 3)))
        idle_horizon = provider.next_frame_expected_by() - now[0]

        client.deliver(f"{BASE}/motion/state", b"ON")
        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((1, 2, 3)))
        await provider.getNextImage()
        live_horizon = provider.next_frame_expected_by() - now[0]

        assert provider._live_provider is not None
        assert live_horizon < idle_horizon, "a live stream must not inherit the idle tier's patience"

    async def test_never_returns_a_deadline_in_the_past(self, fakes):
        """A consumer derives a wait budget from this; a past deadline means zero,
        which would time out instantly on a source that is merely overdue."""
        provider, client = await _started_provider()
        now = [1000.0]
        provider._now = lambda: now[0]
        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((1, 2, 3)))
        now[0] += 30.0
        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((1, 2, 3)))

        now[0] += 10_000.0  # long overdue
        assert provider.next_frame_expected_by() > now[0]

    async def test_min_capture_interval_actually_paces(self, fakes):
        """It is advertised in the source editor, so it has to do something.

        Without this the live tier hands a collector ~20fps of 1080p regardless of
        what it asked for.
        """
        provider, client = await _started_provider(min_capture_interval=5.0)
        now = [1000.0]
        slept: list[float] = []
        provider._now = lambda: now[0]

        async def fake_sleep(d):
            slept.append(d)
            now[0] += d

        provider._sleep = fake_sleep

        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((1, 2, 3)))
        await provider.getNextImage()
        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((4, 5, 6)))
        await provider.getNextImage()

        assert slept and slept[-1] == pytest.approx(5.0), f"expected a 5s pacing sleep, got {slept}"


class TestSleepCourtesy:
    """We only ever put back to sleep what we ourselves woke."""

    async def test_publishes_off_for_a_stream_we_started(self, fakes):
        provider, client = await _started_provider()
        await provider.request_live()
        await provider.getNextImage()
        client.published.clear()

        await provider.release_live()

        assert (f"{BASE}/stream/command", "off") in client.published

    async def test_does_not_publish_off_for_someone_elses_stream(self, fakes):
        """Frigate or Home Assistant may be holding this camera up for its own
        reasons. Releasing our interest must not cancel a stream we never asked for."""
        provider, client = await _started_provider(live_on_motion=False)
        client.deliver(f"{BASE}/stream/state", b"ON")
        client.published.clear()

        await provider.release_live()

        assert client.published == []

    async def test_does_not_cut_the_live_tier_off_mid_event(self, fakes):
        """Closing the live view while motion is running must not end the recording.

        The UI releases the camera when the viewer closes it, which can land in the
        middle of a motion event. Tearing the tier down there drops the rest of the
        event back to 30-90s cloud snapshots — the failure the `_motion_active` gate on
        the expiry path exists to prevent. The hold is still dropped, so the tier comes
        down through that path as soon as motion clears.
        """
        provider, client = await _started_provider(live_on_motion=True)
        client.deliver(f"{BASE}/motion/state", b"ON")
        await provider.getNextImage()
        assert provider._live_provider is not None, "expected motion to have escalated"
        client.published.clear()

        await provider.release_live()

        assert provider._live_provider is not None, "the live tier was cut off mid-event"
        assert (f"{BASE}/stream/command", "off") not in client.published

        # And it does come down, once ring-mqtt says the event is over.
        client.deliver(f"{BASE}/motion/state", b"OFF")
        provider._live_until = 0.0
        client.deliver(f"{BASE}/snapshot/image", _jpeg_bytes((1, 2, 3)))
        await provider.getNextImage()
        assert provider._live_provider is None, "the live tier outlived the event"


class TestSnapshotFrames:
    async def test_snapshot_is_bgr_and_keeps_the_original_jpeg(self, fakes):
        """Channel order is silent when wrong — nothing downstream can tell, and it
        shows up only as worse detections. The published JPEG is carried through
        rather than re-encoded so a clip built from snapshots stays full quality.
        """
        provider, client = await _started_provider()
        payload = _jpeg_bytes((255, 0, 0))  # pure red in RGB
        client.deliver(f"{BASE}/snapshot/image", payload)

        image = await provider.getNextImage()

        blue, green, red = image.image[0, 0]
        assert red > 200 and blue < 60, "red pixel must land in the red channel of a BGR frame"
        assert image.encoded_b64 is not None


class TestReconnectSubscriptionRace:
    """`mqtt_connected_callback` runs on paho's thread while the owner is still
    subscribing from its own.

    `connect()` returns as soon as the socket is up, so a caller that constructs a
    client and immediately subscribes races the CONNACK. Iterating the live dict raised
    part-way through, which killed the connack handler — so every subscription it had
    not yet reached was silently never registered, and that topic stayed quiet for the
    life of the client. It looked like a source that simply never published.
    """

    def test_resubscribes_everything_when_a_topic_is_added_mid_reconnect(self):
        client = object.__new__(MqttClient)
        client._prefix = ""
        client._subMap = {
            f"{BASE}/{suffix}": MqttClient.SubMapData(callback=lambda m: None)
            for suffix in ("snapshot/image", "motion/state", "stream/state")
        }

        resubscribed: list[str] = []

        class RacingPaho:
            def disconnect(self):
                """Only for MqttClient.__del__, which runs when the test's client is
                collected and would otherwise raise into an unraisable-exception warning."""

            def subscribe(self, topic):
                resubscribed.append(topic)
                # Another thread getting a subscription in while the handler runs.
                if len(resubscribed) == 1:
                    client._subMap[f"{BASE}/ding/state"] = MqttClient.SubMapData(callback=lambda m: None)

        client._mqtt = RacingPaho()

        client.mqtt_connected_callback(client._mqtt, None, {}, 0)

        assert len(resubscribed) == 3, "connack handler must resubscribe every known topic"


class TestBufferFactsFollowTheTier:
    """What the watcher reads to decide where a clip is cut from.

    These have to be live reads, not values sampled once: a camera is idle when the
    watcher starts, so a single sample describes it as unbuffered for the whole run and
    every clip is cut from the frame ring at analysis fps while MediaMTX records the
    real stream and deletes it unread.
    """

    async def test_an_idle_camera_reports_no_buffer(self, fakes):
        provider, _client = await _started_provider()

        assert provider.buffered is False
        assert provider.retention_s == 0.0
        assert provider.relay_path == ""

    async def test_an_escalated_camera_reports_the_relays_answer(self, fakes):
        provider, _client = await _started_provider(live_hold_s=10.0)
        await provider.request_live()
        await provider.getNextImage()
        live = FakeLiveProvider.instances[0]
        live.buffered, live.retention_s, live.relay_path = True, 45.0, "relay_master_abc"

        assert provider.buffered is True
        assert provider.retention_s == 45.0
        assert provider.relay_path == "relay_master_abc"

    async def test_the_tier_name_stays_ring_across_an_escalation(self, fakes):
        """The buffer badge keys its "snapshots, low fps" wording off this, and the tier
        a Ring camera spends almost all its time in is the idle one.
        """
        provider, _client = await _started_provider(live_hold_s=10.0)
        assert provider.ingest_tier_name == "ring"

        await provider.request_live()
        await provider.getNextImage()

        assert provider.ingest_tier_name == "ring"

    async def test_holding_the_window_open_publishes_no_command(self, fakes):
        """`hold_live` is re-stated every frame. `stream/command` is the one topic that
        separates a deliberate request for video from the RTSP server's own churn, so
        the hold must not touch it.
        """
        provider, client = await _started_provider(live_hold_s=10.0)
        now = [1000.0]
        provider._now = lambda: now[0]
        await provider.request_live()
        await provider.getNextImage()
        published_before = len(client.published)

        now[0] += 5.0
        provider.hold_live()

        assert len(client.published) == published_before
        assert provider._live_until == 1015.0
