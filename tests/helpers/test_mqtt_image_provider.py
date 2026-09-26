"""MqttImageProvider's loop binding.

The provider is constructed **off** the event loop: `SourceHelper.get_image_provider`
is a blocking DB call, and every caller (the watcher's source setup, the zone editor's
frame grab) hands it to a worker thread. Anything the constructor asks of the running
loop therefore fails with "no running event loop" — which is exactly what stopped an
MQTT source from being opened at all.

No broker is contacted here: `MqttClient` connects inside its own constructor, so it is
replaced with a fake that records the subscription callback.
"""

import asyncio
import base64
import io
import json
import threading

import numpy as np
import pytest
from PIL import Image

from apps.helpers.imageProviders import mqttImageProvider as mqtt_provider_module
from apps.helpers.imageProviders.mqttImageProvider import MqttImageProvider


class FakeMqttClient:
    """Stand-in for MqttClient that never touches a network."""

    def __init__(self, broker_address, prefix="myhome/", broker_port=1883):
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.callbacks: dict[str, object] = {}
        self.unsubscribed: list[str] = []
        self.disconnected = False
        self.connected = True

    def is_connected(self):
        return self.connected and not self.disconnected

    def subscribe(self, topic, callback, absoluteTopic=False):
        self.callbacks[topic] = callback

    def unsubscribe(self, topic):
        self.unsubscribed.append(topic)
        self.callbacks.pop(topic, None)

    def disconnect(self):
        self.disconnected = True


class FakeMessage:
    def __init__(self, payload: bytes):
        self.payload = payload


def _png_bytes(rgb: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((4, 4, 3), rgb, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_clients(monkeypatch):
    made: list[FakeMqttClient] = []

    def factory(*args, **kwargs):
        client = FakeMqttClient(*args, **kwargs)
        made.append(client)
        return client

    monkeypatch.setattr(mqtt_provider_module, "MqttClient", factory)
    return made


class TestConstructedOffTheEventLoop:
    async def test_construction_on_a_worker_thread_succeeds(self, fake_clients):
        """The regression: every real caller builds providers via a worker thread, where
        asyncio.get_running_loop() raises."""
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        assert isinstance(provider, MqttImageProvider)

    async def test_a_frame_published_after_start_is_delivered(self, fake_clients):
        """Binding late must still wire the broker thread to the loop — a provider that
        constructs fine but never yields a frame is the same outage one step later."""
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()
        client = fake_clients[0]

        # Delivered from another thread, as paho does.
        callback = client.callbacks["cam/frames"]
        thread = threading.Thread(target=callback, args=(FakeMessage(_png_bytes((255, 0, 0))),))
        thread.start()
        thread.join()

        image = await asyncio.wait_for(provider.getNextImage(), timeout=5)
        # PIL decodes RGB; the provider reverses it to BGR.
        assert tuple(image.image[0, 0]) == (0, 0, 255)

    async def test_no_subscription_is_made_before_start(self, fake_clients):
        """A frame arriving with no loop bound would have nowhere to go, so the
        subscription waits for the loop rather than dropping frames on the floor."""
        await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        assert fake_clients[0].callbacks == {}

    async def test_stop_unbinds_the_loop(self, fake_clients):
        """After stop() the broker thread must not post into a loop nobody is running."""
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()
        callback = fake_clients[0].callbacks["cam/frames"]
        await provider.stop()

        # Late frame: dropped, not raised, and not queued.
        callback(FakeMessage(_png_bytes((255, 0, 0))))
        assert provider._frameQueue.empty()


class TestBacklog:
    async def test_a_slow_consumer_gets_recent_frames_not_a_growing_backlog(self, fake_clients):
        """A watcher paces itself at its analysis fps. With an unbounded FIFO, a source
        publishing faster than that grows without limit and every frame handed over is
        further behind reality — the frames must be dropped, not banked."""
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()
        callback = fake_clients[0].callbacks["cam/frames"]

        # One stale blue frame, then far more fresh red ones than the queue can hold.
        callback(FakeMessage(_png_bytes((0, 0, 255))))
        for _ in range(MqttImageProvider.QUEUE_SIZE * 5):
            callback(FakeMessage(_png_bytes((255, 0, 0))))
        await asyncio.sleep(0.05)  # let the queued _offer() coroutines run

        assert provider._frameQueue.qsize() <= MqttImageProvider.QUEUE_SIZE
        image = await asyncio.wait_for(provider.getNextImage(), timeout=5)
        # BGR, so a red frame reads (0, 0, 255) and the evicted blue one would be (255, 0, 0).
        assert tuple(image.image[0, 0]) == (0, 0, 255), "served the stale frame from the backlog"


class TestReconnect:
    async def test_reconnect_resubscribes_and_delivers_again(self, fake_clients):
        """The base reconnect() is stop-then-start, and a watcher calls it whenever it
        decides a source has gone stale. A stop() that left the provider unable to
        restart meant the source went silent for good — with no error anywhere."""
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()

        await provider.reconnect()

        client = fake_clients[-1]
        assert "cam/frames" in client.callbacks, "reconnect left the provider unsubscribed"
        client.callbacks["cam/frames"](FakeMessage(_png_bytes((255, 0, 0))))
        image = await asyncio.wait_for(provider.getNextImage(), timeout=5)
        assert tuple(image.image[0, 0]) == (0, 0, 255)

    async def test_reconnect_builds_a_live_client(self, fake_clients):
        """The rebuilt client must be a fresh connection, not the disconnected one."""
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()
        await provider.reconnect()

        assert len(fake_clients) == 2
        assert fake_clients[0].disconnected
        assert not fake_clients[-1].disconnected


def _jpeg_bytes(rgb: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((4, 4, 3), rgb, dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


class TestCadence:
    """The read budget is measured, not fixed.

    A flat timeout cannot serve this source: the same provider has to cope with a topic
    publishing at 20 fps and one publishing every few minutes, and a budget shorter than
    the publisher's own interval reports a perfectly healthy source as failing on every
    single read — then tears down its subscription to "fix" it.
    """

    async def test_a_configured_budget_is_widened_by_a_slower_measured_cadence(self, fake_clients):
        """The measured cadence may only ever widen a configured budget.

        A source told to expect a frame every 60 s but observed publishing every 120 s is
        being read wrong somewhere — and failing it every cycle is the least useful of the
        two available responses.
        """
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames", max_silence_s=60.0)
        await provider.start()
        callback = fake_clients[0].callbacks["cam/frames"]

        clock = [1000.0]
        provider._now = lambda: clock[0]
        callback(FakeMessage(_png_bytes((255, 0, 0))))
        clock[0] += 120.0
        callback(FakeMessage(_png_bytes((255, 0, 0))))

        assert provider._frame_budget_s() == pytest.approx(240.0)

    async def test_a_burst_does_not_shrink_the_budget_below_the_quiet_gap(self, fake_clients):
        """The estimate is the worst gap in the window, not the latest one.

        A source that alternates bursts and silence would otherwise learn the burst rate
        mid-burst and declare itself failed on the next quiet stretch — which is the
        behaviour of a variable publisher, not an edge case.
        """
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()
        callback = fake_clients[0].callbacks["cam/frames"]

        clock = [1000.0]
        provider._now = lambda: clock[0]
        callback(FakeMessage(_png_bytes((255, 0, 0))))
        clock[0] += 90.0  # a quiet stretch
        callback(FakeMessage(_png_bytes((255, 0, 0))))
        for _ in range(5):  # then a burst
            clock[0] += 0.05
            callback(FakeMessage(_png_bytes((255, 0, 0))))

        assert provider._cadence_s == pytest.approx(90.0)

    async def test_the_window_forgets_an_old_outage(self, fake_clients):
        """Unlike the Ring provider's running maximum: a single outage must not leave
        this source promising a twenty-minute budget for the rest of the run."""
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()
        callback = fake_clients[0].callbacks["cam/frames"]

        clock = [1000.0]
        provider._now = lambda: clock[0]
        callback(FakeMessage(_png_bytes((255, 0, 0))))
        clock[0] += 600.0  # one long outage
        callback(FakeMessage(_png_bytes((255, 0, 0))))
        assert provider._cadence_s == pytest.approx(600.0)

        # Well past the window, publishing steadily again.
        for _ in range(12):
            clock[0] += MqttImageProvider.CADENCE_BUCKET_S
            callback(FakeMessage(_png_bytes((255, 0, 0))))

        assert provider._cadence_s == pytest.approx(MqttImageProvider.CADENCE_BUCKET_S)


class TestFreshness:
    async def test_a_queued_backlog_under_the_bound_is_still_skipped(self, fake_clients):
        """Eviction only caps the queue's *length*; draining is what caps a frame's age.

        Two frames fit inside the bound, so a plain FIFO hands over the older one — and
        every detection drawn from it is filed against the moment it was analysed.
        """
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()
        callback = fake_clients[0].callbacks["cam/frames"]

        callback(FakeMessage(_png_bytes((0, 0, 255))))  # blue, stale
        callback(FakeMessage(_png_bytes((255, 0, 0))))  # red, current
        await asyncio.sleep(0.05)
        assert provider._frameQueue.qsize() == 2, "the bound must not be what makes this pass"

        image = await asyncio.wait_for(provider.getNextImage(), timeout=5)
        assert tuple(image.image[0, 0]) == (0, 0, 255), "served a frame that was already superseded"


class TestPublishedEncoding:
    async def test_a_published_jpeg_is_handed_on_unchanged(self, fake_clients):
        """The publisher's own bytes are what make an event clip full-resolution; the
        analysis frame is downscaled, so re-encoding that instead loses the detail for
        good."""
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()
        raw = _jpeg_bytes((255, 0, 0))
        fake_clients[0].callbacks["cam/frames"](FakeMessage(raw))

        image = await asyncio.wait_for(provider.getNextImage(), timeout=5)

        packet = json.loads(image.encoded_b64)
        assert packet["format"] == "jpeg"
        assert base64.b64decode(packet["data"]) == raw

    async def test_a_published_png_is_not_announced_as_jpeg(self, fake_clients):
        """The packet declares its format, so a PNG passed off as JPEG is a lie that only
        surfaces in whatever tries to decode it."""
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()
        fake_clients[0].callbacks["cam/frames"](FakeMessage(_png_bytes((255, 0, 0))))

        image = await asyncio.wait_for(provider.getNextImage(), timeout=5)

        assert image.encoded_b64 is None


class TestReadiness:
    async def test_not_ready_until_subscribed(self, fake_clients):
        """A watcher servicing several cameras from one loop skips a source that is not
        ready; without the gate it instead blocks on this one for the whole frame budget,
        which every other camera pays for."""
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        assert not provider.is_ready()
        await provider.start()
        assert provider.is_ready()
        await provider.stop()
        assert not provider.is_ready()


class TestUnboundedSilence:
    """A motion- or event-triggered topic may publish nothing for days and be healthy.

    This is the case every frame-derived timeout gets wrong, and it is the default: a
    read that gives up after any fixed interval reports a working doorbell as a failed
    source on a schedule, then tears down its subscription to recover from a problem that
    was never there.
    """

    async def test_no_read_deadline_by_default(self, fake_clients):
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()

        assert provider._frame_budget_s() is None

    async def test_a_read_does_not_fail_while_the_topic_is_merely_quiet(self, fake_clients):
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()

        read = asyncio.ensure_future(provider.getNextImage())
        done, _pending = await asyncio.wait({read}, timeout=0.3)
        assert not done, "a quiet topic must not produce a failure"

        # And it is still a working read, not a wedged one.
        fake_clients[0].callbacks["cam/frames"](FakeMessage(_png_bytes((255, 0, 0))))
        image = await asyncio.wait_for(read, timeout=5)
        assert tuple(image.image[0, 0]) == (0, 0, 255)

    async def test_the_deadline_stays_far_out_however_long_the_silence(self, fake_clients):
        """The deadline must not be derived from the observed cadence.

        A topic that published every 30 s this morning and nothing since lunch is an
        event-driven publisher behaving normally. Quoting one cadence out would have the
        watcher — which takes its whole fetch budget from this — declare it failed on the
        first quiet afternoon.
        """
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()
        callback = fake_clients[0].callbacks["cam/frames"]

        clock = [1000.0]
        provider._now = lambda: clock[0]
        callback(FakeMessage(_png_bytes((255, 0, 0))))
        clock[0] += 30.0
        callback(FakeMessage(_png_bytes((255, 0, 0))))
        assert provider._cadence_s == pytest.approx(30.0), "cadence really was measured"

        clock[0] += 7200.0  # two quiet hours
        assert provider.next_frame_expected_by() - clock[0] >= 86400.0


class TestConfiguredSilenceBudget:
    async def test_a_promised_cadence_can_opt_into_failing(self, fake_clients):
        """The operator is the only one who knows the topic has a contract to break."""
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames", max_silence_s=0.2)
        await provider.start()

        with pytest.raises(TimeoutError):
            await provider.getNextImage()

    async def test_the_deadline_follows_the_configured_budget(self, fake_clients):
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames", max_silence_s=600.0)
        await provider.start()
        clock = [1000.0]
        provider._now = lambda: clock[0]
        fake_clients[0].callbacks["cam/frames"](FakeMessage(_png_bytes((255, 0, 0))))

        assert provider.next_frame_expected_by() == pytest.approx(1600.0)


class TestBrokerLiveness:
    """What replaces frame arrivals as the health signal.

    Ignoring silence is only safe because breakage is detected some other way. The broker
    session is that way: it is evidence about this source rather than about whether
    anybody chose to publish, and it needs no frame to answer.
    """

    async def test_a_dead_broker_is_not_ready_even_with_no_expected_frames(self, fake_clients):
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()
        assert provider.is_ready()

        fake_clients[0].connected = False
        assert not provider.is_ready(), "silence being acceptable must not make a dead broker acceptable"

    async def test_pahos_own_retry_window_is_not_a_rebuild(self, fake_clients):
        """MqttClient reconnects and resubscribes on paho's thread. A consumer that
        rebuilt the provider inside that window would tear down a session that was about
        to come back — so the loss reads as 'connecting' until paho has given up."""
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()
        clock = [1000.0]
        provider._now = lambda: clock[0]

        fake_clients[0].connected = False
        assert provider.is_connecting()

        clock[0] += MqttImageProvider.RECONNECT_GRACE_S + 1
        assert not provider.is_connecting(), "past paho's retries only a rebuild recovers the source"

    async def test_a_restored_session_clears_the_loss(self, fake_clients):
        provider = await asyncio.to_thread(MqttImageProvider, "broker.local", 1883, "cam/frames")
        await provider.start()
        fake_clients[0].connected = False
        assert not provider.is_ready()

        fake_clients[0].connected = True

        assert provider.is_ready()
        assert not provider.is_connecting()
