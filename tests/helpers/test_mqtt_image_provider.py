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
import io
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
