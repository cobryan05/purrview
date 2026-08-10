"""A non-blocking MQTT publisher for the watcher.

`MqttClient` connects synchronously with 5 retries × 2 s, which is up to ten seconds of
stalled event loop, and it has no outbound bound at all. Neither is acceptable for a
task whose frame loop must keep running: a broker that is merely *slow* would stall
inference, and a broker that is down would grow memory without limit.

This wrapper gives the watcher three properties instead:

* **Connect happens off the loop**, in a worker thread, so nothing blocks.
* **The outbound queue is bounded and drops oldest**, with a counter surfaced in the
  status panel — losing the oldest state message is strictly better than unbounded RAM
  or a stalled camera.
* **Reconnection is the background worker's problem**, never the caller's.

Publish-only for now. Inbound commands would attach at `subscribe()`, which is left
documented rather than built, so adding them later doesn't require reshaping this.
"""

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ~1000 messages is minutes of event traffic at any sane rate; past that, the broker is
# gone rather than slow and the oldest messages have no value.
DEFAULT_QUEUE_SIZE = 1000


@dataclass
class MqttSettings:
    host: str
    port: int = 1883
    prefix: str = "whiskerwatcher"
    username: Optional[str] = None
    password: Optional[str] = None
    tls: bool = False
    client_id: str = "whiskerwatcher"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MqttSettings":
        return cls(
            host=str(data.get("host") or "localhost"),
            port=int(data.get("port") or 1883),
            prefix=str(data.get("prefix") or "whiskerwatcher").rstrip("/"),
            username=data.get("username") or None,
            password=data.get("password") or None,
            tls=bool(data.get("tls", False)),
            client_id=str(data.get("client_id") or "whiskerwatcher"),
        )


class AsyncMqttPublisher:
    """Bounded, drop-oldest, background-reconnecting MQTT publisher."""

    def __init__(self, settings: MqttSettings, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self._settings = settings
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._worker: Optional[asyncio.Task] = None
        self._client: Any = None
        self._connected = threading.Event()
        # Counters the status panel reads, so a broker problem is visible rather than
        # silently degrading the feature.
        self.published = 0
        self.dropped = 0
        self.failures = 0
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Connect off the event loop and begin draining the queue."""
        await asyncio.to_thread(self._connect_blocking)
        self._worker = asyncio.create_task(self._drain())

    def _connect_blocking(self) -> None:
        """Runs in a worker thread — safe to block here, never on the loop."""
        import paho.mqtt.client as mqtt

        client = mqtt.Client(client_id=self._settings.client_id)
        if self._settings.username:
            client.username_pw_set(self._settings.username, self._settings.password or None)
        if self._settings.tls:
            client.tls_set()

        # Last Will: the broker publishes "offline" for us if we vanish, which is what
        # makes the state topic trustworthy to a Home Assistant user.
        client.will_set(f"{self._settings.prefix}/state", payload="offline", retain=True)
        client.on_connect = lambda *_: self._connected.set()
        client.on_disconnect = lambda *_: self._connected.clear()

        try:
            client.connect(self._settings.host, self._settings.port, keepalive=60)
            # loop_start owns reconnection from here; the caller never waits on it.
            client.loop_start()
            self._client = client
            client.publish(f"{self._settings.prefix}/state", "online", retain=True)
        except Exception as e:
            # A down broker must not prevent the watcher from starting. Queued messages
            # are dropped until paho's own reconnect succeeds.
            self.failures += 1
            self.last_error = str(e)
            logger.warning("MQTT connect failed (%s); publishes will be dropped until it recovers", e)
            self._client = client
            try:
                client.loop_start()
            except Exception:
                pass

    async def close(self) -> None:
        """Explicit lifecycle, tied to the task's `_deinit`.

        Deliberately not `__del__`: destructor-time disconnects fire at GC's whim,
        which for a network client means "at an unpredictable moment, possibly during
        interpreter shutdown".
        """
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
            self._worker = None
        if self._client is not None:
            try:
                self._client.publish(f"{self._settings.prefix}/state", "offline", retain=True)
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                logger.debug("Error during MQTT shutdown", exc_info=True)
            self._client = None

    # ------------------------------------------------------------------ publish

    def publish(self, topic: str, payload: Any, retain: bool = False, qos: int = 0) -> None:
        """Enqueue a message. Never blocks and never raises.

        Called from the frame loop, so both properties matter: this must be the
        cheapest possible operation on the hot path.
        """
        if not isinstance(payload, (str, bytes)):
            payload = json.dumps(payload, default=str)
        message = (topic, payload, retain, qos)
        while True:
            try:
                self._queue.put_nowait(message)
                return
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:
                    return

    async def _drain(self) -> None:
        while True:
            try:
                topic, payload, retain, qos = await self._queue.get()
            except asyncio.CancelledError:
                return
            try:
                full_topic = f"{self._settings.prefix}/{topic.lstrip('/')}"
                if self._client is None:
                    self.dropped += 1
                    continue
                # paho's publish is non-blocking (it queues onto its own network
                # thread), so this does not need a thread hop.
                info = self._client.publish(full_topic, payload, qos=qos, retain=retain)
                if getattr(info, "rc", 0) != 0:
                    self.failures += 1
                    self.last_error = f"publish rc={info.rc}"
                else:
                    self.published += 1
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.failures += 1
                self.last_error = str(e)
                logger.debug("MQTT publish failed", exc_info=True)

    def stats(self) -> Dict[str, Any]:
        return {
            "published": self.published,
            "dropped": self.dropped,
            "failures": self.failures,
            "lastError": self.last_error,
            "queued": self._queue.qsize(),
        }
