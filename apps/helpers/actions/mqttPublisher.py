"""A non-blocking MQTT publisher for the watcher.

`MqttClient` connects synchronously with 5 retries × 2 s, which is up to ten seconds of
stalled event loop, and it has no outbound bound at all. Neither is acceptable for a
task whose frame loop must keep running: a broker that is merely *slow* would stall
inference, and a broker that is down would grow memory without limit.

This wrapper gives the watcher four properties instead:

* **Connect happens off the loop**, in a worker thread, so nothing blocks.
* **The outbound queue is bounded and drops oldest**, with a counter surfaced in the
  status panel — losing the oldest state message is strictly better than unbounded RAM
  or a stalled camera.
* **The bound is on bytes as well as messages.** A message count alone was only ever
  right because every payload was a small JSON event. An annotated-frame feed publishes
  JPEGs, and 1000 queued frames is ~100 MB of RAM held for a broker that is not reading
  — precisely the unbounded growth this class exists to prevent.
* **Reconnection is the background worker's problem**, never the caller's.

Publish-only for now. Inbound commands would attach at `subscribe()`, which is left
documented rather than built, so adding them later doesn't require reshaping this.
"""

import asyncio
import json
import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger(__name__)

# ~1000 messages is minutes of event traffic at any sane rate; past that, the broker is
# gone rather than slow and the oldest messages have no value.
DEFAULT_QUEUE_SIZE = 1000

# The RAM ceiling for everything waiting to go out. Sized for the live-image case: at a
# few tens of KB per annotated 640 px frame this is a few hundred queued frames, which
# is far more backlog than a live feed can ever usefully have and still a bound a
# 24/7 watcher can hold. Event JSON never comes close to it.
DEFAULT_QUEUE_BYTES = 32 * 1024 * 1024


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


@dataclass
class _Pending:
    """One queued message.

    Mutable on purpose: a coalescing publish overwrites the payload of the entry
    already in the queue rather than appending a second one, which is what keeps a
    live feed's backlog at one frame per topic instead of a growing run of stale ones.
    """

    topic: str
    payload: Any
    retain: bool
    qos: int
    size: int
    #: Set once the message has left the queue, so a coalescing publish that still
    #: holds a reference to it appends a fresh entry instead of mutating a sent one.
    gone: bool = False


class AsyncMqttPublisher:
    """Bounded, drop-oldest, background-reconnecting MQTT publisher."""

    def __init__(
        self,
        settings: MqttSettings,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        queue_bytes: int = DEFAULT_QUEUE_BYTES,
    ) -> None:
        self._settings = settings
        self._max_messages = max(1, int(queue_size))
        self._max_bytes = max(1, int(queue_bytes))
        # A deque rather than an asyncio.Queue: coalescing needs to reach into the
        # backlog and replace an entry, which a Queue deliberately does not allow.
        self._pending: Deque[_Pending] = deque()
        # topic → the queued entry a coalescing publish on that topic should overwrite.
        self._coalescing: Dict[str, _Pending] = {}
        self._queued_bytes = 0
        self._ready = asyncio.Event()
        self._worker: Optional[asyncio.Task] = None
        self._client: Any = None
        self._connected = threading.Event()
        # Counters the status panel reads, so a broker problem is visible rather than
        # silently degrading the feature.
        self.published = 0
        self.dropped = 0
        self.coalesced = 0
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

    def publish(self, topic: str, payload: Any, retain: bool = False, qos: int = 0, coalesce: bool = False) -> None:
        """Enqueue a message. Never blocks and never raises.

        Called from the frame loop, so both properties matter: this must be the
        cheapest possible operation on the hot path.

        `coalesce` marks the payload as *state* rather than an *edge*: only the newest
        value for the topic matters, so a publish supersedes whatever is still queued
        for it. Right for a live annotated frame — a subscriber gains nothing from the
        frame before last, and queueing both is how a slow broker turns into a memory
        problem. Wrong for event edges, where every transition is its own fact and
        dropping one loses information.
        """
        if not isinstance(payload, (str, bytes)):
            payload = json.dumps(payload, default=str)
        size = len(payload)

        if coalesce:
            existing = self._coalescing.get(topic)
            if existing is not None and not existing.gone:
                self._queued_bytes += size - existing.size
                existing.payload = payload
                existing.size = size
                existing.retain = retain
                existing.qos = qos
                self.coalesced += 1
                self._evict()
                self._ready.set()
                return

        message = _Pending(topic=topic, payload=payload, retain=retain, qos=qos, size=size)
        self._pending.append(message)
        self._queued_bytes += size
        if coalesce:
            self._coalescing[topic] = message
        self._evict()
        self._ready.set()

    def _forget(self, message: _Pending) -> None:
        """Drop one entry's bookkeeping. Idempotent."""
        if message.gone:
            return
        message.gone = True
        self._queued_bytes -= message.size
        # Only if it is still the entry this topic coalesces onto — a newer append for
        # the same topic must not be unregistered by an older one leaving.
        if self._coalescing.get(message.topic) is message:
            del self._coalescing[message.topic]

    def _evict(self) -> None:
        """Drop oldest until the backlog is inside both bounds.

        Never empties the queue: a single payload larger than the whole byte budget is
        a configuration problem (an enormous JPEG), and silently discarding it every
        time would look like the broker was refusing it. One over-budget message is
        allowed through; the bound still holds against a *backlog* of them.
        """
        while len(self._pending) > 1 and (
            len(self._pending) > self._max_messages or self._queued_bytes > self._max_bytes
        ):
            oldest = self._pending.popleft()
            self._forget(oldest)
            self.dropped += 1

    async def _drain(self) -> None:
        while True:
            if not self._pending:
                self._ready.clear()
                try:
                    await self._ready.wait()
                except asyncio.CancelledError:
                    return
                continue
            message = self._pending.popleft()
            self._forget(message)
            topic, payload, retain, qos = message.topic, message.payload, message.retain, message.qos
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
            # Distinct from `dropped`: a coalesced frame was superseded by a newer one
            # for the same topic, which is the feature working, not a loss. Merging the
            # two counters would make a healthy live feed look like a failing broker.
            "coalesced": self.coalesced,
            "failures": self.failures,
            "lastError": self.last_error,
            "queued": len(self._pending),
            "queuedBytes": self._queued_bytes,
        }
