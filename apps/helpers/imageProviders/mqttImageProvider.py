"""Image source fed by whole images published to an MQTT topic.

This is not a stream and must not be treated as one. A publisher may send several
frames a second, or one every few minutes, and the *same* publisher may do both at
different times of day — a doorbell that idles and then bursts, a script that only
publishes on a sensor trip. Nothing about the topic says which.

The consequence that shapes this file: **silence is not failure here, and no timeout
derived from the frames themselves can be correct.** A motion-triggered topic may
publish nothing for two days and be working perfectly, so any read deadline guessed from
the observed rate reports a healthy source as broken — which a flat 60 s wait did, on
every read, tearing down a good subscription to "fix" it. Only the operator knows whether
a topic has a promised cadence, so `max_silence_s` asks them, and defaults to never.

What replaces it as the health signal is the broker session (`is_ready`), which is
evidence about *this* source rather than about whether anybody chose to publish, and is
answerable at any moment without waiting.

The cadence learned from arrivals is therefore only ever used to *widen* a configured
budget and to pace the consumer — never to fail the source. It is kept over a sliding
window rather than forever, unlike `RingMqttImageProvider`'s running maximum: Ring's
anomalies are always *faster* than its interval (a motion snapshot landing beside a
scheduled one), whereas here a one-off outage is just as likely and a permanent maximum
would leave the source quoting a twenty-minute budget for the rest of the run.
"""

import asyncio
import base64
import io
import json
import logging
import time
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.mqttClient import MqttClient

from .imageProvider import ImageProvider, ImageProviderMetadata, ImageProviderType, ImageWithProviderMetadata
from .Registry import register_image_provider

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


@register_image_provider()
class MqttImageProvider(ImageProvider):
    # A consumer must never be handed an old frame: a watcher paces itself at its
    # analysis fps, so on any source publishing faster than that an unbounded FIFO grows
    # forever and every frame it yields is further behind reality — an NVR quietly
    # detecting what happened ten minutes ago. The bound caps the memory; `getNextImage`
    # drains to the newest frame, which is what caps the *age*. FfmpegImageProvider
    # drains for the same reason.
    QUEUE_SIZE = 4

    #: Floor on how long one read may wait, so a source that has only just started — no
    #: cadence measured yet — cannot fail on a single normal gap. There is deliberately
    #: **no ceiling**: see `max_silence_s`.
    MIN_FRAME_TIMEOUT = 30.0

    #: The deadline reported while silence is unbounded (`max_silence_s = 0`).
    #:
    #: Not an expectation — it is the horizon a consumer is asked to wait out before
    #: asking again, and it is recomputed from *now* on every call, so a consumer that
    #: re-reads it (as `StreamWatcherTask` does per fetch) is patient indefinitely. A
    #: week rather than a day because a consumer whose wait expires records a failure,
    #: and a motion-triggered topic quiet over a holiday must not accumulate one of
    #: those per day.
    UNBOUNDED_HORIZON_S = 604800.0

    #: How long after losing the broker session this source still calls itself
    #: "connecting" rather than broken. `MqttClient` reconnects on paho's own thread
    #: (`CONNECTION_RETRIES` × `CONNECTION_RETRY_DELAY` ≈ 10 s), and a consumer that
    #: rebuilt the provider inside that window would tear down a session that was about
    #: to come back. Past it, paho has given up and only a rebuild helps.
    RECONNECT_GRACE_S = 30.0

    #: Multiplier on the measured cadence before a frame is "overdue". A publisher on a
    #: timer still jitters, and a deadline at exactly one cadence cries wolf every cycle.
    CADENCE_SLACK = 2.0

    #: How far back the cadence estimate looks, and how finely it is bucketed. Ten
    #: minutes so a burst does not erase the knowledge that this source also goes quiet
    #: for minutes at a time; bucketed so a 20 fps burst costs ten entries rather than
    #: twelve thousand.
    CADENCE_WINDOW_S = 600.0
    CADENCE_BUCKET_S = 60.0

    # Injectable seams for tests — patched on the instance, never on the shared modules.
    _sleep = staticmethod(asyncio.sleep)
    _now = staticmethod(time.monotonic)

    @classmethod
    def provider_type(cls) -> ImageProviderType:
        """LIVE_STREAM, even though this source is frame-at-a-time.

        The type picks the consumer's pacing, and of the two candidates only this one is
        safe at both ends of the range: a LIVE_STREAM consumer rate-limits itself, which
        a slow publisher simply never notices, whereas a LIVE_SNAPSHOT consumer polls on
        a slow interval and would throw away nearly all of a fast publisher's frames.
        """
        return ImageProviderType.LIVE_STREAM

    def __init__(
        self,
        broker_address: str,
        broker_port: int,
        topic: str,
        min_capture_interval: float = 0.0,
        max_silence_s: float = 0.0,
    ):
        """Initializes the MQTT source."""
        self._topic: str = topic
        #: How long this topic may be silent before a read fails. **0 means never**, and
        #: that is the default because nothing about a topic reveals what it carries: a
        #: motion-triggered publisher can legitimately go days between images, and any
        #: finite default would report it as broken on a schedule for the rest of its
        #: life. Silence is safe to ignore precisely because it is not this source's
        #: health signal — the broker session is (`is_ready`), and that is checked
        #: independently of whether anybody is publishing.
        #:
        #: Set it only for a topic whose cadence is a known promise (a camera pushing
        #: every 2 s, a script on a cron), where silence really does mean failure and
        #: should be surfaced instead of waited out.
        self._max_silence_s: float = float(max_silence_s)
        # Kept so start() can rebuild the client after a stop(): the base reconnect() is
        # stop-then-start, and a provider that cannot be restarted goes silent for good
        # the first time a watcher decides its source is stale.
        self._broker_address: str = broker_address
        self._broker_port: int = int(broker_port)
        self._min_capture_interval: float = min_capture_interval
        self.min_capture_interval: float = min_capture_interval
        #: Pacing state for `min_capture_interval`, mirroring UrlImageProvider.
        self._next_capture_time: float = 0.0
        # Connecting blocks (five retries, two seconds apart) and this constructor already
        # runs on a worker thread, so this is the right place for it. start() rebuilds on
        # a thread for the same reason.
        self._mqtt_client: Optional[MqttClient] = MqttClient(broker_address, "/", self._broker_port)
        self._frameQueue: asyncio.Queue = asyncio.Queue(maxsize=MqttImageProvider.QUEUE_SIZE)
        self._dropped_frames: int = 0
        self._frame_idx: int = 0
        #: When the last frame arrived, and the cadence learned from arrivals. Both are
        #: written only by the broker thread; the loop side reads the two scalars and
        #: never the bucket list, which is what keeps this lock-free.
        self._last_frame_at: Optional[float] = None
        self._cadence_s: Optional[float] = None
        self._cadence_buckets: List[Tuple[float, float]] = []
        #: When the broker session was first observed down, or None while it is up. Read
        #: by `is_connecting` to tell "paho is retrying" from "paho has given up".
        self._broker_lost_at: Optional[float] = None
        # Bound in start(), NOT here. Providers are constructed off the event loop —
        # `SourceHelper.get_image_provider` is a blocking DB call every caller hands to a
        # worker thread — and `asyncio.get_running_loop()` on a worker thread raises
        # "no running event loop". Doing it in __init__ meant an MQTT source could not be
        # opened at all: the zone editor's frame grab and the watcher's own source setup
        # both died on construction, before anything touched the broker.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribed: bool = False

    def __repr__(self):
        return f"MqttImageProvider [{self._topic}]"

    @property
    def is_live(self) -> bool:
        """Always False: this source has no continuous tier, whatever its current rate.

        Read by `StreamWatcherTask._push_to_ring` to decide whether a frame feeds the
        sticky lead-in cache (`FrameRing._sticky_recent`). A slow publisher's frames age
        out of the dense ring long before the next one arrives, so without this its event
        clips open on no lead-in at all — the exact failure the sticky cache was built
        for on Ring.

        Saying False for a *fast* publisher costs nothing: the cache then holds frames
        from inside the clip's own window, which `lead_in_frames` filters out, and
        `RingClipBackend` only consults it when the dense ring came up short anyway.
        Deriving this from the measured cadence instead would add a threshold that has to
        agree with the pipeline's own, for no change in behaviour at either end.
        """
        return False

    async def start(self) -> None:
        """Bind the loop the broker thread will hand frames to, and subscribe.

        Idempotent, and the only place the loop is captured. Also rebuilds the client
        when called after a stop(), which is what makes reconnect() work.
        """
        self._loop = asyncio.get_running_loop()
        if self._mqtt_client is None:
            self._mqtt_client = await asyncio.to_thread(MqttClient, self._broker_address, "/", self._broker_port)
            self._subscribed = False
        if not self._subscribed:
            self._mqtt_client.subscribe(self._topic, self._pushFrame, absoluteTopic=True)
            self._subscribed = True

    def is_ready(self) -> bool:
        """Subscribed, and the broker session is up.

        The broker half is what makes unbounded silence safe. Frames are evidence that
        the *publisher* is active, not that this source is working — so a provider
        judging itself on arrivals cannot tell a quiet doorbell from a dead broker, and
        has to either fail the quiet one or ignore the dead one. The session answers that
        question directly and never depends on anybody publishing.

        Gated rather than always-True for a second reason: a watcher servicing several
        cameras from one loop can then skip this source while it is coming up, instead of
        discovering it by blocking for the whole frame budget — which every *other*
        camera then pays for too.
        """
        return self._subscribed and self._broker_connected()

    def is_connecting(self) -> bool:
        """True while the session is down but `MqttClient` may still restore it.

        Splits a lost session into its two cases, which want opposite treatment. paho
        reconnects on its own thread and resubscribes from the CONNACK handler, so a
        consumer that rebuilt this provider inside that window would tear down a session
        about to come back. Past `RECONNECT_GRACE_S` paho has exhausted its retries and
        only the consumer's rebuild recovers the source.
        """
        if self._subscribed and self._broker_connected():
            return False
        lost_at = self._broker_lost_at
        return lost_at is not None and self._now() - lost_at < self.RECONNECT_GRACE_S

    def _broker_connected(self) -> bool:
        """Poll the client's session state, remembering when it was first lost."""
        client = self._mqtt_client
        try:
            connected = client is not None and client.is_connected()
        except Exception:
            # A client that cannot answer must not make the source unusable; treating it
            # as connected leaves behaviour exactly as it is without the check.
            logger.debug("MqttClient could not report its connection state", exc_info=True)
            connected = client is not None
        if connected:
            self._broker_lost_at = None
        elif self._broker_lost_at is None:
            self._broker_lost_at = self._now()
        return connected

    def next_frame_expected_by(self) -> Optional[float]:
        """When the next frame should have arrived, on the `time.monotonic()` clock.

        A deadline rather than an interval because the cadence is the publisher's to
        change at any moment; recomputing from the last arrival means a consumer never
        has to know it changed. `StreamWatcherTask` derives its fetch budget from this,
        which is what stops a slow publisher being recorded as a failure on every read.

        With `max_silence_s` unset this reports `UNBOUNDED_HORIZON_S` out, deliberately
        far past any cadence this source has shown. The learned cadence must not be the
        answer here: a topic that published every 30 s this morning and nothing since
        lunch is the *normal* behaviour of an event-driven publisher, and a deadline one
        cadence out would have a consumer declare it failed on the first quiet afternoon.
        """
        now = self._now()
        if self._max_silence_s <= 0:
            return now + self.UNBOUNDED_HORIZON_S
        base = self._last_frame_at if self._last_frame_at is not None else now
        # Never hand back a deadline already in the past: a consumer deriving a wait
        # budget from it would get zero and time out instantly on a source that is
        # merely overdue.
        return max(base + self._max_silence_s, now + self.MIN_FRAME_TIMEOUT)

    def _frame_budget_s(self) -> Optional[float]:
        """How long a single read may wait; None waits indefinitely.

        The measured cadence only ever *widens* this, never narrows it. A steady
        publisher that stops is the one case a cadence-derived timeout would catch, and
        catching it costs failing every publisher whose cadence is legitimately
        irregular — which is most of them, and all of the event-driven ones.
        `max_silence_s` is how a caller who knows the topic's contract opts in.
        """
        if self._max_silence_s <= 0:
            return None
        return max(self.MIN_FRAME_TIMEOUT, self._max_silence_s, (self._cadence_s or 0.0) * self.CADENCE_SLACK)

    def _note_arrival(self) -> None:
        """Learn the publish cadence from consecutive arrivals. Runs on the broker thread.

        The estimate is the *worst* gap in the window, not the latest: a source that
        alternates between bursts and silence would otherwise learn the burst rate
        mid-burst and declare itself failed on the next quiet stretch. Kept in per-minute
        buckets so the window is cheap at any publish rate, and collapsed to a single
        float here so the event-loop side never touches the list.
        """
        now = self._now()
        previous, self._last_frame_at = self._last_frame_at, now
        if previous is None:
            return
        gap = now - previous
        if gap <= 0:
            return
        bucket = now - (now % self.CADENCE_BUCKET_S)
        if self._cadence_buckets and self._cadence_buckets[-1][0] == bucket:
            start, worst = self._cadence_buckets[-1]
            self._cadence_buckets[-1] = (start, max(worst, gap))
        else:
            self._cadence_buckets.append((bucket, gap))
        cutoff = now - self.CADENCE_WINDOW_S
        while self._cadence_buckets and self._cadence_buckets[0][0] < cutoff:
            self._cadence_buckets.pop(0)
        self._cadence_s = max(worst for _, worst in self._cadence_buckets)

    def _pushFrame(self, mqttMsg):
        """Callback for handling incoming MQTT messages. Runs on the broker's thread."""
        try:
            loop = self._loop
            if loop is None:
                # Only reachable if the subscription outlives the loop it was bound to
                # (i.e. after stop()). Dropping is right: nothing is waiting on the queue.
                logger.debug("Dropping an MQTT frame received while no event loop is bound")
                return
            raw = mqttMsg.payload
            image = Image.open(io.BytesIO(raw))
            # Publishers send greyscale and RGBA too, and without the convert both fall
            # through the channel reversal below into nonsense that nothing downstream
            # can detect: a 2-D greyscale array comes out mirrored left-to-right, and
            # RGBA comes out as ABGR. Converting first makes the reversal mean what it
            # says. PIL decodes to RGB; this app treats a frame as BGR.
            frame = np.array(image.convert("RGB"))[..., ::-1]
            self._note_arrival()
            # `put`, not `put_nowait`, would block this coroutine forever once the queue
            # filled — one pending future per published frame, which is the leak the
            # bound was meant to prevent.
            asyncio.run_coroutine_threadsafe(self._offer(frame, raw), loop)
        except Exception as e:
            logger.error(e, exc_info=True)

    def _note_drop(self) -> None:
        self._dropped_frames += 1
        if self._dropped_frames % 100 == 1:
            # Lag is otherwise invisible: the consumer keeps getting frames, just
            # increasingly old ones.
            logger.warning(
                "%s is publishing faster than it is being read; dropped %d frame(s) so far",
                self,
                self._dropped_frames,
            )

    async def _offer(self, frame: np.ndarray, raw: bytes) -> None:
        """Queue a frame, discarding the oldest when the consumer has fallen behind."""
        while self._frameQueue.full():
            try:
                self._frameQueue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._note_drop()
        self._frameQueue.put_nowait((frame, raw))

    async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
        """Asynchronously retrieves the newest image the publisher has sent."""
        if self._loop is None:
            # A caller that skipped start() still gets frames rather than silence.
            await self.start()

        # Self-pace exactly as UrlImageProvider does, so a topic publishing at 20 fps
        # does not flood a collector that wanted a frame every few seconds.
        if self._min_capture_interval > 0:
            now = self._now()
            if now < self._next_capture_time:
                await self._sleep(self._next_capture_time - now)
            self._next_capture_time = self._now() + self._min_capture_interval

        budget = self._frame_budget_s()
        if budget is None:
            # No deadline at all, which is the default: an event-driven topic has nothing
            # to be late for. The caller is not stranded — a watcher wraps this in its own
            # wait derived from `next_frame_expected_by`, and a broker that dies is caught
            # by `is_ready` rather than by waiting for a frame that was never coming.
            frame, raw = await self._frameQueue.get()
        else:
            try:
                frame, raw = await asyncio.wait_for(self._frameQueue.get(), timeout=budget)
            except asyncio.TimeoutError:
                raise TimeoutError(f"No image on {self._topic} within its {budget:.0f}s silence budget")

        # Everything behind the newest frame is already history the consumer cannot act
        # on — it paces itself, so those frames were going to be skipped anyway, and
        # handing one over dates every detection drawn from it. A no-op for a slow
        # publisher, where the queue never holds more than one.
        while True:
            try:
                frame, raw = self._frameQueue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._note_drop()

        frame_idx = self._frame_idx
        self._frame_idx += 1
        provided = ImageWithProviderMetadata(
            frame, metadata=ImageProviderMetadata(source=self.source_name, frame_idx=frame_idx)
        )
        # Hand on the bytes the publisher actually sent rather than re-encoding the
        # decoded array later: this is the full-resolution image, so it is what makes an
        # event clip from this source full-res, and it saves an encode for any consumer
        # that wanted JPEG anyway. Only JPEG qualifies — the packet declares its format,
        # and a PNG announced as JPEG is a lie that only surfaces in whatever decodes it.
        if raw[:2] == b"\xff\xd8":
            provided.encoded_b64 = json.dumps(
                {"format": "jpeg", "metadata": {}, "data": base64.b64encode(raw).decode("ascii")}
            )
        return provided

    async def stop(self) -> None:
        """Stops the image provider and releases any resources"""
        if self._mqtt_client:
            self._mqtt_client.unsubscribe(self._topic)
            self._mqtt_client.disconnect()
            self._mqtt_client = None
        self._subscribed = False
        self._broker_lost_at = None
        # A frame delivered after this point has no loop to be posted to.
        self._loop = None

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        """
        Return a schema describing the parameters for this Provider
        Each key is a parameter name, value is a dict with:
            - type: str
            - required: bool
            - default: Any (optional)
            - help: str (optional)
            - options: list (optional, for enums)
            - schema: dict (optional, for nested objects)
        """
        return {
            "broker_address": {
                "type": "string",
                "label": "Broker Address",
                "required": True,
                "description": "MQTT broker address",
            },
            "broker_port": {
                "type": "int",
                "label": "Broker Port",
                "default": 1883,
                "required": True,
                "description": "MQTT broker port",
            },
            "topic": {
                "type": "string",
                "label": "Mqtt Topic",
                "required": True,
                "description": "MQTT topic to subscribe to",
            },
            "max_silence_s": {
                "label": "Max Silence (s)",
                "type": "float",
                "required": False,
                "default": 0.0,
                "description": (
                    "Seconds of silence before this source is reported as failed. "
                    "0 = never, the right answer for a motion- or event-triggered topic "
                    "that can legitimately publish nothing for hours or days; broker "
                    "disconnection is still detected either way. Set it only for a topic "
                    "with a promised cadence, where silence really does mean failure."
                ),
            },
            "min_capture_interval": {
                "label": "Min Capture Interval (s)",
                "type": "float",
                "required": False,
                "default": 0.0,
                "description": "Minimum seconds between frames returned by this source. 0 = no source-level limit (the task's own interval applies).",
            },
        }
