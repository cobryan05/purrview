"""Ring camera image provider, fed by a ring-mqtt bridge.

A Ring camera is not a camera in the sense the rest of this app assumes. It is asleep
almost all of the time: there is no continuous stream to connect to, waking it costs
battery and cloud round-trips, and it only produces video when something asks it to.
Pointing `FfmpegImageProvider` at its RTSP URL "works", but it works by holding the
camera awake forever, which is the one thing a battery camera must not do.

So this provider has two tiers and moves between them:

* **Idle** — subscribe to ring-mqtt's ``snapshot/image`` topic. ring-mqtt publishes a
  full JPEG on an interval (90s by default) whether or not anyone is watching, and it
  does so from Ring's own cloud snapshot: the camera is not woken, and the frames cost
  us nothing at all. This is the tier a Ring source sits in essentially always.
* **Live** — publish ``stream/command`` and read the resulting RTSP stream at full
  frame rate through an inner `FfmpegImageProvider`, which brings the relay lease,
  the rewind buffer and reconnect handling with it.

Escalation to the live tier happens on either of two signals, both of which arrive
over MQTT:

* ``motion/state`` going ``ON`` — ring-mqtt's own motion event, free and pushed.
* ``stream/command`` being published — *anyone* deliberately asking for the stream,
  including this app's own UI button, Home Assistant, or a shell running
  ``mosquitto_pub``.

Note which topic that second one is. ``stream/state`` would be the obvious choice and
is the wrong one: it also reflects the RTSP server's own on-demand churn, which nobody
requested. On a live system all three cameras were observed cycling ``OFF``→``ON``
about every 20 seconds without a single MQTT command being sent, so a provider keyed
on state would hold the camera awake permanently while believing it was responding to
demand. A command is an intent; a state is not.

That second trigger is why there is no RPC into a running watcher anywhere in this
file. A live request is just a message on the broker, so the thing that wants video
(a UI button, an automation) publishes ``stream/command`` and every provider watching
that camera escalates on its own. The broker is already the integration bus; routing
the trigger through it means an externally-started stream is picked up identically to
one we started ourselves, rather than being a second code path that has to be kept in
agreement with the first.

De-escalation is time-based: the tier drops back to idle once nothing has re-triggered
it for `live_hold_s`. We only ever publish ``off`` for a stream *we* turned on — if
Home Assistant or Frigate is holding the camera up for its own reasons, that is not
ours to cancel.

**`live_hold_s` must stay small, and this is not a tuning preference.** A Ring camera
raises no motion events while a live capture is running, and reading its stream *is* a
live capture — so every extra second held is a second during which the camera cannot
report the next event. Holding "just a bit longer to be safe" is therefore
self-defeating: it buys tail footage of the event we already have by blinding the
camera to the one after it. Worse, it is silent — nothing errors, the stream is fine,
motion simply stops arriving.

The default (15s) matches what the deployment this was built against does elsewhere:
Home Assistant disables the Frigate camera 15s after motion clears. Two consumers on
one camera means it stays awake until *both* let go, so the slowest one sets the
blackout. Raising this above the other consumer's release time makes this provider the
one holding the camera down.

The relay releases correctly on our side — `release_lease` deletes the MediaMTX path
when the last holder lets go, which drops the RTSP connection and lets the camera
sleep. Note the path is created with ``sourceOnDemand: False``, so it pulls the source
the moment it exists: nothing is lazy here, and the only thing that ends the capture is
the release.
"""

import asyncio
import base64
import io
import json
import logging
import time
from typing import Any, Optional

import numpy as np
from PIL import Image

from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.mqttClient import MqttClient

from .ffmpegImageProvider import FfmpegImageProvider
from .imageProvider import (
    ImageProvider,
    ImageProviderMetadata,
    ImageProviderType,
    ImageWithProviderMetadata,
)
from .Registry import register_image_provider

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

#: ring-mqtt's default RTSP path suffixes. `_live` is the camera now; `_event` plays
#: back an already-recorded event from Ring's cloud.
LIVE_SUFFIX = "_live"


@register_image_provider()
class RingMqttImageProvider(ImageProvider):
    #: Snapshots are the idle tier's only source of frames and ring-mqtt publishes them
    #: on a slow interval (90s out of the box). The wait therefore has to be a large
    #: multiple of that interval, not the seconds-scale timeout a continuous stream
    #: would use — an idle Ring camera is *supposed* to be quiet, and treating a
    #: normal gap between snapshots as a failure would report a healthy camera as down
    #: on almost every read.
    IDLE_FRAME_TIMEOUT = 300.0

    #: Bound on the idle snapshot queue. Snapshots are minutes apart, so this only ever
    #: absorbs a burst (an interval snapshot landing next to a motion one); anything
    #: older than a couple of frames is of no use to a live consumer.
    QUEUE_SIZE = 2

    # Injectable seams for tests — patched on the instance rather than on the global
    # modules, which are shared process-wide and unsafe to monkeypatch while other
    # tests run concurrently.
    _sleep = staticmethod(asyncio.sleep)
    _now = staticmethod(time.monotonic)

    @classmethod
    def provider_type(cls) -> ImageProviderType:
        """LIVE_STREAM, even though the idle tier behaves like a snapshot poller.

        The type picks the consumer's pacing, and of the two available answers only
        this one is safe in both tiers: a LIVE_STREAM consumer rate-limits itself,
        which an idle Ring source simply doesn't notice, whereas a LIVE_SNAPSHOT
        consumer polls on a slow interval and would throw away most of the frames of
        an escalated 20fps stream — exactly the ones we woke the camera to get.
        """
        return ImageProviderType.LIVE_STREAM

    @classmethod
    def ingest_tier(cls) -> str:
        """Deliberately understated as the in-RAM frame ring.

        This source's real answer is "it depends": while live it is a genuine RTSP
        stream in a MediaMTX path with recording, and while idle there is no
        continuous stream to record at all. `resolve_ingest_tier` is a pure function
        of class and params, so it cannot express a tier that changes at runtime.

        Given a forced choice, claiming the weaker tier is the only safe direction.
        The badge, the relay's path config and the watcher's clip backend are required
        never to disagree, and the camera is idle the overwhelming majority of the
        time — so "native" would be a promise of rewindable video that is usually
        untrue, while "ring" understates a capability we do briefly have. The inner
        `FfmpegImageProvider` resolves its own tier independently and still gets a
        proper recording path while live; this only affects what the UI claims.
        """
        return "ring"

    def __init__(
        self,
        broker_address: str,
        location_id: str,
        camera_id: str,
        broker_port: int = 1883,
        rtsp_host: str = "",
        rtsp_port: int = 8554,
        live_on_motion: bool = True,
        live_hold_s: float = 15.0,
        min_capture_interval: float = 0.0,
    ):
        self._broker_address: str = broker_address
        self._broker_port: int = int(broker_port)
        self._location_id: str = location_id.strip()
        self._camera_id: str = camera_id.strip()
        # Defaults to the broker host: ring-mqtt serves its RTSP from the same process
        # that talks to the broker, so one host answers for both in every ordinary
        # deployment. Split only when the bridge is behind a different address than
        # the broker it publishes to.
        self._rtsp_host: str = (rtsp_host or broker_address).strip()
        self._rtsp_port: int = int(rtsp_port)
        self._live_on_motion: bool = bool(live_on_motion)
        self._live_hold_s: float = float(live_hold_s)

        self._min_capture_interval: float = min_capture_interval
        self.min_capture_interval: float = min_capture_interval

        self._base: str = f"ring/{self._location_id}/camera/{self._camera_id}"
        self._mqtt_client: Optional[MqttClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribed: bool = False

        self._snapshot_queue: asyncio.Queue = asyncio.Queue(maxsize=self.QUEUE_SIZE)
        #: Pacing state for `min_capture_interval`, mirroring UrlImageProvider.
        self._next_capture_time: float = 0.0
        #: When the last snapshot arrived, and the cadence learned from consecutive
        #: arrivals. Measured rather than configured: ring-mqtt's snapshot interval is
        #: a setting on the bridge (`snapshot_interval/command`), so any constant here
        #: would silently rot the moment someone changed it there.
        self._last_snapshot_at: Optional[float] = None
        self._snapshot_cadence_s: Optional[float] = None
        self._frame_idx: int = 0

        #: The live tier's reader. Built on escalation, torn down on de-escalation, so
        #: an idle source holds no ffmpeg process and no relay lease.
        self._live_provider: Optional[FfmpegImageProvider] = None
        #: Monotonic deadline after which the live tier drops back to idle. Pushed
        #: forward by every fresh trigger.
        self._live_until: float = 0.0
        #: True only when *we* published `on`. Gates whether we may publish `off`:
        #: a stream someone else is holding up is not ours to cancel.
        self._we_requested_live: bool = False
        #: ring-mqtt's own view of the stream, from `stream/state`.
        self._stream_state_on: bool = False
        #: Set by the broker thread when a trigger arrives, so the async side can pick
        #: it up without the broker thread having to touch the event loop.
        self._escalate_requested: bool = False
        #: Whether ring-mqtt currently reports an active motion event. While true the
        #: live tier is held open regardless of `_live_until`, so the bridge's
        #: `motion_duration` decides the event's length rather than our hold setting.
        self._motion_active: bool = False
        #: Serialises escalate/de-escalate so two triggers landing together cannot
        #: build two ffmpeg readers (and two relay leases) for one camera.
        self._tier_lock: Optional[asyncio.Lock] = None
        self._fps_ceiling: float = 0.0

    def __repr__(self):
        tier = "live" if self._live_provider is not None else "idle"
        return f"RingMqttImageProvider [{self._camera_id} ({tier})]"

    @property
    def live_url(self) -> str:
        return f"rtsp://{self._rtsp_host}:{self._rtsp_port}/{self._camera_id}{LIVE_SUFFIX}"

    ################################################################################
    # Lifecycle
    ################################################################################

    async def start(self) -> None:
        """Connect to the broker and subscribe to the camera's topics.

        Idempotent, and the only place the event loop is captured. Providers are
        constructed off the loop — `SourceHelper.get_image_provider` is a blocking DB
        call every caller hands to a worker thread — so `get_running_loop()` in
        `__init__` would raise on a worker thread and make the source impossible to
        open at all.
        """
        self._loop = asyncio.get_running_loop()
        if self._tier_lock is None:
            self._tier_lock = asyncio.Lock()
        if self._mqtt_client is None:
            # Connecting blocks (five retries, two seconds apart), so it goes to a
            # thread rather than stalling the loop of a watcher servicing other cameras.
            self._mqtt_client = await asyncio.to_thread(MqttClient, self._broker_address, "/", self._broker_port)
            self._subscribed = False
        if not self._subscribed:
            self._mqtt_client.subscribe(f"{self._base}/snapshot/image", self._on_snapshot, absoluteTopic=True)
            self._mqtt_client.subscribe(f"{self._base}/motion/state", self._on_motion, absoluteTopic=True)
            self._mqtt_client.subscribe(f"{self._base}/stream/state", self._on_stream_state, absoluteTopic=True)
            self._mqtt_client.subscribe(f"{self._base}/stream/command", self._on_stream_command, absoluteTopic=True)
            self._subscribed = True

    async def stop(self) -> None:
        """Release everything, and put the camera back to sleep if we woke it."""
        await self._deescalate()
        if self._mqtt_client is not None:
            for suffix in ("snapshot/image", "motion/state", "stream/state", "stream/command"):
                try:
                    self._mqtt_client.unsubscribe(f"{self._base}/{suffix}")
                except Exception:
                    # Unsubscribing a topic that was never subscribed (a partially
                    # failed start) must not mask the caller's own teardown.
                    logger.debug("Could not unsubscribe %s/%s", self._base, suffix, exc_info=True)
            self._mqtt_client.disconnect()
            self._mqtt_client = None
        self._subscribed = False
        # Cleared so a reconnect() — which is stop() then start() — cannot inherit a
        # stale "motion is running" flag and hold the next session live forever.
        self._motion_active = False
        # A snapshot delivered after this point has no loop to be posted to.
        self._loop = None

    def is_ready(self) -> bool:
        """Ready as soon as the broker subscription exists.

        Notably *not* gated on the live tier being up: the idle tier can serve frames
        on its own, and a camera that is merely asleep is working exactly as intended.
        """
        return self._subscribed

    def is_connecting(self) -> bool:
        """Only the live tier can be mid-connect; the idle tier is either up or not."""
        return self._live_provider is not None and self._live_provider.is_connecting()

    #: What to promise while the live reader is still coming up. Measured cold-start to
    #: first frame is ~8.5s through the relay; this is generous enough that a normal
    #: connect never looks like a stall.
    CONNECT_EXPECTATION_S = 30.0

    #: Fallback cadence before enough snapshots have arrived to measure one.
    #: ring-mqtt's documented default is 90s; the deployment this was built against
    #: runs 30s. Assume the slower of the two — guessing fast reports a healthy camera
    #: as overdue, guessing slow only delays a real complaint.
    ASSUMED_SNAPSHOT_CADENCE_S = 90.0

    #: Multiplier on the snapshot cadence before the next frame is "expected". Ring's
    #: interval is not tightly periodic and a cloud snapshot can slip a cycle, so a
    #: deadline at exactly one cadence would cry wolf routinely.
    CADENCE_SLACK = 2.0

    def next_frame_expected_by(self) -> Optional[float]:
        """When the next frame should have arrived, given the tier we are in now.

        Three different answers, which is the whole reason this is a deadline and not
        an interval:

        * **live** — frames arrive at stream rate, so anything beyond a couple of
          seconds is a genuine stall worth noticing quickly.
        * **connecting** — neither tier is producing. Not a stall; just wait.
        * **idle** — the next cloud snapshot, from the cadence measured off the
          bridge, with slack.
        """
        now = self._now()
        if self._live_provider is not None:
            if self._live_provider.is_connecting():
                return now + self.CONNECT_EXPECTATION_S
            return now + 2.0
        cadence = self._snapshot_cadence_s or self.ASSUMED_SNAPSHOT_CADENCE_S
        base = self._last_snapshot_at if self._last_snapshot_at is not None else now
        deadline = base + cadence * self.CADENCE_SLACK
        # Never hand back a deadline already in the past: a consumer deriving a wait
        # budget from it would get zero and time out instantly on a source that is
        # merely overdue.
        return max(deadline, now + cadence)

    def set_source_fps_ceiling(self, fps: float) -> None:
        """Remembered, and applied to each live reader as it is built.

        Stored rather than forwarded immediately because the reader that will honour
        it usually does not exist yet — the camera is asleep when a consumer sets its
        ceiling, and the ffmpeg reader is only spawned on escalation.
        """
        self._fps_ceiling = fps
        if self._live_provider is not None:
            self._live_provider.set_source_fps_ceiling(fps)

    ################################################################################
    # MQTT callbacks — all of these run on the broker's thread
    ################################################################################

    def _on_snapshot(self, mqtt_msg) -> None:
        """Decode a published JPEG and queue it as the idle tier's next frame."""
        try:
            loop = self._loop
            if loop is None:
                # Reachable only if the subscription outlives the loop it was bound to
                # (i.e. after stop()). Dropping is right: nothing is waiting.
                logger.debug("Dropping a Ring snapshot received while no event loop is bound")
                return
            raw = mqtt_msg.payload
            image = Image.open(io.BytesIO(raw))
            # ring-mqtt publishes RGB JPEGs; the rest of this app treats a frame as
            # BGR, so the swap has to happen here. Getting it wrong is silent —
            # nothing downstream can tell — and shows up only as worse detections.
            frame = np.array(image.convert("RGB"))[..., ::-1]
            self._note_snapshot_arrival()
            asyncio.run_coroutine_threadsafe(self._offer_snapshot(frame, raw), loop)
        except Exception:
            logger.error("Failed to decode a Ring snapshot", exc_info=True)

    def _on_motion(self, mqtt_msg) -> None:
        """Escalate for as long as ring-mqtt says the motion event is running.

        Ring holds `motion/state` ON for a fixed window after the event starts —
        180s by default, and tunable at the bridge via `motion_duration/command`. So
        the event's length is ring-mqtt's to decide, not ours, and a provider that
        escalated for its own `live_hold_s` instead would cut every event short the
        moment that setting was the smaller of the two (which it is, by default:
        120s against 180s). Tracking the flag rather than starting a fixed timer
        means the bridge's setting is honoured without being duplicated here.

        `live_hold_s` still applies, but as a *grace period after* the event ends,
        which is what it is useful for: motion clearing does not mean the subject has
        left the frame.
        """
        if not self._live_on_motion:
            return
        on = mqtt_msg.payload.decode(errors="replace").strip().upper() == "ON"
        self._motion_active = on
        if on:
            logger.info("Ring camera %s reported motion; escalating to live", self._camera_id)
            self._request_escalation()
        else:
            # Start the grace period from the moment motion cleared, not from the
            # moment it began.
            self._live_until = self._now() + self._live_hold_s

    def _on_stream_state(self, mqtt_msg) -> None:
        """Track whether ring-mqtt currently has a stream up. Never a trigger.

        It is tempting to escalate here — an `OFF`→`ON` transition looks exactly like
        "somebody wants video". It isn't. `stream/state` also reflects the RTSP
        server's own on-demand churn, which is not a request by anyone: measured on a
        live system, all three cameras cycled `OFF`→`ON` roughly every 20 seconds
        (~70 transitions per camera in 14 minutes) with no MQTT command involved at
        all. Escalating on that would wake the camera and run inference on a full-rate
        1080p stream for `live_hold_s` out of every 20 seconds, forever, without
        anybody having asked for a single frame.

        So state is recorded and nothing more; `_on_stream_command` is the trigger.
        """
        self._stream_state_on = mqtt_msg.payload.decode(errors="replace").strip().upper() == "ON"

    def _on_stream_command(self, mqtt_msg) -> None:
        """Escalate when *someone deliberately asks* for this camera's stream.

        This is what makes an external trigger work without any RPC into this process:
        a UI button (or Home Assistant, or `mosquitto_pub`) publishes `stream/command`,
        and every provider watching this camera picks the request up here. A stream
        someone else started is therefore joined rather than duplicated.

        The command topic rather than the state topic precisely because a command is
        an *intent*, and only ever published by something that wants video. Over 15
        minutes of the on-demand churn described above, not one `stream/command`
        message was published — so this discriminates the two cases exactly.

        Only `on` is acted on. An `off` from someone else is deliberately ignored: the
        live window is ours to end, and honouring another consumer's `off` would let
        one client cut short a stream another is mid-way through reading. The hold
        timer is what returns us to idle.
        """
        command = mqtt_msg.payload.decode(errors="replace").strip().lower()
        if command.startswith("on"):
            logger.info("Ring camera %s stream requested; escalating to live", self._camera_id)
            self._request_escalation()

    def _request_escalation(self) -> None:
        """Flag a trigger from the broker thread and extend the live window.

        A flag rather than a task: this runs on paho's thread, and the escalation
        itself has to happen on the event loop. `getNextImage` picks the flag up on
        its next pass, which is also the only place that knows whether the caller
        actually still wants frames.
        """
        self._escalate_requested = True
        self._live_until = self._now() + self._live_hold_s

    def _note_snapshot_arrival(self) -> None:
        """Learn the bridge's snapshot cadence from consecutive arrivals.

        Runs on the broker thread. Uses the max of the observed gaps rather than the
        latest one, because a motion event injects an out-of-band snapshot
        (`attributes.type` flips to "motion") that is far closer to its predecessor
        than the interval — taking the newest gap would teach us a cadence of two
        seconds and then report a perfectly healthy idle camera as overdue.
        """
        now = self._now()
        previous, self._last_snapshot_at = self._last_snapshot_at, now
        if previous is None:
            return
        gap = now - previous
        if gap <= 0:
            return
        if self._snapshot_cadence_s is None or gap > self._snapshot_cadence_s:
            self._snapshot_cadence_s = gap

    async def _offer_snapshot(self, frame: np.ndarray, raw: bytes) -> None:
        """Queue a snapshot, discarding the oldest if the consumer has fallen behind."""
        while self._snapshot_queue.full():
            try:
                self._snapshot_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._snapshot_queue.put_nowait((frame, raw))

    ################################################################################
    # Tier transitions
    ################################################################################

    async def request_live(self, hold_s: Optional[float] = None) -> None:
        """Wake the camera and hold it live for `hold_s` (default `live_hold_s`).

        The externally callable trigger. Publishing the command is enough on its own —
        `_on_stream_state` will see ring-mqtt acknowledge it — but the window is
        extended here too so that a request for a stream that is *already* up still
        keeps it up.
        """
        self._live_until = self._now() + (hold_s if hold_s is not None else self._live_hold_s)
        self._escalate_requested = True
        await self._publish_stream_command("on")
        self._we_requested_live = True

    async def release_live(self) -> None:
        """Drop back to the idle tier immediately, putting the camera back to sleep."""
        self._live_until = 0.0
        await self._deescalate()

    async def _publish_stream_command(self, command: str) -> None:
        if self._mqtt_client is None:
            return
        try:
            await asyncio.to_thread(
                self._mqtt_client.publish,
                f"{self._base}/stream/command",
                command,
                False,
                True,
            )
        except Exception:
            logger.error("Failed to publish stream/%s for %s", command, self._camera_id, exc_info=True)

    async def _escalate(self) -> None:
        """Bring the live tier up: ask ring-mqtt for the stream, then read it.

        Note what this deliberately does *not* do: wait for `stream/state` to confirm
        the camera is up before connecting. That gate looks obvious and is wrong.
        `stream/state` is published non-retained and only *on change*, so a provider
        that escalates while the stream is already running — which is the common case,
        since Home Assistant, Frigate or another watcher may already be holding this
        camera — receives no message at all and would time out waiting for news about
        a stream it could have read immediately.

        Connecting to the RTSP path is itself the on-demand trigger (ring-mqtt's RTSP
        server runs a script on first reader that starts the stream), so simply
        attempting the read is both the wake-up and the test of whether it worked.
        `stream/state` is left as an opportunistic *reason* to escalate, never as a
        precondition for doing so.
        """
        assert self._tier_lock is not None
        async with self._tier_lock:
            if self._live_provider is not None:
                return
            # Best-effort statement of intent. The RTSP connect below would wake the
            # camera on its own, but going through the documented command topic is
            # what makes `stream/state` reflect our claim, so anything else watching
            # this camera can see it.
            await self._publish_stream_command("on")
            self._we_requested_live = True

            provider = FfmpegImageProvider(self.live_url, loop=True, rtsp_relay=True)
            if self.source_name:
                provider.set_source_identity(self.source_name)
            if self._fps_ceiling > 0:
                provider.set_source_fps_ceiling(self._fps_ceiling)
            try:
                await provider.start()
            except Exception:
                # A failed start has already handed back any relay lease it took, so
                # there is nothing to clean up here beyond not keeping the provider.
                logger.error("Could not open the live stream for %s", self._camera_id, exc_info=True)
                return
            self._live_provider = provider
            logger.info("Ring camera %s is live at %s", self._camera_id, self.live_url)

    async def _deescalate(self) -> None:
        """Tear the live tier down and let the camera sleep."""
        assert self._tier_lock is not None or self._live_provider is None
        if self._tier_lock is None:
            return
        async with self._tier_lock:
            if self._live_provider is not None:
                provider, self._live_provider = self._live_provider, None
                try:
                    await provider.stop()
                except Exception:
                    logger.error("Failed to stop the live reader for %s", self._camera_id, exc_info=True)
            if self._we_requested_live:
                # Only ours to cancel — we never publish `off` for a stream we did not
                # ask for. This is safe even when something else is also watching:
                # ring-mqtt's RTSP server re-triggers on demand for any remaining
                # reader, so `off` releases our claim rather than cutting anyone's
                # video (observed directly — a stream with another reader attached
                # returns to `active` about a second after being told to stop).
                await self._publish_stream_command("off")
                self._we_requested_live = False

    ################################################################################
    # Frames
    ################################################################################

    async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
        """Return the next frame from whichever tier is currently appropriate."""
        if self._loop is None:
            # A caller that skipped start() still gets frames rather than silence.
            await self.start()

        # Self-pace exactly as UrlImageProvider does. This matters most on the live
        # tier: without it an escalated camera hands the caller ~20fps of 1080p, and
        # a collector wanting a frame every few seconds gets a flood.
        if self._min_capture_interval > 0:
            now = self._now()
            if now < self._next_capture_time:
                await self._sleep(self._next_capture_time - now)
            self._next_capture_time = self._now() + self._min_capture_interval

        if self._escalate_requested:
            self._escalate_requested = False
            await self._escalate()
        elif self._live_provider is not None and not self._motion_active and self._now() >= self._live_until:
            # `_motion_active` gates this so an event that outlasts `live_hold_s` is
            # not cut off half way through — the grace period only starts once
            # ring-mqtt says the motion has actually stopped.
            logger.info("Ring camera %s live window expired; returning to snapshots", self._camera_id)
            await self._deescalate()

        if self._live_provider is not None:
            try:
                return await self._live_provider.getNextImage()
            except Exception:
                # The live reader failing is not the source failing: the idle tier is
                # still perfectly good, and dropping back to it keeps the camera
                # producing frames instead of reporting a dead source.
                logger.warning("Live read failed for %s; falling back to snapshots", self._camera_id, exc_info=True)
                await self._deescalate()

        return await self._next_snapshot()

    async def _next_snapshot(self) -> ImageWithProviderMetadata:
        try:
            frame, raw = await asyncio.wait_for(self._snapshot_queue.get(), timeout=self.IDLE_FRAME_TIMEOUT)
        except asyncio.TimeoutError:
            raise TimeoutError(f"No snapshot from Ring camera {self._camera_id} within {self.IDLE_FRAME_TIMEOUT:.0f}s")

        frame_idx = self._frame_idx
        self._frame_idx += 1
        provided = ImageWithProviderMetadata(
            frame,
            metadata=ImageProviderMetadata(
                source=self.source_name,
                frame_idx=frame_idx,
                additional_info={"tier": "snapshot", "cameraId": self._camera_id},
            ),
        )
        # Keep the camera's own JPEG rather than re-encoding the decoded array. This is
        # the full-resolution image Ring produced; passing it on is what makes a clip
        # built from snapshots full-res, and it saves an encode for any consumer that
        # wanted JPEG anyway.
        if raw[:2] == b"\xff\xd8":
            provided.encoded_b64 = json.dumps(
                {"format": "jpeg", "metadata": {}, "data": base64.b64encode(raw).decode("ascii")}
            )
        return provided

    ################################################################################
    # Schema
    ################################################################################

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        return {
            "broker_address": {
                "type": "string",
                "label": "Broker Address",
                "required": True,
                "description": "Address of the MQTT broker ring-mqtt publishes to",
            },
            "broker_port": {
                "type": "int",
                "label": "Broker Port",
                "default": 1883,
                "required": False,
                "description": "MQTT broker port",
            },
            "location_id": {
                "type": "string",
                "label": "Ring Location ID",
                "required": True,
                "description": "The UUID in ring/<location>/camera/... — one per Ring location",
            },
            "camera_id": {
                "type": "string",
                "label": "Camera ID",
                "required": True,
                "description": "The camera's ring-mqtt device id, e.g. 9c7613f4632d",
            },
            "rtsp_host": {
                "type": "string",
                "label": "RTSP Host",
                "required": False,
                "description": "Host serving ring-mqtt's RTSP paths. Defaults to the broker address.",
            },
            "rtsp_port": {
                "type": "int",
                "label": "RTSP Port",
                "default": 8554,
                "required": False,
                "description": "Port of ring-mqtt's RTSP server",
            },
            "live_on_motion": {
                "type": "boolean",
                "label": "Go Live On Motion",
                "default": True,
                "required": False,
                "description": "Wake the camera for full-rate video when ring-mqtt reports motion",
            },
            "live_hold_s": {
                "type": "float",
                "label": "Live Hold (s)",
                "default": 15.0,
                "required": False,
                "description": (
                    "Grace period to stay live after a trigger ends before returning to snapshots. "
                    "A motion event is held live for as long as ring-mqtt reports it (its own "
                    "motion_duration, 180s by default); this applies after that. Keep it short — "
                    "reading the stream keeps the camera awake, and an awake Ring camera raises no "
                    "motion events."
                ),
            },
            "min_capture_interval": {
                "label": "Min Capture Interval (s)",
                "type": "float",
                "required": False,
                "default": 0.0,
                "description": (
                    "Minimum seconds between frames returned by this source. "
                    "0 = no source-level limit (the task's own interval applies)."
                ),
            },
        }
