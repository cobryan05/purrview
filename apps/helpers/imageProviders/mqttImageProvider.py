"""Async URL-backed image source class"""

import asyncio
import io
import logging
from typing import Any, Optional

import numpy as np
from PIL import Image

from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.mqttClient import MqttClient

from .imageProvider import ImageProvider, ImageProviderType, ImageWithProviderMetadata
from .Registry import register_image_provider

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


@register_image_provider()
class MqttImageProvider(ImageProvider):
    TIMEOUT = 60
    # A LIVE_STREAM provider must not hand a consumer an old frame. A watcher paces itself
    # at its analysis fps, so on any source publishing faster than that an unbounded FIFO
    # grows forever and every frame it yields is further behind reality — an NVR quietly
    # detecting what happened ten minutes ago. FfmpegImageProvider drains to the newest
    # frame for the same reason; this keeps a couple in hand only to absorb jitter.
    QUEUE_SIZE = 4

    @classmethod
    def provider_type(cls) -> ImageProviderType:
        return ImageProviderType.LIVE_STREAM

    def __init__(self, broker_address: str, broker_port: int, topic: str):
        """Initializes the MQTT source."""
        self._topic: str = topic
        # Kept so start() can rebuild the client after a stop(): the base reconnect() is
        # stop-then-start, and a provider that cannot be restarted goes silent for good
        # the first time a watcher decides its source is stale.
        self._broker_address: str = broker_address
        self._broker_port: int = int(broker_port)
        # Connecting blocks (five retries, two seconds apart) and this constructor already
        # runs on a worker thread, so this is the right place for it. start() rebuilds on
        # a thread for the same reason.
        self._mqtt_client: Optional[MqttClient] = MqttClient(broker_address, "/", self._broker_port)
        self._frameQueue: asyncio.Queue = asyncio.Queue(maxsize=MqttImageProvider.QUEUE_SIZE)
        self._dropped_frames: int = 0
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

    def _pushFrame(self, mqttMsg):
        """Callback for handling incoming MQTT messages. Runs on the broker's thread."""
        try:
            loop = self._loop
            if loop is None:
                # Only reachable if the subscription outlives the loop it was bound to
                # (i.e. after stop()). Dropping is right: nothing is waiting on the queue.
                logger.debug("Dropping an MQTT frame received while no event loop is bound")
                return
            png_data = mqttMsg.payload
            image = Image.open(io.BytesIO(png_data))
            image_bytes = np.array(image)
            # `put`, not `put_nowait`, would block this coroutine forever once the queue
            # filled — one pending future per published frame, which is the leak the
            # bound was meant to prevent.
            asyncio.run_coroutine_threadsafe(self._offer(image_bytes), loop)
        except Exception as e:
            logger.error(e, exc_info=True)

    async def _offer(self, frame: np.ndarray) -> None:
        """Queue a frame, discarding the oldest when the consumer has fallen behind."""
        while self._frameQueue.full():
            try:
                self._frameQueue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._dropped_frames += 1
            if self._dropped_frames % 100 == 1:
                # Lag is otherwise invisible: the consumer keeps getting frames, just
                # increasingly old ones.
                logger.warning(
                    "%s is publishing faster than it is being read; dropped %d frame(s) so far",
                    self,
                    self._dropped_frames,
                )
        self._frameQueue.put_nowait(frame)

    async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
        """Asynchronously retrieves the next image from the queue."""
        if self._loop is None:
            # A caller that skipped start() still gets frames rather than silence.
            await self.start()
        try:
            frame: np.ndarray = await asyncio.wait_for(self._frameQueue.get(), timeout=MqttImageProvider.TIMEOUT)
            # Swap channel ordering
            frame_rgb = frame[..., ::-1]
            return ImageWithProviderMetadata(frame_rgb)
        except asyncio.TimeoutError:
            raise TimeoutError("No image received within the timeout period.")

    async def stop(self) -> None:
        """Stops the image provider and releases any resources"""
        if self._mqtt_client:
            self._mqtt_client.unsubscribe(self._topic)
            self._mqtt_client.disconnect()
            self._mqtt_client = None
        self._subscribed = False
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
        }
