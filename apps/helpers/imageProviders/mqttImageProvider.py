"""Async URL-backed image source class"""

import asyncio
import io
import logging

import numpy as np
from PIL import Image

from apps.helpers.mqttClient import MqttClient

from .imageProvider import ImageProvider

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class MqttImageProvider(ImageProvider):
    TIMEOUT = 60

    def __init__(self, mqtt_client: MqttClient, topic: str):
        """Initializes the MQTT source."""
        self._topic: str = topic
        self._mqtt_client: MqttClient = mqtt_client
        self._frameQueue: asyncio.Queue = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        mqtt_client.subscribe(self._topic, self._pushFrame, absoluteTopic=True)

    def __del__(self):
        """Unsubscribe from the MQTT topic when deleted."""
        self._mqtt_client.unsubscribe(self._topic)

    def __repr__(self):
        return f"MqttImageProvider [{self._topic}]"

    def _pushFrame(self, mqttMsg):
        """Callback for handling incoming MQTT messages."""
        try:
            png_data = mqttMsg.payload
            image = Image.open(io.BytesIO(png_data))
            image_bytes = np.array(image)
            asyncio.run_coroutine_threadsafe(self._frameQueue.put(image_bytes), self._loop)
        except Exception as e:
            logger.exception(e)

    async def getNextImage(self) -> np.array:
        """Asynchronously retrieves the next image from the queue."""
        try:
            return await asyncio.wait_for(self._frameQueue.get(), timeout=MqttImageProvider.TIMEOUT)
        except asyncio.TimeoutError:
            raise TimeoutError("No image received within the timeout period.")