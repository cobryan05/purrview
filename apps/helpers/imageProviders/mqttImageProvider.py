"""Async URL-backed image source class"""

import asyncio
import io
import logging
from typing import Any, Optional

import numpy as np
from PIL import Image

from apps.helpers.mqttClient import MqttClient

from .imageProvider import ImageProvider, ImageWithMetadata
from .Registry import register_image_provider

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@register_image_provider()
class MqttImageProvider(ImageProvider):
    TIMEOUT = 60

    def __init__(self, broker_address: str, broker_port: int, topic: str):
        """Initializes the MQTT source."""
        self._topic: str = topic
        self._mqtt_client: MqttClient = MqttClient(broker_address, "/", int(broker_port))
        self._frameQueue: asyncio.Queue = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._mqtt_client.subscribe(self._topic, self._pushFrame, absoluteTopic=True)

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
            logger.error(e, exc_info=True)

    async def getNextImage(self) -> Optional[ImageWithMetadata]:
        """Asynchronously retrieves the next image from the queue."""
        try:
            frame: np.ndarray = await asyncio.wait_for(self._frameQueue.get(), timeout=MqttImageProvider.TIMEOUT)
            # Swap channel ordering
            frame_rgb = frame[..., ::-1]
            return ImageWithMetadata(frame_rgb)
        except asyncio.TimeoutError:
            raise TimeoutError("No image received within the timeout period.")

    async def stop(self) -> None:
        """Stops the image provider and releases any resources"""
        if self._mqtt_client:
            self._mqtt_client.unsubscribe(self._topic)
            self._mqtt_client.disconnect()
            self._mqtt_client = None

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
            "broker_port": {"type": "int", "label": "Broker Port", "default": 1883, "required": True, "description": "MQTT broker port"},
            "topic": {
                "type": "string",
                "label": "Mqtt Topic",
                "required": True,
                "description": "MQTT topic to subscribe to",
            },
        }
