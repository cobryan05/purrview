import logging
import sys
import time
from dataclasses import dataclass
from typing import Callable

import paho.mqtt.client as mqtt

from apps.helpers.logUtils import LOG_LEVEL

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("MqttClient")
logger.setLevel(LOG_LEVEL)


class MqttClient:
    @dataclass
    class SubMapData:
        callback: Callable[[mqtt.MQTTMessage], None]
        #: The topic actually sent to the broker, prefix already applied (or not, for an
        #: absolute subscription). Kept because the map is keyed on the *caller's* topic,
        #: which is not the wire topic for a relative subscription — resubscribing or
        #: unsubscribing off the key alone addresses a topic that was never subscribed,
        #: and both failures are silent (the topic simply goes quiet / stays live).
        wire_topic: str = ""

    KEY_MQTT_TOPIC = "mqtt-topic"
    KEY_MQTT_RETAIN = "mqtt-retain"

    KEEPALIVE_TIME = 60
    CONNECTION_RETRIES = 5
    CONNECTION_RETRY_DELAY = 2

    def __init__(self, broker_address: str, prefix: str = "myhome/", broker_port: int = 1883):
        self._prefix: str = prefix.rstrip("/")
        self._mqtt: mqtt.Client = mqtt.Client()
        self._mqtt.on_connect = self.mqtt_connected_callback
        self._mqtt.on_message = self.mqtt_message_callback
        self._mqtt.on_disconnect = self.mqtt_disconnect_callback
        self._subMap: dict[str, MqttClient.SubMapData] = {}

        remaining_tries = MqttClient.CONNECTION_RETRIES
        while True:
            try:
                self._mqtt.connect(broker_address, broker_port, keepalive=MqttClient.KEEPALIVE_TIME)
                break
            except Exception as e:
                remaining_tries -= 1
                logger.warning(
                    "Failed to connect to broker: {}.  {} retries remaining.".format(str(e), remaining_tries)
                )
                if remaining_tries == 0:
                    raise
                time.sleep(MqttClient.CONNECTION_RETRY_DELAY)
        self._mqtt.loop_start()

    def __del__(self):
        self.disconnect()
        logger.debug("Deleting mqtt client from MQTT broker")

    def disconnect(self):
        self._mqtt.disconnect()

    def publish(self, topic: str, value: str, retain: bool = False, absoluteTopic=False):
        if absoluteTopic:
            publish_topic = topic
        else:
            publish_topic = "{}/{}".format(self._prefix, topic)
        logger.debug(f"Publishing {publish_topic} value of {value if isinstance(value, str) else 'non-string type'}")
        self._mqtt.publish(publish_topic, value, retain=retain)

    def subscribe(self, topic: str, callback: Callable[[str], None], absoluteTopic=False):
        if absoluteTopic:
            subscribe_topic = topic
        else:
            subscribe_topic = "{}/{}".format(self._prefix, topic)
        assert topic not in self._subMap

        self._subMap[topic] = MqttClient.SubMapData(callback=callback, wire_topic=subscribe_topic)
        self._mqtt.subscribe(subscribe_topic)

    def unsubscribe(self, topic: str):
        # Off the recorded wire topic, not a re-derived one: an absolute subscription
        # carries no prefix, so re-deriving unsubscribed the wrong topic and left the
        # broker happily publishing to a client that had already dropped the callback.
        data = self._subMap.pop(topic, None)
        self._mqtt.unsubscribe(data.wire_topic if data is not None else "{}/{}".format(self._prefix, topic))

    def mqtt_connected_callback(self, client: mqtt.Client, userdata, flags, rc):
        logger.info("MQTT server connect: {}".format(rc))
        # Reconnect subscriptions.
        #
        # Over a copy, because this runs on paho's thread while the owning code is
        # usually still adding subscriptions from its own: `connect()` returns as soon
        # as the socket is up, so a caller that constructs a client and immediately
        # subscribes races the CONNACK. Iterating the live dict raises "dictionary
        # changed size during iteration" *inside* the connack handler, which kills it
        # part-way — so the subscriptions it had not reached yet are silently never
        # registered and that topic goes quiet for the life of the client. The failure
        # looks like a source that never publishes, not like an error.
        #
        # `mqtt_message_callback` already copies for the same reason; this path was
        # simply missed. A provider subscribing to a single topic almost always wins
        # the race, which is why it survived this long.
        #
        # `wire_topic`, not the map key: the key is the topic the *caller* asked for,
        # which for a relative subscription is missing the prefix. Resubscribing off the
        # key put every relative subscription on the wrong topic after a reconnect, so
        # it went quiet for the rest of the session with nothing logged.
        for subTopic, subData in self._subMap.copy().items():
            self._mqtt.subscribe(subData.wire_topic or subTopic)

    def mqtt_disconnect_callback(self, client: mqtt.Client, userdata, rc):
        logger.info("Mqtt disconnected: {}".format(rc))
        if rc != 0:
            remaining_tries = MqttClient.CONNECTION_RETRIES
            while True:
                try:
                    logger.info("Attempting reconnect...")
                    self._mqtt.reconnect()
                    break
                except Exception as e:
                    remaining_tries -= 1
                    logger.warning(
                        "Failed to connect to broker: {}.  {} retries remaining.".format(str(e), remaining_tries)
                    )
                    if remaining_tries == 0:
                        raise
                    time.sleep(MqttClient.CONNECTION_RETRY_DELAY)
        else:
            logger.debug("Stopping Mqtt loop")
            self._mqtt.loop_stop()

    def mqtt_message_callback(self, client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
        # print(f"Received topic|message: {msg.topic}|{msg.payload.decode()}")
        for topic, data in self._subMap.copy().items():
            prefixedTopic = f"{self._prefix}/{topic.rstrip('#')}"
            if msg.topic.startswith(prefixedTopic) and callable(data.callback):
                data.callback(msg)
            else:  # TODO: Deal with prefix?
                topic = f"{topic.rstrip('#')}"
                if msg.topic.startswith(topic) and callable(data.callback):
                    data.callback(msg)
