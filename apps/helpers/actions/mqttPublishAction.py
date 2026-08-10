"""Publish event transitions to MQTT.

Publish-only. Inbound commands (arming a watcher, acknowledging an event) are a
plausible later addition; the seam for them is `AsyncMqttPublisher.subscribe`, which
is deliberately left unbuilt rather than designed out.

Topic scheme — documented and stable, because Home Assistant users will build
automations on it::

    <prefix>/<profile>/<source>/object/<label>   retain, JSON: {trackId, confidence, bbox, ...}
    <prefix>/<profile>/<source>/event/<rule>     JSON: {eventUuid, phase, startedAt, ...}
    <prefix>/state                               retain: online|offline (LWT)

Retain is used for *state-shaped* topics (what is true right now) and not for event
edges (what just happened) — a retained "event started" would replay to every new
subscriber as though it were live.
"""

import re
from typing import Any, Dict

from ..watch.events import WatchEventSignal
from .action import Action, ActionContext, ActionResult
from .Registry import register_action

_TOPIC_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


def slugify_topic(value: str) -> str:
    """Make a name safe for one MQTT topic level.

    `/` and `+`/`#` in a name would silently reshape the topic tree, so they are
    collapsed rather than escaped.
    """
    slug = _TOPIC_SAFE.sub("-", (value or "").strip().lower()).strip("-")
    return slug or "unnamed"


@register_action("publish_mqtt")
class MqttPublishAction(Action):
    """Publishes a JSON payload for each event transition."""

    def __init__(self, retain: bool = False, qos: int = 0, publisher: Any = None, **params: Any) -> None:
        super().__init__(**params)
        self._retain = bool(retain)
        self._qos = int(qos)
        # Injected by the task, which owns the connection: one broker connection per
        # watcher, not one per action.
        self._publisher = publisher

    @classmethod
    def display_name(cls) -> str:
        return "Publish MQTT"

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "retain": {
                "type": "bool",
                "required": False,
                "default": False,
                "help": (
                    "Retain the message on the broker. Appropriate for state-shaped topics; "
                    "a retained event edge would replay to every new subscriber as if live."
                ),
            },
            "qos": {"type": "int", "required": False, "default": 0, "options": [0, 1, 2], "help": "MQTT QoS level."},
        }

    def topic_for(self, signal: WatchEventSignal, context: ActionContext) -> str:
        profile = slugify_topic(context.profile_name or context.profile_uuid)
        source = slugify_topic(context.source_name or context.source_uuid or "unknown")
        rule = slugify_topic(signal.metadata.get("eventLabel") or signal.rule_name)
        return f"{profile}/{source}/event/{rule}"

    def payload_for(self, signal: WatchEventSignal, context: ActionContext) -> Dict[str, Any]:
        payload = signal.to_dict()
        payload["profileUuid"] = context.profile_uuid
        payload["sourceName"] = context.source_name
        payload["eventUuid"] = context.extra.get("event_uuid")
        payload["recordingUuid"] = context.extra.get("recording_uuid")
        return payload

    async def run(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        if self._publisher is None:
            return ActionResult(ok=False, detail="No MQTT broker configured in Settings")
        topic = self.topic_for(signal, context)
        try:
            # Enqueue only — never awaits the broker, so a slow or dead broker cannot
            # stall the frame loop that called this.
            self._publisher.publish(topic, self.payload_for(signal, context), retain=self._retain, qos=self._qos)
            return ActionResult(ok=True, detail=f"queued {topic}", data={"topic": topic})
        except Exception as e:
            return ActionResult(ok=False, detail=str(e))
