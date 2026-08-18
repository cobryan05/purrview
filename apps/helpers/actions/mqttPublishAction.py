"""Publish event transitions to MQTT.

Publish-only. Inbound commands (arming a watcher, acknowledging an event) are a
plausible later addition; the seam for them is `AsyncMqttPublisher.subscribe`, which
is deliberately left unbuilt rather than designed out.

Topic scheme — documented and stable, because Home Assistant users will build
automations on it::

    <prefix>/<profile>/<source>/object/<label>   retain, JSON: {trackId, confidence, bbox, ...}
    <prefix>/<profile>/<source>/event/<rule>     JSON: {eventUuid, phase, startedAt, ...}
    <prefix>/state                               retain: online|offline (LWT)

That scheme is the *default* topic template, not a hard-coded path. A rule matching
several labels and several zones knows which ones actually fired, and the whole point of
exposing those slot values is to be able to route on them — `cats/{rule}/{source}/{subject}`
puts one cat per topic. See `apps.helpers.watch.templating` for the vocabulary.

Retain is used for *state-shaped* topics (what is true right now) and not for event
edges (what just happened) — a retained "event started" would replay to every new
subscriber as though it were live.
"""

from typing import Any, Dict

from ..watch.events import WatchEventSignal
from ..watch.templating import build_template_params, describe_params, render_topic, slugify
from .action import Action, ActionContext, ActionResult
from .Registry import register_action

#: Reproduces the historical hard-coded scheme exactly, so an action saved before topic
#: templates existed publishes to the same place it always did.
DEFAULT_TOPIC_TEMPLATE = "{watcher}/{source}/event/{rule}"


def slugify_topic(value: str) -> str:
    """Make a name safe for one MQTT topic level.

    Kept as a re-export: the implementation moved to `watch.templating` when the topic
    became a template, and callers elsewhere still import it from here.
    """
    return slugify(value)


@register_action("publish_mqtt")
class MqttPublishAction(Action):
    """Publishes a JSON payload for each event transition."""

    def __init__(
        self,
        retain: bool = False,
        qos: int = 0,
        topic_template: str = DEFAULT_TOPIC_TEMPLATE,
        publisher: Any = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._retain = bool(retain)
        self._qos = int(qos)
        # A blank template means "unset", not "publish to the empty topic".
        self._topic_template = (topic_template or "").strip() or DEFAULT_TOPIC_TEMPLATE
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
            "topic_template": {
                "type": "str",
                "required": False,
                "default": DEFAULT_TOPIC_TEMPLATE,
                "help": (
                    "Topic to publish to. Placeholders are substituted and slugified per level. "
                    f"Available: {describe_params()}. An unknown placeholder is left as-is so a "
                    "typo is visible on the broker rather than silently collapsing a level."
                ),
            },
        }

    def topic_for(self, signal: WatchEventSignal, context: ActionContext) -> str:
        return render_topic(self._topic_template, build_template_params(signal, context))

    def payload_for(self, signal: WatchEventSignal, context: ActionContext) -> Dict[str, Any]:
        payload = signal.to_dict()
        payload["profileUuid"] = context.profile_uuid
        payload["sourceName"] = context.source_name
        payload["eventUuid"] = context.extra.get("event_uuid")
        payload["recordingUuid"] = context.extra.get("recording_uuid")
        # The same slot values the topic was built from, so a subscriber that wants to
        # route on payload rather than topic shape doesn't have to parse the topic back
        # apart.
        payload["templateParams"] = build_template_params(signal, context)
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
