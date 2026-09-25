"""Publish the current frame, with detections drawn on it, to MQTT.

A separate action kind from `publish_mqtt` rather than a flag on it, because the two
want *opposite* defaults on the two settings that matter:

* **Retain.** An annotated frame is state — "what this camera looks like right now" —
  and a retained image is what makes a Home Assistant MQTT camera show something the
  moment it subscribes. An event edge is the opposite: retaining "event started" replays
  it to every new subscriber as though it were live, which is the bug `publish_mqtt`'s
  own module docstring warns about.
* **Topic.** Event topics are keyed by rule (`.../event/{rule}`); an image topic is
  keyed by source, because the frame belongs to the camera and not to whichever rule
  happened to fire.

Payloads are **raw JPEG bytes**, not base64 inside JSON. That is what an MQTT camera
subscriber expects, and it avoids inflating every frame by a third. The matching JSON
event still goes out on its own topic via `publish_mqtt`, and the two correlate through
`eventUuid` — one payload trying to be both an image and an event serves neither.

**Resolution** is the camera's, not the watcher's. The frame inference sees is
downscaled to `analysis_width` to keep GPU cost down; that is a decision about the
model, and letting it also decide what the live feed looks like is what made this feed
soft. So the full-resolution frame is published and the detections — which are in the
analysis frame's pixel space — are scaled up to match. `max_width` is the knob for
bandwidth, and it applies before the boxes are drawn so their weight suits the size
actually going out.

Combined with the `frame_analysed` trigger this is a **live feed**: one publish per
analysed frame per source, rate-limited here and coalesced in the publisher, so a slow
broker costs stale frames rather than RAM.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from ..imageUtils import annotate_detections, downscale_to_width, encode_jpg_bytes
from ..watch.events import WatchEventSignal
from ..watch.templating import build_template_params, describe_params, render_topic
from .action import Action, ActionContext, ActionResult
from .Registry import register_action

logger = logging.getLogger(__name__)

#: Keyed by source, not by rule — the frame is the camera's, whatever fired on it.
DEFAULT_IMAGE_TOPIC_TEMPLATE = "{watcher}/{source}/image"

#: 0 means "every analysed frame". A live feed is normally wanted at the analysis rate,
#: and the publisher's coalescing already protects the broker, so throttling is opt-in.
DEFAULT_MAX_FPS = 0.0


@register_action("publish_mqtt_image")
class MqttPublishImageAction(Action):
    """Publishes an annotated JPEG of the current frame."""

    def __init__(
        self,
        topic_template: str = DEFAULT_IMAGE_TOPIC_TEMPLATE,
        annotate: bool = True,
        full_resolution: bool = True,
        quality: int = 80,
        max_width: int = 0,
        max_fps: float = DEFAULT_MAX_FPS,
        retain: bool = True,
        qos: int = 0,
        publisher: Any = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        # A blank template means "unset", not "publish to the empty topic".
        self._topic_template = (topic_template or "").strip() or DEFAULT_IMAGE_TOPIC_TEMPLATE
        self._annotate = bool(annotate)
        self._full_resolution = bool(full_resolution)
        self._quality = max(1, min(100, int(quality)))
        self._max_width = max(0, int(max_width))
        self._max_fps = max(0.0, float(max_fps))
        self._retain = bool(retain)
        self._qos = int(qos)
        # Injected by the task, which owns the one broker connection per watcher.
        self._publisher = publisher
        # source uuid → monotonic time of the last publish. Per source, not global: one
        # busy camera must not consume the rate budget of every other camera in the
        # watcher, which a single timestamp would let it do.
        self._last_published: Dict[str, float] = {}

    @classmethod
    def display_name(cls) -> str:
        return "Publish MQTT image"

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "topic_template": {
                "type": "str",
                "required": False,
                "default": DEFAULT_IMAGE_TOPIC_TEMPLATE,
                "help": (
                    "Topic to publish the JPEG to. Placeholders are substituted and slugified "
                    f"per level. Available: {describe_params()}. Keyed by source by default, "
                    "because the frame belongs to the camera rather than to a rule."
                ),
            },
            "annotate": {
                "type": "bool",
                "required": False,
                "default": True,
                "help": "Draw detection boxes and labels on the frame before publishing.",
            },
            "full_resolution": {
                "type": "bool",
                "required": False,
                "default": True,
                "help": (
                    "Publish the frame at the resolution the camera sent, rather than at the "
                    "watcher's analysis width. On by default: the analysis downscale exists to "
                    "make inference cheap, and there is no reason for that to decide how the live "
                    "feed looks. Use max_width to cap bandwidth instead."
                ),
            },
            "quality": {
                "type": "int",
                "required": False,
                "default": 80,
                "help": "JPEG quality, 1-100. Around 90 looks the same as the original to the eye; 70 roughly halves the file at a visible cost on fine detail. Lower it only if disk or bandwidth is the problem.",
            },
            "max_width": {
                "type": "int",
                "required": False,
                "default": 0,
                "help": (
                    "Shrink the frame to at most this width before encoding. 0 keeps whatever "
                    "full_resolution selected. Never upscales."
                ),
            },
            "max_fps": {
                "type": "float",
                "required": False,
                "default": DEFAULT_MAX_FPS,
                "help": (
                    "Ceiling on publishes per second per source. 0 publishes every analysed "
                    "frame. Lower this to feed a dashboard from a watcher that analyses fast."
                ),
            },
            "retain": {
                "type": "bool",
                "required": False,
                "default": True,
                "help": (
                    "Retain the image on the broker. On by default: an annotated frame is state, "
                    "and retaining it is what makes a subscriber show a picture immediately "
                    "instead of waiting for the next one."
                ),
            },
            "qos": {
                "type": "int",
                "required": False,
                "default": 0,
                "options": [0, 1, 2],
                "help": "How hard the broker tries to deliver. 0 sends once and does not check, which is right for a frequent topic where the next message is along shortly. 1 keeps resending until the broker acknowledges, so a message cannot be lost but may arrive twice. 2 guarantees exactly once and is the slowest. Leave at 0 unless something downstream is genuinely missing messages.",
            },
        }

    def topic_for(self, signal: WatchEventSignal, context: ActionContext) -> str:
        return render_topic(self._topic_template, build_template_params(signal, context))

    def _throttled(self, source_key: str, now: float) -> bool:
        """Whether this source's rate budget is spent.

        Records the publish time only when it lets one through, so a run of throttled
        frames does not push the next allowed one further out.
        """
        if self._max_fps <= 0:
            return False
        last = self._last_published.get(source_key)
        if last is not None and (now - last) < (1.0 / self._max_fps):
            return True
        self._last_published[source_key] = now
        return False

    def render_jpeg(self, context: ActionContext) -> bytes:
        """Select the source frame, downscale, annotate and encode.

        Blocking — call via `to_thread`.

        Downscaling happens *before* annotation, not after. Annotating first and
        shrinking the result thins every line and softens every label by exactly the
        ratio it shrank, so a capped feed would get boxes drawn for a resolution it is
        not published at. Sizing the canvas first lets `annotate_detections` pick stroke
        weights that suit what is actually going out.
        """
        # The analysis frame is the detections' pixel space whichever frame we publish,
        # so it is what any scale factor is measured against.
        analysis = context.frame
        frame = analysis
        if self._full_resolution and context.full_frame is not None:
            frame = context.full_frame
        if self._max_width:
            frame = downscale_to_width(frame, self._max_width)

        if self._annotate and context.detections:
            scale = None
            if analysis is not None and frame is not analysis:
                a_h, a_w = analysis.shape[:2]
                f_h, f_w = frame.shape[:2]
                if a_w > 0 and a_h > 0 and (f_w, f_h) != (a_w, a_h):
                    scale = (f_w / float(a_w), f_h / float(a_h))
            # Copies: `context.frame` and `context.full_frame` are the pipeline's live
            # frames, also written by snapshot actions and held in the rewind ring.
            # Burning boxes into either would put them in the saved training images.
            frame = annotate_detections(frame, context.detections, colours=context.label_colours, scale=scale)
        return encode_jpg_bytes(frame, quality=self._quality)

    async def run(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        if self._publisher is None:
            return ActionResult(ok=False, detail="No MQTT broker configured in Settings")
        if context.frame is None:
            return ActionResult(ok=False, detail="No frame available to publish")

        source_key = context.source_uuid or ""
        if self._throttled(source_key, time.monotonic()):
            # Not a failure: the ceiling is doing its job. Reported ok so a throttled
            # live feed does not fill the run's "actions failed" counter.
            return ActionResult(ok=True, detail="throttled")

        topic = self.topic_for(signal, context)
        try:
            # Annotation and JPEG encoding are real CPU work on a numpy array, and this
            # is called from the frame loop. Off the loop, exactly as the snapshot
            # action does it.
            payload = await asyncio.to_thread(self.render_jpeg, context)
        except Exception as e:
            logger.warning("Could not encode frame for %s: %s", topic, e)
            return ActionResult(ok=False, detail=f"encode failed: {e}")

        try:
            # Coalescing: only the newest frame for a topic is worth sending, so a
            # backlog behind a slow broker collapses to one frame per camera instead
            # of growing.
            self._publisher.publish(topic, payload, retain=self._retain, qos=self._qos, coalesce=True)
            return ActionResult(
                ok=True,
                detail=f"queued {topic} ({len(payload)} bytes)",
                data={"topic": topic, "sizeBytes": len(payload)},
            )
        except Exception as e:
            return ActionResult(ok=False, detail=str(e))
