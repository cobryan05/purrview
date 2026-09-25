"""Save the triggering frame (and its detections) to disk.

The cheapest possible action, and the one that closes the loop: an event-gated
snapshot writer is "collect training images, but only when something interesting
happened", which is a strictly better collector than a time-based one. It is how the
NVR feeds its own retraining.

Writes a JSON sidecar next to the image, matching the convention the rest of the app
already uses for image metadata.

The output directory is a **template**, over the same placeholder vocabulary the MQTT
topics use. That is the difference between one folder of everything and a folder per
camera, per rule or per day, and it costs nothing that a fixed path was buying: a value
that contains a `/` is slugified before substitution, so the shape of the path stays
exactly what the template says. The automatic `YYYY-MM-DD` level is only added when the
template does **not** place `{date}` itself — otherwise saying where the date goes would
also silently get you a second one.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ..watch.events import EventPhase, WatchEventSignal
from ..watch.templating import build_template_params, describe_params, render_path
from .action import Action, ActionContext, ActionResult
from .outputDir import resolve_output_dir
from .Registry import register_action

logger = logging.getLogger(__name__)

#: Relative to `FILES_ROOT`, like every other directory the picker can reach. The old
#: absolute default named the same folder — `resolve_output_dir` still honours that
#: spelling, so a watcher configured before the picker existed writes where it always did.
DEFAULT_SNAPSHOT_DIR = "watch_snapshots"


@register_action("save_snapshot")
class SaveSnapshotAction(Action):
    """Writes the current analysed frame as a jpg plus a metadata sidecar."""

    def __init__(
        self,
        output_dir: str = DEFAULT_SNAPSHOT_DIR,
        quality: int = 90,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        # Kept as written, not resolved here: it is a template, and what it resolves to
        # depends on the event being written. Resolution happens per snapshot.
        self.output_dir = str(output_dir).strip() or DEFAULT_SNAPSHOT_DIR
        self.quality = int(quality)

    @classmethod
    def display_name(cls) -> str:
        return "Save snapshot"

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "output_dir": {
                "type": "directory",
                "required": False,
                "default": DEFAULT_SNAPSHOT_DIR,
                "help": (
                    "Directory to write snapshots into, relative to the image dataset root, so they "
                    "can be curated later. Placeholders are substituted and slugified per path "
                    f"segment. Available: {describe_params()}. A YYYY-MM-DD level is added "
                    "automatically unless the template already uses {date}."
                ),
            },
            "quality": {
                "type": "int",
                "required": False,
                "default": 90,
                "help": "JPEG quality, 1-100. Around 90 looks the same as the original to the eye; 70 roughly halves the file at a visible cost on fine detail. Lower it only if disk or bandwidth is the problem.",
            },
        }

    def snapshot_path(self, signal: WatchEventSignal, context: ActionContext, when: datetime) -> Path:
        from .mqttPublishAction import slugify_topic

        params = build_template_params(signal, context, when=when)
        rendered = render_path(self.output_dir, params)
        root = resolve_output_dir(rendered, DEFAULT_SNAPSHOT_DIR)
        # Only when the user has not placed the date themselves — see the module docs.
        if "{date}" not in self.output_dir:
            root = root / when.strftime("%Y-%m-%d")

        source = slugify_topic(context.source_name or context.source_uuid or "source")
        rule = slugify_topic(signal.rule_name)
        return root / f"{when.strftime('%H-%M-%S')}_{source}_{rule}.jpg"

    async def run(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        if context.frame is None:
            return ActionResult(ok=False, detail="No frame available to snapshot")
        # A snapshot is about the moment something began, so an END edge would capture
        # the empty scene after the subject left.
        if signal.phase is EventPhase.END:
            return ActionResult(ok=True, detail="skipped on event end")

        # Local, and tz-aware. The folder and filename this produces are read by a human
        # browsing a file tree, so they have to agree with the clock on the wall — a UTC
        # day boundary files an evening's snapshots under tomorrow for most of the world.
        # `capturedAt` in the sidecar keeps its offset, so nothing about the instant is
        # ambiguous; only the spelling changed.
        when = datetime.now().astimezone()
        dest = self.snapshot_path(signal, context, when)
        try:
            size = await asyncio.to_thread(self._write, context.frame, dest, signal, context, when)
            return ActionResult(ok=True, detail=f"wrote {dest.name}", data={"path": str(dest), "sizeBytes": size})
        except Exception as e:
            logger.warning("Snapshot failed for %s: %s", signal.rule_name, e)
            return ActionResult(ok=False, detail=str(e))

    def _write(self, frame, dest: Path, signal: WatchEventSignal, context: ActionContext, when: datetime) -> int:
        import cv2

        dest.parent.mkdir(parents=True, exist_ok=True)
        # No colour conversion: providers hand out BGR, which is what cv2.imencode
        # expects, so the frame goes out as it came in. This used to convert RGB→BGR,
        # which was correct only for the one provider that was itself emitting RGB by
        # mistake and blue-shifted every snapshot from the other three. Converting here
        # would also be a per-event full-frame copy for no reason.
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            raise RuntimeError("cv2 failed to encode the snapshot")
        dest.write_bytes(buffer.tobytes())

        sidecar = {
            "capturedAt": when.isoformat(),
            "profileUuid": context.profile_uuid,
            "sourceUuid": context.source_uuid,
            "sourceName": context.source_name,
            "event": signal.to_dict(),
        }
        dest.with_suffix(dest.suffix + ".json").write_text(json.dumps(sidecar, indent=2, default=str))
        return dest.stat().st_size
