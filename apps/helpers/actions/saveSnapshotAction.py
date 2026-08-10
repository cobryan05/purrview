"""Save the triggering frame (and its detections) to disk.

The cheapest possible action, and the one that closes the loop: an event-gated
snapshot writer is "collect training images, but only when something interesting
happened", which is a strictly better collector than a time-based one. It is how the
NVR feeds its own retraining.

Writes a JSON sidecar next to the image, matching the convention the rest of the app
already uses for image metadata.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ..watch.events import EventPhase, WatchEventSignal
from .action import Action, ActionContext, ActionResult
from .Registry import register_action

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_ROOT = "/app/image_datasets/watch_snapshots"


@register_action("save_snapshot")
class SaveSnapshotAction(Action):
    """Writes the current analysed frame as a jpg plus a metadata sidecar."""

    def __init__(
        self,
        output_dir: str = DEFAULT_SNAPSHOT_ROOT,
        quality: int = 90,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.output_dir = Path(output_dir)
        self.quality = int(quality)

    @classmethod
    def display_name(cls) -> str:
        return "Save snapshot"

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "output_dir": {
                "type": "str",
                "required": False,
                "default": DEFAULT_SNAPSHOT_ROOT,
                "help": "Directory to write snapshots into. Under image_datasets so they can be curated later.",
            },
            "quality": {"type": "int", "required": False, "default": 90, "help": "JPEG quality, 1-100."},
        }

    def snapshot_path(self, signal: WatchEventSignal, context: ActionContext, when: datetime) -> Path:
        from .mqttPublishAction import slugify_topic

        source = slugify_topic(context.source_name or context.source_uuid or "source")
        rule = slugify_topic(signal.rule_name)
        return self.output_dir / when.strftime("%Y-%m-%d") / f"{when.strftime('%H-%M-%S')}_{source}_{rule}.jpg"

    async def run(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        if context.frame is None:
            return ActionResult(ok=False, detail="No frame available to snapshot")
        # A snapshot is about the moment something began, so an END edge would capture
        # the empty scene after the subject left.
        if signal.phase is EventPhase.END:
            return ActionResult(ok=True, detail="skipped on event end")

        when = datetime.now(timezone.utc)
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
        # Providers hand out RGB; cv2.imwrite expects BGR, and writing the wrong one
        # produces blue cats.
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
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
