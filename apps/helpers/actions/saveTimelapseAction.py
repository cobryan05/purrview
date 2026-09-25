"""Write one frame every N seconds to disk, in per-day directories.

A sibling of `save_snapshot` rather than a flag on it, because the two want opposite
answers to almost every question they share:

* **When.** A snapshot is event-gated — it exists to capture the moment something began.
  A timelapse is the opposite: it must keep writing through the 99% of the day when
  nothing happens, which is only possible on the `frame_analysed` trigger.
* **What.** A snapshot writes the *analysis* frame, because its purpose is to feed
  retraining with exactly the pixels inference saw. A timelapse is watched by a human at
  24 fps, so it defaults to the camera's own resolution.
* **Sidecars.** A snapshot writes a JSON sidecar so the image can be curated later. At
  one frame per 30 s that would be 2880 extra files per camera per day describing frames
  nobody will ever query individually, so this writes none.

Naming is chosen for `ffmpeg`: files sort chronologically inside a day directory, and the
source slug leads the name so a multi-camera watcher's frames can be globbed per camera
(`ffmpeg -pattern_type glob -i '2026-08-18/back-room_*.jpg'`) without a per-source tree.

The interval is a **wallclock bucket**, not a gap since the last write. Frames therefore
land on the interval boundary (:00 and :30 for 30 s) and stay aligned across a watcher
restart, where a gap-based timer would drift by up to one frame period every restart and
by the frame interval itself over a long run.

Retention is **not** this action's job. It writes; `Manager.prune_expired_timelapse` in
db_server bounds what accumulates, on its own pair of settings (Settings → Timelapse
Retention) separate from the recording ones. They have to be separate: at a 30 s interval
a single camera writes ~2,900 frames a day, so ~500 MB daily and ~15 GB a month at a
typical frame size — one or two orders of magnitude more than the event clips the
recording bounds were sized for. The sweep only touches directories a `save_timelapse`
action names, and only files shaped like the ones written below, because these live under
`FILES_ROOT` beside images a user may have spent months annotating.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..imageUtils import annotate_detections, downscale_to_width, encode_jpg_bytes
from ..watch.events import WatchEventSignal
from ..watch.templating import build_template_params, describe_params, render_path
from .action import Action, ActionContext, ActionResult
from .outputDir import resolve_output_dir, static_prefix
from .Registry import register_action

logger = logging.getLogger(__name__)

#: Relative to `FILES_ROOT`, not absolute — unlike `save_snapshot`, whose path predates
#: the directory picker. Every user-picked directory in this app is FILES_ROOT-relative
#: (see `CollectTrainingImagesTask.output_dir`), because that is the only path space the
#: file browser can show and the only one `/api/images/mkdir` can create in. A timelapse
#: the browser cannot navigate to is one whose frames nobody can find.
DEFAULT_TIMELAPSE_DIR = "timelapse"

#: Seconds between frames. 30 s is one day compressed to ~2 minutes at 24 fps.
DEFAULT_INTERVAL_S = 30.0


@register_action("save_timelapse")
class SaveTimelapseAction(Action):
    """Writes one frame per interval into `output_dir/<YYYY-MM-DD>/`."""

    def __init__(
        self,
        output_dir: str = DEFAULT_TIMELAPSE_DIR,
        interval_s: float = DEFAULT_INTERVAL_S,
        quality: int = 90,
        full_resolution: bool = True,
        max_width: int = 0,
        annotate: bool = False,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        # Kept as written — it is a template, and a watcher over four cameras wants four
        # folders out of one action. Resolution happens per frame.
        self.output_dir = str(output_dir).strip() or DEFAULT_TIMELAPSE_DIR
        # The literal part is still resolved once, so a path that climbs out of
        # FILES_ROOT fails when the watcher starts rather than silently every frame for
        # a week. A template whose first segment is a placeholder has no literal part,
        # and is refused here rather than at retention time: the sweep cannot bound
        # itself to a directory it cannot name, so those frames would never be cleaned
        # up and nothing would say so.
        prefix = static_prefix(self.output_dir)
        if not prefix:
            raise ValueError(
                f"output_dir {output_dir!r} must start with a real folder name before any "
                "placeholder, so its frames can be found again and swept"
            )
        resolve_output_dir(prefix, DEFAULT_TIMELAPSE_DIR)
        # A zero or negative interval would write every analysed frame — several a
        # second, forever. That is never what "timelapse" means, and the disk cost is
        # not recoverable, so it is clamped rather than honoured.
        self.interval_s = max(1.0, float(interval_s or DEFAULT_INTERVAL_S))
        self.quality = max(1, min(100, int(quality)))
        self.full_resolution = bool(full_resolution)
        self.max_width = max(0, int(max_width))
        self.annotate = bool(annotate)
        # source uuid → the wallclock bucket already written. Per source, not global:
        # one camera must not consume the interval budget of every other camera in the
        # watcher, which a single timestamp would let it do.
        self._written_bucket: Dict[str, int] = {}

    @classmethod
    def display_name(cls) -> str:
        return "Save timelapse"

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "output_dir": {
                "type": "directory",
                "required": False,
                "default": DEFAULT_TIMELAPSE_DIR,
                "help": (
                    "Directory to write frames into, relative to the image dataset root. Placeholders "
                    "are substituted and slugified per path segment — use {source} to give each camera "
                    f"its own folder. Available: {describe_params()}. Must start with a real folder "
                    "name before any placeholder. A YYYY-MM-DD subdirectory (local time) is added "
                    "unless the template uses {date}. How much accumulates is bounded by "
                    "Settings → Timelapse Retention."
                ),
            },
            "interval_s": {
                "type": "float",
                "required": False,
                "default": DEFAULT_INTERVAL_S,
                "help": (
                    "Seconds between frames, per source. Aligned to the wallclock, so 30 writes on "
                    ":00 and :30 and stays aligned across a restart. Minimum 1."
                ),
            },
            "quality": {
                "type": "int",
                "required": False,
                "default": 90,
                "help": "JPEG quality, 1-100. Around 90 looks the same as the original to the eye; 70 roughly halves the file at a visible cost on fine detail. Lower it only if disk or bandwidth is the problem.",
            },
            "full_resolution": {
                "type": "bool",
                "required": False,
                "default": True,
                "help": (
                    "Write the frame at the camera's resolution rather than the watcher's analysis "
                    "width. On by default: the analysis downscale exists to make inference cheap, "
                    "and a timelapse is watched by a human."
                ),
            },
            "max_width": {
                "type": "int",
                "required": False,
                "default": 0,
                "help": "Shrink to at most this width before encoding. 0 keeps the frame's own size.",
            },
            "annotate": {
                "type": "bool",
                "required": False,
                "default": False,
                "help": (
                    "Draw detection boxes on the frame. Off by default — burnt-in boxes make the "
                    "frames useless as training data and flicker distractingly at 24 fps."
                ),
            },
        }

    def frame_path(self, signal: WatchEventSignal, context: ActionContext, when: datetime) -> Path:
        """`<rendered output_dir>/YYYY-MM-DD/<source>_HH-MM-SS.jpg`, in local time.

        Local rather than UTC — the day boundary is the one the user's day has, and a
        UTC one cuts an evening in half for most of the world. The containers mount the
        host's `/etc/localtime`, so this is the same day the rest of the machine sees.

        The `YYYY-MM-DD` level is added unless the template places `{date}` itself, so a
        user who says where the date goes does not also get a second one. Either way a
        day folder exists somewhere on the path, which is what the retention sweep reads
        a frame's age from.

        The source stays in the *filename* even when the template already sorts by
        camera. It costs nothing, and it is what keeps a frame identifiable after it has
        been copied out of the tree it was filed in.
        """
        from .mqttPublishAction import slugify_topic

        params = build_template_params(signal, context, when=when)
        root = resolve_output_dir(render_path(self.output_dir, params), DEFAULT_TIMELAPSE_DIR)
        if "{date}" not in self.output_dir:
            root = root / when.strftime("%Y-%m-%d")

        source = slugify_topic(context.source_name or context.source_uuid or "source")
        return root / f"{source}_{when.strftime('%H-%M-%S')}.jpg"

    def _due(self, source_key: str, now: float) -> Optional[int]:
        """The bucket to write for `now`, or None when this one is already written."""
        bucket = int(now // self.interval_s)
        if self._written_bucket.get(source_key) == bucket:
            return None
        return bucket

    async def run(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        # Wallclock, not `signal.at`: the signal's clock is monotonic (see
        # `_dispatch_frame`), and a bucket derived from it would put the frames in
        # whatever day the container happened to boot on.
        when = datetime.now().astimezone()
        source_key = context.source_uuid or ""
        bucket = self._due(source_key, when.timestamp())
        if bucket is None:
            # Not an error, and not counted as one: this is the normal outcome for
            # almost every frame.
            return ActionResult(ok=True, detail="not due")

        frame = context.full_frame if self.full_resolution and context.full_frame is not None else context.frame
        if frame is None:
            return ActionResult(ok=False, detail="No frame available for the timelapse")

        # Claimed before the write, not after: an unwritable directory fails on every
        # frame otherwise, turning a misconfiguration into a hot loop of encode attempts
        # several times a second.
        self._written_bucket[source_key] = bucket

        dest = self.frame_path(signal, context, when)
        try:
            size = await asyncio.to_thread(self._write, frame, dest, context)
            return ActionResult(ok=True, detail=f"wrote {dest.name}", data={"path": str(dest), "sizeBytes": size})
        except Exception as e:
            logger.warning("Timelapse frame failed for %s: %s", context.source_name, e)
            return ActionResult(ok=False, detail=str(e))

    def _write(self, frame, dest: Path, context: ActionContext) -> int:
        # Cap first, then draw — as the MQTT image action does — so the stroke weight
        # suits the size actually written rather than the size it was cropped from.
        # A no-op when `max_width` is 0, and `encode_jpg_bytes` never mutates, so the
        # plain path copies nothing.
        frame = downscale_to_width(frame, self.max_width)
        if self.annotate and context.detections:
            # The detections are in the *analysis* frame's pixel space; any other canvas
            # needs them scaled or every box lands in the top-left quadrant.
            scale = None
            if context.frame is not None and frame.shape[:2] != context.frame.shape[:2]:
                scale = (
                    frame.shape[1] / float(context.frame.shape[1]),
                    frame.shape[0] / float(context.frame.shape[0]),
                )
            frame = annotate_detections(frame, context.detections, context.label_colours, scale=scale)

        dest.parent.mkdir(parents=True, exist_ok=True)
        # No colour conversion: providers hand out BGR, which is what cv2 expects, so the
        # frame goes out as it came in. See `saveSnapshotAction._write` for the history.
        payload = encode_jpg_bytes(frame, self.quality)
        dest.write_bytes(payload)
        return len(payload)
