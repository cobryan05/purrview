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

from apps import FILES_ROOT

from ..imageUtils import annotate_detections, downscale_to_width, encode_jpg_bytes
from ..watch.events import WatchEventSignal
from .action import Action, ActionContext, ActionResult
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


def resolve_output_dir(output_dir: str) -> Path:
    """Turn a configured directory into an absolute path inside `FILES_ROOT`.

    Accepts all three spellings that reach it, because they all occur in practice:

    * ``timelapse`` — a hand-typed relative path.
    * ``/timelapse`` — what the directory picker returns. Its paths are rooted at
      `FILES_ROOT`, so its leading slash means "top of the dataset root", not "top of
      the filesystem". Treating it as absolute would write jpgs into the container's
      own root every 30 seconds.
    * ``/app/image_datasets/timelapse`` — an absolute path that is already inside
      `FILES_ROOT`, honoured as-is so a path copied from `save_snapshot` still works.

    Anything that lands outside `FILES_ROOT` — an absolute path elsewhere, or enough
    ``..`` to climb out — raises. The picker cannot reach outside it, nothing else in
    the app writes outside it, and a watcher is a long-running process writing an
    unbounded number of files: it is the wrong thing to point at an arbitrary path.
    """
    root = FILES_ROOT.resolve()
    raw = (str(output_dir).strip() or DEFAULT_TIMELAPSE_DIR).rstrip("/")
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            candidate.relative_to(root)
        except ValueError:
            # Not under the root, so its leading slash was the picker's, not the
            # filesystem's. Re-root it rather than rejecting: that spelling is what the
            # UI produces, and rejecting it would make the Browse button unusable.
            candidate = root / candidate.relative_to("/")
    else:
        candidate = root / candidate

    resolved = Path(os.path.normpath(str(candidate)))
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"output_dir {output_dir!r} resolves outside {root}")
    return resolved


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
        # Resolved once, at construction, so a path that climbs out of FILES_ROOT fails
        # when the watcher starts rather than silently every frame for a week.
        self.output_dir = resolve_output_dir(output_dir)
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
                    "Directory to write frames into, relative to the image dataset root. A per-day "
                    "subdirectory (YYYY-MM-DD, local time) is created under it. How much accumulates "
                    "is bounded by Settings → Timelapse Retention."
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
            "quality": {"type": "int", "required": False, "default": 90, "help": "JPEG quality, 1-100."},
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

    def frame_path(self, context: ActionContext, when: datetime) -> Path:
        """`output_dir/YYYY-MM-DD/<source>_HH-MM-SS.jpg`, in local time.

        Local rather than UTC — the day boundary is the one the user's day has, and a
        UTC one cuts an evening in half for most of the world. The containers mount the
        host's `/etc/localtime`, so this is the same day the rest of the machine sees.
        """
        from .mqttPublishAction import slugify_topic

        source = slugify_topic(context.source_name or context.source_uuid or "source")
        return self.output_dir / when.strftime("%Y-%m-%d") / f"{source}_{when.strftime('%H-%M-%S')}.jpg"

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

        dest = self.frame_path(context, when)
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
