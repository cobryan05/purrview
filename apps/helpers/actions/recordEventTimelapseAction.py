"""Sample a frame every N seconds *while an event is open*, and compile it at the close.

The answer to "a cat sat in the window for two hours and I want to watch it in ten
seconds". A clip cannot do it — `record_clip` cuts real time out of the rolling buffer,
so two hours of event is two hours of mp4, and the buffer never held two hours anyway.

Not a mode of `save_timelapse`, which is its opposite in every respect that matters:
that one runs on `frame_analysed` and must keep shooting through the 99% of the day when
nothing happens, writes loose JPEGs into a browsable folder under `FILES_ROOT` for a
human to scrub through, and is bounded by its own retention settings. This one exists
only while a rule is holding, writes exactly one mp4, hangs it off the event, and is
swept by the recording bounds. Sharing an implementation would mean a flag on every
question either one asks.

**Three triggers, one action.** It opens a frame buffer at `event_started`, appends on
`frame_analysed` for as long as the event stays open, and compiles at `event_ended` —
declared via `Action.triggers`. `frame_analysed` is the only per-frame trigger there is,
and its signal carries no rule (a frame belongs to no event), which is why the open
sessions are tracked here rather than being read off the signal.

**Frames go to disk, not to RAM.** At one frame per ten seconds a ninety-minute event is
540 frames; at a camera's own resolution that is a few hundred MB per concurrent event,
per camera. The buffer is a temp directory that ffmpeg reads back with `concat`, so the
memory cost is one frame regardless of how long the cat stays.
"""

import asyncio
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from apps import RECORDINGS_TIMELAPSE_DIR

from ..imageUtils import annotate_detections, downscale_to_width, encode_jpg_bytes
from ..watch.events import EventPhase, WatchEventSignal
from ..watch.scoping import EVENT_ENDED, EVENT_STARTED, FRAME_ANALYSED
from .action import Action, ActionContext, ActionResult
from .Registry import register_action

logger = logging.getLogger(__name__)

#: Shared with db_server's retention sweep, which has to find these files to delete
#: them — see `apps/__init__.py`.
DEFAULT_TIMELAPSE_ROOT = RECORDINGS_TIMELAPSE_DIR

#: Seconds between sampled frames. Ten is the interval that makes a cat's half-hour nap
#: into about fifteen watchable seconds.
DEFAULT_INTERVAL_S = 10.0

#: Frames per second of the compiled mp4. Twelve is deliberately below cinema rate: the
#: frames are seconds apart and nothing about the motion is continuous, so a higher rate
#: just makes the result shorter without making it smoother.
DEFAULT_OUTPUT_FPS = 12.0

#: Below this, no file is written at all. A sixty-second event at the default interval
#: is six frames — half a second of video, which is not a thing anyone wants to find in
#: their event list, and which costs a badge, a row and a sweep entry to store.
DEFAULT_MIN_FRAMES = 10

#: Hard ceiling on frames per event, so a stuck detection cannot fill a disk. At the
#: default interval this is ten hours of event.
DEFAULT_MAX_FRAMES = 3600

#: Timelapses are watched, not analysed, but they are also a *summary* — full camera
#: resolution buys nothing at 12 fps and costs the encode. 1280 is generous for the job.
DEFAULT_MAX_WIDTH = 1280

#: One `EventKey`, matching `_watch_event_recorder`'s: one open event per rule per source
#: per matched pair. The subject/target refs are in it so that per-pair identity, when it
#: lands, separates two cats into two timelapses with no change here.
SessionKey = Tuple[str, str, str, str]


def _session_key(signal: WatchEventSignal) -> SessionKey:
    return (
        signal.source_uuid or "",
        signal.rule_uuid or "",
        signal.subject_ref or "",
        signal.target_ref or "",
    )


class _Session:
    """One open event's frame buffer: a temp directory and a count."""

    def __init__(self, directory: Path, started_at: float) -> None:
        self.directory = directory
        self.started_at = started_at
        self.frames = 0
        #: The interval bucket already written, so a watcher analysing at 5 fps writes
        #: one frame per interval rather than fifty. Bucketed off the event's own start
        #: rather than the wallclock, unlike `save_timelapse`: an event is a closed span
        #: whose first frame should be its opening moment, not whenever the next :10
        #: boundary happens to fall.
        self.bucket: Optional[int] = None
        #: Wallclock of the first and last frames actually kept, for the row's span.
        self.first_at: Optional[float] = None
        self.last_at: Optional[float] = None

    def next_path(self) -> Path:
        # Zero-padded and sequential: ffmpeg's concat demuxer reads the list file in
        # order, but the names are what make a half-written buffer inspectable when
        # something has gone wrong.
        return self.directory / f"{self.frames:06d}.jpg"

    def discard(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


@register_action("record_event_timelapse")
class RecordEventTimelapseAction(Action):
    """Compiles one mp4 per event from frames sampled while it was open."""

    def __init__(
        self,
        interval_s: float = DEFAULT_INTERVAL_S,
        output_fps: float = DEFAULT_OUTPUT_FPS,
        min_frames: int = DEFAULT_MIN_FRAMES,
        max_frames: int = DEFAULT_MAX_FRAMES,
        max_width: int = DEFAULT_MAX_WIDTH,
        quality: int = 85,
        full_resolution: bool = False,
        annotate: bool = False,
        output_dir: str = str(DEFAULT_TIMELAPSE_ROOT),
        **params: Any,
    ) -> None:
        super().__init__(**params)
        # `or DEFAULT` then a floor of 1, matching `save_timelapse`. The two halves read
        # different values: 0 and None are what the UI form sends for "not configured"
        # and take the default, while a negative would put every frame in a new bucket —
        # several a second for the length of the event, which is never what "timelapse"
        # means and whose disk cost is not recoverable.
        self.interval_s = max(1.0, float(interval_s or DEFAULT_INTERVAL_S))
        self.output_fps = max(1.0, min(60.0, float(output_fps or DEFAULT_OUTPUT_FPS)))
        self.min_frames = max(1, int(min_frames))
        self.max_frames = max(self.min_frames, int(max_frames))
        self.max_width = max(0, int(max_width))
        self.quality = max(1, min(100, int(quality)))
        self.full_resolution = bool(full_resolution)
        self.annotate = bool(annotate)
        self.output_dir = self._resolve_output_dir(output_dir)
        self._sessions: Dict[SessionKey, _Session] = {}

    @staticmethod
    def _resolve_output_dir(output_dir: Any) -> Path:
        """Keep the video inside the recordings root unless told an absolute path.

        This action's output is not a loose file the user goes and finds: it is a
        `recordings` row hung off the event, played by the file endpoint and deleted by
        the retention sweep. All three know only the roots in `Manager._recording_roots`,
        so a path outside them produces a row pointing at a file the UI cannot play and
        nothing will ever remove.

        A relative path used to be reachable by accident — the Watchers pane had no form
        for this action, so it fell through to the snapshot branch and stored a
        dataset-relative `output_dir`, which `Path()` would then resolve against whatever
        the process's working directory happened to be. Rehomed rather than rejected:
        those rows are already saved, and refusing them would mean an action that raises
        at every event close instead of quietly doing the right thing.

        An absolute path is left alone — someone bind-mounting a NAS path meant it, and
        setting `RECORDINGS_TIMELAPSE_DIR` to the same place is how they keep the sweep.
        """
        text = str(output_dir or "").strip()
        if not text:
            return DEFAULT_TIMELAPSE_ROOT
        path = Path(text)
        return path if path.is_absolute() else DEFAULT_TIMELAPSE_ROOT / path

    @classmethod
    def display_name(cls) -> str:
        return "Event timelapse"

    @classmethod
    def logs_event(cls) -> bool:
        # Compiled per event and attached to its row, exactly as a clip is.
        return True

    def wrote_to_event(self, result) -> bool:
        # The case this whole distinction exists for: a visit too short to reach
        # `min_frames` compiles nothing and reports **ok**, because there is nothing
        # wrong — it is simply not a timelapse. It must not leave a card behind.
        return bool(result.ok and result.data.get("path"))

    @classmethod
    def triggers(cls, row: Any) -> Set[str]:
        """All three, whatever the row says.

        The row's own trigger stays `event_ended` — that is when this action produces
        its artifact, and it is the honest thing for the Watchers pane to show. The
        other two are how it gets there, and are not the user's to choose: an action
        that sampled frames only when someone remembered to tick a second row would
        silently compile nothing.
        """
        return {EVENT_STARTED, FRAME_ANALYSED, EVENT_ENDED}

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "interval_s": {
                "type": "float",
                "required": False,
                "default": DEFAULT_INTERVAL_S,
                "help": (
                    "Seconds of real time between kept frames. Ten turns a half-hour visit into "
                    "roughly fifteen seconds of video. Frames in between are discarded without "
                    "being written. Minimum 1."
                ),
            },
            "output_fps": {
                "type": "float",
                "required": False,
                "default": DEFAULT_OUTPUT_FPS,
                "help": (
                    "Frames per second of the finished video. With the interval, this sets the "
                    "speed-up: 10 s per frame at 12 fps plays 120x. Raising it shortens the "
                    "result rather than smoothing it — the frames are seconds apart either way."
                ),
            },
            "min_frames": {
                "type": "int",
                "required": False,
                "default": DEFAULT_MIN_FRAMES,
                "help": (
                    "Events with fewer kept frames than this produce no video at all. The event, "
                    "its snapshot and any clip are unaffected. Stops a brief visit from filling "
                    "the events list with half-second videos."
                ),
            },
            "max_frames": {
                "type": "int",
                "required": False,
                "default": DEFAULT_MAX_FRAMES,
                "help": (
                    "Hard ceiling per event, so a stuck detection cannot fill the disk. Frames "
                    "past it are dropped and the video ends early; the event is unaffected."
                ),
            },
            "max_width": {
                "type": "int",
                "required": False,
                "default": DEFAULT_MAX_WIDTH,
                "help": "Shrink frames to at most this width before encoding. 0 keeps their own size.",
            },
            "quality": {"type": "int", "required": False, "default": 85, "help": "JPEG quality of each frame, 1-100."},
            "full_resolution": {
                "type": "bool",
                "required": False,
                "default": False,
                "help": (
                    "Sample at the camera's resolution rather than the watcher's analysis width. "
                    "Off by default — unlike a clip, this is a summary watched at speed, and the "
                    "extra pixels cost encode time and disk for very little."
                ),
            },
            "annotate": {
                "type": "bool",
                "required": False,
                "default": False,
                "help": (
                    "Draw detection boxes onto the frames. Unlike a clip, whose boxes are drawn "
                    "over the video at play time from a sidecar, these are burnt in — a "
                    "timelapse's frames are seconds apart, so there is no continuous track to "
                    "draw from."
                ),
            },
            "output_dir": {
                "type": "str",
                "required": False,
                "default": str(DEFAULT_TIMELAPSE_ROOT),
                "help": (
                    "Directory the finished videos are written to. Not offered in the Watchers "
                    "pane, and normally left alone: at the default the videos are recordings "
                    "attached to their event, played from the event list and bounded by the same "
                    "Settings → Recording retention as clips. A relative path is taken as being "
                    "inside that root, so it keeps all three. An absolute path leaves it, and "
                    "only stays swept if RECORDINGS_TIMELAPSE_DIR points at the same place."
                ),
            },
        }

    # ── the three edges ──────────────────────────────────────────────────────

    async def run(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        # A frame signal has no rule and no phase worth reading — it is dispatched with
        # phase UPDATE for every analysed frame, whether or not anything is open. It is
        # told apart by its empty rule, which is what `_dispatch_frame` documents.
        if not signal.rule_uuid:
            return await self._on_frame(signal, context)
        if signal.phase is EventPhase.START:
            return self._on_start(signal, context)
        if signal.phase is EventPhase.END:
            return await self._on_end(signal, context)
        # UPDATE on a real rule re-fires `event_started` for actions that want a repeated
        # nudge. There is nothing to reopen — the session is already running and the
        # frames arrive on their own trigger.
        return ActionResult(ok=True, detail="already recording")

    def _on_start(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        key = _session_key(signal)
        if key in self._sessions:
            # Two STARTs with no END between them means the previous event's close was
            # lost. Keeping the old buffer would splice two visits into one video, so
            # the stale one is discarded rather than continued.
            self._sessions.pop(key).discard()
        try:
            directory = Path(tempfile.mkdtemp(prefix="wwtl-"))
        except OSError as e:
            return ActionResult(ok=False, detail=f"Could not open a frame buffer: {e}")
        self._sessions[key] = _Session(directory, signal.started_at)
        return ActionResult(ok=True, detail="recording")

    async def _on_frame(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        """Offer this frame to every open session on this source.

        Every session, not one: two rules can hold on the same camera at the same time,
        and each wants its own video of its own span. They share the decoded frame and
        pay only for their own encode, which is the cost the interval exists to bound.
        """
        source_uuid = context.source_uuid or ""
        open_here = [s for k, s in self._sessions.items() if k[0] == source_uuid]
        if not open_here:
            # The common case by a wide margin: nothing is holding on this camera, so
            # this costs one dict scan per analysed frame and no encode.
            return ActionResult(ok=True, detail="no open event")

        frame = context.full_frame if self.full_resolution and context.full_frame is not None else context.frame
        if frame is None:
            # Checked before any bucket is claimed, so a source that briefly has no
            # frame resumes on the next one rather than skipping an interval.
            return ActionResult(ok=False, detail="No frame available for the timelapse")

        due = [s for s in open_here if self._due(s, signal.at)]
        if not due:
            return ActionResult(ok=True, detail="not due")

        # Encoded once and written to each session that wants it. Two rules holding on
        # one camera at the same interval otherwise encode the identical pixels twice.
        try:
            payload = await asyncio.to_thread(self._encode, frame, context)
        except Exception as e:
            logger.warning("Timelapse frame encode failed for %s: %s", context.source_name, e)
            return ActionResult(ok=False, detail=str(e))

        now = datetime.now(timezone.utc).timestamp()
        written = 0
        for session in due:
            try:
                session.next_path().write_bytes(payload)
            except OSError as e:
                # One session's buffer being unwritable must not cost the others their
                # frame, and must not stop the event either.
                logger.warning("Timelapse frame not buffered: %s", e)
                continue
            session.frames += 1
            session.first_at = session.first_at if session.first_at is not None else now
            session.last_at = now
            written += 1
        return ActionResult(ok=True, detail=f"buffered {written} frame(s)")

    def _due(self, session: _Session, now: float) -> bool:
        """Whether this session wants the current frame, claiming the bucket if so.

        `now` is the frame signal's `at`, which is monotonic — the same clock
        `session.started_at` came from. Mixing in a wallclock reading here would make
        the bucket jump by the offset between them and take every frame at once.

        Bucketed off the event's own start rather than off the wallclock, unlike
        `save_timelapse`: an event is a closed span whose first frame should be its
        opening moment, and the rest fall at exact multiples of the interval after it
        however far the analysis rate drifts.
        """
        if session.frames >= self.max_frames:
            return False
        bucket = int(max(0.0, now - session.started_at) // self.interval_s)
        if session.bucket == bucket:
            return False
        session.bucket = bucket
        return True

    async def _on_end(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        session = self._sessions.pop(_session_key(signal), None)
        if session is None:
            # An event whose START never opened a session — the watcher resumed
            # mid-event, or the temp directory could not be made. Nothing to compile.
            return ActionResult(ok=True, detail="no frames buffered")
        try:
            if session.frames < self.min_frames:
                return ActionResult(
                    ok=True,
                    detail=f"only {session.frames} frame(s), below the {self.min_frames}-frame minimum",
                )
            dest = self.timelapse_path(signal, context, session)
            size = await asyncio.to_thread(self._compile, session, dest)
            started_at = datetime.fromtimestamp(session.first_at, tz=timezone.utc) if session.first_at else None
            return ActionResult(
                ok=True,
                detail=f"wrote {dest.name} from {session.frames} frames",
                data={
                    "path": str(dest),
                    "sizeBytes": size,
                    "startedAt": started_at.isoformat() if started_at else None,
                    # The *video's* length, not the event's. The row's duration is what
                    # the player's scrubber is measured against, and claiming the two
                    # hours the event lasted would leave the scrubber wrong by 100x.
                    "durationS": session.frames / self.output_fps,
                    "frames": session.frames,
                    # What the event actually spanned, kept beside it so the modal can
                    # say "2h compressed to 18s" rather than making the user work it out.
                    "sourceSpanS": max(0.0, (session.last_at or 0) - (session.first_at or 0)),
                    "intervalS": self.interval_s,
                },
            )
        except Exception as e:
            logger.warning("Timelapse compile failed for %s: %s", signal.rule_name, e)
            return ActionResult(ok=False, detail=str(e))
        finally:
            # The frames are scaffolding, not an artifact — the mp4 is the deliverable
            # and the whole point of compiling server-side. Removed even on failure:
            # a temp directory nothing will ever read again is a leak, and a failed
            # compile is not retryable (the event is over and the session is gone).
            session.discard()

    async def close(self) -> None:
        """Drop every open session's buffer.

        Called from the task's `_deinit`, after the run loop has emitted a closing signal
        for everything still open — so anything left here is an event whose compile
        already failed, plus whatever was open when the watcher was cancelled mid-event.
        Neither will ever be compiled, and both are holding a temp directory.
        """
        for session in self._sessions.values():
            session.discard()
        self._sessions.clear()

    # ── files ────────────────────────────────────────────────────────────────

    def timelapse_path(self, signal: WatchEventSignal, context: ActionContext, session: _Session) -> Path:
        """`<output_dir>/YYYY-MM-DD/HH-MM-SS_<source>_<rule>_timelapse.mp4`.

        The same shape `record_clip` uses, with a `_timelapse` suffix — the two land in
        different roots, but a file copied out of either should still say which it was.
        """
        from .mqttPublishAction import slugify_topic

        when = (
            datetime.fromtimestamp(session.first_at, tz=timezone.utc)
            if session.first_at
            else datetime.now(timezone.utc)
        )
        source = slugify_topic(context.source_name or context.source_uuid or "source")
        rule = slugify_topic(signal.rule_name or "event")
        return (
            self.output_dir / when.strftime("%Y-%m-%d") / f"{when.strftime('%H-%M-%S')}_{source}_{rule}_timelapse.mp4"
        )

    def _encode(self, frame: Any, context: ActionContext) -> bytes:
        # Cap first, then draw, so the stroke weight suits the size actually written —
        # the same order `save_timelapse` and the MQTT image action use.
        frame = downscale_to_width(frame, self.max_width)
        if self.annotate and context.detections:
            # Detections are in the *analysis* frame's pixel space; any other canvas
            # needs them scaled or every box lands in the top-left quadrant.
            scale = None
            if context.frame is not None and frame.shape[:2] != context.frame.shape[:2]:
                scale = (
                    frame.shape[1] / float(context.frame.shape[1]),
                    frame.shape[0] / float(context.frame.shape[0]),
                )
            frame = annotate_detections(frame, context.detections, context.label_colours, scale=scale)
        return encode_jpg_bytes(frame, self.quality)

    def _compile(self, session: _Session, dest: Path) -> int:
        """Encode the buffered frames to an mp4. Runs in a worker thread.

        The concat demuxer rather than `image2pipe`: the frames are already files, and
        piping them would mean reading hundreds of MB back into this process to hand
        straight to ffmpeg. A list file also survives a gap in the numbering, which a
        `%06d` glob would silently truncate at.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        listing = session.directory / "frames.txt"
        listing.write_text("".join(f"file '{p.name}'\n" for p in self._buffered_frames(session)))

        partial = dest.with_suffix(dest.suffix + ".part")
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "concat",
            # The list names files by their bare name, resolved against the directory
            # ffmpeg is told to trust. Without `-safe 0` it refuses relative paths.
            "-safe",
            "0",
            "-r",
            f"{self.output_fps:.3f}",
            "-i",
            str(listing),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            # yuv420p, not the JPEG-native 4:4:4: anything else is unplayable in a
            # browser <video> element, which is the whole point of writing a file.
            "-pix_fmt",
            "yuv420p",
            # Odd-width frames are fatal to libx264 at yuv420p, and a downscale to an
            # arbitrary max_width produces them about half the time.
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            # Without this the container inherits the demuxer's rate only sometimes,
            # and players disagree about the result's duration.
            "-r",
            f"{self.output_fps:.3f}",
            # Lets the browser start playing before the whole file is fetched.
            "-movflags",
            "+faststart",
            # Named explicitly because the output is written to `<name>.mp4.part` and
            # ffmpeg picks its muxer from the extension: `.part` matches nothing, so
            # without this it fails with a bare "Invalid argument" on the output file.
            "-f",
            "mp4",
            str(partial),
        ]
        try:
            proc = subprocess.run(cmd, cwd=str(session.directory), capture_output=True, timeout=300)
            if proc.returncode != 0 or not partial.exists():
                raise RuntimeError(f"ffmpeg failed writing timelapse: {proc.stderr[-300:].decode('utf-8', 'replace')}")
            size = partial.stat().st_size
            # Written to `.part` and renamed, so the orphan sweep and the file endpoint
            # never see a half-written mp4 under the recordings root.
            partial.replace(dest)
            return size
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    @staticmethod
    def _buffered_frames(session: _Session) -> List[Path]:
        return sorted(p for p in session.directory.glob("*.jpg"))
