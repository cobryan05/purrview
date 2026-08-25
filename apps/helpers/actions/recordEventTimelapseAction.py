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
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from apps import RECORDINGS_TIMELAPSE_DIR

from ..imageUtils import annotate_detections, downscale_to_width, encode_jpg_bytes
from ..recordings.detectionTrack import detection_box
from ..watch.events import EventPhase, WatchEventSignal
from ..watch.scoping import EVENT_ENDED, EVENT_STARTED, FRAME_ANALYSED
from .action import Action, ActionContext, ActionResult
from .Registry import register_action

logger = logging.getLogger(__name__)

#: Shared with db_server's retention sweep, which has to find these files to delete
#: them — see `apps/__init__.py`.
DEFAULT_TIMELAPSE_ROOT = RECORDINGS_TIMELAPSE_DIR

#: Prefix every frame buffer shares, so `sweep_frame_buffers` can find them and nothing
#: else. The preview scratch dirs use a longer prefix starting with this one, so they
#: are swept on the same terms — an abandoned preview encode is just as much a leak.
BUFFER_PREFIX = "wwtl-"

#: Written into every frame buffer as it is opened, naming the process that owns it.
#: The sweep keys on this rather than on the directory's age because every watcher
#: shares the tasks_server process: "left behind by the process an OOM killer took" and
#: "being filled right now by another profile's watcher" are otherwise
#: indistinguishable, and guessing wrong deletes a live event's frames. Not a `.jpg`,
#: so `_buffered_frames` never sees it.
BUFFER_OWNER_FILENAME = "owner.pid"

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


def open_buffer_dir(prefix: str = BUFFER_PREFIX) -> Path:
    """A fresh frame buffer, stamped with the pid that owns it.

    The stamp is what makes `sweep_frame_buffers` safe, so it is written before the
    caller gets a chance to put a frame in the directory. A failure to write it is
    logged and swallowed: the buffer is still perfectly usable, and losing an event's
    frames over a bookkeeping file would be much the worse trade.
    """
    directory = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        (directory / BUFFER_OWNER_FILENAME).write_text(str(os.getpid()))
    except OSError as e:
        logger.warning("Timelapse frame buffer %s could not be marked as owned: %s", directory, e)
    return directory


def _buffer_owner_is_alive(directory: Path) -> bool:
    """Whether the process that opened this buffer is still running.

    An absent or unreadable marker reads as *dead*: a live buffer is always stamped
    before it is written to, so an unmarked one can only predate this mechanism, which
    makes it a leftover by definition.

    A pid that has since been recycled onto some unrelated program reads as alive, so
    the ambiguous case errs towards a buffer that outlives its process by one restart
    rather than one deleted out from under a running watcher.
    """
    try:
        pid = int((directory / BUFFER_OWNER_FILENAME).read_text().strip())
    except (OSError, ValueError):
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists; it just belongs to another user.
        return True
    except OSError:
        return False
    return True


def sweep_frame_buffers(root: Optional[Path] = None) -> int:
    """Delete frame buffers whose owning process is gone. Returns how many were removed.

    The ordinary paths already clean up after themselves — `_on_end` discards in a
    `finally`, and a watcher that stops cleanly force-closes everything open on the way
    out. This is for the case where no Python runs at all: an OOM kill, a `docker kill`,
    a power cut. Clip chunks have had `sweep_chunk_root` for exactly this reason;
    these had nothing, and sixteen orphaned buffers were sitting in the container's
    temp directory against zero open events when this was written.

    Called at watcher start, which is safe precisely because the test is ownership
    rather than age: another watcher in this same process keeps its buffers.
    """
    base = Path(root) if root is not None else Path(tempfile.gettempdir())
    if not base.is_dir():
        return 0
    removed = 0
    for directory in base.glob(f"{BUFFER_PREFIX}*"):
        if not directory.is_dir() or _buffer_owner_is_alive(directory):
            continue
        shutil.rmtree(directory, ignore_errors=True)
        removed += 1
    return removed


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
        #: One list of normalised boxes per kept frame, parallel to the frames on disk.
        #: Kept in RAM rather than written beside them: at a few hundred bytes a frame
        #: even the 3600-frame ceiling is under a megabyte, and the sidecar has to be
        #: one document anyway.
        self.boxes: List[List[Dict[str, Any]]] = []

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
        preroll_frames: int = 0,
        save_overlay: bool = True,
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
        self.preroll_frames = max(0, int(preroll_frames or 0))
        self.save_overlay = bool(save_overlay)
        self.output_dir = self._resolve_output_dir(output_dir)
        self._sessions: Dict[SessionKey, _Session] = {}
        #: Per-source ring of the most recent sampled frames while **nothing** is open,
        #: so an opening event can be given the approach that preceded it. Only fed when
        #: `preroll_frames` is set: with it at zero an idle camera must keep costing one
        #: dict scan per analysed frame and no encode, which is the overwhelmingly common
        #: case and the reason the idle path is written the way it is.
        self._preroll: Dict[str, List[Tuple[bytes, List[Dict[str, Any]], float]]] = {}
        #: The interval bucket each source's ring has already taken, so a watcher
        #: analysing at 5 fps still rings one frame per interval rather than fifty.
        self._preroll_buckets: Dict[str, int] = {}

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

    def holds_open_event(self, signal: WatchEventSignal) -> bool:
        """True once this event's session has enough frames to be worth compiling.

        Below `min_frames` nothing will be written, so the event has earned no row yet —
        the same line `_on_end` draws, asked earlier. Above it the video is a certainty
        (bar the watcher dying), and the event can be listed while it is still running
        instead of appearing only once the cat has left.
        """
        session = self._sessions.get(_session_key(signal))
        return session is not None and session.frames >= self.min_frames

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
            "preroll_frames": {
                "type": "int",
                "required": False,
                "default": 0,
                "help": (
                    "Frames from before the event opened to put at the front of the video, so it "
                    "shows the approach rather than starting with the subject already there. "
                    "Counted in sampled frames, not seconds: 3 at a 10 s interval is the previous "
                    "30 seconds. Costs one frame encode per interval on an idle camera, so it is "
                    "off by default."
                ),
            },
            "save_overlay": {
                "type": "bool",
                "required": False,
                "default": True,
                "help": (
                    "Write the detection boxes beside the video as a sidecar, so the player can "
                    "draw them over it and toggle them off. Costs a few tens of KB, re-encodes "
                    "nothing, and leaves the video itself clean for saving training frames."
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
            directory = open_buffer_dir()
        except OSError as e:
            return ActionResult(ok=False, detail=f"Could not open a frame buffer: {e}")
        session = _Session(directory, signal.started_at)
        self._sessions[key] = session
        prerolled = self._seed_preroll(session, context.source_uuid or "")
        return ActionResult(ok=True, detail=f"recording (pre-roll {prerolled} frame(s))" if prerolled else "recording")

    def _seed_preroll(self, session: _Session, source_uuid: str) -> int:
        """Lay the ring's frames down as the first frames of a new session.

        Copied rather than moved: two rules can open on one camera at different moments
        and both deserve the approach, and the ring keeps being fed while an event is
        open precisely so the second one gets a current view rather than whatever was
        happening before the first event began.

        A failed write is skipped rather than fatal — losing the lead-in is a much
        smaller loss than losing the event's own video, which is what raising here would
        cost.
        """
        ring = self._preroll.get(source_uuid) or []
        for payload, boxes, at_wall in ring:
            try:
                session.next_path().write_bytes(payload)
            except OSError as e:
                logger.warning("Timelapse pre-roll frame not buffered: %s", e)
                continue
            session.boxes.append(boxes)
            session.frames += 1
            # The video now begins before the event does, and the row's span has to say
            # so — otherwise the modal reports a start later than the first frame shown.
            if session.first_at is None:
                session.first_at = at_wall
            session.last_at = at_wall
        return session.frames

    async def _on_frame(self, signal: WatchEventSignal, context: ActionContext) -> ActionResult:
        """Offer this frame to every open session on this source.

        Every session, not one: two rules can hold on the same camera at the same time,
        and each wants its own video of its own span. They share the decoded frame and
        pay only for their own encode, which is the cost the interval exists to bound.
        """
        source_uuid = context.source_uuid or ""
        open_here = [s for k, s in self._sessions.items() if k[0] == source_uuid]
        if not open_here and not self.preroll_frames:
            # The common case by a wide margin: nothing is holding on this camera, so
            # this costs one dict scan per analysed frame and no encode. Only a
            # configured pre-roll makes an idle camera worth sampling at all.
            return ActionResult(ok=True, detail="no open event")

        frame = context.full_frame if self.full_resolution and context.full_frame is not None else context.frame
        if frame is None:
            # Checked before any bucket is claimed, so a source that briefly has no
            # frame resumes on the next one rather than skipping an interval.
            return ActionResult(ok=False, detail="No frame available for the timelapse")

        due = [s for s in open_here if self._due(s, signal.at)]
        # The ring runs on its own bucket, off the frame clock rather than any event's
        # start: with nothing open there is no session whose start could anchor it, and
        # while something *is* open the ring must keep pace so a second rule opening
        # later on this camera gets a current lead-in rather than a stale one.
        ring_due = bool(self.preroll_frames) and self._preroll_due(source_uuid, signal.at)
        if not due and not ring_due:
            return ActionResult(ok=True, detail="not due")

        # Encoded once and written to each session that wants it. Two rules holding on
        # one camera at the same interval otherwise encode the identical pixels twice.
        try:
            payload = await asyncio.to_thread(self._encode, frame, context)
        except Exception as e:
            logger.warning("Timelapse frame encode failed for %s: %s", context.source_name, e)
            return ActionResult(ok=False, detail=str(e))

        # Computed once per frame like the encode, and from `context.frame` whatever the
        # frames were written at: detections are in the *analysis* frame's pixel space,
        # and normalising against the encoded frame would scale every box by the
        # downscale ratio. Normalised coordinates then survive `max_width` and
        # `full_resolution` without either of them being consulted here.
        boxes = self._overlay_boxes(context)

        now = datetime.now(timezone.utc).timestamp()

        if ring_due:
            ring = self._preroll.setdefault(source_uuid, [])
            ring.append((payload, boxes, now))
            # Trimmed from the front so the ring is always the *most recent* N frames.
            # Bounded by count rather than age: the interval already decides how much
            # time N frames cover, and a second bound would let the two disagree.
            del ring[: max(0, len(ring) - self.preroll_frames)]

        written = 0
        for session in due:
            try:
                session.next_path().write_bytes(payload)
            except OSError as e:
                # One session's buffer being unwritable must not cost the others their
                # frame, and must not stop the event either.
                logger.warning("Timelapse frame not buffered: %s", e)
                continue
            # Appended only where the frame was, so the two stay index-aligned: a box
            # list one longer than the frames on disk would shift every annotation by a
            # frame for the rest of the video.
            session.boxes.append(boxes)
            session.frames += 1
            session.first_at = session.first_at if session.first_at is not None else now
            session.last_at = now
            written += 1
        return ActionResult(ok=True, detail=f"buffered {written} frame(s)")

    def _preroll_due(self, source_uuid: str, now: float) -> bool:
        """Whether the pre-roll ring wants this frame, claiming the bucket if so.

        Bucketed off the frame clock, which is monotonic and shared with `_due`. The
        ring has no event to anchor to — that is the whole point of it — so it uses
        absolute interval boundaries the way `save_timelapse` does.
        """
        bucket = int(max(0.0, now) // self.interval_s)
        if self._preroll_buckets.get(source_uuid) == bucket:
            return False
        self._preroll_buckets[source_uuid] = bucket
        return True

    def _overlay_boxes(self, context: ActionContext) -> List[Dict[str, Any]]:
        """This frame's detections as normalised overlay boxes.

        Empty when the sidecar is switched off, when nothing was detected, or when there
        is no analysis frame to normalise against — and an empty list is meaningful, not
        missing: it is how the player is told the detector looked and saw nothing here,
        which is what stops it holding the previous frame's boxes over a cat that left.
        """
        if not self.save_overlay or context.frame is None:
            return []
        height, width = context.frame.shape[:2]
        return [b for b in (detection_box(d, width, height) for d in (context.detections or [])) if b]

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
            overlay = self._write_overlay(session, dest, started_at)
            return ActionResult(
                ok=True,
                detail=f"wrote {dest.name} from {session.frames} frames",
                data={
                    "path": str(dest),
                    "sizeBytes": size,
                    # Written here rather than by the recorder, which builds a clip's
                    # sidecar by slicing the source's detection ring over a wallclock
                    # window. That is meaningless for a timelapse — see
                    # `build_timelapse_overlay` — so this action, the only thing that
                    # knows which frames it kept, hands over a finished path.
                    "overlayPath": str(overlay) if overlay else None,
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

    #: A preview must not become a way to run an unbounded encode on demand: at one
    #: frame per interval a long event holds thousands, and re-encoding all of them on
    #: every click would cost more than the watcher itself. Past this the preview shows
    #: the most recent frames, which is the part anyone previewing is looking for.
    PREVIEW_MAX_FRAMES = 900

    async def preview(self, key: SessionKey) -> Optional[Dict[str, Any]]:
        """Compile what one still-open session has collected so far, without disturbing it.

        The answer to "this event has been running for an hour — what does it look like
        so far?". The session keeps sampling: nothing here mutates it, the frames stay
        on disk, and the real compile at the close still sees all of them.

        Returns None when there is no such session or it holds too little to play.

        The newest frame is deliberately left out. `write_bytes` is not atomic and the
        sampler may be part-way through that very file, so including it risks a torn
        JPEG at the end of every preview — one frame is nothing next to a video that is
        seconds long.
        """
        session = self._sessions.get(key)
        if session is None:
            return None
        frames = self._buffered_frames(session)[:-1]
        if len(frames) < 2:
            return None
        frames = frames[-self.PREVIEW_MAX_FRAMES :]

        scratch = open_buffer_dir(f"{BUFFER_PREFIX}preview-")
        dest = scratch / "preview.mp4"
        try:
            await asyncio.to_thread(self._compile, session, dest, frames, "preview.txt")
            return {
                "video": dest.read_bytes(),
                "frames": len(frames),
                "durationS": len(frames) / self.output_fps,
                # What the previewed frames span in real time, so the player can say
                # "the last 40 minutes" rather than only "18 seconds of video".
                "sourceSpanS": max(0.0, (session.last_at or 0) - (session.first_at or 0)),
                "intervalS": self.interval_s,
                # True when the preview is a tail rather than the whole event so far —
                # the modal has to say so, or it reads as the event having started later
                # than it did.
                "partial": len(frames) < session.frames - 1,
            }
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
            # The preview's own list file, not the session's — see `_compile`.
            (session.directory / "preview.txt").unlink(missing_ok=True)

    def open_session_keys(self) -> List[SessionKey]:
        """Every session currently buffering, so a caller can offer a preview of one."""
        return list(self._sessions)

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
        # Held JPEGs, so this is the one piece of this action's state with a real memory
        # cost. A stopped watcher has no use for a lead-in to an event that will never
        # open, and the frames are stale the moment the camera is next read.
        self._preroll.clear()
        self._preroll_buckets.clear()

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
        """Encode one sampled frame. Nothing is ever drawn on it.

        There used to be an `annotate` option here that burnt the detection boxes into
        the pixels. It is gone rather than defaulted off: the boxes are now a sidecar the
        player draws at run time and can toggle, which is strictly better in every
        direction — it re-encodes nothing, it can be turned off after the fact, and it
        keeps every frame of the video usable as a training image, which is what "save
        frame for annotation" in the event modal exists to collect. A burnt-in box cannot
        be removed later, so keeping the option would only preserve a way to permanently
        spoil the recording. A stored `annotate: true` from an older config is inert: it
        lands in the base class's `**params` and nothing reads it.
        """
        return encode_jpg_bytes(downscale_to_width(frame, self.max_width), self.quality)

    def _write_overlay(self, session: _Session, dest: Path, started_at: Optional[datetime]) -> Optional[Path]:
        """Drop the boxes beside the finished mp4. Never raises.

        The video is the artifact and the boxes are a commentary on it — the same
        contract the clip sidecar and the event's card art hold to. A sidecar that
        cannot be written must not cost the user the timelapse.

        Skipped entirely when nothing was detected in any kept frame: an all-empty
        document tells the player "the detector saw nothing in this event", which is a
        stronger claim than "there is no record", and it would put an overlay toggle on
        every timelapse that could never show anything.
        """
        if not self.save_overlay or not any(session.boxes):
            return None
        try:
            from ..recordings.detectionTrack import build_timelapse_overlay, overlay_path_for, write_overlay

            document = build_timelapse_overlay(
                session.boxes,
                started_at or datetime.now(timezone.utc),
                self.output_fps,
            )
            sidecar = overlay_path_for(dest)
            write_overlay(document, sidecar)
            return sidecar
        except Exception as e:
            logger.warning("Timelapse overlay not written for %s: %s", dest.name, e)
            return None

    def _compile(
        self,
        session: _Session,
        dest: Path,
        frames: Optional[List[Path]] = None,
        listing_name: str = "frames.txt",
    ) -> int:
        """Encode the buffered frames to an mp4. Runs in a worker thread.

        The concat demuxer rather than `image2pipe`: the frames are already files, and
        piping them would mean reading hundreds of MB back into this process to hand
        straight to ffmpeg. A list file also survives a gap in the numbering, which a
        `%06d` glob would silently truncate at.

        `frames` and `listing_name` exist for the mid-event preview, which encodes a
        prefix of a session that is still being written to. It needs its own list file
        because the closing compile may run while a preview is in flight, and both
        writing `frames.txt` at once would give one of them a truncated or interleaved
        list — a corrupt video from a race that reproduces about never.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        listing = session.directory / listing_name
        chosen = self._buffered_frames(session) if frames is None else frames
        listing.write_text("".join(f"file '{p.name}'\n" for p in chosen))

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
