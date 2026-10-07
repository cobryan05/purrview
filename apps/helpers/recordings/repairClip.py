"""Re-decode a clip with error concealment, so lost packets read as a held frame.

A camera whose stream loses packets — a Ring camera over RTSP, routinely — hands
MediaMTX H.264 with damaged slices. The recorder stores the bitstream faithfully and a
clip cut from it is stream-copied, so the damage survives into the file and the player's
decoder renders it: macroblocks from the damaged row down become garbage, and inter
prediction carries that garbage forward to the next keyframe. On screen it is half a
frame of noise persisting for seconds.

The decoder already knows how to do better. `-ec favor_inter` fills a damaged macroblock
from the *previous* frame instead of guessing within the current one, which for a static
scene is very nearly the right pixels. But concealment happens at decode time, so it
only reaches the file by re-encoding — which is why this is opt-in per camera rather
than something every clip pays for. The switch is `repair_corruption` on the source,
read off the provider and carried to the clip action on its `ActionContext`.

A camera that has *not* opted in gets `probe_for_corruption` instead: the same decode
scan with no encode, bounded per source, so the app can suggest the switch rather than
leaving the user to guess which of their cameras is lossy. See `clipRepairAdvice`.

Two rules this file exists to keep:

* **Never drop frames.** `-fflags +discardcorrupt` would be the obvious tool and is the
  wrong one: dropping packets shortens the video, which slides every detection box
  after the drop. Concealment preserves frame count and timing, so the sidecar written
  against the original timeline stays correct.
* **A repair that changes the duration is discarded.** The overlay's box offsets are
  bounded by the clip's measured length, so a re-encode that came out a different length
  would invalidate them. Keeping the corrupt original is better than keeping a clean
  one the boxes no longer fit.
"""

import asyncio
import contextlib
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Decoder errors worth repairing. `concealing` appears when ffmpeg already papered over
#: a loss; `error while decoding` and `bytestream` are the damaged-slice cases.
_ERROR_PATTERN = re.compile(r"error while decoding|bytestream|concealing|corrupt|Invalid NAL", re.I)

#: How far the repaired file's length may move before the repair is rejected. The
#: sidecar's offsets are already written against the original, so this is tight.
DURATION_TOLERANCE_S = 0.25

#: Below this many decoder errors a clip is not worth a full re-encode — a handful of
#: damaged macroblocks in one frame is a flicker nobody sees.
MIN_ERRORS_TO_REPAIR = 5

#: Help text for the per-source switch. Lives here rather than in either provider so the
#: two cannot drift; both relay-capable providers import it into their `params_schema`.
REPAIR_CORRUPTION_DESCRIPTION = (
    "Repair frames this camera's stream lost packets in, instead of leaving the picture to "
    "break up. A lossy camera \u2014 anything over wifi, and the Ring cameras in particular "
    "\u2014 drops parts of a frame, and the player renders the damage as a band of noise that "
    "lasts until the next keyframe. With this on, the damaged part of the frame is filled in "
    "from the frame before it, so the picture holds still rather than turning to garbage. Only "
    "clips that are actually damaged are re-encoded, and it does not recover video the camera "
    "never sent \u2014 a clip that jumps still jumps. Costs a minute or two of background CPU on "
    "a long clip, so leave it off for cameras with a clean feed."
)

#: The live app shares this box, and a 1080p re-encode will take every core it is given.
#: Clips have no deadline; the app does.
_NICE = ["nice", "-n", "19", "ionice", "-c3"]


def _run(cmd: list, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def _count_errors(path: Path) -> int:
    proc = _run(_NICE + ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "null", "-"], 600)
    text = proc.stderr.decode("utf-8", "replace")
    return sum(1 for line in text.splitlines() if _ERROR_PATTERN.search(line))


def _probe_duration(path: Path) -> Optional[float]:
    proc = _run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)], 60
    )
    try:
        return float(proc.stdout.decode().strip())
    except ValueError:
        return None


def _conceal(src: Path, dest: Path) -> subprocess.CompletedProcess:
    return _run(
        _NICE
        + [
            "ffmpeg",
            "-nostdin",
            "-y",
            # Decoder options, so they must precede the input they apply to.
            "-err_detect",
            "ignore_err",
            "-ec",
            "guess_mvs+deblock+favor_inter",
            "-i",
            str(src),
            # Frame timing straight through: a re-encode that resamples the frame rate
            # would move every box relative to the picture.
            "-fps_mode",
            "passthrough",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            # Audio is undamaged by a video slice loss and re-encoding it buys nothing.
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(dest),
        ],
        1800,
    )


async def repair_if_corrupt(path: Path, *, enabled: bool = True) -> dict:
    """Conceal decoder errors in `path`, in place. Returns what was done and why.

    Never raises and never leaves `path` worse than it found it: every failure path
    keeps the original file. A clip is worth more corrupt than missing.
    """
    if not enabled:
        return {"repaired": False, "reason": "disabled"}
    if not path.is_file():
        return {"repaired": False, "reason": "missing"}

    try:
        errors = await asyncio.to_thread(_count_errors, path)
    except Exception as exc:
        logger.warning("Could not scan %s for decoder errors: %s", path.name, exc)
        return {"repaired": False, "reason": "scan failed"}
    if errors < MIN_ERRORS_TO_REPAIR:
        return {"repaired": False, "reason": "clean", "decodeErrors": errors}

    partial = path.with_suffix(path.suffix + ".repair")
    try:
        # Inside the try, like every other subprocess here: ffprobe can time out or be
        # missing, and this function's contract is that it never raises.
        before = await asyncio.to_thread(_probe_duration, path)
        proc = await asyncio.to_thread(_conceal, path, partial)
        if proc.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
            tail = proc.stderr[-300:].decode("utf-8", "replace")
            logger.warning("Concealment failed for %s, keeping the original: %s", path.name, tail)
            return {"repaired": False, "reason": "encode failed", "decodeErrors": errors}

        after = await asyncio.to_thread(_probe_duration, partial)
        if before is None or after is None or abs(after - before) > DURATION_TOLERANCE_S:
            logger.warning(
                "Concealment changed %s from %ss to %ss; keeping the original so the overlay stays valid",
                path.name,
                f"{before:.2f}" if before else "?",
                f"{after:.2f}" if after else "?",
            )
            return {"repaired": False, "reason": "duration moved", "decodeErrors": errors}

        remaining = await asyncio.to_thread(_count_errors, partial)
        await asyncio.to_thread(shutil.move, str(partial), str(path))
        logger.info("Concealed %s: %d decoder error(s) -> %d", path.name, errors, remaining)
        return {
            "repaired": True,
            "decodeErrors": errors,
            "decodeErrorsAfter": remaining,
            "durationS": after,
        }
    except Exception as exc:
        logger.warning("Concealment failed for %s, keeping the original: %s", path.name, exc)
        return {"repaired": False, "reason": "error", "decodeErrors": errors}
    finally:
        partial.unlink(missing_ok=True)


async def probe_for_corruption(path: Path) -> Optional[int]:
    """Count `path`'s decoder errors without touching the file. None if it can't be read.

    The scan half of `repair_if_corrupt`, for a camera that has *not* opted in: it is one
    decode pass with no encode, so it costs a fraction of a repair and can be spent on a
    camera that turns out to be clean. Bounded by the caller — see
    `clipRepairAdvice.wants_probe`.
    """
    if not path.is_file():
        return None
    try:
        return await asyncio.to_thread(_count_errors, path)
    except Exception as exc:
        logger.warning("Could not scan %s for decoder errors: %s", path.name, exc)
        return None


#: How deep the backlog may get before new clips are left alone. A repair that never
#: runs leaves the file exactly as the recorder wrote it, which is what every failure
#: path above already settles for.
MAX_PENDING_REPAIRS = 32


class _RepairWorker:
    """Runs queued repairs one at a time, off the submitter's await.

    The recorder awaits its actions inside the watcher's frame loop, so a repair awaited
    there stops that profile analysing *all* of its cameras for the length of a decode
    scan plus a re-encode. Serialised rather than a task per clip: a busy camera closes
    several events at once, and concurrent 1080p re-encodes are what `_NICE` exists to
    prevent.
    """

    def __init__(self) -> None:
        self._queue: Optional[asyncio.Queue] = None
        self._worker: Optional[asyncio.Task] = None
        self._owners = 0

    def retain(self) -> None:
        self._owners += 1

    def submit(self, path: Path, probe: Optional[dict] = None) -> bool:
        """Queue `path`. False when the backlog is full and it was left as-is.

        With `probe` the job is scan-only and the dict says which source to attribute the
        result to; without it the clip is repaired in place. Both share the one queue so a
        probe can never run beside a re-encode — the whole point of serialising here.
        """
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=MAX_PENDING_REPAIRS)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._drain(), name="clip-repair")
        try:
            self._queue.put_nowait((path, probe))
        except asyncio.QueueFull:
            logger.warning("Repair backlog is full; leaving %s as recorded", path.name)
            return False
        return True

    async def _drain(self) -> None:
        queue = self._queue
        assert queue is not None
        while True:
            path, probe = await queue.get()
            try:
                if probe is None:
                    await repair_if_corrupt(path, enabled=True)
                else:
                    await _run_probe(path, probe)
            except Exception:
                logger.exception("Repair worker failed on %s", path.name)
            finally:
                queue.task_done()

    async def release(self) -> None:
        """Drop one owner's claim; the worker stops once the last one is gone.

        Refcounted because the worker is process-wide while its owners are not: two
        watcher profiles share it, and one of them stopping must not cancel the other's
        queued repairs.
        """
        self._owners = max(0, self._owners - 1)
        if self._owners:
            return
        worker, self._worker, self._queue = self._worker, None, None
        if worker is None or worker.done():
            return
        # An in-flight ffmpeg runs in a thread and cannot be cancelled with the await, so
        # it may recreate its `.repair` file after the `finally` above has unlinked one.
        # Nothing reads that path, and the next repair of the same clip passes `-y`.
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker


_repairs = _RepairWorker()


def retain_repairs() -> None:
    """Claim the shared repair worker for an action that will submit to it."""
    _repairs.retain()


def submit_repair(path: Path) -> bool:
    """Hand a finished clip to the background repair worker."""
    return _repairs.submit(path)


def submit_probe(path: Path, *, source_uuid: str, source_name: str = "") -> bool:
    """Queue a scan-only pass over a clip from a camera that has not opted in."""
    return _repairs.submit(path, probe={"sourceUuid": source_uuid, "sourceName": source_name})


async def _run_probe(path: Path, probe: dict) -> None:
    """Scan one clip and, if it is badly damaged, record a finding for the notices bar."""
    from . import clipRepairAdvice

    source_uuid = probe["sourceUuid"]
    # Re-checked here, not only at submit time: several clips for the same camera can sit
    # in the queue together, and the user may have answered the suggestion in between.
    if not clipRepairAdvice.wants_probe(source_uuid):
        return
    errors = await probe_for_corruption(path)
    if errors is None:
        return
    if clipRepairAdvice.record_probe(source_uuid, errors, path.name, source_name=probe.get("sourceName", "")):
        logger.info(
            "%s decoded with %d error(s); suggesting clip repair for it",
            probe.get("sourceName") or source_uuid,
            errors,
        )


async def release_repairs() -> None:
    """Release one claim on the shared worker, stopping it after the last."""
    await _repairs.release()
