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
than something every clip pays for.

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

    before = await asyncio.to_thread(_probe_duration, path)
    partial = path.with_suffix(path.suffix + ".repair")
    try:
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

    def submit(self, path: Path) -> bool:
        """Queue `path` for repair. False when the backlog is full and it was left as-is."""
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=MAX_PENDING_REPAIRS)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._drain(), name="clip-repair")
        try:
            self._queue.put_nowait(path)
        except asyncio.QueueFull:
            logger.warning("Repair backlog is full; leaving %s as recorded", path.name)
            return False
        return True

    async def _drain(self) -> None:
        queue = self._queue
        assert queue is not None
        while True:
            path = await queue.get()
            try:
                await repair_if_corrupt(path, enabled=True)
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


async def release_repairs() -> None:
    """Release one claim on the shared worker, stopping it after the last."""
    await _repairs.release()
