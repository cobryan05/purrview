"""Notice when a camera's recorded clips stop holding continuous video.

A recorder that drops video leaves a file that still plays: the audio runs on and the
picture freezes on its last frame, or there is no picture at all. Nothing upstream
reports it — MediaMTX's recorder discards late samples with only a log line, and its API
has no counter for them — so the clip itself is measured.

One finding per source, memory-resident for the same reason as `clipRepairAdvice`:
`/status` is polled by the notices loop every few seconds. Every clip is checked, not a
capped few, because this is a regression watch: a camera that was clean yesterday is
exactly the one a firmware update breaks. The check reads packet timestamps only, with no
decode, on a file that was just written and is still in the page cache.
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

#: A gap between consecutive video frames at least this long counts as missing video.
#: Healthy cameras here peak around 0.5 s; a Wi-Fi hiccup can reach a second or two.
MIN_HOLE_S = 3.0

_findings: Dict[str, dict] = {}
#: Strong references to in-flight checks; the event loop only keeps weak ones.
_pending: Set[asyncio.Task] = set()


def analyse(video_pts: Optional[List[float]], has_audio: bool) -> dict:
    """Summarise one clip's video coverage from its video packet timestamps.

    `video_pts` is None when the clip has no video stream at all.
    """
    if video_pts is None:
        return {"missingVideo": True, "holes": 0, "longestHoleS": 0.0, "hasAudio": has_audio}
    pts = sorted(video_pts)
    holes = [b - a for a, b in zip(pts, pts[1:]) if b - a >= MIN_HOLE_S]
    return {
        "missingVideo": not pts,
        "holes": len(holes),
        "longestHoleS": max(holes, default=0.0),
        "hasAudio": has_audio,
    }


def is_damaged(report: dict) -> bool:
    return bool(report["missingVideo"] or report["holes"])


def _probe(path: Path) -> Optional[dict]:
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if streams.returncode != 0:
        return None
    kinds = {line.strip() for line in streams.stdout.splitlines()}
    if "video" not in kinds:
        return analyse(None, "audio" in kinds)
    packets = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0"]
        + [str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if packets.returncode != 0:
        return None
    pts = []
    for line in packets.stdout.splitlines():
        try:
            pts.append(float(line.strip().rstrip(",")))
        except ValueError:
            continue
    return analyse(pts, "audio" in kinds)


async def probe(path: Path) -> Optional[dict]:
    """Coverage report for `path`, or None if ffprobe could not read it."""
    if not path.is_file():
        return None
    try:
        return await asyncio.to_thread(_probe, path)
    except Exception as exc:
        logger.warning("Could not check %s for missing video: %s", path.name, exc)
        return None


def record(source_uuid: str, report: dict, clip_name: str, source_name: str = "") -> None:
    """Keep the latest damaged report for a source; a clean clip clears it."""
    if not is_damaged(report):
        _findings.pop(source_uuid, None)
        return
    _findings[source_uuid] = {**report, "sourceUuid": source_uuid, "sourceName": source_name, "clip": clip_name}


async def check(path: Path, source_uuid: str, source_name: str = "") -> None:
    report = await probe(path)
    if report is None:
        return
    if is_damaged(report):
        logger.warning("%s: clip %s is missing video: %s", source_name or source_uuid, path.name, report)
    record(source_uuid, report, path.name, source_name=source_name)


def submit(path: Path, *, source_uuid: Optional[str], source_name: str = "") -> None:
    """Check a finished clip in the background; the caller is inside a frame loop."""
    if not source_uuid:
        return
    task = asyncio.create_task(check(path, source_uuid, source_name), name="clip-video-health")
    _pending.add(task)
    task.add_done_callback(_pending.discard)


def findings() -> list:
    return list(_findings.values())


def reset() -> None:
    """Drop all state. For tests."""
    _findings.clear()


def notice_for(finding: Dict[str, Any]) -> dict:
    name = finding.get("sourceName") or "A camera"
    if finding["missingVideo"]:
        what = "has no video at all"
    else:
        n = finding["holes"]
        what = f"is missing video in {n} place{'' if n == 1 else 's'} (longest {finding['longestHoleS']:.0f}s)"
    if finding["hasAudio"]:
        hint = (
            "Its audio kept recording, which usually means the camera's audio and video "
            "timestamps drift apart; turning off the camera's audio track fixes it."
        )
    else:
        hint = "The camera stopped delivering frames; check its connection and power."
    return {
        "key": f"clip-video-missing:{finding['sourceUuid']}",
        "level": "warning",
        "message": f"{name}'s last clip {what}. {hint}",
    }
