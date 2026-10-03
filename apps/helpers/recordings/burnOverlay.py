"""Render a clip with its detection overlay burnt in, on demand.

`detectionTrack.py` argues at length against storing annotated video, and that argument
still holds: every *stored* clip stays pristine, so "save this frame for annotation"
can never put a boxed image into the dataset, and the overlay stays correctable. This
module is the other half of that trade — a clip someone wants to send to a person who
has no access to the app needs the boxes *in* the picture, so they are composited here,
once, into a file that is handed straight to the browser and deleted.

The drawing deliberately mirrors `clipOverlay.js` rather than improving on it: the same
box colour, the same tag layout, the same red "the rule was firing here" frame. An
export that looked different from the player would read as a second, disagreeing opinion
about what the detector saw.

**Alignment.** ffmpeg honours an mp4 edit list, so for a buffer-cut clip the sidecar's
t=0 is this file's t=0 and no lead correction is needed — the same position Chromium is
in. A chunk-assembled clip (`partCount > 1`) physically plays every part's untrimmed
head and the player declines to correct for it; this does too, so the export shows
exactly the misalignment the player shows rather than a different one.
"""

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Sidecar shapes this renderer understands, kept in step with `SUPPORTED_VERSIONS` in
#: clipOverlay.js. A newer document is refused rather than guessed at.
SUPPORTED_VERSIONS = frozenset({1, 2})

#: Hold applied when a sidecar omits `maxStalenessS`, matching the player's fallback.
DEFAULT_STALENESS_S = 1.5

#: Box colour when the caller supplies no per-label palette — clipOverlay.js's own
#: fallback, which is what the event modal actually draws today.
DEFAULT_BOX_COLOUR = (56, 189, 248)

#: The red frame's thickness and colour, as the player draws them.
EVENT_FRAME_WIDTH = 4
EVENT_FRAME_COLOUR = (239, 68, 68)

#: Tag geometry, mirroring `drawTag`.
TAG_HEIGHT = 16
TAG_PAD = 4
TAG_FONT_PX = 12

#: Segments shorter than this are dropped: they are boundary arithmetic, not moments,
#: and the concat demuxer rounds them away anyway.
MIN_SEGMENT_S = 0.002

#: Ceiling on one render. A clip is minutes long at worst; anything past this is a
#: wedged ffmpeg, and a request that never returns is worse than one that fails.
RENDER_TIMEOUT_S = 900


def _event_window(doc: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """The rule's span in clip seconds, or None when the sidecar marks none."""
    event = doc.get("event") or {}
    start = event.get("startS")
    if start is None:
        return None
    try:
        start_s = float(start)
    except (TypeError, ValueError):
        return None
    end = event.get("endS")
    try:
        end_s = float(end)
    except (TypeError, ValueError):
        end_s = float("inf")
    return start_s, max(start_s, end_s)


def sample_at(doc: Dict[str, Any], t: float) -> Optional[Dict[str, Any]]:
    """The sample in effect at `t`, or None — `sampleAt` in clipOverlay.js, in Python.

    Deliberately a transliteration rather than an improvement. The export and the player
    must agree frame for frame; two independently reasonable implementations of "held
    until the next one" would disagree at exactly the boundaries someone reviewing a
    marginal detection is looking at.
    """
    frames = doc.get("frames") or []
    try:
        max_stale = float(doc.get("maxStalenessS"))
    except (TypeError, ValueError):
        max_stale = DEFAULT_STALENESS_S
    found = None
    for frame in frames:
        if float(frame.get("t", 0.0)) > t:
            break
        found = frame
    if found is None:
        return None
    return found if t - float(found.get("t", 0.0)) <= max_stale else None


def frame_times(path: Path) -> List[float]:
    """Presentation time of every video frame, ascending.

    Needed because the sidecar's times do not land on frame boundaries: a sample is
    stamped when the watcher's copy of a frame came back out of its decoder, a few
    milliseconds *after* that frame's own position in the clip. See
    `_sample_times_by_frame` for what goes wrong without this.

    A full decode, which is why it is worth saying it is cheap in context: it costs a
    fraction of the encode that follows it, and the alternative is guessing at a
    tolerance.
    """
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        timeout=600,
    )
    if proc.returncode != 0:
        return []
    times: List[float] = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        value = line.strip().rstrip(",")
        if not value or value == "N/A":
            continue
        try:
            times.append(float(value))
        except ValueError:
            continue
    return sorted(times)


def _snap(value: float, times: Sequence[float]) -> float:
    """`value` moved back to the last frame at or before it, or 0 when it precedes them all."""
    found = 0.0
    for t in times:
        if t > value:
            break
        found = t
    return found


def _segments_by_frame(
    doc: Dict[str, Any],
    duration_s: float,
    times: Sequence[float],
    max_stale: float,
) -> List[Dict[str, Any]]:
    """Segment on the video's own frame boundaries.

    This exists because of a half-second bug. A sidecar sample describes one decoded
    frame, but its time is a few milliseconds *past* that frame's position in the clip
    (6.6821 for the frame at 6.68, and so on), and ffmpeg's `overlay` only takes up a new
    overlay image on the next main frame at or after its timestamp. Segmenting on the
    sample times therefore drew every box one video frame late — at the couple of frames
    a second these cameras produce, nearly half a second of visible lag.

    The player never shows it: it re-reads the sidecar continuously against
    `currentTime`, so it catches up within those few milliseconds while the same frame is
    still on screen. Only a composite has to commit to one image per frame.

    So each sample is attributed to the frame it was taken from — the last frame at or
    before its timestamp — and the segments run frame to frame. Staleness is still
    measured against the sample's own time, which is what the player does.
    """
    starts = [t for t in times if t < duration_s]
    if not starts:
        return []
    # A clip whose first frame sits past zero still has to be covered: the overlay input
    # must start at 0 or ffmpeg's framesync has nothing to show until it does.
    if starts[0] > 0.0:
        starts.insert(0, 0.0)

    samples = [f for f in (doc.get("frames") or []) if f.get("t") is not None]
    window = _event_window(doc)
    event_from = _snap(window[0], times) if window else None
    event_to = _snap(window[1], times) if window and window[1] != float("inf") else float("inf")

    segments: List[Dict[str, Any]] = []
    # Both lists are in time order, so one pointer walks the samples across the whole
    # clip. Re-scanning them per frame would be frames x samples, which on a 30 fps clip
    # with a full sidecar is a hundred million comparisons to produce the same answer.
    cursor = 0
    held = None
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else float(duration_s)
        if end - start < MIN_SEGMENT_S:
            continue
        # Every sample belonging to this frame or an earlier one; the last wins. `t < end`
        # rather than `t <= start` is the whole correction: it admits the sample stamped
        # just after this frame, which is the one taken *from* it.
        while cursor < len(samples) and float(samples[cursor]["t"]) < end:
            held = samples[cursor]
            cursor += 1
        sample = held
        if sample is not None and start - float(sample["t"]) > max_stale:
            sample = None
        active = event_from is not None and event_from <= start <= event_to

        previous = segments[-1] if segments else None
        if previous is not None and previous["sample"] is sample and previous["eventActive"] == active:
            previous["end"] = end
            continue
        segments.append({"start": start, "end": end, "sample": sample, "eventActive": active})
    return segments


def overlay_segments(
    doc: Dict[str, Any],
    duration_s: float,
    times: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Cut `[0, duration_s]` into stretches over which the overlay never changes.

    The overlay is a step function of time: it changes only when a new sample takes
    effect, when the current one goes stale, or when the event window opens or closes.
    One still per stretch reproduces the player far more cheaply than a still per video
    frame would.

    Adjacent stretches that would draw the same thing are merged, so a clip where the
    detector saw nothing for a minute costs one image rather than a hundred.

    Pass `times` — the video's own frame times — whenever they can be measured: they are
    what puts each box on the frame it was taken from rather than the one after it. The
    boundary-based fallback below is for a clip ffprobe could not enumerate, and it draws
    every box up to one frame late.

    :returns: dicts of `start`, `end`, `sample` (or None) and `eventActive`, in order.
    """
    if duration_s <= 0:
        return []
    try:
        max_stale = float(doc.get("maxStalenessS"))
    except (TypeError, ValueError):
        max_stale = DEFAULT_STALENESS_S

    if times:
        return _segments_by_frame(doc, duration_s, times, max_stale)

    bounds = {0.0, float(duration_s)}
    for frame in doc.get("frames") or []:
        t = float(frame.get("t", 0.0))
        # Both edges of the frame's influence: when it takes effect, and when holding it
        # any longer would invent a detection the watcher never made.
        for edge in (t, t + max_stale):
            if 0.0 < edge < duration_s:
                bounds.add(edge)
    window = _event_window(doc)
    if window:
        for edge in window:
            if 0.0 < edge < duration_s:
                bounds.add(edge)

    ordered = sorted(bounds)
    segments: List[Dict[str, Any]] = []
    for start, end in zip(ordered, ordered[1:]):
        if end - start < MIN_SEGMENT_S:
            continue
        mid = (start + end) / 2.0
        sample = sample_at(doc, mid)
        active = bool(window and window[0] <= mid <= window[1])
        previous = segments[-1] if segments else None
        # Identity, not equality: two distinct samples holding equal boxes are still two
        # moments, and `is` is what makes the merge cheap on a long empty stretch.
        if previous is not None and previous["sample"] is sample and previous["eventActive"] == active:
            previous["end"] = end
            continue
        segments.append({"start": start, "end": end, "sample": sample, "eventActive": active})
    return segments


def _font():
    """A 12px face, or whatever Pillow can give us.

    The tags are the only text in the picture and an export is no place to fail over a
    font, so every step degrades instead of raising: a real face if one is installed,
    Pillow's bundled default at the right size if not, and its bitmap default as the
    last resort.
    """
    from PIL import ImageFont

    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        try:
            return ImageFont.truetype(path, TAG_FONT_PX)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=TAG_FONT_PX)
    except TypeError:
        return ImageFont.load_default()


def _text_width(draw, text: str, font) -> int:
    try:
        return int(draw.textlength(text, font=font))
    except AttributeError:
        return int(draw.textsize(text, font=font)[0])


def _parse_colour(value: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """`#rrggbb` → RGB, or None for anything else."""
    if not isinstance(value, str):
        return None
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except ValueError:
        return None


def render_segment_png(
    segment: Dict[str, Any],
    width: int,
    height: int,
    dest: Path,
    *,
    rule_name: str = "",
    label_colours: Optional[Dict[str, str]] = None,
) -> None:
    """Draw one segment's overlay as a transparent PNG the size of the video frame.

    Full frame rather than letterboxed: `clipOverlay.js` has to do contain-math because
    its canvas covers an element that is larger than the picture, and here the canvas
    *is* the picture.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _font()
    palette = label_colours or {}

    def tag(text: str, x: float, y: float, colour: Tuple[int, int, int]) -> None:
        box_w = _text_width(draw, text, font) + 2 * TAG_PAD
        draw.rectangle([x, y, x + box_w, y + TAG_HEIGHT], fill=(0, 0, 0, 166))
        draw.text((x + TAG_PAD, y + 2), text, font=font, fill=(*colour, 255))

    if segment.get("eventActive"):
        half = EVENT_FRAME_WIDTH / 2
        draw.rectangle(
            [half, half, width - half, height - half],
            outline=(*EVENT_FRAME_COLOUR, 255),
            width=EVENT_FRAME_WIDTH,
        )
        if rule_name:
            tag(f"● {rule_name}", EVENT_FRAME_WIDTH + TAG_PAD, EVENT_FRAME_WIDTH + TAG_PAD, EVENT_FRAME_COLOUR)

    for box in (segment.get("sample") or {}).get("boxes") or []:
        colour = _parse_colour(palette.get(box.get("labelUuid"))) or DEFAULT_BOX_COLOUR
        x = float(box.get("x", 0.0)) * width
        y = float(box.get("y", 0.0)) * height
        w = float(box.get("w", 0.0)) * width
        h = float(box.get("h", 0.0)) * height
        draw.rectangle([x, y, x + w, y + h], outline=(*colour, 255), width=2)
        conf = box.get("conf")
        name = box.get("label") or "?"
        text = f"{name} {round(float(conf) * 100)}%" if conf else name
        tag(text, x, max(0.0, y - TAG_HEIGHT), colour)

    image.save(dest, format="PNG")


def probe_video(path: Path) -> Tuple[int, int, float]:
    """`(width, height, duration_s)` for a video file.

    Measured rather than taken from the sidecar: `durationS` is what the clip was *meant*
    to be, and the overlay has to be as long as the file actually is or the last stretch
    of video plays bare.
    """
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr[-300:].decode('utf-8', 'replace')}")
    info = json.loads(proc.stdout or b"{}")
    stream = (info.get("streams") or [{}])[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration = float((info.get("format") or {}).get("duration") or 0.0)
    if width <= 0 or height <= 0:
        raise RuntimeError("clip has no decodable video stream")
    return width, height, duration


def _concat_list(segments: Sequence[Dict[str, Any]], images: Sequence[Path]) -> str:
    """A concat-demuxer script holding each still for its segment's length.

    The last entry is repeated without a duration because the demuxer applies a
    `duration` to the *preceding* file and drops the final one otherwise, which would
    leave the closing stretch of the clip unannotated.
    """
    lines: List[str] = []
    for segment, image in zip(segments, images):
        lines.append(f"file '{image}'")
        lines.append(f"duration {segment['end'] - segment['start']:.4f}")
    if images:
        lines.append(f"file '{images[-1]}'")
    return "\n".join(lines) + "\n"


def burn_overlay(clip_path: Path, doc: Dict[str, Any], dest: Path, *, label_colours=None) -> Path:
    """Composite `doc`'s boxes onto `clip_path`, writing an mp4 to `dest`.

    Synchronous and CPU-bound — call it off the event loop. `nice`/`ionice` are not
    cosmetic: this runs on the machine serving the live app and the inference loop, and
    an unprioritised x264 encode makes the UI visibly stutter for the length of the
    render. The export has no deadline; the app does.
    """
    if doc.get("version") not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported overlay version: {doc.get('version')!r}")

    width, height, duration = probe_video(clip_path)
    segments = overlay_segments(doc, duration, frame_times(clip_path))
    # Not just "are there segments": a sidecar whose samples all fall outside the clip
    # yields one empty segment covering the whole thing, and rendering that would hand
    # back a re-encode of the original labelled "annotated". The user would report the
    # boxes as missing against a successful download.
    if not any(segment["sample"] or segment["eventActive"] for segment in segments):
        raise ValueError("overlay covers none of this clip")

    rule_name = (doc.get("event") or {}).get("ruleName") or ""
    work = Path(tempfile.mkdtemp(prefix="burn-overlay-"))
    try:
        images: List[Path] = []
        for index, segment in enumerate(segments):
            image = work / f"{index:05d}.png"
            render_segment_png(segment, width, height, image, rule_name=rule_name, label_colours=label_colours)
            images.append(image)

        list_path = work / "segments.txt"
        list_path.write_text(_concat_list(segments, images))

        partial = dest.with_name(dest.name + ".part")
        cmd = [
            "nice",
            "-n",
            "19",
            "ionice",
            "-c3",
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(clip_path),
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-filter_complex",
            # `eof_action=pass` so a clip that outlives its sidecar keeps playing bare
            # rather than being cut short at the last annotated moment.
            f"[1:v]scale={width}:{height},format=rgba[ov];[0:v][ov]overlay=0:0:eof_action=pass[v]",
            "-map",
            "[v]",
            # Optional: these clips usually carry no audio, and `-map 0:a` would fail the
            # whole render on the ones that do not.
            "-map",
            "0:a?",
            # Keep the filter's own timestamps instead of letting the encoder resample to
            # a constant rate. Without it the output is CFR 25 — 943 frames for a clip
            # with 69 — and the picture, which only changes a couple of times a second,
            # is then held across frames on which the overlay is free to change. Stepping
            # through such a file shows the boxes move several frames away from the
            # picture they belong to. With it, and with the segments snapped to the
            # source's frame times, every output frame is one source frame carrying its
            # own sample's boxes. `-vsync 0` rather than `-fps_mode passthrough`: the
            # container ships ffmpeg 4.4, which predates the newer spelling.
            "-vsync",
            "0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "copy",
            "-f",
            "mp4",
            str(partial),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=RENDER_TIMEOUT_S)
        if proc.returncode != 0 or not partial.exists():
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg burn failed: {proc.stderr[-400:].decode('utf-8', 'replace')}")
        partial.replace(dest)
        return dest
    finally:
        shutil.rmtree(work, ignore_errors=True)


async def burn_overlay_async(clip_path: Path, doc: Dict[str, Any], dest: Path, *, label_colours=None) -> Path:
    return await asyncio.to_thread(burn_overlay, clip_path, doc, dest, label_colours=label_colours)
