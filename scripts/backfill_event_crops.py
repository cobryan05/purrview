"""Cut cropped card art for events recorded before the crop existed.

    docker exec -w /project purrview_mono_dev \
        python scripts/backfill_event_crops.py --dry-run
    docker exec -w /project purrview_mono_dev \
        python scripts/backfill_event_crops.py

**Why this is possible at all.** The crop needs to know where the event's subject was,
and `watch_events` has never stored that. The clips do: every clip is written with a
`.boxes.json` sidecar (`apps.helpers.recordings.detectionTrack`) carrying per-frame
normalised boxes, each with the track id, label and confidence of what it framed. On
this install 1,133 of 1,495 events own a clip and 883 of those yield a crop, so most of
the archive can be back-filled from evidence already on disk. Every event this cannot
reach keeps its wide card and nothing regresses.

**Which frame.** The sidecar's frame nearest the event's start, subject to
`MAX_START_DRIFT_S` — the same instant the live path uses, since a snapshot is encoded
at the opening edge. Not the frame with the biggest boxes or the highest confidence:
those pick the most photogenic moment rather than the moment the card claims to show,
and a card whose picture is from twelve seconds into an event is quietly lying about
when it was taken.

**Which boxes.** The event's own tracks, matched by `track_ids_json`, falling back to
every track on the frame when the ids no longer match anything — the same rule the
recorder applies live, in `_focus_rect`. The geometry itself is shared outright
(`apps.helpers.recordings.eventCrop`) so the back-filled crops and the live ones cannot
frame differently.

Safe to re-run: an event that already has a `crop_path` is skipped unless `--force`, and
nothing is deleted.

**Reads and writes the SQLite file directly**, rather than going through db_server's
API. The API was the first choice and was wrong twice over: `WatchEventRead` withholds
`snapshot_path`, so the file could not be named beside the snapshot anyway; and this
endpoint reports failure as `200 {"status": "failure"}`, so a `raise_for_status()` slid
straight past a whole run that wrote 883 files and updated no rows. A maintenance script
that runs once, before the services have even restarted onto the code that knows about
the column, has no business depending on them being up.
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps import DB_DIR, RECORDINGS_SNAPSHOT_DIR  # noqa: E402
from apps.helpers.imageUtils import encode_jpg_bytes  # noqa: E402
from apps.helpers.recordings.detectionTrack import OVERLAY_SUFFIX  # noqa: E402
from apps.helpers.recordings.eventCrop import (  # noqa: E402
    CROP_QUALITY,
    crop_to_rect,
    focus_rect,
    is_worth_cropping,
)
from apps.helpers.timeUtils import parse_iso_utc  # noqa: E402

# An explicit handler, not `basicConfig`: importing `apps` configures the root logger on
# the way in, so by the time this line runs `basicConfig` is a no-op and every line this
# script writes disappears — including the summary, leaving a silent exit 0 that looks
# like "nothing to do".
logger = logging.getLogger("backfill_event_crops")
logger.setLevel(logging.INFO)
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)

#: How far from the event's start a sidecar frame may be and still stand in for it. A
#: clip carries pre-roll and samples at well under 1 Hz, so the nearest frame is usually
#: a second or two out; beyond this it is a different moment of the visit and the crop
#: would frame something the card never claimed.
MAX_START_DRIFT_S = 6.0

#: Suffix the live recorder gives a crop (`_crop` before the extension) — see
#: `WatchEventRecorder._crop_path`. Matched here so back-filled and live files are
#: indistinguishable on disk and the retention sweep needs no special case.
CROP_SUFFIX = "_crop"


def _iso_seconds(value: str) -> Optional[float]:
    parsed = parse_iso_utc(value) if value else None
    return None if parsed is None else parsed.timestamp()


def _pick_frame(sidecar: Dict[str, Any], event_start_s: Optional[float]) -> Optional[Dict[str, Any]]:
    """The sidecar frame closest to the event's start.

    Sidecar times are seconds from the clip's own start, so the clip's `startedAt` is
    what puts them on the same clock as the event. A sidecar with no `startedAt` (or an
    event with no parseable start) falls back to the first frame — for a clip cut around
    one event that is the approach to it, which is a defensible card and is what the
    pre-roll exists to capture.
    """
    frames = [f for f in sidecar.get("frames") or [] if f.get("tracks") or f.get("boxes")]
    if not frames:
        return None

    clip_start_s = _iso_seconds(sidecar.get("startedAt") or "")
    if clip_start_s is None or event_start_s is None:
        return frames[0]

    best = min(frames, key=lambda f: abs((clip_start_s + float(f.get("t", 0.0))) - event_start_s))
    drift = abs((clip_start_s + float(best.get("t", 0.0))) - event_start_s)
    return best if drift <= MAX_START_DRIFT_S else frames[0]


def _boxes_for_event(frame: Dict[str, Any], track_ids: List[Any]) -> List[Tuple[float, float, float, float]]:
    """The event's boxes on this frame, normalised, mirroring the live `_focus_rect`.

    Sidecar `tracks` carry ids and `boxes` do not, so an older sidecar with only raw
    detections gets the whole frame's worth — which is the same fallback the recorder
    takes when the ids match nothing.
    """
    entries = frame.get("tracks") or frame.get("boxes") or []
    wanted = {int(t) for t in track_ids if isinstance(t, (int, float))}
    mine = [b for b in entries if wanted and b.get("id") in wanted]
    chosen = mine or entries
    return [
        (float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"]))
        for b in chosen
        if all(k in b for k in ("x", "y", "w", "h"))
    ]


def _grab_frame(clip: Path, t: float):
    """One decoded frame at `t` seconds into the clip, at its own resolution."""
    capture = cv2.VideoCapture(str(clip))
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def _crop_dest(event: Dict[str, Any]) -> Path:
    """Where a back-filled crop goes: the snapshot root, under the event's own date.

    Keyed by event uuid rather than named like the live recorder's file. The recorder
    composes its name from the source, the rule and a hash of the event key, and
    `WatchEventRead` deliberately does not expose `snapshot_path` — so reproducing that
    name from here would mean either re-deriving it from data the API withholds or
    reading the database directly. The uuid is already unique and already the thing the
    row is found by.

    Inside `RECORDINGS_SNAPSHOT_DIR` because that is one of the manager's managed roots:
    the size cap counts what is written here, the orphan sweep spares it (the path is in
    `crop_path`, which `referenced_recording_paths` covers), and deleting the event
    deletes it.
    """
    day = (event.get("startedAt") or "")[:10] or "unknown"
    return RECORDINGS_SNAPSHOT_DIR / day / f"{event['uuid']}{CROP_SUFFIX}.jpg"


def _events(db: sqlite3.Connection, limit: int) -> List[Dict[str, Any]]:
    rows = db.execute(
        "SELECT uuid, started_at, track_ids_json, crop_path FROM watch_events ORDER BY started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {
            "uuid": uuid,
            "startedAt": started_at,
            "trackIds": json.loads(track_ids or "[]"),
            "hasCrop": bool(crop_path),
        }
        for uuid, started_at, track_ids, crop_path in rows
    ]


def _clips_by_event(db: sqlite3.Connection) -> Dict[str, Path]:
    """Every clip on record, indexed by the event it belongs to."""
    rows = db.execute(
        "SELECT event_uuid, file_path FROM recordings WHERE kind = 'clip' AND event_uuid IS NOT NULL "
        "AND file_path IS NOT NULL"
    ).fetchall()
    clips: Dict[str, Path] = {}
    for event_uuid, file_path in rows:
        clips.setdefault(event_uuid, Path(file_path))
    return clips


def backfill(db: sqlite3.Connection, limit: int, force: bool, dry_run: bool) -> Dict[str, int]:
    stats = {"seen": 0, "written": 0, "already": 0, "no_clip": 0, "no_sidecar": 0, "no_boxes": 0, "failed": 0}
    clips = _clips_by_event(db)

    for event in _events(db, limit):
        stats["seen"] += 1
        uuid = event["uuid"]

        if event.get("hasCrop") and not force:
            stats["already"] += 1
            continue

        clip = clips.get(uuid)
        if clip is None or not clip.is_file():
            stats["no_clip"] += 1
            continue

        sidecar_path = clip.with_name(clip.name + OVERLAY_SUFFIX)
        if not sidecar_path.is_file():
            stats["no_sidecar"] += 1
            continue

        try:
            sidecar = json.loads(sidecar_path.read_text())
            frame_meta = _pick_frame(sidecar, _iso_seconds(event.get("startedAt") or ""))
            if frame_meta is None:
                stats["no_boxes"] += 1
                continue

            boxes = _boxes_for_event(frame_meta, event.get("trackIds") or [])
            if not boxes:
                stats["no_boxes"] += 1
                continue

            if dry_run:
                stats["written"] += 1
                logger.info("would crop %s from %s @ %.2fs", uuid, clip.name, float(frame_meta.get("t", 0.0)))
                continue

            # Decoded before the rect is computed, not after: the rect has to be fitted
            # to the *frame's* aspect (see `focus_rect`), and nothing in the sidecar
            # records the clip's dimensions. The cost is decoding a frame for the
            # ~150 events that turn out not to be worth cropping.
            image = _grab_frame(clip, float(frame_meta.get("t", 0.0)))
            if image is None:
                stats["failed"] += 1
                logger.warning("could not decode %s at %.2fs", clip.name, float(frame_meta.get("t", 0.0)))
                continue

            height, width = image.shape[:2]
            rect = focus_rect(boxes, width / max(height, 1))
            if not is_worth_cropping(rect):
                # The event filled the frame, or its boxes were spread across it. A crop
                # here would be the snapshot again under a second name.
                stats["no_boxes"] += 1
                continue

            dest = _crop_dest(event)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(encode_jpg_bytes(crop_to_rect(image, rect), quality=CROP_QUALITY))

            # The row last, and committed per event: a path in the database that names a
            # file not yet written is a card that 404s, whereas a file with no row is
            # swept up as an orphan — recoverable, and the cheaper way round.
            db.execute("UPDATE watch_events SET crop_path = ? WHERE uuid = ?", (str(dest), uuid))
            db.commit()
            stats["written"] += 1
        except Exception as e:
            stats["failed"] += 1
            logger.warning("event %s: %s", uuid, e)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=100000, help="Most events to consider (newest first).")
    parser.add_argument("--force", action="store_true", help="Re-cut crops for events that already have one.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written, touch nothing.")
    parser.add_argument("--db", default=str(DB_DIR / "db.sqlite"), help="SQLite file to read and update.")
    args = parser.parse_args()

    # `timeout` rather than the default 5s: db_server holds this file open, and a
    # per-event commit that collides with one of its writes should wait rather than
    # abandon the row whose file has already been written.
    db = sqlite3.connect(args.db, timeout=30.0)
    try:
        stats = backfill(db, args.limit, args.force, args.dry_run)
    finally:
        db.close()
    logger.info(
        "%s%s of %s events: %s cropped, %s already had one, %s no clip, %s no sidecar, "
        "%s nothing to frame, %s failed",
        "DRY RUN — " if args.dry_run else "",
        stats["written"],
        stats["seen"],
        stats["written"],
        stats["already"],
        stats["no_clip"],
        stats["no_sidecar"],
        stats["no_boxes"],
        stats["failed"],
    )
    return 1 if stats["failed"] and not stats["written"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
