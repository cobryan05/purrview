"""Catalogue what old clips contain, for events recorded before `watch_event_labels` existed.

    docker exec -w /project purrview_mono_dev \
        python scripts/backfill_event_present_labels.py --dry-run
    docker exec -w /project purrview_mono_dev \
        python scripts/backfill_event_present_labels.py

**Why this is possible at all.** The catalogue records every label present at some point
in an event's clip window, which the live path builds from the watcher's own presence log.
That log is in memory and is gone. The clips are not: each is written with a `.boxes.json`
sidecar (`apps.helpers.recordings.detectionTrack`) holding per-frame tracks over exactly
the clip's window — pre-roll included, since the window is the clip's — so the archive can
be catalogued from evidence already on disk.

**Only `frames[].tracks[]`, never `frames[].boxes[]`** — the same rule the live path
applies to its own feed, spelled here as `presenceLog.labels_from_overlay`. Tracks are what
the tracker confirmed; boxes are raw model output at the 0.1 floor ByteTrack needs, and
cataloguing those would mark nearly every clip as containing nearly every label.

An event with no clip, or a clip with no sidecar, is skipped rather than guessed at: its
span is no longer observable from anything on disk.

Safe to re-run. Rows are unioned per (event, label), keeping the higher confidence, so a
second pass over the same sidecars writes nothing; `--force` is unnecessary and does not
exist. Nothing is deleted.

**Reads and writes the SQLite file directly**, for the reasons `backfill_event_crops.py`
spells out: it runs once and has no business depending on the services being up.

⚠️ **Run it after the container has restarted onto this code, not before.** `watch_event_labels`
is created by the alembic revision db_server applies on startup, and this script deliberately
does not open a `DbClient`, so nothing here would create it.
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps import DB_DIR  # noqa: E402
from apps.helpers.recordings.detectionTrack import OVERLAY_SUFFIX  # noqa: E402
from apps.helpers.recordings.presenceLog import labels_from_overlay, merge_label_confidences  # noqa: E402

# An explicit handler, not `basicConfig`: importing `apps` configures the root logger on
# the way in, so by then `basicConfig` is a no-op and every line this script writes
# disappears — including the summary, leaving a silent exit 0 that looks like "nothing
# to do".
logger = logging.getLogger("backfill_event_present_labels")
logger.setLevel(logging.INFO)
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)


def _sidecars_by_event(db: sqlite3.Connection) -> Dict[str, List[Path]]:
    """Every recording's sidecar path, indexed by the event that owns it.

    Both kinds, not clips only: a timelapse writes its own sidecar over the frames it
    kept, and while its time base is video seconds — meaningless here — the *set* of
    labels in it is exactly as valid as a clip's. An event whose only media is a
    timelapse is precisely the long, quiet event whose catalogue is most worth having.
    """
    rows = db.execute(
        "SELECT event_uuid, file_path, overlay_path FROM recordings "
        "WHERE event_uuid IS NOT NULL AND file_path IS NOT NULL"
    ).fetchall()
    found: Dict[str, List[Path]] = {}
    for event_uuid, file_path, overlay_path in rows:
        sidecar = Path(overlay_path) if overlay_path else Path(file_path + OVERLAY_SUFFIX)
        found.setdefault(event_uuid, []).append(sidecar)
    return found


def _catalogued_peaks(db: sqlite3.Connection) -> Dict[Tuple[str, str], float]:
    """Every (event, label) already catalogued → its recorded peak.

    Read up front so `--dry-run` can report what would actually change. Without it a
    second dry run over a backfilled archive lists every event again, contradicting the
    one promise the flag makes.
    """
    rows = db.execute("SELECT event_uuid, label_uuid, peak_confidence FROM watch_event_labels").fetchall()
    return {(event_uuid, label_uuid): float(peak or 0.0) for event_uuid, label_uuid, peak in rows}


def _events(db: sqlite3.Connection, limit: int) -> List[str]:
    rows = db.execute(
        "SELECT uuid FROM watch_events ORDER BY started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [uuid for (uuid,) in rows]


def _write(db: sqlite3.Connection, event_uuid: str, labels: Dict[str, float]) -> int:
    """Union labels into one event's catalogue. Returns rows inserted or raised.

    `ON CONFLICT ... DO UPDATE` with a `WHERE` on the excluded value, rather than a read
    followed by a write: it keeps the higher confidence in one statement and makes
    re-running the script a no-op rather than a churn of identical updates.
    """
    written = 0
    for label_uuid, confidence in sorted(labels.items()):
        cursor = db.execute(
            "INSERT INTO watch_event_labels (event_uuid, label_uuid, peak_confidence) VALUES (?, ?, ?) "
            "ON CONFLICT(event_uuid, label_uuid) DO UPDATE SET peak_confidence = excluded.peak_confidence "
            "WHERE excluded.peak_confidence > watch_event_labels.peak_confidence",
            (event_uuid, label_uuid, float(confidence)),
        )
        written += cursor.rowcount or 0
    return written


def backfill(db: sqlite3.Connection, limit: int, dry_run: bool) -> Dict[str, int]:
    stats = {"seen": 0, "catalogued": 0, "labels": 0, "no_media": 0, "no_sidecar": 0, "no_labels": 0, "failed": 0}
    sidecars = _sidecars_by_event(db)
    already = _catalogued_peaks(db)

    for event_uuid in _events(db, limit):
        stats["seen"] += 1
        paths = [p for p in sidecars.get(event_uuid, [])]
        if not paths:
            stats["no_media"] += 1
            continue
        present = [p for p in paths if p.is_file()]
        if not present:
            stats["no_sidecar"] += 1
            continue
        try:
            labels: Dict[str, float] = {}
            for path in present:
                found = labels_from_overlay(json.loads(path.read_text()))
                labels = merge_label_confidences(labels.items(), found.items())
            if not labels:
                # The sidecar exists and holds no confirmed track — a clip cut around a
                # detection the tracker never confirmed. A real answer, not a failure.
                stats["no_labels"] += 1
                continue
            if dry_run:
                # The same condition `_write`'s `ON CONFLICT ... WHERE` applies, so the
                # count reported is the count a real run would write.
                new = {uuid: conf for uuid, conf in labels.items() if conf > already.get((event_uuid, uuid), -1.0)}
                if not new:
                    continue
                stats["catalogued"] += 1
                stats["labels"] += len(new)
                logger.info("would catalogue %s: %s labels", event_uuid, len(new))
                continue
            written = _write(db, event_uuid, labels)
            db.commit()
            if written:
                stats["catalogued"] += 1
                stats["labels"] += written
        except Exception as e:
            stats["failed"] += 1
            logger.warning("event %s: %s", event_uuid, e)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=100000, help="Most events to consider (newest first).")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written, touch nothing.")
    parser.add_argument("--db", default=str(DB_DIR / "db.sqlite"), help="SQLite file to read and update.")
    args = parser.parse_args()

    # `timeout` rather than the default 5s: db_server holds this file open, and a commit
    # that collides with one of its writes should wait rather than abandon the event.
    db = sqlite3.connect(args.db, timeout=30.0)
    try:
        # Checked once, up front. The table arrives with the alembic revision db_server
        # applies on startup; run before that restart, every single event would fail with
        # the same "no such table" and the summary would read as a broken archive.
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='watch_event_labels'").fetchone():
            logger.error(
                "%s has no watch_event_labels table — restart the container so db_server "
                "applies the migration, then run this again.",
                args.db,
            )
            return 1
        stats = backfill(db, args.limit, args.dry_run)
    finally:
        db.close()
    logger.info(
        "%s%s of %s events catalogued (%s label rows): %s no media, %s no sidecar, " "%s nothing confirmed, %s failed",
        "DRY RUN — " if args.dry_run else "",
        stats["catalogued"],
        stats["seen"],
        stats["labels"],
        stats["no_media"],
        stats["no_sidecar"],
        stats["no_labels"],
        stats["failed"],
    )
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
