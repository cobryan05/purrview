"""Backfill `suppression_group_uuids`/`suppression_group_iou_thresholds` into the
sidecar JSON of every already-trained object_detection model.

    docker exec -w /project purrview_mono_dev \
        python scripts/backfill_suppression_groups.py --dry-run
    docker exec -w /project purrview_mono_dev \
        python scripts/backfill_suppression_groups.py

**Why this is needed at all.** These two sidecar fields are what lets inference-time
NMS (`apps.helpers.yoloUtils.decode_detections`) suppress a label's `exclusive_variants`
children together instead of by raw class_id — see `Label.exclusive_variants` in
`apps/helpers/db/types.py`. A model trained *before* a label was marked
`exclusive_variants` has a sidecar with no such fields, and `decode_detections` falls
back to today's per-class_id behaviour for it (safe, but not what the flag now asks
for). This script re-derives the fields purely from data already on disk plus the
current label tree — no retraining needed, since suppression groups are a property of
the label hierarchy, not of anything the model itself learned.

**Which models.** Every `<name>.json` under `$MODELS_ROOT` whose `model_kind` is
`object_detection` and which carries a `label_uuids` list — that positional array
(index = class_id, same order as `classes`) is the only thing needed to recompute the
two new fields; the model's own weights are untouched.

**Reads the SQLite file directly**, mirroring `backfill_event_crops.py`'s reasoning:
this is a maintenance script that may run before the services holding the new column
have restarted, and the whole computation (`Label.parent_uuid` + `exclusive_variants` +
`exclusive_variants_iou_threshold`) is a handful of rows, not worth standing up an HTTP
round-trip for.

Safe to re-run: a sidecar whose fields already match the freshly-derived values is
left untouched (mtime included) unless `--force`; nothing is deleted.
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps import DB_DIR  # noqa: E402

logger = logging.getLogger("backfill_suppression_groups")
logger.setLevel(logging.INFO)
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)

MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/models"))


class _LabelRow:
    __slots__ = ("uuid", "parent_uuid", "exclusive_variants", "exclusive_variants_iou_threshold")

    def __init__(self, uuid: str, parent_uuid: Optional[str], exclusive_variants: bool, threshold: Optional[float]):
        self.uuid = uuid
        self.parent_uuid = parent_uuid
        self.exclusive_variants = exclusive_variants
        self.exclusive_variants_iou_threshold = threshold


def _load_labels(db: sqlite3.Connection) -> Dict[str, _LabelRow]:
    rows = db.execute(
        "SELECT uuid, parent_uuid, exclusive_variants, exclusive_variants_iou_threshold FROM labels"
    ).fetchall()
    return {
        uuid: _LabelRow(uuid, parent_uuid, bool(exclusive_variants), threshold)
        for uuid, parent_uuid, exclusive_variants, threshold in rows
    }


def _suppression_group(uuid: str, labels: Dict[str, _LabelRow]) -> Tuple[str, Optional[float]]:
    """Mirrors DbClient._suppression_group (apps/helpers/db/db_client.py) exactly —

    keep the two in sync; this script exists only because that one runs inside a
    request against the live DB, and a sidecar may need updating with no service
    running at all.
    """
    lbl = labels.get(uuid)
    if lbl is not None and lbl.exclusive_variants:
        return uuid, lbl.exclusive_variants_iou_threshold
    parent_uuid = lbl.parent_uuid if lbl is not None else None
    if parent_uuid is not None:
        parent = labels.get(parent_uuid)
        if parent is not None and parent.exclusive_variants:
            return parent_uuid, parent.exclusive_variants_iou_threshold
    return uuid, None


def _sidecar_paths() -> List[Path]:
    if not MODELS_ROOT.is_dir():
        return []
    return sorted(MODELS_ROOT.glob("*.json"))


def backfill(labels: Dict[str, _LabelRow], force: bool, dry_run: bool) -> Dict[str, int]:
    stats = {"seen": 0, "updated": 0, "unchanged": 0, "not_object_detection": 0, "no_label_uuids": 0, "failed": 0}

    for path in _sidecar_paths():
        stats["seen"] += 1
        try:
            sidecar = json.loads(path.read_text())
        except Exception as e:
            stats["failed"] += 1
            logger.warning("%s: could not read sidecar: %s", path.name, e)
            continue

        if sidecar.get("model_kind") != "object_detection":
            stats["not_object_detection"] += 1
            continue

        label_uuids = sidecar.get("label_uuids") or []
        if not label_uuids:
            stats["no_label_uuids"] += 1
            continue

        groups = [_suppression_group(uuid, labels) for uuid in label_uuids]
        new_group_uuids = [g for g, _thr in groups]
        new_thresholds = [thr for _g, thr in groups]

        unchanged = (
            sidecar.get("suppression_group_uuids") == new_group_uuids
            and sidecar.get("suppression_group_iou_thresholds") == new_thresholds
        )
        if unchanged and not force:
            stats["unchanged"] += 1
            continue

        if dry_run:
            stats["updated"] += 1
            grouped = sum(1 for g, uuid in zip(new_group_uuids, label_uuids) if g != uuid)
            logger.info("would update %s (%d/%d classes now grouped)", path.name, grouped, len(label_uuids))
            continue

        sidecar["suppression_group_uuids"] = new_group_uuids
        sidecar["suppression_group_iou_thresholds"] = new_thresholds
        path.write_text(json.dumps(sidecar, indent=2))
        stats["updated"] += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Rewrite sidecars even if already up to date.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change, touch nothing.")
    parser.add_argument("--db", default=str(DB_DIR / "db.sqlite"), help="SQLite file to read labels from.")
    args = parser.parse_args()

    db = sqlite3.connect(args.db, timeout=30.0)
    try:
        labels = _load_labels(db)
    finally:
        db.close()

    stats = backfill(labels, args.force, args.dry_run)
    logger.info(
        "%s%s of %s sidecars: %s updated, %s already current, %s not object_detection, "
        "%s missing label_uuids, %s failed",
        "DRY RUN — " if args.dry_run else "",
        stats["updated"],
        stats["seen"],
        stats["updated"],
        stats["unchanged"],
        stats["not_object_detection"],
        stats["no_label_uuids"],
        stats["failed"],
    )
    return 1 if stats["failed"] and not stats["updated"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
