"""Single source of truth for the `exclusive_variants` grouping computation.

Used by three call sites that cannot share a DB session with each other —
`DbClient.derive_training_targets` (`apps/helpers/db/db_client.py`, has a `Label`
ORM object per uuid), `scripts/backfill_suppression_groups.py` (reads the labels
table directly with sqlite3, no ORM), and `inference_server`'s suppression-groups
preview/apply endpoints (DB-free by design — receives the label tree as a plain
dict from a caller that already fetched it). All three need the exact same answer
for "which group does this class belong to, and at what threshold", so the
decision lives here once rather than three times risking drift.
"""

from typing import Mapping, Optional, Tuple


def suppression_group_for(
    label_uuid: str,
    parent_of: Mapping[str, Optional[str]],
    exclusive_variants_of: Mapping[str, bool],
    threshold_of: Mapping[str, Optional[float]],
) -> Tuple[str, Optional[float]]:
    """Nearest exclusive-variants group `label_uuid` belongs to, or itself if none.

    A label with `exclusive_variants=True` is a group root — it and its direct
    children (siblings, alternate guesses for the same physical object) share its
    group and its threshold, so NMS treats them as mutually exclusive instead of by
    raw class_id. Anything else returns `(label_uuid, None)`, which every caller
    must treat as "ungrouped, use per-class_id NMS" — today's default behavior.

    All three mappings are keyed by label uuid; a uuid absent from any of them is
    treated the same as an explicit False/None (unknown labels never group).
    """
    if exclusive_variants_of.get(label_uuid):
        return label_uuid, threshold_of.get(label_uuid)
    parent_uuid = parent_of.get(label_uuid)
    if parent_uuid is not None and exclusive_variants_of.get(parent_uuid):
        return parent_uuid, threshold_of.get(parent_uuid)
    return label_uuid, None
