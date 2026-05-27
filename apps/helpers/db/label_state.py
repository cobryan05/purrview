"""Pure functions for deriving image-level label state from bbox+tag data.

Extracted from db_server.manager._recompute_image_labels so the logic can be
unit-tested without any database or HTTP infrastructure.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .types import FALSE_POS_TAG_UUID, VERIFIED_TAG_UUID, ImageLabelStatus


def compute_image_label_states(
    bboxes: List[Tuple[Optional[str], Set[str]]],
) -> Dict[str, ImageLabelStatus]:
    """Derive the per-label presence/absence status for an image from its bboxes.

    Args:
        bboxes: List of ``(label_uuid, set_of_tag_uuids)`` for each bbox.
                Entries where *label_uuid* is ``None`` or empty are ignored.

    Returns:
        A dict mapping ``label_uuid -> ImageLabelStatus``.  Only labels that
        appear in at least one bbox are included; callers are responsible for
        removing stale rows for labels that are no longer referenced.

    Rules (in priority order):
        1. If **any** bbox for a label carries ``VERIFIED_TAG_UUID``     → PRESENT
        2. If **all** bboxes for a label carry ``FALSE_POS_TAG_UUID``   → ABSENT
        3. Otherwise                                                     → UNCERTAIN
    """
    label_tag_sets: Dict[str, List[Set[str]]] = {}
    for label_uuid, tag_uuids in bboxes:
        if label_uuid:
            label_tag_sets.setdefault(label_uuid, []).append(set(tag_uuids))

    result: Dict[str, ImageLabelStatus] = {}
    for label_uuid, tag_sets in label_tag_sets.items():
        if any(VERIFIED_TAG_UUID in tags for tags in tag_sets):
            status = ImageLabelStatus.PRESENT
        elif all(FALSE_POS_TAG_UUID in tags for tags in tag_sets):
            status = ImageLabelStatus.ABSENT
        else:
            status = ImageLabelStatus.UNCERTAIN
        result[label_uuid] = status

    return result
