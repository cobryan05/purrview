"""Shared cross-model detection merge/dedup helper.

Extracted from CollectTrainingImagesTask._run_inference (Phase 5A of
LIVE_COLLECTION_PLAN.md). Merges *fresh* per-model detections by IoU overlap
plus label hierarchy — this is a distinct concern from runModelOnResults.js's
relabel-candidate merge (which keys on existing bboxUuid, not IoU; see
NOTES.md "Phase 5's 'shared merge helper' premise is imprecise"). Do not try
to unify the two.

Only CollectTrainingImagesTask consumes this today. `_filter_chain_runner.py`,
PreviewModelRelabelTask, and DeduplicateOverlappingBBoxesTask are plausible
future adopters but each is its own risk surface and is out of scope here.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping, Sequence

from apps.helpers.bboxUtils import iou_xywh


class MergePolicy(str, Enum):
    """Which model's detection wins when two models produce colliding
    (same-label, overlapping) detections.

    FIRST_WINS (default — today's only behaviour, pinned by
    TestInferenceMergeParity): detections are flattened in model-priority
    order (the order models are configured on the ModelGroup) and, for a
    same-label collision, the earlier/higher-priority model's detection
    survives.

    LAST_WINS: the mirror image. Detections are flattened in *reverse*
    model-priority order, so for a same-label collision the later/
    lower-priority model's detection survives instead.

    Hierarchy-aware supersede (a more-specific label beats a coarser
    ancestor label, regardless of which model produced which) is preserved
    identically under both policies: it is not a priority-order rule, it is
    a structural one — whichever detection is processed *later* against an
    already-kept box always supersedes it if more specific, and is always
    dropped if coarser, independent of which model contributed it. Flipping
    the flattening order therefore does not change the ancestor/descendant
    outcome for a given *pair* of detections in isolation — it only changes,
    for same-label collisions, which model's copy is the survivor, and it
    can change *which* detection ends up "already kept" (and therefore doing
    the superseding) when three or more colliding boxes from different
    models interact — the same knock-on effect FIRST_WINS already has today,
    just walked from the opposite end of the model list.
    """

    FIRST_WINS = "first_wins"
    LAST_WINS = "last_wins"


def merge_detections_by_iou_and_hierarchy(
    results_per_model: Sequence[Sequence[object]],
    *,
    label_ancestry: Mapping[str, "set[str]"],
    overlap_iou_threshold: float,
    policy: MergePolicy = MergePolicy.FIRST_WINS,
) -> list:
    """Flattens per-model detection lists (in model-priority order) and applies
    cross-model, hierarchy-aware IoU dedup.

    For each detection (processed in `policy`-determined order), compare
    against already-kept boxes:
      - same label_uuid                       -> drop the losing detection (see MergePolicy)
      - det label is ancestor of kept label    -> kept is more specific -> drop det
      - kept label is ancestor of det label    -> det is more specific -> supersede kept
      - unrelated labels                       -> keep both (different objects)
    """
    ordered_results = list(reversed(list(results_per_model))) if policy is MergePolicy.LAST_WINS else results_per_model
    matching = [det for dets in ordered_results for det in dets]

    deduped: list = []
    for det in matching:
        det_label = det.bbox.label_uuid
        dominated = False
        to_supersede: list[int] = []
        for i, kept in enumerate(deduped):
            if iou_xywh(det.bbox, kept.bbox) < overlap_iou_threshold:
                continue
            kept_label = kept.bbox.label_uuid
            if det_label == kept_label:
                dominated = True
                break
            elif det_label and det_label in label_ancestry.get(kept_label, ()):
                # det's label is an ancestor of kept's -> kept is more specific -> drop det
                dominated = True
                break
            elif kept_label and kept_label in label_ancestry.get(det_label, ()):
                # kept's label is an ancestor of det's -> det is more specific -> supersede kept
                to_supersede.append(i)
            # else: unrelated labels — keep both
        if not dominated:
            for i in reversed(to_supersede):
                deduped.pop(i)
            deduped.append(det)
    return deduped
