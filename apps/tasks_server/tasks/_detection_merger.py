"""Back-compat re-export. The merge helper now lives in `apps.helpers.detectionMerger`.

It moved because `helpers/training/collectorSpec.py` needs `MergePolicy`, and a
`helpers/` module importing from `tasks_server/` inverts the layering. Import from
`apps.helpers.detectionMerger` in new code.
"""

from apps.helpers.detectionMerger import MergePolicy, merge_detections_by_iou_and_hierarchy

__all__ = ["MergePolicy", "merge_detections_by_iou_and_hierarchy"]
