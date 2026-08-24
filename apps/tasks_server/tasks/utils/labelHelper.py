"""Back-compat re-export. These live in `apps.helpers.labelHelper` now.

They moved because the training collector (`apps/helpers/training/`) needs them,
and `helpers/` importing from `tasks_server/` inverts the layering.
"""

from apps.helpers.labelHelper import build_label_ancestry_map, build_label_metadata_map

__all__ = ["build_label_ancestry_map", "build_label_metadata_map"]
