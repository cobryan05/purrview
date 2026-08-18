"""Where an action that writes images is allowed to write.

Shared by `save_snapshot` and `save_timelapse`, which now both take their directory from
the same picker and so must agree on what a path *means*. They did not always: the
snapshot action predates the picker and stored an absolute container path, while the
picker deals only in paths rooted at `FILES_ROOT` — the one path space it can list and
the one `/api/images/mkdir` can create in. Two conventions behind one Browse button is a
trap, so this module holds the single rule both follow.
"""

import os
from pathlib import Path
from typing import Optional

from apps import FILES_ROOT


def static_prefix(template: str) -> str:
    """The leading path segments of a template that contain no placeholder.

    `timelapse/{source}/{date}` → `timelapse`. `timelapse/cams` → `timelapse/cams`.
    `{source}/frames` → `""`.

    This exists for the retention sweep, which has to know which directories an action
    may have written into *without* knowing what the tokens resolved to. Expanding the
    template against the current source list would look more precise and be worse: a
    camera renamed last month leaves a folder the expansion no longer names, so its
    frames would never be swept again. The prefix is stable under renames, and the
    sweep's own shape rails — a `YYYY-MM-DD` ancestor, a frame-shaped filename, no JSON
    sidecar — are what decide whether any particular file may be deleted.

    An empty result means the template starts with a placeholder, and therefore names no
    directory the sweep can bound itself to. Callers refuse that rather than falling back
    to the dataset root: sweeping all of `FILES_ROOT` is not a thing to do by accident.
    """
    kept = []
    for segment in (template or "").strip().strip("/").split("/"):
        if "{" in segment or "}" in segment:
            break
        if segment:
            kept.append(segment)
    return "/".join(kept)


def has_placeholder(template: Optional[str]) -> bool:
    return "{" in (template or "")


def resolve_output_dir(output_dir: str, default: str) -> Path:
    """Turn a configured directory into an absolute path inside `FILES_ROOT`.

    Accepts all three spellings that reach it, because all three occur in practice:

    * ``snapshots`` — a hand-typed relative path.
    * ``/snapshots`` — what the directory picker returns. Its paths are rooted at
      `FILES_ROOT`, so the leading slash means "top of the dataset root", not "top of the
      filesystem". Treating it as absolute would write images into the container's own
      root.
    * ``/app/image_datasets/watch_snapshots`` — an absolute path already inside
      `FILES_ROOT`, honoured as-is so a value saved before the picker existed keeps
      writing exactly where it always did.

    Anything that lands outside `FILES_ROOT` — an absolute path elsewhere, or enough
    ``..`` to climb out — raises. The picker cannot reach outside it, nothing else in the
    app writes outside it, and these are long-running processes writing an unbounded
    number of files: an arbitrary path is the wrong thing to point one at.
    """
    root = FILES_ROOT.resolve()
    raw = (str(output_dir).strip() or default).rstrip("/")
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            candidate.relative_to(root)
        except ValueError:
            # Not under the root, so the leading slash was the picker's, not the
            # filesystem's. Re-rooted rather than rejected: that spelling is what the UI
            # produces, and rejecting it would make the Browse button unusable.
            candidate = root / candidate.relative_to("/")
    else:
        candidate = root / candidate

    resolved = Path(os.path.normpath(str(candidate)))
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"output_dir {output_dir!r} resolves outside {root}")
    return resolved
