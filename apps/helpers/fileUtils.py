"""File utilities"""

from pathlib import Path
from typing import Any, List, Optional, Sequence


def get_safe_path(root: Path, rel_path: Path) -> Optional[Path]:
    """
    Resolve a path safely under the given root directory.

    If rel_path is absolute and under root, use it.
    If rel_path is absolute but outside root, treat it as relative.
    If rel_path is already relative, resolve it under root.

    Args:
        root (Path): Root directory.
        rel_path (str): Relative or absolute path.

    Returns:
        Optional[Path]: Safe absolute path if within root, otherwise None.
    """
    try:
        root = root.resolve(strict=True)
        raw_path = Path(rel_path)

        # If absolute but not under root, treat it as relative
        if raw_path.is_absolute():
            resolved = raw_path.resolve(strict=False)
            if root in resolved.parents or resolved == root:
                return resolved
            raw_path = Path("." + raw_path.as_posix())  # treat as relative

        target = (root / raw_path).resolve(strict=False)

        return target if root in target.parents or target == root else None
    except Exception:
        return None


def filter_stubs_by_directory(root: Path, directory: str, stubs: Sequence[Any]) -> List[Any]:
    """
    Filter image stubs (objects with a `.filename` attribute) down to those whose
    filename is under `directory`.

    The DB stores filenames as absolute paths, so `directory` (which may be
    relative or absolute) is first resolved to an absolute path under `root` via
    `get_safe_path` before prefix-matching. Returns an empty list if `directory`
    resolves outside `root`.
    """
    abs_dir = get_safe_path(root, Path(directory))
    if abs_dir is None:
        return []
    prefix = str(abs_dir) + "/"
    return [s for s in stubs if s.filename.startswith(prefix) or s.filename == str(abs_dir)]
