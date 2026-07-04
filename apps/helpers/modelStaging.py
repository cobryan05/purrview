"""Shared staging-path convention for raw uploaded model files handed off between
services on the same container/filesystem (e.g. inference_server staging a raw
upload for tasks_server's ConvertModelTask to pick up and convert to ONNX).

Deliberately torch-free / no heavy imports — this module is imported by both
inference_server.manager and tasks_server task modules.
"""

from pathlib import Path

from apps import CACHE_DIR
from apps.helpers.metadataUtils import validate_bare_model_name

# Subdirectory (under CACHE_DIR) where raw uploaded model files are staged pending
# an asynchronous conversion task (e.g. pytorch -> onnx) run by the tasks_server.
STAGING_SUBDIR = "model_conversion_uploads"


def staging_dir() -> Path:
    """Path to the shared staging directory (under CACHE_DIR)."""
    return CACHE_DIR / STAGING_SUBDIR


def staged_model_path(staging_id: str) -> Path:
    """Resolve a staging id (as returned by inference_server's stage_source_model) to
    its file path.

    Args:
        staging_id (str): Bare staging id — must not contain path separators or '..'.

    Returns:
        Path: The path to the staged file (may not exist).
    """
    validate_bare_model_name(staging_id)
    return staging_dir() / f"{staging_id}.bin"
