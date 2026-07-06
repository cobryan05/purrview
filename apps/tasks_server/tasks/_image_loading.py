"""Shared image-byte loader for tasks that run inference over saved images.

`filename` as stored/returned by the DB is FILES_ROOT-relative (or, in some
cases, an absolute path *under* FILES_ROOT) — it is never valid to open() it
directly against the task-server process's cwd. Resolve it the same way
db_server/manager.py does (get_safe_path handles both forms).

The task runs where FILES_ROOT is mounted, so a local read avoids a base64 HTTP
round-trip per image; the HTTP path stays as a fallback for image sources other
than "filesystem". Kept in one place so the path-safety logic can't drift
between ApplyFilterChainTask and PreviewModelRelabelTask.
"""

import asyncio
import base64
from pathlib import Path

from apps.helpers.fileUtils import get_safe_path


async def load_image_bytes(filename: str, image_source: str, images_api, files_root) -> bytes:
    if image_source == "filesystem":
        safe_path = get_safe_path(files_root, Path(filename))
        if safe_path is None:
            raise FileNotFoundError(f"Refusing to read path outside FILES_ROOT: {filename!r}")
        return await asyncio.to_thread(Path(safe_path).read_bytes)
    resp = await asyncio.to_thread(images_api.get_image_file, filename, True)
    if not resp.image_base64:
        raise ValueError(f"No image data returned for {filename}")
    return base64.b64decode(resp.image_base64)
