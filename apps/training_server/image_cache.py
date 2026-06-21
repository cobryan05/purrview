"""Persistent, UUID-keyed LRU image cache for the training server.

Images are immutable per UUID: "do I have uuid X?" is a complete and correct
cache check.  This enables true incremental image transfer — only UUIDs absent
from the cache need to be sent.

Layout: flat files at ``<cache_dir>/<uuid>`` (no extension).
Eviction: oldest-mtime files are removed when total size exceeds the limit.
Pinning: UUIDs required by active (awaiting_data / queued / running) jobs are
never evicted, even if the cache is over-limit.
"""

from __future__ import annotations

import io
import logging
import os
import tarfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = "/training-cache"
_DEFAULT_MAX_MB = 5000  # 5 GB
_MAX_IMAGE_MB = 50       # reject members larger than this


class ImageCache:
    """UUID-keyed LRU image cache backed by a directory on a named volume."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_mb: Optional[int] = None,
    ):
        self._dir = Path(
            cache_dir
            or os.environ.get("TRAINING_CACHE_DIR", _DEFAULT_CACHE_DIR)
        )
        self._max_bytes = (
            (max_mb or int(os.environ.get("TRAINING_CACHE_MAX_MB", _DEFAULT_MAX_MB)))
            * 1024
            * 1024
        )
        # UUIDs that belong to live jobs and must not be evicted.
        self._pinned: set[str] = set()

    def ensure_dir(self) -> None:
        """Create the cache directory if it does not exist (called at server start)."""
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Pin/unpin ─────────────────────────────────────────────────────────────

    def pin(self, uuids: set[str]) -> None:
        """Mark UUIDs as in-use by an active job (not evictable)."""
        self._pinned.update(uuids)

    def unpin(self, uuids: set[str]) -> None:
        """Release eviction protection for these UUIDs."""
        self._pinned -= uuids

    # ── Read interface ────────────────────────────────────────────────────────

    def path(self, uuid: str) -> Path:
        """Return the filesystem path for a UUID (may not exist)."""
        return self._dir / uuid

    def has(self, uuid: str) -> bool:
        """Return True if the image is cached; touch mtime for LRU tracking."""
        if not self._dir.exists():
            return False
        p = self._dir / uuid
        if p.exists():
            p.touch()
            return True
        return False

    def missing(self, uuids: list[str]) -> list[str]:
        """Return the subset of *uuids* not present in the cache."""
        return [u for u in uuids if not self.has(u)]

    # ── Write interface ───────────────────────────────────────────────────────

    def ingest_tar(
        self,
        stream: bytes,
        required_uuids: Optional[set[str]] = None,
    ) -> dict:
        """Extract tar members into the cache.

        Each tar member must be named ``<uuid>`` or ``<uuid>.<ext>``.
        Writes atomically: data → temp file → fsync → rename.
        Unknown or oversized members are rejected with a warning.

        Args:
            stream:         Raw tar bytes.
            required_uuids: When provided, reject members whose UUID is not in
                            this set (prevents arbitrary writes).

        Returns:
            ``{"received": n, "cached_total": m}``
        """
        self.ensure_dir()
        received = 0
        max_bytes = _MAX_IMAGE_MB * 1024 * 1024

        with tarfile.open(fileobj=io.BytesIO(stream), mode="r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue

                # Derive UUID: strip directory components and extension.
                name = Path(member.name).name
                uuid = name.split(".")[0]
                if not uuid:
                    logger.warning("Tar member has no UUID-like name: %s", member.name)
                    continue

                if required_uuids is not None and uuid not in required_uuids:
                    logger.warning("Tar member %s not in required uuids, skipping", uuid)
                    continue

                if member.size > max_bytes:
                    logger.warning(
                        "Tar member %s too large (%d bytes > %d), skipping",
                        name,
                        member.size,
                        max_bytes,
                    )
                    continue

                dest = self._dir / uuid
                if dest.exists():
                    # Already cached — count it but don't re-write.
                    dest.touch()
                    received += 1
                    continue

                tmp = self._dir / f".tmp_{uuid}"
                try:
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    data = fobj.read()
                    tmp.write_bytes(data)
                    # fsync before rename so a crash never leaves a partial file.
                    with open(tmp, "rb") as fd:
                        os.fsync(fd.fileno())
                    tmp.rename(dest)
                    received += 1
                except Exception:
                    logger.exception("Failed to ingest tar member %s", name)
                    tmp.unlink(missing_ok=True)

        self._maybe_evict()

        cached_total = sum(
            1 for p in self._dir.iterdir() if p.is_file() and not p.name.startswith(".tmp_")
        ) if self._dir.exists() else 0
        return {"received": received, "cached_total": cached_total}

    # ── Eviction ──────────────────────────────────────────────────────────────

    def _maybe_evict(self) -> None:
        """Evict the oldest files until total size is within the limit.

        Pinned UUIDs are never evicted regardless of size pressure.
        """
        if not self._dir.exists():
            return
        files = []
        total_bytes = 0
        for p in self._dir.iterdir():
            if not p.is_file() or p.name.startswith(".tmp_"):
                continue
            try:
                st = p.stat()
                files.append((st.st_mtime, p, st.st_size))
                total_bytes += st.st_size
            except FileNotFoundError:
                pass

        if total_bytes <= self._max_bytes:
            return

        files.sort()  # oldest first
        for _mtime, p, size in files:
            if total_bytes <= self._max_bytes:
                break
            uuid = p.name.split(".")[0]
            if uuid in self._pinned:
                continue
            try:
                p.unlink()
                total_bytes -= size
                logger.info("Evicted cached image %s (%d bytes)", p.name, size)
            except FileNotFoundError:
                pass
