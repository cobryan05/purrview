"""Persistent, UUID-keyed LRU image cache for the training server.

Images are immutable per UUID: "do I have uuid X?" is a complete and correct
cache check.  This enables true incremental image transfer — only UUIDs absent
from the cache need to be sent.

Layout: flat files at ``<cache_dir>/<uuid>`` (no extension).
Eviction: both a size bound (oldest-mtime files removed once total size
exceeds the limit) and an age bound (files older than the TTL are removed
regardless of size) apply. Pinning: UUIDs required by active
(awaiting_data / queued / running) jobs are never evicted by either bound.
"""

from __future__ import annotations

import errno
import io
import logging
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = "/training-cache"
_DEFAULT_MAX_MB = 10000  # 10 GB — matches TRAINING_CACHE_MAX_MB in docker-compose.yaml
_DEFAULT_MAX_AGE_DAYS = 30  # 0 disables the age bound
_MAX_IMAGE_MB = 50  # reject members larger than this


class ImageCache:
    """UUID-keyed LRU image cache backed by a directory on a named volume."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_mb: Optional[int] = None,
        max_age_days: Optional[int] = None,
    ):
        self._dir = Path(cache_dir or os.environ.get("TRAINING_CACHE_DIR", _DEFAULT_CACHE_DIR))
        self._max_bytes = (max_mb or int(os.environ.get("TRAINING_CACHE_MAX_MB", _DEFAULT_MAX_MB))) * 1024 * 1024
        max_age_days = (
            max_age_days
            if max_age_days is not None
            else int(os.environ.get("TRAINING_CACHE_MAX_AGE_DAYS", _DEFAULT_MAX_AGE_DAYS))
        )
        # 0 (or negative) disables the age bound — size bound still applies.
        self._max_age_seconds = max_age_days * 86400 if max_age_days > 0 else None
        # UUIDs that belong to live jobs and must not be evicted.
        self._pinned: set[str] = set()

    def ensure_dir(self) -> None:
        """Create the cache directory if it does not exist (called at server start)."""
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> str:
        """Return the cache directory path as a string."""
        return str(self._dir)

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

        cached_total = (
            sum(1 for p in self._dir.iterdir() if p.is_file() and not p.name.startswith(".tmp_"))
            if self._dir.exists()
            else 0
        )
        return {"received": received, "cached_total": cached_total}

    # ── Eviction ──────────────────────────────────────────────────────────────

    def _maybe_evict(self) -> None:
        """Run the combined age + size eviction sweep. Called after every ingest."""
        self.evict()

    def evict(self) -> None:
        """Evict files that are past the age TTL, then oldest-first until
        total size is within the limit.

        Pinned UUIDs are never evicted by either bound. Safe to call
        periodically (e.g. from a background task) as well as after ingests.
        """
        if not self._dir.exists():
            return
        files = []
        total_bytes = 0
        now = time.time()
        for p in self._dir.iterdir():
            if not p.is_file() or p.name.startswith(".tmp_"):
                continue
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            uuid = p.name.split(".")[0]
            if self._max_age_seconds is not None and uuid not in self._pinned:
                age = now - st.st_mtime
                if age > self._max_age_seconds:
                    try:
                        p.unlink()
                        logger.info("Evicted cached image %s (age %.0fs > ttl)", p.name, age)
                        continue
                    except FileNotFoundError:
                        continue
            files.append((st.st_mtime, p, st.st_size))
            total_bytes += st.st_size

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

    # ── Local (no-copy) ingest ───────────────────────────────────────────────

    def ingest_local(
        self,
        entries: Iterable[tuple[str, Path]],
        required_uuids: Optional[set[str]] = None,
    ) -> dict:
        """Populate the cache directly from local source paths, without any
        network transfer.

        For each ``(uuid, source_path)`` pair, hardlinks the source into the
        cache when possible (same filesystem — true zero-copy), or falls back
        to a plain file copy when the source lives on a different device
        (``EXDEV``). This only works when *source_path* is actually reachable
        from this process — e.g. training_server and the pusher share a
        filesystem (same container, or a shared bind mount). Callers should
        push any UUID that comes back in ``not_found`` through the ordinary
        tar-upload path instead.

        Returns ``{"received": n, "cached_total": m, "linked": x, "copied": y,
        "not_found": [uuid, ...]}``.
        """
        self.ensure_dir()
        received = 0
        linked = 0
        copied = 0
        not_found: list[str] = []

        for uuid, source_path in entries:
            if required_uuids is not None and uuid not in required_uuids:
                logger.warning("Local ingest uuid %s not in required uuids, skipping", uuid)
                continue

            dest = self._dir / uuid
            if dest.exists():
                dest.touch()
                received += 1
                continue

            source_path = Path(source_path)
            if not source_path.is_file():
                not_found.append(uuid)
                continue

            tmp = self._dir / f".tmp_{uuid}"
            try:
                try:
                    os.link(source_path, tmp)
                    linked += 1
                except OSError as e:
                    if e.errno != errno.EXDEV:
                        raise
                    with open(source_path, "rb") as fsrc, open(tmp, "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                    with open(tmp, "rb") as fd:
                        os.fsync(fd.fileno())
                    copied += 1
                tmp.rename(dest)
                received += 1
            except Exception:
                logger.exception("Failed to locally ingest %s from %s", uuid, source_path)
                tmp.unlink(missing_ok=True)
                not_found.append(uuid)

        self._maybe_evict()

        cached_total = (
            sum(1 for p in self._dir.iterdir() if p.is_file() and not p.name.startswith(".tmp_"))
            if self._dir.exists()
            else 0
        )
        return {
            "received": received,
            "cached_total": cached_total,
            "linked": linked,
            "copied": copied,
            "not_found": not_found,
        }
