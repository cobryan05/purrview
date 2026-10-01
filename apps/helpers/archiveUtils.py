"""Safe extraction of a user-supplied archive into a staging directory.

Zip and tar both let an archive name its own destinations, so an entry can
carry an absolute path, a ``..`` traversal, or a symlink pointing anywhere on
the filesystem — and tar additionally allows device nodes. Every member is
therefore resolved against the destination and rejected if it lands outside,
rather than trusted and written.

Uncompressed size is bounded as well as path: a few hundred kilobytes of zip
can expand to terabytes, and the destination here is the same volume the image
library lives on.

Tar is supported alongside zip because ``datasetExport`` writes tar, so an
export from this app is an archive a user will reasonably try to import back.
"""

from __future__ import annotations

import logging
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

from apps.helpers.logUtils import LOG_LEVEL

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

#: Extensions accepted by the archive-upload path. Checked against the *whole*
#: trailing suffix chain so ``.tar.gz`` is recognised, not read as ``.gz``.
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")

#: Ceiling on total extracted bytes. Well above a realistic annotated dataset
#: while still bounding what one upload can do to the volume.
MAX_EXTRACTED_BYTES = 64 * 1024 * 1024 * 1024  # 64 GiB

#: Ceiling on member count, which bounds the inode cost independently of size.
MAX_MEMBERS = 500_000


class ArchiveError(ValueError):
    """An archive that cannot be safely extracted, with a user-facing reason."""


def is_archive_name(filename: str) -> bool:
    lowered = filename.lower()
    return any(lowered.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _safe_destination(dest_root: Path, member_name: str) -> Path:
    """Resolve ``member_name`` under ``dest_root``, or raise.

    ``dest_root`` is resolved once by the caller; resolving the candidate the
    same way is what catches a traversal that only becomes one after symlink
    expansion of an earlier member.
    """
    name = member_name.replace("\\", "/").lstrip("/")
    if not name or name in {".", ".."}:
        raise ArchiveError(f"Archive contains an unusable entry name: {member_name!r}")
    candidate = (dest_root / name).resolve()
    if candidate != dest_root and dest_root not in candidate.parents:
        raise ArchiveError(f"Archive entry escapes the destination: {member_name!r}")
    return candidate


def _check_budget(total: int, count: int, added: int) -> tuple[int, int]:
    total += added
    count += 1
    if total > MAX_EXTRACTED_BYTES:
        raise ArchiveError("Archive expands to more than the extraction size limit")
    if count > MAX_MEMBERS:
        raise ArchiveError(f"Archive contains more than {MAX_MEMBERS} entries")
    return total, count


def _extract_zip(path: Path, dest_root: Path) -> int:
    written = 0
    total = 0
    count = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                _safe_destination(dest_root, info.filename).mkdir(parents=True, exist_ok=True)
                continue
            # The declared size is what the budget is checked against before a
            # single byte is written; a member that lies about it is caught by
            # ZipFile's own CRC check on close.
            total, count = _check_budget(total, count, info.file_size)
            target = _safe_destination(dest_root, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            written += 1
    return written


def _tar_members(tf: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
    for member in tf:
        if member.isdir():
            yield member
        elif member.isfile():
            yield member
        # Links and device nodes are dropped rather than rejected: a tar made
        # on a machine with symlinked image directories is otherwise
        # unimportable, and a dataset never needs the link itself.
        elif member.issym() or member.islnk():
            logger.info("archiveUtils: skipping link member %r", member.name)
        else:
            logger.info("archiveUtils: skipping special member %r", member.name)


def _extract_tar(path: Path, dest_root: Path) -> int:
    written = 0
    total = 0
    count = 0
    with tarfile.open(path, "r:*") as tf:
        for member in _tar_members(tf):
            target = _safe_destination(dest_root, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total, count = _check_budget(total, count, member.size)
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, target.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            written += 1
    return written


def extract_archive(archive_path: Path, dest_dir: Path) -> int:
    """Extract ``archive_path`` into ``dest_dir``; return the file count.

    ``dest_dir`` is created if absent. Raises ``ArchiveError`` for an
    unreadable archive, an entry that escapes the destination, or one that
    breaches the size/count budget — in every case leaving whatever was already
    written for the caller to clean up, since a partial tree on the staging
    path is not reachable by anything but the caller anyway.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()

    try:
        if zipfile.is_zipfile(archive_path):
            return _extract_zip(archive_path, dest_root)
        if tarfile.is_tarfile(archive_path):
            return _extract_tar(archive_path, dest_root)
    except ArchiveError:
        raise
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ArchiveError(f"Archive could not be extracted: {exc}") from exc

    raise ArchiveError("File is not a readable .zip or .tar archive")
