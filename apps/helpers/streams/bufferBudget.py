"""Sizing arithmetic for the in-RAM stream buffer (the tmpfs MediaMTX records into).

A full tmpfs makes MediaMTX's recorder *fail writes* rather than evict early, so the
relay has to refuse to arm a source whose committed bytes would breach the mount
instead of discovering the limit the hard way. That means two numbers:

* the **cap** — declared by ``STREAM_BUFFER_SIZE`` and, as a cross-check, whatever
  ``statvfs`` reports for the mount. They should agree; when they don't, the smaller
  one wins, because the mount is the thing that actually fails.
* the **committed bytes** — ``Σ bitrate × retention`` over armed sources. Bitrate is an
  estimate until a stream has run long enough to measure, so this is deliberately a
  budget rather than a measurement.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

# What we assume a camera costs before we have measured it. 4 Mbps is a typical
# 1080p H.264 IP camera; the plan's sizing rule (~30 MB per camera-minute) is this
# number. Sources that turn out cheaper simply leave headroom unused.
DEFAULT_ASSUMED_BITRATE_BPS = 4_000_000

# Fraction of the cap we allow to be committed. Segments are keyframe-aligned, so
# retention overshoots the requested window by up to one GOP, and fMP4 carries
# container overhead on top of the elementary stream.
COMMIT_HEADROOM = 0.85

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)b?\s*$", re.IGNORECASE)
_UNIT_MULTIPLIERS = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}


def parse_size(value: str) -> int:
    """Parse a docker-style size string ("512m", "2gb", "1024") into bytes.

    Accepts the same forms ``setup.sh`` validates, so a value that made it into
    ``.env`` parses here. Raises ValueError on anything else rather than guessing —
    a misparsed cap is worse than no cap.
    """
    match = _SIZE_RE.match(value or "")
    if not match:
        raise ValueError(f"Not a size string: {value!r} (expected e.g. '512m', '2gb')")
    number, unit = match.groups()
    return int(float(number) * _UNIT_MULTIPLIERS[unit.lower()])


def bytes_for(bitrate_bps: int, retention_s: float) -> int:
    """Bytes a stream at ``bitrate_bps`` occupies when ``retention_s`` are retained."""
    return int(bitrate_bps / 8.0 * max(0.0, retention_s))


@dataclass
class BufferUsage:
    """A snapshot of the buffer's budget, shaped for the status panel's stat grid."""

    cap_bytes: int
    committed_bytes: int
    used_bytes: Optional[int] = None  # actual on-disk usage; None when unmeasurable

    @property
    def free_bytes(self) -> int:
        return max(0, self.cap_bytes - self.committed_bytes)

    @property
    def committed_fraction(self) -> float:
        return self.committed_bytes / self.cap_bytes if self.cap_bytes else 0.0

    def to_dict(self) -> dict:
        return {
            "capBytes": self.cap_bytes,
            "committedBytes": self.committed_bytes,
            "usedBytes": self.used_bytes,
            "freeBytes": self.free_bytes,
            "committedFraction": round(self.committed_fraction, 4),
        }


def measure_mount_bytes(path: str) -> tuple[Optional[int], Optional[int]]:
    """Return ``(total_bytes, used_bytes)`` for the filesystem holding ``path``.

    Returns ``(None, None)`` when the path doesn't exist — which is the normal case in
    tests and in any deployment that hasn't mounted a /buffer tmpfs — so callers fall
    back to the declared cap rather than treating an unmounted buffer as size zero.
    """
    try:
        st = os.statvfs(path)
    except OSError:
        return None, None
    total = st.f_blocks * st.f_frsize
    free = st.f_bavail * st.f_frsize
    return total, max(0, total - free)
