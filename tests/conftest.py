import struct
import zlib

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-smoke", action="store_true", default=False, help="Run smoke tests against live services")


def make_png() -> bytes:
    """Return a minimal valid 1x1 white PNG."""
    def chunk(tag, data):
        raw = tag + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )
