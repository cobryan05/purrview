import struct
import zlib

import nest_asyncio
import pytest

nest_asyncio.apply()


def pytest_addoption(parser):
    parser.addoption(
        "--run-smoke",
        action="store_true",
        default=False,
        help="Run smoke tests against live services",
    )


def make_png(pixel: bytes = b"\xff\xff\xff") -> bytes:
    """Return a minimal valid 1x1 PNG, white unless another RGB pixel is given.

    The pixel is a parameter so a test can produce two *different* valid images
    -- "did this file get replaced?" cannot be asked with one byte-identical
    fixture.
    """

    def chunk(tag, data):
        raw = tag + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00" + pixel))
        + chunk(b"IEND", b"")
    )
