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
    parser.addoption(
        "--run-quarantined",
        action="store_true",
        default=False,
        help="Also run tests marked @pytest.mark.quarantine (known flaky; see FLAKY_TESTS.md)",
    )


# `@pytest.mark.quarantine("<why>")` takes a known-flaky test out of the default run.
#
# The point is not to hide failures, it is to stop paying for them twice: a flake that
# fails one run in three is rediscovered from scratch every time, and re-verifying it
# costs far more than the coverage it still provides. Every quarantined test has an entry
# in FLAKY_TESTS.md recording the verbatim failure, the observed frequency and the
# mechanism (or UNDIAGNOSED); the reason string points there.
#
# Not xfail: an xfail that starts passing tells no one. Run them deliberately with
# `--run-quarantined`, which is how a fix gets confirmed or a rate gets sampled.
def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-quarantined"):
        return
    for item in items:
        marker = item.get_closest_marker("quarantine")
        if marker is None:
            continue
        why = (marker.args[0] if marker.args else "known flaky") + " [FLAKY_TESTS.md]"
        item.add_marker(pytest.mark.skip(reason=f"quarantined: {why}"))


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
