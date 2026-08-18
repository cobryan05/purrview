"""Unit tests for the RAM-buffer sizing arithmetic.

These matter because a full tmpfs makes MediaMTX's recorder fail writes rather than
evict early — the budget is the only thing standing between "arm one camera too many"
and silently broken recordings.
"""

import pytest

from apps.helpers.streams.bufferBudget import (
    BufferUsage,
    bytes_for,
    measure_mount_bytes,
    parse_size,
)


class TestParseSize:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("512m", 512 * 1024**2),
            ("512M", 512 * 1024**2),
            ("512mb", 512 * 1024**2),
            ("2gb", 2 * 1024**3),
            ("2G", 2 * 1024**3),
            ("1024k", 1024 * 1024),
            ("4096", 4096),
            ("  1g  ", 1024**3),
        ],
    )
    def test_accepts_docker_size_forms(self, text, expected):
        assert parse_size(text) == expected

    @pytest.mark.parametrize("text", ["", "big", "512mib", "-1m", "m512", None])
    def test_rejects_garbage_rather_than_guessing(self, text):
        # A misparsed cap is worse than no cap: it would silently over-commit.
        with pytest.raises(ValueError):
            parse_size(text)


class TestBytesFor:
    def test_bitrate_times_retention(self):
        # 4 Mbps for 60 s = 30 MB, which is the sizing rule the setup prompt quotes.
        assert bytes_for(4_000_000, 60) == 30_000_000

    def test_negative_retention_is_clamped_not_negative(self):
        assert bytes_for(4_000_000, -10) == 0


class TestBufferUsage:
    def test_reports_free_and_fraction(self):
        usage = BufferUsage(cap_bytes=1000, committed_bytes=250)
        assert usage.free_bytes == 750
        assert usage.committed_fraction == 0.25

    def test_over_commit_does_not_report_negative_free(self):
        usage = BufferUsage(cap_bytes=1000, committed_bytes=1500)
        assert usage.free_bytes == 0

    def test_zero_cap_does_not_divide_by_zero(self):
        assert BufferUsage(cap_bytes=0, committed_bytes=10).committed_fraction == 0.0


class TestMeasureMount:
    def test_missing_path_reports_unknown_rather_than_zero(self):
        # An unmounted /buffer must fall back to the declared cap, not look like a
        # zero-byte filesystem that refuses every source.
        assert measure_mount_bytes("/definitely/not/a/mount/point") == (None, None)

    def test_existing_path_reports_totals(self, tmp_path):
        total, used = measure_mount_bytes(str(tmp_path))
        assert total is not None and total > 0
        assert used is not None and 0 <= used <= total
