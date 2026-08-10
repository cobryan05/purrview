"""Tests for the MediaMTX playback client — the retroactive clip cutter.

The buffer is RAM-backed and short, so the interesting cases are all about honesty:
a window that has partly aged out must not come back looking like a complete clip,
and an unknown path must read as "nothing buffered yet" rather than an error.
"""

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from apps.helpers.recordings.playbackClient import (
    BufferedRange,
    PlaybackClient,
    PlaybackError,
    _parse_rfc3339,
)


@pytest.fixture
def patched_httpx(monkeypatch):
    """Route every httpx.AsyncClient created by the module through a MockTransport."""
    routes: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return routes["handler"](request)

    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(handler))
        return real_client(*args, **kwargs)

    monkeypatch.setattr("apps.helpers.recordings.playbackClient.httpx.AsyncClient", factory)
    return routes


class TestParseRfc3339:
    def test_parses_zulu(self):
        assert _parse_rfc3339("2026-08-10T12:00:00Z") == datetime(2026, 8, 10, 12, tzinfo=timezone.utc)

    def test_parses_offset(self):
        parsed = _parse_rfc3339("2026-08-10T12:00:00+02:00")
        assert parsed.utcoffset() == timedelta(hours=2)

    def test_truncates_go_style_nanoseconds(self):
        # Go emits more fractional digits than datetime.fromisoformat accepts.
        parsed = _parse_rfc3339("2026-08-10T12:00:00.123456789Z")
        assert parsed.microsecond == 123456

    def test_naive_timestamps_are_treated_as_utc(self):
        assert _parse_rfc3339("2026-08-10T12:00:00").tzinfo == timezone.utc


class TestBufferedRange:
    def test_end_is_start_plus_duration(self):
        start = datetime(2026, 8, 10, 12, tzinfo=timezone.utc)
        assert BufferedRange(start, 30).end == start + timedelta(seconds=30)

    def test_covers_only_a_fully_contained_window(self):
        start = datetime(2026, 8, 10, 12, tzinfo=timezone.utc)
        rng = BufferedRange(start, 60)
        assert rng.covers(start + timedelta(seconds=10), start + timedelta(seconds=50))
        assert not rng.covers(start - timedelta(seconds=1), start + timedelta(seconds=50))


class TestListRanges:
    async def test_parses_ranges_sorted_by_start(self, patched_httpx):
        patched_httpx["handler"] = lambda req: httpx.Response(
            200,
            json=[
                {"start": "2026-08-10T12:00:30Z", "duration": 15},
                {"start": "2026-08-10T12:00:00Z", "duration": 30},
            ],
        )
        ranges = await PlaybackClient().list_ranges("relay_master_abc")
        assert [r.duration_s for r in ranges] == [30, 15]

    async def test_unknown_path_is_empty_not_an_error(self, patched_httpx):
        # A camera armed a second ago has no segments yet; that is not a failure.
        patched_httpx["handler"] = lambda req: httpx.Response(404)
        assert await PlaybackClient().list_ranges("relay_master_new") == []

    async def test_server_error_raises(self, patched_httpx):
        patched_httpx["handler"] = lambda req: httpx.Response(500, text="boom")
        with pytest.raises(PlaybackError):
            await PlaybackClient().list_ranges("relay_master_abc")

    async def test_unparseable_entry_is_skipped_not_fatal(self, patched_httpx):
        patched_httpx["handler"] = lambda req: httpx.Response(
            200,
            json=[{"start": "not-a-time", "duration": 5}, {"start": "2026-08-10T12:00:00Z", "duration": 30}],
        )
        ranges = await PlaybackClient().list_ranges("relay_master_abc")
        assert len(ranges) == 1


class TestCoverageSeconds:
    async def test_reports_depth_of_the_buffer(self, patched_httpx):
        patched_httpx["handler"] = lambda req: httpx.Response(
            200, json=[{"start": "2026-08-10T12:00:00Z", "duration": 45}]
        )
        assert await PlaybackClient().coverage_seconds("p") == 45

    async def test_empty_buffer_reports_zero(self, patched_httpx):
        patched_httpx["handler"] = lambda req: httpx.Response(200, json=[])
        assert await PlaybackClient().coverage_seconds("p") == 0.0


class TestWriteClip:
    async def test_writes_bytes_and_returns_size(self, patched_httpx, tmp_path):
        patched_httpx["handler"] = lambda req: httpx.Response(200, content=b"mp4-bytes")
        dest = tmp_path / "clip.mp4"
        size = await PlaybackClient().write_clip("p", datetime.now(timezone.utc), 10, dest)
        assert size == len(b"mp4-bytes")
        assert dest.read_bytes() == b"mp4-bytes"

    async def test_failed_extraction_leaves_no_file_behind(self, patched_httpx, tmp_path):
        # A truncated .part must never be left looking like a complete recording.
        patched_httpx["handler"] = lambda req: httpx.Response(500, text="gone")
        dest = tmp_path / "clip.mp4"
        with pytest.raises(PlaybackError):
            await PlaybackClient().write_clip("p", datetime.now(timezone.utc), 10, dest)
        assert not dest.exists()
        assert list(tmp_path.iterdir()) == []

    async def test_empty_response_is_an_error_not_a_zero_byte_clip(self, patched_httpx, tmp_path):
        patched_httpx["handler"] = lambda req: httpx.Response(200, content=b"")
        dest = tmp_path / "clip.mp4"
        with pytest.raises(PlaybackError):
            await PlaybackClient().write_clip("p", datetime.now(timezone.utc), 10, dest)
        assert not dest.exists()

    async def test_sends_utc_rfc3339_start(self, patched_httpx, tmp_path):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(dict(request.url.params))
            return httpx.Response(200, content=b"x")

        patched_httpx["handler"] = handler
        start = datetime(2026, 8, 10, 12, tzinfo=timezone.utc)
        await PlaybackClient().write_clip("relay_master_abc", start, 12.5, tmp_path / "c.mp4")
        assert seen["path"] == "relay_master_abc"
        assert seen["start"] == "2026-08-10T12:00:00Z"
        assert seen["duration"] == "12.500"
        assert seen["format"] == "mp4"
