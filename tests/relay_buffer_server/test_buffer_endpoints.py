"""Endpoint tests for the relay's ensure/release/ranges/clip/buffer API.

These are the surface a StreamWatcher talks to, so the contract that matters is what
the responses *tell the caller*: whether a source actually has a rewind buffer, and
whether a clip came back whole.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from apps.relay_buffer_server.manager import EnsureResult, master_path_name


class TestEnsureEndpoint:
    async def test_returns_master_path_and_lease(self, http_client, manager):
        resp = await http_client.post(
            "/api/streams/ensure",
            data={"source_url": "rtsp://cam/stream", "holder": "watcher-1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["path"] == master_path_name("rtsp://cam/stream")
        assert data["leaseId"]
        assert data["recording"] is True

    async def test_reports_when_a_source_has_no_buffer(self, http_client, manager):
        resp = await http_client.post(
            "/api/streams/ensure",
            data={"source_url": "rtsp://cam/stream", "holder": "w", "record": "false"},
        )
        data = resp.json()
        assert data["success"] is True
        # The watcher shows this to the user instead of quietly cutting empty clips.
        assert data["recording"] is False
        assert data["retentionS"] == 0.0

    async def test_over_commit_is_reported_as_an_error(self, http_client, manager):
        manager._buffer_cap_bytes = 1_000
        resp = await http_client.post(
            "/api/streams/ensure",
            data={"source_url": "rtsp://cam/stream", "holder": "w", "retention_s": "60"},
        )
        data = resp.json()
        assert data["success"] is False
        assert "buffer" in data["error"].lower()

    async def test_two_holders_share_one_path(self, http_client, manager):
        first = (await http_client.post("/api/streams/ensure", data={"source_url": "rtsp://c/s", "holder": "a"})).json()
        second = (
            await http_client.post("/api/streams/ensure", data={"source_url": "rtsp://c/s", "holder": "b"})
        ).json()
        assert first["path"] == second["path"]
        assert first["leaseId"] != second["leaseId"]


class TestReleaseEndpoint:
    async def test_releases_a_lease(self, http_client, manager):
        ensured = (
            await http_client.post("/api/streams/ensure", data={"source_url": "rtsp://c/s", "holder": "a"})
        ).json()
        resp = await http_client.post("/api/streams/release", data={"lease_id": ensured["leaseId"]})
        data = resp.json()
        assert data["success"] is True
        assert data["released"] is True

    async def test_unknown_lease_reports_not_released(self, http_client, manager):
        resp = await http_client.post("/api/streams/release", data={"lease_id": "nope"})
        data = resp.json()
        assert data["success"] is True
        assert data["released"] is False


class TestRangesEndpoint:
    async def test_returns_buffered_ranges(self, http_client, manager, monkeypatch):
        monkeypatch.setattr(
            manager,
            "get_ranges",
            AsyncMock(return_value=[{"start": "2026-08-10T12:00:00+00:00", "durationS": 45.0}]),
        )
        resp = await http_client.get("/api/streams/relay_master_abc/ranges")
        data = resp.json()
        assert data["success"] is True
        assert data["results"][0]["durationS"] == 45.0

    async def test_playback_failure_is_reported(self, http_client, manager, monkeypatch):
        monkeypatch.setattr(manager, "get_ranges", AsyncMock(side_effect=RuntimeError("playback down")))
        resp = await http_client.get("/api/streams/relay_master_abc/ranges")
        data = resp.json()
        assert data["success"] is False
        assert "playback down" in data["error"]


class TestClipEndpoint:
    async def test_extracts_a_clip(self, http_client, manager, monkeypatch, tmp_path):
        monkeypatch.setattr(
            manager,
            "extract_clip",
            AsyncMock(return_value={"path": str(tmp_path / "c.mp4"), "sizeBytes": 10, "partialPreroll": False}),
        )
        resp = await http_client.post(
            "/api/streams/clip",
            data={
                "path": "relay_master_abc",
                "start": "2026-08-10T12:00:00Z",
                "duration_s": "20",
                "dest": str(tmp_path / "c.mp4"),
            },
        )
        data = resp.json()
        assert data["success"] is True
        assert data["partialPreroll"] is False

    async def test_start_is_parsed_as_utc_when_naive(self, http_client, manager, monkeypatch, tmp_path):
        captured = {}

        async def fake_extract(path, start, duration_s, dest):
            captured["start"] = start
            return {"path": str(dest), "sizeBytes": 1, "partialPreroll": False}

        monkeypatch.setattr(manager, "extract_clip", fake_extract)
        await http_client.post(
            "/api/streams/clip",
            data={
                "path": "p",
                "start": "2026-08-10T12:00:00",
                "duration_s": "5",
                "dest": str(tmp_path / "c.mp4"),
            },
        )
        assert captured["start"] == datetime(2026, 8, 10, 12, tzinfo=timezone.utc)

    async def test_aged_out_window_is_reported_as_an_error(self, http_client, manager, monkeypatch, tmp_path):
        monkeypatch.setattr(manager, "extract_clip", AsyncMock(side_effect=RuntimeError("aged out of the buffer")))
        resp = await http_client.post(
            "/api/streams/clip",
            data={"path": "p", "start": "2026-08-10T12:00:00Z", "duration_s": "5", "dest": str(tmp_path / "c.mp4")},
        )
        data = resp.json()
        assert data["success"] is False
        assert "aged out" in data["error"]


class TestBufferUsageEndpoint:
    async def test_reports_cap_and_commitment(self, http_client, manager):
        await http_client.post("/api/streams/ensure", data={"source_url": "rtsp://c/s", "holder": "a"})
        resp = await http_client.get("/api/streams/buffer")
        data = resp.json()
        assert data["success"] is True
        assert data["capBytes"] > 0
        assert data["committedBytes"] > 0
        assert 0 <= data["committedFraction"] <= 1

    async def test_status_includes_buffer(self, http_client, manager):
        resp = await http_client.get("/status")
        assert resp.json()["buffer"]["capBytes"] > 0
