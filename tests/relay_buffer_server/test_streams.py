import pytest

from apps.relay_buffer_server.manager import Manager


async def test_server_config(http_client):
    resp = await http_client.get("/server-config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "mediamtx_server" in data["config"]


async def test_list_streams_empty(http_client):
    resp = await http_client.get("/api/streams/list")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["results"] == []


async def test_list_streams_shows_active_streams(http_client, manager):
    manager._streams["cam1"] = Manager.StreamInfo(name="cam1")
    resp = await http_client.get("/api/streams/list")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    names = [item["name"] for item in data["results"]]
    assert "cam1" in names


async def test_create_stream_calls_relay(http_client, manager):
    resp = await http_client.post(
        "/api/streams/create",
        data={"source_url": "rtsp://cam/stream", "stream_name": "relay1", "delay": "0"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    manager.create_rtsp_relay_stream.assert_called_once_with(
        rtsp_url="rtsp://cam/stream", stream_name="relay1"
    )


async def test_create_stream_with_delay_calls_delay_stream(http_client, manager):
    resp = await http_client.post(
        "/api/streams/create",
        data={"source_url": "rtsp://cam/stream", "stream_name": "delayed1", "delay": "2.5"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    manager.create_delay_stream.assert_called_once_with(
        rtsp_url="rtsp://cam/stream", stream_name="delayed1", delay=2.5
    )


async def test_create_stream_failure_returns_error(http_client, manager):
    from unittest.mock import AsyncMock
    manager.create_rtsp_relay_stream = AsyncMock(side_effect=RuntimeError("mediamtx down"))
    resp = await http_client.post(
        "/api/streams/create",
        data={"source_url": "rtsp://cam/stream", "stream_name": "fail1", "delay": "0"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert "mediamtx down" in data["error"]


async def test_destroy_stream_calls_manager(http_client, manager):
    manager._streams["relay1"] = Manager.StreamInfo(name="relay1")
    resp = await http_client.post("/api/streams/destroy", data={"stream_name": "relay1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    manager.destroy_stream.assert_called_once_with(stream_name="relay1")


async def test_destroy_stream_failure_returns_error(http_client, manager):
    from unittest.mock import AsyncMock
    manager.destroy_stream = AsyncMock(side_effect=RuntimeError("not found"))
    resp = await http_client.post("/api/streams/destroy", data={"stream_name": "ghost"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False


async def test_list_config(http_client):
    resp = await http_client.get("/api/config/list")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "rtspAddress" in data["results"]
