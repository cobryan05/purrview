"""The Service Status panel's detail view for the relay buffer server.

"streams: 3" doesn't say which cameras are up, who is watching them, or which
are recording — which is the whole reason to look at this row.
"""

from unittest.mock import MagicMock

from apps.relay_buffer_server.manager import Manager


def _stream(manager, name: str, **kwargs) -> None:
    manager._streams[name] = Manager.StreamInfo(name=name, **kwargs)


def _mtx(**kwargs):
    """A stand-in for mediamtx_client's Path — get_streams() only calls model_dump()."""
    path = MagicMock()
    path.model_dump.return_value = {"ready": False, "tracks": [], "bytes_received": 0, **kwargs}
    return path


class TestRelayStatusDetails:
    def test_no_streams_reports_an_empty_section(self, manager):
        status = manager.get_status()

        assert status["streams"]["active"] == 0
        assert status["summary"] == "streams: 0"
        streams_section = status["details"][0]
        assert streams_section["rows"] == []
        assert streams_section["empty"] == "No streams"

    def test_each_stream_is_named_with_its_source(self, manager):
        _stream(manager, "front_door", active=True, source_url="rtsp://cam/front")

        row = manager.get_status()["details"][0]["rows"][0]

        assert row["label"] == "front_door"
        assert "rtsp://cam/front" in row["value"]

    def test_recording_streams_report_their_retention(self, manager):
        _stream(manager, "front_door", active=True, recording=True, retention_s=120)

        status = manager.get_status()

        assert status["streams"]["recording"] == 1
        assert "recording (2m 00s retention)" in status["details"][0]["rows"][0]["value"]
        assert "1 recording" in status["summary"]

    def test_mediamtx_readiness_wins_over_our_active_flag(self, manager):
        """``active`` is our bookkeeping; ``ready`` is whether MediaMTX is
        actually publishing. A reconnecting source disagrees, and that
        disagreement is what you open this view to see."""
        _stream(manager, "publishing", active=True, mtx_path=_mtx(ready=True))
        _stream(manager, "reconnecting", active=True, mtx_path=_mtx(ready=False))

        rows = {r["label"]: r["value"] for r in manager.get_status()["details"][0]["rows"]}

        assert rows["publishing"].startswith("ready")
        assert rows["reconnecting"].startswith("active")

    def test_reader_counts_come_from_leases(self, manager, monkeypatch):
        _stream(manager, "front_door", active=True)
        monkeypatch.setattr(manager, "reader_counts", lambda: {"front_door": 2})

        status = manager.get_status()

        assert status["streams"]["readers"] == 2
        assert "2 readers" in status["details"][0]["rows"][0]["value"]
        assert "2 readers" in status["summary"]

    def test_track_and_byte_info_is_surfaced(self, manager):
        _stream(manager, "front_door", active=True, mtx_path=_mtx(ready=True, tracks=["H264", "MPEG4-audio"]))

        value = manager.get_status()["details"][0]["rows"][0]["value"]

        assert "H264/MPEG4-audio" in value

    def test_buffer_section_reports_the_budget(self, manager):
        buffer_section = manager.get_status()["details"][1]

        assert buffer_section["title"] == "Buffer"
        assert [r["label"] for r in buffer_section["rows"]] == ["Committed", "Used on disk", "Cap"]

    async def test_status_endpoint_serves_the_details(self, http_client, manager):
        _stream(manager, "front_door", active=True)

        resp = await http_client.get("/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "relay_buffer_server"
        # The counters the collapsed panel reads must survive the move to details.
        assert data["streams"]["active"] == 1
        assert "buffer" in data
        assert data["details"][0]["rows"][0]["label"] == "front_door"
