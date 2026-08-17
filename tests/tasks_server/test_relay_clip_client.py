"""The watcher can actually reach the relay to cut a clip.

Regression coverage for a bug the existing clip-backend tests could not see: they
built the task with `SimpleNamespace(get_relay_api_client=...)`, supplying a method
the real `Manager` did not have. `_clip_backend_for` looked it up behind a
`hasattr` guard, so in production every buffered (RTSP/MediaMTX) source fell through
to `None` and every `record_clip` event failed with "No clip backend available for
this source" — with a green suite. These tests use the real Manager on purpose.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import db_client
import inference_client
import pytest

from apps.helpers.actions.recordClipAction import PlaybackClipBackend
from apps.helpers.recordings.relayClipClient import RelayClipClient
from apps.helpers.watch.engine import WatchEngine
from apps.tasks_server.manager import Manager
from apps.tasks_server.tasks._watch_pipeline import WatchPipeline
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask


def _make_manager() -> Manager:
    return Manager(
        db_api_client=db_client.ApiClient(db_client.Configuration("http://localhost:8003")),
        inference_api_client=inference_client.ApiClient(inference_client.Configuration("http://localhost:8001")),
    )


def _buffered_pipeline(source_uuid: str = "src-1") -> WatchPipeline:
    return WatchPipeline(
        source_uuid=source_uuid,
        source_name="cam",
        provider=SimpleNamespace(),
        tracker=SimpleNamespace(update=lambda dets, now: []),
        engine=WatchEngine(rules=[], zones=[], source_uuid=source_uuid),
        buffered=True,
    )


class TestManagerExposesTheRelay:
    def test_the_manager_has_a_relay_client(self):
        assert isinstance(_make_manager().get_relay_api_client(), RelayClipClient)

    def test_the_client_is_reused_across_calls(self):
        """One client, so the connection pool is not rebuilt per event."""
        mgr = _make_manager()
        assert mgr.get_relay_api_client() is mgr.get_relay_api_client()

    def test_it_points_at_the_relay_buffer_server(self):
        from apps import APPS, APPS_CONFIG

        cfg = APPS_CONFIG[APPS.RELAY_BUFFER_SERVER]
        host = _make_manager().get_relay_api_client().api_client.configuration.host
        assert host == f"http://{cfg.host}:{cfg.port}"

    def test_a_buffered_source_gets_a_playback_backend_from_the_real_manager(self):
        """The end the mocked tests faked: real Manager in, real backend out."""
        task = StreamWatcherTask.__new__(StreamWatcherTask)
        task._pipelines = [_buffered_pipeline()]
        task._frame_rings = {}
        task._manager = _make_manager()

        backend = task._clip_backend_for(
            SimpleNamespace(profile_uuid="p", profile_name="P", source_uuid="src-1", source_name="cam")
        )

        assert isinstance(backend, PlaybackClipBackend)


class TestRelayClipClient:
    async def test_it_passes_the_window_through_as_the_api_expects(self):
        api = MagicMock()
        api.extract_clip.return_value = {"success": True, "path": "/clips/a.mp4", "sizeBytes": 10}
        start = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)

        with patch("apps.helpers.recordings.relayClipClient.relay_buffer_client.StreamsApi", return_value=api):
            result = await RelayClipClient(MagicMock()).extract_clip(
                path="cam-1", start=start, duration_s=8.0, dest=Path("/clips/a.mp4")
            )

        api.extract_clip.assert_called_once_with(
            path="cam-1", start=start.isoformat(), duration_s=8.0, dest="/clips/a.mp4"
        )
        # The envelope flag is the transport's, not the caller's business.
        assert result == {"path": "/clips/a.mp4", "sizeBytes": 10}

    async def test_a_naive_timestamp_is_read_as_utc(self):
        """Timestamps cross the wire as ISO strings; an unqualified one would be
        interpreted in the relay's local zone and cut the wrong window."""
        api = MagicMock()
        api.extract_clip.return_value = {"success": True}

        with patch("apps.helpers.recordings.relayClipClient.relay_buffer_client.StreamsApi", return_value=api):
            await RelayClipClient(MagicMock()).extract_clip(
                path="cam-1", start=datetime(2026, 8, 17, 12, 0), duration_s=1.0, dest=Path("/clips/a.mp4")
            )

        assert api.extract_clip.call_args.kwargs["start"].endswith("+00:00")

    async def test_a_failed_extraction_raises_rather_than_reporting_a_clip(self):
        """The endpoint reports failure in the body, so an unchecked caller would
        record a recording row pointing at a file that was never written."""
        api = MagicMock()
        api.extract_clip.return_value = {"success": False, "error": "window has aged out"}

        with patch("apps.helpers.recordings.relayClipClient.relay_buffer_client.StreamsApi", return_value=api):
            with pytest.raises(RuntimeError, match="aged out"):
                await RelayClipClient(MagicMock()).extract_clip(
                    path="cam-1", start=datetime.now(timezone.utc), duration_s=1.0, dest=Path("/clips/a.mp4")
                )
