"""Tests for the relay's lease/ensure lifecycle and the rolling-buffer guard rails.

The three properties under test are the ones a 24/7 watcher depends on and that the
previous implementation got wrong:

* master path names are **stable across restarts** (``recordPath`` templates on
  ``%path``, so an unstable name scatters recordings),
* startup **reconciles** surviving leases instead of calling ``destroy_all_streams()``,
* arming a source that would breach the ``/buffer`` mount is **refused up front**,
  because a full tmpfs fails writes rather than evicting early.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from apps.helpers.recordings.playbackClient import PlaybackError
from apps.relay_buffer_server.manager import (
    BufferFullError,
    Lease,
    Manager,
    master_path_name,
)


@pytest.fixture
def lease_manager(tmp_path, monkeypatch):
    """A Manager with mediamtx and the buffer mount stubbed, persisting into tmp_path."""
    api_client = MagicMock()
    api_client.configuration.host = "http://localhost:9997"
    mgr = Manager(media_mtx_api_client=api_client)
    mgr._lease_file = tmp_path / "leases.json"
    monkeypatch.setattr(mgr, "refresh_streams", AsyncMock())
    # _create_master_path is exercised directly in its own tests; here we only care
    # about lease bookkeeping, so stub the mediamtx round-trip.
    monkeypatch.setattr(
        "apps.relay_buffer_server.manager.ConfigurationApi",
        MagicMock(),
    )
    return mgr


class TestMasterPathName:
    def test_is_deterministic(self):
        assert master_path_name("rtsp://cam/stream") == master_path_name("rtsp://cam/stream")

    def test_survives_whitespace_differences(self):
        assert master_path_name("  rtsp://cam/stream ") == master_path_name("rtsp://cam/stream")

    def test_differs_per_source(self):
        assert master_path_name("rtsp://cam/a") != master_path_name("rtsp://cam/b")

    def test_is_a_legal_mediamtx_path_name(self):
        # Credentials and slashes in the URL must not leak into the path name, since
        # recordPath expands %path into a directory.
        name = master_path_name("rtsp://user:p@ss@cam.local:554/h264/ch1")
        assert name.replace("_", "").isalnum()


class TestEnsureStream:
    async def test_returns_master_path_and_lease(self, lease_manager):
        result = await lease_manager.ensure_stream("rtsp://cam/stream", holder="watcher-1")
        assert result.path == master_path_name("rtsp://cam/stream")
        assert result.rtsp_url.endswith(result.path)
        assert result.lease_id
        assert result.recording is True

    async def test_is_idempotent_for_the_same_source(self, lease_manager):
        first = await lease_manager.ensure_stream("rtsp://cam/stream", holder="a")
        second = await lease_manager.ensure_stream("rtsp://cam/stream", holder="b")
        assert first.path == second.path
        assert first.lease_id != second.lease_id
        # One path, one camera connection — two holders.
        assert len(lease_manager.leases_for("rtsp://cam/stream")) == 2

    async def test_two_watchers_share_one_source(self, lease_manager):
        await lease_manager.ensure_stream("rtsp://cam/stream", holder="watcher-1")
        await lease_manager.ensure_stream("rtsp://cam/stream", holder="watcher-2")
        name = master_path_name("rtsp://cam/stream")
        assert lease_manager.reader_counts()[name] == 2

    async def test_persists_leases_for_restart(self, lease_manager):
        await lease_manager.ensure_stream("rtsp://cam/stream", holder="watcher-1")
        assert lease_manager._lease_file.exists()
        assert "rtsp://cam/stream" in lease_manager._lease_file.read_text()

    async def test_non_recording_lease_reports_no_buffer(self, lease_manager):
        result = await lease_manager.ensure_stream("rtsp://cam/stream", holder="a", record=False)
        # The caller must be able to say "this source has no rewind buffer" rather
        # than silently producing zero-pre-roll clips.
        assert result.recording is False
        assert result.retention_s == 0.0


class TestReleaseLease:
    async def test_path_survives_while_another_holder_remains(self, lease_manager):
        first = await lease_manager.ensure_stream("rtsp://cam/stream", holder="a")
        await lease_manager.ensure_stream("rtsp://cam/stream", holder="b")
        await lease_manager.release_lease(first.lease_id)
        assert "rtsp://cam/stream" in lease_manager._source_relays

    async def test_last_release_tears_the_path_down(self, lease_manager):
        result = await lease_manager.ensure_stream("rtsp://cam/stream", holder="a")
        await lease_manager.release_lease(result.lease_id)
        assert "rtsp://cam/stream" not in lease_manager._source_relays
        assert result.path not in lease_manager._streams

    async def test_unknown_lease_is_reported_not_raised(self, lease_manager):
        assert await lease_manager.release_lease("nope") is False


class TestReconcile:
    async def test_startup_rearms_surviving_leases_instead_of_wiping(self, lease_manager, monkeypatch):
        await lease_manager.ensure_stream("rtsp://cam/stream", holder="watcher-1")
        lease_file = lease_manager._lease_file

        # A fresh process reading the same lease file: it must re-create the path, and
        # must NOT call destroy_all_streams(), which would break live consumers.
        api_client = MagicMock()
        api_client.configuration.host = "http://localhost:9997"
        revived = Manager(media_mtx_api_client=api_client)
        revived._lease_file = lease_file
        monkeypatch.setattr(revived, "refresh_streams", AsyncMock())
        destroyer = AsyncMock()
        monkeypatch.setattr(revived, "destroy_all_streams", destroyer)
        creator = AsyncMock()
        monkeypatch.setattr(revived, "_create_master_path", creator)

        await revived._reconcile()

        destroyer.assert_not_called()
        creator.assert_awaited_once()
        assert creator.await_args.args[0] == "rtsp://cam/stream"
        assert len(revived._leases) == 1

    async def test_one_unreachable_camera_does_not_block_the_rest(self, lease_manager, monkeypatch, tmp_path):
        await lease_manager.ensure_stream("rtsp://bad/stream", holder="a")
        await lease_manager.ensure_stream("rtsp://good/stream", holder="b")

        api_client = MagicMock()
        api_client.configuration.host = "http://localhost:9997"
        revived = Manager(media_mtx_api_client=api_client)
        revived._lease_file = lease_manager._lease_file
        monkeypatch.setattr(revived, "refresh_streams", AsyncMock())

        armed = []

        async def flaky(source_url, name, retention_s, record):
            if "bad" in source_url:
                raise RuntimeError("camera unreachable")
            armed.append(source_url)
            revived._source_relays[source_url] = Manager.SourceRelayInfo(master_stream_name=name)

        monkeypatch.setattr(revived, "_create_master_path", flaky)
        await revived._reconcile()
        assert armed == ["rtsp://good/stream"]

    async def test_corrupt_lease_file_does_not_crash_startup(self, lease_manager):
        lease_manager._lease_file.write_text("{not json")
        lease_manager._load_leases()
        assert lease_manager._leases == {}


class TestBufferCommitment:
    async def test_refuses_a_source_that_would_breach_the_mount(self, lease_manager):
        # A tiny cap so one 4 Mbps camera-minute already exceeds it. The point is that
        # this fails at arm time, not later as a mysterious recorder write error.
        lease_manager._buffer_cap_bytes = 1_000_000
        with pytest.raises(BufferFullError):
            await lease_manager.ensure_stream("rtsp://cam/stream", holder="a", retention_s=60)

    async def test_non_recording_sources_commit_nothing(self, lease_manager):
        lease_manager._buffer_cap_bytes = 1_000_000
        result = await lease_manager.ensure_stream("rtsp://cam/stream", holder="a", record=False)
        assert result.lease_id
        assert lease_manager.buffer_usage().committed_bytes == 0

    async def test_committed_bytes_accumulate_across_sources(self, lease_manager):
        await lease_manager.ensure_stream("rtsp://cam/a", holder="a", retention_s=10)
        first = lease_manager.buffer_usage().committed_bytes
        await lease_manager.ensure_stream("rtsp://cam/b", holder="b", retention_s=10)
        assert lease_manager.buffer_usage().committed_bytes == 2 * first

    async def test_releasing_frees_the_commitment(self, lease_manager):
        result = await lease_manager.ensure_stream("rtsp://cam/a", holder="a", retention_s=10)
        assert lease_manager.buffer_usage().committed_bytes > 0
        await lease_manager.release_lease(result.lease_id)
        assert lease_manager.buffer_usage().committed_bytes == 0

    def test_declared_cap_never_exceeds_the_real_mount(self, lease_manager, tmp_path):
        # If someone mounts /buffer smaller than STREAM_BUFFER_SIZE claims, the mount
        # wins — it is the thing that actually fails writes.
        lease_manager._buffer_cap_bytes = 10**15
        lease_manager._buffer_dir = tmp_path
        usage = lease_manager.buffer_usage()
        assert usage.cap_bytes < 10**15


class TestExtractClip:
    """Clip extraction must never present a truncated window as a complete one.

    The buffer is RAM-backed, so a relay restart or a slow dispatch silently shortens
    the available history — the clamp has to be reported, not applied quietly.
    """

    @staticmethod
    def _stub_playback(manager, monkeypatch, ranges, written=1234):
        from unittest.mock import AsyncMock

        client = MagicMock()
        client.list_ranges = AsyncMock(return_value=ranges)
        client.write_clip = AsyncMock(return_value=written)
        monkeypatch.setattr(manager, "playback_client", lambda: client)
        return client

    async def test_full_window_is_not_flagged_partial(self, lease_manager, monkeypatch, tmp_path):
        from apps.helpers.recordings.playbackClient import BufferedRange

        base = datetime(2026, 8, 10, 12, tzinfo=timezone.utc)
        self._stub_playback(lease_manager, monkeypatch, [BufferedRange(base, 60)])
        result = await lease_manager.extract_clip(
            "relay_master_abc", base + timedelta(seconds=10), 20, tmp_path / "c.mp4"
        )
        assert result["partialPreroll"] is False
        assert result["durationS"] == 20

    async def test_preroll_older_than_the_buffer_is_clamped_and_flagged(self, lease_manager, monkeypatch, tmp_path):
        from apps.helpers.recordings.playbackClient import BufferedRange

        base = datetime(2026, 8, 10, 12, tzinfo=timezone.utc)
        self._stub_playback(lease_manager, monkeypatch, [BufferedRange(base, 60)])
        # Asking for 30 s of pre-roll before the buffer even starts.
        result = await lease_manager.extract_clip(
            "relay_master_abc", base - timedelta(seconds=30), 60, tmp_path / "c.mp4"
        )
        assert result["partialPreroll"] is True
        assert result["startedAt"] == base.isoformat()
        assert result["durationS"] == 30  # clamped to what survives

    async def test_window_entirely_aged_out_raises(self, lease_manager, monkeypatch, tmp_path):
        from apps.helpers.recordings.playbackClient import BufferedRange

        base = datetime(2026, 8, 10, 12, tzinfo=timezone.utc)
        self._stub_playback(lease_manager, monkeypatch, [BufferedRange(base, 60)])
        with pytest.raises(PlaybackError):
            await lease_manager.extract_clip("relay_master_abc", base - timedelta(seconds=600), 30, tmp_path / "c.mp4")

    async def test_no_recording_at_all_raises(self, lease_manager, monkeypatch, tmp_path):
        self._stub_playback(lease_manager, monkeypatch, [])
        with pytest.raises(PlaybackError):
            await lease_manager.extract_clip("relay_master_abc", datetime.now(timezone.utc), 30, tmp_path / "c.mp4")


class TestLeaseAccounting:
    def test_recording_lease_commits_bitrate_times_retention(self):
        lease = Lease(
            lease_id="l", source_url="rtsp://c", holder="h", retention_s=60, record=True, assumed_bitrate_bps=4_000_000
        )
        assert lease.committed_bytes() == 30_000_000

    def test_non_recording_lease_commits_nothing(self):
        lease = Lease(lease_id="l", source_url="rtsp://c", holder="h", retention_s=60, record=False)
        assert lease.committed_bytes() == 0
