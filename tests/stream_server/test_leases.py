"""Tests for the relay's lease/ensure lifecycle and the rolling-buffer guard rails.

The three properties under test are the ones a 24/7 watcher depends on and that the
previous implementation got wrong:

* master path names are **stable across restarts** (``recordPath`` templates on
  ``%path``, so an unstable name scatters recordings),
* startup **reconciles** surviving leases instead of calling ``destroy_all_streams()``,
* arming a source that would breach the ``/buffer`` mount is **refused up front**,
  because a full tmpfs fails writes rather than evicting early.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from apps.helpers.recordings.playbackClient import PlaybackError
from apps.stream_server.manager import (
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
        "apps.stream_server.manager.ConfigurationApi",
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


class TestExtractClipInPieces:
    """A cut spanning several advertised ranges must describe itself by measurement.

    `/list` is not a promise — it advertises the unfinalised segment, segments its own
    retention deleted, and overlapping ranges after a re-arm. A window spanning several
    of them can come back short with the shortfall anywhere inside, which leaves a
    detection box with no placeable position; the clip then loses its overlay entirely.
    Cutting per range makes every piece contiguous, so measuring it says exactly which
    wallclock it holds.
    """

    @staticmethod
    def _stub(manager, monkeypatch, ranges, held_by_start=None, refuse=()):
        """Playback stubbed so each piece can be given its own measured length.

        `held_by_start` maps a piece's requested start (seconds past base) to the video
        its file turns out to hold; `refuse` names starts the server will not serve.
        """
        written = {}

        async def write_clip(path, start, duration_s, dest, fmt="mp4"):
            key = round((start - ranges[0].start).total_seconds(), 3)
            if key in refuse:
                raise PlaybackError(f"no segment at {key}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x" * 100)
            written[key] = duration_s
            return 100

        async def probe(dest):
            key = round(written_key_of(dest), 3)
            return (held_by_start or {}).get(key)

        def written_key_of(dest):
            # The staging files are numbered in piece order, so the nth file is the nth
            # key written. Keeps the stub from needing to parse timestamps out of names.
            order = list(written)
            try:
                return order[int(dest.stem)]
            except (ValueError, IndexError):
                return -1.0

        client = MagicMock()
        client.list_ranges = AsyncMock(return_value=list(ranges))
        client.write_clip = AsyncMock(side_effect=write_clip)
        monkeypatch.setattr(manager, "playback_client", lambda: client)
        monkeypatch.setattr("apps.stream_server.manager.probe_duration_s", AsyncMock(side_effect=probe))
        # Patched on `clipDrain` itself, which is where the function-local import in
        # `_write_clip_in_pieces` resolves it from.
        monkeypatch.setattr(
            "apps.helpers.recordings.clipDrain.concat_parts",
            lambda parts, dest, max_length_s=None: (dest.write_bytes(b"y" * 200), 200)[1],
        )
        return client

    @staticmethod
    def _ranges(base):
        """The real deck case: two advertised ranges that overlap each other."""
        from apps.helpers.recordings.playbackClient import BufferedRange

        return [BufferedRange(base, 5.05), BufferedRange(base + timedelta(seconds=3.861), 9.99)]

    async def test_each_range_is_cut_separately(self, lease_manager, monkeypatch, tmp_path):
        base = datetime(2026, 10, 7, 11, 17, 36, tzinfo=timezone.utc)
        client = self._stub(lease_manager, monkeypatch, self._ranges(base), held_by_start={0.0: 5.05, 5.05: 2.898})
        await lease_manager.extract_clip("relay_master_abc", base, 7.948, tmp_path / "c.mp4")
        # Two calls, and the second starts where the first ended: the overlap is clipped
        # rather than served twice.
        starts = [round((call.args[1] - base).total_seconds(), 3) for call in client.write_clip.call_args_list]
        assert starts == [0.0, 5.05]

    async def test_covered_durations_are_measured_not_advertised(self, lease_manager, monkeypatch, tmp_path):
        base = datetime(2026, 10, 7, 11, 17, 36, tzinfo=timezone.utc)
        # The second piece delivers 1.2s of the 2.898s it was asked for.
        self._stub(lease_manager, monkeypatch, self._ranges(base), held_by_start={0.0: 5.05, 5.05: 1.2})
        result = await lease_manager.extract_clip("relay_master_abc", base, 7.948, tmp_path / "c.mp4")
        held = [r["durationS"] for r in result["coveredRanges"]]
        assert held == [5.05, 1.2], "a short piece must report what it holds, not what it claimed"

    async def test_covered_ranges_sum_to_the_measured_file(self, lease_manager, monkeypatch, tmp_path):
        """The invariant the segment map checks before it will place a box."""
        from apps.helpers.recordings.clipDrain import map_from_windows

        base = datetime(2026, 10, 7, 11, 17, 36, tzinfo=timezone.utc)
        self._stub(lease_manager, monkeypatch, self._ranges(base), held_by_start={0.0: 5.05, 5.05: 1.2})
        result = await lease_manager.extract_clip("relay_master_abc", base, 7.948, tmp_path / "c.mp4")
        windows = [(datetime.fromisoformat(r["startedAt"]), r["durationS"]) for r in result["coveredRanges"]]
        total = sum(d for _, d in windows)
        # The map is only emitted when the windows agree with the file, so this is the
        # difference between a clip that keeps its boxes and one that loses them.
        assert map_from_windows(windows, total) is not None
        assert map_from_windows(windows, total + 5) is None, "a map that disagrees with the file must still be refused"

    async def test_a_piece_the_server_refuses_becomes_a_hole(self, lease_manager, monkeypatch, tmp_path):
        base = datetime(2026, 10, 7, 11, 17, 36, tzinfo=timezone.utc)
        self._stub(lease_manager, monkeypatch, self._ranges(base), held_by_start={0.0: 5.05}, refuse={5.05})
        result = await lease_manager.extract_clip("relay_master_abc", base, 7.948, tmp_path / "c.mp4")
        # One evicted segment costs its own stretch, not the whole event.
        assert [r["durationS"] for r in result["coveredRanges"]] == [5.05]

    async def test_every_piece_refused_raises(self, lease_manager, monkeypatch, tmp_path):
        base = datetime(2026, 10, 7, 11, 17, 36, tzinfo=timezone.utc)
        self._stub(lease_manager, monkeypatch, self._ranges(base), refuse={0.0, 5.05})
        with pytest.raises(PlaybackError):
            await lease_manager.extract_clip("relay_master_abc", base, 7.948, tmp_path / "c.mp4")

    async def test_a_single_range_is_still_one_cut(self, lease_manager, monkeypatch, tmp_path):
        """The common case must not pay for staging, concatenation or extra requests."""
        from apps.helpers.recordings.playbackClient import BufferedRange

        base = datetime(2026, 10, 7, 11, 17, 36, tzinfo=timezone.utc)
        client = self._stub(lease_manager, monkeypatch, [BufferedRange(base, 60)], held_by_start={0.0: 20})
        result = await lease_manager.extract_clip("relay_master_abc", base, 20, tmp_path / "c.mp4")
        assert client.write_clip.await_count == 1
        assert [r["durationS"] for r in result["coveredRanges"]] == [20]


class TestLeaseAccounting:
    def test_recording_lease_commits_bitrate_times_retention(self):
        lease = Lease(
            lease_id="l", source_url="rtsp://c", holder="h", retention_s=60, record=True, assumed_bitrate_bps=4_000_000
        )
        assert lease.committed_bytes() == 30_000_000

    def test_non_recording_lease_commits_nothing(self):
        lease = Lease(lease_id="l", source_url="rtsp://c", holder="h", retention_s=60, record=False)
        assert lease.committed_bytes() == 0


class TestPurgingADestroyedPathsBuffer:
    """Segments outlive the path that wrote them, and that is not only a disk leak.

    MediaMTX applies `recordDeleteAfter` only to configured paths, and path names are
    stable across re-arms, so a re-armed source inherits its predecessor's segments.
    `/list` then advertises ranges that overlap each other and leave holes between
    them, and a cut spanning those comes back shorter than it was charged for.
    """

    async def test_segments_of_a_destroyed_path_are_removed(self, lease_manager, tmp_path):
        lease_manager._buffer_dir = tmp_path / "buffer"
        doomed = lease_manager._buffer_dir / "relay_master_dead"
        doomed.mkdir(parents=True)
        (doomed / "2026-10-04_14-05-07-160403.mp4").write_bytes(b"segment")

        await lease_manager._purge_buffer("relay_master_dead")

        assert not doomed.exists()

    async def test_other_paths_are_untouched(self, lease_manager, tmp_path):
        """One camera going to sleep must not take another's rewind with it."""
        lease_manager._buffer_dir = tmp_path / "buffer"
        keep = lease_manager._buffer_dir / "relay_master_live"
        keep.mkdir(parents=True)
        (keep / "seg.mp4").write_bytes(b"segment")
        (lease_manager._buffer_dir / "relay_master_dead").mkdir()

        await lease_manager._purge_buffer("relay_master_dead")

        assert (keep / "seg.mp4").is_file()

    async def test_a_traversing_name_cannot_escape_the_buffer(self, lease_manager, tmp_path):
        """Path names are hashed, so this is defence rather than a live hazard — but
        the argument reaches `rmtree`, so the guard is asserted rather than assumed."""
        lease_manager._buffer_dir = tmp_path / "buffer"
        lease_manager._buffer_dir.mkdir(parents=True)
        outside = tmp_path / "secrets"
        outside.mkdir()
        (outside / "keep.txt").write_bytes(b"data")

        await lease_manager._purge_buffer("../secrets")

        assert (outside / "keep.txt").is_file()

    async def test_a_missing_directory_is_not_an_error(self, lease_manager, tmp_path):
        """A path destroyed before it ever recorded has no directory, and teardown must
        not fail on it."""
        lease_manager._buffer_dir = tmp_path / "buffer"
        lease_manager._buffer_dir.mkdir(parents=True)

        await lease_manager._purge_buffer("relay_master_never_recorded")


class TestSweepingOrphanBuffers:
    """Teardown-time purging misses anything that never had an orderly teardown — a
    crashed relay, a failed `config_paths_delete`, or segments predating the purge.
    Nothing else collects them: MediaMTX applies retention only to configured paths.
    """

    @staticmethod
    def _seg(manager, path_name, age_s):
        import os
        import time as _time

        d = manager._buffer_dir / path_name
        d.mkdir(parents=True, exist_ok=True)
        f = d / "2026-10-04_14-05-07-160403.mp4"
        f.write_bytes(b"segment")
        stamp = _time.time() - age_s
        os.utime(f, (stamp, stamp))
        return d

    async def test_an_unconfigured_paths_old_segments_are_swept(self, lease_manager, tmp_path):
        lease_manager._buffer_dir = tmp_path / "buffer"
        orphan = self._seg(lease_manager, "relay_master_orphan", age_s=3600)
        lease_manager._streams = {}

        assert await lease_manager.sweep_orphan_buffers() == 1
        assert not orphan.exists()

    async def test_a_configured_paths_segments_survive(self, lease_manager, tmp_path):
        """The live case, and the one that matters: sweeping a recording path would
        delete the rewind its own events are about to be cut from."""
        lease_manager._buffer_dir = tmp_path / "buffer"
        live = self._seg(lease_manager, "relay_master_live", age_s=3600)
        lease_manager._streams = {"relay_master_live": object()}

        assert await lease_manager.sweep_orphan_buffers() == 0
        assert live.is_dir()

    async def test_a_recently_written_path_is_left_alone(self, lease_manager, tmp_path):
        """A path can be writing segments before `refresh_streams` has listed it, so
        recency is what keeps a mid-creation path out of range."""
        lease_manager._buffer_dir = tmp_path / "buffer"
        fresh = self._seg(lease_manager, "relay_master_new", age_s=0)
        lease_manager._streams = {}

        assert await lease_manager.sweep_orphan_buffers() == 0
        assert fresh.is_dir()

    async def test_a_missing_buffer_root_is_not_an_error(self, lease_manager, tmp_path):
        lease_manager._buffer_dir = tmp_path / "nonexistent"
        assert await lease_manager.sweep_orphan_buffers() == 0


class TestReleasingTheLastLeasePurgesTheBuffer:
    """`release_lease` tears the path down itself rather than calling `destroy_stream`,
    and it is the teardown every on-demand camera takes on each sleep cycle. Leaving
    its segments lets the next arming of the same path — the name is stable — inherit
    them, which is what makes `/list` report overlapping and gapped ranges.
    """

    async def test_the_last_holder_leaving_takes_the_segments(self, lease_manager, tmp_path):
        lease_manager._buffer_dir = tmp_path / "buffer"
        url = "rtsp://cam/stream"
        name = master_path_name(url)
        segments = lease_manager._buffer_dir / name
        segments.mkdir(parents=True)
        (segments / "seg.mp4").write_bytes(b"video")
        lease_manager._leases = {"L1": Lease(lease_id="L1", source_url=url, holder="h", retention_s=45.0, record=True)}

        await lease_manager.release_lease("L1")

        assert not segments.exists()

    async def test_a_remaining_holder_keeps_the_segments(self, lease_manager, tmp_path):
        """Two watchers on one camera share a path; one leaving must not cut the other's
        rewind out from under it."""
        lease_manager._buffer_dir = tmp_path / "buffer"
        url = "rtsp://cam/stream"
        name = master_path_name(url)
        segments = lease_manager._buffer_dir / name
        segments.mkdir(parents=True)
        (segments / "seg.mp4").write_bytes(b"video")
        lease_manager._leases = {
            "L1": Lease(lease_id="L1", source_url=url, holder="h1", retention_s=45.0, record=True),
            "L2": Lease(lease_id="L2", source_url=url, holder="h2", retention_s=45.0, record=True),
        }

        await lease_manager.release_lease("L1")

        assert (segments / "seg.mp4").is_file()


class TestRearmingPathsTheRelayLost:
    """The relay restarts independently of this process — it exits on a panic in its
    own HTTP handler — and comes back with an empty path table.

    Everything here survives that: the leases, the refcounts, and the `_source_relays`
    entry `ensure_stream` consults. Nothing re-armed, because holders only renew (which
    bumps `last_seen`) and `ensure_stream` short-circuited on the surviving entry. One
    relay crash therefore took every camera off the air until someone restarted this
    service by hand.
    """

    SOURCE = "rtsp://cam.example:8554/unicast"

    def _armed(self, mgr, source=SOURCE, record=True):
        """Put `mgr` in the state it is in after a successful ensure."""
        name = master_path_name(source)
        mgr._source_relays[source] = Manager.SourceRelayInfo(master_stream_name=name, refcount=1)
        mgr._streams[name] = Manager.StreamInfo(name=name)
        lease = Lease(
            lease_id="lease-1",
            source_url=source,
            holder="provider-1",
            retention_s=45.0,
            record=record,
        )
        mgr._leases[lease.lease_id] = lease
        return name

    def _relay_lists(self, monkeypatch, mgr, names):
        """Make the relay report exactly `names` as configured."""
        listed = MagicMock()
        listed.items = [MagicMock(name=n) for n in names]
        # MagicMock(name=...) sets the mock's own name, not an attribute, so set it.
        for item, n in zip(listed.items, names):
            item.name = n
        api = MagicMock()
        api.config_paths_list = MagicMock(return_value=listed)
        monkeypatch.setattr("apps.stream_server.manager.ConfigurationApi", MagicMock(return_value=api))
        return api

    async def test_a_leased_path_the_relay_lost_is_re_armed(self, lease_manager, monkeypatch):
        name = self._armed(lease_manager)
        self._relay_lists(monkeypatch, lease_manager, [])
        created = AsyncMock()
        monkeypatch.setattr(lease_manager, "_create_master_path", created)

        await lease_manager._rearm_lost_paths()

        assert created.await_count == 1, "a leased path missing from the relay must be re-armed"
        assert created.await_args.args[0] == self.SOURCE
        assert created.await_args.args[1] == name

    async def test_a_path_the_relay_still_has_is_left_alone(self, lease_manager, monkeypatch):
        """Re-arming a live path would drop the camera connection it already has."""
        name = self._armed(lease_manager)
        self._relay_lists(monkeypatch, lease_manager, [name])
        created = AsyncMock()
        monkeypatch.setattr(lease_manager, "_create_master_path", created)

        await lease_manager._rearm_lost_paths()

        assert created.await_count == 0

    async def test_an_unleased_lost_path_is_forgotten_rather_than_re_armed(self, lease_manager, monkeypatch):
        """Nobody wants it, so re-arming would reconnect a camera for no one — but the
        stale entry must still go, because that entry is what makes the next `ensure`
        answer success for a path the relay does not have."""
        name = self._armed(lease_manager)
        lease_manager._leases.clear()
        self._relay_lists(monkeypatch, lease_manager, [])
        created = AsyncMock()
        monkeypatch.setattr(lease_manager, "_create_master_path", created)

        await lease_manager._rearm_lost_paths()

        assert created.await_count == 0
        assert self.SOURCE not in lease_manager._source_relays
        assert name not in lease_manager._streams

    async def test_a_relay_that_cannot_be_asked_changes_nothing(self, lease_manager, monkeypatch):
        """An API call that fails says nothing about whether the path is there. Treating
        that as "lost" would tear down and rebuild every camera on a blip."""
        self._armed(lease_manager)
        api = MagicMock()
        api.config_paths_list = MagicMock(side_effect=RuntimeError("relay down"))
        monkeypatch.setattr("apps.stream_server.manager.ConfigurationApi", MagicMock(return_value=api))
        created = AsyncMock()
        monkeypatch.setattr(lease_manager, "_create_master_path", created)

        await lease_manager._rearm_lost_paths()

        assert created.await_count == 0
        assert self.SOURCE in lease_manager._source_relays, "the entry must survive an unanswered relay"

    async def test_one_camera_failing_to_re_arm_does_not_block_the_others(self, lease_manager, monkeypatch):
        other = "rtsp://cam2.example:8554/unicast"
        self._armed(lease_manager)
        name2 = master_path_name(other)
        lease_manager._source_relays[other] = Manager.SourceRelayInfo(master_stream_name=name2, refcount=1)
        lease_manager._leases["lease-2"] = Lease(
            lease_id="lease-2", source_url=other, holder="provider-2", retention_s=45.0, record=True
        )
        self._relay_lists(monkeypatch, lease_manager, [])

        attempted = []

        async def _create(source_url, name, retention, record):
            attempted.append(source_url)
            if source_url == self.SOURCE:
                raise RuntimeError("camera unreachable")

        monkeypatch.setattr(lease_manager, "_create_master_path", _create)

        await lease_manager._rearm_lost_paths()

        assert sorted(attempted) == sorted([self.SOURCE, other])

    async def test_ensure_re_arms_instead_of_reporting_a_path_that_is_gone(self, lease_manager, monkeypatch):
        """The dishonest answer that hid the outage: `ensure` returned success, a path
        name and a lease id for a path the relay had already forgotten."""
        self._armed(lease_manager)
        self._relay_lists(monkeypatch, lease_manager, [])
        created = AsyncMock()
        monkeypatch.setattr(lease_manager, "_create_master_path", created)
        monkeypatch.setattr(lease_manager, "_check_commit", MagicMock())

        await lease_manager.ensure_stream(source_url=self.SOURCE, holder="provider-1")

        assert created.await_count == 1, "ensure must not trust its own memory of having armed the path"

    async def test_a_leased_source_with_no_relay_entry_is_still_re_armed(self, lease_manager, monkeypatch):
        """`_source_relays` is this process's memory of having armed a path; the lease is
        the durable claim a holder is still renewing. Keying recovery on the former
        leaves a renewing holder with no path and nothing that will ever notice."""
        name = self._armed(lease_manager)
        lease_manager._source_relays.clear()
        self._relay_lists(monkeypatch, lease_manager, [])
        created = AsyncMock()
        monkeypatch.setattr(lease_manager, "_create_master_path", created)

        await lease_manager._rearm_lost_paths()

        assert created.await_count == 1, "the lease alone must be enough to bring the path back"
        assert created.await_args.args[1] == name


class TestTheWorkerLoopOutlivesTheRelay:
    """The periodic worker is the only thing that re-arms a path the relay lost, so it
    has to outlive the relay crashing underneath it.

    MediaMTX exits on a panic in its own HTTP handler and supervisord brings it back
    with an empty path table. Any call into it therefore fails for the seconds in
    between — and an unguarded step let that exception end the ``while True``, killing
    lease expiry, the orphan sweep and the re-arm for the life of the process. Holders
    kept renewing leases that no longer had a path behind them, so every camera whose
    consumer only renews stayed off the air until someone restarted this service.
    """

    @pytest.fixture
    def worker(self, manager, monkeypatch):
        """`manager` with the loop sped up and every step but the re-arm stubbed."""
        monkeypatch.setattr(Manager, "POLLING_INTERVAL", 0.01)
        monkeypatch.setattr(manager, "_expire_stale_leases", AsyncMock())
        monkeypatch.setattr(manager, "sweep_orphan_buffers", AsyncMock())
        return manager

    @staticmethod
    def _signal_on_rearm(manager, monkeypatch):
        """An event set the first time the re-arm step runs, so no test needs a sleep."""
        reached = asyncio.Event()
        monkeypatch.setattr(manager, "_rearm_lost_paths", AsyncMock(side_effect=lambda: reached.set()))
        return reached

    async def test_a_step_that_raises_does_not_stop_the_re_arm_behind_it(self, worker, monkeypatch):
        """`refresh_streams` is the step that actually failed: it polls the relay's path
        list, so it is the first thing to raise when the relay is mid-restart — which is
        exactly the tick on which the re-arm is needed."""
        monkeypatch.setattr(worker, "_init", AsyncMock())
        monkeypatch.setattr(worker, "refresh_streams", AsyncMock(side_effect=RuntimeError("connection refused")))
        reached = self._signal_on_rearm(worker, monkeypatch)

        worker.start()
        try:
            await asyncio.wait_for(reached.wait(), timeout=5)
            assert not worker._task.done(), "the loop must still be running after a step raised"
        finally:
            worker._task.cancel()

    async def test_init_against_a_down_relay_is_retried_rather_than_fatal(self, worker, monkeypatch):
        """Supervisord starts this service and the relay together, so the first `_init`
        can land before the relay's API is listening."""
        attempts = []

        async def flaky_init():
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("connection refused")

        monkeypatch.setattr(worker, "_init", flaky_init)
        reached = self._signal_on_rearm(worker, monkeypatch)

        worker.start()
        try:
            await asyncio.wait_for(reached.wait(), timeout=5)
            assert len(attempts) == 3, "the worker must keep retrying init until the relay answers"
        finally:
            worker._task.cancel()
