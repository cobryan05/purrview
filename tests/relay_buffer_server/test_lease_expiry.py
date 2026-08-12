"""Reaping leases whose holder stopped checking in.

Nothing below this layer can tell a live watcher from a crashed one. MediaMTX retries a
dead camera forever — deliberately, because a camera that is merely offline must come
back on its own when it returns — so an idle path proves nothing about its holder. Four
leases against long-dead `localhost` paths survived weeks of restarts on exactly that
gap, each one re-armed at startup on behalf of a process that no longer existed.

The TTL closes it, and the distinction it draws is the whole point:

* a holder that stops renewing is **dead**, and its lease is reaped;
* a *camera* that stops responding is **not**, and its lease is untouched, because its
  live holder still wants the path re-armed when the camera comes back.

Getting that backwards would break the ordinary offline-camera case, so it is pinned
here explicitly.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from apps.relay_buffer_server.manager import Lease, Manager, master_path_name

SOURCE = "rtsp://camera.invalid:8554/stream"
OTHER_SOURCE = "rtsp://other.invalid:8554/stream"


@pytest.fixture
def lease_manager(tmp_path, monkeypatch):
    api_client = MagicMock()
    api_client.configuration.host = "http://localhost:9997"
    mgr = Manager(media_mtx_api_client=api_client)
    mgr._lease_file = tmp_path / "leases.json"
    monkeypatch.setattr(mgr, "refresh_streams", AsyncMock())
    monkeypatch.setattr(mgr, "_create_master_path", AsyncMock())
    monkeypatch.setattr("apps.relay_buffer_server.manager.ConfigurationApi", MagicMock())
    return mgr


def _add_lease(mgr, lease_id, source=SOURCE, holder="watcher-1", age_s=0.0):
    """Insert a lease that last checked in *age_s* seconds ago."""
    lease = Lease(
        lease_id=lease_id,
        source_url=source,
        holder=holder,
        retention_s=45.0,
        record=True,
    )
    lease.last_seen = time.time() - age_s
    mgr._leases[lease_id] = lease
    mgr._source_relays[source] = Manager.SourceRelayInfo(master_stream_name=master_path_name(source), refcount=1)
    return lease


class TestReapingDeadHolders:
    async def test_a_lease_nobody_renews_is_reaped(self, lease_manager, monkeypatch):
        monkeypatch.setattr("apps.relay_buffer_server.manager.LEASE_TTL_S", 300.0)
        _add_lease(lease_manager, "stale-1", age_s=301)

        await lease_manager._expire_stale_leases()

        assert "stale-1" not in lease_manager._leases

    async def test_a_renewed_lease_survives(self, lease_manager, monkeypatch):
        monkeypatch.setattr("apps.relay_buffer_server.manager.LEASE_TTL_S", 300.0)
        _add_lease(lease_manager, "live-1", age_s=299)

        await lease_manager._expire_stale_leases()

        assert "live-1" in lease_manager._leases

    async def test_renewing_resets_the_clock(self, lease_manager, monkeypatch):
        monkeypatch.setattr("apps.relay_buffer_server.manager.LEASE_TTL_S", 300.0)
        _add_lease(lease_manager, "live-1", age_s=299)

        assert lease_manager.renew_lease("live-1") is True
        await lease_manager._expire_stale_leases()

        assert "live-1" in lease_manager._leases

    async def test_renewing_an_unknown_lease_reports_failure(self, lease_manager):
        """The holder's cue to re-ensure, so it must not silently succeed."""
        assert lease_manager.renew_lease("never-existed") is False

    async def test_reaping_the_last_holder_tears_the_path_down(self, lease_manager, monkeypatch):
        monkeypatch.setattr("apps.relay_buffer_server.manager.LEASE_TTL_S", 300.0)
        _add_lease(lease_manager, "stale-1", age_s=999)
        lease_manager._streams[master_path_name(SOURCE)] = Manager.StreamInfo(name=master_path_name(SOURCE))

        await lease_manager._expire_stale_leases()

        assert master_path_name(SOURCE) not in lease_manager._streams
        assert SOURCE not in lease_manager._source_relays

    async def test_one_dead_holder_does_not_evict_a_live_one(self, lease_manager, monkeypatch):
        """Two watchers on one camera: reaping one must leave the path up for the other."""
        monkeypatch.setattr("apps.relay_buffer_server.manager.LEASE_TTL_S", 300.0)
        _add_lease(lease_manager, "stale-1", holder="watcher-1", age_s=999)
        _add_lease(lease_manager, "live-1", holder="watcher-2", age_s=0)
        lease_manager._streams[master_path_name(SOURCE)] = Manager.StreamInfo(name=master_path_name(SOURCE))

        await lease_manager._expire_stale_leases()

        assert "stale-1" not in lease_manager._leases
        assert "live-1" in lease_manager._leases
        assert master_path_name(SOURCE) in lease_manager._streams, "the surviving holder must keep its path"

    async def test_expiry_sweeps_every_source_not_just_the_first(self, lease_manager, monkeypatch):
        monkeypatch.setattr("apps.relay_buffer_server.manager.LEASE_TTL_S", 300.0)
        _add_lease(lease_manager, "stale-1", source=SOURCE, age_s=999)
        _add_lease(lease_manager, "stale-2", source=OTHER_SOURCE, age_s=999)

        await lease_manager._expire_stale_leases()

        assert lease_manager._leases == {}


class TestAnOfflineCameraIsNotADeadHolder:
    async def test_an_unreachable_camera_keeps_its_lease(self, lease_manager, monkeypatch):
        """The case that must never regress.

        A camera offline overnight — PoE cycling, a router reboot — has a live watcher
        that still wants it. Reaping here would mean the camera never comes back on its
        own, defeating the infinite retry MediaMTX is configured for.
        """
        monkeypatch.setattr("apps.relay_buffer_server.manager.LEASE_TTL_S", 300.0)
        lease = _add_lease(lease_manager, "live-1", age_s=0)
        name = master_path_name(SOURCE)
        # Never ready, nothing received, for far longer than the TTL: as dead as a
        # camera can look from here.
        lease_manager._streams[name] = Manager.StreamInfo(name=name, active=False)

        await lease_manager._expire_stale_leases()

        assert lease.lease_id in lease_manager._leases
        assert name in lease_manager._streams


class TestHolderKeyedEnsure:
    """A named holder re-ensuring is checking in, not claiming a second time."""

    async def test_re_ensuring_reuses_the_same_lease(self, lease_manager):
        first = await lease_manager.ensure_stream(SOURCE, holder="watcher-1")
        second = await lease_manager.ensure_stream(SOURCE, holder="watcher-1")

        assert first.lease_id == second.lease_id
        assert len(lease_manager._leases) == 1, "a re-ensure must not stack a second lease"

    async def test_re_ensuring_refreshes_the_clock(self, lease_manager, monkeypatch):
        monkeypatch.setattr("apps.relay_buffer_server.manager.LEASE_TTL_S", 300.0)
        ensured = await lease_manager.ensure_stream(SOURCE, holder="watcher-1")
        lease_manager._leases[ensured.lease_id].last_seen = time.time() - 999

        await lease_manager.ensure_stream(SOURCE, holder="watcher-1")
        await lease_manager._expire_stale_leases()

        assert ensured.lease_id in lease_manager._leases

    async def test_two_different_holders_still_get_two_leases(self, lease_manager):
        first = await lease_manager.ensure_stream(SOURCE, holder="watcher-1")
        second = await lease_manager.ensure_stream(SOURCE, holder="watcher-2")

        assert first.lease_id != second.lease_id
        assert len(lease_manager._leases) == 2

    async def test_anonymous_holders_are_never_merged(self, lease_manager):
        """Anonymous callers are indistinguishable, so collapsing them would let one
        caller's release free a path another is still reading."""
        first = await lease_manager.ensure_stream(SOURCE)
        second = await lease_manager.ensure_stream(SOURCE)

        assert first.lease_id != second.lease_id
        assert len(lease_manager._leases) == 2

    async def test_the_same_holder_on_two_sources_gets_two_leases(self, lease_manager):
        first = await lease_manager.ensure_stream(SOURCE, holder="watcher-1")
        second = await lease_manager.ensure_stream(OTHER_SOURCE, holder="watcher-1")

        assert first.lease_id != second.lease_id


class TestRestartGivesHoldersTimeToCheckIn:
    async def test_reloaded_leases_are_not_instantly_stale(self, lease_manager, monkeypatch):
        """A stored `last_seen` from before a long shutdown must not reap everything.

        Holders live in other processes and have had no chance to renew yet, so the
        clock restarts on load; a genuinely dead holder is reaped one TTL later.
        """
        monkeypatch.setattr("apps.relay_buffer_server.manager.LEASE_TTL_S", 300.0)
        _add_lease(lease_manager, "survivor", age_s=0)
        lease_manager._leases["survivor"].last_seen = time.time() - 99999
        lease_manager._save_leases()

        revived = Manager(media_mtx_api_client=MagicMock())
        revived._lease_file = lease_manager._lease_file
        revived._load_leases()

        assert "survivor" in revived._leases
        await revived._expire_stale_leases()
        assert "survivor" in revived._leases, "holders must get a full TTL to check in after a restart"


class TestRenewEndpoint:
    async def test_renew_reports_success_for_a_live_lease(self, http_client, manager):
        _add_lease(manager, "live-1", age_s=0)

        resp = await http_client.post("/api/streams/renew", data={"lease_id": "live-1"})

        assert resp.status_code == 200
        assert resp.json() == {"success": True, "renewed": True}

    async def test_renew_reports_failure_for_an_unknown_lease(self, http_client):
        resp = await http_client.post("/api/streams/renew", data={"lease_id": "nope"})

        assert resp.status_code == 200
        assert resp.json() == {"success": True, "renewed": False}
