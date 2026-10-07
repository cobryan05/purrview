"""A relay lease must not outlive the `start()` that took it.

`FfmpegImageProvider.start()` takes a lease on a MediaMTX master path *before* it
knows whether the path will come up. If anything after that point raises, the caller
never gets a provider object, so it can never call `stop()` — and `stop()` is the only
thing that releases the lease. The lease then persists to `/data/relay/leases.json`
(bind-mounted, deliberately, so a relay restart re-arms paths that still have holders)
and gets faithfully re-armed on every restart, forever.

That is not hypothetical: four such leases accumulated against dead `localhost` paths
and survived weeks of restarts, each one an idle `rtspSource` retry loop with a phantom
holder. Any camera that fails to come up leaks one the same way.

These tests drive `start()` with a stubbed relay client, so they need neither MediaMTX
nor ffmpeg.
"""

import asyncio
import types

import pytest

from apps.helpers.imageProviders import ffmpegImageProvider as ffmpeg_module
from apps.helpers.imageProviders.ffmpegImageProvider import FfmpegImageProvider
from tests.waits import await_condition

SOURCE_URL = "rtsp://camera.invalid:8554/stream"


class FakeStreamsApi:
    """Records what the provider asked the relay to do."""

    def __init__(self, api_client):
        self.api_client = api_client
        self.state = api_client.state

    def ensure_stream(self, source_url, holder):
        self.state["ensured"].append((source_url, holder))
        return {
            "success": True,
            "leaseId": "lease-1",
            "path": "relay_master_deadbeef",
            "recording": True,
            "retentionS": 45.0,
            "ingestTier": "native",
            "rtspUrl": "rtsp://localhost:8554/relay_master_deadbeef",
        }

    def release_stream(self, lease_id):
        self.state["released"].append(lease_id)
        return {"success": True, "released": True}

    def renew_stream(self, lease_id):
        self.state["renewed"].append(lease_id)
        return {"success": True, "renewed": True}

    def destroy_stream(self, stream_name):
        self.state["destroyed"].append(stream_name)
        return {"success": True}

    def list_streams(self):
        if self.state.get("list_raises"):
            raise RuntimeError("relay unreachable")
        return {"success": True, "results": self.state["listed"]}


class FakeApiClient:
    def __init__(self, configuration=None):
        self.state = {
            "ensured": [],
            "released": [],
            "destroyed": [],
            "renewed": [],
            # What the relay reports as armed, for the start-failure message.
            "listed": [],
            "list_raises": False,
        }


@pytest.fixture
def relay_state(monkeypatch):
    """Swap the generated relay client for a recorder and yield its call log."""
    fake_module = types.SimpleNamespace(
        Configuration=lambda url: url,
        ApiClient=FakeApiClient,
        StreamsApi=FakeStreamsApi,
    )
    monkeypatch.setattr(ffmpeg_module, "stream_client", fake_module)

    provider = FfmpegImageProvider(video_path=SOURCE_URL, rtsp_relay=True)
    yield provider, provider._rtsp_relay_api.state


class TestLeaseIsReleasedWhenStartFails:
    async def test_a_path_that_never_becomes_ready_does_not_strand_its_lease(self, relay_state, monkeypatch):
        """The exact shape of the four leaked leases: ensure succeeded, readiness never came."""
        provider, state = relay_state

        async def never_ready(api, stream_name, timeout, poll_interval=0.5):
            return False

        monkeypatch.setattr(FfmpegImageProvider, "_wait_for_stream", staticmethod(never_ready))

        with pytest.raises(TimeoutError):
            await provider.start()

        assert state["ensured"], "the test is meaningless unless a lease was actually taken"
        assert state["released"] == ["lease-1"], "a failed start must hand its lease back"

    async def test_a_failing_ffmpeg_reader_does_not_strand_its_lease(self, relay_state, monkeypatch):
        """The path came up, but the local reader blew up — the lease is still ours to return."""
        provider, state = relay_state

        async def ready(api, stream_name, timeout, poll_interval=0.5):
            return True

        monkeypatch.setattr(FfmpegImageProvider, "_wait_for_stream", staticmethod(ready))

        class ExplodingStreamer:
            def __init__(self, url, output_args=None):
                pass

            async def astart(self):
                raise RuntimeError("ffmpeg died on startup")

        monkeypatch.setattr(ffmpeg_module, "FFmpegStreamerIn", ExplodingStreamer)

        with pytest.raises(RuntimeError, match="ffmpeg died"):
            await provider.start()

        assert state["released"] == ["lease-1"]

    async def test_a_relay_that_refuses_to_arm_leaves_nothing_behind(self, relay_state, monkeypatch):
        """No lease was issued, so there is nothing to release — and nothing to destroy."""
        provider, state = relay_state

        def refuse(self, source_url, holder):
            state["ensured"].append((source_url, holder))
            return {"success": False, "error": "no such camera"}

        monkeypatch.setattr(FakeStreamsApi, "ensure_stream", refuse)

        with pytest.raises(RuntimeError, match="Relay refused"):
            await provider.start()

        assert state["released"] == []
        assert state["destroyed"] == []

    async def test_cleanup_failure_does_not_mask_the_original_error(self, relay_state, monkeypatch):
        """A relay that is itself down must not turn a TimeoutError into a confusing one."""
        provider, state = relay_state

        async def never_ready(api, stream_name, timeout, poll_interval=0.5):
            return False

        def release_explodes(self, lease_id):
            raise ConnectionError("relay unreachable")

        monkeypatch.setattr(FfmpegImageProvider, "_wait_for_stream", staticmethod(never_ready))
        monkeypatch.setattr(FakeStreamsApi, "release_stream", release_explodes)

        # The caller must still see why the *stream* failed, not why the cleanup did.
        with pytest.raises(TimeoutError):
            await provider.start()


class TestSuccessfulStartIsUnaffected:
    async def test_a_good_start_keeps_its_lease(self, relay_state, monkeypatch):
        provider, state = relay_state

        async def ready(api, stream_name, timeout, poll_interval=0.5):
            return True

        class QuietStreamer:
            def __init__(self, url, output_args=None):
                self.url = url

            async def astart(self):
                return None

        monkeypatch.setattr(FfmpegImageProvider, "_wait_for_stream", staticmethod(ready))
        monkeypatch.setattr(ffmpeg_module, "FFmpegStreamerIn", QuietStreamer)

        await provider.start()

        assert provider._lease_id == "lease-1"
        assert state["released"] == [], "a healthy stream must hold on to its lease"
        assert provider.buffered is True
        # Read off the same ensure response as `buffered`, and dropped for exactly as
        # long as nobody read it: the watcher reports "0s rewind" for a camera with a
        # 45 s buffer, and `plan_window`'s pre-roll clamp is guarded on `> 0` so it
        # never runs. Both failures are silent.
        assert provider.retention_s == 45.0
        assert provider.relay_path == "relay_master_deadbeef"

    async def test_there_is_no_relay_path_before_the_lease(self, relay_state):
        """`__init__` fills `_relayed_name` with a `provider-<uuid>` placeholder that
        `ensure_stream` replaces. Handing that out gets `path '<name>' is not configured`
        from the playback server, which reads as a camera with no footage rather than as
        a clip cut before its path existed.
        """
        provider, _state = relay_state

        assert provider._relayed_name.startswith("provider-")
        assert provider.relay_path == ""

    async def test_a_relay_that_reports_no_retention_leaves_it_at_zero(self, relay_state, monkeypatch):
        """An older relay omits the field; zero is the honest answer, not a made-up 45."""

        def ensure(self, source_url, holder):
            return {
                "success": True,
                "leaseId": "lease-1",
                "path": "relay_master_deadbeef",
                "recording": False,
                "ingestTier": "none",
                "rtspUrl": "rtsp://localhost:8554/relay_master_deadbeef",
            }

        async def ready(api, stream_name, timeout, poll_interval=0.5):
            return True

        class QuietStreamer:
            def __init__(self, url, output_args=None):
                self.url = url

            async def astart(self):
                return None

        provider, _state = relay_state
        monkeypatch.setattr(FakeStreamsApi, "ensure_stream", ensure)
        monkeypatch.setattr(FfmpegImageProvider, "_wait_for_stream", staticmethod(ready))
        monkeypatch.setattr(ffmpeg_module, "FFmpegStreamerIn", QuietStreamer)

        await provider.start()
        assert provider.retention_s == 0.0

    async def test_the_lease_is_released_exactly_once_across_start_failure_and_stop(self, relay_state, monkeypatch):
        """Defensive `stop()` in a caller's `finally` must not double-release."""
        provider, state = relay_state

        async def never_ready(api, stream_name, timeout, poll_interval=0.5):
            return False

        monkeypatch.setattr(FfmpegImageProvider, "_wait_for_stream", staticmethod(never_ready))

        with pytest.raises(TimeoutError):
            await provider.start()
        await provider.stop()

        assert state["released"] == ["lease-1"]
        assert state["destroyed"] == [], "a released lease must not also destroy the shared path"


class TestHeartbeat:
    """The relay reaps leases that stop being renewed, so a live provider must beat."""

    @staticmethod
    def _started(monkeypatch):
        async def ready(api, stream_name, timeout, poll_interval=0.5):
            return True

        class QuietStreamer:
            def __init__(self, url, output_args=None):
                pass

            async def astart(self):
                return None

            async def astop(self):
                return None

        monkeypatch.setattr(FfmpegImageProvider, "_wait_for_stream", staticmethod(ready))
        monkeypatch.setattr(ffmpeg_module, "FFmpegStreamerIn", QuietStreamer)

    async def test_a_running_provider_renews_its_lease(self, relay_state, monkeypatch):
        provider, state = relay_state
        self._started(monkeypatch)

        beats = asyncio.Event()

        async def instant_sleep(_seconds):
            beats.set()
            await asyncio.sleep(0)

        monkeypatch.setattr(FfmpegImageProvider, "_sleep", staticmethod(instant_sleep))

        await provider.start()
        await asyncio.wait_for(beats.wait(), timeout=5)
        # The beat having been *scheduled* is not the beat having *landed*, so poll for
        # the renewal itself rather than sleeping 50ms and hoping it arrived.
        await await_condition(
            lambda: state["renewed"],
            "a live provider must tell the relay it is still there",
        )
        await provider.stop()

        assert state["renewed"], "a live provider must tell the relay it is still there"
        assert set(state["renewed"]) == {"lease-1"}

    async def test_stop_ends_the_heartbeat(self, relay_state, monkeypatch):
        """A beat in flight could otherwise resurrect the lease we just handed back."""
        provider, state = relay_state
        self._started(monkeypatch)

        async def instant_sleep(_seconds):
            await asyncio.sleep(0)

        monkeypatch.setattr(FfmpegImageProvider, "_sleep", staticmethod(instant_sleep))

        await provider.start()
        # Wait for the heartbeat to actually be beating before stopping it — stopping a
        # heartbeat that never started proves nothing about stop().
        await await_condition(
            lambda: state["renewed"],
            "the heartbeat never beat, so stop() had nothing to end",
        )
        await provider.stop()

        assert provider._renew_task is None
        # This one stays a sleep: it asserts that *nothing further* arrives, and there
        # is no way to poll for a non-event. Its failure mode is also the safe one — a
        # slow box makes a late beat more likely to be caught, not less.
        settled = len(state["renewed"])
        await asyncio.sleep(0.05)
        assert len(state["renewed"]) == settled, "no beats may land after stop()"
        assert state["released"] == ["lease-1"]

    async def test_a_failed_start_leaves_no_heartbeat_running(self, relay_state, monkeypatch):
        """A heartbeat here would keep alive a lease whose provider never ran."""
        provider, state = relay_state

        async def never_ready(api, stream_name, timeout, poll_interval=0.5):
            return False

        monkeypatch.setattr(FfmpegImageProvider, "_wait_for_stream", staticmethod(never_ready))

        with pytest.raises(TimeoutError):
            await provider.start()

        assert provider._renew_task is None
        await asyncio.sleep(0.05)
        assert state["renewed"] == []

    async def test_a_transient_relay_failure_does_not_kill_the_heartbeat(self, relay_state, monkeypatch):
        """Giving up on one failed beat would guarantee the reap that retrying avoids."""
        provider, state = relay_state
        self._started(monkeypatch)

        calls = {"n": 0}

        def flaky_renew(self, lease_id):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("relay briefly unreachable")
            state["renewed"].append(lease_id)
            return {"success": True, "renewed": True}

        async def instant_sleep(_seconds):
            await asyncio.sleep(0)

        monkeypatch.setattr(FakeStreamsApi, "renew_stream", flaky_renew)
        monkeypatch.setattr(FfmpegImageProvider, "_sleep", staticmethod(instant_sleep))

        await provider.start()
        # The retry is the thing under test, so wait for the second call rather than
        # for 50ms — on a loaded box the first beat alone can take longer than that.
        await await_condition(
            lambda: calls["n"] > 1,
            lambda: f"the heartbeat must retry after a failed beat (saw {calls['n']} calls)",
        )
        running = provider._renew_task is not None and not provider._renew_task.done()
        await provider.stop()

        assert calls["n"] > 1, "the heartbeat must retry after a failed beat"
        assert running, "a transient failure must not end the heartbeat"


class TestConcurrentStartsDoNotStrandLeases:
    async def test_two_failing_starts_release_both_leases(self, relay_state, monkeypatch):
        """Two watchers on one camera hold two leases; a failure must return both."""
        provider, state = relay_state

        leases = iter(["lease-1", "lease-2"])

        def ensure(self, source_url, holder):
            state["ensured"].append((source_url, holder))
            return {
                "success": True,
                "leaseId": next(leases),
                "path": "relay_master_deadbeef",
                "recording": True,
                "ingestTier": "native",
            }

        async def never_ready(api, stream_name, timeout, poll_interval=0.5):
            return False

        monkeypatch.setattr(FakeStreamsApi, "ensure_stream", ensure)
        monkeypatch.setattr(FfmpegImageProvider, "_wait_for_stream", staticmethod(never_ready))

        second = FfmpegImageProvider(video_path=SOURCE_URL, rtsp_relay=True)
        second._rtsp_relay_api = provider._rtsp_relay_api

        results = await asyncio.gather(provider.start(), second.start(), return_exceptions=True)

        assert all(isinstance(r, TimeoutError) for r in results)
        assert sorted(state["released"]) == ["lease-1", "lease-2"]


class TestWhyLiveVideoDidNotStart:
    """The message a failed `start()` leaves on the camera's live view.

    One message used to cover two unrelated faults and named only the path hash, so a
    relay that had restarted and lost its paths read exactly like a camera that had
    stopped answering — and the relay restart is the one nobody would guess, because
    the camera is fine and says so.
    """

    async def _failure_message(self, provider, monkeypatch):
        async def never_ready(api, stream_name, timeout, poll_interval=0.5):
            return False

        monkeypatch.setattr(FfmpegImageProvider, "_wait_for_stream", staticmethod(never_ready))
        with pytest.raises(TimeoutError) as excinfo:
            await provider.start()
        return str(excinfo.value)

    async def test_a_path_the_relay_never_listed_is_reported_as_lost(self, relay_state, monkeypatch):
        provider, state = relay_state
        state["listed"] = []

        message = await self._failure_message(provider, monkeypatch)

        assert "no path" in message
        assert "relay restart" in message
        assert SOURCE_URL in message, "the message must name the camera, not only the path hash"

    async def test_a_listed_path_that_never_delivers_blames_the_camera(self, relay_state, monkeypatch):
        """The relay has it armed, so the fault is upstream of the relay."""
        provider, state = relay_state
        state["listed"] = [{"name": "relay_master_deadbeef"}]

        message = await self._failure_message(provider, monkeypatch)

        assert "not answering" in message
        assert "relay restart" not in message, "an armed path must not be reported as lost"

    async def test_a_relay_that_cannot_be_asked_says_so(self, relay_state, monkeypatch):
        """Guessing between the two is worse than admitting the relay did not answer."""
        provider, state = relay_state
        state["list_raises"] = True

        message = await self._failure_message(provider, monkeypatch)

        assert "could not be asked" in message
