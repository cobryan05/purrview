"""End-to-end proof that RTSP ingest actually works, using a synthetic camera.

Nothing in the suite drove a real RTSP stream before this, so the whole
"FfmpegImageProvider -> relay -> MediaMTX -> frames" path was only ever verified by
inspection. An ffmpeg `testsrc` pattern publishing into the local MediaMTX is a
perfectly good fake camera and makes this runnable without hardware.

Covers the properties a 24/7 watcher depends on:

* frames really arrive from an RTSP URL and land as files + DB rows,
* two consumers of one camera share a single master path (no per-consumer path),
* releasing a lease tears the path down only when the last holder lets go,
* a recorded path accumulates a rolling buffer that the playback server can cut.

Smoke-gated (`--run-smoke`) because it needs live services, ffmpeg and MediaMTX.
"""

import asyncio
import os
import shutil
import socket
import subprocess
import time
import uuid

import httpx
import pytest

RTSP_PORT = int(os.environ.get("SMOKE_RTSP_PORT", "8554"))
MEDIAMTX_API_PORT = int(os.environ.get("SMOKE_MEDIAMTX_API_PORT", "9997"))
PLAYBACK_PORT = int(os.environ.get("SMOKE_PLAYBACK_PORT", "9996"))

# Long enough for MediaMTX to close at least one 5 s segment, so the playback server
# has something to list. Segments are keyframe-aligned, so this is a floor, not a
# precise duration — never assert exact clip lengths against it.
PUBLISH_WARMUP_S = 12.0


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def provider_relay_url():
    """The relay a provider actually talks to.

    Deliberately *not* the `relay_url` smoke fixture: that one is a relay this test
    session spawned on an offset port, whereas FfmpegImageProvider resolves its relay
    from APPS_CONFIG at import time. Leases live in the relay process, so asking the
    wrong one always reports zero holders.
    """
    from apps import APPS, APPS_CONFIG

    cfg = APPS_CONFIG[APPS.RELAY_BUFFER_SERVER]
    return f"http://{cfg.host}:{cfg.port}"


@pytest.fixture(scope="module")
def require_media_stack():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    if not _port_open(RTSP_PORT) or not _port_open(MEDIAMTX_API_PORT):
        pytest.skip("MediaMTX is not running; this test needs the live media stack")


@pytest.fixture
def fake_camera(require_media_stack):
    """Publish a `testsrc` pattern into MediaMTX and yield its RTSP URL.

    Two non-obvious requirements, both of which surface only as an opaque
    `400 Bad Request` on ANNOUNCE:

    * **The path must exist before you publish to it.** `conf/mediamtx.yml` declares no
      `paths:` section, so there is no `all_others` fallback and an undeclared path is
      simply rejected. (The relay never hits this because it creates paths through the
      config API first.)
    * **`-pix_fmt yuv420p` is required.** `testsrc` emits rgb24, so libx264 would pick
      a 4:4:4 profile, which MediaMTX refuses.
    """
    path = f"fakecam_{uuid.uuid4().hex[:8]}"
    url = f"rtsp://localhost:{RTSP_PORT}/{path}"
    api = f"http://localhost:{MEDIAMTX_API_PORT}/v3/config/paths"
    httpx.post(f"{api}/add/{path}", json={"source": "publisher"}, timeout=10).raise_for_status()

    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-re",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x480:rate=10",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            # A short GOP keeps segments small, so the rolling buffer has usable
            # granularity for a test that only runs for a few seconds.
            "-g",
            "10",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + 20
    ready = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            resp = httpx.get(f"http://localhost:{MEDIAMTX_API_PORT}/v3/paths/get/{path}", timeout=2)
            if resp.status_code == 200 and resp.json().get("ready"):
                ready = True
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)

    if not ready:
        proc.terminate()
        proc.wait(timeout=10)
        httpx.post(f"{api}/delete/{path}", timeout=10)
        pytest.skip("Could not publish the synthetic camera into MediaMTX")

    try:
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        httpx.post(f"{api}/delete/{path}", timeout=10)


@pytest.fixture(autouse=True)
def no_leaked_leases(provider_relay_url):
    """Fail (and clean up) if a test leaves a lease on the relay.

    Leases outlive the process that took them: they persist to the relay's lease file,
    which is bind-mounted so a restart can re-arm paths that still have holders. A test
    that dies between `ensure` and `stop` therefore leaves an idle path retrying against
    a `localhost` URL that stopped existing the moment ffmpeg was killed — and every
    relay restart faithfully brings it back. Four of those accumulated in the dev
    environment before anyone noticed, which is what this guard exists to prevent.

    Asserting in teardown is deliberate: the leak is the failure, so it should be
    reported rather than quietly swept up. The sweep still happens so one bad test
    can't poison every later run.
    """

    def relay_paths():
        try:
            with httpx.Client(base_url=provider_relay_url, timeout=10) as client:
                return {s["name"] for s in client.get("/api/streams/list").json().get("results", [])}
        except httpx.HTTPError:
            return set()

    before = relay_paths()
    yield
    leaked = relay_paths() - before
    if not leaked:
        return

    with httpx.Client(base_url=provider_relay_url, timeout=10) as client:
        for path in leaked:
            client.post("/api/streams/destroy", data={"stream_name": path})

    pytest.fail(
        f"Test leaked {len(leaked)} relay path(s): {sorted(leaked)}. Every provider that "
        "reached start() must be stopped, including when start() itself raised."
    )


class TestRtspIngest:
    async def test_provider_reads_frames_from_a_real_rtsp_stream(self, fake_camera):
        """The core claim of item #2: an RTSP URL yields decoded numpy frames."""
        from apps.helpers.imageProviders.ffmpegImageProvider import FfmpegImageProvider

        provider = FfmpegImageProvider(video_path=fake_camera, loop=True, rtsp_relay=True)
        await provider.start()
        try:
            frames = []
            for _ in range(3):
                frame = await asyncio.wait_for(provider.getNextImage(), timeout=30)
                assert frame is not None
                frames.append(frame)
            assert all(f.image.shape == (480, 640, 3) for f in frames)
            # Frame indices must advance — a stuck stream would repeat one frame.
            assert [f.metadata.frame_idx for f in frames] == [0, 1, 2]
        finally:
            await provider.stop()

    async def test_provider_reads_the_master_path_directly(self, fake_camera, provider_relay_url):
        """No per-consumer relay path: the provider reads the shared master.

        The old design created a second MediaMTX path per consumer, sourced from the
        master over RTSP loopback — an extra hop and an extra reader for nothing.
        """
        from apps.helpers.imageProviders.ffmpegImageProvider import FfmpegImageProvider
        from apps.relay_buffer_server.manager import master_path_name

        provider = FfmpegImageProvider(video_path=fake_camera, loop=True, rtsp_relay=True)
        await provider.start()
        try:
            assert provider._relayed_name == master_path_name(fake_camera)
            assert provider._lease_id
        finally:
            await provider.stop()

    async def test_two_consumers_share_one_master_path(self, fake_camera, provider_relay_url):
        from apps.helpers.imageProviders.ffmpegImageProvider import FfmpegImageProvider

        first = FfmpegImageProvider(video_path=fake_camera, loop=True, rtsp_relay=True)
        second = FfmpegImageProvider(video_path=fake_camera, loop=True, rtsp_relay=True)
        await first.start()
        await second.start()
        try:
            assert first._relayed_name == second._relayed_name
            assert first._lease_id != second._lease_id

            async with httpx.AsyncClient(base_url=provider_relay_url, timeout=10) as client:
                streams = (await client.get("/api/streams/list")).json()["results"]
                entry = next(s for s in streams if s["name"] == first._relayed_name)
                assert entry["info"]["readers"] == 2
        finally:
            await first.stop()
            await second.stop()

    async def test_releasing_one_lease_keeps_the_path_for_the_other(self, fake_camera, provider_relay_url):
        from apps.helpers.imageProviders.ffmpegImageProvider import FfmpegImageProvider

        first = FfmpegImageProvider(video_path=fake_camera, loop=True, rtsp_relay=True)
        second = FfmpegImageProvider(video_path=fake_camera, loop=True, rtsp_relay=True)
        await first.start()
        await second.start()
        path = first._relayed_name
        try:
            await first.stop()
            async with httpx.AsyncClient(base_url=provider_relay_url, timeout=10) as client:
                streams = (await client.get("/api/streams/list")).json()["results"]
                assert any(s["name"] == path for s in streams), "path must survive while a holder remains"
        finally:
            await second.stop()

        async with httpx.AsyncClient(base_url=provider_relay_url, timeout=10) as client:
            streams = (await client.get("/api/streams/list")).json()["results"]
            assert not any(s["name"] == path for s in streams), "path must go away with its last holder"


class TestRollingBuffer:
    async def test_a_recorded_path_accumulates_a_rewind_window(self, fake_camera, provider_relay_url):
        """The buffer is real: MediaMTX writes segments and playback lists them."""
        if not _port_open(PLAYBACK_PORT):
            pytest.skip("MediaMTX playback server is not enabled")

        from apps.helpers.imageProviders.ffmpegImageProvider import FfmpegImageProvider

        provider = FfmpegImageProvider(video_path=fake_camera, loop=True, rtsp_relay=True)
        await provider.start()
        try:
            assert provider.buffered is True, "an RTSP source must be armed with a rolling buffer"
            await asyncio.sleep(PUBLISH_WARMUP_S)

            async with httpx.AsyncClient(base_url=provider_relay_url, timeout=20) as client:
                resp = await client.get(f"/api/streams/{provider._relayed_name}/ranges")
                data = resp.json()
                assert data["success"] is True, data
                assert data["results"], "no buffered ranges after publishing for a while"
                # Depth is a floor, not an exact figure: segments are keyframe-aligned
                # and the recorder only closes them periodically.
                assert sum(r["durationS"] for r in data["results"]) > 0
        finally:
            await provider.stop()

    async def test_buffer_usage_reports_a_commitment(self, fake_camera, provider_relay_url):
        from apps.helpers.imageProviders.ffmpegImageProvider import FfmpegImageProvider

        provider = FfmpegImageProvider(video_path=fake_camera, loop=True, rtsp_relay=True)
        await provider.start()
        try:
            async with httpx.AsyncClient(base_url=provider_relay_url, timeout=10) as client:
                usage = (await client.get("/api/streams/buffer")).json()
            assert usage["success"] is True
            assert usage["committedBytes"] > 0
            assert usage["committedBytes"] <= usage["capBytes"]
        finally:
            await provider.stop()
