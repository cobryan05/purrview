"""FFMpeg-backed Image Provider class"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Optional, Tuple

import numpy as np

from apps import APPS, APPS_CONFIG
from apps.helpers.apiCall import bind_default_timeout
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.streams.ffmpegStreamerIn import FFmpegStreamerIn
from apps.helpers.streams.ingest import resolve_ingest_tier

from .imageProvider import (
    ImageProvider,
    ImageProviderMetadata,
    ImageProviderType,
    ImageWithProviderMetadata,
)
from .Registry import register_image_provider

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

if os.getenv("GENERATING_OPENAPI_CLIENTS") != "1":
    import stream_client

# How often a live provider tells the relay it is still there. Must stay comfortably
# under the relay's RELAY_LEASE_TTL_S (default 300s) so a slow event loop or a brief
# relay restart costs a few missed beats rather than the lease.
LEASE_RENEW_INTERVAL_S = 60.0

STREAM_START_TIMEOUT = 10.0
STREAM_READ_TIMEOUT = 10.0
FFMPEG_START_TIMEOUT = 30.0


@register_image_provider()
class FfmpegImageProvider(ImageProvider):
    # Bound on consecutive stream-restart attempts inside getNextImage() so a stream that
    # stops yielding data (EOF, dead source, etc.) can't spin astop()/astart() forever.
    _MAX_STREAM_RESTARTS = 3
    _RESTART_BACKOFF = 0.5

    # How long a freshly-started reader is allowed to be silent before it counts as
    # broken rather than still connecting.
    #
    # Generous because connecting genuinely is slow: an RTSP camera behind a relay can
    # take two or three minutes to come up from cold, and MediaMTX retries the source
    # in the background the whole time. Tearing the reader down before that finishes
    # does not hurry it — it resets the attempt, and a short grace produces a source
    # that reconnects forever and never once connects.
    CONNECT_GRACE_S: float = 300.0

    # Injectable seams for tests (patch on the class/instance instead of the global
    # asyncio/time modules, which are shared process-wide and unsafe to monkeypatch
    # while other tests may be running concurrently).
    _sleep = staticmethod(asyncio.sleep)
    _now = staticmethod(time.monotonic)

    @classmethod
    def provider_type(cls) -> ImageProviderType:
        return ImageProviderType.LIVE_STREAM

    @classmethod
    def ingest_tier(cls) -> str:
        """Defer to the URL: this provider covers everything ffmpeg can read, which
        spans both what MediaMTX pulls natively and what needs an ffmpeg ingest."""
        return "native"

    def __init__(
        self,
        video_path: str,
        loop: bool = True,
        rtsp_relay: bool = True,
        min_capture_interval: float = 0.0,
    ):
        output_args = {
            "format": "rawvideo",
            # bgr24, NOT rgb24 — the whole app treats a frame as BGR, and nothing
            # downstream can detect the difference. cv2.imencode/imwrite read channel 0
            # as blue, so an RGB frame is written to disk and sent to inference with red
            # and blue swapped; the inference provider then runs cvtColor(BGR2RGB) and
            # hands the model the swapped image, while the models were trained on
            # correctly-ordered ones. This asked for rgb24 from the provider's first
            # commit, and produced no error anywhere — only worse detections.
            #
            # Free to fix: ffmpeg's swscale converts the source's yuv to one packed
            # 24-bit layout or the other at identical cost, so this adds nothing to the
            # hot path and removes a conversion that would otherwise be needed in Python.
            "pix_fmt": "bgr24",
            "codec": "rawvideo",
        }  # decode raw frames
        self._video_path: str = video_path.strip()
        self._loop: bool = loop
        self._frame_size: Optional[Tuple[int, int]] = None
        self._buffer: bytes = b""
        self._rtsp_relay_api: Optional[stream_client.ApiClient] = None
        self._relayed_name: str = ""
        self._stream: Optional[FFmpegStreamerIn] = None
        #: Where the reader reads from, learned in start(). reconnect() rebuilds the
        #: reader against it without re-running the lease handshake.
        self._stream_url: str = ""
        #: When the current reader was spawned, for the connect grace. Reset on every
        #: rebuild, so each attempt gets a full window rather than inheriting the
        #: elapsed time of the one before it.
        self._stream_started_at: Optional[float] = None
        self._video_path: str = video_path.strip()
        self._ffmpeg_args: dict = output_args
        self._min_capture_interval: float = min_capture_interval
        self.min_capture_interval: float = min_capture_interval
        #: Smallest interval, in seconds, between frames ffmpeg is asked to emit.
        #: 0 means "whatever the source sends".
        #:
        #: Deliberately NOT `min_capture_interval`, even though it starts out equal to
        #: it. That one *also* makes getNextImage() sleep to pace its caller, and a
        #: caller that services several sources sequentially from one loop (the
        #: watcher does) pays every source's sleep on all the others. This attribute
        #: only ever tells ffmpeg to send less, which costs the caller nothing.
        self._decimate_interval: float = min_capture_interval
        self._next_capture_time: float = 0.0
        self._frame_idx: int = 0
        self._restart_count: int = 0
        self._lease_id: str = ""
        self._renew_task: Optional[asyncio.Task] = None
        # Set once the lease has been handed back, so a caller's defensive `stop()`
        # after a failed `start()` doesn't fall through to destroying the shared path
        # that other watchers may still be reading.
        self._lease_returned: bool = False
        # Relay whenever the resolved ingest tier can actually be got into a MediaMTX
        # path — not just for `rtsp:`. Recording is a property of the path, so this one
        # condition is what makes MJPEG cameras, V4L2 devices and other ffmpeg-readable
        # sources buffer-capable too.
        self._ingest_plan = resolve_ingest_tier(type(self), {"video_path": self._video_path})
        # What the UI's buffer badge reads. Set from the resolver up front and corrected
        # from the relay's answer in start(), because only the relay knows whether the
        # path is actually recording.
        self.ingest_tier_name: str = self._ingest_plan.tier.value
        self.buffered: bool = False
        #: Seconds of rewind the relay actually armed. Read from the same ensure
        #: response as `buffered` — the relay is the only thing that knows, and a value
        #: inferred here would let the badge and the clip's pre-roll disagree.
        self.retention_s: float = 0.0
        if rtsp_relay and self._ingest_plan.buffered:
            relay_host, relay_port = (
                APPS_CONFIG[APPS.STREAM_SERVER].host,
                APPS_CONFIG[APPS.STREAM_SERVER].port,
            )
            config = stream_client.Configuration(f"http://{relay_host}:{relay_port}")
            self._rtsp_relay_api = bind_default_timeout(stream_client.ApiClient(config))
            self._relayed_name = f"provider-{uuid.uuid4()}"

    def __repr__(self):
        return f"FfmpegImageProvider [{self._stream}]"

    @property
    def relay_path(self) -> str:
        """The MediaMTX path a clip can be cut from, or "" when there is nothing to cut.

        Gated on the lease rather than on the name: `__init__` fills `_relayed_name`
        with a `provider-<uuid>` placeholder that `ensure_stream` replaces with the
        shared master path, and the playback server answers
        `path '<name>' is not configured` for the placeholder.
        """
        return self._relayed_name if self._lease_id else ""

    async def start(self) -> None:
        if self._rtsp_relay_api:
            api: stream_client.StreamsApi = stream_client.StreamsApi(self._rtsp_relay_api)
            # Take a lease on the shared master path and read it directly. The old code
            # created a second, per-consumer path sourced from the master over RTSP
            # loopback — an extra hop and an extra MediaMTX reader per consumer, for
            # nothing, and it got worse the moment two tasks watched one camera.
            ensured = await asyncio.to_thread(
                api.ensure_stream,
                self._video_path,
                self._relayed_name,
            )
            if not ensured.get("success"):
                raise RuntimeError(f"Relay refused to arm {self._video_path}: {ensured.get('error')}")
            self._lease_id = ensured["leaseId"]
            self._relayed_name = ensured["path"]
            self.buffered = bool(ensured.get("recording"))
            self.retention_s = float(ensured.get("retentionS") or 0.0)
            self.ingest_tier_name = ensured.get("ingestTier", self._ingest_plan.tier.value)

            # From here on the lease is ours, and only `stop()` gives it back. A caller
            # whose `start()` raises never receives the provider, so it can never call
            # `stop()` — the lease would outlive the process, persist to the lease file
            # and get re-armed on every relay restart. Hand it back ourselves.
            try:
                if not await self._wait_for_stream(api, self._relayed_name, STREAM_START_TIMEOUT):
                    raise TimeoutError(f"Timed out waiting for '{self._relayed_name}' to become ready on the relay")
                stream_url = ensured.get("rtspUrl") or (
                    f"rtsp://{APPS_CONFIG[APPS.MEDIA_MTX_RTSP].host}:"
                    f"{APPS_CONFIG[APPS.MEDIA_MTX_RTSP].port}/{self._relayed_name}"
                )
                # Kept so reconnect() can rebuild the reader without re-running the
                # whole lease handshake to rediscover where to read from.
                self._stream_url = stream_url
                self._stream = FFmpegStreamerIn(stream_url, output_args=self._spawn_args())

                self._stream_started_at = self._now()
                await self._stream.astart()
            except BaseException:
                await self._release_lease_quietly()
                raise
            # Only once the stream is genuinely up: a heartbeat started earlier would
            # keep a lease alive on behalf of a provider that never actually ran.
            self._renew_task = asyncio.get_running_loop().create_task(self._renew_lease_forever())
            return

        self._stream_url = self._video_path
        self._stream = FFmpegStreamerIn(self._video_path, output_args=self._spawn_args())

        self._stream_started_at = self._now()
        await self._stream.astart()

    async def reconnect(self) -> None:
        """Rebuild the ffmpeg reader, keeping the relay lease.

        A camera going down does not take the relay path with it: MediaMTX retries its
        source forever by design, and our lease is what keeps the path armed and
        recording while the camera is away. So the thing that has actually broken is
        our reader off that path, and that is all this rebuilds.

        The inherited stop()/start() would be wrong here twice over — it would release
        and re-acquire a lease that other watchers on the same camera share the
        refcount of, and it would drop the rewind buffer's continuity every time a
        flapping camera bounced.

        Returns as soon as ffmpeg has been spawned; it does NOT wait for the stream to
        come up. Whether it did is `is_ready()`, asked on a later pass, because a
        source that is genuinely down would otherwise block this call for the full
        metadata timeout — and the caller servicing several cameras from one loop pays
        that stall on all of them.

        A reader still inside its connect grace is left strictly alone. Rebuilding one
        that is mid-handshake does not hurry it up, it restarts it — and a camera that
        takes minutes to come up would then be reconnected forever without ever once
        connecting.
        """
        if self.is_connecting():
            return

        if self._stream is not None:
            stream, self._stream = self._stream, None
            await stream.astop()

        if not self._stream_url:
            # start() never learned where to read from, which means it failed before
            # getting that far — a slow camera missing the relay-readiness window is
            # the ordinary way. A full start() is the only thing that can recover it,
            # and it is safe to repeat: a failed start already handed its lease back.
            await self.start()
            return

        self._stream = FFmpegStreamerIn(self._stream_url, output_args=self._spawn_args())

        self._stream_started_at = self._now()
        await self._stream.astart()
        self._buffer = b""
        # Re-read on the new stream rather than trusting the old geometry: a camera can
        # come back at a different resolution, and a stale frame size silently
        # misaligns every subsequent read off the rawvideo pipe.
        self._frame_size = None
        # The inner per-call restart budget is spent; this attempt is the caller's
        # retry, so it starts with a full one.
        self._restart_count = 0

    def is_connecting(self) -> bool:
        """True while a reader is up but hasn't produced geometry yet, within grace.

        The distinction the caller needs: not-ready means either "coming up" or
        "broken", and those want opposite treatment — patience versus a rebuild.
        Without it a slow camera is indistinguishable from a dead one, and gets
        restarted every retry until someone notices it never works.
        """
        if self._stream is None or self._stream_started_at is None:
            return False
        if self._stream.metadata_ready():
            return False
        return (self._now() - self._stream_started_at) < self.CONNECT_GRACE_S

    def is_ready(self) -> bool:
        """True once ffmpeg has reported geometry — i.e. the source is actually there."""
        return self._stream is not None and self._stream.metadata_ready()

    def set_source_fps_ceiling(self, fps: float) -> None:
        """Ask ffmpeg for at most *fps* frames per second, dropped at the source.

        The point is the pipe, not the CPU. Output here is rawvideo, so one 1280x720
        frame is 2.7 MB and a 25 fps camera is ~69 MB/s down a 64 KiB stdout pipe. A
        caller that only analyses a few frames a second still has to read every byte of
        the rest off that pipe just to throw it away, and when it cannot keep up the
        pipe fills, ffmpeg blocks in write(), stops reading its RTSP input, and MediaMTX
        drops the stalled session. That failure reads as a dead camera while the logs
        show the stream starting cleanly every time. Dropping frames before they are
        written is the only place the saving is real.

        Takes effect on the next spawn, so set it before start(). reconnect() rebuilds
        against it too.
        """
        self._decimate_interval = 1.0 / fps if fps > 0 else 0.0

    def _spawn_args(self) -> dict:
        """Output args for a new reader, including any source-side decimation.

        Built per spawn rather than frozen in __init__ so that a ceiling set after
        construction is honoured, and so reconnect() rebuilds against the current one.
        """
        args = dict(self._ffmpeg_args)
        if self._decimate_interval <= 0:
            return args
        # `select`, not `fps`: the fps filter pads as well as drops, so a source slower
        # than the ceiling — an event-only feed, a camera that only sends during motion
        # — would have its frames duplicated up to the target rate, which is the
        # opposite of the point. `select` only ever drops, so an intermittent source
        # passes through untouched and only a fast one is thinned.
        #
        # The comma is escaped for ffmpeg's filtergraph parser, which would otherwise
        # read it as a filter separator. No shell is involved, so the backslash is a
        # literal argv byte — which is exactly what the parser wants.
        args["vf"] = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{self._decimate_interval})'"
        # Without this, a CFR rawvideo output re-pads the decimated stream back up to
        # the input rate and undoes the whole thing. `-vsync 0` rather than
        # `-fps_mode passthrough` because this image ships ffmpeg 4.4, which predates
        # fps_mode.
        args["vsync"] = "0"
        return args

    async def _renew_lease_forever(self) -> None:
        """Heartbeat the relay for as long as this provider is reading.

        The relay cannot otherwise tell a live watcher from a crashed one — MediaMTX
        retries a dead camera forever by design, so an idle path proves nothing about
        its holder. Silence here is what marks this provider as gone.
        """
        while True:
            await self._sleep(LEASE_RENEW_INTERVAL_S)
            if not self._lease_id:
                return
            try:
                api: stream_client.StreamsApi = stream_client.StreamsApi(self._rtsp_relay_api)
                result = await asyncio.to_thread(api.renew_stream, self._lease_id)
                if not result.get("renewed"):
                    # The relay has forgotten us: reaped, or its lease file was lost.
                    # The path we are reading may already be gone, so say so loudly
                    # rather than heartbeating a lease that no longer exists.
                    logger.warning(
                        f"Relay does not recognise lease {self._lease_id} for {self._relayed_name}; "
                        "the path may have been torn down"
                    )
            except Exception as e:
                # A transient relay outage must not kill the heartbeat — the TTL has
                # room for several missed beats, and giving up here would guarantee
                # the reap that retrying avoids.
                logger.warning(f"Lease renewal failed for {self._relayed_name}: {e}")

    async def _release_lease_quietly(self) -> None:
        """Hand the lease back without letting the cleanup replace the real failure.

        Releases rather than destroys: another watcher may be reading the same camera,
        and the relay's refcount is what decides whether the path goes away.
        """
        if not self._lease_id:
            return
        lease_id, self._lease_id = self._lease_id, ""
        self._lease_returned = True
        try:
            api: stream_client.StreamsApi = stream_client.StreamsApi(self._rtsp_relay_api)
            await asyncio.to_thread(api.release_stream, lease_id)
        except Exception as e:
            # Swallowed on purpose: the caller needs to see why the *stream* failed,
            # not why the tidy-up did. A relay that is down here will have dropped its
            # in-memory leases anyway.
            logger.error(f"Failed to release relay lease {lease_id} for {self._relayed_name}: {e}")

    @staticmethod
    async def _wait_for_stream(api, stream_name: str, timeout: float, poll_interval: float = 0.5) -> bool:
        """Wait for the relay path to be ready to read.

        Keys on the path's `ready` flag rather than `bytes_received > 0`: the latter is
        a snapshot counter that reads 0 for a stream which is connected but momentarily
        idle, so it produced spurious start-up timeouts on low-bitrate cameras.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            streams = await asyncio.to_thread(api.list_streams)
            for stream in streams.get("results", []):
                if stream.get("name") != stream_name:
                    continue
                info = stream.get("info") or {}
                mtx_path = info.get("mtx_path") or {}
                if info.get("active") or mtx_path.get("ready") or (mtx_path.get("bytes_received") or 0) > 0:
                    return True
                break
            await asyncio.sleep(poll_interval)
        return False

    async def stop(self) -> None:
        # Before releasing: a heartbeat still in flight could otherwise resurrect the
        # lease we are about to hand back.
        if self._renew_task is not None:
            self._renew_task.cancel()
            try:
                await self._renew_task
            except (asyncio.CancelledError, Exception):
                pass
            self._renew_task = None
        if self._stream is not None:
            await self._stream.astop()
        if self._rtsp_relay_api:
            if self._lease_id:
                # Release the lease, not the path: another watcher may still be
                # reading this camera, and the relay's refcount decides.
                await self._release_lease_quietly()
            elif not self._lease_returned:
                # No lease was ever issued, so there is no refcount to decrement and the
                # path is ours alone to tear down. Skipped once the lease *has* been
                # returned — a failed `start()` already did the right thing, and
                # destroying here would yank the path out from under other holders.
                try:
                    api: stream_client.StreamsApi = stream_client.StreamsApi(self._rtsp_relay_api)
                    await asyncio.to_thread(api.destroy_stream, self._relayed_name)
                except Exception as e:
                    logger.error(f"Failed to destroy relay stream {self._relayed_name}: {e}")

    async def getNextImage(self) -> Optional[ImageWithProviderMetadata]:
        if self._frame_size is None:
            try:
                await asyncio.wait_for(self._stream.wait_for_metadata(), timeout=FFMPEG_START_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error(f"FFmpeg metadata wait timed out after {FFMPEG_START_TIMEOUT} seconds")
                raise
            self._frame_size = self._stream.get_frame_size()
        width, height = self._frame_size
        frame_size_bytes = width * height * 3

        if self._min_capture_interval > 0:
            now = self._now()
            if now < self._next_capture_time:
                await self._sleep(self._next_capture_time - now)

        while True:
            frame_bytes = await self._read_latest_frame(frame_size_bytes)

            if frame_bytes:
                self._restart_count = 0  # a successful frame clears cumulative restart history

                if self._min_capture_interval > 0:
                    self._next_capture_time = self._now() + self._min_capture_interval

                image = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
                frame_idx = self._frame_idx
                self._frame_idx += 1
                return ImageWithProviderMetadata(
                    image, metadata=ImageProviderMetadata(source=self.source_name, frame_idx=frame_idx)
                )

            if not self._loop:
                return None

            self._restart_count += 1
            if self._restart_count > self._MAX_STREAM_RESTARTS:
                logger.error(
                    f"Giving up on stream {self._stream} after {self._MAX_STREAM_RESTARTS} consecutive "
                    "restart attempts with no data"
                )
                return None

            logger.debug(
                f"Looping stream {self._stream} (restart attempt {self._restart_count}/{self._MAX_STREAM_RESTARTS})"
            )
            await self._stream.astop()
            await self._stream.astart()
            self._buffer = b""  # Clear the buffer
            await self._sleep(self._RESTART_BACKOFF)

    async def _read_latest_frame(self, frame_size_bytes: int) -> Optional[bytes]:
        """Returns the freshest full frame available, draining stale queued frames first.

        When the underlying streamer's queue items are whole frames (the normal case for this
        provider, since output is always rawvideo), drains via read_nowait() to the newest item
        and only blocks on read_async() if nothing was queued. Falls back to chunk accumulation
        if the streamer somehow isn't frame-aligned yet.
        """
        if self._stream.is_frame_aligned():
            latest: Optional[bytes] = None
            drained = self._stream.read_nowait()
            while drained is not None:
                latest = drained
                drained = self._stream.read_nowait()

            if latest is not None:
                return latest

            try:
                return await asyncio.wait_for(self._stream.read_async(), timeout=STREAM_READ_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error(f"FFmpeg stream read timed out after {STREAM_READ_TIMEOUT} seconds")
                return None

        # Not frame-aligned yet (e.g. metadata just arrived) — accumulate chunks the old way.
        while len(self._buffer) < frame_size_bytes:
            try:
                chunk: Optional[bytes] = await asyncio.wait_for(self._stream.read_async(), timeout=STREAM_READ_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error(f"FFmpeg stream read timed out after {STREAM_READ_TIMEOUT} seconds")
                chunk = None
            if not chunk:
                return None
            self._buffer += chunk

        frame_bytes = self._buffer[:frame_size_bytes]
        self._buffer = self._buffer[frame_size_bytes:]
        return frame_bytes

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        """
        Return a schema describing the parameters for this Provider
        Each key is a parameter name, value is a dict with:
            - type: str
            - required: bool
            - default: Any (optional)
            - help: str (optional)
            - options: list (optional, for enums)
            - schema: dict (optional, for nested objects)
        """
        return {
            "video_path": {
                "label": "Video Source Path",
                "type": "string",
                "required": True,
                "description": "Path to pass to ffmpeg (eg, rtsp://, server-local url path, etc)",
            },
            "rtsp_relay": {
                "label": "Use RTSP Proxy Server",
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Proxy RTSP streams through MediaMTX",
            },
            "loop": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Whether to loop the video",
            },
            "min_capture_interval": {
                "label": "Min Capture Interval (s)",
                "type": "float",
                "required": False,
                "default": 0.0,
                "description": "Minimum seconds between frames returned by this source. 0 = no source-level limit (the task's own interval applies).",
            },
        }
