"""Manages streams on the MediaMTX server.

This service does not hold any media itself. It owns the *policy* around the NVR's
rolling buffer while MediaMTX owns the bytes: each master relay path records
keyframe-aligned fMP4 segments into an in-RAM tmpfs (``/buffer``) with a short
``recordDeleteAfter``, and MediaMTX's playback server cuts arbitrary windows back out
of it on demand. That is what lets an event choose its pre-roll *after* it has
happened, instead of a fixed delay decided when the stream was created.

It used to buffer in Python — a `DelayedStreamer` holding chunks in a bounded queue and
re-publishing them N seconds late. That was removed (see `67adce87`), and with it the
whole notion of a stream that is permanently behind. Nothing here queues frames now;
the only reason this process exists between the app and MediaMTX is the state below,
which MediaMTX has no concept of.

Three properties this file is responsible for keeping true:

* **Master path names are stable across restarts.** ``recordPath`` templates on
  ``%path``, so a process-lifetime counter would scatter one camera's recordings
  across a new directory every restart.
* **Startup reconciles, it does not wipe.** A 24/7 watcher outlives relay restarts,
  so leases are persisted and re-armed rather than every path being deleted.
* **The buffer is never over-committed.** A full tmpfs makes MediaMTX's recorder fail
  writes rather than evict early, so arming a source that would breach the mount is
  refused up front.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import mediamtx_client
from mediamtx_client.api.configuration_api import ConfigurationApi
from mediamtx_client.api.paths_api import PathsApi
from mediamtx_client.models.path import Path as MtxPath
from mediamtx_client.models.path_conf import PathConf

from apps import STREAM_BUFFER_DIR, STREAM_BUFFER_SIZE
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.recordings.playbackClient import (
    PlaybackClient,
    PlaybackError,
    piece_windows,
    probe_duration_s,
)
from apps.helpers.serviceStatus import fmt_duration, row, section
from apps.helpers.streams.bufferBudget import (
    COMMIT_HEADROOM,
    DEFAULT_ASSUMED_BITRATE_BPS,
    BufferUsage,
    bytes_for,
    measure_mount_bytes,
    parse_size,
)
from apps.helpers.streams.ingest import IngestPlan, IngestTier, resolve_ingest_tier

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

# How much rewind a recorded path keeps by default. Deliberately under a minute: the
# buffer is RAM, and the design trades depth for costing ~nothing when idle.
DEFAULT_RETENTION_S = float(os.environ.get("STREAM_BUFFER_RETENTION_S", "45"))
# Short segments keep the granularity of a cut fine relative to a <60 s buffer, and
# bound how much a keyframe-aligned start can overshoot the requested window.
SEGMENT_DURATION = "5s"
PART_DURATION = "500ms"
# Where leases survive a relay restart. Under /data because that is bind-mounted;
# /buffer itself is container-local and evaporates with the process (by design).
LEASE_FILE = Path(os.environ.get("RELAY_LEASE_FILE", "/data/relay/leases.json"))
# How long a lease survives without its holder checking in. MediaMTX will retry a dead
# camera forever — correctly, since a camera that is merely offline must come back on
# its own — so nothing below this layer ever concludes that a *consumer* has died. That
# is what the TTL is for: it reaps the holder, not the camera. Generous relative to
# LEASE_RENEW_INTERVAL_S (see ffmpegImageProvider) so a busy event loop or a brief relay
# outage can never expire a lease whose holder is alive and trying.
LEASE_TTL_S = float(os.environ.get("RELAY_LEASE_TTL_S", "300"))
# How far a cut may fall short of the window it was charged for before it is worth
# reporting. Below this is keyframe alignment and muxer rounding, which every cut has.
_CUT_SHORTFALL_TOLERANCE_S = 0.5


def master_path_name(source_url: str) -> str:
    """Deterministic MediaMTX path name for a source URL.

    Stable across restarts because ``recordPath`` templates on ``%path``: a
    process-lifetime counter would move each camera's recordings to a new directory
    every time the relay came back. Hashed rather than slugged because source URLs
    carry credentials and characters MediaMTX path names disallow.
    """
    digest = hashlib.sha1(source_url.strip().encode("utf-8")).hexdigest()[:12]
    return f"relay_master_{digest}"


@dataclass
class Lease:
    """One consumer's claim on a source's relay path.

    Leases are what make ``ensure`` idempotent and refcounted at the same time: two
    watchers on one camera hold two leases against one path, and the path lives until
    the last of them is released. They are persisted so a relay restart re-arms paths
    that still have holders rather than dropping their consumers on the floor.
    """

    lease_id: str
    source_url: str
    holder: str
    retention_s: float
    record: bool
    assumed_bitrate_bps: int = DEFAULT_ASSUMED_BITRATE_BPS
    created_at: float = field(default_factory=time.time)
    # Last time the holder said it was still there. A holder that stops renewing has
    # died, and its lease is reaped — see ``Manager._expire_stale_leases``. Defaults to
    # now so a lease is always born healthy, including one restored from disk.
    last_seen: float = field(default_factory=time.time)

    def committed_bytes(self) -> int:
        return bytes_for(self.assumed_bitrate_bps, self.retention_s) if self.record else 0

    def is_stale(self, now: float, ttl_s: float) -> bool:
        return (now - self.last_seen) > ttl_s


@dataclass
class EnsureResult:
    """What a consumer gets back from ``ensure_stream``.

    ``recording``/``retention_s`` are on here deliberately: a watcher must be able to
    tell the user "this source has no rewind buffer" rather than silently producing
    zero-pre-roll clips.
    """

    path: str
    rtsp_url: str
    lease_id: str
    recording: bool
    retention_s: float
    ingest_tier: str = "native"
    transcoding: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "rtspUrl": self.rtsp_url,
            "leaseId": self.lease_id,
            "recording": self.recording,
            "retentionS": self.retention_s,
            "ingestTier": self.ingest_tier,
            "transcoding": self.transcoding,
        }


class BufferFullError(RuntimeError):
    """Raised when arming a source would commit more than the /buffer mount holds."""


class Manager:
    @dataclass
    class StreamInfo:
        name: str
        url: Optional[str] = None
        mtx_path: Optional[MtxPath] = None
        active: bool = False
        expected_active: Optional[bool] = None
        owned: bool = False  # True if Manager created this
        source_url: Optional[str] = None  # Tracks the original RTSP source (if relay)
        recording: bool = False
        retention_s: float = 0.0

    @dataclass
    class SourceRelayInfo:
        master_stream_name: str  # Internal relay that actually connects to remote source
        refcount: int = 1

    POLLING_INTERVAL: float = 5.0  # Interval in seconds for periodic tasks

    def __init__(self, media_mtx_api_client: mediamtx_client.ApiClient):
        """Initialize the Manager with an API client"""
        self._media_mtx_api_client: mediamtx_client.ApiClient = media_mtx_api_client
        self._streams: Dict[str, Manager.StreamInfo] = {}
        self._source_relays: Dict[str, Manager.SourceRelayInfo] = {}
        self._leases: Dict[str, Lease] = {}
        # Resolved ingest plan per source, so the tier reported to consumers is the
        # same one the path was actually armed with.
        self._ingest_plans: Dict[str, IngestPlan] = {}
        self._hostname: str = "localhost"
        self._api_port: int = 9997
        self._rtsp_port: int = 8554
        self._playback_port: int = 9996
        self._task: Optional[asyncio.Task] = None  # Background task for periodic operations
        self._buffer_dir: Path = STREAM_BUFFER_DIR
        # Instance attribute rather than the module constant directly, so a test (or a
        # second relay in one process) can point persistence somewhere harmless.
        self._lease_file: Path = LEASE_FILE
        try:
            self._buffer_cap_bytes: int = parse_size(STREAM_BUFFER_SIZE)
        except ValueError as exc:
            # A malformed cap must not take the relay down; fall back to the documented
            # default and say so loudly, since the tmpfs mount option is the real limit.
            logger.error(f"Invalid STREAM_BUFFER_SIZE {STREAM_BUFFER_SIZE!r} ({exc}); falling back to 512m")
            self._buffer_cap_bytes = parse_size("512m")
        self._config = {
            "api_server": f"http://{self._hostname}:{self._api_port}",
            "rtsp_server": f"rtsp://{self._hostname}:{self._rtsp_port}",
            "mediamtx_server": self._media_mtx_api_client.configuration.host,
        }

    # ------------------------------------------------------------------ config

    async def get_server_config(self) -> Dict[str, Any]:
        return self._config.copy()

    def playback_client(self) -> PlaybackClient:
        return PlaybackClient(host=self._hostname, port=self._playback_port)

    # ------------------------------------------------------------------ buffer

    def buffer_usage(self) -> BufferUsage:
        """Current budget of the RAM buffer.

        ``cap`` is the smaller of the declared ``STREAM_BUFFER_SIZE`` and what the mount
        actually reports, because the mount is the thing that fails; the declared value
        alone would happily over-commit a tmpfs someone mounted smaller by hand.
        """
        total, used = measure_mount_bytes(str(self._buffer_dir))
        cap = min(self._buffer_cap_bytes, total) if total else self._buffer_cap_bytes
        committed = sum(lease.committed_bytes() for lease in self._leases.values())
        return BufferUsage(cap_bytes=cap, committed_bytes=committed, used_bytes=used)

    def _check_commit(self, additional: int) -> None:
        usage = self.buffer_usage()
        budget = int(usage.cap_bytes * COMMIT_HEADROOM)
        if usage.committed_bytes + additional > budget:
            raise BufferFullError(
                f"Arming this source would commit "
                f"{(usage.committed_bytes + additional) / 1e6:.0f} MB against a "
                f"{budget / 1e6:.0f} MB usable buffer "
                f"(cap {usage.cap_bytes / 1e6:.0f} MB). Raise STREAM_BUFFER_SIZE, "
                f"shorten the retention, or watch fewer cameras."
            )

    def _record_conf(self, retention_s: float) -> dict:
        """The recording half of a master path's PathConf.

        fMP4 is not a preference — the playback server can only serve that format.
        """
        return {
            "record": True,
            "recordPath": str(self._buffer_dir / "%path" / "%Y-%m-%d_%H-%M-%S-%f"),
            "recordFormat": "fmp4",
            "recordPartDuration": PART_DURATION,
            "recordSegmentDuration": SEGMENT_DURATION,
            "recordDeleteAfter": f"{int(retention_s)}s",
        }

    # ------------------------------------------------------------------ leases

    def _load_leases(self) -> None:
        """Restore leases written by a previous process."""
        if not self._lease_file.exists():
            return
        try:
            raw = json.loads(self._lease_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Could not read lease file {self._lease_file}: {exc}")
            return
        for entry in raw.get("leases", []):
            try:
                lease = Lease(**entry)
            except TypeError as exc:
                logger.warning(f"Skipping unreadable lease {entry!r}: {exc}")
                continue
            # Restart the liveness clock rather than trusting the persisted value: the
            # holders are in other processes and have had no chance to check in yet, so
            # a stored `last_seen` from before a long shutdown would reap every lease
            # the instant the relay came back. A genuinely dead holder simply fails to
            # renew and is reaped one TTL later instead.
            lease.last_seen = time.time()
            self._leases[lease.lease_id] = lease
        logger.info(f"Loaded {len(self._leases)} lease(s) from {self._lease_file}")

    def _save_leases(self) -> None:
        """Persist leases so the next process can reconcile rather than wipe."""
        try:
            self._lease_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {"leases": [asdict(lease) for lease in self._leases.values()]}
            tmp = self._lease_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self._lease_file)
        except OSError as exc:
            # Losing persistence degrades restart behaviour but must not fail the
            # request that triggered it — the in-memory lease is still authoritative.
            logger.warning(f"Could not persist leases to {self._lease_file}: {exc}")

    def leases_for(self, source_url: str) -> List[Lease]:
        return [lease for lease in self._leases.values() if lease.source_url == source_url]

    def _find_lease(self, source_url: str, holder: str) -> Optional[Lease]:
        """The lease this holder already has on this source, if any."""
        for lease in self._leases.values():
            if lease.source_url == source_url and lease.holder == holder:
                return lease
        return None

    def renew_lease(self, lease_id: str) -> bool:
        """Mark a holder as still alive. Returns False for an unknown lease.

        A False here is a holder's cue to re-``ensure``: it means the relay no longer
        believes in this lease (it was reaped, or the lease file was lost), so the path
        it is reading may be gone.
        """
        lease = self._leases.get(lease_id)
        if lease is None:
            return False
        lease.last_seen = time.time()
        # Deliberately not persisted on every renew: a heartbeat per holder per minute
        # would rewrite the file constantly for information that is only meaningful
        # while this process is alive. `_load_leases` re-arms the clock on restart.
        return True

    async def _expire_stale_leases(self) -> None:
        """Reap leases whose holder stopped checking in.

        This is the only thing in the stack that concludes a consumer is gone. It must
        never be confused with the camera being unreachable: an offline camera with a
        live watcher keeps its lease and its path, because MediaMTX reconnecting when
        the camera returns is exactly the behaviour a 24/7 watcher wants.
        """
        now = time.time()
        stale = [lease for lease in self._leases.values() if lease.is_stale(now, LEASE_TTL_S)]
        for lease in stale:
            logger.warning(
                f"Reaping lease {lease.lease_id} on {lease.source_url}: holder "
                f"{lease.holder!r} last seen {now - lease.last_seen:.0f}s ago "
                f"(ttl {LEASE_TTL_S:.0f}s)"
            )
            await self.release_lease(lease.lease_id)

    def reader_counts(self) -> Dict[str, int]:
        """Holders per master path — the status panel's "who is watching this camera"."""
        counts: Dict[str, int] = {}
        for lease in self._leases.values():
            name = master_path_name(lease.source_url)
            counts[name] = counts.get(name, 0) + 1
        return counts

    # ------------------------------------------------------------------ ensure

    async def ensure_stream(
        self,
        source_url: str,
        holder: str = "",
        retention_s: Optional[float] = None,
        record: bool = True,
    ) -> EnsureResult:
        """Idempotently arm a relay path for ``source_url`` and take a lease on it.

        Consumers read the master path *directly*. There is deliberately no
        per-consumer path: fanning out from the master is the entire point of the
        relay, and a second path per consumer only added an RTSP hop and another
        MediaMTX reader for nothing.
        """
        source_url = source_url.strip()
        name = master_path_name(source_url)
        retention = DEFAULT_RETENTION_S if retention_s is None else float(retention_s)

        existing = self._source_relays.get(source_url)
        if existing is not None:
            # The relay restarts independently of this process, and an entry here is
            # only a memory of having armed the path once. Trusting it is what let
            # `ensure` answer success for a path MediaMTX had already forgotten.
            names = await self._configured_path_names()
            if names is not None and existing.master_stream_name not in names:
                logger.warning(f"Relay has no path {existing.master_stream_name} for {source_url}; re-arming")
                self._source_relays.pop(source_url, None)
                self._streams.pop(existing.master_stream_name, None)
                existing = None

        if existing is None:
            if record:
                self._check_commit(bytes_for(DEFAULT_ASSUMED_BITRATE_BPS, retention))
            await self._create_master_path(source_url, name, retention, record)
        elif record and not self._streams.get(name, Manager.StreamInfo(name=name)).recording:
            # The path exists but was armed without recording (or with a shorter
            # window); widen it in place rather than tearing the camera's connection
            # down, since other consumers are reading it right now.
            self._check_commit(bytes_for(DEFAULT_ASSUMED_BITRATE_BPS, retention))
            await self._patch_recording(name, source_url, retention)

        # A named holder re-ensuring the same source is checking in, not asking for a
        # second claim: refresh the lease it already has. Without this, the plan's
        # "provider-side re-ensure on read timeout" would mint a fresh lease on every
        # retry and inflate the refcount until nothing could ever release the path.
        # Anonymous callers are exempt — they are indistinguishable from one another,
        # so collapsing them would let one caller's release free another's path.
        existing_lease = self._find_lease(source_url, holder) if holder else None
        if existing_lease is not None:
            existing_lease.last_seen = time.time()
            self._save_leases()
            lease = existing_lease
        else:
            lease = Lease(
                lease_id=str(uuid.uuid4()),
                source_url=source_url,
                holder=holder or "anonymous",
                retention_s=retention,
                record=record,
            )
            self._leases[lease.lease_id] = lease
            self._save_leases()

        info = self._streams.get(name)
        relay = self._source_relays.get(source_url)
        if relay is not None:
            relay.refcount = len(self.leases_for(source_url))

        plan = self._ingest_plans.get(source_url)
        return EnsureResult(
            path=name,
            rtsp_url=f"rtsp://{self._hostname}:{self._rtsp_port}/{name}",
            lease_id=lease.lease_id,
            recording=bool(info and info.recording),
            retention_s=info.retention_s if info else 0.0,
            # The tier drives the UI's four-state buffer badge and the watcher's choice
            # of clip backend. Reporting it here keeps those two from disagreeing.
            ingest_tier=plan.tier.value if plan else IngestTier.NATIVE.value,
            transcoding=bool(plan and plan.transcoding),
        )

    async def _create_master_path(self, source_url: str, name: str, retention_s: float, record: bool) -> None:
        api: ConfigurationApi = ConfigurationApi(self._media_mtx_api_client)
        plan = resolve_ingest_tier(params={"source_url": source_url})

        if plan.tier is IngestTier.FFMPEG:
            # MediaMTX can't pull this URL, but ffmpeg can read it. Let MediaMTX spawn
            # and supervise the ingest via runOnInit — that buys process restart for
            # free, and (unlike runOnDemand) keeps the ingest up when nobody is
            # watching, which continuous recording requires.
            conf: Dict[str, Any] = {
                "name": name,
                "source": "publisher",
                "runOnInit": plan.run_on_init,
                "runOnInitRestart": True,
            }
        elif plan.tier is IngestTier.NATIVE:
            conf = {"name": name, "source": source_url, "sourceOnDemand": False}
        else:
            # Snapshot/MQTT sources have no continuous elementary stream, so there is
            # nothing for MediaMTX to record. The watcher gives them an in-RAM JPEG
            # frame ring instead; arming a path here would just create a dead one.
            raise ValueError(
                f"Source {source_url!r} cannot be ingested into a MediaMTX path "
                f"({plan.reason}); it needs the frame-ring buffer instead."
            )

        if record:
            conf.update(self._record_conf(retention_s))
        path_conf = PathConf(**conf)
        try:
            await asyncio.to_thread(api.config_paths_add, name=name, path_conf=path_conf)
        except Exception as exc:
            # A path left behind by a previous process is the expected reconcile case,
            # not a failure: adopt it by patching it to the config we want.
            logger.info(f"config_paths_add failed for {name} ({exc}); patching existing path instead")
            await asyncio.to_thread(api.config_paths_patch, name=name, path_conf=path_conf)

        self._source_relays[source_url] = Manager.SourceRelayInfo(master_stream_name=name, refcount=0)
        self._streams[name] = Manager.StreamInfo(
            name=name,
            url=f"rtsp://{self._hostname}:{self._rtsp_port}/{name}",
            owned=True,
            source_url=source_url,
            recording=record,
            retention_s=retention_s if record else 0.0,
        )
        self._ingest_plans[source_url] = plan
        logger.info(
            f"Armed master path {name} for {source_url} "
            f"(tier={plan.tier.value}, transcoding={plan.transcoding}, "
            f"record={record}, retention={retention_s}s)"
        )
        await self.refresh_streams()

    async def _patch_recording(self, name: str, source_url: str, retention_s: float) -> None:
        api: ConfigurationApi = ConfigurationApi(self._media_mtx_api_client)
        path_conf = PathConf(name=name, source=source_url, sourceOnDemand=False, **self._record_conf(retention_s))
        await asyncio.to_thread(api.config_paths_patch, name=name, path_conf=path_conf)
        info = self._streams.get(name)
        if info:
            info.recording = True
            info.retention_s = retention_s
        logger.info(f"Enabled recording on existing path {name} (retention={retention_s}s)")

    async def release_lease(self, lease_id: str) -> bool:
        """Drop one consumer's claim; tear the path down when it was the last."""
        lease = self._leases.pop(lease_id, None)
        if lease is None:
            logger.warning(f"Releasing unknown lease {lease_id}")
            return False
        self._save_leases()

        remaining = self.leases_for(lease.source_url)
        relay = self._source_relays.get(lease.source_url)
        if relay is not None:
            relay.refcount = len(remaining)
        if remaining:
            logger.info(f"Released lease {lease_id}; {len(remaining)} holder(s) remain on {lease.source_url}")
            return True

        name = master_path_name(lease.source_url)
        logger.info(f"Last holder of {lease.source_url} released; removing path {name}")
        api: ConfigurationApi = ConfigurationApi(self._media_mtx_api_client)
        try:
            await asyncio.to_thread(api.config_paths_delete, name=name)
        except Exception as exc:
            logger.warning(f"Failed to delete master path {name}: {exc}")
        self._source_relays.pop(lease.source_url, None)
        self._streams.pop(name, None)
        # The dominant teardown in production — every on-demand camera's sleep cycle
        # ends here rather than in `destroy_stream`. `sweep_orphan_buffers` would
        # collect these eventually, but only after the recency guard elapses, leaving a
        # window in which a re-armed path inherits its own predecessor's segments.
        await self._purge_buffer(name)
        await self.refresh_streams()
        return True

    async def _reconcile(self) -> None:
        """Re-arm paths for leases that outlived the previous process.

        This replaces the old ``destroy_all_streams()`` on startup, which would nuke
        paths in use by running tasks and tell no one. Note the buffer itself does not
        survive: it is RAM, so a restart costs every open event its pre-roll, and
        consumers must treat shrunken coverage as a notice rather than silence.
        """
        self._load_leases()
        by_source: Dict[str, List[Lease]] = {}
        for lease in self._leases.values():
            by_source.setdefault(lease.source_url, []).append(lease)

        for source_url, leases in by_source.items():
            name = master_path_name(source_url)
            record = any(lease.record for lease in leases)
            retention = max((lease.retention_s for lease in leases), default=DEFAULT_RETENTION_S)
            try:
                await self._create_master_path(source_url, name, retention, record)
                self._source_relays[source_url].refcount = len(leases)
                logger.info(f"Reconciled {name} for {len(leases)} surviving lease(s)")
            except Exception as exc:
                # One unreachable camera must not stop the rest from coming back.
                logger.error(f"Failed to reconcile {name} for {source_url}: {exc}")

    async def _configured_path_names(self) -> Optional[Set[str]]:
        """Names MediaMTX currently has *configured*, or None if it could not be asked.

        Configured rather than active: a path whose camera is unreachable is still
        configured and is retrying on its own, so treating it as lost would drop the
        connection attempt and start it over.
        """
        api: ConfigurationApi = ConfigurationApi(self._media_mtx_api_client)
        try:
            configured = await asyncio.to_thread(api.config_paths_list)
        except Exception as exc:
            # Nothing is known either way, so the caller must not conclude "lost".
            logger.warning(f"Could not list configured relay paths: {exc}")
            return None
        return {str(item.name) for item in (configured.items or [])}

    async def _rearm_lost_paths(self) -> None:
        """Re-arm leased paths MediaMTX has forgotten while this process kept running.

        `_reconcile` covers the other direction — this process restarting under a live
        relay — and the two are not interchangeable. MediaMTX exits on a panic in its
        own HTTP handler, so it restarts with an empty path table while every lease,
        refcount and `_source_relays` entry here survives. Nothing re-arms on its own:
        holders renew, which only bumps `last_seen`, and `ensure_stream` short-circuits
        on the surviving `_source_relays` entry — so a relay restart that lasts seconds
        becomes an outage that lasts until someone notices.
        """
        if not self._source_relays and not self._leases:
            return
        names = await self._configured_path_names()
        if names is None:
            return

        # Driven by the leases, not by `_source_relays`: a lease is durable state a
        # holder is actively renewing, while an entry here is this process's memory of
        # having armed a path once. A leased source missing from `_source_relays` is
        # otherwise invisible to every recovery path there is.
        sources = set(self._source_relays) | {lease.source_url for lease in self._leases.values()}
        for source_url in sources:
            relay = self._source_relays.get(source_url)
            name = relay.master_stream_name if relay else master_path_name(source_url)
            if name in names:
                continue
            leases = self.leases_for(source_url)
            if not leases:
                # Believed-armed, gone, and unwanted. Dropping the entry is the point:
                # left behind, it is what makes the next `ensure` claim success for a
                # path that does not exist.
                logger.info(f"Dropping stale relay entry {name} for {source_url}; path is gone and unleased")
                self._source_relays.pop(source_url, None)
                self._streams.pop(name, None)
                continue
            record = any(lease.record for lease in leases)
            retention = max((lease.retention_s for lease in leases), default=DEFAULT_RETENTION_S)
            logger.warning(
                f"Relay lost path {name} for {source_url}; re-arming for {len(leases)} lease(s) "
                f"(record={record}, retention={retention}s)"
            )
            try:
                await self._create_master_path(source_url, name, retention, record)
                rearmed = self._source_relays.get(source_url)
                if rearmed is not None:
                    rearmed.refcount = len(leases)
            except Exception as exc:
                # One unreachable camera must not stop the rest from coming back.
                logger.error(f"Failed to re-arm {name} for {source_url}: {exc}")

    # ------------------------------------------------------------------ playback

    async def get_ranges(self, path: str) -> List[dict]:
        """Buffered time ranges for a path, as the clip UI needs them."""
        ranges = await self.playback_client().list_ranges(path)
        return [r.to_dict() for r in ranges]

    async def extract_clip(self, path: str, start: datetime, duration_s: float, dest: Path) -> dict:
        """Cut ``[start, start+duration_s)`` out of the buffer into ``dest``.

        Clamps the request to what the buffer actually still holds and reports the
        clamp, because a RAM buffer silently shrinks across a relay restart and a
        truncated clip must never be presented as if its pre-roll were intact.
        """
        client = self.playback_client()
        ranges = await client.list_ranges(path)
        if not ranges:
            raise PlaybackError(f"No buffered recording for path {path!r}")

        oldest = min(r.start for r in ranges)
        newest_end = max(r.end for r in ranges)
        requested_start = start
        actual_start = max(start, oldest)
        end = min(start + timedelta(seconds=duration_s), newest_end)
        actual_duration = max(0.0, (end - actual_start).total_seconds())
        if actual_duration <= 0:
            raise PlaybackError(
                f"Requested window for {path!r} has already aged out of the buffer "
                f"(buffer covers {oldest.isoformat()}..{newest_end.isoformat()})"
            )

        # One cut per advertised range, not one cut spanning them all. A window that
        # covers several ranges can come back short with the shortfall anywhere inside
        # it, and nothing downstream can then place a box: the hole has no position.
        # Cut per range and every piece is contiguous by construction, so measuring it
        # is enough to say exactly which wallclock it holds.
        pieces = piece_windows(ranges, actual_start, end)
        # The clamp above is to the oldest range's start, which is not the same as the
        # first *surviving* one: `/list` advertises ranges that leave holes, so a window
        # opening inside one is served from later video than it asked for. Taking the
        # start from the first piece is what keeps `startedAt` and `coveredRanges`
        # describing the file — report the clamp instead and every box lands early by
        # the length of the head hole.
        if pieces and pieces[0][0] > actual_start:
            actual_start = pieces[0][0]
            actual_duration = max(0.0, (end - actual_start).total_seconds())

        truncated = actual_start > requested_start
        if len(pieces) > 1:
            size, covered = await self._write_clip_in_pieces(client, path, pieces, dest)
            # A piece the server would not serve is dropped in there, so the file can
            # begin later than even the first piece promised. The cursor still lands on
            # `end` because the span is re-derived, and the head loss is then reported as
            # the clamp it is rather than hidden inside a window that claims to cover it.
            actual_start = covered[0][0]
            actual_duration = max(0.0, (end - actual_start).total_seconds())
            truncated = actual_start > requested_start
        else:
            size = await client.write_clip(path, actual_start, actual_duration, dest)
            held = await probe_duration_s(dest)
            # Measured, not advertised. One range is contiguous, so the video starts at
            # the window's own start — corrected above where that fell in a hole — and
            # runs for as long as the file says, which makes this exact even when the
            # cut came back short.
            covered = [(actual_start, actual_duration if held is None else held)]

        # Measured off the file, never taken from the requested window: `/list` describes
        # more than `/get` serves. It advertises the segment MediaMTX is still writing,
        # and a re-armed path can report ranges that overlap each other and leave holes,
        # so a window inside the advertised envelope still comes back short. Callers
        # place detection boxes by subtracting this from a wallclock, so an overstated
        # duration puts every box after the missing video late by its length.
        #
        # `None` means ffprobe could not read the file; the requested span is then the
        # only figure available.
        measured = await probe_duration_s(dest)
        played = actual_duration if measured is None else measured

        # A cut that came back short says something about the buffer that nothing else
        # observes: the ranges are only visible here, and /buffer is a tmpfs whose
        # evidence is gone within the retention window. Logged with the whole range list
        # because the shape is the diagnosis — a short cut inside one range is the
        # segment MediaMTX has not finalised, while ranges that overlap or leave holes
        # are a re-armed path reading its predecessor's segments.
        if actual_duration - played > _CUT_SHORTFALL_TOLERANCE_S:
            spans = " | ".join(f"{r.start.isoformat()}+{r.duration_s:.2f}s" for r in ranges)
            logger.warning(
                f"Short cut on {path}: charged {actual_duration:.2f}s, file holds {played:.2f}s "
                f"(requested {requested_start.isoformat()}+{duration_s:.2f}s); "
                f"{len(ranges)} range(s): {spans}"
            )
        return {
            "path": str(dest),
            "sizeBytes": size,
            "startedAt": actual_start.isoformat(),
            "durationS": played,
            "requestedStart": requested_start.isoformat(),
            "requestedDurationS": duration_s,
            # What the window would have been had the buffer held all of it. A caller
            # advancing a cursor needs the span it asked for; one placing boxes needs
            # `durationS`. They differ exactly when video is missing.
            "clampedDurationS": actual_duration,
            # The sub-windows the cut actually holds, each measured off its own piece,
            # so a caller placing boxes can map wallclock piecewise. Their durations sum
            # to `durationS` by construction rather than by luck — which is what lets a
            # consumer check them, and what `/list` alone could never support.
            "coveredRanges": [{"startedAt": lo.isoformat(), "durationS": held} for lo, held in covered],
            # The caller writes this onto the recording row: a clip whose pre-roll was
            # cut short by buffer eviction is a different artifact from a complete one.
            "partialPreroll": truncated,
        }

    async def _write_clip_in_pieces(
        self,
        client: PlaybackClient,
        path: str,
        pieces: "List[tuple[datetime, datetime]]",
        dest: Path,
    ) -> "tuple[int, List[tuple[datetime, float]]]":
        """Cut each piece separately, measure it, and join them into ``dest``.

        Returns the byte count and the windows the finished file holds, each paired with
        the video length measured off its own piece. Those lengths are what make the
        result mappable: a piece is one contiguous range, so its video starts where the
        piece was asked to start, and the measurement says how far it runs.

        A piece the server will not serve is a hole, not a failure. It is dropped, the
        remaining pieces still describe themselves correctly, and the clip is still
        worth having — the alternative is losing the whole event to one evicted segment.
        """
        from ..helpers.recordings.clipDrain import concat_parts

        staging = dest.parent / f".{dest.stem}.pieces"
        staging.mkdir(parents=True, exist_ok=True)
        parts: "List[Path]" = []
        covered: "List[tuple[datetime, float]]" = []
        try:
            for index, (lo, hi) in enumerate(pieces):
                span = (hi - lo).total_seconds()
                part = staging / f"{index:03d}.mp4"
                try:
                    await client.write_clip(path, lo, span, part)
                except Exception as exc:
                    logger.warning(f"Piece {index} of {path} ({lo.isoformat()}+{span:.2f}s) could not be cut: {exc}")
                    continue
                held = await probe_duration_s(part)
                # An unmeasurable piece is charged at its requested span: that is the
                # pre-existing behaviour for an unreadable file, and it is the only
                # figure available. The map's own total check is what catches it.
                held = span if held is None else held
                if held <= 0:
                    continue
                parts.append(part)
                covered.append((lo, held))
            if not parts:
                raise PlaybackError(f"No piece of the requested window for {path!r} could be cut")
            if len(parts) == 1:
                size = parts[0].stat().st_size
                parts[0].replace(dest)
            else:
                size = await asyncio.to_thread(concat_parts, parts, dest)
            return size, covered
        finally:
            for part in staging.glob("*.mp4"):
                part.unlink(missing_ok=True)
            try:
                staging.rmdir()
            except OSError:
                logger.debug("Could not remove piece staging dir %s", staging, exc_info=True)

    async def _purge_buffer(self, path_name: str) -> None:
        """Delete a destroyed path's recorded segments.

        MediaMTX applies `recordDeleteAfter` only to paths it still has configured, so
        segments left behind by a deleted path are never collected. They are also
        unreachable while the path is gone — the playback server refuses a path it has
        no config for — but `recordPath` templates on `%path` and path names are stable
        across re-arms, so the next arming of the same source inherits them: `/list`
        then reports ranges that overlap and leave holes, and a cut spanning them comes
        back short.

        Never raises. A failed purge costs RAM on a tmpfs; failing the teardown it is
        part of would strand the path itself.
        """
        target = self._buffer_dir / path_name
        try:
            # Resolved and re-checked, so a path name that ever contained a traversal
            # cannot aim this at the rest of the filesystem.
            resolved = target.resolve()
            if resolved.parent != self._buffer_dir.resolve() or not resolved.is_dir():
                return
            removed = sum(1 for _ in resolved.glob("*.mp4"))
            await asyncio.to_thread(shutil.rmtree, resolved, True)
            if removed:
                logger.info(f"Purged {removed} buffered segment(s) for destroyed path {path_name}")
        except Exception as e:
            logger.warning(f"Could not purge the buffer for {path_name}: {e}")

    async def sweep_orphan_buffers(self) -> int:
        """Purge segments belonging to paths MediaMTX no longer has configured.

        `_purge_buffer` covers an orderly teardown; this covers the rest — a crashed
        relay, a `config_paths_delete` that failed, or segments predating the fix. An
        unconfigured path is not being recorded to and has no retention applied, so its
        segments are dead weight that nothing else will ever collect.

        A directory is only swept once its newest segment is older than the retention
        window, which keeps a path that is mid-creation — configured a moment from now,
        or recording before `refresh_streams` has seen it — out of range.
        """
        root = self._buffer_dir
        if not root.is_dir():
            return 0
        configured = {name for name in self._streams}
        cutoff = time.time() - max(DEFAULT_RETENTION_S, self._max_retention_s()) - Manager.POLLING_INTERVAL
        swept = 0
        for child in root.iterdir():
            if not child.is_dir() or child.name in configured:
                continue
            segments = list(child.glob("*.mp4"))
            # The directory's own mtime stands in when it holds no segment yet: MediaMTX
            # creates it before it flushes the first one, so keying only off segments
            # would sweep a path that is mid-creation on the very pass that created it.
            newest = max((s.stat().st_mtime for s in segments), default=child.stat().st_mtime)
            if newest > cutoff:
                continue
            await self._purge_buffer(child.name)
            swept += 1
        return swept

    def _max_retention_s(self) -> float:
        """The deepest buffer any live lease has asked for."""
        return max((lease.retention_s for lease in self._leases.values()), default=0.0)

    async def destroy_stream(self, stream_name: str) -> None:
        """Destroy a stream with the given name, and possibly its master relay."""
        api: ConfigurationApi = ConfigurationApi(self._media_mtx_api_client)
        stream_info = self._streams.get(stream_name)
        if stream_info is None:
            logger.warning(f"Destroying unknown stream {stream_name}")
            return

        stream_info.active = False

        source_url = stream_info.source_url
        if source_url:
            relay_info = self._source_relays.get(source_url)
            if relay_info:
                if stream_name == relay_info.master_stream_name:
                    if relay_info.refcount > 0:
                        logger.info(f"Cannot destroy master stream {stream_name}, refcount = {relay_info.refcount}")
                        return
                    logger.info(f"Destroying master stream {stream_name}")
                    try:
                        await asyncio.to_thread(api.config_paths_delete, name=stream_name)
                    except Exception as e:
                        logger.warning(f"Failed to destroy master stream {stream_name}: {e}")
                    await self._purge_buffer(stream_name)
                    del self._source_relays[source_url]
                    self._streams.pop(stream_name, None)
                    await self.refresh_streams()
                    return
                else:
                    relay_info.refcount -= 1
                    logger.info(f"Decremented refcount for {relay_info.master_stream_name} → {relay_info.refcount}")
                    if relay_info.refcount <= 0:
                        master_name = relay_info.master_stream_name
                        logger.info(f"No more users of {master_name}; removing")
                        try:
                            await asyncio.to_thread(api.config_paths_delete, name=master_name)
                        except Exception as e:
                            logger.warning(f"Failed to destroy master stream {master_name}: {e}")
                        await self._purge_buffer(master_name)
                        del self._source_relays[source_url]
                        self._streams.pop(master_name, None)

        try:
            await asyncio.to_thread(api.config_paths_delete, name=stream_name)
        except Exception as e:
            logger.warning(f"Failed to destroy stream: {e}")

        self._streams.pop(stream_name, None)
        await self.refresh_streams()

    async def destroy_all_streams(self) -> None:
        """Destroys all streams on the server.

        No longer called on startup — see ``_reconcile``. A 24/7 watcher outlives the
        relay, so wiping every path at boot silently broke live consumers.
        """
        for name in list(self.get_streams().keys()):
            await self.destroy_stream(name)

    async def get_config(self) -> str:
        """Retrieve the global configuration as a string"""
        api: ConfigurationApi = ConfigurationApi(self._media_mtx_api_client)
        config = await asyncio.to_thread(api.config_global_get)
        return config.to_str()

    def get_streams(self) -> Dict[str, StreamInfo]:
        readers = self.reader_counts()
        ret = {}
        for k, v in self._streams.items():
            d = asdict(v)
            if d["mtx_path"] is not None:
                # model_dump, not .dict(): the latter is deprecated on pydantic v2 and
                # asdict() has already turned nested models into plain dicts anyway.
                mtx_path = v.mtx_path
                d["mtx_path"] = mtx_path.model_dump() if hasattr(mtx_path, "model_dump") else d["mtx_path"]
            d["readers"] = readers.get(k, 0)
            ret[k] = d
        return ret

    def get_status(self) -> Dict[str, Any]:
        """Service health, one-line summary, and per-stream detail rows.

        Reads the cached stream table (``refresh_streams`` does the MediaMTX call
        on its own poll), so this stays cheap enough for the notices loop.
        """
        streams = self.get_streams()
        recording = sum(1 for s in streams.values() if s.get("recording"))
        readers = sum(int(s.get("readers") or 0) for s in streams.values())
        usage = self.buffer_usage()

        summary_parts = [f"streams: {len(streams)}"]
        if recording:
            summary_parts.append(f"{recording} recording")
        if readers:
            summary_parts.append(f"{readers} reader{'' if readers == 1 else 's'}")

        rows = []
        for name, s in sorted(streams.items()):
            mtx = s.get("mtx_path") or {}
            parts = []
            # ``active`` is our view; MediaMTX's ``ready`` is whether the path is
            # actually publishing. They disagree while a source is reconnecting,
            # and that disagreement is the thing worth seeing here.
            parts.append("ready" if mtx.get("ready") else ("active" if s.get("active") else "idle"))
            reader_count = int(s.get("readers") or 0)
            if reader_count:
                parts.append(f"{reader_count} reader{'' if reader_count == 1 else 's'}")
            if s.get("recording"):
                retention = s.get("retention_s") or 0
                parts.append(f"recording ({fmt_duration(retention)} retention)" if retention else "recording")
            tracks = mtx.get("tracks")
            if tracks:
                # Codec names are a generated enum, whose str() is "PathTrackCodec.H264".
                # The panel wants the wire value, and plain strings must still pass through.
                parts.append("/".join(str(getattr(t, "value", t)) for t in tracks))
            received = mtx.get("bytes_received")
            if received:
                parts.append(f"{received / 1e6:.0f} MB in")
            source = s.get("source_url") or s.get("url")
            if source:
                parts.append(str(source))
            rows.append(row(name, "  ·  ".join(parts)))

        # used_bytes is None whenever the buffer mount cannot be stat'd, which
        # measure_mount_bytes treats as a supported deployment rather than a fault.
        used = f"{usage.used_bytes / 1e6:.0f} MB" if usage.used_bytes is not None else "unmeasured"
        buffer_rows = [
            row("Committed", f"{usage.committed_bytes / 1e6:.0f} MB"),
            row("Used on disk", used),
            row("Cap", f"{usage.cap_bytes / 1e6:.0f} MB"),
        ]

        return {
            "service": "stream_server",
            "status": "ok",
            "streams": {"active": len(streams), "recording": recording, "readers": readers},
            "buffer": usage.to_dict(),
            "summary": "  ·  ".join(summary_parts),
            "details": [
                section(f"Streams ({len(streams)})", rows, empty="No streams"),
                section("Buffer", buffer_rows),
            ],
        }

    async def refresh_streams(self) -> None:
        """Retrieve the current list of streams, updating the cached list"""
        api: PathsApi = PathsApi(self._media_mtx_api_client)
        streams = await asyncio.to_thread(api.paths_list)
        if streams.items is None:
            raise Exception("Failed to get stream list from server")
        stream_dict: Dict[str, MtxPath] = {str(stream.name): stream for stream in streams.items}

        # Add new streams
        for name, mtx_path in stream_dict.items():
            if name not in self._streams:
                logger.info(f"Unexpected new stream on server: {name}")
                self._streams[name] = Manager.StreamInfo(name=name, owned=False)
            stream_info = self._streams[name]
            stream_info.mtx_path = mtx_path
            stream_info.url = f"rtsp://{self._hostname}:{self._rtsp_port}/{name}"
            if mtx_path.ready != stream_info.active:
                logger.info(f"Stream {name} set to url: {stream_info.url}  active: {mtx_path.ready}")
                stream_info.active = mtx_path.ready
                if stream_info.owned:
                    if stream_info.expected_active is None and stream_info.active:
                        stream_info.expected_active = True
                    elif stream_info.expected_active and not stream_info.active:
                        logger.warning(f"Stream {name} expected active but is not!")
        # Remove unexpectedly closed streams
        for name in list(self._streams.keys()):
            if name not in stream_dict:
                mtx_path = self._streams[name]
                if mtx_path.active:
                    logger.warning(f"Stream unexpectedly removed: {name}")
                else:
                    logger.info(f"Confirming removal of {name}")
                self._streams.pop(name)

    def start(self):
        """Start the periodic worker task, should be called from the event loop to run on"""
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._worker_task())

    async def _init(self):
        """Initialization that should run on event loop"""
        config_api: ConfigurationApi = ConfigurationApi(self._media_mtx_api_client)
        config = await asyncio.to_thread(config_api.config_global_get)
        host_url = self._media_mtx_api_client.configuration.host
        host = host_url.split("://")[1]
        host_name, host_port = host.split(":")
        self._hostname = host_name
        self._api_port = host_port
        self._rtsp_port = config.rtsp_address.split(":")[1]
        playback_address = getattr(config, "playback_address", None)
        if playback_address:
            self._playback_port = int(playback_address.split(":")[-1])
        self._config["api_server"] = f"http://{self._hostname}:{self._api_port}"
        self._config["rtsp_server"] = f"rtsp://{self._hostname}:{self._rtsp_port}"
        await self.refresh_streams()
        await self._reconcile()

    async def _try_init(self) -> bool:
        """Run ``_init``, reporting failure rather than raising.

        Supervisord brings this service and MediaMTX up together, so the first attempt
        can easily land before the relay's API is listening. Raising here leaves the
        worker dead for the life of the process.
        """
        try:
            await self._init()
            return True
        except Exception as exc:
            logger.warning(f"Relay not ready to init ({exc}); retrying in {Manager.POLLING_INTERVAL}s")
            return False

    async def _worker_task(self):
        """Periodic worker task that runs at regular intervals.

        Every step is guarded and the loop exits only on cancellation. MediaMTX exits
        on a panic in its own HTTP handler, so any call into it can fail on any tick;
        an unguarded step lets that kill the loop for good, and with it
        ``_rearm_lost_paths`` — the only thing that brings a camera back after the
        relay restarts with an empty path table.
        """
        try:
            while not await self._try_init():
                await asyncio.sleep(Manager.POLLING_INTERVAL)
            while True:
                await asyncio.sleep(Manager.POLLING_INTERVAL)
                # refresh_streams runs first: the steps after it read self._streams.
                for step in (
                    self.refresh_streams,
                    self._expire_stale_leases,
                    self.sweep_orphan_buffers,
                    self._rearm_lost_paths,
                ):
                    try:
                        await step()
                    except Exception as exc:
                        logger.error(f"Periodic {step.__name__} failed: {exc}")
        except asyncio.CancelledError:
            logger.debug("Periodic worker cancelled.")
        except BaseException:
            logger.exception("Periodic worker died; no lost path will be re-armed until this service restarts")
            raise
