"""Resolves how a source gets into a MediaMTX path — and therefore whether it can have
a rolling buffer at all.

Recording and playback are properties of the *path*, not of the camera protocol, so
"can this source be rewound?" reduces to "can we get it into a path?". Three tiers:

* **native** — MediaMTX pulls the URL itself (``rtsp://``, ``rtmp://``, ``srt://``,
  HLS over http(s), MPEG-TS over ``udp://``, WebRTC/WHEP). Set ``source=<url>`` and
  recording is free: no extra processes at all.
* **ffmpeg** — MediaMTX can't pull it, but ffmpeg can read it (MJPEG-over-HTTP, V4L2
  devices, local files, codec fixups). MediaMTX spawns and *supervises* the ingest
  itself via ``runOnInit``, so we get restart handling for free rather than
  re-inventing process babysitting.
* **ring** — snapshot pollers and MQTT feeds. There is no continuous elementary
  stream, so there is no timeline to record. These get a bounded in-RAM JPEG frame
  ring instead, flushed to mp4 on event start: choppy, honestly labelled, and far
  better than no pre-roll at all.
* **none** — buffering explicitly disabled, or a source we can't ingest by any route.

Exactly one function decides this, because the answer drives the relay's path config,
the watcher's clip backend, *and* the four-state buffer badge in the UI. Those must
never disagree.
"""

import shlex
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse


class IngestTier(str, Enum):
    NATIVE = "native"
    FFMPEG = "ffmpeg"
    RING = "ring"
    NONE = "none"


# Schemes MediaMTX pulls without help. `http(s)` is deliberately absent: it depends on
# what's behind the URL (HLS playlist vs MJPEG vs a still image), so it is resolved
# separately below.
NATIVE_SCHEMES = frozenset({"rtsp", "rtsps", "rtmp", "rtmps", "srt", "udp", "whep"})

# Extensions that mean "HLS playlist", which MediaMTX pulls natively over http(s).
HLS_SUFFIXES = (".m3u8",)

# Codecs we can pass through without re-encoding. Anything else needs a transcode,
# which is real per-camera CPU — unlike everything else in this design — so it has to
# be visible in the UI rather than a silent cost.
PASSTHROUGH_CODECS = frozenset({"h264", "h265", "hevc", "avc"})


@dataclass(frozen=True)
class IngestPlan:
    """How to arm a source, and what that costs."""

    tier: IngestTier
    #: ffmpeg command for Tier B, to be run by MediaMTX as ``runOnInit``.
    run_on_init: Optional[str] = None
    #: True when the ffmpeg ingest must re-encode rather than copy.
    transcoding: bool = False
    #: Human-readable explanation, shown next to the buffer badge.
    reason: str = ""

    @property
    def buffered(self) -> bool:
        """Whether this source gets a MediaMTX-backed rewind timeline."""
        return self.tier in (IngestTier.NATIVE, IngestTier.FFMPEG)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "runOnInit": self.run_on_init,
            "transcoding": self.transcoding,
            "reason": self.reason,
            "buffered": self.buffered,
        }


def build_ffmpeg_ingest(input_url: str, transcode: bool = False) -> str:
    """The ``runOnInit`` command MediaMTX uses to pull a Tier B source into a path.

    ``runOnInit`` rather than ``runOnDemand``: on-demand only runs while a reader is
    attached, and continuous recording needs the ingest up regardless of whether
    anyone is currently watching.
    """
    codec_args = "-c:v libx264 -preset veryfast -tune zerolatency -an" if transcode else "-c copy"
    return (
        f"ffmpeg -nostdin -re -i {shlex.quote(input_url)} "
        f"{codec_args} -f rtsp rtsp://localhost:$RTSP_PORT/$MTX_PATH"
    )


def _resolve_url(url: str, codec: Optional[str] = None) -> IngestPlan:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()

    if scheme in NATIVE_SCHEMES:
        return IngestPlan(tier=IngestTier.NATIVE, reason=f"MediaMTX pulls {scheme}:// natively")

    if scheme in ("http", "https"):
        if any(parsed.path.lower().endswith(sfx) for sfx in HLS_SUFFIXES):
            return IngestPlan(tier=IngestTier.NATIVE, reason="HLS playlist, pulled natively")
        # Anything else over http(s) that is a continuous stream (MJPEG being the
        # common case) needs ffmpeg in front of it.
        transcode = codec is not None and codec.lower() not in PASSTHROUGH_CODECS
        return IngestPlan(
            tier=IngestTier.FFMPEG,
            run_on_init=build_ffmpeg_ingest(url, transcode=transcode),
            transcoding=transcode,
            reason="ffmpeg ingest (transcoding)" if transcode else "ffmpeg ingest (stream copy)",
        )

    if scheme in ("file", "") or url.startswith("/dev/"):
        # Local files and V4L2 devices: ffmpeg reads them, MediaMTX cannot.
        transcode = codec is not None and codec.lower() not in PASSTHROUGH_CODECS
        return IngestPlan(
            tier=IngestTier.FFMPEG,
            run_on_init=build_ffmpeg_ingest(url, transcode=transcode),
            transcoding=transcode,
            reason="ffmpeg ingest from a local device or file",
        )

    return IngestPlan(tier=IngestTier.NONE, reason=f"No ingest route for {scheme or 'this'} source")


def resolve_ingest_tier(provider_cls: Any = None, params: Optional[dict] = None) -> IngestPlan:
    """Decide the ingest tier for a source, from its provider class and params.

    The provider class gets the first say via ``ingest_tier()`` — a snapshot poller
    knows it can never be a continuous stream regardless of its URL. Only providers
    that claim a stream-shaped tier get their URL inspected.
    """
    params = params or {}

    declared: Optional[IngestTier] = None
    if provider_cls is not None and hasattr(provider_cls, "ingest_tier"):
        declared = IngestTier(provider_cls.ingest_tier())

    # A provider that is inherently frame-at-a-time can't be talked into a timeline by
    # its URL, so short-circuit before looking at one.
    if declared in (IngestTier.RING, IngestTier.NONE):
        return IngestPlan(
            tier=declared,
            reason=(
                "Snapshot/frame source — no continuous stream to record, " "using an in-RAM frame ring instead"
                if declared is IngestTier.RING
                else "Source cannot be ingested into a stream"
            ),
        )

    url = params.get("video_path") or params.get("url") or params.get("source_url") or ""
    if not url:
        return IngestPlan(tier=IngestTier.NONE, reason="Source has no stream URL")

    return _resolve_url(str(url), codec=params.get("codec"))
