"""Tests for ingest-tier resolution — "can this source be rewound, and at what cost?".

One resolver decides this because the answer drives three things that must never
disagree: how the relay arms the MediaMTX path, which clip backend the watcher uses,
and the four-state buffer badge the UI shows the user. A source that silently resolves
to the wrong tier produces empty or misleadingly choppy clips.
"""

import pytest

from apps.helpers.imageProviders.ffmpegImageProvider import FfmpegImageProvider
from apps.helpers.imageProviders.imageProvider import ImageProvider
from apps.helpers.imageProviders.mqttImageProvider import MqttImageProvider
from apps.helpers.imageProviders.urlImageProvider import UrlImageProvider
from apps.helpers.streams.ingest import (
    IngestTier,
    build_ffmpeg_ingest,
    resolve_ingest_tier,
)


class TestNativeTier:
    @pytest.mark.parametrize(
        "url",
        [
            "rtsp://cam.local/stream",
            "rtsps://cam.local/stream",
            "rtmp://cam.local/live",
            "srt://cam.local:9000",
            "udp://239.0.0.1:1234",
        ],
    )
    def test_schemes_mediamtx_pulls_itself(self, url):
        plan = resolve_ingest_tier(params={"source_url": url})
        assert plan.tier is IngestTier.NATIVE
        # Zero extra processes is the whole point of this tier.
        assert plan.run_on_init is None
        assert plan.transcoding is False
        assert plan.buffered is True

    def test_hls_over_http_is_native(self):
        # HLS is pulled natively, so it does NOT exercise the ffmpeg ingest path.
        plan = resolve_ingest_tier(params={"source_url": "https://cam.local/live/index.m3u8"})
        assert plan.tier is IngestTier.NATIVE


class TestFfmpegTier:
    def test_mjpeg_over_http_needs_an_ffmpeg_ingest(self):
        plan = resolve_ingest_tier(params={"source_url": "http://cam.local/video.mjpg"})
        assert plan.tier is IngestTier.FFMPEG
        assert plan.buffered is True
        assert "ffmpeg" in plan.run_on_init

    def test_v4l2_device_needs_an_ffmpeg_ingest(self):
        plan = resolve_ingest_tier(params={"source_url": "/dev/video0"})
        assert plan.tier is IngestTier.FFMPEG
        assert "/dev/video0" in plan.run_on_init

    def test_passthrough_codec_does_not_transcode(self):
        plan = resolve_ingest_tier(params={"source_url": "http://cam/video.mjpg", "codec": "h264"})
        assert plan.transcoding is False
        assert "-c copy" in plan.run_on_init

    def test_non_passthrough_codec_transcodes_and_says_so(self):
        # A transcode is real per-camera CPU, unlike everything else in this design,
        # so it must be visible rather than a silent cost.
        plan = resolve_ingest_tier(params={"source_url": "http://cam/video.mjpg", "codec": "mjpeg"})
        assert plan.transcoding is True
        assert "libx264" in plan.run_on_init
        assert "transcoding" in plan.reason


class TestBuildFfmpegIngest:
    def test_uses_run_on_init_target_variables(self):
        cmd = build_ffmpeg_ingest("http://cam/video.mjpg")
        # MediaMTX expands these when it spawns the process.
        assert "$RTSP_PORT" in cmd and "$MTX_PATH" in cmd

    def test_quotes_the_input_url(self):
        # URLs carry query strings and credentials; an unquoted one would be split by
        # the shell MediaMTX runs the hook through.
        cmd = build_ffmpeg_ingest("http://cam/video?user=a&pass=b c")
        assert "'http://cam/video?user=a&pass=b c'" in cmd

    def test_copy_is_the_default(self):
        assert "-c copy" in build_ffmpeg_ingest("rtsp://cam/s")


class TestRingAndNoneTiers:
    def test_snapshot_provider_gets_a_ring_regardless_of_url(self):
        # A still-image camera can't be talked into a timeline by its URL.
        plan = resolve_ingest_tier(UrlImageProvider, params={"url": "http://cam/snapshot.jpg"})
        assert plan.tier is IngestTier.RING
        assert plan.buffered is False

    def test_mqtt_provider_gets_a_ring(self):
        plan = resolve_ingest_tier(MqttImageProvider, params={"topic": "cam/frames"})
        assert plan.tier is IngestTier.RING

    def test_missing_url_is_none_not_a_crash(self):
        assert resolve_ingest_tier(FfmpegImageProvider, params={}).tier is IngestTier.NONE

    def test_unknown_scheme_is_none(self):
        plan = resolve_ingest_tier(params={"source_url": "carrier-pigeon://cam/stream"})
        assert plan.tier is IngestTier.NONE


class TestProviderDeclarations:
    def test_default_is_the_safe_ring_answer(self):
        # A provider that hasn't thought about it is almost certainly frame-at-a-time,
        # so the default must not claim a rewind buffer it can't deliver.
        assert ImageProvider.ingest_tier() == "ring"

    def test_ffmpeg_provider_defers_to_the_url(self):
        # It covers both native-pullable and ffmpeg-only URLs, so it must not
        # short-circuit resolution.
        assert FfmpegImageProvider.ingest_tier() == "native"
        assert resolve_ingest_tier(FfmpegImageProvider, {"video_path": "rtsp://c/s"}).tier is IngestTier.NATIVE
        assert resolve_ingest_tier(FfmpegImageProvider, {"video_path": "http://c/v.mjpg"}).tier is IngestTier.FFMPEG

    def test_video_path_param_name_is_understood(self):
        # FfmpegImageProvider names its URL param `video_path`, not `url`.
        plan = resolve_ingest_tier(FfmpegImageProvider, {"video_path": "rtsp://cam/stream"})
        assert plan.tier is IngestTier.NATIVE
