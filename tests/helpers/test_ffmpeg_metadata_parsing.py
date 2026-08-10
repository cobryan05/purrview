"""Regression tests for FFmpegStreamerIn's stderr metadata parser.

The parser has to pull width/height/fps/pixel format out of ffmpeg's `Stream #...
Video:` banner. Getting it wrong is not a cosmetic failure: `wait_for_metadata()`
never resolves, so a *live* stream hangs forever rather than erroring — the loop only
exits at stderr EOF, which never comes for a camera.

The case that broke it in practice: ffmpeg annotates the resolution with
`[SAR 1:1 DAR 4:3]`, and the old parser rejected any comma-part containing a space,
so the resolution was skipped and every RTSP source hung on start.
"""

import pytest

from apps.helpers.streams.ffmpegStreamerIn import parse_video_stream_banner

# Real banners, copied verbatim from ffmpeg 4.4 output in this container.
RTSP_H264_INPUT = (
    "Stream #0:0: Video: h264 (Constrained Baseline), yuv420p(progressive), "
    "640x480 [SAR 1:1 DAR 4:3], 10 fps, 10 tbr, 90k tbn, 20 tbc"
)
RAWVIDEO_OUTPUT = (
    "Stream #0:0: Video: rawvideo (RGB[24] / 0x18424752), rgb24(pc, gbr/unknown/unknown, progressive), "
    "640x480 [SAR 1:1 DAR 4:3], q=2-31, 73728 kb/s, 10 fps, 10 tbn"
)
NO_SAR = "Stream #0:0: Video: h264 (High), yuv420p, 1920x1080, 25 fps, 25 tbr, 90k tbn"
TBR_ONLY = "Stream #0:0: Video: h264 (Main), yuvj420p(pc), 1280x720 [SAR 1:1 DAR 16:9], 30 tbr, 90k tbn"


class TestParseVideoStreamBanner:
    def test_parses_a_banner_with_sar_and_dar(self):
        """The regression: SAR/DAR annotation must not hide the resolution."""
        meta = parse_video_stream_banner(RTSP_H264_INPUT)
        assert meta is not None
        assert (meta.width, meta.height) == (640, 480)
        assert meta.fps == 10.0
        assert meta.pixel_format == "yuv420p"

    def test_parses_the_rawvideo_output_banner(self):
        # `0x18424752` contains an 'x' and must not be mistaken for a resolution.
        meta = parse_video_stream_banner(RAWVIDEO_OUTPUT)
        assert meta is not None
        assert (meta.width, meta.height) == (640, 480)
        assert meta.pixel_format == "rgb24"
        assert meta.fps == 10.0

    def test_parses_a_banner_without_sar(self):
        meta = parse_video_stream_banner(NO_SAR)
        assert meta is not None
        assert (meta.width, meta.height) == (1920, 1080)
        assert meta.fps == 25.0

    def test_falls_back_to_tbr_when_fps_is_absent(self):
        # Many IP cameras report only tbr. Without this the stream would hang waiting
        # for an fps figure that never comes.
        meta = parse_video_stream_banner(TBR_ONLY)
        assert meta is not None
        assert (meta.width, meta.height) == (1280, 720)
        assert meta.fps == 30.0

    def test_ignores_non_video_lines(self):
        assert parse_video_stream_banner("Stream #0:1: Audio: aac, 48000 Hz, stereo") is None
        assert parse_video_stream_banner("  Duration: N/A, start: 0.000000, bitrate: N/A") is None

    def test_a_video_line_with_no_resolution_is_rejected(self):
        # Better to keep reading stderr than to invent a frame size — a wrong size
        # would silently misalign every frame read off the pipe.
        assert parse_video_stream_banner("Stream #0:0: Video: h264 (High), yuv420p") is None

    @pytest.mark.parametrize(
        "banner,expected",
        [
            (RTSP_H264_INPUT, 640 * 480 * 3),
            (NO_SAR, 1920 * 1080 * 3),
        ],
    )
    def test_frame_size_is_rgb24_bytes(self, banner, expected):
        assert parse_video_stream_banner(banner).rgb24_frame_size == expected
