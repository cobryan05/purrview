"""VAAPI stream decode falls back to the CPU only when the GPU is what failed."""

import asyncio
import shutil
import subprocess

import pytest

from apps.helpers.imageProviders import ffmpegImageProvider as ffmpeg_module
from apps.helpers.imageProviders.ffmpegImageProvider import FfmpegImageProvider, configured_hwaccel_device


@pytest.fixture
def vaapi_on_dev_null(monkeypatch):
    """Configure VAAPI decode on a node that exists but is not a GPU."""
    monkeypatch.setenv("STREAM_HWACCEL", "vaapi")
    monkeypatch.setenv("STREAM_HWACCEL_DEVICE", "/dev/null")
    configured_hwaccel_device.cache_clear()
    yield
    configured_hwaccel_device.cache_clear()


def fake_streamer(created, *, connected, exited):
    """A reader class whose instances report fixed connection state and never yield a frame."""

    class FakeStreamer:
        def __init__(self, url, input_args=None, output_args=None):
            self.input_args = input_args or {}
            created.append(self)

        async def astart(self):
            return None

        async def astop(self):
            return None

        def metadata_ready(self):
            return connected

        def reader_exited(self):
            return exited

        async def wait_for_metadata(self):
            return None

        def get_frame_size(self):
            return (64, 48)

        def is_frame_aligned(self):
            return True

        def read_nowait(self):
            return None

        async def read_async(self, timeout=None):
            await asyncio.sleep(3600)

    return FakeStreamer


class TestVaapiFallback:
    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
    async def test_a_gpu_that_cannot_decode_still_yields_frames(self, tmp_path, vaapi_on_dev_null):
        video = tmp_path / "clip.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=64x48:d=1:r=5", "-pix_fmt", "yuv420p"]
            + [str(video)],
            check=True,
        )

        provider = FfmpegImageProvider(video_path=str(video), loop=False, rtsp_relay=False)
        assert provider._hwaccel_device == "/dev/null"
        await provider.start()
        try:
            provided = await provider.getNextImage()
        finally:
            await provider.stop()

        assert provided is not None, "a failed VAAPI reader must be replaced by a CPU one"
        assert provided.image.shape == (48, 64, 3)
        assert provider._hwaccel_device == ""

    async def test_a_source_that_never_connected_keeps_gpu_decode(self, monkeypatch, vaapi_on_dev_null):
        """No geometry means the camera never answered, which says nothing about the GPU."""
        created = []
        monkeypatch.setattr(ffmpeg_module, "FFmpegStreamerIn", fake_streamer(created, connected=False, exited=True))

        provider = FfmpegImageProvider(video_path="rtsp://camera.invalid/stream", rtsp_relay=False)
        await provider.start()
        await provider.reconnect()

        assert len(created) == 2
        assert created[-1].input_args.get("hwaccel") == "vaapi"

    async def test_a_quiet_camera_keeps_gpu_decode(self, monkeypatch, vaapi_on_dev_null):
        """A running reader that times out waiting for a frame is not a GPU failure."""
        created = []
        monkeypatch.setattr(ffmpeg_module, "FFmpegStreamerIn", fake_streamer(created, connected=True, exited=False))
        monkeypatch.setattr(ffmpeg_module, "STREAM_READ_TIMEOUT", 0.05)

        provider = FfmpegImageProvider(video_path="rtsp://camera.invalid/stream", loop=False, rtsp_relay=False)
        await provider.start()
        assert await provider.getNextImage() is None
        await provider.reconnect()

        assert len(created) == 2
        assert created[-1].input_args.get("hwaccel") == "vaapi"

    async def test_one_reader_dying_after_connecting_keeps_gpu_decode(self, monkeypatch, vaapi_on_dev_null):
        """A single exit after the banner can be the network; it takes two in a row."""
        created = []
        monkeypatch.setattr(ffmpeg_module, "FFmpegStreamerIn", fake_streamer(created, connected=True, exited=True))

        provider = FfmpegImageProvider(video_path="rtsp://camera.invalid/stream", rtsp_relay=False)
        await provider.start()
        await provider.reconnect()
        assert created[-1].input_args.get("hwaccel") == "vaapi"

        await provider.reconnect()
        assert created[-1].input_args == {}
