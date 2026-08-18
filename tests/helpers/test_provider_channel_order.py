"""Every image provider must hand out **BGR**, because everything downstream assumes it.

There is no marker on a numpy frame saying which order its channels are in, so a
provider that disagrees with the rest of the app produces no error anywhere — just
quietly worse detections and blue cats in saved images. `FfmpegImageProvider` requested
`rgb24` from ffmpeg and therefore did exactly that, from its first commit.

The contract, and who relies on it:

  * `cv2.imencode` / `cv2.imwrite` interpret channel 0 as blue. Everything written to
    disk (`CollectTrainingImagesTask._save_image`) and everything sent to inference
    (`base64_encode_jpg`) goes through one of them.
  * `onnxYoloInferenceProvider` decodes that JPEG and runs `cvtColor(BGR2RGB)` before
    the model — so an RGB-ordered frame reaches the model with red and blue swapped,
    while the models were trained on correctly-ordered images.

A test that only checked "a frame comes back" passes in every one of those cases, which
is why these assert on a *known colour* instead.
"""

import io
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from apps.helpers.imageProviders.ffmpegImageProvider import FfmpegImageProvider

# Pure red, in the BGR order the rest of the app speaks.
RED_BGR = (0, 0, 255)


def _red_png_bytes() -> bytes:
    ok, buffer = cv2.imencode(".png", np.full((8, 8, 3), RED_BGR, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


class TestFfmpegProviderChannelOrder:
    def test_requests_bgr_from_ffmpeg(self):
        """The pixel format asked of ffmpeg *is* the channel order of every frame.

        Asserted at the argument rather than on a decoded frame because the raw pipe
        carries whatever ffmpeg was told to emit — there is nothing further downstream
        that could correct or even detect a wrong choice here.
        """
        provider = FfmpegImageProvider(video_path="/tmp/does-not-need-to-exist.mp4", rtsp_relay=False)

        assert provider._ffmpeg_args["pix_fmt"] == "bgr24"

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg binary")
    async def test_a_red_video_decodes_to_red(self, tmp_path):
        """End to end against real ffmpeg, because the unit test above only checks a string.

        This is the test that would have caught the original bug: it compares the
        provider's frame against `cv2.VideoCapture` on the same file, which is BGR by
        definition, so the two agreeing is the whole assertion.
        """
        video = tmp_path / "red.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x48:d=1:r=5",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            check=True,
        )

        capture = cv2.VideoCapture(str(video))
        ok, reference = capture.read()
        capture.release()
        assert ok

        provider = FfmpegImageProvider(video_path=str(video), loop=False, rtsp_relay=False)
        await provider.start()
        try:
            provided = await provider.getNextImage()
        finally:
            await provider.stop()

        assert provided is not None
        pixel = provided.image[0, 0]
        # Channel 2 is red in BGR; a swapped provider puts the 250-ish value in channel 0.
        assert pixel[2] > 200 and pixel[0] < 50, f"expected BGR red, got {pixel}"
        np.testing.assert_allclose(pixel, reference[0, 0], atol=8)


class TestMqttProviderChannelOrder:
    async def test_reverses_pils_rgb_into_bgr(self):
        """PIL decodes to RGB; the provider's `[..., ::-1]` is what makes it BGR.

        Named `frame_rgb` in the source, which reads like the opposite of what it does —
        hence a test pinning the behaviour rather than the name.
        """
        from apps.helpers.imageProviders.mqttImageProvider import MqttImageProvider

        provider = MqttImageProvider.__new__(MqttImageProvider)
        import asyncio

        provider._frameQueue = asyncio.Queue()
        provider._loop = asyncio.get_running_loop()

        from PIL import Image

        decoded = np.array(Image.open(io.BytesIO(_red_png_bytes())))
        # Sanity: PIL really does hand back RGB, so the reversal below is load-bearing.
        assert tuple(decoded[0, 0]) == (255, 0, 0)
        await provider._frameQueue.put(decoded)

        provided = await provider.getNextImage()

        assert tuple(provided.image[0, 0]) == RED_BGR


class TestSnapshotActionChannelOrder:
    async def test_writes_the_frame_it_was_given_without_swapping(self, tmp_path):
        """The snapshot action must not "correct" a frame that is already BGR.

        It used to convert RGB→BGR on the way out, which was right only for the one
        provider that was itself wrong and blue-shifted every snapshot from the other
        three.
        """
        from unittest.mock import patch

        from apps.helpers.actions import outputDir
        from apps.helpers.actions.action import ActionContext
        from apps.helpers.actions.saveSnapshotAction import SaveSnapshotAction
        from apps.helpers.watch.events import EventPhase, WatchEventSignal

        frame = np.full((8, 8, 3), RED_BGR, dtype=np.uint8)
        # A real signal, not a stub: the output folder is a template now, and the
        # renderer reads slots a hand-rolled namespace does not have.
        signal = WatchEventSignal(
            rule_uuid="r1",
            rule_name="cat_in_box",
            phase=EventPhase.START,
            at=10.0,
            started_at=10.0,
            source_uuid="s1",
        )
        context = ActionContext(
            profile_uuid="p1",
            profile_name="Litter box",
            source_uuid="s1",
            source_name="Litter cam",
            frame=frame,
        )

        # The dataset root the action resolves against, so this writes into the test's
        # tmp dir rather than the container's real image_datasets.
        with patch.object(outputDir, "FILES_ROOT", tmp_path):
            action = SaveSnapshotAction(output_dir="snaps")
            result = await action.run(signal, context)

        assert result.ok, result.detail
        written = cv2.imread(result.data["path"])
        pixel = written[0, 0]
        assert pixel[2] > 200 and pixel[0] < 50, f"expected BGR red on disk, got {pixel}"
