"""Noticing a camera whose recorded clips stop holding continuous video.

The failure this guards against still produces a playable file — audio running on under
a frozen picture, or no picture at all — so every test here uses a real mp4 built by
ffmpeg and measured by the real ffprobe.
"""

import subprocess
from pathlib import Path

import pytest

from apps.helpers.recordings import clipVideoHealth


@pytest.fixture(autouse=True)
def _fresh():
    clipVideoHealth.reset()
    yield
    clipVideoHealth.reset()


def _clip(path: Path, *, video: bool = True, audio: bool = False, hole_s: float = 0.0) -> Path:
    """A 6 s mp4; with `hole_s`, every frame from the third second on is pushed later."""
    args = ["ffmpeg", "-nostdin", "-y", "-v", "error"]
    if video:
        args += ["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=6"]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=6"]
    if video:
        if hole_s:
            args += ["-vf", f"setpts=PTS+if(gte(N\\,20)\\,{hole_s}/TB\\,0)", "-vsync", "passthrough"]
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac"]
    subprocess.run(args + ["-f", "mp4", str(path)], capture_output=True, check=True, timeout=60)
    return path


class TestMeasuringAClip:
    async def test_continuous_video_is_healthy(self, tmp_path):
        report = await clipVideoHealth.probe(_clip(tmp_path / "ok.mp4", audio=True))
        assert not clipVideoHealth.is_damaged(report)

    async def test_a_hole_in_the_video_is_found_and_measured(self, tmp_path):
        report = await clipVideoHealth.probe(_clip(tmp_path / "hole.mp4", audio=True, hole_s=5))
        assert report["holes"] == 1
        assert report["longestHoleS"] == pytest.approx(5.1, abs=0.2)
        assert report["hasAudio"] is True

    async def test_a_short_stutter_is_not_a_hole(self, tmp_path):
        """A Wi-Fi hiccup of a second or so must not raise a standing warning."""
        report = await clipVideoHealth.probe(_clip(tmp_path / "stutter.mp4", hole_s=1))
        assert not clipVideoHealth.is_damaged(report)

    async def test_an_audio_only_clip_has_no_video(self, tmp_path):
        report = await clipVideoHealth.probe(_clip(tmp_path / "audio.mp4", video=False, audio=True))
        assert report["missingVideo"] is True

    async def test_an_unreadable_file_is_no_verdict(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not an mp4")
        assert await clipVideoHealth.probe(bad) is None


class TestTheStandingWarning:
    async def test_a_damaged_clip_raises_it_and_a_clean_one_clears_it(self, tmp_path):
        await clipVideoHealth.check(_clip(tmp_path / "a.mp4", video=False, audio=True), "src-1", "Closet")
        assert [f["sourceUuid"] for f in clipVideoHealth.findings()] == ["src-1"]

        await clipVideoHealth.check(_clip(tmp_path / "b.mp4"), "src-1", "Closet")
        assert clipVideoHealth.findings() == []

    async def test_one_cameras_clean_clip_does_not_clear_another(self, tmp_path):
        await clipVideoHealth.check(_clip(tmp_path / "a.mp4", hole_s=5), "src-1")
        await clipVideoHealth.check(_clip(tmp_path / "b.mp4"), "src-2")
        assert [f["sourceUuid"] for f in clipVideoHealth.findings()] == ["src-1"]

    async def test_the_advice_follows_whether_audio_kept_recording(self, tmp_path):
        """Audio running under missing video points at the camera's audio track; no
        audio either points at the camera going away."""
        await clipVideoHealth.check(_clip(tmp_path / "a.mp4", audio=True, hole_s=5), "with-audio")
        await clipVideoHealth.check(_clip(tmp_path / "b.mp4", hole_s=5), "silent")
        notices = {f["sourceUuid"]: clipVideoHealth.notice_for(f)["message"] for f in clipVideoHealth.findings()}

        assert "audio track" in notices["with-audio"]
        assert "audio track" not in notices["silent"]
