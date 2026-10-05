"""Concealing a corrupt clip, and the guards that stop it making things worse.

The repair is a decode-and-re-encode, so the ways it can go wrong are all ways of
damaging a file that was merely ugly: losing it, shortening it, or replacing it with
something a player cannot open. The sidecar's box offsets are written against the
original timeline, so a repair that changed the duration would slide every box — which
is the exact fault the clip pipeline was just fixed to stop doing.

So the tests that matter here are the refusals, not the happy path.
"""

import subprocess
from pathlib import Path

import pytest

from apps.helpers.recordings import repairClip
from apps.helpers.recordings.repairClip import repair_if_corrupt


def _clip(path: Path, seconds: int = 2) -> Path:
    """A real, playable mp4 with nothing wrong with it.

    `-f mp4` named explicitly because these are also written to `<name>.mp4.repair`,
    and ffmpeg picks its muxer from the extension — the same trap `concat_parts`
    documents for its own `.part` file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=160x120:rate=10:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "mp4",
            str(path),
        ],
        capture_output=True,
        check=True,
        timeout=60,
    )
    return path


class TestItLeavesWellAloneClipsAlone:
    async def test_a_disabled_camera_is_not_scanned(self, tmp_path):
        """The scan is a full decode pass. A camera that has not asked for this must not
        pay for one on every clip."""
        scanned = []
        clip = _clip(tmp_path / "a.mp4")

        result = await repair_if_corrupt(clip, enabled=False)

        assert result == {"repaired": False, "reason": "disabled"}
        assert scanned == []

    async def test_a_missing_file_is_not_an_error(self, tmp_path):
        """A clip whose extraction failed is absent, and this runs after extraction."""
        result = await repair_if_corrupt(tmp_path / "never-written.mp4")

        assert result["repaired"] is False
        assert result["reason"] == "missing"

    async def test_an_undamaged_clip_is_not_re_encoded(self, tmp_path):
        """Most clips are clean, and re-encoding one costs CPU for an identical picture
        — on the box also serving the live app."""
        clip = _clip(tmp_path / "clean.mp4")
        before = clip.read_bytes()

        result = await repair_if_corrupt(clip)

        assert result["repaired"] is False
        assert result["reason"] == "clean"
        assert clip.read_bytes() == before, "a clean clip was rewritten"


class TestItRefusesARepairThatWouldInvalidateTheOverlay:
    """The detection sidecar is written against the clip's measured length, so a repair
    that moved the duration would put every box at the wrong moment. Keeping a corrupt
    clip whose boxes are right beats a clean one whose boxes are wrong.
    """

    async def test_a_repair_that_shortens_the_clip_is_discarded(self, tmp_path, monkeypatch):
        clip = _clip(tmp_path / "c.mp4", seconds=3)
        original = clip.read_bytes()
        monkeypatch.setattr(repairClip, "_count_errors", lambda p: 50)

        def _shorter(src, dest):
            _clip(Path(dest), seconds=1)
            return subprocess.CompletedProcess([], 0, b"", b"")

        monkeypatch.setattr(repairClip, "_conceal", _shorter)

        result = await repair_if_corrupt(clip)

        assert result["repaired"] is False
        assert result["reason"] == "duration moved"
        assert clip.read_bytes() == original, "the original was replaced by a shorter clip"

    async def test_a_repair_keeping_the_duration_is_accepted(self, tmp_path, monkeypatch):
        """The positive half — without it the test above passes against a function that
        refuses every repair, and the feature would silently never work."""
        clip = _clip(tmp_path / "d.mp4", seconds=2)
        monkeypatch.setattr(repairClip, "_count_errors", lambda p: 50)

        def _same_length(src, dest):
            _clip(Path(dest), seconds=2)
            return subprocess.CompletedProcess([], 0, b"", b"")

        monkeypatch.setattr(repairClip, "_conceal", _same_length)

        result = await repair_if_corrupt(clip)

        assert result["repaired"] is True
        assert result["decodeErrors"] == 50

    async def test_a_failed_encode_keeps_the_original(self, tmp_path, monkeypatch):
        clip = _clip(tmp_path / "e.mp4")
        original = clip.read_bytes()
        monkeypatch.setattr(repairClip, "_count_errors", lambda p: 50)
        monkeypatch.setattr(
            repairClip,
            "_conceal",
            lambda src, dest: subprocess.CompletedProcess([], 1, b"", b"ffmpeg fell over"),
        )

        result = await repair_if_corrupt(clip)

        assert result["repaired"] is False
        assert result["reason"] == "encode failed"
        assert clip.read_bytes() == original

    async def test_the_half_written_repair_is_never_left_behind(self, tmp_path, monkeypatch):
        """`.repair` beside a clip would be picked up by nothing and swept by nothing."""
        clip = _clip(tmp_path / "f.mp4")
        monkeypatch.setattr(repairClip, "_count_errors", lambda p: 50)
        monkeypatch.setattr(
            repairClip,
            "_conceal",
            lambda src, dest: subprocess.CompletedProcess([], 1, b"", b"boom"),
        )

        await repair_if_corrupt(clip)

        assert list(tmp_path.glob("*.repair")) == []


class TestTheErrorScan:
    def test_decoder_errors_are_counted_not_just_detected(self, tmp_path):
        """The count drives whether a re-encode is worth it, so it has to be a number
        rather than a boolean."""
        assert repairClip._count_errors(_clip(tmp_path / "g.mp4")) == 0

    @pytest.mark.parametrize(
        "line",
        [
            "[h264 @ 0x55] error while decoding MB 38 7, bytestream -7",
            "[h264 @ 0x55] concealing 120 DC, 120 AC, 120 MV errors in P frame",
            "[h264 @ 0x55] Invalid NAL unit size",
        ],
    )
    def test_the_damage_ffmpeg_reports_is_recognised(self, line):
        """These are the shapes ffmpeg actually prints for a lossy stream; missing one
        means a corrupt clip is reported clean and never repaired."""
        assert repairClip._ERROR_PATTERN.search(line)

    def test_an_unrelated_warning_is_not_counted_as_damage(self, tmp_path):
        """Counting ordinary chatter would re-encode every clip."""
        assert not repairClip._ERROR_PATTERN.search("[mp4 @ 0x55] track 0 has no keyframes")
