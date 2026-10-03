"""`/api/recordings/{uuid}/export` — the annotated download."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from apps.helpers.db.types import Recording

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_clip(dest: Path, seconds=1.0):
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=64x48:r=10:d={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ],
        capture_output=True,
        check=True,
        timeout=120,
    )


def _sidecar(dest: Path):
    dest.write_text(
        json.dumps(
            {
                "version": 2,
                "maxStalenessS": 2.0,
                "durationS": 1.0,
                "event": {"startS": 0.0, "endS": 1.0, "ruleName": "cat-in-yard"},
                "frames": [
                    {
                        "t": 0.0,
                        "boxes": [
                            {"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3, "label": "Cat", "labelUuid": "u-1", "conf": 0.8}
                        ],
                    }
                ],
            }
        )
    )


class TestExportRecording:
    async def test_unknown_recording_is_404(self, http_client):
        res = await http_client.get("/api/recordings/nope/export")
        assert res.status_code == 404

    async def test_a_clip_without_an_overlay_is_404(self, http_client, db_client, tmp_path):
        # The export is only offered where the player found a sidecar, so this is the
        # caller asking for a representation that does not exist — not a server fault.
        clip = tmp_path / "bare.mp4"
        clip.write_bytes(b"not really an mp4")
        recording = await db_client.create_recording(Recording(kind="clip", file_path=str(clip), status="complete"))
        res = await http_client.get(f"/api/recordings/{recording.uuid}/export")
        assert res.status_code == 404

    async def test_a_clip_whose_file_is_gone_is_404_not_a_render_failure(self, http_client, db_client, tmp_path):
        overlay = tmp_path / "ghost.mp4.boxes.json"
        _sidecar(overlay)
        recording = await db_client.create_recording(
            Recording(kind="clip", file_path=str(tmp_path / "ghost.mp4"), overlay_path=str(overlay), status="complete")
        )
        res = await http_client.get(f"/api/recordings/{recording.uuid}/export")
        assert res.status_code == 404

    @pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe are not on PATH")
    async def test_the_annotated_clip_comes_back_as_a_playable_mp4(self, http_client, db_client, tmp_path):
        clip = tmp_path / "clip.mp4"
        _make_clip(clip)
        overlay = tmp_path / "clip.mp4.boxes.json"
        _sidecar(overlay)
        recording = await db_client.create_recording(
            Recording(kind="clip", file_path=str(clip), overlay_path=str(overlay), status="complete")
        )

        res = await http_client.get(f"/api/recordings/{recording.uuid}/export")
        assert res.status_code == 200
        assert res.headers["content-type"] == "video/mp4"
        body = res.content
        # A render that failed halfway would still be a 200 with a short body, and the
        # rendered file is necessarily larger than the black source it was drawn over.
        assert len(body) > clip.stat().st_size

        out = tmp_path / "downloaded.mp4"
        out.write_bytes(body)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
            capture_output=True,
            timeout=60,
        )
        assert probe.returncode == 0
        assert float(probe.stdout.strip()) == pytest.approx(1.0, abs=0.3)

    @pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe are not on PATH")
    async def test_the_rendered_file_is_not_left_behind(self, http_client, db_client, tmp_path, monkeypatch):
        # The render goes to a temp directory and the response deletes it once sent.
        # Without that every download leaks a whole clip into /tmp, which on this box
        # is the disk the recordings themselves live on.
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("TMPDIR", str(scratch))
        import tempfile

        monkeypatch.setattr(tempfile, "tempdir", str(scratch))

        clip = tmp_path / "clip.mp4"
        _make_clip(clip)
        overlay = tmp_path / "clip.mp4.boxes.json"
        _sidecar(overlay)
        recording = await db_client.create_recording(
            Recording(kind="clip", file_path=str(clip), overlay_path=str(overlay), status="complete")
        )

        res = await http_client.get(f"/api/recordings/{recording.uuid}/export")
        assert res.status_code == 200
        assert list(scratch.iterdir()) == []
