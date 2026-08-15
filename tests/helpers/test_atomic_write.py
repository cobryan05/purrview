"""Tests for atomic_write_bytes — the landing path for model files.

Model files are written by one process (tasks_server, on retrain) and read by
another (inference_server, which stats and reloads on every inference). A plain
write_bytes() is therefore readable as a truncated ONNX. These tests pin the
property that makes that impossible: at no instant does the destination path
hold anything other than the complete old bytes or the complete new bytes.
"""

import os
from glob import glob
from pathlib import Path

import pytest

from apps.helpers.metadataUtils import atomic_write_bytes


class TestAtomicWriteBytes:
    def test_writes_new_file(self, tmp_path):
        dest = tmp_path / "m.onnx"
        atomic_write_bytes(dest, b"weights")

        assert dest.read_bytes() == b"weights"

    def test_creates_missing_parent_dirs(self, tmp_path):
        dest = tmp_path / "nested" / "deeper" / "m.onnx"
        atomic_write_bytes(dest, b"weights")

        assert dest.read_bytes() == b"weights"

    def test_overwrites_existing_file(self, tmp_path):
        dest = tmp_path / "m.onnx"
        dest.write_bytes(b"old weights")

        atomic_write_bytes(dest, b"new weights")

        assert dest.read_bytes() == b"new weights"

    def test_leaves_no_temp_files_behind(self, tmp_path):
        dest = tmp_path / "m.onnx"
        atomic_write_bytes(dest, b"weights")

        assert [p.name for p in tmp_path.iterdir()] == ["m.onnx"]

    def test_temp_file_is_not_visible_to_the_model_glob(self, tmp_path, monkeypatch):
        """Manager.list_models() globs `*.onnx`. If the in-flight temp file matched
        that pattern it would be advertised as a real model — and then loaded — for
        the whole duration of the write, which is the bug this is meant to prevent."""
        dest = tmp_path / "m.onnx"
        seen: list = []

        real_replace = os.replace

        def _spy_replace(src, dst):
            # Mid-write: the temp exists, the destination does not yet.
            seen.append(sorted(Path(p).name for p in glob(str(tmp_path / "*.onnx"))))
            return real_replace(src, dst)

        monkeypatch.setattr("apps.helpers.metadataUtils.os.replace", _spy_replace)
        atomic_write_bytes(dest, b"weights")

        assert seen == [[]], f"a *.onnx path was visible mid-write: {seen}"

    def test_destination_never_holds_partial_bytes(self, tmp_path, monkeypatch):
        """The destination goes straight from the old contents to the new ones."""
        dest = tmp_path / "m.onnx"
        dest.write_bytes(b"old weights")
        observed: list = []

        real_replace = os.replace

        def _spy_replace(src, dst):
            observed.append(dest.read_bytes())
            return real_replace(src, dst)

        monkeypatch.setattr("apps.helpers.metadataUtils.os.replace", _spy_replace)
        atomic_write_bytes(dest, b"new weights")

        assert observed == [b"old weights"], "destination was mutated before the swap"
        assert dest.read_bytes() == b"new weights"

    def test_mtime_moves_so_the_inference_server_notices(self, tmp_path):
        """The reload check in inference_server keys on mtime_ns — an atomic swap
        must still bump it, or the fix it supports stops firing."""
        dest = tmp_path / "m.onnx"
        dest.write_bytes(b"old weights")
        before = dest.stat().st_mtime_ns
        os.utime(dest, ns=(before - 1_000_000_000, before - 1_000_000_000))
        before = dest.stat().st_mtime_ns

        atomic_write_bytes(dest, b"new weights")

        assert dest.stat().st_mtime_ns > before

    def test_failed_write_leaves_the_old_file_intact(self, tmp_path, monkeypatch):
        dest = tmp_path / "m.onnx"
        dest.write_bytes(b"old weights")

        monkeypatch.setattr(
            "apps.helpers.metadataUtils.os.replace",
            lambda src, dst: (_ for _ in ()).throw(OSError("disk full")),
        )

        with pytest.raises(OSError):
            atomic_write_bytes(dest, b"new weights")

        assert dest.read_bytes() == b"old weights"
        assert [p.name for p in tmp_path.iterdir()] == ["m.onnx"], "temp file was not cleaned up"

    def test_concurrent_writes_do_not_share_a_temp_path(self, tmp_path, monkeypatch):
        """Two retrains landing the same model name must not corrupt each other by
        writing through one shared temp file."""
        dest = tmp_path / "m.onnx"
        temps: list = []

        real_replace = os.replace

        def _spy_replace(src, dst):
            temps.append(str(src))
            return real_replace(src, dst)

        monkeypatch.setattr("apps.helpers.metadataUtils.os.replace", _spy_replace)
        atomic_write_bytes(dest, b"a")
        atomic_write_bytes(dest, b"b")

        assert temps[0] != temps[1]
