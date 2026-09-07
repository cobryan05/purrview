"""Tests for TrainModelTask._save_artifact — how retrained weights land on disk.

Everywhere else in the suite _save_artifact is patched out, so the landing itself
was untested. It matters because the inference server reads these files from a
different process while a detection task is live: it stats the .onnx on every
inference and rebuilds its session when the mtime moves. That makes the write
order and the write atomicity observable behaviour, not implementation detail.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.tasks_server.tasks.TrainModelTask import TrainModelTask

_MOCK_MANAGER = MagicMock()
_MOCK_MANAGER.update_task_heartbeat = AsyncMock()


def _tar(files: dict) -> bytes:
    """Build an artifact tar. dict order becomes tar member order, so a test can
    put the .onnx first and prove _save_artifact reorders it."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _make_task() -> TrainModelTask:
    task = TrainModelTask(task_config_uuid="cfg-uuid", params={}, manager=_MOCK_MANAGER)
    task._training_base_url = "http://training:8005"
    task.log = MagicMock()
    task._push_artifact_to_inference_server = AsyncMock()
    return task


def _patch_download(tar_bytes: bytes):
    """Stand in for the artifact GET."""
    response = MagicMock()
    response.content = tar_bytes
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return patch("apps.tasks_server.tasks._training_job_runner.httpx.AsyncClient", return_value=ctx)


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("apps.tasks_server.tasks._training_job_runner._MODELS_DIR", tmp_path)
    return tmp_path


class TestSaveArtifact:
    async def test_writes_onnx_and_sidecar(self, models_dir):
        task = _make_task()
        with _patch_download(_tar({"m.onnx": b"weights", "m.json": b"{}"})):
            await task._save_artifact("job-1")

        assert (models_dir / "m.onnx").read_bytes() == b"weights"
        assert (models_dir / "m.json").read_bytes() == b"{}"

    async def test_skips_unexpected_files(self, models_dir):
        task = _make_task()
        with _patch_download(_tar({"m.onnx": b"weights", "notes.txt": b"hi"})):
            await task._save_artifact("job-1")

        assert not (models_dir / "notes.txt").exists()

    async def test_overwrites_previous_weights(self, models_dir):
        (models_dir / "m.onnx").write_bytes(b"old weights")
        task = _make_task()
        with _patch_download(_tar({"m.onnx": b"new weights"})):
            await task._save_artifact("job-1")

        assert (models_dir / "m.onnx").read_bytes() == b"new weights"

    async def test_onnx_lands_last(self, models_dir):
        """The inference server reloads on the .onnx's mtime alone, so the sidecar
        and any external-data tensors must already be on disk when it moves.
        Otherwise the reload picks up new weights with the previous class list."""
        task = _make_task()
        order: list = []

        import apps.tasks_server.tasks._training_job_runner as runner

        real = runner.atomic_write_bytes
        with patch.object(
            runner,
            "atomic_write_bytes",
            side_effect=lambda dest, data: (order.append(Path(dest).name), real(dest, data))[1],
        ):
            # .onnx deliberately first in the tar, and note that a plain name sort
            # would also place "m.onnx" before "m.onnx.data".
            with _patch_download(_tar({"m.onnx": b"weights", "m.onnx.data": b"tensors", "m.json": b"{}"})):
                await task._save_artifact("job-1")

        assert order[-1] == "m.onnx", f"the .onnx must land last, got order {order}"
        assert set(order) == {"m.onnx", "m.onnx.data", "m.json"}

    async def test_writes_are_atomic(self, models_dir):
        """A live detection task can read this file mid-write, so it must go through
        the temp-and-rename path rather than a plain write_bytes()."""
        task = _make_task()

        import apps.tasks_server.tasks._training_job_runner as runner

        with patch.object(runner, "atomic_write_bytes", wraps=runner.atomic_write_bytes) as spy:
            with _patch_download(_tar({"m.onnx": b"weights", "m.json": b"{}"})):
                await task._save_artifact("job-1")

        written = sorted(Path(call.args[0]).name for call in spy.call_args_list)
        assert written == ["m.json", "m.onnx"]

    async def test_no_temp_files_survive(self, models_dir):
        task = _make_task()
        with _patch_download(_tar({"m.onnx": b"weights", "m.json": b"{}"})):
            await task._save_artifact("job-1")

        assert sorted(p.name for p in models_dir.iterdir()) == ["m.json", "m.onnx"]


class TestSaveArtifactPushesToInferenceServer:
    """_save_artifact's write to $MODELS_DIR is this host's own copy (tasks_server
    loads models directly for classification); inference_server only ever learns
    about a retrained model through the push. See test_inference_push.py for the
    push mechanics themselves (success/failure handling, tar contents)."""

    async def test_pushes_the_same_tar_it_downloaded(self, models_dir):
        task = TrainModelTask(task_config_uuid="cfg-uuid", params={}, manager=_MOCK_MANAGER)
        task._training_base_url = "http://training:8005"
        task.log = MagicMock()
        task._push_artifact_to_inference_server = AsyncMock()
        tar_bytes = _tar({"m.onnx": b"weights", "m.json": b"{}"})

        with _patch_download(tar_bytes):
            await task._save_artifact("job-1")

        task._push_artifact_to_inference_server.assert_awaited_once_with(tar_bytes)

    async def test_colocated_inference_server_is_not_pushed_to(self):
        """In the single-container deployment the write to $MODELS_DIR already *is*
        inference_server's copy, so posting the same tar back re-writes files it has
        — and a transient HTTP failure would error a finished training run."""
        task = TrainModelTask(task_config_uuid="cfg-uuid", params={}, manager=_MOCK_MANAGER)
        task.log = MagicMock()
        manager = MagicMock()
        manager.get_inference_api_client.return_value = MagicMock(configuration=MagicMock(host="http://localhost:8001"))
        task._manager = manager

        with patch("apps.tasks_server.tasks._training_job_runner.push_artifact_tar_to_inference_server") as push:
            await task._push_artifact_to_inference_server(b"tar")

        push.assert_not_called()

    async def test_remote_inference_server_is_pushed_to(self):
        task = TrainModelTask(task_config_uuid="cfg-uuid", params={}, manager=_MOCK_MANAGER)
        task.log = MagicMock()
        manager = MagicMock()
        manager.get_inference_api_client.return_value = MagicMock(
            configuration=MagicMock(host="http://gpu-box.lan:8001")
        )
        task._manager = manager

        with patch(
            "apps.tasks_server.tasks._training_job_runner.push_artifact_tar_to_inference_server",
            new=AsyncMock(return_value=["m.onnx"]),
        ) as push:
            await task._push_artifact_to_inference_server(b"tar")

        push.assert_awaited_once()
