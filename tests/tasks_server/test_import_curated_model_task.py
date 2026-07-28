"""Tests for ImportCuratedModelTask — curated `requires_conversion` cold-start
import (download .pt -> submit convert_model job -> SSE poll -> pull artifact
-> validate-in-memory -> write_onnx_with_sidecar).

Mirrors ConvertModelTask's SSE-mocking style for the submit/push/poll machinery
(shared via TrainingJobRunnerMixin) and ImportModelTask's httpx-mocking style
for the .pt download step. Unlike ConvertModelTask's tests (which patch out
_save_artifact entirely), here we exercise the REAL overridden _save_artifact
with a fabricated artifact tar, since the whole point of this task's override
is the in-memory validate-then-write path — asserting the family-correctness
propagation (sidecar model_family survives from the training server's
conversion sidecar through to the final on-disk sidecar) is the main point of
this suite.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import onnx
import pytest
from onnx import TensorProto, helper

from apps.helpers.curatedModels import CURATED_BASE_MODELS
from apps.tasks_server.tasks.ImportCuratedModelTask import ImportCuratedModelTask

_MOCK_MANAGER = MagicMock()
_MOCK_MANAGER.update_task_heartbeat = AsyncMock()

_CONVERSION_KEY = next(e["key"] for e in CURATED_BASE_MODELS if e.get("requires_conversion"))


def _make_task(params: dict) -> ImportCuratedModelTask:
    return ImportCuratedModelTask(task_config_uuid="cfg-uuid", params=params, manager=_MOCK_MANAGER)


def _build_onnx_bytes() -> bytes:
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph)
    return model.SerializeToString()


_ONNX_BYTES = _build_onnx_bytes()


def _build_artifact_tar(model_name: str, onnx_bytes: bytes, sidecar: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tf:
        onnx_info = tarfile.TarInfo(name=f"{model_name}.onnx")
        onnx_info.size = len(onnx_bytes)
        tf.addfile(onnx_info, io.BytesIO(onnx_bytes))

        sidecar_bytes = json.dumps(sidecar).encode()
        sidecar_info = tarfile.TarInfo(name=f"{model_name}.json")
        sidecar_info.size = len(sidecar_bytes)
        tf.addfile(sidecar_info, io.BytesIO(sidecar_bytes))
    return buf.getvalue()


def _patch_clients():
    return (
        patch("apps.tasks_server.tasks.ImportCuratedModelTask.APPS_CONFIG"),
        patch("apps.tasks_server.tasks.ImportCuratedModelTask.db_client"),
        patch("apps.tasks_server.tasks.ImportCuratedModelTask.training_client"),
    )


async def _init_task(task, resume=None):
    cfg_patch, db_patch, tc_patch = _patch_clients()
    with cfg_patch as mock_cfg, db_patch, tc_patch:
        mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
        await task._init(task._params, resume)


class TestInit:
    async def test_init_resolves_conversion_entry(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ImportCuratedModelTask._MODELS_ROOT", tmp_path)
        task = _make_task({"curated_key": _CONVERSION_KEY, "model_name": "my_model"})
        await _init_task(task)
        assert task._model_name == "my_model"
        assert task._pt_url.startswith("https://")
        assert task._source_path is None

    async def test_unknown_curated_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ImportCuratedModelTask._MODELS_ROOT", tmp_path)
        task = _make_task({"curated_key": "bogus-key", "model_name": "my_model"})
        with pytest.raises(ValueError, match="Unknown curated_key"):
            await _init_task(task)

    async def test_direct_onnx_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ImportCuratedModelTask._MODELS_ROOT", tmp_path)
        direct_key = next(e["key"] for e in CURATED_BASE_MODELS if not e.get("requires_conversion"))
        task = _make_task({"curated_key": direct_key, "model_name": "my_model"})
        with pytest.raises(ValueError, match="does not require conversion"):
            await _init_task(task)

    async def test_invalid_model_name_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ImportCuratedModelTask._MODELS_ROOT", tmp_path)
        task = _make_task({"curated_key": _CONVERSION_KEY, "model_name": "../escape"})
        with pytest.raises(ValueError, match="Invalid model_name"):
            await _init_task(task)

    async def test_overwrite_false_existing_model_raises_fast(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ImportCuratedModelTask._MODELS_ROOT", tmp_path)
        (tmp_path / "existing.onnx").write_bytes(b"already here")
        task = _make_task({"curated_key": _CONVERSION_KEY, "model_name": "existing", "overwrite": False})
        with pytest.raises(FileExistsError):
            await _init_task(task)

    async def test_resume_restores_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ImportCuratedModelTask._MODELS_ROOT", tmp_path)
        source_file = tmp_path / "resumed_source.pt"
        source_file.write_bytes(b"partial download")
        task = _make_task({})
        resume = {
            "job_id": "existing-job-id",
            "model_name": "m1",
            "overwrite": True,
            "source_path": str(source_file),
        }
        await _init_task(task, resume=resume)

        assert task._job_id == "existing-job-id"
        assert task._model_name == "m1"
        assert task._overwrite is True
        assert task._source_path == source_file


class TestRun:
    def _make_sse_stream(self, events: list[tuple[str, dict]]) -> list[str]:
        lines = []
        for event_type, data in events:
            lines.append(f"event: {event_type}")
            lines.append(f"data: {json.dumps(data)}")
            lines.append("")
        return lines

    async def _run_task(
        self,
        task: ImportCuratedModelTask,
        pt_bytes: bytes,
        sse_lines: list[str],
        artifact_tar_bytes: bytes,
        job_id: str = "job-123",
        cancel_after_submit: bool = False,
    ):
        mock_tc = MagicMock()
        mock_api_client = MagicMock()
        mock_tc.ApiClient.return_value = mock_api_client
        mock_tc.Configuration = MagicMock(return_value=MagicMock())

        mock_submit_resp = MagicMock()
        mock_submit_resp.job_id = job_id

        def _submit(*a, **kw):
            if cancel_after_submit:
                task._cancel_flag.set()
            return mock_submit_resp

        mock_training_api = MagicMock()
        mock_training_api.submit_training_job = MagicMock(side_effect=_submit)
        mock_training_api.start_training_job = MagicMock(return_value=MagicMock())
        mock_training_api.cancel_training_job = MagicMock(return_value=MagicMock())

        async def _aiter_lines():
            for line in sse_lines:
                yield line

        mock_stream_response = MagicMock()
        mock_stream_response.aiter_lines = _aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        push_calls = []

        async def _get(url, *args, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if url.endswith("/artifact"):
                resp.content = artifact_tar_bytes
            else:
                # .pt download
                resp.content = pt_bytes
            return resp

        async def _post(url, content=None, headers=None, **kwargs):
            push_calls.append({"url": url, "content": content, "headers": headers})
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            return resp

        mock_http_client = MagicMock()
        mock_http_client.get = AsyncMock(side_effect=_get)
        mock_http_client.post = AsyncMock(side_effect=_post)
        mock_http_client.stream = MagicMock(return_value=mock_stream_ctx)

        mock_http_ctx = AsyncMock()
        mock_http_ctx.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("apps.tasks_server.tasks.ImportCuratedModelTask.APPS_CONFIG") as mock_cfg,
            patch("apps.tasks_server.tasks.ImportCuratedModelTask.training_client", mock_tc),
            patch("apps.tasks_server.tasks.ImportCuratedModelTask.db_client"),
            patch("apps.tasks_server.tasks.ImportCuratedModelTask.TrainingApi", return_value=mock_training_api),
            patch("apps.tasks_server.tasks.ImportCuratedModelTask.httpx.AsyncClient", return_value=mock_http_ctx),
            patch(
                "apps.tasks_server.tasks._training_job_runner.TrainingApi",
                return_value=mock_training_api,
            ),
            patch(
                "apps.tasks_server.tasks._training_job_runner.httpx.AsyncClient",
                return_value=mock_http_ctx,
            ),
        ):
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=8005))
            result = await task._run()

        return result, mock_training_api, push_calls

    async def test_happy_path_downloads_converts_and_lands_sidecar(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ImportCuratedModelTask._MODELS_ROOT", tmp_path)

        task = _make_task({"curated_key": _CONVERSION_KEY, "model_name": "conv_model", "overwrite": False})
        await _init_task(task)

        artifact_tar = _build_artifact_tar(
            "conv_model",
            _ONNX_BYTES,
            {"model_family": "ultralytics_e2e", "classes": ["person", "bicycle"], "class_map": {}},
        )
        done_event = [("done", {})]
        result, mock_training_api, push_calls = await self._run_task(
            task,
            pt_bytes=b"fake pt bytes",
            sse_lines=self._make_sse_stream(done_event),
            artifact_tar_bytes=artifact_tar,
        )

        assert result["job_id"] == "job-123"
        assert result["model_name"] == "conv_model"

        mock_training_api.submit_training_job.assert_called_once()
        submitted_payload = mock_training_api.submit_training_job.call_args[0][0]
        assert submitted_payload.trainer_type == "convert_model"
        assert submitted_payload.model_name == "conv_model"

        assert len(push_calls) == 1
        assert push_calls[0]["content"] == b"fake pt bytes"
        assert push_calls[0]["url"].endswith("/api/training/jobs/job-123/source-model")

        onnx_path = tmp_path / "conv_model.onnx"
        sidecar_path = tmp_path / "conv_model.json"
        assert onnx_path.exists()
        assert onnx_path.read_bytes() == _ONNX_BYTES
        assert sidecar_path.exists()

        sidecar = json.loads(sidecar_path.read_text())
        # Family-correctness: the training server's own detected family
        # (from ACTUAL exported ONNX shape, not a guess in the catalogue)
        # must survive unchanged into the final on-disk sidecar.
        assert sidecar["model_family"] == "ultralytics_e2e"
        assert sidecar["classes"] == ["person", "bicycle"]

        # The downloaded temp .pt file must be cleaned up after the run.
        assert task._source_path is not None
        await task._deinit()
        assert not task._source_path.exists()

    async def test_garbage_artifact_onnx_rejected_without_writing_files(self, tmp_path, monkeypatch):
        """The overridden _save_artifact must validate in memory before writing —
        a corrupt/truncated ONNX in the artifact tar must not land on disk."""
        monkeypatch.setattr("apps.tasks_server.tasks.ImportCuratedModelTask._MODELS_ROOT", tmp_path)

        task = _make_task({"curated_key": _CONVERSION_KEY, "model_name": "garbage_model", "overwrite": False})
        await _init_task(task)

        artifact_tar = _build_artifact_tar(
            "garbage_model", b"not a valid onnx file", {"model_family": "ultralytics_detect", "classes": []}
        )
        done_event = [("done", {})]

        with pytest.raises(ValueError, match="Not a valid ONNX model"):
            await self._run_task(
                task,
                pt_bytes=b"fake pt bytes",
                sse_lines=self._make_sse_stream(done_event),
                artifact_tar_bytes=artifact_tar,
            )

        assert not (tmp_path / "garbage_model.onnx").exists()
        assert not (tmp_path / "garbage_model.json").exists()

    async def test_cancel_after_submit_skips_push_and_cancels_remote(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ImportCuratedModelTask._MODELS_ROOT", tmp_path)

        task = _make_task({"curated_key": _CONVERSION_KEY, "model_name": "conv_model2", "overwrite": False})
        await _init_task(task)

        result, mock_training_api, push_calls = await self._run_task(
            task, pt_bytes=b"fake pt bytes", sse_lines=[], artifact_tar_bytes=b"", cancel_after_submit=True
        )

        assert result == {"cancelled": True, "job_id": "job-123"}
        assert push_calls == []
        mock_training_api.cancel_training_job.assert_called_once_with("job-123")
        mock_training_api.start_training_job.assert_not_called()

        # Temp .pt was downloaded before cancel was observed (cancel is only
        # checked after submit) — _deinit must still clean it up.
        assert task._source_path is not None
        await task._deinit()
        assert not task._source_path.exists()

    async def test_download_http_error_propagates_and_cleans_temp_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ImportCuratedModelTask._MODELS_ROOT", tmp_path)

        task = _make_task({"curated_key": _CONVERSION_KEY, "model_name": "failed_dl", "overwrite": False})
        await _init_task(task)

        async def _raise_get(self, url, *args, **kwargs):
            raise RuntimeError("simulated network error")

        monkeypatch.setattr("httpx.AsyncClient.get", _raise_get)

        with pytest.raises(RuntimeError, match="simulated network error"):
            await task._run()

        assert task._source_path is None


class TestParamsSchema:
    def test_only_conversion_entries_offered(self):
        schema = ImportCuratedModelTask.params_schema()
        option_values = {o["value"] for o in schema["curated_key"]["options"]}
        assert option_values == {e["key"] for e in CURATED_BASE_MODELS if e.get("requires_conversion")}

    def test_required_params_present(self):
        schema = ImportCuratedModelTask.params_schema()
        assert schema["model_name"]["required"] is True
        assert schema["overwrite"]["default"] is False
