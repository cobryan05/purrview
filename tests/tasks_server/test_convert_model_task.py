"""Tests for ConvertModelTask — push-based model-conversion lifecycle.

Mirrors TrainModelTask's test style: the task is instantiated directly and
_init/_run are driven with the training server API + httpx boundaries mocked.
No real servers/network needed.

The staged source-model file is written directly via
apps.helpers.modelStaging.staged_model_path, with
apps.helpers.modelStaging.CACHE_DIR monkeypatched to a tmp dir (same pattern
tests/inference_server/test_import_model.py uses for
apps.inference_server.manager.CACHE_DIR).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.helpers.modelStaging import staged_model_path
from apps.tasks_server.tasks.ConvertModelTask import ConvertModelTask

_MOCK_MANAGER = MagicMock()
_MOCK_MANAGER.update_task_heartbeat = AsyncMock()


def _make_task(params: dict) -> ConvertModelTask:
    return ConvertModelTask(task_config_uuid="cfg-uuid", params=params, manager=_MOCK_MANAGER)


def _stage_file(monkeypatch, tmp_path: Path, staging_id: str, data: bytes = b"raw source model bytes") -> Path:
    staging_root = tmp_path / "staging"
    monkeypatch.setattr("apps.helpers.modelStaging.CACHE_DIR", staging_root)
    path = staged_model_path(staging_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class TestParamsSchema:
    def test_required_params_present(self):
        schema = ConvertModelTask.params_schema()
        assert schema["staging_id"]["required"] is True
        assert schema["model_name"]["required"] is True
        assert schema["overwrite"]["default"] is False


class TestInit:
    def _patch_clients(self):
        return (
            patch("apps.tasks_server.tasks.ConvertModelTask.APPS_CONFIG"),
            patch("apps.tasks_server.tasks.ConvertModelTask.db_client"),
            patch("apps.tasks_server.tasks.ConvertModelTask.training_client"),
        )

    async def _init_task(self, task, resume=None):
        cfg_patch, db_patch, tc_patch = self._patch_clients()
        with cfg_patch as mock_cfg, db_patch, tc_patch:
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            await task._init(task._params, resume)

    async def test_missing_staging_id_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ConvertModelTask._MODELS_ROOT", tmp_path)
        task = _make_task({"model_name": "m1"})
        with pytest.raises(ValueError, match="staging_id"):
            await self._init_task(task)

    async def test_missing_staged_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ConvertModelTask._MODELS_ROOT", tmp_path)
        monkeypatch.setattr("apps.helpers.modelStaging.CACHE_DIR", tmp_path / "staging")
        task = _make_task({"staging_id": "nonexistent-id", "model_name": "m1"})
        with pytest.raises(ValueError, match="Staged source model not found"):
            await self._init_task(task)

    async def test_missing_model_name_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ConvertModelTask._MODELS_ROOT", tmp_path)
        _stage_file(monkeypatch, tmp_path, "stg-1")
        task = _make_task({"staging_id": "stg-1", "model_name": ""})
        with pytest.raises(ValueError, match="Invalid model_name"):
            await self._init_task(task)

    async def test_bad_model_name_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ConvertModelTask._MODELS_ROOT", tmp_path)
        _stage_file(monkeypatch, tmp_path, "stg-1")
        task = _make_task({"staging_id": "stg-1", "model_name": "../escape"})
        with pytest.raises(ValueError, match="Invalid model_name"):
            await self._init_task(task)

    async def test_existing_target_without_overwrite_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ConvertModelTask._MODELS_ROOT", tmp_path)
        (tmp_path / "existing.onnx").write_bytes(b"already here")
        _stage_file(monkeypatch, tmp_path, "stg-1")
        task = _make_task({"staging_id": "stg-1", "model_name": "existing", "overwrite": False})
        with pytest.raises(FileExistsError):
            await self._init_task(task)

    async def test_existing_target_with_overwrite_ok(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ConvertModelTask._MODELS_ROOT", tmp_path)
        (tmp_path / "existing.onnx").write_bytes(b"already here")
        _stage_file(monkeypatch, tmp_path, "stg-1")
        task = _make_task({"staging_id": "stg-1", "model_name": "existing", "overwrite": True})
        await self._init_task(task)
        assert task._model_name == "existing"
        assert task._overwrite is True

    async def test_happy_init_sets_attrs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ConvertModelTask._MODELS_ROOT", tmp_path)
        staged_path = _stage_file(monkeypatch, tmp_path, "stg-1", data=b"hello bytes")
        task = _make_task({"staging_id": "stg-1", "model_name": "m1", "overwrite": False})
        await self._init_task(task)

        assert task._model_name == "m1"
        assert task._staging_id == "stg-1"
        assert task._source_path == staged_path
        assert task._job_id is None
        assert task._manifest == []

    async def test_resume_restores_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ConvertModelTask._MODELS_ROOT", tmp_path)
        _stage_file(monkeypatch, tmp_path, "stg-1")
        task = _make_task({})
        resume = {"job_id": "existing-job-id", "model_name": "m1", "staging_id": "stg-1"}
        await self._init_task(task, resume=resume)

        assert task._job_id == "existing-job-id"
        assert task._model_name == "m1"
        assert task._staging_id == "stg-1"
        assert task._manifest == []


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
        task: ConvertModelTask,
        sse_lines: list[str],
        job_id: str = "job-123",
        cancel_after_submit: bool = False,
    ):
        mock_tc = MagicMock()
        mock_api_client = MagicMock()
        mock_tc.ApiClient.return_value = mock_api_client
        mock_tc.Configuration = MagicMock(return_value=MagicMock())

        mock_submit_resp = MagicMock()
        mock_submit_resp.job_id = job_id
        mock_training_api = MagicMock()
        mock_training_api.submit_training_job = MagicMock(return_value=mock_submit_resp)
        mock_training_api.start_training_job = MagicMock(return_value=MagicMock())
        mock_training_api.cancel_training_job = MagicMock(return_value=MagicMock())

        async def _aiter_lines():
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.aiter_lines = _aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        # httpx is a single shared module object — ConvertModelTask's direct
        # `httpx.AsyncClient` call (source-model push) and the mixin's
        # `_training_job_runner.httpx.AsyncClient` call (SSE stream) both
        # resolve to the SAME attribute, so they must be patched together with
        # one mock client that supports both `.post` (push) and `.stream` (SSE).
        push_calls = []

        async def _post(url, content=None, headers=None, **kwargs):
            push_calls.append({"url": url, "content": content, "headers": headers})
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            return resp

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(side_effect=_post)
        mock_http_client.stream = MagicMock(return_value=mock_stream_ctx)

        mock_http_ctx = AsyncMock()
        mock_http_ctx.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("apps.tasks_server.tasks.ConvertModelTask.APPS_CONFIG") as mock_cfg,
            patch("apps.tasks_server.tasks.ConvertModelTask.training_client", mock_tc),
            patch("apps.tasks_server.tasks.ConvertModelTask.db_client"),
            patch("apps.tasks_server.tasks.ConvertModelTask.TrainingApi", return_value=mock_training_api),
            patch(
                "apps.tasks_server.tasks._training_job_runner.TrainingApi",
                return_value=mock_training_api,
            ),
            patch(
                "apps.tasks_server.tasks._training_job_runner.httpx.AsyncClient",
                return_value=mock_http_ctx,
            ),
            patch.object(ConvertModelTask, "_save_artifact", new=AsyncMock()),
        ):
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=8005))
            if cancel_after_submit:
                task._cancel_flag.set()
            result = await task._run()

        return result, mock_training_api, push_calls

    async def test_happy_path_pushes_bytes_and_completes(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ConvertModelTask._MODELS_ROOT", tmp_path)
        staged_data = b"the raw checkpoint bytes"
        _stage_file(monkeypatch, tmp_path, "stg-1", data=staged_data)

        task = _make_task({"staging_id": "stg-1", "model_name": "conv_model", "overwrite": False})

        cfg_patch, db_patch, tc_patch = TestInit()._patch_clients()
        with cfg_patch as mock_cfg, db_patch, tc_patch:
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            await task._init(task._params, None)

        done_event = [("done", {"val_accuracy": None})]
        result, mock_training_api, push_calls = await self._run_task(task, self._make_sse_stream(done_event))

        assert result["job_id"] == "job-123"
        assert result["model_name"] == "conv_model"

        mock_training_api.submit_training_job.assert_called_once()
        submitted_payload = mock_training_api.submit_training_job.call_args[0][0]
        assert submitted_payload.trainer_type == "convert_model"
        assert submitted_payload.model_name == "conv_model"

        assert len(push_calls) == 1
        assert push_calls[0]["content"] == staged_data
        assert push_calls[0]["url"].endswith("/api/training/jobs/job-123/source-model")

        mock_training_api.start_training_job.assert_called_once_with("job-123")

    async def test_cancel_after_submit_skips_push_and_cancels_remote(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ConvertModelTask._MODELS_ROOT", tmp_path)
        _stage_file(monkeypatch, tmp_path, "stg-1")

        task = _make_task({"staging_id": "stg-1", "model_name": "conv_model2", "overwrite": False})

        cfg_patch, db_patch, tc_patch = TestInit()._patch_clients()
        with cfg_patch as mock_cfg, db_patch, tc_patch:
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            await task._init(task._params, None)

        result, mock_training_api, push_calls = await self._run_task(task, [], cancel_after_submit=True)

        assert result == {"cancelled": True, "job_id": "job-123"}
        assert push_calls == []
        mock_training_api.cancel_training_job.assert_called_once_with("job-123")
        mock_training_api.start_training_job.assert_not_called()


class TestDeinit:
    async def test_deinit_deletes_staged_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ConvertModelTask._MODELS_ROOT", tmp_path)
        staged_path = _stage_file(monkeypatch, tmp_path, "stg-1")

        task = _make_task({"staging_id": "stg-1", "model_name": "m1", "overwrite": False})
        cfg_patch, db_patch, tc_patch = TestInit()._patch_clients()
        with cfg_patch as mock_cfg, db_patch, tc_patch:
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            await task._init(task._params, None)

        assert staged_path.exists()
        await task._deinit()
        assert not staged_path.exists()

    async def test_deinit_is_noop_when_file_already_gone(self, tmp_path, monkeypatch):
        monkeypatch.setattr("apps.tasks_server.tasks.ConvertModelTask._MODELS_ROOT", tmp_path)
        staged_path = _stage_file(monkeypatch, tmp_path, "stg-1")

        task = _make_task({"staging_id": "stg-1", "model_name": "m1", "overwrite": False})
        cfg_patch, db_patch, tc_patch = TestInit()._patch_clients()
        with cfg_patch as mock_cfg, db_patch, tc_patch:
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            await task._init(task._params, None)

        staged_path.unlink()
        await task._deinit()  # should not raise
