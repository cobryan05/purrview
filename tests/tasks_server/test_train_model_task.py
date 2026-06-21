"""Tests for TrainModelTask — push-based training lifecycle.

Tests directly instantiate the task and call _init/_run.
The training server endpoints and db_client are mocked; no real servers needed.
"""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.tasks_server.tasks.TrainModelTask import TrainModelTask

_MOCK_MANAGER = MagicMock()
_MOCK_MANAGER.update_task_heartbeat = AsyncMock()


def _make_task(params: dict) -> TrainModelTask:
    return TrainModelTask(
        task_config_uuid="cfg-uuid",
        params=params,
        manager=_MOCK_MANAGER,
    )


class TestParamsSchema:
    def test_training_config_uuid_present(self):
        schema = TrainModelTask.params_schema()
        assert "training_config_uuid" in schema

    def test_legacy_keys_present(self):
        """Legacy raw-param keys are still in the schema for backward compat."""
        schema = TrainModelTask.params_schema()
        assert "trainer_type" in schema
        assert "label_uuid" in schema
        assert "model_name" in schema
        assert "max_epochs" in schema

    def test_trainer_type_has_options(self):
        schema = TrainModelTask.params_schema()
        assert "options" in schema["trainer_type"]
        assert "binary_image_classifier" in schema["trainer_type"]["options"]


class TestInit:
    """_init with legacy params (no training_config_uuid)."""

    def _patch_clients(self):
        return patch("apps.tasks_server.tasks.TrainModelTask.APPS_CONFIG"), \
               patch("apps.tasks_server.tasks.TrainModelTask.db_client"), \
               patch("apps.tasks_server.tasks.TrainModelTask.training_client")

    async def test_init_legacy_sets_model_name(self):
        task = _make_task({
            "trainer_type": "binary_image_classifier",
            "label_uuid": "lbl-1",
            "model_name": "m1",
        })
        cfg_patch, db_patch, tc_patch = self._patch_clients()
        with cfg_patch as mock_cfg, db_patch, tc_patch:
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            await task._init(task._params, None)

        assert task._model_name == "m1"
        assert task._manifest == []
        assert task._config["trainer_type"] == "binary_image_classifier"

    async def test_init_resumes_job_id(self):
        task = _make_task({"trainer_type": "binary_image_classifier", "model_name": "m1"})
        resume = {"job_id": "existing-job-id", "model_name": "m1", "manifest": [], "config": {}}
        cfg_patch, db_patch, tc_patch = self._patch_clients()
        with cfg_patch as mock_cfg, db_patch, tc_patch:
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            await task._init(task._params, resume)

        assert task._job_id == "existing-job-id"
        assert task._model_name == "m1"

    async def test_init_no_resume_data_job_id_is_none(self):
        task = _make_task({"trainer_type": "binary_image_classifier", "model_name": "m1"})
        cfg_patch, db_patch, tc_patch = self._patch_clients()
        with cfg_patch as mock_cfg, db_patch, tc_patch:
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            await task._init(task._params, None)

        assert task._job_id is None

    async def test_init_with_training_config_uuid_calls_resolve(self):
        """When training_config_uuid is present, _resolve_training_config is called."""
        task = _make_task({"training_config_uuid": "tc-uuid-1"})
        cfg_patch, db_patch, tc_patch = self._patch_clients()
        with cfg_patch as mock_cfg, db_patch, tc_patch:
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            with patch.object(
                task,
                "_resolve_training_config",
                return_value=("my_model", [{"image_uuid": "u1", "filename": "a.jpg", "status": "present"}], {"trainer_type": "binary_image_classifier"}),
            ) as mock_resolve:
                await task._init(task._params, None)

        mock_resolve.assert_called_once_with("tc-uuid-1")
        assert task._model_name == "my_model"
        assert len(task._manifest) == 1


class TestRun:
    """Tests for _run and _consume_sse using mocked HTTP calls."""

    def _make_sse_stream(self, events: list[tuple[str, dict]]) -> list[str]:
        """Turn (event_type, data) pairs into SSE-format lines."""
        lines = []
        for event_type, data in events:
            lines.append(f"event: {event_type}")
            lines.append(f"data: {json.dumps(data)}")
            lines.append("")  # blank line = event boundary
        return lines

    async def _run_task_with_sse(
        self,
        task: TrainModelTask,
        sse_lines: list[str],
        job_id: str = "job-123",
        manifest: Optional[list] = None,
        missing_uuids: Optional[list] = None,
    ) -> dict:
        """
        Helper: patch training API + httpx stream, init and run the task.
        """
        if manifest is None:
            manifest = []
        if missing_uuids is None:
            missing_uuids = []

        # Patch APPS_CONFIG and clients.
        mock_tc = MagicMock()
        mock_api_client = MagicMock()
        mock_tc.ApiClient.return_value = mock_api_client
        mock_tc.Configuration = MagicMock(return_value=MagicMock())

        # Build mock TrainingApi.
        mock_submit_resp = MagicMock()
        mock_submit_resp.job_id = job_id
        mock_missing_resp = MagicMock()
        mock_missing_resp.uuids = missing_uuids
        mock_training_api = MagicMock()
        mock_training_api.submit_training_job = MagicMock(return_value=mock_submit_resp)
        mock_training_api.get_missing_images = MagicMock(return_value=mock_missing_resp)
        mock_training_api.start_training_job = MagicMock(return_value=MagicMock())
        mock_training_api.cancel_training_job = MagicMock(return_value=MagicMock())

        # Build async line iterator for SSE.
        async def _aiter_lines():
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.aiter_lines = _aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_get_resp = MagicMock()
        mock_get_resp.content = b""  # empty artifact tar
        mock_get_resp.raise_for_status = MagicMock()

        mock_http_client = MagicMock()
        mock_http_client.stream = MagicMock(return_value=mock_stream_ctx)
        mock_http_client.get = AsyncMock(return_value=mock_get_resp)

        mock_http_ctx = AsyncMock()
        mock_http_ctx.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("apps.tasks_server.tasks.TrainModelTask.APPS_CONFIG") as mock_cfg,
            patch("apps.tasks_server.tasks.TrainModelTask.training_client", mock_tc),
            patch("apps.tasks_server.tasks.TrainModelTask.db_client"),
            patch(
                "apps.tasks_server.tasks.TrainModelTask.TrainingApi",
                return_value=mock_training_api,
            ),
            patch(
                "apps.tasks_server.tasks.TrainModelTask.httpx.AsyncClient",
                return_value=mock_http_ctx,
            ),
            patch("apps.tasks_server.tasks.TrainModelTask.SubmitJobPayload"),
            patch("apps.tasks_server.tasks.TrainModelTask.ManifestEntry"),
            patch.object(TrainModelTask, "_save_artifact", new=AsyncMock()),
        ):
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=8005))
            task._manifest = manifest
            task._config = {"trainer_type": "binary_image_classifier"}
            task._model_name = "test_model"
            await task._init(task._params, None)
            task._manifest = manifest
            task._config = {"trainer_type": "binary_image_classifier"}
            task._model_name = "test_model"
            result = await task._run()

        return result

    async def test_run_submits_job_on_first_call(self):
        task = _make_task({
            "trainer_type": "binary_image_classifier",
            "label_uuid": "lbl-1",
            "model_name": "m1",
        })
        done_event = [("done", {"model_path": "/models/m1.onnx", "val_accuracy": 0.9})]
        result = await self._run_task_with_sse(task, self._make_sse_stream(done_event))
        assert result.get("model_name") == "test_model"

    async def test_run_forwards_progress_events(self):
        task = _make_task({
            "trainer_type": "binary_image_classifier",
            "label_uuid": "lbl-1",
            "model_name": "m1",
        })
        events = [
            ("progress", {"progress": 0.1, "log": "Epoch 1"}),
            ("progress", {"progress": 0.5, "log": "Epoch 5"}),
            ("done", {"model_path": "/models/m1.onnx", "val_accuracy": 0.92}),
        ]
        result = await self._run_task_with_sse(task, self._make_sse_stream(events))
        assert result.get("val_accuracy") == 0.92
        # Training phase progress should have been updated.
        # Last progress update was 0.5 → 50 + 0.5*50 = 75.
        assert task._progress == pytest.approx(75.0)

    async def test_run_logs_progress_messages(self):
        task = _make_task({
            "trainer_type": "binary_image_classifier",
            "label_uuid": "lbl-1",
            "model_name": "m1",
        })
        events = [
            ("progress", {"progress": 0.2, "log": "Epoch 2 val_loss=0.31"}),
            ("done", {"model_path": "/models/m1.onnx"}),
        ]
        await self._run_task_with_sse(task, self._make_sse_stream(events))
        log_text = " ".join(task.get_log_lines())
        assert "Epoch 2 val_loss=0.31" in log_text

    async def test_run_raises_on_error_event(self):
        task = _make_task({
            "trainer_type": "binary_image_classifier",
            "label_uuid": "lbl-1",
            "model_name": "m1",
        })
        events = [("error", {"message": "CUDA OOM"})]
        with pytest.raises(RuntimeError, match="CUDA OOM"):
            await self._run_task_with_sse(task, self._make_sse_stream(events))

    async def test_run_cancel_flag_stops_stream(self):
        task = _make_task({
            "trainer_type": "binary_image_classifier",
            "label_uuid": "lbl-1",
            "model_name": "m1",
        })

        cancel_after = 1

        async def _aiter_lines_with_cancel():
            for i in range(10):
                if i >= cancel_after:
                    task._cancel_flag.set()
                yield f"event: progress"
                yield f"data: {json.dumps({'progress': i * 0.1})}"
                yield ""

        mock_tc = MagicMock()
        mock_tc.ApiClient.return_value = MagicMock()
        mock_tc.Configuration = MagicMock(return_value=MagicMock())

        mock_submit_resp = MagicMock()
        mock_submit_resp.job_id = "job-cancel"
        mock_missing_resp = MagicMock()
        mock_missing_resp.uuids = []
        mock_training_api = MagicMock()
        mock_training_api.submit_training_job = MagicMock(return_value=mock_submit_resp)
        mock_training_api.get_missing_images = MagicMock(return_value=mock_missing_resp)
        mock_training_api.start_training_job = MagicMock(return_value=MagicMock())
        mock_training_api.cancel_training_job = MagicMock(return_value=MagicMock())

        mock_response = MagicMock()
        mock_response.aiter_lines = _aiter_lines_with_cancel

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_http_client = MagicMock()
        mock_http_client.stream = MagicMock(return_value=mock_stream_ctx)

        mock_http_ctx = AsyncMock()
        mock_http_ctx.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("apps.tasks_server.tasks.TrainModelTask.APPS_CONFIG") as mock_cfg,
            patch("apps.tasks_server.tasks.TrainModelTask.training_client", mock_tc),
            patch("apps.tasks_server.tasks.TrainModelTask.db_client"),
            patch(
                "apps.tasks_server.tasks.TrainModelTask.TrainingApi",
                return_value=mock_training_api,
            ),
            patch(
                "apps.tasks_server.tasks.TrainModelTask.httpx.AsyncClient",
                return_value=mock_http_ctx,
            ),
            patch("apps.tasks_server.tasks.TrainModelTask.SubmitJobPayload"),
            patch("apps.tasks_server.tasks.TrainModelTask.ManifestEntry"),
            patch.object(TrainModelTask, "_save_artifact", new=AsyncMock()),
        ):
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=8005))
            await task._init(task._params, None)
            task._manifest = []
            task._config = {"trainer_type": "binary_image_classifier"}
            task._model_name = "m1"
            result = await task._run()

        assert result.get("cancelled") is True
        mock_training_api.cancel_training_job.assert_called_once()

    async def test_run_with_resume_skips_submit(self):
        task = _make_task({
            "trainer_type": "binary_image_classifier",
            "label_uuid": "lbl-1",
            "model_name": "m1",
        })
        done_event = [("done", {"val_accuracy": 0.88})]

        mock_tc = MagicMock()
        mock_tc.ApiClient.return_value = MagicMock()
        mock_tc.Configuration = MagicMock(return_value=MagicMock())
        mock_training_api = MagicMock()
        mock_training_api.cancel_training_job = MagicMock(return_value=MagicMock())

        async def _aiter_lines():
            for line in self._make_sse_stream(done_event):
                yield line

        mock_response = MagicMock()
        mock_response.aiter_lines = _aiter_lines
        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_get_resp = MagicMock()
        mock_get_resp.content = b""
        mock_get_resp.raise_for_status = MagicMock()
        mock_http_client = MagicMock()
        mock_http_client.stream = MagicMock(return_value=mock_stream_ctx)
        mock_http_client.get = AsyncMock(return_value=mock_get_resp)
        mock_http_ctx = AsyncMock()
        mock_http_ctx.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("apps.tasks_server.tasks.TrainModelTask.APPS_CONFIG") as mock_cfg,
            patch("apps.tasks_server.tasks.TrainModelTask.training_client", mock_tc),
            patch("apps.tasks_server.tasks.TrainModelTask.db_client"),
            patch(
                "apps.tasks_server.tasks.TrainModelTask.TrainingApi",
                return_value=mock_training_api,
            ),
            patch(
                "apps.tasks_server.tasks.TrainModelTask.httpx.AsyncClient",
                return_value=mock_http_ctx,
            ),
            patch.object(TrainModelTask, "_save_artifact", new=AsyncMock()),
        ):
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=8005))
            # Resume with existing job_id.
            await task._init(
                task._params,
                {"job_id": "resumed-job", "model_name": "m1", "manifest": [], "config": {}},
            )
            result = await task._run()

        # submit should NOT have been called — we resumed.
        mock_training_api.submit_training_job.assert_not_called()
        assert result.get("model_name") == "m1"
