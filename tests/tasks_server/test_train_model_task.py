"""Tests for TrainModelTask.

Tests directly instantiate the task and call _init/_run to avoid needing a
running tasks_server HTTP stack.  The training server endpoints are mocked so
no real training container is required.
"""

import asyncio
import json
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
    def test_required_keys_present(self):
        schema = TrainModelTask.params_schema()
        assert "trainer_type" in schema
        assert "label_uuid" in schema
        assert "model_name" in schema

    def test_optional_keys_present(self):
        schema = TrainModelTask.params_schema()
        assert "max_epochs" in schema
        assert "image_source" in schema

    def test_trainer_type_has_options(self):
        schema = TrainModelTask.params_schema()
        assert "options" in schema["trainer_type"]
        assert "binary_image_classifier" in schema["trainer_type"]["options"]


class TestInit:
    async def test_init_sets_required_attrs(self):
        task = _make_task(
            {"trainer_type": "binary_image_classifier", "label_uuid": "lbl-1", "model_name": "m1"}
        )
        with patch("apps.tasks_server.tasks.TrainModelTask.APPS_CONFIG") as mock_cfg:
            mock_cfg.__getitem__ = MagicMock(
                return_value=MagicMock(host="training_host", port=8005)
            )
            with patch("apps.tasks_server.tasks.TrainModelTask.training_client"):
                await task._init(task._params, None)

        assert task._trainer_type == "binary_image_classifier"
        assert task._label_uuid == "lbl-1"
        assert task._model_name == "m1"
        assert task._max_epochs == 50  # default

    async def test_init_custom_max_epochs(self):
        task = _make_task(
            {
                "trainer_type": "binary_image_classifier",
                "label_uuid": "lbl-1",
                "model_name": "m1",
                "max_epochs": 20,
            }
        )
        with patch("apps.tasks_server.tasks.TrainModelTask.APPS_CONFIG") as mock_cfg:
            mock_cfg.__getitem__ = MagicMock(
                return_value=MagicMock(host="training_host", port=8005)
            )
            with patch("apps.tasks_server.tasks.TrainModelTask.training_client"):
                await task._init(task._params, None)

        assert task._max_epochs == 20

    async def test_init_resumes_job_id(self):
        task = _make_task(
            {"trainer_type": "binary_image_classifier", "label_uuid": "lbl-1", "model_name": "m1"}
        )
        resume = {"job_id": "existing-job-id"}
        with patch("apps.tasks_server.tasks.TrainModelTask.APPS_CONFIG") as mock_cfg:
            mock_cfg.__getitem__ = MagicMock(
                return_value=MagicMock(host="training_host", port=8005)
            )
            with patch("apps.tasks_server.tasks.TrainModelTask.training_client"):
                await task._init(task._params, resume)

        assert task._job_id == "existing-job-id"

    async def test_init_no_resume_data_job_id_is_none(self):
        task = _make_task(
            {"trainer_type": "binary_image_classifier", "label_uuid": "lbl-1", "model_name": "m1"}
        )
        with patch("apps.tasks_server.tasks.TrainModelTask.APPS_CONFIG") as mock_cfg:
            mock_cfg.__getitem__ = MagicMock(
                return_value=MagicMock(host="training_host", port=8005)
            )
            with patch("apps.tasks_server.tasks.TrainModelTask.training_client"):
                await task._init(task._params, None)

        assert task._job_id is None


class TestRun:
    """Tests for _run and _consume_sse using a mocked SSE stream."""

    def _make_sse_stream(self, events: list[tuple[str, dict]]) -> list[str]:
        """Turn a list of (event_type, data_dict) into SSE-format lines."""
        lines = []
        for event_type, data in events:
            lines.append(f"event: {event_type}")
            lines.append(f"data: {json.dumps(data)}")
            lines.append("")  # blank line = event boundary
        return lines

    async def _run_task_with_sse(
        self, task: TrainModelTask, sse_lines: list[str], job_id: str = "job-123"
    ) -> dict:
        """
        Helper: patch training_client submit + httpx stream, run _run(), return result.
        """
        # Patch training_client.ApiClient and Configuration (used in _init)
        mock_tc = MagicMock()
        mock_api_client = MagicMock()
        mock_tc.ApiClient.return_value = mock_api_client
        mock_tc.Configuration = MagicMock(return_value=MagicMock())

        # Patch submit_training_job
        mock_submit_resp = MagicMock()
        mock_submit_resp.job_id = job_id
        mock_training_api = MagicMock()
        mock_training_api.submit_training_job = MagicMock(return_value=mock_submit_resp)

        # Patch cancel_training_job (for cancel path)
        mock_training_api.cancel_training_job = MagicMock(return_value=MagicMock())

        # Build an async line iterator from sse_lines
        async def _aiter_lines():
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.aiter_lines = _aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client_ctx = AsyncMock()
        mock_client_ctx.__aenter__ = AsyncMock(return_value=MagicMock(stream=MagicMock(return_value=mock_stream_ctx)))
        mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("apps.tasks_server.tasks.TrainModelTask.APPS_CONFIG") as mock_cfg,
            patch("apps.tasks_server.tasks.TrainModelTask.training_client", mock_tc),
            patch("apps.tasks_server.tasks.TrainModelTask.TrainingApi", return_value=mock_training_api),
            patch("apps.tasks_server.tasks.TrainModelTask.httpx.AsyncClient", return_value=mock_client_ctx),
        ):
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=8005))
            await task._init(task._params, None)
            result = await task._run()

        return result

    async def test_run_submits_job_on_first_call(self):
        task = _make_task(
            {"trainer_type": "binary_image_classifier", "label_uuid": "lbl-1", "model_name": "m1"}
        )
        done_event = [("done", {"model_path": "/models/m1.onnx", "val_accuracy": 0.9})]
        result = await self._run_task_with_sse(task, self._make_sse_stream(done_event))
        assert result["model_path"] == "/models/m1.onnx"
        assert result["val_accuracy"] == 0.9

    async def test_run_forwards_progress_events(self):
        task = _make_task(
            {"trainer_type": "binary_image_classifier", "label_uuid": "lbl-1", "model_name": "m1"}
        )
        events = [
            ("progress", {"progress": 0.1, "log": "Epoch 1"}),
            ("progress", {"progress": 0.5, "log": "Epoch 5"}),
            ("done", {"model_path": "/models/m1.onnx", "val_accuracy": 0.92}),
        ]
        result = await self._run_task_with_sse(task, self._make_sse_stream(events))
        assert result["val_accuracy"] == 0.92
        # Progress should have been updated (last was 0.5 * 100 = 50)
        assert task._progress == pytest.approx(50.0)

    async def test_run_logs_progress_messages(self):
        task = _make_task(
            {"trainer_type": "binary_image_classifier", "label_uuid": "lbl-1", "model_name": "m1"}
        )
        events = [
            ("progress", {"progress": 0.2, "log": "Epoch 2 val_loss=0.31"}),
            ("done", {"model_path": "/models/m1.onnx"}),
        ]
        await self._run_task_with_sse(task, self._make_sse_stream(events))
        log_text = " ".join(task.get_log_lines())
        assert "Epoch 2 val_loss=0.31" in log_text

    async def test_run_raises_on_error_event(self):
        task = _make_task(
            {"trainer_type": "binary_image_classifier", "label_uuid": "lbl-1", "model_name": "m1"}
        )
        events = [("error", {"message": "CUDA OOM"})]
        with pytest.raises(RuntimeError, match="CUDA OOM"):
            await self._run_task_with_sse(task, self._make_sse_stream(events))

    async def test_run_cancel_flag_stops_stream(self):
        task = _make_task(
            {"trainer_type": "binary_image_classifier", "label_uuid": "lbl-1", "model_name": "m1"}
        )

        cancel_after = 1  # cancel after this many progress events

        async def _aiter_lines_with_cancel():
            for i in range(10):
                if i >= cancel_after:
                    task._cancel_flag.set()
                yield f"event: progress"
                yield f"data: {json.dumps({'progress': i * 0.1})}"
                yield ""

        mock_tc = MagicMock()
        mock_api_client = MagicMock()
        mock_tc.ApiClient.return_value = mock_api_client
        mock_tc.Configuration = MagicMock(return_value=MagicMock())

        mock_submit_resp = MagicMock()
        mock_submit_resp.job_id = "job-cancel"
        mock_training_api = MagicMock()
        mock_training_api.submit_training_job = MagicMock(return_value=mock_submit_resp)
        mock_training_api.cancel_training_job = MagicMock(return_value=MagicMock())

        mock_response = MagicMock()
        mock_response.aiter_lines = _aiter_lines_with_cancel

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client_ctx = AsyncMock()
        mock_client_ctx.__aenter__ = AsyncMock(
            return_value=MagicMock(stream=MagicMock(return_value=mock_stream_ctx))
        )
        mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("apps.tasks_server.tasks.TrainModelTask.APPS_CONFIG") as mock_cfg,
            patch("apps.tasks_server.tasks.TrainModelTask.training_client", mock_tc),
            patch("apps.tasks_server.tasks.TrainModelTask.TrainingApi", return_value=mock_training_api),
            patch("apps.tasks_server.tasks.TrainModelTask.httpx.AsyncClient", return_value=mock_client_ctx),
        ):
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=8005))
            await task._init(task._params, None)
            result = await task._run()

        assert result.get("cancelled") is True
        # Remote job should have been cancelled
        mock_training_api.cancel_training_job.assert_called_once()

    async def test_run_with_resume_skips_submit(self):
        task = _make_task(
            {"trainer_type": "binary_image_classifier", "label_uuid": "lbl-1", "model_name": "m1"}
        )
        done_event = [("done", {"model_path": "/models/m1.onnx", "val_accuracy": 0.88})]

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
        mock_client_ctx = AsyncMock()
        mock_client_ctx.__aenter__ = AsyncMock(
            return_value=MagicMock(stream=MagicMock(return_value=mock_stream_ctx))
        )
        mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("apps.tasks_server.tasks.TrainModelTask.APPS_CONFIG") as mock_cfg,
            patch("apps.tasks_server.tasks.TrainModelTask.training_client", mock_tc),
            patch("apps.tasks_server.tasks.TrainModelTask.TrainingApi", return_value=mock_training_api),
            patch("apps.tasks_server.tasks.TrainModelTask.httpx.AsyncClient", return_value=mock_client_ctx),
        ):
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=8005))
            # Pass resume_data with an existing job_id
            await task._init(task._params, {"job_id": "resumed-job"})
            result = await task._run()

        # submit should NOT have been called — we resumed
        mock_training_api.submit_training_job.assert_not_called()
        assert result["model_path"] == "/models/m1.onnx"
