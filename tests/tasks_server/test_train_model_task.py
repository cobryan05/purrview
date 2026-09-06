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
        return (
            patch("apps.tasks_server.tasks.TrainModelTask.APPS_CONFIG"),
            patch("apps.tasks_server.tasks.TrainModelTask.db_client"),
            patch("apps.tasks_server.tasks.TrainModelTask.training_client"),
        )

    async def test_init_legacy_sets_model_name(self):
        task = _make_task(
            {
                "trainer_type": "binary_image_classifier",
                "label_uuid": "lbl-1",
                "model_name": "m1",
            }
        )
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
                return_value=(
                    "my_model",
                    [{"image_uuid": "u1", "filename": "a.jpg", "status": "present"}],
                    {"trainer_type": "binary_image_classifier"},
                ),
            ) as mock_resolve:
                await task._init(task._params, None)

        mock_resolve.assert_called_once_with("tc-uuid-1")
        assert task._model_name == "my_model"
        assert len(task._manifest) == 1


class TestResolveBundleTargetFields:
    """_resolve_training_target_fields — role/label_uuids lookup for sidecar enrichment."""

    def _task(self) -> TrainModelTask:
        return _make_task({})

    def test_returns_matching_target_role_and_labels(self):
        task = self._task()
        target = MagicMock(
            node_uuid="node-1", role="submodel", label_uuids=["lbl-a", "lbl-b"], class_remap={"lbl-c": "lbl-a"}
        )
        other = MagicMock(node_uuid="node-other", role="top", label_uuids=["lbl-z"], class_remap={})
        mc_api = MagicMock()
        mc_api.get_model_config_training_targets.return_value = MagicMock(targets=[other, target])

        role, label_uuids, class_remap = task._resolve_training_target_fields(mc_api, "cfg-1", "node-1", "lbl-a")

        mc_api.get_model_config_training_targets.assert_called_once_with("cfg-1")
        assert role == "submodel"
        assert label_uuids == ["lbl-a", "lbl-b"]
        assert class_remap == {"lbl-c": "lbl-a"}

    def test_falls_back_when_node_not_a_bundle_target(self):
        """A node whose spec matches its inherited spec isn't its own boundary target —
        fall back to a locally-derived role/label list instead of failing."""
        task = self._task()
        mc_api = MagicMock()
        mc_api.get_model_config_training_targets.return_value = MagicMock(targets=[])

        role, label_uuids, class_remap = task._resolve_training_target_fields(mc_api, "cfg-1", "node-1", "lbl-a")

        assert role == "submodel"
        assert label_uuids == ["lbl-a"]
        assert class_remap == {}

    def test_falls_back_to_top_role_for_root_node(self):
        task = self._task()
        mc_api = MagicMock()
        mc_api.get_model_config_training_targets.return_value = MagicMock(targets=[])

        role, label_uuids, class_remap = task._resolve_training_target_fields(mc_api, "cfg-1", "root-node", "")

        assert role == "top"
        assert label_uuids == []
        assert class_remap == {}

    def test_swallows_lookup_errors_and_falls_back(self):
        task = self._task()
        mc_api = MagicMock()
        mc_api.get_model_config_training_targets.side_effect = RuntimeError("db_server unreachable")

        role, label_uuids, class_remap = task._resolve_training_target_fields(mc_api, "cfg-1", "node-1", "lbl-a")

        assert role == "submodel"
        assert label_uuids == ["lbl-a"]
        assert class_remap == {}


class TestJobConfigBundleFields:
    """job_config from each _resolve_* builder carries role/model_config_uuid/
    model_config_node_uuid (and label_uuids for object_detection)."""

    def test_binary_classifier_job_config_carries_bundle_fields(self):
        """End-to-end through _resolve_training_config: single labeled node → binary
        classifier path is inlined (not its own method), so exercise it via the
        public entry point with all four db_client APIs mocked."""
        task = _make_task({})
        task.log = MagicMock()
        task._db_api_client = MagicMock()

        tc_data = MagicMock(
            status="success",
            dataset_uuids=[],
            hyperparams_json={},
            model_config_node_uuid="node-1",
            name="My Config",
        )
        tc_api = MagicMock()
        tc_api.get_training_config.return_value = tc_data

        node_data = MagicMock(
            model_type="image_classifier",
            architecture="mobilenet_v3_small",
            label_uuid="lbl-a",
            model_config_uuid="cfg-uuid",
        )
        mc_api = MagicMock()
        mc_api.get_model_config_node.return_value = node_data
        target = MagicMock(node_uuid="node-1", role="submodel", label_uuids=["lbl-a"], class_remap={})
        mc_api.get_model_config_training_targets.return_value = MagicMock(targets=[target])

        row = MagicMock(image_uuid="img-1", filename="a.jpg", status=MagicMock(value="present"))
        img_api = MagicMock()
        img_api.list_images_for_label.return_value = MagicMock(images=[row])

        with (
            patch("apps.tasks_server.tasks.TrainModelTask.TrainingConfigsApi", return_value=tc_api),
            patch("apps.tasks_server.tasks.TrainModelTask.ModelConfigsApi", return_value=mc_api),
            patch("apps.tasks_server.tasks.TrainModelTask.ImagesApi", return_value=img_api),
            patch("apps.tasks_server.tasks.TrainModelTask.DatasetsApi"),
        ):
            _, _, job_config = task._resolve_training_config("tc-1")

        assert job_config["role"] == "submodel"
        assert job_config["model_config_uuid"] == "cfg-uuid"
        assert job_config["model_config_node_uuid"] == "node-1"

    def test_root_node_job_config_carries_bundle_fields(self):
        task = _make_task({})
        task.log = MagicMock()

        node = MagicMock(uuid="node-1", label_uuid="lbl-a")
        cfg = MagicMock(nodes=[MagicMock(uuid="root-node", label_uuid=None), node])
        mc_api = MagicMock()
        mc_api.list_model_configs.return_value = MagicMock(configs=[cfg])

        img_api = MagicMock()
        row = MagicMock(image_uuid="img-1", filename="a.jpg", status=MagicMock(value="present"))
        img_api.list_images_for_label.return_value = MagicMock(images=[row])

        _, _, job_config = task._resolve_root_node(
            node_uuid="root-node",
            mc_api=mc_api,
            img_api=img_api,
            dataset_uuids=[],
            dataset_uuids_set=set(),
            model_name="m1",
            architecture="mobilenet_v3_small",
            hyperparams={},
            role="top",
            model_config_uuid="cfg-uuid",
            model_config_node_uuid="root-node",
        )

        assert job_config["role"] == "top"
        assert job_config["model_config_uuid"] == "cfg-uuid"
        assert job_config["model_config_node_uuid"] == "root-node"

    def test_object_detection_job_config_carries_bundle_fields(self):
        task = _make_task({})
        task.log = MagicMock()

        img = MagicMock(image_uuid="img-1", filename="a.jpg", annotations=[])
        mc_api = MagicMock()
        mc_api.get_training_target_detection_dataset.return_value = MagicMock(
            status="success", class_names=["cat", "dog"], images=[img]
        )

        _, _, job_config = task._resolve_object_detection(
            mc_api=mc_api,
            model_name="m1",
            architecture="yolo_v9_c",
            hyperparams={},
            role="submodel",
            model_config_uuid="cfg-uuid",
            node_uuid="node-uuid",
            target_label_uuids=["lbl-a", "lbl-b"],
            class_remap={},
            dataset_uuids=["ds-1"],
        )

        assert job_config["role"] == "submodel"
        assert job_config["model_config_uuid"] == "cfg-uuid"
        assert job_config["model_config_node_uuid"] == "node-uuid"
        assert job_config["label_uuids"] == ["lbl-a", "lbl-b"]
        mc_api.get_training_target_detection_dataset.assert_called_once()
        call_args = mc_api.get_training_target_detection_dataset.call_args[0]
        assert call_args[0] == "cfg-uuid"
        assert call_args[1] == "node-uuid"
        assert call_args[2].label_uuids == ["lbl-a", "lbl-b"]
        assert call_args[2].dataset_uuids == ["ds-1"]


class TestRecordImageManifestFlag:
    """The "remember exact image list" opt-out reaches every trainer path.

    It is stored in hyperparams_json (the training config's only free-form blob)
    but must NOT ride the ``**hyperparams`` merge — see NOTES.md on trainers
    silently disagreeing about merged key names.
    """

    def _resolve_with_hyperparams(self, hyperparams_json: dict):
        """Drive the binary-classifier path end-to-end through the public entry
        point with a given hyperparams_json, returning its job_config."""
        task = _make_task({})
        task.log = MagicMock()
        task._db_api_client = MagicMock()

        tc_data = MagicMock(
            status="success",
            dataset_uuids=[],
            hyperparams_json=hyperparams_json,
            model_config_node_uuid="node-1",
            name="My Config",
        )
        tc_api = MagicMock()
        tc_api.get_training_config.return_value = tc_data

        node_data = MagicMock(
            model_type="image_classifier",
            architecture="mobilenet_v3_small",
            label_uuid="lbl-a",
            model_config_uuid="cfg-uuid",
        )
        mc_api = MagicMock()
        mc_api.get_model_config_node.return_value = node_data
        target = MagicMock(node_uuid="node-1", role="submodel", label_uuids=["lbl-a"], class_remap={})
        mc_api.get_model_config_training_targets.return_value = MagicMock(targets=[target])

        row = MagicMock(image_uuid="img-1", filename="a.jpg", status=MagicMock(value="present"))
        img_api = MagicMock()
        img_api.list_images_for_label.return_value = MagicMock(images=[row])

        with (
            patch("apps.tasks_server.tasks.TrainModelTask.TrainingConfigsApi", return_value=tc_api),
            patch("apps.tasks_server.tasks.TrainModelTask.ModelConfigsApi", return_value=mc_api),
            patch("apps.tasks_server.tasks.TrainModelTask.ImagesApi", return_value=img_api),
            patch("apps.tasks_server.tasks.TrainModelTask.DatasetsApi"),
        ):
            _, _, job_config = task._resolve_training_config("tc-1")
        return job_config

    def test_defaults_to_true_when_absent_from_hyperparams(self):
        """No existing training config carries the key; they must keep recording."""
        assert self._resolve_with_hyperparams({})["record_image_manifest"] is True

    def test_false_in_hyperparams_reaches_job_config(self):
        assert self._resolve_with_hyperparams({"record_image_manifest": False})["record_image_manifest"] is False

    def test_true_in_hyperparams_reaches_job_config(self):
        assert self._resolve_with_hyperparams({"record_image_manifest": True})["record_image_manifest"] is True

    def test_root_node_path_carries_the_flag(self):
        task = _make_task({})
        task.log = MagicMock()

        node = MagicMock(uuid="node-1", label_uuid="lbl-a")
        cfg = MagicMock(nodes=[MagicMock(uuid="root-node", label_uuid=None), node])
        mc_api = MagicMock()
        mc_api.list_model_configs.return_value = MagicMock(configs=[cfg])

        img_api = MagicMock()
        row = MagicMock(image_uuid="img-1", filename="a.jpg", status=MagicMock(value="present"))
        img_api.list_images_for_label.return_value = MagicMock(images=[row])

        _, _, job_config = task._resolve_root_node(
            node_uuid="root-node",
            mc_api=mc_api,
            img_api=img_api,
            dataset_uuids=[],
            dataset_uuids_set=set(),
            model_name="m1",
            architecture="mobilenet_v3_small",
            hyperparams={},
            role="top",
            model_config_uuid="cfg-uuid",
            model_config_node_uuid="root-node",
            record_image_manifest=False,
        )
        assert job_config["record_image_manifest"] is False

    def test_object_detection_path_carries_the_flag(self):
        task = _make_task({})
        task.log = MagicMock()

        img = MagicMock(image_uuid="img-1", filename="a.jpg", annotations=[])
        mc_api = MagicMock()
        mc_api.get_training_target_detection_dataset.return_value = MagicMock(
            status="success", class_names=["cat"], images=[img]
        )

        _, _, job_config = task._resolve_object_detection(
            mc_api=mc_api,
            model_name="m1",
            architecture="yolo_v9_c",
            hyperparams={},
            role="submodel",
            model_config_uuid="cfg-uuid",
            node_uuid="node-uuid",
            target_label_uuids=["lbl-a"],
            class_remap={},
            dataset_uuids=["ds-1"],
            record_image_manifest=False,
        )
        assert job_config["record_image_manifest"] is False

    def test_explicit_flag_wins_over_a_stale_hyperparams_copy(self):
        """The key is threaded AFTER the ``**hyperparams`` splat, so a copy left
        behind in hyperparams_json can never override the resolved value."""
        task = _make_task({})
        task.log = MagicMock()

        img = MagicMock(image_uuid="img-1", filename="a.jpg", annotations=[])
        mc_api = MagicMock()
        mc_api.get_training_target_detection_dataset.return_value = MagicMock(
            status="success", class_names=["cat"], images=[img]
        )

        _, _, job_config = task._resolve_object_detection(
            mc_api=mc_api,
            model_name="m1",
            architecture="yolo_v9_c",
            hyperparams={"record_image_manifest": True},
            role="submodel",
            model_config_uuid="cfg-uuid",
            node_uuid="node-uuid",
            target_label_uuids=["lbl-a"],
            class_remap={},
            dataset_uuids=["ds-1"],
            record_image_manifest=False,
        )
        assert job_config["record_image_manifest"] is False


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
                "apps.tasks_server.tasks._training_job_runner.TrainingApi",
                return_value=mock_training_api,
            ),
            patch(
                "apps.tasks_server.tasks._training_job_runner.httpx.AsyncClient",
                return_value=mock_http_ctx,
            ),
            patch("apps.tasks_server.tasks._training_job_runner.SubmitJobPayload"),
            patch("apps.tasks_server.tasks._training_job_runner.ManifestEntry"),
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
        task = _make_task(
            {
                "trainer_type": "binary_image_classifier",
                "label_uuid": "lbl-1",
                "model_name": "m1",
            }
        )
        done_event = [("done", {"model_path": "/models/m1.onnx", "val_accuracy": 0.9})]
        result = await self._run_task_with_sse(task, self._make_sse_stream(done_event))
        assert result.get("model_name") == "test_model"

    async def test_run_forwards_progress_events(self):
        task = _make_task(
            {
                "trainer_type": "binary_image_classifier",
                "label_uuid": "lbl-1",
                "model_name": "m1",
            }
        )
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
        task = _make_task(
            {
                "trainer_type": "binary_image_classifier",
                "label_uuid": "lbl-1",
                "model_name": "m1",
            }
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
            {
                "trainer_type": "binary_image_classifier",
                "label_uuid": "lbl-1",
                "model_name": "m1",
            }
        )
        events = [("error", {"message": "CUDA OOM"})]
        with pytest.raises(RuntimeError, match="CUDA OOM"):
            await self._run_task_with_sse(task, self._make_sse_stream(events))

    async def test_run_cancel_flag_stops_stream(self):
        task = _make_task(
            {
                "trainer_type": "binary_image_classifier",
                "label_uuid": "lbl-1",
                "model_name": "m1",
            }
        )

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
                "apps.tasks_server.tasks._training_job_runner.TrainingApi",
                return_value=mock_training_api,
            ),
            patch(
                "apps.tasks_server.tasks._training_job_runner.httpx.AsyncClient",
                return_value=mock_http_ctx,
            ),
            patch("apps.tasks_server.tasks._training_job_runner.SubmitJobPayload"),
            patch("apps.tasks_server.tasks._training_job_runner.ManifestEntry"),
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
        task = _make_task(
            {
                "trainer_type": "binary_image_classifier",
                "label_uuid": "lbl-1",
                "model_name": "m1",
            }
        )
        done_event = [("done", {"val_accuracy": 0.88})]

        mock_tc = MagicMock()
        mock_tc.ApiClient.return_value = MagicMock()
        mock_tc.Configuration = MagicMock(return_value=MagicMock())
        mock_training_api = MagicMock()
        mock_training_api.cancel_training_job = MagicMock(return_value=MagicMock())
        mock_job_status = MagicMock()
        mock_job_status.status = "running"
        mock_training_api.get_training_job.return_value = mock_job_status

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
                "apps.tasks_server.tasks._training_job_runner.TrainingApi",
                return_value=mock_training_api,
            ),
            patch(
                "apps.tasks_server.tasks._training_job_runner.httpx.AsyncClient",
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


class TestSaveArtifact:
    """Tests for _save_artifact — verifies correct extraction from artifact tar."""

    def _make_tar(self, files: dict) -> bytes:
        """Build an in-memory tar containing the given filename→content mapping."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:") as tf:
            for name, data in files.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        return buf.getvalue()

    async def _run_save_artifact(self, tar_bytes, models_dir):
        """Call _save_artifact with a fake tar, returns list of filenames written."""
        task = _make_task({"model_name": "m"})
        task._model_name = "m"
        task._training_base_url = "http://training:8005"
        task._job_id = "job-1"
        task._push_artifact_to_inference_server = AsyncMock()

        mock_get_resp = MagicMock()
        mock_get_resp.content = tar_bytes
        mock_get_resp.raise_for_status = MagicMock()

        mock_http_client = MagicMock()
        mock_http_client.get = AsyncMock(return_value=mock_get_resp)
        mock_http_ctx = AsyncMock()
        mock_http_ctx.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("apps.tasks_server.tasks._training_job_runner.httpx.AsyncClient", return_value=mock_http_ctx),
            patch("apps.tasks_server.tasks._training_job_runner._MODELS_DIR", models_dir),
        ):
            await task._save_artifact("job-1")

        return [p.name for p in models_dir.iterdir()]

    async def test_extracts_onnx_and_json(self, tmp_path):
        tar = self._make_tar({"mymodel.onnx": b"onnx", "mymodel.json": b"{}"})
        written = await self._run_save_artifact(tar, tmp_path)
        assert "mymodel.onnx" in written
        assert "mymodel.json" in written

    async def test_extracts_onnx_data_sidecar(self, tmp_path):
        """A .onnx.data file (external ONNX weights) must be extracted alongside .onnx."""
        tar = self._make_tar(
            {
                "mymodel.onnx": b"onnx-graph",
                "mymodel.onnx.data": b"weights",
                "mymodel.json": b"{}",
            }
        )
        written = await self._run_save_artifact(tar, tmp_path)
        assert (
            "mymodel.onnx.data" in written
        ), ".onnx.data sidecar was not extracted — ONNX runtime will fail to load the model"

    async def test_skips_unexpected_extensions(self, tmp_path):
        tar = self._make_tar({"mymodel.onnx": b"onnx", "checkpoint.pt": b"pt"})
        written = await self._run_save_artifact(tar, tmp_path)
        assert "checkpoint.pt" not in written


def _task_with_history(samples: list[dict]) -> TrainModelTask:
    """A task with `_epoch_history` populated by replaying progress-event dicts.

    Exercises the real recorder rather than assigning the list directly, so the
    dedup/filter rules stay covered by every status-data assertion below.
    """
    task = _make_task({"trainer_type": "binary_image_classifier"})
    task._epoch_history = []
    task._total_epochs = None
    task._resume_data = {}
    for sample in samples:
        task._record_epoch_metrics(sample)
    return task


class TestEpochHistory:
    """_record_epoch_metrics — the per-epoch series behind the training curves."""

    def test_records_metrics_from_progress_event(self):
        task = _task_with_history(
            [{"progress": 0.1, "epoch": 1, "total_epochs": 10, "train_loss": 0.9, "val_loss": 0.8, "val_accuracy": 0.5}]
        )
        assert task._epoch_history == [{"epoch": 1, "train_loss": 0.9, "val_loss": 0.8, "val_accuracy": 0.5}]
        assert task._total_epochs == 10

    def test_ignores_events_without_an_epoch(self):
        """Phase messages ("Loading dataset...") are progress events too."""
        task = _task_with_history([{"progress": 0.0, "log": "Loading dataset..."}])
        assert task._epoch_history == []

    def test_zero_val_loss_is_dropped(self):
        """The detection trainers send val_loss=0.0 as a "not reported" placeholder;
        charting it would invent a flat zero line."""
        task = _task_with_history([{"epoch": 1, "train_loss": 2.0, "val_loss": 0.0, "val_accuracy": 0.1}])
        assert "val_loss" not in task._epoch_history[0]
        assert task._epoch_history[0]["train_loss"] == 2.0

    def test_an_unvalidated_epoch_leaves_a_gap_not_a_zero(self):
        """The detection trainers validate on an interval and omit val_accuracy
        on the epochs in between. Those epochs keep their train_loss and simply
        carry no accuracy sample — a real 0.0 (a model detecting nothing) still
        records, so absence and zero stay distinguishable."""
        task = _task_with_history(
            [
                {"epoch": 1, "train_loss": 2.0, "val_accuracy": 0.0},
                {"epoch": 2, "train_loss": 1.5},
                {"epoch": 3, "train_loss": 1.2, "val_accuracy": 0.4},
            ]
        )
        assert task._epoch_history[0]["val_accuracy"] == 0.0
        assert "val_accuracy" not in task._epoch_history[1]
        assert task._epoch_history[1]["train_loss"] == 1.5
        assert task._epoch_history[2]["val_accuracy"] == 0.4

    def test_replayed_epochs_truncate_rather_than_append(self):
        """A restarted job resumes from its checkpoint and replays epoch numbers.
        The series must stay monotonic in x, not zig-zag backwards."""
        task = _task_with_history(
            [
                {"epoch": 1, "train_loss": 1.0},
                {"epoch": 2, "train_loss": 0.9},
                {"epoch": 3, "train_loss": 0.8},
                # ── restart: training resumes from the epoch-2 checkpoint ──
                {"epoch": 3, "train_loss": 0.7},
                {"epoch": 4, "train_loss": 0.6},
            ]
        )
        assert [s["epoch"] for s in task._epoch_history] == [1, 2, 3, 4]
        assert task._epoch_history[2]["train_loss"] == 0.7

    def test_history_is_bounded(self):
        from apps.tasks_server.tasks._training_job_runner import _MAX_EPOCH_HISTORY

        task = _task_with_history([{"epoch": i, "train_loss": 1.0} for i in range(_MAX_EPOCH_HISTORY + 50)])
        assert len(task._epoch_history) == _MAX_EPOCH_HISTORY
        # The oldest samples are the ones dropped.
        assert task._epoch_history[-1]["epoch"] == _MAX_EPOCH_HISTORY + 49

    def test_history_rides_along_in_resume_data(self):
        """So an interrupted-and-resumed run keeps the curve it already drew."""
        task = _task_with_history([{"epoch": 1, "train_loss": 1.0}])
        assert task._resume_data["epoch_history"] == task._epoch_history

    async def test_sse_progress_events_populate_history(self):
        """End-to-end through the real SSE relay, not just the recorder."""
        task = _make_task({"trainer_type": "binary_image_classifier", "label_uuid": "l", "model_name": "m1"})
        events = [
            ("progress", {"progress": 0.5, "epoch": 1, "total_epochs": 2, "train_loss": 0.5, "val_accuracy": 0.7}),
            ("progress", {"progress": 1.0, "epoch": 2, "total_epochs": 2, "train_loss": 0.3, "val_accuracy": 0.8}),
            ("done", {"model_path": "/models/m1.onnx"}),
        ]
        runner = TestRun()
        await runner._run_task_with_sse(task, runner._make_sse_stream(events))
        assert [s["epoch"] for s in task._epoch_history] == [1, 2]
        assert task._epoch_history[1]["val_accuracy"] == 0.8


class TestTrainingCurveStatus:
    """status_schema() / get_status_data() — what the task card renders."""

    def test_schema_declares_one_curve_with_two_axes(self):
        """Loss and accuracy share a plot, but not a scale: accuracy is a 0–1
        fraction on its own right-hand axis, loss is unbounded on the left."""
        schema = {w["id"]: w for w in TrainModelTask.status_schema()}
        assert "accuracy_curve" not in schema
        chart = schema["loss_curve"]
        assert chart["type"] == "line_chart"
        assert [s["key"] for s in chart["series"]] == ["train_loss", "val_loss", "val_accuracy"]

        by_key = {s["key"]: s for s in chart["series"]}
        assert by_key["val_accuracy"]["axis"] == "right"
        assert by_key["train_loss"].get("axis") in (None, "left")
        # Validation runs on an interval for the detection trainers, so the
        # sparse series carries its last measurement forward.
        assert by_key["val_accuracy"]["hold"] is True
        assert chart["yRight"] == {"min": 0, "max": 1, "valueFormat": "percent"}

    def test_schema_keeps_progress_and_logs(self):
        ids = [w["id"] for w in TrainModelTask.status_schema()]
        assert ids[0] == "progress"
        assert "log" in ids

    def test_status_data_empty_before_any_epoch(self):
        task = _task_with_history([])
        assert task.get_status_data() == {}

    def test_status_data_carries_the_whole_history_to_the_curve(self):
        task = _task_with_history(
            [
                {"epoch": 1, "total_epochs": 3, "train_loss": 0.9, "val_loss": 0.8, "val_accuracy": 0.50},
                {"epoch": 2, "total_epochs": 3, "train_loss": 0.7, "val_loss": 0.6, "val_accuracy": 0.75},
            ]
        )
        data = task.get_status_data()
        assert data["loss_curve"]["points"] == task._epoch_history

    def test_status_data_stats_report_latest_and_best(self):
        task = _task_with_history(
            [
                {"epoch": 1, "total_epochs": 3, "train_loss": 0.9, "val_loss": 0.6, "val_accuracy": 0.90},
                {"epoch": 2, "total_epochs": 3, "train_loss": 0.7, "val_loss": 0.8, "val_accuracy": 0.75},
            ]
        )
        stats = task.get_status_data()["epoch_stats"]
        assert stats["epoch"] == "2 / 3"
        assert stats["val_accuracy"] == "75.0%"
        # Best is over the whole run, not the last epoch.
        assert stats["best_val_accuracy"] == "90.0%"
        assert stats["best_val_loss"] == "0.6000"

    def test_status_data_omits_metrics_the_trainer_never_reported(self):
        """A detection run has no val_loss — the stat grid must show "—", not 0."""
        task = _task_with_history([{"epoch": 1, "train_loss": 1.5, "val_loss": 0.0, "val_accuracy": 0.4}])
        stats = task.get_status_data()["epoch_stats"]
        assert "val_loss" not in stats
        assert "best_val_loss" not in stats
        assert stats["train_loss"] == "1.5000"

    def test_status_data_epoch_without_total(self):
        task = _task_with_history([{"epoch": 7, "train_loss": 1.0}])
        assert task.get_status_data()["epoch_stats"]["epoch"] == "7"
