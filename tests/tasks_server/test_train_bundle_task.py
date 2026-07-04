"""Tests for TrainBundleTask — sequential per-target training + bundle assembly.

Tests directly instantiate the task and call _init/_run with all db_client/
training_client APIs mocked; no real servers needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.tasks_server.tasks.TrainBundleTask import TrainBundleTask

_MOCK_MANAGER = MagicMock()
_MOCK_MANAGER.update_task_heartbeat = AsyncMock()


def _make_task(params: dict) -> TrainBundleTask:
    return TrainBundleTask(
        task_config_uuid="cfg-uuid",
        params=params,
        manager=_MOCK_MANAGER,
    )


def _patch_clients():
    return (
        patch("apps.tasks_server.tasks.TrainBundleTask.APPS_CONFIG"),
        patch("apps.tasks_server.tasks.TrainBundleTask.db_client"),
        patch("apps.tasks_server.tasks.TrainBundleTask.training_client"),
    )


def _target(
    *,
    node_uuid="node-1",
    role="top",
    label_uuid=None,
    model_type="object_detection",
    architecture="yolo_v9_c",
    label_uuids=None,
    class_names=None,
    class_remap=None,
):
    return MagicMock(
        node_uuid=node_uuid,
        role=role,
        label_uuid=label_uuid,
        model_type=model_type,
        architecture=architecture,
        label_uuids=label_uuids or [],
        class_names=class_names or [],
        class_remap=class_remap or {},
    )


class TestParamsSchema:
    def test_required_keys_present(self):
        schema = TrainBundleTask.params_schema()
        assert "model_config_uuid" in schema
        assert "dataset_uuids" in schema
        assert "bundle_name" in schema
        assert schema["model_config_uuid"]["required"] is True
        assert schema["bundle_name"]["required"] is True

    def test_hyperparams_json_optional(self):
        schema = TrainBundleTask.params_schema()
        assert schema["hyperparams_json"]["required"] is False


class TestInit:
    async def test_requires_model_config_uuid(self):
        task = _make_task({"bundle_name": "b1"})
        cfg_patch, db_patch, tc_patch = _patch_clients()
        with cfg_patch as mock_cfg, db_patch, tc_patch:
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            with pytest.raises(RuntimeError, match="model_config_uuid"):
                await task._init(task._params, None)

    async def test_requires_bundle_name(self):
        task = _make_task({"model_config_uuid": "mc-1"})
        cfg_patch, db_patch, tc_patch = _patch_clients()
        with cfg_patch as mock_cfg, db_patch, tc_patch:
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            with pytest.raises(RuntimeError, match="bundle_name"):
                await task._init(task._params, None)

    async def test_requires_dataset_uuids(self):
        task = _make_task({"model_config_uuid": "mc-1", "bundle_name": "b1"})
        cfg_patch, db_patch, tc_patch = _patch_clients()
        with cfg_patch as mock_cfg, db_patch, tc_patch:
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            with pytest.raises(RuntimeError, match="dataset_uuids"):
                await task._init(task._params, None)

    async def test_requires_nonempty_dataset_uuids(self):
        task = _make_task({"model_config_uuid": "mc-1", "bundle_name": "b1", "dataset_uuids": []})
        cfg_patch, db_patch, tc_patch = _patch_clients()
        with cfg_patch as mock_cfg, db_patch, tc_patch:
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            with pytest.raises(RuntimeError, match="dataset_uuids"):
                await task._init(task._params, None)

    async def test_fresh_run_fetches_and_seeds_resume_data(self):
        task = _make_task(
            {
                "model_config_uuid": "mc-1",
                "dataset_uuids": ["ds-1"],
                "bundle_name": "My Bundle",
            }
        )
        cfg_patch, db_patch, tc_patch = _patch_clients()
        targets = [_target(node_uuid="root-node", role="top")]
        with (
            cfg_patch as mock_cfg,
            db_patch,
            tc_patch,
            patch.object(task, "_fetch_targets", return_value=targets),
            patch.object(task, "_fetch_dataset_uuids_set", return_value={"img-1"}),
        ):
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            await task._init(task._params, None)

        assert task._targets == targets
        assert task._target_index == 0
        assert task._completed_artifacts == []
        assert task._job_id is None
        assert task._resume_data["targets"] == targets
        assert task._resume_data["bundle_name"] == "My Bundle"

    async def test_fresh_run_raises_when_no_targets(self):
        task = _make_task(
            {
                "model_config_uuid": "mc-1",
                "dataset_uuids": ["ds-1"],
                "bundle_name": "b1",
            }
        )
        cfg_patch, db_patch, tc_patch = _patch_clients()
        with (
            cfg_patch as mock_cfg,
            db_patch,
            tc_patch,
            patch.object(task, "_fetch_targets", return_value=[]),
        ):
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            with pytest.raises(RuntimeError, match="no training targets"):
                await task._init(task._params, None)

    async def test_resume_restores_state(self):
        task = _make_task({})
        resume = {
            "model_config_uuid": "mc-1",
            "dataset_uuids": ["ds-1"],
            "bundle_name": "b1",
            "hyperparams_json": {"batch_size": 4},
            "targets": [{"node_uuid": "n1", "role": "top"}],
            "target_index": 1,
            "completed_artifacts": [{"role": "top", "node_uuid": "n1"}],
            "job_id": "job-in-flight",
            "manifest": [{"image_uuid": "u1", "filename": "a.jpg", "status": "present"}],
            "config": {"trainer_type": "object_detection"},
            "model_name": "b1_top",
        }
        cfg_patch, db_patch, tc_patch = _patch_clients()
        with (
            cfg_patch as mock_cfg,
            db_patch,
            tc_patch,
            patch.object(task, "_fetch_dataset_uuids_set", return_value={"img-1"}),
        ):
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            await task._init(task._params, resume)

        assert task._target_index == 1
        assert task._job_id == "job-in-flight"
        assert task._completed_artifacts == [{"role": "top", "node_uuid": "n1"}]
        assert task._hyperparams == {"batch_size": 4}
        # Derivable, so refetched on resume rather than persisted.
        assert task._dataset_uuids_set == {"img-1"}

    async def test_resume_data_does_not_persist_dataset_uuids_set(self):
        task = _make_task(
            {
                "model_config_uuid": "mc-1",
                "dataset_uuids": ["ds-1"],
                "bundle_name": "b1",
            }
        )
        cfg_patch, db_patch, tc_patch = _patch_clients()
        with (
            cfg_patch as mock_cfg,
            db_patch,
            tc_patch,
            patch.object(task, "_fetch_targets", return_value=[_target()]),
            patch.object(task, "_fetch_dataset_uuids_set", return_value={"img-1", "img-2"}),
        ):
            mock_cfg.__getitem__ = MagicMock(return_value=MagicMock(host="h", port=1234))
            await task._init(task._params, None)

        assert "dataset_uuids_set" not in task._resume_data
        assert task._dataset_uuids_set == {"img-1", "img-2"}


class TestTargetModelName:
    def test_top_role(self):
        task = _make_task({})
        task._bundle_name = "My Bundle!"
        name = task._target_model_name({"role": "top", "node_uuid": "root-node"})
        assert name == "my_bundle_top"

    def test_submodel_role_uses_node_uuid_prefix(self):
        task = _make_task({})
        task._bundle_name = "b1"
        name = task._target_model_name({"role": "submodel", "node_uuid": "abcdef1234567890"})
        assert name == "b1_sub_abcdef12"


class TestResolveTargetJob:
    def test_dispatches_object_detection(self):
        task = _make_task({})
        task.log = MagicMock()
        task._db_api_client = MagicMock()
        task._model_config_uuid = "mc-1"
        task._dataset_uuids = ["ds-1"]
        task._dataset_uuids_set = {"img-1"}
        task._hyperparams = {}
        task._bundle_name = "b1"

        target = {
            "node_uuid": "node-1",
            "role": "top",
            "label_uuid": None,
            "model_type": "object_detection",
            "architecture": "yolo_v9_c",
            "label_uuids": ["lbl-a"],
            "class_names": ["cat"],
            "class_remap": {},
        }
        with patch.object(task, "_resolve_target_object_detection", return_value=("m", [], {})) as mock_od:
            task._resolve_target_job(target)
        mock_od.assert_called_once()

    def test_dispatches_binary_for_single_label(self):
        task = _make_task({})
        task._db_api_client = MagicMock()
        task._bundle_name = "b1"
        target = {
            "node_uuid": "node-1",
            "role": "submodel",
            "label_uuid": "lbl-a",
            "model_type": "image_classifier",
            "architecture": "mobilenet_v3_small",
            "label_uuids": ["lbl-a"],
            "class_names": ["a"],
            "class_remap": {},
        }
        with patch.object(task, "_resolve_target_binary", return_value=("m", [], {})) as mock_bin:
            task._resolve_target_job(target)
        mock_bin.assert_called_once()

    def test_dispatches_multi_label_for_multiple_labels(self):
        task = _make_task({})
        task._db_api_client = MagicMock()
        task._bundle_name = "b1"
        target = {
            "node_uuid": "node-1",
            "role": "top",
            "label_uuid": None,
            "model_type": "image_classifier",
            "architecture": "mobilenet_v3_small",
            "label_uuids": ["lbl-a", "lbl-b"],
            "class_names": ["a", "b"],
            "class_remap": {},
        }
        with patch.object(task, "_resolve_target_multi_label", return_value=("m", [], {})) as mock_ml:
            task._resolve_target_job(target)
        mock_ml.assert_called_once()

    def test_raises_for_empty_label_uuids_on_classification_target(self):
        """A childless sub-model boundary has an empty class list — nothing to train."""
        task = _make_task({})
        task._db_api_client = MagicMock()
        task._bundle_name = "b1"
        target = {
            "node_uuid": "node-1",
            "role": "submodel",
            "label_uuid": "lbl-a",
            "model_type": "image_classifier",
            "architecture": "mobilenet_v3_small",
            "label_uuids": [],
            "class_names": [],
            "class_remap": {},
        }
        with pytest.raises(RuntimeError, match="no in-scope labels"):
            task._resolve_target_job(target)


class TestRun:
    """_run — sequential per-target loop, one job at a time, bundle created at the end."""

    def _base_task(self, targets: list[dict]) -> TrainBundleTask:
        task = _make_task({})
        task.log = MagicMock()
        task._targets = targets
        task._target_index = 0
        task._completed_artifacts = []
        task._job_id = None
        task._manifest = []
        task._config = {}
        task._model_name = ""
        task._resume_data = {}
        task._data_ready_flag = MagicMock()
        task._data_ready_flag.set = MagicMock()
        task._model_config_uuid = "mc-1"
        task._bundle_name = "b1"
        return task

    async def test_trains_each_target_sequentially_one_job_at_a_time(self):
        targets = [
            {
                "node_uuid": "n1",
                "role": "top",
                "label_uuid": None,
                "model_type": "object_detection",
                "architecture": "yolo_v9_c",
                "label_uuids": ["a"],
                "class_names": ["a"],
                "class_remap": {},
            },
            {
                "node_uuid": "n2",
                "role": "submodel",
                "label_uuid": "b",
                "model_type": "image_classifier",
                "architecture": "mobilenet_v3_small",
                "label_uuids": ["b"],
                "class_names": ["b"],
                "class_remap": {},
            },
        ]
        task = self._base_task(targets)

        resolve_calls = []
        run_job_calls = []

        async def fake_run_one_job():
            run_job_calls.append(task._model_name)
            return {"job_id": "job", "model_name": task._model_name, "val_accuracy": 0.9}

        def fake_resolve(target):
            resolve_calls.append(target["node_uuid"])
            return (f"model_{target['node_uuid']}", [], {})

        with (
            patch.object(task, "_resolve_target_job", side_effect=fake_resolve),
            patch.object(task, "_run_one_job", side_effect=fake_run_one_job),
            patch.object(task, "_create_bundle", return_value={"uuid": "bundle-1", "name": "b1"}),
        ):
            result = await task._run()

        # Each target resolved and trained exactly once, in order — never both in flight.
        assert resolve_calls == ["n1", "n2"]
        assert run_job_calls == ["model_n1", "model_n2"]
        assert len(task._completed_artifacts) == 2
        assert result["bundle_uuid"] == "bundle-1"
        assert len(result["artifacts"]) == 2

    async def test_cancellation_stops_loop_without_creating_bundle(self):
        targets = [
            {
                "node_uuid": "n1",
                "role": "top",
                "label_uuid": None,
                "model_type": "object_detection",
                "architecture": "yolo_v9_c",
                "label_uuids": ["a"],
                "class_names": ["a"],
                "class_remap": {},
            },
            {
                "node_uuid": "n2",
                "role": "submodel",
                "label_uuid": "b",
                "model_type": "image_classifier",
                "architecture": "mobilenet_v3_small",
                "label_uuids": ["b"],
                "class_names": ["b"],
                "class_remap": {},
            },
        ]
        task = self._base_task(targets)

        async def fake_run_one_job():
            return {"cancelled": True, "job_id": "job-1"}

        with (
            patch.object(task, "_resolve_target_job", return_value=("m", [], {})),
            patch.object(task, "_run_one_job", side_effect=fake_run_one_job),
            patch.object(task, "_create_bundle") as mock_create,
        ):
            result = await task._run()

        assert result == {"cancelled": True, "job_id": "job-1"}
        mock_create.assert_not_called()

    async def test_target_failure_propagates_without_creating_bundle(self):
        """One target's training fails mid-bundle -> whole task errors, no ModelBundle row,
        but artifacts already saved for prior targets (via _save_artifact inside
        _run_one_job) are left on disk — nothing here deletes them."""
        targets = [
            {
                "node_uuid": "n1",
                "role": "top",
                "label_uuid": None,
                "model_type": "object_detection",
                "architecture": "yolo_v9_c",
                "label_uuids": ["a"],
                "class_names": ["a"],
                "class_remap": {},
            },
            {
                "node_uuid": "n2",
                "role": "submodel",
                "label_uuid": "b",
                "model_type": "image_classifier",
                "architecture": "mobilenet_v3_small",
                "label_uuids": ["b"],
                "class_names": ["b"],
                "class_remap": {},
            },
        ]
        task = self._base_task(targets)

        call_count = {"n": 0}

        async def fake_run_one_job():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"job_id": "job-1", "model_name": task._model_name, "val_accuracy": 0.9}
            raise RuntimeError("training server exploded")

        with (
            patch.object(task, "_resolve_target_job", return_value=("m", [], {})),
            patch.object(task, "_run_one_job", side_effect=fake_run_one_job),
            patch.object(task, "_create_bundle") as mock_create,
        ):
            with pytest.raises(RuntimeError, match="training server exploded"):
                await task._run()

        # First target completed and was recorded; bundle was never created.
        assert len(task._completed_artifacts) == 1
        mock_create.assert_not_called()

    async def test_resumed_mid_target_skips_resolve_and_reuses_job_id(self):
        """If resume_data restored a job_id, _run must NOT re-resolve the target's
        job_config — _run_one_job's own resume branch reconnects to the existing job."""
        targets = [
            {
                "node_uuid": "n1",
                "role": "top",
                "label_uuid": None,
                "model_type": "object_detection",
                "architecture": "yolo_v9_c",
                "label_uuids": ["a"],
                "class_names": ["a"],
                "class_remap": {},
            },
        ]
        task = self._base_task(targets)
        task._job_id = "already-submitted-job"

        async def fake_run_one_job():
            return {"job_id": task._job_id, "model_name": "m", "val_accuracy": 0.5}

        with (
            patch.object(task, "_resolve_target_job") as mock_resolve,
            patch.object(task, "_run_one_job", side_effect=fake_run_one_job),
            patch.object(task, "_create_bundle", return_value={"uuid": "bundle-1", "name": "b1"}),
        ):
            await task._run()

        mock_resolve.assert_not_called()
