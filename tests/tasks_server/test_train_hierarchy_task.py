"""Tests for TrainHierarchyTask — training a whole model config as one run.

The behaviour being pinned is what "train the hierarchy" has to mean: every target gets
trained, in dependency order, and a part that has no TrainingConfig yet gets one rather
than being skipped. Skipping is precisely the old failure — the top model trained, the
sub-model did not, and inference silently stopped at the parent label.

The training server and db_client are mocked; no real servers needed.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.tasks_server.tasks.TrainHierarchyTask import TrainHierarchyTask

_MOCK_MANAGER = MagicMock()
_MOCK_MANAGER.update_task_heartbeat = AsyncMock()


def _target(node_uuid: str, role: str, label_uuid: Optional[str] = None):
    return SimpleNamespace(
        node_uuid=node_uuid,
        role=role,
        label_uuid=label_uuid,
        model_type="object_detection" if role == "top" else "crop_classifier",
        architecture="yolo_v9_s" if role == "top" else "efficientnet_b0",
        label_uuids=[],
        class_names=[],
        class_remap={},
    )


def _training_config(uuid: str, node_uuid: str, name: str = "cfg", datasets=("ds-1",)):
    return SimpleNamespace(
        uuid=uuid,
        name=name,
        model_config_node_uuid=node_uuid,
        dataset_uuids=list(datasets),
        hyperparams_json={},
    )


def _make_task(**params) -> TrainHierarchyTask:
    return TrainHierarchyTask(
        task_config_uuid="cfg-uuid",
        params={"model_config_uuid": "mc-1", **params},
        manager=_MOCK_MANAGER,
    )


class _Harness:
    """Wires the two db APIs TrainHierarchyTask talks to, recording what it creates."""

    def __init__(self, targets, existing):
        self.created: list = []
        self.existing = list(existing)
        # No labels by default: a part whose label cannot be read must still get a
        # usable name rather than failing the run.
        self.labels_api = MagicMock()
        self.labels_api.list_labels.return_value = SimpleNamespace(labels=[])

        self.mc_api = MagicMock()
        self.mc_api.get_model_config_training_targets.return_value = SimpleNamespace(targets=targets)
        self.mc_api.get_model_config.return_value = SimpleNamespace(config=SimpleNamespace(name="Test Nest"))

        self.tc_api = MagicMock()
        self.tc_api.list_training_configs.return_value = SimpleNamespace(configs=self.existing)

        def _create(payload):
            created = _training_config(
                uuid=f"tc-new-{len(self.created) + 1}",
                node_uuid=payload.model_config_node_uuid,
                name=payload.name,
                datasets=payload.dataset_uuids,
            )
            self.created.append(payload)
            self.existing.append(created)
            return SimpleNamespace(config=created)

        self.tc_api.create_training_config.side_effect = _create


def _patched(harness):
    """Patch the task's clients and the API classes it constructs."""
    return (
        patch.object(TrainHierarchyTask, "_setup_clients", lambda self: setattr(self, "_db_api_client", MagicMock())),
        patch("apps.tasks_server.tasks.TrainHierarchyTask.ModelConfigsApi", return_value=harness.mc_api),
        patch("apps.tasks_server.tasks.TrainHierarchyTask.TrainingConfigsApi", return_value=harness.tc_api),
        patch("apps.tasks_server.tasks.TrainHierarchyTask.LabelsApi", return_value=harness.labels_api),
    )


async def _init_task(task, harness, resume_data=None):
    with contextlib.ExitStack() as stack:
        for patcher in _patched(harness):
            stack.enter_context(patcher)
        await task._init(task._params if hasattr(task, "_params") else {"model_config_uuid": "mc-1"}, resume_data)


class TestResolvingTargets:
    async def test_every_target_gets_a_training_config(self):
        """Two targets in, two configs out — the top model already has one, the
        sub-model does not and must be created rather than skipped."""
        harness = _Harness(
            targets=[_target("node-root", "top"), _target("node-cat", "submodel", "label-cat")],
            existing=[_training_config("tc-top", "node-root", name="Test Nest")],
        )
        task = _make_task()
        await _init_task(task, harness)

        assert task._training_config_uuids == ["tc-top", "tc-new-1"]
        assert len(harness.created) == 1
        assert harness.created[0].model_config_node_uuid == "node-cat"

    async def test_created_config_is_named_for_its_label_with_a_safe_artifact_name(self):
        """The display name is for humans; the artifact name is a path segment twice
        over — a file in MODELS_ROOT and part of /api/models/{model_name}/...

        The first real run produced `test_nest_—_3572cc86`: an em-dash in a filename,
        plus the label's uuid prefix, because the artifact name was derived from the
        display name and the label was identified by uuid.
        """
        harness = _Harness(
            targets=[_target("node-root", "top"), _target("node-cat", "submodel", "label-cat")],
            existing=[_training_config("tc-top", "node-root")],
        )
        harness.labels_api = MagicMock()
        harness.labels_api.list_labels.return_value = SimpleNamespace(
            labels=[SimpleNamespace(uuid="label-cat", name="Cat")]
        )
        task = _make_task()
        await _init_task(task, harness)

        created = harness.created[0]
        assert created.name == "Test Nest — Cat", "the display name should name the label"
        artifact = created.hyperparams_json["model_name"]
        assert artifact == "test_nest__cat"
        assert artifact.isascii() and all(c.isalnum() or c == "_" for c in artifact)

    async def test_created_sub_model_config_inherits_the_top_models_datasets(self):
        """A sub-model refines what the top model detects, so it trains on the same
        images. An empty dataset list would train on nothing and report success."""
        harness = _Harness(
            targets=[_target("node-root", "top"), _target("node-cat", "submodel", "label-cat")],
            existing=[_training_config("tc-top", "node-root", datasets=("ds-a", "ds-b"))],
        )
        task = _make_task()
        await _init_task(task, harness)

        assert harness.created[0].dataset_uuids == ["ds-a", "ds-b"]

    async def test_existing_configs_are_reused_not_duplicated(self):
        """Running twice must not accumulate configs — and must honour settings the
        operator has since edited on the ones that exist."""
        harness = _Harness(
            targets=[_target("node-root", "top"), _target("node-cat", "submodel", "label-cat")],
            existing=[
                _training_config("tc-top", "node-root"),
                _training_config("tc-cat", "node-cat"),
            ],
        )
        task = _make_task()
        await _init_task(task, harness)

        assert task._training_config_uuids == ["tc-top", "tc-cat"]
        assert harness.created == []

    async def test_a_config_with_no_targets_is_an_error_not_a_silent_success(self):
        """Reporting success having trained nothing is how a hierarchy ends up looking
        trained while every part is missing."""
        harness = _Harness(targets=[], existing=[])
        task = _make_task()
        with pytest.raises(RuntimeError):
            await _init_task(task, harness)


class TestRunningEveryPart:
    async def _run_with_jobs(self, harness, resume_data=None):
        task = _make_task()
        await _init_task(task, harness, resume_data)

        ran: list = []

        async def _fake_job():
            ran.append(task._model_name)
            return {"job_id": f"job-{len(ran)}", "model_name": task._model_name}

        def _resolve(tc_uuid):
            return (f"model_for_{tc_uuid}", [], {})

        task._run_one_job = _fake_job
        task._resolve_training_config = _resolve
        result = await task._run()
        return task, ran, result

    async def test_runs_one_job_per_target_in_target_order(self):
        """Top model first: a sub-model's training set is defined against the boundary
        the top model detects, and the run order is the contract the whole cascade
        inherits."""
        harness = _Harness(
            targets=[_target("node-root", "top"), _target("node-cat", "submodel", "label-cat")],
            existing=[_training_config("tc-top", "node-root"), _training_config("tc-cat", "node-cat")],
        )
        _, ran, result = await self._run_with_jobs(harness)

        assert ran == ["model_for_tc-top", "model_for_tc-cat"]
        assert result["parts_trained"] == 2
        assert result["cancelled"] is False

    async def test_resume_skips_parts_that_already_finished(self):
        """A container restart mid-hierarchy must not retrain the parts that are done —
        each is potentially hours of GPU time."""
        harness = _Harness(
            targets=[_target("node-root", "top"), _target("node-cat", "submodel", "label-cat")],
            existing=[_training_config("tc-top", "node-root"), _training_config("tc-cat", "node-cat")],
        )
        _, ran, result = await self._run_with_jobs(harness, resume_data={"target_index": 1})

        assert ran == ["model_for_tc-cat"]
        assert result["parts_total"] == 2

    async def test_cancelling_stops_between_parts_and_says_so(self):
        """Cancel has to be observable: a run that stopped after the top model looks
        exactly like a completed one if the result does not distinguish them."""
        harness = _Harness(
            targets=[_target("node-root", "top"), _target("node-cat", "submodel", "label-cat")],
            existing=[_training_config("tc-top", "node-root"), _training_config("tc-cat", "node-cat")],
        )
        task = _make_task()
        await _init_task(task, harness)

        async def _job_then_cancel():
            task._cancel_flag.set()
            return {"job_id": "job-1", "model_name": task._model_name}

        task._run_one_job = _job_then_cancel
        task._resolve_training_config = lambda tc: (f"model_for_{tc}", [], {})
        result = await task._run()

        assert result["parts_trained"] == 1
        assert result["cancelled"] is True

    async def test_a_part_stopped_mid_job_is_not_counted_as_trained(self):
        """Observed for real: the container went down during part 1 and the run reported
        parts_trained: 1 having produced no model at all.

        A cancelled job leaves no artifact, so counting it both overstates the result and
        advances target_index past a part that still needs training — a resume would then
        skip it and call a half-trained hierarchy complete.
        """
        harness = _Harness(
            targets=[_target("node-root", "top"), _target("node-cat", "submodel", "label-cat")],
            existing=[_training_config("tc-top", "node-root"), _training_config("tc-cat", "node-cat")],
        )
        task = _make_task()
        await _init_task(task, harness)

        async def _cancelled_job():
            return {"job_id": "job-1", "cancelled": True}

        task._run_one_job = _cancelled_job
        task._resolve_training_config = lambda tc: (f"model_for_{tc}", [], {})
        result = await task._run()

        assert result["parts_trained"] == 0, "a cancelled job trained nothing"
        assert result["trained"] == []
        assert result["cancelled"] is True
        # Still pointing at the part that did not finish, so a resume retries it.
        assert task._target_index == 0

    async def test_finishing_a_part_clears_its_job_id(self):
        """The job id in resume data belongs to the part at target_index. Left behind, a
        resume reconnects to the finished job and reports the next part as complete."""
        harness = _Harness(
            targets=[_target("node-root", "top"), _target("node-cat", "submodel", "label-cat")],
            existing=[_training_config("tc-top", "node-root"), _training_config("tc-cat", "node-cat")],
        )
        task, _, _ = await self._run_with_jobs(harness)

        assert "job_id" not in task._resume_data
        assert task._resume_data["target_index"] == 2


class TestProgress:
    async def test_progress_spans_the_hierarchy_not_one_job(self):
        """A per-job bar resets to zero on every part and reads as a restart."""
        harness = _Harness(
            targets=[_target("node-root", "top"), _target("node-cat", "submodel", "label-cat")],
            existing=[_training_config("tc-top", "node-root"), _training_config("tc-cat", "node-cat")],
        )
        task = _make_task()
        await _init_task(task, harness)

        assert task.get_progress() == 0.0
        task._target_index = 1
        assert task.get_progress() == 50.0
