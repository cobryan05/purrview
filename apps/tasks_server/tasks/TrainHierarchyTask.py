"""Train every part of a hierarchical model config, in dependency order, as one run.

A model config with sub-models is *used* as a single model, so it should be *trained* as
one action too. Before this, "train" meant one node, and training the root left every
sub-model untrained — the top model detected "Cat" and nothing downstream ever refined
it, with no error anywhere to point at. Training the parts individually is still
supported, and is now the deliberate alternative (retrain one node) rather than the only
way to get a whole hierarchy.

## Why this orchestrates per-node configs rather than owning the settings itself

Each node keeps its own TrainingConfig — a yolo detector and an EfficientNet crop
classifier want genuinely different epochs/batch/imgsz, and burying per-node overrides in
one blob would give them no editable home. This task therefore *finds or creates* a
config per training target and runs them in order. "Retrain just this node" is then
literally running one of those configs, and this task's output is reproducible by hand.

## Ordering, and why it is not parallel

Targets are run in `derive_training_targets` order: top model first, then sub-models
outermost first. Sequential because there is one GPU — two jobs at once would either
OOM or halve each other's throughput — and because the residual-VRAM behaviour after a
job (see NOTES.md) is only survivable one job at a time.

## Resume

`target_index` rides in the same `_resume_data` dict the job runner already persists, so
a container restart mid-hierarchy picks up at the target it was on rather than retraining
the parts that already finished. A job id in the resume data belongs to the target at
that index — both are cleared together when a target completes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from db_client.api.model_configs_api import ModelConfigsApi
from db_client.api.training_configs_api import TrainingConfigsApi
from db_client.models.training_config_create import TrainingConfigCreate

from apps.helpers.consts import TaskAccent
from apps.helpers.logUtils import LOG_LEVEL

from .Registry import register_task
from .TrainModelTask import TrainModelTask

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


@register_task()
class TrainHierarchyTask(TrainModelTask):
    """Run one training job per target of a model config, top model first."""

    CARD_ACCENT = TaskAccent.VIOLET
    HEARTBEAT_INTERVAL = 60

    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        self._setup_clients()

        self._epoch_history: list[dict[str, Any]] = list((resume_data or {}).get("epoch_history") or [])
        self._total_epochs: Optional[int] = None

        self._model_config_uuid: str = str(params.get("model_config_uuid") or "")
        if not self._model_config_uuid:
            raise RuntimeError("TrainHierarchyTask requires a model_config_uuid")

        resume_data = resume_data or {}
        self._target_index: int = int(resume_data.get("target_index") or 0)
        # A job id in the resume data belongs to the target at _target_index, and
        # _run_one_job reconnects to it rather than submitting a duplicate.
        self._job_id: Optional[str] = resume_data.get("job_id")
        self._model_name: str = str(resume_data.get("model_name", "model"))
        self._manifest: list[dict] = resume_data.get("manifest") or []
        self._config: dict = resume_data.get("config") or {}
        self._completed: list[dict[str, Any]] = list(resume_data.get("completed") or [])

        self._training_config_uuids: list[str] = await asyncio.to_thread(
            self._resolve_hierarchy_training_configs, self._model_config_uuid
        )
        if not self._training_config_uuids:
            raise RuntimeError(f"Model config {self._model_config_uuid!r} has no training targets")

        self.log(
            f"Training {len(self._training_config_uuids)} part(s) of this hierarchy, "
            f"starting at part {self._target_index + 1}"
        )

    def _resolve_hierarchy_training_configs(self, model_config_uuid: str) -> list[str]:
        """One TrainingConfig uuid per training target, in target order, creating any
        that do not exist yet.

        Creating them (rather than refusing to run until the operator has made one per
        node) is what makes "Train hierarchy" a single action on a freshly designed
        config. They are real, editable TrainingConfigs afterwards — the next run picks
        the same ones up by node uuid and honours whatever settings have since been
        changed on them.
        """
        mc_api = ModelConfigsApi(self._db_api_client)
        tc_api = TrainingConfigsApi(self._db_api_client)

        resp = mc_api.get_model_config_training_targets(model_config_uuid)
        targets = list(getattr(resp, "targets", None) or [])
        if not targets:
            return []

        config_resp = mc_api.get_model_config(model_config_uuid)
        config = getattr(config_resp, "config", None)
        config_name = getattr(config, "name", None) or "Hierarchy"

        existing = list(getattr(tc_api.list_training_configs(), "configs", None) or [])
        by_node = {c.model_config_node_uuid: c for c in existing if c.model_config_node_uuid}

        # Datasets for an auto-created sub-model config come from the top model's,
        # because a sub-model refines what the top model detects and therefore trains on
        # the same images. An empty dataset list would train on nothing and "succeed".
        top_node_uuid = targets[0].node_uuid
        top_existing = by_node.get(top_node_uuid)
        inherited_datasets = list(getattr(top_existing, "dataset_uuids", None) or [])

        uuids: list[str] = []
        for target in targets:
            found = by_node.get(target.node_uuid)
            if found is not None:
                uuids.append(found.uuid)
                continue

            label_part = getattr(target, "label_uuid", None)
            suffix = "top" if target.role == "top" else (label_part or target.node_uuid)[:8]
            name = f"{config_name} — {suffix}"
            # Names are unique across training configs, so a collision here is a real
            # one (two configs of the same name) rather than something to overwrite.
            if any(c.name == name for c in existing):
                name = f"{name} ({target.node_uuid[:8]})"

            created = tc_api.create_training_config(
                TrainingConfigCreate(
                    name=name,
                    description=f"Auto-created by Train hierarchy for {config_name}",
                    model_config_node_uuid=target.node_uuid,
                    dataset_uuids=inherited_datasets,
                    hyperparams_json={},
                )
            )
            new_config = getattr(created, "config", None)
            if new_config is None:
                raise RuntimeError(f"Could not create a training config for node {target.node_uuid!r}")
            uuids.append(new_config.uuid)
            self.log(f"Created training config '{name}' for this hierarchy's {target.role} model")

        return uuids

    async def _run(self) -> dict[str, Any]:
        total = len(self._training_config_uuids)

        while self._target_index < total:
            if self._cancel_flag.is_set():
                break
            await self._check_pause()

            training_config_uuid = self._training_config_uuids[self._target_index]

            # Only resolve when starting a target fresh. On resume the job is already
            # submitted and its config came back with the resume data — re-resolving
            # would rebuild the manifest from today's datasets and quietly train a
            # different image set than the running job was submitted with.
            if not self._job_id:
                self._model_name, self._manifest, self._config = await asyncio.to_thread(
                    self._resolve_training_config, training_config_uuid
                )
                self._epoch_history = []

            self.log(f"Training part {self._target_index + 1} of {total}: {self._model_name}")
            result = await self._run_one_job() or {}

            # A cancelled job produced no model, so it is not a trained part. Counting
            # it anyway reported "parts_trained: 1" for a run that trained nothing —
            # observed for real when the container went down mid-run — and left
            # target_index pointing past a part that still needs doing. Break rather
            # than advance: the index is where a resume picks up, and this part is
            # exactly where it should pick up.
            if result.get("cancelled"):
                self.log(f"Part {self._target_index + 1} of {total} stopped before it finished")
                break

            self._completed.append({"model_name": self._model_name, **result})

            self._target_index += 1
            self._job_id = None
            self._resume_data["target_index"] = self._target_index
            self._resume_data["completed"] = self._completed
            # Stale keys for a finished job would make a resume reconnect to it and
            # report the hierarchy as further along than it is.
            self._resume_data.pop("job_id", None)
            await self.heartbeat(force=True)

        return {
            "model_config_uuid": self._model_config_uuid,
            "trained": self._completed,
            "parts_trained": len(self._completed),
            "parts_total": total,
            "cancelled": self._target_index < total,
        }

    def get_progress(self) -> Optional[float]:
        """Progress across the whole hierarchy, not within one job.

        A per-job bar would jump back to zero on every part and read as a restart. The
        within-part detail is in the epoch stats and the curve, which are per-part by
        design — they are what you look at to decide whether THIS part is training well.
        """
        total = len(getattr(self, "_training_config_uuids", []) or [])
        if not total:
            return None
        return min(100.0, (self._target_index / total) * 100.0)

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        return {
            "model_config_uuid": {
                "type": "str",
                "required": True,
                "help": "UUID of the model config (hierarchy) to train every part of",
            },
        }
