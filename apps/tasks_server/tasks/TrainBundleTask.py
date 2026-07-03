"""Train every target in a ModelConfig (top model + each sub-model) sequentially,
then assemble the resulting artifacts into a ModelBundle.

Targets come from GET /api/model-configs/{uuid}/training-targets
(DbClient.derive_bundle_training_targets — the single authority on sub-model
boundaries and per-target label scope/class remapping). One target at a time
is resolved into a job_config and driven through TrainingJobRunnerMixin's
submit → push images → start → SSE relay → save-artifact sequence — targets
are never trained concurrently (see NOTES.md-style rationale: a single
training-server slot should not be asked to run multiple jobs at once).

If a target's training fails, the whole task errors out (existing Task error
semantics) but ONNX+sidecar artifacts already written for prior targets stay
on disk in $MODELS_ROOT — no ModelBundle row is created for an incomplete
set. A user can retry the whole bundle, or compose a partial bundle later
from the individual artifacts (each sidecar self-describes its
role/model_config_uuid/model_config_node_uuid/label_uuids, see
TrainModelTask/_training_job_runner).

Resumable: on restart, resume_data restores which target we were on, the
artifacts already completed, and (if a job was mid-flight) that job's
id/manifest/config so TrainingJobRunnerMixin's own resume branch reconnects
to it rather than resubmitting.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import training_client

import db_client
from db_client.api.datasets_api import DatasetsApi
from db_client.api.images_api import ImagesApi
from db_client.api.model_configs_api import ModelConfigsApi
from db_client.models.bundle_target_detection_dataset_payload import BundleTargetDetectionDatasetPayload
from db_client.models.image_item_status import ImageItemStatus
from db_client.models.model_bundle_create import ModelBundleCreate

from apps import APPS, APPS_CONFIG
from apps.helpers.model_type_map import MODEL_TYPE_TO_TRAINER_TYPE

from ._training_job_runner import _MODELS_DIR, TrainingJobRunnerMixin
from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@register_task()
class TrainBundleTask(Task, TrainingJobRunnerMixin):
    # Trains multiple models back to back — extend heartbeat tolerance like TrainModelTask.
    HEARTBEAT_INTERVAL = 60

    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        db_cfg = APPS_CONFIG[APPS.DB_SERVER]
        db_base_url = f"http://{db_cfg.host}:{db_cfg.port}"
        self._db_api_client = db_client.ApiClient(db_client.Configuration(host=db_base_url))

        training_cfg = APPS_CONFIG[APPS.TRAINING_SERVER]
        self._training_base_url: str = f"http://{training_cfg.host}:{training_cfg.port}"
        tc_config = training_client.Configuration(host=self._training_base_url)
        self._api_client = training_client.ApiClient(tc_config)

        if resume_data and resume_data.get("targets"):
            self._model_config_uuid: str = resume_data["model_config_uuid"]
            self._dataset_uuids: list[str] = list(resume_data["dataset_uuids"])
            self._dataset_uuids_set: set[str] = set(resume_data.get("dataset_uuids_set") or [])
            self._bundle_name: str = resume_data["bundle_name"]
            self._hyperparams: dict = dict(resume_data.get("hyperparams_json") or {})
            self._targets: list[dict] = list(resume_data["targets"])
            self._target_index: int = int(resume_data.get("target_index", 0))
            self._completed_artifacts: list[dict] = list(resume_data.get("completed_artifacts") or [])
            self._job_id: Optional[str] = resume_data.get("job_id")
            self._manifest: list[dict] = list(resume_data.get("manifest") or [])
            self._config: dict = dict(resume_data.get("config") or {})
            self._model_name: str = str(resume_data.get("model_name") or "")
            self.log(
                f"Resuming bundle training at target {self._target_index + 1}/{len(self._targets)}"
            )
            return

        model_config_uuid = str(params.get("model_config_uuid") or "")
        dataset_uuids = list(params.get("dataset_uuids") or [])
        bundle_name = str(params.get("bundle_name") or "")
        hyperparams = dict(params.get("hyperparams_json") or {})

        if not model_config_uuid:
            raise RuntimeError("model_config_uuid is required")
        if not bundle_name:
            raise RuntimeError("bundle_name is required")

        self._model_config_uuid = model_config_uuid
        self._dataset_uuids = dataset_uuids
        self._bundle_name = bundle_name
        self._hyperparams = hyperparams

        self._targets = await asyncio.to_thread(self._fetch_targets)
        if not self._targets:
            raise RuntimeError(
                f"ModelConfig {model_config_uuid!r} has no training targets (no root node?)"
            )

        self._dataset_uuids_set = await asyncio.to_thread(self._fetch_dataset_uuids_set)

        self._target_index = 0
        self._completed_artifacts = []
        self._job_id = None
        self._manifest = []
        self._config = {}
        self._model_name = ""

        # Seed resume_data immediately so an early crash (before the first job
        # submits) doesn't lose the resolved target list on restart.
        self._resume_data.update({
            "model_config_uuid": self._model_config_uuid,
            "dataset_uuids": self._dataset_uuids,
            "dataset_uuids_set": list(self._dataset_uuids_set),
            "bundle_name": self._bundle_name,
            "hyperparams_json": self._hyperparams,
            "targets": self._targets,
            "target_index": 0,
            "completed_artifacts": [],
            "job_id": None,
            "manifest": [],
            "config": {},
            "model_name": "",
        })

        self.log(f"Resolved {len(self._targets)} bundle training target(s) for {model_config_uuid}")

    def _fetch_targets(self) -> list[dict]:
        """Called in a thread — synchronous generated client."""
        mc_api = ModelConfigsApi(self._db_api_client)
        resp = mc_api.get_model_config_training_targets(self._model_config_uuid)
        if getattr(resp, "status", None) != "success":
            raise RuntimeError(
                f"Could not fetch training targets for {self._model_config_uuid!r}: "
                f"{getattr(resp, 'message', 'unknown error')}"
            )
        return [
            {
                "node_uuid": t.node_uuid,
                "role": t.role,
                "label_uuid": t.label_uuid,
                "model_type": t.model_type,
                "architecture": t.architecture,
                "label_uuids": list(t.label_uuids or []),
                "class_names": list(t.class_names or []),
                "class_remap": dict(t.class_remap or {}),
            }
            for t in (resp.targets or [])
        ]

    def _fetch_dataset_uuids_set(self) -> set[str]:
        """Called in a thread — synchronous generated client."""
        if not self._dataset_uuids:
            return set()
        ds_api = DatasetsApi(self._db_api_client)
        resp = ds_api.get_dataset_image_uuids(dataset_uuid=self._dataset_uuids)
        return set(resp.image_uuids if hasattr(resp, "image_uuids") else [])

    def _target_model_name(self, target: dict) -> str:
        import re

        base = re.sub(r"[^a-zA-Z0-9_]+", "_", self._bundle_name.strip().lower()).strip("_") or "bundle"
        if target["role"] == "top":
            return f"{base}_top"
        return f"{base}_sub_{target['node_uuid'][:8]}"

    def _resolve_target_job(self, target: dict) -> tuple[str, list[dict], dict]:
        """Called in a thread — synchronous generated clients, mirrors
        TrainModelTask's _resolve_* builders but driven by an already-resolved
        bundle target instead of a TrainingConfig's own node lookup."""
        mc_api = ModelConfigsApi(self._db_api_client)
        img_api = ImagesApi(self._db_api_client)
        model_name = self._target_model_name(target)

        if target["model_type"] == "object_detection":
            return self._resolve_target_object_detection(mc_api, target, model_name)

        label_uuids = target["label_uuids"]
        if not label_uuids:
            raise RuntimeError(
                f"Bundle target node={target['node_uuid']} role={target['role']} has no "
                f"in-scope labels — nothing to train (a childless sub-model boundary has "
                f"an empty class set)."
            )
        if len(label_uuids) == 1:
            return self._resolve_target_binary(img_api, target, model_name)
        return self._resolve_target_multi_label(img_api, target, model_name)

    def _resolve_target_object_detection(self, mc_api, target: dict, model_name: str) -> tuple[str, list[dict], dict]:
        payload = BundleTargetDetectionDatasetPayload(dataset_uuids=self._dataset_uuids)
        result = mc_api.get_bundle_target_detection_dataset(
            self._model_config_uuid, target["node_uuid"], payload
        )
        if getattr(result, "status", None) != "success":
            raise RuntimeError(
                f"Could not assemble detection dataset for bundle target node="
                f"{target['node_uuid']!r}: {getattr(result, 'message', 'unknown error')}"
            )
        class_names = list(result.class_names or [])
        images = list(result.images or [])

        if not images:
            raise RuntimeError(
                f"No eligible images found for object_detection bundle target "
                f"node={target['node_uuid']!r}. Ensure images have verified bboxes in scope."
            )

        manifest = [
            {"image_uuid": img.image_uuid, "filename": img.filename, "status": "present"}
            for img in images
        ]
        annotations = {img.image_uuid: img.annotations for img in images}

        self.log(
            f"Resolved object_detection bundle target {target['role']} "
            f"(node={target['node_uuid']}): {len(class_names)} classes, {len(manifest)} images"
        )

        job_config = {
            "trainer_type": "object_detection",
            "model_name": model_name,
            "architecture": target["architecture"],
            "class_names": class_names,
            "annotations": annotations,
            "role": target["role"],
            "model_config_uuid": self._model_config_uuid,
            "model_config_node_uuid": target["node_uuid"],
            "label_uuids": target["label_uuids"],
            **self._hyperparams,
        }
        return model_name, manifest, job_config

    def _resolve_target_binary(self, img_api, target: dict, model_name: str) -> tuple[str, list[dict], dict]:
        label_uuid = target["label_uuids"][0]
        trainer_type = MODEL_TYPE_TO_TRAINER_TYPE.get(target["model_type"], target["model_type"])

        all_rows_resp = img_api.list_images_for_label(
            label_uuid=label_uuid,
            status=[ImageItemStatus.PRESENT, ImageItemStatus.ABSENT],
        )
        all_rows = all_rows_resp.images if hasattr(all_rows_resp, "images") else []
        if self._dataset_uuids_set:
            all_rows = [r for r in all_rows if r.image_uuid in self._dataset_uuids_set]

        if not all_rows:
            raise RuntimeError(
                f"No labelled images found for bundle target label_uuid={label_uuid!r} "
                f"(node={target['node_uuid']})."
            )

        manifest = [
            {
                "image_uuid": r.image_uuid,
                "filename": r.filename,
                "status": r.status.value if hasattr(r.status, "value") else str(r.status),
            }
            for r in all_rows
        ]

        self.log(
            f"Resolved binary bundle target {target['role']} (node={target['node_uuid']}, "
            f"label={label_uuid}): {len(manifest)} images"
        )

        job_config = {
            "trainer_type": trainer_type,
            "label_uuid": label_uuid,
            "model_name": model_name,
            "architecture": target["architecture"],
            "class_names": ["absent", "present"],
            "role": target["role"],
            "model_config_uuid": self._model_config_uuid,
            "model_config_node_uuid": target["node_uuid"],
            **self._hyperparams,
        }
        return model_name, manifest, job_config

    def _resolve_target_multi_label(self, img_api, target: dict, model_name: str) -> tuple[str, list[dict], dict]:
        label_uuids = list(target["label_uuids"])
        uuid_to_filename: dict[str, str] = {}
        label_targets: dict[str, dict[str, int]] = {}

        for luuid in label_uuids:
            rows_resp = img_api.list_images_for_label(
                label_uuid=luuid,
                status=[ImageItemStatus.PRESENT, ImageItemStatus.ABSENT],
            )
            rows = rows_resp.images if hasattr(rows_resp, "images") else []
            if self._dataset_uuids_set:
                rows = [r for r in rows if r.image_uuid in self._dataset_uuids_set]
            for r in rows:
                uuid_to_filename[r.image_uuid] = r.filename
                status_val = r.status.value if hasattr(r.status, "value") else str(r.status)
                label_targets.setdefault(r.image_uuid, {})[luuid] = (
                    1 if status_val == ImageItemStatus.PRESENT.value else 0
                )

        if not uuid_to_filename:
            raise RuntimeError(
                f"No labeled images found across {len(label_uuids)} label(s) for bundle "
                f"target (node={target['node_uuid']})."
            )

        manifest = [
            {"image_uuid": uuid, "filename": fname, "status": "present"}
            for uuid, fname in uuid_to_filename.items()
        ]

        self.log(
            f"Resolved multi_label bundle target {target['role']} (node={target['node_uuid']}): "
            f"{len(label_uuids)} labels, {len(manifest)} images"
        )

        job_config = {
            "trainer_type": "multi_label_classifier",
            "model_name": model_name,
            "architecture": target["architecture"],
            "label_uuids": label_uuids,
            "label_targets": label_targets,
            "role": target["role"],
            "model_config_uuid": self._model_config_uuid,
            "model_config_node_uuid": target["node_uuid"],
            **self._hyperparams,
        }
        return model_name, manifest, job_config

    async def _run(self) -> dict[str, Any]:
        while self._target_index < len(self._targets):
            target = self._targets[self._target_index]

            if not self._job_id:
                # Not resuming a specific in-flight job for this target — resolve it.
                self._model_name, self._manifest, self._config = await asyncio.to_thread(
                    self._resolve_target_job, target
                )

            self.log(
                f"Training bundle target {self._target_index + 1}/{len(self._targets)}: "
                f"role={target['role']} node={target['node_uuid']}"
            )

            result = await self._run_one_job()

            if result.get("cancelled"):
                return result

            self._completed_artifacts.append({
                "role": target["role"],
                "node_uuid": target["node_uuid"],
                "label_uuid": target["label_uuid"],
                "label_uuids": target["label_uuids"],
                "model_name": result.get("model_name"),
                "val_accuracy": result.get("val_accuracy"),
            })

            # Advance to the next target; clear per-job state so the next
            # iteration resolves a fresh job instead of _run_one_job's own
            # resume-in-place branch.
            self._target_index += 1
            self._job_id = None
            self._manifest = []
            self._config = {}
            self._model_name = ""

            self._resume_data.update({
                "target_index": self._target_index,
                "completed_artifacts": self._completed_artifacts,
                "job_id": None,
                "manifest": [],
                "config": {},
                "model_name": "",
            })
            self._data_ready_flag.set()

        bundle = await asyncio.to_thread(self._create_bundle)
        self.log(f"Bundle {bundle['uuid']} created with {len(self._completed_artifacts)} artifact(s)")
        return {"bundle_uuid": bundle["uuid"], "artifacts": self._completed_artifacts}

    def _create_bundle(self) -> dict:
        """Called in a thread — synchronous generated client."""
        mc_api = ModelConfigsApi(self._db_api_client)
        payload = ModelBundleCreate(
            name=self._bundle_name,
            model_config_uuid=self._model_config_uuid,
            bundle_path=str(_MODELS_DIR),
            manifest_json={"artifacts": self._completed_artifacts},
            training_config_uuids=[],
        )
        resp = mc_api.create_model_bundle(payload)
        if getattr(resp, "status", None) != "success" or resp.bundle is None:
            raise RuntimeError(f"Could not create ModelBundle: {getattr(resp, 'message', 'unknown error')}")
        return {"uuid": resp.bundle.uuid, "name": resp.bundle.name}

    async def _deinit(self) -> None:
        pass

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        return {
            "model_config_uuid": {
                "type": "str",
                "required": True,
                "help": "ModelConfig to train a full bundle for (top model + every sub-model)",
            },
            "dataset_uuids": {
                "type": "list",
                "required": True,
                "help": "Dataset(s) to train from, applied to every target",
            },
            "bundle_name": {
                "type": "str",
                "required": True,
                "help": "Name for the resulting ModelBundle",
            },
            "hyperparams_json": {
                "type": "dict",
                "required": False,
                "default": {},
                "help": "Shared training hyperparameters (batch_size, imgsz, max_epochs, "
                "patience) applied to every target",
            },
        }
