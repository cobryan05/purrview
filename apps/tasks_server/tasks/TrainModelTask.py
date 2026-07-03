"""Train a model via the training server, driven by a TrainingConfig.

Push-based lifecycle (no shared filesystem required):
  1. Resolve TrainingConfig → node → label, manifest
  2. POST /api/training/jobs           (job spec + manifest)  → awaiting_data
  3. GET  /jobs/{id}/missing-images    → uuids not yet cached
  4. POST /jobs/{id}/images            (streamed tar batches, ≤500 images each)
  5. POST /jobs/{id}/start             → queued
  6. GET  /jobs/{id}/stream (SSE)      → relay progress events
  7. On "done": GET /jobs/{id}/artifact → write ONNX + JSON to $MODELS_ROOT
  8. On cancel: DELETE /jobs/{id}

The training server is network-isolated: it never calls back to db_server.
Everything flows one way: tasks_server pushes, training server responds.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import db_client
import training_client
from db_client.api.datasets_api import DatasetsApi
from db_client.api.images_api import ImagesApi
from db_client.api.model_configs_api import ModelConfigsApi
from db_client.api.training_configs_api import TrainingConfigsApi
from db_client.models.image_item_status import ImageItemStatus

from apps import APPS, APPS_CONFIG
from apps.helpers.model_type_map import MODEL_TYPE_TO_TRAINER_TYPE

from ._training_job_runner import TrainingJobRunnerMixin
from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@register_task()
class TrainModelTask(Task, TrainingJobRunnerMixin):
    # Epochs can take minutes each — extend heartbeat tolerance.
    HEARTBEAT_INTERVAL = 60

    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        # ── API client setup ────────────────────────────────────────────────
        db_cfg = APPS_CONFIG[APPS.DB_SERVER]
        db_base_url = f"http://{db_cfg.host}:{db_cfg.port}"
        self._db_api_client = db_client.ApiClient(db_client.Configuration(host=db_base_url))

        training_cfg = APPS_CONFIG[APPS.TRAINING_SERVER]
        self._training_base_url: str = f"http://{training_cfg.host}:{training_cfg.port}"
        tc_config = training_client.Configuration(host=self._training_base_url)
        self._api_client = training_client.ApiClient(tc_config)

        # ── Resume from a previous run ───────────────────────────────────────
        self._job_id: Optional[str] = resume_data.get("job_id") if resume_data else None
        if self._job_id:
            self.log(f"Resuming training job {self._job_id}")
            # Restore state needed for the SSE relay and artifact download.
            self._model_name: str = str(resume_data.get("model_name", "model"))
            self._manifest: list[dict] = resume_data.get("manifest") or []
            self._config: dict = resume_data.get("config") or {}
            return

        # ── Resolve training config (first run) ─────────────────────────────
        training_config_uuid: Optional[str] = params.get("training_config_uuid")

        if training_config_uuid:
            self._model_name, self._manifest, self._config = await asyncio.to_thread(
                self._resolve_training_config, training_config_uuid
            )
        else:
            # Legacy / test path: accept raw params directly.
            self._model_name = str(params.get("model_name", "model"))
            self._config = {
                "trainer_type": str(params.get("trainer_type", "binary_image_classifier")),
                "label_uuid": str(params.get("label_uuid", "")),
                "model_name": self._model_name,
                "architecture": str(params.get("architecture", "")),
                "max_epochs": int(params.get("max_epochs", 50)),
            }
            self._manifest = []

        logger.info(
            "TrainModelTask init: model=%s manifest=%d images",
            self._model_name,
            len(self._manifest),
        )

    def _resolve_training_config(self, training_config_uuid: str) -> tuple[str, list[dict], dict]:
        """Resolve a TrainingConfig UUID to (model_name, manifest, job_config).

        Called in a thread (not async) so it can use the synchronous generated clients.
        """
        tc_api = TrainingConfigsApi(self._db_api_client)
        mc_api = ModelConfigsApi(self._db_api_client)
        img_api = ImagesApi(self._db_api_client)
        ds_api = DatasetsApi(self._db_api_client)

        # 1. Get training config.
        tc = tc_api.get_training_config(training_config_uuid)
        if tc is None or tc.status != "success" if hasattr(tc, "status") else False:
            raise RuntimeError(f"Training config {training_config_uuid!r} not found")

        # The generated client wraps results in a response object.
        # Handle both direct TrainingConfigRead and wrapped responses.
        tc_data = tc if hasattr(tc, "dataset_uuids") else tc.config if hasattr(tc, "config") else None
        if tc_data is None:
            raise RuntimeError(f"Could not read training config {training_config_uuid!r}")

        dataset_uuids: list[str] = list(tc_data.dataset_uuids or [])
        hyperparams: dict = dict(tc_data.hyperparams_json or {})
        node_uuid: Optional[str] = tc_data.model_config_node_uuid
        model_name: str = hyperparams.pop("model_name", None) or tc_data.name.replace(" ", "_").lower()

        if not node_uuid:
            raise RuntimeError(
                f"TrainingConfig {training_config_uuid!r} has no model_config_node_uuid"
            )

        # 2. Get model config node → label_uuid, model_type, architecture.
        node = mc_api.get_model_config_node(node_uuid)
        if node is None:
            raise RuntimeError(f"ModelConfigNode {node_uuid!r} not found")

        # node might be a ModelConfigNodeRead directly.
        node_data = node if hasattr(node, "model_type") else getattr(node, "data", node)
        label_uuid: str = node_data.label_uuid or ""
        model_type: str = node_data.model_type
        architecture: str = node_data.architecture
        model_config_uuid: str = node_data.model_config_uuid

        # Bundle-target identity (role + in-scope label_uuids), sourced from the
        # backend's single authority (derive_bundle_training_targets) so trained
        # artifacts can self-describe their place in the model tree. Threaded into
        # every job_config below and from there into each trainer's sidecar.
        role, bundle_label_uuids = self._resolve_bundle_target_fields(
            mc_api, model_config_uuid, node_uuid, label_uuid
        )

        # Object-detection nodes use a bbox-based dataset assembly; dispatch before the
        # image-label classifier paths below.
        if model_type == "object_detection":
            return self._resolve_object_detection(
                training_config_uuid=training_config_uuid,
                tc_api=tc_api,
                model_name=model_name,
                architecture=architecture,
                hyperparams=hyperparams,
                role=role,
                model_config_uuid=model_config_uuid,
                node_uuid=node_uuid,
                bundle_label_uuids=bundle_label_uuids,
            )

        # Get dataset image UUID set once (shared by both paths below).
        dataset_uuids_set: set[str] = set()
        if dataset_uuids:
            dataset_image_uuids_resp = ds_api.get_dataset_image_uuids(dataset_uuid=dataset_uuids)
            dataset_uuids_set = set(
                dataset_image_uuids_resp.image_uuids
                if hasattr(dataset_image_uuids_resp, "image_uuids")
                else []
            )

        if not label_uuid:
            # Root node → multi-label: train all labeled child nodes as one model.
            return self._resolve_root_node(
                node_uuid=node_uuid,
                mc_api=mc_api,
                img_api=img_api,
                dataset_uuids=dataset_uuids,
                dataset_uuids_set=dataset_uuids_set,
                model_name=model_name,
                architecture=architecture,
                hyperparams=hyperparams,
                role=role,
                model_config_uuid=model_config_uuid,
                model_config_node_uuid=node_uuid,
            )

        # Single labeled node → binary classifier.
        trainer_type: str = MODEL_TYPE_TO_TRAINER_TYPE.get(model_type, model_type)

        # 3. Get all present/absent image-label rows for this label.
        all_rows_resp = img_api.list_images_for_label(
            label_uuid=label_uuid,
            status=[ImageItemStatus.PRESENT, ImageItemStatus.ABSENT],
        )
        all_rows = all_rows_resp.images if hasattr(all_rows_resp, "images") else []

        if dataset_uuids:
            before = len(all_rows)
            all_rows = [r for r in all_rows if r.image_uuid in dataset_uuids_set]
            logger.debug(
                "Dataset filter: %d → %d rows (dataset_set size=%d)",
                before, len(all_rows), len(dataset_uuids_set),
            )

        if not all_rows:
            raise RuntimeError(
                f"No labelled images found for label_uuid={label_uuid!r} "
                f"with {len(dataset_uuids)} dataset(s). "
                f"Found {len(all_rows_resp.images if hasattr(all_rows_resp, 'images') else [])} "
                f"labeled rows before dataset filter (dataset_set size={len(dataset_uuids_set)})."
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
            f"Resolved training config: label={label_uuid} arch={architecture} "
            f"trainer={trainer_type} manifest={len(manifest)} images "
            f"datasets={len(dataset_uuids)}"
        )

        job_config = {
            "trainer_type": trainer_type,
            "label_uuid": label_uuid,
            "model_name": model_name,
            "architecture": architecture,
            "class_names": ["absent", "present"],
            "role": role,
            "model_config_uuid": model_config_uuid,
            "model_config_node_uuid": node_uuid,
            **hyperparams,
        }
        return model_name, manifest, job_config

    def _resolve_bundle_target_fields(
        self, mc_api, model_config_uuid: str, node_uuid: str, label_uuid: str
    ) -> tuple[str, list[str]]:
        """Look up this node's role + in-scope label_uuids from the backend's single
        source of truth (GET .../training-targets, backed by
        DbClient.derive_bundle_training_targets).

        Falls back to a locally-derived role/label list if the node isn't a
        recognised bundle-boundary target (e.g. its spec is identical to its
        inherited spec, so it's not its own target) — training must never fail on
        this lookup, it only enriches descriptive metadata.
        """
        try:
            resp = mc_api.get_model_config_training_targets(model_config_uuid)
            targets = resp.targets if hasattr(resp, "targets") else []
        except Exception:
            logger.warning(
                "Could not fetch bundle training targets for model_config_uuid=%s",
                model_config_uuid,
                exc_info=True,
            )
            targets = []

        for target in targets:
            if getattr(target, "node_uuid", None) == node_uuid:
                return target.role, list(target.label_uuids or [])

        role = "top" if not label_uuid else "submodel"
        return role, ([label_uuid] if label_uuid else [])

    def _resolve_root_node(
        self,
        node_uuid: str,
        mc_api,
        img_api,
        dataset_uuids: list[str],
        dataset_uuids_set: set[str],
        model_name: str,
        architecture: str,
        hyperparams: dict,
        role: str,
        model_config_uuid: str,
        model_config_node_uuid: str,
    ) -> tuple[str, list[dict], dict]:
        """Resolve a root node to a multi-label training job.

        Finds the parent model config, collects all labeled child nodes, builds a
        combined image manifest and per-label status map, and returns a config for
        the ``multi_label_classifier`` trainer.
        """
        # Find the model config that owns this root node.
        all_configs_resp = mc_api.list_model_configs()
        all_configs = getattr(all_configs_resp, "configs", None) or []

        parent_config = None
        for cfg in all_configs:
            nodes = getattr(cfg, "nodes", None) or []
            if any(getattr(n, "uuid", None) == node_uuid for n in nodes):
                parent_config = cfg
                break

        if parent_config is None:
            raise RuntimeError(
                f"Could not find a model config containing node {node_uuid!r}"
            )

        labeled_nodes = [
            n for n in (getattr(parent_config, "nodes", None) or [])
            if getattr(n, "label_uuid", None)
        ]
        if not labeled_nodes:
            raise RuntimeError(
                f"Root node {node_uuid!r} has no labeled child nodes to train"
            )

        label_uuids_list = [n.label_uuid for n in labeled_nodes]

        # Collect per-label image data.
        uuid_to_filename: dict[str, str] = {}
        label_targets: dict[str, dict[str, int]] = {}

        for node in labeled_nodes:
            luuid = node.label_uuid
            rows_resp = img_api.list_images_for_label(
                label_uuid=luuid,
                status=[ImageItemStatus.PRESENT, ImageItemStatus.ABSENT],
            )
            rows = rows_resp.images if hasattr(rows_resp, "images") else []
            raw_count = len(rows)
            if dataset_uuids:
                rows = [r for r in rows if r.image_uuid in dataset_uuids_set]
            logger.debug(
                "Label %s: %d labeled rows, %d after dataset filter",
                luuid, raw_count, len(rows),
            )

            for r in rows:
                uuid_to_filename[r.image_uuid] = r.filename
                if r.image_uuid not in label_targets:
                    label_targets[r.image_uuid] = {}
                status_val = r.status.value if hasattr(r.status, "value") else str(r.status)
                label_targets[r.image_uuid][luuid] = 1 if status_val == ImageItemStatus.PRESENT.value else 0

        if not uuid_to_filename:
            raise RuntimeError(
                f"No labeled images found across {len(labeled_nodes)} label(s) "
                f"with {len(dataset_uuids)} dataset(s). "
                f"dataset_set size={len(dataset_uuids_set)}. "
                f"Check that labeled images are included in the selected dataset(s)."
            )

        # Manifest: unique images (status field unused by multi_label_classifier).
        manifest = [
            {"image_uuid": uuid, "filename": fname, "status": "present"}
            for uuid, fname in uuid_to_filename.items()
        ]

        self.log(
            f"Resolved root node: {len(labeled_nodes)} labels "
            f"({', '.join(label_uuids_list)}), "
            f"{len(manifest)} unique images, datasets={len(dataset_uuids)}"
        )

        job_config = {
            "trainer_type": "multi_label_classifier",
            "model_name": model_name,
            "architecture": architecture,
            "label_uuids": label_uuids_list,
            "label_targets": label_targets,
            "role": role,
            "model_config_uuid": model_config_uuid,
            "model_config_node_uuid": model_config_node_uuid,
            **hyperparams,
        }
        return model_name, manifest, job_config

    def _resolve_object_detection(
        self,
        training_config_uuid: str,
        tc_api,
        model_name: str,
        architecture: str,
        hyperparams: dict,
        role: str,
        model_config_uuid: str,
        node_uuid: str,
        bundle_label_uuids: list[str],
    ) -> tuple[str, list[dict], dict]:
        """Resolve an object_detection training config to a detection training job.

        Calls the detection-dataset endpoint on db_server to assemble the bbox-based
        dataset (class list + per-image annotations) and builds a manifest + job config
        for the object_detection trainer.
        """
        result = tc_api.get_training_config_detection_dataset(training_config_uuid)
        # Result is a DetectionDatasetResponse (generated model).
        class_names = list(result.class_names or [])
        images = list(result.images or [])

        if not images:
            raise RuntimeError(
                f"No eligible images found for object_detection training config "
                f"{training_config_uuid!r}. Ensure images have verified bboxes in scope."
            )

        manifest = [
            {"image_uuid": img.image_uuid, "filename": img.filename, "status": "present"}
            for img in images
        ]
        annotations = {
            img.image_uuid: img.annotations
            for img in images
        }

        self.log(
            f"Resolved object_detection config: {len(class_names)} classes "
            f"({', '.join(class_names)}), {len(manifest)} images, arch={architecture}"
        )

        job_config = {
            "trainer_type": "object_detection",
            "model_name": model_name,
            "architecture": architecture,
            "class_names": class_names,
            "annotations": annotations,
            "role": role,
            "model_config_uuid": model_config_uuid,
            "model_config_node_uuid": node_uuid,
            "label_uuids": bundle_label_uuids,
            **hyperparams,
        }
        return model_name, manifest, job_config

    async def _run(self) -> dict[str, Any]:
        return await self._run_one_job()

    async def _deinit(self) -> None:
        pass

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        return {
            "training_config_uuid": {
                "type": "str",
                "required": False,
                "help": "UUID of the TrainingConfig to train from (preferred)",
            },
            "trainer_type": {
                "type": "str",
                "required": False,
                "default": "binary_image_classifier",
                "options": ["binary_image_classifier", "multi_label_classifier"],
                "help": "Which training algorithm to use (legacy; use training_config_uuid instead). Derived automatically from node type when training_config_uuid is set.",
            },
            "label_uuid": {
                "type": "str",
                "required": False,
                "help": "UUID of the label (legacy; use training_config_uuid instead)",
            },
            "model_name": {
                "type": "str",
                "required": False,
                "help": "Output model filename stem (legacy; use training_config_uuid instead)",
            },
            "max_epochs": {
                "type": "int",
                "required": False,
                "default": 50,
                "help": "Maximum number of training epochs",
            },
        }
