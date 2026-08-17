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
from db_client.models.training_target_detection_dataset_payload import (
    TrainingTargetDetectionDatasetPayload,
)

from apps import APPS, APPS_CONFIG
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.model_type_map import resolve_trainer_type

from ._classifier_job_builders import (
    build_binary_classifier_job,
    build_multi_label_classifier_job,
)
from ._training_job_runner import TrainingJobRunnerMixin, format_eta
from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


@register_task()
class TrainModelTask(Task, TrainingJobRunnerMixin):
    # Epochs can take minutes each — extend heartbeat tolerance.
    HEARTBEAT_INTERVAL = 60

    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        # ── API client setup ────────────────────────────────────────────────
        db_cfg = APPS_CONFIG[APPS.DB_SERVER]
        db_base_url = f"http://{db_cfg.host}:{db_cfg.port}"
        self._db_api_client = db_client.ApiClient(db_client.Configuration(host=db_base_url))

        # Resolved through the manager, not APPS_CONFIG: Settings → Servers can
        # point training at another host, and only the manager tracks that override.
        self._training_base_url: str = self._manager.get_training_base_url()
        tc_config = training_client.Configuration(host=self._training_base_url)
        self._api_client = training_client.ApiClient(tc_config)

        # ── Per-epoch metric history (drives the training-curve widgets) ─────
        # Seeded from resume_data so a job that is interrupted and resumed keeps
        # the curve it had already drawn. Set before the resume early-return
        # below: both paths need it.
        self._epoch_history: list[dict[str, Any]] = list((resume_data or {}).get("epoch_history") or [])
        self._total_epochs: Optional[int] = None

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

        # "Remember exact image list" opt-out. Stored in hyperparams_json (that's
        # the training config's only free-form blob) but popped out and threaded as
        # a named argument rather than riding the **hyperparams merge — see
        # NOTES.md on trainers silently disagreeing over merged key names. Absent
        # means ON: recording the manifest is what every trained model has always
        # done, and an opt-out must not change that by default.
        record_image_manifest: bool = bool(hyperparams.pop("record_image_manifest", True))

        if not node_uuid:
            raise RuntimeError(f"TrainingConfig {training_config_uuid!r} has no model_config_node_uuid")

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

        # Training-target identity (role + in-scope label_uuids), sourced from
        # the backend's single authority (derive_training_targets) so trained
        # artifacts can self-describe their place in the model tree. Threaded into
        # every job_config below and from there into each trainer's sidecar.
        role, target_label_uuids, class_remap = self._resolve_training_target_fields(
            mc_api, model_config_uuid, node_uuid, label_uuid
        )

        # Object-detection nodes use a bbox-based dataset assembly; dispatch before the
        # image-label classifier paths below.
        if model_type == "object_detection":
            return self._resolve_object_detection(
                mc_api=mc_api,
                model_name=model_name,
                architecture=architecture,
                hyperparams=hyperparams,
                role=role,
                model_config_uuid=model_config_uuid,
                node_uuid=node_uuid,
                target_label_uuids=target_label_uuids,
                class_remap=class_remap,
                dataset_uuids=dataset_uuids,
                record_image_manifest=record_image_manifest,
            )

        # Get dataset image UUID set once (shared by both paths below).
        dataset_uuids_set: set[str] = set()
        if dataset_uuids:
            dataset_image_uuids_resp = ds_api.get_dataset_image_uuids(dataset_uuid=dataset_uuids)
            dataset_uuids_set = set(
                dataset_image_uuids_resp.image_uuids if hasattr(dataset_image_uuids_resp, "image_uuids") else []
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
                record_image_manifest=record_image_manifest,
            )

        # Single labeled node → binary classifier.
        trainer_type: str = resolve_trainer_type(model_type, architecture)

        model_name, manifest, job_config = build_binary_classifier_job(
            img_api,
            label_uuid=label_uuid,
            trainer_type=trainer_type,
            model_name=model_name,
            architecture=architecture,
            role=role,
            model_config_uuid=model_config_uuid,
            model_config_node_uuid=node_uuid,
            hyperparams=hyperparams,
            dataset_uuids=dataset_uuids,
            dataset_uuids_set=dataset_uuids_set,
            record_image_manifest=record_image_manifest,
        )
        self.log(
            f"Resolved training config: label={label_uuid} arch={architecture} "
            f"trainer={trainer_type} manifest={len(manifest)} images "
            f"datasets={len(dataset_uuids)}"
        )
        return model_name, manifest, job_config

    def _resolve_training_target_fields(
        self, mc_api, model_config_uuid: str, node_uuid: str, label_uuid: str
    ) -> tuple[str, list[str], dict[str, str]]:
        """Look up this node's role + in-scope label_uuids (+ class_remap) from the
        backend's single source of truth (GET .../training-targets, backed by
        DbClient.derive_training_targets).

        Falls back to a locally-derived role/label list if the node isn't a
        recognised sub-model boundary target (e.g. its spec is identical to its
        inherited spec, so it's not its own target) — training must never fail on
        this lookup, it only enriches descriptive metadata.
        """
        try:
            resp = mc_api.get_model_config_training_targets(model_config_uuid)
            targets = resp.targets if hasattr(resp, "targets") else []
        except Exception:
            logger.warning(
                "Could not fetch training targets for model_config_uuid=%s",
                model_config_uuid,
                exc_info=True,
            )
            targets = []

        for target in targets:
            if getattr(target, "node_uuid", None) == node_uuid:
                return target.role, list(target.label_uuids or []), dict(target.class_remap or {})

        role = "top" if not label_uuid else "submodel"
        return role, ([label_uuid] if label_uuid else []), {}

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
        record_image_manifest: bool = True,
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
            raise RuntimeError(f"Could not find a model config containing node {node_uuid!r}")

        labeled_nodes = [n for n in (getattr(parent_config, "nodes", None) or []) if getattr(n, "label_uuid", None)]
        if not labeled_nodes:
            raise RuntimeError(f"Root node {node_uuid!r} has no labeled child nodes to train")

        label_uuids_list = [n.label_uuid for n in labeled_nodes]

        model_name, manifest, job_config = build_multi_label_classifier_job(
            img_api,
            label_uuids=label_uuids_list,
            model_name=model_name,
            architecture=architecture,
            role=role,
            model_config_uuid=model_config_uuid,
            model_config_node_uuid=model_config_node_uuid,
            hyperparams=hyperparams,
            dataset_uuids=dataset_uuids,
            dataset_uuids_set=dataset_uuids_set,
            record_image_manifest=record_image_manifest,
        )
        self.log(
            f"Resolved root node: {len(labeled_nodes)} labels "
            f"({', '.join(label_uuids_list)}), "
            f"{len(manifest)} unique images, datasets={len(dataset_uuids)}"
        )
        return model_name, manifest, job_config

    def _resolve_object_detection(
        self,
        mc_api,
        model_name: str,
        architecture: str,
        hyperparams: dict,
        role: str,
        model_config_uuid: str,
        node_uuid: str,
        target_label_uuids: list[str],
        class_remap: dict[str, str],
        dataset_uuids: list[str],
        record_image_manifest: bool = True,
    ) -> tuple[str, list[dict], dict]:
        """Resolve an object_detection training config to a detection training job.

        Uses the training-target detection-dataset endpoint (and its
        ``target_label_uuids`` ordering), so the class index assigned to each
        label here is guaranteed to match the order the sidecar's
        ``label_uuids`` are written in — the two must agree, or the label-matchup
        UI pairs the wrong name with the wrong label UUID.
        """
        # Negative-sampling cap, stored alongside the rest of the training config's
        # free-form hyperparams. Popped out (not spread into job_config below) since
        # it's a dataset-assembly concern the trainer itself never needs to see.
        negative_ratio = hyperparams.pop("negative_ratio", None)
        negative_ratio_floor = hyperparams.pop("negative_ratio_floor", None)

        payload = TrainingTargetDetectionDatasetPayload(
            dataset_uuids=dataset_uuids,
            label_uuids=target_label_uuids,
            class_remap=class_remap,
            negative_ratio=negative_ratio,
            negative_ratio_floor=negative_ratio_floor,
        )
        result = mc_api.get_training_target_detection_dataset(model_config_uuid, node_uuid, payload)
        if getattr(result, "status", None) != "success":
            raise RuntimeError(
                f"Could not assemble detection dataset for node={node_uuid!r}: "
                f"{getattr(result, 'message', 'unknown error')}"
            )
        # Result is a DetectionDatasetResponse (generated model).
        class_names = list(result.class_names or [])
        images = list(result.images or [])

        if not images:
            raise RuntimeError(
                f"No eligible images found for object_detection node={node_uuid!r}. "
                f"Ensure images have verified bboxes in scope."
            )

        manifest = [{"image_uuid": img.image_uuid, "filename": img.filename, "status": "present"} for img in images]
        annotations = {img.image_uuid: img.annotations for img in images}

        self.log(
            f"Resolved object_detection config: {len(class_names)} classes "
            f"({', '.join(class_names)}), {len(manifest)} images, arch={architecture}"
        )

        job_config = {
            "trainer_type": resolve_trainer_type("object_detection", architecture),
            "model_name": model_name,
            "architecture": architecture,
            "class_names": class_names,
            "annotations": annotations,
            "role": role,
            "model_config_uuid": model_config_uuid,
            "model_config_node_uuid": node_uuid,
            "label_uuids": target_label_uuids,
            **hyperparams,
            "record_image_manifest": record_image_manifest,
        }
        return model_name, manifest, job_config

    async def _run(self) -> dict[str, Any]:
        return await self._run_one_job()

    async def _deinit(self) -> None:
        pass

    @classmethod
    def status_schema(cls) -> list[dict[str, Any]]:
        """Progress + live epoch stats + the training curve + logs.

        Loss and accuracy share one plot with two y-axes. They have unrelated
        scales (loss is unbounded, accuracy is a 0–1 fraction), so where the
        curves cross is an artefact of how each axis was zeroed and means
        nothing — the widget answers that by painting each axis' ticks in its
        own series' colour and naming the axis in the legend, rather than by
        splitting the run across two charts you have to read in alternation.

        val_accuracy is held: the detection trainers validate on an interval
        (libreyolo's default is every 10th epoch), so it is measured far more
        rarely than train_loss and the chart carries its last measurement
        forward as a dashed step between dots.
        """
        return [
            {"id": "progress", "type": "progress_bar", "source": "summary"},
            {"id": "statusMessage", "type": "text", "source": "summary"},
            {
                "id": "epoch_stats",
                "type": "stat_grid",
                "source": "status",
                "label": "Latest epoch",
                # Superset of what any trainer can report. get_status_data() omits
                # keys the current trainer doesn't produce (e.g. val_loss for the
                # detection trainers) and the grid renders those as "—" rather
                # than a misleading 0.
                "fields": {
                    "epoch": "Epoch",
                    # Only present once two epoch boundaries have been timed in
                    # this process — see estimate_eta_seconds.
                    "eta": "Est. remaining",
                    "train_loss": "Train loss",
                    "val_loss": "Val loss",
                    "val_accuracy": "Val accuracy",
                    "best_val_accuracy": "Best val acc.",
                    "best_val_loss": "Best val loss",
                },
            },
            {
                "id": "loss_curve",
                "type": "line_chart",
                "source": "status",
                "label": "Training curves by epoch",
                "x": "epoch",
                "xLabel": "Epoch",
                "yMin": 0,
                "yRight": {"min": 0, "max": 1, "valueFormat": "percent"},
                "series": [
                    {"key": "train_loss", "label": "Train loss"},
                    {"key": "val_loss", "label": "Val loss"},
                    # One label for two different measures: the classifiers
                    # report a plain accuracy, the detection trainers put mAP50
                    # in the same field. status_schema() is static per typename,
                    # so it can't name whichever one this instance produced.
                    {
                        "key": "val_accuracy",
                        "label": "Val accuracy / mAP50",
                        "axis": "right",
                        "hold": True,
                    },
                ],
            },
            {"id": "log", "type": "log_tail", "source": "summary", "lines": 5},
        ]

    def get_status_data(self) -> dict[str, Any]:
        """Live epoch stats + curve points. Pure in-memory reads, no I/O."""
        history: list[dict[str, Any]] = getattr(self, "_epoch_history", None) or []
        if not history:
            return {}

        latest = history[-1]
        total_epochs = getattr(self, "_total_epochs", None)
        stats: dict[str, Any] = {
            "epoch": f"{latest['epoch']} / {total_epochs}" if total_epochs else str(latest["epoch"]),
        }
        eta_seconds = self.estimate_eta_seconds()
        if eta_seconds is not None:
            stats["eta"] = format_eta(eta_seconds)
        if "train_loss" in latest:
            stats["train_loss"] = f"{latest['train_loss']:.4f}"
        if "val_loss" in latest:
            stats["val_loss"] = f"{latest['val_loss']:.4f}"
        if "val_accuracy" in latest:
            stats["val_accuracy"] = f"{latest['val_accuracy'] * 100:.1f}%"

        accuracies = [s["val_accuracy"] for s in history if "val_accuracy" in s]
        if accuracies:
            stats["best_val_accuracy"] = f"{max(accuracies) * 100:.1f}%"
        val_losses = [s["val_loss"] for s in history if "val_loss" in s]
        if val_losses:
            stats["best_val_loss"] = f"{min(val_losses):.4f}"

        # One chart now, reading the whole sample list and picking its own series
        # out of it. Older instances' persisted snapshots still carry a separate
        # accuracy_curve key; the schema no longer asks for it, and an extra key
        # in the data is ignored.
        return {"epoch_stats": stats, "loss_curve": {"points": history}}

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
