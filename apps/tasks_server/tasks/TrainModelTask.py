"""Train a model via the training server, driven by a TrainingConfig.

Push-based lifecycle (no shared filesystem required):
  1. Resolve TrainingConfig → node → label, manifest
  2. POST /api/training/jobs           (job spec + manifest)  → awaiting_data
  3. GET  /jobs/{id}/missing-images    → uuids not yet cached
  4. POST /jobs/{id}/images            (streamed tar batches, ≤500 images each)
  5. POST /jobs/{id}/start             → queued
  6. GET  /jobs/{id}/stream (SSE)      → relay progress events
  7. On "done": GET /jobs/{id}/artifact → write ONNX + JSON to /models
  8. On cancel: DELETE /jobs/{id}

The training server is network-isolated: it never calls back to db_server.
Everything flows one way: tasks_server pushes, training server responds.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Any, Optional

import httpx
import training_client
from training_client.api.training_api import TrainingApi
from training_client.models.manifest_entry import ManifestEntry
from training_client.models.submit_job_payload import SubmitJobPayload

import db_client
from db_client.api.datasets_api import DatasetsApi
from db_client.api.images_api import ImagesApi
from db_client.api.model_configs_api import ModelConfigsApi
from db_client.api.training_configs_api import TrainingConfigsApi
from db_client.models.image_item_status import ImageItemStatus

from apps import APPS, APPS_CONFIG, FILES_ROOT
from apps.helpers.model_type_map import MODEL_TYPE_TO_TRAINER_TYPE

from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

_MODELS_DIR = Path("/models")
_TAR_BATCH_SIZE = 500  # max images per tar POST


@register_task()
class TrainModelTask(Task):
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
            **hyperparams,
        }
        return model_name, manifest, job_config

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
            **hyperparams,
        }
        return model_name, manifest, job_config

    async def _run(self) -> dict[str, Any]:
        if self._job_id:
            status = await self._check_remote_job_status(self._job_id)
            self.log(f"Resuming training job {self._job_id} (remote status: {status})")
            await self.heartbeat(force=True)

            if status == "done":
                await self._save_artifact(self._job_id)
                return {"job_id": self._job_id, "model_name": self._model_name}

            if status == "interrupted":
                # Server restarted mid-job. Restart the job and consume SSE.
                # Training will resume from the last epoch checkpoint.
                self.log(f"Job {self._job_id} was interrupted; restarting on training server")
                async with httpx.AsyncClient(timeout=30.0) as client:
                    r = await client.post(
                        f"{self._training_base_url}/api/training/jobs/{self._job_id}/restart"
                    )
                    r.raise_for_status()
                return await self._consume_sse(self._job_id)

            if status in ("running", "queued", "awaiting_data"):
                # Already in progress — just reconnect to the stream.
                return await self._consume_sse(self._job_id)

            if status in ("error", "cancelled"):
                raise RuntimeError(
                    f"Training job {self._job_id} is in terminal state: {status}"
                )

            # Status is "not_found" (404) or unknown — server restarted without
            # persisted state. Re-submit the full job; images are likely still cached.
            self.log(
                f"Job {self._job_id} not found on training server; resubmitting from scratch"
            )
            self._job_id = None

        # ── Step 1: Submit job ───────────────────────────────────────────────
        payload = SubmitJobPayload(
            trainer_type=self._config.get("trainer_type", "binary_image_classifier"),
            label_uuid=self._config.get("label_uuid", ""),
            model_name=self._model_name,
            architecture=self._config.get("architecture", ""),
            class_names=self._config.get("class_names") or ["absent", "present"],
            hyperparams={
                k: v for k, v in self._config.items()
                if k not in {"trainer_type", "label_uuid", "model_name", "architecture", "class_names"}
            },
            manifest=[
                ManifestEntry(
                    image_uuid=e["image_uuid"],
                    filename=e["filename"],
                    status=e["status"],
                )
                for e in self._manifest
            ],
        )
        resp = await asyncio.to_thread(
            TrainingApi(self._api_client).submit_training_job,
            payload,
        )
        self._job_id = resp.job_id
        self.log(f"Submitted training job {self._job_id} ({len(self._manifest)} images in manifest)")

        # Persist immediately so cancel can reach the remote job.
        self._resume_data["job_id"] = self._job_id
        self._resume_data["model_name"] = self._model_name
        self._resume_data["manifest"] = self._manifest
        self._resume_data["config"] = self._config
        self._data_ready_flag.set()

        if self._cancel_flag.is_set():
            await self._cancel_remote_job(self._job_id)
            return {"cancelled": True, "job_id": self._job_id}

        # ── Step 2: Push missing images ──────────────────────────────────────
        await self._push_images(self._job_id, self._manifest)

        if self._cancel_flag.is_set():
            await self._cancel_remote_job(self._job_id)
            return {"cancelled": True, "job_id": self._job_id}

        # ── Step 3: Start the job ────────────────────────────────────────────
        await asyncio.to_thread(
            TrainingApi(self._api_client).start_training_job,
            self._job_id,
        )
        self.log("Job started, waiting for training to complete...")

        # ── Step 4: Relay SSE ────────────────────────────────────────────────
        await self.heartbeat(force=True)
        return await self._consume_sse(self._job_id)

    async def _check_remote_job_status(self, job_id: str) -> str:
        """Return the current status of a remote training job, or 'not_found' on any error."""
        try:
            resp = await asyncio.to_thread(
                TrainingApi(self._api_client).get_training_job,
                job_id,
            )
            return resp.status
        except Exception:
            return "not_found"

    async def _push_images(self, job_id: str, manifest: list[dict]) -> None:
        """Determine missing images, read them from FILES_ROOT, POST tar batches."""
        # Build uuid → filename map from manifest.
        uuid_to_filename = {e["image_uuid"]: e["filename"] for e in manifest}

        # Get missing UUIDs from the training server.
        missing_resp = await asyncio.to_thread(
            TrainingApi(self._api_client).get_missing_images,
            job_id,
        )
        missing_uuids = list(missing_resp.uuids or [])

        if not missing_uuids:
            self.log("All images already cached on training server")
            return

        self.log(f"Pushing {len(missing_uuids)} images to training server")

        # Process in batches of _TAR_BATCH_SIZE, reusing one HTTP client.
        pushed = 0
        async with httpx.AsyncClient(timeout=300.0) as client:
            for batch_start in range(0, len(missing_uuids), _TAR_BATCH_SIZE):
                if self._cancel_flag.is_set():
                    return
                batch = missing_uuids[batch_start : batch_start + _TAR_BATCH_SIZE]
                tar_bytes = await asyncio.to_thread(self._build_tar, batch, uuid_to_filename)
                if not tar_bytes:
                    continue
                # Use httpx directly — the generated client can't send raw binary body.
                r = await client.post(
                    f"{self._training_base_url}/api/training/jobs/{job_id}/images",
                    content=tar_bytes,
                    headers={"Content-Type": "application/x-tar"},
                )
                r.raise_for_status()
                pushed += len(batch)
                self._update_progress(min(pushed / len(missing_uuids) * 50.0, 50.0))
                await self.heartbeat()
                self.log(f"Pushed {pushed}/{len(missing_uuids)} images")

    def _build_tar(self, uuids: list[str], uuid_to_filename: dict[str, str]) -> bytes:
        """Build a tar in memory for the given UUIDs, reading bytes from FILES_ROOT."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:") as tf:
            for uuid in uuids:
                filename = uuid_to_filename.get(uuid, "")
                if not filename:
                    logger.warning("No filename for UUID %s, skipping", uuid)
                    continue
                img_path = FILES_ROOT / filename
                if not img_path.exists():
                    logger.warning("Image file not found: %s, skipping", img_path)
                    continue
                info = tarfile.TarInfo(name=uuid)
                info.size = img_path.stat().st_size
                with open(img_path, "rb") as fh:
                    tf.addfile(info, fh)
        return buf.getvalue()

    async def _consume_sse(self, job_id: str) -> dict[str, Any]:
        """Open the SSE stream for *job_id* and relay events until done/error/cancel."""
        stream_url = f"{self._training_base_url}/api/training/jobs/{job_id}/stream"
        cancelled = False

        if self._cancel_flag.is_set():
            cancelled = True
        else:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", stream_url) as response:
                    event_type: Optional[str] = None
                    async for raw_line in response.aiter_lines():
                        if self._cancel_flag.is_set():
                            cancelled = True
                            break

                        line = raw_line.strip()
                        if not line:
                            event_type = None
                            continue
                        if line.startswith("event:"):
                            event_type = line[len("event:") :].strip()
                        elif line.startswith("data:"):
                            data_str = line[len("data:") :].strip()
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            result = await self._handle_event(event_type, data)
                            if result is not None:
                                return result

        if cancelled:
            await self._cancel_remote_job(job_id)
            return {"cancelled": True, "job_id": job_id}

        raise RuntimeError(f"Training job {job_id} stream ended without a terminal event")

    async def _handle_event(self, event_type: Optional[str], data: dict) -> Optional[dict[str, Any]]:
        """Process one SSE event. Returns a result dict if terminal, else None."""
        if event_type == "progress":
            progress = float(data.get("progress", 0.0))
            # Offset by 50 to account for image-push phase (0–50%) + training phase (50–100%).
            self._update_progress(50.0 + progress * 50.0)
            log_msg = data.get("log", "")
            if log_msg:
                self.log(log_msg)
            await self.heartbeat()

        elif event_type == "status":
            status = data.get("status", "")
            if status:
                self.log(f"Status: {status}")

        elif event_type == "done":
            # Fetch and save the artifact.
            await self._save_artifact(self._job_id)
            result: dict[str, Any] = {"job_id": self._job_id, "model_name": self._model_name}
            val_accuracy = data.get("val_accuracy")
            if val_accuracy is not None:
                result["val_accuracy"] = val_accuracy
            self.log(
                f"Training complete — model: {self._model_name}, val_accuracy: {val_accuracy}"
            )
            return result

        elif event_type == "error":
            msg = data.get("message", "unknown error")
            raise RuntimeError(f"Training job failed: {msg}")

        return None

    async def _save_artifact(self, job_id: str) -> None:
        """Download the artifact tar from the training server and write to /models."""
        artifact_url = f"{self._training_base_url}/api/training/jobs/{job_id}/artifact"
        self.log("Downloading trained model artifact...")

        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.get(artifact_url)
            r.raise_for_status()
            tar_bytes = r.content

        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if not name.endswith((".onnx", ".json")):
                    logger.warning("Unexpected artifact file %s, skipping", name)
                    continue
                dest = _MODELS_DIR / name
                fobj = tf.extractfile(member)
                if fobj:
                    dest.write_bytes(fobj.read())
                    self.log(f"Saved model artifact: {dest}")

    async def _cancel_remote_job(self, job_id: str) -> None:
        """Best-effort DELETE of the remote training job."""
        try:
            await asyncio.to_thread(
                TrainingApi(self._api_client).cancel_training_job,
                job_id,
            )
            self.log(f"Cancelled remote training job {job_id}")
        except Exception as exc:
            logger.warning("Could not cancel remote job %s: %s", job_id, exc)

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
