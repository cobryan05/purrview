"""Shared submit/push/SSE/save routine for a single remote training job.

Both TrainModelTask (one job per Task run) and TrainBundleTask (N jobs per Task
run, one per bundle target) drive their remote training-server job through
``TrainingJobRunnerMixin``. The host Task class must set up, before calling
``_run_one_job()``:
  - self._api_client, self._training_base_url — training_client wiring
  - self._cancel_flag (asyncio.Event), self._resume_data (dict)
  - self._data_ready_flag (asyncio.Event)
  - self.log(msg), self.heartbeat(force=False), self._update_progress(pct)
    — provided by the Task base class
  - self._job_id / self._manifest / self._model_name / self._config
    — mutable instance attributes describing the CURRENT job; _run_one_job
      reads and updates them (including on resume) for exactly one job.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import tarfile
from pathlib import Path
from typing import Any, Optional

import httpx
from training_client.api.training_api import TrainingApi
from training_client.models.manifest_entry import ManifestEntry
from training_client.models.submit_job_payload import SubmitJobPayload

from apps import FILES_ROOT

logger = logging.getLogger(__file__)

_MODELS_DIR = Path(os.environ.get("MODELS_ROOT", "/models"))
_TAR_BATCH_SIZE = 500  # max images per tar POST


class TrainingJobRunnerMixin:
    async def _run_one_job(self) -> dict[str, Any]:
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
                    r = await client.post(f"{self._training_base_url}/api/training/jobs/{self._job_id}/restart")
                    r.raise_for_status()
                return await self._consume_sse(self._job_id)

            if status in ("running", "queued", "awaiting_data"):
                # Already in progress — just reconnect to the stream.
                return await self._consume_sse(self._job_id)

            if status in ("error", "cancelled"):
                raise RuntimeError(f"Training job {self._job_id} is in terminal state: {status}")

            # Status is "not_found" (404) or unknown — server restarted without
            # persisted state. Re-submit the full job; images are likely still cached.
            self.log(f"Job {self._job_id} not found on training server; resubmitting from scratch")
            self._job_id = None

        # ── Step 1: Submit job ───────────────────────────────────────────────
        payload = SubmitJobPayload(
            trainer_type=self._config.get("trainer_type", "binary_image_classifier"),
            label_uuid=self._config.get("label_uuid", ""),
            model_name=self._model_name,
            architecture=self._config.get("architecture", ""),
            class_names=self._config.get("class_names") or ["absent", "present"],
            # Everything not a typed SubmitJobPayload field rides in
            # hyperparams — including bundle-target identity keys (role,
            # model_config_uuid, model_config_node_uuid, label_uuids) that the
            # training server merges back into the trainer config
            # (web.py: config.update(payload.hyperparams)). That merge is a
            # load-bearing contract, pinned by
            # tests/training_server/test_training.py::test_submit_job_endpoint_merges_hyperparams_into_trainer_config.
            hyperparams={
                k: v
                for k, v in self._config.items()
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
        skipped = 0
        async with httpx.AsyncClient(timeout=300.0) as client:
            for batch_start in range(0, len(missing_uuids), _TAR_BATCH_SIZE):
                if self._cancel_flag.is_set():
                    return
                batch = missing_uuids[batch_start : batch_start + _TAR_BATCH_SIZE]
                tar_bytes, added, batch_skipped = await asyncio.to_thread(self._build_tar, batch, uuid_to_filename)
                skipped += batch_skipped
                if not added:
                    continue
                # Use httpx directly — the generated client can't send raw binary body.
                r = await client.post(
                    f"{self._training_base_url}/api/training/jobs/{job_id}/images",
                    content=tar_bytes,
                    headers={"Content-Type": "application/x-tar"},
                )
                r.raise_for_status()
                ingest_result = r.json()
                server_received = ingest_result.get("received", 0)
                if server_received < added:
                    raise RuntimeError(
                        f"Training server only cached {server_received}/{added} images in batch. "
                        f"Check training server logs for write errors (permissions, disk space)."
                    )
                pushed += added
                self._update_progress(min(pushed / len(missing_uuids) * 50.0, 50.0))
                await self.heartbeat()
                self.log(f"Pushed {pushed}/{len(missing_uuids)} images")

        if skipped:
            raise RuntimeError(
                f"{skipped} image file(s) not found on disk and could not be pushed. "
                f"Check that FILES_ROOT ({FILES_ROOT}) contains the labeled images."
            )

    def _build_tar(self, uuids: list[str], uuid_to_filename: dict[str, str]) -> tuple[bytes, int, int]:
        """Build a tar in memory for the given UUIDs, reading bytes from FILES_ROOT.

        Returns (tar_bytes, added_count, skipped_count).
        """
        buf = io.BytesIO()
        added = 0
        skipped = 0
        with tarfile.open(fileobj=buf, mode="w:") as tf:
            for uuid in uuids:
                filename = uuid_to_filename.get(uuid, "")
                if not filename:
                    self.log(f"WARNING: no filename recorded for image {uuid}, skipping")
                    skipped += 1
                    continue
                img_path = FILES_ROOT / filename
                if not img_path.exists():
                    self.log(f"WARNING: image file not found: {img_path}, skipping")
                    skipped += 1
                    continue
                info = tarfile.TarInfo(name=uuid)
                info.size = img_path.stat().st_size
                with open(img_path, "rb") as fh:
                    tf.addfile(info, fh)
                added += 1
        return buf.getvalue(), added, skipped

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
            self.log(f"Training complete — model: {self._model_name}, val_accuracy: {val_accuracy}")
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
                if not name.endswith((".onnx", ".onnx.data", ".json")):
                    logger.warning("Unexpected artifact file %s, skipping", name)
                    continue
                dest = _MODELS_DIR / name
                fobj = tf.extractfile(member)
                if fobj:
                    if dest.exists():
                        # Expected on retrain of the same model name; anything
                        # else (e.g. two bundles sharing a name) is a silent
                        # clobber — leave a trace either way.
                        logger.warning("Overwriting existing model artifact %s", dest)
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
