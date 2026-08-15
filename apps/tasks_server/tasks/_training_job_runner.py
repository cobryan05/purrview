"""Shared submit/push/SSE/save routine for a single remote training job.

TrainModelTask (one job per Task run) drives its remote training-server job
through ``TrainingJobRunnerMixin``. The mixin also served the since-removed
TrainBundleTask, which ran N jobs per Task run — hence the shape.
The host Task class must set up, before calling
``_run_one_job()``:
  - self._api_client, self._training_base_url — training_client wiring
  - self._cancel_flag (asyncio.Event), self._resume_data (dict)
  - self._data_ready_flag (asyncio.Event)
  - self.log(msg), self.heartbeat(force=False), self._update_progress(pct)
    — provided by the Task base class
  - self._job_id / self._manifest / self._model_name / self._config
    — mutable instance attributes describing the CURRENT job; _run_one_job
      reads and updates them (including on resume) for exactly one job.
  - self._epoch_history — list of per-epoch metric dicts, seeded from
      resume_data; appended to by _record_epoch_metrics as progress events
      arrive, and read by the host task's get_status_data() to draw the
      training curves.
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
from training_client.models.local_ingest_payload import LocalIngestPayload
from training_client.models.local_manifest_entry import LocalManifestEntry
from training_client.models.manifest_entry import ManifestEntry
from training_client.models.submit_job_payload import SubmitJobPayload

from apps import FILES_ROOT

logger = logging.getLogger(__file__)

_MODELS_DIR = Path(os.environ.get("MODELS_ROOT", "/models"))
_TAR_BATCH_SIZE = 500  # max images per tar POST

# Hard cap on retained per-epoch metric samples. Real runs are tens-to-hundreds of
# epochs, but the history is persisted into resume_data (and snapshotted into the
# task instance row on completion), so a runaway trainer must not grow it without
# bound.
_MAX_EPOCH_HISTORY = 2000

# Metric keys every trainer's progress event carries alongside `epoch`. Kept as a
# tuple rather than "whatever's in the event" so a trainer adding a private
# debugging key doesn't silently start riding along in the persisted history.
_EPOCH_METRIC_KEYS = ("train_loss", "val_loss", "val_accuracy")


class _StreamDropped(Exception):
    """The SSE connection ended without a terminal event.

    Says nothing about the job — only that this end stopped hearing about it.
    """


class TrainingJobRunnerMixin:
    # How many times to reconnect to a job's event stream (or wait out an
    # unreachable server on resume) before giving up on it. With _RECONNECT_DELAYS
    # this is a bit over 20 minutes of outage tolerated, which covers a training
    # server restart or a redeploy on another host without leaving a task retrying
    # into a server that is never coming back.
    MAX_STREAM_RECONNECTS: int = 40
    _RECONNECT_DELAYS: tuple[int, ...] = (2, 5, 10, 20, 30)

    def _reconnect_delay(self, attempt: int) -> int:
        return self._RECONNECT_DELAYS[min(attempt, len(self._RECONNECT_DELAYS) - 1)]

    async def _run_one_job(self) -> dict[str, Any]:
        if self._job_id:
            status = await self._await_reachable_status(self._job_id)
            self.log(f"Resuming training job {self._job_id} (remote status: {status})")
            await self.heartbeat(force=True)

            if status == "done":
                await self._save_artifact(self._job_id)
                return {"job_id": self._job_id, "model_name": self._model_name}

            if status == "interrupted":
                # Server restarted mid-job. Restart the job and consume SSE.
                # Training will resume from the last epoch checkpoint.
                self.log(f"Job {self._job_id} was interrupted; restarting on training server")
                await self._restart_remote_job(self._job_id)
                return await self._consume_sse(self._job_id)

            if status in ("running", "queued", "awaiting_data"):
                # Already in progress — just reconnect to the stream.
                return await self._consume_sse(self._job_id)

            if status in ("error", "cancelled"):
                raise RuntimeError(f"Training job {self._job_id} is in terminal state: {status}")

            if status == "unreachable":
                # Only reachable by cancelling during the retry wait — the poll
                # otherwise raises rather than returning this. Falling through
                # would resubmit a job we never established was gone.
                return {"cancelled": True, "job_id": self._job_id}

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
            # hyperparams — including training-target identity keys (role,
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

        # Persist immediately so cancel — and recovery after a restart — can reach
        # the remote job. Written straight through rather than flagged for the
        # manager's periodic pass: that pass runs once a minute, and a restart
        # inside that window leaves a real training job running on the other host
        # with nothing in the database pointing at it. The flag is still set as a
        # fallback for when the direct write fails (db_server restarting).
        self._resume_data["job_id"] = self._job_id
        self._resume_data["model_name"] = self._model_name
        self._resume_data["manifest"] = self._manifest
        self._resume_data["config"] = self._config
        await self.save_checkpoint_now()

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
        """Return the current status of a remote training job.

        Two distinct failure answers, deliberately not merged: 'not_found' means
        the training server answered and has no such job (resubmit is correct),
        'unreachable' means it did not answer at all (resubmitting would push the
        whole image set again and leave the original job training unattended).
        The training server can be on another host, so "did not answer" is an
        ordinary, recoverable event.
        """
        try:
            resp = await asyncio.to_thread(
                TrainingApi(self._api_client).get_training_job,
                job_id,
            )
            return resp.status
        except Exception as e:
            if getattr(e, "status", None) == 404:
                return "not_found"
            logger.warning("Could not reach the training server for job %s: %s", job_id, e)
            return "unreachable"

    async def _await_reachable_status(self, job_id: str) -> str:
        """Poll the remote job status until the training server answers.

        Bounded by MAX_STREAM_RECONNECTS: a server that is never coming back must
        fail the task rather than hold it open forever.
        """
        for attempt in range(self.MAX_STREAM_RECONNECTS):
            status = await self._check_remote_job_status(job_id)
            if status != "unreachable":
                return status
            if self._cancel_flag.is_set():
                return "unreachable"
            delay = self._reconnect_delay(attempt)
            self.log(f"Training server unreachable; retrying job {job_id} in {delay}s")
            await self.heartbeat(force=True)
            await asyncio.sleep(delay)
        raise RuntimeError(f"Training server unreachable while trying to re-attach to job {job_id}")

    async def _restart_remote_job(self, job_id: str) -> None:
        """Re-queue a job the training server marked 'interrupted' by its own restart."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{self._training_base_url}/api/training/jobs/{job_id}/restart")
            r.raise_for_status()

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
        local_linked = 0
        local_copied = 0
        async with httpx.AsyncClient(timeout=300.0) as client:
            for batch_start in range(0, len(missing_uuids), _TAR_BATCH_SIZE):
                if self._cancel_flag.is_set():
                    return
                batch = missing_uuids[batch_start : batch_start + _TAR_BATCH_SIZE]

                filenameless = [u for u in batch if u not in uuid_to_filename]
                for u in filenameless:
                    self.log(f"WARNING: no filename recorded for image {u}, skipping")
                skipped += len(filenameless)
                batch = [u for u in batch if u in uuid_to_filename]
                if not batch:
                    continue

                # First try zero-copy local ingest: the training server reads
                # straight from FILES_ROOT itself, no bytes cross HTTP at all.
                # This only succeeds for UUIDs it can actually see on its own
                # filesystem (e.g. pusher and training server share a
                # container/bind mount) — anything else comes back in
                # not_found and falls through to the tar-upload path below.
                local_resp = await asyncio.to_thread(
                    TrainingApi(self._api_client).ingest_local_images,
                    job_id,
                    LocalIngestPayload(
                        entries=[LocalManifestEntry(image_uuid=u, filename=uuid_to_filename[u]) for u in batch]
                    ),
                )
                not_found = set(local_resp.not_found or [])
                local_linked += local_resp.linked
                local_copied += local_resp.copied
                pushed += len(batch) - len(not_found)

                remaining = [u for u in batch if u in not_found]
                if remaining:
                    tar_bytes, added, batch_skipped = await asyncio.to_thread(
                        self._build_tar, remaining, uuid_to_filename
                    )
                    skipped += batch_skipped
                    if added:
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

        if local_linked or local_copied:
            self.log(f"Zero-copy local ingest: {local_linked} hardlinked, {local_copied} copied on-disk")

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
        """Follow *job_id*'s event stream to a terminal event, reconnecting as needed.

        Losing the stream is not losing the job: the training runs on the training
        server, which may be on another host, so a dropped connection, a network
        blip or a redeploy over there must be recovered from rather than reported
        as a training failure. On each drop the job's real status decides what
        happens — finished while we were away (collect the artifact), interrupted
        by the server's own restart (re-queue it), still going (reconnect), or
        genuinely terminal (fail).
        """
        for attempt in range(self.MAX_STREAM_RECONNECTS + 1):
            if self._cancel_flag.is_set():
                await self._cancel_remote_job(job_id)
                return {"cancelled": True, "job_id": job_id}

            try:
                return await self._stream_job_events(job_id)
            except (httpx.HTTPError, _StreamDropped) as e:
                if attempt >= self.MAX_STREAM_RECONNECTS:
                    raise RuntimeError(f"Lost the event stream for training job {job_id}: {e}") from e
                self.log(f"Event stream for job {job_id} dropped ({e}); checking whether it is still running")

            if self._cancel_flag.is_set():
                await self._cancel_remote_job(job_id)
                return {"cancelled": True, "job_id": job_id}

            status = await self._check_remote_job_status(job_id)
            if status == "done":
                # It finished while we were disconnected — the terminal event was
                # emitted to nobody, but the artifact is there to collect.
                await self._save_artifact(job_id)
                self.log(f"Training job {job_id} completed while disconnected")
                return {"job_id": job_id, "model_name": self._model_name}
            if status in ("error", "cancelled"):
                raise RuntimeError(f"Training job {job_id} ended in state: {status}")
            if status == "not_found":
                raise RuntimeError(f"Training job {job_id} no longer exists on the training server")
            if status == "interrupted":
                self.log(f"Job {job_id} was interrupted by a training server restart; re-queueing it")
                try:
                    await self._restart_remote_job(job_id)
                except Exception as e:
                    self.log(f"Could not re-queue job {job_id}: {e}")

            delay = self._reconnect_delay(attempt)
            self.log(f"Reconnecting to job {job_id} in {delay}s (attempt {attempt + 1})")
            # The reconnect wait is dead air as far as the stale-heartbeat check is
            # concerned; without this a long outage gets the task marked dead.
            await self.heartbeat(force=True)
            await asyncio.sleep(delay)

        raise RuntimeError(f"Gave up reconnecting to training job {job_id}")

    async def _stream_job_events(self, job_id: str) -> dict[str, Any]:
        """One SSE connection: relay events until a terminal one, or raise.

        Raises _StreamDropped if the connection ends without a terminal event —
        the caller decides whether that is recoverable.
        """
        stream_url = f"{self._training_base_url}/api/training/jobs/{job_id}/stream"

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", stream_url) as response:
                event_type: Optional[str] = None
                async for raw_line in response.aiter_lines():
                    if self._cancel_flag.is_set():
                        await self._cancel_remote_job(job_id)
                        return {"cancelled": True, "job_id": job_id}

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

        raise _StreamDropped(f"stream for job {job_id} ended without a terminal event")

    def _record_epoch_metrics(self, data: dict) -> None:
        """Append one per-epoch metric sample from a progress event to the history.

        Not every progress event is an epoch boundary — the trainers also emit
        phase messages ("Loading dataset...") with no `epoch` — so an event
        without a numeric epoch is simply ignored.

        Re-runs are the interesting case: a job that is interrupted and restarted
        resumes from its last checkpoint and replays epoch numbers we already have.
        Truncating everything at or after the incoming epoch (rather than
        appending) keeps the series monotonic in x, so the chart shows one curve
        per run instead of a sawtooth.
        """
        epoch = data.get("epoch")
        if not isinstance(epoch, (int, float)) or isinstance(epoch, bool):
            return
        epoch = int(epoch)

        sample: dict[str, Any] = {"epoch": epoch}
        for key in _EPOCH_METRIC_KEYS:
            value = data.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            # A hard 0.0 val_loss means "this trainer doesn't report it" — the
            # detection trainers send it as a placeholder. Charting that as a flat
            # zero line would invent a metric, so drop it and let the series be
            # absent (same rule the training server's own status view applies).
            if key == "val_loss" and value == 0.0:
                continue
            sample[key] = float(value)

        history: list[dict[str, Any]] = self._epoch_history
        while history and history[-1]["epoch"] >= epoch:
            history.pop()
        history.append(sample)
        del history[:-_MAX_EPOCH_HISTORY]

        total_epochs = data.get("total_epochs")
        if isinstance(total_epochs, (int, float)) and not isinstance(total_epochs, bool):
            self._total_epochs = int(total_epochs)

        # Ride along with the existing job_id/manifest persistence so an
        # interrupted-and-resumed run keeps the curve it has drawn so far.
        self._resume_data["epoch_history"] = history

    async def _handle_event(self, event_type: Optional[str], data: dict) -> Optional[dict[str, Any]]:
        """Process one SSE event. Returns a result dict if terminal, else None."""
        if event_type == "progress":
            progress = float(data.get("progress", 0.0))
            # Offset by 50 to account for image-push phase (0–50%) + training phase (50–100%).
            self._update_progress(50.0 + progress * 50.0)
            self._record_epoch_metrics(data)
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
                        # else (e.g. two configs sharing a model name) is a silent
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
