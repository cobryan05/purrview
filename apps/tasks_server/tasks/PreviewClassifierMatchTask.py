"""PreviewClassifierMatchTask — "Run Model on Results" for classifiers: run an image
classifier over an explicit, client-enumerated set of images and propose match
candidates for confirmation.

Sibling of PreviewModelRelabelTask (detectors), but substantially simpler since
classification is whole-image only: no bbox candidate set, no IoU matching, no crop
mode, no new-box proposal. The client already knows the image scope (it enumerated it
from the review queue), so this task just fetches + classifies each image once and
returns a list of match candidates. This task never writes to an annotation table —
it only previews; the caller (review browser) confirms via a bulk-apply-tag flow.

Optional `preview_run_uuid`: when set, candidates are streamed into that persisted
preview run (one boxless candidate per matched image) and the run's stats are written
by the task itself, so the result outlives the tab that started it. The one-shot
result then carries `preview_run_uuid` and an empty `candidates` list.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np
from db_client.api.images_api import ImagesApi
from db_client.api.labels_api import LabelsApi
from db_client.api.preview_runs_api import PreviewRunsApi
from db_client.models.add_preview_candidates_payload import AddPreviewCandidatesPayload
from db_client.models.batch_by_uuids_payload import BatchByUuidsPayload
from db_client.models.preview_candidate_create import PreviewCandidateCreate
from db_client.models.preview_run_update import PreviewRunUpdate

from apps import FILES_ROOT
from apps.helpers.consts import TaskAccent
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.metadataUtils import validate_bare_model_name

from ._classifier_runtime import ClassifierRuntime, load_classifier_config
from ._image_loading import load_image_bytes
from .ApplyFilterChainTask import _CLASSIFIER_KINDS, _read_model_kind
from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

# Images fetched per batch_images_by_uuids round-trip (matches PreviewModelRelabelTask).
_IMAGE_BATCH_SIZE = 32

# Candidates buffered before being flushed into preview_candidates (matches
# PreviewModelRelabelTask).
_CANDIDATE_FLUSH_SIZE = 500


@register_task()
class PreviewClassifierMatchTask(Task):
    # A dry run: produces a result to look at, changes nothing.
    CARD_ACCENT = TaskAccent.TEAL

    # Started by Review's "Run Model on Results" when the chosen model is a
    # classifier. See PreviewModelRelabelTask.
    SYSTEM_TASK = True

    async def _init(self, params: Dict[str, Any], resume_data: Optional[Dict[str, Any]]) -> None:
        self._model_name: str = str(params.get("model_name") or "")
        if not self._model_name:
            raise ValueError("model_name is required")
        validate_bare_model_name(self._model_name)

        self._conf_thresh: float = float(params.get("conf_thresh", 0.5))
        self._image_source: str = str(params.get("image_source", "filesystem"))

        self._image_uuids: List[str] = list(params.get("image_uuids") or [])
        if not self._image_uuids:
            raise ValueError("image_uuids is required")

        self._preview_run_uuid: Optional[str] = params.get("preview_run_uuid") or None

        self._images_processed: int = resume_data.get("images_processed", 0) if resume_data else 0
        # Flushed candidates leave the buffer (and the checkpoint), so a resume
        # cannot re-insert them; `_candidates_written` carries their count.
        self._candidates: List[Dict[str, Any]] = list(resume_data.get("candidates", [])) if resume_data else []
        self._candidates_written: int = resume_data.get("candidates_written", 0) if resume_data else 0
        self._processed_image_uuids: Set[str] = (
            set(resume_data.get("processed_image_uuids", [])) if resume_data else set()
        )

        self._images_failed: int = resume_data.get("images_failed", 0) if resume_data else 0
        self._sample_errors: List[str] = list(resume_data.get("sample_errors", [])) if resume_data else []
        self._no_match_count: int = resume_data.get("no_match_count", 0) if resume_data else 0
        self._runtime: Optional[ClassifierRuntime] = None

    async def _run(self) -> Dict[str, Any]:
        db_api_client = self._manager.get_db_api_client()
        images_api = ImagesApi(db_api_client)
        labels_api = LabelsApi(db_api_client)

        model_kind = await asyncio.to_thread(_read_model_kind, self._model_name)
        if model_kind not in _CLASSIFIER_KINDS:
            raise ValueError(
                f"Model {self._model_name!r} has kind {model_kind!r}; PreviewClassifierMatchTask "
                "only supports classifier models (image_classifier / multi_label_classifier)"
            )

        label_names = await self._build_label_name_map(labels_api)

        # Load the model once and hold it for the whole task run (an implicit,
        # in-process pin), batching many images per session.run call — unlike
        # going through inference_server's /classify, which auto-pins/unpins
        # (a full model load+unload cycle) and infers one image at a time.
        self.log(f"Loading model {self._model_name!r}")
        model_path, model_config = await asyncio.to_thread(load_classifier_config, self._model_name)
        self._runtime = ClassifierRuntime(model_path, model_config)
        await self._runtime.load()
        self.log("Model loaded")

        remaining = [u for u in self._image_uuids if u not in self._processed_image_uuids]
        total = len(self._image_uuids)
        self.log(f"Classifying {total} image(s) with {self._model_name!r} (conf>={self._conf_thresh})")

        await self.heartbeat(force=True)

        for batch_start in range(0, len(remaining), _IMAGE_BATCH_SIZE):
            if self._cancel_flag.is_set():
                break
            await self._check_pause()

            batch = remaining[batch_start : batch_start + _IMAGE_BATCH_SIZE]
            try:
                resp = await asyncio.to_thread(images_api.batch_images_by_uuids, BatchByUuidsPayload(image_uuids=batch))
                images_data = resp.images or []
            except Exception as e:
                logger.exception("Failed to fetch image batch starting at index %d", batch_start)
                self._images_failed += len(batch)
                if len(self._sample_errors) < 5:
                    self._sample_errors.append(f"batch fetch at index {batch_start}: {e}")
                self._images_processed += len(batch)
                self._processed_image_uuids.update(batch)
                continue

            images_by_uuid = {img.get("uuid"): img for img in images_data}
            await self._classify_batch(images_api, self._runtime, batch, images_by_uuid, label_names)

            # Before the checkpoint below, so a resume never re-inserts what has
            # already landed in the run.
            try:
                await self._flush_candidates()
            except Exception as e:
                logger.exception("Failed to flush preview candidates; will retry on the next batch")
                if len(self._sample_errors) < 5:
                    self._sample_errors.append(f"preview candidate flush: {e}")

            progress = min(100.0, self._images_processed / max(total, 1) * 100.0)
            self._update_progress(progress)
            self.log(f"Processed {self._images_processed}/{total} image(s), {self._candidate_count()} candidate(s)")

            if self._data_req_flag.is_set():
                self._data_req_flag.clear()
                self._resume_data = self._build_resume_data()
                self._data_ready_flag.set()

            await self.heartbeat()
            await asyncio.sleep(0)

        await self._flush_candidates(force=True)
        await self._persist_run_stats()
        self._resume_data = self._build_resume_data()
        self._data_ready_flag.set()
        self.log(
            f"Complete: {self._candidate_count()} candidate(s) from {self._images_processed} image(s) "
            f"({self._images_failed} failed to load/process, {self._no_match_count} below threshold/unmapped)"
        )
        if self._sample_errors:
            self.log(f"Sample errors: {self._sample_errors}")
        return self._build_result()

    async def _build_label_name_map(self, labels_api: LabelsApi) -> Dict[str, str]:
        resp = await asyncio.to_thread(labels_api.list_labels)
        return {lbl.uuid: lbl.name for lbl in (resp.labels or [])}

    async def _classify_batch(
        self,
        images_api: ImagesApi,
        runtime: ClassifierRuntime,
        image_uuids: List[str],
        images_by_uuid: Dict[str, Any],
        label_names: Dict[str, str],
    ) -> None:
        """Load + preprocess each image in *image_uuids*, then run one batched
        inference call over all of them (instead of one HTTP round-trip per image)."""
        tensors: List[np.ndarray] = []
        valid_images: List[Dict[str, Any]] = []

        for image_uuid in image_uuids:
            if self._cancel_flag.is_set():
                break

            img = images_by_uuid.get(image_uuid)
            if img is None:
                self._images_failed += 1
                if len(self._sample_errors) < 5:
                    self._sample_errors.append(f"{image_uuid}: not found in batch_images_by_uuids response")
                self._images_processed += 1
                self._processed_image_uuids.add(image_uuid)
                continue

            filename = img.get("filename")
            if not filename:
                self._images_processed += 1
                self._processed_image_uuids.add(image_uuid)
                continue

            try:
                image_bytes = await self._load_image_bytes(filename, images_api)
                tensor = await asyncio.to_thread(runtime.preprocess, image_bytes)
            except Exception as e:
                logger.exception("Failed to load/preprocess image %s", image_uuid)
                self._images_failed += 1
                if len(self._sample_errors) < 5:
                    self._sample_errors.append(f"{image_uuid}: {e}")
                self._images_processed += 1
                self._processed_image_uuids.add(image_uuid)
                continue

            if tensor is None:
                self._images_failed += 1
                if len(self._sample_errors) < 5:
                    self._sample_errors.append(f"{image_uuid}: could not decode image")
                self._images_processed += 1
                self._processed_image_uuids.add(image_uuid)
                continue

            tensors.append(tensor)
            valid_images.append(img)

        if not tensors:
            return

        predictions = await runtime.predict_batch(tensors)

        for img, preds in zip(valid_images, predictions):
            image_uuid = img.get("uuid")
            filename = img.get("filename")
            best = max(preds, key=lambda p: p[1], default=None)
            if best is None or best[1] < self._conf_thresh or not best[0]:
                self._no_match_count += 1
            else:
                label_uuid, confidence = best
                self._candidates.append(
                    {
                        "imageUuid": image_uuid,
                        "imagePath": filename,
                        "predictedLabelUuid": label_uuid,
                        "predictedLabelName": label_names.get(label_uuid),
                        "confidence": confidence,
                    }
                )
            self._images_processed += 1
            self._processed_image_uuids.add(image_uuid)

    async def _load_image_bytes(self, filename: str, images_api: ImagesApi) -> bytes:
        return await load_image_bytes(filename, self._image_source, images_api, FILES_ROOT)

    async def _deinit(self) -> None:
        if self._runtime is not None:
            await self._runtime.close()

    def _candidate_count(self) -> int:
        return self._candidates_written + len(self._candidates)

    async def _flush_candidates(self, force: bool = False) -> None:
        """Write buffered candidates into the persisted preview run, if there is one.

        A failed flush keeps the buffer, so the next flush (or the final forced
        one) retries it rather than losing the work.
        """
        if not self._preview_run_uuid or not self._candidates:
            return
        if not force and len(self._candidates) < _CANDIDATE_FLUSH_SIZE:
            return

        batch = self._candidates
        payload = AddPreviewCandidatesPayload(
            candidates=[
                PreviewCandidateCreate(
                    image_uuid=c["imageUuid"],
                    proposed_label_uuid=c["predictedLabelUuid"],
                    confidence=float(c.get("confidence") or 0.0),
                    source_model=self._model_name,
                )
                for c in batch
            ],
        )
        api = PreviewRunsApi(self._manager.get_db_api_client())
        await asyncio.to_thread(api.add_preview_candidates, self._preview_run_uuid, payload)
        self._candidates_written += len(batch)
        self._candidates = []

    async def _persist_run_stats(self) -> None:
        """Written by the task rather than the browser: a PATCH sent after polling
        is lost exactly when the tab is."""
        if not self._preview_run_uuid:
            return
        stats = self._build_result()
        stats.pop("candidates", None)
        api = PreviewRunsApi(self._manager.get_db_api_client())
        try:
            await asyncio.to_thread(api.update_preview_run, self._preview_run_uuid, PreviewRunUpdate(stats=stats))
        except Exception:
            logger.exception("Failed to write stats for preview run %s", self._preview_run_uuid)

    def _build_resume_data(self) -> Dict[str, Any]:
        return {
            "images_processed": self._images_processed,
            "candidates": self._candidates,
            "candidates_written": self._candidates_written,
            "processed_image_uuids": sorted(self._processed_image_uuids),
            "images_failed": self._images_failed,
            "sample_errors": self._sample_errors,
            "no_match_count": self._no_match_count,
        }

    def _build_result(self) -> Dict[str, Any]:
        result = {
            "images_processed": self._images_processed,
            "candidate_count": self._candidate_count(),
            "candidates": self._candidates,
            "images_failed": self._images_failed,
            "sample_errors": self._sample_errors,
            "no_match_count": self._no_match_count,
        }
        if self._preview_run_uuid:
            result["preview_run_uuid"] = self._preview_run_uuid
            result["candidates"] = []
        return result

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "meta": {
                "order": ["model_name", "conf_thresh", "image_uuids", "preview_run_uuid"],
            },
            "model_name": {
                "type": "model_name",
                "required": True,
                "help": "Classifier model (image_classifier or multi_label_classifier) to run against the images.",
            },
            "conf_thresh": {
                "type": "float",
                "label": "Confidence Threshold",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            },
            "image_uuids": {
                "type": "list",
                "label": "Image UUIDs",
                "required": True,
                "description": "Images to fetch and classify (each fetched/inferred once).",
            },
            "preview_run_uuid": {
                "type": "str",
                "label": "Preview Run UUID",
                "required": False,
                "description": (
                    "Persist candidates and stats into this preview run as they are found, instead of "
                    "returning them all in the one-shot result."
                ),
            },
        }
