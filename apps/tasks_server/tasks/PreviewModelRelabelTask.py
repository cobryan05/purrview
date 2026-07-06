"""PreviewModelRelabelTask — "Run Model on Results": run a detector over an
explicit, client-enumerated set of existing bboxes and propose relabels for
confirmation.

Hybrid architecture: the client (review browser) enumerates the candidate bbox
UUIDs from whatever filter it currently has applied — and, because it already
has them in hand, the image UUID each one belongs to — and hands both lists to
this server-side background Task. The task fetches + infers each image exactly
once, runs the given detector model whole-image, matches detections against the
candidate bboxes on that image by IoU (same semantics as
`_filter_chain_runner.py`'s detector-overlap steps — existing bboxes and
detections share the same relative coordinate space), and returns a list of
candidate relabels. This task never writes to the DB — it only previews.

Two crop modes (`crop_mode` param):
  - "image" (default): run the detector once per image, whole-image, then
    match each detection to candidate bboxes on that image by IoU.
  - "bbox": crop each candidate bbox out of its image and run the detector on
    just that crop, taking the crop's single highest-confidence detection as
    the proposed label directly (no IoU matching — the crop *is* the object).
    Costs one inference call per candidate bbox instead of per image.

Detections whose class isn't mapped to a label UUID (`det.label_uuid is None`)
fall back to a name match against the project's label names (mirrors the
manual single-image "Run Model" UI, which already tolerates unmapped classes
by displaying the raw class string) before being discarded.
"""

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from db_client.api.images_api import ImagesApi
from db_client.api.labels_api import LabelsApi
from db_client.models.batch_by_uuids_payload import BatchByUuidsPayload
from inference_client.api.models_api import ModelsApi as InferenceModelsApi
from inference_client.models.pin_model_payload import PinModelPayload

from apps import FILES_ROOT
from apps.helpers.bboxUtils import iou_xywh
from apps.helpers.fileUtils import get_safe_path

from ._filter_chain_runner import Detection
from .ApplyFilterChainTask import _CLASSIFIER_KINDS, _HttpInferenceBackend, _read_model_kind, _validate_model_name
from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

# Hard refusal ceiling on the candidate bbox list. The 1000-item *warning*
# threshold (nudge the user to narrow their filter first) is a client-side
# concern, handled in the review-browser UI, not here.
_MAX_BBOX_UUIDS = 5000

# Images fetched per batch_images_by_uuids round-trip.
_IMAGE_BATCH_SIZE = 32


@register_task()
class PreviewModelRelabelTask(Task):
    async def _init(self, params: Dict[str, Any], resume_data: Optional[Dict[str, Any]]) -> None:
        self._model_name: str = str(params.get("model_name") or "")
        if not self._model_name:
            raise ValueError("model_name is required")
        _validate_model_name(self._model_name)

        self._conf_thresh: float = float(params.get("conf_thresh", 0.25))
        self._iou_thresh: float = float(params.get("iou_thresh", 0.5))
        self._image_source: str = str(params.get("image_source", "filesystem"))

        self._crop_mode: str = str(params.get("crop_mode") or "image")
        if self._crop_mode not in ("image", "bbox"):
            raise ValueError(f"crop_mode must be 'image' or 'bbox', got {self._crop_mode!r}")

        self._image_uuids: List[str] = list(params.get("image_uuids") or [])
        if not self._image_uuids:
            raise ValueError("image_uuids is required")

        # Candidate bboxes to preview relabels for — a subset of the bboxes on the
        # given images. Held as a set for O(1) membership tests in _process_image.
        bbox_uuids: List[str] = list(params.get("bbox_uuids") or [])
        if not bbox_uuids:
            raise ValueError("bbox_uuids is required")
        if len(bbox_uuids) > _MAX_BBOX_UUIDS:
            raise ValueError(
                f"bbox_uuids has {len(bbox_uuids)} entries, exceeding the hard cap of "
                f"{_MAX_BBOX_UUIDS}; narrow the filter before running the model"
            )
        self._bbox_uuids: Set[str] = set(bbox_uuids)

        self._images_processed: int = resume_data.get("images_processed", 0) if resume_data else 0
        self._candidates: List[Dict[str, Any]] = list(resume_data.get("candidates", [])) if resume_data else []
        self._processed_image_uuids: set = set(resume_data.get("processed_image_uuids", [])) if resume_data else set()

        # Diagnostics so a run that finds zero candidates says *why* (image load
        # failures, no matching detection, or the model just agreeing with every
        # existing label) instead of silently completing empty.
        self._images_failed: int = resume_data.get("images_failed", 0) if resume_data else 0
        self._sample_errors: List[str] = list(resume_data.get("sample_errors", [])) if resume_data else []
        self._no_detection_count: int = resume_data.get("no_detection_count", 0) if resume_data else 0
        self._no_op_count: int = resume_data.get("no_op_count", 0) if resume_data else 0

        self._pin_id: Optional[str] = None
        # Populated in _run() once label names are fetched; defaults to empty so
        # _process_image can be exercised directly in tests without a full _run().
        self._name_to_label_uuid: Dict[str, str] = {}

    async def _run(self) -> Dict[str, Any]:
        db_api_client = self._manager.get_db_api_client()
        inference_api_client = self._manager.get_inference_api_client()
        images_api = ImagesApi(db_api_client)
        labels_api = LabelsApi(db_api_client)

        model_kind = await asyncio.to_thread(_read_model_kind, self._model_name)
        if model_kind in _CLASSIFIER_KINDS:
            self.log(
                f"Model {self._model_name!r} is a {model_kind}; relabel preview only supports "
                "detector models — nothing to do"
            )
            return self._build_result()

        label_names = await self._build_label_name_map(labels_api)
        # Reverse (name -> uuid) lookup for the unmapped-class fallback: first
        # label with a given name wins, matching how the manual single-image
        # "Run Model" UI resolves a raw class string to a label.
        name_to_label_uuid: Dict[str, str] = {}
        for uuid, name in label_names.items():
            name_to_label_uuid.setdefault(name, uuid)
        self._name_to_label_uuid = name_to_label_uuid

        await self._pin_model(inference_api_client)
        pin_ids = {self._model_name: self._pin_id} if self._pin_id else {}
        backend = _HttpInferenceBackend(inference_api_client, pin_ids)

        # The client already knows each candidate bbox's image (it enumerated both
        # from the review queue), so no bbox->image resolution is needed here — we
        # fetch the given images directly and _process_image filters each image's
        # bboxes to the candidate set.
        remaining = [u for u in self._image_uuids if u not in self._processed_image_uuids]
        total = len(self._image_uuids)
        self.log(
            f"Previewing {len(self._bbox_uuids)} candidate bbox(es) across {total} image(s); "
            f"running {self._model_name!r} (conf>={self._conf_thresh}, iou>={self._iou_thresh})"
        )

        await self.heartbeat(force=True)

        for batch_start in range(0, len(remaining), _IMAGE_BATCH_SIZE):
            if self._cancel_flag.is_set():
                break
            await self._check_pause()

            batch = remaining[batch_start : batch_start + _IMAGE_BATCH_SIZE]
            try:
                resp = await asyncio.to_thread(
                    images_api.batch_images_by_uuids, BatchByUuidsPayload(image_uuids=batch)
                )
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
            for image_uuid in batch:
                if self._cancel_flag.is_set():
                    break
                img = images_by_uuid.get(image_uuid)
                if img is not None:
                    try:
                        await self._process_image(images_api, backend, img, label_names)
                    except Exception as e:
                        logger.exception("Failed to process image %s", image_uuid)
                        self._images_failed += 1
                        if len(self._sample_errors) < 5:
                            self._sample_errors.append(f"{image_uuid}: {e}")
                else:
                    self._images_failed += 1
                    if len(self._sample_errors) < 5:
                        self._sample_errors.append(f"{image_uuid}: not found in batch_images_by_uuids response")
                self._images_processed += 1
                self._processed_image_uuids.add(image_uuid)

            progress = min(100.0, self._images_processed / max(total, 1) * 100.0)
            self._update_progress(progress)
            self.log(f"Processed {self._images_processed}/{total} image(s), {len(self._candidates)} candidate(s)")

            if self._data_req_flag.is_set():
                self._data_req_flag.clear()
                self._resume_data = self._build_resume_data()
                self._data_ready_flag.set()

            await self.heartbeat()
            await asyncio.sleep(0)

        self._resume_data = self._build_resume_data()
        self._data_ready_flag.set()
        self.log(
            f"Complete: {len(self._candidates)} candidate relabel(s) from {self._images_processed} image(s) "
            f"({self._images_failed} failed to load/process, {self._no_detection_count} bbox(es) unmatched, "
            f"{self._no_op_count} bbox(es) already correctly labeled)"
        )
        if self._sample_errors:
            self.log(f"Sample errors: {self._sample_errors}")
        return self._build_result()

    async def _build_label_name_map(self, labels_api: LabelsApi) -> Dict[str, str]:
        resp = await asyncio.to_thread(labels_api.list_labels)
        return {lbl.uuid: lbl.name for lbl in (resp.labels or [])}

    async def _pin_model(self, inference_api_client) -> None:
        models_api = InferenceModelsApi(inference_api_client)
        try:
            pin_resp = await asyncio.to_thread(
                models_api.pin_model_api_api_models_model_name_pin_post,
                self._model_name,
                PinModelPayload(model_name=self._model_name, duration=3600),
            )
            if pin_resp.status == "success" and pin_resp.pin_id:
                self._pin_id = pin_resp.pin_id
        except Exception:
            logger.exception("Failed to pin model %s", self._model_name)

    def _resolve_label_uuid(self, det: Detection) -> Optional[str]:
        """A detection's class may not be mapped to a label UUID on this model
        (`det.label_uuid is None`) — fall back to matching the raw class string
        against a project label of the same name, same as the manual
        single-image "Run Model" UI already tolerates."""
        if det.label_uuid is not None:
            return det.label_uuid
        if det.class_str is None:
            return None
        return self._name_to_label_uuid.get(det.class_str)

    def _maybe_add_candidate(
        self,
        bbox: Dict[str, Any],
        image_uuid: str,
        filename: str,
        label_names: Dict[str, str],
        best: Optional[Detection],
        best_label_uuid: Optional[str],
    ) -> None:
        if best is None or best_label_uuid is None:
            self._no_detection_count += 1
            return
        current_label_uuid = bbox.get("labelUuid")
        if best_label_uuid == current_label_uuid:
            self._no_op_count += 1
            return  # no-op relabel — model agrees with the existing label
        self._candidates.append(
            {
                "bboxUuid": bbox.get("uuid"),
                "imageUuid": image_uuid,
                "imagePath": filename,
                "currentLabelUuid": current_label_uuid,
                "currentLabelName": label_names.get(current_label_uuid),
                "proposedLabelUuid": best_label_uuid,
                "proposedLabelName": label_names.get(best_label_uuid),
                "confidence": best.confidence,
            }
        )

    async def _process_image(
        self,
        images_api: ImagesApi,
        backend: _HttpInferenceBackend,
        img: Dict[str, Any],
        label_names: Dict[str, str],
    ) -> None:
        filename = img.get("filename")
        if not filename:
            return
        # Only infer if this image actually carries a candidate bbox — the client
        # may pass images whose only candidate got filtered out, and inference is
        # the expensive step.
        image_bboxes = [b for b in (img.get("bboxes") or []) if b.get("uuid") in self._bbox_uuids]
        if not image_bboxes:
            return
        image_bytes = await self._load_image_bytes(filename, images_api)

        if self._crop_mode == "bbox":
            await self._process_image_bbox_crop(image_bboxes, image_bytes, img, backend, label_names)
            return

        detections = await backend.detect(self._model_name, image_bytes, self._conf_thresh)
        if not detections:
            self._no_detection_count += len(image_bboxes)
            return

        # batch_images_by_uuids serializes with by_alias=True (alias_generator=to_camel),
        # so nested bbox dicts use camelCase keys ("labelUuid", not "label_uuid") — see
        # the same trap documented/fixed in DeduplicateOverlappingBBoxesTask.
        image_uuid = img.get("uuid")
        for bbox in image_bboxes:
            best: Optional[Detection] = None
            best_label_uuid: Optional[str] = None
            for det in detections:
                label_uuid = self._resolve_label_uuid(det)
                if label_uuid is None:
                    continue
                if iou_xywh(det, bbox) < self._iou_thresh:
                    continue
                if best is None or det.confidence > best.confidence:
                    best = det
                    best_label_uuid = label_uuid
            self._maybe_add_candidate(bbox, image_uuid, filename, label_names, best, best_label_uuid)

    async def _process_image_bbox_crop(
        self,
        image_bboxes: List[Dict[str, Any]],
        image_bytes: bytes,
        img: Dict[str, Any],
        backend: _HttpInferenceBackend,
        label_names: Dict[str, str],
    ) -> None:
        """Crop each candidate bbox out of the image and run inference on just
        that crop, taking its single highest-confidence detection as the
        proposed label — no IoU matching, since the crop *is* the candidate."""
        from PIL import Image as PILImage

        filename = img.get("filename")
        image_uuid = img.get("uuid")
        try:
            pil_image = await asyncio.to_thread(lambda: PILImage.open(io.BytesIO(image_bytes)).convert("RGB"))
        except Exception as e:
            logger.exception("Failed to open image %s for bbox-crop inference", filename)
            self._images_failed += 1
            if len(self._sample_errors) < 5:
                self._sample_errors.append(f"{filename}: {e}")
            return
        img_w, img_h = pil_image.size

        for bbox in image_bboxes:
            x, y = bbox.get("x", 0.0), bbox.get("y", 0.0)
            width, height = bbox.get("width", 0.0), bbox.get("height", 0.0)
            left = max(0, int(x * img_w))
            top = max(0, int(y * img_h))
            right = min(img_w, int((x + width) * img_w))
            bottom = min(img_h, int((y + height) * img_h))
            if right <= left or bottom <= top:
                self._no_detection_count += 1
                continue

            try:
                crop_bytes = await asyncio.to_thread(self._crop_to_png_bytes, pil_image, (left, top, right, bottom))
            except Exception as e:
                logger.exception("Failed to crop bbox %s on image %s", bbox.get("uuid"), filename)
                if len(self._sample_errors) < 5:
                    self._sample_errors.append(f"{filename} bbox {bbox.get('uuid')}: {e}")
                self._no_detection_count += 1
                continue

            detections = await backend.detect(self._model_name, crop_bytes, self._conf_thresh)
            best: Optional[Detection] = None
            best_label_uuid: Optional[str] = None
            for det in detections:
                label_uuid = self._resolve_label_uuid(det)
                if label_uuid is None:
                    continue
                if best is None or det.confidence > best.confidence:
                    best = det
                    best_label_uuid = label_uuid
            self._maybe_add_candidate(bbox, image_uuid, filename, label_names, best, best_label_uuid)

    @staticmethod
    def _crop_to_png_bytes(pil_image, box: tuple) -> bytes:
        buf = io.BytesIO()
        pil_image.crop(box).save(buf, format="PNG")
        return buf.getvalue()

    async def _load_image_bytes(self, filename: str, images_api: ImagesApi) -> bytes:
        # The task runs where FILES_ROOT is mounted, so a local read avoids a
        # base64 HTTP round-trip per image (mirrors ApplyFilterChainTask). The
        # HTTP path stays as a fallback for image_source != "filesystem".
        #
        # `filename` as stored/returned by the DB is FILES_ROOT-relative (or, in
        # some cases, an absolute path under FILES_ROOT) — it is never valid to
        # open() directly against the task-server process's cwd. Resolve it the
        # same way db_server/manager.py does (get_safe_path handles both forms).
        if self._image_source == "filesystem":
            safe_path = get_safe_path(FILES_ROOT, Path(filename))
            if safe_path is None:
                raise FileNotFoundError(f"Refusing to read path outside FILES_ROOT: {filename!r}")
            return await asyncio.to_thread(_read_file, str(safe_path))
        resp = await asyncio.to_thread(images_api.get_image_file, filename, True)
        if not resp.image_base64:
            raise ValueError(f"No image data returned for {filename}")
        return base64.b64decode(resp.image_base64)

    def _build_resume_data(self) -> Dict[str, Any]:
        return {
            "images_processed": self._images_processed,
            "candidates": self._candidates,
            "processed_image_uuids": sorted(self._processed_image_uuids),
            "images_failed": self._images_failed,
            "sample_errors": self._sample_errors,
            "no_detection_count": self._no_detection_count,
            "no_op_count": self._no_op_count,
        }

    def _build_result(self) -> Dict[str, Any]:
        return {
            "images_processed": self._images_processed,
            "bbox_count": len(self._bbox_uuids),
            "candidate_count": len(self._candidates),
            "candidates": self._candidates,
            "images_failed": self._images_failed,
            "sample_errors": self._sample_errors,
            "no_detection_count": self._no_detection_count,
            "no_op_count": self._no_op_count,
        }

    async def _deinit(self) -> None:
        if not self._pin_id:
            return
        inference_api_client = self._manager.get_inference_api_client()
        models_api = InferenceModelsApi(inference_api_client)
        try:
            await asyncio.to_thread(models_api.unpin_model, self._model_name, self._pin_id)
        except Exception:
            logger.exception("Failed to unpin model %s", self._model_name)

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "meta": {
                "order": ["model_name", "crop_mode", "conf_thresh", "iou_thresh", "image_uuids", "bbox_uuids"],
            },
            "model_name": {
                "type": "model_name",
                "required": True,
                "help": "Detector model to run against the candidate bboxes' images.",
            },
            "crop_mode": {
                "type": "str",
                "label": "Inference Input",
                "default": "image",
                "options": ["image", "bbox"],
                "description": (
                    "'image' runs the detector once per whole image, then matches detections to "
                    "candidate bboxes by IoU. 'bbox' crops each candidate bbox out and runs the "
                    "detector on just that crop (no IoU matching needed) — one inference call per "
                    "bbox instead of per image."
                ),
            },
            "conf_thresh": {
                "type": "float",
                "label": "Confidence Threshold",
                "default": 0.25,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            },
            "iou_thresh": {
                "type": "float",
                "label": "IoU Threshold",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Minimum IoU for a detection to match an existing bbox. Ignored in 'bbox' crop_mode.",
            },
            "image_uuids": {
                "type": "list",
                "label": "Image UUIDs",
                "required": True,
                "description": "Images to fetch and run inference on (each fetched/inferred once).",
            },
            "bbox_uuids": {
                "type": "list",
                "label": "Candidate Bbox UUIDs",
                "required": True,
                "description": f"Explicit list of bbox UUIDs to preview relabels for (hard cap {_MAX_BBOX_UUIDS}).",
            },
        }


def _read_file(filename: str) -> bytes:
    with open(filename, "rb") as f:
        return f.read()
