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

Two independent switches decide what a whole-image run may propose:
`add_new` collects detections that overlap no existing bbox as brand-new boxes,
and `relabel_existing` (default True) proposes relabels for the candidate
bboxes. Clearing the latter gives the "only add new boxes" run — existing boxes
are never touched, so the candidate bbox list stops mattering entirely. Both
off is rejected in `_init`: the run could only ever complete empty.

Detections whose class isn't mapped to a label UUID (`det.label_uuid is None`)
fall back to a name match against the project's label names (mirrors the
manual single-image "Run Model" UI, which already tolerates unmapped classes
by displaying the raw class string) before being discarded.

Optional `preview_run_uuid`: when set, candidates are streamed into the
`preview_candidates` table for that run as they are found and the one-shot
result carries only `{preview_run_uuid, <stats>}` instead of the whole candidate
array — so a 5,000-candidate run isn't serialized through the browser, and the
result survives the modal being closed.

⚠️ "This task never writes to the DB" above is about **annotation** tables —
bboxes, labels and tags are still untouched, and nothing here changes an image.
`preview_candidates` is scratch state describing a proposal; it becomes an
annotation only when the user applies the run.
"""

import asyncio
import io
import logging
from typing import Any, Dict, List, Optional, Set

from db_client.api.images_api import ImagesApi
from db_client.api.labels_api import LabelsApi
from db_client.api.preview_runs_api import PreviewRunsApi
from db_client.models.add_preview_candidates_payload import AddPreviewCandidatesPayload
from db_client.models.batch_by_uuids_payload import BatchByUuidsPayload
from db_client.models.preview_candidate_create import PreviewCandidateCreate
from inference_client.api.models_api import ModelsApi as InferenceModelsApi
from inference_client.models.pin_model_payload import PinModelPayload

from apps import FILES_ROOT
from apps.helpers.bboxUtils import iou_xywh
from apps.helpers.cascade import (
    CROP_CLASSIFIER_KIND,
    CropSubModel,
    order_sub_models,
    refinable_indices,
    resolve_refinement,
)
from apps.helpers.classifier_utils import DEFAULT_CROP_PADDING
from apps.helpers.consts import TaskAccent
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.metadataUtils import get_model_metadata, validate_bare_model_name

from ._filter_chain_runner import Detection
from ._image_loading import load_image_bytes
from .ApplyFilterChainTask import (
    _CLASSIFIER_KINDS,
    _MODELS_ROOT,
    _HttpInferenceBackend,
    _read_model_kind,
)
from .Registry import register_task
from .Task import Task
from .utils.labelHelper import build_label_ancestry_map

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

# No UX-facing ceiling on the candidate bbox list: the review-browser UI *warns*
# the user on large runs (and lets them cap the count), but "process all" is
# always allowed. This sanity cap is not a UX limit — a legitimate UI-driven
# run should never approach it — it only guards against a pathological or
# non-UI caller (e.g. a direct API request) handing this task an unbounded
# list, since self._candidates accumulates every match in memory for the
# life of the task and is serialized whole into checkpoints and the result.
_SANITY_MAX_BBOX_UUIDS = 50_000

# Images fetched per batch_images_by_uuids round-trip.
_IMAGE_BATCH_SIZE = 32

# Candidates buffered before being flushed into preview_candidates. Sized so a
# large run makes tens of round-trips rather than thousands, while keeping the
# in-memory buffer (and therefore what a crash loses) small.
_CANDIDATE_FLUSH_SIZE = 500

# Images in flight at once (see self._concurrency). Same default, and for the
# same reason, as ApplyFilterChainTask: inference is ~97% of the per-image cost
# here (a measured ~35ms round-trip against ~1ms of file read + base64), and
# only ~22ms of that round-trip is ONNX work, so a serial loop leaves the
# inference server idle between requests. Measured against the live server,
# throughput goes 30 -> 62 -> 72 img/s at concurrency 1 -> 2 -> 4 and then
# plateaus (8 gives no further gain) — 4 is the knee, past which the shared
# inference server, not this task, is the limit.
_DEFAULT_CONCURRENCY = 4

# NOTE: the crop-classifier kind is `cascade.CROP_CLASSIFIER_KIND`, deliberately NOT
# a member of ApplyFilterChainTask._CLASSIFIER_KINDS — that set means "scores the
# whole image, so it drives an image-level action", which is the opposite of this.


def _read_parent_label_uuid(model_name: str) -> Optional[str]:
    """The boundary label a crop classifier refines, from its sidecar."""
    validate_bare_model_name(model_name)
    try:
        metadata = get_model_metadata(str(_MODELS_ROOT / f"{model_name}.onnx"), create_json_if_missing=False)
    except FileNotFoundError:
        return None
    return metadata.get("parent_label_uuid") or None


def _read_crop_padding(model_name: str) -> float:
    """The context padding a crop classifier was trained with, from its sidecar.

    Serve-time framing has to match train-time framing or accuracy quietly drops,
    so this is read per model rather than assumed.
    """
    validate_bare_model_name(model_name)
    try:
        metadata = get_model_metadata(str(_MODELS_ROOT / f"{model_name}.onnx"), create_json_if_missing=False)
    except FileNotFoundError:
        return DEFAULT_CROP_PADDING
    return float(metadata.get("crop_padding", DEFAULT_CROP_PADDING))


@register_task()
class PreviewModelRelabelTask(Task):
    # A dry run: produces a result to look at, changes nothing.
    CARD_ACCENT = TaskAccent.TEAL

    # Started by Review's "Run Model on Results", which supplies the image scope
    # the current filter resolved to.
    SYSTEM_TASK = True

    async def _init(self, params: Dict[str, Any], resume_data: Optional[Dict[str, Any]]) -> None:
        self._model_name: str = str(params.get("model_name") or "")
        if not self._model_name:
            raise ValueError("model_name is required")
        validate_bare_model_name(self._model_name)

        self._conf_thresh: float = float(params.get("conf_thresh", 0.25))
        self._iou_thresh: float = float(params.get("iou_thresh", 0.5))
        self._image_source: str = str(params.get("image_source", "filesystem"))

        self._crop_mode: str = str(params.get("crop_mode") or "image")
        if self._crop_mode not in ("image", "bbox"):
            raise ValueError(f"crop_mode must be 'image' or 'bbox', got {self._crop_mode!r}")

        # Second-pass crop classifiers to run on this detector's output, turning one
        # run into both passes of a hierarchical config: the detector finds "Cat",
        # these say which cat, and the candidate that lands in review is already
        # named. Empty (the default) leaves the run exactly as it was.
        self._sub_model_names: List[str] = list(params.get("sub_model_names") or [])
        for name in self._sub_model_names:
            validate_bare_model_name(name)
        if self._sub_model_names and self._crop_mode != "image":
            raise ValueError(
                "sub_model_names applies to whole-image crop_mode only — in bbox mode the "
                "crop classifier IS the model to run, so pass it as model_name instead"
            )
        self._sub_model_min_confidence: float = float(params.get("sub_model_min_confidence", 0.6))

        # When set (whole-image crop_mode only), detections that overlap *no*
        # existing bbox on the image are also proposed as brand-new boxes — the
        # interactive counterpart to the filter-chain add_bbox action.
        self._add_new: bool = bool(params.get("add_new", False))

        # The other half of the "new boxes" decision: with this cleared, existing
        # bboxes are left alone entirely and the run proposes *only* new boxes.
        # Defaults True so every params dict written before this existed (and
        # every resumed run) behaves exactly as it did.
        self._relabel_existing: bool = bool(params.get("relabel_existing", True))
        if not self._relabel_existing and not (self._crop_mode == "image" and self._add_new):
            raise ValueError(
                "relabel_existing=False requires crop_mode 'image' with add_new set — "
                "otherwise the run has nothing it is allowed to propose"
            )

        self._image_uuids: List[str] = list(params.get("image_uuids") or [])
        if not self._image_uuids:
            raise ValueError("image_uuids is required")

        # Candidate bboxes to preview relabels for — a subset of the bboxes on the
        # given images. Held as a set for O(1) membership tests in _process_image.
        # Empty is only meaningful for the image-scoped "run on filtered images"
        # flow: whole-image crop mode with add_new, where every candidate comes
        # from _collect_new_boxes rather than relabeling an existing bbox. In any
        # other combination an empty list can never produce a candidate, so it's
        # rejected up front instead of silently completing with nothing.
        bbox_uuids: List[str] = list(params.get("bbox_uuids") or [])
        if not bbox_uuids and not (self._crop_mode == "image" and self._add_new):
            raise ValueError("bbox_uuids is required unless crop_mode is 'image' and add_new is set")
        if len(bbox_uuids) > _SANITY_MAX_BBOX_UUIDS:
            raise ValueError(
                f"bbox_uuids has {len(bbox_uuids)} entries, exceeding the sanity cap of "
                f"{_SANITY_MAX_BBOX_UUIDS}; narrow the filter before running the model"
            )
        self._bbox_uuids: Set[str] = set(bbox_uuids)

        # When set, candidates are written into this persisted preview run as the
        # task goes and the result carries only the uuid + stats. See the module
        # docstring for why this does not violate the "never writes to the DB"
        # invariant.
        self._preview_run_uuid: Optional[str] = params.get("preview_run_uuid") or None
        # Clamped to >= 1: asyncio.Semaphore(0) would block the batch loop forever.
        self._concurrency: int = max(1, int(params.get("concurrency", _DEFAULT_CONCURRENCY)))

        self._conflict_policy: str = str(params.get("conflict_policy") or "last")
        if self._conflict_policy not in ("last", "first"):
            raise ValueError(f"conflict_policy must be 'last' or 'first', got {self._conflict_policy!r}")

        self._images_processed: int = resume_data.get("images_processed", 0) if resume_data else 0
        # In preview-run mode this is only the *unflushed* tail: flushed
        # candidates are dropped from memory (and from the checkpoint), so a
        # resume cannot re-insert them. `_candidates_written` carries the count
        # the result reports.
        self._candidates: List[Dict[str, Any]] = list(resume_data.get("candidates", [])) if resume_data else []
        self._candidates_written: int = resume_data.get("candidates_written", 0) if resume_data else 0
        self._processed_image_uuids: set = set(resume_data.get("processed_image_uuids", [])) if resume_data else set()

        # Diagnostics so a run that finds zero candidates says *why* (image load
        # failures, no matching detection, or the model just agreeing with every
        # existing label) instead of silently completing empty.
        self._images_failed: int = resume_data.get("images_failed", 0) if resume_data else 0
        self._sample_errors: List[str] = list(resume_data.get("sample_errors", [])) if resume_data else []
        self._no_detection_count: int = resume_data.get("no_detection_count", 0) if resume_data else 0
        self._no_op_count: int = resume_data.get("no_op_count", 0) if resume_data else 0
        self._new_boxes_found: int = resume_data.get("new_boxes_found", 0) if resume_data else 0

        self._pin_id: Optional[str] = None
        # Populated in _run() once label names are fetched; defaults to empty so
        # _process_image can be exercised directly in tests without a full _run().
        self._name_to_label_uuid: Dict[str, str] = {}
        # Resolved from the model sidecar in _run(); the defaults keep a detector
        # model's behaviour identical to before crop classifiers existed.
        self._is_crop_classifier: bool = False
        self._crop_padding: float = 0.0
        # Resolved in _run() from each sub-model's sidecar + the label tree.
        self._sub_models: List[CropSubModel] = []
        self._label_ancestry: Dict[str, set] = {}
        self._refined_count: int = 0

    async def _run(self) -> Dict[str, Any]:
        db_api_client = self._manager.get_db_api_client()
        inference_api_client = self._manager.get_inference_api_client()
        images_api = ImagesApi(db_api_client)
        labels_api = LabelsApi(db_api_client)

        model_kind = await asyncio.to_thread(_read_model_kind, self._model_name)
        if model_kind in _CLASSIFIER_KINDS:
            self.log(
                f"Model {self._model_name!r} is a {model_kind}; relabel preview only supports "
                "detector and crop-classifier models — nothing to do"
            )
            return self._build_result()

        # A crop classifier is the second pass of a hierarchical config: it answers
        # "which child label is this box" and has no opinion about anything outside
        # the box, so it only makes sense in bbox mode. In image mode it would be
        # handed whole frames and would confidently mislabel every one of them.
        self._is_crop_classifier = model_kind == CROP_CLASSIFIER_KIND
        if self._is_crop_classifier:
            if self._crop_mode != "bbox":
                self.log(
                    f"Model {self._model_name!r} is a crop classifier; it classifies one box at a "
                    f"time and cannot run in {self._crop_mode!r} mode — re-run with crop_mode='bbox'"
                )
                return self._build_result()
            self._crop_padding = await asyncio.to_thread(_read_crop_padding, self._model_name)
            self.log(
                f"Model {self._model_name!r} is a crop classifier "
                f"(crop padding {self._crop_padding:.0%}, matching how it was trained)"
            )

        await self._resolve_sub_models(labels_api)

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
        # The *factory*, not the client it returns — `_HttpInferenceBackend` calls this
        # per request so a Settings → Servers change mid-run is picked up (see its
        # docstring). Passing the already-called client made every `detect()` raise
        # `TypeError: 'ApiClient' object is not callable`, which `_run`'s per-image
        # handler swallowed into `images_failed`: the run "succeeded" with zero
        # candidates, every time.
        backend = _HttpInferenceBackend(self._manager.get_inference_api_client, pin_ids)

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

            # Images within a batch are independent (each one only reads its own
            # bboxes and appends its own candidates), so they run concurrently to
            # keep several requests queued at the inference server instead of
            # leaving it idle between round-trips. The batch boundary stays a hard
            # sync point: every image finishes before the flush + checkpoint below,
            # so resume semantics are unchanged by the out-of-order completion.
            #
            # Candidates therefore accumulate in completion order rather than
            # image order. That is safe for `conflict_policy`, which arbitrates
            # between *different models* sharing one preview run — within a single
            # run each candidate bbox is visited exactly once (it belongs to one
            # image), so no two candidates here contend for the same row.
            semaphore = asyncio.Semaphore(self._concurrency)

            async def _run_one(image_uuid: str) -> bool:
                """True = this image was visited (count it as processed); False =
                skipped because the task was cancelled mid-batch, so it must stay
                out of the checkpoint and be retried on resume."""
                async with semaphore:
                    if self._cancel_flag.is_set():
                        return False
                    img = images_by_uuid.get(image_uuid)
                    if img is None:
                        self._images_failed += 1
                        if len(self._sample_errors) < 5:
                            self._sample_errors.append(f"{image_uuid}: not found in batch_images_by_uuids response")
                        return True
                    try:
                        await self._process_image(images_api, backend, img, label_names)
                    except Exception as e:
                        logger.exception("Failed to process image %s", image_uuid)
                        self._images_failed += 1
                        if len(self._sample_errors) < 5:
                            self._sample_errors.append(f"{image_uuid}: {e}")
                    return True

            visited = await asyncio.gather(*(_run_one(u) for u in batch))
            for image_uuid, was_visited in zip(batch, visited):
                if was_visited:
                    self._images_processed += 1
                    self._processed_image_uuids.add(image_uuid)

            # Before the checkpoint below, so a resume never re-inserts what has
            # already landed in the run.
            try:
                await self._flush_candidates()
            except Exception as e:
                logger.exception("Failed to flush preview candidates; will retry on the next batch")
                if len(self._sample_errors) < 5:
                    self._sample_errors.append(f"preview candidate flush: {e}")

            found = self._candidates_written + len(self._candidates)
            progress = min(100.0, self._images_processed / max(total, 1) * 100.0)
            self._update_progress(progress)
            self.log(f"Processed {self._images_processed}/{total} image(s), {found} candidate(s)")

            if self._data_req_flag.is_set():
                self._data_req_flag.clear()
                self._resume_data = self._build_resume_data()
                self._data_ready_flag.set()

            await self.heartbeat()
            await asyncio.sleep(0)

        await self._flush_candidates(force=True)
        self._resume_data = self._build_resume_data()
        self._data_ready_flag.set()
        self.log(
            f"Complete: {self._candidates_written + len(self._candidates)} candidate relabel(s) "
            f"from {self._images_processed} image(s) "
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
        geometry: Optional[Dict[str, float]] = None,
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
                # The refinement model's suggested box (image-relative x/y/width/height,
                # or None if the caller didn't compute one) — the client applies this in
                # place of the original box on relabel, assuming the model that matched
                # is more precise than whatever produced the original.
                "geometry": geometry,
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
        # the expensive step. Exception: whole-image crop mode with add_new set
        # still needs to run — a boxless image can still yield new-box candidates
        # via _collect_new_boxes below, even with zero existing candidate bboxes.
        # In only-add mode nothing is relabelled, so the candidate set is
        # irrelevant to what this image can produce — treat every image as
        # having no candidate bboxes and let the add_new path do all the work.
        image_bboxes = (
            [b for b in (img.get("bboxes") or []) if b.get("uuid") in self._bbox_uuids]
            if self._relabel_existing
            else []
        )
        if not image_bboxes and not (self._crop_mode == "image" and self._add_new):
            return
        image_bytes = await self._load_image_bytes(filename, images_api)

        if self._crop_mode == "bbox":
            if not image_bboxes:
                return
            await self._process_image_bbox_crop(image_bboxes, image_bytes, img, backend, label_names)
            return

        detections = await backend.detect(self._model_name, image_bytes, self._conf_thresh)
        if not detections:
            self._no_detection_count += len(image_bboxes)
            return

        # Second pass runs here, before anything reads a detection's label: both the
        # relabel matching below and _collect_new_boxes then see the refined name, so
        # one run really does produce "Willow" rather than "Cat" plus a follow-up run.
        if self._sub_models:
            await self._refine_detections(backend, image_bytes, detections)

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
            # Whole-image detections already share the bbox's coordinate space
            # (see module docstring), so the geometry needs no remapping.
            geometry = (
                {"x": best.x, "y": best.y, "width": best.width, "height": best.height} if best is not None else None
            )
            self._maybe_add_candidate(bbox, image_uuid, filename, label_names, best, best_label_uuid, geometry)

        if self._add_new:
            self._collect_new_boxes(img, image_uuid, filename, detections, label_names)

    def _collect_new_boxes(
        self,
        img: Dict[str, Any],
        image_uuid: str,
        filename: str,
        detections: List[Detection],
        label_names: Dict[str, str],
    ) -> None:
        """Propose detections that overlap *no* existing bbox on the image as new
        boxes. Overlap is checked against *all* of the image's bboxes (not just
        the candidate set), so a detection that lands on an already-correct,
        non-candidate box isn't re-added."""
        all_bboxes = img.get("bboxes") or []
        for det in detections:
            label_uuid = self._resolve_label_uuid(det)
            if label_uuid is None:
                continue
            if any(iou_xywh(det, b) >= self._iou_thresh for b in all_bboxes):
                continue  # overlaps an existing box — not new
            self._candidates.append(
                {
                    "bboxUuid": None,
                    "imageUuid": image_uuid,
                    "imagePath": filename,
                    "currentLabelUuid": None,
                    "currentLabelName": None,
                    "proposedLabelUuid": label_uuid,
                    "proposedLabelName": label_names.get(label_uuid),
                    "confidence": det.confidence,
                    "geometry": {"x": det.x, "y": det.y, "width": det.width, "height": det.height},
                    "isNew": True,
                }
            )
            self._new_boxes_found += 1

    async def _resolve_sub_models(self, labels_api: LabelsApi) -> None:
        """Turn the requested sub-model names into aimed CropSubModels.

        A name that isn't a crop classifier, or is one with no `parent_label_uuid`,
        is reported and skipped rather than failing the run: the first pass is still
        worth previewing, and a run that refuses to start over one misconfigured
        second-pass model is the less useful failure.
        """
        if not self._sub_model_names:
            return

        self._label_ancestry = await build_label_ancestry_map(labels_api)
        for name in self._sub_model_names:
            kind = await asyncio.to_thread(_read_model_kind, name)
            if kind != CROP_CLASSIFIER_KIND:
                self.log(f"Skipping second-pass model {name!r}: it is a {kind}, not a crop classifier")
                continue
            parent = await asyncio.to_thread(_read_parent_label_uuid, name)
            if not parent:
                self.log(
                    f"Skipping second-pass model {name!r}: its sidecar names no parent label, so there "
                    "is nothing to aim it at"
                )
                continue
            self._sub_models.append(CropSubModel(model_name=name, parent_label_uuid=parent))

        if self._sub_models:
            self.log(
                "Second pass: "
                + ", ".join(f"{s.model_name} refines {s.parent_label_uuid}" for s in self._sub_models)
                + f" (min confidence {self._sub_model_min_confidence})"
            )

    async def _refine_detections(
        self,
        backend: _HttpInferenceBackend,
        image_bytes: bytes,
        detections: List[Detection],
    ) -> None:
        """Replace each detection's coarse label with its sub-model's verdict, in place.

        A detection the sub-model wasn't confident about keeps its coarse label —
        the preview then proposes "Cat", which is still a correct answer and still
        reviewable, rather than dropping the box or guessing a cat's name.
        """
        # Outermost boundary first — see order_sub_models. Sorted here rather than at
        # resolve time so a hierarchy edit between runs can't leave a stale order.
        for sub in order_sub_models(self._sub_models, self._label_ancestry):
            indices = refinable_indices([d.label_uuid for d in detections], sub.parent_label_uuid, self._label_ancestry)
            if not indices:
                continue
            boxes = [
                {
                    "x": detections[i].x,
                    "y": detections[i].y,
                    "width": detections[i].width,
                    "height": detections[i].height,
                }
                for i in indices
            ]
            try:
                per_box = await backend.classify_crops(sub.model_name, image_bytes, boxes)
            except Exception as e:
                logger.exception("Second pass %s failed", sub.model_name)
                if len(self._sample_errors) < 5:
                    self._sample_errors.append(f"second pass {sub.model_name}: {e}")
                continue

            for idx, predictions in zip(indices, per_box):
                refined = resolve_refinement(
                    detections[idx].confidence,
                    predictions,
                    self._sub_model_min_confidence,
                    current_label_uuid=detections[idx].label_uuid,
                )
                if refined is None:
                    continue
                label_uuid, confidence = refined
                detections[idx].label_uuid = label_uuid
                # Clear the detector's class name so _resolve_label_uuid's
                # name-based fallback can't resolve the refined box back to the
                # coarse label it just moved away from.
                detections[idx].class_str = None
                detections[idx].confidence = confidence
                self._refined_count += 1

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
            # Context padding, zero for detectors (which see the bare box, as they
            # always have) and the sidecar's value for a crop classifier, which was
            # trained on padded crops and must be served the same framing.
            pad_x = width * self._crop_padding
            pad_y = height * self._crop_padding
            left = max(0, int((x - pad_x) * img_w))
            top = max(0, int((y - pad_y) * img_h))
            right = min(img_w, int((x + width + pad_x) * img_w))
            bottom = min(img_h, int((y + height + pad_y) * img_h))
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

            if self._is_crop_classifier:
                # A classifier returns scores, not boxes: take the top-scoring class
                # above the threshold and propose it for this bbox unchanged. No
                # geometry — the model localised nothing, so the existing box stands.
                predictions = await backend.classify_labels(self._model_name, crop_bytes)
                top = next(((lu, c) for lu, c in predictions if c >= self._conf_thresh), None)
                if top is None:
                    self._no_detection_count += 1
                    continue
                label_uuid, confidence = top
                self._maybe_add_candidate(
                    bbox,
                    image_uuid,
                    filename,
                    label_names,
                    Detection(x=x, y=y, width=width, height=height, confidence=confidence, label_uuid=label_uuid),
                    label_uuid,
                    None,
                )
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
            # The detector ran on the cropped bbox, so `best`'s coordinates are
            # relative to the crop, not the full image — remap into the same
            # image-relative space the bbox itself uses before proposing it.
            # Use the *clamped* crop region (left/top/right/bottom), not the raw
            # bbox width/height: a bbox extending past the image edge is cropped
            # smaller, so scaling by the unclamped width/height would overshoot.
            geometry = None
            if best is not None:
                crop_rx = left / img_w
                crop_ry = top / img_h
                crop_rw = (right - left) / img_w
                crop_rh = (bottom - top) / img_h
                geometry = {
                    "x": crop_rx + best.x * crop_rw,
                    "y": crop_ry + best.y * crop_rh,
                    "width": best.width * crop_rw,
                    "height": best.height * crop_rh,
                }
            self._maybe_add_candidate(bbox, image_uuid, filename, label_names, best, best_label_uuid, geometry)

    @staticmethod
    def _crop_to_png_bytes(pil_image, box: tuple) -> bytes:
        buf = io.BytesIO()
        pil_image.crop(box).save(buf, format="PNG")
        return buf.getvalue()

    async def _load_image_bytes(self, filename: str, images_api: ImagesApi) -> bytes:
        return await load_image_bytes(filename, self._image_source, images_api, FILES_ROOT)

    async def _flush_candidates(self, force: bool = False) -> None:
        """Write the buffered candidates into the persisted preview run.

        No-op unless `preview_run_uuid` was given — without it the task behaves
        exactly as it always did, accumulating in memory and returning the whole
        array.

        Flushed candidates are removed from `self._candidates` so they leave the
        checkpoint too: a resumed run must not re-insert what it already wrote.
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
                    bbox_uuid=c.get("bboxUuid"),
                    is_new=bool(c.get("isNew")),
                    current_label_uuid=c.get("currentLabelUuid"),
                    proposed_label_uuid=c["proposedLabelUuid"],
                    confidence=float(c.get("confidence") or 0.0),
                    geometry=c.get("geometry"),
                    source_model=self._model_name,
                )
                for c in batch
            ],
            conflict_policy=self._conflict_policy,
        )
        api = PreviewRunsApi(self._manager.get_db_api_client())
        await asyncio.to_thread(api.add_preview_candidates, self._preview_run_uuid, payload)
        self._candidates_written += len(batch)
        self._candidates = []

    def _build_resume_data(self) -> Dict[str, Any]:
        return {
            "images_processed": self._images_processed,
            "candidates": self._candidates,
            "candidates_written": self._candidates_written,
            "processed_image_uuids": sorted(self._processed_image_uuids),
            "images_failed": self._images_failed,
            "sample_errors": self._sample_errors,
            "no_detection_count": self._no_detection_count,
            "no_op_count": self._no_op_count,
            "new_boxes_found": self._new_boxes_found,
        }

    def _build_result(self) -> Dict[str, Any]:
        result = {
            "images_processed": self._images_processed,
            "bbox_count": len(self._bbox_uuids),
            "candidate_count": self._candidates_written + len(self._candidates),
            "candidates": self._candidates,
            "images_failed": self._images_failed,
            "sample_errors": self._sample_errors,
            "no_detection_count": self._no_detection_count,
            "no_op_count": self._no_op_count,
            "new_boxes_found": self._new_boxes_found,
            # How many detections the second pass renamed. Zero with sub-models
            # configured is the signal that the cascade ran but never cleared its
            # confidence bar — a different problem from "the detector found nothing".
            "refined_count": self._refined_count,
        }
        if self._preview_run_uuid:
            # The candidates live in the DB now; shipping them through the
            # one-shot result as well would defeat the point of persisting them.
            result["preview_run_uuid"] = self._preview_run_uuid
            result["candidates"] = []
        return result

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
                "order": [
                    "model_name",
                    "crop_mode",
                    "add_new",
                    "relabel_existing",
                    "conf_thresh",
                    "iou_thresh",
                    "concurrency",
                    "image_uuids",
                    "bbox_uuids",
                    "preview_run_uuid",
                    "conflict_policy",
                ],
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
            "add_new": {
                "type": "boolean",
                "label": "Add New Boxes",
                "default": False,
                "description": (
                    "Also propose detections that overlap no existing bbox as brand-new boxes "
                    "(whole-image crop_mode only)."
                ),
            },
            "relabel_existing": {
                "type": "boolean",
                "label": "Relabel Existing Boxes",
                "default": True,
                "description": (
                    "Propose relabels for the candidate bboxes. Clear it (with add_new set, in "
                    "whole-image crop_mode) to add new boxes only and leave existing boxes untouched."
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
            "concurrency": {
                "type": "int",
                "label": "Concurrency",
                "default": _DEFAULT_CONCURRENCY,
                "description": (
                    "Images processed in parallel. Higher keeps the GPU busier, up to what the "
                    "inference server can absorb."
                ),
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
                "required": False,
                "description": (
                    "Explicit list of bbox UUIDs to preview relabels for. May be empty only when "
                    "crop_mode is 'image' and add_new is set — every candidate then comes from "
                    "new-box detection instead of relabeling an existing bbox."
                ),
            },
            "preview_run_uuid": {
                "type": "str",
                "label": "Preview Run UUID",
                "required": False,
                "description": (
                    "Persist candidates into this preview run as they are found, instead of "
                    "returning them all in the one-shot result. Several models sharing one uuid "
                    "land in the same run."
                ),
            },
            "conflict_policy": {
                "type": "str",
                "label": "Multi-Model Conflict Policy",
                "default": "last",
                "options": ["last", "first"],
                "description": (
                    "When several models propose on the same bbox in one preview run, whether "
                    "the later model overwrites the earlier one ('last') or the first proposal "
                    "is kept ('first'). Only meaningful with preview_run_uuid set."
                ),
            },
        }
