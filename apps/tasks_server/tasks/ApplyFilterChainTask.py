"""ApplyFilterChainTask — retroactively run a FilterChain against saved images.

Scope MUST be bounded (directory / image_uuids / tag_uuid / label_uuid /
dataset_uuid, capped by max_images) — see WORKFLOW_FILTERCHAIN_PLAN.md 2.4:
"match existing bboxes" is an unbounded scan and must never silently run over
the whole DB.

Uses the shared `FilterChainRunner` (`_filter_chain_runner.py`) for match
evaluation; this task owns the DB-write side: applying `discard_image` (a
retroactive no-op — existing images are never deleted), `tag_image`,
`strip_unverified_bboxes`, `relabel_bbox` (apply-now + review tag + prior-label
stash, per plan 2.1a), and `tag_bbox`.
"""

import asyncio
import base64
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from uuid import NAMESPACE_OID, uuid5

from db_client.api.datasets_api import DatasetsApi
from db_client.api.filter_chains_api import FilterChainsApi
from db_client.api.images_api import ImagesApi
from db_client.api.labels_api import LabelsApi
from db_client.api.tags_api import TagsApi
from db_client.models.add_bbox_payload import AddBboxPayload
from db_client.models.add_tag_payload import AddTagPayload
from db_client.models.b_box_tag_term import BBoxTagTerm
from db_client.models.batch_by_uuids_payload import BatchByUuidsPayload
from db_client.models.bbox_patch_payload import BboxPatchPayload
from db_client.models.get_image_metadata_payload import GetImageMetadataPayload
from db_client.models.image_record_read import ImageRecordRead
from db_client.models.image_search_params import ImageSearchParams
from db_client.models.set_image_tag_payload import SetImageTagPayload
from db_client.models.tag_kind import TagKind as ApiTagKind
from db_client.models.terms_inner import TermsInner
from inference_client.api.inference_api import InferenceApi
from inference_client.api.models_api import ModelsApi as InferenceModelsApi
from inference_client.models.classify_payload import ClassifyPayload
from inference_client.models.inference_payload import InferencePayload
from inference_client.models.pin_model_payload import PinModelPayload

from apps import FILES_ROOT
from apps.helpers.db.types import UNVERIFIED_TAG_UUID
from apps.helpers.fileUtils import filter_stubs_by_directory
from apps.helpers.metadataUtils import get_model_metadata, validate_bare_model_name

from ._filter_chain_runner import Detection, FilterChainRunner
from ._image_loading import load_image_bytes
from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

_MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/models"))
_MAX_IMAGES_CEILING = 5000
_CLASSIFIER_KINDS = {"image_classifier", "multi_label_classifier"}
_SCOPE_PAGE_SIZE = 200
# Images in flight at once (see self._concurrency). 4 keeps the inference
# server's single GPU thread fed without piling up enough concurrent decodes to
# swamp its event loop — inference is the shared resource, so raising this only
# helps up to the point that server saturates.
_DEFAULT_CONCURRENCY = 4


def _read_model_kind(model_name: str) -> str:
    validate_bare_model_name(model_name)
    model_path = _MODELS_ROOT / f"{model_name}.onnx"
    try:
        metadata = get_model_metadata(str(model_path), create_json_if_missing=False)
    except FileNotFoundError:
        return "detector"
    return metadata.get("model_kind") or "detector"


class _HttpInferenceBackend:
    """InferenceBackend (see _filter_chain_runner.py) that talks to inference_server
    over HTTP. Detector steps reuse a pin acquired once for the whole task run;
    classifier steps go through /classify, which auto-pins per call (no pin_id
    param exists on that endpoint) — acceptable overhead for a batch task, not a
    hot path like collect-time inference."""

    def __init__(self, inference_api_client, pin_ids: Dict[str, str]):
        self._inference_api = InferenceApi(inference_api_client)
        self._pin_ids = pin_ids
        self._kind_cache: Dict[str, str] = {}

    async def get_model_kind(self, model_name: str) -> str:
        if model_name not in self._kind_cache:
            self._kind_cache[model_name] = await asyncio.to_thread(_read_model_kind, model_name)
        return self._kind_cache[model_name]

    async def classify_whole_image(self, model_name: str, image_bytes: bytes) -> float:
        b64 = base64.b64encode(image_bytes).decode()
        resp = await asyncio.to_thread(self._inference_api.classify, model_name, ClassifyPayload(image_base64=b64))
        if resp.status != "success" or resp.result is None:
            return 0.0
        preds = resp.result.predictions or []
        return max((p.confidence for p in preds), default=0.0)

    async def detect(self, model_name: str, image_bytes: bytes, conf_thresh: float) -> List[Detection]:
        b64 = base64.b64encode(image_bytes).decode()
        resp = await asyncio.to_thread(
            self._inference_api.detect,
            model_name,
            InferencePayload(image_base64=b64, conf_thresh=conf_thresh, pin_id=self._pin_ids.get(model_name)),
        )
        if resp.status != "success" or resp.result is None:
            return []
        return [
            Detection(
                x=d.bbox.x,
                y=d.bbox.y,
                width=d.bbox.width,
                height=d.bbox.height,
                confidence=d.confidence,
                label_uuid=d.bbox.label_uuid,
                class_str=d.bbox.class_str,
            )
            for d in (resp.result.detections or [])
        ]


@register_task()
class ApplyFilterChainTask(Task):
    async def _init(self, params: Dict[str, Any], resume_data: Optional[Dict[str, Any]]) -> None:
        self._filter_chain_uuid: str = str(params.get("filter_chain_uuid") or "")
        if not self._filter_chain_uuid:
            raise ValueError("filter_chain_uuid is required")

        self._directory: str = (params.get("directory") or "").strip().rstrip("/")
        self._image_uuids: List[str] = list(params.get("image_uuids") or [])
        self._tag_uuid: Optional[str] = params.get("tag_uuid") or None
        self._label_uuid: Optional[str] = params.get("label_uuid") or None
        self._dataset_uuid: Optional[str] = params.get("dataset_uuid") or None
        if not any([self._directory, self._image_uuids, self._tag_uuid, self._label_uuid, self._dataset_uuid]):
            raise ValueError(
                "At least one scope filter is required: directory, image_uuids, tag_uuid, label_uuid, or dataset_uuid"
            )

        self._max_images: int = min(int(params.get("max_images", _MAX_IMAGES_CEILING)), _MAX_IMAGES_CEILING)
        self._batch_size: int = int(params.get("batch_size", 32))
        self._dry_run: bool = bool(params.get("dry_run", False))
        self._image_source: str = str(params.get("image_source", "filesystem"))
        # How many images are in flight at once. Each image is overwhelmingly
        # wait (disk read, inference HTTP round-trip, DB writes) rather than CPU
        # in *this* process, so running one at a time left the GPU idle ~90% of
        # the time: it sat between requests with nothing queued. Clamped to >=1
        # so a 0/negative param can't deadlock the batch loop.
        self._concurrency: int = max(1, int(params.get("concurrency", _DEFAULT_CONCURRENCY)))

        self._images_processed: int = resume_data.get("images_processed", 0) if resume_data else 0
        self._images_changed: int = resume_data.get("images_changed", 0) if resume_data else 0
        self._bboxes_relabeled: int = resume_data.get("bboxes_relabeled", 0) if resume_data else 0
        self._bboxes_tagged: int = resume_data.get("bboxes_tagged", 0) if resume_data else 0
        self._bboxes_stripped: int = resume_data.get("bboxes_stripped", 0) if resume_data else 0
        self._bboxes_added: int = resume_data.get("bboxes_added", 0) if resume_data else 0
        self._images_tagged: int = resume_data.get("images_tagged", 0) if resume_data else 0
        self._images_discard_matched: int = resume_data.get("images_discard_matched", 0) if resume_data else 0

        self._pin_ids: Dict[str, str] = {}
        # Per-instance review tags are created lazily; track which uuids we've
        # already ensured so relabel and add_bbox can each have their own tag.
        self._review_tags_ensured: set = set()
        # Serializes _ensure_review_tag: its check-then-create spans awaits, so
        # concurrent images could otherwise both miss the cache and race to
        # create the same tag. Created lazily in _run() — _init() may run off
        # the loop that later owns the task.
        self._review_tag_lock: Optional[asyncio.Lock] = None
        # name -> label_uuid fallback for add_bbox, populated in _run().
        self._name_to_label_uuid: Dict[str, str] = {}

    async def _run(self) -> Dict[str, Any]:
        self._review_tag_lock = asyncio.Lock()
        db_api_client = self._manager.get_db_api_client()
        inference_api_client = self._manager.get_inference_api_client()
        filter_chains_api = FilterChainsApi(db_api_client)
        images_api = ImagesApi(db_api_client)
        tags_api = TagsApi(db_api_client)
        labels_api = LabelsApi(db_api_client)

        chain_resp = await asyncio.to_thread(filter_chains_api.get_filter_chain, self._filter_chain_uuid)
        chain = getattr(chain_resp, "chain", None)
        if chain is None:
            raise ValueError(f"Filter chain not found: {self._filter_chain_uuid}")

        steps = [
            {
                "order": s.order,
                "model_name": s.model_name,
                "confidence": s.confidence,
                "iou_threshold": s.iou_threshold,
                "action": s.action,
                "action_params": s.action_params or {},
                "enabled": s.enabled,
            }
            for s in (chain.steps or [])
            if s.enabled
        ]
        if not steps:
            self.log("Filter chain has no enabled steps; nothing to do")
            return self._build_result()
        if len(steps) > 5:
            self.log(f"Warning: chain has {len(steps)} steps — this multiplies inference cost per image")

        # For add_bbox steps, a detection whose class isn't mapped to a project
        # label falls back to a name match (first label with that name wins),
        # mirroring PreviewModelRelabelTask / the manual single-image Run Model UI.
        if any(s["action"] == "add_bbox" for s in steps):
            self._name_to_label_uuid = await self._build_name_to_label_uuid(labels_api)

        await self._pin_detector_models(inference_api_client, steps)
        backend = _HttpInferenceBackend(inference_api_client, self._pin_ids)
        runner = FilterChainRunner(steps, backend)

        # Deterministic per-instance review tags, so each run's relabels and each
        # run's added boxes are separately reviewable/reversible batches (plan
        # 2.1a / open item 4.2). add_bbox gets its own tag so newly-created boxes
        # aren't conflated with relabeled existing ones.
        review_tag_uuid = str(uuid5(NAMESPACE_OID, self._instance_uuid))
        review_tag_name = f"[Filter Chain] {chain.name} auto-relabeled ({self._instance_uuid[:8]})"
        add_tag_uuid = str(uuid5(NAMESPACE_OID, self._instance_uuid + ":add"))
        add_tag_name = f"[Filter Chain] {chain.name} auto-added ({self._instance_uuid[:8]})"

        stubs = await self._collect_stubs(db_api_client, images_api)
        if self._images_processed > 0 and self._images_processed < len(stubs):
            stubs = stubs[self._images_processed :]
            self.log(f"Resuming from image {self._images_processed}")
        total = len(stubs)
        dry_label = " [DRY RUN]" if self._dry_run else ""
        self.log(f"Applying filter chain '{chain.name}' to {total} images{dry_label}")

        await self.heartbeat(force=True)

        for batch_start in range(0, total, self._batch_size):
            if self._cancel_flag.is_set():
                break
            await self._check_pause()

            batch = stubs[batch_start : batch_start + self._batch_size]
            # Fetch the whole batch's image records in one round-trip instead of
            # one get_image_metadata call per image (an N+1 over up to max_images).
            prefetched = await self._prefetch_batch_records(images_api, batch)

            # Images within a batch are independent (each touches only its own
            # bboxes/tags), so they run concurrently to keep several requests
            # queued at the inference server instead of leaving it idle between
            # round-trips. The batch boundary stays a hard sync point: every
            # image finishes before resume data is written below, so resume
            # semantics are unchanged by the out-of-order completion.
            semaphore = asyncio.Semaphore(self._concurrency)

            async def _run_one(stub: Any) -> Optional[bool]:
                """True/False = processed (changed or not); None = skipped."""
                async with semaphore:
                    if self._cancel_flag.is_set():
                        return None
                    try:
                        return await self._process_image(
                            images_api,
                            tags_api,
                            runner,
                            stub,
                            review_tag_uuid,
                            review_tag_name,
                            add_tag_uuid,
                            add_tag_name,
                            prefetched.get(getattr(stub, "uuid", None)),
                        )
                    except Exception:
                        logger.exception("Failed to process image %s", getattr(stub, "filename", "?"))
                        return False

            outcomes = await asyncio.gather(*(_run_one(stub) for stub in batch))
            self._images_changed += sum(1 for changed in outcomes if changed)
            self._images_processed += sum(1 for changed in outcomes if changed is not None)

            progress = min(100.0, self._images_processed / max(total, 1) * 100.0)
            self._update_progress(progress)
            self.log(f"Processed {self._images_processed}/{total}, {self._images_changed} changed")

            if self._data_req_flag.is_set():
                self._data_req_flag.clear()
                self._resume_data = self._build_resume_data()
                self._data_ready_flag.set()

            await self.heartbeat()
            await asyncio.sleep(0)

        self._resume_data = self._build_resume_data()
        self._data_ready_flag.set()
        self.log(f"Complete{dry_label}: {self._images_processed} processed, {self._images_changed} changed")
        return self._build_result()

    async def _pin_detector_models(self, inference_api_client, steps: List[Dict[str, Any]]) -> None:
        """Pin every detector-step model once for the whole batch (classifier steps
        auto-pin per /classify call — see _HttpInferenceBackend)."""
        models_api = InferenceModelsApi(inference_api_client)
        for name in sorted({s["model_name"] for s in steps}):
            kind = await asyncio.to_thread(_read_model_kind, name)
            if kind in _CLASSIFIER_KINDS:
                continue
            try:
                pin_resp = await asyncio.to_thread(
                    models_api.pin_model_api_api_models_model_name_pin_post,
                    name,
                    PinModelPayload(model_name=name, duration=3600),
                )
                if pin_resp.status == "success" and pin_resp.pin_id:
                    self._pin_ids[name] = pin_resp.pin_id
            except Exception:
                logger.exception("Failed to pin model %s", name)

    async def _prefetch_batch_records(self, images_api: ImagesApi, batch: List[Any]) -> Dict[str, Any]:
        """Batch-fetch full image records (bboxes + tags) for a batch of stubs in one
        round-trip, keyed by uuid. batch_images_by_uuids returns camelCase dicts
        (BatchImagesResponse.images: List[dict]); deserialize each into the same typed
        ImageRecordRead that get_image_metadata returns so _process_image is unchanged.

        Best-effort: any stub without a uuid, or any record that fails to fetch/parse,
        is simply absent from the map — _process_image falls back to get_image_metadata
        for it, so this can never change results, only save round-trips."""
        uuids = [u for u in (getattr(s, "uuid", None) for s in batch) if u]
        if not uuids:
            return {}
        try:
            resp = await asyncio.to_thread(images_api.batch_images_by_uuids, BatchByUuidsPayload(image_uuids=uuids))
        except Exception:
            logger.exception("Batch image prefetch failed; falling back to per-image fetch")
            return {}
        records: Dict[str, Any] = {}
        for raw in resp.images or []:
            try:
                record = ImageRecordRead.from_dict(raw)
            except Exception:
                logger.exception("Failed to parse prefetched image record; will refetch individually")
                continue
            if record is not None and record.uuid:
                records[record.uuid] = record
        return records

    async def _build_name_to_label_uuid(self, labels_api: LabelsApi) -> Dict[str, str]:
        """name -> label_uuid, first label with a given name winning (matches the
        unmapped-class fallback in PreviewModelRelabelTask)."""
        resp = await asyncio.to_thread(labels_api.list_labels)
        name_to_uuid: Dict[str, str] = {}
        for lbl in resp.labels or []:
            name_to_uuid.setdefault(lbl.name, lbl.uuid)
        return name_to_uuid

    async def _ensure_review_tag(self, tags_api: TagsApi, tag_uuid: str, tag_name: str) -> None:
        if tag_uuid in self._review_tags_ensured:
            return
        # Re-check under the lock: with images processed concurrently, several
        # can clear the fast path above before any of them creates the tag.
        assert self._review_tag_lock is not None, "_ensure_review_tag called outside _run()"
        async with self._review_tag_lock:
            if tag_uuid in self._review_tags_ensured:
                return
            existing = await asyncio.to_thread(tags_api.list_tags)
            existing_uuids = {t.uuid for t in (existing.tags or [])}
            if tag_uuid not in existing_uuids:
                await asyncio.to_thread(
                    tags_api.add_tag,
                    AddTagPayload(uuid=tag_uuid, name=tag_name, color="#e8a020", kind=ApiTagKind.TASK_GENERATED),
                )
            self._review_tags_ensured.add(tag_uuid)

    async def _process_image(
        self,
        images_api: ImagesApi,
        tags_api: TagsApi,
        runner: FilterChainRunner,
        stub: Any,
        review_tag_uuid: str,
        review_tag_name: str,
        add_tag_uuid: str,
        add_tag_name: str,
        prefetched_record: Any = None,
    ) -> bool:
        image_bytes = await self._load_image_bytes(stub.filename, images_api)
        # Use the batched record when it already carries bboxes. An empty bbox
        # list can mean the DB row hasn't been synced from its JSON sidecar yet,
        # so fall back to get_image_metadata (which performs that sync) to keep
        # behavior identical to the un-batched path.
        image_record = prefetched_record if (prefetched_record is not None and prefetched_record.bboxes) else None
        if image_record is None:
            meta_resp = await asyncio.to_thread(
                images_api.get_image_metadata, GetImageMetadataPayload(image_path=stub.filename)
            )
            if meta_resp.status != "success" or meta_resp.image is None:
                return False
            image_record = meta_resp.image

        bboxes_by_uuid = {b.uuid: b for b in (image_record.bboxes or [])}
        existing = [
            {"uuid": b.uuid, "x": b.x, "y": b.y, "width": b.width, "height": b.height, "label_uuid": b.label_uuid}
            for b in bboxes_by_uuid.values()
        ]

        matches = await runner.evaluate_image(image_bytes, existing)
        if not matches:
            return False

        image_tag_uuids_to_add: set = set()
        bbox_uuids_to_strip: set = set()
        # bbox_uuid -> {"add_tags": set, "label_uuid"?: str, "prev_label_uuid"?: str}
        bbox_patches: Dict[str, Dict[str, Any]] = {}
        # New boxes to create from add_bbox steps: list of
        # {"label_uuid", "x", "y", "width", "height", "confidence", "model_name"}.
        bboxes_to_add: List[Dict[str, Any]] = []

        for match in matches:
            step = match.step
            action = step["action"]
            action_params = step.get("action_params") or {}

            if match.unmatched_detections:
                # add_bbox: each detection that overlapped no existing box becomes a
                # new box. Label = optional target override, else the detection's own
                # predicted label, else a name match; unresolved detections are skipped.
                forced_label = action_params.get("target_label_uuid")
                for det in match.unmatched_detections:
                    label_uuid = forced_label or self._resolve_detection_label(det)
                    if not label_uuid:
                        continue
                    bboxes_to_add.append(
                        {
                            "label_uuid": label_uuid,
                            "x": det.x,
                            "y": det.y,
                            "width": det.width,
                            "height": det.height,
                            "confidence": det.confidence,
                            "model_name": step["model_name"],
                        }
                    )
                continue

            if match.image_match:
                if action == "discard_image":
                    # Retroactive no-op — existing images are never deleted (plan 2.1).
                    self._images_discard_matched += 1
                elif action == "tag_image":
                    tag_uuid = action_params.get("tag_uuid")
                    if tag_uuid:
                        image_tag_uuids_to_add.add(tag_uuid)
                elif action == "strip_unverified_bboxes":
                    for b in bboxes_by_uuid.values():
                        tag_uuids = {t.uuid for t in (b.tags or [])}
                        if UNVERIFIED_TAG_UUID in tag_uuids and b.uuid not in bbox_uuids_to_strip:
                            bbox_uuids_to_strip.add(b.uuid)
                            self._bboxes_stripped += 1
                else:
                    logger.warning("Action %r is not valid for an image-level (classifier) step; skipping", action)
                continue

            for bbox_uuid in match.bbox_matches:
                if bbox_uuid in bbox_uuids_to_strip:
                    continue
                if action == "relabel_bbox":
                    target = action_params.get("target_label_uuid")
                    if not target:
                        continue
                    patch = bbox_patches.setdefault(bbox_uuid, {"add_tags": set()})
                    if "label_uuid" not in patch:
                        patch["prev_label_uuid"] = bboxes_by_uuid[bbox_uuid].label_uuid
                        self._bboxes_relabeled += 1
                    patch["label_uuid"] = target
                    patch["add_tags"].add(review_tag_uuid)
                elif action == "tag_bbox":
                    tag_uuid = action_params.get("tag_uuid")
                    if not tag_uuid:
                        continue
                    patch = bbox_patches.setdefault(bbox_uuid, {"add_tags": set()})
                    if tag_uuid not in patch["add_tags"]:
                        patch["add_tags"].add(tag_uuid)
                        self._bboxes_tagged += 1
                else:
                    logger.warning("Action %r is not valid for a bbox-level (detector) step; skipping", action)

        changed = bool(image_tag_uuids_to_add or bbox_uuids_to_strip or bbox_patches or bboxes_to_add)
        if self._dry_run or not changed:
            if self._dry_run:
                # Count add candidates so a dry run reports them without writing.
                self._bboxes_added += len(bboxes_to_add)
            return changed

        if image_tag_uuids_to_add:
            for tag_uuid in image_tag_uuids_to_add:
                await asyncio.to_thread(
                    images_api.set_image_tag,
                    SetImageTagPayload(image_path=stub.filename, tag_uuid=tag_uuid, action="add"),
                )
            self._images_tagged += 1

        for bbox_uuid in bbox_uuids_to_strip:
            await asyncio.to_thread(images_api.delete_bbox, bbox_uuid)

        if bbox_patches:
            if any("label_uuid" in p for p in bbox_patches.values()):
                await self._ensure_review_tag(tags_api, review_tag_uuid, review_tag_name)

            # PATCH each affected bbox individually (BboxPatchPayload) rather than
            # a full-image update_image_metadata replace: that endpoint silently
            # drops any bbox lacking a label_uuid from the *entire* image (server
            # filters `if b.label_uuid`), which would permanently delete unrelated
            # unlabeled bboxes on the same image as a side effect of this patch.
            for bbox_uuid, patch in bbox_patches.items():
                if bbox_uuid in bbox_uuids_to_strip:
                    # A later step stripped this bbox after an earlier step queued
                    # a patch for it — it's already deleted above; don't re-touch it.
                    continue
                b = bboxes_by_uuid[bbox_uuid]
                tag_uuids = {t.uuid for t in (b.tags or [])} | patch["add_tags"]
                metadata_json = None
                label_uuid = None
                if "label_uuid" in patch:
                    label_uuid = patch["label_uuid"]
                    metadata_json = {**(b.metadata_json or {}), "relabel_prev_label_uuid": patch["prev_label_uuid"]}
                await asyncio.to_thread(
                    images_api.update_bbox,
                    bbox_uuid,
                    BboxPatchPayload(label_uuid=label_uuid, tag_uuids=sorted(tag_uuids), metadata_json=metadata_json),
                )

        if bboxes_to_add:
            # New boxes are created UNVERIFIED + this run's add tag, so they're a
            # reviewable/reversible batch (a detector can be wrong — an added box
            # should never silently count as a confirmed annotation).
            await self._ensure_review_tag(tags_api, add_tag_uuid, add_tag_name)
            for new in bboxes_to_add:
                await asyncio.to_thread(
                    images_api.add_bbox,
                    AddBboxPayload(
                        image_path=stub.filename,
                        label_uuid=new["label_uuid"],
                        x=new["x"],
                        y=new["y"],
                        width=new["width"],
                        height=new["height"],
                        tag_uuids=[UNVERIFIED_TAG_UUID, add_tag_uuid],
                        metadata_json={
                            "filter_chain_added": True,
                            "source_model": new["model_name"],
                            "source_confidence": new["confidence"],
                        },
                    ),
                )
                self._bboxes_added += 1

        return True

    def _resolve_detection_label(self, det: Detection) -> Optional[str]:
        """A detection's class may be unmapped to a label UUID (det.label_uuid is
        None) — fall back to matching its raw class string against a project label
        of the same name (see PreviewModelRelabelTask._resolve_label_uuid)."""
        if det.label_uuid is not None:
            return det.label_uuid
        if det.class_str is None:
            return None
        return self._name_to_label_uuid.get(det.class_str)

    async def _collect_stubs(self, db_api_client, images_api: ImagesApi) -> List[Any]:
        """Resolve the bounded scope into a list of image stubs (uuid + filename)."""
        if self._image_uuids:
            capped = self._image_uuids[: self._max_images]
            if len(capped) < len(self._image_uuids):
                self.log(f"image_uuids capped at max_images={self._max_images} (had {len(self._image_uuids)})")
            resp = await asyncio.to_thread(images_api.batch_images_by_uuids, BatchByUuidsPayload(image_uuids=capped))
            # batch_images_by_uuids serializes with by_alias=True, so its response
            # images are plain camelCase dicts (BatchImagesResponse.images: List[dict]),
            # not attribute objects. Normalize to the same (uuid, filename) stub shape
            # the other scopes return so _process_image's `stub.filename` works.
            return [SimpleNamespace(uuid=img.get("uuid"), filename=img.get("filename")) for img in (resp.images or [])]

        if self._directory:
            resp = await asyncio.to_thread(images_api.list_image_stubs)
            stubs = filter_stubs_by_directory(FILES_ROOT, self._directory, resp.stubs or [])
            return self._cap_stubs(stubs, "directory")

        if self._dataset_uuid:
            datasets_api = DatasetsApi(db_api_client)
            stubs: List[Any] = []
            after_uuid = None
            while len(stubs) < self._max_images:
                resp = await asyncio.to_thread(
                    datasets_api.get_dataset_images, self._dataset_uuid, limit=_SCOPE_PAGE_SIZE, after_uuid=after_uuid
                )
                page = list(resp.images or [])
                if not page:
                    break
                stubs.extend(page)
                if not resp.has_more:
                    break
                after_uuid = page[-1].uuid
            return self._cap_stubs(stubs, "dataset_uuid")

        if self._label_uuid:
            resp = await asyncio.to_thread(images_api.list_images_for_label, self._label_uuid)
            return self._cap_stubs(list(resp.images or []), "label_uuid")

        # tag_uuid scope
        stubs = []
        after_uuid = None
        after_sort_value = None
        while len(stubs) < self._max_images:
            params = ImageSearchParams(
                terms=[TermsInner(BBoxTagTerm(tag_uuids=[self._tag_uuid], match_mode="any"))],
                limit=_SCOPE_PAGE_SIZE,
                sort_by="uuid",
                sort_order="asc",
                after_uuid=after_uuid,
                after_sort_value=after_sort_value,
            )
            resp = await asyncio.to_thread(images_api.get_review_queue, params)
            page = list(resp.images or [])
            if not page:
                break
            stubs.extend(page)
            if not resp.has_more:
                break
            after_uuid = page[-1].uuid
        return self._cap_stubs(stubs, "tag_uuid")

    def _cap_stubs(self, stubs: List[Any], scope_name: str) -> List[Any]:
        if len(stubs) > self._max_images:
            self.log(f"{scope_name} scope capped at max_images={self._max_images} (matched {len(stubs)})")
            return stubs[: self._max_images]
        return stubs

    async def _load_image_bytes(self, filename: str, images_api: ImagesApi) -> bytes:
        return await load_image_bytes(filename, self._image_source, images_api, FILES_ROOT)

    def _build_resume_data(self) -> Dict[str, Any]:
        return {
            "images_processed": self._images_processed,
            "images_changed": self._images_changed,
            "bboxes_relabeled": self._bboxes_relabeled,
            "bboxes_tagged": self._bboxes_tagged,
            "bboxes_stripped": self._bboxes_stripped,
            "bboxes_added": self._bboxes_added,
            "images_tagged": self._images_tagged,
            "images_discard_matched": self._images_discard_matched,
        }

    def _build_result(self) -> Dict[str, Any]:
        return {
            **self._build_resume_data(),
            "dry_run": self._dry_run,
        }

    async def _deinit(self) -> None:
        if not self._pin_ids:
            return
        inference_api_client = self._manager.get_inference_api_client()
        models_api = InferenceModelsApi(inference_api_client)
        for name, pin_id in self._pin_ids.items():
            try:
                await asyncio.to_thread(models_api.unpin_model, name, pin_id)
            except Exception:
                logger.exception("Failed to unpin model %s", name)

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "meta": {
                "order": [
                    "filter_chain_uuid",
                    "directory",
                    "image_uuids",
                    "tag_uuid",
                    "label_uuid",
                    "dataset_uuid",
                    "max_images",
                    "batch_size",
                    "concurrency",
                    "dry_run",
                ],
            },
            "filter_chain_uuid": {
                "type": "filter_chain_uuid",
                "required": True,
                "help": "UUID of the FilterChain to apply",
            },
            "directory": {
                "type": "directory",
                "label": "Directory Filter",
                "required": False,
                "description": "Scope: only images under this directory.",
            },
            "image_uuids": {
                "type": "list",
                "label": "Explicit Image UUIDs",
                "required": False,
                "description": "Scope: an explicit list of image UUIDs.",
            },
            "tag_uuid": {
                "type": "str",
                "label": "Tag Filter",
                "required": False,
                "description": "Scope: only images carrying this tag.",
            },
            "label_uuid": {
                "type": "str",
                "label": "Label Filter",
                "required": False,
                "description": "Scope: only images with a bbox of this label.",
            },
            "dataset_uuid": {
                "type": "str",
                "label": "Dataset Filter",
                "required": False,
                "description": "Scope: only images belonging to this dataset.",
            },
            "max_images": {
                "type": "int",
                "label": "Max Images",
                "default": _MAX_IMAGES_CEILING,
                "description": f"Hard cap on images processed (ceiling {_MAX_IMAGES_CEILING}).",
            },
            "batch_size": {
                "type": "int",
                "label": "Batch Size",
                "default": 32,
            },
            "concurrency": {
                "type": "int",
                "label": "Concurrency",
                "default": _DEFAULT_CONCURRENCY,
                "description": "Images processed in parallel. Higher keeps the GPU busier, up to what the inference server can absorb.",
            },
            "dry_run": {
                "type": "boolean",
                "label": "Dry Run",
                "default": False,
                "description": "Count matches and report without changing anything.",
            },
        }
