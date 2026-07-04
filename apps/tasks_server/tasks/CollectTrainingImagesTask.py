import asyncio
import base64
import logging
import os
import time
import uuid as uuid_module
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import cv2
import db_client
import inference_client
from db_client.api.filter_chains_api import FilterChainsApi
from db_client.api.images_api import ImagesApi
from db_client.api.labels_api import LabelsApi
from db_client.api.settings_api import SettingsApi
from db_client.api.tags_api import TagsApi
from db_client.models.add_tag_payload import AddTagPayload
from db_client.models.bounding_box_metadata_model import BoundingBoxMetadataModel
from db_client.models.set_image_tag_payload import SetImageTagPayload
from db_client.models.tag_kind import (  # noqa: F401 (used as ApiTagKind.TASK_GENERATED)
    TagKind as ApiTagKind,
)
from db_client.models.update_metadata_payload import UpdateMetadataPayload
from inference_client.api.inference_api import InferenceApi
from inference_client.api.models_api import ModelsApi
from inference_client.models.inference_payload import InferencePayload
from inference_client.models.pin_model_payload import PinModelPayload

from apps.helpers.db.db_client import UNVERIFIED_TAG_UUID
from apps.helpers.hashUtils import (
    PHASH_JPEG_QUALITY,
    compute_phashes_with_jpeg_roundtrip,
)
from apps.helpers.imageProviders.imageProvider import (
    ImageProvider,
    ImageProviderType,
    ImageWithProviderMetadata,
)
from apps.helpers.imageUtils import base64_encode_jpg

from ._filter_chain_runner import FilterChainRunner
from .ApplyFilterChainTask import (
    _CLASSIFIER_KINDS,
    _HttpInferenceBackend,
    _read_model_kind,
)
from .Registry import register_task
from .Task import Task
from .utils.imageHelper import ImageHelper
from .utils.labelHelper import build_label_ancestry_map
from .utils.sourceHelper import SourceHelper

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

# Two detections from different models are treated as the same physical object when
# their boxes overlap at or above this IoU; the higher-priority (earlier-configured)
# model's detection is kept. Kept as a module constant for now — may later be routed
# to the task-config UI.
SAME_BBOX_IOU_THRESHOLD = 0.65


def _bbox_iou(a, b) -> float:
    """Intersection-over-union of two bbox-like objects exposing x/y/width/height."""
    inter_w = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    inter_h = max(0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = (a.width * a.height) + (b.width * b.height) - inter
    return inter / union if union > 0.0 else 0.0


# _compute_jpeg_phashes has moved to apps.helpers.hashUtils as
# compute_phashes_with_jpeg_roundtrip, so the JPEG quality constant and the
# hash algorithm are shared with the DB server's phash pipeline.


@dataclass
class TargetLabel:
    label_uuid: str
    min_confidence: float


@register_task()
class CollectTrainingImagesTask(Task):
    # Per-call retry limits and timeouts for I/O operations that can hang indefinitely.
    _MAX_RETRIES: int = 3
    _FETCH_TIMEOUT_S: float = 30.0  # seconds to wait for a single getNextImage() call
    _INFERENCE_TIMEOUT_S: float = 60.0  # seconds per inference call (allows for model load)
    # Dedup pipeline buffer: allows this many matched frames to queue for dedup before
    # inference blocks.  A small buffer absorbs per-frame jitter without wasting memory
    # (each slot holds a numpy array reference, not a copy).
    _DEDUP_QUEUE_SIZE: int = 4

    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        self._status_msg: str = "Initializing"
        self.log("Initializing task parameters")
        self._source_uuids: list[str] = params["source_uuids"]
        self._output_dir: str = params["output_dir"]
        # output_dir is joined onto files_root when writing collected images.
        # Reject absolute paths and '..' traversal so a task can't write outside
        # the dataset root (raw params are untrusted — validate here per CLAUDE.md).
        _out = Path(self._output_dir)
        if _out.is_absolute() or ".." in _out.parts:
            raise ValueError(
                f"Invalid output_dir {self._output_dir!r}: must be a relative path within the dataset root"
            )

        # model_label_config: [{modelName, labelValues: {labelUuid: confidence}}]
        model_label_config: list[dict] = params.get("model_label_config", [])
        # Preserve config order: it doubles as model priority — earlier-configured
        # models win when two models produce overlapping (same-bbox) detections.
        ordered_model_names: list[str] = []
        # Flatten to label_uuid → min_confidence; skip zero-confidence (disabled in UI)
        label_conf: dict[str, float] = {}
        for item in model_label_config:
            name = item.get("modelName")
            if name and name not in ordered_model_names:
                ordered_model_names.append(name)
            for label_uuid, conf in item.get("labelValues", {}).items():
                conf_f = float(conf)
                if conf_f > 0:
                    label_conf[label_uuid] = min(label_conf.get(label_uuid, 1.0), conf_f)
        self._model_names: list[str] = ordered_model_names
        self._overlap_iou_threshold: float = float(params.get("overlap_iou_threshold", SAME_BBOX_IOU_THRESHOLD))

        self._target_labels: list[TargetLabel] = [
            TargetLabel(label_uuid=u, min_confidence=c) for u, c in label_conf.items()
        ]
        self._target_label_map: dict[str, float] = {t.label_uuid: t.min_confidence for t in self._target_labels}
        self._min_all_conf: float = min((t.min_confidence for t in self._target_labels), default=0.5)

        # Intervals for rate-limited provider types
        self._snapshot_interval: float = float(params.get("snapshot_interval", 15))
        self._stream_interval: float = float(params.get("stream_interval", 1.0))

        self._images_saved: int = resume_data.get("images_saved", 0) if resume_data else 0
        self._images_processed: int = resume_data.get("images_processed", 0) if resume_data else 0
        self._fixed_frames_processed: int = resume_data.get("fixed_frames_processed", 0) if resume_data else 0
        # Set of "source:frame_idx" strings for images already processed this run or
        # in a previous run that was paused.  Lets us skip re-processing on resume.
        self._processed_ids: set[str] = set(resume_data.get("processed_ids", [])) if resume_data else set()
        if self._processed_ids:
            self.log(f"Resuming: {len(self._processed_ids)} images already processed")

        # Tag params — resolved to UUIDs in _run() once API clients are available.
        self._apply_tag_uuids: list[str] = list(params.get("apply_tags", []))
        self._generate_task_tag: bool = bool(params.get("generate_task_tag", False))
        self._task_tag_name_override: str = params.get("task_tag_name", "") or ""
        # _config_name is injected by the manager (reserved key, not user-facing).
        self._config_name: str = params.get("_config_name", "")
        # Set in _run() when a TASK_GENERATED tag is created or found; used in result actions.
        self._task_tag_uuid: Optional[str] = None

        # Optional pre-save filter chain (WORKFLOW_FILTERCHAIN_PLAN.md Phase 4):
        # applied to each frame's in-memory detections before they're saved.
        self._filter_chain_uuid: Optional[str] = params.get("filter_chain_uuid") or None
        self._filter_chain_runner = None  # built in _run() once the chain is resolved
        self._filtered_discarded: int = resume_data.get("filtered_discarded", 0) if resume_data else 0
        self._filtered_relabeled: int = resume_data.get("filtered_relabeled", 0) if resume_data else 0
        self._filtered_tagged: int = resume_data.get("filtered_tagged", 0) if resume_data else 0
        self._filtered_stripped: int = resume_data.get("filtered_stripped", 0) if resume_data else 0

        # Deduplication params
        self._enable_dedup: bool = bool(params.get("enable_dedup", False))
        self._image_dup_threshold: int = int(params.get("image_dup_threshold", 5))
        self._bbox_dup_threshold: int = int(params.get("bbox_dup_threshold", 8))
        self._duplicate_action: str = params.get("duplicate_action", "discard")
        # Dedup outcome stats — persisted across pause/resume so totals are cumulative.
        self._dup_frames_discarded: int = resume_data.get("dup_frames_discarded", 0) if resume_data else 0
        # Finer-grained dedup counters: frames checked, and raw hit counts from the index
        # (independent of what action was taken — useful for diagnosing threshold issues).
        self._dedup_frames_checked: int = resume_data.get("dedup_frames_checked", 0) if resume_data else 0
        self._dedup_image_hits: int = resume_data.get("dedup_image_hits", 0) if resume_data else 0
        self._dedup_bbox_hits: int = resume_data.get("dedup_bbox_hits", 0) if resume_data else 0
        # Hash sizes are global settings (must match the index; changing them requires
        # recomputing all stored hashes). Read from DB settings, fall back to defaults.
        _settings_api = SettingsApi(self._manager.get_db_api_client())
        _labels_api = LabelsApi(self._manager.get_db_api_client())
        self._image_hash_size, self._bbox_hash_size, self._label_ancestry = await asyncio.gather(
            self._read_int_setting(_settings_api, "dedup_image_hash_size", 8),
            self._read_int_setting(_settings_api, "dedup_bbox_hash_size", 16),
            build_label_ancestry_map(_labels_api),
        )

    async def _run(self) -> dict[str, Any]:
        self._unverified_tag_uuid = UNVERIFIED_TAG_UUID

        db_api_client = self._manager.get_db_api_client()
        inference_api_client = self._manager.get_inference_api_client()

        # Resolve extra tags to apply to every saved bbox.
        self._extra_tag_uuids: list[str] = list(self._apply_tag_uuids)
        if self._generate_task_tag:
            task_tag_uuid = str(uuid_module.uuid5(uuid_module.NAMESPACE_OID, self._config_uuid))
            tag_name = self._task_tag_name_override or (
                f"[Task] {self._config_name}" if self._config_name else f"[Task] {self._config_uuid[:8]}"
            )
            tags_api = TagsApi(db_api_client)
            existing_tags = await asyncio.to_thread(tags_api.list_tags)
            existing_by_uuid = {t.uuid: t for t in (existing_tags.tags or [])}
            if task_tag_uuid not in existing_by_uuid:
                try:
                    await asyncio.to_thread(
                        tags_api.add_tag,
                        AddTagPayload(
                            name=tag_name,
                            kind=ApiTagKind.TASK_GENERATED,
                            uuid=task_tag_uuid,
                        ),
                    )
                    self.log(f"Created task tag: {tag_name} ({task_tag_uuid})")
                    self._extra_tag_uuids.append(task_tag_uuid)
                    self._task_tag_uuid = task_tag_uuid
                except Exception:
                    logger.exception(f"Failed to create task tag {tag_name}")
            else:
                self._extra_tag_uuids.append(task_tag_uuid)
                self._task_tag_uuid = task_tag_uuid

        self._status_msg = "Getting image providers"
        source_helper = SourceHelper(db_api_client)
        providers: list[ImageProvider] = []
        for source_uuid in self._source_uuids:
            try:
                provider = source_helper.get_image_provider(source_uuid)
            except ValueError:
                self.log(f"Failed to get provider for {source_uuid}")
                continue
            providers.append(provider)

        self._status_msg = "Pinning models"
        self.log(f"Pinning {len(self._model_names)} model(s)")
        models_api = ModelsApi(inference_api_client)
        pin_ids: dict[str, Optional[str]] = {}
        for name in self._model_names:
            try:
                resp = await asyncio.to_thread(
                    models_api.pin_model_api_api_models_model_name_pin_post,
                    name,
                    PinModelPayload(model_name=name, duration=3600),
                )
                if resp.pin_id:
                    pin_ids[name] = resp.pin_id
                    self.log(f"Pinned model: {name} (pin_id={resp.pin_id})")
                else:
                    # Server returned failure status without raising — pin_id will be None,
                    # causing the model to reload on every inference call.
                    logger.warning(
                        f"Pin request for '{name}' returned no pin_id: status={resp.status} message={resp.message}"
                    )
                    self.log(f"Warning: could not pin model '{name}' — will reload each inference")
                    pin_ids[name] = None
            except Exception:
                logger.exception(f"Failed to pin model {name}")
                self.log(f"Failed to pin model: {name}")
                pin_ids[name] = None

        self._chain_pin_ids: dict[str, str] = {}
        if self._filter_chain_uuid:
            self._status_msg = "Resolving filter chain"
            filter_chains_api = FilterChainsApi(db_api_client)
            chain_resp = await asyncio.to_thread(filter_chains_api.get_filter_chain, self._filter_chain_uuid)
            chain = getattr(chain_resp, "chain", None)
            if chain is None:
                raise ValueError(f"Filter chain not found: {self._filter_chain_uuid}")
            chain_steps = [
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
            if chain_steps:
                # Reuse a model already pinned by model_label_config where possible;
                # only pin the chain's own detector-step models otherwise.
                for name in sorted({s["model_name"] for s in chain_steps}):
                    if pin_ids.get(name):
                        continue
                    try:
                        kind = await asyncio.to_thread(_read_model_kind, name)
                    except ValueError:
                        logger.exception("Invalid filter-chain step model_name %r", name)
                        continue
                    if kind in _CLASSIFIER_KINDS:
                        continue  # /classify auto-pins per call; no long-lived pin needed
                    try:
                        resp = await asyncio.to_thread(
                            models_api.pin_model_api_api_models_model_name_pin_post,
                            name,
                            PinModelPayload(model_name=name, duration=3600),
                        )
                        if resp.pin_id:
                            self._chain_pin_ids[name] = resp.pin_id
                    except Exception:
                        logger.exception("Failed to pin filter-chain model %s", name)

                combined_pin_ids = {**pin_ids, **self._chain_pin_ids}
                backend = _HttpInferenceBackend(inference_api_client, combined_pin_ids)
                self._filter_chain_runner = FilterChainRunner(chain_steps, backend)
                self.log(f"Filter chain '{chain.name}' active ({len(chain_steps)} step(s))")

        images_api = ImagesApi(db_api_client)
        inference_api = InferenceApi(inference_api_client)
        files_root = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))

        # Shared save-task list: appended by _dedup_worker (dedup path) or directly here
        # (no-dedup path), harvested by _collect_saves() on each main-loop iteration.
        self._pending_saves: list[tuple[asyncio.Task, Optional[str]]] = []

        # Separate out FIXED providers so we can compute real progress from their frame counts.
        # For LIVE_SNAPSHOT and LIVE_STREAM providers we track last-fetch time for rate limiting.
        fixed_providers: list[ImageProvider] = [p for p in providers if p.provider_type() == ImageProviderType.FIXED]
        last_fetch: dict[int, float] = {}  # id(provider) → perf_counter time of last successful fetch
        _rate_limits: dict[ImageProviderType, float] = {
            ImageProviderType.LIVE_SNAPSHOT: self._snapshot_interval,
            ImageProviderType.LIVE_STREAM: self._stream_interval,
        }

        def _status_msg_str() -> str:
            # Switch to generic message once all fixed providers are exhausted.
            if fixed_providers and any(p in active for p in fixed_providers):
                total = sum(p.get_total_frames() or 0 for p in fixed_providers)
                return f"Processed {self._fixed_frames_processed}/{total} frames, Saved {self._images_saved} images"
            return f"Processed {self._images_processed} frames, Saved {self._images_saved} images"

        def _dup_stats_str() -> str:
            if not self._enable_dedup:
                return ""
            parts = [
                f"{self._dedup_frames_checked} checked",
                f"{self._dedup_image_hits} img hits",
                f"{self._dedup_bbox_hits} bbox hits",
            ]
            if self._dup_frames_discarded:
                parts.append(f"{self._dup_frames_discarded} discarded")
            queue_depth = self._dedup_queue.qsize()
            if queue_depth > 0:
                parts.append(f"{queue_depth} queued")
            return ", dedup: " + ", ".join(parts)

        def _full_status() -> str:
            return _status_msg_str() + _dup_stats_str()

        def _collect_saves() -> None:
            """Harvest results from any completed save tasks and update counters."""
            still_pending = []
            for task, uid in self._pending_saves:
                if task.done():
                    try:
                        filename = task.result()
                    except Exception:
                        logger.exception("Background save task raised an exception")
                        continue
                    if filename is None:
                        continue
                    if uid:
                        self._processed_ids.add(uid)
                    self._images_saved += 1
                    self.log(f"Saved {filename} ({self._images_saved} total)")
                    # Update progress for FIXED providers based on real frame counts.
                    if fixed_providers:
                        total = sum(p.get_total_frames() or 0 for p in fixed_providers)
                        if total > 0:
                            self._update_progress(min(100.0, self._fixed_frames_processed / total * 100.0))
                else:
                    still_pending.append((task, uid))
            self._pending_saves.clear()
            self._pending_saves.extend(still_pending)

        async def _drain_saves() -> None:
            """Wait for all in-flight saves then collect their results."""
            if self._pending_saves:
                await asyncio.gather(*(t for t, _ in self._pending_saves), return_exceptions=True)
            _collect_saves()

        self._status_msg = "Running"
        self.log(f"Starting collection (resuming from {self._images_saved} images)")
        if self._enable_dedup:
            self.log(
                f"Dedup enabled: action={self._duplicate_action}, "
                f"img_threshold={self._image_dup_threshold}, bbox_threshold={self._bbox_dup_threshold}, "
                f"hash_sizes={self._image_hash_size}x{self._bbox_hash_size}"
            )
        _last_heartbeat: float = 0.0
        # PERF BEGIN — rolling per-frame timing stats (bounded window to cap memory)
        _PERF_SUMMARY_INTERVAL = 50
        _PERF_WINDOW = 200
        _perf_fetch_ms: deque = deque(maxlen=_PERF_WINDOW)
        _perf_encode_ms: deque = deque(maxlen=_PERF_WINDOW)
        _perf_infer_ms: deque = deque(maxlen=_PERF_WINDOW)
        _perf_frame_count: int = 0

        def _ms_stats(lst) -> str:
            n = len(lst)
            return f"min={min(lst):.0f} avg={sum(lst)/n:.0f} max={max(lst):.0f}" if n else "n/a"

        # PERF END
        self._meta_write_lock = asyncio.Lock()

        for provider in providers:
            # Enable pre-encoding in the prefetch worker so encode cost is pipelined
            # with the previous frame's inference rather than sitting on the critical path.
            if hasattr(provider, "pre_encode"):
                provider.pre_encode = True
            await provider.start()

        # Dedup coroutine: started only when dedup is enabled.  Receives matched frames
        # from the main loop via a bounded queue (maxsize=1) and pipelines dedup one
        # frame behind inference so throughput ≈ max(inference_time, dedup_time).
        dedup_task: Optional[asyncio.Task] = None
        if self._enable_dedup:
            self._dedup_queue: asyncio.Queue = asyncio.Queue(maxsize=self._DEDUP_QUEUE_SIZE)
            dedup_task = asyncio.create_task(self._dedup_worker(images_api, files_root))

        try:
            active = set(providers)
            while active and not self._cancel_flag.is_set():
                await self._check_pause()
                _collect_saves()
                self._status_msg = _full_status()

                any_fetched = False
                for provider in list(active):
                    if self._cancel_flag.is_set():
                        break

                    # Rate-limit LIVE providers: skip until enough time has elapsed.
                    ptype = provider.provider_type()
                    if (interval := _rate_limits.get(ptype)) is not None:
                        if time.perf_counter() - last_fetch.get(id(provider), 0) < interval:
                            continue

                    t0 = time.perf_counter()  # PERF
                    provided = None
                    for _attempt in range(self._MAX_RETRIES):
                        try:
                            provided = await asyncio.wait_for(provider.getNextImage(), timeout=self._FETCH_TIMEOUT_S)
                            break
                        except asyncio.TimeoutError:
                            if _attempt < self._MAX_RETRIES - 1:
                                logger.warning(
                                    "Fetch timeout on attempt %d/%d, retrying",
                                    _attempt + 1,
                                    self._MAX_RETRIES,
                                )
                            else:
                                self.log(f"Provider timed out {self._MAX_RETRIES} times — removing from active sources")
                                logger.warning(
                                    "Removing provider after %d consecutive fetch timeouts",
                                    self._MAX_RETRIES,
                                )
                    t_fetch = time.perf_counter()
                    if provided is None:
                        active.discard(provider)
                        continue

                    any_fetched = True
                    last_fetch[id(provider)] = t_fetch

                    # Build a stable unique ID when the provider gives us source + frame_idx.
                    # Used to skip already-processed frames on resume (file-based providers only).
                    unique_id: Optional[str] = None
                    if provided.metadata.source is not None and provided.metadata.frame_idx is not None:
                        unique_id = f"{provided.metadata.source}:{provided.metadata.frame_idx}"
                    if unique_id and unique_id in self._processed_ids:
                        logger.debug(f"Skipping already-processed: {unique_id}")
                        continue

                    self._images_processed += 1
                    if ptype == ImageProviderType.FIXED:
                        self._fixed_frames_processed += 1

                    # Use pre-encoded data from the prefetch worker when available;
                    # otherwise encode here (on the critical path).
                    _pre_encoded = provided.encoded_b64 is not None
                    if _pre_encoded:
                        base64_img = provided.encoded_b64
                        t_encode = time.perf_counter()  # PERF
                    else:
                        base64_img = await asyncio.to_thread(base64_encode_jpg, provided.image)
                        t_encode = time.perf_counter()  # PERF

                    matching = await self._run_inference(inference_api, base64_img, pin_ids)
                    force_save = False
                    if self._filter_chain_runner is not None:
                        matching, _discarded, force_save = await self._apply_filter_chain_to_frame(
                            provided, base64_img, matching
                        )
                    t_infer = time.perf_counter()  # PERF

                    # PERF BEGIN
                    _fetch_ms = 1000 * (t_fetch - t0)
                    _encode_ms = 1000 * (t_encode - t_fetch)
                    _infer_ms = 1000 * (t_infer - t_encode)
                    _perf_fetch_ms.append(_fetch_ms)
                    _perf_encode_ms.append(_encode_ms)
                    _perf_infer_ms.append(_infer_ms)
                    _perf_frame_count += 1
                    _enc_label = "pre_enc" if _pre_encoded else "on_the_fly"
                    logger.debug(
                        "Frame timing (%s): fetch=%.1fms encode=%.1fms(%s) infer=%.1fms%s",
                        "matched" if matching else "no match",
                        _fetch_ms,
                        _encode_ms,
                        _enc_label,
                        _infer_ms,
                        (
                            " (dedup queued)"
                            if (matching and self._enable_dedup)
                            else " (save dispatched)" if matching else ""
                        ),
                    )
                    if _perf_frame_count % _PERF_SUMMARY_INTERVAL == 0:
                        logger.info(
                            "[pipeline summary] frames=%d fetch=%sms encode=%sms(%s) infer=%sms",
                            _perf_frame_count,
                            _ms_stats(_perf_fetch_ms),
                            _ms_stats(_perf_encode_ms),
                            _enc_label,
                            _ms_stats(_perf_infer_ms),
                        )
                    # PERF END

                    if matching:
                        if self._enable_dedup:
                            # Hand off to the dedup coroutine.  This may block briefly if
                            # the worker is still processing the previous frame (queue
                            # maxsize=1), providing natural backpressure.
                            await self._dedup_queue.put((provided, matching))
                        else:
                            save_task = asyncio.create_task(
                                self._save_image(images_api, provided, matching, files_root)
                            )
                            self._pending_saves.append((save_task, unique_id))
                    elif force_save and not self._enable_dedup:
                        # strip_unverified_bboxes emptied `matching`, but per plan 2.1 the
                        # image itself must still be saved (never delete images). Skipped
                        # when dedup is enabled: an empty bbox list makes the dedup worker's
                        # `all(bbox_flags)` vacuously true, which would wrongly discard the
                        # frame under duplicate_action="discard" — narrow edge case, not
                        # worth the complexity of special-casing dedup for an empty bbox set.
                        save_task = asyncio.create_task(self._save_image(images_api, provided, [], files_root))
                        self._pending_saves.append((save_task, unique_id))

                    if self._data_req_flag.is_set():
                        # Drain the dedup queue first so all matched frames are accounted
                        # for in the resume snapshot (join() waits for task_done() in worker).
                        if self._enable_dedup:
                            await self._dedup_queue.join()
                        await _drain_saves()
                        self._data_req_flag.clear()
                        self._resume_data = self._build_resume_data()
                        self._data_ready_flag.set()

                    # Yield to the event loop between frames so HTTP handlers get cycles
                    # even when providers deliver images as fast as they can be processed.
                    await asyncio.sleep(0)

                # If all remaining active providers were rate-limited this iteration,
                # yield briefly to avoid busy-spinning.
                if not any_fetched:
                    await asyncio.sleep(0.05)

                # Heartbeat: log current status every 30 s so the task doesn't look stuck
                # when providers deliver frames but none match the detection criteria.
                _now = time.perf_counter()
                if _now - _last_heartbeat >= 30.0:
                    _last_heartbeat = _now
                    self.log(_full_status())

        finally:
            # Shut down the dedup coroutine cleanly before draining saves so that any
            # frames still queued for dedup get processed and their save tasks are in
            # _pending_saves before we wait on them.
            if dedup_task is not None:
                await self._dedup_queue.put(None)  # sentinel
                await dedup_task
            await _drain_saves()
            for provider in providers:
                await provider.stop()
            # Release long-lived model pins so the inference server can unload promptly.
            unpin_api = ModelsApi(inference_api_client)
            for name, pid in pin_ids.items():
                if pid:
                    try:
                        await asyncio.to_thread(unpin_api.unpin_model, name, pid)
                        self.log(f"Unpinned model: {name}")
                    except Exception:
                        logger.warning(f"Failed to unpin model '{name}' (pin {pid})")
            for name, pid in self._chain_pin_ids.items():
                try:
                    await asyncio.to_thread(unpin_api.unpin_model, name, pid)
                except Exception:
                    logger.warning(f"Failed to unpin filter-chain model '{name}' (pin {pid})")

        self._resume_data = self._build_resume_data()
        self._data_ready_flag.set()
        # PERF BEGIN
        if _perf_frame_count:
            logger.info(
                "[pipeline final] frames=%d fetch=%sms encode=%sms infer=%sms",
                _perf_frame_count,
                _ms_stats(_perf_fetch_ms),
                _ms_stats(_perf_encode_ms),
                _ms_stats(_perf_infer_ms),
            )
        # PERF END
        self.log(
            f"Complete: {self._images_processed} seen, {self._images_saved} saved"
            + (
                f", dedup: {self._dedup_frames_checked} checked"
                f", {self._dedup_image_hits} img hits"
                f", {self._dedup_bbox_hits} bbox hits"
                f", {self._dup_frames_discarded} discarded"
                if self._enable_dedup
                else ""
            )
            + (
                f", filter chain: {self._filtered_discarded} discarded"
                f", {self._filtered_relabeled} relabeled"
                f", {self._filtered_tagged} tagged"
                f", {self._filtered_stripped} stripped"
                if self._filter_chain_runner is not None
                else ""
            )
        )
        result: dict[str, Any] = {"images_saved": self._images_saved}
        if self._enable_dedup:
            result["dedup_frames_checked"] = self._dedup_frames_checked
            result["dedup_image_hits"] = self._dedup_image_hits
            result["dedup_bbox_hits"] = self._dedup_bbox_hits
            result["dup_frames_discarded"] = self._dup_frames_discarded
        if self._filter_chain_uuid:
            result["filtered_discarded"] = self._filtered_discarded
            result["filtered_relabeled"] = self._filtered_relabeled
            result["filtered_tagged"] = self._filtered_tagged
            result["filtered_stripped"] = self._filtered_stripped
        if self._task_tag_uuid:
            result["actions"] = [
                {
                    "type": "open_review",
                    "label": "See results",
                    "payload": {"tag_uuids": [self._task_tag_uuid]},
                }
            ]
        return result

    def _build_resume_data(self) -> dict[str, Any]:
        return {
            "images_saved": self._images_saved,
            "images_processed": self._images_processed,
            "fixed_frames_processed": self._fixed_frames_processed,
            "processed_ids": list(self._processed_ids),
            "dup_frames_discarded": self._dup_frames_discarded,
            "dedup_frames_checked": self._dedup_frames_checked,
            "dedup_image_hits": self._dedup_image_hits,
            "dedup_bbox_hits": self._dedup_bbox_hits,
            "filtered_discarded": self._filtered_discarded,
            "filtered_relabeled": self._filtered_relabeled,
            "filtered_tagged": self._filtered_tagged,
            "filtered_stripped": self._filtered_stripped,
        }

    async def _run_inference(
        self,
        inference_api: InferenceApi,
        base64_img: str,
        pin_ids: dict[str, Optional[str]],
    ) -> list:
        """Runs all configured models in parallel; returns detections meeting per-label confidence thresholds."""

        async def _run_model(name: str) -> list:
            """Call one model with retries; return its filtered detections."""
            for _attempt in range(self._MAX_RETRIES):
                try:
                    t_start = time.perf_counter()  # PERF
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            inference_api.inference,
                            name,
                            InferencePayload(
                                image_base64=base64_img,
                                conf_thresh=self._min_all_conf,
                                pin_id=pin_ids.get(name),
                            ),
                        ),
                        timeout=self._INFERENCE_TIMEOUT_S,
                    )
                    # PERF BEGIN
                    logger.debug(
                        "Inference HTTP round-trip model='%s': %.1fms (%d detections) "
                        "[compare to processImage total= in inference_server log for HTTP overhead]",
                        name,
                        1000 * (time.perf_counter() - t_start),
                        len(result.result.detections or []),
                    )
                    # PERF END
                    dets = []
                    for det in result.result.detections or []:
                        luuid = det.bbox.label_uuid
                        if luuid and luuid in self._target_label_map:
                            if det.confidence >= self._target_label_map[luuid]:
                                dets.append(det)
                    return dets
                except asyncio.TimeoutError:
                    if _attempt < self._MAX_RETRIES - 1:
                        logger.warning(
                            "Inference timeout on model '%s' (attempt %d/%d), retrying",
                            name,
                            _attempt + 1,
                            self._MAX_RETRIES,
                        )
                    else:
                        logger.warning(
                            "Inference timed out %d times for model '%s', skipping frame",
                            self._MAX_RETRIES,
                            name,
                        )
                        self.log(f"Inference timeout on '{name}' after {self._MAX_RETRIES} retries — skipping frame")
                except Exception:
                    logger.exception(f"Inference error on model {name}")
                    break  # don't retry on non-timeout exceptions
            return []

        # Run all models concurrently.  asyncio.gather preserves input order, so
        # results_per_model[i] corresponds to self._model_names[i] (priority order).
        # return_exceptions=True ensures that a CancelledError or unexpected
        # BaseException in one coroutine doesn't discard results from sibling models.
        raw_results = await asyncio.gather(
            *(_run_model(name) for name in self._model_names),
            return_exceptions=True,
        )

        # _run_model catches all Exception internally; only BaseExceptions (e.g.
        # CancelledError) can appear here.  Re-raise CancelledError so the outer
        # task cancellation propagates correctly; treat anything else as empty.
        results_per_model: list[list] = []
        for r in raw_results:
            if isinstance(r, asyncio.CancelledError):
                raise r
            results_per_model.append(r if isinstance(r, list) else [])

        # Flatten in priority order then apply cross-model hierarchy-aware IoU dedup.
        matching = [det for dets in results_per_model for det in dets]

        # For each detection in model-priority order, compare against already-kept boxes:
        #   same label_uuid       → drop the lower-priority (later) detection
        #   det label is ancestor of kept label → kept is more specific → drop det
        #   kept label is ancestor of det label → det is more specific → replace kept
        #   unrelated labels      → keep both (different objects)
        deduped: list = []
        for det in matching:
            det_label = det.bbox.label_uuid
            dominated = False
            to_supersede: list[int] = []
            for i, kept in enumerate(deduped):
                if _bbox_iou(det.bbox, kept.bbox) < self._overlap_iou_threshold:
                    continue
                kept_label = kept.bbox.label_uuid
                if det_label == kept_label:
                    dominated = True
                    break
                elif det_label and det_label in self._label_ancestry.get(kept_label, ()):
                    # det's label is an ancestor of kept's → kept is more specific → drop det
                    dominated = True
                    break
                elif kept_label and kept_label in self._label_ancestry.get(det_label, ()):
                    # kept's label is an ancestor of det's → det is more specific → supersede kept
                    to_supersede.append(i)
                # else: unrelated labels — keep both
            if not dominated:
                for i in reversed(to_supersede):
                    deduped.pop(i)
                deduped.append(det)
        return deduped

    async def _apply_filter_chain_to_frame(
        self, provided: ImageWithProviderMetadata, base64_img: str, matching: list
    ) -> tuple[list, bool, bool]:
        """Run the configured filter chain against this frame's in-memory detections
        before they're saved (WORKFLOW_FILTERCHAIN_PLAN.md Phase 4).

        Returns (possibly-mutated matching list, discard, force_save):
          - discard=True: the whole frame must not be saved at all.
          - force_save=True: `matching` was stripped to [] but the plan requires the
            image itself to still be saved (never delete images) — caller must
            dispatch even though `matching` is empty.

        Bbox-level actions (relabel_bbox / tag_bbox) mutate `matching` in place by
        overlap-matching a chain detector step's new detections against `matching`
        itself (there are no "existing" DB bboxes yet at collect time). `tag_image`
        is an image-level tag — stashed on `provided` and applied via set_image_tag
        once _save_image knows the image's saved path, so it lands on the image
        record rather than being (incorrectly) folded into every bbox's tags.
        """
        if self._filter_chain_runner is None or not matching:
            return matching, False, False

        image_bytes = base64.b64decode(base64_img)
        existing = [
            {
                "uuid": str(i),
                "x": det.bbox.x,
                "y": det.bbox.y,
                "width": det.bbox.width,
                "height": det.bbox.height,
                "label_uuid": det.bbox.label_uuid,
            }
            for i, det in enumerate(matching)
        ]
        try:
            step_matches = await self._filter_chain_runner.evaluate_image(image_bytes, existing)
        except Exception:
            logger.exception("Filter chain evaluation failed for this frame; skipping filter")
            return matching, False, False

        discard = False
        strip_all = False
        image_tag_uuids: set = set()
        for match in step_matches:
            step = match.step
            action = step["action"]
            action_params = step.get("action_params") or {}
            if match.image_match:
                if action == "discard_image":
                    discard = True
                elif action == "tag_image":
                    tag_uuid = action_params.get("tag_uuid")
                    if tag_uuid and tag_uuid not in image_tag_uuids:
                        image_tag_uuids.add(tag_uuid)
                        self._filtered_tagged += 1
                elif action == "strip_unverified_bboxes":
                    # At collect time every detection is unverified by definition —
                    # strip them all, but per plan 2.1 the image itself is still saved
                    # (never delete images, even ones with zero remaining bboxes).
                    strip_all = True
                    self._filtered_stripped += len(matching)
                else:
                    logger.warning("Action %r is not valid for an image-level (classifier) step; skipping", action)
                continue

            for uuid_str, _det_match in match.bbox_matches.items():
                idx = int(uuid_str)
                if action == "relabel_bbox":
                    target = action_params.get("target_label_uuid")
                    if target:
                        matching[idx].bbox.label_uuid = target
                        self._filtered_relabeled += 1
                elif action == "tag_bbox":
                    tag_uuid = action_params.get("tag_uuid")
                    if tag_uuid:
                        matching[idx].bbox.tag_uuids = [*(matching[idx].bbox.tag_uuids or []), tag_uuid]
                        self._filtered_tagged += 1
                else:
                    logger.warning("Action %r is not valid for a bbox-level (detector) step; skipping", action)

        if image_tag_uuids:
            provided._filter_chain_image_tags = image_tag_uuids

        if discard:
            self._filtered_discarded += 1
            return [], True, False
        if strip_all:
            return [], False, True
        return matching, False, False

    async def _save_image(
        self,
        images_api: ImagesApi,
        provided: ImageWithProviderMetadata,
        matching: list,
        files_root: Path,
    ) -> Optional[str]:
        """Writes image to disk and persists bounding-box metadata. Returns filename on success, None on failure."""
        t_save_start = time.perf_counter()  # PERF
        source_name = ImageHelper.sanitize_filename(provided.metadata.source or "unknown")
        frame_idx = provided.metadata.frame_idx or 0
        filename = f"{source_name}_{frame_idx}.jpg"
        rel_path = str(Path(self._output_dir) / filename)
        abs_path = files_root / self._output_dir / filename
        await asyncio.to_thread(abs_path.parent.mkdir, parents=True, exist_ok=True)

        # cv2.imwrite is CPU-bound + I/O; run off the event loop.
        # Save as JPEG (faster write, smaller files; lossless PNG not needed for training data).
        t_write = time.perf_counter()  # PERF
        if not await asyncio.to_thread(
            cv2.imwrite,
            str(abs_path),
            provided.image,
            [cv2.IMWRITE_JPEG_QUALITY, PHASH_JPEG_QUALITY],
        ):
            logger.warning(f"Failed to write image to {abs_path}")
            self.log(f"Failed to write image: {filename}")
            return None
        logger.debug("Image write: %.1fms → %s", 1000 * (time.perf_counter() - t_write), abs_path)  # PERF

        boxes = []
        for det in matching:
            extra_tags = list(self._extra_tag_uuids)
            # Filter-chain tag_bbox matches (Phase 4) stash extra tags directly on
            # the matched detection's own bbox.tag_uuids — fold them in here.
            # (tag_image is image-level — applied via set_image_tag below instead.)
            filter_chain_tags = list(det.bbox.tag_uuids or [])
            boxes.append(
                BoundingBoxMetadataModel(
                    bbox_uuid=str(uuid4()),
                    x=det.bbox.x,
                    y=det.bbox.y,
                    width=det.bbox.width,
                    height=det.bbox.height,
                    label_uuid=det.bbox.label_uuid,
                    class_str=det.bbox.class_str,
                    tag_uuids=[self._unverified_tag_uuid, *extra_tags, *filter_chain_tags],
                    extra={"confidence": str(det.confidence)},
                )
            )

        try:
            t_meta = time.perf_counter()  # PERF
            # Serialise DB writes to avoid SQLite write-lock contention between concurrent saves.
            async with self._meta_write_lock:
                await asyncio.to_thread(
                    images_api.update_image_metadata,
                    UpdateMetadataPayload(image_path=rel_path, boxes=boxes),
                )
            logger.debug(
                "Metadata save: %.1fms for %s",
                1000 * (time.perf_counter() - t_meta),
                filename,
            )  # PERF
        except Exception:
            logger.exception(f"Failed to save metadata for {rel_path}")
            self.log(f"Failed to save metadata: {filename}")
            return None

        # Image-level filter-chain tag_image match, stashed by
        # _apply_filter_chain_to_frame — apply now that the image is saved.
        image_tag_uuids = getattr(provided, "_filter_chain_image_tags", None)
        if image_tag_uuids:
            for tag_uuid in image_tag_uuids:
                try:
                    await asyncio.to_thread(
                        images_api.set_image_tag,
                        SetImageTagPayload(image_path=rel_path, tag_uuid=tag_uuid, action="add"),
                    )
                except Exception:
                    logger.exception(f"Failed to apply filter-chain image tag {tag_uuid} to {rel_path}")

        logger.debug(
            "_save_image total: %.1fms for %s",
            1000 * (time.perf_counter() - t_save_start),
            filename,
        )  # PERF
        return filename

    async def _dedup_worker(
        self,
        images_api: ImagesApi,
        files_root: Path,
    ) -> None:
        """Consume matched frames one at a time, run bulk dedup, dispatch saves.

        Runs as a single background coroutine so dedup is serialised (no thread-pool
        explosion) while still pipelining one frame behind the main loop's inference.
        Shuts down when it receives None from _dedup_queue.
        """
        from db_client.models.bbox_dedup_item import BboxDedupItem
        from db_client.models.check_dedup_bulk_payload import CheckDedupBulkPayload

        while True:
            item = await self._dedup_queue.get()
            try:
                if item is None:
                    break  # sentinel — main loop is done

                provided, matching = item

                # When action is 'ignore', skip dedup entirely and save without any tagging.
                if self._duplicate_action == "ignore":
                    unique_id = None
                    if provided.metadata.source is not None and provided.metadata.frame_idx is not None:
                        unique_id = f"{provided.metadata.source}:{provided.metadata.frame_idx}"
                    save_task = asyncio.create_task(self._save_image(images_api, provided, matching, files_root))
                    self._pending_saves.append((save_task, unique_id))
                    continue

                # Compute image and bbox phashes via JPEG round-trip so they match the
                # phashes stored in the BK-tree (which are derived from the saved JPEG file).
                # Uses compute_phashes_with_jpeg_roundtrip from hashUtils, the same helper
                # the DB server uses, so the JPEG quality is guaranteed to match.
                # PERF
                t_bbox_hash = time.perf_counter()
                # /PERF
                bboxes_xywh = [(d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height) for d in matching]
                image_phash, bbox_phashes = await asyncio.to_thread(
                    compute_phashes_with_jpeg_roundtrip,
                    provided.image,
                    bboxes_xywh,
                    self._image_hash_size,
                    self._bbox_hash_size,
                )

                # Single bulk HTTP call — replaces N+1 separate calls.
                # PERF
                t_http = time.perf_counter()
                # /PERF
                resp = await asyncio.to_thread(
                    images_api.check_dedup_bulk,
                    CheckDedupBulkPayload(
                        image_phash=image_phash,
                        image_threshold=self._image_dup_threshold,
                        bboxes=[
                            BboxDedupItem(
                                phash=bp,
                                label_uuid=det.bbox.label_uuid,
                                threshold=self._bbox_dup_threshold,
                            )
                            for det, bp in zip(matching, bbox_phashes)
                        ],
                    ),
                )
                # PERF
                t_done = time.perf_counter()
                logger.debug(
                    "dedup bulk: bbox_hash=%.1fms http=%.1fms total=%.1fms image_is_dup=%s",
                    1000 * (t_http - t_bbox_hash),
                    1000 * (t_done - t_http),
                    1000 * (t_done - t_bbox_hash),
                    resp.image_is_dup,
                )
                # /PERF

                # bbox_flags: per-detection duplicate flags (pure bbox similarity, image-independent).
                bbox_flags: list[bool] = list(resp.bbox_flags) if resp.bbox_flags else [False] * len(matching)

                # Track raw hit counts regardless of what action is taken — these are the
                # primary diagnostic for "why isn't dedup discarding anything?"
                self._dedup_frames_checked += 1
                if resp.image_is_dup:
                    self._dedup_image_hits += 1
                # Raw bbox index hits (image-independent) — diagnostic only; not a count of
                # bboxes that were tagged/discarded, which requires image_is_dup AND bbox_flag.
                self._dedup_bbox_hits += sum(1 for f in bbox_flags if f)
                logger.debug(
                    "dedup result: image_is_dup=%s bbox_flags=%s (img_thresh=%d bbox_thresh=%d)",
                    resp.image_is_dup,
                    bbox_flags,
                    self._image_dup_threshold,
                    self._bbox_dup_threshold,
                )

                # Image-level discard: skip the frame when the whole scene is a perceptual duplicate.
                if self._duplicate_action == "discard_image" and resp.image_is_dup:
                    self._dup_frames_discarded += 1
                    logger.debug("Discarding image-duplicate frame: %s", provided.metadata.source)
                    if self._dup_frames_discarded == 1:
                        self.log("Dedup: discarding first duplicate frame")
                    elif self._dup_frames_discarded % 50 == 0:
                        self.log(f"Dedup: {self._dup_frames_discarded} duplicate frames discarded so far")
                    continue

                # Bbox-level discard: keep if any detection is new; discard only if all are duplicates.
                # Gate on image_is_dup too — bbox hashes can collide across unrelated scenes
                # (same logic as the 'tag' action below, which uses image_is_dup AND bbox_flag).
                if self._duplicate_action == "discard" and resp.image_is_dup and all(bbox_flags):
                    self._dup_frames_discarded += 1
                    logger.debug("Discarding all-duplicate frame: %s", provided.metadata.source)
                    if self._dup_frames_discarded == 1:
                        self.log("Dedup: discarding first all-duplicate frame")
                    elif self._dup_frames_discarded % 50 == 0:
                        self.log(f"Dedup: {self._dup_frames_discarded} duplicate frames discarded so far")
                    continue

                save_task = asyncio.create_task(self._save_image(images_api, provided, matching, files_root))
                unique_id: Optional[str] = None
                if provided.metadata.source is not None and provided.metadata.frame_idx is not None:
                    unique_id = f"{provided.metadata.source}:{provided.metadata.frame_idx}"
                self._pending_saves.append((save_task, unique_id))
                # Must wait for the save to complete before checking the next frame:
                # update_image_metadata adds the phash to the in-memory BK-tree index, and
                # check_dedup_bulk reads from that same index — if we move on before the save
                # finishes, every subsequent frame sees an empty/stale index and is never
                # flagged as a duplicate.
                await save_task

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("_dedup_worker: unexpected error processing frame")
            finally:
                self._dedup_queue.task_done()

    async def _deinit(self) -> None:
        pass

    def get_status_message(self) -> str:
        return self._status_msg

    def get_status_actions(self) -> list[dict]:
        if self._task_tag_uuid:
            return [
                {
                    "type": "open_review",
                    "label": "Browse results",
                    "payload": {"tag_uuids": [self._task_tag_uuid]},
                }
            ]
        return []

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        return {
            "meta": {
                "order": [
                    "output_dir",
                    "snapshot_interval",
                    "stream_interval",
                    "source_uuids",
                    "model_label_config",
                    "overlap_iou_threshold",
                    "apply_tags",
                    "generate_task_tag",
                    "task_tag_name",
                    "filter_chain_uuid",
                    "dedup_group",
                ]
            },
            "snapshot_interval": {
                "type": "int",
                "label": "Snapshot Interval (s)",
                "description": "Minimum seconds between fetches from snapshot sources (e.g. URL cameras)",
                "default": 15,
            },
            "stream_interval": {
                "type": "int",
                "label": "Stream Min Interval (s)",
                "description": "Minimum seconds between frames collected from live stream sources (e.g. RTSP)",
                "default": 1,
            },
            "source_uuids": {
                "type": "array",
                "required": True,
                "label": "Image Sources",
                "items": {
                    "type": "source_uuid",
                    "label": "Image Source",
                    "required": True,
                    "description": "Pre-configured source to pull images from",
                },
            },
            "output_dir": {
                "type": "directory",
                "label": "Output Directory",
                "required": True,
                "description": "Directory relative to image dataset root where images will be saved",
            },
            "model_label_config": {
                "type": "array",
                "required": "true",
                "label": "Model/Label Configuration",
                "items": {
                    "type": "model_label",
                    "label": "Model/Label selection",
                    "required": True,
                    "description": "Labels to collect",
                },
            },
            "overlap_iou_threshold": {
                "type": "float",
                "label": "Multi-Model Overlap IoU Threshold",
                "description": (
                    "When multiple models detect the same region, boxes from later-configured models are discarded "
                    "if they overlap a higher-priority model's box at or above this IoU value (0–1). "
                    "Lower values discard more aggressively; higher values allow more overlap before deduplication."
                ),
                "default": SAME_BBOX_IOU_THRESHOLD,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            },
            "apply_tags": {
                "type": "tags",
                "label": "Apply Tags",
                "description": "Tags to apply to every bounding box in each saved image.",
                "default": [],
                "allow_add_tag": True,
            },
            "generate_task_tag": {
                "type": "boolean",
                "label": "Tag images with a tag unique to this task",
                "description": "Creates a dedicated tag for this task config and applies it to all saved images. Useful for reviewing just this task's output.",
                "default": False,
            },
            "task_tag_name": {
                "type": "string",
                "label": "Generated tag name (optional)",
                "description": "Custom name for the per-task generated tag. Defaults to the config name if left blank.",
                "default": "",
            },
            "filter_chain_uuid": {
                "type": "str",
                "label": "Filter Chain (optional)",
                "required": False,
                "description": (
                    "Run this chain against each frame's detections before saving — "
                    "discard corrupt frames, relabel known false-positives, or strip/tag "
                    "matches before they ever land in the dataset."
                ),
            },
            "dedup_group": {
                "type": "group",
                "label": "Deduplication",
                "collapsed": True,
                "fields": {
                    "enable_dedup": {
                        "type": "boolean",
                        "label": "Enable Deduplication",
                        "description": "Check each frame against the perceptual hash index and handle duplicate detections.",
                        "default": False,
                    },
                    "duplicate_action": {
                        "type": "select",
                        "label": "Action on Duplicate",
                        "description": "What to do when a detection matches an existing entry in the perceptual hash index.",
                        "options": [
                            {
                                "value": "discard",
                                "label": "Keep if any unique — discard the frame only when all detections are duplicates",
                            },
                            {
                                "value": "discard_image",
                                "label": "Discard if image matches — skip the frame when the whole scene is a perceptual duplicate",
                            },
                            {
                                "value": "ignore",
                                "label": "Ignore — save normally without any duplicate marking",
                            },
                        ],
                        "default": "discard",
                    },
                    "image_dup_threshold": {
                        "type": "int",
                        "label": "Image Duplicate Threshold",
                        "description": "Max Hamming distance for two whole-image phashes to be considered the same scene.",
                        "default": 5,
                    },
                    "bbox_dup_threshold": {
                        "type": "int",
                        "label": "BBox Duplicate Threshold",
                        "description": "Max Hamming distance for two bbox crop phashes to be considered the same object appearance.",
                        "default": 8,
                    },
                },
            },
        }
