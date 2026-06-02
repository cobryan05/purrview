import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import cv2
import db_client
from db_client.api.images_api import ImagesApi
from db_client.models.bounding_box_metadata_model import BoundingBoxMetadataModel
from db_client.models.update_metadata_payload import UpdateMetadataPayload
import inference_client
from inference_client.api.inference_api import InferenceApi
from inference_client.api.models_api import ModelsApi
from inference_client.models.inference_payload import InferencePayload
from inference_client.models.pin_model_payload import PinModelPayload

from apps.helpers.db.db_client import UNVERIFIED_TAG_UUID
from apps.helpers.imageProviders.imageProvider import ImageProvider, ImageProviderType, ImageWithProviderMetadata
from apps.helpers.imageUtils import base64_encode_png

from .Registry import register_task
from .Task import Task
from .utils.imageHelper import ImageHelper
from .utils.sourceHelper import SourceHelper

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


@dataclass
class TargetLabel:
    label_uuid: str
    min_confidence: float


@register_task()
class CollectTrainingImagesTask(Task):
    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        self._status_msg: str = "Initializing"
        self.log("Initializing task parameters")
        self._source_uuids: list[str] = params["source_uuids"]
        self._output_dir: str = params["output_dir"]

        # model_label_config: [{modelName, labelValues: {labelUuid: confidence}}]
        model_label_config: list[dict] = params.get("model_label_config", [])
        model_names_set: set[str] = set()
        # Flatten to label_uuid → min_confidence; skip zero-confidence (disabled in UI)
        label_conf: dict[str, float] = {}
        for item in model_label_config:
            if item.get("modelName"):
                model_names_set.add(item["modelName"])
            for label_uuid, conf in item.get("labelValues", {}).items():
                conf_f = float(conf)
                if conf_f > 0:
                    label_conf[label_uuid] = min(label_conf.get(label_uuid, 1.0), conf_f)
        self._model_names: list[str] = list(model_names_set)

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

    async def _run(self) -> dict[str, Any]:
        self._unverified_tag_uuid = UNVERIFIED_TAG_UUID

        db_api_client = self._manager.get_db_api_client()
        inference_api_client = self._manager.get_inference_api_client()

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
                pin_ids[name] = resp.pin_id
                self.log(f"Pinned model: {name}")
            except Exception:
                logger.exception(f"Failed to pin model {name}")
                self.log(f"Failed to pin model: {name}")
                pin_ids[name] = None

        images_api = ImagesApi(db_api_client)
        inference_api = InferenceApi(inference_api_client)
        files_root = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))

        # Background save tasks: list of (Task, unique_id).  Saves run concurrently
        # with fetch/encode/infer so the ~100 ms metadata write is hidden.
        pending_saves: list[tuple[asyncio.Task, Optional[str]]] = []

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

        def _collect_saves() -> None:
            """Harvest results from any completed save tasks and update counters."""
            still_pending = []
            for task, uid in pending_saves:
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
            pending_saves.clear()
            pending_saves.extend(still_pending)

        async def _drain_saves() -> None:
            """Wait for all in-flight saves then collect their results."""
            if pending_saves:
                await asyncio.gather(*(t for t, _ in pending_saves), return_exceptions=True)
            _collect_saves()

        self._status_msg = "Running"
        self.log(f"Starting collection (resuming from {self._images_saved} images)")
        for provider in providers:
            await provider.start()
        try:
            active = set(providers)
            while active and not self._cancel_flag.is_set():
                _collect_saves()
                self._status_msg = _status_msg_str()

                any_fetched = False
                for provider in list(active):
                    if self._cancel_flag.is_set():
                        break

                    # Rate-limit LIVE providers: skip until enough time has elapsed.
                    ptype = provider.provider_type()
                    if (interval := _rate_limits.get(ptype)) is not None:
                        if time.perf_counter() - last_fetch.get(id(provider), 0) < interval:
                            continue

                    t0 = time.perf_counter()
                    provided = await provider.getNextImage()
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

                    # base64_encode_png is CPU-bound (PNG compression); run off the event loop
                    base64_img = await asyncio.to_thread(base64_encode_png, provided.image)
                    t_encode = time.perf_counter()

                    matching = await self._run_inference(inference_api, base64_img, pin_ids)
                    t_infer = time.perf_counter()

                    if not matching:
                        logger.debug(
                            f"Frame timing (no match): fetch={1000*(t_fetch-t0):.1f}ms "
                            f"encode={1000*(t_encode-t_fetch):.1f}ms "
                            f"infer={1000*(t_infer-t_encode):.1f}ms"
                        )
                        continue

                    logger.debug(
                        f"Frame timing (matched): fetch={1000*(t_fetch-t0):.1f}ms "
                        f"encode={1000*(t_encode-t_fetch):.1f}ms "
                        f"infer={1000*(t_infer-t_encode):.1f}ms (save dispatched)"
                    )
                    save_task = asyncio.create_task(
                        self._save_image(images_api, provided, matching, files_root)
                    )
                    pending_saves.append((save_task, unique_id))

                    if self._data_req_flag.is_set():
                        await _drain_saves()
                        self._data_req_flag.clear()
                        self._resume_data = self._build_resume_data()
                        self._data_ready_flag.set()

                # If all remaining active providers were rate-limited this iteration,
                # yield briefly to avoid busy-spinning.
                if not any_fetched:
                    await asyncio.sleep(0.05)

        finally:
            await _drain_saves()
            for provider in providers:
                await provider.stop()

        self._resume_data = self._build_resume_data()
        self._data_ready_flag.set()
        return {"images_saved": self._images_saved}

    def _build_resume_data(self) -> dict[str, Any]:
        return {
            "images_saved": self._images_saved,
            "images_processed": self._images_processed,
            "fixed_frames_processed": self._fixed_frames_processed,
            "processed_ids": list(self._processed_ids),
        }

    async def _run_inference(
        self,
        inference_api: InferenceApi,
        base64_img: str,
        pin_ids: dict[str, Optional[str]],
    ) -> list:
        """Runs all configured models; returns detections meeting per-label confidence thresholds."""
        matching = []
        for name in self._model_names:
            try:
                t_start = time.perf_counter()
                result = await asyncio.to_thread(
                    inference_api.inference,
                    name,
                    InferencePayload(
                        image_base64=base64_img,
                        conf_thresh=self._min_all_conf,
                        pin_id=pin_ids.get(name),
                    ),
                )
                logger.debug(
                    f"Inference HTTP round-trip model='{name}': {1000*(time.perf_counter()-t_start):.1f}ms "
                    f"({len(result.result.detections or [])} detections)"
                )
                for det in result.result.detections or []:
                    luuid = det.bbox.label_uuid
                    if luuid and luuid in self._target_label_map:
                        if det.confidence >= self._target_label_map[luuid]:
                            matching.append(det)
            except Exception:
                logger.exception(f"Inference error on model {name}")
        return matching

    async def _save_image(
        self,
        images_api: ImagesApi,
        provided: ImageWithProviderMetadata,
        matching: list,
        files_root: Path,
    ) -> Optional[str]:
        """Writes image to disk and persists bounding-box metadata. Returns filename on success, None on failure."""
        source_name = ImageHelper.sanitize_filename(provided.metadata.source or "unknown")
        frame_idx = provided.metadata.frame_idx or 0
        filename = f"{source_name}_{frame_idx}.png"
        rel_path = str(Path(self._output_dir) / filename)
        abs_path = files_root / self._output_dir / filename
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # cv2.imwrite is CPU-bound (PNG compression) + I/O; run off the event loop
        t0 = time.perf_counter()
        if not await asyncio.to_thread(cv2.imwrite, str(abs_path), provided.image):
            logger.warning(f"Failed to write image to {abs_path}")
            self.log(f"Failed to write image: {filename}")
            return None
        logger.debug(f"Image write: {1000*(time.perf_counter()-t0):.1f}ms → {abs_path}")

        boxes = [
            BoundingBoxMetadataModel(
                bbox_uuid=str(uuid4()),
                x=det.bbox.x,
                y=det.bbox.y,
                width=det.bbox.width,
                height=det.bbox.height,
                label_uuid=det.bbox.label_uuid,
                class_str=det.bbox.class_str,
                tag_uuids=[self._unverified_tag_uuid],
                extra={"confidence": str(det.confidence)},
            )
            for det in matching
        ]

        try:
            t_meta = time.perf_counter()
            await asyncio.to_thread(
                images_api.update_image_metadata,
                UpdateMetadataPayload(image_path=rel_path, boxes=boxes),
            )
            logger.debug(f"Metadata save: {1000*(time.perf_counter()-t_meta):.1f}ms for {filename}")
        except Exception:
            logger.exception(f"Failed to save metadata for {rel_path}")
            self.log(f"Failed to save metadata: {filename}")
            return None

        return filename

    async def _deinit(self) -> None:
        pass

    def get_status_message(self) -> str:
        return self._status_msg

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        return {
            "meta": {
                "order": ["output_dir", "snapshot_interval", "stream_interval", "source_uuids", "model_label_config"]
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
                "type": "string",
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
        }
