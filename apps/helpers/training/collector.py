"""The runtime half of training-image collection: everything downstream of
*(frame, detections)*.

`CollectTrainingImagesTask` owned all of this inline. It is being split because a
watcher will drive the same collection over frames it is already pulling
(`COLLECTOR_UNIFICATION_PLAN.md`), and the two halves have genuinely different
shapes: one is a source-rotation loop, the other is "given a frame, decide whether
to keep it and how to annotate it". After the split the task is its providers, its
pacing and its resume bookkeeping, plus `await collector.offer(...)`.

**The saved image is always full resolution**, in every mode. `FrameOffer.provided`
carries the frame the provider returned, and `_save_image` writes that and nothing
else. An analysis downscale (the watcher's 640 px inference frame) must never reach
the training set — a 640 px training set would be worthless, and training data is
the entire point of this class.

## The host seam

The collector takes a `CollectorHost`, not a task manager. Three reasons, and the
third is the one that would otherwise bite:

* it stays constructible in a test without a manager or a DB;
* `helpers/` must not import from `tasks_server/`, so the filter-chain runner and
  its inference backend arrive as factories rather than imports;
* API **clients are rebuilt per use, from factories**. The task deliberately
  reconstructs `InferenceApi` on every loop pass so a Settings → Servers change is
  picked up mid-run; a collector holding a cached client would keep talking to the
  host the user navigated away from, whose connection pool has since been cleared.

## Backpressure

`offer()` blocks when `blocking_offer` is set — correct while the collector *is*
the loop, since a full dedup queue should slow the fetch rather than drop frames.
Inside a watcher it must not: stalling the analysis loop behind a phash makes the
watcher miss events, and a watcher that misses events is broken in a way a
collector that misses frames is not. See `frames_dropped_behind`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid as uuid_module
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import cv2
from db_client.api.filter_chains_api import FilterChainsApi
from db_client.api.images_api import ImagesApi
from db_client.api.labels_api import LabelsApi
from db_client.api.settings_api import SettingsApi
from db_client.api.tags_api import TagsApi
from db_client.models.add_tag_payload import AddTagPayload
from db_client.models.bounding_box_metadata_model import BoundingBoxMetadataModel
from db_client.models.set_image_tag_payload import SetImageTagPayload
from db_client.models.tag_kind import TagKind as ApiTagKind
from db_client.models.update_metadata_payload import UpdateMetadataPayload
from inference_client.api.inference_api import InferenceApi
from inference_client.api.models_api import ModelsApi
from inference_client.models.pin_model_payload import PinModelPayload

from apps.helpers.bboxUtils import iou_xywh
from apps.helpers.db.db_client import CONFUSABLE_TAG_UUID, DETECTOR_MISS_TAG_UUID, UNVERIFIED_TAG_UUID
from apps.helpers.detectionMerger import merge_detections_by_iou_and_hierarchy
from apps.helpers.frameInference import FrameInferenceRequest, run_frame_inference
from apps.helpers.hashUtils import (
    DEFAULT_BBOX_HASH_SIZE,
    DEFAULT_IMAGE_HASH_SIZE,
    PHASH_JPEG_QUALITY,
    compute_phashes_with_jpeg_roundtrip,
    hamming,
    phash_from_numpy_crop,
)
from apps.helpers.imageHelper import ImageHelper
from apps.helpers.imageProviders.imageProvider import ImageProviderType, ImageWithProviderMetadata
from apps.helpers.labelHelper import build_label_ancestry_map, build_label_metadata_map
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.training.collectorSpec import (
    AMBIGUITY_KIND_CROSS_MODEL,
    AMBIGUITY_METRIC_RATIO,
    SAME_BBOX_IOU_THRESHOLD,
    AmbiguityInfo,
    CollectorSpec,
    ambiguity_fires,
    ambiguity_value,
)

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)


@dataclass
class CollectorHost:
    """Everything the collector needs from whoever is running it.

    Deliberately a bag of callables rather than the task or the manager: the point
    of the split is that a watcher can drive collection, and a watcher is not a
    task. Every field here is something only the driver can answer.
    """

    #: Appends to the driver's user-visible log (task log line, watcher log line).
    log: Callable[[str], None]
    #: Called per use, never cached — see the module docstring.
    db_api_client: Callable[[], Any]
    inference_api_client: Callable[[], Any]
    #: Set when the driver wants to stop; the pin-renewal loop waits on it.
    stop_event: asyncio.Event
    #: `model_name -> model_kind`. Supplied by the driver so `helpers/` need not
    #: import the tasks package that owns the model-metadata reader.
    model_kind_reader: Callable[[str], str] = lambda name: "detector"
    #: Kinds that auto-pin per call and therefore need no long-lived pin.
    classifier_kinds: frozenset = frozenset({"image_classifier", "multi_label_classifier"})
    #: `(client_provider, pin_ids) -> InferenceBackend` for the filter chain.
    build_inference_backend: Optional[Callable[[Callable[[], Any], dict], Any]] = None
    #: `(steps, backend) -> FilterChainRunner`.
    build_filter_chain_runner: Optional[Callable[[list, Any], Any]] = None


@dataclass(frozen=True)
class TrackedCrop:
    """One live track's box, normalized to the same frame `matching` detections are
    in — built by the driver, which is the only side that knows `Track.bbox`'s
    pixel space and how to normalize it. The collector never sees a raw `Track`.
    """

    track_id: int
    x: float
    y: float
    width: float
    height: float
    label_uuid: Optional[str] = None
    #: Whether the tracker matched a real detection to this track on the frame
    #: it was captured from — `False` means this is the frame of a detector-miss
    #: gap (plan §4A). Defaults `True` so phase 6's per-track novelty callers
    #: (which don't care about this) don't have to set it.
    matched_this_update: bool = True
    #: Mirrors `Track.confirmed` — plan §4A is explicit that only a *confirmed*
    #: track's gaps are worth collecting; a still-provisional (single-hit) track
    #: is exactly as likely to be tracker noise as a real object. Defaults `True`
    #: for the same reason as `matched_this_update`: callers that don't care
    #: about detector-miss gating shouldn't have to set it.
    confirmed: bool = True


#: Why a frame was kept. Carried from the decision point through the dedup queue
#: to the completed save, so the status card can answer "is this knob producing
#: anything" (plan §6) — a single `saved` total cannot, because the interesting
#: reasons are the rare ones.
SAVE_REASON_BAND = "band"
SAVE_REASON_DETECTOR_MISS = "detector_miss"
SAVE_REASON_TRACK_WORST = "track_worst"
SAVE_REASONS: Tuple[str, ...] = (SAVE_REASON_BAND, SAVE_REASON_DETECTOR_MISS, SAVE_REASON_TRACK_WORST)


@dataclass
class FrameOffer:
    """One frame handed to the collector.

    `provided.image` is the FULL-RESOLUTION frame and is the only thing ever
    written to disk. `base64` is what inference actually saw — which may be a
    downscaled analysis frame when a watcher is driving — and is used for the
    inference call and for the filter chain's image bytes.

    ⚠️ `base64` is the JSON packet `base64_encode_jpg` produces,
    `{"format", "data", "metadata"}`, **not** bare base64. Decoding it directly
    raises `Incorrect padding` on the first real frame; unwrap `["data"]` first.
    """

    provided: ImageWithProviderMetadata
    base64: str
    provider_type: ImageProviderType = ImageProviderType.FIXED
    #: `source:frame_idx` for FIXED sources, so a resume can skip it. None means
    #: "don't record this frame as processed" (live sources reset frame_idx).
    unique_id: Optional[str] = None
    #: Live tracks as of this frame. Empty for the standalone task (no tracker) —
    #: per-track novelty sampling is watcher-only by construction, not by a flag:
    #: an empty list here simply never matches any detection, so `offer()` falls
    #: through to today's behaviour unchanged.
    tracks: Tuple[TrackedCrop, ...] = ()
    #: track_ids that vanished between the previous offer and this one — see
    #: `WatchPipeline.dead_track_ids`.
    dead_track_ids: Tuple[int, ...] = ()


class TrainingCollector:
    """Bands, ambiguity, filter chain, dedup and save — the collect pipeline.

    One entry point on the hot path (`offer`), plus lifecycle (`prepare` / `start` /
    `shutdown`) and read-only reporting (`status_stats`, `notices`,
    `review_actions`, `result_fragment`).

    Everything here was moved wholesale out of `CollectTrainingImagesTask`; the
    comments explaining *why* each rule is the way it is came with it, because
    several of them record decisions that look wrong until you know the reason.
    """

    # Per-call retry limits and timeouts for I/O that can hang indefinitely.
    MAX_RETRIES: int = 3
    INFERENCE_TIMEOUT_S: float = 60.0  # per inference call (allows for model load)
    # Dedup pipeline buffer: this many matched frames may queue before the offer
    # blocks (or drops, when blocking_offer is False). A small buffer absorbs
    # per-frame jitter without wasting memory — each slot holds a reference.
    DEDUP_QUEUE_SIZE: int = 4
    # Seconds a model pin is requested for, renewed at roughly half that so a run
    # longer than the duration never lets a pin lapse silently.
    PIN_DURATION_S: int = 3600
    PIN_RENEWAL_INTERVAL_S: float = PIN_DURATION_S / 2
    # Rolling window of per-frame nearest-neighbour distances behind the
    # threshold-calibration widgets. In-memory only (never checkpointed): it is a
    # "what is the stream doing right now" sample, and replaying distances measured
    # before a long pause would describe a scene that no longer exists.
    DEDUP_SAMPLE_SIZE: int = 200
    # Candidate thresholds shown in the sweep tables. Image phashes are hash_size²
    # bits (64 at the default 8) and bbox phashes 256 at the default 16, so the two
    # live on different scales and get their own ranges rather than a shared one.
    # Both ladders run past any threshold anyone would plausibly configure and cover
    # the same span the histogram does: a sweep that stops at the current setting
    # can't answer "what would raising it do?", which is the whole point of the
    # table. Steps widen with distance because that is where the curve flattens.
    IMAGE_SWEEP_THRESHOLDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 64)
    BBOX_SWEEP_THRESHOLDS: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 90, 100)

    # Actions that actually drop frames. "ignore" and "measure" are pass-through.
    _DISCARD_ACTIONS: tuple[str, ...] = ("discard", "discard_image")

    # Default safety valve for per-track novelty sampling (plan §4B): a track that
    # never stops moving would otherwise keep adding distinct-enough crops forever.
    # This is NOT the rule — appearance novelty is — it only bounds the
    # pathological case. Kept well above a normal visit's pose count. Overridable
    # per collector via `spec.max_frames_per_track`; this constant is the fallback
    # for a spec built before the knob existed.
    MAX_FRAMES_PER_TRACK: int = 25

    def __init__(
        self,
        spec: CollectorSpec,
        host: CollectorHost,
        *,
        files_root: Path,
        tag_scope_uuid: str = "",
        tag_scope_name: str = "",
        blocking_offer: bool = True,
    ) -> None:
        self._spec = spec
        self._host = host
        self._log = host.log
        self._files_root = files_root
        # Identity for the optional per-run TASK_GENERATED tag. For the standalone
        # task this is the task config; for a watcher-attached collector it is the
        # collector row, so the tag names the collector rather than the watcher.
        self._tag_scope_uuid = tag_scope_uuid
        self._tag_scope_name = tag_scope_name
        self._blocking_offer = blocking_offer

        # Safe defaults for everything a status/notice poll can reach. The driver
        # may poll before `prepare()` has run (the task's PENDING window), so none
        # of these may be created lazily.
        self._images_saved: int = 0
        self._filtered_discarded: int = 0
        self._filtered_relabeled: int = 0
        self._filtered_tagged: int = 0
        self._filtered_stripped: int = 0
        self._ambiguity_flagged: int = 0
        self._ambiguity_evaluated: int = 0
        self._ambiguity_samples: deque = deque(maxlen=self.DEDUP_SAMPLE_SIZE)
        self._ambiguity_pairs: dict[tuple[str, str, str], int] = {}
        self._ambiguity_unavailable_models: set[str] = set()
        self._dedup_frames_checked: int = 0
        self._dedup_image_hits: int = 0
        self._dedup_bbox_hits: int = 0
        self._dup_frames_discarded: int = 0
        self._dedup_samples: deque = deque(maxlen=self.DEDUP_SAMPLE_SIZE)
        #: Frames refused because the dedup queue was full. Only ever non-zero for
        #: a non-blocking (watcher-attached) collector; surfaced in status because
        #: silent dropping is indistinguishable from "the config collects nothing".
        self._frames_dropped_behind: int = 0
        #: Total calls to `offer()`, before any filtering — the denominator that
        #: makes "saved" and "dropped_behind" readable as rates rather than bare
        #: counts. Every other counter here is post-some-filter; this is the one
        #: that answers "is the driver even calling offer() at all".
        self._frames_offered: int = 0
        #: track_id -> hex phashes of crops already kept for that track this run
        #: (plan §4B: "is this pose meaningfully different from what's already
        #: kept for THIS track" — intra-visit decimation, complementary to the
        #: global phash's cross-visit dedup, not a replacement for it). Bounded by
        #: `MAX_FRAMES_PER_TRACK`; forgotten once the track dies.
        self._track_hashes: Dict[int, deque] = {}
        #: Resolved once: a spec built before the knob existed leaves it None, and
        #: reading it per frame would re-do that fallback on the hot path.
        self._max_frames_per_track: int = getattr(spec, "max_frames_per_track", None) or self.MAX_FRAMES_PER_TRACK
        #: track_id -> (confidence, frame, detection) for the lowest-confidence
        #: crop offered for that track that the novelty check above has NOT
        #: already let through naturally. Committed on track death so the
        #: hardest example of a visit is never lost to "it looked like an
        #: earlier pose" — see `_handle_dead_tracks`.
        self._track_worst: Dict[int, Tuple[float, "FrameOffer", Any]] = {}
        self._track_novelty_kept: int = 0
        self._track_novelty_dropped: int = 0
        self._track_worst_committed: int = 0
        #: Detector-miss boxes handed to `offer_detector_miss` (plan §4A). Only
        #: ever incremented by a watcher-attached collector with
        #: `enable_detector_miss` on — the standalone task never calls this
        #: method, having no tracker to supply a gap in the first place.
        #: Counts OFFERS, not saves: dedup still gets a veto downstream, so the
        #: matching saved subtotal is `_saved_by_reason[SAVE_REASON_DETECTOR_MISS]`.
        self._detector_miss_offered: int = 0
        #: Saves that actually landed, split by why the frame was kept (plan §6:
        #: "break saves down by why they were collected ... that is what tells you
        #: whether a knob is producing anything"). Attributed where the save
        #: completes, not where it is dispatched, so the subtotals always sum to
        #: `_images_saved` rather than counting frames dedup later discarded.
        self._saved_by_reason: Dict[str, int] = {r: 0 for r in SAVE_REASONS}

        self._last_capture: Optional[dict[str, Any]] = None
        self._filter_chain_runner: Any = None
        self._task_tag_uuid: Optional[str] = None
        self._extra_tag_uuids: list[str] = list(spec.apply_tag_uuids)
        self._unverified_tag_uuid: str = UNVERIFIED_TAG_UUID

        self._label_ancestry: dict[str, set[str]] = {}
        self._label_meta: dict[str, dict[str, Any]] = {}
        self._image_hash_size: int = DEFAULT_IMAGE_HASH_SIZE
        self._bbox_hash_size: int = DEFAULT_BBOX_HASH_SIZE

        #: Widest model input across this collector's stack, read once in `start()`.
        #: None until then, and None for a stack whose shapes could not be read —
        #: both mean "the driver should not change what it already sends".
        self._model_input_width: Optional[int] = None

        self._pin_ids: dict[str, Optional[str]] = {}
        self._chain_pin_ids: dict[str, str] = {}
        self._combined_pin_ids: dict[str, str] = {}

        self._pending_saves: list[tuple[asyncio.Task, Optional[str], str]] = []
        self._dedup_queue: Optional[asyncio.Queue] = None
        self._dedup_task: Optional[asyncio.Task] = None
        self._renew_pins_task: Optional[asyncio.Task] = None
        # Created in start(): an event loop must exist, and the collector is
        # constructed before the driver's loop is necessarily running.
        self._meta_write_lock: Optional[asyncio.Lock] = None

    # ── read-only accessors the driver reports on ────────────────────────────

    @property
    def spec(self) -> CollectorSpec:
        return self._spec

    @property
    def images_saved(self) -> int:
        return self._images_saved

    def send_width(self, native_width: int) -> Optional[int]:
        """Width the driver should encode this frame at, or None for "unchanged".

        Plan §2.3b's arithmetic, and deliberately arithmetic rather than a user
        toggle: `min(source_native_width, model_input_width)`. Returns None when
        the model width is unknown, so an unreadable ONNX graph can never make a
        working collector start re-encoding frames at a guessed size.
        """
        if not self._model_input_width or native_width <= 0:
            return None
        return min(native_width, self._model_input_width)

    @property
    def pending_saves(self) -> list[tuple[asyncio.Task, Optional[str], str]]:
        """In-flight save tasks. Exposed so a driver can tell whether an offer
        dispatched a save without waiting for it to finish."""
        return self._pending_saves

    @property
    def last_capture(self) -> Optional[dict[str, Any]]:
        return self._last_capture

    @property
    def task_tag_uuid(self) -> Optional[str]:
        return self._task_tag_uuid

    @property
    def filter_chain_active(self) -> bool:
        return self._filter_chain_runner is not None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def prepare(self, resume_data: Optional[dict[str, Any]] = None) -> None:
        """Read the settings and label maps collection needs, and restore counters.

        Split from `start()` because it is the part with no side effects on the
        inference server: nothing is pinned, no tag is created, no worker starts.
        A driver that only wants to render status can stop here.
        """
        if resume_data:
            self.apply_resume(resume_data)

        # Hash sizes are global settings (they must match the index; changing them
        # requires recomputing every stored hash). Read from DB settings, falling
        # back to the shared defaults.
        db_client = self._host.db_api_client()
        settings_api = SettingsApi(db_client)
        labels_api = LabelsApi(db_client)
        (
            self._image_hash_size,
            self._bbox_hash_size,
            self._label_ancestry,
            self._label_meta,
        ) = await asyncio.gather(
            _read_int_setting(settings_api, "dedup_image_hash_size", DEFAULT_IMAGE_HASH_SIZE),
            _read_int_setting(settings_api, "dedup_bbox_hash_size", DEFAULT_BBOX_HASH_SIZE),
            build_label_ancestry_map(labels_api),
            build_label_metadata_map(labels_api),
        )

    def apply_resume(self, resume_data: dict[str, Any]) -> None:
        """Restore the cumulative counters from a checkpoint."""
        self._images_saved = resume_data.get("images_saved", 0)
        self._dup_frames_discarded = resume_data.get("dup_frames_discarded", 0)
        self._dedup_frames_checked = resume_data.get("dedup_frames_checked", 0)
        self._dedup_image_hits = resume_data.get("dedup_image_hits", 0)
        self._dedup_bbox_hits = resume_data.get("dedup_bbox_hits", 0)
        self._filtered_discarded = resume_data.get("filtered_discarded", 0)
        self._filtered_relabeled = resume_data.get("filtered_relabeled", 0)
        self._filtered_tagged = resume_data.get("filtered_tagged", 0)
        self._filtered_stripped = resume_data.get("filtered_stripped", 0)
        self._ambiguity_flagged = resume_data.get("ambiguity_flagged", 0)
        self._ambiguity_evaluated = resume_data.get("ambiguity_evaluated", 0)

    async def start(self) -> None:
        """Resolve tags, pin models, build the filter chain, start the workers."""
        self._meta_write_lock = asyncio.Lock()
        db_api_client = self._host.db_api_client()
        inference_api_client = self._host.inference_api_client()

        await self._resolve_tags(db_api_client)
        models_api = ModelsApi(inference_api_client)
        await self._pin_models(models_api)
        await self._read_model_input_width(models_api)
        await self._build_filter_chain(db_api_client, models_api)

        if self._spec.enable_dedup:
            self._dedup_queue = asyncio.Queue(maxsize=self.DEDUP_QUEUE_SIZE)
            self._dedup_task = asyncio.create_task(self._dedup_worker(ImagesApi(db_api_client)))
            self._log(
                f"Dedup enabled: action={self._spec.duplicate_action}, "
                f"img_threshold={self._spec.image_dup_threshold}, "
                f"bbox_threshold={self._spec.bbox_dup_threshold}, "
                f"hash_sizes={self._image_hash_size}x{self._bbox_hash_size}"
            )
        # Independent of the capture loop so a slow/paced source never delays a
        # renewal and a renewal failure never blocks a frame.
        self._renew_pins_task = asyncio.create_task(self._renew_pins_periodically(models_api))

    async def shutdown(self) -> None:
        """Stop the workers, finish the in-flight saves, release the pins.

        Ordered deliberately: the renewal loop dies first so it cannot race the
        unpinning below (renewing a model this same teardown is about to unpin),
        then dedup drains so every queued frame has its save task registered
        before we wait on them.
        """
        if self._renew_pins_task is not None:
            self._renew_pins_task.cancel()
            try:
                await self._renew_pins_task
            except asyncio.CancelledError:
                pass
            self._renew_pins_task = None

        if self._dedup_task is not None:
            assert self._dedup_queue is not None
            await self._dedup_queue.put(None)  # sentinel
            await self._dedup_task
            self._dedup_task = None

        await self.drain_saves()

        unpin_api = ModelsApi(self._host.inference_api_client())
        for name, pid in self._pin_ids.items():
            if pid:
                try:
                    await asyncio.to_thread(unpin_api.unpin_model, name, pid)
                    self._log(f"Unpinned model: {name}")
                except Exception:
                    logger.warning(f"Failed to unpin model '{name}' (pin {pid})")
        for name, pid in self._chain_pin_ids.items():
            try:
                await asyncio.to_thread(unpin_api.unpin_model, name, pid)
            except Exception:
                logger.warning(f"Failed to unpin filter-chain model '{name}' (pin {pid})")

    async def _resolve_tags(self, db_api_client) -> None:
        self._extra_tag_uuids = list(self._spec.apply_tag_uuids)
        if not self._spec.generate_task_tag:
            return
        tag_name = self._spec.task_tag_name_override or (
            f"[Task] {self._tag_scope_name}" if self._tag_scope_name else f"[Task] {self._tag_scope_uuid[:8]}"
        )
        # The tag is identified by (scope, name), not by scope alone. Keying it on
        # the config UUID only meant a name change reused the same DB row and
        # renamed it — retroactively relabelling every image already collected
        # under the old name. A new name must mint a new tag and leave the old one
        # (and its images) alone.
        legacy_tag_uuid = str(uuid_module.uuid5(uuid_module.NAMESPACE_OID, self._tag_scope_uuid))
        task_tag_uuid = str(uuid_module.uuid5(uuid_module.NAMESPACE_OID, f"{self._tag_scope_uuid}:{tag_name}"))
        tags_api = TagsApi(db_api_client)
        existing_tags = await asyncio.to_thread(tags_api.list_tags)
        existing_by_uuid = {t.uuid: t for t in (existing_tags.tags or [])}
        # Configs that predate the name-keyed UUID carry a legacy row. As long as
        # its name still matches, keep using it so their history stays on one tag.
        legacy_tag = existing_by_uuid.get(legacy_tag_uuid)
        if legacy_tag is not None and legacy_tag.name == tag_name:
            task_tag_uuid = legacy_tag_uuid
        if task_tag_uuid not in existing_by_uuid:
            try:
                await asyncio.to_thread(
                    tags_api.add_tag,
                    AddTagPayload(name=tag_name, kind=ApiTagKind.TASK_GENERATED, uuid=task_tag_uuid),
                )
                self._log(f"Created task tag: {tag_name} ({task_tag_uuid})")
                self._extra_tag_uuids.append(task_tag_uuid)
                self._task_tag_uuid = task_tag_uuid
            except Exception:
                logger.exception(f"Failed to create task tag {tag_name}")
        else:
            # Reusing an existing row — never rename it. Its name is what the images
            # already carrying it were collected under.
            self._log(f"Reusing task tag: {tag_name} ({task_tag_uuid})")
            self._extra_tag_uuids.append(task_tag_uuid)
            self._task_tag_uuid = task_tag_uuid

    async def _pin_models(self, models_api: ModelsApi) -> None:
        if self._spec.collect_all:
            self._log("No models configured — collecting every frame (filter chain and dedup still apply)")
        else:
            self._log(f"Pinning {len(self._spec.model_names)} model(s)")
        for name in self._spec.model_names:
            try:
                resp = await asyncio.to_thread(
                    models_api.pin_model_api_api_models_model_name_pin_post,
                    name,
                    PinModelPayload(model_name=name, duration=self.PIN_DURATION_S),
                )
                if resp.pin_id:
                    self._pin_ids[name] = resp.pin_id
                    self._log(f"Pinned model: {name} (pin_id={resp.pin_id})")
                else:
                    # Server returned failure status without raising — pin_id will
                    # be None, causing the model to reload on every inference call.
                    logger.warning(
                        f"Pin request for '{name}' returned no pin_id: status={resp.status} message={resp.message}"
                    )
                    self._log(f"Warning: could not pin model '{name}' — will reload each inference")
                    self._pin_ids[name] = None
            except Exception:
                logger.exception(f"Failed to pin model {name}")
                self._log(f"Failed to pin model: {name}")
                self._pin_ids[name] = None
        # Single mutable dict shared with the filter-chain backend and the renewal
        # loop, so a renewal's new pin_id is visible everywhere without rebuilding
        # the backend.
        self._combined_pin_ids = {k: v for k, v in self._pin_ids.items() if v}

    async def _read_model_input_width(self, models_api: ModelsApi) -> None:
        """Widest input this collector's models can actually use (plan §2.3b).

        `send_width = min(source_native_width, model_input_width)`: below the
        model's input you throw away detail it could have used; above it you pay to
        ship pixels the server resizes away. A watcher drives inference at its own
        `analysis_width` (640 by default), which is already the optimum for a
        640-input model — the common case, where this reads back 640 and the driver
        changes nothing. It only earns its keep for a larger-input model.

        Best-effort by construction: `input_shape` is `None` for any model the
        inference server could not read a graph for, and a missing number must
        leave today's behaviour alone rather than guess a width.
        """
        widths = []
        for name in self._spec.model_names:
            try:
                meta = await asyncio.to_thread(models_api.get_model_meta, name)
            except Exception:
                logger.debug("Could not read input_shape for model '%s'", name, exc_info=True)
                continue
            shape = getattr(meta, "input_shape", None)
            # (height, width) — see ModelMetaResponse.
            if shape and len(shape) == 2 and shape[1]:
                widths.append(int(shape[1]))
        self._model_input_width = max(widths) if widths else None

    async def _build_filter_chain(self, db_api_client, models_api: ModelsApi) -> None:
        if not self._spec.filter_chain_uuid:
            return
        if self._host.build_filter_chain_runner is None or self._host.build_inference_backend is None:
            # The chain is a driver-supplied capability (see the module docstring on
            # layering). Saying so beats silently collecting unfiltered frames.
            raise ValueError("A filter chain is configured but this host cannot build one")
        filter_chains_api = FilterChainsApi(db_api_client)
        chain_resp = await asyncio.to_thread(filter_chains_api.get_filter_chain, self._spec.filter_chain_uuid)
        chain = getattr(chain_resp, "chain", None)
        if chain is None:
            raise ValueError(f"Filter chain not found: {self._spec.filter_chain_uuid}")
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
        if not chain_steps:
            return
        # Reuse a model already pinned by model_label_config where possible; only
        # pin the chain's own detector-step models otherwise.
        for name in sorted({s["model_name"] for s in chain_steps}):
            if self._pin_ids.get(name):
                continue
            try:
                kind = await asyncio.to_thread(self._host.model_kind_reader, name)
            except ValueError:
                logger.exception("Invalid filter-chain step model_name %r", name)
                continue
            if kind in self._host.classifier_kinds:
                continue  # /classify auto-pins per call; no long-lived pin needed
            try:
                resp = await asyncio.to_thread(
                    models_api.pin_model_api_api_models_model_name_pin_post,
                    name,
                    PinModelPayload(model_name=name, duration=self.PIN_DURATION_S),
                )
                if resp.pin_id:
                    self._chain_pin_ids[name] = resp.pin_id
            except Exception:
                logger.exception("Failed to pin filter-chain model %s", name)

        self._combined_pin_ids.update({k: v for k, v in self._chain_pin_ids.items() if v})
        backend = self._host.build_inference_backend(self._host.inference_api_client, self._combined_pin_ids)
        self._filter_chain_runner = self._host.build_filter_chain_runner(chain_steps, backend)
        self._log(f"Filter chain '{chain.name}' active ({len(chain_steps)} step(s))")

    # ── the hot path ─────────────────────────────────────────────────────────

    async def offer(self, frame: FrameOffer) -> None:
        """Decide whether this frame is worth keeping, and dispatch the save.

        Blocks on a full dedup queue only when `blocking_offer` is set; otherwise
        the frame is dropped and counted. See the module docstring.
        """
        self._frames_offered += 1
        # Collect-everything mode skips inference entirely — there is no model to
        # run and nothing to match against, so the frame is saved unannotated
        # unless the filter chain or dedup drops it. _run_inference stashes
        # per-detection model/ambiguity metadata on `provided` so it rides through
        # the dedup queue into _save_image unchanged.
        if self._spec.collect_all:
            matching: list = []
        else:
            inference_api = InferenceApi(self._host.inference_api_client())
            matching = await self._run_inference(inference_api, frame.base64, self._pin_ids, provided=frame.provided)

        discarded = False
        force_save = False
        if self._filter_chain_runner is not None:
            matching, discarded, force_save = await self._apply_filter_chain_to_frame(
                frame.provided, frame.base64, matching
            )

        # Per-track novelty prefilter (plan §4B) — cheap (a handful of Hamming
        # distances against hashes already in hand) and local, so it runs before
        # dedup's network round-trip rather than after it. `frame.tracks` is empty
        # for the standalone task (no tracker), which makes this a no-op for it by
        # construction rather than by branching on which driver is calling.
        if frame.tracks and matching:
            matching = self._apply_track_novelty(frame, matching)
        if frame.dead_track_ids:
            await self._commit_dead_track_worst(frame.dead_track_ids)

        # In collect-everything mode `matching` is always empty, so the save is
        # driven by the frame itself: everything the driver hands over is collected
        # unless the filter chain discarded it (or dedup, below, calls it a dup).
        if matching or (self._spec.collect_all and not discarded):
            if self._spec.enable_dedup:
                await self._enqueue_for_dedup(frame, matching)
            else:
                self._dispatch_save(frame, matching)
        elif force_save and not self._spec.enable_dedup:
            # strip_unverified_bboxes emptied `matching`, but per plan 2.1 the image
            # itself must still be saved (never delete images). Skipped when dedup
            # is enabled: an empty bbox list makes the dedup worker's
            # `all(bbox_flags)` vacuously true, which would wrongly discard the
            # frame under duplicate_action="discard" — a narrow edge case, not worth
            # special-casing dedup for an empty bbox set.
            self._dispatch_save(frame, [])

    def _match_track(self, det: Any, tracks: "Tuple[TrackedCrop, ...]") -> Optional[int]:
        """Best-IoU track for one detection, or None if nothing overlaps enough.

        Reuses `SAME_BBOX_IOU_THRESHOLD` — the same "is this the same object"
        bar the cross-model merge uses — rather than inventing a second one.
        """
        best_iou = 0.0
        best_id: Optional[int] = None
        for t in tracks:
            i = iou_xywh(det.bbox, t)
            if i > best_iou:
                best_iou = i
                best_id = t.track_id
        return best_id if best_iou >= SAME_BBOX_IOU_THRESHOLD else None

    def _apply_track_novelty(self, frame: FrameOffer, matching: list) -> list:
        """Keep a crop only if it's a meaningfully different pose than what's
        already been kept for the SAME track this visit (plan §4B).

        This is intra-visit decimation — a loitering animal's stillness produces
        near-identical crops that add volume without information — and it is
        complementary to, not a replacement for, the global phash dedup below,
        which answers a different question ("have I already got this from last
        Tuesday"). A detection matching no live track (nothing close enough by
        IoU) is left untouched: novelty sampling has nothing to say about an
        object the tracker isn't following.
        """
        image = frame.provided.image
        kept: list = []
        for det in matching:
            track_id = self._match_track(det, frame.tracks)
            if track_id is None:
                kept.append(det)
                continue
            crop_hash = phash_from_numpy_crop(
                image, det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height, self._bbox_hash_size
            )
            self._update_track_worst(track_id, det, frame, is_being_saved=False)
            history = self._track_hashes.get(track_id)
            novel = not history or all(hamming(crop_hash, h) > self._spec.bbox_dup_threshold for h in history)
            if not novel:
                self._track_novelty_dropped += 1
                continue
            kept.append(det)
            self._track_novelty_kept += 1
            self._update_track_worst(track_id, det, frame, is_being_saved=True)
            bucket = self._track_hashes.setdefault(track_id, deque(maxlen=self._max_frames_per_track))
            bucket.append(crop_hash)
        return kept

    def _update_track_worst(self, track_id: int, det: Any, frame: FrameOffer, *, is_being_saved: bool) -> None:
        """Track the lowest-confidence crop seen for `track_id` that has not
        already been (or is not about to be) saved through the ordinary path —
        so `_commit_dead_track_worst` never re-saves a crop the driver already
        has on disk, and never loses the hardest example of a visit to a novelty
        filter that (correctly) called it a redundant pose.
        """
        current = self._track_worst.get(track_id)
        if is_being_saved:
            # This confidence is being saved normally this frame. It only
            # discharges the pending "must commit at death" obligation if it's at
            # least as hard as what's pending — a harder crop offered later still
            # needs to be remembered.
            if current is not None and det.confidence <= current[0]:
                self._track_worst.pop(track_id, None)
            return
        if current is None or det.confidence < current[0]:
            self._track_worst[track_id] = (det.confidence, frame, det)

    async def _commit_dead_track_worst(self, dead_track_ids: "Tuple[int, ...]") -> None:
        """A track just ended — save its held hardest example if the novelty
        filter ever withheld one, then forget the track entirely (plan §4B:
        "always keep the track's lowest-confidence frame ... commit at track
        end"). `track_id`s don't survive a tracker restart, so nothing here needs
        to persist past this run.
        """
        for track_id in dead_track_ids:
            self._track_hashes.pop(track_id, None)
            worst = self._track_worst.pop(track_id, None)
            if worst is None:
                continue
            _, held_frame, held_det = worst
            self._track_worst_committed += 1
            if self._spec.enable_dedup:
                await self._enqueue_for_dedup(held_frame, [held_det], SAVE_REASON_TRACK_WORST)
            else:
                self._dispatch_save(held_frame, [held_det], SAVE_REASON_TRACK_WORST)

    async def offer_detector_miss(self, frame: FrameOffer, det: Any) -> None:
        """Save a tracker-supplied box on a detector-miss frame (plan §4A).

        `det` is a synthetic detection built by the driver (the watch action) —
        the tracker's box interpolated between the two real detections
        bracketing the gap, confidence `None`. Bypasses inference (there is
        nothing to run: the driver already decided this frame is worth
        keeping) and per-track novelty (that prefilter reasons about repeated
        *poses* of the same object; a detector-miss box is a one-off event, not
        a pose sample). Still respects the collector's own dedup setting, the
        same as `_commit_dead_track_worst` — "always keep" is about not losing
        it to the novelty filter, not about exempting it from dedup policy.

        Never called by the standalone task: it has no tracker, so it can never
        observe a gap to seed a box for in the first place (plan §4A: "impossible
        without temporal context").
        """
        self._detector_miss_offered += 1
        if self._spec.enable_dedup:
            await self._enqueue_for_dedup(frame, [det], SAVE_REASON_DETECTOR_MISS)
        else:
            self._dispatch_save(frame, [det], SAVE_REASON_DETECTOR_MISS)

    async def _enqueue_for_dedup(self, frame: FrameOffer, matching: list, reason: str = SAVE_REASON_BAND) -> None:
        assert self._dedup_queue is not None
        if self._blocking_offer:
            # Deliberate backpressure: while the collector IS the loop, slowing the
            # fetch is better than dropping a frame nothing else will see.
            await self._dedup_queue.put((frame, matching, reason))
            return
        try:
            self._dedup_queue.put_nowait((frame, matching, reason))
        except asyncio.QueueFull:
            self._frames_dropped_behind += 1
            if self._frames_dropped_behind == 1:
                self._log("Collector is behind — dropping offered frames rather than stalling the caller")

    def _dispatch_save(self, frame: FrameOffer, matching: list, reason: str = SAVE_REASON_BAND) -> asyncio.Task:
        save_task = asyncio.create_task(self._save_image(ImagesApi(self._host.db_api_client()), frame, matching))
        self._pending_saves.append((save_task, frame.unique_id, reason))
        return save_task

    def collect_saves(self) -> list[str]:
        """Harvest finished saves; returns the unique_ids that are now processed.

        The driver uses the return value to grow its resume set — the collector has
        no opinion about what a frame identity means, only about whether the write
        succeeded.
        """
        completed: list[str] = []
        still_pending = []
        for task, uid, reason in self._pending_saves:
            if task.done():
                try:
                    filename = task.result()
                except Exception:
                    logger.exception("Background save task raised an exception")
                    continue
                if filename is None:
                    continue
                if uid:
                    completed.append(uid)
                self._images_saved += 1
                self._saved_by_reason[reason] = self._saved_by_reason.get(reason, 0) + 1
                self._log(f"Saved {filename} ({self._images_saved} total)")
            else:
                still_pending.append((task, uid, reason))
        self._pending_saves.clear()
        self._pending_saves.extend(still_pending)
        return completed

    async def drain_saves(self) -> list[str]:
        """Wait for every in-flight save, then collect the results."""
        if self._pending_saves:
            await asyncio.gather(*(t for t, _, _ in self._pending_saves), return_exceptions=True)
        return self.collect_saves()

    async def flush(self) -> list[str]:
        """Everything queued has been written by the time this returns.

        Used before a checkpoint: frames sitting in the dedup queue must be
        accounted for in the snapshot, or a resume re-collects them.
        """
        if self._dedup_queue is not None:
            await self._dedup_queue.join()
        return await self.drain_saves()

    def stop_condition_reached(self, elapsed_s: float) -> bool:
        """Has this collector's own budget run out?

        A property of the collection job rather than of the driver: for a
        watcher-attached collector this means "this collector stops after N", not
        "stop the watcher", which is why it lives on the spec.
        """
        if self._spec.max_images is not None and self._images_saved >= self._spec.max_images:
            return True
        if self._spec.max_duration_s is not None and elapsed_s >= self._spec.max_duration_s:
            return True
        return False

    def stop_condition_progress(self, elapsed_s: float) -> Optional[float]:
        """Progress (0-100) against whichever stop condition is set, or None
        (indeterminate) when neither is."""
        ratios: list[float] = []
        if self._spec.max_images:
            ratios.append(min(1.0, self._images_saved / self._spec.max_images))
        if self._spec.max_duration_s:
            ratios.append(min(1.0, elapsed_s / self._spec.max_duration_s))
        if not ratios:
            return None
        return max(ratios) * 100.0

    # ── inference and band/ambiguity evaluation ──────────────────────────────

    async def _run_inference(
        self,
        inference_api: InferenceApi,
        base64_img: str,
        pin_ids: dict[str, Optional[str]],
        provided: Optional[ImageWithProviderMetadata] = None,
    ) -> list:
        """Runs all configured models in parallel; returns detections meeting per-label confidence thresholds.

        When *provided* is passed, per-detection metadata — `{"model": str,
        "ambiguity": AmbiguityInfo|None}` keyed by bbox uuid — is stashed on the
        frame as `_det_meta`, the same convention `_filter_chain_image_tags`
        uses. Deliberately an out-parameter rather than a second return value:
        the return type is load-bearing for ~40 existing call sites, and this
        keeps a minor feature from rewriting all of them.

        Keyed on `det.bbox.uuid` — a uuid4 minted by the inference provider per
        detection — because that key survives the merge (which reorders and drops
        detections but never rewrites their objects), and because the generated
        client model forbids unknown attributes, so there is nowhere on the
        detection itself to hang this.

        Stashed per-FRAME, never on the collector: saves run concurrently, so
        instance state would be overwritten by the next frame before an in-flight
        save had read it.
        """
        spec = self._spec
        det_meta: dict[str, dict[str, Any]] = {}

        async def _fetch_batch() -> dict:
            """One request for the whole model set; returns name -> result (or None).

            Every model here runs over the SAME frame. Sent through
            `run_frame_inference` (apps/helpers/frameInference.py) so the server
            decodes and letterboxes it once and shares that across models that
            agree on input size and anchor, rather than once per model.

            Retries are per-batch rather than per-model: a timeout here means
            the inference server is not keeping up with the frame, which is not
            a condition one model recovers from while its siblings don't. A
            model that fails *individually* still comes back in `errors` and
            only costs its own detections.
            """
            request = FrameInferenceRequest(
                model_names=list(spec.model_names),
                conf_floor=spec.min_all_conf,
                pin_ids={n: p for n in spec.model_names if (p := pin_ids.get(n))},
                # Runner-up classes, asked for only when the confusable-detection
                # feature is on so every other run keeps today's response size.
                top_k=spec.ambiguity_top_k if spec.enable_ambiguity else 0,
            )
            t_start = time.perf_counter()  # PERF
            result = await run_frame_inference(
                inference_api,
                base64_img,
                [request],
                timeout_s=self.INFERENCE_TIMEOUT_S,
                max_retries=self.MAX_RETRIES,
                log=self._log,
            )
            # PERF BEGIN
            logger.debug(
                "Inference HTTP round-trip (batch of %d): %.1fms (%d detections total) "
                "[compare to processImage total= in inference_server log for HTTP overhead]",
                len(spec.model_names),
                1000 * (time.perf_counter() - t_start),
                sum(len(getattr(r, "detections", None) or []) for r in result.results.values()),
            )
            # PERF END
            return result.results

        def _filter_model(name: str, result) -> list:
            """Per-model post-processing of one model's own detections.

            Deliberately still per-model and still run BEFORE the cross-model
            merge: everything here is a statement about which model said what,
            and the merge blends the models together.
            """
            if result is None:
                return []

            # Only restrict to target labels here — NOT to each label's
            # confidence band. Band filtering must happen AFTER the
            # cross-model merge (see _run_inference): filtering first
            # would mean a high-confidence detection from one model no
            # longer suppresses an overlapping low-confidence/out-of-band
            # detection of the same object from another model, since the
            # merge would never even see the suppressed one (Phase 5B,
            # LIVE_COLLECTION_PLAN.md).
            # A model that architecturally cannot rank runners-up (NMS-free
            # /E2E export) must say so out loud — an empty alternates list
            # is indistinguishable from "nothing was confusable", and that
            # ambiguity is exactly how a multi-day run silently collects
            # nothing. Recorded once per model; surfaced by notices().
            if (
                spec.enable_ambiguity
                and result.alternates_available is False
                and name not in self._ambiguity_unavailable_models
            ):
                self._ambiguity_unavailable_models.add(name)
                self._log(
                    f"Model '{name}' cannot report runner-up classes (NMS-free/E2E export) — "
                    f"confusable detection is inactive for it"
                )

            dets = []
            for det in result.detections or []:
                luuid = det.bbox.label_uuid
                if luuid and luuid in spec.target_label_bands:
                    # Computed HERE — per model, on that model's own
                    # detections, before the cross-model merge. The merge
                    # drops losers and blends models together, so anything
                    # computed after it can no longer say WHICH model was
                    # confused, which is the only question this feature
                    # exists to answer.
                    det_meta[det.bbox.uuid] = {
                        "model": name,
                        "ambiguity": self._detect_ambiguity(det, name),
                    }
                    dets.append(det)
            return dets

        batch = await _fetch_batch()

        # Built by walking spec.model_names, NOT the response mapping: the merge
        # below is priority-ordered (first model wins an overlap), so
        # results_per_model[i] must correspond to model_names[i]. Reading the order
        # off the response would make which model wins depend on mapping order
        # across two serializers instead of on configured priority. A model missing
        # from the batch (it errored) contributes an empty list, leaving every other
        # model's position intact.
        results_per_model: list[list] = [_filter_model(name, batch.get(name)) for name in spec.model_names]

        # Cross-model disagreement is measured HERE, on the per-model lists, for
        # the same reason same-model ambiguity is measured inside _filter_model:
        # once merged there is no longer a "model-a said cat, model-b said dog"
        # to report, only a flat set of surviving boxes.
        if spec.enable_ambiguity and spec.ambiguity_cross_model and len(results_per_model) > 1:
            self._detect_cross_model_ambiguity(results_per_model, det_meta)

        # Flatten in priority order then apply cross-model hierarchy-aware IoU dedup.
        # This MUST happen before any confidence-band filtering — see the comment
        # in _filter_model above and _frame_triggers_save below.
        deduped = merge_detections_by_iou_and_hierarchy(
            results_per_model,
            label_ancestry=self._label_ancestry,
            overlap_iou_threshold=spec.overlap_iou_threshold,
            policy=spec.merge_policy,
        )

        # Phase 5B trigger/annotation split (LIVE_COLLECTION_PLAN.md). The two
        # bounds of a label's band do different jobs, and that asymmetry is the
        # whole feature:
        #
        #   min — the TAG floor. A detection at or above it gets an unverified
        #         bbox on any saved frame, however confident it is.
        #   max — the REVIEW ceiling. It only gates whether a frame is worth
        #         saving; it never suppresses a box on a frame saved anyway.
        #
        # So the TRIGGER (did any merged detection land inside its band?) decides
        # whether the frame is saved at all, and the ANNOTATION set is every
        # merged detection at or above its own label's min — deliberately NOT
        # capped at max, so a frame saved for a 0.4 dog still carries a box for
        # the 0.95 cat beside it rather than leaving a known true positive
        # unlabeled. If the trigger doesn't fire, nothing is saved and `matching`
        # is empty — same "no match" contract the caller already understands.
        annotatable = self._annotatable(deduped)
        # Two INDEPENDENT save triggers. The band asks "is this detection weak
        # enough to be worth a look?"; ambiguity asks "is the model, however
        # confident, unable to choose?". A run can use either or both, and a
        # frame that fires only the second is exactly the case the band was
        # structurally unable to catch.
        #
        # Ambiguity is checked against the ANNOTATABLE set, not the raw merged
        # one: a detection that won't be written as a box can't be the reason a
        # human is asked to look at the frame.
        kept_uuids = {d.bbox.uuid for d in annotatable}
        kept_meta = {uuid: meta for uuid, meta in det_meta.items() if uuid in kept_uuids}
        any_ambiguous = any(meta.get("ambiguity") is not None for meta in kept_meta.values())
        if not self._frame_triggers_save(deduped) and not any_ambiguous:
            if provided is not None:
                provided._det_meta = {}
            return []
        if provided is not None:
            provided._det_meta = kept_meta
        return annotatable

    def _frame_triggers_save(self, detections: list) -> bool:
        """True if any (post-merge) detection lands inside its label's confidence
        band — both bounds inclusive (`min <= confidence <= max`). Only this
        decides whether the frame is saved; it does not filter which boxes are
        written (see _annotatable)."""
        for det in detections:
            band = self._spec.target_label_bands.get(det.bbox.label_uuid)
            if band is not None and band.contains(det.confidence):
                return True
        return False

    def _labels_are_unrelated(self, a: Optional[str], b: Optional[str]) -> bool:
        """True when two labels are genuinely incompatible claims about one object.

        Same label is agreement. An ancestor/descendant pair ("Animal" vs "Cat")
        is *hierarchy*, not disagreement — the merge already treats it as one
        being a refinement of the other, and flagging it as confusion would fire
        on every correctly-working hierarchical model.
        """
        if not a or not b or a == b:
            return False
        if a in self._label_ancestry.get(b, ()):
            return False
        if b in self._label_ancestry.get(a, ()):
            return False
        return True

    def _detect_cross_model_ambiguity(self, results_per_model: list, det_meta: dict[str, dict[str, Any]]) -> None:
        """Flag objects two different models labelled incompatibly.

        Runs on the PRE-MERGE per-model lists, pairwise, so both sides can be
        named — after the merge there is no "which model said what" left to
        report. Unlike same-model ambiguity this needs no runner-up scores at
        all, so it is the only signal available for NMS-free/E2E models.

        Both detections are flagged, each naming the other: the merge keeps both
        boxes when their labels are unrelated, so both will be written to the
        image, and a reviewer who opens either one should see why it is marked.

        Mutates *det_meta* in place, never overwriting a same-model verdict —
        that one is the more direct statement about a specific model's own
        decision boundary, so it wins where both apply.

        ⚠️ Cost: this is O(dets_a × dets_b) per model pair, on top of the merge's
        own quadratic pass. `conf_thresh` sent to the network is `min_all_conf`
        — the LOOSEST floor across every configured label — so the raw per-model
        lists can be several times larger than the set that will actually be
        kept. Hence the prefilter below: it is the difference between scanning
        what the network returned and scanning what the run is actually
        collecting, and the saving is quadratic. (A driver that SHARES inference
        with a watcher hands over a list sized by the watcher's 0.1 floor, which
        makes this prefilter mandatory rather than merely worthwhile.)
        """
        spec = self._spec
        # Restrict to detections that will survive _annotatable (at or above
        # their OWN label's floor). Two reasons, and the second is correctness,
        # not speed:
        #   1. Quadratic saving — see the cost note above.
        #   2. A below-floor detection is never written as a box. Letting one act
        #      as a "rival" would produce a flag naming a box the reviewer cannot
        #      see on the image, and would count a detection the config says is
        #      not worth collecting as evidence of disagreement.
        # The MERGE still receives the unfiltered lists: there a below-floor
        # detection must remain able to suppress an overlapping duplicate.
        candidates: list[list] = []
        for dets in results_per_model:
            kept = []
            for det in dets:
                band = spec.target_label_bands.get(det.bbox.label_uuid)
                if band is not None and det.confidence >= band.min_confidence:
                    kept.append(det)
            candidates.append(kept)

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                for det_a in candidates[i]:
                    for det_b in candidates[j]:
                        if not self._labels_are_unrelated(det_a.bbox.label_uuid, det_b.bbox.label_uuid):
                            continue
                        if iou_xywh(det_a.bbox, det_b.bbox) < spec.overlap_iou_threshold:
                            continue
                        # Order by confidence so "winner"/"rival" mean the same
                        # thing here as in the same-model path.
                        hi, lo = (det_a, det_b) if det_a.confidence >= det_b.confidence else (det_b, det_a)
                        value = ambiguity_value(spec.ambiguity_metric, hi.confidence, lo.confidence)
                        self._ambiguity_evaluated += 1
                        self._ambiguity_samples.append((det_meta.get(hi.bbox.uuid, {}).get("model", "?"), value))
                        if not ambiguity_fires(spec.ambiguity_metric, value, spec.ambiguity_threshold):
                            continue
                        # Both boxes carry the verdict — a reviewer opening either one
                        # must see the disagreement — but it is *one* disagreement, and
                        # counting it per box reported more flags than pairs measured
                        # and listed every pair twice, once per direction.
                        keys = [
                            key
                            for key in (
                                self._record_cross_model_flag(hi, lo, value, det_meta),
                                self._record_cross_model_flag(lo, hi, value, det_meta),
                            )
                            if key is not None
                        ]
                        if keys:
                            self._ambiguity_flagged += 1
                            # keys[0] is the higher-confidence side whenever it passed
                            # the gates, so the summary attributes the pair to the model
                            # that actually won it.
                            self._ambiguity_pairs[keys[0]] = self._ambiguity_pairs.get(keys[0], 0) + 1

    def _record_cross_model_flag(
        self, winner, rival, value: float, det_meta: dict[str, dict[str, Any]]
    ) -> Optional[tuple[str, str, str]]:
        """Attach one cross-model verdict to *winner*, respecting the same gates
        the same-model path applies (review_disabled, the confidence floor).

        Returns the pair key for the caller to count, or None if a gate rejected
        this side. Counting is the caller's job because both directions describe
        one disagreement — see the note there.
        """
        meta = det_meta.get(winner.bbox.uuid)
        if meta is None or meta.get("ambiguity") is not None:
            return None  # unknown detection, or same-model already had its say
        band = self._spec.target_label_bands.get(winner.bbox.label_uuid)
        if band is None or band.review_disabled:
            return None
        if winner.confidence < self._spec.ambiguity_min_confidence:
            return None
        winner_model = meta.get("model") or "?"
        rival_model = (det_meta.get(rival.bbox.uuid) or {}).get("model")
        meta["ambiguity"] = AmbiguityInfo(
            kind=AMBIGUITY_KIND_CROSS_MODEL,
            metric=self._spec.ambiguity_metric,
            value=value,
            model=winner_model,
            rival_model=rival_model,
            rival_label_uuid=rival.bbox.label_uuid,
            rival_class_str=rival.bbox.class_str,
            rival_confidence=rival.confidence,
        )
        return (winner_model, winner.bbox.label_uuid or "?", rival.bbox.label_uuid or "?")

    def _label_name(self, label_uuid: str) -> str:
        """Human-readable label name for status text, falling back to whatever
        identifier we have (a raw class string, or the uuid) so a summary row is
        never blank."""
        meta = self._label_meta.get(label_uuid) if self._label_meta else None
        if meta and meta.get("text"):
            return meta["text"]
        return label_uuid

    def _detect_ambiguity(self, det, model_name: str) -> Optional[AmbiguityInfo]:
        """Is *det* one this model couldn't decide? Returns why, or None.

        Called per model on that model's own pre-merge detections (see
        _run_inference). Reads `det.alternates` — the runner-up classes the
        network already scored and postprocessing would otherwise discard.
        """
        spec = self._spec
        if not spec.enable_ambiguity:
            return None
        alternates = getattr(det, "alternates", None) or ()
        if not alternates:
            return None

        band = spec.target_label_bands.get(det.bbox.label_uuid)
        if band is None:
            return None
        # A label the user explicitly told us must never pull a frame into review
        # ("Review up to = 0") doesn't get to do it through this door either. The
        # WINNER's band decides: the runner-up's is irrelevant, or "whose setting
        # applies?" has no defensible answer.
        if band.review_disabled:
            return None
        # Skip what the confidence band is already collecting, so a run using both
        # triggers doesn't re-flag its own weak detections as "confusable" noise.
        if det.confidence < spec.ambiguity_min_confidence:
            return None

        best: Optional[AmbiguityInfo] = None
        for alt in alternates:
            in_scope = spec.ambiguity_scope == "any" or (
                alt.label_uuid is not None and alt.label_uuid in spec.target_label_bands
            )
            if not in_scope:
                continue
            value = ambiguity_value(spec.ambiguity_metric, det.confidence, alt.confidence)
            # Alternates arrive best-first, so the first in-scope one is the
            # strongest rival — sample it (threshold-independently, for the
            # calibration stats) and stop looking.
            self._ambiguity_evaluated += 1
            self._ambiguity_samples.append((model_name, value))
            if ambiguity_fires(spec.ambiguity_metric, value, spec.ambiguity_threshold):
                best = AmbiguityInfo(
                    metric=spec.ambiguity_metric,
                    value=value,
                    model=model_name,
                    rival_label_uuid=alt.label_uuid,
                    rival_class_str=alt.class_str,
                    rival_confidence=alt.confidence,
                )
            break

        if best is not None:
            self._ambiguity_flagged += 1
            key = (model_name, det.bbox.label_uuid or "?", best.rival_label_uuid or best.rival_class_str or "?")
            self._ambiguity_pairs[key] = self._ambiguity_pairs.get(key, 0) + 1
        return best

    def _annotatable(self, detections: list) -> list:
        """The (post-merge) detections that get written as unverified bboxes:
        those at or above their OWN label's min. The band's max is not applied
        here — see _run_inference for why the ceiling is review-only.

        The per-label floor matters because the network-level `conf_thresh` is
        `min_all_conf`, the loosest min across every configured label. Without
        this check, a label configured at min 0.5 would still collect boxes down
        to a *different* label's 0.15 floor, which is not what the config says.

        Must run AFTER the cross-model merge: filtering first would stop a
        below-floor detection from suppressing an overlapping duplicate of the
        same object (the same ordering constraint _frame_triggers_save has)."""
        annotatable = []
        for det in detections:
            band = self._spec.target_label_bands.get(det.bbox.label_uuid)
            if band is not None and det.confidence >= band.min_confidence:
                annotatable.append(det)
        return annotatable

    # ── filter chain ─────────────────────────────────────────────────────────

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
        # An empty `matching` normally means "frame not saved anyway", so the chain is
        # skipped as pure wasted inference. In collect-everything mode an empty list is
        # the normal case for a frame that IS about to be saved, so the chain must still
        # run — its image-level steps (discard_image / tag_image) are the only filtering
        # such a run has.
        if self._filter_chain_runner is None or (not matching and not self._spec.collect_all):
            return matching, False, False

        # base64_img is the JSON packet produced by base64_encode_jpg (imageUtils.py)
        # — `{"format": ..., "data": <b64>, "metadata": {}}` — the same "universal
        # reader" packet shape base64_decode_image() unwraps on the inference-server
        # side, not a bare base64 string. Fall back to treating it as raw base64 for
        # any caller that already passes an unwrapped string.
        try:
            parsed = json.loads(base64_img)
            b64_data = parsed["data"] if isinstance(parsed, dict) else base64_img
        except json.JSONDecodeError:
            b64_data = base64_img
        image_bytes = base64.b64decode(b64_data)
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

    # ── saving ───────────────────────────────────────────────────────────────

    def _build_seed_bbox(self, det, bbox_uuid: str) -> dict[str, Any]:
        """A detection → the plain-object bbox shape `seedRuntimeBboxes()`
        (canvas/image.js) expects: it feeds each entry straight into
        `createUIBBbox(seed)`, which fills in any UIBBox field we don't set
        (uuid/tags/selected/dirty) with its own defaults.

        `label` must be `{uuid, text, color}` or `None` — `BboxView` reads
        `label?.text`/`label?.color` directly, not a bare uuid — so a bare
        label_uuid resolves to name/color via `self._label_meta` (fetched once at
        prepare(); see build_label_metadata_map) rather than a DB round trip here.
        """
        label_uuid = det.bbox.label_uuid
        label = None
        if label_uuid:
            meta = self._label_meta.get(label_uuid, {})
            label = {"uuid": label_uuid, "text": meta.get("text") or label_uuid, "color": meta.get("color")}
        return {
            "uuid": bbox_uuid,
            "label": label,
            "x": det.bbox.x,
            "y": det.bbox.y,
            "width": det.bbox.width,
            "height": det.bbox.height,
        }

    async def _save_image(self, images_api: ImagesApi, frame: FrameOffer, matching: list) -> Optional[str]:
        """Writes image to disk and persists bounding-box metadata. Returns filename on success, None on failure."""
        provided = frame.provided
        provider_type = frame.provider_type
        t_save_start = time.perf_counter()  # PERF
        source_name = ImageHelper.sanitize_filename(provided.metadata.source or "unknown")
        frame_idx = provided.metadata.frame_idx or 0
        if provider_type == ImageProviderType.FIXED:
            # Stable naming preserved exactly: resume-by-processed_ids depends on it.
            filename = f"{source_name}_{frame_idx}.jpg"
        else:
            # Live providers (LIVE_SNAPSHOT / LIVE_STREAM) reset frame_idx to 0 on every
            # restart, so frame_idx alone would overwrite earlier files across restarts.
            # A UTC timestamp (second resolution) makes the filename unique across
            # restarts; frame_idx already guarantees uniqueness within a run.
            utc_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            filename = f"{source_name}_{utc_stamp}_{frame_idx}.jpg"
        rel_path = str(Path(self._spec.output_dir) / filename)
        abs_path = self._files_root / self._spec.output_dir / filename
        await asyncio.to_thread(abs_path.parent.mkdir, parents=True, exist_ok=True)

        # cv2.imwrite is CPU-bound + I/O; run off the event loop.
        # Save as JPEG (faster write, smaller files; lossless PNG not needed for
        # training data). `provided.image` is the FULL-RESOLUTION frame — never an
        # analysis downscale. See the module docstring.
        t_write = time.perf_counter()  # PERF
        if not await asyncio.to_thread(
            cv2.imwrite,
            str(abs_path),
            provided.image,
            [cv2.IMWRITE_JPEG_QUALITY, PHASH_JPEG_QUALITY],
        ):
            logger.warning(f"Failed to write image to {abs_path}")
            self._log(f"Failed to write image: {filename}")
            return None
        logger.debug("Image write: %.1fms → %s", 1000 * (time.perf_counter() - t_write), abs_path)  # PERF

        boxes = []
        seed_bboxes: list[dict[str, Any]] = []
        det_meta: dict[str, dict[str, Any]] = getattr(provided, "_det_meta", None) or {}
        # Provenance a driver stashed on the frame (e.g. a watcher's event-scoped
        # collect_training_images action: which rule, when the event opened —
        # see apps.helpers.actions.collectTrainingImagesAction). Stringified like
        # every other bbox_meta value (metadata_json values are typed str) and
        # merged into EVERY box for the same reason source_model is: the frame
        # this ends up on may carry no ambiguity info, but the question "which
        # rule collected this?" is still answerable from any of its boxes.
        provenance_meta: dict[str, str] = {str(k): str(v) for k, v in (provided.metadata.additional_info or {}).items()}
        for det in matching:
            extra_tags = list(self._extra_tag_uuids)
            this_meta = det_meta.get(det.bbox.uuid) or {}
            ambiguity: Optional[AmbiguityInfo] = this_meta.get("ambiguity")
            # Filter-chain tag_bbox matches (Phase 4) stash extra tags directly on
            # the matched detection's own bbox.tag_uuids — fold them in here.
            # (tag_image is image-level — applied via set_image_tag below instead.)
            filter_chain_tags = list(det.bbox.tag_uuids or [])
            bbox_uuid = str(uuid4())
            # metadata_json is a raw JSON blob whose inner keys are NOT run
            # through the camelCase alias generator — these stay snake_case on
            # both sides (see shared/domain/image/confidence.js).
            # A detector-miss box (plan §4A) has no model score behind it — a
            # synthetic confidence would let it sort/filter alongside real
            # detections while lying about where it came from. `None` is
            # already a first-class state (`confidence_in_band`'s
            # `include_missing`), so neither metadata key is written at all.
            bbox_meta: dict[str, Any] = {} if det.confidence is None else {"confidence": str(det.confidence)}
            # Written on EVERY box, not just confusable ones: a multi-model run
            # was previously an untraceable mix, and "which model produced this?"
            # is a precondition for answering "which model is confused?".
            source_model = this_meta.get("model")
            if source_model:
                bbox_meta["source_model"] = source_model
            if provenance_meta:
                bbox_meta.update(provenance_meta)
            ambiguity_tags: list[str] = []
            if ambiguity is not None:
                bbox_meta.update(ambiguity.as_metadata())
                # The full runner-up list is persisted even though nothing renders
                # it yet: it costs ~100 bytes on an already-flagged box, and
                # without it a finished multi-day run can't be revisited to ask
                # *what* it was confused with. JSON-encoded into a string because
                # metadata_json values are typed `str` (see as_metadata) — readers
                # json.loads / JSON.parse it.
                bbox_meta["alternates"] = json.dumps(
                    [
                        {
                            "label_uuid": alt.label_uuid,
                            "class_str": alt.class_str,
                            "confidence": round(alt.confidence, 4),
                        }
                        for alt in (getattr(det, "alternates", None) or ())
                    ]
                )
                ambiguity_tags.append(CONFUSABLE_TAG_UUID)
            boxes.append(
                BoundingBoxMetadataModel(
                    bbox_uuid=bbox_uuid,
                    x=det.bbox.x,
                    y=det.bbox.y,
                    width=det.bbox.width,
                    height=det.bbox.height,
                    label_uuid=det.bbox.label_uuid,
                    class_str=det.bbox.class_str,
                    tag_uuids=[self._unverified_tag_uuid, *extra_tags, *filter_chain_tags, *ambiguity_tags],
                    # ⚠️ `bbox_meta`, NOT `extra`. The server model declares
                    # AliasChoices("bbox_meta", "extra") and the OpenAPI
                    # generator takes the first choice as the generated field
                    # name. Pydantic ignores unknown kwargs, so `extra=` was
                    # accepted, silently dropped, and every bbox this task wrote
                    # landed with metadata_json = {}.
                    bbox_meta=bbox_meta,
                )
            )
            seed_bboxes.append(self._build_seed_bbox(det, bbox_uuid))

        try:
            t_meta = time.perf_counter()  # PERF
            # Serialise DB writes to avoid SQLite write-lock contention between concurrent saves.
            async with self._meta_write_lock:
                await asyncio.to_thread(
                    images_api.update_image_metadata,
                    UpdateMetadataPayload(
                        image_path=rel_path,
                        boxes=boxes,
                        source_uuid=provided.metadata.source_uuid,
                    ),
                )
            logger.debug(
                "Metadata save: %.1fms for %s",
                1000 * (time.perf_counter() - t_meta),
                filename,
            )  # PERF
            # Status-panel "last capture" widget (Phase 4) — by reference only (path +
            # bbox descriptors), never image bytes. Set only on a successful save, and
            # only here (not in the dedup/discard paths), so the panel always reflects
            # a frame that actually made it into the dataset. Racy across concurrent
            # save tasks (last one to finish wins) — acceptable for a "most recent"
            # display, and avoids adding a lock to the hot save path.
            self._last_capture = {"path": rel_path, "bboxes": seed_bboxes}
        except Exception:
            logger.exception(f"Failed to save metadata for {rel_path}")
            self._log(f"Failed to save metadata: {filename}")
            return None

        # Image-level tags, applied now that the image record exists. Two sources:
        #   - self._extra_tag_uuids — the generated task tag plus any explicit
        #     apply_tags. "This run collected this frame" is a property of the
        #     image, so it belongs here as well as on each bbox above. Bbox-only
        #     was silently a no-op for collect-everything runs (no models → no
        #     bboxes → nothing to hang the tag on), which left the task's own
        #     "See results" filter matching nothing.
        #   - _filter_chain_image_tags — tag_image matches stashed by
        #     _apply_filter_chain_to_frame.
        # Deduplicated (a chain may apply a tag the task already applies) while
        # preserving order, so each tag is written exactly once.
        filter_chain_image_tags = getattr(provided, "_filter_chain_image_tags", None) or ()
        image_tag_uuids = list(dict.fromkeys([*self._extra_tag_uuids, *filter_chain_image_tags]))
        for tag_uuid in image_tag_uuids:
            try:
                await asyncio.to_thread(
                    images_api.set_image_tag,
                    SetImageTagPayload(image_path=rel_path, tag_uuid=tag_uuid, action="add"),
                )
            except Exception:
                logger.exception(f"Failed to apply image tag {tag_uuid} to {rel_path}")

        logger.debug(
            "_save_image total: %.1fms for %s",
            1000 * (time.perf_counter() - t_save_start),
            filename,
        )  # PERF
        return filename

    # ── deduplication ────────────────────────────────────────────────────────

    async def _dedup_worker(self, images_api: ImagesApi) -> None:
        """Consume matched frames one at a time, run bulk dedup, dispatch saves.

        Runs as a single background coroutine so dedup is serialised (no thread-pool
        explosion) while still pipelining one frame behind the caller's inference.
        Shuts down when it receives None from the queue.
        """
        from db_client.models.bbox_dedup_item import BboxDedupItem
        from db_client.models.check_dedup_bulk_payload import CheckDedupBulkPayload

        spec = self._spec
        assert self._dedup_queue is not None
        while True:
            item = await self._dedup_queue.get()
            try:
                if item is None:
                    break  # sentinel — the driver is done

                frame, matching, reason = item
                provided = frame.provided

                # When action is 'ignore', skip dedup entirely and save without any tagging.
                if spec.duplicate_action == "ignore":
                    self._dispatch_save(frame, matching, reason)
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
                        image_threshold=spec.image_dup_threshold,
                        bboxes=[
                            BboxDedupItem(
                                phash=bp,
                                label_uuid=det.bbox.label_uuid,
                                threshold=spec.bbox_dup_threshold,
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
                # Threshold-independent distances feed the calibration widgets. `None`
                # (empty index / unlabelled bbox) means "no comparison was possible" and
                # is kept as None rather than coerced to a number — bucketing it as a
                # large distance would fabricate "these frames are unique" evidence out
                # of the first frames of a run, when the index simply had nothing in it.
                self._dedup_samples.append(
                    (
                        resp.image_min_distance,
                        list(resp.bbox_min_distances) if resp.bbox_min_distances else [],
                    )
                )
                logger.debug(
                    "dedup result: image_is_dup=%s bbox_flags=%s (img_thresh=%d bbox_thresh=%d)",
                    resp.image_is_dup,
                    bbox_flags,
                    spec.image_dup_threshold,
                    spec.bbox_dup_threshold,
                )

                # The "measure" (calibration) action matches neither discard branch below,
                # so it falls through to the save — by design. It pays the full hashing and
                # index-query cost purely to populate the distance sample above, letting the
                # sweep report what each threshold *would* have done without any frame
                # actually being dropped on a guessed threshold.
                #
                # Image-level discard: skip the frame when the whole scene is a perceptual duplicate.
                if spec.duplicate_action == "discard_image" and resp.image_is_dup:
                    self._record_discard("image-duplicate", provided)
                    continue

                # Bbox-level discard: keep if any detection is new; discard only if all are duplicates.
                # Gate on image_is_dup too — bbox hashes can collide across unrelated scenes
                # (same logic as the 'tag' action below, which uses image_is_dup AND bbox_flag).
                if spec.duplicate_action == "discard" and resp.image_is_dup and all(bbox_flags):
                    self._record_discard("all-duplicate", provided)
                    continue

                save_task = self._dispatch_save(frame, matching, reason)
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

    def _record_discard(self, kind: str, provided) -> None:
        self._dup_frames_discarded += 1
        logger.debug("Discarding %s frame: %s", kind, provided.metadata.source)
        if self._dup_frames_discarded == 1:
            self._log(f"Dedup: discarding first {kind} frame")
        elif self._dup_frames_discarded % 50 == 0:
            self._log(f"Dedup: {self._dup_frames_discarded} duplicate frames discarded so far")

    # ── model pins ───────────────────────────────────────────────────────────

    async def _renew_pins_periodically(self, models_api: ModelsApi) -> None:
        """Background loop: re-pins every held model roughly every
        PIN_RENEWAL_INTERVAL_S (~half of PIN_DURATION_S) so a run longer than the
        pin duration never lets a pin lapse and silently reload the model mid-stream.

        Runs as its own task, independent of the capture loop, so a slow/paced
        source can't delay a renewal and a renewal failure can't stall a frame.
        Renewal failures are logged and skipped — never raised — so a transient
        inference-server hiccup can't kill an otherwise-healthy long-running task.

        `_pin_ids` and `_chain_pin_ids` are mutated in place (new pin_id swapped in)
        so `_run_inference` and the filter-chain backend, which hold references to
        these same dicts, immediately see the renewed pin.
        """
        try:
            while True:
                try:
                    await asyncio.wait_for(self._host.stop_event.wait(), timeout=self.PIN_RENEWAL_INTERVAL_S)
                    return  # stopping — shutdown() will unpin
                except asyncio.TimeoutError:
                    pass  # timer elapsed — time to renew

                for name, pid in list(self._pin_ids.items()):
                    if not pid:
                        continue
                    new_pid = await self._renew_one_pin(models_api, name, pid, label="model")
                    if new_pid:
                        self._pin_ids[name] = new_pid
                        self._combined_pin_ids[name] = new_pid

                for name, pid in list(self._chain_pin_ids.items()):
                    if not pid:
                        continue
                    new_pid = await self._renew_one_pin(models_api, name, pid, label="filter-chain model")
                    if new_pid:
                        self._chain_pin_ids[name] = new_pid
                        self._combined_pin_ids[name] = new_pid
        except asyncio.CancelledError:
            raise

    async def _renew_one_pin(self, models_api: ModelsApi, name: str, old_pid: str, label: str) -> Optional[str]:
        """Re-pin `name`, unpin the old pin_id, and return the new pin_id (or None on
        any failure, in which case the old pin is left alone and will keep working
        until it naturally expires)."""
        try:
            resp = await asyncio.to_thread(
                models_api.pin_model_api_api_models_model_name_pin_post,
                name,
                PinModelPayload(model_name=name, duration=self.PIN_DURATION_S),
            )
        except Exception:
            logger.exception(f"Failed to renew pin for {label} '{name}'")
            return None
        new_pid = resp.pin_id
        if not new_pid:
            logger.warning(f"Pin renewal for {label} '{name}' returned no pin_id")
            return None
        self._log(f"Renewed pin for {label}: {name}")
        try:
            await asyncio.to_thread(models_api.unpin_model, name, old_pid)
        except Exception:
            logger.warning(f"Failed to unpin stale pin for {label} '{name}' (pin {old_pid})")
        return new_pid

    # ── dedup threshold calibration ──────────────────────────────────────────

    @staticmethod
    def _percentile(values: list[int], pct: float) -> Optional[int]:
        """Nearest-rank percentile of an already-unsorted int list, or None if empty."""
        if not values:
            return None
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * (len(ordered) - 1)))))
        return ordered[idx]

    def _sweep_action(self) -> str:
        """Which discard rule the sweep tables model.

        The sweep answers a counterfactual — "what would threshold N have
        discarded?" — so it cannot use the configured action when that action is
        a pass-through one: "measure" never discards, which pinned the entire
        "Would discard" column at 0 in exactly the calibration mode the sweep
        exists to serve. When the run is only measuring, model the schema's
        default discard rule instead; _dedup_hint_text() names whichever is used
        so the numbers are never unattributed.
        """
        if self._spec.duplicate_action in self._DISCARD_ACTIONS:
            return self._spec.duplicate_action
        return "discard"

    def _would_discard(
        self,
        image_dist: Optional[int],
        bbox_dists: list,
        image_t: int,
        bbox_t: int,
        action: Optional[str] = None,
    ) -> bool:
        """Replay a duplicate action against one sampled frame.

        *action* defaults to the configured one (what _dedup_worker is doing right
        now); the sweep passes _sweep_action() to model a discard rule even while
        the run is merely measuring.

        Mirrors the branch structure in _dedup_worker exactly — including the fact
        that the 'discard' (keep-if-any-unique) action gates on image_is_dup as well
        as on every bbox being a duplicate. If those rules ever diverge the sweep
        silently starts lying, so they are asserted together in the unit tests.
        """
        action = action if action is not None else self._spec.duplicate_action
        image_is_dup = image_dist is not None and image_dist <= image_t
        if action == "discard_image":
            return image_is_dup
        if action == "discard":
            # A frame with no detections can't have "all detections duplicated";
            # _dedup_worker's all([]) == True would discard it, so match that.
            all_bbox_dup = all(d is not None and d <= bbox_t for d in bbox_dists)
            return image_is_dup and all_bbox_dup
        # "ignore" and "measure" never discard.
        return False

    def _dedup_histogram(self, values: list[int], marker: int, bucket: int = 1) -> dict[str, Any]:
        """Bucket nearest-neighbour distances into a histogram payload for the widget."""
        if not values:
            return {"bins": [], "total": 0}
        top = max(max(values), marker) + 1
        bins: list[dict[str, Any]] = []
        for start in range(0, top, bucket):
            end = start + bucket - 1
            count = sum(1 for v in values if start <= v <= end)
            bins.append(
                {
                    "label": str(start) if bucket == 1 else f"{start}–{end}",
                    "count": count,
                    # The marked bin is where the configured threshold currently cuts:
                    # everything at or left of it is treated as a duplicate.
                    "marked": start <= marker <= end,
                }
            )
        return {"bins": bins, "total": len(values)}

    def _dedup_sweep_rows(self, thresholds: tuple[int, ...], vary: str) -> list[dict[str, Any]]:
        """Counterfactual: what each candidate threshold would have done to the sample.

        *vary* is "image" or "bbox" — the other threshold is held at its configured
        value, because the two hashes have different widths and sweeping them
        together would compare distances that aren't on the same scale.
        """
        samples = list(self._dedup_samples)
        if not samples:
            return []
        action = self._sweep_action()
        # The configured value always gets a row, even if it isn't on the ladder —
        # without it there's nothing to compare the other rows against.
        configured = self._spec.image_dup_threshold if vary == "image" else self._spec.bbox_dup_threshold
        sweep = sorted(set(thresholds) | {configured})
        rows: list[dict[str, Any]] = []
        for t in sweep:
            image_t = t if vary == "image" else self._spec.image_dup_threshold
            bbox_t = t if vary == "bbox" else self._spec.bbox_dup_threshold
            discarded = sum(1 for img, bbs in samples if self._would_discard(img, bbs, image_t, bbox_t, action))
            if vary == "image":
                matched = sum(1 for img, _ in samples if img is not None and img <= t)
            else:
                matched = sum(1 for _, bbs in samples for d in bbs if d is not None and d <= t)
            current = t == configured
            rows.append(
                {
                    "threshold": f"{t} ←current" if current else str(t),
                    "matched": str(matched),
                    "discarded": f"{discarded} ({100.0 * discarded / len(samples):.0f}%)",
                    "kept": f"{len(samples) - discarded} ({100.0 * (len(samples) - discarded) / len(samples):.0f}%)",
                }
            )
        return rows

    def _dedup_hint_text(self) -> str:
        """One-paragraph explanation of how to read the calibration widgets."""
        sample_n = len(self._dedup_samples)
        if sample_n == 0:
            return (
                "No frames measured yet. Once frames start arriving, the tables below show what "
                "each candidate threshold would have done to them."
            )
        action = self._sweep_action()
        if action == "discard_image":
            rule = (
                'The sweep applies "Discard if image matches", so every matching frame is a '
                "discard — the two columns agree by definition."
            )
        else:
            rule = (
                'The sweep applies "Keep if any unique", which drops a frame only when the image '
                "matches AND every detection in it also matches. That is why "
                '"Would discard" can be lower than "Frames matching".'
            )
        base = (
            f"Based on the last {sample_n} checked frame(s). Distance is Hamming distance to the "
            "closest already-stored image (or same-label bbox crop): 0 means pixel-identical after "
            "hashing, and bigger numbers mean more different. A threshold counts every frame at that "
            "distance or less as a duplicate. "
            # Deliberately no "above"/"below": the histogram runs 0 upward, so
            # "below" means a smaller number but a row further down the list —
            # the two readings point in opposite directions.
            "To choose one, find the stretch of empty or near-empty rows in the histogram. Small "
            "distances on the 0 side of that stretch are repeats of scenes already collected; the "
            "bigger distances past it are genuinely new. Set the threshold to the largest distance "
            "that is still on the repeats side — the last populated row before the empty stretch. "
            "If there is no clear empty stretch, the scene is changing continuously and any cut-off "
            "is a judgement call: start low and raise it until the discard rate looks right. " + rule
        )
        if self._spec.duplicate_action == "measure":
            return (
                base + " Calibration mode is on: nothing is actually being discarded, so this sample "
                "is unbiased and the discard columns are hypothetical."
            )
        if self._spec.duplicate_action == "ignore":
            return base + " Duplicates are being ignored, so the discard columns are hypothetical."
        if self._spec.duplicate_action in ("discard", "discard_image"):
            return (
                base + " Note: frames are currently being discarded, so discarded frames never entered "
                "the index and the measured distances drift upward. For an unbiased sample, re-run with "
                'Action on Duplicate set to "Measure only (calibration)".'
            )
        return base

    # ── reporting ────────────────────────────────────────────────────────────

    def status_stats(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """`(stats, calibration_widgets)` for the driver's status panel.

        Reads only in-memory counters — no I/O, no awaits — because the driver
        calls this synchronously from a poll handler.
        """
        spec = self._spec
        stats: dict[str, Any] = {"offered": self._frames_offered, "saved": self._images_saved}
        calibration: dict[str, Any] = {}
        # Saved-by-reason subtotals (plan §6). Only shown once a non-band reason has
        # actually produced something: for the standalone task every save is a band
        # save, so the split would be three rows restating `saved`.
        special = self._saved_by_reason.get(SAVE_REASON_DETECTOR_MISS, 0) + self._saved_by_reason.get(
            SAVE_REASON_TRACK_WORST, 0
        )
        if special:
            stats["saved_band"] = self._saved_by_reason.get(SAVE_REASON_BAND, 0)
            stats["saved_detector_miss"] = self._saved_by_reason.get(SAVE_REASON_DETECTOR_MISS, 0)
            stats["saved_track_worst"] = self._saved_by_reason.get(SAVE_REASON_TRACK_WORST, 0)
        # Only present for a watcher-attached collector — the standalone task never
        # populates `frame.tracks`, so these would otherwise be permanent zeros
        # cluttering the panel of the thing this feature doesn't apply to.
        if self._track_novelty_kept or self._track_novelty_dropped or self._track_worst_committed:
            stats["track_novelty_kept"] = self._track_novelty_kept
            stats["track_novelty_dropped"] = self._track_novelty_dropped
            stats["track_worst_committed"] = self._track_worst_committed
        if self._detector_miss_offered:
            stats["detector_miss_offered"] = self._detector_miss_offered
        if spec.enable_dedup:
            stats["dedup_checked"] = self._dedup_frames_checked
            stats["dedup_image_hits"] = self._dedup_image_hits
            stats["dedup_bbox_hits"] = self._dedup_bbox_hits
            stats["dedup_discarded"] = self._dup_frames_discarded

            samples = list(self._dedup_samples)
            image_dists = [img for img, _ in samples if img is not None]
            bbox_dists = [d for _, bbs in samples for d in bbs if d is not None]
            stats["dedup_sample_size"] = len(samples)
            for key, values, pct in (
                ("dedup_image_dist_median", image_dists, 50),
                ("dedup_image_dist_p10", image_dists, 10),
                ("dedup_bbox_dist_median", bbox_dists, 50),
            ):
                value = self._percentile(values, pct)
                if value is not None:
                    stats[key] = value
            calibration = {
                "dedup_hint": self._dedup_hint_text(),
                "dedup_image_hist": self._dedup_histogram(image_dists, spec.image_dup_threshold),
                "dedup_bbox_hist": self._dedup_histogram(bbox_dists, spec.bbox_dup_threshold, bucket=2),
                "dedup_image_sweep": self._dedup_sweep_rows(self.IMAGE_SWEEP_THRESHOLDS, "image"),
                "dedup_bbox_sweep": self._dedup_sweep_rows(self.BBOX_SWEEP_THRESHOLDS, "bbox"),
            }
        if spec.enable_ambiguity:
            # Deliberately plain scalars in the existing stats dict rather than a
            # new widget: the panel already renders this as key/value, so live
            # threshold feedback costs no UI. Without *some* visibility the
            # threshold is a blind guess that tags everything or nothing.
            stats["confusable_flagged"] = self._ambiguity_flagged
            stats["confusable_measured"] = self._ambiguity_evaluated
            values = [v for _, v in self._ambiguity_samples]
            if values:
                ordered = sorted(values)
                stats["confusable_median_closeness"] = round(ordered[len(ordered) // 2], 3)
                # The extreme end says how close the stream ever actually gets —
                # if even the most confusable detection is far from the threshold,
                # the threshold is simply unreachable and nothing will ever flag.
                extreme = max(values) if spec.ambiguity_metric == AMBIGUITY_METRIC_RATIO else min(values)
                stats["confusable_most_extreme"] = round(extreme, 3)
            if self._ambiguity_unavailable_models:
                stats["confusable_unsupported_models"] = ", ".join(sorted(self._ambiguity_unavailable_models))
            top_pairs = sorted(self._ambiguity_pairs.items(), key=lambda kv: kv[1], reverse=True)[:3]
            if top_pairs:
                stats["confusable_top_pairs"] = "; ".join(
                    f"{model}: {self._label_name(win)}↔{self._label_name(rival)} ×{count}"
                    for (model, win, rival), count in top_pairs
                )

        if self._filter_chain_runner is not None:
            stats["filter_discarded"] = self._filtered_discarded
            stats["filter_relabeled"] = self._filtered_relabeled
            stats["filter_tagged"] = self._filtered_tagged
            stats["filter_stripped"] = self._filtered_stripped

        if self._frames_dropped_behind:
            stats["dropped_behind"] = self._frames_dropped_behind

        return stats, calibration

    def notices(self) -> list[dict]:
        """The silent-failure guard for confusable detection.

        A model exported NMS-free/end-to-end argmaxes inside its own graph, so it
        can never report a runner-up — and the symptom is simply that nothing is
        ever flagged, which is indistinguishable from "nothing was confusable"
        until a multi-day run has already finished. Warn as soon as we know.
        """
        return [
            {
                "key": f"ambiguity_unsupported_{model_name}",
                "level": "warning",
                "message": (
                    f"'{model_name}' cannot report runner-up classes (NMS-free/end-to-end export), "
                    f"so no confusable detections will be found for it. Re-export the model without "
                    f"end-to-end mode, or pair it with another model."
                ),
            }
            for model_name in sorted(self._ambiguity_unavailable_models)
        ]

    def review_actions(self) -> list[dict]:
        """Ways into Review over this collector's images: everything, the part
        still needing a human, and (when enabled) the confusable subset.

        The collector auto-annotates with its models and tags what it draws
        UNVERIFIED, so after a long run "what did I collect" and "what do I still
        have to look at" are different, and usually very different, questions. One
        button could only answer the first.
        """
        if not self._task_tag_uuid:
            return []

        # image_tag_uuids is what actually finds every collected frame: the tag
        # is written to the image record, so it matches whether or not the frame
        # ended up with bboxes. tag_uuids (a bbox-tag filter) is kept for
        # annotated runs, where narrowing to the tagged bboxes is more useful.
        scope = {
            "tag_uuids": [] if self._spec.collect_all else [self._task_tag_uuid],
            "image_tag_uuids": [self._task_tag_uuid],
        }
        actions = [
            {
                "type": "open_review",
                "label": "Browse all results",
                # Sent explicitly rather than omitted so this also *clears* an
                # 'unverified' filter left over from the button next to it.
                "payload": {**scope, "bbox_verification": "any"},
            },
            {
                "type": "open_review",
                "label": "Browse unverified",
                "payload": {**scope, "bbox_verification": "unverified"},
            },
        ]
        if self._spec.enable_ambiguity:
            # The confusable set is usually a small fraction of a run and is the
            # part with the highest review value per box, so it gets its own way
            # in rather than being something to reconstruct by hand in the filter
            # bar. Narrowed by bbox tag, which is what the tag is for.
            actions.append(
                {
                    "type": "open_review",
                    "label": "Browse confusable",
                    "payload": {
                        **scope,
                        "tag_uuids": [*(scope["tag_uuids"] or []), CONFUSABLE_TAG_UUID],
                        "bbox_verification": "any",
                    },
                }
            )
        if self._spec.enable_detector_miss:
            # Always unverified — a detector-miss box is a fresh seed, never a
            # confirmed one (plan §5: "unverified": narrower than confusable's
            # "any", since a verified detector-miss box has already had its
            # one human look).
            actions.append(
                {
                    "type": "open_review",
                    "label": "Browse detector misses",
                    "payload": {
                        **scope,
                        "tag_uuids": [*(scope["tag_uuids"] or []), DETECTOR_MISS_TAG_UUID],
                        "bbox_verification": "unverified",
                    },
                }
            )
        return actions

    def resume_fragment(self) -> dict[str, Any]:
        """The collector's half of a checkpoint."""
        return {
            "images_saved": self._images_saved,
            "dup_frames_discarded": self._dup_frames_discarded,
            "dedup_frames_checked": self._dedup_frames_checked,
            "dedup_image_hits": self._dedup_image_hits,
            "dedup_bbox_hits": self._dedup_bbox_hits,
            "filtered_discarded": self._filtered_discarded,
            "filtered_relabeled": self._filtered_relabeled,
            "filtered_tagged": self._filtered_tagged,
            "filtered_stripped": self._filtered_stripped,
            "ambiguity_flagged": self._ambiguity_flagged,
            "ambiguity_evaluated": self._ambiguity_evaluated,
        }

    def result_fragment(self) -> dict[str, Any]:
        """The collector's contribution to the driver's completion result."""
        result: dict[str, Any] = {"images_saved": self._images_saved}
        if self._spec.enable_ambiguity:
            result["ambiguity_flagged"] = self._ambiguity_flagged
            result["ambiguity_evaluated"] = self._ambiguity_evaluated
            result["ambiguity_pairs"] = [
                {"model": m, "label_uuid": w, "rival_label_uuid": r, "count": c}
                for (m, w, r), c in sorted(self._ambiguity_pairs.items(), key=lambda kv: kv[1], reverse=True)
            ]
        if self._spec.enable_dedup:
            result["dedup_frames_checked"] = self._dedup_frames_checked
            result["dedup_image_hits"] = self._dedup_image_hits
            result["dedup_bbox_hits"] = self._dedup_bbox_hits
            result["dup_frames_discarded"] = self._dup_frames_discarded
        if self._spec.filter_chain_uuid:
            result["filtered_discarded"] = self._filtered_discarded
            result["filtered_relabeled"] = self._filtered_relabeled
            result["filtered_tagged"] = self._filtered_tagged
            result["filtered_stripped"] = self._filtered_stripped
        return result

    def summary_lines(self) -> list[str]:
        """Human-readable end-of-run summary fragments for the driver's log."""
        lines: list[str] = []
        if self._spec.enable_dedup:
            lines.append(
                f"dedup: {self._dedup_frames_checked} checked"
                f", {self._dedup_image_hits} img hits"
                f", {self._dedup_bbox_hits} bbox hits"
                f", {self._dup_frames_discarded} discarded"
            )
        if self._filter_chain_runner is not None:
            lines.append(
                f"filter chain: {self._filtered_discarded} discarded"
                f", {self._filtered_relabeled} relabeled"
                f", {self._filtered_tagged} tagged"
                f", {self._filtered_stripped} stripped"
            )
        return lines

    def ambiguity_summary(self) -> Optional[str]:
        """The per-model confusion pairs — the actual training feedback.

        A flagged-frame count says work is waiting; this says what the model is bad
        at. Ordered by how much review time each pair will cost.
        """
        if not self._spec.enable_ambiguity:
            return None
        top_pairs = sorted(self._ambiguity_pairs.items(), key=lambda kv: kv[1], reverse=True)[:10]
        return f"Confusable: {self._ambiguity_flagged} flagged of {self._ambiguity_evaluated} measured" + (
            " — " + ", ".join(f"{m}: {self._label_name(w)}↔{self._label_name(r)} ×{c}" for (m, w, r), c in top_pairs)
            if top_pairs
            else ""
        )

    def dedup_queue_depth(self) -> int:
        return self._dedup_queue.qsize() if self._dedup_queue is not None else 0


async def _read_int_setting(settings_api: SettingsApi, key: str, default: int) -> int:
    """Read an integer setting from the DB settings API; return default on any failure.

    Same contract as `Task._read_int_setting` — deliberately a copy rather than a
    call into the task base class, which the collector must not depend on.
    """
    try:
        resp = await asyncio.to_thread(settings_api.get_setting, key)
        return int(resp.value)
    except Exception:
        return default
