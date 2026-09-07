import asyncio
import copy
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import db_client
import inference_client

from apps.helpers.consts import TaskAccent, TaskStatus
from apps.helpers.imageProviders.imageProvider import ImageProvider, ImageProviderType
from apps.helpers.imageUtils import base64_encode_jpg
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.training.collector import CollectorHost, FrameOffer, TrainingCollector
from apps.helpers.training.collectorSpec import (
    AMBIGUITY_KIND_CROSS_MODEL,
    AMBIGUITY_KIND_SAME_MODEL,
    AMBIGUITY_METRIC_MARGIN,
    AMBIGUITY_METRIC_RATIO,
    SAME_BBOX_IOU_THRESHOLD,
    AmbiguityInfo,
    CollectorSpec,
    ConfidenceBand,
    TargetLabel,
    ambiguity_fires,
    ambiguity_value,
    collector_params_schema,
    collector_status_schema,
    parse_confidence_band,
)

from ._filter_chain_runner import FilterChainRunner
from .ApplyFilterChainTask import (
    _CLASSIFIER_KINDS,
    _HttpInferenceBackend,
    _read_model_kind,
)
from .Registry import register_task
from .Task import Task
from .utils.sourceHelper import SourceHelper

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

# The configuration half of collection — confidence bands, ambiguity metrics, the
# same-bbox IoU threshold — now lives in apps.helpers.training.collectorSpec, so a
# watcher-driven collector and this task are configured by one code path. The names
# are re-exported above (and via __all__ below) because they were public here first.
_parse_confidence_band = parse_confidence_band
_ambiguity_value = ambiguity_value
_ambiguity_fires = ambiguity_fires

__all__ = [
    "AMBIGUITY_KIND_CROSS_MODEL",
    "AMBIGUITY_KIND_SAME_MODEL",
    "AMBIGUITY_METRIC_MARGIN",
    "AMBIGUITY_METRIC_RATIO",
    "SAME_BBOX_IOU_THRESHOLD",
    "AmbiguityInfo",
    "CollectTrainingImagesTask",
    "ConfidenceBand",
    "TargetLabel",
]


@register_task()
class CollectTrainingImagesTask(Task):
    # Gathering data. Kept apart from training (violet) on purpose:
    # telling a collection from the training it feeds is the whole reason the
    # Jobs pane needed colour.
    CARD_ACCENT = TaskAccent.GREEN

    # Per-call retry limits and timeouts for I/O operations that can hang indefinitely.
    _MAX_RETRIES: int = 3
    _FETCH_TIMEOUT_S: float = 30.0  # seconds to wait for a single getNextImage() call
    _INFERENCE_TIMEOUT_S: float = 60.0  # seconds per inference call (allows for model load)
    # Dedup pipeline buffer: allows this many matched frames to queue for dedup before
    # inference blocks.  A small buffer absorbs per-frame jitter without wasting memory
    # (each slot holds a numpy array reference, not a copy).
    _DEDUP_QUEUE_SIZE: int = 4
    # Live-collection tasks can run for days; checkpoint far more often than the base
    # Task default so an unexpected restart loses at most a few minutes of progress.
    CHECKPOINT_INTERVAL: int = 60
    # Seconds a model pin is requested for. Renewed on a timer at roughly half this
    # duration (_PIN_RENEWAL_INTERVAL_S) so a run longer than this never lets a pin
    # lapse silently.
    _PIN_DURATION_S: int = 3600
    _PIN_RENEWAL_INTERVAL_S: float = _PIN_DURATION_S / 2
    # Stale-source notice escalation thresholds, as a multiple of the source's own
    # configured fetch interval elapsed without a successful frame.
    _STALE_NOTICE_INFO_MULTIPLE: float = 3.0
    _STALE_NOTICE_WARNING_MULTIPLE: float = 10.0
    _STALE_NOTICE_ERROR_MULTIPLE: float = 30.0
    # A source that fails to yield a frame is retried on a doubling backoff rather
    # than dropped: a camera that drops its TLS handshake for a minute is a normal
    # event on a long collection run, and losing that source silently for the rest
    # of the run (or, worse, killing the whole task) throws away everything the
    # other sources would still have produced.
    _SOURCE_RETRY_BACKOFF_S: float = 5.0
    _SOURCE_RETRY_BACKOFF_MAX_S: float = 300.0
    # The task only gives up when EVERY source has been failing continuously for
    # this long — with nothing to capture from, there is no work left to do. Kept
    # generous: the point is to outlast router reboots and camera power-cycles.
    _ALL_SOURCES_FAILING_GRACE_S: float = 900.0
    # Rolling window of per-frame nearest-neighbour distances used to build the
    # threshold-calibration widgets. Deliberately in-memory only (not checkpointed):
    # it's a "what is the stream doing right now" sample, and replaying distances
    # measured before a long pause would describe a scene that no longer exists.
    _DEDUP_SAMPLE_SIZE: int = 200
    # Candidate thresholds shown in the sweep tables. Image phashes are hash_size²
    # bits (64 at the default 8) and bbox phashes 256 at the default 16, so the two
    # live on different scales and get their own ranges rather than a shared one.
    # Both ladders run past any threshold anyone would plausibly configure and cover
    # the same span the histogram does: a sweep that stops at the current setting
    # can't answer "what would raising it do?", which is the whole point of the table.
    # Steps widen with distance because that's where the curve flattens out.
    _IMAGE_SWEEP_THRESHOLDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 64)
    _BBOX_SWEEP_THRESHOLDS: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 90, 100)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # get_status_data()/get_notices() can be polled in the narrow PENDING window
        # before _init() has run (task.start() constructs the instance; _execute()
        # awaits _init() asynchronously right after) — set safe defaults here so
        # that window never raises, rather than only guarding inside _init().
        #
        # The collector cannot be one of them: building it needs the parsed spec,
        # which needs params. Everything that reads it therefore goes through
        # `getattr(self, "_collector", None)` and reports nothing when it is absent.
        self._collector: Optional[TrainingCollector] = None
        self._elapsed_duration_at_start: float = 0.0
        self._run_started_at: float = time.perf_counter()
        self._paused_duration_s: float = 0.0
        self._images_processed: int = 0
        self._fixed_frames_processed: int = 0
        self._processed_ids: set[str] = set()
        self._source_names: list[str] = []
        self._source_types: dict[str, str] = {}
        self._source_intervals: dict[str, float] = {}
        self._last_fetch_by_source: dict[str, float] = {}
        # source name → {"consecutive": int, "error": str, "since": float, "retry_at": float}.
        # Present only while a source is failing; cleared outright on its next good frame.
        self._source_failures: dict[str, dict[str, Any]] = {}
        self._last_any_success_at: float = time.perf_counter()

    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        self._status_msg: str = "Initializing"
        self.log("Initializing task parameters")
        self._source_uuids: list[str] = params["source_uuids"]
        # Everything about *what to keep and how to annotate it* lives in a
        # `training_collectors` row, so a watcher-driven collector and this task
        # configure one entity. This loop owns only its sources and pacing.
        # from_params raises for a traversing output_dir and degrades everything
        # else to a documented default; see its docstring for why the asymmetry.
        collector_params = await self._fetch_collector_params(params)
        self._spec: CollectorSpec = CollectorSpec.from_params(collector_params)
        # _config_name is injected by the manager (reserved key, not user-facing).
        self._config_name: str = params.get("_config_name", "")
        # The collector owns everything downstream of (frame, detections): bands,
        # ambiguity, the filter chain, dedup, saving, the model pins and its own
        # counters. This task is now the provider rotation, the pacing and the
        # resume bookkeeping that feed it.
        self._collector = TrainingCollector(
            self._spec,
            self._build_collector_host(),
            files_root=Path(os.environ.get("FILES_ROOT", "/app/image_datasets")),
            tag_scope_uuid=self._config_uuid,
            tag_scope_name=self._config_name,
            # The collector IS this loop, so a full dedup queue should slow the
            # fetch rather than drop a frame nothing else will ever see.
            blocking_offer=True,
        )
        await self._collector.prepare(resume_data)

        # Intervals for rate-limited provider types
        self._snapshot_interval: float = float(params.get("snapshot_interval", 15))
        self._stream_interval: float = float(params.get("stream_interval", 1.0))

        # Total elapsed run time in seconds, accumulated across restarts (persisted in
        # resume_data) plus whatever this process instance has been running — so
        # max_duration and duration-based progress are correct even after a checkpoint
        # restart, not just within a single process lifetime.
        self._elapsed_duration_at_start: float = resume_data.get("elapsed_duration_s", 0.0) if resume_data else 0.0
        # Set for real at the top of _run(); placeholder here so _elapsed_duration_s()
        # is safe to call before _run() starts (e.g. from tests exercising _init alone).
        self._run_started_at: float = time.perf_counter()
        # Wall time this run has spent paused in place, subtracted from elapsed
        # duration so a pause doesn't burn down the max_duration budget.
        self._paused_duration_s: float = 0.0

        self._images_processed: int = resume_data.get("images_processed", 0) if resume_data else 0
        self._fixed_frames_processed: int = resume_data.get("fixed_frames_processed", 0) if resume_data else 0
        # Set of "source:frame_idx" strings for images already processed this run or
        # in a previous run that was paused.  Lets us skip re-processing on resume.
        self._processed_ids: set[str] = set(resume_data.get("processed_ids", [])) if resume_data else set()
        if self._processed_ids:
            self.log(f"Resuming: {len(self._processed_ids)} images already processed")

        # Status-panel bookkeeping (Phase 4 — live collection panel):
        # _source_names/_types/_intervals and _last_fetch_by_source already have
        # safe pre-run defaults from __init__; populated for real once _run()
        # starts and providers are known.

    async def _fetch_collector_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Resolve `collector_uuid` to the stored collector params.

        Raises rather than falling back to the raw params: a missing collector
        means the run would silently collect with defaults into whatever
        `output_dir` defaults to, which is the failure mode worth being loud about.
        """
        collector_uuid = params.get("collector_uuid")
        if not collector_uuid:
            raise ValueError("No collector selected — pick one in Configuration → Collectors")
        api = db_client.TrainingCollectorsApi(self._manager.get_db_api_client())
        response = await asyncio.to_thread(api.get_training_collector, collector_uuid)
        if response.collector is None:
            raise ValueError(f"Collector {collector_uuid} not found: {response.message or 'unknown error'}")
        if not response.collector.enabled:
            raise ValueError(f'Collector "{response.collector.name}" is disabled — enable it to run this task')
        return dict(response.collector.params or {})

    def _build_collector_host(self) -> CollectorHost:
        """The narrow seam the collector reaches the outside world through.

        The API clients are passed as factories, not instances: a live collect can
        outlive a Settings → Servers change, and a cached client would keep talking
        to the host the user navigated away from. The filter-chain pieces are
        factories for a different reason — they live in this package, and having
        `helpers/` import them would invert the layering the split exists to fix.
        """
        return CollectorHost(
            log=self.log,
            db_api_client=self._manager.get_db_api_client,
            inference_api_client=self._manager.get_inference_api_client,
            stop_event=self._cancel_flag,
            # Lambdas, not the functions themselves: both are patched by name on
            # this module's imports in tests, and a lambda resolves at call time.
            model_kind_reader=lambda name: _read_model_kind(name),
            classifier_kinds=frozenset(_CLASSIFIER_KINDS),
            build_inference_backend=lambda provider, pin_ids: _HttpInferenceBackend(provider, pin_ids),
            build_filter_chain_runner=lambda steps, backend: FilterChainRunner(steps, backend),
        )

    async def _run(self) -> dict[str, Any]:
        # Real start of this process's run — _elapsed_duration_s() adds this delta to
        # whatever was already accumulated in resume_data before this restart.
        self._run_started_at = time.perf_counter()
        collector = self._collector
        assert collector is not None  # _init always builds one

        db_api_client = self._manager.get_db_api_client()

        # Tags, model pins, the filter chain, the dedup worker and the pin-renewal
        # loop all belong to the collector now; this is the one call that brings
        # them up.
        self._status_msg = "Starting collector"
        await collector.start()

        self._status_msg = "Getting image providers"
        source_helper = SourceHelper(db_api_client)
        providers: list[ImageProvider] = []
        for source_uuid in self._source_uuids:
            try:
                provider = await asyncio.to_thread(source_helper.get_image_provider, source_uuid)
            except ValueError:
                self.log(f"Failed to get provider for {source_uuid}")
                continue
            providers.append(provider)

        # Separate out FIXED providers so we can compute real progress from their frame counts.
        # For LIVE_SNAPSHOT and LIVE_STREAM providers we track last-fetch time for rate limiting.
        fixed_providers: list[ImageProvider] = [p for p in providers if p.provider_type() == ImageProviderType.FIXED]
        last_fetch: dict[int, float] = {}  # id(provider) → perf_counter time of last successful fetch
        _rate_limits: dict[ImageProviderType, float] = {
            ImageProviderType.LIVE_SNAPSHOT: self._snapshot_interval,
            ImageProviderType.LIVE_STREAM: self._stream_interval,
        }

        # Stale-source tracking (live providers only — FIXED providers complete and
        # drop out of `active` on their own via the existing timeout/exhaustion path).
        # Keyed by provider.source_name (Phase 1's stable, user-facing identity) rather
        # than id(provider): id() is not stable across GC and isn't meaningful in the UI.
        self._last_fetch_by_source: dict[str, float] = {}
        self._source_intervals: dict[str, float] = {}
        # Every source, in configured order, for the status-panel sources table —
        # unlike _source_intervals (live sources only, used for staleness), this
        # covers FIXED sources too so the table lists everything the task reads from.
        self._source_names = []
        self._source_types = {}
        # Kept so staleness can ask a provider what it currently expects, rather than
        # reading the one number captured here at setup. A source whose cadence changes
        # at runtime (a Ring camera is ~30s idle and ~20fps live) cannot be described by
        # any single value fixed now — see `ImageProvider.next_frame_expected_by`.
        self._providers_by_source: dict[str, ImageProvider] = {}
        for provider in providers:
            name = provider.source_name or "unknown"
            self._source_names.append(name)
            self._source_types[name] = provider.provider_type().value
            self._providers_by_source[name] = provider
            if provider.provider_type() == ImageProviderType.FIXED:
                continue
            interval = provider.min_capture_interval or _rate_limits.get(provider.provider_type(), 0.0)
            if provider.source_name and interval > 0:
                self._source_intervals[provider.source_name] = interval

        def _status_msg_str() -> str:
            # Switch to generic message once all fixed providers are exhausted.
            if fixed_providers and any(p in active for p in fixed_providers):
                total = sum(p.get_total_frames() or 0 for p in fixed_providers)
                return f"Processed {self._fixed_frames_processed}/{total} frames, Saved {collector.images_saved} images"
            return f"Processed {self._images_processed} frames, Saved {collector.images_saved} images"

        def _dup_stats_str() -> str:
            if not self._spec.enable_dedup:
                return ""
            stats, _ = collector.status_stats()
            parts = [
                f"{stats.get('dedup_checked', 0)} checked",
                f"{stats.get('dedup_image_hits', 0)} img hits",
                f"{stats.get('dedup_bbox_hits', 0)} bbox hits",
            ]
            if stats.get("dedup_discarded"):
                parts.append(f"{stats['dedup_discarded']} discarded")
            queue_depth = collector.dedup_queue_depth()
            if queue_depth > 0:
                parts.append(f"{queue_depth} queued")
            return ", dedup: " + ", ".join(parts)

        def _full_status() -> str:
            return _status_msg_str() + _dup_stats_str()

        def _collect_saves() -> None:
            """Harvest finished saves and fold the results into the task's own bookkeeping."""
            for uid in collector.collect_saves():
                self._processed_ids.add(uid)
            # Update progress for FIXED providers based on real frame counts.
            if fixed_providers:
                total = sum(p.get_total_frames() or 0 for p in fixed_providers)
                if total > 0:
                    self._update_progress(min(100.0, self._fixed_frames_processed / total * 100.0))

        async def _drain_saves() -> None:
            for uid in await collector.drain_saves():
                self._processed_ids.add(uid)

        self._status_msg = "Running"
        self.log(f"Starting collection (resuming from {collector.images_saved} images)")
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

        for provider in providers:
            # Enable pre-encoding in the prefetch worker so encode cost is pipelined
            # with the previous frame's inference rather than sitting on the critical path.
            if hasattr(provider, "pre_encode"):
                provider.pre_encode = True
            await provider.start()

        async def _service_checkpoint_request() -> None:
            """Snapshot resume data now if the manager has requested a checkpoint.

            Checked at the top of the OUTER while loop (once per pass over the active
            provider set) rather than only after a provider yields a frame. Sources are
            paced independently (min_capture_interval / task-level rate limits) and can be
            far slower than the checkpoint interval — e.g. a 15-minute snapshot source.
            Checking only after a successful fetch would leave a checkpoint request
            unanswered for the whole fetch interval. Checking here means it's serviced on
            (at most) the next outer-loop tick — a busy-spin every ~50ms when no provider
            has fresh data (see `if not any_fetched: sleep(0.05)` below) — regardless of
            how slowly any individual source is paced.
            """
            if self._data_req_flag.is_set():
                # flush() drains the dedup queue as well as the in-flight saves, so
                # every matched frame is accounted for in the snapshot.
                for uid in await collector.flush():
                    self._processed_ids.add(uid)
                self._data_req_flag.clear()
                self._resume_data = self._build_resume_data()
                self._data_ready_flag.set()

        try:
            active = set(providers)
            while active and not self._cancel_flag.is_set():
                # _check_pause() returns immediately unless the task is paused in place,
                # so this is a no-op delta on the hot path; when it does block, the time
                # is banked as paused and excluded from the max_duration budget.
                _t_pause = time.perf_counter()
                await self._check_pause()
                self._paused_duration_s += time.perf_counter() - _t_pause
                await _service_checkpoint_request()
                _collect_saves()
                self._status_msg = _full_status()

                # Live tasks (no FIXED provider) otherwise report 0% forever. Measure
                # against whichever stop condition is configured; indeterminate (None,
                # which renders the frontend's striped bar) when neither is set. FIXED
                # providers' real frame-count progress (in _collect_saves) is untouched.
                if not fixed_providers:
                    self._update_progress(collector.stop_condition_progress(self._elapsed_duration_s()))

                if collector.stop_condition_reached(self._elapsed_duration_s()):
                    self.log("Stop condition reached (max_images/max_duration) — stopping")
                    break

                any_fetched = False

                # Pick who is due this pass (cheap and sequential), then fetch them all
                # at once. Fetching in a plain loop silently turns each source's timeout
                # into a budget for *everyone*: one quiet source — a Ring camera idling
                # on ~30s cloud snapshots — stalls the whole rotation, and its peers go
                # stale behind it while nothing is actually failing, which reads as
                # several broken cameras rather than one slow one.
                #
                # Only the *fetch* is concurrent. Everything downstream stays sequential
                # and in the original source order, because dedup, inference and the DB
                # writes all share state that is not safe to interleave.
                due: list[tuple[Any, str]] = []
                for provider in list(active):
                    if self._cancel_flag.is_set():
                        break

                    # Rate-limit LIVE providers: skip until enough time has elapsed.
                    # Providers with their own min_capture_interval pace themselves inside
                    # getNextImage() (source-level interval overrides the task-level fallback) —
                    # skip the task-level check for them so the two intervals don't compound.
                    ptype = provider.provider_type()
                    if provider.min_capture_interval <= 0 and (interval := _rate_limits.get(ptype)) is not None:
                        if time.perf_counter() - last_fetch.get(id(provider), 0) < interval:
                            continue

                    # A failing source waits out its backoff instead of being retried
                    # every pass (and instead of being dropped for good).
                    source_key = self._source_key(provider)
                    if self._source_in_backoff(source_key, time.perf_counter()):
                        continue

                    due.append((provider, source_key))

                # gather rather than a TaskGroup: a provider's own failure is already
                # captured and returned by _fetch_with_retries, so the only thing that
                # can escape here is cancellation, which must propagate.
                fetch_results = await asyncio.gather(*(self._fetch_with_retries(p, k) for p, k in due)) if due else []

                for (provider, source_key), (provided, fetch_error, t0, t_fetch) in zip(due, fetch_results):
                    # Re-checked per result, not just before the gather: a long pass can
                    # span a cancel, and the frames already in hand must not be processed
                    # after the user asked to stop.
                    if self._cancel_flag.is_set():
                        break

                    if fetch_error is not None:
                        # Kept in `active` — it gets another go once its backoff expires.
                        consecutive = self._record_source_failure(source_key, fetch_error, t_fetch)
                        if consecutive == 1 or consecutive % 10 == 0:
                            # Log the first failure and then every tenth, so a source that
                            # is down for hours doesn't drown the log it shares with the
                            # sources that are still working.
                            self.log(f"Source '{source_key}' failing ({consecutive}×): {fetch_error}")
                        continue

                    if provided is None:
                        # Not a failure: the provider ran out of frames (FIXED sources
                        # finish this way), so it leaves the rotation for good.
                        active.discard(provider)
                        self._source_failures.pop(source_key, None)
                        continue

                    if self._source_failures.pop(source_key, None) is not None:
                        self.log(f"Source '{source_key}' recovered")
                    self._record_source_success(source_key, t_fetch)
                    any_fetched = True
                    last_fetch[id(provider)] = t_fetch
                    if provider.source_name:
                        self._last_fetch_by_source[provider.source_name] = t_fetch

                    if provided.image is None:
                        # The provider got a response but couldn't decode it into an
                        # image (e.g. a camera returning a transient non-image reply) —
                        # distinct from `provided is None`, which means the source
                        # itself is exhausted/dead. Skip just this frame; don't drop
                        # the source from `active` over one bad frame. The rate-limit
                        # clock must still be updated above, or a source that keeps
                        # returning undecodable frames gets retried on every loop
                        # iteration instead of respecting its configured interval.
                        logger.debug("Skipping frame with no decodable image (source=%s)", provider.source_name)
                        continue

                    # Build a stable unique ID when the provider gives us source + frame_idx.
                    # Used to skip already-processed frames on resume — FIXED providers only:
                    # live providers now also populate source/frame_idx (Phase 1), but their
                    # frame_idx resets on every task restart and runs unbounded, so tracking
                    # them would grow processed_ids/resume-data JSON without bound.
                    unique_id: Optional[str] = None
                    if (
                        ptype == ImageProviderType.FIXED
                        and provided.metadata.source is not None
                        and provided.metadata.frame_idx is not None
                    ):
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

                    _saved_before = collector.images_saved
                    _pending_before = len(collector.pending_saves)
                    await collector.offer(
                        FrameOffer(
                            provided=provided,
                            base64=base64_img,
                            provider_type=ptype,
                            unique_id=unique_id,
                        )
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
                    _dispatched = (
                        collector.images_saved > _saved_before or len(collector.pending_saves) > _pending_before
                    )
                    logger.debug(
                        "Frame timing: fetch=%.1fms encode=%.1fms(%s) infer=%.1fms%s",
                        _fetch_ms,
                        _encode_ms,
                        _enc_label,
                        _infer_ms,
                        " (save dispatched)" if _dispatched else "",
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

                    # Yield to the event loop between frames so HTTP handlers get cycles
                    # even when providers deliver images as fast as they can be processed.
                    await asyncio.sleep(0)

                # If all remaining active providers were rate-limited this iteration,
                # yield briefly to avoid busy-spinning.
                if not any_fetched:
                    await asyncio.sleep(0.05)

                # Give up only when there is nothing left that could produce a frame:
                # every remaining source failing, and none of them having succeeded for
                # the whole grace window. One working source is enough to keep going —
                # the failing ones stay in the rotation, warned about via get_notices().
                _now = time.perf_counter()
                if active and all(self._source_key(p) in self._source_failures for p in active):
                    down_for = _now - self._last_any_success_at
                    if down_for >= self._ALL_SOURCES_FAILING_GRACE_S:
                        detail = "; ".join(
                            f"{name}: {entry['error']}" for name, entry in sorted(self._source_failures.items())
                        )
                        raise RuntimeError(
                            f"All {len(active)} capture source(s) have been failing for {down_for:.0f}s — {detail}"
                        )

                # Heartbeat: log current status every 30 s so the task doesn't look stuck
                # when providers deliver frames but none match the detection criteria.
                _now = time.perf_counter()
                if _now - _last_heartbeat >= 30.0:
                    _last_heartbeat = _now
                    self.log(_full_status())

        finally:
            # Stops the pin-renewal loop, drains dedup, waits out the in-flight saves
            # and releases the pins — in that order, for the reasons its docstring
            # gives. The uids it returns still have to reach _processed_ids, or a
            # resume re-collects the last frames of the run.
            await _drain_saves()
            await collector.shutdown()
            await _drain_saves()
            for provider in providers:
                await provider.stop()

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
        summary = collector.summary_lines()
        self.log(
            f"Complete: {self._images_processed} seen, {collector.images_saved} saved"
            + ("".join(f", {line}" for line in summary))
        )
        ambiguity_summary = collector.ambiguity_summary()
        if ambiguity_summary:
            self.log(ambiguity_summary)

        result: dict[str, Any] = collector.result_fragment()
        if collector.task_tag_uuid:
            result["actions"] = self._review_actions()
        return result

    def _elapsed_duration_s(self) -> float:
        """Total ACTIVE run time in seconds, accumulated across restarts.

        Time spent paused in place is excluded: counting it would mean a task paused
        overnight and resumed in the morning instantly satisfies its max_duration and
        stops collecting, which is both surprising and silently destructive. Time
        between separate runs (suspend to checkpoint, server restart) is likewise not
        counted, since _run_started_at resets on each run while
        _elapsed_duration_at_start carries only the active time already banked.
        """
        return self._elapsed_duration_at_start + (time.perf_counter() - self._run_started_at) - self._paused_duration_s

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Human-readable elapsed duration for the stat-grid 'Uptime' cell."""
        total = max(0, int(seconds))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    @classmethod
    def status_schema(cls) -> list[dict[str, Any]]:
        """`last_capture` and `sources` are this task's own — a watcher has
        neither a single frame nor a fixed source list of its own to show.
        Everything about *collection itself* comes from `collector_status_schema()`
        (collectorSpec.py), shared with the per-collector diagnostics a
        watcher-attached collector renders — this task only splices its own
        loop-owned fields (`processed`, `captures_per_min`, `uptime`) into that
        shared schema's `stats` widget, exactly the way `params_schema()` used
        to splice loop params around `collector_params_schema()`.
        """
        schema = copy.deepcopy(collector_status_schema())
        stats_widget = next(w for w in schema if w["id"] == "stats")
        stats_widget["fields"] = {
            "saved": "Saved",
            "processed": "Processed",
            "captures_per_min": "Captures/min",
            "uptime": "Uptime",
            **stats_widget["fields"],
        }
        return [
            {
                "id": "last_capture",
                "type": "annotated_image",
                "source": "status",
                "label": "Last capture",
                "height": "16em",
            },
            *schema,
            {
                "id": "sources",
                "type": "table",
                "source": "status",
                "label": "Sources",
                "columns": [
                    {"key": "name", "label": "Source"},
                    {"key": "type", "label": "Type"},
                    {"key": "interval", "label": "Interval"},
                    {"key": "last_frame", "label": "Last frame"},
                    {"key": "health", "label": "Health"},
                    {"key": "last_error", "label": "Last error"},
                ],
            },
        ]

    def get_status_data(self) -> dict[str, Any]:
        """Live values for the Phase 4 live-collection status panel.

        Reads only already-in-memory counters/attributes — no I/O, no awaits, no
        DB/filesystem access — per the base class contract (this is called
        synchronously from a poll handler while a task card is expanded).

        `getattr` for the collector, not attribute access: this is reachable in the
        PENDING window before _init() has built one, and a plain read would
        AttributeError far from the change.
        """
        elapsed = self._elapsed_duration_s()
        collector = getattr(self, "_collector", None)

        # Everything the collector counts (saved, dedup, confusable, filter chain)
        # comes back in one call; the task contributes only what it owns — frames
        # seen, uptime, and the sources table.
        stats: dict[str, Any]
        calibration: dict[str, Any]
        if collector is not None:
            stats, calibration = collector.status_stats()
        else:
            stats, calibration = {"saved": 0}, {}
        saved = stats.get("saved", 0)
        stats["processed"] = self._images_processed
        stats["captures_per_min"] = round(saved / (elapsed / 60.0), 1) if elapsed > 0 else 0
        stats["uptime"] = self._format_duration(elapsed)

        now = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for name in self._source_names:
            interval = self._source_intervals.get(name)
            last_fetch = self._last_fetch_by_source.get(name)
            last_frame = f"{now - last_fetch:.0f}s ago" if last_fetch is not None else "never"
            if interval:
                health, _elapsed = self._source_staleness(name, interval)
                interval_str = f"{interval:.0f}s"
            else:
                # FIXED sources (and any live source with no configured interval) have
                # no staleness baseline — mirror get_notices(), which never tracks them.
                health = "n/a"
                interval_str = "—"
            # A source we know is failing reports that outright. Staleness is inferred
            # from silence; this is the actual reason, so it wins over "stale".
            failure = self._source_failures.get(name)
            if failure is not None:
                health = "failing"
                last_error = f"{failure['error']} ({failure['consecutive']}× over {now - failure['since']:.0f}s)"
            else:
                last_error = "—"
            rows.append(
                {
                    "name": name,
                    "type": self._source_types.get(name, ""),
                    "interval": interval_str,
                    "last_frame": last_frame,
                    "health": health,
                    "last_error": last_error,
                }
            )

        return {
            "last_capture": collector.last_capture if collector is not None else None,
            "stats": stats,
            "sources": rows,
            **calibration,
        }

    def _build_resume_data(self) -> dict[str, Any]:
        """The task's own bookkeeping plus the collector's counters.

        One flat dict, unchanged in shape: a checkpoint written before the split
        must still resume a task after it, and `apply_resume` reads the same keys
        back out of exactly this payload.
        """
        collector = getattr(self, "_collector", None)
        return {
            **(collector.resume_fragment() if collector is not None else {}),
            "images_processed": self._images_processed,
            "fixed_frames_processed": self._fixed_frames_processed,
            "processed_ids": list(self._processed_ids),
            "elapsed_duration_s": self._elapsed_duration_s(),
        }

    async def _deinit(self) -> None:
        pass

    async def _fetch_with_retries(self, provider, source_key: str) -> tuple[Any, Optional[str], float, float]:
        """One source's fetch, retries included, as `(provided, error, t0, t_fetch)`.

        Never raises for a provider's own failure — the error is *returned*. These run
        under `asyncio.gather`, and a raise would abandon every sibling fetch in the
        same pass over one bad camera. `asyncio.CancelledError` is deliberately not
        caught (it is a BaseException), so cancellation still propagates.
        """
        # A self-pacing provider sleeps for up to min_capture_interval *inside*
        # getNextImage() before it even starts fetching. That scheduled wait is not a
        # stall, so the fetch budget has to sit on top of it — otherwise any source
        # configured slower than _FETCH_TIMEOUT_S (e.g. a camera polled once a minute)
        # times out on every call, burns all _MAX_RETRIES attempts and is dropped from
        # `active` on its very first fetch despite behaving correctly.
        #
        # A provider that knows its own cadence extends the budget further: an idle
        # Ring camera has nothing to say for a whole snapshot interval (90s out of the
        # ring-mqtt box) and carries no min_capture_interval, so a flat 30s budget times
        # out on every attempt, burns all _MAX_RETRIES, records the source as failing and
        # walks it up the exponential backoff — a healthy camera reported as broken. See
        # `ImageProvider.next_frame_expected_by`, whose first documented use this is.
        fetch_timeout = self._extended_wait(provider, self._FETCH_TIMEOUT_S + max(0.0, provider.min_capture_interval))
        t0 = time.perf_counter()
        provided = None
        fetch_error: Optional[str] = None
        for _attempt in range(self._MAX_RETRIES):
            try:
                provided = await asyncio.wait_for(provider.getNextImage(), timeout=fetch_timeout)
                fetch_error = None
                break
            except asyncio.TimeoutError:
                fetch_error = f"timed out after {fetch_timeout:.0f}s"
            except Exception as e:
                fetch_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    "Fetch from '%s' failed on attempt %d/%d: %s",
                    source_key,
                    _attempt + 1,
                    self._MAX_RETRIES,
                    fetch_error,
                )
            if _attempt < self._MAX_RETRIES - 1:
                logger.warning("Retrying '%s' (attempt %d/%d)", source_key, _attempt + 2, self._MAX_RETRIES)
        # Stamp the frame with the source row it came from. Done here rather than
        # inside each provider because providers fill `metadata.source` with
        # whatever they find natural (a name, a file path, nothing at all) —
        # this is the one place that has both the provider and its frame, and it
        # feeds every _save_image call site, including the dedup worker's.
        if provided is not None and getattr(provided, "metadata", None) is not None:
            provided.metadata.source_uuid = provider.source_uuid
        return provided, fetch_error, t0, time.perf_counter()

    @staticmethod
    def _extended_wait(provider, configured: float) -> float:
        """`configured`, stretched to whatever this provider says it needs.

        Never *shortens* it: a provider is trusted to ask for more patience, never
        less. Shared by the fetch budget and the staleness threshold so the two can
        never disagree about how long silence from a source is normal.
        """
        if provider is None:
            return configured
        try:
            deadline = provider.next_frame_expected_by()
        except Exception:
            # A provider misbehaving here must not take out the status panel, which is
            # often the only way to see that it is misbehaving at all.
            logger.debug("next_frame_expected_by() failed for %r", provider, exc_info=True)
            return configured
        if deadline is None:
            return configured
        # time.monotonic() specifically — the deadline is on that clock, and
        # perf_counter() (used elsewhere in this class) has a different epoch.
        return max(configured, deadline - time.monotonic())

    def _expected_interval(self, source_name: str, configured: float) -> float:
        """How long silence from this source is normal, right now.

        `configured` is the setup-time number: the source's own `min_capture_interval`
        or the task-level rate limit for its provider type. That conflates two things
        which only coincide for a continuous stream — how often we *fetch*, and how
        often frames actually *arrive*. A source that publishes a cloud snapshot every
        30s is not stalled at 1s, but the rate limit says it is, and it gets reported
        as failing forever.

        So a provider that knows better gets to say so.
        """
        return self._extended_wait(getattr(self, "_providers_by_source", {}).get(source_name), configured)

    def _source_staleness(self, source_name: str, interval: float) -> tuple[str, float]:
        """(level, elapsed_s) for one live source, as a function of how many multiples
        of its configured fetch interval have elapsed since its last successful frame.

        Single source of truth for the escalation thresholds (_STALE_NOTICE_*_MULTIPLE)
        — shared by get_notices() (which only surfaces "info" and above) and the
        status-panel sources table (which shows every level, including "ok"), so the
        two can never independently drift apart on what counts as stale.
        """
        interval = self._expected_interval(source_name, interval)
        now = time.perf_counter()
        last = self._last_fetch_by_source.get(source_name)
        baseline = last if last is not None else getattr(self, "_run_started_at", now)
        elapsed = now - baseline
        multiple = (elapsed / interval) if interval > 0 else 0.0
        if multiple >= self._STALE_NOTICE_ERROR_MULTIPLE:
            level = "error"
        elif multiple >= self._STALE_NOTICE_WARNING_MULTIPLE:
            level = "warning"
        elif multiple >= self._STALE_NOTICE_INFO_MULTIPLE:
            level = "info"
        else:
            level = "ok"
        return level, elapsed

    @staticmethod
    def _source_key(provider) -> str:
        """The name a provider is tracked under. Matches how _source_names is built,
        so the failure map and the status table can never key on different things."""
        return provider.source_name or "unknown"

    def _record_source_failure(self, source_name: str, error: str, now: float) -> int:
        """Note a failed fetch and schedule the source's next attempt.

        Backoff doubles from _SOURCE_RETRY_BACKOFF_S per consecutive failure, capped
        at _SOURCE_RETRY_BACKOFF_MAX_S, so a source that is down for hours costs a
        couple of attempts an hour instead of hammering it every loop pass.
        Returns the new consecutive-failure count.
        """
        entry = self._source_failures.get(source_name)
        consecutive = (entry["consecutive"] + 1) if entry else 1
        # The exponent is clamped before it is used, not after. `2 ** (consecutive - 1)`
        # is an arbitrary-precision int evaluated before the multiply, so it keeps
        # doubling long after min() has pinned the backoff at the ceiling — and at 1024
        # consecutive failures the product leaves float range and raises
        # `OverflowError: int too large to convert to float`, ending the whole run over
        # one dead camera. 32 doublings is already ~4 billion times the base delay, so
        # nothing below the clamp is reachable before the ceiling wins.
        doublings = min(consecutive - 1, 32)
        backoff = min(self._SOURCE_RETRY_BACKOFF_S * (2**doublings), self._SOURCE_RETRY_BACKOFF_MAX_S)
        self._source_failures[source_name] = {
            "consecutive": consecutive,
            "error": error,
            "since": entry["since"] if entry else now,
            "retry_at": now + backoff,
        }
        return consecutive

    def _record_source_success(self, source_name: str, now: float) -> None:
        """Clear a source's failure state after a good frame.

        Recovery needs no bookkeeping of its own — the entry simply goes away, which
        is what makes the notice and the table's health cell clear themselves.
        """
        self._source_failures.pop(source_name, None)
        self._last_any_success_at = now

    def _source_in_backoff(self, source_name: str, now: float) -> bool:
        entry = self._source_failures.get(source_name)
        return entry is not None and now < entry["retry_at"]

    def get_notices(self) -> list[dict]:
        """Advisory notices for sources that have gone quiet.

        Escalates info → warning → error as multiples of the source's own configured
        fetch interval elapse without a successful frame (_STALE_NOTICE_*_MULTIPLE, via
        _source_staleness()). Recomputed fresh on every call from _last_fetch_by_source,
        so a notice clears (or downgrades) automatically the moment the source yields a
        frame again — no separate recovery bookkeeping needed.
        """
        status = self.get_status()
        if status not in (TaskStatus.RUNNING, TaskStatus.PAUSED):
            return []
        notices: list[dict] = []
        for source_name, interval in getattr(self, "_source_intervals", {}).items():
            level, elapsed = self._source_staleness(source_name, interval)
            if level == "ok":
                continue
            # Report the interval staleness was actually judged against, not the
            # setup-time one — otherwise a source whose provider asked for more
            # patience is described by a number that contradicts its own level.
            effective = self._expected_interval(source_name, interval)
            notices.append(
                {
                    "key": f"stale_source_{source_name}",
                    "level": level,
                    "message": (
                        f"No frames from '{source_name}' in {elapsed:.0f}s " f"(expected every {effective:.0f}s)"
                    ),
                }
            )
        # Staleness only says a source went quiet. When we know *why* — it raised or
        # timed out — say so, and cover FIXED sources too (they have no interval, so
        # the staleness pass above never speaks for them).
        now = time.perf_counter()
        for source_name, entry in getattr(self, "_source_failures", {}).items():
            down_for = now - entry["since"]
            notices.append(
                {
                    "key": f"failing_source_{source_name}",
                    "level": "error",
                    "message": (
                        f"'{source_name}' has failed {entry['consecutive']}× over {down_for:.0f}s "
                        f"— still retrying. Last error: {entry['error']}"
                    ),
                }
            )
        # The collector's own notices — today, the silent-failure guard for models
        # that cannot report runner-up classes. Sources are ours; what the models
        # can and can't do is its.
        collector = getattr(self, "_collector", None)
        if collector is not None:
            notices.extend(collector.notices())
        return notices

    def get_status_message(self) -> str:
        return self._status_msg

    def _review_actions(self) -> list[dict]:
        """The collector's ways into Review, shared by the live status card and the
        completion result so the two cannot drift apart.

        getattr, not attribute access: this is reachable through
        get_status_actions() before _init() has run — the same PENDING-window
        contract get_status_data() documents.
        """
        collector = getattr(self, "_collector", None)
        return collector.review_actions() if collector is not None else []

    def get_status_actions(self) -> list[dict]:
        return self._review_actions()

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        """A collector to run, plus the three params that belong to this loop.

        The ~30 knobs live on the collector row and are edited in Configuration →
        Collectors; only `source_uuids` / `snapshot_interval` / `stream_interval`
        are ours, because only a driver that rotates providers has sources or a
        pacing interval.
        """
        return {
            "meta": {"order": ["collector_uuid", "snapshot_interval", "stream_interval", "source_uuids"]},
            "collector_uuid": {
                "type": "training_collector_uuid",
                "required": True,
                "label": "Collector",
                "description": (
                    "Which stored collector decides what to keep and how to annotate it. "
                    "Edit collectors in Configuration → Collectors."
                ),
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
        }
