"""StreamWatcherTask — the NVR's live analysis loop.

Orchestration only. Tracking lives in `apps/helpers/tracking`, rules and geometry in
`apps/helpers/watch`, and both are pure; this task's whole job is to pace sources,
run inference, feed the pipeline, and dispatch what comes out. That split is
deliberate — `CollectTrainingImagesTask` is 2400 lines doing strictly less than this.

Per source, per analysed frame::

    provider.getNextImage()      # paced at the source's analysis fps
      → downscale to analysis_width
      → inference (model stack, merged by IoU + label hierarchy)
      → tracker.update(detections, now)     → tracks
      → engine.step(tracks, now)            → event START/UPDATE/END
      → dispatch                            → actions (Phase 6; logged here)

Two properties this loop must keep:

* **Model pins are held and renewed.** Inference auto-unloads a model after ~10
  idle minutes; a 24/7 watcher without renewal would thrash model loads forever.
* **One failing camera never stops the others.** Sources back off independently and
  the task keeps running, because a router reboot must not end a week-long watch.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import db_client
import inference_client
from db_client.api.model_groups_api import ModelGroupsApi
from db_client.api.watch_profiles_api import WatchProfilesApi
from inference_client.api.inference_api import InferenceApi
from inference_client.api.models_api import ModelsApi
from inference_client.models.inference_payload import InferencePayload
from inference_client.models.pin_model_payload import PinModelPayload

from apps.helpers.actions.action import ActionContext
from apps.helpers.actions.mqttPublisher import AsyncMqttPublisher, MqttSettings
from apps.helpers.actions.recordClipAction import PlaybackClipBackend, RecordClipAction
from apps.helpers.actions.Registry import create_action
from apps.helpers.consts import JsonValues, TaskStatus
from apps.helpers.imageUtils import base64_encode_jpg
from apps.helpers.tracking.Registry import create_tracker
from apps.helpers.tracking.tracker import Detection
from apps.helpers.watch.engine import WatchEngine
from apps.helpers.watch.events import EventPhase
from apps.helpers.watch.geometry import zone_from_row
from apps.helpers.watch.rules import RuleSpec

from ._detection_merger import MergePolicy, merge_detections_by_iou_and_hierarchy
from ._watch_pipeline import WatchPipeline, build_label_lifetimes, resolve_model_names
from .Registry import register_task
from .Task import Task
from .utils.sourceHelper import SourceHelper

logger = logging.getLogger(__name__)


@register_task()
class StreamWatcherTask(Task):
    """Watches a Watch Profile's sources and raises events from interaction rules."""

    # A watcher runs for weeks. Checkpoint far more often than the 300 s base default so
    # a restart loses seconds of open-event state, not minutes.
    CHECKPOINT_INTERVAL: int = 60

    # `_run` is `while not self._cancel_flag.is_set()`: this ends when the user
    # ends it, never on its own.
    CONTINUOUS: bool = True

    _PIN_DURATION_S: int = 3600
    _PIN_RENEWAL_INTERVAL_S: float = _PIN_DURATION_S / 2
    _FETCH_TIMEOUT_S: float = 30.0
    _INFERENCE_TIMEOUT_S: float = 60.0
    _MAX_RETRIES: int = 2
    # Ceiling on how long the loop sleeps when nothing is due. Short enough that a
    # newly-due source is serviced promptly, long enough that an idle watcher costs
    # nothing measurable.
    _IDLE_SLEEP_S: float = 0.05
    _MAX_IDLE_SLEEP_S: float = 0.5

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # get_status_data() can be polled in the PENDING window before _init() runs,
        # so every field it reads needs a safe value here rather than only in _init().
        self._profile: Optional[Any] = None
        self._pipelines: List[WatchPipeline] = []
        self._pin_ids: Dict[str, Optional[str]] = {}
        self._dry_run: bool = False
        self._analysis_fps_cap: Optional[float] = None
        self._status_msg: str = "Starting"
        self._events_started: int = 0
        self._events_ended: int = 0
        self._frames_analysed: int = 0
        self._inference_ms: List[float] = []
        self._recent_events: List[dict] = []
        self._open_event_uuids: Dict[str, str] = {}
        self._run_started_at: float = time.monotonic()
        self._label_ancestry: Dict[str, set] = {}
        self._actions: List[tuple] = []
        self._mqtt: Optional[AsyncMqttPublisher] = None
        self._action_successes: int = 0
        self._action_failures: int = 0
        self._last_clip_path: Optional[str] = None
        # Converts the loop's monotonic timestamps to wallclock, which is how the
        # rolling buffer is addressed. Captured once so a clock adjustment mid-run
        # can't make two events disagree about when 'now' was.
        self._wallclock_offset: float = time.time() - time.monotonic()
        self._progress = None  # a watcher has no meaningful completion percentage

    # ------------------------------------------------------------------ schema

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        return {
            "watch_profile_uuid": {
                "type": "watch_profile_uuid",
                "required": True,
                "help": "The Watch Profile this task executes.",
            },
            "dry_run": {
                "type": "bool",
                "required": False,
                "default": False,
                "help": (
                    "Evaluate rules and raise events, but perform no actions — no MQTT "
                    "publishes and no clips written. Use this while tuning thresholds."
                ),
            },
            "analysis_fps_cap": {
                "type": "float",
                "required": False,
                "default": 0.0,
                "help": (
                    "Optional ceiling applied to every source's analysis rate, for "
                    "temporarily reducing load without editing the profile. 0 = no cap."
                ),
            },
        }

    @classmethod
    def status_schema(cls) -> list[dict[str, Any]]:
        return [
            {"id": "statusMessage", "type": "text", "source": "summary"},
            {
                "id": "stats",
                "type": "stat_grid",
                "label": "Activity",
                "stats": [
                    {"id": "framesAnalysed", "label": "Frames analysed"},
                    {"id": "eventsStarted", "label": "Events started"},
                    {"id": "eventsEnded", "label": "Events ended"},
                    {"id": "openEvents", "label": "Open events"},
                    {"id": "activeTracks", "label": "Active tracks"},
                    {"id": "inferenceP50Ms", "label": "Inference p50 (ms)"},
                    {"id": "inferenceP95Ms", "label": "Inference p95 (ms)"},
                    {"id": "actionsOk", "label": "Actions ok"},
                    {"id": "actionsFailed", "label": "Actions failed"},
                    {"id": "mqttPublished", "label": "MQTT published"},
                    {"id": "mqttDropped", "label": "MQTT dropped"},
                ],
            },
            {
                "id": "sources",
                "type": "table",
                "label": "Sources",
                "columns": [
                    {"id": "source", "label": "Source"},
                    {"id": "analysisFps", "label": "fps"},
                    {"id": "framesAnalysed", "label": "Frames"},
                    {"id": "lastFrameAgoS", "label": "Last frame (s)"},
                    {"id": "buffer", "label": "Buffer"},
                    {"id": "tracks", "label": "Tracks"},
                    {"id": "health", "label": "Health"},
                    {"id": "lastError", "label": "Last error"},
                ],
            },
            {
                "id": "activeTracksTable",
                "type": "table",
                "label": "Active tracks",
                "columns": [
                    {"id": "source", "label": "Source"},
                    {"id": "trackId", "label": "Track"},
                    {"id": "label", "label": "Label"},
                    {"id": "ageS", "label": "Age (s)"},
                    {"id": "confidence", "label": "Conf"},
                    {"id": "anchor", "label": "Anchor"},
                ],
            },
            {
                "id": "openEventsTable",
                "type": "table",
                "label": "Open events",
                "columns": [
                    {"id": "ruleName", "label": "Rule"},
                    {"id": "sourceUuid", "label": "Source"},
                    {"id": "state", "label": "State"},
                    {"id": "startedAt", "label": "Started"},
                ],
            },
            {
                "id": "recentEvents",
                "type": "table",
                "label": "Recent events",
                "columns": [
                    {"id": "ruleName", "label": "Rule"},
                    {"id": "phase", "label": "Phase"},
                    {"id": "durationS", "label": "Duration (s)"},
                    {"id": "truncated", "label": "Capped"},
                ],
            },
            {"id": "tuningHints", "type": "text", "label": "Tuning"},
            {"id": "log", "type": "log_tail", "source": "summary", "lines": 8},
        ]

    # ------------------------------------------------------------------ init

    async def _init(self, params: dict[str, Any], resume_data: Optional[dict[str, Any]]) -> None:
        # Params arrive as raw JSON and are unvalidated by the framework — cast here.
        profile_uuid = str(params.get("watch_profile_uuid") or "").strip()
        if not profile_uuid:
            raise ValueError("StreamWatcherTask requires a watch_profile_uuid")
        self._profile_uuid = profile_uuid
        self._dry_run = bool(params.get("dry_run", False))
        cap = params.get("analysis_fps_cap")
        try:
            cap_value = float(cap) if cap not in (None, "") else 0.0
        except (TypeError, ValueError):
            cap_value = 0.0
        self._analysis_fps_cap = cap_value if cap_value > 0 else None

        # resume_data is wholesale-replaced on every checkpoint, so it may only hold
        # state this task itself owns. Open events are exactly that.
        self._open_event_uuids = dict((resume_data or {}).get("open_event_uuids") or {})

        db_api_client = self._manager.get_db_api_client()
        watch_api = WatchProfilesApi(db_api_client)
        resp = await asyncio.to_thread(watch_api.get_watch_profile, profile_uuid)
        if resp.status != JsonValues.SUCCESS or resp.profile is None:
            raise ValueError(f"Watch profile {profile_uuid} not found: {resp.message}")
        self._profile = resp.profile
        if not self._profile.sources:
            raise ValueError(f"Watch profile {self._profile.name!r} has no sources to watch")

        self.log(f"Loaded watch profile {self._profile.name!r} with {len(self._profile.sources)} source(s)")
        if self._dry_run:
            self.log("DRY RUN — events will be raised but no actions performed")

    # ------------------------------------------------------------------ setup

    async def _build_pipelines(self, db_api_client: db_client.ApiClient) -> None:
        """Construct one pipeline per enabled source in the profile."""
        groups_resp = await asyncio.to_thread(ModelGroupsApi(db_api_client).list_model_groups)
        groups_by_uuid = {g.uuid: g for g in (groups_resp.groups or [])}

        source_helper = SourceHelper(db_api_client)
        lifetimes = build_label_lifetimes(self._profile.label_lifetimes)
        rule_specs = [RuleSpec.from_row(r) for r in self._profile.rules if r.enabled]

        for src in sorted(self._profile.sources, key=lambda s: s.order):
            if not src.enabled:
                continue
            try:
                provider = await asyncio.to_thread(source_helper.get_image_provider, src.source_uuid)
            except Exception as e:
                # One unresolvable source must not stop the rest of the watcher.
                logger.exception("Could not build provider for source %s", src.source_uuid)
                self.log(f"Skipping source {src.source_uuid}: {e}")
                continue

            fps = src.analysis_fps or self._profile.default_analysis_fps
            if self._analysis_fps_cap:
                fps = min(fps, self._analysis_fps_cap)

            tracker = create_tracker(
                self._profile.tracker_typename,
                label_lifetimes=lifetimes,
                **(src.tracker_params_json or self._profile.tracker_params_json or {}),
            )
            zones = [zone_from_row(z) for z in src.zones]
            engine = WatchEngine(rules=rule_specs, zones=zones, source_uuid=src.source_uuid)

            # A source that names either a stack or a model overrides the profile
            # wholesale; only a source that names neither inherits the default. Falling
            # back field-by-field would let a source's single model be silently union-ed
            # with the profile's default stack.
            if src.model_group_uuid or src.model_name:
                model_names = resolve_model_names(src.model_group_uuid, src.model_name, groups_by_uuid)
            else:
                model_names = resolve_model_names(
                    self._profile.default_model_group_uuid, self._profile.default_model_name, groups_by_uuid
                )
            if not model_names:
                self.log(f"Source {src.source_uuid} has no models configured — nothing will be detected on it")

            self._pipelines.append(
                WatchPipeline(
                    source_uuid=src.source_uuid,
                    source_name=getattr(provider, "source_name", None) or src.source_uuid,
                    provider=provider,
                    tracker=tracker,
                    engine=engine,
                    model_names=model_names,
                    analysis_fps=fps,
                    analysis_width=src.analysis_width or self._profile.default_analysis_width,
                )
            )

        if not self._pipelines:
            raise RuntimeError("No usable sources — every source in this profile failed to build a provider")

    async def _build_actions(self, db_api_client: db_client.ApiClient) -> None:
        """Instantiate the profile's actions, wiring in shared resources.

        The MQTT connection and the clip backend are owned here rather than inside each
        action: a watcher needs one broker connection, not one per action row, and the
        clip backend depends on the relay client the task already holds.
        """
        mqtt_settings = await self._load_mqtt_settings(db_api_client)
        needs_mqtt = any(a.kind == "publish_mqtt" and a.enabled for a in self._profile.actions)
        if needs_mqtt and mqtt_settings is not None:
            self._mqtt = AsyncMqttPublisher(mqtt_settings)
            await self._mqtt.start()
        elif needs_mqtt:
            self.log("An MQTT action is configured but no broker is set in Settings — publishes will be skipped")

        relay_client = self._manager.get_relay_api_client() if hasattr(self._manager, "get_relay_api_client") else None

        for row in sorted(self._profile.actions, key=lambda a: a.order):
            if not row.enabled:
                continue
            params = dict(row.params_json or {})
            try:
                if row.kind == "publish_mqtt":
                    action = create_action(row.kind, publisher=self._mqtt, **params)
                elif row.kind == "record_clip":
                    backend = PlaybackClipBackend(relay_client) if relay_client else None
                    action = create_action(row.kind, backend=backend, **params)
                else:
                    action = create_action(row.kind, **params)
            except Exception as e:
                # An unknown or misconfigured action must not stop the watcher starting;
                # the other actions are still worth running.
                logger.exception("Could not build action %s", row.kind)
                self.log(f"Skipping action {row.kind}: {e}")
                continue
            self._actions.append((row, action))

        if self._actions:
            self.log(f"Loaded {len(self._actions)} action(s)")

    async def _load_mqtt_settings(self, db_api_client: db_client.ApiClient) -> Optional[MqttSettings]:
        """Read broker config from global Settings.

        Deliberately not duplicated into every action's params: one broker per
        deployment, referenced by actions rather than restated by them.
        """
        try:
            from db_client.api.settings_api import SettingsApi

            resp = await asyncio.to_thread(SettingsApi(db_api_client).get_setting, "mqtt_broker")
            value = getattr(resp, "value", None)
            if not value:
                return None
            return MqttSettings.from_dict(value if isinstance(value, dict) else {})
        except Exception:
            logger.debug("No MQTT broker configured in Settings", exc_info=True)
            return None

    async def _pin_models(self, inference_api_client: inference_client.ApiClient) -> None:
        """Hold a pin on every model for the task's lifetime.

        Without this the ~10-minute idle auto-unload would evict models between frames
        on a slow-fps watcher, so every inference call would pay a model load.
        """
        models_api = ModelsApi(inference_api_client)
        names = {name for p in self._pipelines for name in p.model_names}
        for name in sorted(names):
            try:
                resp = await asyncio.to_thread(
                    models_api.pin_model_api_api_models_model_name_pin_post,
                    name,
                    PinModelPayload(model_name=name, duration=self._PIN_DURATION_S),
                )
                self._pin_ids[name] = resp.pin_id
                if not resp.pin_id:
                    self.log(f"Warning: could not pin '{name}' — it will reload on each inference")
            except Exception:
                logger.exception("Failed to pin model %s", name)
                self._pin_ids[name] = None

    async def _renew_pins_periodically(self, inference_api_client: inference_client.ApiClient) -> None:
        """Re-pin on a timer so a run longer than the pin duration never lapses."""
        models_api = ModelsApi(inference_api_client)
        try:
            while not self._cancel_flag.is_set():
                await asyncio.sleep(self._PIN_RENEWAL_INTERVAL_S)
                for name in list(self._pin_ids):
                    try:
                        resp = await asyncio.to_thread(
                            models_api.pin_model_api_api_models_model_name_pin_post,
                            name,
                            PinModelPayload(model_name=name, duration=self._PIN_DURATION_S),
                        )
                        old = self._pin_ids.get(name)
                        self._pin_ids[name] = resp.pin_id
                        if old and resp.pin_id and old != resp.pin_id:
                            try:
                                await asyncio.to_thread(models_api.unpin_model, name, old)
                            except Exception:
                                logger.debug("Could not release superseded pin %s for %s", old, name)
                    except Exception:
                        # A failed renewal must never kill the watcher; the worst case
                        # is a model reload on the next inference.
                        logger.warning("Pin renewal failed for %s", name, exc_info=True)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------ inference

    async def _run_inference(
        self, inference_api: InferenceApi, base64_img: str, model_names: List[str]
    ) -> List[Detection]:
        """Run a source's model stack and merge the results."""
        if not model_names:
            return []

        async def _run_model(name: str) -> list:
            for attempt in range(self._MAX_RETRIES):
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            inference_api.detect,
                            name,
                            InferencePayload(image_base64=base64_img, pin_id=self._pin_ids.get(name)),
                        ),
                        timeout=self._INFERENCE_TIMEOUT_S,
                    )
                    return list(result.result.detections or [])
                except asyncio.TimeoutError:
                    if attempt == self._MAX_RETRIES - 1:
                        self.log(f"Inference timed out on '{name}' — skipping frame")
                except Exception:
                    logger.exception("Inference error on model %s", name)
                    break
            return []

        started = time.perf_counter()
        results_per_model = await asyncio.gather(*(_run_model(n) for n in model_names))
        self._record_inference_time((time.perf_counter() - started) * 1000.0)

        merged = merge_detections_by_iou_and_hierarchy(
            results_per_model,
            label_ancestry=self._label_ancestry,
            overlap_iou_threshold=0.5,
            policy=MergePolicy.FIRST_WINS,
        )
        return [
            Detection(
                bbox=(det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2),
                label_uuid=det.bbox.label_uuid,
                label_name=getattr(det.bbox, "label_name", None),
                confidence=getattr(det, "confidence", 0.0) or 0.0,
            )
            for det in merged
        ]

    def _record_inference_time(self, ms: float) -> None:
        # Bounded: this is a "what is it doing now" sample, not a full history.
        self._inference_ms.append(ms)
        if len(self._inference_ms) > 500:
            del self._inference_ms[:-500]

    # ------------------------------------------------------------------ run

    async def _run(self) -> dict[str, Any]:
        self._run_started_at = time.monotonic()
        db_api_client = self._manager.get_db_api_client()
        inference_api_client = self._manager.get_inference_api_client()
        inference_api = InferenceApi(inference_api_client)

        self._status_msg = "Building pipelines"
        await self._build_pipelines(db_api_client)

        self._status_msg = "Loading actions"
        await self._build_actions(db_api_client)

        self._status_msg = "Pinning models"
        await self._pin_models(inference_api_client)
        renewal = asyncio.create_task(self._renew_pins_periodically(inference_api_client))

        self._status_msg = "Starting sources"
        for pipeline in self._pipelines:
            try:
                await pipeline.provider.start()
            except Exception as e:
                logger.exception("Failed to start source %s", pipeline.source_uuid)
                pipeline.health.record_failure(time.monotonic(), str(e))
            # The relay tells us whether this source actually has a rewind buffer;
            # never infer it locally, or the UI badge and the clip backend can disagree.
            pipeline.buffered = bool(getattr(pipeline.provider, "buffered", False))
            pipeline.ingest_tier = getattr(pipeline.provider, "ingest_tier_name", "none")

        self._status_msg = "Watching"
        try:
            while not self._cancel_flag.is_set():
                await self._check_pause()
                if self._data_req_flag.is_set():
                    # Mid-run persistence takes priority over cancel, per the Task contract.
                    self.set_resume_data(self._build_watch_resume_data())
                    self.clear_data_request()

                now = time.monotonic()
                due = [p for p in self._pipelines if p.due(now)]
                if not due:
                    await asyncio.sleep(self._idle_sleep(now))
                    continue

                for pipeline in due:
                    if self._cancel_flag.is_set():
                        break
                    await self._service_source(pipeline, inference_api)

                await self.heartbeat()
        finally:
            renewal.cancel()
            self._status_msg = "Closing open events"
            # An event open at shutdown must be closed, not stranded — its clip is
            # still extractable while the buffer window holds.
            for pipeline in self._pipelines:
                for signal in pipeline.close_open_events(time.monotonic()):
                    self._note_event(signal)
                    await self._dispatch(signal, pipeline)

        return {
            "eventsStarted": self._events_started,
            "eventsEnded": self._events_ended,
            "framesAnalysed": self._frames_analysed,
        }

    def _idle_sleep(self, now: float) -> float:
        """Sleep until the next source is due, bounded so cancel stays responsive."""
        upcoming = [p._next_frame_at for p in self._pipelines if p._next_frame_at > now]
        if not upcoming:
            return self._IDLE_SLEEP_S
        return max(self._IDLE_SLEEP_S, min(self._MAX_IDLE_SLEEP_S, min(upcoming) - now))

    async def _service_source(self, pipeline: WatchPipeline, inference_api: InferenceApi) -> None:
        """Fetch, analyse and step one source. Never raises — failures back off."""
        now = time.monotonic()
        pipeline.schedule_next(now)
        try:
            provided = await asyncio.wait_for(pipeline.provider.getNextImage(), timeout=self._FETCH_TIMEOUT_S)
        except asyncio.TimeoutError:
            pipeline.health.record_failure(now, "frame fetch timed out")
            return
        except Exception as e:
            logger.exception("Error fetching from source %s", pipeline.source_uuid)
            pipeline.health.record_failure(now, str(e))
            return

        if provided is None:
            pipeline.health.record_failure(now, "source returned no frame")
            return

        analysed = pipeline.prepare_frame(provided.image)
        # The prefetched encoding is of the FULL-size frame, so it may only be reused
        # when no downscale happened. Detections come back in the coordinate space of
        # whatever image was sent, and zones are scaled to the analysis frame — sending
        # the full-size image would silently offset every rule.
        if provided.encoded_b64 is not None and analysed.shape == provided.image.shape:
            base64_img = provided.encoded_b64
        else:
            base64_img = await asyncio.to_thread(base64_encode_jpg, analysed)

        detections = await self._run_inference(inference_api, base64_img, pipeline.model_names)

        now = time.monotonic()
        pipeline.health.record_success(now)
        self._frames_analysed += 1

        for signal in pipeline.step(detections, now):
            self._note_event(signal)
            await self._dispatch(signal, pipeline)

    def _note_event(self, signal) -> None:
        if signal.phase is EventPhase.START:
            self._events_started += 1
        elif signal.phase is EventPhase.END:
            self._events_ended += 1
        self._recent_events.append(signal.to_dict())
        if len(self._recent_events) > 50:
            del self._recent_events[:-50]

    def _actions_for(self, signal) -> List[Any]:
        """Actions that should fire for this transition.

        An action with no `rule_uuid` applies to every rule in the profile; one with a
        `rule_uuid` is scoped to that rule alone.
        """
        trigger = {
            EventPhase.START: "event_started",
            EventPhase.UPDATE: "event_started",
            EventPhase.END: "event_ended",
        }[signal.phase]
        return [
            (row, action)
            for row, action in self._actions
            if row.enabled and row.trigger == trigger and (row.rule_uuid in (None, "", signal.rule_uuid))
        ]

    async def _dispatch(self, signal, pipeline: Optional[WatchPipeline] = None) -> None:
        """Hand an event transition to the action layer."""
        verb = {EventPhase.START: "started", EventPhase.UPDATE: "updated", EventPhase.END: "ended"}[signal.phase]
        detail = f" ({signal.duration_s:.1f}s)" if signal.phase is EventPhase.END else ""
        capped = " [capped by max duration]" if signal.truncated else ""
        self.log(f"Event {verb}: {signal.rule_name}{detail}{capped} tracks={signal.track_ids}")
        if self._dry_run:
            return

        context = self._action_context(signal, pipeline)
        for row, action in self._actions_for(signal):
            try:
                result = await action.run(signal, context)
            except Exception as e:
                # An action must never be able to kill the frame loop. Defensive even
                # though Action.run promises not to raise — a third-party action might.
                logger.exception("Action %s raised", row.kind)
                self._action_failures += 1
                self.log(f"Action {row.kind} errored: {e}")
                continue
            if result.ok:
                self._action_successes += 1
                if result.data.get("path"):
                    self._last_clip_path = result.data["path"]
            else:
                self._action_failures += 1
                self.log(f"Action {row.kind} failed: {result.detail}")

    def _action_context(self, signal, pipeline: Optional[WatchPipeline]) -> ActionContext:
        return ActionContext(
            profile_uuid=self._profile.uuid,
            profile_name=self._profile.name,
            source_uuid=signal.source_uuid,
            source_name=pipeline.source_name if pipeline else None,
            relay_path=getattr(pipeline.provider, "_relayed_name", None) if pipeline else None,
            buffered=bool(pipeline and pipeline.buffered),
            retention_s=pipeline.retention_s if pipeline else 0.0,
            frame=pipeline.last_frame if pipeline else None,
            # Rule timing is monotonic; the buffer is addressed by wallclock. Carry the
            # offset once rather than making every action re-derive it.
            wallclock_offset=self._wallclock_offset,
        )

    def _build_watch_resume_data(self) -> dict[str, Any]:
        """State this task owns, for checkpointing.

        Track ids deliberately do NOT survive a restart and must not pretend to; only
        open events and their start times are worth carrying across.
        """
        open_events = {}
        for pipeline in self._pipelines:
            for entry in pipeline.engine.open_events():
                open_events[f"{pipeline.source_uuid}:{entry['ruleUuid']}"] = entry["startedAt"]
        return {
            "open_event_uuids": self._open_event_uuids,
            "open_events": open_events,
            "events_started": self._events_started,
            "events_ended": self._events_ended,
        }

    async def _deinit(self) -> None:
        for _row, action in self._actions:
            try:
                await action.close()
            except Exception:
                logger.debug("Error closing action", exc_info=True)
        if self._mqtt is not None:
            # Explicit lifecycle rather than __del__: a destructor-time disconnect
            # fires at GC's whim, which for a network client is unpredictable.
            await self._mqtt.close()
            self._mqtt = None

        for pipeline in self._pipelines:
            try:
                await pipeline.provider.stop()
            except Exception:
                logger.warning("Error stopping source %s", pipeline.source_uuid, exc_info=True)

        models_api = ModelsApi(self._manager.get_inference_api_client())
        for name, pin_id in self._pin_ids.items():
            if not pin_id:
                continue
            try:
                await asyncio.to_thread(models_api.unpin_model, name, pin_id)
            except Exception:
                logger.debug("Could not unpin %s", name, exc_info=True)

    # ------------------------------------------------------------------ status

    def get_status_data(self) -> dict[str, Any]:
        """Cheap and non-blocking — this runs inside a poll handler."""
        now = time.monotonic()
        tracks_rows = []
        open_rows = []
        hints: List[str] = []
        for pipeline in self._pipelines:
            for track in pipeline.tracks:
                tracks_rows.append(
                    {
                        "source": pipeline.source_name,
                        "trackId": track.track_id,
                        "label": track.label_name or track.label_uuid,
                        "ageS": round(track.age_s(now), 1),
                        "confidence": round(track.confidence, 3),
                        "anchor": "yes" if track.is_anchor else "",
                    }
                )
            open_rows.extend(pipeline.engine.open_events())
            hints.extend(pipeline.engine.tuning_hints())

        return {
            "stats": {
                "framesAnalysed": self._frames_analysed,
                "eventsStarted": self._events_started,
                "eventsEnded": self._events_ended,
                "openEvents": len(open_rows),
                "activeTracks": len(tracks_rows),
                "inferenceP50Ms": self._percentile(50),
                "inferenceP95Ms": self._percentile(95),
                "actionsOk": self._action_successes,
                "actionsFailed": self._action_failures,
                "mqttPublished": (self._mqtt.published if self._mqtt else 0),
                # A dropped publish means the broker is gone rather than slow; surfacing
                # it is the difference between a visible problem and a silent one.
                "mqttDropped": (self._mqtt.dropped if self._mqtt else 0),
            },
            "sources": [p.status_row(now) for p in self._pipelines],
            "activeTracksTable": tracks_rows,
            "openEventsTable": open_rows,
            "recentEvents": list(reversed(self._recent_events[-15:])),
            "tuningHints": "\n".join(hints) if hints else "No tuning suggestions yet.",
        }

    def _percentile(self, pct: int) -> Optional[float]:
        if not self._inference_ms:
            return None
        ordered = sorted(self._inference_ms)
        idx = min(len(ordered) - 1, int(len(ordered) * pct / 100))
        return round(ordered[idx], 1)

    def get_status_message(self) -> str:
        return self._status_msg
