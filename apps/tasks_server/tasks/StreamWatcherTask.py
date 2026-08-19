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
import base64
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import db_client
import inference_client
from db_client.api.labels_api import LabelsApi
from db_client.api.model_groups_api import ModelGroupsApi
from db_client.api.sources_api import SourcesApi
from db_client.api.watch_profiles_api import WatchProfilesApi
from db_client.models.get_source_regions_payload import GetSourceRegionsPayload
from inference_client.api.inference_api import InferenceApi
from inference_client.api.models_api import ModelsApi
from inference_client.models.batch_inference_payload import BatchInferencePayload
from inference_client.models.pin_model_payload import PinModelPayload

from apps.helpers.actions.action import ActionContext
from apps.helpers.actions.mqttPublisher import AsyncMqttPublisher, MqttSettings
from apps.helpers.actions.recordClipAction import FrameRing, PlaybackClipBackend, RecordClipAction, RingClipBackend
from apps.helpers.actions.Registry import create_action
from apps.helpers.consts import JsonValues, TaskStatus
from apps.helpers.imageUtils import base64_encode_jpg
from apps.helpers.recordings.clipDrain import ClipDrain, sweep_chunk_root
from apps.helpers.recordings.detectionTrack import DEFAULT_TRACK_SECONDS, DetectionTrack
from apps.helpers.tracking.Registry import create_tracker
from apps.helpers.tracking.tracker import Detection
from apps.helpers.watch.engine import WatchEngine
from apps.helpers.watch.events import EventPhase, WatchEventSignal
from apps.helpers.watch.geometry import zone_from_row
from apps.helpers.watch.rules import RuleSpec, expand_descendants
from apps.helpers.watch.scoping import FRAME_ANALYSED, action_applies, in_scope

from ._detection_merger import MergePolicy, merge_detections_by_iou_and_hierarchy
from ._watch_event_recorder import NullWatchEventRecorder, WatchEventRecorder
from ._watch_pipeline import WatchPipeline, _format_duration, build_label_lifetimes, resolve_model_names
from .Registry import register_task
from .Task import Task
from .utils.labelHelper import build_label_ancestry_map, build_label_metadata_map
from .utils.sourceHelper import SourceHelper

logger = logging.getLogger(__name__)

#: Action kinds that need the watcher's shared broker connection injected. Membership,
#: not an equality check, because there are two of them now and a third would otherwise
#: silently build with `publisher=None` and report "no broker configured" forever.
MQTT_ACTION_KINDS = frozenset({"publish_mqtt", "publish_mqtt_image"})
#: The one action kind whose output is a playable clip, and so the only one that needs
#: the rolling buffer, the drain and a detection overlay.
CLIP_ACTION_KIND = "record_clip"
#: Action kind → the `recordings.kind` its result becomes. Membership is what decides
#: whether a result's `path` may become a `recordings` row at all: `save_snapshot` and
#: `save_timelapse` both report a `path` too, and treating either as an event's video
#: would put a jpg behind a `<video>` element.
MEDIA_ACTION_KINDS = {
    CLIP_ACTION_KIND: "clip",
    "record_event_timelapse": "timelapse",
}
#: Shared, stateless, and never used for a real run — see the class docstring.
_NULL_RECORDER = NullWatchEventRecorder()


@register_task()
class StreamWatcherTask(Task):
    """Watches a Watch Profile's sources and raises events from interaction rules."""

    # A watcher runs for weeks. Checkpoint far more often than the 300 s base default so
    # a restart loses seconds of open-event state, not minutes.
    CHECKPOINT_INTERVAL: int = 60

    # `_run` is `while not self._cancel_flag.is_set()`: this ends when the user
    # ends it, never on its own.
    CONTINUOUS: bool = True

    # A watcher is the one task whose whole value is that it is still running.
    # Left non-persistent, a service restart stops it silently: nothing errors,
    # the cameras are simply no longer watched until somebody notices. It
    # checkpoints every 60 s (above), so it is genuinely resumable.
    DEFAULT_PERSISTENT: bool = True

    _PIN_DURATION_S: int = 3600
    _PIN_RENEWAL_INTERVAL_S: float = _PIN_DURATION_S / 2
    _FETCH_TIMEOUT_S: float = 30.0
    # Rebuilding a reader is spawning a process, not waiting for a camera — this bounds
    # the spawn only, so it stays short. The waiting is the caller's backoff.
    _RECONNECT_TIMEOUT_S: float = 15.0
    _INFERENCE_TIMEOUT_S: float = 60.0
    _MAX_RETRIES: int = 2
    # Deliberately low: thresholding is the *tracker's* job, not this call's.
    # ByteTrack's second association pass exists to recover boxes that a conventional
    # threshold throws away — filtering here would quietly reduce it to plain SORT, and
    # the loss would be invisible (fewer track ids, no error). Each tracker applies its
    # own real threshold, exposed through `params_schema()` and tunable in the UI.
    _INFERENCE_CONF_FLOOR: float = 0.1
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
        #: Source uuids currently past the failed threshold. Drives the task-level
        #: status message, so a watcher with a dead camera stops reporting the
        #: unqualified "Watching" it reports when everything is fine.
        self._failed_sources: Set[str] = set()
        self._pin_ids: Dict[str, Optional[str]] = {}
        self._dry_run: bool = False
        self._analysis_fps_cap: Optional[float] = None
        self._status_msg: str = "Starting"
        self._events_started: int = 0
        self._events_ended: int = 0
        self._frames_analysed: int = 0
        self._inference_ms: List[float] = []
        self._recent_events: List[dict] = []
        # Writes the durable event history. Built in `_run` rather than here because it
        # needs the resolved profile uuid. `_open_event_uuids` used to live here as
        # unused scaffolding for exactly this job; the recorder owns that map now.
        self._recorder: WatchEventRecorder = _NULL_RECORDER
        self._run_started_at: float = time.monotonic()
        self._label_ancestry: Dict[str, set] = {}
        # label_uuid → {"text": name, "color": color}. Inference returns a bare
        # label_uuid, so without this every detection, track and live-view box would
        # be captioned with a raw UUID (or nothing at all).
        self._label_meta: Dict[str, Dict[str, Optional[str]]] = {}
        self._actions: List[tuple] = []
        # The `frame_analysed` subset, kept apart so the per-frame path can skip its
        # whole dispatch with one truth test. Every watcher pays that test 5x a second
        # per source; almost none of them have a live feed configured.
        self._frame_actions: List[tuple] = []
        self._mqtt: Optional[AsyncMqttPublisher] = None
        self._action_successes: int = 0
        self._action_failures: int = 0
        # action uuid → the last failure message logged for it, so a per-frame action
        # against a dead broker reports once rather than flooding the log ring.
        self._last_action_failure: Dict[str, str] = {}
        self._last_clip_path: Optional[str] = None
        # Converts the loop's monotonic timestamps to wallclock, which is how the
        # rolling buffer is addressed. Captured once so a clock adjustment mid-run
        # can't make two events disagree about when 'now' was.
        self._wallclock_offset: float = time.time() - time.monotonic()
        # source_uuid → (frame_seq, quality, base64 jpeg). Populated only by the live
        # view, so a watcher with no viewer never allocates one.
        self._preview_cache: Dict[str, tuple] = {}
        # source_uuid → FrameRing, for sources with no rewind buffer. Created on first
        # frame, so a profile of RTSP cameras allocates none of them.
        self._frame_rings: Dict[str, FrameRing] = {}
        # source_uuid → ClipDrain, for buffered sources with an event open. Created on
        # the first open event and discarded when the last one closes, so a watcher
        # whose events all fit inside the buffer never allocates one — which is most of
        # them. See `apps.helpers.recordings.clipDrain` for why this exists at all.
        self._clip_drains: Dict[str, ClipDrain] = {}
        # source_uuid → DetectionTrack, for sources whose clips carry a detection
        # overlay. Created on first frame and only where some in-scope `record_clip`
        # asked for one, so a watcher with no clip actions — or one with the checkbox
        # off — allocates nothing and pays one dict lookup per frame. A None value means
        # "asked, and this source wants none"; see `_detection_track_for`.
        self._detection_tracks: Dict[str, Optional[DetectionTrack]] = {}
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
                    {"id": "mqttCoalesced", "label": "MQTT superseded"},
                    {"id": "mqttQueuedKb", "label": "MQTT queued (KB)"},
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
        # Both maps are keyed by label uuid and read on the per-frame path, so they are
        # fetched exactly once here. The ancestry map is what lets the cross-model merge
        # treat "cat" and "animal" as the same object rather than two overlapping ones —
        # left empty (as it was) the hierarchy arm of the merge silently never fires.
        labels_api = LabelsApi(db_api_client)
        self._label_ancestry = await build_label_ancestry_map(labels_api)
        self._label_meta = await build_label_metadata_map(labels_api)

        groups_resp = await asyncio.to_thread(ModelGroupsApi(db_api_client).list_model_groups)
        groups_by_uuid = {g.uuid: g for g in (groups_resp.groups or [])}

        source_helper = SourceHelper(db_api_client)
        lifetimes = build_label_lifetimes(self._profile.label_lifetimes)
        # Rules whose label slots opted into "include children" are widened here, once,
        # against the ancestry map fetched just above. Expanding at start rather than
        # storing the closure means a label added to the tree later is picked up on the
        # watcher's next restart, instead of never.
        rule_specs = [
            expand_descendants(RuleSpec.from_row(r), self._label_ancestry) for r in self._profile.rules if r.enabled
        ]
        for spec in rule_specs:
            if spec.subject_a_include_descendants or spec.subject_b_include_descendants:
                self.log(
                    f"Rule {spec.name!r} covers {len(spec.subject_a_refs)} subject label(s) "
                    f"and {len(spec.subject_b_refs)} target(s) after including descendants"
                )

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
            # Regions belong to the source, so they are fetched per source rather than
            # read off the profile. Masks are filtered by this watcher's opt-out list
            # here, once, rather than per frame.
            regions = await asyncio.to_thread(self._fetch_source_regions, db_api_client, src.source_uuid)
            disabled_masks = set(src.disabled_mask_uuids_json or [])
            zones = [zone_from_row(r) for r in regions if r.role == "zone"]
            masks = [zone_from_row(r) for r in regions if r.role == "mask" and r.uuid not in disabled_masks]
            # Every rule is handed to every source's engine; the engine keeps only the
            # ones scoped to it (empty scope = all). Filtering there rather than here
            # keeps "which rules run on this camera" a single answer, so a preview or a
            # future second caller cannot get it subtly wrong.
            engine = WatchEngine(rules=rule_specs, zones=zones, source_uuid=src.source_uuid)
            skipped = len(rule_specs) - len(engine.rule_uuids)
            if skipped:
                self.log(f"Source {src.source_uuid} skips {skipped} rule(s) scoped to other sources")

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
                    mqtt_name=getattr(src, "mqtt_name", None) or None,
                    provider=provider,
                    tracker=tracker,
                    engine=engine,
                    model_names=model_names,
                    analysis_fps=fps,
                    analysis_width=src.analysis_width or self._profile.default_analysis_width,
                    ring_seconds=getattr(src, "ring_seconds", 20.0) or 20.0,
                    ring_max_mb=getattr(src, "ring_max_mb", 32.0) or 32.0,
                    buffer_mode=str(getattr(src, "buffer_mode", "auto") or "auto"),
                    masks=masks,
                )
            )
            if masks:
                self.log(f"Source {src.source_uuid}: {len(masks)} mask(s) active")

        if not self._pipelines:
            raise RuntimeError("No usable sources — every source in this profile failed to build a provider")

    @staticmethod
    def _fetch_source_regions(db_api_client: db_client.ApiClient, source_uuid: str) -> List[Any]:
        """Zones and masks drawn on one source. Blocking — called via `to_thread`."""
        response = SourcesApi(db_api_client).get_source_regions(
            GetSourceRegionsPayload(source_uuid=source_uuid),
        )
        if response.status != JsonValues.SUCCESS:
            # A source whose regions cannot be read must not silently analyse with no
            # masks: that publishes exactly the events the masks existed to suppress.
            raise RuntimeError(f"Could not load regions for source {source_uuid}: {response.message}")
        return list(response.regions or [])

    async def _build_actions(self, db_api_client: db_client.ApiClient) -> None:
        """Instantiate the profile's actions, wiring in shared resources.

        The MQTT connection and the clip backend are owned here rather than inside each
        action: a watcher needs one broker connection, not one per action row, and the
        clip backend depends on the relay client the task already holds.
        """
        mqtt_settings = await self._load_mqtt_settings(db_api_client)
        needs_mqtt = any(a.kind in MQTT_ACTION_KINDS and a.enabled for a in self._profile.actions)
        if needs_mqtt and mqtt_settings is not None:
            self._mqtt = AsyncMqttPublisher(mqtt_settings)
            await self._mqtt.start()
        elif needs_mqtt:
            self.log("An MQTT action is configured but no broker is set in Settings — publishes will be skipped")

        for row in sorted(self._profile.actions, key=lambda a: a.order):
            if not row.enabled:
                continue
            params = dict(row.params_json or {})
            try:
                if row.kind in MQTT_ACTION_KINDS:
                    action = create_action(row.kind, publisher=self._mqtt, **params)
                elif row.kind == "record_clip":
                    # A resolver, not a backend: which one is right depends on the source
                    # the event fired on, and a profile may mix a relayed camera with a
                    # snapshot poller. Resolved at event time because a source's buffered
                    # flag isn't known until its provider has started, which is after this.
                    action = create_action(row.kind, backend_for=self._clip_backend_for, **params)
                else:
                    action = create_action(row.kind, **params)
            except Exception as e:
                # An unknown or misconfigured action must not stop the watcher starting;
                # the other actions are still worth running.
                logger.exception("Could not build action %s", row.kind)
                self.log(f"Skipping action {row.kind}: {e}")
                continue
            self._actions.append((row, action))

        # Asked of the action rather than of the row, for the same reason `_actions_for`
        # is: `record_event_timelapse` sits on the `event_ended` trigger and still needs
        # every frame. Reading `row.trigger` here left it never sampling anything, and
        # the failure was silent — every event compiled from zero frames.
        self._frame_actions = [entry for entry in self._actions if FRAME_ANALYSED in entry[1].triggers(entry[0])]

        if self._actions:
            self.log(f"Loaded {len(self._actions)} action(s)")
        if self._frame_actions:
            self.log(f"{len(self._frame_actions)} action(s) will run on every analysed frame")

    def _clip_backend_for(self, context: ActionContext):
        """Pick the clip backend for the source an event fired on.

        Buffered sources (MediaMTX pulls or ffmpeg-ingests them) are cut retroactively
        out of the relay's recorded segments — full resolution, full frame rate, with
        real pre-roll. Everything else has no timeline to rewind, so it falls back to
        its frame ring, which is choppy and honest about it.
        """
        pipeline = next((p for p in self._pipelines if p.source_uuid == context.source_uuid), None)
        if pipeline is None:
            return None
        if pipeline.buffered:
            # No hasattr guard: a manager without this method is a wiring bug, and
            # swallowing it here is what previously made every clip on a relayed
            # camera fail with "no clip backend" instead of saying why.
            return PlaybackClipBackend(self._manager.get_relay_api_client())
        ring = self._frame_rings.get(pipeline.source_uuid)
        return RingClipBackend(ring) if ring is not None else None

    #: Where drained chunks live while an event is open. Under the clips root so the
    #: retention sweep's disk accounting sees them, dot-prefixed so the media browser's
    #: day folders do not.
    CHUNK_DIRNAME = ".drain"

    def _chunk_root(self):
        from apps import RECORDINGS_CLIP_DIR

        return RECORDINGS_CLIP_DIR / self.CHUNK_DIRNAME

    def _drain_for(self, pipeline: WatchPipeline) -> Optional[ClipDrain]:
        """The clip drain for a buffered source, created on first need.

        Only buffered sources get one. A ring-backed source has no timeline to drain —
        its whole history is the RAM ring, which is bounded by its own settings.
        """
        if not pipeline.buffered or not pipeline.retention_s:
            return None
        drain = self._clip_drains.get(pipeline.source_uuid)
        if drain is None:
            relay_path = getattr(pipeline.provider, "_relayed_name", None)
            if not relay_path:
                return None
            relay = self._manager.get_relay_api_client()

            async def extract(start, duration_s, dest):
                return await relay.extract_clip(path=relay_path, start=start, duration_s=duration_s, dest=dest)

            drain = ClipDrain(
                source_uuid=pipeline.source_uuid,
                retention_s=pipeline.retention_s,
                extract=extract,
                chunk_root=self._chunk_root(),
                lead_s=self._max_preroll_for(pipeline.source_uuid),
                log=self.log,
            )
            self._clip_drains[pipeline.source_uuid] = drain
        return drain

    def _max_preroll_for(self, source_uuid: Optional[str]) -> float:
        """The largest pre-roll any clip action on this source asks for.

        The drain has to start that far ahead of the event, because pre-roll sits in
        front of the event and is the one part of a clip that cannot be re-derived once
        it has aged out. Taken across actions rather than per action so one drain serves
        all of them.
        """
        prerolls = [
            float(getattr(action, "preroll_s", 0.0))
            for row, action in self._actions
            if row.kind == "record_clip" and in_scope(getattr(row, "source_uuids_json", None), source_uuid)
        ]
        return max(prerolls) if prerolls else 0.0

    async def _service_drain(self, pipeline: WatchPipeline) -> None:
        """Keep a long event's video alive, one chunk at a time.

        Called every pass for every source. The common case — no event open — costs one
        dict lookup and returns.
        """
        open_events = pipeline.engine.open_events()
        drain = self._clip_drains.get(pipeline.source_uuid)

        if not open_events:
            if drain is not None:
                drain.note_idle()
            return

        drain = drain or self._drain_for(pipeline)
        if drain is None:
            return

        starts = [e["startedAt"] for e in open_events if e.get("startedAt") is not None]
        if not starts:
            return
        # Event timing is monotonic; the buffer is addressed by wallclock.
        earliest = datetime.fromtimestamp(min(starts) + self._wallclock_offset, tz=timezone.utc)
        drain.note_open(earliest)

        now_wall = datetime.fromtimestamp(time.monotonic() + self._wallclock_offset, tz=timezone.utc)
        if drain.due(now_wall):
            await drain.drain(now_wall)

    def _wants_overlay(self, source_uuid: Optional[str]) -> bool:
        """Whether any in-scope clip action on this source asked for a detection overlay."""
        return any(
            getattr(action, "save_overlay", False)
            for row, action in self._actions
            if row.kind == CLIP_ACTION_KIND and in_scope(getattr(row, "source_uuids_json", None), source_uuid)
        )

    def _detection_track_for(self, pipeline: WatchPipeline) -> Optional[DetectionTrack]:
        """The detection ring for a source whose clips want an overlay, or None.

        Sized to comfortably outreach the source's own rewind retention: pre-roll is cut
        from the buffer, so a ring that only reached back as far as the *default* would
        leave the lead-in of a long-retention camera's clips unannotated — and the
        lead-in is where you look to see what the detector made of an approach.

        The *negative* answer is cached as well as the positive one. This is called for
        every analysed frame of every source, and `_wants_overlay` walks the whole action
        list — a scan a watcher with no clip actions would otherwise pay for, forever, to
        be told "no" every time. Presence in the dict is the cache; the value may be None.
        """
        source_uuid = pipeline.source_uuid
        if source_uuid in self._detection_tracks:
            return self._detection_tracks[source_uuid]
        track = None
        if self._wants_overlay(source_uuid):
            window = max(DEFAULT_TRACK_SECONDS, (pipeline.retention_s or 0.0) + self._max_preroll_for(source_uuid))
            track = DetectionTrack(max_seconds=window)
        self._detection_tracks[source_uuid] = track
        return track

    def _ring_for(self, pipeline: WatchPipeline) -> Optional[FrameRing]:
        """The frame ring for an unbuffered source, created on first use.

        Lazily, and only for sources that need one: a profile of RTSP cameras must not
        allocate tens of megabytes of ring it will never read from.
        """
        if pipeline.buffered or pipeline.buffer_mode == "off":
            return None
        ring = self._frame_rings.get(pipeline.source_uuid)
        if ring is None:
            ring = FrameRing(max_seconds=pipeline.ring_seconds, max_mb=pipeline.ring_max_mb)
            self._frame_rings[pipeline.source_uuid] = ring
            self.log(
                f"Source {pipeline.source_name} has no rewind buffer — keeping a "
                f"{pipeline.ring_seconds:g}s / {pipeline.ring_max_mb:g}MB frame ring for clip pre-roll"
            )
        return ring

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
            # The settings table stores strings, so the Settings UI writes this key as
            # JSON. Accepting a dict too keeps a caller that hands us an already-decoded
            # value working, rather than silently falling back to localhost defaults.
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, dict) or not value.get("host"):
                return None
            return MqttSettings.from_dict(value)
        except Exception:
            logger.warning("Could not read MQTT broker settings; MQTT actions will not publish", exc_info=True)
            return None

    async def _pin_models(self) -> None:
        """Hold a pin on every model for the task's lifetime.

        Without this the ~10-minute idle auto-unload would evict models between frames
        on a slow-fps watcher, so every inference call would pay a model load.
        """
        models_api = ModelsApi(self._manager.get_inference_api_client())
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

    async def _renew_pins_periodically(self) -> None:
        """Re-pin on a timer so a run longer than the pin duration never lapses."""
        try:
            while not self._cancel_flag.is_set():
                await asyncio.sleep(self._PIN_RENEWAL_INTERVAL_S)
                # Re-read the client each cycle: a watcher outlives a Settings →
                # Servers change, and this is also what re-pins on the new host.
                models_api = ModelsApi(self._manager.get_inference_api_client())
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
        self,
        inference_api: InferenceApi,
        base64_img: str,
        model_names: List[str],
        frame_size: Tuple[int, int],
    ) -> List[Detection]:
        """Run a source's model stack and merge the results.

        *frame_size* is the (width, height) IN PIXELS of the image that was sent to
        inference — i.e. the analysis frame, after any downscale. It is required,
        not inferred: the inference client returns boxes as RELATIVE x/y/width/
        height, while `tracker.BBox` and `watch.geometry` are both documented in
        pixels, and zones are scaled to exactly this frame size by
        `WatchPipeline.prepare_frame`. Getting it from anywhere else (the camera's
        native size, say) would offset every rule.
        """
        if not model_names:
            return []

        async def _run_stack() -> List[list]:
            # ONE request for the whole stack. Built once, outside the retry loop
            # and outside the try. A malformed payload is a programming error, not
            # a transient inference failure, and building it inside the
            # `except Exception` below is exactly how this call spent its first
            # week returning zero detections on every frame.
            payload = BatchInferencePayload(
                image_base64=base64_img,
                model_names=list(model_names),
                pin_ids={n: p for n in model_names if (p := self._pin_ids.get(n))},
                conf_thresh=self._INFERENCE_CONF_FLOOR,
            )
            for attempt in range(self._MAX_RETRIES):
                try:
                    resp = await asyncio.wait_for(
                        asyncio.to_thread(inference_api.detect_batch, payload),
                        timeout=self._INFERENCE_TIMEOUT_S,
                    )
                    results = resp.results or {}
                    for name, message in (resp.errors or {}).items():
                        logger.error("Inference error on model %s: %s", name, message)
                    # Rebuilt from model_names, NOT by iterating the response: the
                    # merge below is FIRST_WINS, so the order of this list decides
                    # which model's label survives an overlap. Taking it from the
                    # response would make that depend on mapping order across two
                    # serializers rather than on the stack's configured priority.
                    # A model that failed contributes an empty list, keeping the
                    # positions of the ones that succeeded intact.
                    return [list(getattr(results.get(n), "detections", None) or []) for n in model_names]
                except asyncio.TimeoutError:
                    if attempt == self._MAX_RETRIES - 1:
                        self.log("Inference timed out — skipping frame")
                except Exception:
                    logger.exception("Inference error on model stack %s", list(model_names))
                    break
            return [[] for _ in model_names]

        started = time.perf_counter()
        results_per_model = await _run_stack()
        self._record_inference_time((time.perf_counter() - started) * 1000.0)

        merged = merge_detections_by_iou_and_hierarchy(
            results_per_model,
            label_ancestry=self._label_ancestry,
            overlap_iou_threshold=0.5,
            policy=MergePolicy.FIRST_WINS,
        )
        # ⚠️ The client model has x/y/width/height only, and they are RELATIVE.
        # There is no x1/y1/x2/y2 on it — reading those raised AttributeError out
        # of the first frame of every watcher (the caller's try/except covers only
        # the frame fetch, so nothing caught it). Convert to absolute corners here,
        # once, at the boundary where the model's coordinate space meets the
        # tracker's pixel space.
        frame_w, frame_h = frame_size
        return [
            Detection(
                bbox=(
                    det.bbox.x * frame_w,
                    det.bbox.y * frame_h,
                    (det.bbox.x + det.bbox.width) * frame_w,
                    (det.bbox.y + det.bbox.height) * frame_h,
                ),
                label_uuid=det.bbox.label_uuid,
                label_name=self._label_display_name(det.bbox),
                confidence=getattr(det, "confidence", 0.0) or 0.0,
            )
            for det in merged
        ]

    def _label_display_name(self, bbox: Any) -> Optional[str]:
        """A human name for one inference bbox.

        The inference client's bbox carries `label_uuid` and `class_str` and no name, so
        the DB's label name is the only real answer. `class_str` is the fallback rather
        than the uuid on purpose: a model class with no label mapping is a *configuration*
        problem, and showing "person" (the model's own word) makes that visible, whereas
        a uuid tells the user nothing and looks like a bug in the view.
        """
        meta = self._label_meta.get(getattr(bbox, "label_uuid", None) or "")
        if meta and meta.get("text"):
            return meta["text"]
        return getattr(bbox, "label_name", None) or getattr(bbox, "class_str", None)

    def _label_colour(self, label_uuid: Optional[str]) -> Optional[str]:
        return (self._label_meta.get(label_uuid or "") or {}).get("color")

    def _record_inference_time(self, ms: float) -> None:
        # Bounded: this is a "what is it doing now" sample, not a full history.
        self._inference_ms.append(ms)
        if len(self._inference_ms) > 500:
            del self._inference_ms[:-500]

    # ------------------------------------------------------------------ run

    async def _run(self) -> dict[str, Any]:
        self._run_started_at = time.monotonic()
        db_api_client = self._manager.get_db_api_client()

        # A factory, not the client itself, for the same reason `inference_api` is built
        # per pass: a watcher runs for weeks, and a client captured at start-up keeps
        # addressing whatever host Settings pointed at when the run began.
        self._recorder = WatchEventRecorder(
            profile_uuid=self._profile.uuid,
            api_client_factory=self._manager.get_db_api_client,
            log=self.log,
        )

        # Chunks from a previous process are worthless — their events were force-closed
        # and their clips already cut or lost — but they are full-rate video, so leaving
        # them is a disk leak nothing else would ever clean up.
        swept = await asyncio.to_thread(sweep_chunk_root, self._chunk_root())
        if swept:
            self.log(f"Swept {swept} clip chunk(s) left behind by a previous run")

        self._status_msg = "Building pipelines"
        await self._build_pipelines(db_api_client)

        self._status_msg = "Loading actions"
        await self._build_actions(db_api_client)

        self._status_msg = "Pinning models"
        await self._pin_models()
        renewal = asyncio.create_task(self._renew_pins_periodically())

        self._status_msg = "Starting sources"
        for pipeline in self._pipelines:
            try:
                await pipeline.provider.start()
            except Exception as e:
                logger.exception("Failed to start source %s", pipeline.source_uuid)
                pipeline.health.record_failure(time.monotonic(), str(e))
            # The relay tells us whether this source actually has a rewind buffer and
            # how deep it is; never infer either locally, or the UI badge, the clip
            # backend and the pre-roll clamp can disagree.
            pipeline.adopt_buffer_capabilities(pipeline.provider)

        self._status_msg = "Watching"
        try:
            while not self._cancel_flag.is_set():
                await self._check_pause()
                # Mid-run persistence takes priority over cancel, per the Task contract.
                self._maybe_checkpoint()

                now = time.monotonic()
                due = [p for p in self._pipelines if p.due(now)]
                if not due:
                    await asyncio.sleep(self._idle_sleep(now))
                    continue

                # Built per pass, not once: a watcher runs for weeks, and a client
                # cached at start-up would keep addressing the host that Settings →
                # Servers pointed away from (and whose pool has since been cleared).
                inference_api = InferenceApi(self._manager.get_inference_api_client())
                for pipeline in due:
                    if self._cancel_flag.is_set():
                        break
                    await self._service_source(pipeline, inference_api)

                self._status_msg = self._watching_status_msg()
                await self.heartbeat()
        finally:
            renewal.cancel()
            self._status_msg = "Closing open events"
            # Order matters: the events below are force-closed and still cut their
            # clips, so the chunks must outlive them. Discarding happens after.
            # An event open at shutdown must be closed, not stranded — its clip is
            # still extractable while the buffer window holds.
            for pipeline in self._pipelines:
                for signal in pipeline.close_open_events(time.monotonic()):
                    self._resolve_slot_names(signal, pipeline)
                    self._note_event(signal)
                    await self._dispatch(signal, pipeline)
            # Now that every clip has been cut, the chunks are dead weight — full-rate
            # video that nothing will ever read again.
            for drain in self._clip_drains.values():
                drain.discard()
            self._clip_drains.clear()
            # Same reasoning, and same ordering: the closing edges above wrote their
            # sidecars out of these rings, so they are only dead weight afterwards.
            self._detection_tracks.clear()

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

    def _watching_status_msg(self) -> str:
        """The one line the run's card gets. A dead camera has to reach it.

        A watcher over four cameras with one unplugged is still "Watching" by any
        internal measure, and that is exactly the reading that let a source stay down
        unnoticed for the rest of a run.
        """
        failed = len(self._failed_sources)
        if not failed:
            return "Watching"
        if failed == len(self._pipelines):
            return f"No sources available — retrying {failed} failed source(s)"
        return f"Watching ({failed} of {len(self._pipelines)} source(s) failed)"

    async def _reconnect_source(self, pipeline: WatchPipeline, now: float) -> bool:
        """Try to bring one failed source back. True if it is ready to be read.

        Deliberately two passes for a source that is genuinely down: this rebuilds the
        connection and returns, and readiness is asked on a later attempt. Waiting here
        for a camera that is not coming back would block for the metadata timeout, and
        the loop that calls this services every other camera too — one dead source
        would cost the healthy ones their frame rate.
        """
        try:
            await asyncio.wait_for(pipeline.provider.reconnect(), timeout=self._RECONNECT_TIMEOUT_S)
        except (asyncio.TimeoutError, TimeoutError):
            self._record_source_failure(pipeline, now, "reconnect timed out")
            return False
        except Exception as e:
            # Expected while a camera is off: the relay path has no source to read.
            logger.debug("Reconnect failed for source %s: %s", pipeline.source_uuid, e)
            self._record_source_failure(pipeline, now, f"reconnect failed: {e}")
            return False

        if not pipeline.provider.is_ready():
            self._record_source_failure(pipeline, now, "source has not come back yet")
            return False
        return True

    def _record_source_failure(self, pipeline: WatchPipeline, now: float, error: str) -> None:
        """Record a failed attempt, and say so the first time it becomes a fault.

        Logged on the transition only. A source retried every five minutes for a day
        would otherwise write 288 identical lines into a ring buffer that holds far
        fewer, pushing out the history that explains what happened.
        """
        pipeline.health.record_failure(now, error)
        # "Have we said this already?" is the membership, not a recomputation of
        # is_failed() from before the call: downtime has already crossed the threshold
        # by the time the crossing attempt is recorded, so comparing before-and-after
        # never sees an edge and the line was never logged at all.
        if pipeline.health.is_failed(now) and pipeline.source_uuid not in self._failed_sources:
            self._failed_sources.add(pipeline.source_uuid)
            self.log(
                f"Source '{pipeline.source_name}' has had no frames for "
                f"{_format_duration(pipeline.health.downtime_s(now))} — marking it failed. "
                f"Still retrying every {int(pipeline.health.BACKOFF_MAX_S)}s. Last error: {error}"
            )

    async def _service_source(self, pipeline: WatchPipeline, inference_api: InferenceApi) -> None:
        """Fetch, analyse and step one source. Never raises — failures back off."""
        now = time.monotonic()
        pipeline.schedule_next(now)

        provider = pipeline.provider
        # Two questions, and both have to be asked. Readiness catches a source that
        # never came up; health catches one that came up and then went silent — and
        # readiness cannot see that second case at all. `is_ready()` means "ffmpeg
        # reported geometry", which it does once, at the start; a camera that stopped
        # sending an hour ago still answers yes. Gating the rebuild on readiness alone
        # therefore skipped it for precisely the failure it exists to fix, and by then
        # the provider's own restart budget is spent and only reconnect() clears it, so
        # the source stayed dark for the rest of the run while the status surfaces
        # cheerfully reported it as "failed, still retrying".
        if not provider.is_ready() or not pipeline.health.healthy:
            # Not ready splits two ways, and they want opposite treatment.
            if provider.is_connecting():
                # Coming up. Leave it entirely alone — an RTSP camera behind a relay
                # can take minutes from cold, and rebuilding a reader mid-handshake
                # restarts that clock rather than hurrying it. Poll, do not touch.
                pipeline.health.record_connecting(now)
                return

            # Out of its connect window and still silent: genuinely broken, so rebuild
            # it. Without this the retry was theatre — the provider gives up on its own
            # stream after a few frameless reads and returns None from then on, so the
            # backoff was re-asking a reader that had permanently stopped trying.
            if not await self._reconnect_source(pipeline, now):
                return

        try:
            provided = await asyncio.wait_for(pipeline.provider.getNextImage(), timeout=self._FETCH_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._record_source_failure(pipeline, now, "frame fetch timed out")
            return
        except Exception as e:
            logger.exception("Error fetching from source %s", pipeline.source_uuid)
            self._record_source_failure(pipeline, now, str(e))
            return

        if provided is None:
            self._record_source_failure(pipeline, now, "source returned no frame")
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

        # A fresh reading, not the `now` from before the fetch: a slow camera can make
        # that stale by seconds, and these timestamps are what the clip window indexes.
        self._push_to_ring(pipeline, base64_img, time.monotonic(), full_res=provided.encoded_b64)

        # analysed.shape is (h, w, c); the detections must land in the same pixel
        # space the zones were scaled to by prepare_frame() just above.
        analysed_h, analysed_w = analysed.shape[:2]
        detections = await self._run_inference(
            inference_api, base64_img, pipeline.model_names, frame_size=(analysed_w, analysed_h)
        )

        now = time.monotonic()
        was_failing = not pipeline.health.healthy
        recovered_after = pipeline.health.downtime_s(now)
        pipeline.health.record_success(now)
        if was_failing:
            # Announced, because the matching "went down" line is in the log above it
            # and a run that only ever reports losses reads as though nothing ever
            # comes back.
            self.log(
                f"Source '{pipeline.source_name}' is back after " f"{_format_duration(recovered_after)} — reconnected"
            )
            self._failed_sources.discard(pipeline.source_uuid)
        self._frames_analysed += 1

        signals = pipeline.step(detections, now)

        # Between `step` and dispatch, and both halves of that matter. After `step`,
        # because `pipeline.detections`/`.tracks` hold the *previous* frame until it has
        # run — pushing earlier would file this frame's timestamp against last frame's
        # boxes and offset the whole overlay by one analysis interval. Before dispatch,
        # because dispatch is where a closing event cuts its clip and writes the sidecar
        # out of this ring, and the frame that closed the event is one of the frames you
        # most want to see boxes on.
        self._push_detection_track(pipeline, now, (analysed_w, analysed_h))

        for signal in signals:
            self._resolve_slot_names(signal, pipeline)
            self._note_event(signal)
            await self._dispatch(signal, pipeline)

        # After dispatch, so an event that just closed has already had its clip cut from
        # the chunks before they are discarded — and an event that just opened arms the
        # drain on this same pass rather than one frame late.
        await self._service_drain(pipeline)

        # After `step`, not before: the pipeline's `detections` are the mask-filtered
        # ones the tracker actually saw, so a live feed draws the same boxes the rules
        # were evaluated against rather than raw model output the watcher then ignored.
        await self._dispatch_frame(pipeline)

    def _push_detection_track(self, pipeline: WatchPipeline, now: float, frame_size: Tuple[int, int]) -> None:
        """Record one frame's boxes for the clip overlay, if this source keeps one.

        Timestamped in wallclock because that is how a clip's window is addressed, while
        the loop's clock is monotonic — the same reason `_push_to_ring` carries the
        offset.
        """
        track = self._detection_track_for(pipeline)
        if track is None:
            return
        track.push(now + self._wallclock_offset, pipeline.detections, pipeline.tracks, frame_size)

    def _push_to_ring(
        self, pipeline: WatchPipeline, base64_img: str, now: float, full_res: Optional[str] = None
    ) -> None:
        """Keep an unbuffered source's recent frames, so its events can still have pre-roll.

        Prefers the frame as the *camera* sent it (`full_res`) over the downscaled
        analysis frame. For a snapshot camera that is the difference between a 640 px
        clip and a full-resolution one, and it costs nothing: those bytes were already
        downloaded, and using them avoids an encode rather than adding one. Falls back to
        the analysis frame's JPEG, which was encoded for inference anyway.

        Either way the ring costs one base64 decode per analysed frame (tens of
        microseconds at these sizes) and no second encode.

        Timestamps are wallclock because that is how clip windows are addressed; the
        loop's own clock is monotonic, hence the offset.
        """
        ring = self._ring_for(pipeline)
        if ring is None:
            return
        try:
            # base64_encode_jpg wraps the image in a JSON packet ({"format", "data", ...})
            # rather than returning bare base64 — see imageUtils. Unwrap it here.
            payload = json.loads(full_res or base64_img)
            jpeg = base64.b64decode(payload["data"])
        except Exception:
            logger.debug("Could not unpack frame for the ring on %s", pipeline.source_uuid, exc_info=True)
            return
        ring.push(now + self._wallclock_offset, jpeg)

    def _resolve_slot_names(self, signal, pipeline: Optional[WatchPipeline]) -> None:
        """Turn the engine's slot uuids into display names, in place.

        The engine reports *which* label and *which* zone satisfied a multi-ref rule, but
        it is pure and has no vocabulary to resolve them with. Names are attached here,
        once per transition, so every downstream consumer — the MQTT topic template, the
        event log, the status panel — sees the same words the user typed.

        Falls back to the uuid rather than to a placeholder: a template that prints a raw
        uuid is obviously wrong, whereas one printing "unknown" looks like a real value.
        """
        if signal.subject_ref:
            meta = self._label_meta.get(signal.subject_ref) or {}
            signal.subject_name = meta.get("text") or signal.subject_ref

        if not signal.target_ref:
            return
        # The target is a zone for a zone-kind rule and a label otherwise; try both
        # rather than re-deriving the rule's arity here.
        zone = (pipeline.engine.zones.get(signal.target_ref) if pipeline else None) if signal.target_ref else None
        if zone is not None:
            signal.target_name = zone.name
        else:
            meta = self._label_meta.get(signal.target_ref) or {}
            signal.target_name = meta.get("text") or signal.target_ref

    def _note_event(self, signal) -> None:
        if signal.phase is EventPhase.START:
            self._events_started += 1
        elif signal.phase is EventPhase.END:
            self._events_ended += 1
        self._recent_events.append(signal.to_dict())
        if len(self._recent_events) > 50:
            del self._recent_events[:-50]

    def _actions_for(self, signal, trigger: str) -> List[Any]:
        """Actions that should fire for this transition, after rule and source scoping.

        Both scope lists are empty-means-all, so an action nobody has narrowed still
        fires everywhere. See `apps.helpers.watch.scoping` for why the scope hangs off
        the action rather than off the rule.
        """
        return [
            (row, action)
            for row, action in self._actions
            if action_applies(
                row,
                trigger,
                rule_uuid=signal.rule_uuid or None,
                source_uuid=signal.source_uuid or None,
                # Asked of the action, not read off the row: an action that spans an
                # event's life fires on triggers the user never picked. See
                # `Action.triggers`.
                fires_on=action.triggers(row),
            )
        ]

    async def _dispatch(self, signal, pipeline: Optional[WatchPipeline] = None) -> None:
        """Hand an event transition to the action layer, and record it.

        Persistence brackets the actions rather than following them, because the two
        edges need opposite orderings. On START the row must exist before anything runs
        so an artifact produced by an action has an event to hang off. On END the clip
        does not exist until `record_clip` has run, so the row is closed *after* — with
        the event uuid captured beforehand, since closing it forgets the mapping.
        """
        verb = {EventPhase.START: "started", EventPhase.UPDATE: "updated", EventPhase.END: "ended"}[signal.phase]
        detail = f" ({signal.duration_s:.1f}s)" if signal.phase is EventPhase.END else ""
        capped = " [capped by max duration]" if signal.truncated else ""
        self.log(f"Event {verb}: {signal.rule_name}{detail}{capped} tracks={signal.track_ids}")
        if self._dry_run:
            return

        trigger = {
            EventPhase.START: "event_started",
            EventPhase.UPDATE: "event_started",
            EventPhase.END: "event_ended",
        }[signal.phase]
        context = self._action_context(signal, pipeline)

        if signal.phase is EventPhase.START:
            await self._recorder.record(signal, context)
        event_uuid = self._recorder.event_uuid_for(signal)

        media = await self._run_actions(self._actions_for(signal, trigger), signal, context)

        if signal.phase is EventPhase.END:
            for item in media:
                await self._recorder.attach_recording(
                    signal,
                    event_uuid,
                    item,
                    context=context,
                    # Only a clip has one. A timelapse's frames are seconds apart, so
                    # there is no continuous detection track to lay over it — anything
                    # it wants drawn is burnt in at sample time.
                    detection_track=(
                        self._detection_tracks.get(signal.source_uuid or "")
                        if item.get("kind") == MEDIA_ACTION_KINDS[CLIP_ACTION_KIND]
                        else None
                    ),
                )
            await self._recorder.record(signal, context)

    async def _dispatch_frame(self, pipeline: WatchPipeline) -> None:
        """Fire `frame_analysed` actions for one just-analysed frame.

        Separate from `_dispatch` because a frame is not an event: nothing opened, so
        there is no rule, no phase and nothing to log or persist. It exists so a live
        annotated feed can publish continuously — an image action on an event trigger
        would show a frozen frame for the whole of the day that nothing happens.

        Cheap when unused: `_frame_actions` is empty for every watcher that has no
        such action, so the common case is one truth test per frame.
        """
        if self._dry_run or not self._frame_actions:
            return
        now = time.monotonic()
        signal = WatchEventSignal(
            # A frame belongs to no rule. Empty rather than a placeholder name so a
            # topic template's `{rule}` level collapses instead of publishing a level
            # that looks deliberate.
            rule_uuid="",
            rule_name="",
            phase=EventPhase.UPDATE,
            # Monotonic, like every signal the engine emits. `ActionContext` carries the
            # wallclock offset for anything that needs real time, and a signal mixing
            # the two clocks would make one action's timestamps silently disagree with
            # another's.
            at=now,
            started_at=now,
            source_uuid=pipeline.source_uuid,
        )
        context = self._action_context(signal, pipeline)
        await self._run_actions(self._actions_for(signal, FRAME_ANALYSED), signal, context)

    async def _run_actions(self, rows: List[Any], signal, context: ActionContext) -> List[Dict[str, Any]]:
        """Run a resolved action list, counting outcomes. Never raises.

        Returns the result data of every media-producing action, each tagged with the
        `recordings.kind` it becomes, so the caller can register the files they wrote.
        A list rather than the single clip this used to return: an event can own both a
        clip and a timelapse, and they are peers.

        Two actions of the *same* kind scoped to one rule and source is still a
        misconfiguration, and still resolves as last-wins — but now only against its own
        kind, so a timelapse can no longer displace a clip.
        """
        media: Dict[str, Dict[str, Any]] = {}
        for row, action in rows:
            try:
                result = await action.run(signal, context)
            except Exception as e:
                # An action must never be able to kill the frame loop. Defensive even
                # though Action.run promises not to raise — a third-party action might.
                logger.exception("Action %s raised", row.kind)
                self._action_failures += 1
                self._log_action_failure(row, f"Action {row.kind} errored: {e}")
                continue
            if result.ok:
                self._action_successes += 1
                self._last_action_failure.pop(row.uuid, None)
                # Gated on kind, not merely on a `path` key: `save_snapshot` also
                # reports one, and treating a jpg as the event's clip put a still image
                # behind the status panel's "last clip" and would now write a
                # `recordings` row pointing at something no <video> can play.
                kind = MEDIA_ACTION_KINDS.get(row.kind)
                if kind and result.data.get("path"):
                    if row.kind == CLIP_ACTION_KIND:
                        # The status panel's "last clip" means the last *clip*. A
                        # timelapse is not one, and would make the panel claim a
                        # 120x-speed summary as the thing the watcher just recorded.
                        self._last_clip_path = result.data["path"]
                    media[kind] = dict(result.data, kind=kind)
            else:
                self._action_failures += 1
                self._log_action_failure(row, f"Action {row.kind} failed: {result.detail}")
        return list(media.values())

    def _log_action_failure(self, row: Any, message: str) -> None:
        """Log an action failure once per distinct message, not once per occurrence.

        A `frame_analysed` action fires several times a second on every source. One
        broken broker would otherwise write hundreds of identical lines into a ring
        buffer that holds far fewer, pushing out the history that explains what went
        wrong — the same reason `_record_source_failure` logs only on the transition.
        The counter still increments every time, so the rate stays visible.
        """
        if self._last_action_failure.get(row.uuid) == message:
            return
        self._last_action_failure[row.uuid] = message
        self.log(message)

    def _action_context(self, signal, pipeline: Optional[WatchPipeline]) -> ActionContext:
        return ActionContext(
            profile_uuid=self._profile.uuid,
            profile_name=self._profile.name,
            source_uuid=signal.source_uuid,
            source_name=pipeline.source_name if pipeline else None,
            source_mqtt_name=pipeline.mqtt_name if pipeline else None,
            relay_path=getattr(pipeline.provider, "_relayed_name", None) if pipeline else None,
            buffered=bool(pipeline and pipeline.buffered),
            retention_s=pipeline.retention_s if pipeline else 0.0,
            frame=pipeline.last_frame if pipeline else None,
            full_frame=pipeline.last_full_frame if pipeline else None,
            detections=list(pipeline.detections) if pipeline else [],
            label_colours=self._label_colours(),
            # Rule timing is monotonic; the buffer is addressed by wallclock. Carry the
            # offset once rather than making every action re-derive it.
            wallclock_offset=self._wallclock_offset,
            # Video already rescued from the buffer for this source, for an event that
            # outlived it. Empty for every event that closed inside the window, which is
            # what keeps the ordinary case on the ordinary single-cut path.
            clip_chunks=self._chunks_for_signal(signal),
        )

    def _chunks_for_signal(self, signal) -> List[Any]:
        """Drained chunks covering this event's window, oldest first."""
        drain = self._clip_drains.get(signal.source_uuid) if signal.source_uuid else None
        if drain is None or not drain.chunks:
            return []
        start = datetime.fromtimestamp(signal.started_at + self._wallclock_offset, tz=timezone.utc)
        end = datetime.fromtimestamp(signal.at + self._wallclock_offset, tz=timezone.utc)
        return drain.chunks_for(start, end)

    def _label_colours(self) -> dict:
        """label uuid → colour, for actions that draw boxes.

        Built from the same `_label_meta` the status panel and slot-name resolution use,
        so a box drawn onto an MQTT frame is the colour the label has in the UI.
        """
        return {uuid: meta["color"] for uuid, meta in self._label_meta.items() if meta.get("color")}

    def _maybe_checkpoint(self) -> bool:
        """Answer a pending checkpoint request. Returns whether one was answered.

        Checkpointing is a two-part handshake with
        ``Manager._checkpoint_running_tasks``: fill ``_resume_data`` **and** raise
        ``_data_ready_flag``, because the manager writes nothing until
        ``is_data_ready()`` says there is something to write.

        This did only the first half. Every checkpoint request built resume data
        and dropped it on the floor, so the instance reached a restart with
        ``resume_data=None`` and startup recovery marked it "interrupted without
        checkpoint — marking error" rather than resuming it. Nothing logged,
        nothing raised; the only symptom was a watcher that did not come back.

        It also quietly defeated DEFAULT_PERSISTENT, since auto-resume is gated
        on resume_data — a task that never checkpoints cannot be persistent
        however it is declared.

        Extracted from the run loop so this is reachable by a test without
        standing up sources, models and a database.
        """
        if not self._data_req_flag.is_set():
            return False
        self.set_resume_data(self._build_watch_resume_data())
        self._data_ready_flag.set()
        self.clear_data_request()
        return True

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
        # After the run loop's `finally`, which has already emitted a closing signal for
        # everything still open. Anything left in the map failed to persist and has no
        # row to patch, so this forgets rather than closes.
        await self._recorder.close_all()

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
                # Counted apart from dropped: a superseded frame is a live feed working
                # as designed, and merging the two would make a healthy feed read as a
                # failing broker. Rising *dropped* on a live feed is the real warning.
                "mqttCoalesced": (self._mqtt.coalesced if self._mqtt else 0),
                # In KB because the interesting number for an image feed is a backlog in
                # megabytes, which a raw byte count buries in digits.
                "mqttQueuedKb": (round(self._mqtt.stats()["queuedBytes"] / 1024, 1) if self._mqtt else 0),
            },
            "sources": [p.status_row(now) for p in self._pipelines],
            "activeTracksTable": tracks_rows,
            "openEventsTable": open_rows,
            "recentEvents": list(reversed(self._recent_events[-15:])),
            "tuningHints": "\n".join(hints) if hints else "No tuning suggestions yet.",
        }

    # ------------------------------------------------------------------ live view

    #: Preview JPEG quality. Lower than the analysis path's: this is a monitoring
    #: view polled a few times a second, not a training image.
    _PREVIEW_QUALITY: int = 70

    async def get_live_preview(self, source_uuid: Optional[str] = None, **options: Any) -> Optional[dict[str, Any]]:
        """The most recent analysed frame for one source, with everything seen in it.

        Three layers of "don't do the work":

        1. **Nothing is precomputed.** A watcher nobody is looking at pays nothing for
           this method existing — the frame loop stores a reference it already held.
        2. **`since_seq`** — a caller that already has frame N gets metadata only, no
           image. Polling at 5 Hz against a 1 fps camera therefore costs one encode per
           second, not five, and transfers no image bytes on the other four.
        3. **The encode is cached per (frame, quality)** so two viewers, or a caller
           that doesn't pass `since_seq`, still only cost one encode per frame.

        Deliberately reports the frame *as analysed* — downscaled, and byte-identical to
        what was handed to inference — because the whole point of this view is to answer
        "is the model being shown what I think it is?". A prettier full-resolution frame
        would show coordinates that do not line up with the boxes drawn on it.
        """
        pipeline = self._pipeline_for(source_uuid)
        payload: dict[str, Any] = {
            "profileUuid": getattr(self._profile, "uuid", None),
            "profileName": getattr(self._profile, "name", None),
            "dryRun": self._dry_run,
            # The picker's whole option list, so the view needs exactly one call per poll.
            "sources": [{"sourceUuid": p.source_uuid, "sourceName": p.source_name} for p in self._pipelines],
            "sourceUuid": pipeline.source_uuid if pipeline else None,
            "sourceName": pipeline.source_name if pipeline else None,
            "frame": None,
        }
        if pipeline is None:
            return payload

        now = time.monotonic()
        # One reference, grabbed once: the loop replaces this attribute between frames,
        # and reading it twice could encode one frame and report another's geometry.
        frame = pipeline.last_frame
        quality = int(options.get("quality") or self._PREVIEW_QUALITY)

        open_events = pipeline.engine.open_events()
        # track_id → the rules currently holding an event open on it. The view draws
        # these boxes differently, which is the fastest way to answer the question the
        # live view is really asked: "of everything moving out there, what is the
        # watcher actually reacting to?"
        rules_by_track: Dict[int, List[str]] = {}
        for event in open_events:
            for track_id in event.get("trackIds") or ():
                rules_by_track.setdefault(int(track_id), []).append(str(event.get("ruleName") or "rule"))

        payload.update(
            {
                "analysisFps": pipeline.analysis_fps,
                "analysisWidth": pipeline.analysis_width,
                "modelNames": list(pipeline.model_names),
                "framesAnalysed": pipeline.health.frames_analysed,
                "healthy": pipeline.health.healthy,
                # `healthy` alone cannot distinguish a source that blinked from one
                # that has been gone for an hour, and the live view is where someone
                # looks precisely when a camera has stopped showing anything.
                "sourceState": pipeline.health.state(now),
                "downtimeS": round(pipeline.health.downtime_s(now), 1),
                "lastError": pipeline.health.last_error,
                "buffer": pipeline._buffer_label(),
                "frameAgeS": (round(now - pipeline.health.last_frame_at, 2) if pipeline.health.last_frame_at else None),
                # Pre-tracker, so an empty tracks list next to a populated detections
                # list localises the problem to the tracker rather than the model.
                "detections": [
                    {
                        "bbox": [round(v, 2) for v in det.bbox],
                        "labelUuid": det.label_uuid,
                        "labelName": det.label_name,
                        "labelColor": self._label_colour(det.label_uuid),
                        "confidence": round(det.confidence, 4),
                    }
                    for det in pipeline.detections
                ],
                "tracks": [
                    {
                        **t.to_dict(),
                        "ageS": round(t.age_s(now), 2),
                        "labelColor": self._label_colour(t.label_uuid),
                        "eventRules": rules_by_track.get(t.track_id, []),
                    }
                    for t in pipeline.tracks
                ],
                "zones": [
                    {"uuid": z.uuid, "name": z.name, "kind": z.kind, "points": [list(p) for p in z.points]}
                    for z in pipeline.engine.zones.values()
                ],
                "openEvents": open_events,
            }
        )

        if frame is None:
            return payload

        seq = pipeline.frame_seq
        height, width = frame.shape[:2]
        payload["frameSeq"] = seq
        payload["frameWidth"] = width
        payload["frameHeight"] = height

        since = options.get("since_seq")
        if since is not None and int(since) == seq:
            # The caller already has these pixels. Everything above still moves —
            # tracks age, events open — so the metadata is worth returning; the image
            # is not worth re-encoding or re-sending.
            payload["frameUnchanged"] = True
            return payload

        cached_seq, cached_quality, cached_b64 = self._preview_cache.get(pipeline.source_uuid, (None, None, None))
        if cached_seq == seq and cached_quality == quality:
            payload["frame"] = cached_b64
            return payload

        encoded = await asyncio.to_thread(self._encode_preview, frame, quality)
        # One entry per source, overwritten — a bounded ~50 KB per watched source, and
        # only for sources someone has actually opened the live view on.
        self._preview_cache[pipeline.source_uuid] = (seq, quality, encoded)
        payload["frame"] = encoded
        return payload

    def _pipeline_for(self, source_uuid: Optional[str]) -> Optional[WatchPipeline]:
        if not self._pipelines:
            return None
        if not source_uuid:
            return self._pipelines[0]
        return next((p for p in self._pipelines if p.source_uuid == source_uuid), None)

    def get_source_frame(self, source_uuid: str):
        """The most recent analysed frame for one source, or None if not watching it.

        Lets the region editor draw against a live still without opening a second
        connection to a camera this watcher is already reading — which some IP cameras
        refuse outright, having a hard cap on concurrent RTSP sessions.
        """
        for pipeline in self._pipelines:
            if pipeline.source_uuid == source_uuid:
                return pipeline.last_frame
        return None

    @staticmethod
    def _encode_preview(frame, quality: int) -> Optional[str]:
        """Bare base64 JPEG — not `base64_encode_jpg`, which wraps it in a JSON packet
        for the inference API. An <img> src wants the bytes, not the envelope.

        No colour conversion, on purpose: this shows the array exactly as
        `cv2.imencode` in the inference path interprets it, so a provider handing out
        RGB where the rest of the app assumes BGR shows up here as blue cats instead of
        being silently corrected in the one view meant to reveal it.
        """
        import base64

        import cv2

        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        try:
            # No copy: the frame is passed straight through. cv2.imencode treats its
            # input as const, so nothing here needs a writable or re-laid-out array.
            ok, buffer = cv2.imencode(".jpg", frame, params)
        except cv2.error:
            # Some providers hand out a read-only view over a buffer (ffmpeg's
            # np.frombuffer). Copying is the fallback, not the default, so the common
            # case stays zero-copy.
            import numpy as np

            ok, buffer = cv2.imencode(".jpg", np.ascontiguousarray(frame), params)
        if not ok:
            return None
        return base64.b64encode(buffer).decode("ascii")

    def _percentile(self, pct: int) -> Optional[float]:
        if not self._inference_ms:
            return None
        ordered = sorted(self._inference_ms)
        idx = min(len(ordered) - 1, int(len(ordered) * pct / 100))
        return round(ordered[idx], 1)

    def get_status_message(self) -> str:
        return self._status_msg
