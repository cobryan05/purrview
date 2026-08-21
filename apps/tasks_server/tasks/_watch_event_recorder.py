"""Turn the rule engine's event signals into durable `watch_events` rows.

Why this is a task concern and not an action's: `ActionContext` states outright that
the task keeps sole ownership of its API clients and DB writes, so actions are pure
side effects on the outside world. Persistence has to live above them.

It is not unconditional, and it is not written at the opening edge. Both used to be
true, on the reasoning that the review page is a record of what the watcher saw rather
than of which actions someone enabled — which is how the list filled with rows holding
nothing to look at. A rule that matches all day is normal and wanted (it still drives
MQTT); a row per match is not.

So an event is filed only once something has actually been written *to* it, and that is
knowable only after the actions have run. Two gates, one layer up in
`StreamWatcherTask`: `_event_is_logged` (could any in-scope action file this?) and
`Action.wrote_to_event` (did one?). The second is what stops an event timelapse that
compiled nothing — a legitimate outcome for a visit below its frame minimum, reported
as success — from leaving a card that points at no media.

Two artifacts per event:

* **The row.** Created when the first thing is written to the event, from material held
  since the opening edge, and patched at the close with the ending time, peak confidence
  and the runaway-cap flag. For a clip or a timelapse that means the row appears when
  the event ends; for `save_event` it appears at once and is visible while the event is
  still open. Rule and slot names are denormalised onto it — see the model docstring.
* **The card art.** A small annotated JPEG *encoded* at the opening edge, because that
  is the only moment the right frame exists, and written when the row is created — so
  an event that files nothing costs no disk. Distinct from what a `save_snapshot` action
  writes: that is a full-quality *training image* under `image_datasets` which the user
  curates and which must never have boxes burnt into it. This one is thumbnail-sized,
  has the boxes drawn on deliberately (the whole point of the card is to show what acted
  on what), lives outside the image datasets, and is deleted with its event.

Nothing else in the system closes an event, so this module owns that invariant at both
ends: `close_all` closes what a clean stop leaves open, and `close_stranded` closes what
a killed process left behind, on the next start.

Nothing here may raise into the frame loop. A watcher that stops recording events
because the DB hiccuped is a bug; a watcher that stops *watching* for the same reason
is a much worse one, so every failure is logged and swallowed exactly as an action
failure is.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import db_client
from db_client.api.recordings_api import RecordingsApi
from db_client.api.watch_profiles_api import WatchProfilesApi

from apps import RECORDINGS_SNAPSHOT_DIR
from apps.helpers.actions.action import ActionContext
from apps.helpers.consts import JsonValues
from apps.helpers.imageUtils import annotate_detections, downscale_to_width, encode_jpg_bytes
from apps.helpers.timeUtils import parse_iso_utc
from apps.helpers.watch.events import EventPhase, WatchEventSignal

logger = logging.getLogger(__name__)

#: Outside `image_datasets` on purpose — these are not training images and must not turn
#: up in the annotation file browser. Defined in `apps/__init__.py` because db_server's
#: retention sweep has to find these files to delete them and cannot import this module.
DEFAULT_SNAPSHOT_ROOT = RECORDINGS_SNAPSHOT_DIR

#: Card art, not evidence. The clip is the evidence, and at 480 px a snapshot is a few
#: tens of KB, so a busy week of events costs megabytes rather than gigabytes.
SNAPSHOT_WIDTH = 480
SNAPSHOT_QUALITY = 80

#: Mirrors `RecordingKind.CLIP`. Spelled out rather than imported for the same reason the
#: trigger names are in `scoping.py`: this module talks to db_server through the
#: generated client, where every field is already a plain string.
RECORDING_KIND_CLIP = "clip"

#: Result keys a media action reports that are worth keeping on the row for the player
#: to show. Everything else in the result — the path, the sizes, the requested window —
#: either has its own column or is about the extraction rather than the file.
MEDIA_METADATA_KEYS = ("frames", "sourceSpanS", "intervalS", "lowFps")

#: A signal carries no event id, so the row a closing edge belongs to has to be found by
#: what identifies the event. This is the *engine's* current identity — one open event
#: per rule per source. When per-pair identity lands (WATCH_OUTPUTS_PLAN §2) the refs
#: already in the key make it correct with no change here.
EventKey = Tuple[str, str, str, str]


def _payload_models() -> Tuple[Any, Any, Any]:
    """Import the generated request bodies at call time, not import time.

    `generate_openapi_modules.sh` builds the spec by importing this app, then generates
    the client from it. A module-level import of a model this module's own endpoints
    define is therefore a bootstrap cycle: the generator cannot run to create the model
    because importing the app needs the model that does not exist yet. Deferring it
    breaks the cycle, at the cost of one dict lookup per event.
    """
    from db_client.models.recording_create import RecordingCreate
    from db_client.models.watch_event_create import WatchEventCreate
    from db_client.models.watch_event_update import WatchEventUpdate

    return WatchEventCreate, WatchEventUpdate, RecordingCreate


def _blip_payload_models() -> Tuple[Any, Any]:
    """Deferred for the same bootstrap reason as `_payload_models` above."""
    from db_client.models.event_blip_payload import EventBlipPayload
    from db_client.models.replace_event_blips_request import ReplaceEventBlipsRequest

    return ReplaceEventBlipsRequest, EventBlipPayload


@dataclass
class _PendingEvent:
    """An event that has begun but written nothing yet.

    Everything a row needs that is only knowable at the opening edge: the wallclock the
    event began, and its card art already encoded (destination and bytes). Held in RAM
    until an action writes something, then flushed by `ensure_event` — or dropped at the
    closing edge, having cost one jpg encode and nothing on disk.
    """

    started_at: Optional[datetime] = None
    snapshot: Optional[Tuple[Path, bytes]] = None


def _event_key(signal: WatchEventSignal) -> EventKey:
    return (
        signal.source_uuid or "",
        signal.rule_uuid or "",
        signal.subject_ref or "",
        signal.target_ref or "",
    )


class WatchEventRecorder:
    """Writes one watcher's event history. One instance per running task."""

    def __init__(
        self,
        profile_uuid: str,
        api_client_factory: Callable[[], db_client.ApiClient],
        log: Callable[[str], None],
        snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
        analyse_blips: bool = True,
        blip_thresholds: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._profile_uuid = profile_uuid
        self._api_client_factory = api_client_factory
        self._log = log
        self._snapshot_root = Path(snapshot_root)
        #: Both come from the watch profile and are read once per run, not per clip: a
        #: watcher runs for weeks and a fetch per closing edge would be a request per
        #: event to learn something that almost never changes. Editing a profile
        #: restarts its watcher, which is what picks the change up.
        self._analyse_blips = bool(analyse_blips)
        self._blip_threshold_overrides = dict(blip_thresholds or {})
        #: Open events by key, so a closing edge can find the row it opened.
        self._open: Dict[EventKey, str] = {}
        #: Events that have begun but written nothing yet — see `_PendingEvent`.
        self._pending: Dict[EventKey, _PendingEvent] = {}
        self._failures = 0

    @property
    def failures(self) -> int:
        return self._failures

    # ── the two edges ─────────────────────────────────────────────────────────

    async def record(self, signal: WatchEventSignal, context: ActionContext) -> Optional[str]:
        """Note one transition. Returns the event uuid, when there is one.

        START writes nothing. It captures what a row *would* need and holds it, because
        whether the event deserves one is not known yet: a clip action can fail to cut,
        and an event timelapse legitimately compiles nothing when the visit was too
        short for its frame minimum. Both used to leave a card with no media behind it —
        which is the whole complaint this design answers. The row is created by
        `ensure_event`, once an action reports it actually wrote something.

        UPDATE is deliberately ignored: it re-fires the `event_started` trigger for
        actions that want a repeated nudge, but it opens nothing, so persisting it
        would turn one visit into a row per frame.
        """
        try:
            if signal.phase is EventPhase.START:
                await self._hold_event(signal, context)
                return None
            if signal.phase is EventPhase.END:
                return await self._close_event(signal)
        except Exception as e:
            # Same contract as an action failure: counted and logged, never raised.
            self._failures += 1
            logger.exception("Could not persist watch event for rule %s", signal.rule_name)
            self._log(f"Event not recorded ({signal.rule_name}): {e}")
        return None

    def event_uuid_for(self, signal: WatchEventSignal) -> Optional[str]:
        """The row a still-open signal belongs to, for callers linking artifacts to it."""
        return self._open.get(_event_key(signal))

    async def ensure_event(self, signal: WatchEventSignal, context: ActionContext) -> Optional[str]:
        """The event's row, creating it if this is the first thing written to it.

        Called when an action reports that it put something in the event — a clip, a
        compiled timelapse, or an explicit `save_event`. Everything the row needs was
        captured at the opening edge, so the card art is still the frame the rule fired
        on rather than whatever the camera happens to show now.

        Idempotent: an event that owns both a clip and a timelapse calls this twice and
        gets one row. Never raises, for the same reason `record` does not.
        """
        key = _event_key(signal)
        existing = self._open.get(key)
        if existing is not None:
            return existing
        try:
            pending = self._pending.pop(key, None)
            if pending is None:
                # No opening edge was held for this event — the watcher started while it
                # was already open, or the capture failed. Still worth a row: something
                # was written and a recording with no event to hang off is worse than a
                # card with no art.
                pending = _PendingEvent(started_at=self._wallclock(signal.started_at, context))
            uuid = await self._open_event(signal, pending)
            self._open[key] = uuid
            return uuid
        except Exception as e:
            self._failures += 1
            logger.exception("Could not persist watch event for rule %s", signal.rule_name)
            self._log(f"Event not recorded ({signal.rule_name}): {e}")
            return None

    async def _hold_event(self, signal: WatchEventSignal, context: ActionContext) -> None:
        """Capture what the row will need, without writing anything.

        The card art is *encoded* here and written later, rather than keeping the frame
        and encoding at flush time. An action's frame is a live buffer some providers
        reuse in place, so a reference held for the length of an event can decode as
        whatever the camera showed minutes afterwards — a card art bug that would only
        appear on some sources, long after the event it misrepresents.
        """
        self._pending[_event_key(signal)] = _PendingEvent(
            started_at=self._wallclock(signal.started_at, context),
            snapshot=await self._encode_snapshot(signal, context),
        )

    async def _open_event(self, signal: WatchEventSignal, pending: "_PendingEvent") -> str:
        # The snapshot is written first so the row is never briefly snapshot-less: a
        # card that pops into existence without art and gains it a moment later flickers
        # in a list that polls. A failure here costs the art, not the event.
        snapshot_path = await self._flush_snapshot(signal, pending)

        WatchEventCreate, _, _ = _payload_models()
        api = WatchProfilesApi(self._api_client_factory())
        payload = WatchEventCreate(
            profile_uuid=self._profile_uuid,
            source_uuid=signal.source_uuid or None,
            rule_uuid=signal.rule_uuid or None,
            rule_name=signal.rule_name or None,
            event_label=(signal.metadata or {}).get("eventLabel"),
            started_at=pending.started_at,
            peak_confidence=signal.peak_confidence,
            track_ids_json=list(signal.track_ids),
            subject_ref=signal.subject_ref,
            subject_name=signal.subject_name,
            target_ref=signal.target_ref,
            target_name=signal.target_name,
            snapshot_path=str(snapshot_path) if snapshot_path else None,
            metadata_json=dict(signal.metadata or {}),
        )
        response = await asyncio.to_thread(api.create_watch_event, payload)
        if response.status != JsonValues.SUCCESS or response.event is None:
            raise RuntimeError(response.message or "db refused the event")
        return response.event.uuid

    async def _close_event(self, signal: WatchEventSignal) -> Optional[str]:
        # Whatever was held for an event that wrote nothing is dropped here. This is the
        # ordinary path for a rule that matches all day with no recording action in
        # scope — nothing was written, so there is nothing to close and nothing to show.
        self._pending.pop(_event_key(signal), None)
        uuid = self._open.pop(_event_key(signal), None)
        if uuid is None:
            # Either the event wrote nothing, or its row failed to insert. Closing
            # nothing is correct; inventing a zero-length row at the closing edge would
            # be worse than the gap.
            return None
        _, WatchEventUpdate, _ = _payload_models()
        api = WatchProfilesApi(self._api_client_factory())
        payload = WatchEventUpdate(
            ended_at=datetime.now(timezone.utc),
            peak_confidence=signal.peak_confidence,
            truncated=signal.truncated,
        )
        response = await asyncio.to_thread(api.update_watch_event, uuid, payload)
        if response.status != JsonValues.SUCCESS:
            raise RuntimeError(response.message or "db refused the event update")
        return uuid

    async def close_all(self) -> None:
        """Close out anything still open, and drop anything still merely pending.

        Called from `_deinit` *after* the run loop has emitted a closing signal for
        everything open, so a row still here is one whose close did not persist. It used
        to be forgotten at this point, which is how a stopped watcher could leave an
        event showing "in progress" forever — nothing else in the system ever closes
        one, and the row outlives the process that could have.

        `now` is the honest end: the watcher held the event open until it stopped. Each
        row is closed independently and a failure is logged rather than raised — this
        runs during shutdown, where giving up on the second row because the first failed
        would strand exactly what this exists to prevent.
        """
        self._pending.clear()
        if not self._open:
            return
        _, WatchEventUpdate, _ = _payload_models()
        api = WatchProfilesApi(self._api_client_factory())
        payload = WatchEventUpdate(ended_at=datetime.now(timezone.utc))
        for key, uuid in list(self._open.items()):
            try:
                await asyncio.to_thread(api.update_watch_event, uuid, payload)
            except Exception as e:
                self._failures += 1
                logger.warning("Could not close stranded event %s: %s", uuid, e)
                self._log(f"Event {uuid} left open: {e}")
            self._open.pop(key, None)

    async def close_stranded(self) -> int:
        """Close events a previous run left open, and say how many there were.

        Called once as the watcher starts. Nothing else in the system ever closes one: a
        clean stop closes its own events, but a killed process cannot, and the row then
        claims to be in progress for as long as it exists. This is the only moment the
        claim is provably false — this run has just begun and owns the profile's sources,
        so no event of this profile can legitimately be open.

        Best effort by design: a watcher must start whether or not the tidy-up worked.
        """
        try:
            from db_client.models.close_stranded_events_request import CloseStrandedEventsRequest

            api = WatchProfilesApi(self._api_client_factory())
            payload = CloseStrandedEventsRequest(profile_uuid=self._profile_uuid)
            response = await asyncio.to_thread(api.close_stranded_watch_events, payload)
            if response.status != JsonValues.SUCCESS:
                raise RuntimeError(response.message or "db refused the request")
            return int(response.closed or 0)
        except Exception as e:
            logger.warning("Could not close stranded events for profile %s: %s", self._profile_uuid, e)
            self._log(f"Could not close events left open by a previous run: {e}")
            return 0

    # ── clips ────────────────────────────────────────────────────────────────

    async def attach_recording(
        self,
        signal: WatchEventSignal,
        event_uuid: Optional[str],
        clip: Dict[str, Any],
        context: Optional[ActionContext] = None,
        detection_track: Optional[Any] = None,
    ) -> Optional[str]:
        """Register a written media file and link it to its event.

        `clip` is a media action's result data, carrying the `kind` the row takes —
        `clip` for `record_clip`, `timelapse` for `record_event_timelapse`. An event may
        have one of each, so this may be called more than once per closing edge; the
        link is `recordings.event_uuid` and nothing on the event has to be patched.

        For a clip, `startedAt` is the clip's
        own first frame, which is *earlier* than the event's start by the pre-roll that
        was actually available — MediaMTX's playback server clamps the request to what
        the buffer still holds and says so via `partialPreroll`. Storing the clamped
        start rather than the event's is what lets the review page mark where in the
        clip the rule fired, and what keeps a truncated clip from being presented as
        though its lead-in were intact.

        `detection_track` is the source's ring of what the detector saw. When the action
        asked for one, its slice over this clip's window is written beside the mp4 and
        named on the row — see `_write_overlay`.
        """
        if not clip.get("path"):
            return None
        try:
            _, WatchEventUpdate, RecordingCreate = _payload_models()
            started_at = _parse_iso(clip.get("startedAt"))
            duration_s = float(clip.get("durationS") or 0.0)
            overlay_path = self._write_overlay(signal, event_uuid, clip, context, detection_track)
            api = RecordingsApi(self._api_client_factory())
            payload = RecordingCreate(
                source_uuid=signal.source_uuid or None,
                profile_uuid=self._profile_uuid,
                event_uuid=event_uuid,
                kind=str(clip.get("kind") or RECORDING_KIND_CLIP),
                file_path=str(clip["path"]),
                overlay_path=str(overlay_path) if overlay_path else None,
                started_at=started_at,
                ended_at=(started_at + timedelta(seconds=duration_s)) if started_at else None,
                duration_s=duration_s,
                size_bytes=int(clip.get("sizeBytes") or 0),
                partial_preroll=bool(clip.get("partialPreroll")),
                metadata_json=_media_metadata(clip),
            )
            response = await asyncio.to_thread(api.create_recording, payload)
            if response.status != JsonValues.SUCCESS or response.recording is None:
                raise RuntimeError(response.message or "db refused the recording")
            recording_uuid = response.recording.uuid
            if self._analyse_blips and overlay_path and event_uuid:
                await self._file_blips(event_uuid, recording_uuid, overlay_path)
            return recording_uuid
        except Exception as e:
            self._failures += 1
            logger.exception("Could not register recording for rule %s", signal.rule_name)
            self._log(f"Recording written but not registered ({signal.rule_name}): {e}")
            return None

    def _write_overlay(
        self,
        signal: WatchEventSignal,
        event_uuid: Optional[str],
        clip: Dict[str, Any],
        context: Optional[ActionContext],
        detection_track: Optional[Any],
    ) -> Optional[Path]:
        """Write the clip's detection sidecar. Returns its path, or None.

        Never raises and never fails the clip: the mp4 is the artifact, the boxes are a
        commentary on it, and losing the commentary must not lose the recording — the
        same contract the card art has.

        Written *before* the row is created, not after, so `overlay_path` is set in the
        same insert. Setting it in a follow-up update would leave a window in which the
        file exists and no row names it, which is precisely what the orphan sweep is
        built to delete.

        Deliberately synchronous. It is a few tens of KB of JSON serialised from data
        already in memory — a thread hop would cost more than the write.
        """
        # An action that knows its own time base writes its own sidecar and hands over
        # the path — `record_event_timelapse` does, because a timelapse's video seconds
        # are not world seconds and the ring slice below would place every box hours
        # past the end of an eighteen-second video.
        written = clip.get("overlayPath")
        if written:
            return Path(written)
        if not clip.get("overlay") or detection_track is None or context is None:
            return None
        started_at = _parse_iso(clip.get("startedAt"))
        duration_s = float(clip.get("durationS") or 0.0)
        if started_at is None or duration_s <= 0:
            return None
        try:
            from apps.helpers.recordings.detectionTrack import (
                build_overlay,
                event_window,
                overlay_path_for,
                write_overlay,
            )

            samples = detection_track.samples_between(started_at.timestamp(), started_at.timestamp() + duration_s)
            if not samples:
                # Nothing was analysed inside the clip's window — a source whose watcher
                # started mid-buffer, say. An empty sidecar would tell the player the
                # detector saw nothing, which is a stronger claim than "no record".
                return None
            document = build_overlay(
                samples,
                started_at,
                duration_s,
                event=event_window(
                    started_at,
                    duration_s,
                    self._wallclock(signal.started_at, context),
                    self._wallclock(signal.at, context),
                    rule_name=signal.rule_name,
                    event_uuid=event_uuid,
                ),
            )
            dest = overlay_path_for(Path(clip["path"]))
            write_overlay(document, dest)
            return dest
        except Exception as e:
            logger.warning("Could not write detection overlay for %s: %s", signal.rule_name, e)
            self._log(f"Clip overlay not written ({signal.rule_name}): {e}")
            return None

    async def _file_blips(self, event_uuid: str, recording_uuid: str, overlay_path: Path) -> None:
        """Find where tracking went wrong in the clip just registered, and file it.

        Runs here, inline, rather than in a task of its own. The analysis is a JSON read
        and some arithmetic over a few hundred samples — no inference, no video decode —
        so the orchestration a separate task would need costs more than the work it
        would schedule, and this is the one moment where the clip, its sidecar and its
        recording uuid are all in hand at once.

        The read is threaded even so. It is small, but it is disk, and the frame loop is
        the one thing in this process that must never wait on disk.

        **Never raises**, on the same contract as everything else in this module: a
        watcher that stops recording events because an analysis hiccuped is a bug, and
        one that stops *watching* for it is far worse. A missing analysis costs a badge
        on a review card; there is no correctness the frame loop depends on here.
        """
        try:
            from apps.helpers.recordings.blips import BlipThresholds, analyse_overlay

            thresholds = BlipThresholds.from_overrides(self._blip_threshold_overrides)
            document = await asyncio.to_thread(lambda: json.loads(Path(overlay_path).read_text()))
            blips = await asyncio.to_thread(analyse_overlay, document, thresholds)
            if not blips:
                return

            ReplaceEventBlipsRequest, EventBlipPayload = _blip_payload_models()
            api = WatchProfilesApi(self._api_client_factory())
            payload = ReplaceEventBlipsRequest(
                event_uuid=event_uuid,
                recording_uuid=recording_uuid,
                blips=[
                    EventBlipPayload(
                        kind=blip.kind,
                        track_id=blip.track_id,
                        start_s=blip.start_s,
                        end_s=blip.end_s,
                        severity=blip.severity,
                        fault=blip.fault,
                        harvestable=blip.harvestable,
                        detail=dict(blip.detail),
                    )
                    for blip in blips
                ],
            )
            response = await asyncio.to_thread(api.replace_event_blips, payload)
            if response.status != JsonValues.SUCCESS:
                raise RuntimeError(response.message or "db refused the blips")
            logger.info("Filed %d tracking blip(s) for event %s", len(blips), event_uuid)
        except Exception as e:
            self._failures += 1
            logger.warning("Could not analyse tracking for event %s: %s", event_uuid, e)
            self._log(f"Tracking analysis skipped for this clip: {e}")

    # ── card art ─────────────────────────────────────────────────────────────

    async def _encode_snapshot(self, signal: WatchEventSignal, context: ActionContext) -> Optional[Tuple[Path, bytes]]:
        """Render the event's thumbnail and its destination, without touching the disk.

        Encoded at the opening edge because that is the only moment the right frame
        exists, but not *written* there: most events never earn a row, and writing a jpg
        for each of them fills a directory the retention sweep then has to walk with
        files no card will ever point at.
        """
        frame = context.frame
        if frame is None:
            return None
        when = self._wallclock(signal.started_at, context) or datetime.now(timezone.utc)
        dest = self._snapshot_path(signal, context, when)
        try:
            payload = await asyncio.to_thread(self._encode, frame, context)
            return dest, payload
        except Exception as e:
            # The row is worth more than the picture, so this never propagates.
            logger.warning("Event snapshot failed for %s: %s", signal.rule_name, e)
            self._log(f"Event snapshot failed ({signal.rule_name}): {e}")
            return None

    async def _flush_snapshot(self, signal: WatchEventSignal, pending: "_PendingEvent") -> Optional[Path]:
        """Write the held card art now that the event has earned a row."""
        if pending.snapshot is None:
            return None
        dest, payload = pending.snapshot
        try:
            await asyncio.to_thread(self._write_bytes, dest, payload)
            return dest
        except Exception as e:
            logger.warning("Event snapshot not written for %s: %s", signal.rule_name, e)
            self._log(f"Event snapshot not written ({signal.rule_name}): {e}")
            return None

    @staticmethod
    def _encode(frame: Any, context: ActionContext) -> bytes:
        # Annotate the *analysis* frame, not `full_frame`: detections are in the
        # analysis frame's pixel space, so this needs no `scale` and cannot land every
        # box in the top-left quadrant. Card art wants legibility, not resolution — the
        # clip is where the full-resolution record lives.
        canvas = frame
        if context.detections:
            canvas = annotate_detections(canvas, context.detections, colours=context.label_colours)
        canvas = downscale_to_width(canvas, SNAPSHOT_WIDTH)
        return encode_jpg_bytes(canvas, quality=SNAPSHOT_QUALITY)

    @staticmethod
    def _write_bytes(dest: Path, payload: bytes) -> int:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        return len(payload)

    def _snapshot_path(self, signal: WatchEventSignal, context: ActionContext, when: datetime) -> Path:
        from apps.helpers.actions.mqttPublishAction import slugify_topic

        source = slugify_topic(context.source_name or context.source_uuid or "source")
        rule = slugify_topic(signal.rule_name or "event")
        # The timestamp alone is not unique. It is the event's *start*, not the moment
        # this file is written, so two rules opening on the same analysed frame for the
        # same camera produce the identical instant — and if they also share a name, the
        # identical path, leaving one event's card showing the other's picture. The
        # discriminator is the event key, so it separates exactly the events the rest of
        # this class treats as distinct (and keeps doing so when per-pair identity lands
        # and the refs start varying).
        discriminator = hashlib.sha1("\x1f".join(_event_key(signal)).encode()).hexdigest()[:6]
        stamp = when.strftime("%H-%M-%S")
        return self._snapshot_root / when.strftime("%Y-%m-%d") / f"{stamp}_{source}_{rule}_{discriminator}.jpg"

    # ── clocks ───────────────────────────────────────────────────────────────

    @staticmethod
    def _wallclock(monotonic_s: float, context: ActionContext) -> Optional[datetime]:
        """Rule timing is monotonic; rows are wallclock.

        The task supplies the offset rather than every consumer re-deriving it, for the
        same reason the clip backend takes it: two consumers computing it independently
        would drift apart, and an event whose row and clip disagree about when it
        happened is unreviewable.
        """
        if not context.wallclock_offset:
            return None
        return datetime.fromtimestamp(monotonic_s + context.wallclock_offset, tz=timezone.utc)


class NullWatchEventRecorder(WatchEventRecorder):
    """Records nothing. The state a task is in before its run loop has started.

    A null object rather than an `Optional`: `_dispatch` would otherwise need a guard on
    every one of its persistence calls, and a missed guard would be an `AttributeError`
    inside the frame loop — the one place this module promises never to raise.
    """

    def __init__(self) -> None:
        super().__init__(profile_uuid="", api_client_factory=lambda: None, log=lambda _msg: None)  # type: ignore[arg-type]

    async def record(self, signal: WatchEventSignal, context: ActionContext) -> Optional[str]:
        return None

    async def ensure_event(self, signal: WatchEventSignal, context: ActionContext) -> Optional[str]:
        return None

    async def close_all(self) -> None:
        return None

    async def close_stranded(self) -> int:
        return 0

    async def attach_recording(
        self,
        signal: WatchEventSignal,
        event_uuid: Optional[str],
        clip: Dict[str, Any],
        context: Optional[ActionContext] = None,
        detection_track: Optional[Any] = None,
    ) -> Optional[str]:
        return None


def _media_metadata(media: Dict[str, Any]) -> Dict[str, Any]:
    """The display-worthy part of a media action's result.

    Filtered to a known list rather than passed through wholesale: the result dict is an
    action's private return value and gains keys freely, and a row that mirrors all of
    them turns every internal rename into a schema change nobody noticed.
    """
    return {key: media[key] for key in MEDIA_METADATA_KEYS if media.get(key) is not None}


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    # The clip backends emit UTC-aware strings, but a naive one must not be silently
    # read as local time by whatever reads the row back.
    return parse_iso_utc(value)
