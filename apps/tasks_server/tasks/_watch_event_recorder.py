"""Turn the rule engine's event signals into durable `watch_events` rows.

Why this is a task concern and not an action's: `ActionContext` states outright that
the task keeps sole ownership of its API clients and DB writes, so actions are pure
side effects on the outside world. Persistence has to live above them — and it must be
unconditional, because the review page is a record of *what the watcher saw*, not a
record of which actions someone happened to enable. A profile with no actions at all
still produces a reviewable history.

Two artifacts per event:

* **The row.** Written on START so an event is visible while it is still open, and
  patched on END with the closing time, peak confidence and the runaway-cap flag.
  Rule and slot names are denormalised onto it — see the model docstring.
* **The card art.** A small annotated JPEG written on START. Distinct from what a
  `save_snapshot` action writes: that is a full-quality *training image* under
  `image_datasets` which the user curates and which must never have boxes burnt into
  it. This one is thumbnail-sized, has the boxes drawn on deliberately (the whole
  point of the card is to show what acted on what), lives outside the image datasets,
  and is deleted with its event.

Nothing here may raise into the frame loop. A watcher that stops recording events
because the DB hiccuped is a bug; a watcher that stops *watching* for the same reason
is a much worse one, so every failure is logged and swallowed exactly as an action
failure is.
"""

import asyncio
import hashlib
import logging
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
    ) -> None:
        self._profile_uuid = profile_uuid
        self._api_client_factory = api_client_factory
        self._log = log
        self._snapshot_root = Path(snapshot_root)
        #: Open events by key, so a closing edge can find the row it opened.
        self._open: Dict[EventKey, str] = {}
        self._failures = 0

    @property
    def failures(self) -> int:
        return self._failures

    # ── the two edges ─────────────────────────────────────────────────────────

    async def record(self, signal: WatchEventSignal, context: ActionContext) -> Optional[str]:
        """Persist one transition. Returns the event uuid, when there is one.

        UPDATE is deliberately ignored: it re-fires the `event_started` trigger for
        actions that want a repeated nudge, but it opens nothing, so persisting it
        would turn one visit into a row per frame.
        """
        try:
            if signal.phase is EventPhase.START:
                return await self._open_event(signal, context)
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

    async def _open_event(self, signal: WatchEventSignal, context: ActionContext) -> str:
        # The snapshot is written first so the row is never briefly snapshot-less: a
        # card that pops into existence without art and gains it a moment later flickers
        # in a list that polls. A failure here costs the art, not the event.
        snapshot_path = await self._write_snapshot(signal, context)

        WatchEventCreate, _, _ = _payload_models()
        api = WatchProfilesApi(self._api_client_factory())
        payload = WatchEventCreate(
            profile_uuid=self._profile_uuid,
            source_uuid=signal.source_uuid or None,
            rule_uuid=signal.rule_uuid or None,
            rule_name=signal.rule_name or None,
            event_label=(signal.metadata or {}).get("eventLabel"),
            started_at=self._wallclock(signal.started_at, context),
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
        uuid = response.event.uuid
        self._open[_event_key(signal)] = uuid
        return uuid

    async def _close_event(self, signal: WatchEventSignal) -> Optional[str]:
        uuid = self._open.pop(_event_key(signal), None)
        if uuid is None:
            # An event whose START was never persisted (DB down at the time, or a
            # watcher resumed mid-event). Closing nothing is correct; inventing a
            # zero-length row at the closing edge would be worse than the gap.
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
        """Forget open events without closing them.

        Called from `_deinit` *after* the run loop has already emitted a closing signal
        for everything still open, so anything left here failed to persist and has no
        row to patch.
        """
        self._open.clear()

    # ── clips ────────────────────────────────────────────────────────────────

    async def attach_recording(
        self,
        signal: WatchEventSignal,
        event_uuid: Optional[str],
        clip: Dict[str, Any],
    ) -> Optional[str]:
        """Register a written clip and link it to its event.

        `clip` is a `record_clip` action's result data. Its `startedAt` is the clip's
        own first frame, which is *earlier* than the event's start by the pre-roll that
        was actually available — MediaMTX's playback server clamps the request to what
        the buffer still holds and says so via `partialPreroll`. Storing the clamped
        start rather than the event's is what lets the review page mark where in the
        clip the rule fired, and what keeps a truncated clip from being presented as
        though its lead-in were intact.
        """
        if not clip.get("path"):
            return None
        try:
            _, WatchEventUpdate, RecordingCreate = _payload_models()
            started_at = _parse_iso(clip.get("startedAt"))
            duration_s = float(clip.get("durationS") or 0.0)
            api = RecordingsApi(self._api_client_factory())
            payload = RecordingCreate(
                source_uuid=signal.source_uuid or None,
                profile_uuid=self._profile_uuid,
                event_uuid=event_uuid,
                file_path=str(clip["path"]),
                started_at=started_at,
                ended_at=(started_at + timedelta(seconds=duration_s)) if started_at else None,
                duration_s=duration_s,
                size_bytes=int(clip.get("sizeBytes") or 0),
                partial_preroll=bool(clip.get("partialPreroll")),
            )
            response = await asyncio.to_thread(api.create_recording, payload)
            if response.status != JsonValues.SUCCESS or response.recording is None:
                raise RuntimeError(response.message or "db refused the recording")
            recording_uuid = response.recording.uuid

            if event_uuid:
                # Both directions: `recordings.event_uuid` answers "what fired this
                # clip", `watch_events.recording_uuid` answers "does this card have a
                # video", and the card asks the second one on every row.
                watch_api = WatchProfilesApi(self._api_client_factory())
                await asyncio.to_thread(
                    watch_api.update_watch_event,
                    event_uuid,
                    WatchEventUpdate(recording_uuid=recording_uuid),
                )
            return recording_uuid
        except Exception as e:
            self._failures += 1
            logger.exception("Could not register clip for rule %s", signal.rule_name)
            self._log(f"Clip written but not recorded ({signal.rule_name}): {e}")
            return None

    # ── card art ─────────────────────────────────────────────────────────────

    async def _write_snapshot(self, signal: WatchEventSignal, context: ActionContext) -> Optional[Path]:
        """Write the event's thumbnail. Returns None if there was nothing to write."""
        frame = context.frame
        if frame is None:
            return None
        when = self._wallclock(signal.started_at, context) or datetime.now(timezone.utc)
        dest = self._snapshot_path(signal, context, when)
        try:
            await asyncio.to_thread(self._encode_and_write, frame, context, dest)
            return dest
        except Exception as e:
            # The row is worth more than the picture, so this never propagates.
            logger.warning("Event snapshot failed for %s: %s", signal.rule_name, e)
            self._log(f"Event snapshot failed ({signal.rule_name}): {e}")
            return None

    @staticmethod
    def _encode_and_write(frame: Any, context: ActionContext, dest: Path) -> int:
        # Annotate the *analysis* frame, not `full_frame`: detections are in the
        # analysis frame's pixel space, so this needs no `scale` and cannot land every
        # box in the top-left quadrant. Card art wants legibility, not resolution — the
        # clip is where the full-resolution record lives.
        canvas = frame
        if context.detections:
            canvas = annotate_detections(canvas, context.detections, colours=context.label_colours)
        canvas = downscale_to_width(canvas, SNAPSHOT_WIDTH)
        payload = encode_jpg_bytes(canvas, quality=SNAPSHOT_QUALITY)
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

    async def attach_recording(self, signal: WatchEventSignal, event_uuid: Optional[str], clip: Dict[str, Any]):
        return None


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    # The clip backends emit UTC-aware strings, but a naive one must not be silently
    # read as local time by whatever reads the row back.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
