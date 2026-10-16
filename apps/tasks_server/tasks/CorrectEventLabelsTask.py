"""CorrectEventLabelsTask — "the watcher called all of these the wrong thing".

The bulk half of the correction loop. One event is corrected in the browser, where the
clip is already open and the reviewer is looking at the frame they disagree with
(`labelCorrectionModal.js`). A *selection* cannot be: that path drives a real `<video>`,
needs the tab foregrounded, and spends several seconds of seeking and canvas encoding
per clip. Forty events is minutes of a tab nobody may touch, and a group card stands for
383 of them.

So the selection bar hands the same remap to this task and it does the identical work
from the files — the clip and its detection sidecar — with nothing on screen.

**Two halves, and the row half is not conditional on the frame half.** An event whose
subject was wrong reads wrong in the grid, groups under the wrong identity and answers
the wrong filter; that correction is what the user sees immediately, and it must not be
skipped for a clip whose sidecar yields no frames worth saving. So rows are patched
first, per event, before the harvest that follows them.

**Boxes are written Unverified**, exactly as `saveFrameForAnnotation` writes any model
output. A corrected label on a detector's rectangle is still a guess about *where*, and
the review queue is where that gets settled.

Which frames, and why not all of them: see `apps.helpers.recordings.labelCorrection`.
Its spread is deliberately simpler than the browser's hard-moment ranking, which costs
a blips request per event.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from db_client.api.images_api import ImagesApi
from db_client.api.labels_api import LabelsApi
from db_client.api.recordings_api import RecordingsApi
from db_client.api.watch_profiles_api import WatchProfilesApi

from apps.helpers.consts import JsonValues, TaskAccent
from apps.helpers.db.types import UNVERIFIED_TAG_UUID
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.recordings.labelCorrection import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_MIN_FRAME_GAP_S,
    DEFAULT_WEAK_CONF_MAX,
    NO_BBOX,
    WEAK,
    apply_label_remap,
    dedupe_overlapping_boxes,
    event_correction_for,
    frames_holding_labels,
    pick_training_frames,
    sample_at,
)

from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

#: Where corrected frames land under FILES_ROOT. The same directory the browser's
#: single-event correction writes to, deliberately: a correction made from the grid and
#: one made from the player are the same artifact, and the `corrected_from` metadata is
#: what distinguishes either from an ordinary capture.
DEFAULT_CORRECTION_DIR = "watch_captures"

#: Sanity bound on one run. A group card can stand for several hundred events, which is
#: a legitimate selection; past this it is an API caller asking for the whole history.
_SANITY_MAX_EVENTS = 2_000


@register_task()
class CorrectEventLabelsTask(Task):
    # Gathering training data — same family as the blip harvest and the collector.
    CARD_ACCENT = TaskAccent.GREEN

    # Started from the events grid's selection bar, on a remap the correction dialog
    # produced. There is no way to name a label remap by hand.
    SYSTEM_TASK = True

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "meta": {"order": ["event_uuids", "remap", "fix_events", "max_frames_per_event", "dest_path"]},
            "event_uuids": {
                "type": "list",
                "required": True,
                "help": "Which events to correct. Selected in the review UI.",
            },
            "remap": {
                "type": "dict",
                "required": True,
                "help": (
                    "Wrong label uuid → what it should have been. Two values are not "
                    f"labels: '{NO_BBOX}' removes the box instead of relabelling it, and "
                    f"'{WEAK}' keeps the label and collects only the frames the detector "
                    "hedged on."
                ),
            },
            "fix_events": {
                "type": "bool",
                "default": True,
                "help": (
                    "Also rewrite each event's own subject/target, so the rows read, "
                    "group and filter as the corrected label rather than the mistake."
                ),
            },
            "max_frames_per_event": {
                "type": "int",
                "default": DEFAULT_MAX_FRAMES,
                "min": 1,
                "max": 40,
                "help": (
                    "Frames to harvest from each event's clip. Consecutive frames of one "
                    "subject are near-duplicates, so more rarely means more signal."
                ),
            },
            "min_frame_gap_s": {
                "type": "float",
                "default": DEFAULT_MIN_FRAME_GAP_S,
                "help": "Closest two harvested frames may sit in clip time.",
            },
            "weak_conf_max": {
                "type": "float",
                "default": DEFAULT_WEAK_CONF_MAX,
                "help": "Confidence below which a box counts as one the detector hedged on.",
            },
            "dest_path": {
                "type": "str",
                "default": DEFAULT_CORRECTION_DIR,
                "help": "Folder under the image root where corrected frames are written.",
            },
        }

    async def _init(self, params: Dict[str, Any], resume_data: Optional[Dict[str, Any]]) -> None:
        self._event_uuids: List[str] = [str(u) for u in (params.get("event_uuids") or []) if u]
        if not self._event_uuids:
            raise ValueError("event_uuids is required")
        if len(self._event_uuids) > _SANITY_MAX_EVENTS:
            raise ValueError(
                f"event_uuids has {len(self._event_uuids)} entries, over the sanity cap of "
                f"{_SANITY_MAX_EVENTS}; narrow the selection before correcting"
            )

        raw_remap = params.get("remap") or {}
        if not isinstance(raw_remap, dict) or not raw_remap:
            raise ValueError("remap is required, as {wrong_label_uuid: corrected_label_uuid}")
        # A remap entry pointing a label at itself is a no-op dressed as a correction: it
        # would file frames of a label nobody questioned and flag the events for review.
        self._remap: Dict[str, str] = {str(k): str(v) for k, v in raw_remap.items() if k and v and str(k) != str(v)}
        if not self._remap:
            raise ValueError("remap changes nothing")

        self._fix_events: bool = bool(params.get("fix_events", True))
        self._max_frames: int = max(1, int(params.get("max_frames_per_event") or DEFAULT_MAX_FRAMES))
        self._min_gap_s: float = float(params.get("min_frame_gap_s") or DEFAULT_MIN_FRAME_GAP_S)
        self._weak_conf_max: float = float(params.get("weak_conf_max") or DEFAULT_WEAK_CONF_MAX)
        self._dest_path: str = str(params.get("dest_path") or DEFAULT_CORRECTION_DIR)

        resume = resume_data or {}
        self._frames_saved: int = int(resume.get("frames_saved", 0))
        self._boxes_written: int = int(resume.get("boxes_written", 0))
        self._boxes_removed: int = int(resume.get("boxes_removed", 0))
        self._events_corrected: int = int(resume.get("events_corrected", 0))
        self._duplicates: int = int(resume.get("duplicates", 0))
        self._no_frames: int = int(resume.get("no_frames", 0))
        self._frames_undecodable: int = int(resume.get("frames_undecodable", 0))
        self._no_clip: int = int(resume.get("no_clip", 0))
        self._failed: int = int(resume.get("failed", 0))
        self._sample_errors: List[str] = list(resume.get("sample_errors", []))
        self._done_events: set = set(resume.get("done_events", []))
        self._label_names: Dict[str, str] = dict(resume.get("label_names", {}))

    def _api(self, cls):
        return cls(self._manager.get_db_api_client())

    # ── the run ──────────────────────────────────────────────────────────────

    async def _run(self) -> Dict[str, Any]:
        if not self._label_names:
            self._label_names = await self._fetch_label_names()
        self._drop_entries_that_are_not_labels()

        self._describe_remap()

        for index, event_uuid in enumerate(self._event_uuids):
            if self._cancel_flag.is_set():
                break
            await self._check_pause()
            if event_uuid in self._done_events:
                continue
            try:
                await self._correct_event(event_uuid)
            except Exception as e:
                logger.exception("Correction failed for event %s", event_uuid)
                self._note_failure(f"{event_uuid}: {e}")
                self._failed += 1
            # Marked done even on failure: a retry of the whole selection should not
            # redo the events that already landed, and a failure is reported in the
            # result rather than silently retried by a resume.
            self._done_events.add(event_uuid)
            self._update_progress(min(100.0, (index + 1) / len(self._event_uuids) * 100.0))

        return self._build_result()

    def _drop_entries_that_are_not_labels(self) -> None:
        """Keep only the remap entries that name labels on both sides.

        An event's `target_ref` is not necessarily a label: a zone-kind rule stores the
        *zone's* uuid there. A remap keyed on one would rewrite `target_ref` from a zone
        to a label, leaving a row that points at something which is not a place — and a
        zone uuid matches no sidecar box either, so the entry could only ever do harm.

        The dialog already filters its rows this way; this is the guard for a direct API
        caller, and for a label deleted between the dialog and the run.
        """
        kept = {
            k: v
            for k, v in self._remap.items()
            if k in self._label_names and (v in (NO_BBOX, WEAK) or v in self._label_names)
        }
        dropped = set(self._remap) - set(kept)
        if dropped:
            self.log(
                f"{len(dropped)} remap entry/entries name something that is not a label "
                f"(a zone, or a deleted label) and were skipped: {sorted(dropped)}"
            )
        self._remap = kept
        if not self._remap:
            raise ValueError("no remap entry names a label on both sides")

    def _describe_remap(self) -> None:
        """Say what the run is about to do, in label names, once."""
        self.log(
            f"Correcting {len(self._event_uuids)} event(s): "
            + ", ".join(f"{self._name_of(k)} → {self._describe_target(v)}" for k, v in self._remap.items())
        )

    async def _correct_event(self, event_uuid: str) -> None:
        events = self._api(WatchProfilesApi)
        response = await asyncio.to_thread(events.get_watch_event, event_uuid)
        if response.status != JsonValues.SUCCESS or response.event is None:
            raise RuntimeError(response.message or "the event is gone")
        event = response.event

        # The row first, and independently of the frames below — see the module docstring.
        if self._fix_events:
            await self._correct_event_row(event)

        clip_path, overlay_path, source_uuid = await self._clip_for_event(event_uuid)
        if not clip_path or not overlay_path:
            # Not a failure. A clip the recorder never wrote, or one with no sidecar, has
            # no per-frame boxes to relabel — and the row correction above is the half
            # that was visible anyway.
            self._no_clip += 1
            return

        document = await asyncio.to_thread(_read_json, overlay_path)
        times = self._times_for(document)
        if not times:
            self._no_frames += 1
            return

        for at_s in times:
            if self._cancel_flag.is_set():
                return
            await self._check_pause()
            await self._harvest_frame(event, document, clip_path, at_s, source_uuid)

    def _times_for(self, document: Dict[str, Any]) -> List[float]:
        """The moments this event's correction harvests.

        A relabelled label contributes every frame holding it; a `WEAK` one contributes
        only the frames it hedged on, since the rest are frames of a label the detector
        already gets right.
        """
        relabelled = [k for k, v in self._remap.items() if v != WEAK]
        weak_only = [k for k, v in self._remap.items() if v == WEAK]
        candidates = set(frames_holding_labels(document, relabelled))
        candidates |= set(frames_holding_labels(document, weak_only, max_conf=self._weak_conf_max))
        return pick_training_frames(sorted(candidates), self._max_frames, self._min_gap_s)

    async def _harvest_frame(
        self,
        event: Any,
        document: Dict[str, Any],
        clip_path: str,
        at_s: float,
        source_uuid: str,
    ) -> None:
        """One moment: its picture, saved, wearing the corrected boxes."""
        frame_sample = sample_at(document, at_s)
        original = (frame_sample or {}).get("boxes") or []
        remapped = dedupe_overlapping_boxes(apply_label_remap(original, self._remap))
        removed = len([b for b in original if self._remap.get(b.get("labelUuid")) == NO_BBOX])
        # A sample that never held anything is skipped; one the correction *emptied* is
        # not. A frame whose only box was a false positive is a picture of the detector
        # being wrong about an empty room, which is exactly the negative example a
        # false-positive correction exists to produce — see `apply_label_remap`. Only
        # the first of those two cases is clutter.
        if not original:
            return

        jpeg = await asyncio.to_thread(_grab_frame, clip_path, at_s)
        if jpeg is None:
            # Routine rather than exceptional: a sidecar describes the *event*, and the
            # mp4 beside it is whatever the recorder managed to write — a Ring clip is
            # its own fixed length while the watcher's event ran minutes. A moment past
            # the end of the file cannot be captured, and `grab_frame_array` says so by
            # returning nothing rather than clamping to the final frame.
            self._frames_undecodable += 1
            return

        uploaded = await self._upload(jpeg, event, at_s, source_uuid)
        if uploaded is None:
            return

        self._frames_saved += 1
        self._boxes_removed += removed
        await self._write_boxes(uploaded, event, remapped, at_s)

    # ── the writes ───────────────────────────────────────────────────────────

    async def _correct_event_row(self, event: Any) -> None:
        """Rewrite one event's subject/target to what the correction says they were.

        Reported and swallowed on failure: the frame harvest that follows is
        independently valuable and must still run.
        """
        from db_client.models.watch_event_update import WatchEventUpdate

        correction = event_correction_for(event, self._remap, self._label_names)
        if not correction:
            return
        try:
            api = self._api(WatchProfilesApi)
            response = await asyncio.to_thread(
                api.update_watch_event, event.uuid, WatchEventUpdate(**correction["patch"])
            )
            if response.status != JsonValues.SUCCESS:
                raise RuntimeError(response.message or "the event refused the correction")
            self._events_corrected += 1
        except Exception as e:
            logger.warning("Could not correct event %s: %s", event.uuid, e)
            self._note_failure(f"{event.uuid}: the row was not corrected ({e})")

    async def _upload(self, jpeg: bytes, event: Any, at_s: float, source_uuid: str) -> Optional[Any]:
        """Write one frame into the dataset, letting the endpoint dedup it.

        Nobody chose these frames one by one, so a near-duplicate is dilution rather
        than a second opinion — and the spread is by clip time, which says nothing about
        a subject that did not move.
        """
        images = self._api(ImagesApi)
        name = f"corrected_{event.uuid[:8]}_t{at_s:.2f}.jpg".replace(".", "-", 1)
        response = await asyncio.to_thread(
            images.upload_images,
            files=[(name, jpeg)],
            path=self._dest_path,
            overwrite=False,
            source_uuid=source_uuid or "",
            skip_duplicates=True,
        )
        if response.status != JsonValues.SUCCESS or not response.results:
            self._failed += 1
            self._note_failure(response.message or "the upload was refused")
            return None
        result = response.results[0]
        if not result.ok:
            self._failed += 1
            self._note_failure(result.message or f"{name} was refused")
            return None
        if result.skipped or result.duplicate_of:
            self._duplicates += 1
            return None
        return result

    async def _write_boxes(self, uploaded: Any, event: Any, boxes: List[Dict[str, Any]], at_s: float) -> None:
        """Seed the saved frame with the corrected boxes, Unverified.

        Sidecar boxes are already normalised to 0..1 (`detectionTrack._normalise`), which
        is the shape the column stores — so unlike the browser path there is no
        pixel-space conversion here, and nothing to get wrong about the analysis frame's
        size versus the clip's.
        """
        from db_client.models.add_bbox_payload import AddBboxPayload

        images = self._api(ImagesApi)
        path = getattr(uploaded, "path", None)
        if not path:
            return
        corrected_from = ", ".join(f"{self._name_of(k)} → {self._describe_target(v)}" for k, v in self._remap.items())
        for box in boxes:
            label_uuid = box.get("labelUuid")
            if not label_uuid:
                continue
            try:
                await asyncio.to_thread(
                    images.add_bbox,
                    AddBboxPayload(
                        image_path=path,
                        label_uuid=label_uuid,
                        x=float(box.get("x", 0.0)),
                        y=float(box.get("y", 0.0)),
                        width=float(box.get("w", 0.0)),
                        height=float(box.get("h", 0.0)),
                        tag_uuids=[UNVERIFIED_TAG_UUID],
                        metadata_json={
                            # Stringified to match what the collector writes, so the
                            # confidence filter reads both producers' boxes alike.
                            **({"confidence": str(box["conf"])} if box.get("conf") is not None else {}),
                            "source_event": event.uuid,
                            "source_event_rule": getattr(event, "rule_name", None) or "",
                            "source_clip_offset_s": f"{at_s:.2f}",
                            # Which guess this replaced: a reviewer meeting the frame
                            # later can see it was a correction rather than a detection
                            # the model got right, and it is the record of what the
                            # model is bad at.
                            "corrected_from": corrected_from,
                            "pick_reason": "spread",
                        },
                    ),
                )
                self._boxes_written += 1
            except Exception as e:
                # The frame is already filed; a box that will not write costs that box.
                self._note_failure(f"{path}: a box was refused ({e})")

    # ── lookups ──────────────────────────────────────────────────────────────

    async def _clip_for_event(self, event_uuid: str) -> Tuple[Optional[str], Optional[str], str]:
        """The event's clip, its sidecar, and the camera it came from.

        The clip, not the timelapse: a timelapse packs minutes into seconds and its
        sidecar holds only the frames it sampled, so the spread would be over a handful
        of moments minutes apart. `kind` is left unfiltered and the first clip-kind
        recording with a sidecar wins, because an event with both owns two rows.
        """
        api = self._api(RecordingsApi)
        response = await asyncio.to_thread(api.list_recordings, event_uuid=event_uuid)
        if response.status != JsonValues.SUCCESS:
            raise RuntimeError(response.message or "could not read the event's recordings")
        for recording in response.recordings or []:
            # `.value` first: the generated client types this as an enum, and comparing
            # the member against a bare string works only because it happens to subclass
            # str — which is not a property of the spec worth depending on.
            kind = getattr(recording, "kind", None)
            if getattr(kind, "value", kind) == "timelapse":
                continue
            if recording.file_path and recording.overlay_path:
                return recording.file_path, recording.overlay_path, getattr(recording, "source_uuid", "") or ""
        return None, None, ""

    async def _fetch_label_names(self) -> Dict[str, str]:
        """uuid → name, for the corrected rows and the provenance line.

        Fetched once per run, not per event: the names go into every event's patch and
        every box's metadata.
        """
        api = self._api(LabelsApi)
        response = await asyncio.to_thread(api.list_labels)
        if response.status != JsonValues.SUCCESS:
            raise RuntimeError(response.message or "could not read the labels")
        return {label.uuid: label.name for label in (response.labels or []) if label.uuid}

    def _name_of(self, label_uuid: str) -> str:
        return self._label_names.get(label_uuid, label_uuid)

    def _describe_target(self, target: str) -> str:
        if target == NO_BBOX:
            return "no bbox"
        if target == WEAK:
            return "kept, low confidence"
        return self._name_of(target)

    # ── reporting ────────────────────────────────────────────────────────────

    def _note_failure(self, message: str) -> None:
        if len(self._sample_errors) < 10:
            self._sample_errors.append(message)
        self.log(message)

    def _result_actions(self) -> List[Dict[str, Any]]:
        """The way back to what this run produced.

        Without it the correction ends in a toast and a pile of frames three levels down
        in Review. The boxes are written Unverified, so that filter is the durable
        address for them — it outlives this task's result, which is one-shot.
        """
        return [
            {
                "type": "open_review",
                "label": "Review corrected frames",
                "payload": {"tag_uuids": [UNVERIFIED_TAG_UUID], "bbox_verification": "unverified"},
            }
        ]

    def get_status_actions(self) -> List[Dict[str, Any]]:
        return self._result_actions()

    def _build_result(self) -> Dict[str, Any]:
        return {
            "eventsCorrected": self._events_corrected,
            "framesSaved": self._frames_saved,
            "boxesWritten": self._boxes_written,
            "boxesRemoved": self._boxes_removed,
            "duplicatesSkipped": self._duplicates,
            "eventsWithNoFrames": self._no_frames,
            "framesPastEndOfClip": self._frames_undecodable,
            "eventsWithNoClip": self._no_clip,
            "failed": self._failed,
            "sampleErrors": list(self._sample_errors),
            "actions": self._result_actions(),
        }

    def get_resume_data(self) -> Dict[str, Any]:
        """Checkpointed state. `done_events` is what makes a resumed run skip the events
        it already corrected, rather than patching rows twice and leaning on dedup to
        clean up the frames."""
        return {
            "frames_saved": self._frames_saved,
            "boxes_written": self._boxes_written,
            "boxes_removed": self._boxes_removed,
            "events_corrected": self._events_corrected,
            "duplicates": self._duplicates,
            "no_frames": self._no_frames,
            "frames_undecodable": self._frames_undecodable,
            "no_clip": self._no_clip,
            "failed": self._failed,
            "sample_errors": list(self._sample_errors),
            "done_events": list(self._done_events),
            "label_names": dict(self._label_names),
        }

    async def _deinit(self) -> None:
        return None


def _read_json(path: str) -> Dict[str, Any]:
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text())


def _grab_frame(clip_path: str, at_s: float) -> Optional[bytes]:
    """One frame of the clip as JPEG bytes, at `at_s` seconds from its start.

    `grab_frame_array` counts in frames rather than milliseconds, so the four-decimal
    rounding in the sidecar's own times cannot land the seek on a neighbour — and it
    returns None past the end of the file rather than clamping to the final frame, which
    is what makes a clip shorter than its event safe to harvest from.
    """
    import cv2

    from apps.helpers.recordings.frameGrab import grab_frame_array

    frame = grab_frame_array(clip_path, at_s)
    if frame is None:
        return None
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return buffer.tobytes() if ok else None
