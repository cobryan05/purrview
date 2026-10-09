"""HarvestBlipFramesTask — turn tracking blips into training frames awaiting review.

The last step of the blip pipeline. The analyser has already said *where* the detector
went wrong; this pulls the actual pixels out of the clip and files a proposal for each,
so the frames arrive in the review browser as ordinary annotatable images with a
suggested correction attached.

**Three frames per incident, not every frame.** Onset, middle and recovery — the
boundaries are where the detector's decision actually flips, and a four-second dropout
at 30 fps is a hundred and twenty near-identical pictures that add annotation work and
no information. `harvest_times` picks the moments.

**Nothing is written that a human has not asked for.** This runs on an explicit list of
blip uuids, never on "everything found". The analysis that produced them is automatic;
turning them into dataset images is not.

**Dedup is the upload endpoint's, not ours.** `/api/images/upload` already runs the
perceptual hash and reports `duplicateOf`, so a frame that matches something already
collected is counted and skipped rather than filed twice. Re-running a harvest is
therefore close to free and does not grow the dataset.

**Only detector faults are eligible.** A blip the tracker caused is a frame the detector
already got right, and harvesting it would pad the training set with cases the model
handles — see `apps.helpers.recordings.blips` on why that distinction is the load-
bearing part of the whole feature. The task refuses them rather than silently dropping
them, because a user who selected one deserves to know why nothing happened.

The proposal itself is a `preview_candidates` row, not a bbox: an annotation only exists
once the user applies the run. Geometry the detector never produced is interpolated and
flagged `estimated`, which the review UI must render differently — a box twenty pixels
out is worse for training than no box at all.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from db_client.api.images_api import ImagesApi
from db_client.api.preview_runs_api import PreviewRunsApi
from db_client.api.recordings_api import RecordingsApi
from db_client.api.watch_profiles_api import WatchProfilesApi

from apps.helpers.consts import JsonValues, TaskAccent
from apps.helpers.db.types import HARVESTED_TAG_UUID
from apps.helpers.logUtils import LOG_LEVEL
from apps.helpers.recordings.blips import harvest_times, proposed_box_at

from .Registry import register_task
from .Task import Task

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(LOG_LEVEL)

#: Where harvested frames land under FILES_ROOT. Its own directory, for the same reason
#: `watch_captures` is separate from the user's upload target: these arrive by the
#: dozen from an automated pass, and mixing them into a hand-curated folder is a mess
#: nobody asked for.
DEFAULT_HARVEST_DIR = "blip_harvest"

#: Sanity bound on one run. Not a UX limit — the UI never approaches it — but a guard
#: against a direct API caller asking for the entire history in one task.
_SANITY_MAX_BLIPS = 2_000


def _payload_models() -> Tuple[Any, Any, Any]:
    """Deferred for the bootstrap reason the watch recorder documents: the spec is built
    by importing this app, and the client is generated from that spec."""
    from db_client.models.add_preview_candidates_payload import AddPreviewCandidatesPayload
    from db_client.models.preview_candidate_create import PreviewCandidateCreate
    from db_client.models.preview_run_create import PreviewRunCreate

    return PreviewRunCreate, PreviewCandidateCreate, AddPreviewCandidatesPayload


@register_task()
class HarvestBlipFramesTask(Task):
    # Gathering data — same family as CollectTrainingImages.
    CARD_ACCENT = TaskAccent.GREEN

    # Started from the event modal's blips panel, on a selection of blip uuids.
    # There is no way to name those by hand.
    SYSTEM_TASK = True

    @classmethod
    def params_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "meta": {"order": ["blip_uuids", "frames_per_incident", "dest_path", "run_name"]},
            "blip_uuids": {
                "type": "list",
                "required": True,
                "help": "Which tracking blips to pull frames for. Selected in the review UI.",
            },
            "frames_per_incident": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10,
                "help": (
                    "Frames to take from each incident — onset, middle and recovery at the "
                    "default. More rarely helps: consecutive frames of one dropout look alike."
                ),
            },
            "dest_path": {
                "type": "str",
                "default": DEFAULT_HARVEST_DIR,
                "help": "Folder under the image root where harvested frames are written.",
            },
            "run_name": {
                "type": "str",
                "default": "",
                "help": "Name for the preview run these proposals land in. Defaults to the date.",
            },
        }

    async def _init(self, params: Dict[str, Any], resume_data: Optional[Dict[str, Any]]) -> None:
        self._blip_uuids: List[str] = [str(u) for u in (params.get("blip_uuids") or []) if u]
        if not self._blip_uuids:
            raise ValueError("blip_uuids is required")
        if len(self._blip_uuids) > _SANITY_MAX_BLIPS:
            raise ValueError(
                f"blip_uuids has {len(self._blip_uuids)} entries, over the sanity cap of "
                f"{_SANITY_MAX_BLIPS}; narrow the selection before harvesting"
            )

        self._frames_per_incident: int = max(1, int(params.get("frames_per_incident", 3) or 3))
        self._dest_path: str = str(params.get("dest_path") or DEFAULT_HARVEST_DIR)
        self._run_name: str = str(params.get("run_name") or "")

        resume = resume_data or {}
        self._harvested: int = int(resume.get("harvested", 0))
        self._duplicates: int = int(resume.get("duplicates", 0))
        self._no_proposal: int = int(resume.get("no_proposal", 0))
        self._skipped_not_harvestable: int = int(resume.get("skipped_not_harvestable", 0))
        self._failed: int = int(resume.get("failed", 0))
        self._sample_errors: List[str] = list(resume.get("sample_errors", []))
        self._done_blips: set = set(resume.get("done_blips", []))
        self._preview_run_uuid: Optional[str] = resume.get("preview_run_uuid")

    def _api(self, cls):
        return cls(self._manager.get_db_api_client())

    async def _run(self) -> Dict[str, Any]:
        blips = await self._fetch_blips()
        if not blips:
            raise ValueError("none of the requested blips still exist")

        eligible = [b for b in blips if getattr(b, "harvestable", False)]
        self._skipped_not_harvestable = len(blips) - len(eligible)
        if self._skipped_not_harvestable:
            self.log(
                f"{self._skipped_not_harvestable} blip(s) skipped: the tracker was at fault, "
                "so the detector already got those frames right"
            )
        if not eligible:
            return self._build_result()

        if self._preview_run_uuid is None:
            self._preview_run_uuid = await self._create_run(eligible)

        # Grouped by recording so one clip is opened once, however many incidents it holds.
        by_recording: Dict[str, List[Any]] = {}
        for blip in eligible:
            if blip.uuid in self._done_blips:
                continue
            by_recording.setdefault(blip.recording_uuid or "", []).append(blip)

        for recording_uuid, group in by_recording.items():
            if self._cancel_flag.is_set():
                break
            await self._check_pause()
            if not recording_uuid:
                self._note_failure("a blip names no recording, so there is no clip to pull from")
                self._failed += len(group)
                continue
            try:
                await self._harvest_recording(recording_uuid, group)
            except Exception as e:
                logger.exception("Harvest failed for recording %s", recording_uuid)
                self._note_failure(f"{recording_uuid}: {e}")
                self._failed += len(group)

        return self._build_result()

    # ── fetching ─────────────────────────────────────────────────────────────

    async def _fetch_blips(self) -> List[Any]:
        api = self._api(WatchProfilesApi)
        response = await asyncio.to_thread(api.list_event_blips, uuids=self._blip_uuids, limit=_SANITY_MAX_BLIPS)
        if response.status != JsonValues.SUCCESS:
            raise RuntimeError(response.message or "could not read the blips")
        return list(response.blips or [])

    async def _create_run(self, blips: List[Any]) -> str:
        PreviewRunCreate, _, _ = _payload_models()
        api = self._api(PreviewRunsApi)
        # Plural, unlike the plan's `{eventUuid, blipUuids}`: blips are selected by uuid
        # and nothing constrains a selection to one event, so a run routinely spans
        # several. A single uuid here would name whichever event happened to sort first
        # and quietly misattribute the rest.
        event_uuids = sorted({b.event_uuid for b in blips if getattr(b, "event_uuid", None)})
        payload = PreviewRunCreate(
            name=self._run_name or "Tracking blips",
            # No model ran here. The proposal comes from the track's own majority, and
            # naming a model would imply a second opinion that was never asked for.
            model_name="",
            task_typename=type(self).__name__,
            params_json={"framesPerIncident": self._frames_per_incident},
            scope_json={"blipUuids": self._blip_uuids, "eventUuids": event_uuids},
        )
        response = await asyncio.to_thread(api.create_preview_run, payload)
        if response.status != JsonValues.SUCCESS or response.run is None:
            raise RuntimeError(response.message or "could not open a preview run")
        return response.run.uuid

    # ── the work ─────────────────────────────────────────────────────────────

    async def _harvest_recording(self, recording_uuid: str, blips: List[Any]) -> None:
        """Pull every requested frame out of one clip."""
        recordings = self._api(RecordingsApi)
        response = await asyncio.to_thread(recordings.get_recording, recording_uuid)
        if response.status != JsonValues.SUCCESS or response.recording is None:
            raise RuntimeError(response.message or "the recording is gone")
        recording = response.recording

        clip_path = recording.file_path
        overlay_path = recording.overlay_path
        if not clip_path:
            raise RuntimeError("the recording has no file on disk")
        if not overlay_path:
            # Without the sidecar there is no geometry to propose. The frames could still
            # be pulled, but a bare image with no suggestion is not what was asked for.
            raise RuntimeError("the clip has no detection sidecar, so nothing can be proposed")

        document = await asyncio.to_thread(_read_json, overlay_path)

        candidates: List[Dict[str, Any]] = []
        for blip in blips:
            if self._cancel_flag.is_set():
                break
            await self._check_pause()
            candidates.extend(await self._harvest_blip(blip, clip_path, document, source_uuid=recording.source_uuid))
            self._done_blips.add(blip.uuid)
            self._update_progress(min(100.0, len(self._done_blips) / max(len(self._blip_uuids), 1) * 100.0))

        if candidates:
            await self._add_candidates(candidates)
        await self._mark_harvested([b.uuid for b in blips if b.uuid in self._done_blips])

    async def _harvest_blip(
        self,
        blip: Any,
        clip_path: str,
        document: Dict[str, Any],
        source_uuid: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """One incident: its frames, uploaded, with a proposal each."""
        label_uuid = str((blip.detail_json or {}).get("modalLabel") or "")
        if not label_uuid:
            self._no_proposal += 1
            self._note_failure(f"blip {blip.uuid} records no label to propose")
            return []

        moments = harvest_times(blip.start_s, blip.end_s, self._frames_per_incident)
        proposals: List[Dict[str, Any]] = []

        for at_s in moments:
            box = proposed_box_at(document, blip.track_id, label_uuid, at_s)
            if box is None:
                # Nothing honest to suggest here — see `proposed_box_at`. The frame is
                # not pulled either: an image with no proposal is just clutter in the
                # review browser.
                self._no_proposal += 1
                continue

            frame = await asyncio.to_thread(_grab_frame, clip_path, at_s)
            if frame is None:
                self._failed += 1
                self._note_failure(f"could not decode {clip_path} at {at_s:.2f}s")
                continue

            uploaded = await self._upload(frame, blip, at_s, source_uuid=source_uuid)
            if uploaded is None:
                continue

            proposals.append(
                {
                    "image_uuid": uploaded.uuid,
                    "proposed_label_uuid": label_uuid,
                    "geometry": box.as_geometry(),
                    "estimated": box.estimated,
                    "blip": blip,
                    "at_s": at_s,
                }
            )
            self._harvested += 1

        return proposals

    async def _upload(self, jpeg: bytes, blip: Any, at_s: float, source_uuid: Optional[str] = None) -> Optional[Any]:
        """Write one frame into the dataset, letting the endpoint dedup it.

        `source_uuid` is the camera the clip was recorded from, carried through so a
        harvested frame is filterable by origin exactly like a collected one.
        """
        images = self._api(ImagesApi)
        name = f"blip_{blip.event_uuid[:8]}_{blip.kind}_t{at_s:.2f}.jpg".replace(".", "-", 1)
        response = await asyncio.to_thread(
            images.upload_images,
            files=[(name, jpeg)],
            path=self._dest_path,
            overwrite=False,
            source_uuid=source_uuid or "",
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
        if result.duplicate_of:
            # Already in the dataset. Counted rather than filed again, which is what
            # makes re-running a harvest close to free.
            self._duplicates += 1
            return None
        await self._tag_harvested(result)
        return result

    async def _tag_harvested(self, uploaded: Any) -> None:
        """Mark the image as harvested, so it stays findable after the run is gone.

        A preview run is a temporary address: applying it or deleting it takes these
        frames out of Previous Model Results, and what is left is an image in `blip_harvest`
        that looks like any other. The tag is the durable answer to "where did the
        frames I harvested go?" — Review can filter on it forever.

        Best-effort on purpose. The frame and its proposal are already filed by this
        point, and failing the whole harvest over a missing tag would throw away the
        work that actually matters. It is logged, not raised.
        """
        from db_client.models.set_image_tag_payload import SetImageTagPayload

        path = getattr(uploaded, "path", None)
        if not path:
            return
        try:
            images = self._api(ImagesApi)
            await asyncio.to_thread(
                images.set_image_tag,
                SetImageTagPayload(image_path=path, tag_uuid=HARVESTED_TAG_UUID, action="add"),
            )
        except Exception as e:
            logger.warning("Could not tag harvested frame %s: %s", path, e)

    async def _add_candidates(self, proposals: List[Dict[str, Any]]) -> None:
        _, PreviewCandidateCreate, AddPreviewCandidatesPayload = _payload_models()
        api = self._api(PreviewRunsApi)
        rows = [
            PreviewCandidateCreate(
                image_uuid=p["image_uuid"],
                # A proposed *new* box: there is no bbox row to point at until the run
                # is applied.
                is_new=True,
                proposed_label_uuid=p["proposed_label_uuid"],
                geometry_json={
                    **p["geometry"],
                    # Carried on the geometry rather than in a column of its own: the
                    # review UI needs it exactly where it draws the box, and nothing
                    # filters or sorts on it.
                    "estimated": p["estimated"],
                    # Provenance, and navigation. Without the timestamp a harvested frame
                    # is stranded the moment it leaves the event it came from.
                    "sourceEventUuid": p["blip"].event_uuid,
                    "sourceBlipUuid": p["blip"].uuid,
                    "sourceBlipKind": p["blip"].kind,
                    "sourceClipTimeS": round(p["at_s"], 3),
                    "sourceTrackId": p["blip"].track_id,
                    # The claim the analyser made, in numbers rather than prose, so the
                    # review card can state *why* this frame is being questioned without
                    # a second request. Reviewing a cropped frame on its own is the
                    # ambiguity that caused the mistake in the first place; "9 of 121
                    # frames disagree" is something you can sanity-check at a glance.
                    "sourceBlipSamples": (p["blip"].detail_json or {}).get("samples"),
                    "sourceTrackSamples": (p["blip"].detail_json or {}).get("trackSamples"),
                    "sourceSawLabel": (p["blip"].detail_json or {}).get("sawLabel"),
                },
                confidence=float(getattr(p["blip"], "severity", 0.0) or 0.0),
                source_model="tracking-blips",
            )
            for p in proposals
        ]
        response = await asyncio.to_thread(
            api.add_preview_candidates,
            self._preview_run_uuid,
            AddPreviewCandidatesPayload(candidates=rows),
        )
        if response.status != JsonValues.SUCCESS:
            raise RuntimeError(response.message or "the preview run refused the proposals")

    async def _mark_harvested(self, uuids: List[str]) -> None:
        if not uuids:
            return
        from db_client.models.set_event_blip_states_request import SetEventBlipStatesRequest

        api = self._api(WatchProfilesApi)
        response = await asyncio.to_thread(
            api.set_event_blip_states, SetEventBlipStatesRequest(uuids=uuids, state="harvested")
        )
        if response.status != JsonValues.SUCCESS:
            # Not fatal: the frames and proposals are already filed, and a blip stuck in
            # `new` shows up again rather than losing anything.
            self._note_failure(response.message or "could not mark the blips harvested")

    # ── reporting ────────────────────────────────────────────────────────────

    def _note_failure(self, message: str) -> None:
        if len(self._sample_errors) < 10:
            self._sample_errors.append(message)
        self.log(message)

    def _result_actions(self) -> List[Dict[str, Any]]:
        """The two ways back to what this run produced.

        Without these the harvest ended with a toast and a preview run whose uuid
        appears nowhere a person can click. The frames were three levels down —
        Review → Previous Model Results → find the right run — which is close enough to
        "lost" that it was reported as lost.

        Shared by the live status card and the completion result so the two cannot
        drift apart (the pattern CollectTrainingImagesTask sets).
        """
        actions: List[Dict[str, Any]] = []
        # getattr, not attribute access: get_status_actions() is reachable while the
        # task is still PENDING, before _init has assigned anything.
        run_uuid = getattr(self, "_preview_run_uuid", None)
        if run_uuid:
            actions.append(
                {
                    "type": "open_preview_run",
                    "label": "Review harvested frames",
                    "payload": {"run_uuid": run_uuid},
                }
            )
        # The second one outlives the first. Once the run is applied or deleted the
        # preview run is gone, and this tag is the only thing still naming these
        # images. Image-scope: a harvested frame has no bbox until the run is applied,
        # so a bbox-tag filter would match nothing.
        actions.append(
            {
                "type": "open_review",
                "label": "Browse all harvested frames",
                "payload": {
                    "tag_uuids": [],
                    "image_tag_uuids": [HARVESTED_TAG_UUID],
                    "bbox_verification": "any",
                },
            }
        )
        return actions

    def get_status_actions(self) -> List[Dict[str, Any]]:
        return self._result_actions()

    def _build_result(self) -> Dict[str, Any]:
        return {
            "previewRunUuid": self._preview_run_uuid,
            "framesHarvested": self._harvested,
            "duplicatesSkipped": self._duplicates,
            "noProposal": self._no_proposal,
            "skippedNotHarvestable": self._skipped_not_harvestable,
            "failed": self._failed,
            "sampleErrors": list(self._sample_errors),
            "actions": self._result_actions(),
        }

    def get_resume_data(self) -> Dict[str, Any]:
        """Checkpointed state. `done_blips` is what makes a resumed run skip the
        incidents it already pulled, rather than uploading them again and leaning on
        dedup to clean up after itself."""
        return {
            "harvested": self._harvested,
            "duplicates": self._duplicates,
            "no_proposal": self._no_proposal,
            "skipped_not_harvestable": self._skipped_not_harvestable,
            "failed": self._failed,
            "sample_errors": list(self._sample_errors),
            "done_blips": list(self._done_blips),
            "preview_run_uuid": self._preview_run_uuid,
        }

    async def _deinit(self) -> None:
        return None


def _read_json(path: str) -> Dict[str, Any]:
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text())


def _grab_frame(clip_path: str, at_s: float) -> Optional[bytes]:
    """One frame of the clip as JPEG bytes, at `at_s` seconds from its start.

    The seek itself is `grab_frame_array` (shared with the blip crop-preview endpoint,
    which crops before encoding); this just encodes the whole frame for a proposal that
    keeps the geometry separate.
    """
    import cv2

    from apps.helpers.recordings.frameGrab import grab_frame_array

    frame = grab_frame_array(clip_path, at_s)
    if frame is None:
        return None
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return buffer.tobytes() if ok else None
