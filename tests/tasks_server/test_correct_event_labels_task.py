"""Correcting a whole selection's labels, from the clip files.

The bulk path exists because the browser's cannot be looped, so what is worth
protecting here is everything the single-event path got for free by having a person
watching it happen:

* **The row correction is not conditional on the frames.** An event whose subject was
  wrong reads wrong in the grid, groups under the wrong identity and answers the wrong
  filter. That half must land for a clip the recorder never wrote, which is the case a
  "harvest, then patch" ordering silently drops.
* **A frame the correction emptied is still filed; one that was always empty is not.**
  A picture of the detector being wrong about an empty room is the negative example a
  false-positive correction exists to produce, and the single-event path files it for
  that reason — the bulk path must not quietly disagree.
* **One bad event must not take the selection with it.** A group card stands for several
  hundred rows, and failing the run on the first swept clip would abandon the rest.
* **Boxes are written Unverified.** A corrected label on a detector's rectangle is still
  a guess about *where*.

The API round-trips are faked; the picking and remapping logic they sit on is covered
directly in `tests/helpers/test_label_correction.py`.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from apps.helpers.db.types import UNVERIFIED_TAG_UUID
from apps.helpers.recordings.labelCorrection import NO_BBOX, WEAK
from apps.tasks_server.tasks.CorrectEventLabelsTask import CorrectEventLabelsTask

_MOCK_MANAGER = MagicMock()
_MOCK_MANAGER.update_task_heartbeat = AsyncMock()

CAT = "label-cat"
DOG = "label-dog"
BOWL = "label-bowl"


def _make_task(**overrides) -> CorrectEventLabelsTask:
    params = {"event_uuids": ["event-1"], "remap": {CAT: DOG}}
    params.update(overrides)
    task = CorrectEventLabelsTask(task_config_uuid="cfg", params=params, manager=_MOCK_MANAGER)
    task.set_instance_uuid("33333333-3333-3333-3333-333333333333")
    return task


def _event(uuid="event-1", **overrides):
    fields = {
        "uuid": uuid,
        "subject_ref": CAT,
        "subject_name": "Clover",
        "target_ref": None,
        "target_name": None,
        "rule_name": "cat at bowl",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _box(label=CAT, conf=0.9, x=0.1, y=0.1, w=0.2, h=0.2):
    return {"labelUuid": label, "conf": conf, "x": x, "y": y, "w": w, "h": h}


def _doc(*frames):
    return {"frames": [{"t": t, "boxes": list(boxes)} for t, boxes in frames]}


def _upload_result(ok=True, skipped=False, duplicate_of=None, path="watch_captures/f.jpg"):
    return SimpleNamespace(ok=ok, skipped=skipped, duplicate_of=duplicate_of, path=path, uuid="img-1", message=None)


async def _init(task):
    await task._init(task._params, None)
    task._label_names = {CAT: "Clover", DOG: "Luna", BOWL: "Bowl"}
    return task


class TestRefusingAnEmptyCorrection:
    async def test_no_events_is_refused(self):
        with pytest.raises(ValueError, match="event_uuids"):
            await _make_task(event_uuids=[])._init({"event_uuids": [], "remap": {CAT: DOG}}, None)

    async def test_no_remap_is_refused(self):
        task = _make_task(remap={})
        with pytest.raises(ValueError, match="remap"):
            await task._init(task._params, None)

    async def test_a_remap_pointing_a_label_at_itself_changes_nothing_and_is_refused(self):
        """A no-op dressed as a correction. Run, it would file frames of a label nobody
        questioned and rewrite rows to what they already said."""
        task = _make_task(remap={CAT: CAT})
        with pytest.raises(ValueError, match="changes nothing"):
            await task._init(task._params, None)

    async def test_a_self_mapping_entry_is_dropped_but_the_rest_of_the_remap_stands(self):
        """The conditional: one dead entry must not cost the correction beside it."""
        task = await _init(_make_task(remap={CAT: CAT, BOWL: DOG}))
        assert task._remap == {BOWL: DOG}

    async def test_a_selection_past_the_sanity_cap_is_refused_before_anything_runs(self):
        task = _make_task(event_uuids=[f"e{i}" for i in range(2001)])
        with pytest.raises(ValueError, match="sanity cap"):
            await task._init(task._params, None)


class TestRefusingToRemapSomethingThatIsNotALabel:
    """A zone-kind rule stores the *zone's* uuid in `target_ref`.

    A remap keyed on one would rewrite that slot from a zone to a label, leaving a row
    pointing at something which is not a place — and a zone uuid matches no sidecar box,
    so the entry could only ever do harm. The dialog filters its rows this way; these
    cover the direct API caller and the label deleted between dialog and run.
    """

    async def test_a_zone_keyed_entry_is_dropped(self):
        task = await _init(_make_task(remap={CAT: DOG, "zone-bowl": DOG}))
        task._drop_entries_that_are_not_labels()
        assert task._remap == {CAT: DOG}

    async def test_a_correction_to_a_deleted_label_is_dropped(self):
        task = await _init(_make_task(remap={CAT: "label-gone"}))
        with pytest.raises(ValueError, match="names a label"):
            task._drop_entries_that_are_not_labels()

    async def test_the_sentinels_survive_the_filter(self):
        """The conditional that stops the guard eating the two values which are
        deliberately not label uuids."""
        task = await _init(_make_task(remap={CAT: NO_BBOX, BOWL: WEAK}))
        task._drop_entries_that_are_not_labels()
        assert task._remap == {CAT: NO_BBOX, BOWL: WEAK}


class TestTheRowCorrection:
    async def test_the_event_row_is_rewritten_with_the_corrected_name(self, monkeypatch):
        """The ref and its name travel together: a row that filters as one label and
        reads as another is the worst of the three states."""
        task = await _init(_make_task())
        sent = []

        async def _to_thread(fn, uuid, payload, *args, **kwargs):
            sent.append((uuid, payload))
            return SimpleNamespace(status="success", message=None)

        monkeypatch.setattr(task, "_api", MagicMock())
        monkeypatch.setattr("asyncio.to_thread", _to_thread)

        await task._correct_event_row(_event())

        [(uuid, payload)] = sent
        assert uuid == "event-1"
        assert payload.subject_ref == DOG
        assert payload.subject_name == "Luna"
        assert task._events_corrected == 1

    async def test_a_row_the_remap_does_not_touch_is_not_patched_at_all(self, monkeypatch):
        """The conditional against the test above. Correcting a label that was never the
        event's own subject is the common case, and a PATCH per event of a 383-event
        group that changes nothing is a request storm for no effect."""
        task = await _init(_make_task(remap={BOWL: DOG}))
        sent = AsyncMock()
        monkeypatch.setattr(task, "_api", MagicMock())
        monkeypatch.setattr("asyncio.to_thread", sent)

        await task._correct_event_row(_event())

        sent.assert_not_awaited()
        assert task._events_corrected == 0

    async def test_a_clip_that_was_never_written_still_gets_its_row_corrected(self, monkeypatch):
        """The load-bearing ordering. A swept or sidecar-less clip has no frames to
        relabel, and the row is the half the user sees immediately — so it must not be
        skipped for the half that cannot run."""
        task = await _init(_make_task())
        patched = []
        monkeypatch.setattr(task, "_correct_event_row", AsyncMock(side_effect=lambda e: patched.append(e.uuid)))
        monkeypatch.setattr(task, "_clip_for_event", AsyncMock(return_value=(None, None, "")))
        monkeypatch.setattr(task, "_api", MagicMock())
        monkeypatch.setattr(
            "asyncio.to_thread",
            AsyncMock(return_value=SimpleNamespace(status="success", event=_event(), message=None)),
        )

        await task._correct_event("event-1")

        assert patched == ["event-1"]
        assert task._no_clip == 1

    async def test_fix_events_off_leaves_every_row_alone(self, monkeypatch):
        """The conditional against the two above: the frames are still harvested, and
        the rows keep saying what they said."""
        task = await _init(_make_task(fix_events=False))
        patch = AsyncMock()
        monkeypatch.setattr(task, "_correct_event_row", patch)
        monkeypatch.setattr(task, "_clip_for_event", AsyncMock(return_value=(None, None, "")))
        monkeypatch.setattr(task, "_api", MagicMock())
        monkeypatch.setattr(
            "asyncio.to_thread",
            AsyncMock(return_value=SimpleNamespace(status="success", event=_event(), message=None)),
        )

        await task._correct_event("event-1")

        patch.assert_not_awaited()

    async def test_a_row_that_will_not_patch_does_not_abort_the_harvest(self, monkeypatch):
        """Reported and swallowed: the frames are independently valuable."""
        task = await _init(_make_task())
        monkeypatch.setattr(task, "_api", MagicMock())
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=RuntimeError("db is down")))

        await task._correct_event_row(_event())

        assert task._events_corrected == 0
        assert any("db is down" in e for e in task._sample_errors)


class TestWhichFramesAreFiled:
    async def test_a_frame_the_correction_emptied_is_still_filed_with_no_boxes(self, monkeypatch):
        """A picture of the detector being wrong about an empty room is the negative
        example a false-positive correction exists to produce — and the single-event path
        files it for exactly this reason (see `applyLabelRemap`). The bulk path must not
        quietly disagree with it."""
        task = await _init(_make_task(remap={CAT: NO_BBOX}))
        monkeypatch.setattr(
            "apps.tasks_server.tasks.CorrectEventLabelsTask._grab_frame", MagicMock(return_value=b"jpeg")
        )
        monkeypatch.setattr(task, "_upload", AsyncMock(return_value=_upload_result()))
        written = AsyncMock()
        monkeypatch.setattr(task, "_write_boxes", written)

        await task._harvest_frame(_event(), _doc((0.0, [_box(CAT)])), "/clip.mp4", 0.0, "src-1")

        assert task._frames_saved == 1
        assert written.await_args.args[2] == [], "the removed box must not come back"
        assert task._boxes_removed == 1

    async def test_a_sample_that_never_held_a_box_is_skipped(self, monkeypatch):
        """The conditional against the test above, and the reason the guard is on the
        *original* boxes rather than the remapped ones: a frame with nothing in it was
        never worth saving, as opposed to one this correction deliberately emptied."""
        task = await _init(_make_task(remap={CAT: NO_BBOX}))
        grabbed = MagicMock()
        monkeypatch.setattr("apps.tasks_server.tasks.CorrectEventLabelsTask._grab_frame", grabbed)
        upload = AsyncMock()
        monkeypatch.setattr(task, "_upload", upload)

        await task._harvest_frame(_event(), _doc((0.0, [])), "/clip.mp4", 0.0, "src-1")

        grabbed.assert_not_called(), "the clip should not even be opened"
        upload.assert_not_awaited()

    async def test_a_frame_keeping_one_box_is_still_filed_when_another_was_removed(self, monkeypatch):
        """The conditional: same remap, one surviving box, opposite outcome. Removing
        the bowl must not cost the frame of the cat beside it."""
        task = await _init(_make_task(remap={BOWL: NO_BBOX, CAT: DOG}))
        monkeypatch.setattr(
            "apps.tasks_server.tasks.CorrectEventLabelsTask._grab_frame", MagicMock(return_value=b"jpeg")
        )
        monkeypatch.setattr(task, "_upload", AsyncMock(return_value=_upload_result()))
        written = AsyncMock()
        monkeypatch.setattr(task, "_write_boxes", written)

        doc = _doc((0.0, [_box(CAT), _box(BOWL, x=0.7, y=0.7)]))
        await task._harvest_frame(_event(), doc, "/clip.mp4", 0.0, "src-1")

        written.assert_awaited_once()
        assert [b["labelUuid"] for b in written.await_args.args[2]] == [DOG]
        assert task._frames_saved == 1

    async def test_a_moment_past_the_end_of_the_clip_is_counted_not_filed(self, monkeypatch):
        """A sidecar describes the event; the mp4 is whatever the recorder wrote. A Ring
        clip is its own fixed length while the watcher's event ran minutes, so the
        sidecar routinely names moments the file does not contain."""
        task = await _init(_make_task())
        monkeypatch.setattr("apps.tasks_server.tasks.CorrectEventLabelsTask._grab_frame", MagicMock(return_value=None))
        upload = AsyncMock()
        monkeypatch.setattr(task, "_upload", upload)

        await task._harvest_frame(_event(), _doc((0.0, [_box(CAT)])), "/clip.mp4", 0.0, "src-1")

        upload.assert_not_awaited()
        assert task._frames_undecodable == 1

    async def test_a_duplicate_frame_writes_no_boxes(self, monkeypatch):
        """With `skip_duplicates` nothing was written, so there is no image to attach
        boxes to — and attaching them to the image that was kept instead would annotate
        someone else's frame."""
        task = await _init(_make_task())
        monkeypatch.setattr(
            "apps.tasks_server.tasks.CorrectEventLabelsTask._grab_frame", MagicMock(return_value=b"jpeg")
        )
        monkeypatch.setattr(task, "_api", MagicMock())
        monkeypatch.setattr(
            "asyncio.to_thread",
            AsyncMock(
                return_value=SimpleNamespace(status="success", results=[_upload_result(skipped=True)], message=None)
            ),
        )
        written = AsyncMock()
        monkeypatch.setattr(task, "_write_boxes", written)

        await task._harvest_frame(_event(), _doc((0.0, [_box(CAT)])), "/clip.mp4", 0.0, "src-1")

        written.assert_not_awaited()
        assert task._duplicates == 1
        assert task._frames_saved == 0

    async def test_a_weak_selection_harvests_only_the_hedged_moments(self, monkeypatch):
        """`WEAK` names the right label, so the frames it was sure about are frames of a
        label it already gets right."""
        task = await _init(_make_task(remap={CAT: WEAK}))
        doc = _doc((0.0, [_box(CAT, conf=0.95)]), (2.0, [_box(CAT, conf=0.2)]))
        assert task._times_for(doc) == [2.0]

    async def test_the_frame_cap_bounds_one_event(self, monkeypatch):
        task = await _init(_make_task(max_frames_per_event=3))
        doc = _doc(*[(float(i), [_box(CAT)]) for i in range(20)])
        assert len(task._times_for(doc)) == 3


class TestTheBoxesThatGetWritten:
    async def test_every_box_is_written_unverified(self, monkeypatch):
        """A corrected label on a model's rectangle is still a guess about *where*, and
        the whole review workflow is built around that tag."""
        task = await _init(_make_task())
        payloads = []

        async def _to_thread(fn, payload, *args, **kwargs):
            payloads.append(payload)

        monkeypatch.setattr(task, "_api", MagicMock())
        monkeypatch.setattr("asyncio.to_thread", _to_thread)

        await task._write_boxes(_upload_result(), _event(), [_box(DOG)], 1.25)

        assert [p.tag_uuids for p in payloads] == [[UNVERIFIED_TAG_UUID]]

    async def test_the_box_records_what_the_correction_replaced(self, monkeypatch):
        """A reviewer meeting the frame later needs to know it was a correction rather
        than a detection the model got right — it is the record of what the model is
        bad at."""
        task = await _init(_make_task())
        payloads = []

        async def _to_thread(fn, payload, *args, **kwargs):
            payloads.append(payload)

        monkeypatch.setattr(task, "_api", MagicMock())
        monkeypatch.setattr("asyncio.to_thread", _to_thread)

        await task._write_boxes(_upload_result(), _event(), [_box(DOG)], 1.25)

        meta = payloads[0].metadata_json
        assert meta["corrected_from"] == "Clover → Luna"
        assert meta["source_event"] == "event-1"
        assert meta["source_clip_offset_s"] == "1.25"

    async def test_the_geometry_is_passed_through_unconverted(self, monkeypatch):
        """Sidecar boxes are already normalised to 0..1, which is the shape the column
        stores. A conversion here would be a second, divergent one."""
        task = await _init(_make_task())
        payloads = []

        async def _to_thread(fn, payload, *args, **kwargs):
            payloads.append(payload)

        monkeypatch.setattr(task, "_api", MagicMock())
        monkeypatch.setattr("asyncio.to_thread", _to_thread)

        await task._write_boxes(_upload_result(), _event(), [_box(DOG, x=0.25, y=0.5, w=0.1, h=0.2)], 0.0)

        p = payloads[0]
        assert (p.x, p.y, p.width, p.height) == (0.25, 0.5, 0.1, 0.2)

    async def test_a_box_that_will_not_write_costs_that_box_not_the_frame(self, monkeypatch):
        """The image is already filed by this point."""
        task = await _init(_make_task())
        monkeypatch.setattr(task, "_api", MagicMock())
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=RuntimeError("refused")))

        await task._write_boxes(_upload_result(), _event(), [_box(DOG), _box(CAT)], 0.0)

        assert task._boxes_written == 0
        assert task._sample_errors


class TestOneBadEventDoesNotTakeTheRest:
    async def test_a_failing_event_is_counted_and_the_run_continues(self, monkeypatch):
        """A group card stands for several hundred rows. Failing on the first swept clip
        would abandon every event behind it."""
        task = await _init(_make_task(event_uuids=["event-1", "event-2"]))
        reached = []

        async def _correct(uuid):
            reached.append(uuid)
            if uuid == "event-1":
                raise RuntimeError("the event is gone")

        monkeypatch.setattr(task, "_correct_event", _correct)
        monkeypatch.setattr(task, "_fetch_label_names", AsyncMock(return_value={}))

        result = await task._run()

        assert reached == ["event-1", "event-2"]
        assert result["failed"] == 1

    async def test_a_resumed_run_skips_the_events_it_already_corrected(self, monkeypatch):
        """Rows are patched and frames uploaded per event, so redoing one means a second
        patch and a dedup pass cleaning up after it."""
        task = _make_task(event_uuids=["event-1", "event-2"])
        await task._init(task._params, {"done_events": ["event-1"]})
        task._label_names = {CAT: "Clover", DOG: "Luna"}
        reached = []
        monkeypatch.setattr(task, "_correct_event", AsyncMock(side_effect=lambda u: reached.append(u)))

        await task._run()

        assert reached == ["event-2"]


class TestWhichRecordingIsRead:
    async def test_the_clip_is_preferred_over_the_timelapse(self, monkeypatch):
        """A timelapse packs minutes into seconds and its sidecar holds only the frames
        it sampled, so a spread over it is a handful of moments minutes apart."""
        task = await _init(_make_task())
        recordings = [
            SimpleNamespace(kind="timelapse", file_path="/t.mp4", overlay_path="/t.json", source_uuid="src-1"),
            SimpleNamespace(kind="clip", file_path="/c.mp4", overlay_path="/c.json", source_uuid="src-1"),
        ]
        monkeypatch.setattr(task, "_api", MagicMock())
        monkeypatch.setattr(
            "asyncio.to_thread",
            AsyncMock(return_value=SimpleNamespace(status="success", recordings=recordings, message=None)),
        )

        clip, overlay, source = await task._clip_for_event("event-1")

        assert (clip, overlay, source) == ("/c.mp4", "/c.json", "src-1")

    async def test_a_clip_with_no_sidecar_is_not_offered(self, monkeypatch):
        """Without per-frame boxes there is nothing to relabel — the event's own subject
        describes one analysed instant and cannot say where in the clip the box was."""
        task = await _init(_make_task())
        recordings = [SimpleNamespace(kind="clip", file_path="/c.mp4", overlay_path=None, source_uuid="src-1")]
        monkeypatch.setattr(task, "_api", MagicMock())
        monkeypatch.setattr(
            "asyncio.to_thread",
            AsyncMock(return_value=SimpleNamespace(status="success", recordings=recordings, message=None)),
        )

        assert await task._clip_for_event("event-1") == (None, None, "")
