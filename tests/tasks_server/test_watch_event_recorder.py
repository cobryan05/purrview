"""The watcher's event history writer.

The behaviours under test are the ones that are invisible when wrong:

* an event that opens and closes must produce **one** row, patched — not two,
* a clip's `recordings` row must carry the clip's own start, which is *earlier* than
  the event's by however much pre-roll the buffer actually had. Storing the event's
  start instead still looks right in a list and silently breaks the review page's
  "the rule fired here" marker,
* card art must never be confused with a `save_snapshot` action's training image,
* nothing here may raise into the frame loop, whatever the DB does.
"""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import apps.tasks_server.tasks._watch_event_recorder as _watch_event_recorder
from apps.helpers.actions.action import ActionContext
from apps.helpers.recordings.detectionTrack import DetectionTrack
from apps.helpers.watch.events import EventCloseReason, EventPhase, WatchEventSignal
from apps.tasks_server.tasks._watch_event_recorder import WatchEventRecorder


class _Det:
    """A stand-in for `apps.helpers.tracking.Detection`, which is read by attribute."""

    def __init__(self, bbox, label_name="cat", label_uuid="lbl-1", confidence=0.9):
        self.bbox = bbox
        self.label_name = label_name
        self.label_uuid = label_uuid
        self.confidence = confidence


PROFILE = "profile-1"
#: The recorder converts monotonic event timestamps to wallclock with this. A round
#: number keeps the expected instants readable.
WALLCLOCK_OFFSET = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc).timestamp()


class _FakeResponse(SimpleNamespace):
    pass


class _FakeApi:
    """Stands in for both generated API classes, recording what it was asked to do."""

    def __init__(self, store):
        self._store = store

    # watch profiles
    def create_watch_event(self, payload):
        if self._store.fail_create:
            return _FakeResponse(status="failure", message="db is down", event=None)
        uuid = f"event-{len(self._store.created) + 1}"
        self._store.created.append(payload)
        return _FakeResponse(status="success", event=SimpleNamespace(uuid=uuid))

    def update_watch_event(self, uuid, payload):
        self._store.updated.append((uuid, payload))
        return _FakeResponse(status="success", event=SimpleNamespace(uuid=uuid))

    # recordings
    def create_recording(self, payload):
        self._store.recordings.append(payload)
        return _FakeResponse(status="success", recording=SimpleNamespace(uuid="recording-1"))

    # tracking blips
    def replace_event_blips(self, payload):
        if self._store.fail_blips:
            return _FakeResponse(status="failure", message="db is down")
        self._store.blips.append(payload)
        return _FakeResponse(status="success", blips=[])


class _Store:
    def __init__(self):
        self.created = []
        self.updated = []
        self.recordings = []
        self.blips = []
        self.fail_create = False
        self.fail_blips = False
        self.logs = []


@pytest.fixture
def store():
    return _Store()


@pytest.fixture
def recorder(store, tmp_path, monkeypatch):
    import apps.tasks_server.tasks._watch_event_recorder as module

    monkeypatch.setattr(module, "WatchProfilesApi", lambda _client: _FakeApi(store))
    monkeypatch.setattr(module, "RecordingsApi", lambda _client: _FakeApi(store))
    # The generated request bodies are plain data holders here; SimpleNamespace keeps
    # the test independent of whether the client has been regenerated.
    monkeypatch.setattr(module, "_payload_models", lambda: (SimpleNamespace, SimpleNamespace, SimpleNamespace))
    monkeypatch.setattr(module, "_blip_payload_models", lambda: (SimpleNamespace, SimpleNamespace))
    return WatchEventRecorder(
        profile_uuid=PROFILE,
        api_client_factory=lambda: None,
        log=store.logs.append,
        snapshot_root=tmp_path / "snapshots",
    )


def _signal(phase=EventPhase.START, *, at=100.0, started_at=100.0, **overrides) -> WatchEventSignal:
    kwargs = dict(
        rule_uuid="rule-1",
        rule_name="cat using litter box",
        phase=phase,
        at=at,
        started_at=started_at,
        source_uuid="src-1",
        track_ids=[7],
        peak_confidence=0.8,
        subject_ref="label-cat",
        subject_name="Tabby",
        target_ref="zone-box",
        target_name="litter box",
    )
    kwargs.update(overrides)
    return WatchEventSignal(**kwargs)


def _context(*, frame=None, detections=()) -> ActionContext:
    return ActionContext(
        profile_uuid=PROFILE,
        profile_name="House cams",
        source_uuid="src-1",
        source_name="Litter cam",
        frame=frame,
        detections=list(detections),
        wallclock_offset=WALLCLOCK_OFFSET,
    )


async def _open(recorder, signal, context) -> str:
    """Hold the opening edge, then file it — what the task does once an action reports it
    wrote something to the event.

    The two steps are deliberately separate in the recorder: an event that writes nothing
    never reaches the second one and never touches the database. Every test below that is
    about a *filed* event goes through here, so the ordering it depends on is the real
    one rather than a shortcut.
    """
    await recorder.record(signal, context)
    return await recorder.ensure_event(signal, context)


class TestDeferredCreation:
    """Nothing is written until an action reports having written something.

    Both halves matter and neither is visible from the row alone. An event that produced
    no media used to leave a card pointing at nothing — the complaint that motivated
    this — and an event that *did* must still carry the frame the rule fired on, not
    whichever frame was current when the row was finally created.
    """

    async def test_the_opening_edge_alone_writes_nothing(self, recorder, store):
        await recorder.record(_signal(EventPhase.START), _context())
        assert store.created == []

    async def test_an_event_that_writes_nothing_leaves_no_row(self, recorder, store):
        """The timelapse-below-minimum case: the action ran, reported success, compiled
        no video, and the event must not appear at all."""
        await recorder.record(_signal(EventPhase.START), _context())
        assert await recorder.record(_signal(EventPhase.END, at=130.0), _context()) is None
        assert store.created == []
        assert store.updated == []

    async def test_filing_twice_creates_one_row(self, recorder, store):
        """An event owning both a clip and a timelapse files twice on one closing edge."""
        signal = _signal(EventPhase.START)
        await recorder.record(signal, _context())
        first = await recorder.ensure_event(signal, _context())
        assert await recorder.ensure_event(signal, _context()) == first
        assert len(store.created) == 1

    async def test_the_card_art_is_the_frame_the_rule_fired_on(self, recorder, store, tmp_path):
        """Encoded at the opening edge, written at filing time. Holding the frame instead
        would decode whatever the camera showed later — some providers reuse the buffer
        in place, so this would be wrong on some sources and right on others."""
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        signal = _signal(EventPhase.START)
        await recorder.record(signal, _context(frame=frame))
        # The frame is overwritten in place after the event opened, exactly as a
        # provider reusing its buffer would.
        frame[:] = 255
        await recorder.ensure_event(signal, _context(frame=frame))

        written = list((tmp_path / "snapshots").rglob("*.jpg"))
        assert len(written) == 1
        decoded = cv2.imdecode(np.frombuffer(written[0].read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded.max() == 0, "the card art shows a frame from after the event opened"

    async def test_an_unfiled_event_writes_no_card_art(self, recorder, tmp_path):
        await recorder.record(_signal(EventPhase.START), _context(frame=np.zeros((120, 160, 3), dtype=np.uint8)))
        await recorder.record(_signal(EventPhase.END, at=130.0), _context())
        assert list((tmp_path / "snapshots").rglob("*.jpg")) == []


class TestEventLifecycle:
    async def test_a_filed_event_is_one_row_patched_at_the_end(self, recorder, store):
        start = _signal(EventPhase.START)
        uuid = await _open(recorder, start, _context())
        assert uuid == "event-1"
        assert len(store.created) == 1

        end = _signal(EventPhase.END, at=130.0, peak_confidence=0.95, close_reason=EventCloseReason.CAP)
        assert await recorder.record(end, _context()) == "event-1"
        # One row, patched — not a second row for the closing edge.
        assert len(store.created) == 1
        patched_uuid, patch = store.updated[-1]
        assert patched_uuid == "event-1"
        assert patch.peak_confidence == pytest.approx(0.95)
        assert patch.truncated is True
        # The *reason* rides along with the flag, so a card can say which of the two
        # cut-short endings this was instead of asserting the cap fired.
        assert patch.close_reason == "cap"

    async def test_update_phase_persists_nothing(self, recorder, store):
        """UPDATE re-fires the `event_started` trigger for actions that want a repeated
        nudge. Persisting it would turn one visit into a row per analysed frame."""
        await _open(recorder, _signal(EventPhase.START), _context())
        await recorder.record(_signal(EventPhase.UPDATE, at=110.0), _context())
        assert len(store.created) == 1
        assert store.updated == []

    async def test_the_row_carries_denormalised_names(self, recorder, store):
        await _open(recorder, _signal(EventPhase.START), _context())
        row = store.created[0]
        assert row.rule_name == "cat using litter box"
        assert (row.subject_ref, row.subject_name) == ("label-cat", "Tabby")
        assert (row.target_ref, row.target_name) == ("zone-box", "litter box")

    async def test_monotonic_event_time_is_stored_as_wallclock(self, recorder, store):
        """Rule timing is monotonic and rows are wallclock; the offset is carried on the
        context so two consumers cannot derive it differently and disagree."""
        await _open(recorder, _signal(EventPhase.START, started_at=100.0), _context())
        assert store.created[0].started_at == datetime.fromtimestamp(WALLCLOCK_OFFSET + 100.0, tz=timezone.utc)

    async def test_closing_an_event_that_never_opened_does_nothing(self, recorder, store):
        """A watcher whose DB was down at the opening edge must not invent a
        zero-length row at the closing one — the gap is more honest."""
        assert await recorder.record(_signal(EventPhase.END), _context()) is None
        assert store.updated == []

    async def test_two_rules_on_one_source_are_separate_events(self, recorder, store):
        await _open(recorder, _signal(EventPhase.START, rule_uuid="rule-1"), _context())
        await _open(recorder, _signal(EventPhase.START, rule_uuid="rule-2"), _context())
        assert len(store.created) == 2

        await recorder.record(_signal(EventPhase.END, rule_uuid="rule-2"), _context())
        assert store.updated[-1][0] == "event-2"

    async def test_a_db_failure_is_swallowed_and_logged(self, recorder, store):
        """A watcher that stops watching because the DB hiccuped is far worse than one
        that loses a row, so this must not raise into the frame loop."""
        store.fail_create = True
        assert await _open(recorder, _signal(EventPhase.START), _context()) is None
        assert recorder.failures == 1
        assert any("not recorded" in line for line in store.logs)


class TestShutdown:
    """An event must not be left claiming to be in progress once the watcher has stopped.

    Nothing else in the system closes one, so a row forgotten here says "in progress"
    for as long as it exists — which is exactly what happened before `close_all` closed
    rather than cleared.
    """

    async def test_an_event_still_open_at_shutdown_is_closed(self, recorder, store):
        await _open(recorder, _signal(EventPhase.START), _context())
        await recorder.close_all()

        assert len(store.updated) == 1
        uuid, patch = store.updated[0]
        assert uuid == "event-1"
        assert patch.ended_at is not None

    async def test_an_unfiled_event_is_dropped_not_closed(self, recorder, store):
        await recorder.record(_signal(EventPhase.START), _context())
        await recorder.close_all()
        assert store.created == []
        assert store.updated == []

    async def test_one_failure_does_not_strand_the_next_event(self, recorder, store, monkeypatch):
        """This runs during shutdown; giving up on the second row because the first threw
        would leave behind exactly what the method exists to prevent."""
        await _open(recorder, _signal(EventPhase.START, rule_uuid="rule-1"), _context())
        await _open(recorder, _signal(EventPhase.START, rule_uuid="rule-2"), _context())

        calls = []

        def _flaky(uuid, payload):
            calls.append(uuid)
            if len(calls) == 1:
                raise RuntimeError("db is down")
            return _FakeResponse(status="success", event=SimpleNamespace(uuid=uuid))

        monkeypatch.setattr(_FakeApi, "update_watch_event", staticmethod(_flaky))
        await recorder.close_all()

        assert calls == ["event-1", "event-2"]
        assert any("left open" in line for line in store.logs)


class TestClipLinking:
    async def test_recording_stores_the_clips_own_start_not_the_events(self, recorder, store):
        """Pre-roll is cut retroactively out of the rewind buffer, so the clip begins
        before the event does. The review page subtracts the two to mark where in the
        video the rule fired; storing the event's start collapses that to zero."""
        start = _signal(EventPhase.START)
        event_uuid = await _open(recorder, start, _context())
        clip_start = datetime.fromtimestamp(WALLCLOCK_OFFSET + 90.0, tz=timezone.utc)

        end = _signal(EventPhase.END, at=130.0)
        await recorder.attach_recording(
            end,
            event_uuid,
            {
                "path": "/data/recordings/clips/a.mp4",
                "startedAt": clip_start.isoformat(),
                "durationS": 45.0,
                "sizeBytes": 1234,
                "partialPreroll": False,
            },
        )

        recording = store.recordings[0]
        assert recording.started_at == clip_start
        assert recording.ended_at == clip_start + timedelta(seconds=45.0)
        # 10s of lead-in before the event's own start at offset 100.
        assert (store.created[0].started_at - recording.started_at).total_seconds() == pytest.approx(10.0)

    async def test_a_clamped_preroll_is_recorded_as_partial(self, recorder, store):
        """The buffer is RAM and ages out; a truncated clip must never be presented as
        though its lead-in were intact."""
        event_uuid = await _open(recorder, _signal(EventPhase.START), _context())
        await recorder.attach_recording(
            _signal(EventPhase.END),
            event_uuid,
            {"path": "/x.mp4", "startedAt": "2026-08-18T12:01:35+00:00", "durationS": 5.0, "partialPreroll": True},
        )
        assert store.recordings[0].partial_preroll is True

    async def test_the_clip_names_the_event_and_the_event_is_not_patched(self, recorder, store):
        """One direction only: `recordings.event_uuid`.

        This used to write the link both ways, so a card could ask "do I have a video"
        of a single column. An event can own a clip *and* a timelapse now, so the
        reverse pointer could not answer it and is gone — patching the event here would
        be writing to a column that no longer exists.
        """
        event_uuid = await _open(recorder, _signal(EventPhase.START), _context())
        await recorder.attach_recording(_signal(EventPhase.END), event_uuid, {"path": "/x.mp4", "durationS": 5.0})
        assert store.recordings[0].event_uuid == event_uuid
        assert store.updated == []

    async def test_an_events_clip_and_timelapse_are_separate_rows(self, recorder, store):
        """Both attach to one event, each keeping its own kind."""
        event_uuid = await _open(recorder, _signal(EventPhase.START), _context())
        await recorder.attach_recording(
            _signal(EventPhase.END), event_uuid, {"path": "/x.mp4", "durationS": 5.0, "kind": "clip"}
        )
        await recorder.attach_recording(
            _signal(EventPhase.END), event_uuid, {"path": "/x-tl.mp4", "durationS": 2.0, "kind": "timelapse"}
        )
        assert [r.kind for r in store.recordings] == ["clip", "timelapse"]
        assert {r.event_uuid for r in store.recordings} == {event_uuid}

    async def test_a_recording_with_no_kind_is_a_clip(self, recorder, store):
        """The kind comes from the action's result, and only media actions produce one.
        Defaulting to `clip` keeps a result that predates the field playable."""
        event_uuid = await _open(recorder, _signal(EventPhase.START), _context())
        await recorder.attach_recording(_signal(EventPhase.END), event_uuid, {"path": "/x.mp4", "durationS": 5.0})
        assert store.recordings[0].kind == "clip"

    async def test_a_timelapses_display_metadata_reaches_the_row(self, recorder, store):
        """The player says "N frames over two hours" off this; without it the duration
        alone claims the visit lasted as long as the video does."""
        event_uuid = await _open(recorder, _signal(EventPhase.START), _context())
        await recorder.attach_recording(
            _signal(EventPhase.END),
            event_uuid,
            {
                "path": "/x-tl.mp4",
                "durationS": 10.0,
                "kind": "timelapse",
                "frames": 120,
                "sourceSpanS": 1200.0,
                "intervalS": 10.0,
                # Not in the kept list — an action's result dict is private and gains
                # keys freely, so the row must not mirror it wholesale.
                "requestedStart": "2026-08-18T12:00:00+00:00",
            },
        )
        metadata = store.recordings[0].metadata_json
        assert metadata == {"frames": 120, "sourceSpanS": 1200.0, "intervalS": 10.0}

    async def test_a_clip_with_no_path_registers_nothing(self, recorder, store):
        event_uuid = await _open(recorder, _signal(EventPhase.START), _context())
        assert await recorder.attach_recording(_signal(EventPhase.END), event_uuid, {"expired": True}) is None
        assert store.recordings == []


class TestTheDetectionSidecar:
    """The clip's overlay: what the detector saw, written beside the mp4 and named on the
    row.

    Naming it on the row is not bookkeeping. The orphan sweep deletes any file under the
    recordings roots that no row points at, so a sidecar written and not recorded is a
    file with about an hour to live — and the bug would present as an overlay that works
    right after an event and mysteriously stops later.
    """

    @pytest.fixture(autouse=True)
    def _no_ingest_lag(self, monkeypatch):
        """Pin the overlay lag, rather than inheriting whatever the container sets.

        `OVERLAY_LAG_S` is read from `WATCH_OVERLAY_LAG_S` at import, so without this the
        sidecar's box times and event window shift by whatever the deployment happens to
        be calibrated to and these assertions pass or fail by environment. It cost a real
        failure: the suite was green, a rebuild put 1.0 into the container, and the event
        window moved to 9.0 s with no code change behind it.
        """
        monkeypatch.setattr(_watch_event_recorder, "OVERLAY_LAG_S", 0.0)

    def _track(self, *, first_at=90.0, count=40):
        """A ring covering the clip's whole window, one sample a second."""
        track = DetectionTrack()
        for i in range(count):
            track.push(WALLCLOCK_OFFSET + first_at + i, [_Det((160, 90, 320, 180))], [], (640, 360))
        return track

    def _clip(self, tmp_path, *, overlay=True, start_offset=90.0, duration=45.0, low_fps=False):
        clip = tmp_path / "a.mp4"
        clip.write_bytes(b"\x00")
        return {
            "path": str(clip),
            "startedAt": datetime.fromtimestamp(WALLCLOCK_OFFSET + start_offset, tz=timezone.utc).isoformat(),
            "durationS": duration,
            "sizeBytes": 1,
            "overlay": overlay,
            # The ring backend's marker — see RingClipBackend.extract.
            "lowFps": low_fps,
        }

    async def _attach(self, recorder, tmp_path, *, track=..., **clip_kw):
        event_uuid = await _open(recorder, _signal(EventPhase.START), _context())
        await recorder.attach_recording(
            _signal(EventPhase.END, at=130.0),
            event_uuid,
            self._clip(tmp_path, **clip_kw),
            context=_context(),
            detection_track=self._track() if track is ... else track,
        )

    async def test_a_ring_cut_clip_is_never_lag_compensated(self, recorder, tmp_path, monkeypatch):
        """A ring clip's frames and its boxes are stamped by the same `arrived_at`, so
        they are aligned by construction. Compensating one of them would *introduce* the
        error the setting exists to remove — and the setting is global, so this exemption
        is the only thing standing between a calibrated deployment and broken ring clips.
        """
        monkeypatch.setattr(_watch_event_recorder, "OVERLAY_LAG_S", 1.0)
        await self._attach(recorder, tmp_path, low_fps=True)

        assert json.loads((tmp_path / "a.mp4.boxes.json").read_text())["lagS"] == 0.0

    async def test_a_buffer_cut_clip_carries_the_configured_lag(self, recorder, tmp_path, monkeypatch):
        """The other side of the branch above: the setting has to actually reach the
        sidecar, and be recorded on it, or a clip written under one calibration is
        indistinguishable from one written under another."""
        monkeypatch.setattr(_watch_event_recorder, "OVERLAY_LAG_S", 1.0)
        await self._attach(recorder, tmp_path, low_fps=False)

        assert json.loads((tmp_path / "a.mp4.boxes.json").read_text())["lagS"] == 1.0

    async def test_the_sidecar_lands_beside_the_clip_and_is_named_on_the_row(self, recorder, store, tmp_path):
        await self._attach(recorder, tmp_path)

        sidecar = tmp_path / "a.mp4.boxes.json"
        assert sidecar.exists()
        assert store.recordings[0].overlay_path == str(sidecar)

    async def test_the_boxes_are_timed_from_the_clips_start_not_the_events(self, recorder, tmp_path):
        """The clip begins 10 s before the event. A sidecar timed off the event would put
        every box 10 s late — late enough to read as the detector lagging."""
        await self._attach(recorder, tmp_path)

        doc = json.loads((tmp_path / "a.mp4.boxes.json").read_text())

        assert doc["frames"][0]["t"] == 0.0
        assert doc["frames"][0]["boxes"][0]["label"] == "cat"

    async def test_the_rules_own_span_is_marked(self, recorder, tmp_path):
        """Pre-roll and post-roll are indistinguishable from the event in the video
        itself, so the window is the only thing that says when the rule was firing."""
        await self._attach(recorder, tmp_path)

        event = json.loads((tmp_path / "a.mp4.boxes.json").read_text())["event"]

        assert event["startS"] == pytest.approx(10.0)
        assert event["endS"] == pytest.approx(40.0)
        assert event["ruleName"] == "cat using litter box"

    async def test_the_action_can_turn_it_off(self, recorder, store, tmp_path):
        """The checkbox on the clip action, all the way through to the disk."""
        await self._attach(recorder, tmp_path, overlay=False)

        assert not (tmp_path / "a.mp4.boxes.json").exists()
        assert store.recordings[0].overlay_path is None

    async def test_no_track_means_no_sidecar(self, recorder, store, tmp_path):
        await self._attach(recorder, tmp_path, track=None)

        assert not (tmp_path / "a.mp4.boxes.json").exists()
        assert store.recordings[0].overlay_path is None

    async def test_a_window_the_watcher_never_analysed_writes_nothing(self, recorder, store, tmp_path):
        """An empty sidecar would claim the detector saw nothing, which is a stronger
        statement than "there is no record" and would read as a broken model."""
        await self._attach(recorder, tmp_path, track=DetectionTrack())

        assert not (tmp_path / "a.mp4.boxes.json").exists()
        assert store.recordings[0].overlay_path is None

    async def test_a_failed_sidecar_still_registers_the_clip(self, recorder, store, tmp_path, monkeypatch):
        """The mp4 is the artifact; the boxes are a commentary on it. Losing the
        commentary must never lose the recording."""
        import apps.helpers.recordings.detectionTrack as dt

        def _boom(*_a, **_kw):
            raise OSError("disk full")

        monkeypatch.setattr(dt, "write_overlay", _boom)

        await self._attach(recorder, tmp_path)

        assert store.recordings[0].overlay_path is None
        assert store.recordings[0].file_path == str(tmp_path / "a.mp4")


class TestCardArt:
    def _frame(self):
        return np.zeros((64, 96, 3), dtype=np.uint8)

    async def test_a_snapshot_is_written_for_every_filed_event(self, recorder, store, tmp_path):
        """Every event that earns a row gets card art, and it is independent of whether a
        `save_snapshot` action exists — that action's output is a training image the user
        curates, and using it here would put boxes into the dataset."""
        await _open(recorder, _signal(EventPhase.START), _context(frame=self._frame()))
        written = list((tmp_path / "snapshots").rglob("*.jpg"))
        assert len(written) == 1
        assert store.created[0].snapshot_path == str(written[0])

    async def test_card_art_lives_outside_the_image_datasets(self, recorder, tmp_path):
        """It is not a training image and must not appear in the annotation browser."""
        await _open(recorder, _signal(EventPhase.START), _context(frame=self._frame()))
        written = list((tmp_path / "snapshots").rglob("*.jpg"))[0]
        assert "image_datasets" not in str(written)

    async def test_an_event_with_no_frame_still_gets_a_row(self, recorder, store):
        """A source that has not yet delivered a frame must not cost us the event."""
        await _open(recorder, _signal(EventPhase.START), _context(frame=None))
        assert len(store.created) == 1
        assert store.created[0].snapshot_path is None

    async def test_two_events_on_the_same_second_do_not_share_a_file(self, recorder, tmp_path):
        """Two rules can open on one frame for one camera; a same-second filename would
        have one event's card show the other's picture."""
        await _open(recorder, _signal(EventPhase.START, rule_uuid="rule-1"), _context(frame=self._frame()))
        await _open(recorder, _signal(EventPhase.START, rule_uuid="rule-2"), _context(frame=self._frame()))
        assert len(list((tmp_path / "snapshots").rglob("*.jpg"))) == 2


class _Trk(_Det):
    """A stand-in for `apps.helpers.tracking.Track`, which is also read by attribute."""

    def __init__(self, bbox, track_id=7, **kwargs):
        super().__init__(bbox, **kwargs)
        self.track_id = track_id


class TestTheTrackingAnalysis:
    """Blips are found and filed at the moment the clip is registered.

    The contract that matters is the same one the rest of this module lives under:
    nothing here may raise into the frame loop. An analysis that fails costs a badge on
    a review card; a watcher that stops recording because of it is a bug, and one that
    stops watching is much worse. So the failure paths are asserted as hard as the
    happy one.
    """

    BOX = (160, 90, 320, 180)

    def _ring(self, *, flip_at=None, count=40):
        """A ring of one well-behaved track, optionally flipping identity mid-way.

        Detection and track agree on every sample, so nothing here reads as a dropout —
        the only anomaly is the one a test asks for.
        """
        track = DetectionTrack()
        for i in range(count):
            label = "lbl-b" if flip_at and flip_at <= i < flip_at + 3 else "lbl-a"
            track.push(
                WALLCLOCK_OFFSET + 90.0 + i,
                [_Det(self.BOX, label_uuid=label)],
                [_Trk(self.BOX, label_uuid=label)],
                (640, 360),
            )
        return track

    def _clip(self, tmp_path):
        clip = tmp_path / "a.mp4"
        clip.write_bytes(b"\x00")
        return {
            "path": str(clip),
            "startedAt": datetime.fromtimestamp(WALLCLOCK_OFFSET + 90.0, tz=timezone.utc).isoformat(),
            "durationS": 45.0,
            "sizeBytes": 1,
            "overlay": True,
        }

    async def _attach(self, recorder, tmp_path, ring):
        event_uuid = await _open(recorder, _signal(EventPhase.START), _context())
        await recorder.attach_recording(
            _signal(EventPhase.END, at=130.0),
            event_uuid,
            self._clip(tmp_path),
            context=_context(),
            detection_track=ring,
        )

    async def test_a_flipping_identity_is_found_and_filed(self, recorder, store, tmp_path):
        await self._attach(recorder, tmp_path, self._ring(flip_at=10))

        assert len(store.blips) == 1
        filed = store.blips[0]
        assert filed.recording_uuid == "recording-1"
        assert [b.kind for b in filed.blips] == ["identity_flip"]
        assert filed.blips[0].harvestable

    async def test_a_clean_clip_files_nothing_at_all(self, recorder, store, tmp_path):
        """The conditional that makes the test above mean something: the same path, the
        same clip, no anomaly — and no request. A badge on every event would be noise."""
        await self._attach(recorder, tmp_path, self._ring())

        assert store.blips == []
        assert len(store.recordings) == 1, "the recording is registered either way"

    async def test_a_missing_sidecar_does_not_cost_us_the_recording(self, recorder, store, tmp_path):
        """No overlay means nothing to analyse. The clip is still the artifact."""
        event_uuid = await _open(recorder, _signal(EventPhase.START), _context())
        clip = self._clip(tmp_path)
        clip["overlay"] = False
        await recorder.attach_recording(
            _signal(EventPhase.END, at=130.0), event_uuid, clip, context=_context(), detection_track=None
        )

        assert store.blips == []
        assert len(store.recordings) == 1

    async def test_an_unreadable_sidecar_is_swallowed(self, recorder, store, tmp_path, monkeypatch):
        """Whatever the disk does, the frame loop must not see it."""
        import apps.tasks_server.tasks._watch_event_recorder as module

        def _boom(*_args, **_kwargs):
            raise OSError("disk gone")

        monkeypatch.setattr(module.Path, "read_text", _boom)
        await self._attach(recorder, tmp_path, self._ring(flip_at=10))

        assert store.blips == []
        assert len(store.recordings) == 1, "the recording still landed"

    async def test_a_database_refusal_is_swallowed(self, recorder, store, tmp_path):
        store.fail_blips = True
        await self._attach(recorder, tmp_path, self._ring(flip_at=10))

        assert store.blips == []
        assert len(store.recordings) == 1
        assert any("Tracking analysis skipped" in line for line in store.logs)


class TestTheAnalysisSwitch:
    """Blip analysis is per watcher, read once at start rather than per clip.

    Both halves are asserted against the *same* clip and the same ring, because "no
    blips were filed" is only meaningful next to a run that files some — otherwise it
    would pass against a recorder whose analysis had quietly stopped working.
    """

    def _recorder(self, store, tmp_path, **kwargs):
        return WatchEventRecorder(
            profile_uuid=PROFILE,
            api_client_factory=lambda: None,
            log=store.logs.append,
            snapshot_root=tmp_path / "snapshots",
            **kwargs,
        )

    async def _run(self, recorder, tmp_path):
        analysis = TestTheTrackingAnalysis()
        await analysis._attach(recorder, tmp_path, analysis._ring(flip_at=10))

    async def test_a_watcher_with_analysis_off_files_nothing(self, store, tmp_path, monkeypatch):
        import apps.tasks_server.tasks._watch_event_recorder as module

        monkeypatch.setattr(module, "WatchProfilesApi", lambda _client: _FakeApi(store))
        monkeypatch.setattr(module, "RecordingsApi", lambda _client: _FakeApi(store))
        monkeypatch.setattr(module, "_payload_models", lambda: (SimpleNamespace, SimpleNamespace, SimpleNamespace))
        monkeypatch.setattr(module, "_blip_payload_models", lambda: (SimpleNamespace, SimpleNamespace))

        await self._run(self._recorder(store, tmp_path, analyse_blips=False), tmp_path)

        assert store.blips == []
        assert len(store.recordings) == 1, "the clip is still recorded either way"

    async def test_the_same_clip_with_analysis_on_does_file(self, recorder, store, tmp_path):
        await self._run(recorder, tmp_path)

        assert len(store.blips) == 1

    async def test_a_watchers_thresholds_reach_the_detectors(self, store, tmp_path, monkeypatch):
        """A flip of three samples is found by default and suppressed by this watcher's
        own tuning — which is the only thing differing between the two runs."""
        import apps.tasks_server.tasks._watch_event_recorder as module

        monkeypatch.setattr(module, "WatchProfilesApi", lambda _client: _FakeApi(store))
        monkeypatch.setattr(module, "RecordingsApi", lambda _client: _FakeApi(store))
        monkeypatch.setattr(module, "_payload_models", lambda: (SimpleNamespace, SimpleNamespace, SimpleNamespace))
        monkeypatch.setattr(module, "_blip_payload_models", lambda: (SimpleNamespace, SimpleNamespace))

        strict = self._recorder(store, tmp_path, blip_thresholds={"min_flip_samples": 5})
        await self._run(strict, tmp_path)

        assert store.blips == []
