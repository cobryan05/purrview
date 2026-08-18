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

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from apps.helpers.actions.action import ActionContext
from apps.helpers.watch.events import EventPhase, WatchEventSignal
from apps.tasks_server.tasks._watch_event_recorder import WatchEventRecorder

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


class _Store:
    def __init__(self):
        self.created = []
        self.updated = []
        self.recordings = []
        self.fail_create = False
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


class TestEventLifecycle:
    async def test_start_writes_one_row_and_end_patches_it(self, recorder, store):
        start = _signal(EventPhase.START)
        uuid = await recorder.record(start, _context())
        assert uuid == "event-1"
        assert len(store.created) == 1

        end = _signal(EventPhase.END, at=130.0, peak_confidence=0.95, truncated=True)
        assert await recorder.record(end, _context()) == "event-1"
        # One row, patched — not a second row for the closing edge.
        assert len(store.created) == 1
        patched_uuid, patch = store.updated[-1]
        assert patched_uuid == "event-1"
        assert patch.peak_confidence == pytest.approx(0.95)
        assert patch.truncated is True

    async def test_update_phase_persists_nothing(self, recorder, store):
        """UPDATE re-fires the `event_started` trigger for actions that want a repeated
        nudge. Persisting it would turn one visit into a row per analysed frame."""
        await recorder.record(_signal(EventPhase.START), _context())
        await recorder.record(_signal(EventPhase.UPDATE, at=110.0), _context())
        assert len(store.created) == 1
        assert store.updated == []

    async def test_the_row_carries_denormalised_names(self, recorder, store):
        await recorder.record(_signal(EventPhase.START), _context())
        row = store.created[0]
        assert row.rule_name == "cat using litter box"
        assert (row.subject_ref, row.subject_name) == ("label-cat", "Tabby")
        assert (row.target_ref, row.target_name) == ("zone-box", "litter box")

    async def test_monotonic_event_time_is_stored_as_wallclock(self, recorder, store):
        """Rule timing is monotonic and rows are wallclock; the offset is carried on the
        context so two consumers cannot derive it differently and disagree."""
        await recorder.record(_signal(EventPhase.START, started_at=100.0), _context())
        assert store.created[0].started_at == datetime.fromtimestamp(WALLCLOCK_OFFSET + 100.0, tz=timezone.utc)

    async def test_closing_an_event_that_never_opened_does_nothing(self, recorder, store):
        """A watcher whose DB was down at the opening edge must not invent a
        zero-length row at the closing one — the gap is more honest."""
        assert await recorder.record(_signal(EventPhase.END), _context()) is None
        assert store.updated == []

    async def test_two_rules_on_one_source_are_separate_events(self, recorder, store):
        await recorder.record(_signal(EventPhase.START, rule_uuid="rule-1"), _context())
        await recorder.record(_signal(EventPhase.START, rule_uuid="rule-2"), _context())
        assert len(store.created) == 2

        await recorder.record(_signal(EventPhase.END, rule_uuid="rule-2"), _context())
        assert store.updated[-1][0] == "event-2"

    async def test_a_db_failure_is_swallowed_and_logged(self, recorder, store):
        """A watcher that stops watching because the DB hiccuped is far worse than one
        that loses a row, so this must not raise into the frame loop."""
        store.fail_create = True
        assert await recorder.record(_signal(EventPhase.START), _context()) is None
        assert recorder.failures == 1
        assert any("not recorded" in line for line in store.logs)


class TestClipLinking:
    async def test_recording_stores_the_clips_own_start_not_the_events(self, recorder, store):
        """Pre-roll is cut retroactively out of the rewind buffer, so the clip begins
        before the event does. The review page subtracts the two to mark where in the
        video the rule fired; storing the event's start collapses that to zero."""
        start = _signal(EventPhase.START)
        event_uuid = await recorder.record(start, _context())
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
        event_uuid = await recorder.record(_signal(EventPhase.START), _context())
        await recorder.attach_recording(
            _signal(EventPhase.END),
            event_uuid,
            {"path": "/x.mp4", "startedAt": "2026-08-18T12:01:35+00:00", "durationS": 5.0, "partialPreroll": True},
        )
        assert store.recordings[0].partial_preroll is True

    async def test_the_event_is_linked_back_to_its_clip(self, recorder, store):
        """`watch_events.recording_uuid` is what every card in the list checks for its
        video badge, so the link has to exist in both directions."""
        event_uuid = await recorder.record(_signal(EventPhase.START), _context())
        await recorder.attach_recording(_signal(EventPhase.END), event_uuid, {"path": "/x.mp4", "durationS": 5.0})
        assert store.updated[-1][0] == event_uuid
        assert store.updated[-1][1].recording_uuid == "recording-1"

    async def test_a_clip_with_no_path_registers_nothing(self, recorder, store):
        event_uuid = await recorder.record(_signal(EventPhase.START), _context())
        assert await recorder.attach_recording(_signal(EventPhase.END), event_uuid, {"expired": True}) is None
        assert store.recordings == []


class TestCardArt:
    def _frame(self):
        return np.zeros((64, 96, 3), dtype=np.uint8)

    async def test_a_snapshot_is_written_for_every_event(self, recorder, store, tmp_path):
        """Unconditional, and independent of whether a `save_snapshot` action exists —
        the review page lists events with pictures whatever the profile is configured
        to do."""
        await recorder.record(_signal(EventPhase.START), _context(frame=self._frame()))
        written = list((tmp_path / "snapshots").rglob("*.jpg"))
        assert len(written) == 1
        assert store.created[0].snapshot_path == str(written[0])

    async def test_card_art_lives_outside_the_image_datasets(self, recorder, tmp_path):
        """It is not a training image and must not appear in the annotation browser."""
        await recorder.record(_signal(EventPhase.START), _context(frame=self._frame()))
        written = list((tmp_path / "snapshots").rglob("*.jpg"))[0]
        assert "image_datasets" not in str(written)

    async def test_an_event_with_no_frame_still_gets_a_row(self, recorder, store):
        """A source that has not yet delivered a frame must not cost us the event."""
        await recorder.record(_signal(EventPhase.START), _context(frame=None))
        assert len(store.created) == 1
        assert store.created[0].snapshot_path is None

    async def test_two_events_on_the_same_second_do_not_share_a_file(self, recorder, tmp_path):
        """Two rules can open on one frame for one camera; a same-second filename would
        have one event's card show the other's picture."""
        await recorder.record(_signal(EventPhase.START, rule_uuid="rule-1"), _context(frame=self._frame()))
        await recorder.record(_signal(EventPhase.START, rule_uuid="rule-2"), _context(frame=self._frame()))
        assert len(list((tmp_path / "snapshots").rglob("*.jpg"))) == 2
