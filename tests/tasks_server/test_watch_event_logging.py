"""Which matches become rows in the Events view, and which only fire their actions.

A rule matching is not by itself worth a `watch_events` row: rules like "a cat is
present" hold for hours a day, and filing every one of them buried the events that had
a clip to watch. The row now follows the artifact — an event is recorded where some
in-scope action writes to it — while everything else about the match (MQTT, the task
log, the live view) is untouched.

The failure this guards against is silence in both directions: an event that stops
being recorded when something *did* write to it looks exactly like a quiet camera, and
one that is recorded anyway looks exactly like the list this replaced.
"""

from types import SimpleNamespace

from apps.helpers.actions.action import Action, ActionContext, ActionResult
from apps.helpers.actions.Registry import action_registry
from apps.helpers.watch.events import EventPhase, WatchEventSignal
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask

SOURCE_A = "source-a"
SOURCE_B = "source-b"
RULE_A = "rule-a"
RULE_B = "rule-b"


class _Silent(Action):
    """An action with a side effect that leaves nothing in the Events view — an MQTT
    publish, in every respect this module cares about."""

    def __init__(self, **params):
        super().__init__(**params)
        self.calls = 0

    async def run(self, signal, context):
        self.calls += 1
        return ActionResult(ok=True, detail="published")


class _Filing(_Silent):
    """A stand-in for `record_clip` — writes to the event, so the event exists."""

    @classmethod
    def logs_event(cls) -> bool:
        return True


def _row(kind, *, trigger="event_started", rules=(), sources=(), enabled=True):
    return SimpleNamespace(
        uuid=f"action-{kind}-{trigger}",
        kind=kind,
        enabled=enabled,
        trigger=trigger,
        order=0,
        rule_uuids_json=list(rules),
        source_uuids_json=list(sources),
        params_json={},
    )


def _signal(phase=EventPhase.START, rule_uuid=RULE_A, source_uuid=SOURCE_A):
    return WatchEventSignal(
        rule_uuid=rule_uuid,
        rule_name="a cat is present",
        phase=phase,
        at=100.0 if phase is EventPhase.START else 130.0,
        started_at=100.0,
        source_uuid=source_uuid,
        track_ids=[1],
    )


class _SpyRecorder:
    """Counts what would have been persisted, without a database."""

    def __init__(self):
        self.recorded = []

    async def record(self, signal, context):
        self.recorded.append(signal.phase)
        return "event-1"

    def event_uuid_for(self, signal):
        return "event-1" if EventPhase.START in self.recorded else None

    async def attach_recording(self, *args, **kwargs):
        return None


def _task(actions, recorder=None):
    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._dry_run = False
    task._actions = list(actions)
    task._recorder = recorder or _SpyRecorder()
    task._detection_tracks = {}
    task._action_successes = 0
    task._action_failures = 0
    task._last_action_failure = {}
    task._last_clip_path = None
    task.log = lambda _msg: None
    task._action_context = lambda signal, pipeline=None: ActionContext(profile_uuid="p", profile_name="p")
    return task


class TestEventIsLogged:
    def test_an_action_with_no_output_does_not_file_the_event(self):
        task = _task([(_row("publish_mqtt"), _Silent())])
        assert task._event_is_logged(_signal()) is False

    def test_a_filing_action_files_it(self):
        task = _task([(_row("record_clip", trigger="event_ended"), _Filing())])
        assert task._event_is_logged(_signal()) is True

    def test_save_event_files_it(self):
        save_event = action_registry["save_event"]()
        task = _task([(_row("save_event"), save_event)])
        assert task._event_is_logged(_signal()) is True

    def test_rule_scope_is_respected(self):
        task = _task([(_row("record_clip", trigger="event_ended", rules=[RULE_B]), _Filing())])
        assert task._event_is_logged(_signal(rule_uuid=RULE_A)) is False
        assert task._event_is_logged(_signal(rule_uuid=RULE_B)) is True

    def test_source_scope_is_respected(self):
        # The camera pointed at the litter box records; the other four watch the same
        # rule and file nothing. Scoping the action now scopes the history with it.
        task = _task([(_row("record_clip", trigger="event_ended", sources=[SOURCE_A]), _Filing())])
        assert task._event_is_logged(_signal(source_uuid=SOURCE_A)) is True
        assert task._event_is_logged(_signal(source_uuid=SOURCE_B)) is False

    def test_a_per_frame_action_files_nothing(self):
        # A `frame_analysed` row is scoped to no rule, so counting it would file every
        # event on the source — the behaviour this replaces, reintroduced sideways.
        task = _task([(_row("save_event", trigger="frame_analysed"), _Filing())])
        assert task._event_is_logged(_signal()) is False


class TestValidatorMirror:
    def test_db_servers_copy_names_the_same_kinds(self):
        """`db_server` cannot import the action package (it would drag cv2 and an MQTT
        client in for one predicate), so it keeps a hand-written copy of the set to warn
        about a rule nothing files. A new logging action that is not added there would
        leave the hint quietly wrong — warning about a rule that is in fact recorded."""
        from apps.db_server.manager import EVENT_LOGGING_ACTION_KINDS

        registered = {kind for kind, cls in action_registry.items() if cls.logs_event()}
        assert registered == set(EVENT_LOGGING_ACTION_KINDS)


class TestDispatch:
    async def test_unfiled_event_still_runs_its_actions(self):
        action = _Silent()
        recorder = _SpyRecorder()
        task = _task([(_row("publish_mqtt"), action)], recorder)

        await task._dispatch(_signal(EventPhase.START))
        await task._dispatch(_signal(EventPhase.END))

        # The whole point: the match is still broadcast, it just leaves no row.
        assert action.calls == 1  # event_started only; this row is not on the end trigger
        assert recorder.recorded == []

    async def test_filed_event_is_opened_and_closed(self):
        recorder = _SpyRecorder()
        task = _task([(_row("save_event"), action_registry["save_event"]())], recorder)

        await task._dispatch(_signal(EventPhase.START))
        await task._dispatch(_signal(EventPhase.END))

        # One row, opened at the start and patched at the end — not two, and not one
        # that never closes.
        assert recorder.recorded == [EventPhase.START, EventPhase.END]
