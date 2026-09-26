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

import logging
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
    """Records what the task asked for, without a database.

    `held` and `filed` are kept apart because that is the distinction under test: the
    opening edge always *holds* an event, and only a write turns it into a row.
    """

    def __init__(self):
        self.held = []
        self.filed = []
        self.closed = []
        self.attached = []

    async def record(self, signal, context):
        if signal.phase is EventPhase.START:
            self.held.append(signal.rule_uuid)
            return None
        if signal.phase is EventPhase.END and self.filed:
            self.closed.append(signal.rule_uuid)
            return "event-1"
        return None

    async def ensure_event(self, signal, context):
        self.filed.append(signal.rule_uuid)
        return "event-1"

    def event_uuid_for(self, signal):
        return "event-1" if self.filed else None

    async def attach_recording(self, signal, event_uuid, clip, **kwargs):
        self.attached.append((event_uuid, clip.get("kind")))
        return "recording-1"


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
    task.log = lambda _msg, _level=logging.INFO: None
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
        assert recorder.filed == []
        # Not even held: no action in scope could have filed it, so there was nothing to
        # capture card art for.
        assert recorder.held == []

    async def test_filed_event_is_opened_and_closed(self):
        recorder = _SpyRecorder()
        task = _task([(_row("save_event"), action_registry["save_event"]())], recorder)

        await task._dispatch(_signal(EventPhase.START))
        await task._dispatch(_signal(EventPhase.END))

        # Held at the start, filed the moment the action reported the write, closed at
        # the end — one row, not two, and not one that never closes.
        assert recorder.held == [RULE_A]
        assert recorder.filed == [RULE_A]
        assert recorder.closed == [RULE_A]


class TestOnlyASuccessfulWriteFiles:
    """A filing action that produced nothing must leave no card behind.

    This is the bug that shipped: `record_event_timelapse` returns **ok** with no file
    when the visit was shorter than its frame minimum, which is a legitimate outcome and
    not an error. Gating on the action's *kind* filed a card anyway, and the card pointed
    at nothing. A clip that could not be cut did the same.
    """

    def _end(self, kind, result):
        """A real action of this kind whose `run` returns a fixed result.

        Real and not a stand-in on purpose: the rule under test is that action's own
        `wrote_to_event`, and a fake would only assert what this test declared.
        """
        action = action_registry[kind]()

        async def _run(signal, context, _result=result):
            return _result

        action.run = _run
        recorder = _SpyRecorder()
        task = _task([(_row(kind, trigger="event_ended"), action)], recorder)
        return recorder, task

    async def test_a_timelapse_below_its_frame_minimum_files_nothing(self):
        # Exactly what the action returns for a visit too short to compile.
        recorder, task = self._end(
            "record_event_timelapse",
            ActionResult(ok=True, detail="only 3 frame(s), below the 10-frame minimum"),
        )
        await task._dispatch(_signal(EventPhase.START))
        await task._dispatch(_signal(EventPhase.END))

        assert recorder.held == [RULE_A]
        assert recorder.filed == []
        assert recorder.attached == []

    async def test_a_clip_that_could_not_be_cut_files_nothing(self):
        recorder, task = self._end("record_clip", ActionResult(ok=False, detail="no relay path"))
        await task._dispatch(_signal(EventPhase.START))
        await task._dispatch(_signal(EventPhase.END))

        assert recorder.filed == []

    async def test_a_timelapse_that_compiled_files_the_event(self):
        recorder, task = self._end(
            "record_event_timelapse",
            ActionResult(ok=True, detail="wrote it", data={"path": "/tmp/a.mp4", "sizeBytes": 10}),
        )
        await task._dispatch(_signal(EventPhase.START))
        await task._dispatch(_signal(EventPhase.END))

        assert recorder.filed == [RULE_A]
        # And the recording is linked to the row that was just created for it.
        assert recorder.attached == [("event-1", "timelapse")]


class _Accumulating(_Filing):
    """A stand-in for `record_event_timelapse`: spends the event filling a buffer, and
    only becomes a certainty once that buffer is big enough to compile."""

    def __init__(self, **params):
        super().__init__(**params)
        self.committed = False

    def holds_open_event(self, signal) -> bool:
        return self.committed


class _Pipeline(SimpleNamespace):
    def open_event_signals(self, now):
        return list(self.signals)


class TestPromotingAnOpenEvent:
    """A long event has to be listed while it is happening, not only once it ends.

    An event timelapse accumulates its video for the whole event, so under the
    "row follows the artifact" rule its row appeared only at the close — and a
    timelapse of a cat asleep for two hours is exactly the event a user wants to see in
    progress. The row is brought forward once the action promises it will produce
    something, and not one frame before that.
    """

    def _task_and_pipeline(self, action, *, rule=RULE_A):
        recorder = _SpyRecorder()
        task = _task([(_row("record_event_timelapse", trigger="event_ended"), action)], recorder)
        pipeline = _Pipeline(signals=[_signal(EventPhase.UPDATE, rule_uuid=rule)])
        task._resolve_slot_names = lambda signal, pipeline: None
        return recorder, task, pipeline

    async def test_an_open_event_is_not_filed_before_the_action_commits(self):
        """The empty-card bug in its mid-event form: a session below its frame minimum
        still compiles nothing, so a row now would point at a video that never arrives."""
        action = _Accumulating()
        recorder, task, pipeline = self._task_and_pipeline(action)

        await task._promote_open_events(pipeline)
        assert recorder.filed == []

    async def test_an_open_event_is_filed_once_the_action_commits(self):
        action = _Accumulating()
        recorder, task, pipeline = self._task_and_pipeline(action)

        action.committed = True
        await task._promote_open_events(pipeline)
        assert recorder.filed == [RULE_A]

    async def test_the_row_is_created_once_however_long_the_event_runs(self):
        """Every analysed frame passes through here, and the promotion is an HTTP call.
        Filing per frame would be a request per frame for the length of the event."""
        action = _Accumulating()
        recorder, task, pipeline = self._task_and_pipeline(action)

        action.committed = True
        for _ in range(5):
            await task._promote_open_events(pipeline)
        assert recorder.filed == [RULE_A]

    async def test_an_action_that_files_nothing_never_promotes(self):
        """`_Silent` is an MQTT publish: it holds no artifact, so an event whose only
        action is one of these has no row at all — open or closed."""
        recorder = _SpyRecorder()
        task = _task([(_row("publish_mqtt"), _Silent())])
        task._recorder = recorder
        task._resolve_slot_names = lambda signal, pipeline: None
        pipeline = _Pipeline(signals=[_signal(EventPhase.UPDATE)])

        await task._promote_open_events(pipeline)
        assert recorder.filed == []
