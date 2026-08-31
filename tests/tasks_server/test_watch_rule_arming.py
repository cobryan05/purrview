"""The watcher half of Home Assistant rule arming.

The engine's own gate is covered in `tests/helpers/test_watch_engine.py` and the MQTT
side in `tests/helpers/test_mqtt_bridge.py`. What is worth pinning here is the wiring
between them: that the watcher actually applies what the bridge holds, and that a
disarmed rule is *visible in this app* rather than only in HA.

That last part is the whole point of the status surfacing. `enabled` and `armed` are
deliberately separate, so a disarmed rule still shows its checkbox ticked in the Watchers
editor — without the panel below, a rule recording nothing has no explanation anywhere in
the app.
"""

from types import SimpleNamespace

import pytest

from apps.helpers.tracking.tracker import Track
from apps.helpers.watch.engine import WatchEngine
from apps.helpers.watch.geometry import Zone
from apps.helpers.watch.rules import RuleSpec
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask

ZONE = Zone(uuid="zone-1", name="hallway", kind="box", points=[(0.4, 0.4), (0.8, 0.8)])
FRAME = 1000.0


def _spec(uuid="rule-1", name="person in house") -> RuleSpec:
    return RuleSpec(
        uuid=uuid,
        name=name,
        subject_a_kind="label",
        subject_a_refs=["label-person"],
        relation="contains",
        subject_b_kind="zone",
        subject_b_refs=["zone-1"],
        threshold=0.6,
        min_duration_s=1.0,
        rearm_delay_s=5.0,
        max_duration_s=300.0,
    )


def _engine(*specs) -> WatchEngine:
    return WatchEngine(
        rules=list(specs) or [_spec()],
        zones=[ZONE],
        source_uuid="src-1",
        frame_width=FRAME,
        frame_height=FRAME,
    )


def _inside() -> list:
    return [
        Track(track_id=1, bbox=(450, 450, 550, 550), label_uuid="label-person", label_name="person", confidence=0.9)
    ]


def _pipeline(name: str, engine: WatchEngine) -> SimpleNamespace:
    return SimpleNamespace(source_name=name, engine=engine, tracks=[], status_row=lambda now: {"source": name})


def _task(pipelines) -> StreamWatcherTask:
    """A task with only the fields `get_status_data` reads — no DB, no inference."""
    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._pipelines = list(pipelines)
    task._recent_events = []
    task._inference_ms = []
    task._frames_analysed = 0
    task._events_started = 0
    task._events_ended = 0
    task._action_successes = 0
    task._action_failures = 0
    task._mqtt = None
    task._collectors = {}
    task._collector_names = {}
    return task


class TestTheStatusPanelShowsArming:
    def test_an_armed_rule_reads_as_armed_and_a_disarmed_one_does_not(self):
        """Asserted as a pair. A panel that always printed "Armed" would satisfy either
        half on its own, and it is the *disarmed* reading that has to be right — that is
        the state with no other explanation anywhere in the app.
        """
        engine = _engine()
        task = _task([_pipeline("Hall cam", engine)])
        assert task.get_status_data()["rulesTable"] == [
            {"rule": "person in house", "armed": "Armed", "cameras": "Hall cam"}
        ]

        engine.set_rule_armed("rule-1", False, now=0.0)

        assert task.get_status_data()["rulesTable"] == [
            {"rule": "person in house", "armed": "Disarmed", "cameras": "Hall cam"}
        ]

    def test_the_disarmed_count_tracks_the_table(self):
        """The glanceable number and the table must not be able to disagree — a stat
        saying zero over a table showing "Disarmed" is worse than neither.
        """
        engine = _engine(_spec("rule-1", "person in house"), _spec("rule-2", "cat at bowl"))
        task = _task([_pipeline("Hall cam", engine)])
        assert task.get_status_data()["stats"]["rulesDisarmed"] == 0

        engine.set_rule_armed("rule-1", False, now=0.0)

        data = task.get_status_data()
        assert data["stats"]["rulesDisarmed"] == 1
        assert sum(1 for r in data["rulesTable"] if r["armed"] == "Disarmed") == 1

    def test_one_rule_on_two_cameras_is_one_row_naming_both(self):
        """A row per (rule, camera) would repeat one Armed value per source and bury the
        column that actually differs. The cameras belong in a cell."""
        task = _task([_pipeline("Hall cam", _engine()), _pipeline("Kitchen cam", _engine())])

        rows = task.get_status_data()["rulesTable"]

        assert len(rows) == 1
        assert rows[0]["cameras"] == "Hall cam, Kitchen cam"

    def test_a_watcher_with_no_pipelines_reports_nothing_rather_than_failing(self):
        """`get_status_data` is polled in the PENDING window, before `_init` has built
        anything."""
        data = _task([]).get_status_data()

        assert data["rulesTable"] == []
        assert data["stats"]["rulesDisarmed"] == 0


class TestTheWatcherAppliesArming:
    def _armed_task(self, engine, armed: bool):
        task = _task([_pipeline("Hall cam", engine)])
        task._manager = SimpleNamespace(get_mqtt_bridge=lambda: SimpleNamespace(is_rule_armed=lambda _uuid: armed))
        task.logged = []
        task.log = task.logged.append
        task.dispatched = []
        task._resolve_slot_names = lambda signal, pipeline: None
        task._note_event = lambda signal: None

        async def _dispatch(signal, pipeline):
            task.dispatched.append(signal)

        task._dispatch = _dispatch
        return task

    async def test_a_disarm_closes_the_open_event_and_dispatches_it(self):
        """The end-to-end slice: HA holds the rule disarmed, and the event the watcher
        was recording is closed *and* filed rather than dropped on the floor.
        """
        engine = _engine()
        engine.step(_inside(), now=0.0)
        engine.step(_inside(), now=1.5)
        assert len(engine.open_events()) == 1
        task = self._armed_task(engine, armed=False)

        await task._apply_rule_arming(now=2.0)

        assert engine.open_events() == []
        assert len(task.dispatched) == 1, "a closed event must still reach the actions"
        assert task.logged, "closing someone's recording from outside the app should say so"

    async def test_a_steady_armed_state_dispatches_nothing(self):
        """This runs every pass of the watcher loop, so the no-change path has to be
        genuinely silent — not merely harmless."""
        engine = _engine()
        engine.step(_inside(), now=0.0)
        engine.step(_inside(), now=1.5)
        task = self._armed_task(engine, armed=True)

        await task._apply_rule_arming(now=2.0)

        assert task.dispatched == []
        assert len(engine.open_events()) == 1

    async def test_a_bridge_that_is_unavailable_does_not_stop_the_watcher(self):
        """MQTT is optional. A profile running without a broker must not have its loop
        broken by the arming poll."""
        engine = _engine()
        task = self._armed_task(engine, armed=True)

        def _boom():
            raise RuntimeError("no bridge")

        task._manager = SimpleNamespace(get_mqtt_bridge=_boom)

        await task._apply_rule_arming(now=2.0)  # must not raise

        assert task.dispatched == []


class TestArmingIsReachableFromTheApp:
    """Home Assistant must not be the only writer.

    The Watchers editor holds `enabled`, which is config; arming is runtime, so it lives
    on the running watcher's card instead — as buttons, next to the table that reports it.
    """

    def _task_with_bridge(self, engine):
        task = _task([_pipeline("Hall cam", engine)])
        task.armed_calls = []
        bridge = SimpleNamespace(
            is_rule_armed=lambda _uuid: True,
            set_rule_armed=lambda uuid, armed: task.armed_calls.append((uuid, armed)),
        )
        task._manager = SimpleNamespace(get_mqtt_bridge=lambda: bridge)
        return task

    def test_an_armed_rule_offers_to_disarm_it_and_the_reverse(self):
        """The label names the effect, not the state — a button reading "Armed" would be
        ambiguous about whether it reports or acts. Asserted both ways round so it cannot
        pass by hard-coding either verb."""
        engine = _engine()
        task = self._task_with_bridge(engine)

        armed_action = [a for a in task.get_status_actions() if a["command"] == "set_rule_armed"]
        assert [a["label"] for a in armed_action] == ["Disarm 'person in house'"]
        assert armed_action[0]["params"] == {"ruleUuid": "rule-1", "armed": False}

        engine.set_rule_armed("rule-1", False, now=0.0)

        disarmed_action = [a for a in task.get_status_actions() if a["command"] == "set_rule_armed"]
        assert [a["label"] for a in disarmed_action] == ["Arm 'person in house'"]
        assert disarmed_action[0]["params"] == {"ruleUuid": "rule-1", "armed": True}

    def test_a_rule_on_several_cameras_gets_one_button(self):
        engine_a, engine_b = _engine(), _engine()
        task = _task([_pipeline("Hall cam", engine_a), _pipeline("Kitchen cam", engine_b)])

        actions = [a for a in task.get_status_actions() if a.get("command") == "set_rule_armed"]

        assert len(actions) == 1

    async def test_the_command_reaches_the_bridge_rather_than_the_engine(self):
        """`handle_command` runs in a request handler, so it may only record the change —
        the running loop applies it on its next pass. Going straight to the engines here
        would both break that contract and skip the retained publish that moves the
        Home Assistant switch."""
        engine = _engine()
        task = self._task_with_bridge(engine)

        await task.handle_command("set_rule_armed", {"ruleUuid": "rule-1", "armed": False})

        assert task.armed_calls == [("rule-1", False)]

    async def test_an_unknown_command_is_rejected(self):
        task = self._task_with_bridge(_engine())

        with pytest.raises(ValueError):
            await task.handle_command("delete_everything", {})

        assert task.armed_calls == []

    async def test_a_command_without_a_rule_is_rejected(self):
        """The params come off an HTTP request, so a missing uuid must be a clean error
        rather than an arming entry under an empty key that nothing can ever clear."""
        task = self._task_with_bridge(_engine())

        with pytest.raises(ValueError):
            await task.handle_command("set_rule_armed", {"armed": False})

        assert task.armed_calls == []
