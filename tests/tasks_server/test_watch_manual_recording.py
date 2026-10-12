"""Manual recording — the Record button's half of the watch pipeline.

What is worth pinning here is the *conditional* behaviour, not the wiring: that a
recording ends on its own without frames arriving, that an app-started one survives a
pass in which Home Assistant said nothing, and that a stop and a cap are reported
differently. Each of those is a path that produced a wrong clip (or no clip) while this
was being built, and each would break silently.
"""

from types import SimpleNamespace

from apps.helpers.watch.engine import WatchEngine
from apps.helpers.watch.events import EventCloseReason, EventPhase
from apps.helpers.watch.manualRecording import (
    MANUAL_RULE_NAME,
    MANUAL_RULE_UUID,
    MIN_MAX_DURATION_S,
    ManualRecording,
    clamp_max_duration,
)
from apps.tasks_server.tasks._watch_pipeline import WatchPipeline
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask, _ManualClipRow


def _pipeline(source_uuid="src-1"):
    return WatchPipeline(
        source_uuid=source_uuid,
        source_name=f"cam-{source_uuid}",
        provider=SimpleNamespace(),
        tracker=SimpleNamespace(update=lambda dets, now: []),
        engine=WatchEngine(rules=[], zones=[], source_uuid=source_uuid),
    )


def _task(pipelines, *, bridge=None, with_clip_action=True):
    """A watcher with the manual-recording state `_init` would have built.

    `_dispatch` is replaced by a recorder: every test here asks what signals the
    service emitted, and the real dispatch would need the action layer, the event
    recorder and a DB to answer the same question.
    """
    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._pipelines = list(pipelines)
    task._manual_requests = {}
    task._manual_open = {}
    task._bridge_recording_seen = {}
    task._live_requests = set()
    task._release_requests = set()
    task._actions = []
    task._profile = SimpleNamespace(uuid="p1", name="P", default_manual_recording_s=300.0)
    task._manual_clip = (_ManualClipRow(), SimpleNamespace(max_length_s=180.0)) if with_clip_action else None
    task._manager = SimpleNamespace(get_mqtt_bridge=lambda: bridge)
    task.log = lambda *a, **k: None

    task.dispatched = []

    async def _record(signal, pipeline=None):
        task.dispatched.append(signal)

    task._dispatch = _record
    return task


class TestClamp:
    def test_a_request_below_the_floor_is_raised_not_refused(self):
        # An automation asking for five seconds wants a short recording. Refusing it
        # fails somewhere the automation will never look.
        assert clamp_max_duration(5, 300) == MIN_MAX_DURATION_S

    def test_no_request_takes_the_profile_default(self):
        assert clamp_max_duration(None, 120) == 120
        assert clamp_max_duration(0, 120) == 120

    def test_a_request_above_the_default_is_honoured(self):
        # The profile default is a default, not a ceiling — only the floor clamps.
        assert clamp_max_duration(600, 300) == 600


class TestOpeningAndClosing:
    async def test_a_request_opens_an_event_under_the_manual_rule(self):
        pipeline = _pipeline()
        task = _task([pipeline])
        task._manual_requests["src-1"] = 60.0

        await task._service_manual_recordings(pipeline)

        assert [s.phase for s in task.dispatched] == [EventPhase.START]
        assert task.dispatched[0].rule_uuid == MANUAL_RULE_UUID
        assert task.dispatched[0].rule_name == MANUAL_RULE_NAME
        assert "src-1" in task._manual_open

    async def test_a_second_pass_does_not_reopen_the_same_recording(self):
        pipeline = _pipeline()
        task = _task([pipeline])
        task._manual_requests["src-1"] = 60.0

        await task._service_manual_recordings(pipeline)
        await task._service_manual_recordings(pipeline)

        assert [s.phase for s in task.dispatched] == [EventPhase.START]

    async def test_the_cap_closes_a_recording_with_no_frames_arriving(self):
        # The whole reason this is serviced at loop level: a camera that stops
        # delivering frames must still stop recording, and `step()` never runs.
        pipeline = _pipeline()
        task = _task([pipeline])
        task._manual_requests["src-1"] = 60.0
        await task._service_manual_recordings(pipeline)
        # Reach back past the cap rather than sleeping through it.
        task._manual_open["src-1"].started_at -= 61.0

        await task._service_manual_recordings(pipeline)

        assert [s.phase for s in task.dispatched] == [EventPhase.START, EventPhase.END]
        assert task.dispatched[-1].close_reason is EventCloseReason.CAP
        assert "src-1" not in task._manual_open

    async def test_a_hand_stop_is_not_reported_as_capped(self):
        # A truncation claim puts "[capped by max duration]" on a clip the user ended
        # deliberately, which sends the next reader looking for a setting to change.
        pipeline = _pipeline()
        task = _task([pipeline])
        task._manual_requests["src-1"] = 60.0
        await task._service_manual_recordings(pipeline)
        task._manual_open["src-1"].stop_requested = True

        await task._service_manual_recordings(pipeline)

        assert task.dispatched[-1].close_reason is EventCloseReason.PREDICATE

    async def test_closing_hands_the_camera_back(self):
        # A Ring camera held live reports no motion. Leaving the request set would keep
        # it blind until live_hold_s lapsed on its own.
        pipeline = _pipeline()
        task = _task([pipeline])
        task._live_requests.add("src-1")
        task._manual_requests["src-1"] = 60.0
        await task._service_manual_recordings(pipeline)
        task._manual_open["src-1"].stop_requested = True

        await task._service_manual_recordings(pipeline)

        assert "src-1" not in task._live_requests
        assert "src-1" in task._release_requests

    async def test_both_edges_carry_the_same_event_identity(self):
        # instance_id is the event's identity — the recorder's open map keys on it, so a
        # START and END that disagreed would file two half-events.
        pipeline = _pipeline()
        task = _task([pipeline])
        task._manual_requests["src-1"] = 60.0
        await task._service_manual_recordings(pipeline)
        task._manual_open["src-1"].stop_requested = True
        await task._service_manual_recordings(pipeline)

        assert task.dispatched[0].instance_id == task.dispatched[-1].instance_id


class TestHomeAssistantSync:
    """The edge-vs-level distinction in `_apply_bridge_recording`."""

    @staticmethod
    def _bridge(wanted):
        calls = []
        return SimpleNamespace(
            connected=True,
            wants_recording=lambda uuid: wanted.get(uuid, False),
            set_source_recording=lambda uuid, on: calls.append((uuid, on)),
            calls=calls,
        )

    async def test_an_app_started_recording_survives_a_pass_with_no_ha_opinion(self):
        # `wants_recording` defaults False, so a level test reads "HA has never mentioned
        # this camera" as "HA wants it stopped" and cancels every recording started from
        # the app or the service on the very next pass.
        pipeline = _pipeline()
        bridge = self._bridge({})
        task = _task([pipeline], bridge=bridge)
        task._manual_requests["src-1"] = 60.0
        await task._service_manual_recordings(pipeline)

        await task._service_manual_recordings(pipeline)

        assert task._manual_open["src-1"].stop_requested is False
        assert [s.phase for s in task.dispatched] == [EventPhase.START]

    async def test_an_app_started_recording_is_echoed_to_the_switch(self):
        pipeline = _pipeline()
        bridge = self._bridge({})
        task = _task([pipeline], bridge=bridge)
        task._manual_requests["src-1"] = 60.0
        await task._service_manual_recordings(pipeline)

        await task._service_manual_recordings(pipeline)

        assert ("src-1", True) in bridge.calls

    async def test_switching_on_in_ha_starts_a_recording(self):
        pipeline = _pipeline()
        task = _task([pipeline], bridge=self._bridge({"src-1": True}))

        await task._service_manual_recordings(pipeline)

        assert "src-1" in task._manual_open
        assert "src-1" in task._live_requests

    async def test_switching_off_in_ha_stops_the_recording(self):
        pipeline = _pipeline()
        wanted = {"src-1": True}
        bridge = self._bridge(wanted)
        task = _task([pipeline], bridge=bridge)
        await task._service_manual_recordings(pipeline)
        assert "src-1" in task._manual_open

        wanted["src-1"] = False
        await task._service_manual_recordings(pipeline)

        assert task.dispatched[-1].phase is EventPhase.END
        assert "src-1" not in task._manual_open


class TestBridgeCounters:
    def test_a_manual_recording_is_not_published_as_a_rule_transition(self):
        # `publish_rule_transition` counts the event and tracks it as open *before* it
        # checks whether the rule has any topics, so passing the sentinel through would
        # add a non-rule to Home Assistant's cross-camera totals and strand it in
        # `_open_rules` under a uuid no reconcile can clear.
        calls = []
        task = _task([_pipeline()])
        task._manager = SimpleNamespace(
            get_mqtt_bridge=lambda: SimpleNamespace(publish_rule_transition=lambda **kw: calls.append(kw))
        )

        task._publish_bridge_transition(
            SimpleNamespace(source_uuid="src-1", rule_uuid=MANUAL_RULE_UUID, phase=EventPhase.START)
        )

        assert calls == []

    def _dispatch(self, phase, *, subject_ref, started_at=10.0, at=70.0):
        """One transition through the real dispatch, with a bridge that only records.

        Returns the rule-transition calls as well, because
        `_publish_bridge_transition` swallows every exception — without them an
        assertion that nothing was published would pass just as happily for a fake
        that raised on its first attribute.
        """
        subject_calls, rule_calls = [], []
        task = _task([_pipeline()])
        task._wallclock_offset = 0.0
        task._manager = SimpleNamespace(
            get_mqtt_bridge=lambda: SimpleNamespace(
                publish_rule_transition=lambda **kw: rule_calls.append(kw),
                publish_subject_event=lambda **kw: subject_calls.append(kw),
            )
        )

        task._publish_bridge_transition(
            SimpleNamespace(
                source_uuid="src-1",
                rule_uuid="rule-1",
                phase=phase,
                at=at,
                started_at=started_at,
                subject_ref=subject_ref,
                subject_name="Maple",
                target_name=None,
                track_ids=(),
                peak_confidence=0.9,
                duration_s=60.0,
            )
        )
        return task, subject_calls, rule_calls

    def test_a_subject_is_counted_when_its_event_closes_not_when_it_opens(self):
        """`subject_ref` is a running argmax over accumulated label mass, so the opening
        edge carries a provisional guess while the closing one carries what the event was
        decided to be — which is the value `_close_watch_event_counter` stores. Counting
        on the opening edge left the live total above the durable one for good, and a
        reseed then stepped it back down, which `total_increasing` reads as a reset.
        """
        _open_task, on_open, open_rule_calls = self._dispatch(EventPhase.START, subject_ref="provisional")
        _close_task, on_close, close_rule_calls = self._dispatch(EventPhase.END, subject_ref="final")

        assert len(open_rule_calls) == 1 and len(close_rule_calls) == 1
        assert on_open == []
        assert [c["subject_uuid"] for c in on_close] == ["final"]

    def test_the_counted_event_is_stamped_with_its_start_not_its_close(self):
        """The durable row keeps the event's `started_at`, so stamping the close time
        here would make `last seen` walk backwards on the next reseed."""
        task, calls, rule_calls = self._dispatch(EventPhase.END, subject_ref="final", started_at=10.0, at=70.0)

        assert len(rule_calls) == 1
        assert calls[0]["last_seen"] == task._wall_iso(10.0)
        assert calls[0]["last_seen"] != task._wall_iso(70.0)


class TestClipLength:
    def test_the_manual_cap_extends_the_overlay_reach(self):
        # The spool and the detection overlay are bounded by this. A manual recording
        # longer than every rule's clip would otherwise lose its head.
        pipeline = _pipeline()
        task = _task([pipeline])
        assert task._max_clip_length_for("src-1") == 0.0

        task._manual_open["src-1"] = ManualRecording(
            source_uuid="src-1", instance_id="i", started_at=0.0, max_duration_s=600.0
        )

        assert task._max_clip_length_for("src-1") == 600.0


class TestStatus:
    def test_a_watcher_with_no_clip_action_reports_unavailable(self):
        # The button has three states, not two. A watcher that cannot record must say so
        # rather than offering a control that silently does nothing.
        task = _task([_pipeline()], with_clip_action=False)

        assert task._manual_recording_status("src-1")["available"] is False

    def test_an_idle_source_reports_the_profile_default(self):
        task = _task([_pipeline()])
        status = task._manual_recording_status("src-1")

        assert status["active"] is False
        assert status["defaultMaxDurationS"] == 300.0
