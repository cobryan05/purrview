"""Taking one camera out of a running watcher, and putting it back.

The MQTT half — the switch, its retained state, the command topic — is covered in
`tests/helpers/test_mqtt_bridge.py`. What is worth pinning here is what "off" actually
does to a running watcher, because the whole feature is a promise about teardown: the
camera stops being read *and* hands back its stream, so a Ring camera gets its own motion
detection back rather than being read and ignored.

The ordering is the part that cannot be allowed to drift. A clip is cut out of the stream
buffer by the END signal's action, and releasing the lease takes that buffer with it — so
the events have to close before the provider stops, or the recording the user was keeping
is cut from nothing.
"""

import asyncio
from types import SimpleNamespace

from apps.helpers.watch.events import EventCloseReason
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask


def _pipeline(name="Hall cam", *, signals=None) -> SimpleNamespace:
    """A stand-in carrying only what the watching path touches."""
    closed: list = []
    pipeline = SimpleNamespace(
        source_name=name,
        source_uuid="src-1",
        closed_with=closed,
        dispatch_lock=asyncio.Lock(),
        close_open_events=lambda now, reason: (closed.append(reason), list(signals or []))[1],
    )
    return pipeline


def _provider(log: list) -> SimpleNamespace:
    async def _stop():
        log.append("provider.stop")

    async def _start():
        log.append("provider.start")

    return SimpleNamespace(stop=_stop, start=_start)


def _task(pipeline, *, watching: bool, log=None) -> StreamWatcherTask:
    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._pipelines = [pipeline]
    task._source_watching = {}
    task._inflight = {}
    task._manual_open = {}
    task._manual_requests = {}
    task._logs = []
    task.log = task._logs.append
    pipeline.provider = _provider(log if log is not None else [])
    forgotten: list = []
    bridge = SimpleNamespace(
        is_source_watching=lambda _uuid: watching,
        forget_source_presence=forgotten.append,
    )
    task.forgotten_presence = forgotten
    task._manager = SimpleNamespace(get_mqtt_bridge=lambda: bridge)
    return task


class TestTurningACameraOff:
    async def test_it_closes_the_events_before_it_releases_the_stream(self):
        """The ordering the feature depends on. A clip is cut by the END signal's action
        out of the stream buffer, and the buffer goes away with the lease — so a release
        that happened first would leave the user's recording cut from nothing.
        """
        log: list = []
        signal = SimpleNamespace(rule_uuid="rule-1", rule_name="r")
        pipeline = _pipeline(signals=[signal])
        task = _task(pipeline, watching=False, log=log)
        task._resolve_slot_names = lambda *_: None
        task._note_event = lambda *_: None

        async def _dispatch(_signal, _pipeline):
            log.append("dispatch")

        task._dispatch = _dispatch
        task._service_manual_recordings = lambda _p: asyncio.sleep(0)

        await task._apply_source_watching(now=100.0)

        assert log == ["dispatch", "provider.stop"]

    async def test_the_events_are_closed_as_turned_off_not_as_a_fault(self):
        """`forced` would read as the watcher having stopped and `source_lost` as the
        camera having failed. Neither happened, and the reasons exist so a reader is not
        sent looking for a fault that does not exist.
        """
        pipeline = _pipeline(signals=[])
        task = _task(pipeline, watching=False)
        task._service_manual_recordings = lambda _p: asyncio.sleep(0)

        await task._apply_source_watching(now=100.0)

        assert pipeline.closed_with == [EventCloseReason.SOURCE_DISABLED]

    async def test_it_withdraws_the_camera_s_contribution_to_presence(self):
        """A subject's presence is a union across cameras with no per-camera availability
        behind it, so a camera that stops without withdrawing pins whatever it last saw
        as present forever, with nothing left running to contradict it.
        """
        task = _task(_pipeline(signals=[]), watching=False)
        task._service_manual_recordings = lambda _p: asyncio.sleep(0)

        await task._apply_source_watching(now=100.0)

        assert task.forgotten_presence == ["src-1"]

    async def test_a_camera_that_is_off_is_not_serviced_again(self):
        """The reconcile acts on transitions, so a steady state must emit nothing — the
        alternative is tearing the same camera down on every pass for the life of the run.
        """
        log: list = []
        task = _task(_pipeline(signals=[]), watching=False, log=log)
        task._service_manual_recordings = lambda _p: asyncio.sleep(0)

        await task._apply_source_watching(now=100.0)
        await task._apply_source_watching(now=101.0)

        assert log == ["provider.stop"]


class TestTurningACameraBackOn:
    async def test_it_takes_the_stream_again(self):
        log: list = []
        pipeline = _pipeline(signals=[])
        task = _task(pipeline, watching=True, log=log)
        # Already off, so this pass is the transition back on.
        task._source_watching = {"src-1": False}

        await task._apply_source_watching(now=100.0)

        assert log == ["provider.start"]

    async def test_a_camera_nobody_has_an_opinion_about_is_left_running(self):
        """The default that keeps the feature from ever being the reason a camera fails to
        start: an unknown source is watched, so a bridge with no state for it is a no-op
        rather than a teardown.
        """
        log: list = []
        task = _task(_pipeline(signals=[]), watching=True, log=log)

        await task._apply_source_watching(now=100.0)

        assert log == []


class TestItDoesNotTearACameraDownUnderItself:
    async def test_the_source_s_own_read_is_stopped_before_anything_else(self):
        """That task closes events and cuts clips too, so it must be gone before this
        closes the same pipeline's events — and stopping the provider under a read in
        progress would report a torn read as a camera fault, which is exactly what
        turning a camera off must not look like.
        """
        log: list = []
        pipeline = _pipeline(signals=[])
        task = _task(pipeline, watching=False, log=log)
        task._service_manual_recordings = lambda _p: asyncio.sleep(0)

        async def _never_finishes():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                log.append("read.cancelled")
                raise

        task._inflight = {"src-1": asyncio.ensure_future(_never_finishes())}
        # Let it reach its await: a task cancelled before it has ever run never enters
        # the body, so without this the assertion below would be about scheduling.
        await asyncio.sleep(0)

        await task._apply_source_watching(now=100.0)

        assert log == ["read.cancelled", "provider.stop"]
        assert "src-1" not in task._inflight
