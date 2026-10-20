"""A camera that goes down has to come back on its own.

The watcher shipped unable to recover from a lost stream, in a way that looked like it
could: `SourceHealth` scheduled retries with exponential backoff, and the retries
happened — but `FfmpegImageProvider` permanently gives up on its own stream after
`_MAX_STREAM_RESTARTS` frameless reads and returns None from then on, and nothing ever
re-armed it. `provider.start()` was called once at task start-up and `provider.stop()`
once at teardown. So the backoff was re-asking a reader that had stopped trying,
forever, at ever-longer intervals, and the source stayed dark for the rest of the run.

What these pin:

  - a failed source gets its connection rebuilt before it is read again;
  - retries never stop, they only slow down (there is no give-up state);
  - past a downtime threshold the source is *reported* failed, without that reporting
    stopping the retries;
  - recovery is noticed and announced.
"""

import asyncio
import time
from types import SimpleNamespace
from typing import Optional

import pytest

from apps.tasks_server.tasks._watch_pipeline import SourceHealth, _format_duration
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask


class _FakeProvider:
    """Stands in for an ImageProvider whose camera can be turned off and on."""

    def __init__(
        self,
        *,
        ready: bool = True,
        connecting: bool = False,
        reconnect_error: Optional[Exception] = None,
    ) -> None:
        self.ready = ready
        self.connecting = connecting
        self.reconnect_error = reconnect_error
        self.reconnect_calls = 0
        self.stop_calls = 0
        self.start_calls = 0
        self.fetch_calls = 0
        self.frames: list = []

    async def reconnect(self) -> None:
        self.reconnect_calls += 1
        if self.reconnect_error is not None:
            raise self.reconnect_error

    def is_ready(self) -> bool:
        return self.ready

    def is_connecting(self) -> bool:
        return self.connecting

    async def getNextImage(self):
        self.fetch_calls += 1
        return self.frames.pop(0) if self.frames else None

    async def stop(self) -> None:
        self.stop_calls += 1

    async def start(self) -> None:
        self.start_calls += 1


def _pipeline(provider=None, name="Front door", uuid="src-1") -> SimpleNamespace:
    pipeline = SimpleNamespace(
        source_uuid=uuid,
        source_name=name,
        provider=provider or _FakeProvider(),
        health=SourceHealth(),
        # None is the real value for a source with no motion rule; reconnect clears the
        # differencer so a gap in the frames is not read as the whole scene moving.
        motion_detector=None,
    )
    # Only what _service_source touches before the first frame arrives.
    pipeline.schedule_next = lambda now: None
    # Sources are serviced concurrently, so the event-dispatch paths take the real
    # pipeline's lock; a stand-in has to carry one too.
    pipeline.dispatch_lock = asyncio.Lock()
    return pipeline


def _task() -> StreamWatcherTask:
    """A watcher with only the state the recovery path touches."""
    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._failed_sources = set()
    task._pipelines = []
    task._inflight = {}
    task._log_lines = []
    task.log = task._log_lines.append
    return task


class TestSourceHealthStates:
    def test_a_fresh_source_is_ok(self):
        health = SourceHealth()
        assert health.state(now=0.0) == "ok"
        assert health.downtime_s(0.0) == 0.0
        assert not health.is_failed(0.0)

    def test_a_source_that_just_dropped_is_reconnecting_not_failed(self):
        """A camera blinking for ten seconds is a normal event, not a fault."""
        health = SourceHealth()
        health.record_failure(100.0, "no frame")

        assert health.state(110.0) == "reconnecting"
        assert not health.is_failed(110.0)

    def test_a_source_down_past_the_threshold_is_failed(self):
        health = SourceHealth()
        health.record_failure(100.0, "no frame")

        later = 100.0 + SourceHealth.FAILED_AFTER_S
        assert health.state(later) == "failed"
        assert health.is_failed(later)

    def test_downtime_is_measured_from_the_first_failure_not_the_last(self):
        """Each retry must not reset the clock, or nothing would ever reach failed."""
        health = SourceHealth()
        health.record_failure(100.0, "no frame")
        health.record_failure(160.0, "no frame")
        health.record_failure(220.0, "no frame")

        assert health.downtime_s(220.0) == pytest.approx(120.0)
        assert health.is_failed(220.0)

    def test_a_recovered_source_forgets_it_was_ever_down(self):
        health = SourceHealth()
        health.record_failure(100.0, "no frame")
        health.record_success(400.0)

        assert health.state(400.0) == "ok"
        assert health.downtime_s(400.0) == 0.0
        assert health.last_error is None


class TestRetriesNeverStop:
    def test_backoff_grows_but_is_capped(self):
        health = SourceHealth()
        now = 0.0
        delays = []
        for _ in range(20):
            health.record_failure(now, "no frame")
            delays.append(health.backoff_s)
            now += health.backoff_s

        assert delays[0] == pytest.approx(SourceHealth.BACKOFF_BASE_S)
        # Growth is asserted over the whole ladder rather than at the second rung: the
        # early rungs are deliberately flat (see TestShortOutagesStayOnAFlatInterval),
        # so "the next one is longer" is only true once the flat window is behind it.
        assert max(delays) > delays[0], "backoff must grow"
        assert delays == sorted(delays), "backoff must never shrink"
        assert max(delays) == pytest.approx(SourceHealth.BACKOFF_MAX_S)

    def test_a_source_down_for_a_day_is_still_being_retried(self):
        """The key property: there is no failure count at which retrying stops.

        A camera unplugged for a day is still expected to come back when it is
        plugged in again, so `next_attempt_at` must always be a finite time away.
        """
        health = SourceHealth()
        now = 0.0
        for _ in range(2000):
            health.record_failure(now, "no frame")
            assert health.next_attempt_at - now <= SourceHealth.BACKOFF_MAX_S
            now = health.next_attempt_at

        assert now > 86400, "the simulation should have covered more than a day"
        assert health.is_failed(now), "and it should still be reported as failed"
        # Still scheduled, still finite — never a give-up sentinel.
        health.record_failure(now, "no frame")
        assert health.next_attempt_at == pytest.approx(now + SourceHealth.BACKOFF_MAX_S)


class TestShortOutagesStayOnAFlatInterval:
    """A drop the camera recovers from in seconds must not be waited out.

    The retry interval is derived from elapsed downtime, so the attempts that land
    while a short outage is still recovering are the ones most likely to find the
    source back. Those have to stay close together: spacing them by a multiple of a
    downtime that has barely started leaves a source unread long after its relay path
    is available again.
    """

    @staticmethod
    def _attempt_times(span_s: float) -> list:
        """When a source that never comes back would be retried, over `span_s`."""
        health = SourceHealth()
        now = 0.0
        times = [now]
        while now < span_s:
            health.record_failure(now, "no frame")
            now = health.next_attempt_at
            times.append(now)
        return times

    def test_every_attempt_inside_the_flat_window_is_one_base_interval_apart(self):
        times = self._attempt_times(SourceHealth.BACKOFF_FLAT_UNTIL_S)
        gaps = [b - a for a, b in zip(times, times[1:])]

        assert gaps, "the source should have been retried at least once"
        assert max(gaps) == pytest.approx(
            SourceHealth.BACKOFF_BASE_S
        ), "no attempt inside the flat window may be spaced out by growth"

    def test_a_camera_back_inside_the_flat_window_is_retried_within_one_interval(self):
        times = self._attempt_times(SourceHealth.BACKOFF_FLAT_UNTIL_S)

        for recovered_at in (1.0, 5.0, 12.0, 29.0):
            waited = next(t for t in times if t >= recovered_at) - recovered_at
            assert waited <= SourceHealth.BACKOFF_BASE_S, (
                f"a source readable again {recovered_at}s in waited a further " f"{waited}s to be retried"
            )

    def test_a_long_outage_still_widens_to_the_cap(self):
        """The flat window must not become a flat ladder.

        A camera that is genuinely gone has to cost less and less, or a watcher over
        four dead cameras spends the whole run reconnecting them.
        """
        times = self._attempt_times(3600.0)
        gaps = [b - a for a, b in zip(times, times[1:])]

        assert gaps[-1] > SourceHealth.BACKOFF_BASE_S, "the interval must widen past the flat window"
        assert gaps[-1] == pytest.approx(SourceHealth.BACKOFF_MAX_S)


class TestReconnectDrive:
    async def test_a_failed_source_is_reconnected_before_it_is_read(self):
        task = _task()
        provider = _FakeProvider(ready=True)
        pipeline = _pipeline(provider)
        pipeline.health.record_failure(100.0, "no frame")

        ok = await task._reconnect_source(pipeline, now=200.0)

        assert ok is True
        assert provider.reconnect_calls == 1

    async def test_a_source_that_has_not_come_back_is_not_read(self):
        """is_ready() is asked instead of blocking on the fetch timeout to find out.

        One dead camera must not cost the healthy ones their frame rate, and the
        watcher services every source from a single loop.
        """
        task = _task()
        provider = _FakeProvider(ready=False)
        pipeline = _pipeline(provider)
        pipeline.health.record_failure(100.0, "no frame")

        ok = await task._reconnect_source(pipeline, now=200.0)

        assert ok is False
        # Not read, but not penalised either: a rebuilt reader is never ready on the pass
        # that rebuilds it, so charging that as a failure widened the backoff for normal
        # behaviour and deferred the look-in that would have found it up.
        assert pipeline.health.consecutive_failures == 1, "a rebuild in progress must not widen the backoff"
        assert pipeline.health.connecting is True
        assert pipeline.health.next_attempt_at == pytest.approx(200.0 + SourceHealth.CONNECT_POLL_S)

    async def test_a_raising_reconnect_is_one_more_failed_attempt_not_a_crash(self):
        task = _task()
        provider = _FakeProvider(reconnect_error=ConnectionError("camera is off"))
        pipeline = _pipeline(provider)
        pipeline.health.record_failure(100.0, "no frame")

        ok = await task._reconnect_source(pipeline, now=200.0)

        assert ok is False
        assert "camera is off" in (pipeline.health.last_error or "")

    async def test_a_hanging_reconnect_is_bounded(self):
        task = _task()

        class _Hangs(_FakeProvider):
            async def reconnect(self):
                self.reconnect_calls += 1
                await asyncio.sleep(3600)

        provider = _Hangs()
        pipeline = _pipeline(provider)
        pipeline.health.record_failure(100.0, "no frame")

        task._RECONNECT_TIMEOUT_S = 0.05
        ok = await asyncio.wait_for(task._reconnect_source(pipeline, now=200.0), timeout=5.0)

        assert ok is False
        assert "timed out" in (pipeline.health.last_error or "")


class TestASilentButReadySourceIsRebuilt:
    """The rebuild must key off "has this source delivered?", not "is the reader up?".

    `is_ready()` is `ffmpeg reported geometry`, and geometry is reported once, at the
    start. A camera that connects and then goes silent keeps answering yes forever, so
    a readiness-only gate skips the rebuild for exactly the failure it exists to fix —
    and the provider's own restart budget is spent by then and is cleared only by
    `reconnect()`. The source stays dark for the rest of the run while every status
    surface reports it as merely "failed, still retrying".
    """

    async def test_a_ready_source_that_returns_no_frames_is_still_reconnected(self):
        task = _task()
        provider = _FakeProvider(ready=True)
        pipeline = _pipeline(provider)
        task._pipelines = [pipeline]

        # First pass: healthy and "ready", so nothing is rebuilt — the fetch is what
        # discovers the silence.
        await task._service_source(pipeline, inference_api=None)
        assert provider.reconnect_calls == 0
        assert not pipeline.health.healthy

        # Second pass: still "ready", but now known to be delivering nothing. This is
        # the one that has to rebuild the reader.
        await task._service_source(pipeline, inference_api=None)

        assert provider.reconnect_calls == 1, "a silent source must be rebuilt even while it reports ready"

    async def test_a_source_down_for_an_hour_keeps_being_rebuilt(self):
        """Not one rebuild and then silence: every scheduled retry must re-arm it."""
        task = _task()
        provider = _FakeProvider(ready=True)
        pipeline = _pipeline(provider)
        task._pipelines = [pipeline]

        for _ in range(12):
            await task._service_source(pipeline, inference_api=None)

        assert provider.reconnect_calls >= 10

    async def test_a_healthy_source_is_never_rebuilt(self):
        """The gate must not tear down a working reader between good frames."""
        task = _task()
        provider = _FakeProvider(ready=True)
        pipeline = _pipeline(provider)
        pipeline.health.record_success(0.0)
        task._pipelines = [pipeline]

        # A healthy, ready source that happens to have no frame queued this instant.
        await task._service_source(pipeline, inference_api=None)

        assert provider.reconnect_calls == 0

    async def test_a_source_still_connecting_is_left_alone_even_when_unhealthy(self):
        """Patience outranks the rebuild: restarting a handshake restarts its clock."""
        task = _task()
        provider = _FakeProvider(ready=False, connecting=True)
        pipeline = _pipeline(provider)
        pipeline.health.record_failure(0.0, "no frame")
        task._pipelines = [pipeline]

        await task._service_source(pipeline, inference_api=None)

        assert provider.reconnect_calls == 0
        assert pipeline.health.connecting is True


class _AsyncGeometry(_FakeProvider):
    """The real provider's timing: a rebuilt reader is not ready on the pass that built it.

    `FFmpegStreamerIn.astart()` returns as soon as ffmpeg is spawned. The banner
    carrying width/height/fps is parsed afterwards, off the stderr pump, so
    `is_ready()` is False for a moment after every rebuild and True by the next pass.
    """

    async def reconnect(self) -> None:
        self.reconnect_calls += 1
        self.ready = False


class TestASourceRecoversWithoutARestart:
    """A source that fails once must be able to come back on its own.

    It could not. `health.healthy` is cleared only by `record_success()`, and that
    needs a frame — so a source with one failure against it reached the rebuild gate on
    every retry, tore down the reader it had built on the previous retry, and never got
    as far as the read that would have cleared the failure. The backoff saturated at
    `BACKOFF_MAX_S` and the source rebuilt on that cycle indefinitely.

    Seen in production as four cameras cycling every 300 s for nine hours, ffmpeg
    reporting `1280x720@25.0fps` cleanly on all 108 attempts per camera, zero frames
    read, nothing recorded, and the run ending only because the task was restarted by
    hand — a fresh `start()` being the one path back to healthy.
    """

    async def test_a_rebuilt_reader_is_read_from_before_it_is_rebuilt_again(self):
        task = _task()
        provider = _AsyncGeometry(ready=True)
        pipeline = _pipeline(provider)
        task._pipelines = [pipeline]

        # One failed read is the whole entry price into the trap.
        await task._service_source(pipeline, inference_api=None)
        assert not pipeline.health.healthy
        assert provider.fetch_calls == 1

        # The rebuild. Geometry has not arrived yet, so there is nothing to read.
        await task._service_source(pipeline, inference_api=None)
        assert provider.reconnect_calls == 1
        assert provider.fetch_calls == 1, "the pass that rebuilds cannot also read"

        # The stderr pump parses the banner between passes, as it does in the field.
        provider.ready = True
        await task._service_source(pipeline, inference_api=None)

        assert provider.reconnect_calls == 1, "the reader it just built must not be torn down unread"
        assert provider.fetch_calls == 2, "the rebuilt reader has to be given its read"

    async def test_a_long_outage_keeps_both_rebuilding_and_reading(self):
        """Retrying means alternating the two, not repeating one of them.

        Rebuild-only was the bug and read-only was the bug before it (a reader that
        has given up answers every read with None). Neither alone recovers anything.
        """
        task = _task()
        provider = _AsyncGeometry(ready=True)
        pipeline = _pipeline(provider)
        task._pipelines = [pipeline]

        for _ in range(20):
            await task._service_source(pipeline, inference_api=None)
            # Geometry always lands between passes; the camera is simply sending nothing.
            provider.ready = True

        assert provider.reconnect_calls >= 8, "a silent source keeps being rebuilt"
        assert provider.fetch_calls >= 8, "and keeps being read — otherwise it can never come back"


class TestFailureReporting:
    def test_becoming_failed_is_logged_once_not_every_retry(self):
        """A source retried every five minutes for a day would otherwise write 288
        identical lines into a ring buffer that holds far fewer."""
        task = _task()
        pipeline = _pipeline()
        task._pipelines = [pipeline]

        now = 0.0
        for _ in range(30):
            task._record_source_failure(pipeline, now, "no frame")
            now += SourceHealth.BACKOFF_MAX_S

        failed_lines = [line for line in task._log_lines if "marking it failed" in line]
        assert len(failed_lines) == 1, f"expected one transition line, got {task._log_lines}"
        assert "Front door" in failed_lines[0]
        assert "still retrying" in failed_lines[0].lower()

    def test_a_brief_blip_is_never_reported_as_failed(self):
        task = _task()
        pipeline = _pipeline()
        task._pipelines = [pipeline]

        task._record_source_failure(pipeline, 0.0, "no frame")
        task._record_source_failure(pipeline, 10.0, "no frame")

        assert task._failed_sources == set()
        assert not any("failed" in line for line in task._log_lines)

    def test_the_task_status_names_the_failed_sources(self):
        task = _task()
        good = _pipeline(name="Kitchen", uuid="src-kitchen")
        bad = _pipeline(name="Front door", uuid="src-door")
        task._pipelines = [good, bad]

        assert task._watching_status_msg() == "Watching"

        task._record_source_failure(bad, 0.0, "no frame")
        task._record_source_failure(bad, SourceHealth.FAILED_AFTER_S, "no frame")

        assert task._watching_status_msg() == "Watching (1 of 2 source(s) failed)"

    def test_every_source_failing_says_so_plainly(self):
        task = _task()
        one = _pipeline(name="Kitchen", uuid="src-kitchen")
        two = _pipeline(name="Front door", uuid="src-door")
        task._pipelines = [one, two]

        for pipeline in (one, two):
            task._record_source_failure(pipeline, 0.0, "no frame")
            task._record_source_failure(pipeline, SourceHealth.FAILED_AFTER_S, "no frame")

        assert "No sources available" in task._watching_status_msg()

    def test_a_recovered_source_clears_the_failed_mark(self):
        task = _task()
        pipeline = _pipeline()
        task._pipelines = [pipeline]
        task._record_source_failure(pipeline, 0.0, "no frame")
        task._record_source_failure(pipeline, SourceHealth.FAILED_AFTER_S, "no frame")
        assert task._failed_sources == {"src-1"}

        # What _service_source does on a successful fetch.
        pipeline.health.record_success(SourceHealth.FAILED_AFTER_S + 10)
        task._failed_sources.discard(pipeline.source_uuid)

        assert task._watching_status_msg() == "Watching"


class TestHealthLabel:
    def test_the_label_leads_with_how_long_not_how_many(self):
        """ "failing (37)" read as "something went wrong 37 times". It meant one
        camera had been away for an hour and a half."""
        pipeline = _pipeline()
        pipeline.health.record_failure(0.0, "no frame")

        from apps.tasks_server.tasks._watch_pipeline import WatchPipeline

        label = WatchPipeline._health_label(pipeline, now=5400.0)

        assert "failed" in label
        assert "1h 30m" in label
        assert "still retrying" in label


class TestFormatDuration:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "0s"),
            (8.4, "8s"),
            (59, "59s"),
            (60, "1m"),
            (245, "4m"),
            (3600, "1h"),
            (5400, "1h 30m"),
        ],
    )
    def test_durations_read_naturally(self, seconds, expected):
        assert _format_duration(seconds) == expected

    def test_a_negative_duration_does_not_render_as_nonsense(self):
        assert _format_duration(-5) == "0s"


class TestConnectingIsNotFailing:
    """A camera that is slow to come up must not be treated as one that is broken.

    An RTSP camera behind a relay can take two or three minutes from cold. Before
    this distinction existed, the watcher saw "no frames", counted a failure, widened
    the backoff, declared the source failed at two minutes, and — worst of all — tore
    the reader down and rebuilt it on every retry, so the three minutes it needed
    never elapsed uninterrupted. A camera slower than the retry interval could
    reconnect forever without ever once connecting.
    """

    def test_connecting_does_not_count_as_a_failure(self):
        health = SourceHealth()

        health.record_connecting(100.0)

        assert health.consecutive_failures == 0, "a connect must not widen the backoff"
        assert health.state(100.0) == "connecting"

    def test_a_long_connect_is_never_reported_failed(self):
        """Coming up can legitimately outlast FAILED_AFTER_S. Reporting a camera as
        failed while it is in the middle of successfully connecting is how a health
        signal teaches people to ignore it."""
        health = SourceHealth()
        health.record_connecting(0.0)

        later = SourceHealth.FAILED_AFTER_S * 3
        health.record_connecting(later)

        assert not health.is_failed(later)
        assert health.state(later) == "connecting"

    def test_connecting_polls_often_rather_than_backing_off(self):
        """A camera that came up two seconds ago must not wait out a failure backoff
        before anyone looks at it."""
        health = SourceHealth()

        health.record_connecting(100.0)

        assert health.next_attempt_at == pytest.approx(100.0 + SourceHealth.CONNECT_POLL_S)
        assert health.next_attempt_at < 100.0 + SourceHealth.BACKOFF_BASE_S

    def test_a_real_failure_supersedes_a_connect(self):
        health = SourceHealth()
        health.record_connecting(0.0)

        health.record_failure(10.0, "stream gone")

        assert health.connecting is False
        assert health.consecutive_failures == 1
        assert health.state(10.0) == "reconnecting"

    def test_a_connect_that_never_completes_eventually_reports_failed(self):
        """Patience is bounded. Once the provider stops calling itself connecting, the
        ordinary failure path takes over and the downtime already accrued applies."""
        health = SourceHealth()
        health.record_connecting(0.0)
        health.record_failure(SourceHealth.FAILED_AFTER_S + 1, "never came up")

        assert health.is_failed(SourceHealth.FAILED_AFTER_S + 1)

    def test_a_successful_frame_clears_the_connecting_state(self):
        health = SourceHealth()
        health.record_connecting(0.0)

        health.record_success(5.0)

        assert health.connecting is False
        assert health.state(5.0) == "ok"

    def test_the_label_says_connecting_not_failing(self):
        from apps.tasks_server.tasks._watch_pipeline import WatchPipeline

        pipeline = _pipeline()
        pipeline.health.record_connecting(0.0)

        label = WatchPipeline._health_label(pipeline, now=90.0)

        assert "connecting" in label
        assert "failed" not in label


class TestALostSourceDoesNotHoldItsEventsOpen:
    """An unbounded event is a legitimate setting; an event nothing can ever close is not.

    `max_duration_s = 0` means "no cap", and a cat asleep in a window for four hours is
    exactly what an event timelapse exists to record. But the cap is evaluated inside
    `RuleMachine.step()`, and `step()` only runs when a frame has been analysed — so it
    is a ceiling only for as long as frames keep arriving, and it can never fire for the
    one failure that most needs it. When the camera stops delivering, every event it had
    open stays open: the row sits "in progress" until something restarts the watcher,
    and the timelapse buffering behind it neither compiles nor releases its frames.

    Closing is driven off `SourceHealth.is_failed`, which is deliberately not the first
    dropped frame — a camera blinking for ten seconds is an ordinary event, and a source
    still working through a slow connect is excluded outright.
    """

    def _armed_task(self):
        """A watcher holding one open event on one source, ready to lose it."""
        from apps.helpers.tracking.tracker import Track
        from apps.helpers.watch.engine import WatchEngine
        from apps.helpers.watch.rules import RuleSpec

        spec = RuleSpec(
            uuid="rule-1",
            name="Cat Present",
            subject_a_kind="label",
            subject_a_refs=["label-cat"],
            relation="present",
            subject_b_kind="none",
            subject_b_refs=[],
            threshold=0.6,
            min_duration_s=0.0,
            rearm_delay_s=10.0,
            max_duration_s=0.0,  # the unbounded case this is all about
        )
        engine = WatchEngine([spec], source_uuid="src-1")
        cat = Track(track_id=1, bbox=(0, 0, 20, 20), label_uuid="label-cat", label_name="cat", confidence=0.9)
        engine.step([cat], now=0.0)
        engine.step([cat], now=0.5)
        assert engine.open_signals(0.5), "the event should be open before the camera is lost"

        task = _task()
        pipeline = _pipeline()
        pipeline.engine = engine
        pipeline.close_open_events = lambda now, reason=None: engine.close_all(now, reason=reason)
        task._pipelines = [pipeline]

        dispatched = []
        task._resolve_slot_names = lambda signal, pipeline=None: None
        task._note_event = lambda signal: None

        async def _dispatch(signal, pipeline=None):
            dispatched.append(signal)

        task._dispatch = _dispatch
        return task, pipeline, engine, dispatched

    async def test_a_blink_does_not_close_anything(self):
        """The negative half, and the reason this hangs off `is_failed` at all."""
        task, pipeline, engine, dispatched = self._armed_task()
        pipeline.health.record_failure(100.0, "no frame")

        await task._close_events_for_lost_sources(110.0)

        assert dispatched == []
        assert engine.open_signals(110.0), "a ten-second drop must not end the event"

    async def test_a_source_down_past_the_threshold_closes_its_events(self):
        from apps.helpers.watch.events import EventCloseReason, EventPhase

        task, pipeline, engine, dispatched = self._armed_task()
        pipeline.health.record_failure(100.0, "no frame")

        await task._close_events_for_lost_sources(100.0 + SourceHealth.FAILED_AFTER_S)

        assert len(dispatched) == 1
        assert dispatched[0].phase is EventPhase.END
        assert dispatched[0].close_reason is EventCloseReason.SOURCE_LOST
        assert dispatched[0].truncated, "the subject may well still have been there"
        assert not engine.open_signals(1000.0), "the event must actually be closed"

    async def test_a_source_still_connecting_keeps_its_events(self):
        """Coming up slowly is not the same as being gone — see `record_connecting`."""
        task, pipeline, engine, dispatched = self._armed_task()
        pipeline.health.record_connecting(100.0)

        await task._close_events_for_lost_sources(100.0 + SourceHealth.FAILED_AFTER_S * 3)

        assert dispatched == []
        assert engine.open_signals(1000.0)

    async def test_it_does_not_close_the_same_event_twice(self):
        """The loop calls this every pass; a second close would double every artifact."""
        task, pipeline, engine, dispatched = self._armed_task()
        pipeline.health.record_failure(100.0, "no frame")
        failed_at = 100.0 + SourceHealth.FAILED_AFTER_S

        await task._close_events_for_lost_sources(failed_at)
        await task._close_events_for_lost_sources(failed_at + 60.0)

        assert len(dispatched) == 1

    async def test_a_healthy_source_is_left_alone(self):
        task, pipeline, engine, dispatched = self._armed_task()

        await task._close_events_for_lost_sources(500.0)

        assert dispatched == []
        assert engine.open_signals(500.0)


class TestASlowSourceDoesNotStallTheOthers:
    """The watcher services every camera from one loop, and it used to `await` each
    source in turn.

    That made the whole watcher run at the pace of its slowest camera. It is not a
    theoretical cost: an idle Ring source blocks until its next cloud snapshot, tens
    of seconds away, and on a live profile with three of them the four RTSP cameras
    configured at 1 fps were getting one frame every 30 s — about 3% of their
    configured rate. Nothing reported a fault, because a source waiting for a frame
    is not an error; every one of the seven sources read `health: ok` throughout.

    So the property worth pinning is not "sources are read concurrently" (an
    implementation detail) but the thing that failed: a blocked source must not
    prevent another source from being read and finishing.
    """

    async def test_a_fast_source_finishes_while_a_slow_one_is_still_blocked(self):
        task = _task()
        slow_started = asyncio.Event()
        release_slow = asyncio.Event()
        finished: list = []

        async def _service(pipeline, inference_api):
            if pipeline.source_uuid == "slow":
                slow_started.set()
                await release_slow.wait()
            finished.append(pipeline.source_uuid)

        task._service_source = _service

        task._dispatch_source(_pipeline(uuid="slow"), inference_api=None)
        await slow_started.wait()
        task._dispatch_source(_pipeline(uuid="fast"), inference_api=None)
        # Hand control back to the loop without releasing the slow source.
        for _ in range(4):
            await asyncio.sleep(0)

        assert finished == ["fast"], "a blocked source still gates the others"
        assert "slow" in task._inflight, "the blocked source should still be in flight"

        release_slow.set()
        for _ in range(4):
            await asyncio.sleep(0)
        assert finished == ["fast", "slow"]
        # Retired on completion, or the loop would never look at them again.
        assert task._inflight == {}

    async def test_drain_cancels_reads_still_in_flight(self):
        """Shutdown force-closes every open event, so a source read must not still be
        stepping trackers and dispatching events while that happens."""
        task = _task()
        started = asyncio.Event()
        cancelled = []

        async def _service(pipeline, inference_api):
            started.set()
            try:
                await asyncio.Event().wait()  # never completes on its own
            except asyncio.CancelledError:
                cancelled.append(pipeline.source_uuid)
                raise

        task._service_source = _service
        task._dispatch_source(_pipeline(uuid="stuck"), inference_api=None)
        await started.wait()

        await task._drain_inflight()

        assert cancelled == ["stuck"]
        assert task._inflight == {}


class TestFetchBudget:
    """How long one source is given to produce a frame.

    A flat 30 s is right for a continuous stream and wrong for anything slower. An
    idle Ring camera publishes a cloud snapshot on the bridge's own interval (~30 s),
    so a flat budget races the cadence: whether a perfectly healthy camera is recorded
    as a success or a failure comes down to which lands first. Providers answer this
    themselves via `next_frame_expected_by()`.
    """

    def test_a_provider_with_no_opinion_keeps_the_flat_timeout(self):
        task = _task()
        pipeline = _pipeline()  # _FakeProvider has no next_frame_expected_by
        assert task._fetch_budget_s(pipeline) == StreamWatcherTask._FETCH_TIMEOUT_S

    def test_a_slow_provider_is_given_until_its_own_deadline(self):
        task = _task()
        pipeline = _pipeline()
        far = StreamWatcherTask._FETCH_TIMEOUT_S * 4
        pipeline.provider.next_frame_expected_by = lambda: time.monotonic() + far
        budget = task._fetch_budget_s(pipeline)
        assert budget > StreamWatcherTask._FETCH_TIMEOUT_S
        assert budget == pytest.approx(far, abs=1.0)

    def test_a_deadline_already_passed_never_shortens_the_budget(self):
        """Only ever more patient than the flat timeout, never less — otherwise a
        provider reporting an overdue frame would get a shorter fuse than every
        source has always had."""
        task = _task()
        pipeline = _pipeline()
        pipeline.provider.next_frame_expected_by = lambda: time.monotonic() - 500.0
        assert task._fetch_budget_s(pipeline) == StreamWatcherTask._FETCH_TIMEOUT_S


class TestBackoffTracksDowntimeNotAttemptCount:
    """The retry interval is bounded by how long the source has actually been away.

    A reconnect against a relay path that is momentarily not-ready fails in
    milliseconds, so an attempt-counted ladder reaches its ceiling while the camera
    has been gone for under a second — and then ignores it for the ceiling's length.
    """

    def test_instant_failures_do_not_push_the_next_attempt_far_out(self):
        health = SourceHealth()
        now = 0.0
        for _ in range(12):
            health.record_failure(now, "no bytes returned")
            now += 0.05

        # Down for 0.6s: the next look-in must be seconds away, not minutes.
        assert health.next_attempt_at - now <= SourceHealth.BACKOFF_BASE_S

    def test_the_wait_never_greatly_exceeds_the_downtime_so_far(self):
        health = SourceHealth()
        now = 0.0
        # Fail far faster than the backoff asks, as a failing reconnect does.
        for _ in range(400):
            health.record_failure(now, "no frame")
            downtime = health.downtime_s(now)
            wait = health.next_attempt_at - now
            assert wait <= max(
                SourceHealth.BACKOFF_BASE_S, downtime * 2 + 1e-6
            ), f"after {downtime:.2f}s down the next attempt is {wait:.2f}s away"
            now += 0.25

    def test_a_source_back_after_a_brief_blip_is_retried_within_seconds(self):
        """The laundry camera's real shape: drops, and the relay has it back in ~5s."""
        health = SourceHealth()
        health.record_failure(0.0, "read timed out")
        health.record_failure(0.3, "no bytes returned")

        assert health.next_attempt_at <= SourceHealth.BACKOFF_BASE_S + 0.3


class TestADeadReaderIsNotAConnectForever:
    """A rebuild whose reader dies without geometry has to reach the failure ladder.

    `_reconnect_source` reported every not-yet-ready rebuild as a connect in progress,
    which is right for a camera that is slow to come up and wrong for one that is not
    there at all. An unreachable source's reader is spawned, finds no stream on the
    relay path and stops at once — so the next pass found it neither ready nor
    connecting, rebuilt it, and reported a connect again. `connecting` resets
    `next_attempt_at` to `CONNECT_POLL_S`, so the backoff was never entered and the
    source was respawned every two seconds for as long as the run lasted.

    Observed on a live install as one unplugged camera spawning 29 ffmpeg processes a
    minute for days, each dying in under a millisecond.
    """

    async def test_a_reader_that_dies_without_geometry_is_charged_as_a_failure(self):
        task = _task()
        # Never ready, never connecting: the shape of a reader that exits on spawn.
        provider = _FakeProvider(ready=False, connecting=False)
        pipeline = _pipeline(provider)
        task._pipelines = [pipeline]

        # The rebuild, reported as a connect because geometry legitimately has not
        # arrived yet on the pass that does it.
        await task._service_source(pipeline, inference_api=None)
        assert provider.reconnect_calls == 1
        assert pipeline.health.connecting is True
        assert pipeline.health.consecutive_failures == 0

        # The reader built above is neither ready nor connecting, so it has stopped.
        await task._service_source(pipeline, inference_api=None)

        assert pipeline.health.connecting is False, "a stopped reader is not a connect in progress"
        assert pipeline.health.consecutive_failures == 1

    async def test_the_next_attempt_is_a_backoff_not_a_connect_poll(self):
        """The whole point: the wait between respawns has to grow."""
        task = _task()
        pipeline = _pipeline(_FakeProvider(ready=False, connecting=False))
        task._pipelines = [pipeline]

        await task._service_source(pipeline, inference_api=None)
        connect_poll_until = pipeline.health.next_attempt_at
        await task._service_source(pipeline, inference_api=None)

        assert pipeline.health.next_attempt_at > connect_poll_until
        assert pipeline.health.backoff_s >= SourceHealth.BACKOFF_BASE_S
        assert pipeline.health.backoff_s > SourceHealth.CONNECT_POLL_S

    async def test_an_unreachable_source_is_not_rebuilt_on_every_pass(self):
        """Charging the dead attempt is what stops the respawn loop.

        The charge and the rebuild alternate, so a source that never comes up costs
        half as many spawns as passes — and in the running watcher the backoff the
        charge sets is what keeps those passes minutes apart (`WatchPipeline.due`).
        """
        task = _task()
        provider = _FakeProvider(ready=False, connecting=False)
        pipeline = _pipeline(provider)
        task._pipelines = [pipeline]

        for _ in range(10):
            await task._service_source(pipeline, inference_api=None)

        assert provider.reconnect_calls <= 5, "every pass rebuilt the reader — the backoff is not being entered"

    async def test_a_source_that_is_merely_slow_is_still_left_alone(self):
        """The charge must not reach the case the connecting state exists for.

        A camera behind a relay can take minutes from cold, and its reader is up and
        handshaking the whole time. Charging that is the regression this guards: it
        widens the backoff for doing exactly what coming up looks like.
        """
        task = _task()
        provider = _FakeProvider(ready=False, connecting=True)
        pipeline = _pipeline(provider)
        pipeline.health.record_connecting(time.monotonic())
        task._pipelines = [pipeline]

        for _ in range(5):
            await task._service_source(pipeline, inference_api=None)

        assert provider.reconnect_calls == 0
        assert pipeline.health.consecutive_failures == 0
        assert pipeline.health.connecting is True
