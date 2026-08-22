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
    return pipeline


def _task() -> StreamWatcherTask:
    """A watcher with only the state the recovery path touches."""
    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._failed_sources = set()
    task._pipelines = []
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
        assert delays[1] > delays[0], "backoff must grow"
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
        assert pipeline.health.consecutive_failures == 2, "the attempt counts as a failure"

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
