"""Regression tests: stopping a streamer must be bounded, whatever ffmpeg does.

The bug these guard against wedged a watcher permanently. `astop()` did
`process.terminate()` followed by an unbounded `await asyncio.to_thread(process.wait)`.
ffmpeg blocked on a read from an RTSP source that had stopped delivering frames never
acts on SIGTERM, so that wait never returned — and because `asyncio.to_thread` is not
cancellable, the task awaiting it could not be cancelled either. The user's Stop click
did nothing, forever, with no way back short of restarting the service.

No real ffmpeg subprocess is used: the fakes below stand in for `Popen` and model the
two failure modes that matter — ignoring SIGTERM, and ignoring SIGKILL too.
"""

import asyncio
import subprocess
import threading

import pytest

from apps.helpers.streams.ffmpegStreamerIn import FFmpegStreamerIn


class FakePopen:
    """A Popen stand-in whose responsiveness to each signal is configurable."""

    def __init__(self, *, dies_on_terminate: bool = True, dies_on_kill: bool = True) -> None:
        self.pid = 4242
        self._dies_on_terminate = dies_on_terminate
        self._dies_on_kill = dies_on_kill
        self._dead = False
        self.terminate_calls = 0
        self.kill_calls = 0
        self.waits: list[float] = []

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self._dies_on_terminate:
            self._dead = True

    def kill(self) -> None:
        self.kill_calls += 1
        if self._dies_on_kill:
            self._dead = True

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self._dead:
            return 0
        # Real Popen.wait blocks for `timeout` then raises; the test only needs the
        # raise, and actually sleeping would make the suite pay the whole budget.
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout)


@pytest.fixture
async def streamer(monkeypatch):
    # Async because FFmpegStreamerIn.__init__ grabs the running loop.
    s = FFmpegStreamerIn("rtsp://example/stalled")
    # Shrink the budget: the assertions are about escalation and boundedness, not
    # about the production numbers, and a test must not wait 10 real seconds.
    monkeypatch.setattr(FFmpegStreamerIn, "TERMINATE_TIMEOUT_S", 0.01)
    monkeypatch.setattr(FFmpegStreamerIn, "KILL_TIMEOUT_S", 0.01)
    monkeypatch.setattr(FFmpegStreamerIn, "WORKER_JOIN_TIMEOUT_S", 0.01)
    return s


class TestStopEscalation:
    async def test_astop_returns_when_ffmpeg_ignores_sigterm(self, streamer):
        """The core regression: an unkillable-by-SIGTERM process must not hang astop()."""
        proc = FakePopen(dies_on_terminate=False, dies_on_kill=True)
        streamer._process = proc

        await asyncio.wait_for(streamer.astop(), timeout=5.0)

        assert proc.terminate_calls == 1, "SIGTERM is still tried first"
        assert proc.kill_calls == 1, "a process that ignored SIGTERM must be escalated to SIGKILL"
        assert streamer._process is None

    async def test_astop_gives_up_rather_than_hanging_on_an_unkillable_process(self, streamer):
        """Even SIGKILL can be unreapable (uninterruptible I/O). Leaking beats wedging."""
        proc = FakePopen(dies_on_terminate=False, dies_on_kill=False)
        streamer._process = proc

        await asyncio.wait_for(streamer.astop(), timeout=5.0)

        assert proc.kill_calls == 1
        assert streamer._process is None, "the handle is dropped so a retry isn't blocked by it"

    async def test_astop_never_waits_unbounded(self, streamer):
        """Every wait() call must carry a timeout — an unbounded one is the original bug."""
        proc = FakePopen(dies_on_terminate=False, dies_on_kill=False)
        streamer._process = proc

        await streamer.astop()

        assert proc.waits, "the process must actually be reaped"
        assert all(t is not None for t in proc.waits), f"unbounded wait() found: {proc.waits}"

    async def test_astop_does_not_escalate_a_cooperative_process(self, streamer):
        """The normal path is unchanged: SIGTERM is enough, and SIGKILL is never sent."""
        proc = FakePopen(dies_on_terminate=True)
        streamer._process = proc

        await streamer.astop()

        assert proc.terminate_calls == 1
        assert proc.kill_calls == 0

    async def test_astop_does_not_hang_on_a_worker_that_swallows_cancellation(self, streamer):
        """A worker stuck in a blocking call must be abandoned, not awaited forever."""

        # `release` exists only so the test can clean up after itself: without it the
        # fake worker would outlive the test and stall the loop's teardown.
        release = asyncio.Event()

        async def _uncancellable() -> None:
            while not release.is_set():
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    continue  # models a worker that never actually unwinds

        streamer._process = None
        task = asyncio.ensure_future(_uncancellable())
        streamer._task = task
        try:
            await asyncio.wait_for(streamer.astop(), timeout=5.0)
            assert streamer._task is None
            assert not task.done(), "the detached worker is abandoned, not awaited"
        finally:
            release.set()
            task.cancel()
            await asyncio.wait({task}, timeout=1.0)

    async def test_sync_stop_escalates_too(self, streamer):
        """stop() and astop() are the same policy — the sync path had the same hang."""
        proc = FakePopen(dies_on_terminate=False, dies_on_kill=True)
        streamer._process = proc

        streamer.stop()

        assert proc.terminate_calls == 1
        assert proc.kill_calls == 1
        assert all(t is not None for t in proc.waits)


class TestCancelledStart:
    """`astart()` spawns ffmpeg in a thread that cancellation cannot stop.

    Under heavy disk load `Popen` blocks until the child's exec has paged ffmpeg in,
    which can outlast the watcher's fetch budget. The awaiting task is then cancelled,
    the thread finishes the spawn anyway, and nothing holds the process to stop it.
    """

    async def test_a_process_spawned_after_cancellation_is_reaped(self, streamer, monkeypatch):
        release = threading.Event()
        proc = FakePopen(dies_on_terminate=True)

        def slow_spawn():
            release.wait(timeout=5)
            return proc

        monkeypatch.setattr(streamer, "_build_process", slow_spawn)
        start = asyncio.create_task(streamer.astart())
        await asyncio.sleep(0.05)
        start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start

        release.set()
        for _ in range(200):
            if proc.terminate_calls:
                break
            await asyncio.sleep(0.01)

        assert proc.terminate_calls == 1
        assert streamer._process is None

    async def test_an_uncancelled_start_keeps_its_process(self, streamer, monkeypatch):
        proc = FakePopen(dies_on_terminate=True)
        monkeypatch.setattr(streamer, "_build_process", lambda: proc)
        monkeypatch.setattr(streamer, "_worker_task", lambda: asyncio.sleep(0))

        await streamer.astart()

        assert streamer._process is proc
        assert proc.terminate_calls == 0
