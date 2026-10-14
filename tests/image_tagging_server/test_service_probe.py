"""Unit tests for Manager.get_all_service_status()'s soft/hard probe timeouts.

A service that's just busy (mid-inference, mid-training-step) can be slow to
answer its own /status health check. These tests cover the fix for probes
flagging a busy-but-fine service "not responding" after only a few seconds:
now a slow probe reports "still waiting" (info, not a failure) once
PROBE_SOFT_TIMEOUT elapses, and is only marked unreachable after the much
longer PROBE_HARD_TIMEOUT.
"""

import asyncio

import httpx
import pytest

import apps.image_tagging_server.manager as manager_module

# Scaled-down thresholds so tests run in a fraction of a second. Keep a wide
# margin between them (and above PROBE_INLINE_WAIT) so scheduler jitter under
# a loaded test run can't flip which side of a boundary a sleep lands on.
_SOFT = 0.15
_HARD = 0.6
_INLINE = 0.02


@pytest.fixture(autouse=True)
def _fast_timeouts(monkeypatch):
    monkeypatch.setattr(manager_module, "PROBE_SOFT_TIMEOUT", _SOFT)
    monkeypatch.setattr(manager_module, "PROBE_HARD_TIMEOUT", _HARD)
    monkeypatch.setattr(manager_module, "PROBE_INLINE_WAIT", _INLINE)


def _install_slow_get(manager, delay: float, *, status_code: int = 200, payload: dict | None = None):
    """Replace manager._test_client.get with one that sleeps `delay` before
    responding — or raises httpx.TimeoutException if `delay` exceeds the
    caller's `timeout` kwarg, matching real httpx behavior."""

    async def _slow_get(url, timeout=None):
        if timeout is not None and delay > timeout:
            await asyncio.sleep(timeout)
            raise httpx.TimeoutException("simulated timeout", request=httpx.Request("GET", url))
        await asyncio.sleep(delay)
        return httpx.Response(status_code, json=payload or {"service": "x"}, request=httpx.Request("GET", url))

    manager._test_client.get = _slow_get


def _install_gated_get(manager, release: asyncio.Event, *, raise_timeout: bool = False):
    """Replace manager._test_client.get with one that blocks until `release` is
    set, then answers (or raises the timeout httpx would raise).

    The probe's completion is driven by the test instead of by a sleep, so a
    stalled scheduler can only make the test slower — it can never carry the
    probe past a boundary the assertion depends on being on the near side of.
    """

    async def _gated_get(url, timeout=None):
        await release.wait()
        if raise_timeout:
            raise httpx.TimeoutException("simulated timeout", request=httpx.Request("GET", url))
        return httpx.Response(200, json={"service": "x"}, request=httpx.Request("GET", url))

    manager._test_client.get = _gated_get


async def _drain_probes(manager):
    """Await every in-flight probe task, so the next poll sees a resolved probe."""
    tasks = list(manager._probe_tasks.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.fixture
async def gate(manager):
    """The event a gated probe blocks on. Released and drained at teardown so
    no probe task is still pending when the event loop closes."""
    release = asyncio.Event()
    yield release
    release.set()
    await _drain_probes(manager)


def _probed(statuses: list[dict]) -> list[dict]:
    """Only the entries that came from an HTTP probe.

    get_all_service_status() also reports image_tagging_server itself, built
    in-memory rather than probed (it is the process answering the request, so
    asking itself over HTTP would spend a worker to learn nothing). These tests
    are about probe timeout behaviour, so that entry is not one of their
    subjects — without this filter they assert "every service has a
    still-waiting notice" against a service that was never waiting.
    """
    return [s for s in statuses if s.get("service") != "image_tagging_server"]


class TestProbeSoftHardTimeouts:
    async def test_fast_response_reports_success_with_no_notice(self, manager):
        _install_slow_get(manager, delay=0.0)
        statuses = await manager.get_all_service_status()
        for svc in _probed(statuses):
            assert svc.get("notices", []) == []

    async def test_slow_response_under_soft_timeout_is_silent(self, manager):
        """A response that resolves before PROBE_SOFT_TIMEOUT should not flap
        the UI with any notice at all, even mid-flight."""
        _install_slow_get(manager, delay=_SOFT / 2)
        statuses = await manager.get_all_service_status()
        for svc in _probed(statuses):
            assert svc.get("notices", []) == []

    async def test_slow_response_past_soft_timeout_reports_still_waiting(self, manager, gate):
        """Once PROBE_SOFT_TIMEOUT elapses but the probe is still in flight
        (and under PROBE_HARD_TIMEOUT), report an info-level 'still waiting'
        notice — not the old 'not responding' warning."""
        _install_gated_get(manager, gate)

        # First poll starts the probe; it won't have hit the soft threshold yet.
        await manager.get_all_service_status()
        await asyncio.sleep(_SOFT * 1.5)

        statuses = await manager.get_all_service_status()
        assert len(statuses) > 0
        for svc in _probed(statuses):
            notices = svc.get("notices", [])
            assert len(notices) == 1
            assert notices[0]["level"] == "info"
            assert "still waiting" in notices[0]["message"].lower()

    async def test_response_beyond_hard_timeout_is_marked_unreachable(self, manager, gate):
        """A probe that never resolves within PROBE_HARD_TIMEOUT is finally
        flagged unreachable (warning level) — the hard failure case still exists,
        it just takes much longer to trigger than before."""
        _install_gated_get(manager, gate, raise_timeout=True)

        await manager.get_all_service_status()  # kicks off the probe
        await asyncio.sleep(_SOFT * 1.5)
        statuses = await manager.get_all_service_status()
        for svc in _probed(statuses):
            assert svc["notices"][0]["level"] == "info"

        # Release the probe into the failure httpx would raise at its own
        # request timeout, which resolves it to "unreachable".
        gate.set()
        await _drain_probes(manager)
        statuses = await manager.get_all_service_status()
        for svc in _probed(statuses):
            notices = svc.get("notices", [])
            assert len(notices) == 1
            assert notices[0]["level"] == "warning"
            assert "not responding" in notices[0]["message"].lower()

    async def test_concurrent_pollers_do_not_double_delete_probe_task(self, manager):
        """The cached /notices endpoint and the uncached SSE notices stream can
        both call get_all_service_status concurrently. When they await the same
        in-flight probe task and it resolves during the inline-wait window, both
        reach the cleanup branch — removing the task from _probe_tasks must be
        idempotent, else the second poller raises KeyError deleting a key the
        first already removed."""
        # Small nonzero delay: long enough for every concurrent poller to reach
        # `await asyncio.wait(...)` on the shared task before it resolves, short
        # enough to resolve inside PROBE_INLINE_WAIT so they all hit the branch.
        _install_slow_get(manager, delay=_INLINE * 0.4)

        results = await asyncio.gather(*[manager.get_all_service_status() for _ in range(16)])
        assert len(results) == 16
        for statuses in results:
            assert isinstance(statuses, list)

    async def test_recovers_to_healthy_after_waiting(self, manager, gate):
        """Once a slow-but-in-flight probe finally resolves successfully, the
        next poll should reflect that — no notice."""
        _install_gated_get(manager, gate)

        await manager.get_all_service_status()
        await asyncio.sleep(_SOFT * 1.5)
        statuses = await manager.get_all_service_status()
        for svc in _probed(statuses):
            assert svc["notices"][0]["level"] == "info"

        gate.set()
        await _drain_probes(manager)

        statuses = await manager.get_all_service_status()
        for svc in _probed(statuses):
            assert svc.get("notices", []) == []
