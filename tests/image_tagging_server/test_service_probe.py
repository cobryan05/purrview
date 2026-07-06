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


class TestProbeSoftHardTimeouts:
    async def test_fast_response_reports_success_with_no_notice(self, manager):
        _install_slow_get(manager, delay=0.0)
        statuses = await manager.get_all_service_status()
        for svc in statuses:
            assert svc.get("notices", []) == []

    async def test_slow_response_under_soft_timeout_is_silent(self, manager):
        """A response that resolves before PROBE_SOFT_TIMEOUT should not flap
        the UI with any notice at all, even mid-flight."""
        _install_slow_get(manager, delay=_SOFT / 2)
        statuses = await manager.get_all_service_status()
        for svc in statuses:
            assert svc.get("notices", []) == []

    async def test_slow_response_past_soft_timeout_reports_still_waiting(self, manager):
        """Once PROBE_SOFT_TIMEOUT elapses but the probe is still in flight
        (and under PROBE_HARD_TIMEOUT), report an info-level 'still waiting'
        notice — not the old 'not responding' warning."""
        _install_slow_get(manager, delay=(_SOFT + _HARD) / 2)

        # First poll starts the probe; it won't have hit the soft threshold yet.
        await manager.get_all_service_status()
        await asyncio.sleep(_SOFT * 1.5)

        statuses = await manager.get_all_service_status()
        assert len(statuses) > 0
        for svc in statuses:
            notices = svc.get("notices", [])
            assert len(notices) == 1
            assert notices[0]["level"] == "info"
            assert "still waiting" in notices[0]["message"].lower()

    async def test_response_beyond_hard_timeout_is_marked_unreachable(self, manager):
        """A probe that never resolves within PROBE_HARD_TIMEOUT is finally
        flagged unreachable (warning level) — the hard failure case still exists,
        it just takes much longer to trigger than before."""
        _install_slow_get(manager, delay=_HARD * 5)  # never completes within a poll's patience

        await manager.get_all_service_status()  # kicks off the probe
        await asyncio.sleep(_SOFT * 1.5)
        statuses = await manager.get_all_service_status()
        for svc in statuses:
            assert svc["notices"][0]["level"] == "info"

        # Wait past PROBE_HARD_TIMEOUT — httpx's own request timeout fires
        # inside the background task, resolving it to "unreachable".
        await asyncio.sleep(_HARD * 1.5)
        statuses = await manager.get_all_service_status()
        for svc in statuses:
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

    async def test_recovers_to_healthy_after_waiting(self, manager):
        """Once a slow-but-in-flight probe finally resolves successfully, the
        next poll should reflect that — no notice."""
        _install_slow_get(manager, delay=(_SOFT + _HARD) / 2)

        await manager.get_all_service_status()
        await asyncio.sleep(_SOFT * 1.5)
        statuses = await manager.get_all_service_status()
        for svc in statuses:
            assert svc["notices"][0]["level"] == "info"

        await asyncio.sleep(_HARD)  # let the in-flight probe resolve successfully

        statuses = await manager.get_all_service_status()
        for svc in statuses:
            assert svc.get("notices", []) == []
