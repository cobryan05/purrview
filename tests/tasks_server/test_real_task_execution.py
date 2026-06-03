"""Tests for real Task execution lifecycle.

Unlike the existing tasks_server tests (which mock everything), these tests
exercise the actual Task._execute() path: init → run → deinit, cancel flag,
error handling, and the data-request/data-ready protocol.

No HTTP server is spun up here.  Tasks are instantiated directly.
"""

import asyncio
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from apps.helpers.consts import TaskStatus
from apps.tasks_server.tasks.Registry import register_task, task_registry
from apps.tasks_server.tasks.Task import Task

# A fake manager — test tasks that don't call manager methods need only a stub.
_MOCK_MANAGER = MagicMock()


# ──────────────────────────────────────────────────────────────────────────────
# Minimal test task definitions (registered only if not already present)
# ──────────────────────────────────────────────────────────────────────────────

if "StepsTask" not in task_registry:

    @register_task("StepsTask")
    class StepsTask(Task):
        """Completes N steps, honouring the cancel flag each step."""

        async def _init(self, params: dict, resume_data: Optional[dict]) -> None:
            self.steps = int(params.get("steps", 3))
            self.executed = 0

        async def _run(self) -> dict[str, Any]:
            for _ in range(self.steps):
                if self._cancel_flag.is_set():
                    break
                await asyncio.sleep(0)  # yield to allow cancel to propagate
                self.executed += 1
            return {"executed": self.executed}

        async def _deinit(self) -> None:
            pass


if "ErrorTask" not in task_registry:

    @register_task("ErrorTask")
    class ErrorTask(Task):
        """Always raises in _run."""

        async def _init(self, params: dict, resume_data: Optional[dict]) -> None:
            pass

        async def _run(self) -> dict[str, Any]:
            raise ValueError("deliberate error")

        async def _deinit(self) -> None:
            pass


if "DataFlagTask" not in task_registry:

    @register_task("DataFlagTask")
    class DataFlagTask(Task):
        """Waits for data_req_flag, writes resume_data, signals data_ready."""

        async def _init(self, params: dict, resume_data: Optional[dict]) -> None:
            self.saved_resume = resume_data or {}

        async def _run(self) -> dict[str, Any]:
            # Spin until cancelled or data_req comes in
            while not self._cancel_flag.is_set():
                if self._data_req_flag.is_set():
                    self._resume_data["checkpoint"] = 42
                    self._data_ready_flag.set()
                    self._data_req_flag.clear()
                await asyncio.sleep(0)
            return {}

        async def _deinit(self) -> None:
            pass


if "ResumeTask" not in task_registry:

    @register_task("ResumeTask")
    class ResumeTask(Task):
        """Records whether resume_data was injected before _run."""

        async def _init(self, params: dict, resume_data: Optional[dict]) -> None:
            self.got_resume = bool(resume_data)
            self.resume_checkpoint = (resume_data or {}).get("checkpoint", 0)

        async def _run(self) -> dict[str, Any]:
            return {"checkpoint": self.resume_checkpoint}

        async def _deinit(self) -> None:
            pass


def _make_task(cls_name: str, params: dict | None = None) -> Task:
    return task_registry[cls_name](
        task_config_uuid="test-config-uuid",
        params=params or {},
        manager=_MOCK_MANAGER,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Happy path: full lifecycle
# ──────────────────────────────────────────────────────────────────────────────


async def test_task_completes_successfully():
    task = _make_task("StepsTask", {"steps": 3})
    await task.start()
    await task.wait_for_task_done(timeout=5.0)

    assert task.get_status() == TaskStatus.COMPLETED
    assert task.is_task_done()
    results = task.get_results()
    assert results is not None
    assert results["data"]["executed"] == 3


async def test_task_progress_reaches_100():
    task = _make_task("StepsTask", {"steps": 2})
    await task.start()
    await task.wait_for_task_done(timeout=5.0)
    assert task.get_progress() == 100.0


async def test_task_logs_lifecycle_messages():
    task = _make_task("StepsTask", {"steps": 1})
    await task.start()
    await task.wait_for_task_done(timeout=5.0)
    logs = task.get_log_lines()
    assert any("Initializing" in line for line in logs)
    assert any("Running" in line for line in logs)
    assert any("Completed" in line for line in logs)


# ──────────────────────────────────────────────────────────────────────────────
# Cancel
# ──────────────────────────────────────────────────────────────────────────────


async def test_cancel_stops_task_early():
    # Use many steps so it won't finish before we cancel
    task = _make_task("StepsTask", {"steps": 1000})
    await task.start()
    await task.stop()  # sets _cancel_flag
    await task.wait_for_task_done(timeout=5.0)

    assert task.is_task_done()
    # Status will be COMPLETED because the task completed its loop (cancel
    # checked per step, not via exception).  Executed count must be < 1000.
    results = task.get_results()
    assert results is not None
    assert results["data"]["executed"] < 1000


async def test_start_twice_raises():
    task = _make_task("StepsTask", {"steps": 1})
    await task.start()
    with pytest.raises(RuntimeError):
        await task.start()
    # Clean up
    await task.wait_for_task_done(timeout=5.0)


# ──────────────────────────────────────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────────────────────────────────────


async def test_error_in_run_sets_error_status():
    task = _make_task("ErrorTask")
    await task.start()
    await task.wait_for_task_done(timeout=5.0)

    assert task.get_status() == TaskStatus.ERROR
    results = task.get_results()
    assert results is not None
    assert "deliberate error" in results.get("message", "")


async def test_error_task_logs_error_message():
    task = _make_task("ErrorTask")
    await task.start()
    await task.wait_for_task_done(timeout=5.0)
    logs = task.get_log_lines()
    assert any("Error" in line for line in logs)


# ──────────────────────────────────────────────────────────────────────────────
# Data request / data ready protocol
# ──────────────────────────────────────────────────────────────────────────────


async def test_data_request_flag_triggers_resume_data_save():
    task = _make_task("DataFlagTask")
    await task.start()

    # Request data persistence
    task.request_data_update()
    # Wait until data is ready
    await task.wait_for_data_ready(timeout=5.0)

    assert task.is_data_ready()
    assert task._resume_data.get("checkpoint") == 42

    # Clean up
    task.cancel_task()
    await task.wait_for_task_done(timeout=5.0)


async def test_clear_data_ready(task=None):
    t = _make_task("DataFlagTask")
    await t.start()
    t.request_data_update()
    await t.wait_for_data_ready(timeout=5.0)

    t.clear_data_ready()
    assert not t.is_data_ready()

    t.cancel_task()
    await t.wait_for_task_done(timeout=5.0)


# ──────────────────────────────────────────────────────────────────────────────
# Resume data injection
# ──────────────────────────────────────────────────────────────────────────────


async def test_resume_data_injected_before_init():
    task = _make_task("ResumeTask")
    task.set_resume_data({"checkpoint": 7})

    await task.start()
    await task.wait_for_task_done(timeout=5.0)

    results = task.get_results()
    assert results["data"]["checkpoint"] == 7


async def test_no_resume_data_gives_zero_checkpoint():
    task = _make_task("ResumeTask")
    await task.start()
    await task.wait_for_task_done(timeout=5.0)

    results = task.get_results()
    assert results["data"]["checkpoint"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# wait_for_task_done timeout
# ──────────────────────────────────────────────────────────────────────────────


async def test_wait_for_task_done_timeout():
    # DataFlagTask never finishes unless cancelled — timeout should raise
    task = _make_task("DataFlagTask")
    await task.start()

    with pytest.raises(asyncio.TimeoutError):
        await task.wait_for_task_done(timeout=0.05)

    # Clean up
    task.cancel_task()
    await task.wait_for_task_done(timeout=5.0)
