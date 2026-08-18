"""The manager's own bookkeeping runs on threads nothing else can exhaust.

`asyncio.to_thread` uses one process-wide executor of `min(32, cpu + 4)` threads,
shared by inference requests, training submissions, image encodes and file writes.
That is fine until something saturates it, and then what suffers is whatever is
smallest and most important — the heartbeat.

Not hypothetical: it is what made a wedged watcher invisible. Leaked inference threads
filled the pool, `update_task_heartbeat` could not get a slot, and the task went on
looking alive in every view the app has while nothing was happening. Bounding those
calls makes saturation unlikely; a private executor makes starving the *reporting*
path impossible.
"""

import asyncio
import threading

import pytest

from apps.tasks_server.manager import Manager
from tests.tasks_server.conftest import CONFIG_UUID, TASK_UUID


class TestIsolation:
    async def test_bookkeeping_does_not_run_on_the_default_executor(self, manager):
        """The property in one assertion: these calls use the private pool."""
        names = []

        def _who_am_i():
            names.append(threading.current_thread().name)

        await manager.db_call(_who_am_i)

        assert names and names[0].startswith("tasks-bookkeeping"), names

    async def test_bookkeeping_still_works_with_the_default_executor_saturated(self, manager):
        """The scenario that actually happened, reproduced.

        The default pool is filled with work that will not finish until released. A
        heartbeat write must still get through — if it went through `to_thread` it
        would queue behind all of it and never land.
        """
        release = threading.Event()
        occupied = threading.Semaphore(0)

        def _hog():
            occupied.release()
            release.wait(timeout=10)

        # Comfortably more than min(32, cpu + 4) so the shared pool has nothing left.
        hogs = [asyncio.ensure_future(asyncio.to_thread(_hog)) for _ in range(40)]
        try:
            # Wait until the pool is genuinely busy rather than assuming it.
            for _ in range(100):
                if not occupied.acquire(blocking=False):
                    await asyncio.sleep(0.02)
                    continue
                break

            done = []
            await asyncio.wait_for(manager.db_call(lambda: done.append(True)), timeout=5.0)

            assert done == [True], "bookkeeping must not queue behind saturated task threads"
        finally:
            release.set()
            await asyncio.gather(*hogs, return_exceptions=True)

    async def test_a_heartbeat_goes_through_the_private_pool(self, manager, tasks_api_mock):
        """The specific call whose starvation hid the original outage."""
        names = []
        tasks_api_mock.set_task_instance_heartbeat.side_effect = lambda *a, **k: names.append(
            threading.current_thread().name
        )

        await manager.update_task_heartbeat(TASK_UUID)

        assert names and names[0].startswith("tasks-bookkeeping"), names

    async def test_task_work_is_left_on_the_shared_pool(self):
        """Isolation is worth having because this side is small and predictable.
        Moving bulk work in would rebuild the same problem inside the reserved pool."""
        import inspect

        from apps.tasks_server.tasks import StreamWatcherTask

        source = inspect.getsource(StreamWatcherTask)
        assert "db_call" not in source, "task work must not borrow the bookkeeping threads"


class TestLifecycle:
    async def test_the_executor_is_small(self):
        """Short writes and reads — the point is isolation, not throughput."""
        assert 1 <= Manager._BOOKKEEPING_THREADS <= 8

    async def test_stop_shuts_the_executor_down(self, manager):
        await manager.stop()

        with pytest.raises(RuntimeError):
            manager._bookkeeping_executor.submit(lambda: None)

    async def test_calls_still_surface_their_exceptions(self, manager):
        """Swapping the executor must not swallow anything — every caller's error
        handling keys off the exception coming back."""

        def _boom():
            raise ValueError("db said no")

        with pytest.raises(ValueError, match="db said no"):
            await manager.db_call(_boom)

    async def test_arguments_are_passed_through(self, manager):
        seen = {}

        def _record(a, b, key=None):
            seen.update({"a": a, "b": b, "key": key})
            return "result"

        assert await manager.db_call(_record, 1, 2, key="value") == "result"
        assert seen == {"a": 1, "b": 2, "key": "value"}


class TestEveryManagerDbCallIsIsolated:
    """A guard on the pattern. The natural thing to type is `asyncio.to_thread(...)`,
    it looks identical to the legitimate uses elsewhere, and nothing fails until the
    pool is saturated in production — which is the one moment this has to work.

    Scoped to every module in `apps/tasks_server/` *except* `tasks/`, rather than to
    `manager.py` alone. The rule is about service-level bookkeeping, not about one
    filename, and a file-scoped guard silently stops applying the moment someone factors
    manager logic into a helper module beside it. Task implementations under `tasks/` are
    excluded deliberately: bulk work belongs on the shared pool, and moving it into the
    reserved one would rebuild the same starvation inside it.
    """

    #: Files under `apps/tasks_server/` that own service bookkeeping. Anything added here
    #: later is covered automatically — the sweep is by directory, not by name.
    @staticmethod
    def _bookkeeping_modules():
        import pathlib

        import apps.tasks_server

        # `__path__`, not `__file__`: these are namespace packages (no `__init__.py`), so
        # `__file__` is None and `Path(None)` raises a TypeError that reads like a
        # pathlib bug rather than a missing file.
        root = pathlib.Path(list(apps.tasks_server.__path__)[0])
        # Top level only: `tasks/` is the excluded subtree.
        return sorted(p for p in root.glob("*.py") if p.name != "__init__.py")

    def test_no_bookkeeping_module_uses_the_shared_pool(self):
        import ast

        modules = self._bookkeeping_modules()
        # A sweep that silently matches nothing passes forever; assert it found the work.
        assert any(p.name == "manager.py" for p in modules), modules

        offenders = []
        for path in modules:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                fn = node.func
                if isinstance(fn, ast.Attribute) and fn.attr == "to_thread":
                    offenders.append(f"{path.name}:{node.lineno}  {ast.unparse(node.args[0])[:60]}")

        assert not offenders, (
            "Service bookkeeping must go through `self.db_call(...)` so it cannot be "
            "starved by task work on the shared executor.\n\n"
            "Replace:  await asyncio.to_thread(fn, *args)\n"
            "With:     await self.db_call(fn, *args)\n\n"
            "(Task implementations under tasks/ are exempt and are not scanned.)\n\n"
            "Offending calls:\n  " + "\n  ".join(offenders)
        )
