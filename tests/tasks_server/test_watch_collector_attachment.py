"""Watcher-side collector ownership (phase 4, COLLECTOR_UNIFICATION_PLAN.md).

`TrainingCollector` itself is exercised elsewhere (tests/helpers/,
test_collect_training_images_task.py). What's worth pinning here is the
wiring `StreamWatcherTask` owns on top of it: one collector instance per
distinct `collector_uuid` shared across every action row naming it, a bad
attachment that doesn't stop the watcher starting, and shutdown reaching
every collector that was built — not the action framework's generic
behaviour, which the other watch-action tests already cover.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import apps.tasks_server.tasks.StreamWatcherTask as watcher_module
from apps.tasks_server.tasks.StreamWatcherTask import StreamWatcherTask


class _FakeCollector:
    """Records its own lifecycle calls instead of touching the DB/inference server."""

    instances: list = []

    def __init__(self, spec, host, **kwargs):
        self.spec = spec
        self.host = host
        self.kwargs = kwargs
        self.prepared = False
        self.started = False
        self.shutdown_called = False
        _FakeCollector.instances.append(self)

    async def prepare(self, resume_data=None):
        self.prepared = True

    async def start(self):
        self.started = True

    async def shutdown(self):
        self.shutdown_called = True


class _FakeCollectorsApi:
    """Answers `get_training_collector` from a canned table, keyed by uuid."""

    def __init__(self, table):
        self._table = table

    def get_training_collector(self, collector_uuid):
        row = self._table.get(collector_uuid)
        if row is None:
            return SimpleNamespace(collector=None, message="not found")
        return SimpleNamespace(collector=row)


def _row(uuid: str, name: str = "", enabled: bool = True, params=None):
    return SimpleNamespace(uuid=uuid, name=name, enabled=enabled, params=params or {})


def _action_row(
    kind: str, collector_uuid: str = "", enabled: bool = True, order: int = 0, min_interval_s: float = 10.0
):
    return SimpleNamespace(
        uuid=f"row-{collector_uuid}-{order}",
        kind=kind,
        enabled=enabled,
        order=order,
        params_json={"collector_uuid": collector_uuid, "min_interval_s": min_interval_s} if collector_uuid else {},
    )


def _task(actions, collectors_table, monkeypatch) -> StreamWatcherTask:
    """A task with only what `_build_collectors`/`_collector_for` touch."""
    _FakeCollector.instances = []
    monkeypatch.setattr(watcher_module, "TrainingCollector", _FakeCollector)
    monkeypatch.setattr(
        watcher_module.db_client, "TrainingCollectorsApi", lambda client: _FakeCollectorsApi(collectors_table)
    )

    task = StreamWatcherTask.__new__(StreamWatcherTask)
    task._collectors = {}
    task._profile = SimpleNamespace(actions=actions)
    task._manager = SimpleNamespace(
        get_db_api_client=lambda: object(),
        get_inference_api_client=lambda: object(),
    )
    task._cancel_flag = asyncio.Event()
    task._logs: list = []
    task.log = task._logs.append
    return task


class TestBuildCollectors:
    async def test_builds_one_collector_per_distinct_uuid(self, monkeypatch):
        table = {"c1": _row("c1", "Collector One")}
        actions = [_action_row("collect_training_images", "c1")]
        task = _task(actions, table, monkeypatch)

        await task._build_collectors(object())

        assert set(task._collectors) == {"c1"}
        collector = task._collectors["c1"]
        assert collector.prepared and collector.started

    async def test_two_rows_naming_the_same_collector_share_one_instance(self, monkeypatch):
        table = {"c1": _row("c1", "Collector One")}
        actions = [
            _action_row("collect_training_images", "c1", order=0),
            _action_row("collect_training_images", "c1", order=1),
        ]
        task = _task(actions, table, monkeypatch)

        await task._build_collectors(object())

        assert len(_FakeCollector.instances) == 1
        assert task._collector_for("c1") is _FakeCollector.instances[0]

    async def test_a_missing_collector_is_skipped_without_raising(self, monkeypatch):
        actions = [_action_row("collect_training_images", "does-not-exist")]
        task = _task(actions, {}, monkeypatch)

        await task._build_collectors(object())  # must not raise

        assert task._collectors == {}
        assert any("does-not-exist" in line for line in task._logs)

    async def test_a_disabled_collector_is_skipped_without_raising(self, monkeypatch):
        table = {"c1": _row("c1", "Collector One", enabled=False)}
        actions = [_action_row("collect_training_images", "c1")]
        task = _task(actions, table, monkeypatch)

        await task._build_collectors(object())

        assert task._collectors == {}

    async def test_a_disabled_action_row_is_not_attached(self, monkeypatch):
        table = {"c1": _row("c1", "Collector One")}
        actions = [_action_row("collect_training_images", "c1", enabled=False)]
        task = _task(actions, table, monkeypatch)

        await task._build_collectors(object())

        assert task._collectors == {}

    async def test_other_action_kinds_do_not_attach_a_collector(self, monkeypatch):
        actions = [_action_row("publish_mqtt")]
        task = _task(actions, {}, monkeypatch)

        await task._build_collectors(object())

        assert task._collectors == {}

    async def test_unresolved_collector_lookup_falls_through_to_collector_for_none(self, monkeypatch):
        task = _task([], {}, monkeypatch)
        assert task._collector_for("never-built") is None


class TestDeinitShutsDownCollectors:
    async def test_deinit_shuts_down_every_attached_collector(self, monkeypatch):
        table = {"c1": _row("c1", "One"), "c2": _row("c2", "Two")}
        actions = [_action_row("collect_training_images", "c1"), _action_row("collect_training_images", "c2")]
        task = _task(actions, table, monkeypatch)
        await task._build_collectors(object())
        collectors = list(task._collectors.values())
        assert len(collectors) == 2

        # Everything else _deinit touches, stubbed to no-ops.
        task._recorder = SimpleNamespace(close_all=AsyncMock())
        task._actions = []
        task._mqtt = None
        task._pipelines = []
        task._pin_ids = {}
        monkeypatch.setattr(watcher_module, "ModelsApi", lambda client: SimpleNamespace())

        await task._deinit()

        assert all(c.shutdown_called for c in collectors)

    async def test_a_collector_that_fails_to_shut_down_does_not_block_the_others(self, monkeypatch):
        table = {"c1": _row("c1", "One"), "c2": _row("c2", "Two")}
        actions = [_action_row("collect_training_images", "c1"), _action_row("collect_training_images", "c2")]
        task = _task(actions, table, monkeypatch)
        await task._build_collectors(object())
        broken, healthy = task._collectors["c1"], task._collectors["c2"]

        async def _raise():
            raise RuntimeError("boom")

        broken.shutdown = _raise

        task._recorder = SimpleNamespace(close_all=AsyncMock())
        task._actions = []
        task._mqtt = None
        task._pipelines = []
        task._pin_ids = {}
        monkeypatch.setattr(watcher_module, "ModelsApi", lambda client: SimpleNamespace())

        await task._deinit()  # must not raise

        assert healthy.shutdown_called
