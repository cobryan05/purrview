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
    task._collector_names = {}
    task._profile = SimpleNamespace(actions=actions, name="test profile")
    task._manager = SimpleNamespace(
        get_db_api_client=lambda: object(),
        get_inference_api_client=lambda: object(),
        get_mqtt_bridge=lambda: SimpleNamespace(set_watcher_availability=lambda *a, **k: None),
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

    async def test_collector_name_is_recorded_for_the_status_card(self, monkeypatch):
        table = {"c1": _row("c1", "Litter box collector")}
        actions = [_action_row("collect_training_images", "c1")]
        task = _task(actions, table, monkeypatch)

        await task._build_collectors(object())

        assert task._collector_names["c1"] == "Litter box collector"

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


class TestBuildActionsPassesTrigger:
    """`_build_actions` must hand the row's own trigger to
    `collect_training_images`, so `CollectTrainingImagesAction.triggers()` can
    decide continuous vs event-scoped mode (phase 5)."""

    async def _task_for_build_actions(self, actions, collectors_table, monkeypatch):
        task = _task(actions, collectors_table, monkeypatch)
        monkeypatch.setattr(StreamWatcherTask, "_load_mqtt_settings", AsyncMock(return_value=None))
        task._actions = []
        task._frame_actions = []
        task._mqtt = None
        return task

    async def test_frame_analysed_row_builds_a_continuous_action(self, monkeypatch):
        table = {"c1": _row("c1", "One")}
        actions = [_action_row("collect_training_images", "c1")]
        actions[0].trigger = "frame_analysed"
        task = await self._task_for_build_actions(actions, table, monkeypatch)

        await task._build_actions(object())

        assert len(task._actions) == 1
        _row_built, action = task._actions[0]
        assert action._continuous is True

    async def test_event_started_row_builds_an_event_scoped_action(self, monkeypatch):
        table = {"c1": _row("c1", "One")}
        actions = [_action_row("collect_training_images", "c1")]
        actions[0].trigger = "event_started"
        task = await self._task_for_build_actions(actions, table, monkeypatch)

        await task._build_actions(object())

        assert len(task._actions) == 1
        _row_built, action = task._actions[0]
        assert action._continuous is False


class TestStatusDataCollectorsTable:
    """`get_status_data()` gains one row per attached collector, rendered by
    `status_schema()`'s "collectorsTable" widget."""

    def _task_for_status(self):
        task = StreamWatcherTask.__new__(StreamWatcherTask)
        task._pipelines = []
        task._recent_events = []
        task._inference_ms = []
        task._frames_analysed = 0
        task._events_started = 0
        task._events_ended = 0
        task._action_successes = 0
        task._action_failures = 0
        task._mqtt = None
        task._collectors = {}
        task._collector_names = {}
        return task

    def test_one_row_per_attached_collector(self):
        task = self._task_for_status()

        def _stats(offered, saved, dropped, image_hits, bbox_hits, filter_discarded):
            return (
                {
                    "offered": offered,
                    "saved": saved,
                    "dropped_behind": dropped,
                    "dedup_image_hits": image_hits,
                    "dedup_bbox_hits": bbox_hits,
                    "filter_discarded": filter_discarded,
                },
                {},
            )

        c1 = SimpleNamespace(status_stats=lambda: _stats(40, 12, 1, 3, 2, 7))
        task._collectors = {"c1": c1}
        task._collector_names = {"c1": "Litter box collector"}

        data = task.get_status_data()

        assert data["collectorsTable"] == [
            {
                "name": "Litter box collector",
                "uuid": "c1",
                "offered": 40,
                "saved": 12,
                # Blank, not 0: this collector has never saved for a non-band
                # reason, so a zero would read as "the knob ran and found nothing"
                # rather than "the knob isn't on".
                "savedBand": "",
                "savedDetectorMiss": "",
                "savedTrackWorst": "",
                "noveltyDropped": "",
                "droppedBehind": 1,
                "dedupHits": 5,  # image_hits(3) + bbox_hits(2)
                "filterDiscarded": 7,
            }
        ]

    def test_dropped_frames_raise_a_tuning_hint_naming_the_collector(self):
        """A non-blocking offer sheds frames silently by design (plan §7.1), and
        silence there is indistinguishable from "this config collects nothing"."""
        task = self._task_for_status()
        task._collectors = {"c1": SimpleNamespace(status_stats=lambda: ({"dropped_behind": 4}, {}))}
        task._collector_names = {"c1": "Doorway"}

        hints = task.get_status_data()["tuningHints"]

        assert "Doorway" in hints and "4" in hints

    def test_no_hint_while_nothing_is_being_dropped(self):
        task = self._task_for_status()
        task._collectors = {"c1": SimpleNamespace(status_stats=lambda: ({"dropped_behind": 0}, {}))}
        task._collector_names = {"c1": "Doorway"}

        assert "Doorway" not in task.get_status_data()["tuningHints"]

    def test_save_reason_subtotals_reach_the_table(self):
        task = self._task_for_status()
        task._collectors = {
            "c1": SimpleNamespace(
                status_stats=lambda: (
                    {"saved": 10, "saved_band": 7, "saved_detector_miss": 2, "saved_track_worst": 1},
                    {},
                )
            )
        }
        task._collector_names = {"c1": "Doorway"}

        row = task.get_status_data()["collectorsTable"][0]

        assert (row["savedBand"], row["savedDetectorMiss"], row["savedTrackWorst"]) == (7, 2, 1)

    def test_no_attached_collectors_is_an_empty_table(self):
        task = self._task_for_status()
        assert task.get_status_data()["collectorsTable"] == []

    def test_falls_back_to_the_uuid_when_no_name_was_recorded(self):
        task = self._task_for_status()
        task._collectors = {"c1": SimpleNamespace(status_stats=lambda: ({}, {}))}
        # _collector_names deliberately left empty.

        data = task.get_status_data()

        assert data["collectorsTable"][0]["name"] == "c1"


class TestStatusSchemaCoversStatusData:
    """Every key `get_status_data()` returns must have a widget declaring it.

    The frontend renders `statusData[widget.id]` per declared widget, so a key
    with no widget is computed and thrown away with nothing raising anywhere —
    which is exactly what happened to `collectorsTable` for three phases.
    """

    def test_every_status_data_key_is_declared_by_a_widget(self):
        task = TestStatusDataCollectorsTable()._task_for_status()

        declared = {w["id"] for w in StreamWatcherTask.status_schema()}
        # `source="summary"` widgets read the task's summary text, not status data.
        produced = set(task.get_status_data())

        assert produced <= declared, f"computed but never rendered: {sorted(produced - declared)}"


class TestReviewActions:
    """A watcher-attached collector's ways into Review.

    Without these the whole detector-miss feature is unreachable from the UI:
    only a watcher can produce a detector miss (it needs a tracker), and only
    the standalone task's Jobs card rendered `review_actions()` before.
    """

    @staticmethod
    def _task(collectors, names):
        task = StreamWatcherTask.__new__(StreamWatcherTask)
        task._collectors = collectors
        task._collector_names = names
        return task

    @staticmethod
    def _collector(*labels):
        return SimpleNamespace(
            review_actions=lambda: [{"type": "open_review", "label": lbl, "payload": {}} for lbl in labels]
        )

    def test_a_single_collectors_labels_are_not_prefixed(self):
        task = self._task({"c1": self._collector("Browse detector misses")}, {"c1": "Doorway"})

        actions = task.get_status_actions()

        assert [a["label"] for a in actions] == ["Browse detector misses"]

    def test_two_collectors_are_disambiguated_by_name(self):
        task = self._task(
            {"c1": self._collector("Browse unverified"), "c2": self._collector("Browse unverified")},
            {"c1": "Doorway", "c2": "Litter box"},
        )

        assert [a["label"] for a in task.get_status_actions()] == [
            "Doorway — Browse unverified",
            "Litter box — Browse unverified",
        ]

    def test_payloads_are_carried_through_untouched(self):
        payload = {"tag_uuids": ["t"], "bbox_verification": "unverified"}
        collector = SimpleNamespace(review_actions=lambda: [{"type": "open_review", "label": "x", "payload": payload}])
        task = self._task({"c1": collector}, {"c1": "Doorway"})

        assert task.get_status_actions()[0]["payload"] is payload

    def test_a_task_built_before_init_has_no_actions_rather_than_raising(self):
        """Reachable through the PENDING window, and tests build this class with
        `__new__` — an attribute read here would AttributeError far from the change."""
        task = StreamWatcherTask.__new__(StreamWatcherTask)

        assert task.get_status_actions() == []
