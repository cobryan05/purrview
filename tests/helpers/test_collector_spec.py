"""Tests for `apps.helpers.training.collectorSpec`.

The schema tests here guard the one invariant that makes the split worth doing:
the ~30 collect knobs have a single definition, rendered by two different forms.
A knob that exists but never renders, or a driver-only param that leaks into the
collector editor, are both silent — nothing raises, the form is just wrong.
"""

from apps.helpers.training.collectorSpec import (
    AMBIGUITY_METRIC_MARGIN,
    AMBIGUITY_METRIC_RATIO,
    EVICTION_FIFO,
    EVICTION_STRATEGIES,
    CollectorSpec,
    collector_params_schema,
    parse_confidence_band,
)
from apps.tasks_server.tasks.CollectTrainingImagesTask import CollectTrainingImagesTask

# The params that belong to whoever drives the loop, not to collection.
DRIVER_PARAMS = {"source_uuids", "snapshot_interval", "stream_interval"}


class TestCollectorParamsSchema:
    def test_every_knob_is_rendered(self):
        """meta.order drives the form, so a key missing from it never renders."""
        schema = collector_params_schema()
        assert set(schema["meta"]["order"]) == set(schema) - {"meta"}

    def test_driver_params_are_not_the_collectors(self):
        """A collector attached to a watcher has no sources and no pacing interval;
        offering either in its editor would be a control that cannot work."""
        schema = collector_params_schema()
        assert DRIVER_PARAMS.isdisjoint(schema)
        assert DRIVER_PARAMS.isdisjoint(schema["meta"]["order"])

    def test_task_names_a_collector_instead_of_carrying_its_knobs(self):
        """The knobs live on the collector row now. A task form that still rendered
        them would let a run's settings diverge from the collector it names."""
        task = CollectTrainingImagesTask.params_schema()
        assert set(task) - {"meta"} == DRIVER_PARAMS | {"collector_uuid"}
        assert set(collector_params_schema()).isdisjoint(set(task) - {"meta"})


class TestParseConfidenceBand:
    def test_bare_scalar_is_a_floor_with_an_open_top(self):
        band = parse_confidence_band(0.25)
        assert (band.min_confidence, band.max_confidence) == (0.25, 1.0)
        assert band.contains(0.9)

    def test_zero_floor_disables_the_label(self):
        assert parse_confidence_band(0) is None
        assert parse_confidence_band({"min": 0}) is None

    def test_garbage_disables_rather_than_raising(self):
        assert parse_confidence_band("nonsense") is None
        assert parse_confidence_band(None) is None

    def test_zero_max_disables_review_but_not_tagging(self):
        band = parse_confidence_band({"min": 0.3, "max": 0})
        assert band.review_disabled
        assert not band.contains(0.5)
        assert band.min_confidence == 0.3

    def test_inverted_band_degrades_to_a_point_rather_than_matching_nothing(self):
        band = parse_confidence_band({"min": 0.6, "max": 0.2})
        assert band.contains(0.6)
        assert not band.contains(0.7)


class TestCollectorSpecFromParams:
    def test_traversing_output_dir_is_rejected(self):
        for bad in ("../etc", "/etc", "a/../../b"):
            try:
                CollectorSpec.from_params({"output_dir": bad})
            except ValueError:
                continue
            raise AssertionError(f"{bad!r} was accepted as an output_dir")

    def test_no_models_means_collect_everything(self):
        spec = CollectorSpec.from_params({"output_dir": "out"})
        assert spec.collect_all
        assert spec.model_names == []

    def test_a_model_with_only_zero_labels_is_disabled_not_collect_all(self):
        """'No models' and 'a model whose labels are all off' are different states:
        the second is an explicitly disabled model and collects nothing."""
        spec = CollectorSpec.from_params(
            {"output_dir": "out", "model_label_config": [{"modelName": "m", "labelValues": {"l": 0}}]}
        )
        assert not spec.collect_all
        assert spec.model_names == ["m"]
        assert spec.target_labels == []

    def test_min_all_conf_is_the_loosest_floor(self):
        spec = CollectorSpec.from_params(
            {
                "output_dir": "out",
                "model_label_config": [{"modelName": "m", "labelValues": {"a": 0.7, "b": 0.35}}],
            }
        )
        assert spec.min_all_conf == 0.35

    def test_two_configs_on_one_label_widen_to_the_union(self):
        spec = CollectorSpec.from_params(
            {
                "output_dir": "out",
                "model_label_config": [
                    {"modelName": "m1", "labelValues": {"a": {"min": 0.4, "max": 0.6}}},
                    {"modelName": "m2", "labelValues": {"a": {"min": 0.2, "max": 0.5}}},
                ],
            }
        )
        band = spec.target_label_bands["a"]
        assert (band.min_confidence, band.max_confidence) == (0.2, 0.6)

    def test_review_stays_enabled_if_either_config_wants_it(self):
        spec = CollectorSpec.from_params(
            {
                "output_dir": "out",
                "model_label_config": [
                    {"modelName": "m1", "labelValues": {"a": {"min": 0.4, "max": 0}}},
                    {"modelName": "m2", "labelValues": {"a": {"min": 0.4, "max": 0.9}}},
                ],
            }
        )
        assert not spec.target_label_bands["a"].review_disabled

    def test_ambiguity_top_k_floors_at_two(self):
        """k counts the winner, so k=1 leaves no runner-up to be confused with."""
        spec = CollectorSpec.from_params({"output_dir": "out", "enable_ambiguity": True, "ambiguity_top_k": 1})
        assert spec.ambiguity_top_k == 2

    def test_unknown_metric_falls_back_rather_than_raising(self):
        spec = CollectorSpec.from_params({"output_dir": "out", "ambiguity_metric": "vibes"})
        assert spec.ambiguity_metric == AMBIGUITY_METRIC_RATIO
        spec = CollectorSpec.from_params({"output_dir": "out", "ambiguity_metric": AMBIGUITY_METRIC_MARGIN})
        assert spec.ambiguity_metric == AMBIGUITY_METRIC_MARGIN

    def test_review_queue_limits_default_to_unlimited(self):
        spec = CollectorSpec.from_params({"output_dir": "out"})
        assert spec.max_unverified_images is None
        assert spec.max_unverified_bytes is None
        assert spec.eviction_strategy == EVICTION_FIFO

    def test_zero_review_queue_limits_mean_unlimited_not_zero(self):
        """The UI's blank/0 convention for 'no limit' — matches max_images above."""
        spec = CollectorSpec.from_params({"output_dir": "out", "max_unverified_images": 0, "max_unverified_mb": 0})
        assert spec.max_unverified_images is None
        assert spec.max_unverified_bytes is None

    def test_max_unverified_mb_is_converted_to_bytes(self):
        spec = CollectorSpec.from_params({"output_dir": "out", "max_unverified_mb": 5})
        assert spec.max_unverified_bytes == 5 * 1024 * 1024

    def test_unknown_eviction_strategy_falls_back_to_fifo(self):
        spec = CollectorSpec.from_params({"output_dir": "out", "eviction_strategy": "vibes"})
        assert spec.eviction_strategy == EVICTION_FIFO

    def test_eviction_strategy_round_trips_for_every_known_value(self):
        for strategy in EVICTION_STRATEGIES:
            spec = CollectorSpec.from_params({"output_dir": "out", "eviction_strategy": strategy})
            assert spec.eviction_strategy == strategy

    def test_confusable_protection_defaults_to_a_fifth_of_the_cap(self):
        # Not 1.0: confusable frames are evicted last, so unlimited protection
        # lets them accumulate until the queue stops rolling.
        spec = CollectorSpec.from_params({"output_dir": "out"})
        assert spec.confusable_protection_fraction == 0.2

    def test_confusable_protection_is_clamped_to_a_fraction(self):
        for raw, expected in ((-1, 0.0), (0, 0.0), (0.5, 0.5), (1, 1.0), (7, 1.0)):
            spec = CollectorSpec.from_params({"output_dir": "out", "confusable_protection_fraction": raw})
            assert spec.confusable_protection_fraction == expected

    def test_unparseable_confusable_protection_falls_back_to_the_default(self):
        spec = CollectorSpec.from_params({"output_dir": "out", "confusable_protection_fraction": "most of it"})
        assert spec.confusable_protection_fraction == 0.2


class TestSchemaGrouping:
    """The form is grouped (plan: the flat run of a dozen fields had no shape), and a
    `group` is value-transparent — GroupField merges its fields into the parent, so
    `from_params` still reads every key flat. That transparency is the thing that can
    silently break.
    """

    @staticmethod
    def _knobs(schema: dict) -> list[str]:
        """Every field a user can set, flattened across groups — in declaration order,
        so duplicates survive to be counted rather than collapsing into a set."""
        found = []
        for key, meta in schema.items():
            if key == "meta":
                continue
            if meta.get("type") == "group":
                found.extend(meta.get("fields", {}))
            else:
                found.append(key)
        return found

    def test_no_knob_is_declared_in_two_places(self):
        """Two widgets bound to one key both write it, and which one wins depends on
        render order — the saved value would then depend on which group you opened."""
        knobs = self._knobs(collector_params_schema())
        duplicated = {k for k in knobs if knobs.count(k) > 1}
        assert not duplicated, f"declared more than once: {sorted(duplicated)}"

    def test_every_group_actually_contains_something(self):
        """An empty group renders as a collapsible that opens onto nothing — the
        shape a moved-out-but-not-deleted field leaves behind."""
        schema = collector_params_schema()
        empty = [k for k, m in schema.items() if m.get("type") == "group" and not m.get("fields")]
        assert not empty, f"groups with no fields: {empty}"

    def test_grouping_does_not_nest_the_stored_params(self):
        """The whole reason grouping was safe. A spec built from flat params must still
        read a knob that now lives inside a group — if this fails, every existing
        collector row silently reverts that knob to its default."""
        spec = CollectorSpec.from_params(
            {"output_dir": "out", "max_images": 12, "overlap_iou_threshold": 0.8, "max_frames_per_track": 7}
        )

        assert (spec.max_images, spec.overlap_iou_threshold, spec.max_frames_per_track) == (12, 0.8, 7)
