"""Tests for `apps.helpers.training.collectorSpec`.

The schema tests here guard the one invariant that makes the split worth doing:
the ~30 collect knobs have a single definition, rendered by two different forms.
A knob that exists but never renders, or a driver-only param that leaks into the
collector editor, are both silent — nothing raises, the form is just wrong.
"""

from apps.helpers.training.collectorSpec import (
    AMBIGUITY_METRIC_MARGIN,
    AMBIGUITY_METRIC_RATIO,
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

    def test_task_schema_is_the_collectors_plus_its_loop_params(self):
        task = CollectTrainingImagesTask.params_schema()
        collector = collector_params_schema()
        assert set(task) - {"meta"} == (set(collector) - {"meta"}) | DRIVER_PARAMS
        for key, defn in collector.items():
            if key != "meta":
                assert task[key] == defn, f"{key} drifted between the two renderers"

    def test_task_keeps_the_collectors_relative_ordering(self):
        """Splicing the loop params in must not reshuffle the rest of the form."""
        task_order = CollectTrainingImagesTask.params_schema()["meta"]["order"]
        collector_order = collector_params_schema()["meta"]["order"]
        assert [k for k in task_order if k not in DRIVER_PARAMS] == collector_order
        assert DRIVER_PARAMS.issubset(task_order)


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
