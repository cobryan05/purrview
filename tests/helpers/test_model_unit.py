"""Tests for resolving a model config into the artifacts that implement it.

This is what makes a hierarchy usable as one model: the operator selects a config and
these functions work out which files to run, in what order, without asking the database.
A regression here either drops a pass (detections silently stop at a parent label) or
runs a classifier as a detector (plausible boxes that mean nothing).
"""

from apps.helpers.cascade import CropSubModel, order_sub_models
from apps.helpers.modelUnit import intra_unit_ancestry, resolve_unit

CFG = "cfg-1"


def _top(name="top_model", **kw):
    return {"model_config_uuid": CFG, "role": "top", "model_kind": "object_detection", **kw}


def _sub(parent, node="node-1", **kw):
    return {
        "model_config_uuid": CFG,
        "role": "submodel",
        "model_kind": "crop_classifier",
        "parent_label_uuid": parent,
        "model_config_node_uuid": node,
        **kw,
    }


class TestResolveUnit:
    def test_splits_a_config_into_detectors_and_sub_models(self):
        unit = resolve_unit({"top_model": _top(), "cat_namer": _sub("cat")}, CFG)
        assert unit.detector_names == ["top_model"]
        assert unit.sub_models == [CropSubModel(model_name="cat_namer", parent_label_uuid="cat")]
        assert unit.problems == []

    def test_ignores_artifacts_from_other_configs(self):
        """Two hierarchies in one models directory must not bleed into each other —
        the sidecars sit side by side with nothing but this uuid separating them."""
        other = dict(_sub("dog"), model_config_uuid="cfg-2")
        unit = resolve_unit({"top_model": _top(), "cat_namer": _sub("cat"), "dog_namer": other}, CFG)
        assert [s.model_name for s in unit.sub_models] == ["cat_namer"]
        assert "dog_namer" not in unit.detector_names

    def test_sub_model_without_a_parent_label_is_dropped_and_reported(self):
        """Running it as a detector is the failure mode being prevented: the detection
        endpoint accepts a classifier and returns meaningless boxes rather than erroring."""
        unit = resolve_unit({"top_model": _top(), "orphan": _sub(None)}, CFG)
        assert unit.sub_models == []
        assert unit.detector_names == ["top_model"]
        assert any("orphan" in p for p in unit.problems)

    def test_two_artifacts_for_one_node_keeps_only_the_newest(self):
        """Retraining a node under a new name leaves the old file in place. Running both
        puts two disagreeing opinions of the same boundary in one stack, and which won
        would depend on directory order."""
        unit = resolve_unit(
            {
                "top_model": _top(),
                "cat_v1": _sub("cat", node="n1", trained_at="2026-01-01T00:00:00Z"),
                "cat_v2": _sub("cat", node="n1", trained_at="2026-06-01T00:00:00Z"),
            },
            CFG,
        )
        assert [s.model_name for s in unit.sub_models] == ["cat_v2"]
        assert any("cat_v1" in p for p in unit.problems)

    def test_different_nodes_both_run(self):
        """The same guard must not collapse a genuine two-level hierarchy."""
        unit = resolve_unit(
            {"top_model": _top(), "animal_namer": _sub("animal", node="n1"), "cat_namer": _sub("cat", node="n2")},
            CFG,
        )
        assert sorted(s.model_name for s in unit.sub_models) == ["animal_namer", "cat_namer"]

    def test_legacy_artifact_without_role_falls_back_to_model_kind(self):
        """Artifacts trained before `role` was stamped still exist and are runnable."""
        legacy = {
            "model_config_uuid": CFG,
            "model_kind": "crop_classifier",
            "parent_label_uuid": "cat",
        }
        unit = resolve_unit({"top_model": _top(), "old_cat": legacy}, CFG)
        assert [s.model_name for s in unit.sub_models] == ["old_cat"]

    def test_config_with_no_artifacts_resolves_empty_rather_than_raising(self):
        """An untrained config is a normal state (it is what you have before the first
        run), so it must come back empty for the caller to report, not explode."""
        unit = resolve_unit({"top_model": _top()}, "cfg-never-trained")
        assert unit.is_empty
        assert unit.parts == []


class TestIntraUnitAncestry:
    def test_classes_inherit_their_boundary(self):
        anc = intra_unit_ancestry({"cat_namer": ["willow", "luna"]}, {"cat_namer": "cat"})
        assert anc["willow"] == {"cat"}
        assert anc["luna"] == {"cat"}

    def test_nested_units_chain_without_the_label_tree(self):
        """The point of the whole module: animal -> cat -> willow is reconstructed from
        two sidecars, so inference_server needs no database to order the passes."""
        anc = intra_unit_ancestry(
            {"animal_namer": ["cat"], "cat_namer": ["willow"]},
            {"animal_namer": "animal", "cat_namer": "cat"},
        )
        assert anc["willow"] == {"cat", "animal"}
        assert anc["cat"] == {"animal"}

    def test_derived_ancestry_orders_the_passes_outermost_first(self):
        """Ancestry exists to feed order_sub_models; asserting the map alone would not
        prove the passes actually run in an order that resolves the chain."""
        subs = [
            CropSubModel(model_name="cat_namer", parent_label_uuid="cat"),
            CropSubModel(model_name="animal_namer", parent_label_uuid="animal"),
        ]
        anc = intra_unit_ancestry(
            {"animal_namer": ["cat"], "cat_namer": ["willow"]},
            {"animal_namer": "animal", "cat_namer": "cat"},
        )
        assert [s.parent_label_uuid for s in order_sub_models(subs, anc)] == ["animal", "cat"]

    def test_caller_supplied_ancestry_is_kept(self):
        """The watcher passes the real label tree so a detection emitted by a foreign
        model still qualifies for refinement; deriving must add to that, not replace it."""
        anc = intra_unit_ancestry(
            {"cat_namer": ["willow"]},
            {"cat_namer": "cat"},
            base={"stray": {"cat"}},
        )
        assert anc["stray"] == {"cat"}
        assert anc["willow"] == {"cat"}

    def test_a_cycle_degrades_instead_of_hanging(self):
        """Two corrupt sidecars naming each other's boundary must cost a level of
        hierarchy, not the process."""
        anc = intra_unit_ancestry(
            {"a": ["y"], "b": ["x"]},
            {"a": "x", "b": "y"},
        )
        assert anc["y"] == {"x", "y"}
