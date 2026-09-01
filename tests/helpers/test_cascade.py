"""Tests for the hierarchical second-pass (sub-model) resolution helpers.

These decide, at inference time, which detections a crop classifier is handed and
what its verdict means. Both the live watcher and the retroactive preview path run
on this, so a regression here is a regression in both.
"""

import pytest

from apps.helpers.cascade import (
    CROP_CLASSIFIER_KIND,
    CropSubModel,
    order_sub_models,
    refinable_indices,
    resolve_refinement,
    split_model_stack,
)


class TestSplitModelStack:
    def test_crop_classifier_leaves_the_detector_list(self):
        """The whole point: a crop classifier must not reach the detection endpoint.

        It would answer rather than error there — a classifier run as a detector
        produces boxes that mean nothing — so this split is the only thing standing
        between a configured sub-model and silent garbage on every frame.
        """
        detectors, subs = split_model_stack(
            ["yolo_cats", "cat_namer"],
            {"yolo_cats": "object_detection", "cat_namer": CROP_CLASSIFIER_KIND},
            {"cat_namer": "cat-uuid"},
        )
        assert detectors == ["yolo_cats"]
        assert subs == [CropSubModel(model_name="cat_namer", parent_label_uuid="cat-uuid")]

    def test_detector_order_is_preserved(self):
        """Detector order is the watcher's stack priority (FIRST_WINS merge), so the
        split must not reorder it while filtering."""
        detectors, _ = split_model_stack(
            ["c", "sub", "a", "b"],
            {"c": "object_detection", "a": "object_detection", "b": "detector", "sub": CROP_CLASSIFIER_KIND},
            {"sub": "p"},
        )
        assert detectors == ["c", "a", "b"]

    def test_parentless_crop_classifier_is_dropped_from_both_lists(self):
        """A sub-model with no boundary label cannot be aimed at anything. Leaving it
        among the detectors would run it as one; leaving it among the sub-models
        would refine nothing. It belongs in neither."""
        detectors, subs = split_model_stack(
            ["yolo", "orphan"],
            {"yolo": "object_detection", "orphan": CROP_CLASSIFIER_KIND},
            {"orphan": None},
        )
        assert detectors == ["yolo"]
        assert subs == []

    def test_unknown_kind_is_treated_as_a_detector(self):
        """Metadata can fail to load (a remote inference server, a missing sidecar).
        The fallback must be today's behaviour — run it as a detector — not "drop it",
        which would silently stop detecting on that camera."""
        detectors, subs = split_model_stack(["mystery"], {}, {})
        assert detectors == ["mystery"]
        assert subs == []


class TestRefinableIndices:
    ANCESTRY = {
        "willow": {"cat"},
        "maple": {"cat"},
        "cat": set(),
        "dog": set(),
        "puppy": {"dog"},
    }

    def test_boundary_label_itself_is_selected(self):
        assert refinable_indices(["cat"], "cat", self.ANCESTRY) == [0]

    def test_descendants_are_selected_too(self):
        """A stack can already emit a specific label (an older per-cat detector
        running alongside the collapsed one). Re-classifying it is the second
        opinion the cascade exists to give, so it must not be skipped."""
        assert refinable_indices(["willow", "maple"], "cat", self.ANCESTRY) == [0, 1]

    def test_unrelated_and_missing_labels_are_left_alone(self):
        """The sub-model knows nothing about dogs; handing it a dog crop would get a
        confident cat name back."""
        labels = ["cat", "dog", None, "puppy", "willow"]
        assert refinable_indices(labels, "cat", self.ANCESTRY) == [0, 4]

    def test_returns_positions_not_labels(self):
        """Callers pair results back to detections by index, so the contract is
        positional — a filtered list of labels would be unusable."""
        assert refinable_indices(["dog", "dog", "cat"], "cat", self.ANCESTRY) == [2]


class TestOrderSubModels:
    """Nesting depth is unbounded, but only if the passes run outside-in."""

    # animal -> cat -> breed -> (persian, siamese)
    ANCESTRY = {
        "cat": {"animal"},
        "breed": {"cat", "animal"},
        "persian": {"breed", "cat", "animal"},
        "animal": set(),
    }

    def test_outermost_boundary_runs_first(self):
        """Configured order carries no hierarchy information — it is whatever order
        the operator added models to the stack. Run inner-first, the inner pass finds
        nothing to refine yet and the detection silently stops one level short."""
        configured = [
            CropSubModel(model_name="breed_namer", parent_label_uuid="breed"),
            CropSubModel(model_name="animal_namer", parent_label_uuid="animal"),
            CropSubModel(model_name="cat_namer", parent_label_uuid="cat"),
        ]
        ordered = order_sub_models(configured, self.ANCESTRY)
        assert [s.parent_label_uuid for s in ordered] == ["animal", "cat", "breed"]

    def test_siblings_keep_their_configured_order(self):
        """Same depth means neither feeds the other, so there is no reason to
        reorder them — and a stable order keeps runs reproducible."""
        configured = [
            CropSubModel(model_name="b", parent_label_uuid="cat"),
            CropSubModel(model_name="a", parent_label_uuid="cat"),
        ]
        assert [s.model_name for s in order_sub_models(configured, self.ANCESTRY)] == ["b", "a"]

    def test_unknown_ancestry_sorts_first_without_raising(self):
        """A label missing from the map (deleted mid-run, or a stale sidecar) must not
        take the frame down."""
        configured = [
            CropSubModel(model_name="deep", parent_label_uuid="persian"),
            CropSubModel(model_name="mystery", parent_label_uuid="not-in-map"),
        ]
        ordered = order_sub_models(configured, self.ANCESTRY)
        assert [s.model_name for s in ordered] == ["mystery", "deep"]


class TestArbitraryNestingDepth:
    """Three levels resolve in one pass sequence: animal -> cat -> willow.

    Each pass reads the labels the previous pass wrote, so depth is bounded only by
    how many sub-models are configured — there is no fixed limit in the code.
    """

    ANCESTRY = {
        "cat": {"animal"},
        "willow": {"cat", "animal"},
        "animal": set(),
    }

    def test_each_level_feeds_the_next(self):
        labels = ["animal"]

        # Level 1: the animal sub-model says "cat".
        idx = refinable_indices(labels, "animal", self.ANCESTRY)
        assert idx == [0]
        refined = resolve_refinement(1.0, [("cat", 1.0)], 0.5, current_label_uuid=labels[0])
        assert refined is not None
        labels[0] = refined[0]
        assert labels == ["cat"]

        # Level 2: the cat sub-model now matches the label level 1 produced.
        idx = refinable_indices(labels, "cat", self.ANCESTRY)
        assert idx == [0], "the second level found nothing — the chain broke"
        refined = resolve_refinement(1.0, [("willow", 1.0)], 0.5, current_label_uuid=labels[0])
        assert refined is not None
        labels[0] = refined[0]
        assert labels == ["willow"]

    def test_a_descendant_label_still_matches_an_ancestor_boundary(self):
        """The reason the chain works at all: after level 1 rewrites the label, the
        deeper sub-model's boundary is an ancestor of it, not equal to it."""
        assert refinable_indices(["willow"], "animal", self.ANCESTRY) == [0]


class TestResolveRefinement:
    def test_top_class_wins_regardless_of_input_order(self):
        got = resolve_refinement(1.0, [("maple", 0.2), ("willow", 0.7), ("luna", 0.1)], min_confidence=0.5)
        assert got is not None
        assert got[0] == "willow"

    def test_declines_when_the_classifier_is_unsure(self):
        """An unsure second pass must leave the coarse label in place. "A cat, not
        sure which" is true and useful; naming the wrong cat in a notification is
        worse than not naming one."""
        assert resolve_refinement(0.9, [("willow", 0.51), ("maple", 0.49)], min_confidence=0.6) is None

    def test_confidence_is_the_product_of_both_passes(self):
        """A weak detection must not come back strong just because the classifier was
        sure about the crop — every downstream confidence band would then be reading
        a number that ignores whether there was an animal there at all."""
        got = resolve_refinement(0.4, [("willow", 0.9)], min_confidence=0.5)
        assert got == ("willow", 0.4 * 0.9)

    def test_no_predictions_declines(self):
        assert resolve_refinement(0.9, [], min_confidence=0.0) is None

    def test_agreeing_with_the_current_label_declines(self):
        """A no-op relabel must be reported as "no change", not applied.

        Applying it would multiply the confidence for a label that never changed —
        a 0.8 detection coming back at 0.48 can fall out of a watch rule's
        confidence band without anything about it having actually moved.
        """
        assert resolve_refinement(0.8, [("cat", 0.9)], min_confidence=0.6, current_label_uuid="cat") is None

    def test_still_refines_when_the_label_does_change(self):
        got = resolve_refinement(0.8, [("willow", 0.9)], min_confidence=0.6, current_label_uuid="cat")
        assert got == ("willow", pytest.approx(0.8 * 0.9))
