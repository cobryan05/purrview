"""Regression tests for DeduplicateOverlappingBBoxesTask._find_redundant_bboxes.

Guards the camelCase-key contract: the bbox dicts come from
`batch_images_by_uuids`, which serializes with `alias_generator=to_camel`, so the
label key is `"labelUuid"` (not `"label_uuid"`). Reading the wrong key silently
returns None for every label, which disables all same-label / ancestor-label
matching and leaves only pure-IoU dedup — a silent correctness bug this test
locks against.
"""

from apps.tasks_server.tasks.DeduplicateOverlappingBBoxesTask import (
    _find_redundant_bboxes,
)


def _box(uuid, label, x=0, y=0, w=10, h=10):
    return {"uuid": uuid, "labelUuid": label, "x": x, "y": y, "width": w, "height": h}


def test_same_label_overlap_deletes_later():
    # Two fully-overlapping bboxes with the SAME label: the later one is redundant.
    # Before the labelUuid fix, both labels read as None -> "unrelated, keep both"
    # -> this returned an empty set.
    bboxes = [_box("a", "cat"), _box("b", "cat")]
    assert _find_redundant_bboxes(bboxes, {}, iou_threshold=0.5) == {"b"}


def test_ancestor_label_overlap_deletes_ancestor():
    # "animal" is an ancestor of "cat"; the ancestor-labelled bbox is the redundant one.
    bboxes = [_box("a", "animal"), _box("b", "cat")]
    ancestry = {"cat": {"animal"}, "animal": set()}
    assert _find_redundant_bboxes(bboxes, ancestry, iou_threshold=0.5) == {"a"}


def test_unrelated_labels_keep_both():
    bboxes = [_box("a", "cat"), _box("b", "dog")]
    assert _find_redundant_bboxes(bboxes, {"cat": set(), "dog": set()}, iou_threshold=0.5) == set()


def test_non_overlapping_same_label_keep_both():
    bboxes = [_box("a", "cat", x=0), _box("b", "cat", x=100)]
    assert _find_redundant_bboxes(bboxes, {}, iou_threshold=0.5) == set()
