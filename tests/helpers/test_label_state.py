"""Unit tests for compute_image_label_states — apps/helpers/db/label_state.py.

These are pure-function tests: no database, no HTTP, no I/O.
The function is the extracted core of Manager._recompute_image_labels.
"""
from apps.helpers.db.label_state import compute_image_label_states
from apps.helpers.db.types import (
    FALSE_POS_TAG_UUID,
    UNVERIFIED_TAG_UUID,
    VERIFIED_TAG_UUID,
    ImageLabelStatus,
)


def test_empty_bboxes_returns_empty():
    assert compute_image_label_states([]) == {}


def test_bbox_no_label_ignored():
    result = compute_image_label_states([(None, {VERIFIED_TAG_UUID})])
    assert result == {}


def test_empty_label_uuid_ignored():
    result = compute_image_label_states([("", {VERIFIED_TAG_UUID})])
    assert result == {}


def test_single_verified_bbox_gives_present():
    result = compute_image_label_states([("label-1", {VERIFIED_TAG_UUID})])
    assert result["label-1"] == ImageLabelStatus.PRESENT


def test_single_unverified_bbox_gives_uncertain():
    result = compute_image_label_states([("label-1", {UNVERIFIED_TAG_UUID})])
    assert result["label-1"] == ImageLabelStatus.UNCERTAIN


def test_single_false_pos_bbox_gives_absent():
    result = compute_image_label_states([("label-1", {FALSE_POS_TAG_UUID})])
    assert result["label-1"] == ImageLabelStatus.ABSENT


def test_all_false_pos_multiple_bboxes_gives_absent():
    result = compute_image_label_states([
        ("label-1", {FALSE_POS_TAG_UUID}),
        ("label-1", {FALSE_POS_TAG_UUID}),
    ])
    assert result["label-1"] == ImageLabelStatus.ABSENT


def test_any_verified_beats_all_false_pos_gives_present():
    result = compute_image_label_states([
        ("label-1", {FALSE_POS_TAG_UUID}),
        ("label-1", {VERIFIED_TAG_UUID}),
    ])
    assert result["label-1"] == ImageLabelStatus.PRESENT


def test_verified_beats_unverified():
    result = compute_image_label_states([
        ("label-1", {UNVERIFIED_TAG_UUID}),
        ("label-1", {VERIFIED_TAG_UUID}),
    ])
    assert result["label-1"] == ImageLabelStatus.PRESENT


def test_mixed_false_pos_and_unverified_gives_uncertain():
    # Not all false-pos, and no verified → UNCERTAIN
    result = compute_image_label_states([
        ("label-1", {FALSE_POS_TAG_UUID}),
        ("label-1", {UNVERIFIED_TAG_UUID}),
    ])
    assert result["label-1"] == ImageLabelStatus.UNCERTAIN


def test_no_tags_gives_uncertain():
    result = compute_image_label_states([("label-1", set())])
    assert result["label-1"] == ImageLabelStatus.UNCERTAIN


def test_multiple_labels_independent():
    result = compute_image_label_states([
        ("cat", {VERIFIED_TAG_UUID}),
        ("dog", {FALSE_POS_TAG_UUID}),
        ("bird", {UNVERIFIED_TAG_UUID}),
    ])
    assert result["cat"] == ImageLabelStatus.PRESENT
    assert result["dog"] == ImageLabelStatus.ABSENT
    assert result["bird"] == ImageLabelStatus.UNCERTAIN


def test_multiple_bboxes_same_label_one_verified():
    result = compute_image_label_states([
        ("cat", {UNVERIFIED_TAG_UUID}),
        ("cat", {UNVERIFIED_TAG_UUID}),
        ("cat", {VERIFIED_TAG_UUID}),  # one verified → PRESENT
    ])
    assert result["cat"] == ImageLabelStatus.PRESENT


def test_extra_tags_do_not_affect_logic():
    # A bbox can have multiple tags; the logic only cares about specific UUIDs
    RANDOM_TAG = "aaaaaaaa-0000-0000-0000-000000000001"
    result = compute_image_label_states([
        ("label-1", {RANDOM_TAG, VERIFIED_TAG_UUID}),
    ])
    assert result["label-1"] == ImageLabelStatus.PRESENT
