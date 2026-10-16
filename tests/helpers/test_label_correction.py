"""Correcting a watcher's label: which frames, which boxes, and which rows.

Everything here is pure computation over a sidecar document — no DB, no clip. What is
worth protecting is that each rule is *conditional*, because every one of them has an
opposite case that must behave differently and none of them raises when it stops:

* `NO_BBOX` removes a box and `WEAK` leaves one alone. Collapse the two and a false
  positive is filed as training data under a label someone confirmed.
* The cap is spent on *distinct* frames. Thin after capping instead of before and eight
  slots fill with one pose from whichever burst was densest.
* Neither sentinel may touch an event row. A row rewritten from "this box shouldn't
  exist" says the event was something it never was.

The two sentinel strings are also a cross-language mirror — the browser chooses them and
sends them — so the last class asserts the two sides agree with *each other* rather than
pinning either to a literal.
"""

import re
from pathlib import Path
from types import SimpleNamespace

from apps.helpers.recordings.labelCorrection import (
    NO_BBOX,
    WEAK,
    apply_label_remap,
    dedupe_overlapping_boxes,
    event_correction_for,
    frames_holding_labels,
    pick_training_frames,
    sample_at,
    weakest_conf,
)

_JS_MODAL = (
    Path(__file__).resolve().parents[2]
    / "apps"
    / "image_tagging_server"
    / "static"
    / "js"
    / "app"
    / "nvr"
    / "events"
    / "labelCorrectionModal.js"
)

CAT = "label-cat"
DOG = "label-dog"
BOWL = "label-bowl"


def _box(label=CAT, conf=0.9, x=0.1, y=0.1, w=0.2, h=0.2):
    return {"labelUuid": label, "conf": conf, "x": x, "y": y, "w": w, "h": h}


def _doc(*frames):
    """A sidecar document from `(t, [boxes])` pairs."""
    return {"frames": [{"t": t, "boxes": list(boxes)} for t, boxes in frames]}


class TestWhichFramesHoldTheLabel:
    def test_only_the_frames_carrying_the_label_are_candidates(self):
        doc = _doc((0.0, [_box(CAT)]), (1.0, [_box(DOG)]), (2.0, [_box(CAT), _box(BOWL)]))
        assert frames_holding_labels(doc, [CAT]) == [0.0, 2.0]

    def test_no_labels_asked_for_means_no_candidates(self):
        """The guard that stops an empty remap harvesting the whole clip."""
        doc = _doc((0.0, [_box(CAT)]))
        assert frames_holding_labels(doc, []) == []

    def test_a_weak_selection_takes_only_the_frames_the_detector_hedged_on(self):
        """The conditional that makes `WEAK` worth having: the same label, the same
        frames, and a different answer — the frames it was sure about are frames of a
        label it already gets right."""
        doc = _doc((0.0, [_box(CAT, conf=0.95)]), (1.0, [_box(CAT, conf=0.3)]))
        assert frames_holding_labels(doc, [CAT], max_conf=0.6) == [1.0]

    def test_a_box_carrying_no_confidence_never_counts_as_weak(self):
        """Absent is not low. A sidecar frame whose box has no `conf` would otherwise
        read as maximally hedged and dominate the harvest."""
        doc = _doc((0.0, [{"labelUuid": CAT, "x": 0, "y": 0, "w": 0.1, "h": 0.1}]))
        assert frames_holding_labels(doc, [CAT], max_conf=0.6) == []

    def test_the_weakest_box_in_a_frame_decides_it(self):
        """A frame holding a confident box and a hedged one still holds a hedged one."""
        frame = {"t": 0.0, "boxes": [_box(CAT, conf=0.9), _box(CAT, conf=0.2)]}
        assert weakest_conf(frame, {CAT}) == 0.2
        assert frames_holding_labels(_doc((0.0, frame["boxes"])), [CAT], max_conf=0.6) == [0.0]

    def test_a_frame_with_no_time_is_not_a_candidate(self):
        """Its picture cannot be found. Dropped rather than harvested at t=0, which would
        pair a box from an unknown instant with the clip's opening frame."""
        doc = {"frames": [{"boxes": [_box(CAT)]}, {"t": 1.0, "boxes": [_box(CAT)]}]}
        assert frames_holding_labels(doc, [CAT]) == [1.0]


class TestSpreadingTheHarvest:
    def test_a_short_run_is_taken_whole(self):
        """Three candidates under a cap of eight is three frames, not eight."""
        assert pick_training_frames([0.0, 1.0, 2.0], max_frames=8, min_gap_s=0.75) == [0.0, 1.0, 2.0]

    def test_frames_closer_than_the_gap_are_thinned(self):
        """Consecutive sidecar samples show one pose. At 5 fps a burst spans barely a
        second, and filing all of it is one pose weighted tenfold."""
        times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        assert pick_training_frames(times, max_frames=8, min_gap_s=0.75) == [0.0, 0.8]

    def test_the_cap_is_spent_on_distinct_frames_not_on_the_densest_burst(self):
        """The conditional that decides the pass order. Capping before thinning would
        take the first three samples — 0.0, 0.1, 0.2 — which are one pose."""
        times = [0.0, 0.1, 0.2, 0.3, 5.0, 10.0]
        assert pick_training_frames(times, max_frames=3, min_gap_s=0.75) == [0.0, 5.0, 10.0]

    def test_the_last_pick_lands_on_the_final_frame(self):
        """Fenceposts, not buckets: the final frame is the leaving pose, and dividing by
        `room` instead of `room - 1` stops one step short of it every time."""
        times = [float(i) for i in range(10)]
        picked = pick_training_frames(times, max_frames=4, min_gap_s=0.75)
        assert picked[0] == 0.0
        assert picked[-1] == 9.0
        assert len(picked) == 4

    def test_a_cap_of_one_takes_a_single_frame(self):
        assert pick_training_frames([0.0, 5.0, 10.0], max_frames=1, min_gap_s=0.75) == [0.0]

    def test_a_cap_of_zero_harvests_nothing(self):
        assert pick_training_frames([0.0, 5.0], max_frames=0, min_gap_s=0.75) == []

    def test_no_frame_is_picked_twice(self):
        """A step below 1 rounds two indices onto one frame. The cap is a ceiling on
        distinct pictures, so the same moment must not be seeked and filed twice."""
        picked = pick_training_frames([0.0, 1.0, 2.0], max_frames=8, min_gap_s=0.75)
        assert len(picked) == len(set(picked))


class TestApplyingTheRemap:
    def test_a_relabelled_box_keeps_its_geometry(self):
        """The correction is about *what*, never about *where* — the rectangle is the
        detector's and the review queue is where it gets settled."""
        [out] = apply_label_remap([_box(CAT, x=0.3, y=0.4, w=0.1, h=0.2)], {CAT: DOG})
        assert out["labelUuid"] == DOG
        assert (out["x"], out["y"], out["w"], out["h"]) == (0.3, 0.4, 0.1, 0.2)

    def test_a_no_bbox_label_is_removed_rather_than_relabelled(self):
        assert apply_label_remap([_box(CAT)], {CAT: NO_BBOX}) == []

    def test_a_weak_label_passes_through_untouched(self):
        """The conditional against `NO_BBOX`: same dialog, same remap, opposite outcome.
        `WEAK` says the label was right, so there is nothing to change."""
        [out] = apply_label_remap([_box(CAT)], {CAT: WEAK})
        assert out["labelUuid"] == CAT

    def test_a_label_nobody_corrected_is_left_alone(self):
        """The bowl beside the cat. A single-label form would rewrite it while looking
        like it had worked."""
        boxes = [_box(CAT), _box(BOWL)]
        out = apply_label_remap(boxes, {CAT: DOG})
        assert [b["labelUuid"] for b in out] == [DOG, BOWL]

    def test_the_caller_s_boxes_are_not_rewritten_in_place(self):
        """The sidecar document is read again for the next frame, and for the next
        event in the selection."""
        boxes = [_box(CAT)]
        apply_label_remap(boxes, {CAT: DOG})
        assert boxes[0]["labelUuid"] == CAT


class TestDedupingBoxes:
    def test_one_object_described_twice_is_filed_once(self):
        """A sidecar frame carries the detections *and* the tracks, so the same animal
        routinely appears twice. Two boxes hands the trainer one object annotated twice."""
        boxes = [_box(CAT, conf=0.5), _box(CAT, conf=0.9)]
        assert len(dedupe_overlapping_boxes(boxes, iou_threshold=0.75)) == 1

    def test_the_more_confident_box_is_the_one_kept(self):
        boxes = [_box(CAT, conf=0.5), _box(CAT, conf=0.9)]
        [kept] = dedupe_overlapping_boxes(boxes, iou_threshold=0.75)
        assert kept["conf"] == 0.9

    def test_two_animals_side_by_side_both_survive(self):
        """The conditional: same label, no overlap, both kept. A dedupe that collapsed
        these would silently halve a two-cat frame."""
        boxes = [_box(CAT, x=0.0, y=0.0), _box(CAT, x=0.6, y=0.6)]
        assert len(dedupe_overlapping_boxes(boxes, iou_threshold=0.75)) == 2

    def test_overlapping_boxes_of_different_labels_both_survive(self):
        """A cat in front of its bowl is two annotations, not a duplicate."""
        boxes = [_box(CAT), _box(BOWL)]
        assert len(dedupe_overlapping_boxes(boxes, iou_threshold=0.75)) == 2

    def test_a_remap_that_collapses_two_labels_onto_one_dedupes_afterwards(self):
        """The reason the two passes run in this order. Before the remap these are
        different labels and not duplicates; after it they are one object twice."""
        boxes = [_box(CAT, conf=0.9), _box(DOG, conf=0.4)]
        remapped = apply_label_remap(boxes, {DOG: CAT})
        assert len(dedupe_overlapping_boxes(remapped, iou_threshold=0.75)) == 1


class TestFindingTheSample:
    def test_the_frame_at_that_exact_time_is_returned(self):
        doc = _doc((0.0, [_box(CAT)]), (1.5, [_box(DOG)]))
        assert sample_at(doc, 1.5)["boxes"][0]["labelUuid"] == DOG

    def test_a_time_no_sample_describes_returns_nothing(self):
        """Rather than the nearest. A box from a neighbouring instant describes
        somewhere the subject no longer is."""
        assert sample_at(_doc((0.0, [_box(CAT)])), 1.5) is None


class TestCorrectingTheEventRow:
    def _event(self, **overrides):
        fields = {
            "uuid": "event-1",
            "subject_ref": CAT,
            "subject_name": "Clover",
            "target_ref": None,
            "target_name": None,
            "rule_name": "cat at bowl",
        }
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def test_the_subject_is_rewritten_with_its_name(self):
        """The name travels with the ref, always: a row that filters as one label and
        reads as another is the worst of the three states."""
        correction = event_correction_for(self._event(), {CAT: DOG}, {DOG: "Luna"})
        assert correction["patch"] == {"subject_ref": DOG, "subject_name": "Luna"}

    def test_both_slots_can_be_corrected_at_once(self):
        event = self._event(target_ref=BOWL, target_name="Bowl")
        correction = event_correction_for(event, {CAT: DOG, BOWL: CAT}, {DOG: "Luna", CAT: "Clover"})
        assert correction["patch"] == {
            "subject_ref": DOG,
            "subject_name": "Luna",
            "target_ref": CAT,
            "target_name": "Clover",
        }

    def test_a_remap_touching_neither_slot_offers_nothing(self):
        """Correcting a label that was never the event's own subject — the common case.
        Returning a patch here would offer to change nothing."""
        assert event_correction_for(self._event(), {BOWL: DOG}, {}) is None

    def test_no_bbox_never_rewrites_the_row(self):
        """ "This box shouldn't exist" says nothing about what actually happened, so
        there is no corrected identity to write."""
        assert event_correction_for(self._event(), {CAT: NO_BBOX}, {}) is None

    def test_weak_never_rewrites_the_row(self):
        """The opposite reason to `NO_BBOX`, same outcome: the identity the row already
        carries is the one being confirmed."""
        assert event_correction_for(self._event(), {CAT: WEAK}, {}) is None

    def test_an_unnamed_label_falls_back_to_its_uuid(self):
        """A row must never be left with a ref and no name."""
        correction = event_correction_for(self._event(), {CAT: DOG}, {})
        assert correction["patch"]["subject_name"] == DOG


class TestFrontendMirrorMatches:
    """The two sentinels exist twice — the browser picks them and sends them, and this
    module decides what they mean — because no build step shares constants between the
    languages. Divergence is silent: a `NO_BBOX` the task does not recognise is treated
    as a label uuid, so a false positive is relabelled to a label that does not exist
    instead of being removed.

    Asserted against each other, never pinned to a literal.
    """

    def _js_const(self, name: str) -> str:
        match = re.search(rf"^export const {name} = '([^']+)';", _JS_MODAL.read_text(), re.MULTILINE)
        assert match, f"{name} not found in {_JS_MODAL.name}"
        return match.group(1)

    def test_no_bbox_matches(self):
        assert self._js_const("NO_BBOX") == NO_BBOX

    def test_weak_matches(self):
        assert self._js_const("WEAK") == WEAK

    def test_the_two_sentinels_are_distinguishable(self):
        """They mean opposite things — remove the box, keep the box — so one value for
        both would file every false positive as confirmed training data."""
        assert NO_BBOX != WEAK
