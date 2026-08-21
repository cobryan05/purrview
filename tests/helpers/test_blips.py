"""Finding bad inference in a clip's detection sidecar.

The bug that matters most here is **misattribution**, and it is silent. A gap where the
detector went blind and a gap where the tracker dropped a thread look identical in the
video and identical in a naive reading of the sidecar — but one is prime training data
and the other is a frame the model already gets right. Harvest the second kind and you
quietly pad the training set with cases that teach nothing, which shows up months later
as a model that stopped improving.

So most of these tests are *conditional*: the same gap, differing only in whether a raw
detection was present, must produce a different kind, a different fault, and a different
answer to `harvestable`. A test that merely asserted "a blip was found" would pass
against a version that got every attribution backwards.

Pure unit tests over a JSON document. No clip, no database, no inference.
"""

import pytest

from apps.helpers.recordings.blips import (
    FAULT_DETECTOR,
    FAULT_TRACKER,
    KIND_CONFIDENCE_DIP,
    KIND_DETECTOR_DROPOUT,
    KIND_FULL_DROPOUT,
    KIND_GHOST,
    KIND_IDENTITY_FLIP,
    KIND_IDENTITY_SWITCH,
    KIND_OCCLUSION_MERGE,
    KIND_TRACKER_DROPOUT,
    BlipThresholds,
    analyse_overlay,
)

CAT_A = "cat_a"
CAT_B = "cat_b"
BOX = "litter_box"

#: Sample spacing. 5 fps is the watcher's default analysis rate.
STEP = 0.2


def box(label=CAT_A, conf=0.9, x=0.10, y=0.10, w=0.20, h=0.20):
    return {"x": x, "y": y, "w": w, "h": h, "label": label, "labelUuid": label, "conf": conf}


def track(track_id=7, **kwargs):
    return {**box(**kwargs), "id": track_id}


def doc(frames, step=STEP):
    """Wrap per-frame `{boxes, tracks}` dicts into an overlay document."""
    return {
        "version": 1,
        "durationS": round(len(frames) * step, 4),
        "frames": [
            {"t": round(index * step, 4), "boxes": f.get("boxes", []), "tracks": f.get("tracks", [])}
            for index, f in enumerate(frames)
        ],
    }


def steady(count, label=CAT_A, conf=0.9, track_id=7):
    """`count` frames of one well-behaved track — detection and track agreeing."""
    return [{"boxes": [box(label=label, conf=conf)], "tracks": [track(track_id, label=label, conf=conf)]}] * count


def kinds_of(blips):
    return [b.kind for b in blips]


class TestATrackThatBehaves:
    def test_a_clean_track_produces_nothing(self):
        assert analyse_overlay(doc(steady(20))) == []

    def test_a_document_with_one_frame_produces_nothing(self):
        assert analyse_overlay(doc(steady(1))) == []

    def test_a_track_seen_too_few_times_is_not_analysed(self):
        """Below `min_track_samples` the modal label is not a majority, and treating a
        near-tie as truth manufactures flips that were never there."""
        frames = steady(3) + steady(2, label=CAT_B)
        assert analyse_overlay(doc(frames)) == []

    def test_a_track_split_evenly_between_two_labels_reports_no_flip(self):
        """No majority means no inferable truth. Picking the first-seen label would
        invent a flip out of a genuine ambiguity."""
        frames = steady(8) + steady(8, label=CAT_B)
        assert KIND_IDENTITY_FLIP not in kinds_of(analyse_overlay(doc(frames)))


class TestIdentity:
    def test_a_label_that_departs_and_returns_is_a_flip(self):
        frames = steady(8) + steady(4, label=CAT_B) + steady(8)
        blips = [b for b in analyse_overlay(doc(frames)) if b.kind == KIND_IDENTITY_FLIP]

        assert len(blips) == 1
        assert blips[0].fault == FAULT_DETECTOR
        assert blips[0].detail["modalLabel"] == CAT_A
        assert blips[0].detail["sawLabel"] == CAT_B
        assert blips[0].harvestable

    def test_a_flip_spans_only_the_frames_that_disagreed(self):
        frames = steady(8) + steady(4, label=CAT_B) + steady(8)
        flip = next(b for b in analyse_overlay(doc(frames)) if b.kind == KIND_IDENTITY_FLIP)

        assert flip.start_s == pytest.approx(8 * STEP)
        assert flip.end_s == pytest.approx(11 * STEP)

    def test_a_label_that_changes_and_stays_changed_is_a_switch_blamed_on_tracking(self):
        """Track-id reuse across two objects, not the detector misreading one — so it
        must not become training data."""
        frames = steady(12) + steady(8, label=CAT_B)
        blips = [b for b in analyse_overlay(doc(frames)) if b.kind == KIND_IDENTITY_SWITCH]

        assert len(blips) == 1
        assert blips[0].fault == FAULT_TRACKER
        assert not blips[0].harvestable

    def test_a_single_frame_blink_is_below_the_conservative_floor(self):
        frames = steady(8) + steady(1, label=CAT_B) + steady(8)
        assert KIND_IDENTITY_FLIP not in kinds_of(analyse_overlay(doc(frames)))

    def test_lowering_the_floor_surfaces_the_same_blink(self):
        """The floor is the only thing suppressing it — proves the detector works and
        the threshold is genuinely doing the filtering."""
        frames = steady(8) + steady(1, label=CAT_B) + steady(8)
        loose = BlipThresholds(min_flip_samples=1)

        assert KIND_IDENTITY_FLIP in kinds_of(analyse_overlay(doc(frames), loose))

    def test_a_confidently_wrong_flip_outranks_an_unsure_one(self):
        """Severity encodes how sure the detector was about what it got wrong, because
        confidently-wrong frames are the most valuable training data in the pipeline."""
        sure = steady(8) + steady(4, label=CAT_B, conf=0.95) + steady(8)
        unsure = steady(8) + steady(4, label=CAT_B, conf=0.30) + steady(8)

        sure_flip = next(b for b in analyse_overlay(doc(sure)) if b.kind == KIND_IDENTITY_FLIP)
        unsure_flip = next(b for b in analyse_overlay(doc(unsure)) if b.kind == KIND_IDENTITY_FLIP)

        assert sure_flip.severity > unsure_flip.severity


class TestWhoActuallyFailed:
    """The distinction the whole feature rests on. Same gap, three different causes."""

    def test_a_coasting_track_with_no_detection_blames_the_detector(self):
        blind = [{"boxes": [], "tracks": [track(7)]}] * 4
        blips = analyse_overlay(doc(steady(8) + blind + steady(8)))
        dropouts = [b for b in blips if b.kind == KIND_DETECTOR_DROPOUT]

        assert len(dropouts) == 1
        assert dropouts[0].fault == FAULT_DETECTOR
        assert dropouts[0].harvestable

    def test_a_vanished_track_over_a_live_detection_blames_the_tracker(self):
        lost = [{"boxes": [box()], "tracks": []}] * 4
        blips = analyse_overlay(doc(steady(8) + lost + steady(8)))
        dropouts = [b for b in blips if b.kind == KIND_TRACKER_DROPOUT]

        assert len(dropouts) == 1
        assert dropouts[0].fault == FAULT_TRACKER

    def test_a_tracker_fault_is_never_harvestable(self):
        """Harvesting it would pad the training set with frames the detector already
        got right — the failure this attribution exists to prevent."""
        lost = [{"boxes": [box()], "tracks": []}] * 4
        blips = analyse_overlay(doc(steady(8) + lost + steady(8)))

        assert all(not b.harvestable for b in blips if b.fault == FAULT_TRACKER)

    def test_nothing_at_all_is_a_full_dropout_blamed_on_the_detector(self):
        gone = [{"boxes": [], "tracks": []}] * 4
        blips = analyse_overlay(doc(steady(8) + gone + steady(8)))
        dropouts = [b for b in blips if b.kind == KIND_FULL_DROPOUT]

        assert len(dropouts) == 1
        assert dropouts[0].fault == FAULT_DETECTOR
        assert dropouts[0].harvestable

    def test_the_same_gap_is_attributed_differently_by_the_detections_alone(self):
        """The one test that would fail if attribution were inverted: identical gaps,
        differing only in whether a raw box was present."""
        gap_with_box = [{"boxes": [box()], "tracks": []}] * 4
        gap_without = [{"boxes": [], "tracks": []}] * 4

        with_box = analyse_overlay(doc(steady(8) + gap_with_box + steady(8)))
        without = analyse_overlay(doc(steady(8) + gap_without + steady(8)))

        assert [b.fault for b in with_box] == [FAULT_TRACKER]
        assert [b.fault for b in without] == [FAULT_DETECTOR]

    def test_a_gap_shorter_than_the_floor_is_ignored(self):
        blind = [{"boxes": [], "tracks": [track(7)]}] * 2
        assert analyse_overlay(doc(steady(8) + blind + steady(8))) == []

    def test_a_worse_miss_scores_higher_than_a_marginal_one(self):
        """Losing a subject held at 0.95 is a worse failure than losing one at 0.3."""
        gone = [{"boxes": [], "tracks": []}] * 4
        confident = analyse_overlay(doc(steady(8, conf=0.95) + gone + steady(8, conf=0.95)))
        marginal = analyse_overlay(doc(steady(8, conf=0.30) + gone + steady(8, conf=0.30)))

        assert confident[0].severity > marginal[0].severity


class TestOcclusion:
    def test_a_track_absorbed_by_an_overlapping_neighbour_is_flagged(self):
        """The litter-box case: two boxes overlap heavily and one stops being reported
        while the other carries on."""
        together = [
            {
                "boxes": [box(), box(label=BOX)],
                "tracks": [track(7), track(12, label=BOX)],
            }
        ] * 8
        alone = [{"boxes": [box(label=BOX)], "tracks": [track(12, label=BOX)]}] * 8

        blips = analyse_overlay(doc(together + alone))
        merges = [b for b in blips if b.kind == KIND_OCCLUSION_MERGE]

        assert len(merges) == 1
        assert merges[0].track_id == 7
        assert merges[0].detail["absorbedByTrackId"] == 12
        assert merges[0].harvestable

    def test_two_tracks_that_never_overlap_are_not_a_merge(self):
        apart = [
            {
                "boxes": [box(), box(label=BOX, x=0.70)],
                "tracks": [track(7), track(12, label=BOX, x=0.70)],
            }
        ] * 8
        alone = [{"boxes": [box(label=BOX, x=0.70)], "tracks": [track(12, label=BOX, x=0.70)]}] * 8

        assert KIND_OCCLUSION_MERGE not in kinds_of(analyse_overlay(doc(apart + alone)))


class TestKindsThatAreOffByDefault:
    def test_a_short_lived_track_is_not_reported_under_the_conservative_default(self):
        frames = steady(20)
        frames[5] = {
            "boxes": [box(), box(label=CAT_B, x=0.60)],
            "tracks": [track(7), track(99, label=CAT_B, x=0.60)],
        }
        assert KIND_GHOST not in kinds_of(analyse_overlay(doc(frames)))

    def test_enabling_ghosts_surfaces_it(self):
        frames = steady(20)
        frames[5] = {
            "boxes": [box(), box(label=CAT_B, x=0.60)],
            "tracks": [track(7), track(99, label=CAT_B, x=0.60)],
        }
        loose = BlipThresholds(enabled_kinds=frozenset({KIND_GHOST}))
        ghosts = [b for b in analyse_overlay(doc(frames), loose) if b.kind == KIND_GHOST]

        assert [g.track_id for g in ghosts] == [99]

    def test_a_low_confidence_but_correct_run_is_silent_by_default(self):
        frames = steady(8) + steady(4, conf=0.2) + steady(8)
        assert KIND_CONFIDENCE_DIP not in kinds_of(analyse_overlay(doc(frames)))

    def test_enabling_dips_surfaces_the_same_run(self):
        frames = steady(8) + steady(4, conf=0.2) + steady(8)
        loose = BlipThresholds(enabled_kinds=frozenset({KIND_CONFIDENCE_DIP}))

        assert KIND_CONFIDENCE_DIP in kinds_of(analyse_overlay(doc(frames), loose))


class TestTimelapseIsRefused:
    def test_frames_spaced_minutes_apart_produce_nothing(self):
        """A "dropout" between two frames ten seconds apart means the subject left, not
        that anything failed. Every temporal detector would report confident nonsense."""
        gone = [{"boxes": [], "tracks": []}] * 4
        frames = steady(8) + gone + steady(8)

        assert analyse_overlay(doc(frames, step=10.0)) == []

    def test_the_same_frames_at_clip_rate_do_report(self):
        """Proves the refusal above is about spacing, not about the frames themselves."""
        gone = [{"boxes": [], "tracks": []}] * 4
        frames = steady(8) + gone + steady(8)

        assert analyse_overlay(doc(frames, step=STEP)) != []
