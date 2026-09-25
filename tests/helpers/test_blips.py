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
    harvest_times,
    proposed_box_at,
)

CAT_A = "cat_a"
CAT_B = "cat_b"
BOX = "litter_box"

#: Sample spacing. 5 fps is the watcher's default analysis rate.
STEP = 0.2


def box(label=CAT_A, conf=0.9, x=0.10, y=0.10, w=0.20, h=0.20):
    return {"x": x, "y": y, "w": w, "h": h, "label": label, "labelUuid": label, "conf": conf}


def track(track_id=7, saw=None, saw_conf=None, margin=None, **kwargs):
    """One track on one frame.

    `saw` is what the detector claimed here, which a v2 sidecar records beside the
    identity the track settled on. It defaults to agreement, because that is the
    ordinary case and a test that has to spell it out every time buries the one line
    that matters in the tests where they differ.
    """
    entry = {**box(**kwargs), "id": track_id}
    entry["sawLabelUuid"] = saw if saw is not None else entry["labelUuid"]
    entry["sawConf"] = saw_conf if saw_conf is not None else entry["conf"]
    if margin is not None:
        entry["margin"] = margin
    return entry


def doc(frames, step=STEP):
    """Wrap per-frame `{boxes, tracks}` dicts into an overlay document."""
    return {
        "version": 2,
        "durationS": round(len(frames) * step, 4),
        "frames": [
            {"t": round(index * step, 4), "boxes": f.get("boxes", []), "tracks": f.get("tracks", [])}
            for index, f in enumerate(frames)
        ],
    }


def steady(count, label=CAT_A, conf=0.9, track_id=7):
    """`count` frames of one well-behaved track — detection and track agreeing."""
    return [{"boxes": [box(label=label, conf=conf)], "tracks": [track(track_id, label=label, conf=conf)]}] * count


def misread(count, settled=CAT_A, saw=CAT_B, conf=0.9, track_id=7, margin=None):
    """`count` frames where the detector disagreed with the identity the track holds.

    The track keeps reporting `settled` — its vote across the whole clip — while the
    model's own output for these frames says `saw`. That pairing is what makes the
    frames harvestable: the box is exact and the right answer is known.
    """
    return [
        {
            "boxes": [box(label=saw, conf=conf)],
            "tracks": [track(track_id, label=settled, conf=conf, saw=saw, saw_conf=conf, margin=margin)],
        }
    ] * count


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
    """Two different failures that used to be conflated, and one that never fired.

    Before a track's label became a vote, `identity_flip` was defined as a track's label
    departing from its own modal value — which could not happen, because association was
    gated on label equality and the label was fixed at birth. It fired zero times across
    1,963 real events. A flickering identity forked a second track instead, so the single
    most informative failure in a clip produced no training data at all.

    It is now the disagreement between what the track settled on and what the detector
    said frame by frame, and `identity_switch` keeps the other half: the settled label
    itself changing hands, which is a tracking fault and must never be harvested.
    """

    def test_the_detector_contradicting_the_settled_identity_is_a_harvestable_flip(self):
        frames = steady(8) + misread(4) + steady(8)
        blips = [b for b in analyse_overlay(doc(frames)) if b.kind == KIND_IDENTITY_FLIP]

        assert len(blips) == 1
        assert blips[0].fault == FAULT_DETECTOR
        assert blips[0].detail["modalLabel"] == CAT_A, "the proposal must be the corrected label"
        assert blips[0].detail["sawLabel"] == CAT_B
        assert blips[0].harvestable

    def test_a_flip_spans_only_the_frames_that_disagreed(self):
        frames = steady(8) + misread(4) + steady(8)
        flip = next(b for b in analyse_overlay(doc(frames)) if b.kind == KIND_IDENTITY_FLIP)

        assert flip.start_s == pytest.approx(8 * STEP)
        assert flip.end_s == pytest.approx(11 * STEP)

    def test_agreement_reports_nothing(self):
        """The negative that keeps the detector honest: a clip the model got right must
        not manufacture training data out of itself."""
        assert KIND_IDENTITY_FLIP not in kinds_of(analyse_overlay(doc(steady(20))))

    def test_a_single_disagreeing_frame_is_reported(self):
        """One frame is a fifth of a second of the model being wrong, with an exact box
        and a known right answer. The old two-sample floor existed only because the
        reference used to be as noisy as the observation."""
        frames = steady(8) + misread(1) + steady(8)
        assert KIND_IDENTITY_FLIP in kinds_of(analyse_overlay(doc(frames)))

    def test_raising_the_floor_suppresses_a_single_frame(self):
        frames = steady(8) + misread(1) + steady(8)
        strict = BlipThresholds(min_disagreement_samples=2)

        assert KIND_IDENTITY_FLIP not in kinds_of(analyse_overlay(doc(frames), strict))

    def test_a_confidently_wrong_flip_outranks_an_unsure_one(self):
        """Severity encodes how sure the detector was about what it got wrong, because
        confidently-wrong frames are the most valuable training data in the pipeline."""
        sure = steady(8) + misread(4, conf=0.95) + steady(8)
        unsure = steady(8) + misread(4, conf=0.30) + steady(8)

        sure_flip = next(b for b in analyse_overlay(doc(sure)) if b.kind == KIND_IDENTITY_FLIP)
        unsure_flip = next(b for b in analyse_overlay(doc(unsure)) if b.kind == KIND_IDENTITY_FLIP)

        assert sure_flip.severity > unsure_flip.severity

    def test_the_confusion_margin_is_carried_for_triage(self):
        """A near-tie between two names for one object is a systematic confusion, not a
        hard frame, and no absolute confidence threshold can tell them apart."""
        frames = steady(8) + misread(4, conf=0.91, margin=0.03) + steady(8)
        flip = next(b for b in analyse_overlay(doc(frames)) if b.kind == KIND_IDENTITY_FLIP)

        assert flip.detail["margin"] == pytest.approx(0.03)

    def test_a_sidecar_that_predates_the_record_reports_no_flips(self):
        """Silence in a v1 document means "not recorded", not "the detector agreed".
        Inferring agreement would fabricate a clean bill of health for every old clip."""
        frames = steady(8) + misread(4) + steady(8)
        for frame in frames:
            for entry in frame["tracks"]:
                entry.pop("sawLabelUuid", None)

        assert KIND_IDENTITY_FLIP not in kinds_of(analyse_overlay(doc(frames)))

    def test_a_coasting_frame_is_not_a_disagreement(self):
        """ "The detector said nothing" is a dropout, and a different fault entirely."""
        blind = [{"boxes": [], "tracks": [{**track(7), "sawLabelUuid": None, "sawConf": None}]}] * 4
        blips = analyse_overlay(doc(steady(8) + blind + steady(8)))

        assert KIND_IDENTITY_FLIP not in kinds_of(blips)
        assert KIND_DETECTOR_DROPOUT in kinds_of(blips)

    def test_a_label_that_changes_and_stays_changed_is_a_switch_blamed_on_tracking(self):
        """Track-id reuse across two objects, not the detector misreading one — so it
        must not become training data."""
        frames = steady(12) + steady(8, label=CAT_B)
        blips = [b for b in analyse_overlay(doc(frames)) if b.kind == KIND_IDENTITY_SWITCH]

        assert len(blips) == 1
        assert blips[0].fault == FAULT_TRACKER
        assert not blips[0].harvestable

    def test_a_switch_is_not_also_reported_as_a_flip(self):
        """One incident, one blip. Double-reporting would harvest a tracking fault
        through the flip door, which is exactly what fault attribution exists to stop."""
        frames = steady(12) + steady(8, label=CAT_B)
        assert KIND_IDENTITY_FLIP not in kinds_of(analyse_overlay(doc(frames)))


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


class TestTuningFromAStoredBundle:
    """Thresholds come from a JSON column a user can edit.

    The failure to avoid is a watcher that refuses to analyse — or worse, refuses to
    *record* — because one key was misspelt or typed as a string, so this degrades
    toward the conservative default rather than toward broken.
    """

    def test_an_empty_bundle_is_the_conservative_default(self):
        assert BlipThresholds.from_overrides({}) == BlipThresholds()
        assert BlipThresholds.from_overrides(None) == BlipThresholds()

    def test_an_override_takes_effect_and_the_rest_stay_default(self):
        tuned = BlipThresholds.from_overrides({"min_flip_samples": 5})

        assert tuned.min_flip_samples == 5
        assert tuned.min_dropout_samples == BlipThresholds().min_dropout_samples

    def test_raising_the_flip_floor_suppresses_a_flip_that_the_default_finds(self):
        """The override reaching the detectors, not just the dataclass."""
        frames = steady(8) + misread(3) + steady(8)

        assert KIND_IDENTITY_FLIP in kinds_of(analyse_overlay(doc(frames)))
        strict = BlipThresholds.from_overrides({"min_disagreement_samples": 5})
        assert KIND_IDENTITY_FLIP not in kinds_of(analyse_overlay(doc(frames), strict))

    def test_enabled_kinds_arrives_as_a_list_because_json_has_no_set(self):
        tuned = BlipThresholds.from_overrides({"enabled_kinds": [KIND_GHOST, KIND_IDENTITY_FLIP]})

        assert tuned.enabled(KIND_GHOST)
        assert not tuned.enabled(KIND_FULL_DROPOUT)

    def test_an_empty_kind_list_is_ignored_rather_than_silencing_everything(self):
        """Disabling every detector is what the per-watcher switch is for. An empty list
        in a tuning bundle is far more likely to be a mistake than an intent."""
        tuned = BlipThresholds.from_overrides({"enabled_kinds": []})

        assert tuned.enabled_kinds == BlipThresholds().enabled_kinds

    def test_an_unusable_value_falls_back_instead_of_raising(self):
        tuned = BlipThresholds.from_overrides({"min_dropout_samples": "three", "occlusion_iou": None})

        assert tuned.min_dropout_samples == BlipThresholds().min_dropout_samples
        assert tuned.occlusion_iou == BlipThresholds().occlusion_iou

    def test_an_unknown_key_is_ignored(self):
        assert BlipThresholds.from_overrides({"speed": "fast"}) == BlipThresholds()


class TestChoosingWhichFramesToPull:
    def test_an_incident_yields_its_onset_middle_and_recovery(self):
        assert harvest_times(4.0, 6.0, 3) == [4.0, 5.0, 6.0]

    def test_a_single_instant_yields_one_moment_not_three_copies(self):
        assert harvest_times(4.0, 4.0, 3) == [4.0]

    def test_moments_inside_one_frame_interval_collapse(self):
        """Two "different" moments closer than a millisecond are the same picture, and
        uploading it twice just leans on dedup to notice."""
        assert len(harvest_times(4.0, 4.0005, 3)) == 1


class TestProposingTheCorrectedBox:
    """What gets offered for a harvested frame.

    The distinction that matters is not the blip kind but whether the track was *seen*
    at that moment: seen means the detector's own box with only the label corrected,
    unseen means an interpolation that must be flagged so the reviewer opens it on the
    canvas rather than accepting it from a thumbnail.
    """

    def _doc(self, frames):
        return doc(frames)

    def test_a_seen_track_proposes_the_detectors_own_box_exactly(self):
        frames = steady(20)
        box = proposed_box_at(self._doc(frames), track_id=7, label_uuid=CAT_A, at_s=1.0)

        assert box is not None
        assert not box.estimated
        assert (box.x, box.y, box.w, box.h) == (0.10, 0.10, 0.20, 0.20)

    def test_the_label_is_the_correction_not_what_was_seen(self):
        """The whole point of a flip proposal: geometry from the detector, label from
        the identity the track settled on."""
        frames = steady(8) + misread(4) + steady(8)
        box = proposed_box_at(self._doc(frames), track_id=7, label_uuid=CAT_A, at_s=8 * STEP)

        assert box is not None
        assert box.label_uuid == CAT_A
        assert not box.estimated

    def test_a_flip_proposes_a_corrected_box_at_every_harvested_moment(self):
        """End to end: the blip the analysis found must yield a usable proposal at each
        moment the harvest pulls a frame from, carrying the corrected label rather than
        the one the detector got wrong.

        The onset is asserted exact separately — it is a real sample, where the geometry
        is the detector's own box and only the label changes. Interior moments may fall
        between samples and be interpolated across a fifth of a second, which is honest
        and is flagged as such.
        """
        frames = steady(8) + misread(4) + steady(8)
        document = self._doc(frames)
        flip = next(b for b in analyse_overlay(document) if b.kind == KIND_IDENTITY_FLIP)

        proposals = [
            proposed_box_at(document, track_id=flip.track_id, label_uuid=flip.detail["modalLabel"], at_s=at)
            for at in harvest_times(flip.start_s, flip.end_s)
        ]

        assert proposals and all(p is not None for p in proposals)
        assert {p.label_uuid for p in proposals} == {CAT_A}
        assert not proposals[0].estimated

    def test_an_unseen_moment_is_interpolated_and_flagged(self):
        moving = [
            {
                "boxes": [box(x=0.1 + i * 0.02)],
                "tracks": [track(7, x=0.1 + i * 0.02)],
            }
            for i in range(20)
        ]
        # Drop the track from the middle so there is a gap to cross.
        for i in (8, 9, 10):
            moving[i] = {"boxes": [], "tracks": []}

        box_at = proposed_box_at(self._doc(moving), track_id=7, label_uuid=CAT_A, at_s=9 * STEP)

        assert box_at is not None
        assert box_at.estimated
        # Between the box at t=7 (x=0.24) and the one at t=11 (x=0.32).
        assert 0.24 < box_at.x < 0.32

    def test_nothing_is_proposed_when_only_one_side_exists(self):
        """Extrapolating from one side puts a box where the subject may not be. A
        missing proposal costs one frame; a confidently wrong box costs training data."""
        frames = steady(10)
        after_the_end = proposed_box_at(self._doc(frames), track_id=7, label_uuid=CAT_A, at_s=30.0)

        assert after_the_end is None

    def test_nothing_is_proposed_across_a_gap_too_long_to_estimate(self):
        frames = steady(6) + [{"boxes": [], "tracks": []}] * 30 + steady(6)
        box_at = proposed_box_at(self._doc(frames), track_id=7, label_uuid=CAT_A, at_s=10 * STEP)

        assert box_at is None

    def test_an_unknown_track_proposes_nothing(self):
        assert proposed_box_at(self._doc(steady(20)), track_id=999, label_uuid=CAT_A, at_s=1.0) is None
