"""The event card's cropped art.

The wide snapshot answers "where in the room?" once. The grid asks "what was this?", and
a wall of identical shots of the same room cannot answer it — which is what the crop is
for. Three things have to hold or it silently stops being an answer:

* it frames **the event's** tracks, not everything the detector saw. A rect around all
  the boxes is a rect around the room, i.e. the picture this replaces, and the failure is
  invisible: you get a crop, it is just useless;
* it is cut from the **camera's** frame, not the analysis downscale. Both produce a
  picture, and the wrong one produces a 2–3× enlargement of pixels already thrown away;
* it carries **no boxes**. A stroke sized for a whole room covers the animal when the
  same frame is magnified onto a card.

None of the three raises when it goes wrong, and none of them is visible in a row.
"""

import cv2
import numpy as np
import pytest

from apps.helpers.recordings.eventCrop import CARD_ASPECT, focus_rect, is_worth_cropping

from .test_watch_event_recorder import _context, _open, _signal, _Trk, recorder, store  # noqa: F401

#: BGR markers painted into the frame, chosen to be far apart in every channel so a
#: JPEG round trip cannot turn one into the other.
SUBJECT_COLOUR = (0, 0, 255)
BYSTANDER_COLOUR = (0, 255, 0)


def _frame(width, height, patches=()):
    """A black frame with solid colour patches at `(x1, y1, x2, y2, colour)`."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for x1, y1, x2, y2, colour in patches:
        canvas[y1:y2, x1:x2] = colour
    return canvas


def _has_colour(image, colour, tolerance=60):
    """Whether any pixel is close to `colour`, allowing for JPEG ringing."""
    distance = np.abs(image.astype(np.int16) - np.array(colour, dtype=np.int16)).sum(axis=2)
    return bool((distance < tolerance).any())


def _crops(tmp_path):
    return sorted((tmp_path / "snapshots").rglob("*_crop.jpg"))


def _the_crop(tmp_path):
    """The one crop written, decoded — with a legible failure when there isn't one.

    Worth the helper: several ways of getting the framing wrong end up producing a rect
    that covers the whole frame, which is then correctly skipped, so the symptom of a
    framing bug is *no file* rather than a wrong one. Indexing straight into the glob
    reports that as `IndexError: list index out of range`, which points at the test.
    """
    crops = _crops(tmp_path)
    assert crops, "no crop was written — the rect probably grew to cover the whole frame"
    return cv2.imread(str(crops[0]))


class TestWhatTheCropFrames:
    async def test_it_frames_the_events_tracks_and_not_the_bystanders(self, recorder, tmp_path):
        """The whole feature is "this card shows *its* event".

        A crop taken around every track on the frame is a crop around the room — the
        exact picture the wide snapshot already gives, produced at extra cost and stored
        in a second file. Nothing about that failure is visible from the row.
        """
        # Subject far left, bystander far right, both small enough that a rect around
        # only one of them cannot reach the other.
        frame = _frame(
            960,
            540,
            [
                (40, 220, 120, 300, SUBJECT_COLOUR),
                (840, 220, 920, 300, BYSTANDER_COLOUR),
            ],
        )
        context = _context(
            frame=frame,
            tracks=[
                _Trk((40, 220, 120, 300), track_id=7),
                _Trk((840, 220, 920, 300), track_id=99),
            ],
        )

        await _open(recorder, _signal(track_ids=[7]), context)

        crop = _the_crop(tmp_path)
        assert _has_colour(crop, SUBJECT_COLOUR), "the event's own subject is missing from its card"
        assert not _has_colour(crop, BYSTANDER_COLOUR), "the crop reached a track the event never fired on"

    async def test_unmatched_track_ids_fall_back_to_every_track(self, recorder, tmp_path):
        """A signal can name tracks the tracker has already retired by the time an action
        runs. A plausible crop beats no crop — the alternative is a card that silently
        stays wide for a whole class of events."""
        frame = _frame(960, 540, [(400, 220, 520, 320, SUBJECT_COLOUR)])
        context = _context(frame=frame, tracks=[_Trk((400, 220, 520, 320), track_id=3)])

        await _open(recorder, _signal(track_ids=[404]), context)

        assert _has_colour(_the_crop(tmp_path), SUBJECT_COLOUR)

    async def test_a_subject_at_the_edge_stays_whole(self, recorder, tmp_path):
        """The rect slides inside the frame rather than being trimmed to fit. Trimming
        is what a naive clamp does, and it cuts the subject in half on exactly the
        events where the animal is entering or leaving."""
        frame = _frame(960, 540, [(0, 0, 90, 90, SUBJECT_COLOUR)])
        context = _context(frame=frame, tracks=[_Trk((0, 0, 90, 90), track_id=7)])

        await _open(recorder, _signal(track_ids=[7]), context)

        crop = _the_crop(tmp_path)
        assert _has_colour(crop, SUBJECT_COLOUR)


class TestWhichFrameItIsCutFrom:
    async def test_the_crop_comes_from_the_full_resolution_frame(self, recorder, tmp_path):
        """`full_frame` is the camera's own picture; `frame` is the downscale inference
        ran on. Both crop successfully, and the wrong one throws away the resolution
        that is the entire reason this is a second file rather than a CSS zoom."""
        analysis = _frame(96, 54, [(40, 22, 52, 32, SUBJECT_COLOUR)])
        full = _frame(960, 540, [(400, 220, 520, 320, SUBJECT_COLOUR)])
        context = _context(
            frame=analysis,
            full_frame=full,
            tracks=[_Trk((40, 22, 52, 32), track_id=7)],
        )

        await _open(recorder, _signal(track_ids=[7]), context)

        crop = _the_crop(tmp_path)
        # A third of the 96 px analysis frame is 32 px; a third of the 960 px camera
        # frame is 320. Anything in between means neither, and is worth failing on.
        assert crop.shape[1] > 100, f"crop is {crop.shape[1]}px wide — cut from the analysis frame"

    async def test_the_analysis_frame_is_used_when_there_is_no_other(self, recorder, tmp_path):
        """A source that never supplies a full-resolution frame still gets a card that
        shows its subject. Softer, and still not the same picture as every other card."""
        analysis = _frame(320, 180, [(140, 70, 180, 110, SUBJECT_COLOUR)])
        context = _context(frame=analysis, full_frame=None, tracks=[_Trk((140, 70, 180, 110), track_id=7)])

        await _open(recorder, _signal(track_ids=[7]), context)

        assert _has_colour(_the_crop(tmp_path), SUBJECT_COLOUR)


class TestTheWrittenFilesShape:
    async def test_the_crop_file_is_the_cards_shape(self, recorder, tmp_path):
        """The end-to-end assertion for the letterbox bug — the unit test above covers
        the rect, this covers the pixels that actually reached disk.

        `.event-card__media` is `aspect-ratio: 16 / 9` with `object-fit: cover`, so a
        crop of the wrong shape is not letterboxed on screen: it is *cropped again* by
        the browser, along the axis the server had no say in.
        """
        frame = _frame(960, 540, [(400, 220, 520, 320, SUBJECT_COLOUR)])
        context = _context(frame=frame, tracks=[_Trk((400, 220, 520, 320), track_id=7)])

        await _open(recorder, _signal(track_ids=[7]), context)

        crop = _the_crop(tmp_path)
        height, width = crop.shape[:2]
        assert width / height == pytest.approx(CARD_ASPECT, rel=0.02), f"crop is {width}x{height}"


class TestWhenNoCropIsWritten:
    async def test_an_event_with_no_tracks_files_a_row_and_no_crop(self, recorder, store, tmp_path):
        """Ordinary, not a failure: the card falls back to the wide shot. Costing the
        event its row over a missing crop would be the real bug."""
        await _open(recorder, _signal(), _context(frame=_frame(960, 540), tracks=[]))

        assert len(store.created) == 1
        assert store.created[0].crop_path is None
        assert _crops(tmp_path) == []

    async def test_boxes_spanning_the_frame_write_no_crop(self, recorder, store, tmp_path):
        """A crop that shows what the snapshot already shows is a second file for
        nothing — and a second file the retention sweep then has to carry."""
        frame = _frame(960, 540, [(10, 10, 950, 530, SUBJECT_COLOUR)])
        context = _context(frame=frame, tracks=[_Trk((10, 10, 950, 530), track_id=7)])

        await _open(recorder, _signal(track_ids=[7]), context)

        assert store.created[0].crop_path is None
        assert _crops(tmp_path) == []

    async def test_the_snapshot_is_still_written_when_the_crop_is_not(self, recorder, store, tmp_path):
        """The two are independent. A card with no crop must still have a picture."""
        await _open(recorder, _signal(), _context(frame=_frame(960, 540), tracks=[]))

        assert store.created[0].snapshot_path is not None


class TestTheCropCarriesNoBoxes:
    async def test_no_annotation_is_drawn_on_the_crop(self, recorder, tmp_path):
        """The snapshot has boxes on purpose; the crop must not. At this magnification a
        stroke sized for the whole room sits on top of the animal it points at.

        Asserted through the label colour: `annotate_detections` strokes and letters
        every box in it, so its total absence from the crop is what "nothing was drawn"
        looks like from outside.
        """
        frame = _frame(960, 540, [(400, 220, 520, 320, SUBJECT_COLOUR)])
        context = _context(
            frame=frame,
            tracks=[_Trk((400, 220, 520, 320), track_id=7, label_uuid="lbl-1")],
            label_colours={"lbl-1": "#00ff00"},
        )

        await _open(recorder, _signal(track_ids=[7]), context)

        crop = _the_crop(tmp_path)
        assert not _has_colour(crop, (0, 255, 0)), "the crop was annotated"

    async def test_the_wide_snapshot_still_is_annotated(self, recorder, tmp_path):
        """The counterpart, so "draw nothing" cannot be satisfied by drawing nothing
        anywhere. The detail modal and the timeline hover want the boxes."""
        frame = _frame(960, 540, [(400, 220, 520, 320, SUBJECT_COLOUR)])
        context = _context(
            frame=frame,
            detections=[_Trk((400, 220, 520, 320), track_id=7, label_uuid="lbl-1")],
            tracks=[_Trk((400, 220, 520, 320), track_id=7, label_uuid="lbl-1")],
            label_colours={"lbl-1": "#00ff00"},
        )

        await _open(recorder, _signal(track_ids=[7]), context)

        wide = [p for p in (tmp_path / "snapshots").rglob("*.jpg") if not p.name.endswith("_crop.jpg")][0]
        assert _has_colour(cv2.imread(str(wide)), (0, 255, 0))


class TestTheCropLivesBesideItsSnapshot:
    async def test_two_rules_on_one_frame_do_not_share_a_crop(self, recorder, tmp_path):
        """Same trap the snapshot has: two rules opening on the same frame for the same
        camera produce the same instant, and a name built from the instant alone would
        have one event's card show the other's picture."""
        frame = _frame(960, 540, [(400, 220, 520, 320, SUBJECT_COLOUR)])
        context = _context(frame=frame, tracks=[_Trk((400, 220, 520, 320), track_id=7)])

        await _open(recorder, _signal(track_ids=[7], rule_uuid="rule-1"), context)
        await _open(recorder, _signal(track_ids=[7], rule_uuid="rule-2"), context)

        assert len(_crops(tmp_path)) == 2


class TestFocusRectGeometry:
    """The shared geometry, which the live recorder and the backfill both call.

    Unit-level because the properties below are what stop a crop from being unusable,
    and each of them has a plausible wrong implementation that still returns a rect.
    """

    #: A 16:9 camera, the shape every source here produces.
    WIDE = 16 / 9

    def test_a_tiny_box_does_not_become_a_full_card_enlargement(self):
        """Without a floor, a confident 30 px detection on a distant camera is magnified
        to fill a card — technically the subject, visually a smear."""
        rect = focus_rect([(0.50, 0.50, 0.01, 0.01)], self.WIDE)
        assert rect is not None
        assert rect[2] >= 1.0 / 3.0 - 1e-9

    @pytest.mark.parametrize("frame_aspect", [16 / 9, 4 / 3, 1.0, 9 / 16])
    def test_the_rect_is_the_cards_shape_in_pixels(self, frame_aspect):
        """**In pixels, not in normalised units** — the two are only the same on a
        square frame.

        A normalised `w/h` of 16/9 on a 16/9 camera is 3.16:1 on screen: a letterbox
        strip with the animal's head cut off. That shipped, and 883 back-filled crops
        had to be re-cut. Parametrised across frame shapes because the bug is invisible
        at exactly one of them.
        """
        _x, _y, w, h = focus_rect([(0.4, 0.4, 0.05, 0.1)], frame_aspect)
        assert (w * frame_aspect) / h == pytest.approx(CARD_ASPECT, rel=1e-6)

    def test_saturating_one_axis_does_not_flatten_the_shape(self):
        """A rect wider than the frame must shrink on both axes, not be capped on one.
        Capping is the obvious spelling and it silently undoes the fitting above — and
        it fires on the common case, a wide rect on a frame only so wide."""
        _x, _y, w, h = focus_rect([(0.02, 0.45, 0.96, 0.1)], self.WIDE)
        assert (w * self.WIDE) / h == pytest.approx(CARD_ASPECT, rel=1e-6)
        assert w <= 1.0 + 1e-9 and h <= 1.0 + 1e-9

    def test_the_rect_stays_inside_the_frame(self):
        for box in [(0.0, 0.0, 0.05, 0.05), (0.95, 0.95, 0.05, 0.05)]:
            x, y, w, h = focus_rect([box], self.WIDE)
            assert x >= 0 and y >= 0
            assert x + w <= 1.0 + 1e-9 and y + h <= 1.0 + 1e-9

    def test_it_covers_every_box_it_is_given(self):
        """Subject *and* target: the card headline says "A → B", and a crop showing only
        A contradicts its own caption."""
        x, y, w, h = focus_rect([(0.05, 0.4, 0.1, 0.1), (0.8, 0.4, 0.1, 0.1)], self.WIDE)
        assert x <= 0.05 and x + w >= 0.9

    def test_no_boxes_is_no_rect(self):
        assert focus_rect([], self.WIDE) is None
        assert focus_rect([(0.1, 0.1, 0.0, 0.0)], self.WIDE) is None

    def test_a_full_frame_rect_is_not_worth_cropping(self):
        assert not is_worth_cropping((0.0, 0.0, 1.0, 1.0))
        assert not is_worth_cropping(None)
        assert is_worth_cropping((0.25, 0.25, 0.5, 0.28))
