"""Clip overlay boxes glide between samples instead of holding.

The watcher samples at ~1 fps, so a held box sits where the cat was up to a second ago.
`boxesAtTime` moves a box towards where the next sample found the same object. What
must not happen is a box gliding into a *different* object, or across a gap in which
the watcher was not analysing.

Run with:
    pytest playwright_tests/test_clip_overlay_glide.py -p no:asyncio -v
"""

import pytest
from playwright.sync_api import Page

_CALL = """
    async ([doc, t]) => {
        const { boxesAtTime } = await import('/app-static/js/app/nvr/events/clipOverlay.js');
        return boxesAtTime(doc, t);
    }
"""


def _box(x, label="cat", y=0.1, w=0.2, h=0.2):
    return {"labelUuid": label, "label": label, "x": x, "y": y, "w": w, "h": h, "conf": 0.9}


def _doc(*frames, stale=1.5):
    return {"maxStalenessS": stale, "frames": [{"t": t, "boxes": boxes} for t, boxes in frames]}


@pytest.fixture
def boxes_at(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    return lambda doc, t: page.evaluate(_CALL, [doc, t])


class TestGlide:
    def test_a_box_is_placed_between_where_two_samples_saw_it(self, boxes_at):
        doc = _doc((1.0, [_box(0.1)]), (2.0, [_box(0.3)]))
        [box] = boxes_at(doc, 1.25)
        assert box["x"] == pytest.approx(0.15)

    def test_each_of_two_cats_follows_its_own_next_position(self, boxes_at):
        doc = _doc((1.0, [_box(0.1), _box(0.6)]), (2.0, [_box(0.7), _box(0.2)]))
        xs = sorted(b["x"] for b in boxes_at(doc, 1.5))
        assert xs == [pytest.approx(0.15), pytest.approx(0.65)]

    def test_a_box_does_not_glide_into_an_object_of_another_label(self, boxes_at):
        doc = _doc((1.0, [_box(0.1)]), (2.0, [_box(0.3, label="dog")]))
        [box] = boxes_at(doc, 1.5)
        assert box["x"] == pytest.approx(0.1)

    def test_a_box_does_not_sweep_across_the_frame_to_a_distant_match(self, boxes_at):
        doc = _doc((1.0, [_box(0.0)]), (2.0, [_box(0.8)]))
        [box] = boxes_at(doc, 1.5)
        assert box["x"] == pytest.approx(0.0)

    def test_samples_further_apart_than_the_hold_are_not_bridged(self, boxes_at):
        """The watcher was not analysing in between; the box is held until it goes stale,
        exactly as before, rather than invented in motion through the gap."""
        doc = _doc((1.0, [_box(0.1)]), (4.0, [_box(0.3)]), stale=1.5)
        [box] = boxes_at(doc, 1.5)
        assert box["x"] == pytest.approx(0.1)
        assert boxes_at(doc, 3.0) == []
