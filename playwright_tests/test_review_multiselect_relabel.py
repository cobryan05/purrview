"""
Multi-select relabel from the review grid's right-click menu.

A relabel from a multi-selection has to reach EVERY selected card, not just the
one the menu was opened on — the per-card refresh resolves the other cards by
`grid.querySelector('[data-bbox-uuid=...]')`, which is easy to break. These pin
both surfaces that carry a label colour: the bbox card's server-rendered crop
and the parent image card's client-drawn overlay.

Run with:
    pytest playwright_tests/test_review_multiselect_relabel.py -p no:asyncio -v
"""

import base64

import pytest
from playwright.sync_api import Page

from playwright_tests.test_review_workflow_ui import (
    LABEL_CAT,
    LABEL_DOG,
    PAGE_1_IMAGES,
    _mock_labels,
    _mock_misc_review_endpoints,
    _mock_review_queue,
    _mock_tags,
    _navigate_to_review,
    _show_all_bboxes,
    _switch_view,
)

PNG_16_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAGUlEQVR4nGMUERFhIAUwkaR6VMOoh"
    "iGlAQDrWABcshm/EQAAAABJRU5ErkJggg=="
)

REVIEWED_TAG = {"uuid": "tag-reviewed", "name": "Reviewed", "color": "#4c4", "icon": None, "kind": "user"}


# Reports whether the image card's bbox overlay canvas has more red or more
# green in it — the two mock labels are pure #ff0000 / #00ff00.
_DOMINANT_CHANNEL_JS = """(imageUuid) => {
  const card = document.querySelector(`#review-root [data-kind='image'][data-image-uuid='${imageUuid}']`);
  const cv = card?.querySelector('canvas.thumb-overlay');
  if (!cv || !cv.width || !cv.height) return 'no-canvas';
  const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
  let r = 0, g = 0;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] === 0) continue;
    r += d[i];
    g += d[i + 1];
  }
  if (r === 0 && g === 0) return 'blank';
  return r > g ? 'red' : 'green';
}"""


def _mock_thumbnails(page: Page):
    """Serve a real (16x16) PNG for every thumbnail request.

    The image-card overlay bails out when the thumbnail has no natural size, so
    without a decodable image the canvas stays empty and proves nothing.
    """
    png = base64.b64decode(PNG_16_B64)
    page.route(
        "**/thumbnail?*",
        lambda route: route.fulfill(status=200, content_type="image/png", body=png),
    )


def _thumb_srcs(page: Page, uuids):
    """Current .thumb src for each bbox card, keyed by bbox uuid."""
    return {u: page.locator(f"#review-root [data-bbox-uuid='{u}'] .thumb").get_attribute("src") or "" for u in uuids}


def _open_menu_and_apply_label(page: Page, on_bbox_uuid: str, label_name: str):
    page.locator(f"#review-root [data-bbox-uuid='{on_bbox_uuid}'] .thumb").click(button="right")
    page.wait_for_selector(".ctx-menu")
    page.locator(".ctx-menu-item", has_text="Apply label").first.hover()
    # The flyout is an async panel — wait for the picker input to exist.
    picker = page.locator(".ctx-menu input[placeholder='Search or create label…']")
    picker.wait_for(state="visible", timeout=4000)
    picker.fill(label_name)
    page.wait_for_timeout(150)
    page.locator(".ctx-menu div", has_text=label_name).last.click()
    page.wait_for_timeout(500)


class TestMultiSelectRelabelRefreshesEveryCard:
    @pytest.mark.quarantine("one card of several kept its old blob src; 1 of 3 runs")
    def test_every_selected_card_gets_a_fresh_crop(self, page: Page, base_url: str):
        """Applying a label to a 2-card selection re-requests BOTH crops.

        The crop is drawn server-side with the label's colour baked in, so a card
        whose src is unchanged is still showing the previous label's colour.
        """
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [REVIEWED_TAG])
        _mock_misc_review_endpoints(page)
        _mock_thumbnails(page)
        _navigate_to_review(page, base_url)
        _show_all_bboxes(page)
        _switch_view(page, "bboxes")

        # bbox-1 and bbox-3 are both "Cat"; relabel them to "Dog".
        page.locator("#review-root [data-bbox-uuid='bbox-1'] input[type=checkbox]").click()
        page.locator("#review-root [data-bbox-uuid='bbox-3'] input[type=checkbox]").click()

        before = _thumb_srcs(page, ["bbox-1", "bbox-3"])
        _open_menu_and_apply_label(page, "bbox-1", "Dog")
        after = _thumb_srcs(page, ["bbox-1", "bbox-3"])

        # Both chips must say Dog (the part that already worked)...
        for u in ("bbox-1", "bbox-3"):
            chip = page.locator(f"#review-root [data-bbox-uuid='{u}'] .label-chip").inner_text()
            assert chip == "Dog", f"{u} chip should read Dog, got {chip!r}"

        # ...and both crops must have been re-requested.
        for u in ("bbox-1", "bbox-3"):
            assert after[u] != before[u], f"{u} crop was not refreshed (src unchanged: {after[u]!r})"

    def test_parent_image_card_overlay_recolours(self, page: Page, base_url: str):
        """The image card draws each box's outline client-side from bbox.label.color.

        Relabelling a multi-selection of bbox cards must recolour those outlines
        on the parent image cards too, not just the bbox cards.
        """
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [REVIEWED_TAG])
        _mock_misc_review_endpoints(page)
        _mock_thumbnails(page)
        _navigate_to_review(page, base_url)
        _show_all_bboxes(page)
        _switch_view(page, "all")

        page.locator("#review-root [data-bbox-uuid='bbox-1'] input[type=checkbox]").click()
        page.locator("#review-root [data-bbox-uuid='bbox-3'] input[type=checkbox]").click()
        _open_menu_and_apply_label(page, "bbox-1", "Dog")

        # Cat is #ff0000, Dog is #00ff00 — so "any pixel greener than it is red"
        # is a reliable read of which colour the overlay was stroked with, even
        # with antialiasing.
        dominant = page.evaluate(_DOMINANT_CHANNEL_JS, "img-1")
        assert dominant == "green", f"image card overlay still stroked with the old label colour ({dominant})"
