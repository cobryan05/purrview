"""
Tests for the training-run gap row on review image cards.

A "gap" is a label the selected training run still needs resolved on an image
(the per-image slice of the co-label gap counts on the coverage table). The card
renders them as a capped row of actionable chips; clicking a chip writes the
image label and re-reads that one image's gaps.

Covers:
  - the row renders one chip per gap, capped at 3 with a "+N" overflow chip,
  - image-scope gaps offer both present and absent; object-scope gaps only absent,
  - clicking a chip POSTs the image-label write and re-reads gaps for THAT image only,
  - clearing the last gap removes the row and marks the card verified,
  - no gapsConfig → no row and no /image-gaps request at all.

Run with:
    docker compose exec -T whiskerwatcher-dev bash -lc \
      'cd /project && python -m pytest playwright_tests/test_review_card_gaps.py -p no:asyncio -q'
"""

import base64
import io
import json
import re

import pytest
from PIL import Image
from playwright.sync_api import Page, Route, expect

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}
LABEL_DOG = {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00", "parentUuid": None}
LABEL_BIRD = {"uuid": "lbl-bird", "name": "Bird", "color": "#0000ff", "parentUuid": None}
LABEL_FOX = {"uuid": "lbl-fox", "name": "Fox", "color": "#ff00ff", "parentUuid": None}
LABEL_WEATHER = {"uuid": "lbl-weather", "name": "Weather", "color": "#00ffff", "parentUuid": None}

UNVERIFIED_TAG = {"uuid": UNVERIFIED_TAG_UUID, "name": "Unverified", "color": "#888", "kind": "system", "icon": None}

BBOX_1 = {
    "uuid": "bbox-1",
    "x": 0.1,
    "y": 0.1,
    "width": 0.2,
    "height": 0.2,
    "label": {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000"},
    "labelUuid": "lbl-cat",
    "tags": [UNVERIFIED_TAG],
}
IMAGE_1 = {"uuid": "img-1", "filename": "folder/image1.jpg", "bboxes": [BBOX_1]}

CONFIG = {"uuid": "tc-1", "name": "Nightly Detector"}


def _gap(label_uuid: str, label_name: str, scope: str = "object") -> dict:
    return {"labelUuid": label_uuid, "labelName": label_name, "labelScope": scope}


def _mock_labels(page: Page, labels: list):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": labels}),
        ),
    )


def _mock_tags(page: Page, tags: list):
    page.route(
        "**/api/tags/list",
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"tags": tags})),
    )


def _mock_image_labels_batch(page: Page):
    page.route(
        "**/api/images/labels/batch*",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "imageLabels": {}})
        ),
    )


def _mock_image_gaps(page: Page, responses: list):
    """Serve /image-gaps, returning `responses[i]` for the i-th request (last one repeats).

    Each response is a {imageUuid: [gap, ...]} dict — the same shape the endpoint
    returns, where an image with nothing outstanding is simply absent.
    Returns the list of recorded request bodies.
    """
    calls = []

    def handle(route: Route):
        calls.append(json.loads(route.request.post_data or "{}"))
        gaps = responses[min(len(calls) - 1, len(responses) - 1)]
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "gaps": gaps}),
        )

    page.route("**/api/training-configs/*/image-gaps", handle)
    return calls


def _mock_set_image_label(page: Page):
    calls = []

    def handle(route: Route):
        calls.append(json.loads(route.request.post_data or "{}"))
        route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "skipped": False})
        )

    page.route("**/api/images/labels/set", handle)
    return calls


def _open_modal(page: Page, base_url: str, *, with_config: bool = True, images=None):
    images = images if images is not None else [IMAGE_1]
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.evaluate(
        """async ({ images, config }) => {
            const { showReviewModal } = await import('/app-static/js/app/review/main.js');
            window.__refresh = await showReviewModal({
                title: 'Gap Row Test',
                initialViewFilter: 'images',
                gapsConfig: config,
                fetchImages: async () => images,
            });
        }""",
        {"images": images, "config": CONFIG if with_config else None},
    )
    page.wait_for_selector(".review-grid")


class TestGapRowRendering:
    def _setup(self, page: Page):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG, LABEL_BIRD, LABEL_FOX, LABEL_WEATHER])
        _mock_tags(page, [UNVERIFIED_TAG])
        _mock_image_labels_batch(page)

    def test_renders_a_chip_per_gap(self, page: Page, base_url: str):
        self._setup(page)
        _mock_image_gaps(page, [{"img-1": [_gap("lbl-dog", "Dog"), _gap("lbl-bird", "Bird")]}])

        _open_modal(page, base_url)

        row = page.locator("[data-testid='card-gaps-row']").first
        expect(row).to_be_visible()
        expect(row.locator(".gap-chip")).to_have_count(2)
        expect(row).to_contain_text("Dog")
        expect(row).to_contain_text("Bird")

    def test_caps_chips_and_shows_overflow(self, page: Page, base_url: str):
        """Five gaps render three chips plus a "+2" — the full list lives in the hover panel."""
        self._setup(page)
        _mock_image_gaps(
            page,
            [
                {
                    "img-1": [
                        _gap("lbl-dog", "Dog"),
                        _gap("lbl-bird", "Bird"),
                        _gap("lbl-fox", "Fox"),
                        _gap("lbl-weather", "Weather", "image"),
                        _gap("lbl-cat", "Cat"),
                    ]
                }
            ],
        )

        _open_modal(page, base_url)

        row = page.locator("[data-testid='card-gaps-row']").first
        expect(row.locator(".gap-chip")).to_have_count(4)  # 3 real + 1 overflow
        overflow = row.locator(".gap-chip--more")
        expect(overflow).to_have_text("+2")
        # The overflow chip names what it hides rather than just counting it.
        expect(overflow).to_have_attribute("title", "Weather, Cat")

    def test_no_row_when_image_has_no_gaps(self, page: Page, base_url: str):
        self._setup(page)
        _mock_image_gaps(page, [{}])

        _open_modal(page, base_url)

        expect(page.locator(".review-card--image").first).to_be_visible()
        expect(page.locator("[data-testid='card-gaps-row'] .gap-chip")).to_have_count(0)

    def test_no_gaps_config_means_no_request(self, page: Page, base_url: str):
        """Without a training run to judge against, the card must not render the row
        OR issue the batch request."""
        self._setup(page)
        calls = _mock_image_gaps(page, [{"img-1": [_gap("lbl-dog", "Dog")]}])

        _open_modal(page, base_url, with_config=False)

        expect(page.locator(".review-card--image").first).to_be_visible()
        expect(page.locator("[data-testid='card-gaps-row'] .gap-chip")).to_have_count(0)
        assert calls == [], f"image-gaps must not be requested without a config: {calls}"

    def test_scope_determines_available_actions(self, page: Page, base_url: str):
        """Image-scope gaps clear via present OR absent, so both are offered; object-scope
        ones can only be cleared here by absent (a verified box needs the canvas)."""
        self._setup(page)
        _mock_image_gaps(page, [{"img-1": [_gap("lbl-dog", "Dog"), _gap("lbl-weather", "Weather", "image")]}])

        _open_modal(page, base_url)

        chips = page.locator("[data-testid='card-gaps-row'] .gap-chip")
        object_chip = chips.nth(0)
        image_chip = chips.nth(1)
        expect(object_chip.locator(".gap-chip__act")).to_have_count(1)
        expect(object_chip.locator(".gap-chip__act")).to_have_text("✕")
        expect(image_chip.locator(".gap-chip__act")).to_have_count(2)


class TestGapChipResolves:
    def _setup(self, page: Page):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG, LABEL_BIRD, LABEL_WEATHER])
        _mock_tags(page, [UNVERIFIED_TAG])
        _mock_image_labels_batch(page)

    def test_absent_click_writes_and_refetches_one_image(self, page: Page, base_url: str):
        self._setup(page)
        gap_calls = _mock_image_gaps(
            page,
            [
                {"img-1": [_gap("lbl-dog", "Dog"), _gap("lbl-bird", "Bird")]},
                {"img-1": [_gap("lbl-bird", "Bird")]},
            ],
        )
        label_calls = _mock_set_image_label(page)

        _open_modal(page, base_url)

        row = page.locator("[data-testid='card-gaps-row']").first
        expect(row.locator(".gap-chip")).to_have_count(2)
        row.locator(".gap-chip", has_text="Dog").locator(".gap-chip__act").click()

        expect(row.locator(".gap-chip")).to_have_count(1)
        expect(row).to_contain_text("Bird")
        expect(row).not_to_contain_text("Dog")

        assert len(label_calls) == 1, label_calls
        assert label_calls[0]["labelUuid"] == "lbl-dog"
        assert label_calls[0]["status"] == "absent"

        # The re-read is scoped to the one image that changed, not the whole page.
        assert len(gap_calls) == 2, gap_calls
        assert gap_calls[1]["imageUuids"] == ["img-1"]

    def test_present_click_on_image_scope_gap(self, page: Page, base_url: str):
        self._setup(page)
        _mock_image_gaps(page, [{"img-1": [_gap("lbl-weather", "Weather", "image")]}, {}])
        label_calls = _mock_set_image_label(page)

        _open_modal(page, base_url)

        chip = page.locator("[data-testid='card-gaps-row'] .gap-chip").first
        chip.locator(".gap-chip__act").first.click()  # the ✓ (present) button

        expect(page.locator("[data-testid='card-gaps-row'] .gap-chip")).to_have_count(0)
        assert label_calls[0]["labelUuid"] == "lbl-weather"
        assert label_calls[0]["status"] == "present"

    def test_clearing_last_gap_marks_the_card(self, page: Page, base_url: str):
        """The card gets the shared stale treatment once nothing is outstanding, so a
        burned-down grid reads at a glance."""
        self._setup(page)
        _mock_image_gaps(page, [{"img-1": [_gap("lbl-dog", "Dog")]}, {}])
        _mock_set_image_label(page)

        _open_modal(page, base_url)

        card = page.locator(".review-card--image").first
        expect(card).not_to_have_class(re.compile(r"\bstale-verified\b"))

        page.locator("[data-testid='card-gaps-row'] .gap-chip__act").first.click()

        expect(card).to_have_class(re.compile(r"\bstale-verified\b"))
        expect(page.locator("[data-testid='card-gaps-row'] .gap-chip")).to_have_count(0)


class TestGapRowHoverPreviewOverlay:
    """The image-card hover popover draws the bbox overlay on a canvas sized to
    its wrapper, not to the <img> it sits on top of. Adding the gaps side panel
    turns the popover into a flex row, and the wrapper (a flex item with default
    `align-items: stretch`) stretches to match the taller side panel — so the
    canvas ends up taller than the image beneath it and every bbox is drawn in
    the wrong place."""

    def _setup(self, page: Page):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG, LABEL_BIRD, LABEL_FOX, LABEL_WEATHER])
        _mock_tags(page, [UNVERIFIED_TAG])
        _mock_image_labels_batch(page)

        # A very wide, short image: max-width:492 caps its rendered width, so its
        # rendered height stays small — well under the side panel's height once it
        # lists several gaps. That gap makes the wrapper-stretch bug unmissable.
        wide_png = io.BytesIO()
        Image.new("RGB", (800, 40), color="red").save(wide_png, format="PNG")
        page.route(
            "**/thumbnail?*",
            lambda route: route.fulfill(status=200, content_type="image/png", body=wide_png.getvalue()),
        )

    def test_overlay_canvas_matches_image_height_with_side_panel(self, page: Page, base_url: str):
        self._setup(page)
        _mock_image_gaps(
            page,
            [
                {
                    "img-1": [
                        _gap("lbl-dog", "Dog"),
                        _gap("lbl-bird", "Bird"),
                        _gap("lbl-fox", "Fox"),
                        _gap("lbl-weather", "Weather", "image"),
                    ]
                }
            ],
        )

        _open_modal(page, base_url)

        thumb = page.locator(".review-card--image .thumb").first
        thumb.hover()

        popover = page.locator(".thumb-hover-popover")
        expect(popover).to_have_class(re.compile(r"\bvisible\b"))
        expect(popover).to_have_class(re.compile(r"\bhas-side\b"))

        img_box = page.locator(".thumb-hover-popover img.popover-img").bounding_box()
        canvas_box = page.locator(".thumb-hover-popover canvas.popover-canvas").bounding_box()

        # The overlay canvas must be sized to the actual rendered image, not to
        # whatever height the flex row stretched its wrapper to.
        assert canvas_box["height"] == pytest.approx(img_box["height"], abs=1.0), (
            f"overlay canvas height {canvas_box['height']} does not match "
            f"image height {img_box['height']} — bboxes will be drawn misaligned"
        )
