"""
Tests for Staging step 6 of the preview-card-sharing refactor: the classifier
branch of the Run-Model-on-Results preview
(`runClassifierOnResults.js`'s `openClassifierTagConfirmationModal`) giving its
IMAGE-LEVEL candidate cards the same select + relabel-context-menu parity the
detector preview got in step 5 — MINUS geometry (a classifier proposes a
whole-image label, not a box).

Asserts that a classifier candidate preview card can be:
  (a) click-selected (gets the `.selected` class via the shared cardGrid
      controller, independent of the "will Apply" checkbox),
  (b) right-clicked to change its proposed label (the proposed chip updates in
      place, mutating only the in-memory candidate — no DB write), and that the
      context menu offers ONLY Relabel + Remove-from-preview — no geometry /
      similar-search / verification / tags (image-level, scope decision 3).

The preview candidates are transient in-memory objects; nothing touches the DB
until Apply. We drive straight into `openClassifierTagConfirmationModal`
(bypassing the run-dialog/task-poll plumbing that
test_run_classifier_on_results.py already covers) and mock the small set of API
endpoints the cards need.

Run with:
    docker compose --profile dev run --rm --no-deps -w /project whiskerwatcher-dev \
        python3.11 -m pytest playwright_tests/test_run_classifier_preview_cards.py -q
"""

import json
import re

from playwright.sync_api import Page, expect

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}
LABEL_CORRUPT = {"uuid": "lbl-corrupt", "name": "Corrupt", "color": "#888888", "parentUuid": None}
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
IMAGE_2 = {"uuid": "img-2", "filename": "folder/image2.jpg", "bboxes": [BBOX_1 | {"uuid": "bbox-2"}]}
PAGE_1_IMAGES = [IMAGE_1, IMAGE_2]

# Two image-level classifier candidates predicting the "Corrupt" label.
CANDIDATE_1 = {
    "imageUuid": "img-1",
    "imagePath": "folder/image1.jpg",
    "predictedLabelUuid": "lbl-corrupt",
    "predictedLabelName": "Corrupt",
    "confidence": 0.91,
}
CANDIDATE_2 = {
    "imageUuid": "img-2",
    "imagePath": "folder/image2.jpg",
    "predictedLabelUuid": "lbl-corrupt",
    "predictedLabelName": "Corrupt",
    "confidence": 0.68,
}

_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _mock_review_queue(page: Page, images: list):
    page.route(
        "**/api/images/review",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "images": images, "hasMore": False, "totalCount": len(images)}),
        ),
    )


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


def _mock_image_thumbnails(page: Page):
    # Registered after navigation — "/thumbnail" is a substring of the
    # thumbnailHoverPreview.js module URL too, so mocking it before the page's
    # own module graph loads breaks that import.
    page.route(
        lambda url: "/thumbnail" in url and "/bbox/" not in url,
        lambda route: route.fulfill(status=200, content_type="image/png", body=_TINY_PNG),
    )


def _navigate_to_review(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-review']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#review-root .review-grid", timeout=4000)


def _open_confirm_modal(page: Page, candidates: list):
    page.evaluate(
        """async (candidates) => {
            const { openClassifierTagConfirmationModal } =
                await import('/app-static/js/app/review/runClassifierOnResults.js');
            const labels = new Map([
                ['lbl-cat', { uuid: 'lbl-cat', name: 'Cat', color: '#ff0000', scope: 'object' }],
                ['lbl-corrupt', { uuid: 'lbl-corrupt', name: 'Corrupt', color: '#888888', scope: 'object' }],
            ]);
            window.__refreshCount = 0;
            openClassifierTagConfirmationModal(candidates, {
                modelName: 'corrupt-classifier',
                refresh: () => { window.__refreshCount++; },
                labels,
            });
        }""",
        candidates,
    )


class TestClassifierPreviewCardBehavior:
    def _setup(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_CORRUPT])
        _mock_tags(page, [UNVERIFIED_TAG])
        _navigate_to_review(page, base_url)
        _mock_image_thumbnails(page)

    def test_card_click_selects(self, page: Page, base_url: str):
        """Clicking a classifier candidate card body toggles the shared-controller
        selection (the `.selected` class), independent of the Apply checkbox."""
        self._setup(page, base_url)
        _open_confirm_modal(page, [CANDIDATE_1, CANDIDATE_2])

        card = page.locator(".classifier-match-confirm-modal .relabel-modal-card").first
        expect(card).to_be_visible()
        expect(card).not_to_have_class(re.compile(r"\bselected\b"))

        # Click the card body (not the checkbox). Selection is applied on a
        # 220ms debounce, so wait for the class to appear.
        card.locator(".card-info").click()
        expect(card).to_have_class(re.compile(r"\bselected\b"), timeout=2000)

        # The Apply checkbox stays checked (its "will Apply" state is unchanged
        # by a selection click).
        expect(card.locator("input[type=checkbox]")).to_be_checked()

    def test_right_click_relabel_updates_proposed_chip(self, page: Page, base_url: str):
        """Right-click -> Apply label -> pick Cat updates the proposed chip in
        place, mutating only the in-memory candidate (no DB call). The menu offers
        ONLY relabel + remove — no geometry / similar / verification / tags."""
        self._setup(page, base_url)
        _open_confirm_modal(page, [CANDIDATE_1])

        modal = page.locator(".classifier-match-confirm-modal")
        chip = modal.locator(".relabel-modal-proposed-chip").first
        expect(chip).to_have_text("Corrupt")

        card = modal.locator(".relabel-modal-card").first
        card.locator(".card-info").click(button="right")

        # The candidateImage menu is image-level: Relabel + Remove-from-preview
        # ONLY — no geometry / similar-search / verification / tag editing.
        menu = page.locator(".ctx-menu").first
        expect(menu).to_be_visible(timeout=2000)
        item_text = menu.inner_text()
        assert "Apply label" in item_text
        assert "Remove from preview" in item_text
        assert "Edit geometry" not in item_text
        assert "similar" not in item_text.lower()
        assert "Verification" not in item_text
        assert "Tags" not in item_text

        # Hover "Apply label" to open the label-picker flyout, type to filter,
        # and click the "Cat" row.
        menu.locator(".ctx-menu-item", has_text="Apply label").hover()
        picker_input = page.locator(".ctx-menu input").first
        expect(picker_input).to_be_visible(timeout=2000)
        picker_input.fill("Cat")
        page.locator(".ctx-menu").last.get_by_text("Cat", exact=False).first.click()

        expect(chip).to_have_text("Cat", timeout=2000)

    def test_right_click_remove_drops_candidate_from_preview(self, page: Page, base_url: str):
        """Remove-from-preview drops the candidate card and reduces the Apply
        count — the candidate is excluded from a subsequent Apply, with no DB
        call (batch contract preserved)."""
        self._setup(page, base_url)
        _open_confirm_modal(page, [CANDIDATE_1, CANDIDATE_2])

        modal = page.locator(".classifier-match-confirm-modal")
        cards = modal.locator(".relabel-modal-card")
        expect(cards).to_have_count(2)
        apply_btn = modal.locator("button", has_text="Apply to")
        expect(apply_btn).to_contain_text("Apply to 2 checked")

        cards.first.locator(".card-info").click(button="right")
        menu = page.locator(".ctx-menu").first
        expect(menu).to_be_visible(timeout=2000)
        menu.locator(".ctx-menu-item", has_text="Remove from preview").click()

        expect(cards).to_have_count(1)
        expect(apply_btn).to_contain_text("Apply to 1 checked")
