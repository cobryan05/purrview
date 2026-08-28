"""
Regression test for a keyboard-focus bug in the modals built by hand in
app/review/main.js (and imageBulkActionModal.js): opening the modal never
moved focus into it, so document.activeElement stayed on whatever was
focused behind the overlay (e.g. a search input). Arrow keys / Tab then kept
acting on that background element — invisible, since the modal covers it —
which read as "the modal stopped responding to keypresses and the search
results behind it scrolled instead."

The fix is `trapModalFocus()` in ui/utils/modalShell.js, now called by every
hand-rolled modal right after it's attached to the document: focus moves
inside the dialog on open, and is trapped there (Tab included) until close.

Run with:
    docker compose exec -T encounterwatch-dev bash -lc \
      'cd /project && python -m pytest playwright_tests/test_modal_focus_trap.py -p no:asyncio -q'
"""

import json

from playwright.sync_api import Page, expect

IMAGE_1 = {"uuid": "img-1", "filename": "folder/image1.jpg", "bboxes": []}


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


class TestShowReviewModalFocusTrap:
    def _setup(self, page: Page):
        _mock_labels(page, [])
        _mock_tags(page, [])
        _mock_image_labels_batch(page)

    def test_opening_the_modal_moves_focus_inside_it(self, page: Page, base_url: str):
        """A background input focused before the modal opens must lose focus
        to the dialog — otherwise arrow keys/Tab silently act on the page
        behind the (visually on-top) modal instead of the modal itself."""
        self._setup(page)
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        # Simulate a background element (e.g. a search box) holding focus
        # when the modal is triggered.
        page.evaluate("""() => {
                const decoy = document.createElement('input');
                decoy.id = '__decoy_search_input';
                document.body.appendChild(decoy);
                decoy.focus();
            }""")
        expect(page.locator("#__decoy_search_input")).to_be_focused()

        page.evaluate(
            """async ({ images }) => {
                const { showReviewModal } = await import('/app-static/js/app/review/main.js');
                window.__refresh = await showReviewModal({
                    title: 'Focus Trap Test',
                    initialViewFilter: 'images',
                    fetchImages: async () => images,
                });
            }""",
            {"images": [IMAGE_1]},
        )
        page.wait_for_selector(".review-grid")

        focus_is_in_decoy = page.evaluate(
            "() => document.activeElement && document.activeElement.id === '__decoy_search_input'"
        )
        assert not focus_is_in_decoy, "focus stayed on the background input after the modal opened"

        focus_is_in_dialog = page.evaluate(
            "() => !!document.activeElement && !!document.activeElement.closest('.compare-modal-dialog')"
        )
        assert focus_is_in_dialog, "focus did not move inside the modal dialog"

    def test_closing_the_modal_restores_focus_to_the_trigger(self, page: Page, base_url: str):
        self._setup(page)
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        page.evaluate("""() => {
                const trigger = document.createElement('button');
                trigger.id = '__trigger_btn';
                document.body.appendChild(trigger);
                trigger.focus();
            }""")

        page.evaluate(
            """async ({ images }) => {
                const { showReviewModal } = await import('/app-static/js/app/review/main.js');
                window.__refresh = await showReviewModal({
                    title: 'Focus Restore Test',
                    initialViewFilter: 'images',
                    fetchImages: async () => images,
                });
            }""",
            {"images": [IMAGE_1]},
        )
        page.wait_for_selector(".review-grid")

        page.keyboard.press("Escape")
        page.wait_for_selector(".compare-modal-dialog", state="detached", timeout=4000)

        expect(page.locator("#__trigger_btn")).to_be_focused()
