"""A label created while Review is open reaches Review's own consumers.

`createReviewPanel` fetches the label map once and hands it *by reference* to the bulk
apply modals, the canvas modal, run-model and verify-model. A label created afterwards
was missing from every one of them until the page was reloaded — hit in practice by
creating a new label and immediately trying to bulk-mark it absent on existing images,
which is the whole reason to add a label to a workspace that already has pictures in it.

The fix refreshes that map in place rather than replacing it, so the assertion worth
pinning is that the modal's picker gains the label *without a reload*. Asserting the
before state too is what makes this a regression test rather than a check that the mock
works: the picker must not list the label until the event fires.

Run with:
    pytest playwright_tests/test_review_new_label_available.py -p no:asyncio -v
"""

import json

from playwright.sync_api import Page

from playwright_tests.test_review_workflow_ui import (
    LABEL_CAT,
    LABEL_DOG,
    PAGE_1_IMAGES,
    _mock_misc_review_endpoints,
    _mock_review_queue,
    _mock_tags,
    _navigate_to_review,
)
from playwright_tests.waits import wait_until

LABEL_POSSUM = {"uuid": "lbl-possum", "name": "Possum", "color": "#8888ff", "parentUuid": None}


def _label_names_in_bulk_modal(page: Page) -> list:
    """Every label the image bulk-apply modal offers, by name.

    Stripped: the row renders the name beside a colour swatch, so each entry arrives with
    leading whitespace and a bare `in` check against the raw text silently never matches.
    """
    return [n.strip() for n in page.locator(".bulk-modal-picker--labels .ms-item__name").all_text_contents()]


def _open_image_bulk_modal(page: Page):
    """Select every image on the page and open Apply to Selected.

    The check is retried rather than asserted once: a label change re-renders the grid and
    resets select-all, so a tick placed while that is in flight is silently undone and the
    Apply button stays disabled with nothing selected.
    """
    cb = page.locator(".review-toolbar__select-all input[type=checkbox]")

    def _ticked():
        if not cb.is_checked():
            cb.click()
        return cb.is_checked()

    wait_until(page, _ticked, "select-all never stayed ticked")
    page.locator("[data-testid='review-bulk-apply']").click()
    page.wait_for_selector(".bulk-modal-overlay .bulk-modal-picker--labels")


def _close_image_bulk_modal(page: Page):
    """Dismiss it and wait for the overlay to actually go.

    Waiting for detachment, not just pressing Escape: the next open queries the same
    selector, and a modal still on its way out answers it with the rows it is about to
    lose — which reads as "the label never arrived" rather than as a timing problem.
    """
    page.keyboard.press("Escape")
    page.wait_for_selector(".bulk-modal-overlay", state="detached")


class TestANewLabelReachesTheBulkApplyModal:
    def test_the_picker_gains_a_label_created_after_the_panel_mounted(self, page: Page, base_url: str):
        served = [LABEL_CAT, LABEL_DOG]

        # One route for both verbs, so the POST that creates the label is the same thing
        # that changes what the GET returns — the sequence the user actually performed.
        def labels_route(route):
            if route.request.method == "POST":
                served.append(LABEL_POSSUM)
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "label": LABEL_POSSUM}),
                )
                return
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "labels": served}),
            )

        page.route(
            lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
            labels_route,
        )
        _mock_tags(page, [])
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_misc_review_endpoints(page)
        _navigate_to_review(page, base_url)

        _open_image_bulk_modal(page)
        before = _label_names_in_bulk_modal(page)
        assert "Possum" not in before, f"the new label must not be listed before it exists: {before}"
        assert "Cat" in before, f"the picker should already be populated: {before}"
        _close_image_bulk_modal(page)

        # The real creation path: it clears the label cache and publishes LABELS_UPDATED.
        page.evaluate("""async () => {
                const { createNewLabel } = await import('/app-static/js/shared/api/labels.js');
                await createNewLabel('Possum', '#8888ff', null, 'standard', 'object');
            }""")

        _open_image_bulk_modal(page)
        after = _label_names_in_bulk_modal(page)
        assert "Possum" in after, f"a label created while Review was open must be selectable: {after}"

    def test_a_bare_labels_updated_does_not_empty_the_picker(self, page: Page, base_url: str):
        """The hazard in refreshing a Map in place.

        `fetchLabels` returns the cache's own Map, so a publisher that skips the cache
        clear hands the refresh the very object it is about to overwrite. Clearing it
        first and then copying out of it wipes the label list instead of refreshing it,
        and the picker comes up empty — worse than the staleness being fixed.
        """
        page.route(
            lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "labels": [LABEL_CAT, LABEL_DOG]}),
            ),
        )
        _mock_tags(page, [])
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_misc_review_endpoints(page)
        _navigate_to_review(page, base_url)

        page.evaluate("""async () => {
                const { events, EventTypes } = await import('/app-static/js/shared/events/index.js');
                events.publish(EventTypes.LABELS_UPDATED, {});
            }""")
        _open_image_bulk_modal(page)
        assert _label_names_in_bulk_modal(page) == ["Cat", "Dog"]
