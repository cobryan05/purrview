"""
Tests that the Review tab's "★ Most Unique First" button honours the active filters.

The button used to call openDissimilarGlobalModal() with no scope at all, so the
server always ranked its own unverified-bbox pool and the result ignored whatever
the user had filtered the grid down to. It now pushes the filtered image UUID set
down as `candidateUuids` — but only when a filter is actually active, so an
untouched panel keeps the server's unverified-first default.

Mocking follows test_review_workflow_ui.py (route interception over the conftest's
empty-data proxies).

Run with:
    pytest playwright_tests/test_most_unique_first_filters.py -p no:xdist -p no:asyncio -v
"""

import json

from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}
LABEL_DOG = {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00", "parentUuid": None}
UNVERIFIED_TAG = {"uuid": UNVERIFIED_TAG_UUID, "name": "Unverified", "color": "#888", "kind": "system"}

BBOX_1 = {
    "uuid": "bbox-1",
    "x": 0.1,
    "y": 0.1,
    "width": 0.2,
    "height": 0.2,
    "label": {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000"},
    "labelUuid": "lbl-cat",
    "tags": [{"uuid": UNVERIFIED_TAG_UUID, "icon": None}],
}
IMAGE_1 = {"uuid": "img-1", "filename": "folder/image1.jpg", "bboxes": [BBOX_1]}
IMAGE_2 = {"uuid": "img-2", "filename": "folder/image2.jpg", "bboxes": []}
QUEUE_IMAGES = [IMAGE_1, IMAGE_2]

IMAGES_BY_UUID = {img["uuid"]: img for img in QUEUE_IMAGES}


def _mock_review_queue(page: Page):
    page.route(
        "**/api/images/review",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {"status": "success", "images": QUEUE_IMAGES, "hasMore": False, "totalCount": len(QUEUE_IMAGES)}
            ),
        ),
    )


def _mock_labels(page: Page):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": [LABEL_CAT, LABEL_DOG]}),
        ),
    )


def _mock_tags(page: Page):
    page.route(
        "**/api/tags/list",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"tags": [UNVERIFIED_TAG]})
        ),
    )


def _mock_batch_by_uuids(page: Page):
    def _handle(route: Route):
        req = json.loads(route.request.post_data or "{}")
        images = [IMAGES_BY_UUID[u] for u in req.get("imageUuids", []) if u in IMAGES_BY_UUID]
        route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "images": images})
        )

    page.route("**/api/images/batch-by-uuids", _handle)


def _mock_dissimilar_global(page: Page, captured: list):
    """Record each request body and answer with the queue images, most-unique first."""

    def _handle(route: Route):
        captured.append(json.loads(route.request.post_data or "{}"))
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {"status": "success", "matches": [{"imageUuid": "img-1", "rank": 0}, {"imageUuid": "img-2", "rank": 1}]}
            ),
        )

    page.route("**/api/images/find-dissimilar-global", _handle)


def _navigate_to_review(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
    page.locator("#tab-image-tagging .section-tab-button[data-tab='tab-review']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#review-root .review-grid")


def _apply_label_filter(page: Page):
    """Check the Cat label in the Labels dropdown — enough to make isDefault() false."""
    # The Labels dropdown lives in the filter popover (it used to sit in the
    # toolbar, always visible), so the popover has to be opened to reach it.
    page.get_by_test_id("review-filter-toggle").click()
    trigger = page.locator("#review-root .ms-dropdown[data-filter-key='labels-v2'] .ms-dropdown__trigger")
    trigger.wait_for(state="visible", timeout=5000)
    trigger.click()
    page.wait_for_timeout(150)
    page.locator("#review-root .ms-dropdown[data-filter-key='labels-v2'] input[value='lbl-cat']").check()
    page.mouse.click(0, 0)
    page.wait_for_timeout(300)


def _click_unique_first(page: Page):
    page.locator("#review-root button:has-text('Most Unique First')").click()
    page.wait_for_selector(".modal-count-input")


def _setup(page: Page, captured: list):
    _mock_review_queue(page)
    _mock_labels(page)
    _mock_tags(page)
    _mock_batch_by_uuids(page)
    _mock_dissimilar_global(page, captured)


class TestMostUniqueFirstRespectsFilters:
    def test_no_filter_sends_no_candidates(self, page: Page, base_url: str):
        """An untouched filter panel leaves the server's unverified scoping alone."""
        captured = []
        _setup(page, captured)
        _navigate_to_review(page, base_url)

        _click_unique_first(page)

        # `_click_unique_first` waits for the modal's count input, which the modal
        # renders on open — the dissimilar-global fetch is issued alongside it, not
        # before it. Poll for the request that is actually under test.
        wait_until(page, lambda: len(captured) == 1, lambda: f"expected 1 dissimilar query, saw {captured}")
        assert captured[0].get("candidateUuids") is None

    def test_active_filter_sends_filtered_candidates(self, page: Page, base_url: str):
        captured = []
        _setup(page, captured)
        _navigate_to_review(page, base_url)
        _apply_label_filter(page)

        _click_unique_first(page)

        wait_until(page, lambda: len(captured) == 1, lambda: f"expected 1 dissimilar query, saw {captured}")
        assert sorted(captured[0].get("candidateUuids") or []) == ["img-1", "img-2"]

    def test_button_is_restored_after_the_filtered_fetch(self, page: Page, base_url: str):
        """The click shows a busy state while resolving the filtered set; it must not
        stay disabled or keep the 'Loading…' label once the modal opens."""
        captured = []
        _setup(page, captured)
        _navigate_to_review(page, base_url)
        _apply_label_filter(page)

        _click_unique_first(page)

        btn = page.locator("#review-root button:has-text('Most Unique First')")
        expect(btn).to_be_enabled()
        expect(btn).to_have_text("★ Most Unique First")

    def test_filtered_modal_title_marks_the_scope(self, page: Page, base_url: str):
        captured = []
        _setup(page, captured)
        _navigate_to_review(page, base_url)
        _apply_label_filter(page)

        _click_unique_first(page)

        expect(page.locator(".compare-modal-header strong").last).to_have_text("Label Most Unique First (filtered)")
