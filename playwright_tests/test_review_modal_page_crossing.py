"""
The canvas modal's prev/next must cross results-page boundaries.

The nav list handed to the modal is built from the cards on the *current* page,
so stepping off the end of page 1 stopped dead with a toast telling the user to
go and turn the page themselves — close the modal, click Next, re-open. Which
page a result landed on is an artefact of how it was fetched; nobody browsing
search results thinks in pages of fifty.

The server is stubbed with a two-page keyset fake: the bug is entirely in what
the client does at the boundary, and two pages is the smallest set that has one.
"""

import base64
import json

from playwright.sync_api import Page, Route, expect

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"
TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk" "+A8AAQUBAScY42YAAAAASUVORK5CYII="

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}

PAGE_1 = [{"uuid": f"img-a{i}", "filename": f"deck/a{i}.jpg", "bboxes": []} for i in range(3)]
PAGE_2 = [{"uuid": f"img-b{i}", "filename": f"deck/b{i}.jpg", "bboxes": []} for i in range(3)]
BY_UUID = {img["uuid"]: img for img in PAGE_1 + PAGE_2}
BY_PATH = {img["filename"]: img for img in PAGE_1 + PAGE_2}
TOTAL = len(PAGE_1) + len(PAGE_2)


def _mock_review_queue(page: Page):
    """Keyset pagination over two pages, cursor-driven like the real endpoint."""

    def _handle(route: Route):
        after = json.loads(route.request.post_data or "{}").get("afterUuid")
        if after is None:
            images, has_more = PAGE_1, True
        elif after == PAGE_1[-1]["uuid"]:
            images, has_more = PAGE_2, False
        else:
            images, has_more = [], False
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                # The real endpoint only counts on the first page (-1 elsewhere).
                {
                    "status": "success",
                    "images": images,
                    "hasMore": has_more,
                    "totalCount": TOTAL if after is None else -1,
                }
            ),
        )

    page.route("**/api/images/review", _handle)


def _mock_supporting_endpoints(page: Page):
    png_bytes = base64.b64decode(TINY_PNG_B64)
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": [LABEL_CAT]}),
        ),
    )
    page.route(
        "**/api/tags/list",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "tags": []})
        ),
    )
    page.route(
        "**/thumbnail?*",
        lambda route: route.fulfill(status=200, content_type="image/png", body=png_bytes),
    )

    def _file(route: Route):
        # ".../api/images/<urlencoded path>/file"
        path = route.request.url.split("/api/images/")[1].rsplit("/file", 1)[0]
        path = path.replace("%2F", "/")
        img = BY_PATH.get(path, PAGE_1[0])
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "imageBase64": TINY_PNG_B64,
                    "mimeType": "image/png",
                    "image": {"uuid": img["uuid"], "filename": img["filename"], "bboxes": []},
                }
            ),
        )

    page.route("**/api/images/*/file", _file)


def _navigate_to_review(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
    page.locator("#tab-image-tagging .section-tab-button[data-tab='tab-review']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#review-root .review-grid")


def _open_modal_on(page: Page, base_url: str, image_uuid: str):
    _mock_supporting_endpoints(page)
    _mock_review_queue(page)
    _navigate_to_review(page, base_url)
    page.locator(f"#review-root [data-image-uuid='{image_uuid}'] img.thumb").first.dblclick()
    modal = page.get_by_test_id("canvas-edit-modal")
    expect(modal).to_be_visible(timeout=8000)
    return modal


def _goto_page_2(page: Page):
    page.locator("#review-root [data-testid='review-page-next']").click()
    expect(page.locator("#review-root .review-toolbar__page-info")).to_contain_text("Page 2")


class TestModalNavigationCrossesPageBoundaries:
    def test_next_on_the_last_card_of_a_page_opens_the_next_page(self, page: Page, base_url: str):
        """The reported bug. Nothing about the last card of page 1 should be
        different from the ones before it."""
        modal = _open_modal_on(page, base_url, PAGE_1[-1]["uuid"])
        expect(modal.get_by_test_id("canvas-edit-modal-title")).to_contain_text(PAGE_1[-1]["filename"])

        modal.get_by_test_id("canvas-edit-modal-next").click()

        expect(modal.get_by_test_id("canvas-edit-modal-title")).to_contain_text(PAGE_2[0]["filename"])

    def test_the_grid_behind_follows_onto_the_new_page(self, page: Page, base_url: str):
        """Crossing forward moves the results page too, so closing the modal
        leaves the user where they navigated to rather than where they started."""
        modal = _open_modal_on(page, base_url, PAGE_1[-1]["uuid"])
        expect(page.locator("#review-root .review-toolbar__page-info")).to_contain_text("Page 1")

        modal.get_by_test_id("canvas-edit-modal-next").click()
        expect(modal.get_by_test_id("canvas-edit-modal-title")).to_contain_text(PAGE_2[0]["filename"])

        expect(page.locator("#review-root .review-toolbar__page-info")).to_contain_text("Page 2")

    def test_prev_on_the_first_card_of_a_page_opens_the_previous_page(self, page: Page, base_url: str):
        """The symmetric case, and the one with the fiddly landing spot: going
        back must arrive at the LAST result of the previous page, not its first."""
        _mock_supporting_endpoints(page)
        _mock_review_queue(page)
        _navigate_to_review(page, base_url)
        _goto_page_2(page)

        page.locator(f"#review-root [data-image-uuid='{PAGE_2[0]['uuid']}'] img.thumb").first.dblclick()
        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)
        expect(modal.get_by_test_id("canvas-edit-modal-title")).to_contain_text(PAGE_2[0]["filename"])

        modal.get_by_test_id("canvas-edit-modal-prev").click()

        expect(modal.get_by_test_id("canvas-edit-modal-title")).to_contain_text(PAGE_1[-1]["filename"])
        expect(page.locator("#review-root .review-toolbar__page-info")).to_contain_text("Page 1")

    def test_the_end_of_the_last_page_is_still_the_end(self, page: Page, base_url: str):
        """The counterweight: paging past a boundary must not turn into paging
        past the result set. On the final card, Next stays put and disabled."""
        _mock_supporting_endpoints(page)
        _mock_review_queue(page)
        _navigate_to_review(page, base_url)
        _goto_page_2(page)

        page.locator(f"#review-root [data-image-uuid='{PAGE_2[-1]['uuid']}'] img.thumb").first.dblclick()
        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)

        expect(modal.get_by_test_id("canvas-edit-modal-next")).to_be_disabled()
        expect(modal.get_by_test_id("canvas-edit-modal-title")).to_contain_text(PAGE_2[-1]["filename"])

    def test_next_is_enabled_on_the_last_card_of_a_non_final_page(self, page: Page, base_url: str):
        """Disabled must mean "nothing further exists", not "nothing further on
        this page" — otherwise the button contradicts the arrow keys."""
        modal = _open_modal_on(page, base_url, PAGE_1[-1]["uuid"])

        expect(modal.get_by_test_id("canvas-edit-modal-next")).to_be_enabled()

    def test_prev_is_disabled_on_the_very_first_card(self, page: Page, base_url: str):
        """The other end of the same rule: page 1's first card has no previous
        page to reach for."""
        modal = _open_modal_on(page, base_url, PAGE_1[0]["uuid"])

        expect(modal.get_by_test_id("canvas-edit-modal-prev")).to_be_disabled()
