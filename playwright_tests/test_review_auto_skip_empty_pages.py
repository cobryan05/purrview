"""
The review grid's auto-skip must skip pages that are *blank*, not pages that
merely have no bounding boxes.

render() re-fetches the next page when the page it just built "looks empty".
That check counted bbox entries only, regardless of which view was active — so
in the All/Images views, where every returned image is a card in its own right,
a page full of perfectly good un-annotated images counted as empty and was
thrown away. Filter the Review tab by a camera source whose images are mostly
un-annotated (the real case: 1535 images, 1 of them with a box) and the grid
burned all ten skips in a row and landed on the *last* page of the result set.

The user-visible damage was worse than a wasted fetch, because a skip
overwrites the current page cursor instead of pushing one: the pager still read
"Page 1 of 7" with Prev disabled while the grid showed the tail of the set, and
the counts on "Apply to filtered images" — which come from the server and were
right all along — disagreed with everything on screen. No click was involved,
so there was nothing to undo and nothing to explain it.

The server is stubbed: the bug is entirely in how the client decides a page is
empty, and a fake keyset lets a three-page result set fit in a test.
"""

import json

import pytest
from playwright.sync_api import Page, Route, expect

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}

# Three pages. Only the last carries a bbox — the shape that made the old check
# walk the whole set. Page 1 and 2 are exactly what the user should be looking
# at in the All/Images views.
PAGE_1 = [{"uuid": f"img-a{i}", "filename": f"deck/a{i}.jpg", "bboxes": []} for i in range(3)]
PAGE_2 = [{"uuid": f"img-b{i}", "filename": f"deck/b{i}.jpg", "bboxes": []} for i in range(3)]
BBOX = {
    "uuid": "bbox-1",
    "x": 10,
    "y": 10,
    "width": 50,
    "height": 50,
    "labelUuid": "lbl-cat",
    "tags": [{"uuid": UNVERIFIED_TAG_UUID, "icon": None}],
}
PAGE_3 = [{"uuid": "img-c0", "filename": "deck/c0.jpg", "bboxes": [BBOX]}]

TOTAL = len(PAGE_1) + len(PAGE_2) + len(PAGE_3)


def _mock_review_queue(page: Page, requests: list):
    """A keyset-paginated fake: the cursor picks the page, `requireBboxes` drops
    the un-annotated images the way the real endpoint does."""

    def _handle(route: Route):
        req = json.loads(route.request.post_data or "{}")
        requests.append(req)

        if req.get("requireBboxes"):
            # Server-side narrowing: only the annotated image survives, so the
            # whole result set is one page and auto-skip has nothing to skip.
            images, has_more = (PAGE_3, False) if req.get("afterUuid") is None else ([], False)
            total = len(PAGE_3)
        else:
            after = req.get("afterUuid")
            if after is None:
                images, has_more = PAGE_1, True
            elif after == PAGE_1[-1]["uuid"]:
                images, has_more = PAGE_2, True
            elif after == PAGE_2[-1]["uuid"]:
                images, has_more = PAGE_3, False
            else:
                images, has_more = [], False
            # The real endpoint only counts on the first page (-1 elsewhere).
            total = TOTAL if after is None else -1

        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "images": images, "hasMore": has_more, "totalCount": total}),
        )

    page.route("**/api/images/review", _handle)


def _mock_labels(page: Page):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": [LABEL_CAT]}),
        ),
    )


def _mock_tags(page: Page):
    page.route(
        "**/api/tags/list",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "tags": []})
        ),
    )


def _navigate_to_review(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
    page.locator("#tab-image-tagging .section-tab-button[data-tab='tab-review']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#review-root .review-grid")


def _shown_image_uuids(page: Page) -> list:
    return page.locator("#review-root [data-testid='review-card'][data-kind='image']").evaluate_all(
        "els => els.map(e => e.dataset.imageUuid)"
    )


def _setup(page: Page, base_url: str, requests: list):
    _mock_labels(page)
    _mock_tags(page)
    _mock_review_queue(page, requests)
    _navigate_to_review(page, base_url)


class TestAutoSkipOnlySkipsBlankPages:
    @pytest.mark.parametrize("view", ["all", "images"])
    def test_bbox_less_page_is_not_skipped(self, page: Page, base_url: str, view: str):
        """Page 1 has three images and no boxes. In a view that renders image
        cards that is a full page, so it must stay put."""
        requests = []
        _setup(page, base_url, requests)
        if view != "all":
            page.locator(f"#review-root .view-btn[data-view-key='{view}']").click()
            page.wait_for_timeout(400)

        expect(page.locator("#review-root [data-testid='review-card']")).to_have_count(len(PAGE_1))
        # The decisive assertion: page 1's images, not page 3's. Before the fix
        # this was ['img-c0'] — the tail of the set, reached without a click.
        assert _shown_image_uuids(page) == [i["uuid"] for i in PAGE_1]

    def test_pager_and_next_agree_with_what_is_shown(self, page: Page, base_url: str):
        """A silent skip left Next disabled on a set with two more pages in it,
        stranding the user on the last page while the label still said page 1."""
        requests = []
        _setup(page, base_url, requests)

        expect(page.locator("#review-root [data-testid='review-page-next']")).to_be_enabled()
        expect(page.locator("#review-root .review-toolbar__page-info")).to_contain_text("Page 1")

    def test_next_still_advances_through_the_bbox_less_pages(self, page: Page, base_url: str):
        """Not-skipping must not turn into not-paging: Next still walks forward."""
        requests = []
        _setup(page, base_url, requests)

        page.locator("#review-root [data-testid='review-page-next']").click()
        page.wait_for_timeout(500)

        assert _shown_image_uuids(page) == [i["uuid"] for i in PAGE_2]

    @pytest.mark.quarantine("a second request raced the assertion; 1 of 1 sighting")
    def test_first_page_costs_one_request(self, page: Page, base_url: str):
        """The skip loop fired a fetch per skipped page. Landing on page 1 should
        take exactly one review query."""
        requests = []
        _setup(page, base_url, requests)

        assert len(requests) == 1, f"auto-skip re-fetched: {[r.get('afterUuid') for r in requests]}"

    def test_bboxes_view_shows_only_the_annotated_image(self, page: Page, base_url: str):
        """The counterweight: in the Bboxes view a page with no boxes genuinely is
        blank, and that view must still end up on the annotated image."""
        requests = []
        _setup(page, base_url, requests)
        page.locator("#review-root .view-btn[data-view-key='bboxes']").click()
        page.wait_for_timeout(500)

        cards = page.locator("#review-root [data-testid='review-card'][data-kind='bbox']")
        expect(cards).to_have_count(1)
        assert cards.first.get_attribute("data-image-uuid") == "img-c0"
