"""
Tests for M2 of the "unify Review filters" plan — infinite scroll in
custom/fetchImages mode (similarity/dissimilarity modals), replacing fixed
Prev/Next pagination with a scroll-triggered `_loadMoreCurated()` (main.js)
that fetches chunks via a stubbed `fetchImages({ afterUuid, limit })` and
appends them (entryView.appendEntries) without a full DOM rebuild.

The candidate stub below implements real afterUuid-cursor pagination (unlike
test_modal_curated_filters.py's fixed unpaginated CANDIDATES list) so the
chunked bootstrap loop and _loadMoreCurated actually have multiple pages to
walk through.

Run with:
    docker compose exec -T mono-dev bash -lc \
      'cd /project/playwright_tests && python -m pytest test_modal_infinite_scroll.py \
       -o addopts="" -p no:xdist -q'
"""

import json
import re

import pytest
from playwright.sync_api import Page, expect

_SELECTED_RE = re.compile(r"selected")

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}

N_CANDIDATES = 25
CANDIDATES = [
    {
        "uuid": f"img-{i:03d}",
        "filename": f"folder/image{i:03d}.jpg",
        "bboxes": [
            {
                "uuid": f"bbox-{i:03d}",
                "x": 0.1,
                "y": 0.1,
                "width": 0.2,
                "height": 0.2,
                "label": {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000"},
                "labelUuid": "lbl-cat",
                "tags": [],
            }
        ],
    }
    for i in range(N_CANDIDATES)
]


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
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"tags": []})),
    )


def _mock_image_labels_batch(page: Page):
    page.route(
        "**/api/images/labels/batch*",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "imageLabels": {}})
        ),
    )


def _open_paginated_modal(page: Page, base_url: str):
    """Open showReviewModal with a stub fetchImages that implements real
    afterUuid-cursor pagination over CANDIDATES, so main.js's chunked
    bootstrap + _loadMoreCurated have multiple pages to walk."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.evaluate(
        """async (candidates) => {
            const { showReviewModal } = await import('/app-static/js/app/review/main.js');
            window.__refresh = await showReviewModal({
                title: 'Infinite Scroll Test',
                initialViewFilter: 'images',
                fetchImages: async ({ afterUuid, limit }) => {
                    const startIdx = afterUuid ? candidates.findIndex(c => c.uuid === afterUuid) + 1 : 0;
                    return candidates.slice(startIdx, startIdx + limit);
                },
            });
        }""",
        CANDIDATES,
    )
    page.wait_for_selector(".review-grid", timeout=4000)


def _select_chunk_size(page: Page, value: str):
    """The page-size <select> (className 'filter-page-size') doubles as the
    infinite-scroll chunk size in fetchImages mode — main.js's `_pageSize`."""
    page.locator("select.filter-page-size").select_option(value)


def _scroll_content_to_bottom(page: Page):
    page.evaluate("""() => {
            let el = document.querySelector('.review-grid');
            while (el && getComputedStyle(el).overflow !== 'auto') el = el.parentElement;
            if (el) el.scrollTop = el.scrollHeight;
        }""")


def _scroll_until_count(page: Page, locator, target_count: int, attempts: int = 10):
    """Repeatedly scroll to the (growing) bottom, like a real user would —
    M2 only preloads until the viewport is full, so each load can grow the
    container past a single scrollTop=scrollHeight snapshot, requiring
    another scroll to reach newly-appended content below the fold."""
    for _ in range(attempts):
        if locator.count() >= target_count:
            break
        _scroll_content_to_bottom(page)
        page.wait_for_timeout(150)
    expect(locator).to_have_count(target_count)


class TestInfiniteScrollNoPagination:
    def test_prev_next_hidden_in_modal_mode(self, page: Page, base_url: str):
        _mock_labels(page)
        _mock_tags(page)
        _mock_image_labels_batch(page)

        _open_paginated_modal(page, base_url)

        expect(page.locator(".review-toolbar__btn", has_text="Prev")).to_be_hidden()
        expect(page.locator(".review-toolbar__btn", has_text="Next")).to_be_hidden()


class TestInfiniteScrollLoadsMoreOnScroll:
    """Uses Playwright's auto-retrying `expect()` rather than `wait_for_timeout()`
    + a point-in-time `.count()` snapshot (NOTES.md anti-pattern) — the scroll
    sentinel's generous rootMargin auto-preloads a chunk or two ahead of the
    literal scroll position (intentional: M2's "keep pulling until the
    viewport is full"), so intermediate chunk-boundary counts are inherently
    racy and only the eventual settled state is worth asserting on.
    """

    def test_all_candidates_eventually_load_via_scroll(self, page: Page, base_url: str):
        _mock_labels(page)
        _mock_tags(page)
        _mock_image_labels_batch(page)

        _open_paginated_modal(page, base_url)

        # Shrink chunk size to 10 so 25 candidates span 3 chunks (10, 10, 5) —
        # too tall to all auto-preload within the sentinel's rootMargin, so a
        # real scroll is required to reach the last chunk.
        _select_chunk_size(page, "10")

        # Preload settles at some partial count (rootMargin proximity) short of
        # the full 25 — assert that plateau is real (not already complete)
        # before proving repeated scrolling pushes it the rest of the way.
        expect(page.locator("[data-image-uuid]")).not_to_have_count(25, timeout=2000)

        _scroll_until_count(page, page.locator("[data-image-uuid]"), 25)

        # No more candidates left — further scrolling must not error or duplicate cards.
        _scroll_content_to_bottom(page)
        page.wait_for_timeout(200)
        expect(page.locator("[data-image-uuid]")).to_have_count(25)

    def test_selection_survives_a_scroll_load(self, page: Page, base_url: str):
        """Regression guard: appendEntries (not a full rerender) must not wipe
        a selection made before the scroll-triggered load fires."""
        _mock_labels(page)
        _mock_tags(page)
        _mock_image_labels_batch(page)

        _open_paginated_modal(page, base_url)
        _select_chunk_size(page, "10")

        first_card = page.locator("[data-image-uuid]").first
        first_card.locator("input[type=checkbox]").click()
        expect(first_card).to_have_class(_SELECTED_RE)

        _scroll_until_count(page, page.locator("[data-image-uuid]"), 25)
        expect(first_card).to_have_class(_SELECTED_RE)
