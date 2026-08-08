"""
Regression tests for the curated-filter chunk loop in `createReviewPanel`
(fetchImages / modal mode).

`render()` and `_loadMoreCurated()` page the candidate list until the first
screenful is filled or the list is exhausted, applying the curated predicate
in memory to each chunk. When a filter rejects a whole chunk the loop keeps
going — so the cursor it sends has to actually advance.

It did not. Both loops sent `afterSortValue: null`, and every server-side
custom fetcher (the training-config / coverage modals) keyset-paginates on
`(created_at, uuid)`. Given a uuid with no sort value the server fell back to
comparing uuids against a created_at-ordered list, the page stopped advancing,
and `has_more` stayed true forever. Live symptom: opening "Needs Review" for a
1,294-image training config and typing a filename filter never finished — the
same 251 rows were re-served ~200 times in 69s before the walk was killed.

`test_modal_curated_filters.py` missed this because its stub returns 3
candidates and ignores the cursor: 3 < the 250 page size, so the loop exits
after one iteration and never paginates at all. These tests use a candidate
set larger than the page size plus a filter that matches nothing, which is the
combination that triggers the walk.

The filter used here is the bbox **verification** control set to "Verified"
against candidates whose boxes are all unverified. That rejects every image —
`imageMatchesFilter` returns `!hasActiveBboxFilter(state)` when no bbox matches
— so each chunk contributes zero entries and the loop keeps paging. (The
reported hang came via the filename control, which gates images the same way;
verification is used here because it drives the identical loop.)

Run with:
    pytest playwright_tests/test_modal_curated_pagination.py -p no:asyncio -v
"""

import json

from playwright.sync_api import Page, expect

PAGE_SIZE = 250
TOTAL_CANDIDATES = 600

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"
VERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000005"
UNVERIFIED_TAG = {"uuid": UNVERIFIED_TAG_UUID, "name": "Unverified", "color": "#888", "kind": "system", "icon": None}
VERIFIED_TAG = {"uuid": VERIFIED_TAG_UUID, "name": "Verified", "color": "#2c2", "kind": "system", "icon": "✓"}
LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}


def _mock_supporting_endpoints(page: Page):
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
            status=200, content_type="application/json", body=json.dumps({"tags": [UNVERIFIED_TAG, VERIFIED_TAG]})
        ),
    )
    page.route(
        "**/api/images/labels/batch*",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "imageLabels": {}})
        ),
    )


def _open_paginating_modal(page: Page, base_url: str, verified_index: int | None = None):
    """Open a modal whose stub `fetchImages` is a *correctly* keyset-paginated
    source over TOTAL_CANDIDATES images, recording every cursor it is handed.

    The stub resumes by locating `afterUuid` directly, so it always terminates
    regardless of whether `afterSortValue` was sent — the test asserts on the
    recorded cursors rather than relying on a hang, which would just time out.
    """
    _mock_supporting_endpoints(page)
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.evaluate(
        """async ({ total, verifiedIndex }) => {
            const all = Array.from({ length: total }, (_, i) => ({
                uuid: `img-${String(i).padStart(4, '0')}`,
                // Descending created_at, matching the server fetchers' default sort.
                createdAt: new Date(Date.UTC(2026, 0, 1) - i * 60000).toISOString().replace('Z', ''),
                filename: `folder/photo_${String(i).padStart(4, '0')}.jpg`,
                // One unverified box each, so "Box status: Verified" rejects every image
                // — except `verifiedIndex`, when the test wants exactly one survivor.
                bboxes: [{
                    uuid: `bbox-${String(i).padStart(4, '0')}`,
                    x: 0.1, y: 0.1, width: 0.2, height: 0.2,
                    label: { uuid: 'lbl-cat', name: 'Cat', color: '#ff0000' },
                    labelUuid: 'lbl-cat',
                    tags: i === verifiedIndex
                        ? [{ uuid: '00000000-0000-0000-0000-000000000005', name: 'Verified',
                             color: '#2c2', kind: 'system', icon: '✓' }]
                        : [{ uuid: '00000000-0000-0000-0000-000000000001', name: 'Unverified',
                             color: '#888', kind: 'system', icon: null }],
                }],
            }));
            window.__calls = [];
            const { showReviewModal } = await import('/app-static/js/app/review/main.js');
            window.__refresh = await showReviewModal({
                title: 'Curated Pagination Test',
                initialViewFilter: 'images',
                fetchImages: async ({ afterUuid = null, afterSortValue = null, limit }) => {
                    window.__calls.push({ afterUuid, afterSortValue });
                    if (window.__calls.length > 40) throw new Error('runaway pagination');
                    let start = 0;
                    if (afterUuid !== null) {
                        const idx = all.findIndex(x => x.uuid === afterUuid);
                        if (idx === -1) return [];
                        start = idx + 1;
                    }
                    return all.slice(start, start + limit);
                },
            });
        }""",
        {"total": TOTAL_CANDIDATES, "verifiedIndex": verified_index},
    )
    page.wait_for_selector(".review-grid", timeout=6000)


def _reject_everything(page: Page):
    """Set Box status = Verified. Every candidate box is unverified, so no image
    survives the predicate and the chunk loop has to walk the whole list."""
    page.locator(".bbox-verification-select").select_option("verified")
    page.wait_for_timeout(1400)


class TestCuratedCursorAdvances:
    def test_every_paging_call_carries_a_sort_value(self, page: Page, base_url: str):
        """The regression itself. A filter matching nothing forces the loop to walk
        the whole candidate list; every call past the first must carry a non-null
        `afterSortValue`, or the real server cannot advance its keyset."""
        _open_paginating_modal(page, base_url)
        _reject_everything(page)

        calls = page.evaluate("() => window.__calls")
        assert len(calls) > 1, f"expected the filter to force a multi-chunk walk, got {len(calls)} call(s)"

        missing = [i for i, c in enumerate(calls) if c["afterUuid"] is not None and c["afterSortValue"] is None]
        assert missing == [], (
            "these paging calls sent a uuid-only cursor, which the server keyset cannot "
            f"resume from: {[calls[i] for i in missing]}"
        )

    def test_walk_terminates_and_does_not_refetch(self, page: Page, base_url: str):
        """The loop must exhaust the candidate list in a bounded number of chunks and
        never re-request a cursor it has already used."""
        _open_paginating_modal(page, base_url)
        _reject_everything(page)

        calls = page.evaluate("() => window.__calls")
        # A null cursor is a render *restart* (initial render, then the filter change),
        # so it legitimately repeats; only the resume cursors must be unique.
        resumed = [c["afterUuid"] for c in calls if c["afterUuid"] is not None]
        assert len(resumed) == len(set(resumed)), f"a cursor was re-requested (loop not advancing): {resumed}"

        # 600 candidates / 250 per chunk = 3 chunks to exhaust, plus the initial render.
        assert len(calls) <= 8, f"walk took {len(calls)} chunks for {TOTAL_CANDIDATES} candidates: {resumed}"

        expect(page.locator(".review-grid .review-card")).to_have_count(0)

    def test_selective_filter_finds_its_matches_across_chunks(self, page: Page, base_url: str):
        """A filter whose only matches live beyond the first chunk still finds them —
        the walk has to actually reach them, not stall on chunk 1."""
        # The only verified box is on the last candidate, well past the 250-image
        # first chunk — the walk has to reach chunk 3 to find it.
        _open_paginating_modal(page, base_url, verified_index=TOTAL_CANDIDATES - 1)
        _reject_everything(page)

        expect(page.locator(".review-grid .review-card")).to_have_count(1)

        calls = page.evaluate("() => window.__calls")
        resumed = [c["afterUuid"] for c in calls if c["afterUuid"] is not None]
        assert len(resumed) == len(set(resumed)), f"a cursor was re-requested: {resumed}"
