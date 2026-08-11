"""
Tests that the "Run Model on Results" scope resolvers send the *same* filter to
the review-queue endpoint that the toolbar's counts were built from.

Both resolvers (`resolveCandidateScope` / `resolveImageScope`) used to hand-roll
their own subset of `filterPanel.getState()` on the way to `fetchReviewQueue`.
Every dimension they forgot silently changed the result set — and for the
*negate* flags it did worse than narrow it, it inverted it: a
"filename does NOT contain office" filter reached the server as
"filename contains office", so the modal reported the complement of what the
toolbar counted.

These drive the real modules against a stubbed `window.fetch` and assert on the
request body actually put on the wire.
"""

import json

import pytest
from playwright.sync_api import Page

_MODULE = "/app-static/js/app/review/runModelOnResults.js"

# A filter state exercising every dimension the resolvers were dropping.
_STATE = {
    "showAllBboxes": False,
    "tagUuids": [],
    "tagExcludeUuids": [],
    "tagMatchMode": "any",
    "labelUuids": [],
    "labelExcludeUuids": [],
    "labelMatchMode": "any",
    "labelIncludeChildren": False,
    "imageLabelConditions": [],
    "imageTagConditions": [
        {"tagUuid": "11111111-1111-1111-1111-111111111111", "state": "present"},
    ],
    "filenameSearch": "office",
    "filenameNegate": True,
    "dateFrom": "2026-01-01T00:00:00.000Z",
    "dateTo": None,
    "dateNegate": True,
    "verificationCheck": None,
    "bboxVerification": "verified",
}


def _capture_body(page: Page, fn_name: str, state: dict) -> dict:
    """Call a scope resolver with `state` and return the review-queue request body."""
    return page.evaluate(
        """async ({ fnName, state }) => {
            const m = await import('%s');
            const realFetch = window.fetch;
            let captured = null;
            window.fetch = async (url, opts) => {
                const href = typeof url === 'string' ? url : url.url;
                if (href.includes('/api/images/review')) {
                    captured = JSON.parse(opts.body);
                    return new Response(
                        JSON.stringify({ status: 'success', images: [], hasMore: false, totalCount: 0 }),
                        { status: 200, headers: { 'Content-Type': 'application/json' } },
                    );
                }
                return realFetch(url, opts);
            };
            try {
                await m[fnName]({
                    fetchImages: null,
                    filterPanel: { getState: () => state },
                    viewFilter: 'all',
                    selectionKind: null,
                    selected: new Set(),
                    visibleEntries: [],
                    sortBy: 'created_at',
                    sortOrder: 'desc',
                    bboxFilterFn: null,
                    labels: new Map(),
                });
            } finally {
                window.fetch = realFetch;
            }
            return captured;
        }"""
        % _MODULE,
        {"fnName": fn_name, "state": state},
    )


def _term(body: dict, kind: str) -> dict | None:
    matches = [t for t in body.get("terms", []) if t.get("kind") == kind]
    return matches[0] if matches else None


@pytest.mark.parametrize("fn_name", ["resolveCandidateScope", "resolveImageScope"])
class TestScopeResolverFilterParity:
    def test_filename_negate_is_sent(self, shared_page: Page, fn_name: str):
        """The regression: a negated filename filter must not reach the server
        un-negated, or the modal resolves the exact complement of the filter."""
        body = _capture_body(shared_page, fn_name, _STATE)
        term = _term(body, "filename")
        assert term is not None, f"{fn_name} dropped the filename filter entirely: {json.dumps(body)}"
        assert term.get("negate") is True, f"{fn_name} sent the filename filter un-negated: {term}"

    def test_date_negate_is_sent(self, shared_page: Page, fn_name: str):
        body = _capture_body(shared_page, fn_name, _STATE)
        term = _term(body, "date")
        assert term is not None, f"{fn_name} dropped the date filter"
        assert term.get("negate") is True, f"{fn_name} sent the date filter un-negated: {term}"

    def test_image_tag_conditions_are_sent(self, shared_page: Page, fn_name: str):
        body = _capture_body(shared_page, fn_name, _STATE)
        assert _term(body, "imageTag") is not None, f"{fn_name} dropped imageTagConditions"
