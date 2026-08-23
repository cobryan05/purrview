"""
The Source filter must reach the wire, for every review-query wrapper.

shared/api/review.js has one body builder (buildReviewFilterBody) and four
public wrappers that hand it their options *by hand*, relisting every field
instead of spreading. A dimension missing from that relist is dropped in
silence: the panel holds the selection, getState/reviewStateToFilterOpts carry
it, the server understands the term — and the request simply never mentions it.
That is exactly how the Source filter shipped broken (selecting a camera
changed nothing on the Review page).

So these assert the outgoing request body, which is the only place the drop is
visible. `fetch` is stubbed rather than hitting the server: the point is what
was *sent*, independent of what any particular DB holds.
"""

import json

import pytest
from playwright.sync_api import Page

_MODULE = "/app-static/js/shared/api/review.js"

_SRC_A = "11111111-1111-1111-1111-111111111111"
_SRC_B = "22222222-2222-2222-2222-222222222222"

# fn name -> the call that exercises it, plus where the filter body sits in the
# request payload. The three bulk/count wrappers nest it under `filter`;
# fetchReviewQueue sends it at the top level.
_WRAPPERS = [
    ("fetchReviewQueue", None),
    ("bulkApplyAll", "filter"),
    ("countMatchingBboxes", "filter"),
    ("bulkApplyImages", "filter"),
]


def _capture_body(page: Page, fn: str, opts: dict) -> dict:
    """Call review.js's `fn` with `opts` and return the JSON body it POSTed."""
    return page.evaluate(
        """async ({ fn, opts }) => {
            const m = await import('%s');
            let captured = null;
            const realFetch = globalThis.fetch;
            globalThis.fetch = async (url, init) => {
                captured = JSON.parse(init.body);
                return new Response(
                    JSON.stringify({
                        status: 'success',
                        images: [], hasMore: false, totalCount: 0,
                        count: 0, modifiedCount: 0, failCount: 0, skippedCount: 0,
                    }),
                    { status: 200, headers: { 'Content-Type': 'application/json' } },
                );
            };
            try {
                await m[fn]({ ...opts, action: 'tag' });
            } finally {
                globalThis.fetch = realFetch;
            }
            return captured;
        }""" % _MODULE,
        {"fn": fn, "opts": opts},
    )


def _source_terms(body: dict, nest_key) -> list:
    filter_body = body[nest_key] if nest_key else body
    return [t for t in (filter_body.get("terms") or []) if t.get("kind") == "imageSource"]


class TestSourceFilterReachesTheWire:
    @pytest.mark.parametrize("fn,nest_key", _WRAPPERS)
    def test_selected_sources_are_sent(self, shared_page: Page, fn: str, nest_key):
        body = _capture_body(shared_page, fn, {"sourceUuids": [_SRC_A, _SRC_B]})
        terms = _source_terms(body, nest_key)
        assert terms, f"{fn} dropped the Source selection — request body: {json.dumps(body)}"
        assert sorted(terms[0]["sourceUuids"]) == sorted([_SRC_A, _SRC_B])

    @pytest.mark.parametrize("fn,nest_key", _WRAPPERS)
    def test_excluded_sources_are_sent(self, shared_page: Page, fn: str, nest_key):
        """Exclusions are their own negated terms — dropping them is the same
        silent failure, and the complement of what the user asked for."""
        body = _capture_body(shared_page, fn, {"sourceExcludeUuids": [_SRC_A]})
        negated = [t for t in _source_terms(body, nest_key) if t.get("negate")]
        assert negated, f"{fn} dropped the Source exclusion — request body: {json.dumps(body)}"
        assert negated[0]["sourceUuids"] == [_SRC_A]

    @pytest.mark.parametrize("fn,nest_key", _WRAPPERS)
    def test_no_selection_sends_no_source_term(self, shared_page: Page, fn: str, nest_key):
        """The conditional half: an empty picker must not narrow anything. A
        term built unconditionally would filter every query by 'no sources'."""
        body = _capture_body(shared_page, fn, {})
        assert not _source_terms(body, nest_key)
