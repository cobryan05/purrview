"""
Tests for shared/filterUtils.js round-tripping.

reviewStateToDatasetForm / datasetFormToReviewState are the save/load path for
Views (which replaced the old localStorage filter presets). The presets stored
filterPanel.getState() verbatim, so every filter dimension round-tripped for
free; the converters have to carry each dimension explicitly, and a dimension
that is silently dropped means a saved View quietly filters differently from
the filter that was saved.

These are pure functions, but the module imports from shared/api/review.js, so
we navigate to base_url first (same rationale as test_mapper.py).
"""

import pytest
from playwright.sync_api import Page

_MODULE = "/app-static/js/shared/filterUtils.js"


def _to_form(page: Page, review_state: dict) -> dict:
    return page.evaluate(
        """async (state) => {
            const m = await import('%s');
            return m.reviewStateToDatasetForm(state);
        }""" % _MODULE,
        review_state,
    )


def _to_state(page: Page, filter_json: dict, label_uuids=None) -> dict:
    return page.evaluate(
        """async ({ filterJson, labelUuids }) => {
            const m = await import('%s');
            return m.datasetFormToReviewState(filterJson, labelUuids);
        }""" % _MODULE,
        {"filterJson": filter_json, "labelUuids": label_uuids or []},
    )


class TestBBoxVerificationRoundTrip:
    """`bboxVerification` is a live filter dimension — review/main.js feeds it
    straight into the review query — so it must survive save/load."""

    @pytest.mark.parametrize("state", ["verified", "unverified"])
    def test_serialized_as_term(self, shared_page: Page, state: str):
        form = _to_form(shared_page, {"bboxVerification": state})
        terms = form["filterJson"]["terms"]
        match = [t for t in terms if t.get("kind") == "bboxVerification"]
        assert match, f"bboxVerification={state} must be serialized into filterJson.terms"
        assert match[0]["state"] == state

    def test_any_is_not_serialized(self, shared_page: Page):
        """'any' is the no-op default — it must not add a term."""
        form = _to_form(shared_page, {"bboxVerification": "any"})
        terms = form["filterJson"]["terms"]
        assert not [t for t in terms if t.get("kind") == "bboxVerification"]

    def test_missing_is_not_serialized(self, shared_page: Page):
        form = _to_form(shared_page, {})
        terms = form["filterJson"]["terms"]
        assert not [t for t in terms if t.get("kind") == "bboxVerification"]

    @pytest.mark.parametrize("state", ["verified", "unverified"])
    def test_restored_from_term(self, shared_page: Page, state: str):
        restored = _to_state(shared_page, {"terms": [{"kind": "bboxVerification", "state": state}]})
        assert restored["bboxVerification"] == state

    def test_restored_defaults_to_any(self, shared_page: Page):
        """No term must reset the panel to 'any', not leave the previous value
        in place — setState is partial, so an omitted key is a stale filter."""
        restored = _to_state(shared_page, {"terms": []})
        assert restored["bboxVerification"] == "any"

    @pytest.mark.parametrize("state", ["verified", "unverified"])
    def test_full_round_trip_preserves_other_dimensions(self, shared_page: Page, state: str):
        original = {
            "bboxVerification": state,
            "tagUuids": ["tag-a", "tag-b"],
            "tagMatchMode": "all",
            "filenameSearch": "cat",
            "labelUuids": ["lbl-1"],
            "labelMatchMode": "all",
        }
        form = _to_form(shared_page, original)
        restored = _to_state(shared_page, form["filterJson"], form["labelUuids"])

        assert restored["bboxVerification"] == state
        assert sorted(restored["tagUuids"]) == ["tag-a", "tag-b"]
        assert restored["tagMatchMode"] == "all"
        assert restored["filenameSearch"] == "cat"
        assert restored["labelUuids"] == ["lbl-1"]
        assert restored["labelMatchMode"] == "all"
