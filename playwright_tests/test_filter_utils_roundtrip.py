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
        assert restored["filenameConditions"] == [{"pattern": "cat", "negate": False}]
        assert restored["labelUuids"] == ["lbl-1"]
        assert restored["labelMatchMode"] == "all"


class TestImageLabelConditionRoundTrip:
    """`rollUp` (+sub) and `negate` (¬) are per-condition controls on the image
    label filter rows (ui/imageTagConditions.js) and are rendered into the chip
    bar (review/filters.js), so they are user-visible filter state. The server
    defaults them to roll_up=true / negate=false, which means omitting them on
    save does not "keep the default" — it saves the *complement* of a negated
    condition and re-widens an exact-label one."""

    def test_negate_is_serialized(self, shared_page: Page):
        form = _to_form(
            shared_page,
            {"imageLabelConditions": [{"labelUuid": "lbl-1", "state": "present", "negate": True}]},
        )
        term = [t for t in form["filterJson"]["terms"] if t.get("kind") == "imageLabel"][0]
        assert term["negate"] is True, "a negated condition saved without negate matches the opposite images"

    def test_roll_up_false_is_serialized(self, shared_page: Page):
        form = _to_form(
            shared_page,
            {"imageLabelConditions": [{"labelUuid": "lbl-1", "state": "present", "rollUp": False}]},
        )
        term = [t for t in form["filterJson"]["terms"] if t.get("kind") == "imageLabel"][0]
        assert term["rollUp"] is False

    def test_defaults_match_the_server_defaults(self, shared_page: Page):
        """A condition predating the UI controls must serialize to the server's
        own defaults, so older saved Views keep behaving exactly as before."""
        form = _to_form(shared_page, {"imageLabelConditions": [{"labelUuid": "lbl-1", "state": "present"}]})
        term = [t for t in form["filterJson"]["terms"] if t.get("kind") == "imageLabel"][0]
        assert term["rollUp"] is True
        assert term["negate"] is False

    def test_negate_is_restored(self, shared_page: Page):
        restored = _to_state(
            shared_page,
            {"terms": [{"kind": "imageLabel", "labelUuid": "lbl-1", "state": "present", "negate": True}]},
        )
        assert restored["imageLabelConditions"][0]["negate"] is True

    def test_roll_up_false_is_restored(self, shared_page: Page):
        restored = _to_state(
            shared_page,
            {"terms": [{"kind": "imageLabel", "labelUuid": "lbl-1", "state": "present", "rollUp": False}]},
        )
        assert restored["imageLabelConditions"][0]["rollUp"] is False

    def test_restored_defaults_match_the_server_defaults(self, shared_page: Page):
        """imageTagConditions.setState() re-defaults any key it isn't given, so
        a term stored before these fields existed must read back as the server
        defaults rather than as undefined."""
        restored = _to_state(
            shared_page,
            {"terms": [{"kind": "imageLabel", "labelUuid": "lbl-1", "state": "present"}]},
        )
        cond = restored["imageLabelConditions"][0]
        assert cond["rollUp"] is True
        assert cond["negate"] is False

    @pytest.mark.parametrize("negate", [True, False])
    @pytest.mark.parametrize("roll_up", [True, False])
    @pytest.mark.parametrize("state", ["present", "absent", "uncertain"])
    def test_full_round_trip(self, shared_page: Page, state: str, roll_up: bool, negate: bool):
        """Save → load must be the identity on every combination. Fixing only one
        side of the conversion leaves the round trip passing while a real save
        still filters wrongly, so this asserts both halves at once."""
        original = {
            "imageLabelConditions": [
                {"labelUuid": "lbl-1", "state": state, "rollUp": roll_up, "negate": negate},
            ],
        }
        form = _to_form(shared_page, original)
        restored = _to_state(shared_page, form["filterJson"], form["labelUuids"])

        assert restored["imageLabelConditions"] == [
            {"labelUuid": "lbl-1", "state": state, "rollUp": roll_up, "negate": negate},
        ]


class TestFilenameNegateRoundTrip:
    def test_filename_negate_true_is_serialized(self, shared_page: Page):
        form = _to_form(shared_page, {"filenameSearch": "foo", "filenameNegate": True})
        term = [t for t in form["filterJson"]["terms"] if t.get("kind") == "filename"][0]
        assert term["negate"] is True

    def test_filename_negate_defaults_false(self, shared_page: Page):
        """Omitted rather than written as false — the same omit-when-default
        rule the other terms follow; the server defaults negate to false."""
        form = _to_form(shared_page, {"filenameSearch": "foo"})
        term = [t for t in form["filterJson"]["terms"] if t.get("kind") == "filename"][0]
        assert term.get("negate", False) is False

    def test_no_filename_search_means_no_term_regardless_of_negate(self, shared_page: Page):
        """negate is a modifier on the filename search, not an independent
        filter — an empty search must not produce a term just because negate
        is set, or a stray checkbox click would silently start filtering."""
        form = _to_form(shared_page, {"filenameNegate": True})
        assert not [t for t in form["filterJson"]["terms"] if t.get("kind") == "filename"]

    def test_filename_negate_is_restored(self, shared_page: Page):
        restored = _to_state(
            shared_page,
            {"terms": [{"kind": "filename", "patterns": ["foo"], "negate": True}]},
        )
        assert restored["filenameConditions"] == [{"pattern": "foo", "negate": True}]

    def test_filename_negate_restored_defaults_false(self, shared_page: Page):
        restored = _to_state(shared_page, {"terms": [{"kind": "filename", "patterns": ["foo"]}]})
        assert restored["filenameConditions"] == [{"pattern": "foo", "negate": False}]

    @pytest.mark.parametrize("negate", [True, False])
    def test_full_round_trip(self, shared_page: Page, negate: bool):
        form = _to_form(shared_page, {"filenameSearch": "foo", "filenameNegate": negate})
        restored = _to_state(shared_page, form["filterJson"], form["labelUuids"])
        assert restored["filenameConditions"] == [{"pattern": "foo", "negate": negate}]


class TestMultipleFilenameRoundTrip:
    """A saved View must keep every filename filter, not just the first — the
    include patterns share one term, each exclusion gets its own, and reading
    back only terms[0] would silently drop the exclusions on load."""

    def test_all_filenames_survive_a_round_trip(self, shared_page: Page):
        original = {
            "filenameConditions": [
                {"pattern": "cam1", "negate": False},
                {"pattern": "cam2", "negate": False},
                {"pattern": "thumb", "negate": True},
            ],
            "filenameMatchMode": "any",
        }
        form = _to_form(shared_page, original)
        restored = _to_state(shared_page, form["filterJson"], form["labelUuids"])
        assert restored["filenameConditions"] == original["filenameConditions"]
        assert restored["filenameMatchMode"] == "any"

    def test_match_mode_all_survives_a_round_trip(self, shared_page: Page):
        form = _to_form(
            shared_page,
            {
                "filenameConditions": [{"pattern": "cam1"}, {"pattern": "2026-08"}],
                "filenameMatchMode": "all",
            },
        )
        restored = _to_state(shared_page, form["filterJson"], form["labelUuids"])
        assert restored["filenameMatchMode"] == "all"


class TestDateNegateRoundTrip:
    def test_date_negate_true_is_serialized(self, shared_page: Page):
        form = _to_form(shared_page, {"dateFrom": "2026-01-01T00:00:00.000Z", "dateNegate": True})
        term = [t for t in form["filterJson"]["terms"] if t.get("kind") == "date"][0]
        assert term["negate"] is True

    def test_date_negate_omitted_when_false(self, shared_page: Page):
        """Matches the omit-when-default pattern used elsewhere in this module —
        dateNegate: False must not add a `negate` key at all."""
        form = _to_form(shared_page, {"dateFrom": "2026-01-01T00:00:00.000Z", "dateNegate": False})
        term = [t for t in form["filterJson"]["terms"] if t.get("kind") == "date"][0]
        assert "negate" not in term

    def test_no_date_range_means_no_term_regardless_of_negate(self, shared_page: Page):
        form = _to_form(shared_page, {"dateNegate": True})
        assert not [t for t in form["filterJson"]["terms"] if t.get("kind") == "date"]

    def test_date_negate_is_restored(self, shared_page: Page):
        restored = _to_state(
            shared_page,
            {"terms": [{"kind": "date", "field": "created", "from": "2026-01-01T00:00:00.000Z", "negate": True}]},
        )
        assert restored["dateNegate"] is True
        assert restored["dateFrom"] == "2026-01-01T00:00:00.000Z"

    def test_date_negate_restored_defaults_false(self, shared_page: Page):
        restored = _to_state(
            shared_page,
            {"terms": [{"kind": "date", "field": "created", "from": "2026-01-01T00:00:00.000Z"}]},
        )
        assert restored["dateNegate"] is False

    @pytest.mark.parametrize("negate", [True, False])
    def test_full_round_trip(self, shared_page: Page, negate: bool):
        original = {"dateFrom": "2026-01-01T00:00:00.000Z", "dateTo": "2026-02-01T00:00:00.000Z", "dateNegate": negate}
        form = _to_form(shared_page, original)
        restored = _to_state(shared_page, form["filterJson"], form["labelUuids"])
        assert restored["dateFrom"] == original["dateFrom"]
        assert restored["dateTo"] == original["dateTo"]
        assert restored["dateNegate"] == negate


class TestConfidenceBandRoundTrip:
    """The confidence band is a live filter dimension (review/main.js feeds it
    into the review query and into Apply-to-All's bbox-level narrowing), so a
    saved View has to carry it or it silently filters differently on load."""

    def test_serialized_as_term(self, shared_page: Page):
        form = _to_form(shared_page, {"confidenceMin": 0.25, "confidenceMax": 0.75})
        match = [t for t in form["filterJson"]["terms"] if t.get("kind") == "bboxConfidence"]
        assert match, "a confidence band must be serialized into filterJson.terms"
        assert match[0]["min"] == 0.25
        assert match[0]["max"] == 0.75

    def test_one_sided_band_omits_the_other_bound(self, shared_page: Page):
        form = _to_form(shared_page, {"confidenceMax": 0.5})
        term = [t for t in form["filterJson"]["terms"] if t.get("kind") == "bboxConfidence"][0]
        assert term["max"] == 0.5
        assert "min" not in term

    def test_no_bounds_is_not_serialized(self, shared_page: Page):
        # Including includeMissing on its own — with no bounds it matches
        # everything, so it is not a filter.
        for state in ({}, {"confidenceMin": None, "confidenceMax": None}, {"confidenceIncludeMissing": True}):
            form = _to_form(shared_page, state)
            assert not [t for t in form["filterJson"]["terms"] if t.get("kind") == "bboxConfidence"], state

    def test_restored_from_term(self, shared_page: Page):
        restored = _to_state(
            shared_page,
            {"terms": [{"kind": "bboxConfidence", "min": 0.4, "max": 0.6, "includeMissing": True}]},
        )
        assert restored["confidenceMin"] == 0.4
        assert restored["confidenceMax"] == 0.6
        assert restored["confidenceIncludeMissing"] is True

    def test_restored_defaults_to_no_band(self, shared_page: Page):
        """No term must reset the panel to an empty band, not leave whatever the
        user had typed before loading the View (setState is partial)."""
        restored = _to_state(shared_page, {"terms": []})
        assert restored["confidenceMin"] is None
        assert restored["confidenceMax"] is None
        assert restored["confidenceIncludeMissing"] is False

    @pytest.mark.parametrize("include_missing", [True, False])
    def test_full_round_trip(self, shared_page: Page, include_missing: bool):
        original = {
            "confidenceMin": 0.1,
            "confidenceMax": 0.9,
            "confidenceIncludeMissing": include_missing,
            "bboxVerification": "unverified",
            "filenameSearch": "cam1",
        }
        form = _to_form(shared_page, original)
        restored = _to_state(shared_page, form["filterJson"], form["labelUuids"])
        assert restored["confidenceMin"] == 0.1
        assert restored["confidenceMax"] == 0.9
        assert restored["confidenceIncludeMissing"] is include_missing
        assert restored["bboxVerification"] == "unverified"
        assert restored["filenameConditions"] == [{"pattern": "cam1", "negate": False}]
