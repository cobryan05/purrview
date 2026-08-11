"""
Pure-logic unit tests for the shared curated filter predicate (F1).

Covers `bboxMatchesFilter` / `imageMatchesFilter` /
`defaultCuratedFilterState` in
apps/image_tagging_server/static/js/app/review/filterPredicate.js.

Same style as test_pure_logic.py: each test imports the module inside the
browser via page.evaluate so /app-static/ imports resolve. The predicate is
pure (no DOM, no network), so these are true unit tests.

Verification tag UUIDs (must match shared/api/review.js):
  VERIFIED    = 00000000-0000-0000-0000-000000000005
  UNVERIFIED  = 00000000-0000-0000-0000-000000000001
"""

import pytest
from playwright.sync_api import Page

VERIFIED = "00000000-0000-0000-0000-000000000005"
UNVERIFIED = "00000000-0000-0000-0000-000000000001"

# A small JS preamble shared by every test: imports the module and defines
# fixtures (a label hierarchy + helpers to build bboxes/images) so the actual
# test body is a single expression using `defaultCuratedFilterState()`.
_PREAMBLE = """
const M = await import('/app-static/js/app/review/filterPredicate.js');
const { defaultCuratedFilterState, bboxMatchesFilter, imageMatchesFilter } = M;
// Verification tag UUIDs, mirrored from shared/api/review.js so test bodies
// can reference them as bare identifiers.
const VERIFIED   = '00000000-0000-0000-0000-000000000005';
const UNVERIFIED = '00000000-0000-0000-0000-000000000001';
// Label hierarchy:  animal -> cat -> kitten ;  animal -> dog ;  vehicle
const labels = new Map([
  ['animal',  { uuid: 'animal',  name: 'Animal',  parentUuid: null }],
  ['cat',     { uuid: 'cat',     name: 'Cat',     parentUuid: 'animal' }],
  ['kitten',  { uuid: 'kitten',  name: 'Kitten',  parentUuid: 'cat' }],
  ['dog',     { uuid: 'dog',     name: 'Dog',     parentUuid: 'animal' }],
  ['vehicle', { uuid: 'vehicle', name: 'Vehicle', parentUuid: null }],
]);
const bbox = (labelUuid, ...tagUuids) => ({
  uuid: 'b-' + Math.random().toString(36).slice(2),
  labelUuid,
  tags: tagUuids.map(u => ({ uuid: u })),
});
const img = (...bboxes) => ({ uuid: 'i1', bboxes });
const st = (over) => ({ ...defaultCuratedFilterState(), ...(over || {}) });
// A bbox carrying a raw metadata blob, for the confidence-band tests. `meta`
// is passed through verbatim so a test can use either confidence key and
// either JSON type (both occur in the real data — see CONFIDENCE_KEYS).
const cbox = (meta) => ({ ...bbox('cat'), metadataJson: meta });
"""


def _eval(page: Page, body: str):
    """Run `body` (a JS expression, returned) after the shared preamble."""
    return page.evaluate("async () => {\n" + _PREAMBLE + "\n return (" + body + ");\n}")


# ---------------------------------------------------------------------------
# defaultCuratedFilterState — shape
# ---------------------------------------------------------------------------


class TestDefaultState:
    def test_shape(self, shared_page: Page):
        result = _eval(shared_page, "defaultCuratedFilterState()")
        assert result == {
            "bboxVerification": "any",
            "labelUuids": [],
            "labelExcludeUuids": [],
            "labelMatchMode": "any",
            "labelIncludeChildren": False,
            "tagUuids": [],
            "tagExcludeUuids": [],
            "tagMatchMode": "any",
            "filenameSearch": "",
            "dateFrom": None,
            "dateTo": None,
            "confidenceMin": None,
            "confidenceMax": None,
            "confidenceIncludeMissing": False,
        }

    def test_empty_state_bbox_matches_everything(self, shared_page: Page):
        # Any bbox (even one with no label and no tags) matches the default state.
        assert _eval(shared_page, "bboxMatchesFilter(bbox(null), st(), labels)") is True
        assert _eval(shared_page, "bboxMatchesFilter(bbox('cat', UNVERIFIED), st(), labels)") is True

    def test_empty_state_image_matches_when_any_bbox(self, shared_page: Page):
        assert _eval(shared_page, "imageMatchesFilter(img(bbox('cat')), st(), labels)") is True

    def test_empty_state_image_with_no_bboxes_matches(self, shared_page: Page):
        # Under the no-op default there is nothing to filter on, so a bbox-free
        # image must pass — image-oriented modals (similar/dissimilar/diversity)
        # return whole images that frequently have zero bboxes, and gating them
        # out would silently hide unfiltered results.
        assert _eval(shared_page, "imageMatchesFilter(img(), st(), labels)") is True

    def test_active_filter_image_with_no_bboxes_is_false(self, shared_page: Page):
        # With any active bbox-scoped constraint, a bbox-free image has nothing
        # that can satisfy it and is correctly rejected.
        assert _eval(shared_page, "imageMatchesFilter(img(), st({bboxVerification:'verified'}), labels)") is False
        assert _eval(shared_page, "imageMatchesFilter(img(), st({labelUuids:['cat']}), labels)") is False
        assert _eval(shared_page, "imageMatchesFilter(img(), st({tagUuids:[VERIFIED]}), labels)") is False

    def test_active_filter_image_with_only_nonmatching_bbox_is_false(self, shared_page: Page):
        # An image that has bboxes but none satisfying the active filter is
        # still rejected (unchanged behavior).
        assert _eval(shared_page, "imageMatchesFilter(img(bbox('dog')), st({labelUuids:['cat']}), labels)") is False


# ---------------------------------------------------------------------------
# Verification tri-state
# ---------------------------------------------------------------------------


class TestVerification:
    def test_uses_real_review_api_tag_uuids(self, shared_page: Page):
        # Self-validating guard: drive the predicate with the tag UUIDs actually
        # exported by shared/api/review.js (not the local mirror), so if those
        # constants ever change the predicate wiring is still exercised.
        result = shared_page.evaluate("""async () => {
            const P = await import('/app-static/js/app/review/filterPredicate.js');
            const R = await import('/app-static/js/shared/api/review.js');
            const st = { ...P.defaultCuratedFilterState(), bboxVerification: 'verified' };
            const good = { uuid: 'b1', labelUuid: 'x', tags: [{ uuid: R.VERIFIED_TAG_UUID }] };
            const bad  = { uuid: 'b2', labelUuid: 'x', tags: [{ uuid: R.UNVERIFIED_TAG_UUID }] };
            return {
                good: P.bboxMatchesFilter(good, st, new Map()),
                bad: P.bboxMatchesFilter(bad, st, new Map()),
            };
        }""")
        assert result == {"good": True, "bad": False}

    def test_any_ignores_tags(self, shared_page: Page):
        assert _eval(shared_page, "bboxMatchesFilter(bbox('cat'), st({bboxVerification:'any'}), labels)") is True

    def test_verified_requires_verified_tag(self, shared_page: Page):
        assert (
            _eval(shared_page, f"bboxMatchesFilter(bbox('cat', VERIFIED), st({{bboxVerification:'verified'}}), labels)")
            is True
        )
        assert (
            _eval(
                shared_page, f"bboxMatchesFilter(bbox('cat', UNVERIFIED), st({{bboxVerification:'verified'}}), labels)"
            )
            is False
        )
        assert _eval(shared_page, "bboxMatchesFilter(bbox('cat'), st({bboxVerification:'verified'}), labels)") is False

    def test_unverified_requires_unverified_tag(self, shared_page: Page):
        assert (
            _eval(
                shared_page, "bboxMatchesFilter(bbox('cat', UNVERIFIED), st({bboxVerification:'unverified'}), labels)"
            )
            is True
        )
        assert (
            _eval(shared_page, "bboxMatchesFilter(bbox('cat', VERIFIED), st({bboxVerification:'unverified'}), labels)")
            is False
        )

    def test_states_are_exclusive(self, shared_page: Page):
        # A bbox with VERIFIED does not satisfy 'unverified', and vice versa.
        assert (
            _eval(shared_page, "bboxMatchesFilter(bbox('cat', VERIFIED), st({bboxVerification:'unverified'}), labels)")
            is False
        )
        assert (
            _eval(shared_page, "bboxMatchesFilter(bbox('cat', UNVERIFIED), st({bboxVerification:'verified'}), labels)")
            is False
        )


# ---------------------------------------------------------------------------
# Label include / exclude / child expansion
# ---------------------------------------------------------------------------


class TestLabelInclude:
    def test_include_exact_match(self, shared_page: Page):
        assert _eval(shared_page, "bboxMatchesFilter(bbox('cat'), st({labelUuids:['cat']}), labels)") is True
        assert _eval(shared_page, "bboxMatchesFilter(bbox('dog'), st({labelUuids:['cat']}), labels)") is False

    def test_include_no_children_excludes_descendants(self, shared_page: Page):
        # kitten is a descendant of cat; without includeChildren it doesn't match.
        assert _eval(shared_page, "bboxMatchesFilter(bbox('kitten'), st({labelUuids:['cat']}), labels)") is False

    def test_include_children_matches_descendants(self, shared_page: Page):
        assert (
            _eval(
                shared_page,
                "bboxMatchesFilter(bbox('kitten'), st({labelUuids:['cat'], labelIncludeChildren:true}), labels)",
            )
            is True
        )
        # grandchild + child both covered by the ancestor 'animal'
        assert (
            _eval(
                shared_page,
                "bboxMatchesFilter(bbox('kitten'), st({labelUuids:['animal'], labelIncludeChildren:true}), labels)",
            )
            is True
        )
        assert (
            _eval(
                shared_page,
                "bboxMatchesFilter(bbox('dog'),    st({labelUuids:['animal'], labelIncludeChildren:true}), labels)",
            )
            is True
        )
        # vehicle is not under animal
        assert (
            _eval(
                shared_page,
                "bboxMatchesFilter(bbox('vehicle'), st({labelUuids:['animal'], labelIncludeChildren:true}), labels)",
            )
            is False
        )


class TestLabelExclude:
    def test_exclude_removes_matching_label(self, shared_page: Page):
        assert _eval(shared_page, "bboxMatchesFilter(bbox('cat'), st({labelExcludeUuids:['cat']}), labels)") is False
        assert _eval(shared_page, "bboxMatchesFilter(bbox('dog'), st({labelExcludeUuids:['cat']}), labels)") is True

    def test_exclude_is_not_child_expanded(self, shared_page: Page):
        # SERVER SEMANTICS: label exclusions are exact-uuid, never child-expanded,
        # even when labelIncludeChildren is True. kitten (child of cat) is NOT
        # excluded by excluding 'cat'.
        assert (
            _eval(
                shared_page,
                "bboxMatchesFilter(bbox('kitten'), st({labelExcludeUuids:['cat'], labelIncludeChildren:true}), labels)",
            )
            is True
        )


# ---------------------------------------------------------------------------
# Tag include (any / all) + exclude
# ---------------------------------------------------------------------------


class TestTagInclude:
    def test_any_requires_at_least_one(self, shared_page: Page):
        assert _eval(shared_page, "bboxMatchesFilter(bbox('cat','t1'), st({tagUuids:['t1','t2']}), labels)") is True
        assert _eval(shared_page, "bboxMatchesFilter(bbox('cat','t3'), st({tagUuids:['t1','t2']}), labels)") is False

    def test_all_requires_every_tag(self, shared_page: Page):
        assert (
            _eval(
                shared_page,
                "bboxMatchesFilter(bbox('cat','t1','t2'), st({tagUuids:['t1','t2'], tagMatchMode:'all'}), labels)",
            )
            is True
        )
        assert (
            _eval(
                shared_page,
                "bboxMatchesFilter(bbox('cat','t1'),      st({tagUuids:['t1','t2'], tagMatchMode:'all'}), labels)",
            )
            is False
        )


class TestTagExclude:
    def test_exclude_rejects_any_present(self, shared_page: Page):
        assert _eval(shared_page, "bboxMatchesFilter(bbox('cat','t1'), st({tagExcludeUuids:['t1']}), labels)") is False
        assert _eval(shared_page, "bboxMatchesFilter(bbox('cat','t2'), st({tagExcludeUuids:['t1']}), labels)") is True
        # none-of: any excluded tag present -> reject
        assert (
            _eval(shared_page, "bboxMatchesFilter(bbox('cat','t2'), st({tagExcludeUuids:['t1','t2']}), labels)")
            is False
        )


# ---------------------------------------------------------------------------
# imageMatchesFilter — labelMatchMode 'all' coverage gate
# ---------------------------------------------------------------------------


class TestImageAll:
    def test_any_mode_one_matching_bbox_passes(self, shared_page: Page):
        # 'any' (default): image passes if any bbox matches, even if only one of
        # the selected labels is present.
        assert (
            _eval(
                shared_page,
                "imageMatchesFilter(img(bbox('cat'), bbox('vehicle')), st({labelUuids:['cat','dog']}), labels)",
            )
            is True
        )

    def test_all_mode_requires_every_selected_label_covered(self, shared_page: Page):
        # 'all': image must have a matching bbox for BOTH cat and dog.
        assert (
            _eval(
                shared_page,
                "imageMatchesFilter(img(bbox('cat'), bbox('dog')), st({labelUuids:['cat','dog'], labelMatchMode:'all'}), labels)",
            )
            is True
        )

    def test_all_mode_fails_when_a_label_uncovered(self, shared_page: Page):
        # only cat present -> the 'dog' requirement fails.
        assert (
            _eval(
                shared_page,
                "imageMatchesFilter(img(bbox('cat'), bbox('vehicle')), st({labelUuids:['cat','dog'], labelMatchMode:'all'}), labels)",
            )
            is False
        )

    def test_all_mode_with_children_covers_via_descendant(self, shared_page: Page):
        # kitten (descendant of cat) covers the 'cat' requirement when
        # includeChildren is on; dog covers 'dog'.
        assert (
            _eval(
                shared_page,
                "imageMatchesFilter(img(bbox('kitten'), bbox('dog')), st({labelUuids:['cat','dog'], labelMatchMode:'all', labelIncludeChildren:true}), labels)",
            )
            is True
        )

    def test_all_mode_no_matching_bbox_is_false(self, shared_page: Page):
        assert (
            _eval(
                shared_page,
                "imageMatchesFilter(img(bbox('vehicle')), st({labelUuids:['cat','dog'], labelMatchMode:'all'}), labels)",
            )
            is False
        )


# ---------------------------------------------------------------------------
# Combined AND-ing of conditions
# ---------------------------------------------------------------------------


class TestCombined:
    def test_all_conditions_and_together(self, shared_page: Page):
        # verified + label cat + tag t1, all satisfied
        assert (
            _eval(
                shared_page,
                f"bboxMatchesFilter(bbox('cat', VERIFIED, 't1'), st({{bboxVerification:'verified', labelUuids:['cat'], tagUuids:['t1']}}), labels)",
            )
            is True
        )
        # same but label is dog -> fails the label condition
        assert (
            _eval(
                shared_page,
                f"bboxMatchesFilter(bbox('dog', VERIFIED, 't1'), st({{bboxVerification:'verified', labelUuids:['cat'], tagUuids:['t1']}}), labels)",
            )
            is False
        )


# ---------------------------------------------------------------------------
# Filename substring (image-scoped, not bbox-scoped)
# ---------------------------------------------------------------------------


class TestFilenameFilter:
    """`filenameSearch` is the client-side counterpart of the server's
    FilenameTerm LIKE. It is applied by `imageMatchesFilter` alone — bboxes
    carry no filename, so `bboxMatchesFilter` must ignore it entirely."""

    # An image with a filename plus one plain bbox.
    NAMED = "{uuid:'i1', filename:'datasets/cams/front_2024.jpg', bboxes:[bbox('cat')]}"

    def test_empty_search_is_a_noop(self, shared_page: Page):
        assert _eval(shared_page, f"imageMatchesFilter({self.NAMED}, st({{filenameSearch:''}}), labels)") is True

    def test_substring_matches_anywhere_in_the_path(self, shared_page: Page):
        for needle in ("front", "cams/", "2024.jpg", "datasets"):
            assert (
                _eval(shared_page, f"imageMatchesFilter({self.NAMED}, st({{filenameSearch:'{needle}'}}), labels)")
                is True
            ), needle

    def test_non_matching_substring_rejects(self, shared_page: Page):
        assert _eval(shared_page, f"imageMatchesFilter({self.NAMED}, st({{filenameSearch:'rear'}}), labels)") is False

    def test_case_insensitive_both_directions(self, shared_page: Page):
        # SQLite's LIKE is case-insensitive for ASCII; String.includes is not,
        # so the predicate lowercases both sides to stay in step with it.
        assert _eval(shared_page, f"imageMatchesFilter({self.NAMED}, st({{filenameSearch:'FRONT'}}), labels)") is True
        assert (
            _eval(
                shared_page,
                "imageMatchesFilter({uuid:'i1', filename:'A/B/CAM.JPG', bboxes:[bbox('cat')]}, st({filenameSearch:'cam'}), labels)",
            )
            is True
        )

    def test_whitespace_only_search_is_a_noop(self, shared_page: Page):
        assert _eval(shared_page, f"imageMatchesFilter({self.NAMED}, st({{filenameSearch:'   '}}), labels)") is True

    def test_rejects_bbox_free_image(self, shared_page: Page):
        # The bbox-scoped filters deliberately let bbox-free images through
        # (hasActiveBboxFilter is false under the default state). Filename is
        # image-scoped, so it must still reject one whose path doesn't match.
        assert (
            _eval(
                shared_page,
                "imageMatchesFilter({uuid:'i1', filename:'a/b.jpg', bboxes:[]}, st({filenameSearch:'zzz'}), labels)",
            )
            is False
        )
        assert (
            _eval(
                shared_page,
                "imageMatchesFilter({uuid:'i1', filename:'a/b.jpg', bboxes:[]}, st({filenameSearch:'b.jpg'}), labels)",
            )
            is True
        )

    def test_missing_filename_field_rejects_when_searching(self, shared_page: Page):
        assert _eval(shared_page, "imageMatchesFilter(img(bbox('cat')), st({filenameSearch:'x'}), labels)") is False
        assert _eval(shared_page, "imageMatchesFilter(img(bbox('cat')), st(), labels)") is True

    def test_bbox_predicate_ignores_filename(self, shared_page: Page):
        # A bbox has no filename; the bbox predicate must not start rejecting
        # every bbox just because a filename search is active.
        assert _eval(shared_page, "bboxMatchesFilter(bbox('cat'), st({filenameSearch:'zzz'}), labels)") is True

    def test_ands_with_bbox_filters(self, shared_page: Page):
        expr = "imageMatchesFilter({uuid:'i1', filename:'cams/front.jpg', bboxes:[bbox('cat')]}, st({filenameSearch:'%s', labelUuids:['%s']}), labels)"
        assert _eval(shared_page, expr % ("front", "cat")) is True
        assert _eval(shared_page, expr % ("front", "dog")) is False  # bbox filter fails
        assert _eval(shared_page, expr % ("rear", "cat")) is False  # filename fails


# ---------------------------------------------------------------------------
# Created-date range (image-scoped, not bbox-scoped)
# ---------------------------------------------------------------------------


class TestDateFilter:
    """`dateFrom`/`dateTo` mirror the server's DateTerm on `created`. Like
    filename they are image-scoped and applied by `imageMatchesFilter` alone."""

    # createdAt as the API actually serialises it: naive, no UTC offset.
    NAIVE = "{uuid:'i1', filename:'a.jpg', createdAt:'2026-06-15T12:00:00.123456', bboxes:[bbox('cat')]}"

    def test_no_bounds_is_a_noop(self, shared_page: Page):
        assert _eval(shared_page, f"imageMatchesFilter({self.NAIVE}, st(), labels)") is True

    def test_offsetless_timestamps_are_read_as_utc(self, shared_page: Page):
        # The regression this guards: `new Date('2026-06-15T12:00:00')` is
        # LOCAL time, so in any non-UTC browser an image stored at 12:00 UTC
        # would fall on the wrong side of a 12:00 UTC bound. parseRecordDate
        # pins offset-less values to UTC.
        assert (
            _eval(
                shared_page,
                "(await import('/app-static/js/app/review/filterPredicate.js'))"
                ".parseRecordDate('2026-06-15T12:00:00') === Date.UTC(2026, 5, 15, 12, 0, 0)",
            )
            is True
        )

    def test_explicit_offset_is_respected(self, shared_page: Page):
        assert (
            _eval(
                shared_page,
                "(await import('/app-static/js/app/review/filterPredicate.js'))"
                ".parseRecordDate('2026-06-15T12:00:00Z') === Date.UTC(2026, 5, 15, 12, 0, 0)",
            )
            is True
        )
        assert (
            _eval(
                shared_page,
                "(await import('/app-static/js/app/review/filterPredicate.js'))"
                ".parseRecordDate('2026-06-15T14:00:00+02:00') === Date.UTC(2026, 5, 15, 12, 0, 0)",
            )
            is True
        )

    def test_from_bound_is_inclusive(self, shared_page: Page):
        # Exactly on the bound passes (server uses >=).
        assert (
            _eval(shared_page, f"imageMatchesFilter({self.NAIVE}, st({{dateFrom:'2026-06-15T12:00:00.123Z'}}), labels)")
            is True
        )
        assert (
            _eval(shared_page, f"imageMatchesFilter({self.NAIVE}, st({{dateFrom:'2026-06-14T00:00:00Z'}}), labels)")
            is True
        )
        assert (
            _eval(shared_page, f"imageMatchesFilter({self.NAIVE}, st({{dateFrom:'2026-06-16T00:00:00Z'}}), labels)")
            is False
        )

    def test_to_bound_is_inclusive(self, shared_page: Page):
        assert (
            _eval(shared_page, f"imageMatchesFilter({self.NAIVE}, st({{dateTo:'2026-06-16T00:00:00Z'}}), labels)")
            is True
        )
        assert (
            _eval(shared_page, f"imageMatchesFilter({self.NAIVE}, st({{dateTo:'2026-06-14T00:00:00Z'}}), labels)")
            is False
        )

    def test_both_bounds_form_a_range(self, shared_page: Page):
        inside = "st({dateFrom:'2026-06-01T00:00:00Z', dateTo:'2026-07-01T00:00:00Z'})"
        outside = "st({dateFrom:'2026-01-01T00:00:00Z', dateTo:'2026-02-01T00:00:00Z'})"
        assert _eval(shared_page, f"imageMatchesFilter({self.NAIVE}, {inside}, labels)") is True
        assert _eval(shared_page, f"imageMatchesFilter({self.NAIVE}, {outside}, labels)") is False

    def test_missing_created_at_fails_an_active_bound(self, shared_page: Page):
        # Mirrors SQL: a NULL never satisfies a comparison.
        assert (
            _eval(shared_page, "imageMatchesFilter(img(bbox('cat')), st({dateFrom:'2026-01-01T00:00:00Z'}), labels)")
            is False
        )
        assert _eval(shared_page, "imageMatchesFilter(img(bbox('cat')), st(), labels)") is True

    def test_rejects_bbox_free_image(self, shared_page: Page):
        assert (
            _eval(
                shared_page,
                "imageMatchesFilter({uuid:'i1', createdAt:'2026-06-15T12:00:00', bboxes:[]}, st({dateFrom:'2026-07-01T00:00:00Z'}), labels)",
            )
            is False
        )

    def test_bbox_predicate_ignores_dates(self, shared_page: Page):
        assert (
            _eval(shared_page, "bboxMatchesFilter(bbox('cat'), st({dateFrom:'2030-01-01T00:00:00Z'}), labels)") is True
        )

    def test_ands_with_filename_and_bbox_filters(self, shared_page: Page):
        base = "{uuid:'i1', filename:'cams/front.jpg', createdAt:'2026-06-15T12:00:00', bboxes:[bbox('cat')]}"
        ok = "st({dateFrom:'2026-06-01T00:00:00Z', filenameSearch:'cams', labelUuids:['cat']})"
        bad_date = "st({dateFrom:'2026-07-01T00:00:00Z', filenameSearch:'cams', labelUuids:['cat']})"
        bad_name = "st({dateFrom:'2026-06-01T00:00:00Z', filenameSearch:'nope', labelUuids:['cat']})"
        assert _eval(shared_page, f"imageMatchesFilter({base}, {ok}, labels)") is True
        assert _eval(shared_page, f"imageMatchesFilter({base}, {bad_date}, labels)") is False
        assert _eval(shared_page, f"imageMatchesFilter({base}, {bad_name}, labels)") is False


# ---------------------------------------------------------------------------
# Confidence band — bboxMatchesConfidence / bboxMatchesFilter
# ---------------------------------------------------------------------------


class TestConfidenceBand:
    """The client-side mirror of the server's BBoxConfidenceTerm. Confidence
    lives in the bbox's raw `metadataJson`, under either of two keys, as either
    a string or a number."""

    def test_no_bounds_is_a_noop(self, shared_page: Page):
        # Even a bbox with no confidence at all passes when no bound is set.
        assert _eval(shared_page, "bboxMatchesFilter(cbox(null), st(), labels)") is True
        assert _eval(shared_page, "bboxMatchesFilter(cbox({confidence: 0.1}), st(), labels)") is True

    def test_include_missing_alone_is_not_an_active_filter(self, shared_page: Page):
        # Matches the server: a bounds-less term with include_missing is true().
        assert _eval(shared_page, "M.hasActiveConfidenceFilter(st({confidenceIncludeMissing: true}))") is False
        assert _eval(shared_page, "M.hasActiveConfidenceFilter(st({confidenceMin: 0.5}))") is True
        assert _eval(shared_page, "M.hasActiveConfidenceFilter(st({confidenceMax: 0.5}))") is True

    def test_bounds_are_inclusive(self, shared_page: Page):
        assert (
            _eval(shared_page, "bboxMatchesFilter(cbox({confidence: 0.5}), st({confidenceMin: 0.5}), labels)") is True
        )
        assert (
            _eval(shared_page, "bboxMatchesFilter(cbox({confidence: 0.5}), st({confidenceMax: 0.5}), labels)") is True
        )
        assert (
            _eval(shared_page, "bboxMatchesFilter(cbox({confidence: 0.49}), st({confidenceMin: 0.5}), labels)") is False
        )
        assert (
            _eval(shared_page, "bboxMatchesFilter(cbox({confidence: 0.51}), st({confidenceMax: 0.5}), labels)") is False
        )

    def test_band_needs_both_bounds_satisfied(self, shared_page: Page):
        band = "st({confidenceMin: 0.4, confidenceMax: 0.6})"
        assert _eval(shared_page, f"bboxMatchesFilter(cbox({{confidence: 0.5}}), {band}, labels)") is True
        assert _eval(shared_page, f"bboxMatchesFilter(cbox({{confidence: 0.3}}), {band}, labels)") is False
        assert _eval(shared_page, f"bboxMatchesFilter(cbox({{confidence: 0.7}}), {band}, labels)") is False

    def test_reads_string_values_and_the_source_confidence_key(self, shared_page: Page):
        # The canvas save path writes a string; ApplyFilterChainTask writes
        # `source_confidence`. Both must filter identically to a plain float.
        band = "st({confidenceMin: 0.4, confidenceMax: 0.6})"
        assert _eval(shared_page, f"bboxMatchesFilter(cbox({{confidence: '0.5'}}), {band}, labels)") is True
        assert _eval(shared_page, f"bboxMatchesFilter(cbox({{source_confidence: 0.5}}), {band}, labels)") is True
        assert _eval(shared_page, f"bboxMatchesFilter(cbox({{source_confidence: '0.9'}}), {band}, labels)") is False

    def test_missing_confidence_fails_an_active_band(self, shared_page: Page):
        # An absent confidence is not 0 — it must not fall into a low band.
        for meta in ("null", "{}", "{other: 1}", "{confidence: null}", "{confidence: 'n/a'}"):
            assert (
                _eval(shared_page, f"bboxMatchesFilter(cbox({meta}), st({{confidenceMax: 0.9}}), labels)") is False
            ), meta

    def test_include_missing_opts_unscored_boxes_back_in(self, shared_page: Page):
        state = "st({confidenceMax: 0.9, confidenceIncludeMissing: true})"
        assert _eval(shared_page, f"bboxMatchesFilter(cbox(null), {state}, labels)") is True
        # Still a real filter for boxes that DO have a confidence.
        assert _eval(shared_page, f"bboxMatchesFilter(cbox({{confidence: 0.95}}), {state}, labels)") is False

    def test_ands_with_verification(self, shared_page: Page):
        """The headline use case: unverified boxes the model was unsure about."""
        state = "st({bboxVerification: 'unverified', confidenceMax: 0.5})"
        low_unverified = "{...cbox({confidence: 0.2}), tags: [{uuid: UNVERIFIED}]}"
        low_verified = "{...cbox({confidence: 0.2}), tags: [{uuid: VERIFIED}]}"
        high_unverified = "{...cbox({confidence: 0.9}), tags: [{uuid: UNVERIFIED}]}"
        assert _eval(shared_page, f"bboxMatchesFilter({low_unverified}, {state}, labels)") is True
        assert _eval(shared_page, f"bboxMatchesFilter({low_verified}, {state}, labels)") is False
        assert _eval(shared_page, f"bboxMatchesFilter({high_unverified}, {state}, labels)") is False

    def test_image_with_no_bboxes_is_rejected_by_an_active_band(self, shared_page: Page):
        # An active band is a bbox-scoped constraint, so hasActiveBboxFilter must
        # see it — otherwise bbox-free images would leak through the filter.
        assert (
            _eval(shared_page, "imageMatchesFilter({uuid:'i1', bboxes:[]}, st({confidenceMin: 0.5}), labels)") is False
        )
        assert _eval(shared_page, "imageMatchesFilter({uuid:'i1', bboxes:[]}, st(), labels)") is True

    def test_image_matches_when_any_bbox_is_in_band(self, shared_page: Page):
        both = "{uuid:'i1', bboxes:[cbox({confidence: 0.1}), cbox({confidence: 0.9})]}"
        assert _eval(shared_page, f"imageMatchesFilter({both}, st({{confidenceMin: 0.8}}), labels)") is True
        assert _eval(shared_page, f"imageMatchesFilter({both}, st({{confidenceMin: 0.95}}), labels)") is False
