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
