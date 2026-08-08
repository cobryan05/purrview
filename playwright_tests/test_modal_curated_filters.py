"""
Tests for M1 of the "unify Review filters" plan — the curated result-filter
surface (bbox Verification + Label + Tag) that `createReviewPanel` builds in
custom/fetchImages mode (similarity/dissimilarity modals), reusing V1's
`createBboxVerificationControl` and F1's `filterPredicate.js` predicate.

There were previously NO Playwright tests for the similarity/dissimilarity
modals (`ui/similarSearch.js` / `ui/dissimilarSearch.js`), so this exercises
the panel itself (`showReviewModal({ fetchImages, ... })`) with a stub
`fetchImages` returning a small fixed candidate set — hermetic, no BK-tree.

Layout (see main.js's fetchImages-mode curated-surface block):
  .bbox-verification-select        — V1 tri-state control
  .ms-dropdown[data-filter-key='curated-labels']
  .ms-dropdown[data-filter-key='curated-tags']

Run with:
    docker compose exec -T mono-dev bash -lc \
      'cd /project/playwright_tests && python -m pytest test_modal_curated_filters.py \
       test_run_model_on_results.py test_filter_panel.py test_review_workflow_ui.py \
       -o addopts="" -p no:xdist -q'
"""

import json

import pytest
from playwright.sync_api import Page, Route, expect

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"
VERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000005"

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}
LABEL_DOG = {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00", "parentUuid": None}

UNVERIFIED_TAG = {"uuid": UNVERIFIED_TAG_UUID, "name": "Unverified", "color": "#888", "kind": "system", "icon": None}
VERIFIED_TAG = {"uuid": VERIFIED_TAG_UUID, "name": "Verified", "color": "#2c2", "kind": "system", "icon": "✓"}
REVIEWED_TAG = {"uuid": "tag-reviewed", "name": "Reviewed", "color": "#4c4", "kind": "user", "icon": None}

# bbox-1: Cat, unverified, no extra tag
# bbox-2: Dog, verified, "Reviewed" tag
# bbox-3: Cat, no verification tag, no extra tag
BBOX_1 = {
    "uuid": "bbox-1",
    "x": 0.1,
    "y": 0.1,
    "width": 0.2,
    "height": 0.2,
    "label": {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000"},
    "labelUuid": "lbl-cat",
    "tags": [UNVERIFIED_TAG],
}
BBOX_2 = {
    "uuid": "bbox-2",
    "x": 0.3,
    "y": 0.3,
    "width": 0.1,
    "height": 0.1,
    "label": {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00"},
    "labelUuid": "lbl-dog",
    "tags": [VERIFIED_TAG, REVIEWED_TAG],
}
BBOX_3 = {
    "uuid": "bbox-3",
    "x": 0.5,
    "y": 0.5,
    "width": 0.1,
    "height": 0.1,
    "label": {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000"},
    "labelUuid": "lbl-cat",
    "tags": [],
}

IMAGE_1 = {"uuid": "img-1", "filename": "folder/image1.jpg", "bboxes": [BBOX_1]}
IMAGE_2 = {"uuid": "img-2", "filename": "folder/image2.jpg", "bboxes": [BBOX_2]}
IMAGE_3 = {"uuid": "img-3", "filename": "folder/image3.jpg", "bboxes": [BBOX_3]}

CANDIDATES = [IMAGE_1, IMAGE_2, IMAGE_3]


def _mock_labels(page: Page, labels: list):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": labels}),
        ),
    )


def _mock_tags(page: Page, tags: list):
    page.route(
        "**/api/tags/list",
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"tags": tags})),
    )


def _mock_image_labels_batch(page: Page):
    page.route(
        "**/api/images/labels/batch*",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success", "imageLabels": {}})
        ),
    )


def _open_curated_modal(page: Page, base_url: str, candidates=None):
    """Open showReviewModal with a stub fetchImages returning `candidates` (a
    fixed, unpaginated set — the stub ignores afterUuid/limit for simplicity,
    matching how small a fixture-driven test needs to be)."""
    candidates = candidates if candidates is not None else CANDIDATES
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.evaluate(
        """async (candidates) => {
            const { showReviewModal } = await import('/app-static/js/app/review/main.js');
            window.__refresh = await showReviewModal({
                title: 'Curated Filter Test',
                initialViewFilter: 'bboxes',
                fetchImages: async () => candidates,
            });
        }""",
        candidates,
    )
    page.wait_for_selector(".review-grid", timeout=4000)


class TestCuratedSurfaceRenders:
    def test_verification_label_tag_controls_present(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, REVIEWED_TAG])
        _mock_image_labels_batch(page)

        _open_curated_modal(page, base_url)

        expect(page.locator(".bbox-verification-select")).to_be_visible()
        expect(page.locator(".ms-dropdown[data-filter-key='curated-labels']")).to_be_visible()
        expect(page.locator(".ms-dropdown[data-filter-key='curated-tags']")).to_be_visible()

    def test_all_three_bboxes_visible_by_default(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, REVIEWED_TAG])
        _mock_image_labels_batch(page)

        _open_curated_modal(page, base_url)

        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_be_visible()


class TestCuratedVerificationFilter:
    def test_filter_to_verified_shows_only_bbox2(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, REVIEWED_TAG])
        _mock_image_labels_batch(page)

        _open_curated_modal(page, base_url)

        page.locator(".bbox-verification-select").select_option("verified")
        page.wait_for_timeout(200)

        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_have_count(0)
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_have_count(0)


class TestCuratedLabelFilter:
    def test_filter_to_dog_label_shows_only_bbox2(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, REVIEWED_TAG])
        _mock_image_labels_batch(page)

        _open_curated_modal(page, base_url)

        page.locator(".ms-dropdown[data-filter-key='curated-labels'] .ms-dropdown__trigger").click()
        page.wait_for_timeout(100)
        page.locator(".ms-dropdown[data-filter-key='curated-labels'] input[value='lbl-dog']").check()
        page.wait_for_timeout(200)

        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_have_count(0)
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_have_count(0)


class TestCuratedTagFilter:
    def test_include_reviewed_tag_shows_only_bbox2(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, REVIEWED_TAG])
        _mock_image_labels_batch(page)

        _open_curated_modal(page, base_url)

        page.locator(".ms-dropdown[data-filter-key='curated-tags'] .ms-dropdown__trigger").click()
        page.wait_for_timeout(100)
        page.locator(".ms-dropdown[data-filter-key='curated-tags'] input[value='tag-reviewed']").check()
        page.wait_for_timeout(200)

        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_have_count(0)
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_have_count(0)

    def test_exclude_reviewed_tag_hides_bbox2(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, REVIEWED_TAG])
        _mock_image_labels_batch(page)

        _open_curated_modal(page, base_url)

        # Verification tags (UNVERIFIED/VERIFIED) must NOT appear in the curated
        # Tags dropdown — that is the verification control's exclusive domain
        # (V1 decision, mirrored from filters.js's _visibleTags()).
        expect(
            page.locator(".ms-dropdown[data-filter-key='curated-tags'] input[value='" + VERIFIED_TAG_UUID + "']")
        ).to_have_count(0)
        expect(
            page.locator(".ms-dropdown[data-filter-key='curated-tags'] input[value='" + UNVERIFIED_TAG_UUID + "']")
        ).to_have_count(0)

        page.locator(".ms-dropdown[data-filter-key='curated-tags'] .ms-dropdown__trigger").click()
        page.wait_for_timeout(100)
        # Click the "exclude" (NOT) toggle for the Reviewed tag row.
        row = page.locator(
            ".ms-dropdown[data-filter-key='curated-tags'] .ms-item", has=page.locator("input[value='tag-reviewed']")
        )
        row.locator(".ms-item__not-btn").click()
        page.wait_for_timeout(200)

        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_have_count(0)
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_be_visible()


class TestApplyToAllRespectsCuratedFilter:
    """M3: Apply-to-All (fetchImages/custom mode) must apply the curated
    Verification/Label/Tag predicate before building its candidate list, so
    the count shown in the bulk-action modal reflects the *filtered* set —
    not every candidate the stub fetchImages returns."""

    def test_apply_to_all_count_reflects_verification_filter(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, REVIEWED_TAG])
        _mock_image_labels_batch(page)

        _open_curated_modal(page, base_url)

        # Narrow to "verified" only — of the 3 candidates, only bbox-2 qualifies.
        page.locator(".bbox-verification-select").select_option("verified")
        page.wait_for_timeout(200)

        page.locator("button", has_text="Apply to All").click()

        # showBulkActionModal's title reflects entries.length from the curated,
        # not raw, candidate set — asserts M3's fix in applyToAllCandidates.
        expect(page.locator("strong", has_text="Apply to all 1 matched bbox")).to_be_visible()


class TestCuratedFilenameFilter:
    """The filename substring control is image-scoped: it lives in the same
    curated group but is applied by `imageMatchesFilter` as its own gate, not
    by the bbox predicate. Matching is against the FILES_ROOT-*relative* path
    the server already puts on each record."""

    # Same three bboxes, but each image in a distinguishable path.
    IMAGES = [
        {"uuid": "img-1", "filename": "cams/front/shot1.jpg", "bboxes": [BBOX_1]},
        {"uuid": "img-2", "filename": "cams/rear/shot2.jpg", "bboxes": [BBOX_2]},
        {"uuid": "img-3", "filename": "archive/rear_old.jpg", "bboxes": [BBOX_3]},
    ]

    def _open(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, REVIEWED_TAG])
        _mock_image_labels_batch(page)
        _open_curated_modal(page, base_url, candidates=self.IMAGES)

    def _search(self, page: Page, text: str):
        page.locator("[data-testid='curated-filename-input']").fill(text)
        # The input is debounced by 400ms before it triggers a re-render.
        page.wait_for_timeout(700)

    def test_control_is_present(self, page: Page, base_url: str):
        self._open(page, base_url)
        expect(page.locator("[data-testid='curated-filename-input']")).to_be_visible()

    def test_substring_narrows_to_one_image(self, page: Page, base_url: str):
        self._open(page, base_url)
        self._search(page, "shot1")

        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_have_count(0)
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_have_count(0)

    def test_directory_segment_matches_multiple(self, page: Page, base_url: str):
        self._open(page, base_url)
        self._search(page, "cams/")

        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_have_count(0)

    def test_case_insensitive(self, page: Page, base_url: str):
        self._open(page, base_url)
        self._search(page, "SHOT2")

        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_have_count(0)

    def test_clearing_restores_all(self, page: Page, base_url: str):
        self._open(page, base_url)
        self._search(page, "shot1")
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_have_count(0)

        self._search(page, "")
        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_be_visible()

    def test_ands_with_the_bbox_filters(self, page: Page, base_url: str):
        self._open(page, base_url)
        # "rear" matches img-2 and img-3; verified narrows to img-2's bbox only.
        self._search(page, "rear")
        page.locator(".bbox-verification-select").select_option("verified")
        page.wait_for_timeout(300)

        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_have_count(0)
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_have_count(0)

    def test_apply_to_all_count_reflects_filename_filter(self, page: Page, base_url: str):
        # M3's rule: the whole-candidate-set path must run the same predicate.
        self._open(page, base_url)
        self._search(page, "cams/")

        page.locator("button", has_text="Apply to All").click()
        expect(page.locator("strong", has_text="Apply to all 2 matched bbox")).to_be_visible()


class TestCuratedDateFilter:
    """The created-date range, also image-scoped and also client-side. The
    fixture timestamps are naive (no UTC offset) because that is exactly what
    the API serialises — `parseRecordDate` pins them to UTC."""

    IMAGES = [
        {"uuid": "img-1", "filename": "a.jpg", "createdAt": "2026-01-10T12:00:00", "bboxes": [BBOX_1]},
        {"uuid": "img-2", "filename": "b.jpg", "createdAt": "2026-06-15T12:00:00", "bboxes": [BBOX_2]},
        {"uuid": "img-3", "filename": "c.jpg", "createdAt": "2026-12-20T12:00:00", "bboxes": [BBOX_3]},
    ]

    def _open(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, REVIEWED_TAG])
        _mock_image_labels_batch(page)
        _open_curated_modal(page, base_url, candidates=self.IMAGES)

    def _set_range(self, page: Page, testid: str, value: str):
        page.locator(f"[data-testid='{testid}']").fill(value)
        page.locator(f"[data-testid='{testid}']").dispatch_event("change")
        page.wait_for_timeout(300)

    def test_controls_are_present(self, page: Page, base_url: str):
        self._open(page, base_url)
        expect(page.locator("[data-testid='curated-date-from']")).to_be_visible()
        expect(page.locator("[data-testid='curated-date-to']")).to_be_visible()

    def test_from_bound_drops_earlier_images(self, page: Page, base_url: str):
        self._open(page, base_url)
        self._set_range(page, "curated-date-from", "2026-03-01T00:00")

        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_have_count(0)
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_be_visible()

    def test_to_bound_drops_later_images(self, page: Page, base_url: str):
        self._open(page, base_url)
        self._set_range(page, "curated-date-to", "2026-09-01T00:00")

        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_have_count(0)

    def test_both_bounds_form_a_window(self, page: Page, base_url: str):
        self._open(page, base_url)
        self._set_range(page, "curated-date-from", "2026-03-01T00:00")
        self._set_range(page, "curated-date-to", "2026-09-01T00:00")

        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_have_count(0)
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_have_count(0)

    def test_clearing_restores_all(self, page: Page, base_url: str):
        self._open(page, base_url)
        self._set_range(page, "curated-date-from", "2026-03-01T00:00")
        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_have_count(0)

        self._set_range(page, "curated-date-from", "")
        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_be_visible()

    def test_apply_to_all_count_reflects_date_filter(self, page: Page, base_url: str):
        self._open(page, base_url)
        self._set_range(page, "curated-date-from", "2026-03-01T00:00")

        page.locator("button", has_text="Apply to All").click()
        expect(page.locator("strong", has_text="Apply to all 2 matched bbox")).to_be_visible()
