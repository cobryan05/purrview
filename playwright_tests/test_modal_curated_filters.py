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
FALSE_POS_TAG_UUID = "00000000-0000-0000-0000-000000000003"

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}
LABEL_DOG = {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00", "parentUuid": None}

UNVERIFIED_TAG = {"uuid": UNVERIFIED_TAG_UUID, "name": "Unverified", "color": "#888", "kind": "system", "icon": None}
VERIFIED_TAG = {"uuid": VERIFIED_TAG_UUID, "name": "Verified", "color": "#2c2", "kind": "system", "icon": "✓"}
FALSE_POS_TAG = {"uuid": FALSE_POS_TAG_UUID, "name": "False Pos", "color": "#c22", "kind": "system", "icon": "✗"}
REVIEWED_TAG = {"uuid": "tag-reviewed", "name": "Reviewed", "color": "#4c4", "kind": "user", "icon": None}

# bbox-1: Cat, unverified, no extra tag
# bbox-2: Dog, verified, "Reviewed" tag
# bbox-3: Cat, false-pos, no extra tag
BBOX_1 = {
    "uuid": "bbox-1", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2,
    "label": {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000"},
    "labelUuid": "lbl-cat",
    "tags": [UNVERIFIED_TAG],
}
BBOX_2 = {
    "uuid": "bbox-2", "x": 0.3, "y": 0.3, "width": 0.1, "height": 0.1,
    "label": {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00"},
    "labelUuid": "lbl-dog",
    "tags": [VERIFIED_TAG, REVIEWED_TAG],
}
BBOX_3 = {
    "uuid": "bbox-3", "x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1,
    "label": {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000"},
    "labelUuid": "lbl-cat",
    "tags": [FALSE_POS_TAG],
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
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success", "imageLabels": {}})),
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
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, FALSE_POS_TAG, REVIEWED_TAG])
        _mock_image_labels_batch(page)

        _open_curated_modal(page, base_url)

        expect(page.locator(".bbox-verification-select")).to_be_visible()
        expect(page.locator(".ms-dropdown[data-filter-key='curated-labels']")).to_be_visible()
        expect(page.locator(".ms-dropdown[data-filter-key='curated-tags']")).to_be_visible()

    def test_all_three_bboxes_visible_by_default(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, FALSE_POS_TAG, REVIEWED_TAG])
        _mock_image_labels_batch(page)

        _open_curated_modal(page, base_url)

        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_be_visible()


class TestCuratedVerificationFilter:
    def test_filter_to_verified_shows_only_bbox2(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, FALSE_POS_TAG, REVIEWED_TAG])
        _mock_image_labels_batch(page)

        _open_curated_modal(page, base_url)

        page.locator(".bbox-verification-select").select_option("verified")
        page.wait_for_timeout(200)

        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_have_count(0)
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_have_count(0)

    def test_filter_to_false_pos_shows_only_bbox3(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, FALSE_POS_TAG, REVIEWED_TAG])
        _mock_image_labels_batch(page)

        _open_curated_modal(page, base_url)

        page.locator(".bbox-verification-select").select_option("false_pos")
        page.wait_for_timeout(200)

        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_have_count(0)
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_have_count(0)
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_be_visible()


class TestCuratedLabelFilter:
    def test_filter_to_dog_label_shows_only_bbox2(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, FALSE_POS_TAG, REVIEWED_TAG])
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
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, FALSE_POS_TAG, REVIEWED_TAG])
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
        _mock_tags(page, [UNVERIFIED_TAG, VERIFIED_TAG, FALSE_POS_TAG, REVIEWED_TAG])
        _mock_image_labels_batch(page)

        _open_curated_modal(page, base_url)

        # Verification tags (UNVERIFIED/VERIFIED/FALSE_POS) must NOT appear in
        # the curated Tags dropdown — that's the tri-state control's exclusive
        # domain (V1 decision, mirrored from filters.js's _visibleTags()).
        expect(page.locator(".ms-dropdown[data-filter-key='curated-tags'] input[value='" + VERIFIED_TAG_UUID + "']")).to_have_count(0)
        expect(page.locator(".ms-dropdown[data-filter-key='curated-tags'] input[value='" + UNVERIFIED_TAG_UUID + "']")).to_have_count(0)
        expect(page.locator(".ms-dropdown[data-filter-key='curated-tags'] input[value='" + FALSE_POS_TAG_UUID + "']")).to_have_count(0)

        page.locator(".ms-dropdown[data-filter-key='curated-tags'] .ms-dropdown__trigger").click()
        page.wait_for_timeout(100)
        # Click the "exclude" (NOT) toggle for the Reviewed tag row.
        row = page.locator(".ms-dropdown[data-filter-key='curated-tags'] .ms-item", has=page.locator("input[value='tag-reviewed']"))
        row.locator(".ms-item__not-btn").click()
        page.wait_for_timeout(200)

        expect(page.locator("[data-bbox-uuid='bbox-1']")).to_be_visible()
        expect(page.locator("[data-bbox-uuid='bbox-2']")).to_have_count(0)
        expect(page.locator("[data-bbox-uuid='bbox-3']")).to_be_visible()
