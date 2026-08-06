"""
Tests for the Review tab UI — apps/image_tagging_server/static/js/app/review/main.js

Exercises createReviewPanel() rendered inside #review-root:
  - Queue rendering (image + bbox cards from mocked /api/images/review data)
  - Verify-state chips
  - Selection (single, shift-click range, select-all) and bulk-action modals
  - Pagination
  - Filter-toggle button state
  - Canvas-modal round trip from a thumbnail double-click
  - Group highlight on hover

Follows the route-interception pattern from test_configuration_tags.py: the
conftest proxies default to empty data, so each test mocks exactly the
endpoints it needs via page.route().

Run with:
    pytest playwright_tests/test_review_workflow_ui.py -p no:xdist -p no:asyncio -v
"""

import base64
import json
import re

import pytest
from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"
VERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000005"

# A 1x1 transparent PNG, base64-encoded — small valid image for canvas loadImage().
TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk" "+A8AAQUBAScY42YAAAAASUVORK5CYII="

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}
LABEL_DOG = {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00", "parentUuid": None}

UNVERIFIED_TAG = {"uuid": UNVERIFIED_TAG_UUID, "icon": None}
VERIFIED_TAG = {"uuid": VERIFIED_TAG_UUID, "icon": "✓"}
REVIEWED_TAG = {"uuid": "tag-reviewed", "name": "Reviewed", "color": "#4c4", "icon": None, "kind": "user"}

# bbox-1, bbox-3: still unverified (carry the UNVERIFIED tag)
# bbox-2: verified (carries the VERIFIED tag)
BBOX_1 = {
    "uuid": "bbox-1",
    "x": 0.1,
    "y": 0.1,
    "width": 0.2,
    "height": 0.2,
    "label": {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000"},
    "labelUuid": "lbl-cat",
    "tags": [UNVERIFIED_TAG],
    "metadataJson": {"confidence": "0.92"},
}
BBOX_2 = {
    "uuid": "bbox-2",
    "x": 0.3,
    "y": 0.3,
    "width": 0.1,
    "height": 0.1,
    "label": {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00"},
    "labelUuid": "lbl-dog",
    "tags": [VERIFIED_TAG],
    "metadataJson": None,
}
BBOX_3 = {
    "uuid": "bbox-3",
    "x": 0.2,
    "y": 0.2,
    "width": 0.15,
    "height": 0.15,
    "label": {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000"},
    "labelUuid": "lbl-cat",
    "tags": [UNVERIFIED_TAG],
    "metadataJson": None,
}

IMAGE_1 = {"uuid": "img-1", "filename": "folder/image1.jpg", "bboxes": [BBOX_1, BBOX_2]}
IMAGE_2 = {"uuid": "img-2", "filename": "folder/image2.jpg", "bboxes": [BBOX_3]}

PAGE_1_IMAGES = [IMAGE_1, IMAGE_2]

IMAGE_3 = {"uuid": "img-3", "filename": "folder/image3.jpg", "bboxes": []}
PAGE_2_IMAGES = [IMAGE_3]


# ---------------------------------------------------------------------------
# Navigation / mocking helpers
# ---------------------------------------------------------------------------


def _navigate_to_review(page: Page, base_url: str):
    """Open the app and activate the Review tab."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-review']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#review-root .review-grid", timeout=4000)


def _mock_review_queue(page: Page, images: list, has_more: bool = False, total_count: int = None):
    """Intercept POST /api/images/review with a fixed page of results."""
    if total_count is None:
        total_count = len(images)
    page.route(
        "**/api/images/review",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "images": images,
                    "hasMore": has_more,
                    "totalCount": total_count,
                }
            ),
        ),
    )


def _mock_paged_review_queue(page: Page, page1: list, page2: list):
    """Serve `page1` for the initial fetch (after_uuid is null) and `page2` after that."""

    def handle(route: Route):
        body = json.loads(route.request.post_data or "{}")
        images = page2 if body.get("afterUuid") else page1
        has_more = images is page1
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "images": images,
                    "hasMore": has_more,
                    "totalCount": len(page1) + len(page2),
                }
            ),
        )

    page.route("**/api/images/review", handle)


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
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"tags": tags}),
        ),
    )


def _mock_misc_review_endpoints(page: Page):
    """Stub out side-effect endpoints (thumbnail invalidation, bbox/image updates)."""
    page.route(
        "**/api/images/bbox/thumbnails/invalidate",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success"})
        ),
    )
    page.route(
        "**/api/images/bbox/*",
        lambda route: (
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))
            if route.request.method in ("PATCH", "DELETE")
            else route.fallback()
        ),
    )
    page.route(
        "**/api/images/labels/set",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"status": "success"})
        ),
    )


def _mock_canvas_modal_image(page: Page):
    """Serve everything the canvas modal needs for img-1: thumbnails (fetched as
    raw bytes via fetch().blob(), so they need a real PNG to have nonzero size and
    be double-clickable), the image file itself, and the metadata-update save."""
    png_bytes = base64.b64decode(TINY_PNG_B64)
    page.route(
        "**/thumbnail?*",
        lambda route: route.fulfill(status=200, content_type="image/png", body=png_bytes),
    )
    page.route(
        "**/api/images/*/file",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "imageBase64": TINY_PNG_B64,
                    "mimeType": "image/png",
                    "image": {"uuid": "img-1", "filename": "folder/image1.jpg", "bboxes": [BBOX_1, BBOX_2]},
                }
            ),
        ),
    )
    page.route(
        "**/api/images/metadata/update",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success"}),
        ),
    )


def _show_all_bboxes(page: Page):
    """Clear the tag filter so all bboxes render regardless of verification state.

    System tags are hidden from the dropdown by default, so we use 'Select None'
    to clear all tag filters (including the hidden Unverified default), which
    triggers showAllBboxes=true on the backend.
    """
    # The tags dropdown is added asynchronously after the tags fetch — wait for it.
    trigger = page.locator("#review-root .ms-dropdown[data-filter-key='tags'] .ms-dropdown__trigger")
    trigger.wait_for(state="visible", timeout=5000)
    trigger.click()
    page.wait_for_timeout(150)
    # Click "Select None" to clear all tag selections (including hidden system tags).
    page.locator(
        "#review-root .ms-dropdown[data-filter-key='tags'] .ms-select-all-row button:text('Select None')"
    ).click()
    # Click outside the dropdown to close it.
    page.mouse.click(0, 0)
    page.wait_for_timeout(300)


def _switch_view(page: Page, key: str):
    page.locator(f"#review-root .view-btn[data-view-key='{key}']").click()
    page.wait_for_timeout(200)


# ---------------------------------------------------------------------------
# 1. Queue rendering
# ---------------------------------------------------------------------------


class TestReviewQueueRendering:
    def test_queue_renders_one_card_per_entry(self, page: Page, base_url: str):
        """Each image gets an image-level card, plus one card per matching bbox."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(
            page,
            [
                (
                    UNVERIFIED_TAG | {"name": "Unverified", "color": "#888"}
                    if False
                    else {**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}
                ),
                REVIEWED_TAG,
            ],
        )

        _navigate_to_review(page, base_url)
        # Switch to 'all' to see both image-level and bbox-level cards.
        _switch_view(page, "all")

        # Default: no tag filter → all bboxes visible (bbox-1, bbox-2, bbox-3).
        cards = page.locator("#review-root [data-testid='review-card']")
        expect(cards).to_have_count(5)  # 2 image cards + 3 bbox cards
        expect(page.locator("#review-root [data-image-uuid='img-1'][data-kind='image']")).to_be_visible()
        expect(page.locator("#review-root [data-image-uuid='img-2'][data-kind='image']")).to_be_visible()
        expect(page.locator("#review-root [data-bbox-uuid='bbox-1']")).to_be_visible()
        expect(page.locator("#review-root [data-bbox-uuid='bbox-2']")).to_be_visible()
        expect(page.locator("#review-root [data-bbox-uuid='bbox-3']")).to_be_visible()


# ---------------------------------------------------------------------------
# 2. Verify chip
# ---------------------------------------------------------------------------


class TestVerifyChip:
    def test_verify_chip_reflects_bbox_state(self, page: Page, base_url: str):
        """The verify chip reads '✓ Verified' for verified bboxes and '◌ Unverified' otherwise."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(
            page,
            [
                {**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"},
                {**VERIFIED_TAG, "name": "Verified", "color": "#27ae60", "kind": "system"},
                REVIEWED_TAG,
            ],
        )

        _navigate_to_review(page, base_url)
        _show_all_bboxes(page)
        _switch_view(page, "bboxes")

        unverified_chip = page.locator("#review-root [data-bbox-uuid='bbox-1'] .verify-chip")
        verified_chip = page.locator("#review-root [data-bbox-uuid='bbox-2'] .verify-chip")

        expect(unverified_chip).to_have_text("◌ Unverified")
        expect(unverified_chip).to_have_class("verify-chip")
        expect(verified_chip).to_have_text("✓ Verified")
        expect(verified_chip).to_have_class("verify-chip verify-chip--verified")

    def test_double_click_verify_chip_advances_state(self, page: Page, base_url: str):
        """Double-clicking an unverified chip advances it Unverified → Verified.

        Exercises the new buildVerifyChip onToggle path, which drives the same
        setVerification adapter as the right-click menu (PATCHes the bbox and
        rebuilds the chip via the IMAGE_BBOXES_CHANGED refresh path).
        """
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(
            page,
            [
                {**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"},
                {**VERIFIED_TAG, "name": "Verified", "color": "#27ae60", "kind": "system"},
                REVIEWED_TAG,
            ],
        )
        _mock_misc_review_endpoints(page)

        _navigate_to_review(page, base_url)
        _show_all_bboxes(page)
        _switch_view(page, "bboxes")

        chip = page.locator("#review-root [data-bbox-uuid='bbox-1'] .verify-chip")
        expect(chip).to_have_text("◌ Unverified")

        # Re-query after each toggle: setVerification replaces the chip element.
        chip.dblclick()
        chip = page.locator("#review-root [data-bbox-uuid='bbox-1'] .verify-chip")
        expect(chip).to_have_text("✓ Verified")
        expect(chip).to_have_class("verify-chip verify-chip--verified")


# ---------------------------------------------------------------------------
# 3. Single checkbox selection
# ---------------------------------------------------------------------------


class TestSingleSelection:
    def test_single_checkbox_selection_updates_bulk_button(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")
        bulk_btn = page.get_by_test_id("review-bulk-apply")
        expect(bulk_btn).to_have_text("Bulk Apply (0)")
        expect(bulk_btn).to_be_disabled()

        page.locator("#review-root [data-bbox-uuid='bbox-1'] input[type=checkbox]").click()

        expect(bulk_btn).to_have_text("Bulk Apply (1)")
        expect(bulk_btn).to_be_enabled()


# ---------------------------------------------------------------------------
# 4. Shift-click range selection
# ---------------------------------------------------------------------------


class TestShiftClickRangeSelection:
    def test_shift_click_selects_contiguous_range(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        _navigate_to_review(page, base_url)
        _show_all_bboxes(page)
        _switch_view(page, "bboxes")

        cards = page.locator("#review-root [data-testid='review-card']")
        expect(cards).to_have_count(3)  # bbox-1, bbox-2, bbox-3

        cards.nth(0).locator("input[type=checkbox]").click()
        cards.nth(2).locator("input[type=checkbox]").click(modifiers=["Shift"])

        for i in range(3):
            expect(cards.nth(i).locator("input[type=checkbox]")).to_be_checked()
        bulk_btn = page.get_by_test_id("review-bulk-apply")
        expect(bulk_btn).to_have_text("Bulk Apply (3)")


# ---------------------------------------------------------------------------
# 5. Select-all
# ---------------------------------------------------------------------------


class TestSelectAll:
    def test_select_all_toggles_all_visible_cards(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        _navigate_to_review(page, base_url)
        _show_all_bboxes(page)
        _switch_view(page, "bboxes")

        cards = page.locator("#review-root [data-testid='review-card']")
        expect(cards).to_have_count(3)

        select_all = page.locator("#review-root label", has_text="Select all").locator("input[type=checkbox]")
        select_all.check()

        for i in range(3):
            expect(cards.nth(i).locator("input[type=checkbox]")).to_be_checked()
        bulk_btn = page.get_by_test_id("review-bulk-apply")
        expect(bulk_btn).to_have_text("Bulk Apply (3)")

        # Unchecking clears the selection again
        select_all.uncheck()
        for i in range(3):
            expect(cards.nth(i).locator("input[type=checkbox]")).not_to_be_checked()
        expect(bulk_btn).to_have_text("Bulk Apply (0)")


# ---------------------------------------------------------------------------
# 6. Bulk relabel (bbox) modal
# ---------------------------------------------------------------------------


class TestBulkBboxAction:
    def test_bulk_relabel_modal_fires_api_with_selected_entries(self, page: Page, base_url: str):
        """Selecting bboxes, choosing 'Change Label' and a label PATCHes each selected bbox."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)

        calls = []

        def handle_patch(route: Route):
            if route.request.method == "PATCH":
                calls.append({"url": route.request.url, "body": json.loads(route.request.post_data)})
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

        page.route("**/api/images/bbox/bbox-1", handle_patch)
        page.route("**/api/images/bbox/bbox-3", handle_patch)

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")

        # Default tag filter shows the two unverified bboxes (bbox-1, bbox-3).
        page.locator("#review-root [data-bbox-uuid='bbox-1'] input[type=checkbox]").click()
        page.locator("#review-root [data-bbox-uuid='bbox-3'] input[type=checkbox]").click()

        page.get_by_test_id("review-bulk-apply").click()

        modal = page.locator(
            "[data-testid='modal-shell-overlay'], div:has(> strong:text-is('Apply to 2 selected bboxes'))"
        ).last
        expect(page.locator("strong", has_text="Apply to 2 selected bboxes")).to_be_visible()

        # 'Change Label' (relabel) is selected by default.
        # Use the dynamic label picker: type to filter then click the Dog result.
        modal_locator = page.locator("div:has(> strong:text-is('Apply to 2 selected bboxes'))")
        picker_input = modal_locator.locator("input[type=text]")
        picker_input.click()
        picker_input.fill("Dog")
        # The results dropdown appears; click the result row for Dog scoped to the modal.
        modal_locator.get_by_text("Dog", exact=True).click()
        page.locator("div:has(> strong:text-is('Apply to 2 selected bboxes')) button", has_text="Apply").click()
        wait_until(page, lambda: len(calls) >= 2, lambda: f"Expected 2 PATCH calls, got {calls}")

        assert len(calls) == 2, f"Expected 2 PATCH calls, got {calls}"
        for call in calls:
            assert call["body"]["labelUuid"] == "lbl-dog"
            # Relabeling to a real label strips UNVERIFIED and adds VERIFIED.
            assert UNVERIFIED_TAG_UUID not in call["body"]["tagUuids"]
            assert VERIFIED_TAG_UUID in call["body"]["tagUuids"]


class TestBulkApplyAllScope:
    """ "Apply to All" must send the same filter the grid is showing.

    The call site used to spell out its own subset of the filter opts and had
    fallen two dimensions behind _baseReviewFilterOpts — bboxVerification and
    imageLabelConditions — so filtering to "Unverified only" and applying to
    all also hit verified bboxes, with nothing in the UI saying so.
    Follow-up PR A, UI_REDESIGN §32.5.
    """

    def test_apply_to_all_carries_the_bbox_verification_filter(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)

        calls = []

        def handle_bulk(route: Route):
            calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "modifiedCount": 1, "failCount": 0}),
            )

        page.route("**/api/images/bbox/bulk-apply", handle_bulk)

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")

        # Narrow the grid to verified bboxes only, via the filter popover.
        page.get_by_test_id("review-filter-toggle").click()
        page.locator("#review-root .bbox-verification-select").select_option("verified")
        page.get_by_test_id("review-filter-toggle").click()  # close the popover

        page.get_by_test_id("review-bulk-apply-all").click()
        page.locator("input[type=radio][name='bulk-all-action'][value='markVerified']").click()
        page.locator("button", has_text="Apply to All").last.click()

        wait_until(page, lambda: calls, lambda: f"Expected POST to bulk-apply, got {calls}")
        body = calls[0]
        # Bbox-level: without this the action reaches the unverified bboxes
        # sharing those images, which the grid does not show.
        assert body.get("bboxVerification") == "verified", body
        # Image-level: the same predicate as a filter term, so paging skips
        # images with no verified bbox at all.
        assert {"kind": "bboxVerification", "state": "verified"} in body["filter"]["terms"], body


# ---------------------------------------------------------------------------
# 6b. Apply to Parent Images (bbox selection -> distinct parent images)
# ---------------------------------------------------------------------------


class TestBulkParentImagesAction:
    def test_apply_to_parent_images_resolves_distinct_images_from_bbox_selection(self, page: Page, base_url: str):
        """Selecting bboxes from DIFFERENT images and using 'Apply to Parent
        Images' routes the DISTINCT parent image uuids into the image-level
        bulk endpoint — not one call per bbox, and not the bbox bulk endpoint.
        """
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)

        calls = []

        def handle_bulk_apply(route: Route):
            calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "modifiedCount": 2, "failCount": 0, "skippedCount": 0}),
            )

        page.route("**/api/images/bulk-apply-by-uuids", handle_bulk_apply)

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")

        # Default tag filter shows the two unverified bboxes: bbox-1 (img-1),
        # bbox-3 (img-2) — two DIFFERENT parent images.
        page.locator("#review-root [data-bbox-uuid='bbox-1'] input[type=checkbox]").click()
        page.locator("#review-root [data-bbox-uuid='bbox-3'] input[type=checkbox]").click()

        parent_btn = page.get_by_test_id("review-bulk-apply-parent")
        expect(parent_btn).to_be_visible()
        expect(parent_btn).to_have_text(re.compile(r"\(2\)"))
        parent_btn.click()

        # Title must be explicit about acting on 2 distinct IMAGES, not bboxes.
        modal_title = page.locator("strong", has_text="distinct image")
        expect(modal_title).to_be_visible()
        expect(modal_title).to_contain_text("2 distinct image")
        expect(modal_title).to_contain_text("2 selected bbox")

        page.locator("input[type=radio][value=setAbsent]").click()
        modal_locator = page.locator("div:has(> strong:has-text('distinct image'))")
        modal_locator.locator("select").select_option("lbl-cat")
        modal_locator.locator("button", has_text="Apply").click()
        wait_until(page, lambda: len(calls) >= 1, lambda: f"Expected 1 batched bulk-apply-by-uuids call, got {calls}")

        assert len(calls) == 1, f"Expected 1 batched bulk-apply-by-uuids call, got {calls}"
        call = calls[0]
        assert call["action"] == "setAbsent"
        assert call["labelUuid"] == "lbl-cat"
        assert sorted(call["imageUuids"]) == ["img-1", "img-2"]


# ---------------------------------------------------------------------------
# 7. Bulk image action modal
# ---------------------------------------------------------------------------


class TestBulkImageAction:
    def test_image_bulk_action_modal_fires_api_with_selected_entries(self, page: Page, base_url: str):
        """Selecting image cards and confirming 'Mark label Absent' calls the bulk-apply-by-uuids API.

        Label actions route through the same server-side bulk endpoint as tag
        actions (one request for the whole selection), not one
        /api/images/labels/set call per image.
        """
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)

        calls = []

        def handle_bulk_apply(route: Route):
            calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "modifiedCount": 2, "failCount": 0, "skippedCount": 0}),
            )

        page.route("**/api/images/bulk-apply-by-uuids", handle_bulk_apply)

        _navigate_to_review(page, base_url)
        _switch_view(page, "images")

        cards = page.locator("#review-root [data-testid='review-card']")
        expect(cards).to_have_count(2)
        for i in range(2):
            cards.nth(i).locator("input[type=checkbox]").click()

        page.get_by_test_id("review-bulk-apply").click()

        modal_title = page.locator("strong", has_text="Apply to 2 selected images")
        expect(modal_title).to_be_visible()

        # Explicitly select 'Mark label Absent' and choose Cat.
        page.locator("input[type=radio][value=setAbsent]").click()
        page.locator("div:has(> strong:text-is('Apply to 2 selected images')) select").select_option("lbl-cat")
        page.locator("div:has(> strong:text-is('Apply to 2 selected images')) button", has_text="Apply").click()
        wait_until(page, lambda: len(calls) >= 1, lambda: f"Expected 1 batched bulk-apply-by-uuids call, got {calls}")

        assert len(calls) == 1, f"Expected 1 batched bulk-apply-by-uuids call, got {calls}"
        call = calls[0]
        assert call["action"] == "setAbsent"
        assert call["labelUuid"] == "lbl-cat"
        assert sorted(call["imageUuids"]) == ["img-1", "img-2"]


# ---------------------------------------------------------------------------
# 8. Pagination
# ---------------------------------------------------------------------------


class TestPagination:
    def test_next_and_prev_advance_and_return(self, page: Page, base_url: str):
        _mock_paged_review_queue(page, PAGE_1_IMAGES, PAGE_2_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        _navigate_to_review(page, base_url)
        # Image cards only appear in 'all' view (default is 'bboxes').
        _switch_view(page, "all")

        prev_btn = page.get_by_test_id("review-page-prev")
        next_btn = page.get_by_test_id("review-page-next")

        img_card_1 = page.locator("#review-root [data-image-uuid='img-1'][data-kind='image']")
        img_card_3 = page.locator("#review-root [data-image-uuid='img-3'][data-kind='image']")

        expect(prev_btn).to_be_disabled()
        expect(next_btn).to_be_enabled()
        expect(img_card_1).to_be_visible()

        next_btn.click()
        page.wait_for_timeout(500)
        expect(img_card_3).to_be_visible()
        expect(img_card_1).to_have_count(0)
        expect(prev_btn).to_be_enabled()

        prev_btn.click()
        page.wait_for_timeout(500)
        expect(img_card_1).to_be_visible()
        expect(img_card_3).to_have_count(0)
        expect(prev_btn).to_be_disabled()


# ---------------------------------------------------------------------------
# 9. Filter toggle button
# ---------------------------------------------------------------------------


class TestFilterToggleButton:
    def test_filter_chip_bar_reflects_active_filters(self, page: Page, base_url: str):
        """Slice 3, PR 3.2: the "+ Add filter" button text no longer carries an
        active-filter count — the always-visible chip bar does, one chip per
        active filter dimension, each independently removable."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        _navigate_to_review(page, base_url)
        toggle_btn = page.get_by_test_id("review-filter-toggle")
        chip_bar = page.get_by_test_id("review-filter-chip-bar")

        # Default state: no filters active, no chips.
        expect(toggle_btn).to_have_text("+ Add filter")
        expect(toggle_btn).not_to_have_class(re.compile(r"\bfilter-active\b"))
        expect(chip_bar.locator(".review-filter-chip")).to_have_count(0)

        # The Labels dropdown is in the toolbar — open it and check 'Cat'.
        labels_trigger = page.locator("#review-root .ms-dropdown[data-filter-key='labels-v2'] .ms-dropdown__trigger")
        labels_trigger.wait_for(state="visible", timeout=5000)
        labels_trigger.click()
        page.locator("#review-root .ms-dropdown[data-filter-key='labels-v2'] input[value='lbl-cat']").check()
        page.mouse.click(0, 0)  # close dropdown
        page.wait_for_timeout(300)

        expect(toggle_btn).to_have_text("+ Add filter")
        expect(toggle_btn).to_have_class(re.compile(r"\bfilter-active\b"))
        chips = chip_bar.locator(".review-filter-chip")
        expect(chips).to_have_count(1)
        expect(chips.first).to_contain_text("Cat")

        # The chip's own remove button clears the filter without opening the
        # popover editor.
        chips.first.locator(".review-filter-chip__remove").click()
        page.wait_for_timeout(300)
        expect(chip_bar.locator(".review-filter-chip")).to_have_count(0)
        expect(toggle_btn).not_to_have_class(re.compile(r"\bfilter-active\b"))


# ---------------------------------------------------------------------------
# 10. Saved Views (slice 5, PR 5.2 — replaces the localStorage preset store)
# ---------------------------------------------------------------------------

VIEW_CATS_ONLY = {
    "uuid": "view-1",
    "name": "Cats only",
    "description": "",
    "labelUuids": ["lbl-cat"],
    "filterJson": {
        "labelExcludeUuids": [],
        "labelMatchMode": "any",
        "labelIncludeChildren": False,
        "verificationCheck": None,
        "terms": [],
    },
    "promoted": False,
}


class TestSavedViews:
    def test_view_picker_applies_saved_filter_state(self, page: Page, base_url: str):
        """Picking a saved View from the dropdown restores its filter state
        into the panel — shown here as a label chip in the always-visible bar."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        page.route(
            lambda url: url.split("?")[0].endswith("/api/datasets") and "include_views=true" in url,
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "datasets": [VIEW_CATS_ONLY]}),
            ),
        )

        _navigate_to_review(page, base_url)
        views_select = page.get_by_test_id("review-views-select")
        expect(views_select).to_be_visible()
        # The dropdown populates from an async fetch that must not block the
        # rest of toolbar setup — wait for the option rather than assuming
        # it's there the instant the select renders.
        expect(views_select.locator("option", has_text="Cats only")).to_have_count(1)
        views_select.select_option(label="Cats only")
        page.wait_for_timeout(300)

        chips = page.get_by_test_id("review-filter-chip-bar").locator(".review-filter-chip")
        expect(chips).to_have_count(1)
        expect(chips.first).to_contain_text("Cat")

    def test_save_as_view_posts_promoted_false(self, page: Page, base_url: str):
        """Save as View must create an *unpromoted* dataset row — a View, not
        a trainable Dataset (decision register #10, one table + flag) — from
        whatever filters are currently active."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        create_calls: list[dict] = []

        def handle_datasets(route: Route):
            if route.request.method == "POST":
                create_calls.append(json.loads(route.request.post_data))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "dataset": {**VIEW_CATS_ONLY, "name": "My View"}}),
                )
            else:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "datasets": []}),
                )

        page.route(lambda url: url.split("?")[0].endswith("/api/datasets"), handle_datasets)

        _navigate_to_review(page, base_url)

        # Select 'Cat' via the Labels dropdown so the saved View isn't empty.
        labels_trigger = page.locator("#review-root .ms-dropdown[data-filter-key='labels-v2'] .ms-dropdown__trigger")
        labels_trigger.wait_for(state="visible", timeout=5000)
        labels_trigger.click()
        page.locator("#review-root .ms-dropdown[data-filter-key='labels-v2'] input[value='lbl-cat']").check()
        page.mouse.click(0, 0)
        page.wait_for_timeout(300)

        page.once("dialog", lambda dialog: dialog.accept("My View"))
        page.get_by_test_id("review-save-as-view").click()
        wait_until(page, lambda: create_calls, "Expected POST to /api/datasets")
        payload = create_calls[0]
        assert payload.get("name") == "My View"
        assert payload.get("promoted") is False
        assert payload.get("labelUuids") == ["lbl-cat"]


# ---------------------------------------------------------------------------
# 11. Canvas modal round trip
# ---------------------------------------------------------------------------


class TestCanvasModalRoundTrip:
    def test_double_click_thumbnail_opens_canvas_modal_and_delete_round_trips(self, page: Page, base_url: str):
        """Double-clicking a bbox thumbnail opens the canvas modal; deleting the
        highlighted bbox and saving marks the originating card as deleted —
        proving the onSaved round trip back into _refreshBboxCard/markCardStale."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)
        _mock_canvas_modal_image(page)

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")

        card = page.locator("#review-root [data-bbox-uuid='bbox-1']")
        card.locator("img.thumb").dblclick()

        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)
        expect(modal.get_by_test_id("canvas-edit-modal-title")).to_have_text("Edit: folder/image1.jpg")

        # Delete the highlighted (originally-opened) bbox, then Save & Close.
        # Assert the delete actually took (canvas went dirty, so Save lights up)
        # rather than sleeping and hoping — a no-op delete would otherwise sail
        # through to a no-op save and only fail at the very last assertion.
        modal.locator("button[data-toolbar-key='delete']").click()
        save_btn = modal.locator("button", has_text="Save & Close")
        expect(save_btn).to_have_class("save-btn--dirty")
        save_btn.click()

        # Don't blind-sleep then hard-assert: let Playwright's auto-retrying expect
        # poll for the modal's removal instead, matching the modal-open assertion's
        # own timeout above. A fixed sleep here either wastes time (fast path) or,
        # under rare heavy parallel-suite load, could fire the assertion before an
        # in-flight save-then-close finishes — auto-retry removes that race outright
        # rather than gambling on a longer fixed delay.
        expect(modal).to_have_count(0, timeout=8000)
        expect(card).to_have_class("review-card stale-deleted")
        expect(card.locator(".card-state-badge")).to_have_text("🗑 Deleted")
        expect(card.locator("input[type=checkbox]")).to_be_disabled()

    def test_delete_works_as_soon_as_the_toolbar_is_clickable(self, page: Page, base_url: str):
        """Regression: showCanvasModal appends the overlay (and, right after
        loadImage, the toolbar) well before it highlights the opened bbox —
        `_highlightBbox` used to run only after `createSidebarPanels` and
        `createInferencePanel` had both been awaited, and the latter awaits three
        cold model fetches. Throughout that window the Delete button is present
        and enabled, but `deleteSelected` is a silent no-op with nothing selected,
        so a fast click deleted nothing and the card never went stale.

        Delaying the model fetches widens that window deterministically. The delay
        is applied in-page rather than via `page.route` + `time.sleep`: the sync
        Playwright API runs route handlers on the same thread that drives the
        click, so a sleeping handler would stall the click too and hide the race."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)
        _mock_canvas_modal_image(page)

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")

        # Patched after the review panel has loaded so only the modal's own
        # (cold) model fetches are slowed.
        page.evaluate("""() => {
                const orig = window.fetch;
                window.fetch = (input, init) => {
                    const url = String(input?.url ?? input);
                    if (!/model/i.test(url)) return orig(input, init);
                    return new Promise(r => setTimeout(r, 1500)).then(() => orig(input, init));
                };
            }""")

        card = page.locator("#review-root [data-bbox-uuid='bbox-1']")
        card.locator("img.thumb").dblclick()

        modal = page.get_by_test_id("canvas-edit-modal")
        # Click Delete the instant Playwright considers the button actionable —
        # i.e. exactly what a fast machine (or a fast test) does.
        modal.locator("button[data-toolbar-key='delete']").click(timeout=8000)
        modal.locator("button", has_text="Save & Close").click()

        expect(modal).to_have_count(0, timeout=8000)
        expect(card).to_have_class("review-card stale-deleted")


# ---------------------------------------------------------------------------
# 11. Group highlight on hover
# ---------------------------------------------------------------------------


class TestGroupHighlight:
    def test_hovering_a_card_highlights_related_cards(self, page: Page, base_url: str):
        """Hovering a bbox card highlights itself and other cards sharing its image-uuid."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        _navigate_to_review(page, base_url)
        _show_all_bboxes(page)
        # Image cards are only visible in 'all' view (default is 'bboxes').
        _switch_view(page, "all")

        bbox_card = page.locator("#review-root [data-bbox-uuid='bbox-1']")
        image_card = page.locator("#review-root [data-image-uuid='img-1'][data-kind='image']")

        bbox_card.hover()
        page.wait_for_timeout(200)

        expect(bbox_card).to_have_class("review-card group-highlight-self-bbox")
        expect(image_card).to_have_class("review-card review-card--image group-highlight-image")

        # Move away — highlight should clear
        page.get_by_test_id("review-toolbar").hover()
        page.wait_for_timeout(200)
        expect(bbox_card).to_have_class("review-card")
        expect(image_card).to_have_class("review-card review-card--image")


# ---------------------------------------------------------------------------
# 12. List view (ui/entryView.js)
# ---------------------------------------------------------------------------


class TestListView:
    def test_toggle_to_list_view_renders_rows_and_persists(self, page: Page, base_url: str):
        """Switching the entryView toggle to list view renders `.review-row`
        elements (instead of grid `.review-card` cells), and the choice survives
        a reload via localStorage `review-view` — mirroring the file browser's
        `fb-view` persistence."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")

        # Explicitly in bboxes view, no tag filter → all 3 bbox cards visible.
        expect(page.locator("#review-root [data-testid='review-card']")).to_have_count(3)
        expect(page.locator("#review-root [data-testid='review-row']")).to_have_count(0)

        page.locator("#review-root .fb-view-toggle button[title='List view']").click()
        page.wait_for_timeout(200)

        expect(page.locator("#review-root [data-testid='review-row']")).to_have_count(3)
        expect(page.locator("#review-root .review-list")).to_have_count(1)
        expect(page.locator("#review-root .review-grid")).to_have_count(0)
        assert page.evaluate("() => localStorage.getItem('review-view')") == "list"

        # Reload — list view should be remembered; viewFilter resets to 'images' default → 2 rows.
        page.reload()
        page.wait_for_load_state("networkidle")
        page.locator(".main-tab-button[data-tab='tab-review']").click()
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("#review-root .review-list", timeout=4000)
        expect(page.locator("#review-root [data-testid='review-row']")).to_have_count(2)

    def test_list_view_selection_and_canvas_modal(self, page: Page, base_url: str):
        """In list view, checking a row's checkbox enables Bulk Apply (same
        `applySelect` wiring as grid cards), and double-clicking a row's
        thumbnail opens the canvas modal — proving `entryView.layout` queries
        and handlers work identically across both render paths."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)

        # Thumbnails are fetched as raw bytes via fetch().blob() — serve a tiny real PNG
        # so the <img class="thumb"> has nonzero size and can be double-clicked.
        png_bytes = base64.b64decode(TINY_PNG_B64)
        page.route(
            "**/thumbnail?*",
            lambda route: route.fulfill(status=200, content_type="image/png", body=png_bytes),
        )
        page.route(
            "**/api/images/*/file",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "imageBase64": TINY_PNG_B64,
                        "mimeType": "image/png",
                        "image": {"uuid": "img-1", "filename": "folder/image1.jpg", "bboxes": [BBOX_1, BBOX_2]},
                    }
                ),
            ),
        )

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")
        page.locator("#review-root .fb-view-toggle button[title='List view']").click()
        page.wait_for_timeout(200)

        rows = page.locator("#review-root [data-testid='review-row']")
        # Explicitly in bboxes view, no tag filter → all 3 bbox rows visible.
        expect(rows).to_have_count(3)

        bulk_btn = page.get_by_test_id("review-bulk-apply")
        expect(bulk_btn).to_have_text("Bulk Apply (0)")

        row = page.locator("#review-root [data-bbox-uuid='bbox-1']")
        row.locator("input[type=checkbox]").click()
        expect(bulk_btn).to_have_text("Bulk Apply (1)")
        expect(bulk_btn).to_be_enabled()

        row.locator("img.thumb").dblclick()
        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)
        expect(modal.get_by_test_id("canvas-edit-modal-title")).to_have_text("Edit: folder/image1.jpg")


# ---------------------------------------------------------------------------
# 11. "Config Ready" (V2) — image-view-only gating in the live review panel
# ---------------------------------------------------------------------------
#
# filters.js unit tests (test_filter_panel.py) cover the control's own
# disabled/hint mechanism in isolation. These tests confirm main.js actually
# drives it: setConfigReadyAvailable(false) on switching to bboxes view, and
# back to enabled (assuming an active config) on switching away — following
# the same _navigate_to_review / _switch_view helpers used elsewhere in this file.


class TestConfigReadyViewGating:
    def test_disabled_in_bboxes_view(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")

        # Excludes .bbox-verification-select — the bbox tri-state control (V1)
        # shares the base .filter-verification-select class name.
        select = page.locator("#review-root .filter-verification-select:not(.bbox-verification-select)")
        expect(select).to_be_disabled()

    def test_reenabled_switching_back_to_images_view(self, page: Page, base_url: str):
        """Without an active training config the control stays disabled for that
        reason regardless of view — so this drives an active config first (same
        activeTrainingConfig singleton used by test_filter_panel.py) to isolate
        the view-gating behavior specifically."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        page.route(
            lambda url: url.split("?")[0].endswith("/api/model-configs"),
            lambda route: (
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "status": "success",
                            "configs": [
                                {
                                    "uuid": "mc-1",
                                    "name": "My Model",
                                    "description": "",
                                    "nodes": [
                                        {
                                            "uuid": "node-cat",
                                            "labelUuid": "lbl-cat",
                                            "modelType": "object_detection",
                                            "architecture": "yolo_v9_c",
                                        },
                                    ],
                                },
                            ],
                        }
                    ),
                )
                if route.request.method == "GET"
                else route.fallback()
            ),
        )
        page.route(
            "**/api/training-configs/tc-1/label-stats",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "labelStats": [], "totalImages": 0}),
            ),
        )

        _navigate_to_review(page, base_url)
        page.evaluate("""async () => {
            const { activeTrainingConfig } = await import('/app-static/js/shared/activeTrainingConfig.js');
            await activeTrainingConfig.setActive({
                uuid: 'tc-1', name: 'Cat Detector', description: '', datasetUuids: [],
                modelConfigNodeUuid: 'node-cat', hyperparamsJson: {},
                createdAt: '2026-01-01T00:00:00', updatedAt: '2026-01-01T00:00:00',
            });
        }""")

        # Excludes .bbox-verification-select — the bbox tri-state control (V1)
        # shares the base .filter-verification-select class name.
        select = page.locator("#review-root .filter-verification-select:not(.bbox-verification-select)")
        expect(select).to_be_enabled()

        _switch_view(page, "bboxes")
        expect(select).to_be_disabled()

        _switch_view(page, "images")
        expect(select).to_be_enabled()

        # Cleanup: clear the singleton so it doesn't leak into later tests.
        page.evaluate("""async () => {
            const { activeTrainingConfig } = await import('/app-static/js/shared/activeTrainingConfig.js');
            await activeTrainingConfig.setActive(null);
        }""")

    def test_verification_check_and_label_uuids_omitted_in_bboxes_view(self, page: Page, base_url: str):
        """Gate the request itself, not just the control: in bboxes view,
        verificationCheck/verificationLabelUuids must not be sent even if a
        Config Ready selection + active config are already in place from a
        prior images-view visit."""
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        page.route(
            lambda url: url.split("?")[0].endswith("/api/model-configs"),
            lambda route: (
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "status": "success",
                            "configs": [
                                {
                                    "uuid": "mc-1",
                                    "name": "My Model",
                                    "description": "",
                                    "nodes": [
                                        {
                                            "uuid": "node-cat",
                                            "labelUuid": "lbl-cat",
                                            "modelType": "object_detection",
                                            "architecture": "yolo_v9_c",
                                        },
                                    ],
                                },
                            ],
                        }
                    ),
                )
                if route.request.method == "GET"
                else route.fallback()
            ),
        )
        page.route(
            "**/api/training-configs/tc-1/label-stats",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "labelStats": [], "totalImages": 0}),
            ),
        )

        captured = {}

        def handle(route: Route):
            captured["body"] = route.request.post_data_json
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "images": PAGE_1_IMAGES, "hasMore": False, "totalCount": 2}),
            )

        page.route("**/api/images/review", handle)

        _navigate_to_review(page, base_url)
        page.evaluate("""async () => {
            const { activeTrainingConfig } = await import('/app-static/js/shared/activeTrainingConfig.js');
            await activeTrainingConfig.setActive({
                uuid: 'tc-1', name: 'Cat Detector', description: '', datasetUuids: [],
                modelConfigNodeUuid: 'node-cat', hyperparamsJson: {},
                createdAt: '2026-01-01T00:00:00', updatedAt: '2026-01-01T00:00:00',
            });
        }""")

        # The filter panel starts hidden — open it via the ☰ Filters toggle
        # before interacting with the Config Ready select.
        page.get_by_test_id("review-filter-toggle").click()

        # Excludes .bbox-verification-select — the bbox tri-state control (V1)
        # shares the base .filter-verification-select class name.
        select = page.locator("#review-root .filter-verification-select:not(.bbox-verification-select)")
        select.select_option("all_verified")
        page.wait_for_timeout(300)
        assert {
            "kind": "labelVerification",
            "check": "all_verified",
            "labelUuids": ["lbl-cat"],
        } in captured[
            "body"
        ].get("terms", [])

        _switch_view(page, "bboxes")
        page.wait_for_timeout(300)
        assert not any(t.get("kind") == "labelVerification" for t in captured["body"].get("terms", []))

        # Cleanup.
        page.evaluate("""async () => {
            const { activeTrainingConfig } = await import('/app-static/js/shared/activeTrainingConfig.js');
            await activeTrainingConfig.setActive(null);
        }""")
