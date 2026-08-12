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
import urllib.parse

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
        """The verify chip reads '✓ Verified' for verified bboxes and '? Unverified' otherwise."""
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

        expect(unverified_chip).to_have_text("? Unverified")
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
        expect(chip).to_have_text("? Unverified")

        # Re-query after each toggle: setVerification replaces the chip element.
        chip.dblclick()
        chip = page.locator("#review-root [data-bbox-uuid='bbox-1'] .verify-chip")
        expect(chip).to_have_text("✓ Verified")
        expect(chip).to_have_class("verify-chip verify-chip--verified")


class TestImageCardLabelChipVerifyMarks:
    """Image-card label chips carry a ✓/? verification marker per label."""

    def _setup(self, page: Page, base_url: str, images: list):
        _mock_review_queue(page, images)
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
        _switch_view(page, "images")

    def test_chip_marks_reflect_per_label_verification(self, page: Page, base_url: str):
        """img-1 holds an unverified Cat and a verified Dog — each chip marks its own state."""
        self._setup(page, base_url, PAGE_1_IMAGES)

        chips = page.locator("#review-root [data-image-uuid='img-1'][data-kind='image'] .label-chip")
        expect(chips).to_have_count(2)
        expect(chips.nth(0)).to_have_text("Cat?")
        expect(chips.nth(0).locator(".label-chip__verify--unverified")).to_have_count(1)
        expect(chips.nth(1)).to_have_text("Dog✓")
        expect(chips.nth(1).locator(".label-chip__verify--verified")).to_have_count(1)

    def test_whole_image_chip_suppressed_when_a_bbox_shows_the_label(self, page: Page, base_url: str):
        """Image-label presence is derived from verified bboxes, so a label with a box
        on the image must not also render a dashed whole-image chip. Only image labels
        with no box of their own stay dashed."""
        page.route(
            "**/api/images/labels/batch",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        # Cat has boxes on img-1 → duplicate. Dog has none → genuine image label.
                        "imageLabels": {
                            "img-1": [
                                {"labelUuid": "lbl-cat", "status": "present"},
                                {"labelUuid": "lbl-dog", "status": "present"},
                            ]
                        },
                    }
                ),
            ),
        )
        cat_only_image = {"uuid": "img-1", "filename": "folder/image1.jpg", "bboxes": [BBOX_1]}
        self._setup(page, base_url, [cat_only_image])

        card = page.locator("#review-root [data-image-uuid='img-1'][data-kind='image']")
        expect(card.locator(".label-chip")).to_have_text(["Cat?"])
        expect(card.locator(".image-label-chip")).to_have_text(["Dog"])

    def test_mixed_label_shows_both_counts(self, page: Page, base_url: str):
        """Two Cat boxes with differing verification collapse to one chip reading '✓1 ?1'."""
        verified_cat = {**BBOX_1, "uuid": "bbox-cat-verified", "tags": [VERIFIED_TAG]}
        mixed_image = {"uuid": "img-mixed", "filename": "folder/mixed.jpg", "bboxes": [BBOX_1, verified_cat]}
        self._setup(page, base_url, [mixed_image])

        chip = page.locator("#review-root [data-image-uuid='img-mixed'][data-kind='image'] .label-chip")
        expect(chip).to_have_count(1)
        expect(chip).to_have_text("Cat✓1 ?1")

    def test_double_click_mark_verifies_every_box_behind_the_chip(self, page: Page, base_url: str):
        """A mixed chip resolves upward in one gesture: both Cat boxes go Verified."""
        _mock_misc_review_endpoints(page)
        verified_cat = {**BBOX_1, "uuid": "bbox-cat-verified", "tags": [VERIFIED_TAG]}
        mixed_image = {"uuid": "img-mixed", "filename": "folder/mixed.jpg", "bboxes": [BBOX_1, verified_cat]}
        self._setup(page, base_url, [mixed_image])

        card = page.locator("#review-root [data-image-uuid='img-mixed'][data-kind='image']")
        card.locator(".label-chip__verify").dblclick()
        expect(card.locator(".label-chip")).to_have_text(["Cat✓"])
        # The card is flagged as touched on this page.
        expect(card.locator(".card-state-badge.state-verified")).to_have_count(1)

        # Fully verified → the next double-click sends them all back to Unverified.
        card.locator(".label-chip__verify").dblclick()
        expect(card.locator(".label-chip")).to_have_text(["Cat?"])


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
        expect(bulk_btn).to_have_text("Apply to Selected (0)")
        expect(bulk_btn).to_be_disabled()

        page.locator("#review-root [data-bbox-uuid='bbox-1'] input[type=checkbox]").click()

        expect(bulk_btn).to_have_text("Apply to Selected (1)")
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
        expect(bulk_btn).to_have_text("Apply to Selected (3)")


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
        expect(bulk_btn).to_have_text("Apply to Selected (3)")

        # Unchecking clears the selection again
        select_all.uncheck()
        for i in range(3):
            expect(cards.nth(i).locator("input[type=checkbox]")).not_to_be_checked()
        expect(bulk_btn).to_have_text("Apply to Selected (0)")


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


def _record_status_and_unload_guard(page: Page):
    """Record every status the toolbar shows, plus beforeunload add/remove.

    Both are transient — the status is overwritten as soon as the run lands and
    the guard is removed in a `finally` — so polling from the test client is
    hopelessly racy; the page records them as they happen instead.
    """
    page.add_init_script("""
        window.__reviewStatuses = [];
        window.__unloadGuardLog = [];
        const _add = window.addEventListener.bind(window);
        const _remove = window.removeEventListener.bind(window);
        window.addEventListener = (type, fn, opts) => {
          if (type === 'beforeunload') window.__unloadGuardLog.push('add');
          return _add(type, fn, opts);
        };
        window.removeEventListener = (type, fn, opts) => {
          if (type === 'beforeunload') window.__unloadGuardLog.push('remove');
          return _remove(type, fn, opts);
        };
        // Init scripts run before the document has a body, so arm the
        // observer once there is something to observe.
        document.addEventListener('DOMContentLoaded', () => {
          new MutationObserver(() => {
            const el = document.querySelector("[data-testid='review-status']");
            if (el) window.__reviewStatuses.push(`${el.className}|${el.textContent}`);
          }).observe(document.body, { subtree: true, childList: true, characterData: true });
        });
        """)


class TestPerItemApplyAsksYouToWait:
    """The per-item counterpart of TestBulkApplyAllStatusSaysItRunsOnTheServer.

    "Apply to Selected" on bboxes is a loop of one PATCH per bbox issued from
    the tab, so leaving really does abandon the rest — its status must read
    differently from the server-side runs, count progress, and hold a
    beforeunload guard for as long as the loop is in flight.
    """

    def _select_two_bboxes_and_relabel(self, page: Page, base_url: str):
        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")
        page.locator("#review-root [data-bbox-uuid='bbox-1'] input[type=checkbox]").click()
        page.locator("#review-root [data-bbox-uuid='bbox-3'] input[type=checkbox]").click()
        page.get_by_test_id("review-bulk-apply").click()

        modal_locator = page.locator("div:has(> strong:text-is('Apply to 2 selected bboxes'))")
        picker_input = modal_locator.locator("input[type=text]")
        picker_input.click()
        picker_input.fill("Dog")
        modal_locator.get_by_text("Dog", exact=True).click()
        page.locator("div:has(> strong:text-is('Apply to 2 selected bboxes')) button", has_text="Apply").click()

    def test_status_says_please_wait_and_counts_progress(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)
        _record_status_and_unload_guard(page)

        self._select_two_bboxes_and_relabel(page, base_url)

        status = page.get_by_test_id("review-status")
        expect(status).not_to_have_class(re.compile(r"--loading"), timeout=6000)

        seen = page.evaluate("window.__reviewStatuses")
        applying = [s for s in seen if "Applying to" in s]
        assert applying, f"No in-flight status was shown: {seen}"
        assert all("please wait" in s for s in applying), applying
        # Must never claim the server will finish it — nothing else will.
        assert not any("you can leave this page" in s for s in applying), applying
        # Progress is visible, not an opaque pulse.
        assert any("of 2 bboxes" in s for s in applying), applying
        assert any("Applying to 2 of 2 bboxes" in s for s in applying), applying

    def test_unload_is_guarded_while_the_loop_runs_and_released_after(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)
        _record_status_and_unload_guard(page)

        self._select_two_bboxes_and_relabel(page, base_url)
        expect(page.get_by_test_id("review-status")).not_to_have_class(re.compile(r"--loading"), timeout=6000)

        log = page.evaluate("window.__unloadGuardLog")
        assert log == ["add", "remove"], log


class TestBulkApplyAllStatusSaysItRunsOnTheServer:
    """The pulsing "Applying…" status reads as "don't touch anything", but
    Apply to All is one request the server runs to completion — the tab is only
    waiting for the reply. The in-flight message must say the page can be left,
    so nobody sits watching a run over tens of thousands of rows.
    """

    def test_status_says_the_page_can_be_left_while_applying(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)

        page.route(
            "**/api/images/bbox/bulk-apply",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "modifiedCount": 1, "failCount": 0}),
            ),
        )

        # The in-flight message is replaced by the result (and then by render()'s
        # own count) as soon as the request lands, which is far too fast to catch
        # by polling from here — record every status the element ever shows.
        page.add_init_script("""
            window.__reviewStatuses = [];
            // Init scripts run before the document has a body, so arm the
            // observer once there is something to observe.
            document.addEventListener('DOMContentLoaded', () => {
              new MutationObserver(() => {
                const el = document.querySelector("[data-testid='review-status']");
                if (el) window.__reviewStatuses.push(`${el.className}|${el.textContent}`);
              }).observe(document.body, { subtree: true, childList: true, characterData: true });
            });
            """)

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")

        page.get_by_test_id("review-bulk-apply-all").click()
        page.locator("input[type=radio][name='bulk-all-action'][value='markVerified']").click()
        page.locator("button", has_text="Apply to All").last.click()

        status = page.get_by_test_id("review-status")
        # The result message is itself replaced by render()'s page count moments
        # later, so wait on the busy state clearing rather than on any one text.
        expect(status).not_to_have_class(re.compile(r"--loading"), timeout=6000)

        seen = page.evaluate("window.__reviewStatuses")
        applying = [s for s in seen if "Applying to all" in s]
        assert applying, f"No in-flight 'Applying to all' status was shown: {seen}"
        assert all("you can leave this page" in s for s in applying), applying
        # It is the busy/pulsing message that carries the hint, not a leftover.
        assert all("review-toolbar__status--loading" in s for s in applying), applying

        # Once it lands, the hint goes away with the rest of the busy state.
        expect(status).not_to_contain_text("you can leave this page")


class TestFilenameDateNegateReachesTheGridQuery:
    """_baseReviewFilterOpts is the single hand-picked field list every real
    grid query goes through (main render, bulk-apply-all, bulk-apply-images,
    "See results", etc.) — see TestBulkApplyAllScope above for the same class
    of bug (a filter dimension added elsewhere but never forwarded here). The
    negate checkboxes update filterPanel's own state and chip correctly, but
    _baseReviewFilterOpts silently dropped filenameNegate/dateNegate, so the
    actual /api/images/review request never carried them — the checkbox
    looked like it worked but the search results were unaffected."""

    def test_filename_negate_checkbox_reaches_the_actual_search_request(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        calls = []

        def handle(route: Route):
            calls.append(json.loads(route.request.post_data or "{}"))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "images": [], "hasMore": False, "totalCount": 0}),
            )

        page.route("**/api/images/review", handle)

        _navigate_to_review(page, base_url)
        calls.clear()  # drop the initial default-filter load

        page.get_by_test_id("review-filter-toggle").click()
        page.locator("#review-root .review-filter-panel input[placeholder='substring…']").fill("closet")
        page.locator("#review-root .filter-filename-negate input[type=checkbox]").check()
        page.wait_for_timeout(500)  # filenameSearch is debounced 400ms

        wait_until(page, lambda: calls, lambda: f"Expected POST to /api/images/review, got {calls}")
        terms = calls[-1].get("terms", [])
        filename_terms = [t for t in terms if t.get("kind") == "filename"]
        assert filename_terms == [{"kind": "filename", "patterns": ["closet"], "negate": True}], calls[-1]

    def test_date_negate_checkbox_reaches_the_actual_search_request(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        calls = []

        def handle(route: Route):
            calls.append(json.loads(route.request.post_data or "{}"))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "images": [], "hasMore": False, "totalCount": 0}),
            )

        page.route("**/api/images/review", handle)

        _navigate_to_review(page, base_url)
        calls.clear()

        page.get_by_test_id("review-filter-toggle").click()
        page.locator("#review-root .review-filter-panel input[type=datetime-local]").first.fill("2026-01-01T00:00")
        page.locator("#review-root .filter-date-negate input[type=checkbox]").check()
        page.wait_for_timeout(200)

        wait_until(page, lambda: calls, lambda: f"Expected POST to /api/images/review, got {calls}")
        terms = calls[-1].get("terms", [])
        date_terms = [t for t in terms if t.get("kind") == "date"]
        assert len(date_terms) == 1, calls[-1]
        assert date_terms[0].get("negate") is True, calls[-1]


class TestConfidenceBandReachesTheGridQuery:
    """Same class of bug as TestFilenameDateNegateReachesTheGridQuery above: the
    confidence band is a real filter dimension, so it has to survive the trip
    from the panel control through _baseReviewFilterOpts into the actual
    /api/images/review body — and, for Apply to All, into the bbox-level field
    that stops the action hitting out-of-band boxes on matching images."""

    def _fill_band(self, page: Page, min_pct: str, max_pct: str):
        inputs = page.locator("#review-root .review-filter-panel .confidence-range-input")
        inputs.nth(0).fill(min_pct)
        inputs.nth(0).dispatch_event("change")
        inputs.nth(1).fill(max_pct)
        inputs.nth(1).dispatch_event("change")

    def test_band_reaches_the_actual_search_request(self, page: Page, base_url: str):
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        calls = []

        def handle(route: Route):
            calls.append(json.loads(route.request.post_data or "{}"))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "images": [], "hasMore": False, "totalCount": 0}),
            )

        page.route("**/api/images/review", handle)

        _navigate_to_review(page, base_url)
        calls.clear()

        page.get_by_test_id("review-filter-toggle").click()
        self._fill_band(page, "20", "60")
        page.wait_for_timeout(200)

        wait_until(page, lambda: calls, lambda: f"Expected POST to /api/images/review, got {calls}")
        terms = calls[-1].get("terms", [])
        conf_terms = [t for t in terms if t.get("kind") == "bboxConfidence"]
        # Percent in the input, 0–1 on the wire — the unit the metadata stores.
        assert conf_terms == [{"kind": "bboxConfidence", "min": 0.2, "max": 0.6}], calls[-1]

    def test_apply_to_all_carries_the_band_as_a_bbox_level_field(self, page: Page, base_url: str):
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

        page.get_by_test_id("review-filter-toggle").click()
        self._fill_band(page, "", "50")
        page.get_by_test_id("review-filter-toggle").click()  # close the popover

        page.get_by_test_id("review-bulk-apply-all").click()
        page.locator("input[type=radio][name='bulk-all-action'][value='markVerified']").click()
        page.locator("button", has_text="Apply to All").last.click()

        wait_until(page, lambda: calls, lambda: f"Expected POST to bulk-apply, got {calls}")
        body = calls[0]
        # Bbox-level: without this the action reaches the high-confidence boxes
        # sharing those images, which the grid does not show.
        assert body.get("bboxConfidence") == {"min": None, "max": 0.5, "includeMissing": False}, body
        # Image-level: the same band as a filter term, so paging skips images
        # with no in-band bbox at all.
        assert {"kind": "bboxConfidence", "max": 0.5} in body["filter"]["terms"], body


class TestBulkAllButtonViewGating:
    """The bbox bulk button appears only in a view that shows bboxes — it applies
    bbox-level actions no matter what's on screen, which is exactly the confusion
    the gating removes. The image one is offered everywhere: in bboxes view it
    acts on the images CONTAINING the matched bboxes, and says so."""

    def _setup(self, page: Page, base_url: str, bbox_count: int = 7):
        _mock_review_queue(page, PAGE_1_IMAGES, total_count=2)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [])
        _mock_misc_review_endpoints(page)
        page.route(
            "**/api/images/bbox/count",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "count": bbox_count}),
            ),
        )
        _navigate_to_review(page, base_url)

    def test_bbox_button_shows_a_real_bbox_count_not_the_image_count(self, page: Page, base_url: str):
        """The review queue's totalCount counts images (2 here); the bbox button
        must show the separate bbox count (7) or it understates its own scope."""
        self._setup(page, base_url)
        _switch_view(page, "bboxes")

        expect(page.get_by_test_id("review-bulk-apply-all")).to_have_text("Apply to All Filtered Bboxes (7)")

    def test_bboxes_view_offers_both_buttons_with_a_containing_images_label(self, page: Page, base_url: str):
        """The image button stays available in bboxes view (acting on the images
        behind the matched bboxes), but must never read as if it acts on the
        bboxes themselves — hence the distinct label and the image count (2),
        not the bbox count (7)."""
        self._setup(page, base_url)
        _switch_view(page, "bboxes")

        expect(page.get_by_test_id("review-bulk-apply-all")).to_be_visible()
        images_btn = page.get_by_test_id("review-bulk-apply-filtered-images")
        expect(images_btn).to_be_visible()
        expect(images_btn).to_have_text("Apply to Images Containing Filtered Bboxes (2)")

    def test_only_the_images_button_shows_in_images_view(self, page: Page, base_url: str):
        self._setup(page, base_url)
        _switch_view(page, "images")

        expect(page.get_by_test_id("review-bulk-apply-filtered-images")).to_be_visible()
        expect(page.get_by_test_id("review-bulk-apply-all")).to_be_hidden()

    def test_both_buttons_show_in_all_view(self, page: Page, base_url: str):
        self._setup(page, base_url)
        _switch_view(page, "all")

        expect(page.get_by_test_id("review-bulk-apply-all")).to_be_visible()
        expect(page.get_by_test_id("review-bulk-apply-filtered-images")).to_be_visible()


class TestApplyToFilteredImagesScope:
    """ "Apply to Filtered Images" must say how many images it will hit and
    show the same first-3/last-3 range preview as "Apply to All"."""

    def test_button_shows_the_filtered_image_count(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES, total_count=1234)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [])
        _mock_misc_review_endpoints(page)

        _navigate_to_review(page, base_url)

        expect(page.get_by_test_id("review-bulk-apply-filtered-images")).to_have_text(
            "Apply to Filtered Images (1,234)"
        )

    def test_modal_shows_count_and_first_last_range_preview(self, page: Page, base_url: str):
        """The reverse-sorted query supplies the "Last" group, so the modal
        shows both ends of the filtered range, not just the open page."""
        png_bytes = base64.b64decode(TINY_PNG_B64)
        page.route(
            "**/thumbnail?*",
            lambda route: route.fulfill(status=200, content_type="image/png", body=png_bytes),
        )

        def handle_queue(route: Route):
            body = json.loads(route.request.post_data or "{}")
            # Default sort order is 'desc'; the preview's tail query flips it.
            images = PAGE_2_IMAGES if body.get("sortOrder") == "asc" else PAGE_1_IMAGES
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "images": images, "hasMore": False, "totalCount": 3}),
            )

        page.route("**/api/images/review", handle_queue)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [])
        _mock_misc_review_endpoints(page)

        _navigate_to_review(page, base_url)
        page.get_by_test_id("review-bulk-apply-filtered-images").click()

        dialog = page.locator("div:has(> strong:text-is('Apply to All 3 Filtered Images'))").last
        expect(dialog).to_be_visible()

        preview = dialog.locator(".bulk-modal-preview-row")
        expect(preview.locator(".image-compare-group-label", has_text="First")).to_be_visible()
        expect(preview.locator(".image-compare-group-label", has_text="Last")).to_be_visible()
        # 2 images from the head query + 1 distinct image from the tail query.
        expect(preview.locator(".image-compare-thumb")).to_have_count(3)


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
        modal_locator.locator(".bulk-modal-picker--labels input[type=checkbox][value='lbl-cat']").check()
        modal_locator.locator("button", has_text="Apply").click()
        wait_until(page, lambda: len(calls) >= 1, lambda: f"Expected 1 batched bulk-apply-by-uuids call, got {calls}")

        assert len(calls) == 1, f"Expected 1 batched bulk-apply-by-uuids call, got {calls}"
        call = calls[0]
        assert call["action"] == "setAbsent"
        assert call["labelUuids"] == ["lbl-cat"]
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

        # Explicitly select 'Mark label Absent' and check Cat in the label picker.
        page.locator("input[type=radio][value=setAbsent]").click()
        page.locator(".bulk-modal-picker--labels input[type=checkbox][value='lbl-cat']").check()
        page.locator("div:has(> strong:text-is('Apply to 2 selected images')) button", has_text="Apply").click()
        wait_until(page, lambda: len(calls) >= 1, lambda: f"Expected 1 batched bulk-apply-by-uuids call, got {calls}")

        assert len(calls) == 1, f"Expected 1 batched bulk-apply-by-uuids call, got {calls}"
        call = calls[0]
        assert call["action"] == "setAbsent"
        assert call["labelUuids"] == ["lbl-cat"]
        assert sorted(call["imageUuids"]) == ["img-1", "img-2"]

    def test_remove_from_db_hits_the_db_only_endpoint_not_the_destructive_delete(self, page: Page, base_url: str):
        """'Remove Images from Database' must go to DELETE /api/images/<uuid>/db-only.
        Falling through to the plain DELETE would silently erase the files the
        user explicitly chose to keep."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)

        db_only_calls = []
        hard_delete_calls = []

        def handle_db_only(route: Route):
            db_only_calls.append(route.request.url)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "imageUuid": "x", "filename": "x.png"}),
            )

        def handle_hard_delete(route: Route):
            hard_delete_calls.append(route.request.url)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "imageUuid": "x", "filename": "x.png"}),
            )

        page.route("**/api/images/*/db-only", handle_db_only)
        page.route(
            "**/api/images/*",
            lambda route: (handle_hard_delete(route) if route.request.method == "DELETE" else route.fallback()),
        )

        _navigate_to_review(page, base_url)
        _switch_view(page, "images")

        cards = page.locator("#review-root [data-testid='review-card']")
        expect(cards).to_have_count(2)
        for i in range(2):
            cards.nth(i).locator("input[type=checkbox]").click()

        page.get_by_test_id("review-bulk-apply").click()
        page.locator("input[type=radio][value=removeFromDb]").click()
        page.locator("div:has(> strong:text-is('Apply to 2 selected images')) button", has_text="Apply").click()

        # Confirmation spells out that the files stay on disk.
        confirm = page.locator("div", has_text="the image files stay on disk").last
        expect(confirm).to_be_visible()
        page.locator("button", has_text="Remove from DB").last.click()

        wait_until(
            page,
            lambda: len(db_only_calls) >= 2,
            lambda: f"Expected 2 db-only deletes, got {db_only_calls}",
        )
        assert sorted(u.rsplit("/", 2)[-2] for u in db_only_calls) == ["img-1", "img-2"]
        assert hard_delete_calls == [], f"must not hit the file-deleting endpoint: {hard_delete_calls}"


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


class TestFilterPopoverDismissal:
    """Pressing "+ Add filter" again was the only way to close the popover, and
    nothing on screen said so. ESC, an outside click and an explicit ✕ all close
    it now, and the toggle reports its state via aria-expanded."""

    def _open(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _navigate_to_review(page, base_url)

        toggle = page.get_by_test_id("review-filter-toggle")
        popover = page.locator("#review-root .review-filter-popover")
        expect(toggle).to_have_attribute("aria-expanded", "false")
        expect(popover).to_be_hidden()

        toggle.click()
        expect(popover).to_be_visible()
        expect(toggle).to_have_attribute("aria-expanded", "true")
        return toggle, popover

    def test_escape_closes_the_popover(self, page: Page, base_url: str):
        toggle, popover = self._open(page, base_url)
        page.keyboard.press("Escape")
        expect(popover).to_be_hidden()
        expect(toggle).to_have_attribute("aria-expanded", "false")

    def test_outside_click_closes_the_popover(self, page: Page, base_url: str):
        toggle, popover = self._open(page, base_url)
        page.mouse.click(5, 5)
        expect(popover).to_be_hidden()
        expect(toggle).to_have_attribute("aria-expanded", "false")

    def test_close_button_closes_the_popover(self, page: Page, base_url: str):
        toggle, popover = self._open(page, base_url)
        page.get_by_test_id("review-filter-popover-close").click()
        expect(popover).to_be_hidden()
        expect(toggle).to_have_attribute("aria-expanded", "false")

    def test_clicking_inside_keeps_the_popover_open(self, page: Page, base_url: str):
        """The outside-click listener must not fire for the panel's own controls,
        or every filter edit would dismiss the editor mid-use."""
        _toggle, popover = self._open(page, base_url)
        popover.locator(".bbox-verification-select").click()
        expect(popover).to_be_visible()

    def test_toggle_button_still_closes_it(self, page: Page, base_url: str):
        """The original affordance must keep working: the outside-click handler
        ignores the toggle so it doesn't close-then-reopen and look dead."""
        toggle, popover = self._open(page, base_url)
        toggle.click()
        expect(popover).to_be_hidden()
        expect(toggle).to_have_attribute("aria-expanded", "false")


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
        save_btn = modal.get_by_test_id("canvas-edit-modal-save-close")
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

    def test_double_click_opens_canvas_for_an_image_with_no_bboxes(self, page: Page, base_url: str):
        """Regression: the image-card double-click handler bailed out unless the
        image already had a bbox, so every un-annotated image in a search result —
        exactly the ones a reviewer needs to open and draw on — did nothing at all."""
        boxless = {"uuid": "img-boxless", "filename": "folder/boxless.jpg", "bboxes": []}
        _mock_review_queue(page, [boxless])
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)
        _mock_canvas_modal_image(page)

        _navigate_to_review(page, base_url)
        _switch_view(page, "images")

        card = page.locator("#review-root [data-image-uuid='img-boxless'][data-kind='image']")
        expect(card).to_have_count(1)
        card.locator("img.thumb").dblclick()

        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)
        expect(modal.get_by_test_id("canvas-edit-modal-title")).to_have_text("Edit: folder/boxless.jpg")

    def test_save_from_image_card_updates_that_card(self, page: Page, base_url: str):
        """Regression: opening the canvas from an IMAGE card and hitting Save & Close
        left the card showing its pre-save boxes. IMAGE_ANNOTATIONS_SAVED fired and
        the panel heard it, but the card's refresher only redrew the bbox overlay —
        and did so from the stale in-memory `img.bboxes`, so the count badge never
        moved. Drawing a new box on img-1 and saving must bump the badge to
        '3 boxes' (drawn, not deleted — see TestImageInitiatedOpenHasNoPreselection
        for why this no longer uses the toolbar Delete button)."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(
            page,
            [
                {**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"},
                {**VERIFIED_TAG, "name": "Verified", "color": "#27ae60", "kind": "system"},
            ],
        )
        _mock_misc_review_endpoints(page)
        _mock_canvas_modal_image(page)

        _navigate_to_review(page, base_url)
        _show_all_bboxes(page)
        _switch_view(page, "images")

        card = page.locator("#review-root [data-image-uuid='img-1'][data-kind='image']")
        expect(card.locator(".bbox-count-badge")).to_have_text("2 boxes")

        card.locator("img.thumb").dblclick()
        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)

        # Draw a new box — any rectangle inside the canvas works, its exact
        # placement isn't what this test is about.
        modal.locator("button[data-toolbar-key='bbox']").click()
        canvas_box = modal.locator(".canvas-modal-canvas").bounding_box()
        x0 = canvas_box["x"] + canvas_box["width"] * 0.2
        y0 = canvas_box["y"] + canvas_box["height"] * 0.2
        x1 = canvas_box["x"] + canvas_box["width"] * 0.6
        y1 = canvas_box["y"] + canvas_box["height"] * 0.6
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move(x1, y1)
        page.mouse.up()

        save_btn = modal.get_by_test_id("canvas-edit-modal-save-close")
        expect(save_btn).to_have_class("save-btn--dirty")
        save_btn.click()
        expect(modal).to_have_count(0, timeout=8000)

        # The card reflects the save without a re-fetch.
        expect(card.locator(".bbox-count-badge")).to_have_text("3 boxes")

    def test_image_initiated_open_has_no_preselected_box(self, page: Page, base_url: str):
        """Opening the canvas by double-clicking an IMAGE card (not a specific
        bbox's own card) used to pass img.bboxes[0] straight into the modal as
        its "originally opened" bbox, which highlighted it with transformer
        handles on open — as if the user had clicked it. Nothing should be
        selected until the user actually selects something; the toolbar
        Delete button (a no-op with nothing selected) proves it."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)
        _mock_canvas_modal_image(page)

        _navigate_to_review(page, base_url)
        _show_all_bboxes(page)
        _switch_view(page, "images")

        card = page.locator("#review-root [data-image-uuid='img-1'][data-kind='image']")
        expect(card.locator(".bbox-count-badge")).to_have_text("2 boxes")

        card.locator("img.thumb").dblclick()
        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)
        expect(modal.locator("button[data-toolbar-key='delete']")).to_be_enabled(timeout=8000)

        modal.locator("button[data-toolbar-key='delete']").click()
        page.wait_for_timeout(300)

        save_btn = modal.get_by_test_id("canvas-edit-modal-save-close")
        expect(save_btn).not_to_have_class(re.compile(r"\bsave-btn--dirty\b"))

    def test_arrow_keys_do_not_scroll_the_background_grid(self, page: Page, base_url: str):
        """Reported bug: with the canvas modal open, pressing ArrowDown scrolled
        the review grid behind it.

        Trapping focus inside the dialog (trapModalFocus) stops the browser
        from scrolling some OTHER focused/scrollable element, but arrow keys /
        Space / Page Up-Down still fall through to their *default* action of
        scrolling the viewport whenever nothing else handles the key — that
        fallback doesn't care what has focus, only whether the key was
        consumed. A dialog full of buttons never consumes it. The fix locks
        html/body overflow while any dialog is open (acquireModalLayer in
        modalZIndex.js), which this asserts against real page-level scroll,
        not a focus check — that's what let the first (focus-only) fix look
        complete while the bug was still fully reproducible."""
        many_images = [{"uuid": f"img-{i}", "filename": f"folder/image{i}.jpg", "bboxes": []} for i in range(1, 60)]
        many_images[0] = PAGE_1_IMAGES[0]  # img-1 keeps its real bboxes for _mock_canvas_modal_image
        _mock_review_queue(page, many_images)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)
        _mock_canvas_modal_image(page)

        _navigate_to_review(page, base_url)
        _switch_view(page, "images")

        assert page.evaluate(
            "() => document.documentElement.scrollHeight > window.innerHeight"
        ), "page must actually be tall enough to scroll for this test to mean anything"

        card = page.locator("#review-root [data-image-uuid='img-1'][data-kind='image']")
        card.locator("img.thumb").dblclick()
        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)
        modal.locator("button[data-toolbar-key='bbox']").wait_for(state="visible", timeout=8000)

        for _ in range(15):
            page.keyboard.press("ArrowDown")
        page.wait_for_timeout(200)

        scroll_y = page.evaluate("() => window.scrollY")
        assert scroll_y == 0, f"background page scrolled to {scroll_y} while the modal was open"

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
        modal.get_by_test_id("canvas-edit-modal-save-close").click()

        expect(modal).to_have_count(0, timeout=8000)
        expect(card).to_have_class("review-card stale-deleted")

    def test_delete_image_button_removes_the_whole_image_and_closes(self, page: Page, base_url: str):
        """The header's 🗑 button deletes the whole image (not just the
        highlighted bbox), skips the unsaved-changes prompt, and marks the
        originating card as deleted via the same IMAGE_DELETED broadcast the
        grid's own bulk-delete path uses."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)
        _mock_canvas_modal_image(page)
        page.route(
            "**/api/images/img-1",
            lambda route: (
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "imageUuid": "img-1", "filename": "folder/image1.jpg"}),
                )
                if route.request.method == "DELETE"
                else route.fallback()
            ),
        )

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")

        card = page.locator("#review-root [data-bbox-uuid='bbox-1']")
        card.locator("img.thumb").dblclick()

        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)

        modal.get_by_test_id("canvas-edit-modal-delete-image").click()
        page.get_by_test_id("modal-shell-dialog").locator("button", has_text="Delete").click()

        expect(modal).to_have_count(0, timeout=8000)
        expect(card).to_have_class("review-card stale-deleted")
        expect(card.locator(".card-state-badge")).to_have_text("🗑 Deleted")

    def test_delete_image_button_cancel_leaves_modal_open(self, page: Page, base_url: str):
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

        modal.get_by_test_id("canvas-edit-modal-delete-image").click()
        page.get_by_test_id("modal-shell-dialog").locator("button", has_text="Cancel").click()

        expect(modal).to_be_visible()
        expect(card).not_to_have_class("review-card stale-deleted")


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
# 12. Grid-only results (list view removed)
# ---------------------------------------------------------------------------


class TestGridOnlyResults:
    def test_no_list_toggle_and_stored_list_pref_still_renders_grid(self, page: Page, base_url: str):
        """The grid/list toggle is gone from the review toolbar, and a stale
        `review-view=list` in localStorage (written before the removal) must not
        leave the panel rendering nothing — the panel forces itself back to grid."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        page.goto(base_url)
        page.evaluate("() => localStorage.setItem('review-view', 'list')")
        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")

        expect(page.locator("#review-root .fb-view-toggle")).to_have_count(0)
        expect(page.locator("#review-root .review-list")).to_have_count(0)
        expect(page.locator("#review-root .review-grid")).to_have_count(1)
        expect(page.locator("#review-root [data-testid='review-card']")).to_have_count(3)


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


# ---------------------------------------------------------------------------
# 13. Image context menu — server-refused label writes
# ---------------------------------------------------------------------------

# Image-scope label — only these appear in the Mark-label flyout.
LABEL_INDOOR = {
    "uuid": "lbl-indoor",
    "name": "Indoor",
    "color": "#3388ff",
    "parentUuid": None,
    "scope": "image",
}


def _pick_label_from_flyout(page: Page, item_label: str, label_name: str):
    """Right-click the first image card, open `item_label`'s flyout, pick a label."""
    page.locator("#review-root [data-image-uuid='img-1'][data-kind='image']").click(button="right")
    page.locator(".ctx-menu-item", has_text=item_label).first.click()
    page.locator(".ctx-menu div", has_text=re.compile(f"^{label_name}$")).last.dispatch_event("mousedown")


class TestImageContextMenuLabelRefusal:
    """POST /api/images/labels/set answers `status: success, skipped: true` when
    it declines a write (ABSENT blocked by a verified bbox carrying that label).
    Nothing throws, so the Review adapter used to await-and-discard the response
    and the menu just closed — no toast, no change, no explanation. This drives
    the real grid so the adapter in app/review/main.js is what's under test, not
    a stand-in (contextMenu/providers/imageProvider.js's own handling is covered
    by test_context_menu_label_skipped.py)."""

    def _setup(self, page: Page, base_url: str, skipped: bool):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG, LABEL_INDOOR])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)

        body = {"status": "success"}
        if skipped:
            body["skipped"] = True
            body["message"] = "Cannot mark label absent: the image has a verified bbox with this label"
        # Registered after _mock_misc_review_endpoints so this handler wins.
        page.route(
            "**/api/images/labels/set",
            lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(body)),
        )

        _navigate_to_review(page, base_url)
        _switch_view(page, "images")
        expect(page.locator("#review-root [data-testid='review-card']")).to_have_count(2)

    def test_refused_absent_write_tells_the_user_nothing_changed(self, page: Page, base_url: str):
        self._setup(page, base_url, skipped=True)
        _pick_label_from_flyout(page, "Mark label absent", "Indoor")

        toast = page.locator("#toast-container", has_text="left unchanged")
        expect(toast).to_be_visible(timeout=3000)
        expect(toast).to_contain_text("verified bbox")

    def test_accepted_write_shows_no_toast(self, page: Page, base_url: str):
        """The control case — an ordinary write must stay silent, so the toast
        above is proof of the refusal and not of any label write at all."""
        self._setup(page, base_url, skipped=False)
        _pick_label_from_flyout(page, "Mark label absent", "Indoor")

        page.wait_for_timeout(500)
        assert page.locator("#toast-container", has_text="left unchanged").count() == 0


# ---------------------------------------------------------------------------
# 14. Bbox-card verification after a canvas save resynced the image
# ---------------------------------------------------------------------------


class TestVerificationAfterAnnotationsSaved:
    """IMAGE_ANNOTATIONS_SAVED rebuilds `entry.img.bboxes` from the save payload,
    but a bbox card's adapter stays closed over the bbox object it was built with.
    Verifying through that adapter used to mutate only the orphaned object, so the
    image card — which renders its per-label ✓/? marks from `img.bboxes` — kept
    showing the pre-toggle state until the next full re-render."""

    def _setup(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(
            page,
            [
                {**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"},
                {**VERIFIED_TAG, "name": "Verified", "color": "#27ae60", "kind": "system"},
            ],
        )
        _mock_misc_review_endpoints(page)
        _navigate_to_review(page, base_url)
        _show_all_bboxes(page)
        _switch_view(page, "all")

    def test_image_card_chip_follows_a_bbox_card_verify(self, page: Page, base_url: str):
        self._setup(page, base_url)

        image_card = page.locator("#review-root [data-image-uuid='img-1'][data-kind='image']")
        expect(image_card.locator(".label-chip")).to_have_text(["Cat?", "Dog✓"])

        # Stand in for a canvas save: the panel resyncs img.bboxes from this
        # payload, replacing the objects the bbox cards' adapters hold.
        page.evaluate(
            """async ([bbox1, bbox2]) => {
            const { events } = await import('/app-static/js/shared/events/bus.js');
            const { EventTypes } = await import('/app-static/js/shared/events/types.js');
            const toUi = (b) => ({
                uuid: b.uuid, x: b.x, y: b.y, width: b.width, height: b.height,
                label: { uuid: b.label.uuid, text: b.label.name, color: b.label.color },
                tags: b.tags, metadataJson: b.metadataJson,
            });
            events.publish(EventTypes.IMAGE_ANNOTATIONS_SAVED, {
                imageUuid: 'img-1',
                imageFilename: 'folder/image1.jpg',
                bboxes: new Map([bbox1, bbox2].map(b => [b.uuid, toUi(b)])),
            });
        }""",
            [BBOX_1, BBOX_2],
        )
        page.wait_for_timeout(200)
        expect(image_card.locator(".label-chip")).to_have_text(["Cat?", "Dog✓"])

        # Now verify the Cat box through its own bbox card's context menu.
        page.locator("#review-root [data-bbox-uuid='bbox-1']").click(button="right")
        page.locator(".ctx-menu-item", has_text="Verification").first.click()
        page.locator(".ctx-menu-item", has_text="Mark Verified").last.click()

        expect(image_card.locator(".label-chip")).to_have_text(["Cat✓", "Dog✓"])


# ---------------------------------------------------------------------------
# 12. Canvas modal hotkeys
# ---------------------------------------------------------------------------


class TestCanvasModalHotkeys:
    """Hotkeys must keep working for the whole life of the modal.

    Reported bug: open the modal, press ']' once to advance, and every hotkey
    (']', '[', Delete, tool keys) goes dead until the modal is closed and
    reopened.
    """

    def _open_modal(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)
        _mock_canvas_modal_image(page)

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")

        page.locator("#review-root [data-bbox-uuid='bbox-1'] img.thumb").dblclick()
        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)
        # Wait for the sidebar/inference panels, i.e. the fully-wired modal.
        expect(modal.locator("button[data-toolbar-key='bbox']")).to_be_enabled(timeout=8000)
        return modal

    def test_tool_hotkey_works_when_modal_opens(self, page: Page, base_url: str):
        modal = self._open_modal(page, base_url)

        page.keyboard.press("r")

        expect(modal.locator("button[data-tool='bbox']")).to_have_class(re.compile(r"\bselected\b"))

    def test_tool_hotkey_still_works_after_bracket_navigation(self, page: Page, base_url: str):
        modal = self._open_modal(page, base_url)

        page.keyboard.press("]")
        page.wait_for_timeout(600)  # let the nav-triggered image load settle
        page.keyboard.press("r")

        expect(modal.locator("button[data-tool='bbox']")).to_have_class(re.compile(r"\bselected\b"))

    def test_bracket_navigation_still_works_after_bracket_navigation(self, page: Page, base_url: str):
        """']' twice must reach the third entry (bbox-3, on image2)."""
        modal = self._open_modal(page, base_url)
        title = modal.get_by_test_id("canvas-edit-modal-title")
        expect(title).to_have_text("Edit: folder/image1.jpg")

        page.keyboard.press("]")
        page.wait_for_timeout(600)
        page.keyboard.press("]")

        expect(title).to_have_text("Edit: folder/image2.jpg", timeout=5000)

    def test_tool_hotkey_still_works_after_navigating_to_another_image(self, page: Page, base_url: str):
        modal = self._open_modal(page, base_url)
        title = modal.get_by_test_id("canvas-edit-modal-title")

        page.keyboard.press("]")
        page.wait_for_timeout(600)
        page.keyboard.press("]")
        expect(title).to_have_text("Edit: folder/image2.jpg", timeout=5000)
        page.wait_for_timeout(600)

        page.keyboard.press("r")

        expect(modal.locator("button[data-tool='bbox']")).to_have_class(re.compile(r"\bselected\b"))


# ---------------------------------------------------------------------------
# 14. Canvas modal next/prev prefetch
# ---------------------------------------------------------------------------


class TestCanvasNavPrefetch:
    """Stepping through the canvas modal must not pay a full round trip per step.

    `/file` reads the whole image, base64s it AND does the metadata DB read in
    one request, so on a slow DB that latency is the entire cost of a `]` press.
    The modal knows the whole nav list up front, so the neighbours can be warmed
    while the user is looking at the current image.
    """

    def _mock_files(self, page: Page):
        """Serve a distinct body per image path and count requests per path."""
        calls: list[str] = []

        def handle(route: Route):
            # ".../api/images/<urlencoded path>/file"
            path = urllib.parse.unquote(route.request.url.split("/api/images/")[1].rsplit("/file", 1)[0])
            calls.append(path)
            uuid = "img-1" if path.endswith("image1.jpg") else "img-2"
            bboxes = [BBOX_1, BBOX_2] if uuid == "img-1" else [BBOX_3]
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "imageBase64": TINY_PNG_B64,
                        "mimeType": "image/png",
                        "image": {"uuid": uuid, "filename": path, "bboxes": bboxes},
                    }
                ),
            )

        page.route("**/api/images/*/file", handle)
        return calls

    def _open_modal_on_first_bbox(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)
        png_bytes = base64.b64decode(TINY_PNG_B64)
        page.route(
            "**/thumbnail?*",
            lambda route: route.fulfill(status=200, content_type="image/png", body=png_bytes),
        )
        calls = self._mock_files(page)

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")
        page.locator("#review-root [data-bbox-uuid='bbox-1'] img.thumb").dblclick()
        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)
        return modal, calls

    def test_the_next_image_is_fetched_before_the_user_asks_for_it(self, page: Page, base_url: str):
        modal, calls = self._open_modal_on_first_bbox(page, base_url)

        # No navigation has happened — image2 is fetched purely speculatively.
        wait_until(
            page,
            lambda: any(c.endswith("image2.jpg") for c in calls),
            lambda: f"Expected a prefetch of the next image, got {calls}",
        )

    def test_navigating_to_a_prefetched_image_issues_no_new_request(self, page: Page, base_url: str):
        modal, calls = self._open_modal_on_first_bbox(page, base_url)
        wait_until(
            page,
            lambda: any(c.endswith("image2.jpg") for c in calls),
            lambda: f"Expected a prefetch of the next image, got {calls}",
        )
        before = len([c for c in calls if c.endswith("image2.jpg")])

        modal.get_by_test_id("canvas-edit-modal-next").click()
        expect(modal.get_by_test_id("canvas-edit-modal-title")).to_have_text("Edit: folder/image2.jpg")

        after = len([c for c in calls if c.endswith("image2.jpg")])
        assert after == before, f"navigation re-fetched a prefetched image: {calls}"


# ---------------------------------------------------------------------------
# 15. Pre-highlighted bbox is a real selection
# ---------------------------------------------------------------------------


class TestOpenedBboxIsProperlySelected:
    """Opening the canvas modal from a bbox card highlights that bbox, which
    puts live resize anchors on screen before the user has clicked anything.

    Those anchors have to behave like anchors on a box the user selected by
    hand: the highlight attaches the transformer directly, so unless it also
    does the rest of the selection bookkeeping, dragging them silently mutates
    the box without marking the canvas dirty — the edit then looks saved but is
    discarded on close.
    """

    def _open_modal(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)
        _mock_canvas_modal_image(page)

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")
        page.locator("#review-root [data-bbox-uuid='bbox-1'] img.thumb").dblclick()

        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)
        page.wait_for_function(
            """() => {
                const stages = (window.Konva && window.Konva.stages) || [];
                for (const stage of stages) {
                    if (!stage.container()?.closest('.canvas-modal-canvas')) continue;
                    if (stage.findOne('#bbox-1')) return true;
                }
                return false;
            }""",
            timeout=8000,
        )
        return modal

    _RESIZE = """() => {
        const stages = (window.Konva && window.Konva.stages) || [];
        for (const stage of stages) {
            if (!stage.container()?.closest('.canvas-modal-canvas')) continue;
            const node = stage.findOne('#bbox-1');
            if (!node) continue;
            // Grow the node the way an anchor drag does, then fire the event
            // Konva's Transformer fires on the attached node when the drag ends.
            node.width(node.width() * 1.5);
            node.height(node.height() * 1.5);
            node.fire('transformend', { target: node }, true);
            return true;
        }
        return false;
    }"""

    def test_the_opened_bbox_is_attached_to_the_transformer(self, page: Page, base_url: str):
        """The premise of the bug: anchors really are live before any click."""
        self._open_modal(page, base_url)
        attached = page.evaluate("""() => {
                const stages = (window.Konva && window.Konva.stages) || [];
                for (const stage of stages) {
                    if (!stage.container()?.closest('.canvas-modal-canvas')) continue;
                    const tr = stage.findOne('Transformer');
                    if (tr) return tr.nodes().map(n => n.id());
                }
                return [];
            }""")
        assert attached == ["bbox-1"], attached

    def test_resizing_the_pre_highlighted_bbox_enables_save(self, page: Page, base_url: str):
        modal = self._open_modal(page, base_url)
        save_btn = modal.get_by_test_id("canvas-edit-modal-save-close")
        expect(save_btn).to_have_class("save-btn--clean")

        assert page.evaluate(self._RESIZE), "modal canvas stage / bbox-1 node not found"

        expect(save_btn).to_have_class("save-btn--dirty")


# ---------------------------------------------------------------------------
# 16. Save & Close reads "Close" with nothing to save
# ---------------------------------------------------------------------------


class TestSaveAndCloseBecomesClose:
    """With a clean canvas there is nothing to save, so the footer's primary
    button says "Close" and closes. It used to keep the save wording and return
    early on click — the most prominent button in the footer was dead whenever
    the user was only looking at an image."""

    def _open_modal(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)
        _mock_canvas_modal_image(page)

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")
        page.locator("#review-root [data-bbox-uuid='bbox-1'] img.thumb").dblclick()
        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)
        return modal

    def test_a_clean_canvas_offers_close_and_closes(self, page: Page, base_url: str):
        modal = self._open_modal(page, base_url)
        btn = modal.get_by_test_id("canvas-edit-modal-save-close")
        expect(btn).to_have_text("Close")

        btn.click()
        expect(modal).to_have_count(0, timeout=5000)

    def test_the_label_follows_the_dirty_state_both_ways(self, page: Page, base_url: str):
        """It must go back to "Close" once the changes are saved, or the button
        would keep promising a save it no longer has anything to do.

        Note the reverse direction is driven by an actual save, not by undo:
        `runtime.undo()` sets `_dirty` unconditionally, so dirty means "edited
        since load or save", not "differs from the loaded state".
        """
        modal = self._open_modal(page, base_url)
        page.route(
            "**/api/images/metadata/update",
            lambda route: route.fulfill(
                status=200, content_type="application/json", body=json.dumps({"status": "success"})
            ),
        )
        btn = modal.get_by_test_id("canvas-edit-modal-save-close")

        modal.locator("button[data-toolbar-key='delete']").click()
        expect(btn).to_have_text("Save & Close")

        # Save-only keeps the modal open and marks the canvas clean.
        modal.locator("button", has_text=re.compile(r"^Save$")).click()
        expect(btn).to_have_text("Close")

    def test_a_dirty_canvas_still_saves_before_closing(self, page: Page, base_url: str):
        """The save path must survive the label switch."""
        saves: list[dict] = []

        def handle_update(route: Route):
            saves.append(json.loads(route.request.post_data or "{}"))
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

        modal = self._open_modal(page, base_url)
        page.route("**/api/images/metadata/update", handle_update)

        modal.locator("button[data-toolbar-key='delete']").click()
        btn = modal.get_by_test_id("canvas-edit-modal-save-close")
        expect(btn).to_have_text("Save & Close")
        btn.click()

        expect(modal).to_have_count(0, timeout=8000)
        assert len(saves) == 1, f"expected exactly one save, got {saves}"


# ---------------------------------------------------------------------------
# 17. Dirty means "differs from what was loaded/saved"
# ---------------------------------------------------------------------------


class TestUndoneEditIsNotDirty:
    """Editing and then undoing back to the loaded state must count as never
    modified — no unsaved-changes prompt, and the footer offers plain "Close".

    Dirty is compared against the loaded/saved annotation content rather than
    tracked as a latch, so anything that ends up back where it started is
    clean however it got there.
    """

    def _open_modal(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)
        _mock_canvas_modal_image(page)

        _navigate_to_review(page, base_url)
        _switch_view(page, "bboxes")
        page.locator("#review-root [data-bbox-uuid='bbox-1'] img.thumb").dblclick()
        modal = page.get_by_test_id("canvas-edit-modal")
        expect(modal).to_be_visible(timeout=8000)
        page.wait_for_function(
            """() => {
                const stages = (window.Konva && window.Konva.stages) || [];
                for (const stage of stages) {
                    if (!stage.container()?.closest('.canvas-modal-canvas')) continue;
                    if (stage.findOne('#bbox-1')) return true;
                }
                return false;
            }""",
            timeout=8000,
        )
        return modal

    def _bbox_count(self, page: Page) -> int:
        return page.evaluate("""() => {
                const stages = (window.Konva && window.Konva.stages) || [];
                for (const stage of stages) {
                    if (!stage.container()?.closest('.canvas-modal-canvas')) continue;
                    return stage.find('.bbox').length;
                }
                return -1;
            }""")

    def test_undoing_a_delete_returns_the_canvas_to_clean(self, page: Page, base_url: str):
        modal = self._open_modal(page, base_url)
        btn = modal.get_by_test_id("canvas-edit-modal-save-close")
        assert self._bbox_count(page) == 2

        modal.locator("button[data-toolbar-key='delete']").click()
        expect(btn).to_have_text("Save & Close")

        page.keyboard.press("Control+z")
        # Assert the undo actually happened, so a dead hotkey can't make the
        # dirty assertion below pass for the wrong reason.
        assert self._bbox_count(page) == 2
        expect(btn).to_have_text("Close")

    def test_moving_a_box_and_putting_it_back_is_clean(self, page: Page, base_url: str):
        """Not undo-specific: the comparison is against content, so a manual
        revert counts too."""
        modal = self._open_modal(page, base_url)
        btn = modal.get_by_test_id("canvas-edit-modal-save-close")

        move = """(dx) => {
            const stages = (window.Konva && window.Konva.stages) || [];
            for (const stage of stages) {
                if (!stage.container()?.closest('.canvas-modal-canvas')) continue;
                const node = stage.findOne('#bbox-1');
                if (!node) continue;
                node.x(node.x() + dx);
                node.fire('dragend', { target: node }, true);
                return true;
            }
            return false;
        }"""
        assert page.evaluate(move, 40)
        expect(btn).to_have_text("Save & Close")

        assert page.evaluate(move, -40)
        expect(btn).to_have_text("Close")

    def test_an_image_level_label_change_still_marks_dirty(self, page: Page, base_url: str):
        """Image-level label presence lives outside `bboxes`, so no bbox
        comparison can see it — it goes through markDirty and must stay dirty
        even though the boxes themselves are untouched."""
        modal = self._open_modal(page, base_url)
        btn = modal.get_by_test_id("canvas-edit-modal-save-close")
        expect(btn).to_have_text("Close")

        # The Labels sidebar tab's status button is the real path into markDirty.
        clicked = page.evaluate("""() => {
                const row = document.querySelector('.canvas-modal-sidebar .label-panel-row');
                const btn = row?.querySelector('.label-status-btn');
                if (!btn) return false;
                btn.click();
                return true;
            }""")
        assert clicked, "no label row in the modal sidebar"
        expect(btn).to_have_text("Save & Close")
