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

import pytest
from playwright.sync_api import Page, Route, expect

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"

# A 1x1 transparent PNG, base64-encoded — small valid image for canvas loadImage().
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parent_uuid": None}
LABEL_DOG = {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00", "parent_uuid": None}

UNVERIFIED_TAG = {"uuid": UNVERIFIED_TAG_UUID, "icon": None}
REVIEWED_TAG = {"uuid": "tag-reviewed", "name": "Reviewed", "color": "#4c4", "icon": None, "kind": "user"}

# bbox-1, bbox-3: still unverified (carry the UNVERIFIED tag)
# bbox-2: verified (no UNVERIFIED tag)
BBOX_1 = {
    "uuid": "bbox-1", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2,
    "label": {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000"},
    "labelUuid": "lbl-cat",
    "tags": [UNVERIFIED_TAG],
    "metadataJson": {"confidence": "0.92"},
}
BBOX_2 = {
    "uuid": "bbox-2", "x": 0.3, "y": 0.3, "width": 0.1, "height": 0.1,
    "label": {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00"},
    "labelUuid": "lbl-dog",
    "tags": [],
    "metadataJson": None,
}
BBOX_3 = {
    "uuid": "bbox-3", "x": 0.2, "y": 0.2, "width": 0.15, "height": 0.15,
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
            body=json.dumps({
                "status": "success",
                "images": images,
                "hasMore": has_more,
                "totalCount": total_count,
            }),
        ),
    )


def _mock_paged_review_queue(page: Page, page1: list, page2: list):
    """Serve `page1` for the initial fetch (after_uuid is null) and `page2` after that."""
    def handle(route: Route):
        body = json.loads(route.request.post_data or "{}")
        images = page2 if body.get("after_uuid") else page1
        has_more = images is page1
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "status": "success",
                "images": images,
                "hasMore": has_more,
                "totalCount": len(page1) + len(page2),
            }),
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
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"})),
    )
    page.route(
        "**/api/images/bbox/*",
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))
        if route.request.method in ("PATCH", "DELETE") else route.fallback(),
    )
    page.route(
        "**/api/images/labels/set",
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"})),
    )


def _show_all_bboxes(page: Page):
    """Uncheck the 'Unverified' tag filter so verified bboxes also render."""
    cb = page.locator(f"#review-root input[value='{UNVERIFIED_TAG_UUID}']")
    cb.uncheck()
    page.wait_for_timeout(400)


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
        _mock_tags(page, [UNVERIFIED_TAG | {"name": "Unverified", "color": "#888"} if False else
                          {**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}, REVIEWED_TAG])

        _navigate_to_review(page, base_url)

        # Default filter shows only bboxes carrying the UNVERIFIED tag: bbox-1, bbox-3.
        cards = page.locator("#review-root .review-card")
        expect(cards).to_have_count(4)  # 2 image cards + 2 unverified-bbox cards
        expect(page.locator("#review-root [data-image-uuid='img-1'][data-kind='image']")).to_be_visible()
        expect(page.locator("#review-root [data-image-uuid='img-2'][data-kind='image']")).to_be_visible()
        expect(page.locator("#review-root [data-bbox-uuid='bbox-1']")).to_be_visible()
        expect(page.locator("#review-root [data-bbox-uuid='bbox-3']")).to_be_visible()
        expect(page.locator("#review-root [data-bbox-uuid='bbox-2']")).to_have_count(0)


# ---------------------------------------------------------------------------
# 2. Verify chip
# ---------------------------------------------------------------------------

class TestVerifyChip:
    def test_verify_chip_reflects_bbox_state(self, page: Page, base_url: str):
        """The verify chip reads '✓ Verified' for verified bboxes and '◌ Unverified' otherwise."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}, REVIEWED_TAG])

        _navigate_to_review(page, base_url)
        _show_all_bboxes(page)

        unverified_chip = page.locator("#review-root [data-bbox-uuid='bbox-1'] .verify-chip")
        verified_chip = page.locator("#review-root [data-bbox-uuid='bbox-2'] .verify-chip")

        expect(unverified_chip).to_have_text("◌ Unverified")
        expect(unverified_chip).to_have_class("verify-chip")
        expect(verified_chip).to_have_text("✓ Verified")
        expect(verified_chip).to_have_class("verify-chip verify-chip--verified")


# ---------------------------------------------------------------------------
# 3. Single checkbox selection
# ---------------------------------------------------------------------------

class TestSingleSelection:
    def test_single_checkbox_selection_updates_bulk_button(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        _navigate_to_review(page, base_url)
        bulk_btn = page.locator("#review-root .review-toolbar button", has_text="Bulk Apply")
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

        cards = page.locator("#review-root .review-card")
        expect(cards).to_have_count(3)  # bbox-1, bbox-2, bbox-3

        cards.nth(0).locator("input[type=checkbox]").click()
        cards.nth(2).locator("input[type=checkbox]").click(modifiers=["Shift"])

        for i in range(3):
            expect(cards.nth(i).locator("input[type=checkbox]")).to_be_checked()
        bulk_btn = page.locator("#review-root .review-toolbar button", has_text="Bulk Apply")
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

        cards = page.locator("#review-root .review-card")
        expect(cards).to_have_count(3)

        select_all = page.locator("#review-root label", has_text="Select all").locator("input[type=checkbox]")
        select_all.check()

        for i in range(3):
            expect(cards.nth(i).locator("input[type=checkbox]")).to_be_checked()
        bulk_btn = page.locator("#review-root .review-toolbar button", has_text="Bulk Apply")
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

        # Default filter shows the two unverified bboxes (bbox-1, bbox-3).
        page.locator("#review-root [data-bbox-uuid='bbox-1'] input[type=checkbox]").click()
        page.locator("#review-root [data-bbox-uuid='bbox-3'] input[type=checkbox]").click()

        page.locator("#review-root .review-toolbar button", has_text="Bulk Apply").click()

        modal = page.locator(".app-modal-overlay, div:has(> strong:text-is('Apply to 2 selected bboxes'))").last
        expect(page.locator("strong", has_text="Apply to 2 selected bboxes")).to_be_visible()

        # 'Change Label' (relabel) is selected by default — choose Dog.
        page.locator("div:has(> strong:text-is('Apply to 2 selected bboxes')) select").select_option("lbl-dog")
        page.locator("div:has(> strong:text-is('Apply to 2 selected bboxes')) button", has_text="Apply").click()
        page.wait_for_timeout(600)

        assert len(calls) == 2, f"Expected 2 PATCH calls, got {calls}"
        for call in calls:
            assert call["body"]["labelUuid"] == "lbl-dog"
            # Relabeling to a real label strips the UNVERIFIED tag — i.e. "verifies" the bbox.
            assert UNVERIFIED_TAG_UUID not in call["body"]["tagUuids"]


# ---------------------------------------------------------------------------
# 7. Bulk image action modal
# ---------------------------------------------------------------------------

class TestBulkImageAction:
    def test_image_bulk_action_modal_fires_api_with_selected_entries(self, page: Page, base_url: str):
        """Selecting image cards and confirming 'Mark label Absent' calls the image-labels API."""
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])
        _mock_misc_review_endpoints(page)

        calls = []

        def handle_set(route: Route):
            calls.append(json.loads(route.request.post_data))
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

        page.route("**/api/images/labels/set", handle_set)

        _navigate_to_review(page, base_url)
        _switch_view(page, "images")

        cards = page.locator("#review-root .review-card")
        expect(cards).to_have_count(2)
        for i in range(2):
            cards.nth(i).locator("input[type=checkbox]").click()

        page.locator("#review-root .review-toolbar button", has_text="Bulk Apply").click()

        modal_title = page.locator("strong", has_text="Apply to 2 selected images")
        expect(modal_title).to_be_visible()

        # 'Mark label Absent' is selected by default — choose Cat.
        page.locator("div:has(> strong:text-is('Apply to 2 selected images')) select").select_option("lbl-cat")
        page.locator("div:has(> strong:text-is('Apply to 2 selected images')) button", has_text="Apply").click()
        page.wait_for_timeout(600)

        assert len(calls) == 2, f"Expected 2 setImageLabel calls, got {calls}"
        for call in calls:
            assert call["labelUuid"] == "lbl-cat"
            assert call["status"] == "absent"


# ---------------------------------------------------------------------------
# 8. Pagination
# ---------------------------------------------------------------------------

class TestPagination:
    def test_next_and_prev_advance_and_return(self, page: Page, base_url: str):
        _mock_paged_review_queue(page, PAGE_1_IMAGES, PAGE_2_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        _navigate_to_review(page, base_url)

        prev_btn = page.locator("#review-root .review-toolbar button", has_text="Prev")
        next_btn = page.locator("#review-root .review-toolbar button", has_text="Next")

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
    def test_filter_toggle_button_reflects_active_count(self, page: Page, base_url: str):
        _mock_review_queue(page, PAGE_1_IMAGES)
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [{**UNVERIFIED_TAG, "name": "Unverified", "color": "#888", "kind": "system"}])

        _navigate_to_review(page, base_url)
        toggle_btn = page.locator("#review-root .review-toolbar button", has_text="Filters")

        # Default state: only the Unverified tag checked → count of 1.
        expect(toggle_btn).to_have_text("☰ Filters (1)")

        # The filter panel starts visible — pick a bbox-label filter directly.
        page.locator("#review-root select.filter-label-select").select_option("lbl-cat")
        page.wait_for_timeout(300)

        expect(toggle_btn).to_have_text("☰ Filters (2)")
        expect(toggle_btn).to_have_class("filter-active")


# ---------------------------------------------------------------------------
# 10. Canvas modal round trip
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

        # Thumbnails are fetched as raw bytes via fetch().blob() — serve a tiny real PNG
        # so the <img class="thumb"> has nonzero size and can be double-clicked.
        png_bytes = base64.b64decode(TINY_PNG_B64)
        page.route(
            "**/thumbnail*",
            lambda route: route.fulfill(status=200, content_type="image/png", body=png_bytes),
        )
        page.route(
            "**/api/images/*/file",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({
                    "status": "success",
                    "imageBase64": TINY_PNG_B64,
                    "mimeType": "image/png",
                    "image": {"uuid": "img-1", "filename": "folder/image1.jpg", "bboxes": [BBOX_1, BBOX_2]},
                }),
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

        _navigate_to_review(page, base_url)

        card = page.locator("#review-root [data-bbox-uuid='bbox-1']")
        card.locator("img.thumb").dblclick()

        modal = page.locator(".canvas-modal-overlay")
        expect(modal).to_be_visible(timeout=8000)
        expect(modal.locator(".canvas-modal-title")).to_have_text("Edit: folder/image1.jpg")

        # Delete the highlighted (originally-opened) bbox, then Save & Close.
        modal.locator("button[data-toolbar-key='delete']").click()
        page.wait_for_timeout(300)
        save_btn = modal.locator("button", has_text="Save & Close")
        save_btn.click()
        page.wait_for_timeout(800)

        expect(modal).to_have_count(0)
        expect(card).to_have_class("review-card stale-deleted")
        expect(card.locator(".card-state-badge")).to_have_text("🗑 Deleted")
        expect(card.locator("input[type=checkbox]")).to_be_disabled()


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

        bbox_card = page.locator("#review-root [data-bbox-uuid='bbox-1']")
        image_card = page.locator("#review-root [data-image-uuid='img-1'][data-kind='image']")

        bbox_card.hover()
        page.wait_for_timeout(200)

        expect(bbox_card).to_have_class("review-card group-highlight-self-bbox")
        expect(image_card).to_have_class("review-card review-card--image group-highlight-image")

        # Move away — highlight should clear
        page.locator("#review-root .review-toolbar").hover()
        page.wait_for_timeout(200)
        expect(bbox_card).to_have_class("review-card")
        expect(image_card).to_have_class("review-card review-card--image")
