"""
Tests for the image "Apply to All / Apply to Selected" modal's label picker —
apps/image_tagging_server/static/js/app/review/imageBulkActionModal.js

The label picker used to be a single <select>, so marking six labels absent
across a search result meant six trips through the modal. It is now the shared
multi-select dropdown (createMultiSelectDropdown), which also badges the labels
belonging to the active training run.

Covers:
  - several labels checked → one request carrying every labelUuid
  - nothing checked → no request at all (a label action with no labels would
    otherwise be a silent no-op on the server)
  - labels in the active training run carry a "training" badge

Run with:
    pytest playwright_tests/test_image_bulk_action_labels.py -p no:asyncio -v
"""

import json

import pytest
from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None}
LABEL_DOG = {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00", "parentUuid": None}
LABEL_FOX = {"uuid": "lbl-fox", "name": "Fox", "color": "#ffaa00", "parentUuid": None}

IMAGE_1 = {
    "uuid": "img-1",
    "filename": "one.jpg",
    "bboxes": [],
    "tags": [],
    "metadataJson": None,
}
IMAGE_2 = {
    "uuid": "img-2",
    "filename": "two.jpg",
    "bboxes": [],
    "tags": [],
    "metadataJson": None,
}


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _mock_review_queue(page: Page, images: list):
    page.route(
        "**/api/images/review",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "images": images, "hasMore": False, "totalCount": len(images)}),
        ),
    )


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


def _navigate_to_review(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
    page.locator("#tab-image-tagging .section-tab-button[data-tab='tab-review']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#review-root .review-grid", timeout=4000)


def _open_bulk_modal_for_selected_images(page: Page, base_url: str, image_uuids: list):
    """Select the given image cards and open the bulk-action modal."""
    _navigate_to_review(page, base_url)
    for uuid in image_uuids:
        page.locator(f"#review-root [data-image-uuid='{uuid}'][data-kind='image'] input[type=checkbox]").click()
    page.get_by_test_id("review-bulk-apply").click()


def _modal(page: Page):
    return page.locator("div:has(> strong)", has=page.locator(".bulk-modal-picker")).last


# ---------------------------------------------------------------------------
# Multi-select
# ---------------------------------------------------------------------------


class TestImageBulkActionLabelMultiSelect:
    def test_checking_two_labels_sends_both_in_one_request(self, page: Page, base_url: str):
        _mock_review_queue(page, [IMAGE_1, IMAGE_2])
        _mock_labels(page, [LABEL_CAT, LABEL_DOG, LABEL_FOX])
        _mock_tags(page, [])

        calls = []

        def handle_bulk(route: Route):
            calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "modifiedCount": 2, "failCount": 0, "skippedCount": 0}),
            )

        page.route("**/api/images/bulk-apply-by-uuids", handle_bulk)

        _open_bulk_modal_for_selected_images(page, base_url, ["img-1", "img-2"])

        picker = page.locator(".bulk-modal-picker .ms-dropdown").first
        expect(picker).to_be_visible()
        picker.locator("input[type=checkbox][value='lbl-cat']").check()
        picker.locator("input[type=checkbox][value='lbl-fox']").check()

        page.locator("button", has_text="Apply").last.click()

        wait_until(page, lambda: calls, lambda: f"Expected a bulk-apply request, got {calls}")
        body = calls[0]
        assert body["action"] == "setPresent", body
        assert sorted(body["labelUuids"]) == ["lbl-cat", "lbl-fox"], body
        assert sorted(body["imageUuids"]) == ["img-1", "img-2"], body

    def test_applying_with_no_label_checked_sends_nothing(self, page: Page, base_url: str):
        """Guard: the server treats a label action with no labels as a no-op, so
        the modal must refuse rather than fire a request that does nothing."""
        _mock_review_queue(page, [IMAGE_1])
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [])

        calls = []
        page.route(
            "**/api/images/bulk-apply-by-uuids",
            lambda route: (
                calls.append(json.loads(route.request.post_data)),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "modifiedCount": 0, "failCount": 0}),
                ),
            )[-1],
        )

        _open_bulk_modal_for_selected_images(page, base_url, ["img-1"])

        page.locator("button", has_text="Apply").last.click()
        page.wait_for_timeout(400)

        assert calls == [], f"Expected no request without a label selected, got {calls}"
        # The modal stays open so the user can correct the selection.
        expect(page.locator(".bulk-modal-picker .ms-dropdown").first).to_be_visible()

    def test_label_filter_box_narrows_the_list(self, page: Page, base_url: str):
        _mock_review_queue(page, [IMAGE_1])
        _mock_labels(page, [LABEL_CAT, LABEL_DOG, LABEL_FOX])
        _mock_tags(page, [])

        _open_bulk_modal_for_selected_images(page, base_url, ["img-1"])

        picker = page.locator(".bulk-modal-picker .ms-dropdown").first
        rows = picker.locator(".ms-item")
        expect(rows).to_have_count(3)

        picker.locator(".ms-dropdown__search").fill("do")
        expect(picker.locator(".ms-item:visible")).to_have_count(1)
        expect(picker.locator(".ms-item:visible")).to_contain_text("Dog")


# ---------------------------------------------------------------------------
# Active-training-run badge
# ---------------------------------------------------------------------------


class TestTrainingRunBadge:
    def _mock_training_config(self, page: Page):
        page.route(
            "**/api/training-configs",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "configs": [
                            {
                                "uuid": "tc-1",
                                "name": "Nightly",
                                "modelConfigNodeUuid": "node-root",
                                "datasetUuids": [],
                                "hyperparamsJson": {},
                            }
                        ],
                    }
                ),
            ),
        )
        page.route(
            "**/api/model-configs",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "configs": [
                            {
                                "uuid": "mc-1",
                                "name": "Shape",
                                "nodes": [
                                    {"uuid": "node-root", "labelUuid": None},
                                    {"uuid": "node-cat", "labelUuid": "lbl-cat"},
                                ],
                            }
                        ],
                    }
                ),
            ),
        )
        # Selected before the app boots, so activeTrainingConfig.init() picks it up.
        page.add_init_script("sessionStorage.setItem('wtch_active_tc_uuid', 'tc-1');")

    def test_labels_in_the_active_run_are_badged(self, page: Page, base_url: str):
        _mock_review_queue(page, [IMAGE_1])
        _mock_labels(page, [LABEL_CAT, LABEL_DOG])
        _mock_tags(page, [])
        self._mock_training_config(page)

        _open_bulk_modal_for_selected_images(page, base_url, ["img-1"])

        picker = page.locator(".bulk-modal-picker .ms-dropdown").first
        cat_row = picker.locator(".ms-item", has_text="Cat")
        dog_row = picker.locator(".ms-item", has_text="Dog")

        expect(cat_row.locator(".ms-item__badge--training")).to_have_count(1)
        expect(dog_row.locator(".ms-item__badge--training")).to_have_count(0)
