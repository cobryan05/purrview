"""Tests for comparing an archived model version in the Verify Model dialog.

A version is not a model — nothing can load it by name — so picking one in the
dialog materialises a temporary copy and deletes it again. That create/cleanup
pairing is the part only a browser proves, and the part whose failure leaves a
full-size model file behind, so it is what these assert.
"""

import json
from urllib.parse import unquote

from playwright.sync_api import Page, Route, expect

from playwright_tests.test_run_model_on_results import (
    LABEL_CAT,
    LABEL_DOG,
    PAGE_1_IMAGES,
    UNVERIFIED_TAG,
    _mock_labels,
    _mock_review_queue,
    _mock_tags,
    _navigate_to_review,
)
from playwright_tests.waits import wait_until

VERSIONS = [
    {
        "versionId": "20260901T101500Z",
        "archivedAt": "2026-09-05T12:00:00Z",
        "createdAt": "2026-09-01T10:15:00Z",
        "bytes": 1000,
        "files": ["aged.onnx"],
    }
]


def _mock_models_with_history(page: Page, *, with_history=("aged",), without_history=("fresh",)):
    """Two detectors: one with an archived generation, one without."""
    all_names = list(with_history) + list(without_history)

    page.route(
        lambda url: url.split("?")[0].endswith("/api/models"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "models": all_names}),
        ),
    )
    page.route(
        "**/api/models/*/meta",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "modelKind": "detector", "labelUuids": ["lbl-cat"]}),
        ),
    )

    def handle_versions(route: Route):
        name = route.request.url.rstrip("/").split("/")[-2]
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {"status": "success", "modelName": name, "versions": VERSIONS if name in with_history else []}
            ),
        )

    page.route("**/api/models/*/versions", handle_versions)


def _mock_preview_endpoints(page: Page, created: list, deleted: list):
    def handle_create(route: Route):
        name = route.request.url.rstrip("/").split("/")
        model, version = name[-4], name[-2]
        preview = f"{model}@{version}"
        created.append(preview)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "previewName": preview}),
        )

    page.route("**/api/models/*/versions/*/preview", handle_create)

    def handle_delete(route: Route):
        # The preview name contains "@", which encodeURIComponent percent-escapes
        # on the way into the path segment.
        deleted.append(unquote(route.request.url.rstrip("/").split("/")[-2]))
        route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

    page.route("**/api/models/*/preview", handle_delete)


def _setup(page: Page, base_url: str, created: list, deleted: list):
    _mock_review_queue(page, PAGE_1_IMAGES)
    _mock_labels(page, [LABEL_CAT, LABEL_DOG])
    _mock_tags(page, [UNVERIFIED_TAG])
    _mock_models_with_history(page)
    _mock_preview_endpoints(page, created, deleted)
    _navigate_to_review(page, base_url)
    page.get_by_test_id("review-verify-model").click()
    expect(page.get_by_test_id("compare-models-primary-a")).to_be_visible()


def _version_select(page: Page, side="a"):
    return page.get_by_test_id(f"compare-models-version-{side}")


class TestVersionPicker:
    def test_version_picker_appears_only_for_a_model_that_has_history(self, page: Page, base_url: str):
        """Most models have never been retrained; an empty dropdown is just noise."""
        created, deleted = [], []
        _setup(page, base_url, created, deleted)

        expect(_version_select(page)).to_be_hidden()

        page.get_by_test_id("compare-models-primary-a").select_option("fresh")
        expect(_version_select(page)).to_be_hidden()

        page.get_by_test_id("compare-models-primary-a").select_option("aged")
        expect(_version_select(page)).to_be_visible()
        # "Current version (live)" plus the one archived generation.
        expect(_version_select(page).locator("option")).to_have_count(2)

    def test_switching_to_a_model_without_history_hides_the_picker_again(self, page: Page, base_url: str):
        created, deleted = [], []
        _setup(page, base_url, created, deleted)

        page.get_by_test_id("compare-models-primary-a").select_option("aged")
        expect(_version_select(page)).to_be_visible()
        page.get_by_test_id("compare-models-primary-a").select_option("fresh")
        expect(_version_select(page)).to_be_hidden()

    def test_picking_a_version_materialises_it_and_says_it_runs_alone(self, page: Page, base_url: str):
        created, deleted = [], []
        _setup(page, base_url, created, deleted)

        page.get_by_test_id("compare-models-primary-a").select_option("aged")
        _version_select(page).select_option(VERSIONS[0]["versionId"])

        wait_until(page, lambda: created == [f"aged@{VERSIONS[0]['versionId']}"], "version was not materialised")
        expect(page.get_by_test_id("compare-models-version-note-a")).to_be_visible()

    def test_going_back_to_the_live_model_deletes_the_copy(self, page: Page, base_url: str):
        """A preview is a full copy of the weights; nothing should hold one it isn't using."""
        created, deleted = [], []
        _setup(page, base_url, created, deleted)

        page.get_by_test_id("compare-models-primary-a").select_option("aged")
        _version_select(page).select_option(VERSIONS[0]["versionId"])
        wait_until(page, lambda: created, "version was not materialised")

        _version_select(page).select_option("")

        wait_until(page, lambda: deleted == created, "preview was not cleaned up")
        expect(page.get_by_test_id("compare-models-version-note-a")).to_be_hidden()

    def test_closing_the_dialog_cleans_up_a_selected_version(self, page: Page, base_url: str):
        created, deleted = [], []
        _setup(page, base_url, created, deleted)

        page.get_by_test_id("compare-models-primary-a").select_option("aged")
        _version_select(page).select_option(VERSIONS[0]["versionId"])
        wait_until(page, lambda: created, "version was not materialised")

        page.get_by_role("button", name="Close").last.click()

        wait_until(page, lambda: deleted == created, "closing the dialog left the preview behind")

    def test_both_sides_on_the_same_version_share_one_copy_and_free_it_once(self, page: Page, base_url: str):
        """The server hands both panels the same materialised copy.

        Deleting it when the first panel lets go would pull the weights out from
        under the second one mid-run, so the dialog refcounts.
        """
        created, deleted = [], []
        _setup(page, base_url, created, deleted)

        for side in ("a", "b"):
            page.get_by_test_id(f"compare-models-primary-{side}").select_option("aged")
            _version_select(page, side).select_option(VERSIONS[0]["versionId"])
            wait_until(page, lambda side=side: len(created) >= (1 if side == "a" else 2), "not materialised")

        # A releases; B is still holding the same name, so nothing is deleted yet.
        _version_select(page, "a").select_option("")
        page.wait_for_timeout(500)
        assert deleted == []

        _version_select(page, "b").select_option("")
        wait_until(page, lambda: len(deleted) == 1, "the last holder did not free the preview")
