"""Tests for the model History modal — the rollback path in the browser.

Only the wiring that needs a browser is asserted here: that History is offered
for a selected model, that the list renders what the server returned, and that
Restore posts to the version the row is for. The archiving rules themselves are
covered in `tests/helpers/test_model_versions.py`.
"""

import json

from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

FAKE_VERSIONS = [
    {
        "versionId": "20260901T101500Z",
        "archivedAt": "2026-09-05T12:00:00Z",
        "createdAt": "2026-09-01T10:15:00Z",
        "bytes": 28_000_000,
        "files": ["cats.onnx", "cats.json"],
    },
    {
        "versionId": "20260820T090000Z",
        "archivedAt": "2026-09-01T10:15:00Z",
        "createdAt": "2026-08-20T09:00:00Z",
        "bytes": 27_500_000,
        "files": ["cats.onnx", "cats.json"],
    },
]


def _mock_models(page: Page, models=("cats",)):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/models"),
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "models": list(models)}),
            )
            if route.request.method == "GET"
            else route.continue_()
        ),
    )
    page.route(
        "**/api/models/classes/bulk",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "models": {m: {} for m in models}}),
        ),
    )
    page.route(
        "**/api/models/*/device",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "device": "auto"}),
        ),
    )


def _mock_versions(page: Page, versions=None):
    page.route(
        "**/api/models/*/versions",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {"status": "success", "modelName": "cats", "versions": FAKE_VERSIONS if versions is None else versions}
            ),
        ),
    )


def _open_models_pane(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-tag-management']").click()
    page.wait_for_selector("#model-labels-box")


def _select_model(page: Page):
    page.locator("#model-labels-box select").first.select_option("cats")
    page.wait_for_timeout(700)


def _history_btn(page: Page):
    # An emoji button: the glyph is the label, so the testid is the only handle.
    return page.get_by_test_id("model-history-open")


class TestModelHistoryModal:
    def test_history_is_offered_only_once_a_model_is_selected(self, page: Page, base_url: str):
        """With no model chosen there is no history to show — and no model to roll back."""
        _mock_models(page)
        _mock_versions(page)
        _open_models_pane(page, base_url)
        page.wait_for_timeout(700)
        expect(_history_btn(page)).to_have_count(0)

        _select_model(page)
        expect(_history_btn(page)).to_be_visible()

    def test_versions_render_newest_first_from_the_server(self, page: Page, base_url: str):
        _mock_models(page)
        _mock_versions(page)
        _open_models_pane(page, base_url)
        _select_model(page)
        _history_btn(page).click()

        rows = page.get_by_test_id("model-version-row")
        expect(rows).to_have_count(len(FAKE_VERSIONS))
        expect(rows.first).to_have_attribute("data-version-id", FAKE_VERSIONS[0]["versionId"])
        expect(page.get_by_test_id("model-versions-current")).to_contain_text("2 previous versions")

    def test_a_model_with_no_history_says_so_rather_than_looking_broken(self, page: Page, base_url: str):
        _mock_models(page)
        _mock_versions(page, versions=[])
        _open_models_pane(page, base_url)
        _select_model(page)
        _history_btn(page).click()

        expect(page.get_by_test_id("model-version-row")).to_have_count(0)
        expect(page.get_by_test_id("model-versions-current")).to_contain_text("No previous versions")

    def test_restore_posts_to_the_version_of_the_row_it_was_clicked_on(self, page: Page, base_url: str):
        _mock_models(page)
        _mock_versions(page)

        posted: list = []

        def handle_restore(route: Route):
            posted.append(route.request.url)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "restored": FAKE_VERSIONS[1]["versionId"],
                        "archived": "20260907T000000Z",
                        "files": ["cats.onnx"],
                    }
                ),
            )

        page.route("**/api/models/*/versions/*/restore", handle_restore)

        _open_models_pane(page, base_url)
        _select_model(page)
        _history_btn(page).click()

        # The second row — restoring the *older* of the two, so a handler that
        # always posts the first row's id fails this.
        page.get_by_test_id("model-version-row").nth(1).get_by_test_id("model-version-restore").click()
        page.get_by_role("button", name="Restore", exact=True).last.click()

        wait_until(page, lambda: len(posted) == 1, message="restore was never posted")
        assert FAKE_VERSIONS[1]["versionId"] in posted[0]
        assert FAKE_VERSIONS[0]["versionId"] not in posted[0]

    def test_restore_is_confirmed_before_it_happens(self, page: Page, base_url: str):
        """Rolling weights back changes what every running detection sees."""
        _mock_models(page)
        _mock_versions(page)

        posted: list = []
        page.route(
            "**/api/models/*/versions/*/restore",
            lambda route: (posted.append(route.request.url), route.abort())[1],
        )

        _open_models_pane(page, base_url)
        _select_model(page)
        _history_btn(page).click()
        page.get_by_test_id("model-version-row").first.get_by_test_id("model-version-restore").click()
        page.get_by_role("button", name="Cancel").last.click()
        page.wait_for_timeout(400)

        assert posted == []
