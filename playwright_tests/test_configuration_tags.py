"""
Tests for the Tags configuration UI.

Exercises the renderTagList() component rendered inside
Configuration > Labels & Tags > tags-list:
  - DOM structure (add/refresh buttons present)
  - Tag list loading (user tags and system tags)
  - Create flow (name required; API called with correct payload)
  - Delete flow (confirm dialog; API called or suppressed)

Run with:
    pytest tests/playwright/test_configuration_tags.py -p no:xdist -v
"""
import json

import pytest
from playwright.sync_api import Page, Route, expect

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

FAKE_TAG = {
    "uuid": "aaaaaaaa-0000-0000-0000-000000000001",
    "name": "Fluffy",
    "color": "#ff5500",
    "icon": None,
    "kind": "user",
}

SYSTEM_TAG = {
    "uuid": "00000000-0000-0000-0000-000000000001",
    "name": "Unverified",
    "color": "#888888",
    "icon": None,
    "kind": "system",
}

# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------


def _navigate_to_tags(page: Page, base_url: str):
    """Open the app and activate Configuration > Labels & Tags."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    # Labels & Tags is the default workspace sub-tab — already active.


def _mock_tags(page: Page, tags: list):
    """Intercept /api/tags/list to return the given tag list."""
    page.route(
        "**/api/tags/list",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"tags": tags}),
        ),
    )


# ---------------------------------------------------------------------------
# DOM structure
# ---------------------------------------------------------------------------


class TestTagPaneStructure:
    def test_pane_structure(self, page: Page, base_url: str):
        """The tags pane renders its container, controls, and inputs in one navigation."""
        _navigate_to_tags(page, base_url)
        expect(page.locator("#tags-list")).to_be_visible()
        page.wait_for_selector("#tags-list button[title='Add tag']", timeout=4000)
        expect(page.locator("#tags-list button[title='Add tag']")).to_be_visible()
        expect(page.locator("#tags-list button[title='Refresh']")).to_be_visible()
        expect(page.locator("#tags-list input[placeholder='New Tag Name']")).to_be_visible()

    def test_no_js_errors_on_tags_tab(self, page: Page, base_url: str):
        """No console errors while loading the Tags pane."""
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        _navigate_to_tags(page, base_url)
        page.wait_for_selector("#tags-list button[title='Add tag']", timeout=4000)
        assert errors == [], f"Console errors: {errors}"


# ---------------------------------------------------------------------------
# Tag list loading
# ---------------------------------------------------------------------------


class TestTagListLoading:
    def test_user_tags_rendered_as_rows(self, page: Page, base_url: str):
        """Tags returned from /api/tags/list appear as rows with the correct data-uuid."""
        _mock_tags(page, [FAKE_TAG])
        _navigate_to_tags(page, base_url)
        row = page.locator(f"#tags-list [data-uuid='{FAKE_TAG['uuid']}']")
        expect(row).to_be_visible()

    def test_system_tags_are_rendered(self, page: Page, base_url: str):
        """System tags are shown in the list."""
        _mock_tags(page, [SYSTEM_TAG])
        _navigate_to_tags(page, base_url)
        row = page.locator(f"#tags-list [data-uuid='{SYSTEM_TAG['uuid']}']")
        expect(row).to_be_visible()

    def test_system_tags_have_no_delete_button(self, page: Page, base_url: str):
        """System tags do not expose a Delete button."""
        _mock_tags(page, [SYSTEM_TAG])
        _navigate_to_tags(page, base_url)
        row = page.locator(f"#tags-list [data-uuid='{SYSTEM_TAG['uuid']}']")
        expect(row).to_be_visible()
        expect(row.locator("button[title='Delete']")).to_have_count(0)

    def test_user_tags_have_delete_button(self, page: Page, base_url: str):
        """User (non-system) tags expose a Delete button."""
        _mock_tags(page, [FAKE_TAG])
        _navigate_to_tags(page, base_url)
        row = page.locator(f"#tags-list [data-uuid='{FAKE_TAG['uuid']}']")
        expect(row).to_be_visible()
        expect(row.locator("button[title='Delete']")).to_have_count(1)

    def test_multiple_tags_all_rendered(self, page: Page, base_url: str):
        """All tags returned from the API are rendered."""
        tags = [
            {**FAKE_TAG, "uuid": f"aaaaaaaa-0000-0000-0000-00000000000{i}", "name": f"Tag{i}"}
            for i in range(1, 4)
        ]
        _mock_tags(page, tags)
        _navigate_to_tags(page, base_url)
        for tag in tags:
            expect(page.locator(f"#tags-list [data-uuid='{tag['uuid']}']")).to_be_visible()

    def test_api_failure_does_not_crash_page(self, page: Page, base_url: str):
        """A 500 from /api/tags/list doesn't produce uncaught JS exceptions."""
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.route(
            "**/api/tags/list",
            lambda route: route.fulfill(
                status=500,
                content_type="application/json",
                body=json.dumps({"message": "DB error"}),
            ),
        )
        _navigate_to_tags(page, base_url)
        page.wait_for_timeout(1000)
        # The container stays visible; no uncaught TypeError etc.
        expect(page.locator("#tags-list")).to_be_visible()
        uncaught = [e for e in errors if "TypeError" in e or "Uncaught" in e]
        assert uncaught == [], f"Uncaught JS errors: {uncaught}"


# ---------------------------------------------------------------------------
# Create flow
# ---------------------------------------------------------------------------


class TestTagCreate:
    def test_create_tag_calls_add_endpoint(self, page: Page, base_url: str):
        """Filling in a name and clicking 'Add tag' POSTs to /api/tags/add."""
        calls: list[dict] = []

        def handle_add(route: Route):
            body = json.loads(route.request.post_data)
            calls.append(body)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "tag": {**FAKE_TAG, "name": body["name"]}}),
            )

        _mock_tags(page, [])
        page.route("**/api/tags/add", handle_add)

        _navigate_to_tags(page, base_url)
        page.wait_for_selector("#tags-list input[placeholder='New Tag Name']", timeout=4000)

        page.locator("#tags-list input[placeholder='New Tag Name']").fill("Whiskers")
        page.locator("#tags-list button[title='Add tag']").click()
        page.wait_for_timeout(600)

        assert calls, "Expected POST to /api/tags/add"
        assert calls[0]["name"] == "Whiskers"

    def test_create_tag_empty_name_blocked(self, page: Page, base_url: str):
        """Clicking 'Add tag' with an empty name does not call /api/tags/add."""
        calls: list[bool] = []
        page.route("**/api/tags/add", lambda route: (calls.append(True), route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "tag": FAKE_TAG}),
        )))
        _mock_tags(page, [])

        _navigate_to_tags(page, base_url)
        page.wait_for_selector("#tags-list button[title='Add tag']", timeout=4000)

        # Leave the name field empty and click Add tag
        page.locator("#tags-list button[title='Add tag']").click()
        page.wait_for_timeout(500)

        assert not calls, "API must NOT be called when name is empty"

    def test_create_tag_sends_color(self, page: Page, base_url: str):
        """The create call includes a color field."""
        calls: list[dict] = []

        def handle_add(route: Route):
            body = json.loads(route.request.post_data)
            calls.append(body)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "tag": FAKE_TAG}),
            )

        _mock_tags(page, [])
        page.route("**/api/tags/add", handle_add)

        _navigate_to_tags(page, base_url)
        page.wait_for_selector("#tags-list input[placeholder='New Tag Name']", timeout=4000)
        page.locator("#tags-list input[placeholder='New Tag Name']").fill("Mittens")
        page.locator("#tags-list button[title='Add tag']").click()
        page.wait_for_timeout(600)

        assert calls, "Expected POST to /api/tags/add"
        assert "color" in calls[0], "Payload must include 'color'"

    def test_refresh_button_clears_cache_and_rerenders(self, page: Page, base_url: str):
        """Clicking Refresh re-fetches the tag list from the API."""
        call_count = [0]

        def handle_list(route: Route):
            call_count[0] += 1
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"tags": []}),
            )

        page.route("**/api/tags/list", handle_list)
        _navigate_to_tags(page, base_url)
        page.wait_for_selector("#tags-list button[title='Refresh']", timeout=4000)

        calls_before = call_count[0]
        page.locator("#tags-list button[title='Refresh']").click()
        page.wait_for_timeout(600)

        assert call_count[0] > calls_before, "Refresh should trigger a new /api/tags/list request"


# ---------------------------------------------------------------------------
# Delete flow
# ---------------------------------------------------------------------------


class TestTagDelete:
    def test_delete_shows_confirm_dialog(self, page: Page, base_url: str):
        """Clicking Delete on a user tag opens a browser confirm() dialog."""
        _mock_tags(page, [FAKE_TAG])
        _navigate_to_tags(page, base_url)
        row = page.locator(f"#tags-list [data-uuid='{FAKE_TAG['uuid']}']")
        expect(row).to_be_visible()

        dialogs: list[str] = []
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
        row.locator("button[title='Delete']").click()
        page.wait_for_timeout(400)

        assert dialogs, "Expected a confirm() dialog"
        assert FAKE_TAG["name"] in dialogs[0], "Dialog should mention the tag name"

    def test_delete_accepted_calls_api(self, page: Page, base_url: str):
        """Accepting the confirm dialog sends DELETE payload to /api/tags/delete."""
        calls: list[dict] = []

        def handle_delete(route: Route):
            calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success"}),
            )

        _mock_tags(page, [FAKE_TAG])
        page.route("**/api/tags/delete", handle_delete)

        _navigate_to_tags(page, base_url)
        row = page.locator(f"#tags-list [data-uuid='{FAKE_TAG['uuid']}']")
        expect(row).to_be_visible()

        page.on("dialog", lambda d: d.accept())
        row.locator("button[title='Delete']").click()
        page.wait_for_timeout(600)

        assert calls, "Expected POST to /api/tags/delete"
        assert calls[0].get("tagUuid") == FAKE_TAG["uuid"]

    def test_delete_dismissed_does_not_call_api(self, page: Page, base_url: str):
        """Dismissing the confirm dialog does NOT send a delete request."""
        calls: list[bool] = []
        page.route("**/api/tags/delete", lambda route: (calls.append(True), route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success"}),
        )))
        _mock_tags(page, [FAKE_TAG])

        _navigate_to_tags(page, base_url)
        row = page.locator(f"#tags-list [data-uuid='{FAKE_TAG['uuid']}']")
        expect(row).to_be_visible()

        page.on("dialog", lambda d: d.dismiss())
        row.locator("button[title='Delete']").click()
        page.wait_for_timeout(500)

        assert not calls, "API must NOT be called when delete is cancelled"
