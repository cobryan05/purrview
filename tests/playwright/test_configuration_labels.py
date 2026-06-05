"""
Tests for the Labels configuration UI.

Exercises the renderLabelList() → createLabelList() component rendered inside
Configuration > Labels & Tags > #labels-list:
  - DOM structure (list container present)
  - Label list loading (hierarchical labels appear)
  - Create root label flow (API called with correct payload)
  - Add child label flow
  - Edit label flow
  - Delete flow (confirm dialog; API called or suppressed)

Run with:
    pytest tests/playwright/test_configuration_labels.py -p no:xdist -v
"""
import json

import pytest
from playwright.sync_api import Page, Route, expect

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

ROOT_LABEL = {
    "uuid": "cccccccc-0000-0000-0000-000000000001",
    "name": "Cat",
    "color": "#ff0000",
    "parent_uuid": None,
}

CHILD_LABEL = {
    "uuid": "cccccccc-0000-0000-0000-000000000002",
    "name": "Kitten",
    "color": "#00ff00",
    "parent_uuid": "cccccccc-0000-0000-0000-000000000001",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _navigate_to_labels(page: Page, base_url: str):
    """Open the app and activate Configuration > Labels & Tags."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    # Labels & Tags is the default sub-tab.


def _mock_labels(page: Page, labels: list):
    """Intercept GET /api/labels to return the given label list."""
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": labels}),
        ),
    )


# ---------------------------------------------------------------------------
# DOM structure
# ---------------------------------------------------------------------------


class TestLabelPaneStructure:
    def test_labels_list_container_visible(self, page: Page, base_url: str):
        """The #labels-list div is present after entering Labels & Tags."""
        _navigate_to_labels(page, base_url)
        expect(page.locator("#labels-list")).to_be_visible()

    def test_no_js_errors_on_labels_tab(self, page: Page, base_url: str):
        """No console errors during Labels pane load."""
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        _navigate_to_labels(page, base_url)
        page.wait_for_timeout(1000)
        uncaught = [e for e in errors if "TypeError" in e or "Uncaught" in e]
        assert uncaught == [], f"Uncaught JS errors: {uncaught}"

    def test_add_root_label_button_present(self, page: Page, base_url: str):
        """An 'Add label' button (title='Add label') appears for creating root labels."""
        _mock_labels(page, [])
        _navigate_to_labels(page, base_url)
        # Wait for the label list component to render
        page.wait_for_selector("#labels-list button[title='Add label']", timeout=4000)
        expect(page.locator("#labels-list button[title='Add label']")).to_be_visible()


# ---------------------------------------------------------------------------
# Label list loading
# ---------------------------------------------------------------------------


class TestLabelListLoading:
    def test_root_labels_rendered(self, page: Page, base_url: str):
        """Root labels returned from the API appear as rows with the correct data-uuid."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

    def test_child_labels_rendered(self, page: Page, base_url: str):
        """Child labels appear nested under their parent."""
        _mock_labels(page, [ROOT_LABEL, CHILD_LABEL])
        _navigate_to_labels(page, base_url)
        # Both parent and child should be rendered
        expect(page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")).to_be_visible()
        expect(page.locator(f"#labels-list [data-uuid='{CHILD_LABEL['uuid']}']")).to_be_visible()

    def test_label_name_displayed(self, page: Page, base_url: str):
        """The label name is visible in the rendered row."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_contain_text(ROOT_LABEL["name"])

    def test_each_label_has_delete_button(self, page: Page, base_url: str):
        """Each label row provides a Delete button."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()
        expect(row.locator("button[title='Delete']")).to_have_count(1)

    def test_each_label_has_edit_button(self, page: Page, base_url: str):
        """Each label row provides an Edit button."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()
        expect(row.locator("button[title='Edit']")).to_have_count(1)

    def test_each_label_has_add_child_button(self, page: Page, base_url: str):
        """Each label row provides an 'Add Child' button to create sub-labels."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()
        expect(row.locator("button[title='Add Child']")).to_have_count(1)


# ---------------------------------------------------------------------------
# Create root label flow
# ---------------------------------------------------------------------------


class TestCreateRootLabel:
    def test_create_root_label_calls_api(self, page: Page, base_url: str):
        """Filling a name and clicking the Save button POSTs to /api/labels."""
        calls: list[dict] = []

        def handle_create(route: Route):
            if route.request.method == "POST":
                calls.append(json.loads(route.request.post_data))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "label": ROOT_LABEL}),
                )
            else:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labels": []}),
                )

        _mock_labels(page, [])
        page.route(
            lambda url: url.split("?")[0].endswith("/api/labels"),
            handle_create,
        )

        _navigate_to_labels(page, base_url)
        page.wait_for_selector("#labels-list input[placeholder='New Root Label Name']", timeout=4000)

        # Fill in the label name (the root-label create form is always visible)
        page.locator("#labels-list input[placeholder='New Root Label Name']").fill("Dog")

        # Click the 'Add label' button to submit (or the Save button inside the field)
        page.locator("#labels-list button[title='Add label']").click()
        page.wait_for_timeout(600)

        assert calls, "Expected POST to /api/labels"
        assert calls[0]["name"] == "Dog"
        assert calls[0].get("parentUuid") is None

    def test_create_root_label_empty_name_blocked(self, page: Page, base_url: str):
        """Clicking Save with an empty name does not call the API."""
        calls: list[bool] = []

        def handle_create(route: Route):
            if route.request.method == "POST":
                calls.append(True)
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps({"status": "success", "label": ROOT_LABEL}))
            else:
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps({"status": "success", "labels": []}))

        _mock_labels(page, [])
        page.route(lambda url: url.split("?")[0].endswith("/api/labels"), handle_create)

        _navigate_to_labels(page, base_url)
        # The create form is always visible; leave the name field empty
        page.wait_for_selector("#labels-list input[placeholder='New Root Label Name']", timeout=4000)

        # Click Add label without filling in a name
        page.locator("#labels-list button[title='Add label']").click()
        page.wait_for_timeout(500)

        assert not calls, "API must NOT be called when name is empty"


# ---------------------------------------------------------------------------
# Add child label flow
# ---------------------------------------------------------------------------


class TestAddChildLabel:
    def test_add_child_opens_inline_form(self, page: Page, base_url: str):
        """Clicking 'Add Child' on a label opens an inline child-create form."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

        row.locator("button[title='Add Child']").click()
        page.wait_for_timeout(400)

        # An inline input should appear somewhere after the parent row
        page.wait_for_selector("#labels-list input[placeholder='New Label Name']", timeout=3000)
        expect(page.locator("#labels-list input[placeholder='New Label Name']")).to_be_visible()

    def test_create_child_label_sends_parent_uuid(self, page: Page, base_url: str):
        """Creating a child label sends the parent's UUID in the API payload."""
        calls: list[dict] = []

        def handle_create(route: Route):
            if route.request.method == "POST":
                calls.append(json.loads(route.request.post_data))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "label": CHILD_LABEL}),
                )
            else:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labels": [ROOT_LABEL]}),
                )

        _mock_labels(page, [ROOT_LABEL])
        page.route(lambda url: url.split("?")[0].endswith("/api/labels"), handle_create)

        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

        row.locator("button[title='Add Child']").click()
        page.wait_for_selector("#labels-list input[placeholder='New Label Name']", timeout=3000)
        page.locator("#labels-list input[placeholder='New Label Name']").fill("Kitten")
        page.locator("#labels-list button[title='Save']").first.click()
        page.wait_for_timeout(600)

        assert calls, "Expected POST to /api/labels"
        assert calls[0].get("parentUuid") == ROOT_LABEL["uuid"]
        assert calls[0]["name"] == "Kitten"


# ---------------------------------------------------------------------------
# Delete label flow
# ---------------------------------------------------------------------------


class TestDeleteLabel:
    def test_delete_shows_confirm_dialog(self, page: Page, base_url: str):
        """Clicking Delete on a label opens a browser confirm() dialog."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

        dialogs: list[str] = []
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
        row.locator("button[title='Delete']").click()
        page.wait_for_timeout(400)

        assert dialogs, "Expected a confirm() dialog"

    def test_delete_accepted_calls_api(self, page: Page, base_url: str):
        """Accepting delete sends a DELETE request to /api/labels/{uuid}."""
        calls: list[str] = []

        def handle_all(route: Route):
            if route.request.method == "DELETE":
                calls.append(route.request.url)
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success"}),
                )
            else:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labels": [ROOT_LABEL]}),
                )

        _mock_labels(page, [ROOT_LABEL])
        page.route("**/api/labels/**", handle_all)

        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

        page.on("dialog", lambda d: d.accept())
        row.locator("button[title='Delete']").click()
        page.wait_for_timeout(600)

        assert calls, "Expected DELETE request to /api/labels/{uuid}"
        assert ROOT_LABEL["uuid"] in calls[0]

    def test_delete_dismissed_does_not_call_api(self, page: Page, base_url: str):
        """Dismissing the delete dialog does NOT send a DELETE request."""
        calls: list[bool] = []
        page.route("**/api/labels/**", lambda route: (
            calls.append(True) if route.request.method == "DELETE" else None,
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps({"status": "success", "labels": [ROOT_LABEL]})),
        ))
        _mock_labels(page, [ROOT_LABEL])

        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

        page.on("dialog", lambda d: d.dismiss())
        row.locator("button[title='Delete']").click()
        page.wait_for_timeout(500)

        assert not calls, "API must NOT be called when delete is cancelled"
