"""
Tests for the Dataset Management configuration UI.

Exercises renderDatasetManager() inside
Configuration > Dataset Management > #dataset-management-box:
  - DOM structure (container present, create form visible)
  - Dataset list loading (datasets appear with edit/delete buttons)
  - Create flow (fill name → Create Dataset → /api/datasets POST called)
  - Delete flow (confirm dialog; API called or suppressed)
  - Label checkbox picker (Select All / Clear All controls)

Run with:
    pytest tests/playwright/test_configuration_datasets.py -p no:xdist -v
"""
import json

import pytest
from playwright.sync_api import Page, Route, expect

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

FAKE_DATASET = {
    "uuid": "11111111-aaaa-0000-0000-000000000001",
    "name": "Training Set Alpha",
    "description": "Main training dataset",
    "label_uuids": [],
    "filter_json": {"conditions": []},
}

ROOT_LABEL = {
    "uuid": "cccccccc-0000-0000-0000-000000000001",
    "name": "Cat",
    "color": "#ff0000",
    "parent_uuid": None,
}

# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------


def _navigate_to_datasets(page: Page, base_url: str):
    """Open the app and activate Configuration > Dataset Management."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-dataset-management']").click()
    page.wait_for_load_state("networkidle")


def _mock_datasets(page: Page, datasets: list | None = None):
    """Intercept GET /api/datasets to return the given datasets list."""
    datasets = datasets or []
    datasets_map = {d["uuid"]: d for d in datasets}
    page.route(
        lambda url: url.split("?")[0].endswith("/api/datasets") and not any(
            s in url for s in ["/images", "/labels", "/preview"]
        ),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "datasets": datasets}),
        )
        if route.request.method == "GET"
        else route.continue_(),
    )


def _mock_labels(page: Page, labels: list | None = None):
    """Intercept GET /api/labels."""
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "/api/labels/" not in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": labels or []}),
        )
        if route.request.method == "GET"
        else route.continue_(),
    )


def _mock_tags_empty(page: Page):
    """Return empty tags list for the dataset label picker."""
    page.route(
        "**/api/tags/list",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"tags": []}),
        ),
    )


# ---------------------------------------------------------------------------
# DOM structure
# ---------------------------------------------------------------------------


class TestDatasetPaneStructure:
    def test_pane_structure(self, page: Page, base_url: str):
        """The dataset pane renders its container and creation affordances in one navigation."""
        _mock_datasets(page, [])
        _navigate_to_datasets(page, base_url)
        expect(page.locator("#dataset-management-box")).to_be_visible()

        page.wait_for_timeout(1000)
        content = page.locator("#dataset-management-box").inner_text()
        assert "New Dataset" in content or "Create Dataset" in content or "Dataset name" in content, (
            f"No dataset creation affordance found. Content: {content!r}"
        )

        # Click "+ New Dataset" if it exists to reveal the create form
        new_btn = page.locator("#dataset-management-box button", has_text="New Dataset")
        if new_btn.count() > 0:
            new_btn.first.click()
            page.wait_for_timeout(500)

        page.wait_for_selector(
            "#dataset-management-box button, #dataset-management-box input[placeholder='Dataset name']",
            timeout=3000,
        )
        content = page.locator("#dataset-management-box").inner_text()
        assert "Create Dataset" in content or "Dataset name" in content, (
            f"No 'Create Dataset' button or name input. Content: {content!r}"
        )

    def test_no_js_errors_on_datasets_tab(self, page: Page, base_url: str):
        """No console errors while loading the Dataset Management pane."""
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        _navigate_to_datasets(page, base_url)
        page.wait_for_timeout(1200)
        uncaught = [e for e in errors if "TypeError" in e or "Uncaught" in e]
        assert uncaught == [], f"Uncaught JS errors: {uncaught}"


# ---------------------------------------------------------------------------
# Dataset list loading
# ---------------------------------------------------------------------------


class TestDatasetListLoading:
    def test_existing_datasets_shown(self, page: Page, base_url: str):
        """Datasets returned from /api/datasets appear in the list."""
        _mock_datasets(page, [FAKE_DATASET])
        _navigate_to_datasets(page, base_url)
        page.wait_for_timeout(1000)
        content = page.locator("#dataset-management-box").inner_text()
        assert FAKE_DATASET["name"] in content, (
            f"Dataset name '{FAKE_DATASET['name']}' not found in: {content!r}"
        )

    def test_dataset_has_delete_affordance(self, page: Page, base_url: str):
        """Clicking a dataset name opens a detail panel that contains a Delete button."""
        _mock_datasets(page, [FAKE_DATASET])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        _navigate_to_datasets(page, base_url)
        page.wait_for_timeout(1000)

        # The dataset nameLine div has exactly the dataset name as text;
        # clicking it (via event bubbling) triggers the row's click handler.
        page.get_by_text(FAKE_DATASET["name"], exact=True).first.click()
        page.wait_for_timeout(600)

        # The detail panel has a red "Delete" button
        delete_btn = page.locator("#dataset-management-box button", has_text="Delete")
        expect(delete_btn.first).to_be_visible()

    def test_multiple_datasets_all_shown(self, page: Page, base_url: str):
        """Multiple datasets are all rendered in the list."""
        datasets = [
            {**FAKE_DATASET, "uuid": f"11111111-aaaa-0000-0000-00000000000{i}", "name": f"Dataset {i}"}
            for i in range(1, 4)
        ]
        _mock_datasets(page, datasets)
        _navigate_to_datasets(page, base_url)
        page.wait_for_timeout(1000)
        content = page.locator("#dataset-management-box").inner_text()
        for ds in datasets:
            assert ds["name"] in content, f"Dataset '{ds['name']}' not in content: {content!r}"


# ---------------------------------------------------------------------------
# Create dataset flow
# ---------------------------------------------------------------------------


class TestCreateDataset:
    def test_create_dataset_calls_api(self, page: Page, base_url: str):
        """Filling in a name and clicking Create Dataset POSTs to /api/datasets."""
        create_calls: list[dict] = []

        def handle_create(route: Route):
            if route.request.method == "POST" and not any(
                s in route.request.url for s in ["/images", "/labels", "/preview"]
            ):
                create_calls.append(json.loads(route.request.post_data))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "dataset": FAKE_DATASET}),
                )
            else:
                route.continue_()

        _mock_datasets(page, [])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        page.route(
            lambda url: "/api/datasets" in url,
            handle_create,
        )

        _navigate_to_datasets(page, base_url)
        page.wait_for_timeout(800)

        # Reveal form if hidden behind a "New Dataset" button
        new_btn = page.locator("#dataset-management-box button", has_text="New Dataset")
        if new_btn.count() > 0:
            new_btn.first.click()
            page.wait_for_timeout(500)

        # Fill the name
        page.wait_for_selector(
            "#dataset-management-box input[placeholder='Dataset name']", timeout=4000
        )
        page.locator("#dataset-management-box input[placeholder='Dataset name']").fill("My New Dataset")

        # Click Create Dataset
        page.locator("#dataset-management-box button", has_text="Create Dataset").click()
        page.wait_for_timeout(800)

        assert create_calls, "Expected POST to /api/datasets"
        payload = create_calls[0]
        assert payload.get("name") == "My New Dataset"

    def test_create_dataset_name_required(self, page: Page, base_url: str):
        """Clicking Create Dataset without a name does not call the API."""
        calls: list[bool] = []

        def handle_create(route: Route):
            if route.request.method == "POST":
                calls.append(True)
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "dataset": FAKE_DATASET}),
                )
            else:
                route.continue_()

        _mock_datasets(page, [])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        page.route(lambda url: "/api/datasets" in url and "/preview" not in url, handle_create)

        _navigate_to_datasets(page, base_url)
        page.wait_for_timeout(800)

        new_btn = page.locator("#dataset-management-box button", has_text="New Dataset")
        if new_btn.count() > 0:
            new_btn.first.click()
            page.wait_for_timeout(500)

        # Click Create Dataset without filling the name
        create_btn = page.locator("#dataset-management-box button", has_text="Create Dataset")
        if create_btn.count() > 0:
            create_btn.first.click()
            page.wait_for_timeout(500)
            assert not calls, "API must NOT be called when name is empty"

    def test_label_checkbox_picker_shows_labels(self, page: Page, base_url: str):
        """Labels from /api/labels appear as checkboxes in the create form."""
        _mock_datasets(page, [])
        _mock_labels(page, [ROOT_LABEL])
        _mock_tags_empty(page)

        _navigate_to_datasets(page, base_url)
        page.wait_for_timeout(800)

        new_btn = page.locator("#dataset-management-box button", has_text="New Dataset")
        if new_btn.count() > 0:
            new_btn.first.click()
            page.wait_for_timeout(600)

        # The label should appear as a checkbox label somewhere in the box
        page.wait_for_timeout(800)
        content = page.locator("#dataset-management-box").inner_text()
        assert ROOT_LABEL["name"] in content, (
            f"Label '{ROOT_LABEL['name']}' not found in dataset form. Content: {content!r}"
        )

    def test_label_select_all_button_present(self, page: Page, base_url: str):
        """A 'Select All' button is visible in the label picker."""
        _mock_datasets(page, [])
        _mock_labels(page, [ROOT_LABEL])
        _mock_tags_empty(page)

        _navigate_to_datasets(page, base_url)
        page.wait_for_timeout(800)

        new_btn = page.locator("#dataset-management-box button", has_text="New Dataset")
        if new_btn.count() > 0:
            new_btn.first.click()
            page.wait_for_timeout(600)

        page.wait_for_timeout(600)
        select_all = page.locator("#dataset-management-box button", has_text="Select All")
        expect(select_all.first).to_be_visible()


# ---------------------------------------------------------------------------
# Delete dataset flow
# ---------------------------------------------------------------------------


class TestDeleteDataset:
    def _open_detail_panel(self, page: Page):
        """Click the bold dataset name div to reveal the detail panel with Save/Delete buttons."""
        page.wait_for_timeout(800)
        # The nameLine div has exactly the dataset name as its text content.
        # Clicking it bubbles to the row's click handler which calls showDetail().
        page.get_by_text(FAKE_DATASET["name"], exact=True).first.click()
        page.wait_for_timeout(600)

    def test_delete_shows_confirm_dialog(self, page: Page, base_url: str):
        """Clicking Delete in the dataset detail panel opens a confirm() dialog."""
        _mock_datasets(page, [FAKE_DATASET])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        _navigate_to_datasets(page, base_url)
        self._open_detail_panel(page)

        delete_btn = page.locator("#dataset-management-box button", has_text="Delete")
        expect(delete_btn.first).to_be_visible()

        dialogs: list[str] = []
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
        delete_btn.first.click()
        page.wait_for_timeout(400)

        assert dialogs, "Expected a confirm() dialog"
        assert FAKE_DATASET["name"] in dialogs[0], "Dialog should mention dataset name"

    def test_delete_accepted_calls_api(self, page: Page, base_url: str):
        """Accepting the delete dialog calls DELETE /api/datasets/{uuid}."""
        delete_calls: list[str] = []

        def handle_delete(route: Route):
            if route.request.method == "DELETE":
                delete_calls.append(route.request.url)
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success"}),
                )
            else:
                route.continue_()

        _mock_datasets(page, [FAKE_DATASET])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        page.route("**/api/datasets/**", handle_delete)

        _navigate_to_datasets(page, base_url)
        self._open_detail_panel(page)

        delete_btn = page.locator("#dataset-management-box button", has_text="Delete")
        expect(delete_btn.first).to_be_visible()

        page.on("dialog", lambda d: d.accept())
        delete_btn.first.click()
        page.wait_for_timeout(600)

        assert delete_calls, "Expected DELETE request to /api/datasets/{uuid}"
        assert FAKE_DATASET["uuid"] in delete_calls[0]

    def test_delete_dismissed_does_not_call_api(self, page: Page, base_url: str):
        """Dismissing the confirm dialog does NOT delete the dataset."""
        calls: list[bool] = []

        def handle_delete(route: Route):
            if route.request.method == "DELETE":
                calls.append(True)
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps({"status": "success"}))
            else:
                route.continue_()

        _mock_datasets(page, [FAKE_DATASET])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        page.route("**/api/datasets/**", handle_delete)

        _navigate_to_datasets(page, base_url)
        self._open_detail_panel(page)

        delete_btn = page.locator("#dataset-management-box button", has_text="Delete")
        expect(delete_btn.first).to_be_visible()

        page.on("dialog", lambda d: d.dismiss())
        delete_btn.first.click()
        page.wait_for_timeout(500)

        assert not calls, "API must NOT be called when delete is dismissed"
