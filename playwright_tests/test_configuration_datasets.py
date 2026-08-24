"""
Tests for the Dataset Management configuration UI.

Exercises renderDatasetManager() inside
Configuration > Dataset Management > #dataset-management-box:
  - DOM structure (list panel + editor panel, create form visible by default)
  - Dataset list loading (datasets appear in the list)
  - Create flow (fill name → Create → /api/datasets POST called)
  - Delete flow (modal confirm; API called or suppressed)
  - Label checkbox picker (Select All / Clear All controls)

Run with:
    pytest playwright_tests/test_configuration_datasets.py -p no:xdist -p no:asyncio -v
"""

import json
import re

import pytest
from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

FAKE_DATASET = {
    "uuid": "11111111-aaaa-0000-0000-000000000001",
    "name": "Training Set Alpha",
    "description": "Main training dataset",
    "labelUuids": [],
    "filterJson": {"conditions": []},
    "promoted": True,
}

FAKE_VIEW = {
    "uuid": "22222222-bbbb-0000-0000-000000000001",
    "name": "Unreviewed Cats",
    "description": "Saved search from the Review tab",
    "labelUuids": [],
    "filterJson": {"conditions": []},
    "promoted": False,
}

ROOT_LABEL = {
    "uuid": "cccccccc-0000-0000-0000-000000000001",
    "name": "Cat",
    "color": "#ff0000",
    "parentUuid": None,
}

# ---------------------------------------------------------------------------
# Navigation / mocking helpers
# ---------------------------------------------------------------------------


def _navigate_to_datasets(page: Page, base_url: str):
    """Open the app and activate Configuration > Dataset Management."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-dataset-management']").click()
    page.wait_for_load_state("networkidle")
    # New DOM: panel-box structure
    page.wait_for_selector("#dataset-management-box .ds-list-panel")


def _mock_datasets(page: Page, datasets: list | None = None):
    """Intercept GET /api/datasets to return the given datasets list."""
    datasets = datasets or []
    page.route(
        lambda url: url.split("?")[0].endswith("/api/datasets")
        and not any(s in url for s in ["/images", "/labels", "/preview"]),
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "datasets": datasets}),
            )
            if route.request.method == "GET"
            else route.continue_()
        ),
    )


def _mock_labels(page: Page, labels: list | None = None):
    """Intercept GET /api/labels."""
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "/api/labels/" not in url,
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "labels": labels or []}),
            )
            if route.request.method == "GET"
            else route.continue_()
        ),
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


def _editor(page: Page):
    """The editor panel (right side)."""
    return page.locator("#dataset-management-box .ds-editor")


def _editor_bar(page: Page):
    """The sticky header bar of the editor panel."""
    return page.locator("#dataset-management-box .ds-editor .panel-box__bar")


def _editor_body(page: Page):
    """The scrollable body of the editor panel."""
    return page.locator("#dataset-management-box .ds-editor .panel-box__body")


def _open_detail_panel(page: Page):
    """Click the bold dataset name item to reveal the detail panel with Save/Delete buttons."""
    page.wait_for_timeout(800)
    # The ds-list-item__name div has exactly the dataset name as its text content.
    page.get_by_text(FAKE_DATASET["name"], exact=True).first.click()
    page.wait_for_timeout(600)


# ---------------------------------------------------------------------------
# DOM structure
# ---------------------------------------------------------------------------


class TestDatasetPaneStructure:
    def test_pane_structure(self, page: Page, base_url: str):
        """The dataset pane renders list panel + editor panel; create form is shown by default."""
        _mock_datasets(page, [])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        _navigate_to_datasets(page, base_url)

        # Left panel: list panel exists
        expect(page.locator("#dataset-management-box .ds-list-panel")).to_be_visible()

        # Footer has a "+ New" button
        expect(page.locator("#dataset-management-box .ds-list-panel button", has_text="+ New")).to_be_visible()

        # Editor panel defaults to create form; title is in the panel bar
        expect(_editor_bar(page)).to_contain_text("New Dataset")

        # Create button is in the editor bar (disabled until name entered)
        expect(_editor_bar(page).locator("button", has_text="Create")).to_be_visible()

        # Name input with "Required" placeholder is in the editor body
        expect(_editor_body(page).locator("input[placeholder='Required']")).to_be_visible()

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
        list_item = page.locator("#dataset-management-box [data-dataset-uuid]")
        expect(list_item.locator(".ds-list-item__name")).to_have_text(FAKE_DATASET["name"])

    def test_dataset_has_delete_affordance(self, page: Page, base_url: str):
        """Clicking a dataset item opens the edit form in the editor panel with a Delete button."""
        _mock_datasets(page, [FAKE_DATASET])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        _navigate_to_datasets(page, base_url)
        _open_detail_panel(page)

        # The editor bar should show the dataset name and a Delete button
        expect(_editor_bar(page)).to_contain_text(FAKE_DATASET["name"])
        expect(_editor_bar(page).locator("button", has_text="Delete")).to_be_visible()

    def test_multiple_datasets_all_shown(self, page: Page, base_url: str):
        """Multiple datasets are all rendered in the list."""
        datasets = [
            {**FAKE_DATASET, "uuid": f"11111111-aaaa-0000-0000-00000000000{i}", "name": f"Dataset {i}"}
            for i in range(1, 4)
        ]
        _mock_datasets(page, datasets)
        _navigate_to_datasets(page, base_url)
        # `inner_text()` takes one snapshot and does not retry, so the 1s sleep it sat
        # behind was the only thing making this pass. `expect` retries instead.
        panel = page.locator("#dataset-management-box .ds-list-panel")
        for ds in datasets:
            expect(panel).to_contain_text(ds["name"])


# ---------------------------------------------------------------------------
# Create dataset flow
# ---------------------------------------------------------------------------


class TestCreateDataset:
    def test_create_dataset_calls_api(self, page: Page, base_url: str):
        """Filling in a name and clicking Create POSTs to /api/datasets."""
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

        # Create form is visible by default — fill the name input
        _editor_body(page).locator("input[placeholder='Required']").fill("My New Dataset")

        # Click Create in the sticky editor header bar
        _editor_bar(page).locator("button", has_text="Create").click()
        wait_until(page, lambda: create_calls, "Expected POST to /api/datasets")
        payload = create_calls[0]
        assert payload.get("name") == "My New Dataset"

    def test_create_dataset_name_required(self, page: Page, base_url: str):
        """Clicking Create without a name does not call the API."""
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

        # Create button in the sticky bar is disabled until name is entered
        create_btn = _editor_bar(page).locator("button", has_text="Create")
        expect(create_btn).to_be_disabled()
        assert not calls, "API must NOT be called when name is empty"

    def test_label_checkbox_picker_shows_labels(self, page: Page, base_url: str):
        """Labels from /api/labels appear as checkboxes in the create form."""
        _mock_datasets(page, [])
        _mock_labels(page, [ROOT_LABEL])
        _mock_tags_empty(page)

        _navigate_to_datasets(page, base_url)
        page.wait_for_timeout(800)

        # The label should appear somewhere in the editor body
        expect(_editor_body(page)).to_contain_text(ROOT_LABEL["name"])


# ---------------------------------------------------------------------------
# Delete dataset flow
# ---------------------------------------------------------------------------


class TestDeleteDataset:
    def test_delete_shows_confirm_dialog(self, page: Page, base_url: str):
        """Clicking Delete in the editor bar opens a modal dialog naming the dataset."""
        _mock_datasets(page, [FAKE_DATASET])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        _navigate_to_datasets(page, base_url)
        _open_detail_panel(page)

        _editor_bar(page).locator("button", has_text="Delete").click()
        page.wait_for_timeout(300)

        # Custom modal (not window.confirm) — verify it appears with the dataset name
        modal = page.get_by_test_id("modal-shell-dialog")
        expect(modal).to_be_visible()
        expect(modal).to_contain_text(FAKE_DATASET["name"])

    def test_delete_accepted_calls_api(self, page: Page, base_url: str):
        """Confirming the delete modal sends a DELETE to /api/datasets/{uuid}."""
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
        _open_detail_panel(page)

        _editor_bar(page).locator("button", has_text="Delete").click()

        # Confirm by clicking "Delete" inside the modal, once it is actually there.
        modal = page.get_by_test_id("modal-shell-dialog")
        expect(modal).to_be_visible()
        modal.locator("button", has_text="Delete").click()
        wait_until(page, lambda: delete_calls, "Expected DELETE request to /api/datasets/{uuid}")
        assert FAKE_DATASET["uuid"] in delete_calls[0]

    def test_delete_dismissed_does_not_call_api(self, page: Page, base_url: str):
        """Cancelling the delete modal must NOT delete the dataset."""
        calls: list[bool] = []

        def handle_delete(route: Route):
            if route.request.method == "DELETE":
                calls.append(True)
                route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))
            else:
                route.continue_()

        _mock_datasets(page, [FAKE_DATASET])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        page.route("**/api/datasets/**", handle_delete)

        _navigate_to_datasets(page, base_url)
        _open_detail_panel(page)

        _editor_bar(page).locator("button", has_text="Delete").click()
        page.wait_for_timeout(300)

        # Cancel the modal — no API call should happen
        page.get_by_test_id("modal-shell-dialog").locator("button", has_text="Cancel").click()
        page.wait_for_timeout(500)

        assert not calls, "API must NOT be called when delete is cancelled"


# ---------------------------------------------------------------------------
# Views + promotion (slice 5, PR 5.3)
# ---------------------------------------------------------------------------


def _promote_btn(page: Page):
    return page.get_by_test_id("dataset-promote-toggle")


class TestViewsAndPromotion:
    """Views and Datasets are one table separated by `promoted`, and this pane is
    the only screen that shows both — so grouping and the promote/demote flip
    both live here."""

    def _setup(self, page: Page, base_url: str, rows: list):
        _mock_datasets(page, rows)
        _mock_labels(page, [])
        _mock_tags_empty(page)
        _navigate_to_datasets(page, base_url)
        page.wait_for_timeout(900)

    def test_views_render_in_their_own_section(self, page: Page, base_url: str):
        self._setup(page, base_url, [FAKE_DATASET, FAKE_VIEW])

        expect(page.get_by_test_id("dataset-section-datasets")).to_contain_text("(1)")
        expect(page.get_by_test_id("dataset-section-views")).to_contain_text("(1)")

        dataset_item = page.locator("#dataset-management-box [data-dataset-uuid='%s']" % FAKE_DATASET["uuid"])
        view_item = page.locator("#dataset-management-box [data-dataset-uuid='%s']" % FAKE_VIEW["uuid"])
        expect(dataset_item).to_have_attribute("data-promoted", "true")
        expect(view_item).to_have_attribute("data-promoted", "false")

    def test_empty_views_section_still_renders(self, page: Page, base_url: str):
        """The Views section is a permanent affordance — its absence when empty
        would hide that saved searches exist at all."""
        self._setup(page, base_url, [FAKE_DATASET])
        expect(page.get_by_test_id("dataset-section-views")).to_contain_text("(0)")
        expect(page.locator("#dataset-management-box .ds-list-empty")).to_be_visible()

    def test_promote_button_reflects_the_selected_row_kind(self, page: Page, base_url: str):
        self._setup(page, base_url, [FAKE_DATASET, FAKE_VIEW])

        page.get_by_text(FAKE_VIEW["name"], exact=True).first.click()
        page.wait_for_timeout(600)
        expect(_promote_btn(page)).to_contain_text("Promote to Dataset")

        page.get_by_text(FAKE_DATASET["name"], exact=True).first.click()
        page.wait_for_timeout(600)
        expect(_promote_btn(page)).to_contain_text("Demote to View")

    def test_new_form_has_no_promote_button(self, page: Page, base_url: str):
        """Promotion is meaningless before the row exists."""
        self._setup(page, base_url, [FAKE_DATASET])
        expect(_promote_btn(page)).to_have_count(0)

    def test_promoting_a_view_patches_promoted_true(self, page: Page, base_url: str):
        patches: list[dict] = []

        def handle_patch(route: Route):
            if route.request.method == "PATCH":
                patches.append(json.loads(route.request.post_data))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "dataset": {**FAKE_VIEW, "promoted": True}}),
                )
            else:
                route.continue_()

        self._setup(page, base_url, [FAKE_DATASET, FAKE_VIEW])
        page.route("**/api/datasets/**", handle_patch)

        page.get_by_text(FAKE_VIEW["name"], exact=True).first.click()
        # `click()` auto-waits for actionability, so the button being there is enough of
        # a barrier for the first sleep; the modal assertion below covers the second.
        _promote_btn(page).click()

        modal = page.get_by_test_id("modal-shell-dialog")
        expect(modal).to_contain_text(FAKE_VIEW["name"])
        modal.locator("button", has_text="Promote").click()
        wait_until(page, lambda: patches, "Expected a PATCH to /api/datasets/{uuid}")
        assert patches[0].get("promoted") is True, patches[0]

    def test_cancelling_the_confirm_does_not_patch(self, page: Page, base_url: str):
        """Promotion changes what training can consume, so it is always confirmed."""
        patches: list[dict] = []

        def handle_patch(route: Route):
            if route.request.method == "PATCH":
                patches.append(json.loads(route.request.post_data))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "dataset": FAKE_VIEW}),
                )
            else:
                route.continue_()

        self._setup(page, base_url, [FAKE_DATASET, FAKE_VIEW])
        page.route("**/api/datasets/**", handle_patch)

        page.get_by_text(FAKE_VIEW["name"], exact=True).first.click()
        page.wait_for_timeout(600)
        _promote_btn(page).click()
        page.wait_for_timeout(300)
        page.get_by_test_id("modal-shell-dialog").locator("button", has_text="Cancel").click()
        page.wait_for_timeout(500)

        assert not patches, "Promotion must not fire when the confirm is cancelled"

    def test_demotion_refused_by_server_surfaces_the_reason(self, page: Page, base_url: str):
        """The server refuses demoting a Dataset a training config references;
        that reason must reach the user, not a generic failure."""

        def handle_patch(route: Route):
            if route.request.method == "PATCH":
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "status": "failure",
                            "message": 'Dataset "Training Set Alpha" is referenced by one or more '
                            "training configs and cannot be demoted to a View",
                        }
                    ),
                )
            else:
                route.continue_()

        self._setup(page, base_url, [FAKE_DATASET, FAKE_VIEW])
        page.route("**/api/datasets/**", handle_patch)

        page.get_by_text(FAKE_DATASET["name"], exact=True).first.click()
        page.wait_for_timeout(600)
        _promote_btn(page).click()
        page.wait_for_timeout(300)
        page.get_by_test_id("modal-shell-dialog").locator("button", has_text="Demote").click()
        page.wait_for_timeout(700)

        expect(page.locator("body")).to_contain_text("referenced by one or more training configs")


# ---------------------------------------------------------------------------
# Unsaved-changes navigation guard
#
# Shared by all four list-editor screens via ui/utils/unsavedNavGuard.js. It had
# no coverage at all before that extraction, so these pin the behaviour the four
# duplicated copies used to provide.
# ---------------------------------------------------------------------------


class TestUnsavedNavGuard:
    def _make_dirty(self, page: Page):
        _editor_body(page).locator("input[placeholder='Required']").fill("Half-typed name")

    def test_navigating_away_with_unsaved_changes_prompts(self, page: Page, base_url: str):
        _mock_labels(page, [ROOT_LABEL])
        _mock_tags_empty(page)
        _mock_datasets(page, [FAKE_DATASET])
        _navigate_to_datasets(page, base_url)

        self._make_dirty(page)
        page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-model-management']").click()

        modal = page.get_by_test_id("modal-shell-dialog")
        expect(modal).to_be_visible()
        expect(modal).to_contain_text("unsaved changes")
        modal.locator("button", has_text="Cancel").click()

    def test_cancel_keeps_you_on_the_pane(self, page: Page, base_url: str):
        _mock_labels(page, [ROOT_LABEL])
        _mock_tags_empty(page)
        _mock_datasets(page, [FAKE_DATASET])
        _navigate_to_datasets(page, base_url)

        self._make_dirty(page)
        page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-model-management']").click()
        page.get_by_test_id("modal-shell-dialog").locator("button", has_text="Cancel").click()

        expect(page.locator("#tab-dataset-management")).to_have_class(re.compile(r"\bactive\b"))

    def test_discard_lets_navigation_through(self, page: Page, base_url: str):
        _mock_labels(page, [ROOT_LABEL])
        _mock_tags_empty(page)
        _mock_datasets(page, [FAKE_DATASET])
        _navigate_to_datasets(page, base_url)

        self._make_dirty(page)
        page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-model-management']").click()
        page.get_by_test_id("modal-shell-dialog").locator("button", has_text="Don't save & continue").click()

        expect(page.locator("#tab-model-management")).to_have_class(re.compile(r"\bactive\b"))

    def test_clean_editor_navigates_without_prompting(self, page: Page, base_url: str):
        """The guard must not fire when there is nothing to lose."""
        _mock_labels(page, [ROOT_LABEL])
        _mock_tags_empty(page)
        _mock_datasets(page, [FAKE_DATASET])
        _navigate_to_datasets(page, base_url)

        page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-model-management']").click()

        expect(page.locator("#tab-model-management")).to_have_class(re.compile(r"\bactive\b"))
        expect(page.get_by_test_id("modal-shell-dialog")).to_have_count(0)

    def test_own_subtab_is_not_treated_as_navigating_away(self, page: Page, base_url: str):
        """Clicking the pane you are already on must not prompt (ownTabId)."""
        _mock_labels(page, [ROOT_LABEL])
        _mock_tags_empty(page)
        _mock_datasets(page, [FAKE_DATASET])
        _navigate_to_datasets(page, base_url)

        self._make_dirty(page)
        page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-dataset-management']").click()

        expect(page.get_by_test_id("modal-shell-dialog")).to_have_count(0)
