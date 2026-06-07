"""
Tests for the Model Configurations UI (Configuration > Models).

Exercises renderModelConfigManager() rendered inside
Configuration > Models > #model-config-box (apps/.../ui/modelConfigs.js):
  - DOM structure (config list + create form, hierarchical label tree)
  - Create flow: select labels + root model type → POST /api/model-configs
  - Edit flow: opening an existing config pre-fills the form and a submodel panel
  - Save flow: editing & saving PATCHes /api/model-configs/{uuid} with new nodes
  - Clearing an assignment removes it from the saved payload
  - Delete flow (modal confirm; API called or suppressed)

Mirrors the structure/style of test_configuration_datasets.py. Follows the
route-interception pattern from test_configuration_labels.py / _sources.py —
the model-configs/labels proxies default to empty in conftest.

Run with:
    pytest playwright_tests/test_configuration_model_configs.py -p no:xdist -p no:asyncio -v
"""
import json

import pytest
from playwright.sync_api import Page, Route, expect

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

LABEL_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parent_uuid": None}
LABEL_KITTEN = {"uuid": "lbl-kitten", "name": "Kitten", "color": "#ff8888", "parent_uuid": "lbl-cat"}
LABEL_DOG = {"uuid": "lbl-dog", "name": "Dog", "color": "#00ff00", "parent_uuid": None}

TYPENAMES = ["yolov8n", "yolov8s", "yolov5m"]

EXISTING_CONFIG = {
    "uuid": "cfg-1",
    "name": "Animal Detector",
    "description": "Cats & dogs",
    "nodes": [
        {"label_uuid": None, "model_typename": "yolov8n"},
        {"label_uuid": "lbl-cat", "model_typename": "yolov8s"},
    ],
}

# ---------------------------------------------------------------------------
# Navigation / mocking helpers
# ---------------------------------------------------------------------------


def _navigate_to_model_configs(page: Page, base_url: str):
    """Open the app and activate Configuration > Models."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-model-management']").click()
    page.wait_for_load_state("networkidle")
    # New DOM: panel-box structure instead of config-panel
    page.wait_for_selector("#model-config-box .mc-list-panel", timeout=4000)


def _mock_model_configs(page: Page, configs: list | None = None):
    """Intercept GET /api/model-configs (list) — CRUD endpoints are mocked separately."""
    configs = configs or []
    page.route(
        lambda url: url.split("?")[0].endswith("/api/model-configs"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": configs}),
        )
        if route.request.method == "GET"
        else route.fallback(),
    )


def _mock_typenames(page: Page, typenames: list | None = None):
    page.route(
        "**/api/model-configs/typenames",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "typenames": typenames or TYPENAMES}),
        ),
    )


def _mock_labels(page: Page, labels: list | None = None):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "/api/labels/" not in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": labels or []}),
        ),
    )


def _label_row(page: Page, uuid: str):
    return page.locator(f"#model-config-box [data-uuid='{uuid}']")


def _editor(page: Page):
    """The editor panel (right side)."""
    return page.locator("#model-config-box .mc-editor")


def _editor_bar(page: Page):
    """The sticky header bar of the editor panel."""
    return page.locator("#model-config-box .mc-editor .panel-box__bar")


def _editor_body(page: Page):
    """The scrollable body of the editor panel."""
    return page.locator("#model-config-box .mc-editor .panel-box__body")


# ---------------------------------------------------------------------------
# 1. DOM structure
# ---------------------------------------------------------------------------


class TestModelConfigPaneStructure:
    def test_pane_renders_list_and_default_create_form(self, page: Page, base_url: str):
        """The pane shows the config list (with assignment counts) and defaults to the
        create-new-config form with a hierarchical, fully-expanded label tree."""
        _mock_model_configs(page, [EXISTING_CONFIG])
        _mock_typenames(page)
        _mock_labels(page, [LABEL_CAT, LABEL_KITTEN, LABEL_DOG])

        _navigate_to_model_configs(page, base_url)

        # Left panel: list item with name and assignment count
        list_row = page.locator("#model-config-box [data-config-uuid='cfg-1']")
        expect(list_row.locator(".mc-list-item__name")).to_have_text("Animal Detector")
        expect(list_row).to_contain_text("2 assignments")

        # Footer button for creating a new config
        expect(page.locator("#model-config-box .mc-list-panel button", has_text="+ New")).to_be_visible()

        # Right panel defaults to the "create" form; title is in the panel bar span
        expect(_editor_bar(page)).to_contain_text("New Configuration")

        for lbl in (LABEL_CAT, LABEL_KITTEN, LABEL_DOG):
            row = _label_row(page, lbl["uuid"])
            expect(row).to_be_visible()
            expect(row).to_contain_text(lbl["name"])
            expect(row.locator("input[type=checkbox]").first).not_to_be_checked()

        # Cat has a child (Kitten) so it shows the "sub model" toggle; Dog/Kitten don't.
        expect(_label_row(page, "lbl-cat").locator("label", has_text="sub model")).to_be_visible()
        expect(_label_row(page, "lbl-dog").locator("label", has_text="sub model")).to_have_count(0)
        expect(_label_row(page, "lbl-kitten").locator("label", has_text="sub model")).to_have_count(0)


# ---------------------------------------------------------------------------
# 2. Create flow
# ---------------------------------------------------------------------------


class TestCreateModelConfig:
    def test_create_sends_selected_labels_and_root_typename(self, page: Page, base_url: str):
        """Naming the config, picking a root model type, and checking two labels
        sends a create payload whose nodes are the root + the two selected labels,
        all inheriting the chosen root typename (no submodels enabled)."""
        _mock_model_configs(page, [])
        _mock_typenames(page)
        _mock_labels(page, [LABEL_CAT, LABEL_KITTEN, LABEL_DOG])

        calls = []

        def handle_create(route: Route):
            calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({
                    "status": "success",
                    "config": {"uuid": "cfg-new", "name": "My Config", "description": "", "nodes": []},
                }),
            )

        page.route(
            lambda url: url.split("?")[0].endswith("/api/model-configs"),
            lambda route: handle_create(route) if route.request.method == "POST" else route.fallback(),
        )

        _navigate_to_model_configs(page, base_url)

        # Fill form fields in the editor body
        _editor_body(page).locator("input[type=text]").first.fill("My Config")

        root_select = page.locator("#model-config-box select").first
        root_select.select_option("yolov5m")

        _label_row(page, "lbl-cat").locator("input[type=checkbox]").first.click()
        _label_row(page, "lbl-dog").locator("input[type=checkbox]").first.click()

        # Create button lives in the sticky editor header bar
        _editor_bar(page).locator("button", has_text="Create").click()
        page.wait_for_timeout(700)

        assert calls, "Expected POST to /api/model-configs"
        payload = calls[0]
        assert payload["name"] == "My Config"

        nodes_by_label = {n["label_uuid"]: n["model_typename"] for n in payload["nodes"]}
        assert nodes_by_label[None] == "yolov5m"
        assert nodes_by_label["lbl-cat"] == "yolov5m"
        assert nodes_by_label["lbl-dog"] == "yolov5m"
        assert "lbl-kitten" not in nodes_by_label


# ---------------------------------------------------------------------------
# 3. Edit flow — pre-fill + submodel panel
# ---------------------------------------------------------------------------


class TestEditModelConfig:
    def test_selecting_config_prefills_form_and_submodel_panel(self, page: Page, base_url: str):
        """Clicking a config in the list opens its edit form, pre-filled with its
        name/description/root-typename, the tree reflects its node selections, and
        a submodel configuration card appears for the label with a non-inherited typename."""
        _mock_model_configs(page, [EXISTING_CONFIG])
        _mock_typenames(page)
        _mock_labels(page, [LABEL_CAT, LABEL_KITTEN, LABEL_DOG])

        _navigate_to_model_configs(page, base_url)
        page.locator("#model-config-box [data-config-uuid='cfg-1']").click()
        page.wait_for_timeout(500)

        # Title is now in the panel bar span (not an h4)
        expect(_editor_bar(page)).to_contain_text("Animal Detector")

        # Form fields in the editor body
        expect(_editor_body(page).locator("input[type=text]").first).to_have_value("Animal Detector")
        expect(_editor_body(page).locator("input[type=text]").nth(1)).to_have_value("Cats & dogs")
        expect(page.locator("#model-config-box select").first).to_have_value("yolov8n")

        cat_row = _label_row(page, "lbl-cat")
        expect(cat_row.locator("input[type=checkbox]").first).to_be_checked()
        sub_toggle = cat_row.locator("label", has_text="sub model").locator("input[type=checkbox]")
        expect(sub_toggle).to_be_checked()

        # Submodel panel: "Sub-model overrides" section label (was "Sub Model Configurations" h4)
        expect(page.locator("#model-config-box .panel-section-label", has_text="Sub-model overrides")).to_be_visible()
        submodel_select = page.locator("#model-config-box select").nth(1)
        expect(submodel_select).to_have_value("yolov8s")

        # Delete button is in the sticky editor header bar (no emoji)
        expect(_editor_bar(page).locator("button", has_text="Delete")).to_be_visible()

    def test_save_sends_updated_name_and_nodes(self, page: Page, base_url: str):
        """Editing the name and changing the submodel's typename PATCHes the config
        with the new name and a node reflecting the new submodel typename."""
        _mock_model_configs(page, [EXISTING_CONFIG])
        _mock_typenames(page)
        _mock_labels(page, [LABEL_CAT, LABEL_KITTEN, LABEL_DOG])

        calls = []

        def handle_patch(route: Route):
            calls.append(json.loads(route.request.post_data))
            updated = {**EXISTING_CONFIG, "name": "Animal Detector v2"}
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success", "config": updated}))

        page.route("**/api/model-configs/cfg-1", lambda route: handle_patch(route) if route.request.method == "PATCH" else route.fallback())

        _navigate_to_model_configs(page, base_url)
        page.locator("#model-config-box [data-config-uuid='cfg-1']").click()
        page.wait_for_timeout(500)

        name_input = _editor_body(page).locator("input[type=text]").first
        name_input.fill("Animal Detector v2")

        submodel_select = page.locator("#model-config-box select").nth(1)
        submodel_select.select_option("yolov5m")

        # Save button is in the sticky editor header bar
        _editor_bar(page).locator("button", has_text="Save").click()
        page.wait_for_timeout(700)

        assert calls, "Expected PATCH to /api/model-configs/cfg-1"
        payload = calls[0]
        assert payload["name"] == "Animal Detector v2"
        nodes_by_label = {n["label_uuid"]: n["model_typename"] for n in payload["nodes"]}
        assert nodes_by_label["lbl-cat"] == "yolov5m"


# ---------------------------------------------------------------------------
# 4. Clearing an assignment
# ---------------------------------------------------------------------------


class TestClearAssignment:
    def test_unchecking_a_label_removes_its_node_from_the_payload(self, page: Page, base_url: str):
        """Unchecking a previously-assigned label clears both its selection and any
        submodel override, and the saved payload no longer includes a node for it."""
        _mock_model_configs(page, [EXISTING_CONFIG])
        _mock_typenames(page)
        _mock_labels(page, [LABEL_CAT, LABEL_KITTEN, LABEL_DOG])

        calls = []

        def handle_patch(route: Route):
            calls.append(json.loads(route.request.post_data))
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success", "config": EXISTING_CONFIG}))

        page.route("**/api/model-configs/cfg-1", lambda route: handle_patch(route) if route.request.method == "PATCH" else route.fallback())

        _navigate_to_model_configs(page, base_url)
        page.locator("#model-config-box [data-config-uuid='cfg-1']").click()
        page.wait_for_timeout(500)

        cat_row = _label_row(page, "lbl-cat")
        expect(cat_row.locator("input[type=checkbox]").first).to_be_checked()
        cat_row.locator("input[type=checkbox]").first.click()  # uncheck

        # Unchecking clears the submodel too — its section label disappears
        expect(page.locator("#model-config-box .panel-section-label", has_text="Sub-model overrides")).to_have_count(0)

        # Save button is in the sticky editor header bar
        _editor_bar(page).locator("button", has_text="Save").click()
        page.wait_for_timeout(700)

        assert calls, "Expected PATCH to /api/model-configs/cfg-1"
        payload = calls[0]
        labels_in_payload = {n["label_uuid"] for n in payload["nodes"]}
        assert "lbl-cat" not in labels_in_payload
        assert None in labels_in_payload  # root node always present


# ---------------------------------------------------------------------------
# 5. Delete flow
# ---------------------------------------------------------------------------


class TestDeleteModelConfig:
    def test_delete_shows_confirm_dialog(self, page: Page, base_url: str):
        """Clicking Delete opens a modal dialog naming the config."""
        _mock_model_configs(page, [EXISTING_CONFIG])
        _mock_typenames(page)
        _mock_labels(page, [LABEL_CAT, LABEL_KITTEN, LABEL_DOG])

        _navigate_to_model_configs(page, base_url)
        page.locator("#model-config-box [data-config-uuid='cfg-1']").click()
        page.wait_for_timeout(500)

        # Delete button is in the sticky editor header bar (no emoji)
        _editor_bar(page).locator("button", has_text="Delete").click()
        page.wait_for_timeout(300)

        # Custom modal (not window.confirm) — verify it appears with the config name
        modal = page.locator(".app-modal-dialog")
        expect(modal).to_be_visible()
        expect(modal).to_contain_text("Animal Detector")

    def test_delete_accepted_calls_api(self, page: Page, base_url: str):
        """Confirming the delete modal sends a DELETE to /api/model-configs/{uuid}."""
        _mock_model_configs(page, [EXISTING_CONFIG])
        _mock_typenames(page)
        _mock_labels(page, [LABEL_CAT, LABEL_KITTEN, LABEL_DOG])

        calls = []

        def handle_delete(route: Route):
            calls.append(route.request.url)
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

        page.route(
            "**/api/model-configs/cfg-1",
            lambda route: handle_delete(route) if route.request.method == "DELETE" else route.fallback(),
        )

        _navigate_to_model_configs(page, base_url)
        page.locator("#model-config-box [data-config-uuid='cfg-1']").click()
        page.wait_for_timeout(500)

        _editor_bar(page).locator("button", has_text="Delete").click()
        page.wait_for_timeout(300)

        # Confirm by clicking "Delete" inside the modal
        page.locator(".app-modal-dialog button", has_text="Delete").click()
        page.wait_for_timeout(600)

        assert calls, "Expected DELETE to /api/model-configs/cfg-1"
        assert calls[0].rstrip("/").endswith("/cfg-1")

    def test_delete_dismissed_does_not_call_api(self, page: Page, base_url: str):
        """Cancelling the delete modal must NOT send a delete request."""
        _mock_model_configs(page, [EXISTING_CONFIG])
        _mock_typenames(page)
        _mock_labels(page, [LABEL_CAT, LABEL_KITTEN, LABEL_DOG])

        calls = []
        page.route(
            "**/api/model-configs/cfg-1",
            lambda route: (calls.append(True), route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"})))
            if route.request.method == "DELETE"
            else route.fallback(),
        )

        _navigate_to_model_configs(page, base_url)
        page.locator("#model-config-box [data-config-uuid='cfg-1']").click()
        page.wait_for_timeout(500)

        _editor_bar(page).locator("button", has_text="Delete").click()
        page.wait_for_timeout(300)

        # Cancel the modal — no API call should happen
        page.locator(".app-modal-dialog button", has_text="Cancel").click()
        page.wait_for_timeout(500)

        assert not calls, "API must NOT be called when delete is cancelled"
