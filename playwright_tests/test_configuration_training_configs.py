"""
Tests for the Training Configuration UI (Configuration > Training).

Exercises renderTrainingConfigManager() rendered inside
Configuration > Training > #training-config-box
(apps/.../ui/trainingConfigs.js):

  - DOM structure: list panel + create form
  - Create flow: fill form → POST /api/training-configs with correct payload
    (modelConfigNodeUuid, datasetUuids, hyperparamsJson — no trainerType/labelUuids)
  - Readiness panel: shows ready/needs-review counts when a config with a node is selected
  - "Review unverified images" button: fires OPEN_REVIEW_WITH_FILTERS event and
    activates the Review tab

Run with:
    pytest playwright_tests/test_configuration_training_configs.py -p no:xdist -p no:asyncio -n 0 -v
"""

import json

import pytest
from playwright.sync_api import Page, Route, expect

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

DATASET_ALPHA = {"uuid": "ds-alpha", "name": "Alpha Dataset", "description": "", "filterJson": {}, "labelUuids": []}

MODEL_CFG_NODE = {
    "uuid": "node-root",
    "labelUuid": None,
    "modelType": "object_detection",
    "architecture": "yolo_v8_n",
}
MODEL_CFG = {"uuid": "mc-1", "name": "My Model", "description": "", "nodes": [MODEL_CFG_NODE]}

EXISTING_TC = {
    "uuid": "tc-1",
    "name": "Corrupt Detector",
    "description": "Finds corrupt images",
    "datasetUuids": ["ds-alpha"],
    "modelConfigNodeUuid": "node-root",
    "hyperparamsJson": {},
    "createdAt": "2026-01-01T00:00:00",
    "updatedAt": "2026-01-01T00:00:00",
}


# ---------------------------------------------------------------------------
# Route-interception helpers
# ---------------------------------------------------------------------------


def _mock_datasets(page: Page, datasets: list | None = None):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/datasets"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "datasets": datasets or []}),
        )
        if route.request.method == "GET"
        else route.fallback(),
    )


def _mock_model_configs(page: Page, configs: list | None = None):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/model-configs"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": configs or []}),
        )
        if route.request.method == "GET"
        else route.fallback(),
    )


def _mock_training_configs_list(page: Page, configs: list | None = None):
    """Intercept GET /api/training-configs (list only)."""
    page.route(
        lambda url: url.split("?")[0].endswith("/api/training-configs"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": configs or []}),
        )
        if route.request.method == "GET"
        else route.fallback(),
    )


def _mock_training_config_get(page: Page, config: dict):
    """Intercept GET /api/training-configs/{uuid}."""
    page.route(
        f"**/api/training-configs/{config['uuid']}",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "config": config}),
        )
        if route.request.method == "GET"
        else route.fallback(),
    )


def _mock_readiness(page: Page, config_uuid: str, ready: int = 3, needs_review: int = 2):
    page.route(
        f"**/api/training-configs/{config_uuid}/readiness",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "status": "success",
                "readyCount": ready,
                "needsReviewCount": needs_review,
                "datasetUuids": ["ds-alpha"],
                "labelUuids": [],
            }),
        ),
    )


def _mock_label_stats(page: Page, config_uuid: str):
    """Return empty label stats to suppress the panel's loading state."""
    page.route(
        f"**/api/training-configs/{config_uuid}/label-stats",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labelStats": [], "totalImages": 0}),
        ),
    )


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------


def _navigate_to_training_configs(page: Page, base_url: str):
    """Open the app and activate Configuration > Training."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-training-management']").click()
    page.wait_for_selector("#training-config-box .panel-box", timeout=4000)


def _box(page: Page):
    return page.locator("#training-config-box")


def _list_item(page: Page, uuid: str):
    """List item for the given config UUID."""
    return _box(page).locator(f"[data-config-uuid='{uuid}']")


def _editor_bar(page: Page):
    return _box(page).locator(".mc-editor .panel-box__bar")


def _editor_body(page: Page):
    return _box(page).locator(".mc-editor .panel-box__body")


# ---------------------------------------------------------------------------
# 1. Structure
# ---------------------------------------------------------------------------


class TestTrainingConfigPaneStructure:
    def test_pane_renders_list_and_create_form(self, page: Page, base_url: str):
        """The pane shows an existing config in the list (with dataset name) and
        defaults to a create form with a model-config selector."""
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])

        _navigate_to_training_configs(page, base_url)

        # Left panel: existing config name and dataset meta visible in list
        item = _list_item(page, "tc-1")
        expect(item).to_be_visible()
        expect(item).to_contain_text("Corrupt Detector")
        expect(item).to_contain_text("Alpha Dataset")

        # Footer new-config button
        expect(_box(page).locator("button", has_text="+ New")).to_be_visible()

        # Right panel defaults to create form
        expect(_editor_bar(page)).to_contain_text("New Training Config")

    def test_model_config_selector_is_present(self, page: Page, base_url: str):
        """The create form has a model config selector that lists available configs."""
        _mock_datasets(page, [])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [])

        _navigate_to_training_configs(page, base_url)

        # Wait for editor to render
        _editor_body(page).wait_for(timeout=4000)

        # The model config select should list our config
        config_sel = _editor_body(page).locator("select").first
        expect(config_sel).to_be_visible(timeout=4000)
        options = config_sel.locator("option").all_text_contents()
        assert any("My Model" in o for o in options), \
            f"Expected 'My Model' in config selector options; got: {options}"

    def test_node_selector_updates_when_config_chosen(self, page: Page, base_url: str):
        """Choosing a model config from the first selector populates the node selector."""
        _mock_datasets(page, [])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [])

        _navigate_to_training_configs(page, base_url)

        # Select the model config
        config_sel = _editor_body(page).locator("select").first
        config_sel.select_option("mc-1")
        page.wait_for_timeout(300)

        # Node selector (second select) should now have "Root" option
        node_sel = _editor_body(page).locator("select").nth(1)
        node_options = node_sel.locator("option").all_text_contents()
        assert any("Root" in o for o in node_options), \
            f"Expected a Root node option; got: {node_options}"


# ---------------------------------------------------------------------------
# 2. Create flow
# ---------------------------------------------------------------------------


class TestCreateTrainingConfig:
    def test_create_sends_correct_payload(self, page: Page, base_url: str):
        """Filling the form and saving POSTs modelConfigNodeUuid + datasetUuids;
        no trainerType or labelUuids in the payload."""
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [])

        captured = []

        def handle_create(route: Route):
            captured.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "config": {
                    **EXISTING_TC, "uuid": "tc-new", "name": "My Run",
                }}),
            )

        page.route("**/api/training-configs",
                   lambda r: handle_create(r) if r.request.method == "POST" else r.fallback())

        _navigate_to_training_configs(page, base_url)

        # Fill name
        name_input = _editor_body(page).locator("input[placeholder='Required']").first
        name_input.fill("My Run")

        # Select model config, then node
        config_sel = _editor_body(page).locator("select").first
        config_sel.select_option("mc-1")
        page.wait_for_timeout(200)
        node_sel = _editor_body(page).locator("select").nth(1)
        node_sel.select_option("node-root")

        # Check the dataset checkbox
        dataset_cb = _box(page).locator("input[type='checkbox'][value='ds-alpha']")
        dataset_cb.check()

        _editor_bar(page).locator("button", has_text="Create").or_(
            _editor_bar(page).locator("button", has_text="Save")
        ).click()
        page.wait_for_timeout(800)

        assert len(captured) == 1, f"Expected 1 POST, got {len(captured)}"
        payload = captured[0]
        assert payload.get("name") == "My Run"
        assert payload.get("modelConfigNodeUuid") == "node-root"
        assert "ds-alpha" in payload.get("datasetUuids", [])
        # Old fields must NOT be present
        assert "trainerType" not in payload, "trainerType should not be in payload"
        assert "labelUuids" not in payload, "labelUuids should not be in payload"

    def test_existing_config_node_is_preselected(self, page: Page, base_url: str):
        """Opening an existing training config pre-selects its modelConfigNodeUuid."""
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1")
        _mock_label_stats(page, "tc-1")

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()
        page.wait_for_timeout(400)

        # Config selector should show mc-1
        config_sel = _editor_body(page).locator("select").first
        expect(config_sel).to_have_value("mc-1", timeout=3000)

        # Node selector should show node-root
        node_sel = _editor_body(page).locator("select").nth(1)
        expect(node_sel).to_have_value("node-root", timeout=3000)


# ---------------------------------------------------------------------------
# 3. Readiness panel
# ---------------------------------------------------------------------------


class TestReadinessPanel:
    def test_readiness_counts_are_displayed(self, page: Page, base_url: str):
        """Selecting an existing config (with a node) shows the readiness panel."""
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=3, needs_review=2)
        _mock_label_stats(page, "tc-1")

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

        readiness_panel = _editor_body(page).locator("div", has_text="Training readiness").first
        expect(readiness_panel).to_be_visible(timeout=3000)
        expect(readiness_panel).to_contain_text("3")
        expect(readiness_panel).to_contain_text("2")

    def test_review_button_hidden_when_no_needs_review(self, page: Page, base_url: str):
        """'Review unverified images' button is hidden when needsReviewCount is 0."""
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=5, needs_review=0)
        _mock_label_stats(page, "tc-1")

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

        page.wait_for_selector(
            "#training-config-box div:has-text('Training readiness')", timeout=3000
        )
        btn = _box(page).locator("button", has_text="Review unverified images")
        expect(btn).to_have_count(1)
        expect(btn).to_be_hidden()

    def test_review_button_visible_when_needs_review(self, page: Page, base_url: str):
        """'Review unverified images' button appears when needsReviewCount > 0."""
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=3, needs_review=2)
        _mock_label_stats(page, "tc-1")

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

        btn = _box(page).locator("button", has_text="Review unverified images")
        expect(btn).to_be_visible(timeout=3000)

    def test_review_button_switches_to_review_tab(self, page: Page, base_url: str):
        """Clicking 'Review unverified images' activates the Review tab and the
        subsequent review-queue POST includes labelVerificationCheck=has_unverified."""
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=1, needs_review=3)
        _mock_label_stats(page, "tc-1")

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

        btn = _box(page).locator("button", has_text="Review unverified images")
        expect(btn).to_be_visible(timeout=3000)

        captured_queue_requests = []

        def handle_queue(route: Route):
            try:
                captured_queue_requests.append(json.loads(route.request.post_data))
            except Exception:
                captured_queue_requests.append({})
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({
                    "status": "success", "images": [], "hasMore": False, "totalCount": 0,
                }),
            )

        page.route("**/api/images/review", handle_queue)

        with page.expect_request(
            lambda r: "/api/images/review" in r.url and r.method == "POST",
            timeout=8000,
        ):
            btn.click()

        import re as _re
        expect(page.locator("#tab-review")).to_have_class(_re.compile(r"\bactive\b"), timeout=4000)

        page.wait_for_timeout(600)
        last = captured_queue_requests[-1]
        assert last.get("labelVerificationCheck") == "has_unverified", \
            f"Expected labelVerificationCheck='has_unverified'; got: {last}"
        # labelUuids is now empty (derived from model tree at runtime, not stored)
        assert last.get("labelUuids", []) == [], \
            f"Expected empty labelUuids; got: {last.get('labelUuids')}"
