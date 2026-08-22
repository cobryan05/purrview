"""
Tests for the Training Configuration UI (Configuration > Training).

Exercises renderTrainingConfigManager() rendered inside
Configuration > Training > #training-config-box
(apps/.../ui/trainingConfigs.js):

  - DOM structure: list panel + create form
  - Create flow: fill form → POST /api/training-configs with correct payload
    (modelConfigNodeUuid, datasetUuids, hyperparamsJson — no trainerType/labelUuids)
  - Readiness panel: shows ready/needs-review/not-started counts when a config
    with a node is selected
  - Clickable readiness counts: clicking "need review" opens a showReviewModal
    overlay backed by POST /api/training-configs/{uuid}/partial-review-images

Run with:
    pytest playwright_tests/test_configuration_training_configs.py -p no:xdist -p no:asyncio -n 0 -v
"""

import json
import re

import pytest
from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

DATASET_ALPHA = {"uuid": "ds-alpha", "name": "Alpha Dataset", "description": "", "filterJson": {}, "labelUuids": []}

MODEL_CFG_NODE = {
    "uuid": "node-root",
    "labelUuid": None,
    "modelType": "object_detection",
    "architecture": "yolo_v9_c",
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
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "datasets": datasets or []}),
            )
            if route.request.method == "GET"
            else route.fallback()
        ),
    )


def _mock_model_configs(page: Page, configs: list | None = None):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/model-configs"),
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configs": configs or []}),
            )
            if route.request.method == "GET"
            else route.fallback()
        ),
    )


def _mock_training_configs_list(page: Page, configs: list | None = None):
    """Intercept GET /api/training-configs (list only)."""
    page.route(
        lambda url: url.split("?")[0].endswith("/api/training-configs"),
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configs": configs or []}),
            )
            if route.request.method == "GET"
            else route.fallback()
        ),
    )


def _mock_training_config_get(page: Page, config: dict):
    """Intercept GET /api/training-configs/{uuid}."""
    page.route(
        f"**/api/training-configs/{config['uuid']}",
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "config": config}),
            )
            if route.request.method == "GET"
            else route.fallback()
        ),
    )


def _mock_readiness(
    page: Page,
    config_uuid: str,
    ready: int = 3,
    needs_review: int = 2,
    not_started: int = 0,
    ready_positive: int | None = None,
    ready_negative: int = 0,
):
    # The split defaults to "all positive" so callers that only care about the
    # totals keep describing the same scenario they did before it existed.
    if ready_positive is None:
        ready_positive = ready - ready_negative
    page.route(
        f"**/api/training-configs/{config_uuid}/readiness",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "readyCount": ready,
                    "readyPositiveCount": ready_positive,
                    "readyNegativeCount": ready_negative,
                    "needsReviewCount": needs_review,
                    "notStartedCount": not_started,
                    "datasetUuids": ["ds-alpha"],
                    "labelUuids": [],
                }
            ),
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

        # Footer new-config button. Targeted by its block class, not by its text:
        # the form also carries a "+ New shape" button for the Model Shape field,
        # and `has_text` is a substring match, so "+ New" alone matches both.
        expect(_box(page).locator("button.panel-btn--block", has_text="+ New")).to_be_visible()

        # Right panel defaults to create form
        expect(_editor_bar(page)).to_contain_text("New Training Run")

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
        assert any("My Model" in o for o in options), f"Expected 'My Model' in config selector options; got: {options}"

    def test_node_selector_updates_when_config_chosen(self, page: Page, base_url: str):
        """Choosing a model config from the first selector populates the node selector."""
        _mock_datasets(page, [])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [])

        _navigate_to_training_configs(page, base_url)

        # Select the model config
        config_sel = _editor_body(page).locator("select").first
        config_sel.select_option("mc-1")

        # Node selector (second select) should now have "Root" option. `expect` retries
        # until the dependent select has been repopulated; the 300ms sleep it replaces
        # was a bet on that round trip fitting in 300ms.
        node_sel = _editor_body(page).locator("select").nth(1)
        expect(node_sel.locator("option").filter(has_text="Root").first).to_be_attached()


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
                body=json.dumps(
                    {
                        "status": "success",
                        "config": {
                            **EXISTING_TC,
                            "uuid": "tc-new",
                            "name": "My Run",
                        },
                    }
                ),
            )

        page.route(
            "**/api/training-configs", lambda r: handle_create(r) if r.request.method == "POST" else r.fallback()
        )

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
        wait_until(page, lambda: len(captured) >= 1, lambda: f"Expected 1 POST, got {len(captured)}")

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
# 2b. Training settings (hyperparams) fieldset
# ---------------------------------------------------------------------------


class TestTrainingSettingsFieldset:
    def test_fields_prefilled_from_existing_hyperparams(self, page: Page, base_url: str):
        """Opening an existing config with hyperparamsJson set pre-fills the
        batch size / image size / max epochs / patience inputs."""
        tc_with_hyperparams = {
            **EXISTING_TC,
            "hyperparamsJson": {"batch_size": 4, "imgsz": 512, "max_epochs": 30, "early_stopping_patience": 15},
        }
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [tc_with_hyperparams])
        _mock_training_config_get(page, tc_with_hyperparams)
        _mock_readiness(page, "tc-1")
        _mock_label_stats(page, "tc-1")

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()
        page.wait_for_timeout(400)

        body = _editor_body(page)
        expect(body.locator("input[type='number']").nth(0)).to_have_value("4")
        expect(body.locator("input[type='number']").nth(1)).to_have_value("512")
        expect(body.locator("input[type='number']").nth(2)).to_have_value("30")
        expect(body.locator("input[type='number']").nth(3)).to_have_value("15")

    def test_save_sends_batch_size_and_both_patience_keys(self, page: Page, base_url: str):
        """Filling in the training-settings fields and saving sends batch_size/
        imgsz/max_epochs plus BOTH early_stopping_patience and patience (the
        object-detection trainer reads `patience`, classifiers read
        `early_stopping_patience`) inside hyperparamsJson."""
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [])

        captured = []

        def handle_create(route: Route):
            captured.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "config": {
                            **EXISTING_TC,
                            "uuid": "tc-new",
                            "name": "My Run",
                        },
                    }
                ),
            )

        page.route(
            "**/api/training-configs", lambda r: handle_create(r) if r.request.method == "POST" else r.fallback()
        )

        _navigate_to_training_configs(page, base_url)

        name_input = _editor_body(page).locator("input[placeholder='Required']").first
        name_input.fill("My Run")

        config_sel = _editor_body(page).locator("select").first
        config_sel.select_option("mc-1")
        page.wait_for_timeout(200)
        node_sel = _editor_body(page).locator("select").nth(1)
        node_sel.select_option("node-root")

        number_inputs = _editor_body(page).locator("input[type='number']")
        number_inputs.nth(0).fill("4")
        number_inputs.nth(1).fill("512")
        number_inputs.nth(2).fill("30")
        number_inputs.nth(3).fill("15")

        dataset_cb = _box(page).locator("input[type='checkbox'][value='ds-alpha']")
        dataset_cb.check()

        _editor_bar(page).locator("button", has_text="Create").or_(
            _editor_bar(page).locator("button", has_text="Save")
        ).click()
        wait_until(page, lambda: len(captured) >= 1, lambda: f"Expected 1 POST, got {len(captured)}")

        assert len(captured) == 1, f"Expected 1 POST, got {len(captured)}"
        hyperparams = captured[0].get("hyperparamsJson", {})
        assert hyperparams.get("batch_size") == 4
        assert hyperparams.get("imgsz") == 512
        assert hyperparams.get("max_epochs") == 30
        assert hyperparams.get("early_stopping_patience") == 15
        assert hyperparams.get("patience") == 15


class TestRecordImageManifestOptOut:
    """ "Remember exact image list" is an opt-OUT: ticked by default, and the key
    only appears in hyperparamsJson when the user unticks it (no saved config
    carries the key, and absent must keep meaning "record")."""

    def _manifest_checkbox(self, page: Page):
        return _editor_body(page).locator("input[data-testid='hyperparams-record-image-manifest']")

    def _create_and_capture(self, page: Page, base_url: str, untick: bool):
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [])

        captured = []

        def handle_create(route: Route):
            captured.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "config": {**EXISTING_TC, "uuid": "tc-new", "name": "My Run"}}),
            )

        page.route(
            "**/api/training-configs", lambda r: handle_create(r) if r.request.method == "POST" else r.fallback()
        )

        _navigate_to_training_configs(page, base_url)
        _editor_body(page).locator("input[placeholder='Required']").first.fill("My Run")
        _editor_body(page).locator("select").first.select_option("mc-1")
        page.wait_for_timeout(200)
        _editor_body(page).locator("select").nth(1).select_option("node-root")

        if untick:
            self._manifest_checkbox(page).uncheck()

        _box(page).locator("input[type='checkbox'][value='ds-alpha']").check()
        _editor_bar(page).locator("button", has_text="Create").or_(
            _editor_bar(page).locator("button", has_text="Save")
        ).click()
        wait_until(page, lambda: len(captured) >= 1, lambda: f"Expected 1 POST, got {len(captured)}")

        assert len(captured) == 1, f"Expected 1 POST, got {len(captured)}"
        return captured[0].get("hyperparamsJson", {})

    def test_checked_by_default(self, page: Page, base_url: str):
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [])

        _navigate_to_training_configs(page, base_url)
        expect(self._manifest_checkbox(page)).to_be_checked()

    def test_leaving_it_ticked_writes_no_key(self, page: Page, base_url: str):
        hyperparams = self._create_and_capture(page, base_url, untick=False)
        assert "record_image_manifest" not in hyperparams

    def test_unticking_writes_false(self, page: Page, base_url: str):
        hyperparams = self._create_and_capture(page, base_url, untick=True)
        assert hyperparams.get("record_image_manifest") is False

    def test_prefilled_unticked_from_existing_config(self, page: Page, base_url: str):
        tc_opted_out = {**EXISTING_TC, "hyperparamsJson": {"record_image_manifest": False}}
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [tc_opted_out])
        _mock_training_config_get(page, tc_opted_out)
        _mock_readiness(page, "tc-1")
        _mock_label_stats(page, "tc-1")

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()
        page.wait_for_timeout(400)

        expect(self._manifest_checkbox(page)).not_to_be_checked()


# ---------------------------------------------------------------------------
# 2c. VRAM-tiered hyperparameter suggestions
# ---------------------------------------------------------------------------


class TestVramHyperparamsSelect:
    def _vram_select(self, page: Page):
        """The VRAM tier select, identified by its '— none —' placeholder option
        (it's the only select with that text; config/node selects use
        '— select a config/node —')."""
        return _editor_body(page).locator("select").filter(has_text="— none —")

    def test_vram_select_hidden_until_object_detection_node_chosen(self, page: Page, base_url: str):
        """No config/node selected yet -> architecture is unknown -> the VRAM
        select stays hidden rather than guessing."""
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [])

        _navigate_to_training_configs(page, base_url)
        _editor_body(page).wait_for(timeout=4000)

        expect(self._vram_select(page)).to_be_hidden()

        config_sel = _editor_body(page).locator("select").first
        config_sel.select_option("mc-1")
        page.wait_for_timeout(200)
        node_sel = _editor_body(page).locator("select").nth(1)
        node_sel.select_option("node-root")  # MODEL_CFG_NODE arch = yolo_v9_c (object_detection)
        page.wait_for_timeout(200)

        expect(self._vram_select(page)).to_be_visible()

    def test_selecting_vram_tier_prefills_batch_size_and_imgsz_and_stays_editable(self, page: Page, base_url: str):
        """Picking a VRAM tier prefills batch_size/imgsz from the per-architecture
        table (yolo_v9_c @ 8GB -> batch_size=8, imgsz=640 per vramHyperparams.js),
        and the fields remain enabled/editable afterward — selecting a tier
        must not lock them."""
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [])

        _navigate_to_training_configs(page, base_url)
        _editor_body(page).wait_for(timeout=4000)

        config_sel = _editor_body(page).locator("select").first
        config_sel.select_option("mc-1")
        page.wait_for_timeout(200)
        node_sel = _editor_body(page).locator("select").nth(1)
        node_sel.select_option("node-root")
        page.wait_for_timeout(200)

        vram_sel = self._vram_select(page)
        expect(vram_sel).to_be_visible()
        vram_sel.select_option("8")
        page.wait_for_timeout(200)

        number_inputs = _editor_body(page).locator("input[type='number']")
        batch_size_input = number_inputs.nth(0)
        imgsz_input = number_inputs.nth(1)

        expect(batch_size_input).to_have_value("8")
        expect(imgsz_input).to_have_value("640")

        # Fields stay editable — not disabled by the VRAM pick.
        expect(batch_size_input).to_be_enabled()
        expect(imgsz_input).to_be_enabled()
        batch_size_input.fill("2")
        expect(batch_size_input).to_have_value("2")


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

    def test_needs_review_count_not_clickable_when_zero(self, page: Page, base_url: str):
        """The '⚠ N need review' count is not a clickable link when needsReviewCount is 0."""
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=5, needs_review=0)
        _mock_label_stats(page, "tc-1")

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

        page.wait_for_selector("#training-config-box div:has-text('Training readiness')", timeout=3000)
        needs_review_span = _box(page).get_by_test_id("training-readiness-needs-review-count")
        expect(needs_review_span).to_contain_text("0 need review")
        style = needs_review_span.get_attribute("style") or ""
        assert "cursor: pointer" not in style, "count should not be clickable when count is 0"

    def test_needs_review_count_clickable_when_nonzero(self, page: Page, base_url: str):
        """The '⚠ N need review' count is clickable when needsReviewCount > 0."""
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=3, needs_review=2)
        _mock_label_stats(page, "tc-1")

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

        needs_review_span = _box(page).get_by_test_id("training-readiness-needs-review-count")
        expect(needs_review_span).to_be_visible(timeout=3000)
        style = needs_review_span.get_attribute("style") or ""
        assert "cursor: pointer" in style, "count should be clickable when count > 0"

    def test_needs_review_count_click_opens_modal(self, page: Page, base_url: str):
        """Clicking the '⚠ N need review' count opens a review modal and issues a POST
        to /api/training-configs/{uuid}/partial-review-images (not the global review queue).
        """
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=1, needs_review=3)
        _mock_label_stats(page, "tc-1")

        # Mock the partial-review-images endpoint so the modal can load
        captured_requests = []

        def handle_partial_review(route: Route):
            try:
                captured_requests.append(json.loads(route.request.post_data))
            except Exception:
                captured_requests.append({})
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "images": [],
                        "hasMore": False,
                        "totalCount": 0,
                    }
                ),
            )

        page.route("**/api/training-configs/tc-1/partial-review-images", handle_partial_review)

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

        needs_review_span = _box(page).get_by_test_id("training-readiness-needs-review-count")
        expect(needs_review_span).to_be_visible(timeout=3000)

        with page.expect_request(
            lambda r: "partial-review-images" in r.url and r.method == "POST",
            timeout=8000,
        ):
            needs_review_span.click()

        # Modal is now visible — detect it by its title strong element
        modal_title = page.locator("strong", has_text="Needs Review")
        expect(modal_title).to_be_visible(timeout=4000)

        wait_until(page, lambda: len(captured_requests) >= 1, "partial-review-images endpoint was not called")


class TestReadyPositiveNegativeSplit:
    """A mostly-background dataset reads as thousands of "ready" images when only a
    handful carry an annotation, so the ready count shows its positive/background split."""

    def _setup(self, page: Page, base_url: str, ready: int, ready_negative: int, not_started: int = 0):
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(
            page,
            "tc-1",
            ready=ready,
            needs_review=0,
            not_started=not_started,
            ready_negative=ready_negative,
        )
        _mock_label_stats(page, "tc-1")
        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

    def test_split_is_shown_alongside_the_ready_count(self, page: Page, base_url: str):
        self._setup(page, base_url, ready=3564, ready_negative=3524)

        expect(_box(page).get_by_test_id("training-readiness-ready-count")).to_contain_text("3564 ready")
        expect(_box(page).get_by_test_id("training-readiness-ready-positive-count")).to_contain_text("40 positive")
        expect(_box(page).get_by_test_id("training-readiness-ready-negative-count")).to_contain_text("3524 background")

    def test_split_is_hidden_when_nothing_is_ready(self, page: Page, base_url: str):
        """With no ready images there is no split to make — the parenthetical would
        just be "(0 positive · 0 background)" noise. The dataset still has images —
        an entirely empty one renders a different message and wouldn't test this."""
        self._setup(page, base_url, ready=0, ready_negative=0, not_started=7)

        expect(_box(page).get_by_test_id("training-readiness-ready-count")).to_be_visible(timeout=4000)
        expect(_box(page).get_by_test_id("training-readiness-ready-split")).to_have_count(0)

    def test_positive_count_opens_the_positive_bucket(self, page: Page, base_url: str):
        self._setup(page, base_url, ready=100, ready_negative=90)

        positive = _box(page).get_by_test_id("training-readiness-ready-positive-count")
        expect(positive).to_be_visible(timeout=4000)

        with page.expect_request(
            lambda r: "ready-positive-images" in r.url and r.method == "POST",
            timeout=8000,
        ):
            positive.click()

        expect(page.locator("strong", has_text="Positive Samples")).to_be_visible(timeout=4000)

    def test_background_count_opens_the_background_bucket(self, page: Page, base_url: str):
        self._setup(page, base_url, ready=100, ready_negative=90)

        negative = _box(page).get_by_test_id("training-readiness-ready-negative-count")
        expect(negative).to_be_visible(timeout=4000)

        with page.expect_request(
            lambda r: "ready-negative-images" in r.url and r.method == "POST",
            timeout=8000,
        ):
            negative.click()

        expect(page.locator("strong", has_text="Background Samples")).to_be_visible(timeout=4000)

    def test_zero_side_of_the_split_is_not_clickable(self, page: Page, base_url: str):
        """A fully-positive run still shows the split (it's reassuring), but the
        background half has nothing to browse."""
        self._setup(page, base_url, ready=12, ready_negative=0)

        negative = _box(page).get_by_test_id("training-readiness-ready-negative-count")
        expect(negative).to_contain_text("0 background")
        style = negative.get_attribute("style") or ""
        assert "cursor: pointer" not in style, "count should not be clickable when count is 0"


# ---------------------------------------------------------------------------
# 4. Label stats panel — clickable bbox columns
# ---------------------------------------------------------------------------

OBJECT_LABEL = {"uuid": "lbl-obj", "name": "Cat", "color": "#ff0000", "scope": "object", "parentUuid": None}

LABEL_STATS_WITH_OBJECT = {
    "labelStats": [
        {
            "labelUuid": "lbl-obj",
            "labelScope": "object",
            "verifiedBboxCount": 5,
            "unverifiedBboxCount": 2,
            "coLabelsCoveredCount": 3,
            "coLabelGapsCount": 1,
            "hasRollup": False,
            "rollUpParentUuid": None,
        }
    ],
    "totalImages": 8,
}


def _mock_labels(page: Page, labels: list | None = None):
    page.route(
        "**/api/labels",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": labels or []}),
        ),
    )


def _mock_label_stats_object(page: Page, config_uuid: str, stats: dict | None = None):
    page.route(
        f"**/api/training-configs/{config_uuid}/label-stats",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", **(stats or LABEL_STATS_WITH_OBJECT)}),
        ),
    )


def _mock_images_by_label_status(page: Page, config_uuid: str, captured: list):
    def handler(route: Route):
        try:
            captured.append(json.loads(route.request.post_data or "{}"))
        except Exception:
            captured.append({})
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "images": [], "hasMore": False, "totalCount": 0}),
        )

    page.route(f"**/api/training-configs/{config_uuid}/images-by-label-status", handler)


class TestLabelStatsPanelBboxColumns:
    """Clicking the Verified and All clear columns opens the right modal
    and calls the right endpoint with the correct status value."""

    def _setup(self, page: Page, base_url: str, captured: list):
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=3, needs_review=0)
        _mock_label_stats_object(page, "tc-1")
        _mock_labels(page, [OBJECT_LABEL])
        _mock_images_by_label_status(page, "tc-1", captured)
        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()
        # Wait for the label stats table to appear
        page.wait_for_selector("text=Dataset coverage by label", timeout=5000)
        page.wait_for_selector("thead th[title^='Verified boxes']", timeout=3000)

    def test_verified_boxes_click_calls_has_verified_bbox_status(self, page: Page, base_url: str):
        """Clicking the Verified boxes count calls images-by-label-status with status=has_verified_bbox."""
        captured = []
        self._setup(page, base_url, captured)

        # Find the Verified boxes cell (non-zero count) and click it
        verified_cell = page.locator("td", has_text="5").first
        with page.expect_request(
            lambda r: "images-by-label-status" in r.url and r.method == "POST",
            timeout=6000,
        ):
            verified_cell.click()

        wait_until(page, lambda: len(captured) >= 1, "images-by-label-status was not called")
        assert (
            captured[0].get("status") == "has_verified_bbox"
        ), f"Expected status=has_verified_bbox, got: {captured[0].get('status')}"
        assert captured[0].get("labelUuid") == "lbl-obj"

    def test_verified_boxes_click_opens_bbox_view_modal(self, page: Page, base_url: str):
        """The modal opened by Verified boxes click starts in Bboxes view (active view-btn)."""
        captured = []
        self._setup(page, base_url, captured)

        verified_cell = page.locator("td", has_text="5").first
        with page.expect_request(
            lambda r: "images-by-label-status" in r.url and r.method == "POST",
            timeout=6000,
        ):
            verified_cell.click()

        # Modal title should be visible
        modal_title = page.locator("strong", has_text="verified boxes")
        expect(modal_title).to_be_visible(timeout=4000)

        # The Bboxes view button should be active
        bbox_btn = page.locator(".view-btn", has_text="Bboxes")
        expect(bbox_btn).to_have_class(re.compile(r"\bactive\b"), timeout=3000)

        # Close modal
        page.locator("button[title='Close']").click()

    def test_imgs_all_clear_click_calls_co_labels_covered_status(self, page: Page, base_url: str):
        """Clicking the Imgs all clear count calls images-by-label-status with status=co_labels_covered."""
        captured = []
        self._setup(page, base_url, captured)

        # Find the Imgs all clear cell (count=3) and click it
        clear_cell = page.locator("td", has_text="3").first
        with page.expect_request(
            lambda r: "images-by-label-status" in r.url and r.method == "POST",
            timeout=6000,
        ):
            clear_cell.click()

        wait_until(page, lambda: len(captured) >= 1, "images-by-label-status was not called")
        assert (
            captured[0].get("status") == "co_labels_covered"
        ), f"Expected status=co_labels_covered, got: {captured[0].get('status')}"
        assert captured[0].get("labelUuid") == "lbl-obj"

    def test_imgs_all_clear_click_opens_image_view_modal(self, page: Page, base_url: str):
        """The modal opened by Imgs all clear click starts in Images view (default)."""
        captured = []
        self._setup(page, base_url, captured)

        clear_cell = page.locator("td", has_text="3").first
        with page.expect_request(
            lambda r: "images-by-label-status" in r.url and r.method == "POST",
            timeout=6000,
        ):
            clear_cell.click()

        # Modal title should be visible
        modal_title = page.locator("strong", has_text="all clear")
        expect(modal_title).to_be_visible(timeout=4000)

        # The Images view button should be active (default)
        images_btn = page.locator(".view-btn", has_text="Images")
        expect(images_btn).to_have_class(re.compile(r"\bactive\b"), timeout=3000)

        # Close modal
        page.locator("button[title='Close']").click()


# ---------------------------------------------------------------------------
# 6. Compact model-tree preview under the node selector
# ---------------------------------------------------------------------------

PREVIEW_CAT = {"uuid": "lbl-cat", "name": "Cat", "color": "#f00", "parentUuid": None, "scope": "object"}
PREVIEW_KITTEN = {"uuid": "lbl-kitten", "name": "Kitten", "color": "#f88", "parentUuid": "lbl-cat", "scope": "object"}

MODEL_CFG_WITH_SUB = {
    "uuid": "mc-2",
    "name": "Hier Model",
    "description": "",
    "nodes": [
        {"uuid": "n-root", "labelUuid": None, "modelType": "object_detection", "architecture": "yolo_v9_s"},
        {"uuid": "n-cat", "labelUuid": "lbl-cat", "modelType": "crop_classifier", "architecture": "resnet_18"},
        {"uuid": "n-kitten", "labelUuid": "lbl-kitten", "modelType": "crop_classifier", "architecture": "resnet_18"},
    ],
}


def _preview(page: Page):
    return _box(page).locator(".mtp-box--preview .model-tree-panel")


class TestModelTreePreview:
    def test_preview_renders_compact_tree_for_chosen_config(self, page: Page, base_url: str):
        """Choosing a model config renders a compact tree preview: top model with class
        count and the cat sub-model with its child-label count."""
        _mock_datasets(page, [])
        _mock_model_configs(page, [MODEL_CFG_WITH_SUB])
        _mock_training_configs_list(page, [])
        _mock_labels(page, [PREVIEW_CAT, PREVIEW_KITTEN])

        _navigate_to_training_configs(page, base_url)

        config_sel = _editor_body(page).locator("select").first
        config_sel.select_option("mc-2")
        page.wait_for_timeout(300)

        preview = _preview(page)
        expect(preview).to_be_visible()
        expect(preview.get_by_test_id("model-tree-root-node")).to_contain_text("Top model")
        # Compact mode shows counts, not chips: kitten rolls up into cat → 1 class
        expect(preview.get_by_test_id("model-tree-root-node")).to_contain_text("detects: 1 class")
        sub_row = preview.get_by_test_id("model-tree-sub-node")
        expect(sub_row).to_have_count(1)
        expect(sub_row).to_contain_text("Cat")
        expect(sub_row).to_contain_text("distinguishes: 1 child label")

    def test_preview_marks_selected_node_as_training_target(self, page: Page, base_url: str):
        """Selecting the cat node in the node selector highlights it in the preview."""
        _mock_datasets(page, [])
        _mock_model_configs(page, [MODEL_CFG_WITH_SUB])
        _mock_training_configs_list(page, [])
        _mock_labels(page, [PREVIEW_CAT, PREVIEW_KITTEN])

        _navigate_to_training_configs(page, base_url)

        config_sel = _editor_body(page).locator("select").first
        config_sel.select_option("mc-2")
        page.wait_for_timeout(300)

        node_sel = _editor_body(page).locator("select").nth(1)
        node_sel.select_option("n-cat")
        page.wait_for_timeout(300)

        target = _preview(page).locator(".mtp-node--target")
        expect(target).to_have_count(1)
        expect(target).to_contain_text("Cat")
        expect(target).to_contain_text("training target")

    def test_preview_absent_until_config_chosen(self, page: Page, base_url: str):
        """With no model config selected the preview box stays empty."""
        _mock_datasets(page, [])
        _mock_model_configs(page, [MODEL_CFG_WITH_SUB])
        _mock_training_configs_list(page, [])
        _mock_labels(page, [PREVIEW_CAT, PREVIEW_KITTEN])

        _navigate_to_training_configs(page, base_url)
        _editor_body(page).wait_for(timeout=4000)

        expect(_preview(page)).to_have_count(0)


# ---------------------------------------------------------------------------
# Unsaved changes vs. "Start run"
# ---------------------------------------------------------------------------


class TestUnsavedChangesBeforeStartRun:
    """The trainer reads the config from the server, so starting with a dirty
    form would train the previous settings. Editing must be visible, and
    "Start run" must ask before launching."""

    @staticmethod
    def _open_existing(page: Page, base_url: str):
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1")
        _mock_label_stats(page, "tc-1")
        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()
        page.wait_for_timeout(600)

    @staticmethod
    def _make_dirty(page: Page):
        _editor_body(page).locator("input[placeholder='Required']").first.fill("Corrupt Detector v2")
        page.wait_for_timeout(200)

    @staticmethod
    def _start_btn(page: Page):
        return _editor_body(page).locator("[data-testid='training-start-run']")

    def test_editing_shows_unsaved_badge_and_lights_save(self, page: Page, base_url: str):
        self._open_existing(page, base_url)
        badge = page.get_by_test_id("panel-dirty-label")
        expect(badge).to_be_hidden()
        self._make_dirty(page)
        expect(badge).to_be_visible()
        expect(_editor_bar(page).locator("button", has_text="Save")).to_have_class(re.compile("panel-btn--attention"))

    def test_start_while_dirty_prompts_and_cancel_aborts(self, page: Page, base_url: str):
        """Cancelling the prompt creates no task config and starts nothing."""
        start_calls: list = []
        page.route(
            "**/api/tasks/configs/create",
            lambda route: (
                start_calls.append("create"),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "configUuid": "tmp-cfg"}),
                ),
            ),
        )
        self._open_existing(page, base_url)
        self._make_dirty(page)

        self._start_btn(page).click()
        modal = page.get_by_test_id("modal-shell-dialog")
        expect(modal).to_be_visible()
        expect(modal).to_contain_text("unsaved changes")
        modal.locator("button", has_text="Cancel").click()
        page.wait_for_timeout(500)

        assert not start_calls, "Training must not start while the prompt is cancelled"
        # The button is usable again after backing out of the prompt.
        expect(self._start_btn(page)).to_be_enabled()

    def test_save_and_start_saves_before_launching(self, page: Page, base_url: str):
        """'Save & start run' PUTs the config first, then starts the task."""
        calls: list[str] = []

        def _record(name: str, body: dict, methods=None):
            def handler(route: Route):
                if methods and route.request.method not in methods:
                    route.fallback()
                    return
                calls.append(name)
                route.fulfill(status=200, content_type="application/json", body=json.dumps(body))

            return handler

        self._open_existing(page, base_url)
        page.route(
            "**/api/training-configs/tc-1",
            _record("save", {"status": "success", "config": EXISTING_TC}, methods={"PUT"}),
        )
        page.route(
            "**/api/tasks/configs/create",
            _record("create", {"status": "success", "configUuid": "tmp-cfg"}),
        )
        page.route(
            "**/api/tasks/instances/start",
            _record("start", {"status": "success", "taskUuid": "task-1"}),
        )

        self._make_dirty(page)
        self._start_btn(page).click()
        page.get_by_test_id("modal-shell-dialog").locator("button", has_text="Save & start").click()

        wait_until(page, lambda: "start" in calls, "Expected training to start after saving")
        assert calls.index("save") < calls.index("start"), f"Save must precede start, got {calls}"

    def test_start_when_clean_does_not_prompt(self, page: Page, base_url: str):
        """An untouched config starts immediately, as before."""
        calls: list[str] = []
        page.route(
            "**/api/tasks/configs/create",
            lambda route: (
                calls.append("create"),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "configUuid": "tmp-cfg"}),
                ),
            ),
        )
        page.route(
            "**/api/tasks/instances/start",
            lambda route: (
                calls.append("start"),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "taskUuid": "task-1"}),
                ),
            ),
        )
        self._open_existing(page, base_url)

        self._start_btn(page).click()
        wait_until(page, lambda: "start" in calls, "Expected an immediate start with no prompt")

    def test_editor_does_not_embed_job_monitoring(self, page: Page, base_url: str):
        """A configuration screen must not host a progress bar, a log tail or a
        Cancel — that is what made Configuration own job monitoring.

        Those belong to Runs, which owns every running thing uniformly.
        """
        self._open_existing(page, base_url)

        body = _editor_body(page)
        expect(body.locator("[data-testid='training-start-run']")).to_be_visible()
        expect(body.locator("progress")).to_have_count(0)
        expect(body.locator("button", has_text="Cancel")).to_have_count(0)

    def test_start_hands_off_to_runs(self, page: Page, base_url: str):
        """After starting, the user lands on Runs, where the run is visible."""
        page.route(
            "**/api/tasks/configs/create",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configUuid": "tmp-cfg"}),
            ),
        )
        page.route(
            "**/api/tasks/instances/start",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "taskUuid": "task-1"}),
            ),
        )
        self._open_existing(page, base_url)

        self._start_btn(page).click()

        expect(page.locator("#tab-runs")).to_have_class(re.compile(r"\bactive\b"))
        expect(page.locator("#tab-runs-active")).to_have_class(re.compile(r"\bactive\b"))

    def test_start_lands_on_active_even_after_a_visit_to_the_launcher(self, page: Page, base_url: str):
        """Runs remembers its sub-tab, so clicking the main tab alone lands on
        the launcher for anyone who reached this editor from a launch card's ⚙ —
        you press Start and arrive at a list of things to start."""
        page.route(
            "**/api/tasks/configs/create",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configUuid": "tmp-cfg"}),
            ),
        )
        page.route(
            "**/api/tasks/instances/start",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "taskUuid": "task-1"}),
            ),
        )
        self._open_existing(page, base_url)

        # Leave Jobs sitting on "Start a job", the way the ⚙ round trip does.
        page.locator(".main-tab-button[data-tab='tab-runs']").click()
        page.locator("#tab-runs .workspace-tab-button[data-tab='tab-runs-launch']").click()
        expect(page.locator("#tab-runs-launch")).to_have_class(re.compile(r"\bactive\b"))
        page.locator(".main-tab-button[data-tab='tab-configuration']").click()

        self._start_btn(page).click()

        expect(page.locator("#tab-runs-active")).to_have_class(re.compile(r"\bactive\b"))
        expect(page.locator("#tab-runs-launch")).not_to_have_class(re.compile(r"\bactive\b"))


# ---------------------------------------------------------------------------
# Previous training run ("how many more images since we last trained?")
# ---------------------------------------------------------------------------


def _mock_model_training_images(
    page: Page,
    *,
    train: list | None = None,
    val: list | None = None,
    trained_at: str | None = "2026-07-01T09:00:00+00:00",
    node_uuid: str | None = "node-root",
    failure: bool = False,
):
    """Stand in for a trained model's sidecar manifest.

    Deliberately camelCase, matching the real response: ModelTrainingImagesResponse
    inherits StatusResponse's to_camel alias generator.
    """

    def handler(route: Route):
        if failure:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "failure", "message": "Model not found"}),
            )
            return
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "trainImageUuids": train if train is not None else [],
                    "valImageUuids": val if val is not None else [],
                    "trainedAt": trained_at,
                    "modelConfigNodeUuid": node_uuid,
                }
            ),
        )

    page.route("**/api/models/*/training-images", handler)


def _mock_label_stats_for_images(page: Page, config_uuid: str, stats: dict | None = None, captured: list | None = None):
    def handler(route: Route):
        if captured is not None:
            try:
                captured.append(json.loads(route.request.post_data or "{}"))
            except Exception:
                captured.append({})
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", **(stats or {"labelStats": [], "totalImages": 0})}),
        )

    page.route(f"**/api/training-configs/{config_uuid}/label-stats-for-images", handler)


class TestPreviousRunInReadinessPanel:
    """A config that has been trained before shows what went into that run, so the
    number of images added since is readable at a glance rather than inferred."""

    def _setup(self, page: Page, base_url: str, ready: int = 10, **manifest_kwargs):
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=ready, needs_review=0, not_started=0)
        _mock_label_stats(page, "tc-1")
        _mock_label_stats_for_images(page, "tc-1")
        _mock_model_training_images(page, **manifest_kwargs)
        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

    def test_previous_run_image_count_is_shown(self, page: Page, base_url: str):
        self._setup(page, base_url, ready=10, train=["a", "b", "c", "d", "e", "f"], val=["g"])

        line = _box(page).get_by_test_id("training-previous-run")
        expect(line).to_be_visible(timeout=4000)
        expect(_box(page).get_by_test_id("training-previous-run-count")).to_contain_text("7 images")

    def test_delta_since_last_run_is_shown(self, page: Page, base_url: str):
        """10 ready now against 7 last time — the delta is the whole point."""
        self._setup(page, base_url, ready=10, train=["a", "b", "c", "d", "e", "f"], val=["g"])

        delta = _box(page).get_by_test_id("training-previous-run-delta")
        expect(delta).to_be_visible(timeout=4000)
        expect(delta).to_contain_text("+3 ready since")

    def test_shrinking_dataset_is_not_rendered_as_a_gain(self, page: Page, base_url: str):
        """Images get deleted and datasets get re-scoped; "+-4" would be nonsense."""
        self._setup(page, base_url, ready=3, train=["a", "b", "c", "d", "e", "f", "g"])

        delta = _box(page).get_by_test_id("training-previous-run-delta")
        expect(delta).to_be_visible(timeout=4000)
        expect(delta).to_contain_text("-4")
        expect(delta).not_to_contain_text("+")

    def test_no_previous_run_line_when_model_does_not_exist(self, page: Page, base_url: str):
        """The common case for a new config — it must stay silent, not error."""
        self._setup(page, base_url, ready=10, failure=True)

        expect(_box(page).get_by_test_id("training-readiness-ready-count")).to_be_visible(timeout=4000)
        expect(_box(page).get_by_test_id("training-previous-run")).to_have_count(0)

    def test_no_previous_run_line_when_manifest_was_not_recorded(self, page: Page, base_url: str):
        """ "Remember exact image list" can be turned off — then there is nothing to
        compare against and the line must not claim a run of 0 images."""
        self._setup(page, base_url, ready=10, train=[], val=[])

        expect(_box(page).get_by_test_id("training-readiness-ready-count")).to_be_visible(timeout=4000)
        expect(_box(page).get_by_test_id("training-previous-run")).to_have_count(0)

    def test_model_from_a_different_shape_node_is_ignored(self, page: Page, base_url: str):
        """The model name is derived from the config name, so an unrelated model can
        collide with it. The sidecar's node uuid is what rules that out — without
        this check the panel would report a stranger's numbers as this config's."""
        self._setup(page, base_url, ready=10, train=["a", "b"], node_uuid="some-other-node")

        expect(_box(page).get_by_test_id("training-readiness-ready-count")).to_be_visible(timeout=4000)
        expect(_box(page).get_by_test_id("training-previous-run")).to_have_count(0)

    def test_manifest_is_sent_to_the_scoped_label_stats_endpoint(self, page: Page, base_url: str):
        """The per-label previous-run numbers come from replaying the manifest
        through the config's own label-stats query, so the manifest has to reach it."""
        captured: list = []
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=4, needs_review=0)
        _mock_label_stats(page, "tc-1")
        _mock_label_stats_for_images(page, "tc-1", captured=captured)
        _mock_model_training_images(page, train=["img-1", "img-2"], val=["img-3"])

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

        expect(_box(page).get_by_test_id("training-previous-run")).to_be_visible(timeout=4000)
        wait_until(page, lambda: captured, "no POST to /label-stats-for-images")
        assert captured[0]["imageUuids"] == ["img-1", "img-2", "img-3"]


class TestPreviousRunColumnInLabelStats:
    """The coverage table gains a "Last run" column so the delta is visible per
    class, not just in total."""

    def _setup(self, page: Page, base_url: str, prev_all_clear: int = 1, prev_verified: int = 2):
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=8, needs_review=0)
        _mock_labels(page, [{"uuid": "lbl-obj", "name": "Cat", "color": "#ff0000", "parentUuid": None}])
        _mock_label_stats_object(page, "tc-1")
        _mock_label_stats_for_images(
            page,
            "tc-1",
            stats={
                "labelStats": [
                    {
                        "labelUuid": "lbl-obj",
                        "labelScope": "object",
                        "verifiedBboxCount": prev_verified,
                        "unverifiedBboxCount": 0,
                        "coLabelsCoveredCount": prev_all_clear,
                        "coLabelGapsCount": 0,
                        "hasRollup": False,
                        "rollUpParentUuid": None,
                    }
                ],
                "totalImages": 4,
            },
        )
        _mock_model_training_images(page, train=["a", "b", "c", "d"])
        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

    def test_last_run_column_shows_previous_count_and_delta(self, page: Page, base_url: str):
        """Current all-clear images for the label is 3 (LABEL_STATS_WITH_OBJECT); the
        previous run had 1, so the column reads 1 with a +2."""
        self._setup(page, base_url, prev_all_clear=1)

        header = _box(page).locator("th", has_text="Last run").first
        expect(header).to_be_visible(timeout=4000)

        row = _box(page).locator("tr", has=page.locator("td", has_text="Cat")).first
        expect(row).to_contain_text("+2")

    def test_delta_is_measured_against_all_clear_not_verified_boxes(self, page: Page, base_url: str):
        """The per-class delta must count the same thing as the "+N ready since"
        headline — training-ready images. Here all-clear is unchanged (3 then, 3 now)
        while verified boxes grew 2 → 5, so the column must show no delta at all."""
        self._setup(page, base_url, prev_all_clear=3, prev_verified=2)

        header = _box(page).locator("th", has_text="Last run").first
        expect(header).to_be_visible(timeout=4000)

        row = _box(page).locator("tr", has=page.locator("td", has_text="Cat")).first
        expect(row).not_to_contain_text("+3")

    def test_column_is_absent_without_a_previous_run(self, page: Page, base_url: str):
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=8, needs_review=0)
        _mock_labels(page, [{"uuid": "lbl-obj", "name": "Cat", "color": "#ff0000", "parentUuid": None}])
        _mock_label_stats_object(page, "tc-1")
        _mock_model_training_images(page, failure=True)

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

        expect(_box(page).locator("th", has_text="Verified").first).to_be_visible(timeout=4000)
        expect(_box(page).locator("th", has_text="Last run")).to_have_count(0)
