"""
Tests for the Training Configuration UI (Configuration > Training).

Exercises renderTrainingConfigManager() rendered inside
Configuration > Training > #training-config-box
(apps/.../ui/trainingConfigs.js):

  - DOM structure: list panel + create form
  - Create flow: fill form → POST /api/training-configs with correct payload
  - Readiness panel: shows ready/needs-review counts when a config is selected
  - "Review unverified images" button: fires OPEN_REVIEW_WITH_FILTERS event,
    activates the Review tab, and causes the review queue fetch to include
    labelUuids + labelVerificationCheck=has_unverified

Run with:
    pytest playwright_tests/test_configuration_training_configs.py -p no:xdist -p no:asyncio -n 0 -v
"""

import json

import pytest
from playwright.sync_api import Page, Route, expect

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

LABEL_CORRUPT = {"uuid": "lbl-corrupt", "name": "Corrupt", "color": "#ff0000", "parentUuid": None}
LABEL_BLURRY = {"uuid": "lbl-blurry", "name": "Blurry", "color": "#0000ff", "parentUuid": None}

DATASET_ALPHA = {"uuid": "ds-alpha", "name": "Alpha Dataset", "description": "", "filterJson": {}, "labelUuids": []}

MODEL_CFG = {"uuid": "mc-1", "name": "My Model", "description": "", "nodes": []}

EXISTING_TC = {
    "uuid": "tc-1",
    "name": "Corrupt Detector",
    "description": "Finds corrupt images",
    "trainerType": "image_classifier",
    "datasetUuids": ["ds-alpha"],
    "modelConfigUuid": None,
    "hyperparamsJson": {},
    "labelUuids": ["lbl-corrupt", "lbl-blurry"],
    "createdAt": "2026-01-01T00:00:00",
    "updatedAt": "2026-01-01T00:00:00",
}


# ---------------------------------------------------------------------------
# Route-interception helpers
# ---------------------------------------------------------------------------


def _mock_labels(page: Page, labels: list | None = None):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "/api/labels/" not in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": labels or []}),
        ),
    )


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


def _mock_readiness(page: Page, config_uuid: str, ready: int = 3, needs_review: int = 2,
                    label_uuids: list | None = None):
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
                "labelUuids": label_uuids or ["lbl-corrupt", "lbl-blurry"],
            }),
        ),
    )


def _mock_review_queue(page: Page):
    """Intercept review-queue POST with an empty result."""
    page.route(
        "**/api/images/review",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "status": "success",
                "images": [],
                "hasMore": False,
                "totalCount": 0,
            }),
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


def _label_checkbox(page: Page, label_uuid: str):
    """The checkbox for a label in the target-labels tree."""
    return _box(page).locator(f".panel-tree-box input[value='{label_uuid}']")


# ---------------------------------------------------------------------------
# 1. Structure
# ---------------------------------------------------------------------------


class TestTrainingConfigPaneStructure:
    def test_pane_renders_list_and_create_form(self, page: Page, base_url: str):
        """The pane shows an existing config in the list and defaults to a create form."""
        _mock_labels(page, [LABEL_CORRUPT, LABEL_BLURRY])
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])

        _navigate_to_training_configs(page, base_url)

        # Left panel: existing config name and meta visible in list
        item = _list_item(page, "tc-1")
        expect(item).to_be_visible()
        expect(item).to_contain_text("Corrupt Detector")
        expect(item).to_contain_text("image_classifier")

        # Footer new-config button
        expect(_box(page).locator("button", has_text="+ New")).to_be_visible()

        # Right panel defaults to create form (bar says "New Training Config")
        expect(_editor_bar(page)).to_contain_text("New Training Config")

    def test_trainer_type_select_is_present(self, page: Page, base_url: str):
        """The create form includes a trainer type select with known options."""
        _mock_labels(page, [])
        _mock_datasets(page, [])
        _mock_model_configs(page, [])
        _mock_training_configs_list(page, [])

        _navigate_to_training_configs(page, base_url)

        # Wait for the editor form to render (it's async — _showForm is awaited)
        trainer_select = _editor_body(page).locator("select").first
        expect(trainer_select).to_be_visible(timeout=4000)
        options = trainer_select.locator("option").all_text_contents()
        assert any("yolo" in o.lower() or "classifier" in o.lower() for o in options), \
            f"Expected trainer-type options, got: {options}"

    def test_target_labels_appear_as_checkboxes(self, page: Page, base_url: str):
        """Labels from the API appear as checkboxes in the target-label tree."""
        _mock_labels(page, [LABEL_CORRUPT, LABEL_BLURRY])
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [])
        _mock_training_configs_list(page, [])

        _navigate_to_training_configs(page, base_url)

        expect(_label_checkbox(page, "lbl-corrupt")).to_be_visible()
        expect(_label_checkbox(page, "lbl-blurry")).to_be_visible()
        # Both unchecked on the create form
        expect(_label_checkbox(page, "lbl-corrupt")).not_to_be_checked()
        expect(_label_checkbox(page, "lbl-blurry")).not_to_be_checked()


# ---------------------------------------------------------------------------
# 2. Create flow
# ---------------------------------------------------------------------------


class TestCreateTrainingConfig:
    def test_create_sends_correct_payload(self, page: Page, base_url: str):
        """Filling the form and saving POSTs the right payload."""
        _mock_labels(page, [LABEL_CORRUPT, LABEL_BLURRY])
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [])
        _mock_training_configs_list(page, [])

        captured = []

        def handle_create(route: Route):
            captured.append(json.loads(route.request.post_data))
            # Return the "created" config so the UI can refresh
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "config": {
                    **EXISTING_TC, "uuid": "tc-new", "name": "My Classifier",
                    "labelUuids": ["lbl-corrupt"],
                }}),
            )

        page.route("**/api/training-configs",
                   lambda r: handle_create(r) if r.request.method == "POST" else r.fallback())

        _navigate_to_training_configs(page, base_url)

        # Fill in name
        name_input = _editor_body(page).locator("input[placeholder='Required']").first
        name_input.fill("My Classifier")

        # Check the dataset checkbox (datasets now use checkboxes, not a select)
        dataset_cb = _box(page).locator(f"input[type='checkbox'][value='ds-alpha']")
        dataset_cb.check()

        # Check the Corrupt label checkbox
        _label_checkbox(page, "lbl-corrupt").check()

        # Click the create/save button in the bar
        _editor_bar(page).locator("button", has_text="Create").or_(
            _editor_bar(page).locator("button", has_text="Save")
        ).click()

        page.wait_for_timeout(800)

        assert len(captured) == 1, f"Expected 1 POST, got {len(captured)}"
        payload = captured[0]
        assert payload.get("name") == "My Classifier"
        assert "lbl-corrupt" in payload.get("labelUuids", [])
        assert "ds-alpha" in payload.get("datasetUuids", []), \
            f"Expected ds-alpha in datasetUuids; got: {payload.get('datasetUuids')}"

    def test_existing_config_labels_pre_checked(self, page: Page, base_url: str):
        """Opening an existing config pre-checks its labelUuids in the tree."""
        _mock_labels(page, [LABEL_CORRUPT, LABEL_BLURRY])
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1")

        _navigate_to_training_configs(page, base_url)

        _list_item(page, "tc-1").click()
        page.wait_for_timeout(400)

        # Both labels should be checked since EXISTING_TC has both
        expect(_label_checkbox(page, "lbl-corrupt")).to_be_checked(timeout=3000)
        expect(_label_checkbox(page, "lbl-blurry")).to_be_checked(timeout=3000)


# ---------------------------------------------------------------------------
# 3. Readiness panel
# ---------------------------------------------------------------------------


class TestReadinessPanel:
    def test_readiness_counts_are_displayed(self, page: Page, base_url: str):
        """Selecting an existing config shows the readiness panel with counts."""
        _mock_labels(page, [LABEL_CORRUPT, LABEL_BLURRY])
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [MODEL_CFG])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=3, needs_review=2)

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

        readiness_panel = _editor_body(page).locator("div", has_text="Training readiness").first
        expect(readiness_panel).to_be_visible(timeout=3000)
        # Both counts appear somewhere in the panel
        expect(readiness_panel).to_contain_text("3")
        expect(readiness_panel).to_contain_text("2")

    def test_review_button_hidden_when_no_needs_review(self, page: Page, base_url: str):
        """'Review unverified images' button is hidden when needsReviewCount is 0."""
        _mock_labels(page, [LABEL_CORRUPT])
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=5, needs_review=0)

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

        # Wait for readiness panel to finish loading
        page.wait_for_selector(
            "#training-config-box div:has-text('Training readiness')", timeout=3000
        )
        # Button must exist but stay hidden (display:none)
        btn = _box(page).locator("button", has_text="Review unverified images")
        expect(btn).to_have_count(1)
        expect(btn).to_be_hidden()

    def test_review_button_visible_when_needs_review(self, page: Page, base_url: str):
        """'Review unverified images' button appears when needsReviewCount > 0."""
        _mock_labels(page, [LABEL_CORRUPT])
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=3, needs_review=2)

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

        btn = _box(page).locator("button", has_text="Review unverified images")
        expect(btn).to_be_visible(timeout=3000)

    def test_review_button_switches_to_review_tab_with_filters(self, page: Page, base_url: str):
        """Clicking 'Review unverified images' activates the Review tab and the
        subsequent review-queue POST includes labelUuids + labelVerificationCheck."""
        _mock_labels(page, [LABEL_CORRUPT, LABEL_BLURRY])
        _mock_datasets(page, [DATASET_ALPHA])
        _mock_model_configs(page, [])
        _mock_training_configs_list(page, [EXISTING_TC])
        _mock_training_config_get(page, EXISTING_TC)
        _mock_readiness(page, "tc-1", ready=1, needs_review=3)

        _navigate_to_training_configs(page, base_url)
        _list_item(page, "tc-1").click()

        btn = _box(page).locator("button", has_text="Review unverified images")
        expect(btn).to_be_visible(timeout=3000)

        # Capture all review-queue POSTs; use wait_for_request to block until one arrives.
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

        # Use expect_request to block until the first queue POST arrives (up to 8 s).
        with page.expect_request(
            lambda r: "/api/images/review" in r.url and r.method == "POST",
            timeout=8000,
        ):
            btn.click()

        # Review tab pane must be active
        import re as _re
        expect(page.locator("#tab-review")).to_have_class(_re.compile(r"\bactive\b"), timeout=4000)

        # The last captured request must carry the correct filter fields.
        # (There may be an initial render with default state and then a re-render
        # with the applied state; we check the final request.)
        page.wait_for_timeout(600)  # allow any second render pass to complete
        last = captured_queue_requests[-1]
        assert last.get("labelVerificationCheck") == "has_unverified", \
            f"Expected labelVerificationCheck='has_unverified'; got: {last}"
        assert set(last.get("labelUuids", [])) == {"lbl-corrupt", "lbl-blurry"}, \
            f"Expected labelUuids to match config labels; got: {last.get('labelUuids')}"
