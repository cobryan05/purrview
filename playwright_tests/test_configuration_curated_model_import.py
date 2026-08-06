"""
Tests for the "Curated" mode of the "Import model" form on
Configuration > Labels & Tags (apps/.../ui/models.js:
renderModelLabelAssignments -> buildCuratedForm).

Covers MORE_YOLO_PLAN.md Section 6's UI requirements: a "via training server"
badge on requires_conversion entries, and the Download & Import button being
disabled (with an explanatory tooltip) when the training server health check
(`GET /api/training/status`, proxied straight through) fails, while direct-ONNX
entries stay fully usable regardless of training-server reachability.

Run with:
    pytest playwright_tests/test_configuration_curated_model_import.py -p no:asyncio -v
"""

import json
import re

import pytest
from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

_DOWNLOAD_BUTTON_TEXT = re.compile(r"^Download & Import$")

_CURATED_MODELS_RESPONSE = {
    "status": "success",
    "models": [
        {"key": "direct-onnx-key", "displayName": "Direct ONNX Model", "requiresConversion": False},
        {"key": "needs-conversion-key", "displayName": "Needs Conversion Model", "requiresConversion": True},
    ],
}


def _navigate_to_curated_form(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-tag-management']").click()
    page.wait_for_selector("#model-labels-box", timeout=4000)

    page.locator("#model-labels-box button", has_text="+ Import model").click()
    page.wait_for_selector("#model-labels-box input[type=file]", timeout=4000)

    page.locator("#model-labels-box button", has_text=re.compile(r"^Curated$")).click()
    page.wait_for_selector("#model-labels-box select", timeout=4000)


def _mock_curated_models(page: Page, response: dict = None):
    body = response if response is not None else _CURATED_MODELS_RESPONSE
    page.route(
        "**/api/models/curated",
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(body)),
    )


def _mock_training_status(page: Page, reachable: bool):
    status_body = {"status": "success", "message": "ok"} if reachable else {"status": "failure", "message": "down"}
    status_code = 200 if reachable else 503

    def handle(route: Route):
        route.fulfill(status=status_code, content_type="application/json", body=json.dumps(status_body))

    page.route("**/api/training/status", handle)


class TestCuratedFormConversionBadgeAndHealthGate:
    def test_direct_onnx_selection_has_no_badge_and_button_enabled(self, page: Page, base_url: str):
        _mock_curated_models(page)
        _mock_training_status(page, reachable=False)  # even unreachable, direct-ONNX must stay usable

        _navigate_to_curated_form(page, base_url)

        select = page.locator("#model-labels-box select")
        select.select_option(label="Direct ONNX Model")

        download_btn = page.locator("#model-labels-box button", has_text=_DOWNLOAD_BUTTON_TEXT)
        expect(download_btn).to_be_enabled()

    def test_conversion_selection_shows_badge_and_disables_button_when_unreachable(self, page: Page, base_url: str):
        _mock_curated_models(page)
        _mock_training_status(page, reachable=False)

        _navigate_to_curated_form(page, base_url)

        select = page.locator("#model-labels-box select")
        select.select_option(label="Needs Conversion Model (via training server)")

        download_btn = page.locator("#model-labels-box button", has_text=_DOWNLOAD_BUTTON_TEXT)
        expect(download_btn).to_be_disabled()

    def test_conversion_selection_enabled_when_training_server_reachable(self, page: Page, base_url: str):
        _mock_curated_models(page)
        _mock_training_status(page, reachable=True)

        _navigate_to_curated_form(page, base_url)

        select = page.locator("#model-labels-box select")
        select.select_option(label="Needs Conversion Model (via training server)")

        download_btn = page.locator("#model-labels-box button", has_text=_DOWNLOAD_BUTTON_TEXT)
        expect(download_btn).to_be_enabled()

    def test_selecting_conversion_entry_starts_import_curated_model_task(self, page: Page, base_url: str):
        """Clicking Download & Import on a requires_conversion entry starts
        ImportCuratedModelTask (not ImportModelTask)."""
        _mock_curated_models(page)
        _mock_training_status(page, reachable=True)

        create_calls = []

        def handle_create(route: Route):
            create_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configUuid": "tc-curated-1"}),
            )

        page.route("**/api/tasks/configs/create", handle_create)
        page.route(
            "**/api/tasks/instances/start",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "taskUuid": "task-curated-1"}),
            ),
        )
        page.route(
            "**/api/tasks/stream",
            lambda route: route.fulfill(
                status=200,
                content_type="text/event-stream",
                body=(
                    "data: "
                    + json.dumps(
                        {
                            "tasks": {
                                "task-curated-1": {
                                    "instance": {"uuid": "task-curated-1", "status": "completed"},
                                    "progress": 100,
                                    "logLines": ["done"],
                                }
                            }
                        }
                    )
                    + "\n\n"
                ),
            ),
        )

        _navigate_to_curated_form(page, base_url)

        select = page.locator("#model-labels-box select")
        select.select_option(label="Needs Conversion Model (via training server)")
        page.locator("#model-labels-box input[type=text]").first.fill("my-converted-model")

        page.locator("#model-labels-box button", has_text=_DOWNLOAD_BUTTON_TEXT).click()
        wait_until(
            page,
            lambda: len(create_calls) >= 1,
            lambda: f"Expected 1 POST to /api/tasks/configs/create, got {len(create_calls)}",
        )

        assert len(create_calls) == 1, f"Expected 1 POST to /api/tasks/configs/create, got {len(create_calls)}"
        payload = create_calls[0]
        assert payload.get("typename") == "ImportCuratedModelTask"
        params = payload.get("params", {})
        assert params.get("curated_key") == "needs-conversion-key"
        assert params.get("model_name") == "my-converted-model"
