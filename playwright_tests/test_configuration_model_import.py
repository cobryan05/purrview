"""
Tests for the "Import model" form on Configuration > Labels & Tags
(apps/.../ui/models.js: renderModelLabelAssignments -> buildUploadForm).

Covers the Chunk E frontend wiring: uploading a non-ONNX (pytorch) model
file detects its format via /api/models/detect, confirms with the user,
stages it via /api/models/import/stage, and starts a background
ConvertModelTask via the shared task-runner section (createTaskConfig +
startTask), the same plumbing used by the "Curated" download form.

Run with:
    pytest playwright_tests/test_configuration_model_import.py -p no:asyncio -v
"""

import json
import re

import pytest
from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

_IMPORT_BUTTON_TEXT = re.compile(r"^Import$")

# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------


def _navigate_to_model_import_form(page: Page, base_url: str):
    """Open the app, land on Configuration > Labels & Tags (the default
    sub-tab), and expand the "+ Import model" form."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-tag-management']").click()
    page.wait_for_selector("#model-labels-box", timeout=4000)

    page.locator("#model-labels-box button", has_text="+ Import model").click()
    page.wait_for_selector("#model-labels-box input[type=file]", timeout=4000)


# ---------------------------------------------------------------------------
# Mocking helpers
# ---------------------------------------------------------------------------


def _mock_detect(page: Page, response: dict):
    page.route(
        "**/api/models/detect",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(response),
        ),
    )


def _mock_stage(page: Page, calls: list, staging_id: str = "deadbeef"):
    def handle(route: Route):
        calls.append(json.loads(route.request.post_data))
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "message": "staged", "stagingId": staging_id}),
        )

    page.route("**/api/models/import/stage", handle)


def _mock_task_create_and_start(page: Page, create_calls: list):
    def handle_create(route: Route):
        create_calls.append(json.loads(route.request.post_data))
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configUuid": "tc-convert-1"}),
        )

    page.route("**/api/tasks/configs/create", handle_create)
    page.route(
        "**/api/tasks/instances/start",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "taskUuid": "task-convert-1"}),
        ),
    )
    # The task-runner section opens an SSE stream immediately after starting;
    # fulfill it with a single terminal "completed" event so the EventSource
    # settles instead of leaving an unmocked connection hanging in the
    # background for the rest of the test.
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
                            "task-convert-1": {
                                "instance": {"uuid": "task-convert-1", "status": "completed"},
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImportModelConversion:
    def test_pytorch_upload_stages_and_starts_convert_task(self, page: Page, base_url: str):
        """Choosing a non-ONNX (pytorch) file, confirming the conversion
        prompt, and clicking Import stages the file and starts a
        ConvertModelTask with staging_id/model_name/overwrite params."""
        _mock_detect(
            page,
            {
                "status": "success",
                "message": "detected",
                "format": "pytorch",
                "isOnnx": False,
                "conversionSupported": True,
            },
        )
        stage_calls = []
        _mock_stage(page, stage_calls)
        create_calls = []
        _mock_task_create_and_start(page, create_calls)

        # Auto-accept the "convert on training server?" confirm dialog.
        page.on("dialog", lambda dialog: dialog.accept())

        _navigate_to_model_import_form(page, base_url)

        file_input = page.locator("#model-labels-box input[type=file]")
        file_input.set_input_files(
            {
                "name": "my_model.pt",
                "mimeType": "application/octet-stream",
                "buffer": b"\x80\x02fake-pytorch-bytes",
            }
        )
        page.locator("#model-labels-box input[type=text]").first.fill("converted-model")

        page.locator("#model-labels-box button", has_text=_IMPORT_BUTTON_TEXT).click()
        wait_until(
            page,
            lambda: len(stage_calls) >= 1,
            lambda: f"Expected 1 POST to /api/models/import/stage, got {len(stage_calls)}",
        )

        assert len(stage_calls) == 1, f"Expected 1 POST to /api/models/import/stage, got {len(stage_calls)}"
        assert "sourceBase64" in stage_calls[0]

        assert len(create_calls) == 1, f"Expected 1 POST to /api/tasks/configs/create, got {len(create_calls)}"
        payload = create_calls[0]
        assert payload.get("typename") == "ConvertModelTask"
        params = payload.get("params", {})
        assert params.get("staging_id") == "deadbeef"
        assert params.get("model_name") == "converted-model"
        assert params.get("overwrite") is False

    def test_unsupported_format_does_not_stage_or_start_task(self, page: Page, base_url: str):
        """An unsupported/unknown format shows an error and never stages the
        file or starts a task."""
        _mock_detect(
            page,
            {
                "status": "success",
                "message": "detected",
                "format": "tflite",
                "isOnnx": False,
                "conversionSupported": False,
            },
        )
        stage_calls = []
        _mock_stage(page, stage_calls)
        create_calls = []
        _mock_task_create_and_start(page, create_calls)

        dialogs = []
        page.on("dialog", lambda dialog: (dialogs.append(dialog), dialog.dismiss()))

        _navigate_to_model_import_form(page, base_url)

        file_input = page.locator("#model-labels-box input[type=file]")
        file_input.set_input_files(
            {
                "name": "my_model.tflite",
                "mimeType": "application/octet-stream",
                "buffer": b"fake-tflite-bytes",
            }
        )
        page.locator("#model-labels-box input[type=text]").first.fill("converted-model")

        page.locator("#model-labels-box button", has_text=_IMPORT_BUTTON_TEXT).click()
        page.wait_for_timeout(500)

        assert not stage_calls, "Unsupported format must not be staged"
        assert not create_calls, "Unsupported format must not start a task"
        assert not dialogs, "Unsupported format must not prompt for confirmation"
