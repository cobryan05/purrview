"""
Tests for the Task Management configuration UI.

Exercises renderTaskConfigs() and renderActiveTasks() inside
Configuration > Task Management:
  - DOM structure (task-config-list and active-tasks-list present)
  - Task config list loading (configs appear with Start/Delete buttons)
  - Task type dropdown populated from /api/tasks/types/list
  - Full create flow: select type → schema fields appear → fill name → Add Config → API called
  - Task name required validation
  - Active task list loading (running/completed rows with correct action buttons)
  - Task lifecycle button flows (start, stop/cancel, delete)
  - Delete config flow (confirm dialog)

Run with:
    pytest tests/playwright/test_configuration_tasks.py -p no:xdist -v
"""
import json

import pytest
from playwright.sync_api import Page, Route, expect

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

TASK_TYPE = "RefreshDatabaseTask"

# Minimal schema — boolean + select fields; tests boolean-checkbox and select rendering.
REFRESH_DB_SCHEMA = {
    "meta": {"order": ["remove_deleted", "sync_mode"]},
    "remove_deleted": {
        "type": "boolean",
        "label": "Remove deleted images",
        "description": "Delete DB records for images that are no longer on disk.",
        "default": True,
    },
    "sync_mode": {
        "type": "select",
        "label": "Sync mode",
        "options": [
            {"value": "merge", "label": "Merge"},
            {"value": "replace_db_with_json", "label": "Replace DB with JSON"},
        ],
        "default": "merge",
    },
}

FAKE_CONFIG = {
    "uuid": "eeeeeeee-0000-0000-0000-000000000001",
    "name": "Nightly Refresh",
    "typename": TASK_TYPE,
    "params_json": {"remove_deleted": True, "sync_mode": "merge"},
}

RUNNING_TASK = {
    "instance": {
        "uuid": "ffffffff-0000-0000-0000-000000000001",
        "status": "running",
        "started_at": "2024-01-01T00:00:00",
        "ended_at": None,
    },
    "config": {"name": "Nightly Refresh", "uuid": FAKE_CONFIG["uuid"]},
    "progress": 42.0,
    "is_paused_in_place": False,
    "log_lines": ["Starting…", "Processing images…"],
}

COMPLETED_TASK = {
    "instance": {
        "uuid": "ffffffff-0000-0000-0000-000000000002",
        "status": "completed",
        "started_at": "2024-01-01T00:00:00",
        "ended_at": "2024-01-01T00:05:00",
    },
    "config": {"name": "Nightly Refresh", "uuid": FAKE_CONFIG["uuid"]},
    "progress": 100.0,
    "is_paused_in_place": False,
    "log_lines": ["Done."],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _navigate_to_tasks(page: Page, base_url: str):
    """Open the app and activate Configuration > Task Management."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-task-management']").click()
    page.wait_for_load_state("networkidle")


def _mock_task_types(page: Page, types: list | None = None):
    page.route(
        "**/api/tasks/types/list",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "types": types or [TASK_TYPE]}),
        ),
    )


def _mock_task_schema(page: Page, schema: dict | None = None):
    page.route(
        "**/api/tasks/types/schema",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"schemas": {TASK_TYPE: schema or REFRESH_DB_SCHEMA}}),
        ),
    )


def _mock_task_configs(page: Page, configs: dict | None = None):
    configs = configs or {}
    page.route(
        "**/api/tasks/configs/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": configs}),
        ),
    )


def _mock_active_tasks(page: Page, tasks: dict | None = None):
    tasks = tasks or {}
    page.route(
        "**/api/tasks/instances/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "tasks": tasks}),
        ),
    )


def _setup_full_task_mocks(page: Page, configs=None, tasks=None):
    """Set up all task-management API mocks at once."""
    _mock_task_types(page)
    _mock_task_schema(page)
    _mock_task_configs(page, configs)
    _mock_active_tasks(page, tasks)


# ---------------------------------------------------------------------------
# DOM structure
# ---------------------------------------------------------------------------


class TestTaskPaneStructure:
    def test_task_config_list_container_visible(self, page: Page, base_url: str):
        """The #task-config-list div is present on the Task Management tab."""
        _setup_full_task_mocks(page)
        _navigate_to_tasks(page, base_url)
        expect(page.locator("#task-config-list")).to_be_visible()

    def test_active_tasks_list_container_present(self, page: Page, base_url: str):
        """The #active-tasks-list div is present in the DOM on the Task Management tab.
        The div may be empty (zero height) when no tasks exist, but it must be attached."""
        _setup_full_task_mocks(page)
        _navigate_to_tasks(page, base_url)
        # Use to_be_attached: an empty div is not "visible" to Playwright (zero height)
        expect(page.locator("#active-tasks-list")).to_be_attached()

    def test_create_task_config_header_present(self, page: Page, base_url: str):
        """A 'Create New Task Config' heading appears in the task config section."""
        _setup_full_task_mocks(page)
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list h3", timeout=4000)
        headers = page.locator("#task-config-list h3").all_text_contents()
        assert any("Create" in h for h in headers), f"No 'Create' header. Got: {headers}"

    def test_no_js_errors_on_task_tab(self, page: Page, base_url: str):
        """No console errors while loading the Task Management pane."""
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        _setup_full_task_mocks(page)
        _navigate_to_tasks(page, base_url)
        page.wait_for_timeout(1200)
        uncaught = [e for e in errors if "TypeError" in e or "Uncaught" in e]
        assert uncaught == [], f"Uncaught JS errors: {uncaught}"


# ---------------------------------------------------------------------------
# Task config list loading
# ---------------------------------------------------------------------------


class TestTaskConfigList:
    def test_existing_configs_rendered(self, page: Page, base_url: str):
        """Configs returned from /api/tasks/configs/get appear with the config name."""
        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG})
        _navigate_to_tasks(page, base_url)
        page.wait_for_timeout(800)
        content = page.locator("#task-config-list").inner_text()
        assert FAKE_CONFIG["name"] in content, f"Config name not found in: {content!r}"

    def test_existing_config_has_start_button(self, page: Page, base_url: str):
        """Each task config row provides a Start button."""
        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list button[title='Start']", timeout=4000)
        expect(page.locator("#task-config-list button[title='Start']").first).to_be_visible()

    def test_existing_config_has_delete_button(self, page: Page, base_url: str):
        """Each task config row provides a Delete button."""
        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list button[title='Delete']", timeout=4000)
        expect(page.locator("#task-config-list button[title='Delete']").first).to_be_visible()


# ---------------------------------------------------------------------------
# Task type dropdown
# ---------------------------------------------------------------------------


class TestTaskTypeDropdown:
    def test_type_dropdown_populated_from_api(self, page: Page, base_url: str):
        """The task type <select> lists types returned by /api/tasks/types/list."""
        types = ["RefreshDatabaseTask", "CollectTrainingImagesTask"]
        _mock_task_types(page, types)
        _mock_task_schema(page)
        _mock_task_configs(page, {})
        _mock_active_tasks(page, {})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list select", timeout=5000)

        # The last select in the create form (not an existing config's select)
        options = page.locator("#task-config-list select").last.locator("option").all_text_contents()
        for t in types:
            assert t in options, f"Type '{t}' not in dropdown options: {options}"

    def test_selecting_type_fetches_schema(self, page: Page, base_url: str):
        """Selecting a task type triggers a /api/tasks/types/schema call."""
        schema_calls: list[str] = []

        def handle_schema(route: Route):
            body = json.loads(route.request.post_data)
            schema_calls.append(str(body))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"schemas": {TASK_TYPE: REFRESH_DB_SCHEMA}}),
            )

        _mock_task_types(page)
        page.route("**/api/tasks/types/schema", handle_schema)
        _mock_task_configs(page, {})
        _mock_active_tasks(page, {})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list select", timeout=5000)
        page.locator("#task-config-list select").last.select_option(TASK_TYPE)
        page.wait_for_timeout(800)

        assert schema_calls, "Expected /api/tasks/types/schema to be called after type selection"
        assert TASK_TYPE in schema_calls[0]

    def test_schema_fields_appear_after_type_selection(self, page: Page, base_url: str):
        """After selecting a task type, schema-driven fields appear in the create form."""
        _setup_full_task_mocks(page, configs={}, tasks={})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list select", timeout=5000)

        page.locator("#task-config-list select").last.select_option(TASK_TYPE)
        page.wait_for_timeout(1000)

        # The schema has a "Remove deleted images" boolean → rendered as a checkbox with that label
        content = page.locator("#task-config-list").inner_html()
        assert "Remove deleted images" in content or "remove_deleted" in content, (
            f"Expected schema field label in content; got snippet: {content[:500]!r}"
        )


# ---------------------------------------------------------------------------
# Full create task config flow
# ---------------------------------------------------------------------------


class TestCreateTaskConfig:
    def test_create_calls_api_with_correct_payload(self, page: Page, base_url: str):
        """Full create flow: select type → fill name → Add Config → /api/tasks/configs/create called."""
        create_calls: list[dict] = []

        def handle_create(route: Route):
            create_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configUuid": FAKE_CONFIG["uuid"]}),
            )

        _setup_full_task_mocks(page, configs={}, tasks={})
        page.route("**/api/tasks/configs/create", handle_create)

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list select", timeout=5000)

        # 1. Select task type
        page.locator("#task-config-list select").last.select_option(TASK_TYPE)
        page.wait_for_timeout(1000)  # wait for schema fetch + render

        # 2. Fill in config name
        page.locator("#task-config-list input[placeholder='Enter new config name']").fill("My Refresh Job")

        # 3. Click Add Config
        page.locator("#task-config-list button[title='Add Config']").click()
        page.wait_for_timeout(800)

        assert create_calls, "Expected POST to /api/tasks/configs/create"
        payload = create_calls[0]
        assert payload.get("name") == "My Refresh Job"
        assert payload.get("typename") == TASK_TYPE

    def test_create_requires_name(self, page: Page, base_url: str):
        """Clicking Add Config without a name does not call /api/tasks/configs/create."""
        calls: list[bool] = []
        page.route("**/api/tasks/configs/create", lambda route: (calls.append(True), route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configUuid": FAKE_CONFIG["uuid"]}),
        )))
        _setup_full_task_mocks(page, configs={}, tasks={})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list select", timeout=5000)
        page.locator("#task-config-list select").last.select_option(TASK_TYPE)
        page.wait_for_timeout(800)

        # Leave the name field empty and click Add Config
        page.locator("#task-config-list button[title='Add Config']").click()
        page.wait_for_timeout(500)

        assert not calls, "API must NOT be called when name is empty"

    def test_create_sends_schema_params(self, page: Page, base_url: str):
        """The create payload includes the schema field values."""
        create_calls: list[dict] = []

        def handle_create(route: Route):
            create_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configUuid": FAKE_CONFIG["uuid"]}),
            )

        _setup_full_task_mocks(page, configs={}, tasks={})
        page.route("**/api/tasks/configs/create", handle_create)

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list select", timeout=5000)
        page.locator("#task-config-list select").last.select_option(TASK_TYPE)
        page.wait_for_timeout(1000)

        page.locator("#task-config-list input[placeholder='Enter new config name']").fill("Refresh Job")
        page.locator("#task-config-list button[title='Add Config']").click()
        page.wait_for_timeout(800)

        assert create_calls, "Expected POST to /api/tasks/configs/create"
        payload = create_calls[0]
        assert "params" in payload, f"Payload missing 'params': {payload}"
        # Default boolean field 'remove_deleted' should be present
        assert "remove_deleted" in payload["params"], (
            f"'remove_deleted' not in params: {payload['params']}"
        )


# ---------------------------------------------------------------------------
# Active tasks list
# ---------------------------------------------------------------------------


class TestActiveTasksList:
    def test_active_tasks_rendered(self, page: Page, base_url: str):
        """Running tasks appear in the #active-tasks-list section."""
        task_id = RUNNING_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: RUNNING_TASK})

        _navigate_to_tasks(page, base_url)
        page.wait_for_timeout(1200)

        content = page.locator("#active-tasks-list").inner_text()
        assert RUNNING_TASK["config"]["name"] in content or "running" in content.lower(), (
            f"Running task not visible. Content: {content!r}"
        )

    def test_running_task_has_stop_button(self, page: Page, base_url: str):
        """A running task shows a Stop button."""
        task_id = RUNNING_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: RUNNING_TASK})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#active-tasks-list button[title='Stop']", timeout=5000)
        expect(page.locator("#active-tasks-list button[title='Stop']").first).to_be_visible()

    def test_running_task_has_pause_button(self, page: Page, base_url: str):
        """A running task shows a Pause button (pause-in-place)."""
        task_id = RUNNING_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: RUNNING_TASK})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#active-tasks-list button[title='Pause']", timeout=5000)
        expect(page.locator("#active-tasks-list button[title='Pause']").first).to_be_visible()

    def test_completed_task_has_delete_button(self, page: Page, base_url: str):
        """A completed task shows a Delete button."""
        task_id = COMPLETED_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: COMPLETED_TASK})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#active-tasks-list button[title='Delete']", timeout=5000)
        expect(page.locator("#active-tasks-list button[title='Delete']").first).to_be_visible()

    def test_completed_section_has_clear_completed_button(self, page: Page, base_url: str):
        """When completed tasks exist, a 'Clear completed' button appears."""
        task_id = COMPLETED_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: COMPLETED_TASK})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#active-tasks-list button", timeout=5000)
        # Button text is rendered as text (not emoji-button), so use has_text
        clear_btn = page.locator("#active-tasks-list button", has_text="Clear completed")
        expect(clear_btn).to_be_visible()


# ---------------------------------------------------------------------------
# Task lifecycle button flows
# ---------------------------------------------------------------------------


class TestTaskLifecycleButtons:
    def test_start_task_calls_api(self, page: Page, base_url: str):
        """Clicking Start on a task config POSTs to /api/tasks/instances/start."""
        start_calls: list[dict] = []

        def handle_start(route: Route):
            start_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "taskUuid": "new-task-uuid"}),
            )

        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG}, tasks={})
        page.route("**/api/tasks/instances/start", handle_start)

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list button[title='Start']", timeout=5000)
        page.locator("#task-config-list button[title='Start']").first.click()
        page.wait_for_timeout(600)

        assert start_calls, "Expected POST to /api/tasks/instances/start"
        assert start_calls[0].get("configUuid") == FAKE_CONFIG["uuid"]

    def test_stop_running_task_calls_cancel(self, page: Page, base_url: str):
        """Clicking Stop on a running task POSTs to /api/tasks/instances/cancel."""
        cancel_calls: list[dict] = []

        def handle_cancel(route: Route):
            cancel_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success"}),
            )

        task_id = RUNNING_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: RUNNING_TASK})
        page.route("**/api/tasks/instances/cancel", handle_cancel)

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#active-tasks-list button[title='Stop']", timeout=5000)
        page.locator("#active-tasks-list button[title='Stop']").first.click()
        page.wait_for_timeout(600)

        assert cancel_calls, "Expected POST to /api/tasks/instances/cancel"
        assert task_id in cancel_calls[0].get("taskUuids", [])

    def test_delete_completed_task_shows_confirm(self, page: Page, base_url: str):
        """Clicking Delete on a completed task opens a confirm() dialog."""
        task_id = COMPLETED_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: COMPLETED_TASK})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#active-tasks-list button[title='Delete']", timeout=5000)

        dialogs: list[str] = []
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
        page.locator("#active-tasks-list button[title='Delete']").first.click()
        page.wait_for_timeout(400)

        assert dialogs, "Expected a confirm() dialog"

    def test_delete_completed_task_accepted_calls_api(self, page: Page, base_url: str):
        """Accepting delete on a completed task POSTs to /api/tasks/instances/delete."""
        delete_calls: list[dict] = []

        def handle_delete(route: Route):
            delete_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success"}),
            )

        task_id = COMPLETED_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: COMPLETED_TASK})
        page.route("**/api/tasks/instances/delete", handle_delete)

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#active-tasks-list button[title='Delete']", timeout=5000)

        page.on("dialog", lambda d: d.accept())
        page.locator("#active-tasks-list button[title='Delete']").first.click()
        page.wait_for_timeout(600)

        assert delete_calls, "Expected POST to /api/tasks/instances/delete"
        assert task_id in delete_calls[0].get("taskUuids", [])


# ---------------------------------------------------------------------------
# Delete task config flow
# ---------------------------------------------------------------------------


class TestDeleteTaskConfig:
    def test_delete_config_shows_confirm(self, page: Page, base_url: str):
        """Clicking Delete on a task config opens a confirm() dialog."""
        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG}, tasks={})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list button[title='Delete']", timeout=5000)

        dialogs: list[str] = []
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
        page.locator("#task-config-list button[title='Delete']").first.click()
        page.wait_for_timeout(400)

        assert dialogs, "Expected a confirm() dialog"
        assert FAKE_CONFIG["name"] in dialogs[0], "Dialog should mention the config name"

    def test_delete_config_accepted_calls_api(self, page: Page, base_url: str):
        """Accepting delete on a config POSTs to /api/tasks/configs/delete."""
        delete_calls: list[dict] = []

        def handle_delete(route: Route):
            delete_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success"}),
            )

        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG}, tasks={})
        page.route("**/api/tasks/configs/delete", handle_delete)

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list button[title='Delete']", timeout=5000)

        page.on("dialog", lambda d: d.accept())
        page.locator("#task-config-list button[title='Delete']").first.click()
        page.wait_for_timeout(600)

        assert delete_calls, "Expected POST to /api/tasks/configs/delete"
        assert FAKE_CONFIG["uuid"] in delete_calls[0].get("configUuids", [])
