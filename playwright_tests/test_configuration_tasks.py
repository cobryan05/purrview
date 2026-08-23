"""
Tests for the Task Management UI.

Exercises renderTaskConfigs() (Operations > Tasks) and renderActiveTasks()
(Operations > Activity):
  - DOM structure (list-editor screen on Tasks, instance list on Activity)
  - Task config list loading (configs appear as list rows)
  - Task type dropdown populated from /api/tasks/types/list
  - Full create flow: select type -> schema fields appear -> fill name -> Create -> API called
  - Task name required validation
  - Active task list loading (running/completed cards with correct action buttons)
  - Task lifecycle button flows (start, stop/cancel, delete)
  - Delete config flow (modal confirm)

Task configs are edited in a list-beside-editor screen (ui/utils/listEditorScreen.js),
the same shape as Datasets and Filter Chains: pick a row on the left, edit it on the
right, Save/Delete live in the editor's sticky bar. There is no always-expanded
"create" form any more -- the editor shows a blank New Task Config by default, and
"+ New Config" in the list footer returns to it.

Run with:
    pytest playwright_tests/test_configuration_tasks.py -p no:asyncio -v
"""

import json
import re

import pytest
from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

TASK_TYPE = "DatabaseMaintenanceTask"

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
    "params": {"remove_deleted": True, "sync_mode": "merge"},
}

RUNNING_TASK = {
    "instance": {
        "uuid": "ffffffff-0000-0000-0000-000000000001",
        "status": "running",
        "startedAt": "2024-01-01T00:00:00",
        "endedAt": None,
    },
    "config": {"name": "Nightly Refresh", "uuid": FAKE_CONFIG["uuid"]},
    "progress": 42.0,
    "isPausedInPlace": False,
    "logLines": ["Starting…", "Processing images…"],
}

INDETERMINATE_TASK = {
    "instance": {
        "uuid": "ffffffff-0000-0000-0000-000000000003",
        "status": "running",
        "startedAt": "2024-01-01T00:00:00",
        "endedAt": None,
    },
    "config": {"name": "Live Collector", "uuid": FAKE_CONFIG["uuid"]},
    "progress": None,
    "isPausedInPlace": False,
    "logLines": ["Watching camera…"],
}

ZERO_PROGRESS_TASK = {
    "instance": {
        "uuid": "ffffffff-0000-0000-0000-000000000004",
        "status": "running",
        "startedAt": "2024-01-01T00:00:00",
        "endedAt": None,
    },
    "config": {"name": "Batch Job", "uuid": FAKE_CONFIG["uuid"]},
    "progress": 0.0,
    "isPausedInPlace": False,
    "logLines": ["Starting…"],
}

COMPLETED_TASK = {
    "instance": {
        "uuid": "ffffffff-0000-0000-0000-000000000002",
        "status": "completed",
        "startedAt": "2024-01-01T00:00:00",
        "endedAt": "2024-01-01T00:05:00",
    },
    "config": {"name": "Nightly Refresh", "uuid": FAKE_CONFIG["uuid"]},
    "progress": 100.0,
    "isPausedInPlace": False,
    "logLines": ["Done."],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _navigate_to_configuration(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")


def _navigate_to_tasks(page: Page, base_url: str):
    """Open the app and activate Configuration > Tasks (the config editor)."""
    _navigate_to_configuration(page, base_url)
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-task-management']").click()
    page.wait_for_load_state("networkidle")


def _navigate_to_activity(page: Page, base_url: str):
    """Open the app and activate Runs (the task-instance list).

    Task instances used to be an Operations sub-tab. Runs is a main tab now and
    holds the instance list directly, so there is no sub-tab to click.
    """
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-runs']").click()
    page.wait_for_load_state("networkidle")


def _editor_bar(page: Page):
    """The sticky header bar of the task-config editor (holds Save/Create/Delete)."""
    return page.locator("#task-config-list .le-editor .panel-box__bar")


def _editor_footer(page: Page):
    """The sticky footer of the task-config editor (holds Start)."""
    return page.locator("#task-config-list .le-editor .panel-box__footer")


def _type_select(page: Page):
    return page.get_by_test_id("task-type-select")


def _name_input(page: Page):
    return page.get_by_test_id("task-name-input")


def _select_type(page: Page, typename: str = TASK_TYPE):
    """Pick a task type and wait for its schema form to actually render.

    Every caller used to follow `select_option` with `wait_for_timeout(800..1000)` —
    a bet that one fetch plus a render always fits in under a second. It does on an
    idle box; the suite runs 16-way parallel on a host that also runs the user's live
    services, and there the bet is simply lost sometimes. Waiting on the rendered form
    is both faster on the fast path and load-proof on the slow one.
    """
    _type_select(page).select_option(typename)
    expect(page.locator("#task-config-list .le-editor .schema-form")).to_be_visible()


def _open_config(page: Page, name: str = FAKE_CONFIG["name"]):
    """Click a config row in the list to load it into the editor."""
    page.wait_for_selector("#task-config-list .le-list-item", timeout=5000)
    page.locator("#task-config-list .le-list-item", has_text=name).first.click()
    expect(page.locator("#task-config-list .le-editor")).to_be_visible()


def _mock_task_types(page: Page, types: list | None = None, system_types: list | None = None):
    page.route(
        "**/api/tasks/types/list",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "types": types or [TASK_TYPE],
                    "systemTypes": system_types or [],
                }
            ),
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
    def test_pane_structure(self, page: Page, base_url: str):
        """The Tasks pane renders the list-editor screen: a list panel and an editor."""
        _setup_full_task_mocks(page)
        _navigate_to_tasks(page, base_url)
        expect(page.locator("#task-config-list")).to_be_visible()
        expect(page.locator("#task-config-list .le-list-panel")).to_be_visible()
        expect(page.locator("#task-config-list .le-editor")).to_be_visible()
        # A blank "New Task Config" is loaded by default, so the editor is usable
        # without first picking anything.
        expect(_editor_bar(page)).to_contain_text("New Task Config")

    def test_no_js_errors_on_task_tab(self, page: Page, base_url: str):
        """No console errors while loading the Tasks pane."""
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        _setup_full_task_mocks(page)
        _navigate_to_tasks(page, base_url)
        page.wait_for_timeout(1200)
        uncaught = [e for e in errors if "TypeError" in e or "Uncaught" in e]
        assert uncaught == [], f"Uncaught JS errors: {uncaught}"

    def test_no_js_errors_on_activity_tab(self, page: Page, base_url: str):
        """No console errors while loading the Activity pane."""
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        _setup_full_task_mocks(page, tasks={RUNNING_TASK["instance"]["uuid"]: RUNNING_TASK})
        _navigate_to_activity(page, base_url)
        page.wait_for_timeout(1200)
        uncaught = [e for e in errors if "TypeError" in e or "Uncaught" in e]
        assert uncaught == [], f"Uncaught JS errors: {uncaught}"


# ---------------------------------------------------------------------------
# Task config list loading
# ---------------------------------------------------------------------------


class TestTaskConfigList:
    def test_existing_configs_rendered(self, page: Page, base_url: str):
        """Configs returned from /api/tasks/configs/get appear as list rows."""
        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list .le-list-item", timeout=5000)
        content = page.locator("#task-config-list .le-list-panel").inner_text()
        assert FAKE_CONFIG["name"] in content, f"Config name not found in: {content!r}"

    def test_list_row_shows_task_type(self, page: Page, base_url: str):
        """A row's meta line names the task type, so the list is scannable."""
        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list .le-list-item", timeout=5000)
        expect(page.locator("#task-config-list .le-list-item").first).to_contain_text(TASK_TYPE)

    def test_selecting_config_loads_it_into_editor(self, page: Page, base_url: str):
        """Clicking a row loads that config's name and type into the editor."""
        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG})
        _navigate_to_tasks(page, base_url)
        _open_config(page)
        expect(_editor_bar(page)).to_contain_text(FAKE_CONFIG["name"])
        expect(_name_input(page)).to_have_value(FAKE_CONFIG["name"])

    def test_new_config_has_no_start_button(self, page: Page, base_url: str):
        """There is nothing to start until a config exists server-side."""
        _setup_full_task_mocks(page, configs={})
        _navigate_to_tasks(page, base_url)
        page.wait_for_timeout(800)
        expect(_editor_footer(page).locator("button[title='Start']")).to_have_count(0)

    def test_filter_box_narrows_the_list(self, page: Page, base_url: str):
        """The list's filter box hides non-matching rows."""
        other = {**FAKE_CONFIG, "uuid": "eeeeeeee-0000-0000-0000-000000000002", "name": "Weekly Vacuum"}
        _setup_full_task_mocks(
            page,
            configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG, other["uuid"]: other},
        )
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list .le-list-item", timeout=5000)
        expect(page.locator("#task-config-list .le-list-item:visible")).to_have_count(2)

        page.locator("#task-config-list .le-list-search input").fill("Weekly")
        expect(page.locator("#task-config-list .le-list-item:visible")).to_have_count(1)
        expect(page.locator("#task-config-list .le-list-item:visible")).to_contain_text("Weekly Vacuum")


# ---------------------------------------------------------------------------
# Task type dropdown
# ---------------------------------------------------------------------------


class TestTaskTypeDropdown:
    def test_type_dropdown_populated_from_api(self, page: Page, base_url: str):
        """The task type <select> lists types returned by /api/tasks/types/list."""
        types = ["DatabaseMaintenanceTask", "CollectTrainingImagesTask"]
        _mock_task_types(page, types)
        _mock_task_schema(page)
        _mock_task_configs(page, {})
        _mock_active_tasks(page, {})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)

        options = _type_select(page).locator("option").all_text_contents()
        for t in types:
            assert t in options, f"Type '{t}' not in dropdown options: {options}"

    def test_system_types_are_hidden_until_show_system_is_ticked(self, page: Page, base_url: str):
        """Types a dedicated screen owns are not offered for a hand-built config.

        The conditional is the point: `StreamWatcherTask` must be absent while the
        toggle is off and present while it is on. Asserting only that the ordinary
        type is listed would pass just as happily with no filtering at all.
        """
        types = ["DatabaseMaintenanceTask", "StreamWatcherTask"]
        _mock_task_types(page, types, system_types=["StreamWatcherTask"])
        _mock_task_schema(page)
        _mock_task_configs(page, {})
        _mock_active_tasks(page, {})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)

        options = _type_select(page).locator("option").all_text_contents()
        assert "DatabaseMaintenanceTask" in options, options
        assert "StreamWatcherTask" not in options, options

        page.locator("#task-config-list .tc-system-toggle input").check()

        wait_until(
            page,
            lambda: "StreamWatcherTask" in _type_select(page).locator("option").all_text_contents(),
            "Expected 'Show system' to add StreamWatcherTask to the type dropdown",
        )

    def test_editing_a_system_config_keeps_its_own_type_offered(self, page: Page, base_url: str):
        """Opening a saved config of a hidden type must not blank its dropdown.

        Filtering the option away would leave the <select> with no matching value,
        and saving from there would rewrite the config as whatever happened to be
        first in the list.
        """
        watcher_config = {
            **FAKE_CONFIG,
            "uuid": "cfg-watcher",
            "name": "Armed watcher",
            "typename": "StreamWatcherTask",
        }
        _mock_task_types(page, ["DatabaseMaintenanceTask", "StreamWatcherTask"], system_types=["StreamWatcherTask"])
        page.route(
            "**/api/tasks/types/schema",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"schemas": {"StreamWatcherTask": {}, TASK_TYPE: REFRESH_DB_SCHEMA}}),
            ),
        )
        _mock_task_configs(page, {watcher_config["uuid"]: watcher_config})
        _mock_active_tasks(page, {})

        _navigate_to_tasks(page, base_url)
        _open_config(page, "Armed watcher")

        expect(_type_select(page)).to_have_value("StreamWatcherTask")

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
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)
        _type_select(page).select_option(TASK_TYPE)
        wait_until(page, lambda: schema_calls, "Expected /api/tasks/types/schema to be called after type selection")
        assert TASK_TYPE in schema_calls[0]

    def test_schema_fields_appear_after_type_selection(self, page: Page, base_url: str):
        """After selecting a task type, schema-driven fields appear in the editor."""
        _setup_full_task_mocks(page, configs={}, tasks={})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)

        _type_select(page).select_option(TASK_TYPE)

        # `expect` auto-retries, so it is both the barrier and the assertion; the fixed
        # 1s sleep it replaces was a bet that the schema fetch plus render always fits
        # in a second, which under a 16-way parallel run it need not.
        expect(page.locator("#task-config-list .le-editor")).to_contain_text(
            re.compile("Remove deleted images|remove_deleted")
        )

    def test_boolean_field_is_not_double_labelled(self, page: Page, base_url: str):
        """A boolean renders its label once, beside the checkbox — not also above it.

        SchemaField used to add its own bold label over every field, including
        booleans whose control already draws `description || fieldName` next to
        the box. That turned each one-line toggle into a three-line block.
        """
        _setup_full_task_mocks(page, configs={}, tasks={})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)
        _select_type(page)

        editor = page.locator("#task-config-list .le-editor")
        # The schema-level label element must not be emitted for the boolean.
        expect(editor.locator(".schema-field--self-labelled")).to_have_count(1)
        expect(editor.locator(".schema-field--self-labelled .schema-field__label")).to_have_count(0)

    def test_boolean_checkbox_is_a_visible_box(self, page: Page, base_url: str):
        """A schema-form checkbox must render as a square box, not a thin sliver.

        Pico draws checkboxes with `appearance: none`, so their only size is the
        width/height it sets. `.schema-form input[type=checkbox] { width: auto }`
        — added to stop flex layouts stretching them — collapsed them to zero
        width instead, leaving a 1px-wide vertical line nobody could recognise.
        """
        _setup_full_task_mocks(page, configs={}, tasks={})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)
        _type_select(page).select_option(TASK_TYPE)

        box = page.locator("#task-config-list .le-editor .schema-form input[type='checkbox']").first
        # Wait on the checkbox itself rather than a sleep: `bounding_box()` does not
        # retry, so a slow schema render made this read None and fail on the next line.
        expect(box).to_be_visible()
        rect = box.bounding_box()
        assert rect is not None, "boolean checkbox is not rendered"
        assert rect["width"] >= 10, f"checkbox collapsed to {rect['width']}px wide"
        assert rect["height"] >= 10, f"checkbox collapsed to {rect['height']}px tall"
        # Square-ish: a stretched box reads as a bar, not a checkbox.
        assert abs(rect["width"] - rect["height"]) <= 3, f"checkbox is not square: {rect}"


# ---------------------------------------------------------------------------
# Full create task config flow
# ---------------------------------------------------------------------------


class TestCreateTaskConfig:
    def test_create_calls_api_with_correct_payload(self, page: Page, base_url: str):
        """Full create flow: select type → fill name → Create → /api/tasks/configs/create called."""
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
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)

        _select_type(page)
        _name_input(page).fill("My Refresh Job")

        _editor_bar(page).locator("button", has_text="Create").click()
        wait_until(page, lambda: create_calls, "Expected POST to /api/tasks/configs/create")
        payload = create_calls[0]
        assert payload.get("name") == "My Refresh Job"
        assert payload.get("typename") == TASK_TYPE

    def test_create_requires_name(self, page: Page, base_url: str):
        """Create stays unavailable / does not call the API without a name."""
        calls: list[bool] = []
        page.route(
            "**/api/tasks/configs/create",
            lambda route: (
                calls.append(True),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "configUuid": FAKE_CONFIG["uuid"]}),
                ),
            ),
        )
        _setup_full_task_mocks(page, configs={}, tasks={})

        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)
        _select_type(page)

        create_btn = _editor_bar(page).locator("button", has_text="Create")
        # Selecting a type marks the form dirty, so the button is enabled; the
        # name check is what has to stop the request.
        if create_btn.is_enabled():
            create_btn.click()
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
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)
        _select_type(page)

        _name_input(page).fill("Refresh Job")
        _editor_bar(page).locator("button", has_text="Create").click()
        wait_until(page, lambda: create_calls, "Expected POST to /api/tasks/configs/create")
        payload = create_calls[0]
        assert "params" in payload, f"Payload missing 'params': {payload}"
        # Default boolean field 'remove_deleted' should be present
        assert "remove_deleted" in payload["params"], f"'remove_deleted' not in params: {payload['params']}"


# ---------------------------------------------------------------------------
# Persistent (auto-restart-after-restart) flag on task configs
# ---------------------------------------------------------------------------


class TestPersistentFlag:
    def test_persistent_checkbox_present_and_unchecked_by_default(self, page: Page, base_url: str):
        """The config form offers a persistent checkbox, defaulting unchecked."""
        _setup_full_task_mocks(page, configs={}, tasks={})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)
        _select_type(page)

        chk = page.locator("#task-config-list").get_by_label("Restart automatically after a server restart")
        expect(chk).to_be_attached()
        expect(chk).not_to_be_checked()

    def test_create_defaults_persistent_false_in_payload(self, page: Page, base_url: str):
        """Leaving the checkbox untouched sends persistent: false to the API."""
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
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)
        _select_type(page)
        _name_input(page).fill("Persist Test")
        _editor_bar(page).locator("button", has_text="Create").click()
        wait_until(page, lambda: create_calls, "Expected POST to /api/tasks/configs/create")
        payload = create_calls[0]
        assert payload.get("persistent") is False, f"Expected persistent=False by default, got: {payload}"
        assert payload["params"].get("_persistent") is False

    def test_checking_persistent_reaches_create_payload(self, page: Page, base_url: str):
        """Checking the box sends persistent: true, both top-level and inside params."""
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
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)
        _select_type(page)
        _name_input(page).fill("Persist Test")
        page.locator("#task-config-list").get_by_label("Restart automatically after a server restart").check()
        _editor_bar(page).locator("button", has_text="Create").click()
        wait_until(page, lambda: create_calls, "Expected POST to /api/tasks/configs/create")
        payload = create_calls[0]
        assert payload.get("persistent") is True, f"Expected persistent=True after checking box, got: {payload}"
        assert payload["params"].get("_persistent") is True

    def test_persistent_config_shows_chip_in_list(self, page: Page, base_url: str):
        """A persistent config is marked in the list, without opening it."""
        persistent = {
            **FAKE_CONFIG,
            "params": {**FAKE_CONFIG["params"], "_persistent": True},
        }
        _setup_full_task_mocks(page, configs={persistent["uuid"]: persistent}, tasks={})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("#task-config-list .le-list-item", timeout=5000)
        expect(page.locator("#task-config-list .le-list-item .le-chip")).to_have_text("persistent")


# ---------------------------------------------------------------------------
# Active tasks list (Activity sub-tab)
# ---------------------------------------------------------------------------


class TestActiveTasksList:
    def test_active_tasks_rendered(self, page: Page, base_url: str):
        """Running tasks appear in the #active-tasks-list section."""
        task_id = RUNNING_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: RUNNING_TASK})

        _navigate_to_activity(page, base_url)

        expect(page.locator("#active-tasks-list")).to_contain_text(
            re.compile(f"{re.escape(RUNNING_TASK['config']['name'])}|running", re.IGNORECASE)
        )

    def test_activity_badge_counts_live_tasks(self, page: Page, base_url: str):
        """The Activity tab badge reports live tasks so leaving the pane doesn't
        mean losing sight of what is running."""
        _setup_full_task_mocks(
            page,
            configs={},
            tasks={
                RUNNING_TASK["instance"]["uuid"]: RUNNING_TASK,
                COMPLETED_TASK["instance"]["uuid"]: COMPLETED_TASK,
            },
        )
        _navigate_to_activity(page, base_url)
        badge = page.locator("#task-activity-badge")
        # Only the running task counts; the completed one does not.
        wait_until(
            page,
            lambda: badge.is_visible() and badge.inner_text() == "1",
            "Expected the Activity badge to show 1 live task",
        )


# ---------------------------------------------------------------------------
# Active task progress bar — indeterminate vs. real 0%
# ---------------------------------------------------------------------------


class TestActiveTaskProgressBar:
    def test_null_progress_renders_indeterminate_striped_bar(self, page: Page, base_url: str):
        """A running task with progress: null (no denominator, e.g. a live capture
        task) renders the striped/animated indeterminate bar, not a stuck 0%."""
        task_id = INDETERMINATE_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: INDETERMINATE_TASK})

        _navigate_to_activity(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").click()

        bar = page.locator(f"[data-task-uuid='{task_id}'] progress")
        expect(bar).to_be_visible()
        expect(bar).to_have_class(re.compile("progress-indeterminate"))
        assert bar.get_attribute("value") is None, "Indeterminate bar must not have a numeric value attribute"
        assert bar.get_attribute("role") == "progressbar"

    def test_zero_progress_renders_normal_bar(self, page: Page, base_url: str):
        """A running task genuinely at progress: 0 (start of a batch task) still
        renders a normal (non-indeterminate) 0% bar — this is the regression that
        matters: don't collapse 'no denominator' and 'real 0%' into the same thing."""
        task_id = ZERO_PROGRESS_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: ZERO_PROGRESS_TASK})

        _navigate_to_activity(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").click()

        bar = page.locator(f"[data-task-uuid='{task_id}'] progress")
        expect(bar).to_be_visible()
        expect(bar).not_to_have_class(re.compile("progress-indeterminate"))
        assert bar.get_attribute("value") == "0"

    def test_running_progress_renders_normal_bar(self, page: Page, base_url: str):
        """Sanity check: a running task with real, positive progress keeps
        rendering the normal numeric bar (pre-existing behavior)."""
        task_id = RUNNING_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: RUNNING_TASK})

        _navigate_to_activity(page, base_url)
        page.wait_for_selector(f"[data-task-uuid='{task_id}'] summary", timeout=5000)
        page.locator(f"[data-task-uuid='{task_id}'] summary").click()

        bar = page.locator(f"[data-task-uuid='{task_id}'] progress")
        expect(bar).not_to_have_class(re.compile("progress-indeterminate"))
        assert bar.get_attribute("value") == "42"


# ---------------------------------------------------------------------------
# Task lifecycle button flows
# ---------------------------------------------------------------------------


class TestTaskLifecycleButtons:
    def test_start_task_calls_api(self, page: Page, base_url: str):
        """Clicking Start on a selected task config POSTs to /api/tasks/instances/start."""
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
        _open_config(page)
        _editor_footer(page).locator("button[title='Start']").click()
        wait_until(page, lambda: start_calls, "Expected POST to /api/tasks/instances/start")
        assert start_calls[0].get("configUuid") == FAKE_CONFIG["uuid"]

    def test_start_keeps_the_config_open_in_the_editor(self, page: Page, base_url: str):
        """Starting must not throw you back to a blank New Task Config form.

        A full pane refresh rebuilds renderTaskConfigs from scratch, which clears
        the list selection and reopens the empty form — losing the config you were
        looking at, on the one screen where Start is reachable.
        """
        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG}, tasks={})
        page.route(
            "**/api/tasks/instances/start",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "taskUuid": "new-task-uuid"}),
            ),
        )

        _navigate_to_tasks(page, base_url)
        _open_config(page)
        _editor_footer(page).locator("button[title='Start']").click()
        page.wait_for_timeout(900)

        expect(_name_input(page)).to_have_value(FAKE_CONFIG["name"])
        expect(_editor_footer(page).locator("button[title='Start']")).to_have_count(1)
        expect(page.locator("#task-config-list .le-list-item.active")).to_have_count(1)

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

        _navigate_to_activity(page, base_url)
        page.wait_for_selector("#active-tasks-list button[title='Stop']", timeout=5000)
        page.locator("#active-tasks-list button[title='Stop']").first.click()
        wait_until(page, lambda: cancel_calls, "Expected POST to /api/tasks/instances/cancel")
        assert task_id in cancel_calls[0].get("taskUuids", [])

    def test_delete_completed_task_shows_confirm(self, page: Page, base_url: str):
        """Clicking Delete on a completed task opens a confirm() dialog."""
        task_id = COMPLETED_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: COMPLETED_TASK})

        _navigate_to_activity(page, base_url)
        page.wait_for_selector("#active-tasks-list button[title='Delete']", timeout=5000)

        dialogs: list[str] = []
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
        page.locator("#active-tasks-list button[title='Delete']").first.click()
        wait_until(page, lambda: dialogs, "Expected a confirm() dialog")

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

        _navigate_to_activity(page, base_url)
        page.wait_for_selector("#active-tasks-list button[title='Delete']", timeout=5000)

        page.on("dialog", lambda d: d.accept())
        page.locator("#active-tasks-list button[title='Delete']").first.click()
        wait_until(page, lambda: delete_calls, "Expected POST to /api/tasks/instances/delete")
        assert task_id in delete_calls[0].get("taskUuids", [])


# ---------------------------------------------------------------------------
# Delete task config flow
# ---------------------------------------------------------------------------


class TestDeleteTaskConfig:
    def test_delete_config_shows_confirm(self, page: Page, base_url: str):
        """Clicking Delete in the editor bar opens a modal naming the config."""
        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG}, tasks={})

        _navigate_to_tasks(page, base_url)
        _open_config(page)
        _editor_bar(page).locator("button", has_text="Delete").click()

        modal = page.get_by_test_id("modal-shell-dialog")
        expect(modal).to_be_visible()
        expect(modal).to_contain_text(FAKE_CONFIG["name"])

    def test_delete_config_accepted_calls_api(self, page: Page, base_url: str):
        """Confirming delete POSTs to /api/tasks/configs/delete."""
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
        _open_config(page)
        _editor_bar(page).locator("button", has_text="Delete").click()
        page.get_by_test_id("modal-shell-dialog").locator("button", has_text="Delete").click()

        wait_until(page, lambda: delete_calls, "Expected POST to /api/tasks/configs/delete")
        assert FAKE_CONFIG["uuid"] in delete_calls[0].get("configUuids", [])

    def test_delete_config_cancelled_does_not_call_api(self, page: Page, base_url: str):
        """Dismissing the confirm modal leaves the config alone."""
        delete_calls: list[dict] = []
        page.route(
            "**/api/tasks/configs/delete",
            lambda route: (
                delete_calls.append(json.loads(route.request.post_data)),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success"}),
                ),
            ),
        )

        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG}, tasks={})

        _navigate_to_tasks(page, base_url)
        _open_config(page)
        _editor_bar(page).locator("button", has_text="Delete").click()
        page.get_by_test_id("modal-shell-dialog").locator("button", has_text="Cancel").click()
        page.wait_for_timeout(500)

        assert not delete_calls, "API must NOT be called when the confirm modal is cancelled"


class TestUnsavedChangesBeforeStart:
    """Starting runs the config as stored on the server, so unsaved form edits
    would silently not be part of the run. Editing must be visible, and Start
    must ask."""

    @staticmethod
    def _make_dirty(page: Page):
        _name_input(page).fill("Nightly Refresh (edited)")
        page.wait_for_timeout(200)

    def test_editing_shows_unsaved_badge_and_lights_save(self, page: Page, base_url: str):
        """A worded badge appears in the bar and Save gets the attention style."""
        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG}, tasks={})
        _navigate_to_tasks(page, base_url)
        _open_config(page)

        badge = page.get_by_test_id("panel-dirty-label")
        expect(badge).to_be_hidden()
        self._make_dirty(page)
        expect(badge).to_be_visible()
        save_btn = _editor_bar(page).locator("button", has_text="Save")
        expect(save_btn).to_have_class(re.compile("panel-btn--attention"))

    def test_start_while_dirty_prompts_and_cancel_aborts(self, page: Page, base_url: str):
        """Cancelling the unsaved-changes prompt starts nothing."""
        start_calls: list[dict] = []
        page.route(
            "**/api/tasks/instances/start",
            lambda route: (
                start_calls.append(json.loads(route.request.post_data)),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "taskUuid": "new-task-uuid"}),
                ),
            ),
        )
        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG}, tasks={})
        _navigate_to_tasks(page, base_url)
        _open_config(page)
        self._make_dirty(page)

        _editor_footer(page).locator("button[title='Start']").click()
        modal = page.get_by_test_id("modal-shell-dialog")
        expect(modal).to_be_visible()
        expect(modal).to_contain_text("unsaved changes")
        modal.locator("button", has_text="Cancel").click()
        page.wait_for_timeout(500)

        assert not start_calls, "Start must not run while the prompt is cancelled"

    def test_save_and_start_updates_then_starts(self, page: Page, base_url: str):
        """'Save & start' persists the edits first, then launches the task."""
        calls: list[str] = []

        def _record(name: str, body: dict):
            def handler(route: Route):
                calls.append(name)
                route.fulfill(status=200, content_type="application/json", body=json.dumps(body))

            return handler

        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG}, tasks={})
        page.route("**/api/tasks/configs/update", _record("update", {"status": "success"}))
        page.route(
            "**/api/tasks/instances/start",
            _record("start", {"status": "success", "taskUuid": "new-task-uuid"}),
        )

        _navigate_to_tasks(page, base_url)
        _open_config(page)
        self._make_dirty(page)

        _editor_footer(page).locator("button[title='Start']").click()
        page.get_by_test_id("modal-shell-dialog").locator("button", has_text="Save & start").click()

        wait_until(page, lambda: "start" in calls, "Expected the task to start after saving")
        assert calls.index("update") < calls.index("start"), f"Save must precede start, got {calls}"

    def test_start_when_clean_does_not_prompt(self, page: Page, base_url: str):
        """An untouched config starts immediately, as before."""
        start_calls: list[dict] = []
        page.route(
            "**/api/tasks/instances/start",
            lambda route: (
                start_calls.append(json.loads(route.request.post_data)),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "taskUuid": "new-task-uuid"}),
                ),
            ),
        )
        _setup_full_task_mocks(page, configs={FAKE_CONFIG["uuid"]: FAKE_CONFIG}, tasks={})
        _navigate_to_tasks(page, base_url)
        _open_config(page)

        _editor_footer(page).locator("button[title='Start']").click()
        wait_until(page, lambda: start_calls, "Expected an immediate start with no prompt")


class TestSchemaFieldHelpText:
    """`help` is discoverable prose under a field, unlike `description`, which is
    only a tooltip on the label. Some settings (the duplicate thresholds) need a
    whole paragraph of guidance that a user cannot be expected to find by
    hovering — but that paragraph is a wall to scroll past on every later visit,
    so it is a disclosure: closed by default, one click away, never a tooltip."""

    _HELP = "Run once in calibration mode.\n\nNothing is discarded while calibrating."
    _SCHEMA = {
        "meta": {"order": ["mode", "plain", "eager"]},
        "mode": {
            "type": "select",
            "label": "Mode",
            "options": [{"value": "a", "label": "A"}, {"value": "b", "label": "B"}],
            "default": "a",
            "help": _HELP,
        },
        "plain": {
            "type": "select",
            "label": "Plain",
            "description": "Tooltip only.",
            "options": [{"value": "x", "label": "X"}],
            "default": "x",
        },
        "eager": {
            "type": "select",
            "label": "Eager",
            "options": [{"value": "y", "label": "Y"}],
            "default": "y",
            "help": "Read me first.",
            "help_open": True,
        },
    }

    def _open_editor(self, page: Page, base_url: str):
        _mock_task_types(page)
        _mock_task_schema(page, self._SCHEMA)
        _mock_task_configs(page, {})
        _mock_active_tasks(page, {})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)
        _select_type(page)
        return page.locator("#task-config-list .le-editor")

    def test_help_starts_collapsed_but_is_advertised(self, page: Page, base_url: str):
        """The toggle is always visible (so the prose is discoverable without
        hovering), while the prose itself is folded away until asked for."""
        editor = self._open_editor(page, base_url)
        help_el = editor.locator(".schema-field__help").first
        expect(help_el.locator("summary")).to_be_visible()
        expect(help_el).not_to_have_attribute("open", re.compile(".*"))
        expect(help_el.locator(".schema-field__help-body")).to_be_hidden()

    def test_clicking_the_toggle_reveals_the_prose(self, page: Page, base_url: str):
        editor = self._open_editor(page, base_url)
        help_el = editor.locator(".schema-field__help").first
        help_el.locator("summary").click()
        expect(help_el).to_contain_text("Run once in calibration mode.")
        expect(help_el.locator(".schema-field__help-body")).to_be_visible()

    def test_help_open_schema_flag_starts_expanded(self, page: Page, base_url: str):
        """Prose that has to be read before the field can be answered at all can
        opt out of the default."""
        editor = self._open_editor(page, base_url)
        eager = editor.locator(".schema-field__help", has_text="Read me first.")
        expect(eager.locator(".schema-field__help-body")).to_be_visible()

    def test_blank_lines_become_separate_paragraphs(self, page: Page, base_url: str):
        editor = self._open_editor(page, base_url)
        help_el = editor.locator(".schema-field__help").first
        help_el.locator("summary").click()
        expect(help_el.locator("p")).to_have_count(2)

    def test_fields_without_help_render_none(self, page: Page, base_url: str):
        """A `description` must not start rendering as a visible block."""
        editor = self._open_editor(page, base_url)
        expect(editor.locator(".schema-field__help")).to_have_count(2)
        expect(editor).not_to_contain_text("Tooltip only.")


# ---------------------------------------------------------------------------
# Error message on a failed task's card
# ---------------------------------------------------------------------------

ERRORED_TASK = {
    "instance": {
        "uuid": "ffffffff-0000-0000-0000-000000000005",
        "status": "error",
        "startedAt": "2024-01-01T00:00:00",
        "endedAt": "2024-01-01T05:34:00",
        "errorMessage": "URLError: <urlopen error _ssl.c:999: The handshake operation timed out>",
    },
    # typename matters: without it the card falls back to DEFAULT_STATUS_SCHEMA, and
    # the "survives a custom status schema" test below would prove nothing.
    "config": {"name": "Nightly Refresh", "uuid": FAKE_CONFIG["uuid"], "typename": TASK_TYPE},
    "progress": 100.0,
    "isPausedInPlace": False,
    "logLines": ["Error: <urlopen error _ssl.c:999: The handshake operation timed out>"],
}


class TestErroredTaskShowsWhyItFailed:
    """A failed task used to say only "● error". The reason lived in the one-shot
    result (deleted on first read) and in the log tail, which a task type with its
    own status_schema() doesn't necessarily render."""

    def _open_errored_card(self, page: Page, base_url: str, status_schema=None):
        task_id = ERRORED_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: ERRORED_TASK})
        page.route(
            "**/api/tasks/types/status-schema",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"schemas": {TASK_TYPE: status_schema or []}}),
            ),
        )
        _navigate_to_activity(page, base_url)
        page.wait_for_selector("#active-tasks-list summary", timeout=5000)
        page.locator("#active-tasks-list summary").first.click()

    def test_error_message_is_shown_on_the_card(self, page: Page, base_url: str):
        self._open_errored_card(page, base_url)
        expect(page.get_by_test_id("task-error-message")).to_contain_text("handshake operation timed out")

    def test_error_message_survives_a_custom_status_schema(self, page: Page, base_url: str):
        """The row is rendered by the card itself, not as a status widget — a task
        type whose schema omits the default statusMessage widget (every real one
        that defines its own) must still explain the failure."""
        self._open_errored_card(
            page,
            base_url,
            status_schema=[{"id": "stats", "type": "stat_grid", "source": "summary", "fields": {"a": "A"}}],
        )
        expect(page.get_by_test_id("task-error-message")).to_contain_text("handshake operation timed out")

    def test_completed_task_has_no_error_row(self, page: Page, base_url: str):
        task_id = COMPLETED_TASK["instance"]["uuid"]
        _setup_full_task_mocks(page, configs={}, tasks={task_id: COMPLETED_TASK})
        _navigate_to_activity(page, base_url)
        page.wait_for_selector("#active-tasks-list summary", timeout=5000)
        page.locator("#active-tasks-list summary").first.click()
        expect(page.get_by_test_id("task-error-message")).to_have_count(0)


# ---------------------------------------------------------------------------
# Completed tasks keep their status widgets
# ---------------------------------------------------------------------------


class TestCompletedTaskKeepsItsStatusWidgets:
    """Status widgets read from /status-data, which used to be live-only — so a
    finished task's histograms and stat grids rendered empty and everything the run
    measured was lost. The server now serves the snapshot persisted when it ended;
    the card doesn't care where the numbers came from, but it must stop polling for
    them, since a finished task's data can never change."""

    STATUS_SCHEMA = [
        {"id": "stats", "type": "stat_grid", "label": "Totals", "fields": {"images_saved": "Saved"}},
        {"id": "dedup_image_hist", "type": "histogram", "label": "Distance"},
    ]
    SNAPSHOT = {
        "stats": {"images_saved": "306"},
        "dedup_image_hist": {"bins": [{"label": "0", "count": 42, "marked": True}], "total": 42},
    }

    def _setup(self, page: Page, base_url: str, status_data_calls: list):
        task_id = COMPLETED_TASK["instance"]["uuid"]
        # typename is what makes the card resolve THIS schema instead of the default.
        task = {**COMPLETED_TASK, "config": {**COMPLETED_TASK["config"], "typename": TASK_TYPE}}
        _setup_full_task_mocks(page, configs={}, tasks={task_id: task})
        page.route(
            "**/api/tasks/types/status-schema",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"schemas": {TASK_TYPE: self.STATUS_SCHEMA}}),
            ),
        )

        def handle_status_data(route: Route):
            status_data_calls.append(1)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "statusData": {task_id: self.SNAPSHOT}}),
            )

        page.route("**/api/tasks/instances/status-data", handle_status_data)
        _navigate_to_activity(page, base_url)
        page.wait_for_selector("#active-tasks-list summary", timeout=5000)
        page.locator("#active-tasks-list summary").first.click()

    def test_persisted_stats_and_histogram_render(self, page: Page, base_url: str):
        self._setup(page, base_url, [])
        card = page.locator("#active-tasks-list .task-instance-card")
        expect(card).to_contain_text("306")
        expect(card).to_contain_text("Saved")
        # The histogram widget rendered its bin rather than its empty state.
        expect(card).not_to_contain_text("No measurements yet.")

    def test_finished_card_fetches_once_and_stops_polling(self, page: Page, base_url: str):
        calls: list = []
        self._setup(page, base_url, calls)
        expect(page.locator("#active-tasks-list .task-instance-card")).to_contain_text("306")
        after_open = len(calls)
        # Well past the 5s poll cadence — a running card would have polled again.
        page.wait_for_timeout(6000)
        assert len(calls) == after_open, f"expected no further polls, got {len(calls) - after_open}"


# ---------------------------------------------------------------------------
# `tags` schema field — the compact multi-select, not the old checkbox column
# ---------------------------------------------------------------------------


class TestTagsSchemaField:
    """A `tags` field (e.g. CollectTrainingImages' "Apply Tags") renders as the
    shared compact multi-select dropdown.

    It used to render as CheckboxField: one full-width `<label><input></label><br>`
    per tag, so a form with a real tag vocabulary became a wall of rows. The
    field is now MultiSelectField, wrapping the same `.ms-dropdown` widget the
    review filter toolbar uses.
    """

    _SCHEMA = {
        "meta": {"order": ["apply_tags"]},
        "apply_tags": {
            "type": "tags",
            "label": "Apply Tags",
            "description": "Tags to apply to every bounding box in each saved image.",
            "default": [],
        },
    }

    _TAGS = [
        {"uuid": "tag-aaa", "name": "Daytime", "color": "#00ff00", "icon": None},
        {"uuid": "tag-bbb", "name": "Nighttime", "color": "#0000ff", "icon": None},
    ]

    def _open_editor(self, page: Page, base_url: str, tags: list | None = None):
        page.route(
            "**/api/tags/list",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"tags": self._TAGS if tags is None else tags}),
            ),
        )
        _mock_task_types(page)
        _mock_task_schema(page, self._SCHEMA)
        _mock_task_configs(page, {})
        _mock_active_tasks(page, {})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)
        _select_type(page)
        return page.locator("#task-config-list .le-editor")

    def test_renders_a_dropdown_trigger_not_a_checkbox_column(self, page: Page, base_url: str):
        editor = self._open_editor(page, base_url)
        expect(editor.locator(".ms-dropdown__trigger")).to_have_count(1)
        # The panel is closed, so no tag row is on screen until it is opened.
        expect(editor.locator(".ms-item")).to_have_count(2)
        expect(editor.locator(".ms-item").first).to_be_hidden()

    def test_trigger_summarises_the_empty_selection(self, page: Page, base_url: str):
        editor = self._open_editor(page, base_url)
        # 'None selected', not the filter widget's 'All' — an empty tag list
        # applies no tags, it does not apply every tag.
        expect(editor.locator(".ms-dropdown__trigger")).to_contain_text("None selected")

    def test_field_label_is_not_repeated_inside_the_trigger(self, page: Page, base_url: str):
        editor = self._open_editor(page, base_url)
        expect(editor.locator(".schema-field__label", has_text="Apply Tags")).to_have_count(1)
        expect(editor.locator(".ms-dropdown__trigger")).not_to_contain_text("Apply Tags")

    def test_no_exclude_toggles_on_a_picker(self, page: Page, base_url: str):
        """The `≠` per-row exclude button builds queries; it is meaningless when
        the list is choosing which tags to apply."""
        editor = self._open_editor(page, base_url)
        editor.locator(".ms-dropdown__trigger").click()
        expect(editor.locator(".ms-item__not-btn")).to_have_count(0)
        expect(editor.locator(".ms-mode-btn")).to_have_count(0)

    def test_selecting_a_tag_reaches_the_create_payload(self, page: Page, base_url: str):
        create_calls: list[dict] = []

        def handle_create(route: Route):
            create_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configUuid": FAKE_CONFIG["uuid"]}),
            )

        editor = self._open_editor(page, base_url)
        page.route("**/api/tasks/configs/create", handle_create)

        editor.locator(".ms-dropdown__trigger").click()
        editor.locator(".ms-item input[value='tag-bbb']").check()
        expect(editor.locator(".ms-dropdown__trigger")).to_contain_text("Nighttime")

        _name_input(page).fill("Collector")
        _editor_bar(page).locator("button", has_text="Create").click()
        wait_until(page, lambda: create_calls, "Expected POST to /api/tasks/configs/create")
        assert create_calls[0]["params"]["apply_tags"] == ["tag-bbb"], create_calls[0]["params"]

    def test_empty_vocabulary_shows_a_placeholder(self, page: Page, base_url: str):
        editor = self._open_editor(page, base_url, tags=[])
        editor.locator(".ms-dropdown__trigger").click()
        expect(editor.locator(".ms-dropdown__empty")).to_be_visible()
        expect(editor.locator(".ms-dropdown__empty")).to_contain_text("No tags defined yet")


# Python spells these types `bool` and `str`; the JS factory table was keyed only on
# `boolean` and `string`. StreamWatcherTask declares `dry_run` as `bool`, so its config
# form rendered a *text box* placeheld "Unhandled type: bool" — which then round-trips a
# truthy string into a flag meant to suppress every MQTT publish and clip write.
PY_SPELLED_SCHEMA = {
    "meta": {"order": ["dry_run", "note"]},
    "dry_run": {
        "type": "bool",
        "label": "Dry run",
        "description": "Evaluate rules but perform no actions.",
        "default": False,
    },
    "note": {"type": "str", "label": "Note", "description": "Free text."},
}


class TestPythonSpelledSchemaTypes:
    def _open_form(self, page: Page, base_url: str):
        _mock_task_types(page)
        _mock_task_schema(page, PY_SPELLED_SCHEMA)
        _mock_task_configs(page, {})
        _mock_active_tasks(page, {})
        _navigate_to_tasks(page, base_url)
        page.wait_for_selector("[data-testid='task-type-select']", timeout=5000)
        _select_type(page)
        return page.locator("#task-config-list .le-editor")

    def test_bool_renders_a_checkbox_not_a_text_box(self, page: Page, base_url: str):
        editor = self._open_form(page, base_url)
        expect(editor.locator(".schema-form input[type='checkbox']")).to_have_count(1)

    def test_no_unhandled_type_placeholder_is_shown(self, page: Page, base_url: str):
        """The fallback field advertises the bug in its own placeholder."""
        editor = self._open_form(page, base_url)
        placeholders = editor.locator(".schema-form [placeholder*='Unhandled type']")
        expect(placeholders).to_have_count(0)

    def test_bool_is_not_double_labelled(self, page: Page, base_url: str):
        """`bool` must join `boolean` in SELF_LABELLING_TYPES, or the checkbox comes
        back as the three-line block that set has always existed to prevent."""
        editor = self._open_form(page, base_url)
        expect(editor.locator(".schema-field--self-labelled")).to_have_count(1)
        expect(editor.locator(".schema-field--self-labelled .schema-field__label")).to_have_count(0)
