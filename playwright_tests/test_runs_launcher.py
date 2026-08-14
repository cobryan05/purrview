"""Runs > Start a run — the one place anything is launched from.

Configuration defines recipes; this pane turns one into a running instance and
hands the user to Active, where it can be watched. It exists because starting a
training used to be a button inside its Configuration editor, complete with a
progress bar and a Cancel — job monitoring inside a configuration screen.

Run with:
    pytest playwright_tests/test_runs_launcher.py -p no:asyncio -n 0 -v
"""

import json
import re

from playwright.sync_api import Page, expect

from playwright_tests.waits import wait_until

TRAINING_CONFIGS = [
    {
        "uuid": "tc-1",
        "name": "Primary Model",
        "description": "",
        "datasetUuids": ["ds-1"],
        "modelConfigNodeUuid": "node-1",
        "hyperparamsJson": {},
    }
]

FILTER_CHAINS = [{"uuid": "fc-1", "name": "Remove Corrupt Images", "description": "", "steps": []}]

WATCH_PROFILES = [
    {
        "uuid": "wp-1",
        "name": "Test",
        "description": "",
        "enabled": True,
        "trackerTypename": "IouTracker",
        "defaultAnalysisFps": 5.0,
        "rules": [],
    }
]

SAVED_TASK_CONFIGS = {
    "task-cfg-1": {
        "uuid": "task-cfg-1",
        "name": "Clean Database",
        "typename": "DatabaseMaintenanceTask",
        "params": {},
        "isTemporary": False,
    }
}


def _mock_launchables(page: Page):
    page.route(
        "**/api/training-configs*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": TRAINING_CONFIGS}),
        ),
    )
    page.route(
        "**/api/filter-chains*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "chains": FILTER_CHAINS}),
        ),
    )
    page.route(
        "**/api/watch-profiles*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "profiles": WATCH_PROFILES}),
        ),
    )
    page.route(
        "**/api/tasks/configs/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": SAVED_TASK_CONFIGS}),
        ),
    )


def _open_launcher(page: Page, base_url: str):
    _mock_launchables(page)
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-runs']").click()
    page.locator("#tab-runs .workspace-tab-button[data-tab='tab-runs-launch']").click()
    page.wait_for_load_state("networkidle")


def _card(page: Page, name: str):
    return page.locator(f"#runs-launcher-box .runs-launch-card[data-run-name='{name}']")


class TestLauncherContents:
    def test_lists_every_kind_of_launchable_thing(self, page: Page, base_url: str):
        _open_launcher(page, base_url)

        expect(_card(page, "Primary Model")).to_be_visible()
        expect(_card(page, "Remove Corrupt Images")).to_be_visible()
        expect(_card(page, "Test")).to_be_visible()
        expect(_card(page, "Clean Database")).to_be_visible()

    def test_a_failing_group_does_not_blank_the_others(self, page: Page, base_url: str):
        """One dead endpoint must not cost you the launcher — the other groups
        are still launchable."""
        _mock_launchables(page)
        page.route("**/api/filter-chains*", lambda route: route.fulfill(status=500, body="nope"))

        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.locator(".main-tab-button[data-tab='tab-runs']").click()
        page.locator("#tab-runs .workspace-tab-button[data-tab='tab-runs-launch']").click()
        page.wait_for_load_state("networkidle")

        expect(_card(page, "Primary Model")).to_be_visible()
        expect(page.locator("#runs-launcher-box")).to_contain_text("Could not load filter chains")


class TestLaunching:
    def test_training_start_creates_a_temporary_config_then_starts_it(self, page: Page, base_url: str):
        """A training config is a recipe, not a task, so starting one needs a
        throwaway task config carrying its uuid."""
        calls: list[str] = []
        payloads: list[dict] = []

        def _create(route):
            calls.append("create")
            payloads.append(route.request.post_data_json)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configUuid": "tmp-cfg"}),
            )

        page.route("**/api/tasks/configs/create", _create)
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

        _open_launcher(page, base_url)
        _card(page, "Primary Model").click()

        wait_until(page, lambda: "start" in calls, "Expected the training run to start")
        assert calls.index("create") < calls.index("start")
        assert payloads[0]["typename"] == "TrainModelTask"
        assert payloads[0]["params"]["training_config_uuid"] == "tc-1"
        # Scaffolding, not a definition the user authored.
        assert payloads[0]["isTemporary"] is True
        assert payloads[0]["persistent"] is False

    def test_saved_task_starts_without_cloning_itself(self, page: Page, base_url: str):
        """A config the user authored is started directly. Cloning one per press
        is what littered the Tasks list with duplicate 'Train ...' rows."""
        created: list[str] = []
        started: list[dict] = []

        page.route("**/api/tasks/configs/create", lambda route: (created.append("x"), route.abort()))
        page.route(
            "**/api/tasks/instances/start",
            lambda route: (
                started.append(route.request.post_data_json),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "taskUuid": "task-9"}),
                ),
            ),
        )

        _open_launcher(page, base_url)
        _card(page, "Clean Database").click()

        wait_until(page, lambda: bool(started), "Expected the saved task to start")
        assert started[0]["configUuid"] == "task-cfg-1"
        assert not created, "A saved config must not be cloned to start it"

    def test_starting_lands_on_active(self, page: Page, base_url: str):
        page.route(
            "**/api/tasks/instances/start",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "taskUuid": "task-9"}),
            ),
        )
        _open_launcher(page, base_url)
        _card(page, "Clean Database").click()

        expect(page.locator("#tab-runs-active")).to_have_class(re.compile(r"\bactive\b"))


class TestUnscopedFilterChainGuard:
    """ApplyFilterChainTask requires only filter_chain_uuid; every scope param
    is optional, so a one-click run applies the chain to the whole database.
    A chain can discard and relabel images, so that is confirmed first."""

    def test_filter_chain_asks_before_running_over_everything(self, page: Page, base_url: str):
        created: list[str] = []
        page.route("**/api/tasks/configs/create", lambda route: (created.append("x"), route.abort()))

        _open_launcher(page, base_url)
        _card(page, "Remove Corrupt Images").click()

        modal = page.get_by_test_id("modal-shell-dialog")
        expect(modal).to_be_visible()
        expect(modal).to_contain_text("every image in the database")

        modal.locator("button", has_text="Cancel").click()
        page.wait_for_timeout(300)
        assert not created, "Cancelling the confirm must not start the chain"
