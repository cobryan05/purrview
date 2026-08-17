"""An active run is a dead end without a way back to what it is running.

The Active pane names the run and shows its progress, but the recipe behind it —
the training config, the watch profile — was only reachable by remembering which
Configuration screen defines that task type and finding it there by name. And a
watcher, whose whole output is a live picture, had no way to show it: the live
view existed on the NVR pane and on the launcher card, but not on the card for
the watcher that is actually running.

Run with:
    pytest playwright_tests/test_runs_active_card_links.py -p no:asyncio -n 0 -v
"""

import json
import re

from playwright.sync_api import Page, expect

WATCHER = {
    "instance": {"uuid": "t-watch", "status": "running", "startedAt": "2026-01-01T00:00:00", "endedAt": None},
    "config": {
        "name": "Watch Test",
        "uuid": "cfg-watch",
        "typename": "StreamWatcherTask",
        "isTemporary": True,
        "params": {"watch_profile_uuid": "profile-1"},
    },
    "progress": None,
    "continuous": True,
    "isPausedInPlace": False,
    "logLines": [],
    "statusMessage": "Watching…",
    "statusActions": [],
}

TRAINING = {
    "instance": {"uuid": "t-train", "status": "running", "startedAt": "2026-01-01T00:00:00", "endedAt": None},
    "config": {
        "name": "Train Primary Model",
        "uuid": "cfg-train",
        "typename": "TrainModelTask",
        "isTemporary": True,
        "params": {"training_config_uuid": "tc-1"},
    },
    "progress": 42.0,
    "continuous": False,
    "isPausedInPlace": False,
    "logLines": [],
    "statusMessage": "epoch 46/200",
    "statusActions": [],
}

SAVED_JOB = {
    "instance": {"uuid": "t-collect", "status": "running", "startedAt": "2026-01-01T00:00:00", "endedAt": None},
    "config": {
        "name": "Collect Run 1",
        "uuid": "cfg-collect",
        "typename": "CollectTrainingImagesTask",
        "isTemporary": False,
        "params": {},
    },
    "progress": 10.0,
    "continuous": False,
    "isPausedInPlace": False,
    "logLines": [],
    "statusMessage": "",
    "statusActions": [],
}

FINISHED_TRAINING = {
    "instance": {
        "uuid": "t-old-train",
        "status": "error",
        "startedAt": "2026-01-01T00:00:00",
        "endedAt": "2026-01-02T00:00:00",
    },
    "config": {
        "name": "Train Primary Model",
        "uuid": "cfg-old",
        "typename": "TrainModelTask",
        "isTemporary": True,
        "params": {"training_config_uuid": "tc-1"},
    },
    "progress": 12.0,
    "continuous": False,
    "isPausedInPlace": False,
    "logLines": [],
    "statusMessage": "",
    "statusActions": [],
}

TEMPORARY_UNKNOWN = {
    "instance": {"uuid": "t-temp", "status": "running", "startedAt": "2026-01-01T00:00:00", "endedAt": None},
    "config": {
        "name": "Preview run",
        "uuid": "cfg-temp",
        "typename": "SomePreviewTask",
        "isTemporary": True,
        "params": {},
    },
    "progress": None,
    "continuous": False,
    "isPausedInPlace": False,
    "logLines": [],
    "statusMessage": "",
    "statusActions": [],
}


def _open_runs(page: Page, base_url: str, tasks: dict):
    body = json.dumps({"status": "success", "tasks": tasks})
    page.route(
        "**/api/tasks/instances/get",
        lambda route: route.fulfill(status=200, content_type="application/json", body=body),
    )
    page.route(
        "**/api/tasks/configs/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": {}}),
        ),
    )
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-runs']").click()
    page.wait_for_load_state("networkidle")


def _card(page: Page, task_uuid: str):
    return page.locator(f"#active-tasks-list [data-task-uuid='{task_uuid}']")


class TestConfigurationLink:
    def test_every_active_card_offers_its_configuration(self, page: Page, base_url: str):
        _open_runs(page, base_url, {"t-watch": WATCHER, "t-train": TRAINING, "t-collect": SAVED_JOB})

        for uuid in ("t-watch", "t-train", "t-collect"):
            expect(_card(page, uuid).locator("[data-testid='task-config-jump']")).to_have_count(1)

    def test_a_finished_run_still_links_to_its_recipe(self, page: Page, base_url: str):
        """A failed training is exactly when you want to look at its settings."""
        _open_runs(page, base_url, {"t-old-train": FINISHED_TRAINING})

        expect(_card(page, "t-old-train").locator("[data-testid='task-config-jump']")).to_have_count(1)

    def test_a_run_with_no_reachable_definition_offers_nothing(self, page: Page, base_url: str):
        """A temporary config of an unrecognised type has no screen to send the
        user to — the Tasks list hides temporary configs, so the button would
        land them on a blank editor."""
        _open_runs(page, base_url, {"t-temp": TEMPORARY_UNKNOWN})

        expect(_card(page, "t-temp").locator("[data-testid='task-config-jump']")).to_have_count(0)

    def test_a_watcher_lands_on_its_watch_profile_not_its_task_config(self, page: Page, base_url: str):
        """The task config behind a launcher-started run is throwaway
        scaffolding; the profile is the thing the user authored."""
        _open_runs(page, base_url, {"t-watch": WATCHER})

        _card(page, "t-watch").locator("[data-testid='task-config-jump']").click()

        expect(page.locator("#tab-configuration")).to_have_class(re.compile(r"\bactive\b"))
        expect(page.locator("#tab-watcher-management")).to_have_class(re.compile(r"\bactive\b"))

    def test_a_training_lands_on_its_training_config(self, page: Page, base_url: str):
        _open_runs(page, base_url, {"t-train": TRAINING})

        _card(page, "t-train").locator("[data-testid='task-config-jump']").click()

        expect(page.locator("#tab-training-management")).to_have_class(re.compile(r"\bactive\b"))


class TestWatcherLiveView:
    def test_a_running_watcher_offers_its_live_view(self, page: Page, base_url: str):
        _open_runs(page, base_url, {"t-watch": WATCHER})

        expect(_card(page, "t-watch").locator("[data-testid='task-live-view']")).to_have_count(1)

    def test_pressing_it_opens_the_live_view(self, page: Page, base_url: str):
        page.route(
            "**/api/tasks/live-preview**",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "available": False}),
            ),
        )
        _open_runs(page, base_url, {"t-watch": WATCHER})

        _card(page, "t-watch").locator("[data-testid='task-live-view']").click()

        expect(page.locator("[data-testid='watch-live-dialog']")).to_be_visible()
        expect(page.locator("[data-testid='watch-live-dialog']")).to_contain_text("Watch Test")

    def test_only_watchers_get_one(self, page: Page, base_url: str):
        _open_runs(page, base_url, {"t-train": TRAINING})

        expect(_card(page, "t-train").locator("[data-testid='task-live-view']")).to_have_count(0)

    def test_a_stopped_watcher_has_nothing_to_show(self, page: Page, base_url: str):
        """A finished watcher produces no frames, so the modal would open empty."""
        stopped = json.loads(json.dumps(WATCHER))
        stopped["instance"]["status"] = "completed"
        stopped["instance"]["endedAt"] = "2026-01-02T00:00:00"
        _open_runs(page, base_url, {"t-watch": stopped})

        expect(_card(page, "t-watch").locator("[data-testid='task-live-view']")).to_have_count(0)
