"""Runs separates work that finishes from work that runs until stopped.

A training runs a fixed number of epochs and finishes; a stream watcher never
does. Listing them together makes "what is running?" unanswerable, because the
answer is always "at least the watcher" — and folding the watcher into the tab
badge pins it at >=1 forever, at which point it has stopped being a signal.

The backend declares which is which (Task.CONTINUOUS, reported as `continuous`
on the task status payload); this asserts the frontend acts on it.

Run with:
    pytest playwright_tests/test_runs_sections.py -p no:asyncio -n 0 -v
"""

import json

from playwright.sync_api import Page, expect

WATCHER = {
    "instance": {"uuid": "t-watch", "status": "running", "startedAt": "2026-01-01T00:00:00", "endedAt": None},
    "config": {"name": "Watcher Test", "uuid": "cfg-watch", "typename": "StreamWatcherTask"},
    "progress": None,
    "continuous": True,
    "isPausedInPlace": False,
    "logLines": [],
    "statusMessage": "Watching…",
    "statusActions": [],
}

TRAINING = {
    "instance": {"uuid": "t-train", "status": "running", "startedAt": "2026-01-01T00:00:00", "endedAt": None},
    "config": {"name": "Train Primary Model", "uuid": "cfg-train", "typename": "TrainModelTask"},
    "progress": 42.0,
    "continuous": False,
    "isPausedInPlace": False,
    "logLines": [],
    "statusMessage": "epoch 46/200",
    "statusActions": [],
}

FINISHED_WATCHER = {
    "instance": {
        "uuid": "t-old-watch",
        "status": "completed",
        "startedAt": "2026-01-01T00:00:00",
        "endedAt": "2026-01-02T00:00:00",
    },
    "config": {"name": "Old Watcher", "uuid": "cfg-old", "typename": "StreamWatcherTask"},
    "progress": 100.0,
    "continuous": True,
    "isPausedInPlace": False,
    "logLines": [],
    "statusMessage": "",
    "statusActions": [],
}


def _mock_tasks(page: Page, tasks: dict):
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


def _open_runs(page: Page, base_url: str, tasks: dict):
    _mock_tasks(page, tasks)
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-runs']").click()
    page.wait_for_load_state("networkidle")


def _section_headers(page: Page):
    return page.locator("#active-tasks-list .task-section-header")


class TestSectioning:
    def test_continuous_and_jobs_are_separate_sections(self, page: Page, base_url: str):
        _open_runs(page, base_url, {"t-watch": WATCHER, "t-train": TRAINING})

        expect(_section_headers(page).filter(has_text="Continuous")).to_have_count(1)
        expect(_section_headers(page).filter(has_text="Jobs")).to_have_count(1)

    def test_a_section_with_nothing_in_it_is_not_rendered(self, page: Page, base_url: str):
        """No empty "Continuous" heading on an install that has never run one."""
        _open_runs(page, base_url, {"t-train": TRAINING})

        expect(_section_headers(page).filter(has_text="Jobs")).to_have_count(1)
        expect(_section_headers(page).filter(has_text="Continuous")).to_have_count(0)

    def test_a_stopped_continuous_task_is_history_not_a_live_service(self, page: Page, base_url: str):
        """Once it has ended it is history like any other run, so it belongs in
        Completed rather than lingering as a dead entry under Continuous."""
        _open_runs(page, base_url, {"t-old-watch": FINISHED_WATCHER})

        expect(_section_headers(page).filter(has_text="Completed")).to_have_count(1)
        expect(_section_headers(page).filter(has_text="Continuous")).to_have_count(0)


class TestBadge:
    def _badge(self, page: Page):
        return page.locator("#task-activity-badge")

    def test_badge_counts_jobs_only(self, page: Page, base_url: str):
        _open_runs(page, base_url, {"t-watch": WATCHER, "t-train": TRAINING})
        expect(self._badge(page)).to_have_text("1")

    def test_an_armed_watcher_alone_leaves_the_badge_hidden(self, page: Page, base_url: str):
        """The whole point: a watcher is supposed to stay armed for weeks, so it
        must not permanently light the badge."""
        _open_runs(page, base_url, {"t-watch": WATCHER})
        expect(self._badge(page)).to_be_hidden()
