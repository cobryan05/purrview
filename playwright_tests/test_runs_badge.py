"""The Runs badge, before anyone has opened Runs.

Two defects this covers:

  1. The count was painted by Runs' own lazy init, so a freshly loaded page read
     a bare "Runs" until you visited the tab — which is exactly backwards for a
     badge whose whole job is telling you something is running from elsewhere.
  2. It counted jobs only. A running watcher appeared under Active but not in
     the number describing Active, with no rule visible from the screen.

Run with:
    pytest playwright_tests/test_runs_badge.py -p no:asyncio -n 0 -v
"""

import json

from playwright.sync_api import Page, expect


def _task(uuid, *, status="running", continuous=False, typename="CollectTrainingImagesTask"):
    return {
        "instance": {"uuid": uuid, "status": status, "startedAt": "2026-01-01T00:00:00", "endedAt": None},
        "config": {"name": uuid, "uuid": f"cfg-{uuid}", "typename": typename, "isTemporary": True, "params": {}},
        "progress": None,
        "continuous": continuous,
        "isPausedInPlace": False,
        "logLines": [],
        "statusMessage": "",
        "statusActions": [],
    }


WATCHER = _task("t-watch", continuous=True, typename="StreamWatcherTask")
JOB = _task("t-job")
PAUSED_JOB = _task("t-paused", status="paused")
DONE_JOB = _task("t-done", status="completed")


def _load(page: Page, base_url: str, tasks: dict):
    """Load the app with a fixed task list and DO NOT open Runs."""
    body = json.dumps({"status": "success", "tasks": tasks})
    page.route(
        "**/api/tasks/instances/get",
        lambda route: route.fulfill(status=200, content_type="application/json", body=body),
    )
    page.goto(base_url)
    page.wait_for_load_state("networkidle")


def _badge(page: Page):
    return page.locator("#task-activity-badge")


class TestSeededAtPageLoad:
    def test_the_count_is_shown_without_ever_opening_runs(self, page: Page, base_url: str):
        _load(page, base_url, {"t-job": JOB, "t-paused": PAUSED_JOB})

        expect(_badge(page)).to_be_visible()
        expect(_badge(page)).to_have_text("2")

    def test_nothing_running_shows_no_badge(self, page: Page, base_url: str):
        """Hidden at zero: a badge that is always there stops being a signal."""
        _load(page, base_url, {"t-done": DONE_JOB})

        expect(_badge(page)).to_be_hidden()

    def test_finished_runs_do_not_count(self, page: Page, base_url: str):
        _load(page, base_url, {"t-job": JOB, "t-done": DONE_JOB})

        expect(_badge(page)).to_have_text("1")


class TestCountsEverythingUnderActive:
    def test_a_running_watcher_counts(self, page: Page, base_url: str):
        """It is listed under Active, so it belongs in Active's number."""
        _load(page, base_url, {"t-watch": WATCHER})

        expect(_badge(page)).to_have_text("1")

    def test_jobs_and_continuous_runs_are_summed(self, page: Page, base_url: str):
        _load(page, base_url, {"t-watch": WATCHER, "t-job": JOB})

        expect(_badge(page)).to_have_text("2")

    def test_the_tooltip_explains_the_number(self, page: Page, base_url: str):
        """'Why does it say 2?' should be answerable without opening the tab."""
        _load(page, base_url, {"t-watch": WATCHER, "t-job": JOB})

        title = _badge(page).get_attribute("title") or ""
        assert "1 job" in title, title
        assert "1 continuous run" in title, title


class TestBothBadgesAgree:
    def test_the_active_sub_tab_badge_matches(self, page: Page, base_url: str):
        _load(page, base_url, {"t-watch": WATCHER, "t-job": JOB})
        page.locator(".main-tab-button[data-tab='tab-runs']").click()
        page.wait_for_load_state("networkidle")

        expect(page.locator("#task-activity-badge-active")).to_have_text("2")
        expect(_badge(page)).to_have_text("2")
