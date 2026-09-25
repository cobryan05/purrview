"""The Runs badge, before anyone has opened Runs.

Two defects this covers:

  1. The count was painted by Runs' own lazy init, so a freshly loaded page read
     a bare "Jobs" until you visited the tab — which is exactly backwards for a
     badge whose whole job is telling you something is running from elsewhere.
  2. It counted jobs only. A running watcher appeared under Running but not in
     the number describing Running, with no rule visible from the screen.

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


class TestCountsEverythingUnderRunning:
    def test_a_running_watcher_counts(self, page: Page, base_url: str):
        """It is listed under Running, so it belongs in Running's number."""
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
        assert "1 continuous job" in title, title


class TestBothBadgesAgree:
    def test_the_active_sub_tab_badge_matches(self, page: Page, base_url: str):
        _load(page, base_url, {"t-watch": WATCHER, "t-job": JOB})
        page.locator(".main-tab-button[data-tab='tab-runs']").click()
        page.wait_for_load_state("networkidle")

        expect(page.locator("#task-activity-badge-active")).to_have_text("2")
        expect(_badge(page)).to_have_text("2")


def _with_status(task, message, level):
    return {**task, "statusMessage": message, "statusLevel": level}


class TestTheWarningState:
    """A CPU-only training job is 10-50x slower and otherwise looks like any
    other run. The badge borrows the warning colour so it is visible from every
    tab — but only for tasks that asked to be noticed, which is why the level is
    an explicit field rather than "statusMessage is set"."""

    def test_a_warning_task_colours_the_badge(self, page: Page, base_url: str):
        task = _with_status(JOB, "WARNING: no GPU detected — training on CPU", "warning")
        _load(page, base_url, {"t-job": task})

        expect(_badge(page)).to_have_class("tab-badge tab-badge--warning")
        expect(_badge(page)).to_have_attribute("title", "WARNING: no GPU detected — training on CPU")

    def test_an_ordinary_status_message_does_not(self, page: Page, base_url: str):
        """The half that matters: a running watcher narrates through the same
        field all day. Keying the colour on the message being non-empty would
        pin the badge amber permanently and make it mean nothing."""
        task = _with_status(WATCHER, "Watching 7 cameras", "info")
        _load(page, base_url, {"t-watch": task})

        expect(_badge(page)).not_to_have_class("tab-badge tab-badge--warning")
        expect(_badge(page)).to_have_attribute("title", "1 continuous job — click to see them")

    def test_a_finished_warning_run_does_not_colour_it(self, page: Page, base_url: str):
        """The warning describes work in progress; a completed job is not live
        and does not count towards the badge, so it must not tint it either."""
        done = _with_status(DONE_JOB, "WARNING: no GPU detected", "warning")
        _load(page, base_url, {"t-job": JOB, "t-done": done})

        expect(_badge(page)).to_have_text("1")
        expect(_badge(page)).not_to_have_class("tab-badge tab-badge--warning")
