"""Acting on a run from its card has to visibly do something.

The bug behind these tests: a watcher wedged mid-shutdown could not be stopped, and
the UI said nothing at all about it. Stop sent a cancel, the server answered 200, and
the card carried on reading "Running" — identical to the case where the stop worked,
and identical again to the case where the server had no such task to stop. The only
feedback any card action gave was an error toast; success and no-op were both silent.

So the cancel endpoint now reports three outcomes (`taskUuids` = stopped, `stopping`,
`notRunning`) and the card says which one it got. These tests pin all three, plus the
"Stopping…" mark surviving the poll that rebuilds every card.

Run with:
    pytest playwright_tests/test_task_action_feedback.py -p no:asyncio -n 0 -v
"""

import json

from playwright.sync_api import Page, expect

TASK_UUID = "t-watch"

WATCHER = {
    "instance": {"uuid": TASK_UUID, "status": "running", "startedAt": "2026-01-01T00:00:00", "endedAt": None},
    "config": {"name": "Watch Test", "uuid": "cfg-watch", "typename": "StreamWatcherTask"},
    "progress": None,
    "continuous": True,
    "isPausedInPlace": False,
    "logLines": [],
    "statusMessage": "Watching",
    "statusActions": [],
}


def _mock(page: Page, *, cancel_response: dict):
    """Serve one running watcher, and a cancel that answers with *cancel_response*."""
    seen = {"cancels": []}

    page.route(
        "**/api/tasks/instances/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "tasks": {TASK_UUID: WATCHER}}),
        ),
    )
    page.route(
        "**/api/tasks/configs/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": {}}),
        ),
    )
    page.route(
        "**/api/tasks/instances/status-data",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "statusData": {}}),
        ),
    )

    def _cancel(route):
        try:
            seen["cancels"].append(route.request.post_data_json)
        except Exception:
            seen["cancels"].append(None)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", **cancel_response}),
        )

    page.route("**/api/tasks/instances/cancel", _cancel)
    return seen


def _open_runs(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-runs']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("[data-task-uuid='t-watch']", timeout=5000)


def _toast(page: Page):
    return page.locator("#toast-container > div")


def _card(page: Page):
    return page.locator("[data-task-uuid='t-watch']")


class TestStopFeedback:
    def test_a_stop_that_worked_says_so(self, page: Page, base_url: str):
        seen = _mock(page, cancel_response={"taskUuids": [TASK_UUID], "stopping": [], "notRunning": []})
        _open_runs(page, base_url)

        _card(page).locator("[data-testid='task-stop']").click()

        expect(_toast(page).filter(has_text='Stopped "Watch Test"')).to_be_visible()
        assert seen["cancels"], "no cancel was sent"
        assert seen["cancels"][0]["taskUuids"] == [TASK_UUID]

    def test_a_stop_still_in_progress_is_not_reported_as_stopped(self, page: Page, base_url: str):
        """The wedged-watcher case. It must not look like a clean stop."""
        _mock(page, cancel_response={"taskUuids": [], "stopping": [TASK_UUID], "notRunning": []})
        _open_runs(page, base_url)

        _card(page).locator("[data-testid='task-stop']").click()

        expect(_toast(page).filter(has_text="still shutting down")).to_be_visible()
        expect(_toast(page).filter(has_text='Stopped "Watch Test"')).to_have_count(0)

    def test_a_task_the_server_could_not_find_is_not_reported_as_stopped(self, page: Page, base_url: str):
        """The state an orphaned task left behind: Stop used to succeed and change nothing."""
        _mock(page, cancel_response={"taskUuids": [], "stopping": [], "notRunning": [TASK_UUID]})
        _open_runs(page, base_url)

        _card(page).locator("[data-testid='task-stop']").click()

        expect(_toast(page).filter(has_text="was not running")).to_be_visible()

    def test_a_failing_stop_still_reports(self, page: Page, base_url: str):
        """These endpoints answer 200 with a failure body — checking the status code
        alone reported every server-side error as a successful stop."""
        _mock(page, cancel_response={})
        page.route(
            "**/api/tasks/instances/cancel",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "failure", "message": "boom"}),
            ),
        )
        _open_runs(page, base_url)

        _card(page).locator("[data-testid='task-stop']").click()

        expect(_toast(page).filter(has_text="boom")).to_be_visible()


class TestStoppingMark:
    def test_the_card_is_marked_while_it_is_still_stopping(self, page: Page, base_url: str):
        """A toast fades; a shutdown may not. The card has to carry the state.

        The poll rebuilds every card from scratch, so this also pins that the mark
        survives a re-render — the reason it is held outside the card at all.
        """
        _mock(page, cancel_response={"taskUuids": [], "stopping": [TASK_UUID], "notRunning": []})
        _open_runs(page, base_url)

        expect(_card(page)).not_to_have_attribute("data-stopping", "true")

        _card(page).locator("[data-testid='task-stop']").click()

        expect(_card(page)).to_have_attribute("data-stopping", "true")
        expect(_card(page)).to_contain_text("Stopping…")

    def test_a_clean_stop_does_not_mark_the_card(self, page: Page, base_url: str):
        _mock(page, cancel_response={"taskUuids": [TASK_UUID], "stopping": [], "notRunning": []})
        _open_runs(page, base_url)

        _card(page).locator("[data-testid='task-stop']").click()
        expect(_toast(page).filter(has_text='Stopped "Watch Test"')).to_be_visible()

        expect(_card(page)).not_to_have_attribute("data-stopping", "true")
