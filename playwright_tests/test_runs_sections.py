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

from playwright_tests.waits import wait_until

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


# Persistence ("_persistent" on the config's params) is independent of
# CONTINUOUS: this watcher runs until stopped but will NOT come back after a
# server restart, while PERSISTENT_JOB below runs to completion and will.
NON_PERSISTENT_WATCHER = {
    **WATCHER,
    "config": {
        "name": "One-off Watcher",
        "uuid": "cfg-watch-oneoff",
        "typename": "StreamWatcherTask",
        "params": {"_persistent": False},
    },
}

PERSISTENT_JOB = {
    **TRAINING,
    "config": {
        "name": "Resuming Training",
        "uuid": "cfg-train-persist",
        "typename": "TrainModelTask",
        "params": {"_persistent": True},
    },
}

FINISHED_PERSISTENT_JOB = {
    **PERSISTENT_JOB,
    "instance": {
        "uuid": "t-train-done",
        "status": "completed",
        "startedAt": "2026-01-01T00:00:00",
        "endedAt": "2026-01-02T00:00:00",
    },
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


class TestPersistenceBadge:
    """ "Will this come back on its own?" is not what the sections answer.

    Continuous/Jobs splits by *when a run ends*; persistence is about whether the
    server starts it again afterwards. A card that reads the wrong one of the two
    tells the user a watcher will survive a restart when it will not.
    """

    def _badges(self, page: Page):
        return page.locator("#active-tasks-list [data-testid='task-persistent-badge']")

    def _card(self, page: Page, task_uuid: str):
        return page.locator(f"#active-tasks-list [data-task-uuid='{task_uuid}']")

    def test_only_the_persistent_run_is_badged(self, page: Page, base_url: str):
        """Both are live; one is continuous-but-not-persistent and the other is
        persistent-but-not-continuous, so keying off `continuous` badges exactly
        the wrong card."""
        _open_runs(
            page,
            base_url,
            {"t-watch": NON_PERSISTENT_WATCHER, "t-train": PERSISTENT_JOB},
        )

        expect(self._badges(page)).to_have_count(1)
        expect(self._card(page, "t-train").locator("[data-testid='task-persistent-badge']")).to_be_visible()
        expect(self._card(page, "t-watch").locator("[data-testid='task-persistent-badge']")).to_have_count(0)

    def test_a_finished_run_is_not_badged(self, page: Page, base_url: str):
        """The flag is still true on the config, but a completed instance is not
        resumed on startup — badging it would promise a restart that never comes."""
        _open_runs(page, base_url, {"t-train-done": FINISHED_PERSISTENT_JOB})

        expect(_section_headers(page).filter(has_text="Completed")).to_have_count(1)
        expect(self._badges(page)).to_have_count(0)


class TestBadge:
    def _badge(self, page: Page):
        return page.locator("#task-activity-badge")

    def test_badge_counts_everything_listed_under_active(self, page: Page, base_url: str):
        """Jobs and continuous runs alike. Counting jobs only made the number
        unexplainable from the pane it describes: Active listed the watcher and
        the badge silently disagreed."""
        _open_runs(page, base_url, {"t-watch": WATCHER, "t-train": TRAINING})
        expect(self._badge(page)).to_have_text("2")

    def test_an_armed_watcher_alone_still_lights_the_badge(self, page: Page, base_url: str):
        _open_runs(page, base_url, {"t-watch": WATCHER})
        expect(self._badge(page)).to_have_text("1")

    def _active_tab_badge(self, page: Page):
        return page.locator("#task-activity-badge-active")

    def test_the_active_sub_tab_carries_the_same_count(self, page: Page, base_url: str):
        """From inside Runs the main-tab badge is out of the eyeline, and the
        count is about the Active pane — so it belongs beside it too."""
        _open_runs(page, base_url, {"t-watch": WATCHER, "t-train": TRAINING})
        expect(self._active_tab_badge(page)).to_have_text("2")

    def test_the_active_sub_tab_badge_hides_at_zero(self, page: Page, base_url: str):
        _open_runs(page, base_url, {})
        expect(self._active_tab_badge(page)).to_be_hidden()

    def test_the_badge_is_reseeded_on_return(self, page: Page, base_url: str):
        """Returning re-seeds from a fresh fetch rather than trusting whatever
        was last on screen — the stream may have been closed the whole time."""
        _open_runs(page, base_url, {"t-train": TRAINING})
        expect(self._badge(page)).to_have_text("1")

        page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
        page.wait_for_timeout(300)
        _mock_tasks(page, {})
        page.locator(".main-tab-button[data-tab='tab-runs']").click()

        expect(self._badge(page)).to_be_hidden()
        expect(self._active_tab_badge(page)).to_be_hidden()


class TestStreamLifetimeFollowsTheCount:
    """The stream lives off-tab exactly as long as something can still finish.

    Closing it unconditionally froze the badge — a job that ended while the user
    was annotating still read as running. Holding it open unconditionally costs
    a connection per browser tab for the life of the app, feeding a badge that
    reads zero and cannot change (see test_leaks_and_races.py, which pins the
    idle case). A live count is what separates the two.
    """

    _COUNTER = """
        window.__taskStreams = { opened: 0, closed: 0 };
        const Native = window.EventSource;
        window.EventSource = function (url, init) {
            const es = new Native(url, init);
            const isTaskStream = String(url).includes('/api/tasks/stream');
            if (isTaskStream) window.__taskStreams.opened += 1;
            const close = es.close.bind(es);
            es.close = () => { if (isTaskStream) window.__taskStreams.closed += 1; return close(); };
            return es;
        };
        window.EventSource.prototype = Native.prototype;
    """

    def test_a_running_job_keeps_the_stream_alive_off_tab(self, page: Page, base_url: str):
        page.add_init_script(self._COUNTER)
        _open_runs(page, base_url, {"t-train": TRAINING})
        expect(page.locator("#task-activity-badge")).to_have_text("1")

        page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
        page.wait_for_timeout(700)

        counts = page.evaluate("window.__taskStreams")
        assert counts["closed"] == 0, "A job is still running — the badge must keep counting"
        assert counts["opened"] == 1

    def test_nothing_running_closes_the_stream_off_tab(self, page: Page, base_url: str):
        """Nothing can finish behind your back, so an idle app holds no socket.

        A watcher is the interesting case rather than an empty list: it lights
        the badge (it *is* running) but never finishes, so there is no
        completion to stay connected for."""
        page.add_init_script(self._COUNTER)
        _open_runs(page, base_url, {"t-watch": WATCHER})  # continuous: never finishes
        expect(page.locator("#task-activity-badge")).to_have_text("1")

        page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()

        wait_until(
            page,
            lambda: page.evaluate("window.__taskStreams.closed") == 1,
            lambda: "leaving Runs did not close the task stream, closed="
            + str(page.evaluate("window.__taskStreams.closed")),
        )

    # NOT covered here: the last job *ending* while the user is away, which
    # `updateBadge` handles by dropping the stream at that moment. The fixture
    # cannot deliver a second SSE event mid-test — `page.route` fulfils a body
    # in one shot, and the server-side stream reads the mocked tasks proxy
    # rather than anything the test can change — so the assertion would be
    # vacuous. The tab-change branches below are the two that are reachable.
