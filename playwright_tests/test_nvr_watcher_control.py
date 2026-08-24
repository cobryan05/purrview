"""
Starting and stopping a watcher from the NVR → Live pane, and noticing when it happens.

Two things this pane got wrong, both of which made it read as a dead end:

  - a card said "stopped" and offered no way to do anything about it, sending the user
    to Runs to start the very watcher the card is about;
  - the pane only learned that a watcher had started on a full page reload — including
    one started from this pane a second earlier.

Starting deliberately goes through the same `launchRun` path as the Runs launcher
(a temporary task config plus an instance). Two mechanisms for starting a watcher is how
you end up with watchers that behave differently depending on which button started them,
so the test asserts on the *request*, not just on the button.

Run with:
    pytest playwright_tests/test_nvr_watcher_control.py -p no:asyncio -v
"""

import json

from playwright.sync_api import Page, expect

from playwright_tests.test_nvr_live_view import PROFILE_UUID, TASK_UUID, _profile, _task


def _mock(page: Page, *, task_status=None):
    """Wire the pane. `task_status=None` means the watcher is not running.

    Returns a dict collecting the write requests the buttons make.
    """
    seen = {"configs": [], "starts": [], "cancels": []}

    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/watch-profiles"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "profiles": [_profile()]}),
        ),
    )
    tasks = {} if task_status is None else {TASK_UUID: _task(task_status)}
    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/tasks/instances/get"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "tasks": tasks}),
        ),
    )
    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/tasks/instances/status-data"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "statusData": {}}),
        ),
    )
    page.route(
        lambda url: "/live-preview" in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "preview": None}),
        ),
    )

    def _record(bucket, route, body=None):
        try:
            seen[bucket].append(route.request.post_data_json)
        except Exception:  # a body-less request is still worth counting
            seen[bucket].append(None)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(body or {"status": "success"}),
        )

    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/tasks/configs/create"),
        lambda route: _record("configs", route, {"status": "success", "uuid": "cfg-new"}),
    )
    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/tasks/instances/start"),
        lambda route: _record("starts", route, {"status": "success", "uuid": TASK_UUID}),
    )
    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/tasks/instances/cancel"),
        lambda route: _record("cancels", route),
    )
    return seen


def _open_live_pane(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-nvr']").click()
    page.wait_for_selector("#nvr-live-box .nvr-profile-card")


class TestStartStopButtons:
    def test_a_stopped_watcher_offers_start(self, page: Page, base_url: str):
        _mock(page)
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-start-watcher']")).to_be_visible()
        expect(page.locator("[data-testid='nvr-stop-watcher']")).to_have_count(0)

    def test_a_running_watcher_offers_stop(self, page: Page, base_url: str):
        _mock(page, task_status="running")
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-stop-watcher']")).to_be_visible()
        expect(page.locator("[data-testid='nvr-start-watcher']")).to_have_count(0)

    def test_start_launches_this_profile(self, page: Page, base_url: str):
        """The same config Runs would write — not a second way to start a watcher."""
        seen = _mock(page)
        _open_live_pane(page, base_url)
        page.locator("[data-testid='nvr-start-watcher']").click()
        page.wait_for_function("() => true")
        expect(page.locator("[data-testid='nvr-start-watcher']")).to_be_disabled()

        assert seen["configs"], "no task config was created"
        config = seen["configs"][0]
        assert config["typename"] == "StreamWatcherTask"
        assert config["params"]["watch_profile_uuid"] == PROFILE_UUID
        # Scaffolding, so it stays out of the Tasks list — same as the Runs launcher.
        assert config["isTemporary"] is True

    def test_stop_cancels_the_running_instance(self, page: Page, base_url: str):
        seen = _mock(page, task_status="running")
        _open_live_pane(page, base_url)
        page.locator("[data-testid='nvr-stop-watcher']").click()
        expect(page.locator("[data-testid='nvr-stop-watcher']")).to_be_disabled()

        assert seen["cancels"], "no cancel was sent"
        assert seen["cancels"][0]["taskUuids"] == [TASK_UUID]


class TestLiveUpdates:
    def test_the_pane_notices_a_watcher_starting_without_a_reload(self, page: Page, base_url: str):
        """The pane subscribes to the shared task stream; a start elsewhere must land.

        The stream frame is delivered by hand — capturing the EventSource, as
        test_leaks_and_races does — so the moment the watcher "starts" is under the
        test's control rather than racing the pane's first render.

        The API is flipped to running *before* the frame is delivered, because the pane
        must re-read rather than render from the frame's contents: a pane that trusted
        the frame alone would look right here and still be stale about everything the
        frame does not carry (the profile's sources, its rules).
        """
        _mock(page)
        page.add_init_script("""
            window.__taskStreams = [];
            const OrigES = window.EventSource;
            window.EventSource = function(url, opts) {
                const es = new OrigES(url, opts);
                if (String(url).includes('/api/tasks/stream')) window.__taskStreams.push(es);
                return es;
            };
            window.EventSource.prototype = OrigES.prototype;
            """)
        _open_live_pane(page, base_url)
        expect(page.locator("[data-testid='nvr-start-watcher']")).to_be_visible()

        page.route(
            lambda url: url.split("?")[0].rstrip("/").endswith("/api/tasks/instances/get"),
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "tasks": {TASK_UUID: _task("running")}}),
            ),
        )

        delivered = page.evaluate(
            """(taskUuid) => {
                const frame = {tasks: {[taskUuid]: {
                    config: {typename: 'StreamWatcherTask', params: {}},
                    instance: {uuid: taskUuid, status: 'running'},
                }}};
                let n = 0;
                for (const es of window.__taskStreams || []) {
                    if (es.onmessage) { es.onmessage({data: JSON.stringify(frame)}); n += 1; }
                }
                return n;
            }""",
            TASK_UUID,
        )
        assert delivered >= 1, "the NVR pane never opened the task stream"

        expect(page.locator("[data-testid='nvr-stop-watcher']")).to_be_visible(timeout=10000)
        expect(page.locator(".nvr-state-pill").first).to_have_text("running")

    def test_a_stream_tick_during_a_render_does_not_duplicate_the_cards(self, page: Page, base_url: str):
        """Every watcher appeared twice once the stream subscription was added.

        `renderLive` awaits three APIs between clearing the pane and appending to it, so
        two overlapping runs — the tab activation and a stream tick — both appended, and
        one watch profile produced two cards. The regression is only visible when the
        second render starts while the first is still in flight, so the API calls are
        slowed down to guarantee the overlap.
        """
        _mock(page, task_status="running")
        page.add_init_script("""
            window.__taskStreams = [];
            const OrigES = window.EventSource;
            window.EventSource = function(url, opts) {
                const es = new OrigES(url, opts);
                if (String(url).includes('/api/tasks/stream')) window.__taskStreams.push(es);
                return es;
            };
            window.EventSource.prototype = OrigES.prototype;
            """)
        # Wide enough that a second render definitely starts mid-flight.
        page.route(
            lambda url: url.split("?")[0].rstrip("/").endswith("/api/watch-profiles"),
            lambda route: (
                page.wait_for_timeout(300),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "profiles": [_profile()]}),
                ),
            ),
        )
        _open_live_pane(page, base_url)

        # Several ticks, each carrying a *different* set of running watchers so the
        # signature changes every time and each one really does request a render.
        for status in ("paused", "running", "paused", "running"):
            page.evaluate(
                """([taskUuid, status]) => {
                    const frame = {tasks: {[taskUuid]: {
                        config: {typename: 'StreamWatcherTask', params: {}},
                        instance: {uuid: taskUuid, status},
                    }}};
                    for (const es of window.__taskStreams || []) {
                        if (es.onmessage) es.onmessage({data: JSON.stringify(frame)});
                    }
                }""",
                [TASK_UUID, status],
            )

        page.wait_for_timeout(1500)
        expect(page.locator(".nvr-profile-card")).to_have_count(1)
        expect(page.locator(".nvr-live-list")).to_have_count(1)
