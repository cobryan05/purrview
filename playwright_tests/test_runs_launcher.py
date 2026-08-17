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


def _readiness_body(positive: int, negative: int) -> str:
    return json.dumps(
        {
            "status": "success",
            "readyCount": positive + negative,
            "readyPositiveCount": positive,
            "readyNegativeCount": negative,
            "needsReviewCount": 0,
            "notStartedCount": 0,
        }
    )


def _mock_previous_run(
    page: Page,
    *,
    now: tuple = (10, 30),
    before: tuple = (5, 15),
    manifest: list | None = None,
    calls: list | None = None,
    model_status: int = 200,
):
    """The three endpoints behind a training card's "+N pos / +N neg since last run".

    `now` and `before` are (positive, negative) ready counts for the config's
    datasets today and over the previous run's manifest. `manifest=None` stands
    for a config that has never produced a model, the normal state of a freshly
    defined recipe.
    """

    def _readiness(route):
        if calls is not None:
            calls.append(route.request.url)
        route.fulfill(status=200, content_type="application/json", body=_readiness_body(*now))

    def _readiness_for_images(route):
        if calls is not None:
            calls.append(route.request.url)
        route.fulfill(status=200, content_type="application/json", body=_readiness_body(*before))

    def _training_images(route):
        if calls is not None:
            calls.append(route.request.url)
        if manifest is None:
            route.fulfill(
                status=model_status,
                content_type="application/json",
                body=json.dumps({"status": "failure", "message": "Model 'primary_model' not found"}),
            )
            return
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "trainImageUuids": manifest,
                    "valImageUuids": [],
                    "trainedAt": "2026-01-02T03:04:05",
                    "modelConfigNodeUuid": "node-1",
                }
            ),
        )

    page.route("**/api/training-configs/*/readiness", _readiness)
    page.route("**/api/training-configs/*/readiness-for-images", _readiness_for_images)
    page.route("**/api/models/*/training-images", _training_images)


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


def _training_instance(status: str, config_uuid: str = "tc-1", typename: str = "TrainModelTask") -> dict:
    """One entry of the task-status payload, shaped like a run the launcher
    started: a throwaway task config carrying the recipe's uuid."""
    return {
        "config": {
            "uuid": "tmp-cfg",
            "name": "Train Primary Model",
            "typename": typename,
            "params": {"training_config_uuid": config_uuid},
        },
        "instance": {"uuid": "task-1", "status": status},
        "continuous": False,
    }


def _mock_task_status(page: Page, instances: dict):
    page.route(
        "**/api/tasks/instances/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "tasks": instances}),
        ),
    )


def _open_launcher(page: Page, base_url: str, **previous_run):
    _mock_launchables(page)
    _mock_previous_run(page, **previous_run)
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-runs']").click()
    page.locator("#tab-runs .workspace-tab-button[data-tab='tab-runs-launch']").click()
    page.wait_for_load_state("networkidle")


def _card(page: Page, name: str):
    return page.locator(f"#runs-launcher-box .runs-launch-card[data-run-name='{name}']")


def _start(page: Page, name: str):
    """The start target. The card is a div wrapping two buttons, so clicking the
    card itself would land on whichever child is under the cursor."""
    return _card(page, name).locator(".runs-launch-card__start")


def _config(page: Page, name: str):
    return _card(page, name).locator(".runs-launch-card__config")


def _active_row(page: Page, task_uuid: str):
    """A run's card on Runs > Active, where a live launch card sends you."""
    return page.locator(f"#active-tasks-list [data-task-uuid='{task_uuid}']")


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
        _mock_previous_run(page)
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
        _start(page, "Primary Model").click()

        wait_until(page, lambda: "start" in calls, "Expected the training run to start")
        assert calls.index("create") < calls.index("start")
        assert payloads[0]["typename"] == "TrainModelTask"
        assert payloads[0]["params"]["training_config_uuid"] == "tc-1"
        # Scaffolding, not a definition the user authored.
        assert payloads[0]["isTemporary"] is True
        # Training is the one launch that opts into persistence: the job runs on
        # the training server, so a tasks_server restart detaches the watcher
        # rather than killing the work, and the recovered task can re-attach to
        # the still-running job instead of waiting for someone to press Resume.
        assert payloads[0]["persistent"] is True

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
        _start(page, "Clean Database").click()

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
        _start(page, "Clean Database").click()

        expect(page.locator("#tab-runs-active")).to_have_class(re.compile(r"\bactive\b"))


class TestTrainingCardDelta:
    """ "+N since last run" on a training card.

    The same number the training config's coverage panel shows — images ready
    now minus the images that trained the model this recipe last produced —
    surfaced where the decision to start a run is actually made.
    """

    def _detail(self, page: Page, name: str):
        return _card(page, name).get_by_test_id("runs-launch-detail")

    def _lead(self, page: Page, name: str):
        """The coloured half — the positives. The negatives trail it in a
        separate, always-muted span."""
        return _card(page, name).get_by_test_id("runs-launch-detail-lead")

    def test_positives_and_negatives_are_counted_separately(self, page: Page, base_url: str):
        """A combined total is misleading: these datasets are overwhelmingly
        background, so most of any growth is confirmed-absence images that add no
        training signal. 1,100 "new" images meaning 12 new positives is exactly
        the number a card must not print."""
        _open_launcher(
            page,
            base_url,
            before=(8, 100),
            now=(20, 1200),
            manifest=[f"img-{i}" for i in range(108)],
        )

        expect(self._detail(page, "Primary Model")).to_have_text("+12 pos · +1,100 neg since last run")
        # Only the positives are coloured, and only they decide the colour.
        expect(self._lead(page, "Primary Model")).to_have_text("+12 pos")
        expect(self._lead(page, "Primary Model")).to_have_class(re.compile(r"\bis-up\b"))

    def test_growth_in_negatives_alone_is_not_coloured_as_progress(self, page: Page, base_url: str):
        """1,100 more confirmed-absence images is real dataset growth and no new
        training signal, so the card must not go green for it."""
        _open_launcher(
            page,
            base_url,
            before=(8, 100),
            now=(8, 1200),
            manifest=[f"img-{i}" for i in range(108)],
        )

        expect(self._detail(page, "Primary Model")).to_have_text("0 pos · +1,100 neg since last run")
        lead = self._lead(page, "Primary Model")
        expect(lead).to_have_class(re.compile(r"\bis-muted\b"))
        expect(lead).not_to_have_class(re.compile(r"\bis-up\b"))

    def test_a_shrunken_dataset_reads_as_negative_not_plus_minus(self, page: Page, base_url: str):
        """Images get deleted and datasets get re-scoped, so the delta is signed."""
        _open_launcher(
            page,
            base_url,
            before=(20, 40),
            now=(5, 40),
            manifest=[f"img-{i}" for i in range(60)],
        )

        expect(self._detail(page, "Primary Model")).to_have_text("-15 pos · 0 neg since last run")

    def test_a_never_trained_recipe_says_so_instead_of_claiming_everything_is_new(self, page: Page, base_url: str):
        _open_launcher(page, base_url, now=(12, 28), manifest=None)

        expect(self._detail(page, "Primary Model")).to_have_text("12 pos · 28 neg ready · not trained yet")

    def test_the_card_paints_before_its_number_arrives(self, page: Page, base_url: str):
        """The launcher must render at the speed of the list endpoints, not at
        the speed of one readiness query per card — there can be dozens."""
        _mock_launchables(page)
        # Never fulfilled: the readiness query stays outstanding for the whole
        # test, which is the condition being asserted.
        held: list = []
        page.route("**/api/training-configs/*/readiness", lambda route: held.append(route))

        page.goto(base_url)
        page.locator(".main-tab-button[data-tab='tab-runs']").click()
        page.locator("#tab-runs .workspace-tab-button[data-tab='tab-runs-launch']").click()

        # No networkidle wait: the point is that the card is there while the
        # readiness call is still in flight.
        expect(_start(page, "Primary Model")).to_be_visible()
        expect(self._detail(page, "Primary Model")).to_have_class(re.compile(r"\bis-pending\b"))
        assert held, "Expected the card to have painted before its count resolved"

        page.unroute_all(behavior="ignoreErrors")

    def test_a_failed_count_leaves_the_card_startable(self, page: Page, base_url: str):
        started: list[dict] = []
        page.route(
            "**/api/tasks/instances/start",
            lambda route: (
                started.append(route.request.post_data_json),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "taskUuid": "task-1"}),
                ),
            ),
        )
        _mock_launchables(page)
        page.route("**/api/training-configs/*/readiness", lambda route: route.fulfill(status=500, body="nope"))
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.locator(".main-tab-button[data-tab='tab-runs']").click()
        page.locator("#tab-runs .workspace-tab-button[data-tab='tab-runs-launch']").click()
        page.wait_for_load_state("networkidle")

        expect(self._detail(page, "Primary Model")).to_have_text("count unavailable")
        _start(page, "Primary Model").click()
        wait_until(page, lambda: bool(started), "A card whose count failed must still start its run")

    def test_the_expensive_per_label_replay_is_not_run_for_a_card(self, page: Page, base_url: str):
        """A card shows two numbers, both of them image counts. The per-label
        stats query over the previous manifest is the slow half of
        loadPreviousRun (0.05-0.7s per config) and nothing on the card needs
        it."""
        label_stats_calls: list[str] = []
        _mock_launchables(page)
        _mock_previous_run(page, before=(8, 100), now=(20, 1200), manifest=[f"img-{i}" for i in range(108)])
        page.route(
            "**/api/training-configs/*/label-stats-for-images",
            lambda route: (
                label_stats_calls.append(route.request.url),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labelStats": [], "totalImages": 0}),
                ),
            ),
        )
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.locator(".main-tab-button[data-tab='tab-runs']").click()
        page.locator("#tab-runs .workspace-tab-button[data-tab='tab-runs-launch']").click()

        expect(self._detail(page, "Primary Model")).to_have_text("+12 pos · +1,100 neg since last run")
        assert not label_stats_calls, "A launcher card must not run the per-label replay"

    def test_refresh_recounts_rather_than_serving_the_cached_number(self, page: Page, base_url: str):
        """The counts are cached so that flipping tabs does not re-query the
        database; ↻ is how you get a fresh reading after annotating."""
        positives = {"count": 12}

        _mock_launchables(page)
        # No previous run, so the card reports today's split on its own — one
        # moving number is all this test needs.
        page.route("**/api/models/*/training-images", lambda route: route.fulfill(status=404, body="{}"))
        page.route(
            "**/api/training-configs/*/readiness",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=_readiness_body(positives["count"], 28),
            ),
        )
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.locator(".main-tab-button[data-tab='tab-runs']").click()
        page.locator("#tab-runs .workspace-tab-button[data-tab='tab-runs-launch']").click()

        expect(self._detail(page, "Primary Model")).to_contain_text("12 pos · 28 neg")

        positives["count"] = 13
        page.locator("#runs-launcher-box .runs-launch-group").first.get_by_test_id("runs-launch-group-action").click()

        expect(self._detail(page, "Primary Model")).to_contain_text("13 pos · 28 neg")


class TestTrainingInProgressBadge:
    """A recipe whose model is training right now is the one card you should not
    be pressing, and the run was visible only on the Active sub-tab."""

    def _badge(self, page: Page, name: str):
        return _card(page, name).get_by_test_id("runs-launch-run-badge")

    def test_a_running_training_is_named_on_its_card(self, page: Page, base_url: str):
        _mock_task_status(page, {"task-1": _training_instance("running")})
        _open_launcher(page, base_url)

        expect(self._badge(page, "Primary Model")).to_have_text("Training in progress")
        expect(_card(page, "Primary Model")).to_have_class(re.compile(r"\bis-live\b"))

    def test_a_queued_run_counts_too(self, page: Page, base_url: str):
        """A queued run is as much a reason not to press Start as a running one."""
        _mock_task_status(page, {"task-1": _training_instance("pending")})
        _open_launcher(page, base_url)

        expect(self._badge(page, "Primary Model")).to_have_text("Queued")

    def test_a_finished_run_leaves_no_badge(self, page: Page, base_url: str):
        _mock_task_status(page, {"task-1": _training_instance("completed")})
        _open_launcher(page, base_url)

        expect(self._badge(page, "Primary Model")).to_be_hidden()
        expect(_card(page, "Primary Model")).not_to_have_class(re.compile(r"\bis-live\b"))

    def test_another_recipes_run_does_not_mark_this_card(self, page: Page, base_url: str):
        """The badge keys on the training config the run was launched from, not
        on 'some training is happening somewhere'."""
        _mock_task_status(page, {"task-1": _training_instance("running", config_uuid="tc-other")})
        _open_launcher(page, base_url)

        expect(self._badge(page, "Primary Model")).to_be_hidden()

    def test_a_non_training_task_does_not_mark_a_training_card(self, page: Page, base_url: str):
        _mock_task_status(
            page,
            {"task-1": _training_instance("running", typename="ApplyFilterChainTask")},
        )
        _open_launcher(page, base_url)

        expect(self._badge(page, "Primary Model")).to_be_hidden()


def _watcher_instance(status: str = "running", profile_uuid: str = "wp-1") -> dict:
    """A running watcher, shaped the way the launcher starts one: a throwaway
    task config carrying the watch profile's uuid."""
    return {
        "config": {
            "uuid": "tmp-watch-cfg",
            "name": "Watch Test",
            "typename": "StreamWatcherTask",
            "params": {"watch_profile_uuid": profile_uuid},
        },
        "instance": {"uuid": "task-w1", "status": status},
        "continuous": True,
    }


class TestWatcherAlreadyRunning:
    """A watcher is the case where a second copy is actively harmful: both open
    the same cameras, both evaluate the same rules, and both fire the same
    actions — duplicate clips, duplicate MQTT, and a stop that stops one of them.
    The card has to say so, and refuse.
    """

    def _badge(self, page: Page, name: str):
        return _card(page, name).get_by_test_id("runs-launch-run-badge")

    def test_a_running_watcher_is_named_on_its_card(self, page: Page, base_url: str):
        _mock_task_status(page, {"task-w1": _watcher_instance("running")})
        _open_launcher(page, base_url)

        expect(self._badge(page, "Test")).to_have_text("Watching now")
        expect(_card(page, "Test")).to_have_class(re.compile(r"\bis-live\b"))

    def test_a_paused_watcher_reads_as_a_watcher_not_a_training(self, page: Page, base_url: str):
        # The wording is per task type on purpose; a wall of watcher cards all
        # saying "Training paused" would be worse than no badge.
        _mock_task_status(page, {"task-w1": _watcher_instance("paused")})
        _open_launcher(page, base_url)

        expect(self._badge(page, "Test")).to_have_text("Watcher paused")

    def test_another_profiles_watcher_does_not_mark_this_card(self, page: Page, base_url: str):
        _mock_task_status(page, {"task-w1": _watcher_instance("running", profile_uuid="wp-other")})
        _open_launcher(page, base_url)

        expect(self._badge(page, "Test")).to_be_hidden()

    def test_pressing_start_does_not_launch_a_second_watcher(self, page: Page, base_url: str):
        started: list = []
        page.route(
            "**/api/tasks/configs/create",
            lambda route: (
                started.append(route.request.post_data),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "configUuid": "should-not-happen"}),
                ),
            )[-1],
        )
        _mock_task_status(page, {"task-w1": _watcher_instance("running")})
        _open_launcher(page, base_url)

        _start(page, "Test").click()

        # It opens the live view instead (see TestWatcherDrillDown) — the point
        # here is only that it does not start a second watcher.
        expect(page.get_by_test_id("watch-live-dialog")).to_be_visible()
        assert started == []

    def test_the_card_invites_the_press_rather_than_blocking_it(self, page: Page, base_url: str):
        """A running watcher card is not "unavailable" — pressing it does the
        useful thing (shows you the watcher), so it must not look disabled."""
        _mock_task_status(page, {"task-w1": _watcher_instance("running")})
        _open_launcher(page, base_url)

        start = _start(page, "Test")
        expect(start).to_have_attribute("title", re.compile(r"live view"))
        expect(start).to_be_enabled()

    def test_an_idle_watcher_card_still_starts(self, page: Page, base_url: str):
        """The guard must not leak into the normal case."""
        _mock_task_status(page, {})
        _open_launcher(page, base_url)

        expect(_start(page, "Test")).not_to_have_attribute("title", re.compile(r".+"))
        expect(_card(page, "Test")).not_to_have_class(re.compile(r"\bis-live\b"))

    def test_the_launcher_opens_no_stream_of_its_own(self, page: Page, base_url: str):
        """Runs owns exactly one task SSE stream, closed when the tab is left
        (test_leaks_and_races.py guards it). The launcher is pushed that stream's
        messages instead of subscribing a second time, so entering and leaving
        the pane must not add a connection.
        """
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
        _mock_task_status(page, {"task-1": _training_instance("running")})
        _open_launcher(page, base_url)
        expect(self._badge(page, "Primary Model")).to_be_visible()

        opened_with_launcher = page.evaluate("() => window.__taskStreams.length")

        # Bounce off Runs and back into the launcher: a per-render subscription
        # would show up as a second connection here.
        page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
        page.wait_for_timeout(300)
        page.locator(".main-tab-button[data-tab='tab-runs']").click()
        page.locator("#tab-runs .workspace-tab-button[data-tab='tab-runs-launch']").click()
        expect(self._badge(page, "Primary Model")).to_be_visible()

        states = page.evaluate("() => window.__taskStreams.map(es => es.readyState)")
        # One stream per Runs visit, and every earlier one closed (readyState 2).
        assert opened_with_launcher == 1, f"launcher visit opened {opened_with_launcher} streams"
        assert states.count(1) <= 1, f"more than one live task stream: {states}"


def _mock_live_preview(page: Page):
    """The one endpoint the drill-down modal polls once it opens."""
    page.route(
        "**/api/tasks/instances/*/live-preview**",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "preview": {
                        "profileName": "Test",
                        "sources": [{"sourceUuid": "src-1", "sourceName": "Cam"}],
                        "sourceUuid": "src-1",
                        "sourceName": "Cam",
                        "frame": None,
                        "detections": [],
                        "tracks": [],
                    },
                }
            ),
        ),
    )


class TestWatcherDrillDown:
    """Pressing a watcher that is already watching should show it to you.

    The launcher knew a watcher was running (it badges "Watching now") and its
    only answer to a press was to refuse. The useful answer is the drill-down
    the NVR pane already has — the thing the user was reaching for.
    """

    def test_pressing_a_running_watcher_opens_the_live_view(self, page: Page, base_url: str):
        _mock_live_preview(page)
        _mock_task_status(page, {"task-w1": _watcher_instance("running")})
        _open_launcher(page, base_url)

        _start(page, "Test").click()

        expect(page.get_by_test_id("watch-live-dialog")).to_be_visible()

    def test_it_opens_the_live_view_for_that_watcher(self, page: Page, base_url: str):
        """Keyed on the running instance, not the profile: opening the wrong
        watcher's frames would be worse than opening none."""
        asked: list = []
        page.route(
            "**/api/tasks/instances/*/live-preview**",
            lambda route: (
                asked.append(route.request.url),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "preview": {"sources": [], "frame": None}}),
                ),
            )[-1],
        )
        _mock_task_status(page, {"task-w1": _watcher_instance("running")})
        _open_launcher(page, base_url)

        _start(page, "Test").click()
        expect(page.get_by_test_id("watch-live-dialog")).to_be_visible()

        wait_until(
            page,
            lambda: any("task-w1" in url for url in asked),
            message=lambda: f"live-preview never asked for the running instance: {asked}",
        )

    def test_an_idle_watcher_card_does_not_open_the_live_view(self, page: Page, base_url: str):
        """Nothing is running, so the press must start a run as it always did."""
        _mock_live_preview(page)
        _mock_task_status(page, {})
        page.route(
            "**/api/tasks/configs/create",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configUuid": "new-cfg"}),
            ),
        )
        _open_launcher(page, base_url)

        _start(page, "Test").click()

        expect(page.get_by_test_id("watch-live-dialog")).not_to_be_visible()

    def test_a_paused_watcher_offers_no_live_view(self, page: Page, base_url: str):
        """A paused watcher has stopped producing frames, so the modal would be
        empty — and a button that opens an empty modal teaches people to ignore
        it. It still gets the default: the run, on the pane that lists it."""
        _mock_live_preview(page)
        _mock_task_status(page, {"task-w1": _watcher_instance("paused")})
        _open_launcher(page, base_url)

        _start(page, "Test").click()

        expect(page.get_by_test_id("watch-live-dialog")).not_to_be_visible()
        expect(page.get_by_text(re.compile(r"no live view"))).to_be_visible()
        expect(page.locator("#tab-runs-active")).to_have_class(re.compile(r"\bactive\b"))

    def test_a_training_card_shows_the_run_on_active_instead(self, page: Page, base_url: str):
        """Only a watcher has a live view. Every other live card used to answer
        a press with a toast saying what would not happen, leaving the user to
        find the run themselves on the next pane over — so it hands them to it."""
        _mock_task_status(page, {"task-1": _training_instance("running")})
        _open_launcher(page, base_url)

        _start(page, "Primary Model").click()

        expect(page.locator("#tab-runs-active")).to_have_class(re.compile(r"\bactive\b"))
        expect(_active_row(page, "task-1")).to_have_class(re.compile(r"\bis-highlighted\b"))
        expect(page.get_by_test_id("watch-live-dialog")).not_to_be_visible()


def _saved_task_instance(status: str = "running", config_uuid: str = "task-cfg-1") -> dict:
    """A running saved task, shaped the way the launcher starts one: the saved
    config itself, not a throwaway clone of it."""
    return {
        "config": {
            "uuid": config_uuid,
            "name": "Clean Database",
            "typename": "DatabaseMaintenanceTask",
            "params": {},
            "isTemporary": False,
        },
        "instance": {"uuid": "task-s1", "status": status},
        "continuous": False,
    }


class TestSavedTaskAlreadyRunning:
    """Trainings and watchers say on the card when they are already running;
    a saved task said nothing, so the only way to know you were about to start a
    second copy of a maintenance job was to check the Active sub-tab first.

    A saved task is started as itself rather than cloned, so the link back to
    the card is the task config's own uuid.
    """

    def _badge(self, page: Page, name: str):
        return _card(page, name).get_by_test_id("runs-launch-run-badge")

    def test_a_running_saved_task_is_named_on_its_card(self, page: Page, base_url: str):
        _mock_task_status(page, {"task-s1": _saved_task_instance("running")})
        _open_launcher(page, base_url)

        expect(self._badge(page, "Clean Database")).to_have_text("Running now")
        expect(_card(page, "Clean Database")).to_have_class(re.compile(r"\bis-live\b"))

    def test_a_paused_saved_task_still_counts(self, page: Page, base_url: str):
        _mock_task_status(page, {"task-s1": _saved_task_instance("paused")})
        _open_launcher(page, base_url)

        expect(self._badge(page, "Clean Database")).to_have_text("Paused")

    def test_a_queued_saved_task_counts_too(self, page: Page, base_url: str):
        _mock_task_status(page, {"task-s1": _saved_task_instance("pending")})
        _open_launcher(page, base_url)

        expect(self._badge(page, "Clean Database")).to_have_text("Queued")

    def test_a_finished_saved_task_leaves_no_badge(self, page: Page, base_url: str):
        _mock_task_status(page, {"task-s1": _saved_task_instance("completed")})
        _open_launcher(page, base_url)

        expect(self._badge(page, "Clean Database")).to_be_hidden()
        expect(_card(page, "Clean Database")).not_to_have_class(re.compile(r"\bis-live\b"))

    def test_another_saved_tasks_run_does_not_mark_this_card(self, page: Page, base_url: str):
        _mock_task_status(page, {"task-s1": _saved_task_instance("running", config_uuid="task-cfg-other")})
        _open_launcher(page, base_url)

        expect(self._badge(page, "Clean Database")).to_be_hidden()

    def test_a_launcher_scaffolding_run_marks_no_saved_task_card(self, page: Page, base_url: str):
        """A training started from here runs through a temporary config. It is
        badged on the training card that launched it and must not leak into the
        Saved-tasks group."""
        _mock_task_status(page, {"task-1": _training_instance("running")})
        _open_launcher(page, base_url)

        expect(self._badge(page, "Primary Model")).to_be_visible()
        expect(self._badge(page, "Clean Database")).to_be_hidden()

    def test_pressing_it_does_not_start_a_second_copy(self, page: Page, base_url: str):
        """The same guard the other kinds get: a saved task already in flight
        would simply do its work twice."""
        started: list = []
        page.route(
            "**/api/tasks/instances/start",
            lambda route: (
                started.append(route.request.post_data_json),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "taskUuid": "should-not-happen"}),
                ),
            )[-1],
        )
        _mock_task_status(page, {"task-s1": _saved_task_instance("running")})
        _open_launcher(page, base_url)

        expect(self._badge(page, "Clean Database")).to_have_text("Running now")
        _start(page, "Clean Database").click()

        expect(page.locator("#tab-runs-active")).to_have_class(re.compile(r"\bactive\b"))
        assert started == []

    def test_pressing_it_shows_the_run_on_active(self, page: Page, base_url: str):
        """A saved task has no live view of its own, so the useful answer to
        "it is already running" is the run itself, marked in a pane that can
        list dozens."""
        _mock_task_status(page, {"task-s1": _saved_task_instance("running")})
        _open_launcher(page, base_url)

        _start(page, "Clean Database").click()

        expect(_active_row(page, "task-s1")).to_be_visible()
        expect(_active_row(page, "task-s1")).to_have_class(re.compile(r"\bis-highlighted\b"))

    def test_a_paused_saved_task_is_blocked_and_shown_too(self, page: Page, base_url: str):
        """Restarting a paused run is not "starting" it — it would leave the
        paused one behind. Resume lives on Active, which is where this lands."""
        started: list = []
        page.route(
            "**/api/tasks/instances/start",
            lambda route: (started.append(1), route.fulfill(status=500, body="{}"))[-1],
        )
        _mock_task_status(page, {"task-s1": _saved_task_instance("paused")})
        _open_launcher(page, base_url)

        _start(page, "Clean Database").click()

        expect(_active_row(page, "task-s1")).to_have_class(re.compile(r"\bis-highlighted\b"))
        assert started == []

    def test_an_idle_saved_task_still_starts(self, page: Page, base_url: str):
        """The guard must not leak into the normal case."""
        started: list = []
        page.route(
            "**/api/tasks/instances/start",
            lambda route: (
                started.append(route.request.post_data_json),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "taskUuid": "task-s2"}),
                ),
            )[-1],
        )
        _mock_task_status(page, {})
        _open_launcher(page, base_url)

        _start(page, "Clean Database").click()

        wait_until(page, lambda: bool(started), "An idle saved task must still start")
        assert started[0]["configUuid"] == "task-cfg-1"


class TestUnscopedFilterChainGuard:
    """ApplyFilterChainTask requires only filter_chain_uuid; every scope param
    is optional, so a one-click run applies the chain to the whole database.
    A chain can discard and relabel images, so that is confirmed first."""

    def test_filter_chain_asks_before_running_over_everything(self, page: Page, base_url: str):
        created: list[str] = []
        page.route("**/api/tasks/configs/create", lambda route: (created.append("x"), route.abort()))

        _open_launcher(page, base_url)
        _start(page, "Remove Corrupt Images").click()

        modal = page.get_by_test_id("modal-shell-dialog")
        expect(modal).to_be_visible()
        expect(modal).to_contain_text("every image in the database")

        modal.locator("button", has_text="Cancel").click()
        page.wait_for_timeout(300)
        assert not created, "Cancelling the confirm must not start the chain"


class TestConfigLink:
    """Every card links to what defines it.

    Starting something you have not touched in weeks is exactly when you want to
    check its settings first, and this pane is the only screen that lists all of
    them — so it is where "wait, how is this one set up?" gets asked.
    """

    def test_each_group_points_at_its_own_configuration_pane(self, page: Page, base_url: str):
        cases = [
            ("Primary Model", "tab-training-management"),
            ("Remove Corrupt Images", "tab-filterchain-management"),
            ("Test", "tab-watcher-management"),
            ("Clean Database", "tab-task-management"),
        ]
        for name, expected_tab in cases:
            _open_launcher(page, base_url)
            _config(page, name).click()

            expect(page.locator("#tab-configuration")).to_have_class(re.compile(r"\bactive\b"))
            expect(page.locator(f"#{expected_tab}")).to_have_class(re.compile(r"\bactive\b"))

    def test_the_clicked_card_arrives_selected(self, page: Page, base_url: str):
        """Landing on the right pane is not enough — the ⚙ names one recipe, so
        that recipe must be the one open, not whatever was last edited."""
        cases = [
            ("Primary Model", "tab-training-management", "config-uuid", "tc-1"),
            ("Remove Corrupt Images", "tab-filterchain-management", "chain-uuid", "fc-1"),
            ("Test", "tab-watcher-management", "watch-uuid", "wp-1"),
            ("Clean Database", "tab-task-management", "config-uuid", "task-cfg-1"),
        ]
        for name, tab, attr, uuid in cases:
            _open_launcher(page, base_url)
            _config(page, name).click()

            expect(page.locator(f"#{tab} [data-{attr}='{uuid}']")).to_have_class(re.compile(r"\bactive\b"))

    def test_the_config_link_does_not_start_anything(self, page: Page, base_url: str):
        """The two controls sit in one card, so the gear must not fall through
        to the start target underneath it."""
        started: list[str] = []
        page.route("**/api/tasks/configs/create", lambda route: (started.append("x"), route.abort()))
        page.route("**/api/tasks/instances/start", lambda route: (started.append("x"), route.abort()))

        _open_launcher(page, base_url)
        _config(page, "Primary Model").click()
        page.wait_for_timeout(400)

        assert not started, "Opening the config must not launch the run"

    def test_the_card_is_not_a_button_inside_a_button(self, page: Page, base_url: str):
        """A <button> cannot legally contain another; browsers recover by
        hoisting the inner one out of the card, which loses its handler."""
        _open_launcher(page, base_url)

        assert _card(page, "Primary Model").evaluate("el => el.tagName") == "DIV"
        assert _start(page, "Primary Model").evaluate("el => el.tagName") == "BUTTON"
        assert _config(page, "Primary Model").evaluate("el => el.tagName") == "BUTTON"
