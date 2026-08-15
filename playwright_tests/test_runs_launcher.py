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
