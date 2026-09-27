"""Guided tours (ui/tour/).

The mocked workspace has no labels and no sources, so it is exactly the fresh install
the overview tour exists for. `conftest` sets the app's per-browser kill switch so the
rest of the suite never boots under a spotlight; `_boot` here clears it and routes
`tour_state` to say what a real install would.

Tours branch by calling each other: the overview visits each tab and offers that tab's
own tour, and finishing the branch returns to the step *after* the one that offered it.
What matters here is conditional — the tour appears on an empty, unseen workspace and
not once it has been declined; a branch returns to its parent but pausing one does not;
the ? offers the page you are on rather than every tour; a paused tour resumes itself on
returning to its page while a dismissed one stays quiet; and a step's own navigation
does not count as the user arriving somewhere.
"""

import json
import re
import time

from playwright.sync_api import Page, expect

TOUR_STATE_RE = re.compile(r"/api/settings/tour_state$")


def _route_tour_state(page: Page, value: str | None, puts: list | None = None):
    """Serve `tour_state` as a fresh install (None → 404) or a stored value, and
    record what the app writes back."""

    def handle(route):
        if route.request.method == "PUT":
            if puts is not None:
                puts.append(json.loads(route.request.post_data)["value"])
            route.fulfill(status=200, content_type="application/json", body='{"status":"success"}')
        elif value is None:
            route.fulfill(status=404, content_type="application/json", body='{"detail":"not found"}')
        else:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "value": value}),
            )

    page.route(TOUR_STATE_RE, handle)


def _stored(**overrides) -> str:
    """A tour_state payload with every field the app expects.

    `fabHintSeen` defaults to True: the one-time callout opens a popup anchored to the
    ?, which would swallow the first click of any test that opens the menu. The test
    that covers the callout passes False for itself.
    """
    state = {
        "neverShow": False,
        "completed": [],
        "dismissed": [],
        "paused": [],
        "progress": {},
        "stack": [],
        "fabHintSeen": True,
    }
    state.update(overrides)
    return json.dumps(state)


def _saved_matching(page: Page, puts: list, predicate, timeout_ms: int = 10000) -> dict:
    """The first written state satisfying `predicate`.

    Never "the last PUT": several things write this setting around one user action —
    the pause or completion itself, and the ? callout recording that it was shown —
    so the write under test is rarely the newest one.
    """
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        for raw in puts:
            state = json.loads(raw)
            if predicate(state):
                return state
        assert time.monotonic() < deadline, f"no matching tour_state PUT within {timeout_ms}ms; saw {puts}"
        page.wait_for_timeout(50)


def _route_labels(page: Page, labels: list):
    page.route(
        re.compile(r"/api/labels$"),
        lambda r: r.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": labels}),
        ),
    )


def _boot(page: Page, base_url: str):
    # Init scripts run in registration order, so this undoes the fixture's switch.
    page.add_init_script("try { localStorage.removeItem('ww_tours_auto'); } catch (_) {}")
    page.goto(base_url)
    page.wait_for_load_state("networkidle")


def _progress(page: Page):
    return page.locator("[data-testid='tour-progress']")


#: A step renders only once its anchor resolves, and the engine waits up to 4 s for
#: one that never appears before rendering without a spotlight — longer than
#: Playwright's 5 s default leaves room for.
_STEP_TIMEOUT_MS = 8000


def _advance(page: Page, times: int = 1):
    """Press Next `times`, waiting for each step to land before the next press.

    See `_walk_to_end`: a press arriving before the previous step rendered is handled
    by the step still on screen, which advances the tour twice.

    The baseline is `text_content`, not `inner_text`: the progress line is
    `text-transform: uppercase`, and `to_have_text` matches against textContent — so an
    inner_text baseline can never equal it and every wait would pass instantly.
    """
    for _ in range(times):
        before = _progress(page).text_content()
        page.locator("[data-testid='tour-next']").click()
        expect(_progress(page)).not_to_have_text(before, timeout=_STEP_TIMEOUT_MS)


def _walk_to_end(page: Page, limit: int = 24):
    """Press Next until the button offers to finish, and leave it there.

    Steps whose anchor is absent on an empty workspace are passed over, so the number
    of presses is a property of the workspace, not of the tour.

    Each press waits for the progress line to actually change before the next one.
    A step renders only once its anchor resolves, which can outlast any fixed delay —
    and a second press against the still-displayed previous step advances *twice*,
    walking straight out of the tour under test.
    """
    next_btn = page.locator("[data-testid='tour-next']")
    for _ in range(limit):
        text = next_btn.inner_text().strip()
        if text == "Done" or text.startswith("Back to "):
            return next_btn
        before = _progress(page).text_content()
        next_btn.click()
        expect(_progress(page)).not_to_have_text(before, timeout=_STEP_TIMEOUT_MS)
    raise AssertionError("the tour never reached its last step")


class TestTheTourOffersItselfOnlyWhenWanted:
    def test_a_fresh_empty_workspace_gets_the_tour(self, page: Page, base_url: str):
        _route_tour_state(page, None)
        _boot(page, base_url)

        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        expect(_progress(page)).to_have_text("Getting started · 1 of 7")
        # Step two is the one that lives on Configuration; the tour must take the
        # user there from the default Watch tab rather than point at a hidden tab.
        page.locator("[data-testid='tour-next']").click()
        expect(page.locator("#tab-configuration")).to_be_visible()

    def test_a_workspace_that_declined_is_left_alone(self, page: Page, base_url: str):
        _route_tour_state(page, _stored(dismissed=["overview"]))
        _boot(page, base_url)

        page.wait_for_timeout(1200)
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)
        # And it stayed on the landing tab, since nothing navigated.
        expect(page.locator("#tab-nvr")).to_be_visible()

    def test_a_page_tour_never_starts_itself(self, page: Page, base_url: str):
        """Only the overview carries `autoStart`. A workspace that has finished it
        must not then be ambushed by the four page tours, which are unseen."""
        _route_tour_state(page, _stored(completed=["overview"]))
        _boot(page, base_url)

        page.wait_for_timeout(1200)
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)

    def test_a_workspace_with_labels_is_not_a_fresh_install(self, page: Page, base_url: str):
        """Eligibility is "is anything set up", not "first visit": an install someone
        else configured needs no walkthrough."""
        _route_tour_state(page, None)
        _route_labels(
            page, [{"uuid": "lbl-1", "name": "Tabby", "color": "#fff", "parentUuid": None, "scope": "object"}]
        )
        _boot(page, base_url)

        page.wait_for_timeout(1200)
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)

    def test_seeded_image_labels_do_not_count_as_setup(self, page: Page, base_url: str):
        """Corrupt/Blurry exist on every install; their presence says nothing."""
        _route_tour_state(page, None)
        _route_labels(
            page,
            [
                {"uuid": "img-1", "name": "Corrupt", "color": "#fff", "parentUuid": None, "scope": "image"},
                {"uuid": "img-2", "name": "Blurry", "color": "#fff", "parentUuid": None, "scope": "image"},
            ],
        )
        _boot(page, base_url)

        expect(page.locator("[data-testid='tour-card']")).to_be_visible()


class TestPausingAndDismissingAreDifferentThings:
    """Pause means "I'll come back"; dismiss means "stop asking". The two are stored
    apart because only one of them lets a tour start itself again."""

    def test_pause_records_a_pause_not_a_dismissal(self, page: Page, base_url: str):
        puts: list = []
        _route_tour_state(page, None, puts)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        page.locator("[data-testid='tour-next']").click()
        expect(_progress(page)).to_have_text("Getting started · 2 of 7")

        page.locator("[data-testid='tour-skip']").click()

        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)
        expect(page.locator("[data-testid='tour-spotlight']")).to_have_count(0)
        saved = _saved_matching(page, puts, lambda st: "overview" in st["paused"])
        assert "overview" not in saved["dismissed"], saved
        assert saved["progress"]["overview"] == 1, saved

    def test_the_corner_cross_dismisses_for_good(self, page: Page, base_url: str):
        puts: list = []
        _route_tour_state(page, None, puts)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()

        page.locator("[data-testid='tour-dismiss']").click()

        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)
        saved = _saved_matching(page, puts, lambda st: "overview" in st["dismissed"])
        assert "overview" not in saved["paused"], saved

    def test_escape_pauses_rather_than_dismisses(self, page: Page, base_url: str):
        puts: list = []
        _route_tour_state(page, None, puts)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        expect(page.locator("[data-testid='tour-back']")).to_be_hidden()

        page.locator("[data-testid='tour-next']").click()
        expect(_progress(page)).to_have_text("Getting started · 2 of 7")
        page.locator("[data-testid='tour-back']").click()
        expect(_progress(page)).to_have_text("Getting started · 1 of 7")

        page.keyboard.press("Escape")

        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)
        saved = _saved_matching(page, puts, lambda st: "overview" in st["paused"])
        assert "overview" not in saved["dismissed"], saved

    def test_the_opt_out_box_switches_tours_off(self, page: Page, base_url: str):
        puts: list = []
        _route_tour_state(page, None, puts)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()

        page.locator("[data-testid='tour-never-show']").check()

        assert _saved_matching(page, puts, lambda st: st["neverShow"] is True)

    def test_done_on_the_last_step_records_completion(self, page: Page, base_url: str):
        puts: list = []
        _route_tour_state(page, None, puts)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()

        next_btn = _walk_to_end(page)
        expect(_progress(page)).to_have_text("Getting started · 7 of 7")
        next_btn.click()

        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)
        assert _saved_matching(page, puts, lambda st: "overview" in st["completed"])


class TestBranchesReturnToTheTourThatOfferedThem:
    """The point of the whole design: diving into a tab's tour must not cost you the
    overview. Jobs is the branch under test because it is the shortest."""

    def _to_the_jobs_step(self, page: Page):
        _advance(page, 4)
        expect(_progress(page)).to_have_text("Getting started · 5 of 7")

    def test_finishing_a_branch_lands_on_the_step_after_the_one_that_offered_it(self, page: Page, base_url: str):
        _route_tour_state(page, None)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        self._to_the_jobs_step(page)

        page.locator("[data-testid='tour-branch-0']").click()

        # The branch counts its own steps, and says what it will return to.
        expect(_progress(page)).to_have_text("Jobs · 1 of 2")
        last = _walk_to_end(page)
        expect(last).to_have_text("Back to Getting started")
        last.click()

        # Step 6, not step 5 again and not step 1: the overview resumes *after* the
        # step that branched.
        expect(_progress(page)).to_have_text("Getting started · 6 of 7")
        expect(page.locator("[data-testid='tour-title']")).to_have_text("Settings — and the keyboard shortcuts")

    def test_pausing_a_branch_does_not_return_to_the_parent(self, page: Page, base_url: str):
        """Pausing is "stop", not "go up one level" — the parent must stay off screen."""
        puts: list = []
        _route_tour_state(page, None, puts)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        self._to_the_jobs_step(page)
        page.locator("[data-testid='tour-branch-0']").click()
        expect(_progress(page)).to_have_text("Jobs · 1 of 2")

        page.locator("[data-testid='tour-skip']").click()

        page.wait_for_timeout(800)
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)
        saved = _saved_matching(page, puts, lambda s: "runs" in s["paused"])
        # The overview is still on the stack, so the ? can still hand it back.
        assert [f["tourId"] for f in saved["stack"]] == ["overview"], saved

    def test_a_branching_step_offers_both_ways_on(self, page: Page, base_url: str):
        _route_tour_state(page, None)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        self._to_the_jobs_step(page)

        # Going deeper and going along are distinct controls, and Next is the one
        # that stays in the overview.
        expect(page.locator("[data-testid='tour-branch-0']")).to_be_visible()
        expect(page.locator("[data-testid='tour-next']")).to_have_text("Next tab →")
        page.locator("[data-testid='tour-next']").click()
        expect(_progress(page)).to_have_text("Getting started · 6 of 7")

    def test_the_last_overview_step_offers_no_branch(self, page: Page, base_url: str):
        _route_tour_state(page, None)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        _walk_to_end(page)

        expect(page.locator("[data-testid='tour-branches']")).to_be_hidden()


class TestTheHelpButtonIsPerPage:
    def test_it_offers_this_page_and_the_overview_but_not_other_pages(self, page: Page, base_url: str):
        _route_tour_state(page, _stored(completed=["overview"]))
        _boot(page, base_url)
        page.wait_for_timeout(800)

        page.locator(".main-tab-button[data-tab='tab-nvr']").click()
        fab = page.locator("[data-testid='tour-help-fab']")
        expect(fab).to_be_visible()
        fab.click()

        expect(page.locator("[data-testid='tour-help-run-watch']")).to_be_visible()
        # The overview has no `pages`, so it follows you everywhere.
        expect(page.locator("[data-testid='tour-help-run-overview']")).to_be_visible()
        # Configure's tour belongs to Configure, not here.
        expect(page.locator("[data-testid='tour-help-run-configuration']")).to_have_count(0)

    def test_the_dot_marks_a_page_whose_tour_is_unseen(self, page: Page, base_url: str):
        """The dot is the entire invitation, so it has to distinguish a page with an
        unseen tour from one whose tours are all done."""
        _route_tour_state(page, _stored(completed=["overview", "watch"]))
        _boot(page, base_url)
        page.wait_for_timeout(800)
        fab = page.locator("[data-testid='tour-help-fab']")

        page.locator(".main-tab-button[data-tab='tab-nvr']").click()
        expect(fab).to_be_visible()
        expect(fab).not_to_have_class(re.compile(r"tour-help-fab--unseen"))

        page.locator(".main-tab-button[data-tab='tab-runs']").click()
        expect(fab).to_have_class(re.compile(r"tour-help-fab--unseen"))

    def test_it_offers_the_step_the_user_walked_away_from(self, page: Page, base_url: str):
        _route_tour_state(page, _stored(paused=["overview"], progress={"overview": 1}))
        _boot(page, base_url)
        page.wait_for_timeout(800)
        # A paused overview does not auto-start: it has no `pages`, so there is no
        # page to come back to and the ? is the way in.
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)

        page.locator("[data-testid='tour-help-fab']").click()
        expect(page.locator("[data-testid='tour-help-run-overview']")).to_contain_text("Resume at step 2")
        page.locator("[data-testid='tour-help-run-overview']").click()

        expect(_progress(page)).to_have_text("Getting started · 2 of 7")

    def test_restart_is_offered_separately_and_goes_to_step_one(self, page: Page, base_url: str):
        _route_tour_state(page, _stored(neverShow=True, paused=["overview"], progress={"overview": 3}))
        _boot(page, base_url)
        # neverShow is on, so nothing auto-started — but the button still offers it,
        # because the list only ever appears when someone asks for it.
        page.wait_for_timeout(800)
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)

        page.locator("[data-testid='tour-help-fab']").click()
        page.locator("[data-testid='tour-help-restart-overview']").click()

        expect(_progress(page)).to_have_text("Getting started · 1 of 7")

    def test_the_button_stands_down_while_a_tour_is_running(self, page: Page, base_url: str):
        _route_tour_state(page, None)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()

        # Not merely unstyled: hidden, so it cannot be clicked out from under the card.
        expect(page.locator("[data-testid='tour-help-fab']")).to_be_hidden()

        page.locator("[data-testid='tour-skip']").click()
        expect(page.locator("[data-testid='tour-help-fab']")).to_be_visible()


class TestAPausedTourComesBackOnItsOwnPage:
    def test_arriving_on_the_page_resumes_where_it_stopped(self, page: Page, base_url: str):
        _route_tour_state(page, _stored(completed=["overview"], paused=["runs"], progress={"runs": 1}))
        _boot(page, base_url)
        page.wait_for_timeout(800)
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)

        page.locator(".main-tab-button[data-tab='tab-runs']").click()

        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        expect(_progress(page)).to_have_text("Jobs · 2 of 2")

    def test_a_dismissed_tour_stays_quiet_on_the_same_navigation(self, page: Page, base_url: str):
        """The only difference from the test above is pause vs dismiss."""
        _route_tour_state(page, _stored(completed=["overview"], dismissed=["runs"], progress={"runs": 1}))
        _boot(page, base_url)
        page.wait_for_timeout(800)

        page.locator(".main-tab-button[data-tab='tab-runs']").click()

        page.wait_for_timeout(1200)
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)

    def test_an_unseen_tour_stays_quiet_and_only_marks_the_button(self, page: Page, base_url: str):
        """Unseen is not paused: landing on the page must not start a walkthrough the
        user never asked for."""
        _route_tour_state(page, _stored(completed=["overview"]))
        _boot(page, base_url)
        page.wait_for_timeout(800)

        page.locator(".main-tab-button[data-tab='tab-runs']").click()

        page.wait_for_timeout(1200)
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)
        expect(page.locator("[data-testid='tour-help-fab']")).to_have_class(re.compile(r"tour-help-fab--unseen"))

    def test_a_steps_own_navigation_does_not_hijack_the_running_tour(self, page: Page, base_url: str):
        """The engine navigates by clicking tabs, which fires the same event the
        resume listener watches. Without the guard the paused Configure tour would
        take over the moment the overview stepped onto Configuration."""
        # The overview is marked complete so it does not auto-start and pull the ?
        # out from under the click below; it is then run deliberately.
        _route_tour_state(
            page,
            _stored(completed=["overview"], paused=["configuration"], progress={"configuration": 2}),
        )
        _boot(page, base_url)
        page.wait_for_timeout(800)
        page.locator("[data-testid='tour-help-fab']").click()
        page.locator("[data-testid='tour-help-run-overview']").click()
        expect(_progress(page)).to_have_text("Getting started · 1 of 7")

        # Step two navigates to Configuration, which is the paused tour's page.
        page.locator("[data-testid='tour-next']").click()

        expect(page.locator("#tab-configuration")).to_be_visible()
        page.wait_for_timeout(1500)
        expect(_progress(page)).to_have_text("Getting started · 2 of 7")


class TestStepsReachTheirAnchors:
    def _run_labels_tour(self, page: Page, base_url: str):
        """The label steps live on the Labels & Tags screen's own tour now."""
        _route_tour_state(page, _stored(completed=["overview"]))
        _boot(page, base_url)
        page.wait_for_timeout(800)
        page.locator(".main-tab-button[data-tab='tab-configuration']").click()
        page.locator("#tab-configuration [data-tab='tab-tag-management']").click()
        page.wait_for_timeout(400)
        page.locator("[data-testid='tour-help-fab']").click()
        page.locator("[data-testid='tour-help-run-labels']").click()
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()

    def test_a_step_on_another_pane_navigates_there_and_frames_the_control(self, page: Page, base_url: str):
        self._run_labels_tour(page, base_url)
        next_btn = page.locator("[data-testid='tour-next']")

        # 1 → 2: the label-creation step, whose anchor is a row that only exists
        # once the Labels & Tags pane has mounted.
        _advance(page)

        expect(page.locator("[data-testid='tour-title']")).to_have_text("Create the top of the hierarchy")
        row = page.locator("[data-tour='config.labels.newObject']")
        expect(row).to_be_visible()
        spot = page.locator("[data-testid='tour-spotlight']")
        expect(spot).to_be_visible()
        page.wait_for_timeout(600)
        rb = row.bounding_box()
        sb = spot.bounding_box()
        # The spotlight is the row plus its padding — not somewhere else on the page.
        assert abs(sb["x"] - rb["x"]) < 12 and abs(sb["y"] - rb["y"]) < 12, (sb, rb)
        assert abs(sb["width"] - rb["width"]) < 24, (sb, rb)
        # The framed control is usable through the dim: the input takes focus.
        row.locator("input[type='text']").click()
        expect(row.locator("input[type='text']")).to_be_focused()

    def test_a_waiting_step_advances_when_the_user_does_the_thing(self, page: Page, base_url: str):
        self._run_labels_tour(page, base_url)
        next_btn = page.locator("[data-testid='tour-next']")
        _advance(page)
        expect(page.locator("[data-testid='tour-title']")).to_have_text("Create the top of the hierarchy")
        # A step that waits offers to be skipped, not "Next": the point is doing it.
        expect(next_btn).to_have_text("Skip this step")

        # Serve the label the creation is about to make — the same route answers the
        # POST. Routed only now because the pane had to render empty first.
        _route_labels(page, [{"uuid": "lbl-1", "name": "Cat", "color": "#fff", "parentUuid": None, "scope": "object"}])
        name = page.locator("[data-tour='config.labels.newObject'] input[type='text']")
        name.fill("Cat")
        name.press("Enter")

        expect(page.locator("[data-testid='tour-title']")).to_have_text("Add a variant under it")
        # The step that follows points at a control on the new label's own row, which
        # is opacity:0 until the row is hovered — the pointer is nowhere near it, so
        # the spotlight has to reveal the cluster or it would ring an invisible button.
        expect(page.locator("[data-tour='config.labels.addChild']")).to_be_visible()
        page.wait_for_function(
            "() => {"
            " const el = document.querySelector('[data-tour=\"config.labels.addChild\"]');"
            " return el && getComputedStyle(el.closest('.lt-acts')).opacity === '1';"
            "}"
        )

    def test_the_annotate_tour_reaches_a_pane_three_tiers_down(self, page: Page, base_url: str):
        """Annotate nests main → section → workspace, so a step there is only
        reachable once `navigate` walks a path rather than a single sub-tab."""
        _route_tour_state(page, _stored(completed=["overview"]))
        _boot(page, base_url)
        page.wait_for_timeout(800)
        page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
        page.locator("[data-testid='tour-help-fab']").click()
        page.locator("[data-testid='tour-help-run-annotate']").click()

        expect(page.locator("[data-testid='tour-title']")).to_have_text("Files is the image browser")
        expect(page.locator("#tab-files")).to_have_class(re.compile(r"\bactive\b"))
        expect(page.locator("[data-testid='tour-spotlight']")).to_be_visible()


class TestSettingsCanRunATour:
    def test_run_from_settings_starts_it_even_when_tours_are_off(self, page: Page, base_url: str):
        _route_tour_state(page, _stored(neverShow=True, completed=["overview"]))
        _boot(page, base_url)
        page.wait_for_timeout(800)
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)

        page.locator(".main-tab-button[data-tab='tab-settings']").click()
        page.locator("#tab-settings [data-tab='tab-settings-interface']").click()
        auto = page.locator("[data-testid='settings-tours-auto']")
        expect(auto).not_to_be_checked()

        page.locator("[data-testid='settings-tours-run']").click()

        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        expect(_progress(page)).to_have_text("Getting started · 1 of 7")


class TestConfigurationScreensHaveTheirOwnTours:
    """Configuration has eleven screens behind one main tab. One tour across all of
    them told a user standing on Watch Rules nothing about watch rules."""

    def _open_config(self, page: Page, base_url: str, screen: str):
        _route_tour_state(page, _stored(completed=["overview"]))
        _boot(page, base_url)
        page.wait_for_timeout(800)
        page.locator(".main-tab-button[data-tab='tab-configuration']").click()
        page.locator(f"#tab-configuration [data-tab='{screen}']").click()
        page.wait_for_timeout(400)

    def test_the_offer_follows_the_rail_without_a_main_tab_change(self, page: Page, base_url: str):
        """Moving between screens is not a MAIN_TAB_CHANGED, so this only works if
        every tab tier announces itself."""
        self._open_config(page, base_url, "tab-tag-management")
        fab = page.locator("[data-testid='tour-help-fab']")
        fab.click()
        expect(page.locator("[data-testid='tour-help-run-labels']")).to_be_visible()
        expect(page.locator("[data-testid='tour-help-run-watch-rules']")).to_have_count(0)
        fab.click()

        page.locator("#tab-configuration [data-tab='tab-watcher-management']").click()
        page.wait_for_timeout(400)
        fab.click()

        expect(page.locator("[data-testid='tour-help-run-watch-rules']")).to_be_visible()
        expect(page.locator("[data-testid='tour-help-run-labels']")).to_have_count(0)

    def test_the_screens_tour_is_offered_above_the_tab_and_overview_ones(self, page: Page, base_url: str):
        """All three are relevant on Watch Rules; the narrowest is the one that
        answers "what is this screen"."""
        self._open_config(page, base_url, "tab-watcher-management")
        page.locator("[data-testid='tour-help-fab']").click()

        names = page.locator("[data-testid='tour-help-menu'] .tour-help-menu__name")
        expect(names.first).to_have_text("Watch Rules")
        # The broader two are still reachable from here.
        expect(page.locator("[data-testid='tour-help-run-configuration']")).to_be_visible()
        expect(page.locator("[data-testid='tour-help-run-overview']")).to_be_visible()

    def test_finishing_a_screen_tour_returns_to_the_configure_walk(self, page: Page, base_url: str):
        """The same branch-and-return as the top level, one tier down."""
        self._open_config(page, base_url, "tab-config-overview")
        page.locator("[data-testid='tour-help-fab']").click()
        page.locator("[data-testid='tour-help-run-configuration']").click()
        expect(_progress(page)).to_have_text("Configure · 1 of 7")

        # 1 → 2 is the Sources step; branch into it because it is the shortest.
        _advance(page, 2)
        expect(_progress(page)).to_have_text("Configure · 3 of 7")

        page.locator("[data-testid='tour-branch-0']").click()
        expect(_progress(page)).to_have_text("Sources · 1 of 2")
        last = _walk_to_end(page)
        expect(_progress(page)).to_have_text("Sources · 2 of 2")
        expect(last).to_have_text("Back to Configure")
        last.click()

        expect(_progress(page)).to_have_text("Configure · 4 of 7")

    def test_a_paused_screen_tour_resumes_on_returning_to_that_screen(self, page: Page, base_url: str):
        _route_tour_state(
            page,
            _stored(completed=["overview"], paused=["watch-rules"], progress={"watch-rules": 1}),
        )
        _boot(page, base_url)
        page.wait_for_timeout(800)
        page.locator(".main-tab-button[data-tab='tab-configuration']").click()
        page.locator("#tab-configuration [data-tab='tab-tag-management']").click()
        page.wait_for_timeout(600)
        # Not this screen, so nothing starts.
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)

        page.locator("#tab-configuration [data-tab='tab-watcher-management']").click()

        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        expect(_progress(page)).to_have_text("Watch Rules · 2 of 5")


class TestTheOneTimeHintExplainsTheHighlightedButton:
    """The dot on the ? is easy to miss, and a user who never notices it never learns
    the per-screen tours exist. It is said once, and never auto-opens a tour."""

    def test_it_appears_once_on_the_first_page_with_an_unseen_tour(self, page: Page, base_url: str):
        puts: list = []
        _route_tour_state(page, _stored(completed=["overview"], fabHintSeen=False), puts)
        _boot(page, base_url)

        hint = page.locator("[data-testid='tour-help-hint']")
        expect(hint).to_be_visible()
        # Explaining the button is the whole job — it must not start the tour.
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)
        assert _saved_matching(page, puts, lambda st: st.get("fabHintSeen") is True)

        page.locator("[data-testid='tour-help-hint-dismiss']").click()
        expect(hint).to_have_count(0)

    def test_it_offers_to_start_the_broadest_unseen_tour(self, page: Page, base_url: str):
        """On a configured workspace the overview never auto-starts, so this callout is
        the only walkthrough anyone is ever offered. It has to open the lap of the app,
        not the one screen they happen to be standing on: Configuration Overview is the
        narrower unseen tour here, and starting that instead would explain a corner to
        someone who has not been shown the tabs."""
        _route_tour_state(page, _stored(completed=["overview", "watch"], fabHintSeen=False))
        _boot(page, base_url)
        page.wait_for_timeout(600)
        # Nothing unseen on the landing page, so the callout has not fired yet.
        expect(page.locator("[data-testid='tour-help-hint']")).to_have_count(0)

        # Configuration's own screen tour is unseen, so arriving here fires the callout.
        page.locator(".main-tab-button[data-tab='tab-configuration']").click()

        start = page.locator("[data-testid='tour-help-hint-start']")
        expect(start).to_be_visible()
        start.click()

        expect(page.locator("[data-testid='tour-card']")).to_be_visible(timeout=_STEP_TIMEOUT_MS)
        expect(_progress(page)).to_have_text("Configure · 1 of 7")

    def test_a_workspace_that_has_seen_it_is_not_told_again(self, page: Page, base_url: str):
        _route_tour_state(page, _stored(completed=["overview"], fabHintSeen=True))
        _boot(page, base_url)
        page.wait_for_timeout(1200)

        expect(page.locator("[data-testid='tour-help-hint']")).to_have_count(0)
        # …and the button is still marked, which is what the callout was about.
        expect(page.locator("[data-testid='tour-help-fab']")).to_have_class(re.compile(r"tour-help-fab--unseen"))

    def test_a_page_with_nothing_unseen_does_not_trigger_it(self, page: Page, base_url: str):
        """It is tied to the highlight, not merely to having a ?."""
        every = ["overview", "watch", "configuration", "labels", "sources", "model-shapes", "watch-rules"]
        _route_tour_state(page, _stored(completed=every, fabHintSeen=False))
        _boot(page, base_url)
        page.wait_for_timeout(1200)

        expect(page.locator("[data-testid='tour-help-fab']")).to_be_visible()
        expect(page.locator("[data-testid='tour-help-hint']")).to_have_count(0)


class TestEveryConfigurationScreenHasATour:
    def test_each_rail_screen_offers_its_own(self, page: Page, base_url: str):
        """Landing on Training must offer Training, not a walk through all of Configure."""
        screens = {
            "tab-config-overview": "config-overview",
            "tab-tag-management": "labels",
            "tab-dataset-management": "datasets",
            "tab-source-management": "sources",
            "tab-model-management": "model-shapes",
            "tab-modelstack-management": "model-stacks",
            "tab-training-management": "training",
            "tab-filterchain-management": "filter-chains",
            "tab-collector-management": "collectors",
            "tab-task-management": "tasks",
            "tab-watcher-management": "watch-rules",
        }
        _route_tour_state(page, _stored(completed=["overview"]))
        _boot(page, base_url)
        page.wait_for_timeout(800)
        page.locator(".main-tab-button[data-tab='tab-configuration']").click()
        fab = page.locator("[data-testid='tour-help-fab']")

        for tab, tour_id in screens.items():
            page.locator(f"#tab-configuration [data-tab='{tab}']").click()
            page.wait_for_timeout(300)
            fab.click()
            expect(page.locator(f"[data-testid='tour-help-run-{tour_id}']")).to_be_visible()
            fab.click()

    def test_a_screen_tour_starts_on_that_screen_without_the_configure_walk(self, page: Page, base_url: str):
        _route_tour_state(page, _stored(completed=["overview"]))
        _boot(page, base_url)
        page.wait_for_timeout(800)
        page.locator(".main-tab-button[data-tab='tab-configuration']").click()
        page.locator("#tab-configuration [data-tab='tab-training-management']").click()
        page.wait_for_timeout(300)

        page.locator("[data-testid='tour-help-fab']").click()
        page.locator("[data-testid='tour-help-run-training']").click()

        expect(_progress(page)).to_have_text("Training · 1 of 3")


class TestTheTourPointsOutKeyboardShortcuts:
    def test_the_overview_stops_at_settings_and_branches_to_the_shortcuts(self, page: Page, base_url: str):
        _route_tour_state(page, _stored(fabHintSeen=True))
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        _advance(page, 5)

        expect(_progress(page)).to_have_text("Getting started · 6 of 7")
        page.locator("[data-testid='tour-branch-0']").click()

        expect(_progress(page)).to_have_text("Settings · 1 of 3")
        expect(page.locator("#tab-settings-hotkeys")).to_have_class(re.compile(r"\bactive\b"))
        expect(page.locator("[data-testid='tour-body']")).to_contain_text("keyboard")

    def test_it_says_labels_join_the_shortcut_list_once_they_exist(self, page: Page, base_url: str):
        """A user looking at a short list on a fresh install needs telling why."""
        _route_tour_state(page, _stored(completed=["overview"]))
        _boot(page, base_url)
        page.wait_for_timeout(800)
        page.locator(".main-tab-button[data-tab='tab-settings']").click()
        page.wait_for_timeout(300)
        page.locator("[data-testid='tour-help-fab']").click()
        page.locator("[data-testid='tour-help-run-settings']").click()

        expect(_progress(page)).to_have_text("Settings · 1 of 3")
        _advance(page)
        expect(page.locator("[data-testid='tour-body']")).to_contain_text("labels exist")
