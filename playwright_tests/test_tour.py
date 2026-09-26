"""Guided tours (ui/tour/).

The mocked workspace has no labels and no sources, so it is exactly the fresh install
the getting-started tour exists for. `conftest` sets the app's per-browser kill switch
so the rest of the suite never boots under a spotlight; `_boot` here clears it and
routes `tour_state` to say what a real install would.

What matters here is conditional: the tour appears on an empty, unseen workspace and
does NOT appear once it has been declined or once the workspace has something in it;
a step gets its anchor on screen from another tab; and the walkthrough step advances
itself when the user does the thing rather than when they press Next.
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


def _saved_state(page: Page, puts: list, timeout_ms: int = 10000) -> dict:
    """The state the app wrote back, waited for rather than slept for.

    Every write in `ui/tour/state.js` reads the setting before it PUTs, so the write
    is a second round-trip that nothing on screen is gated on: the card disappears
    first. A fixed delay here is a guess about how fast the machine is.
    """
    deadline = time.monotonic() + timeout_ms / 1000
    while not puts:
        assert time.monotonic() < deadline, f"no tour_state PUT arrived within {timeout_ms}ms"
        page.wait_for_timeout(50)
    return json.loads(puts[-1])


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


class TestTheTourOffersItselfOnlyWhenWanted:
    def test_a_fresh_empty_workspace_gets_the_tour(self, page: Page, base_url: str):
        _route_tour_state(page, None)
        _boot(page, base_url)

        card = page.locator("[data-testid='tour-card']")
        expect(card).to_be_visible()
        expect(page.locator("[data-testid='tour-progress']")).to_have_text("1 of 10")
        # Step one lives on Configuration: the tour must have taken the user there
        # from the default Watch tab, not pointed at a tab that is not on screen.
        expect(page.locator("#tab-configuration")).to_be_visible()

    def test_a_workspace_that_declined_is_left_alone(self, page: Page, base_url: str):
        _route_tour_state(page, json.dumps({"neverShow": False, "dismissed": ["getting-started"], "completed": []}))
        _boot(page, base_url)

        page.wait_for_timeout(1200)
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)
        # And it stayed on the landing tab, since nothing navigated.
        expect(page.locator("#tab-nvr")).to_be_visible()

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


class TestSkippingAndFinishingArePersisted:
    def test_skip_records_a_dismissal(self, page: Page, base_url: str):
        puts: list = []
        _route_tour_state(page, None, puts)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()

        page.locator("[data-testid='tour-skip']").click()

        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)
        expect(page.locator("[data-testid='tour-spotlight']")).to_have_count(0)
        saved = _saved_state(page, puts)
        assert "getting-started" in saved["dismissed"]
        assert "getting-started" not in saved["completed"]

    def test_the_opt_out_box_switches_tours_off(self, page: Page, base_url: str):
        puts: list = []
        _route_tour_state(page, None, puts)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()

        page.locator("[data-testid='tour-never-show']").check()

        assert _saved_state(page, puts)["neverShow"] is True

    def test_done_on_the_last_step_records_completion(self, page: Page, base_url: str):
        puts: list = []
        _route_tour_state(page, None, puts)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()

        next_btn = page.locator("[data-testid='tour-next']")
        for _ in range(9):
            next_btn.click()
            page.wait_for_timeout(150)
        expect(page.locator("[data-testid='tour-progress']")).to_have_text("10 of 10")
        expect(next_btn).to_have_text("Done")

        next_btn.click()

        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)
        saved = _saved_state(page, puts)
        assert "getting-started" in saved["completed"]


class TestStepsReachTheirAnchors:
    def test_a_step_on_another_pane_navigates_there_and_frames_the_control(self, page: Page, base_url: str):
        _route_tour_state(page, None)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        next_btn = page.locator("[data-testid='tour-next']")

        # 1 → 2 → 3 → 4: "Create your first label", whose anchor is a row that only
        # exists once the Labels & Tags pane has mounted.
        for _ in range(3):
            next_btn.click()
            page.wait_for_timeout(150)

        expect(page.locator("[data-testid='tour-title']")).to_have_text("Create your first label")
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
        _route_tour_state(page, None)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        next_btn = page.locator("[data-testid='tour-next']")
        for _ in range(3):
            next_btn.click()
            page.wait_for_timeout(150)
        expect(page.locator("[data-testid='tour-title']")).to_have_text("Create your first label")
        # A step that waits offers to be skipped, not "Next": the point is doing it.
        expect(next_btn).to_have_text("Skip this step")

        page.evaluate("""async () => {
                const { events, EventTypes } = await import('/app-static/js/shared/events/index.js');
                events.publish(EventTypes.LABELS_UPDATED);
            }""")

        expect(page.locator("[data-testid='tour-progress']")).to_have_text("5 of 10")
        expect(page.locator("[data-testid='tour-title']")).to_have_text("Sources are where frames come from")

    def test_escape_skips_and_back_returns(self, page: Page, base_url: str):
        _route_tour_state(page, None)
        _boot(page, base_url)
        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        expect(page.locator("[data-testid='tour-back']")).to_be_hidden()

        page.locator("[data-testid='tour-next']").click()
        expect(page.locator("[data-testid='tour-progress']")).to_have_text("2 of 10")
        page.locator("[data-testid='tour-back']").click()
        expect(page.locator("[data-testid='tour-progress']")).to_have_text("1 of 10")

        page.keyboard.press("Escape")
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)


class TestSettingsCanRunATour:
    def test_run_from_settings_starts_it_even_when_tours_are_off(self, page: Page, base_url: str):
        _route_tour_state(page, json.dumps({"neverShow": True, "dismissed": [], "completed": ["getting-started"]}))
        _boot(page, base_url)
        page.wait_for_timeout(800)
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)

        page.locator(".main-tab-button[data-tab='tab-settings']").click()
        page.locator("#tab-settings [data-tab='tab-settings-interface']").click()
        auto = page.locator("[data-testid='settings-tours-auto']")
        expect(auto).not_to_be_checked()

        page.locator("[data-testid='settings-tours-run']").click()

        expect(page.locator("[data-testid='tour-card']")).to_be_visible()
        expect(page.locator("[data-testid='tour-progress']")).to_have_text("1 of 10")
