"""The guided tour that lives inside the event modal (ui/tour/registry.js → EVENT_DETAIL).

Every other tour belongs to a page: the ? in the corner lists it, and the engine can
walk the tabs to reach each step's anchor. This one belongs to a modal, which the
engine cannot open, so it is reached from a ? in the modal's own header instead — and
its anchors only exist while that modal is up.

Three things can break without anyone noticing, and each is asserted here:

  - the modal tour leaking into the page ? menu, where every step would point at an
    anchor that is not in the document;
  - the tour card ending up *under* the modal overlay (`--layer-tour` sits above
    `--layer-modal-max`), which leaves a visible tour nobody can press Next on;
  - closing the modal losing the walk, so the ? restarts from step one instead of
    picking up where the reader left off.

Run with:
    pytest playwright_tests/test_event_detail_tour.py -p no:asyncio -v
"""

import json
import re
import time

from playwright.sync_api import Page, expect

from playwright_tests.test_nvr_events import _mock, _open_events_pane, _overlay, _route_labels
from playwright_tests.test_tour import _STEP_TIMEOUT_MS, _progress, _stored

TOUR_STATE_RE = re.compile(r"/api/settings/tour_state$")


def _route_tour_state_rw(page: Page, initial: str):
    """Serve `tour_state` and remember what the app writes back.

    Unlike `test_tour._route_tour_state`, a PUT here changes what the next GET answers.
    The pause this file is about is only observable through a *later* read: the app
    records the step on close and reads it again when the ? is pressed, so a mock that
    kept serving the original value could never show a resume.
    """
    box = {"value": initial}

    def handle(route):
        if route.request.method == "PUT":
            box["value"] = json.loads(route.request.post_data)["value"]
            route.fulfill(status=200, content_type="application/json", body='{"status":"success"}')
        else:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "value": box["value"]}),
            )

    page.route(TOUR_STATE_RE, handle)
    return box


def _wait_for_recorded_pause(page: Page, box: dict, timeout_ms: int = 5000):
    """Block until the app has written the paused step back.

    The pause is a read-modify-write against one setting, so it lands a round trip after
    the close. Reopening the modal and pressing ? before it does would read the state
    from before the pause and restart the walk — a flake, not a finding.
    """
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        state = json.loads(box["value"])
        if "event-detail" in state.get("progress", {}):
            return state
        assert time.monotonic() < deadline, f"no paused step written within {timeout_ms}ms; saw {box['value']}"
        page.wait_for_timeout(50)


def _open_event_modal(page: Page, base_url: str):
    _mock(page, overlay=_overlay())
    _route_labels(page)
    _open_events_pane(page, base_url)
    page.locator("[data-testid='event-card']").first.click()
    page.wait_for_selector("[data-testid='event-detail-modal']")


def _start_modal_tour(page: Page):
    page.locator("[data-testid='event-detail-tour']").click()
    expect(page.locator("[data-testid='tour-card']")).to_be_visible()


class TestTheModalTourStaysInItsModal:
    def test_the_page_button_offers_the_watch_tour_but_not_this_one(self, page: Page, base_url: str):
        """`pages: []` is what keeps it out — every step here anchors inside the modal,
        so a reader who started it from the page ? would get a tour of nothing."""
        _route_tour_state_rw(page, _stored(completed=["overview"]))
        _mock(page)
        _open_events_pane(page, base_url)

        page.locator("[data-testid='tour-help-fab']").click()
        expect(page.locator("[data-testid='tour-help-run-watch']")).to_be_visible()
        expect(page.locator("[data-testid='tour-help-run-event-detail']")).to_have_count(0)

    def test_the_button_is_in_the_modal_and_starts_at_step_one(self, page: Page, base_url: str):
        _route_tour_state_rw(page, _stored(completed=["overview"]))
        _open_event_modal(page, base_url)
        _start_modal_tour(page)

        expect(_progress(page)).to_have_text(re.compile(r"^Inside an event · 1 of \d+$"))


class TestTheTourIsUsableOverTheModal:
    def test_next_can_actually_be_pressed_through_the_overlay(self, page: Page, base_url: str):
        """The modal overlay covers the viewport, so a tour card at a lower layer would
        render and then swallow every click. Playwright refuses a click on a covered
        element, which is exactly the failure this asserts against."""
        _route_tour_state_rw(page, _stored(completed=["overview"]))
        _open_event_modal(page, base_url)
        _start_modal_tour(page)

        before = _progress(page).text_content()
        page.locator("[data-testid='tour-next']").click()
        expect(_progress(page)).not_to_have_text(before, timeout=_STEP_TIMEOUT_MS)

    def test_the_spotlight_frames_something_inside_the_modal(self, page: Page, base_url: str):
        """The first step points at the media pane. An anchor resolved from the page
        behind the modal would put the spotlight somewhere else entirely."""
        _route_tour_state_rw(page, _stored(completed=["overview"]))
        _open_event_modal(page, base_url)
        _start_modal_tour(page)

        page.wait_for_timeout(600)
        spot = page.locator("[data-testid='tour-spotlight']").bounding_box()
        media = page.locator("[data-tour='eventDetail.media']").bounding_box()
        assert abs(spot["x"] - media["x"]) < 16 and abs(spot["y"] - media["y"]) < 16, (spot, media)


class TestClosingTheModalKeepsThePlace:
    def test_the_tour_goes_with_the_modal_rather_than_hanging_over_the_grid(self, page: Page, base_url: str):
        _route_tour_state_rw(page, _stored(completed=["overview"]))
        _open_event_modal(page, base_url)
        _start_modal_tour(page)

        page.locator("[data-testid='event-detail-close']").click()
        expect(page.locator("[data-testid='event-detail-modal']")).to_have_count(0)
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)

    def test_reopening_resumes_the_step_it_was_left_on(self, page: Page, base_url: str):
        """A pause, not a stop: the reader who closes an event to look at the thing a
        step just described should not have to walk the tour again to get back."""
        box = _route_tour_state_rw(page, _stored(completed=["overview"]))
        _open_event_modal(page, base_url)
        _start_modal_tour(page)

        before = _progress(page).text_content()
        page.locator("[data-testid='tour-next']").click()
        expect(_progress(page)).not_to_have_text(before, timeout=_STEP_TIMEOUT_MS)
        left_on = _progress(page).text_content()

        page.locator("[data-testid='event-detail-close']").click()
        expect(page.locator("[data-testid='tour-card']")).to_have_count(0)
        _wait_for_recorded_pause(page, box)

        page.locator("[data-testid='event-card']").first.click()
        page.wait_for_selector("[data-testid='event-detail-modal']")
        _start_modal_tour(page)

        expect(_progress(page)).to_have_text(left_on, timeout=_STEP_TIMEOUT_MS)
