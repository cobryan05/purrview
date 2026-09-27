"""The Review screen's own guided tour.

Review carries more unlabelled machinery than any other screen — the verification marks
are on a right-click, the editor on a double-click, and half the toolbar acts on the whole
filter rather than on the page — so it has a tour of its own rather than one step inside
Annotate's.

Driven against mocked review data rather than an empty workspace, because most of what
the tour points at does not exist until there is something to act on: the bulk-apply
controls appear only once cards are selected. An empty screen would skip exactly the steps
most worth checking.

Run with:
    pytest playwright_tests/test_review_tour.py -p no:asyncio -v
"""

import re
import time

from playwright.sync_api import Page, expect

from playwright_tests.test_review_workflow_ui import (
    LABEL_CAT,
    LABEL_DOG,
    PAGE_1_IMAGES,
    _mock_labels,
    _mock_misc_review_endpoints,
    _mock_review_queue,
    _mock_tags,
)
from playwright_tests.test_tour import (
    _STEP_TIMEOUT_MS,
    _boot,
    _progress,
    _route_tour_state,
    _stored,
    _walk_to_end,
)
from playwright_tests.waits import wait_until

#: Every step the tour declares. Asserted as a number rather than a list because the
#: engine renders the count itself, and a step added without a thought about where it
#: belongs in the order should be a visible change here.
REVIEW_STEP_COUNT = 14


def _open_review_with_data(page: Page, base_url: str):
    _mock_labels(page, [LABEL_CAT, LABEL_DOG])
    _mock_tags(page, [])
    _mock_review_queue(page, PAGE_1_IMAGES)
    _mock_misc_review_endpoints(page)
    _route_tour_state(page, _stored(completed=["overview"]))
    _boot(page, base_url)
    page.wait_for_timeout(800)
    page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
    page.locator("#tab-image-tagging .section-tab-button[data-tab='tab-review']").click()
    page.wait_for_selector("#review-root .review-grid")


def _tick_select_all(page: Page):
    """Tick select-all and make sure it stays ticked.

    The grid re-renders as its thumbnails settle and resets the box, so a single `check()`
    placed during that window is silently undone — and the bulk controls this unhides stay
    hidden, which reads as a missing anchor rather than a timing problem.
    """
    cb = page.locator(".review-toolbar__select-all input[type=checkbox]")

    def _ticked():
        if not cb.is_checked():
            cb.click()
        return cb.is_checked()

    wait_until(page, _ticked, "select-all never stayed ticked")


def _start_review_tour(page: Page):
    page.locator("[data-testid='tour-help-fab']").click()
    page.locator("[data-testid='tour-help-run-review']").click()
    expect(page.locator("[data-testid='tour-card']")).to_be_visible()


def _advance_until_title(page: Page, title: str, limit: int = 20):
    next_btn = page.locator("[data-testid='tour-next']")
    for _ in range(limit):
        if page.locator("[data-testid='tour-title']").text_content() == title:
            return
        before = _progress(page).text_content()
        next_btn.click()
        expect(_progress(page)).not_to_have_text(before, timeout=_STEP_TIMEOUT_MS)
    raise AssertionError(f"never reached the {title!r} step")


class TestReviewIsOfferedItsOwnTour:
    def test_standing_on_review_offers_review_not_only_annotate(self, page: Page, base_url: str):
        """The ? reads the *section* tier to answer this.

        Annotate splits into Editor and Review above the Canvas/Files tier, and a page
        lookup that saw only the lower one answered "Canvas" for a user standing on
        Review — so this tour could never be offered however it was registered.
        """
        _open_review_with_data(page, base_url)
        page.locator("[data-testid='tour-help-fab']").click()
        expect(page.locator("[data-testid='tour-help-run-review']")).to_be_visible()

    def test_the_editor_side_still_offers_the_annotate_tour(self, page: Page, base_url: str):
        """The other half of that fix: reading the section tier must not cost the lower
        one its answer. Editor's own screens are still what the ? offers there."""
        _open_review_with_data(page, base_url)
        page.locator("#tab-image-tagging .section-tab-button[data-tab='tab-annotate-workspace']").click()
        page.wait_for_timeout(400)
        page.locator("[data-testid='tour-help-fab']").click()
        expect(page.locator("[data-testid='tour-help-run-annotate']")).to_be_visible()
        expect(page.locator("[data-testid='tour-help-run-review']")).to_have_count(0)


class TestEveryStepReachesAnAnchor:
    def test_the_tour_walks_to_its_last_step(self, page: Page, base_url: str):
        """A step whose anchor never resolves is not an error — the engine gives up after
        four seconds and renders it pointing at nothing. Finishing the walk inside the
        press limit is what catches a broken anchor, because a tour full of those cannot.
        """
        _open_review_with_data(page, base_url)
        _start_review_tour(page)
        expect(_progress(page)).to_have_text(f"Review · 1 of {REVIEW_STEP_COUNT}")

        _walk_to_end(page)
        expect(page.locator("[data-testid='tour-next']")).to_have_text("Done")

    def test_the_bulk_step_frames_the_bulk_button(self, page: Page, base_url: str):
        """One anchor checked properly rather than fifteen checked by their absence.

        These anchors are set in JS on toolbar controls the grid re-renders around, and
        the bulk button is hidden until something is selected — so this pins both that the
        anchor resolves and that the spotlight lands on the control rather than at the top
        of the page, where an unresolved one leaves it.
        """
        _open_review_with_data(page, base_url)
        _tick_select_all(page)
        _start_review_tour(page)
        _advance_until_title(page, "Select many, change them together")

        btn = page.locator("[data-tour='review.bulkApply']")
        expect(btn).to_be_visible()
        spot = page.locator("[data-testid='tour-spotlight']")
        expect(spot).to_be_visible()
        page.wait_for_timeout(600)
        bb, sb = btn.bounding_box(), spot.bounding_box()
        assert abs(sb["x"] - bb["x"]) < 16 and abs(sb["y"] - bb["y"]) < 16, (sb, bb)
        assert abs(sb["width"] - bb["width"]) < 32, (sb, bb)

    def test_the_bulk_step_falls_back_to_the_control_that_summons_the_button(self, page: Page, base_url: str):
        """ "Apply to Selected" lives in a bar that only exists once something is ticked.

        The step still has to happen on an untouched screen — it is how a reader learns
        that bar is there at all — so with no selection it spotlights the Select all
        checkbox instead of being passed over.
        """
        _open_review_with_data(page, base_url)
        _start_review_tour(page)
        _advance_until_title(page, "Select many, change them together")

        expect(page.locator("[data-tour='review.bulkApply']")).to_be_hidden()
        fallback = page.locator("[data-tour='review.select']")
        expect(fallback).to_be_visible()
        spot = page.locator("[data-testid='tour-spotlight']")
        expect(spot).to_be_visible()
        page.wait_for_timeout(600)
        fb, sb = fallback.bounding_box(), spot.bounding_box()
        assert abs(sb["x"] - fb["x"]) < 16 and abs(sb["y"] - fb["y"]) < 16, (sb, fb)

    def test_passing_over_an_unavailable_step_does_not_stall(self, page: Page, base_url: str):
        """A step whose control is simply not there must be decided, not waited out.

        The optional-anchor timeout exists to outlast a pane that is still mounting; when
        the step does not navigate and nothing is settling, the pane is already painted
        and the rest of that wait is dead time with the card unresponsive. Bounded well
        below the 1.5s a single timed-out anchor costs on its own, so the assertion fails
        if the wait comes back.

        In the Images view "Apply to All" is hidden — asserted, not assumed, because the
        press only has something to pass over while that holds.
        """
        _open_review_with_data(page, base_url)
        page.locator("[data-tour='review.viewToggle'] [data-view-key='images']").click()
        page.wait_for_selector("[data-tour='review.viewToggle'] [data-view-key='images'].active")
        _start_review_tour(page)
        _advance_until_title(page, "Select many, change them together")
        expect(page.locator("[data-tour='review.bulkApplyAll']")).to_be_hidden()

        started = time.monotonic()
        before = _progress(page).text_content()
        page.locator("[data-testid='tour-next']").click()
        expect(page.locator("[data-testid='tour-title']")).to_have_text(
            "Let a model pre-fill the whole queue", timeout=_STEP_TIMEOUT_MS
        )
        elapsed = time.monotonic() - started
        assert _progress(page).text_content() != before
        assert elapsed < 1.0, f"passing over an absent anchor took {elapsed:.2f}s"


class TestTheCopyNamesTheUsersOwnKeys:
    def test_a_rebound_key_is_what_the_step_says(self, page: Page, base_url: str):
        """`[` and `]` are the defaults, not facts. A step that prints them is wrong for
        anyone who has rebound the action — which the Hotkeys screen exists to let them
        do — so the copy carries a token the engine resolves against the real binding."""
        # Answered as "never stored", so the app falls back to the localStorage mirror
        # seeded below. The PUT is caught too: the merge of new defaults persists the map
        # on load, and this must not reach the real workspace's settings.
        page.route(
            re.compile(r"/api/settings/hotkey_bindings$"),
            lambda r: (
                r.fulfill(status=200, content_type="application/json", body='{"status":"success"}')
                if r.request.method == "PUT"
                else r.fulfill(status=404, content_type="application/json", body='{"detail":"not found"}')
            ),
        )
        page.add_init_script(
            "try { localStorage.setItem('ww_hotkey_bindings',"
            " JSON.stringify({'n': 'action:prevImage', 'm': 'action:nextImage'})); } catch (_) {}"
        )
        _open_review_with_data(page, base_url)
        _start_review_tour(page)
        _advance_until_title(page, "Double-click to fix it properly")

        body = page.locator("[data-testid='tour-body']").text_content()
        assert "N and M move through the grid" in body, body
        assert "[ and ]" not in body, body
