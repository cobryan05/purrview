"""
The global busy indicator (`ui/utils/busyIndicator.js`).

What is worth pinning here is not that the chip can be shown — it is the
conditionals that decide whether it *should* be, each of which is a real
regression if it inverts:

  - an operation that finishes promptly never shows it at all (otherwise every
    navigation flashes, and a signal that fires constantly stops being read);
  - one that outlives the delay does show it;
  - two overlapping operations are refcounted, so the first to finish does not
    clear the indicator the second still needs;
  - the label names the newest operation, and falls back to one still running
    rather than going blank.

The timings are driven from the test rather than waited on: an operation is held
open explicitly, and the "fast" case begins and ends inside one synchronous
block, so none of these assertions depend on how loaded the box is.
"""

import pytest
from playwright.sync_api import Page, expect

BUSY = "[data-testid=busy-indicator]"

# Comfortably past SHOW_DELAY_MS (180ms) and MIN_VISIBLE_MS (350ms) in busyIndicator.js.
PAST_SHOW_DELAY_MS = 450


@pytest.fixture(autouse=True)
def _reset_indicator(shared_page: Page):
    """The chip is body-appended and `position: fixed` via a class, so the autouse
    DOM cleanup (which matches an inline style) does not reach it."""
    yield
    shared_page.evaluate("""async () => {
            const m = await import('/app-static/js/ui/utils/busyIndicator.js');
            m._resetBusyIndicator();
        }""")


def _begin(page: Page, label: str, key: str) -> None:
    """Start a busy span and keep its end function on `window` under `key`."""
    page.evaluate(
        """async ([label, key]) => {
            const m = await import('/app-static/js/ui/utils/busyIndicator.js');
            window[key] = m.beginBusy(label);
        }""",
        [label, key],
    )


def _end(page: Page, key: str) -> None:
    page.evaluate("key => { window[key](); delete window[key]; }", key)


class TestBusyIndicator:
    def test_operation_outliving_the_delay_is_announced(self, shared_page: Page):
        _begin(shared_page, "Running inference…", "_busyA")
        try:
            expect(shared_page.locator(BUSY)).to_be_visible()
            expect(shared_page.locator(BUSY)).to_contain_text("Running inference…")
        finally:
            _end(shared_page, "_busyA")

    def test_prompt_operation_is_never_announced(self, shared_page: Page):
        """The whole span runs in one synchronous block, so the show must never fire.

        Asserted as "the element was never built", not "it ended up hidden": a chip
        that flashed and then hid is also hidden by the time anything can look at
        it, so only its absence distinguishes deferring the show from doing it
        immediately — and showing immediately is precisely the flash-on-every-
        navigation this delay exists to prevent.
        """
        shared_page.evaluate("""async () => {
                const m = await import('/app-static/js/ui/utils/busyIndicator.js');
                m.beginBusy('Loading image…')();
            }""")
        shared_page.wait_for_timeout(PAST_SHOW_DELAY_MS)
        expect(shared_page.locator(BUSY)).to_have_count(0)

    def test_overlapping_operations_are_refcounted(self, shared_page: Page):
        _begin(shared_page, "Loading image…", "_busyA")
        _begin(shared_page, "Running inference…", "_busyB")
        try:
            expect(shared_page.locator(BUSY)).to_be_visible()
            # The first to finish must not clear the indicator the second needs.
            _end(shared_page, "_busyA")
            shared_page.wait_for_timeout(PAST_SHOW_DELAY_MS)
            expect(shared_page.locator(BUSY)).to_be_visible()
        except BaseException:
            shared_page.evaluate("() => { window._busyA?.(); }")
            raise
        _end(shared_page, "_busyB")
        expect(shared_page.locator(BUSY)).to_be_hidden()

    def test_label_names_the_most_recent_operation(self, shared_page: Page):
        """A navigation started while inference runs should say so: the newest
        operation is the one the user just asked for."""
        _begin(shared_page, "Running inference…", "_busyA")
        try:
            expect(shared_page.locator(BUSY)).to_be_visible()
            _begin(shared_page, "Loading image…", "_busyB")
            try:
                expect(shared_page.locator(BUSY)).to_contain_text("Loading image…")
            finally:
                _end(shared_page, "_busyB")
            # Falling back to the one still running, rather than going blank.
            expect(shared_page.locator(BUSY)).to_contain_text("Running inference…")
        finally:
            _end(shared_page, "_busyA")
