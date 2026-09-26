"""Polling helpers for state Playwright's auto-retrying `expect` can't see.

`expect(locator)` retries until the DOM agrees, so DOM assertions are already
load-proof. Assertions about *Python-side* state are not: the common pattern

    page.wait_for_timeout(300)
    assert create_calls, "Expected POST to /api/thing"

is a bet that 300ms is always enough for the click to reach a `page.route`
handler. On a loaded machine (the full suite runs 16-way parallel) it isn't,
and the test fails for reasons that have nothing to do with the code it covers.

`wait_until` is the `expect`-equivalent for those capture lists: poll until the
condition holds, fail with a useful message if it never does. Passing quickly on
the fast path is a bonus; not failing on the slow path is the point.
"""

import os
from contextlib import contextmanager
from typing import Callable, TypeVar, Union

from playwright.sync_api import Page

T = TypeVar("T")

DEFAULT_TIMEOUT_MS = 5000
POLL_INTERVAL_MS = 50

# Near-miss tracing. A `wait_until` that succeeds at 4.9s of a 5s ceiling *passed*, but
# it was one scheduling hiccup away from failing — it is a flake that has not happened
# yet, and it is invisible in a green run. Set FLAKE_TRACE to a file path to record
# every wait that consumed more than FLAKE_TRACE_RATIO of its timeout:
#
#   docker exec -e FLAKE_TRACE=/project/near-misses.txt -w /project … pytest playwright_tests/
#
# Off unless the env var is set, so it costs one `os.environ` lookup per wait otherwise.
_TRACE_PATH = os.environ.get("FLAKE_TRACE")
_TRACE_RATIO = float(os.environ.get("FLAKE_TRACE_RATIO", "0.5"))


def _record_near_miss(waited: int, timeout: int) -> None:
    """Append a line for any wait that ate a significant fraction of its budget."""
    if not _TRACE_PATH or waited < timeout * _TRACE_RATIO:
        return
    test_id = os.environ.get("PYTEST_CURRENT_TEST", "<unknown>").split(" (")[0]
    try:
        with open(_TRACE_PATH, "a") as fh:
            fh.write(f"{waited}\t{timeout}\t{test_id}\n")
    except OSError:
        pass  # tracing must never fail a run


def wait_until(
    page: Page,
    condition: Callable[[], T],
    message: Union[str, Callable[[], str]] = "",
    timeout: int = DEFAULT_TIMEOUT_MS,
) -> T:
    """Poll `condition` until it returns a truthy value, then return that value.

    Polls via `page.wait_for_timeout` rather than `time.sleep`: the sync
    Playwright API runs route handlers on this same thread, so sleeping here
    would also stall the very handlers we're waiting on.

    Raises AssertionError with `message` if the condition never holds. Pass a
    callable to describe the *final* state — a plain f-string is interpolated
    before the first poll, so it would report the state we started from.
    """
    waited = 0
    while True:
        value = condition()
        if value:
            _record_near_miss(waited, timeout)
            return value
        if waited >= timeout:
            text = message() if callable(message) else message
            raise AssertionError(text or f"Condition never became true within {timeout}ms")
        page.wait_for_timeout(POLL_INTERVAL_MS)
        waited += POLL_INTERVAL_MS


@contextmanager
def issues_a_request(page: Page, captured: list, what: str = "request"):
    """Barrier: the block must cause at least one *new* entry in `captured`.

    The trap this exists for is a wait that is already satisfied. `wait_until(page,
    lambda: calls)` looks like a barrier but returns instantly whenever the page has
    already made one of these requests on open — so the assertion that follows reads
    the *initial* payload, not the one the interaction produced. It passes only
    because the fetch normally beats the next Python statement, and it goes on
    passing when the interaction is broken outright.

    Keying on growth from a baseline taken *before* the block is what makes the
    following assertion about the right request, and falsifiable.

    Note this brackets the interaction, so it cannot be bolted onto an assertion
    after the fact — the baseline has to be taken first:

        with issues_a_request(page, calls, "a review search"):
            page.locator("...").check()
        assert calls[-1]["terms"] == ...
    """
    before = len(captured)
    yield
    wait_until(
        page,
        lambda: len(captured) > before,
        lambda: f"the interaction issued no new {what} (still {len(captured)} captured)",
    )


def requests_settle(page: Page, captured: list, quiet_ms: int = 250, timeout: int = DEFAULT_TIMEOUT_MS) -> list:
    """Wait until `captured` has stopped growing for `quiet_ms`, then return it.

    The companion to `issues_a_request`, for use after *setup* that fires fetches
    nothing on screen is gated on. That barrier keys on growth from a baseline taken
    when the block is entered, so a setup request still in flight at that moment is
    captured afterwards and becomes `captured[-1]` — the following assertion then
    reads the setup's payload instead of the interaction's, and fails claiming the
    interaction did nothing. Draining setup first is what makes the baseline honest.

    Unlike `wait_until` this cannot fail: no further request is the expected outcome.
    """
    waited = 0
    seen = len(captured)
    quiet = 0
    while quiet < quiet_ms and waited < timeout:
        page.wait_for_timeout(POLL_INTERVAL_MS)
        waited += POLL_INTERVAL_MS
        if len(captured) != seen:
            seen = len(captured)
            quiet = 0
        else:
            quiet += POLL_INTERVAL_MS
    _record_near_miss(waited, timeout)
    return captured
