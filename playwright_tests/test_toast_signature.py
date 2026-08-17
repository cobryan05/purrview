"""`toast(message, duration, type)` tolerates a type in the duration slot.

That argument order is easy to get wrong and ~7 call sites did, writing
`toast('Failed to save', 'error')`. It failed twice over, silently:

* `type` kept its default, so error toasts rendered neutral grey rather than
  red — the message said "failed", the colour said "fine";
* `setTimeout(fn, 'error')` coerces to NaN, which the spec clamps to 0, so the
  toast appeared and began disappearing in the same frame. An error message
  that flashes for one frame is worse than one that never shows, because the
  user sees *something* happen and cannot read it.

The call sites are fixed, but the signature stays forgiving so the next one
does not reintroduce the same silent failure.
"""

import re

import pytest
from playwright.sync_api import Page

_IMPORT = "const { toast } = await import('/app-static/js/ui/utils/logging.js');"

_CLEANUP = "document.getElementById('toast-container')?.remove();"


def _show(page: Page, args_js: str) -> dict:
    """Call toast(...) and report back what it actually rendered."""
    return page.evaluate(f"""async () => {{
            {_CLEANUP}
            {_IMPORT}
            toast({args_js});
            const el = document.querySelector('#toast-container > div');
            const out = {{
                text: el?.textContent ?? null,
                background: el?.style.background ?? null,
            }};
            {_CLEANUP}
            return out;
        }}""")


# Backgrounds are asserted as "not the neutral default" / "same as the explicit
# form" rather than as literal hex, so restyling a toast doesn't fail these.
_NEUTRAL = "rgb(85, 85, 85)"


class TestTypeInTheDurationSlot:
    def test_type_as_second_arg_is_understood(self, shared_page: Page):
        misordered = _show(shared_page, "'Failed to save', 'error'")
        correct = _show(shared_page, "'Failed to save', 5000, 'error'")

        assert misordered["background"] == correct["background"]
        assert misordered["background"] != _NEUTRAL

    @pytest.mark.parametrize("kind", ["error", "warning", "success", "info"])
    def test_every_kind_is_accepted_in_either_position(self, shared_page: Page, kind: str):
        a = _show(shared_page, f"'msg', '{kind}'")
        b = _show(shared_page, f"'msg', 5000, '{kind}'")

        assert a["background"] == b["background"]

    def test_the_message_is_still_the_message(self, shared_page: Page):
        assert _show(shared_page, "'Select at least one tag', 'error'")["text"] == "Select at least one tag"

    def test_an_explicit_duration_still_works(self, shared_page: Page):
        # The forgiving branch must not swallow callers who pass a real duration.
        assert _show(shared_page, "'quick', 2000, 'info'")["background"] == _NEUTRAL


class TestColours:
    def test_error_is_not_the_neutral_default(self, shared_page: Page):
        assert _show(shared_page, "'boom', 5000, 'error'")["background"] != _NEUTRAL

    def test_success_is_distinct_from_error(self, shared_page: Page):
        """Call sites have passed 'success' all along; it had no colour, so
        "Saved" and "Failed to save" were the same shade of grey."""
        ok = _show(shared_page, "'saved', 5000, 'success'")
        bad = _show(shared_page, "'failed', 5000, 'error'")

        assert ok["background"] != bad["background"]
        assert ok["background"] != _NEUTRAL

    def test_info_stays_neutral(self, shared_page: Page):
        assert _show(shared_page, "'fyi', 5000, 'info'")["background"] == _NEUTRAL


class TestDurationIsSane:
    def test_a_bad_duration_does_not_become_an_instant_dismissal(self, shared_page: Page):
        """The original bug: NaN clamps to 0, so the toast vanished immediately.
        Anything unusable must fall back to a readable default."""
        remaining = shared_page.evaluate(f"""async () => {{
                {_CLEANUP}
                {_IMPORT}
                const delays = [];
                const realSetTimeout = window.setTimeout;
                window.setTimeout = (fn, ms, ...rest) => {{ delays.push(ms); return realSetTimeout(fn, ms, ...rest); }};
                toast('boom', 'error');
                toast('boom', NaN, 'error');
                toast('boom', -1, 'error');
                window.setTimeout = realSetTimeout;
                {_CLEANUP}
                return delays;
            }}""")

        assert all(isinstance(d, (int, float)) and d > 0 for d in remaining), remaining

    def test_an_unrecognised_string_is_not_silently_treated_as_a_type(self, shared_page: Page):
        # A typo'd kind should not quietly pick a colour; it falls back to the
        # default and warns, rather than pretending it understood.
        assert _show(shared_page, "'hm', 'not-a-kind'")["background"] == _NEUTRAL
