"""Source-consistency test: the JS code that removes the .recently-viewed CSS
class must not cut off the animation early.

The correct pattern is to remove the class on the `animationend` event rather
than via a hardcoded setTimeout — that way changes to the CSS duration never
silently re-introduce a mismatch.

This test verifies that:
  1. The animationend listener pattern is used (not a hardcoded setTimeout).
  2. There is no bare `setTimeout(..., <N>)` removal of the class left over.
"""

import re
from pathlib import Path

_CSS = Path("apps/image_tagging_server/static/css/app.css")
_JS = Path("apps/image_tagging_server/static/js/app/review/main.js")


def test_recently_viewed_uses_animationend_not_hardcoded_timeout():
    """Removal of .recently-viewed must use animationend, not a fixed setTimeout."""
    js_text = _JS.read_text()

    # After the fix: animationend listener must be present.
    assert "animationend" in js_text, (
        "review/main.js must remove .recently-viewed via an 'animationend' "
        "event listener so it stays in sync with the CSS duration automatically."
    )

    # Ensure no leftover hardcoded setTimeout for this class remains.
    stale_timeout = re.search(
        r"setTimeout\([^)]*classList\.remove\(['\"]recently-viewed['\"]\)[^)]*,\s*\d+\s*\)",
        js_text,
    )
    assert not stale_timeout, (
        "Found a hardcoded setTimeout that removes .recently-viewed. "
        "Use animationend instead to avoid duration mismatches."
    )


def test_recently_viewed_animation_duration_in_css():
    """The CSS animation must declare a non-zero duration for the test above to be meaningful."""
    css_text = _CSS.read_text()
    match = re.search(r"animation:\s*recently-viewed-flash\s+([\d.]+)s", css_text)
    assert match, "recently-viewed-flash animation rule not found in app.css"
    assert float(match.group(1)) > 0, "Animation duration must be positive"
