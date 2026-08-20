"""UTC normalisation for datetimes crossing a boundary.

Naive datetimes are the problem this module exists for. Python compares and
formats them happily, so a lost `tzinfo` never raises — it just makes the value
mean "these digits, zone unknown", and the error only becomes visible much later
(a timestamp rendered hours off in the browser, a comparison against an aware
`now()` raising `TypeError`, a clip cut from the wrong second).

Three boundaries drop the zone, and every one of them had grown its own inline
`if x.tzinfo is None: x = x.replace(tzinfo=timezone.utc)`:

* **SQLite**, which has no offset-bearing storage type — handled once and for
  all by `UtcDateTime` in `apps/helpers/db/types.py`, which calls in here.
* **`datetime.fromisoformat`** on strings from MediaMTX, sidecar JSON, and
  query params, which are UTC by convention but not always by spelling.
* **Generated API clients**, whose parsing depends on what the server spelled.

The convention everywhere in this codebase is that a naive datetime is UTC —
every writer uses `datetime.now(timezone.utc)`. These helpers make that reading
explicit at the point the value arrives, rather than leaving each caller to
remember.
"""

from datetime import datetime, timezone
from typing import Optional

__all__ = ["ensure_utc", "parse_iso_utc"]


def ensure_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Return `value` as a timezone-aware UTC datetime.

    A naive input is *stamped* as UTC (its digits are already UTC by this
    codebase's convention); an aware input in another zone is *converted*. `None`
    passes through so callers can stay branch-free on optional columns.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_iso_utc(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string into an aware UTC datetime, or `None`.

    Accepts the `Z` suffix, which `datetime.fromisoformat` rejects before Python
    3.11 and which several of our upstreams emit. Returns `None` rather than
    raising for absent or unparseable input: every caller here treats a bad
    timestamp as a missing one.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return ensure_utc(parsed)
