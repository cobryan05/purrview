"""Shared shape for the Operations tab's Service Status panel.

Every service's ``GET /status`` may return two optional keys on top of
``service``/``status``/``notices``:

``summary``
    The one-line string shown on the collapsed row ("running: 2",
    "3 streams · 1 recording"). Previously the frontend derived this from
    whichever well-known keys it recognised (``tasks``, ``models``, …), which
    meant every new counter needed a matching edit in ``serviceStatus.js``. The
    service owns its own wording now; the frontend still falls back to the
    derived text when ``summary`` is absent, so a service that hasn't opted in
    renders exactly as before.

``details``
    A list of *sections* — see :func:`section` — rendered as an expandable
    sub-panel under the row. The shape is deliberately generic (title + label/
    value rows) so the frontend needs no per-service knowledge: a service adds
    a section and it renders.

Everything here must stay cheap to compute. ``/status`` is polled by the
notices SSE loop every few seconds, not just when the panel is refreshed, so a
detail builder that touches the DB, the filesystem, or the MediaMTX API would
put that cost on every poll. In-memory reads only.
"""

from typing import Any, Iterable, Optional


def row(label: str, value: Any) -> dict:
    """One ``label: value`` line inside a section. *value* is stringified."""
    return {"label": str(label), "value": "" if value is None else str(value)}


def section(title: str, rows: Iterable[dict], empty: Optional[str] = None) -> dict:
    """A titled group of rows.

    Args:
        title: Heading for the group, e.g. ``"Loaded models (2)"``.
        rows: Iterable of :func:`row` dicts.
        empty: Placeholder shown when *rows* is empty. A section with no rows
            and no *empty* text is still returned — the caller decides whether
            an empty group is worth showing — so pass *empty* whenever "nothing
            here" is itself the useful answer.
    """
    out: dict = {"title": title, "rows": list(rows)}
    if empty is not None:
        out["empty"] = empty
    return out


def fmt_duration(seconds: Optional[float]) -> str:
    """Compact human duration for status rows: ``42s``, ``7m 12s``, ``2h 3m``."""
    if seconds is None:
        return "—"
    total = int(max(seconds, 0))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total // 3600}h {(total % 3600) // 60:02d}m"
