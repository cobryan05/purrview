"""Tests for ui/collectorDiagnosticsModal.js — the collector tuning modal.

The modal renders `collector_status_schema()` against the payload
`get_collector_diagnostics()` returns, and those two are written in different
files by different code paths. The shipped bug these tests pin was exactly that
seam: the schema's `stat_grid` is `{id: "stats"}` and the widget renderer looks
its data up by widget id, but the modal spread the stats dict *flat* into the
data source, so `dataSource["stats"]` was undefined and all 29 fields rendered
their "—" placeholder on a collector that was actively collecting.

`window.fetch` is stubbed, so nothing here needs a running watcher.
"""

import pytest
from playwright.sync_api import Page

# The real payload shape: status_stats() returns (stats, calibration) where
# `stats` is flat scalars and `calibration`'s own keys are already widget ids.
_SETUP = """
    const { openCollectorDiagnosticsModal } =
        await import('/app-static/js/ui/collectorDiagnosticsModal.js');

    const _realFetch = window.fetch;
    window.fetch = async (url, opts) => {
        let payload = { status: 'success' };
        const u = String(url);
        if (u.includes('/api/training-collectors/status-schema')) {
            payload = { status: 'success', statusSchema: window.__schema };
        } else if (u.includes('/diagnostics')) {
            payload = { status: 'success', diagnostics: window.__diagnostics };
        } else if (u.includes('/api/training-collectors/') && u.includes('/review-actions')) {
            payload = { status: 'success', actions: [] };
        }
        return new Response(JSON.stringify(payload), {
            status: 200, headers: { 'Content-Type': 'application/json' },
        });
    };
    window.__restoreFetch = () => { window.fetch = _realFetch; };
"""

_TEARDOWN = """
    window.__restoreFetch();
    delete window.__schema; delete window.__diagnostics;
    document.querySelectorAll('[data-testid="modal-shell-overlay"]').forEach(d => d.remove());
    document.querySelectorAll('[data-testid="collector-diag-dialog"]').forEach(d => d.remove());
"""

_SCHEMA = """[
    { id: 'stats', type: 'stat_grid', source: 'status', label: 'Collection stats',
      fields: { offered: 'Offered', saved: 'Saved', dedup_checked: 'Dedup checked' } },
    { id: 'dedup_hint', type: 'text', source: 'status', label: 'How to read this' }
]"""


def _cells(page: Page, diagnostics: str, schema: str = _SCHEMA) -> dict:
    """Open the modal against a scripted payload; read back {label: value}."""
    return page.evaluate(f"""async () => {{
            {_SETUP}
            window.__schema = {schema};
            window.__diagnostics = {diagnostics};
            openCollectorDiagnosticsModal({{
                collectorUuid: 'coll-1', collectorName: 'Cats', taskUuid: 'task-1',
            }});
            // One poll runs on open; give it room to fetch schema + diagnostics.
            await new Promise(r => setTimeout(r, 300));
            const host = document.querySelector('[data-testid="collector-diag-widgets"]');
            const out = {{}};
            for (const cell of host?.querySelectorAll('div > div > div') ?? []) {{
                const value = cell.children[0]?.textContent;
                const label = cell.children[1]?.textContent;
                if (label != null && value != null) out[label] = value;
            }}
            {_TEARDOWN}
            return out;
        }}""")


class TestStatGridData:
    """The regression: a collector with real counters must not render "—"."""

    _LIVE = """{
        stats: { offered: 42, saved: 7, dedup_checked: 12 },
        calibration: { dedup_hint: 'read it like this' },
        notices: [],
    }"""

    def test_counters_reach_the_grid(self, shared_page: Page):
        cells = _cells(shared_page, self._LIVE)
        assert cells.get("Offered") == "42"
        assert cells.get("Saved") == "7"
        assert cells.get("Dedup checked") == "12"

    def test_a_field_the_collector_never_sent_stays_a_dash(self, shared_page: Page):
        """The grid lists every field status_stats() *can* populate. A disabled
        feature omits its key, and "—" is the honest answer — a 0 would read as
        "dedup ran and found nothing"."""
        cells = _cells(
            shared_page,
            "{ stats: { offered: 42, saved: 7 }, calibration: {}, notices: [] }",
        )
        assert cells.get("Offered") == "42"
        assert cells.get("Dedup checked") == "—"

    def test_zero_renders_as_zero_not_a_dash(self, shared_page: Page):
        """A real 0 and a missing key are different answers, and `??` is what
        keeps them apart — `||` here would collapse both to "—"."""
        cells = _cells(
            shared_page,
            "{ stats: { offered: 0, saved: 0, dedup_checked: 0 }, calibration: {}, notices: [] }",
        )
        assert cells.get("Offered") == "0"
        assert cells.get("Saved") == "0"
