"""Regression test for the silently-swallowed "Mark label absent" refusal.

POST /api/images/labels/set answers `status: success, skipped: true` when it
deliberately declines the write — marking a label ABSENT on an image that has a
verified bbox carrying that label. Because that is a *success* response nothing
throws, so the Review tab's context menu used to just close: no toast, no error,
and the label silently stayed as it was. ui/labelList.js already surfaced the
same server contract with an info toast; the context-menu path did not.

The fix moved the handling into contextMenu/providers/imageProvider.js, which is
the single consumer of every `setLabelPresence` adapter, so the multi-select loop
is covered by the same code. These tests drive the real menu DOM (open menu →
"Mark label absent" flyout → pick a label) rather than calling the helper
directly, so an adapter that goes back to swallowing its result still fails.

Run with:
    pytest playwright_tests/test_context_menu_label_skipped.py -p no:asyncio -v
"""

import json

import pytest
from playwright.sync_api import Page

LABEL_UUID = "44444444-5555-6666-7777-000000000042"

_LABELS_PAYLOAD = {
    "status": "success",
    "labels": [
        {
            "uuid": LABEL_UUID,
            "name": "verified-conflict-label",
            "color": "#ff8800",
            "labelType": "standard",
            # Only image-scope labels are listed in the Mark-label flyout.
            "scope": "image",
            "parentUuid": None,
        },
    ],
}

# Opens the 'image' context menu over two fake adapters, clicks the given
# top-level item to open its flyout, then picks the single label row.
# Returns the toasts that resulted plus what each adapter was asked to do.
_DRIVE_MENU = """
async ({ itemLabel, adapterCount, skippedFlags }) => {
    const { openContextMenuForObjects } = await import('/app-static/js/ui/contextMenu/registry.js');
    const { clearLabelsCache } = await import('/app-static/js/shared/api/labels.js');
    await import('/app-static/js/ui/contextMenu/providers/imageProvider.js');

    clearLabelsCache();
    document.getElementById('toast-container')?.remove();

    const calls = [];
    const objects = [];
    for (let i = 0; i < adapterCount; i++) {
        objects.push({
            adapters: {
                image: {
                    getPath: () => `img-${i}.png`,
                    getUuid: () => `uuid-${i}`,
                    openOnCanvas: () => {},
                    getTagUuids: () => [],
                    toggleTag: async () => {},
                    setTagStatus: async () => {},
                    async setLabelPresence(uuid, status) {
                        calls.push({ i, uuid, status });
                        // Mirrors shared/api/imageLabels.js's return shape.
                        return { skipped: skippedFlags[i] };
                    },
                },
            },
        });
    }

    await openContextMenuForObjects(objects, { x: 50, y: 50 });

    const rows = [...document.querySelectorAll('.ctx-menu-item')];
    const target = rows.find(r => r.textContent.includes(itemLabel));
    if (!target) return { error: `no menu item matching ${itemLabel}` };
    target.click();                                  // opens the flyout

    // The scrolling wrapper around the rows has the same textContent as its
    // single row, so take the innermost match — the wrapper carries no click
    // handler and a mousedown on it would silently do nothing.
    const findLabelRow = () => [...document.querySelectorAll('.ctx-menu')]
        .flatMap(f => [...f.querySelectorAll('div')])
        .filter(d => d.textContent.trim() === 'verified-conflict-label')
        .filter(d => !d.querySelector('div'))
        .pop();

    // Poll rather than sleep: asyncPanel resolves fetchLabels off a promise, and
    // a fixed wait is a race that only loses when the box is busy — which is
    // exactly when the whole suite runs.
    const waitFor = async (fn, ms) => {
        const deadline = Date.now() + ms;
        for (;;) {
            const v = fn();
            if (v) return v;
            if (Date.now() > deadline) return null;
            await new Promise(r => setTimeout(r, 20));
        }
    };

    const labelRow = await waitFor(findLabelRow, 10000);
    if (!labelRow) return { error: 'label row not found in flyout' };
    labelRow.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));

    // Every adapter must have been asked before the toast decision is readable.
    await waitFor(() => calls.length === adapterCount, 10000);
    // The toast (if any) is raised by the same handler, immediately after the
    // last adapter resolves. Absence is the assertion in one test, so this can
    // only be a settle — but it is now the sole fixed wait, and nothing has to
    // happen inside it for the accepted-write case to be correct.
    await waitFor(() => document.getElementById('toast-container')?.children.length, 500);

    const container = document.getElementById('toast-container');
    const toasts = container ? [...container.children].map(c => c.textContent) : [];

    document.querySelectorAll('.ctx-menu').forEach(m => m.remove());
    return { toasts, calls };
}
"""


@pytest.fixture
def menu_page(page: Page, base_url: str) -> Page:
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.route(
        "**/api/labels",
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(_LABELS_PAYLOAD)),
    )
    return page


class TestMarkLabelAbsentRefusal:
    def test_skipped_response_surfaces_a_toast(self, menu_page: Page):
        """The whole point: a refused write must not look like it succeeded."""
        result = menu_page.evaluate(
            _DRIVE_MENU,
            {"itemLabel": "Mark label absent", "adapterCount": 1, "skippedFlags": [True]},
        )
        assert not result.get("error"), result.get("error")
        assert result["calls"] == [{"i": 0, "uuid": LABEL_UUID, "status": "absent"}]
        assert any(
            "left unchanged" in t for t in result["toasts"]
        ), f"a skipped write must tell the user nothing changed; toasts={result['toasts']}"
        assert any(
            "verified bbox" in t for t in result["toasts"]
        ), "the toast must say why, so the user knows retrying won't help"

    def test_accepted_write_is_silent(self, menu_page: Page):
        """No toast on the ordinary path — the grid re-render is the feedback."""
        result = menu_page.evaluate(
            _DRIVE_MENU,
            {"itemLabel": "Mark label absent", "adapterCount": 1, "skippedFlags": [False]},
        )
        assert not result.get("error"), result.get("error")
        assert result["calls"] == [{"i": 0, "uuid": LABEL_UUID, "status": "absent"}]
        assert result["toasts"] == []

    def test_multi_select_counts_only_the_refused_images(self, menu_page: Page):
        """Selecting three images where two are blocked must report two, not
        three and not one — the multi-select loop is where a per-adapter check
        is easiest to get wrong."""
        result = menu_page.evaluate(
            _DRIVE_MENU,
            {
                "itemLabel": "Mark label absent",
                "adapterCount": 3,
                "skippedFlags": [True, False, True],
            },
        )
        assert not result.get("error"), result.get("error")
        assert [c["i"] for c in result["calls"]] == [0, 1, 2], "every selected image is still written"
        assert any("2 image(s) left unchanged" in t for t in result["toasts"]), result["toasts"]

    @pytest.mark.parametrize(
        "item_label,status",
        [("Mark label present", "present"), ("Clear label status (unreviewed)", "uncertain")],
    )
    def test_other_presence_actions_report_refusals_too(self, menu_page: Page, item_label, status):
        """All three presence actions go through the same server endpoint, so
        all three must handle `skipped` — not just the one that triggers it
        today."""
        result = menu_page.evaluate(
            _DRIVE_MENU,
            {"itemLabel": item_label, "adapterCount": 1, "skippedFlags": [True]},
        )
        assert not result.get("error"), result.get("error")
        assert result["calls"] == [{"i": 0, "uuid": LABEL_UUID, "status": status}]
        assert any("left unchanged" in t for t in result["toasts"]), result["toasts"]
