"""
Regression test for the Labels panel's sticky header (header controls + the
Mark absent / Reset / Verify all bulk-op row).

The panel used to give its own scroll container (`overflow-y:auto`) an
*inner* wrapper div that was already inside an outer `.sidebar-tab-pane`
which also scrolls. Nested auto-overflow ancestors break CSS `position:
sticky` — sticky anchors to the *nearest* scrolling ancestor, and the inner
wrapper never itself grows a scrollbar (unconstrained height), so the
"sticky" header just scrolled away with the outer pane instead of pinning to
its top.

Run with:
    pytest playwright_tests/test_label_panel_sticky_header.py -p no:xdist -v
"""

import json

import pytest
from playwright.sync_api import Page


def _mock_many_labels(page: Page, count: int = 30):
    labels = [
        {
            "uuid": f"11111111-2222-3333-4444-{i:012d}",
            "name": f"label-{i}",
            "color": "#123456",
            "labelType": "standard",
            "scope": "object",
            "parentUuid": None,
        }
        for i in range(count)
    ]
    payload = {"status": "success", "labels": labels}
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
        lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(payload)),
    )


class TestLabelPanelStickyHeader:
    def test_sticky_header_stays_pinned_while_list_scrolls(self, page: Page, base_url: str):
        """Scrolling the panel's list must not carry the sticky header away."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        _mock_many_labels(page)

        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createLabelList } = await import('/app-static/js/ui/labelList.js');
            const { clearLabelsCache } = await import('/app-static/js/shared/api/labels.js');
            clearLabelsCache();

            // Mimics the real relationship: an outer bounded, scrolling pane
            // (like .sidebar-tab-pane.active) hosting the panel's container,
            // which must NOT itself be a second overflow:auto box.
            const pane = document.createElement('div');
            Object.assign(pane.style, {
                height: '250px', width: '240px', overflowY: 'auto', position: 'relative',
            });
            document.body.appendChild(pane);

            const container = document.createElement('div');
            pane.appendChild(container);

            const runtime = createImageTaggingRuntime();
            await createLabelList(container, { mode: 'select', runtime });
            await new Promise(r => setTimeout(r, 50));

            const sticky = container.querySelector('.label-panel-sticky-header');
            if (!sticky) return { error: 'no .label-panel-sticky-header found' };

            const paneTopBefore = pane.getBoundingClientRect().top;
            const stickyTopBefore = sticky.getBoundingClientRect().top;

            pane.scrollTop = 300;
            await new Promise(r => requestAnimationFrame(r));

            const scrolled = pane.scrollTop;
            const stickyTopAfter = sticky.getBoundingClientRect().top;
            const paneTopAfter = pane.getBoundingClientRect().top;

            pane.remove();

            return {
                paneTopBefore, stickyTopBefore, scrolled, stickyTopAfter, paneTopAfter,
                containerOverflowY: getComputedStyle(container).overflowY,
            };
        }""")

        assert "error" not in result, result.get("error")
        # The container itself must not be a second scroll box — that's what
        # broke sticky positioning before this fix.
        assert result["containerOverflowY"] != "auto" and result["containerOverflowY"] != "scroll"
        # The scroll actually happened (enough labels to overflow the 250px pane).
        assert result["scrolled"] > 0, "test setup didn't produce enough rows to scroll"
        # Sticky header must remain pinned to the pane's top edge after scrolling,
        # not have scrolled up and out of view with the rest of the content.
        assert result["stickyTopAfter"] == pytest.approx(result["paneTopAfter"], abs=1)
