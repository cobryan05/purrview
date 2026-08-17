"""Rapid repeat saves must not put two metadata writes in flight at once.

POST /api/images/metadata/update replaces an image's whole bbox set, and nothing
orders two overlapping requests: clicking Save (or triggering the unsaved-changes
save on navigation) while an earlier save is still in flight could let the older
payload land last and silently undo everything drawn in between. saveAnnotations
queues saves behind each other instead — this asserts that at the network level.
"""

import json

from playwright.sync_api import Page


def test_repeat_saves_are_queued_not_concurrent(page: Page, base_url: str):
    def _serve_ok(route):
        route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.route("**/api/images/metadata/update", _serve_ok)

    result = page.evaluate("""async () => {
        const { saveAnnotations } = await import('/app-static/js/app/image-tagging/canvas/image.js');
        const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');

        const runtime = createImageTaggingRuntime();
        runtime.state.image = { uuid: 'img1', image_path: 'a.png', bboxes: new Map() };

        // Hold each update request open long enough that a concurrent second
        // save would overlap it, and record how many are open at once.
        const stats = { calls: 0, inFlight: 0, maxInFlight: 0, order: [] };
        const realFetch = window.fetch.bind(window);
        window.fetch = async (input, init) => {
            const url = typeof input === 'string' ? input : input.url;
            if (!url.includes('/api/images/metadata/update')) return realFetch(input, init);
            const n = ++stats.calls;
            stats.inFlight++;
            stats.maxInFlight = Math.max(stats.maxInFlight, stats.inFlight);
            await new Promise(r => setTimeout(r, 120));
            stats.inFlight--;
            stats.order.push(n);
            return realFetch(input, init);
        };

        try {
            // Deliberately not awaited in turn — this is the double-click.
            const first = saveAnnotations(runtime);
            const second = saveAnnotations(runtime);
            await Promise.all([first, second]);
        } finally {
            window.fetch = realFetch;
        }
        return stats;
    }""")

    assert result["calls"] == 2, "both saves should reach the server"
    assert result["maxInFlight"] == 1, f"saves overlapped: {result}"
    assert result["order"] == [1, 2], f"saves completed out of order: {result}"
