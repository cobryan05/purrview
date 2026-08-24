"""Regression tests for EventBus subscription leaks and tab-init races.

Uses the bus's internal _listeners map as the observable: a component that
subscribes on create must remove exactly its own handlers on destroy, and a
feature tab must initialise exactly once no matter how fast the user switches.
"""

import re

from playwright.sync_api import Page, expect

from playwright_tests.waits import wait_until

_COUNT_LISTENERS = """async () => {
    const { events } = await import('/app-static/js/shared/events/bus.js');
    let n = 0;
    for (const handlers of events._listeners.values()) n += handlers.size;
    return n;
}"""


def _listener_count(page: Page) -> int:
    return page.evaluate(_COUNT_LISTENERS)


class TestReviewPanelDestroy:
    def test_destroy_releases_all_subscriptions(self, page: Page, base_url: str):
        """createReviewPanel().destroy() must remove every EventBus handler it
        added — including the filter panel's five subscriptions."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        before = _listener_count(page)
        leaked = page.evaluate(
            """async (before) => {
            const { createReviewPanel } = await import('/app-static/js/app/review/main.js');
            const { events } = await import('/app-static/js/shared/events/bus.js');
            const div = document.createElement('div');
            document.body.appendChild(div);
            const panel = await createReviewPanel(div);
            panel.destroy();
            div.remove();
            let after = 0;
            for (const handlers of events._listeners.values()) after += handlers.size;
            return after - before;
        }""",
            before,
        )
        assert leaked == 0, f"review panel destroy leaked {leaked} EventBus subscriptions"


class TestTabInitRace:
    def test_rapid_tab_switching_initializes_feature_once(self, page: Page, base_url: str):
        """A→B→A tab clicks before the first init resolves must not run the
        feature's init() twice (double init leaks subscriptions and DOM)."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        # Racy: bounce back onto Review while its first init is still awaiting.
        # Review is a section tab inside Annotate now, so the race to guard is
        # the section-tier one — Canvas→Review→Canvas→Review before the first
        # init resolves — plus Annotate's own lazy init underneath it.
        page.evaluate("""() => {
            document.querySelector(".main-tab-button[data-tab='tab-image-tagging']").click();
            const sect = t => document.querySelector(
                `#tab-image-tagging .section-tab-button[data-tab='${t}']`);
            sect('tab-review')?.click();
            sect('tab-annotate-workspace')?.click();
            sect('tab-review')?.click();
        }""")
        # Wait for Review to actually be on screen before starting the settle window,
        # so the 2500ms is spent quiescing rather than partly waiting for a lazy module
        # to load — the difference between the two sides is what this test measures, and
        # a slow load on one side alone would read as a leak.
        expect(page.locator("#tab-review")).to_be_visible()
        page.wait_for_timeout(2500)
        racy = _listener_count(page)

        # Clean baseline: fresh page, single deliberate Review activation.
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
        page.locator("#tab-image-tagging .section-tab-button[data-tab='tab-review']").click()
        expect(page.locator("#tab-review")).to_be_visible()
        page.wait_for_timeout(2500)
        clean = _listener_count(page)

        assert (
            racy == clean
        ), f"racy tab switching left {racy} listeners vs {clean} after a clean init — feature init() ran twice"


class TestRunsTaskStreamLifecycle:
    def test_task_stream_closed_when_leaving_runs_tab(self, page: Page, base_url: str):
        """The task SSE stream opened by Runs must be closed when the user
        leaves it — otherwise the server keeps pushing to a hidden tab forever.

        This used to be asserted against Configuration > Task Management, back
        when Operations opened one tab-wide stream to feed its Activity badge
        and the Tasks sub-tab sat inside it. The stream moved with the thing
        that consumes it: the task-config editor never read it (ActiveTaskField
        polls on its own timer, and taskRunnerSection opens its own stream when
        a task is actually started), so Runs is now its only owner.
        """
        page.add_init_script("""
            window.__taskStreams = [];
            const OrigES = window.EventSource;
            window.EventSource = function(url, opts) {
                const es = new OrigES(url, opts);
                if (String(url).includes('/api/tasks/stream')) window.__taskStreams.push(es);
                return es;
            };
            window.EventSource.prototype = OrigES.prototype;
        """)
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        page.locator(".main-tab-button[data-tab='tab-runs']").click()
        wait_until(
            page,
            lambda: page.evaluate("() => window.__taskStreams.length"),
            "task SSE stream never opened on the Runs tab",
        )

        page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()

        # `all()` is vacuously true over an empty list, so "every stream is closed" has
        # to also require that there *is* one — otherwise a run where the init script
        # never saw a stream would report success. readyState 2 = CLOSED.
        def _all_closed():
            states = page.evaluate("() => window.__taskStreams.map(es => es.readyState)")
            return states if states and all(s == 2 for s in states) else None

        wait_until(
            page,
            _all_closed,
            lambda: "task SSE stream left open after leaving Runs tab: "
            + str(page.evaluate("() => window.__taskStreams.map(es => es.readyState)")),
        )

    def test_configuration_tasks_pane_opens_no_stream(self, page: Page, base_url: str):
        """Editing task configs must not open an SSE stream of its own.

        The pane has no consumer for it, and opening one per visit is how the
        leak this class guards against comes back.
        """
        page.add_init_script("""
            window.__taskStreams = [];
            const OrigES = window.EventSource;
            window.EventSource = function(url, opts) {
                const es = new OrigES(url, opts);
                if (String(url).includes('/api/tasks/stream')) window.__taskStreams.push(es);
                return es;
            };
            window.EventSource.prototype = OrigES.prototype;
        """)
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        page.locator(".main-tab-button[data-tab='tab-configuration']").click()
        page.wait_for_timeout(300)
        page.locator("#tab-configuration [data-tab='tab-task-management']").click()
        page.wait_for_timeout(500)

        assert page.evaluate("() => window.__taskStreams.length") == 0


class TestAsyncRenderReentrancy:
    def test_label_list_double_labels_updated_renders_once(self, page: Page, base_url: str):
        """Two synchronous LABELS_UPDATED events must not interleave two async
        mount() runs — that duplicates the rendered list and orphans the first
        mount's destroy handle."""
        import json
        import time

        labels_payload = {
            "status": "success",
            "labels": [
                {
                    "uuid": "11111111-2222-3333-4444-000000000001",
                    "name": "re-a",
                    "color": "#123456",
                    "labelType": "standard",
                    "scope": "object",
                    "parentUuid": None,
                },
                {
                    "uuid": "11111111-2222-3333-4444-000000000002",
                    "name": "re-b",
                    "color": "#654321",
                    "labelType": "standard",
                    "scope": "object",
                    "parentUuid": None,
                },
            ],
        }

        def _serve_labels(route):
            # Small delay so concurrent mounts genuinely suspend and interleave.
            time.sleep(0.15)
            route.fulfill(status=200, content_type="application/json", body=json.dumps(labels_payload))

        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.route("**/api/labels", _serve_labels)

        result = page.evaluate("""async () => {
            const { createLabelList } = await import('/app-static/js/ui/labelList.js');
            const { events, EventTypes } = await import('/app-static/js/shared/events/index.js');
            const { clearLabelsCache } = await import('/app-static/js/shared/api/labels.js');
            // Drop the cache warmed at page load so mounts hit the stub route.
            clearLabelsCache();
            const div = document.createElement('div');
            document.body.appendChild(div);
            const list = await createLabelList(div, { mode: 'model-config' });
            const single = div.children.length;
            // Cold cache: both re-mounts suspend on a real (slow) fetch,
            // exposing the interleave that warm-cache resolution masks.
            clearLabelsCache();
            events.publish(EventTypes.LABELS_UPDATED, {});
            clearLabelsCache();
            events.publish(EventTypes.LABELS_UPDATED, {});
            await new Promise(r => setTimeout(r, 1200));
            const after = div.children.length;
            list.destroy();
            div.remove();
            return { single, after };
        }""")
        assert result["single"] > 0, "label list rendered nothing — stubbed labels not picked up"
        assert (
            result["after"] == result["single"]
        ), f"double LABELS_UPDATED interleaved mounts: {result['after']} children vs {result['single']} after one"

    def test_tag_panel_double_tags_updated_renders_once(self, page: Page, base_url: str):
        """Same re-entrancy race on the inspector tag panel's _mount()."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        result = page.evaluate("""async () => {
            const { createTagPanel } = await import('/app-static/js/ui/inspector.js');
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { events, EventTypes } = await import('/app-static/js/shared/events/index.js');
            const div = document.createElement('div');
            document.body.appendChild(div);
            const runtime = createImageTaggingRuntime();
            const panel = createTagPanel(div, runtime);
            await new Promise(r => setTimeout(r, 800));  // initial _mount() is not awaited
            const single = div.children.length;
            events.publish(EventTypes.TAGS_UPDATED, {});
            events.publish(EventTypes.TAGS_UPDATED, {});
            await new Promise(r => setTimeout(r, 800));
            const after = div.children.length;
            panel.destroy();
            div.remove();
            return { single, after };
        }""")
        assert (
            result["after"] == result["single"]
        ), f"double TAGS_UPDATED interleaved mounts: {result['after']} children vs {result['single']} after one"


class TestMergeExternalBboxChanges:
    def test_merged_label_normalized_to_canvas_shape(self, page: Page, base_url: str):
        """Review-API bboxes carry label.name; canvas renderers read label.text.
        Merging external changes must normalize the shape or label chips render
        blank ("Loading…") after a merge."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        result = page.evaluate("""async () => {
            const { _mergeExternalBboxChanges } = await import('/app-static/js/app/review/main.js');
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');

            const div = document.createElement('div');
            div.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
            document.body.appendChild(div);
            const runtime = createImageTaggingRuntime();
            runtime.setCanvasContainer(div);

            const b = createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 });
            b.label = { uuid: 'l-old', text: 'old', color: '#111111' };
            runtime.state.image = { uuid: 'img', bboxes: new Map([['b1', b]]) };

            // API-shaped update: label has .name, no .text.
            const apiBbox = { label: { uuid: 'l-new', name: 'new-label', color: '#222222' }, labelUuid: 'l-new' };
            _mergeExternalBboxChanges(runtime, new Map([['b1', apiBbox]]), [], new Set());

            const merged = runtime.state.image.bboxes.get('b1');
            return { text: merged.label.text ?? null, uuid: merged.label.uuid };
        }""")
        assert result["uuid"] == "l-new"
        assert (
            result["text"] == "new-label"
        ), f"merged label.text is {result['text']!r} — canvas chip would render blank"


class TestImageLabelChipsBatch:
    def test_review_page_uses_one_batch_request_not_per_card_gets(self, page: Page, base_url: str):
        """Whole-image label chips must come from ONE POST /api/images/labels/batch
        per page render — not one GET /api/images/<path>/labels per card."""
        import json

        images = [{"uuid": f"img-{i}", "filename": f"folder/im{i}.jpg", "bboxes": []} for i in range(12)]
        page.route(
            "**/api/images/review",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "images": images, "hasMore": False, "totalCount": len(images)}),
            ),
        )
        batch_calls = []
        per_card_calls = []
        page.route(
            "**/api/images/labels/batch",
            lambda route: (
                batch_calls.append(json.loads(route.request.post_data or "{}")),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {"status": "success", "imageLabels": {"img-0": [{"labelUuid": "lbl-x", "status": "present"}]}}
                    ),
                ),
            )[-1],
        )
        page.route(
            re.compile(r".*/api/images/.+/labels$"),
            lambda route: (
                per_card_calls.append(route.request.url),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labels": []}),
                ),
            )[-1],
        )

        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
        page.locator("#tab-image-tagging .section-tab-button[data-tab='tab-review']").click()
        page.wait_for_selector("#review-root .review-grid")
        wait_until(
            page, lambda: len(batch_calls) >= 1, lambda: f"expected exactly one batch call, got {len(batch_calls)}"
        )

        assert len(batch_calls) == 1, f"expected exactly one batch call, got {len(batch_calls)}"
        assert set(batch_calls[0]["imageUuids"]) == {img["uuid"] for img in images}
        assert per_card_calls == [], f"per-card label GETs still issued: {per_card_calls[:3]}"


class TestViewToggleSubscriptionLeak:
    def test_view_filter_toggle_does_not_duplicate_card_subscriptions(self, page: Page, base_url: str):
        """Switching All/Images/Bboxes rebuilds every card; each rebuild must
        drain the previous cards' EventBus subscriptions or they double per
        click. (Was the grid↔list toggle, before list view was removed.)"""
        import json

        images = [{"uuid": "img-1", "filename": "folder/im1.jpg", "bboxes": []}]
        page.route(
            "**/api/images/review",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "images": images, "hasMore": False, "totalCount": 1}),
            ),
        )
        page.route(
            "**/api/images/labels/batch",
            lambda route: route.fulfill(
                status=200, content_type="application/json", body=json.dumps({"status": "success", "imageLabels": {}})
            ),
        )

        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
        page.locator("#tab-image-tagging .section-tab-button[data-tab='tab-review']").click()
        page.wait_for_selector("#review-root .review-grid")
        page.wait_for_timeout(500)

        baseline = _listener_count(page)
        # Toggle back and forth a few times; listener count must not grow.
        for _ in range(3):
            page.locator("#review-root .view-btn[data-view-key='bboxes']").click()
            page.wait_for_timeout(250)
            page.locator("#review-root .view-btn[data-view-key='images']").click()
            page.wait_for_timeout(250)
        after = _listener_count(page)
        assert after == baseline, f"view toggling grew listeners {baseline} → {after}"
