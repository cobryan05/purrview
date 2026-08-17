"""
Pure-logic unit tests — no real network, minimal DOM.

Each test imports a specific module inside the browser via page.evaluate so
/app-static/ imports resolve correctly. All tests share one pre-navigated
`shared_page` (module-scoped, see conftest.py) — they create fresh instances
or call `events.clear()` for the EventBus singleton, so no module-level state
leaks between tests.
"""

import re

import pytest
from playwright.sync_api import Page

# ---------------------------------------------------------------------------
# viewport.js — computeViewport
# ---------------------------------------------------------------------------


class TestComputeViewport:
    # computeViewport applies a small FIT_MARGIN (0.96) below the raw
    # fit-to-container scale so the default zoom leaves a visible margin
    # around the image instead of running edge-to-edge (WW-xxx: bboxes
    # touching the top/bottom edge were getting truncated under the
    # bottom status bar without the user noticing).
    FIT_MARGIN = 0.96

    def test_landscape_image_fits_by_width(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { computeViewport } = await import('/app-static/js/app/image-tagging/canvas/viewport.js');
            // container 800x600, image 400x200 (landscape) → scale limited by width
            return computeViewport(800, 600, 400, 200);
        }""")
        assert result["scale"] == pytest.approx(2.0 * self.FIT_MARGIN)
        assert result["offsetY"] > 0  # vertically centred
        assert result["offsetX"] > 0  # margin leaves a gap on the fitted axis too

    def test_portrait_image_fits_by_height(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { computeViewport } = await import('/app-static/js/app/image-tagging/canvas/viewport.js');
            // container 400x800, image 100x400 → height-limited
            return computeViewport(400, 800, 100, 400);
        }""")
        assert result["scale"] == pytest.approx(2.0 * self.FIT_MARGIN)
        assert result["offsetX"] > 0
        assert result["offsetY"] > 0

    def test_square_image_square_container(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { computeViewport } = await import('/app-static/js/app/image-tagging/canvas/viewport.js');
            return computeViewport(500, 500, 500, 500);
        }""")
        assert result["scale"] == pytest.approx(self.FIT_MARGIN)
        assert result["offsetX"] == pytest.approx(result["offsetY"])
        assert result["offsetX"] > 0

    def test_large_image_scales_down(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { computeViewport } = await import('/app-static/js/app/image-tagging/canvas/viewport.js');
            return computeViewport(200, 200, 1000, 1000);
        }""")
        assert result["scale"] == pytest.approx(0.2 * self.FIT_MARGIN)

    def test_image_width_and_height_preserved(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { computeViewport } = await import('/app-static/js/app/image-tagging/canvas/viewport.js');
            return computeViewport(800, 600, 320, 240);
        }""")
        assert result["imageWidth"] == 320
        assert result["imageHeight"] == 240

    def test_fitted_image_never_touches_container_edge(self, shared_page: Page):
        """Both axes must have a strictly positive offset so the rendered
        image (imageDim * scale) sits fully inside the container, with room
        to spare on the axis that would otherwise fill it exactly."""
        page = shared_page
        result = page.evaluate("""async () => {
            const { computeViewport } = await import('/app-static/js/app/image-tagging/canvas/viewport.js');
            const vp = computeViewport(800, 600, 400, 200);
            return {
                offsetX: vp.offsetX,
                offsetY: vp.offsetY,
                renderedWidth: vp.imageWidth * vp.scale,
                renderedHeight: vp.imageHeight * vp.scale,
            };
        }""")
        assert result["offsetX"] > 0
        assert result["offsetY"] > 0
        assert result["renderedWidth"] < 800
        assert result["renderedHeight"] < 600


class TestScreenToImage:
    def test_zero_offset_identity(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { screenToImage } = await import('/app-static/js/app/image-tagging/canvas/viewport.js');
            const vp = { offsetX: 0, offsetY: 0, scale: 1 };
            return screenToImage(100, 200, vp);
        }""")
        assert result["x"] == pytest.approx(100)
        assert result["y"] == pytest.approx(200)

    def test_with_offset_and_scale(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { screenToImage } = await import('/app-static/js/app/image-tagging/canvas/viewport.js');
            const vp = { offsetX: 50, offsetY: 100, scale: 2 };
            // screenX=150 → (150-50)/2 = 50
            return screenToImage(150, 300, vp);
        }""")
        assert result["x"] == pytest.approx(50)
        assert result["y"] == pytest.approx(100)


# ---------------------------------------------------------------------------
# tools.js — isDrawTool, isTagTool, parseToolUuid
# ---------------------------------------------------------------------------


class TestToolPredicates:
    def test_isDrawTool_bare_bbox(self, shared_page: Page):
        page = shared_page
        assert page.evaluate("""async () => {
            const { isDrawTool } = await import('/app-static/js/app/image-tagging/canvas/tools.js');
            return isDrawTool('bbox');
        }""") is True

    def test_isDrawTool_bbox_with_uuid(self, shared_page: Page):
        page = shared_page
        assert page.evaluate("""async () => {
            const { isDrawTool } = await import('/app-static/js/app/image-tagging/canvas/tools.js');
            return isDrawTool('bbox:some-uuid-here');
        }""") is True

    def test_isDrawTool_select_is_false(self, shared_page: Page):
        page = shared_page
        assert page.evaluate("""async () => {
            const { isDrawTool } = await import('/app-static/js/app/image-tagging/canvas/tools.js');
            return isDrawTool('select');
        }""") is False

    def test_isDrawTool_tag_is_false(self, shared_page: Page):
        page = shared_page
        assert page.evaluate("""async () => {
            const { isDrawTool } = await import('/app-static/js/app/image-tagging/canvas/tools.js');
            return isDrawTool('tag:some-uuid');
        }""") is False

    def test_isTagTool_with_uuid(self, shared_page: Page):
        page = shared_page
        assert page.evaluate("""async () => {
            const { isTagTool } = await import('/app-static/js/app/image-tagging/canvas/tools.js');
            return isTagTool('tag:some-uuid');
        }""") is True

    def test_isTagTool_select_is_false(self, shared_page: Page):
        page = shared_page
        assert page.evaluate("""async () => {
            const { isTagTool } = await import('/app-static/js/app/image-tagging/canvas/tools.js');
            return isTagTool('select');
        }""") is False

    def test_isTagTool_bbox_is_false(self, shared_page: Page):
        page = shared_page
        assert page.evaluate("""async () => {
            const { isTagTool } = await import('/app-static/js/app/image-tagging/canvas/tools.js');
            return isTagTool('bbox:uuid');
        }""") is False


class TestParseToolUuid:
    def test_select_has_null_uuid(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { parseToolUuid } = await import('/app-static/js/app/image-tagging/canvas/tools.js');
            return parseToolUuid('select');
        }""")
        assert result == {"tool": "select", "uuid": None}

    def test_bare_bbox_has_null_uuid(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { parseToolUuid } = await import('/app-static/js/app/image-tagging/canvas/tools.js');
            return parseToolUuid('bbox:');
        }""")
        assert result["tool"] == "bbox"
        assert result["uuid"] is None

    def test_bbox_with_uuid(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { parseToolUuid } = await import('/app-static/js/app/image-tagging/canvas/tools.js');
            return parseToolUuid('bbox:abc-123');
        }""")
        assert result == {"tool": "bbox", "uuid": "abc-123"}

    def test_tag_with_uuid(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { parseToolUuid } = await import('/app-static/js/app/image-tagging/canvas/tools.js');
            return parseToolUuid('tag:def-456');
        }""")
        assert result == {"tool": "tag", "uuid": "def-456"}


# ---------------------------------------------------------------------------
# shared/events/bus.js — EventBus
# ---------------------------------------------------------------------------


class TestEventBus:
    def test_subscribe_and_publish_delivers_payload(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { events } = await import('/app-static/js/shared/events/bus.js');
            events.clear();
            let received = null;
            events.subscribe('test:event', (payload) => { received = payload; });
            events.publish('test:event', { value: 42 });
            return received;
        }""")
        assert result == {"value": 42}

    def test_unsubscribe_stops_delivery(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { events } = await import('/app-static/js/shared/events/bus.js');
            events.clear();
            let count = 0;
            const unsub = events.subscribe('test:evt', () => count++);
            events.publish('test:evt', {});
            unsub();
            events.publish('test:evt', {});
            return count;
        }""")
        assert result == 1

    def test_multiple_handlers_all_called(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { events } = await import('/app-static/js/shared/events/bus.js');
            events.clear();
            const calls = [];
            events.subscribe('multi', () => calls.push('a'));
            events.subscribe('multi', () => calls.push('b'));
            events.publish('multi', {});
            return calls;
        }""")
        assert sorted(result) == ["a", "b"]

    def test_subscribeToCanvas_passes_global_event(self, shared_page: Page):
        """Events with no runtimeId in payload reach all subscribeToCanvas listeners."""
        page = shared_page
        result = page.evaluate("""async () => {
            const { events } = await import('/app-static/js/shared/events/bus.js');
            events.clear();
            let received = false;
            events.subscribeToCanvas('runtime-A', 'canvas:evt', () => { received = true; });
            events.publish('canvas:evt', {});   // no runtimeId
            return received;
        }""")
        assert result is True

    def test_subscribeToCanvas_filters_wrong_runtimeId(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { events } = await import('/app-static/js/shared/events/bus.js');
            events.clear();
            let received = false;
            events.subscribeToCanvas('runtime-A', 'canvas:evt', () => { received = true; });
            events.publish('canvas:evt', { runtimeId: 'runtime-B' });
            return received;
        }""")
        assert result is False

    def test_subscribeToCanvas_accepts_matching_runtimeId(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { events } = await import('/app-static/js/shared/events/bus.js');
            events.clear();
            let received = false;
            events.subscribeToCanvas('runtime-A', 'canvas:evt', () => { received = true; });
            events.publish('canvas:evt', { runtimeId: 'runtime-A' });
            return received;
        }""")
        assert result is True

    def test_clear_removes_all_listeners(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { events } = await import('/app-static/js/shared/events/bus.js');
            events.clear();
            let count = 0;
            events.subscribe('test:clear', () => count++);
            events.clear();
            events.publish('test:clear', {});
            return count;
        }""")
        assert result == 0


# ---------------------------------------------------------------------------
# createCachedFetcher.js
# ---------------------------------------------------------------------------


class TestCreateCachedFetcher:
    def test_first_fetch_calls_fn(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCachedFetcher } = await import('/app-static/js/ui/utils/data/createCachedFetcher.js');
            let callCount = 0;
            const fetcher = createCachedFetcher(async (keys) => {
                callCount++;
                const m = new Map();
                for (const k of keys) m.set(k, 'v_' + k);
                return m;
            });
            const val = await fetcher.fetch('key1');
            return { val, callCount };
        }""")
        assert result["val"] == "v_key1"
        assert result["callCount"] == 1

    def test_second_fetch_is_cache_hit(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCachedFetcher } = await import('/app-static/js/ui/utils/data/createCachedFetcher.js');
            let callCount = 0;
            const fetcher = createCachedFetcher(async (keys) => {
                callCount++;
                const m = new Map(); for (const k of keys) m.set(k, 'v_' + k); return m;
            });
            await fetcher.fetch('key1');
            await fetcher.fetch('key1');
            return callCount;
        }""")
        assert result == 1

    def test_concurrent_fetches_deduplicated(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCachedFetcher } = await import('/app-static/js/ui/utils/data/createCachedFetcher.js');
            let callCount = 0;
            const fetcher = createCachedFetcher(async (keys) => {
                callCount++;
                await new Promise(r => setTimeout(r, 20));
                const m = new Map(); for (const k of keys) m.set(k, 'v_' + k); return m;
            });
            const [r1, r2] = await Promise.all([fetcher.fetch('key1'), fetcher.fetch('key1')]);
            return { r1, r2, callCount };
        }""")
        assert result["r1"] == "v_key1"
        assert result["r2"] == "v_key1"
        assert result["callCount"] == 1

    def test_clear_forces_refetch(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCachedFetcher } = await import('/app-static/js/ui/utils/data/createCachedFetcher.js');
            let callCount = 0;
            const fetcher = createCachedFetcher(async (keys) => {
                callCount++;
                const m = new Map(); for (const k of keys) m.set(k, callCount + '_' + k); return m;
            });
            await fetcher.fetch('k');
            fetcher.clear();
            await fetcher.fetch('k');
            return callCount;
        }""")
        assert result == 2

    def test_set_preloads_cache(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCachedFetcher } = await import('/app-static/js/ui/utils/data/createCachedFetcher.js');
            let callCount = 0;
            const fetcher = createCachedFetcher(async (keys) => { callCount++; return new Map(); });
            fetcher.set('myKey', 'preloaded');
            const val = await fetcher.fetch('myKey');
            return { val, callCount };
        }""")
        assert result["val"] == "preloaded"
        assert result["callCount"] == 0

    def test_array_key_returns_map(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCachedFetcher } = await import('/app-static/js/ui/utils/data/createCachedFetcher.js');
            const fetcher = createCachedFetcher(async (keys) => {
                const m = new Map(); for (const k of keys) m.set(k, 'v_' + k); return m;
            });
            const result = await fetcher.fetch(['a', 'b']);
            return [result.get('a'), result.get('b')];
        }""")
        assert result == ["v_a", "v_b"]


# ---------------------------------------------------------------------------
# fileNavigation.js — FileNavigation
# ---------------------------------------------------------------------------


class TestFileNavigation:
    def test_empty_navigation(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { FileNavigation } = await import('/app-static/js/ui/utils/fileNavigation.js');
            const nav = new FileNavigation();
            return { current: nav.current(), hasNext: nav.hasNext(), hasPrev: nav.hasPrev() };
        }""")
        assert result["current"] is None
        assert result["hasNext"] is False
        assert result["hasPrev"] is False

    def test_setDirectory_resets_index(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { FileNavigation } = await import('/app-static/js/ui/utils/fileNavigation.js');
            const nav = new FileNavigation(['a', 'b', 'c'], '/dir');
            nav.select('b');
            nav.setDirectory('/new', ['x', 'y']);
            return { current: nav.current(), files: nav.getFiles() };
        }""")
        assert result["current"] is None
        assert result["files"] == ["x", "y"]

    def test_select_by_name(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { FileNavigation } = await import('/app-static/js/ui/utils/fileNavigation.js');
            const nav = new FileNavigation(['a.jpg', 'b.jpg', 'c.jpg'], '/dir');
            nav.select('b.jpg');
            return nav.current();
        }""")
        assert result == "b.jpg"

    def test_select_nonexistent_does_nothing(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { FileNavigation } = await import('/app-static/js/ui/utils/fileNavigation.js');
            const nav = new FileNavigation(['a.jpg'], '/dir');
            nav.select('missing.jpg');
            return nav.current();
        }""")
        assert result is None

    def test_next_and_prev(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { FileNavigation } = await import('/app-static/js/ui/utils/fileNavigation.js');
            const nav = new FileNavigation(['a', 'b', 'c'], '/dir');
            nav.select('a');
            const r1 = nav.next();   // → b
            const r2 = nav.next();   // → c
            const r3 = nav.next();   // → c (capped)
            const r4 = nav.prev();   // → b
            return [r1, r2, r3, r4];
        }""")
        assert result == ["b", "c", "c", "b"]

    def test_has_next_has_prev_boundaries(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { FileNavigation } = await import('/app-static/js/ui/utils/fileNavigation.js');
            const nav = new FileNavigation(['a', 'b'], '/dir');
            nav.select('a');
            const atFirst = { hasNext: nav.hasNext(), hasPrev: nav.hasPrev() };
            nav.next();
            const atLast = { hasNext: nav.hasNext(), hasPrev: nav.hasPrev() };
            return { atFirst, atLast };
        }""")
        assert result["atFirst"] == {"hasNext": True, "hasPrev": False}
        assert result["atLast"] == {"hasNext": False, "hasPrev": True}

    def test_currentPath_joins_dir_and_file(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { FileNavigation } = await import('/app-static/js/ui/utils/fileNavigation.js');
            const nav = new FileNavigation(['img.jpg'], '/my/dir');
            nav.select('img.jpg');
            return nav.currentPath();
        }""")
        assert result == "/my/dir/img.jpg"

    def test_currentPath_root_dir_no_double_slash(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { FileNavigation } = await import('/app-static/js/ui/utils/fileNavigation.js');
            const nav = new FileNavigation(['img.jpg'], '/');
            nav.select('img.jpg');
            return nav.currentPath();
        }""")
        assert result == "img.jpg"

    def test_getFiles_returns_copy(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { FileNavigation } = await import('/app-static/js/ui/utils/fileNavigation.js');
            const nav = new FileNavigation(['a', 'b', 'c'], '/dir');
            const files = nav.getFiles();
            files.push('extra');   // mutate the copy
            return nav.getFiles().length;  // original should be unaffected
        }""")
        assert result == 3


# ---------------------------------------------------------------------------
# app/image-tagging/files.js — next_image / prev_image
# ---------------------------------------------------------------------------


class TestFilesNavigationGuard:
    # A canvas modal opened with `allowNavigation: false` (e.g. the "run model
    # on results" single-candidate editor) never calls setFileNavigation, so
    # runtime.fileNavigation stays null. The global '[' / ']' hotkeys still
    # dispatch to whatever runtime is on top of the hotkeys stack regardless
    # of whether it supports navigation, so next_image/prev_image must no-op
    # instead of crashing when there's no FileNavigation to consult.
    def test_next_image_noop_when_no_file_navigation(self, shared_page: Page):
        page = shared_page
        threw = page.evaluate("""async () => {
            const { next_image } = await import('/app-static/js/app/image-tagging/files.js');
            try {
                await next_image({ fileNavigation: null, state: {} });
                return false;
            } catch (e) {
                return true;
            }
        }""")
        assert threw is False

    def test_prev_image_noop_when_no_file_navigation(self, shared_page: Page):
        page = shared_page
        threw = page.evaluate("""async () => {
            const { prev_image } = await import('/app-static/js/app/image-tagging/files.js');
            try {
                await prev_image({ fileNavigation: null, state: {} });
                return false;
            } catch (e) {
                return true;
            }
        }""")
        assert threw is False


# ---------------------------------------------------------------------------
# app/hotkeys.js — who owns the hotkeys
# ---------------------------------------------------------------------------


class TestHotkeyOwnership:
    # An `annotated_image` status widget builds a read-only canvas preview via
    # createCanvasInstance, and ActiveTaskField rebuilds it on every 5s poll
    # tick whose data changed — with no user action involved. attachHotkeys is
    # "most recently attached wins", so each rebuild silently took ownership of
    # every hotkey away from the canvas modal the user had open. The keys still
    # fired and still called preventDefault(); they just drove an invisible
    # 16em preview, which looked exactly like the hotkeys being dead.
    def test_canvas_instance_opting_out_does_not_steal_ownership(self, shared_page: Page):
        page = shared_page
        stole = page.evaluate("""async () => {
            const { createCanvasInstance } = await import('/app-static/js/app/image-tagging/canvas/instance.js');
            const { attachHotkeys, activeHotkeyRuntime } = await import('/app-static/js/app/hotkeys.js');
            const owner = { id: 'owner-probe', fileNavigation: null, state: {} };
            const ownerHandle = attachHotkeys(owner);
            const div = document.createElement('div');
            div.style.cssText = 'position:fixed; left:-9999px; width:400px; height:300px;';
            document.body.appendChild(div);
            try {
                const inst = await createCanvasInstance(
                    div, { enhance: 'none', lockEnhance: true, hotkeys: false });
                const stole = activeHotkeyRuntime() !== owner;
                inst.destroy();
                return stole;
            } finally {
                ownerHandle.destroy();
                div.remove();
            }
        }""")
        assert stole is False

    # Guards the test above against passing for the wrong reason: prove that a
    # canvas which does NOT opt out still takes ownership, so the assertion is
    # measuring the opt-out and not a broken/no-op probe.
    def test_canvas_instance_takes_ownership_by_default(self, shared_page: Page):
        page = shared_page
        stole = page.evaluate("""async () => {
            const { createCanvasInstance } = await import('/app-static/js/app/image-tagging/canvas/instance.js');
            const { attachHotkeys, activeHotkeyRuntime } = await import('/app-static/js/app/hotkeys.js');
            const owner = { id: 'owner-probe', fileNavigation: null, state: {} };
            const ownerHandle = attachHotkeys(owner);
            const div = document.createElement('div');
            div.style.cssText = 'position:fixed; left:-9999px; width:400px; height:300px;';
            document.body.appendChild(div);
            try {
                const inst = await createCanvasInstance(div, { enhance: 'none', lockEnhance: true });
                const stole = activeHotkeyRuntime() !== owner;
                inst.destroy();
                return stole;
            } finally {
                ownerHandle.destroy();
                div.remove();
            }
        }""")
        assert stole is True

    # Second half of the same failure: a runtime with no FileNavigation can't
    # navigate, so next_image/prev_image no-op on it — but dispatchAction still
    # reported "handled" and called preventDefault(). That turned a misrouted
    # hotkey into a completely silent one. Reporting it unhandled instead means
    # the key falls through to its default action, which is at least visible.
    def test_navigation_hotkey_is_not_swallowed_without_file_navigation(self, shared_page: Page):
        page = shared_page
        prevented = page.evaluate("""async () => {
            const { attachHotkeys } = await import('/app-static/js/app/hotkeys.js');
            const handle = attachHotkeys({ id: 'nav-less', fileNavigation: null, state: {} });
            try {
                const results = {};
                for (const key of [']', '[']) {
                    const ev = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true });
                    document.dispatchEvent(ev);
                    results[key] = ev.defaultPrevented;
                }
                return results;
            } finally {
                handle.destroy();
            }
        }""")
        assert prevented == {"]": False, "[": False}


# ---------------------------------------------------------------------------
# ui/utils/utils.js — generateUUID
# ---------------------------------------------------------------------------


class TestGenerateUUID:
    def test_format_matches_rfc4122(self, shared_page: Page):
        page = shared_page
        uuid = page.evaluate("""async () => {
            const { generateUUID } = await import('/app-static/js/ui/utils/utils.js');
            return generateUUID();
        }""")
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            uuid,
        ), f"UUID format wrong: {uuid}"

    def test_two_calls_are_distinct(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { generateUUID } = await import('/app-static/js/ui/utils/utils.js');
            return [generateUUID(), generateUUID()];
        }""")
        assert result[0] != result[1]


# ---------------------------------------------------------------------------
# shared/api/apiUtils.js — ignoreKeyReturn
# ---------------------------------------------------------------------------


class TestIgnoreKeyReturn:
    def test_single_key_returns_value(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { ignoreKeyReturn } = await import('/app-static/js/shared/api/apiUtils.js');
            const m = ignoreKeyReturn(['k1'], 'hello');
            return m.get('k1');
        }""")
        assert result == "hello"

    def test_multiple_keys_all_get_same_value(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { ignoreKeyReturn } = await import('/app-static/js/shared/api/apiUtils.js');
            const m = ignoreKeyReturn(['a', 'b', 'c'], 42);
            return [m.get('a'), m.get('b'), m.get('c')];
        }""")
        assert result == [42, 42, 42]


# ---------------------------------------------------------------------------
# shared/prefs.js — parse-safe storage wrappers
# ---------------------------------------------------------------------------


class TestPrefs:
    """The point of the helper is that a value a *previous* version of the app
    wrote in a different shape can never throw at a call site. Every module used
    to hand-roll this; these tests pin the one implementation."""

    # The shared_page is module-scoped, so each test cleans up its scratch key
    # rather than relying on a fresh store.

    def test_read_returns_fallback_when_absent(self, shared_page: Page):
        result = shared_page.evaluate("""async () => {
            const { prefs } = await import('/app-static/js/shared/prefs.js');
            localStorage.removeItem('ww_test_pref');
            return prefs.read('ww_test_pref', { fallback: true });
        }""")
        assert result == {"fallback": True}

    def test_read_returns_fallback_on_corrupt_json(self, shared_page: Page):
        """The hazard the helper exists for: a key written in an older shape."""
        result = shared_page.evaluate("""async () => {
            const { prefs } = await import('/app-static/js/shared/prefs.js');
            localStorage.setItem('ww_test_pref', 'not{valid]json');
            const out = prefs.read('ww_test_pref', ['safe']);
            localStorage.removeItem('ww_test_pref');
            return out;
        }""")
        assert result == ["safe"]

    def test_read_returns_fallback_for_stored_null(self, shared_page: Page):
        result = shared_page.evaluate("""async () => {
            const { prefs } = await import('/app-static/js/shared/prefs.js');
            localStorage.setItem('ww_test_pref', 'null');
            const out = prefs.read('ww_test_pref', 'fb');
            localStorage.removeItem('ww_test_pref');
            return out;
        }""")
        assert result == "fb"

    def test_write_read_round_trip(self, shared_page: Page):
        result = shared_page.evaluate("""async () => {
            const { prefs } = await import('/app-static/js/shared/prefs.js');
            prefs.write('ww_test_pref', { a: [1, 2], b: false });
            const out = prefs.read('ww_test_pref', null);
            prefs.remove('ww_test_pref');
            return { out, afterRemove: prefs.read('ww_test_pref', 'gone') };
        }""")
        assert result["out"] == {"a": [1, 2], "b": False}
        assert result["afterRemove"] == "gone"

    def test_write_reports_failure_instead_of_throwing(self, shared_page: Page):
        """A circular structure is unstorable — the caller gets false, not an
        exception that takes down whatever it was called from."""
        result = shared_page.evaluate("""async () => {
            const { prefs } = await import('/app-static/js/shared/prefs.js');
            const circular = {}; circular.self = circular;
            const ok = prefs.write('ww_test_pref', circular);
            prefs.remove('ww_test_pref');
            return ok;
        }""")
        assert result is False

    def test_raw_read_write_bypasses_json(self, shared_page: Page):
        """Keys whose stored format predates the helper stay byte-compatible:
        a sidebar width is `240`, not `"240"`."""
        result = shared_page.evaluate("""async () => {
            const { prefs } = await import('/app-static/js/shared/prefs.js');
            prefs.writeRaw('ww_test_pref', 240);
            const stored = localStorage.getItem('ww_test_pref');
            const out = prefs.readRaw('ww_test_pref');
            prefs.remove('ww_test_pref');
            return { stored, out, missing: prefs.readRaw('ww_test_pref') };
        }""")
        assert result["stored"] == "240"
        assert result["out"] == "240"
        assert result["missing"] is None

    def test_boolean_write_stays_compatible_with_the_old_string_format(self, shared_page: Page):
        """activeModel used to write `String(bool)`. JSON booleans serialise
        identically, so previously-stored values still read back correctly."""
        result = shared_page.evaluate("""async () => {
            const { prefs } = await import('/app-static/js/shared/prefs.js');
            localStorage.setItem('ww_test_pref', String(true));   // legacy write
            const out = prefs.read('ww_test_pref', false);
            prefs.write('ww_test_pref', false);
            const reWritten = localStorage.getItem('ww_test_pref');
            prefs.remove('ww_test_pref');
            return { out, reWritten };
        }""")
        assert result["out"] is True
        assert result["reWritten"] == "false"

    def test_local_and_session_are_separate_stores(self, shared_page: Page):
        result = shared_page.evaluate("""async () => {
            const { prefs, sessionPrefs } = await import('/app-static/js/shared/prefs.js');
            prefs.write('ww_test_pref', 'local');
            sessionPrefs.write('ww_test_pref', 'session');
            const out = {
                local: prefs.read('ww_test_pref', null),
                session: sessionPrefs.read('ww_test_pref', null),
            };
            prefs.remove('ww_test_pref');
            sessionPrefs.remove('ww_test_pref');
            return out;
        }""")
        assert result == {"local": "local", "session": "session"}


# ---------------------------------------------------------------------------
# ui/labelList.js — buildHierarchy (pure function, exported separately)
# ---------------------------------------------------------------------------


class TestBuildHierarchy:
    def test_root_and_children(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { buildHierarchy } = await import('/app-static/js/ui/labelList.js');
            const labels = new Map([
                ['uuid1', { uuid: 'uuid1', name: 'Animal', parentUuid: null}],
                ['uuid2', { uuid: 'uuid2', name: 'Cat',    parentUuid: 'uuid1' }],
                ['uuid3', { uuid: 'uuid3', name: 'Dog',    parentUuid: 'uuid1' }],
            ]);
            const h = buildHierarchy(labels, null);
            return {
                rootUuids:    (h.get(null)    || []).map(l => l.uuid),
                uuid1Children: (h.get('uuid1') || []).map(l => l.uuid),
            };
        }""")
        assert result["rootUuids"] == ["uuid1"]
        assert sorted(result["uuid1Children"]) == ["uuid2", "uuid3"]

    def test_children_sorted_alphabetically(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { buildHierarchy } = await import('/app-static/js/ui/labelList.js');
            const labels = new Map([
                ['p', { uuid: 'p', name: 'Parent', parentUuid: null}],
                ['z', { uuid: 'z', name: 'Zebra', parentUuid: 'p' }],
                ['a', { uuid: 'a', name: 'Aardvark', parentUuid: 'p' }],
                ['m', { uuid: 'm', name: 'Mouse', parentUuid: 'p' }],
            ]);
            const h = buildHierarchy(labels, null);
            return (h.get('p') || []).map(l => l.name);
        }""")
        assert result == ["Aardvark", "Mouse", "Zebra"]

    def test_filter_excludes_labels(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { buildHierarchy } = await import('/app-static/js/ui/labelList.js');
            const labels = new Map([
                ['a', { uuid: 'a', name: 'Alpha', parentUuid: null}],
                ['b', { uuid: 'b', name: 'Beta',  parentUuid: null}],
            ]);
            // filter: only include labels whose name starts with 'A'
            const h = buildHierarchy(labels, lbl => lbl.name.startsWith('A'));
            return (h.get(null) || []).map(l => l.name);
        }""")
        assert result == ["Alpha"]

    def test_empty_map_returns_empty_hierarchy(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { buildHierarchy } = await import('/app-static/js/ui/labelList.js');
            const h = buildHierarchy(new Map(), null);
            return h.size;
        }""")
        assert result == 0


# ---------------------------------------------------------------------------
# ui/tasks.js — the two task counts
# ---------------------------------------------------------------------------


class TestTaskCounts:
    """Two counts over the same task map, answering two different questions.

    The badge asks "how many things are running?" — a paused job has not
    finished and an armed watcher is running, so both count, and both appear
    under Active where the badge's number is read. The off-tab SSE stream asks
    "can anything finish behind the user's back?" — a paused job cannot (only
    the user can restart it) and a continuous run never finishes at all, so
    neither may hold it. Conflating them held an EventSource open per browser
    tab for the life of the app, because restart recovery writes `paused` for
    any interrupted task that had a checkpoint.
    """

    TASKS = """{
        running:    { instance: { status: 'running' },   continuous: false },
        paused:     { instance: { status: 'paused' },    continuous: false },
        done:       { instance: { status: 'completed' }, continuous: false },
        failed:     { instance: { status: 'error' },     continuous: false },
        watcher:    { instance: { status: 'running' },   continuous: true },
    }"""

    def _counts(self, page: Page, tasks: str):
        return page.evaluate(
            """async (tasksSrc) => {
            const { countLiveTasks, countLiveTasksByKind, countAdvancingTasks } =
                await import('/app-static/js/ui/tasks.js');
            const tasks = eval('(' + tasksSrc + ')');
            return {
                live: countLiveTasks(tasks),
                advancing: countAdvancingTasks(tasks),
                byKind: countLiveTasksByKind(tasks),
            };
        }""",
            tasks,
        )

    def test_the_badge_counts_a_paused_job(self, shared_page: Page):
        assert self._counts(shared_page, self.TASKS)["live"] == 3  # running + paused + watcher

    def test_the_stream_condition_does_not_count_a_paused_job(self, shared_page: Page):
        assert self._counts(shared_page, self.TASKS)["advancing"] == 1  # running only

    def test_a_lone_paused_job_does_not_hold_the_stream_open(self, shared_page: Page):
        """The leak itself: one paused instance and an otherwise idle app."""
        counts = self._counts(shared_page, "{ p: { instance: { status: 'paused' }, continuous: false } }")
        assert counts["advancing"] == 0, "a paused job kept the off-tab stream alive"
        assert counts["live"] == 1, "a paused job is still unfinished and belongs on the badge"

    def test_a_watcher_counts_on_the_badge_but_never_holds_the_stream(self, shared_page: Page):
        """It is listed under Active, so it belongs in Active's number — but it
        never finishes, so there is no completion to stay connected for."""
        counts = self._counts(shared_page, "{ w: { instance: { status: 'running' }, continuous: true } }")
        assert counts["live"] == 1
        assert counts["advancing"] == 0

    def test_the_badge_count_splits_into_jobs_and_continuous_runs(self, shared_page: Page):
        """The tooltip explains the number; the split is what it says."""
        assert self._counts(shared_page, self.TASKS)["byKind"] == {"jobs": 2, "continuous": 1}

    def test_terminal_jobs_are_excluded_from_both(self, shared_page: Page):
        counts = self._counts(
            shared_page,
            "{ a: { instance: { status: 'completed' } }, b: { instance: { status: 'error' } } }",
        )
        assert counts["live"] == 0
        assert counts["advancing"] == 0
