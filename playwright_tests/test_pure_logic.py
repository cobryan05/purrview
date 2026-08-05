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
