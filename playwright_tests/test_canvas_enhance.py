"""
Tests for canvas/enhance.js — the display-only image enhancement modes offered
by the ✨ toolbar button on the main and modal canvases.

The pixel maths is exercised directly against synthetic ImageData (no image
fetch needed); the last class checks the toolbar button actually cycles.

Run with:
    pytest playwright_tests/test_canvas_enhance.py -p no:asyncio -v
"""

import pytest
from playwright.sync_api import Page

ENHANCE = "/app-static/js/app/image-tagging/canvas/enhance.js"

# JS helpers injected into each evaluate() that needs them.
JS_HELPERS = """
    // Build an ImageData whose pixels come from fn(x, y) -> luminance byte.
    const makeGray = (w, h, fn) => {
        const img = new ImageData(w, h);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const p = (y * w + x) * 4;
                const v = fn(x, y);
                img.data[p] = img.data[p + 1] = img.data[p + 2] = v;
                img.data[p + 3] = 255;
            }
        }
        return img;
    };
    const lumaRange = (img) => {
        let lo = 255, hi = 0;
        for (let p = 0; p < img.data.length; p += 4) {
            const v = img.data[p];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        return { lo, hi, span: hi - lo };
    };
"""


# ---------------------------------------------------------------------------
# Mode list / cycling
# ---------------------------------------------------------------------------


class TestModeCycle:
    def test_cycles_through_every_mode_and_wraps(self, shared_page: Page):
        result = shared_page.evaluate(
            """async (mod) => {
            const { ENHANCE_MODES, nextEnhanceModeId, DEFAULT_ENHANCE_MODE } = await import(mod);

            const visited = [];
            let id = DEFAULT_ENHANCE_MODE;
            for (let i = 0; i < ENHANCE_MODES.length; i++) {
                visited.push(id);
                id = nextEnhanceModeId(id);
            }
            return {
                defaultMode: DEFAULT_ENHANCE_MODE,
                visited,
                allIds: ENHANCE_MODES.map(m => m.id),
                wrapped: id,
            };
        }""",
            ENHANCE,
        )
        assert result["defaultMode"] == "none"
        # Every mode is reachable, and one full lap returns to the start.
        assert sorted(result["visited"]) == sorted(result["allIds"])
        assert result["wrapped"] == "none"

    def test_unknown_mode_falls_back_to_off(self, shared_page: Page):
        result = shared_page.evaluate(
            """async (mod) => {
            const { getEnhanceMode } = await import(mod);
            return getEnhanceMode('not-a-mode').id;
        }""",
            ENHANCE,
        )
        assert result == "none"


# ---------------------------------------------------------------------------
# Point operations
# ---------------------------------------------------------------------------


class TestAutoLevels:
    def test_stretches_low_contrast_frame_to_full_range(self, shared_page: Page):
        result = shared_page.evaluate(
            """async (mod) => {
            %s
            const { autoLevels } = await import(mod);
            // A dark, low-contrast ramp: everything between 40 and 71.
            const img = makeGray(64, 64, (x) => 40 + (x >> 1));
            const before = lumaRange(img);
            autoLevels(img.data);
            return { before, after: lumaRange(img) };
        }""" % JS_HELPERS,
            ENHANCE,
        )
        assert result["before"]["span"] == 31
        assert result["after"]["lo"] == 0
        assert result["after"]["hi"] == 255

    def test_flat_frame_is_left_alone(self, shared_page: Page):
        # hi <= lo means there is nothing to stretch — must not divide by zero
        # and blow the frame out to noise.
        result = shared_page.evaluate(
            """async (mod) => {
            %s
            const { autoLevels } = await import(mod);
            const img = makeGray(32, 32, () => 17);
            autoLevels(img.data);
            return lumaRange(img);
        }""" % JS_HELPERS,
            ENHANCE,
        )
        assert result == {"lo": 17, "hi": 17, "span": 0}


class TestGamma:
    def test_lifts_midtones_and_pins_the_endpoints(self, shared_page: Page):
        result = shared_page.evaluate(
            """async (mod) => {
            %s
            const { applyGamma } = await import(mod);
            const img = makeGray(3, 1, (x) => [0, 64, 255][x]);
            applyGamma(img.data, 0.45);
            return [img.data[0], img.data[4], img.data[8]];
        }""" % JS_HELPERS,
            ENHANCE,
        )
        black, mid, white = result
        assert black == 0
        assert white == 255
        assert mid > 64  # shadows opened up


class TestClahe:
    def test_expands_local_contrast(self, shared_page: Page):
        result = shared_page.evaluate(
            """async (mod) => {
            %s
            const { clahe } = await import(mod);
            // Each 8x8 tile holds the same narrow 20..27 local ramp, so a
            // *global* stretch has little to work with but a local one does.
            const img = makeGray(64, 64, (x) => 20 + (x %% 8));
            const before = lumaRange(img);
            clahe(img, 8, 3.0);
            return { before, after: lumaRange(img) };
        }""" % JS_HELPERS,
            ENHANCE,
        )
        assert result["before"]["span"] == 7
        # Contrast-limited, so not a full 0..255 blowout — but clearly expanded.
        assert result["after"]["span"] >= result["before"]["span"] * 3

    def test_preserves_hue(self, shared_page: Page):
        result = shared_page.evaluate(
            """async (mod) => {
            const { clahe } = await import(mod);
            const w = 32, h = 32;
            const img = new ImageData(w, h);
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const p = (y * w + x) * 4;
                    // Strongly red-tinted, and dark.
                    img.data[p] = 30 + (x % 8);
                    img.data[p + 1] = 10;
                    img.data[p + 2] = 10;
                    img.data[p + 3] = 255;
                }
            }
            clahe(img, 8, 3.0);
            const p = (16 * w + 16) * 4;
            return { r: img.data[p], g: img.data[p + 1], b: img.data[p + 2] };
        }""",
            ENHANCE,
        )
        # The red cast survives: scaling by the luminance ratio keeps channel
        # ordering intact rather than washing out to grey.
        assert result["r"] > result["g"]
        assert result["g"] == result["b"]


# ---------------------------------------------------------------------------
# Edge modes
# ---------------------------------------------------------------------------


class TestEdgeOverlay:
    def test_marks_edges_in_cyan_and_keeps_the_scene(self, shared_page: Page):
        result = shared_page.evaluate(
            """async (mod) => {
            %s
            const { edgeOverlay } = await import(mod);
            const w = 64, h = 64;
            const inside = (x, y) => x >= 16 && x < 48 && y >= 16 && y < 48;
            const img = makeGray(w, h, (x, y) => (inside(x, y) ? 200 : 40));
            edgeOverlay(img);

            const at = (x, y) => {
                const p = (y * w + x) * 4;
                return { r: img.data[p], g: img.data[p + 1], b: img.data[p + 2] };
            };
            return { edge: at(16, 32), interior: at(32, 32), outside: at(4, 4) };
        }""" % JS_HELPERS,
            ENHANCE,
        )
        # On the rectangle's boundary the overlay pushes green/blue well past red.
        assert result["edge"]["g"] > result["edge"]["r"] + 50
        assert result["edge"]["g"] == result["edge"]["b"]
        # Flat areas stay neutral, and the scene is still visible underneath
        # (dimmed, not blacked out) so you can see where to draw a box.
        assert result["interior"]["g"] == result["interior"]["r"]
        assert result["interior"]["r"] > 0


class TestCanny:
    def test_outlines_a_rectangle_and_leaves_flat_areas_black(self, shared_page: Page):
        result = shared_page.evaluate(
            """async (mod) => {
            %s
            const { canny } = await import(mod);
            const w = 64, h = 64;
            const inside = (x, y) => x >= 16 && x < 48 && y >= 16 && y < 48;
            const img = makeGray(w, h, (x, y) => (inside(x, y) ? 200 : 40));
            canny(img);

            const lit = (x, y) => img.data[(y * w + x) * 4] > 0;
            let total = 0;
            for (let p = 0; p < img.data.length; p += 4) if (img.data[p] > 0) total++;

            // Look for a lit pixel within +/-2px of each nominal border.
            const near = (cx, cy) => {
                for (let d = -2; d <= 2; d++) {
                    if (lit(cx + d, cy) || lit(cx, cy + d)) return true;
                }
                return false;
            };
            return {
                left: near(16, 32), right: near(47, 32),
                top: near(32, 16), bottom: near(32, 47),
                interior: lit(32, 32), background: lit(4, 4),
                total,
            };
        }""" % JS_HELPERS,
            ENHANCE,
        )
        assert result["left"] and result["right"] and result["top"] and result["bottom"]
        assert not result["interior"]
        assert not result["background"]
        # Thin outlines, not a filled blob: the rectangle's perimeter is ~128px,
        # so anything near the 4096-pixel frame size means NMS/hysteresis broke.
        assert 50 < result["total"] < 600

    def test_binary_output_only(self, shared_page: Page):
        result = shared_page.evaluate(
            """async (mod) => {
            %s
            const { canny } = await import(mod);
            const img = makeGray(48, 48, (x, y) => (x < 24 ? 20 : 220));
            canny(img);
            const seen = new Set();
            for (let p = 0; p < img.data.length; p += 4) seen.add(img.data[p]);
            return [...seen].sort((a, b) => a - b);
        }""" % JS_HELPERS,
            ENHANCE,
        )
        assert result == [0, 255]

    def test_flat_frame_produces_no_edges(self, shared_page: Page):
        # No gradients at all — the percentile threshold must not degenerate
        # into "everything is an edge".
        result = shared_page.evaluate(
            """async (mod) => {
            %s
            const { canny } = await import(mod);
            const img = makeGray(32, 32, () => 12);
            canny(img);
            let total = 0;
            for (let p = 0; p < img.data.length; p += 4) if (img.data[p] > 0) total++;
            return total;
        }""" % JS_HELPERS,
            ENHANCE,
        )
        assert result == 0


# ---------------------------------------------------------------------------
# renderEnhanced / caching
# ---------------------------------------------------------------------------


class TestRenderEnhanced:
    def test_off_returns_the_original_element_and_other_modes_are_cached(self, shared_page: Page):
        result = shared_page.evaluate(
            """async (mod) => {
            const { renderEnhanced } = await import(mod);

            // 8x8 red PNG via a canvas -> data URL (same-origin, so getImageData
            // on it is not tainted).
            const src = document.createElement('canvas');
            src.width = src.height = 8;
            const sctx = src.getContext('2d');
            sctx.fillStyle = '#301010';
            sctx.fillRect(0, 0, 8, 8);

            const img = new Image();
            img.src = src.toDataURL();
            await img.decode();

            const off = renderEnhanced(img, 'none');
            const a = renderEnhanced(img, 'levels');
            const b = renderEnhanced(img, 'levels');
            const c = renderEnhanced(img, 'canny');

            return {
                offIsOriginal: off === img,
                cached: a === b,
                distinctModes: a !== c,
                dims: [a.width, a.height],
            };
        }""",
            ENHANCE,
        )
        assert result["offIsOriginal"] is True
        assert result["cached"] is True
        assert result["distinctModes"] is True
        assert result["dims"] == [8, 8]


# ---------------------------------------------------------------------------
# Toolbar button
# ---------------------------------------------------------------------------


class TestToolbarButton:
    def test_button_cycles_modes_and_reports_them(self, shared_page: Page):
        result = shared_page.evaluate(
            """async (mod) => {
            const { ENHANCE_MODES } = await import(mod);
            const { createImageTaggingRuntime } = await import(
                '/app-static/js/app/image-tagging/runtime.js');
            const { init } = await import('/app-static/js/app/image-tagging/toolbar.js');

            const host = document.createElement('div');
            host.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
            document.body.appendChild(host);
            const canvasDiv = document.createElement('div');
            canvasDiv.style.cssText = 'width:800px;height:500px;';
            const bar = document.createElement('div');
            host.append(canvasDiv, bar);

            const runtime = createImageTaggingRuntime();
            runtime.setCanvasContainer(canvasDiv);
            runtime.canvas.enhanceMode = 'none';
            const handle = init(runtime, bar, { buttons: ['enhance'] });

            const btn = bar.querySelector('[data-toolbar-key="enhance"]');
            const seen = [{ mode: btn.dataset.enhanceMode, label: btn.textContent }];
            for (let i = 0; i < ENHANCE_MODES.length; i++) {
                btn.click();
                seen.push({ mode: btn.dataset.enhanceMode, label: btn.textContent });
            }
            const selectedWhileOn = btn.classList.contains('selected');

            handle.destroy();
            runtime.destroyCanvas();
            host.remove();
            return { seen, modeIds: ENHANCE_MODES.map(m => m.id), selectedWhileOn };
        }""",
            ENHANCE,
        )
        modes = [s["mode"] for s in result["seen"]]
        # Starts off, walks the whole cycle, and lands back on off.
        assert modes[0] == "none"
        assert modes[:-1] == result["modeIds"]
        assert modes[-1] == "none"
        # The label always names the active mode.
        assert result["seen"][1]["label"].endswith("Levels")
        # Back at 'none' after a full lap, so the highlight is off again.
        assert result["selectedWhileOn"] is False


# ---------------------------------------------------------------------------
# Locked canvases (read-only previews)
# ---------------------------------------------------------------------------


class TestLockedCanvas:
    def test_locked_canvas_ignores_cycle_and_keeps_session_default(self, shared_page: Page):
        """A canvas created with lockEnhance has no enhance control.

        The enhance hotkey is global (initCanvasCore attaches it to every canvas,
        including read-only status-widget previews), so without the lock, pressing
        it while such a preview had focus would both enhance a widget nobody asked
        to enhance and change the default mode for every canvas created afterwards.
        """
        result = shared_page.evaluate(
            """async (mod) => {
            const { cycleCanvasEnhancement, defaultEnhanceMode } = await import(mod);

            const before = defaultEnhanceMode();
            const locked = { canvas: { enhanceMode: 'none', enhanceLocked: true } };
            const returned = cycleCanvasEnhancement(locked);
            return {
                returned,
                modeAfter: locked.canvas.enhanceMode,
                defaultBefore: before,
                defaultAfter: defaultEnhanceMode(),
            };
        }""",
            ENHANCE,
        )
        assert result["returned"] is None
        assert result["modeAfter"] == "none"
        assert result["defaultAfter"] == result["defaultBefore"]

    def test_unlocked_canvas_still_cycles(self, shared_page: Page):
        """The lock is opt-in — ordinary canvases keep the sticky-mode behaviour."""
        result = shared_page.evaluate(
            """async (mod) => {
            const { cycleCanvasEnhancement, defaultEnhanceMode, nextEnhanceModeId,
                    ENHANCE_MODES, DEFAULT_ENHANCE_MODE } = await import(mod);
            const canvas = { canvas: { enhanceMode: DEFAULT_ENHANCE_MODE } };
            const returned = cycleCanvasEnhancement(canvas);
            const out = { returned, expected: nextEnhanceModeId(DEFAULT_ENHANCE_MODE), def: defaultEnhanceMode() };
            // _lastMode is module state on a shared page: walk the rest of the
            // cycle so the session default is back where the next test found it.
            for (let i = 1; i < ENHANCE_MODES.length; i++) cycleCanvasEnhancement(canvas);
            return out;
        }""",
            ENHANCE,
        )
        assert result["returned"] == result["expected"]
        assert result["def"] == result["expected"]
