"""
Regression test for the review-modal "first image zoomed in too far" bug.

The canvas modal (review/main.js) creates an empty `.canvas-modal-toolbar`
div, appends the modal to the DOM, then immediately loads the first image
onto the canvas (measuring `.canvas-modal-canvas` for the fit-to-view
viewport) *before* `initToolbar()` populates the toolbar with buttons and
`initSplitResize()` sets the sidebar's persisted width. Since the toolbar
grows taller once populated, the content row (and therefore the canvas)
has less height than it did at the moment of the first measurement, so the
first image's fit-to-view scale is too large (zoomed in too far). Scrubbing
to the next/prev image re-measures the now-stable layout and looks correct.

The fix (review/main.js) calls `resetCanvasZoom()` right after the toolbar
and sidebar-resize setup, once the modal chrome has reached its final size.

This test rebuilds the same DOM skeleton (same CSS classes, so the real
app.css flex rules apply) to reproduce the timing gap deterministically,
without needing the full review panel/queue machinery.

Run with:
    pytest playwright_tests/test_canvas_modal_zoom.py -p no:xdist --override-ini="addopts=" -v
"""

from playwright.sync_api import Page

# Builds a `.canvas-modal-dialog` skeleton (header/toolbar/content/sidebar/footer)
# using the real app.css classes, with an EMPTY toolbar — mirrors review/main.js's
# DOM before initToolbar() populates it. Loads a blank 400x300 image onto
# canvasWrap, exactly as the first `instance.loadImage()` call does.
_SETUP = """
    const overlay = document.createElement('div');
    overlay.className = 'app-modal-overlay';
    document.body.appendChild(overlay);

    const dialog = document.createElement('div');
    dialog.className = 'canvas-modal-dialog';
    overlay.appendChild(dialog);

    const header = document.createElement('div');
    header.className = 'canvas-modal-header';
    header.innerHTML = '<span class="canvas-modal-title">test</span>';

    const toolbarEl = document.createElement('div');
    toolbarEl.className = 'canvas-modal-toolbar';
    // Empty, as it is at the moment createCanvasInstance()/loadImage() first run.

    const contentArea = document.createElement('div');
    contentArea.className = 'canvas-modal-content';

    const canvasWrap = document.createElement('div');
    canvasWrap.className = 'canvas-modal-canvas';

    const resizeHandleEl = document.createElement('div');
    resizeHandleEl.className = 'split-resize-handle';

    const sidebarEl = document.createElement('div');
    sidebarEl.className = 'canvas-modal-sidebar';

    const footer = document.createElement('div');
    footer.className = 'canvas-modal-footer';
    footer.innerHTML = '<button>Cancel</button>';

    contentArea.append(canvasWrap, resizeHandleEl, sidebarEl);
    dialog.append(header, toolbarEl, contentArea, footer);

    const runtime = createImageTaggingRuntime();
    runtime.setCanvasContainer(canvasWrap);

    const imgCanvas = document.createElement('canvas');
    imgCanvas.width = 400; imgCanvas.height = 300;
    const img = new Image();
    img.src = imgCanvas.toDataURL();
    await new Promise(r => { img.onload = r; });

    // First load: happens while the toolbar is still empty, exactly like the
    // real `await instance.loadImage(img.filename)` call in review/main.js.
    runtime.setImage({
        uuid: 'img', filename: 'test.png', image_path: '/test.png',
        img, bboxes: new Map(), dirty: false,
    });
"""

# Mimics initToolbar() populating the toolbar with real buttons (making it
# taller than the empty placeholder) and initSplitResize() setting the
# sidebar's persisted width — the two layout-affecting steps that, in the
# real modal, run AFTER the first loadImage() call.
_SIMULATE_CHROME_SETTLING = """
    toolbarEl.style.cssText = 'display:flex;gap:0.5em;padding:5px;width:100%;box-sizing:border-box;';
    for (const label of ['Undo', 'Redo', 'Select', 'BBox', 'Clear', 'Delete']) {
        const btn = document.createElement('button');
        btn.textContent = label;
        btn.style.cssText = 'padding:8px 16px;font-size:14px;';
        toolbarEl.appendChild(btn);
    }
    sidebarEl.style.width = '220px';
"""


class TestModalFirstImageZoom:
    def test_toolbar_populating_changes_available_canvas_height(self, shared_page: Page):
        """Sanity check that the timing gap is real: populating the toolbar
        measurably shrinks canvasWrap's height, which is why measuring the
        container before that happens gives an oversized (too-zoomed) fit."""
        page = shared_page
        result = page.evaluate(f"""async () => {{
            const {{ createImageTaggingRuntime }} = await import('/app-static/js/app/image-tagging/runtime.js');
            {_SETUP}
            const heightBefore = canvasWrap.clientHeight;
            {_SIMULATE_CHROME_SETTLING}
            const heightAfter = canvasWrap.clientHeight;
            return {{ heightBefore, heightAfter }};
        }}""")
        assert result["heightAfter"] < result["heightBefore"]

    def test_reset_canvas_zoom_after_chrome_settles_matches_final_container(self, shared_page: Page):
        """The fix: after the toolbar/sidebar reach their final size,
        resetCanvasZoom() must recompute a scale matching a fresh
        computeViewport() call against the now-current container size —
        not the stale, oversized one from the first load."""
        page = shared_page
        result = page.evaluate(f"""async () => {{
            const {{ createImageTaggingRuntime }} = await import('/app-static/js/app/image-tagging/runtime.js');
            const {{ resetCanvasZoom }} = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const {{ computeViewport }} = await import('/app-static/js/app/image-tagging/canvas/viewport.js');
            {_SETUP}

            const scaleBeforeFix = runtime.canvas.viewport.scale;

            {_SIMULATE_CHROME_SETTLING}
            resetCanvasZoom(runtime.canvas);

            const scaleAfterFix = runtime.canvas.viewport.scale;
            const expected = computeViewport(
                canvasWrap.clientWidth, canvasWrap.clientHeight, img.width, img.height,
            ).scale;

            return {{ scaleBeforeFix, scaleAfterFix, expected }};
        }}""")
        # Without the fix, the stale first-load scale is larger (too zoomed in)
        # than what the final, settled container size warrants.
        assert result["scaleBeforeFix"] > result["expected"]
        # With the fix, resetCanvasZoom's scale matches the final expected fit.
        assert result["scaleAfterFix"] == result["expected"]
