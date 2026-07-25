"""
Tests for the default-zoom margin (viewport.js FIT_MARGIN) and the dashed
image-bounds border (manager.js setCanvasViewport) added so users can always
see the true edges of the image, even at fit-to-view zoom or after a
manual zoom/pan.

Run with:
    pytest playwright_tests/test_canvas_border_zoom.py -p no:xdist --override-ini="addopts=" -v
"""
from playwright.sync_api import Page

# Creates a runtime + canvas + blank 400x300 image loaded via setImage
# (mirrors the _SETUP helper in test_canvas_interaction.py).
_SETUP = """
    const div = document.createElement('div');
    div.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
    document.body.appendChild(div);

    const runtime = createImageTaggingRuntime();
    runtime.setCanvasContainer(div);

    const imgCanvas = document.createElement('canvas');
    imgCanvas.width = 400; imgCanvas.height = 300;
    const img = new Image();
    img.src = imgCanvas.toDataURL();
    await new Promise(r => { img.onload = r; });

    runtime.setImage({
        uuid: 'img', filename: 'test.png', image_path: '/test.png',
        img, bboxes: new Map(), dirty: false,
    });
"""


class TestImageBorder:
    def test_border_created_on_image_load(self, shared_page: Page):
        page = shared_page
        count = page.evaluate(f"""async () => {{
            const {{ createImageTaggingRuntime }} = await import('/app-static/js/app/image-tagging/runtime.js');
            {_SETUP}
            return runtime.canvas.layer.find('.image-border').length;
        }}""")
        assert count == 1

    def test_border_matches_background_image_bounds(self, shared_page: Page):
        page = shared_page
        result = page.evaluate(f"""async () => {{
            const {{ createImageTaggingRuntime }} = await import('/app-static/js/app/image-tagging/runtime.js');
            {_SETUP}
            const border = runtime.canvas.layer.findOne('.image-border');
            const bg = runtime.canvas.backgroundImage;
            return {{
                borderX: border.x(), borderY: border.y(),
                borderW: border.width(), borderH: border.height(),
                bgX: bg.x(), bgY: bg.y(),
                bgW: bg.width(), bgH: bg.height(),
                dash: border.dash(),
            }};
        }}""")
        assert result["borderX"] == result["bgX"]
        assert result["borderY"] == result["bgY"]
        assert result["borderW"] == result["bgW"]
        assert result["borderH"] == result["bgH"]
        assert result["dash"] == [6, 4]  # dashed, not a solid line

    def test_border_reused_not_duplicated_on_rezoom(self, shared_page: Page):
        """resetCanvasZoom (and any other viewport update) must reposition the
        existing border node rather than stacking a new one on every call."""
        page = shared_page
        count = page.evaluate(f"""async () => {{
            const {{ createImageTaggingRuntime }} = await import('/app-static/js/app/image-tagging/runtime.js');
            const {{ resetCanvasZoom }} = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            {_SETUP}
            resetCanvasZoom(runtime.canvas);
            resetCanvasZoom(runtime.canvas);
            return runtime.canvas.layer.find('.image-border').length;
        }}""")
        assert count == 1

    def test_border_stays_below_bbox_views(self, shared_page: Page):
        """The border must render above the background image but below any
        bbox views, so it never visually obscures selection handles."""
        page = shared_page
        result = page.evaluate(f"""async () => {{
            const {{ createImageTaggingRuntime }} = await import('/app-static/js/app/image-tagging/runtime.js');
            const {{ createUIBBbox }} = await import('/app-static/js/shared/domain/image/mapper.js');
            const {{ BboxView }} = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');
            {_SETUP}

            const b = createUIBBbox({{ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 }});
            const bv = new BboxView(b, runtime.canvas.viewport);
            runtime.canvas.layer.add(bv);

            const bgIndex = runtime.canvas.backgroundImage.zIndex();
            const borderIndex = runtime.canvas.layer.findOne('.image-border').zIndex();
            const bboxIndex = bv.zIndex();
            return {{ bgIndex, borderIndex, bboxIndex }};
        }}""")
        assert result["bgIndex"] < result["borderIndex"] < result["bboxIndex"]


class TestFitToViewMargin:
    def test_default_zoom_leaves_margin_on_load(self, shared_page: Page):
        """The image drawn at default (fit-to-view) zoom must not fill the
        container exactly on either axis, so bboxes at the true image edges
        stay visible instead of running under the container/status bar."""
        page = shared_page
        result = page.evaluate(f"""async () => {{
            const {{ createImageTaggingRuntime }} = await import('/app-static/js/app/image-tagging/runtime.js');
            {_SETUP}
            const bg = runtime.canvas.backgroundImage;
            return {{
                offsetX: bg.x(), offsetY: bg.y(),
                renderedW: bg.width(), renderedH: bg.height(),
                containerW: div.clientWidth, containerH: div.clientHeight,
            }};
        }}""")
        assert result["offsetX"] > 0
        assert result["offsetY"] > 0
        assert result["renderedW"] < result["containerW"]
        assert result["renderedH"] < result["containerH"]
