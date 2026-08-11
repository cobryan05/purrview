"""
Tests for canvas/views/bboxView.js — BboxView construction, viewport updates,
appearance changes, icon computation, and drag/transform normalization.
Phase 2 canvas tests — require Konva as a global (navigate to base_url).

Run with:
    pytest tests/playwright/test_bboxview.py -p no:xdist --override-ini="addopts=" -v
"""

import pytest
from playwright.sync_api import Page

# ---------------------------------------------------------------------------
# Helper — JS snippet that creates a bare Konva stage+layer inside the page.
# Included at the start of every evaluate block (after imports).
# ---------------------------------------------------------------------------
_STAGE_SETUP = """
    const div = document.createElement('div');
    div.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
    document.body.appendChild(div);
    const stage = new Konva.Stage({ container: div, width: 800, height: 600 });
    const layer = new Konva.Layer();
    stage.add(layer);
"""


# ---------------------------------------------------------------------------
# Constructor — position, size, name, id, stroke colour
# ---------------------------------------------------------------------------


class TestBboxViewConstructor:
    def test_positions_correctly_with_viewport(self, shared_page: Page):
        """x = uiBBox.x * imageWidth * scale + offsetX  (and similar for y/w/h)."""
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            const vp = { offsetX: 10, offsetY: 20, scale: 2, imageWidth: 400, imageHeight: 300 };
            // x = 0.25 * 400 * 2 + 10 = 210
            // y = 0.5  * 300 * 2 + 20 = 320
            // w = 0.5  * 400 * 2      = 400
            // h = 0.5  * 300 * 2      = 300
            const uiBBox = { uuid: 'u', x: 0.25, y: 0.5, width: 0.5, height: 0.5, label: null, tags: [] };
            const view = new BboxView(uiBBox, vp);
            layer.add(view);

            return { x: view.x(), y: view.y(), w: view.width(), h: view.height() };
        }"""
        )
        assert result["x"] == pytest.approx(210)
        assert result["y"] == pytest.approx(320)
        assert result["w"] == pytest.approx(400)
        assert result["h"] == pytest.approx(300)

    def test_name_is_bbox_and_id_is_uuid(self, shared_page: Page):
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            const uiBBox = { uuid: 'my-id', x: 0, y: 0, width: 0.5, height: 0.5, label: null, tags: [] };
            const view = new BboxView(uiBBox, vp);

            return { name: view.name(), id: view.id() };
        }"""
        )
        assert result["name"] == "bbox"
        assert result["id"] == "my-id"

    def test_label_color_used_for_rect_stroke(self, shared_page: Page):
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            const uiBBox = {
                uuid: 'u', x: 0, y: 0, width: 0.5, height: 0.5,
                label: { uuid: 'l', text: 'Cat', color: '#ff0000' },
                tags: [],
            };
            const view = new BboxView(uiBBox, vp);
            layer.add(view);

            return view.findOne('.box').stroke();
        }"""
        )
        assert result == "#ff0000"

    def test_grey_stroke_when_no_label(self, shared_page: Page):
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            const uiBBox = { uuid: 'u', x: 0, y: 0, width: 0.5, height: 0.5, label: null, tags: [] };
            const view = new BboxView(uiBBox, vp);
            layer.add(view);

            return view.findOne('.box').stroke();
        }"""
        )
        assert result == "grey"

    def test_icon_text_built_from_tags(self, shared_page: Page):
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            const uiBBox = {
                uuid: 'u', x: 0, y: 0, width: 0.5, height: 0.5, label: null,
                tags: [{ uuid: 't1', icon: '🐈' }, { uuid: 't2', icon: '✓' }],
            };
            const view = new BboxView(uiBBox, vp);

            return view.findOne('.icon').text();
        }"""
        )
        assert result == "🐈✓"

    def test_duplicate_icons_deduplicated(self, shared_page: Page):
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            const uiBBox = {
                uuid: 'u', x: 0, y: 0, width: 0.5, height: 0.5, label: null,
                tags: [{ uuid: 't1', icon: '✓' }, { uuid: 't2', icon: '✓' }],
            };
            const view = new BboxView(uiBBox, vp);

            return view.findOne('.icon').text();
        }"""
        )
        assert result == "✓"

    def test_null_icons_excluded_from_icon_text(self, shared_page: Page):
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            const uiBBox = {
                uuid: 'u', x: 0, y: 0, width: 0.5, height: 0.5, label: null,
                tags: [{ uuid: 't1', icon: null }, { uuid: 't2', icon: '✓' }],
            };
            const view = new BboxView(uiBBox, vp);

            return view.findOne('.icon').text();
        }"""
        )
        assert result == "✓"

    def test_no_tags_gives_empty_icon_text(self, shared_page: Page):
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            const uiBBox = { uuid: 'u', x: 0, y: 0, width: 0.5, height: 0.5, label: null, tags: [] };
            const view = new BboxView(uiBBox, vp);

            return view.findOne('.icon').text();
        }"""
        )
        assert result == ""


# ---------------------------------------------------------------------------
# updateViewport
# ---------------------------------------------------------------------------


class TestBboxViewUpdateViewport:
    def test_repositions_to_new_viewport(self, shared_page: Page):
        """After zooming/panning the viewport, updateViewport moves the view."""
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            const vp1 = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            const uiBBox = { uuid: 'u', x: 0.25, y: 0.25, width: 0.5, height: 0.5, label: null, tags: [] };
            const view = new BboxView(uiBBox, vp1);
            layer.add(view);

            // New viewport: scale=2, offset=(100, 50)
            // x = 0.25 * 400 * 2 + 100 = 300
            // y = 0.25 * 300 * 2 + 50  = 200
            const vp2 = { offsetX: 100, offsetY: 50, scale: 2, imageWidth: 400, imageHeight: 300 };
            view.updateViewport(vp2);

            return { x: view.x(), y: view.y() };
        }"""
        )
        assert result["x"] == pytest.approx(300)
        assert result["y"] == pytest.approx(200)

    def test_resizes_to_new_scale(self, shared_page: Page):
        """Dimensions scale with the viewport."""
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            const vp1 = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            const uiBBox = { uuid: 'u', x: 0, y: 0, width: 0.5, height: 0.5, label: null, tags: [] };
            const view = new BboxView(uiBBox, vp1);
            layer.add(view);

            const vp2 = { offsetX: 0, offsetY: 0, scale: 2, imageWidth: 400, imageHeight: 300 };
            view.updateViewport(vp2);

            return { w: view.width(), h: view.height() };
        }"""
        )
        # w = 0.5 * 400 * 2 = 400, h = 0.5 * 300 * 2 = 300
        assert result["w"] == pytest.approx(400)
        assert result["h"] == pytest.approx(300)


# ---------------------------------------------------------------------------
# updateAppearance
# ---------------------------------------------------------------------------


class TestBboxViewUpdateAppearance:
    def test_updates_label_text_and_color(self, shared_page: Page):
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            const uiBBox = { uuid: 'u', x: 0, y: 0, width: 0.5, height: 0.5, label: null, tags: [] };
            const view = new BboxView(uiBBox, vp);
            layer.add(view);

            view.updateAppearance('Dog', '#0000ff');

            return {
                labelText: view.findOne('.label').text(),
                labelFill: view.findOne('.label').fill(),
                rectStroke: view.findOne('.box').stroke(),
            };
        }"""
        )
        assert result["labelText"] == "Dog"
        assert result["labelFill"] == "#0000ff"
        assert result["rectStroke"] == "#0000ff"


# ---------------------------------------------------------------------------
# refreshIcons
# ---------------------------------------------------------------------------


class TestBboxViewRefreshIcons:
    def test_updates_icon_text_from_new_tags(self, shared_page: Page):
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            const uiBBox = { uuid: 'u', x: 0, y: 0, width: 0.5, height: 0.5, label: null, tags: [] };
            const view = new BboxView(uiBBox, vp);
            layer.add(view);

            const before = view.findOne('.icon').text();
            view.refreshIcons([{ uuid: 't1', icon: '🐾' }]);
            const after = view.findOne('.icon').text();

            return { before, after };
        }"""
        )
        assert result["before"] == ""
        assert result["after"] == "🐾"

    def test_refreshIcons_deduplicates(self, shared_page: Page):
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            const uiBBox = { uuid: 'u', x: 0, y: 0, width: 0.5, height: 0.5, label: null, tags: [] };
            const view = new BboxView(uiBBox, vp);
            layer.add(view);

            view.refreshIcons([
                { uuid: 't1', icon: '★' },
                { uuid: 't2', icon: '★' },
                { uuid: 't3', icon: '✓' },
            ]);

            return view.findOne('.icon').text();
        }"""
        )
        assert result == "★✓"


# ---------------------------------------------------------------------------
# Drag / transform normalization (dragend / transformend handlers)
# ---------------------------------------------------------------------------


class TestBboxViewDragNormalization:
    def test_drag_updates_uibbox_coordinates(self, shared_page: Page):
        """Firing 'dragend' after moving the group normalises uiBBox.x/y back to [0,1] space."""
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            // viewport: offsetX=0, offsetY=0, scale=1, imageWidth=400, imageHeight=300
            // sw = 400, sh = 300
            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            const uiBBox = { uuid: 'u', x: 0.2, y: 0.2, width: 0.3, height: 0.3, label: null, tags: [] };
            const view = new BboxView(uiBBox, vp);
            layer.add(view);

            // Simulate a drag: move group to (200, 150)
            view.x(200);
            view.y(150);
            view.fire('dragend');

            // Expected: uiBBox.x = (200-0)/400 = 0.5, uiBBox.y = (150-0)/300 = 0.5
            return { x: uiBBox.x, y: uiBBox.y };
        }"""
        )
        assert result["x"] == pytest.approx(0.5)
        assert result["y"] == pytest.approx(0.5)

    def test_transform_resets_scale_and_updates_size(self, shared_page: Page):
        """After transform, scaleX/Y must be reset to 1 and width/height baked in."""
        page = shared_page
        result = page.evaluate(
            """async () => {
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            """
            + _STAGE_SETUP
            + """

            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            const uiBBox = { uuid: 'u', x: 0, y: 0, width: 0.3, height: 0.3, label: null, tags: [] };
            // initial: w = 0.3*400 = 120, h = 0.3*300 = 90
            const view = new BboxView(uiBBox, vp);
            layer.add(view);

            // Simulate Konva transform: scale up 2× wide, 1.5× tall
            view.scaleX(2);
            view.scaleY(1.5);
            view.fire('transformend');

            return {
                scaleX: view.scaleX(),
                scaleY: view.scaleY(),
                width: view.width(),
                height: view.height(),
                uiWidth: uiBBox.width,
                uiHeight: uiBBox.height,
            };
        }"""
        )
        assert result["scaleX"] == pytest.approx(1.0)
        assert result["scaleY"] == pytest.approx(1.0)
        # width = 120 * 2 = 240, height = 90 * 1.5 = 135
        assert result["width"] == pytest.approx(240)
        assert result["height"] == pytest.approx(135)
        # normalised: 240/400 = 0.6, 135/300 = 0.45
        assert result["uiWidth"] == pytest.approx(0.6)
        assert result["uiHeight"] == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Persisted confidence in the label text
# ---------------------------------------------------------------------------


def _label_text(page: Page, ui_bbox_js: str) -> str:
    return page.evaluate(
        """async (uiBBoxJson) => {
        const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

        """
        + _STAGE_SETUP
        + """

        const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
        const view = new BboxView(JSON.parse(uiBBoxJson), vp);
        layer.add(view);
        return view.findOne('Text').text();
    }""",
        ui_bbox_js,
    )


class TestBboxViewConfidenceLabel:
    """A reloaded, still-unverified box shows what the model thought of it.

    Only reloaded boxes carry `confidence`: a box straight out of live inference
    has the percentage baked into `label.text` by canvas/image.js and no
    `confidence` field, so exactly one of the two paths applies per box.
    """

    _BASE = {
        "uuid": "u",
        "x": 0,
        "y": 0,
        "width": 0.5,
        "height": 0.5,
        "label": {"uuid": "l", "text": "Cat", "color": "#ff0000"},
        "tags": [],
    }

    def _bbox(self, **overrides):
        import json

        return json.dumps({**self._BASE, **overrides})

    def test_confidence_appended_to_label(self, shared_page: Page):
        assert _label_text(shared_page, self._bbox(confidence=0.824)) == "Cat (82%)"

    def test_no_confidence_leaves_label_clean(self, shared_page: Page):
        assert _label_text(shared_page, self._bbox()) == "Cat"

    def test_zero_confidence_is_shown_not_dropped(self, shared_page: Page):
        """0 is a real confidence — a truthiness check here would hide it."""
        assert _label_text(shared_page, self._bbox(confidence=0)) == "Cat (0%)"

    def test_label_already_carrying_a_pct_is_not_double_suffixed(self, shared_page: Page):
        """Guards the seam between the two paths: image.js bakes the suffix into
        label.text for live detections and leaves `confidence` unset, so a box
        can never pick up "(82%) (82%)"."""
        text = _label_text(shared_page, self._bbox(label={"uuid": "l", "text": "Cat (82%)", "color": "#ff0000"}))
        assert text == "Cat (82%)"
        assert text.count("%") == 1

    def test_unlabelled_bbox_still_shows_loading(self, shared_page: Page):
        assert _label_text(shared_page, self._bbox(label=None)) == "Loading..."
