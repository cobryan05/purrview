"""
Tests for canvas/state.js (createCanvasState) and canvas/manager.js (rerenderBboxViews).
Phase 2 canvas tests — require Konva to be available as a global (navigate to base_url).

Uses the shared-page fixture with the autouse _canvas_dom_cleanup fixture
(conftest.py) to remove fixed-position test containers after each test, so
Konva stages don't accumulate across tests.

Run with:
    pytest tests/playwright/test_canvas_state.py -p no:xdist --override-ini="addopts=" -v
"""
import pytest
from playwright.sync_api import Page

# ---------------------------------------------------------------------------
# createCanvasState
# ---------------------------------------------------------------------------

class TestCreateCanvasState:
    def test_returns_required_fields(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCanvasState } = await import('/app-static/js/app/image-tagging/canvas/state.js');

            const div = document.createElement('div');
            div.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
            document.body.appendChild(div);

            const state = createCanvasState(div);

            return {
                hasStage: !!state.stage,
                hasLayer: !!state.layer,
                hasTransformer: !!state.transformer,
                hasViewport: !!state.viewport,
                hasInteractionState: !!state.interactionState,
                containerIsDiv: state.container === div,
            };
        }""")
        assert result == {
            'hasStage': True,
            'hasLayer': True,
            'hasTransformer': True,
            'hasViewport': True,
            'hasInteractionState': True,
            'containerIsDiv': True,
        }

    def test_stage_sized_to_container(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCanvasState } = await import('/app-static/js/app/image-tagging/canvas/state.js');

            const div = document.createElement('div');
            div.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
            document.body.appendChild(div);

            const state = createCanvasState(div);

            return { w: state.stage.width(), h: state.stage.height() };
        }""")
        assert result['w'] == 800
        assert result['h'] == 600

    def test_transformer_config(self, shared_page: Page):
        """Transformer must never rotate, must have correct anchor/ratio settings."""
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCanvasState } = await import('/app-static/js/app/image-tagging/canvas/state.js');

            const div = document.createElement('div');
            div.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
            document.body.appendChild(div);

            const { transformer } = createCanvasState(div);

            return {
                rotateEnabled: transformer.rotateEnabled(),
                anchorSize: transformer.anchorSize(),
                keepRatio: transformer.keepRatio(),
            };
        }""")
        assert result['rotateEnabled'] is False
        assert result['anchorSize'] == 10
        assert result['keepRatio'] is False

    def test_viewport_initialized_to_zeros(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCanvasState } = await import('/app-static/js/app/image-tagging/canvas/state.js');

            const div = document.createElement('div');
            div.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
            document.body.appendChild(div);

            return createCanvasState(div).viewport;
        }""")
        assert result == {
            'offsetX': 0, 'offsetY': 0, 'scale': 1.0,
            'imageHeight': 0, 'imageWidth': 0,
        }

    def test_interaction_state_initialized_to_null_and_false(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCanvasState } = await import('/app-static/js/app/image-tagging/canvas/state.js');

            const div = document.createElement('div');
            div.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
            document.body.appendChild(div);

            const ist = createCanvasState(div).interactionState;

            return {
                pendingDraggedBbox: ist.pendingDraggedBbox,
                selectedBbox: ist.selectedBbox,
                isPanning: ist.isPanning,
                startPos: ist.startPos,
                lastClickTime: ist.lastClickTime,
                lastWasDoubleClick: ist.lastWasDoubleClick,
                cycleIndex: ist.cycleIndex,
                cycleAnchor: ist.cycleAnchor,
            };
        }""")
        assert result['pendingDraggedBbox'] is None
        assert result['selectedBbox'] is None
        assert result['isPanning'] is False
        assert result['startPos'] is None
        assert result['lastClickTime'] == 0
        assert result['lastWasDoubleClick'] is False
        assert result['cycleIndex'] == 0
        assert result['cycleAnchor'] is None


# ---------------------------------------------------------------------------
# rerenderBboxViews
# ---------------------------------------------------------------------------

class TestRerenderBboxViews:
    def test_creates_one_view_per_bbox(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCanvasState } = await import('/app-static/js/app/image-tagging/canvas/state.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');

            const div = document.createElement('div');
            div.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
            document.body.appendChild(div);

            const canvas = createCanvasState(div);
            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            canvas.viewport = vp;

            const bboxes = new Map([
                ['b1', createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 })],
                ['b2', createUIBBbox({ uuid: 'b2', x: 0.5, y: 0.5, width: 0.2, height: 0.2 })],
            ]);
            rerenderBboxViews(canvas, bboxes, vp);

            return canvas.layer.find('.bbox').length;
        }""")
        assert result == 2

    def test_destroys_old_views_before_new_render(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCanvasState } = await import('/app-static/js/app/image-tagging/canvas/state.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');

            const div = document.createElement('div');
            div.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
            document.body.appendChild(div);

            const canvas = createCanvasState(div);
            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            canvas.viewport = vp;

            // First render: 2 bboxes
            rerenderBboxViews(canvas, new Map([
                ['b1', createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 })],
                ['b2', createUIBBbox({ uuid: 'b2', x: 0.5, y: 0.5, width: 0.2, height: 0.2 })],
            ]), vp);
            const afterFirst = canvas.layer.find('.bbox').length;

            // Second render with only 1 bbox — old 2 must be destroyed
            rerenderBboxViews(canvas, new Map([
                ['b3', createUIBBbox({ uuid: 'b3', x: 0.3, y: 0.3, width: 0.1, height: 0.1 })],
            ]), vp);
            const afterSecond = canvas.layer.find('.bbox').length;

            return { afterFirst, afterSecond };
        }""")
        assert result == {'afterFirst': 2, 'afterSecond': 1}

    def test_empty_map_removes_all_views(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCanvasState } = await import('/app-static/js/app/image-tagging/canvas/state.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');

            const div = document.createElement('div');
            div.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
            document.body.appendChild(div);

            const canvas = createCanvasState(div);
            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            canvas.viewport = vp;

            rerenderBboxViews(canvas, new Map([
                ['b1', createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 })],
            ]), vp);

            rerenderBboxViews(canvas, new Map(), vp);  // clear all

            return canvas.layer.find('.bbox').length;
        }""")
        assert result == 0

    def test_clears_transformer_selection_on_rerender(self, shared_page: Page):
        """Transformer must be cleared so the selection handles don't outlive the view."""
        page = shared_page
        result = page.evaluate("""async () => {
            const { createCanvasState } = await import('/app-static/js/app/image-tagging/canvas/state.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { BboxView } = await import('/app-static/js/app/image-tagging/canvas/views/bboxView.js');

            const div = document.createElement('div');
            div.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
            document.body.appendChild(div);

            const canvas = createCanvasState(div);
            const vp = { offsetX: 0, offsetY: 0, scale: 1, imageWidth: 400, imageHeight: 300 };
            canvas.viewport = vp;

            // Create a view and pretend it's selected
            const b = createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 });
            const bv = new BboxView(b, vp);
            canvas.layer.add(bv);
            canvas.transformer.nodes([bv]);

            // Rerender should clear the transformer
            rerenderBboxViews(canvas, new Map(), vp);

            return canvas.transformer.nodes().length;
        }""")
        assert result == 0
