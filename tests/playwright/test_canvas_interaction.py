"""
Behavioural tests for the canvas interaction layer:
  - Drawing bounding boxes (mousedown → mousemove → mouseup)
  - Label assignment via double-click
  - Tag toggling via double-click
  - Select-tool click / clear-selection
  - Stacked-bbox cycling on double-click
  - Right-click context-menu tool switching
  - deleteSelected / clearAnnotations helpers

Handlers are called directly (not through DOM events) so we can control the
pointer position by setting stage._pointerPos and await the async handlers.
Label / tag cache is preloaded via the exported fetcher .set() API so no real
network call is needed.

UNVERIFIED_TAG_UUID = '00000000-0000-0000-0000-000000000001'

Run with:
    pytest tests/playwright/test_canvas_interaction.py -p no:xdist --override-ini="addopts=" -v
"""
import pytest
from playwright.sync_api import Page

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"



# ---------------------------------------------------------------------------
# JS helpers embedded in every evaluate block
# ---------------------------------------------------------------------------

# Creates a runtime + canvas + blank 400×300 image (scale=2, offset=0,0).
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

    // Helpers
    function setPtr(x, y) { runtime.canvas.stage._pointerPositions = [{ x, y }]; }
    function makeEvt(button = 0, clientX = 0, clientY = 0) {
        return { evt: { button, clientX, clientY, preventDefault: () => {} } };
    }
    async function drawBbox(x1, y1, x2, y2) {
        setPtr(x1, y1);
        await handleMouseDown(makeEvt(0, x1, y1), runtime);
        setPtr(x2, y2);
        await handleMouseMove(makeEvt(0, x2, y2), runtime);
        await handleMouseUp(makeEvt(0, x2, y2), runtime);
    }
"""


# ---------------------------------------------------------------------------
# stripUnverifiedTag (pure exported function)
# ---------------------------------------------------------------------------

class TestStripUnverifiedTag:
    def test_removes_unverified_tag(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate(f"""async () => {{
            const {{ stripUnverifiedTag }} = await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            const UNVERIFIED = '{UNVERIFIED_TAG_UUID}';
            const uiBBox = {{ tags: [{{ uuid: UNVERIFIED, icon: null }}] }};
            stripUnverifiedTag(uiBBox);
            return uiBBox.tags.map(t => t.uuid);
        }}""")
        assert UNVERIFIED_TAG_UUID not in result

    def test_leaves_non_verification_tags_intact(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate(f"""async () => {{
            const {{ stripUnverifiedTag }} = await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            const UNVERIFIED = '{UNVERIFIED_TAG_UUID}';
            const uiBBox = {{
                tags: [
                    {{ uuid: UNVERIFIED, icon: null }},
                    {{ uuid: 'keep-me', icon: '★' }},
                ],
            }};
            stripUnverifiedTag(uiBBox);
            return uiBBox.tags.map(t => t.uuid);
        }}""")
        assert result == ['keep-me']

    def test_noop_when_no_unverified_tag(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { stripUnverifiedTag } = await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            const uiBBox = { tags: [{ uuid: 'other-tag', icon: '✓' }] };
            stripUnverifiedTag(uiBBox);
            return uiBBox.tags.map(t => t.uuid);
        }""")
        assert result == ['other-tag']


# ---------------------------------------------------------------------------
# Drawing bounding boxes
# ---------------------------------------------------------------------------

class TestDrawBbox:
    def test_drawn_bbox_appears_in_runtime_state(self, page: Page, base_url: str):
        """mousedown → mousemove → mouseup with draw tool registers bbox in state."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { handleMouseDown, handleMouseMove, handleMouseUp } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            runtime.setTool('bbox');
            // Draw from canvas (100,100) to (250,200) — well above BBOX_MIN_PX=10
            await drawBbox(100, 100, 250, 200);

            return runtime.state.image.bboxes.size;
        }""")
        assert result == 1

    def test_drawn_bbox_without_label_gets_unverified_tag(self, page: Page, base_url: str):
        """applyDefaultVerificationTag fires inside handleMouseUp on a new unlabelled bbox."""
        page.goto(base_url)
        result = page.evaluate(f"""async () => {{
            const {{ createImageTaggingRuntime }} = await import('/app-static/js/app/image-tagging/runtime.js');
            const {{ handleMouseDown, handleMouseMove, handleMouseUp }} =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            {_SETUP}

            runtime.setTool('bbox');
            await drawBbox(100, 100, 250, 200);

            const bbox = [...runtime.state.image.bboxes.values()][0];
            return bbox.tags.map(t => t.uuid);
        }}""")
        assert UNVERIFIED_TAG_UUID in result

    def test_drawing_pushes_state_and_marks_dirty(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { handleMouseDown, handleMouseMove, handleMouseUp } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            runtime.setTool('bbox');
            await drawBbox(100, 100, 250, 200);

            return { canUndo: runtime.canUndo, isDirty: runtime.isDirty };
        }""")
        assert result['canUndo'] is True
        assert result['isDirty'] is True

    def test_too_small_drag_does_not_register_bbox(self, page: Page, base_url: str):
        """A drag of < BBOX_MIN_PX (10 px) in each axis must be discarded."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { handleMouseDown, handleMouseMove, handleMouseUp } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            runtime.setTool('bbox');
            // Only 5 px total — below BBOX_MIN_PX=10
            await drawBbox(100, 100, 105, 105);

            return runtime.state.image.bboxes.size;
        }""")
        assert result == 0

    def test_drawn_bbox_coordinates_normalised_to_image_space(self, page: Page, base_url: str):
        """After drawing, uiBBox.{x,y,width,height} are in [0,1] image space."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { handleMouseDown, handleMouseMove, handleMouseUp } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            // Viewport: scale=2, offset=0,0  →  sw=800, sh=600
            // Draw from (0,0) to (400,300): covers exactly half the image in each axis
            runtime.setTool('bbox');
            await drawBbox(0, 0, 400, 300);

            const bbox = [...runtime.state.image.bboxes.values()][0];
            return { x: bbox.x, y: bbox.y, w: bbox.width, h: bbox.height };
        }""")
        assert result['x'] == pytest.approx(0.0)
        assert result['y'] == pytest.approx(0.0)
        assert result['w'] == pytest.approx(0.5)  # 400/800
        assert result['h'] == pytest.approx(0.5)  # 300/600


# ---------------------------------------------------------------------------
# Label assignment (double-click with bbox:<uuid> tool)
# ---------------------------------------------------------------------------

class TestLabelAssignment:
    def test_double_click_assigns_label(self, page: Page, base_url: str):
        """Double-clicking a bbox with bbox:<uuid> tool assigns the label."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { labelsFetcher } = await import('/app-static/js/shared/api/labels.js');
            const { handleMouseUp } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            // Preload the label cache (avoids real network call)
            labelsFetcher.set('labelsMap', new Map([
                ['cat-uuid', { uuid: 'cat-uuid', name: 'Cat', color: '#ff0000' }],
            ]));

            // Place a bbox on the canvas: at canvas (80,60)→(240,180)
            // With vp scale=2 offset=0,0: uiBBox.x=80/800=0.1, w=160/800=0.2 etc.
            const b = createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 });
            runtime.state.image.bboxes.set('b1', b);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            // Set label-draw tool
            runtime.setTool('bbox:cat-uuid');

            // BboxView centre in canvas coords: x=0.1*800+0.5*0.2*800=80+80=160, y=0.1*600+60=120
            const cx = 160, cy = 120;

            // First click — sets lastClickTime; isDoubleClick=false
            setPtr(cx, cy);
            await handleMouseUp(makeEvt(0, cx, cy), runtime);

            // Second immediate click — isDoubleClick=true (timeDelta << 400ms)
            await handleMouseUp(makeEvt(0, cx, cy), runtime);

            return runtime.state.image.bboxes.get('b1').label;
        }""")
        assert result is not None
        assert result['uuid'] == 'cat-uuid'
        assert result['text'] == 'Cat'

    def test_double_click_label_strips_unverified_tag(self, page: Page, base_url: str):
        """Assigning a label removes the UNVERIFIED tag (human confirmed the detection)."""
        page.goto(base_url)
        result = page.evaluate(f"""async () => {{
            const {{ createImageTaggingRuntime }} = await import('/app-static/js/app/image-tagging/runtime.js');
            const {{ createUIBBbox }} = await import('/app-static/js/shared/domain/image/mapper.js');
            const {{ rerenderBboxViews }} = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const {{ labelsFetcher }} = await import('/app-static/js/shared/api/labels.js');
            const {{ handleMouseUp }} =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            {_SETUP}

            labelsFetcher.set('labelsMap', new Map([
                ['cat-uuid', {{ uuid: 'cat-uuid', name: 'Cat', color: '#ff0000' }}],
            ]));

            const UNVERIFIED = '{UNVERIFIED_TAG_UUID}';
            const b = createUIBBbox({{
                uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2,
                tags: [{{ uuid: UNVERIFIED, icon: null }}],
            }});
            runtime.state.image.bboxes.set('b1', b);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            runtime.setTool('bbox:cat-uuid');
            const cx = 160, cy = 120;
            setPtr(cx, cy);
            await handleMouseUp(makeEvt(0, cx, cy), runtime);
            await handleMouseUp(makeEvt(0, cx, cy), runtime);

            return runtime.state.image.bboxes.get('b1').tags.map(t => t.uuid);
        }}""")
        assert UNVERIFIED_TAG_UUID not in result


# ---------------------------------------------------------------------------
# Tag toggling (double-click with tag:<uuid> tool)
# ---------------------------------------------------------------------------

class TestTagToggle:
    def test_double_click_adds_tag(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { tagsFetcher } = await import('/app-static/js/shared/api/tags.js');
            const { handleMouseUp } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            tagsFetcher.set('tagsMap', new Map([
                ['reviewed-uuid', { uuid: 'reviewed-uuid', name: 'Reviewed', icon: '✓', color: '#0f0' }],
            ]));

            const b = createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2, tags: [] });
            runtime.state.image.bboxes.set('b1', b);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            runtime.setTool('tag:reviewed-uuid');
            const cx = 160, cy = 120;
            setPtr(cx, cy);
            await handleMouseUp(makeEvt(0, cx, cy), runtime);
            await handleMouseUp(makeEvt(0, cx, cy), runtime);

            return runtime.state.image.bboxes.get('b1').tags.map(t => t.uuid);
        }""")
        assert 'reviewed-uuid' in result

    def test_double_click_removes_existing_tag(self, page: Page, base_url: str):
        """Double-clicking again toggles the tag off."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { tagsFetcher } = await import('/app-static/js/shared/api/tags.js');
            const { handleMouseUp } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            tagsFetcher.set('tagsMap', new Map([
                ['reviewed-uuid', { uuid: 'reviewed-uuid', name: 'Reviewed', icon: '✓', color: '#0f0' }],
            ]));

            // Start with the tag already present
            const b = createUIBBbox({
                uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2,
                tags: [{ uuid: 'reviewed-uuid', icon: '✓' }],
            });
            runtime.state.image.bboxes.set('b1', b);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            runtime.setTool('tag:reviewed-uuid');
            const cx = 160, cy = 120;
            setPtr(cx, cy);
            await handleMouseUp(makeEvt(0, cx, cy), runtime);
            await handleMouseUp(makeEvt(0, cx, cy), runtime);

            return runtime.state.image.bboxes.get('b1').tags.map(t => t.uuid);
        }""")
        assert 'reviewed-uuid' not in result

    def test_toggle_on_verification_tag_preserves_nonverification_tags(self, page: Page, base_url: str):
        """Toggle-on of a verification tag uses VERIFICATION_TAG_UUIDS to build the base:
        existing verification tags are stripped while non-verification tags are kept."""
        page.goto(base_url)
        result = page.evaluate(f"""async () => {{
            const {{ createImageTaggingRuntime }} = await import('/app-static/js/app/image-tagging/runtime.js');
            const {{ createUIBBbox }} = await import('/app-static/js/shared/domain/image/mapper.js');
            const {{ rerenderBboxViews }} = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const {{ tagsFetcher }} = await import('/app-static/js/shared/api/tags.js');
            const {{ handleMouseUp }} =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            {_SETUP}

            const UNVERIFIED = '{UNVERIFIED_TAG_UUID}';

            tagsFetcher.set('tagsMap', new Map([
                [UNVERIFIED, {{ uuid: UNVERIFIED, name: 'Unverified', icon: '?', color: '#f00' }}],
            ]));

            // Start with a non-verification tag present and no UNVERIFIED.
            // The toggle-on path runs: base = tags.filter(t => !VERIFICATION_TAG_UUIDS.has(t.uuid))
            // which must preserve 'extra-tag' (not in the set) while stripping any
            // verification peers (none here, but the filter runs on a non-empty array).
            const b = createUIBBbox({{
                uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2,
                tags: [{{ uuid: 'extra-tag', icon: '★' }}],
            }});
            runtime.state.image.bboxes.set('b1', b);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            // Double-click to toggle UNVERIFIED on
            runtime.setTool('tag:' + UNVERIFIED);
            const cx = 160, cy = 120;
            setPtr(cx, cy);
            await handleMouseUp(makeEvt(0, cx, cy), runtime);
            await handleMouseUp(makeEvt(0, cx, cy), runtime);

            const finalTags = runtime.state.image.bboxes.get('b1').tags.map(t => t.uuid);
            return {{
                unverifiedCount: finalTags.filter(u => u === UNVERIFIED).length,
                hasExtraTag:     finalTags.includes('extra-tag'),
            }};
        }}""")
        # UNVERIFIED added exactly once; non-verification 'extra-tag' preserved
        assert result['unverifiedCount'] == 1
        assert result['hasExtraTag'] is True


# ---------------------------------------------------------------------------
# Select-tool click / clear selection
# ---------------------------------------------------------------------------

class TestSelection:
    def test_click_on_bbox_selects_it_in_transformer(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { handleMouseUp, handleClick } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            const b = createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 });
            runtime.state.image.bboxes.set('b1', b);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            runtime.setTool('select');
            const cx = 160, cy = 120;
            setPtr(cx, cy);
            await handleMouseUp(makeEvt(0, cx, cy), runtime);
            handleClick(makeEvt(0), runtime);

            return runtime.canvas.transformer.nodes().length;
        }""")
        assert result == 1

    def test_click_empty_canvas_clears_selection(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { handleMouseUp, handleClick } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            const b = createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 });
            runtime.state.image.bboxes.set('b1', b);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            runtime.setTool('select');

            // Select the bbox first
            setPtr(160, 120);
            await handleMouseUp(makeEvt(0, 160, 120), runtime);
            handleClick(makeEvt(0), runtime);
            const afterSelect = runtime.canvas.transformer.nodes().length;

            // Click far outside any bbox
            setPtr(700, 550);
            await handleMouseUp(makeEvt(0, 700, 550), runtime);
            handleClick(makeEvt(0), runtime);
            const afterClear = runtime.canvas.transformer.nodes().length;

            return { afterSelect, afterClear };
        }""")
        assert result == {'afterSelect': 1, 'afterClear': 0}

    def test_click_fires_bbox_clicked_event(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { handleMouseUp, handleClick } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');
            const { events, EventTypes } = await import('/app-static/js/shared/events/index.js');

            """ + _SETUP + """

            const b = createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 });
            runtime.state.image.bboxes.set('b1', b);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            runtime.setTool('select');

            let clickedId = null;
            events.subscribeToCanvas(runtime.id, EventTypes.CANVAS_BBOX_CLICKED, ({ bboxId }) => {
                clickedId = bboxId;
            });

            setPtr(160, 120);
            await handleMouseUp(makeEvt(0, 160, 120), runtime);
            handleClick(makeEvt(0), runtime);

            return clickedId;
        }""")
        assert result == 'b1'


# ---------------------------------------------------------------------------
# Stacked-bbox cycling
# ---------------------------------------------------------------------------

class TestOverlapCycling:
    def test_first_click_selects_smallest_bbox(self, page: Page, base_url: str):
        """With two overlapping bboxes, the first click picks the smallest (by area)."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { handleMouseUp, handleClick } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            // big: covers (0,0)→(0.5,0.5) image space → canvas (0,0)→(400,300), area=120000
            // small: covers (0.1,0.1)→(0.2,0.2) → canvas (80,60)→(160,120), area=4800
            // Point (120, 90) is inside both.
            const big   = createUIBBbox({ uuid: 'big',   x: 0, y: 0, width: 0.5, height: 0.5 });
            const small = createUIBBbox({ uuid: 'small', x: 0.1, y: 0.1, width: 0.1, height: 0.1 });
            runtime.state.image.bboxes.set('big', big);
            runtime.state.image.bboxes.set('small', small);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            runtime.setTool('select');
            setPtr(120, 90);
            await handleMouseUp(makeEvt(0, 120, 90), runtime);
            handleClick(makeEvt(0), runtime);

            return runtime.canvas.interactionState.selectedBbox?.id();
        }""")
        assert result == 'small'

    def test_double_click_cycles_to_larger_bbox(self, page: Page, base_url: str):
        """Double-clicking at the same spot advances the cycle index to the larger bbox."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { handleMouseUp, handleClick } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            const big   = createUIBBbox({ uuid: 'big',   x: 0, y: 0, width: 0.5, height: 0.5 });
            const small = createUIBBbox({ uuid: 'small', x: 0.1, y: 0.1, width: 0.1, height: 0.1 });
            runtime.state.image.bboxes.set('big', big);
            runtime.state.image.bboxes.set('small', small);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            runtime.setTool('select');
            const cx = 120, cy = 90;

            // First click → small selected
            setPtr(cx, cy);
            await handleMouseUp(makeEvt(0, cx, cy), runtime);
            handleClick(makeEvt(0), runtime);

            // Double-click (second handleMouseUp + handleClick immediately after)
            await handleMouseUp(makeEvt(0, cx, cy), runtime);
            handleClick(makeEvt(0), runtime);

            return runtime.canvas.interactionState.selectedBbox?.id();
        }""")
        assert result == 'big'

    def test_third_double_click_wraps_back_to_small(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { handleMouseUp, handleClick } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            const big   = createUIBBbox({ uuid: 'big',   x: 0, y: 0, width: 0.5, height: 0.5 });
            const small = createUIBBbox({ uuid: 'small', x: 0.1, y: 0.1, width: 0.1, height: 0.1 });
            runtime.state.image.bboxes.set('big', big);
            runtime.state.image.bboxes.set('small', small);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            runtime.setTool('select');
            const cx = 120, cy = 90;

            // Click 1 → small
            setPtr(cx, cy);
            await handleMouseUp(makeEvt(0, cx, cy), runtime);
            handleClick(makeEvt(0), runtime);
            // Click 2 (double) → big
            await handleMouseUp(makeEvt(0, cx, cy), runtime);
            handleClick(makeEvt(0), runtime);
            // Click 3 (triple = another double) → wraps back to small
            await handleMouseUp(makeEvt(0, cx, cy), runtime);
            handleClick(makeEvt(0), runtime);

            return runtime.canvas.interactionState.selectedBbox?.id();
        }""")
        assert result == 'small'

    def test_click_different_spot_resets_cycle(self, page: Page, base_url: str):
        """Clicking at a different location starts a fresh cycle (smallest first)."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { handleMouseUp, handleClick } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            const big   = createUIBBbox({ uuid: 'big',   x: 0, y: 0, width: 0.5, height: 0.5 });
            const small = createUIBBbox({ uuid: 'small', x: 0.1, y: 0.1, width: 0.1, height: 0.1 });
            runtime.state.image.bboxes.set('big', big);
            runtime.state.image.bboxes.set('small', small);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            runtime.setTool('select');

            // First double-click at (120,90) → cycle advances to big
            setPtr(120, 90);
            await handleMouseUp(makeEvt(0, 120, 90), runtime);
            handleClick(makeEvt(0), runtime);
            await handleMouseUp(makeEvt(0, 120, 90), runtime);
            handleClick(makeEvt(0), runtime);
            const afterDoubleClick = runtime.canvas.interactionState.selectedBbox?.id();

            // Now click a clearly different spot (far away) — only big contains it
            setPtr(300, 200);
            await handleMouseUp(makeEvt(0, 300, 200), runtime);
            handleClick(makeEvt(0), runtime);
            const afterReset = runtime.canvas.interactionState.selectedBbox?.id();

            return { afterDoubleClick, afterReset };
        }""")
        assert result['afterDoubleClick'] == 'big'
        assert result['afterReset'] == 'big'   # only big covers (300,200)


# ---------------------------------------------------------------------------
# Right-click context menu (handleContextMenu)
# ---------------------------------------------------------------------------

class TestContextMenu:
    def test_right_click_empty_canvas_toggles_bbox_select(self, page: Page, base_url: str):
        """Right-clicking empty canvas: select→bbox or bbox→select."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { handleContextMenu } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            runtime.setTool('select');

            // Right-click on empty canvas → should switch to 'bbox'
            setPtr(700, 500);
            handleContextMenu(makeEvt(2), runtime);
            const after1 = runtime.tool;

            // Right-click again → back to 'select'
            handleContextMenu(makeEvt(2), runtime);
            const after2 = runtime.tool;

            return { after1, after2 };
        }""")
        assert result['after1'] == 'bbox:'
        assert result['after2'] == 'select'

    def test_right_click_labelled_bbox_toggles_label_tool(self, page: Page, base_url: str):
        """Right-clicking a labelled bbox switches to bbox:<label-uuid>; right-clicking again returns to select."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { handleContextMenu } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            const b = createUIBBbox({
                uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2,
                label: { uuid: 'cat-uuid', text: 'Cat', color: '#f00' },
            });
            runtime.state.image.bboxes.set('b1', b);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            runtime.setTool('select');

            // handleContextMenu detects the clicked bbox via e.target (Konva node walk).
            // Use the actual Konva BboxView as e.target so the labelled-bbox branch fires.
            const bboxView = runtime.canvas.stage.findOne('#b1');
            const ctxEvt = { target: bboxView, evt: { preventDefault: () => {} } };

            // First right-click on the labelled bbox: select → bbox:cat-uuid
            handleContextMenu(ctxEvt, runtime);
            const afterFirst = runtime.tool;

            // Second right-click: bbox:cat-uuid → select
            handleContextMenu(ctxEvt, runtime);
            const afterSecond = runtime.tool;

            return { afterFirst, afterSecond };
        }""")
        assert result['afterFirst'] == 'bbox:cat-uuid'
        assert result['afterSecond'] == 'select'


# ---------------------------------------------------------------------------
# deleteSelected
# ---------------------------------------------------------------------------

class TestDeleteSelected:
    def test_deletes_selected_bbox(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { deleteSelected } = await import('/app-static/js/app/image-tagging/canvas/image.js');
            const { handleMouseUp, handleClick } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            """ + _SETUP + """

            const b = createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 });
            runtime.state.image.bboxes.set('b1', b);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            // Select the bbox
            runtime.setTool('select');
            setPtr(160, 120);
            await handleMouseUp(makeEvt(0, 160, 120), runtime);
            handleClick(makeEvt(0), runtime);

            const beforeDelete = runtime.state.image.bboxes.size;
            deleteSelected(runtime);
            const afterDelete = runtime.state.image.bboxes.size;

            return { beforeDelete, afterDelete };
        }""")
        assert result == {'beforeDelete': 1, 'afterDelete': 0}

    def test_delete_with_nothing_selected_is_noop(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { deleteSelected } = await import('/app-static/js/app/image-tagging/canvas/image.js');

            """ + _SETUP + """

            const b = createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 });
            runtime.state.image.bboxes.set('b1', b);
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            // Nothing selected — transformer.nodes() is empty
            let threw = false;
            try { deleteSelected(runtime); } catch (e) { threw = true; }

            return { threw, bboxCount: runtime.state.image.bboxes.size };
        }""")
        assert result['threw'] is False
        assert result['bboxCount'] == 1  # bbox untouched


# ---------------------------------------------------------------------------
# clearAnnotations
# ---------------------------------------------------------------------------

class TestClearAnnotations:
    def test_clear_removes_all_bboxes(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { clearAnnotations } = await import('/app-static/js/app/image-tagging/canvas/image.js');

            """ + _SETUP + """

            runtime.state.image.bboxes.set('b1', createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 }));
            runtime.state.image.bboxes.set('b2', createUIBBbox({ uuid: 'b2', x: 0.5, y: 0.5, width: 0.2, height: 0.2 }));
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            clearAnnotations(runtime);

            return {
                bboxMapSize: runtime.state.image.bboxes.size,
                layerBboxCount: runtime.canvas.layer.find('.bbox').length,
            };
        }""")
        assert result == {'bboxMapSize': 0, 'layerBboxCount': 0}

    def test_clear_pushes_state_so_undo_restores(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { clearAnnotations } = await import('/app-static/js/app/image-tagging/canvas/image.js');

            """ + _SETUP + """

            runtime.state.image.bboxes.set('b1', createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 }));
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);
            runtime.pushState();  // snapshot with 1 bbox

            clearAnnotations(runtime);  // pushes cleared state
            const afterClear = runtime.state.image.bboxes.size;

            runtime.undo();
            const afterUndo = runtime.state.image.bboxes.size;

            return { afterClear, afterUndo };
        }""")
        assert result == {'afterClear': 0, 'afterUndo': 1}
