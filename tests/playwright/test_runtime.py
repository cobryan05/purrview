"""
Tests for image-tagging/runtime.js — undo / redo, dirty-state lifecycle,
and undo-stack capping.
Phase 2 canvas tests — require Konva as a global (navigate to base_url).

Run with:
    pytest tests/playwright/test_runtime.py -p no:xdist --override-ini="addopts=" -v
"""
import pytest
from playwright.sync_api import Page




# ---------------------------------------------------------------------------
# Shared JS that creates a runtime with a canvas container and a blank image.
# The 400×300 image fills the 800×600 container exactly at scale=2 / offset=0.
# Imported at the start (after all imports) inside every evaluate block.
# ---------------------------------------------------------------------------
_RUNTIME_SETUP = """
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
        uuid: 'test-img', filename: 'test.png', image_path: '/test.png',
        img, bboxes: new Map(), dirty: false,
    });

    const makeBbox = (uuid) => ({
        uuid, x: 0.1, y: 0.1, width: 0.2, height: 0.2,
        label: null, tags: [], selected: false, dirty: false,
    });
"""


# ---------------------------------------------------------------------------
# pushState / undo / redo
# ---------------------------------------------------------------------------

class TestPushStateAndUndo:
    def test_no_undo_after_fresh_setImage(self, page: Page, base_url: str):
        """setImage calls clearHistory+pushState; that baseline is not undoable."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');

            """ + _RUNTIME_SETUP + """

            return { canUndo: runtime.canUndo, canRedo: runtime.canRedo };
        }""")
        assert result['canUndo'] is False
        assert result['canRedo'] is False

    def test_push_state_enables_undo(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');

            """ + _RUNTIME_SETUP + """

            runtime.state.image.bboxes.set('b1', makeBbox('b1'));
            runtime.pushState();

            return runtime.canUndo;
        }""")
        assert result is True

    def test_undo_restores_previous_bbox_map(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');

            """ + _RUNTIME_SETUP + """

            // Add a bbox and take a snapshot
            runtime.state.image.bboxes.set('b1', makeBbox('b1'));
            runtime.pushState();
            const afterAdd = runtime.state.image.bboxes.size;   // 1

            // Undo → should be empty again
            runtime.undo();
            const afterUndo = runtime.state.image.bboxes.size;  // 0

            return { afterAdd, afterUndo };
        }""")
        assert result == {'afterAdd': 1, 'afterUndo': 0}

    def test_redo_reapplies_undone_state(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');

            """ + _RUNTIME_SETUP + """

            runtime.state.image.bboxes.set('b1', makeBbox('b1'));
            runtime.pushState();
            runtime.undo();
            const beforeRedo = runtime.state.image.bboxes.size;  // 0
            runtime.redo();
            const afterRedo = runtime.state.image.bboxes.size;   // 1

            return { beforeRedo, afterRedo };
        }""")
        assert result == {'beforeRedo': 0, 'afterRedo': 1}

    def test_push_after_undo_clears_redo_stack(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');

            """ + _RUNTIME_SETUP + """

            runtime.state.image.bboxes.set('b1', makeBbox('b1'));
            runtime.pushState();
            runtime.undo();

            const canRedoBefore = runtime.canRedo;  // true

            // Now push a divergent state — redo stack must be cleared
            runtime.state.image.bboxes.set('b2', makeBbox('b2'));
            runtime.pushState();

            return { canRedoBefore, canRedoAfter: runtime.canRedo };
        }""")
        assert result['canRedoBefore'] is True
        assert result['canRedoAfter'] is False

    def test_undo_stack_capped_at_101_entries(self, page: Page, base_url: str):
        """After 110 pushes only 100 undos are available (stack capped at 101)."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');

            """ + _RUNTIME_SETUP + """

            for (let i = 0; i < 110; i++) {
                runtime.state.image.bboxes.set(`b${i}`, makeBbox(`b${i}`));
                runtime.pushState();
            }

            let undoCount = 0;
            while (runtime.canUndo) {
                runtime.undo();
                undoCount++;
            }
            return undoCount;
        }""")
        assert result == 100  # 101-entry stack → 100 undos from the top

    def test_multiple_undo_steps(self, page: Page, base_url: str):
        """Two pushes → two undos → canUndo becomes false."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');

            """ + _RUNTIME_SETUP + """

            runtime.state.image.bboxes.set('b1', makeBbox('b1'));
            runtime.pushState();
            runtime.state.image.bboxes.set('b2', makeBbox('b2'));
            runtime.pushState();

            runtime.undo();
            const after1 = runtime.state.image.bboxes.size;  // 1

            runtime.undo();
            const after2 = runtime.state.image.bboxes.size;  // 0

            return { after1, after2, canUndo: runtime.canUndo };
        }""")
        assert result == {'after1': 1, 'after2': 0, 'canUndo': False}

    def test_undo_at_baseline_is_noop(self, page: Page, base_url: str):
        """Calling undo when canUndo is false must not throw."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');

            """ + _RUNTIME_SETUP + """

            let threw = false;
            try { runtime.undo(); } catch (e) { threw = true; }
            return { threw, bboxes: runtime.state.image.bboxes.size };
        }""")
        assert result['threw'] is False
        assert result['bboxes'] == 0


# ---------------------------------------------------------------------------
# Dirty state
# ---------------------------------------------------------------------------

class TestDirtyState:
    def test_starts_clean_after_setImage(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');

            """ + _RUNTIME_SETUP + """

            return runtime.isDirty;
        }""")
        assert result is False

    def test_dirty_after_pushState(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');

            """ + _RUNTIME_SETUP + """

            runtime.state.image.bboxes.set('b1', makeBbox('b1'));
            runtime.pushState();

            return runtime.isDirty;
        }""")
        assert result is True

    def test_markClean_clears_dirty_flag(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');

            """ + _RUNTIME_SETUP + """

            runtime.state.image.bboxes.set('b1', makeBbox('b1'));
            runtime.pushState();
            runtime.markClean();

            return runtime.isDirty;
        }""")
        assert result is False

    def test_markDirty_sets_dirty_flag(self, page: Page, base_url: str):
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');

            """ + _RUNTIME_SETUP + """

            runtime.markDirty();

            return runtime.isDirty;
        }""")
        assert result is True

    def test_dirty_after_undo(self, page: Page, base_url: str):
        """Undo counts as a mutation — isDirty becomes true."""
        page.goto(base_url)
        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');

            """ + _RUNTIME_SETUP + """

            runtime.state.image.bboxes.set('b1', makeBbox('b1'));
            runtime.pushState();
            runtime.markClean();          // pretend we just saved

            runtime.undo();

            return runtime.isDirty;      // undo should have set dirty again
        }""")
        assert result is True
