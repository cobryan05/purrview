"""
Keyboard-hotkey integration tests for the canvas.

These tests fire real DOM keydown events via page.keyboard, which triggers
the document-level listener registered by hotkeys.js.  The runtime is stored
on window._testRuntime so state can be read back after the keypress.

Default bindings (from hotkeyBindings.js):
  ctrl+z       → action:undo
  ctrl+shift+z → action:redo
  v            → action:select
  r            → action:bbox
  delete       → action:delete
  backspace    → action:delete

Run with:
    pytest tests/playwright/test_canvas_hotkeys.py -p no:xdist --override-ini="addopts=" -v
"""

import pytest
from playwright.sync_api import Page

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_runtime(page: Page, base_url: str) -> None:
    """
    Navigate to base_url, create a canvas runtime with one bbox in state,
    attach hotkeys and expose the runtime as window._testRuntime.
    """
    page.goto(base_url)

    page.evaluate("""async () => {
        const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
        const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
        const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
        const { attachHotkeys } = await import('/app-static/js/app/hotkeys.js');

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

        // Add a bbox and take a snapshot so undo/redo has something to do
        const b = createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 });
        runtime.state.image.bboxes.set('b1', b);
        rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);
        runtime.pushState();

        // Expose and register as the active hotkey recipient
        window._testRuntime = runtime;
        attachHotkeys(runtime);
    }""")


# ---------------------------------------------------------------------------
# Undo / redo via keyboard
# ---------------------------------------------------------------------------


class TestKeyboardUndo:
    def test_ctrl_z_triggers_undo(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)

        page.keyboard.press("Control+z")

        result = page.evaluate("""() => ({
            bboxes: window._testRuntime.state.image.bboxes.size,
            canUndo: window._testRuntime.canUndo,
        })""")
        assert result["bboxes"] == 0
        assert result["canUndo"] is False

    def test_ctrl_z_does_nothing_when_nothing_to_undo(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)

        # First undo exhausts the stack
        page.keyboard.press("Control+z")
        # Second undo should be a no-op
        page.keyboard.press("Control+z")

        result = page.evaluate("""() => window._testRuntime.state.image.bboxes.size""")
        assert result == 0  # still 0, did not go negative / throw


class TestKeyboardRedo:
    def test_ctrl_shift_z_triggers_redo(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)

        page.keyboard.press("Control+z")
        before = page.evaluate("""() => window._testRuntime.state.image.bboxes.size""")
        assert before == 0

        page.keyboard.press("Control+Shift+Z")
        after = page.evaluate("""() => window._testRuntime.state.image.bboxes.size""")
        assert after == 1

    def test_redo_does_nothing_when_stack_is_empty(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)

        # No undo first → redo stack is empty → no-op
        page.keyboard.press("Control+Shift+Z")

        result = page.evaluate("""() => window._testRuntime.state.image.bboxes.size""")
        assert result == 1  # unchanged


# ---------------------------------------------------------------------------
# Tool switching via keyboard
# ---------------------------------------------------------------------------


class TestKeyboardToolSwitch:
    def test_v_switches_to_select_tool(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)

        # Start on a non-select tool
        page.evaluate("""() => window._testRuntime.setTool('bbox')""")

        page.keyboard.press("v")

        result = page.evaluate("""() => window._testRuntime.tool""")
        assert result == "select"

    def test_r_switches_to_bbox_tool(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)

        page.evaluate("""() => window._testRuntime.setTool('select')""")

        page.keyboard.press("r")

        result = page.evaluate("""() => window._testRuntime.tool""")
        assert result == "bbox"

    def test_pressing_tool_key_when_already_active_is_idempotent(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)

        page.evaluate("""() => window._testRuntime.setTool('select')""")
        page.keyboard.press("v")

        result = page.evaluate("""() => window._testRuntime.tool""")
        assert result == "select"


# ---------------------------------------------------------------------------
# Delete via keyboard
# ---------------------------------------------------------------------------


class TestKeyboardDelete:
    def test_delete_key_removes_selected_bbox(self, page: Page, base_url: str):
        """Press Delete when a bbox is selected → bbox removed from state."""
        _setup_runtime(page, base_url)

        # Select the bbox via the transformer directly
        page.evaluate("""async () => {
            const { handleMouseUp, handleClick } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            const runtime = window._testRuntime;
            runtime.setTool('select');
            runtime.canvas.stage._pointerPositions = [{ x: 160, y: 120 }];
            const ev = { evt: { button: 0, clientX: 160, clientY: 120, preventDefault: () => {} } };
            await handleMouseUp(ev, runtime);
            handleClick(ev, runtime);
        }""")

        selected = page.evaluate("""() => window._testRuntime.canvas.transformer.nodes().length""")
        assert selected == 1

        page.keyboard.press("Delete")

        result = page.evaluate("""() => ({
            bboxes: window._testRuntime.state.image.bboxes.size,
            transformerNodes: window._testRuntime.canvas.transformer.nodes().length,
        })""")
        assert result["bboxes"] == 0
        assert result["transformerNodes"] == 0

    def test_backspace_key_also_deletes(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)

        page.evaluate("""async () => {
            const { handleMouseUp, handleClick } =
                await import('/app-static/js/app/image-tagging/canvas/interaction.js');

            const runtime = window._testRuntime;
            runtime.setTool('select');
            runtime.canvas.stage._pointerPositions = [{ x: 160, y: 120 }];
            const ev = { evt: { button: 0, clientX: 160, clientY: 120, preventDefault: () => {} } };
            await handleMouseUp(ev, runtime);
            handleClick(ev, runtime);
        }""")

        page.keyboard.press("Backspace")

        result = page.evaluate("""() => window._testRuntime.state.image.bboxes.size""")
        assert result == 0

    def test_delete_with_nothing_selected_is_noop(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)

        # Ensure nothing is selected
        page.evaluate("""() => window._testRuntime.canvas.transformer.nodes([])""")

        page.keyboard.press("Delete")

        result = page.evaluate("""() => window._testRuntime.state.image.bboxes.size""")
        assert result == 1  # bbox still there
