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


# ---------------------------------------------------------------------------
# Focus handling
# ---------------------------------------------------------------------------


def _focus_element(page: Page, html: str) -> None:
    """Append an element to the body and give it focus."""
    page.evaluate(
        """(html) => {
        const host = document.createElement('div');
        host.style.cssText = 'position:fixed;bottom:0;right:0;';
        host.innerHTML = html;
        document.body.appendChild(host);
        window._testFocused = host.firstElementChild;
        window._testFocused.focus();
    }""",
        html,
    )


class TestHotkeysAndFocus:
    """Only *text entry* focus may swallow hotkeys.

    The inference panel's model checkboxes and confidence slider are <input>s
    too, and clicking one used to kill every hotkey for the rest of the modal's
    life: Konva's stage calls preventDefault() on pointerdown, so a subsequent
    click on the canvas never moves focus off the input.
    """

    def test_text_input_focus_suppresses_hotkeys(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)
        page.evaluate("""() => window._testRuntime.setTool('select')""")

        _focus_element(page, "<input type='text'>")
        page.keyboard.press("r")

        assert page.evaluate("""() => window._testRuntime.tool""") == "select"

    def test_textarea_focus_suppresses_hotkeys(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)
        page.evaluate("""() => window._testRuntime.setTool('select')""")

        _focus_element(page, "<textarea></textarea>")
        page.keyboard.press("r")

        assert page.evaluate("""() => window._testRuntime.tool""") == "select"

    def test_checkbox_focus_does_not_suppress_hotkeys(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)
        page.evaluate("""() => window._testRuntime.setTool('select')""")

        _focus_element(page, "<input type='checkbox'>")
        page.keyboard.press("r")

        assert page.evaluate("""() => window._testRuntime.tool""") == "bbox"

    def test_range_slider_focus_does_not_suppress_hotkeys(self, page: Page, base_url: str):
        """The inference panel's confidence slider."""
        _setup_runtime(page, base_url)
        page.evaluate("""() => window._testRuntime.setTool('select')""")

        _focus_element(page, "<input type='range' min='0' max='100'>")
        page.keyboard.press("Delete")

        assert page.evaluate("""() => window._testRuntime.state.image.bboxes.size""") == 1
        page.keyboard.press("r")
        assert page.evaluate("""() => window._testRuntime.tool""") == "bbox"

    def test_button_focus_does_not_suppress_hotkeys(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)
        page.evaluate("""() => window._testRuntime.setTool('select')""")

        _focus_element(page, "<button>Save</button>")
        page.keyboard.press("r")

        assert page.evaluate("""() => window._testRuntime.tool""") == "bbox"

    def test_middle_mouse_pan_blurs_a_focused_text_input(self, page: Page, base_url: str):
        """Middle-drag panning calls preventDefault(), so the browser never moves
        focus off the input — which is why "clicking around on the canvas" did not
        revive the hotkeys. The stage has to blur text entry itself."""
        page.goto(base_url)
        page.evaluate("""async () => {
            const { createCanvasInstance } =
                await import('/app-static/js/app/image-tagging/canvas/instance.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } =
                await import('/app-static/js/app/image-tagging/canvas/manager.js');

            const div = document.createElement('div');
            div.style.cssText = 'width:400px;height:300px;position:fixed;top:0;left:0;';
            document.body.appendChild(div);

            const instance = await createCanvasInstance(div, { tool: 'select', enhance: 'none' });
            const runtime = instance.runtime;

            const imgCanvas = document.createElement('canvas');
            imgCanvas.width = 400; imgCanvas.height = 300;
            const img = new Image();
            img.src = imgCanvas.toDataURL();
            await new Promise(r => { img.onload = r; });

            const b = createUIBBbox({ uuid: 'b1', x: 0.25, y: 0.25, width: 0.5, height: 0.5 });
            runtime.setImage({
                uuid: 'img', filename: 'test.png', image_path: '/test.png',
                img, bboxes: new Map([['b1', b]]), dirty: false,
            });
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);
            window._testRuntime = runtime;
        }""")

        # Centre of the bbox, in page coordinates.
        point = page.evaluate("""() => {
            const r = window._testRuntime.canvas.layer.findOne('#b1').getClientRect();
            const c = window._testRuntime.canvas.container.getBoundingClientRect();
            return { x: c.left + r.x + r.width / 2, y: c.top + r.y + r.height / 2 };
        }""")

        _focus_element(page, "<input type='text'>")
        assert page.evaluate("""() => document.activeElement.tagName""") == "INPUT"

        # Middle-button drag = pan.
        page.mouse.move(point["x"], point["y"])
        page.mouse.down(button="middle")
        page.mouse.move(point["x"] + 10, point["y"] + 10)
        page.mouse.up(button="middle")

        assert page.evaluate("""() => document.activeElement.tagName""") != "INPUT"
        page.keyboard.press("r")
        assert page.evaluate("""() => window._testRuntime.tool""") == "bbox"


# ---------------------------------------------------------------------------
# Hotkeys must not fire while a dialog is open
# ---------------------------------------------------------------------------


class TestHotkeysBlockedWhileDialogOpen:
    """A dialog built only of buttons (e.g. the "unsaved changes?" confirm)
    never becomes text-entry focus, so isTextEntryElement's guard alone let
    hotkeys keep reaching the canvas underneath it. Concretely: pressing ']'
    (next image) while that confirm was open re-ran the guarded navigation on
    every keypress, popping another stacked copy of the same dialog each time.
    """

    def test_bound_key_is_ignored_while_a_dialog_is_open(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)
        page.evaluate("""() => window._testRuntime.setTool('bbox')""")

        page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const shell = createModalShell();
            shell.open();
            window._testShell = shell;
        }""")

        page.keyboard.press("v")

        result = page.evaluate("""() => window._testRuntime.tool""")
        assert result == "bbox"  # unchanged — the hotkey must not have fired

    def test_bound_key_works_again_once_the_dialog_closes(self, page: Page, base_url: str):
        _setup_runtime(page, base_url)
        page.evaluate("""() => window._testRuntime.setTool('bbox')""")

        page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const shell = createModalShell();
            shell.open();
            window._testShell = shell;
        }""")
        page.evaluate("""() => window._testShell.close()""")

        page.keyboard.press("v")

        result = page.evaluate("""() => window._testRuntime.tool""")
        assert result == "select"

    def test_repeated_keypresses_do_not_open_additional_dialogs(self, page: Page, base_url: str):
        """Direct regression for the reported bug: mashing the bound key while
        the confirm dialog is open must not stack up more copies of it."""
        _setup_runtime(page, base_url)

        page.evaluate("""async () => {
            const { showUnsavedChangesModal } = await import('/app-static/js/ui/utils/confirmModal.js');
            window._testModalPromise = showUnsavedChangesModal();
        }""")

        for _ in range(5):
            page.keyboard.press("v")

        count = page.evaluate("""() => document.querySelectorAll('.app-modal-overlay').length""")
        assert count == 1
