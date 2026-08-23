"""
Behavioural tests for canvas multi-select and the bbox copy/paste clipboard.

  - Ctrl/Shift+click adds to and removes from the selection
  - A modifier click on empty canvas must NOT clear a selection being built
  - deleteSelected removes every selected box, and one undo brings them all back
  - Shift+drag marquee selects the boxes it intersects and no others
  - Copy survives an image change; paste creates fresh boxes at the same
    normalized coordinates
  - Paste skips boxes already present unless forced

Handlers are called directly (not through DOM events) so the pointer position
and modifier keys can be controlled exactly — same approach as
test_canvas_interaction.py, whose _SETUP block is reused here.
"""

from playwright.sync_api import Page

from playwright_tests.test_canvas_interaction import _SETUP

# Modifier-carrying event factory. The shared _SETUP's makeEvt has no
# ctrlKey/shiftKey at all, which is exactly what "a plain click" must look like,
# so modifier clicks need their own.
_MODIFIERS = """
    function makeModEvt({ ctrl = false, shift = false, button = 0, clientX = 0, clientY = 0 } = {}) {
        return { evt: { button, clientX, clientY, ctrlKey: ctrl, shiftKey: shift, preventDefault: () => {} } };
    }
    function selectedIds() {
        return runtime.canvas.interactionState.selectedBboxes.map(v => v.id());
    }
    async function clickAt(x, y, mods = {}) {
        setPtr(x, y);
        await handleMouseUp(makeModEvt({ ...mods, clientX: x, clientY: y }), runtime);
        handleClick(makeModEvt(mods), runtime);
    }
"""

# Two non-overlapping boxes: b1 at canvas (80,60)-(160,120), b2 at (400,300)-(480,360).
_TWO_BOXES = """
    const b1 = createUIBBbox({ uuid: 'b1', x: 0.1, y: 0.1, width: 0.1, height: 0.1 });
    const b2 = createUIBBbox({ uuid: 'b2', x: 0.5, y: 0.5, width: 0.1, height: 0.1 });
    runtime.state.image.bboxes.set('b1', b1);
    runtime.state.image.bboxes.set('b2', b2);
    rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);
    runtime.setTool('select');
"""

_IMPORTS = """
    const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
    const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
    const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
    const { handleMouseDown, handleMouseMove, handleMouseUp, handleClick } =
        await import('/app-static/js/app/image-tagging/canvas/interaction.js');
    const { setSelection, getSelectedViews } =
        await import('/app-static/js/app/image-tagging/canvas/selection.js');
"""


class TestModifierClickSelection:
    def test_ctrl_click_adds_second_box_and_plain_click_collapses(self, shared_page: Page):
        """Ctrl+click extends the selection; a plain click afterwards replaces it.

        The collapse half is what makes this a real assertion: a selection model
        that only ever grows would pass the first check.
        """
        page = shared_page
        result = page.evaluate("""async () => {""" + _IMPORTS + _SETUP + _MODIFIERS + _TWO_BOXES + """
            await clickAt(120, 90);                        // plain → b1 only
            const afterFirst = selectedIds();

            await clickAt(440, 330, { ctrl: true });       // ctrl → adds b2
            const afterCtrl = selectedIds();
            const transformerCount = runtime.canvas.transformer.nodes().length;

            await clickAt(120, 90);                        // plain → back to b1 alone
            const afterPlain = selectedIds();

            return { afterFirst, afterCtrl, transformerCount, afterPlain };
        }""")
        assert result["afterFirst"] == ["b1"]
        assert sorted(result["afterCtrl"]) == ["b1", "b2"]
        assert result["transformerCount"] == 2
        assert result["afterPlain"] == ["b1"]

    def test_shift_click_on_selected_box_removes_it(self, shared_page: Page):
        """A modifier click toggles: the second one takes the box back out."""
        page = shared_page
        result = page.evaluate("""async () => {""" + _IMPORTS + _SETUP + _MODIFIERS + _TWO_BOXES + """
            await clickAt(120, 90);
            await clickAt(440, 330, { shift: true });
            const both = selectedIds();
            await clickAt(440, 330, { shift: true });
            return { both, afterToggleOff: selectedIds() };
        }""")
        assert sorted(result["both"]) == ["b1", "b2"]
        assert result["afterToggleOff"] == ["b1"]

    def test_modifier_click_on_empty_canvas_keeps_selection(self, shared_page: Page):
        """A missed modifier click must not discard a selection being built.

        A plain click on empty canvas still clears — that contrast is the point.
        """
        page = shared_page
        result = page.evaluate("""async () => {""" + _IMPORTS + _SETUP + _MODIFIERS + _TWO_BOXES + """
            await clickAt(120, 90);
            await clickAt(440, 330, { ctrl: true });
            await clickAt(700, 500, { ctrl: true });      // empty space, modifier held
            const afterModifierMiss = selectedIds();
            await clickAt(700, 500);                      // empty space, no modifier
            return { afterModifierMiss, afterPlainMiss: selectedIds() };
        }""")
        assert sorted(result["afterModifierMiss"]) == ["b1", "b2"]
        assert result["afterPlainMiss"] == []


class TestDeleteSelection:
    def test_delete_removes_every_selected_box_and_one_undo_restores_all(self, shared_page: Page):
        """deleteSelected acts on the whole selection, as a single undo step.

        It used to read `transformer.nodes()[0]` and delete exactly one box no
        matter how many were selected.
        """
        page = shared_page
        result = page.evaluate(
            """async () => {"""
            + _IMPORTS
            + """
            const { deleteSelected } = await import('/app-static/js/app/image-tagging/canvas/image.js');
            """
            + _SETUP
            + _MODIFIERS
            + _TWO_BOXES
            + """
            runtime.pushState();   // baseline holding both boxes

            await clickAt(120, 90);
            await clickAt(440, 330, { ctrl: true });

            deleteSelected(runtime);
            const afterDelete = runtime.state.image.bboxes.size;
            const viewsAfterDelete = runtime.canvas.layer.find('.bbox').length;
            const selectionAfterDelete = selectedIds().length;

            runtime.undo();
            const afterUndo = [...runtime.state.image.bboxes.keys()].sort();

            return { afterDelete, viewsAfterDelete, selectionAfterDelete, afterUndo };
        }"""
        )
        assert result["afterDelete"] == 0
        assert result["viewsAfterDelete"] == 0
        assert result["selectionAfterDelete"] == 0
        assert result["afterUndo"] == ["b1", "b2"]


class TestMarqueeSelection:
    def test_marquee_selects_only_intersecting_boxes(self, shared_page: Page):
        """Shift+drag over empty canvas selects what the band touches, and nothing else."""
        page = shared_page
        result = page.evaluate("""async () => {""" + _IMPORTS + _SETUP + _MODIFIERS + _TWO_BOXES + """
            // Band (40,20)→(200,160) covers b1's (80,60)-(160,120) and misses
            // b2's (400,300)-(480,360) entirely.
            setPtr(40, 20);
            await handleMouseDown(makeModEvt({ shift: true, clientX: 40, clientY: 20 }), runtime);
            const startedMarquee = runtime.canvas.interactionState.marqueeStart !== null;
            const startedPanning = runtime.canvas.interactionState.isPanning;

            setPtr(200, 160);
            await handleMouseMove(makeModEvt({ shift: true, clientX: 200, clientY: 160 }), runtime);
            await handleMouseUp(makeModEvt({ shift: true, clientX: 200, clientY: 160 }), runtime);

            const selected = selectedIds();
            // The synthesized click that follows a drag must not wipe the result.
            handleClick(makeModEvt({ shift: true }), runtime);
            const afterTrailingClick = selectedIds();
            const marqueeCleanedUp = runtime.canvas.layer.find('.marquee').length;

            return { startedMarquee, startedPanning, selected, afterTrailingClick, marqueeCleanedUp };
        }""")
        assert result["startedMarquee"] is True
        # Shift+drag must claim the gesture instead of panning the viewport.
        assert result["startedPanning"] is False
        assert result["selected"] == ["b1"]
        assert result["afterTrailingClick"] == ["b1"]
        assert result["marqueeCleanedUp"] == 0

    def test_marquee_adds_to_an_existing_selection(self, shared_page: Page):
        """Shift is the extend modifier on click, so a shift-drag extends too."""
        page = shared_page
        result = page.evaluate("""async () => {""" + _IMPORTS + _SETUP + _MODIFIERS + _TWO_BOXES + """
            await clickAt(440, 330);            // b2 selected first

            setPtr(40, 20);
            await handleMouseDown(makeModEvt({ shift: true, clientX: 40, clientY: 20 }), runtime);
            setPtr(200, 160);
            await handleMouseMove(makeModEvt({ shift: true, clientX: 200, clientY: 160 }), runtime);
            await handleMouseUp(makeModEvt({ shift: true, clientX: 200, clientY: 160 }), runtime);

            return selectedIds().sort();
        }""")
        assert result == ["b1", "b2"]


class TestBboxClipboard:
    def test_copy_then_paste_onto_another_image_recreates_boxes(self, shared_page: Page):
        """Copied boxes survive an image change and paste as new boxes at the same coords."""
        page = shared_page
        result = page.evaluate(
            """async () => {"""
            + _IMPORTS
            + """
            const { copySelectedBboxes, pasteBboxes } =
                await import('/app-static/js/app/image-tagging/canvas/bboxClipboard.js');
            """
            + _SETUP
            + _MODIFIERS
            + _TWO_BOXES
            + """
            setSelection(runtime, runtime.canvas.layer.find('.bbox'));
            const copied = copySelectedBboxes(runtime);

            // Same canvas, different image — the clipboard must outlive it.
            runtime.setImage({
                uuid: 'img2', filename: 'test2.png', image_path: '/test2.png',
                img, bboxes: new Map(), dirty: false,
            });
            const beforePaste = runtime.state.image.bboxes.size;

            const pasted = await pasteBboxes(runtime);
            const boxes = [...runtime.state.image.bboxes.values()]
                .map(b => ({ uuid: b.uuid, x: b.x, y: b.y, w: b.width, h: b.height }))
                .sort((a, b) => a.x - b.x);

            return {
                copied, beforePaste, pasted, boxes,
                views: runtime.canvas.layer.find('.bbox').length,
                selectedAfterPaste: selectedIds().length,
            };
        }"""
        )
        assert result["copied"] == 2
        assert result["beforePaste"] == 0
        assert result["pasted"] == 2
        # Fresh identities — pasting must never reuse the source uuids, or the
        # two images would fight over the same bbox rows on save.
        assert {b["uuid"] for b in result["boxes"]}.isdisjoint({"b1", "b2"})
        assert result["boxes"][0]["x"] == 0.1
        assert result["boxes"][0]["w"] == 0.1
        assert result["boxes"][1]["x"] == 0.5
        assert result["views"] == 2
        assert result["selectedAfterPaste"] == 2

    def test_paste_skips_boxes_already_present_unless_forced(self, shared_page: Page):
        """The duplicate check is what makes paste safe to mash on every frame."""
        page = shared_page
        result = page.evaluate(
            """async () => {"""
            + _IMPORTS
            + """
            const { copySelectedBboxes, pasteBboxes } =
                await import('/app-static/js/app/image-tagging/canvas/bboxClipboard.js');
            """
            + _SETUP
            + _MODIFIERS
            + _TWO_BOXES
            + """
            setSelection(runtime, runtime.canvas.layer.find('.bbox'));
            copySelectedBboxes(runtime);

            // New image already carrying a box in b1's exact position.
            const existing = createUIBBbox({ uuid: 'existing', x: 0.1, y: 0.1, width: 0.1, height: 0.1 });
            runtime.setImage({
                uuid: 'img2', filename: 'test2.png', image_path: '/test2.png',
                img, bboxes: new Map([['existing', existing]]), dirty: false,
            });

            const pastedSkipping = await pasteBboxes(runtime);
            const afterSkipping = runtime.state.image.bboxes.size;

            const pastedForced = await pasteBboxes(runtime, { force: true });
            const afterForced = runtime.state.image.bboxes.size;

            return { pastedSkipping, afterSkipping, pastedForced, afterForced };
        }"""
        )
        # b1 duplicates `existing`; b2 does not.
        assert result["pastedSkipping"] == 1
        assert result["afterSkipping"] == 2
        # Forced ignores the overlap and pastes both again.
        assert result["pastedForced"] == 2
        assert result["afterForced"] == 4
