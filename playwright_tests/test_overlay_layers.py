"""
Tests for the unified overlay layer stack (slice 6, PR 6.1).

Covers ui/utils/modalZIndex.js + ui/utils/modalShell.js:
  - the modal z-index band is bounded and stays below the bottom-bars layer
  - the counter resets once every modal has closed
  - ESC closes strictly top-down, across *all* overlay kinds (modal + popup)
  - arbitrary modal nesting and the cascade offset are preserved
  - the dialog traps Tab focus and restores focus to its trigger on close

These build DOM inside the live page so /app-static/ imports resolve, following
the pattern in test_dom_components.py.

Run with:
    pytest playwright_tests/test_overlay_layers.py -p no:asyncio -v
"""

from playwright.sync_api import Page

# ---------------------------------------------------------------------------
# Layer scale — the modal band must be bounded
# ---------------------------------------------------------------------------


class TestModalLayerBand:
    def test_modal_zindex_never_reaches_the_bottom_bars_layer(self, shared_page: Page):
        """#bottom-bars sits above modals — that must be an invariant, not arithmetic.

        The counter used to increment forever, so ~900 modal opens in one
        session silently inverted the documented ordering.
        """
        result = shared_page.evaluate("""async () => {
            const { acquireModalLayer, LAYERS } = await import('/app-static/js/ui/utils/modalZIndex.js');
            let max = 0;
            const held = [];
            for (let i = 0; i < 1000; i++) {
                const l = acquireModalLayer();
                held.push(l);
                max = Math.max(max, l.zIndex);
            }
            // Release so the band rewinds for the tests that follow.
            held.forEach(l => l.release());
            return { max, bottomBars: LAYERS.bottomBars };
        }""")
        assert (
            result["max"] < result["bottomBars"]
        ), f"modal z-index reached {result['max']}, at or above #bottom-bars ({result['bottomBars']})"

    def test_counter_resets_once_all_modals_close(self, shared_page: Page):
        """Every layer released means nothing is open, so the band can start over."""
        result = shared_page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const a = createModalShell(); a.open();
            const first = parseInt(a.overlay.style.zIndex, 10);
            a.close();
            const b = createModalShell(); b.open();
            const second = parseInt(b.overlay.style.zIndex, 10);
            b.close();
            return { first, second };
        }""")
        assert (
            result["second"] == result["first"]
        ), f"expected the band to reset to {result['first']}, got {result['second']}"

    def test_nested_modals_still_stack_upward(self, shared_page: Page):
        """⛔ Arbitrary nesting is a feature — the inner modal must sit above."""
        result = shared_page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const outer = createModalShell(); outer.open();
            const inner = createModalShell(); inner.open();
            const res = {
                outer: parseInt(outer.overlay.style.zIndex, 10),
                inner: parseInt(inner.overlay.style.zIndex, 10),
                cascaded: inner.dialog.style.transform !== '',
            };
            inner.close(); outer.close();
            return res;
        }""")
        assert result["inner"] > result["outer"]
        assert result["cascaded"], "the nested dialog lost its cascade offset"

    def test_band_is_not_rewound_while_a_layer_is_still_held(self, shared_page: Page):
        """Layer lifetime is the acquire/release pair, not ESC registration.

        The canvas modal in app/review/main.js allocates its z-index, appends
        itself, and only registers an ESC handler after two awaits. Rewinding the
        band on an empty ESC stack handed the next overlay the same z-index while
        that modal was still on screen.
        """
        result = shared_page.evaluate("""async () => {
            const { acquireModalLayer } = await import('/app-static/js/ui/utils/modalZIndex.js');
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            // An overlay on screen that has not (yet) registered an ESC handler.
            const held = acquireModalLayer({ dialog: true });
            // A whole overlay opens and closes during that window.
            const s = createModalShell(); s.open(); s.close();
            const after = acquireModalLayer({ dialog: true });
            const res = { held: held.zIndex, after: after.zIndex };
            after.release(); held.release();
            return res;
        }""")
        assert (
            result["after"] > result["held"]
        ), f"band rewound to {result['after']} while a layer at {result['held']} was still held"

    def test_double_release_does_not_rewind_the_band(self, shared_page: Page):
        """release() is idempotent — a double close must not free someone else's slot."""
        result = shared_page.evaluate("""async () => {
            const { acquireModalLayer } = await import('/app-static/js/ui/utils/modalZIndex.js');
            const outer = acquireModalLayer({ dialog: true });
            const inner = acquireModalLayer({ dialog: true });
            inner.release();
            inner.release();  // double close
            const next = acquireModalLayer({ dialog: true });
            const res = { outer: outer.zIndex, next: next.zIndex };
            next.release(); outer.release();
            return res;
        }""")
        assert result["next"] > result["outer"], "a double release rewound the band while an outer layer was still open"

    def test_a_popup_does_not_cascade_the_dialog_opened_from_it(self, shared_page: Page):
        """Cascade depth counts dialogs only.

        Context-menu items await onSelect *before* close(), so a menu that opens
        a confirm dialog is still on the ESC stack at that point. Counting it
        staircased an otherwise-centred first-level dialog 20px off-centre.
        """
        result = shared_page.evaluate("""async () => {
            const { openAnchoredPopup } = await import('/app-static/js/ui/utils/anchoredPopup.js');
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');

            const anchor = document.createElement('button');
            anchor.textContent = 'anchor';
            document.body.appendChild(anchor);
            const popup = openAnchoredPopup(anchor, document.createElement('div'));

            // Dialog opened from the popup, while the popup is still open.
            const shell = createModalShell(); shell.open();
            const transform = shell.dialog.style.transform;
            shell.close(); popup.close(); anchor.remove();
            return transform;
        }""")
        assert result == "", f"a first-level dialog was cascaded by an open popup (transform={result!r})"


# ---------------------------------------------------------------------------
# Anchored popups above the bottom bars
# ---------------------------------------------------------------------------


class TestAnchoredPopupLayer:
    def test_a_caller_set_zindex_survives(self, shared_page: Page):
        """Popups anchored inside #bottom-bars must clear it, so they set their own.

        openAnchoredPopup used to overwrite that with a modal-band value, which
        dropped activeModelBar's .vis-popup / .tc-run-picker and labelList's
        .label-preset-menu behind the very bar they are anchored to.
        """
        result = shared_page.evaluate("""async () => {
            const { openAnchoredPopup } = await import('/app-static/js/ui/utils/anchoredPopup.js');
            const { LAYERS } = await import('/app-static/js/ui/utils/modalZIndex.js');

            const anchor = document.createElement('button');
            document.body.appendChild(anchor);

            // (a) inline z-index, as activeModelBar sets it
            const inlineEl = document.createElement('div');
            inlineEl.style.cssText = 'z-index:var(--layer-bottom-bars-menu)';
            const a = openAnchoredPopup(anchor, inlineEl);
            const inlineZ = parseInt(getComputedStyle(inlineEl).zIndex, 10);
            a.close();

            // (b) z-index from a class rule, as labelList's preset menu has
            const classEl = document.createElement('div');
            classEl.className = 'label-preset-menu';
            const b = openAnchoredPopup(anchor, classEl);
            const classZ = parseInt(getComputedStyle(classEl).zIndex, 10);
            b.close();

            // (c) no opinion → allocated inside the modal band
            const plainEl = document.createElement('div');
            const c = openAnchoredPopup(anchor, plainEl);
            const plainZ = parseInt(getComputedStyle(plainEl).zIndex, 10);
            c.close();

            anchor.remove();
            return { inlineZ, classZ, plainZ, bottomBars: LAYERS.bottomBars, modalMax: LAYERS.modalMax };
        }""")
        assert result["inlineZ"] > result["bottomBars"], "inline z-index was overwritten with a modal-band value"
        assert result["classZ"] > result["bottomBars"], "a class-rule z-index was overwritten"
        assert (
            result["plainZ"] <= result["modalMax"]
        ), "a popup with no caller z-index should still be allocated inside the modal band"


# ---------------------------------------------------------------------------
# ESC ordering across overlay kinds
# ---------------------------------------------------------------------------


class TestEscStack:
    def test_esc_closes_only_the_topmost_modal(self, shared_page: Page):
        result = shared_page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            let outerClosed = false, innerClosed = false;
            const outer = createModalShell({ onClose: () => { outerClosed = true; } }); outer.open();
            const inner = createModalShell({ onClose: () => { innerClosed = true; } }); inner.open();
            window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
            const res = { outerClosed, innerClosed };
            outer.close(); inner.close();
            return res;
        }""")
        assert result["innerClosed"] is True
        assert result["outerClosed"] is False

    def test_esc_closes_an_anchored_popup_before_the_modal_beneath_it(self, shared_page: Page):
        """anchoredPopup had no ESC handling at all, so ordering was undefined."""
        result = shared_page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const { openAnchoredPopup } = await import('/app-static/js/ui/utils/anchoredPopup.js');

            let modalClosed = false;
            const modal = createModalShell({ onClose: () => { modalClosed = true; } });
            modal.open();

            const anchor = document.createElement('button');
            anchor.textContent = 'anchor';
            modal.dialog.appendChild(anchor);
            const content = document.createElement('div');
            content.textContent = 'popup';
            const popup = openAnchoredPopup(anchor, content);

            window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
            const res = {
                popupClosed: !document.body.contains(content),
                modalClosed,
            };
            popup.close(); modal.close();
            return res;
        }""")
        assert result["popupClosed"] is True, "ESC did not close the anchored popup"
        assert result["modalClosed"] is False, "ESC closed the modal underneath the popup"


# ---------------------------------------------------------------------------
# Focus management
# ---------------------------------------------------------------------------


class TestModalFocusTrap:
    def test_tab_from_the_last_focusable_wraps_to_the_first(self, shared_page: Page):
        result = shared_page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const shell = createModalShell();
            const first = document.createElement('button'); first.textContent = 'first';
            const last = document.createElement('button'); last.textContent = 'last';
            shell.dialog.append(first, last);
            shell.open();

            last.focus();
            shell.dialog.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'Tab', bubbles: true, cancelable: true,
            }));
            const wrapped = document.activeElement === first;
            shell.close();
            return wrapped;
        }""")
        assert result is True, "Tab escaped the modal instead of wrapping to the first control"

    def test_shift_tab_from_the_first_focusable_wraps_to_the_last(self, shared_page: Page):
        result = shared_page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const shell = createModalShell();
            const first = document.createElement('button'); first.textContent = 'first';
            const last = document.createElement('button'); last.textContent = 'last';
            shell.dialog.append(first, last);
            shell.open();

            first.focus();
            shell.dialog.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'Tab', shiftKey: true, bubbles: true, cancelable: true,
            }));
            const wrapped = document.activeElement === last;
            shell.close();
            return wrapped;
        }""")
        assert result is True

    def test_open_moves_focus_into_the_dialog(self, shared_page: Page):
        """The trap listens on the dialog, so it is inert until focus is inside.

        Opening from the keyboard used to leave focus on the trigger behind the
        overlay: _trapTab never fired and Tab walked the page underneath.
        """
        result = shared_page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const trigger = document.createElement('button');
            trigger.textContent = 'open me';
            document.body.appendChild(trigger);
            trigger.focus();

            const shell = createModalShell();
            const inner = document.createElement('button'); inner.textContent = 'inner';
            shell.dialog.appendChild(inner);
            shell.open();
            const res = {
                onInner: document.activeElement === inner,
                insideDialog: shell.dialog.contains(document.activeElement),
            };
            shell.close(); trigger.remove();
            return res;
        }""")
        assert result["insideDialog"] is True, "focus stayed outside the dialog, leaving the Tab trap inert"
        assert result["onInner"] is True, "focus should land on the dialog's first focusable control"

    def test_an_empty_dialog_still_takes_focus(self, shared_page: Page):
        """No focusable content is the case where the trap matters most."""
        result = shared_page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const trigger = document.createElement('button');
            document.body.appendChild(trigger);
            trigger.focus();

            const shell = createModalShell();
            shell.dialog.textContent = 'nothing focusable here';
            shell.open();
            const onDialog = document.activeElement === shell.dialog;
            shell.close(); trigger.remove();
            return onDialog;
        }""")
        assert result is True, "an empty dialog left focus on the trigger behind the overlay"

    def test_focus_returns_to_the_trigger_on_close(self, shared_page: Page):
        result = shared_page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const trigger = document.createElement('button');
            trigger.textContent = 'open me';
            document.body.appendChild(trigger);
            trigger.focus();

            const shell = createModalShell();
            const inner = document.createElement('button'); inner.textContent = 'inner';
            shell.dialog.appendChild(inner);
            shell.open();
            inner.focus();
            shell.close();

            const restored = document.activeElement === trigger;
            trigger.remove();
            return restored;
        }""")
        assert result is True, "focus was not restored to the element that opened the modal"
