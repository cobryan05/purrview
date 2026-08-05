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
            const { nextModalZIndex, LAYERS } = await import('/app-static/js/ui/utils/modalZIndex.js');
            let max = 0;
            for (let i = 0; i < 1000; i++) max = Math.max(max, nextModalZIndex());
            // Rewind the shared band for the tests that follow — allocating
            // without registering leaves the counter parked at the ceiling.
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const s = createModalShell(); s.open(); s.close();
            return { max, bottomBars: LAYERS.bottomBars };
        }""")
        assert (
            result["max"] < result["bottomBars"]
        ), f"modal z-index reached {result['max']}, at or above #bottom-bars ({result['bottomBars']})"

    def test_counter_resets_once_all_modals_close(self, shared_page: Page):
        """An empty ESC stack means nothing is open, so the band can start over."""
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
