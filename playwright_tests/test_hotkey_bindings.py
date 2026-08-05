"""
Tests for app/image-tagging/hotkeyBindings.js

Each test gets a fresh page so the module-level _bindings singleton starts from
localStorage (which is empty in a fresh context → DEFAULT_BINDINGS apply).

The module persists to both localStorage and the DB settings API.  The DB
settings proxy in our test server returns {"status":"success","settings":{}} so
getSetting returns null → the module falls back to localStorage defaults.
"""

import pytest
from playwright.sync_api import Page


def goto(page: Page, base_url: str):
    page.goto(base_url)


# ---------------------------------------------------------------------------
# eventToCombo
# ---------------------------------------------------------------------------


class TestEventToCombo:
    def test_bare_key(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { eventToCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return eventToCombo({ key: 'v', ctrlKey: false, metaKey: false, altKey: false, shiftKey: false });
        }""")
        assert result == "v"

    def test_ctrl_z(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { eventToCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return eventToCombo({ key: 'z', ctrlKey: true, metaKey: false, altKey: false, shiftKey: false });
        }""")
        assert result == "ctrl+z"

    def test_meta_treated_as_ctrl(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { eventToCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return eventToCombo({ key: 'z', ctrlKey: false, metaKey: true, altKey: false, shiftKey: false });
        }""")
        assert result == "ctrl+z"

    def test_ctrl_shift_z(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { eventToCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return eventToCombo({ key: 'z', ctrlKey: true, metaKey: false, altKey: false, shiftKey: true });
        }""")
        assert result == "ctrl+shift+z"

    def test_key_lowercased(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { eventToCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return eventToCombo({ key: 'Delete', ctrlKey: false, metaKey: false, altKey: false, shiftKey: false });
        }""")
        assert result == "delete"

    def test_alt_key_included(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { eventToCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return eventToCombo({ key: 'x', ctrlKey: false, metaKey: false, altKey: true, shiftKey: false });
        }""")
        assert result == "alt+x"


# ---------------------------------------------------------------------------
# comboLabel
# ---------------------------------------------------------------------------


class TestComboLabel:
    def test_single_lowercase_char_uppercased(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { comboLabel } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return comboLabel('v');
        }""")
        assert result == "V"

    def test_multi_modifier_formatted(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { comboLabel } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return comboLabel('ctrl+shift+z');
        }""")
        assert result == "Ctrl+Shift+Z"

    def test_empty_combo_returns_dash(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { comboLabel } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return comboLabel('');
        }""")
        assert result == "—"

    def test_null_combo_returns_dash(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { comboLabel } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return comboLabel(null);
        }""")
        assert result == "—"


# ---------------------------------------------------------------------------
# actionForCombo / combosForAction — default bindings
# ---------------------------------------------------------------------------


class TestDefaultBindings:
    def test_ctrl_z_maps_to_undo(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return actionForCombo('ctrl+z');
        }""")
        assert result == "action:undo"

    def test_ctrl_shift_z_maps_to_redo(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return actionForCombo('ctrl+shift+z');
        }""")
        assert result == "action:redo"

    def test_unknown_combo_returns_null(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return actionForCombo('ctrl+q');
        }""")
        assert result is None

    def test_combos_for_action_undo(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { combosForAction } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return combosForAction('action:undo');
        }""")
        assert "ctrl+z" in result

    def test_delete_key_maps_to_delete(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return actionForCombo('delete');
        }""")
        assert result == "action:delete"


# ---------------------------------------------------------------------------
# setBinding / clearBinding / resetBindings
# ---------------------------------------------------------------------------


class TestSetBinding:
    def test_new_binding_registered(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { setBinding, actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            setBinding('action:select', 'q');
            return actionForCombo('q');
        }""")
        assert result == "action:select"

    def test_set_binding_removes_old_combo_for_action(self, page: Page, base_url: str):
        """When an action gets a new combo, the prior combo is unbound."""
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { setBinding, actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            // 'v' defaults to action:select
            setBinding('action:select', 'q');    // q → select; v should now be unbound
            return actionForCombo('v');
        }""")
        assert result is None

    def test_set_binding_steals_combo_from_other_action(self, page: Page, base_url: str):
        """Assigning an already-used combo reassigns it, leaving the previous action without that combo."""
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { setBinding, actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            // 'v' → action:select by default; steal it for undo
            setBinding('action:undo', 'v');
            return { undoCombo: actionForCombo('v'), selectCombo: actionForCombo('v') !== 'action:select' };
        }""")
        assert result["undoCombo"] == "action:undo"
        assert result["selectCombo"] is True


class TestClearBinding:
    def test_clear_removes_all_combos_for_action(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { clearBinding, actionForCombo, combosForAction } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            clearBinding('action:undo');
            return {
                ctrlZ: actionForCombo('ctrl+z'),
                combos: combosForAction('action:undo'),
            };
        }""")
        assert result["ctrlZ"] is None
        assert result["combos"] == []

    def test_clear_does_not_affect_other_actions(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { clearBinding, actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            clearBinding('action:undo');
            return actionForCombo('v');   // action:select should be unaffected
        }""")
        assert result == "action:select"


class TestResetBindings:
    def test_reset_restores_defaults(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { setBinding, clearBinding, resetBindings, actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            // scramble the bindings
            clearBinding('action:undo');
            setBinding('action:select', 'x');
            resetBindings();
            return {
                ctrlZ: actionForCombo('ctrl+z'),
                v: actionForCombo('v'),
                x: actionForCombo('x'),   // should be gone after reset
            };
        }""")
        assert result["ctrlZ"] == "action:undo"
        assert result["v"] == "action:select"
        assert result["x"] is None
