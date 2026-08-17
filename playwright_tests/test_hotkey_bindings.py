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

    def test_s_maps_to_save(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return actionForCombo('s');
        }""")
        assert result == "action:save"

    def test_i_maps_to_run_inference(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return actionForCombo('i');
        }""")
        assert result == "action:runInference"

    def test_a_maps_to_mark_absent(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return actionForCombo('a');
        }""")
        assert result == "action:markAbsent"

    def test_r_maps_to_reset_labels(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return actionForCombo('r');
        }""")
        assert result == "action:resetLabels"

    def test_v_maps_to_verify_all(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return actionForCombo('v');
        }""")
        assert result == "action:verifyAll"

    def test_tools_moved_to_q_and_b(self, page: Page, base_url: str):
        """Select and Draw-bbox vacated V and R for the bulk actions."""
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return { q: actionForCombo('q'), b: actionForCombo('b') };
        }""")
        assert result["q"] == "action:select"
        assert result["b"] == "action:bbox"

    def test_no_fixed_action_ships_without_a_key(self, page: Page, base_url: str):
        """The defaults are an object literal keyed by combo, so reusing a key is
        not an error — the later entry silently wins and the earlier action ships
        unreachable. That shows up here as an action with no combo at all.

        Deliberately not asserting one combo per action: Delete and Delete-image
        each answer to two keys (Delete and Backspace), which is intended.
        """
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const m = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            await m.initHotkeyBindings();
            return m.FIXED_ACTIONS.map(a => a.id).filter(a => m.combosForAction(a).length === 0);
        }""")
        assert result == []

    def test_save_and_inference_listed_as_configurable_actions(self, page: Page, base_url: str):
        """Both appear in FIXED_ACTIONS, which is what the config panel renders."""
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { FIXED_ACTIONS } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return FIXED_ACTIONS.map(a => a.id);
        }""")
        assert "action:save" in result
        assert "action:runInference" in result
        assert "action:markAbsent" in result
        assert "action:resetLabels" in result
        assert "action:verifyAll" in result

    def test_delete_key_maps_to_delete(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            return actionForCombo('delete');
        }""")
        assert result == "action:delete"


# ---------------------------------------------------------------------------
# New defaults merged into an already-stored binding map
# ---------------------------------------------------------------------------


class TestNewDefaultsMerged:
    """Bindings are persisted whole, so an action added to DEFAULT_BINDINGS after
    a user already has a saved map must be merged in on load — otherwise the new
    hotkey silently does nothing for every existing user."""

    STORAGE_KEY = "ww_hotkey_bindings"

    def _seed(self, page: Page, mapping: dict):
        import json

        page.add_init_script(f"localStorage.setItem({json.dumps(self.STORAGE_KEY)}, {json.dumps(json.dumps(mapping))})")

    def _init_and_read(self, page: Page, combos: list):
        return page.evaluate(
            """async (combos) => {
            const m = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            await m.initHotkeyBindings();
            const out = {};
            for (const c of combos) out[c] = m.actionForCombo(c);
            return out;
        }""",
            combos,
        )

    def test_stored_map_without_the_new_actions_gains_them(self, page: Page, base_url: str):
        self._seed(page, {"k": "action:select", "ctrl+z": "action:undo"})
        goto(page, base_url)
        result = self._init_and_read(page, ["s", "i", "k"])
        assert result["s"] == "action:save"
        assert result["i"] == "action:runInference"
        assert result["k"] == "action:select"  # existing bindings untouched

    def test_the_legacy_tool_layout_frees_v_and_r(self, page: Page, base_url: str):
        """Every pre-existing user held V=Select and R=Draw-bbox, the old defaults.
        Because _mergeNewDefaults skips a default whose combo is taken, Verify and
        Reset arrived unbound for all of them — the two headline actions of the
        release that added them were unreachable without manual rebinding."""
        self._seed(page, {"v": "action:select", "r": "action:bbox", "ctrl+z": "action:undo"})
        goto(page, base_url)
        result = self._init_and_read(page, ["v", "r", "q", "b", "ctrl+z"])
        assert result["v"] == "action:verifyAll"
        assert result["r"] == "action:resetLabels"
        # The tools move to their new defaults rather than being dropped.
        assert result["q"] == "action:select"
        assert result["b"] == "action:bbox"
        assert result["ctrl+z"] == "action:undo"  # unrelated bindings untouched

    def test_a_tool_rebound_by_hand_is_not_migrated(self, page: Page, base_url: str):
        """Only the exact old layout migrates. Someone who chose V for Draw-bbox
        keeps it, and Verify stays unbound rather than stealing their key."""
        self._seed(page, {"v": "action:bbox"})
        goto(page, base_url)
        result = self._init_and_read(page, ["v", "b"])
        assert result["v"] == "action:bbox"
        assert result["b"] is None

    def test_migration_skips_a_tool_whose_new_combo_is_taken(self, page: Page, base_url: str):
        """Q already spoken for means there is nowhere free to move Select to, so
        nothing moves — a migration must never silently drop a binding."""
        self._seed(page, {"v": "action:select", "q": "action:undo"})
        goto(page, base_url)
        result = self._init_and_read(page, ["v", "q"])
        assert result["v"] == "action:select"
        assert result["q"] == "action:undo"

    def test_default_combo_already_used_is_not_stolen(self, page: Page, base_url: str):
        self._seed(page, {"s": "action:select"})
        goto(page, base_url)
        result = self._init_and_read(page, ["s"])
        assert result["s"] == "action:select"

    def test_action_bound_to_another_key_keeps_that_key(self, page: Page, base_url: str):
        """A user who rebound Save to K must not also get S."""
        self._seed(page, {"k": "action:save"})
        goto(page, base_url)
        result = self._init_and_read(page, ["k", "s"])
        assert result["k"] == "action:save"
        assert result["s"] is None


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
            // 'q' defaults to action:select
            setBinding('action:select', 'x');    // x → select; q should now be unbound
            return actionForCombo('q');
        }""")
        assert result is None

    def test_set_binding_steals_combo_from_other_action(self, page: Page, base_url: str):
        """Assigning an already-used combo reassigns it, leaving the previous action without that combo."""
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { setBinding, actionForCombo } = await import('/app-static/js/app/image-tagging/hotkeyBindings.js');
            // 'q' → action:select by default; steal it for undo
            setBinding('action:undo', 'q');
            return { undoCombo: actionForCombo('q'), selectCombo: actionForCombo('q') !== 'action:select' };
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
            return actionForCombo('q');   // action:select should be unaffected
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
                q: actionForCombo('q'),
                x: actionForCombo('x'),   // should be gone after reset
            };
        }""")
        assert result["ctrlZ"] == "action:undo"
        assert result["q"] == "action:select"
        assert result["x"] is None
