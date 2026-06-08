"""
Tests for app/review/filters.js — createFilterPanel

The filter panel is a pure DOM component: it accepts label/tag Maps, builds a
UI, and exposes getState/setState/isDefault/getActiveFilterCount.  No network
calls are made by the panel itself; all API dependency is injected via the Maps.

Layout after the multi-select refactor:
  toolbarControls[0] — Tags multi-select dropdown  (.ms-dropdown[data-filter-key='tags'])
  toolbarControls[1] — Labels multi-select dropdown (.ms-dropdown[data-filter-key='labels'])
  panel.element      — Date range + Filename + Advanced (image-label) filters

All tests share one pre-navigated `shared_page` (module-scoped, see
conftest.py); each test builds and removes its own panel element so the
shared page stays clean.
"""
import pytest
from playwright.sync_api import Page

UNVERIFIED_UUID = "00000000-0000-0000-0000-000000000001"


# Shared JS preamble — builds filter panel + mounts ALL elements into the DOM.
# Stores the panel on window.__fp so it can be accessed from separate evaluate() calls.
_SETUP = f"""
    const {{ createFilterPanel }} = await import('/app-static/js/app/review/filters.js');

    const labels = new Map([
        ['lbl-cat', {{ uuid: 'lbl-cat', name: 'Cat', color: '#ff0000', parentUuid: null }}],
        ['lbl-dog', {{ uuid: 'lbl-dog', name: 'Dog', color: '#00ff00', parentUuid: null }}],
    ]);
    const tags = new Map([
        ['{UNVERIFIED_UUID}', {{ uuid: '{UNVERIFIED_UUID}', name: 'Unverified', color: null }}],
        ['tag-reviewed',      {{ uuid: 'tag-reviewed',      name: 'Reviewed',   color: '#4c4' }}],
    ]);

    let lastState = null;
    const panel = createFilterPanel(labels, tags, (s) => {{ lastState = s; }});
    window.__fp = panel;
    window.__fpLastState = () => lastState;

    // Mount panel element and toolbar controls (Tags + Labels dropdowns) into DOM.
    document.body.appendChild(panel.element);
    panel.toolbarControls.forEach(el => document.body.appendChild(el));
"""

# For use within the SAME evaluate call as _SETUP (panel is in-scope).
_TEARDOWN = """
    panel.element.remove();
    panel.toolbarControls.forEach(el => el.remove());
    panel.destroy && panel.destroy();
    window.__fp = null;
"""

# For use in a SEPARATE evaluate call (panel is accessed via window.__fp).
_TEARDOWN_EXT = """
    if (window.__fp) {
        window.__fp.element.remove();
        window.__fp.toolbarControls.forEach(el => el.remove());
        window.__fp.destroy && window.__fp.destroy();
        window.__fp = null;
    }
"""


# ── Helper: open a ms-dropdown by data-filter-key, interact, close ─────────

def _open_dropdown(page, filter_key):
    """Open the named dropdown (click its trigger)."""
    page.locator(f".ms-dropdown[data-filter-key='{filter_key}'] .ms-dropdown__trigger").click()
    page.wait_for_timeout(100)


def _close_dropdown(page, filter_key):
    """Close the named dropdown (click outside)."""
    page.mouse.click(0, 0)
    page.wait_for_timeout(100)


def _check_item(page, filter_key, uuid):
    cb = page.locator(f".ms-dropdown[data-filter-key='{filter_key}'] input[value='{uuid}']")
    cb.check()
    page.wait_for_timeout(100)


def _uncheck_item(page, filter_key, uuid):
    cb = page.locator(f".ms-dropdown[data-filter-key='{filter_key}'] input[value='{uuid}']")
    cb.uncheck()
    page.wait_for_timeout(100)


# ── Default state ────────────────────────────────────────────────────────────

class TestFilterPanelDefaults:
    def test_default_tag_uuids_contains_unverified(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const s = panel.getState();
            {_TEARDOWN}
            return s.tagUuids;
        }}""")
        assert UNVERIFIED_UUID in result

    def test_default_label_uuids_empty(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const s = panel.getState();
            {_TEARDOWN}
            return s.labelUuids;
        }}""")
        assert result == []

    def test_is_default_true_initially(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const d = panel.isDefault();
            {_TEARDOWN}
            return d;
        }}""")
        assert result is True

    def test_active_filter_count_is_one_initially(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const c = panel.getActiveFilterCount();
            {_TEARDOWN}
            return c;
        }}""")
        assert result == 1

    def test_filename_search_empty_initially(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const s = panel.getState();
            {_TEARDOWN}
            return s.filenameSearch;
        }}""")
        assert result == ""

    def test_match_modes_default_to_any(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const s = panel.getState();
            {_TEARDOWN}
            return {{ tagMatchMode: s.tagMatchMode, labelMatchMode: s.labelMatchMode }};
        }}""")
        assert result["tagMatchMode"] == "any"
        assert result["labelMatchMode"] == "any"


# ── Tags dropdown ────────────────────────────────────────────────────────────

class TestFilterPanelTagsDropdown:
    def test_unverified_checkbox_is_checked_by_default(self, shared_page: Page):
        page = shared_page
        page.evaluate(f"async () => {{ {_SETUP} }}")

        _open_dropdown(page, "tags")
        cb = page.locator(f".ms-dropdown[data-filter-key='tags'] input[value='{UNVERIFIED_UUID}']")
        assert cb.is_checked()

        page.evaluate(f"() => {{ {_TEARDOWN_EXT} }}")

    def test_unchecking_unverified_removes_from_state(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            // Simulate unchecking by driving the checkbox directly.
            const cb = panel.toolbarControls[0].querySelector('input[value="{UNVERIFIED_UUID}"]');
            cb.checked = false;
            cb.dispatchEvent(new Event('change'));
            const s = panel.getState();
            {_TEARDOWN}
            return {{ tagUuids: s.tagUuids, showAllBboxes: s.showAllBboxes }};
        }}""")
        assert UNVERIFIED_UUID not in result["tagUuids"]
        assert result["showAllBboxes"] is True

    def test_no_tags_checked_sets_show_all(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.toolbarControls[0].querySelectorAll('input[type="checkbox"]').forEach(cb => {{
                cb.checked = false;
                cb.dispatchEvent(new Event('change'));
            }});
            const s = panel.getState();
            {_TEARDOWN}
            return s.showAllBboxes;
        }}""")
        assert result is True

    def test_checking_second_tag_adds_to_state(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const cb = panel.toolbarControls[0].querySelector('input[value="tag-reviewed"]');
            cb.checked = true;
            cb.dispatchEvent(new Event('change'));
            const s = panel.getState();
            {_TEARDOWN}
            return s.tagUuids;
        }}""")
        assert "tag-reviewed" in result
        assert UNVERIFIED_UUID in result

    def test_unchecking_makes_not_default(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const cb = panel.toolbarControls[0].querySelector('input[value="{UNVERIFIED_UUID}"]');
            cb.checked = false;
            cb.dispatchEvent(new Event('change'));
            {_TEARDOWN}
            return panel.isDefault();
        }}""")
        assert result is False

    def test_trigger_shows_selection_summary(self, shared_page: Page):
        """Trigger button text summarises the current selection."""
        page = shared_page
        page.evaluate(f"async () => {{ {_SETUP} }}")

        trigger = page.locator(".ms-dropdown[data-filter-key='tags'] .ms-dropdown__trigger")
        text = trigger.text_content()
        # Default: one item selected → shows the tag name
        assert "Unverified" in text

        page.evaluate(f"() => {{ {_TEARDOWN_EXT} }}")

    def test_all_of_mode_updates_state(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            // Click the "All of" mode button
            const allBtn = panel.toolbarControls[0].querySelector('.ms-mode-btn[data-mode="all"]');
            allBtn.click();
            const s = panel.getState();
            {_TEARDOWN}
            return s.tagMatchMode;
        }}""")
        assert result == "all"


# ── Labels dropdown ───────────────────────────────────────────────────────────

class TestFilterPanelLabelsDropdown:
    def test_no_labels_selected_by_default(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const s = panel.getState();
            {_TEARDOWN}
            return s.labelUuids;
        }}""")
        assert result == []

    def test_selecting_label_updates_state(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const cb = panel.toolbarControls[1].querySelector('input[value="lbl-cat"]');
            cb.checked = true;
            cb.dispatchEvent(new Event('change'));
            const s = panel.getState();
            {_TEARDOWN}
            return s.labelUuids;
        }}""")
        assert "lbl-cat" in result

    def test_selecting_label_increments_count(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const cb = panel.toolbarControls[1].querySelector('input[value="lbl-dog"]');
            cb.checked = true;
            cb.dispatchEvent(new Event('change'));
            {_TEARDOWN}
            return panel.getActiveFilterCount();
        }}""")
        assert result == 2  # UNVERIFIED tag + one label

    def test_deselecting_label_decrements_count(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const cb = panel.toolbarControls[1].querySelector('input[value="lbl-cat"]');
            cb.checked = true;
            cb.dispatchEvent(new Event('change'));
            cb.checked = false;
            cb.dispatchEvent(new Event('change'));
            {_TEARDOWN}
            return panel.getActiveFilterCount();
        }}""")
        assert result == 1  # back to just the tag

    def test_all_of_mode_updates_label_match_mode(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const allBtn = panel.toolbarControls[1].querySelector('.ms-mode-btn[data-mode="all"]');
            allBtn.click();
            const s = panel.getState();
            {_TEARDOWN}
            return s.labelMatchMode;
        }}""")
        assert result == "all"

    def test_selecting_label_makes_not_default(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const cb = panel.toolbarControls[1].querySelector('input[value="lbl-cat"]');
            cb.checked = true;
            cb.dispatchEvent(new Event('change'));
            {_TEARDOWN}
            return panel.isDefault();
        }}""")
        assert result is False


# ── Advanced / image-label filters ───────────────────────────────────────────

class TestFilterPanelImageLabelFilters:
    def test_add_image_label_filter(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const labelSel = panel.element.querySelector('.filter-imlabel-builder select');
            labelSel.value = 'lbl-cat';
            panel.element.querySelector('.filter-imlabel-builder button').click();
            const s = panel.getState();
            {_TEARDOWN}
            return s.imageLabelFilters;
        }}""")
        assert len(result) == 1
        assert result[0]["labelUuid"] == "lbl-cat"

    def test_add_filter_increments_count(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const labelSel = panel.element.querySelector('.filter-imlabel-builder select');
            labelSel.value = 'lbl-cat';
            panel.element.querySelector('.filter-imlabel-builder button').click();
            {_TEARDOWN}
            return panel.getActiveFilterCount();
        }}""")
        assert result == 2  # tag + image-label filter

    def test_add_without_label_does_nothing(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.element.querySelector('.filter-imlabel-builder button').click();
            {_TEARDOWN}
            return panel.getState().imageLabelFilters.length;
        }}""")
        assert result == 0

    def test_remove_chip_removes_from_state(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const labelSel = panel.element.querySelector('.filter-imlabel-builder select');
            labelSel.value = 'lbl-cat';
            panel.element.querySelector('.filter-imlabel-builder button').click();
            panel.element.querySelector('.filter-imlabel-list button').click();
            {_TEARDOWN}
            return panel.getState().imageLabelFilters.length;
        }}""")
        assert result == 0


# ── setState ──────────────────────────────────────────────────────────────────

class TestFilterPanelSetState:
    def test_set_state_updates_tag_checkboxes(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ tagUuids: ['tag-reviewed'] }});
            const reviewedCb = panel.toolbarControls[0].querySelector('input[value="tag-reviewed"]');
            const unverifiedCb = panel.toolbarControls[0].querySelector('input[value="{UNVERIFIED_UUID}"]');
            {_TEARDOWN}
            return {{ reviewed: reviewedCb.checked, unverified: unverifiedCb.checked }};
        }}""")
        assert result["reviewed"] is True
        assert result["unverified"] is False

    def test_set_state_updates_label_checkboxes(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ labelUuids: ['lbl-dog'] }});
            const dogCb = panel.toolbarControls[1].querySelector('input[value="lbl-dog"]');
            const catCb = panel.toolbarControls[1].querySelector('input[value="lbl-cat"]');
            {_TEARDOWN}
            return {{ dog: dogCb.checked, cat: catCb.checked }};
        }}""")
        assert result["dog"] is True
        assert result["cat"] is False

    def test_set_state_fires_on_filter_change(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ labelUuids: ['lbl-cat'] }});
            {_TEARDOWN}
            return lastState?.labelUuids;
        }}""")
        assert result == ["lbl-cat"]

    def test_set_state_updates_tag_match_mode(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ tagMatchMode: 'all' }});
            const s = panel.getState();
            {_TEARDOWN}
            return s.tagMatchMode;
        }}""")
        assert result == "all"
