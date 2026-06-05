"""
Tests for app/review/filters.js — createFilterPanel

The filter panel is a pure DOM component: it accepts label/tag Maps, builds a
UI, and exposes getState/setState/isDefault/getActiveFilterCount.  No network
calls are made by the panel itself; all API dependency is injected via the Maps.
"""
import pytest
from playwright.sync_api import Page

UNVERIFIED_UUID = "00000000-0000-0000-0000-000000000001"


def goto(page: Page, base_url: str):
    page.goto(base_url)


# Shared JS preamble used in many tests — builds a filter panel with one tag.
_SETUP = f"""
    const {{ createFilterPanel }} = await import('/app-static/js/app/review/filters.js');

    const labels = new Map([
        ['lbl-cat', {{ uuid: 'lbl-cat', name: 'Cat', parent_uuid: null }}],
        ['lbl-dog', {{ uuid: 'lbl-dog', name: 'Dog', parent_uuid: null }}],
    ]);
    const tags = new Map([
        ['{UNVERIFIED_UUID}', {{ uuid: '{UNVERIFIED_UUID}', name: 'Unverified', color: null }}],
        ['tag-reviewed',     {{ uuid: 'tag-reviewed',     name: 'Reviewed',   color: '#4c4' }}],
    ]);

    let lastState = null;
    const panel = createFilterPanel(labels, tags, (s) => {{ lastState = s; }});
    document.body.appendChild(panel.element);
"""

_TEARDOWN = "panel.element.remove();"


class TestFilterPanelDefaults:
    def test_default_is_unverified_tag_checked(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            const state = panel.getState();
            {_TEARDOWN}
            return state.tagUuids;
        }}""")
        assert UNVERIFIED_UUID in result

    def test_is_default_true_initially(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            const d = panel.isDefault();
            {_TEARDOWN}
            return d;
        }}""")
        assert result is True

    def test_active_filter_count_is_one_initially(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            const c = panel.getActiveFilterCount();
            {_TEARDOWN}
            return c;
        }}""")
        assert result == 1

    def test_label_uuid_empty_initially(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            const state = panel.getState();
            {_TEARDOWN}
            return state.labelUuid;
        }}""")
        assert result == ""

    def test_filename_search_empty_initially(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            const state = panel.getState();
            {_TEARDOWN}
            return state.filenameSearch;
        }}""")
        assert result == ""


class TestFilterPanelTagCheckbox:
    def test_unchecking_unverified_removes_from_state(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            // Uncheck the UNVERIFIED checkbox
            const cb = panel.element.querySelector('input[value="{UNVERIFIED_UUID}"]');
            cb.checked = false;
            cb.dispatchEvent(new Event('change'));
            const state = panel.getState();
            {_TEARDOWN}
            return {{ tagUuids: state.tagUuids, showAllBboxes: state.showAllBboxes }};
        }}""")
        assert UNVERIFIED_UUID not in result["tagUuids"]

    def test_no_tags_checked_sets_show_all(self, page: Page, base_url: str):
        """When no tag checkboxes are checked, showAllBboxes should be true."""
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            // Uncheck all checkboxes
            panel.element.querySelectorAll('input[type="checkbox"]').forEach(cb => {{
                cb.checked = false;
                cb.dispatchEvent(new Event('change'));
            }});
            const state = panel.getState();
            {_TEARDOWN}
            return state.showAllBboxes;
        }}""")
        assert result is True

    def test_checking_second_tag_adds_to_state(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            const cb = panel.element.querySelector('input[value="tag-reviewed"]');
            cb.checked = true;
            cb.dispatchEvent(new Event('change'));
            const state = panel.getState();
            {_TEARDOWN}
            return state.tagUuids;
        }}""")
        assert "tag-reviewed" in result
        assert UNVERIFIED_UUID in result  # was already checked

    def test_unchecking_makes_not_default(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            const cb = panel.element.querySelector('input[value="{UNVERIFIED_UUID}"]');
            cb.checked = false;
            cb.dispatchEvent(new Event('change'));
            const isDefault = panel.isDefault();
            {_TEARDOWN}
            return isDefault;
        }}""")
        assert result is False


class TestFilterPanelLabelSelect:
    def test_selecting_label_updates_state(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            const sel = panel.element.querySelector('select.filter-label-select');
            sel.value = 'lbl-cat';
            sel.dispatchEvent(new Event('change'));
            const state = panel.getState();
            {_TEARDOWN}
            return {{ labelUuid: state.labelUuid }};
        }}""")
        assert result["labelUuid"] == "lbl-cat"

    def test_selecting_label_increments_count(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            const sel = panel.element.querySelector('select.filter-label-select');
            sel.value = 'lbl-dog';
            sel.dispatchEvent(new Event('change'));
            {_TEARDOWN}
            return panel.getActiveFilterCount();
        }}""")
        assert result == 2  # UNVERIFIED tag + label

    def test_clearing_label_decrements_count(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            const sel = panel.element.querySelector('select.filter-label-select');
            sel.value = 'lbl-cat';
            sel.dispatchEvent(new Event('change'));
            sel.value = '';
            sel.dispatchEvent(new Event('change'));
            {_TEARDOWN}
            return panel.getActiveFilterCount();
        }}""")
        assert result == 1  # back to just the tag


class TestFilterPanelImageLabelFilters:
    def test_add_image_label_filter(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            // Select the label in the builder
            const labelSel = panel.element.querySelector('.filter-imlabel-builder select');
            labelSel.value = 'lbl-cat';
            // Click Add
            const addBtn = panel.element.querySelector('.filter-imlabel-builder button');
            addBtn.click();
            const state = panel.getState();
            {_TEARDOWN}
            return state.imageLabelFilters;
        }}""")
        assert len(result) == 1
        assert result[0]["labelUuid"] == "lbl-cat"

    def test_add_filter_increments_count(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            const labelSel = panel.element.querySelector('.filter-imlabel-builder select');
            labelSel.value = 'lbl-cat';
            const addBtn = panel.element.querySelector('.filter-imlabel-builder button');
            addBtn.click();
            {_TEARDOWN}
            return panel.getActiveFilterCount();
        }}""")
        assert result == 2  # tag + image-label filter

    def test_add_without_label_does_nothing(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            // Don't select a label — click Add anyway
            const addBtn = panel.element.querySelector('.filter-imlabel-builder button');
            addBtn.click();
            {_TEARDOWN}
            return panel.getState().imageLabelFilters.length;
        }}""")
        assert result == 0

    def test_remove_chip_removes_from_state(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            const labelSel = panel.element.querySelector('.filter-imlabel-builder select');
            labelSel.value = 'lbl-cat';
            panel.element.querySelector('.filter-imlabel-builder button').click();

            // Click the × remove button on the chip
            const removeBtn = panel.element.querySelector('.filter-imlabel-list button');
            removeBtn.click();

            {_TEARDOWN}
            return panel.getState().imageLabelFilters.length;
        }}""")
        assert result == 0


class TestFilterPanelSetState:
    def test_set_state_updates_tag_checkboxes(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ tagUuids: ['tag-reviewed'] }});
            // Verify the DOM checkbox reflects the new state
            const reviewedCb = panel.element.querySelector('input[value="tag-reviewed"]');
            const unverifiedCb = panel.element.querySelector('input[value="{UNVERIFIED_UUID}"]');
            {_TEARDOWN}
            return {{ reviewed: reviewedCb.checked, unverified: unverifiedCb.checked }};
        }}""")
        assert result["reviewed"] is True
        assert result["unverified"] is False

    def test_set_state_updates_label_select(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ labelUuid: 'lbl-dog' }});
            const sel = panel.element.querySelector('select.filter-label-select');
            {_TEARDOWN}
            return sel.value;
        }}""")
        assert result == "lbl-dog"

    def test_set_state_fires_on_filter_change(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ labelUuid: 'lbl-cat' }});
            {_TEARDOWN}
            return lastState?.labelUuid;
        }}""")
        assert result == "lbl-cat"
