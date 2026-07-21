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
    def test_default_tag_uuids_empty(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const s = panel.getState();
            {_TEARDOWN}
            return s.tagUuids;
        }}""")
        assert result == []

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

    def test_active_filter_count_is_zero_initially(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const c = panel.getActiveFilterCount();
            {_TEARDOWN}
            return c;
        }}""")
        assert result == 0

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
    def test_unverified_tag_absent_from_tags_dropdown(self, shared_page: Page):
        """
        V1: the bbox-verification tri-state control (filter-section
        '.bbox-verification-control') now owns verification filtering — the
        ad-hoc UNVERIFIED entry in the Tags dropdown is removed so there's
        only one way to filter by verification status.
        """
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const cb = panel.toolbarControls[0].querySelector('input[value="{UNVERIFIED_UUID}"]');
            {_TEARDOWN}
            return cb === null;
        }}""")
        assert result is True

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

    def test_checking_tag_adds_to_state(self, shared_page: Page):
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
        assert UNVERIFIED_UUID not in result

    def test_checking_tag_makes_not_default(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const cb = panel.toolbarControls[0].querySelector('input[value="tag-reviewed"]');
            cb.checked = true;
            cb.dispatchEvent(new Event('change'));
            {_TEARDOWN}
            return panel.isDefault();
        }}""")
        assert result is False

    def test_trigger_shows_all_by_default(self, shared_page: Page):
        """Trigger button text reflects no selection (show all) by default."""
        page = shared_page
        page.evaluate(f"async () => {{ {_SETUP} }}")

        trigger = page.locator(".ms-dropdown[data-filter-key='tags'] .ms-dropdown__trigger")
        text = trigger.text_content()
        assert "Unverified" not in text

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
        assert result == 1  # one label

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
        assert result == 0  # back to no filters

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


# ── setState ──────────────────────────────────────────────────────────────────

class TestFilterPanelSetState:
    def test_set_state_updates_tag_checkboxes(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ tagUuids: ['tag-reviewed'] }});
            const reviewedCb = panel.toolbarControls[0].querySelector('input[value="tag-reviewed"]');
            {_TEARDOWN}
            return {{ reviewed: reviewedCb.checked }};
        }}""")
        assert result["reviewed"] is True

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


# ── Bbox verification tri-state control (V1) ────────────────────────────────
#
# The control (app/review/bboxVerificationControl.js) is not one of
# toolbarControls[0/1] — it's rendered inside panel.element alongside the
# existing Verification select. Locate it via its own class name, not by
# position, so it can't collide with the toolbarControls positional contract.

class TestFilterPanelBboxVerification:
    def test_default_is_any(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const s = panel.getState();
            {_TEARDOWN}
            return s.bboxVerification;
        }}""")
        assert result == "any"

    def test_default_view_is_unchanged_show_all_bboxes(self, shared_page: Page):
        """bboxVerification: 'any' must not affect showAllBboxes derivation."""
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const s = panel.getState();
            {_TEARDOWN}
            return {{ tagUuids: s.tagUuids, showAllBboxes: s.showAllBboxes, bboxVerification: s.bboxVerification }};
        }}""")
        assert result["tagUuids"] == []
        assert result["showAllBboxes"] is True
        assert result["bboxVerification"] == "any"

    def test_control_renders_with_four_options(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const select = panel.element.querySelector('.bbox-verification-select');
            const texts = [...select.querySelectorAll('option')].map(o => o.textContent);
            {_TEARDOWN}
            return texts;
        }}""")
        assert result == ["Any", "Verified", "Unverified", "False-Pos"]

    def test_control_label_is_distinct_from_config_ready(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const label = panel.element.querySelector('.bbox-verification-control .filter-section-label');
            const text = label.textContent;
            {_TEARDOWN}
            return text;
        }}""")
        assert "Config Ready" not in result
        assert result.strip() != ""

    def test_selecting_verified_updates_state_and_fires_change(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const select = panel.element.querySelector('.bbox-verification-select');
            select.value = 'verified';
            select.dispatchEvent(new Event('change'));
            const s = panel.getState();
            const emitted = lastState?.bboxVerification;
            {_TEARDOWN}
            return {{ stateValue: s.bboxVerification, emitted }};
        }}""")
        assert result["stateValue"] == "verified"
        assert result["emitted"] == "verified"

    def test_selecting_unverified_increments_active_filter_count(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const before = panel.getActiveFilterCount();
            const select = panel.element.querySelector('.bbox-verification-select');
            select.value = 'unverified';
            select.dispatchEvent(new Event('change'));
            const after = panel.getActiveFilterCount();
            {_TEARDOWN}
            return {{ before, after }};
        }}""")
        assert result["before"] == 0
        assert result["after"] == 1

    def test_selecting_false_pos_makes_not_default(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const select = panel.element.querySelector('.bbox-verification-select');
            select.value = 'false_pos';
            select.dispatchEvent(new Event('change'));
            const d = panel.isDefault();
            {_TEARDOWN}
            return d;
        }}""")
        assert result is False

    def test_set_state_syncs_control(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ bboxVerification: 'false_pos' }});
            const select = panel.element.querySelector('.bbox-verification-select');
            {_TEARDOWN}
            return select.value;
        }}""")
        assert result == "false_pos"

    def test_verification_tags_absent_from_tags_dropdown(self, shared_page: Page):
        """
        V1 decision: the tri-state control REPLACES the ad-hoc UNVERIFIED entry
        that used to live in the Tags dropdown — verification tags must not
        appear there anymore (single way to filter by verification status).
        """
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const cb = panel.toolbarControls[0].querySelector('input[value="{UNVERIFIED_UUID}"]');
            {_TEARDOWN}
            return cb === null;
        }}""")
        assert result is True


# ── bboxVerificationControl.js — standalone reusable control ───────────────
#
# Exercised directly (no filter panel) since it's designed to be dependency-
# light for reuse in modals (unit M1).

class TestBboxVerificationControlStandalone:
    _STANDALONE_SETUP = """
        const { createBboxVerificationControl } = await import('/app-static/js/app/review/bboxVerificationControl.js');
        let calls = [];
        const ctl = createBboxVerificationControl({ onChange: (v) => calls.push(v) });
        document.body.appendChild(ctl.element);
    """
    _STANDALONE_TEARDOWN = """
        ctl.element.remove();
        ctl.destroy();
    """

    def test_get_value_defaults_to_any(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {self._STANDALONE_SETUP}
            const v = ctl.getValue();
            {self._STANDALONE_TEARDOWN}
            return v;
        }}""")
        assert result == "any"

    def test_set_value_updates_get_value(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {self._STANDALONE_SETUP}
            ctl.setValue('verified');
            const v = ctl.getValue();
            {self._STANDALONE_TEARDOWN}
            return v;
        }}""")
        assert result == "verified"

    def test_set_value_does_not_emit_by_default(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {self._STANDALONE_SETUP}
            ctl.setValue('verified');
            const n = calls.length;
            {self._STANDALONE_TEARDOWN}
            return n;
        }}""")
        assert result == 0

    def test_changing_select_calls_on_change(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {self._STANDALONE_SETUP}
            const select = ctl.element.querySelector('.bbox-verification-select');
            select.value = 'false_pos';
            select.dispatchEvent(new Event('change'));
            const captured = calls.slice();
            {self._STANDALONE_TEARDOWN}
            return captured;
        }}""")
        assert result == ["false_pos"]


# ── fetchReviewQueue request mapping (bboxVerification -> tagPresenceFilters) ──
#
# Route-intercept the review-queue endpoint to capture the outgoing request
# body, following the test_run_model_on_results.py pattern of stubbing the
# network boundary rather than exercising the real backend.

class TestFetchReviewQueueBboxVerificationMapping:
    def test_any_does_not_add_tag_presence_filter(self, page: Page, base_url: str):
        captured = {}

        def handle(route):
            captured["body"] = route.request.post_data_json
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"status": "success", "images": [], "hasMore": false, "totalCount": 0}',
            )

        page.goto(base_url)
        page.route("**/api/images/review", handle)

        page.evaluate("""async () => {
            const { fetchReviewQueue } = await import('/app-static/js/shared/api/review.js');
            await fetchReviewQueue({ tagUuids: [], showAllBboxes: true, bboxVerification: 'any' });
        }""")

        body = captured["body"]
        assert "tagPresenceFilters" not in body

    def test_verified_appends_tag_presence_filter(self, page: Page, base_url: str):
        captured = {}
        VERIFIED_UUID = "00000000-0000-0000-0000-000000000005"

        def handle(route):
            captured["body"] = route.request.post_data_json
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"status": "success", "images": [], "hasMore": false, "totalCount": 0}',
            )

        page.goto(base_url)
        page.route("**/api/images/review", handle)

        page.evaluate("""async () => {
            const { fetchReviewQueue } = await import('/app-static/js/shared/api/review.js');
            await fetchReviewQueue({ tagUuids: [], showAllBboxes: true, bboxVerification: 'verified' });
        }""")

        body = captured["body"]
        assert body["tagPresenceFilters"] == [{"tagUuid": VERIFIED_UUID, "exclude": False}]

    def test_bbox_verification_merges_with_tag_exclude_uuids(self, page: Page, base_url: str):
        """
        The builder used to ASSIGN body.tagPresenceFilters = tagExcludeUuids.map(...)
        — a naive fix could overwrite that instead of merging. Confirm both the
        exclude entry (from tagExcludeUuids) and the verification entry (from
        bboxVerification) are present together.
        """
        captured = {}
        FALSE_POS_UUID = "00000000-0000-0000-0000-000000000003"

        def handle(route):
            captured["body"] = route.request.post_data_json
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"status": "success", "images": [], "hasMore": false, "totalCount": 0}',
            )

        page.goto(base_url)
        page.route("**/api/images/review", handle)

        page.evaluate("""async () => {
            const { fetchReviewQueue } = await import('/app-static/js/shared/api/review.js');
            await fetchReviewQueue({
                tagUuids: [],
                showAllBboxes: true,
                tagExcludeUuids: ['some-other-tag'],
                bboxVerification: 'false_pos',
            });
        }""")

        body = captured["body"]
        assert {"tagUuid": "some-other-tag", "exclude": True} in body["tagPresenceFilters"]
        assert {"tagUuid": FALSE_POS_UUID, "exclude": False} in body["tagPresenceFilters"]
        assert len(body["tagPresenceFilters"]) == 2
