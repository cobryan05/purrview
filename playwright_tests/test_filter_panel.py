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

import json

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
        assert result == ["Any", "Verified", "Unverified"]

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

    def test_selecting_unverified_makes_not_default(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const select = panel.element.querySelector('.bbox-verification-select');
            select.value = 'unverified';
            select.dispatchEvent(new Event('change'));
            const d = panel.isDefault();
            {_TEARDOWN}
            return d;
        }}""")
        assert result is False

    def test_set_state_syncs_control(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ bboxVerification: 'unverified' }});
            const select = panel.element.querySelector('.bbox-verification-select');
            {_TEARDOWN}
            return select.value;
        }}""")
        assert result == "unverified"

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
            select.value = 'unverified';
            select.dispatchEvent(new Event('change'));
            const captured = calls.slice();
            {self._STANDALONE_TEARDOWN}
            return captured;
        }}""")
        assert result == ["unverified"]


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
        assert {"kind": "bboxVerification", "state": "verified"} in body["terms"]

    def test_bbox_verification_merges_with_tag_exclude_uuids(self, page: Page, base_url: str):
        """
        Confirm both the exclude entry (from tagExcludeUuids) and the
        verification entry (from bboxVerification) are present together as
        independent terms — one doesn't overwrite the other.
        """
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
            await fetchReviewQueue({
                tagUuids: [],
                showAllBboxes: true,
                tagExcludeUuids: ['some-other-tag'],
                bboxVerification: 'unverified',
            });
        }""")

        body = captured["body"]
        assert {"kind": "bboxTag", "tagUuids": ["some-other-tag"], "negate": True} in body["terms"]
        assert {"kind": "bboxVerification", "state": "unverified"} in body["terms"]


# ── "Config Ready" (V2) — Verification select renamed + repointed ──────────
#
# V2 renames the whole-image Verification select to "Config Ready" in place
# (same DOM node / class names, same underlying state field `verificationCheck`
# and values '' | 'all_verified' | 'has_unverified' — only display text
# changes) and repoints its target from the Labels-dropdown selection to the
# ACTIVE TRAINING CONFIG's target labels via `verificationLabelUuids`. It is
# disabled with a hint when there's no active config, and disabled in bboxes
# view (whole-image-only check).
#
# activeTrainingConfig is a session-scoped ES-module singleton — setting it via
# activeTrainingConfig.setActive(...) in one evaluate() call affects every
# subsequent evaluate() on the same page (shared module cache), so these tests
# use the plain `page` fixture (not `shared_page`) and reset activeTrainingConfig
# to null in teardown so state doesn't leak across tests.

MODEL_CFG_NODE = {
    "uuid": "node-cat",
    "labelUuid": "lbl-cat",
    "modelType": "object_detection",
    "architecture": "yolo_v9_c",
}
MODEL_CFG = {"uuid": "mc-1", "name": "My Model", "description": "", "nodes": [MODEL_CFG_NODE]}
ACTIVE_TC = {
    "uuid": "tc-1",
    "name": "Cat Detector",
    "description": "",
    "datasetUuids": [],
    "modelConfigNodeUuid": "node-cat",
    "hyperparamsJson": {},
    "createdAt": "2026-01-01T00:00:00",
    "updatedAt": "2026-01-01T00:00:00",
}


def _mock_model_configs_and_stats(page: Page):
    """Stub the network calls activeTrainingConfig.setActive() triggers:
    fetchModelConfigs() (to derive label UUIDs from the TC's model config node)
    and fetchTrainingConfigLabelStats() (fire-and-forget stats refresh)."""
    page.route(
        lambda url: url.split("?")[0].endswith("/api/model-configs"),
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "configs": [MODEL_CFG]}),
            )
            if route.request.method == "GET"
            else route.fallback()
        ),
    )
    page.route(
        "**/api/training-configs/tc-1/label-stats",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labelStats": [], "totalImages": 0}),
        ),
    )


# JS snippets: activate/deactivate ACTIVE_TC via the real activeTrainingConfig
# singleton, awaiting the publish so getActive()/getLabelUuids() are settled
# before returning. Each is wrapped in its own block scope `{ ... }` because
# some tests use BOTH snippets within the same evaluate() call, where two
# top-level `const activeTrainingConfig` decls would be a SyntaxError.
_ACTIVATE_TC = f"""
    {{
        const {{ activeTrainingConfig }} = await import('/app-static/js/shared/activeTrainingConfig.js');
        await activeTrainingConfig.setActive({json.dumps(ACTIVE_TC)});
    }}
"""

_DEACTIVATE_TC = """
    {
        const { activeTrainingConfig } = await import('/app-static/js/shared/activeTrainingConfig.js');
        await activeTrainingConfig.setActive(null);
    }
"""


@pytest.fixture()
def _reset_active_tc(page: Page):
    """Ensure activeTrainingConfig is cleared after each test in this module,
    since it's a page-global singleton that would otherwise leak into the next test."""
    yield
    page.evaluate(f"async () => {{ {_DEACTIVATE_TC} }}")


class TestConfigReadySection:
    def test_section_label_is_config_ready(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const sectionLabels = [...panel.element.querySelectorAll('.filter-section-label')].map(l => l.textContent);
            {_TEARDOWN}
            return sectionLabels;
        }}""")
        assert "Config Ready" in result
        assert "Verification" not in result

    def test_options_are_any_ready_needs_review(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const select = panel.element.querySelector('.filter-verification-select:not(.bbox-verification-select)');
            const texts = [...select.querySelectorAll('option')].map(o => o.textContent);
            {_TEARDOWN}
            return texts;
        }}""")
        assert result == ["Any", "Ready", "Needs review"]

    def test_option_values_unchanged(self, shared_page: Page):
        """Display text changes but underlying values must stay the server contract."""
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const select = panel.element.querySelector('.filter-verification-select:not(.bbox-verification-select)');
            const values = [...select.querySelectorAll('option')].map(o => o.value);
            {_TEARDOWN}
            return values;
        }}""")
        assert result == ["", "all_verified", "has_unverified"]

    def test_state_field_name_still_verification_check(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const select = panel.element.querySelector('.filter-verification-select:not(.bbox-verification-select)');
            select.value = 'all_verified';
            select.dispatchEvent(new Event('change'));
            const s = panel.getState();
            {_TEARDOWN}
            return s.verificationCheck;
        }}""")
        assert result == "all_verified"

    def test_selecting_ready_still_counts_as_active_filter(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const select = panel.element.querySelector('.filter-verification-select:not(.bbox-verification-select)');
            select.value = 'has_unverified';
            select.dispatchEvent(new Event('change'));
            const count = panel.getActiveFilterCount();
            const isDefault = panel.isDefault();
            {_TEARDOWN}
            return {{ count, isDefault }};
        }}""")
        assert result["count"] == 1
        assert result["isDefault"] is False


class TestConfigReadyDisabledWithoutActiveConfig:
    def test_disabled_by_default_no_active_config(self, page: Page, base_url: str, _reset_active_tc):
        page.goto(base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            const select = panel.element.querySelector('.filter-verification-select:not(.bbox-verification-select)');
            const disabled = select.disabled;
            const title = select.title;
            {_TEARDOWN}
            return {{ disabled, title }};
        }}""")
        assert result["disabled"] is True
        assert "active training config" in result["title"].lower()

    def test_enabled_once_config_becomes_active(self, page: Page, base_url: str, _reset_active_tc):
        _mock_model_configs_and_stats(page)
        page.goto(base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            {_ACTIVATE_TC}
            const select = panel.element.querySelector('.filter-verification-select:not(.bbox-verification-select)');
            const disabled = select.disabled;
            {_TEARDOWN}
            return disabled;
        }}""")
        assert result is False

    def test_disabled_again_after_config_cleared(self, page: Page, base_url: str, _reset_active_tc):
        _mock_model_configs_and_stats(page)
        page.goto(base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            {_ACTIVATE_TC}
            const enabledAfterActivate = !panel.element.querySelector('.filter-verification-select:not(.bbox-verification-select)').disabled;
            {_DEACTIVATE_TC}
            const disabledAfterClear = panel.element.querySelector('.filter-verification-select:not(.bbox-verification-select)').disabled;
            {_TEARDOWN}
            return {{ enabledAfterActivate, disabledAfterClear }};
        }}""")
        assert result["enabledAfterActivate"] is True
        assert result["disabledAfterClear"] is True

    def test_stats_only_refresh_does_not_reset_enabled_state(self, page: Page, base_url: str, _reset_active_tc):
        """ACTIVE_TRAINING_CONFIG_CHANGED also fires for stats-only refreshes
        (statsUpdated: true) — the enabled/disabled state must still reflect
        reality after one of those (NOTES.md: naive subscribers wipe UI state)."""
        _mock_model_configs_and_stats(page)
        page.goto(base_url)
        result = page.evaluate(f"""async () => {{
            {_SETUP}
            {_ACTIVATE_TC}
            const {{ events, EventTypes }} = await import('/app-static/js/shared/events/index.js');
            events.publish(EventTypes.ACTIVE_TRAINING_CONFIG_CHANGED, {{ statsUpdated: true }});
            const disabled = panel.element.querySelector('.filter-verification-select:not(.bbox-verification-select)').disabled;
            {_TEARDOWN}
            return disabled;
        }}""")
        assert result is False


# ── verificationLabelUuids sourcing + threading (V2) ────────────────────────
#
# main.js sources verificationLabelUuids from activeTrainingConfig.getLabelUuids()
# and threads it through fetchReviewQueue/bulkApplyAll only when verificationCheck
# is set and _viewFilter !== 'bboxes'. Test at the fetchReviewQueue/bulkApplyAll
# API-boundary level (matches the existing TestFetchReviewQueueBboxVerificationMapping
# pattern) since that's the shared contract main.js and runModelOnResults.js both use.


class TestVerificationLabelUuidsMapping:
    def test_omitted_when_verification_check_unset(self, page: Page, base_url: str):
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
            await fetchReviewQueue({ tagUuids: [], showAllBboxes: true, verificationLabelUuids: ['lbl-cat'] });
        }""")

        # Without verificationCheck the caller should not have supplied the
        # override either. Without verificationCheck there's nothing for
        # verificationLabelUuids to parametrize — buildReviewFilterBody only
        # emits a labelVerification term when verificationCheck is set.
        body = captured["body"]
        assert not any(t.get("kind") == "labelVerification" for t in body.get("terms", []))

    def test_fetch_review_queue_sends_verification_label_uuids(self, page: Page, base_url: str):
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
            await fetchReviewQueue({
                tagUuids: [],
                showAllBboxes: true,
                verificationCheck: 'all_verified',
                verificationLabelUuids: ['lbl-cat', 'lbl-dog'],
            });
        }""")

        body = captured["body"]
        assert {
            "kind": "labelVerification",
            "check": "all_verified",
            "labelUuids": ["lbl-cat", "lbl-dog"],
        } in body["terms"]

    def test_fetch_review_queue_omits_empty_verification_label_uuids(self, page: Page, base_url: str):
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
            await fetchReviewQueue({ tagUuids: [], showAllBboxes: true, verificationCheck: 'all_verified' });
        }""")

        body = captured["body"]
        assert "verificationLabelUuids" not in body

    def test_bulk_apply_all_sends_verification_label_uuids(self, page: Page, base_url: str):
        captured = {}

        def handle(route):
            captured["body"] = route.request.post_data_json
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"status": "success", "modifiedCount": 0, "failCount": 0}',
            )

        page.goto(base_url)
        page.route("**/api/images/bbox/bulk-apply", handle)

        page.evaluate("""async () => {
            const { bulkApplyAll } = await import('/app-static/js/shared/api/review.js');
            await bulkApplyAll({
                tagUuids: [],
                showAllBboxes: true,
                verificationCheck: 'has_unverified',
                verificationLabelUuids: ['lbl-cat'],
                action: 'markVerified',
            });
        }""")

        body = captured["body"]
        assert {
            "kind": "labelVerification",
            "check": "has_unverified",
            "labelUuids": ["lbl-cat"],
        } in body[
            "filter"
        ]["terms"]

    def test_bulk_apply_all_omits_empty_verification_label_uuids(self, page: Page, base_url: str):
        captured = {}

        def handle(route):
            captured["body"] = route.request.post_data_json
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"status": "success", "modifiedCount": 0, "failCount": 0}',
            )

        page.goto(base_url)
        page.route("**/api/images/bbox/bulk-apply", handle)

        page.evaluate("""async () => {
            const { bulkApplyAll } = await import('/app-static/js/shared/api/review.js');
            await bulkApplyAll({
                tagUuids: [],
                showAllBboxes: true,
                verificationCheck: 'has_unverified',
                action: 'markVerified',
            });
        }""")

        body = captured["body"]
        assert "verificationLabelUuids" not in body["filter"]


# ── Bulk-apply must scope to the SAME filter the grid shows ────────────────
#
# Follow-up PR A (UI_REDESIGN §32.5): bulkApplyAll and bulkApplyImages
# destructure their params explicitly, and both dropped filter dimensions the
# visible grid IS filtered by — bulkApplyAll dropped bboxVerification *and*
# imageLabelConditions, bulkApplyImages dropped bboxVerification. The result
# was a silent over-apply: "filter to Unverified only, click Apply to All" hit
# verified bboxes too. These pin every dimension _baseReviewFilterOpts sends.


class TestBulkApplyFilterParity:
    def _route(self, page: Page, url: str, captured: dict):
        def handle(route):
            captured["body"] = route.request.post_data_json
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"status": "success", "modifiedCount": 0, "failCount": 0, "skippedCount": 0}',
            )

        page.route(url, handle)

    def test_bulk_apply_all_sends_bbox_verification(self, page: Page, base_url: str):
        captured = {}
        page.goto(base_url)
        self._route(page, "**/api/images/bbox/bulk-apply", captured)

        page.evaluate("""async () => {
            const { bulkApplyAll } = await import('/app-static/js/shared/api/review.js');
            await bulkApplyAll({
                tagUuids: [],
                showAllBboxes: true,
                bboxVerification: 'unverified',
                action: 'markVerified',
            });
        }""")

        terms = captured["body"]["filter"]["terms"]
        assert {"kind": "bboxVerification", "state": "unverified"} in terms

    def test_bulk_apply_all_ignores_bbox_verification_any(self, page: Page, base_url: str):
        captured = {}
        page.goto(base_url)
        self._route(page, "**/api/images/bbox/bulk-apply", captured)

        page.evaluate("""async () => {
            const { bulkApplyAll } = await import('/app-static/js/shared/api/review.js');
            await bulkApplyAll({
                tagUuids: [],
                showAllBboxes: true,
                bboxVerification: 'any',
                action: 'markVerified',
            });
        }""")

        terms = captured["body"]["filter"].get("terms", [])
        assert not any(t.get("kind") == "bboxVerification" for t in terms)

    def test_bulk_apply_all_sends_image_label_conditions(self, page: Page, base_url: str):
        captured = {}
        page.goto(base_url)
        self._route(page, "**/api/images/bbox/bulk-apply", captured)

        page.evaluate("""async () => {
            const { bulkApplyAll } = await import('/app-static/js/shared/api/review.js');
            await bulkApplyAll({
                tagUuids: [],
                showAllBboxes: true,
                imageLabelConditions: [{ labelUuid: 'lbl-cat', state: 'present', rollUp: false, negate: true }],
                action: 'markVerified',
            });
        }""")

        terms = captured["body"]["filter"]["terms"]
        assert {
            "kind": "imageLabel",
            "labelUuid": "lbl-cat",
            "state": "present",
            "rollUp": False,
            "negate": True,
        } in terms

    def test_bulk_apply_images_sends_bbox_verification(self, page: Page, base_url: str):
        captured = {}
        page.goto(base_url)
        self._route(page, "**/api/images/bulk-apply", captured)

        page.evaluate("""async () => {
            const { bulkApplyImages } = await import('/app-static/js/shared/api/review.js');
            await bulkApplyImages({
                tagUuids: [],
                showAllBboxes: true,
                bboxVerification: 'verified',
                action: 'setAbsent',
                labelUuid: 'lbl-cat',
            });
        }""")

        terms = captured["body"]["filter"]["terms"]
        assert {"kind": "bboxVerification", "state": "verified"} in terms


# ── fetchReviewQueue: imageLabelFilters → imageLabel term mapping ──────────────
#
# `imageLabelFilters` is the direct-format option on buildReviewFilterBody (the
# pre-1.4b flat ImageLabelFilter shape), kept alongside the panel's own
# `imageLabelConditions`. Its term conversion was not semantics-preserving:
# 'uncertain' was passed through as an ImageLabelState (which the server rejects
# — the term model spells it 'unreviewed'), and rollUp was left off entirely, so
# ImageLabelTerm.roll_up's server-side default of true silently widened what had
# been an exact-label predicate. No in-repo caller passes the option today, so
# these guard a documented public option that nothing else would catch.


class TestFetchReviewQueueImageLabelFilterMapping:
    @staticmethod
    def _capture(page: Page, base_url: str, image_label_filters: list) -> dict:
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
        page.evaluate(
            """async (filters) => {
                const { fetchReviewQueue } = await import('/app-static/js/shared/api/review.js');
                await fetchReviewQueue({ tagUuids: [], showAllBboxes: true, imageLabelFilters: filters });
            }""",
            image_label_filters,
        )
        terms = [t for t in captured["body"].get("terms", []) if t.get("kind") == "imageLabel"]
        assert len(terms) == 1, captured["body"]
        return terms[0]

    def test_uncertain_maps_to_unreviewed(self, page: Page, base_url: str):
        """'uncertain' is the ImageItemStatus spelling; the term model's
        equivalent is 'unreviewed'. Passing it through raw 422s server-side."""
        term = self._capture(page, base_url, [{"labelUuid": "lbl-1", "status": "uncertain"}])
        assert term["state"] == "unreviewed"

    @pytest.mark.parametrize("status", ["present", "absent"])
    def test_other_states_pass_through(self, page: Page, base_url: str, status: str):
        term = self._capture(page, base_url, [{"labelUuid": "lbl-1", "status": status}])
        assert term["state"] == status

    def test_roll_up_defaults_to_false(self, page: Page, base_url: str):
        """Matches the deliberate rollUp: false on labelExcludeUuids — the flat
        field this option preserves was never child-expanded, and the server
        default is true, so it has to be explicit."""
        term = self._capture(page, base_url, [{"labelUuid": "lbl-1", "status": "present"}])
        assert term["rollUp"] is False

    def test_explicit_roll_up_is_honoured(self, page: Page, base_url: str):
        term = self._capture(page, base_url, [{"labelUuid": "lbl-1", "status": "present", "rollUp": True}])
        assert term["rollUp"] is True

    def test_exclude_maps_to_negate(self, page: Page, base_url: str):
        term = self._capture(page, base_url, [{"labelUuid": "lbl-1", "status": "present", "exclude": True}])
        assert term["negate"] is True


# ── Filename / date negation ────────────────────────────────────────────────
#
# "does not contain" / "not in this range" — a negate checkbox next to each
# control, mirroring the pattern already used for image-label/image-tag
# conditions and tag/label excludes (see imageTagConditions.js's allowNegate).
# Backend support (negate on FilenameTerm/DateTerm) already existed and was
# tested (tests/db_server/test_filter_terms.py); this covers the panel UI and
# the wire mapping in buildReviewFilterBody / filterUtils.js that were missing.


class TestFilterPanelNegateCheckboxes:
    def test_filename_negate_defaults_false(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const s = panel.getState();
            {_TEARDOWN}
            return s.filenameNegate;
        }}""")
        assert result is False

    def test_date_negate_defaults_false(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            const s = panel.getState();
            {_TEARDOWN}
            return s.dateNegate;
        }}""")
        assert result is False

    def test_checking_filename_negate_updates_state(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            document.querySelector('.filter-filename-negate input').click();
            const s = window.__fpLastState();
            {_TEARDOWN}
            return s.filenameNegate;
        }}""")
        assert result is True

    def test_checking_date_negate_updates_state(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            document.querySelector('.filter-date-negate input').click();
            const s = window.__fpLastState();
            {_TEARDOWN}
            return s.dateNegate;
        }}""")
        assert result is True

    def test_set_state_syncs_filename_negate_checkbox(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ filenameSearch: 'foo', filenameNegate: true }});
            const checked = document.querySelector('.filter-filename-negate input').checked;
            {_TEARDOWN}
            return checked;
        }}""")
        assert result is True

    def test_set_state_syncs_date_negate_checkbox(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ dateFrom: '2026-01-01T00:00:00.000Z', dateNegate: true }});
            const checked = document.querySelector('.filter-date-negate input').checked;
            {_TEARDOWN}
            return checked;
        }}""")
        assert result is True

    def test_filename_negate_makes_panel_non_default(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ filenameNegate: true }});
            const d = panel.isDefault();
            {_TEARDOWN}
            return d;
        }}""")
        assert result is False

    def test_date_negate_makes_panel_non_default(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ dateNegate: true }});
            const d = panel.isDefault();
            {_TEARDOWN}
            return d;
        }}""")
        assert result is False

    def test_filename_chip_shows_negate_marker(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ filenameSearch: 'foo', filenameNegate: true }});
            const chips = panel.getChipDescriptors();
            {_TEARDOWN}
            return chips.find(c => c.key === 'filename').label;
        }}""")
        assert result == "¬File: foo"

    def test_date_chip_shows_negate_marker(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ dateFrom: '2026-01-01T00:00:00.000Z', dateNegate: true }});
            const chips = panel.getChipDescriptors();
            {_TEARDOWN}
            return chips.find(c => c.key === 'date').label.startsWith('¬Date:');
        }}""")
        assert result is True

    def test_removing_filename_chip_clears_negate(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            panel.setState({{ filenameSearch: 'foo', filenameNegate: true }});
            const chip = panel.getChipDescriptors().find(c => c.key === 'filename');
            chip.onRemove();
            const s = panel.getState();
            {_TEARDOWN}
            return {{ filenameSearch: s.filenameSearch, filenameNegate: s.filenameNegate }};
        }}""")
        assert result == {"filenameSearch": "", "filenameNegate": False}


class TestFetchReviewQueueFilenameDateNegateMapping:
    @staticmethod
    def _capture(page: Page, base_url: str, opts: dict) -> dict:
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
        page.evaluate(
            """async (opts) => {
                const { fetchReviewQueue } = await import('/app-static/js/shared/api/review.js');
                await fetchReviewQueue({ tagUuids: [], showAllBboxes: true, ...opts });
            }""",
            opts,
        )
        return captured["body"]

    def test_filename_negate_true_is_sent(self, page: Page, base_url: str):
        body = self._capture(page, base_url, {"filenameSearch": "foo", "filenameNegate": True})
        terms = [t for t in body["terms"] if t["kind"] == "filename"]
        assert terms == [{"kind": "filename", "patterns": ["foo"], "negate": True}]

    def test_filename_negate_defaults_false(self, page: Page, base_url: str):
        body = self._capture(page, base_url, {"filenameSearch": "foo"})
        terms = [t for t in body["terms"] if t["kind"] == "filename"]
        assert terms == [{"kind": "filename", "patterns": ["foo"], "negate": False}]

    def test_date_negate_true_is_sent(self, page: Page, base_url: str):
        body = self._capture(page, base_url, {"dateFrom": "2026-01-01T00:00:00.000Z", "dateNegate": True})
        terms = [t for t in body["terms"] if t["kind"] == "date"]
        assert terms == [{"kind": "date", "field": "created", "from": "2026-01-01T00:00:00.000Z", "negate": True}]

    def test_date_negate_omitted_when_false(self, page: Page, base_url: str):
        """dateNegate defaults false — matches the omit-when-default pattern used
        elsewhere in buildReviewFilterBody rather than always sending the key."""
        body = self._capture(page, base_url, {"dateFrom": "2026-01-01T00:00:00.000Z", "dateNegate": False})
        terms = [t for t in body["terms"] if t["kind"] == "date"]
        assert terms == [{"kind": "date", "field": "created", "from": "2026-01-01T00:00:00.000Z"}]
