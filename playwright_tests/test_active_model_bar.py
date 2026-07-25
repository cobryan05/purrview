"""
Tests for the training-run picker in the bottom-left bar —
apps/image_tagging_server/static/js/ui/activeModelBar.js

Previously the only way to change/clear the active training run was via
navigateTabs() to the Training Runs management page, which is useless while
a modal is open (the destination content renders behind the modal overlay).
These tests cover the replacement: a dropdown, anchored in the bar itself,
that lists available runs, lets you switch/clear the active one in place,
and still offers a link to the management page.

Run with:
    pytest playwright_tests/test_active_model_bar.py -p no:xdist -p no:asyncio -v
"""
import json
import re

import pytest
from playwright.sync_api import Page, expect

TC_1 = {
    "uuid": "tc-1", "name": "Cat Detector", "description": "", "datasetUuids": [],
    "modelConfigNodeUuid": None, "hyperparamsJson": {},
    "createdAt": "2026-01-01T00:00:00", "updatedAt": "2026-01-01T00:00:00",
}
TC_2 = {
    "uuid": "tc-2", "name": "Dog Detector", "description": "", "datasetUuids": [],
    "modelConfigNodeUuid": None, "hyperparamsJson": {},
    "createdAt": "2026-01-01T00:00:00", "updatedAt": "2026-01-01T00:00:00",
}


def _mock_training_configs(page: Page, configs: list):
    """Intercept GET /api/training-configs with a fixed list."""
    page.route(
        lambda url: url.split("?")[0].endswith("/api/training-configs"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": configs}),
        ) if route.request.method == "GET" else route.fallback(),
    )


def _mock_label_stats(page: Page, tc_uuid: str, label_stats: list, total_images: int = 20):
    page.route(
        f"**/api/training-configs/{tc_uuid}/label-stats",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labelStats": label_stats, "totalImages": total_images}),
        ),
    )


def _mock_readiness(page: Page, tc_uuid: str, ready=0, needs_review=0, not_started=0, roll_up_active=False):
    page.route(
        f"**/api/training-configs/{tc_uuid}/readiness",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "status": "success",
                "readyCount": ready,
                "needsReviewCount": needs_review,
                "notStartedCount": not_started,
                "datasetUuids": [],
                "labelUuids": [],
                "rollUpActive": roll_up_active,
            }),
        ),
    )


def _mock_labels(page: Page, labels: list):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": labels}),
        ) if route.request.method == "GET" else route.fallback(),
    )


def _set_active(page: Page, tc: dict | None):
    """Drive the activeTrainingConfig singleton directly, as done elsewhere
    (test_review_workflow_ui.py) to seed bar state without a full UI flow."""
    page.evaluate("""async (tc) => {
        const { activeTrainingConfig } = await import('/app-static/js/shared/activeTrainingConfig.js');
        await activeTrainingConfig.setActive(tc);
    }""", tc)


def _navigate(page: Page, base_url: str, configs: list = ()):
    _mock_training_configs(page, list(configs))
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#active-model-bar")


@pytest.fixture(autouse=True)
def _cleanup(page: Page):
    yield
    # Clear the session-scoped singleton so state doesn't leak between tests.
    page.evaluate("""async () => {
        const { activeTrainingConfig } = await import('/app-static/js/shared/activeTrainingConfig.js');
        await activeTrainingConfig.setActive(null);
    }""")


class TestRunPickerContents:
    def test_lists_all_runs_and_marks_active(self, page: Page, base_url: str):
        _navigate(page, base_url, [TC_1, TC_2])
        _set_active(page, TC_1)
        expect(page.locator(".active-model-bar__name")).to_have_text("Cat Detector")

        page.locator(".active-model-bar__picker-btn").click()
        rows = page.locator(".tc-run-picker__row")
        expect(rows).to_have_count(3)  # Cat Detector, Dog Detector, "No active"

        cat_row = page.locator(".tc-run-picker__row", has_text="Cat Detector")
        dog_row = page.locator(".tc-run-picker__row", has_text="Dog Detector")
        expect(cat_row).to_contain_text("✓")
        expect(dog_row).not_to_contain_text("✓")

    def test_no_active_option_absent_when_nothing_active(self, page: Page, base_url: str):
        _navigate(page, base_url, [TC_1, TC_2])
        page.locator(".active-model-bar__set-tc-btn").click()
        expect(page.locator(".tc-run-picker__row")).to_have_count(2)
        expect(page.locator(".tc-run-picker__row", has_text="No active training run")).to_have_count(0)

    def test_manage_link_present(self, page: Page, base_url: str):
        _navigate(page, base_url, [TC_1])
        page.locator(".active-model-bar__set-tc-btn").click()
        expect(page.get_by_text("Manage training runs →")).to_be_visible()


class TestRunPickerSelection:
    def test_selecting_a_run_switches_active_and_closes_popup(self, page: Page, base_url: str):
        _navigate(page, base_url, [TC_1, TC_2])
        _set_active(page, TC_1)

        page.locator(".active-model-bar__picker-btn").click()
        page.locator(".tc-run-picker__row", has_text="Dog Detector").click()

        expect(page.locator(".tc-run-picker")).to_have_count(0)
        expect(page.locator(".active-model-bar__name")).to_have_text("Dog Detector")

        active_uuid = page.evaluate("""async () => {
            const { activeTrainingConfig } = await import('/app-static/js/shared/activeTrainingConfig.js');
            return activeTrainingConfig.getActive()?.uuid;
        }""")
        assert active_uuid == "tc-2"

    def test_no_active_row_clears_selection(self, page: Page, base_url: str):
        _navigate(page, base_url, [TC_1, TC_2])
        _set_active(page, TC_1)

        page.locator(".active-model-bar__picker-btn").click()
        page.locator(".tc-run-picker__row", has_text="No active training run").click()

        expect(page.locator(".tc-run-picker")).to_have_count(0)
        expect(page.locator(".active-model-bar__none")).to_have_text("No active training run")

    def test_selecting_from_no_active_state_activates_it(self, page: Page, base_url: str):
        _navigate(page, base_url, [TC_1, TC_2])
        page.locator(".active-model-bar__set-tc-btn").click()
        page.locator(".tc-run-picker__row", has_text="Cat Detector").click()
        expect(page.locator(".active-model-bar__name")).to_have_text("Cat Detector")

    def test_manage_link_navigates_to_training_runs_tab(self, page: Page, base_url: str):
        _navigate(page, base_url, [TC_1])
        page.locator(".active-model-bar__set-tc-btn").click()
        page.get_by_text("Manage training runs →").click()

        expect(page.locator("#tab-configuration")).to_have_class(re.compile(r"\bactive\b"))
        page.wait_for_selector("#tab-training-management.active", timeout=2000)


class TestRunPickerAboveModals:
    """The whole point of the picker: it must stay usable while a modal
    overlay covers the page, since #bottom-bars already renders above
    .app-modal-overlay (z-index 10000 vs 9999) — see app.css."""

    def test_switch_run_while_modal_open(self, page: Page, base_url: str):
        _navigate(page, base_url, [TC_1, TC_2])
        _set_active(page, TC_1)

        # Open a real modal shell, same as any in-app modal would.
        page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            window.__testModal = createModalShell({ onClose: () => {} });
            window.__testModal.dialog.textContent = 'Test modal';
            window.__testModal.open();
        }""")
        expect(page.locator(".app-modal-overlay")).to_be_visible()

        page.locator(".active-model-bar__picker-btn").click()
        expect(page.locator(".tc-run-picker")).to_be_visible()

        page.locator(".tc-run-picker__row", has_text="Dog Detector").click()

        expect(page.locator(".active-model-bar__name")).to_have_text("Dog Detector")
        # The modal is untouched — no navigation happened underneath it.
        expect(page.locator(".app-modal-overlay")).to_be_visible()


LABEL_STATS = [
    {
        "labelUuid": "lbl-cat", "labelScope": "image", "hasRollup": False, "rollUpParentUuid": None,
        "presentImageCount": 12, "absentImageCount": 0, "unverifiedImageCount": 3,
    },
    {
        "labelUuid": "lbl-dog-box", "labelScope": "object", "hasRollup": False, "rollUpParentUuid": None,
        "verifiedBboxCount": 8, "unverifiedBboxCount": 2,
        "coLabelsCoveredCount": 6, "coLabelGapsCount": 2,
        "coLabelGapBreakdown": [{"labelUuid": "lbl-other", "labelName": "Other Label", "count": 2}],
    },
]
LABELS = [
    {"uuid": "lbl-cat", "name": "Cat", "color": "#ff0000", "parentUuid": None},
    {"uuid": "lbl-dog-box", "name": "Dog Box", "color": "#00ff00", "parentUuid": None},
]


class TestCoveragePopupTable:
    """The Coverage popup (opened by clicking the training-run name) shares
    ui/utils/coverageStatsTable.js with the full Training Runs page, so its
    columns are labeled and its cells clickable exactly like that page."""

    def _open_coverage_popup(self, page: Page, base_url: str):
        _navigate(page, base_url, [TC_1])
        _mock_label_stats(page, "tc-1", LABEL_STATS)
        _mock_labels(page, LABELS)
        page.route(
            lambda url: "/api/model-configs" in url.split("?")[0],
            lambda route: route.fulfill(
                status=200, content_type="application/json",
                body=json.dumps({"status": "success", "configs": []}),
            ) if route.request.method == "GET" else route.fallback(),
        )
        _set_active(page, TC_1)
        page.locator(".active-model-bar__name").click()
        page.wait_for_selector(".tc-stats-panel table")

    def test_columns_are_labeled(self, page: Page, base_url: str):
        self._open_coverage_popup(page, base_url)
        image_table_headers = page.locator(".tc-stats-panel table").nth(0).locator("thead th")
        expect(image_table_headers).to_have_text(["Label", "Positive", "Negative", "Unverified"])

        object_table_headers = page.locator(".tc-stats-panel table").nth(1).locator("thead th")
        expect(object_table_headers).to_have_text(
            ["Label", "Verified", "Unverified", "All clear", "Gaps"]
        )
        # Shortened headers still carry the full explanation as a hover tooltip.
        expect(object_table_headers.nth(1)).to_have_attribute("title", re.compile(r"^Verified boxes"))

    def test_nonzero_cell_is_clickable_and_opens_review_modal(self, page: Page, base_url: str):
        self._open_coverage_popup(page, base_url)
        page.route(
            "**/api/images/review*",
            lambda route: route.fulfill(
                status=200, content_type="application/json",
                body=json.dumps({"status": "success", "images": [], "hasMore": False, "totalCount": 0}),
            ),
        )

        cat_row = page.locator(".tc-stats-panel tr", has_text="Cat")
        cat_row.locator("td").nth(1).click()  # Positive column, value 12

        expect(page.locator(".canvas-modal-title, strong", has_text="Cat — positive (Cat Detector)")).to_be_visible()
        # The popup closes itself so it doesn't sit on top of the modal it opened.
        expect(page.locator(".tc-stats-panel")).to_have_count(0)

    def test_zero_value_cell_is_not_clickable(self, page: Page, base_url: str):
        self._open_coverage_popup(page, base_url)
        cat_row = page.locator(".tc-stats-panel tr", has_text="Cat")
        negative_cell = cat_row.locator("td").nth(2)  # Negative column, value 0
        expect(negative_cell).not_to_have_css("cursor", "pointer")


class TestReadinessInPopup:
    """The Coverage popup also shows the "images ready to train" readiness
    summary — previously only on the full Training Runs page — sharing
    ui/utils/coverageStatsTable.js's renderReadinessStats() with it."""

    def _open_popup_with_readiness(self, page: Page, base_url: str, **readiness_kwargs):
        _navigate(page, base_url, [TC_1])
        _mock_label_stats(page, "tc-1", LABEL_STATS)
        _mock_readiness(page, "tc-1", **readiness_kwargs)
        _mock_labels(page, LABELS)
        page.route(
            lambda url: "/api/model-configs" in url.split("?")[0],
            lambda route: route.fulfill(
                status=200, content_type="application/json",
                body=json.dumps({"status": "success", "configs": []}),
            ) if route.request.method == "GET" else route.fallback(),
        )
        _set_active(page, TC_1)
        page.locator(".active-model-bar__name").click()
        page.wait_for_selector(".tc-stats-panel table")

    def test_readiness_counts_are_displayed(self, page: Page, base_url: str):
        self._open_popup_with_readiness(page, base_url, ready=5, needs_review=2, not_started=3)
        panel = page.locator(".tc-stats-panel")
        expect(panel.get_by_text("Training readiness")).to_be_visible()
        expect(panel.get_by_text("✓ 5 ready", exact=False)).to_be_visible()
        expect(panel.get_by_text("⚠ 2 need review", exact=False)).to_be_visible()
        expect(panel.get_by_text("◌ 3 not started", exact=False)).to_be_visible()

    def test_ready_count_click_opens_review_modal_and_closes_popup(self, page: Page, base_url: str):
        self._open_popup_with_readiness(page, base_url, ready=5, needs_review=0, not_started=0)
        page.route(
            "**/api/training-configs/tc-1/ready-images",
            lambda route: route.fulfill(
                status=200, content_type="application/json",
                body=json.dumps({"status": "success", "images": []}),
            ),
        )
        page.locator(".tc-stats-panel", has_text="✓ 5 ready").get_by_text("✓ 5 ready", exact=False).click()
        expect(page.locator(".canvas-modal-title, strong", has_text="Ready for Training")).to_be_visible()
        expect(page.locator(".tc-stats-panel")).to_have_count(0)
