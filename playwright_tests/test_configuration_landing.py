"""
Tests for the Configuration > Overview landing (PR 5.4).

Exercises renderConfigLanding() inside #config-overview-box:
  - every Configuration rail section has an entry (the landing must not go stale)
  - counts come from the live API, not hardcoded numbers
  - Datasets vs Views are split by the `promoted` flag
  - a failing count endpoint degrades to "—" instead of blanking the pane
  - clicking an entry navigates to the sub-tab that holds it

Run with:
    pytest playwright_tests/test_configuration_landing.py -p no:xdist -p no:asyncio -v
"""

import json
import re

from playwright.sync_api import Page, expect

# ---------------------------------------------------------------------------
# Fake data — deliberately distinct counts per entry so a mixed-up wiring
# (e.g. labels count rendered under Tags) fails instead of coincidentally passing.
# ---------------------------------------------------------------------------

FAKE_LABELS = [
    {"uuid": f"aaaaaaaa-0000-0000-0000-00000000000{i}", "name": f"L{i}", "color": "#fff", "parentUuid": None}
    for i in range(1, 4)  # 3 labels
]

FAKE_TAGS = [{"uuid": f"bbbbbbbb-0000-0000-0000-00000000000{i}", "name": f"T{i}"} for i in range(1, 3)]  # 2

FAKE_MODELS = ["model_a.onnx", "model_b.onnx", "model_c.onnx", "model_d.onnx"]  # 4

FAKE_ROWS = [
    {"uuid": "cccccccc-0000-0000-0000-000000000001", "name": "DS one", "promoted": True},
    {"uuid": "cccccccc-0000-0000-0000-000000000002", "name": "DS two", "promoted": True},
    {"uuid": "cccccccc-0000-0000-0000-000000000003", "name": "DS three", "promoted": True},
    {"uuid": "cccccccc-0000-0000-0000-000000000004", "name": "DS four", "promoted": True},
    {"uuid": "cccccccc-0000-0000-0000-000000000005", "name": "DS five", "promoted": True},
    {"uuid": "dddddddd-0000-0000-0000-000000000001", "name": "View one", "promoted": False},
]  # 5 datasets / 1 view

FAKE_MODEL_CONFIGS = [{"uuid": f"eeeeeeee-0000-0000-0000-00000000000{i}", "name": f"BP{i}"} for i in range(1, 8)]  # 7

FAKE_TRAINING_CONFIGS = [{"uuid": f"ffffffff-0000-0000-0000-00000000000{i}", "name": f"TR{i}"} for i in range(1, 10)]

FAKE_FILTER_CHAINS = [{"uuid": "99999999-0000-0000-0000-000000000001", "name": "Chain"}]  # 1

FAKE_SOURCES = {
    f"11111111-0000-0000-0000-00000000000{i}": {"uuid": f"11111111-0000-0000-0000-00000000000{i}", "name": f"S{i}"}
    for i in range(1, 3)
}  # 2

FAKE_TASK_CONFIGS = {
    f"22222222-0000-0000-0000-00000000000{i}": {"uuid": f"22222222-0000-0000-0000-00000000000{i}", "name": f"TC{i}"}
    for i in range(1, 7)
}  # 6

FAKE_WATCH_PROFILES = [{"uuid": f"33333333-0000-0000-0000-00000000000{i}", "name": f"W{i}"} for i in range(1, 9)]  # 8


# ---------------------------------------------------------------------------
# Mocking helpers
# ---------------------------------------------------------------------------


def _json_route(page: Page, url_pattern, body: dict, status: int = 200):
    page.route(
        url_pattern,
        lambda route: (
            route.fulfill(status=status, content_type="application/json", body=json.dumps(body))
            if route.request.method == "GET"
            else route.continue_()
        ),
    )


def _mock_counts(page: Page, *, break_labels: bool = False):
    """Intercept every endpoint the landing counts from."""
    _json_route(
        page,
        lambda url: url.split("?")[0].endswith("/api/labels") and "/api/labels/" not in url,
        {"status": "failure", "message": "boom"} if break_labels else {"status": "success", "labels": FAKE_LABELS},
        status=500 if break_labels else 200,
    )
    _json_route(page, "**/api/tags/list", {"tags": FAKE_TAGS})
    _json_route(page, "**/api/models", {"status": "success", "models": FAKE_MODELS})
    _json_route(
        page,
        lambda url: url.split("?")[0].endswith("/api/datasets"),
        {"status": "success", "datasets": FAKE_ROWS},
    )
    _json_route(page, "**/api/model-configs", {"status": "success", "configs": FAKE_MODEL_CONFIGS})
    _json_route(page, "**/api/training-configs", {"status": "success", "configs": FAKE_TRAINING_CONFIGS})
    _json_route(page, "**/api/filter-chains", {"status": "success", "chains": FAKE_FILTER_CHAINS})
    _json_route(page, "**/api/sources/get", {"status": "success", "sources": FAKE_SOURCES})
    _json_route(page, "**/api/watch-profiles", {"status": "success", "profiles": FAKE_WATCH_PROFILES})
    # Task configs are fetched with POST, so this one cannot go through _json_route.
    page.route(
        "**/api/tasks/configs/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": FAKE_TASK_CONFIGS}),
        ),
    )


def _navigate_to_landing(page: Page, base_url: str, *, break_labels: bool = False):
    """Open the app on the Configuration tab, which lands on Overview by default."""
    _mock_counts(page, break_labels=break_labels)
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_selector("[data-testid='config-landing'] .config-landing__grid", timeout=4000)


def _entry(page: Page, tab: str, name: str):
    slug = name.lower().replace(" ", "-")
    return page.locator(f"[data-testid='config-landing-entry-{tab}-{slug}']")


def _count_text(page: Page, tab: str, name: str) -> str:
    return _entry(page, tab, name).locator(".config-landing__count").inner_text().strip()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfigLandingStructure:
    def test_landing_is_the_default_configuration_subtab(self, page: Page, base_url: str):
        _navigate_to_landing(page, base_url)
        expect(page.locator("#tab-config-overview")).to_have_class(re.compile(r"\bactive\b"))

    def test_sibling_subtabs_remain_directly_reachable(self, page: Page, base_url: str):
        """The landing is an extra sub-tab, not a gate — nothing gains a click."""
        _navigate_to_landing(page, base_url)
        for tab in (
            "tab-tag-management",
            "tab-dataset-management",
            "tab-model-management",
            "tab-training-management",
            "tab-filterchain-management",
        ):
            expect(page.locator(f"#tab-configuration .workspace-tab-button[data-tab='{tab}']")).to_be_visible()


class TestConfigLandingCounts:
    def test_counts_come_from_the_api(self, page: Page, base_url: str):
        _navigate_to_landing(page, base_url)
        assert _count_text(page, "tab-tag-management", "Labels") == "3"
        assert _count_text(page, "tab-tag-management", "Tags") == "2"
        assert _count_text(page, "tab-tag-management", "Inference Class Map") == "4"
        assert _count_text(page, "tab-model-management", "Model Shapes") == "7"
        assert _count_text(page, "tab-training-management", "Training") == "9"
        assert _count_text(page, "tab-filterchain-management", "Filter Chains") == "1"

    def test_every_configuration_subtab_has_a_landing_entry(self, page: Page, base_url: str):
        """The landing is the map of Configuration — a new rail section must appear here too."""
        _navigate_to_landing(page, base_url)
        rail_tabs = page.eval_on_selector_all(
            "#tab-configuration .config-rail__item",
            "els => els.map(e => e.dataset.tab)",
        )
        entry_tabs = page.eval_on_selector_all(
            "[data-testid='config-landing'] .config-landing__entry",
            "els => els.map(e => e.dataset.testid)",
        )
        missing = [
            tab
            for tab in rail_tabs
            if tab != "tab-config-overview"
            and not any(t.startswith(f"config-landing-entry-{tab}-") for t in entry_tabs)
        ]
        assert missing == [], f"Configuration sub-tabs with no Overview entry: {missing}"

    def test_the_newer_sections_are_counted(self, page: Page, base_url: str):
        _navigate_to_landing(page, base_url)
        assert _count_text(page, "tab-source-management", "Sources") == "2"
        assert _count_text(page, "tab-task-management", "Tasks") == "6"
        assert _count_text(page, "tab-watcher-management", "Watch Rules") == "8"

    def test_datasets_and_views_are_split_by_the_promoted_flag(self, page: Page, base_url: str):
        """A View must never be counted as a trainable Dataset (the slice 5 trap)."""
        _navigate_to_landing(page, base_url)
        assert _count_text(page, "tab-dataset-management", "Datasets") == "5"
        assert _count_text(page, "tab-dataset-management", "Views") == "1"

    def test_a_failing_count_degrades_to_a_dash(self, page: Page, base_url: str):
        """One dead endpoint must not blank the whole landing."""
        _navigate_to_landing(page, base_url, break_labels=True)
        assert _count_text(page, "tab-tag-management", "Labels") == "—"
        # every other count still renders
        assert _count_text(page, "tab-model-management", "Model Shapes") == "7"


class TestConfigLandingNavigation:
    def test_entry_navigates_to_its_subtab(self, page: Page, base_url: str):
        _navigate_to_landing(page, base_url)
        _entry(page, "tab-model-management", "Model Shapes").click()
        expect(page.locator("#tab-model-management")).to_have_class(re.compile(r"\bactive\b"))

    def test_views_entry_navigates_to_the_dataset_pane(self, page: Page, base_url: str):
        """Views live in the Dataset Management pane (PR 5.3) — no separate screen."""
        _navigate_to_landing(page, base_url)
        _entry(page, "tab-dataset-management", "Views").click()
        expect(page.locator("#tab-dataset-management")).to_have_class(re.compile(r"\bactive\b"))
