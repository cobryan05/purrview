"""
Playwright smoke tests for the image_tagging_server SPA.

Run with:
    pytest tests/playwright/ -p no:xdist -v
"""

import re

import pytest
from playwright.sync_api import Page, expect

from playwright_tests.waits import wait_until

# ---------------------------------------------------------------------------
# Page load
# ---------------------------------------------------------------------------


def test_page_loads(page: Page, base_url: str):
    """The root URL returns HTML with the expected tab container."""
    page.goto(base_url)
    expect(page.locator("#main-tabs")).to_be_visible()


def test_page_title(page: Page, base_url: str):
    page.goto(base_url)
    expect(page).to_have_title("PurrView")


# ---------------------------------------------------------------------------
# Main tab switching
# ---------------------------------------------------------------------------


def test_initial_tab_is_image_tagging(page: Page, base_url: str):
    """Image Tagging should be the active tab on first load."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    button = page.locator(".main-tab-button[data-tab='tab-image-tagging']")
    expect(button).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#tab-image-tagging")).to_have_class(re.compile(r"\bactive\b"))


def test_switch_to_configuration_tab(page: Page, base_url: str):
    """Clicking Configuration activates that tab and deactivates Image Tagging."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    page.locator(".main-tab-button[data-tab='tab-configuration']").click()

    expect(page.locator(".main-tab-button[data-tab='tab-configuration']")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#tab-configuration")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#tab-image-tagging")).not_to_have_class(re.compile(r"\bactive\b"))


def test_switch_to_review_section_tab(page: Page, base_url: str):
    """Review is a section tab inside Annotate, not a main tab.

    It shares canvas/, files.js and inferencePanel.js with Canvas by direct
    import — it is the same annotation work at grid zoom.
    """
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
    page.locator("#tab-image-tagging .section-tab-button[data-tab='tab-review']").click()

    expect(page.locator(".section-tab-button[data-tab='tab-review']")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#tab-review")).to_have_class(re.compile(r"\bactive\b"))
    # Review is no longer its own top-level destination.
    expect(page.locator(".main-tab-button[data-tab='tab-review']")).to_have_count(0)


def test_annotate_section_tabs_are_canvas_and_review(page: Page, base_url: str):
    """Annotate holds exactly Canvas and Review at the section tier."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    header = page.locator("#tab-image-tagging .section-tab-header")
    expect(header.locator(".section-tab-button")).to_have_count(2)
    for tab in ["tab-annotate-workspace", "tab-review"]:
        expect(header.locator(f".section-tab-button[data-tab='{tab}']")).to_have_count(1)


def test_switch_tabs_and_back(page: Page, base_url: str):
    """Navigate between all main tabs and return to Annotate."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    for tab in ["tab-configuration", "tab-nvr", "tab-runs", "tab-image-tagging"]:
        page.locator(f".main-tab-button[data-tab='{tab}']").click()
        expect(page.locator(f"#{tab}")).to_have_class(re.compile(r"\bactive\b"))

    # Only one main pane should be active at the end
    active_panes = page.locator(".main-tab-pane.active")
    expect(active_panes).to_have_count(1)


# ---------------------------------------------------------------------------
# Configuration tab — sub-tab and API calls
# ---------------------------------------------------------------------------


def test_configuration_default_subtab_is_overview(page: Page, base_url: str):
    """The Overview landing should be active by default inside Configuration."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    page.locator(".main-tab-button[data-tab='tab-configuration']").click()

    # Wait for the sub-tab header to appear
    overview_btn = page.locator(".workspace-tab-button[data-tab='tab-config-overview']")
    expect(overview_btn).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#tab-config-overview")).to_have_class(re.compile(r"\bactive\b"))


def test_configuration_subtab_switching(page: Page, base_url: str):
    """Each configuration sub-tab can be activated."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    page.locator(".main-tab-button[data-tab='tab-configuration']").click()

    sub_tabs = [
        "tab-dataset-management",
        "tab-model-management",
        "tab-modelstack-management",
        "tab-training-management",
        "tab-filterchain-management",
        "tab-tag-management",
        "tab-source-management",
        "tab-task-management",
        "tab-config-overview",
    ]
    for tab in sub_tabs:
        page.locator(f"#tab-configuration .workspace-tab-button[data-tab='{tab}']").click()
        expect(page.locator(f"#{tab}")).to_have_class(re.compile(r"\bactive\b"))


# ---------------------------------------------------------------------------
# Runs tab — Operations dissolved into Configuration + Runs
# ---------------------------------------------------------------------------


def test_runs_holds_the_task_instance_list(page: Page, base_url: str):
    """Runs is a main tab holding the instance list directly, with no sub-tabs.

    It was Operations > Activity. Operations' other two panes (Tasks, Sources)
    were definition editors and moved to Configuration, leaving the one pane
    that was genuinely about running.
    """
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    page.locator(".main-tab-button[data-tab='tab-runs']").click()

    expect(page.locator("#tab-runs")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#active-tasks-list")).to_be_visible()
    # Operations is gone entirely.
    expect(page.locator(".main-tab-button[data-tab='tab-operations']")).to_have_count(0)


def test_nvr_tab_has_live_and_events(page: Page, base_url: str):
    """NVR exists with both sub-tabs, even though Events is a placeholder."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    page.locator(".main-tab-button[data-tab='tab-nvr']").click()
    expect(page.locator("#tab-nvr")).to_have_class(re.compile(r"\bactive\b"))

    for tab in ["tab-nvr-live", "tab-nvr-events"]:
        page.locator(f"#tab-nvr .workspace-tab-button[data-tab='{tab}']").click()
        expect(page.locator(f"#{tab}")).to_have_class(re.compile(r"\bactive\b"))


def test_sources_and_tasks_are_in_configuration(page: Page, base_url: str):
    """Sources and Tasks are definition editors, so they live in Configuration.

    They were under Operations, which put two config editors on a surface named
    for running things — the mirror of Training's Start button living in
    Configuration. Eleven rail destinations: Overview, Labels & Tags, Datasets,
    Sources, Model Shapes, Model Stacks, Training, Filter Chains, Collectors,
    Tasks, Watch Rules.
    """
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    page.locator(".main-tab-button[data-tab='tab-configuration']").click()

    rail = page.locator("#tab-configuration .config-rail")
    expect(rail.locator(".workspace-tab-button")).to_have_count(11)
    for tab in ["tab-source-management", "tab-task-management"]:
        expect(rail.locator(f".workspace-tab-button[data-tab='{tab}']")).to_have_count(1)


def test_labels_api_called_when_entering_configuration(page: Page, base_url: str):
    """Switching to Configuration triggers a call to /api/labels."""
    api_requests: list[str] = []
    page.on("request", lambda req: api_requests.append(req.url))

    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()

    # `networkidle` is not a barrier on the request under test — it resolves whenever
    # the network is momentarily quiet, which under load can be before the tab's lazy
    # module has even been fetched. Poll the capture list instead.
    wait_until(
        page,
        lambda: [u for u in api_requests if "/api/labels" in u],
        "Expected at least one /api/labels request when entering Configuration",
    )


def test_server_config_endpoint_responds(page: Page, base_url: str):
    """The /server-config endpoint returns a success response with host info."""
    import httpx

    resp = httpx.get(f"{base_url}/server-config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "inference_server" in data["config"]
    assert "db_server" in data["config"]
    assert "tasks_server" in data["config"]


# ---------------------------------------------------------------------------
# No JS console errors on tab switch
# ---------------------------------------------------------------------------


def test_no_console_errors_on_load(page: Page, base_url: str):
    """No JS console errors (level 'error') during initial page load."""
    errors: list[str] = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    assert errors == [], f"Console errors on load: {errors}"


def test_no_console_errors_switching_tabs(page: Page, base_url: str):
    """No JS console errors while switching through all main tabs."""
    errors: list[str] = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    for tab in ["tab-configuration", "tab-nvr", "tab-runs", "tab-image-tagging"]:
        page.locator(f".main-tab-button[data-tab='{tab}']").click()
        page.wait_for_load_state("networkidle")

    # Review is a section tab now, and is the tab most likely to throw on a
    # structural change — it is lazily initialised one tier deeper than before.
    page.locator("#tab-image-tagging .section-tab-button[data-tab='tab-review']").click()
    page.wait_for_load_state("networkidle")

    assert errors == [], f"Console errors while switching tabs: {errors}"
