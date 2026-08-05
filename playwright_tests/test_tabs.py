"""
Playwright smoke tests for the image_tagging_server SPA.

Run with:
    pytest tests/playwright/ -p no:xdist -v
"""

import re

import pytest
from playwright.sync_api import Page, expect

# ---------------------------------------------------------------------------
# Page load
# ---------------------------------------------------------------------------


def test_page_loads(page: Page, base_url: str):
    """The root URL returns HTML with the expected tab container."""
    page.goto(base_url)
    expect(page.locator("#main-tabs")).to_be_visible()


def test_page_title(page: Page, base_url: str):
    page.goto(base_url)
    expect(page).to_have_title("Annotation Tool")


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


def test_switch_to_review_tab(page: Page, base_url: str):
    """Clicking Review activates the Review pane."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    page.locator(".main-tab-button[data-tab='tab-review']").click()

    expect(page.locator(".main-tab-button[data-tab='tab-review']")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#tab-review")).to_have_class(re.compile(r"\bactive\b"))


def test_switch_tabs_and_back(page: Page, base_url: str):
    """Navigate between all main tabs and return to Image Tagging."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    for tab in ["tab-configuration", "tab-review", "tab-operations", "tab-image-tagging"]:
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
        "tab-training-management",
        "tab-filterchain-management",
        "tab-tag-management",
        "tab-config-overview",
    ]
    for tab in sub_tabs:
        page.locator(f"#tab-configuration .workspace-tab-button[data-tab='{tab}']").click()
        expect(page.locator(f"#{tab}")).to_have_class(re.compile(r"\bactive\b"))


# ---------------------------------------------------------------------------
# Operations tab — Sources and Tasks moved here out of Configuration
# ---------------------------------------------------------------------------


def test_operations_default_subtab_is_sources(page: Page, base_url: str):
    """The Sources sub-tab should be active by default inside Operations."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    page.locator(".main-tab-button[data-tab='tab-operations']").click()

    sources_btn = page.locator("#tab-operations .workspace-tab-button[data-tab='tab-source-management']")
    expect(sources_btn).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#tab-source-management")).to_have_class(re.compile(r"\bactive\b"))


def test_operations_subtab_switching(page: Page, base_url: str):
    """Each operations sub-tab can be activated."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    page.locator(".main-tab-button[data-tab='tab-operations']").click()

    for tab in ["tab-task-management", "tab-source-management"]:
        page.locator(f"#tab-operations .workspace-tab-button[data-tab='{tab}']").click()
        expect(page.locator(f"#{tab}")).to_have_class(re.compile(r"\bactive\b"))


def test_sources_and_tasks_are_not_in_configuration(page: Page, base_url: str):
    """Sources and Tasks moved to Operations; Configuration keeps its setup sub-tabs.

    5 setup panes plus the Overview landing added in PR 5.4.
    """
    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    page.locator(".main-tab-button[data-tab='tab-configuration']").click()

    header = page.locator("#tab-configuration .workspace-tab-header")
    expect(header.locator(".workspace-tab-button")).to_have_count(6)
    for tab in ["tab-source-management", "tab-task-management"]:
        expect(header.locator(f".workspace-tab-button[data-tab='{tab}']")).to_have_count(0)


def test_labels_api_called_when_entering_configuration(page: Page, base_url: str):
    """Switching to Configuration triggers a call to /api/labels."""
    api_requests: list[str] = []
    page.on("request", lambda req: api_requests.append(req.url))

    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")

    label_calls = [u for u in api_requests if "/api/labels" in u]
    assert label_calls, "Expected at least one /api/labels request when entering Configuration"


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

    for tab in ["tab-configuration", "tab-review", "tab-operations", "tab-image-tagging"]:
        page.locator(f".main-tab-button[data-tab='{tab}']").click()
        page.wait_for_load_state("networkidle")

    assert errors == [], f"Console errors while switching tabs: {errors}"
