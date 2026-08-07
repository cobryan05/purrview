"""Service Status lives in Operations, not Settings.

Decision register #8 moves service health to the Operations surface; UI_REDESIGN
§12 calls its presence in Settings the original miscategorisation (it persists
nothing and is pure live monitoring).
"""

import json

from playwright.sync_api import Page, expect

from playwright_tests.waits import wait_until

SERVICES = [
    {"service": "db_server", "status": "ok", "hash_index": {"image_valid": 42, "needs_rebuild": False}},
    {"service": "inference_server", "status": "ok", "models": {"loaded": 1, "available": 3}},
    {"service": "tasks_server", "status": "ok", "tasks": {"running": 2}},
    {"service": "training_server", "status": "ok"},
    {"service": "relay_buffer_server", "status": "unreachable", "message": "connection refused"},
]


def _mock_service_status(page: Page, services=None):
    page.route(
        "**/api/service-status",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "services": services if services is not None else SERVICES}),
        ),
    )


def _navigate_to_operations(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-operations']").click()
    page.wait_for_load_state("networkidle")


def _expand_service_status(page: Page):
    """Open the service-status disclosure.

    The panel is a <details> collapsed by default -- expanded it is five rows
    above every Operations sub-tab, and the answer is almost always "all green".
    The per-service rows only exist to be looked at on demand, so tests that
    assert on them have to open it first.
    """
    panel = page.locator("[data-testid='service-status-panel']")
    expect(panel).to_be_visible()
    if not panel.evaluate("el => el.open"):
        page.locator(".service-status-panel__header").click()
    page.wait_for_selector(".service-status-row", state="visible", timeout=4000)
    # `<details>`'s `toggle` event is fired *asynchronously* (queued as a task),
    # so the pref write it drives has not necessarily run by the time `open` is
    # already true and the rows are already visible. Navigating inside that
    # window drops the write. Wait for the value to actually land.
    wait_until(
        page,
        lambda: page.evaluate("localStorage.getItem('ww.operations.serviceStatusExpanded')") == "true",
        "Service-status expanded state was never persisted",
    )


class TestServiceStatusInOperations:
    def test_panel_renders_a_row_per_service(self, page: Page, base_url: str):
        _mock_service_status(page)
        _navigate_to_operations(page, base_url)

        _expand_service_status(page)
        panel = page.locator("[data-testid='service-status-panel']")
        expect(panel.locator(".service-status-row")).to_have_count(len(SERVICES))

    def test_row_detail_reflects_service_payload(self, page: Page, base_url: str):
        _mock_service_status(page)
        _navigate_to_operations(page, base_url)

        _expand_service_status(page)
        expect(page.locator("[data-testid='service-status-row-tasks_server']")).to_contain_text("running: 2")
        expect(page.locator("[data-testid='service-status-row-inference_server']")).to_contain_text("loaded: 1/3")
        # Unreachable services surface their message rather than a detail summary.
        expect(page.locator("[data-testid='service-status-row-relay_buffer_server']")).to_contain_text(
            "connection refused"
        )

    def test_refresh_button_refetches(self, page: Page, base_url: str):
        calls: list[str] = []
        page.on("request", lambda req: calls.append(req.url) if "/api/service-status" in req.url else None)

        _mock_service_status(page)
        _navigate_to_operations(page, base_url)
        expect(page.locator("[data-testid='service-status-panel']")).to_be_visible()
        before = len(calls)

        page.locator("[data-testid='service-status-refresh']").click()
        page.wait_for_load_state("networkidle")

        assert len(calls) > before, "Refresh did not re-request /api/service-status"

    def test_training_server_row_opens_the_job_modal(self, page: Page, base_url: str):
        """The training-server row is the one clickable row, and its modal holds a
        per-job DELETE — the piece most easily lost when moving this panel."""
        _mock_service_status(page)
        page.route(
            "**/api/training/jobs",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"jobs": []}),
            ),
        )
        _navigate_to_operations(page, base_url)

        _expand_service_status(page)
        page.locator("[data-testid='service-status-row-training_server']").click()
        expect(page.locator("[data-testid='modal-shell-dialog']")).to_be_visible()


class TestServiceStatusCollapsedSummary:
    def test_panel_is_collapsed_by_default(self, page: Page, base_url: str):
        """Collapsed is the default: the strip must not eat a third of the pane
        above every Operations sub-tab just to say everything is fine."""
        _mock_service_status(page)
        _navigate_to_operations(page, base_url)

        panel = page.locator("[data-testid='service-status-panel']")
        expect(panel).to_be_visible()
        assert panel.evaluate("el => el.open") is False
        expect(page.locator(".service-status-row").first).not_to_be_visible()

    def test_summary_names_the_unreachable_service(self, page: Page, base_url: str):
        """The collapsed line must say *which* service is unhealthy — a bare
        count would send you into the list anyway, defeating the collapse."""
        _mock_service_status(page)
        _navigate_to_operations(page, base_url)

        summary = page.locator("[data-testid='service-status-summary']")
        expect(summary).to_be_visible()
        expect(summary).to_contain_text("Relay Buffer Server")
        expect(summary).to_have_attribute("data-tone", "warn")

    def test_expanded_state_survives_a_reload(self, page: Page, base_url: str):
        """Whether the panel is open is a user preference, not per-visit state."""
        _mock_service_status(page)
        _navigate_to_operations(page, base_url)
        _expand_service_status(page)

        _navigate_to_operations(page, base_url)
        panel = page.locator("[data-testid='service-status-panel']")
        expect(panel).to_be_visible()
        assert panel.evaluate("el => el.open") is True


class TestServiceStatusNotInSettings:
    def test_settings_no_longer_shows_service_status(self, page: Page, base_url: str):
        _mock_service_status(page)
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.locator(".main-tab-button[data-tab='tab-settings']").click()
        page.wait_for_selector("#tab-settings .settings-wrap", timeout=4000)

        expect(page.locator("#tab-settings [data-testid='service-status-panel']")).to_have_count(0)
