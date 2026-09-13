"""
Tests for the Settings tab and the file browser.

Covers two recently-added, browser-untested areas:
  - apps/image_tagging_server/static/js/app/settings/main.js
    (server-config override rows, save/test flows, hotkeys collapse panel)
  - apps/image_tagging_server/static/js/ui/fileBrowser.js
    (entry rendering, pagination "Load more", directory navigation, file selection)

Both proxies default to empty data in conftest, so each test mocks exactly the
endpoints it needs via page.route() — following the interception pattern in
test_configuration_sources.py / test_configuration_labels.py.

Run with:
    pytest playwright_tests/test_settings.py -p no:xdist -p no:asyncio -v
"""

import base64
import json

import pytest
from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------


def _navigate_to_settings(page: Page, base_url: str):
    """Open the app and activate the Settings tab.

    Lands on the rail's default pane (Service Status) — every other section's
    fields are built into the DOM already but sit in a hidden pane until its
    rail item is clicked, see `_open_settings_section`.
    """
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-settings']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#tab-settings .config-rail")


def _open_settings_section(page: Page, tab: str):
    """Click a Settings rail item and wait for its pane to become the visible one."""
    page.locator(f"#tab-settings .config-rail__item[data-tab='{tab}']").click()
    expect(page.locator(f"#{tab}")).to_be_visible()


def _put_for(put_calls, key):
    """Filter captured (url, body) PUTs down to the ones for settings key `key`.

    All Settings panes build concurrently (settings/main.js), and Keyboard Shortcuts
    fires its own background `hotkey_bindings` migration PUT independent of anything
    a test clicked. Under load that PUT can land after a test's own save PUT is
    issued, so indexing `put_calls[0]` is a race — filter by key instead. See
    FLAKY_TESTS.md (test_save_writes_json_under_mqtt_broker).
    """
    return [c for c in put_calls if c[0].rstrip("/").endswith(f"/{key}")]


def _navigate_to_files(page: Page, base_url: str):
    """Open the app and activate Image Tagging > Files."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
    page.wait_for_load_state("networkidle")
    # `[data-tab-wired]`, because Annotate is lazy-initialised: until its init has
    # attached the listeners the button is in the DOM but inert, and a click on it does
    # nothing at all, silently. Same guard navigateTabs uses.
    page.locator(".workspace-tab-button[data-tab='tab-files'][data-tab-wired]").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#tab-files .fb-container")


# ---------------------------------------------------------------------------
# Settings mocking helpers
# ---------------------------------------------------------------------------


def _mock_server_config(page: Page, overrides: dict | None = None):
    overrides = overrides or {}
    config = {
        "inference_server": "http://localhost:8001",
        "training_server": "http://localhost:8005",
        "inference_server_url_override": overrides.get("inference_server_url_override"),
        "training_server_url_override": overrides.get("training_server_url_override"),
    }
    page.route(
        "**/server-config",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "config": config}),
        ),
    )


def _mock_settings_get(page: Page, values: dict | None = None):
    """Intercept GET /api/settings/{key} -> {"value": ...} (404 when absent)."""
    values = values or {}

    def handle(route: Route):
        if route.request.method != "GET":
            return route.fallback()
        key = route.request.url.rstrip("/").rsplit("/", 1)[-1]
        if key in values:
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"value": values[key]}))
        else:
            route.fulfill(status=404, content_type="application/json", body=json.dumps({"detail": "not found"}))

    page.route(lambda url: "/api/settings/" in url, handle)


def _mock_test_server(page: Page, success=True, message="Connected OK"):
    calls = []

    def handle(route: Route):
        calls.append(json.loads(route.request.post_data))
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"success": success, "message": message}),
        )

    page.route("**/api/test-server", handle)
    return calls


def _mock_reload_server_config(page: Page, new_config: dict):
    calls = []

    def handle(route: Route):
        calls.append(True)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "config": new_config}),
        )

    page.route("**/api/reload-server-config", handle)
    return calls


# ---------------------------------------------------------------------------
# 1. Server-config rows render
# ---------------------------------------------------------------------------


class TestServerConfigRows:
    def test_rows_render_with_defaults(self, page: Page, base_url: str):
        """Both server rows render with their label, default placeholder, and status text."""
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-servers")

        rows = page.locator("#settings-section-server .settings-server-row")
        expect(rows).to_have_count(2)

        labels = rows.locator(".settings-server-label").all_text_contents()
        assert labels == ["Inference Server", "Training Server"]

        inf_row = rows.nth(0)
        expect(inf_row.locator(".settings-server-input")).to_have_attribute(
            "placeholder", "http://localhost:8001  (default)"
        )
        expect(inf_row.locator(".settings-server-status")).to_have_text("Using default: http://localhost:8001")
        expect(inf_row.locator(".settings-server-status")).to_have_attribute("data-state", "muted")

    def test_existing_override_shown(self, page: Page, base_url: str):
        """When an override is configured, the input is pre-filled and status reflects it."""
        _mock_server_config(page, {"inference_server_url_override": "http://my-inference:9001"})
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-servers")

        inf_row = page.locator("#tab-settings .settings-server-row").nth(0)
        expect(inf_row.locator(".settings-server-input")).to_have_value("http://my-inference:9001")
        expect(inf_row.locator(".settings-server-status")).to_have_text("Using: http://my-inference:9001")
        expect(inf_row.locator(".settings-server-status")).to_have_attribute("data-state", "info")


# ---------------------------------------------------------------------------
# 2. Save flow
# ---------------------------------------------------------------------------


class TestSaveOverride:
    def test_save_persists_override_and_reloads_config(self, page: Page, base_url: str):
        """Filling the input and clicking Save calls setSetting then reload-server-config,
        and the status line updates to reflect the new override."""
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-servers")

        put_calls = []

        def handle_put(route: Route):
            if route.request.method == "PUT":
                put_calls.append((route.request.url, json.loads(route.request.post_data)))
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

        page.route(lambda url: "/api/settings/" in url, handle_put)
        reload_calls = _mock_reload_server_config(
            page, {"inference_server": "http://localhost:8001", "training_server": "http://localhost:8005"}
        )

        inf_row = page.locator("#tab-settings .settings-server-row").nth(0)
        inf_row.locator(".settings-server-input").fill("http://override.example.com:9999")
        inf_row.locator("button", has_text="Save").click()
        wait_until(
            page,
            lambda: _put_for(put_calls, "inference_server_url"),
            "Expected PUT to /api/settings/inference_server_url",
        )
        _, body = _put_for(put_calls, "inference_server_url")[0]
        assert body == {"value": "http://override.example.com:9999"}
        # The reload POST is fired *after* the PUT resolves, so it needs its own
        # wait — arriving at the PUT says nothing about the reload having landed.
        wait_until(page, lambda: reload_calls, "Expected POST to /api/reload-server-config")

        expect(inf_row.locator(".settings-server-status")).to_have_text("Using: http://override.example.com:9999")
        expect(inf_row.locator(".settings-server-status")).to_have_attribute("data-state", "info")
        expect(inf_row.locator("button", has_text="Saved")).to_be_visible()

    def test_save_empty_clears_override(self, page: Page, base_url: str):
        """Saving with an empty input persists an empty string (clears the override)."""
        _mock_server_config(page, {"inference_server_url_override": "http://old-override:1234"})
        _mock_settings_get(page, {"inference_server_url": "http://old-override:1234"})
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-servers")

        put_calls = []

        def handle_put(route: Route):
            if route.request.method == "PUT":
                put_calls.append((route.request.url, json.loads(route.request.post_data)))
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

        page.route(lambda url: "/api/settings/" in url, handle_put)
        _mock_reload_server_config(
            page, {"inference_server": "http://localhost:8001", "training_server": "http://localhost:8005"}
        )

        inf_row = page.locator("#tab-settings .settings-server-row").nth(0)
        inf_row.locator(".settings-server-input").fill("")
        inf_row.locator("button", has_text="Save").click()
        wait_until(
            page,
            lambda: _put_for(put_calls, "inference_server_url"),
            "Expected PUT to /api/settings/inference_server_url",
        )
        _, body = _put_for(put_calls, "inference_server_url")[0]
        assert body == {"value": ""}
        expect(inf_row.locator(".settings-server-status")).to_have_text("Using default: http://localhost:8001")
        expect(inf_row.locator(".settings-server-status")).to_have_attribute("data-state", "muted")


# ---------------------------------------------------------------------------
# 3. Test-connection flow
# ---------------------------------------------------------------------------


class TestTestConnection:
    def test_test_button_calls_api_and_shows_success(self, page: Page, base_url: str):
        """Clicking 'Test' POSTs the current URL + server type and shows a success message."""
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-servers")
        calls = _mock_test_server(page, success=True, message="Connected OK")

        train_row = page.locator("#tab-settings .settings-server-row").nth(1)
        train_row.locator(".settings-server-input").fill("http://my-training:7777")
        train_row.locator("button", has_text="Test").click()
        wait_until(page, lambda: calls, "Expected POST to /api/test-server")
        assert calls[0] == {"url": "http://my-training:7777", "server_type": "training_server"}
        expect(train_row.locator(".settings-server-status")).to_have_text("✓ Connected OK")
        expect(train_row.locator(".settings-server-status")).to_have_attribute("data-state", "ok")

    def test_test_button_shows_failure_message(self, page: Page, base_url: str):
        """A failed connectivity check renders the ✗ message in error state."""
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-servers")
        _mock_test_server(page, success=False, message="Connection refused")

        inf_row = page.locator("#tab-settings .settings-server-row").nth(0)
        inf_row.locator("button", has_text="Test").click()
        page.wait_for_timeout(500)

        expect(inf_row.locator(".settings-server-status")).to_have_text("✗ Connection refused")
        expect(inf_row.locator(".settings-server-status")).to_have_attribute("data-state", "error")


# ---------------------------------------------------------------------------
# 3b. Layout — a left rail of destinations, one pane visible at a time
# ---------------------------------------------------------------------------

# Every rail destination the tab is expected to expose, grouped the way the
# rail groups them. A section silently dropped from the rail (e.g. a typo in
# its `data-tab`) would otherwise only be noticed by whatever test happens to
# open it directly — this is the one place that checks the full roster.
SETTINGS_RAIL_SECTIONS = [
    "tab-settings-status",
    "tab-settings-servers",
    "tab-settings-mqtt",
    "tab-settings-retention",
    "tab-settings-timelapse",
    "tab-settings-models",
    "tab-settings-storage",
    "tab-settings-security",
    "tab-settings-dedup",
    "tab-settings-interface",
    "tab-settings-hotkeys",
]


class TestSettingsLayout:
    def test_rail_lists_every_section_and_defaults_to_service_status(self, page: Page, base_url: str):
        """Mirrors Configuration's left-rail layout: a nav of destinations, one active pane."""
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)

        items = page.locator("#tab-settings .config-rail__item")
        expect(items).to_have_count(len(SETTINGS_RAIL_SECTIONS))
        tabs = items.evaluate_all("els => els.map(el => el.dataset.tab)")
        assert set(tabs) == set(SETTINGS_RAIL_SECTIONS)

        # Lands on Service Status without any click, and it's the only visible pane.
        expect(page.locator("#tab-settings-status")).to_be_visible()
        expect(page.locator("#tab-settings-servers")).to_be_hidden()

    def test_selecting_a_rail_item_shows_only_that_pane(self, page: Page, base_url: str):
        """Every section is built into the DOM up front (so switching is instant), but
        only the selected one is on screen — the rest sit behind `display: none`."""
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)

        _open_settings_section(page, "tab-settings-mqtt")
        expect(page.locator("#tab-settings-mqtt")).to_be_visible()
        expect(page.locator("#tab-settings-status")).to_be_hidden()
        expect(page.locator("#tab-settings-servers")).to_be_hidden()

        _open_settings_section(page, "tab-settings-servers")
        expect(page.locator("#tab-settings-servers")).to_be_visible()
        expect(page.locator("#tab-settings-mqtt")).to_be_hidden()

    def test_narrow_controls_stay_narrow_and_inside_their_panel(self, page: Page, base_url: str):
        """Layout invariant, not a regression test: the sized controls stay small and
        inside their panel. The `--narrow`/`--select` rules use a flex-basis rather than
        a `width` precisely because Pico sets `width: 100%` on inputs and selects from a
        far higher-specificity selector, and a losing `width` stretches the control
        across the row and pushes the Save button past the panel edge."""
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)

        # Checked one pane at a time: a hidden pane's controls report a zero-size
        # rect, which would misread as "narrow" rather than "not on screen".
        sections_with_narrow_controls = [
            "tab-settings-mqtt",
            "tab-settings-retention",
            "tab-settings-timelapse",
            "tab-settings-models",
            "tab-settings-dedup",
            "tab-settings-interface",
        ]
        all_measured = []
        for tab in sections_with_narrow_controls:
            _open_settings_section(page, tab)
            measured = page.evaluate(f"""() => [...document.querySelectorAll(
                    '#{tab} .settings-server-input--narrow, #{tab} .settings-server-input--select'
                )].map(el => {{
                    const panel = el.closest('.panel-box').getBoundingClientRect();
                    const r = el.getBoundingClientRect();
                    return {{ cls: el.className, width: r.width, overflow: r.right - panel.right,
                             panelWidth: panel.width }};
                }})""")
            all_measured.extend(measured)

        assert all_measured, "expected narrow controls somewhere across the sections checked"
        for m in all_measured:
            assert m["overflow"] <= 0, f"{m['cls']} overflows its panel by {m['overflow']:.0f}px"
            # A control that lost the width fight fills the row; a sized one is a
            # small fraction of the panel.
            assert m["width"] < m["panelWidth"] * 0.5, f"{m['cls']} is not sized narrow: {m['width']:.0f}px"


# ---------------------------------------------------------------------------
# 3c. MQTT broker panel
# ---------------------------------------------------------------------------


def _mock_test_mqtt(page: Page, success=True, message="Connected to broker.local:1883"):
    calls = []

    def handle(route: Route):
        calls.append(json.loads(route.request.post_data))
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"success": success, "message": message}),
        )

    page.route("**/api/test-mqtt", handle)
    return calls


def _mqtt_field(page: Page, testid: str):
    return page.locator(f"#settings-section-mqtt [data-testid='{testid}']")


def _mqtt_status(page: Page):
    return page.locator("#settings-section-mqtt [data-testid='settings-mqtt-actions'] .settings-server-status")


class TestMqttSettings:
    def test_empty_config_renders_blank_fields_with_defaults(self, page: Page, base_url: str):
        """With no stored broker, the panel shows the MqttSettings defaults and says so."""
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-mqtt")

        expect(_mqtt_field(page, "settings-mqtt-host")).to_have_value("")
        expect(_mqtt_field(page, "settings-mqtt-port")).to_have_value("1883")
        expect(_mqtt_field(page, "settings-mqtt-prefix")).to_have_value("purrview")
        expect(_mqtt_field(page, "settings-mqtt-client-id")).to_have_value("purrview")
        expect(_mqtt_field(page, "settings-mqtt-tls")).not_to_be_checked()
        expect(_mqtt_status(page)).to_have_text("No broker configured.")

    def test_stored_json_config_populates_the_form(self, page: Page, base_url: str):
        """The key is stored as a JSON string, since the settings table holds strings."""
        _mock_server_config(page)
        _mock_settings_get(
            page,
            {
                "mqtt_broker": json.dumps(
                    {
                        "host": "broker.local",
                        "port": 8883,
                        "prefix": "home/cams",
                        "username": "user1",
                        "password": "secret",
                        "tls": True,
                        "client_id": "ww-garage",
                    }
                )
            },
        )
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-mqtt")

        expect(_mqtt_field(page, "settings-mqtt-host")).to_have_value("broker.local")
        expect(_mqtt_field(page, "settings-mqtt-port")).to_have_value("8883")
        expect(_mqtt_field(page, "settings-mqtt-prefix")).to_have_value("home/cams")
        expect(_mqtt_field(page, "settings-mqtt-username")).to_have_value("user1")
        expect(_mqtt_field(page, "settings-mqtt-password")).to_have_value("secret")
        expect(_mqtt_field(page, "settings-mqtt-tls")).to_be_checked()
        expect(_mqtt_field(page, "settings-mqtt-client-id")).to_have_value("ww-garage")
        expect(_mqtt_status(page)).to_have_text("Configured: broker.local:8883")

    def test_unparseable_stored_value_falls_back_to_defaults(self, page: Page, base_url: str):
        """A hand-edited row must not break the whole Settings tab render."""
        _mock_server_config(page)
        _mock_settings_get(page, {"mqtt_broker": "not json"})
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-mqtt")

        expect(_mqtt_field(page, "settings-mqtt-host")).to_have_value("")
        # Every other section still built fine — a hand-edited row must not break the
        # whole tab's render. All panes build up front regardless of which is visible.
        expect(page.locator("#tab-settings .config-rail-body .panel-box.settings-panel")).to_have_count(10)

    def test_save_writes_json_under_mqtt_broker(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-mqtt")

        put_calls = []

        def handle_put(route: Route):
            if route.request.method == "PUT":
                put_calls.append((route.request.url, json.loads(route.request.post_data)))
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

        page.route(lambda url: "/api/settings/" in url, handle_put)

        _mqtt_field(page, "settings-mqtt-host").fill("broker.local")
        _mqtt_field(page, "settings-mqtt-port").fill("8883")
        _mqtt_field(page, "settings-mqtt-username").fill("user1")
        _mqtt_field(page, "settings-mqtt-password").fill("secret")
        _mqtt_field(page, "settings-mqtt-tls").check()
        page.locator("[data-testid='settings-mqtt-save']").click()

        wait_until(page, lambda: _put_for(put_calls, "mqtt_broker"), "Expected PUT to /api/settings/mqtt_broker")
        _, body = _put_for(put_calls, "mqtt_broker")[0]
        saved = json.loads(body["value"])
        assert saved["host"] == "broker.local"
        assert saved["port"] == 8883
        assert saved["username"] == "user1"
        assert saved["password"] == "secret"
        assert saved["tls"] is True
        assert saved["prefix"] == "purrview"
        assert saved["client_id"] == "purrview"
        expect(_mqtt_status(page)).to_contain_text("Saved: broker.local:8883")

    def test_save_with_blank_host_disables_mqtt(self, page: Page, base_url: str):
        """A blank host stores an empty string — the watcher's "no broker" signal —
        rather than JSON that would parse into a localhost connection attempt."""
        _mock_server_config(page)
        _mock_settings_get(page, {"mqtt_broker": json.dumps({"host": "broker.local"})})
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-mqtt")

        put_calls = []

        def handle_put(route: Route):
            if route.request.method == "PUT":
                put_calls.append((route.request.url, json.loads(route.request.post_data)))
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

        page.route(lambda url: "/api/settings/" in url, handle_put)

        _mqtt_field(page, "settings-mqtt-host").fill("")
        page.locator("[data-testid='settings-mqtt-save']").click()

        wait_until(page, lambda: _put_for(put_calls, "mqtt_broker"), "Expected PUT to /api/settings/mqtt_broker")
        assert _put_for(put_calls, "mqtt_broker")[0][1] == {"value": ""}
        expect(_mqtt_status(page)).to_have_text("Saved: MQTT publishing disabled.")

    def test_test_button_probes_the_form_values(self, page: Page, base_url: str):
        """Test probes what is typed, not what is saved — otherwise it could only ever
        confirm settings already committed."""
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-mqtt")
        calls = _mock_test_mqtt(page, success=True, message="Connected to broker.local:1883")

        _mqtt_field(page, "settings-mqtt-host").fill("broker.local")
        page.locator("[data-testid='settings-mqtt-test']").click()

        wait_until(page, lambda: calls, "Expected POST to /api/test-mqtt")
        assert calls[0] == {
            "host": "broker.local",
            "port": 1883,
            "username": None,
            "password": None,
            "tls": False,
        }
        expect(_mqtt_status(page)).to_have_text("✓ Connected to broker.local:1883")
        expect(_mqtt_status(page)).to_have_attribute("data-state", "ok")

    def test_test_button_reports_failure(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-mqtt")
        _mock_test_mqtt(page, success=False, message="bad username or password")

        _mqtt_field(page, "settings-mqtt-host").fill("broker.local")
        page.locator("[data-testid='settings-mqtt-test']").click()

        expect(_mqtt_status(page)).to_have_text("✗ bad username or password")
        expect(_mqtt_status(page)).to_have_attribute("data-state", "error")

    def test_test_without_host_does_not_call_the_api(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-mqtt")
        calls = _mock_test_mqtt(page)

        page.locator("[data-testid='settings-mqtt-test']").click()
        page.wait_for_timeout(300)

        assert calls == []
        expect(_mqtt_status(page)).to_have_text("Enter a broker host first.")


# ---------------------------------------------------------------------------
# 4. Hotkeys collapsible panel
# ---------------------------------------------------------------------------


class TestHotkeysPanel:
    """Reset-to-defaults semantics (resetBindings) are covered by
    test_hotkey_bindings.py; this file only checks the panel is reachable and
    renders as its own rail destination — it no longer collapses behind a
    chevron, since Configuration's rail already hides everything but the
    selected section."""

    def test_selecting_the_section_shows_the_hotkey_grid_immediately(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)

        expect(page.locator("#tab-settings-hotkeys")).to_be_hidden()
        _open_settings_section(page, "tab-settings-hotkeys")

        # Rows flow into a grid rather than one row-per-line column — with one row
        # per label *and* per visible tag, a single column runs arbitrarily tall.
        grid = page.locator("#tab-settings-hotkeys .hotkey-grid").first
        expect(grid).to_be_visible()
        expect(grid).to_have_css("display", "grid")
        assert grid.locator(".hotkey-row").count() > 0

    def test_reset_button_stays_on_the_hotkeys_section(self, page: Page, base_url: str):
        """Clicking 'Reset to defaults' must not navigate away from the section."""
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-hotkeys")

        page.locator("#tab-settings-hotkeys button", has_text="Reset to defaults").click()

        expect(page.locator("#tab-settings-hotkeys")).to_be_visible()
        expect(page.locator("#tab-settings-hotkeys .hotkey-grid").first).to_be_visible()


# ---------------------------------------------------------------------------
# File browser mocking helpers
# ---------------------------------------------------------------------------

ROOT_ENTRIES_PAGE_1 = [
    {"name": "subdir", "path": "subdir", "isDir": True, "uuid": None},
    {"name": "a.jpg", "path": "a.jpg", "isDir": False, "uuid": "img-uuid-a", "bboxCount": 2},
]
ROOT_ENTRIES_PAGE_2 = [
    {"name": "b.jpg", "path": "b.jpg", "isDir": False, "uuid": None},
]
SUBDIR_ENTRIES = [
    {"name": "c.png", "path": "subdir/c.png", "isDir": False, "uuid": "img-uuid-c", "bboxCount": 0},
]


def _mock_image_list(page: Page, pages_by_path: dict):
    """Intercept GET /api/images/list and serve the page (with hasMore flag) for the
    requested `path` query param. `pages_by_path` maps path -> list[{entries, has_more}]."""
    call_counts = {}

    def handle(route: Route):
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(route.request.url).query)
        path = qs.get("path", ["/"])[0]
        offset = int(qs.get("offset", ["0"])[0])
        plan = pages_by_path.get(path, [])
        idx = call_counts.get((path, offset), None)
        # Pick the page whose declared offset matches; fall back to last.
        page_entry = next(
            (p for p in plan if p.get("offset", 0) == offset), plan[-1] if plan else {"entries": [], "has_more": False}
        )
        call_counts[(path, offset)] = True
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "files": page_entry["entries"], "hasMore": page_entry["has_more"]}),
        )

    page.route(lambda url: "/api/images/list" in url, handle)


# ---------------------------------------------------------------------------
# 5. File browser entry rendering
# ---------------------------------------------------------------------------


class TestFileBrowserRendering:
    def test_entries_render_with_names_and_db_badges(self, page: Page, base_url: str):
        """Listed entries render as rows with name + folder/image icon, and DB-status badges."""
        _mock_image_list(page, {"/": [{"offset": 0, "entries": ROOT_ENTRIES_PAGE_1, "has_more": False}]})
        _navigate_to_files(page, base_url)

        dir_row = page.locator("#tab-files .fb-entry", has_text="subdir")
        file_row = page.locator("#tab-files .fb-entry", has_text="a.jpg")

        expect(dir_row).to_be_visible()
        expect(dir_row.locator(".fb-entry__icon")).to_have_text("📁")
        expect(file_row).to_be_visible()
        expect(file_row.locator(".fb-entry__icon")).to_have_text("🖼️")

        badge = file_row.locator(".fb-entry__db-badge")
        expect(badge).to_have_text("2 boxes")
        expect(badge).to_have_attribute("title", "In database — 2 bounding boxes")


# ---------------------------------------------------------------------------
# 6. Pagination
# ---------------------------------------------------------------------------


class TestFileBrowserPagination:
    def test_load_more_appends_next_page(self, page: Page, base_url: str):
        """Clicking 'Load more…' fetches the next page and appends its entries."""
        _mock_image_list(
            page,
            {
                "/": [
                    {"offset": 0, "entries": ROOT_ENTRIES_PAGE_1, "has_more": True},
                    {"offset": len(ROOT_ENTRIES_PAGE_1), "entries": ROOT_ENTRIES_PAGE_2, "has_more": False},
                ]
            },
        )
        _navigate_to_files(page, base_url)

        expect(page.locator("#tab-files .fb-entry", has_text="a.jpg")).to_be_visible()
        load_more = page.locator("#tab-files .fb-load-more-btn")
        expect(load_more).to_be_visible()
        expect(load_more).to_have_text("Load more…")

        load_more.click()
        page.wait_for_timeout(500)

        expect(page.locator("#tab-files .fb-entry", has_text="b.jpg")).to_be_visible()
        # All of page 1's entries remain (appended, not replaced).
        expect(page.locator("#tab-files .fb-entry", has_text="a.jpg")).to_be_visible()
        expect(page.locator("#tab-files .fb-entry", has_text="subdir")).to_be_visible()
        # No more pages -> the Load more control disappears.
        expect(page.locator("#tab-files .fb-load-more-btn")).to_have_count(0)


# ---------------------------------------------------------------------------
# 7. Directory navigation
# ---------------------------------------------------------------------------


class TestFileBrowserNavigation:
    def test_clicking_directory_navigates_into_it(self, page: Page, base_url: str):
        """Clicking a directory entry re-fetches the listing for that path and updates the breadcrumb."""
        _mock_image_list(
            page,
            {
                "/": [{"offset": 0, "entries": ROOT_ENTRIES_PAGE_1, "has_more": False}],
                "subdir": [{"offset": 0, "entries": SUBDIR_ENTRIES, "has_more": False}],
            },
        )
        _navigate_to_files(page, base_url)

        page.locator("#tab-files .fb-entry", has_text="subdir").click()
        page.wait_for_timeout(500)

        expect(page.locator("#tab-files .fb-entry", has_text="c.png")).to_be_visible()
        expect(page.locator("#tab-files .fb-breadcrumb")).to_contain_text("subdir")
        # ".." nav entry appears once we're below root.
        expect(page.locator("#tab-files .fb-entry--nav .fb-entry__name", has_text="..")).to_be_visible()

    def test_clicking_file_loads_it_onto_the_canvas(self, page: Page, base_url: str):
        """Clicking a file entry publishes LOAD_IMAGE_ONTO_CANVAS, which fetches that file."""
        _mock_image_list(page, {"/": [{"offset": 0, "entries": ROOT_ENTRIES_PAGE_1, "has_more": False}]})

        file_calls = []

        def handle_file(route: Route):
            file_calls.append(route.request.url)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "imageBase64": "",
                        "mimeType": "image/jpeg",
                        "image": {"uuid": "img-uuid-a", "filename": "a.jpg", "bboxes": []},
                    }
                ),
            )

        page.route(lambda url: url.split("?")[0].endswith("/file") and "/api/images/" in url, handle_file)

        _navigate_to_files(page, base_url)
        page.locator("#tab-files .fb-entry", has_text="a.jpg").click()
        wait_until(
            page, lambda: file_calls, "Expected a fetch to /api/images/{path}/file after clicking the file entry"
        )
        assert "a.jpg" in file_calls[0]


# ---------------------------------------------------------------------------
# 8. Hover preview (ui/fileEntryHoverPreview.js)
# ---------------------------------------------------------------------------

TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk" "+A8AAQUBAScY42YAAAAASUVORK5CYII="


class TestFileBrowserHoverPreview:
    def test_hover_shows_annotated_thumbnail_for_db_image(self, page: Page, base_url: str):
        """Hovering a filename for an image registered in the DB pops up a thumbnail
        with bboxes drawn from /api/images/metadata/get."""
        _mock_image_list(page, {"/": [{"offset": 0, "entries": ROOT_ENTRIES_PAGE_1, "has_more": False}]})

        png_bytes = base64.b64decode(TINY_PNG_B64)
        page.route(
            "**/thumbnail?*",
            lambda route: route.fulfill(status=200, content_type="image/png", body=png_bytes),
        )

        metadata_calls = []

        def handle_metadata(route: Route):
            body = json.loads(route.request.post_data or "{}")
            metadata_calls.append(body.get("imagePath"))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "image": {
                            "uuid": "img-uuid-a",
                            "filename": "a.jpg",
                            "bboxes": [
                                {
                                    "uuid": "bbox-1",
                                    "x": 0.1,
                                    "y": 0.1,
                                    "width": 0.2,
                                    "height": 0.2,
                                    "label": {"uuid": "label-1", "name": "Cat", "color": "#ff0000"},
                                    "tags": [],
                                },
                            ],
                        },
                    }
                ),
            )

        page.route("**/api/images/metadata/get", handle_metadata)

        _navigate_to_files(page, base_url)

        name = page.locator("#tab-files .fb-entry", has_text="a.jpg").locator(".fb-entry__name")
        name.hover()
        page.wait_for_selector(".fb-hover-popover.visible")

        popover = page.locator(".fb-hover-popover")
        expect(popover.locator("img.fb-popover-img")).to_be_visible()
        expect(popover.locator("canvas.fb-popover-canvas")).to_be_visible()
        wait_until(page, lambda: metadata_calls, "Expected a metadata fetch for the hovered file")
        assert metadata_calls[0] == "a.jpg"

    def test_hover_does_not_attach_preview_to_directories(self, page: Page, base_url: str):
        """Directory entries get no hover preview — the popover is never created."""
        _mock_image_list(page, {"/": [{"offset": 0, "entries": ROOT_ENTRIES_PAGE_1, "has_more": False}]})
        _navigate_to_files(page, base_url)

        dir_name = page.locator("#tab-files .fb-entry", has_text="subdir").locator(".fb-entry__name")
        dir_name.hover()
        page.wait_for_timeout(600)

        expect(page.locator(".fb-hover-popover")).to_have_count(0)


# ---------------------------------------------------------------------------
# 8. Grid/list view toggle (ui/entryView.js, gated by `showThumbnails`)
# ---------------------------------------------------------------------------
# `showThumbnails` is never enabled on the Files tab in the live app today, so
# the toggle is unreachable through normal navigation — instantiate FileBrowser
# directly with `showThumbnails: true` to exercise entryView's grid/list paths.


class TestFileBrowserViewToggle:
    def test_toggle_switches_views_and_persists(self, page: Page, base_url: str):
        """With `showThumbnails: true`, the grid/list toggle (shared `entryView`)
        re-renders entries as `.fb-grid-cell`/`.fb-entry`, and the chosen view is
        remembered across instances via localStorage `fb-view` — mirroring the
        review page's `review-view` persistence."""
        _mock_image_list(page, {"/": [{"offset": 0, "entries": ROOT_ENTRIES_PAGE_1, "has_more": False}]})
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.evaluate("() => localStorage.removeItem('fb-view')")

        page.evaluate("""async () => {
            const { FileBrowser } = await import('/app-static/js/ui/fileBrowser.js');
            const container = document.createElement('div');
            container.id = 'fb-toggle-test';
            document.body.appendChild(container);
            window.__fb = new FileBrowser({ container, showThumbnails: true });
            await window.__fb.init();
        }""")

        try:
            root = page.locator("#fb-toggle-test")
            expect(root.locator(".fb-entry", has_text="a.jpg")).to_be_visible()
            expect(root.locator(".fb-grid-cell")).to_have_count(0)

            root.locator(".fb-view-toggle button[title='Grid view']").click()

            # The two `expect`s below retry on their own; only the localStorage read does
            # not, so it gets its own poll rather than riding on a 150ms sleep.
            expect(root.locator(".fb-grid-cell", has_text="a.jpg")).to_be_visible()
            expect(root.locator(".fb-entry")).to_have_count(0)
            wait_until(
                page,
                lambda: page.evaluate("() => localStorage.getItem('fb-view')") == "grid",
                lambda: "fb-view was never persisted as 'grid', got "
                + repr(page.evaluate("() => localStorage.getItem('fb-view')")),
            )

            # A fresh instance (simulating reload) should pick up the persisted choice
            page.evaluate("""async () => {
                window.__fb.container.innerHTML = '';
                const { FileBrowser } = await import('/app-static/js/ui/fileBrowser.js');
                window.__fb2 = new FileBrowser({ container: window.__fb.container, showThumbnails: true });
                await window.__fb2.init();
            }""")
            page.wait_for_timeout(150)
            expect(root.locator(".fb-grid-cell", has_text="a.jpg")).to_be_visible()
            expect(root.locator(".fb-entry")).to_have_count(0)
        finally:
            page.evaluate("""() => {
                window.__fb?.container?.remove();
                delete window.__fb;
                delete window.__fb2;
            }""")


# ---------------------------------------------------------------------------
# 3d. Recording retention panel
# ---------------------------------------------------------------------------


def _retention_field(page: Page):
    return page.locator("#settings-section-retention [data-testid='settings-retention-days']")


def _retention_size_field(page: Page):
    return page.locator("#settings-section-retention [data-testid='settings-retention-max-mb']")


def _retention_status(page: Page, index: int = 0):
    """The panel now has a row per bound, so a status line has to be picked by row:
    0 is the age bound, 1 is the size bound."""
    return page.locator("#settings-section-retention .settings-server-status").nth(index)


class TestRecordingRetention:
    """Event clips and card art are the only unbounded writer in the app, so the window
    that bounds them has to be visible and changeable rather than a constant in a file."""

    def test_defaults_to_thirty_days_when_unset(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-retention")

        expect(_retention_field(page)).to_have_value("30")
        expect(_retention_status(page)).to_contain_text("30 days")

    def test_a_stored_value_populates_the_field(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page, {"recording_retention_days": "7"})
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-retention")

        expect(_retention_field(page)).to_have_value("7")
        expect(_retention_status(page)).to_contain_text("7 days")

    def test_zero_is_shown_as_keeping_everything(self, page: Page, base_url: str):
        """`Number('')` is a finite 0, so "unset" and "keep forever" must not collapse."""
        _mock_server_config(page)
        _mock_settings_get(page, {"recording_retention_days": "0"})
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-retention")

        expect(_retention_field(page)).to_have_value("0")
        expect(_retention_status(page)).to_contain_text("No age limit")

    def test_an_unreadable_stored_value_falls_back_to_the_default(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page, {"recording_retention_days": "forever"})
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-retention")

        expect(_retention_field(page)).to_have_value("30")

    def test_save_writes_the_day_count(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-retention")

        put_calls = []

        def handle_put(route: Route):
            if route.request.method == "PUT":
                put_calls.append((route.request.url, json.loads(route.request.post_data)))
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

        page.route(lambda url: "/api/settings/" in url, handle_put)

        _retention_field(page).fill("14")
        page.locator("[data-testid='settings-retention-days-save']").click()

        wait_until(
            page,
            lambda: _put_for(put_calls, "recording_retention_days"),
            "Expected PUT to /api/settings/recording_retention_days",
        )
        _, body = _put_for(put_calls, "recording_retention_days")[0]
        assert body["value"] == "14"

    def test_a_negative_value_is_refused_rather_than_saved(self, page: Page, base_url: str):
        """Negative days would silently read as "keep everything" in the backend."""
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-retention")

        put_calls = []
        page.route(
            lambda url: "/api/settings/" in url,
            lambda route: (
                put_calls.append(route.request.url) if route.request.method == "PUT" else None,
                route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"})),
            )[-1],
        )

        _retention_field(page).fill("-5")
        page.locator("[data-testid='settings-retention-days-save']").click()

        expect(_retention_status(page)).to_contain_text("whole number of days")
        assert not put_calls

    def test_the_size_bound_defaults_to_ten_gigabytes(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-retention")

        expect(_retention_size_field(page)).to_have_value("10240")
        expect(_retention_status(page, 1)).to_contain_text("10240 MB")

    def test_a_stored_size_bound_populates_the_field(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page, {"recording_max_total_mb": "512"})
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-retention")

        expect(_retention_size_field(page)).to_have_value("512")
        expect(_retention_status(page, 1)).to_contain_text("512 MB")

    def test_the_two_bounds_are_independent(self, page: Page, base_url: str):
        """Switching one off must not read as switching both off — that is the whole
        point of having two, and the copy has to say so."""
        _mock_server_config(page)
        _mock_settings_get(page, {"recording_retention_days": "0", "recording_max_total_mb": "2048"})
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-retention")

        expect(_retention_status(page, 0)).to_contain_text("No age limit")
        expect(_retention_status(page, 0)).to_contain_text("size limit")
        expect(_retention_status(page, 1)).to_contain_text("2048 MB")

    def test_saving_the_size_bound_writes_its_own_key(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-retention")

        put_calls = []

        def handle_put(route: Route):
            if route.request.method == "PUT":
                put_calls.append((route.request.url, json.loads(route.request.post_data)))
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

        page.route(lambda url: "/api/settings/" in url, handle_put)

        _retention_size_field(page).fill("4096")
        page.locator("[data-testid='settings-retention-max-mb-save']").click()

        wait_until(
            page,
            lambda: _put_for(put_calls, "recording_max_total_mb"),
            "Expected PUT to /api/settings/recording_max_total_mb",
        )
        _, body = _put_for(put_calls, "recording_max_total_mb")[0]
        assert body["value"] == "4096"

    def test_the_unmanaged_folder_caveat_is_stated(self, page: Page, base_url: str):
        """A custom clip `output_dir` is neither counted nor swept, which is surprising
        enough that the panel has to say it rather than leave it to be discovered."""
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-retention")

        expect(page.locator("[data-testid='settings-retention-caveat']")).to_contain_text("custom output folder")


# ---------------------------------------------------------------------------
# 3e. Timelapse retention panel
# ---------------------------------------------------------------------------


def _timelapse_field(page: Page):
    return page.locator("#settings-section-timelapse-retention [data-testid='settings-timelapse-days']")


def _timelapse_size_field(page: Page):
    return page.locator("#settings-section-timelapse-retention [data-testid='settings-timelapse-max-mb']")


def _timelapse_status(page: Page, index: int = 0):
    return page.locator("#settings-section-timelapse-retention .settings-server-status").nth(index)


class TestTimelapseRetention:
    """Timelapse frames get their own bounds, in their own panel.

    Not two more rows under Recording Retention: a clip is kept because something
    happened and a month of them is a modest folder, while a timelapse keeps shooting
    when nothing does — ~2,900 frames per camera per day. One pair of numbers cannot
    serve both without the bulkier material silently evicting the other.
    """

    def test_it_is_a_rail_destination_of_its_own(self, page: Page, base_url: str):
        """Its own rail item, distinct from Recording Retention — selecting one hides
        the other, so the two panels' values are never both on screen to conflate."""
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)

        _open_settings_section(page, "tab-settings-timelapse")
        expect(page.locator("#settings-section-timelapse-retention")).to_be_visible()
        expect(page.locator("#settings-section-retention")).to_be_hidden()

        _open_settings_section(page, "tab-settings-retention")
        expect(page.locator("#settings-section-retention")).to_be_visible()
        expect(page.locator("#settings-section-timelapse-retention")).to_be_hidden()

    def test_defaults_when_unset(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-timelapse")

        expect(_timelapse_field(page)).to_have_value("14")
        expect(_timelapse_size_field(page)).to_have_value("20480")

    def test_stored_values_populate_the_fields(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page, {"timelapse_retention_days": "3", "timelapse_max_total_mb": "512"})
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-timelapse")

        expect(_timelapse_field(page)).to_have_value("3")
        expect(_timelapse_status(page, 0)).to_contain_text("3 days")
        expect(_timelapse_size_field(page)).to_have_value("512")
        expect(_timelapse_status(page, 1)).to_contain_text("512 MB")

    def test_it_does_not_read_the_recording_keys(self, page: Page, base_url: str):
        """Sharing a key would make one panel silently rewrite the other's bound."""
        _mock_server_config(page)
        _mock_settings_get(page, {"recording_retention_days": "99", "recording_max_total_mb": "99"})
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-timelapse")

        expect(_timelapse_field(page)).to_have_value("14")
        expect(_timelapse_size_field(page)).to_have_value("20480")

    def test_zero_is_shown_as_keeping_everything(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page, {"timelapse_retention_days": "0"})
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-timelapse")

        expect(_timelapse_status(page, 0)).to_contain_text("No age limit")

    def test_save_writes_the_timelapse_keys(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-timelapse")

        put_calls = []

        def handle_put(route: Route):
            if route.request.method == "PUT":
                put_calls.append((route.request.url, json.loads(route.request.post_data)))
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

        page.route(lambda url: "/api/settings/" in url, handle_put)

        _timelapse_field(page).fill("7")
        page.locator("[data-testid='settings-timelapse-days-save']").click()
        wait_until(
            page,
            lambda: _put_for(put_calls, "timelapse_retention_days"),
            "Expected PUT to /api/settings/timelapse_retention_days",
        )
        assert _put_for(put_calls, "timelapse_retention_days")[0][1]["value"] == "7"

        _timelapse_size_field(page).fill("4096")
        page.locator("[data-testid='settings-timelapse-max-mb-save']").click()
        wait_until(
            page,
            lambda: _put_for(put_calls, "timelapse_max_total_mb"),
            "Expected PUT to /api/settings/timelapse_max_total_mb",
        )
        assert _put_for(put_calls, "timelapse_max_total_mb")[0][1]["value"] == "4096"

    def test_the_sweep_scope_is_stated(self, page: Page, base_url: str):
        """These folders live under the image datasets, beside a user's annotated images.
        What the sweep will and will not touch is the thing they need to know."""
        _mock_server_config(page)
        _mock_settings_get(page)
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-timelapse")

        caveat = page.locator("[data-testid='settings-timelapse-caveat']")
        expect(caveat).to_contain_text("JSON sidecar")
        expect(caveat).to_contain_text("YYYY-MM-DD")


class TestModelHistorySettings:
    """The two bounds on what training leaves behind.

    Both rows are built from the same helper, so the risk they carry is that one
    saves under the other's key — which would look like a working input that
    silently changes nothing.
    """

    def test_each_row_saves_under_its_own_key(self, page: Page, base_url: str):
        _mock_server_config(page)
        _mock_settings_get(page, {"model_versions_keep": "3", "training_artifacts_keep": "3"})
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-models")

        put_calls = []

        def handle_put(route: Route):
            if route.request.method == "PUT":
                put_calls.append((route.request.url, json.loads(route.request.post_data)))
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

        page.route(lambda url: "/api/settings/" in url, handle_put)

        page.get_by_test_id("settings-model-versions-keep").fill("5")
        page.get_by_test_id("settings-model-versions-keep-save").click()
        wait_until(
            page, lambda: _put_for(put_calls, "model_versions_keep"), "Expected PUT for the model-versions bound"
        )

        page.get_by_test_id("settings-training-artifacts-keep").fill("1")
        page.get_by_test_id("settings-training-artifacts-keep-save").click()
        wait_until(page, lambda: _put_for(put_calls, "training_artifacts_keep"), "Expected PUT for the artifacts bound")

        assert _put_for(put_calls, "model_versions_keep")[0][1] == {"value": "5"}
        assert _put_for(put_calls, "training_artifacts_keep")[0][1] == {"value": "1"}

    def test_zero_versions_is_described_as_no_rollback(self, page: Page, base_url: str):
        """0 is a legal value with real consequences, so the row has to say what it means."""
        _mock_server_config(page)
        _mock_settings_get(page, {"model_versions_keep": "0"})
        _navigate_to_settings(page, base_url)
        _open_settings_section(page, "tab-settings-models")

        row = page.locator(".settings-server-row").filter(has=page.get_by_test_id("settings-model-versions-keep"))
        expect(row.locator(".settings-server-status")).to_contain_text("No history")
