"""
Tests for the Sources configuration UI.

Exercises the renderSourceManager() component rendered inside
Configuration > Source Management > #source-management-box:
  - DOM structure (list and create-form sections visible)
  - Source list loading (sources appear as rows with Edit/Test buttons)
  - Provider dropdown populated from /api/sources/image-providers/list
  - Full create flow: select provider → schema fields appear → fill details → Save → API called
  - Delete flow (confirm dialog; API called or suppressed)

Run with:
    pytest tests/playwright/test_configuration_sources.py -p no:xdist -v
"""

import json

import pytest
from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

FAKE_SOURCE = {
    "uuid": "dddddddd-0000-0000-0000-000000000001",
    "name": "My Camera",
    "typename": "UrlImageProvider",
    "params": {"url": "http://cam.example.com/snap.jpg"},
}

URL_PROVIDER_SCHEMA = {
    "url": {
        "type": "string",
        "label": "URL",
        "required": True,
        "description": "URL of the image",
    },
    "user": {
        "type": "string",
        "label": "Username",
        "required": False,
        "description": "Username for authentication",
    },
}

URL_PROVIDER_SCHEMA_WITH_INTERVAL = {
    **URL_PROVIDER_SCHEMA,
    "min_capture_interval": {
        "type": "float",
        "label": "Min Capture Interval (s)",
        "required": False,
        "default": 0.0,
        "description": "Minimum seconds between frames.",
    },
}

# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------


def _navigate_to_sources(page: Page, base_url: str):
    """Open the app and activate Configuration > Source Management."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-operations']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-operations .workspace-tab-button[data-tab='tab-source-management']").click()
    page.wait_for_load_state("networkidle")


def _mock_sources(page: Page, sources: dict | None = None):
    """Intercept /api/sources/get to return the given sources dict (uuid → source)."""
    sources = sources or {}
    page.route(
        "**/api/sources/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "sources": sources}),
        ),
    )


def _mock_providers(page: Page, providers: list | None = None):
    """Intercept /api/sources/image-providers/list to return the given provider names."""
    providers = providers or ["UrlImageProvider"]
    page.route(
        "**/api/sources/image-providers/list",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "imageProviders": providers}),
        ),
    )


def _mock_provider_schema(page: Page, schema: dict | None = None):
    """Intercept /api/sources/image-providers/schema to return the given schema."""
    schema = schema or URL_PROVIDER_SCHEMA
    page.route(
        "**/api/sources/image-providers/schema",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "imageProviderSchema": schema}),
        ),
    )


# ---------------------------------------------------------------------------
# DOM structure
# ---------------------------------------------------------------------------


class TestSourcePaneStructure:
    def test_pane_structure(self, page: Page, base_url: str):
        """The sources pane renders its container and key headings in one navigation."""
        _navigate_to_sources(page, base_url)
        expect(page.locator("#source-management-box")).to_be_visible()
        page.wait_for_selector("#source-management-box h3", timeout=4000)
        headers = page.locator("#source-management-box h3").all_text_contents()
        assert any("Sources" in h for h in headers), f"No 'Sources' header found. Got: {headers}"
        assert any("Create" in h for h in headers), f"No 'Create' header found. Got: {headers}"

    def test_no_js_errors_on_sources_tab(self, page: Page, base_url: str):
        """No console errors while loading the Sources pane."""
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        _navigate_to_sources(page, base_url)
        page.wait_for_timeout(1000)
        uncaught = [e for e in errors if "TypeError" in e or "Uncaught" in e]
        assert uncaught == [], f"Uncaught JS errors: {uncaught}"


# ---------------------------------------------------------------------------
# Source list loading
# ---------------------------------------------------------------------------


class TestSourceListLoading:
    def test_existing_sources_render_as_rows(self, page: Page, base_url: str):
        """Sources returned from /api/sources/get are rendered in the list.

        Note: SourceConfigField.create() calls fetchImageProviderList() internally,
        so the provider list endpoint must be mocked for existing source rows to render.
        """
        _mock_sources(page, {FAKE_SOURCE["uuid"]: FAKE_SOURCE})
        _mock_providers(page, ["UrlImageProvider"])
        _mock_provider_schema(page)
        _navigate_to_sources(page, base_url)
        # Source rows render asynchronously; give them time to settle
        page.wait_for_timeout(1200)
        content = page.locator("#source-management-box").inner_text()
        assert FAKE_SOURCE["name"] in content, f"Source name not found in: {content!r}"

    def test_existing_source_has_test_button(self, page: Page, base_url: str):
        """Each source row shows a 'Test Source' button."""
        _mock_sources(page, {FAKE_SOURCE["uuid"]: FAKE_SOURCE})
        _mock_providers(page, ["UrlImageProvider"])
        _mock_provider_schema(page)
        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box button[title='Test Source']", timeout=5000)
        expect(page.locator("#source-management-box button[title='Test Source']").first).to_be_visible()

    def test_existing_source_has_delete_button(self, page: Page, base_url: str):
        """Each source row shows a Delete button."""
        _mock_sources(page, {FAKE_SOURCE["uuid"]: FAKE_SOURCE})
        _mock_providers(page, ["UrlImageProvider"])
        _mock_provider_schema(page)
        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box button[title='Delete']", timeout=5000)
        expect(page.locator("#source-management-box button[title='Delete']").first).to_be_visible()


# ---------------------------------------------------------------------------
# Provider dropdown
# ---------------------------------------------------------------------------


class TestProviderDropdown:
    def test_provider_dropdown_populated_from_api(self, page: Page, base_url: str):
        """The provider <select> lists the providers returned by /api/sources/image-providers/list."""
        providers = ["UrlImageProvider", "FfmpegImageProvider"]
        _mock_providers(page, providers)
        _mock_sources(page, {})
        _navigate_to_sources(page, base_url)

        # Wait for the create-section dropdown to appear
        page.wait_for_selector("#source-management-box select", timeout=4000)
        options = page.locator("#source-management-box select").nth(-1).locator("option").all_text_contents()
        for provider in providers:
            assert provider in options, f"Provider '{provider}' not in dropdown options {options}"

    def test_selecting_provider_fetches_schema(self, page: Page, base_url: str):
        """Selecting a provider from the dropdown triggers a /api/sources/image-providers/schema call."""
        schema_calls: list[str] = []

        def handle_schema(route: Route):
            body = json.loads(route.request.post_data)
            schema_calls.append(body.get("imageProvider", ""))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "imageProviderSchema": URL_PROVIDER_SCHEMA}),
            )

        _mock_providers(page, ["UrlImageProvider"])
        _mock_sources(page, {})
        page.route("**/api/sources/image-providers/schema", handle_schema)

        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box select", timeout=4000)

        # Select the provider in the last dropdown (the create-form one)
        page.locator("#source-management-box select").last.select_option("UrlImageProvider")
        wait_until(page, lambda: schema_calls, "Expected /api/sources/image-providers/schema to be called")
        assert "UrlImageProvider" in schema_calls[0]

    def test_schema_fields_appear_after_provider_selection(self, page: Page, base_url: str):
        """After selecting a provider, schema-driven input fields appear in the form."""
        _mock_providers(page, ["UrlImageProvider"])
        _mock_sources(page, {})
        _mock_provider_schema(page, URL_PROVIDER_SCHEMA)

        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box select", timeout=4000)

        page.locator("#source-management-box select").last.select_option("UrlImageProvider")
        page.wait_for_timeout(800)

        # The URL schema field renders a labeled input with "URL" label
        content = page.locator("#source-management-box").inner_html()
        assert "URL" in content, f"Expected 'URL' schema field label in: {content[:500]!r}"


# ---------------------------------------------------------------------------
# Full create flow
# ---------------------------------------------------------------------------


class TestCreateSource:
    def test_create_source_calls_api(self, page: Page, base_url: str):
        """Full create flow: pick provider → fill name + URL field → Save → /api/sources/create called."""
        create_calls: list[dict] = []

        def handle_create(route: Route):
            create_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "source": FAKE_SOURCE}),
            )

        _mock_providers(page, ["UrlImageProvider"])
        _mock_sources(page, {})
        _mock_provider_schema(page, URL_PROVIDER_SCHEMA)
        page.route("**/api/sources/create", handle_create)

        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box select", timeout=4000)

        # 1. Select provider
        page.locator("#source-management-box select").last.select_option("UrlImageProvider")
        page.wait_for_timeout(800)

        # 2. Fill source name (second text input in the create area — first might be the provider name)
        name_inputs = page.locator(
            "#source-management-box input[type='text'], #source-management-box input:not([type])"
        ).all()
        # The source-name field has placeholder "Enter new source name"
        name_input = page.locator("#source-management-box input[placeholder='Enter new source name']")
        name_input.fill("My New Camera")

        # 3. Fill the URL field from the schema
        url_input = page.locator("#source-management-box input").filter(has_text="").last
        # Use placeholder or description to identify URL input; schema field description is "URL of the image"
        # Find any input that appeared after schema loaded (not the source name or provider)
        all_inputs = page.locator(
            "#source-management-box input[type='text'], #source-management-box input:not([type='checkbox']):not([placeholder='Enter new source name'])"
        ).all()
        for inp in all_inputs:
            placeholder = inp.get_attribute("placeholder") or ""
            if "url" in placeholder.lower() or "URL" in placeholder or not inp.input_value():
                inp.fill("http://camera.example.com/snap.jpg")
                break

        # 4. Click Save
        page.locator("#source-management-box button[title='Save']").first.click()
        wait_until(page, lambda: create_calls, "Expected POST to /api/sources/create")
        payload = create_calls[0]
        assert payload.get("sourceName") == "My New Camera"
        assert payload.get("imageProvider") == "UrlImageProvider"

    def test_create_source_test_button_present_in_form(self, page: Page, base_url: str):
        """The 'Test Source' button is visible in the create form alongside Save."""
        _mock_providers(page, ["UrlImageProvider"])
        _mock_sources(page, {})
        _mock_provider_schema(page, URL_PROVIDER_SCHEMA)

        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box button[title='Test Source']", timeout=4000)
        expect(page.locator("#source-management-box button[title='Test Source']").first).to_be_visible()


# ---------------------------------------------------------------------------
# Numeric ("float"/"int") schema field type coercion
# ---------------------------------------------------------------------------


class TestNumericSchemaFieldCoercion:
    """Regression tests for a bug where a schema `float` field (e.g.
    UrlImageProvider's `min_capture_interval`) was sent to the backend as a
    string instead of a number, crashing the provider with
    `TypeError: '>' not supported between instances of 'str' and 'int'`.

    Two bugs compounded: `fieldFactories.int` seeded the TextField's initial
    value with `values[fieldName] || ''`, so a falsy schema default of `0`
    silently became `''`; and `TextField.getValue()` returned the raw string
    typed into the input rather than a parsed number, so even a hand-edited
    value like "5" stayed a string all the way into the create payload.
    """

    def test_untouched_default_zero_saves_as_number_not_empty_string(self, page: Page, base_url: str):
        """A field left untouched, with a schema default of 0.0, must save
        as the number 0 — not '' (the `||` vs `??` fallback bug)."""
        create_calls: list[dict] = []

        def handle_create(route: Route):
            create_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "source": FAKE_SOURCE}),
            )

        _mock_providers(page, ["UrlImageProvider"])
        _mock_sources(page, {})
        _mock_provider_schema(page, URL_PROVIDER_SCHEMA_WITH_INTERVAL)
        page.route("**/api/sources/create", handle_create)

        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box select", timeout=4000)

        page.locator("#source-management-box select").last.select_option("UrlImageProvider")
        page.wait_for_timeout(800)

        page.locator("#source-management-box input[placeholder='Enter new source name']").fill("My New Camera")
        page.locator("#source-management-box input[type='text']").last.fill("http://camera.example.com/snap.jpg")

        page.locator("#source-management-box button[title='Save']").first.click()
        wait_until(page, lambda: create_calls, "Expected POST to /api/sources/create")
        interval = create_calls[0]["providerParams"]["min_capture_interval"]
        assert interval == 0 and isinstance(interval, (int, float)), (
            f"Expected numeric 0 for untouched default, got {interval!r} "
            "(likely the ||-vs-?? default-0-becomes-empty-string bug)"
        )

    def test_typed_value_saves_as_number_not_string(self, page: Page, base_url: str):
        """Typing a value into a `float` schema field must save as a number,
        not the raw string from the input element."""
        create_calls: list[dict] = []

        def handle_create(route: Route):
            create_calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "source": FAKE_SOURCE}),
            )

        _mock_providers(page, ["UrlImageProvider"])
        _mock_sources(page, {})
        _mock_provider_schema(page, URL_PROVIDER_SCHEMA_WITH_INTERVAL)
        page.route("**/api/sources/create", handle_create)

        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box select", timeout=4000)

        page.locator("#source-management-box select").last.select_option("UrlImageProvider")
        page.wait_for_timeout(800)

        page.locator("#source-management-box input[placeholder='Enter new source name']").fill("My New Camera")
        page.locator("#source-management-box input[type='text']").last.fill("http://camera.example.com/snap.jpg")
        page.locator("#source-management-box input[type='number']").first.fill("2.5")

        page.locator("#source-management-box button[title='Save']").first.click()
        wait_until(page, lambda: create_calls, "Expected POST to /api/sources/create")
        interval = create_calls[0]["providerParams"]["min_capture_interval"]
        assert interval == 2.5 and isinstance(
            interval, (int, float)
        ), f"Expected numeric 2.5, got {interval!r} (getValue() returning the raw input string)"


# ---------------------------------------------------------------------------
# Delete flow
# ---------------------------------------------------------------------------


class TestDeleteSource:
    def test_delete_shows_confirm_dialog(self, page: Page, base_url: str):
        """Clicking Delete on a source opens a browser confirm() dialog."""
        _mock_sources(page, {FAKE_SOURCE["uuid"]: FAKE_SOURCE})
        _mock_providers(page, ["UrlImageProvider"])
        _mock_provider_schema(page)
        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box button[title='Delete']", timeout=5000)

        dialogs: list[str] = []
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
        page.locator("#source-management-box button[title='Delete']").first.click()
        wait_until(page, lambda: dialogs, "Expected a confirm() dialog")
        assert FAKE_SOURCE["name"] in dialogs[0], "Dialog should mention the source name"

    def test_delete_accepted_calls_api(self, page: Page, base_url: str):
        """Accepting the delete dialog POSTs to /api/sources/delete."""
        calls: list[dict] = []

        def handle_delete(route: Route):
            calls.append(json.loads(route.request.post_data))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success"}),
            )

        _mock_sources(page, {FAKE_SOURCE["uuid"]: FAKE_SOURCE})
        _mock_providers(page, ["UrlImageProvider"])
        _mock_provider_schema(page)
        page.route("**/api/sources/delete", handle_delete)

        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box button[title='Delete']", timeout=5000)

        page.on("dialog", lambda d: d.accept())
        page.locator("#source-management-box button[title='Delete']").first.click()
        wait_until(page, lambda: calls, "Expected POST to /api/sources/delete")
        assert FAKE_SOURCE["uuid"] in calls[0].get("sourceUuids", [])

    def test_delete_dismissed_does_not_call_api(self, page: Page, base_url: str):
        """Dismissing the confirm dialog does NOT send a delete request."""
        calls: list[bool] = []
        page.route(
            "**/api/sources/delete",
            lambda route: (
                calls.append(True),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success"}),
                ),
            ),
        )
        _mock_sources(page, {FAKE_SOURCE["uuid"]: FAKE_SOURCE})
        _mock_providers(page, ["UrlImageProvider"])
        _mock_provider_schema(page)

        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box button[title='Delete']", timeout=5000)

        page.on("dialog", lambda d: d.dismiss())
        page.locator("#source-management-box button[title='Delete']").first.click()
        page.wait_for_timeout(500)

        assert not calls, "API must NOT be called when delete is cancelled"
