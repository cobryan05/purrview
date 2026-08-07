"""
Tests for the Sources configuration UI.

Exercises the renderSourceManager() component rendered inside
Operations > Sources > #source-management-box:
  - DOM structure (list panel beside editor panel)
  - Source list loading (sources appear as list rows)
  - Provider dropdown populated from /api/sources/image-providers/list
  - Full create flow: select provider -> schema fields appear -> fill details -> Create -> API called
  - Delete flow (modal confirm; API called or suppressed)

Sources are edited in a list-beside-editor screen (ui/utils/listEditorScreen.js),
the same shape as Datasets and Filter Chains: pick a row on the left, edit it on the
right, Save/Delete in the editor's sticky bar, Test Source in its sticky footer.
There is no always-expanded "Create Source" form any more -- the editor shows a blank
New Source by default, and "+ New Source" in the list footer returns to it.

Run with:
    pytest playwright_tests/test_configuration_sources.py -p no:asyncio -v
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
# Navigation / locator helpers
# ---------------------------------------------------------------------------


def _navigate_to_sources(page: Page, base_url: str):
    """Open the app and activate Operations > Sources."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-operations']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-operations .workspace-tab-button[data-tab='tab-source-management']").click()
    page.wait_for_load_state("networkidle")


def _editor_bar(page: Page):
    """The sticky header bar of the source editor (holds Create/Save/Delete)."""
    return page.locator("#source-management-box .le-editor .panel-box__bar")


def _editor_footer(page: Page):
    """The sticky footer of the source editor (holds Test Source)."""
    return page.locator("#source-management-box .le-editor .panel-box__footer")


def _provider_select(page: Page):
    return page.get_by_test_id("source-provider-select")


def _name_input(page: Page):
    return page.get_by_test_id("source-name-input")


def _open_source(page: Page, name: str = FAKE_SOURCE["name"]):
    """Click a source row in the list to load it into the editor."""
    page.wait_for_selector("#source-management-box .le-list-item", timeout=5000)
    page.locator("#source-management-box .le-list-item", has_text=name).first.click()
    page.wait_for_timeout(700)


def _fill_url_field(page: Page, value: str):
    """Fill the schema-driven URL text input (not the source-name box)."""
    page.locator("#source-management-box .schema-form input[type='text']").first.fill(value)


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
        """The sources pane renders the list-editor screen: a list panel and an editor."""
        _mock_sources(page, {})
        _mock_providers(page)
        _navigate_to_sources(page, base_url)
        expect(page.locator("#source-management-box")).to_be_visible()
        expect(page.locator("#source-management-box .le-list-panel")).to_be_visible()
        expect(page.locator("#source-management-box .le-editor")).to_be_visible()
        # A blank "New Source" is loaded by default, so the editor is usable
        # without first picking anything.
        expect(_editor_bar(page)).to_contain_text("New Source")

    def test_list_panel_titled_sources(self, page: Page, base_url: str):
        _mock_sources(page, {})
        _mock_providers(page)
        _navigate_to_sources(page, base_url)
        expect(page.locator("#source-management-box .le-list-panel .panel-box__title")).to_contain_text("Sources")

    def test_no_js_errors_on_sources_tab(self, page: Page, base_url: str):
        """No console errors while loading the Sources pane."""
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        _mock_sources(page, {})
        _mock_providers(page)
        _navigate_to_sources(page, base_url)
        page.wait_for_timeout(1000)
        uncaught = [e for e in errors if "TypeError" in e or "Uncaught" in e]
        assert uncaught == [], f"Uncaught JS errors: {uncaught}"


# ---------------------------------------------------------------------------
# Source list loading
# ---------------------------------------------------------------------------


class TestSourceListLoading:
    def test_existing_sources_render_as_rows(self, page: Page, base_url: str):
        """Sources returned from /api/sources/get are rendered in the list."""
        _mock_sources(page, {FAKE_SOURCE["uuid"]: FAKE_SOURCE})
        _mock_providers(page, ["UrlImageProvider"])
        _mock_provider_schema(page)
        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box .le-list-item", timeout=5000)
        content = page.locator("#source-management-box .le-list-panel").inner_text()
        assert FAKE_SOURCE["name"] in content, f"Source name not found in: {content!r}"

    def test_list_row_shows_provider(self, page: Page, base_url: str):
        """A row's meta line names the provider, so the list is scannable."""
        _mock_sources(page, {FAKE_SOURCE["uuid"]: FAKE_SOURCE})
        _mock_providers(page, ["UrlImageProvider"])
        _mock_provider_schema(page)
        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box .le-list-item", timeout=5000)
        expect(page.locator("#source-management-box .le-list-item").first).to_contain_text("UrlImageProvider")

    def test_selecting_source_loads_it_into_editor(self, page: Page, base_url: str):
        """Clicking a row loads that source's name into the editor."""
        _mock_sources(page, {FAKE_SOURCE["uuid"]: FAKE_SOURCE})
        _mock_providers(page, ["UrlImageProvider"])
        _mock_provider_schema(page)
        _navigate_to_sources(page, base_url)
        _open_source(page)
        expect(_editor_bar(page)).to_contain_text(FAKE_SOURCE["name"])
        expect(_name_input(page)).to_have_value(FAKE_SOURCE["name"])

    def test_test_source_button_in_editor_footer(self, page: Page, base_url: str):
        """'Test Source' is always available in the editor's sticky footer."""
        _mock_sources(page, {FAKE_SOURCE["uuid"]: FAKE_SOURCE})
        _mock_providers(page, ["UrlImageProvider"])
        _mock_provider_schema(page)
        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box button[title='Test Source']", timeout=5000)
        expect(_editor_footer(page).locator("button[title='Test Source']")).to_be_visible()

    def test_selected_source_has_delete_button(self, page: Page, base_url: str):
        """A saved source offers Delete in the editor bar."""
        _mock_sources(page, {FAKE_SOURCE["uuid"]: FAKE_SOURCE})
        _mock_providers(page, ["UrlImageProvider"])
        _mock_provider_schema(page)
        _navigate_to_sources(page, base_url)
        _open_source(page)
        expect(_editor_bar(page).locator("button", has_text="Delete")).to_be_visible()

    def test_filter_box_narrows_the_list(self, page: Page, base_url: str):
        """The list's filter box hides non-matching rows."""
        other = {**FAKE_SOURCE, "uuid": "dddddddd-0000-0000-0000-000000000002", "name": "Back Yard"}
        _mock_sources(page, {FAKE_SOURCE["uuid"]: FAKE_SOURCE, other["uuid"]: other})
        _mock_providers(page, ["UrlImageProvider"])
        _mock_provider_schema(page)
        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box .le-list-item", timeout=5000)
        expect(page.locator("#source-management-box .le-list-item:visible")).to_have_count(2)

        page.locator("#source-management-box .le-list-search input").fill("Back")
        expect(page.locator("#source-management-box .le-list-item:visible")).to_have_count(1)
        expect(page.locator("#source-management-box .le-list-item:visible")).to_contain_text("Back Yard")

    def test_empty_list_shows_one_empty_message_while_filtering(self, page: Page, base_url: str):
        """An empty list already says "No sources yet." — don't stack "No matches." under it."""
        _mock_sources(page, {})
        _mock_providers(page, ["UrlImageProvider"])
        _mock_provider_schema(page)
        _navigate_to_sources(page, base_url)
        page.wait_for_selector("#source-management-box .le-list-empty", timeout=5000)

        page.locator("#source-management-box .le-list-search input").fill("anything")
        page.wait_for_timeout(200)
        expect(page.locator("#source-management-box .le-list-empty:visible")).to_have_count(1)


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

        page.wait_for_selector("[data-testid='source-provider-select']", timeout=5000)
        options = _provider_select(page).locator("option").all_text_contents()
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
        page.wait_for_selector("[data-testid='source-provider-select']", timeout=5000)

        _provider_select(page).select_option("UrlImageProvider")
        wait_until(page, lambda: schema_calls, "Expected /api/sources/image-providers/schema to be called")
        assert "UrlImageProvider" in schema_calls[0]

    def test_schema_fields_appear_after_provider_selection(self, page: Page, base_url: str):
        """After selecting a provider, schema-driven input fields appear in the form."""
        _mock_providers(page, ["UrlImageProvider"])
        _mock_sources(page, {})
        _mock_provider_schema(page, URL_PROVIDER_SCHEMA)

        _navigate_to_sources(page, base_url)
        page.wait_for_selector("[data-testid='source-provider-select']", timeout=5000)

        _provider_select(page).select_option("UrlImageProvider")
        page.wait_for_timeout(800)

        content = page.locator("#source-management-box .le-editor").inner_html()
        assert "URL" in content, f"Expected 'URL' schema field label in: {content[:500]!r}"

    def test_schema_form_uses_the_compact_grid(self, page: Page, base_url: str):
        """Schema fields render into the shared `.schema-form` grid rather than a
        stack of full-width label-above-input rows."""
        _mock_providers(page, ["UrlImageProvider"])
        _mock_sources(page, {})
        _mock_provider_schema(page, URL_PROVIDER_SCHEMA)

        _navigate_to_sources(page, base_url)
        page.wait_for_selector("[data-testid='source-provider-select']", timeout=5000)
        _provider_select(page).select_option("UrlImageProvider")
        page.wait_for_selector("#source-management-box .schema-form", timeout=5000)

        # url + user, both narrow string fields → ordinary grid cells, not full-width.
        fields = page.locator("#source-management-box .schema-form > .schema-field")
        expect(fields).to_have_count(2)
        expect(page.locator("#source-management-box .schema-form > .schema-field--full")).to_have_count(0)


# ---------------------------------------------------------------------------
# Full create flow
# ---------------------------------------------------------------------------


class TestCreateSource:
    def test_create_source_calls_api(self, page: Page, base_url: str):
        """Full create flow: pick provider → fill name + URL → Create → /api/sources/create called."""
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
        page.wait_for_selector("[data-testid='source-provider-select']", timeout=5000)

        _provider_select(page).select_option("UrlImageProvider")
        page.wait_for_selector("#source-management-box .schema-form input", timeout=5000)

        _name_input(page).fill("My New Camera")
        _fill_url_field(page, "http://camera.example.com/snap.jpg")

        _editor_bar(page).locator("button", has_text="Create").click()
        wait_until(page, lambda: create_calls, "Expected POST to /api/sources/create")
        payload = create_calls[0]
        assert payload.get("sourceName") == "My New Camera"
        assert payload.get("imageProvider") == "UrlImageProvider"

    def test_create_requires_name(self, page: Page, base_url: str):
        """Create without a name does not call the API."""
        calls: list[bool] = []
        page.route(
            "**/api/sources/create",
            lambda route: (
                calls.append(True),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "source": FAKE_SOURCE}),
                ),
            ),
        )
        _mock_providers(page, ["UrlImageProvider"])
        _mock_sources(page, {})
        _mock_provider_schema(page, URL_PROVIDER_SCHEMA)

        _navigate_to_sources(page, base_url)
        page.wait_for_selector("[data-testid='source-provider-select']", timeout=5000)
        _provider_select(page).select_option("UrlImageProvider")
        page.wait_for_selector("#source-management-box .schema-form input", timeout=5000)

        create_btn = _editor_bar(page).locator("button", has_text="Create")
        if create_btn.is_enabled():
            create_btn.click()
        page.wait_for_timeout(500)

        assert not calls, "API must NOT be called when name is empty"


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

    def _setup(self, page: Page, base_url: str, create_calls: list):
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
        page.wait_for_selector("[data-testid='source-provider-select']", timeout=5000)
        _provider_select(page).select_option("UrlImageProvider")
        page.wait_for_selector("#source-management-box .schema-form input[type='number']", timeout=5000)

        _name_input(page).fill("My New Camera")
        _fill_url_field(page, "http://camera.example.com/snap.jpg")

    def test_untouched_default_zero_saves_as_number_not_empty_string(self, page: Page, base_url: str):
        """A field left untouched, with a schema default of 0.0, must save
        as the number 0 — not '' (the `||` vs `??` fallback bug)."""
        create_calls: list[dict] = []
        self._setup(page, base_url, create_calls)

        _editor_bar(page).locator("button", has_text="Create").click()
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
        self._setup(page, base_url, create_calls)

        page.locator("#source-management-box .schema-form input[type='number']").first.fill("2.5")

        _editor_bar(page).locator("button", has_text="Create").click()
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
        """Clicking Delete in the editor bar opens a modal naming the source."""
        _mock_sources(page, {FAKE_SOURCE["uuid"]: FAKE_SOURCE})
        _mock_providers(page, ["UrlImageProvider"])
        _mock_provider_schema(page)
        _navigate_to_sources(page, base_url)
        _open_source(page)

        _editor_bar(page).locator("button", has_text="Delete").click()
        modal = page.get_by_test_id("modal-shell-dialog")
        expect(modal).to_be_visible()
        expect(modal).to_contain_text(FAKE_SOURCE["name"])

    def test_delete_accepted_calls_api(self, page: Page, base_url: str):
        """Confirming delete POSTs to /api/sources/delete."""
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
        _open_source(page)

        _editor_bar(page).locator("button", has_text="Delete").click()
        page.get_by_test_id("modal-shell-dialog").locator("button", has_text="Delete").click()

        wait_until(page, lambda: calls, "Expected POST to /api/sources/delete")
        assert FAKE_SOURCE["uuid"] in calls[0].get("sourceUuids", [])

    def test_delete_dismissed_does_not_call_api(self, page: Page, base_url: str):
        """Dismissing the confirm modal does NOT send a delete request."""
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
        _open_source(page)

        _editor_bar(page).locator("button", has_text="Delete").click()
        page.get_by_test_id("modal-shell-dialog").locator("button", has_text="Cancel").click()
        page.wait_for_timeout(500)

        assert not calls, "API must NOT be called when delete is cancelled"
