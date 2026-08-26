"""Browser tests for the Collectors configuration UI.

The pane renders `collector_params_schema()` fetched from the server, so what is
worth testing here is the wiring that only exists in the browser: that a stored
param survives the round trip through a schema-built form, and that the form
posts what the user actually sees.

Run with:
    pytest playwright_tests/test_configuration_collectors.py -p no:xdist -p no:asyncio -v
"""

import json

from playwright.sync_api import Page, Route, expect

FAKE_COLLECTOR = {
    "uuid": "33333333-cccc-0000-0000-000000000001",
    "name": "Ring collector",
    "description": "Doorbell frames",
    "enabled": True,
    "params": {
        "output_dir": "collected/ring",
        "max_images": 250,
        "enable_dedup": True,
    },
}


def _mock_collectors(page: Page, collectors: list | None = None):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/training-collectors"),
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "collectors": collectors or []}),
            )
            if route.request.method == "GET"
            else route.continue_()
        ),
    )


def _navigate_to_collectors(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-collector-management']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#collector-management-box .col-list-panel")


def _open_collector(page: Page):
    page.get_by_text(FAKE_COLLECTOR["name"], exact=True).first.click()
    page.wait_for_timeout(600)


class TestCollectorPane:
    def test_no_js_errors_on_collectors_tab(self, page: Page, base_url: str):
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        _mock_collectors(page, [])
        _navigate_to_collectors(page, base_url)
        page.wait_for_timeout(1200)
        uncaught = [e for e in errors if "TypeError" in e or "Uncaught" in e]
        assert uncaught == [], f"Uncaught JS errors: {uncaught}"

    def test_the_list_shows_where_a_collector_writes(self, page: Page, base_url: str):
        """Two collectors differ by output directory more often than by name, so
        that is the line that has to be on the row."""
        _mock_collectors(page, [FAKE_COLLECTOR])
        _navigate_to_collectors(page, base_url)
        page.wait_for_timeout(800)

        row = page.locator("#collector-management-box [data-collector-uuid]")
        expect(row.locator(".col-list-item__name")).to_have_text(FAKE_COLLECTOR["name"])
        expect(row.locator(".col-list-item__meta")).to_have_text("collected/ring")

    def test_editing_one_field_resends_every_other_stored_param(self, page: Page, base_url: str):
        """The form is built from the schema and seeded from the stored params, so a
        knob the form failed to seed would be silently rewritten to its default the
        next time anything at all was edited — a saved run's settings quietly
        changing under it. Editing the description is the least related edit there
        is, which is exactly why it is the one to save after."""
        _mock_collectors(page, [FAKE_COLLECTOR])
        _navigate_to_collectors(page, base_url)
        page.wait_for_timeout(800)
        _open_collector(page)

        patched: list[dict] = []

        def _handle_patch(route: Route):
            if route.request.method == "PATCH":
                patched.append(json.loads(route.request.post_data or "{}"))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "collector": FAKE_COLLECTOR}),
                )
            else:
                route.continue_()

        page.route(
            lambda url: url.split("?")[0].endswith(f"/api/training-collectors/{FAKE_COLLECTOR['uuid']}"),
            _handle_patch,
        )

        page.locator("#collector-management-box .col-editor input[placeholder='Optional']").fill("Edited")
        save = page.locator("#collector-management-box .col-editor .panel-box__bar button", has_text="Save")
        expect(save).to_be_enabled()
        save.click()
        page.wait_for_timeout(800)

        assert patched, "Save sent no PATCH"
        assert patched[0]["description"] == "Edited"
        params = patched[0]["params"]
        assert params["output_dir"] == "collected/ring"
        assert params["max_images"] == 250
        assert params["enable_dedup"] is True
