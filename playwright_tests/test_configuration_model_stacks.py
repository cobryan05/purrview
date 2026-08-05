"""
Smoke tests for the Model Stacks configuration UI.

Exercises renderModelStackManager() inside
Configuration > Model Stacks > #model-stack-management-box:
  - DOM structure (list panel + editor panel, create form visible by default)
  - Stack list loading, including the member-count meta line
  - Member rows: add / reorder / remove, and the create POST payload
  - A stack can never nest itself

The wire entity is still `model_groups` / `/api/model-groups`; "stack" is only
the display noun, so the mocks below speak the old names deliberately.

Run with:
    pytest playwright_tests/test_configuration_model_stacks.py -p no:asyncio -v
"""

import json

from playwright.sync_api import Page, expect

FAKE_STACK = {
    "uuid": "33333333-cccc-0000-0000-000000000001",
    "name": "Cats then Litter",
    "description": "Detector cascade",
    "members": [
        {"uuid": "m1", "order": 0, "modelName": "cats", "childGroupUuid": None},
        {"uuid": "m2", "order": 1, "modelName": "laundry", "childGroupUuid": None},
    ],
}


def _mock_model_groups(page: Page, groups: list | None = None):
    groups = groups if groups is not None else []
    page.route(
        lambda url: url.split("?")[0].endswith("/api/model-groups"),
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "groups": groups}),
            )
            if route.request.method == "GET"
            else route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "group": {**FAKE_STACK, "name": "Created"}}),
            )
        ),
    )


def _mock_models(page: Page, models: list | None = None):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/models"),
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "models": models or []}),
            )
            if route.request.method == "GET"
            else route.continue_()
        ),
    )


def _navigate(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-modelstack-management']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#model-stack-management-box .mc-list-panel", timeout=4000)


def _editor_bar(page: Page):
    return page.locator("#model-stack-management-box .mc-editor .panel-box__bar")


def _editor_body(page: Page):
    return page.locator("#model-stack-management-box .mc-editor .panel-box__body")


class TestModelStackPaneStructure:
    def test_pane_structure(self, page: Page, base_url: str):
        _mock_model_groups(page, [])
        _mock_models(page, ["cats", "laundry"])
        _navigate(page, base_url)

        expect(page.locator("#model-stack-management-box .mc-list-panel")).to_be_visible()
        expect(page.locator("#model-stack-management-box .mc-list-panel button", has_text="+ New")).to_be_visible()
        expect(_editor_bar(page)).to_contain_text("New Model Stack")

    def test_empty_state_when_no_stacks_exist(self, page: Page, base_url: str):
        """The pane ships before the concept has instances, so 0 stacks is the
        expected steady state for a while — it must read as empty, not broken."""
        _mock_model_groups(page, [])
        _mock_models(page, ["cats"])
        _navigate(page, base_url)

        expect(page.locator("#model-stack-management-box .mstack-empty").first).to_contain_text("No model stacks yet")

    def test_no_js_errors_on_model_stacks_tab(self, page: Page, base_url: str):
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        _mock_model_groups(page, [])
        _mock_models(page, ["cats"])
        _navigate(page, base_url)
        page.wait_for_timeout(1200)
        uncaught = [e for e in errors if "TypeError" in e or "Uncaught" in e]
        assert uncaught == [], f"Uncaught JS errors: {uncaught}"


class TestModelStackList:
    def test_existing_stack_listed_with_member_count(self, page: Page, base_url: str):
        _mock_model_groups(page, [FAKE_STACK])
        _mock_models(page, ["cats", "laundry"])
        _navigate(page, base_url)

        item = page.locator("#model-stack-management-box .mc-list-item").first
        expect(item).to_contain_text("Cats then Litter")
        expect(item).to_contain_text("2 members")

    def test_selecting_a_stack_loads_its_members(self, page: Page, base_url: str):
        _mock_model_groups(page, [FAKE_STACK])
        _mock_models(page, ["cats", "laundry"])
        _navigate(page, base_url)

        page.locator("#model-stack-management-box .mc-list-item").first.click()
        page.wait_for_timeout(500)
        rows = _editor_body(page).locator(".mstack-member-row")
        expect(rows).to_have_count(2)
        expect(rows.nth(0).locator("select")).to_have_value("model:cats")
        expect(rows.nth(1).locator("select")).to_have_value("model:laundry")

    def test_a_stack_cannot_nest_itself(self, page: Page, base_url: str):
        """Self-nesting is an obvious cycle; the server rejects cycles too, but
        the picker must not offer the one everybody would try first."""
        _mock_model_groups(page, [FAKE_STACK])
        _mock_models(page, ["cats"])
        _navigate(page, base_url)

        page.locator("#model-stack-management-box .mc-list-item").first.click()
        page.wait_for_timeout(500)
        options = _editor_body(page).locator(".mstack-member-row select").first.locator("option")
        values = options.evaluate_all("els => els.map(e => e.value)")
        assert f"stack:{FAKE_STACK['uuid']}" not in values, values


class TestModelStackEditing:
    def test_add_and_remove_members(self, page: Page, base_url: str):
        _mock_model_groups(page, [])
        _mock_models(page, ["cats", "laundry"])
        _navigate(page, base_url)

        body = _editor_body(page)
        expect(body.locator(".mstack-member-row")).to_have_count(0)

        body.locator("button", has_text="+ Add member").click()
        body.locator("button", has_text="+ Add member").click()
        expect(body.locator(".mstack-member-row")).to_have_count(2)

        body.locator(".mstack-member-row button[title='Remove from stack']").first.click()
        expect(body.locator(".mstack-member-row")).to_have_count(1)

    def test_reorder_swaps_members(self, page: Page, base_url: str):
        _mock_model_groups(page, [FAKE_STACK])
        _mock_models(page, ["cats", "laundry"])
        _navigate(page, base_url)

        page.locator("#model-stack-management-box .mc-list-item").first.click()
        page.wait_for_timeout(500)
        body = _editor_body(page)
        rows = body.locator(".mstack-member-row")

        # Move the second member up; the two selects swap values.
        rows.nth(1).locator("button[title='Move up']").click()
        page.wait_for_timeout(300)
        expect(rows.nth(0).locator("select")).to_have_value("model:laundry")
        expect(rows.nth(1).locator("select")).to_have_value("model:cats")

    def test_create_posts_ordered_members(self, page: Page, base_url: str):
        posted: list[dict] = []

        def _capture(route):
            if route.request.method == "POST":
                posted.append(json.loads(route.request.post_data or "{}"))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "group": {**FAKE_STACK, "name": "Cascade"}}),
                )
            else:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "groups": []}),
                )

        page.route(lambda url: url.split("?")[0].endswith("/api/model-groups"), _capture)
        _mock_models(page, ["cats", "laundry"])
        _navigate(page, base_url)

        body = _editor_body(page)
        body.locator("input[placeholder='Required']").fill("Cascade")
        body.locator("button", has_text="+ Add member").click()
        body.locator("button", has_text="+ Add member").click()
        body.locator(".mstack-member-row select").nth(1).select_option("model:laundry")

        _editor_bar(page).locator("button", has_text="Create").click()
        page.wait_for_timeout(800)

        assert posted, "no POST to /api/model-groups was made"
        members = posted[-1].get("members", [])
        # Members go out in the wire shape the server takes (snake_case
        # `model_name`), matching what the Inference panel's save already sends.
        assert [m.get("order") for m in members] == [0, 1], members
        assert members[0].get("model_name") == "cats", members
        assert members[1].get("model_name") == "laundry", members
