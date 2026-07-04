"""
Smoke tests for the Filter Chains configuration UI.

Exercises renderFilterChainManager() inside
Configuration > Filter Chains > #filter-chain-management-box:
  - DOM structure (list panel + editor panel, create form visible by default)
  - Filter chain list loading (chains appear in the list)
  - Create flow (fill name → Create → /api/filter-chains POST called)

Run with:
    pytest playwright_tests/test_configuration_filter_chains.py -p no:xdist -p no:asyncio -v
"""

import json

import pytest
from playwright.sync_api import Page, Route, expect

FAKE_CHAIN = {
    "uuid": "22222222-bbbb-0000-0000-000000000001",
    "name": "Corrupt Image Cleanup",
    "description": "Discards corrupt images and strips unverified bboxes",
    "steps": [
        {
            "uuid": "step-uuid-1",
            "filterChainUuid": "22222222-bbbb-0000-0000-000000000001",
            "order": 0,
            "modelName": "corrupt_classifier",
            "confidence": 0.8,
            "iouThreshold": 0.5,
            "action": "discard_image",
            "actionParams": {},
            "enabled": True,
        }
    ],
}


def _navigate_to_filter_chains(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-filterchain-management']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#filter-chain-management-box .fc-list-panel", timeout=4000)


def _mock_filter_chains(page: Page, chains: list | None = None):
    chains = chains or []
    page.route(
        lambda url: url.split("?")[0].endswith("/api/filter-chains"),
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "chains": chains}),
            )
            if route.request.method == "GET"
            else route.continue_()
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


def _mock_labels(page: Page, labels: list | None = None):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "/api/labels/" not in url,
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "labels": labels or []}),
            )
            if route.request.method == "GET"
            else route.continue_()
        ),
    )


def _mock_tags_empty(page: Page):
    page.route(
        "**/api/tags/list",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"tags": []}),
        ),
    )


def _editor_bar(page: Page):
    return page.locator("#filter-chain-management-box .fc-editor .panel-box__bar")


def _editor_body(page: Page):
    return page.locator("#filter-chain-management-box .fc-editor .panel-box__body")


class TestFilterChainPaneStructure:
    def test_pane_structure(self, page: Page, base_url: str):
        _mock_filter_chains(page, [])
        _mock_models(page, [])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        _navigate_to_filter_chains(page, base_url)

        expect(page.locator("#filter-chain-management-box .fc-list-panel")).to_be_visible()
        expect(page.locator("#filter-chain-management-box .fc-list-panel button", has_text="+ New")).to_be_visible()
        expect(_editor_bar(page)).to_contain_text("New Filter Chain")
        expect(_editor_bar(page).locator("button", has_text="Create")).to_be_visible()
        expect(_editor_body(page).locator("input[placeholder='Required']")).to_be_visible()

    def test_no_js_errors_on_filter_chains_tab(self, page: Page, base_url: str):
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        _mock_filter_chains(page, [])
        _mock_models(page, [])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        _navigate_to_filter_chains(page, base_url)
        page.wait_for_timeout(1200)
        uncaught = [e for e in errors if "TypeError" in e or "Uncaught" in e]
        assert uncaught == [], f"Uncaught JS errors: {uncaught}"


class TestFilterChainListLoading:
    def test_existing_chain_shown(self, page: Page, base_url: str):
        _mock_filter_chains(page, [FAKE_CHAIN])
        _mock_models(page, ["corrupt_classifier"])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        _navigate_to_filter_chains(page, base_url)
        page.wait_for_timeout(1000)
        list_item = page.locator("#filter-chain-management-box [data-chain-uuid]")
        expect(list_item.locator(".fc-list-item__name")).to_have_text(FAKE_CHAIN["name"])
        expect(list_item.locator(".fc-list-item__meta")).to_have_text("1 step")

    def test_chain_has_delete_affordance(self, page: Page, base_url: str):
        _mock_filter_chains(page, [FAKE_CHAIN])
        _mock_models(page, ["corrupt_classifier"])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        _navigate_to_filter_chains(page, base_url)
        page.wait_for_timeout(800)
        page.get_by_text(FAKE_CHAIN["name"], exact=True).first.click()
        page.wait_for_timeout(600)

        expect(_editor_bar(page)).to_contain_text(FAKE_CHAIN["name"])
        expect(_editor_bar(page).locator("button", has_text="Delete")).to_be_visible()

    def test_editing_existing_chain_preserves_its_steps_on_save(self, page: Page, base_url: str):
        """Regression test: opening a chain with steps and saving an unrelated
        scalar-field edit must resend the existing steps, not an empty list —
        a prior bug clobbered _stepData on the very first render pass."""
        _mock_filter_chains(page, [FAKE_CHAIN])
        _mock_models(page, ["corrupt_classifier"])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        _navigate_to_filter_chains(page, base_url)
        page.wait_for_timeout(800)
        page.get_by_text(FAKE_CHAIN["name"], exact=True).first.click()
        page.wait_for_timeout(600)

        # Exactly one step row rendered from the fetched chain.
        expect(page.locator("#filter-chain-management-box .fc-step-row")).to_have_count(1)

        patched_payloads = []

        def _handle_patch(route: Route):
            if route.request.method == "PATCH":
                patched_payloads.append(json.loads(route.request.post_data or "{}"))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "chain": FAKE_CHAIN}),
                )
            else:
                route.continue_()

        page.route(
            lambda url: url.split("?")[0].endswith(f"/api/filter-chains/{FAKE_CHAIN['uuid']}"),
            _handle_patch,
        )

        desc_input = _editor_body(page).locator("input").nth(1)
        desc_input.fill("updated description")
        _editor_bar(page).locator("button", has_text="Save").click()
        page.wait_for_timeout(800)

        assert len(patched_payloads) == 1
        assert len(patched_payloads[0]["steps"]) == 1
        assert patched_payloads[0]["steps"][0]["model_name"] == "corrupt_classifier"


class TestFilterChainCreateFlow:
    def test_create_chain_posts_to_api(self, page: Page, base_url: str):
        _mock_filter_chains(page, [])
        _mock_models(page, ["corrupt_classifier"])
        _mock_labels(page, [])
        _mock_tags_empty(page)
        _navigate_to_filter_chains(page, base_url)

        created_payloads = []

        def _handle_post(route: Route):
            if route.request.method == "POST":
                created_payloads.append(json.loads(route.request.post_data or "{}"))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "status": "success",
                            "chain": {**FAKE_CHAIN, "name": created_payloads[-1].get("name"), "steps": []},
                        }
                    ),
                )
            else:
                route.continue_()

        page.route(lambda url: url.split("?")[0].endswith("/api/filter-chains"), _handle_post)

        _editor_body(page).locator("input[placeholder='Required']").fill("New Chain")
        _editor_bar(page).locator("button", has_text="Create").click()
        page.wait_for_timeout(800)

        assert len(created_payloads) == 1
        assert created_payloads[0]["name"] == "New Chain"
