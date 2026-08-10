"""
Smoke tests for the Watchers configuration UI (Configuration → Watchers).

Exercises renderWatchProfileManager() inside #watcher-management-box:
  - the pane renders as a list-beside-editor screen and its rows are visible
  - an existing watcher loads with its sources, rules and actions
  - the create flow POSTs to /api/watch-profiles
  - a rule row reads as a sentence and hides its second subject for `present`
  - the zones modal opens, draws a zone and returns it

Run with:
    pytest playwright_tests/test_configuration_watchers.py -p no:xdist -p no:asyncio -v
"""

import json

import pytest
from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

ZONE_UUID = "11111111-1111-1111-1111-111111111111"

FAKE_PROFILE = {
    "uuid": "99999999-cccc-0000-0000-000000000001",
    "name": "House cams",
    "description": "",
    "enabled": True,
    "trackerTypename": "IouTracker",
    "trackerParamsJson": {},
    "defaultModelGroupUuid": None,
    "defaultAnalysisFps": 5.0,
    "defaultAnalysisWidth": 640,
    "sources": [
        {
            "uuid": "ps-1",
            "profileUuid": "99999999-cccc-0000-0000-000000000001",
            "sourceUuid": "src-1",
            "order": 0,
            "enabled": True,
            "modelGroupUuid": None,
            "analysisFps": 5.0,
            "analysisWidth": 640,
            "recordRetentionS": 45.0,
            "trackerParamsJson": None,
            "bufferMode": "auto",
            "ringSeconds": 20.0,
            "ringMaxMb": 32.0,
            "zones": [
                {
                    "uuid": ZONE_UUID,
                    "profileSourceUuid": "ps-1",
                    "name": "litter box area",
                    "kind": "box",
                    "pointsJson": [[0.4, 0.4], [0.8, 0.8]],
                    "referenceWidth": 1920,
                    "referenceHeight": 1080,
                }
            ],
        }
    ],
    "rules": [
        {
            "uuid": "rule-1",
            "profileUuid": "99999999-cccc-0000-0000-000000000001",
            "order": 0,
            "name": "cat using litter box",
            "enabled": True,
            "subjectAKind": "label",
            "subjectARef": "label-cat",
            "relation": "contains",
            "subjectBKind": "zone",
            "subjectBRef": ZONE_UUID,
            "threshold": 0.6,
            "minDurationS": 2.0,
            "rearmDelayS": 10.0,
            "maxDurationS": 300.0,
            "eventLabel": "litterbox",
        }
    ],
    "actions": [
        {
            "uuid": "act-1",
            "profileUuid": "99999999-cccc-0000-0000-000000000001",
            "ruleUuid": None,
            "order": 0,
            "enabled": True,
            "kind": "publish_mqtt",
            "paramsJson": {"retain": False},
            "trigger": "event_started",
        }
    ],
    "labelLifetimes": [],
}


def _mock_watch_apis(page: Page, profiles: list | None = None, on_create=None):
    profiles = profiles or []

    def handle_collection(route: Route):
        if route.request.method == "POST":
            if on_create:
                on_create(json.loads(route.request.post_data or "{}"))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "profile": {**FAKE_PROFILE, "name": "New watcher"}}),
            )
            return
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "profiles": profiles}),
        )

    page.route(lambda url: url.split("?")[0].rstrip("/").endswith("/api/watch-profiles"), handle_collection)

    page.route(
        lambda url: "/api/watch-profiles/" in url and url.endswith("/validate"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "valid": True, "errors": [], "warnings": []}),
        ),
    )

    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/sources"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "sources": {
                        "src-1": {
                            "uuid": "src-1",
                            "name": "Litter cam",
                            "typename": "FfmpegImageProvider",
                            "params": {},
                        }
                    },
                }
            ),
        ),
    )


def _navigate_to_watchers(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-watcher-management']").click()
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("#watcher-management-box .wp-list-panel", timeout=5000)


class TestWatchersPane:
    def test_pane_renders_as_a_list_editor_screen(self, page: Page, base_url: str):
        _mock_watch_apis(page)
        _navigate_to_watchers(page, base_url)
        box = page.locator("#watcher-management-box")
        expect(box.locator(".list-editor-split")).to_be_visible()
        expect(box.locator(".wp-list-panel")).to_be_visible()

    def test_existing_watchers_appear_in_the_list(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        item = page.locator("[data-testid='watcher-list-item']").first
        expect(item).to_be_visible()
        expect(item).to_contain_text("House cams")

    def test_list_rows_are_visible_not_just_present(self, page: Page, base_url: str):
        """A new listEditorScreen prefix gets working layout but invisible rows unless
        its CSS exists — assert visibility, not just presence."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        item = page.locator("[data-testid='watcher-list-item']").first
        expect(item).to_be_visible()
        box = item.bounding_box()
        assert box and box["height"] > 0 and box["width"] > 0

    def test_empty_state_is_shown_when_there_are_no_watchers(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[])
        _navigate_to_watchers(page, base_url)
        expect(page.locator("#watcher-management-box")).to_contain_text("No watchers yet")


class TestWatcherEditor:
    def test_opening_a_watcher_loads_its_sources_rules_and_actions(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()

        expect(page.locator("[data-testid='watcher-name']")).to_have_value("House cams")
        expect(page.locator("[data-testid='watch-source-row']")).to_have_count(1)
        expect(page.locator("[data-testid='watch-rule-row']")).to_have_count(1)
        expect(page.locator("[data-testid='watch-action-row']")).to_have_count(1)

    def test_a_rule_row_reads_as_a_sentence(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-rule-relation']")).to_have_value("contains")
        expect(page.locator(".watch-rule-sentence")).to_contain_text("When")

    def test_present_relation_hides_the_second_subject(self, page: Page, base_url: str):
        """`present` is arity 1 — showing a second subject would imply a comparison
        that never happens."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-rule-relation']").select_option("present")
        expect(page.locator("[data-testid='watch-rule-subject-b']")).to_be_hidden()

    def test_overlaps_is_offered_and_keeps_the_second_subject(self, page: Page, base_url: str):
        """`overlaps` is the "cat at the water bowl" relation — arity 2, unlike present."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-rule-relation']").select_option("overlaps")
        expect(page.locator("[data-testid='watch-rule-relation']")).to_have_value("overlaps")
        expect(page.locator("[data-testid='watch-rule-subject-b']")).to_be_visible()

    def test_iou_is_not_worded_as_plain_overlap(self, page: Page, base_url: str):
        """IoU reads as "overlaps" only for same-size boxes; the old wording sent the
        water-bowl case to the one relation that cannot detect it."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        iou_option = page.locator("[data-testid='watch-rule-relation'] option[value='iou']")
        expect(iou_option).to_have_text("closely matches (same size)")

    def test_adding_a_rule_appends_a_row(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-add-rule']").click()
        expect(page.locator("[data-testid='watch-rule-row']")).to_have_count(2)

    def test_new_rules_default_to_hysteresis_values(self, page: Page, base_url: str):
        """A bare threshold flaps into an MQTT storm; the defaults must guard it."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-add-rule']").click()
        row = page.locator("[data-testid='watch-rule-row']").nth(1)
        inputs = row.locator(".watch-rule-tuning input")
        # threshold, starts-after, ends-after-clear, max-duration
        expect(inputs.nth(1)).to_have_value("2")
        expect(inputs.nth(2)).to_have_value("10")
        expect(inputs.nth(3)).to_have_value("300")

    def test_creating_a_watcher_posts_the_payload(self, page: Page, base_url: str):
        created = {}
        _mock_watch_apis(page, profiles=[], on_create=lambda body: created.update(body))
        _navigate_to_watchers(page, base_url)

        page.locator("#watcher-management-box button:has-text('New watcher')").first.click()
        page.locator("[data-testid='watcher-name']").fill("Back door watch")
        # A brand-new item's button reads "Create"; only an existing one says "Save".
        page.locator("#watcher-management-box button:has-text('Create')").first.click()

        wait_until(page, lambda: created.get("name") == "Back door watch")
        assert created["name"] == "Back door watch"

    def test_action_params_change_with_the_action_kind(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        kind = page.locator("[data-testid='watch-action-kind']").first
        kind.select_option("record_clip")
        expect(page.locator(".watch-action-params")).to_contain_text("Pre-roll")


class TestZonesModal:
    def test_zones_button_opens_the_modal(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-zones-btn']").first.click()
        expect(page.locator("[data-testid='watch-zones-stage']")).to_be_visible()

    def test_the_stage_has_real_dimensions(self, page: Page, base_url: str):
        """A zero-sized container yields a permanent scale of 0 for anything that
        measures it — the classic canvas-in-a-modal trap."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-zones-btn']").first.click()
        box = page.locator("[data-testid='watch-zones-canvas']").bounding_box()
        assert box and box["width"] > 100 and box["height"] > 100

    def test_existing_zones_are_listed(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-zones-btn']").first.click()
        expect(page.locator("[data-testid='watch-zone-name']").first).to_have_value("litter box area")

    def test_dragging_on_the_canvas_creates_a_zone(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-zones-btn']").first.click()

        canvas = page.locator("[data-testid='watch-zones-canvas']")
        box = canvas.bounding_box()
        page.mouse.move(box["x"] + 40, box["y"] + 40)
        page.mouse.down()
        page.mouse.move(box["x"] + 200, box["y"] + 160, steps=5)
        page.mouse.up()

        expect(page.locator("[data-testid='watch-zone-row']")).to_have_count(2)

    def test_a_stray_click_does_not_create_a_degenerate_zone(self, page: Page, base_url: str):
        """A zero-area zone would silently match nothing."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-zones-btn']").first.click()

        canvas = page.locator("[data-testid='watch-zones-canvas']")
        box = canvas.bounding_box()
        page.mouse.move(box["x"] + 50, box["y"] + 50)
        page.mouse.down()
        page.mouse.up()

        expect(page.locator("[data-testid='watch-zone-row']")).to_have_count(1)

    def test_done_returns_the_zones_and_updates_the_button(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        zones_btn = page.locator("[data-testid='watch-zones-btn']").first
        zones_btn.click()

        canvas = page.locator("[data-testid='watch-zones-canvas']")
        box = canvas.bounding_box()
        page.mouse.move(box["x"] + 40, box["y"] + 40)
        page.mouse.down()
        page.mouse.move(box["x"] + 200, box["y"] + 160, steps=5)
        page.mouse.up()

        page.locator("[data-testid='watch-zones-save']").click()
        expect(zones_btn).to_contain_text("Zones (2)")
