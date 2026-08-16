"""
Smoke tests for the NVR → Live pane (app/nvr/main.js).

Both properties here are ones the pane got *wrong* while still looking fine:

  - the card reported `defaultAnalysisFps` — the profile-level fallback — so a
    watcher whose source is configured at 1 fps advertised the default 5;
  - "Edit rules" navigated to the Watchers pane without naming the profile, so it
    opened whatever was last selected rather than the card you clicked.

Run with:
    pytest playwright_tests/test_nvr_live.py -p no:asyncio -v
"""

import json

from playwright.sync_api import Page, expect

PROFILE_UUID = "99999999-dddd-0000-0000-000000000001"
OTHER_UUID = "99999999-dddd-0000-0000-000000000002"


def _profile(uuid: str, name: str, sources: list) -> dict:
    return {
        "uuid": uuid,
        "name": name,
        "description": "",
        "enabled": True,
        "trackerTypename": "IouTracker",
        "trackerParamsJson": {},
        "defaultModelGroupUuid": None,
        "defaultModelName": None,
        # The trap: 5 is only the fallback for a source that sets no rate of its own.
        "defaultAnalysisFps": 5.0,
        "defaultAnalysisWidth": 640,
        "sources": sources,
        "rules": [],
        "actions": [],
        "labelLifetimes": [],
    }


def _source(uuid: str, analysis_fps, enabled: bool = True) -> dict:
    return {
        "uuid": uuid,
        "profileUuid": PROFILE_UUID,
        "sourceUuid": "src-1",
        "order": 0,
        "enabled": enabled,
        "modelGroupUuid": None,
        "modelName": None,
        "analysisFps": analysis_fps,
        "analysisWidth": 640,
        "recordRetentionS": 45.0,
        "trackerParamsJson": None,
        "bufferMode": "auto",
        "ringSeconds": 20.0,
        "ringMaxMb": 32.0,
        "zones": [],
    }


def _mock_nvr_apis(page: Page, profiles: list):
    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/watch-profiles"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "profiles": profiles}),
        ),
    )
    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/tasks/instances/get"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "tasks": {}}),
        ),
    )


def _navigate_to_nvr(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-nvr']").click()
    page.wait_for_selector("#nvr-live-box .nvr-profile-card", timeout=5000)


class TestNvrLiveFps:
    def test_a_source_rate_overrides_the_profile_default(self, page: Page, base_url: str):
        """A watcher configured at 1 fps must not advertise the 5 fps fallback."""
        _mock_nvr_apis(page, [_profile(PROFILE_UUID, "House cams", [_source("ps-1", 1.0)])])
        _navigate_to_nvr(page, base_url)
        card = page.locator(".nvr-profile-card").first
        expect(card).to_contain_text("1 fps")
        expect(card).not_to_contain_text("5 fps")

    def test_a_source_without_its_own_rate_shows_the_default(self, page: Page, base_url: str):
        _mock_nvr_apis(page, [_profile(PROFILE_UUID, "House cams", [_source("ps-1", None)])])
        _navigate_to_nvr(page, base_url)
        expect(page.locator(".nvr-profile-card").first).to_contain_text("5 fps")

    def test_mixed_rates_are_shown_as_a_range(self, page: Page, base_url: str):
        """Collapsing two different rates to one number would be the same lie again."""
        sources = [_source("ps-1", 1.0), _source("ps-2", 4.0)]
        _mock_nvr_apis(page, [_profile(PROFILE_UUID, "House cams", sources)])
        _navigate_to_nvr(page, base_url)
        expect(page.locator(".nvr-profile-card").first).to_contain_text("1–4 fps")

    def test_a_disabled_source_does_not_count(self, page: Page, base_url: str):
        sources = [_source("ps-1", 1.0), _source("ps-2", 9.0, enabled=False)]
        _mock_nvr_apis(page, [_profile(PROFILE_UUID, "House cams", sources)])
        _navigate_to_nvr(page, base_url)
        expect(page.locator(".nvr-profile-card").first).to_contain_text("1 fps")


class TestNvrEditRules:
    def test_edit_rules_opens_the_watcher_you_clicked(self, page: Page, base_url: str):
        """Without a pending selection the pane opens on whatever was last selected."""
        _mock_nvr_apis(
            page,
            [
                _profile(PROFILE_UUID, "House cams", [_source("ps-1", 1.0)]),
                _profile(OTHER_UUID, "Zzz garden cam", [_source("ps-2", 2.0)]),
            ],
        )
        _navigate_to_nvr(page, base_url)

        # The second card, so a pane that ignores the selection lands on the first.
        card = page.locator(".nvr-profile-card", has_text="Zzz garden cam").first
        card.locator("button:has-text('Edit rules')").click()

        page.wait_for_selector("#watcher-management-box .wp-list-panel", timeout=5000)
        expect(page.locator("[data-testid='watcher-name']")).to_have_value("Zzz garden cam")
