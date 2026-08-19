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
import re

import pytest
from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

ZONE_UUID = "11111111-1111-1111-1111-111111111111"
MASK_UUID = "22222222-2222-2222-2222-222222222222"

FAKE_PROFILE = {
    "uuid": "99999999-cccc-0000-0000-000000000001",
    "name": "House cams",
    "description": "",
    "enabled": True,
    "trackerTypename": "IouTracker",
    "trackerParamsJson": {},
    "defaultModelGroupUuid": None,
    "defaultModelName": None,
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
            "modelName": None,
            "analysisFps": 5.0,
            "analysisWidth": 640,
            "recordRetentionS": 45.0,
            "trackerParamsJson": None,
            "bufferMode": "auto",
            "ringSeconds": 20.0,
            "ringMaxMb": 32.0,
            "disabledMaskUuidsJson": [],
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
            "subjectARefsJson": ["label-cat"],
            "relation": "contains",
            "subjectBKind": "zone",
            "subjectBRefsJson": [ZONE_UUID],
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
            "ruleUuidsJson": [],
            "sourceUuidsJson": [],
            "order": 0,
            "enabled": True,
            "kind": "publish_mqtt",
            "paramsJson": {"retain": False},
            "trigger": "event_started",
        }
    ],
    "labelLifetimes": [],
}


MODELS = ["cat_detector.onnx", "yolov8n.onnx"]

MODEL_GROUPS = [
    {"uuid": "grp-1", "name": "House stack", "description": "", "mergePolicy": "first_wins", "members": []},
]

LABELS = [
    {"uuid": "label-cat", "name": "Cat"},
    {"uuid": "label-tabby", "name": "Tabby", "parentUuid": "label-cat"},
    {"uuid": "label-ginger", "name": "Ginger", "parentUuid": "label-cat"},
    {"uuid": "label-litterbox", "name": "Litter Box"},
]

# Zones and masks belong to the source, not to a watcher's use of it, so they are served
# from their own endpoint rather than nested in the profile.
REGIONS = [
    {
        "uuid": ZONE_UUID,
        "sourceUuid": "src-1",
        "name": "litter box area",
        "role": "zone",
        "kind": "box",
        "pointsJson": [[0.4, 0.4], [0.8, 0.8]],
        "referenceWidth": 1920,
        "referenceHeight": 1080,
        "order": 0,
    },
    {
        "uuid": MASK_UUID,
        "sourceUuid": "src-1",
        "name": "doorway",
        "role": "mask",
        "kind": "box",
        "pointsJson": [[0.0, 0.0], [0.2, 0.2]],
        "referenceWidth": 1920,
        "referenceHeight": 1080,
        "order": 1,
    },
]

# As `GET /api/watch-profiles/trackers` returns them. Schema keys stay snake_case —
# they are data the tracker declared, not model fields, so they are not camelised.
TRACKERS = [
    {
        "typename": "ByteTracker",
        "displayName": "ByteTrack (two-stage)",
        "paramsSchema": {
            "track_thresh": {"type": "float", "required": False, "default": 0.5, "help": "High threshold."},
            "low_thresh": {"type": "float", "required": False, "default": 0.1, "help": "Low threshold."},
        },
    },
    {
        "typename": "IouTracker",
        "displayName": "IoU / SORT-lite",
        "paramsSchema": {
            "min_confidence": {"type": "float", "required": False, "default": 0.25, "help": "Floor."},
        },
    },
]

# Same profile, but the source runs one bare model rather than a stack.
SINGLE_MODEL_PROFILE = {
    **FAKE_PROFILE,
    "sources": [{**FAKE_PROFILE["sources"][0], "modelGroupUuid": None, "modelName": "cat_detector.onnx"}],
}

# Same profile, but the rule's second subject is a *label* rather than a zone —
# "Tabby is inside Litter Box".
LABEL_SUBJECT_PROFILE = {
    **FAKE_PROFILE,
    "rules": [
        {
            **FAKE_PROFILE["rules"][0],
            "name": "Tabby using litter box",
            "subjectARefsJson": ["label-tabby"],
            "subjectBKind": "label",
            "subjectBRefsJson": ["label-litterbox"],
        }
    ],
}

# A rule whose subject slot names several labels at once — one rule for three cats
# rather than three near-identical rules.
MULTI_LABEL_PROFILE = {
    **FAKE_PROFILE,
    "rules": [
        {
            **FAKE_PROFILE["rules"][0],
            "subjectARefsJson": ["label-tabby", "label-ginger"],
        }
    ],
}

# A rule whose subject slot opted into "include child labels" — stored as a flag and
# expanded when the watcher starts, not as an expanded ref list.
DESCENDANTS_PROFILE = {
    **FAKE_PROFILE,
    "rules": [{**FAKE_PROFILE["rules"][0], "subjectAIncludeDescendants": True}],
}

# A watcher that has opted out of the source's mask.
MASK_DISABLED_PROFILE = {
    **FAKE_PROFILE,
    "sources": [{**FAKE_PROFILE["sources"][0], "disabledMaskUuidsJson": [MASK_UUID]}],
}


def _mock_watch_apis(
    page: Page,
    profiles: list | None = None,
    on_create=None,
    on_update=None,
    ingest_tiers: dict | None = None,
    regions: list | None = None,
):
    profiles = profiles or []

    # Must be routed before the bare `/api/sources` matcher below, and is a distinct
    # path from it: the tier is derived per request, not a field on the source.
    tiers_body = (
        {"status": "success", "tiers": ingest_tiers}
        if ingest_tiers is not None
        else {
            "status": "success",
            "tiers": {
                "src-1": {
                    "tier": "native",
                    "reason": "MediaMTX pulls rtsp:// natively",
                    "buffered": True,
                    "transcoding": False,
                }
            },
        }
    )
    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/sources/ingest-tiers"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(tiers_body),
        ),
    )

    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/labels"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": LABELS}),
        ),
    )

    # Regions live on the source, so they come from their own endpoint rather than from
    # the profile. Routed before the bare `/api/sources` matcher, which would otherwise
    # answer this with a source list and leave every zone picker empty.
    #
    # Stateful on purpose: a save has to be visible to the next fetch, or the property
    # these tests exist to protect — a newly drawn zone reaching the rules without a
    # reload — cannot be observed at all.
    region_store = list(REGIONS if regions is None else regions)

    def handle_regions_get(route: Route):
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "regions": region_store}),
        )

    def handle_regions_set(route: Route):
        body = json.loads(route.request.post_data or "{}")
        source_uuid = body.get("sourceUuid")
        saved = [{**r, "sourceUuid": source_uuid} for r in body.get("regions", [])]
        region_store[:] = [r for r in region_store if r.get("sourceUuid") != source_uuid] + saved
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "regions": saved, "orphanedRules": []}),
        )

    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/sources/regions/get"),
        handle_regions_get,
    )
    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/sources/regions/set"),
        handle_regions_set,
    )

    # The region editor grabs a frame on open. Answered with a failure rather than a
    # fake JPEG: the drawing surface must work against a blank stage anyway (regions are
    # normalised), and asserting that keeps the modal usable when a camera is down.
    page.route(
        lambda url: "/grab-frame" in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "failure", "message": "camera unreachable in tests"}),
        ),
    )

    def handle_single(route: Route):
        if route.request.method in ("PUT", "PATCH", "POST") and on_update:
            on_update(json.loads(route.request.post_data or "{}"))
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "profile": profiles[0] if profiles else FAKE_PROFILE}),
        )

    # Must be excluded from the single-profile matcher below, which would otherwise
    # answer it with a profile and leave the picker silently falling back to its default.
    page.route(
        lambda url: url.rstrip("/").endswith("/api/watch-profiles/trackers"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "trackers": TRACKERS}),
        ),
    )

    page.route(
        lambda url: (
            "/api/watch-profiles/" in url
            and not url.endswith("/validate")
            and not url.rstrip("/").endswith("/trackers")
        ),
        handle_single,
    )

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

    # A watcher can select a Model Stack *or* a single model, so the editor reads both
    # lists. Unmocked, fetchModelsList() falls back to [] and the Models optgroup would
    # silently not exist — which is the thing under test.
    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/models"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "models": MODELS}),
        ),
    )

    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/model-groups"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "groups": MODEL_GROUPS}),
        ),
    )

    page.route(
        # `/api/sources/get`, not `/api/sources` — `fetchSourceList()` calls the former,
        # so a matcher for the latter never fired and every test in this file was
        # quietly running against the developer's real source list instead of this
        # fixture. Both are matched now so the fixture is what the editor actually sees.
        lambda url: url.split("?")[0].rstrip("/").endswith(("/api/sources", "/api/sources/get")),
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

    def test_a_label_subject_b_loads_its_own_label(self, page: Page, base_url: str):
        """A rule whose second subject is a label must reopen showing *that* label.

        The picker was built from the zone options, so a label uuid never matched and
        the row silently fell back to the first label — "Litter Box" became "Cat".
        """
        _mock_watch_apis(page, profiles=[LABEL_SUBJECT_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        subject_b = page.locator("[data-testid='watch-rule-subject-b']")
        expect(subject_b).to_contain_text("Litter Box")

    def test_saving_keeps_a_label_subject_b(self, page: Page, base_url: str):
        """Round-trip: the payload sent back must still name the chosen label."""
        saved = {}
        _mock_watch_apis(page, profiles=[LABEL_SUBJECT_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        # Save is disabled until the editor is dirty; an unrelated edit arms it.
        page.locator("[data-testid='watch-rule-name']").first.fill("Cat using litter box!")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        rule = saved["rules"][0]
        assert rule["subjectBKind"] == "label", rule
        assert rule["subjectBRefsJson"] == ["label-litterbox"], rule

    def test_a_rule_can_name_several_labels_at_once(self, page: Page, base_url: str):
        """One rule for three cats, rather than three near-identical rules."""
        _mock_watch_apis(page, profiles=[MULTI_LABEL_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        subject_a = page.locator("[data-testid='watch-rule-subject-a']")
        expect(subject_a).to_contain_text("Tabby")
        expect(subject_a).to_contain_text("Ginger")

    def test_the_label_picker_shows_the_hierarchy(self, page: Page, base_url: str):
        """A vocabulary of cats nested under Cat is unnavigable as a flat list."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-rule-subject-a'] .ms-dropdown__trigger").first.click()
        panel = page.locator("[data-testid='watch-rule-subject-a'] .ms-dropdown__panel")
        expect(panel).to_contain_text("Cat")
        expect(panel).to_contain_text("Tabby")

    def test_selecting_a_second_label_sends_both(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-rule-subject-a'] .ms-dropdown__trigger").first.click()
        # Scoped to the item list, not the whole panel: the panel also holds the
        # "Include child labels" toggle, which would shift every index by one.
        page.locator("[data-testid='watch-rule-subject-a'] .ms-dropdown__items input[type='checkbox']").nth(1).check()
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        assert len(saved["rules"][0]["subjectARefsJson"]) == 2, saved["rules"][0]

    def test_a_source_can_choose_an_individual_model_not_just_a_stack(self, page: Page, base_url: str):
        """Running one model used to require wrapping it in a one-member Model Stack."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        choice = page.locator("[data-testid='watch-model-choice']").first
        expect(choice.locator("optgroup[label='Model stacks'] option")).to_have_count(1)
        expect(choice.locator("optgroup[label='Models'] option")).to_have_count(len(MODELS))

    def test_a_source_reopens_on_its_chosen_model(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[SINGLE_MODEL_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        choice = page.locator("[data-testid='watch-model-choice']").first
        expect(choice).to_have_value("m:cat_detector.onnx")

    def test_saving_an_individual_model_sends_it_and_clears_the_stack(self, page: Page, base_url: str):
        """Both keys must go out: omitting the stack would leave the old one in place."""
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-model-choice']").first.select_option("m:yolov8n.onnx")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("sources")))
        source = saved["sources"][0]
        assert source["modelName"] == "yolov8n.onnx", source
        assert source["modelGroupUuid"] is None, source

    def test_saving_a_stack_clears_any_individual_model(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[SINGLE_MODEL_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-model-choice']").first.select_option("g:grp-1")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("sources")))
        source = saved["sources"][0]
        assert source["modelGroupUuid"] == "grp-1", source
        assert source["modelName"] is None, source

    def test_the_profile_default_can_also_be_an_individual_model(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-default-model-choice']").select_option("m:cat_detector.onnx")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: "defaultModelName" in saved)
        assert saved["defaultModelName"] == "cat_detector.onnx", saved
        assert saved["defaultModelGroupUuid"] is None, saved

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
        # threshold, nearby-margin, starts-after, ends-after-clear, max-duration
        expect(inputs.nth(1)).to_have_value("0")
        expect(inputs.nth(2)).to_have_value("2")
        expect(inputs.nth(3)).to_have_value("10")
        expect(inputs.nth(4)).to_have_value("300")

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

    def test_existing_regions_are_listed(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-zones-btn']").first.click()
        expect(page.locator("[data-testid='watch-zone-name']").first).to_have_value("litter box area")

    def test_zones_and_masks_are_listed_separately(self, page: Page, base_url: str):
        """A mask suppresses detections; reading one as a zone is the mistake that
        silently produces no events at all."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-zones-btn']").first.click()
        expect(page.locator("[data-testid='watch-zone-row'][data-role='zone']")).to_have_count(1)
        expect(page.locator("[data-testid='watch-zone-row'][data-role='mask']")).to_have_count(1)

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

        # Two seeded regions (one zone, one mask) plus the new one.
        expect(page.locator("[data-testid='watch-zone-row']")).to_have_count(3)

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

        expect(page.locator("[data-testid='watch-zone-row']")).to_have_count(2)

    def test_drawing_a_mask_adds_it_to_the_mask_list(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-zones-btn']").first.click()
        page.locator("[data-testid='watch-zones-role']").select_option("mask")

        canvas = page.locator("[data-testid='watch-zones-canvas']")
        box = canvas.bounding_box()
        page.mouse.move(box["x"] + 40, box["y"] + 40)
        page.mouse.down()
        page.mouse.move(box["x"] + 200, box["y"] + 160, steps=5)
        page.mouse.up()

        expect(page.locator("[data-testid='watch-zone-row'][data-role='mask']")).to_have_count(2)

    def test_saving_posts_the_regions(self, page: Page, base_url: str):
        """Regions save to the source directly — the camera's geometry is not the
        watcher's to hold hostage until someone saves the watcher."""
        posted = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        page.route(
            lambda url: url.split("?")[0].rstrip("/").endswith("/api/sources/regions/set"),
            lambda route: (
                posted.update(json.loads(route.request.post_data or "{}")),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "regions": REGIONS, "orphanedRules": []}),
                ),
            )[-1],
        )
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-zones-btn']").first.click()

        canvas = page.locator("[data-testid='watch-zones-canvas']")
        box = canvas.bounding_box()
        page.mouse.move(box["x"] + 40, box["y"] + 40)
        page.mouse.down()
        page.mouse.move(box["x"] + 200, box["y"] + 160, steps=5)
        page.mouse.up()

        page.locator("[data-testid='watch-zones-save']").click()
        wait_until(page, lambda: bool(posted.get("regions")))
        assert len(posted["regions"]) == 3, posted
        assert posted["sourceUuid"] == "src-1", posted

    def test_normalised_coordinates_are_sent_not_pixels(self, page: Page, base_url: str):
        """Regions are stored normalised so they survive a camera resolution change."""
        posted = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        page.route(
            lambda url: url.split("?")[0].rstrip("/").endswith("/api/sources/regions/set"),
            lambda route: (
                posted.update(json.loads(route.request.post_data or "{}")),
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "regions": REGIONS, "orphanedRules": []}),
                ),
            )[-1],
        )
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-zones-btn']").first.click()

        canvas = page.locator("[data-testid='watch-zones-canvas']")
        box = canvas.bounding_box()
        page.mouse.move(box["x"] + 40, box["y"] + 40)
        page.mouse.down()
        page.mouse.move(box["x"] + 200, box["y"] + 160, steps=5)
        page.mouse.up()

        page.locator("[data-testid='watch-zones-save']").click()
        wait_until(page, lambda: bool(posted.get("regions")))
        drawn = posted["regions"][-1]
        assert all(0.0 <= c <= 1.0 for point in drawn["pointsJson"] for c in point), drawn


class TestMaskOptOut:
    """Masks belong to the camera but apply per watcher.

    The default is what matters: everything ticked, so a mask drawn on a camera tomorrow
    protects the watchers written today without anyone revisiting them.
    """

    def _open_regions(self, page: Page, base_url: str, profile=None):
        _mock_watch_apis(page, profiles=[profile or FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-zones-btn']").first.click()
        expect(page.locator("[data-testid='watch-zones-stage']")).to_be_visible()

    def test_each_mask_offers_an_apply_toggle(self, page: Page, base_url: str):
        self._open_regions(page, base_url)
        expect(page.locator("[data-testid='watch-mask-toggle']")).to_have_count(1)

    def test_zones_have_no_apply_toggle(self, page: Page, base_url: str):
        """A zone is inert until a rule names it, so there is nothing to opt out of."""
        self._open_regions(page, base_url)
        zone_row = page.locator("[data-testid='watch-zone-row'][data-role='zone']").first
        expect(zone_row.locator("[data-testid='watch-mask-toggle']")).to_have_count(0)

    def test_masks_apply_by_default(self, page: Page, base_url: str):
        self._open_regions(page, base_url)
        expect(page.locator("[data-testid='watch-mask-toggle'] input").first).to_be_checked()

    def test_a_disabled_mask_loads_unticked(self, page: Page, base_url: str):
        self._open_regions(page, base_url, profile=MASK_DISABLED_PROFILE)
        expect(page.locator("[data-testid='watch-mask-toggle'] input").first).not_to_be_checked()

    def test_unticking_a_mask_sends_it_as_disabled(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-zones-btn']").first.click()
        page.locator("[data-testid='watch-mask-toggle'] input").first.uncheck()
        page.locator("[data-testid='watch-zones-save']").click()
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("sources")))
        assert saved["sources"][0]["disabledMaskUuidsJson"] == [MASK_UUID], saved["sources"][0]

    def test_an_applied_mask_is_not_sent_as_disabled(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-rule-name']").first.fill("armed")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("sources")))
        assert saved["sources"][0]["disabledMaskUuidsJson"] == [], saved["sources"][0]

    def test_the_source_row_stays_one_control_line(self, page: Page, base_url: str):
        """Why the opt-out lives in the modal: a permanent control here cost a line."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        box = page.locator("[data-testid='watch-source-row']").first.bounding_box()
        assert box["height"] < 90, f"the source row is {box['height']}px tall"


class TestTopicTemplate:
    def test_the_default_template_is_shown(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-topic']")).to_have_value("{watcher}/{source}/event/{rule}")

    def test_the_example_updates_as_you_type(self, page: Page, base_url: str):
        """The placeholders are only useful if you can see what they resolve to."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-action-topic']").fill("cats/{rule}/{subject}")
        expect(page.locator("[data-testid='watch-action-topic-example']")).to_contain_text(
            "cats/cat-using-litter-box/tabby"
        )

    def test_the_template_is_saved(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-action-topic']").fill("cats/{subject}/{target}")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("actions")))
        assert saved["actions"][0]["paramsJson"]["topic_template"] == "cats/{subject}/{target}", saved


class TestWatcherEditorDensity:
    """The editor's controls are bare Pico form elements, and Pico ships
    `input, select { width: 100% }`. Left alone, every control in a row claims the
    full width and drops onto its own line: the rule sentence — five short words —
    became five stacked lines, and one rule filled the viewport with whitespace.

    These bounds are deliberately loose. They are not pixel-perfect design assertions;
    they fail only if a control goes full-width again, which is the one regression that
    made this pane unusable.
    """

    def _open_profile(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-rule-row']").first).to_be_visible()

    def test_the_rule_sentence_sits_on_one_line(self, page: Page, base_url: str):
        """ "When [cats] [is inside] [Zone] [litter boxes]" reads as a sentence only
        while its controls share a line — the whole point of the sentence layout.

        The two subject slots are multi-select dropdowns rather than `<select>`s now, so
        this counts both kinds: they are still controls that must not go full-width.
        """
        self._open_profile(page, base_url)
        tops = page.locator(".watch-rule-sentence select, .watch-rule-sentence .ms-dropdown__trigger").evaluate_all(
            "els => els.map(e => Math.round(e.getBoundingClientRect().top))"
        )
        assert len(tops) >= 4, f"expected the sentence's controls, got {tops}"
        # Rounded tops: a dropdown trigger and a select of slightly different heights are
        # still on one line, and demanding identical pixels would fail on that alone.
        assert max(tops) - min(tops) <= 6, f"the rule sentence wrapped onto several lines: tops={tops}"

    def test_a_rule_row_does_not_fill_the_screen(self, page: Page, base_url: str):
        self._open_profile(page, base_url)
        box = page.locator("[data-testid='watch-rule-row']").first.bounding_box()
        assert box["height"] < 150, f"a single rule row is {box['height']}px tall"

    def test_source_and_action_rows_are_single_line(self, page: Page, base_url: str):
        self._open_profile(page, base_url)
        for testid in ("watch-source-row", "watch-action-row"):
            box = page.locator(f"[data-testid='{testid}']").first.bounding_box()
            assert box["height"] < 90, f"{testid} is {box['height']}px tall"

    def test_a_small_profile_fits_without_scrolling(self, page: Page, base_url: str):
        """One source, one rule and one action used to overflow the editor pane.

        The allowance was 2px and is now 30px. Two deliberate additions spent it: the
        MQTT topic template is a real new control in the action row, and the subject
        slots became multi-select dropdowns, whose trigger is a few pixels taller than
        the `<select>` it replaced. Every row is still one line per control group, which
        is the regression this test actually exists to catch — a control going
        full-width and stacking, which costs ~35px each and hundreds in total.
        """
        self._open_profile(page, base_url)
        body = page.locator("#watcher-management-box .le-editor-body").first
        overflow = body.evaluate("e => e.scrollHeight - e.clientHeight")
        assert overflow <= 30, f"the editor scrolls by {overflow}px for a 1/1/1 profile"

    def test_checkboxes_keep_their_box(self, page: Page, base_url: str):
        """Overriding Pico's explicit checkbox width collapses it to a thin bar."""
        self._open_profile(page, base_url)
        box = page.locator(".watch-identity input[type='checkbox']").first.bounding_box()
        assert box["width"] > 8, f"the Enabled checkbox is {box['width']}px wide"
        assert abs(box["width"] - box["height"]) < 4, f"checkbox is not square: {box}"


class TestZonesReachTheRules:
    """A zone is drawn on a source but referenced by a rule, so the two lists have to
    stay in step while the editor is open — the profile is not reloaded in between."""

    def _open(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-rule-row']").first).to_be_visible()

    def _draw_a_zone(self, page: Page):
        page.locator("[data-testid='watch-zones-btn']").first.click()
        canvas = page.locator("[data-testid='watch-zones-canvas']")
        box = canvas.bounding_box()
        page.mouse.move(box["x"] + 40, box["y"] + 40)
        page.mouse.down()
        page.mouse.move(box["x"] + 200, box["y"] + 160, steps=5)
        page.mouse.up()
        page.locator("[data-testid='watch-zones-save']").click()

    def _zone_choices(self, page: Page):
        """The zone picker's checkbox rows, opening the panel if it is closed.

        Scoped to `__items` rather than the whole panel: a label-kind slot also renders
        an "Include child labels" checkbox in there, and counting both would make this
        assert on a control that is not a zone.
        """
        subject_b = page.locator("[data-testid='watch-rule-subject-b']").first
        panel = subject_b.locator(".ms-dropdown__panel")
        if not panel.is_visible():
            subject_b.locator(".ms-dropdown__trigger").click()
        return panel.locator(".ms-dropdown__items input[type='checkbox']")

    def test_a_new_zone_becomes_selectable_in_an_existing_rule(self, page: Page, base_url: str):
        """Drawing a zone and then being unable to point a rule at it without saving
        and reopening the whole watcher is the wrong order to build a rule in."""
        self._open(page, base_url)
        expect(self._zone_choices(page)).to_have_count(1)

        self._draw_a_zone(page)

        expect(self._zone_choices(page)).to_have_count(2)

    def test_the_rules_existing_zone_choice_survives_the_refresh(self, page: Page, base_url: str):
        """Rebuilding the picker must not silently repoint the rule at zone #1."""
        self._open(page, base_url)
        subject_b = page.locator("[data-testid='watch-rule-subject-b']").first
        expect(subject_b).to_contain_text("litter box area")

        self._draw_a_zone(page)

        expect(subject_b).to_contain_text("litter box area")

    def test_removing_a_source_drops_its_zones_from_the_rules(self, page: Page, base_url: str):
        """The same staleness in the other direction: a rule must not keep offering a
        zone belonging to a source that is no longer in the profile."""
        self._open(page, base_url)
        expect(self._zone_choices(page)).to_have_count(1)

        page.locator("[data-testid='watch-source-row'] .watch-row-remove").first.click()

        expect(self._zone_choices(page)).to_have_count(0)


class TestTrackingSection:
    """Tracker choice and per-label lifetimes.

    Both were configurable in the schema and read by the task from the day the watcher
    shipped, but the editor never rendered or sent them — so the tracker was permanently
    the default and lifetimes were unreachable.
    """

    def _open_profile(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-rule-row']").first).to_be_visible()

    def test_the_picker_is_populated_from_the_registry(self, page: Page, base_url: str):
        self._open_profile(page, base_url)
        options = page.locator("[data-testid='watch-tracker'] option").all_text_contents()
        assert "ByteTrack (two-stage)" in options
        assert "IoU / SORT-lite" in options

    def test_params_follow_the_selected_tracker(self, page: Page, base_url: str):
        """Schema-driven: switching trackers must re-render the fields, not keep stale ones."""
        self._open_profile(page, base_url)
        page.locator("[data-testid='watch-tracker-tuning'] summary").click()

        page.locator("[data-testid='watch-tracker']").select_option("ByteTracker")
        expect(page.locator("[data-testid='watch-tracker-param-track_thresh']")).to_be_visible()

        page.locator("[data-testid='watch-tracker']").select_option("IouTracker")
        expect(page.locator("[data-testid='watch-tracker-param-track_thresh']")).to_have_count(0)
        expect(page.locator("[data-testid='watch-tracker-param-min_confidence']")).to_be_visible()

    def test_tuning_is_collapsed_so_the_editor_still_fits(self, page: Page, base_url: str):
        self._open_profile(page, base_url)
        expect(page.locator("[data-testid='watch-tracker-params']")).to_be_hidden()

    def test_a_label_lifetime_can_be_added_and_is_saved(self, page: Page, base_url: str):
        """The headline case: a litter box outlives a cat by 30x."""
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=saved.update)
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-rule-row']").first).to_be_visible()

        page.locator("[data-testid='watch-add-lifetime']").click()
        row = page.locator("[data-testid='watch-lifetime-row']").first
        row.locator("[data-testid='watch-lifetime-label']").select_option("label-litterbox")
        ttl = row.locator("[data-testid='watch-lifetime-ttl']")
        ttl.fill("30")
        row.locator("[data-testid='watch-lifetime-anchor']").check()
        page.locator("[data-testid='watch-tracker']").select_option("ByteTracker")

        page.get_by_role("button", name="Save", exact=True).first.click()

        # Poll rather than assert immediately: the save is a round trip through the
        # mocked route, which runs on this same thread.
        wait_until(page, lambda: saved.get("trackerTypename"), "the save never reached the API")
        assert saved.get("trackerTypename") == "ByteTracker"
        lifetimes = saved.get("labelLifetimes") or []
        assert lifetimes and lifetimes[0]["labelUuid"] == "label-litterbox"
        assert lifetimes[0]["ttlS"] == 30
        assert lifetimes[0]["isAnchor"] is True


class TestBufferTierBadge:
    """The pre-roll field's honesty affordance.

    "buffer s" is a request for pre-roll, and a snapshot camera cannot honour it at any
    value. Before this badge the mistake was invisible until an event fired and the clip
    that was supposed to appear simply never did.
    """

    RING_TIERS = {
        "src-1": {
            "tier": "ring",
            "reason": "Snapshot/frame source — no continuous stream to record",
            "buffered": False,
            "transcoding": False,
        }
    }

    def _open_profile(self, page: Page, base_url: str, ingest_tiers=None):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], ingest_tiers=ingest_tiers)
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-source-row']").first).to_be_visible()

    def test_a_streamable_source_reports_a_rewind_buffer(self, page: Page, base_url: str):
        self._open_profile(page, base_url)
        badge = page.locator("[data-testid='watch-buffer-badge']").first
        expect(badge).to_contain_text("rewind")
        expect(badge).not_to_have_class("watch-buffer-badge--unbuffered")

    def test_a_snapshot_source_warns_that_pre_roll_cannot_be_honoured(self, page: Page, base_url: str):
        self._open_profile(page, base_url, ingest_tiers=self.RING_TIERS)
        badge = page.locator("[data-testid='watch-buffer-badge']").first
        expect(badge).to_contain_text("no rewind")
        expect(badge).to_have_class("watch-buffer-badge watch-buffer-badge--unbuffered")

    def test_the_reason_is_available_to_the_user_not_just_the_state(self, page: Page, base_url: str):
        """A bare "no buffer" invites "why not?" — the reason is what makes it actionable."""
        self._open_profile(page, base_url, ingest_tiers=self.RING_TIERS)
        badge = page.locator("[data-testid='watch-buffer-badge']").first
        assert "no continuous stream" in (badge.get_attribute("title") or "")

    def test_the_pre_roll_field_is_marked_but_still_editable(self, page: Page, base_url: str):
        """Marked, not disabled: the value keeps its meaning if the source is repointed."""
        self._open_profile(page, base_url, ingest_tiers=self.RING_TIERS)
        row = page.locator("[data-testid='watch-source-row']").first
        retention = row.locator("input[type='number']").nth(2)
        expect(retention).to_be_editable()
        assert "watch-input--inert" in (retention.get_attribute("class") or "")

    def test_no_badge_is_shown_when_the_tier_is_unknown(self, page: Page, base_url: str):
        """An empty answer must render nothing, not assert "no buffer" about a source
        the server simply had no answer for."""
        self._open_profile(page, base_url, ingest_tiers={})
        badge = page.locator("[data-testid='watch-buffer-badge']").first
        expect(badge).to_have_text("")


class TestRingFieldsSurviveASave:
    """A whole-list replace drops what the editor does not send.

    `ringSeconds` / `ringMaxMb` / `bufferMode` were omitted by the editor, so every save
    reset them to defaults. That was invisible while nothing read them; the watcher now
    reads all three to size an unbuffered source's frame ring, so a save would quietly
    undo a configured ring.
    """

    RING_TIERS = {
        "src-1": {"tier": "ring", "reason": "Snapshot source", "buffered": False, "transcoding": False},
    }

    def _open(self, page: Page, base_url: str, saved: dict, ingest_tiers=None):
        _mock_watch_apis(
            page,
            profiles=[FAKE_PROFILE],
            on_update=lambda body: saved.update(body),
            ingest_tiers=ingest_tiers,
        )
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-source-row']").first).to_be_visible()

    def test_ring_bounds_are_shown_only_for_an_unbuffered_source(self, page: Page, base_url: str):
        saved: dict = {}
        self._open(page, base_url, saved)
        expect(page.locator("[data-testid='watch-ring-fields']").first).to_be_hidden()

        page.reload()
        self._open(page, base_url, saved, ingest_tiers=self.RING_TIERS)
        expect(page.locator("[data-testid='watch-ring-fields']").first).to_be_visible()

    def test_an_edited_ring_size_is_saved(self, page: Page, base_url: str):
        saved: dict = {}
        self._open(page, base_url, saved, ingest_tiers=self.RING_TIERS)

        page.locator("[data-testid='watch-ring-seconds']").first.fill("45")
        page.locator("[data-testid='watch-ring-max-mb']").first.fill("64")
        page.get_by_role("button", name="Save", exact=True).first.click()

        wait_until(page, lambda: saved.get("sources"), "the save never reached the API")
        source = saved["sources"][0]
        assert source["ringSeconds"] == 45
        assert source["ringMaxMb"] == 64

    def test_a_save_preserves_the_stored_values_it_does_not_render(self, page: Page, base_url: str):
        """`bufferMode` has no input; the save must still carry it through unchanged."""
        saved: dict = {}
        self._open(page, base_url, saved)

        page.locator("[data-testid='watcher-name']").fill("Renamed")
        page.get_by_role("button", name="Save", exact=True).first.click()

        wait_until(page, lambda: saved.get("sources"), "the save never reached the API")
        source = saved["sources"][0]
        assert source["bufferMode"] == FAKE_PROFILE["sources"][0]["bufferMode"]
        assert source["ringSeconds"] == FAKE_PROFILE["sources"][0]["ringSeconds"]
        assert source["ringMaxMb"] == FAKE_PROFILE["sources"][0]["ringMaxMb"]


class TestIncludeChildLabels:
    """The label pickers reuse the shared multi-select's own hierarchy toggle.

    Stored as a flag and expanded when the watcher starts, so a label added to the tree
    later is covered without editing the rule.
    """

    def _open(self, page: Page, base_url: str, profile=None):
        _mock_watch_apis(page, profiles=[profile or FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-rule-row']").first).to_be_visible()

    def _toggle(self, page: Page, slot: str):
        host = page.locator(f"[data-testid='watch-rule-subject-{slot}']").first
        panel = host.locator(".ms-dropdown__panel")
        if not panel.is_visible():
            host.locator(".ms-dropdown__trigger").click()
        return panel.locator(".ms-hierarchy-toggle input[type='checkbox']")

    def test_the_subject_a_picker_offers_it(self, page: Page, base_url: str):
        self._open(page, base_url)
        expect(self._toggle(page, "a")).to_have_count(1)

    def test_it_is_off_by_default(self, page: Page, base_url: str):
        self._open(page, base_url)
        expect(self._toggle(page, "a")).not_to_be_checked()

    def test_a_saved_flag_loads_ticked(self, page: Page, base_url: str):
        self._open(page, base_url, profile=DESCENDANTS_PROFILE)
        expect(self._toggle(page, "a")).to_be_checked()

    def test_a_zone_subject_b_has_no_toggle(self, page: Page, base_url: str):
        """Zones have no hierarchy — the backend rejects the flag there."""
        self._open(page, base_url)
        expect(self._toggle(page, "b")).to_have_count(0)

    def test_a_label_subject_b_has_one(self, page: Page, base_url: str):
        self._open(page, base_url, profile=LABEL_SUBJECT_PROFILE)
        expect(self._toggle(page, "b")).to_have_count(1)

    def test_ticking_it_is_saved(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        self._toggle(page, "a").check()
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        rule = saved["rules"][0]
        assert rule["subjectAIncludeDescendants"] is True, rule
        # Stored as a flag, not as an expanded list.
        assert rule["subjectARefsJson"] == ["label-cat"], rule

    def test_switching_subject_b_to_a_zone_clears_the_flag(self, page: Page, base_url: str):
        """A leftover true would fail the save with a message about a hidden control."""
        saved = {}
        _mock_watch_apis(page, profiles=[LABEL_SUBJECT_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        self._toggle(page, "b").check()
        page.locator("[data-testid='watch-rule-subject-b-kind']").first.select_option("zone")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        assert saved["rules"][0]["subjectBIncludeDescendants"] is False, saved["rules"][0]


class TestActionScoping:
    """Per-rule and per-source action scope.

    This is what lets one watcher serve several cameras: the rule "cats using the litter
    box" is written once, and only the *recording* is narrowed to the camera pointed at
    it. Both pickers default to "All", so an action nobody has narrowed still fires
    everywhere — the behaviour every watcher written before scoping existed relies on.
    """

    def _open(self, page: Page, base_url: str):
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()

    def test_both_scope_pickers_are_shown(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        expect(page.locator("[data-testid='watch-action-rule-scope']").first).to_be_visible()
        expect(page.locator("[data-testid='watch-action-source-scope']").first).to_be_visible()

    def test_an_unscoped_action_reads_as_all(self, page: Page, base_url: str):
        """ "All" is the honest summary — an empty scope really does fire everywhere."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        # The trigger, not the container — the container's hidden panel has a "Select
        # All" row that would satisfy this assertion however the trigger read.
        trigger = page.locator("[data-testid='watch-action-source-scope'] .ms-dropdown__trigger").first
        expect(trigger).to_contain_text("All")

    def test_an_unscoped_action_saves_empty_lists(self, page: Page, base_url: str):
        """Empty means all, and it has to survive a round trip unchanged.

        Edits an unrelated field first: Save is disabled until the form is dirty, so a
        test that touches nothing can never click it.
        """
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)
        page.locator("[data-testid='watch-action-topic']").fill("cats/{rule}")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("actions")))
        assert saved["actions"][0]["ruleUuidsJson"] == []
        assert saved["actions"][0]["sourceUuidsJson"] == []

    def test_picking_a_source_scopes_the_action_to_it(self, page: Page, base_url: str):
        """The whole feature: record on this camera, not on the other four."""
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)

        scope = page.locator("[data-testid='watch-action-source-scope']").first
        scope.locator("button").first.click()
        scope.locator("input[type='checkbox']").first.check()
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("actions")))
        assert saved["actions"][0]["sourceUuidsJson"] == ["src-1"], saved["actions"][0]

    def test_a_saved_source_scope_loads_back(self, page: Page, base_url: str):
        scoped = {
            **FAKE_PROFILE,
            "actions": [{**FAKE_PROFILE["actions"][0], "sourceUuidsJson": ["src-1"]}],
        }
        _mock_watch_apis(page, profiles=[scoped])
        self._open(page, base_url)
        # The trigger button, not the container: the container also holds the (hidden)
        # panel, whose "Select All" row matches any assertion about the word "All".
        trigger = page.locator("[data-testid='watch-action-source-scope'] .ms-dropdown__trigger").first
        # Names the camera, rather than reading as unscoped — a scope that silently
        # displayed as "All" would be worse than no scoping at all.
        expect(trigger).to_contain_text("Litter cam")


class TestRuleSourceScope:
    """Per-source *rule* scope — the sharper sibling of action scope.

    Scoping the action leaves the event raised, logged and browsable on every camera and
    only silences the side effect. Scoping the rule stops the event existing, which is
    what "record a clip when a cat is present in this one room" actually needs: a cat is
    present in every room all day.
    """

    def _open(self, page: Page, base_url: str):
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-rule-row']").first).to_be_visible()

    def test_the_picker_is_shown_on_a_rule(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        expect(page.locator("[data-testid='watch-rule-source-scope']").first).to_be_visible()

    def test_an_unscoped_rule_reads_as_all_sources(self, page: Page, base_url: str):
        """The empty state must not read as the opposite of what it does."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        trigger = page.locator("[data-testid='watch-rule-source-scope'] .ms-dropdown__trigger").first
        expect(trigger).to_contain_text("All sources")

    def test_an_unscoped_rule_saves_an_empty_list(self, page: Page, base_url: str):
        """A nested write is a whole-list replace, so the field must always be sent."""
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)
        page.locator("[data-testid='watch-rule-name']").first.fill("cat in the litter box")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        assert saved["rules"][0]["sourceUuidsJson"] == [], saved["rules"][0]

    def test_picking_a_source_scopes_the_rule_to_it(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)

        scope = page.locator("[data-testid='watch-rule-source-scope']").first
        scope.locator("button").first.click()
        scope.locator("input[type='checkbox']").first.check()
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        assert saved["rules"][0]["sourceUuidsJson"] == ["src-1"], saved["rules"][0]

    def test_a_saved_scope_loads_back_naming_the_camera(self, page: Page, base_url: str):
        """A scope that silently reopened as "All" would be worse than no scoping."""
        scoped = {
            **FAKE_PROFILE,
            "rules": [{**FAKE_PROFILE["rules"][0], "sourceUuidsJson": ["src-1"]}],
        }
        _mock_watch_apis(page, profiles=[scoped])
        self._open(page, base_url)
        trigger = page.locator("[data-testid='watch-rule-source-scope'] .ms-dropdown__trigger").first
        expect(trigger).to_contain_text("Litter cam")

    def test_the_scope_survives_a_save_untouched(self, page: Page, base_url: str):
        """Load, edit something else, save — the scope must come back out unchanged."""
        saved = {}
        scoped = {
            **FAKE_PROFILE,
            "rules": [{**FAKE_PROFILE["rules"][0], "sourceUuidsJson": ["src-1"]}],
        }
        _mock_watch_apis(page, profiles=[scoped], on_update=lambda body: saved.update(body))
        self._open(page, base_url)
        page.locator("[data-testid='watch-rule-name']").first.fill("cat in the litter box")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        assert saved["rules"][0]["sourceUuidsJson"] == ["src-1"], saved["rules"][0]


class TestLiveImageFeed:
    """The per-frame annotated MQTT feed."""

    def _open(self, page: Page, base_url: str):
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()

    def test_the_image_action_kind_is_offered(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        kind = page.locator("[data-testid='watch-action-kind']").first
        kind.select_option("publish_mqtt_image")
        expect(page.locator("[data-testid='watch-action-image-topic']")).to_be_visible()

    def test_the_image_topic_defaults_to_a_source_keyed_one(self, page: Page, base_url: str):
        """The frame belongs to the camera, not to whichever rule fired."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        page.locator("[data-testid='watch-action-kind']").first.select_option("publish_mqtt_image")
        expect(page.locator("[data-testid='watch-action-image-topic']")).to_have_value("{watcher}/{source}/image")

    def test_the_per_frame_trigger_is_offered(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        trigger = page.locator("[data-testid='watch-action-trigger']").first
        trigger.select_option("frame_analysed")
        expect(trigger).to_have_value("frame_analysed")

    def test_the_rule_scope_is_hidden_for_a_per_frame_action(self, page: Page, base_url: str):
        """A frame belongs to no rule, so there is nothing to scope it against."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        page.locator("[data-testid='watch-action-trigger']").first.select_option("frame_analysed")
        expect(page.locator("[data-testid='watch-action-rule-scope']").first).to_be_hidden()
        # The source scope stays: it is exactly how a live feed is limited to one camera.
        expect(page.locator("[data-testid='watch-action-source-scope']").first).to_be_visible()

    def test_a_live_feed_action_saves_its_params(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)

        page.locator("[data-testid='watch-action-kind']").first.select_option("publish_mqtt_image")
        page.locator("[data-testid='watch-action-trigger']").first.select_option("frame_analysed")
        page.locator("[data-testid='watch-action-image-max-fps']").fill("2")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("actions")))
        action = saved["actions"][0]
        assert action["kind"] == "publish_mqtt_image"
        assert action["trigger"] == "frame_analysed"
        assert action["paramsJson"]["max_fps"] == 2
        assert action["paramsJson"]["annotate"] is True
        # Not sent even though the trigger changed after a scope could have been picked.
        assert action["ruleUuidsJson"] == []


class TestSourceMqttName:
    """A watcher's per-source MQTT name.

    The name that reads well in the Sources list is not necessarily the short, stable
    topic level a Home Assistant automation should be written against.
    """

    def _open(self, page: Page, base_url: str):
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-source-row']").first).to_be_visible()

    def test_the_field_is_shown(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        expect(page.locator("[data-testid='watch-source-mqtt-name']").first).to_be_visible()

    def test_it_is_blank_by_default_and_hints_the_sources_own_name(self, page: Page, base_url: str):
        """The placeholder is what makes "blank means the source's name" self-evident."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        field = page.locator("[data-testid='watch-source-mqtt-name']").first
        expect(field).to_have_value("")
        expect(field).to_have_attribute("placeholder", "Litter cam")

    def test_a_saved_name_loads_back(self, page: Page, base_url: str):
        named = {
            **FAKE_PROFILE,
            "sources": [{**FAKE_PROFILE["sources"][0], "mqttName": "litterbox"}],
        }
        _mock_watch_apis(page, profiles=[named])
        self._open(page, base_url)
        expect(page.locator("[data-testid='watch-source-mqtt-name']").first).to_have_value("litterbox")

    def test_it_is_saved(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)
        page.locator("[data-testid='watch-source-mqtt-name']").first.fill("litterbox")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("sources")))
        assert saved["sources"][0]["mqttName"] == "litterbox"

    def test_a_blank_name_is_sent_as_no_override(self, page: Page, base_url: str):
        """ "" and null must not be two states that behave alike but compare differently."""
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)
        page.locator("[data-testid='watch-source-mqtt-name']").first.fill("   ")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("sources")))
        assert saved["sources"][0]["mqttName"] is None


class TestTemplateHelp:
    """The placeholder vocabulary used to be reachable only as the topic input's
    `title`, which is invisible until you already suspect it is there — so the whole
    templating feature read as a free-text box you had to guess at."""

    def _open(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-topic']")).to_be_visible()

    def test_a_visible_affordance_sits_beside_the_topic_field(self, page: Page, base_url: str):
        self._open(page, base_url)
        expect(page.locator("[data-testid='watch-template-help-btn']").first).to_be_visible()

    def test_the_panel_is_closed_until_asked_for(self, page: Page, base_url: str):
        self._open(page, base_url)
        expect(page.locator("[data-testid='watch-template-help-panel']").first).to_be_hidden()

    def test_opening_it_lists_the_placeholders_with_meanings(self, page: Page, base_url: str):
        self._open(page, base_url)
        page.locator("[data-testid='watch-template-help-btn']").first.click()
        panel = page.locator("[data-testid='watch-template-help-panel']").first
        expect(panel).to_be_visible()
        expect(panel).to_contain_text("{subject}")
        # The meaning, not just the name — which for {source} vs {source_config_name} is
        # the only thing worth knowing.
        expect(panel).to_contain_text("MQTT name")

    def test_clicking_a_placeholder_inserts_it(self, page: Page, base_url: str):
        """The list is the fastest way to write a template, not documentation about one."""
        self._open(page, base_url)
        topic = page.locator("[data-testid='watch-action-topic']")
        topic.fill("")
        page.locator("[data-testid='watch-template-help-btn']").first.click()
        page.locator("[data-testid='watch-template-help-panel'] .watch-template-help__item").first.click()
        expect(topic).not_to_have_value("")

    def test_inserting_updates_the_live_example(self, page: Page, base_url: str):
        """A programmatic value change fires no input event on its own."""
        self._open(page, base_url)
        topic = page.locator("[data-testid='watch-action-topic']")
        topic.fill("")
        page.locator("[data-testid='watch-template-help-btn']").first.click()
        panel = page.locator("[data-testid='watch-template-help-panel']").first
        panel.locator(".watch-template-help__item", has_text="{subject}").first.click()
        expect(page.locator("[data-testid='watch-action-topic-example']")).to_contain_text("tabby")

    def test_clicking_outside_closes_it(self, page: Page, base_url: str):
        self._open(page, base_url)
        page.locator("[data-testid='watch-template-help-btn']").first.click()
        expect(page.locator("[data-testid='watch-template-help-panel']").first).to_be_visible()
        page.locator("[data-testid='watcher-name']").click()
        expect(page.locator("[data-testid='watch-template-help-panel']").first).to_be_hidden()

    def test_the_image_action_gets_one_too(self, page: Page, base_url: str):
        self._open(page, base_url)
        page.locator("[data-testid='watch-action-kind']").first.select_option("publish_mqtt_image")
        expect(page.locator("[data-testid='watch-template-help-btn']").first).to_be_visible()


class TestClipTriggerIsConstrained:
    """ "Record clip" only works on the closing edge, so it only offers that trigger.

    `RecordClipAction.run` returns early unless the phase is END — the window to cut is
    not known until the event closes. Offering the other two triggers let a watcher be
    saved with a clip action that could never fire: no error at save time, no clip at run
    time, and nothing on screen to explain why.
    """

    def _open(self, page: Page, base_url: str):
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()

    def _kind(self, page: Page):
        return page.locator("[data-testid='watch-action-kind']").first

    def _trigger(self, page: Page):
        return page.locator("[data-testid='watch-action-trigger']").first

    def test_a_clip_action_offers_only_the_end_trigger(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        self._kind(page).select_option("record_clip")
        options = self._trigger(page).locator("option")
        expect(options).to_have_count(1)
        expect(options.first).to_have_attribute("value", "event_ended")

    def test_an_unconstrained_action_still_offers_every_trigger(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        self._kind(page).select_option("publish_mqtt")
        expect(self._trigger(page).locator("option")).to_have_count(3)

    def test_switching_to_clip_coerces_an_impossible_trigger(self, page: Page, base_url: str):
        """Otherwise the row keeps a stored value the dropdown no longer offers."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        self._trigger(page).select_option("frame_analysed")
        self._kind(page).select_option("record_clip")
        expect(self._trigger(page)).to_have_value("event_ended")

    def test_switching_away_restores_the_full_list(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        self._kind(page).select_option("record_clip")
        self._kind(page).select_option("save_snapshot")
        expect(self._trigger(page).locator("option")).to_have_count(3)

    def test_the_shortened_list_explains_itself(self, page: Page, base_url: str):
        """A dropdown that silently has one entry reads as an unbuilt feature."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        self._kind(page).select_option("record_clip")
        help_btn = page.locator("[data-testid='watch-action-trigger-help-btn']").first
        expect(help_btn).to_be_visible()

        help_btn.click()
        panel = page.locator("[data-testid='watch-action-trigger-help-panel']").first
        expect(panel).to_be_visible()
        expect(panel).to_contain_text("only be cut when the event ends")

    def test_no_help_marker_when_nothing_is_constrained(self, page: Page, base_url: str):
        """A `(?)` beside a full dropdown promises information that isn't there."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        self._kind(page).select_option("publish_mqtt")
        expect(page.locator("[data-testid='watch-action-trigger-help-btn']")).to_have_count(0)


class TestTimelapseAction:
    """ "Save timelapse" — one frame every N seconds into per-day directories.

    Constrained to the per-frame trigger for the opposite reason "Record clip" is
    constrained to the end edge: a timelapse has to keep shooting through the hours when
    no event is open, and an event trigger would give it nothing to shoot on.
    """

    def _open(self, page: Page, base_url: str):
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()

    def _kind(self, page: Page):
        return page.locator("[data-testid='watch-action-kind']").first

    def test_it_offers_only_the_per_frame_trigger(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        self._kind(page).select_option("save_timelapse")
        options = page.locator("[data-testid='watch-action-trigger']").first.locator("option")
        expect(options).to_have_count(1)
        expect(options.first).to_have_attribute("value", "frame_analysed")

    def test_switching_to_it_coerces_an_impossible_trigger(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        page.locator("[data-testid='watch-action-trigger']").first.select_option("event_ended")
        self._kind(page).select_option("save_timelapse")
        expect(page.locator("[data-testid='watch-action-trigger']").first).to_have_value("frame_analysed")

    def test_the_shortened_list_explains_itself(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        self._kind(page).select_option("save_timelapse")
        page.locator("[data-testid='watch-action-trigger-help-btn']").first.click()
        panel = page.locator("[data-testid='watch-action-trigger-help-panel']").first
        expect(panel).to_contain_text("keep shooting through the hours when nothing happens")

    def test_the_per_day_layout_is_shown_before_saving(self, page: Page, base_url: str):
        """The day subdirectory is the action's doing, so nothing else reveals it."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        self._kind(page).select_option("save_timelapse")
        hint = page.locator("[data-testid='watch-action-timelapse-layout']").first
        expect(hint).to_contain_text("timelapse/")
        page.locator("[data-testid='watch-action-dir']").first.fill("cams/front")
        expect(hint).to_contain_text("cams/front/")

    def test_it_saves_its_params(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)

        self._kind(page).select_option("save_timelapse")
        page.locator("[data-testid='watch-action-dir']").first.fill("cams/front")
        page.locator("[data-testid='watch-action-timelapse-interval']").first.fill("60")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("actions")))
        action = saved["actions"][0]
        assert action["kind"] == "save_timelapse"
        assert action["trigger"] == "frame_analysed"
        assert action["paramsJson"]["output_dir"] == "cams/front"
        assert action["paramsJson"]["interval_s"] == 60
        # Defaults the row never touched still reach the backend, so the action does not
        # silently fall back to a different set of them.
        assert action["paramsJson"]["full_resolution"] is True
        assert action["paramsJson"]["annotate"] is False

    def test_browse_opens_the_shared_directory_picker(self, page: Page, base_url: str):
        """The picker is what makes a directory that does not exist yet creatable."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        page.route(
            lambda url: url.split("?")[0].rstrip("/").endswith("/api/images/list"),
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "success",
                        "files": [{"name": "timelapse", "path": "/timelapse", "is_dir": True}],
                        "has_more": False,
                    }
                ),
            ),
        )
        self._open(page, base_url)

        self._kind(page).select_option("save_timelapse")
        page.locator("[data-testid='watch-action-dir-browse']").first.click()
        # "New Folder" is the affordance this reuse exists for — a timelapse directory
        # normally does not exist until the user makes it here.
        expect(page.get_by_text("Select Directory").last).to_be_visible()
        expect(page.get_by_role("button", name="+ New Folder").last).to_be_visible()


class TestSnapshotDirectoryField:
    """ "Save snapshot" gets the same directory affordances as the timelapse action.

    It was a bare text box holding an absolute container path — no way to see what
    existed, no way to create a folder, and no hint that anything could be substituted
    into it.
    """

    def _open(self, page: Page, base_url: str):
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()
        page.locator("[data-testid='watch-action-kind']").first.select_option("save_snapshot")

    def test_it_has_a_browse_button(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        expect(page.locator("[data-testid='watch-action-dir-browse']").first).to_be_visible()

    def test_it_has_a_token_help_marker(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        expect(page.locator("[data-testid='watch-template-help-btn']").first).to_be_visible()

    def test_the_date_token_is_offered_and_inserts_at_the_caret(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        field = page.locator("[data-testid='watch-action-dir']").first
        field.fill("snaps/")
        page.locator("[data-testid='watch-template-help-btn']").first.click()
        panel = page.locator("[data-testid='watch-template-help-panel']").first
        expect(panel).to_be_visible()
        panel.locator("button", has_text="{date}").first.click()
        expect(field).to_have_value("snaps/{date}")

    def test_the_preview_shows_where_a_snapshot_lands(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        field = page.locator("[data-testid='watch-action-dir']").first
        hint = page.locator("[data-testid='watch-action-snapshot-example']").first
        field.fill("snaps/{source}")
        expect(hint).to_contain_text("snaps/back-room/2026-08-18/")

    def test_placing_the_date_drops_the_automatic_day_level_from_the_preview(self, page: Page, base_url: str):
        """The preview must not claim a folder the action will not create."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        field = page.locator("[data-testid='watch-action-dir']").first
        hint = page.locator("[data-testid='watch-action-snapshot-example']").first
        field.fill("snaps/{date}/{source}")
        expect(hint).to_contain_text("snaps/2026-08-18/back-room/14-30-00_")

    def test_it_saves_the_folder_template(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)

        page.locator("[data-testid='watch-action-dir']").first.fill("snaps/{source}")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("actions")))
        assert saved["actions"][0]["paramsJson"]["output_dir"] == "snaps/{source}"


class TestHelpPopoverStaysOnScreen:
    """The `(?)` panels used to open ~23rem to the left of a 1.35em button.

    `position: absolute; right: 0` inside the button's own wrapper meant the panel hung
    off the left of the viewport for every field that was not near the right edge of a
    wide window — visible in the DOM, unreadable on screen.
    """

    def _open(self, page: Page, base_url: str):
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()

    def _assert_within_viewport(self, page: Page, panel):
        box = panel.bounding_box()
        assert box is not None, "panel should be laid out"
        width = page.evaluate("() => window.innerWidth")
        height = page.evaluate("() => window.innerHeight")
        assert box["x"] >= 0, f"panel starts {box['x']}px off the left edge"
        assert box["y"] >= 0, f"panel starts {box['y']}px off the top edge"
        assert box["x"] + box["width"] <= width + 1, "panel runs off the right edge"
        assert box["y"] + box["height"] <= height + 1, "panel runs off the bottom edge"

    def test_the_topic_help_opens_inside_the_viewport(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        page.locator("[data-testid='watch-template-help-btn']").first.click()
        panel = page.locator("[data-testid='watch-template-help-panel']").first
        expect(panel).to_be_visible()
        self._assert_within_viewport(page, panel)

    def test_it_stays_on_screen_in_a_narrow_window(self, page: Page, base_url: str):
        """The case that was worst: no room to hang left, and none to hang right either."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        page.set_viewport_size({"width": 700, "height": 800})
        self._open(page, base_url)

        page.locator("[data-testid='watch-template-help-btn']").first.click()
        panel = page.locator("[data-testid='watch-template-help-panel']").first
        expect(panel).to_be_visible()
        self._assert_within_viewport(page, panel)

    def test_the_trigger_help_opens_inside_the_viewport(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        page.locator("[data-testid='watch-action-kind']").first.select_option("record_clip")
        page.locator("[data-testid='watch-action-trigger-help-btn']").first.click()
        panel = page.locator("[data-testid='watch-action-trigger-help-panel']").first
        expect(panel).to_be_visible()
        self._assert_within_viewport(page, panel)

    def test_clicking_the_marker_again_closes_it(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        btn = page.locator("[data-testid='watch-template-help-btn']").first
        btn.click()
        expect(page.locator("[data-testid='watch-template-help-panel']").first).to_be_visible()
        btn.click()
        expect(page.locator("[data-testid='watch-template-help-panel']")).to_have_count(0)


class TestActionName:
    """An action can carry a note, so a list of them is readable.

    Three "Save snapshot" rows scoped to different cameras are otherwise
    indistinguishable, and what tells them apart is spread across a dense row of
    dropdowns.
    """

    def _open(self, page: Page, base_url: str):
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()

    def test_the_field_is_offered_on_every_action(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        expect(page.locator("[data-testid='watch-action-name']").first).to_be_visible()

    def test_it_saves_the_note(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)

        page.locator("[data-testid='watch-action-name']").first.fill("Shout when the cat arrives")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("actions")))
        assert saved["actions"][0]["name"] == "Shout when the cat arrives"

    def test_a_blank_note_is_sent_as_null(self, page: Page, base_url: str):
        """An empty string would render as a blank label rather than as "unnamed"."""
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)

        page.locator("[data-testid='watch-action-name']").first.fill("   ")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("actions")))
        assert saved["actions"][0]["name"] is None


class TestTimelapseFolderTokens:
    """One timelapse action covers every camera in the watcher, so the folder field has
    to be able to split them — and has to say so when it isn't."""

    def _open(self, page: Page, base_url: str):
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()
        page.locator("[data-testid='watch-action-kind']").first.select_option("save_timelapse")

    def test_the_token_help_is_offered(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        expect(page.locator("[data-testid='watch-template-help-btn']").first).to_be_visible()

    def test_the_preview_shows_a_folder_per_camera(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        page.locator("[data-testid='watch-action-dir']").first.fill("timelapse/{source}")
        hint = page.locator("[data-testid='watch-action-timelapse-layout']").first
        expect(hint).to_contain_text("timelapse/back-room/2026-08-18/")

    def test_it_says_when_every_camera_shares_one_folder(self, page: Page, base_url: str):
        """The whole trap: without {source} four cameras interleave, and nothing on
        screen said so."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        page.locator("[data-testid='watch-action-dir']").first.fill("timelapse")
        hint = page.locator("[data-testid='watch-action-timelapse-layout']").first
        expect(hint).to_have_attribute("title", re.compile("add \\{source\\}"))

    def test_it_saves_the_template(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)

        page.locator("[data-testid='watch-action-dir']").first.fill("timelapse/{source}")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("actions")))
        assert saved["actions"][0]["paramsJson"]["output_dir"] == "timelapse/{source}"


class TestMatchPreview:
    """The match-threshold sandbox.

    The threshold is the one number in this editor with no intuition behind it — a
    fraction of an area the user never sees — so what these tests protect is that the
    readout actually responds to the boxes and that "use these settings" writes back the
    values the user just dragged into place.
    """

    def _open_preview(self, page: Page, base_url: str, profile=None):
        _mock_watch_apis(page, profiles=[profile or FAKE_PROFILE])
        # The modal samples the DB for a crop of each label. Answered empty so the boxes
        # stay plain rectangles and their geometry is exactly what the test set up.
        page.route(
            lambda url: url.split("?")[0].rstrip("/").endswith("/api/images/review"),
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "images": [], "hasMore": False, "totalCount": 0}),
            ),
        )
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-rule-preview']").first.click()
        expect(page.locator("[data-testid='overlap-preview']")).to_be_visible()

    def _drag_a_onto_b(self, page: Page):
        """Drag subject A's box to the centre of B's, which contains it completely."""
        box_a = page.locator("[data-testid='overlap-preview-box-a']").bounding_box()
        box_b = page.locator("[data-testid='overlap-preview-box-b']").bounding_box()
        page.mouse.move(box_a["x"] + box_a["width"] / 2, box_a["y"] + box_a["height"] / 2)
        page.mouse.down()
        # Two moves: a single one from press to target can be swallowed as a click.
        page.mouse.move(box_b["x"] + box_b["width"] / 2, box_b["y"] + box_b["height"] / 2, steps=10)
        page.mouse.up()

    def test_the_button_opens_the_preview(self, page: Page, base_url: str):
        self._open_preview(page, base_url)
        expect(page.locator("[data-testid='overlap-preview-stage']")).to_be_visible()
        expect(page.locator("[data-testid='overlap-preview-box-a']")).to_be_visible()
        expect(page.locator("[data-testid='overlap-preview-box-b']")).to_be_visible()

    def test_the_verdict_follows_the_boxes(self, page: Page, base_url: str):
        """Boxes barely touching do not match at 0.6; fully inside does."""
        self._open_preview(page, base_url)
        badge = page.locator("[data-testid='overlap-preview-badge']")
        expect(badge).to_have_text("no match")
        self._drag_a_onto_b(page)
        expect(badge).to_have_text("FIRES")
        expect(page.locator("[data-testid='overlap-preview-score']")).to_contain_text("100%")

    def test_a_nearby_margin_can_make_disjoint_boxes_match(self, page: Page, base_url: str):
        """The whole point of the margin: "close" becomes a match without moving a box."""
        self._open_preview(page, base_url)
        page.locator("[data-testid='overlap-preview-relation']").select_option("overlaps")
        # The boxes start barely lapping — about 27% of the smaller one.
        page.locator("[data-testid='overlap-preview-threshold']").fill("0.5")
        expect(page.locator("[data-testid='overlap-preview-badge']")).to_have_text("no match")

        # One subject-width of slack takes the same boxes past 60%.
        page.locator("[data-testid='overlap-preview-margin']").fill("1")
        expect(page.locator("[data-testid='overlap-preview-badge']")).to_have_text("FIRES")

    def test_the_margin_note_explains_the_units(self, page: Page, base_url: str):
        """A bare "0.75" says nothing; the note is where subject-widths are named."""
        self._open_preview(page, base_url)
        page.locator("[data-testid='overlap-preview-margin']").fill("0.75")
        expect(page.locator("[data-testid='overlap-preview-margin-note']")).to_contain_text("smaller box")

    def test_is_near_takes_no_margin(self, page: Page, base_url: str):
        """It compares centre points, which do not grow — and the backend rejects one."""
        self._open_preview(page, base_url)
        page.locator("[data-testid='overlap-preview-relation']").select_option("distance")
        expect(page.locator("[data-testid='overlap-preview-margin']")).to_be_hidden()

    def test_applying_writes_the_values_back_to_the_rule(self, page: Page, base_url: str):
        self._open_preview(page, base_url)
        page.locator("[data-testid='overlap-preview-relation']").select_option("overlaps")
        page.locator("[data-testid='overlap-preview-threshold']").fill("0.35")
        page.locator("[data-testid='overlap-preview-margin']").fill("0.5")
        page.locator("[data-testid='overlap-preview-apply']").click()

        expect(page.locator("[data-testid='overlap-preview']")).to_have_count(0)
        row = page.locator("[data-testid='watch-rule-row']").first
        expect(row.locator("[data-testid='watch-rule-relation']")).to_have_value("overlaps")
        inputs = row.locator(".watch-rule-tuning input")
        expect(inputs.nth(0)).to_have_value("0.35")
        expect(inputs.nth(1)).to_have_value("0.5")

    def test_the_margin_is_saved(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        row = page.locator("[data-testid='watch-rule-row']").first
        row.locator(".watch-rule-tuning input").nth(1).fill("0.8")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        assert saved["rules"][0]["margin"] == 0.8, saved["rules"][0]

    def test_a_relation_that_ignores_the_margin_saves_zero(self, page: Page, base_url: str):
        """`present` compares no boxes, and the backend rejects a margin there — so a
        value typed under another relation must not survive into the payload."""
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        row = page.locator("[data-testid='watch-rule-row']").first
        row.locator(".watch-rule-tuning input").nth(1).fill("0.8")
        row.locator("[data-testid='watch-rule-relation']").select_option("present")
        expect(row.locator("[data-testid='watch-rule-margin-field']")).to_be_hidden()
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        assert saved["rules"][0]["margin"] == 0, saved["rules"][0]


class TestWatcherSections:
    def test_each_group_is_its_own_panel(self, page: Page, base_url: str):
        """Sources, Tracking, Rules and Actions all render as near-identical rows, so
        without a panel apiece the editor reads as one long undifferentiated list."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-section']")).to_have_count(4)
        for box in ("watch-sources-box", "watch-rules-box", "watch-actions-box", "watch-lifetimes-box"):
            expect(page.locator(f"[data-testid='watch-section'] [data-testid='{box}']")).to_have_count(1)


class TestThresholdCaption:
    """The threshold field is one number with six meanings, so its caption is not fixed.

    Under `present` there is no second box at all — `evaluate_predicate` compares the
    number against the best track's *detection confidence* — and captioning that `match`
    described geometry that is never measured.
    """

    def _rule_row(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        return page.locator("[data-testid='watch-rule-row']").first

    def test_present_captions_the_threshold_as_confidence(self, page: Page, base_url: str):
        row = self._rule_row(page, base_url)
        row.locator("[data-testid='watch-rule-relation']").select_option("present")
        field = row.locator("[data-testid='watch-rule-threshold-field']")
        expect(field.locator(".watch-field__label")).to_have_text("conf")
        assert "confidence" in (field.get_attribute("title") or "")

    def test_a_geometric_relation_still_reads_as_match(self, page: Page, base_url: str):
        row = self._rule_row(page, base_url)
        row.locator("[data-testid='watch-rule-relation']").select_option("overlaps")
        field = row.locator("[data-testid='watch-rule-threshold-field']")
        expect(field.locator(".watch-field__label")).to_have_text("match")
        assert "smaller box" in (field.get_attribute("title") or "")

    def test_distance_says_within_because_higher_is_looser(self, page: Page, base_url: str):
        """Its threshold is an upper bound, so raising it loosens the rule — the
        opposite of every other relation, which is worth saying in the caption."""
        row = self._rule_row(page, base_url)
        row.locator("[data-testid='watch-rule-relation']").select_option("distance")
        field = row.locator("[data-testid='watch-rule-threshold-field']")
        expect(field.locator(".watch-field__label")).to_have_text("within")

    def test_the_caption_follows_the_relation_back(self, page: Page, base_url: str):
        """Switching away from `present` must not leave the confidence caption behind."""
        row = self._rule_row(page, base_url)
        relation = row.locator("[data-testid='watch-rule-relation']")
        relation.select_option("present")
        relation.select_option("contains")
        expect(row.locator("[data-testid='watch-rule-threshold-field'] .watch-field__label")).to_have_text("match")

    def test_renaming_the_caption_does_not_move_the_value(self, page: Page, base_url: str):
        """Caption only: the field still saves as `threshold`, whatever it is called."""
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        row = page.locator("[data-testid='watch-rule-row']").first
        row.locator("[data-testid='watch-rule-relation']").select_option("present")
        row.locator("[data-testid='watch-rule-threshold']").fill("0.85")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        assert saved["rules"][0]["threshold"] == 0.85, saved["rules"][0]


class TestEverySettableThingExplainsItself:
    """A watcher is configured almost entirely through unlabelled short captions —
    `fps`, `ring MB`, `ends` — so a hover explanation is not a nicety here.

    Asserted structurally rather than field by field: the point is that *nothing* is
    left bare, and a per-field list would be a test that passes while the next control
    added ships with no help at all.
    """

    def _open(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()

    def test_no_field_in_the_editor_hovers_blank(self, page: Page, base_url: str):
        self._open(page, base_url)
        bare = page.evaluate(
            """() => [...document.querySelectorAll('#watcher-management-box .watch-field')]
                .filter(f => !(f.title || '').trim() && !f.hidden)
                .map(f => f.querySelector('.watch-field__label')?.textContent || '?')"""
        )
        assert bare == [], f"watch fields with no tooltip: {bare}"

    def test_the_rule_sentence_slots_explain_themselves(self, page: Page, base_url: str):
        """The three multi-selects carry no caption at all — the sentence around them is
        the label — so the tooltip is the only place they can be explained."""
        self._open(page, base_url)
        for testid in (
            "watch-rule-subject-a",
            "watch-rule-subject-b",
            "watch-rule-source-scope",
        ):
            title = page.locator(f"[data-testid='{testid}']").first.get_attribute("title")
            assert title and title.strip(), f"{testid} has no tooltip"

    def test_the_action_scope_pickers_explain_empty_means_all(self, page: Page, base_url: str):
        """"All" is a summary a user reads as "nothing chosen"; the tooltip is where the
        difference between scoping a rule and scoping an action lives."""
        self._open(page, base_url)
        for testid in ("watch-action-rule-scope", "watch-action-source-scope"):
            title = page.locator(f"[data-testid='{testid}']").first.get_attribute("title")
            assert title and "all" in title.lower(), f"{testid}: {title!r}"

    def test_the_add_buttons_say_what_they_add(self, page: Page, base_url: str):
        self._open(page, base_url)
        for testid in (
            "watch-add-source",
            "watch-add-rule",
            "watch-add-action",
            "watch-add-lifetime",
        ):
            title = page.locator(f"[data-testid='{testid}']").get_attribute("title")
            assert title and title.strip(), f"{testid} has no tooltip"
