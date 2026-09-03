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

import copy
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


def _real_action_schemas() -> list:
    """The registry's own declarations, as `GET /api/watch-profiles/actions` returns them.

    Built by importing the actions rather than hand-copied into this file, and that is
    the whole point: a hand-written fixture is a third list to keep in step with the
    action and its form, and it would go stale in exactly the way that made these tests
    necessary — silently, and in the direction that makes them pass.

    Keys stay snake_case; the API camelises only the envelope around them.
    """
    import apps.helpers.actions  # noqa: F401  — populates the registry by autodiscovery
    from apps.helpers.actions.Registry import action_registry

    return [
        {"kind": kind, "displayName": cls.display_name(), "paramsSchema": cls.params_schema()}
        for kind, cls in sorted(action_registry.items())
    ]


ACTION_SCHEMAS = _real_action_schemas()

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
    action_schemas: list | None = None,
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

    # Same exclusion, same reason as `/trackers`: answered with a profile, the pane's
    # action rows would render from an empty schema and show no fields at all.
    page.route(
        lambda url: url.rstrip("/").endswith("/api/watch-profiles/actions"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "actions": action_schemas or ACTION_SCHEMAS}),
        ),
    )

    page.route(
        lambda url: (
            "/api/watch-profiles/" in url
            and not url.endswith("/validate")
            and not url.rstrip("/").endswith("/trackers")
            and not url.rstrip("/").endswith("/actions")
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
    page.wait_for_selector("#watcher-management-box .wp-list-panel")


class TestWatchersPane:
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


class TestHomeAssistantPublishFlags:
    """`publishToHa` is a different question from `enabled`, and the payload has to keep
    them apart: `enabled` off stops the rule running, this only stops it reaching Home
    Assistant. A save that conflated them would silently stop recording."""

    def test_unpublishing_a_rule_leaves_it_enabled(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()

        page.locator("[data-testid='watch-rule-publish']").first.uncheck()
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        rule = saved["rules"][0]
        assert rule["publishToHa"] is False, rule
        assert rule["enabled"] is True, rule

    def test_unpublishing_a_camera_leaves_the_rule_publishing(self, page: Page, base_url: str):
        """The two flags are independent axes — silencing a camera must not silence the
        rules that run on it, which is what keeps a rule-based automation working when a
        camera is taken out of Home Assistant."""
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()

        page.locator("[data-testid='watch-source-publish']").first.uncheck()
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("sources")))
        assert saved["sources"][0]["publishToHa"] is False, saved["sources"][0]
        assert saved["rules"][0]["publishToHa"] is True, saved["rules"][0]

    def test_a_save_that_ignores_the_flags_still_sends_them_as_publishing(self, page: Page, base_url: str):
        """Two regressions in one, both silent.

        FAKE_PROFILE predates these fields, exactly like a row saved before the
        migration — absent has to read as "publish", or upgrading would empty everyone's
        Home Assistant. And a profile saves as a whole-list replace, so a flag the editor
        forgets to send is a flag the save resets; editing something unrelated must not
        quietly turn publishing on or off.
        """
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()

        expect(page.locator("[data-testid='watch-rule-publish']").first).to_be_checked()
        expect(page.locator("[data-testid='watch-source-publish']").first).to_be_checked()

        # Dirty the form via a field that has nothing to do with publishing — Save stays
        # disabled until something changes, and that unrelated edit is the scenario.
        page.locator("[data-testid='watch-rule-name']").first.fill("cat at the litter box")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        assert saved["rules"][0]["publishToHa"] is True, saved["rules"][0]
        assert saved["sources"][0]["publishToHa"] is True, saved["sources"][0]


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

    def test_clicking_the_first_point_again_closes_the_polygon(self, page: Page, base_url: str):
        """Clicking back on the start vertex is the only close affordance a user finds
        without being told; a click anywhere else must still just add a vertex."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-zones-btn']").first.click()
        page.locator("[data-testid='watch-zones-kind']").select_option("polygon")

        canvas = page.locator("[data-testid='watch-zones-canvas']")
        box = canvas.bounding_box()
        for dx, dy in [(60, 60), (220, 60), (220, 200)]:
            page.mouse.click(box["x"] + dx, box["y"] + dy)

        # A fourth point away from the start does not close it.
        page.mouse.click(box["x"] + 60, box["y"] + 200)
        expect(page.locator("[data-testid='watch-zone-row']")).to_have_count(2)

        # Back on the first point, it does.
        page.mouse.click(box["x"] + 62, box["y"] + 62)
        expect(page.locator("[data-testid='watch-zone-row']")).to_have_count(3)
        # The list renders zones before masks, so the new one is the last *zone* row.
        expect(page.locator("[data-testid='watch-zone-row'][data-role='zone']").last).to_contain_text("poly")

    def test_right_click_closes_the_polygon(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-zones-btn']").first.click()
        page.locator("[data-testid='watch-zones-kind']").select_option("polygon")

        canvas = page.locator("[data-testid='watch-zones-canvas']")
        box = canvas.bounding_box()
        for dx, dy in [(60, 60), (220, 60), (140, 200)]:
            page.mouse.click(box["x"] + dx, box["y"] + dy)
        expect(page.locator("[data-testid='watch-zone-row']")).to_have_count(2)

        page.mouse.click(box["x"] + 140, box["y"] + 200, button="right")
        expect(page.locator("[data-testid='watch-zone-row']")).to_have_count(3)

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
        # Only the controls actually on screen. The sentence legitimately carries hidden
        # slots — subject B is hidden for `present`, the motion zone picker for every
        # relation but `motion` — and a hidden element reports a zero rect, which reads
        # as "wrapped to the top of the page" rather than "not rendered".
        tops = page.locator(".watch-rule-sentence select, .watch-rule-sentence .ms-dropdown__trigger").evaluate_all(
            "els => els.filter(e => e.offsetParent !== null)"
            "        .map(e => Math.round(e.getBoundingClientRect().top))"
        )
        assert len(tops) >= 4, f"expected the sentence's controls, got {tops}"
        # Rounded tops: a dropdown trigger and a select of slightly different heights are
        # still on one line, and demanding identical pixels would fail on that alone.
        assert max(tops) - min(tops) <= 6, f"the rule sentence wrapped onto several lines: tops={tops}"

    def test_a_rule_row_does_not_fill_the_screen(self, page: Page, base_url: str):
        """175px, up from 150: the row now carries a summary line that states what its
        tuning numbers do, and a line of prose is 25px whichever way it is written.

        That is a deliberate purchase, not drift. The numbers alone hid a real behaviour
        — a 30s cap on a `present` rule recorded a clip every thirty seconds while the
        car stayed parked, and nothing on screen connected the two fields that produced
        it. The bound still catches what it was written for: a control going full-width
        and stacking costs ~35px and would clear this on its own.
        """
        self._open_profile(page, base_url)
        box = page.locator("[data-testid='watch-rule-row']").first.bounding_box()
        assert box["height"] < 175, f"a single rule row is {box['height']}px tall"

    def test_source_and_action_rows_are_single_line(self, page: Page, base_url: str):
        self._open_profile(page, base_url)
        for testid in ("watch-source-row", "watch-action-row"):
            box = page.locator(f"[data-testid='{testid}']").first.bounding_box()
            assert box["height"] < 90, f"{testid} is {box['height']}px tall"

    def test_a_small_profile_fits_without_scrolling(self, page: Page, base_url: str):
        """One source, one rule and one action used to overflow the editor pane.

        The allowance was 2px, then 30px, then 50px, and is now 80px. Two deliberate additions
        spent the first rise: the MQTT topic template is a real new control in the
        action row, and the subject slots became multi-select dropdowns, whose trigger
        is a few pixels taller than the `<select>` it replaced.

        The rise to 50px is not extra content — it is the editor pane getting *shorter*
        by the amount it was previously overhanging the viewport. `--le-chrome-h` did
        not count the in-flow padding below the split, so the pane ran ~48px past the
        bottom of the window and the whole page scrolled to compensate, taking the
        pinned Save bar with it. Sizing the pane correctly hands those pixels back to
        the page, and the content that used to overflow the *window* now overflows this
        scroller instead. Same content, one scroller instead of two, Save always
        visible. (Half of it was then reclaimed by dropping a duplicated
        `padding-bottom` — 78px measured, 46px after.)

        The rise to 80px is the rule row's summary line — one line of prose per rule,
        stating what its tuning numbers do. See
        `test_a_rule_row_does_not_fill_the_screen` for why that line is worth its 25px.

        Every row is still one line per control group, which is the regression this
        test actually exists to catch — a control going full-width and stacking, which
        costs ~35px each and hundreds in total, and would clear this on its own.

        Note the viewport: Playwright's default is 720px tall, which is the tightest
        window anyone would realistically use, so this is the worst case rather than
        the typical one — the pane grows with the window.
        """
        self._open_profile(page, base_url)
        body = page.locator("#watcher-management-box .le-editor-body").first
        overflow = body.evaluate("e => e.scrollHeight - e.clientHeight")
        assert overflow <= 80, f"the editor scrolls by {overflow}px for a 1/1/1 profile"

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
        host = row.locator("[data-testid='watch-lifetime-label']")
        host.locator(".ms-dropdown__trigger").click()
        host.locator(".ms-dropdown__items input[type='checkbox']").first.check()
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
        assert lifetimes and len(lifetimes[0]["labelUuidsJson"]) == 1
        assert lifetimes[0]["ttlS"] == 30
        assert lifetimes[0]["isAnchor"] is True

    def test_a_lifetime_can_cover_several_labels_and_their_children(self, page: Page, base_url: str):
        """The reason the flat `<select>` had to go: "every cat survives 1 s" is one row.

        Two labels *and* the hierarchy toggle, because losing either on the way to the
        API looks at runtime like a tracker that ignored the setting rather than an
        editor that dropped it.
        """
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=saved.update)
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-rule-row']").first).to_be_visible()

        page.locator("[data-testid='watch-add-lifetime']").click()
        host = page.locator("[data-testid='watch-lifetime-row']").first.locator("[data-testid='watch-lifetime-label']")
        host.locator(".ms-dropdown__trigger").click()
        panel = host.locator(".ms-dropdown__panel")
        # The hierarchy the flat list could not show, and the toggle that goes with it.
        expect(panel).to_contain_text("Tabby")
        # Scoped to the item list: the panel also holds the toggle's own checkbox.
        host.locator(".ms-dropdown__items input[type='checkbox']").nth(0).check()
        host.locator(".ms-dropdown__items input[type='checkbox']").nth(1).check()
        panel.locator(".ms-hierarchy-toggle input[type='checkbox']").check()

        page.get_by_role("button", name="Save", exact=True).first.click()

        wait_until(page, lambda: saved.get("labelLifetimes"), "the save never reached the API")
        lifetime = saved["labelLifetimes"][0]
        assert len(lifetime["labelUuidsJson"]) == 2, lifetime
        assert lifetime["includeDescendants"] is True


class TestBufferTierBadge:
    """The rewind field's honesty affordance.

    "rewind s" asks for a rewindable buffer, and a snapshot camera cannot keep one at any
    value. Before this badge the mistake was invisible until an event fired and the clip
    that was supposed to appear simply never did.

    (The field was captioned "buffer s" and described as a pre-roll request, which is
    what led a user to shorten it to their preferred lead-in — pre-roll is set on the
    Record clip action. The badge's job is unchanged either way.)
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


class TestUnsavedRulesAreNotYetScopeable:
    """A rule added but not saved cannot be scoped to — and must say so.

    An action's rule scope is a list of rule uuids, and a rule has no uuid until the
    backend gives it one, so a just-added rule genuinely cannot be picked. The failure
    mode this guards is silence: the rule used to be dropped from the picker entirely,
    which reads as a bug rather than as "not yet".
    """

    def _open(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()

    def test_an_unsaved_rule_is_listed_but_not_pickable(self, page: Page, base_url: str):
        self._open(page, base_url)
        scope = page.locator("[data-testid='watch-action-rule-scope']").first
        scope.locator(".ms-dropdown__trigger").click()
        # The saved rule is the baseline: whatever the new row does, this one stays
        # selectable, or the fix has broken scoping outright.
        saved_row = scope.locator(".ms-item", has_text="cat using litter box")
        expect(saved_row.locator("input[type='checkbox']")).to_be_enabled()

        page.locator("[data-testid='watch-add-rule']").click()
        page.locator("[data-testid='watch-rule-name']").last.fill("dog at the door")
        # Tab out rather than dispatching `change`: committing the name rebuilds every
        # scope picker, and a rebuild triggered by the blur of the *next* click detaches
        # the control that click was aimed at.
        page.locator("[data-testid='watch-rule-name']").last.press("Tab")

        scope.locator(".ms-dropdown__trigger").click()
        new_row = scope.locator(".ms-item", has_text="dog at the door")
        expect(new_row).to_be_visible()
        expect(new_row.locator("input[type='checkbox']")).to_be_disabled()
        expect(saved_row.locator("input[type='checkbox']")).to_be_enabled()

    def test_select_all_cannot_pick_an_unsaved_rule(self, page: Page, base_url: str):
        """The greyed row must not be a uuid the save then rejects."""
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-add-rule']").click()
        page.locator("[data-testid='watch-rule-name']").last.fill("dog at the door")
        # Tab out rather than dispatching `change`: committing the name rebuilds every
        # scope picker, and a rebuild triggered by the blur of the *next* click detaches
        # the control that click was aimed at.
        page.locator("[data-testid='watch-rule-name']").last.press("Tab")

        scope = page.locator("[data-testid='watch-action-rule-scope']").first
        scope.locator(".ms-dropdown__trigger").click()
        scope.locator(".ms-select-all-btn", has_text="Select All").click()
        expect(scope.locator(".ms-dropdown__trigger")).to_contain_text("cat using litter box")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("actions")))
        # Only the rule that has an id. A synthetic placeholder reaching the payload is
        # a 422 from a control the user cannot see anything wrong with.
        assert saved["actions"][0]["ruleUuidsJson"] == ["rule-1"], saved["actions"][0]

    def test_renaming_a_rule_does_not_swallow_the_next_click(self, page: Page, base_url: str):
        """Rename a rule, then scope an action to it — in one go, as a user would.

        A rule name commits on blur, and the first click that actually moves focus is
        the one *inside* the picker. Relabelling used to rebuild the whole dropdown, so
        that blur destroyed the very checkbox being clicked, between its mousedown and
        its mouseup: no click event was ever dispatched and the box stayed unticked.
        Nothing about it looked broken — the click simply did nothing.
        """
        self._open(page, base_url)
        # Typed, not filled: `fill` sets the value without the browser's dirty flag, so
        # blur never fires `change` and the commit this test is about never happens.
        name = page.locator("[data-testid='watch-rule-name']").first
        name.click()
        name.press("Control+a")
        page.keyboard.type("dog at the door")

        scope = page.locator("[data-testid='watch-action-rule-scope']").first
        # Opening the panel does not blur the field — the trigger preventDefaults its
        # mousedown to keep the panel from closing — so the commit is still pending here.
        scope.locator(".ms-dropdown__trigger").click()
        # By id, not by name: the rename has not been committed yet, so the row still
        # reads by its old one at the moment this click is aimed.
        box = scope.locator("input[type='checkbox'][value='rule-1']")
        box.click()

        expect(box).to_be_checked()
        expect(scope.locator(".ms-dropdown__trigger")).to_contain_text("dog at the door")

    def test_the_rules_header_warns_only_while_a_rule_is_unsaved(self, page: Page, base_url: str):
        """Conditional by design: a watcher whose rules all came from the server is not
        in this state, and a hint that is always on is a hint nobody reads."""
        self._open(page, base_url)
        hint = page.locator("[data-testid='watch-rules-unsaved-hint']")
        expect(hint).to_be_hidden()

        page.locator("[data-testid='watch-add-rule']").click()
        expect(hint).to_be_visible()

        page.locator("[data-testid='watch-rule-row']").last.locator(".watch-row-remove").click()
        expect(hint).to_be_hidden()


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


class TestEnablingAndDisablingRows:
    """Per-row on/off, so silencing something does not mean deleting it.

    The flags already existed and the watcher already honoured them — `StreamWatcherTask`
    builds its rule list from `profile.rules if r.enabled` and skips disabled action rows
    — but nothing rendered them and both `getRule`/`getAction` hardcoded `enabled: true`.
    So the only way to turn a rule off was to delete it and rebuild it from memory later,
    and a profile saved from this editor silently re-enabled anything switched off
    through the API.
    """

    def _open(self, page: Page, base_url: str):
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-rule-row']").first).to_be_visible()

    def test_rows_start_enabled(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        expect(page.locator("[data-testid='watch-rule-enabled']").first).to_be_checked()
        expect(page.locator("[data-testid='watch-action-enabled']").first).to_be_checked()

    def test_a_disabled_rule_saves_as_disabled(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)

        page.locator("[data-testid='watch-rule-enabled']").first.uncheck()
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        assert saved["rules"][0]["enabled"] is False, saved["rules"][0]

    def test_a_disabled_action_saves_as_disabled(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)

        page.locator("[data-testid='watch-action-enabled']").first.uncheck()
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("actions")))
        assert saved["actions"][0]["enabled"] is False, saved["actions"][0]

    def test_an_untouched_row_still_saves_as_enabled(self, page: Page, base_url: str):
        """Guards the inversion: the flag used to be hardcoded true, so a bug that always
        sent false would disable a user's whole watcher on the next unrelated save."""
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)

        # Save stays disabled until the editor is dirty, so an unrelated edit is what
        # makes this the realistic case: someone renames a rule and every other row must
        # come back exactly as enabled as it went in.
        page.locator("[data-testid='watch-rule-name']").first.fill("renamed")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        assert saved["rules"][0]["enabled"] is True
        assert saved["actions"][0]["enabled"] is True

    def test_a_disabled_row_is_dimmed_but_still_editable(self, page: Page, base_url: str):
        """Dimmed, never hidden or collapsed — keeping the configuration is the point."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        row = page.locator("[data-testid='watch-rule-row']").first
        page.locator("[data-testid='watch-rule-enabled']").first.uncheck()
        expect(row).to_have_class(re.compile(r"watch-row--disabled"))
        # Still reachable: the name field accepts input while the row is off.
        name = page.locator("[data-testid='watch-rule-name']").first
        expect(name).to_be_visible()
        name.fill("off for now")
        expect(name).to_have_value("off for now")

    def test_the_toggle_itself_is_not_dimmed(self, page: Page, base_url: str):
        """It is the one control you need in order to switch the row back on; dimming it
        with everything else is how a disabled row becomes a lost one."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        box = page.locator("[data-testid='watch-rule-enabled']").first
        box.uncheck()
        opacity = box.evaluate("el => getComputedStyle(el).opacity")
        assert float(opacity) == 1.0, f"the enable toggle is dimmed at {opacity}"


class TestAnUncappedRule:
    """`max` = 0 means "this event lasts as long as its subject is there".

    The field is a plain number input, so the risk is that 0 reads as "empty" somewhere
    on the way out and is quietly replaced by the 300 default — which would look like the
    setting simply not working.
    """

    def test_zero_is_sent_rather_than_coerced_to_the_default(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-rule-row']").first).to_be_visible()

        # Targeted by testid rather than `.last`: the tuning group grew a motion-only
        # sensitivity field on the end, and "the last input" stopped being max-duration.
        page.locator("[data-testid='watch-rule-max-duration']").first.fill("0")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        assert saved["rules"][0]["maxDurationS"] == 0, saved["rules"][0]


class TestTheActionNoteIsLegible:
    """The note is free-text prose in the narrowest field on the row.

    At the row's inherited font size roughly ten characters were visible, which is not
    enough of "timelapse of the back garden every 30s" to tell two rows apart.
    """

    def test_the_note_uses_a_smaller_font_than_the_rows_other_inputs(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()

        note = page.locator("[data-testid='watch-action-name']").first
        note_px = float(note.evaluate("el => parseFloat(getComputedStyle(el).fontSize)"))
        # The action's own kind <select> sits on the same row and takes the inherited
        # size, so it is the honest comparison — an absolute pixel assertion would just
        # pin whatever the root font size happens to be.
        other = page.locator("[data-testid='watch-action-kind']").first
        other_px = float(other.evaluate("el => parseFloat(getComputedStyle(el).fontSize)"))
        assert note_px < other_px, f"note is {note_px}px, siblings are {other_px}px"


class TestEventTimelapseActionParams:
    """ "Event timelapse" had a kind in the dropdown and no form behind it.

    Every branch in `syncParams` is keyed on the action kind, and this one had none — so
    picking it fell through to the trailing `else`, which is the *snapshot* form. The
    user was asked for an output directory (this action's video is a recording attached
    to its event, not a file they go and find), the capture interval and playback rate
    were unreachable at any value but their defaults, and the row saved
    `output_dir: "snapshots"` — a dataset-relative path that put the mp4 outside the
    recordings root, where the retention sweep does not walk and the file endpoint does
    not serve.
    """

    def _open(self, page: Page, base_url: str):
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()
        page.locator("[data-testid='watch-action-kind']").first.select_option("record_event_timelapse")

    def test_it_offers_a_capture_interval_and_a_playback_rate(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        expect(page.locator("[data-testid='watch-action-event-timelapse-interval']").first).to_have_value("10")
        expect(page.locator("[data-testid='watch-action-event-timelapse-fps']").first).to_have_value("12")

    def test_it_does_not_ask_for_an_output_directory(self, page: Page, base_url: str):
        """The video is a recording hung off the event, bounded by the recording
        retention settings. A directory field both implies otherwise and, when saved,
        made it true."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        expect(page.locator("[data-testid='watch-action-dir']")).to_have_count(0)

    def test_the_speed_up_is_shown_before_saving(self, page: Page, base_url: str):
        """Neither field means much alone — the product is the number the user has in
        mind, and working it out by hand is what makes people leave the defaults."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        hint = page.locator("[data-testid='watch-action-event-timelapse-speed']").first
        expect(hint).to_contain_text("120×")

        page.locator("[data-testid='watch-action-event-timelapse-interval']").first.fill("30")
        expect(hint).to_contain_text("360×")
        expect(hint).to_contain_text("5 s")

    def test_the_pre_roll_says_how_much_time_it_covers(self, page: Page, base_url: str):
        """The field counts frames but the user is thinking in seconds, and the answer
        depends on the interval field next to it — the same reason the speed-up hint
        exists."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        hint = page.locator("[data-testid='watch-action-event-timelapse-preroll-hint']").first
        # Zero is the default, and it has to read as a state rather than as "0 s".
        expect(hint).to_contain_text("starts when the event does")

        page.locator("[data-testid='watch-action-event-timelapse-preroll']").first.fill("3")
        expect(hint).to_contain_text("30 s")

        # Follows the interval, not just its own value.
        page.locator("[data-testid='watch-action-event-timelapse-interval']").first.fill("20")
        expect(hint).to_contain_text("60 s")

    def test_the_burn_in_option_is_gone(self, page: Page, base_url: str):
        """It was the only way to see the boxes and it ruined the frames for training.
        The event view draws them at play time now, so offering it would only offer a
        way to spoil a recording permanently."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        expect(page.locator("[data-testid='watch-action-event-timelapse-annotate']")).to_have_count(0)
        # Nor is there a switch for the sidecar that replaced it: it costs a few tens of
        # KB, re-encodes nothing, and the player can already hide the boxes, so a control
        # for it would only offer a way to lose the detections permanently.
        expect(page.locator("[data-testid='watch-action-event-timelapse-overlay']")).to_have_count(0)

    def test_it_saves_its_params(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        self._open(page, base_url)

        page.locator("[data-testid='watch-action-event-timelapse-interval']").first.fill("20")
        page.locator("[data-testid='watch-action-event-timelapse-fps']").first.fill("24")
        page.locator("[data-testid='watch-action-event-timelapse-preroll']").first.fill("3")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("actions")))
        action = saved["actions"][0]
        assert action["kind"] == "record_event_timelapse"
        assert action["trigger"] == "event_ended"
        assert action["paramsJson"]["interval_s"] == 20
        assert action["paramsJson"]["output_fps"] == 24
        assert action["paramsJson"]["preroll_frames"] == 3
        # Not sent at all, which is how it stays on: the action defaults it to True, and
        # an omitted key lets that default apply — including for a row that had
        # previously stored it off, back when the form offered a switch.
        assert "save_overlay" not in action["paramsJson"], action["paramsJson"]
        # Gone, not merely defaulted off: burning boxes into the pixels is permanent and
        # ruins every frame for training, and the player now draws them at play time.
        assert "annotate" not in action["paramsJson"], action["paramsJson"]
        # Defaults the row never touched still reach the backend, so the action does not
        # silently fall back to a different set of them.
        assert action["paramsJson"]["min_frames"] == 10
        assert action["paramsJson"]["max_frames"] == 3600
        # The snapshot form's field, which this row used to inherit. Sending it at all
        # is what took the video out of the recordings root.
        assert "output_dir" not in action["paramsJson"], action["paramsJson"]


class TestSchemaDrivenActionParams:
    """The form is generated from the action's `params_schema()`.

    This is the property the whole arrangement exists for, and it is the one that was
    missing: the pane used to hold a hand-written field list per action kind, so the two
    could not disagree out loud. A parameter added on the Python side simply never
    appeared, and one removed from it carried on being written into saved config by a
    form that still offered it. Both happened within a day of each other.
    """

    def _open(self, page: Page, base_url: str):
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()
        # The fixture profile's first action is an MQTT publish; the schema-driven form
        # is the timelapse one, so switch the row's kind to it.
        page.locator("[data-testid='watch-action-kind']").first.select_option("record_event_timelapse")

    def _schema_with(self, **extra_fields) -> list:
        """ACTION_SCHEMAS with extra params bolted onto the timelapse action.

        Found by kind rather than by position — indexing into the list silently attaches
        the fields to whichever action happens to be first, which is how this test last
        started asserting against a form it was not looking at.
        """
        clone = copy.deepcopy(ACTION_SCHEMAS)
        target = next(a for a in clone if a["kind"] == "record_event_timelapse")
        target["paramsSchema"].update(extra_fields)
        return clone

    def test_a_parameter_the_form_has_never_heard_of_still_renders(self, page: Page, base_url: str):
        """No decorator, no branch, no JS change — the point of the exercise. If this
        fails, adding a parameter to an action is once again a two-file job with no
        error when you forget the second one."""
        schemas = self._schema_with(
            wobble_factor={"type": "int", "required": False, "default": 7, "help": "How much wobble."},
        )
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], action_schemas=schemas)
        self._open(page, base_url)

        field = page.locator("[data-testid='watch-action-param-wobble_factor']").first
        expect(field).to_be_visible()
        expect(field).to_have_value("7")
        # The schema's own help reaches the control, so a new parameter explains itself.
        assert "How much wobble." in (field.get_attribute("title") or "")

    def test_a_new_parameter_is_saved_under_its_declared_name(self, page: Page, base_url: str):
        """Rendering it is half the job; it has to round-trip under the snake_case key
        the Python side looks it up by."""
        saved = {}
        schemas = self._schema_with(
            wobble_factor={"type": "int", "required": False, "default": 7, "help": "How much wobble."},
        )
        _mock_watch_apis(
            page, profiles=[FAKE_PROFILE], action_schemas=schemas, on_update=lambda body: saved.update(body)
        )
        self._open(page, base_url)

        page.locator("[data-testid='watch-action-param-wobble_factor']").first.fill("12")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("actions")))
        assert saved["actions"][0]["paramsJson"]["wobble_factor"] == 12

    def test_a_hint_stays_on_the_same_line_as_the_field_it_explains(self, page: Page, base_url: str):
        """ "→ the 30 s before it opened" is meaningless next to some other field.

        The params row is a wrapping flex container, and the field and its hint used to
        be separate flex items — free to break onto different lines, which is exactly
        what happened at narrower widths. Asserted geometrically rather than by class
        name: what matters is that they end up on the same line, not how that is done.
        """
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)
        page.locator("[data-testid='watch-action-event-timelapse-preroll']").first.fill("3")

        field = page.locator("[data-testid='watch-action-event-timelapse-preroll']").first
        hint = page.locator("[data-testid='watch-action-event-timelapse-preroll-hint']").first
        expect(hint).to_contain_text("30 s")

        field_box = field.bounding_box()
        hint_box = hint.bounding_box()
        # Vertical centres within a few pixels of each other — the same visual line.
        field_mid = field_box["y"] + field_box["height"] / 2
        hint_mid = hint_box["y"] + hint_box["height"] / 2
        assert abs(field_mid - hint_mid) < 10, f"hint is {abs(field_mid - hint_mid)}px off the field's line"
        # And after it, not before — a preview that precedes its input reads as a label.
        assert hint_box["x"] > field_box["x"]

    def test_a_hidden_parameter_is_declared_but_never_offered(self, page: Page, base_url: str):
        """`output_dir` is real and the action honours it; this form must not ask for it.
        A typed path puts the mp4 outside the tree the retention sweep walks and the file
        endpoint serves."""
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        self._open(page, base_url)

        expect(page.locator("[data-testid='watch-action-param-output_dir']")).to_have_count(0)
        # The fields around it still render, so this is an omission rather than an
        # action whose form failed to build at all.
        expect(page.locator("[data-testid='watch-action-event-timelapse-interval']").first).to_be_visible()

    def test_a_checkbox_follows_the_schemas_default_in_both_directions(self, page: Page, base_url: str):
        """An unset value has to mean "the schema's default", not "off".

        Read as off, an action would quietly stop doing something it declares on by
        default, for every watcher that had never touched the field — and the form would
        then write that off back into the saved config, making it true.

        Asserted on `retain`, which the two MQTT actions declare with *opposite*
        defaults — off for the event action, where a retained edge would replay to every
        new subscriber as though it were live, and on for the image one, where a retained
        frame is what makes a dashboard show a picture the moment it connects. Same field
        name, same form, two answers: a bug that always checked, always cleared, or
        ignored the schema fails one half or the other.
        """
        # A row with nothing stored, so the schema's default is the only thing that can
        # decide the box. The shared fixture saves `retain: False` explicitly, and a
        # saved value rightly wins over a default — which would make this test pass for
        # the wrong reason on one half and fail on the other.
        profile = copy.deepcopy(FAKE_PROFILE)
        profile["actions"][0]["paramsJson"] = {}
        _mock_watch_apis(page, profiles=[profile])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()
        kind = page.locator("[data-testid='watch-action-kind']").first
        retain = page.locator("[data-testid='watch-action-param-retain']").first

        kind.select_option("publish_mqtt")
        expect(retain).not_to_be_checked()

        kind.select_option("publish_mqtt_image")
        expect(retain).to_be_checked()


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
        bare = page.evaluate("""() => [...document.querySelectorAll('#watcher-management-box .watch-field')]
                .filter(f => !(f.title || '').trim() && !f.hidden)
                .map(f => f.querySelector('.watch-field__label')?.textContent || '?')""")
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
        """ "All" is a summary a user reads as "nothing chosen"; the tooltip is where the
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


class TestEventFilingHint:
    """Whether the watcher will put anything in Events at all.

    An event is only recorded where an in-scope action writes to it, so a watcher whose
    actions all publish and none record leaves the Events list empty — which looks
    exactly like a camera that saw nothing. The hint is the only place that distinction
    is visible while the actions are being chosen, and it has to track the kind and the
    enabled box live, before anything is saved.
    """

    def test_the_hint_tracks_whether_any_action_files_events(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        hint = page.locator("[data-testid='watch-actions-event-hint']")
        kind = page.locator("[data-testid='watch-action-kind']").first

        # The fixture's only action is an MQTT publish, which files nothing.
        expect(hint).to_be_visible()

        kind.select_option("save_event")
        expect(hint).to_be_hidden()

        # A disabled action is not constructed at run time, so it files nothing either.
        page.locator("[data-testid='watch-action-enabled']").first.uncheck()
        expect(hint).to_be_visible()


class TestTheTrackingMistakesToggle:
    """Per-watcher switch for analysing clips against their own detection sidecar.

    Only the browser-side wiring is asserted here — that the control reflects the stored
    flag and that flipping it reaches the save. What the flag *does* is a backend rule
    and is covered in `tests/`.
    """

    def _open(self, page: Page, base_url: str, profile=None, saved=None):
        _mock_watch_apis(
            page,
            profiles=[profile or FAKE_PROFILE],
            on_update=(lambda body: saved.update(body)) if saved is not None else None,
        )
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        return page.locator("[data-testid='watch-analyse-blips']")

    def test_a_watcher_that_has_never_set_it_shows_it_on(self, page: Page, base_url: str):
        """Existing watchers adopt the feature switched on, matching the migration."""
        expect(self._open(page, base_url)).to_be_checked()

    def test_a_stored_off_flag_loads_unticked(self, page: Page, base_url: str):
        profile = {**FAKE_PROFILE, "blipAnalysisEnabled": False}
        expect(self._open(page, base_url, profile=profile)).not_to_be_checked()

    def test_switching_it_off_reaches_the_save(self, page: Page, base_url: str):
        saved: dict = {}
        toggle = self._open(page, base_url, saved=saved)
        toggle.uncheck()
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: "blipAnalysisEnabled" in saved, "the save never reached the API")
        assert saved["blipAnalysisEnabled"] is False, saved

    def test_a_save_preserves_thresholds_the_form_cannot_show(self, page: Page, base_url: str):
        """There is no editor for the tuning bundle yet. Collecting `{}` from the form
        would silently wipe thresholds set through the API."""
        saved: dict = {}
        profile = {**FAKE_PROFILE, "blipThresholdsJson": {"minFlipSamples": 5}}
        self._open(page, base_url, profile=profile, saved=saved)
        page.locator("[data-testid='watcher-name']").fill("Renamed")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: "blipThresholdsJson" in saved, "the save never reached the API")
        assert saved["blipThresholdsJson"] == {"minFlipSamples": 5}, saved


class TestCollectorPicker:
    """`collect_training_images` picks its collector from a list, not a pasted uuid.

    The action declares `collector_uuid` as its own schema type so the generic
    renderer routes it to a picker. Before that it was `str`, which fell through to
    a plain text box: a uuid typed wrong is accepted by the form and only fails
    when the WATCHER builds, with nothing on screen connecting the two.
    """

    def _route_collectors(self, page: Page, collectors: list):
        page.route(
            "**/api/training-collectors",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "collectors": collectors}),
            ),
        )

    def _open(self, page: Page, base_url: str, collectors: list):
        self._route_collectors(page, collectors)
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()
        page.locator("[data-testid='watch-action-kind']").first.select_option("collect_training_images")
        return page.locator("[data-testid='watch-action-collector']").first

    def test_the_collectors_that_exist_are_offered_as_options(self, page: Page, base_url: str):
        picker = self._open(
            page,
            base_url,
            [
                {"uuid": "c1", "name": "Doorway", "enabled": True, "params": {}},
                {"uuid": "c2", "name": "Litter box", "enabled": True, "params": {}},
            ],
        )

        expect(picker).to_be_visible()
        wait_until(
            page,
            lambda: picker.locator("option[value='c1']").count() == 1,
            "the collector list never populated the picker",
        )
        expect(picker.locator("option[value='c2']")).to_have_count(1)

    def test_a_disabled_collector_is_not_offered(self, page: Page, base_url: str):
        """Picking one would save an attachment the watcher refuses to build."""
        picker = self._open(
            page,
            base_url,
            [
                {"uuid": "c1", "name": "Doorway", "enabled": True, "params": {}},
                {"uuid": "c2", "name": "Retired", "enabled": False, "params": {}},
            ],
        )

        wait_until(page, lambda: picker.locator("option[value='c1']").count() == 1, "the picker never populated")
        expect(picker.locator("option[value='c2']")).to_have_count(0)

    def test_with_no_collectors_the_picker_says_so_rather_than_sitting_empty(self, page: Page, base_url: str):
        picker = self._open(page, base_url, [])

        wait_until(
            page,
            lambda: "No collectors yet" in (picker.locator("option").first.text_content() or ""),
            "an empty list left the picker showing nothing actionable",
        )
        # Where to fix it lives in the tooltip and the Edit… link rather than in the
        # option text, which sizes the <select> and made it three times the width of
        # every other control in the row.
        expect(page.locator("[data-testid='watch-action-collector-edit']").first).to_be_visible()
        assert "Collectors" in (picker.get_attribute("title") or "")


class TestEveryRegisteredActionKindIsOffered:
    """The kind dropdown is a hand-written list in `watchers.js`; the registry is the
    real one. Nothing compared them, and `collect_training_images` was registered,
    schema'd, tested and completely unreachable from the UI for four phases because
    of it — the exact failure mode the schema-driven params form was built to end.
    """

    def test_the_dropdown_offers_every_kind_the_registry_declares(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-action-row']").first).to_be_visible()

        offered = {
            option.get_attribute("value")
            for option in page.locator("[data-testid='watch-action-kind']").first.locator("option").all()
        }
        registered = {schema["kind"] for schema in ACTION_SCHEMAS}

        assert registered <= offered, f"registered but not offered: {sorted(registered - offered)}"


class TestRuleTrigger:
    """Whether the rule records the whole time it holds, or one instant at an edge.

    The engine half is covered in `tests/helpers/test_watch_engine.py`; what needs a
    browser is that the choice reaches the payload, that a saved impulse loads back as
    one, and that the cap — which does not apply to an instant — leaves the row.
    """

    def _open(self, page: Page, base_url: str, profile=None):
        _mock_watch_apis(page, profiles=[profile or FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-rule-row']").first).to_be_visible()

    def test_a_rule_records_the_whole_time_by_default(self, page: Page, base_url: str):
        self._open(page, base_url)
        expect(page.locator("[data-testid='watch-rule-trigger']").first).to_have_value("while")

    def test_choosing_an_edge_is_saved(self, page: Page, base_url: str):
        saved = {}
        _mock_watch_apis(page, profiles=[FAKE_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        page.locator("[data-testid='watch-rule-trigger']").first.select_option("on_exit")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        assert saved["rules"][0]["trigger"] == "on_exit", saved["rules"][0]

    def test_the_cap_leaves_the_row_for_an_instant(self, page: Page, base_url: str):
        """An instant has no span to cap, and leaving the field on screen invites the one
        setting that would stop an `on_exit` rule firing at all — a 15-minute dwell under
        a 300s cap."""
        self._open(page, base_url)
        cap = page.locator("[data-testid='watch-rule-max-duration-field']").first
        expect(cap).to_be_visible()

        page.locator("[data-testid='watch-rule-trigger']").first.select_option("on_enter")
        expect(cap).to_be_hidden()

        page.locator("[data-testid='watch-rule-trigger']").first.select_option("while")
        expect(cap).to_be_visible()


class TestRuleSummary:
    """The sentence under the tuning row, restating what its numbers do.

    It exists because the numbers alone hid a real behaviour for an evening: a 30s cap
    on a `present` rule recorded a clip every thirty seconds while the car was parked,
    and nothing on screen connected the two fields that produced that.
    """

    def _open(self, page: Page, base_url: str):
        _mock_watch_apis(page, profiles=[FAKE_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        expect(page.locator("[data-testid='watch-rule-row']").first).to_be_visible()

    def test_it_states_what_the_delays_do(self, page: Page, base_url: str):
        self._open(page, base_url)
        summary = page.locator("[data-testid='watch-rule-summary']").first
        expect(summary).to_contain_text("starts after 2s true")
        expect(summary).to_contain_text("ends after 10s clear")
        # The part that was invisible: what happens *after* the cap fires.
        expect(summary).to_contain_text("waits for the subject to leave")

    def test_it_follows_a_changed_delay(self, page: Page, base_url: str):
        """A summary that only rendered once would describe the saved rule while the user
        typed a different one — worse than no summary."""
        self._open(page, base_url)
        summary = page.locator("[data-testid='watch-rule-summary']").first
        expect(summary).not_to_contain_text("47s")

        # nth(2) is the start delay: threshold, nearby, starts, ends, max, sens.
        page.locator(".watch-rule-tuning input").nth(2).fill("47")
        expect(summary).to_contain_text("starts after 47s true")

    def test_it_describes_an_instant_differently(self, page: Page, base_url: str):
        """The surprising part of an impulse is that it records nothing by itself, so the
        summary has to say where the footage comes from."""
        self._open(page, base_url)
        page.locator("[data-testid='watch-rule-trigger']").first.select_option("on_exit")

        summary = page.locator("[data-testid='watch-rule-summary']").first
        expect(summary).to_contain_text("fires when this stops")
        # An instant records nothing by itself, which is the surprising part.
        expect(summary).to_contain_text("Length comes from the action")
        # The cap belongs to a span, and repeating it would describe a field the row no
        # longer shows.
        expect(summary).not_to_contain_text("at most")
