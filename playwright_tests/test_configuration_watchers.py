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


MODELS = ["cat_detector.onnx", "yolov8n.onnx"]

MODEL_GROUPS = [
    {"uuid": "grp-1", "name": "House stack", "description": "", "mergePolicy": "first_wins", "members": []},
]

LABELS = [
    {"uuid": "label-cat", "name": "Cat"},
    {"uuid": "label-tabby", "name": "Tabby"},
    {"uuid": "label-litterbox", "name": "Litter Box"},
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
            "subjectARef": "label-tabby",
            "subjectBKind": "label",
            "subjectBRef": "label-litterbox",
        }
    ],
}


def _mock_watch_apis(page: Page, profiles: list | None = None, on_create=None, on_update=None):
    profiles = profiles or []

    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/labels"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": LABELS}),
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

    def test_a_label_subject_b_loads_its_own_label(self, page: Page, base_url: str):
        """A rule whose second subject is a label must reopen showing *that* label.

        The select was built from the zone options, so a label uuid never matched and
        the row silently fell back to the first label — "Litter Box" became "Cat".
        """
        _mock_watch_apis(page, profiles=[LABEL_SUBJECT_PROFILE])
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        subject_b = page.locator("[data-testid='watch-rule-subject-b']")
        expect(subject_b).to_have_value("label-litterbox")

    def test_saving_keeps_a_label_subject_b(self, page: Page, base_url: str):
        """Round-trip: the payload sent back must still name the chosen label."""
        saved = {}
        _mock_watch_apis(page, profiles=[LABEL_SUBJECT_PROFILE], on_update=lambda body: saved.update(body))
        _navigate_to_watchers(page, base_url)
        page.locator("[data-testid='watcher-list-item']").first.click()
        # Save is disabled until the editor is dirty; an unrelated edit arms it.
        page.locator("[data-testid='watch-rule-name']").first.fill("Tabby using litter box!")
        page.locator("#watcher-management-box button:has-text('Save')").first.click()

        wait_until(page, lambda: bool(saved.get("rules")))
        rule = saved["rules"][0]
        assert rule["subjectBKind"] == "label", rule
        assert rule["subjectBRef"] == "label-litterbox", rule

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
        """ "When [cat] [is inside] [Zone] [litter box]" reads as a sentence only while
        its selects share a line — the whole point of the sentence layout."""
        self._open_profile(page, base_url)
        tops = page.locator(".watch-rule-sentence select").evaluate_all(
            "els => els.map(e => Math.round(e.getBoundingClientRect().top))"
        )
        assert len(tops) >= 3, f"expected the sentence's selects, got {tops}"
        assert len(set(tops)) == 1, f"the rule sentence wrapped onto several lines: tops={tops}"

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
        """One source, one rule and one action used to overflow the editor pane."""
        self._open_profile(page, base_url)
        body = page.locator("#watcher-management-box .le-editor-body").first
        overflow = body.evaluate("e => e.scrollHeight - e.clientHeight")
        assert overflow <= 2, f"the editor scrolls by {overflow}px for a 1/1/1 profile"

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

    def test_a_new_zone_becomes_selectable_in_an_existing_rule(self, page: Page, base_url: str):
        """Drawing a zone and then being unable to point a rule at it without saving
        and reopening the whole watcher is the wrong order to build a rule in."""
        self._open(page, base_url)
        subject_b = page.locator("[data-testid='watch-rule-subject-b']").first
        expect(subject_b.locator("option")).to_have_count(1)

        self._draw_a_zone(page)

        expect(subject_b.locator("option")).to_have_count(2)

    def test_the_rules_existing_zone_choice_survives_the_refresh(self, page: Page, base_url: str):
        """Re-populating the dropdown must not silently repoint the rule at zone #1."""
        self._open(page, base_url)
        subject_b = page.locator("[data-testid='watch-rule-subject-b']").first
        expect(subject_b).to_have_value(ZONE_UUID)

        self._draw_a_zone(page)

        expect(subject_b).to_have_value(ZONE_UUID)

    def test_removing_a_source_drops_its_zones_from_the_rules(self, page: Page, base_url: str):
        """The same staleness in the other direction: a rule must not keep offering a
        zone belonging to a source that is no longer in the profile."""
        self._open(page, base_url)
        subject_b = page.locator("[data-testid='watch-rule-subject-b']").first
        expect(subject_b.locator("option")).to_have_count(1)

        page.locator("[data-testid='watch-source-row'] .watch-row-remove").first.click()

        expect(subject_b.locator("option")).to_have_count(1)
        expect(subject_b.locator("option").first).to_have_text("(no zones defined)")


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
