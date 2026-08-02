"""
Tests for the "Insert models from group" control on the `model_label_config`
array field (apps/image_tagging_server/static/js/ui/utils/fields/ArrayField.js).

Phase 5C of LIVE_COLLECTION_PLAN.md: `CollectTrainingImagesTask.params_schema()`
marks `model_label_config` with `allow_group_insert: True`. The `array` field
factory (Factories.js) pulls that flag out of the schema entry and passes it
into `ArrayField.create` as `allowGroupInsert`; `ArrayField` renders the extra
control only when the flag is set, and only for this field — it must not
appear on ordinary array fields (e.g. `source_uuids`), so a second schema
without the flag is used as a negative control.

This is a one-shot insert (append one `{modelName, labelValues: {}}` entry per
model in the picked group, in group order), not a live binding, and it must
never touch the sibling `merge_policy` param, which already has its own
first-class `select` widget in the same form.

Run with:
    pytest playwright_tests/test_model_group_insert.py -p no:asyncio -v
"""

import json

import pytest
from playwright.sync_api import Page, Route, expect

TASK_TYPE = "FakeGroupInsertTask"
MODEL_A = "yolo-model-a"
MODEL_B = "yolo-model-b"
MODEL_C = "yolo-model-c"
CAT_UUID = "11111111-0000-0000-0000-000000000001"
GROUP_UUID = "22222222-0000-0000-0000-000000000002"
EMPTY_GROUP_UUID = "33333333-0000-0000-0000-000000000003"
OTHER_GROUP_UUID = "44444444-0000-0000-0000-000000000004"

# Mirrors CollectTrainingImagesTask.params_schema(): model_label_config is an
# array of model_label items with allow_group_insert, sitting next to a plain
# merge_policy select and a plain array field (source_uuids-alike) that must
# NOT get the group-insert control.
GROUP_INSERT_SCHEMA = {
    "meta": {"order": ["model_label_config", "merge_policy", "plain_list"]},
    "model_label_config": {
        "type": "array",
        "label": "Model/Label Configuration",
        "allow_group_insert": True,
        "items": {
            "type": "model_label",
            "label": "Model/Label selection",
        },
    },
    "merge_policy": {
        "type": "select",
        "label": "Multi-Model Merge Policy",
        "default": "first_wins",
        "options": [
            {"value": "first_wins", "label": "First model wins (priority order)"},
            {"value": "last_wins", "label": "Last model wins (reverse priority order)"},
        ],
    },
    "plain_list": {
        "type": "array",
        "label": "Plain list (no group insert)",
        "items": {"type": "string", "label": "Item"},
    },
}

FAKE_CONFIG_UUID = "eeeeeeee-0000-0000-0000-000000000099"


def _config(model_label_config: list, merge_policy: str = "first_wins") -> dict:
    return {
        "uuid": FAKE_CONFIG_UUID,
        "name": "Group Insert Test Config",
        "typename": TASK_TYPE,
        "params": {
            "model_label_config": model_label_config,
            "merge_policy": merge_policy,
            "plain_list": [],
        },
    }


def _navigate_to_tasks(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-task-management']").click()
    page.wait_for_load_state("networkidle")


def _mock_apis(page: Page, configs: dict | None = None, groups: list | None = None):
    configs = configs or {}
    groups = groups if groups is not None else []

    page.route(
        "**/api/tasks/types/list",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "types": [TASK_TYPE]}),
        ),
    )
    page.route(
        "**/api/tasks/types/schema",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"schemas": {TASK_TYPE: GROUP_INSERT_SCHEMA}}),
        ),
    )
    page.route(
        "**/api/tasks/configs/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": configs}),
        ),
    )
    page.route(
        "**/api/tasks/instances/get",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "tasks": {}}),
        ),
    )
    page.route(
        "**/api/models",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "models": [MODEL_A, MODEL_B, MODEL_C]}),
        ),
    )
    page.route(
        "**/api/models/classes/bulk",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    # Real shape: Dict[modelName, Dict[classIndex, labelUuid]].
                    "status": "success",
                    "models": {
                        MODEL_A: {"0": CAT_UUID},
                        MODEL_B: {"0": CAT_UUID},
                        MODEL_C: {"0": CAT_UUID},
                    },
                }
            ),
        ),
    )
    page.route(
        "**/api/labels",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "labels": [
                        {"uuid": CAT_UUID, "name": "cat", "color": "#ff0000", "labelType": "standard"},
                    ],
                }
            ),
        ),
    )
    page.route(
        "**/api/model-groups",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "groups": groups}),
        ),
    )


def _group(uuid: str, name: str, member_model_names: list) -> dict:
    return {
        "uuid": uuid,
        "name": name,
        "description": "",
        "mergePolicy": "first_wins",
        "members": [
            {"uuid": f"member-{i}", "modelGroupUuid": uuid, "order": i, "modelName": m, "childGroupUuid": None}
            for i, m in enumerate(member_model_names)
        ],
    }


def _open_edit(page: Page):
    page.wait_for_selector("#task-config-list button[title='Edit']", timeout=5000)
    page.locator("#task-config-list button[title='Edit']").first.click()


class TestGroupInsertAppendsInOrder:
    def test_insert_appends_one_entry_per_model_in_group_order(self, page: Page, base_url: str):
        groups = [_group(GROUP_UUID, "My Group", [MODEL_C, MODEL_A, MODEL_B])]
        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config([])}, groups=groups)
        _navigate_to_tasks(page, base_url)
        _open_edit(page)

        insert_btn = page.locator("#task-config-list button", has_text="Insert models from group")
        expect(insert_btn).to_be_visible(timeout=5000)

        group_select = insert_btn.locator("xpath=preceding-sibling::select[1]")
        group_select.select_option(GROUP_UUID)
        insert_btn.click()

        # One model dropdown per inserted entry, in group order (C, A, B).
        model_selects = page.locator("#task-config-list .array-field > div:not(.array-group-insert-row) select")
        expect(model_selects).to_have_count(3, timeout=5000)
        assert [model_selects.nth(i).input_value() for i in range(3)] == [MODEL_C, MODEL_A, MODEL_B]

    def test_insert_appends_after_existing_entries_not_replacing(self, page: Page, base_url: str):
        groups = [_group(GROUP_UUID, "My Group", [MODEL_B])]
        existing_entry = {"modelName": MODEL_A, "labelValues": {CAT_UUID: 0.5}}
        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config([existing_entry])}, groups=groups)
        _navigate_to_tasks(page, base_url)
        _open_edit(page)

        insert_btn = page.locator("#task-config-list button", has_text="Insert models from group")
        expect(insert_btn).to_be_visible(timeout=5000)
        group_select = insert_btn.locator("xpath=preceding-sibling::select[1]")
        group_select.select_option(GROUP_UUID)
        insert_btn.click()

        model_selects = page.locator("#task-config-list .array-field > div:not(.array-group-insert-row) select")
        expect(model_selects).to_have_count(2, timeout=5000)
        # Existing entry (model A) stays first; the inserted model (B) is appended.
        assert model_selects.nth(0).input_value() == MODEL_A
        assert model_selects.nth(1).input_value() == MODEL_B

    def test_dropdown_selection_survives_the_post_insert_rerender(self, page: Page, base_url: str):
        """Regression test: `_renderGroupInsertControl()` runs inside
        `renderItems()`, so the whole control — including the group <select> —
        is rebuilt after every insert. Before the fix, the rebuilt select had
        no memory of what was picked and silently snapped back to the first
        option, so a second click would insert the wrong group's models
        without any error. The picked group must still be selected after the
        insert completes."""
        # OTHER_GROUP_UUID sorts first in the groups map (JS insertion order,
        # since Map preserves it and the mocked response lists it first), so a
        # naive "select defaults to first option" bug would show OTHER_GROUP
        # selected instead of GROUP_UUID after the rebuild.
        groups = [
            _group(OTHER_GROUP_UUID, "Other Group", [MODEL_C]),
            _group(GROUP_UUID, "My Group", [MODEL_A, MODEL_B]),
        ]
        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config([])}, groups=groups)
        _navigate_to_tasks(page, base_url)
        _open_edit(page)

        insert_btn = page.locator("#task-config-list button", has_text="Insert models from group")
        expect(insert_btn).to_be_visible(timeout=5000)
        group_select = insert_btn.locator("xpath=preceding-sibling::select[1]")
        group_select.select_option(GROUP_UUID)
        insert_btn.click()
        page.wait_for_timeout(300)

        # Re-fetch the (rebuilt) button/select after the rerender.
        insert_btn = page.locator("#task-config-list button", has_text="Insert models from group")
        expect(insert_btn).to_be_visible(timeout=5000)
        group_select = insert_btn.locator("xpath=preceding-sibling::select[1]")
        expect(group_select).to_have_value(GROUP_UUID)

    def test_merge_policy_untouched_by_insert(self, page: Page, base_url: str):
        groups = [_group(GROUP_UUID, "My Group", [MODEL_A, MODEL_B])]
        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config([], merge_policy="last_wins")}, groups=groups)
        _navigate_to_tasks(page, base_url)
        _open_edit(page)

        # merge_policy's own select should already show "last_wins" and remain so
        # after the group insert — the insert must not overwrite it.
        merge_policy_select = page.locator("#task-config-list select").filter(
            has=page.locator("option[value='last_wins']")
        )
        expect(merge_policy_select).to_have_value("last_wins")

        insert_btn = page.locator("#task-config-list button", has_text="Insert models from group")
        expect(insert_btn).to_be_visible(timeout=5000)
        group_select = insert_btn.locator("xpath=preceding-sibling::select[1]")
        group_select.select_option(GROUP_UUID)
        insert_btn.click()
        page.wait_for_timeout(300)

        expect(merge_policy_select).to_have_value("last_wins")


class TestGroupInsertEdgeCases:
    def test_no_groups_disables_control(self, page: Page, base_url: str):
        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config([])}, groups=[])
        _navigate_to_tasks(page, base_url)
        _open_edit(page)

        insert_btn = page.locator("#task-config-list button", has_text="Insert models from group")
        expect(insert_btn).to_be_visible(timeout=5000)
        expect(insert_btn).to_be_disabled()

    def test_empty_group_inserts_nothing(self, page: Page, base_url: str):
        groups = [_group(EMPTY_GROUP_UUID, "Empty Group", [])]
        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config([])}, groups=groups)
        _navigate_to_tasks(page, base_url)
        _open_edit(page)

        insert_btn = page.locator("#task-config-list button", has_text="Insert models from group")
        expect(insert_btn).to_be_visible(timeout=5000)
        group_select = insert_btn.locator("xpath=preceding-sibling::select[1]")
        group_select.select_option(EMPTY_GROUP_UUID)
        insert_btn.click()
        page.wait_for_timeout(300)

        model_selects = page.locator("#task-config-list .array-field > div:not(.array-group-insert-row) select")
        expect(model_selects).to_have_count(0)

    def test_plain_array_field_has_no_group_insert_control(self, page: Page, base_url: str):
        """Opt-in check: an ordinary array field (no allow_group_insert in its
        schema entry) must not render the group-insert control at all."""
        _mock_apis(page, configs={FAKE_CONFIG_UUID: _config([])}, groups=[_group(GROUP_UUID, "My Group", [MODEL_A])])
        _navigate_to_tasks(page, base_url)
        _open_edit(page)

        # Exactly one "Insert models from group" button should exist (on
        # model_label_config), not one per array field.
        insert_btns = page.locator("#task-config-list button", has_text="Insert models from group")
        expect(insert_btns).to_have_count(1, timeout=5000)
