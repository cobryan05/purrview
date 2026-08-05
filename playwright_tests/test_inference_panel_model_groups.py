"""
Tests for saved "model group" presets in the inference side panel —
apps/image_tagging_server/static/js/app/image-tagging/inferencePanel.js
+ apps/image_tagging_server/static/js/shared/api/modelGroups.js

Covers: saving the current checked detector selection (in check order) as a
named group via POST /api/model-groups, the group then appearing in a "Groups"
section, and selecting a saved group on Run expanding it back to its member
models in the saved order — both immediately after saving and after
"reopening" the panel (simulating a reload) against a group that already
exists server-side.

Run with:
    pytest playwright_tests/test_inference_panel_model_groups.py -p no:xdist --override-ini="addopts=" -v
"""

import json

import pytest
from playwright.sync_api import Page


def _mock_models(page: Page, models: list, kinds: dict = None):
    kinds = kinds or {}
    page.route(
        lambda url: url.split("?")[0].endswith("/api/models"),
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "models": models}),
            )
            if route.request.method == "GET"
            else route.fallback()
        ),
    )

    def _meta_handler(route):
        name = route.request.url.rstrip("/").split("/")[-2]
        kind = kinds.get(name, "detector")
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "modelKind": kind}),
        )

    page.route(lambda url: "/api/models/" in url and url.endswith("/meta"), _meta_handler)


def _mock_model_groups(page: Page, initial_groups: list | None = None):
    """Mocks GET (list) + POST (create) /api/model-groups against an in-memory
    list, so a create followed by the panel's own re-fetch (cache-cleared)
    sees the newly created group — mirrors the mutable-list pattern used for
    `/api/models` in test_run_model_on_results.py's refresh test."""
    state = {"groups": list(initial_groups or [])}

    def handle(route):
        req = route.request
        if req.method == "GET":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "groups": state["groups"]}),
            )
        elif req.method == "POST":
            body = json.loads(req.post_data or "{}")
            uuid = f"grp-{len(state['groups']) + 1}"
            group = {
                "uuid": uuid,
                "name": body["name"],
                "description": body.get("description"),
                "members": [
                    {
                        "uuid": f"{uuid}-mem-{i}",
                        "modelGroupUuid": uuid,
                        "order": m["order"],
                        "modelName": m.get("model_name"),
                        "childGroupUuid": m.get("child_group_uuid"),
                    }
                    for i, m in enumerate(body["members"])
                ],
            }
            state["groups"].append(group)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "group": group}),
            )
        else:
            route.fallback()

    page.route(lambda url: url.split("?")[0].endswith("/api/model-groups"), handle)
    return state


def _reset_session(page: Page):
    page.evaluate("""() => {
        sessionStorage.removeItem('wtch_inference_selected_detectors');
        sessionStorage.removeItem('wtch_inference_selected_classifier');
    }""")


def _make_panel(page: Page):
    return page.evaluate("""async () => {
        const { createInferencePanel } = await import('/app-static/js/app/image-tagging/inferencePanel.js');
        const { clearModelsCache } = await import('/app-static/js/shared/api/models.js');
        const { clearModelGroupsCache } = await import('/app-static/js/shared/api/modelGroups.js');
        clearModelsCache();
        clearModelGroupsCache();

        document.getElementById('test-inference-panel')?.remove();
        const container = document.createElement('div');
        container.id = 'test-inference-panel';
        document.body.appendChild(container);

        const runtime = { id: 'test-runtime', state: {} };
        const handle = await createInferencePanel(runtime, container);
        window.__testPanelHandle = handle;
        return true;
    }""")


def _check_item(page: Page, name: str, checked: bool = True):
    """Toggles the checkbox for any item (detector or group) by its display name."""
    page.evaluate(
        """([name, checked]) => {
            const items = [...document.querySelectorAll('#test-inference-panel .inference-model-item')];
            const item = items.find(el => el.querySelector('.inference-model-name')?.textContent === name);
            const cb = item.querySelector('input[type=checkbox]');
            if (cb.checked !== checked) cb.click();
        }""",
        [name, checked],
    )


def _capture_run_event(page: Page):
    return page.evaluate("""async () => {
        const { events, EventTypes } = await import('/app-static/js/shared/events/index.js');
        const payload = await new Promise(resolve => {
            const unsub = events.subscribeToCanvas('test-runtime', EventTypes.RUN_INFERENCE_ON_CANVAS, (p) => {
                unsub();
                resolve(p);
            });
            document.querySelector('#test-inference-panel .inference-buttons-row button').click();
        });
        return payload;
    }""")


@pytest.fixture(autouse=True)
def _cleanup(shared_page: Page):
    yield
    shared_page.evaluate("""() => {
        document.getElementById('test-inference-panel')?.remove();
        window.__testPanelHandle?.destroy?.();
    }""")
    shared_page.unroute_all(behavior="ignoreErrors")


class TestSaveAndRunModelGroup:
    def test_save_selection_as_group_then_run_expands_saved_order(self, shared_page: Page):
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["det-a", "det-b"])
        groups_state = _mock_model_groups(page)
        _make_panel(page)

        _check_item(page, "det-a")
        _check_item(page, "det-b")

        page.once("dialog", lambda dialog: dialog.accept("MyGroup"))
        page.click("#test-inference-panel button:has-text('Save as stack')")
        page.wait_for_timeout(300)  # let the async create + re-fetch settle

        assert len(groups_state["groups"]) == 1
        created = groups_state["groups"][0]
        assert created["name"] == "MyGroup"
        members_by_order = {m["order"]: m for m in created["members"]}
        assert members_by_order[0]["modelName"] == "det-a"
        assert members_by_order[1]["modelName"] == "det-b"

        # The saved group now shows up in the panel's Groups section.
        group_names = page.eval_on_selector_all(
            "#test-inference-panel .inference-model-item .inference-model-name",
            "els => els.map(el => el.textContent)",
        )
        assert "MyGroup" in group_names

        # Swap the raw detector checks for the saved group and confirm Run
        # expands it back to both models, in the order they were saved.
        _check_item(page, "det-a", checked=False)
        _check_item(page, "det-b", checked=False)
        _check_item(page, "MyGroup")

        payload = _capture_run_event(page)
        assert payload["modelNames"] == ["det-a", "det-b"]

    def test_group_saved_in_a_prior_session_is_selectable_on_reload(self, shared_page: Page):
        """Simulates reopening the panel (e.g. after a page reload) against a
        group that already exists server-side from an earlier session."""
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["det-a", "det-b"])
        _mock_model_groups(
            page,
            initial_groups=[
                {
                    "uuid": "grp-1",
                    "name": "SavedGroup",
                    "description": None,
                    "members": [
                        {
                            "uuid": "m1",
                            "modelGroupUuid": "grp-1",
                            "order": 0,
                            "modelName": "det-b",
                            "childGroupUuid": None,
                        },
                        {
                            "uuid": "m2",
                            "modelGroupUuid": "grp-1",
                            "order": 1,
                            "modelName": "det-a",
                            "childGroupUuid": None,
                        },
                    ],
                }
            ],
        )
        _make_panel(page)

        group_names = page.eval_on_selector_all(
            "#test-inference-panel .inference-model-item .inference-model-name",
            "els => els.map(el => el.textContent)",
        )
        assert "SavedGroup" in group_names

        # First-ever-load default-select-first-detector doesn't fire here since a
        # persisted (albeit empty) selection key isn't set — clear any default pick.
        _check_item(page, "det-a", checked=False)
        _check_item(page, "SavedGroup")

        payload = _capture_run_event(page)
        assert payload["modelNames"] == ["det-b", "det-a"]
