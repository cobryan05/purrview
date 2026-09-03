"""A trained hierarchical model config appears in the inference panel as ONE model.

apps/image_tagging_server/static/js/app/image-tagging/inferencePanel.js
+ apps/image_tagging_server/static/js/shared/api/inference.js

The behaviour that matters here is the wiring the browser owns: the unit is named from
the model-configs API (the inference server returns bare uuids, having no database), and
Run publishes its *uuid* rather than the names of its parts. If it ever published the
parts, the second pass would be back in the caller's hands and the canvas would silently
disagree with the watcher.

Run with:
    pytest playwright_tests/test_inference_panel_units.py -p no:xdist --override-ini="addopts=" -v
"""

import json

import pytest
from playwright.sync_api import Page

CFG_UUID = "cfg-nest"


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
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "modelKind": kinds.get(name, "detector")}),
        )

    page.route(lambda url: "/api/models/" in url and url.endswith("/meta"), _meta_handler)
    page.route(
        lambda url: url.split("?")[0].endswith("/api/model-groups"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "groups": []}),
        ),
    )


def _mock_units(page: Page, units: list):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/models/units"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "units": units}),
        ),
    )


def _mock_model_configs(page: Page, configs: list):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/model-configs"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "configs": configs}),
        ),
    )


def _unit(uuid=CFG_UUID, detectors=("test_nest",), subs=("test_nest__cat",), notes=()):
    return {
        "modelConfigUuid": uuid,
        "detectorNames": list(detectors),
        "subModelNames": list(subs),
        "parts": [],
        "notes": list(notes),
    }


def _reset_session(page: Page):
    page.evaluate("""() => {
        localStorage.removeItem('ww_inference_selected_detectors');
        localStorage.removeItem('ww_inference_selected_classifier');
        sessionStorage.removeItem('wtch_inference_selected_detectors');
        sessionStorage.removeItem('wtch_inference_selected_classifier');
    }""")


def _make_panel(page: Page):
    return page.evaluate("""async () => {
        const { createInferencePanel } = await import('/app-static/js/app/image-tagging/inferencePanel.js');
        const { clearModelsCache } = await import('/app-static/js/shared/api/models.js');
        const { clearModelGroupsCache } = await import('/app-static/js/shared/api/modelGroups.js');
        const { clearModelConfigsCache } = await import('/app-static/js/shared/api/modelConfigs.js');
        clearModelsCache();
        clearModelGroupsCache();
        clearModelConfigsCache();

        document.getElementById('test-inference-panel')?.remove();
        const container = document.createElement('div');
        container.id = 'test-inference-panel';
        document.body.appendChild(container);

        const runtime = { id: 'test-runtime', state: {} };
        window.__testPanelHandle = await createInferencePanel(runtime, container);
        return true;
    }""")


def _items(page: Page):
    return page.eval_on_selector_all(
        "#test-inference-panel .inference-model-item",
        """els => els.map(el => ({
            name: el.querySelector('.inference-model-name')?.textContent,
            badge: el.querySelector('.inference-model-badge')?.textContent,
            warn: el.classList.contains('inference-model-item--warn'),
        }))""",
    )


def _check_item(page: Page, name: str, checked: bool = True):
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
        return await new Promise(resolve => {
            const unsub = events.subscribeToCanvas('test-runtime', EventTypes.RUN_INFERENCE_ON_CANVAS, (p) => {
                unsub();
                resolve(p);
            });
            document.querySelector('#test-inference-panel .inference-buttons-row button').click();
        });
    }""")


@pytest.fixture(autouse=True)
def _cleanup(shared_page: Page):
    yield
    shared_page.evaluate("""() => {
        document.getElementById('test-inference-panel')?.remove();
        window.__testPanelHandle?.destroy?.();
    }""")
    shared_page.unroute_all(behavior="ignoreErrors")


class TestHierarchyUnits:
    def test_unit_is_named_from_its_config_and_shows_its_sub_model_count(self, shared_page: Page):
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["test_nest", "test_nest__cat"], {"test_nest__cat": "crop_classifier"})
        _mock_units(page, [_unit()])
        _mock_model_configs(page, [{"uuid": CFG_UUID, "name": "Test Nest", "nodes": []}])
        _make_panel(page)

        by_name = {i["name"]: i for i in _items(page)}
        assert "Test Nest" in by_name, "the hierarchy should appear as one selectable entry"
        # The count is the difference between "a hierarchy" and "a top model with nothing
        # under it" — the exact distinction that was invisible before.
        assert by_name["Test Nest"]["badge"] == "+1 sub"
        assert not by_name["Test Nest"]["warn"]

    def test_a_hierarchy_with_no_trained_sub_models_is_marked(self, shared_page: Page):
        """The failure that started this work: the top trained, the sub-model did not,
        and inference silently stopped at the parent label. It must still be runnable —
        the coarse answer is a real answer — but it must not look healthy."""
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["test_nest"])
        _mock_units(page, [_unit(subs=())])
        _mock_model_configs(page, [{"uuid": CFG_UUID, "name": "Test Nest", "nodes": []}])
        _make_panel(page)

        by_name = {i["name"]: i for i in _items(page)}
        assert by_name["Test Nest"]["badge"] == "top only"
        assert by_name["Test Nest"]["warn"]

    def test_running_a_unit_publishes_its_uuid_not_its_parts(self, shared_page: Page):
        """The whole contract of the unit. Publishing the part names instead would put
        the cascade back in the browser, where it would drift from the watcher's."""
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["test_nest", "test_nest__cat"], {"test_nest__cat": "crop_classifier"})
        _mock_units(page, [_unit()])
        _mock_model_configs(page, [{"uuid": CFG_UUID, "name": "Test Nest", "nodes": []}])
        _make_panel(page)

        _check_item(page, "test_nest", checked=False)  # clear the default first-detector pick
        _check_item(page, "Test Nest")
        payload = _capture_run_event(page)

        assert payload["unitUuids"] == [CFG_UUID]
        assert payload["modelNames"] == []

    def test_a_unit_whose_config_was_deleted_is_not_offered(self, shared_page: Page):
        """Artifacts outlive the config that defined them. Showing one with a raw uuid
        for a name would offer a hierarchy nobody can open or edit."""
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["test_nest"])
        _mock_units(page, [_unit(uuid="cfg-deleted")])
        _mock_model_configs(page, [])
        _make_panel(page)

        names = [i["name"] for i in _items(page)]
        assert "cfg-deleted" not in names
        assert names == ["test_nest"]
