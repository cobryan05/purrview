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


# A shape that genuinely defines a sub-model. Three non-root nodes on purpose: parseNodes
# sorts them by label depth, and Array.sort skips its comparator for a single element — a
# one-node fixture never runs the label lookup that walks the tree.
HIER_NODES = [
    {"uuid": "n-root", "labelUuid": None, "modelType": "object_detection", "architecture": "yolo_v9_s"},
    {
        "uuid": "n-cat",
        "labelUuid": "lbl-cat",
        "modelType": "crop_classifier",
        "architecture": "efficientnet_b0",
        "isBoundary": True,
    },
    {
        "uuid": "n-willow",
        "labelUuid": "lbl-willow",
        "modelType": "crop_classifier",
        "architecture": "efficientnet_b0",
        "isBoundary": False,
    },
]

HIER_LABELS = [
    {"uuid": "lbl-cat", "name": "Cat", "color": "#ff8800", "parentUuid": None},
    {"uuid": "lbl-willow", "name": "Willow", "color": "#ff0000", "parentUuid": "lbl-cat"},
]


def _hier_config(uuid=CFG_UUID, name="Test Nest"):
    return {"uuid": uuid, "name": name, "description": "", "nodes": HIER_NODES}


def _mock_labels(page: Page, labels=None):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": HIER_LABELS if labels is None else labels}),
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
        _mock_model_configs(page, [_hier_config()])
        _mock_labels(page)
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
        _mock_model_configs(page, [_hier_config()])
        _mock_labels(page)
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
        _mock_model_configs(page, [_hier_config()])
        _mock_labels(page)
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

    def test_a_crop_classifier_is_never_listed_as_a_detector(self, shared_page: Page):
        """Reported for real: running the hierarchy's sub-model produced a single box
        filling the whole image, labelled with the parent.

        A crop classifier listed among detectors goes through the DETECTION path, where
        an EfficientNet head decodes into one full-image box. It answers instead of
        failing, so the result looks like a confident detection rather than a mistake.
        """
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["test_nest", "test_nest__cat"], {"test_nest__cat": "crop_classifier"})
        _mock_units(page, [_unit(subs=("test_nest__cat",))])
        _mock_model_configs(page, [_hier_config()])
        _mock_labels(page)
        _make_panel(page)

        sections = page.eval_on_selector_all(
            "#test-inference-panel .inference-model-section",
            """els => els.map(s => ({
                header: s.querySelector('.inference-model-section-header')?.textContent,
                items: [...s.querySelectorAll('.inference-model-name')].map(n => n.textContent),
            }))""",
        )
        by_header = {s["header"]: s["items"] for s in sections}
        assert "test_nest__cat" not in by_header.get("Detectors", []), by_header
        assert "test_nest" in by_header.get("Detectors", []), by_header
        assert "test_nest__cat" in by_header.get("Sub-models (refine a detector)", []), by_header

    def test_a_flat_shape_is_not_listed_as_a_hierarchy(self, shared_page: Page):
        """Every shape with a trained artifact used to appear under Hierarchies, so each
        flat shape showed up as a "hierarchy" duplicating the detector right below it."""
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["flat_model"])
        _mock_units(page, [_unit(uuid="cfg-flat", detectors=("flat_model",), subs=())])
        _mock_model_configs(
            page,
            [
                {
                    "uuid": "cfg-flat",
                    "name": "Flat Shape",
                    "description": "",
                    # No sub-model nodes: this shape defines no hierarchy.
                    "nodes": [
                        {
                            "uuid": "n-root",
                            "labelUuid": None,
                            "modelType": "object_detection",
                            "architecture": "yolo_v9_s",
                        }
                    ],
                }
            ],
        )
        _mock_labels(page)
        _make_panel(page)

        names = [i["name"] for i in _items(page)]
        assert "Flat Shape" not in names, names
        assert "flat_model" in names, names

    def test_a_hierarchy_with_untrained_sub_models_is_still_listed(self, shared_page: Page):
        """The counterpart to the filter above: "defines sub-models" is the test, not
        "has trained ones". Filtering on the artifacts would hide a hierarchy exactly
        when it most needs to say its sub-model is missing."""
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["test_nest"])
        _mock_units(page, [_unit(subs=())])
        _mock_model_configs(page, [_hier_config()])
        _mock_labels(page)
        _make_panel(page)

        by_name = {i["name"]: i for i in _items(page)}
        assert "Test Nest" in by_name, by_name
        assert by_name["Test Nest"]["badge"] == "top only"
        assert by_name["Test Nest"]["warn"]
