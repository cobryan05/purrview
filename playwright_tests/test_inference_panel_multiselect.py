"""
Tests for the multi-select detector list in the inference side panel —
apps/image_tagging_server/static/js/app/image-tagging/inferencePanel.js

Covers: multi-selecting detectors publishes all of them on Run, classifier
selection stays single/exclusive with detectors, and the checked set is
persisted to sessionStorage and reconciled against the live model list on
reload/refresh (stale models dropped, new models arrive unchecked, existing
checks survive).

Run with:
    pytest playwright_tests/test_inference_panel_multiselect.py -p no:xdist --override-ini="addopts=" -v
"""

import json

import pytest
from playwright.sync_api import Page


def _mock_models(page: Page, models: list, kinds: dict):
    """Mock GET /api/models and GET /api/models/<name>/meta for the given models."""
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


def _reset_session(page: Page):
    """Clear persisted selection. Call once at the start of each test, not on every
    _make_panel() call — the latter simulates *reopening* the panel within a session,
    so it must not wipe the very persistence it's meant to exercise."""
    page.evaluate("""() => {
        sessionStorage.removeItem('wtch_inference_selected_detectors');
        sessionStorage.removeItem('wtch_inference_selected_classifier');
    }""")


def _make_panel(page: Page):
    """
    Create a fresh container + inference panel bound to a dummy runtime, clearing
    the module-level model-kind/list cache so each call re-fetches from the current
    mocked /api/models response.
    """
    return page.evaluate("""async () => {
        const { createInferencePanel } = await import('/app-static/js/app/image-tagging/inferencePanel.js');
        const { clearModelsCache } = await import('/app-static/js/shared/api/models.js');
        clearModelsCache();

        document.getElementById('test-inference-panel')?.remove();
        const container = document.createElement('div');
        container.id = 'test-inference-panel';
        document.body.appendChild(container);

        const runtime = { id: 'test-runtime', state: {} };
        const handle = await createInferencePanel(runtime, container);
        window.__testPanelHandle = handle;
        return true;
    }""")


def _checkbox_names(page: Page):
    return page.eval_on_selector_all(
        "#test-inference-panel .inference-model-item input[type=checkbox]",
        "els => els.map(el => el.closest('.inference-model-item').querySelector('.inference-model-name').textContent)",
    )


def _check_model(page: Page, name: str, checked: bool = True):
    page.evaluate(
        """([name, checked]) => {
            const items = [...document.querySelectorAll('#test-inference-panel .inference-model-item')];
            const item = items.find(el => el.querySelector('.inference-model-name')?.textContent === name);
            const cb = item.querySelector('input[type=checkbox]');
            if (cb.checked !== checked) cb.click();
        }""",
        [name, checked],
    )


def _click_classifier(page: Page, name: str):
    page.evaluate(
        """(name) => {
            const items = [...document.querySelectorAll('#test-inference-panel .inference-model-item')];
            const item = items.find(el => el.querySelector('.inference-model-name')?.textContent === name);
            item.click();
        }""",
        name,
    )


def _capture_run_event(page: Page):
    """Subscribe to RUN_INFERENCE_ON_CANVAS for 'test-runtime' and click Run; return the payload."""
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


class TestMultiSelectDetectors:
    def test_checking_multiple_detectors_runs_all_of_them(self, shared_page: Page):
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["det-a", "det-b", "det-c"], {})
        _make_panel(page)

        _check_model(page, "det-a")
        _check_model(page, "det-b")

        payload = _capture_run_event(page)
        assert sorted(payload["modelNames"]) == ["det-a", "det-b"]

    def test_unchecking_a_detector_removes_it_from_run(self, shared_page: Page):
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["det-a", "det-b"], {})
        _make_panel(page)

        _check_model(page, "det-a")
        _check_model(page, "det-b")
        _check_model(page, "det-a", checked=False)

        payload = _capture_run_event(page)
        assert payload["modelNames"] == ["det-b"]

    def test_selecting_classifier_clears_detector_selection(self, shared_page: Page):
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["det-a", "clf-a"], {"clf-a": "image_classifier"})
        _make_panel(page)

        _check_model(page, "det-a")
        _click_classifier(page, "clf-a")

        selected_detectors = page.eval_on_selector_all(
            "#test-inference-panel .inference-model-item input[type=checkbox]:checked",
            "els => els.length",
        )
        assert selected_detectors == 0


class TestVerifyAllButton:
    def test_button_present_alongside_clear(self, shared_page: Page):
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["det-a"], {})
        _make_panel(page)

        labels = page.eval_on_selector_all(
            "#test-inference-panel .inference-buttons-row button",
            "els => els.map(el => el.textContent)",
        )
        assert "Verify all" in labels

    def test_click_marks_every_bbox_verified(self, shared_page: Page):
        """The button reuses canvas/image.js's verifyAllBboxes against whatever
        runtime the panel was constructed with — same helper the Labels panel's
        bulk-op row calls, exercised end-to-end here against a real canvas."""
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["det-a"], {})

        result = page.evaluate("""async () => {
            const { createImageTaggingRuntime } = await import('/app-static/js/app/image-tagging/runtime.js');
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const { rerenderBboxViews } = await import('/app-static/js/app/image-tagging/canvas/manager.js');
            const { createInferencePanel } = await import('/app-static/js/app/image-tagging/inferencePanel.js');
            const { clearModelsCache } = await import('/app-static/js/shared/api/models.js');
            clearModelsCache();

            const div = document.createElement('div');
            div.style.cssText = 'width:800px;height:600px;position:fixed;top:0;left:0;';
            document.body.appendChild(div);

            const runtime = createImageTaggingRuntime();
            runtime.setCanvasContainer(div);

            const imgCanvas = document.createElement('canvas');
            imgCanvas.width = 400; imgCanvas.height = 300;
            const img = new Image();
            img.src = imgCanvas.toDataURL();
            await new Promise(r => { img.onload = r; });
            runtime.setImage({
                uuid: 'img', filename: 'test.png', image_path: '/test.png',
                img, bboxes: new Map(), dirty: false,
            });

            runtime.state.image.bboxes.set('b1', createUIBBbox({
                uuid: 'b1', x: 0.1, y: 0.1, width: 0.2, height: 0.2,
                tags: [{ uuid: '00000000-0000-0000-0000-000000000001', icon: null }],
            }));
            rerenderBboxViews(runtime.canvas, runtime.state.image.bboxes, runtime.canvas.viewport);

            const container = document.createElement('div');
            document.body.appendChild(container);
            await createInferencePanel(runtime, container);

            const btn = [...container.querySelectorAll('.inference-buttons-row button')]
                .find(b => b.textContent === 'Verify all');
            btn.click();

            return runtime.state.image.bboxes.get('b1').tags.map(t => t.uuid);
        }""")
        assert "00000000-0000-0000-0000-000000000001" not in result  # UNVERIFIED
        assert "00000000-0000-0000-0000-000000000005" in result  # VERIFIED


class TestSessionPersistence:
    """
    These set sessionStorage directly (rather than driving checkboxes + reopening
    the panel) so the assertions test reconciliation-against-live-models in
    isolation, without interaction from the "first-ever-load defaults to the first
    detector" fallback covered separately below.
    """

    def test_first_ever_load_with_nothing_persisted_defaults_to_first_detector(self, shared_page: Page):
        page = shared_page
        _reset_session(page)
        _mock_models(page, ["det-a", "det-b"], {})
        _make_panel(page)

        payload = _capture_run_event(page)
        assert payload["modelNames"] == ["det-a"]

    def test_persisted_selection_is_restored_on_next_load(self, shared_page: Page):
        page = shared_page
        _reset_session(page)
        page.evaluate("() => sessionStorage.setItem('wtch_inference_selected_detectors', JSON.stringify(['det-b']))")
        _mock_models(page, ["det-a", "det-b"], {})
        _make_panel(page)

        payload = _capture_run_event(page)
        assert payload["modelNames"] == ["det-b"]

    def test_removed_model_is_dropped_without_disturbing_other_selections(self, shared_page: Page):
        page = shared_page
        _reset_session(page)
        page.evaluate(
            "() => sessionStorage.setItem('wtch_inference_selected_detectors', JSON.stringify(['det-a', 'det-b']))"
        )
        # det-b no longer exists.
        _mock_models(page, ["det-a"], {})
        _make_panel(page)

        payload = _capture_run_event(page)
        assert payload["modelNames"] == ["det-a"]

    def test_newly_added_model_arrives_unchecked(self, shared_page: Page):
        page = shared_page
        _reset_session(page)
        page.evaluate("() => sessionStorage.setItem('wtch_inference_selected_detectors', JSON.stringify(['det-a']))")

        # det-new appears alongside the previously-selected det-a.
        _mock_models(page, ["det-a", "det-new"], {})
        _make_panel(page)

        names = _checkbox_names(page)
        assert set(names) == {"det-a", "det-new"}

        payload = _capture_run_event(page)
        assert payload["modelNames"] == ["det-a"]
