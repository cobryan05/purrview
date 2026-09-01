"""
The canvas inference panel keeps up with the models directory —
apps/image_tagging_server/static/js/app/image-tagging/inferencePanel.js
+ notifyModelsChanged() in apps/image_tagging_server/static/js/shared/api/models.js

The panel is built once, on the first visit to Annotate, and then lives for the
rest of the session. The model list behind it is cached for the session too. So
a model made *after* that point — a finished training run, an import done on the
Configuration tab — did not exist as far as this panel was concerned until the
page was reloaded, and the manual ↻ button was the only way to see it.

Two signals now keep it current, and this pins both:

  - MODELS_UPDATED, for a change made elsewhere in this tab;
  - returning to Annotate, for everything else (a run that finished while the tab
    was closed, a file dropped into the models directory by hand).

Both assertions are conditional on the mocked list actually having changed: the
"before" check is what stops them passing against a panel that simply refetches
nothing, and against one that refetches through a cache nobody cleared.

Run with:
    pytest playwright_tests/test_inference_panel_model_refresh.py -p no:xdist --override-ini="addopts=" -v
"""

import json

import pytest
from playwright.sync_api import Page

from playwright_tests.waits import wait_until


def _mock_models(page: Page, state: dict):
    """GET /api/models served from a mutable list, so the test can add a model
    mid-session the way a finished training run does."""
    page.route(
        lambda url: url.split("?")[0].endswith("/api/models"),
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "models": state["models"]}),
            )
            if route.request.method == "GET"
            else route.fallback()
        ),
    )
    page.route(
        lambda url: "/api/models/" in url and url.endswith("/meta"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "modelKind": "detector"}),
        ),
    )


def _mock_model_groups(page: Page):
    page.route(
        lambda url: url.split("?")[0].endswith("/api/model-groups"),
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "groups": []}),
            )
            if route.request.method == "GET"
            else route.fallback()
        ),
    )


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
        window.__testPanelHandle = await createInferencePanel(runtime, container);
        return true;
    }""")


def _listed_models(page: Page) -> list:
    return page.evaluate("""() => [...document.querySelectorAll(
        '#test-inference-panel .inference-model-item .inference-model-name')].map(el => el.textContent)""")


@pytest.fixture(autouse=True)
def _cleanup(shared_page: Page):
    yield
    shared_page.evaluate("""() => {
        window.__testPanelHandle?.destroy?.();
        document.getElementById('test-inference-panel')?.remove();
        localStorage.removeItem('ww_inference_selected_detectors');
        localStorage.removeItem('ww_inference_selected_classifier');
        sessionStorage.removeItem('wtch_inference_selected_detectors');
        sessionStorage.removeItem('wtch_inference_selected_classifier');
    }""")
    shared_page.unroute_all(behavior="ignoreErrors")


class TestInferencePanelModelRefresh:
    def test_models_updated_event_adds_the_new_model(self, shared_page: Page):
        page = shared_page
        state = {"models": ["det-a"]}
        _mock_models(page, state)
        _mock_model_groups(page)
        _make_panel(page)

        assert _listed_models(page) == ["det-a"]

        state["models"] = ["det-a", "det-fresh"]
        page.evaluate("""async () => {
            const { notifyModelsChanged } = await import('/app-static/js/shared/api/models.js');
            notifyModelsChanged();
        }""")

        wait_until(
            page,
            lambda: "det-fresh" in _listed_models(page),
            "Expected MODELS_UPDATED to bring a newly created model into the panel",
        )

    def test_returning_to_annotate_picks_up_a_model_made_elsewhere(self, shared_page: Page):
        """The out-of-band case: nothing in this tab published anything, because
        nothing in this tab knew. Only the tab switch can catch it, and it has to
        drop the cache on the way — refetching through it would return the same
        list forever."""
        page = shared_page
        state = {"models": ["det-a"]}
        _mock_models(page, state)
        _mock_model_groups(page)
        _make_panel(page)

        assert _listed_models(page) == ["det-a"]

        state["models"] = ["det-a", "det-elsewhere"]
        # Leave, then return. Arriving without having left is the initial mount, which
        # deliberately does not refetch — the panel loaded the list moments earlier.
        page.evaluate("""async () => {
            const { events, EventTypes } = await import('/app-static/js/shared/events/index.js');
            events.publish(EventTypes.MAIN_TAB_CHANGED, { tabName: 'tab-configuration' });
            events.publish(EventTypes.MAIN_TAB_CHANGED, { tabName: 'tab-image-tagging' });
        }""")

        wait_until(
            page,
            lambda: "det-elsewhere" in _listed_models(page),
            "Expected returning to Annotate to re-read the model list",
        )

    def test_a_tab_switch_to_somewhere_else_does_not_reload(self, shared_page: Page):
        """The negative: only Annotate's own tab event triggers the refetch. A
        panel that reloaded on every MAIN_TAB_CHANGED would issue an /api/models
        request on every click in the tab bar for the life of the session."""
        page = shared_page
        state = {"models": ["det-a"]}
        _mock_models(page, state)
        _mock_model_groups(page)
        _make_panel(page)

        state["models"] = ["det-a", "det-should-not-appear"]
        page.evaluate("""async () => {
            const { events, EventTypes } = await import('/app-static/js/shared/events/index.js');
            events.publish(EventTypes.MAIN_TAB_CHANGED, { tabName: 'tab-configuration' });
        }""")
        page.wait_for_timeout(500)

        assert _listed_models(page) == ["det-a"]

    def test_arriving_on_annotate_without_having_left_does_not_refetch(self, shared_page: Page):
        """The panel is built during the first activation of Annotate, so the tab
        event that mounts it must not immediately re-read the list it just read.

        This is not micro-optimisation: those requests land inside the page-load
        settle that every Annotate test waits on, and one of them (/api/model-groups)
        is not mocked by most of them."""
        page = shared_page
        state = {"models": ["det-a"]}
        _mock_models(page, state)
        _mock_model_groups(page)
        _make_panel(page)

        state["models"] = ["det-a", "det-should-not-appear"]
        page.evaluate("""async () => {
            const { events, EventTypes } = await import('/app-static/js/shared/events/index.js');
            events.publish(EventTypes.MAIN_TAB_CHANGED, { tabName: 'tab-image-tagging' });
        }""")
        page.wait_for_timeout(500)

        assert _listed_models(page) == ["det-a"]
