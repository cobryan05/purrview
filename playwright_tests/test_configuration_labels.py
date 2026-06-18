"""
Tests for the Labels configuration UI.

Exercises the renderLabelList() → createLabelList() component rendered inside
Configuration > Labels & Tags > #labels-list:
  - DOM structure (list container present)
  - Label list loading (hierarchical labels appear)
  - Create root label flow (API called with correct payload)
  - Add child label flow
  - Edit label flow
  - Delete flow (confirm dialog; API called or suppressed)

Run with:
    pytest tests/playwright/test_configuration_labels.py -p no:xdist -v
"""
import json

import pytest
from playwright.sync_api import Page, Route, expect

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

ROOT_LABEL = {
    "uuid": "cccccccc-0000-0000-0000-000000000001",
    "name": "Cat",
    "color": "#ff0000",
    "parentUuid": None,
}

CHILD_LABEL = {
    "uuid": "cccccccc-0000-0000-0000-000000000002",
    "name": "Kitten",
    "color": "#00ff00",
    "parentUuid": "cccccccc-0000-0000-0000-000000000001",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _navigate_to_labels(page: Page, base_url: str):
    """Open the app and activate Configuration > Labels & Tags."""
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-configuration']").click()
    page.wait_for_load_state("networkidle")
    # Labels & Tags is the default sub-tab.


def _mock_labels(page: Page, labels: list):
    """Intercept GET /api/labels to return the given label list."""
    page.route(
        lambda url: url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "labels": labels}),
        ),
    )


# ---------------------------------------------------------------------------
# DOM structure
# ---------------------------------------------------------------------------


class TestLabelPaneStructure:
    def test_pane_structure(self, page: Page, base_url: str):
        """The labels pane renders its container and add-label control in one navigation."""
        _mock_labels(page, [])
        _navigate_to_labels(page, base_url)
        expect(page.locator("#labels-list")).to_be_visible()
        page.wait_for_selector("#labels-list button[title='Add label']", timeout=4000)
        expect(page.locator("#labels-list button[title='Add label']")).to_be_visible()

    def test_no_js_errors_on_labels_tab(self, page: Page, base_url: str):
        """No console errors during Labels pane load."""
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        _navigate_to_labels(page, base_url)
        page.wait_for_timeout(1000)
        uncaught = [e for e in errors if "TypeError" in e or "Uncaught" in e]
        assert uncaught == [], f"Uncaught JS errors: {uncaught}"


# ---------------------------------------------------------------------------
# Label list loading
# ---------------------------------------------------------------------------


class TestLabelListLoading:
    def test_root_labels_rendered(self, page: Page, base_url: str):
        """Root labels returned from the API appear as rows with the correct data-uuid."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

    def test_child_labels_rendered(self, page: Page, base_url: str):
        """Child labels appear nested under their parent."""
        _mock_labels(page, [ROOT_LABEL, CHILD_LABEL])
        _navigate_to_labels(page, base_url)
        # Both parent and child should be rendered
        expect(page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")).to_be_visible()
        expect(page.locator(f"#labels-list [data-uuid='{CHILD_LABEL['uuid']}']")).to_be_visible()

    def test_label_name_displayed(self, page: Page, base_url: str):
        """The label name is visible in the rendered row."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_contain_text(ROOT_LABEL["name"])

    def test_each_label_has_delete_button(self, page: Page, base_url: str):
        """Each label row provides a Delete button."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()
        expect(row.locator("button[title='Delete']")).to_have_count(1)

    def test_each_label_has_edit_button(self, page: Page, base_url: str):
        """Each label row provides an Edit button."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()
        expect(row.locator("button[title='Edit']")).to_have_count(1)

    def test_each_label_has_add_child_button(self, page: Page, base_url: str):
        """Each label row provides an 'Add Child' button to create sub-labels."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()
        expect(row.locator("button[title='Add Child']")).to_have_count(1)


# ---------------------------------------------------------------------------
# Create root label flow
# ---------------------------------------------------------------------------


class TestCreateRootLabel:
    def test_create_root_label_calls_api(self, page: Page, base_url: str):
        """Filling a name and clicking the Save button POSTs to /api/labels."""
        calls: list[dict] = []

        def handle_create(route: Route):
            if route.request.method == "POST":
                calls.append(json.loads(route.request.post_data))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "label": ROOT_LABEL}),
                )
            else:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labels": []}),
                )

        _mock_labels(page, [])
        page.route(
            lambda url: url.split("?")[0].endswith("/api/labels"),
            handle_create,
        )

        _navigate_to_labels(page, base_url)
        page.wait_for_selector("#labels-list input[placeholder='New Root Label Name']", timeout=4000)

        # Fill in the label name (the root-label create form is always visible)
        page.locator("#labels-list input[placeholder='New Root Label Name']").fill("Dog")

        # Click the 'Add label' button to submit (or the Save button inside the field)
        page.locator("#labels-list button[title='Add label']").click()
        page.wait_for_timeout(600)

        assert calls, "Expected POST to /api/labels"
        assert calls[0]["name"] == "Dog"
        assert calls[0].get("parentUuid") is None

    def test_create_root_label_empty_name_blocked(self, page: Page, base_url: str):
        """Clicking Save with an empty name does not call the API."""
        calls: list[bool] = []

        def handle_create(route: Route):
            if route.request.method == "POST":
                calls.append(True)
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps({"status": "success", "label": ROOT_LABEL}))
            else:
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps({"status": "success", "labels": []}))

        _mock_labels(page, [])
        page.route(lambda url: url.split("?")[0].endswith("/api/labels"), handle_create)

        _navigate_to_labels(page, base_url)
        # The create form is always visible; leave the name field empty
        page.wait_for_selector("#labels-list input[placeholder='New Root Label Name']", timeout=4000)

        # Click Add label without filling in a name
        page.locator("#labels-list button[title='Add label']").click()
        page.wait_for_timeout(500)

        assert not calls, "API must NOT be called when name is empty"


# ---------------------------------------------------------------------------
# Add child label flow
# ---------------------------------------------------------------------------


class TestAddChildLabel:
    def test_add_child_opens_inline_form(self, page: Page, base_url: str):
        """Clicking 'Add Child' on a label opens an inline child-create form."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

        row.locator("button[title='Add Child']").click()
        page.wait_for_timeout(400)

        # An inline input should appear somewhere after the parent row
        page.wait_for_selector("#labels-list input[placeholder='New Label Name']", timeout=3000)
        expect(page.locator("#labels-list input[placeholder='New Label Name']")).to_be_visible()

    def test_create_child_label_sends_parent_uuid(self, page: Page, base_url: str):
        """Creating a child label sends the parent's UUID in the API payload."""
        calls: list[dict] = []

        def handle_create(route: Route):
            if route.request.method == "POST":
                calls.append(json.loads(route.request.post_data))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "label": CHILD_LABEL}),
                )
            else:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labels": [ROOT_LABEL]}),
                )

        _mock_labels(page, [ROOT_LABEL])
        page.route(lambda url: url.split("?")[0].endswith("/api/labels"), handle_create)

        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

        row.locator("button[title='Add Child']").click()
        page.wait_for_selector("#labels-list input[placeholder='New Label Name']", timeout=3000)
        page.locator("#labels-list input[placeholder='New Label Name']").fill("Kitten")
        page.locator("#labels-list button[title='Save']").first.click()
        page.wait_for_timeout(600)

        assert calls, "Expected POST to /api/labels"
        assert calls[0].get("parentUuid") == ROOT_LABEL["uuid"]
        assert calls[0]["name"] == "Kitten"


# ---------------------------------------------------------------------------
# Delete label flow
# ---------------------------------------------------------------------------


class TestDeleteLabel:
    def test_delete_shows_confirm_dialog(self, page: Page, base_url: str):
        """Clicking Delete on a label opens a browser confirm() dialog."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

        dialogs: list[str] = []
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
        row.locator("button[title='Delete']").click()
        page.wait_for_timeout(400)

        assert dialogs, "Expected a confirm() dialog"

    def test_delete_accepted_calls_api(self, page: Page, base_url: str):
        """Accepting delete sends a DELETE request to /api/labels/{uuid}."""
        calls: list[str] = []

        def handle_all(route: Route):
            if route.request.method == "DELETE":
                calls.append(route.request.url)
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success"}),
                )
            else:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labels": [ROOT_LABEL]}),
                )

        _mock_labels(page, [ROOT_LABEL])
        page.route("**/api/labels/**", handle_all)

        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

        page.on("dialog", lambda d: d.accept())
        row.locator("button[title='Delete']").click()
        page.wait_for_timeout(600)

        assert calls, "Expected DELETE request to /api/labels/{uuid}"
        assert ROOT_LABEL["uuid"] in calls[0]

    def test_delete_dismissed_does_not_call_api(self, page: Page, base_url: str):
        """Dismissing the delete dialog does NOT send a DELETE request."""
        calls: list[bool] = []
        page.route("**/api/labels/**", lambda route: (
            calls.append(True) if route.request.method == "DELETE" else None,
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps({"status": "success", "labels": [ROOT_LABEL]})),
        ))
        _mock_labels(page, [ROOT_LABEL])

        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

        page.on("dialog", lambda d: d.dismiss())
        row.locator("button[title='Delete']").click()
        page.wait_for_timeout(500)

        assert not calls, "API must NOT be called when delete is cancelled"


# ---------------------------------------------------------------------------
# Label scope — edit-mode UI
# ---------------------------------------------------------------------------

IMAGE_SCOPE_LABEL = {
    "uuid": "cccccccc-0000-0000-0000-000000000010",
    "name": "Corrupt",
    "color": "#888888",
    "parentUuid": None,
    "scope": "image",
}

OBJECT_SCOPE_LABEL = {
    "uuid": "cccccccc-0000-0000-0000-000000000011",
    "name": "Cat",
    "color": "#ff0000",
    "parentUuid": None,
    "scope": "object",
}


class TestLabelScopeEditMode:
    def test_scope_selector_visible_in_new_label_row(self, page: Page, base_url: str):
        """The 'add new root label' row contains an Object/Image scope select."""
        _mock_labels(page, [])
        _navigate_to_labels(page, base_url)
        page.wait_for_selector("#labels-list input[placeholder='New Root Label Name']", timeout=4000)

        scope_sel = page.locator("#labels-list select[title*='Scope']")
        expect(scope_sel).to_be_visible()
        options = scope_sel.locator("option").all_text_contents()
        assert "Object" in options, f"Expected 'Object' option; got: {options}"
        assert "Image" in options, f"Expected 'Image' option; got: {options}"

    def test_create_image_scope_label_sends_scope_in_payload(self, page: Page, base_url: str):
        """Creating a new label with 'Image' scope selected sends scope='image' in POST."""
        calls: list[dict] = []

        def handle(route: Route):
            if route.request.method == "POST":
                calls.append(json.loads(route.request.post_data))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "label": IMAGE_SCOPE_LABEL}),
                )
            else:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labels": []}),
                )

        _mock_labels(page, [])
        page.route(lambda url: url.split("?")[0].endswith("/api/labels"), handle)
        _navigate_to_labels(page, base_url)
        page.wait_for_selector("#labels-list input[placeholder='New Root Label Name']", timeout=4000)

        # Switch scope to Image
        page.locator("#labels-list select[title*='Scope']").select_option("image")
        page.locator("#labels-list input[placeholder='New Root Label Name']").fill("Corrupt")
        page.locator("#labels-list button[title='Add label']").click()
        page.wait_for_timeout(600)

        assert calls, "Expected POST to /api/labels"
        assert calls[0].get("scope") == "image", \
            f"Expected scope='image' in POST payload; got: {calls[0]}"

    def test_create_object_scope_label_sends_scope_in_payload(self, page: Page, base_url: str):
        """Default scope (Object) is sent as scope='object' in POST."""
        calls: list[dict] = []

        def handle(route: Route):
            if route.request.method == "POST":
                calls.append(json.loads(route.request.post_data))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "label": OBJECT_SCOPE_LABEL}),
                )
            else:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labels": []}),
                )

        _mock_labels(page, [])
        page.route(lambda url: url.split("?")[0].endswith("/api/labels"), handle)
        _navigate_to_labels(page, base_url)
        page.wait_for_selector("#labels-list input[placeholder='New Root Label Name']", timeout=4000)

        page.locator("#labels-list input[placeholder='New Root Label Name']").fill("Cat")
        page.locator("#labels-list button[title='Add label']").click()
        page.wait_for_timeout(600)

        assert calls, "Expected POST to /api/labels"
        assert calls[0].get("scope") == "object", \
            f"Expected scope='object' in POST payload; got: {calls[0]}"

    def test_existing_image_scope_label_shows_img_selected(self, page: Page, base_url: str):
        """An existing label with scope='image' shows 'Img' selected in its scope dropdown."""
        _mock_labels(page, [IMAGE_SCOPE_LABEL])
        _navigate_to_labels(page, base_url)

        row = page.locator(f"#labels-list [data-uuid='{IMAGE_SCOPE_LABEL['uuid']}']")
        expect(row).to_be_visible()
        scope_sel = row.locator("select[title*='scope']")
        expect(scope_sel).to_have_value("image", timeout=3000)

    def test_changing_scope_to_image_calls_update_api(self, page: Page, base_url: str):
        """Switching the scope dropdown on an existing label sends scope in the PATCH payload."""
        calls: list[dict] = []

        def handle(route: Route):
            if route.request.method == "PATCH":
                calls.append(json.loads(route.request.post_data))
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "label": {**OBJECT_SCOPE_LABEL, "scope": "image"}}),
                )
            else:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labels": [OBJECT_SCOPE_LABEL]}),
                )

        _mock_labels(page, [OBJECT_SCOPE_LABEL])
        page.route("**/api/labels/**", handle)
        _navigate_to_labels(page, base_url)

        row = page.locator(f"#labels-list [data-uuid='{OBJECT_SCOPE_LABEL['uuid']}']")
        expect(row).to_be_visible()
        row.locator("select[title*='scope']").select_option("image")
        page.wait_for_timeout(600)

        assert calls, "Expected PATCH to /api/labels/{uuid}"
        assert calls[0].get("scope") == "image", \
            f"Expected scope='image' in PATCH payload; got: {calls[0]}"


# ---------------------------------------------------------------------------
# Label scope — select-mode behavior (canvas panel)
# ---------------------------------------------------------------------------


class TestLabelScopeSelectMode:
    """
    Test scope-gating in select mode via page.evaluate using createLabelList
    (the public export from labelList.js; mountSelectMode is not exported).
    setTool() in tools.js accesses runtime.canvas.stage.container() for cursor
    style, so the fake runtime must include that stub.
    """

    def _fake_runtime_js(self, runtime_id: str) -> str:
        """Return JS that creates a minimal runtime compatible with setTool."""
        return f"""{{
            id: '{runtime_id}',
            tool: 'select',
            state: {{ image: null }},
            canvas: {{
                layer: null,
                stage: {{ container: () => ({{ style: {{}} }}) }},
            }},
            setTool(t) {{ this._toolCalls.push(t); this.tool = t; }},
            markDirty() {{}},
            _toolCalls: [],
        }}"""

    def test_image_scope_label_click_does_not_set_tool(self, shared_page: Page):
        """Clicking a label with scope='image' must NOT call setTool (no bbox draw)."""
        result = shared_page.evaluate(f"""async () => {{
            const {{ labelsFetcher }} = await import('/app-static/js/shared/api/labels.js');
            const {{ createLabelList }} = await import('/app-static/js/ui/labelList.js');

            labelsFetcher.set('labelsMap', new Map([
                ['img-lbl-1', {{
                    uuid: 'img-lbl-1', name: 'Corrupt', color: '#888',
                    parentUuid: null, labelType: 'standard', scope: 'image',
                }}],
            ]));

            const runtime = {self._fake_runtime_js('rt-scope-img')};
            const container = document.createElement('div');
            document.body.appendChild(container);
            const handle = await createLabelList(container, {{ mode: 'select', runtime, filter: null }});

            const row = container.querySelector('[data-uuid="img-lbl-1"]');
            if (!row) {{ handle.destroy(); container.remove(); return {{ error: 'row not found', html: container.innerHTML }}; }}
            row.click();

            handle.destroy();
            container.remove();
            return {{ toolCalls: runtime._toolCalls }};
        }}""")
        assert result.get("toolCalls") == [], \
            f"Expected no setTool calls for image-scope label; got: {result}"

    def test_object_scope_label_click_sets_bbox_tool(self, shared_page: Page):
        """Clicking a label with scope='object' arms the bbox draw tool."""
        result = shared_page.evaluate(f"""async () => {{
            const {{ labelsFetcher }} = await import('/app-static/js/shared/api/labels.js');
            const {{ createLabelList }} = await import('/app-static/js/ui/labelList.js');

            labelsFetcher.set('labelsMap', new Map([
                ['obj-lbl-1', {{
                    uuid: 'obj-lbl-1', name: 'Cat', color: '#f00',
                    parentUuid: null, labelType: 'standard', scope: 'object',
                }}],
            ]));

            const runtime = {self._fake_runtime_js('rt-scope-obj')};
            const container = document.createElement('div');
            document.body.appendChild(container);
            const handle = await createLabelList(container, {{ mode: 'select', runtime, filter: null }});

            const row = container.querySelector('[data-uuid="obj-lbl-1"]');
            if (!row) {{ handle.destroy(); container.remove(); return {{ error: 'row not found' }}; }}
            row.click();

            handle.destroy();
            container.remove();
            return {{ toolCalls: runtime._toolCalls }};
        }}""")
        assert "bbox:obj-lbl-1" in result.get("toolCalls", []), \
            f"Expected setTool('bbox:obj-lbl-1'); got: {result}"

    def test_image_scope_labels_appear_under_image_classification_section(self, shared_page: Page):
        """Image-scope labels render under an 'Image Classification' section header."""
        result = shared_page.evaluate(f"""async () => {{
            const {{ labelsFetcher }} = await import('/app-static/js/shared/api/labels.js');
            const {{ createLabelList }} = await import('/app-static/js/ui/labelList.js');

            labelsFetcher.set('labelsMap', new Map([
                ['obj-lbl-2', {{
                    uuid: 'obj-lbl-2', name: 'Cat', color: '#f00',
                    parentUuid: null, labelType: 'standard', scope: 'object',
                }}],
                ['img-lbl-2', {{
                    uuid: 'img-lbl-2', name: 'Corrupt', color: '#888',
                    parentUuid: null, labelType: 'standard', scope: 'image',
                }}],
            ]));

            const runtime = {self._fake_runtime_js('rt-scope-sect')};
            const container = document.createElement('div');
            document.body.appendChild(container);
            const handle = await createLabelList(container, {{ mode: 'select', runtime, filter: null }});

            // Find the "Image Classification" section header by its text content.
            // (Object.assign on element.style normalizes hex colors to rgb(), so
            //  a style-attribute selector like [style*="88aaff"] won't match.)
            const header = Array.from(container.querySelectorAll('div'))
                .find(d => d.textContent.trim() === 'Image Classification') ?? null;
            const imgRow = container.querySelector('[data-uuid="img-lbl-2"]');
            const objRow = container.querySelector('[data-uuid="obj-lbl-2"]');

            handle.destroy();
            container.remove();
            return {{
                hasSectionHeader: !!header,
                sectionText: header?.textContent ?? null,
                hasImgRow: !!imgRow,
                hasObjRow: !!objRow,
            }};
        }}""")
        assert result.get("hasSectionHeader"), \
            f"Expected 'Image Classification' section header; got: {result}"
        assert "Image" in (result.get("sectionText") or ""), \
            f"Section header text should contain 'Image'; got: {result}"
        assert result.get("hasImgRow"), "Image-scope label row should be rendered"
        assert result.get("hasObjRow"), "Object-scope label row should be rendered"

    def test_object_scope_labels_not_in_image_classification_section(self, shared_page: Page):
        """With only object-scope labels, no 'Image Classification' header is shown."""
        result = shared_page.evaluate(f"""async () => {{
            const {{ labelsFetcher }} = await import('/app-static/js/shared/api/labels.js');
            const {{ createLabelList }} = await import('/app-static/js/ui/labelList.js');

            labelsFetcher.set('labelsMap', new Map([
                ['obj-only', {{
                    uuid: 'obj-only', name: 'Dog', color: '#0f0',
                    parentUuid: null, labelType: 'standard', scope: 'object',
                }}],
            ]));

            const runtime = {self._fake_runtime_js('rt-scope-noimg')};
            const container = document.createElement('div');
            document.body.appendChild(container);
            const handle = await createLabelList(container, {{ mode: 'select', runtime, filter: null }});

            const header = Array.from(container.querySelectorAll('div'))
                .find(d => d.textContent.trim() === 'Image Classification') ?? null;
            handle.destroy();
            container.remove();
            return {{ hasSectionHeader: !!header }};
        }}""")
        assert not result.get("hasSectionHeader"), \
            "Should NOT show 'Image Classification' header when no image-scope labels exist"
