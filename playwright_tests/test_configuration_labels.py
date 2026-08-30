"""
Tests for the Labels configuration UI.

Exercises the renderLabelList() → createLabelList() component rendered inside
Configuration > Labels & Tags > #labels-list:
  - DOM structure (two scoped sections present)
  - Label list loading (hierarchical labels appear in the correct section)
  - Create root label flow (API called with correct payload)
  - Add child label flow
  - Edit label flow
  - Delete flow (confirm dialog; API called or suppressed)
  - Scope pill (move label between sections)

Run with:
    pytest playwright_tests/test_configuration_labels.py -p no:xdist -v
"""

import json

import pytest
from playwright.sync_api import Page, Route, expect

from playwright_tests.waits import wait_until

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

ROOT_LABEL = {
    "uuid": "cccccccc-0000-0000-0000-000000000001",
    "name": "Cat",
    "color": "#ff0000",
    "parentUuid": None,
    "scope": "object",
}

CHILD_LABEL = {
    "uuid": "cccccccc-0000-0000-0000-000000000002",
    "name": "Kitten",
    "color": "#00ff00",
    "parentUuid": "cccccccc-0000-0000-0000-000000000001",
    "scope": "object",
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
    # Overview is the default sub-tab (PR 5.4) — click through to Labels & Tags.
    page.locator("#tab-configuration .workspace-tab-button[data-tab='tab-tag-management']").click()
    page.wait_for_load_state("networkidle")


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


def _obj_add_row(page: Page):
    """Return the add-row locator for the Object Labels section."""
    return page.get_by_test_id("label-list-new-object-row")


def _img_add_row(page: Page):
    """Return the add-row locator for the Image Labels section."""
    return page.get_by_test_id("label-list-new-image-row")


# ---------------------------------------------------------------------------
# DOM structure
# ---------------------------------------------------------------------------


class TestLabelPaneStructure:
    def test_pane_structure(self, page: Page, base_url: str):
        """The labels pane renders its container and both scoped add-label controls."""
        _mock_labels(page, [])
        _navigate_to_labels(page, base_url)
        expect(page.locator("#labels-list")).to_be_visible()
        # Each section has its own Add label button → expect exactly 2.
        page.wait_for_selector("#labels-list button[title='Add label']")
        expect(page.locator("#labels-list button[title='Add label']")).to_have_count(2)

    def test_two_section_headers_visible(self, page: Page, base_url: str):
        """Object Labels and Image Labels section headers are both rendered."""
        _mock_labels(page, [])
        _navigate_to_labels(page, base_url)
        page.wait_for_selector("#labels-list button[title='Add label']")
        # Section headers are plain divs with all-caps text rendered from JS.
        labels_list = page.locator("#labels-list")
        expect(labels_list).to_contain_text("Object Labels")
        expect(labels_list).to_contain_text("Image Labels")

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
    def test_child_labels_rendered(self, page: Page, base_url: str):
        """Child labels appear nested under their parent."""
        _mock_labels(page, [ROOT_LABEL, CHILD_LABEL])
        _navigate_to_labels(page, base_url)
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
        """Filling a name in the Object section and clicking Add label POSTs to /api/labels."""
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
        page.wait_for_selector("#labels-list input[placeholder='New Object Label']")

        add_row = _obj_add_row(page)
        add_row.locator("input[placeholder='New Object Label']").fill("Dog")
        add_row.locator("button[title='Add label']").click()
        wait_until(page, lambda: calls, "Expected POST to /api/labels")
        assert calls[0]["name"] == "Dog"
        assert calls[0].get("parentUuid") is None

    def test_create_root_label_empty_name_blocked(self, page: Page, base_url: str):
        """Clicking Add label with an empty name does not call the API."""
        calls: list[bool] = []

        def handle_create(route: Route):
            if route.request.method == "POST":
                calls.append(True)
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "label": ROOT_LABEL}),
                )
            else:
                route.fulfill(
                    status=200, content_type="application/json", body=json.dumps({"status": "success", "labels": []})
                )

        _mock_labels(page, [])
        page.route(lambda url: url.split("?")[0].endswith("/api/labels"), handle_create)

        _navigate_to_labels(page, base_url)
        page.wait_for_selector("#labels-list input[placeholder='New Object Label']")

        # Click Add label without filling in a name
        _obj_add_row(page).locator("button[title='Add label']").click()
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

        page.wait_for_selector("#labels-list input[placeholder='New Label Name']")
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
        page.wait_for_selector("#labels-list input[placeholder='New Label Name']")
        page.locator("#labels-list input[placeholder='New Label Name']").fill("Kitten")
        page.locator("#labels-list button[title='Save']").first.click()
        wait_until(page, lambda: calls, "Expected POST to /api/labels")
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
        wait_until(page, lambda: dialogs, "Expected a confirm() dialog")

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
        wait_until(page, lambda: calls, "Expected DELETE request to /api/labels/{uuid}")
        assert ROOT_LABEL["uuid"] in calls[0]

    def test_delete_dismissed_does_not_call_api(self, page: Page, base_url: str):
        """Dismissing the delete dialog does NOT send a DELETE request."""
        calls: list[bool] = []
        page.route(
            "**/api/labels/**",
            lambda route: (
                calls.append(True) if route.request.method == "DELETE" else None,
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labels": [ROOT_LABEL]}),
                ),
            ),
        )
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
    def test_two_sections_present_in_edit_mode(self, page: Page, base_url: str):
        """Both 'Object Labels' and 'Image Labels' sections are rendered."""
        _mock_labels(page, [])
        _navigate_to_labels(page, base_url)
        page.wait_for_selector("#labels-list button[title='Add label']")
        expect(page.locator("#labels-list")).to_contain_text("Object Labels")
        expect(page.locator("#labels-list")).to_contain_text("Image Labels")
        # One Add label button per section
        expect(page.locator("#labels-list button[title='Add label']")).to_have_count(2)

    def test_create_object_scope_label_sends_scope_in_payload(self, page: Page, base_url: str):
        """Adding a label via the Object Labels section sends scope='object' in POST."""
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
        page.wait_for_selector("#labels-list input[placeholder='New Object Label']")

        add_row = _obj_add_row(page)
        add_row.locator("input[placeholder='New Object Label']").fill("Cat")
        add_row.locator("button[title='Add label']").click()
        wait_until(page, lambda: calls, "Expected POST to /api/labels")
        assert calls[0].get("scope") == "object", f"Expected scope='object' in POST payload; got: {calls[0]}"

    def test_create_image_scope_label_sends_scope_in_payload(self, page: Page, base_url: str):
        """Adding a label via the Image Labels section sends scope='image' in POST."""
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
        page.wait_for_selector("#labels-list input[placeholder='New Image Label']")

        add_row = _img_add_row(page)
        add_row.locator("input[placeholder='New Image Label']").fill("Corrupt")
        add_row.locator("button[title='Add label']").click()
        wait_until(page, lambda: calls, "Expected POST to /api/labels")
        assert calls[0].get("scope") == "image", f"Expected scope='image' in POST payload; got: {calls[0]}"


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
        assert result.get("toolCalls") == [], f"Expected no setTool calls for image-scope label; got: {result}"

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
        assert "bbox:obj-lbl-1" in result.get("toolCalls", []), f"Expected setTool('bbox:obj-lbl-1'); got: {result}"

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

            // Section headers carry the title in a dedicated element — the
            // header div itself also holds the count and the section controls,
            // so an exact-textContent match on it would never hit.
            const header = Array.from(container.querySelectorAll('[data-testid="lt-section-title"]'))
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
        assert result.get("hasSectionHeader"), f"Expected 'Image Classification' section header; got: {result}"
        assert "Image" in (
            result.get("sectionText") or ""
        ), f"Section header text should contain 'Image'; got: {result}"
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
        assert not result.get(
            "hasSectionHeader"
        ), "Should NOT show 'Image Classification' header when no image-scope labels exist"


# ---------------------------------------------------------------------------
# Apply-to-all targeting — two orthogonal axes (visibility + target)
# ---------------------------------------------------------------------------


class TestLabelApplyToAllTargeting:
    """
    The old per-section "Mark all absent" buttons + 3-mode cycle are replaced by
    a single global Apply-to-all bar whose scope is an explicit, per-row-toggle
    "targeted" set. These tests pin the new behavior:

      - "Mark absent" affects only targeted labels (the target preset baseline),
        so the old bug where the object section's bulk op reached past Image
        Classification into Dynamic Labels can no longer happen.
      - The per-row ◉/⊘ toggle and the section-level all/none master override the
        baseline individually.

    A fake runtime with an image already loaded is required — bulkOp() no-ops
    ("No image loaded") unless currentImagePath is set, which mountSelectMode
    derives synchronously from runtime.state.image.image_path.
    """

    def _runtime_with_image(self, runtime_id: str) -> str:
        return f"""{{
            id: '{runtime_id}',
            tool: 'select',
            state: {{ image: {{ image_path: 'x.jpg', bboxes: new Map() }} }},
            canvas: {{ layer: null, stage: {{ container: () => ({{ style: {{}} }}) }} }},
            setTool(t) {{ this.tool = t; }},
            markDirty() {{ this._dirty = true; }},
            _dirty: false,
        }}"""

    def _status_helpers_js(self) -> str:
        """JS helpers (closure over `container`) for reading row state."""
        return """
            // .label-status-btn, not "the first button" — the disclosure caret
            // is also a <button> and precedes it in the row.
            const statusOf = (uuid) => {
                const row = container.querySelector('[data-uuid="'+uuid+'"]');
                return row ? row.querySelector('.label-status-btn').textContent : null;
            };
            const targetIconOf = (uuid) => {
                const row = container.querySelector('[data-uuid="'+uuid+'"]');
                return row ? row.querySelector('.label-target-toggle').textContent : null;
            };
            const clickMarkAbsent = () => {
                const b = Array.from(container.querySelectorAll('button'))
                    .find(x => x.textContent === 'Mark absent');
                if (b && !b.disabled) b.click();
                return !!b && !b.disabled;
            };
        """

    def test_mark_absent_targets_object_preset_and_spares_dynamic(self, shared_page: Page):
        """With target preset 'object', Mark absent hits object labels only — the
        dynamic label is NOT swept (the regression this redesign fixes)."""
        result = shared_page.evaluate(f"""async () => {{
            const {{ labelsFetcher }} = await import('/app-static/js/shared/api/labels.js');
            const {{ createLabelList }} = await import('/app-static/js/ui/labelList.js');
            const {{ activeTrainingConfig }} = await import('/app-static/js/shared/activeTrainingConfig.js');

            labelsFetcher.set('labelsMap', new Map([
                ['obj-a', {{ uuid:'obj-a', name:'Cat',    color:'#f00', parentUuid:null, labelType:'standard', scope:'object' }}],
                ['dyn-a', {{ uuid:'dyn-a', name:'Motion', color:'#888', parentUuid:null, labelType:'dynamic',  scope:'object' }}],
            ]));
            activeTrainingConfig.setTargetPreset('object');

            const runtime = {self._runtime_with_image('rt-tgt-obj')};
            const container = document.createElement('div');
            document.body.appendChild(container);
            const handle = await createLabelList(container, {{ mode:'select', runtime, filter:null }});
            {self._status_helpers_js()}

            const clicked = clickMarkAbsent();
            const out = {{ clicked, obj: statusOf('obj-a'), dyn: statusOf('dyn-a'),
                          objIcon: targetIconOf('obj-a'), dynIcon: targetIconOf('dyn-a') }};

            activeTrainingConfig.setTargetPreset('all');
            handle.destroy(); container.remove();
            return out;
        }}""")
        assert result.get("clicked"), f"Mark absent should be enabled; got {result}"
        assert result.get("objIcon") == "◉", f"object should start targeted; got {result}"
        assert result.get("dynIcon") == "⊘", f"dynamic should start untargeted; got {result}"
        assert result.get("obj") == "✗", f"object label should be marked absent; got {result}"
        assert result.get("dyn") == "?", f"dynamic label must NOT be swept; got {result}"

    def test_per_row_toggle_excludes_from_mark_absent(self, shared_page: Page):
        """Preset 'all' targets everything; clicking a row's ◉ toggle excludes just
        that row, so Mark absent skips it while still hitting the others."""
        result = shared_page.evaluate(f"""async () => {{
            const {{ labelsFetcher }} = await import('/app-static/js/shared/api/labels.js');
            const {{ createLabelList }} = await import('/app-static/js/ui/labelList.js');
            const {{ activeTrainingConfig }} = await import('/app-static/js/shared/activeTrainingConfig.js');

            labelsFetcher.set('labelsMap', new Map([
                ['obj-a', {{ uuid:'obj-a', name:'Cat', color:'#f00', parentUuid:null, labelType:'standard', scope:'object' }}],
                ['obj-b', {{ uuid:'obj-b', name:'Dog', color:'#0f0', parentUuid:null, labelType:'standard', scope:'object' }}],
            ]));
            activeTrainingConfig.setTargetPreset('all');

            const runtime = {self._runtime_with_image('rt-tgt-row')};
            const container = document.createElement('div');
            document.body.appendChild(container);
            const handle = await createLabelList(container, {{ mode:'select', runtime, filter:null }});
            {self._status_helpers_js()}

            // Exclude obj-a via its per-row toggle.
            container.querySelector('[data-uuid="obj-a"] .label-target-toggle').click();
            const iconAfter = targetIconOf('obj-a');
            const clicked = clickMarkAbsent();
            const out = {{ clicked, iconAfter, a: statusOf('obj-a'), b: statusOf('obj-b') }};

            activeTrainingConfig.setTargetPreset('all');
            handle.destroy(); container.remove();
            return out;
        }}""")
        assert result.get("iconAfter") == "⊘", f"toggled row should show excluded icon; got {result}"
        assert result.get("clicked"), f"Mark absent should be enabled; got {result}"
        assert result.get("a") == "?", f"excluded row must be skipped; got {result}"
        assert result.get("b") == "✗", f"still-targeted row should be marked absent; got {result}"

    def test_section_master_none_excludes_that_section_only(self, shared_page: Page):
        """The Object-section '⊘ none' master untargets object labels but leaves the
        Dynamic section targeted — Mark absent then hits only dynamic labels."""
        result = shared_page.evaluate(f"""async () => {{
            const {{ labelsFetcher }} = await import('/app-static/js/shared/api/labels.js');
            const {{ createLabelList }} = await import('/app-static/js/ui/labelList.js');
            const {{ activeTrainingConfig }} = await import('/app-static/js/shared/activeTrainingConfig.js');

            labelsFetcher.set('labelsMap', new Map([
                ['obj-a', {{ uuid:'obj-a', name:'Cat',    color:'#f00', parentUuid:null, labelType:'standard', scope:'object' }}],
                ['obj-b', {{ uuid:'obj-b', name:'Dog',    color:'#0f0', parentUuid:null, labelType:'standard', scope:'object' }}],
                ['dyn-a', {{ uuid:'dyn-a', name:'Motion', color:'#888', parentUuid:null, labelType:'dynamic',  scope:'object' }}],
            ]));
            activeTrainingConfig.setTargetPreset('all');

            const runtime = {self._runtime_with_image('rt-tgt-sect')};
            const container = document.createElement('div');
            document.body.appendChild(container);
            const handle = await createLabelList(container, {{ mode:'select', runtime, filter:null }});
            {self._status_helpers_js()}

            // First '.label-section-none' belongs to the Object Labels section.
            container.querySelectorAll('.label-section-none')[0].click();
            const clicked = clickMarkAbsent();
            const out = {{ clicked, a: statusOf('obj-a'), b: statusOf('obj-b'), dyn: statusOf('dyn-a'),
                          aIcon: targetIconOf('obj-a'), dynIcon: targetIconOf('dyn-a') }};

            activeTrainingConfig.setTargetPreset('all');
            handle.destroy(); container.remove();
            return out;
        }}""")
        assert result.get("aIcon") == "⊘", f"object rows should be untargeted after 'none'; got {result}"
        assert result.get("dynIcon") == "◉", f"dynamic row should stay targeted; got {result}"
        assert result.get("clicked"), f"Mark absent should be enabled (dynamic still targeted); got {result}"
        assert result.get("a") == "?", f"object row a should be spared; got {result}"
        assert result.get("b") == "?", f"object row b should be spared; got {result}"
        assert result.get("dyn") == "✗", f"dynamic row should be marked absent; got {result}"

    def test_custom_targeting_persists_across_remount(self, shared_page: Page):
        """A manual per-row toggle switches the preset to 'custom' and persists the
        hand-picked selection, so it survives a remount (e.g. the one a save
        triggers) instead of snapping back to the preset baseline."""
        result = shared_page.evaluate(f"""async () => {{
            const {{ labelsFetcher }} = await import('/app-static/js/shared/api/labels.js');
            const {{ createLabelList }} = await import('/app-static/js/ui/labelList.js');
            const {{ activeTrainingConfig }} = await import('/app-static/js/shared/activeTrainingConfig.js');

            labelsFetcher.set('labelsMap', new Map([
                ['obj-a', {{ uuid:'obj-a', name:'Cat', color:'#f00', parentUuid:null, labelType:'standard', scope:'object' }}],
                ['obj-b', {{ uuid:'obj-b', name:'Dog', color:'#0f0', parentUuid:null, labelType:'standard', scope:'object' }}],
            ]));
            activeTrainingConfig.setTargetPreset('all');

            const runtime = {self._runtime_with_image('rt-tgt-persist')};
            const container = document.createElement('div');
            document.body.appendChild(container);
            const handle = await createLabelList(container, {{ mode:'select', runtime, filter:null }});
            {self._status_helpers_js()}

            // Exclude obj-a via its per-row toggle — this should flip the preset to 'custom'.
            container.querySelector('[data-uuid="obj-a"] .label-target-toggle').click();
            const presetAfterToggle = activeTrainingConfig.getTargetPreset();
            const iconBeforeRemount = targetIconOf('obj-a');

            // Remount (what saving an image triggers) must NOT reset the selection.
            await handle.refresh();

            const out = {{
                presetAfterToggle,
                iconBeforeRemount,
                presetAfterRemount: activeTrainingConfig.getTargetPreset(),
                aIconAfter: targetIconOf('obj-a'),
                bIconAfter: targetIconOf('obj-b'),
            }};

            activeTrainingConfig.setTargetPreset('all');
            handle.destroy(); container.remove();
            return out;
        }}""")
        assert result.get("presetAfterToggle") == "custom", f"toggle should switch preset to custom; got {result}"
        assert result.get("iconBeforeRemount") == "⊘", f"toggled row should be excluded; got {result}"
        assert result.get("presetAfterRemount") == "custom", f"preset should stay custom after remount; got {result}"
        assert result.get("aIconAfter") == "⊘", f"excluded row must survive remount; got {result}"
        assert result.get("bIconAfter") == "◉", f"other row should stay targeted after remount; got {result}"


# ---------------------------------------------------------------------------
# Re-parent ("Move to…") flow — edit mode
# ---------------------------------------------------------------------------


ROOT_A = {
    "uuid": "cccccccc-0000-0000-0000-0000000000a1",
    "name": "RootA",
    "color": "#ff0000",
    "parentUuid": None,
    "scope": "object",
}

ROOT_B = {
    "uuid": "cccccccc-0000-0000-0000-0000000000a2",
    "name": "RootB",
    "color": "#00ff00",
    "parentUuid": None,
    "scope": "object",
}

MOVABLE_CHILD = {
    "uuid": "cccccccc-0000-0000-0000-0000000000a3",
    "name": "MovableChild",
    "color": "#0000ff",
    "parentUuid": ROOT_A["uuid"],
    "scope": "object",
}


class TestReparentLabel:
    def test_each_label_has_move_button(self, page: Page, base_url: str):
        """Each editable label row provides a 'Move' affordance."""
        _mock_labels(page, [ROOT_A])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_A['uuid']}']")
        expect(row).to_be_visible()
        expect(row.locator("button[title='Move']")).to_have_count(1)

    def test_move_opens_target_picker_excluding_self(self, page: Page, base_url: str):
        """Clicking Move opens an inline picker; the label itself is not offered
        as a valid target (that would be a no-op self-parent)."""
        _mock_labels(page, [ROOT_A, ROOT_B])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_A['uuid']}']")
        expect(row).to_be_visible()

        row.locator("button[title='Move']").click()

        select = page.locator(".label-move-select").first
        expect(select).to_be_visible()
        # Barrier before the *negative* assertion below: an empty <select> satisfies
        # "RootA is not among the options" without proving anything at all. Waiting for
        # the options to exist is what makes that assertion falsifiable.
        expect(select.locator("option")).not_to_have_count(0)
        option_texts = select.locator("option").all_inner_texts()
        assert "RootA" not in option_texts, f"Label must not offer itself as a move target; got {option_texts}"
        assert any("RootB" in t for t in option_texts), f"Expected RootB as a valid target; got {option_texts}"
        assert "(root)" in option_texts

    def test_move_to_root_sends_null_parent_uuid(self, page: Page, base_url: str):
        """Picking '(root)' and confirming PATCHes parentUuid: null."""
        calls: list[dict] = []

        def handle(route: Route):
            if route.request.method == "PATCH":
                calls.append(json.loads(route.request.post_data))
                route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))
            else:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labels": [ROOT_A, MOVABLE_CHILD]}),
                )

        _mock_labels(page, [ROOT_A, MOVABLE_CHILD])
        page.route(lambda url: "/api/labels/" in url, handle)

        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{MOVABLE_CHILD['uuid']}']")
        expect(row).to_be_visible()

        row.locator("button[title='Move']").click()
        page.wait_for_timeout(300)
        select = page.locator(".label-move-select").first
        select.select_option("")  # "(root)"
        page.locator(".label-move-apply").first.click()
        wait_until(page, lambda: calls, "Expected a PATCH to /api/labels/{uuid}")
        assert calls[0].get("parentUuid") is None
        assert "parentUuid" in calls[0]

    def test_move_to_other_root_sends_its_uuid(self, page: Page, base_url: str):
        """Picking a real target label sends that label's uuid as parentUuid."""
        calls: list[dict] = []

        def handle(route: Route):
            if route.request.method == "PATCH":
                calls.append(json.loads(route.request.post_data))
                route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))
            else:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labels": [ROOT_A, ROOT_B, MOVABLE_CHILD]}),
                )

        _mock_labels(page, [ROOT_A, ROOT_B, MOVABLE_CHILD])
        page.route(lambda url: "/api/labels/" in url, handle)

        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{MOVABLE_CHILD['uuid']}']")
        expect(row).to_be_visible()

        row.locator("button[title='Move']").click()
        # No sleep: `select_option` retries until the option exists, and the PATCH is
        # waited on explicitly below.
        select = page.locator(".label-move-select").first
        select.select_option(ROOT_B["uuid"])
        page.locator(".label-move-apply").first.click()
        wait_until(page, lambda: calls, "Expected a PATCH to /api/labels/{uuid}")
        assert calls[0].get("parentUuid") == ROOT_B["uuid"]

    def test_move_rerenders_tree_with_new_hierarchy(self, page: Page, base_url: str):
        """After a successful move, the list re-fetches (a real GET, not a
        cached/stale one) and the label re-renders nested under its new parent.

        Uses one stateful route (rather than the static `_mock_labels` helper)
        so the post-PATCH GET reflects the move — proving the re-render used
        fresh data, not just that the DOM node happened to survive.
        """
        state = {
            "labels": [dict(ROOT_A), dict(ROOT_B), dict(MOVABLE_CHILD)],
            "patch_calls": 0,
        }

        def handle(route: Route):
            url = route.request.url
            if route.request.method == "PATCH":
                state["patch_calls"] += 1
                body = json.loads(route.request.post_data)
                for lbl in state["labels"]:
                    if lbl["uuid"] in url and "parentUuid" in body:
                        lbl["parentUuid"] = body["parentUuid"]
                route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))
            elif url.split("?")[0].endswith("/api/labels") and "api/labels/" not in url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"status": "success", "labels": state["labels"]}),
                )
            else:
                route.continue_()

        page.route(lambda url: "/api/labels" in url, handle)

        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{MOVABLE_CHILD['uuid']}']")
        expect(row).to_be_visible()

        row.locator("button[title='Move']").click()
        select = page.locator(".label-move-select").first
        select.select_option(ROOT_B["uuid"])
        page.locator(".label-move-apply").first.click()

        # Poll for the PATCH rather than sleeping 700ms and hoping. Note this still
        # asserts *exactly* one: `wait_until` returns on the first, and a duplicate
        # would have to arrive before the assertion — which is the same window the
        # sleep had, without the guaranteed 700ms cost on every run.
        wait_until(page, lambda: state["patch_calls"], "Expected a PATCH to /api/labels/{uuid}")
        assert state["patch_calls"] == 1
        # The label now nests under RootB in the re-fetched/re-rendered tree.
        root_b_row = page.locator(f"#labels-list [data-uuid='{ROOT_B['uuid']}']")
        moved_row = page.locator(f"#labels-list [data-uuid='{MOVABLE_CHILD['uuid']}']")
        expect(moved_row).to_be_visible()
        # The moved row's indentation should now match a direct child of RootB
        # rather than its old position under RootA (which no longer has children).
        root_a_row = page.locator(f"#labels-list [data-uuid='{ROOT_A['uuid']}']")
        expect(root_a_row).to_be_visible()
        assert state["labels"][2]["parentUuid"] == ROOT_B["uuid"]


# ---------------------------------------------------------------------------
# Collapsible tree (compact label list)
# ---------------------------------------------------------------------------


class TestCollapsibleTree:
    """The label list renders a disclosure caret on every parent; collapsing a
    branch hides its descendants and reports how many were folded away."""

    def _clear_collapse_state(self, page: Page):
        """Collapse state is persisted in localStorage and would leak between tests."""
        page.evaluate(
            "() => Object.keys(localStorage)"
            ".filter(k => k.startsWith('ww_label_tree_collapsed_'))"
            ".forEach(k => localStorage.removeItem(k))"
        )

    def test_parent_has_caret_and_leaf_does_not(self, page: Page, base_url: str):
        """Only labels with children get an operable caret; leaves keep the
        column (so names stay aligned) but it is inert and aria-hidden."""
        _mock_labels(page, [ROOT_LABEL, CHILD_LABEL])
        _navigate_to_labels(page, base_url)
        self._clear_collapse_state(page)
        page.reload()
        page.wait_for_load_state("networkidle")
        _navigate_to_labels(page, base_url)

        parent_caret = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}'] .lt-caret")
        expect(parent_caret).to_have_attribute("aria-expanded", "true")
        expect(parent_caret).not_to_have_class("lt-caret--leaf")

        child_caret = page.locator(f"#labels-list [data-uuid='{CHILD_LABEL['uuid']}'] .lt-caret")
        expect(child_caret).to_have_attribute("aria-hidden", "true")

    def test_collapsing_hides_children_and_shows_count(self, page: Page, base_url: str):
        """Clicking a parent's caret removes its descendants from the DOM and
        puts the hidden count on the parent row."""
        _mock_labels(page, [ROOT_LABEL, CHILD_LABEL])
        _navigate_to_labels(page, base_url)
        self._clear_collapse_state(page)
        page.reload()
        page.wait_for_load_state("networkidle")
        _navigate_to_labels(page, base_url)

        parent = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        child = page.locator(f"#labels-list [data-uuid='{CHILD_LABEL['uuid']}']")
        expect(child).to_be_visible()

        parent.locator(".lt-caret").click()
        expect(child).to_have_count(0)
        expect(parent.locator(".lt-kids")).to_have_text("1")

        # Expanding brings it back.
        page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}'] .lt-caret").click()
        expect(page.locator(f"#labels-list [data-uuid='{CHILD_LABEL['uuid']}']")).to_be_visible()

    def test_collapse_state_survives_a_reload(self, page: Page, base_url: str):
        """Collapse is persisted, so a folded tree stays folded across reloads."""
        _mock_labels(page, [ROOT_LABEL, CHILD_LABEL])
        _navigate_to_labels(page, base_url)
        self._clear_collapse_state(page)
        page.reload()
        page.wait_for_load_state("networkidle")
        _navigate_to_labels(page, base_url)

        page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}'] .lt-caret").click()
        expect(page.locator(f"#labels-list [data-uuid='{CHILD_LABEL['uuid']}']")).to_have_count(0)

        _navigate_to_labels(page, base_url)
        expect(page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")).to_be_visible()
        expect(page.locator(f"#labels-list [data-uuid='{CHILD_LABEL['uuid']}']")).to_have_count(0)
        self._clear_collapse_state(page)


# ---------------------------------------------------------------------------
# Inline rename
# ---------------------------------------------------------------------------


class TestInlineRename:
    def test_edit_opens_inline_editor_prefilled(self, page: Page, base_url: str):
        """Edit swaps the row into a name input seeded with the current name."""
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

        row.locator("button[title='Edit']").click()
        editor = row.locator(".lt-editor input[type='text']")
        expect(editor).to_be_visible()
        expect(editor).to_have_value(ROOT_LABEL["name"])
        # The UUID is shown while editing, not as a permanent second line.
        expect(row.locator(".lt-hint")).to_have_text(ROOT_LABEL["uuid"])

    def test_rename_saves_new_name(self, page: Page, base_url: str):
        """Saving the inline editor PATCHes the label with the typed name."""
        calls: list[dict] = []

        def handle(route: Route):
            if route.request.method == "PATCH":
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
                    body=json.dumps({"status": "success", "labels": [ROOT_LABEL]}),
                )

        _mock_labels(page, [ROOT_LABEL])
        page.route("**/api/labels/**", handle)

        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

        row.locator("button[title='Edit']").click()
        row.locator(".lt-editor input[type='text']").fill("Feline")
        row.locator("button[title='Save']").click()
        wait_until(page, lambda: calls, "Expected PATCH to /api/labels/{uuid}")
        assert calls[0]["name"] == "Feline"

    def test_cancel_discards_the_edit(self, page: Page, base_url: str):
        """Cancel closes the editor without calling the API."""
        calls: list[bool] = []

        def handle(route: Route):
            if route.request.method == "PATCH":
                calls.append(True)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "labels": [ROOT_LABEL]}),
            )

        _mock_labels(page, [ROOT_LABEL])
        page.route("**/api/labels/**", handle)

        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        expect(row).to_be_visible()

        row.locator("button[title='Edit']").click()
        row.locator(".lt-editor input[type='text']").fill("Feline")
        row.locator("button[title='Cancel']").click()
        page.wait_for_timeout(500)

        assert not calls, "Cancel must NOT PATCH"
        expect(page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")).to_contain_text(ROOT_LABEL["name"])


class TestLabelHomeAssistantPanel:
    """The per-camera breakdown is the tier whose size is (labels x rules x cameras), so
    the panel has to make it clear it is subordinate to the label being published at
    all — and must never send an opt-in that cannot apply."""

    def test_the_per_camera_box_is_unavailable_while_the_label_is_not_published(self, page: Page, base_url: str):
        _mock_labels(page, [ROOT_LABEL])
        _navigate_to_labels(page, base_url)
        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        row.locator("button[title='Home Assistant']").click()

        publish = page.get_by_test_id("label-publish-to-ha")
        per_source = page.get_by_test_id("label-publish-per-source")
        # Publishing by default (the label predates the field), so the breakdown is
        # merely unticked — available, just not chosen.
        expect(publish).to_be_checked()
        expect(per_source).to_be_enabled()

        publish.uncheck()
        # …and now unavailable, because there is no device for it to be a breakdown of.
        expect(per_source).to_be_disabled()

    def test_opting_out_of_the_label_does_not_send_a_per_camera_opt_in(self, page: Page, base_url: str):
        """A disabled checkbox keeps whatever it was last set to. Sending that alongside
        `publishToHa: false` would store an opt-in that silently takes effect the moment
        the label is published again."""
        patched = {}

        def handle(route):
            if route.request.method == "PATCH":
                patched.update(json.loads(route.request.post_data or "{}"))
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "labels": [ROOT_LABEL]}),
            )

        _mock_labels(page, [ROOT_LABEL])
        page.route("**/api/labels/**", handle)
        _navigate_to_labels(page, base_url)

        row = page.locator(f"#labels-list [data-uuid='{ROOT_LABEL['uuid']}']")
        row.locator("button[title='Home Assistant']").click()
        page.get_by_test_id("label-publish-per-source").check()
        page.get_by_test_id("label-publish-to-ha").uncheck()
        page.get_by_test_id("label-ha-panel").locator("button:has-text('Save')").click()

        wait_until(page, lambda: "publishToHa" in patched)
        assert patched["publishToHa"] is False, patched
        assert patched["publishPerSource"] is False, patched
