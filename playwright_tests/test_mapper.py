"""
Tests for shared/domain/image/mapper.js

createUIBBbox and bboxReadToUI are pure data transformers; they do import
BboxView (which references the Konva global) so we must navigate to base_url
first so Konva is available.

All tests share one pre-navigated `shared_page` (module-scoped, see
conftest.py) instead of reloading the SPA per test — they only call pure
functions and touch no DOM/SPA state, so a single navigation suffices.
"""

import pytest
from playwright.sync_api import Page

UNVERIFIED_TAG_UUID = "00000000-0000-0000-0000-000000000001"


# ---------------------------------------------------------------------------
# createUIBBbox
# ---------------------------------------------------------------------------


class TestCreateUIBBbox:
    def test_defaults(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            const bbox = createUIBBbox();
            return {
                hasUuid: typeof bbox.uuid === 'string' && bbox.uuid.length > 0,
                label: bbox.label,
                tags: bbox.tags,
                x: bbox.x, y: bbox.y, width: bbox.width, height: bbox.height,
                selected: bbox.selected,
                dirty: bbox.dirty,
            };
        }""")
        assert result["hasUuid"] is True
        assert result["label"] is None
        assert result["tags"] == []
        assert result["x"] == 0
        assert result["y"] == 0
        assert result["width"] == 0
        assert result["height"] == 0
        assert result["selected"] is False
        assert result["dirty"] is True

    def test_overrides_applied(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            return createUIBBbox({
                uuid: 'my-uuid',
                x: 0.1, y: 0.2, width: 0.3, height: 0.4,
                label: { uuid: 'lbl', text: 'Cat', color: '#f00' },
                selected: true,
                dirty: false,
            });
        }""")
        assert result["uuid"] == "my-uuid"
        assert result["x"] == pytest.approx(0.1)
        assert result["y"] == pytest.approx(0.2)
        assert result["width"] == pytest.approx(0.3)
        assert result["height"] == pytest.approx(0.4)
        assert result["label"]["uuid"] == "lbl"
        assert result["selected"] is True
        assert result["dirty"] is False

    def test_partial_overrides_leave_defaults(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { createUIBBbox } = await import('/app-static/js/shared/domain/image/mapper.js');
            return createUIBBbox({ x: 0.5 });
        }""")
        assert result["x"] == pytest.approx(0.5)
        assert result["y"] == 0  # default
        assert result["tags"] == []  # default


# ---------------------------------------------------------------------------
# bboxReadToUI
# ---------------------------------------------------------------------------


class TestBboxReadToUI:
    def test_with_label_maps_fields(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { bboxReadToUI } = await import('/app-static/js/shared/domain/image/mapper.js');
            const raw = {
                uuid: 'bbox-1',
                x: 0.1, y: 0.2, width: 0.3, height: 0.4,
                label: { uuid: 'lbl-uuid', name: 'Cat', color: '#ff0000' },
                tags: [],
            };
            const result = bboxReadToUI(raw);
            return {
                uuid: result.uuid,
                label: result.label,
                x: result.x, y: result.y,
                dirty: result.dirty,
                selected: result.selected,
            };
        }""")
        assert result["uuid"] == "bbox-1"
        assert result["label"]["uuid"] == "lbl-uuid"
        assert result["label"]["text"] == "Cat"
        assert result["label"]["color"] == "#ff0000"
        assert result["x"] == pytest.approx(0.1)
        assert result["dirty"] is False
        assert result["selected"] is False

    def test_no_label_no_verification_tag_adds_unverified(self, shared_page: Page):
        """A bbox with no label and no existing verification tag gets UNVERIFIED added."""
        page = shared_page
        result = page.evaluate("""async () => {
            const { bboxReadToUI } = await import('/app-static/js/shared/domain/image/mapper.js');
            const raw = { uuid: 'b', x: 0, y: 0, width: 0.1, height: 0.1, label: null, tags: [] };
            const ui = bboxReadToUI(raw);
            return ui.tags.map(t => t.uuid);
        }""")
        assert UNVERIFIED_TAG_UUID in result

    def test_with_label_does_not_add_unverified(self, shared_page: Page):
        """A human-assigned label means the bbox was reviewed — UNVERIFIED stays out."""
        page = shared_page
        result = page.evaluate(f"""async () => {{
            const {{ bboxReadToUI }} = await import('/app-static/js/shared/domain/image/mapper.js');
            const raw = {{
                uuid: 'b', x: 0, y: 0, width: 0.1, height: 0.1,
                label: {{ uuid: 'lbl', name: 'Cat', color: '#f00' }},
                tags: [],
            }};
            const ui = bboxReadToUI(raw);
            return ui.tags.map(t => t.uuid);
        }}""")
        assert UNVERIFIED_TAG_UUID not in result

    def test_existing_verification_tag_not_duplicated(self, shared_page: Page):
        """If the DB row already has a verification tag, UNVERIFIED is not added."""
        page = shared_page
        result = page.evaluate(f"""async () => {{
            const {{ bboxReadToUI }} = await import('/app-static/js/shared/domain/image/mapper.js');
            const UNVERIFIED = '{UNVERIFIED_TAG_UUID}';
            const raw = {{
                uuid: 'b', x: 0, y: 0, width: 0.1, height: 0.1,
                label: null,
                tags: [{{ uuid: UNVERIFIED, icon: null }}],
            }};
            const ui = bboxReadToUI(raw);
            const unverifiedCount = ui.tags.filter(t => t.uuid === UNVERIFIED).length;
            return unverifiedCount;
        }}""")
        assert result == 1  # present but not duplicated

    def test_tag_icon_null_preserved(self, shared_page: Page):
        """Null icons in raw tags are kept as null, not coerced."""
        page = shared_page
        result = page.evaluate(f"""async () => {{
            const {{ bboxReadToUI }} = await import('/app-static/js/shared/domain/image/mapper.js');
            const UNVERIFIED = '{UNVERIFIED_TAG_UUID}';
            const raw = {{
                uuid: 'b', x: 0, y: 0, width: 0.1, height: 0.1,
                label: {{ uuid: 'l', name: 'X', color: '#fff' }},
                tags: [{{ uuid: 'tag-uuid', icon: null }}],
            }};
            const ui = bboxReadToUI(raw);
            const tag = ui.tags.find(t => t.uuid === 'tag-uuid');
            return tag?.icon;
        }}""")
        assert result is None

    def test_coordinates_preserved(self, shared_page: Page):
        page = shared_page
        result = page.evaluate("""async () => {
            const { bboxReadToUI } = await import('/app-static/js/shared/domain/image/mapper.js');
            const raw = {
                uuid: 'b', x: 0.12, y: 0.34, width: 0.56, height: 0.78,
                label: null, tags: [],
            };
            const ui = bboxReadToUI(raw);
            return { x: ui.x, y: ui.y, w: ui.width, h: ui.height };
        }""")
        assert result["x"] == pytest.approx(0.12)
        assert result["y"] == pytest.approx(0.34)
        assert result["w"] == pytest.approx(0.56)
        assert result["h"] == pytest.approx(0.78)
