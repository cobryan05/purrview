"""
Tests for the "Show dynamic labels" toggle in shared/activeModel.js and the rule
every label picker depends on: the label a control is currently displaying stays
in its options even while dynamic labels are hidden.

Without that rule a hidden assignment cannot be represented. A <select> with no
matching option shows its first entry instead, and DropDownField.getValue()
returns null for a value it cannot match — so the task-config label field read
back null and dropped the assignment on save.

Built in the live browser page so /app-static/ imports resolve, following the
pattern in test_dom_components.py. /api/labels is mocked with one standard and
one dynamic label; a picker holding the dynamic one is the whole scenario.
"""

import json

import pytest
from playwright.sync_api import Page

_STANDARD_UUID = "aaaaaaaa-0000-0000-0000-000000000001"
_DYNAMIC_UUID = "bbbbbbbb-0000-0000-0000-000000000002"
_DYNAMIC_NAME = "laundrymachine"

_LABELS = [
    {"uuid": _STANDARD_UUID, "name": "Cat", "color": "#ff0000", "labelType": "standard", "scope": "object"},
    {"uuid": _DYNAMIC_UUID, "name": _DYNAMIC_NAME, "color": "#888888", "labelType": "dynamic", "scope": "object"},
]


@pytest.fixture(scope="module", autouse=True)
def _one_dynamic_label(shared_page: Page):
    shared_page.route(
        lambda url: url.split("?")[0].endswith("/api/labels"),
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "labels": _LABELS}),
            )
            if route.request.method == "GET"
            else route.continue_()
        ),
    )
    # No unroute: `shared_page` is module-scoped, so the route dies with the page.


@pytest.fixture(autouse=True)
def _restore_visibility(shared_page: Page):
    """Each test sets the toggle itself; put it back so the module-scoped page
    doesn't carry one test's choice into the next."""
    yield
    shared_page.evaluate("""async () => {
            const m = await import('/app-static/js/shared/activeModel.js');
            m.setDynamicLabelsVisible(true);
        }""")


def _visible_names(page: Page, *, show: bool, keep_uuid: str | None) -> list[str]:
    keep = "null" if keep_uuid is None else f"'{keep_uuid}'"
    return page.evaluate(f"""async () => {{
            const m = await import('/app-static/js/shared/activeModel.js');
            const {{ fetchLabels, clearLabelsCache }} = await import('/app-static/js/shared/api/labels.js');
            clearLabelsCache();
            m.setDynamicLabelsVisible({str(show).lower()});
            const labels = await fetchLabels();
            return m.filterVisibleLabels(labels.values(), {keep}).map(l => l.name);
        }}""")


class TestDynamicLabelDefault:
    def test_dynamic_labels_are_visible_with_no_stored_preference(self, shared_page: Page):
        """Assigning to a dynamic label must not require finding a toggle first,
        so a browser that has never touched the preference shows them."""
        visible = shared_page.evaluate("""async () => {
                const m = await import('/app-static/js/shared/activeModel.js');
                const { fetchLabels, clearLabelsCache } = await import('/app-static/js/shared/api/labels.js');
                sessionStorage.removeItem('wtch_show_dynamic_labels');
                m.init();
                clearLabelsCache();
                const labels = await fetchLabels();
                return m.filterVisibleLabels(labels.values()).map(l => l.name);
            }""")
        assert _DYNAMIC_NAME in visible

    def test_a_stored_opt_out_still_hides_them(self, shared_page: Page):
        """The toggle has to survive a reload, or turning it off is pointless."""
        visible = shared_page.evaluate("""async () => {
                const m = await import('/app-static/js/shared/activeModel.js');
                const { fetchLabels, clearLabelsCache } = await import('/app-static/js/shared/api/labels.js');
                m.setDynamicLabelsVisible(false);
                m.init();
                clearLabelsCache();
                const labels = await fetchLabels();
                return m.filterVisibleLabels(labels.values()).map(l => l.name);
            }""")
        assert _DYNAMIC_NAME not in visible


class TestCurrentValueSurvivesHiding:
    def test_hiding_removes_a_dynamic_label_nothing_is_holding(self, shared_page: Page):
        assert _DYNAMIC_NAME not in _visible_names(shared_page, show=False, keep_uuid=None)

    def test_hiding_keeps_the_dynamic_label_the_picker_is_displaying(self, shared_page: Page):
        """The mirror of the test above: same preference, different answer,
        because this picker's current value is that label."""
        assert _DYNAMIC_NAME in _visible_names(shared_page, show=False, keep_uuid=_DYNAMIC_UUID)

    def test_keeping_one_label_does_not_reveal_the_rest(self, shared_page: Page):
        """keepUuid is an exemption for one label, not a second way to turn the
        preference off — a second dynamic label stays hidden."""
        names = shared_page.evaluate(f"""async () => {{
                const m = await import('/app-static/js/shared/activeModel.js');
                m.setDynamicLabelsVisible(false);
                const labels = [
                    {{ uuid: 'kept', name: 'kept', labelType: 'dynamic' }},
                    {{ uuid: 'other', name: 'other', labelType: 'dynamic' }},
                ];
                return m.filterVisibleLabels(labels, 'kept').map(l => l.name);
            }}""")
        assert names == ["kept"]


class TestTaskConfigLabelFieldKeepsItsValue:
    def _get_value(self, page: Page, *, show: bool) -> str | None:
        return page.evaluate(f"""async () => {{
                const m = await import('/app-static/js/shared/activeModel.js');
                const {{ fieldFactories }} = await import('/app-static/js/ui/utils/fields/Factories.js');
                const {{ clearLabelsCache }} = await import('/app-static/js/shared/api/labels.js');
                clearLabelsCache();
                m.setDynamicLabelsVisible({str(show).lower()});
                const field = await fieldFactories.label('lbl', {{}}, {{ lbl: '{_DYNAMIC_UUID}' }}, () => {{}});
                return field.getValue();
            }}""")

    def test_a_dynamic_assignment_reads_back_while_dynamic_labels_are_hidden(self, shared_page: Page):
        """This is the save path: SchemaField persists whatever getValue()
        returns, so a null here silently clears the task config's label."""
        assert self._get_value(shared_page, show=False) == _DYNAMIC_UUID

    def test_a_dynamic_assignment_reads_back_while_they_are_shown(self, shared_page: Page):
        assert self._get_value(shared_page, show=True) == _DYNAMIC_UUID
