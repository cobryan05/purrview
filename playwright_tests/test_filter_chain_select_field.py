"""
Tests for ui/utils/fields/FilterChainSelectField.js and the DropDownField
placeholder rule it depends on.

The bug these guard against: the field's options were built purely from the
existing chains, and DropDownField's only "empty" entry is a *disabled*
placeholder rendered solely while the value is already empty. Together that made
the field a one-way door — once a chain was selected there was no option left
that meant "none", so an optional filter chain could never be removed from a
task config again.

Built in the live browser page so /app-static/ imports resolve, following the
pattern in test_dom_components.py. The shared server mocks /api/filter-chains as
an empty list, so these install their own route with one chain in it — a field
holding a real selection is the whole scenario.
"""

import json

import pytest
from playwright.sync_api import Page

_CHAIN_UUID = "11111111-2222-3333-4444-555555555555"
_CHAIN_NAME = "Remove Corrupt Images"


@pytest.fixture(scope="module", autouse=True)
def _one_filter_chain(shared_page: Page):
    shared_page.route(
        lambda url: url.split("?")[0].endswith("/api/filter-chains"),
        lambda route: (
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {"status": "success", "chains": [{"uuid": _CHAIN_UUID, "name": _CHAIN_NAME, "steps": []}]}
                ),
            )
            if route.request.method == "GET"
            else route.continue_()
        ),
    )
    # No unroute: `shared_page` is module-scoped, so the route dies with the page
    # rather than leaking into another test file.


_MAKE_FIELD = """
    const { FilterChainSelectField } = await import(
        '/app-static/js/ui/utils/fields/FilterChainSelectField.js');
    const { clearFilterChainsCache } = await import(
        '/app-static/js/shared/api/filterChains.js');
    clearFilterChainsCache();
    const field = await FilterChainSelectField.create({ value: VALUE, allowNone: ALLOW_NONE });
    const select = (await field.renderEdit()).querySelector('select');
    document.body.appendChild(select);
"""


def _setup(value: str | None, allow_none: bool) -> str:
    return _MAKE_FIELD.replace("VALUE", "null" if value is None else f"'{value}'").replace(
        "ALLOW_NONE", "true" if allow_none else "false"
    )


class TestNoneOptionGating:
    def test_optional_field_offers_a_none_option(self, shared_page: Page):
        """allowNone=true is what an optional chain (required: False) gets."""
        values = shared_page.evaluate(f"""async () => {{
            {_setup(None, True)}
            const opts = [...select.options].filter(o => !o.disabled).map(o => o.value);
            select.remove();
            return opts;
        }}""")
        assert "" in values
        assert _CHAIN_UUID in values

    def test_required_field_offers_no_none_option(self, shared_page: Page):
        """The mirror image: ApplyFilterChainTask has nothing to do without a
        chain, so the option must not be offered there."""
        values = shared_page.evaluate(f"""async () => {{
            {_setup(None, False)}
            const opts = [...select.options].filter(o => !o.disabled).map(o => o.value);
            select.remove();
            return opts;
        }}""")
        assert "" not in values
        assert _CHAIN_UUID in values


class TestClearingASelectedChain:
    def test_a_selected_chain_can_be_cleared_again(self, shared_page: Page):
        """The regression itself. With a chain already selected, picking the
        none option must return the field to an empty value."""
        result = shared_page.evaluate(f"""async () => {{
            {_setup(_CHAIN_UUID, True)}
            const before = field.getValue();
            select.value = '';
            select.dispatchEvent(new Event('change'));
            const after = field.getValue();
            select.remove();
            return {{ before, after }};
        }}""")
        assert result["before"] == _CHAIN_UUID
        assert result["after"] == ""


class TestPlaceholderDoesNotFightTheNoneOption:
    def test_no_disabled_placeholder_when_an_option_covers_the_empty_value(self, shared_page: Page):
        """With a real ''-keyed option present, DropDownField must not also emit
        its disabled placeholder — otherwise two options claim `selected` and
        which one wins is the browser's choice, not ours."""
        disabled = shared_page.evaluate(f"""async () => {{
            {_setup('', True)}
            const n = [...select.options].filter(o => o.disabled).length;
            select.remove();
            return n;
        }}""")
        assert disabled == 0

    def test_placeholder_still_shown_when_nothing_covers_the_empty_value(self, shared_page: Page):
        """The unchanged path — every other dropdown in the app relies on it."""
        disabled = shared_page.evaluate(f"""async () => {{
            {_setup(None, False)}
            const n = [...select.options].filter(o => o.disabled).length;
            select.remove();
            return n;
        }}""")
        assert disabled == 1
