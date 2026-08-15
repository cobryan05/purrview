"""`navigateTabs` — jumping from anywhere in the app to a specific sub-tab, with
an item pre-selected in it.

Navigation is a DOM click, so the uuid to open with cannot be passed down a call
chain: it goes into a drop box (`_pendingSelections`) that the destination pane
reads on its next render. That makes *when* the click lands part of the
contract. A main tab's lazy init awaits several fetches before its workspace
tabs are wired, so a click that fires too early hits a button that is in the DOM
and does nothing — leaving the entry in the drop box for whichever visit comes
next to consume.

Run with:
    pytest playwright_tests/test_tab_navigation.py -p no:asyncio -v
"""

from playwright.sync_api import Page

# A main-tab pane with two workspace tabs inside it, detached from the real app's
# tabs so activating them cannot disturb the page under test.
#
# Two, not one: `setupWorkspaceTabs` activates the *first* tab on setup, so with
# a single tab a wiring pass alone would open the requested pane by coincidence
# and the test could not tell a landed click from a missed one. The tests
# navigate to the second.
FIXTURE = """
  const host = document.createElement('div');
  host.innerHTML = `
    <button class="main-tab-button" data-tab="tab-navtest"></button>
    <div id="tab-navtest">
      <span class="workspace-tab-button" data-tab="tab-navtest-first">First</span>
      <span class="workspace-tab-button" data-tab="tab-navtest-pane">Pane</span>
      <div id="tab-navtest-first" class="workspace-tab-pane"></div>
      <div id="tab-navtest-pane" class="workspace-tab-pane"></div>
    </div>`;
  document.body.appendChild(host);
  window.__navHost = host;
"""


def _prepare(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")


class TestLateWiringPane:
    def test_the_click_waits_for_the_pane_to_wire_its_tabs(self, page: Page, base_url: str):
        """The pane is mounted but its listeners land 400ms later — as they do
        after a first-visit lazy init that awaits its fetches."""
        _prepare(page, base_url)
        result = page.evaluate("""async () => {
            %s
            const { navigateTabs, setupWorkspaceTabs, takePendingSelection } =
                await import('/app-static/js/app/utils/tabs.js');

            const opened = [];
            navigateTabs('tab-navtest', 'tab-navtest-pane', { selectUuid: 'cfg-42' });

            // Init finishes well after the old fixed 50ms delay would have fired.
            await new Promise(r => setTimeout(r, 400));
            setupWorkspaceTabs('#tab-navtest', (name) => opened.push(name));

            await new Promise(r => setTimeout(r, 300));
            window.__navHost.remove();
            return { opened, selection: takePendingSelection('tab-navtest-pane') };
        }""" % FIXTURE)
        assert (
            result["opened"][-1] == "tab-navtest-pane"
        ), f"navigation left the user on the default pane: {result['opened']}"
        assert result["selection"] == "cfg-42", "the pane opened without the item it was asked to select"

    def test_a_pane_that_never_wires_does_not_leave_a_selection_behind(self, page: Page, base_url: str):
        """Otherwise the entry ambushes the user's next ordinary visit to that
        pane, opening a config they never asked for."""
        _prepare(page, base_url)
        selection = page.evaluate("""async () => {
            %s
            const { navigateTabs, takePendingSelection } =
                await import('/app-static/js/app/utils/tabs.js');

            navigateTabs('tab-navtest', 'tab-navtest-pane', { selectUuid: 'ghost' });
            // Past NAV_WIRE_TIMEOUT_MS with setupWorkspaceTabs never called.
            await new Promise(r => setTimeout(r, 3400));
            window.__navHost.remove();
            return takePendingSelection('tab-navtest-pane');
        }""" % FIXTURE)
        assert selection is None, "a selection for a pane that never opened was left to be consumed later"

    def test_an_already_wired_pane_is_opened_promptly(self, page: Page, base_url: str):
        """The common case must not pay the poll timeout."""
        _prepare(page, base_url)
        result = page.evaluate("""async () => {
            %s
            const { navigateTabs, setupWorkspaceTabs, takePendingSelection } =
                await import('/app-static/js/app/utils/tabs.js');

            const opened = [];
            setupWorkspaceTabs('#tab-navtest', (name) => opened.push(name));
            opened.length = 0;   // drop the initial activation done by setup

            navigateTabs('tab-navtest', 'tab-navtest-pane', { selectUuid: 'cfg-7' });
            await new Promise(r => setTimeout(r, 100));
            window.__navHost.remove();
            return { opened, selection: takePendingSelection('tab-navtest-pane') };
        }""" % FIXTURE)
        assert result["opened"] == ["tab-navtest-pane"]
        assert result["selection"] == "cfg-7"
