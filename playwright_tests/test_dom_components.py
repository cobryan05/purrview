"""
Tests for:
  - app/utils/tabs.js  (setupMainTabs / setupWorkspaceTabs)
  - ui/utils/modalShell.js  (createModalShell)

These tests build minimal DOM structures inside the live browser page so that
/app-static/ imports resolve, then exercise the component behaviour via JS.
"""
import pytest
from playwright.sync_api import Page


def goto(page: Page, base_url: str):
    page.goto(base_url)


# ---------------------------------------------------------------------------
# Tabs — setupMainTabs (uses .main-tab-button / .main-tab-pane selectors)
# ---------------------------------------------------------------------------

# Reusable JS to create a container with two tabs
_TAB_SETUP = """
    const { setupMainTabs } = await import('/app-static/js/app/utils/tabs.js');

    const container = document.createElement('div');
    container.innerHTML = `
        <span class="main-tab-button" data-tab="tab-a">A</span>
        <span class="main-tab-button" data-tab="tab-b">B</span>
        <div id="tab-a" class="main-tab-pane">Pane A</div>
        <div id="tab-b" class="main-tab-pane">Pane B</div>
    `;
    document.body.appendChild(container);
    const callbacks = [];
    const { activateTab } = setupMainTabs(container, (tabName) => callbacks.push(tabName));
"""

_TAB_TEARDOWN = "container.remove();"


class TestTabsInitialState:
    def test_first_button_active_on_setup(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_TAB_SETUP}
            const btnA = container.querySelector('[data-tab="tab-a"]');
            const active = btnA.classList.contains('active');
            {_TAB_TEARDOWN}
            return active;
        }}""")
        assert result is True

    def test_first_pane_active_on_setup(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_TAB_SETUP}
            const paneA = container.querySelector('#tab-a');
            const active = paneA.classList.contains('active');
            {_TAB_TEARDOWN}
            return active;
        }}""")
        assert result is True

    def test_second_pane_not_active_initially(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_TAB_SETUP}
            const paneB = container.querySelector('#tab-b');
            const active = paneB.classList.contains('active');
            {_TAB_TEARDOWN}
            return active;
        }}""")
        assert result is False

    def test_onTabChange_fires_for_initial_activation(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_TAB_SETUP}
            {_TAB_TEARDOWN}
            return callbacks;
        }}""")
        assert "tab-a" in result  # first tab activated on setup


class TestTabsSwitching:
    def test_click_second_button_activates_second_pane(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_TAB_SETUP}
            container.querySelector('[data-tab="tab-b"]').click();
            const result = {{
                paneA: container.querySelector('#tab-a').classList.contains('active'),
                paneB: container.querySelector('#tab-b').classList.contains('active'),
                btnA:  container.querySelector('[data-tab="tab-a"]').classList.contains('active'),
                btnB:  container.querySelector('[data-tab="tab-b"]').classList.contains('active'),
            }};
            {_TAB_TEARDOWN}
            return result;
        }}""")
        assert result["paneA"] is False
        assert result["paneB"] is True
        assert result["btnA"] is False
        assert result["btnB"] is True

    def test_exactly_one_pane_active_after_switch(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_TAB_SETUP}
            container.querySelector('[data-tab="tab-b"]').click();
            const activePanes = container.querySelectorAll('.main-tab-pane.active').length;
            const activeBtns  = container.querySelectorAll('.main-tab-button.active').length;
            {_TAB_TEARDOWN}
            return {{ activePanes, activeBtns }};
        }}""")
        assert result["activePanes"] == 1
        assert result["activeBtns"] == 1

    def test_onTabChange_fires_with_correct_name(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_TAB_SETUP}
            callbacks.length = 0;   // clear setup callbacks
            container.querySelector('[data-tab="tab-b"]').click();
            {_TAB_TEARDOWN}
            return callbacks;
        }}""")
        assert result == ["tab-b"]

    def test_clicking_active_tab_does_not_fire_callback(self, page: Page, base_url: str):
        """Re-clicking the already-active tab is a no-op (activateTab guards on same-tab)."""
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_TAB_SETUP}
            callbacks.length = 0;
            container.querySelector('[data-tab="tab-a"]').click();   // already active
            {_TAB_TEARDOWN}
            return callbacks.length;
        }}""")
        assert result == 0

    def test_activateTab_api_works(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_TAB_SETUP}
            activateTab('tab-b');
            const active = container.querySelector('#tab-b').classList.contains('active');
            {_TAB_TEARDOWN}
            return active;
        }}""")
        assert result is True

    def test_activateTab_nonexistent_does_nothing(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate(f"""async () => {{
            {_TAB_SETUP}
            activateTab('tab-xyz');   // no such tab — should not crash
            const activePanes = container.querySelectorAll('.main-tab-pane.active').length;
            {_TAB_TEARDOWN}
            return activePanes;
        }}""")
        assert result == 1   # tab-a still active, nothing changed


class TestWorkspaceTabs:
    """setupWorkspaceTabs uses .workspace-tab-button / .workspace-tab-pane selectors."""

    def test_workspace_tabs_work_independently(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { setupWorkspaceTabs } = await import('/app-static/js/app/utils/tabs.js');

            const container = document.createElement('div');
            container.innerHTML = `
                <span class="workspace-tab-button" data-tab="ws-x">X</span>
                <span class="workspace-tab-button" data-tab="ws-y">Y</span>
                <div id="ws-x" class="workspace-tab-pane">X pane</div>
                <div id="ws-y" class="workspace-tab-pane">Y pane</div>
            `;
            document.body.appendChild(container);
            setupWorkspaceTabs(container);

            container.querySelector('[data-tab="ws-y"]').click();
            const result = {
                xActive: container.querySelector('#ws-x').classList.contains('active'),
                yActive: container.querySelector('#ws-y').classList.contains('active'),
            };
            container.remove();
            return result;
        }""")
        assert result["xActive"] is False
        assert result["yActive"] is True


# ---------------------------------------------------------------------------
# Modal shell — createModalShell
# ---------------------------------------------------------------------------

class TestModalShell:
    def test_open_appends_overlay_to_body(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const { overlay, open, close } = createModalShell();
            open();
            const present = document.body.contains(overlay);
            close();
            return present;
        }""")
        assert result is True

    def test_close_removes_overlay_from_body(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const { overlay, open, close } = createModalShell();
            open();
            close();
            return document.body.contains(overlay);
        }""")
        assert result is False

    def test_close_is_idempotent(self, page: Page, base_url: str):
        """Calling close() twice should not throw."""
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const { overlay, open, close } = createModalShell();
            open();
            close();
            close();   // second close should be a no-op
            return 'ok';
        }""")
        assert result == "ok"

    def test_onClose_called_exactly_once(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            let callCount = 0;
            const { open, close } = createModalShell({ onClose: () => callCount++ });
            open();
            close();
            close();   // second close must not fire onClose again
            return callCount;
        }""")
        assert result == 1

    def test_backdrop_click_closes_modal(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            let closed = false;
            const { overlay, open } = createModalShell({ onClose: () => { closed = true; } });
            open();
            // Simulate a click directly on the overlay (not the dialog)
            overlay.dispatchEvent(new MouseEvent('click', { bubbles: true }));
            return { closed, stillInDom: document.body.contains(overlay) };
        }""")
        assert result["closed"] is True
        assert result["stillInDom"] is False

    def test_dialog_click_does_not_close(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            let closed = false;
            const { overlay, dialog, open } = createModalShell({ onClose: () => { closed = true; } });
            open();
            // Click inside the dialog — target is the dialog, not the overlay
            dialog.dispatchEvent(new MouseEvent('click', { bubbles: true }));
            const stillOpen = document.body.contains(overlay);
            if (stillOpen) overlay.remove();   // cleanup
            return { closed, stillOpen };
        }""")
        assert result["closed"] is False
        assert result["stillOpen"] is True

    def test_overlay_and_dialog_are_elements(self, page: Page, base_url: str):
        goto(page, base_url)
        result = page.evaluate("""async () => {
            const { createModalShell } = await import('/app-static/js/ui/utils/modalShell.js');
            const { overlay, dialog } = createModalShell();
            return {
                overlayClass: overlay.className,
                dialogClass:  dialog.className,
                dialogIsChildOfOverlay: overlay.contains(dialog),
            };
        }""")
        assert "app-modal-overlay" in result["overlayClass"]
        assert "app-modal-dialog" in result["dialogClass"]
        assert result["dialogIsChildOfOverlay"] is True
