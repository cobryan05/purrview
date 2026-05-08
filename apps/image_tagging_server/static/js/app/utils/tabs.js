/**
 * Tab setup helper
 * @param {HTMLElement|string} containerOrSelector - container or selector
 * @param {string} buttonSelector - CSS selector for tab buttons
 * @param {string} paneSelector - CSS selector for tab panes
 * @param {(tabName: string, button?: HTMLElement, pane?: HTMLElement) => void} [onTabChange] - callback
 * @returns {{container: HTMLElement|null, activateTab: (tabName: string) => void}}
 */
function setupTabs(containerOrSelector, buttonSelector, paneSelector, onTabChange) {
  const container = typeof containerOrSelector === 'string'
    ? document.querySelector(containerOrSelector)
    : containerOrSelector;

  if (!container) return { container: null, activateTab: () => { } };

  // Lazy query functions
  const getButtons = () => Array.from(container.querySelectorAll(buttonSelector));
  const getPanes = () => Array.from(container.querySelectorAll(paneSelector));

  function getActiveTab() {
    const activeButton = getButtons().find(b => b.classList.contains('active'));
    return activeButton?.dataset.tab ?? null;
  }

  function activateTab(tabName) {
    if (getActiveTab() === tabName) return;

    const button = getButtons().find(b => b.dataset.tab === tabName);
    if (!button) return;

    // deactivate all
    getButtons().forEach(b => b.classList.remove('active'));
    getPanes().forEach(p => p.classList.remove('active'));

    // activate chosen
    button.classList.add('active');
    const pane = container.querySelector(`#${tabName}`);
    if (pane) pane.classList.add('active');

    onTabChange?.(tabName, button, pane);
  }

  // attach click listeners
  getButtons().forEach(button => {
    button.addEventListener('click', () => activateTab(button.dataset.tab));
  });

  // ensure at least one tab is active on setup
  const initialTab = getActiveTab() ?? getButtons()[0]?.dataset.tab;
  if (initialTab) activateTab(initialTab);

  return { container, activateTab };
}

/** Top-level main tabs (Image Tagging / Configuration) */
export function setupMainTabs(containerOrSelector, onTabChange) {
  return setupTabs(containerOrSelector, '.main-tab-button', '.main-tab-pane', onTabChange);
}

/** Workspace tabs (Canvas / Files, Configuration subtabs) */
export function setupWorkspaceTabs(containerOrSelector, onTabChange) {
  return setupTabs(containerOrSelector, '.workspace-tab-button', '.workspace-tab-pane', onTabChange);
}

/** Sidebar tabs (Models / Classes / Inspector) */
export function setupSidebarTabs(containerOrSelector, onTabChange) {
  return setupTabs(containerOrSelector, '.sidebar-tab-button', '.sidebar-tab-pane', onTabChange);
}
