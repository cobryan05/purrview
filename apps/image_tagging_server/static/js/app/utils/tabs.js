/**
 * Tab setup helper
 * @param {HTMLElement|string} containerOrSelector - container or selector
 * @param {string} buttonSelector - CSS selector for tab buttons
 * @param {string} paneSelector - CSS selector for tab panes
 * @param {(tabName: string, button?: HTMLElement) => void} [onTabChange] - callback
 * @returns {HTMLElement|null} container element
 */
function setupTabs(containerOrSelector, buttonSelector, paneSelector, onTabChange) {
  const container = typeof containerOrSelector === 'string'
    ? document.querySelector(containerOrSelector)
    : containerOrSelector;

  if (!container) return null;

  const buttons = container.querySelectorAll(buttonSelector);
  const panes = container.querySelectorAll(paneSelector);

  buttons.forEach(button => {
    button.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      panes.forEach(p => p.classList.remove('active'));

      button.classList.add('active');
      const tabName = button.dataset.tab;
      const pane = container.querySelector(`#${tabName}`);
      if (pane) pane.classList.add('active');

      if (onTabChange) onTabChange(tabName, button);
    });
  });

  return container;
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
