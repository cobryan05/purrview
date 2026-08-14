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

/**
 * Section tabs — the tier between main tabs and workspace tabs.
 *
 * Only Annotate needs it, and only because it nests: its Canvas pane contains a
 * *second* workspace-tab level (Canvas / Files beside the sidebar). `setupTabs`
 * matches every descendant, so reusing `.workspace-tab-*` for both would make
 * the outer row toggle the inner panes as well.
 */
export function setupSectionTabs(containerOrSelector, onTabChange) {
  return setupTabs(containerOrSelector, '.section-tab-button', '.section-tab-pane', onTabChange);
}

/** Sidebar tabs (Models / Classes / Inspector) */
export function setupSidebarTabs(containerOrSelector, onTabChange) {
  return setupTabs(containerOrSelector, '.sidebar-tab-button', '.sidebar-tab-pane', onTabChange);
}

/**
 * Item a workspace tab should open with selected, keyed by that tab's id.
 *
 * Navigation is a DOM click, so there is no call chain to hand an argument
 * down: the pane's render runs later, from the tab's own onTabChange. This is
 * the drop box between the two. Entries are consumed once — a later plain
 * visit to the same tab must not resurrect a stale selection.
 */
const _pendingSelections = new Map();

/**
 * Read and clear the item a caller asked this tab to open with.
 *
 * @param {string} tabId  the workspace tab's data-tab value
 * @returns {?string} uuid, or null if the tab was opened without one
 */
export function takePendingSelection(tabId) {
  const uuid = _pendingSelections.get(tabId) ?? null;
  _pendingSelections.delete(tabId);
  return uuid;
}

/**
 * Navigate to a main tab then, after a short delay, to a workspace sub-tab.
 * The delay lets the main-tab's lazy-init mount its pane before the workspace
 * click fires.  Pass the bare tab id (data-tab value) for each argument.
 *
 * @param {string} mainTabId       e.g. 'tab-configuration'
 * @param {string} workspaceTabId  e.g. 'tab-dataset-management'
 * @param {Object} [opts]
 * @param {?string} [opts.selectUuid] - item the destination pane should open
 *   with selected, for panes that support it (the list-editor screens).
 */
export function navigateTabs(mainTabId, workspaceTabId, { selectUuid = null } = {}) {
  if (selectUuid) _pendingSelections.set(workspaceTabId, selectUuid);
  document.querySelector(`.main-tab-button[data-tab="${mainTabId}"]`)?.click();
  setTimeout(() => {
    const button = document.querySelector(`#${mainTabId} [data-tab="${workspaceTabId}"]`);
    // Clicking the already-active tab is a no-op, so the pane would never
    // re-render and the pending selection would sit unconsumed until some
    // later visit picked it up. Drop `active` first to force the re-render.
    if (selectUuid) button?.classList.remove('active');
    button?.click();
  }, 50);
}
