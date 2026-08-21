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
    // Marked so navigateTabs can tell a wired button from one that is merely in
    // the DOM — clicking the latter does nothing at all, silently.
    button.dataset.tabWired = '1';
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
 * Pass as `selectUuid` to mean "open this pane's editor on a *new* item".
 *
 * Rides the same drop box as a uuid rather than adding a second channel: the
 * destination pane already reads one value on arrival, and the wait-for-wired
 * machinery in navigateTabs (including dropping the request if the pane never
 * comes up) is exactly what a "start a new one" request needs too. Not a
 * plausible uuid, so a pane that hasn't been taught about it looks the value up,
 * finds nothing, and opens normally.
 */
export const NEW_ITEM_SELECTION = '__new__';

/**
 * Read and clear the item a caller asked this tab to open with.
 *
 * @param {string} tabId  the workspace tab's data-tab value
 * @returns {?string} uuid, {@link NEW_ITEM_SELECTION}, or null if the tab was
 *   opened without one
 */
export function takePendingSelection(tabId) {
  const uuid = _pendingSelections.get(tabId) ?? null;
  _pendingSelections.delete(tabId);
  return uuid;
}

/** How long navigateTabs waits for a lazy-initialised pane to wire its tabs. */
const NAV_WIRE_TIMEOUT_MS = 3000;
const NAV_POLL_MS = 25;

/**
 * Navigate to a main tab then, once its pane is ready, to a workspace sub-tab.
 *
 * Waits for the destination button to be *wired* rather than firing after a
 * fixed delay. A main tab's lazy init awaits several fetches before
 * `setupWorkspaceTabs` attaches its listeners, so a timed click could land on a
 * button that is in the DOM but inert: the user stayed on the default pane, and
 * the pending selection stayed in the map to be consumed — and acted on — by
 * their next ordinary visit to that pane, opening something they never asked
 * for. If the pane never wires up, the selection is dropped for the same
 * reason.
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

  const deadline = Date.now() + NAV_WIRE_TIMEOUT_MS;
  const attempt = () => {
    const button = document.querySelector(`#${mainTabId} [data-tab="${workspaceTabId}"][data-tab-wired]`);
    if (!button) {
      if (Date.now() < deadline) {
        setTimeout(attempt, NAV_POLL_MS);
      } else if (selectUuid && _pendingSelections.get(workspaceTabId) === selectUuid) {
        // Nobody is going to consume it now, and leaving it would ambush a
        // later visit. Only drop our own entry — a newer navigation may have
        // overwritten it in the meantime.
        _pendingSelections.delete(workspaceTabId);
      }
      return;
    }
    // Clicking the already-active tab is a no-op, so the pane would never
    // re-render and the pending selection would sit unconsumed until some
    // later visit picked it up. Drop `active` first to force the re-render.
    if (selectUuid) button.classList.remove('active');
    button.click();
  };
  setTimeout(attempt, 0);
}
