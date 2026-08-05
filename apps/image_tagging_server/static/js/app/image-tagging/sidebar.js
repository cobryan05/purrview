import { createInferencePanel } from './inferencePanel.js';
import { createSidebarPanels, wireSidebarTabRules } from './canvasWiring.js';
import { setupSidebarTabs } from '/app-static/js/app/utils/tabs.js';

// Module-level so a repeat init() can tear down the previous mounting's panels
// and subscriptions rather than stacking a second set on top.
let _panels = null;
let _unsubTabRules = null;

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export async function init(runtime) {
  let _inferenceInit = false;

  // Setup sidebar tabs — inference panel remains lazy (heavy); the others are
  // initialized eagerly below so they can subscribe to events immediately.
  const { container, activateTab } = setupSidebarTabs('#sidebar', async (tabName) => {
    if (tabName === 'tab-inference') {
      if (!_inferenceInit) {
        _inferenceInit = true;
        const panelContainer = document.getElementById('inference-panel-container');
        if (panelContainer) await createInferencePanel(runtime, panelContainer);
      }
    }
  });

  // ── Eager init: label list + tags/bbox info panel ─────────────────────────
  // Both must be alive before the first tool/bbox event — see createSidebarPanels.
  // Clean up any prior instances if init() is called again. (This now tears down
  // the label list too, where it previously leaked its three event subscriptions
  // on every re-init; only the tag panel was being destroyed.)
  _panels?.destroy();
  _panels = await createSidebarPanels(runtime, {
    labelsListEl: document.getElementById('labels-tool-list'),
    tagsPaneEl: document.getElementById('tab-inspector'),
  });

  // ── Default tab ───────────────────────────────────────────────────────────
  if (container) {
    activateTab(container.querySelector('.sidebar-tab-button')?.dataset.tab);
  }

  // ── Tab switching rules ───────────────────────────────────────────────────
  _unsubTabRules?.();
  _unsubTabRules = wireSidebarTabRules(runtime, {
    activateTab,
    labelsTabId: 'tab-labels-tool',
    tagsTabId: 'tab-inspector',
  });
}
