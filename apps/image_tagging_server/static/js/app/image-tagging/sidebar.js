import { parseToolUuid } from './canvas/tools.js';
import { createInferencePanel } from './inferencePanel.js';
import { setupSidebarTabs } from '/app-static/js/app/utils/tabs.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { createLabelList } from '/app-static/js/ui/labelList.js';
import { createTagPanel } from '/app-static/js/ui/inspector.js';

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
let _tagPanel = null;

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

  // ── Eager init: label list ────────────────────────────────────────────────
  // Must be alive before the first CANVAS_TOOL_CHANGED event so the label
  // panel's own "auto-select first label on bare bbox tool" logic fires.
  const listEl = document.getElementById('labels-tool-list');
  if (listEl) await createLabelList(listEl, { mode: 'select', runtime });

  // ── Eager init: bbox info / tag panel ─────────────────────────────────────
  // Must be alive before the first CANVAS_BBOX_CLICKED or CANVAS_TOOL_CHANGED
  // event so single-click updates and tag-tool highlights work immediately.
  const paneEl = document.getElementById('tab-inspector');
  if (paneEl) {
    _tagPanel?.destroy();  // clean up any prior instance if init() is called again
    _tagPanel = createTagPanel(paneEl, runtime);
  }

  // ── Default tab ───────────────────────────────────────────────────────────
  if (container) {
    activateTab(container.querySelector('.sidebar-tab-button')?.dataset.tab);
  }

  // ── Tab switching rules ───────────────────────────────────────────────────

  // Bbox tool → Labels tab; tag tool → Bbox Info tab.
  events.subscribeToCanvas(runtime.id, EventTypes.CANVAS_TOOL_CHANGED, ({ tool }) => {
    const { tool: kind } = parseToolUuid(tool);
    if (kind === 'bbox') {
      activateTab('tab-labels-tool');
    } else if (kind === 'tag') {
      activateTab('tab-inspector');
    }
  });

  // Double-clicking a bbox in select mode → switch to Bbox Info tab.
  events.subscribeToCanvas(runtime.id, EventTypes.CANVAS_BBOX_DOUBLE_CLICKED, () => {
    activateTab('tab-inspector');
  });
}
