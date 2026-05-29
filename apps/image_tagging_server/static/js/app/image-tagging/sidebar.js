import { parseToolUuid } from './canvas/tools.js';
import { createInferencePanel } from './inferencePanel.js';
import { setupSidebarTabs } from '/app-static/js/app/utils/tabs.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { createLabelList } from '/app-static/js/ui/labelList.js';
import { initInspector } from '/app-static/js/ui/inspector.js';

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export async function init(runtime) {
  const _unsubInspector = initInspector(runtime);

  let _labelListInit = false;
  let _inferenceInit = false;

  // Setup sidebar tabs (Inference / Labels / Inspector)
  const { container, activateTab } = setupSidebarTabs('#sidebar', async (tabName, button, pane) => {
    if (tabName === 'tab-labels-tool') {
      if (!_labelListInit) {
        _labelListInit = true;
        const listEl = document.getElementById('labels-tool-list');
        if (listEl) await createLabelList(listEl, { mode: 'select', runtime });
      }
    } else if (tabName === 'tab-inference') {
      if (!_inferenceInit) {
        _inferenceInit = true;
        const panelContainer = document.getElementById('inference-panel-container');
        if (panelContainer) await createInferencePanel(runtime, panelContainer);
      }
    }
  });

  // activate default workspace tab
  if (container) {
    activateTab(container.querySelector('.sidebar-tab-button')?.dataset.tab);
  }

  // Show Labels tab when switching to bbox tool; label selection is handled by the label panel
  events.subscribeToCanvas(runtime.id, EventTypes.CANVAS_TOOL_CHANGED, ({ tool }) => {
    if (parseToolUuid(tool).tool === 'bbox') {
      activateTab('tab-labels-tool');
    }
  });
}

