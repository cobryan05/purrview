import { setTool, parseToolUuid } from './canvas/tools.js';
import { setupSidebarTabs } from '/app-static/js/app/utils/tabs.js';
import { clearModelsCache, fetchModelsList } from '/app-static/js/shared/api/models.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { createLabelList } from '/app-static/js/ui/labelList.js';
import { initInspector } from '/app-static/js/ui/inspector.js';
import { Logger } from '/app-static/js/ui/utils/index.js';

let _modelsInit = false;

/**
 * @param {import('@image_tagging_types').ImageTaggingRuntime} runtime
 */
export async function init(runtime) {
  const _unsubInspector = initInspector(runtime);

  let _labelListInit = false;

  // Setup sidebar tabs (Models / Labels / Annotations / Inspector)
  const { container, activateTab } = setupSidebarTabs('#sidebar', async (tabName, button, pane) => {
    if (tabName === 'tab-labels-tool') {
      if (!_labelListInit) {
        _labelListInit = true;
        const listEl = document.getElementById('labels-tool-list');
        if (listEl) await createLabelList(listEl, { mode: 'select', runtime });
      }
    } else if (tabName === 'tab-models') {
      if (!_modelsInit) {
        _modelsInit = true;
        await refreshModelList();
      }
    }
  });

  function handleRecognizeClick() {
    const selected = document.querySelector('input[name="model"]:checked');
    if (!selected) return alert('Please select a model');
    events.publish(EventTypes.RUN_INFERENCE_ON_CANVAS, { modelName: selected.value, runtimeId: runtime.id });
  }

  // Sidebar buttons
  document.getElementById('recognize-button')?.addEventListener('click', handleRecognizeClick);
  document.getElementById('refresh-models')?.addEventListener('click', refreshModelList);

  // activate default workspace tab
  if (container) {
    activateTab(container.querySelector('.sidebar-tab-button')?.dataset.tab);
  }

  // Show Labels tab and auto-select first label when switching to bbox tool
  events.subscribeToCanvas(runtime.id, EventTypes.CANVAS_TOOL_CHANGED, ({ tool }) => {
    const toolInfo = parseToolUuid(tool);
    if (toolInfo.tool === 'bbox') {
      activateTab('tab-labels-tool');
      if (!toolInfo.uuid) {
        const firstRow = document.querySelector('#labels-tool-list .label-panel-row');
        if (firstRow) firstRow.click();
      }
    }
  });
}


async function refreshModelList() {
  try {
    await clearModelsCache();
    const models = await fetchModelsList();
    renderModelList(models);
  } catch (err) {
    Logger.error('Failed to fetch models:', err);
  }
}


function renderModelList(models) {
  const container = document.getElementById('model-list');
  if (!container) return;

  container.innerHTML = '';
  models.forEach(model => {
    const cls = document.createElement('div');
    cls.style.display = 'block';

    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'model';
    radio.value = model;

    cls.appendChild(radio);
    cls.appendChild(document.createTextNode(` ${model}`));
    container.appendChild(cls);
  });

  const firstRadio = container.querySelector('input[type="radio"]');
  if (firstRadio) firstRadio.checked = true;
}
