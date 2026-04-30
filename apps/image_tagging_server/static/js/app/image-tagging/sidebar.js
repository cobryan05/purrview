import { appState } from '../state.js';
import { setTool, parseToolUuid } from './canvas/tools.js';
import { setupSidebarTabs } from '/app-static/js/app/utils/tabs.js';
import { clearModelsCache, fetchModelsList } from '/app-static/js/shared/api/models.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { renderLabelList } from '/app-static/js/ui/labels.js';
import '/app-static/js/ui/inspector.js';
import { Logger } from '/app-static/js/ui/utils/index.js';

let _modelsInit = false;

/**
 * @param {import('@image_tagging_types').ImageTaggingState} state
 */
export async function init(state) {
  // Setup sidebar tabs (Models / Labels / Annotations / Inspector)
  const { container, activateTab } = setupSidebarTabs('#sidebar', async (tabName) => {
    if (tabName === 'tab-labels-tool') {
      renderLabelList({
        target: "labels-tool-list",
        editable: false,
        clickable: true,
        onSelectCallback: uuid => setTool(state, `bbox:${uuid}`)
      });
    } else if (tabName === 'tab-models') {
      if (!_modelsInit) {
        _modelsInit = true;
        await refreshModelList();
      }
    }
  });

  // Sidebar buttons
  document.getElementById('recognize-button')?.addEventListener('click', handleRecognizeClick);
  document.getElementById('refresh-models')?.addEventListener('click', refreshModelList);

  // activate default workspace tab
  if (container) {
    activateTab(container.querySelector('.sidebar-tab-button')?.dataset.tab);
  }

  // Show 'Labels' tab when tool changes to BBox
  events.subscribe(EventTypes.CANVAS_TOOL_CHANGED, ({ tool }) => {
    // Switch to the Labels tab if the tool is BBox
    const toolInfo = parseToolUuid(tool);
    if (toolInfo.tool === 'bbox') {
      activateTab('tab-labels-tool');
      const uuid = toolInfo.uuid ?? appState.imageTagging.currentLabelUuid;

      // If there is no currently selected label then select the first one
      if (!uuid) {
        // defer until after rendering
        requestAnimationFrame(() => {
          const firstLabelRow = document.querySelector('#labels-tool-list .label-row');
          if (firstLabelRow) firstLabelRow.click();
        });
      }
    }
  });
}


function handleRecognizeClick() {
  const selected = document.querySelector('input[name="model"]:checked');
  if (!selected) return alert('Please select a model');

  events.publish(EventTypes.RUN_INFERENCE_ON_CANVAS, { modelName: selected.value });
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
