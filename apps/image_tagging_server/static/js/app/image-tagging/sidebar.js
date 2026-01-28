import { setTool } from './canvas/tools.js';
import { setupSidebarTabs } from '/app-static/js/app/utils/tabs.js';
import { clearModelsCache, fetchModelsList } from '/app-static/js/shared/api/models.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { renderClassList } from '/app-static/js/ui/classes.js';
import '/app-static/js/ui/inspector.js';
import { Logger } from '/app-static/js/ui/utils/index.js';

let _modelsInit = false;

export async function init() {
  // Setup sidebar tabs (Models / Classes / Annotations / Inspector)
  const container = setupSidebarTabs('#sidebar', async (tabName) => {
    if (tabName === 'tab-classes-tool') {
      renderClassList({
        target: "classes-tool-list",
        editable: false,
        clickable: true,
        onSelectCallback: uuid => setTool(`bbox:${uuid}`)
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
  const defaultTabBtn = container?.querySelector('.sidebar-tab-button');
  if (defaultTabBtn) defaultTabBtn.click();
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
