import { fetchModelsList, clearModelsCache } from '/app-static/js/shared/api/models.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { Logger } from '/app-static/js/ui/utils/index.js';
import { clearUnmatched } from './canvas/image.js';

/**
 * Creates a self-contained inference panel: model list + Run/Refresh buttons.
 * Follows the same pattern as initLabelPanel / initToolbar.
 *
 * @param {import('../image-tagging/runtime.js').ImageTaggingRuntime} runtime
 * @param {HTMLElement} container
 * @returns {Promise<{ destroy: () => void }>}
 */
export async function createInferencePanel(runtime, container) {
  let _models = [];
  let _selected = null;

  container.innerHTML = '';
  container.classList.add('inference-panel');

  // ── Action buttons ────────────────────────────────────────────────
  const actionsEl = document.createElement('div');
  actionsEl.className = 'inference-panel-actions';

  const runBtn = document.createElement('button');
  runBtn.textContent = 'Run Inference';
  runBtn.addEventListener('click', handleRun);

  const refreshBtn = document.createElement('button');
  refreshBtn.textContent = 'Refresh';
  refreshBtn.addEventListener('click', handleRefresh);

  // Removes inference detections whose class never matched a real label (grey boxes).
  const clearGreyBtn = document.createElement('button');
  clearGreyBtn.textContent = 'Clear Grey';
  clearGreyBtn.title = 'Remove unmatched (grey) detections from the canvas';
  clearGreyBtn.addEventListener('click', () => clearUnmatched(runtime));

  actionsEl.append(runBtn, refreshBtn, clearGreyBtn);

  // ── Model list ────────────────────────────────────────────────────
  const listEl = document.createElement('div');
  listEl.className = 'inference-model-list';

  container.append(actionsEl, listEl);

  // ── Helpers ───────────────────────────────────────────────────────
  function renderList() {
    listEl.innerHTML = '';
    if (_models.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'inference-model-empty';
      empty.textContent = 'No models found';
      listEl.appendChild(empty);
      return;
    }
    _models.forEach(model => {
      const item = document.createElement('div');
      item.className = 'inference-model-item' + (_selected === model ? ' selected' : '');
      item.title = model;
      item.textContent = model;
      item.addEventListener('click', () => {
        _selected = model;
        renderList();
      });
      listEl.appendChild(item);
    });
  }

  async function handleRefresh() {
    refreshBtn.disabled = true;
    try {
      await clearModelsCache();
      _models = await fetchModelsList();
      if (_models.length > 0 && !_models.includes(_selected)) {
        _selected = _models[0];
      }
      renderList();
    } catch (err) {
      Logger.error('Failed to refresh models:', err);
    } finally {
      refreshBtn.disabled = false;
    }
  }

  function handleRun() {
    if (!_selected) {
      alert('Please select a model first');
      return;
    }
    events.publish(EventTypes.RUN_INFERENCE_ON_CANVAS, { modelName: _selected, runtimeId: runtime.id });
  }

  // ── Initial load ──────────────────────────────────────────────────
  try {
    _models = await fetchModelsList();
    if (_models.length > 0) _selected = _models[0];
    renderList();
  } catch (err) {
    Logger.error('Failed to load models:', err);
    const errEl = document.createElement('div');
    errEl.className = 'inference-model-empty';
    errEl.textContent = 'Failed to load models';
    listEl.appendChild(errEl);
  }

  return {
    destroy() {
      container.innerHTML = '';
      container.classList.remove('inference-panel');
    },
  };
}
