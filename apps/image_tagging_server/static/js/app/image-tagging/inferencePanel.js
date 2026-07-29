import { fetchModelsList, fetchModelsKinds, clearModelsCache } from '/app-static/js/shared/api/models.js';
import { runClassify } from '/app-static/js/shared/api/inference.js';
import { setImageLabel } from '/app-static/js/shared/api/imageLabels.js';
import { fetchLabelByUuid } from '/app-static/js/shared/api/labels.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { Logger } from '/app-static/js/ui/utils/index.js';
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { clearUnmatched } from './canvas/image.js';

/**
 * Creates a self-contained inference panel: detector + classifier model lists and Run button.
 *
 * @param {import('../image-tagging/runtime.js').ImageTaggingRuntime} runtime
 * @param {HTMLElement} container
 * @returns {Promise<{ destroy: () => void }>}
 */
export async function createInferencePanel(runtime, container) {
  /** @type {string[]} */ let _detectors = [];
  /** @type {string[]} */ let _classifiers = [];
  /** @type {string|null} */ let _selected = null;
  /** @type {'detector'|'classifier'|null} */ let _selectedKind = null;
  /** @type {number} */ let _confThresh = 0.25;
  /** @type {boolean} */ let _replaceExisting = false;

  container.innerHTML = '';
  container.classList.add('inference-panel');

  // ── Action buttons ────────────────────────────────────────────────
  const actionsEl = document.createElement('div');
  actionsEl.className = 'inference-panel-actions';

  const runBtn = document.createElement('button');
  runBtn.textContent = 'Run';
  runBtn.addEventListener('click', handleRun);

  const refreshBtn = document.createElement('button');
  refreshBtn.textContent = 'Refresh';
  refreshBtn.addEventListener('click', handleRefresh);

  const clearGreyBtn = document.createElement('button');
  clearGreyBtn.textContent = 'Clear Grey';
  clearGreyBtn.title = 'Remove unmatched (grey) detections from the canvas';
  clearGreyBtn.addEventListener('click', () => clearUnmatched(runtime));

  // ── Min confidence slider ─────────────────────────────────────────
  const confRow = document.createElement('div');
  confRow.className = 'inference-conf-row';

  const confLabel = document.createElement('label');
  confLabel.className = 'inference-conf-label';
  confLabel.textContent = 'Min confidence:';

  const confSlider = document.createElement('input');
  confSlider.type = 'range';
  confSlider.min = '0';
  confSlider.max = '1';
  confSlider.step = '0.05';
  confSlider.value = String(_confThresh);
  confSlider.className = 'inference-conf-slider';

  const confValue = document.createElement('span');
  confValue.className = 'inference-conf-value';
  confValue.textContent = `${Math.round(_confThresh * 100)}%`;

  confSlider.addEventListener('input', () => {
    _confThresh = parseFloat(confSlider.value);
    confValue.textContent = `${Math.round(_confThresh * 100)}%`;
  });

  confRow.append(confLabel, confSlider, confValue);

  // ── Replace-existing-bboxes checkbox ──────────────────────────────
  const replaceRow = document.createElement('label');
  replaceRow.className = 'inference-replace-row';

  const replaceCheckbox = document.createElement('input');
  replaceCheckbox.type = 'checkbox';
  replaceCheckbox.checked = _replaceExisting;
  replaceCheckbox.addEventListener('change', () => {
    _replaceExisting = replaceCheckbox.checked;
  });

  const replaceLabel = document.createElement('span');
  replaceLabel.textContent = 'Replace existing bboxes';

  replaceRow.append(replaceCheckbox, replaceLabel);
  replaceRow.title = 'When a detection significantly overlaps an existing box with a different label, replace it instead of leaving it as-is';

  actionsEl.append(runBtn, refreshBtn, clearGreyBtn, confRow, replaceRow);

  // ── Model sections ────────────────────────────────────────────────
  const listEl = document.createElement('div');
  listEl.className = 'inference-model-list';

  container.append(actionsEl, listEl);

  // ── Helpers ───────────────────────────────────────────────────────

  function _updateRunBtn() {
    if (!_selected) {
      runBtn.textContent = 'Run';
      runBtn.disabled = false;
      clearGreyBtn.style.display = '';
      replaceRow.style.display = '';
      return;
    }
    if (_selectedKind === 'classifier') {
      runBtn.textContent = 'Classify';
      clearGreyBtn.style.display = 'none';
      replaceRow.style.display = 'none';
    } else {
      runBtn.textContent = 'Run Detection';
      clearGreyBtn.style.display = '';
      replaceRow.style.display = '';
    }
  }

  function _makeItem(model, kind) {
    const item = document.createElement('div');
    item.className = 'inference-model-item' + (_selected === model ? ' selected' : '');
    item.title = model;

    const nameSpan = document.createElement('span');
    nameSpan.textContent = model;

    const badge = document.createElement('span');
    badge.className = 'inference-model-badge';
    badge.textContent = kind === 'image_classifier' ? 'binary'
      : kind === 'multi_label_classifier' ? 'multi'
      : 'yolo';

    item.append(nameSpan, badge);
    item.addEventListener('click', () => {
      _selected = model;
      _selectedKind = (kind === 'image_classifier' || kind === 'multi_label_classifier') ? 'classifier' : 'detector';
      renderList();
      _updateRunBtn();
    });
    return item;
  }

  function _makeSection(title, models, kindMap) {
    if (models.length === 0) return null;
    const section = document.createElement('div');
    section.className = 'inference-model-section';

    const header = document.createElement('div');
    header.className = 'inference-model-section-header';
    header.textContent = title;
    section.appendChild(header);

    for (const model of models) {
      section.appendChild(_makeItem(model, kindMap.get(model) ?? 'detector'));
    }
    return section;
  }

  function renderList() {
    listEl.innerHTML = '';
    if (_detectors.length === 0 && _classifiers.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'inference-model-empty';
      empty.textContent = 'No models found';
      listEl.appendChild(empty);
      return;
    }

    if (_kindMap) {
      const detSection = _makeSection('Detectors', _detectors, _kindMap);
      const clfSection = _makeSection('Classifiers', _classifiers, _kindMap);
      if (detSection) listEl.appendChild(detSection);
      if (clfSection) listEl.appendChild(clfSection);
    }
  }

  /** @type {Map<string, string>|null} */
  let _kindMap = null;

  async function _loadModels() {
    const allModels = await fetchModelsList();
    _kindMap = await fetchModelsKinds(allModels);
    _detectors = allModels.filter(m => {
      const k = _kindMap.get(m);
      return k !== 'image_classifier' && k !== 'multi_label_classifier';
    });
    _classifiers = allModels.filter(m => {
      const k = _kindMap.get(m);
      return k === 'image_classifier' || k === 'multi_label_classifier';
    });

    // Preserve selection if still valid, otherwise default to first detector.
    if (_selected && !allModels.includes(_selected)) _selected = null;
    if (!_selected) {
      if (_detectors.length > 0) { _selected = _detectors[0]; _selectedKind = 'detector'; }
      else if (_classifiers.length > 0) { _selected = _classifiers[0]; _selectedKind = 'classifier'; }
    }
  }

  async function handleRefresh() {
    refreshBtn.disabled = true;
    try {
      clearModelsCache();
      await _loadModels();
      renderList();
      _updateRunBtn();
    } catch (err) {
      Logger.error('Failed to refresh models:', err);
    } finally {
      refreshBtn.disabled = false;
    }
  }

  async function handleRun() {
    if (!_selected) { alert('Please select a model first'); return; }

    if (_selectedKind === 'classifier') {
      await handleClassify();
    } else {
      events.publish(EventTypes.RUN_INFERENCE_ON_CANVAS, {
        modelName: _selected,
        runtimeId: runtime.id,
        confThresh: _confThresh,
        replaceExisting: _replaceExisting,
      });
    }
  }

  async function handleClassify() {
    const img = runtime.state?.image?.img;
    const imagePath = runtime.state?.image?.image_path;
    if (!img) { alert('No image loaded'); return; }

    runBtn.disabled = true;
    try {
      const predictions = await runClassify(_selected, img);
      if (!predictions.length) { alert('Model returned no predictions'); return; }
      await openClassifyModal(predictions, imagePath);
    } catch (err) {
      Logger.error('Classify failed:', err);
      alert(`Classification failed: ${err.message}`);
    } finally {
      runBtn.disabled = false;
    }
  }

  /**
   * Show a modal with classifier predictions. Checked labels are applied on confirm.
   * @param {Array<{labelUuid: string, confidence: number, isPresent: boolean}>} predictions
   * @param {string|null} imagePath
   */
  async function openClassifyModal(predictions, imagePath) {
    const { dialog, open, close } = createModalShell();

    // ── Title ────────────────────────────────────────────────────────
    const title = document.createElement('h3');
    title.className = 'app-modal-title';
    title.textContent = `Classifier results — ${_selected}`;
    dialog.appendChild(title);

    // ── Prediction rows ──────────────────────────────────────────────
    const body = document.createElement('div');
    body.className = 'classify-modal-body';
    dialog.appendChild(body);

    /** @type {Array<{labelUuid: string, checkbox: HTMLInputElement}>} */
    const rows = [];

    for (const pred of predictions) {
      const label = await fetchLabelByUuid(pred.labelUuid).catch(() => null);
      const labelName = label?.name ?? pred.labelUuid;
      const pct = Math.round(pred.confidence * 100);

      const row = document.createElement('label');
      row.className = 'classify-modal-row';

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = pred.isPresent;

      const nameSpan = document.createElement('span');
      nameSpan.className = 'classify-modal-label-name';
      nameSpan.textContent = labelName;

      const barWrap = document.createElement('div');
      barWrap.className = 'classify-modal-bar-wrap';
      const bar = document.createElement('div');
      bar.className = 'classify-modal-bar';
      bar.style.width = `${pct}%`;
      bar.style.background = pred.isPresent ? 'var(--pico-primary)' : 'var(--pico-muted-border-color)';
      barWrap.appendChild(bar);

      const pctSpan = document.createElement('span');
      pctSpan.className = 'classify-modal-pct';
      pctSpan.textContent = `${pct}%`;

      row.append(cb, nameSpan, barWrap, pctSpan);
      body.appendChild(row);
      rows.push({ labelUuid: pred.labelUuid, checkbox: cb });
    }

    // ── Buttons ───────────────────────────────────────────────────────
    const footer = document.createElement('div');
    footer.className = 'classify-modal-footer';

    const applyBtn = document.createElement('button');
    applyBtn.textContent = 'Apply';
    applyBtn.addEventListener('click', async () => {
      if (!imagePath) { alert('No image path available'); return; }
      applyBtn.disabled = true;
      try {
        const checked = rows.filter(r => r.checkbox.checked);
        await Promise.all(
          checked.map(r => setImageLabel(imagePath, r.labelUuid, 'present'))
        );
        close();
      } catch (err) {
        Logger.error('Failed to apply labels:', err);
        alert(`Failed to apply: ${err.message}`);
        applyBtn.disabled = false;
      }
    });

    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'secondary';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.addEventListener('click', close);

    footer.append(applyBtn, cancelBtn);
    dialog.appendChild(footer);

    open();
  }

  // ── Initial load ──────────────────────────────────────────────────
  try {
    await _loadModels();
    renderList();
    _updateRunBtn();
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
