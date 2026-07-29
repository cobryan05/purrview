import { fetchModelsList, fetchModelsKinds, clearModelsCache } from '/app-static/js/shared/api/models.js';
import { runClassify } from '/app-static/js/shared/api/inference.js';
import { setImageLabel } from '/app-static/js/shared/api/imageLabels.js';
import { fetchLabelByUuid } from '/app-static/js/shared/api/labels.js';
import { fetchModelGroups, createModelGroup, clearModelGroupsCache } from '/app-static/js/shared/api/modelGroups.js';
import { expandModelGroupMembers } from '/app-static/js/shared/modelGroups/expand.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { Logger } from '/app-static/js/ui/utils/index.js';
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { clearUnmatched, verifyAllBboxes } from './canvas/image.js';

const SESSION_SELECTED_DETECTORS_KEY = 'wtch_inference_selected_detectors';
const SESSION_SELECTED_CLASSIFIER_KEY = 'wtch_inference_selected_classifier';

const MODEL_ID_PREFIX = 'model:';
const GROUP_ID_PREFIX = 'group:';

/**
 * Creates a self-contained inference panel: detector + classifier model lists and Run button.
 *
 * Detectors and saved model groups support multi-select (checkboxes) so several can be run
 * back-to-back; classifiers stay single-select since a match opens a review modal for that
 * one model. The detector/group selection is a single ordered `Set` of tagged ids
 * (`model:<name>` / `group:<uuid>`) — `Set` preserves insertion order, so a mix of raw
 * models and groups still runs in the order they were checked, and a saved group expands to
 * its member models (recursively, for nested groups) at Run time.
 * Selections persist in sessionStorage and are reconciled against the live model/group list
 * on every load/refresh, so models or groups that disappear are silently dropped and new
 * ones simply arrive unchecked.
 *
 * @param {import('../image-tagging/runtime.js').ImageTaggingRuntime} runtime
 * @param {HTMLElement} container
 * @returns {Promise<{ destroy: () => void }>}
 */
export async function createInferencePanel(runtime, container) {
  /** @type {string[]} */ let _detectors = [];
  /** @type {string[]} */ let _classifiers = [];
  /** @type {Array<{uuid: string, name: string}>} */ let _groups = [];
  /** @type {Map<string, Object>} */ let _groupsMap = new Map();
  /** @type {Set<string>} tagged ids: "model:<name>" or "group:<uuid>"; insertion order = run order */
  let _selectedItems = new Set();
  /** @type {string|null} */ let _selectedClassifier = null;
  /** @type {number} */ let _confThresh = 0.25;
  /** @type {boolean} */ let _replaceExisting = false;

  container.innerHTML = '';
  container.classList.add('inference-panel');

  // ── Action buttons ────────────────────────────────────────────────
  const actionsEl = document.createElement('div');
  actionsEl.className = 'inference-panel-actions';

  const buttonsRow = document.createElement('div');
  buttonsRow.className = 'inference-buttons-row';

  const runBtn = document.createElement('button');
  runBtn.textContent = 'Run';
  runBtn.addEventListener('click', handleRun);

  const refreshBtn = document.createElement('button');
  refreshBtn.textContent = '↻';
  refreshBtn.title = 'Refresh: reload the model list from the server';
  refreshBtn.addEventListener('click', handleRefresh);

  const clearGreyBtn = document.createElement('button');
  clearGreyBtn.textContent = 'Clear';
  clearGreyBtn.title = 'Clear Grey: remove unmatched (grey) detections from the canvas';
  clearGreyBtn.addEventListener('click', () => clearUnmatched(runtime));

  // Whole-image action, not detector-specific — shared with the Labels panel's
  // bulk-op row (same underlying verifyAllBboxes helper) so it's reachable
  // from wherever the user happens to be reviewing detections from.
  const verifyAllBtn = document.createElement('button');
  verifyAllBtn.textContent = 'Verify all';
  verifyAllBtn.title = 'Mark every bbox on this image as verified';
  verifyAllBtn.addEventListener('click', () => verifyAllBboxes(runtime));

  const saveGroupBtn = document.createElement('button');
  saveGroupBtn.className = 'secondary';
  saveGroupBtn.textContent = 'Save as group';
  saveGroupBtn.title = 'Save the currently checked detectors/groups, in their current order, as a reusable named group';
  saveGroupBtn.addEventListener('click', handleSaveAsGroup);

  buttonsRow.append(runBtn, refreshBtn, clearGreyBtn, verifyAllBtn, saveGroupBtn);

  // ── Min confidence slider + replace-existing checkbox (one row) ───
  const optionsRow = document.createElement('div');
  optionsRow.className = 'inference-options-row';

  const confRow = document.createElement('div');
  confRow.className = 'inference-conf-row';
  confRow.title = 'Minimum confidence threshold: detections below this score are discarded';

  const confLabel = document.createElement('label');
  confLabel.className = 'inference-conf-label';
  confLabel.textContent = 'Conf:';

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
  replaceRow.title = 'Replace existing bboxes: when a detection significantly overlaps an existing box with a different label, replace it instead of leaving it as-is. With multiple models running, this applies independently for each model in whatever order it happens to finish.';

  const replaceCheckbox = document.createElement('input');
  replaceCheckbox.type = 'checkbox';
  replaceCheckbox.checked = _replaceExisting;
  replaceCheckbox.addEventListener('change', () => {
    _replaceExisting = replaceCheckbox.checked;
  });

  const replaceLabel = document.createElement('span');
  replaceLabel.textContent = 'Replace';

  replaceRow.append(replaceCheckbox, replaceLabel);

  optionsRow.append(confRow, replaceRow);
  actionsEl.append(buttonsRow, optionsRow);

  // ── Model sections ────────────────────────────────────────────────
  const listEl = document.createElement('div');
  listEl.className = 'inference-model-list';

  container.append(actionsEl, listEl);

  // ── Persistence ───────────────────────────────────────────────────

  function _loadPersistedSelection() {
    try {
      const arr = JSON.parse(sessionStorage.getItem(SESSION_SELECTED_DETECTORS_KEY) || '[]');
      if (Array.isArray(arr)) _selectedItems = new Set(arr);
    } catch { /* ignore parse errors */ }
    _selectedClassifier = sessionStorage.getItem(SESSION_SELECTED_CLASSIFIER_KEY) || null;
  }

  function _persistSelection() {
    sessionStorage.setItem(SESSION_SELECTED_DETECTORS_KEY, JSON.stringify([..._selectedItems]));
    if (_selectedClassifier) sessionStorage.setItem(SESSION_SELECTED_CLASSIFIER_KEY, _selectedClassifier);
    else sessionStorage.removeItem(SESSION_SELECTED_CLASSIFIER_KEY);
  }

  // ── Helpers ───────────────────────────────────────────────────────

  /** Resolve a tagged id back to its display name (model name, or group name). */
  function _displayNameFor(id) {
    if (id.startsWith(MODEL_ID_PREFIX)) return id.slice(MODEL_ID_PREFIX.length);
    const group = _groupsMap.get(id.slice(GROUP_ID_PREFIX.length));
    return group ? group.name : id;
  }

  function _updateRunBtn() {
    if (_selectedClassifier) {
      runBtn.title = `Run classifier: ${_selectedClassifier}`;
      clearGreyBtn.style.display = 'none';
      replaceRow.style.display = 'none';
    } else if (_selectedItems.size > 0) {
      runBtn.title = `Run Detection: ${[..._selectedItems].map(_displayNameFor).join(', ')}`;
      clearGreyBtn.style.display = '';
      replaceRow.style.display = '';
    } else {
      runBtn.title = 'Select one or more models/groups below first';
      clearGreyBtn.style.display = '';
      replaceRow.style.display = '';
    }
    saveGroupBtn.disabled = _selectedClassifier != null || _selectedItems.size === 0;
  }

  function _makeToggleItem({ id, label, badgeText }) {
    const item = document.createElement('label');
    item.className = 'inference-model-item' + (_selectedItems.has(id) ? ' selected' : '');
    item.title = label;

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = _selectedItems.has(id);
    cb.addEventListener('change', () => {
      if (cb.checked) _selectedItems.add(id);
      else _selectedItems.delete(id);
      _selectedClassifier = null;
      _persistSelection();
      renderList();
      _updateRunBtn();
    });

    const nameSpan = document.createElement('span');
    nameSpan.className = 'inference-model-name';
    nameSpan.textContent = label;

    const badge = document.createElement('span');
    badge.className = 'inference-model-badge';
    badge.textContent = badgeText;

    item.append(cb, nameSpan, badge);
    return item;
  }

  function _makeDetectorItem(model) {
    return _makeToggleItem({ id: MODEL_ID_PREFIX + model, label: model, badgeText: 'yolo' });
  }

  function _makeGroupItem(group) {
    return _makeToggleItem({ id: GROUP_ID_PREFIX + group.uuid, label: group.name, badgeText: 'group' });
  }

  function _makeClassifierItem(model, kind) {
    const item = document.createElement('div');
    item.className = 'inference-model-item' + (_selectedClassifier === model ? ' selected' : '');
    item.title = model;

    const nameSpan = document.createElement('span');
    nameSpan.className = 'inference-model-name';
    nameSpan.textContent = model;

    const badge = document.createElement('span');
    badge.className = 'inference-model-badge';
    badge.textContent = kind === 'image_classifier' ? 'binary' : 'multi';

    item.append(nameSpan, badge);
    item.addEventListener('click', () => {
      _selectedClassifier = model;
      _selectedItems.clear();
      _persistSelection();
      renderList();
      _updateRunBtn();
    });
    return item;
  }

  function _makeSection(title, models, makeItem) {
    if (models.length === 0) return null;
    const section = document.createElement('div');
    section.className = 'inference-model-section';

    const header = document.createElement('div');
    header.className = 'inference-model-section-header';
    header.textContent = title;
    section.appendChild(header);

    for (const model of models) {
      section.appendChild(makeItem(model));
    }
    return section;
  }

  function renderList() {
    listEl.innerHTML = '';
    if (_detectors.length === 0 && _classifiers.length === 0 && _groups.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'inference-model-empty';
      empty.textContent = 'No models found';
      listEl.appendChild(empty);
      return;
    }

    if (_kindMap) {
      const detSection = _makeSection('Detectors', _detectors, _makeDetectorItem);
      const clfSection = _makeSection('Classifiers', _classifiers, m => _makeClassifierItem(m, _kindMap.get(m)));
      const groupSection = _makeSection('Groups', _groups, _makeGroupItem);
      if (detSection) listEl.appendChild(detSection);
      if (clfSection) listEl.appendChild(clfSection);
      if (groupSection) listEl.appendChild(groupSection);
    }
  }

  /** @type {Map<string, string>|null} */
  let _kindMap = null;

  async function _loadModels({ isInitial = false } = {}) {
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

    _groupsMap = await fetchModelGroups();
    _groups = [..._groupsMap.values()].sort((a, b) => a.name.localeCompare(b.name));

    // Drop persisted selections for models/groups that no longer exist, without disturbing
    // selections that are still present (new models/groups simply arrive unchecked).
    for (const id of [..._selectedItems]) {
      if (id.startsWith(MODEL_ID_PREFIX)) {
        if (!_detectors.includes(id.slice(MODEL_ID_PREFIX.length))) _selectedItems.delete(id);
      } else if (id.startsWith(GROUP_ID_PREFIX)) {
        if (!_groupsMap.has(id.slice(GROUP_ID_PREFIX.length))) _selectedItems.delete(id);
      } else {
        // Pre-groups-feature sessionStorage held bare model names — migrate them to
        // the tagged form so existing persisted selections survive the upgrade.
        _selectedItems.delete(id);
        if (_detectors.includes(id)) _selectedItems.add(MODEL_ID_PREFIX + id);
      }
    }
    if (_selectedClassifier && !_classifiers.includes(_selectedClassifier)) _selectedClassifier = null;

    // First-ever load with nothing persisted: default to the first detector, matching
    // the previous single-select panel's behavior.
    if (isInitial && _selectedItems.size === 0 && !_selectedClassifier && _detectors.length > 0) {
      _selectedItems.add(MODEL_ID_PREFIX + _detectors[0]);
    }

    _persistSelection();
  }

  async function handleRefresh() {
    refreshBtn.disabled = true;
    try {
      clearModelsCache();
      clearModelGroupsCache();
      await _loadModels();
      renderList();
      _updateRunBtn();
    } catch (err) {
      Logger.error('Failed to refresh models:', err);
    } finally {
      refreshBtn.disabled = false;
    }
  }

  /** Expand the current selection (models pass through, groups recurse) into a flat, ordered model-name list. */
  function _resolveSelectedModelNames() {
    const names = [];
    for (const id of _selectedItems) {
      if (id.startsWith(MODEL_ID_PREFIX)) {
        names.push(id.slice(MODEL_ID_PREFIX.length));
      } else if (id.startsWith(GROUP_ID_PREFIX)) {
        names.push(...expandModelGroupMembers(id.slice(GROUP_ID_PREFIX.length), _groupsMap));
      }
    }
    return names;
  }

  async function handleRun() {
    if (_selectedClassifier) {
      await handleClassify();
      return;
    }
    if (_selectedItems.size === 0) { alert('Please select at least one model or group first'); return; }

    events.publish(EventTypes.RUN_INFERENCE_ON_CANVAS, {
      modelNames: _resolveSelectedModelNames(),
      runtimeId: runtime.id,
      confThresh: _confThresh,
      replaceExisting: _replaceExisting,
    });
  }

  async function handleSaveAsGroup() {
    if (_selectedItems.size === 0) return;
    const name = prompt('Model group name:');
    if (!name?.trim()) return;

    const members = [..._selectedItems].map((id, order) =>
      id.startsWith(MODEL_ID_PREFIX)
        ? { order, model_name: id.slice(MODEL_ID_PREFIX.length) }
        : { order, child_group_uuid: id.slice(GROUP_ID_PREFIX.length) }
    );

    saveGroupBtn.disabled = true;
    try {
      await createModelGroup({ name: name.trim(), members });
      _groupsMap = await fetchModelGroups();
      _groups = [..._groupsMap.values()].sort((a, b) => a.name.localeCompare(b.name));
      renderList();
      _updateRunBtn();
    } catch (err) {
      Logger.error('Failed to save model group:', err);
      alert(`Failed to save group: ${err.message}`);
    } finally {
      saveGroupBtn.disabled = false;
    }
  }

  async function handleClassify() {
    const img = runtime.state?.image?.img;
    const imagePath = runtime.state?.image?.image_path;
    if (!img) { alert('No image loaded'); return; }

    runBtn.disabled = true;
    try {
      const predictions = await runClassify(_selectedClassifier, img);
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
    title.textContent = `Classifier results — ${_selectedClassifier}`;
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
  _loadPersistedSelection();
  try {
    await _loadModels({ isInitial: true });
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
