/**
 * Training configuration manager panel.
 *
 * Rendered inside the Configuration tab's "Training" pane.
 *
 * Each training config ties together:
 *   - A dataset (the pool of images)
 *   - A set of target labels (whose verification state drives readiness)
 *   - A trainer type (yolo_v8 | image_classifier)
 *   - An optional model config reference
 *   - Optional hyperparameters (raw JSON)
 *
 * The readiness panel shows how many dataset images have all target labels
 * explicitly verified (ready to train) vs. still unreviewed.
 */

import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { fetchDatasets } from '/app-static/js/shared/api/datasets.js';
import { fetchModelConfigs } from '/app-static/js/shared/api/modelConfigs.js';
import {
  fetchTrainingConfigs,
  createTrainingConfig,
  updateTrainingConfig,
  deleteTrainingConfig,
  fetchTrainingConfigReadiness,
  clearTrainingConfigsCache,
} from '/app-static/js/shared/api/trainingConfigs.js';
import { flattenHierarchy } from '/app-static/js/ui/labelList.js';
import { makeFieldRow, buildPanelEditorBar, toast } from '/app-static/js/ui/utils/index.js';
import { showUnsavedConfigChangesModal } from '/app-static/js/ui/utils/confirmModal.js';

const TRAINER_TYPENAMES = ['yolo_v8', 'image_classifier'];

let _rendering = false;
let _navGuardController = null;

// ── Form ─────────────────────────────────────────────────────────────────────

function renderTrainingConfigForm({ config, labelsMap, datasetsMap, modelConfigsMap, onDirtyChange }) {
  const el = document.createElement('div');

  function _markDirty() { onDirtyChange?.(true); }

  // Name
  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.value = config?.name ?? '';
  nameInput.placeholder = 'Required';
  nameInput.addEventListener('input', _markDirty);
  el.appendChild(makeFieldRow('Name', nameInput));

  // Description
  const descInput = document.createElement('input');
  descInput.type = 'text';
  descInput.value = config?.description ?? '';
  descInput.placeholder = 'Optional';
  descInput.addEventListener('input', _markDirty);
  el.appendChild(makeFieldRow('Description', descInput));

  // Trainer type
  const trainerSel = document.createElement('select');
  trainerSel.style.cssText = 'margin-bottom: 0; width: auto;';
  TRAINER_TYPENAMES.forEach(tn => {
    const o = document.createElement('option');
    o.value = tn;
    o.textContent = tn;
    if (tn === (config?.trainerType ?? TRAINER_TYPENAMES[0])) o.selected = true;
    trainerSel.appendChild(o);
  });
  trainerSel.addEventListener('change', _markDirty);
  el.appendChild(makeFieldRow('Trainer type', trainerSel));

  // Dataset (required)
  const datasetSel = document.createElement('select');
  datasetSel.style.cssText = 'margin-bottom: 0; width: auto;';
  const blankDataset = document.createElement('option');
  blankDataset.value = '';
  blankDataset.textContent = '— select dataset —';
  datasetSel.appendChild(blankDataset);
  for (const [uuid, ds] of datasetsMap) {
    const o = document.createElement('option');
    o.value = uuid;
    o.textContent = ds.name;
    if (uuid === config?.datasetUuid) o.selected = true;
    datasetSel.appendChild(o);
  }
  datasetSel.addEventListener('change', _markDirty);
  el.appendChild(makeFieldRow('Dataset', datasetSel));

  // Model config (optional)
  const modelSel = document.createElement('select');
  modelSel.style.cssText = 'margin-bottom: 0; width: auto;';
  const blankModel = document.createElement('option');
  blankModel.value = '';
  blankModel.textContent = '— none —';
  modelSel.appendChild(blankModel);
  for (const [uuid, mc] of modelConfigsMap) {
    const o = document.createElement('option');
    o.value = uuid;
    o.textContent = mc.name;
    if (uuid === config?.modelConfigUuid) o.selected = true;
    modelSel.appendChild(o);
  }
  modelSel.addEventListener('change', _markDirty);
  el.appendChild(makeFieldRow('Model config (optional)', modelSel));

  // Hyperparameters (raw JSON textarea)
  const hyperText = document.createElement('textarea');
  hyperText.rows = 4;
  hyperText.style.cssText = 'font-family: monospace; font-size: 0.8em; width: 100%; box-sizing: border-box;';
  hyperText.placeholder = '{}';
  hyperText.value = JSON.stringify(config?.hyperparamsJson ?? {}, null, 2);
  hyperText.addEventListener('input', _markDirty);
  el.appendChild(makeFieldRow('Hyperparameters (JSON)', hyperText));

  // ── Label multi-select ────────────────────────────────────────────────────
  const hr = document.createElement('hr');
  hr.style.cssText = 'margin: 0.6em 0; opacity: 0.4;';
  el.appendChild(hr);

  const lblHeader = document.createElement('p');
  lblHeader.className = 'panel-section-label';
  lblHeader.style.marginTop = '0';
  lblHeader.textContent = 'Target labels (readiness tracking)';
  el.appendChild(lblHeader);

  const lblNote = document.createElement('p');
  lblNote.style.cssText = 'font-size: 0.8em; color: var(--pico-muted-color, #888); margin: 0 0 0.5em;';
  lblNote.textContent = 'Images in the dataset are "ready" when all selected labels have been explicitly verified (Present or Confirmed Not Present).';
  el.appendChild(lblNote);

  const checkboxes = new Map(); // uuid → checkbox el
  const flatLabels = flattenHierarchy(labelsMap);
  const currentLabelUuids = new Set(config?.labelUuids ?? []);

  const treeBox = document.createElement('div');
  treeBox.className = 'panel-tree-box';
  treeBox.style.cssText = 'max-height: 200px; overflow-y: auto; border: 1px solid var(--pico-border-color, #444); border-radius: 4px; padding: 4px;';

  for (const label of flatLabels) {
    const row = document.createElement('label');
    row.style.cssText = [
      'display: flex', 'align-items: center', 'gap: 0.4em',
      `padding-left: ${6 + label.depth * 14}px`,
      'cursor: pointer', 'user-select: none', 'font-size: 0.88em',
      'padding-top: 2px', 'padding-bottom: 2px',
    ].join('; ');

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = label.uuid;
    cb.checked = currentLabelUuids.has(label.uuid);
    cb.style.cssText = 'margin: 0; flex-shrink: 0;';
    cb.addEventListener('change', _markDirty);

    if (label.color) {
      const dot = document.createElement('span');
      dot.style.cssText = `display:inline-block;width:8px;height:8px;border-radius:50%;background:${label.color};flex-shrink:0;`;
      row.appendChild(cb);
      row.appendChild(dot);
    } else {
      row.appendChild(cb);
    }

    const nameSpan = document.createElement('span');
    nameSpan.textContent = label.name;
    row.appendChild(nameSpan);

    checkboxes.set(label.uuid, cb);
    treeBox.appendChild(row);
  }
  el.appendChild(treeBox);

  return {
    el,
    getSaveData() {
      const name = nameInput.value.trim();
      const description = descInput.value.trim();
      const trainerType = trainerSel.value;
      const datasetUuid = datasetSel.value;
      const modelConfigUuid = modelSel.value || null;
      const labelUuids = [...checkboxes.entries()]
        .filter(([, cb]) => cb.checked)
        .map(([uuid]) => uuid);
      let hyperparamsJson = {};
      try { hyperparamsJson = JSON.parse(hyperText.value || '{}'); } catch {}
      return { name, description, trainerType, datasetUuid, modelConfigUuid, labelUuids, hyperparamsJson };
    },
    destroy() {},
  };
}

// ── Readiness panel ───────────────────────────────────────────────────────────

function renderReadinessPanel(config) {
  const wrap = document.createElement('div');
  wrap.style.cssText = 'margin-top: 0.75em; padding: 0.6em 0.75em; border: 1px solid var(--pico-border-color, #444); border-radius: 4px; font-size: 0.85em;';

  const title = document.createElement('div');
  title.style.cssText = 'font-weight: bold; margin-bottom: 0.4em;';
  title.textContent = 'Training readiness';
  wrap.appendChild(title);

  const status = document.createElement('div');
  status.textContent = 'Loading…';
  wrap.appendChild(status);

  const reviewBtn = document.createElement('button');
  reviewBtn.type = 'button';
  reviewBtn.textContent = 'Review unverified images';
  reviewBtn.className = 'panel-btn';
  reviewBtn.style.cssText = 'margin-top: 0.4em; font-size: 0.8em;';
  reviewBtn.style.display = 'none';
  reviewBtn.addEventListener('click', () => {
    // Switch to review tab; user can apply the dataset filter there.
    document.querySelector('[data-tab="tab-review"]')?.click();
  });
  wrap.appendChild(reviewBtn);

  fetchTrainingConfigReadiness(config.uuid).then(({ readyCount, needsReviewCount }) => {
    const total = readyCount + needsReviewCount;
    if (total === 0) {
      status.textContent = 'No images in dataset.';
    } else {
      status.innerHTML = `<span style="color:#4c4">✓ ${readyCount} ready</span> &nbsp;·&nbsp; <span style="color:#f90">⚠ ${needsReviewCount} need review</span> &nbsp;(${total} total)`;
      if (needsReviewCount > 0) reviewBtn.style.display = '';
    }
  }).catch(err => {
    status.textContent = `Could not load readiness: ${err.message}`;
  });

  return wrap;
}

// ── Public entry point ────────────────────────────────────────────────────────

export async function renderTrainingConfigManager({ target = 'training-config-box' } = {}) {
  if (_rendering) return;
  _rendering = true;

  _navGuardController?.abort();
  _navGuardController = new AbortController();

  const container = document.getElementById(target);
  if (!container) { _rendering = false; return; }
  container.innerHTML = '';

  try {
    const split = document.createElement('div');
    split.style.cssText = 'display: flex; gap: 1em; align-items: flex-start;';
    container.appendChild(split);

    // ── Left: list panel ──────────────────────────────────────────────────
    const listPanel = document.createElement('div');
    listPanel.className = 'panel-box mc-list-panel';
    split.appendChild(listPanel);

    const listBar = document.createElement('div');
    listBar.className = 'panel-box__bar';
    const listTitle = document.createElement('span');
    listTitle.className = 'panel-box__title';
    listTitle.textContent = 'Training Configs';
    listBar.appendChild(listTitle);
    listPanel.appendChild(listBar);

    const listBody = document.createElement('div');
    listBody.className = 'panel-box__body';
    listPanel.appendChild(listBody);

    const listFooter = document.createElement('div');
    listFooter.className = 'panel-box__footer';
    listPanel.appendChild(listFooter);

    const newBtn = document.createElement('button');
    newBtn.type = 'button';
    newBtn.textContent = '+ New';
    newBtn.className = 'panel-btn';
    newBtn.style.cssText = 'width: 100%; margin-bottom: 0;';
    newBtn.addEventListener('click', () => _showForm(null));
    listFooter.appendChild(newBtn);

    // ── Right: editor panel ───────────────────────────────────────────────
    const editorPanel = document.createElement('div');
    editorPanel.className = 'panel-box mc-editor';
    split.appendChild(editorPanel);

    const editorBar = document.createElement('div');
    editorBar.className = 'panel-box__bar';
    editorPanel.appendChild(editorBar);

    const editorBody = document.createElement('div');
    editorBody.className = 'panel-box__body';
    editorPanel.appendChild(editorBody);

    // ── Shared state ──────────────────────────────────────────────────────
    let _selectedUuid = null;
    let _barHandle = null;
    let _currentFormHandle = null;
    let _currentDoSave = null;
    let _switching = false;

    async function _confirmIfDirty() {
      if (!_barHandle?.getDirty()) return true;
      const result = await showUnsavedConfigChangesModal();
      if (result === 'cancel') return false;
      if (result === 'save') return await (_currentDoSave?.() ?? false);
      return true;
    }

    function _updateListSelection(uuid) {
      for (const item of listBody.querySelectorAll('.mc-list-item')) {
        item.classList.toggle('active', item.dataset.configUuid === uuid);
      }
    }

    async function _showForm(config, { force = false } = {}) {
      if (_switching && !force) return;
      _switching = true;
      try {
        if (!force && !await _confirmIfDirty()) return;
        _currentFormHandle?.destroy();
        _currentFormHandle = null;
        _currentDoSave = null;
        _barHandle = null;
        _selectedUuid = config?.uuid ?? null;

        const [labelsMap, datasetsMap, modelConfigsMap] = await Promise.all([
          fetchLabels(),
          fetchDatasets(),
          fetchModelConfigs(),
        ]);
        editorBody.innerHTML = '';

        _barHandle = buildPanelEditorBar({
          container: editorBar,
          item: config,
          newTitle: 'New Training Config',
          deleteTitle: 'Delete training config',
          onSave: async () => {
            const data = _currentFormHandle.getSaveData();
            if (!data.name) { toast('Name is required', 3000, 'error'); return; }
            if (!data.datasetUuid) { toast('Dataset is required', 3000, 'error'); return; }
            setSaving(true);
            try {
              if (config) {
                const updated = await updateTrainingConfig(config.uuid, data);
                setDirty(false);
                toast('Saved', 3000);
                const nameEl = listBody.querySelector(`[data-config-uuid="${config.uuid}"] .mc-list-item__name`);
                if (nameEl) nameEl.textContent = updated.name;
                await _showForm(updated, { force: true });
              } else {
                const created = await createTrainingConfig(data);
                setDirty(false);
                toast(`Created "${created.name}"`, 3000);
                await refreshList();
                await _showForm(created, { force: true });
              }
            } catch (err) {
              toast(err.message, 5000, 'error');
              setSaving(false);
            }
          },
          onDelete: config ? async () => {
            await deleteTrainingConfig(config.uuid);
            toast('Deleted', 3000);
            await refreshList();
            await _showForm(null, { force: true });
          } : undefined,
        });
        const { setDirty, setSaving } = _barHandle;

        const formHandle = renderTrainingConfigForm({
          config,
          labelsMap,
          datasetsMap,
          modelConfigsMap,
          onDirtyChange: setDirty,
        });
        _currentFormHandle = formHandle;
        editorBody.appendChild(formHandle.el);

        // Readiness panel (only for existing configs with a dataset and labels)
        if (config?.uuid && config.datasetUuid && (config.labelUuids?.length ?? 0) > 0) {
          editorBody.appendChild(renderReadinessPanel(config));
        }

        _currentDoSave = async () => {
          const data = formHandle.getSaveData();
          if (!data.name) { toast('Name is required', 3000, 'error'); return false; }
          if (!data.datasetUuid) { toast('Dataset is required', 3000, 'error'); return false; }
          setSaving(true);
          try {
            if (config) {
              await updateTrainingConfig(config.uuid, data);
              setDirty(false);
              toast('Saved', 3000);
              const nameEl = listBody.querySelector(`[data-config-uuid="${config.uuid}"] .mc-list-item__name`);
              if (nameEl) nameEl.textContent = data.name;
            } else {
              await createTrainingConfig(data);
              setDirty(false);
              await refreshList();
            }
            setSaving(false);
            return true;
          } catch (err) {
            toast(err.message, 5000, 'error');
            setSaving(false);
            return false;
          }
        };

        _updateListSelection(config?.uuid ?? null);
      } finally {
        _switching = false;
      }
    }

    async function refreshList() {
      listBody.innerHTML = '';
      clearTrainingConfigsCache();
      const configsMap = await fetchTrainingConfigs();

      if (configsMap.size === 0) {
        const empty = document.createElement('p');
        empty.style.cssText = 'color: var(--pico-muted-color, #888); font-style: italic; font-size: 0.85em; padding: 0.5em 0.75em; margin: 0;';
        empty.textContent = 'No training configs yet.';
        listBody.appendChild(empty);
      }

      const datasetsMap = await fetchDatasets().catch(() => new Map());

      for (const config of configsMap.values()) {
        const item = document.createElement('div');
        item.className = 'mc-list-item';
        item.dataset.configUuid = config.uuid;

        const nameLine = document.createElement('div');
        nameLine.className = 'mc-list-item__name';
        nameLine.textContent = config.name;
        item.appendChild(nameLine);

        const metaLine = document.createElement('div');
        metaLine.className = 'mc-list-item__meta';
        const dsName = datasetsMap.get(config.datasetUuid)?.name ?? config.datasetUuid;
        const nLabels = config.labelUuids?.length ?? 0;
        metaLine.textContent = `${config.trainerType} · ${dsName} · ${nLabels} label${nLabels === 1 ? '' : 's'}`;
        item.appendChild(metaLine);

        item.addEventListener('click', () => _showForm(config));
        listBody.appendChild(item);
      }

      _updateListSelection(_selectedUuid);
    }

    // ── Navigation guard ──────────────────────────────────────────────────
    const guardSignal = _navGuardController.signal;
    document.addEventListener('click', async (e) => {
      const btn = e.target.closest('.workspace-tab-button, .main-tab-button');
      if (!btn) return;
      const targetTab = btn.dataset.tab;
      if (targetTab === 'tab-training-management' || targetTab === 'tab-configuration') return;
      if (!_barHandle?.getDirty()) return;

      e.stopPropagation();
      const result = await showUnsavedConfigChangesModal();
      if (result === 'cancel') return;
      if (result === 'save') {
        const ok = await _currentDoSave?.();
        if (!ok) return;
      }
      if (guardSignal.aborted) return;
      _barHandle?.setDirty(false);
      btn.click();
    }, { capture: true, signal: guardSignal });

    await refreshList();
    await _showForm(null, { force: true });
  } catch (err) {
    const errMsg = document.createElement('p');
    errMsg.style.color = 'red';
    errMsg.textContent = `Error loading training configurations: ${err.message}`;
    container.appendChild(errMsg);
  } finally {
    _rendering = false;
  }
}
