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

import { fetchDatasets } from '/app-static/js/shared/api/datasets.js';
import { fetchModelConfigs } from '/app-static/js/shared/api/modelConfigs.js';
import {
  fetchTrainingConfigs,
  createTrainingConfig,
  updateTrainingConfig,
  deleteTrainingConfig,
  fetchTrainingConfigReadiness,
  fetchTrainingConfigLabelStats,
  fetchTrainingConfigNeedsReview,
  clearTrainingConfigsCache,
} from '/app-static/js/shared/api/trainingConfigs.js';
import { makeFieldRow, buildPanelEditorBar, toast } from '/app-static/js/ui/utils/index.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { showUnsavedConfigChangesModal } from '/app-static/js/ui/utils/confirmModal.js';

let _rendering = false;
let _navGuardController = null;

// ── Form ─────────────────────────────────────────────────────────────────────

function renderTrainingConfigForm({ config, datasetsMap, modelConfigsMap, onDirtyChange }) {
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

  // ── Model config → node selector (2-step) ────────────────────────────────
  const hr1 = document.createElement('hr');
  hr1.style.cssText = 'margin: 0.6em 0; opacity: 0.4;';
  el.appendChild(hr1);

  const nodeHdr = document.createElement('p');
  nodeHdr.className = 'panel-section-label';
  nodeHdr.style.marginTop = '0';
  nodeHdr.textContent = 'Target model node';
  el.appendChild(nodeHdr);

  // Derive which model config owns the stored node (if any).
  const initNodeUuid = config?.modelConfigNodeUuid ?? '';
  let initConfigUuid = '';
  if (initNodeUuid) {
    for (const [uuid, mc] of modelConfigsMap) {
      if (mc.nodes?.some(n => n.uuid === initNodeUuid)) { initConfigUuid = uuid; break; }
    }
  }

  // Step 1: pick the model config
  const configSel = document.createElement('select');
  configSel.style.cssText = 'margin-bottom: 0; width: auto; flex: none;';
  const blankConfigOpt = document.createElement('option');
  blankConfigOpt.value = '';
  blankConfigOpt.textContent = '— select a config —';
  configSel.appendChild(blankConfigOpt);
  for (const [uuid, mc] of modelConfigsMap) {
    const o = document.createElement('option');
    o.value = uuid;
    o.textContent = mc.name;
    if (uuid === initConfigUuid) o.selected = true;
    configSel.appendChild(o);
  }
  configSel.addEventListener('change', () => { refreshNodeSel(); _markDirty(); });
  el.appendChild(makeFieldRow('Model config', configSel));

  // Step 2: pick the node within that config
  const nodeSel = document.createElement('select');
  nodeSel.style.cssText = 'margin-bottom: 0; width: auto; flex: none;';

  function nodeOptionLabel(node, modelConfigsMap) {
    const mc = modelConfigsMap.get(configSel.value);
    if (!node.labelUuid) {
      return `Root — ${node.modelType ?? node.modelTypename ?? '?'} / ${node.architecture ?? ''}`.trim().replace(/ \/$/, '');
    }
    // We don't have labelsMap here, so use labelUuid as fallback (will improve post-backend update)
    const labelHint = node.labelName ?? node.labelUuid;
    return `${labelHint} — ${node.modelType ?? node.modelTypename ?? '?'} / ${node.architecture ?? ''}`.trim().replace(/ \/$/, '');
  }

  function refreshNodeSel() {
    nodeSel.innerHTML = '';
    const blankOpt = document.createElement('option');
    blankOpt.value = '';
    blankOpt.textContent = '— select a node —';
    nodeSel.appendChild(blankOpt);

    const mc = modelConfigsMap.get(configSel.value);
    for (const node of (mc?.nodes ?? [])) {
      const o = document.createElement('option');
      o.value = node.uuid;
      o.textContent = nodeOptionLabel(node, modelConfigsMap);
      if (node.uuid === config?.modelConfigNodeUuid) o.selected = true;
      nodeSel.appendChild(o);
    }
    nodeSel.disabled = (mc?.nodes?.length ?? 0) === 0;
  }

  refreshNodeSel();
  nodeSel.addEventListener('change', _markDirty);
  el.appendChild(makeFieldRow('Model node', nodeSel));

  // ── Datasets ──────────────────────────────────────────────────────────────
  const hr2 = document.createElement('hr');
  hr2.style.cssText = 'margin: 0.6em 0; opacity: 0.4;';
  el.appendChild(hr2);

  const dsHdr = document.createElement('p');
  dsHdr.className = 'panel-section-label';
  dsHdr.style.marginTop = '0';
  dsHdr.textContent = 'Datasets';
  el.appendChild(dsHdr);

  const currentDatasetUuids = new Set(config?.datasetUuids ?? []);
  const datasetCheckboxes = new Map();
  const datasetBox = document.createElement('div');
  datasetBox.className = 'panel-tree-box';
  if (datasetsMap.size === 0) {
    const none = document.createElement('p');
    none.style.cssText = 'font-size: 0.85em; color: var(--pico-muted-color, #888); margin: 0; padding: 2px 4px;';
    none.textContent = 'No datasets defined yet.';
    datasetBox.appendChild(none);
  }
  for (const [uuid, ds] of datasetsMap) {
    const row = document.createElement('label');
    row.style.cssText = 'display: flex; align-items: center; gap: 0.4em; padding: 2px 4px; cursor: pointer; font-size: 0.88em; user-select: none;';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = uuid;
    cb.checked = currentDatasetUuids.has(uuid);
    cb.style.cssText = 'margin: 0; flex-shrink: 0;';
    cb.addEventListener('change', _markDirty);
    datasetCheckboxes.set(uuid, cb);
    row.appendChild(cb);
    const nameSpan = document.createElement('span');
    nameSpan.textContent = ds.name;
    row.appendChild(nameSpan);
    datasetBox.appendChild(row);
  }
  const datasetRow = makeFieldRow('Select ≥1', datasetBox);
  datasetRow.style.alignItems = 'flex-start';
  el.appendChild(datasetRow);

  // ── Hyperparameters ───────────────────────────────────────────────────────
  const hr3 = document.createElement('hr');
  hr3.style.cssText = 'margin: 0.6em 0; opacity: 0.4;';
  el.appendChild(hr3);

  const hyperText = document.createElement('textarea');
  hyperText.rows = 4;
  hyperText.style.cssText = 'font-family: monospace; font-size: 0.8em; width: 100%; box-sizing: border-box;';
  hyperText.placeholder = '{}';
  hyperText.value = JSON.stringify(config?.hyperparamsJson ?? {}, null, 2);
  hyperText.addEventListener('input', _markDirty);
  el.appendChild(makeFieldRow('Hyperparameters (JSON)', hyperText));

  return {
    el,
    getSaveData() {
      const datasetUuids = [...datasetCheckboxes.entries()]
        .filter(([, cb]) => cb.checked)
        .map(([uuid]) => uuid);
      let hyperparamsJson = {};
      try { hyperparamsJson = JSON.parse(hyperText.value || '{}'); } catch {}
      return {
        name: nameInput.value.trim(),
        description: descInput.value.trim(),
        modelConfigNodeUuid: nodeSel.value || null,
        datasetUuids,
        hyperparamsJson,
      };
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
  reviewBtn.addEventListener('click', async () => {
    const { showReviewModal } = await import('/app-static/js/app/review/main.js');
    await showReviewModal({
      title: `Needs Review — ${config.name}`,
      fetchImages: async ({ afterUuid = null, afterSortValue = null, limit }) => {
        return fetchTrainingConfigNeedsReview({ uuid: config.uuid, afterUuid, afterSortValue, limit });
      },
    });
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

function renderLabelStatsPanel(config, labelsMap) {
  const wrap = document.createElement('div');
  wrap.style.cssText = 'margin-top: 0.75em; padding: 0.6em 0.75em; border: 1px solid var(--pico-border-color, #444); border-radius: 4px; font-size: 0.85em;';

  const title = document.createElement('div');
  title.style.cssText = 'font-weight: bold; margin-bottom: 0.5em;';
  title.textContent = 'Dataset coverage by label';
  wrap.appendChild(title);

  const status = document.createElement('div');
  status.textContent = 'Loading…';
  wrap.appendChild(status);

  fetchTrainingConfigLabelStats(config.uuid).then(({ labelStats, totalImages }) => {
    status.remove();

    if (labelStats.length === 0) {
      const none = document.createElement('div');
      none.style.cssText = 'color: var(--pico-muted-color, #888); font-style: italic;';
      none.textContent = 'No relevant labels found for this model config / dataset combination.';
      wrap.appendChild(none);
      return;
    }

    const imageLabels = labelStats.filter(s => s.labelScope === 'image');
    const objectLabels = labelStats.filter(s => s.labelScope === 'object');

    function _labelName(uuid) {
      return labelsMap?.get(uuid)?.name ?? uuid;
    }

    function _makeTable(stats, columns) {
      const table = document.createElement('table');
      table.style.cssText = 'width: 100%; border-collapse: collapse; font-size: 0.9em;';

      const thead = document.createElement('thead');
      const headerRow = document.createElement('tr');
      const labelTh = document.createElement('th');
      labelTh.textContent = 'Label';
      labelTh.style.cssText = 'text-align: left; padding: 2px 6px; border-bottom: 1px solid var(--pico-border-color, #444);';
      headerRow.appendChild(labelTh);
      for (const col of columns) {
        const th = document.createElement('th');
        th.textContent = col.header;
        th.title = col.title ?? '';
        th.style.cssText = 'text-align: right; padding: 2px 6px; border-bottom: 1px solid var(--pico-border-color, #444); white-space: nowrap;';
        headerRow.appendChild(th);
      }
      thead.appendChild(headerRow);
      table.appendChild(thead);

      const tbody = document.createElement('tbody');
      for (const stat of stats) {
        const tr = document.createElement('tr');

        const nameTd = document.createElement('td');
        nameTd.style.cssText = 'padding: 2px 6px;';
        const dot = labelsMap?.get(stat.labelUuid)?.color;
        if (dot) {
          const dotEl = document.createElement('span');
          dotEl.style.cssText = `display:inline-block;width:7px;height:7px;border-radius:50%;background:${dot};margin-right:4px;vertical-align:middle;`;
          nameTd.appendChild(dotEl);
        }
        nameTd.appendChild(document.createTextNode(_labelName(stat.labelUuid)));
        tr.appendChild(nameTd);

        for (const col of columns) {
          const td = document.createElement('td');
          td.style.cssText = 'text-align: right; padding: 2px 6px;';
          const val = stat[col.key] ?? 0;
          td.textContent = val.toLocaleString();
          if (col.warnIf?.(stat)) td.style.color = '#f90';
          if (col.goodIf?.(stat)) td.style.color = '#4c4';
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      return table;
    }

    const totalNote = document.createElement('div');
    totalNote.style.cssText = 'color: var(--pico-muted-color, #888); margin-bottom: 0.5em; font-size: 0.9em;';
    totalNote.textContent = `${totalImages.toLocaleString()} image${totalImages === 1 ? '' : 's'} across selected dataset${(config.datasetUuids?.length ?? 0) === 1 ? '' : 's'}`;
    wrap.appendChild(totalNote);

    if (imageLabels.length > 0) {
      const sectionHead = document.createElement('div');
      sectionHead.style.cssText = 'font-weight: 600; margin: 0.5em 0 0.25em; font-size: 0.9em;';
      sectionHead.textContent = 'Image labels';
      wrap.appendChild(sectionHead);
      wrap.appendChild(_makeTable(imageLabels, [
        { header: 'Verified imgs', key: 'verifiedImageCount', title: 'Images with this label explicitly set to Present or Absent', goodIf: s => s.unverifiedImageCount === 0 },
        { header: 'Missing verif.', key: 'unverifiedImageCount', title: 'Images missing verification for this label', warnIf: s => s.unverifiedImageCount > 0 },
      ]));
    }

    if (objectLabels.length > 0) {
      const sectionHead = document.createElement('div');
      sectionHead.style.cssText = 'font-weight: 600; margin: 0.6em 0 0.25em; font-size: 0.9em;';
      sectionHead.textContent = 'Object labels';
      wrap.appendChild(sectionHead);
      wrap.appendChild(_makeTable(objectLabels, [
        { header: 'Verified boxes', key: 'verifiedBboxCount', title: 'Bounding boxes marked Verified', goodIf: s => s.unverifiedBboxCount === 0 && s.verifiedBboxCount > 0 },
        { header: 'Unverified boxes', key: 'unverifiedBboxCount', title: 'Bounding boxes still marked Unverified', warnIf: s => s.unverifiedBboxCount > 0 },
        { header: 'Imgs all clear', key: 'fullyVerifiedImageCount', title: 'Images where every box for this label is verified', goodIf: s => s.imagesMissingVerificationCount === 0 && s.fullyVerifiedImageCount > 0 },
        { header: 'Imgs need work', key: 'imagesMissingVerificationCount', title: 'Images with at least one unverified box for this label', warnIf: s => s.imagesMissingVerificationCount > 0 },
      ]));
    }
  }).catch(err => {
    status.textContent = `Could not load label stats: ${err.message}`;
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

        const [datasetsMap, modelConfigsMap] = await Promise.all([
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
            if (!data.modelConfigNodeUuid) { toast('A model config node is required', 3000, 'error'); return; }
            if (!data.datasetUuids?.length) { toast('At least one dataset is required', 3000, 'error'); return; }
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
          datasetsMap,
          modelConfigsMap,
          onDirtyChange: setDirty,
        });
        _currentFormHandle = formHandle;
        editorBody.appendChild(formHandle.el);

        // Readiness panel (only for existing configs with datasets and a node)
        if (config?.uuid && (config.datasetUuids?.length ?? 0) > 0 && config.modelConfigNodeUuid) {
          editorBody.appendChild(renderReadinessPanel(config));
        }

        // Label stats panel (only for existing configs with datasets selected)
        if (config?.uuid && (config.datasetUuids?.length ?? 0) > 0) {
          editorBody.appendChild(renderLabelStatsPanel(config, new Map()));
        }

        _currentDoSave = async () => {
          const data = formHandle.getSaveData();
          if (!data.name) { toast('Name is required', 3000, 'error'); return false; }
          if (!data.modelConfigNodeUuid) { toast('A model config node is required', 3000, 'error'); return false; }
          if (!data.datasetUuids?.length) { toast('At least one dataset is required', 3000, 'error'); return false; }
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
        const dsNames = (config.datasetUuids ?? []).map(u => datasetsMap.get(u)?.name ?? u).join(', ') || '—';
        metaLine.textContent = dsNames;
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
