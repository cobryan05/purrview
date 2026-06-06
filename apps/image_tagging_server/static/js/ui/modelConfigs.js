import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { createLabelList } from '/app-static/js/ui/labelList.js';
import {
  createModelConfig,
  deleteModelConfig,
  fetchModelConfigs,
  fetchModelTypenames,
  updateModelConfig,
} from '/app-static/js/shared/api/modelConfigs.js';
import { toast } from '/app-static/js/ui/utils/index.js';

let _rendering = false;

// -------------------------------------------------------
// Node ↔ UI state conversion
// -------------------------------------------------------

/**
 * Walk up the label tree to find the nearest ancestor with a submodel.
 * Falls back to rootTypename.
 */
function inheritedTypename(uuid, labelsMap, submodels, rootTypename) {
  let p = labelsMap.get(uuid)?.parent_uuid ?? null;
  while (p !== null) {
    if (submodels.has(p)) return submodels.get(p);
    p = labelsMap.get(p)?.parent_uuid ?? null;
  }
  return rootTypename;
}

/**
 * Parse a flat nodes array from the API into form UI state.
 * Processes ancestors first so inheritance is resolved correctly.
 *
 * @returns {{ rootTypename: string, selected: Set<string>, submodels: Map<string,string> }}
 */
function parseNodes(nodes, labelsMap) {
  let rootTypename = '';
  const nonRoot = [];

  for (const n of (nodes ?? [])) {
    if (n.label_uuid === null) rootTypename = n.model_typename;
    else nonRoot.push(n);
  }

  // Sort by depth so ancestor submodels are registered before descendants.
  const depthOf = (uuid) => {
    let d = 0, p = labelsMap.get(uuid)?.parent_uuid ?? null;
    while (p) { d++; p = labelsMap.get(p)?.parent_uuid ?? null; }
    return d;
  };
  nonRoot.sort((a, b) => depthOf(a.label_uuid) - depthOf(b.label_uuid));

  const submodels = new Map();
  const selected = new Set();

  for (const { label_uuid, model_typename } of nonRoot) {
    selected.add(label_uuid);
    const inh = inheritedTypename(label_uuid, labelsMap, submodels, rootTypename);
    if (model_typename !== inh) {
      submodels.set(label_uuid, model_typename);
    }
  }

  return { rootTypename, selected, submodels };
}

/**
 * Build the nodes array to send to the API.
 * Union of selected labels and submodel-boundary labels.
 */
function computeNodes(selected, submodels, rootTypename, labelsMap) {
  const nodes = [{ label_uuid: null, model_typename: rootTypename }];
  const allUuids = new Set([...selected, ...submodels.keys()]);
  for (const uuid of allUuids) {
    const typename = submodels.has(uuid)
      ? submodels.get(uuid)
      : inheritedTypename(uuid, labelsMap, submodels, rootTypename);
    nodes.push({ label_uuid: uuid, model_typename: typename });
  }
  return nodes;
}

// -------------------------------------------------------
// Submodel config panels (rendered below the label tree)
// -------------------------------------------------------

/**
 * Renders the "Sub Model Configurations" section, one card per enabled submodel.
 * Uses no explicit background — inherits theming from the page.
 */
function renderSubmodelPanels(container, submodels, labelsMap, typenames, onTypenameChange) {
  container.innerHTML = '';
  if (submodels.size === 0) return;

  const hdr = document.createElement('h4');
  hdr.style.marginBottom = '8px';
  hdr.textContent = 'Sub Model Configurations';
  container.appendChild(hdr);

  for (const [uuid, typename] of submodels) {
    const labelName = labelsMap.get(uuid)?.name ?? uuid;
    const labelColor = labelsMap.get(uuid)?.color ?? '#888';

    const card = document.createElement('div');
    card.style.cssText = [
      'display: flex', 'align-items: center', 'gap: 10px',
      'padding: 8px 12px', 'margin-bottom: 8px',
      'border: 1px solid var(--pico-border-color)',
      'border-radius: 5px',
    ].join('; ');

    // Color swatch
    const swatch = document.createElement('span');
    swatch.style.cssText = [
      `background: ${labelColor}`,
      'width: 12px', 'height: 12px', 'border-radius: 50%',
      'flex-shrink: 0', 'display: inline-block',
    ].join('; ');

    const nameSpan = document.createElement('span');
    nameSpan.style.cssText = 'font-weight: bold; font-size: 0.9em; flex: 1;';
    nameSpan.textContent = labelName;

    const lbl = document.createElement('label');
    lbl.textContent = 'Model type:';
    lbl.style.cssText = 'font-size: 0.85em; white-space: nowrap; margin-bottom: 0;';

    const sel = document.createElement('select');
    sel.style.cssText = 'margin-bottom: 0;';
    typenames.forEach(tn => {
      const o = document.createElement('option');
      o.value = tn; o.textContent = tn;
      if (tn === typename) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener('change', () => onTypenameChange(uuid, sel.value));

    card.appendChild(swatch);
    card.appendChild(nameSpan);
    card.appendChild(lbl);
    card.appendChild(sel);
    container.appendChild(card);
  }
}

// -------------------------------------------------------
// Config form (create / edit)
// -------------------------------------------------------

function renderConfigForm({ config, typenames, labelsMap, onSave, onCancel, onDirtyChange }) {
  const form = document.createElement('div');

  let _dirty = false;
  function _markDirty() {
    if (!_dirty) { _dirty = true; saveBtn.disabled = false; onDirtyChange?.(() => _dirty); }
  }

  // ---- Name + Description -------------------------------------------------
  function makeInputRow(labelText, input) {
    const row = document.createElement('div');
    row.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 6px;';
    const lbl = document.createElement('label');
    lbl.textContent = labelText;
    lbl.style.cssText = 'width: 100px; flex-shrink: 0; font-size: 0.9em; margin-bottom: 0;';
    input.style.cssText = 'flex: 1; min-width: 0; margin-bottom: 0;';
    row.appendChild(lbl); row.appendChild(input);
    return row;
  }

  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.value = config?.name ?? '';
  nameInput.placeholder = 'Required';
  nameInput.addEventListener('input', _markDirty);
  form.appendChild(makeInputRow('Name:', nameInput));

  const descInput = document.createElement('input');
  descInput.type = 'text';
  descInput.value = config?.description ?? '';
  descInput.placeholder = 'Optional';
  descInput.addEventListener('input', _markDirty);
  form.appendChild(makeInputRow('Description:', descInput));

  // ---- Root model type ----------------------------------------------------
  const { rootTypename: initRoot, selected: initSelected, submodels: initSubmodels } =
    parseNodes(config?.nodes, labelsMap);
  let rootTypename = initRoot || typenames[0] || '';

  const rootRow = document.createElement('div');
  rootRow.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 14px;';
  const rootLbl = document.createElement('label');
  rootLbl.textContent = 'Root model type:';
  rootLbl.style.cssText = 'white-space: nowrap; font-size: 0.9em; margin-bottom: 0;';
  const rootSel = document.createElement('select');
  rootSel.style.cssText = 'margin-bottom: 0;';
  typenames.forEach(tn => {
    const o = document.createElement('option');
    o.value = tn; o.textContent = tn;
    if (tn === rootTypename) o.selected = true;
    rootSel.appendChild(o);
  });
  rootSel.addEventListener('change', () => { rootTypename = rootSel.value; _markDirty(); });
  rootRow.appendChild(rootLbl);
  rootRow.appendChild(rootSel);
  form.appendChild(rootRow);

  // ---- Label tree (shared component) -------------------------------------
  const treeSection = document.createElement('div');
  treeSection.style.cssText = [
    'border: 1px solid var(--pico-border-color)',
    'border-radius: 5px',
    'padding: 8px',
    'margin-bottom: 12px',
    'max-height: 380px',
    'overflow-y: auto',
  ].join('; ');
  form.appendChild(treeSection);

  treeSection.addEventListener('change', _markDirty);

  // ---- Submodel panels (populated dynamically) ---------------------------
  const submodelsContainer = document.createElement('div');
  submodelsContainer.style.marginBottom = '12px';
  form.appendChild(submodelsContainer);

  // Track live submodel state (updated by the tree callbacks).
  let currentSubmodels = new Map(initSubmodels);

  function refreshPanels(submodels) {
    currentSubmodels = submodels;
    renderSubmodelPanels(
      submodelsContainer, submodels, labelsMap, typenames,
      (uuid, newTypename) => { currentSubmodels.set(uuid, newTypename); },
    );
  }

  // Mount the label tree via the shared component.
  let treeHandle = null;
  createLabelList(treeSection, {
    mode: 'model-config',
    labelsMap,
    initialSelected: initSelected,
    initialSubmodels: initSubmodels,
    typenames,
    onSubmodelsChanged: (submodels) => { refreshPanels(submodels); _markDirty(); },
  }).then(handle => {
    treeHandle = handle;
    refreshPanels(initSubmodels); // render initial panels
  });

  // ---- Buttons ------------------------------------------------------------
  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'display: flex; gap: 8px;';

  const saveBtn = document.createElement('button');
  saveBtn.type = 'button';
  saveBtn.textContent = config ? 'Save' : 'Create';
  saveBtn.disabled = true;
  saveBtn.addEventListener('click', async () => {
    const name = nameInput.value.trim();
    if (!name) { toast('Name is required', 3000, 'error'); return; }
    if (!rootTypename) { toast('Root model type is required', 3000, 'error'); return; }
    saveBtn.disabled = true;
    try {
      const selected = treeHandle?.getSelected() ?? new Set();
      await onSave({
        name,
        description: descInput.value.trim(),
        nodes: computeNodes(selected, currentSubmodels, rootTypename, labelsMap),
      });
    } catch (err) {
      toast(err.message, 5000, 'error');
      saveBtn.disabled = false;
    }
  });
  btnRow.appendChild(saveBtn);

  if (onCancel) {
    const cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.addEventListener('click', () => { treeHandle?.destroy(); onCancel(); });
    btnRow.appendChild(cancelBtn);
  }

  form.appendChild(btnRow);
  return form;
}

// -------------------------------------------------------
// Public entry point
// -------------------------------------------------------

export async function renderModelConfigManager({ target = 'model-config-box' } = {}) {
  if (_rendering) return;
  _rendering = true;

  const container = document.getElementById(target);
  if (!container) { _rendering = false; return; }
  container.innerHTML = '';

  try {
    const panel = document.createElement('div');
    panel.className = 'config-panel';
    container.appendChild(panel);

    const leftCol = document.createElement('div');
    leftCol.className = 'config-column';
    panel.appendChild(leftCol);

    const rightCol = document.createElement('div');
    rightCol.className = 'config-column';
    panel.appendChild(rightCol);

    let _selectedUuid = null;
    let _getDirtyState = () => false;

    function _confirmIfDirty() {
      if (!_getDirtyState()) return true;
      return window.confirm('You have unsaved changes. Discard them?');
    }

    function _updateListSelection(uuid) {
      for (const row of leftCol.querySelectorAll('[data-config-uuid]')) {
        const sel = row.dataset.configUuid === uuid;
        row.style.background = sel ? 'var(--pico-primary-background, #eef)' : '';
        row.style.borderColor = sel ? 'var(--pico-primary-border, #aad)' : 'transparent';
      }
    }

    async function showEdit(config, { force = false } = {}) {
      if (!force && !_confirmIfDirty()) return;
      _getDirtyState = () => false;
      _selectedUuid = config.uuid;
      const [labelsMap, typenames] = await Promise.all([fetchLabels(), fetchModelTypenames()]);
      rightCol.innerHTML = '';

      const h4 = document.createElement('h4');
      h4.style.marginBottom = '8px';
      h4.textContent = config.name;
      rightCol.appendChild(h4);

      rightCol.appendChild(renderConfigForm({
        config, typenames, labelsMap,
        onDirtyChange: (getter) => { _getDirtyState = getter; },
        onSave: async ({ name, description, nodes }) => {
          const updated = await updateModelConfig(config.uuid, { name, description, nodes });
          _getDirtyState = () => false;
          toast('Saved', 3000);
          const nameEl = leftCol.querySelector(`[data-config-uuid="${config.uuid}"] .mc-name`);
          if (nameEl) nameEl.textContent = updated.name;
          await showEdit(updated, { force: true });
        },
      }));

      const deleteBtn = document.createElement('button');
      deleteBtn.type = 'button';
      deleteBtn.textContent = '🗑️ Delete';
      deleteBtn.style.cssText = 'margin-top:8px;';
      deleteBtn.addEventListener('click', async () => {
        if (!window.confirm(`Delete model config "${config.name}"?`)) return;
        deleteBtn.disabled = true;
        try {
          await deleteModelConfig(config.uuid);
          toast('Deleted', 3000);
          await refreshList();
          await showCreate({ force: true });
        } catch (err) {
          toast(err.message, 5000, 'error');
          deleteBtn.disabled = false;
        }
      });
      rightCol.appendChild(deleteBtn);

      _updateListSelection(config.uuid);
    }

    async function showCreate({ force = false } = {}) {
      if (!force && !_confirmIfDirty()) return;
      _getDirtyState = () => false;
      _selectedUuid = null;
      const [labelsMap, typenames] = await Promise.all([fetchLabels(), fetchModelTypenames()]);
      rightCol.innerHTML = '';

      const h4 = document.createElement('h4');
      h4.style.marginBottom = '8px';
      h4.textContent = 'New Model Configuration';
      rightCol.appendChild(h4);

      rightCol.appendChild(renderConfigForm({
        config: null, typenames, labelsMap,
        onDirtyChange: (getter) => { _getDirtyState = getter; },
        onSave: async ({ name, description, nodes }) => {
          const created = await createModelConfig({ name, description, nodes });
          _getDirtyState = () => false;
          toast(`Created "${created.name}"`, 3000);
          await refreshList();
          await showEdit(created, { force: true });
        },
      }));

      _updateListSelection(null);
    }

    async function refreshList() {
      leftCol.innerHTML = '';

      const h4 = document.createElement('h4');
      h4.style.marginBottom = '8px';
      h4.textContent = 'Model Configurations';
      leftCol.appendChild(h4);

      const configsMap = await fetchModelConfigs();
      const listEl = document.createElement('div');
      leftCol.appendChild(listEl);

      if (configsMap.size === 0) {
        const empty = document.createElement('p');
        empty.style.cssText = 'color:var(--pico-muted-color,#888); font-style:italic; font-size:0.85em;';
        empty.textContent = 'No configurations yet.';
        listEl.appendChild(empty);
      }

      for (const config of configsMap.values()) {
        const row = document.createElement('div');
        row.style.cssText = 'cursor:pointer; padding:6px 8px; margin-bottom:4px; border-radius:4px; border:1px solid transparent; background:var(--pico-card-background-color);';
        row.dataset.configUuid = config.uuid;

        const nameLine = document.createElement('div');
        nameLine.className = 'mc-name';
        nameLine.style.fontWeight = 'bold';
        nameLine.textContent = config.name;
        row.appendChild(nameLine);

        if (config.description) {
          const descLine = document.createElement('div');
          descLine.style.cssText = 'font-size:0.82em; color:var(--pico-muted-color,#666); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:220px;';
          descLine.textContent = config.description;
          row.appendChild(descLine);
        }

        const metaLine = document.createElement('div');
        metaLine.style.cssText = 'font-size:0.75em; color:var(--pico-muted-color,#999);';
        const n = config.nodes?.length ?? 0;
        metaLine.textContent = `${n} assignment${n === 1 ? '' : 's'}`;
        row.appendChild(metaLine);

        row.addEventListener('mouseenter', () => {
          if (row.dataset.configUuid !== _selectedUuid) row.style.background = 'var(--pico-card-sectioning-background-color, #f5f5f5)';
        });
        row.addEventListener('mouseleave', () => {
          if (row.dataset.configUuid !== _selectedUuid) row.style.background = 'var(--pico-card-background-color)';
        });
        row.addEventListener('click', () => showEdit(config));
        listEl.appendChild(row);
      }

      const newBtn = document.createElement('button');
      newBtn.type = 'button';
      newBtn.textContent = '+ New Configuration';
      newBtn.style.cssText = 'margin-top:8px; width:100%;';
      newBtn.addEventListener('click', () => showCreate());
      leftCol.appendChild(newBtn);

      _updateListSelection(_selectedUuid);
    }

    await refreshList();
    await showCreate({ force: true });
  } catch (err) {
    const errMsg = document.createElement('p');
    errMsg.style.color = 'red';
    errMsg.textContent = `Error loading model configurations: ${err.message}`;
    container.appendChild(errMsg);
  } finally {
    _rendering = false;
  }
}
