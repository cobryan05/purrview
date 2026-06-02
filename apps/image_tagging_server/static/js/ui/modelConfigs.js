import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { createLabelList } from '/app-static/js/ui/labelList.js';
import {
  clearModelConfigsCache,
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

function renderConfigForm({ config, typenames, labelsMap, onSave, onCancel }) {
  const form = document.createElement('div');

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
  form.appendChild(makeInputRow('Name:', nameInput));

  const descInput = document.createElement('input');
  descInput.type = 'text';
  descInput.value = config?.description ?? '';
  descInput.placeholder = 'Optional';
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
  rootSel.addEventListener('change', () => { rootTypename = rootSel.value; });
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
    onSubmodelsChanged: (submodels) => refreshPanels(submodels),
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
// Config card (view mode + edit toggle)
// -------------------------------------------------------

function renderConfigCard({ config, labelsMap, typenames, onRefresh }) {
  const card = document.createElement('div');
  card.style.cssText = [
    'border: 1px solid var(--pico-border-color)',
    'border-radius: 6px',
    'padding: 12px',
    'margin-bottom: 8px',
  ].join('; ');

  function showView() {
    card.innerHTML = '';

    const header = document.createElement('div');
    header.style.cssText = 'display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;';

    const left = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = config.name;
    left.appendChild(title);
    if (config.description) {
      const desc = document.createElement('div');
      desc.style.cssText = 'font-size: 0.85em; color: var(--pico-muted-color); margin-top: 2px;';
      desc.textContent = config.description;
      left.appendChild(desc);
    }

    const btns = document.createElement('div');
    btns.style.cssText = 'display: flex; gap: 6px; margin-left: 12px;';

    const editBtn = document.createElement('button');
    editBtn.type = 'button';
    editBtn.textContent = '✏️ Edit';
    editBtn.style.cssText = 'font-size: 0.8em; padding: 2px 8px;';
    editBtn.addEventListener('click', showEdit);

    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.textContent = '🗑️ Delete';
    deleteBtn.style.cssText = 'font-size: 0.8em; padding: 2px 8px;';
    deleteBtn.addEventListener('click', async () => {
      if (!window.confirm(`Delete model config "${config.name}"?`)) return;
      try { await deleteModelConfig(config.uuid); onRefresh(); }
      catch (err) { toast(err.message, 5000, 'error'); }
    });

    btns.appendChild(editBtn);
    btns.appendChild(deleteBtn);
    header.appendChild(left);
    header.appendChild(btns);
    card.appendChild(header);

    if (config.nodes.length === 0) {
      const empty = document.createElement('div');
      empty.style.cssText = 'font-size: 0.85em; color: var(--pico-muted-color);';
      empty.textContent = 'No assignments defined.';
      card.appendChild(empty);
    } else {
      const table = document.createElement('div');
      table.style.cssText = 'font-size: 0.85em; display: flex; flex-direction: column; gap: 3px;';
      for (const node of config.nodes) {
        const row = document.createElement('div');
        row.style.cssText = 'display: flex; align-items: center; gap: 8px;';

        // Color swatch
        if (node.label_uuid) {
          const color = labelsMap.get(node.label_uuid)?.color ?? '#888';
          const swatch = document.createElement('span');
          swatch.style.cssText = `background:${color}; width:8px; height:8px; border-radius:50%; flex-shrink:0; display:inline-block;`;
          row.appendChild(swatch);
        }

        const lbl = document.createElement('span');
        lbl.style.cssText = 'color: var(--pico-muted-color); min-width: 130px;';
        lbl.textContent = node.label_uuid
          ? (labelsMap.get(node.label_uuid)?.name ?? '(unknown)')
          : 'Root';
        const arrow = document.createElement('span'); arrow.textContent = '→';
        const type = document.createElement('span'); type.textContent = node.model_typename;
        row.appendChild(lbl); row.appendChild(arrow); row.appendChild(type);
        table.appendChild(row);
      }
      card.appendChild(table);
    }
  }

  function showEdit() {
    card.innerHTML = '';
    card.appendChild(renderConfigForm({
      config, typenames, labelsMap,
      onSave: async ({ name, description, nodes }) => {
        await updateModelConfig(config.uuid, { name, description, nodes });
        onRefresh();
      },
      onCancel: showView,
    }));
  }

  showView();
  return card;
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
    const [configsMap, typenames, labelsMap] = await Promise.all([
      fetchModelConfigs(),
      fetchModelTypenames(),
      fetchLabels(),
    ]);

    const refresh = () => { clearModelConfigsCache(); renderModelConfigManager({ target }); };

    const listHeader = document.createElement('h3');
    listHeader.textContent = 'Model Configurations';
    container.appendChild(listHeader);

    if (configsMap.size === 0) {
      const empty = document.createElement('p');
      empty.style.color = 'var(--pico-muted-color)';
      empty.textContent = 'No model configurations defined yet.';
      container.appendChild(empty);
    } else {
      for (const config of configsMap.values()) {
        container.appendChild(renderConfigCard({ config, typenames, labelsMap, onRefresh: refresh }));
      }
    }

    container.appendChild(document.createElement('hr'));

    const createHeader = document.createElement('h3');
    createHeader.textContent = 'Create Model Configuration';
    container.appendChild(createHeader);

    container.appendChild(renderConfigForm({
      config: null, typenames, labelsMap,
      onSave: async ({ name, description, nodes }) => {
        await createModelConfig({ name, description, nodes });
        refresh();
      },
    }));
  } catch (err) {
    const errMsg = document.createElement('p');
    errMsg.style.color = 'red';
    errMsg.textContent = `Error loading model configurations: ${err.message}`;
    container.appendChild(errMsg);
  } finally {
    _rendering = false;
  }
}
