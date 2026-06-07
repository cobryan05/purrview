import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { createLabelList } from '/app-static/js/ui/labelList.js';
import {
  createModelConfig,
  deleteModelConfig,
  fetchModelConfigs,
  fetchModelTypenames,
  updateModelConfig,
} from '/app-static/js/shared/api/modelConfigs.js';
import { showUnsavedConfigChangesModal } from '/app-static/js/ui/utils/confirmModal.js';
import { toast, makeFieldRow, buildPanelEditorBar } from '/app-static/js/ui/utils/index.js';

let _rendering = false;
let _navGuardController = null; // AbortController for the tab-navigation guard

// -------------------------------------------------------
// Node ↔ UI state conversion
// -------------------------------------------------------

/**
 * Walk up the label tree to find the nearest ancestor with a submodel.
 * Falls back to rootTypename.
 */
function inheritedTypename(uuid, labelsMap, submodels, rootTypename) {
  let p = labelsMap.get(uuid)?.parentUuid ?? null;
  while (p !== null) {
    if (submodels.has(p)) return submodels.get(p);
    p = labelsMap.get(p)?.parentUuid ?? null;
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
    if (n.labelUuid === null) rootTypename = n.modelTypename;
    else nonRoot.push(n);
  }

  // Sort by depth so ancestor submodels are registered before descendants.
  const depthOf = (uuid) => {
    let d = 0, p = labelsMap.get(uuid)?.parentUuid ?? null;
    while (p) { d++; p = labelsMap.get(p)?.parentUuid ?? null; }
    return d;
  };
  nonRoot.sort((a, b) => depthOf(a.labelUuid) - depthOf(b.labelUuid));

  const submodels = new Map();
  const selected = new Set();

  for (const { labelUuid, modelTypename } of nonRoot) {
    selected.add(labelUuid);
    const inh = inheritedTypename(labelUuid, labelsMap, submodels, rootTypename);
    if (modelTypename !== inh) {
      submodels.set(labelUuid, modelTypename);
    }
  }

  return { rootTypename, selected, submodels };
}

/**
 * Build the nodes array to send to the API.
 * Union of selected labels and submodel-boundary labels.
 */
function computeNodes(selected, submodels, rootTypename, labelsMap) {
  const nodes = [{ labelUuid: null, modelTypename: rootTypename }];
  const allUuids = new Set([...selected, ...submodels.keys()]);
  for (const uuid of allUuids) {
    const typename = submodels.has(uuid)
      ? submodels.get(uuid)
      : inheritedTypename(uuid, labelsMap, submodels, rootTypename);
    nodes.push({ labelUuid: uuid, modelTypename: typename });
  }
  return nodes;
}

// -------------------------------------------------------
// Submodel panels
// -------------------------------------------------------

function renderSubmodelPanels(container, submodels, labelsMap, typenames, onTypenameChange) {
  container.innerHTML = '';
  if (submodels.size === 0) return;

  const hdr = document.createElement('p');
  hdr.className = 'panel-section-label';
  hdr.textContent = 'Sub-model overrides';
  container.appendChild(hdr);

  for (const [uuid, typename] of submodels) {
    const labelName = labelsMap.get(uuid)?.name ?? uuid;
    const labelColor = labelsMap.get(uuid)?.color ?? '#888';

    const card = document.createElement('div');
    card.className = 'mc-submodel-card';

    const swatch = document.createElement('span');
    swatch.className = 'mc-submodel-swatch';
    swatch.style.background = labelColor;

    const nameSpan = document.createElement('span');
    nameSpan.className = 'mc-submodel-name';
    nameSpan.textContent = labelName;

    const typeLabel = document.createElement('label');
    typeLabel.className = 'mc-submodel-type-label';
    typeLabel.textContent = 'Model:';

    const sel = document.createElement('select');
    // 'width: auto' overrides Pico's 'width: 100%' which would collapse the name span to zero
    sel.style.cssText = 'margin-bottom: 0; width: auto; flex-shrink: 0;';
    typenames.forEach(tn => {
      const o = document.createElement('option');
      o.value = tn; o.textContent = tn;
      if (tn === typename) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener('change', () => onTypenameChange(uuid, sel.value));

    card.append(swatch, nameSpan, typeLabel, sel);
    container.appendChild(card);
  }
}

// -------------------------------------------------------
// Config form body
// Returns { el, getSaveData, destroy }
// Action buttons live in the panel header bar; not included here.
// -------------------------------------------------------

function renderConfigForm({ config, typenames, labelsMap, onDirtyChange }) {
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

  // Root model type
  const { rootTypename: initRoot, selected: initSelected, submodels: initSubmodels } =
    parseNodes(config?.nodes, labelsMap);
  let rootTypename = initRoot || typenames[0] || '';

  const rootSel = document.createElement('select');
  typenames.forEach(tn => {
    const o = document.createElement('option');
    o.value = tn; o.textContent = tn;
    if (tn === rootTypename) o.selected = true;
    rootSel.appendChild(o);
  });
  rootSel.addEventListener('change', () => { rootTypename = rootSel.value; _markDirty(); });
  el.appendChild(makeFieldRow('Root model', rootSel));

  // Divider
  const hr = document.createElement('hr');
  hr.style.cssText = 'margin: 0.6em 0; opacity: 0.4;';
  el.appendChild(hr);

  // Label tree
  const treeLabel = document.createElement('p');
  treeLabel.className = 'panel-section-label';
  treeLabel.style.marginTop = '0';
  treeLabel.textContent = 'Label assignments';
  el.appendChild(treeLabel);

  const treeBox = document.createElement('div');
  treeBox.className = 'panel-tree-box';
  treeBox.addEventListener('change', _markDirty);
  el.appendChild(treeBox);

  // Submodel panels (populated dynamically by tree callbacks)
  const submodelsContainer = document.createElement('div');
  el.appendChild(submodelsContainer);

  let currentSubmodels = new Map(initSubmodels);
  let treeHandle = null;

  function refreshPanels(submodels) {
    currentSubmodels = submodels;
    renderSubmodelPanels(submodelsContainer, submodels, labelsMap, typenames,
      (uuid, newTypename) => { currentSubmodels.set(uuid, newTypename); _markDirty(); });
  }

  createLabelList(treeBox, {
    mode: 'model-config',
    labelsMap,
    initialSelected: initSelected,
    initialSubmodels: initSubmodels,
    typenames,
    onSubmodelsChanged: (submodels) => { refreshPanels(submodels); _markDirty(); },
  }).then(handle => {
    treeHandle = handle;
    refreshPanels(initSubmodels);
  });

  return {
    el,
    getSaveData() {
      return {
        name: nameInput.value.trim(),
        description: descInput.value.trim(),
        nodes: computeNodes(treeHandle?.getSelected() ?? new Set(), currentSubmodels, rootTypename, labelsMap),
      };
    },
    destroy() { treeHandle?.destroy(); },
  };
}

// -------------------------------------------------------
// Public entry point
// -------------------------------------------------------

export async function renderModelConfigManager({ target = 'model-config-box' } = {}) {
  if (_rendering) return;
  _rendering = true;

  // Replace any previously-installed nav guard (handles re-render on tab revisit)
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
    listTitle.textContent = 'Configurations';
    listBar.appendChild(listTitle);
    listPanel.appendChild(listBar);

    const listBody = document.createElement('div');
    listBody.className = 'panel-box__body';
    listPanel.appendChild(listBody);

    const listFooter = document.createElement('div');
    listFooter.className = 'panel-box__footer';
    listPanel.appendChild(listFooter);

    // Footer "+ New" button is static — created once here, not on every refreshList.
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
    let _barHandle = null; // set by _showForm; tracks dirty state via getDirty()/setDirty()
    let _currentFormHandle = null;
    let _currentDoSave = null; // async () => boolean — current form's save-only fn (no re-render)
    let _switching = false; // true while a showEdit/showCreate transition is in progress

    async function _confirmIfDirty() {
      if (!_barHandle?.getDirty()) return true;
      const result = await showUnsavedConfigChangesModal();
      if (result === 'cancel') return false;
      if (result === 'save') return await (_currentDoSave?.() ?? false);
      return true; // 'discard'
    }

    function _updateListSelection(uuid) {
      for (const item of listBody.querySelectorAll('.mc-list-item')) {
        item.classList.toggle('active', item.dataset.configUuid === uuid);
      }
    }

    /**
     * Rebuild the editor header bar. Returns handles to update dynamic parts.
     * onDelete is optional (omit for the "new config" form).
     */
    // Single entry point for both edit (config != null) and create (config == null).
    async function _showForm(config, { force = false } = {}) {
      if (_switching && !force) return;
      _switching = true;
      try {
        if (!force && !await _confirmIfDirty()) return;
        _currentFormHandle?.destroy();
        _currentFormHandle = null;
        _currentDoSave = null;
        _barHandle = null; // mark clean while async fetches are in flight
        _selectedUuid = config?.uuid ?? null;

        const [labelsMap, typenames] = await Promise.all([fetchLabels(), fetchModelTypenames()]);
        editorBody.innerHTML = '';

        _barHandle = buildPanelEditorBar({
          container: editorBar,
          item: config,
          newTitle: 'New Configuration',
          deleteTitle: 'Delete configuration',
          onSave: async () => {
            const { name, description, nodes } = _currentFormHandle.getSaveData();
            if (!name) { toast('Name is required', 3000, 'error'); return; }
            if (!nodes[0]?.modelTypename) { toast('Root model type is required', 3000, 'error'); return; }
            setSaving(true);
            try {
              if (config) {
                const updated = await updateModelConfig(config.uuid, { name, description, nodes });
                setDirty(false);
                toast('Saved', 3000);
                const nameEl = listBody.querySelector(`[data-config-uuid="${config.uuid}"] .mc-list-item__name`);
                if (nameEl) nameEl.textContent = updated.name;
                await _showForm(updated, { force: true });
              } else {
                const created = await createModelConfig({ name, description, nodes });
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
            await deleteModelConfig(config.uuid);
            toast('Deleted', 3000);
            await refreshList();
            await _showForm(null, { force: true });
          } : undefined,
        });
        const { setDirty, setSaving } = _barHandle;

        const formHandle = renderConfigForm({
          config, typenames, labelsMap,
          onDirtyChange: setDirty,
        });
        _currentFormHandle = formHandle;
        editorBody.appendChild(formHandle.el);

        // Save-only function for the dirty-navigation guard (no re-render after save).
        _currentDoSave = async () => {
          const { name, description, nodes } = formHandle.getSaveData();
          if (!name) { toast('Name is required', 3000, 'error'); return false; }
          if (!nodes[0]?.modelTypename) { toast('Root model type is required', 3000, 'error'); return false; }
          setSaving(true);
          try {
            if (config) {
              const updated = await updateModelConfig(config.uuid, { name, description, nodes });
              setDirty(false);
              toast('Saved', 3000);
              const nameEl = listBody.querySelector(`[data-config-uuid="${config.uuid}"] .mc-list-item__name`);
              if (nameEl) nameEl.textContent = updated.name;
            } else {
              const created = await createModelConfig({ name, description, nodes });
              setDirty(false);
              toast(`Created "${created.name}"`, 3000);
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

      const configsMap = await fetchModelConfigs();

      if (configsMap.size === 0) {
        const empty = document.createElement('p');
        empty.style.cssText = 'color: var(--pico-muted-color, #888); font-style: italic; font-size: 0.85em; padding: 0.5em 0.75em; margin: 0;';
        empty.textContent = 'No configurations yet.';
        listBody.appendChild(empty);
      }

      for (const config of configsMap.values()) {
        const item = document.createElement('div');
        item.className = 'mc-list-item';
        item.dataset.configUuid = config.uuid;

        const nameLine = document.createElement('div');
        nameLine.className = 'mc-list-item__name';
        nameLine.textContent = config.name;
        item.appendChild(nameLine);

        if (config.description) {
          const descLine = document.createElement('div');
          descLine.className = 'mc-list-item__desc';
          descLine.textContent = config.description;
          item.appendChild(descLine);
        }

        const metaLine = document.createElement('div');
        metaLine.className = 'mc-list-item__meta';
        const n = config.nodes?.length ?? 0;
        metaLine.textContent = `${n} assignment${n === 1 ? '' : 's'}`;
        item.appendChild(metaLine);

        item.addEventListener('click', () => _showForm(config));
        listBody.appendChild(item);
      }

      _updateListSelection(_selectedUuid);
    }

    // ── Navigation guard ─────────────────────────────────────────────────
    // Intercept workspace-tab and main-tab clicks while there are unsaved changes.
    // Uses capture phase so we can stop propagation before the tab system sees it.
    // guardSignal is captured now so the async continuation can detect if this guard
    // has been superseded by a re-render (signal.aborted → stale, don't act).
    const guardSignal = _navGuardController.signal;
    document.addEventListener('click', async (e) => {
      const btn = e.target.closest('.workspace-tab-button, .main-tab-button');
      if (!btn) return;
      const targetTab = btn.dataset.tab;
      // Allow clicking the Models tab itself or the Configuration main tab (no navigation away)
      if (targetTab === 'tab-model-management' || targetTab === 'tab-configuration') return;
      if (!_barHandle?.getDirty()) return;

      e.stopPropagation();
      const result = await showUnsavedConfigChangesModal();
      if (result === 'cancel') return;
      if (result === 'save') {
        const ok = await _currentDoSave?.();
        if (!ok) return;
      }
      if (guardSignal.aborted) return; // guard superseded by a re-render; discard is stale
      _barHandle?.setDirty(false);
      btn.click(); // re-fire now that guard is satisfied
    }, { capture: true, signal: guardSignal });

    await refreshList();
    await _showForm(null, { force: true });
  } catch (err) {
    const errMsg = document.createElement('p');
    errMsg.style.color = 'red';
    errMsg.textContent = `Error loading model configurations: ${err.message}`;
    container.appendChild(errMsg);
  } finally {
    _rendering = false;
  }
}
