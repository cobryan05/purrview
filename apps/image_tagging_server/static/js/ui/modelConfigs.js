import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { createLabelList } from '/app-static/js/ui/labelList.js';
import {
  createModelConfig,
  deleteModelConfig,
  fetchModelConfigs,
  updateModelConfig,
} from '/app-static/js/shared/api/modelConfigs.js';
import { createListEditorScreen } from '/app-static/js/ui/utils/listEditorScreen.js';
import { toast, makeFieldRow, buildPanelEditorBar } from '/app-static/js/ui/utils/index.js';
import { MODEL_SPEC, scopeForModelType, parseNodes, computeNodes } from '/app-static/js/ui/modelSpec.js';
import { renderModelTreePanel } from '/app-static/js/ui/modelTreePanel.js';

let _rendering = false;
let _navGuardController = null; // AbortController for the tab-navigation guard

/** Returns a pair of linked <select> elements for choosing model type + architecture.
 *  When the type changes the architecture list repopulates automatically.
 *  @param {?{modelType:string,architecture:string}} initialSpec
 *  @param {(spec:{modelType:string,architecture:string})=>void} onChange
 *  @returns {{ el: HTMLElement, getSpec: ()=>{modelType:string,architecture:string} }}
 */
function makeModelSpecSelectors(initialSpec, onChange) {
  const wrap = document.createElement('div');
  wrap.style.cssText = 'display: flex; gap: 0.5em; align-items: center;';

  const typeSel = document.createElement('select');
  typeSel.style.cssText = 'margin-bottom: 0; width: auto;';

  const archSel = document.createElement('select');
  archSel.style.cssText = 'margin-bottom: 0; width: auto;';

  function populateArchs(modelType, selectArch) {
    archSel.innerHTML = '';
    const archs = MODEL_SPEC[modelType]?.architectures ?? {};
    for (const [key, label] of Object.entries(archs)) {
      const o = document.createElement('option');
      o.value = key;
      o.textContent = label;
      if (key === selectArch) o.selected = true;
      archSel.appendChild(o);
    }
  }

  for (const [key, { label }] of Object.entries(MODEL_SPEC)) {
    const o = document.createElement('option');
    o.value = key;
    o.textContent = label;
    if (key === initialSpec?.modelType) o.selected = true;
    typeSel.appendChild(o);
  }

  const initType = initialSpec?.modelType ?? Object.keys(MODEL_SPEC)[0];
  populateArchs(initType, initialSpec?.architecture);

  typeSel.addEventListener('change', () => {
    populateArchs(typeSel.value, null);
    onChange({ modelType: typeSel.value, architecture: archSel.value });
  });
  archSel.addEventListener('change', () => {
    onChange({ modelType: typeSel.value, architecture: archSel.value });
  });

  wrap.append(typeSel, archSel);
  return {
    el: wrap,
    getSpec: () => ({ modelType: typeSel.value, architecture: archSel.value }),
  };
}

// -------------------------------------------------------
// Submodel panels
// -------------------------------------------------------

function renderSubmodelPanels(container, submodels, labelsMap, onSpecChange) {
  container.innerHTML = '';
  if (submodels.size === 0) return;

  const hdr = document.createElement('p');
  hdr.className = 'panel-section-label';
  hdr.textContent = 'Sub-model overrides';
  container.appendChild(hdr);

  for (const [uuid, spec] of submodels) {
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

    const { el: specEl } = makeModelSpecSelectors(spec, (newSpec) => onSpecChange(uuid, newSpec));

    card.append(swatch, nameSpan, specEl);
    container.appendChild(card);
  }
}

// -------------------------------------------------------
// Config form body
// Returns { el, getSaveData, destroy }
// Action buttons live in the panel header bar; not included here.
// -------------------------------------------------------

function renderConfigForm({ config, labelsMap, onDirtyChange }) {
  const el = document.createElement('div');

  function _markDirty() { onDirtyChange?.(true); refreshTreeSummary(); }

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

  // Root model type + architecture
  const { rootSpec: initRootSpec, selected: initSelected, submodels: initSubmodels, rollUpUnchecked: initRollUpUnchecked } =
    parseNodes(config?.nodes, labelsMap);

  let currentScope = scopeForModelType(initRootSpec.modelType);
  let currentSubmodels = new Map(initSubmodels);
  let currentRollUpUnchecked = new Set(initRollUpUnchecked);
  let treeHandle = null;

  // Default spec seeded when the user first enables a sub-model toggle on a label.
  // Crop classifier is the most common sub-model type (refining a detected parent).
  const defaultSubSpec = { modelType: 'crop_classifier', architecture: 'efficientnet_b0' };

  // Submodel panels (populated dynamically by tree callbacks)
  const submodelsContainer = document.createElement('div');

  function refreshPanels(submodels) {
    currentSubmodels = submodels;
    // Sub-model overrides only make sense for object/crop models (not image classifiers).
    if (currentScope === 'image') {
      submodelsContainer.innerHTML = '';
    } else {
      renderSubmodelPanels(submodelsContainer, submodels, labelsMap,
        (uuid, newSpec) => { currentSubmodels.set(uuid, newSpec); _markDirty(); });
    }
  }

  const treeBox = document.createElement('div');
  treeBox.className = 'panel-tree-box';
  treeBox.addEventListener('change', _markDirty);

  async function buildLabelTree(scope, initialSelected, initialSubmodels, initialRollUpUnchecked) {
    treeHandle?.destroy();
    treeHandle = null;
    treeBox.innerHTML = '';
    const handle = await createLabelList(treeBox, {
      mode: 'model-config',
      labelsMap,
      initialSelected,
      initialSubmodels,
      initialRollUpUnchecked,
      defaultSpec: defaultSubSpec,
      scopeFilter: scope,
      onSubmodelsChanged: (submodels) => { refreshPanels(submodels); _markDirty(); },
      onRollUpUncheckedChanged: (set) => { currentRollUpUnchecked = set; _markDirty(); },
      onLabelsChanged: onLabelsMoved,
    });
    treeHandle = handle;
    refreshPanels(new Map(initialSubmodels));
    refreshTreeSummary();
  }

  // A re-parent moved a label in the hierarchy: `labelsMap` (closed over by
  // refreshPanels/treeSummary/getSaveData) is a snapshot taken when the form
  // opened, so it must be refreshed before rebuilding the tree — otherwise the
  // rebuild would just redraw the old hierarchy shape. ModelConfigNode is
  // keyed by label_uuid, so selection/submodel/roll-up state (all keyed the
  // same way) survives the rebuild untouched.
  async function onLabelsMoved() {
    labelsMap = await fetchLabels();
    // The move itself is already persisted server-side (a direct PATCH), and
    // ModelConfigNode is keyed by label_uuid, so this config's own save data
    // (nodes list) is unaffected — don't mark the form dirty over it.
    await buildLabelTree(currentScope, treeHandle?.getSelected() ?? new Set(), currentSubmodels, currentRollUpUnchecked);
  }

  const rootSpecHandle = makeModelSpecSelectors(initRootSpec, (newSpec) => {
    _markDirty();
    const newScope = scopeForModelType(newSpec.modelType);
    if (newScope !== currentScope) {
      currentScope = newScope;
      // Scope changed — old label selections are for a different annotation type,
      // so clear them and rebuild the tree with the correct label scope.
      buildLabelTree(newScope, new Set(), new Map(), new Set());
    }
  });
  el.appendChild(makeFieldRow('Root model', rootSpecHandle.el));

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

  el.appendChild(treeBox);
  el.appendChild(submodelsContainer);

  // ── Live model-tree summary ───────────────────────────────────────────────
  // Visualises the sub-model hierarchy this form will save (NOTES.md contract).
  const treeSummaryHdr = document.createElement('p');
  treeSummaryHdr.className = 'panel-section-label';
  treeSummaryHdr.textContent = 'Model tree';
  el.appendChild(treeSummaryHdr);
  const treeSummaryBox = document.createElement('div');
  treeSummaryBox.className = 'mtp-box';
  el.appendChild(treeSummaryBox);

  let _treeRefreshQueued = false;
  function refreshTreeSummary() {
    if (_treeRefreshQueued) return; // coalesce bursts (e.g. submodel change → markDirty)
    _treeRefreshQueued = true;
    queueMicrotask(() => {
      _treeRefreshQueued = false;
      const nodes = computeNodes(
        treeHandle?.getSelected() ?? new Set(),
        currentSubmodels,
        currentRollUpUnchecked,
        rootSpecHandle.getSpec(),
        labelsMap,
      );
      treeSummaryBox.replaceChildren(renderModelTreePanel({ nodes, labelsMap }));
    });
  }

  // Kick off the initial tree build (async, but non-blocking for the rest of the form).
  buildLabelTree(currentScope, initSelected, initSubmodels, initRollUpUnchecked);

  return {
    el,
    getSaveData() {
      const rootSpec = rootSpecHandle.getSpec();
      return {
        name: nameInput.value.trim(),
        description: descInput.value.trim(),
        nodes: computeNodes(
          treeHandle?.getSelected() ?? new Set(),
          currentSubmodels,
          currentRollUpUnchecked,
          rootSpec,
          labelsMap,
        ),
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
    const screen = createListEditorScreen({
      container,
      prefix: 'mc',
      title: 'Model Shapes',
      itemUuidAttr: 'configUuid',
      onNew: () => _showForm(null),
      ownTabId: 'tab-model-management',
      navGuardSignal: _navGuardController.signal,
    });
    const { listBody, editorBar, editorBody } = screen;
    const _confirmIfDirty = screen.confirmIfDirty;
    const _updateListSelection = screen.updateListSelection;

    // ── Shared state ──────────────────────────────────────────────────────
    let _selectedUuid = null;
    let _currentFormHandle = null;
    let _switching = false; // true while a showEdit/showCreate transition is in progress

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
        screen.setDoSave(null);
        screen.setBarHandle(null); // mark clean while async fetches are in flight
        _selectedUuid = config?.uuid ?? null;

        const labelsMap = await fetchLabels();
        editorBody.innerHTML = '';

        const barHandle = buildPanelEditorBar({
          container: editorBar,
          item: config,
          newTitle: 'New Model Shape',
          deleteTitle: 'Delete model shape',
          onSave: async () => {
            const { name, description, nodes } = _currentFormHandle.getSaveData();
            if (!name) { toast('Name is required', 3000, 'error'); return; }
            if (!nodes[0]?.modelType) { toast('Root model type is required', 3000, 'error'); return; }
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
        screen.setBarHandle(barHandle);
        const { setDirty, setSaving } = barHandle;

        const formHandle = renderConfigForm({
          config, labelsMap,
          onDirtyChange: setDirty,
        });
        _currentFormHandle = formHandle;
        editorBody.appendChild(formHandle.el);

        // Save-only function for the dirty-navigation guard (no re-render after save).
        screen.setDoSave(async () => {
          const { name, description, nodes } = formHandle.getSaveData();
          if (!name) { toast('Name is required', 3000, 'error'); return false; }
          if (!nodes[0]?.modelType) { toast('Root model type is required', 3000, 'error'); return false; }
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
        });

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
        empty.textContent = 'No model shapes yet — use the form on the right to create your first one.';
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
        // Root node (labelUuid=null) is the base model, not a label assignment.
        const n = (config.nodes ?? []).filter((node) => node.labelUuid).length;
        metaLine.textContent = `${n} assignment${n === 1 ? '' : 's'}`;
        item.appendChild(metaLine);

        item.addEventListener('click', () => _showForm(config));
        listBody.appendChild(item);
      }

      _updateListSelection(_selectedUuid);
    }

    await refreshList();
    await _showForm(null, { force: true });
  } catch (err) {
    const errMsg = document.createElement('p');
    errMsg.style.color = 'red';
    errMsg.textContent = `Error loading model shapes: ${err.message}`;
    container.appendChild(errMsg);
  } finally {
    _rendering = false;
  }
}
