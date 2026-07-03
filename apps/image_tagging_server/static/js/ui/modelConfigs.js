import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { createLabelList } from '/app-static/js/ui/labelList.js';
import {
  createModelConfig,
  deleteModelConfig,
  fetchModelConfigs,
  updateModelConfig,
} from '/app-static/js/shared/api/modelConfigs.js';
import { showUnsavedConfigChangesModal } from '/app-static/js/ui/utils/confirmModal.js';
import { toast, makeFieldRow, buildPanelEditorBar } from '/app-static/js/ui/utils/index.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { MODEL_SPEC, scopeForModelType, parseNodes, computeNodes } from '/app-static/js/ui/modelSpec.js';
import { renderModelTreePanel } from '/app-static/js/ui/modelTreePanel.js';
import { fetchModelBundlesForConfig } from '/app-static/js/shared/api/modelBundles.js';

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
    });
    treeHandle = handle;
    refreshPanels(new Map(initialSubmodels));
    refreshTreeSummary();
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

  // ── Trained bundles (saved configs only) ──────────────────────────────────
  if (config) {
    const bundlesHdr = document.createElement('p');
    bundlesHdr.className = 'panel-section-label';
    bundlesHdr.textContent = 'Trained bundles';
    el.appendChild(bundlesHdr);
    const bundlesBox = document.createElement('div');
    bundlesBox.className = 'mtp-bundles';
    bundlesBox.textContent = 'Loading…';
    el.appendChild(bundlesBox);
    fetchModelBundlesForConfig(config.uuid).then((bundles) => {
      bundlesBox.innerHTML = '';
      if (bundles.length === 0) {
        bundlesBox.textContent = 'No bundles trained from this configuration yet.';
        return;
      }
      for (const b of bundles) {
        const row = document.createElement('div');
        row.className = 'mtp-bundle-row';
        const name = document.createElement('span');
        name.className = 'mtp-bundle-row__name';
        name.textContent = b.name;
        name.title = b.description || b.bundlePath || '';
        row.appendChild(name);
        const meta = document.createElement('span');
        meta.className = 'mtp-bundle-row__meta';
        const parts = [];
        if (b.createdAt) parts.push(new Date(b.createdAt).toLocaleDateString());
        const artifacts = b.manifestJson?.artifacts;
        if (Array.isArray(artifacts)) parts.push(`${artifacts.length} artifact${artifacts.length === 1 ? '' : 's'}`);
        meta.textContent = parts.join(' · ');
        row.appendChild(meta);
        bundlesBox.appendChild(row);
      }
    }).catch((err) => {
      bundlesBox.textContent = `Failed to load bundles: ${err.message}`;
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

        const labelsMap = await fetchLabels();
        editorBody.innerHTML = '';

        _barHandle = buildPanelEditorBar({
          container: editorBar,
          item: config,
          newTitle: 'New Configuration',
          deleteTitle: 'Delete configuration',
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
        const { setDirty, setSaving } = _barHandle;

        const formHandle = renderConfigForm({
          config, labelsMap,
          onDirtyChange: setDirty,
        });
        _currentFormHandle = formHandle;
        editorBody.appendChild(formHandle.el);

        // Save-only function for the dirty-navigation guard (no re-render after save).
        _currentDoSave = async () => {
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
        empty.textContent = 'No configurations yet — use the form on the right to create your first one.';
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
