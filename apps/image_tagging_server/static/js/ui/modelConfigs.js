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
import { NEW_ITEM_SELECTION, takePendingSelection, navigateTabs } from '/app-static/js/app/utils/tabs.js';
import { launchRun } from '/app-static/js/ui/utils/launchRun.js';
import { fetchModelUnits } from '/app-static/js/shared/api/inference.js';

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
  // Purely a convenience default: the sub-model stays a sub-model whatever spec it
  // ends up with, including one identical to its parent's.
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

  // The label tree here is read-only with respect to the hierarchy: re-parenting
  // lives solely on Configuration › Labels & Tags, so there is no in-form move
  // that could invalidate the `labelsMap` snapshot this form closed over.

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

  // ── Train the whole tree ──────────────────────────────────────────────────
  // The primary way to train a hierarchy, sitting under the tree it trains. One run
  // per node, in dependency order, creating a training config for any node that lacks
  // one. Training the parts individually is still possible from the Training tab; it is
  // the "retrain one node" path, not the way to build the thing in the first place —
  // which it used to be, and which left every sub-model untrained.
  if (config?.uuid) {
    const trainRow = document.createElement('div');
    trainRow.className = 'training-start-run';

    const trainBtn = document.createElement('button');
    trainBtn.type = 'button';
    trainBtn.className = 'panel-btn';
    trainBtn.textContent = '▶ Train hierarchy';
    trainBtn.dataset.testid = 'model-config-train-hierarchy';
    trainBtn.dataset.tour = 'config.modelShapes.trainHierarchy';
    trainBtn.title = 'Train every part of this model tree, top model first';
    trainBtn.addEventListener('click', async () => {
      trainBtn.disabled = true;
      try {
        await launchRun({
          typename: 'TrainHierarchyTask',
          name: `Train ${config.name}`,
          params: { model_config_uuid: config.uuid },
          // Same reason TrainModelTask is persistent: the jobs run on the training
          // server, so a tasks_server restart only detaches the thing watching them.
          persistent: true,
        });
        toast(`Started Train ${config.name}`, 3000);
        navigateTabs('tab-runs', 'tab-runs-active');
      } catch (err) {
        toast(err.message || 'Could not start the training run', 5000, 'error');
      } finally {
        trainBtn.disabled = false;
      }
    });

    const trainHint = document.createElement('span');
    trainHint.className = 'training-start-run__hint';
    trainHint.textContent = 'Runs one training job per part, in order. Progress and cancel live in Jobs.';

    trainRow.append(trainBtn, trainHint);
    el.appendChild(trainRow);
  }

  // Which parts of THIS config have a trained artifact, from the inference server's
  // view of its own models directory. Null until it answers (and if it never does),
  // because "we don't know yet" must not render as "not trained" — that would cry wolf
  // on every open and teach the warning away.
  /** @type {?{top: boolean, subLabelUuids: Set<string>}} */
  let _trained = null;
  if (config?.uuid) {
    fetchModelUnits()
      .then(units => {
        const unit = units.find(u => u.modelConfigUuid === config.uuid);
        if (!unit) {
          // No artifacts at all for this config: nothing has ever been trained, which is
          // exactly what a freshly designed hierarchy looks like and worth saying.
          _trained = { top: false, subLabelUuids: new Set() };
        } else {
          const parts = unit.parts || [];
          _trained = {
            top: parts.some(p => p.role === 'top'),
            subLabelUuids: new Set(parts.filter(p => p.role === 'submodel' && p.labelUuid).map(p => p.labelUuid)),
          };
        }
        refreshTreeSummary();
      })
      .catch(() => { /* inference server unreachable — say nothing rather than lie */ });
  }

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
      treeSummaryBox.replaceChildren(renderModelTreePanel({ nodes, labelsMap, trained: _trained }));
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

    // Last list fetch, so arrival-with-a-selection can resolve a uuid to its
    // config without a second round trip.
    let _configsMap = new Map();

    async function refreshList() {
      listBody.innerHTML = '';

      const configsMap = await fetchModelConfigs();
      _configsMap = configsMap;

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

    // Somewhere else (e.g. the Training config's Model Shape field) may have
    // sent the user here to look at — or create — a specific shape.
    // NEW_ITEM_SELECTION needs no branch: the create form is what _showForm(null)
    // already opens, and an unknown uuid falls through to the same place.
    const wantedUuid = takePendingSelection('tab-model-management');
    const wantedConfig = wantedUuid && wantedUuid !== NEW_ITEM_SELECTION
      ? (_configsMap.get(wantedUuid) ?? null)
      : null;
    await _showForm(wantedConfig, { force: true });
  } catch (err) {
    const errMsg = document.createElement('p');
    errMsg.style.color = 'red';
    errMsg.textContent = `Error loading model shapes: ${err.message}`;
    container.appendChild(errMsg);
  } finally {
    _rendering = false;
  }
}
