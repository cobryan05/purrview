import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import { createLabelList } from '/app-static/js/ui/labelList.js';
import {
  createModelConfig,
  deleteModelConfig,
  fetchModelConfigs,
  fetchModelConfigTrainingTargets,
  updateModelConfig,
} from '/app-static/js/shared/api/modelConfigs.js';
import { fetchDatasets } from '/app-static/js/shared/api/datasets.js';
import { TaskStatus } from '/app-static/js/shared/api/tasks.js';
import { showUnsavedConfigChangesModal } from '/app-static/js/ui/utils/confirmModal.js';
import { toast, makeFieldRow, buildPanelEditorBar } from '/app-static/js/ui/utils/index.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { MODEL_SPEC, scopeForModelType, parseNodes, computeNodes } from '/app-static/js/ui/modelSpec.js';
import { renderModelTreePanel } from '/app-static/js/ui/modelTreePanel.js';
import { createTaskRunnerSection } from '/app-static/js/ui/utils/taskRunnerSection.js';
import { createHyperparamsFieldset } from '/app-static/js/ui/utils/hyperparamsFieldset.js';
import {
  createModelBundle,
  fetchModelBundlesForConfig,
} from '/app-static/js/shared/api/modelBundles.js';

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
// Trained bundles + "Train all → bundle" + compose-from-artifacts
// -------------------------------------------------------

/** Labels an artifact's scope: "Top model" for the root, or the joined
 * sub-model label names it discriminates (falling back to raw UUIDs if a
 * label was since deleted). Artifact dicts come straight from
 * ModelBundle.manifest_json (a raw JSON blob, not alias-generated — keys are
 * snake_case as TrainBundleTask wrote them). */
function _artifactLabelSummary(artifact, labelsMap) {
  if (artifact.role === 'top') return 'Top model';
  const uuids = artifact.label_uuids?.length ? artifact.label_uuids : (artifact.label_uuid ? [artifact.label_uuid] : []);
  if (uuids.length === 0) return 'Sub-model';
  return uuids.map(u => labelsMap?.get(u)?.name ?? u).join(', ');
}

function _renderBundleArtifactsList(artifacts, labelsMap) {
  const list = document.createElement('div');
  list.className = 'mtp-bundle-artifacts';
  for (const a of artifacts) {
    const chip = document.createElement('span');
    chip.className = 'mtp-chip mtp-bundle-artifact';
    const roleText = a.role === 'top' ? 'Top' : 'Sub';
    const acc = typeof a.val_accuracy === 'number' ? ` · ${(a.val_accuracy * 100).toFixed(1)}%` : '';
    chip.textContent = `${roleText}: ${_artifactLabelSummary(a, labelsMap)}${acc}`;
    chip.title = a.model_name ?? '';
    list.appendChild(chip);
  }
  return list;
}

/** Renders the "Trained bundles" list for a saved config, with a header
 * button that toggles the compose-from-artifacts panel. Returns a refresh()
 * callers invoke after a new bundle is created (train-all or compose). */
function renderBundlesSection({ config, labelsMap, onComposeClick }) {
  const wrap = document.createElement('div');

  const hdrRow = document.createElement('div');
  hdrRow.style.cssText = 'display:flex; align-items:center; justify-content:space-between; gap:0.5em;';
  const bundlesHdr = document.createElement('p');
  bundlesHdr.className = 'panel-section-label';
  bundlesHdr.style.margin = '0';
  bundlesHdr.textContent = 'Trained bundles';
  hdrRow.appendChild(bundlesHdr);

  const composeBtn = document.createElement('button');
  composeBtn.type = 'button';
  composeBtn.className = 'panel-btn';
  composeBtn.style.cssText = 'font-size:0.8em; padding:1px 8px; margin:0;';
  composeBtn.textContent = '+ New bundle from artifacts';
  composeBtn.addEventListener('click', () => onComposeClick?.());
  hdrRow.appendChild(composeBtn);
  wrap.appendChild(hdrRow);

  const bundlesBox = document.createElement('div');
  bundlesBox.className = 'mtp-bundles';
  bundlesBox.textContent = 'Loading…';
  wrap.appendChild(bundlesBox);

  async function refresh() {
    bundlesBox.textContent = 'Loading…';
    try {
      const bundles = await fetchModelBundlesForConfig(config.uuid);
      bundlesBox.innerHTML = '';
      if (bundles.length === 0) {
        bundlesBox.textContent = 'No bundles trained from this configuration yet.';
        return;
      }
      for (const b of bundles) {
        const row = document.createElement('div');
        row.className = 'mtp-bundle-row';

        const topLine = document.createElement('div');
        topLine.style.cssText = 'display:flex; justify-content:space-between; gap:1em;';
        const name = document.createElement('span');
        name.className = 'mtp-bundle-row__name';
        name.textContent = b.name;
        name.title = b.description || b.bundlePath || '';
        topLine.appendChild(name);

        const meta = document.createElement('span');
        meta.className = 'mtp-bundle-row__meta';
        const artifacts = Array.isArray(b.manifestJson?.artifacts) ? b.manifestJson.artifacts : [];
        const parts = [];
        if (b.createdAt) parts.push(new Date(b.createdAt).toLocaleDateString());
        parts.push(`${artifacts.length} artifact${artifacts.length === 1 ? '' : 's'}`);
        meta.textContent = parts.join(' · ');
        topLine.appendChild(meta);
        row.appendChild(topLine);

        if (artifacts.length > 0) row.appendChild(_renderBundleArtifactsList(artifacts, labelsMap));
        bundlesBox.appendChild(row);
      }
    } catch (err) {
      bundlesBox.textContent = `Failed to load bundles: ${err.message}`;
    }
  }

  refresh();
  return { el: wrap, refresh };
}

/** True when both label-UUID sets are the same regardless of order. */
function _sameLabelSet(a, b) {
  if (a.length !== b.length) return false;
  const sa = [...a].sort();
  const sb = [...b].sort();
  return sa.every((v, i) => v === sb[i]);
}

/** Hidden-until-opened panel: pick one existing artifact per current
 * training target (top model + each sub-model) from across ALL bundles ever
 * trained for this config, then assemble them into a new immutable
 * ModelBundle snapshot. Lets a user retrain just one sub-model (via a
 * Training Config targeting that node) and compose a new bundle reusing the
 * other, unchanged artifacts — no retraining of the whole tree required.
 * Targets come from GET /api/model-configs/{uuid}/training-targets, the same
 * authority TrainBundleTask itself uses, so "coverage" here means the same
 * thing it means during a real training run. */
function renderComposeBundlePanel({ config, labelsMap, onCreated }) {
  const wrap = document.createElement('div');
  wrap.className = 'mc-compose-box';
  wrap.style.display = 'none';

  const hdr = document.createElement('p');
  hdr.className = 'panel-section-label';
  hdr.textContent = 'New bundle from artifacts';
  wrap.appendChild(hdr);

  const body = document.createElement('div');
  wrap.appendChild(body);

  let _visible = false;

  async function _load() {
    body.innerHTML = '';
    const loading = document.createElement('div');
    loading.textContent = 'Loading…';
    body.appendChild(loading);

    let targets, bundles;
    try {
      [targets, bundles] = await Promise.all([
        fetchModelConfigTrainingTargets(config.uuid),
        fetchModelBundlesForConfig(config.uuid),
      ]);
    } catch (err) {
      body.innerHTML = '';
      body.textContent = `Failed to load artifacts: ${err.message}`;
      return;
    }
    body.innerHTML = '';

    if (targets.length === 0) {
      body.textContent = 'No training targets for this configuration yet (add a root model first).';
      return;
    }

    // Every past artifact across every bundle for this config, keyed by the
    // model-config node it was trained for.
    const artifactsByNode = new Map();
    for (const bundle of bundles) {
      const artifacts = Array.isArray(bundle.manifestJson?.artifacts) ? bundle.manifestJson.artifacts : [];
      for (const artifact of artifacts) {
        const list = artifactsByNode.get(artifact.node_uuid) ?? [];
        list.push({ artifact, bundle });
        artifactsByNode.set(artifact.node_uuid, list);
      }
    }

    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.placeholder = `${config.name} bundle`;
    body.appendChild(makeFieldRow('Bundle name', nameInput));

    const targetPicks = []; // { target, sel, optionMap }
    for (const target of targets) {
      const candidates = artifactsByNode.get(target.nodeUuid) ?? [];
      const rowLabel = target.role === 'top'
        ? 'Top model'
        : (labelsMap?.get(target.labelUuid)?.name ?? target.labelUuid ?? target.nodeUuid);

      const sel = document.createElement('select');
      sel.style.cssText = 'margin-bottom:0; width:auto;';
      const optionMap = new Map();

      const blank = document.createElement('option');
      blank.value = '';
      blank.textContent = candidates.length ? '— none —' : '— no artifacts available —';
      sel.appendChild(blank);

      candidates.forEach(({ artifact, bundle }, i) => {
        const value = `${bundle.uuid}:${i}`;
        const opt = document.createElement('option');
        opt.value = value;
        const acc = typeof artifact.val_accuracy === 'number' ? ` (${(artifact.val_accuracy * 100).toFixed(1)}%)` : '';
        opt.textContent = `${artifact.model_name ?? bundle.name}${acc} — from "${bundle.name}"`;
        optionMap.set(value, { artifact, bundle });
        sel.appendChild(opt);
      });

      const warnLine = document.createElement('div');
      warnLine.className = 'mc-compose-warn';
      warnLine.style.display = 'none';

      function _refreshWarn() {
        const picked = optionMap.get(sel.value);
        if (!picked) { warnLine.style.display = 'none'; return; }
        const wantUuids = target.labelUuids ?? [];
        const gotUuids = picked.artifact.label_uuids?.length ? picked.artifact.label_uuids
          : (picked.artifact.label_uuid ? [picked.artifact.label_uuid] : []);
        if (_sameLabelSet(wantUuids, gotUuids)) {
          warnLine.style.display = 'none';
        } else {
          warnLine.style.display = '';
          warnLine.textContent =
            `⚠ label coverage differs from this config's current target (artifact covers ${gotUuids.length}, expects ${wantUuids.length})`;
        }
      }
      sel.addEventListener('change', _refreshWarn);
      _refreshWarn();

      const rowWrap = document.createElement('div');
      rowWrap.appendChild(makeFieldRow(rowLabel, sel));
      rowWrap.appendChild(warnLine);
      body.appendChild(rowWrap);

      targetPicks.push({ target, sel, optionMap });
    }

    const createBtn = document.createElement('button');
    createBtn.type = 'button';
    createBtn.className = 'panel-btn';
    createBtn.textContent = 'Create bundle';
    createBtn.style.cssText = 'margin-top:0.5em;';
    createBtn.addEventListener('click', async () => {
      const name = nameInput.value.trim() || `${config.name} bundle`;

      const chosenArtifacts = [];
      let bundlePath = null;
      for (const { target, sel, optionMap } of targetPicks) {
        const picked = optionMap.get(sel.value);
        if (!picked) continue;
        bundlePath = bundlePath ?? picked.bundle.bundlePath;
        chosenArtifacts.push({ ...picked.artifact, node_uuid: target.nodeUuid, role: target.role });
      }
      if (chosenArtifacts.length === 0) { toast('Select at least one artifact', 3000, 'error'); return; }

      createBtn.disabled = true;
      try {
        await createModelBundle({
          name,
          modelConfigUuid: config.uuid,
          bundlePath: bundlePath ?? '',
          manifestJson: { artifacts: chosenArtifacts },
        });
        toast(`Created bundle "${name}"`, 3000);
        hide();
        await onCreated?.();
      } catch (err) {
        toast(err.message, 5000, 'error');
      } finally {
        createBtn.disabled = false;
      }
    });
    body.appendChild(createBtn);
  }

  function show() { wrap.style.display = ''; _visible = true; _load(); }
  function hide() { wrap.style.display = 'none'; _visible = false; }
  function toggle() { if (_visible) hide(); else show(); }

  return { el: wrap, toggle, hide };
}

/** "Train every target (top + each sub-model) sequentially, then assemble a
 * ModelBundle" — the TrainBundleTask entry point. Mirrors the Training
 * Config "Start Training" button (ui/trainingConfigs.js), sharing the same
 * SSE-driven runner and hyperparams fieldset. */
function renderTrainAllSection({ config, onBundleCreated }) {
  const wrap = document.createElement('div');
  wrap.style.cssText = 'margin-top: 1em; padding-top: 0.75em; border-top: 1px solid var(--pico-muted-border-color, #ccc);';

  const hdr = document.createElement('p');
  hdr.className = 'panel-section-label';
  hdr.style.marginTop = '0';
  hdr.textContent = 'Train all → bundle';
  wrap.appendChild(hdr);

  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.placeholder = `${config.name} bundle`;
  wrap.appendChild(makeFieldRow('Bundle name', nameInput));

  const datasetBox = document.createElement('div');
  datasetBox.className = 'panel-tree-box';
  datasetBox.textContent = 'Loading datasets…';
  const datasetRow = makeFieldRow('Datasets (≥1)', datasetBox);
  datasetRow.style.alignItems = 'flex-start';
  wrap.appendChild(datasetRow);

  const datasetCheckboxes = new Map();
  fetchDatasets().then((datasetsMap) => {
    datasetBox.innerHTML = '';
    if (datasetsMap.size === 0) {
      datasetBox.textContent = 'No datasets defined yet.';
      return;
    }
    for (const [uuid, ds] of datasetsMap) {
      const row = document.createElement('label');
      row.style.cssText = 'display: flex; align-items: center; gap: 0.4em; padding: 2px 4px; cursor: pointer; font-size: 0.88em; user-select: none;';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = uuid;
      row.appendChild(cb);
      const nameSpan = document.createElement('span');
      nameSpan.textContent = ds.name;
      row.appendChild(nameSpan);
      datasetBox.appendChild(row);
      datasetCheckboxes.set(uuid, cb);
    }
  }).catch((err) => { datasetBox.textContent = `Failed to load datasets: ${err.message}`; });

  const hyperparamsFieldset = createHyperparamsFieldset({});
  for (const row of hyperparamsFieldset.rows) wrap.appendChild(row);

  const { el: runnerEl } = createTaskRunnerSection({
    buttonLabel: '▶ Train All → Bundle',
    runningLabel: 'Training…',
    buildStartArgs: () => {
      const datasetUuids = [...datasetCheckboxes.entries()].filter(([, cb]) => cb.checked).map(([uuid]) => uuid);
      if (datasetUuids.length === 0) { toast('Select at least one dataset', 3000, 'error'); return null; }
      const bundleName = nameInput.value.trim() || `${config.name} bundle`;

      return {
        typename: 'TrainBundleTask',
        name: `Bundle ${bundleName}`,
        params: {
          model_config_uuid: config.uuid,
          dataset_uuids: datasetUuids,
          bundle_name: bundleName,
          hyperparams_json: hyperparamsFieldset.getHyperparamsJson(),
        },
      };
    },
    onSettled: async (finalTaskInfo) => {
      if (finalTaskInfo?.instance?.status === TaskStatus.COMPLETED) await onBundleCreated?.();
    },
  });
  wrap.appendChild(runnerEl);

  return { el: wrap };
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

  // ── Trained bundles + train-all + compose-from-artifacts (saved configs only) ──
  if (config) {
    const bundlesSection = renderBundlesSection({
      config, labelsMap,
      onComposeClick: () => composePanel.toggle(),
    });
    el.appendChild(bundlesSection.el);

    const composePanel = renderComposeBundlePanel({
      config, labelsMap,
      onCreated: () => bundlesSection.refresh(),
    });
    el.appendChild(composePanel.el);

    const trainAllSection = renderTrainAllSection({
      config,
      onBundleCreated: () => bundlesSection.refresh(),
    });
    el.appendChild(trainAllSection.el);
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
