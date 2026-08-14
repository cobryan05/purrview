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
import { fetchLabels } from '/app-static/js/shared/api/labels.js';
import {
  fetchTrainingConfigs,
  createTrainingConfig,
  updateTrainingConfig,
  deleteTrainingConfig,
  fetchTrainingConfigReadiness,
  fetchTrainingConfigLabelStats,
  clearTrainingConfigsCache,
} from '/app-static/js/shared/api/trainingConfigs.js';
import { renderCoverageStats, renderReadinessStats } from '/app-static/js/ui/utils/coverageStatsTable.js';
import { loadPreviousRun } from '/app-static/js/ui/utils/previousTrainingRun.js';
import { makeFieldRow, buildPanelEditorBar, confirmStartIfDirty, toast, showPaneError } from '/app-static/js/ui/utils/index.js';
import { navigateTabs } from '/app-static/js/app/utils/tabs.js';
import { activeTrainingConfig } from '/app-static/js/shared/activeTrainingConfig.js';
import { launchRun } from '/app-static/js/ui/utils/launchRun.js';
import { createHyperparamsFieldset } from '/app-static/js/ui/utils/hyperparamsFieldset.js';
import { createDatasetChecklist } from '/app-static/js/ui/utils/datasetChecklist.js';
import { events, EventTypes } from '/app-static/js/shared/events/index.js';
import { createListEditorScreen } from '/app-static/js/ui/utils/listEditorScreen.js';
import { renderModelTreePanel } from '/app-static/js/ui/modelTreePanel.js';

let _rendering = false;
let _navGuardController = null;
let _lastSelectedUuid = null;

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Muted hint line with a link that navigates to another Configuration workspace tab. */
function makeNavHint(text, linkText, workspaceTabId) {
  const hint = document.createElement('p');
  hint.style.cssText = 'font-size: 0.85em; color: var(--pico-muted-color, #888); margin: 0; padding: 2px 4px;';
  hint.append(`${text} `);
  const link = document.createElement('a');
  link.href = '#';
  link.textContent = linkText;
  link.addEventListener('click', (e) => { e.preventDefault(); navigateTabs('tab-configuration', workspaceTabId); });
  hint.appendChild(link);
  return hint;
}

/**
 * "Start run" affordance for a saved training config.
 *
 * Deliberately not a task runner: it saves any pending edits (a run of a stale
 * recipe is never what was meant), starts the run, and hands the user to Runs,
 * which owns progress, logs and cancel for everything the app runs.
 *
 * @param {Object} opts
 * @param {Object} opts.config - the saved training config
 * @param {() => boolean} opts.getDirty
 * @param {() => Promise<boolean>} opts.save
 * @returns {HTMLElement}
 */
function buildStartRunLink({ config, getDirty, save }) {
  const section = document.createElement('div');
  section.className = 'training-start-run';

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'panel-btn';
  btn.textContent = '▶ Start run';
  btn.dataset.testid = 'training-start-run';
  btn.dataset.tour = 'config.training.startRun';

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    try {
      const proceed = await confirmStartIfDirty({ getDirty, save, actionText: 'Start run' });
      if (!proceed) return;

      await launchRun({
        typename: 'TrainModelTask',
        name: `Train ${config.name}`,
        params: { training_config_uuid: config.uuid },
      });
      toast(`Started Train ${config.name}`, 3000);
      // Runs is where it can actually be watched.
      document.querySelector('.main-tab-button[data-tab="tab-runs"]')?.click();
    } catch (err) {
      toast(err.message || 'Could not start the run', 5000, 'error');
    } finally {
      btn.disabled = false;
    }
  });

  const hint = document.createElement('span');
  hint.className = 'training-start-run__hint';
  hint.textContent = 'Progress, logs and cancel live in Runs.';

  section.append(btn, hint);
  return section;
}

// ── Form ─────────────────────────────────────────────────────────────────────

function renderTrainingConfigForm({ config, datasetsMap, modelConfigsMap, labelsMap, onDirtyChange }) {
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
  configSel.addEventListener('change', () => { refreshNodeSel(); refreshTreePreview(); _markDirty(); _refreshHyperparamsArchitecture(); });
  el.appendChild(makeFieldRow('Model Shape', configSel));
  if (modelConfigsMap.size === 0) {
    el.appendChild(makeNavHint(
      'No model shapes exist yet — a training run needs one.',
      'Create one in the Model Shapes tab →', 'tab-model-management'));
  }

  // Step 2: pick the node within that config
  const nodeSel = document.createElement('select');
  nodeSel.style.cssText = 'margin-bottom: 0; width: auto; flex: none;';

  function nodeOptionLabel(node) {
    const arch = [node.modelType ?? node.modelTypename ?? '?', node.architecture].filter(Boolean).join(' / ');
    if (!node.labelUuid) {
      return `All labels (Root) — ${arch}`;
    }
    const labelName = labelsMap?.get(node.labelUuid)?.name ?? node.labelUuid;
    return `${labelName} — ${arch}`;
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
      o.textContent = nodeOptionLabel(node);
      if (node.uuid === config?.modelConfigNodeUuid) o.selected = true;
      nodeSel.appendChild(o);
    }
    nodeSel.disabled = (mc?.nodes?.length ?? 0) === 0;
  }

  refreshNodeSel();
  nodeSel.addEventListener('change', () => { _markDirty(); refreshTreePreview(); _refreshHyperparamsArchitecture(); });
  const nodeRow = makeFieldRow('Model node', nodeSel);
  const nodeHelp = document.createElement('span');
  nodeHelp.textContent = ' (?)';
  nodeHelp.title =
    'Root node ("All labels"): trains the top-level model over every label in the config. Labels that belong to a sub-model are rolled up into their parent (e.g. cat-A, cat-B count as cat). Use this for normal training.\n\n' +
    'Labeled node: trains that label’s dedicated sub-model on boxes under its label sub-tree, learning to distinguish the child labels from each other. Use this to (re)train one sub-model in isolation.';
  nodeHelp.style.cssText = 'cursor: help; opacity: 0.55; font-size: 0.85em;';
  nodeRow.querySelector('label').appendChild(nodeHelp);
  el.appendChild(nodeRow);

  // Compact preview of the chosen config's model tree, marking the target node.
  const treePreviewBox = document.createElement('div');
  treePreviewBox.className = 'mtp-box mtp-box--preview';
  el.appendChild(treePreviewBox);

  function refreshTreePreview() {
    treePreviewBox.innerHTML = '';
    const mc = modelConfigsMap.get(configSel.value);
    if (!mc?.nodes?.length) return;
    treePreviewBox.appendChild(renderModelTreePanel({
      nodes: mc.nodes,
      labelsMap,
      compact: true,
      highlightNodeUuid: nodeSel.value || null,
    }));
  }
  refreshTreePreview();

  // ── Training settings ────────────────────────────────────────────────────
  const hrSettings = document.createElement('hr');
  hrSettings.style.cssText = 'margin: 0.6em 0; opacity: 0.4;';
  el.appendChild(hrSettings);

  const settingsHdr = document.createElement('p');
  settingsHdr.className = 'panel-section-label';
  settingsHdr.style.marginTop = '0';
  settingsHdr.textContent = 'Training settings';
  el.appendChild(settingsHdr);

  /** Resolve the currently-selected node's architecture (object_detection
   * only — the fieldset's VRAM select degrades gracefully to hidden for
   * anything else), or null if no node is selected yet. */
  function _selectedArchitecture() {
    const mc = modelConfigsMap.get(configSel.value);
    const node = mc?.nodes?.find(n => n.uuid === nodeSel.value);
    return node?.modelType === 'object_detection' ? node.architecture : null;
  }

  const hyperparamsFieldset = createHyperparamsFieldset({
    initial: config?.hyperparamsJson ?? {},
    onChange: _markDirty,
    architecture: _selectedArchitecture(),
  });
  for (const row of hyperparamsFieldset.rows) el.appendChild(row);

  function _refreshHyperparamsArchitecture() {
    hyperparamsFieldset.setArchitecture(_selectedArchitecture());
  }

  // ── Datasets ──────────────────────────────────────────────────────────────
  const hr2 = document.createElement('hr');
  hr2.style.cssText = 'margin: 0.6em 0; opacity: 0.4;';
  el.appendChild(hr2);

  const dsHdr = document.createElement('p');
  dsHdr.className = 'panel-section-label';
  dsHdr.style.marginTop = '0';
  dsHdr.textContent = 'Datasets';
  el.appendChild(dsHdr);

  const datasetChecklist = createDatasetChecklist({
    datasetsMap,
    selectedUuids: config?.datasetUuids ?? [],
    onChange: _markDirty,
  });
  const datasetRow = makeFieldRow('Select ≥1', datasetChecklist.el);
  datasetRow.style.alignItems = 'flex-start';
  el.appendChild(datasetRow);

  return {
    el,
    getSaveData() {
      const datasetUuids = datasetChecklist.getSelectedUuids();
      return {
        name: nameInput.value.trim(),
        description: descInput.value.trim(),
        modelConfigNodeUuid: nodeSel.value || null,
        datasetUuids,
        hyperparamsJson: hyperparamsFieldset.getHyperparamsJson(),
      };
    },
    destroy() {},
  };
}

// ── Readiness panel ───────────────────────────────────────────────────────────

function renderReadinessPanel(config) {
  const wrap = document.createElement('div');
  wrap.style.cssText = 'margin-top: 0.75em; padding: 0.6em 0.75em; border: 1px solid var(--pico-border-color, #444); border-radius: 4px; font-size: 0.85em;';

  // ── Title row ──────────────────────────────────────────────────
  const titleRow = document.createElement('div');
  titleRow.style.cssText = 'display: flex; align-items: center; gap: 0.5em; margin-bottom: 0.4em;';

  const title = document.createElement('div');
  title.style.cssText = 'font-weight: bold;';
  title.textContent = 'Training readiness';
  titleRow.appendChild(title);

  const refreshBtn = document.createElement('button');
  refreshBtn.type = 'button';
  refreshBtn.textContent = '↻';
  refreshBtn.title = 'Refresh';
  refreshBtn.className = 'panel-btn';
  refreshBtn.style.cssText = 'font-size: 0.9em; padding: 0 5px; margin: 0; line-height: 1.4;';
  titleRow.appendChild(refreshBtn);

  wrap.appendChild(titleRow);

  // ── Content area (replaced on each refresh) ───────────────────
  const content = document.createElement('div');
  wrap.appendChild(content);

  async function _doLoad() {
    content.innerHTML = '';
    refreshBtn.disabled = true;

    const loading = document.createElement('div');
    loading.textContent = 'Loading…';
    content.appendChild(loading);

    try {
      // Both in flight together: the previous run costs two extra round trips
      // (sidecar, then stats over its manifest) and nothing about the readiness
      // numbers depends on it. loadPreviousRun resolves to null rather than
      // throwing when this config has never been trained, so it can't fail the
      // panel.
      const [readiness, previousRun] = await Promise.all([
        fetchTrainingConfigReadiness(config.uuid),
        loadPreviousRun(config),
      ]);
      content.innerHTML = '';
      renderReadinessStats(content, { config, readiness, previousRun });
    } catch (err) {
      showPaneError(content, `Could not load readiness: ${err.message}`);
    } finally {
      refreshBtn.disabled = false;
    }
  }

  refreshBtn.addEventListener('click', _doLoad);
  _doLoad();

  return wrap;
}

function renderLabelStatsPanel(config, labelsMap) {
  const wrap = document.createElement('div');
  wrap.style.cssText = 'margin-top: 0.75em; padding: 0.6em 0.75em; border: 1px solid var(--pico-border-color, #444); border-radius: 4px; font-size: 0.85em;';

  // ── Title row ──────────────────────────────────────────────────
  const titleRow = document.createElement('div');
  titleRow.style.cssText = 'display: flex; align-items: center; gap: 0.5em; margin-bottom: 0.5em;';

  const title = document.createElement('div');
  title.style.cssText = 'font-weight: bold;';
  title.textContent = 'Dataset coverage by label';
  titleRow.appendChild(title);

  const refreshBtn = document.createElement('button');
  refreshBtn.type = 'button';
  refreshBtn.textContent = '↻';
  refreshBtn.title = 'Refresh';
  refreshBtn.className = 'panel-btn';
  refreshBtn.style.cssText = 'font-size: 0.9em; padding: 0 5px; margin: 0; line-height: 1.4;';
  titleRow.appendChild(refreshBtn);

  wrap.appendChild(titleRow);

  // ── Content area (replaced on each refresh) ───────────────────
  const content = document.createElement('div');
  wrap.appendChild(content);

  async function _doLoad() {
    content.innerHTML = '';
    refreshBtn.disabled = true;

    const loading = document.createElement('div');
    loading.textContent = 'Loading…';
    content.appendChild(loading);

    try {
      const [{ labelStats, totalImages }, previousRun] = await Promise.all([
        fetchTrainingConfigLabelStats(config.uuid),
        loadPreviousRun(config),
      ]);
      content.innerHTML = '';
      renderCoverageStats(content, { config, labelStats, totalImages, labelsMap, previousRun });
    } catch (err) {
      showPaneError(content, `Could not load label stats: ${err.message}`);
    } finally {
      refreshBtn.disabled = false;
    }
  }

  refreshBtn.addEventListener('click', _doLoad);
  _doLoad();

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
    // Training Runs deliberately reuses the `mc-*` classes: it shares Model
    // Model Shapes' list/editor CSS rather than duplicating it under an `tc-` prefix.
    const screen = createListEditorScreen({
      container,
      prefix: 'mc',
      title: 'Training Runs',
      itemUuidAttr: 'configUuid',
      onNew: () => _showForm(null),
      ownTabId: 'tab-training-management',
      navGuardSignal: _navGuardController.signal,
    });
    const { listBody, editorBar, editorBody } = screen;
    const _confirmIfDirty = screen.confirmIfDirty;
    const _updateListSelection = screen.updateListSelection;

    // ── Shared state ──────────────────────────────────────────────────────
    let _selectedUuid = null;
    let _currentFormHandle = null;
    let _switching = false;

    async function _showForm(config, { force = false } = {}) {
      if (_switching && !force) return;
      _switching = true;
      try {
        if (!force && !await _confirmIfDirty()) return;
        _currentFormHandle?.destroy();
        _currentFormHandle = null;
        screen.setDoSave(null);
        screen.setBarHandle(null);
        _selectedUuid = config?.uuid ?? null;
        _lastSelectedUuid = _selectedUuid;

        const [datasetsMap, modelConfigsMap, labelsMap] = await Promise.all([
          fetchDatasets(),
          fetchModelConfigs(),
          fetchLabels().catch(() => new Map()),
        ]);
        editorBody.innerHTML = '';

        const barHandle = buildPanelEditorBar({
          container: editorBar,
          item: config,
          newTitle: 'New Training Run',
          deleteTitle: 'Delete training run',
          onSave: async () => {
            const data = _currentFormHandle.getSaveData();
            if (!data.name) { toast('Name is required', 3000, 'error'); return; }
            if (!data.modelConfigNodeUuid) { toast('A model shape node is required', 3000, 'error'); return; }
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
        screen.setBarHandle(barHandle);
        const { setDirty, setSaving } = barHandle;

        const formHandle = renderTrainingConfigForm({
          config,
          datasetsMap,
          modelConfigsMap,
          labelsMap,
          onDirtyChange: setDirty,
        });
        _currentFormHandle = formHandle;
        editorBody.appendChild(formHandle.el);

        // ── "Set as active training config" button ────────────────────────
        if (config?.uuid) {
          const activeRow = document.createElement('div');
          activeRow.style.cssText = 'display:flex;align-items:center;gap:0.5em;margin:0.5em 0 0.25em;';

          const activeBtn = document.createElement('button');
          activeBtn.type = 'button';
          activeBtn.className = 'panel-btn';
          activeBtn.style.cssText = 'margin-bottom:0;';

          function _refreshActiveBtn() {
            const current = activeTrainingConfig.getActive();
            const isActive = current?.uuid === config.uuid;
            activeBtn.textContent = isActive ? '✓ Active training run' : 'Set as active training run';
            activeBtn.style.opacity = isActive ? '0.7' : '1';
          }
          _refreshActiveBtn();

          activeBtn.addEventListener('click', async () => {
            const current = activeTrainingConfig.getActive();
            if (current?.uuid === config.uuid) {
              await activeTrainingConfig.setActive(null);
            } else {
              await activeTrainingConfig.setActive(config);
            }
            _refreshActiveBtn();
          });

          const unsubActiveBtn = events.subscribe(EventTypes.ACTIVE_TRAINING_CONFIG_CHANGED, ({ statsUpdated }) => {
            if (!statsUpdated) _refreshActiveBtn();
          });
          // Clean up subscription when form is destroyed
          const origDestroy = formHandle.destroy.bind(formHandle);
          formHandle.destroy = () => { unsubActiveBtn(); origDestroy(); };

          activeRow.appendChild(activeBtn);
          editorBody.appendChild(activeRow);
        }

        // Readiness panel (only for existing configs with datasets and a node)
        if (config?.uuid && (config.datasetUuids?.length ?? 0) > 0 && config.modelConfigNodeUuid) {
          editorBody.appendChild(renderReadinessPanel(config));
        }

        // Label stats panel (only for existing configs with datasets selected)
        if (config?.uuid && (config.datasetUuids?.length ?? 0) > 0) {
          editorBody.appendChild(renderLabelStatsPanel(config, labelsMap));
        }

        // Declared before the train section so "Start Training" can offer to run
        // it — the trainer reads the config from the server, so unsaved edits
        // would otherwise be left out of the run without a word.
        const _doSave = async () => {
          const data = formHandle.getSaveData();
          if (!data.name) { toast('Name is required', 3000, 'error'); return false; }
          if (!data.modelConfigNodeUuid) { toast('A model shape node is required', 3000, 'error'); return false; }
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
        screen.setDoSave(_doSave);

        // Start-run link (only for fully-configured existing configs).
        //
        // This used to be the whole task runner: a Start button plus a live
        // progress bar, a log tail and a Cancel, embedded in the editor. That
        // made a *configuration* screen own job monitoring, which is the same
        // miscategorisation that had task definitions filed under Operations.
        // The recipe is defined here; the run starts and is watched in Runs.
        if (config?.uuid && (config.datasetUuids?.length ?? 0) > 0 && config.modelConfigNodeUuid) {
          editorBody.appendChild(buildStartRunLink({
            config,
            getDirty: barHandle.getDirty,
            save: _doSave,
          }));
        }

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
        empty.textContent = 'No training runs yet — use the form on the right to create your first one.';
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
      return configsMap;
    }

    const configsMap = await refreshList();
    const autoConfig = _lastSelectedUuid ? configsMap.get(_lastSelectedUuid) ?? null : null;
    await _showForm(autoConfig, { force: true });
  } catch (err) {
    showPaneError(container, `Error loading training runs: ${err.message}`);
  } finally {
    _rendering = false;
  }
}
