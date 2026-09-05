/**
 * "Compare Models" — evaluate two model configurations against the same
 * verified ground truth and show which one actually predicts more correctly.
 *
 * Two evaluation modes, chosen once for the whole comparison (both sides must
 * use the same one for the numbers to mean anything against each other):
 *
 *   - "image" (default): run each configuration's detector on the whole
 *     image and match its detections to the candidate boxes by IoU — the
 *     real, deployed system's behaviour, including its own localisation. A
 *     configuration here is a top-level model **plus** any crop-classifier
 *     sub-models chained onto it (`sub_model_names`), so "all-in-one flat
 *     model" vs "detector + crop-classifier hierarchy" is a genuine face-off
 *     of the two actual systems, not an idealised shortcut.
 *   - "bbox": crop each candidate box directly and classify just that crop —
 *     no localisation involved, purest read on "given a known box, which
 *     model names it better." No sub-model chaining here: the task itself
 *     rejects `sub_model_names` in bbox mode ("the crop classifier IS the
 *     model to run"), since the box is already known.
 *
 * Every model is labelled with its place in a hierarchy (top / sub-model /
 * standalone), read from `model_config_uuid`/`role` in its own sidecar
 * (`/api/models/{name}/meta`) — the exact fields every trainer writes when a
 * model is trained as part of a `TrainHierarchyTask` config. Picking a "top"
 * model in image mode pre-selects its paired sub-model(s), so it's obvious
 * whether you're evaluating the full hierarchy or accidentally running the
 * outer detector alone.
 *
 * This never proposes new boxes or writes anything back — it only ever reads.
 * Accuracy comes from data `PreviewModelRelabelTask` already computes: a
 * "no-op" candidate (the model agrees with the current/ground-truth label)
 * and the `confusion` map (added alongside `no_op_count` specifically for
 * this — see that task's own docstring) together give exact per-class
 * correct/incorrect counts with zero new backend inference logic.
 */

import { fetchModelsList } from '/app-static/js/shared/api/models.js';
import { apiFetch } from '/app-static/js/shared/api/apiUtils.js';
import {
  createTaskConfig,
  startTask,
  fetchTasksStatus,
  fetchTasksResult,
  deleteTasks,
  TaskStatus,
} from '/app-static/js/shared/api/tasks.js';
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { toast, Logger } from '/app-static/js/ui/utils/logging.js';
import { resolveCandidateScope, confirmLargeRun, truncateScope, WARN_THRESHOLD } from '/app-static/js/app/review/runModelOnResults.js';

const POLL_INTERVAL_MS = 1000;
const _CROP_CLASSIFIER_KIND = 'crop_classifier';
// Only whole-image-scoring kinds are excluded: both modes run a detector or a
// crop-classifier directly (image mode may additionally chain sub-models
// onto a detector), so both remain valid primary picks.
const _NON_PRIMARY_KINDS = new Set(['image_classifier', 'multi_label_classifier']);

/**
 * Entry point wired to a review-browser toolbar button. Resolves the current
 * bbox scope exactly like "Run Model on Results" does, then opens the
 * two-configuration comparison dialog.
 *
 * @param {Object} ctx - same shape `runModelOnResultsFlow` takes.
 */
export async function compareModelsFlow(ctx) {
  let scope;
  try {
    scope = await resolveCandidateScope(ctx);
  } catch (err) {
    Logger.error('Failed to resolve candidate scope for model comparison:', err);
    toast(`Could not resolve the current filter: ${err.message}`, 5000, 'error');
    return;
  }

  if (scope.bboxUuids.length === 0) {
    toast('No candidate bboxes in the current filter. Filter to Verified boxes first — accuracy needs ground truth.', 6000, 'warning');
    return;
  }

  let { bboxUuids, imageUuids } = scope;
  if (bboxUuids.length > WARN_THRESHOLD) {
    const limit = await confirmLargeRun(bboxUuids.length);
    if (limit === null) return; // user cancelled
    ({ bboxUuids, imageUuids } = truncateScope(scope, limit));
  }

  let modelNames, meta;
  try {
    modelNames = await fetchModelsList();
    meta = await _fetchModelMeta(modelNames);
  } catch (err) {
    Logger.error('Failed to fetch models list for comparison:', err);
    toast(`Could not load models: ${err.message}`, 5000, 'error');
    return;
  }

  const primaryOptions = modelNames.filter(n => !_NON_PRIMARY_KINDS.has(meta.get(n)?.kind));

  if (primaryOptions.length === 0) {
    toast('No detector or crop-classifier models available to compare.', 5000, 'warning');
    return;
  }

  _showCompareDialog({ ctx, bboxUuids, imageUuids, primaryOptions, meta, hierarchy: _buildHierarchyIndex(meta) });
}

/** Fetches `/api/models/{name}/meta` for every model in parallel. */
async function _fetchModelMeta(modelNames) {
  const meta = new Map();
  await Promise.all(modelNames.map(async name => {
    try {
      const res = await apiFetch(`/api/models/${encodeURIComponent(name)}/meta`);
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.status === 'success') {
        meta.set(name, {
          kind: data.modelKind ?? 'detector',
          labelUuids: data.labelUuids ?? [],
          parentLabelUuid: data.parentLabelUuid ?? null,
          modelConfigUuid: data.modelConfigUuid ?? null,
          role: data.role ?? null,
        });
      } else {
        meta.set(name, { kind: 'detector', labelUuids: [], parentLabelUuid: null, modelConfigUuid: null, role: null });
      }
    } catch {
      meta.set(name, { kind: 'detector', labelUuids: [], parentLabelUuid: null, modelConfigUuid: null, role: null });
    }
  }));
  return meta;
}

/** modelConfigUuid -> { top: name|null, subs: string[] } — the hierarchy pairing. */
function _buildHierarchyIndex(meta) {
  const byConfig = new Map();
  for (const [name, m] of meta.entries()) {
    if (!m.modelConfigUuid || !m.role) continue;
    if (!byConfig.has(m.modelConfigUuid)) byConfig.set(m.modelConfigUuid, { top: null, subs: [] });
    const entry = byConfig.get(m.modelConfigUuid);
    if (m.role === 'top') entry.top = name;
    else if (m.role === 'submodel') entry.subs.push(name);
  }
  return byConfig;
}

/** A short suffix describing where `name` sits in a hierarchy, or '' if standalone. */
function _hierarchyTag(name, meta, hierarchy) {
  const m = meta.get(name);
  if (!m?.modelConfigUuid) return '';
  const entry = hierarchy.get(m.modelConfigUuid);
  if (m.role === 'top') {
    return entry?.subs.length ? ` — hierarchical top (+ ${entry.subs.join(', ')})` : ' — top (no sub-models found)';
  }
  if (m.role === 'submodel') {
    return entry?.top ? ` — sub-model (refines ${entry.top})` : ' — sub-model';
  }
  return '';
}

function _buildConfigPanel(label, primaryOptions, meta, hierarchy, cropClassifierOptions, getCropMode, labels) {
  const panel = document.createElement('div');
  panel.className = 'compare-models-config';
  panel.style.cssText = 'flex: 1; min-width: 260px; display: flex; flex-direction: column; gap: 0.5rem;';

  const heading = document.createElement('strong');
  heading.textContent = label;
  panel.appendChild(heading);

  const modelSelect = document.createElement('select');
  modelSelect.dataset.testid = `compare-models-primary-${label.replace(/\s+/g, '-').toLowerCase()}`;
  const blankOpt = document.createElement('option');
  blankOpt.value = '';
  blankOpt.textContent = '— pick a model —';
  modelSelect.appendChild(blankOpt);
  for (const name of primaryOptions) {
    const o = document.createElement('option');
    o.value = name;
    o.textContent = name + _hierarchyTag(name, meta, hierarchy);
    modelSelect.appendChild(o);
  }
  panel.appendChild(modelSelect);

  const subWrap = document.createElement('div');
  subWrap.style.display = 'none';
  const subLabel = document.createElement('small');
  subLabel.textContent = 'Chain crop-classifier sub-model(s) (image mode only; ctrl/cmd-click for multiple). '
    + 'Each one only ever refines the label named after it — a "Cat" detection never reaches a "Dog" sub-model, '
    + 'no matter how many are chained together.';
  subWrap.appendChild(subLabel);
  const subModelSelect = document.createElement('select');
  subModelSelect.multiple = true;
  subModelSelect.size = Math.min(4, Math.max(2, cropClassifierOptions.length));
  subModelSelect.dataset.testid = `compare-models-submodels-${label.replace(/\s+/g, '-').toLowerCase()}`;
  for (const name of cropClassifierOptions) {
    const o = document.createElement('option');
    o.value = name;
    const parentLabelUuid = meta.get(name)?.parentLabelUuid;
    const parentName = parentLabelUuid ? _labelName(labels, parentLabelUuid) : null;
    o.textContent = parentName ? `${name} (refines ${parentName})` : `${name} (no parent label — won't match anything)`;
    subModelSelect.appendChild(o);
  }
  subWrap.appendChild(subModelSelect);
  panel.appendChild(subWrap);

  function _refreshSubModelVisibility() {
    subWrap.style.display = getCropMode() === 'image' && cropClassifierOptions.length > 0 ? 'block' : 'none';
  }
  _refreshSubModelVisibility();

  // Picking a "top" model in image mode pre-selects its paired sub-model(s) —
  // this is the whole point of surfacing the hierarchy at all: it should be
  // obvious, not something the user has to remember to wire up by hand.
  modelSelect.addEventListener('change', () => {
    const m = meta.get(modelSelect.value);
    const entry = m?.modelConfigUuid ? hierarchy.get(m.modelConfigUuid) : null;
    const pairedSubs = new Set(m?.role === 'top' ? entry?.subs ?? [] : []);
    for (const opt of subModelSelect.options) opt.selected = pairedSubs.has(opt.value);
  });

  return {
    panel,
    refreshSubModelVisibility: _refreshSubModelVisibility,
    getConfig() {
      const modelName = modelSelect.value;
      const subModelNames = getCropMode() === 'image' ? [...subModelSelect.selectedOptions].map(o => o.value) : [];
      return { modelName, subModelNames };
    },
  };
}

function _showCompareDialog({ ctx, bboxUuids, imageUuids, primaryOptions, meta, hierarchy }) {
  const cropClassifierOptions = primaryOptions.filter(n => meta.get(n)?.kind === _CROP_CLASSIFIER_KIND);

  const { dialog, open, close } = createModalShell();
  // .app-modal-dialog's own max-width (--modal-w-sm, 560px) clamps any wider
  // inline `width` set at the call site — the size-modifier classes exist
  // for exactly this (see app.css's comment on .app-modal-dialog). --lg
  // (960px cap) is wide enough to show two confusion matrices side by side;
  // a matrix wider than that scrolls inside its own wrapper (see
  // _renderConfigResult) instead of squeezing every column unreadable.
  dialog.classList.add('app-modal-dialog--lg');
  dialog.style.cssText = 'overflow-y: auto;';

  const title = document.createElement('h3');
  title.textContent = `Compare Models (${bboxUuids.length} verified box${bboxUuids.length === 1 ? '' : 'es'})`;
  dialog.appendChild(title);

  const hint = document.createElement('p');
  hint.innerHTML = 'Accuracy is measured against each box\'s <strong>current label</strong> — this only means something '
    + 'if the current filter is scoped to Verified boxes.';
  dialog.appendChild(hint);

  const modeRow = document.createElement('div');
  modeRow.style.cssText = 'margin-bottom: 0.75rem;';
  const modeLabel = document.createElement('label');
  modeLabel.style.cssText = 'display: flex; align-items: center; gap: 0.5rem;';
  modeLabel.textContent = 'Evaluation mode:';
  const modeSelect = document.createElement('select');
  modeSelect.dataset.testid = 'compare-models-crop-mode';
  for (const [value, text] of [
    ['image', 'Full pipeline — detect on the whole image, match to the box by overlap (real deployed behaviour)'],
    ['bbox', 'Identification only — classify the already-known box directly (no localisation)'],
  ]) {
    const o = document.createElement('option');
    o.value = value;
    o.textContent = text;
    modeSelect.appendChild(o);
  }
  modeLabel.appendChild(modeSelect);
  modeRow.appendChild(modeLabel);
  dialog.appendChild(modeRow);

  const getCropMode = () => modeSelect.value;

  const configRow = document.createElement('div');
  configRow.style.cssText = 'display: flex; gap: 1.5rem; flex-wrap: wrap;';
  const configA = _buildConfigPanel('Configuration A', primaryOptions, meta, hierarchy, cropClassifierOptions, getCropMode, ctx.labels);
  const configB = _buildConfigPanel('Configuration B', primaryOptions, meta, hierarchy, cropClassifierOptions, getCropMode, ctx.labels);
  configRow.appendChild(configA.panel);
  configRow.appendChild(configB.panel);
  dialog.appendChild(configRow);

  modeSelect.addEventListener('change', () => {
    configA.refreshSubModelVisibility();
    configB.refreshSubModelVisibility();
  });

  const threshRow = document.createElement('div');
  threshRow.style.cssText = 'margin-top: 1rem; display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap;';
  const threshLabel = document.createElement('label');
  threshLabel.textContent = 'Confidence threshold:';
  threshLabel.style.cssText = 'display: flex; align-items: center; gap: 0.5rem;';
  const threshInput = document.createElement('input');
  threshInput.type = 'number';
  threshInput.min = '0';
  threshInput.max = '1';
  threshInput.step = '0.05';
  threshInput.value = '0.25';
  threshInput.style.width = '6rem';
  threshLabel.appendChild(threshInput);
  threshRow.appendChild(threshLabel);

  const iouLabel = document.createElement('label');
  iouLabel.textContent = 'IoU threshold (image mode):';
  iouLabel.style.cssText = 'display: flex; align-items: center; gap: 0.5rem;';
  const iouInput = document.createElement('input');
  iouInput.type = 'number';
  iouInput.min = '0';
  iouInput.max = '1';
  iouInput.step = '0.05';
  iouInput.value = '0.5';
  iouInput.style.width = '6rem';
  iouLabel.appendChild(iouInput);
  threshRow.appendChild(iouLabel);
  dialog.appendChild(threshRow);

  const progressArea = document.createElement('div');
  progressArea.style.cssText = 'margin-top: 1rem; display: none;';
  const progressLabel = document.createElement('div');
  const progressBar = document.createElement('progress');
  progressBar.max = 100;
  progressBar.style.width = '100%';
  progressArea.appendChild(progressLabel);
  progressArea.appendChild(progressBar);
  dialog.appendChild(progressArea);

  const resultsArea = document.createElement('div');
  resultsArea.style.marginTop = '1rem';
  dialog.appendChild(resultsArea);

  const footer = document.createElement('div');
  footer.style.cssText = 'margin-top: 1.25rem; display: flex; justify-content: flex-end; gap: 0.5rem;';
  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Close';
  cancelBtn.className = 'secondary';
  cancelBtn.addEventListener('click', close);
  const runBtn = document.createElement('button');
  runBtn.textContent = 'Compare';
  runBtn.dataset.testid = 'compare-models-run';
  footer.appendChild(cancelBtn);
  footer.appendChild(runBtn);
  dialog.appendChild(footer);

  runBtn.addEventListener('click', async () => {
    const cropMode = getCropMode();
    const a = configA.getConfig();
    const b = configB.getConfig();
    if (!a.modelName || !b.modelName) {
      toast('Pick a model for both configurations.', 4000, 'warning');
      return;
    }

    runBtn.disabled = true;
    cancelBtn.disabled = true;
    progressArea.style.display = 'block';
    resultsArea.innerHTML = '';

    try {
      const confThresh = Math.min(1, Math.max(0, parseFloat(threshInput.value) || 0.25));
      const iouThresh = Math.min(1, Math.max(0, parseFloat(iouInput.value) || 0.5));
      const labelA = a.modelName + (a.subModelNames.length ? ` → ${a.subModelNames.join(', ')}` : '');
      const labelB = b.modelName + (b.subModelNames.length ? ` → ${b.subModelNames.join(', ')}` : '');

      progressLabel.textContent = `Running Configuration A (${labelA})…`;
      progressBar.value = 0;
      const resultA = await _runOneConfig({ ...a, cropMode, bboxUuids, imageUuids, confThresh, iouThresh, onProgress: p => { progressBar.value = p; } });

      progressLabel.textContent = `Running Configuration B (${labelB})…`;
      progressBar.value = 0;
      const resultB = await _runOneConfig({ ...b, cropMode, bboxUuids, imageUuids, confThresh, iouThresh, onProgress: p => { progressBar.value = p; } });

      progressArea.style.display = 'none';
      resultsArea.appendChild(_renderComparison({ labelA, labelB, resultA, resultB, labels: ctx.labels }));
    } catch (err) {
      Logger.error('Model comparison run failed:', err);
      toast(`Comparison failed: ${err.message}`, 6000, 'error');
      progressArea.style.display = 'none';
    } finally {
      runBtn.disabled = false;
      cancelBtn.disabled = false;
    }
  });

  open();
}

/** Launches one PreviewModelRelabelTask run and returns its parsed stats. */
async function _runOneConfig({ modelName, subModelNames, cropMode, bboxUuids, imageUuids, confThresh, iouThresh, onProgress }) {
  const params = {
    model_name: modelName,
    crop_mode: cropMode,
    add_new: false,
    relabel_existing: true,
    conf_thresh: confThresh,
    iou_thresh: iouThresh,
    image_uuids: imageUuids,
    bbox_uuids: bboxUuids,
    // Only meaningful (and only accepted by the task) in image mode — see
    // the module docstring on why bbox mode never chains a sub-model.
    ...(cropMode === 'image' && subModelNames.length ? { sub_model_names: subModelNames } : {}),
  };

  const configUuid = await createTaskConfig({
    typename: 'PreviewModelRelabelTask',
    name: `Compare Models: ${modelName}`,
    params,
    persistent: false,
    isTemporary: true,
  });
  const taskUuid = await startTask({ configUuid });

  let info;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { tasks } = await fetchTasksStatus({ taskUuids: [taskUuid] });
    info = tasks?.[taskUuid];
    onProgress?.(info?.progress ?? 0);
    const status = info?.instance?.status;
    if (status === TaskStatus.COMPLETED || status === TaskStatus.ERROR || status === TaskStatus.CANCELLED) break;
    await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
  }

  const status = info?.instance?.status;
  if (status !== TaskStatus.COMPLETED) {
    await deleteTasks({ taskUuids: [taskUuid] }).catch(() => {});
    throw new Error(info?.instance?.errorMessage || `Task for ${modelName} ended with status: ${status}`);
  }

  const resultsMap = await fetchTasksResult({ taskUuids: [taskUuid], cacheResults: false });
  const data = resultsMap.get(taskUuid)?.data ?? {};
  await deleteTasks({ taskUuids: [taskUuid] }).catch(e => Logger.error('Cleanup failed', e));

  const noOp = data.no_op_count ?? 0;
  const wrong = (data.candidates ?? []).length;
  const noDetection = data.no_detection_count ?? 0;
  const total = noOp + wrong + noDetection;
  return {
    accuracy: total > 0 ? noOp / total : null,
    correct: noOp,
    wrong,
    noDetection,
    total,
    confusion: data.confusion ?? {},
    imagesFailed: data.images_failed ?? 0,
  };
}

function _labelName(labels, uuid) {
  return labels?.get(uuid)?.name ?? uuid ?? '(none)';
}

/** One config's accuracy summary + confusion-matrix table. */
function _renderConfigResult(label, result, labels) {
  const wrap = document.createElement('div');
  wrap.style.cssText = 'flex: 1 1 420px; min-width: 320px; max-width: 100%; overflow-x: auto;';

  const heading = document.createElement('h4');
  heading.textContent = label;
  wrap.appendChild(heading);

  const pct = result.accuracy == null ? 'n/a' : `${(result.accuracy * 100).toFixed(1)}%`;
  const summary = document.createElement('p');
  summary.innerHTML = `<strong>${pct}</strong> accurate (${result.correct}/${result.total}) — `
    + `${result.wrong} wrong, ${result.noDetection} no detection`
    + (result.imagesFailed ? `, ${result.imagesFailed} image(s) failed to load` : '');
  wrap.appendChild(summary);

  // Confusion matrix: rows = true label, cols = predicted label.
  const trueLabels = Object.keys(result.confusion);
  if (trueLabels.length > 0) {
    const predLabelSet = new Set();
    for (const t of trueLabels) for (const p of Object.keys(result.confusion[t])) predLabelSet.add(p);
    const predLabels = [...predLabelSet];

    const table = document.createElement('table');
    // No fixed width: with many labels this table is wider than its column —
    // the wrapping div's overflow-x lets it scroll instead of squeezing every
    // cell down to unreadable widths.
    table.style.cssText = 'font-size: 0.85rem; white-space: nowrap; border-collapse: collapse;';
    const cellPad = 'padding: 0.2rem 0.6rem; border: 1px solid var(--pico-muted-border-color, #444);';

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    headRow.appendChild(document.createElement('th'));
    for (const p of predLabels) {
      const th = document.createElement('th');
      th.textContent = _labelName(labels, p);
      th.style.cssText = cellPad;
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    for (const t of trueLabels) {
      const row = document.createElement('tr');
      const rowHead = document.createElement('th');
      rowHead.textContent = _labelName(labels, t);
      rowHead.scope = 'row';
      rowHead.style.cssText = cellPad;
      row.appendChild(rowHead);
      for (const p of predLabels) {
        const td = document.createElement('td');
        const count = result.confusion[t][p] || 0;
        td.textContent = count || '';
        td.style.cssText = cellPad + (p === t && count ? ' font-weight: bold;' : '');
        row.appendChild(td);
      }
      tbody.appendChild(row);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
  }

  return wrap;
}

function _renderComparison({ labelA, labelB, resultA, resultB, labels }) {
  const container = document.createElement('div');

  if (resultA.accuracy != null && resultB.accuracy != null) {
    const deltaPts = (resultB.accuracy - resultA.accuracy) * 100;
    const verdict = document.createElement('p');
    if (Math.abs(deltaPts) < 0.05) {
      verdict.innerHTML = '<strong>Configurations score about the same.</strong>';
    } else {
      const winner = deltaPts > 0 ? 'B' : 'A';
      verdict.innerHTML = `<strong>Configuration ${winner} is ${Math.abs(deltaPts).toFixed(1)} points more accurate.</strong>`;
    }
    container.appendChild(verdict);
  }

  const row = document.createElement('div');
  row.style.cssText = 'display: flex; gap: 1.5rem; flex-wrap: wrap; overflow-x: auto;';
  row.appendChild(_renderConfigResult(labelA, resultA, labels));
  row.appendChild(_renderConfigResult(labelB, resultB, labels));
  container.appendChild(row);

  return container;
}
