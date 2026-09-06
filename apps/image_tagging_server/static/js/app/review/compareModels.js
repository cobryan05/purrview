/**
 * "Verify Model" — evaluate one model configuration against the same
 * verified ground truth (how accurate is it?), or two side by side (which one
 * predicts more correctly?). The second model is optional.
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
 * model is trained as part of a `TrainHierarchyTask` config. Which sub-model
 * refines which label is fixed by that sidecar pairing, not something picked
 * per run — a crop classifier only ever gets detections under its own
 * `parent_label_uuid`, no matter what else is chained alongside it. So
 * picking a "top" model in image mode doesn't offer a list of sub-models to
 * choose from; it offers one on/off switch — "use the full hierarchy" (its
 * paired sub-model(s), on by default) vs "the top detector by itself."
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
// Only whole-image-scoring kinds are excluded: both modes run a detector or a
// crop-classifier directly (image mode may additionally chain sub-models
// onto a detector), so both remain valid primary picks.
const _NON_PRIMARY_KINDS = new Set(['image_classifier', 'multi_label_classifier']);

/**
 * Entry point wired to a review-browser toolbar button. Resolves the current
 * bbox scope exactly like "Run Model on Results" does, then opens the
 * verification dialog (one model, or two side by side).
 *
 * @param {Object} ctx - same shape `runModelOnResultsFlow` takes.
 */
export async function verifyModelFlow(ctx) {
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

  _showVerifyDialog({ ctx, bboxUuids, imageUuids, primaryOptions, meta, hierarchy: _buildHierarchyIndex(meta) });
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

function _pairedSubs(modelName, meta, hierarchy) {
  const m = meta.get(modelName);
  if (m?.role !== 'top' || !m.modelConfigUuid) return [];
  return hierarchy.get(m.modelConfigUuid)?.subs ?? [];
}

/** Every label uuid this configuration can actually predict — the top model's
 * own label set plus whatever its chained sub-models add. A label absent from
 * this set can never be a correct prediction no matter how much ground truth
 * for it is in the verification set — that's a coverage gap, not an accuracy
 * problem, and the results should say so instead of just scoring it 0%. */
function _supportedLabelUuids(modelName, subModelNames, meta) {
  const set = new Set(meta.get(modelName)?.labelUuids ?? []);
  for (const sub of subModelNames) {
    for (const uuid of meta.get(sub)?.labelUuids ?? []) set.add(uuid);
  }
  return set;
}

/** Whether `uuid` is known to be outside `supportedLabelUuids` — false (i.e.
 * "don't flag it") whenever that set is empty, since an empty set means the
 * model's label list failed to load, not that it truly supports nothing. */
function _isUnsupportedLabel(uuid, supportedLabelUuids) {
  return !!supportedLabelUuids && supportedLabelUuids.size > 0 && !supportedLabelUuids.has(uuid);
}

function _buildConfigPanel(label, primaryOptions, meta, hierarchy, getCropMode, { optional = false, testidSuffix = 'a' } = {}) {
  const panel = document.createElement('div');
  panel.className = 'compare-models-config';
  panel.style.cssText = 'flex: 1; min-width: 260px; display: flex; flex-direction: column; gap: 0.5rem;';

  const heading = document.createElement('strong');
  heading.textContent = label;
  panel.appendChild(heading);

  const modelSelect = document.createElement('select');
  modelSelect.dataset.testid = `compare-models-primary-${testidSuffix}`;
  const blankOpt = document.createElement('option');
  blankOpt.value = '';
  blankOpt.textContent = optional ? '— none (verify one model) —' : '— pick a model —';
  modelSelect.appendChild(blankOpt);
  for (const name of primaryOptions) {
    const o = document.createElement('option');
    o.value = name;
    o.textContent = name + _hierarchyTag(name, meta, hierarchy);
    modelSelect.appendChild(o);
  }
  panel.appendChild(modelSelect);

  // No raw sub-model list: which sub-model refines which label is fixed by
  // the sidecar (see the module docstring), not something to pick per run —
  // the only real decision here is on/off, "the whole hierarchy" vs "the
  // top detector by itself." One checkbox says that; a multi-select of bare
  // model names, with no way to change what each one is wired to, didn't.
  const hierarchyToggleWrap = document.createElement('label');
  hierarchyToggleWrap.style.cssText = 'display: none; align-items: center; gap: 0.4rem;';
  const hierarchyToggle = document.createElement('input');
  hierarchyToggle.type = 'checkbox';
  hierarchyToggle.checked = true;
  hierarchyToggle.dataset.testid = `compare-models-use-hierarchy-${testidSuffix}`;
  const hierarchyToggleText = document.createElement('span');
  hierarchyToggleWrap.appendChild(hierarchyToggle);
  hierarchyToggleWrap.appendChild(hierarchyToggleText);
  panel.appendChild(hierarchyToggleWrap);

  function _refresh() {
    const subs = _pairedSubs(modelSelect.value, meta, hierarchy);
    const show = getCropMode() === 'image' && subs.length > 0;
    hierarchyToggleWrap.style.display = show ? 'flex' : 'none';
    if (show) hierarchyToggleText.textContent = `Use full hierarchy (+ ${subs.join(', ')})`;
  }
  // A newly picked model starts with the hierarchy on — that's the real
  // deployed behaviour, and the box to uncheck for "detector alone" instead.
  modelSelect.addEventListener('change', () => {
    hierarchyToggle.checked = true;
    _refresh();
  });
  _refresh();

  return {
    panel,
    refreshSubModelVisibility: _refresh,
    getConfig() {
      const modelName = modelSelect.value;
      const subModelNames = getCropMode() === 'image' && hierarchyToggle.checked
        ? _pairedSubs(modelName, meta, hierarchy)
        : [];
      return { modelName, subModelNames, supportedLabelUuids: _supportedLabelUuids(modelName, subModelNames, meta) };
    },
  };
}

function _showVerifyDialog({ ctx, bboxUuids, imageUuids, primaryOptions, meta, hierarchy }) {
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
  title.textContent = `Verify Model (${bboxUuids.length} verified box${bboxUuids.length === 1 ? '' : 'es'})`;
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
  const configA = _buildConfigPanel('Model', primaryOptions, meta, hierarchy, getCropMode, { testidSuffix: 'a' });
  // Second model is entirely optional — leave it unset to just verify Model
  // A's own accuracy; pick one to compare the two side by side.
  const configB = _buildConfigPanel('Compare against (optional)', primaryOptions, meta, hierarchy, getCropMode, { optional: true, testidSuffix: 'b' });
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
  runBtn.textContent = 'Verify';
  runBtn.dataset.testid = 'compare-models-run';
  footer.appendChild(cancelBtn);
  footer.appendChild(runBtn);
  dialog.appendChild(footer);

  runBtn.addEventListener('click', async () => {
    const cropMode = getCropMode();
    const a = configA.getConfig();
    const b = configB.getConfig();
    if (!a.modelName) {
      toast('Pick a model to verify.', 4000, 'warning');
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

      progressLabel.textContent = `Running ${labelA}…`;
      progressBar.value = 0;
      const resultA = await _runOneConfig({ ...a, cropMode, bboxUuids, imageUuids, confThresh, iouThresh, onProgress: p => { progressBar.value = p; } });
      resultA.supportedLabelUuids = a.supportedLabelUuids;

      let resultB = null;
      let labelB = null;
      if (b.modelName) {
        labelB = b.modelName + (b.subModelNames.length ? ` → ${b.subModelNames.join(', ')}` : '');
        progressLabel.textContent = `Running ${labelB}…`;
        progressBar.value = 0;
        resultB = await _runOneConfig({ ...b, cropMode, bboxUuids, imageUuids, confThresh, iouThresh, onProgress: p => { progressBar.value = p; } });
        resultB.supportedLabelUuids = b.supportedLabelUuids;
      }

      progressArea.style.display = 'none';
      resultsArea.appendChild(resultB
        ? _renderComparison({ labelA, labelB, resultA, resultB, labels: ctx.labels })
        : _renderSingleResult({ labelA, resultA, labels: ctx.labels }));
    } catch (err) {
      Logger.error('Model verification run failed:', err);
      toast(`Verification failed: ${err.message}`, 6000, 'error');
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
    name: `Verify Model: ${modelName}`,
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

// Must match PreviewModelRelabelTask.NO_PREDICTION_KEY exactly — the
// confusion map's sentinel "predicted label" for a box the model produced no
// attributable detection for at all (below threshold, or no match at all).
const NO_PREDICTION_KEY = '__no_prediction__';

function _labelName(labels, uuid) {
  if (uuid === NO_PREDICTION_KEY) return 'No detection';
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
      const unsupported = _isUnsupportedLabel(t, result.supportedLabelUuids);
      rowHead.textContent = unsupported ? `${_labelName(labels, t)} (not in model)` : _labelName(labels, t);
      rowHead.scope = 'row';
      rowHead.style.cssText = cellPad + (unsupported ? ' color: var(--pico-muted-color, #888); font-style: italic;' : '');
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

/** true_label_uuid -> {correct, total} — the confusion matrix's own diagonal
 * vs. row sum, i.e. this model's accuracy on just that one label. */
function _perLabelStats(confusion) {
  const stats = new Map();
  for (const [trueUuid, preds] of Object.entries(confusion)) {
    let total = 0;
    let correct = 0;
    for (const [predUuid, count] of Object.entries(preds)) {
      total += count;
      if (predUuid === trueUuid) correct += count;
    }
    stats.set(trueUuid, { correct, total });
  }
  return stats;
}

/** The actual model-vs-model comparison: one row per label, both configs'
 * accuracy on it side by side, sorted so the labels they disagree about most
 * float to the top — that's the part worth looking at first. */
function _renderPerLabelComparison(resultA, resultB, labelA, labelB, labels) {
  const statsA = _perLabelStats(resultA.confusion);
  const statsB = _perLabelStats(resultB.confusion);
  const allLabelUuids = new Set([...statsA.keys(), ...statsB.keys()]);

  const rows = [...allLabelUuids].map(uuid => {
    const a = statsA.get(uuid) ?? { correct: 0, total: 0 };
    const b = statsB.get(uuid) ?? { correct: 0, total: 0 };
    const accA = a.total > 0 ? a.correct / a.total : null;
    const accB = b.total > 0 ? b.correct / b.total : null;
    const deltaPts = accA != null && accB != null ? (accB - accA) * 100 : null;
    return {
      name: _labelName(labels, uuid), a, b, accA, accB, deltaPts,
      unsupportedA: _isUnsupportedLabel(uuid, resultA.supportedLabelUuids),
      unsupportedB: _isUnsupportedLabel(uuid, resultB.supportedLabelUuids),
    };
  });
  rows.sort((r1, r2) => {
    const d1 = r1.deltaPts == null ? -1 : Math.abs(r1.deltaPts);
    const d2 = r2.deltaPts == null ? -1 : Math.abs(r2.deltaPts);
    return d2 - d1 || r1.name.localeCompare(r2.name);
  });

  const cellPad = 'padding: 0.3rem 0.8rem; border: 1px solid var(--pico-muted-border-color, #444); text-align: left;';
  // "Not in model" isn't a wrong prediction — it's a coverage gap: the model
  // was never trained to say this label at all, so 0% there means "can't",
  // not "doesn't." Muted styling + an explicit note keeps that legible
  // instead of reading as ordinary poor accuracy.
  const fmtCell = (acc, s, unsupported) => {
    const pct = acc == null ? 'n/a' : `${(acc * 100).toFixed(0)}% (${s.correct}/${s.total})`;
    return unsupported ? `${pct} — not in model` : pct;
  };
  const cellStyle = unsupported => cellPad + (unsupported ? ' color: var(--pico-muted-color, #888); font-style: italic;' : '');

  const wrap = document.createElement('div');
  wrap.style.cssText = 'overflow-x: auto; margin: 0.75rem 0 1.25rem;';
  const table = document.createElement('table');
  table.style.cssText = 'font-size: 0.85rem; border-collapse: collapse; width: 100%;';

  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  for (const text of ['Label', labelA, labelB, 'Difference']) {
    const th = document.createElement('th');
    th.textContent = text;
    th.style.cssText = cellPad;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  for (const row of rows) {
    const tr = document.createElement('tr');

    const nameCell = document.createElement('td');
    nameCell.textContent = row.name;
    nameCell.style.cssText = cellPad;
    tr.appendChild(nameCell);

    const cellA = document.createElement('td');
    cellA.textContent = fmtCell(row.accA, row.a, row.unsupportedA);
    cellA.style.cssText = cellStyle(row.unsupportedA);
    tr.appendChild(cellA);

    const cellB = document.createElement('td');
    cellB.textContent = fmtCell(row.accB, row.b, row.unsupportedB);
    cellB.style.cssText = cellStyle(row.unsupportedB);
    tr.appendChild(cellB);

    const deltaCell = document.createElement('td');
    if (row.deltaPts == null) {
      deltaCell.textContent = 'n/a';
      deltaCell.style.cssText = cellPad;
    } else {
      const sign = row.deltaPts > 0 ? '+' : '';
      deltaCell.textContent = `${sign}${row.deltaPts.toFixed(0)} pts`;
      const emphasize = Math.abs(row.deltaPts) >= 0.5;
      deltaCell.style.cssText = cellPad
        + (emphasize ? ` font-weight: bold; color: ${row.deltaPts > 0 ? 'var(--pico-ins-color, #4caf50)' : 'var(--pico-del-color, #f44336)'};` : '');
    }
    tr.appendChild(deltaCell);

    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
  return wrap;
}

/** Single-model verification: overall accuracy, per-label breakdown (no
 * second column to diff against), and the full confusion matrix. */
function _renderSingleResult({ labelA, resultA, labels }) {
  const container = document.createElement('div');

  const pct = resultA.accuracy == null ? 'n/a' : `${(resultA.accuracy * 100).toFixed(1)}%`;
  const accSummary = document.createElement('p');
  accSummary.innerHTML = `<strong>${pct}</strong> accurate (${resultA.correct}/${resultA.total}) — `
    + `${resultA.wrong} wrong, ${resultA.noDetection} no detection`
    + (resultA.imagesFailed ? `, ${resultA.imagesFailed} image(s) failed to load` : '');
  container.appendChild(accSummary);

  const statsA = _perLabelStats(resultA.confusion);
  const rows = [...statsA.entries()]
    .map(([uuid, s]) => ({
      name: _labelName(labels, uuid), s, acc: s.total > 0 ? s.correct / s.total : null,
      unsupported: _isUnsupportedLabel(uuid, resultA.supportedLabelUuids),
    }))
    .sort((r1, r2) => r1.name.localeCompare(r2.name));

  if (rows.length > 0) {
    const cellPad = 'padding: 0.3rem 0.8rem; border: 1px solid var(--pico-muted-border-color, #444); text-align: left;';
    const wrap = document.createElement('div');
    wrap.style.cssText = 'overflow-x: auto; margin: 0.75rem 0 1.25rem;';
    const table = document.createElement('table');
    table.style.cssText = 'font-size: 0.85rem; border-collapse: collapse; width: 100%;';

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const text of ['Label', 'Accuracy']) {
      const th = document.createElement('th');
      th.textContent = text;
      th.style.cssText = cellPad;
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    for (const row of rows) {
      const tr = document.createElement('tr');
      const nameCell = document.createElement('td');
      nameCell.textContent = row.name;
      nameCell.style.cssText = cellPad;
      tr.appendChild(nameCell);

      const accCell = document.createElement('td');
      const pctText = row.acc == null ? 'n/a' : `${(row.acc * 100).toFixed(0)}% (${row.s.correct}/${row.s.total})`;
      accCell.textContent = row.unsupported ? `${pctText} — not in model` : pctText;
      accCell.style.cssText = cellPad + (row.unsupported ? ' color: var(--pico-muted-color, #888); font-style: italic;' : '');
      tr.appendChild(accCell);

      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    container.appendChild(wrap);
  }

  const details = document.createElement('details');
  const summary = document.createElement('summary');
  summary.textContent = 'Full confusion matrix';
  details.appendChild(summary);
  details.appendChild(_renderConfigResult(labelA, resultA, labels));
  container.appendChild(details);

  return container;
}

function _renderComparison({ labelA, labelB, resultA, resultB, labels }) {
  const container = document.createElement('div');

  if (resultA.accuracy != null && resultB.accuracy != null) {
    const deltaPts = (resultB.accuracy - resultA.accuracy) * 100;
    const verdict = document.createElement('p');
    if (Math.abs(deltaPts) < 0.05) {
      verdict.innerHTML = '<strong>Both models score about the same.</strong>';
    } else {
      const winner = deltaPts > 0 ? labelB : labelA;
      verdict.innerHTML = `<strong>${winner} is ${Math.abs(deltaPts).toFixed(1)} points more accurate.</strong>`;
    }
    container.appendChild(verdict);
  }

  // The actual comparison — per-label accuracy, both configs side by side,
  // biggest disagreements first. This is the part that answers "which one
  // is better," which two independent confusion matrices didn't.
  container.appendChild(_renderPerLabelComparison(resultA, resultB, labelA, labelB, labels));

  // Full confusion matrices stay available (what each model actually
  // predicted, not just whether it was right) but collapsed by default —
  // detail to check when a label's numbers surprise you, not the headline.
  const details = document.createElement('details');
  const summary = document.createElement('summary');
  summary.textContent = 'Full confusion matrices';
  details.appendChild(summary);
  const row = document.createElement('div');
  row.style.cssText = 'display: flex; gap: 1.5rem; flex-wrap: wrap; overflow-x: auto; margin-top: 0.75rem;';
  row.appendChild(_renderConfigResult(labelA, resultA, labels));
  row.appendChild(_renderConfigResult(labelB, resultB, labels));
  details.appendChild(row);
  container.appendChild(details);

  return container;
}
