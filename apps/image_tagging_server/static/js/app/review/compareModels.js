/**
 * "Compare Models" — evaluate two model configurations against the same
 * verified ground truth and show which one actually predicts more correctly.
 *
 * Deliberately narrower than "Run Model on Results": this only makes sense
 * against boxes whose current label is trustworthy (i.e. a Verified-filtered
 * scope), always runs in `crop_mode: "bbox"` (crop the known-correct box and
 * classify just that crop — the true apples-to-apples comparison; "image"
 * mode's IoU-matching conflates localisation with identification), and never
 * proposes new boxes or writes anything back — it only ever reads.
 *
 * A "configuration" is a single model name — a detector or a crop classifier,
 * run directly as `PreviewModelRelabelTask`'s `model_name`. No sub-model
 * chaining: the task itself rejects `sub_model_names` in bbox mode ("the crop
 * classifier IS the model to run"), which is the right rule here too — the
 * box is already known, so comparing "flat model" vs "detector + crop-
 * classifier hierarchy" reduces to comparing the flat model directly against
 * the hierarchy's crop-classifier stage on the same known boxes. The upstream
 * detector's own find-the-box accuracy is a separate question this scope
 * (ground-truth boxes only) isn't set up to answer.
 *
 * Accuracy comes from data `PreviewModelRelabelTask` already computes: a
 * "no-op" candidate (the model agrees with the current/ground-truth label)
 * and the `confusion` map (added alongside `no_op_count` specifically for
 * this — see that task's own docstring) together give exact per-class
 * correct/incorrect counts with zero new backend inference logic.
 */

import { fetchModelsList, fetchModelsKinds } from '/app-static/js/shared/api/models.js';
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
// Only whole-image-scoring kinds are excluded: bbox mode runs a detector or a
// crop-classifier directly on the known box, so both remain valid picks —
// see the module docstring on why bbox mode never chains a sub-model.
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

  let modelNames, kinds;
  try {
    modelNames = await fetchModelsList();
    kinds = await fetchModelsKinds(modelNames);
  } catch (err) {
    Logger.error('Failed to fetch models list for comparison:', err);
    toast(`Could not load models: ${err.message}`, 5000, 'error');
    return;
  }

  const primaryOptions = modelNames.filter(n => !_NON_PRIMARY_KINDS.has(kinds.get(n)));

  if (primaryOptions.length === 0) {
    toast('No detector or crop-classifier models available to compare.', 5000, 'warning');
    return;
  }

  _showCompareDialog({ ctx, bboxUuids, imageUuids, primaryOptions });
}

function _buildConfigPanel(label, primaryOptions) {
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
    o.textContent = name;
    modelSelect.appendChild(o);
  }
  panel.appendChild(modelSelect);

  const hint = document.createElement('small');
  hint.style.color = 'var(--pico-muted-color, #888)';
  hint.textContent = 'Pick any detector or crop-classifier model — since the box is already known, '
    + 'a crop classifier evaluates directly (no detector cascade needed here).';
  panel.appendChild(hint);

  return {
    panel,
    getConfig() {
      return { modelName: modelSelect.value };
    },
  };
}

function _showCompareDialog({ ctx, bboxUuids, imageUuids, primaryOptions }) {
  const { dialog, open, close } = createModalShell();
  dialog.style.cssText = 'width: min(900px, 92vw); max-height: 88vh; overflow-y: auto; padding: 1.25rem;';

  const title = document.createElement('h3');
  title.textContent = `Compare Models (${bboxUuids.length} verified box${bboxUuids.length === 1 ? '' : 'es'})`;
  dialog.appendChild(title);

  const hint = document.createElement('p');
  hint.innerHTML = 'Accuracy is measured against each box\'s <strong>current label</strong> — this only means something '
    + 'if the current filter is scoped to Verified boxes.';
  dialog.appendChild(hint);

  const configRow = document.createElement('div');
  configRow.style.cssText = 'display: flex; gap: 1.5rem; flex-wrap: wrap;';
  const configA = _buildConfigPanel('Configuration A', primaryOptions);
  const configB = _buildConfigPanel('Configuration B', primaryOptions);
  configRow.appendChild(configA.panel);
  configRow.appendChild(configB.panel);
  dialog.appendChild(configRow);

  const threshRow = document.createElement('div');
  threshRow.style.cssText = 'margin-top: 1rem; display: flex; align-items: center; gap: 0.5rem;';
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
      const labelA = a.modelName;
      const labelB = b.modelName;

      progressLabel.textContent = `Running Configuration A (${labelA})…`;
      progressBar.value = 0;
      const resultA = await _runOneConfig({ ...a, bboxUuids, imageUuids, confThresh, onProgress: p => { progressBar.value = p; } });

      progressLabel.textContent = `Running Configuration B (${labelB})…`;
      progressBar.value = 0;
      const resultB = await _runOneConfig({ ...b, bboxUuids, imageUuids, confThresh, onProgress: p => { progressBar.value = p; } });

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
async function _runOneConfig({ modelName, bboxUuids, imageUuids, confThresh, onProgress }) {
  const params = {
    model_name: modelName,
    crop_mode: 'bbox',
    add_new: false,
    relabel_existing: true,
    conf_thresh: confThresh,
    iou_thresh: 0.5,
    image_uuids: imageUuids,
    bbox_uuids: bboxUuids,
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
  wrap.style.cssText = 'flex: 1; min-width: 320px;';

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
    table.style.cssText = 'width: 100%; font-size: 0.85rem;';
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    headRow.appendChild(document.createElement('th'));
    for (const p of predLabels) {
      const th = document.createElement('th');
      th.textContent = _labelName(labels, p);
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
      row.appendChild(rowHead);
      for (const p of predLabels) {
        const td = document.createElement('td');
        const count = result.confusion[t][p] || 0;
        td.textContent = count || '';
        if (p === t && count) td.style.fontWeight = 'bold';
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
