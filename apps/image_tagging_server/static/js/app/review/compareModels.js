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
 * Either side can be an **archived version** of its model instead of the live
 * one, which is what makes the version history worth keeping: "did the retrain
 * actually help?" is this dialog's question asked across time rather than across
 * models. A version is not a model — it lives outside the models directory and
 * nothing can load it by name — so picking one materialises a temporary copy
 * (`<model>@<versionId>`, see `apps/helpers/modelVersions.py`) that runs like any
 * other model and is deleted when the dialog closes. Sub-model chaining is off for
 * a version: the sub-models on disk are the current ones, and pairing an archived
 * detector with them would compare a hierarchy that never existed.
 *
 * This never proposes new boxes or writes anything back — it only ever reads.
 * Accuracy comes from data `PreviewModelRelabelTask` already computes: a
 * "no-op" candidate (the model agrees with the current/ground-truth label)
 * and the `confusion` map (added alongside `no_op_count` specifically for
 * this — see that task's own docstring) together give exact per-class
 * correct/incorrect counts with zero new backend inference logic.
 */

import {
  fetchModelsList,
  fetchModelVersions,
  materializeModelVersion,
  deleteModelPreview,
} from '/app-static/js/shared/api/models.js';
import { parseApiDate } from '/app-static/js/shared/time.js';
import { apiFetch } from '/app-static/js/shared/api/apiUtils.js';
import {
  createTaskConfig,
  startTask,
  fetchTasksStatus,
  fetchTasksResult,
  deleteTasks,
  TaskStatus,
} from '/app-static/js/shared/api/tasks.js';
import {
  createPreviewRun,
  fetchPreviewRun,
  fetchPreviewRuns,
  REPORT_PARAM,
} from '/app-static/js/shared/api/previewRuns.js';
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { toast, Logger } from '/app-static/js/ui/utils/logging.js';
import { resolveCandidateScope, confirmLargeRun, truncateScope, WARN_THRESHOLD } from '/app-static/js/app/review/runModelOnResults.js';

const POLL_INTERVAL_MS = 1000;
// Consecutive polls in which a started task is absent from the status map before
// the run is declared lost — see the wait loop in `_runOneConfig`.
const MAX_MISSING_POLLS = 5;
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

/** "Trained 3 Sep 2026, 10:15" for a version option, so the list reads as dates. */
function _versionOptionText(version) {
  const d = parseApiDate(version.createdAt);
  return d ? `Trained ${d.toLocaleString()}` : `Version ${version.versionId}`;
}

function _buildConfigPanel(label, primaryOptions, meta, hierarchy, getCropMode, { optional = false, testidSuffix = 'a', onPreviewCreated = () => {} } = {}) {
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

  // Which generation of that model. Hidden until the chosen model actually has
  // history — most won't, and an empty "Current" dropdown is just noise.
  const versionSelect = document.createElement('select');
  versionSelect.dataset.testid = `compare-models-version-${testidSuffix}`;
  versionSelect.style.display = 'none';

  const versionNote = document.createElement('small');
  versionNote.dataset.testid = `compare-models-version-note-${testidSuffix}`;
  versionNote.style.cssText = 'display: none; opacity: 0.75;';
  versionNote.textContent = 'Archived versions run on their own — the sub-models on disk belong to the '
    + 'current model, not to this generation.';

  panel.appendChild(versionSelect);
  panel.appendChild(versionNote);

  // The name the run actually uses: the model itself, or the temporary copy of an
  // archived generation. Everything downstream (meta, sub-model pairing, the task)
  // takes a model name, so resolving the choice to one name here keeps the version
  // case from leaking into any of it.
  let effectiveName = '';
  let previewName = null;

  /** Materialise the picked version (or drop back to the live model). */
  async function _applyVersionChoice() {
    const modelName = modelSelect.value;
    const versionId = versionSelect.value;

    if (previewName) {
      // The panel never deletes: both panels can be showing the same version of
      // the same model, and the server hands them one shared copy. Deleting on
      // release would pull it out from under the other side mid-run, so the
      // dialog refcounts and deletes when the last holder lets go.
      onPreviewCreated(previewName, 'released');
      previewName = null;
    }

    if (!modelName || !versionId) {
      effectiveName = modelName;
      versionNote.style.display = 'none';
      return;
    }

    versionSelect.disabled = true;
    try {
      previewName = await materializeModelVersion(modelName, versionId);
      onPreviewCreated(previewName, 'created');
      // The preview carries the sidecar it was trained with, so its label set and
      // kind are read from it rather than assumed to match the live model's —
      // a class list that changed between generations is exactly what a
      // comparison is looking for.
      const previewMeta = await _fetchModelMeta([previewName]);
      meta.set(previewName, previewMeta.get(previewName));
      effectiveName = previewName;
      versionNote.style.display = 'block';
    } catch (err) {
      toast(err.message || 'Could not load that version', 5000, 'error');
      versionSelect.value = '';
      effectiveName = modelName;
      versionNote.style.display = 'none';
    } finally {
      versionSelect.disabled = false;
    }
  }

  /** Refill the version list for the currently selected model. */
  async function _loadVersions() {
    versionSelect.innerHTML = '';
    const modelName = modelSelect.value;
    if (!modelName) {
      versionSelect.style.display = 'none';
      return;
    }

    let versions = [];
    try {
      versions = await fetchModelVersions(modelName);
    } catch (err) {
      // A model whose history can't be read is still comparable as it is now.
      Logger.warn('Could not load versions for', modelName, err);
    }
    if (modelSelect.value !== modelName) return; // selection moved on while fetching

    if (versions.length === 0) {
      versionSelect.style.display = 'none';
      return;
    }

    const live = document.createElement('option');
    live.value = '';
    live.textContent = 'Current version (live)';
    versionSelect.appendChild(live);
    for (const version of versions) {
      const o = document.createElement('option');
      o.value = version.versionId;
      o.textContent = _versionOptionText(version);
      versionSelect.appendChild(o);
    }
    versionSelect.style.display = '';
  }

  versionSelect.addEventListener('change', async () => {
    await _applyVersionChoice();
    _refresh();
  });

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
    // An archived generation never chains: `_pairedSubs` would pair it with the
    // *current* sub-models via its sidecar's config uuid, which is a hierarchy
    // that never ran. The note beside the picker says so.
    const subs = versionSelect.value ? [] : _pairedSubs(modelSelect.value, meta, hierarchy);
    const show = getCropMode() === 'image' && subs.length > 0;
    hierarchyToggleWrap.style.display = show ? 'flex' : 'none';
    if (show) hierarchyToggleText.textContent = `Use full hierarchy (+ ${subs.join(', ')})`;
  }
  // A newly picked model starts with the hierarchy on — that's the real
  // deployed behaviour, and the box to uncheck for "detector alone" instead.
  modelSelect.addEventListener('change', async () => {
    hierarchyToggle.checked = true;
    effectiveName = modelSelect.value;
    await _loadVersions();
    await _applyVersionChoice();
    _refresh();
  });
  _refresh();

  return {
    panel,
    refreshSubModelVisibility: _refresh,
    getConfig() {
      const modelName = effectiveName || modelSelect.value;
      const isVersion = Boolean(versionSelect.value);
      const subModelNames = !isVersion && getCropMode() === 'image' && hierarchyToggle.checked
        ? _pairedSubs(modelName, meta, hierarchy)
        : [];
      return {
        modelName,
        subModelNames,
        supportedLabelUuids: _supportedLabelUuids(modelName, subModelNames, meta),
        // What to call it in the results: `<model>@<timestamp>` is a filename, not
        // something anyone picked, and the two sides of a same-model comparison are
        // otherwise indistinguishable in the output.
        displayName: isVersion
          ? `${modelSelect.value} — ${versionSelect.selectedOptions[0]?.textContent ?? versionSelect.value}`
          : modelName,
      };
    },
  };
}

function _showVerifyDialog({ ctx, bboxUuids, imageUuids, primaryOptions, meta, hierarchy }) {
  // Version previews are full copies of a model's weights. They are deleted as
  // soon as a panel stops using one, and this set is the backstop for the rest:
  // closing the dialog mid-comparison, or with a version still selected.
  // name -> how many panels are holding it. Two panels comparing the same version
  // of the same model share one materialised copy (the server is idempotent), so
  // this is a refcount, not a set: whoever lets go last does the deleting.
  const livePreviews = new Map();
  const _releasePreview = (name) => {
    deleteModelPreview(name).catch(err => Logger.warn('Preview cleanup failed:', err));
  };
  const _trackPreview = (name, action) => {
    const held = livePreviews.get(name) ?? 0;
    if (action === 'created') {
      livePreviews.set(name, held + 1);
      return;
    }
    if (held <= 1) {
      livePreviews.delete(name);
      _releasePreview(name);
    } else {
      livePreviews.set(name, held - 1);
    }
  };

  const { dialog, open, close } = createModalShell({
    onClose: () => {
      for (const name of livePreviews.keys()) _releasePreview(name);
      livePreviews.clear();
    },
  });
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
  const configA = _buildConfigPanel('Model', primaryOptions, meta, hierarchy, getCropMode, {
    testidSuffix: 'a', onPreviewCreated: _trackPreview,
  });
  // Second model is entirely optional — leave it unset to just verify Model
  // A's own accuracy; pick one to compare the two side by side.
  const configB = _buildConfigPanel('Compare against (optional)', primaryOptions, meta, hierarchy, getCropMode, {
    optional: true, testidSuffix: 'b', onPreviewCreated: _trackPreview,
  });
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
  // Filled in once the runs exist, because what it can promise depends on
  // whether they were saved and on whether an archived version is involved.
  const savedNote = document.createElement('div');
  savedNote.dataset.testid = 'compare-models-saved-note';
  savedNote.style.cssText = 'display: none; margin-bottom: 0.5rem; font-size: 0.9em;';
  progressArea.appendChild(savedNote);
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
      const labelA = _configLabel(a);
      const labelB = b.modelName ? _configLabel(b) : null;

      // Both runs are created before either starts, so the second one is already
      // on record if the page goes away while the first is still running.
      const group = `verify-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const scope = { bbox_uuid_count: bboxUuids.length, image_uuid_count: imageUuids.length };
      const runA = await _createReportRun({ label: labelA, cfg: a, cropMode, confThresh, iouThresh, scope, group, slot: 'A' });
      const runB = labelB
        ? await _createReportRun({ label: labelB, cfg: b, cropMode, confThresh, iouThresh, scope, group, slot: 'B' })
        : null;
      _showSavedNote(savedNote, [runA, runB].filter(Boolean), livePreviews.size > 0);

      progressLabel.textContent = `Running ${labelA}…`;
      progressBar.value = 0;
      const resultA = await _runOneConfig({ ...a, cropMode, bboxUuids, imageUuids, confThresh, iouThresh, previewRunUuid: runA?.uuid ?? null, onProgress: p => { progressBar.value = p; } });

      let resultB = null;
      if (labelB) {
        progressLabel.textContent = `Running ${labelB}…`;
        progressBar.value = 0;
        resultB = await _runOneConfig({ ...b, cropMode, bboxUuids, imageUuids, confThresh, iouThresh, previewRunUuid: runB?.uuid ?? null, onProgress: p => { progressBar.value = p; } });
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

/** "model → sub, sub" — what a configuration is called in the results. */
function _configLabel(cfg) {
  return (cfg.displayName ?? cfg.modelName) + (cfg.subModelNames.length ? ` → ${cfg.subModelNames.join(', ')}` : '');
}

/**
 * Register one configuration's report before its task starts.
 *
 * The run is what makes the result outlive this dialog: the task writes its
 * summary into `stats` as it finishes, so a verification survives closing the
 * page that started it. Returns null if the run could not be created — the
 * comparison still runs, it just won't be saved.
 */
async function _createReportRun({ label, cfg, cropMode, confThresh, iouThresh, scope, group, slot }) {
  try {
    return await createPreviewRun({
      name: `Verify: ${label}`,
      modelName: cfg.modelName,
      taskTypename: 'PreviewModelRelabelTask',
      params: {
        [REPORT_PARAM]: true,
        display_label: label,
        crop_mode: cropMode,
        conf_thresh: confThresh,
        iou_thresh: iouThresh,
        sub_model_names: cfg.subModelNames,
        // Array, not the Set the dialog carries: this round-trips through JSON.
        supported_label_uuids: [...(cfg.supportedLabelUuids ?? [])],
        comparison_group: group,
        slot,
      },
      scope,
    });
  } catch (err) {
    Logger.error('Failed to create verification run:', err);
    toast('Could not save this verification; its results will be lost if you close this window.', 6000, 'warning');
    return null;
  }
}

/** Tell the user whether they still have to sit here while the run finishes. */
function _showSavedNote(noteEl, runs, holdsVersionPreview) {
  noteEl.style.display = 'block';
  if (runs.length === 0) {
    noteEl.textContent = '⚠ This run is not being saved — keep this window open or the results are lost.';
    return;
  }
  const names = runs.map(r => `“${r.name}”`).join(' and ');
  if (holdsVersionPreview) {
    // Closing the dialog releases the materialised version copy, and the task
    // then fails on a model that no longer exists.
    noteEl.textContent = `Saving as ${names}. This run uses an archived version, so keep this window open until it finishes — closing it deletes the temporary copy the run needs.`;
    return;
  }
  noteEl.textContent = `✓ Saving as ${names} — you can close this window. The run keeps going and the report is waiting under “Model Run Results” in the review toolbar.`;
}

/** Launches one PreviewModelRelabelTask run and returns its parsed stats. */
async function _runOneConfig({ modelName, subModelNames, supportedLabelUuids, cropMode, bboxUuids, imageUuids, confThresh, iouThresh, previewRunUuid, onProgress }) {
  const params = {
    model_name: modelName,
    crop_mode: cropMode,
    add_new: false,
    relabel_existing: true,
    conf_thresh: confThresh,
    iou_thresh: iouThresh,
    image_uuids: imageUuids,
    bbox_uuids: bboxUuids,
    // A verification's "candidates" are its disagreements with ground truth, so
    // they are counted and never persisted: offered as triagable relabels they
    // would propose overwriting the very labels being measured against.
    ...(previewRunUuid ? { preview_run_uuid: previewRunUuid, persist_candidates: false, persist_stats: true } : {}),
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
  // A task that stops appearing in the status map has no terminal status to
  // report, so "keep polling until it's terminal" would spin here forever with
  // the dialog stuck on its progress bar — reachable whenever the instance goes
  // away underneath us (another surface cleared it, tasks_server restarted).
  // Tolerate a few missing polls (the instance can lag a beat behind startTask)
  // and then give up with a real error instead.
  let missingPolls = 0;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { tasks } = await fetchTasksStatus({ taskUuids: [taskUuid] });
    info = tasks?.[taskUuid];
    onProgress?.(info?.progress ?? 0);
    const status = info?.instance?.status;
    if (status === TaskStatus.COMPLETED || status === TaskStatus.ERROR || status === TaskStatus.CANCELLED) break;
    if (status === undefined) {
      missingPolls += 1;
      if (missingPolls > MAX_MISSING_POLLS) {
        throw new Error(`Task for ${modelName} disappeared before it finished`);
      }
    } else {
      missingPolls = 0;
    }
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

  return _resultFromStats(data, supportedLabelUuids);
}

/**
 * Score one run's raw counts.
 *
 * Takes the task result or a saved run's `stats` — they are the same dict,
 * which is the whole reason a report reads back identically to a live run.
 */
function _resultFromStats(data, supportedLabelUuids = null) {
  const noOp = data.no_op_count ?? 0;
  // `candidate_count`, not `candidates.length`: the task empties `candidates`
  // out of its result whenever the run is persisted to a preview run, and a
  // zero-length list there would read as "nothing wrong" — i.e. 100% accurate.
  const wrong = data.candidate_count ?? (data.candidates ?? []).length;
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
    supportedLabelUuids: supportedLabelUuids ?? null,
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

// ── Saved reports ──────────────────────────────────────────────────────────
//
// A verification is a measurement, not a proposal: it has no candidates to
// triage, so a saved run carries its whole result in `stats` and the triage
// panel has nothing to show for it. These two exports are what the Saved
// Previews list uses instead — a one-line summary for the row, and this
// read-only rebuild of the same report the dialog rendered live.

const _REPORT_POLL_MS = 2000;

/** One line for a saved report's row in the Model Run Results list. */
export function reportRowSummary(run) {
  const stats = run?.stats ?? {};
  if (!stats.state) return 'never started';
  if (stats.state === 'running') return 'still running…';
  const result = _resultFromStats(stats);
  const pct = result.accuracy == null ? 'no scorable boxes' : `${(result.accuracy * 100).toFixed(1)}% accurate`;
  const detail = `${result.correct}/${result.total}`;
  if (stats.state === 'completed') return `${pct} (${detail})`;
  // cancelled / interrupted: the numbers are real but cover only part of the scope.
  return `${stats.state} — ${pct} so far (${detail})`;
}

function _reportResult(run) {
  const supported = run.params?.supported_label_uuids;
  return _resultFromStats(run.stats ?? {}, Array.isArray(supported) ? new Set(supported) : null);
}

function _reportLabel(run) {
  return run.params?.display_label || run.modelName || run.name;
}

/**
 * Re-open a saved verification. Still-running runs are polled to completion
 * here, which is what makes closing the dialog mid-run safe in the first place.
 */
export async function openSavedVerifyReport({ run, labels = null }) {
  // ESC and a backdrop click dismiss the modal too, so the poll loop below has
  // to key off the shell's own teardown rather than the Close button.
  let closed = false;
  const { dialog, open, close } = createModalShell({ onClose: () => { closed = true; } });
  dialog.classList.add('app-modal-dialog--lg');
  dialog.style.cssText = 'overflow-y: auto;';

  const title = document.createElement('h3');
  title.textContent = run.name;
  dialog.appendChild(title);

  const meta = document.createElement('p');
  meta.style.cssText = 'font-size: 0.85em; color: var(--pico-muted-color, #888);';
  const p = run.params ?? {};
  const scope = run.scope ?? {};
  meta.textContent = [
    p.crop_mode === 'bbox' ? 'identification only' : 'full pipeline',
    `conf ≥ ${p.conf_thresh ?? '?'}`,
    p.crop_mode === 'bbox' ? null : `IoU ≥ ${p.iou_thresh ?? '?'}`,
    `${scope.bbox_uuid_count ?? 0} box(es)`,
  ].filter(Boolean).join(' · ');
  dialog.appendChild(meta);

  const body = document.createElement('div');
  body.dataset.testid = 'verify-report-body';
  body.style.marginTop = '1rem';
  dialog.appendChild(body);

  const footer = document.createElement('div');
  footer.style.cssText = 'margin-top: 1.25rem; display: flex; justify-content: flex-end;';
  const closeBtn = document.createElement('button');
  closeBtn.className = 'secondary';
  closeBtn.textContent = 'Close';
  closeBtn.addEventListener('click', close);
  footer.appendChild(closeBtn);
  dialog.appendChild(footer);

  open();

  let current = run;
  if (current.stats?.state === 'running') {
    current = await _awaitReport(current, body, () => closed);
    if (closed) return;
  }

  body.innerHTML = '';
  if (!current.stats?.state) {
    body.textContent = 'This verification never started, so there is nothing to show.';
    return;
  }
  if (current.stats.state === 'interrupted') {
    const warn = document.createElement('p');
    warn.textContent = '⚠ This run stopped before it finished — the numbers below cover only the boxes it got through.';
    body.appendChild(warn);
  }

  // A two-model comparison is two runs sharing a `comparison_group`; rendering
  // the pair reproduces the side-by-side verdict the dialog showed live.
  const sibling = await _findSibling(current);
  const labelA = _reportLabel(current);
  if (sibling && sibling.stats?.state === 'completed' && current.stats.state === 'completed') {
    const [first, second] = current.params?.slot === 'B' ? [sibling, current] : [current, sibling];
    body.appendChild(_renderComparison({
      labelA: _reportLabel(first),
      labelB: _reportLabel(second),
      resultA: _reportResult(first),
      resultB: _reportResult(second),
      labels,
    }));
    return;
  }
  body.appendChild(_renderSingleResult({ labelA, resultA: _reportResult(current), labels }));
}

/** The other half of a two-model comparison, if it is saved and readable. */
async function _findSibling(run) {
  const group = run.params?.comparison_group;
  if (!group) return null;
  try {
    const runs = await fetchPreviewRuns();
    return runs.find(r => r.uuid !== run.uuid && r.params?.comparison_group === group) ?? null;
  } catch (err) {
    Logger.error('Failed to look up the other half of this comparison:', err);
    return null;
  }
}

/**
 * Watch a run that is still in flight. The run's own `stats.state` is the only
 * authority on being finished — the task writes it as it exits — so a task that
 * vanished (tasks_server restarted under it) is detected by its absence from
 * the status map rather than by waiting for a terminal status it will never get.
 */
async function _awaitReport(run, body, isClosed) {
  body.innerHTML = '';
  const label = document.createElement('div');
  label.textContent = 'Still running — this picks up where it left off.';
  const bar = document.createElement('progress');
  bar.max = 100;
  bar.style.width = '100%';
  body.append(label, bar);

  const taskUuid = run.stats?.task_uuid;
  let missingPolls = 0;
  let current = run;
  while (!isClosed()) {
    if (taskUuid) {
      try {
        const { tasks } = await fetchTasksStatus({ taskUuids: [taskUuid] });
        const info = tasks?.[taskUuid];
        if (info) {
          missingPolls = 0;
          bar.value = info.progress ?? 0;
        } else {
          missingPolls += 1;
        }
      } catch (err) {
        Logger.warn('Report progress poll failed:', err);
      }
    }
    await new Promise(r => setTimeout(r, _REPORT_POLL_MS));
    if (isClosed()) break;
    try {
      current = await fetchPreviewRun(run.uuid);
    } catch (err) {
      Logger.error('Failed to re-read the verification run:', err);
      break;
    }
    if (current.stats?.state !== 'running') break;
    if (missingPolls > MAX_MISSING_POLLS) {
      // The task is gone and nothing will ever update the run again.
      current = { ...current, stats: { ...current.stats, state: 'interrupted' } };
      break;
    }
  }
  return current;
}
