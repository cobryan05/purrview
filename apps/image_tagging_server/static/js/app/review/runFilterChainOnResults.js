/**
 * "Run Filter Chain on Results" — run a stored FilterChain over the images
 * behind the review browser's current selection/filter.
 *
 * Reuses `resolveCandidateScope` (runModelOnResults.js) to enumerate the scope,
 * but takes only its `imageUuids`: filter chains are *image-scoped* (they run
 * every enabled step over each whole image), so this runs the chain over the
 * images behind the current results, not just the selected bboxes. The chain is
 * launched as a one-shot `ApplyFilterChainTask` (the same task the Tasks tab
 * uses), polled to completion, and its per-action counts are summarized.
 */

import { resolveCandidateScope } from '/app-static/js/app/review/runModelOnResults.js';
import { fetchFilterChains } from '/app-static/js/shared/api/filterChains.js';
import {
  createTaskConfig,
  startTask,
  fetchTasksStatus,
  fetchTasksResult,
  cancelTasks,
  deleteTasks,
  TaskStatus,
} from '/app-static/js/shared/api/tasks.js';
import { createModalShell } from '/app-static/js/ui/utils/modalShell.js';
import { toast, Logger } from '/app-static/js/ui/utils/logging.js';

const POLL_INTERVAL_MS = 700;
// ApplyFilterChainTask enforces its own ceiling (5000); this is the dialog default.
const DEFAULT_MAX_IMAGES = 5000;

/**
 * Builds a human-readable summary line from the task's snake_case result dict.
 * TaskResult.data is a raw passthrough (no camelCase alias conversion).
 */
function _summarize(data) {
  const parts = [];
  if (data.bboxes_added) parts.push(`${data.bboxes_added} added`);
  if (data.bboxes_relabeled) parts.push(`${data.bboxes_relabeled} relabeled`);
  if (data.bboxes_tagged) parts.push(`${data.bboxes_tagged} tagged`);
  if (data.bboxes_stripped) parts.push(`${data.bboxes_stripped} stripped`);
  if (data.images_tagged) parts.push(`${data.images_tagged} image(s) tagged`);
  const changed = data.images_changed ?? 0;
  const processed = data.images_processed ?? 0;
  const head = data.dry_run
    ? `Dry run: ${changed}/${processed} image(s) would change`
    : `${changed}/${processed} image(s) changed`;
  return parts.length > 0 ? `${head} — ${parts.join(', ')}` : head;
}

/**
 * Entry point wired to the review-browser toolbar button.
 * @param {Object} ctx - same shape `runModelOnResultsFlow` receives (see
 *   runModelOnResults.js's `resolveCandidateScope` for field docs).
 */
export async function runFilterChainOnResultsFlow(ctx) {
  let scope;
  try {
    scope = await resolveCandidateScope(ctx);
  } catch (err) {
    Logger.error('Failed to resolve image scope:', err);
    toast(`Failed to resolve images: ${err.message}`, 5000, 'error');
    return;
  }
  const imageUuids = scope.imageUuids;
  if (imageUuids.length === 0) {
    toast('No images in the current selection/filter.', 4000, 'warning');
    return;
  }

  let chains;
  try {
    chains = await fetchFilterChains();
  } catch (err) {
    Logger.error('Failed to load filter chains:', err);
    toast(`Failed to load filter chains: ${err.message}`, 5000, 'error');
    return;
  }
  const chainList = [...chains.values()];

  await _showRunDialog(imageUuids, chainList, ctx);
}

async function _showRunDialog(imageUuids, chainList, ctx) {
  const { dialog, open, close } = createModalShell();
  dialog.style.cssText = 'max-width:520px;max-height:85vh;display:flex;flex-direction:column;gap:0.8em;overflow-y:auto;';
  dialog.classList.add('run-filter-chain-on-results-dialog');

  const title = document.createElement('h3');
  title.className = 'app-modal-title';
  title.textContent = 'Run Filter Chain on Results';
  dialog.appendChild(title);

  const countLine = document.createElement('div');
  countLine.textContent = `Runs the whole chain over ${imageUuids.length.toLocaleString()} image(s) behind the current results (all their bboxes, not just selected ones).`;
  countLine.style.cssText = 'font-size:0.85em;color:#aaa;';
  dialog.appendChild(countLine);

  // ── Chain picker ──────────────────────────────────────────────────────
  const chainRow = document.createElement('div');
  chainRow.style.cssText = 'display:flex;flex-direction:column;gap:0.3em;';
  const chainLabel = document.createElement('div');
  chainLabel.textContent = 'Filter chain:';
  chainLabel.style.cssText = 'font-size:0.85em;color:#aaa;';
  const chainSelect = document.createElement('select');
  chainSelect.className = 'run-filter-chain-on-results-select';
  if (chainList.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '(no filter chains defined)';
    chainSelect.appendChild(opt);
  } else {
    for (const c of chainList) {
      const opt = document.createElement('option');
      opt.value = c.uuid;
      const stepCount = (c.steps || []).length;
      opt.textContent = `${c.name} (${stepCount} step${stepCount === 1 ? '' : 's'})`;
      chainSelect.appendChild(opt);
    }
  }
  chainRow.append(chainLabel, chainSelect);
  dialog.appendChild(chainRow);

  // ── Max images + dry-run ──────────────────────────────────────────────
  const maxRow = document.createElement('div');
  maxRow.style.cssText = 'display:flex;flex-direction:column;gap:0.3em;';
  const maxLabel = document.createElement('label');
  maxLabel.style.cssText = 'font-size:0.85em;color:#aaa;';
  maxLabel.textContent = 'Max images';
  const maxInput = document.createElement('input');
  maxInput.type = 'number';
  maxInput.min = '1';
  maxInput.value = String(Math.min(imageUuids.length, DEFAULT_MAX_IMAGES));
  maxRow.append(maxLabel, maxInput);
  dialog.appendChild(maxRow);

  const dryLabel = document.createElement('label');
  dryLabel.style.cssText = 'display:flex;align-items:center;gap:0.4em;cursor:pointer;';
  const dryCheckbox = document.createElement('input');
  dryCheckbox.type = 'checkbox';
  dryLabel.append(dryCheckbox, document.createTextNode('Dry run (count matches, change nothing)'));
  dialog.appendChild(dryLabel);

  // ── Progress (hidden until Run) ───────────────────────────────────────
  const progressWrap = document.createElement('div');
  progressWrap.style.cssText = 'display:none;flex-direction:column;gap:0.4em;';
  const progressLabel = document.createElement('div');
  progressLabel.style.cssText = 'font-size:0.85em;color:#aaa;';
  const progressBar = document.createElement('progress');
  progressBar.max = 100;
  progressBar.value = 0;
  progressBar.style.width = '100%';
  const progressCancelBtn = document.createElement('button');
  progressCancelBtn.className = 'secondary';
  progressCancelBtn.textContent = 'Cancel';
  progressWrap.append(progressLabel, progressBar, progressCancelBtn);
  dialog.appendChild(progressWrap);

  // ── Buttons ───────────────────────────────────────────────────────────
  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'display:flex;gap:0.5em;justify-content:flex-end;';
  const runBtn = document.createElement('button');
  runBtn.textContent = 'Run';
  runBtn.disabled = chainList.length === 0;
  const cancelBtn = document.createElement('button');
  cancelBtn.className = 'secondary';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', close);
  btnRow.append(cancelBtn, runBtn);
  dialog.appendChild(btnRow);

  let taskUuid = null;
  let pollTimer = null;
  let cancelled = false;

  function _stopPolling() {
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  }

  progressCancelBtn.addEventListener('click', async () => {
    cancelled = true;
    progressCancelBtn.disabled = true;
    _stopPolling();
    if (taskUuid) {
      try {
        await cancelTasks({ taskUuids: [taskUuid] });
        await deleteTasks({ taskUuids: [taskUuid] });
      } catch (err) {
        toast(`Failed to cancel task: ${err.message}`, 5000, 'error');
      }
    }
    close();
  });

  runBtn.addEventListener('click', async () => {
    if (!chainSelect.value) { toast('Please select a filter chain', 3000, 'warning'); return; }
    const maxImages = Math.max(1, Math.floor(Number(maxInput.value)) || imageUuids.length);
    const cappedImageUuids = imageUuids.slice(0, maxImages);

    runBtn.disabled = true;
    cancelBtn.disabled = true;
    chainSelect.disabled = true;
    maxInput.disabled = true;
    dryCheckbox.disabled = true;
    progressWrap.style.display = 'flex';
    progressLabel.textContent = `Starting… (0/${cappedImageUuids.length} images)`;

    const params = {
      filter_chain_uuid: chainSelect.value,
      image_uuids: cappedImageUuids,
      max_images: maxImages,
      dry_run: dryCheckbox.checked,
    };

    try {
      const configUuid = await createTaskConfig({
        typename: 'ApplyFilterChainTask',
        name: 'Run Filter Chain on Results',
        params,
        persistent: false,
        isTemporary: true,
      });
      taskUuid = await startTask({ configUuid });

      const finalInfo = await _pollUntilDone(taskUuid, cappedImageUuids.length, (info) => {
        if (cancelled) return;
        const progress = info?.progress ?? 0;
        const processed = Math.round((progress / 100) * cappedImageUuids.length);
        progressBar.value = progress;
        progressLabel.textContent = `Processing… (${processed}/${cappedImageUuids.length} images)`;
      });

      if (cancelled) return;

      const status = finalInfo?.instance?.status;
      if (status !== TaskStatus.COMPLETED) {
        const msg = finalInfo?.instance?.errorMessage || `Task ended with status: ${status}`;
        toast(`Run failed: ${msg}`, 5000, 'error');
        await deleteTasks({ taskUuids: [taskUuid] }).catch(() => {});
        return;
      }

      const resultsMap = await fetchTasksResult({ taskUuids: [taskUuid], cacheResults: false });
      const data = resultsMap.get(taskUuid)?.data ?? {};
      await deleteTasks({ taskUuids: [taskUuid] }).catch((e) => Logger.error('Cleanup failed', e));

      close();
      toast(_summarize(data), 6000, 'success');
      if (!data.dry_run) ctx.refresh?.();
    } catch (err) {
      Logger.error('Run Filter Chain on Results failed:', err);
      toast(`Run failed: ${err.message}`, 5000, 'error');
      runBtn.disabled = false;
      cancelBtn.disabled = false;
      chainSelect.disabled = false;
      maxInput.disabled = false;
      dryCheckbox.disabled = false;
      progressWrap.style.display = 'none';
    }
  });

  async function _pollUntilDone(uuid, totalImages, onUpdate) {
    return new Promise((resolve, reject) => {
      const _tick = async () => {
        if (cancelled) { resolve(null); return; }
        try {
          const { tasks } = await fetchTasksStatus({ taskUuids: [uuid] });
          const info = tasks?.[uuid];
          if (info) onUpdate(info);
          const status = info?.instance?.status;
          if (status === TaskStatus.COMPLETED || status === TaskStatus.ERROR || status === TaskStatus.CANCELLED) {
            resolve(info);
            return;
          }
        } catch (err) {
          reject(err);
          return;
        }
        if (!cancelled) pollTimer = setTimeout(_tick, POLL_INTERVAL_MS);
      };
      _tick();
    });
  }

  open();
}
