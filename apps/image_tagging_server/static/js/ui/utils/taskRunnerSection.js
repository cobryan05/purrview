/**
 * Reusable "start a task, stream its progress via SSE, allow cancel" section.
 *
 * Extracted from the Training Config "Start Training" button (originally
 * inline in ui/trainingConfigs.js) so a second caller could drive the same
 * start/stream/cancel/render sequence against a different task typename
 * without duplicating the SSE plumbing.
 *
 * Training no longer uses it: a run started from a config editor put progress,
 * logs and cancel inside a *configuration* screen, so training now starts via
 * launchRun and is watched in Runs.
 *
 * Its remaining callers are ui/models.js — model import and conversion. Those
 * are infrastructure actions on the model itself rather than jobs you would go
 * to Runs to supervise, and they stay inline deliberately.
 */

import { TaskStatus, cancelTasks, createTaskConfig, startTask } from '/app-static/js/shared/api/tasks.js';
import { ActiveTaskField } from '/app-static/js/ui/utils/fields/ActiveTaskField.js';
import { toast } from '/app-static/js/ui/utils/logging.js';
import { subscribeTaskStream } from '/app-static/js/shared/api/taskStream.js';

/**
 * @param {Object} opts
 * @param {string} opts.buttonLabel - Idle-state button text (e.g. "▶ Start Training").
 * @param {string} opts.runningLabel - Button text while a task is running (e.g. "Training…").
 * @param {() => (Promise<?{typename:string, name:string, params:object}> | ?{typename:string, name:string, params:object})} opts.buildStartArgs
 *   Called on click. Return null to abort the start (caller must have already
 *   surfaced its own validation toast). Returning a value starts the task.
 * @param {(finalTaskInfo: ?object) => (Promise<void> | void)} [opts.onSettled]
 *   Called after the task reaches a terminal state (or errors/is cancelled),
 *   with the last task info seen (null if the task never got far enough to report one).
 * @returns {{ el: HTMLElement }}
 */
export function createTaskRunnerSection({ buttonLabel, runningLabel, buildStartArgs, onSettled }) {
  const section = document.createElement('div');
  section.style.cssText = 'margin-top: 1em; padding-top: 0.75em; border-top: 1px solid var(--pico-muted-border-color, #ccc);';

  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'display:flex; gap:0.5em; align-items:center; margin-bottom:0.5em;';

  const startBtn = document.createElement('button');
  startBtn.type = 'button';
  startBtn.textContent = buttonLabel;
  startBtn.className = 'panel-btn';
  startBtn.style.cssText = 'margin-bottom: 0;';

  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.className = 'secondary panel-btn';
  cancelBtn.style.cssText = 'margin-bottom:0; display:none;';

  btnRow.append(startBtn, cancelBtn);
  section.appendChild(btnRow);

  const taskDisplay = document.createElement('div');
  section.appendChild(taskDisplay);

  let _activeTaskUuid = null;
  let _abortSSE = null;
  let _lastFingerprint = null;

  function _fp(task) {
    const logs = task.logLines ?? [];
    const progressPart = (task.progress === null || task.progress === undefined) ? 'ind' : Math.round(task.progress);
    return [
      task.instance?.status,
      task.isPausedInPlace ?? false,
      progressPart,
      logs.length,
      logs[logs.length - 1] ?? '',
    ].join(':');
  }

  async function _renderTaskStatus(taskInfo) {
    const fp = _fp(taskInfo);
    if (fp === _lastFingerprint) return;
    _lastFingerprint = fp;

    const details = taskDisplay.querySelector('details');
    const pre = details?.querySelector('pre');
    const savedOpen = details?.open ?? true;
    const savedScroll = pre?.scrollTop ?? 0;

    const field = await ActiveTaskField.create({ value: taskInfo });
    const el = await field.renderView();

    if (_lastFingerprint !== fp) return;

    const newDetails = el.querySelector('details');
    if (newDetails) {
      newDetails.open = savedOpen;
      if (savedOpen && savedScroll) {
        requestAnimationFrame(() => {
          const newPre = newDetails.querySelector('pre');
          if (newPre) newPre.scrollTop = savedScroll;
        });
      }
    }
    taskDisplay.replaceChildren(el);
  }

  cancelBtn.onclick = async () => {
    cancelBtn.disabled = true;
    _abortSSE?.();
    if (_activeTaskUuid) {
      try { await cancelTasks({ taskUuids: [_activeTaskUuid] }); } catch { /* best-effort */ }
    }
  };

  startBtn.addEventListener('click', async () => {
    // buildStartArgs may await a user prompt (e.g. the unsaved-changes modal),
    // so the button has to be held down for that window too — otherwise a second
    // click stacks a second modal on the first.
    startBtn.disabled = true;
    let startArgs;
    try {
      startArgs = await buildStartArgs();
    } finally {
      if (!startArgs) startBtn.disabled = false;
    }
    if (!startArgs) return;

    startBtn.disabled = true;
    startBtn.textContent = 'Starting…';
    taskDisplay.innerHTML = '';
    _lastFingerprint = null;
    _activeTaskUuid = null;
    let finalTaskInfo = null;

    try {
      const configUuid = await createTaskConfig({
        typename: startArgs.typename,
        name: startArgs.name,
        params: startArgs.params,
        // Transient runner (e.g. "Start Training") — must never silently
        // auto-resume across a server restart.
        persistent: false,
        isTemporary: true,
      });
      const taskUuid = await startTask({ configUuid });
      _activeTaskUuid = taskUuid;

      startBtn.textContent = runningLabel;
      cancelBtn.style.display = '';

      await new Promise((resolve, reject) => {
        let settled = false;
        let unsubscribe = null;

        const _settle = (fn, ...args) => {
          if (settled) return;
          settled = true;
          unsubscribe?.();
          fn(...args);
        };

        _abortSSE = () => _settle(resolve);

        unsubscribe = subscribeTaskStream({
          onMessage: async (data) => {
            const taskInfo = data.tasks?.[taskUuid];
            if (!taskInfo) return;

            await _renderTaskStatus(taskInfo);

            const status = taskInfo.instance?.status;
            if (status === TaskStatus.COMPLETED || status === TaskStatus.ERROR || status === TaskStatus.CANCELLED) {
              finalTaskInfo = taskInfo;
              _settle(resolve);
            }
          },
          onError: (err) => _settle(reject, err),
        });
      });
    } catch (err) {
      toast(err.message || 'Failed to start task', 5000, 'error');
    } finally {
      _abortSSE = null;
      startBtn.disabled = false;
      startBtn.textContent = buttonLabel;
      cancelBtn.style.display = 'none';
      cancelBtn.disabled = false;
      await onSettled?.(finalTaskInfo);
    }
  });

  return { el: section };
}
