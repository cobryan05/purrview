import { TaskStatus, cancelTasks, createTaskConfig, deleteTasks, fetchTasksResult, startTask } from '/app-static/js/shared/api/tasks.js';
import { decodeBase64JsonImage, toast } from '/app-static/js/ui/utils/index.js';
import { ActiveTaskField } from '/app-static/js/ui/utils/fields/ActiveTaskField.js';
import { subscribeTaskStream } from '/app-static/js/shared/api/taskStream.js';

// Global guard for only one running preview task
let previewTaskActive = false;

/** Minimal fingerprint so unchanged events skip a pointless re-render. */
function _fp(task) {
  const logs = task.logLines ?? [];
  const progressPart = (task.progress === null || task.progress === undefined) ? 'ind' : Math.round(task.progress);
  return [
    task.instance?.status,
    task.isPausedInPlace ?? false,
    progressPart,
    logs.length,
    logs[logs.length - 1] ?? '',
    task.instance?.startedAt ?? '',
    task.instance?.endedAt ?? '',
  ].join(':');
}

export async function runPreviewSourceTest({ provider, providerParams, target = 'source-preview-box' }) {
  if (previewTaskActive) {
    console.warn("Preview task already running, ignoring request");
    toast("A preview task is already running. Please cancel it first.", 4000, "warning");
    return;
  }
  if (!provider) {
    toast("Please fill in provider before testing", 3000, "warning");
    return;
  }

  const targetDiv = document.getElementById(target);
  targetDiv.innerHTML = '';

  let cancelled = false;
  previewTaskActive = true;

  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';

  const taskDisplay = document.createElement('div');

  let taskUuid;
  let lastFingerprint = null;
  let lastTaskInfo = null;
  // Filled in when the SSE Promise starts; called by the cancel handler so
  // cancellation takes effect immediately rather than waiting for the 30s timeout.
  let _abortSSE = null;

  cancelBtn.onclick = async () => {
    cancelled = true;
    cancelBtn.disabled = true;
    _abortSSE?.();   // immediately break out of the SSE wait
    try {
      await cancelTasks({ taskUuids: [taskUuid] });
      await deleteTasks({ taskUuids: [taskUuid] });
    } catch (err) {
      toast(`Failed to cancel task: ${err.message}`, 5000, 'error');
    }
  };

  targetDiv.appendChild(cancelBtn);
  targetDiv.appendChild(taskDisplay);

  /**
   * Re-render the task card, preserving open/scroll state so a refresh
   * doesn't collapse an expanded <details> element.
   * Returns true if the DOM was actually updated, false if skipped (stale or no change).
   */
  async function renderTaskStatus(taskInfo) {
    const fp = _fp(taskInfo);
    if (fp === lastFingerprint) return false;
    lastFingerprint = fp;
    lastTaskInfo = taskInfo;

    // Snapshot open/scroll state before the async render so we can restore it
    const details = taskDisplay.querySelector('details');
    const pre = details?.querySelector('pre');
    const savedOpen = details?.open ?? true;
    const savedScroll = pre?.scrollTop ?? 0;

    const field = await ActiveTaskField.create({ value: taskInfo });
    const el = await field.renderView();

    // If a concurrent render ran while we were awaiting, discard our stale result
    if (lastFingerprint !== fp) return false;

    // Restore open/scroll on the freshly rendered element
    const newDetails = el.querySelector('details');
    if (newDetails && details !== null) {
      newDetails.open = savedOpen;
      if (savedOpen && savedScroll) {
        requestAnimationFrame(() => {
          const newPre = newDetails.querySelector('pre');
          if (newPre) newPre.scrollTop = savedScroll;
        });
      }
    }

    taskDisplay.replaceChildren(el);
    return true;
  }

  try {
    const configUuid = await createTaskConfig({
      typename: 'PreviewSourceTask',
      name: "Source Preview: " + provider,
      params: { source_config: { provider, provider_params: providerParams } },
      persistent: false,
      isTemporary: true,
    });

    taskUuid = await startTask({ configUuid });
    // No manual config delete needed — is_temporary configs auto-delete when their last instance is removed.

    await new Promise((resolve, reject) => {
      // Guard: if cancel was pressed during the task-setup awaits above, bail out
      // immediately rather than starting the SSE stream and waiting up to 30 s.
      if (cancelled) { resolve(); return; }

      let timeoutId;
      let settled = false;
      let unsubscribe = null;

      const _settle = (fn, ...args) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeoutId);
        unsubscribe?.();
        fn(...args);
      };

      _abortSSE = () => _settle(resolve);

      timeoutId = setTimeout(async () => {
        if (lastTaskInfo) {
          const timedOut = { ...lastTaskInfo, instance: { ...lastTaskInfo.instance, status: TaskStatus.ERROR, errorMessage: 'Timed out after 30s' } };
          lastFingerprint = null;
          await renderTaskStatus(timedOut);
        }
        _settle(resolve);
      }, 30_000);

      unsubscribe = subscribeTaskStream({
        onMessage: async (data) => {
          if (cancelled) { _settle(resolve); return; }

          const taskInfo = data.tasks?.[taskUuid];
          if (!taskInfo) return;

          await renderTaskStatus(taskInfo);

          const status = taskInfo.instance?.status;
          if (status === TaskStatus.COMPLETED || status === TaskStatus.ERROR) {
            _settle(resolve);
          }
        },
        // Transient drops are absorbed by EventSource's own reconnect; this only
        // fires once the shared connection is fully closed.
        onError: (err) => _settle(reject, err),
      });
    });

    if (!cancelled && lastTaskInfo?.instance?.status === TaskStatus.COMPLETED) {
      const res = await fetchTasksResult({ taskUuids: [taskUuid], cacheResults: false });
      const result = res.get(taskUuid);

      if (result?.data?.image) {
        const img_data = await decodeBase64JsonImage(result.data.image);
        if (img_data) {
          img_data.alt = 'Preview result';
          img_data.style.maxWidth = '100%';
          taskDisplay.insertBefore(img_data, taskDisplay.firstChild);
        }
      }
    }
  } catch (err) {
    toast(`Preview failed: ${err.message}`, 5000, "error");
  } finally {
    _abortSSE = null;
    cancelBtn.remove();
    previewTaskActive = false;

    if (taskUuid && !cancelled) {
      await deleteTasks({ taskUuids: [taskUuid] }).catch(e =>
        console.error("Cleanup failed", e)
      );
    }
  }
}
