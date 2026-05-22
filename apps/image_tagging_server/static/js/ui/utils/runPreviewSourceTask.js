import { TaskStatus, cancelTasks, createTaskConfig, deleteTasks, deleteTaskConfigs, fetchTasksResult, fetchTasksStatus, startTask } from '/app-static/js/shared/api/tasks.js';
import { toast } from '/app-static/js/ui/utils/index.js';
import { ActiveTaskField } from '/app-static/js/ui/utils/fields/ActiveTaskField.js';

// Global guard for only one running preview task
let previewTaskActive = false;

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
  cancelBtn.onclick = async () => {
    cancelled = true;
    cancelBtn.disabled = true;
    try {
      await cancelTasks({ taskUuids: [taskUuid] });
      await deleteTasks({ taskUuids: [taskUuid] });
    } catch (err) {
      toast(`Failed to cancel task: ${err.message}`, 5000, 'error');
    }
  };

  targetDiv.appendChild(cancelBtn);
  targetDiv.appendChild(taskDisplay);

  async function renderTaskStatus(taskInfo) {
    const field = await ActiveTaskField.create({ value: taskInfo });
    const el = await field.renderView();
    taskDisplay.replaceChildren(el);
  }

  try {
    const configUuid = await createTaskConfig({
      typename: 'PreviewSourceTask',
      name: "Source Preview: " + provider,
      params: { source_config: { provider, provider_params: providerParams } },
      persistent: false,
    });

    taskUuid = await startTask({ configUuid });
    await deleteTaskConfigs({ uuids: [configUuid] });

    const startTime = Date.now();
    let taskInfo;
    while (true) {
      if (cancelled) break;

      const res = await fetchTasksStatus({ taskUuids: [taskUuid] });
      taskInfo = res.tasks[taskUuid];
      await renderTaskStatus(taskInfo);

      const taskStatus = taskInfo.instance.status;
      if (taskStatus === TaskStatus.COMPLETED || taskStatus === TaskStatus.ERROR) {
        break;
      }

      if (Date.now() - startTime > 30_000) {
        taskInfo.instance.status = TaskStatus.ERROR;
        taskInfo.instance.error_message = 'Timed out after 30s';
        await renderTaskStatus(taskInfo);
        break;
      }

      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    if (!cancelled && taskInfo.instance.status === TaskStatus.COMPLETED) {
      const res = await fetchTasksResult({ taskUuids: [taskUuid], cacheResults: false });
      const result = res.get(taskUuid);

      if (result?.data?.image) {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${result.data.image}`;
        img.alt = 'Preview result';
        img.style.maxWidth = '100%';
        taskDisplay.insertBefore(img, taskDisplay.firstChild);
      }
    }
  } catch (err) {
    toast(`Preview failed: ${err.message}`, 5000, "error");
  } finally {
    cancelBtn.remove();
    previewTaskActive = false;

    if (taskUuid) {
      await deleteTasks({ taskUuids: [taskUuid] }).catch(e =>
        console.error("Cleanup failed", e)
      );
    }
  }
}
