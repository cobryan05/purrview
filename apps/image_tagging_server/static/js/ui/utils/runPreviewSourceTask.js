import { TaskStatus, cancelTasks, createTaskConfig, deleteTasks, deleteTaskConfigs, fetchTasksResult, fetchTasksStatus, startTask } from '/app-static/js/shared/api/tasks.js';
import { toast } from '/app-static/js/ui/utils/index.js';

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
  let lastMessage = '';

  previewTaskActive = true; // mark task as running

  // Cancel button at top
  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';

  // Message container
  const messagesContainer = document.createElement('div');

  let taskUuid;
  cancelBtn.onclick = async () => {
    cancelled = true;
    cancelBtn.disabled = true;
    addMessage('⏹ Cancelling task...');
    try {
      await cancelTasks({ taskUuids: [taskUuid] });
      await deleteTasks({ taskUuids: [taskUuid] });
      addMessage('❌ Task cancelled by user');
    } catch (err) {
      toast(`Failed to cancel task: ${err.message}`, 5000, 'error');
    }
  };

  targetDiv.appendChild(cancelBtn);
  targetDiv.appendChild(messagesContainer);

  function addMessage(msg) {
    const newestChild = messagesContainer.firstChild;
    if (msg === lastMessage) {
      if (newestChild) newestChild.textContent += '.';
    } else {
      const p = document.createElement('div');
      p.textContent = msg;
      messagesContainer.insertBefore(p, messagesContainer.firstChild);
      lastMessage = msg;
    }
  }

  try {
    const configUuid = await createTaskConfig({
      typename: 'PreviewSourceTask',
      name: "Source Preview: " + provider,
      params: { source_config: { provider, provider_params: providerParams } },
      persistent: false,
    });

    taskUuid = await startTask({ configUuid });
    addMessage(`Task started: ${taskUuid}`);
    addMessage(`Status: pending...`);

    await deleteTaskConfigs({ uuids: [configUuid] });

    const startTime = Date.now();
    let taskInfo;
    while (true) {
      if (cancelled) break; // stop polling if cancelled

      const res = await fetchTasksStatus({ taskUuids: [taskUuid] });
      taskInfo = res.tasks[taskUuid];
      const taskStatus = taskInfo.instance.status;
      addMessage(`Status update: ${taskStatus}`);

      if (taskStatus === TaskStatus.COMPLETED || taskStatus === TaskStatus.ERROR) {
        break;
      }

      if (Date.now() - startTime > 30_000) {
        addMessage('❌ Task timed out');
        break;
      }

      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    // Remove cancel button once task finishes
    if (!cancelled && taskInfo.instance.status === TaskStatus.COMPLETED) {
      const res = await fetchTasksResult({ taskUuids: [taskUuid], cacheResults: false });
      const result = res.get(taskUuid);
      addMessage('✅ Task completed');

      if (result?.data?.image) {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${result.data.image}`;
        img.alt = 'Preview result';
        img.style.maxWidth = '100%';
        messagesContainer.insertBefore(img, messagesContainer.firstChild);
      }
    } else if (!cancelled && taskInfo.instance.status === TaskStatus.ERROR) {
      addMessage('❌ Task failed');
    }
  } catch (err) {
    toast(`Preview failed: ${err.message}`, 5000, "error");
    addMessage(`Error: ${err.message}`);
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
