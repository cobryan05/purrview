import { TaskStatus, cancelTask, createTaskConfig, deleteTask, deleteTaskConfigs, fetchTaskResult, fetchTaskStatus, startTask, toast } from '/app-static/js/ui/utils/index.js';

// Global guard for only one running preview task
let previewTaskActive = false;

export async function runPreviewSourceTest({ providerName, params, target = 'source-preview-box' }) {
  if (previewTaskActive) {
    console.warn("Preview task already running, ignoring request");
    toast("A preview task is already running. Please cancel it first.", 4000, "warning");
    return;
  }
  if (!providerName) {
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

  let taskId;
  cancelBtn.onclick = async () => {
    cancelled = true;
    cancelBtn.disabled = true;
    addMessage('⏹ Cancelling task...');
    try {
      await cancelTask(taskId);
      await deleteTask(taskId);
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
      params: { provider: providerName, provider_params: params },
      persistent: false,
    });

    taskId = await startTask({ uuid: configUuid });
    addMessage(`Task started: ${taskId}`);
    addMessage(`Status: pending...`);

    await deleteTaskConfigs({ uuids: [configUuid] });

    const startTime = Date.now();
    let taskInfo;

    while (true) {
      if (cancelled) break; // stop polling if cancelled
      taskInfo = await fetchTaskStatus(taskId);
      addMessage(`Status update: ${taskInfo.message || taskInfo.status}`);

      if (taskInfo.status === TaskStatus.COMPLETED || taskInfo.status === TaskStatus.ERROR) break;

      if (Date.now() - startTime > 30_000) {
        addMessage('❌ Task timed out');
        break;
      }

      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    // Remove cancel button once task finishes
    cancelBtn.remove();
    previewTaskActive = false;

    if (!cancelled && taskInfo.status === TaskStatus.COMPLETED) {
      const results = await fetchTaskResult({ taskIds: [taskId], cacheResults: false });
      const result = results.get(taskId);
      addMessage('✅ Task completed');

      if (result?.data?.image) {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${result.data.image}`;
        img.alt = 'Preview result';
        img.style.maxWidth = '100%';
        messagesContainer.insertBefore(img, messagesContainer.firstChild);
      }
    } else if (!cancelled && taskInfo.status === TaskStatus.ERROR) {
      addMessage('❌ Task failed');
    }
  } catch (err) {
    toast(`Preview failed: ${err.message}`, 5000, "error");
    addMessage(`Error: ${err.message}`);
    cancelBtn.remove();
    previewTaskActive = false;
  }
}
