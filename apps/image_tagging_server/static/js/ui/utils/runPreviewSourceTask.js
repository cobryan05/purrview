import { TaskStatus, createTaskConfig, fetchTaskResult, fetchTaskStatus, startTask, toast } from '/app-static/js/ui/utils/index.js';


export async function runPreviewSourceTest({ providerName, params, target = 'source-preview-box' }) {
  if (!providerName) {
    toast("Please fill in provider before testing", 3000, "warning");
    return;
  }
  const targetDiv = document.getElementById(target);
  targetDiv.innerHTML = '';

  let cancelled = false;
  let lastMessage = '';

  // Cancel button at top
  const cancelBtn = document.createElement('button');
  cancelBtn.textContent = 'Cancel';

  // Message container
  const messagesContainer = document.createElement('div');

  cancelBtn.onclick = async () => {
    cancelled = true;
    cancelBtn.disabled = true;
    try {
      await cancelTask(taskId);
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

    const taskId = await startTask({ uuid: configUuid });
    addMessage(`Task started: ${taskId}`);
    addMessage(`Status: pending...`);

    const startTime = Date.now();
    let taskInfo;

    while (true) {
      taskInfo = await fetchTaskStatus(taskId);
      addMessage(`Status update: ${taskInfo.message || taskInfo.status}`);

      if (taskInfo.status === TaskStatus.COMPLETED || taskInfo.status === TaskStatus.ERROR || cancelled) break;

      if (Date.now() - startTime > 30_000) {
        addMessage('❌ Task timed out');
        break;
      }

      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    // Remove cancel button once task finishes
    cancelBtn.remove();

    if (!cancelled && taskInfo.status === TaskStatus.COMPLETED) {
      const result = await fetchTaskResult(taskId);
      addMessage('✅ Task completed');

      if (result?.data?.image) {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${result.data.image}`;
        img.alt = 'Preview result';
        img.style.maxWidth = '100%';
        // Insert image at top of messages container
        messagesContainer.insertBefore(img, messagesContainer.firstChild);
      }
    } else if (!cancelled && taskInfo.status === TaskStatus.ERROR) {
      addMessage('❌ Task failed');
    }
  } catch (err) {
    toast(`Preview failed: ${err.message}`, 5000, "error");
    addMessage(`Error: ${err.message}`);
    cancelBtn.remove();
  }
}
