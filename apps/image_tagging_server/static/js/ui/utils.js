// ================= Utility Functions =================


/**
 * Extracts form parameters from a given container element.
 *
 * This function iterates over all input, select, and textarea elements
 * within the specified container and collects their values into an object.
 * Checkbox values are stored as booleans, number inputs are parsed as floats,
 * and other input types are stored as strings.
 *
 * @param {HTMLElement} container - The container element containing form inputs.
 * @returns {Object} An object where keys are input names and values are their corresponding values.
 */
export function getParamsFromForm(container) {
  const params = {};
  container.querySelectorAll('input, select, textarea').forEach(el => {
    if (!el.name) return;
    if (el.type === 'checkbox') params[el.name] = el.checked;
    else if (el.type === 'number') params[el.name] = parseFloat(el.value);
    else params[el.name] = el.value;
  });
  return params;
}


/**
 * Utility: Fetch list of image providers
 */
export async function fetchImageProviderList() {
  const res = await fetch('/api/sources/image-providers/list');
  const data = await res.json();
  return data.providers;
}

/**
 * Renders form inputs from JSON schema
 */
export function renderSchemaForm(schema, container) {
  container.innerHTML = '';
  Object.entries(schema).forEach(([name, field]) => {
    const wrapper = document.createElement('div');
    wrapper.style.display = 'flex';
    wrapper.style.flexDirection = 'column';
    wrapper.style.marginBottom = '1em';

    const label = document.createElement('label');
    label.textContent = field.title || name;
    label.htmlFor = name;
    if (field.required) label.innerHTML += ' <span style="color: red">*</span>';
    wrapper.appendChild(label);

    let input;
    if (field.type === 'boolean') {
      const checkboxWrapper = document.createElement('label');
      checkboxWrapper.style.display = 'flex';
      checkboxWrapper.style.alignItems = 'center';
      checkboxWrapper.style.gap = '0.5em';

      input = document.createElement('input');
      input.type = 'checkbox';
      input.name = name;
      input.checked = !!field.default;
      checkboxWrapper.appendChild(input);
      checkboxWrapper.appendChild(document.createTextNode(field.description || field.title || name));
      wrapper.appendChild(checkboxWrapper);
    } else if (field.type === 'array' && field.items?.type === 'string') {
      input = document.createElement('textarea');
      input.name = name;
      input.placeholder = (field.items.description || field.description || '') + ' (one per line)';
      input.rows = 3;
      wrapper.appendChild(input);
    } else {
      input = document.createElement('input');
      input.type = 'text';
      input.name = name;
      input.placeholder = field.description || '';
      if (field.default != null) input.value = field.default;
      wrapper.appendChild(input);
    }

    if (field.required) input.required = true;
    container.appendChild(wrapper);
  });
}

export function pollTaskStatus(taskUuid, resultBox, onComplete, timeoutMs = 30000) {
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/api/tasks/status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_ids: [taskUuid] }),
      });
      const result = await res.json();

      if (result.status === "success") {
        const taskStatus = result.tasks[taskUuid];
        resultBox.textContent = taskStatus.message;

        if (taskStatus.status === 'completed' || taskStatus.status === 'error') {
          clearInterval(interval);
          clearTimeout(timeout);
          const taskRes = await fetch(`/api/tasks/result`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: taskUuid }),
          });
          const taskResult = await taskRes.json();
          onComplete(taskResult.result);
        }
      }
    } catch (err) {
      console.error("Polling error:", err);
    }
  }, 1000);

  const timeout = setTimeout(() => {
    clearInterval(interval);
    resultBox.textContent = "Task timed out.";
    onComplete({ status: 'timeout', message: 'The task did not complete in time.' });
  }, timeoutMs);
}
